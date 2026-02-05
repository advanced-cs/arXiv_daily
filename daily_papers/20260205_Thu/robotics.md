# 机器人 cs.RO

- **最新发布 52 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] Control and State Estimation of Vehicle-Mounted Aerial Systems in GPS-Denied, Non-Inertial Environments
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于无人机控制任务，解决GPS拒止环境下姿态估计与控制问题。通过融合外部位置信息和EKF-UI方法，提升跟踪稳定性，无需惯性传感器。**

- **链接: [https://arxiv.org/pdf/2602.04057v1](https://arxiv.org/pdf/2602.04057v1)**

> **作者:** Riming Xu; Obadah Wali; Yasmine Marani; Eric Feron
>
> **备注:** 10 pages 8 figures
>
> **摘要:** We present a robust control and estimation framework for quadrotors operating in Global Navigation Satellite System(GNSS)-denied, non-inertial environments where inertial sensors such as Inertial Measurement Units (IMUs) become unreliable due to platform-induced accelerations. In such settings, conventional estimators fail to distinguish whether the measured accelerations arise from the quadrotor itself or from the non-inertial platform, leading to drift and control degradation. Unlike conventional approaches that depend heavily on IMU and GNSS, our method relies exclusively on external position measurements combined with a Extended Kalman Filter with Unknown Inputs (EKF-UI) to account for platform motion. The estimator is paired with a cascaded PID controller for full 3D tracking. To isolate estimator performance from localization errors, all tests are conducted using high-precision motion capture systems. Experimental results in a moving-cart testbed validate our approach under both translational in X-axis and Y-axis dissonance. Compared to standard EKF, the proposed method significantly improves stability and trajectory tracking without requiring inertial feedback, enabling practical deployment on moving platforms such as trucks or elevators.
>
---
#### [new 002] GeoLanG: Geometry-Aware Language-Guided Grasping with Unified RGB-D Multimodal Learning
- **分类: cs.RO**

- **简介: 该论文属于语言引导抓取任务，旨在解决复杂场景下抓取精度不足的问题。提出GeoLanG框架，融合视觉与语言信息，提升抓取鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.04231v1](https://arxiv.org/pdf/2602.04231v1)**

> **作者:** Rui Tang; Guankun Wang; Long Bai; Huxin Gao; Jiewen Lai; Chi Kit Ng; Jiazheng Wang; Fan Zhang; Hongliang Ren
>
> **备注:** IEEE ICRA 2025
>
> **摘要:** Language-guided grasping has emerged as a promising paradigm for enabling robots to identify and manipulate target objects through natural language instructions, yet it remains highly challenging in cluttered or occluded scenes. Existing methods often rely on multi-stage pipelines that separate object perception and grasping, which leads to limited cross-modal fusion, redundant computation, and poor generalization in cluttered, occluded, or low-texture scenes. To address these limitations, we propose GeoLanG, an end-to-end multi-task framework built upon the CLIP architecture that unifies visual and linguistic inputs into a shared representation space for robust semantic alignment and improved generalization. To enhance target discrimination under occlusion and low-texture conditions, we explore a more effective use of depth information through the Depth-guided Geometric Module (DGGM), which converts depth into explicit geometric priors and injects them into the attention mechanism without additional computational overhead. In addition, we propose Adaptive Dense Channel Integration, which adaptively balances the contributions of multi-layer features to produce more discriminative and generalizable visual representations. Extensive experiments on the OCID-VLG dataset, as well as in both simulation and real-world hardware, demonstrate that GeoLanG enables precise and robust language-guided grasping in complex, cluttered environments, paving the way toward more reliable multimodal robotic manipulation in real-world human-centric settings.
>
---
#### [new 003] TACO: Temporal Consensus Optimization for Continual Neural Mapping
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决动态环境中持续学习的神经映射问题。提出TACO框架，通过时间共识优化实现无需回放的历史知识利用，平衡记忆效率与适应性。**

- **链接: [https://arxiv.org/pdf/2602.04516v1](https://arxiv.org/pdf/2602.04516v1)**

> **作者:** Xunlan Zhou; Hongrui Zhao; Negar Mehr
>
> **摘要:** Neural implicit mapping has emerged as a powerful paradigm for robotic navigation and scene understanding. However, real-world robotic deployment requires continual adaptation to changing environments under strict memory and computation constraints, which existing mapping systems fail to support. Most prior methods rely on replaying historical observations to preserve consistency and assume static scenes. As a result, they cannot adapt to continual learning in dynamic robotic settings. To address these challenges, we propose TACO (TemporAl Consensus Optimization), a replay-free framework for continual neural mapping. We reformulate mapping as a temporal consensus optimization problem, where we treat past model snapshots as temporal neighbors. Intuitively, our approach resembles a model consulting its own past knowledge. We update the current map by enforcing weighted consensus with historical representations. Our method allows reliable past geometry to constrain optimization while permitting unreliable or outdated regions to be revised in response to new observations. TACO achieves a balance between memory efficiency and adaptability without storing or replaying previous data. Through extensive simulated and real-world experiments, we show that TACO robustly adapts to scene changes, and consistently outperforms other continual learning baselines.
>
---
#### [new 004] GeneralVLA: Generalizable Vision-Language-Action Models with Knowledge-Guided Trajectory Planning
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出GeneralVLA，解决机器人零样本操作问题。通过视觉-语言-动作模型和知识引导轨迹规划，实现无需真实数据的通用任务执行与数据生成。**

- **链接: [https://arxiv.org/pdf/2602.04315v1](https://arxiv.org/pdf/2602.04315v1)**

> **作者:** Guoqing Ma; Siheng Wang; Zeyu Zhang; Shan Yu; Hao Tang
>
> **摘要:** Large foundation models have shown strong open-world generalization to complex problems in vision and language, but similar levels of generalization have yet to be achieved in robotics. One fundamental challenge is that the models exhibit limited zero-shot capability, which hampers their ability to generalize effectively to unseen scenarios. In this work, we propose GeneralVLA (Generalizable Vision-Language-Action Models with Knowledge-Guided Trajectory Planning), a hierarchical vision-language-action (VLA) model that can be more effective in utilizing the generalization of foundation models, enabling zero-shot manipulation and automatically generating data for robotics. In particular, we study a class of hierarchical VLA model where the high-level ASM (Affordance Segmentation Module) is finetuned to perceive image keypoint affordances of the scene; the mid-level 3DAgent carries out task understanding, skill knowledge, and trajectory planning to produce a 3D path indicating the desired robot end-effector trajectory. The intermediate 3D path prediction is then served as guidance to the low-level, 3D-aware control policy capable of precise manipulation. Compared to alternative approaches, our method requires no real-world robotic data collection or human demonstration, making it much more scalable to diverse tasks and viewpoints. Empirically, GeneralVLA successfully generates trajectories for 14 tasks, significantly outperforming state-of-the-art methods such as VoxPoser. The generated demonstrations can train more robust behavior cloning policies than training with human demonstrations or from data generated by VoxPoser, Scaling-up, and Code-As-Policies. We believe GeneralVLA can be the scalable method for both generating data for robotics and solving novel tasks in a zero-shot setting. Code: https://github.com/AIGeeksGroup/GeneralVLA. Website: https://aigeeksgroup.github.io/GeneralVLA.
>
---
#### [new 005] MA3DSG: Multi-Agent 3D Scene Graph Generation for Large-Scale Indoor Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于3D场景图生成任务，解决单智能体方法在大规模环境中的可扩展性问题。提出MA3DSG模型，通过多智能体协作生成统一场景图。**

- **链接: [https://arxiv.org/pdf/2602.04152v1](https://arxiv.org/pdf/2602.04152v1)**

> **作者:** Yirum Kim; Jaewoo Kim; Ue-Hwan Kim
>
> **摘要:** Current 3D scene graph generation (3DSGG) approaches heavily rely on a single-agent assumption and small-scale environments, exhibiting limited scalability to real-world scenarios. In this work, we introduce Multi-Agent 3D Scene Graph Generation (MA3DSG) model, the first framework designed to tackle this scalability challenge using multiple agents. We develop a training-free graph alignment algorithm that efficiently merges partial query graphs from individual agents into a unified global scene graph. Leveraging extensive analysis and empirical insights, our approach enables conventional single-agent systems to operate collaboratively without requiring any learnable parameters. To rigorously evaluate 3DSGG performance, we propose MA3DSG-Bench-a benchmark that supports diverse agent configurations, domain sizes, and environmental conditions-providing a more general and extensible evaluation framework. This work lays a solid foundation for scalable, multi-agent 3DSGG research.
>
---
#### [new 006] SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出SCALE方法，用于视觉-语言-动作模型的推理阶段，解决测试时扩展效率低和感知不确定性问题，通过自适应调整感知与动作提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.04208v1](https://arxiv.org/pdf/2602.04208v1)**

> **作者:** Hyeonbeom Choi; Daechul Ahn; Youhan Lee; Taewook Kang; Seongwon Cho; Jonghyun Choi
>
> **备注:** 20 pages, 8 figures
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising paradigm for general-purpose robotic control, with test-time scaling (TTS) gaining attention to enhance robustness beyond training. However, existing TTS methods for VLAs require additional training, verifiers, and multiple forward passes, making them impractical for deployment. Moreover, they intervene only at action decoding while keeping visual representations fixed-insufficient under perceptual ambiguity, where reconsidering how to perceive is as important as deciding what to do. To address these limitations, we propose SCALE, a simple inference strategy that jointly modulates visual perception and action based on 'self-uncertainty', inspired by uncertainty-driven exploration in Active Inference theory-requiring no additional training, no verifier, and only a single forward pass. SCALE broadens exploration in both perception and action under high uncertainty, while focusing on exploitation when confident-enabling adaptive execution across varying conditions. Experiments on simulated and real-world benchmarks demonstrate that SCALE improves state-of-the-art VLAs and outperforms existing TTS methods while maintaining single-pass efficiency.
>
---
#### [new 007] FDA Flocking: Future Direction-Aware Flocking via Velocity Prediction
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于群体智能任务，旨在解决传统 flocking 模型反应性不足的问题。通过引入基于速度预测的前瞻性机制，提升群体协调与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.04012v1](https://arxiv.org/pdf/2602.04012v1)**

> **作者:** Hossein B. Jond; Martin Saska
>
> **摘要:** Understanding self-organization in natural collectives such as bird flocks inspires swarm robotics, yet most flocking models remain reactive, overlooking anticipatory cues that enhance coordination. Motivated by avian postural and wingbeat signals, as well as multirotor attitude tilts that precede directional changes, this work introduces a principled, bio-inspired anticipatory augmentation of reactive flocking termed Future Direction-Aware (FDA) flocking. In the proposed framework, agents blend reactive alignment with a predictive term based on short-term estimates of neighbors' future velocities, regulated by a tunable blending parameter that interpolates between reactive and anticipatory behaviors. This predictive structure enhances velocity consensus and cohesion-separation balance while mitigating the adverse effects of sensing and communication delays and measurement noise that destabilize reactive baselines. Simulation results demonstrate that FDA achieves faster and higher alignment, enhanced translational displacement of the flock, and improved robustness to delays and noise compared to a purely reactive model. Future work will investigate adaptive blending strategies, weighted prediction schemes, and experimental validation on multirotor drone swarms.
>
---
#### [new 008] Act, Sense, Act: Learning Non-Markovian Active Perception Strategies from Large-Scale Egocentric Human Data
- **分类: cs.RO**

- **简介: 该论文属于机器人主动感知任务，解决复杂环境中信息不确定性问题。通过构建CoMe-VLA框架，融合人类数据学习多样化操作策略，提升机器人感知与决策能力。**

- **链接: [https://arxiv.org/pdf/2602.04600v1](https://arxiv.org/pdf/2602.04600v1)**

> **作者:** Jialiang Li; Yi Qiao; Yunhan Guo; Changwen Chen; Wenzhao Lian
>
> **摘要:** Achieving generalizable manipulation in unconstrained environments requires the robot to proactively resolve information uncertainty, i.e., the capability of active perception. However, existing methods are often confined in limited types of sensing behaviors, restricting their applicability to complex environments. In this work, we formalize active perception as a non-Markovian process driven by information gain and decision branching, providing a structured categorization of visual active perception paradigms. Building on this perspective, we introduce CoMe-VLA, a cognitive and memory-aware vision-language-action (VLA) framework that leverages large-scale human egocentric data to learn versatile exploration and manipulation priors. Our framework integrates a cognitive auxiliary head for autonomous sub-task transitions and a dual-track memory system to maintain consistent self and environmental awareness by fusing proprioceptive and visual temporal contexts. By aligning human and robot hand-eye coordination behaviors in a unified egocentric action space, we train the model progressively in three stages. Extensive experiments on a wheel-based humanoid have demonstrated strong robustness and adaptability of our proposed method across diverse long-horizon tasks spanning multiple active perception scenarios.
>
---
#### [new 009] VLS: Steering Pretrained Robot Policies via Vision-Language Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人控制任务，解决预训练策略在测试时因环境变化失效的问题。提出VLS框架，在不修改策略参数的情况下，通过视觉-语言模型引导生成策略，提升适应性。**

- **链接: [https://arxiv.org/pdf/2602.03973v1](https://arxiv.org/pdf/2602.03973v1)**

> **作者:** Shuo Liu; Ishneet Sukhvinder Singh; Yiqing Xu; Jiafei Duan; Ranjay Krishna
>
> **备注:** 11 Pages, Project page: https://vision-language-steering.github.io/webpage/
>
> **摘要:** Why do pretrained diffusion or flow-matching policies fail when the same task is performed near an obstacle, on a shifted support surface, or amid mild clutter? Such failures rarely reflect missing motor skills; instead, they expose a limitation of imitation learning under train-test shifts, where action generation is tightly coupled to training-specific spatial configurations and task specifications. Retraining or fine-tuning to address these failures is costly and conceptually misaligned, as the required behaviors already exist but cannot be selectively adapted at test time. We propose Vision-Language Steering (VLS), a training-free framework for inference-time adaptation of frozen generative robot policies. VLS treats adaptation as an inference-time control problem, steering the sampling process of a pretrained diffusion or flow-matching policy in response to out-of-distribution observation-language inputs without modifying policy parameters. By leveraging vision-language models to synthesize trajectory-differentiable reward functions, VLS guides denoising toward action trajectories that satisfy test-time spatial and task requirements. Across simulation and real-world evaluations, VLS consistently outperforms prior steering methods, achieving a 31% improvement on CALVIN and a 13% gain on LIBERO-PRO. Real-world deployment on a Franka robot further demonstrates robust inference-time adaptation under test-time spatial and semantic shifts. Project page: https://vision-language-steering.github.io/webpage/
>
---
#### [new 010] PDF-HR: Pose Distance Fields for Humanoid Robots
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出PDF-HR，用于人形机器人姿态优先建模。解决姿态合理性评估问题，通过连续距离场提升运动控制与优化效果。**

- **链接: [https://arxiv.org/pdf/2602.04851v1](https://arxiv.org/pdf/2602.04851v1)**

> **作者:** Yi Gu; Yukang Gao; Yangchen Zhou; Xingyu Chen; Yixiao Feng; Mingle Zhao; Yunyang Mo; Zhaorui Wang; Lixin Xu; Renjing Xu
>
> **备注:** \href{https://gaoyukang33.github.io/PDF-HR/}{Project page}
>
> **摘要:** Pose and motion priors play a crucial role in humanoid robotics. Although such priors have been widely studied in human motion recovery (HMR) domain with a range of models, their adoption for humanoid robots remains limited, largely due to the scarcity of high-quality humanoid motion data. In this work, we introduce Pose Distance Fields for Humanoid Robots (PDF-HR), a lightweight prior that represents the robot pose distribution as a continuous and differentiable manifold. Given an arbitrary pose, PDF-HR predicts its distance to a large corpus of retargeted robot poses, yielding a smooth measure of pose plausibility that is well suited for optimization and control. PDF-HR can be integrated as a reward shaping term, a regularizer, or a standalone plausibility scorer across diverse pipelines. We evaluate PDF-HR on various humanoid tasks, including single-trajectory motion tracking, general motion tracking, style-based motion mimicry, and general motion retargeting. Experiments show that this plug-and-play prior consistently and substantially strengthens strong baselines. Code and models will be released.
>
---
#### [new 011] Safe and Stylized Trajectory Planning for Autonomous Driving via Diffusion Model
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶轨迹规划任务，旨在解决安全与驾驶风格平衡的问题。提出SDD Planner框架，通过扩散模型实现安全且风格化的实时轨迹生成。**

- **链接: [https://arxiv.org/pdf/2602.04329v1](https://arxiv.org/pdf/2602.04329v1)**

> **作者:** Shuo Pei; Yong Wang; Yuanchen Zhu; Chen Sun; Qin Li; Yanan Zhao; Huachun Tan
>
> **备注:** 12 pages, 7 figures, submitted to IEEE Transactions on Intelligent Transportation Systems
>
> **摘要:** Achieving safe and stylized trajectory planning in complex real-world scenarios remains a critical challenge for autonomous driving systems. This paper proposes the SDD Planner, a diffusion-based framework designed to effectively reconcile safety constraints with driving styles in real time. The framework integrates two core modules: a Multi-Source Style-Aware Encoder, which employs distance-sensitive attention to fuse dynamic agent data and environmental contexts for heterogeneous safety-style perception; and a Style-Guided Dynamic Trajectory Generator, which adaptively modulates priority weights within the diffusion denoising process to generate user-preferred yet safe trajectories. Extensive experiments demonstrate that SDD Planner achieves state-of-the-art performance. On the StyleDrive benchmark, it improves the SM-PDMS metric by 3.9% over WoTE, the strongest baseline. Furthermore, on the NuPlan Test14 and Test14-hard benchmarks, SDD Planner ranks first with overall scores of 91.76 and 80.32, respectively, outperforming leading methods such as PLUTO. Real-vehicle closed-loop tests further confirm that SDD Planner maintains high safety standards while aligning with preset driving styles, validating its practical applicability for real-world deployment.
>
---
#### [new 012] Gust Estimation and Rejection with a Disturbance Observer for Proprioceptive Underwater Soft Morphing Wings
- **分类: cs.RO**

- **简介: 该论文属于水下机器人控制任务，旨在解决水流扰动影响车辆稳定的问题。通过仿生软翼与本体感觉传感结合，实现扰动估计与抑制。**

- **链接: [https://arxiv.org/pdf/2602.04438v1](https://arxiv.org/pdf/2602.04438v1)**

> **作者:** Tobias Cook; Leo Micklem; Huazhi Dong; Yunjie Yang; Michael Mistry; Francesco Giorgio Serchi
>
> **备注:** 2026 IEEE International Conference on Robotics & Automation (ICRA)
>
> **摘要:** Unmanned underwater vehicles are increasingly employed for maintenance and surveying tasks at sea, but their operation in shallow waters is often hindered by hydrodynamic disturbances such as waves, currents, and turbulence. These unsteady flows can induce rapid changes in direction and speed, compromising vehicle stability and manoeuvrability. Marine organisms contend with such conditions by combining proprioceptive feedback with flexible fins and tails to reject disturbances. Inspired by this strategy, we propose soft morphing wings endowed with proprioceptive sensing to mitigate environmental perturbations. The wing's continuous deformation provides a natural means to infer dynamic disturbances: sudden changes in camber directly reflect variations in the oncoming flow. By interpreting this proprioceptive signal, a disturbance observer can reconstruct flow parameters in real time. To enable this, we develop and experimentally validate a dynamic model of a hydraulically actuated soft wing with controllable camber. We then show that curvature-based sensing allows accurate estimation of disturbances in the angle of attack. Finally, we demonstrate that a controller leveraging these proprioceptive estimates can reject disturbances in the lift response of the soft wing. By combining proprioceptive sensing with a disturbance observer, this technique mirrors biological strategies and provides a pathway for soft underwater vehicles to maintain stability in hazardous environments.
>
---
#### [new 013] Dull, Dirty, Dangerous: Understanding the Past, Present, and Future of a Key Motivation for Robotics
- **分类: cs.RO**

- **简介: 该论文属于机器人学研究，旨在解决DDD概念在机器人应用中的定义与实践问题。通过分析文献，提出框架以更好理解机器人对劳动的影响。**

- **链接: [https://arxiv.org/pdf/2602.04746v1](https://arxiv.org/pdf/2602.04746v1)**

> **作者:** Nozomi Nakajima; Pedro Reynolds-Cuéllar; Caitrin Lynch; Kate Darling
>
> **摘要:** In robotics, the concept of "dull, dirty, and dangerous" (DDD) work has been used to motivate where robots might be useful. In this paper, we conduct an empirical analysis of robotics publications between 1980 and 2024 that mention DDD, and find that only 2.7% of publications define DDD and 8.7% of publications provide concrete examples of tasks or jobs that are DDD. We then review the social science literature on "dull," "dirty," and "dangerous" work to provide definitions and guidance on how to conceptualize DDD for robotics. Finally, we propose a framework that helps the robotics community consider the job context for our technology, encouraging a more informed perspective on how robotics may impact human labor.
>
---
#### [new 014] Radar-Inertial Odometry For Computationally Constrained Aerial Navigation
- **分类: cs.RO**

- **简介: 该论文属于无人机导航任务，解决在恶劣环境下精准定位问题。融合IMU与雷达数据，提出RIO算法，实现实时、低成本的导航状态估计。**

- **链接: [https://arxiv.org/pdf/2602.04631v1](https://arxiv.org/pdf/2602.04631v1)**

> **作者:** Jan Michalczyk
>
> **摘要:** Recently, the progress in the radar sensing technology consisting in the miniaturization of the packages and increase in measuring precision has drawn the interest of the robotics research community. Indeed, a crucial task enabling autonomy in robotics is to precisely determine the pose of the robot in space. To fulfill this task sensor fusion algorithms are often used, in which data from one or several exteroceptive sensors like, for example, LiDAR, camera, laser ranging sensor or GNSS are fused together with the Inertial Measurement Unit (IMU) measurements to obtain an estimate of the navigation states of the robot. Nonetheless, owing to their particular sensing principles, some exteroceptive sensors are often incapacitated in extreme environmental conditions, like extreme illumination or presence of fine particles in the environment like smoke or fog. Radars are largely immune to aforementioned factors thanks to the characteristics of electromagnetic waves they use. In this thesis, we present Radar-Inertial Odometry (RIO) algorithms to fuse the information from IMU and radar in order to estimate the navigation states of a (Uncrewed Aerial Vehicle) UAV capable of running on a portable resource-constrained embedded computer in real-time and making use of inexpensive, consumer-grade sensors. We present novel RIO approaches relying on the multi-state tightly-coupled Extended Kalman Filter (EKF) and Factor Graphs (FG) fusing instantaneous velocities of and distances to 3D points delivered by a lightweight, low-cost, off-the-shelf Frequency Modulated Continuous Wave (FMCW) radar with IMU readings. We also show a novel way to exploit advances in deep learning to retrieve 3D point correspondences in sparse and noisy radar point clouds.
>
---
#### [new 015] How Users Understand Robot Foundation Model Performance through Task Success Rates and Beyond
- **分类: cs.RO; cs.HC**

- **简介: 论文研究非专家用户如何理解机器人基础模型的性能，解决用户对模型能力认知不足的问题。通过实验分析用户对任务成功率及其他信息的使用情况。**

- **链接: [https://arxiv.org/pdf/2602.03920v1](https://arxiv.org/pdf/2602.03920v1)**

> **作者:** Isaac Sheidlower; Jindan Huang; James Staley; Bingyu Wu; Qicong Chen; Reuben Aronson; Elaine Short
>
> **备注:** Submitted to IJCAI 2026
>
> **摘要:** Robot Foundation Models (RFMs) represent a promising approach to developing general-purpose home robots. Given the broad capabilities of RFMs, users will inevitably ask an RFM-based robot to perform tasks that the RFM was not trained or evaluated on. In these cases, it is crucial that users understand the risks associated with attempting novel tasks due to the relatively high cost of failure. Furthermore, an informed user who understands an RFM's capabilities will know what situations and tasks the robot can handle. In this paper, we study how non-roboticists interpret performance information from RFM evaluations. These evaluations typically report task success rate (TSR) as the primary performance metric. While TSR is intuitive to experts, it is necessary to validate whether novices also use this information as intended. Toward this end, we conducted a study in which users saw real evaluation data, including TSR, failure case descriptions, and videos from multiple published RFM research projects. The results highlight that non-experts not only use TSR in a manner consistent with expert expectations but also highly value other information types, such as failure cases that are not often reported in RFM evaluations. Furthermore, we find that users want access to both real data from previous evaluations of the RFM and estimates from the robot about how well it will do on a novel task.
>
---
#### [new 016] Efficient Long-Horizon Vision-Language-Action Models via Static-Dynamic Disentanglement
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人控制任务，旨在解决VLA模型长周期上下文有限和推理效率低的问题。通过分离静态与动态视觉信息，提升模型效率和性能。**

- **链接: [https://arxiv.org/pdf/2602.03983v1](https://arxiv.org/pdf/2602.03983v1)**

> **作者:** Weikang Qiu; Tinglin Huang; Aosong Feng; Rex Ying
>
> **摘要:** Vision-Language-Action (VLA) models have recently emerged as a promising paradigm for generalist robotic control. Built upon vision-language model (VLM) architectures, VLAs predict actions conditioned on visual observations and language instructions, achieving strong performance and generalization across tasks. However, VLAs face two major challenges: limited long-horizon context and inefficient inference due to the quadratic attention complexity and large parameter counts. Our work is motivated by the observation that much of the visual information in a trajectory remains static across timesteps (e.g., the background). Leveraging this property, we propose SD-VLA, a framework that disentangles visual inputs into multi-level static and dynamic tokens, which enables (1) retaining a single copy of static tokens across frames to significantly reduce context length, and (2) reusing the key-value (KV) cache of static tokens through a lightweight recache gate that updates only when necessary. This design enables efficient multi-frame integration and efficient inference. In addition, we introduce a new benchmark that more effectively evaluates the long-horizon temporal dependency modeling ability of VLAs. Experimental results show that our approach outperforms baselines on this benchmark by 39.8% absolute improvement in success rate, and achieves a 3.9% gain on the SimplerEnv benchmark. Moreover, SD-VLA delivers a 2.26x inference speedup over the base VLA model on the same benchmark, enabling faster and more practical real-world deployment.
>
---
#### [new 017] ALORE: Autonomous Large-Object Rearrangement with a Legged Manipulator
- **分类: cs.RO**

- **简介: 该论文提出ALORE系统，解决腿式机械臂自主 rearrange 大型物体的任务。针对多物体、复杂环境下的高效、无碰撞操作问题，设计了分层强化学习、统一交互表示和任务规划框架，提升系统通用性与效率。**

- **链接: [https://arxiv.org/pdf/2602.04214v1](https://arxiv.org/pdf/2602.04214v1)**

> **作者:** Zhihai Bi; Yushan Zhang; Kai Chen; Guoyang Zhao; Yulin Li; Jun Ma
>
> **摘要:** Endowing robots with the ability to rearrange various large and heavy objects, such as furniture, can substantially alleviate human workload. However, this task is extremely challenging due to the need to interact with diverse objects and efficiently rearrange multiple objects in complex environments while ensuring collision-free loco-manipulation. In this work, we present ALORE, an autonomous large-object rearrangement system for a legged manipulator that can rearrange various large objects across diverse scenarios. The proposed system is characterized by three main features: (i) a hierarchical reinforcement learning training pipeline for multi-object environment learning, where a high-level object velocity controller is trained on top of a low-level whole-body controller to achieve efficient and stable joint learning across multiple objects; (ii) two key modules, a unified interaction configuration representation and an object velocity estimator, that allow a single policy to regulate planar velocity of diverse objects accurately; and (iii) a task-and-motion planning framework that jointly optimizes object visitation order and object-to-target assignment, improving task efficiency while enabling online replanning. Comparisons against strong baselines show consistent superiority in policy generalization, object-velocity tracking accuracy, and multi-object rearrangement efficiency. Key modules are systematically evaluated, and extensive simulations and real-world experiments are conducted to validate the robustness and effectiveness of the entire system, which successfully completes 8 continuous loops to rearrange 32 chairs over nearly 40 minutes without a single failure, and executes long-distance autonomous rearrangement over an approximately 40 m route. The open-source packages are available at https://zhihaibi.github.io/Alore/.
>
---
#### [new 018] Shaping Expressiveness in Robotics: The Role of Design Tools in Crafting Embodied Robot Movements
- **分类: cs.RO**

- **简介: 该论文属于机器人表达性运动设计任务，旨在解决如何将人类意图转化为机器人具身表达的问题。通过设计工具和交互方法，提升机器人运动的直观性和吸引力。**

- **链接: [https://arxiv.org/pdf/2602.04137v1](https://arxiv.org/pdf/2602.04137v1)**

> **作者:** Elisabetta Zibetti; Alexandra Mercader; Hélène Duval; Florent Levillain; Audrey Rochette; David St-Onge
>
> **摘要:** As robots increasingly become part of shared human spaces, their movements must transcend basic functionality by incorporating expressive qualities to enhance engagement and communication. This paper introduces a movement-centered design pedagogy designed to support engineers in creating expressive robotic arm movements. Through a hands-on interactive workshop informed by interdisciplinary methodologies, participants explored various creative possibilities, generating valuable insights into expressive motion design. The iterative approach proposed integrates analytical frameworks from dance, enabling designers to examine motion through dynamic and embodied dimensions. A custom manual remote controller facilitates interactive, real-time manipulation of the robotic arm, while dedicated animation software supports visualization, detailed motion sequencing, and precise parameter control. Qualitative analysis of this interactive design process reveals that the proposed "toolbox" effectively bridges the gap between human intent and robotic expressiveness resulting in more intuitive and engaging expressive robotic arm movements.
>
---
#### [new 019] Relational Scene Graphs for Object Grounding of Natural Language Commands
- **分类: cs.RO**

- **简介: 该论文属于自然语言指令理解任务，旨在提升机器人对自然语言命令的物体定位能力。通过引入空间关系增强3D场景图，改进LLM的物体接地效果。**

- **链接: [https://arxiv.org/pdf/2602.04635v1](https://arxiv.org/pdf/2602.04635v1)**

> **作者:** Julia Kuhn; Francesco Verdoja; Tsvetomila Mihaylova; Ville Kyrki
>
> **备注:** In review for RA-L
>
> **摘要:** Robots are finding wider adoption in human environments, increasing the need for natural human-robot interaction. However, understanding a natural language command requires the robot to infer the intended task and how to decompose it into executable actions, and to ground those actions in the robot's knowledge of the environment, including relevant objects, agents, and locations. This challenge can be addressed by combining the capabilities of Large language models (LLMs) to understand natural language with 3D scene graphs (3DSGs) for grounding inferred actions in a semantic representation of the environment. However, many 3DSGs lack explicit spatial relations between objects, even though humans often rely on these relations to describe an environment. This paper investigates whether incorporating open- or closed-vocabulary spatial relations into 3DSGs can improve the ability of LLMs to interpret natural language commands. To address this, we propose an LLM-based pipeline for target object grounding from open-vocabulary language commands and a vision language model (VLM)-based pipeline to add open-vocabulary spatial edges to 3DSGs from images captured while mapping. Finally, two LLMs are evaluated in a study assessing their performance on the downstream task of target object grounding. Our study demonstrates that explicit spatial relations improve the ability of LLMs to ground objects. Moreover, open-vocabulary relation generation with VLMs proves feasible from robot-captured images, but their advantage over closed-vocabulary relations is found to be limited.
>
---
#### [new 020] KGLAMP: Knowledge Graph-guided Language model for Adaptive Multi-robot Planning and Replanning
- **分类: cs.RO; cs.AI; cs.ET; cs.MA**

- **简介: 该论文提出KGLAMP，用于异构多机器人系统的自适应路径规划与重规划任务。解决动态环境中符号表示不准确和计划不一致的问题，通过知识图谱引导语言模型生成精确的PDDL规范。**

- **链接: [https://arxiv.org/pdf/2602.04129v1](https://arxiv.org/pdf/2602.04129v1)**

> **作者:** Chak Lam Shek; Faizan M. Tariq; Sangjae Bae; David Isele; Piyush Gupta
>
> **摘要:** Heterogeneous multi-robot systems are increasingly deployed in long-horizon missions that require coordination among robots with diverse capabilities. However, existing planning approaches struggle to construct accurate symbolic representations and maintain plan consistency in dynamic environments. Classical PDDL planners require manually crafted symbolic models, while LLM-based planners often ignore agent heterogeneity and environmental uncertainty. We introduce KGLAMP, a knowledge-graph-guided LLM planning framework for heterogeneous multi-robot teams. The framework maintains a structured knowledge graph encoding object relations, spatial reachability, and robot capabilities, which guides the LLM in generating accurate PDDL problem specifications. The knowledge graph serves as a persistent, dynamically updated memory that incorporates new observations and triggers replanning upon detecting inconsistencies, enabling symbolic plans to adapt to evolving world states. Experiments on the MAT-THOR benchmark show that KGLAMP improves performance by at least 25.5% over both LLM-only and PDDL-based variants.
>
---
#### [new 021] GenMRP: A Generative Multi-Route Planning Framework for Efficient and Personalized Real-Time Industrial Navigation
- **分类: cs.RO; cs.GR; cs.IR**

- **简介: 该论文属于工业导航任务，解决实时路径规划中效率与个性化不足的问题。提出GenMRP框架，通过生成多路径提升效率与多样性。**

- **链接: [https://arxiv.org/pdf/2602.04174v1](https://arxiv.org/pdf/2602.04174v1)**

> **作者:** Chengzhang Wang; Chao Chen; Jun Tao; Tengfei Liu; He Bai; Song Wang; Longfei Xu; Kaikui Liu; Xiangxiang Chu
>
> **摘要:** Existing industrial-scale navigation applications contend with massive road networks, typically employing two main categories of approaches for route planning. The first relies on precomputed road costs for optimal routing and heuristic algorithms for generating alternatives, while the second, generative methods, has recently gained significant attention. However, the former struggles with personalization and route diversity, while the latter fails to meet the efficiency requirements of large-scale real-time scenarios. To address these limitations, we propose GenMRP, a generative framework for multi-route planning. To ensure generation efficiency, GenMRP first introduces a skeleton-to-capillary approach that dynamically constructs a relevant sub-network significantly smaller than the full road network. Within this sub-network, routes are generated iteratively. The first iteration identifies the optimal route, while the subsequent ones generate alternatives that balance quality and diversity using the newly proposed correctional boosting approach. Each iteration incorporates road features, user historical sequences, and previously generated routes into a Link Cost Model to update road costs, followed by route generation using the Dijkstra algorithm. Extensive experiments show that GenMRP achieves state-of-the-art performance with high efficiency in both offline and online environments. To facilitate further research, we have publicly released the training and evaluation dataset. GenMRP has been fully deployed in a real-world navigation app, demonstrating its effectiveness and benefits.
>
---
#### [new 022] Quantile Transfer for Reliable Operating Point Selection in Visual Place Recognition
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉定位任务，解决VPR系统中阈值选择问题。通过量化转移方法自动选择操作点，提高召回率并适应环境变化。**

- **链接: [https://arxiv.org/pdf/2602.04401v1](https://arxiv.org/pdf/2602.04401v1)**

> **作者:** Dhyey Manish Rajani; Michael Milford; Tobias Fischer
>
> **摘要:** Visual Place Recognition (VPR) is a key component for localisation in GNSS-denied environments, but its performance critically depends on selecting an image matching threshold (operating point) that balances precision and recall. Thresholds are typically hand-tuned offline for a specific environment and fixed during deployment, leading to degraded performance under environmental change. We propose a method that, given a user-defined precision requirement, automatically selects the operating point of a VPR system to maximise recall. The method uses a small calibration traversal with known correspondences and transfers thresholds to deployment via quantile normalisation of similarity score distributions. This quantile transfer ensures that thresholds remain stable across calibration sizes and query subsets, making the method robust to sampling variability. Experiments with multiple state-of-the-art VPR techniques and datasets show that the proposed approach consistently outperforms the state-of-the-art, delivering up to 25% higher recall in high-precision operating regimes. The method eliminates manual tuning by adapting to new environments and generalising across operating conditions. Our code will be released upon acceptance.
>
---
#### [new 023] Towards Next-Generation SLAM: A Survey on 3DGS-SLAM Focusing on Performance, Robustness, and Future Directions
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM任务，旨在解决传统SLAM在动态环境中的性能与鲁棒性问题，通过3DGS技术提升重建质量与效率。**

- **链接: [https://arxiv.org/pdf/2602.04251v1](https://arxiv.org/pdf/2602.04251v1)**

> **作者:** Li Wang; Ruixuan Gong; Yumo Han; Lei Yang; Lu Yang; Ying Li; Bin Xu; Huaping Liu; Rong Fu
>
> **摘要:** Traditional Simultaneous Localization and Mapping (SLAM) systems often face limitations including coarse rendering quality, insufficient recovery of scene details, and poor robustness in dynamic environments. 3D Gaussian Splatting (3DGS), with its efficient explicit representation and high-quality rendering capabilities, offers a new reconstruction paradigm for SLAM. This survey comprehensively reviews key technical approaches for integrating 3DGS with SLAM. We analyze performance optimization of representative methods across four critical dimensions: rendering quality, tracking accuracy, reconstruction speed, and memory consumption, delving into their design principles and breakthroughs. Furthermore, we examine methods for enhancing the robustness of 3DGS-SLAM in complex environments such as motion blur and dynamic environments. Finally, we discuss future challenges and development trends in this area. This survey aims to provide a technical reference for researchers and foster the development of next-generation SLAM systems characterized by high fidelity, efficiency, and robustness.
>
---
#### [new 024] Can We Redesign a Shoulder Exosuit to Enhance Comfort and Usability Without Losing Assistance?
- **分类: cs.RO**

- **简介: 该论文属于康复工程任务，旨在提升肩部外骨骼的舒适性与实用性。研究设计了新型软式肩部外骨骼（v2），在不牺牲辅助效果的前提下，改善了佩戴舒适度和功能表现。**

- **链接: [https://arxiv.org/pdf/2602.04625v1](https://arxiv.org/pdf/2602.04625v1)**

> **作者:** Roberto Ferroni; Daniele Filippo Mauceri; Jacopo Carpaneto; Alessandra Pedrocchi; Tommaso Proietti
>
> **摘要:** Reduced shoulder mobility limits upper-limb function and the performance of activities of daily living across a wide range of conditions. Wearable exosuits have shown promise in assisting arm elevation, reducing muscle effort, and supporting functional movements; however, comfort is rarely prioritized as an explicit design objective, despite it strongly affects real-life, long-term usage. This study presents a redesigned soft shoulder exosuit (Soft Shoulder v2) developed to address comfort-related limitations identified in our previous version, while preserving assistive performance. In parallel, assistance was also improved, shifting from the coronal plane to the sagittal plane to better support functionally relevant hand positioning. A controlled comparison between the previous (v1) and redesigned (v2) modules was conducted in eight healthy participants, who performed static holding, dynamic lifting, and a functional pick and place task. Muscle activity, kinematics, and user-reported outcomes were assessed. Both versions increased endurance time, reduced deltoid activation, and preserved transparency during unpowered shoulder elevation. However, the difference between them emerged most clearly during functional tasks and comfort evaluation. The redesigned module facilitated forward arm positioning and increased transverse plane mobility by up to 30 deg, without increasing muscular demand. User-reported outcomes further indicated a substantial improvement in wearability, with markedly lower perceived pressure and higher ratings in effectiveness, ease of use, and comfort compared to the previous design. Taken together, these findings show that targeted, user-centered design refinements can improve comfort and functional interaction without compromising assistive performance, advancing the development of soft exosuits suitable for prolonged and daily use.
>
---
#### [new 025] Beyond the Vehicle: Cooperative Localization by Fusing Point Clouds for GPS-Challenged Urban Scenarios
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于定位任务，解决GPS信号弱的城区环境下的车辆定位问题。通过融合V2V/V2I数据与点云SLAM，提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.03908v1](https://arxiv.org/pdf/2602.03908v1)**

> **作者:** Kuo-Yi Chao; Ralph Rasshofer; Alois Christian Knoll
>
> **备注:** 8 pages, 2 figures, Driving the Future Symposium 2025
>
> **摘要:** Accurate vehicle localization is a critical challenge in urban environments where GPS signals are often unreliable. This paper presents a cooperative multi-sensor and multi-modal localization approach to address this issue by fusing data from vehicle-to-vehicle (V2V) and vehicle-to-infrastructure (V2I) systems. Our approach integrates cooperative data with a point cloud registration-based simultaneous localization and mapping (SLAM) algorithm. The system processes point clouds generated from diverse sensor modalities, including vehicle-mounted LiDAR and stereo cameras, as well as sensors deployed at intersections. By leveraging shared data from infrastructure, our method significantly improves localization accuracy and robustness in complex, GPS-noisy urban scenarios.
>
---
#### [new 026] HoRD: Robust Humanoid Control via History-Conditioned Reinforcement Learning and Online Distillation
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决人形机器人在动态变化环境中的适应性问题。通过两阶段学习框架HoRD，提升其鲁棒性和迁移能力。**

- **链接: [https://arxiv.org/pdf/2602.04412v1](https://arxiv.org/pdf/2602.04412v1)**

> **作者:** Puyue Wang; Jiawei Hu; Yan Gao; Junyan Wang; Yu Zhang; Gillian Dobbie; Tao Gu; Wafa Johal; Ting Dang; Hong Jia
>
> **摘要:** Humanoid robots can suffer significant performance drops under small changes in dynamics, task specifications, or environment setup. We propose HoRD, a two-stage learning framework for robust humanoid control under domain shift. First, we train a high-performance teacher policy via history-conditioned reinforcement learning, where the policy infers latent dynamics context from recent state--action trajectories to adapt online to diverse randomized dynamics. Second, we perform online distillation to transfer the teacher's robust control capabilities into a transformer-based student policy that operates on sparse root-relative 3D joint keypoint trajectories. By combining history-conditioned adaptation with online distillation, HoRD enables a single policy to adapt zero-shot to unseen domains without per-domain retraining. Extensive experiments show HoRD outperforms strong baselines in robustness and transfer, especially under unseen domains and external perturbations. Code and project page are available at \href{https://tonywang-0517.github.io/hord/}{https://tonywang-0517.github.io/hord/}.
>
---
#### [new 027] A Unified Complementarity-based Approach for Rigid-Body Manipulation and Motion Prediction
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决非结构环境中自由运动与摩擦接触的联合建模问题。提出统一框架Unicomp，结合互补理论，实现物理一致的接触模式转换。**

- **链接: [https://arxiv.org/pdf/2602.04522v1](https://arxiv.org/pdf/2602.04522v1)**

> **作者:** Bingkun Huang; Xin Ma; Nilanjan Chakraborty; Riddhiman Laha
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Robotic manipulation in unstructured environments requires planners to reason jointly about free-space motion and sustained, frictional contact with the environment. Existing (local) planning and simulation frameworks typically separate these regimes or rely on simplified contact representations, particularly when modeling non-convex or distributed contact patches. Such approximations limit the fidelity of contact-mode transitions and hinder the robust execution of contact-rich behaviors in real time. This paper presents a unified discrete-time modeling framework for robotic manipulation that consistently captures both free motion and frictional contact within a single mathematical formalism (Unicomp). Building on complementarity-based rigid-body dynamics, we formulate free-space motion and contact interactions as coupled linear and nonlinear complementarity problems, enabling principled transitions between contact modes without enforcing fixed-contact assumptions. For planar patch contact, we derive a frictional contact model from the maximum power dissipation principle in which the set of admissible contact wrenches is represented by an ellipsoidal limit surface. This representation captures coupled force-moment effects, including torsional friction, while remaining agnostic to the underlying pressure distribution across the contact patch. The resulting formulation yields a discrete-time predictive model that relates generalized velocities and contact wrenches through quadratic constraints and is suitable for real-time optimization-based planning. Experimental results show that the proposed approach enables stable, physically consistent behavior at interactive speeds across tasks, from planar pushing to contact-rich whole-body maneuvers.
>
---
#### [new 028] Integrated Exploration and Sequential Manipulation on Scene Graph with LLM-based Situated Replanning
- **分类: cs.RO**

- **简介: 该论文属于机器人任务规划与导航领域，解决部分已知环境中探索与操作的协同问题。提出EPoG框架，结合图规划与大语言模型，实现高效路径规划与任务执行。**

- **链接: [https://arxiv.org/pdf/2602.04419v1](https://arxiv.org/pdf/2602.04419v1)**

> **作者:** Heqing Yang; Ziyuan Jiao; Shu Wang; Yida Niu; Si Liu; Hangxin Liu
>
> **备注:** 8 pages, 7 figures; accepted by ICRA 2026
>
> **摘要:** In partially known environments, robots must combine exploration to gather information with task planning for efficient execution. To address this challenge, we propose EPoG, an Exploration-based sequential manipulation Planning framework on Scene Graphs. EPoG integrates a graph-based global planner with a Large Language Model (LLM)-based situated local planner, continuously updating a belief graph using observations and LLM predictions to represent known and unknown objects. Action sequences are generated by computing graph edit operations between the goal and belief graphs, ordered by temporal dependencies and movement costs. This approach seamlessly combines exploration and sequential manipulation planning. In ablation studies across 46 realistic household scenes and 5 long-horizon daily object transportation tasks, EPoG achieved a success rate of 91.3%, reducing travel distance by 36.1% on average. Furthermore, a physical mobile manipulator successfully executed complex tasks in unknown and dynamic environments, demonstrating EPoG's potential for real-world applications.
>
---
#### [new 029] Comparative Analysis of Autonomous Robotic and Manual Techniques for Ultrasonic Sacral Osteotomy: A Preliminary Study
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决手动骨切术精度不足的问题。通过对比自主机器人与手动操作的超声骨切效果，验证了机器人系统的高精度和稳定性。**

- **链接: [https://arxiv.org/pdf/2602.04076v1](https://arxiv.org/pdf/2602.04076v1)**

> **作者:** Daniyal Maroufi; Yash Kulkarni; Justin E. Bird; Jeffrey H. Siewerdsen; Farshid Alambeigi
>
> **备注:** 17 pages, 6 figures, Accepted or publication in 2026 International Symposium on Medical Robotics (ISMR)
>
> **摘要:** In this paper, we introduce an autonomous Ultrasonic Sacral Osteotomy (USO) robotic system that integrates an ultrasonic osteotome with a seven-degree-of-freedom (DoF) robotic manipulator guided by an optical tracking system. To assess multi-directional control along both the surface trajectory and cutting depth of this system, we conducted quantitative comparisons between manual USO (MUSO) and robotic USO (RUSO) in Sawbones phantoms under identical osteotomy conditions. The RUSO system achieved sub-millimeter trajectory accuracy (0.11 mm RMSE), an order of magnitude improvement over MUSO (1.10 mm RMSE). Moreover, MUSO trials showed substantial over-penetration (16.0 mm achieved vs. 8.0 mm target), whereas the RUSO system maintained precise depth control (8.1 mm). These results demonstrate that robotic procedures can effectively overcome the critical limitations of manual osteotomy, establishing a foundation for safer and more precise sacral resections.
>
---
#### [new 030] Viewpoint Matters: Dynamically Optimizing Viewpoints with Masked Autoencoder for Visual Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉任务，解决单摄像头系统视角固定导致的适应性差问题。通过动态选择最优视角，提升机器人操作能力。**

- **链接: [https://arxiv.org/pdf/2602.04243v1](https://arxiv.org/pdf/2602.04243v1)**

> **作者:** Pengfei Yi; Yifan Han; Junyan Li; Litao Liu; Wenzhao Lian
>
> **备注:** 5 pages, 2 figures, 3 tables
>
> **摘要:** Robotic manipulation continues to be a challenge, and imitation learning (IL) enables robots to learn tasks from expert demonstrations. Current IL methods typically rely on fixed camera setups, where cameras are manually positioned in static locations, imposing significant limitations on adaptability and coverage. Inspired by human active perception, where humans dynamically adjust their viewpoint to capture the most relevant and least noisy information, we propose MAE-Select, a novel framework for active viewpoint selection in single-camera robotic systems. MAE-Select fully leverages pre-trained multi-view masked autoencoder representations and dynamically selects the next most informative viewpoint at each time chunk without requiring labeled viewpoints. Extensive experiments demonstrate that MAE-Select improves the capabilities of single-camera systems and, in some cases, even surpasses multi-camera setups. The project will be available at https://mae-select.github.io.
>
---
#### [new 031] From Vision to Assistance: Gaze and Vision-Enabled Adaptive Control for a Back-Support Exoskeleton
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在提升背支持外骨骼的辅助效果。通过融合视觉与凝视追踪，实现更及时、适应性的控制，解决现有方法依赖负载估计或缺乏上下文感知的问题。**

- **链接: [https://arxiv.org/pdf/2602.04648v1](https://arxiv.org/pdf/2602.04648v1)**

> **作者:** Alessandro Leanza; Paolo Franceschi; Blerina Spahiu; Loris Roveda
>
> **摘要:** Back-support exoskeletons have been proposed to mitigate spinal loading in industrial handling, yet their effectiveness critically depends on timely and context-aware assistance. Most existing approaches rely either on load-estimation techniques (e.g., EMG, IMU) or on vision systems that do not directly inform control. In this work, we present a vision-gated control framework for an active lumbar occupational exoskeleton that leverages egocentric vision with wearable gaze tracking. The proposed system integrates real-time grasp detection from a first-person YOLO-based perception system, a finite-state machine (FSM) for task progression, and a variable admittance controller to adapt torque delivery to both posture and object state. A user study with 15 participants performing stooping load lifting trials under three conditions (no exoskeleton, exoskeleton without vision, exoskeleton with vision) shows that vision-gated assistance significantly reduces perceived physical demand and improves fluency, trust, and comfort. Quantitative analysis reveals earlier and stronger assistance when vision is enabled, while questionnaire results confirm user preference for the vision-gated mode. These findings highlight the potential of egocentric vision to enhance the responsiveness, ergonomics, safety, and acceptance of back-support exoskeletons.
>
---
#### [new 032] EgoActor: Grounding Task Planning into Spatial-aware Egocentric Actions for Humanoid Robots via Visual-Language Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出EgoActor，解决人形机器人在部分信息环境下任务规划与空间感知动作执行的问题，通过视觉语言模型实现高效、实时的行动推理。**

- **链接: [https://arxiv.org/pdf/2602.04515v1](https://arxiv.org/pdf/2602.04515v1)**

> **作者:** Yu Bai; MingMing Yu; Chaojie Li; Ziyi Bai; Xinlong Wang; Börje F. Karlsson
>
> **摘要:** Deploying humanoid robots in real-world settings is fundamentally challenging, as it demands tight integration of perception, locomotion, and manipulation under partial-information observations and dynamically changing environments. As well as transitioning robustly between sub-tasks of different types. Towards addressing these challenges, we propose a novel task - EgoActing, which requires directly grounding high-level instructions into various, precise, spatially aware humanoid actions. We further instantiate this task by introducing EgoActor, a unified and scalable vision-language model (VLM) that can predict locomotion primitives (e.g., walk, turn, move sideways, change height), head movements, manipulation commands, and human-robot interactions to coordinate perception and execution in real-time. We leverage broad supervision over egocentric RGB-only data from real-world demonstrations, spatial reasoning question-answering, and simulated environment demonstrations, enabling EgoActor to make robust, context-aware decisions and perform fluent action inference (under 1s) with both 8B and 4B parameter models. Extensive evaluations in both simulated and real-world environments demonstrate that EgoActor effectively bridges abstract task planning and concrete motor execution, while generalizing across diverse tasks and unseen environments.
>
---
#### [new 033] Reshaping Action Error Distributions for Reliable Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决连续动作预测中的误差分布问题，引入最小误差熵方法提升模型可靠性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.04228v1](https://arxiv.org/pdf/2602.04228v1)**

> **作者:** Shuanghao Bai; Dakai Wang; Cheng Chi; Wanqi Zhou; Jing Lyu; Xiaoguang Zhao; Pengwei Wang; Zhongyuan Wang; Lei Xing; Shanghang Zhang; Badong Chen
>
> **摘要:** In robotic manipulation, vision-language-action (VLA) models have emerged as a promising paradigm for learning generalizable and scalable robot policies. Most existing VLA frameworks rely on standard supervised objectives, typically cross-entropy for discrete actions and mean squared error (MSE) for continuous action regression, which impose strong pointwise constraints on individual predictions. In this work, we focus on continuous-action VLA models and move beyond conventional MSE-based regression by reshaping action error distributions during training. Drawing on information-theoretic principles, we introduce Minimum Error Entropy (MEE) into modern VLA architectures and propose a trajectory-level MEE objective, together with two weighted variants, combined with MSE for continuous-action VLA training. We evaluate our approaches across standard, few-shot, and noisy settings on multiple representative VLA architectures, using simulation benchmarks such as LIBERO and SimplerEnv as well as real-world robotic manipulation tasks. Experimental results demonstrate consistent improvements in success rates and robustness across these settings. Under imbalanced data regimes, the gains persist within a well-characterized operating range, while incurring negligible additional training cost and no impact on inference efficiency. We further provide theoretical analyses that explain why MEE-based supervision is effective and characterize its practical range. Project Page: https://cognition2actionlab.github.io/VLA-TMEE.github.io/
>
---
#### [new 034] Capturing Visual Environment Structure Correlates with Control Performance
- **分类: cs.RO**

- **简介: 该论文研究视觉表示对机器人控制性能的影响，旨在解决如何有效评估视觉编码器的问题。通过分析编码器对环境状态的解码能力，提升策略泛化性。**

- **链接: [https://arxiv.org/pdf/2602.04880v1](https://arxiv.org/pdf/2602.04880v1)**

> **作者:** Jiahua Dong; Yunze Man; Pavel Tokmakov; Yu-Xiong Wang
>
> **摘要:** The choice of visual representation is key to scaling generalist robot policies. However, direct evaluation via policy rollouts is expensive, even in simulation. Existing proxy metrics focus on the representation's capacity to capture narrow aspects of the visual world, like object shape, limiting generalization across environments. In this paper, we take an analytical perspective: we probe pretrained visual encoders by measuring how well they support decoding of environment state -- including geometry, object structure, and physical attributes -- from images. Leveraging simulation environments with access to ground-truth state, we show that this probing accuracy strongly correlates with downstream policy performance across diverse environments and learning settings, significantly outperforming prior metrics and enabling efficient representation selection. More broadly, our study provides insight into the representational properties that support generalizable manipulation, suggesting that learning to encode the latent physical state of the environment is a promising objective for control.
>
---
#### [new 035] AppleVLM: End-to-end Autonomous Driving with Advanced Perception and Planning-Enhanced Vision-Language Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，旨在解决端到端驾驶中的感知与规划问题。提出AppleVLM模型，融合视觉、语言和规划信息，提升驾驶性能。**

- **链接: [https://arxiv.org/pdf/2602.04256v1](https://arxiv.org/pdf/2602.04256v1)**

> **作者:** Yuxuan Han; Kunyuan Wu; Qianyi Shao; Renxiang Xiao; Zilu Wang; Cansen Jiang; Yi Xiao; Liang Hu; Yunjiang Lou
>
> **摘要:** End-to-end autonomous driving has emerged as a promising paradigm integrating perception, decision-making, and control within a unified learning framework. Recently, Vision-Language Models (VLMs) have gained significant attention for their potential to enhance the robustness and generalization of end-to-end driving models in diverse and unseen scenarios. However, existing VLM-based approaches still face challenges, including suboptimal lane perception, language understanding biases, and difficulties in handling corner cases. To address these issues, we propose AppleVLM, an advanced perception and planning-enhanced VLM model for robust end-to-end driving. AppleVLM introduces a novel vision encoder and a planning strategy encoder to improve perception and decision-making. Firstly, the vision encoder fuses spatial-temporal information from multi-view images across multiple timesteps using a deformable transformer mechanism, enhancing robustness to camera variations and facilitating scalable deployment across different vehicle platforms. Secondly, unlike traditional VLM-based approaches, AppleVLM introduces a dedicated planning modality that encodes explicit Bird's-Eye-View spatial information, mitigating language biases in navigation instructions. Finally, a VLM decoder fine-tuned by a hierarchical Chain-of-Thought integrates vision, language, and planning features to output robust driving waypoints. We evaluate AppleVLM in closed-loop experiments on two CARLA benchmarks, achieving state-of-the-art driving performance. Furthermore, we deploy AppleVLM on an AGV platform and successfully showcase real-world end-to-end autonomous driving in complex outdoor environments.
>
---
#### [new 036] OAT: Ordered Action Tokenization
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出OAT，解决机器人动作离散化问题，属于机器人学习任务。通过有序动作分词，提升自回归策略的效率与灵活性。**

- **链接: [https://arxiv.org/pdf/2602.04215v1](https://arxiv.org/pdf/2602.04215v1)**

> **作者:** Chaoqi Liu; Xiaoshen Han; Jiawei Gao; Yue Zhao; Haonan Chen; Yilun Du
>
> **摘要:** Autoregressive policies offer a compelling foundation for scalable robot learning by enabling discrete abstraction, token-level reasoning, and flexible inference. However, applying autoregressive modeling to continuous robot actions requires an effective action tokenization scheme. Existing approaches either rely on analytical discretization methods that produce prohibitively long token sequences, or learned latent tokenizers that lack structure, limiting their compatibility with next-token prediction. In this work, we identify three desiderata for action tokenization - high compression, total decodability, and a left-to-right causally ordered token space - and introduce Ordered Action Tokenization (OAT), a learned action tokenizer that satisfies all three. OAT discretizes action chunks into an ordered sequence of tokens using transformer with registers, finite scalar quantization, and ordering-inducing training mechanisms. The resulting token space aligns naturally with autoregressive generation and enables prefix-based detokenization, yielding an anytime trade-off between inference cost and action fidelity. Across more than 20 tasks spanning four simulation benchmarks and real-world settings, autoregressive policies equipped with OAT consistently outperform prior tokenization schemes and diffusion-based baselines, while offering significantly greater flexibility at inference time.
>
---
#### [new 037] An Anatomy-specific Guidewire Shaping Robot for Improved Vascular Navigation
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决神经血管导航中导丝形状依赖医生经验的问题。提出一种可自主塑形的导丝机器人，实现精准、标准化的导丝配置。**

- **链接: [https://arxiv.org/pdf/2602.04050v1](https://arxiv.org/pdf/2602.04050v1)**

> **作者:** Aabha Tamhankar; Jay Patil; Giovanni Pittiglio
>
> **备注:** 7 pages, 7 figures, ISMR2026
>
> **摘要:** Neuroendovascular access often relies on passive microwires that are hand-shaped at the back table and then used to track a microcatheter to the target. Neuroendovascular surgeons determine the shape of the wire by examining the patient pre-operative images and using their experience to identify anatomy specific shapes of the wire that would facilitate reaching the target. This procedure is particularly complex in convoluted anatomical structures and is heavily dependent on the level of expertise of the surgeon. Towards enabling standardized autonomous shaping, we present a bench-top guidewire shaping robot capable of producing navigation-specific desired wire configurations. We present a model that can map the desired wire shape into robot actions, calibrated using experimental data. We show that the robot can produce clinically common tip geometries (C, S, Angled, Hook) and validate them with respect to the model-predicted shapes in 2D. Our model predicts the shape with a Root Mean Square (RMS) error of 0.56mm across all shapes when compared to the experimental results. We also demonstrate 3D tip shaping capabilities and the ability to traverse complex endoluminal navigation from the petrous Internal Carotid Artery (ICA) to the Posterior Communicating Artery (PComm).
>
---
#### [new 038] A Modern System Recipe for Situated Embodied Human-Robot Conversation with Real-Time Multimodal LLMs and Tool-Calling
- **分类: cs.RO**

- **简介: 该论文属于人机对话任务，解决机器人在实时场景中进行具身对话的问题。通过结合多模态大模型和工具调用，提升机器人的感知与交互能力。**

- **链接: [https://arxiv.org/pdf/2602.04157v1](https://arxiv.org/pdf/2602.04157v1)**

> **作者:** Dong Won Lee; Sarah Gillet; Louis-Philippe Morency; Cynthia Breazeal; Hae Won Park
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Situated embodied conversation requires robots to interleave real-time dialogue with active perception: deciding what to look at, when to look, and what to say under tight latency constraints. We present a simple, minimal system recipe that pairs a real-time multimodal language model with a small set of tool interfaces for attention and active perception. We study six home-style scenarios that require frequent attention shifts and increasing perceptual scope. Across four system variants, we evaluate turn-level tool-decision correctness against human annotations and collect subjective ratings of interaction quality. Results indicate that real-time multimodal large language models and tool use for active perception is a promising direction for practical situated embodied conversation.
>
---
#### [new 039] DADP: Domain Adaptive Diffusion Policy
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习中的领域自适应任务，旨在解决策略在未见动态环境下泛化能力差的问题。通过解耦静态领域信息与动态特性，提升策略的零样本适应能力。**

- **链接: [https://arxiv.org/pdf/2602.04037v1](https://arxiv.org/pdf/2602.04037v1)**

> **作者:** Pengcheng Wang; Qinghang Liu; Haotian Lin; Yiheng Li; Guojian Zhan; Masayoshi Tomizuka; Yixiao Wang
>
> **摘要:** Learning domain adaptive policies that can generalize to unseen transition dynamics, remains a fundamental challenge in learning-based control. Substantial progress has been made through domain representation learning to capture domain-specific information, thus enabling domain-aware decision making. We analyze the process of learning domain representations through dynamical prediction and find that selecting contexts adjacent to the current step causes the learned representations to entangle static domain information with varying dynamical properties. Such mixture can confuse the conditioned policy, thereby constraining zero-shot adaptation. To tackle the challenge, we propose DADP (Domain Adaptive Diffusion Policy), which achieves robust adaptation through unsupervised disentanglement and domain-aware diffusion injection. First, we introduce Lagged Context Dynamical Prediction, a strategy that conditions future state estimation on a historical offset context; by increasing this temporal gap, we unsupervisedly disentangle static domain representations by filtering out transient properties. Second, we integrate the learned domain representations directly into the generative process by biasing the prior distribution and reformulating the diffusion target. Extensive experiments on challenging benchmarks across locomotion and manipulation demonstrate the superior performance, and the generalizability of DADP over prior methods. More visualization results are available on the https://outsider86.github.io/DomainAdaptiveDiffusionPolicy/.
>
---
#### [new 040] Towards X-embodiment safety: A control theory perspective on transferring safety certificates across dynamical systems
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于控制安全领域，解决不同动力系统间安全证书转移问题。提出tCBF框架，通过模拟函数和边界项实现安全约束的系统性传递。**

- **链接: [https://arxiv.org/pdf/2602.03987v1](https://arxiv.org/pdf/2602.03987v1)**

> **作者:** Nikolaos Bousias; George Pappas
>
> **摘要:** Control barrier functions (CBFs) provide a powerful tool for enforcing safety constraints in control systems, but their direct application to complex, high-dimensional dynamics is often challenging. In many settings, safety certificates are more naturally designed for simplified or alternative system models that do not exactly match the dynamics of interest. This paper addresses the problem of transferring safety guarantees between dynamical systems with mismatched dynamics. We propose a transferred control barrier function (tCBF) framework that enables safety constraints defined on one system to be systematically enforced on another system using a simulation function and an explicit margin term. The resulting transferred barrier accounts for model mismatch and induces a safety condition that can be enforced on the target system via a quadratic-program-based safety filter. The proposed approach is general and does not require the two systems to share the same state dimension or dynamics. We demonstrate the effectiveness of the framework on a quadrotor navigation task with the transferred barrier ensuring collision avoidance for the target system, while remaining minimally invasive to a nominal controller. These results highlight the potential of transferred control barrier functions as a general mechanism for enforcing safety across heterogeneous dynamical systems.
>
---
#### [new 041] AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出AGILE框架，解决单目视频中手-物体交互重建问题，通过生成式方法和鲁棒跟踪策略，提升几何精度与物理合理性。**

- **链接: [https://arxiv.org/pdf/2602.04672v1](https://arxiv.org/pdf/2602.04672v1)**

> **作者:** Jin-Chuan Shi; Binhong Ye; Tao Liu; Junzhe He; Yangjinhui Xu; Xiaoyang Liu; Zeju Li; Hao Chen; Chunhua Shen
>
> **备注:** 11 pages
>
> **摘要:** Reconstructing dynamic hand-object interactions from monocular videos is critical for dexterous manipulation data collection and creating realistic digital twins for robotics and VR. However, current methods face two prohibitive barriers: (1) reliance on neural rendering often yields fragmented, non-simulation-ready geometries under heavy occlusion, and (2) dependence on brittle Structure-from-Motion (SfM) initialization leads to frequent failures on in-the-wild footage. To overcome these limitations, we introduce AGILE, a robust framework that shifts the paradigm from reconstruction to agentic generation for interaction learning. First, we employ an agentic pipeline where a Vision-Language Model (VLM) guides a generative model to synthesize a complete, watertight object mesh with high-fidelity texture, independent of video occlusions. Second, bypassing fragile SfM entirely, we propose a robust anchor-and-track strategy. We initialize the object pose at a single interaction onset frame using a foundation model and propagate it temporally by leveraging the strong visual similarity between our generated asset and video observations. Finally, a contact-aware optimization integrates semantic, geometric, and interaction stability constraints to enforce physical plausibility. Extensive experiments on HO3D, DexYCB, and in-the-wild videos reveal that AGILE outperforms baselines in global geometric accuracy while demonstrating exceptional robustness on challenging sequences where prior art frequently collapses. By prioritizing physical validity, our method produces simulation-ready assets validated via real-to-sim retargeting for robotic applications.
>
---
#### [new 042] The Supportiveness-Safety Tradeoff in LLM Well-Being Agents
- **分类: cs.HC; cs.RO**

- **简介: 论文研究LLM在心理健康支持中的支持性与安全性权衡问题，评估不同提示对安全性和关怀质量的影响，旨在优化提示设计与模型选择。**

- **链接: [https://arxiv.org/pdf/2602.04487v1](https://arxiv.org/pdf/2602.04487v1)**

> **作者:** Himanshi Lalwani; Hanan Salam
>
> **摘要:** Large language models (LLMs) are being integrated into socially assistive robots (SARs) and other conversational agents providing mental health and well-being support. These agents are often designed to sound empathic and supportive in order to maximize user's engagement, yet it remains unclear how increasing the level of supportive framing in system prompts influences safety relevant behavior. We evaluated 6 LLMs across 3 system prompts with varying levels of supportiveness on 80 synthetic queries spanning 4 well-being domains (1440 responses). An LLM judge framework, validated against human ratings, assessed safety and care quality. Moderately supportive prompts improved empathy and constructive support while maintaining safety. In contrast, strongly validating prompts significantly degraded safety and, in some cases, care across all domains, with substantial variation across models. We discuss implications for prompt design, model selection, and domain specific safeguards in SARs deployment.
>
---
#### [new 043] S-MUSt3R: Sliding Multi-view 3D Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于单目3D重建任务，解决大尺度RGB序列重建的内存限制问题。提出S-MUSt3R方法，通过序列分割与优化实现高效重建。**

- **链接: [https://arxiv.org/pdf/2602.04517v1](https://arxiv.org/pdf/2602.04517v1)**

> **作者:** Leonid Antsfeld; Boris Chidlovskii; Yohann Cabon; Vincent Leroy; Jerome Revaud
>
> **备注:** 8 pages, 5 figures, 5 tables
>
> **摘要:** The recent paradigm shift in 3D vision led to the rise of foundation models with remarkable capabilities in 3D perception from uncalibrated images. However, extending these models to large-scale RGB stream 3D reconstruction remains challenging due to memory limitations. This work proposes S-MUSt3R, a simple and efficient pipeline that extends the limits of foundation models for monocular 3D reconstruction. Our approach addresses the scalability bottleneck of foundation models through a simple strategy of sequence segmentation followed by segment alignment and lightweight loop closure optimization. Without model retraining, we benefit from remarkable 3D reconstruction capacities of MUSt3R model and achieve trajectory and reconstruction performance comparable to traditional methods with more complex architecture. We evaluate S-MUSt3R on TUM, 7-Scenes and proprietary robot navigation datasets and show that S-MUSt3R runs successfully on long RGB sequences and produces accurate and consistent 3D reconstruction. Our results highlight the potential of leveraging the MUSt3R model for scalable monocular 3D scene in real-world settings, with an important advantage of making predictions directly in the metric space.
>
---
#### [new 044] PuppetAI: A Customizable Platform for Designing Tactile-Rich Affective Robot Interaction
- **分类: cs.HC; cs.RO**

- **简介: 该论文介绍PuppetAI平台，用于设计具有触觉反馈的情感机器人交互。属于机器人交互任务，解决情感表达与手势定制问题，通过模块化架构和情感循环实现灵活交互。**

- **链接: [https://arxiv.org/pdf/2602.04787v1](https://arxiv.org/pdf/2602.04787v1)**

> **作者:** Jiaye Li; Tongshun Chen; Siyi Ma; Elizabeth Churchill; Ke Wu
>
> **摘要:** We introduce PuppetAI, a modular soft robot interaction platform. This platform offers a scalable cable-driven actuation system and a customizable, puppet-inspired robot gesture framework, supporting a multitude of interaction gesture robot design formats. The platform comprises a four-layer decoupled software architecture that includes perceptual processing, affective modeling, motion scheduling, and low-level actuation. We also implemented an affective expression loop that connects human input to the robot platform by producing real-time emotional gestural responses to human vocal input. For our own designs, we have worked with nuanced gestures enacted by "soft robots" with enhanced dexterity and "pleasant-to-touch" plush exteriors. By reducing operational complexity and production costs while enhancing customizability, our work creates an adaptable and accessible foundation for future tactile-based expressive robot research. Our goal is to provide a platform that allows researchers to independently construct or refine highly specific gestures and movements performed by social robots.
>
---
#### [new 045] Multi-threaded Recast-Based A* Pathfinding for Scalable Navigation in Dynamic Game Environments
- **分类: cs.GR; cs.RO**

- **简介: 该论文属于游戏路径规划任务，解决动态环境中A*算法性能与真实感的平衡问题。通过多线程和Recast技术提升路径生成与人群导航效率。**

- **链接: [https://arxiv.org/pdf/2602.04130v1](https://arxiv.org/pdf/2602.04130v1)**

> **作者:** Tiroshan Madushanka; Sakuna Madushanka
>
> **摘要:** While the A* algorithm remains the industry standard for game pathfinding, its integration into dynamic 3D environments faces trade-offs between computational performance and visual realism. This paper proposes a multi-threaded framework that enhances standard A* through Recast-based mesh generation, Bezier-curve trajectory smoothing, and density analysis for crowd coordination. We evaluate our system across ten incremental phases, from 2D mazes to complex multi-level dynamic worlds. Experimental results demonstrate that the framework maintains 350+ FPS with 1000 simultaneous agents and achieves collision-free crowd navigation through density-aware path coordination.
>
---
#### [new 046] Modular Safety Guardrails Are Necessary for Foundation-Model-Enabled Robots in the Real World
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于机器人安全领域，解决基础模型赋能机器人在真实世界中的安全问题。提出模块化安全护栏，提升行动、决策和人机协同安全性。**

- **链接: [https://arxiv.org/pdf/2602.04056v1](https://arxiv.org/pdf/2602.04056v1)**

> **作者:** Joonkyung Kim; Wenxi Chen; Davood Soleymanzadeh; Yi Ding; Xiangbo Gao; Zhengzhong Tu; Ruqi Zhang; Fan Fei; Sushant Veer; Yiwei Lyu; Minghui Zheng; Yan Gu
>
> **摘要:** The integration of foundation models (FMs) into robotics has accelerated real-world deployment, while introducing new safety challenges arising from open-ended semantic reasoning and embodied physical action. These challenges require safety notions beyond physical constraint satisfaction. In this paper, we characterize FM-enabled robot safety along three dimensions: action safety (physical feasibility and constraint compliance), decision safety (semantic and contextual appropriateness), and human-centered safety (conformance to human intent, norms, and expectations). We argue that existing approaches, including static verification, monolithic controllers, and end-to-end learned policies, are insufficient in settings where tasks, environments, and human expectations are open-ended, long-tailed, and subject to adaptation over time. To address this gap, we propose modular safety guardrails, consisting of monitoring (evaluation) and intervention layers, as an architectural foundation for comprehensive safety across the autonomy stack. Beyond modularity, we highlight possible cross-layer co-design opportunities through representation alignment and conservatism allocation to enable faster, less conservative, and more effective safety enforcement. We call on the community to explore richer guardrail modules and principled co-design strategies to advance safe real-world physical AI deployment.
>
---
#### [new 047] Lyapunov Constrained Soft Actor-Critic (LC-SAC) using Koopman Operator Theory for Quadrotor Trajectory Tracking
- **分类: eess.SY; cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决安全关键系统中策略稳定性问题。通过引入Lyapunov约束和Koopman理论，提出LC-SAC算法，提升四旋翼轨迹跟踪的稳定性与收敛性。**

- **链接: [https://arxiv.org/pdf/2602.04132v1](https://arxiv.org/pdf/2602.04132v1)**

> **作者:** Dhruv S. Kushwaha; Zoleikha A. Biron
>
> **备注:** 12 pages, 7 Figures, submitted to IEEE RA-L
>
> **摘要:** Reinforcement Learning (RL) has achieved remarkable success in solving complex sequential decision-making problems. However, its application to safety-critical physical systems remains constrained by the lack of stability guarantees. Standard RL algorithms prioritize reward maximization, often yielding policies that may induce oscillations or unbounded state divergence. There has significant work in incorporating Lyapunov-based stability guarantees in RL algorithms with key challenges being selecting a candidate Lyapunov function, computational complexity by using excessive function approximators and conservative policies by incorporating stability criterion in the learning process. In this work we propose a novel Lyapunov-constrained Soft Actor-Critic (LC-SAC) algorithm using Koopman operator theory. We propose use of extended dynamic mode decomposition (EDMD) to produce a linear approximation of the system and use this approximation to derive a closed form solution for candidate Lyapunov function. This derived Lyapunov function is incorporated in the SAC algorithm to further provide guarantees for a policy that stabilizes the nonlinear system. The results are evaluated trajectory tracking of a 2D Quadrotor environment based on safe-control-gym. The proposed algorithm shows training convergence and decaying violations for Lyapunov stability criterion compared to baseline vanilla SAC algorithm. GitHub Repository: https://github.com/DhruvKushwaha/LC-SAC-Quadrotor-Trajectory-Tracking
>
---
#### [new 048] Beyond the Control Equations: An Artifact Study of Implementation Quality in Robot Control Software
- **分类: cs.SE; cs.RO**

- **简介: 该论文属于机器人控制软件质量研究任务，旨在解决控制器实现与理论保证不一致的问题。通过分析184个开源控制器实现，发现其存在离散化处理不当、测试不足等问题。**

- **链接: [https://arxiv.org/pdf/2602.04799v1](https://arxiv.org/pdf/2602.04799v1)**

> **作者:** Nils Chur; Thorsten Berger; Einar Broch Johnsen; Andrzej Wąsowski
>
> **摘要:** A controller -- a software module managing hardware behavior -- is a key component of a typical robot system. While control theory gives safety guarantees for standard controller designs, the practical implementation of controllers in software introduces complexities that are often overlooked. Controllers are often designed in continuous space, while the software is executed in discrete space, undermining some of the theoretical guarantees. Despite extensive research on control theory and control modeling, little attention has been paid to the implementations of controllers and how their theoretical guarantees are ensured in real-world software systems. We investigate 184 real-world controller implementations in open-source robot software. We examine their application context, the implementation characteristics, and the testing methods employed to ensure correctness. We find that the implementations often handle discretization in an ad hoc manner, leading to potential issues with real-time reliability. Challenges such as timing inconsistencies, lack of proper error handling, and inadequate consideration of real-time constraints further complicate matters. Testing practices are superficial, no systematic verification of theoretical guarantees is used, leaving possible inconsistencies between expected and actual behavior. Our findings highlight the need for improved implementation guidelines and rigorous verification techniques to ensure the reliability and safety of robotic controllers in practice.
>
---
#### [new 049] Robot-Assisted Group Tours for Blind People
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决盲人参与混合视觉群体活动的困难。通过设计机器人系统支持盲人加入团体导览，提升其安全性和信息获取能力。**

- **链接: [https://arxiv.org/pdf/2602.04458v1](https://arxiv.org/pdf/2602.04458v1)**

> **作者:** Yaxin Hu; Masaki Kuribayashi; Allan Wang; Seita Kayukawa; Daisuke Sato; Bilge Mutlu; Hironobu Takagi; Chieko Asakawa
>
> **备注:** In Proceedings of ACM CHI 2026 conference on Human Factors in Computing Systems
>
> **摘要:** Group interactions are essential to social functioning, yet effective engagement relies on the ability to recognize and interpret visual cues, making such engagement a significant challenge for blind people. In this paper, we investigate how a mobile robot can support group interactions for blind people. We used the scenario of a guided tour with mixed-visual groups involving blind and sighted visitors. Based on insights from an interview study with blind people (n=5) and museum experts (n=5), we designed and prototyped a robotic system that supported blind visitors to join group tours. We conducted a field study in a science museum where each blind participant (n=8) joined a group tour with one guide and two sighted participants (n=8). Findings indicated users' sense of safety from the robot's navigational support, concerns in the group participation, and preferences for obtaining environmental information. We present design implications for future robotic systems to support blind people's mixed-visual group participation.
>
---
#### [new 050] eCP: Informative uncertainty quantification via Equivariantized Conformal Prediction with pre-trained models
- **分类: cs.LG; cs.RO; eess.SY**

- **简介: 该论文属于不确定性量化任务，旨在解决长时序预测中置信区域过大的问题。通过引入对称性平均，优化非符合度评分，提升预测置信集的精度。**

- **链接: [https://arxiv.org/pdf/2602.03986v1](https://arxiv.org/pdf/2602.03986v1)**

> **作者:** Nikolaos Bousias; Lars Lindemann; George Pappas
>
> **摘要:** We study the effect of group symmetrization of pre-trained models on conformal prediction (CP), a post-hoc, distribution-free, finite-sample method of uncertainty quantification that offers formal coverage guarantees under the assumption of data exchangeability. Unfortunately, CP uncertainty regions can grow significantly in long horizon missions, rendering the statistical guarantees uninformative. To that end, we propose infusing CP with geometric information via group-averaging of the pretrained predictor to distribute the non-conformity mass across the orbits. Each sample now is treated as a representative of an orbit, thus uncertainty can be mitigated by other samples entangled to it via the orbit inducing elements of the symmetry group. Our approach provably yields contracted non-conformity scores in increasing convex order, implying improved exponential-tail bounds and sharper conformal prediction sets in expectation, especially at high confidence levels. We then propose an experimental design to test these theoretical claims in pedestrian trajectory prediction.
>
---
#### [new 051] SPOT-Occ: Sparse Prototype-guided Transformer for Camera-based 3D Occupancy Prediction
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于3D占用预测任务，解决从相机数据中高效准确预测3D空间的问题。提出SPOT-Occ模型，通过原型引导的Transformer解码器提升速度与精度。**

- **链接: [https://arxiv.org/pdf/2602.04240v1](https://arxiv.org/pdf/2602.04240v1)**

> **作者:** Suzeyu Chen; Leheng Li; Ying-Cong Chen
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Achieving highly accurate and real-time 3D occupancy prediction from cameras is a critical requirement for the safe and practical deployment of autonomous vehicles. While this shift to sparse 3D representations solves the encoding bottleneck, it creates a new challenge for the decoder: how to efficiently aggregate information from a sparse, non-uniformly distributed set of voxel features without resorting to computationally prohibitive dense attention. In this paper, we propose a novel Prototype-based Sparse Transformer Decoder that replaces this costly interaction with an efficient, two-stage process of guided feature selection and focused aggregation. Our core idea is to make the decoder's attention prototype-guided. We achieve this through a sparse prototype selection mechanism, where each query adaptively identifies a compact set of the most salient voxel features, termed prototypes, for focused feature aggregation. To ensure this dynamic selection is stable and effective, we introduce a complementary denoising paradigm. This approach leverages ground-truth masks to provide explicit guidance, guaranteeing a consistent query-prototype association across decoder layers. Our model, dubbed SPOT-Occ, outperforms previous methods with a significant margin in speed while also improving accuracy. Source code is released at https://github.com/chensuzeyu/SpotOcc.
>
---
#### [new 052] Natural Language Instructions for Scene-Responsive Human-in-the-Loop Motion Planning in Autonomous Driving using Vision-Language-Action Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决真实场景下语言指令指导路径规划的问题。通过引入doScenes数据集和OpenEMMA框架，实现指令条件下的运动规划，提升驾驶轨迹的鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2602.04184v1](https://arxiv.org/pdf/2602.04184v1)**

> **作者:** Angel Martinez-Sanchez; Parthib Roy; Ross Greer
>
> **摘要:** Instruction-grounded driving, where passenger language guides trajectory planning, requires vehicles to understand intent before motion. However, most prior instruction-following planners rely on simulation or fixed command vocabularies, limiting real-world generalization. doScenes, the first real-world dataset linking free-form instructions (with referentiality) to nuScenes ground-truth motion, enables instruction-conditioned planning. In this work, we adapt OpenEMMA, an open-source MLLM-based end-to-end driving framework that ingests front-camera views and ego-state and outputs 10-step speed-curvature trajectories, to this setting, presenting a reproducible instruction-conditioned baseline on doScenes and investigate the effects of human instruction prompts on predicted driving behavior. We integrate doScenes directives as passenger-style prompts within OpenEMMA's vision-language interface, enabling linguistic conditioning before trajectory generation. Evaluated on 849 annotated scenes using ADE, we observe that instruction conditioning substantially improves robustness by preventing extreme baseline failures, yielding a 98.7% reduction in mean ADE. When such outliers are removed, instructions still influence trajectory alignment, with well-phrased prompts improving ADE by up to 5.1%. We use this analysis to discuss what makes a "good" instruction for the OpenEMMA framework. We release the evaluation prompts and scripts to establish a reproducible baseline for instruction-aware planning. GitHub: https://github.com/Mi3-Lab/doScenes-VLM-Planning
>
---
## 更新

#### [replaced 001] Learning-based Force Sensing and Impedance Matching for Safe Haptic Feedback in Robot-assisted Laparoscopic Surgery
- **分类: cs.RO**

- **简介: 该论文属于机器人辅助手术中的触觉反馈任务，旨在解决力渲染不准确和系统安全性问题。通过引入非线性阻抗匹配方法（NIMA），提升触觉反馈的精度与安全性。**

- **链接: [https://arxiv.org/pdf/2601.14445v2](https://arxiv.org/pdf/2601.14445v2)**

> **作者:** Aiden; Mazidi; Majid Roshanfar; Amir Sayadi; Javad Dargahi; Jake Barralet; Liane S. Feldman; Amir Hooshiar
>
> **摘要:** Integrating accurate haptic feedback into robot-assisted minimally invasive surgery (RAMIS) remains challenging due to difficulties in precise force rendering and ensuring system safety during teleoperation. We present a Nonlinear Impedance Matching Approach (NIMA) that extends our previously validated Impedance Matching Approach (IMA) by incorporating nonlinear dynamics to accurately model and render complex tool-tissue interactions in real-time. NIMA achieves a mean absolute error of 0.01 (std 0.02 N), representing a 95% reduction compared to IMA. Additionally, NIMA eliminates haptic "kickback" by ensuring zero force is applied to the user's hand when they release the handle, enhancing both patient safety and operator comfort. By accounting for nonlinearities in tool-tissue interactions, NIMA significantly improves force fidelity, responsiveness, and precision across various surgical conditions, advancing haptic feedback systems for reliable robot-assisted surgical procedures.
>
---
#### [replaced 002] Analytical Inverse Kinematic Solution for "Moz1" NonSRS 7-DOF Robot arm with novel arm angle
- **分类: cs.RO; math.OC**

- **简介: 该论文属于机器人逆运动学问题，旨在解决7-DOF Moz1机械臂的逆解问题，提出一种新的臂角表示方法，实现无奇异性和完整解空间。**

- **链接: [https://arxiv.org/pdf/2511.22996v2](https://arxiv.org/pdf/2511.22996v2)**

> **作者:** Ke Chen
>
> **摘要:** This paper presents an analytical solution to the inverse kinematic problem(IKP) for the seven degree-of-freedom (7-DOF) Moz1 Robot Arm with offsets on wrist. We provide closed-form solutions with the novel arm angle . it allow fully self-motion and solve the problem of algorithmic singularities within the workspace. It also provides information on how the redundancy is resolved in a new arm angle representation where traditional SEW angle faied to be defined and how singularities are handled. The solution is simple, fast and exact, providing full solution space (i.e. all 16 solutions) per pose.
>
---
#### [replaced 003] MIGHTY: Hermite Spline-based Efficient Trajectory Planning
- **分类: cs.RO**

- **简介: 该论文提出MIGHTY，一种基于赫米特样条的轨迹规划方法，解决高效且满足约束的路径规划问题，通过联合时空优化提升计算效率和飞行性能。**

- **链接: [https://arxiv.org/pdf/2511.10822v3](https://arxiv.org/pdf/2511.10822v3)**

> **作者:** Kota Kondo; Yuwei Wu; Vijay Kumar; Jonathan P. How
>
> **备注:** 10 pages, 12 figures
>
> **摘要:** Hard-constraint trajectory planners often rely on commercial solvers and demand substantial computational resources. Existing soft-constraint methods achieve faster computation, but either (1) decouple spatial and temporal optimization or (2) restrict the search space. To overcome these limitations, we introduce MIGHTY, a Hermite spline-based planner that performs spatiotemporal optimization while fully leveraging the continuous search space of a spline. In simulation, MIGHTY achieves a 9.3% reduction in computation time and a 13.1% reduction in travel time over state-of-the-art baselines, with a 100% success rate. In hardware, MIGHTY completes multiple high-speed flights up to 6.7 m/s in a cluttered static environment and long-duration flights with dynamically added obstacles.
>
---
#### [replaced 004] Realistic adversarial scenario generation via human-like pedestrian model for autonomous vehicle control parameter optimisation
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于自动驾驶安全测试任务，旨在解决传统模拟场景过于复杂而缺乏真实性的难题。通过引入类人行人模型生成更真实的对抗场景，优化AV控制策略。**

- **链接: [https://arxiv.org/pdf/2601.02082v2](https://arxiv.org/pdf/2601.02082v2)**

> **作者:** Yueyang Wang; Mehmet Dogar; Russell Darling; Gustav Markkula
>
> **摘要:** Autonomous vehicles (AVs) are rapidly advancing and are expected to play a central role in future mobility. Ensuring their safe deployment requires reliable interaction with other road users, not least pedestrians. Direct testing on public roads is costly and unsafe for rare but critical interactions, making simulation a practical alternative. Within simulation-based testing, adversarial scenarios are widely used to probe safety limits, but many prioritise difficulty over realism, producing exaggerated behaviours which may result in AV controllers that are overly conservative. We propose an alternative method, instead using a cognitively inspired pedestrian model featuring both inter-individual and intra-individual variability to generate behaviourally plausible adversarial scenarios. We provide a proof of concept demonstration of this method's potential for AV control optimisation, in closed-loop testing and tuning of an AV controller. Our results show that replacing the rule-based CARLA pedestrian with the human-like model yields more realistic gap acceptance patterns and smoother vehicle decelerations. Unsafe interactions occur only for certain pedestrian individuals and conditions, underscoring the importance of human variability in AV testing. Adversarial scenarios generated by this model can be used to optimise AV control towards safer and more efficient behaviour. Overall, this work illustrates how incorporating human-like road user models into simulation-based adversarial testing can enhance the credibility of AV evaluation and provide a practical basis to behaviourally informed controller optimisation.
>
---
#### [replaced 005] ProAct: A Benchmark and Multimodal Framework for Structure-Aware Proactive Response
- **分类: cs.RO**

- **简介: 该论文提出ProAct-75基准和ProAct-Helper框架，解决 proactive agent 开发中资源不足的问题，通过结构化任务图提升智能体的主动响应能力。**

- **链接: [https://arxiv.org/pdf/2602.03430v2](https://arxiv.org/pdf/2602.03430v2)**

> **作者:** Xiaomeng Zhu; Fengming Zhu; Weijie Zhou; Ye Tian; Zhenlin Hu; Yufei Huang; Yuchun Guo; Xinyu Wu; Zhengyou Zhang; Fangzhen Lin; Xuantang Xiong
>
> **摘要:** While passive agents merely follow instructions, proactive agents align with higher-level objectives, such as assistance and safety by continuously monitoring the environment to determine when and how to act. However, developing proactive agents is hindered by the lack of specialized resources. To address this, we introduce ProAct-75, a benchmark designed to train and evaluate proactive agents across diverse domains, including assistance, maintenance, and safety monitoring. Spanning 75 tasks, our dataset features 91,581 step-level annotations enriched with explicit task graphs. These graphs encode step dependencies and parallel execution possibilities, providing the structural grounding necessary for complex decision-making. Building on this benchmark, we propose ProAct-Helper, a reference baseline powered by a Multimodal Large Language Model (MLLM) that grounds decision-making in state detection, and leveraging task graphs to enable entropy-driven heuristic search for action selection, allowing agents to execute parallel threads independently rather than mirroring the human's next step. Extensive experiments demonstrate that ProAct-Helper outperforms strong closed-source models, improving trigger detection mF1 by 6.21%, saving 0.25 more steps in online one-step decision, and increasing the rate of parallel actions by 15.58%.
>
---
#### [replaced 006] Geometry-aware 4D Video Generation for Robot Manipulation
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于4D视频生成任务，旨在解决多视角下视频时空一致性问题。通过几何监督学习3D场景表示，生成从新视角出发的连贯视频序列。**

- **链接: [https://arxiv.org/pdf/2507.01099v3](https://arxiv.org/pdf/2507.01099v3)**

> **作者:** Zeyi Liu; Shuang Li; Eric Cousineau; Siyuan Feng; Benjamin Burchfiel; Shuran Song
>
> **备注:** ICLR 2026; Project website: https://robot4dgen.github.io
>
> **摘要:** Understanding and predicting dynamics of the physical world can enhance a robot's ability to plan and interact effectively in complex environments. While recent video generation models have shown strong potential in modeling dynamic scenes, generating videos that are both temporally coherent and geometrically consistent across camera views remains a significant challenge. To address this, we propose a 4D video generation model that enforces multi-view 3D consistency of generated videos by supervising the model with cross-view pointmap alignment during training. Through this geometric supervision, the model learns a shared 3D scene representation, enabling it to generate spatio-temporally aligned future video sequences from novel viewpoints given a single RGB-D image per view, and without relying on camera poses as input. Compared to existing baselines, our method produces more visually stable and spatially aligned predictions across multiple simulated and real-world robotic datasets. We further show that the predicted 4D videos can be used to recover robot end-effector trajectories using an off-the-shelf 6DoF pose tracker, yielding robot manipulation policies that generalize well to novel camera viewpoints.
>
---
#### [replaced 007] Mixed-Density Diffuser: Efficient Planning with Non-Uniform Temporal Resolution
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决规划中时间密度不均的问题。提出MDD模型，通过调整不同阶段的密度提升性能，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2510.23026v4](https://arxiv.org/pdf/2510.23026v4)**

> **作者:** Crimson Stambaugh; Rajesh P. N. Rao
>
> **备注:** European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN) (under review)
>
> **摘要:** Recent studies demonstrate that diffusion planners benefit from sparse-step planning over single-step planning. Training models to skip steps in their trajectories helps capture long-term dependencies without additional memory or computational cost. However, predicting excessively sparse plans degrades performance. We hypothesize this temporal density threshold is non-uniform across a planning horizon and that certain parts of a predicted trajectory should be more densely generated. We propose Mixed-Density Diffuser (MDD), a diffusion planner where the densities throughout the horizon are tunable hyperparameters. We show that MDD surpasses the SOTA Diffusion Veteran (DV) framework across the Maze2D, Franka Kitchen, and Antmaze Datasets for Deep Data-Driven Reinforcement Learning (D4RL) task domains, achieving a new SOTA on the D4RL benchmark.
>
---
#### [replaced 008] Model Reconciliation through Explainability and Collaborative Recovery in Assistive Robotics
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机协作任务，解决机器人与人类模型不一致问题。通过框架实现模型协调，利用大语言模型解释差异，并允许人类修正机器人模型。**

- **链接: [https://arxiv.org/pdf/2601.06552v3](https://arxiv.org/pdf/2601.06552v3)**

> **作者:** Britt Besch; Tai Mai; Jeremias Thun; Markus Huff; Jörn Vogel; Freek Stulp; Samuel Bustamante
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Whenever humans and robots work together, it is essential that unexpected robot behavior can be explained to the user. Especially in applications such as shared control the user and the robot must share the same model of the objects in the world, and the actions that can be performed on these objects. In this paper, we achieve this with a so-called model reconciliation framework. We leverage a Large Language Model to predict and explain the difference between the robot's and the human's mental models, without the need of a formal mental model of the user. Furthermore, our framework aims to solve the model divergence after the explanation by allowing the human to correct the robot. We provide an implementation in an assistive robotics domain, where we conduct a set of experiments with a real wheelchair-based mobile manipulator and its digital twin.
>
---
#### [replaced 009] Game-Based and Gamified Robotics Education: A Comparative Systematic Review and Design Guidelines
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于教育技术领域，旨在比较游戏化与游戏化教学在机器人教育中的效果。通过系统综述分析95项研究，提出设计指南与研究方向。**

- **链接: [https://arxiv.org/pdf/2601.22199v2](https://arxiv.org/pdf/2601.22199v2)**

> **作者:** Syed T. Mubarrat; Byung-Cheol Min; Tianyu Shao; E. Cho Smith; Bedrich Benes; Alejandra J. Magana; Christos Mousas; Dominic Kao
>
> **备注:** Accepted for publication at Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems. 26 pages, 14 figures, 7 tables;
>
> **摘要:** Robotics education fosters computational thinking, creativity, and problem-solving, but remains challenging due to technical complexity. Game-based learning (GBL) and gamification offer engagement benefits, yet their comparative impact remains unclear. We present the first PRISMA-aligned systematic review and comparative synthesis of GBL and gamification in robotics education, analyzing 95 studies from 12,485 records across four databases (2014-2025). We coded each study's approach, learning context, skill level, modality, pedagogy, and outcomes (k = .918). Three patterns emerged: (1) approach-context-pedagogy coupling (GBL more prevalent in informal settings, while gamification dominated formal classrooms [p < .001] and favored project-based learning [p = .009]); (2) emphasis on introductory programming and modular kits, with limited adoption of advanced software (~17%), advanced hardware (~5%), or immersive technologies (~22%); and (3) short study horizons, relying on self-report. We propose eight research directions and a design space outlining best practices and pitfalls, offering actionable guidance for robotics education.
>
---
#### [replaced 010] Statistical Guarantees for Offline Domain Randomization
- **分类: cs.LG; cs.RO**

- **简介: 该论文研究离线域随机化（ODR）的理论基础，解决模拟到现实部署中的差距问题。通过最大似然估计提供统计保证，明确ODR在何种条件下有效。**

- **链接: [https://arxiv.org/pdf/2506.10133v2](https://arxiv.org/pdf/2506.10133v2)**

> **作者:** Arnaud Fickinger; Abderrahim Bendahi; Stuart Russell
>
> **备注:** ICLR 2026
>
> **摘要:** Reinforcement-learning (RL) agents often struggle when deployed from simulation to the real-world. A dominant strategy for reducing the sim-to-real gap is domain randomization (DR) which trains the policy across many simulators produced by sampling dynamics parameters, but standard DR ignores offline data already available from the real system. We study offline domain randomization (ODR), which first fits a distribution over simulator parameters to an offline dataset. While a growing body of empirical work reports substantial gains with algorithms such as DROPO, the theoretical foundations of ODR remain largely unexplored. In this work, we cast ODR as a maximum-likelihood estimation over a parametric simulator family and provide statistical guarantees: under mild regularity and identifiability conditions, the estimator is weakly consistent (it converges in probability to the true dynamics as data grows), and it becomes strongly consistent (i.e., it converges almost surely to the true dynamics) when an additional uniform Lipschitz continuity assumption holds. We examine the practicality of these assumptions and outline relaxations that justify ODR's applicability across a broader range of settings. Taken together, our results place ODR on a principled footing and clarify when offline data can soundly guide the choice of a randomization distribution for downstream offline RL.
>
---
#### [replaced 011] Autonomous Navigation at the Nano-Scale: Algorithms, Architectures, and Constraints
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究纳米级无人机自主导航任务，解决SWaP约束下的算法与架构设计问题，探讨了从传统方法到边缘AI的转型及硬件软件协同优化。**

- **链接: [https://arxiv.org/pdf/2601.13252v2](https://arxiv.org/pdf/2601.13252v2)**

> **作者:** Mahmud S. Zango; Jianglin Lan
>
> **备注:** 30 pages, 5 figures, 2 table. Review article
>
> **摘要:** Autonomous navigation for nano-scale unmanned aerial vehicles (nano-UAVs) is governed by extreme Size, Weight, and Power (SWaP) constraints (with the weight < 50 g and sub-100 mW onboard processor), distinguishing it fundamentally from standard robotic paradigms. This review synthesizes the state-of-the-art in sensing, computing, and control architectures designed specifically for these sub- 100mW computational envelopes. We critically analyse the transition from classical geometry-based methods to emerging "Edge AI" paradigms, including quantized deep neural networks deployed on ultra-low-power System-on-Chips (SoCs) and neuromorphic event-based control. Beyond algorithms, we evaluate the hardware-software co-design requisite for autonomy, covering advancements in dense optical flow, optimized Simultaneous Localization and Mapping (SLAM), and learning-based flight control. While significant progress has been observed in visual navigation and relative pose estimation, our analysis reveals persistent gaps in long-term endurance, robust obstacle avoidance in dynamic environments, and the "Sim-to-Real" transfer of reinforcement learning policies. This survey provides a roadmap for bridging these gaps, advocating for hybrid architectures that fuse lightweight classical control with data-driven perception to enable fully autonomous, agile nano-UAVs in GPS-denied environments.
>
---
#### [replaced 012] A Unified Candidate Set with Scene-Adaptive Refinement via Diffusion for End-to-End Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶轨迹规划任务，解决传统候选集在复杂场景下表现不足的问题。提出CdDrive方法，结合固定候选与场景自适应生成，提升规划效果。**

- **链接: [https://arxiv.org/pdf/2602.03112v2](https://arxiv.org/pdf/2602.03112v2)**

> **作者:** Zhengfei Wu; Shuaixi Pan; Shuohan Chen; Shuo Yang; Yanjun Huang
>
> **摘要:** End-to-end autonomous driving is increasingly adopting a multimodal planning paradigm that generates multiple trajectory candidates and selects the final plan, making candidate-set design critical. A fixed trajectory vocabulary provides stable coverage in routine driving but often misses optimal solutions in complex interactions, while scene-adaptive refinement can cause over-correction in simple scenarios by unnecessarily perturbing already strong vocabulary trajectories.We propose CdDrive, which preserves the original vocabulary candidates and augments them with scene-adaptive candidates generated by vocabulary-conditioned diffusion denoising. Both candidate types are jointly scored by a shared selection module, enabling reliable performance across routine and highly interactive scenarios. We further introduce HATNA (Horizon-Aware Trajectory Noise Adapter) to improve the smoothness and geometric continuity of diffusion candidates via temporal smoothing and horizon-aware noise modulation. Experiments on NAVSIM v1 and NAVSIM v2 demonstrate leading performance, and ablations verify the contribution of each component. Code: https://github.com/WWW-TJ/CdDrive.
>
---
#### [replaced 013] TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance
- **分类: cs.RO**

- **简介: 该论文提出TouchGuide，解决机器人精细操作中触觉反馈利用不足的问题。通过视觉与触觉融合，提升操作的物理可行性，实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2601.20239v2](https://arxiv.org/pdf/2601.20239v2)**

> **作者:** Zhemeng Zhang; Jiahua Ma; Xincheng Yang; Xin Wen; Yuzhi Zhang; Boyan Li; Yiran Qin; Jin Liu; Can Zhao; Li Kang; Haoqin Hong; Zhenfei Yin; Philip Torr; Hao Su; Ruimao Zhang; Daolin Ma
>
> **摘要:** Fine-grained and contact-rich manipulation remain challenging for robots, largely due to the underutilization of tactile feedback. To address this, we introduce TouchGuide, a novel cross-policy visuo-tactile fusion paradigm that fuses modalities within a low-dimensional action space. Specifically, TouchGuide operates in two stages to guide a pre-trained diffusion or flow-matching visuomotor policy at inference time. First, the policy produces a coarse, visually-plausible action using only visual inputs during early sampling. Second, a task-specific Contact Physical Model (CPM) provides tactile guidance to steer and refine the action, ensuring it aligns with realistic physical contact conditions. Trained through contrastive learning on limited expert demonstrations, the CPM provides a tactile-informed feasibility score to steer the sampling process toward refined actions that satisfy physical contact constraints. Furthermore, to facilitate TouchGuide training with high-quality and cost-effective data, we introduce TacUMI, a data collection system. TacUMI achieves a favorable trade-off between precision and affordability; by leveraging rigid fingertips, it obtains direct tactile feedback, thereby enabling the collection of reliable tactile data. Extensive experiments on five challenging contact-rich tasks, such as shoe lacing and chip handover, show that TouchGuide consistently and significantly outperforms state-of-the-art visuo-tactile policies.
>
---
#### [replaced 014] Learning-based Observer for Coupled Disturbance
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于控制任务，旨在解决机器人系统中耦合扰动的高精度估计问题。通过结合控制与学习方法，提出一种有效算法，实现扰动的准确识别与补偿。**

- **链接: [https://arxiv.org/pdf/2407.13229v3](https://arxiv.org/pdf/2407.13229v3)**

> **作者:** Jindou Jia; Meng Wang; Zihan Yang; Bin Yang; Yuhang Liu; Kexin Guo; Xiang Yu
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Achieving high-precision control for robotic systems is hindered by the low-fidelity dynamical model and external disturbances. Especially, the intricate coupling between internal uncertainties and external disturbances further exacerbates this challenge. This study introduces an effective and convergent algorithm enabling accurate estimation of the coupled disturbance via combining control and learning philosophies. Concretely, by resorting to Chebyshev series expansion, the coupled disturbance is firstly decomposed into an unknown parameter matrix and two known structures dependent on system state and external disturbance respectively. A regularized least squares algorithm is subsequently formalized to learn the parameter matrix using historical time-series data. Finally, a polynomial disturbance observer is specifically devised to achieve a high-precision estimation of the coupled disturbance by utilizing the learned portion. The proposed algorithm is evaluated through extensive simulations and real flight tests. We believe this work can offer a new pathway to integrate learning approaches into control frameworks for addressing longstanding challenges in robotic applications.
>
---
#### [replaced 015] Improved Bag-of-Words Image Retrieval with Geometric Constraints for Ground Texture Localization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，解决地面纹理定位中的图像检索问题。通过改进BoW方法，结合几何约束提升定位精度与闭环检测效果。**

- **链接: [https://arxiv.org/pdf/2505.11620v2](https://arxiv.org/pdf/2505.11620v2)**

> **作者:** Aaron Wilhelm; Nils Napp
>
> **备注:** Accepted to ICRA 2025
>
> **摘要:** Ground texture localization using a downward-facing camera offers a low-cost, high-precision localization solution that is robust to dynamic environments and requires no environmental modification. We present a significantly improved bag-of-words (BoW) image retrieval system for ground texture localization, achieving substantially higher accuracy for global localization and higher precision and recall for loop closure detection in SLAM. Our approach leverages an approximate $k$-means (AKM) vocabulary with soft assignment, and exploits the consistent orientation and constant scale constraints inherent to ground texture localization. Identifying the different needs of global localization vs. loop closure detection for SLAM, we present both high-accuracy and high-speed versions of our algorithm. We test the effect of each of our proposed improvements through an ablation study and demonstrate our method's effectiveness for both global localization and loop closure detection. With numerous ground texture localization systems already using BoW, our method can readily replace other generic BoW systems in their pipeline and immediately improve their results.
>
---
#### [replaced 016] LiDAR, GNSS and IMU Sensor Fine Alignment through Dynamic Time Warping to Construct 3D City Maps
- **分类: cs.RO**

- **简介: 该论文属于3D城市建图任务，解决GNSS受限环境下LiDAR地图的全局错位问题，通过DTW和滤波方法实现传感器精配准。**

- **链接: [https://arxiv.org/pdf/2507.08420v3](https://arxiv.org/pdf/2507.08420v3)**

> **作者:** Haitian Wang; Hezam Albaqami; Xinyu Wang; Muhammad Ibrahim; Zainy M. Malakan; Abdullah M. Algamdi; Mohammed H. Alghamdi; Ajmal Mian
>
> **备注:** This paper has been submitted to IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS) and is currently under review
>
> **摘要:** LiDAR-based 3D mapping suffers from cumulative drift causing global misalignment, particularly in GNSS-constrained environments. To address this, we propose a unified framework that fuses LiDAR, GNSS, and IMU data for high-resolution city-scale mapping. The method performs velocity-based temporal alignment using Dynamic Time Warping and refines GNSS and IMU signals via extended Kalman filtering. Local maps are built using Normal Distributions Transform-based registration and pose graph optimization with loop closure detection, while global consistency is enforced using GNSS-constrained anchors followed by fine registration of overlapping segments. We also introduce a large-scale multimodal dataset captured in Perth, Western Australia to facilitate future research in this direction. Our dataset comprises 144,000 frames acquired with a 128-channel Ouster LiDAR, synchronized RTK-GNSS trajectories, and MEMS-IMU measurements across 21 urban loops. To assess geometric consistency, we evaluated our method using alignment metrics based on road centerlines and intersections to capture both global and local accuracy. The proposed framework reduces the average global alignment error from 3.32m to 1.24m, achieving a 61.4% improvement, and significantly decreases the intersection centroid offset from 13.22m to 2.01m, corresponding to an 84.8% enhancement. The constructed high-fidelity map and raw dataset are publicly available through https://ieee-dataport.org/documents/perth-cbd-high-resolution-lidar-map-gnss-and-imu-calibration, and its visualization can be viewed at https://www.youtube.com/watch?v=-ZUgs1KyMks. The source code is available at https://github.com/HaitianWang/LiDAR-GNSS-and-IMU-Sensor-Fine-Alignment-through-Dynamic-Time-Warping-to-Construct-3D-City-Maps. This dataset and method together establish a new benchmark for evaluating 3D city mapping in GNSS-constrained environments.
>
---
#### [replaced 017] PhysBrain: Human Egocentric Data as a Bridge from Vision Language Models to Physical Intelligence
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉语言动作任务，旨在解决机器人物理智能不足的问题。通过将人类第一视角视频转化为机器人可用数据，构建E2E-3M数据集，并训练出PhysBrain模型，提升机器人规划与控制能力。**

- **链接: [https://arxiv.org/pdf/2512.16793v2](https://arxiv.org/pdf/2512.16793v2)**

> **作者:** Xiaopeng Lin; Shijie Lian; Bin Yu; Ruoqi Yang; Zhaolong Shen; Changti Wu; Yuzhuo Miao; Yurun Jin; Yukun Shi; Jiyan He; Cong Huang; Bojun Cheng; Kai Chen
>
> **备注:** 21 pages, 8 figures
>
> **摘要:** Robotic generalization relies on physical intelligence: the ability to reason about state changes, contact-rich interactions, and long-horizon planning under egocentric perception and action. Vision Language Models (VLMs) are essential to Vision-Language-Action (VLA) systems, but the reliance on third-person training data creates a viewpoint gap for humanoid robots. Collecting massive robot-centric data is an ideal but impractical solution due to cost and diversity constraints. Conversely, human egocentric videos offer a highly scalable data source with rich interaction context, yet the embodiment mismatch prevents the direct application. To bridge this gap, we propose an Egocentric2Embodiment Translation Pipeline that transforms raw human egocentric videos into multi-level, schema-driven embodiment supervision with enforced evidence grounding and temporal consistency, enabling the construction of the Egocentric2Embodiment dataset (E2E-3M) at scale. An egocentric-aware embodied brain, termed PhysBrain, is obtained by training on the E2E-3M dataset. PhysBrain exhibits substantially improved egocentric understanding, particularly for planning. It provides an egocentric-aware initialization that enables more sample-efficient VLA fine-tuning and higher success rates, demonstrating effective transfer from human egocentric supervision to downstream robot control.
>
---
