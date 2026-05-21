# 机器人 cs.RO

- **最新发布 62 篇**

- **更新 24 篇**

## 最新发布

#### [new 001] DISC: Decoupling Instruction from State-Conditioned Control via Policy Generation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出DISC，解决语言条件操作策略中的观察泄露问题。通过超网络从指令生成特定策略参数，确保任务意识来自语言而非视觉捷径，提升复杂任务表现。**

- **链接: [https://arxiv.org/pdf/2605.20856](https://arxiv.org/pdf/2605.20856)**

> **作者:** Hanxiang Ren; Pei Zhou; Xunzhe Zhou; Yanchao Yang
>
> **摘要:** Language-conditioned manipulation policies typically process instructions and observations through shared network parameters. This task-state entanglement provides a pathway for observation leakage -- networks learn scene-to-action shortcuts that bypass language grounding entirely. DISC eliminates this failure structurally. Rather than conditioning a universal policy on language, DISC uses a hypernetwork to generate the entire parameter set of a task-specific visuomotor policy from the instruction alone. The generated policy never directly accesses language; therefore, its task-awareness must come from the language. Consequently, observation leakage has no pathway to emerge. On the other hand, generating coherent high-dimensional policy weights is itself a challenging problem. We address it with a two-stage hypernetwork whose refinement stage embeds the structure of gradient-based optimization as a feed-forward inductive bias, producing globally consistent parameters without actual gradient computation. Trained entirely from scratch on standard data budgets, DISC outperforms all entangled baselines on LIBERO-90 and Meta-World, with advantages that widen on complex, long-horizon tasks -- and surpasses the large-scale pretrained $\pi_0$ despite using no external pretraining data. On a real-world benchmark where all tasks share identical visual context, DISC substantially outperforms entangled alternatives, directly confirming that language-generated policy parameters, not visual shortcuts, drive behavior. The hypernetwork further learns a semantically structured parameter manifold that enables few-shot adaptation from minimal demonstrations and robust generalization across paraphrased instructions. Our code is available at: {this https URL}.
>
---
#### [new 002] Benchmarking Empirical and Learning-Based Approaches for Feedforward Steering Control in Autonomous Racing
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究自主赛车中的前馈转向控制任务，旨在减少反馈控制器的修正需求。通过对比学习与经验方法，提出一种新模型，验证其在闭环测试中的性能优势。**

- **链接: [https://arxiv.org/pdf/2605.21111](https://arxiv.org/pdf/2605.21111)**

> **作者:** Georg Jank; Mattia Piccinini; Sebastian Wenk; Phillip Pitschi; Johannes Betz; Boris Lohmann
>
> **备注:** 8 pages, 12 figures, Accepted to be published as part of the 2026 IEEE International Conference on Intelligent Transportation Systems (ITSC 2026), Naples, Italy, September 15-18, 2026
>
> **摘要:** Feedforward steering control is a key component of hierarchical control architectures for autonomous racing. The goal is to reduce steering corrections from the feedback controllers by predicting the vehicle's inverse lateral dynamics. This paper presents a systematic benchmark of two learning-based and two empirical (analytical) feedforward steering controllers. We introduce a new \acf{ehd} formulation based on a polynomial surface fit that captures velocity-dependent nonlinear steering behavior with minimal parametrization. We test the feedforward controllers in a high-fidelity simulation framework based on the real-world Abu Dhabi Autonomous Racing League competition, using a high-fidelity double-track vehicle dynamics simulator. Open-loop evaluation shows that the learning-based controllers achieve the lowest prediction errors; however, closed-loop testing reveals that this improved accuracy does not translate into superior path tracking performance or lap times, even after iterative fine-tuning. In contrast, the proposed EHD approach achieves the best overall closed-loop robustness and lap time, highlighting the necessity of evaluating feedforward strategies within the complete trajectory planning and control software stack. Our code is available at this https URL.
>
---
#### [new 003] Learning Robust Dexterous In-Hand Manipulation from Joint Sensors with Proprioceptive Transformer
- **分类: cs.RO**

- **简介: 该论文研究机械手的物体操控任务，解决仅依赖关节传感实现精准控制的问题。提出Proprioceptive Transformer模型，利用关节信息进行物体状态估计和控制，提升操控速度与精度。**

- **链接: [https://arxiv.org/pdf/2605.21330](https://arxiv.org/pdf/2605.21330)**

> **作者:** Senlan Yao; Chenyu Yang; Jaehoon Kim; Aristotelis Sympetheros; Robert K. Katzschmann
>
> **备注:** 8 pages, 6 figures, 3 tables
>
> **摘要:** In-hand object manipulation is a fundamental yet challenging capability for dexterous robots. Despite significant progress in dexterous manipulation, existing approaches rely heavily on vision or tactile sensing to track object states, while joint sensing -- the most readily available modality on any robotic hand -- remains largely overlooked, particularly for tendon-driven hands. In this paper, we study how far joint sensing alone can go by asking: (i) whether motor encoders or direct joint sensing provides better proprioceptive feedback, (ii) how to extract environment information from joint measurements, and (iii) whether joint-only control can achieve competitive real-world performance without external perception. We present the Proprioceptive Transformer (PT), an exteroceptive-free approach for continuous cube rotation on a tendon-driven dexterous hand that uses only joint sensing feedback. A teacher policy is first trained via reinforcement learning with privileged object information, then distilled into PT, which operates solely on joint position and velocity histories. The Transformer architecture effectively extracts implicit object state information from temporal patterns in joint sensor readings. Experiments on the real ORCA hand show that our approach achieves 3.1x higher rotation speed than baselines. We also demonstrate that our PT achieves a 23.4% lower RMSE for cube position estimation than the MLP baseline, indicating superior extraction of exteroceptive information from proprioceptive sources.
>
---
#### [new 004] Scalable Multi-robot Motion Planning via Hierarchical Subproblem Expansion and Workspace Decomposition Refinement
- **分类: cs.RO**

- **简介: 该论文属于多机器人运动规划任务，旨在解决机器人间协调与计算效率问题。通过分层分解工作空间，提升规划效率。**

- **链接: [https://arxiv.org/pdf/2605.20395](https://arxiv.org/pdf/2605.20395)**

> **作者:** Isaac Ngui; Courtney McBeth; James D. Motes; Marco Morales; Nancy M. Amato
>
> **备注:** Accepted to WAFR 2026
>
> **摘要:** A fundamental challenge in multi-robot motion planning is achieving sufficient coordination to avoid inter-robot conflicts without incurring the large computational expense of searching the joint configuration space of the robot group. In this work, we present a method for multiple mobile robot motion planning that achieves an improvement in planning time up to an order of magnitude by leveraging the insight that we can use discrete search over a workspace decomposition to provide coordination between robots during planning. While prior work uses workspace topology to inform when coordination between robots is needed and then composes robots into their joint configuration space, we take a step further by iteratively refining our workspace representation to allow our planner to search smaller, decoupled configuration spaces.
>
---
#### [new 005] SmoCap: Unified Scale-Pose Canonicalization with Proxy-Mapped Trust-Region QP
- **分类: cs.RO**

- **简介: 该论文提出SmoCap框架，解决运动数据归一化中的形态与姿态耦合问题，通过联合优化提升准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2605.20850](https://arxiv.org/pdf/2605.20850)**

> **作者:** Shihao Li; Naohiko Sugita
>
> **备注:** 11 pages, 6 figures, 4 tables
>
> **摘要:** Objective: Stage-wise workflows that separate model scaling and inverse kinematics can induce morphology-posture compensation, resulting in anatomically inconsistent yet numerically acceptable solutions, especially in weakly observed directions. We present SmoCap, a leakage-resistant canonicalization framework that estimates morphology and posture jointly in each local trust-region quadratic program (QP) within a sparse control subspace. Methods: SmoCap solves a constrained trust-region QP with analytical proxy-mapped pose and scale Jacobians. The low dimensional proxy map stabilizes weakly observed directions and drives coordinated structures. An optional pre-solve provides warm starts in difficult configurations. The framework is evaluated using cohort fluoroscopy knee motion, anthropometric ground truth, and extreme yoga sequences. Results: SmoCap achieved 2.9 degree knee flexion RMSE against fluoroscopy, and a pooled anthropometric endpoint error around 3%. In the leakage audit against segment wise scaling, SmoCap also reduced marker RMSE, FE error, and anthropometric endpoint error. Proxy coupling preserved expressive and coordinated spine motion with marginal fitting error increase (+0.14 mm, +0.6%) against baseline models in yoga ablation. Median marker RMSE was around 20 mm, and median runtime was 0.204-0.332 ms/frame, achieved with consistently 2-3 iterations. Conclusion: SmoCap provides an externally validated unified coupling-aware scale-pose framework, making externally consistent motion canonicalization practical at dataset scale.
>
---
#### [new 006] Modeling and Control of a Pneumatic Morphing Soft Quadrotor based on the SOFA Framework for Dynamic Soft Robotic Simulation
- **分类: cs.RO**

- **简介: 该论文属于软体机器人动态建模与控制任务，旨在解决气动软臂的形态变化与控制问题。工作包括基于SOFA框架的有限元建模及比例积分控制器设计。**

- **链接: [https://arxiv.org/pdf/2605.21031](https://arxiv.org/pdf/2605.21031)**

> **作者:** F. Labra Caso; V. Sumathy; P. Ferrentino; V. Vanderborght; J. Haluska; G. Nikolakopoulos
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** This article presents a novel SOFA based finite element method for the soft body modeling and the corresponding dynamic simulation and control of a pneumatic morphing soft quadrotor. The proposed modeling preserves the physical interpretability and control structure of traditional quadrotor dynamics, while capturing the complex, time-varying behavior of pneumatically actuated soft arms. In SOFA, the soft pneumatically actuated arms are discretized as a tetrahedral mesh following an elastic material law that produces internal forces adequate to the real dynamic behavior of the body. Pneumatic actuation governed by both periodic and error-based control signals is applied within the internal cavities to analyze the morphing capability. Finally, a proportional-integral controller is proposed to study the controlled dynamic behavior and morphing capabilities of the pneumatic arm, wherein the pneumatic actuation to the soft arm is controlled to achieve the desired target position. The simulation results show the effectiveness of the proposed novel modeling framework and the related controller design.
>
---
#### [new 007] Anomaly-Informed Confidence Calibration for Vision-Based Safety Prediction
- **分类: cs.RO**

- **简介: 该论文属于安全预测任务，解决视觉控制器在分布偏移下的过度自信问题。通过融合感知与动态异常信号，提出在线校准方法，提升预测可靠性。**

- **链接: [https://arxiv.org/pdf/2605.21109](https://arxiv.org/pdf/2605.21109)**

> **作者:** Zhenjiang Mao; Jiawen Wu; Gabriel Wagner; Zhongzheng Zhang; Ivan Ruchkin
>
> **摘要:** Reliable confidence estimates are important for safely deploying vision-based controllers in autonomous racing, where safety predictions must be derived from camera images, yet modern predictors become dangerously overconfident under test-time distribution shifts. We identify a critical perception-dynamics gap in existing anomaly signals: widely used scores, such as autoencoder reconstruction error, capture visual corruptions but miss dynamics anomalies (e.g., actuation bias, latency), where images remain plausible while the trajectory degrades. To address this, we propose an Anomaly-Informed Online Calibration approach that, without retraining any model component, fuses two complementary anomaly scores extracted from a world model: a perceptual score from reconstruction error and a dynamics score from epistemic uncertainty and control-stream statistics. Based on these fused scores, a lightweight temperature-scaling calibrator leverages test-time augmentation to selectively reduce overconfidence under shift while preserving nominal-condition performance. Experiments on a physical DonkeyCar under four real-world anomaly protocols unseen during training (darkness, blur, actuation bias, processing latency) reduce average expected calibration error from 0.184 to 0.116, a 37% improvement over the best baseline, without modifying the base safety predictor.
>
---
#### [new 008] Component Influence-Driven Fastener Reduction for Robotic Disassemblability-Aware Design Simplification
- **分类: cs.RO**

- **简介: 该论文属于产品设计任务，旨在解决机器人拆卸效率低的问题。通过分析组件影响，减少紧固件，优化拆卸路径，提升自动化拆卸效率。**

- **链接: [https://arxiv.org/pdf/2605.21026](https://arxiv.org/pdf/2605.21026)**

> **作者:** Takuya Kiyokawa; Tomoki Ishikura; Shingo Hamada; Genichiro Matsuda; Kensuke Harada
>
> **备注:** 7 pages, 8 figures
>
> **摘要:** To accelerate automated remanufacturing, robotic disassembly must be considered during the product design phase. However, designers currently lack quantitative feedback to identify which structural elements hinder robotic operations. To address this, this study proposes an analytical framework that provides actionable redesign guidance focused on fastener reduction, as fasteners are numerous and ubiquitous components found in almost all manufactured products. Using a Computer-Aided Design (CAD) model and its automatically generated Contact-Connection-Constraint (CCC) graph, the framework translates robotic disassembly sequence planning outcomes into component influence scores. These scores reflect how often a component causes structural constraint violations or evaluation objective deteriorations in the robotic disassembly sequence. To visually highlight structural hindrances, the framework projects these scores onto the CAD geometry as 3D heatmaps. The system then analytically simulates the removal of highly influential fasteners. It reports the expected reductions in structural constraints, tool changes, and robot travel distances, while preventing structurally unsafe modifications by evaluating geometric stability metrics. Experiments on seven household appliances demonstrate that the framework successfully targets redundant fasteners. Removing the recommended fasteners simplified the structural dependencies by eliminating between 8 and 132 structural constraints on the graph depending on each product's structural configuration. Furthermore, it improved robotic operational efficiency by eliminating unnecessary tool change operations and shortening travel distances by 165 to 1675 millimeters wherever structurally permissible.
>
---
#### [new 009] MC-Risk: Multi-Component Risk Fields for Risk Identification and Motion Planning
- **分类: cs.RO**

- **简介: 该论文提出MC-Risk，用于风险识别与运动规划的任务，解决风险定位与早期预警问题，通过多组件风险场实现精准风险建模与轨迹生成。**

- **链接: [https://arxiv.org/pdf/2605.21406](https://arxiv.org/pdf/2605.21406)**

> **作者:** Maximilian Link; Yingjie Xu; Yingbai Hu; Yinlong Liu
>
> **摘要:** We present MC-Risk, a planner-aligned, multi-component risk field on a bird's-eye-view grid that yields early, calibrated, and class-aware risk localization. MC-Risk linearly composes three interpretable modules: (i) a motorized-agent field that fuses a black-box multimodal trajectory predictor with an analytic Gaussian-torus construction whose lateral width grows with speed/curvature and whose height attenuates with look-ahead; (ii) a VRU risk field that replaces isotropic pedestrian blobs with a forward-biased anisotropic kernel aligned to heading and speed; and (iii) a road penalty field that exploits full HD-map topology, imposing an off-road penalty and lane-aware risk exposure for same/opposite directions. We conduct, to our knowledge, the first standardized quantitative evaluation of a risk-field formulation on RiskBench's collision subset. MC-Risk attains the best overall risk localization and the earliest hazard indication. Finally, we demonstrate a plug-and-play planning interface by using the field as an MPC cost density, enabling risk-aware trajectory generation without additional training.
>
---
#### [new 010] Lost in Fog: Sensor Perturbations Expose Reasoning Fragility in Driving VLAs
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，研究传感器扰动下VLA的推理脆弱性。通过实验分析不同传感器干扰对轨迹可靠性的影响，提出CoC一致性作为安全评估指标。**

- **链接: [https://arxiv.org/pdf/2605.21446](https://arxiv.org/pdf/2605.21446)**

> **作者:** Abhinaw Priyadershi; Jelena Frtunikj
>
> **摘要:** Interpretable autonomous driving planners depend not only on generating explanations, but also on those explanations remaining reliable under real-world sensor degradation. In this paper we present a controlled perturbation study of Vision-Language-Action (VLA) robustness in autonomous driving, evaluating Alpamayo R1 (10B parameters) across 1,996 scenarios under eight sensor perturbations (Gaussian noise at four intensities, two lighting extremes, and two fog levels; ${\sim}18{,}000$ inference trials). We find that reasoning consistency is a high-fidelity indicator of trajectory reliability: when Chain-of-Causation (CoC) explanations change after perturbation, trajectory deviation spikes $5.3{\times}$ (21.8m vs 4.1m), with $r\!=\!0.99$ across attack types and $r_{pb}\!=\!0.53$ per-sample (Cohen's $d\!=\!1.12$). A controlled ablation provides evidence that enabling CoC generation is associated with improved trajectory accuracy (11.8% on average across conditions; $p < 0.0001$) under matched inference settings. Over the tested noise range ($\sigma \in \{10, 30, 50, 70\}$), degradation is approximately linear ($R^2\!=\!0.957$), while standard input preprocessing defenses provide only marginal relief. Together, these results establish CoC consistency as a quantitative proxy for planning safety and motivate reasoning-based runtime monitoring for safer VLA deployment.
>
---
#### [new 011] STEAM: A Training-Free Congestion-Aware Enhancement Framework for Decentralized Multi-Agent Path Finding
- **分类: cs.RO**

- **简介: 该论文提出STEAM，用于去中心化多智能体路径规划（MAPF）的无训练增强框架，解决路径拥堵问题。通过轻量级引导提升路径规划效果。**

- **链接: [https://arxiv.org/pdf/2605.20929](https://arxiv.org/pdf/2605.20929)**

> **作者:** Mingyang Feng; Mengnuo Zhang; Shaoyuan Li; Xiang Yin
>
> **摘要:** We propose STEAM (Spatial, Temporal, and Emergent congestion Awareness for MAPF), a training-free test-time enhancement framework for learning-based decentralized Multi-Agent Path Finding (MAPF) in discrete environments. Given a pretrained decentralized policy, STEAM requires no retraining, architectural modification, or replacement by a centralized planner. Instead, it injects lightweight congestion-aware guidance into the original policy execution. STEAM first rolls out the shortest paths induced by the current cost-to-go maps to identify potential future congestion hotspots. Spatially avoidable congestion is mitigated by updating agent-specific cost-to-go information, while spatially unavoidable bottlenecks are handled through temporal logit correction. In addition, emergent local congestion is reduced by a density-aware logit correction based on neighboring agents' corrected cost-to-go maps. Extensive experiments on representative learning-based decentralized MAPF algorithms show that STEAM consistently improves success rate, makespan, and solution cost, with success-rate gains of up to 60% and only minor computational overhead. The implementation is available at this https URL.
>
---
#### [new 012] Intent-First Aerial V2V for Tactical Coordination and Separation: Protocol and Performance Under Density and Disturbance
- **分类: cs.RO; cs.MA; cs.NI**

- **简介: 该论文属于无人机交通管理任务，解决密集空域下的战术协同与分离问题。提出一种基于意图的V2V通信协议，实现局部协调与感知，提升安全性和效率。**

- **链接: [https://arxiv.org/pdf/2605.20595](https://arxiv.org/pdf/2605.20595)**

> **作者:** Mehrnaz Sabet
>
> **备注:** Submitted to IEEE Transactions on Intelligent Transportation Systems
>
> **摘要:** Dense low-altitude aerial operations require more than pre-flight route coordination and last-resort collision avoidance. Once aircraft are airborne, disturbances can emerge on timescales shorter than strategic reauthorization can absorb, while collision avoidance is too late and disruptive to serve as routine traffic management. Although tactical separation is recognized as the intermediate layer, realizing it at scale requires a deployable neighborhood communication mechanism that provides fresh, trusted information for local coordination. This paper presents what is, to our knowledge, the first controller-coupled characterization of an all-airborne, sidelink-class, intent-first vehicle-to-vehicle (V2V) tactical neighborhood exchange stack for dense Unmanned Aircraft System Traffic Management (UTM) operations. Unlike awareness-only broadcast, the proposed exchange combines refreshed state and intent beacons for local awareness, cooperative perception, and degraded-mode assessment with event-triggered messages for yielding, sequencing, release, and contingency coordination. We implement and evaluate this model on an all-airborne V2V stack using sidelink-class C-V2X modules with authenticated freshness checks. Evaluation uses a scenario-driven, high-volume stress campaign supported by real-time, field-anchored infrastructure. Results show that V2V reduces stale-belief divergence, preserves observability through cooperative perception, rejects invalid tactical messages, suppresses false local inference, and structures shared-resource coordination. The implemented stack provides a viable communication layer for tactical separation in lower-to-moderate regimes, but transitions toward guarded fallback as density, impairment, and complexity increase. These findings position intent-first aerial V2V as a bounded enabler for scaling tactical coordination in disturbance-driven urban airspace.
>
---
#### [new 013] SUGAR: A Scalable Human-Video-Driven Generalizable Humanoid Loco-Manipulation Learning Framework
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出SUGAR框架，解决人形机器人通用运动操作问题。通过处理人类视频生成可部署技能，无需任务特定奖励或参考动作，实现高效、稳定的人形机器人控制。**

- **链接: [https://arxiv.org/pdf/2605.20373](https://arxiv.org/pdf/2605.20373)**

> **作者:** Tianshu Wu; Xiangqi Kong; Yue Chen; Qize Yu; Hang Ye; Jia Li; Yizhou Wang; Hao Dong
>
> **备注:** Project Page: this https URL
>
> **摘要:** Building humanoid robots capable of generalizable whole-body loco-manipulation in the real world remains a fundamental challenge. Existing methods either rely on laborious task-specific reward engineering, rigidly replay reference motions that fail to generalize, or depend on costly teleoperation that limits scalability. While human videos capture diverse human behaviors, motion priors inferred from them are inherently imperfect, suffering from occlusion, contact artifacts, and retargeting errors that render them unsuitable for direct policy learning. To address this, we present SUGAR, a scalable data-driven framework that converts diverse human videos into deployable humanoid loco-manipulation skills, without any task-specific reward engineering or reference-motion conditioning at inference. SUGAR proceeds in three stages. First, a fully automated pipeline extracts kinematic interaction priors including human-object motion trajectories and contact labels from unstructured human videos. Second, a privileged physics-based refiner uses a unified mimic reward and progressive state pool to transform imperfect priors into physically feasible, high-fidelity skills. Third, refined skills are distilled into a hierarchical autonomous policy consisting of a command generator and a command tracker. We evaluate SUGAR on six representative loco-manipulation tasks in simulation and real-world humanoid hardware. Our method substantially outperforms reference-tracking baselines, and performance scales clearly with the amount of human video data. It also achieves zero-shot real-world transfer with reliable closed-loop execution, autonomous failure recovery, and stable long-horizon performance under external perturbations. Project Page: this https URL
>
---
#### [new 014] Enhancing Graph-Based SLAM in GNSS-Denied environments by leveraging leg odometry
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决GNSS拒收环境下定位漂移问题。通过融合腿部本体感知数据，提升SLAM精度，减少高度偏差。**

- **链接: [https://arxiv.org/pdf/2605.20484](https://arxiv.org/pdf/2605.20484)**

> **作者:** Léon Perruchot-Triboulet; Luc Jaulin; Kai Xiao
>
> **备注:** 4 pages, 3 figures, 2 tables, for ICRA workshop on Robot Meets GNSS and Ranging for Seamless Autonomy
>
> **摘要:** Autonomous navigation in GNSS-denied environments remains a core challenge for legged robots, where exteroceptive sensors such as LiDAR are prone to elevation drift in geometrically sparse or repetitive scenes. We present a factor graph architecture that augments the LIO-SAM framework with a parallel kinematic lane driven by proprioceptive leg odometry, coupled to the main LiDAR-inertial lane via an identity relative pose constraint with a selective noise model. Applied to a Linxai D50 quadruped platform across two outdoor loops totaling over one kilometer, our approach reduces elevation drift from over 30m to under 30cm and enables convergence in a scene where the baseline pipeline fails entirely. These results suggest that proprioceptive data, already computed onboard for gait control, constitutes a lightweight and effective vertical anchor for SLAM in GNSS-denied settings.
>
---
#### [new 015] PointACT: Vision-Language-Action Models with Multi-Scale Point-Action Interaction
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出PointACT，解决3D环境中机器人操作的视觉-语言-动作对齐问题，通过融合3D点云与语言信息提升操作精度。**

- **链接: [https://arxiv.org/pdf/2605.21414](https://arxiv.org/pdf/2605.21414)**

> **作者:** Shizhe Chen; Paul Pacaud; Cordelia Schmid
>
> **备注:** Accepted to RSS 2026; project webpage: this https URL
>
> **摘要:** Vision-Language-Action (VLA) models have shown strong potential for general-purpose robotic manipulation by leveraging large pretrained vision-language backbones. However, most existing VLAs rely primarily on 2D visual representations, which limit their ability to reason about fine-grained geometry and spatial grounding - capabilities that are essential for precise and robust manipulation in 3D environments. In this paper, we propose PointACT, a dual-system 3D-aware VLA policy that integrates hierarchical 3D point cloud representations directly into the action decoding process. PointACT employs a multi-scale point-action interaction mechanism with efficient bottleneck window self-attention, enabling evolving action tokens to densely attend to both local geometric detail and global scene structure. We evaluate PointACT on the LIBERO and RLBench benchmarks and systematically compare it against monolithic and dual-system VLA baselines, including variants augmented with point cloud inputs. PointACT achieves consistent improvements across both benchmarks, increasing success rates by 10% on the challenging RLBench-10Tasks suite over state-of-the-art pretrained VLAs, with even larger gains when the vision-language backbone is frozen and the action expert is trained from scratch. Extensive ablation studies demonstrate that tightly coupling hierarchical 3D geometry with pretrained 2D semantic representations is critical for robust and spatially grounded robot control. Our results also highlight the promise of pretrained 3D representations for 3D-aware VLA policies.
>
---
#### [new 016] Learning Structural Latent Points for Efficient Visual Representations in Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在解决3D感知与操作中隐式和显式表示的局限。提出一种混合结构潜在点表示，结合两者优势，提升任务成功率与效率。**

- **链接: [https://arxiv.org/pdf/2605.21258](https://arxiv.org/pdf/2605.21258)**

> **作者:** Yicheng Jiang; Jiaxu Wang; Junhao He; Zesen Gan; Junhao Li; Qiang Zhang; Jingkai Sun; Jiahang Cao; Mingyuan Sun; Xiangyu Yue; Qiming Shao
>
> **摘要:** Current 3D-aware pretraining methods for embodied perception and manipulation are largely built on differentiable rendering frameworks, producing either fully implicit neural fields or fully explicit geometric primitives. Implicit representations, while expressive, lack explicit structural cues, whereas explicit ones preserve geometry but suffer from resolution limits and weak generalization. To address these limitations, we propose a novel pretraining framework that learns a hybrid representation-structural latent points. Specifically, we insert a point-wise latent variational autoencoder into the latent space of a point-cloud autoencoder, jointly regularizing point-wise features and coordinates toward a Gaussian prior. The resulting compact latent preserves coarse structural tendencies, which do not encode precise geometry but capture richer rough shape and semantic information, effectively combining the expressiveness of implicit representations with the structural priors of explicit ones. In addition, informed by shared design choices in prior work, we develop a streamlined, efficient 3DGS-based rendering pipeline that is deliberately kept lightweight, improving efficiency while leaving greater representational capacity to the front-end latent module. Extensive evaluations on RLBench, ManiSkill2, and a real-robot platform demonstrate consistent gains in task success, sample efficiency, and robustness to viewpoint and scene variations over strong baselines. Ablation studies further confirm that each component of our framework is critical to overall performance.
>
---
#### [new 017] To Select or not to Select, that is the Question: Distilling Robot Skill Prediction into a Small Ensemble
- **分类: cs.RO**

- **简介: 论文研究机器人技能预测任务，解决如何为任务选择合适机器人的问题。通过合成数据训练小型模型，提升任务与机器人技能的匹配效果。**

- **链接: [https://arxiv.org/pdf/2605.21242](https://arxiv.org/pdf/2605.21242)**

> **作者:** Haechan Mark Bong; Simon Roy; Euhid Aman; Giovanni Beltrame
>
> **摘要:** As robot fleets become more heterogeneous, including humanoids, rovers, quadrupeds, and drones, selecting the right robot for a task becomes a core systems problem. We study robot skill prediction: mapping a natural-language task description to the physical capabilities required to execute it, such as fly, wheels, legs, surface water, under water and hands. Since labelled data that maps natural-language task descriptions to robot's physical capabilities does not exist, we construct a synthetic task-to-skill dataset using LLM-assisted generation and targeted label auditing. Trained on this data, a ~133M-parameter ensemble of two fine-tuned sentence encoders (mpnet + MiniLM) reaches 83.5% task-to-skill matching on a stratified 200 task dataset, outperforming Kimi K2 (1T MoE) at 72.0%, GPT-OSS-120B at 71.5%, and Llama-4-Scout-17B at 69.0% under the same zero-shot prompt. These results suggest that, for fixed robot skill taxonomies, small specialized models trained on synthetic data can outperform much larger general-purpose LLMs for fleet-level task routing.
>
---
#### [new 018] SubTGraph: Large-Scale Subterranean Environment Synthesis with Controllable Topological Variability for Robotic Autonomy Validation
- **分类: cs.RO**

- **简介: 该论文提出SubTGraph，用于生成多样化地下环境，解决缺乏大规模仿真基准的问题。任务为机器人自主性验证，工作包括构建可控制拓扑的仿真环境并进行多场景测试。**

- **链接: [https://arxiv.org/pdf/2605.20917](https://arxiv.org/pdf/2605.20917)**

> **作者:** F. Labra Caso; A. Saradagi; S. Fredriksson; S. Nordström; A. Koval; G. Nikolakopoulos
>
> **备注:** 16 pages, 18 figures
>
> **摘要:** Subterranean (SubT) environments have been a frontier for autonomous robotics, driven by the push for automation of mining operations and the interest in planetary exploration (Martian Lava Tubes). Due to the challenges involved in accessing real SubT environments, rigorous hardening of autonomy stacks in realistic simulation environments is critical. This article fills a well-known gap, which relates to the unavailability of a large-scale simulation-based benchmarking infrastructure for rigorous statistical evaluation of robotic autonomy, due to which it is common for SubT research articles to present validation results in a few environments at best. This article presents SubTGraph, a novel framework for rapid synthesis of multi-level SubT environments with high variability, incorporating user specifications related to topology, dimensionality, textures, etc., to generate distinct environments such as operational mines, natural caves and lava tubes. SubTGraph builds a cost matrix from user-specified structural constraints to guide the classical Dijkstra algorithm to procedurally generate SubT worlds utilizing topometric tiles from the DARPA World Generator. Three robotics case-studies are investigated to demonstrate the utility of SubTGraph for rigorous validation of different layers in the robotic autonomy stack. Structural semantic segmentation is validated against topometric ground truths, multi-agent path planning is widely tested for identification of patterns and trends in the algorithm behavior and LIO SLAM is stress-tested in challenging subterranean sections to identify failure cases. The SubTGraph world creation codebase is open-sourced (this https URL) along with a database consisting of 150 highly variable underground worlds.
>
---
#### [new 019] Terrestrial Soft Mobile Robots: A Review
- **分类: cs.RO**

- **简介: 本文综述了陆地软体移动机器人的研究现状，聚焦于无轮运动系统。论文旨在解决软体机器人在多种应用场景中的技术挑战，涵盖运动策略、驱动方法、建模与控制等方面的工作。**

- **链接: [https://arxiv.org/pdf/2605.20304](https://arxiv.org/pdf/2605.20304)**

> **作者:** Dimuthu D. K. Arachchige
>
> **摘要:** Soft mobile robots have emerged as a promising area of research with potential applications in various disciplines including but not limited to search-and-rescue, service, surveillance, explorations, and manufacturing. In this article, we provide a comprehensive review of the current state of soft mobile robot research, focusing on wheelless terrestrial locomotive systems. We include past and present developments in locomotion strategies, actuation methods, modeling approaches, and control systems. Further, we identify key research challenges that must be overcome to enable the widespread adoption of soft mobile robots in various applications. Overall, this article provides a valuable resource for researchers and practitioners interested in the field of soft mobile robots and soft robotics.
>
---
#### [new 020] VBT-MPC: Vision-Based Tactile MPC for Contour Following
- **分类: cs.RO**

- **简介: 该论文属于机器人轮廓跟踪任务，解决接触保持与精确跟踪的问题。提出VBT-MPC框架，直接在轮廓特征空间控制，无需额外位姿估计或复杂力控。**

- **链接: [https://arxiv.org/pdf/2605.20392](https://arxiv.org/pdf/2605.20392)**

> **作者:** Edison Velasco-Sanchez; Luis F. Recalde; Guanrui Li; Pablo Gil
>
> **备注:** This article has been accepted for publication in IEEE Robotics and Automation Letters. This is a preprint version. This work was supported by the Interreg-VI Sudoe and European Regional Development Funds through the REMAIN Project under Grant S1/1.1/E0111
>
> **摘要:** Tactile sensing plays a key role in robotic manipulation, particularly in tasks like surface inspection. Successful execution requires maintaining contact while accurately tracking object contours. In this work, we propose a Vision-Based Tactile Model Predictive Control (VBT-MPC) framework for robotic contour following using a Vision-Based Tactile Sensor (VBTS) mounted in an eye-in-hand configuration. The proposed controller operates directly in contour features space, thereby avoiding the need for separate pose-estimation modules or complex force-control architectures. We further compare our VBT-MPC with visual-servoing strategies adapted to tactile features, and evaluate contour tracking on objects with diverse geometries and materials in both simulation and real-world experiments.
>
---
#### [new 021] roto 2.0: The Robot Tactile Olympiad
- **分类: cs.RO; cs.LG**

- **简介: 该论文介绍roto 2.0，一个用于触觉强化学习的基准平台，解决盲操作任务中的算法挑战，通过标准化环境提升研究效率。**

- **链接: [https://arxiv.org/pdf/2605.21429](https://arxiv.org/pdf/2605.21429)**

> **作者:** Elle Miller; Jayaram Reddy; Ayush Deshmukh; Trevor McInroe; David Abel; Oisin Mac Aodha; Sethu Vijayakumar
>
> **备注:** Accepted to 7th ViTac Workshop, ICRA 2026
>
> **摘要:** Tactile-based reinforcement learning (RL) is currently hindered by fragmented research and a focus on over-saturated orientation tasks. We introduce v2 of the Robot Tactile Olympiad (\texttt{roto 2.0}), a GPU-parallelised benchmark designed to standardise tactile-based RL across four distinct robotic morphologies (16-DOF to 24-DOF). Unlike prior benchmarks, roto focuses on end-to-end "blind" manipulation, utilising only proprioception and tactile sensing without state information or distillation. We demonstrate a significant performance leap, with our blind agents achieving 13 Baoding ball rotations in 10 seconds, an order of magnitude faster than current state-of-the-art speeds. By open-sourcing our environments and robustly tuned baselines, we reduce the barrier to entry and enable researchers to prioritise fundamental algorithmic challenges over tedious RL tuning. Website: this https URL
>
---
#### [new 022] Jointly Learning Predicates and Actions Enables Zero-Shot Skill Composition
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人技能学习任务，旨在解决现有方法无法泛化新组合技能的问题。提出PACTS模型，联合建模动作与符号结果，实现零样本技能组合。**

- **链接: [https://arxiv.org/pdf/2605.20648](https://arxiv.org/pdf/2605.20648)**

> **作者:** Benedict Quartey; Sebastian Castro; Eric Rosen; Wil Thomason; George Konidaris; Stefanie Tellex
>
> **摘要:** Learning from Demonstration (LfD) enables robots to learn complex behaviors from expert examples, yet existing approaches often fail to generalize to new compositions of known skills without retraining. Modern generative policies model distributions over action trajectories alone, thus are unable to reason about the symbolic outcomes required for robust composition. We propose that skills should jointly model action trajectories and the symbolic outcomes they induce. To address this gap, we introduce Predicate Action Skills (PACTS), a class of closed-loop visuomotor policies that model skills as a joint generative process over action and predicate belief trajectories, producing coherent action-outcome rollouts within a single model. Jointly generating actions and predicates enables PACTS to learn internal representations that improve both action generation and predicate classification. Furthermore, we demonstrate zero-shot composition of learned skills via planning by leveraging online predicate predictions from PACTS as a symbolic interface for sequencing and monitoring execution. Project website: this https URL
>
---
#### [new 023] GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出GaussianDream，用于机器人操作任务，解决3D几何和视觉结构监督不足的问题。通过构建3D高斯世界模型，提供密集的视觉和场景流监督，提升操作精度。**

- **链接: [https://arxiv.org/pdf/2605.20752](https://arxiv.org/pdf/2605.20752)**

> **作者:** Zijian Zhang; Yuqing Jiang; Qian Cheng; Si Liu; Ding Zhao; Ping Luo; Weitao Zhou; Haibao Yu
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** Vision-language-action (VLA) policies have advanced language-conditioned robotic manipulation by transferring semantic priors from pretrained vision-language models to action generation. Yet, standard action-imitation training often provides limited explicit supervision for 3D geometry, dense visual structure, and short-horizon environment evolution, which are critical for physically precise manipulation. We introduce \textbf{GaussianDream}, a feed-forward 3D Gaussian world-model plug-in that turns robot trajectories into structured spatial-temporal supervision. The key idea is to couple current Gaussian reconstruction with horizon-conditioned future Gaussian prediction during training, forcing a compact spatio-temporal prefix to be decodable into renderable 3D Gaussian states. This enables dense RGB rendering, depth, and pseudo 3D scene-flow supervision without requiring test-time Gaussian decoding. At inference, GaussianDream discards all auxiliary decoding heads and retains only the learned prefix to condition action generation, avoiding rendering, video rollout, or additional planning during closed-loop control. Experiments on LIBERO, RoboCasa Human-50, and real-robot tasks demonstrate strong and highly competitive performance, achieving \textbf{98.4\%} average success on LIBERO, \textbf{52.6\%} on RoboCasa Human-50, and \textbf{50.0\%} in real-world evaluation.
>
---
#### [new 024] Q-SpiRL: Quantum Spiking Reinforcement Learning for Adaptive Robot Navigation
- **分类: cs.RO; quant-ph**

- **简介: 该论文属于机器人导航任务，旨在解决动态环境中高效、稳定路径规划问题。提出Q-SpiRL框架，结合量子计算与脉冲神经网络，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2605.20801](https://arxiv.org/pdf/2605.20801)**

> **作者:** Mohamed Khair Altrabulsi; Nouhaila Innan; Alberto Marchisio; Muhammad Kashif; Muhammad Shafique
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Adaptive robot navigation in dynamic environments requires policies that can reach the target reliably while producing efficient and stable trajectories. This paper presents Q-SpiRL, a quantum spiking reinforcement learning framework for obstacle-aware robot navigation. The framework develops and evaluates five agent families: tabular Q-learning, classical MLP, classical SNN, quantum-enhanced MLP (QMLP), and quantum-enhanced spiking neural network (QSNN). While all models are implemented under a unified training and evaluation pipeline, the QSNN is the central architecture of interest, as it combines spike-based temporal processing with variational quantum feature transformation. Experiments are conducted across three grid-world environments of increasing size, namely 20x20, 30x30, and 40x40, with both static and dynamic obstacles. Performance is assessed using success rate, success-weighted path length, path length, and turn rate under deterministic inference. Results show that QSNN achieves the strongest overall trade-off between task completion, trajectory efficiency, and motion smoothness, reaching up to 99% success rate while maintaining high path efficiency in the most challenging setting. Execution on IBM quantum hardware further demonstrates the feasibility of deploying the proposed hybrid policy under real-device conditions.
>
---
#### [new 025] Fault-Tolerant, Rigidity-Preserving Control of Inflatable Truss Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决 inflatable truss 机器人在电机故障下的可靠性问题。通过引入容错控制、约束优化和闭环位置控制，提升系统鲁棒性与定位精度。**

- **链接: [https://arxiv.org/pdf/2605.20561](https://arxiv.org/pdf/2605.20561)**

> **作者:** James Wade; Isaac Weaver; Mihai Stanciu; Nathan Usevitch
>
> **摘要:** Isoperimetric robotic trusses can adapt to different tasks and environments because they have a high strength-to-weight ratio, can change their own shape dramatically, and can be reconfigured into a variety of different shapes. However, motor failures in operational environments can severely limit operational capabilities if not properly addressed. This paper presents a fault-tolerant control framework for an inflatable robotic truss that maintains functionality despite motor failures, shown through three key contributions. First, we extend the kinematic optimization to handle arbitrary combinations of motor failures by imposing equality constraints to ensure failed actuators are not used. Second, we introduce discrete-time control barrier function (DTCBF) constraints that mathematically guarantee structural rigidity while maximizing workspace utilization, a critical requirement for reliable operation of truss robots under discrete-time control. Third, we implement closed-loop position control using onboard encoder feedback and a forward kinematics-based state estimator, improving positional accuracy in the presence of disturbances. We validate our approach through simulation and hardware experiments on a 2D isoperimetric truss testbed. For a 2D configuration with 6 actuators, we demonstrate >69% workspace preservation under single-motor failures and a >25% improvement in tracking accuracy with closed-loop control. These results establish a foundation for more robust and resilient isoperimetric truss robots operating under degraded actuation.
>
---
#### [new 026] A Semantic and Occlusion-Aware GM-PHD Filter
- **分类: cs.RO**

- **简介: 该论文属于目标跟踪任务，旨在解决复杂场景下的目标初始化问题。通过引入语义信息和遮挡感知的出生模型，提升GM-PHD滤波器的跟踪性能。**

- **链接: [https://arxiv.org/pdf/2605.20666](https://arxiv.org/pdf/2605.20666)**

> **作者:** Jovan Menezes; Mark Campbell
>
> **备注:** Accepted at ICRA 2026
>
> **摘要:** This paper proposes a new birth model including semantic information derived from deep learning to create an occlusion-aware Gaussian Mixture Probability Hypothesis Density (GM-PHD) filter. Unlike prior approaches that rely on simplistic or uniform assumptions, the proposed Semantic-Occlusion Aware (S-OA) birth model defines initialization terms by explicitly considering regions of occlusion and by leveraging semantic information about the environment. This enables the filter to accurately represent where new objects are more likely to appear, thereby improving tracking performance in complex and high-density driving scenarios. The method is evaluated through Monte Carlo simulations and experiments on the KITTI dataset. Performance is assessed by measuring the latency between first detection and track initiation, along with the mean absolute cardinality error and the Optimal Subpattern Assignment (OSPA) metric. Results demonstrate that the S-OA birth model reduces initialization delay in occlusion-heavy settings, matching or outperforming the strongest baseline in approximately 70% of cases. A sensitivity analysis of birth model weights is also provided. Overall, the findings underscore the benefits of integrating occlusion reasoning and semantic priors into Bayesian tracking frameworks for autonomous driving.
>
---
#### [new 027] Spacetime Optimal-Transport Attention for Visuo-Haptic Imitation Learning of Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文针对高接触任务的模仿学习，解决多模态感知融合与安全控制问题，提出SO-TA模型提升操作成功率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.20433](https://arxiv.org/pdf/2605.20433)**

> **作者:** Yue Feng; Weicheng Huang; I-Ming Chen
>
> **备注:** 8 pages, 16 figures, 3 tables. Preprint
>
> **摘要:** Contact-rich manipulation tasks such as tight-clearance insertion, connector mating, polishing, and surface-conforming wiping remain difficult for data-driven controllers because they couple discontinuous contact dynamics, partial observability, and strict safety constraints. No single sensing modality suffices: vision supplies global context before contact, force/torque (F/T) feedback governs interaction after contact, and proprioceptive pose provides a consistent kinematic backbone. Most prior imitation-learning policies for contact-rich tasks operate on uni- or bi-modal signals, and the few that fuse three modalities typically adopt off-the-shelf attention modules with no explicit prior on how attention mass should be distributed across task-relevant regions. We present Spacetime Optimal-Transport Attention (SO-TA), a tri-modal fusion backbone that replaces softmax-normalized patch attention by an entropy-regularized Optimal Transport (OT) alignment between force-pose-derived sub-queries and visual patches. Explicit marginal constraints act as a structured inductive bias for contact-rich tasks, encouraging conditioning-aware spatial selection that is stable across illumination, distractors, and partial occlusion. SO-TA is paired with a diffusion-based sequence policy mapping observation windows to pose-action chunks. We evaluate SO-TA on three real-robot tasks: tight peg-in-hole assembly, BCM wiring-connector insertion, and curved-surface mark erasing. With ~200 rollouts per condition, SO-TA reaches 100% success on tight peg-in-hole versus 93% for cross-attention at matched capacity, and retains 82.5% success under illumination, distractor, and partial-occlusion perturbations where a concatenation baseline drops to 43.5%. OT-derived patch heatmaps and leave-one-out modality-influence ratios provide interpretable, phase-dependent diagnostics.
>
---
#### [new 028] Safety-Critical Control for Smoothed Implicit Contact Dynamics
- **分类: cs.RO**

- **简介: 该论文属于安全控制任务，旨在解决平滑隐式接触动力学中的安全约束问题。通过引入边界聚焦滚动和控制屏障函数，确保接触力在安全范围内。**

- **链接: [https://arxiv.org/pdf/2605.21138](https://arxiv.org/pdf/2605.21138)**

> **作者:** Haegu Lee; Yitaek Kim; Christoffer Sloth
>
> **摘要:** Smoothed implicit contact dynamics enables gradient-based planning and control for contact-rich tasks without predefined mode sequences. However, safety-critical control remains challenging because implicit contact dynamics makes safety-filter design nontrivial. The smoothing parameter $\kappa$ relaxes contact complementarity constraints, which makes the dynamics smooth but affects the contact force. This paper provides a method for bounding the actual contact force despite the use of relaxed complementarity constraints. We show that constraint violations can be non-monotonic in $\kappa$. Smaller $\kappa$ reduces force-approximation error, but it does not necessarily improve safety performance. To address this issue, we introduce boundary-focused rollouts to screen $\kappa$ by comparing the safety margin with the approximation error. We then develop a discrete-time control barrier function (CBF) framework based on a first-order Taylor approximation of the implicitly defined contact force. To account for possible force under-prediction, we augment the resulting safety constraint with a fixed robust margin. Simulations on four contact-rich systems show that the proposed method eliminates force violations observed under a standard CBF.
>
---
#### [new 029] Adaptive Human-Robot Collaboration for Masonry Construction Under Material and Assembly Uncertainty
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机协作任务，旨在解决建筑施工中材料和装配不确定性带来的误差累积问题。通过投影引导和激光反馈调整，提升协作精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.20264](https://arxiv.org/pdf/2605.20264)**

> **作者:** Jutang Gao; Arash Adel
>
> **备注:** Accepted for publication in Proceedings of the 43rd International Symposium on Automation and Robotics in Construction (ISARC 2026)
>
> **摘要:** Human-robot collaboration in construction is often challenged by limited robot-to-human communication and the need to adapt to tolerance accumulation arising from material and assembly uncertainties. We present an adaptive human-robot collaborative workflow for masonry construction that addresses communication limitations and tolerance accumulation, demonstrated through a brickwork case study in which a robot places bricks while a human applies adhesive. This workflow is enabled by two complementary mechanisms: 1) an end-effector-mounted projector that provides spatially registered, just-in-time projection guidance for manual adhesive application, and 2) laser scanning for feedback-driven grasping and placement pose correction. Together, these mechanisms enable adjustment of human and robotic actions in response to material variability and accumulated assembly tolerances. Full-scale experiments across conventional running-bond and nonstandard configurations demonstrate that projection guidance improves adhesive application consistency and reduces application time, while laser-based correction maintains level courses and avoids collision-prone failures associated with open-loop execution. These results indicate that integrating spatial projection with feedback-driven adaptation, enabled by material and as-built sensing, can mitigate tolerance accumulation and improve precision and robustness in human-robot collaborative construction.
>
---
#### [new 030] VLA-REPLICA: A Low-Cost, Reproducible Benchmark for Real-World Evaluation of Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出VLA-REPLICA，一个低成本、可复现的真实世界基准，用于评估视觉-语言-动作模型。解决真实环境评估不足的问题，通过构建一致的实验环境和多样化任务，提升模型评估的可靠性与通用性。**

- **链接: [https://arxiv.org/pdf/2605.20774](https://arxiv.org/pdf/2605.20774)**

> **作者:** Alex S. Huang; Jiahui Zhang; Shiqing Tang; Yu Xiang
>
> **摘要:** Vision-Language-Action (VLA) models have shown strong promise for general-purpose robotic manipulation, but their real-world evaluation remains limited by a lack of accessible, reproducible, and consistent benchmarks. Simulation benchmarks fail to capture real-world complexity, while existing real-world benchmarks often require expensive hardware, centralized evaluation, or are limited in task diversity. We introduce VLA-REPLICA, a low-cost, easily reproducible real-world benchmark for evaluating VLA models. Built from off-the-shelf components, our system can be quickly assembled and replicated across laboratories, providing a consistent environment for policy evaluation anywhere in the world. VLA-REPLICA includes a diverse suite of manipulation tasks and a small-scale demonstration dataset for target-domain adaptation, with real-world evaluation protocols for both in-distribution and out-of-distribution settings. Experiments with imitation learning and state-of-the-art VLA models reveal model strengths and limitations, while consistent results across independently constructed setups demonstrate the reproducibility of our benchmark.
>
---
#### [new 031] From swept contact to pose: Probe-aware registration via complementary-shape docking
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，解决高精度模型与真实场景的配准问题。通过互补形状对接方法，无需外部传感器即可实现精确位姿估计。**

- **链接: [https://arxiv.org/pdf/2605.21398](https://arxiv.org/pdf/2605.21398)**

> **作者:** Chen Chen; Yunwen Li; Yifan Xu; Xiangjie Yan; Chang Shu; Jianxia Hou; Shiji Song; Xiang Li
>
> **备注:** 8 pages, 9 figures, accepted to ICRA 2026
>
> **摘要:** Accurate registration between a prior model and the real scene is essential for high-precision robotic manipulation, yet optical methods suffer from long calibration chains, line-of-sight constraints, and fabrication errors. We propose a calibration-free alternative that reformulates contact registration as complementary-shape docking between the object and the probe's swept volume, explicitly accounting for probe geometry and leveraging both contact and non-contact evidence. Our solver integrates a global-to-local search via 3D FFT correlation over low-discrepancy SO(3) samples, then followed by continuous SE(3) refinement using Lie-algebra updates and analytic contact sensitivities. This pipeline yields efficient exploration and metric-grade convergence without fragile point correspondences. Simulation across free-form meshes achieved sub-0.04 mm and sub-0.4° accuracy and robustness to pose noise and contact loss. On a tooth-preparation robot, our method attained 0.42 mm and 3.75°, outperforming an optical tracker registration while requiring no external sensors. These results demonstrate a practical and precise registration strategy for surgical and industrial robots.
>
---
#### [new 032] Reinforcement Learning for Risk Adaptation via Differentiable CVaR Barrier Functions
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决不确定环境下避障效率与安全问题。通过结合强化学习与CVaR安全层，实现风险自适应导航。**

- **链接: [https://arxiv.org/pdf/2605.21257](https://arxiv.org/pdf/2605.21257)**

> **作者:** Xinyi Wang; Taekyung Kim; Bardh Hoxha; Georgios Fainekos; Dimitra Panagou
>
> **备注:** Project page: this https URL
>
> **摘要:** Planning through crowded environments under uncertain obstacle motions remains difficult, as stochastic interactions often induce overly conservative behavior or reduced efficiency. To address this challenge, we propose an end-to-end risk adaptation framework for crowd navigation under obstacle-motion uncertainty modeled by a Gaussian mixture model. The framework combines reinforcement learning~(RL) with a differentiable quadratic-program safety layer based on Conditional Value-at-Risk~(CVaR) barrier functions, jointly learning nominal control input, risk level, and safety margin and enforcing explicit probabilistic safety constraints. This design enables context-aware adaptation, promoting efficient behavior while invoking caution only when necessary. We conduct extensive evaluations in dynamic, uncertain, and crowded environments across varying obstacle densities and robot models, and further assess generalization under three out-of-distribution cases. Comparisons across optimization-based, RL-based, and integrated RL and optimization methods are provided, and the proposed method is shown to deliver the strongest overall performance in safety, efficiency, and generalization under uncertainty.
>
---
#### [new 033] A Terrain-Adaptive epsilon-Constraint MPC for Uneven Terrain Kinodynamic Planning
- **分类: cs.RO**

- **简介: 该论文属于车辆路径规划任务，解决不平地形下车辆动力学规划问题。通过自适应epsilon约束MPC方法，提升导航成功率和姿态稳定性。**

- **链接: [https://arxiv.org/pdf/2605.21188](https://arxiv.org/pdf/2605.21188)**

> **作者:** Otobong Jerome; Geesara Kalathunga; Tiago Nascimento
>
> **摘要:** Kinodynamic planning for car-like vehicles on uneven terrain requires simultaneously optimizing competing objectives such as path efficiency and pose stability. This work presents an adaptive epsilon-constraint method integrated into a Model Predictive Control (MPC) framework, where the epsilon bounds are dynamically adjusted based on terrain descriptors to explore the Pareto front in real time. To capture vehicle-terrain dynamics, we develop a semi-parametric model combining analytical vehicle dynamics with a Sparse Gaussian Process (SGP) trained on the same terrain descriptors. The proposed epsilon-MPC is evaluated against MPPI and GAKD baselines, achieving a 94% navigation success rate while reducing maximum orientation deviation by 24% and improving multi-objective trade-off quality by 23%.
>
---
#### [new 034] Proximal State Nudging: Reducing Skill Atrophy from AI Assistance
- **分类: cs.RO; cs.HC; cs.LG**

- **简介: 该论文属于人机协作任务，解决AI辅助导致的技能退化问题。提出PSN算法，在提升任务性能的同时促进用户技能发展。**

- **链接: [https://arxiv.org/pdf/2605.20355](https://arxiv.org/pdf/2605.20355)**

> **作者:** Megha Srivastava; Jonathan Ouyang; Eric Zhou; Andrew Silva; Emily Sumner; Dorsa Sadigh; Yuchen Cui; Deepak Gopinath; Guy Rosman
>
> **备注:** 9 pages
>
> **摘要:** Skill atrophy, the gradual decline of human capability under AI assistance, poses a safety risk in shared-control of semi-autonomous systems, where operators may be unable to distinguish their own inputs from autonomous corrections. We propose Proximal State Nudging (PSN), a shared autonomy algorithm that jointly optimizes for skill development and task performance by nudging users toward states estimated to be most learnable. We first show that PSN outperforms existing shared autonomy baselines in balancing student improvement in unassisted reward with overall shared performance, using simulated students in the classic LunarLander environment. We then present, to the best of our knowledge, the first human subject studies of a planner incorporating learning-compatible shared autonomy: across two driving tasks in the CARLA simulator (High Performance Racing and Parallel Parking, n = 60), PSN produces up to 7x larger gains in unassisted skill than standard blended shared autonomy, while incurring 50% fewer collisions than unassisted self-practice.
>
---
#### [new 035] Humanoid Whole-Body Manipulation via Active Spatial Brain and Generalizable Action Cerebellum
- **分类: cs.RO**

- **简介: 该论文研究人形机器人全身操作任务，解决复杂环境下的空间理解和动作泛化问题。提出双模块框架，提升机器人空间感知与动作生成能力。**

- **链接: [https://arxiv.org/pdf/2605.21133](https://arxiv.org/pdf/2605.21133)**

> **作者:** Zhizhao Liang; Yi-Lin Wei; Xuhang Chen; Mu Lin; Yi-Xiang He; Zhexi Luo; Jun-Hui Liu; Kun-Yu Lin; Wei-Shi Zheng
>
> **备注:** Project page: this https URL
>
> **摘要:** In this paper, we explore spatial-aware humanoid whole-body manipulation task. Compared with tabletop settings, this task poses two key challenges: 1) Spatial understanding is challenging in complex 3D environments with diverse spatial relations. 2) Action generation is difficult to generalize, as limited and costly real-robot data restricts data-driven models generalization. To address these challenges, we propose a generalizable humanoid loco-manipulation framework that leverages the spatial perception and action generation capabilities of multi-agent large models. Specifically, our framework includes two components: Active Spatial Brain for active spatial perception and decision-making, and Generalizable Action Cerebellum for executable robot action generation. The first component actively perceives the spatial scene and makes decisions on task planning and subtask decomposition. The second component generate executable robot actions based on the decisions made by the first module without needs of task-specific real robot data. To benchmark our framework, we design a set of spatial manipulation tasks from two perspectives: evaluating spatial perception and understanding, and assessing real-robot task performance. The results demonstrate strong performance on both aspects across diverse tasks and environments.
>
---
#### [new 036] HITL-D: Human In The Loop Diffusion Assisted Shared Control
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文提出HITL-D框架，解决人机协作中的多步骤操作问题。通过结合扩散策略与人类控制，提升操作效率与用户体验。**

- **链接: [https://arxiv.org/pdf/2605.21460](https://arxiv.org/pdf/2605.21460)**

> **作者:** Riley Zilka; Sergey Khlynovskiy; Allie Wang; Martin Jagersand
>
> **备注:** Accepted for presentation at ICRA 2026
>
> **摘要:** Autonomous manipulation systems have achieved remarkable capabilities, yet the integration of human expertise with diffusion-based policies in shared control remains relatively unexplored. In this paper, we propose Human-In-The-Loop Diffusion (HITL-D), a shared control framework that enhances user performance in multi-step, insertion, and fine manipulation tasks. HITL-D leverages a novel combination of diffusion-based policies and human control to provide autonomous end effector orientation updates conditioned on a scene point cloud and the Cartesian position of the end effector. This approach reduces the number of joystick control axes required, thereby lowering mental workload. In a multi-task user study with 12 participants, HITL-D reduced average task completion times by 40%, decreased perceived workload by 37%, and improved Likert-scale ratings for independence, intuitiveness, and confidence compared to traditional teleoperation methods. These results demonstrate that HITL-D effectively integrates human expertise with autonomous assistance, improving both objective and subjective aspects of teleoperation.
>
---
#### [new 037] Perception of Social Robots as Communication Partners in Healthcare for Older Adults
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决老年人与社交机器人互动的有效性问题。通过实验对比分析，发现机器人可作为有效的沟通伙伴，减轻护理负担。**

- **链接: [https://arxiv.org/pdf/2605.21053](https://arxiv.org/pdf/2605.21053)**

> **作者:** Hana Yamamoto; Carlotta Julia Mayer; Charlotte Raithel; Theresa Buchner; Christian Werner; Yasuhisa Hirata; Monika Eckstein; Katja Mombaur
>
> **备注:** 31 pages, 10 figures, Under review at International Journal of Social Robotics
>
> **摘要:** Addressing the global caregiver shortage through socially assistive robots necessitates a deep understanding of their psychological and physiological impacts on older adults during human-robot interaction (HRI). This study addresses whether social robots can serve as effective interaction partners compared to humans, and if "positive prompts" can similarly enhance these interactions. We conducted a comparative study with 35 participants (aged 70+). Our multi-modal analysis, integrating facial expression data, heart rate variability, and subjective questionnaires, revealed no significant differences in overall stress levels between human and robot interactions. Facial expression analysis confirmed that the robot was accepted as a valid interaction partner, while physiological data showed slightly lower heart rates during robot interactions, suggesting a more relaxed state compared to human-led sessions. These findings indicate that social robots can engage older adults without inducing psychological strain and are capable of alleviating caregiver burden by performing structured tasks, such as health-sensing surveys. Future work should address the identified "appearance-content mismatch" in robot design to facilitate even more natural and effective interactions.
>
---
#### [new 038] The Yes-Man Syndrome: Benchmarking Abstention in Embodied Robotic Agents
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉语言规划任务，旨在解决 embodied agents 在面对模糊或不可行指令时的拒绝回答问题。工作包括提出分类体系和生成拒绝指令的框架 RoboAbstention。**

- **链接: [https://arxiv.org/pdf/2605.20544](https://arxiv.org/pdf/2605.20544)**

> **作者:** Doguhan Yeke; Elif Su Temirel; Ananth Shreekumar; Brandon Lee; Dongyan Xu; Z Berkay Celik
>
> **摘要:** Vision-language models (VLMs) are used as high-level planners for embodied agents, translating natural language instructions and visual observations into action plans. While prior work has studied abstention in LLMs, existing benchmarks are largely text-only and do not capture the perceptual grounding and physical constraints inherent to embodied robotics environments. In such settings, abstention requires recognizing when instructions are ambiguous, physically infeasible, based on false premises, or otherwise unresolvable given the available sensory modalities and context. To address this gap, we introduce a taxonomy to categorize abstention in the context of embodied robotics and present RoboAbstention, a scalable and auditable framework for generating abstention instructions grounded in images gathered from five robotics datasets. RoboAbstention instantiates the taxonomy through a three-phase pipeline: (1) structured visual grounding, (2) deterministic constraint derivation, and (3) controlled instruction generation via category-specific templates. This enables the construction of a diverse dataset with verifiable abstention conditions. We evaluate several frontier VLMs and find that all models exhibit significant weaknesses in abstention, including those with advanced reasoning capabilities. The best-performing model, Gemini 2.5 Flash, abstains on only 39.0% of our 6,069 benchmark instructions, while the embodied planner Gemini Robotics ER 1.6 Preview abstains on just 16.5%. We further explore methods for improving abstention in VLM planners, such as defensive prompting and in-context learning, and find that these interventions substantially improve performance, reaching 93.6% abstention rate for Gemini Robotics ER 1.6 Preview and 88.6% for GPT 5.4 Mini, yet no approach fully solves the problem. We open-source RoboAbstention at this https URL.
>
---
#### [new 039] CMC-Opt: Constraint Manifold with Corners for Inequality-Constrained Optimization
- **分类: cs.RO**

- **简介: 该论文属于优化问题领域，解决机器人中等式与不等式约束下的优化问题。通过引入“带角的约束流形”框架，将原问题转化为无约束优化，提升轨迹生成的可行性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.20796](https://arxiv.org/pdf/2605.20796)**

> **作者:** Yetong Zhang; Frank Dellaert
>
> **摘要:** We introduce a manifold-based framework for addressing optimization problems with equality and inequality constraints found in robotics. Our approach transforms the original problem into an unconstrained optimization problem directly on the constrained state space. To achieve this, we introduce ``constraint manifolds with corners" to represent the state space satisfying mixed nonlinear equality and inequality constraints. We further extend manifold optimization algorithms to operate on this new topological structure. We demonstrate the power and robustness of our framework in the context of a large-scale kinodynamic planning problem, successfully generating dynamically feasible trajectories where standard methods fail.
>
---
#### [new 040] Conflict-Aware Active Perception and Control in 3D Gaussian Splatting Fields via Control Barrier Functions
- **分类: cs.RO**

- **简介: 该论文属于机器人主动感知任务，解决不确定环境中安全与信息获取的冲突问题。提出基于CBF的安全框架和风险感知的信息增益方法，实现安全导航与有效观测。**

- **链接: [https://arxiv.org/pdf/2605.20566](https://arxiv.org/pdf/2605.20566)**

> **作者:** Amirhossein Mollaei Khass; Athanasios Cosse; Vivek Pandey; Nader Motee
>
> **备注:** Project website: this https URL
>
> **摘要:** Active perception in uncertain environments requires robots to navigate safely while acquiring informative observations to reduce map uncertainty. These objectives inherently conflict, as informative viewpoints often lie near uncertain regions with higher collision risk. To address this challenge, we develop a conflict-aware active perception and control framework for robotic systems operating in environments represented by 3D Gaussian Splatting (3DGS). Safety is enforced using a Control Barrier Function (CBF) derived from an Average Value-at-Risk AV@R collision-risk metric that accounts for geometric uncertainty and guarantees forward invariance of a safe set. To improve perception, we propose a risk-aware Expected Information Gain (EIG) formulation for selecting the next-best-view and introduce perception barrier functions that align the camera orientation with the local information-ascent direction. To obtain a tractable formulation for these conflicting safety and perception objectives, we propose a unified safety-critical, perception-aware quadratic program that enforces safety as a hard constraint while relaxing perception constraints through slack variables. Simulation results demonstrate that the proposed method improves both safety and information acquisition compared to existing 3DGS-based approaches.
>
---
#### [new 041] Mobile UMI: Cross-View Diffusion Policy with Decoupled Kinematics for Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文针对移动操作中的模仿学习任务，解决动作标签污染和执行延迟问题。通过双摄像头、空间锚定和异步执行器实现解耦运动控制，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2605.20894](https://arxiv.org/pdf/2605.20894)**

> **作者:** Haoran Huang; Haonan Dong; Huixu Dong
>
> **摘要:** Mobile imitation learning on portable demonstration interfaces faces two coupled bottlenecks: locomotion-contaminated action labels and inference-induced execution latency on a continuously moving base. Recent wrist-mounted interfaces lower the cost of tabletop data collection, yet a single wrist view does not capture the global context required for base navigation. Adding a body-mounted camera entangles human walking with hand motion. Meanwhile, generative policies introduce hundreds of milliseconds of inference latency, during which the base advances past predicted waypoints, forcing backward corrections at action splices. This paper presents Mobile UMI, a hardware-free demonstration framework that addresses both gaps through three components. First, a dual-camera capture system records chest-centric global context and wrist-centric local interaction without any robot present. Second, a one-shot ChArUco-based spatial anchor unifies the chest and hand visual-inertial frames; the hand pose is then re-expressed relative to the chest to extract decoupled SE(3) manipulation and SE(2) base trajectories. Third, an asynchronous receding-horizon executor performs online state matching: each generated action chunk is realigned with the current physical pose so that expired waypoints are discarded before execution. The full system is evaluated on four long-horizon household tasks, achieving an average success rate of 83.8% over 100 trials per task. Controlled comparisons against ACT and Diffusion Policy show that the chest-relative label alone closes much of the gap; online state matching closes the remainder. These results indicate that, for mobile imitation learning under the tested conditions, explicit kinematic factorization combined with state-level latency alignment provides an effective solution without requiring architectural changes to the underlying policy class.
>
---
#### [new 042] EllipseLIO: Adaptive LiDAR Inertial Odometry with an Ellipsoid Representation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决LIO在异构环境与传感器下的适应性问题。通过自适应的LiDAR扫描过滤与配准方法，提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.21150](https://arxiv.org/pdf/2605.21150)**

> **作者:** Rowan Border; Margarita Chli
>
> **备注:** 8 pages, 6 figures, 2 tables
>
> **摘要:** LiDAR Inertial Odometry (LIO) is a critical component for many mobile robots that need to navigate without relying on external positioning (e.g., GPS). Platforms that operate autonomously in different environments and with heterogeneous LiDAR sensors require a LIO approach that can adapt to these different scenarios without human intervention. Existing LIO approaches can typically provide reliable and accurate odometry in scenarios with similar environments and sensors when suitably tuned. However, many approaches struggle to retain robust odometry across heterogeneous environments and sensors while using a consistent configuration. This paper presents EllipseLIO, a real-time LIO approach that generalises between scenarios by using methods for LiDAR scan filtering and registration that adapt to the sensor capabilities and environment without requiring scenario-specific tuning. Experiments with EllipseLIO and state-of-the-art LIO approaches on five datasets with diverse and challenging scenarios demonstrate that EllipseLIO is the best-performing approach overall. It achieves a 38% lower odometry error on average than the second-best approach and is the only approach that does not diverge in any experiment. An open-source version of EllipseLIO will be available at this http URL.
>
---
#### [new 043] WiXus: A Wheeled-Legged Robot with Wire-Driven Environmental Utilizing to Integrate Mobility and Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人设计任务，旨在解决 wheeled-legged 机器人功能单一的问题。通过引入线驱动环境利用机制，实现移动与操作的结合。**

- **链接: [https://arxiv.org/pdf/2605.20932](https://arxiv.org/pdf/2605.20932)**

> **作者:** Shintaro Inoue; Kento Kawaharazuka; Temma Suzuki; Sota Yuzaki; Kei Okada
>
> **备注:** Accepted at ICRA2026, website - this https URL, YouTube - this https URL
>
> **摘要:** Wheeled-legged robots, which have wheels at their feet and achieve high mobility by coordinating wheel drive and leg drive, have been developed. These robots have been developed purely as platforms specialized for locomotion. Therefore, they do not have a means to repurpose their legs for roles other than locomotion, such as object manipulation or tool utilization. In this paper, we address the problem of how to draw out the potential task-execution capability of the legs by freeing them from the roles of locomotion through external body support. To this end, we propose and develop a new robot, WiXus, which fuses a wheeled-legged mechanism with a wire-driven mechanism that utilizes the external environment. The developed WiXus demonstrates not only planar locomotion with wheeled-legged drive, but also three-dimensional mobility such as cliff climbing by coordinating wire-driven and wheeled-legged actuation. Furthermore, by suspending the body with wire-driven actuation, WiXus successfully repurpose its legs as arms to perform object manipulation, (e.g., rescuing a dog (stuffed animal)), and tool utilization (e.g., harvesting an apple (mockup) with loppers). This study demonstrates that the approach of utilizing the environment with wire-driven actuation is a new design principle that extends the operational domain of wheeled-legged robots.
>
---
#### [new 044] Demo-JEPA: Joint-Embedding Predictive Architecture for One-shot Cross-Embodiment Imitation
- **分类: cs.RO**

- **简介: 该论文提出Demo-JEPA，解决跨形态机器人模仿学习问题。通过解耦演示意图与执行方式，实现不同机器人间的灵活模仿。**

- **链接: [https://arxiv.org/pdf/2605.20811](https://arxiv.org/pdf/2605.20811)**

> **作者:** Jingyang He; Guangrun Li; Jieyu Zhang; Chengkai Hou; Zhengping Che; Shanghang Zhang
>
> **摘要:** Robotic imitation learning is often treated as reproducing demonstrated actions, but actions are inherently embodiment-specific. When demonstrations come from humans or robots with different morphology, kinematics, or action spaces, this action-centric view requires shared action spaces, heuristic retargeting, or large-scale multi-embodiment co-training. We instead view demonstrations as implicit specifications of future goals: the target agent should infer what state the demonstrator is trying to realize, rather than how the demonstrator executes it. We propose Demo-JEPA, a cross-embodiment imitation framework that decouples demonstration intent from embodiment-specific execution. Built on a JEPA-based world model, Demo-JEPA translates source visual demonstrations into target-compatible future latent trajectories in a shared predictive representation space. The target agent then uses these latent trajectories as subgoals and realizes them through planning under its own learned forward dynamics. Because Demo-JEPA avoids action-level correspondence and requires only visual demonstrations plus the target agent's own interaction experience, it supports flexible imitation across heterogeneous embodiments. Experiments on RLBench and real-world manipulation tasks show that Demo-JEPA matches specialized in-domain planners and generalizes to unseen tasks and embodiment configurations where prior methods fail.
>
---
#### [new 045] LiteViLNet: Lightweight Vision-LiDAR Fusion Network for Efficient Road Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于道路分割任务，旨在解决多模态融合在资源受限设备上的实时性与精度平衡问题。提出LiteViLNet，通过轻量编码器和特征融合模块实现高效准确的分割。**

- **链接: [https://arxiv.org/pdf/2605.21007](https://arxiv.org/pdf/2605.21007)**

> **作者:** Daojie Peng; Bingtao Wang; Fulong Ma; Liang Zhang; Jun Ma
>
> **摘要:** Road segmentation is a fundamental perception task for autonomous driving and intelligent robotic systems, requiring both high accuracy and real-time inference, especially for deployment on resource-constrained edge devices. Existing multi-modal road segmentation methods often rely on heavy transformer-based encoders to achieve state-of-the-art performance, but their enormous computational cost prohibits real-time deployment on embedded platforms. To address this dilemma, we propose \textbf{LiteViLNet}, a lightweight multi-modal network that fuses RGB texture information and LiDAR geometric information for efficient road segmentation. Specifically, we design a dual-stream lightweight encoder and depth-wise separable convolutions to extract hierarchical features from both modalities with minimal parameters. We further propose a Multi-Scale Feature Fusion Module (MSFM) to facilitate cross-modal interaction at different levels, and a large-kernel-bridge module to capture long-range dependencies with linear complexity. Extensive experiments on the KITTI Road dataset and real-world applications demonstrate that LiteViLNet achieves a promising balance between accuracy and efficiency. Notably, with only 14.04M parameters, our model attains a 96.36\% MaxF score, ranking the best among all CNN-based methods and being comparable to larger transformer-based models, and runs at 163.79 FPS in model-only inference on RTX 4060 Ti (22.18 FPS on Jetson Orin NX). It outperforms numerous heavy-weight methods in inference speed while maintaining highly competitive accuracy, fully validating the potential of LiteViLNet for real-time embedded deployment in autonomous driving and intelligent robotics.
>
---
#### [new 046] Design for Manufacturing: A Manufacturability Knowledge-Integrated Reinforcement Learning Framework for Free-Form Pipe Routing in Aeroengines
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于航空发动机管路设计任务，解决传统管路设计与制造脱节的问题。通过集成制造知识的强化学习框架，优化自由曲面管路路径，提升可制造性与效率。**

- **链接: [https://arxiv.org/pdf/2605.20644](https://arxiv.org/pdf/2605.20644)**

> **作者:** Caicheng Wang; Zili Wang; Shuyou Zhang; Yongzhe Xiang; Zheyi Li; Liangyou Li; Jianrong Tan
>
> **摘要:** Design for manufacturing plays a critical role in advanced aeroengine development, where complex components necessitate careful consideration of manufacturability. However, current practices in pipe routing remain largely decoupled from down-stream manufacturing, leading to labor-intensive, trial-and-error iterations to achieve manufacturable designs. To address this problem, this study proposes the Frenet-based pipe routing optimization (FPRO) framework, a manufacturability knowledge-integrated reinforcement learning approach for free-form pipe design in aeroengines. FPRO formulates the routing problem as a boundary value problem in the Frenet frame. In this framework, the pipe path is represented by curvature and torsion profiles, which are generated using cubic Hermite interpolation. To integrate design and manufacturing, domain-specific manufacturing knowledge is embedded as constraints on the permissible ranges of curvature and torsion. The path optimization is performed using the proximal policy optimization algorithm with stochastic exploration and a stage-guided reward mechanism. A unified mapping formulation then translates the optimized path into motion trajectories for the bending die, enabling direct fabrication on a six-axis free-bending machine. Experimental results demonstrate that FPRO consistently generates collision-free, manufacturable paths with smoother geometric profiles compared to Cartesian-based methods. It also achieves faster convergence and superior performance in terminal alignment, path length, obstacle avoidance, and manufacturability compared to state-of-the-art reinforcement learning baselines. Real-world validation confirms the close geometric correspondence between the manufactured pipe and its digital design, validating the practical feasibility of FPRO.
>
---
#### [new 047] Faster or Stronger: Towards Flexible Visual Place Recognition via Weighted Aggregation and Token Pruning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对视觉位置识别任务，解决特征聚合不均衡和计算效率低的问题，提出WeiAD和WeiToP方法，提升识别精度并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2605.20551](https://arxiv.org/pdf/2605.20551)**

> **作者:** Zichao Zeng; June Moh Goo; Junwei Zheng; Weijia Fan; Jiaming Zhang; Rainer Stiefelhagen; Jan Boehm
>
> **摘要:** Visual Place Recognition (VPR) aims to match a query image to reference images of the same place in a large-scale database. Recent state-of-the-art methods employ Vision Transformers (ViTs) as backbone foundation models to extract patch-level features that are robust to viewpoint, illumination, and seasonal variations, which are then aggregated into a compact global descriptor for retrieval. Most existing aggregation methods uniformly pool patch tokens into learned clusters, despite the fact that different clusters often encode distinct spatial or semantic patterns and contribute unequally to VPR performance. To address this limitation, we propose Weighted Aggregated Descriptor (WeiAD), which assigns weights to clusters during aggregation, producing more discriminative global representations. Beyond accuracy, retrieval latency is a critical concern for large-scale deployments and resource-constrained edge devices. Prior work mainly reduces latency by compressing global descriptors, while overlooking the cost of feature extraction, an issue exacerbated by ViT-based backbones. We therefore introduce WeiToP, a VPR-oriented token pruning framework that reduces feature extraction cost via self-distillation, where aggregation-induced token importance supervises a lightweight pruning module attached to an early transformer layer, enabling inference-time token pruning. After a single joint training phase, WeiToP enables plug-and-play token pruning at inference time, allowing flexible and on-demand control over the accuracy-efficiency trade-off without additional training. Moreover, WeiToP outperforms existing token pruning methods adapted from general vision tasks.
>
---
#### [new 048] Multi-Agent Reinforcement Learning for Safe Autonomous Driving Under Pedestrian Behavioral Uncertainty
- **分类: cs.LG; cs.AI; cs.HC; cs.RO**

- **简介: 该论文属于自动驾驶安全评估任务，解决行人行为不确定性带来的挑战。通过多智能体强化学习，协同训练自动驾驶车辆与行人，提升交互真实性并降低碰撞率。**

- **链接: [https://arxiv.org/pdf/2605.20255](https://arxiv.org/pdf/2605.20255)**

> **作者:** Prakash Aryan; Kaushik Raghupathruni; Timo Kehrer; Sebastiano Panichella
>
> **备注:** Submitted to ICRA 2026 Workshop "8th Workshop on Long-term Human Motion Prediction"
>
> **摘要:** Simulation-based testing of self-driving cars (SDCs) typically relies on scripted or simplified pedestrian models that do not capture the heterogeneity and uncertainty of real human crossing behavior. This limits the realism of safety assessments, especially in scenarios involving jaywalking, which is governed by latent personality traits that the vehicle cannot observe. We hypothesize that jointly training pedestrians and the SDC with multi-agent reinforcement learning (MARL) produces more realistic interaction scenarios than training the SDC against fixed pedestrian policies, and that the resulting behavior gap between predictable and unpredictable crossings can be measured directly from trajectories. This paper describes a MARL environment in which an SDC and 12 pedestrians are co-trained using Multi-Agent Proximal Policy Optimization (MAPPO). Pedestrian locomotion follows scripted Dijkstra pathfinding, while an RL policy controls high-level go/wait decisions. Jaywalking probability depends on a per-pedestrian personality trait sampled at episode start and hidden from the SDC. In 500-episode evaluations, the co-trained SDC reached 78% of goals with a 14% collision rate, compared to 35% goals and 33% collisions for the best rule-based baseline. A speed differential metric shows that the SDC traveled 2.65 m/s faster near jaywalkers than near crosswalk users at close range (0-3 m), indicating that jaywalking encounters were not anticipated. Jaywalking accounted for 13% of crossing events but was associated with 62% of collisions. Co-training with MARL pedestrians reduced collisions by 30% relative to single-agent RL, as pedestrians learned to wait when the SDC approached at speed.
>
---
#### [new 049] Mechanistic Interpretability for Learning Assurance of a Vision-Based Landing System
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文属于航空视觉系统安全验证任务，旨在解决神经网络缺乏可解释性证据的问题。通过分离内容与风格表示，构建运行时保障方法，提供符合EASA要求的证据。**

- **链接: [https://arxiv.org/pdf/2605.20607](https://arxiv.org/pdf/2605.20607)**

> **作者:** Romeo Valentin; Olivia Beyer Bruvik; Marc R. Schlichting; Mykel J. Kochenderfer
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** EASA's learning-assurance guidance requires data-driven aviation systems to build and monitor their own situation representation, yet for neural networks the technical means to provide such evidence remain an open problem. We address this gap for a vision-based aircraft landing system: we propose that a minimally assurable model must at least be shown to separate content from style in its own situation representation. Showing that the model's predictions then rely largely on the contentful representation components leads to a concrete assurance path. To demonstrate this assurance path on a concrete model we train a vision transformer model for runway keypoint regression on the LARDv2 dataset. The model, which acts as the subject for our assurance demonstration, produces per-patch embeddings that we decompose into interpretable atoms via K-SVD sparse dictionary learning. A qualitative visualization confirms that contentful atoms track task-relevant runway structure and stylistic atoms track domain-specific appearance, and the regression head is shown to place almost all of its linear weight on contentful atoms. We further build on the content/style separation and define out-of-model-scope (OOMS) detection, a novel runtime assurance approach directly monitoring the model's situation representation. OOMS monitoring is complementary to operational design domain and output-space out-of-distribution monitoring and addresses concrete requirements of the recent EASA guidance. By directly analyzing a model's situation representation both at test time and runtime, this work delivers the first concrete piece of the representation-level evidence that EASA learning-assurance guidance demands, and points to mechanistic interpretability as a practical building block of future aviation safety cases.
>
---
#### [new 050] Multi-Week, In-Class Deployments of Telepresence Robots With Four Homebound K-12 Students: Benefits, Challenges, and Recommendations
- **分类: cs.HC; cs.RO**

- **简介: 论文研究了通过远程机器人让学生参与课堂的可行性，探讨其优势与挑战。任务是评估 telepresence robots 在K-12教育中的应用，解决学生缺课带来的社交与学习问题。工作包括多周部署和案例分析。**

- **链接: [https://arxiv.org/pdf/2605.20431](https://arxiv.org/pdf/2605.20431)**

> **作者:** Matthew Rueben; Rhianna Lee; Thomas R. Groechel; Hengzhi Chen; Haemi Lee; Gisele Ragusa; Maja J. Matarić
>
> **摘要:** Missing significant amounts of school during K-12 education is known to put students' cognitive and social development at risk. Alternatives such as home instruction and online learning are common, but lack sufficient interaction with peers and teachers in the classroom. Mobile remote presence systems, or telepresence robots, are promising for homebound students because they provide embodiment and mobility in addition to the real-time participation offered by video conferencing technologies. Research is needed, however, for telepresence robots to meet the complex needs of homebound students participating remotely in the K-12 classroom context. We present findings from four multi-week deployments with homebound K-12 students attending classes via telepresence robots. The homebound students' experiences were documented in a total of 15 interviews and analyzed qualitatively as case studies. The homebound student participants and their deployment contexts differed from one another along multiple dimensions, and while some benefits of mobile remote attendance were enjoyed by all participants, each participant also experienced unique benefits. Some challenges with hearing, seeing, and moving the robot around the classroom warranted improvements to the design of the telepresence system. Other challenges suggested priorities for managing a classroom deployment, such as ensuring that the remote student is included in classroom activities, accountable to the teacher, and treated with respect by classmates. Based on insights from the study, we make recommendations for real-world deployment procedures in similar contexts.
>
---
#### [new 051] Hyper-V2X: Hypernetworks for Estimating Epistemic and Aleatoric Uncertainty in Cooperative Bird's-Eye-View Semantic Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的协同感知任务，旨在解决V2X环境下语义分割的不确定性问题。提出Hyper-V2X框架，同时估计认知和随机不确定性，提升感知可靠性。**

- **链接: [https://arxiv.org/pdf/2605.21309](https://arxiv.org/pdf/2605.21309)**

> **作者:** Abhishek Dinkar Jagtap; Sanath Tiptur Sadashivaiah; Andreas Festag
>
> **备注:** Accepted for IEEE Intelligent Vehicle Symposium (IV) 2026
>
> **摘要:** Cooperative perception enabled by Vehicle-to-Everything (V2X) communication enhances autonomous driving safety by creating a unified environmental representation through shared sensory data. While recent works have advanced multi-agent fusion for improved perception, uncertainty quantification in such cooperative frameworks remains largely unexplored. This paper introduces Hyper-V2X, a hypernetwork-based framework for estimating both epistemic and aleatoric uncertainties in V2X-based perception. Specifically, we propose a partial weight generation scheme and V2X context embedding module that conditions a Bayesian hypernetwork on fused multi-agent features to generate weight distributions for stochastic Bird's-Eye-View (BEV) segmentation. Unlike existing deterministic BEV models, Hyper-V2X enables efficient uncertainty estimation with little computation overhead. Our approach is architecture-agnostic, and can be seamlessly integrating with modern cooperative backbones such as CoBEVT. Experiments on the OPV2V benchmark demonstrate that Hyper-V2X provides accurate, well-calibrated uncertainty estimates and improves overall perception reliability. Our code and benchmark are publicly available under an open-source license: this https URL
>
---
#### [new 052] STELLAR: Scaling 3D Perception Large Models for Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶感知任务，解决传感器数据融合与3D空间理解问题。通过扩展输入模态并训练大规模模型，提升感知性能。**

- **链接: [https://arxiv.org/pdf/2605.20390](https://arxiv.org/pdf/2605.20390)**

> **作者:** Yingwei Li; Xin Huang; Yang Liu; Yang Fu; Alex Zihao Zhu; Chen Song; Junwen Yao; Anant Subramanian; Hao Xiang; Weijing Shi; Yuliang Zou; Tom Hoddes; Zhaoqi Leng; Govind Thattai; Dragomir Anguelov; Mingxing Tan
>
> **摘要:** Model scaling has demonstrated remarkable success through large-scale training on diverse datasets. It remains an open question whether the same paradigm would apply to autonomous driving perception systems due to unique challenges, such as fusing heterogeneous sensor data and the need for sophisticated 3D spatial understanding. To bridge this gap, we present a comprehensive study on systematically analyzing the impact of scale on these systems. We develop our STELLAR model based on Sparse Window Transformer, by extending the input modalities to include LiDAR, radar, camera, and map prior. We train the model on a large-scale dataset of 50 million driving examples with up to 500 million parameters. Our large-scale experiments reveal empirical scaling trends that connect model performance to model size, data, and compute. The resulting model establishes a new state-of-the-art on the Waymo Open Dataset challenge, outperforming prior arts by a large margin. Our work demonstrates that large-scale training is a highly promising path for advancing the capabilities of perception models for autonomous driving.
>
---
#### [new 053] Time-To-Reach Separation and Safety Filtering for Safe, Fair, and Efficient Multi-Agent Coordination
- **分类: eess.SY; cs.MA; cs.RO**

- **简介: 该论文属于多智能体协调任务，旨在解决城市空域中多飞行器安全、公平、高效融合的问题。通过TTR指标实现优先级分配和安全过滤，提升系统性能。**

- **链接: [https://arxiv.org/pdf/2605.20625](https://arxiv.org/pdf/2605.20625)**

> **作者:** Matthew Low; Jasmine Jerry Aloor; Victoria Marie Tuck; Pierluigi Nuzzo; Jason J. Choi
>
> **备注:** 9 pages, 3 figures. Extended version (including appendix) of a paper submitted to the 65th IEEE Conf. on Decision and Control (2026)
>
> **摘要:** Advanced Air Mobility (AAM) operations are expected to significantly increase aerial traffic in urban airspace, requiring autonomous traffic management systems to ensure collision-free operations in highly congested environments. In this paper, we propose a multi-agent coordination framework that uses minimum time-to-reach (TTR) as a unifying metric for priority assignment, temporal separation, and safety filtering. We focus on the problem of coordinating multiple aerial vehicles merging into an air corridor while maintaining safe separation between vehicles. Vehicles are assigned arrival-consistent priority based on TTR, and target TTR values are used to enforce temporal spacing that induces spatial separation. A priority-consistent safety filtering layer based on Hamilton-Jacobi reachability value functions ensures collision avoidance while minimally modifying the reference guidance. Simulation results in a highly congested corridor merging scenario show that the proposed method improves safety, fairness, and efficiency compared to time-optimal guidance and priority-agnostic safety filtering.
>
---
#### [new 054] Mechanisms of Misgeneralization in Physical Sequence Modeling
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 论文研究物理序列建模中的错误泛化问题，属于生成模型领域。解决生成轨迹分布与预期物理量不一致的问题，通过分析误差传播机制并提出干预策略。**

- **链接: [https://arxiv.org/pdf/2605.20299](https://arxiv.org/pdf/2605.20299)**

> **作者:** Kento Nishi; Raphael Tang; Karun Kumar; Core Francisco Park; Hidenori Tanaka
>
> **备注:** Preprint. this http URL
>
> **摘要:** Generative sequence models are often trained to plan motion in physical domains, from robotics to mechanical simulations. When constructing a dataset to train such a model, engineers may curate demonstrations to specify how trajectories should be distributed over a physical quantity like travel distance or mechanical energy. For example, a roboticist building a maze navigation agent might choose demonstrations whose travel distances cover a fixed range uniformly, hoping to constrain the agent's expected power usage. We find that standard deep learning can violate this intent: each generated trajectory can seem plausible on its own, but the aggregate distribution over the physical quantity is wrong. We call this failure physical misgeneralization, and develop an account of its mechanism. Using controlled synthetic tasks, we show that physical misgeneralization arises when local errors typical of the model class propagate through the physical measurement to shift the recovered distribution. We estimate these errors with a data deviation kernel, and we use it to predict which physical quantities gain or lose mass in both our synthetic and more applied maze navigation and double-pendulum motion tasks. Finally, our mechanistic interpretation helps identify which mitigation strategies are structurally promising, and we use it to propose a kernel-informed intervention.
>
---
#### [new 055] Validating Navmesh using Geometry: Voxel-Based Analysis with Prioritized Exploration
- **分类: cs.SE; cs.RO**

- **简介: 该论文属于游戏开发中的导航验证任务，旨在解决Navmesh与实际环境不一致的问题。通过几何驱动的体素分析，验证Navmesh正确性，并利用强化学习优化检测效率。**

- **链接: [https://arxiv.org/pdf/2605.21397](https://arxiv.org/pdf/2605.21397)**

> **作者:** Ramesh Raghavan; Ojas Sharma; Sebastien Larrue; Alan Isaac Kunder; Aakash Sai; Rishi Mathur
>
> **摘要:** Navigation mesh (Navmesh) inconsistencies affect the player experience by directly impacting the navigation systems used by non-playable characters (NPCs) in game environments. While navmeshes are generated from world geometry using well-established algorithms, environments change throughout development as terrain is adjusted and assets are moved or replaced, resulting in mismatches between the navmesh and the actual environment. Existing automated approaches attempt to detect navigation issues using exploration agents and reinforcement learning techniques. However, since these methods rely on the navigation data itself or evaluate navigation behavior indirectly, they do not explicitly verify whether the navigation representation reflects the walkable space defined by underlying geometry. This paper presents a framework for validating navigation meshes through an independent, geometry-driven analysis of navmesh correctness. The approach reconstructs walkable space directly from environment geometry using a voxel-based representation, followed by constraint-aware traversal and connectivity evaluation. Validation is formulated as a prioritized search problem over the voxel space, where reinforcement learning guides sampling toward regions more likely to exhibit inconsistencies. At each sampled location, reachability derived from the voxel representation is compared against reachability obtained from the navmesh via engine-level queries. Experiments across multiple large-scale open-world game environments show that the approach consistently lowers exploration effort while maintaining similar defect detection coverage. The framework runs offline within the game engine and can be integrated into automated quality assurance pipelines. Since the method relies on geometry, it can be adapted across game engines with minimal changes, making it suitable for production deployment.
>
---
#### [new 056] NaP-Control: Navigating Diffusion Prior for Versatile and Fast Character Control
- **分类: cs.GR; cs.LG; cs.RO**

- **简介: 该论文属于物理动画中的角色控制任务，旨在解决精确、快速且通用的全身动作控制问题。提出NaP-Control方法，通过强化学习优化扩散先验，提升控制效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.20209](https://arxiv.org/pdf/2605.20209)**

> **作者:** Chia-Wen Chen; Yan Wu; Korrawe Karunratanakul; Siyu Tang
>
> **摘要:** Achieving precise, versatile whole-body character control in physics-based animation remains challenging. Recent diffusion-based policies generate rich and expressive motions but typically rely on gradient-based test-time guidance to satisfy task objectives, which is slow and can reduce robustness. We introduce NaP-Control (Navigating Diffusion Prior for Versatile and Fast Character Control), abbreviated as NaP. Our method uses reinforcement learning to manipulate the latent noise of a task-agnostic diffusion policy prior, steering it toward task-specific behaviors for fast, robust control with high motion fidelity. In contrast to methods that rely solely on offline training, NaP interacts with the environment during training to correct motions and optimize task rewards, improving success rates and enabling adaptation to challenging scenarios. By directly predicting task-optimized diffusion noise, NaP eliminates iterative guidance during denoising and enables efficient inference. Experiments show that NaP attains higher success rates and faster inference while preserving natural motion across diverse tasks.
>
---
#### [new 057] Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards
- **分类: cs.AI; cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于生成模型控制任务，解决多约束下生成偏离数据流形的问题。提出Conflict-Aware Additive Guidance方法，动态检测并解决梯度冲突，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.20758](https://arxiv.org/pdf/2605.20758)**

> **作者:** Xuehui Yu; Fucheng Cai; Meiyi Wang; Xiaopeng Fan; Harold Soh
>
> **备注:** Forty-Third International Conference on Machine Learning (ICML 2026)
>
> **摘要:** Inference-time guided sampling steers state-of-the-art diffusion and flow models without fine-tuning by interpreting the generation process as a controllable trajectory. This provides a simple and flexible way to inject external constraints (e.g., cost functions or pre-trained verifiers) for controlled generation. However, existing methods often fail when composing multiple constraints simultaneously, which leads to deviations from the true data manifold. In this work, we identify root causes of this off-manifold drift and find that the approximation error scales severely with gradient misalignment. Building on these findings, we propose Conflict-Aware Additive Guidance ($g^\text{car}$), a lightweight and learnable method, which actively rectifies off-manifold drift by dynamically detecting and resolving gradient conflicts. We validate $g^\text{car}$ across diverse domains, ranging from synthetic datasets and image editing to generative decision-making for planning and control. Our results demonstrate that $g^\text{car}$ effectively rectifies off-manifold drift, surpassing baselines in generation fidelity while using light compute. Code is available at this https URL.
>
---
#### [new 058] Closed Loop Dynamic Driving Data Mixture for Real-Synthetic Co-Training
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶领域，解决真实与合成数据联合训练中的数据混合问题。提出AutoScale框架，通过动态优化提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.21372](https://arxiv.org/pdf/2605.21372)**

> **作者:** Hongzhi Ruan; Pei Liu; Weiliang Ma; Zhengning Li; Xueyang Zhang; Jun Ma; Dan Xu; Kun Zhan
>
> **摘要:** Data scaling is fundamental to modern deep learning, and grows increasingly critical as autonomous driving shifts to end-to-end learning. Real-world driving data is expensive to annotate and scene-biased, making real-synthetic co-training with near-infinite synthetic data a promising direction. However, naively incorporating all available synthetic data is inefficient and leads to distribution shifts, and optimizing data mixture under practical training budgets remains a critical yet under-explored problem. In this sense, we claim that the mixture of training data requires clear guidance in terms of scene types and quantities. Particularly in this work, we conceptualize the data mixture approximately as a dynamic optimization process that iteratively adjusts the training data mixture to maximize model performance, guided by closed-loop evaluation feedback, and propose AutoScale, a fully automated closed-loop data engine unifying scene representation, data mixture optimization and retrieval, as well as model training and evaluation. Specifically, we propose Graph Regularized AutoEncoder (Graph-RAE) for driving scene representations, introduce Cluster-aware Gradient Ascent (Cluster-GA) for cluster-wise importance estimation and reweighting, and perform cluster-guided vector retrieval to select high-value samples. Experiments on NavSim demonstrate that AutoScale outperforms vanilla co-training and cross-domain baselines, achieving better performance with fewer synthetic samples under constrained budgets.
>
---
#### [new 059] VSCD: Video-based Scene Change Detection in Unaligned Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出VSCD任务，解决非对齐场景下的视频变化检测问题。通过多参考模型和局部匹配，生成像素级变化掩码，提升真实场景下的检测效果。**

- **链接: [https://arxiv.org/pdf/2605.20821](https://arxiv.org/pdf/2605.20821)**

> **作者:** Jiae Yoon; Ue-Hwan Kim
>
> **备注:** 18 pages, 7 figures. Accepted to the 43rd International Conference on Machine Learning (ICML 2026)
>
> **摘要:** Detecting what has changed in an environment is essential for long-term autonomy, yet most change detection settings assume fixed viewpoints, mild misalignment, or only a few changed objects. We introduce Video-based Scene Change Detection (VSCD), which predicts a pixel-wise change mask for each query frame, given a reference and a query RGB video of the same indoor space recorded at different times under unconstrained camera motion. The two videos are not temporally synchronized, and many object instances may appear or disappear. To study this setting, we build a large-scale benchmark with over 1.1 million frames annotated with pixel-accurate change masks, together with a real-world test set for evaluating transfer beyond simulation. We propose a query-centric multi-reference model that learns temporal matching implicitly from change-mask supervision, aligns candidate reference features to the query via local patch correspondence, and fuses per-candidate change features using frame-level and patch-level confidence before decoding a high-resolution mask once per frame. Our approach achieves state-of-the-art performance against strong image- and video-based baselines, and we validate its real-world impact by deploying it on a mobile robot for two downstream applications -- visual surveillance and object incremental learning.
>
---
#### [new 060] Comparative Analysis of Military Detection Using Drone Imagery Across Multiple Visual Spectrums
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于目标检测任务，旨在提升无人机在不同视觉条件下（如夜间、热成像等）的军事目标识别能力。通过构建多光谱数据集并使用YOLOv11-small模型进行训练，增强无人机作战的可靠性。**

- **链接: [https://arxiv.org/pdf/2605.21157](https://arxiv.org/pdf/2605.21157)**

> **作者:** Sourov Roy Shuvo; Prajwal Panth; Rajesh Chowdhury; Sorup Chakraborty; Sudip Chakrabarty; Prasant Kumar Pattnaik
>
> **备注:** 6 pages, 7 figures. Accepted at the 16th International Conference on Computing, Communication and Networking Technologies (ICCCNT), July 6-11, 2025, IIT Indore. Proceedings pending publication
>
> **摘要:** In modern warfare, drones are becoming an essential part of intelligence gathering and carrying out precise attacks in different kinds of hostile environments. Their ability to operate in real-time and hostile environments from a safe distance makes them invaluable for surveillance and military operations. The KIIT-MiTA dataset is comprised of images of different military scenarios taken from drones, and these provide a foundation for detecting military objects, but it does not take into account the various types of real-world scenarios. With that in mind, to evaluate how the models are performing under varying conditions, four different types of datasets are created: Gray Scale, Thermal Vision, Night Vision, and Obscura Vision. These simulate the real-world environments such as low visibility, heat-based imagery, and nighttime conditions. The YOLOv11-small model is trained and used to detect objects across diverse settings. This research boosts the performance and reliability of drone-based operations by contributing to the development of advanced detection systems in both defensive and offensive missions.
>
---
#### [new 061] Grounding Driving VLA via Inverse Kinematics
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉导航任务，解决现有驾驶VLA忽视视觉信息的问题。通过引入逆运动学思路，设计新模型提升视觉接地和轨迹规划性能。**

- **链接: [https://arxiv.org/pdf/2605.21061](https://arxiv.org/pdf/2605.21061)**

> **作者:** Junsung Park; Hyunjung Shim
>
> **摘要:** Existing Driving VLAs predict trajectories while largely ignoring their visual tokens -- a phenomenon we trace not to insufficient training but to a structurally ill-posed task formulation. We show that trajectory recovery, when viewed through the lens of inverse kinematics, requires both a current and a future visual state as boundary conditions; existing VLAs supply only the former, which encourages the model to shortcut through ego status and text commands alone. To address this, we re-design Driving VLA in the style of an inverse kinematics solver. First, a next visual state prediction objective that requires the LLM to predict the future visual scene provides dense visual supervision and suppresses shortcut paths. Second, a separate Inverse Kinematics Network (a cross-attention-based conditional diffusion model) that takes only the current and future visual states as input is designed to suppress reliance on ego status and textual shortcuts during trajectory decoding. With this simple prescription alone, our 0.5B-scale model recovers visual grounding and reaches trajectory planning performance comparable to 7B--8B VLAs more than an order of magnitude larger, on both the closed-loop NAVSIM-v2 and the nuScenes benchmarks. Extensive analysis further shows that this improvement stems from a recovered ability to exploit visual features, with the effect being most pronounced in dynamic driving situations such as turning.
>
---
#### [new 062] Fully Actuated Manifold Constraint Based Output Feedback Control for Input-Constrained Uncertain Nonlinear Systems
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于控制任务，解决输入受限非线性系统的控制问题。提出一种低复杂度、无需模型的输出反馈控制器，实现有限时间内的精确控制。**

- **链接: [https://arxiv.org/pdf/2605.21439](https://arxiv.org/pdf/2605.21439)**

> **作者:** Dianrui Mu; Changchun Hua; Yafeng Li; Jiannan Chen; Rao Wei
>
> **备注:** 22 pages, 12 figures, 2 tables
>
> **摘要:** This paper presents a low-complexity, model-free, output-feedback controller for a class of unknown time-varying nonlinear systems with unknown input constraints. The controller achieves the preset control accuracy when the actuator is not saturated and maintains flexible control accuracy after actuator saturation. This result extends existing constraint control methods for linear manifolds to a more general form, including the construction of nonlinear manifolds and various types of constraints, thereby achieving preset control accuracy within finite or fixed time. Additionally, flexible control under unknown saturation is achieved through the construction of an error-driven flexible constraint. Finally, second-order and higher-order control examples and simulations are provided.
>
---
## 更新

#### [replaced 001] FUSE: A Framework for Unified State Estimation in Vehicular and Robotic SLAM Systems
- **分类: cs.RO**

- **简介: 该论文提出FUSE框架，用于车辆和机器人的统一状态估计，解决多速率传感下的状态估计问题，通过模块化设计提升系统灵活性与精度。**

- **链接: [https://arxiv.org/pdf/2605.18047](https://arxiv.org/pdf/2605.18047)**

> **作者:** Wei Wu; Honglin Chen; Wenhan Cao; Yao Lyu; Shaobing Xu; Kun Jiang; Jiangtao Li; Tao Zhang; Shengbo Eben Li
>
> **摘要:** Tightly coupled SLAM formulations under mixed-rate sensing often bind temporal processing, local geometric association, estimator formulation, and map-update policy into method-specific designs. Such binding makes it difficult to vary one design choice without re-engineering the rest of the state-estimation process. This paper presents FUSE, a framework for unified state estimation in vehicular and robotic SLAM systems. FUSE organizes the state-estimation interface around observation ingestion, propagation, update, and state query, and uses this interface to separate temporal processing, residual-ready local geometric association, estimator formulation, and map-update policy. A LiDAR--IMU instantiation is developed to examine the framework under mixed-rate sensing and directional degeneracy, where high-rate inertial propagation, LiDAR-triggered geometric update, residual screening, and degeneracy-aware correction operate through the same interface boundaries. On a 418~m loop-corridor sequence, the instantiation reports a 1.626 m end-to-end trajectory error, corresponding to a 7.9% relative error reduction compared with Faster-LIO, the lowest-error baseline on this sequence. The results support FUSE as a framework for organizing state-estimation design choices and show how the evaluated instantiation regularizes updates along weakly observable directions.
>
---
#### [replaced 002] DeformMaster: An Interactive Physics-Neural World Model for Deformable Objects from Videos
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DeformMaster，属于物理-神经世界模型任务，旨在从视频中学习可变形物体的物理动态与外观，解决真实视频中高维变形和复杂材料响应的建模问题。**

- **链接: [https://arxiv.org/pdf/2605.09586](https://arxiv.org/pdf/2605.09586)**

> **作者:** Can Li; Zhoujian Li; Ren Li; Jie Gu; Lei Lei; Jingmin Chen; Lei Sun
>
> **备注:** Project page: this https URL
>
> **摘要:** World models for deformable objects should recover not only geometry and appearance, but also underlying physical dynamics, interaction grounding, and material behavior. Learning such a model from real videos is challenging because deformable linear, planar, and volumetric objects evolve under high-dimensional deformation, noisy interactions, and complex material response. The model must therefore infer a physical state from visual observations, roll it forward under new interactions, and render the resulting dynamics with high visual fidelity. We present DeformMaster, a video-derived interactive physics-neural world model that turns real interaction videos into an online interactive model of deformable objects within a unified dynamics-and-appearance framework. DeformMaster preserves structured physical rollout while using a neural residual to compensate for unmodeled effects, grounds sparse hand motion as distributed compliant actuator for hand-continuum interaction, represents material response with spatially varying constitutive experts, and drives high-fidelity 4D appearance from the predicted physical evolution. Experiments on real-world deformable-object sequences demonstrate DeformMaster's ability to roll out future dynamics and render dynamic appearance, outperforming state-of-the-art baselines while supporting novel action rollout, material-parameter variation, and dynamic novel-view synthesis. Project page: this https URL
>
---
#### [replaced 003] Temporal Counterfactual Explanations of Behaviour Tree Decisions
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于机器人可解释性任务，旨在解决行为树决策的因果解释问题。通过构建因果模型，生成多样化的反事实解释，提升机器人决策的透明度与可信度。**

- **链接: [https://arxiv.org/pdf/2509.07674](https://arxiv.org/pdf/2509.07674)**

> **作者:** Tamlin Love; Antonio Andriella; Guillem Alenyà
>
> **备注:** 33 pages, 7 figures + 4 figures in appendices
>
> **摘要:** Explainability, in particular, the ability for robots to explain why they have made a decision or behaved in a certain way, is a critical tool in helping users understand the robots they interact and coexist with. Behaviour trees are a popular framework for controlling the decision-making of robots, and thus a natural question to ask is whether or not a system driven by a behaviour tree is capable of answering "why" questions. While explainability for behaviour tree-driven robots has seen some prior attention, no existing methods are capable of generating causal, counterfactual explanations which detail the reasons for robot decisions and behaviour. Therefore, in this work, we introduce a novel approach which automatically generates counterfactual explanations in response to contrastive "why" questions. Our method achieves this by first automatically building a causal model from the structure of the behaviour tree as well as domain knowledge about the state and individual behaviour tree nodes. The resultant causal model is then queried and searched to find a set of diverse counterfactual explanations. We demonstrate that our approach is able to correctly explain the behaviour of a wide range of behaviour tree structures and states in real time, unlike previous methods which are either unable to answer contrastive questions with causal explanations, or are not guaranteed to provide consistent and accurate explanations. By being able to answer a wide range of causal queries, our approach represents a step towards more transparent, understandable, and ultimately safe and trustworthy robotic systems.
>
---
#### [replaced 004] Before the Body Moves: Learning Anticipatory Joint Intent for Language-Conditioned Humanoid Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人形机器人控制任务，旨在解决语言指令下机器人动作的前瞻性问题。提出DAJI框架，实现语言与控制的 anticipatory joint-intent 对齐。**

- **链接: [https://arxiv.org/pdf/2605.14417](https://arxiv.org/pdf/2605.14417)**

> **作者:** Haozhe Jia; Honglei Jin; Yuan Zhang; Youcheng Fan; Shaofeng Liang; Lei Wang; Shuxu Jin; Kuimou Yu; Zinuo Zhang; Jianfei Song; Wenshuo Chen; Yutao Yue
>
> **摘要:** Natural language is an intuitive interface for humanoid robots, yet streaming whole-body control requires control representations that are executable now and anticipatory of future physical transitions. Existing language-conditioned humanoid systems typically generate kinematic references that a low-level tracker must repair reactively, or use latent/action policies whose outputs do not explicitly encode upcoming contact changes, support transfers, and balance preparation. We propose \textbf{DAJI} (\emph{Dynamics-Aligned Joint Intent}), a hierarchical framework that learns an anticipatory joint-intent interface between language generation and closed-loop control. DAJI-Act distills a future-aware teacher into a deployable diffusion action policy through student-driven rollouts, while DAJI-Flow autoregressively generates future intent chunks from language and intent history. Experiments show that DAJI achieves strong results in anticipatory latent learning, single-instruction generation, and streaming instruction following, reaching 94.42\% rollout success on HumanML3D-style generation and 0.152 subsequence FID on BABEL.
>
---
#### [replaced 005] Affordance-R1: Reinforcement Learning for Generalizable Affordance Reasoning in Multimodal Large Language Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言任务，解决机器人操作中对象可操作性识别问题。提出Affordance-R1框架，结合强化学习与思维链推理，提升模型泛化与推理能力。**

- **链接: [https://arxiv.org/pdf/2508.06206](https://arxiv.org/pdf/2508.06206)**

> **作者:** Hanqing Wang; Shaoyang Wang; Yiming Zhong; Zemin Yang; Jiamin Wang; Zhiqing Cui; Jiahao Yuan; Yifan Han; Mingyu Liu; Yuexin Ma
>
> **摘要:** Affordance grounding focuses on predicting the specific regions of objects that are associated with the actions to be performed by robots. It plays a vital role in the fields of human-robot interaction, human-object interaction, embodied manipulation, and embodied perception. Existing models often neglect the affordance shared among different objects because they lack the Chain-of-Thought(CoT) reasoning abilities, limiting their out-of-domain (OOD) generalization and explicit reasoning capabilities. To address these challenges, we propose Affordance-R1, the first unified affordance grounding framework that integrates cognitive CoT guided Group Relative Policy Optimization (GRPO) within a reinforcement learning paradigm. Specifically, we designed a sophisticated affordance function, which contains format, perception, and cognition rewards to effectively guide optimization directions. Furthermore, we constructed a high-quality affordance-centric reasoning dataset, ReasonAff, to support training. Trained exclusively via reinforcement learning with GRPO and without explicit reasoning data, Affordance-R1 achieves robust zero-shot generalization and exhibits emergent test-time reasoning capabilities. Comprehensive experiments demonstrate that our model outperforms well-established methods and exhibits open-world generalization. To the best of our knowledge, Affordance-R1 is the first to integrate GRPO-based RL with reasoning into affordance reasoning. The code of our method and our dataset is released on this https URL.
>
---
#### [replaced 006] COBALT: Crowdsourcing Robot Learning via Cloud-Based Teleoperation with Smartphones
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出COBALT平台，解决机器人学习中高质量示范数据不足的问题。通过云协作操作，实现多用户高效数据收集，提升数据质量与系统扩展性。**

- **链接: [https://arxiv.org/pdf/2605.19138](https://arxiv.org/pdf/2605.19138)**

> **作者:** Ayush Agarwal; Ansh Gandhi; Jeremy A. Collins; Omar Rayyan; Aryan Sarswat; Ranjani Koushik; Masoud Moghani; Ajay Mandlekar; Animesh Garg
>
> **摘要:** The scarcity of large-scale, high-quality demonstration data remains a bottleneck in scaling imitation learning for robotic manipulation. We present COBALT, a teleoperation platform designed to democratize robot learning at scale both in simulation and in the real world. By leveraging vectorized environments, our scalable, load-balanced infrastructure supports concurrent teleoperation by multiple users on a single GPU, yielding a significant reduction in teleoperation cost. Operators can connect from nearly anywhere on Earth using commonly available devices, including single or dual smartphones, VR headsets, 3D mice, and keyboards. An inmemory data cache and efficient video streaming keep control and rendering synchronous, sustaining dozens of concurrent users at 20 Hz with sub-100 ms end-to-end latency for up to 8 concurrent users per GPU. We also demonstrate stable operation supporting 256 simulated clients across 8 GPUs, underscoring the system's ability to scale across hardware and within individual servers. We perform a comprehensive user study showing that phone-based teleoperation performs comparably to or better than specialized hardware, enabling faster, more ergonomic data collection. To ensure data quality, COBALT logs a suite of real-time metrics to automatically filter suboptimal demonstrations. We further demonstrate that a structured user training curriculum significantly improves data collection quality. Guided by insights from our user study, we crowdsource the collection of a large-scale, high-quality pilot dataset with 7500+ demonstrations (50+ hours) collected with smartphones across nine countries over five days. We validate the dataset's quality by training state-of-the-art imitation learning algorithms. Please visit this https URL for more details.
>
---
#### [replaced 007] VLANeXt: Recipes for Building Strong VLA Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型（VLA）研究，旨在解决VLA设计不统一的问题。通过系统分析设计选择，提出12项关键发现，并构建了性能更强的VLANeXt模型。**

- **链接: [https://arxiv.org/pdf/2602.18532](https://arxiv.org/pdf/2602.18532)**

> **作者:** Xiao-Ming Wu; Bin Fan; Kang Liao; Jian-Jian Jiang; Runze Yang; Yihang Luo; Zhonghua Wu; Wei-Shi Zheng; Chen Change Loy
>
> **备注:** Accepted in ICML 2026, Project Page: this https URL
>
> **摘要:** Following the rise of large foundation models, Vision-Language-Action models (VLAs) emerged, leveraging strong visual and language understanding from Vision-Language Models for general-purpose policy learning. Yet, the current VLA landscape remains fragmented and exploratory. Although many groups have proposed their own VLA models, inconsistencies in training protocols and evaluation settings make it difficult to identify which design choices truly matter. To bring structure to this evolving space, we reexamine the VLA design space under a unified framework and evaluation setup. Starting from a simple VLA baseline similar to RT-2, which is the origin of VLA, we systematically dissect design choices along three dimensions: foundational components, perception essentials, and action modelling perspectives. From this study, we distill 12 key findings that together form a practical recipe for building strong VLA models. The outcome of this exploration is a simple yet effective model, VLANeXt. It outperforms the state-of-the-art methods on the LIBERO and LIBERO-plus benchmarks and demonstrates strong performance in real-world experiments. We release a unified and easy-to-use codebase to reproduce our findings, explore the design space, and develop new VLA variants on top of a shared foundation. The codebase is available at this https URL.
>
---
#### [replaced 008] Active Defense Against False Data Injection Attacks in Robotic Manipulators
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人安全任务，解决FDIA导致的传感器信号篡改问题，提出两种防御方法以提升机械臂抗攻击能力。**

- **链接: [https://arxiv.org/pdf/2605.17950](https://arxiv.org/pdf/2605.17950)**

> **作者:** Gabriele Gualandi; Carl Mikael Larsson; Alessandro V. Papadopoulos
>
> **备注:** Extended 8-page version containing full proofs. An abridged 6-page version has been accepted for publication in the Proceedings of the 23rd IFAC World Congress (2026). v2: Minor typographical fixes and updated reference formatting
>
> **摘要:** Robotic systems are vulnerable to False Data Injection Attacks (FDIAs), where adversaries corrupt sensor signals to gain malicious control. Feedback linearization exposes robotic systems to integrator vulnerability, making them susceptible to stealthy attacks that can cause significant deviations in end-effector behavior without raising alarms. This paper addresses the resilience of manipulators against finite-horizon FDIAs by formalizing two defense methods, namely anomaly-aware virtual damping and manipulability reduction, with probabilistic guarantees on nominal task execution. Simulations on a 7-DOF redundant manipulator show that the proposed defenses substantially reduce the impact of FDIA compared to using solely a threshold-based ADS like the Chi-squared, while preserving nominal task performance in the absence of attack.
>
---
#### [replaced 009] Constrained Policy Optimization via Sampling-Based Weight-Space Projection
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于安全强化学习任务，解决约束下策略优化问题。提出SCPO方法，在参数空间直接约束安全，无需梯度信息，确保训练过程安全与稳定。**

- **链接: [https://arxiv.org/pdf/2512.13788](https://arxiv.org/pdf/2512.13788)**

> **作者:** Shengfan Cao; Francesco Borrelli; Eunhyek Joa
>
> **备注:** Accepted for publication at IFAC World Congress 2026; fixed minor notation inconsistencies
>
> **摘要:** Safety-critical learning requires policies that improve performance without leaving the safe operating regime. We study constrained policy learning where model parameters must satisfy rollout-based safety constraints that can be evaluated but not differentiated analytically. We propose SCPO, a sampling-based weight-space projection method that enforces safety directly in parameter space without requiring gradient access to the constraint functions. SCPO constructs a local safe region by combining rollout-based safety evaluations with smoothness bounds relating parameter perturbations to changes in safety metrics, and projects each gradient update via a convex QCQP. We establish a safe-by-induction guarantee: starting from any safe initialization, all intermediate policies remain safe given feasible projections. In constrained control settings with a stabilizing backup policy, SCPO further ensures closed-loop stability while enabling safe adaptation beyond the conservative backup. Experiments on constrained regression with harmful supervision and double-integrator imitation with a malicious expert show that SCPO rejects unsafe updates, maintains feasibility throughout training, and achieves meaningful objective improvement.
>
---
#### [replaced 010] Multimodal Fusion for Sim2real Transfer in Visual Reinforcement Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉强化学习任务，旨在提升Sim2Real迁移效果。通过融合RGB与深度信息，增强模型泛化能力，并设计对比学习与课程域随机化方案，提高样本效率和实际应用可行性。**

- **链接: [https://arxiv.org/pdf/2507.09180](https://arxiv.org/pdf/2507.09180)**

> **作者:** Zichun Xu; Jingdong Zhao; Chenyu Guo; Qianxue Zhang; Liao Zhang; Xiao Zhang; Yiming Ren; Lian Zhang; Zengren Zhao
>
> **摘要:** Depth information is robust to scene appearance variations and inherently carries 3D spatial details. Thus, a visual backbone based on the vision transformer is proposed to fuse RGB and depth modalities for enhancing generalization in this paper. Different modalities are first processed by separate CNN stems, and the combined convolutional features are delivered to the scalable vision transformer to obtain visual representations. Moreover, a contrastive learning scheme is designed with masked and unmasked tokens to enhance the sample efficiency and generalization performance. A curriculum-based domain randomization scheme is used to flexibly stabilize the training process. Finally, simulation results demonstrate that our fusion scheme outperforms the other baselines. The feasibility of our model is validated to perform real-world manipulation tasks via zero-shot transfer.
>
---
#### [replaced 011] Query-Calibrated Segmental Admission for Descriptor-Agnostic LiDAR Loop Closure in Repetitive Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM任务，解决重复环境中LiDAR回环检测的误检问题。提出QCSA方法，通过校准查询段提升回环插入精度，减少错误因子。**

- **链接: [https://arxiv.org/pdf/2512.09447](https://arxiv.org/pdf/2512.09447)**

> **作者:** Jaehyun Kim; Seungwon Choi; Wonseok Kang; Tae-Wan Kim
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Structurally repetitive environments produce visually plausible but aliased LiDAR loop candidates that can destabilize pose-graph optimization when admitted as loop factors. We propose Query-Calibrated Segmental Admission (QCSA), a descriptor-agnostic sparse loop-admission policy for graph-stability-oriented insertion. The policy scores short descriptor segments against hard negatives, calibrates which query-level segment hypotheses reach geometry, and inserts representative pairs validated by Generalized Iterative Closest Point (G-ICP). We evaluate it on the SNU Library Dataset (SNULib) and HeLiPR overlap routes. Aggregated over seven LiDAR descriptor families on SNULib, QCSA reduces inserted loop factors by 3.8 times, raises factor precision from 0.542 to 0.717, and sharply lowers false admissions per query group. With this sparser graph, it maintains comparable mean absolute trajectory error (ATE) and substantially reduces worst-sequence ATE versus dense Top1+G-ICP, from 1.064 to 0.778 m. The aggregate mean and worst-sequence ATE remain lower than the odometry-only reference. Under a matched factor budget, QCSA also attains lower trajectory error than SeqSLAM and sparse Top1+G-ICP selections. Fixed-transfer validation on HeLiPR, with no route-specific tuning, likewise suppresses hard-negative admissions. These results support the proposed admission layer for aliasing-heavy simultaneous localization and mapping (SLAM). Our implementation and dataset will be released at: this https URL.
>
---
#### [replaced 012] Can VLMs Unlock Semantic Anomaly Detection? A Framework for Structured Reasoning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于异常检测任务，旨在解决自动驾驶中语义异常检测的可靠性问题。提出SAVANT框架，通过语义一致性验证提升VLM的检测效果，并实现高效数据标注与模型优化。**

- **链接: [https://arxiv.org/pdf/2510.18034](https://arxiv.org/pdf/2510.18034)**

> **作者:** Roberto Brusnicki; David Pop; Yuan Gao; Mattia Piccinini; Johannes Betz
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Autonomous driving systems remain critically vulnerable to the long-tail of rare, out-of-distribution semantic anomalies. While VLMs have emerged as promising tools for perception, their application in anomaly detection remains largely restricted to prompting proprietary models - limiting reliability, reproducibility, and deployment feasibility. To address this gap, we introduce SAVANT (Semantic Anomaly Verification/Analysis Toolkit), a novel model-agnostic reasoning framework that reformulates anomaly detection as a layered semantic consistency verification. By applying SAVANT's two-phase pipeline - structured scene description extraction and multi-modal evaluation - existing VLMs improve their scores in detecting anomalous driving scenarios from input images. Our approach replaces ad hoc prompting with semantic-aware reasoning, transforming VLM-based detection into a principled decomposition across four semantic domains. We show that across a balanced set of real-world driving scenarios, applying SAVANT improves VLM's absolute recall by approximately 18.5% compared to prompting baselines. Moreover, this gain enables reliable large-scale annotation: leveraging the best proprietary model within our framework, we automatically labeled around 10,000 real-world images with high confidence. We use the resulting high-quality dataset to fine-tune a 7B open-source model (Qwen2.5-VL) to perform single-shot anomaly detection, achieving 90.8% recall and 93.8% accuracy - surpassing all models evaluated while enabling local deployment at near-zero cost. By coupling structured semantic reasoning with scalable data curation, we provide a practical solution to data scarcity in semantic anomaly detection for autonomous systems. Supplementary material: this https URL.
>
---
#### [replaced 013] FocalPolicy: Frequency-Optimized Chunking and Locally Anchored Flow Matching for Coherent Visuomotor Policy
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于视觉-运动策略任务，旨在解决长时序动作的连贯性问题。提出FocalPolicy，结合频率优化分块和局部锚定流匹配，提升动作一致性与预测精度。**

- **链接: [https://arxiv.org/pdf/2605.15944](https://arxiv.org/pdf/2605.15944)**

> **作者:** Qian He; Zhenshuo Yang; Wenqi Liang; Chunhui Hao; Nicu Sebe; Jiandong Tian
>
> **摘要:** Visuomotor policies aim to learn complex manipulation tasks from expert demonstrations. However, generating smooth and coherent trajectories remains challenging, as it requires balancing proximal precision with distal foresight. Existing approaches typically focus on optimizing intra-chunk action distributions, often neglecting the inter-chunk coherence. Consequently, inter-chunk discontinuities significantly impede the learning of coherent long-horizon actions. To overcome this limitation and achieve a synergetic balance between precision and foresight, we propose FocalPolicy, a foresight-aware visuomotor policy that combines Frequency-Optimized Chunking with Locally Anchored flow matching. We introduce a foresight composite objective that supervises time-domain alignment within the proximal actions while regularizing frequency-domain structure over multiple future action chunks to improve cross-chunk coherence. To efficiently learn complex action distributions, we design locally anchored sampling to enhance target signal propagation efficiency during consistency flow matching training. Extensive experiments demonstrate that FocalPolicy outperforms existing approaches and confirm the generalizability of our modules to other baselines. Project website: this https URL
>
---
#### [replaced 014] Hand-in-the-Loop: Improving VLA Policies for Dexterous Manipulation via Seamless Hand-Arm Intervention
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人操作任务，解决高维动作空间中策略偏差导致的误差累积问题。提出HandITL方法，实现人机无缝协作，减少干预抖动，提升操作鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.15157](https://arxiv.org/pdf/2605.15157)**

> **作者:** Zhuohang Li; Liqun Huang; Wei Xu; Zhengming Zhu; Nie Lin; Xiao Ma; Xinjun Sheng; Ruoshi Wen
>
> **摘要:** Vision-Language-Action (VLA) models are prone to compounding errors in dexterous manipulation, where high-dimensional action spaces and contact-rich dynamics amplify small policy deviations over long horizons. While Interactive Imitation Learning (IIL) can refine policies through human correction data, applying it to high-degree-of-freedom (DoF) robotic hands remains challenging due to a command mismatch between human teleoperation and policy execution at the intervention moment, which causes abrupt robot-hand configuration changes, or "gesture jumps". We present Hand-in-the-Loop (HandITL), a seamless human-in-the-loop intervention method that blends human corrective intent with autonomous policy execution to avoid gesture jumps during bimanual dexterous manipulation. Compared with taking over control using direct teleoperation, HandITL reduces intervention jitter by 99.8% and preserves robust post-intervention manipulation, reducing grasp failures by 87.5% and mean completion time by 19.1%. We validate HandITL on tasks requiring bimanual coordination, tool use, and fine-grained long-horizon manipulation. When used to collect correction data for policy refinement, HandITL yields policies that outperform those trained with standard teleoperation data by 19% on average across three long-horizon dexterous tasks.
>
---
#### [replaced 015] ARC-RL: A Reinforcement Learning Playground Inspired by ARC Raiders
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出ARC-RL，一个基于游戏角色的强化学习环境，用于研究仿生机器人运动控制，解决形态多样性与风格约束下的训练问题。**

- **链接: [https://arxiv.org/pdf/2605.19503](https://arxiv.org/pdf/2605.19503)**

> **作者:** Carlo Romeo; Andrew D. Bagdanov
>
> **摘要:** Reinforcement learning for legged locomotion has matured into a stack of multi-component reward functions and physics-engine benchmarks whose morphologies are uniformly derived from real commercial hardware. Game NPCs, however, are bound by stylistic constraints absent from sim-to-real robotics and routinely take the form of creatures with no real-robot counterpart. We introduce ARC-RL, a suite of four MuJoCo continuous-control environments featuring robotic morphologies inspired by the bestiary of ARC Raiders: the 18-DoF tall hexapod Queen, the 12-DoF armoured hexapod Bastion, the 18-DoF compact hexapod Tick, and the 12-DoF quadruped Leaper. All four robots share a unified observation template, action convention, simulation cadence, and a single closed-form multi-component reward function whose only per-morphology variation lives in a small set of weights and parameters. The reward fuses a velocity-tracking tent, a healthy survive bonus, a phase-locked gait-compliance bonus/cost pair, action regularisers, three safety penalties, and a posture anchor; no motion-capture data enters the reward at any point. We additionally provide hand-crafted Central Pattern Generator demonstrators per morphology, which serve both as fixed expert references and as sources of prior data for offline-to-online training. On this playground, we conduct a controlled empirical study comparing standard online algorithms (SAC, SPEQ, SOPE-EO) and methods augmented with prior data (SACfD, SPEQ-O2O, SOPE), and characterise how each paradigm copes with the playground's morphological diversity and animation-style stylistic constraints. Source code is available at this https URL.
>
---
#### [replaced 016] How to Utilize Failure Demo Data?: Effective Data Selection for Imitation Learning Using Distribution Differences in Attention Mechanism
- **分类: cs.RO**

- **简介: 该论文属于机器人模仿学习任务，旨在解决如何有效利用失败数据的问题。通过学习成功与失败的差异表示，并结合注意力机制提升动作稳定性。**

- **链接: [https://arxiv.org/pdf/2605.07560](https://arxiv.org/pdf/2605.07560)**

> **作者:** Kana Miyamoto; Kanata Suzuki; Tetsuya Ogata
>
> **备注:** 15 pages, 6 figures, 2 tables
>
> **摘要:** Imitation learning for robotic tasks has relied primarily on policies trained only on successful demonstrations, although failures are unavoidable during human data collection. Many existing approaches for exploiting failure data require additional data processing or iterative policy updates through autonomous rollouts, making it difficult to directly and stably utilize failure data accumulated during data collection. In this work, we propose a method that learns latent representations of success-failure discrepancies and incorporates them into the attention mechanism. During inference, an appropriate latent mode is selected from the initial observation to improve action stability. Furthermore, we introduce a post-training metric that quantifies the attention discrepancy between each failure sample and successful demonstrations to select failure data. Simulation results show that the proposed method improves task success rates when trained with failure data and that the proposed metric identifies failure samples that are beneficial for learning when combined with successful demonstrations. These results suggest that the proposed method can support more efficient use of collected demonstrations in robotic data collection pipelines.
>
---
#### [replaced 017] Dual Quaternion Based Contact Modeling for Fast and Smooth Collision Recovery of Quadrotors
- **分类: cs.RO**

- **简介: 该论文属于无人机碰撞恢复任务，解决传统模型解耦问题，提出基于双四元数的接触建模方法，提升碰撞后稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2603.14698](https://arxiv.org/pdf/2603.14698)**

> **作者:** Valentin Gaucher; Wenlong Zhang
>
> **备注:** 8 pages, 5 figures, 2 tables
>
> **摘要:** Unmanned aerial vehicles (UAVs) operating in cluttered environments require efficient and accurate impact modeling to maintain stability post collisions, however classical impulse contact models decouple the normal and tangential components. This letter presents a dual quaternion impulse reset map directly on the SE(3) manifold. By operating on the unified spatial twist (unified linear and angular velocities), the proposed formulation retains the cross-coupling between normal and tangential impulse components in a single closed-form expression, and recovers the classical decoupled Newton impulse model as a special case. A recovery controller is designed that couples linear and angular momentum to enforce kinetic energy dissipation across impacts. Hardware-in-the-loop benchmarks demonstrate a 24\% reduction in execution latency compared to an optimized matrix-based implementation, and a 20\% reduction relative to a position-plus-quaternion (PQ) formulation. MuJoCo simulations across Monte Carlo sweeps over impact angles and friction coefficients show a 50.8\%-75.1\% reduction in position root-mean-square error (RMSE) and a 68.7\%-85\% decrease in peak kinetic energy compared to published linear-admittance baselines.
>
---
#### [replaced 018] WestWorld: A Knowledge-Encoded Scalable Trajectory World Model for Diverse Robotic Systems
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出WestWorld，解决轨迹世界模型在多样机器人系统中的可扩展性和知识融合问题。通过引入系统感知的专家混合和结构嵌入，提升零样本泛化与控制性能。**

- **链接: [https://arxiv.org/pdf/2603.14392](https://arxiv.org/pdf/2603.14392)**

> **作者:** Yuchen Wang; Jiangtao Kong; Sizhe Wei; Xiaochang Li; Haohong Lin; Hongjue Zhao; Tianyi Zhou; Lu Gan; Huajie Shao
>
> **备注:** ICML 2026 spotlight
>
> **摘要:** Trajectory world models play a crucial role in robotic dynamics learning, planning, and control. While recent works have explored trajectory world models for diverse robotic systems, they struggle to scale to a large number of distinct system dynamics and overlook domain knowledge of physical structures. To address these limitations, we introduce WestWorld, a knoWledge-Encoded Scalable Trajectory World model for diverse robotic systems. To tackle the scalability challenge, we propose a novel system-aware Mixture-of-Experts (Sys-MoE) that dynamically combines and routes specialized experts for different robotic systems via a learnable system embedding. To further enhance zero-shot generalization, we incorporate domain knowledge of robot physical structures by introducing a structural embedding that aligns trajectory representations with morphological information. After pretraining on 89 complex environments spanning diverse morphologies across both simulation and real-world settings, WestWorld achieves significant improvements over competitive baselines in zero- and few-shot trajectory prediction. Additionally, it shows strong scalability across a wide range of robotic environments and significantly improves performance on downstream model-based control for different robots. Finally, we deploy our model on a real-world Unitree Go1, where it demonstrates stable locomotion performance. The code is available at this https URL.
>
---
#### [replaced 019] CosFly-Track: A Large-Scale Multi-Modal Dataset for UAV Visual Tracking via Multi-Constraint Trajectory Optimization
- **分类: cs.RO**

- **简介: 该论文提出CosFlyTrack数据集，用于解决无人机视觉跟踪任务中的缺乏专用训练数据问题，通过多约束轨迹优化生成高质量轨迹。**

- **链接: [https://arxiv.org/pdf/2605.17776](https://arxiv.org/pdf/2605.17776)**

> **作者:** Xiangyue Wang; Hanxuan Chen; Songsheng Cheng; Ruilong Ren; Jie Zheng; Shuai Yuan; Tianle Zeng; Hanzhong Guo; Kangli Wang; Ji Pei
>
> **摘要:** Recent aerial vision-language navigation (VLN) datasets have grown rapidly, but they primarily address goal-oriented navigation to static destinations, leaving UAV visual tracking -- continuously following a moving target while maintaining visibility -- largely without dedicated training data. We introduce CosFlyTrack, a large-scale multi-modal dataset and scalable generation pipeline for UAV visual tracking in urban environments. The dataset provides approximately 12,000 expert and perturbed UAV trajectories generated from 6,000 pedestrian paths, comprising 2.4 million timesteps (approximately 334 hours) with seven aligned data channels: RGB, metric depth, semantic segmentation, six-degree-of-freedom drone pose, target state with visibility flag, bilingual (Chinese-English) instructions, and trajectory-pair metadata. To generate high-quality expert trajectories, we develop MuCO, a multi-constraint optimizer that plans directly in continuous three-dimensional space with BVH-accelerated collision and visibility queries, jointly enforcing target visibility, viewpoint quality, collision avoidance, smoothness, and kinematic feasibility, avoiding the discretization artifacts and post-hoc smoothing of grid-based planners. Fine-tuning experiments on seven vision-language models show that CosFlyTrack improves tracking performance to 78.3 to 95.6 percent SR@1 meter, a 53 to 69 percentage point gain over zero-shot baselines, supporting the dataset as a training resource for dynamic target-following agents. The dataset is publicly available at this https URL evaluation scripts and pre-trained checkpoints are hosted at this https URL.
>
---
#### [replaced 020] TimeRewarder: Learning Dense Reward from Passive Videos via Frame-wise Temporal Distance
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出TimeRewarder，用于从视频中学习密集奖励，解决强化学习中奖励设计困难的问题。通过帧间时间距离建模，生成任务进度信号，提升稀疏奖励任务的性能。**

- **链接: [https://arxiv.org/pdf/2509.26627](https://arxiv.org/pdf/2509.26627)**

> **作者:** Yuyang Liu; Chuan Wen; Yihang Hu; Dinesh Jayaraman; Yang Gao
>
> **备注:** ICML 2026 spotlight paper
>
> **摘要:** Designing dense rewards is crucial for reinforcement learning (RL), yet in robotics it often demands extensive manual effort and lacks scalability. One promising solution is to view task progress as a dense reward signal, as it quantifies the degree to which actions advance the system toward task completion over time. We present TimeRewarder, a simple yet effective reward learning method that derives progress estimation signals from passive videos, including robot demonstrations and human videos, by modeling temporal distances between frame pairs. We then demonstrate how TimeRewarder can supply step-wise proxy rewards to guide reinforcement learning. In our comprehensive experiments on ten challenging Meta-World tasks, we show that TimeRewarder dramatically improves RL for sparse-reward tasks, achieving nearly perfect success in 9/10 tasks with only 200,000 environment interactions per task. This approach outperformed previous methods and even the manually designed environment dense reward on both the final success rate and sample efficiency. Moreover, we show that TimeRewarder pretraining can exploit real-world human videos, highlighting its potential as a scalable approach to rich reward signals from diverse video sources.
>
---
#### [replaced 021] SPARC: Spatial-Aware Path Planning via Attentive Robot Communication
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多机器人路径规划任务，解决通信效率低下问题。提出RMHA机制，通过空间距离增强注意力，提升密集环境下的协作成功率。**

- **链接: [https://arxiv.org/pdf/2603.02845](https://arxiv.org/pdf/2603.02845)**

> **作者:** Sayang Mu; Xiangyu Wu; Bo An
>
> **备注:** The manuscript is being withdrawn at the request of the first author for the purpose of revising content and re-uploading a revised version with updated data/figures/text . The revised manuscript will be resubmitted to arXiv promptly with the same author list and research theme
>
> **摘要:** Efficient communication is critical for decentralized Multi-Robot Path Planning (MRPP), yet existing learned communication methods treat all neighboring robots equally regardless of their spatial proximity, leading to diluted attention in congested regions where coordination matters most. We propose Relation enhanced Multi Head Attention (RMHA), a communication mechanism that explicitly embeds pairwise Manhattan distances into the attention weight computation, enabling each robot to dynamically prioritize messages from spatially relevant neighbors. Combined with a distance-constrained attention mask and GRU gated message fusion, RMHA integrates seamlessly with MAPPO for stable end-to-end training. In zero-shot generalization from 8 training robots to 128 test robots on 40x40 grids, RMHA achieves approximately 75 percent success rate at 30 percent obstacle density outperforming the best baseline by over 25 percentage points. Ablation studies confirm that distance-relation encoding is the key contributor to success rate improvement in high-density environments. Index Terms-Multi-robot path planning, graph attention mechanism, multi-head attention, communication optimization, cooperative decision-making
>
---
#### [replaced 022] Depth Completion in Unseen Field Robotics Environments Using Extremely Sparse Depth Measurements
- **分类: cs.RO**

- **简介: 该论文属于深度补全任务，解决野外机器人环境中深度感知不足的问题。通过合成数据训练，利用稀疏深度测量预测密集深度，实现实时部署。**

- **链接: [https://arxiv.org/pdf/2602.03209](https://arxiv.org/pdf/2602.03209)**

> **作者:** Marco Job; Thomas Stastny; Eleni Kelasidi; Roland Siegwart; Michael Pantic
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Autonomous field robots operating in unstructured environments require robust perception to ensure safe and reliable operations. Recent advances in monocular depth estimation have demonstrated the potential of low-cost cameras as depth sensors; however, their adoption in field robotics remains limited due to the absence of reliable scale cues, ambiguous or low-texture conditions, and the scarcity of large-scale datasets. To address these challenges, we propose a depth completion model that trains on synthetic data and uses extremely sparse measurements from depth sensors to predict dense metric depth in unseen field robotics environments. A synthetic dataset generation pipeline tailored to field robotics enables the creation of multiple realistic datasets for training purposes. This dataset generation approach utilizes textured 3D meshes from Structure from Motion and photorealistic rendering with novel viewpoint synthesis to simulate diverse field robotics scenarios. Our approach achieves an end-to-end latency of 53 ms per frame on a Nvidia Jetson AGX Orin, enabling real-time deployment on embedded platforms. Extensive evaluation demonstrates competitive performance across diverse real-world field robotics scenarios.
>
---
#### [replaced 023] MAPLE: Latent Multi-Agent Play for End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MAPLE框架，用于解决端到端自动驾驶中的多智能体闭环控制问题，通过latent空间训练提升模型鲁棒性与交互 realism。**

- **链接: [https://arxiv.org/pdf/2605.14201](https://arxiv.org/pdf/2605.14201)**

> **作者:** Rajeev Yasarla; Deepti Hegde; Hsin-Pai Cheng; Shizhong Han; Yunxiao Shi; Meysam Sadeghigooghari; Hanno Ackermann; Litian Liu; Pranav Desai; Fatih Porikli; Mohammad Ghavamzadeh; Hong Cai
>
> **备注:** 19 pages, 9 figures
>
> **摘要:** Vision-language-action (VLA) models are effective as end-to-end motion planners, but can be brittle when evaluated in closed-loop settings due to being trained under traditional imitation learning framework. Existing closed-loop supervision approaches lack scalability and fail to completely model a reactive environment. We propose MAPLE, a novel framework for reactive, multi-agent rollout of a dynamic driving scenario in the latent space of the VLA model. The ego vehicle and nearby traffic agents are independently controlled over multi-step horizons, while being reactive to other agents in the scene, enabling closed-loop training. MAPLE consists of two training stages: (1) supervised fine-tuning on the latent rollouts based on ground-truth trajectories, followed by (2) reinforcement learning with global and agent -specific rewards that encourage safety, progress, and interaction realism. We further propose diversity rewards that encourage the model to generate planning behaviors that may not be present in logged driving data. Notably, our closed-loop training framework is scalable and does not require external simulators, which can be computationally expensive to run and have limited visual fidelity to the real-world. MAPLE achieves state-of-the-art driving performance on Bench2Drive and demonstrates scalable, closed-loop multi-agent play for robust E2E autonomous driving systems.
>
---
#### [replaced 024] RankQ: Offline-to-Online Reinforcement Learning via Self-Supervised Action Ranking
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出RankQ，解决离线到在线强化学习中的样本效率问题，通过自监督动作排序提升策略性能。**

- **链接: [https://arxiv.org/pdf/2605.11151](https://arxiv.org/pdf/2605.11151)**

> **作者:** Andrew Choi; Wei Xu
>
> **摘要:** Offline-to-online reinforcement learning (RL) improves sample efficiency by leveraging pre-collected datasets prior to online interaction. A key challenge, however, is learning an accurate critic in large state--action spaces with limited dataset coverage. To mitigate harmful updates from value overestimation, prior methods impose pessimism by down-weighting out-of-distribution (OOD) actions relative to dataset actions. While effective, this essentially acts as a behavior cloning anchor and can hinder downstream online policy improvement when dataset actions are suboptimal. We propose RankQ, an offline-to-online Q-learning objective that augments temporal-difference learning with a self-supervised multi-term ranking loss to enforce structured action ordering. By learning relative action preferences rather than uniformly penalizing unseen actions, RankQ shapes the Q-function such that action gradients are directed toward higher-quality behaviors. Across sparse reward D4RL benchmarks, RankQ achieves performance competitive with or superior to seven prior methods. In vision-based robot learning, RankQ enables effective offline-to-online fine-tuning of a pretrained vision-language-action (VLA) model in a low-data regime, achieving on average a 42.7% higher simulation success rate than the next best method. In a high-data setting, RankQ improves simulation performance by 13.7% over the next best method and achieves strong sim-to-real transfer, increasing real-world cube stacking success from 43.1% to 88.9% relative to the VLA's initial performance.
>
---
