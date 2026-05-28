# 机器人 cs.RO

- **最新发布 60 篇**

- **更新 25 篇**

## 最新发布

#### [new 001] SARAD: LLM-Based Safety-Aware Hybrid Reinforcement Learning with Collision Prediction for Autonomous Driving
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 该论文属于自动驾驶任务，旨在解决安全与效率问题。提出SARAD框架，结合LLM与DRL，提升决策安全性与效率。**

- **链接: [https://arxiv.org/pdf/2605.28583](https://arxiv.org/pdf/2605.28583)**

> **作者:** Kangyu Wu; Peng Cui; Guoxi Chen; Ya Zhang
>
> **备注:** 7 pages, 4 figures, accepted by IJCNN 2026
>
> **摘要:** Ensuring both safety and efficiency in decision-making for autonomous driving systems remains a fundamental challenge. Traditional Deep Reinforcement Learning (DRL) suffers from unsafe random exploration and slow convergence, while Large Language Models (LLMs) demonstrate inherent latency in real-time inference operations. To address these limitations, this paper proposes SARAD, a novel safety-aware hybrid framework that synergizes LLMs and DRL for autonomous driving. SARAD substitutes the random exploration of DRL with Retrieval-Augmented Generation (RAG)-enhanced, LLM-guided decisions sourced from a dynamic expert knowledge repository. An attention discriminator is proposed to integrate the prior knowledge of LLMs into DRL policy optimization. A collision predictor module, fine-tuned with historical collision data, is further designed to improve vehicle safety. Extensive experiments show that SARAD achieves significant performance improvements in the Highway-Env simulator, validating the effectiveness of the proposed model in autonomous driving.
>
---
#### [new 002] STR Robot: Design of an Autonomous Mobile Robot from Simulation to Reality
- **分类: cs.RO**

- **简介: 该论文属于自主移动机器人任务，解决从仿真到现实的部署问题。工作包括开发控制、定位和导航系统，并在仿真后实测验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.28110](https://arxiv.org/pdf/2605.28110)**

> **作者:** Vinh Nguyen; Gia-Uy Le; Tien-Dat Nguyen; Tri-Tin Nguyen; Vinh-Hao Nguyen
>
> **摘要:** With the rapid development of simulation tools, the development and validation of autonomous robotic systems have become more efficient before real-world deployment. This paper presents a simulation-to-real implementation of an autonomous mobile robot based on an existing mechanical platform. Instead of focusing on mechanical design, our work concentrates on the development of the onboard control, self-localization, and autonomous navigation system. The proposed robot is equipped with onboard sensing and computation to estimate its pose and navigate autonomously in the environment. The overall framework is first developed and tested in simulation, and then deployed on the real robot for experimental evaluation. The results demonstrate the feasibility of the proposed approach and show that simulation provides an effective foundation for developing reliable autonomous mobile robot systems. The source code will be released at this https URL.
>
---
#### [new 003] SAFEVPR: Patch-Based Conformal Verification for Safe Cross-Condition Sequence Visual Place Recognition
- **分类: cs.RO**

- **简介: 该论文属于视觉定位任务，解决跨条件序列视觉定位中的安全验证问题。提出SAFEVPR方法，通过改进匹配分数和校准策略，实现可靠的安全决策。**

- **链接: [https://arxiv.org/pdf/2605.28048](https://arxiv.org/pdf/2605.28048)**

> **作者:** Ha Sier; Jiaqiang Zhang; Zhuo Zou; Xianjia Yu; Tomi Westerlund
>
> **摘要:** Sequence-based visual place recognition (VPR) for SLAM and robot relocalization must decide whether the retrieved top-1 candidate is safe to accept. Conformal prediction is a natural framework for this accept/reject decision, but its finite-sample guarantees rely on exchangeability between calibration and deployment (test) data, which is violated under cross-condition deployment. We introduce SAFEVPR, a non-trainable verification-and-calibration pipeline for safe cross-condition sequence VPR. SAFEVPR replaces the standard backbone cosine similarity with a mutual-nearest-neighbour (MNN) patch-matching score computed from frozen DINOv2 ViT features, and replaces flat Learn-Then-Test calibration with Mondrian conformal LTT, fitting separate Bonferroni-corrected thresholds across score bins. Under exchangeability, these thresholds would provide finite-sample false-discovery-rate (FDR) control; under condition shift, we evaluate empirical validity per deployment. Across 23 cross-condition setups from Oxford RobotCar, NCLT, and St Lucia datasets, using three frozen VPR backbones, SAFEVPR is empirically valid on 23/23 setups at target FDR alpha = 0.10, achieving mean accepted FDR 0.014 and mean true-positive rate (TPR) 0.75. The results show that raw discrimination alone is not sufficient for conformal validity: AnyLoc-VLAD and Super-Point+LightGlue reach comparable area under the receiver operating characteristic curve (AUROC) but fail more setups under the same calibration. On textureless repetitive scenery, SAFEVPR safely abstains rather than accepting unreliable matches. Code is available at this https URL.
>
---
#### [new 004] VLM-Based Advanced Rider Assistance System for Motorcycle Safety
- **分类: cs.RO**

- **简介: 该论文属于摩托车安全任务，旨在解决ARAS发展不足的问题。通过VLM构建风险地图，结合动态规划提升骑行安全。**

- **链接: [https://arxiv.org/pdf/2605.27948](https://arxiv.org/pdf/2605.27948)**

> **作者:** Mohamed Elnoor; Francesca Baldini; Ananya Trivedi; Faizan M. Tariq; Jovin D'sa; David Isele; Sangjae Bae; Dinesh Manocha; Yosuke Sakamoto
>
> **备注:** Accepted to IEEE IV 2026
>
> **摘要:** Motorcycles face disproportionately high crash risks compared to cars due to limited protection and heightened sensitivity to surface hazards, yet Advanced Rider Assistance Systems (ARAS) remain underdeveloped relative to Advanced Driver Assistance Systems (ADAS). We propose a novel ARAS that enhances motorcycle safety through semantic perception and risk-aware planning. Our approach leverages Vision-Language Models (VLMs) for contextual hazard reasoning and integrates them with segmentation-based detection to construct dense risk maps. These maps encode both semantic characteristics (e.g., pothole severity, puddle slipperiness) and physical attributes (e.g., size, depth), which produce per-pixel hazard costs that capture motorcycle-specific risks. These maps are used by a sampling-based planner tailored to motorcycle dynamics to recommend throttle and steering actions that minimize hazard exposure while advancing toward the destination. We evaluate our system in different scenarios in the CARLA simulator. Compared to the baseline method, our method achieves higher success rates and lower hazard exposure, while qualitative results demonstrate interpretable risk maps and safe trajectory recommendations.
>
---
#### [new 005] Imitation Learning for Robot Assistance in Open Surgery: A Multi-Policy Evaluation on Suture Following
- **分类: cs.RO**

- **简介: 该论文研究机器人在开放手术中通过模仿学习辅助缝合的任务，解决如何提高机器人协作精度与稳定性的问题。通过评估多种模仿学习策略，验证了其在临床场景中的可行性。**

- **链接: [https://arxiv.org/pdf/2605.28736](https://arxiv.org/pdf/2605.28736)**

> **作者:** Xucheng Wang; Zhizhou Yang; Xiaoman Zhang; Sung Eun Kim; Romain Hardy; Pranav Rajpurkar
>
> **摘要:** This study presents the first evaluation of general-purpose imitation learning for surgeon-robot collaborative assistance in open surgery, targeting suture following: the grab-pull-release motion an assistant performs at every stitch. We collect 160 teleoperated demonstrations (32,374 frames) on an open-source robot arm, benchmark four architecturally diverse imitation learning policies (ACT, Diffusion Policy, SmolVLA, $\pi_0$) across 28 trained models evaluated in 32 configurations along three clinically motivated dimensions: dataset size, camera viewpoint, and background variation. Our results demonstrate that under ideal conditions, the four policies achieve $50$-$75\%$ task success, with depth error as the dominant failure mode across all architectures. Among all policies, $\pi_0$ achieves the strongest results with a pretrained vision-language backbone, demonstrating superior data efficiency, greater robustness to background variation, and smoother trajectories compatible with surgical workflow. When deployed in a surgeon-robot suturing trial, $\pi_0$ yields a $92\%$ stitch completion rate. These findings establish collaborative robotic assistance in open surgery as a feasible target for imitation learning and highlight depth perception and end-effector design as key priorities for clinical translation.
>
---
#### [new 006] Integrated Exploration-Aware UAV Route Optimization and Path Planning
- **分类: cs.RO; eess.SY; math.OC**

- **简介: 该论文属于无人机路径规划任务，解决在不确定环境中高效监测危险区域的问题。通过集成探索与路径优化，提升信息收集效率。**

- **链接: [https://arxiv.org/pdf/2605.28654](https://arxiv.org/pdf/2605.28654)**

> **作者:** Jimin Choi; Grant Stagg; Cameron K. Peterson; Max Z. Li
>
> **摘要:** Uncrewed aerial vehicles (UAVs) are increasingly used for exploration-driven monitoring in hazardous environments such as disaster zones, contaminated sites, wildfire areas, and damaged infrastructure, where limited flight endurance must be allocated between visiting reported locations and gathering new information. In these settings, prior information regarding hazards is often incomplete, spatially imprecise, and subject to change during execution. For example, initial reports may identify a region where a hazard is likely to exist, but the actual hazard may be displaced, partially observed, or entirely unreported. We present an integrated exploration-aware UAV route optimization and path planning framework for hazard monitoring under uncertain and evolving prior information. The environment is represented as a spatial risk map, where each location has an associated belief of hazardous conditions. Reported hazards are modeled as uncertain regions of interest (ROIs) rather than confirmed target locations, requiring the UAV to inspect reported areas while also using its limited flight endurance to explore informative regions. The proposed method solves a vehicle routing problem over reported ROIs, augments the route with auxiliary pseudo-nodes to improve spatial coverage, allocates the remaining flight distance budget across route segments, and optimizes dynamically feasible B-spline trajectories for local exploration. During execution, UAV measurements update a grid-based belief map, and the remaining trajectory is replanned when new information and the remaining budget justify adaptation. Across 48 scenario configurations, online replanning improves average KL reduction by 15.9% over the offline optimized planner and 48.6% over straight-line traversal.
>
---
#### [new 007] Safety-Critical Adaptive Impedance Control via Nonsmooth Control Barrier Functions under State and Input Constraints
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人安全控制任务，旨在解决不确定动态下的安全交互问题。通过结合模糊系统与非光滑控制屏障函数，实现关节状态约束与阻抗跟踪的在线自适应控制。**

- **链接: [https://arxiv.org/pdf/2605.28367](https://arxiv.org/pdf/2605.28367)**

> **作者:** Faisal Lawan; Xiaoran Han; Joaquin Carrasco; Barry Lennox; Xiaoxiao Cheng
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Safe physical interaction is critical for deploying robotic manipulators in human-robot interaction and contact-rich tasks, where uncertainty, external forces, and actuator limitations can compromise both performance and safety. We propose an online adaptive impedance control framework that enforces joint-state safety while achieving compliant interaction under uncertain dynamics. The approach combines a quadratic-program-based safety filter with a novel composed position-velocity non-smooth control barrier function (NCBF), enabling joint position and velocity constraints to be enforced through a unified relative-degree-one barrier. Unknown dynamics are compensated online using an interval type-2 fuzzy logic system, while actuator torque limits are handled through soft constraints with exact penalty recovery of feasible solutions. A disturbance-observer-enhanced safety mechanism improves robustness against modelling errors and external interaction forces. Using composite Lyapunov analysis, we prove forward invariance of the safe set and the uniform ultimately boundedness of the impedance-tracking error. Simulations on a 7-DOF manipulator with severe parametric uncertainty and external interaction wrenches demonstrate safe constraint satisfaction and robust impedance tracking.
>
---
#### [new 008] A Surveillance Evasion Game with Continuous Sensor Redeployment via Bilevel Optimization
- **分类: cs.RO**

- **简介: 该论文属于反无人机任务，解决UAS绕过传感器网络的问题。通过双层优化实现传感器连续部署，提升防御效果。**

- **链接: [https://arxiv.org/pdf/2605.27917](https://arxiv.org/pdf/2605.27917)**

> **作者:** Jaehyeok Kim; Kartik A. Pant; Joseph Kinerson; Kylie Sommer-Kohrt; Worawis Sribunma; Li-Yu Lin; James M. Goppert
>
> **备注:** 8 pages, 8 figures, submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Uncrewed Aerial Systems (UASs) have become a growing threat to the security of critical infrastructure, exploiting spatiotemporal gaps in sensor perimeters to infiltrate restricted airspace undetected. We formulate this interaction as a two-player zero-sum differential game between an adversarial UAS and a heterogeneous sensor network of directional and omnidirectional sensors. Unlike earlier game-theoretic approaches that restrict the defender to discrete placement graphs or fixed configurations, we introduce a continuous sensor redeployment technique in which each sensor slides freely along the convex building boundaries. This is enforced via a log-sum-exp smooth approximation that preserves differentiability at polygon vertices, enabling optimization with gradient-based methods. The attacker's best response is computed via a two-step approach combining STP-RRT* for feasible trajectory initialization and nonlinear programming for detection-minimization refinement. The joint optimization converges to a Local Nash Equilibrium (LNE) via alternating bilevel optimization, with analytical first-order stationarity conditions derived for both players, thereby establishing a deployable baseline for heterogeneous sensor placements in CUAS missions.
>
---
#### [new 009] Chance-Constrained MPPI under State and Dynamic Object Prediction Uncertainty and the Evaluation of Collision Risk Calibration
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决动态环境中碰撞风险的不确定性问题。提出DUCCT-MPPI方法，融合定位与障碍物预测不确定性，提升导航安全性与效率。**

- **链接: [https://arxiv.org/pdf/2605.28330](https://arxiv.org/pdf/2605.28330)**

> **作者:** Benjamin Serfling; Konrad Doll; Kati Radkhah-Lens
>
> **备注:** Submitted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)
>
> **摘要:** Chance-constrained Model Predictive Path Integral (MPPI) control is increasingly adopted for navigation in dynamic environments to explicitly bound collision risk. However, these probabilistic guarantees implicitly assume that upstream uncertainties from localization and perception are well-calibrated. In practice, estimators are often miscalibrated, inducing characteristic closed-loop failure modes: overconfidence leads to systematic safety violations, while underconfidence triggers overly conservative freezing or probability dilution. To address this critical gap, our primary contribution is a rigorous evaluation methodology applying proper scoring rules to assess the statistical validity of predicted collision risks during closed-loop execution. Concurrently, Dual-Uncertainty Chance-Constrained Tube MPPI (DUCCT-MPPI) is proposed as a real-time, risk-aware planning architecture. DUCCT-MPPI jointly integrates localization uncertainty via a one-tube Unscented Transform (UT) approximation and dynamic obstacle prediction uncertainty via Monte Carlo aggregation. Through extensive physics-based simulations, the framework demonstrates robust failure-mitigation, seamlessly transitioning to safe, conservative maneuvering without succumbing to functional deadlocks in highly cluttered environments. In highly cluttered environments, DUCCT-MPPI achieves superior robustness, outperforming established Monte Carlo MPPI baselines by nearly 28\% in navigation success rate, while simultaneously recording the lowest travel times and minimizing induced social forces. Ultimately, these findings establish that reliable probabilistic safety in autonomous navigation dictates not only expressive risk models but statistically valid uncertainty estimates throughout the entire autonomy stack.
>
---
#### [new 010] Tabero: Learning Gentle Manipulation with Closed-Loop Force Feedback from Vision, Touch, and Language
- **分类: cs.RO**

- **简介: 该论文提出Tabero，解决机器人在语言指令下实现柔和操作的问题，通过视觉、触觉和语言的闭环力反馈提升操作精度与安全性。**

- **链接: [https://arxiv.org/pdf/2605.27886](https://arxiv.org/pdf/2605.27886)**

> **作者:** Qiwei Wu; Rui Zhang; Xin Xiang; Tao Li; Weihua Zhang; Junjie Lai; Renjing Xu
>
> **备注:** Code:this https URL
>
> **摘要:** Tactile sensing is essential for robots to achieve human-like gentle manipulation. However, existing Vision-Language-Action (VLA) models struggle to exploit tactile feedback for gentle manipulation due to scarce aligned vision-tactile-language data and the lack of effective closed-loop force feedback mechanisms. To address these challenges, we introduce Tabero, a benchmark and model suite for gentle, language-conditioned robotic manipulation that demands fine-grained contact force perception. First, the Tabero benchmark addresses the scarcity of tactile data by presenting a data-efficient pipeline that repurposes open-source robot manipulation trajectories to generate diverse vision-tactile-language tasks, and establishes a multidimensional evaluation protocol that measures task success alongside physical interaction quality. Second, we propose Tabero-VTLA, an architecture with a decoupled force-position command interface; the resulting force-position commands are executed by a fixed hybrid controller to enable real-time, force-aware manipulation. Evaluated on Tabero, our model maintains high task success while reducing average grip force by over 70\% under gentle instructions, demonstrating its ability to modulate interaction forces based on multimodal experience. Our code is publicly available at this https URL.
>
---
#### [new 011] Self-Supervised Online Robot-Agnostic Traversability Estimation for Open-World Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人导航任务，解决开放环境中路径可行驶性估计问题。提出COTRATE框架，实现在线学习与知识迁移，提升导航安全性与效率。**

- **链接: [https://arxiv.org/pdf/2605.28442](https://arxiv.org/pdf/2605.28442)**

> **作者:** Julia Hindel; Simon Bultmann; Houman Masnavi; Daniele Cattaneo; Abhinav Valada
>
> **备注:** 14 pages, 16 Figures
>
> **摘要:** Self-supervised online traversability estimation enables robots to continuously learn from unlabeled open-world experiences and adapt their navigation behavior toward safe and efficient trajectories. Existing approaches either rely on handcrafted proprioceptive traversability scores, limiting robot-agnosticism, or cluster prior data, preventing online learning. Moreover, many continual learning methods incur substantial memory and computational costs, hindering onboard deployment. We introduce COTRATE, an online learning framework for continuous traversability estimation from multimodal, unlabeled robot experience. Our method first infers robust traversability scores using a robot-agnostic, learning-based online terrain assessment module operating on proprioceptiveand inertial signals. These scores then supervise a visual traversability network through a novel alignment loss that associates visual embeddings with online terrain this http URL mitigate forgetting during continual learning with minimal overhead, we propose a diversity-aware feature selection strategythat preserves performance using a compact replay memory. We further show that the learned traversability representation supports knowledge transfer across different robot platforms with different locomotion kinematics. We evaluate COTRATE on a dataset of \approx 50,000 images collected with two robotic platforms across 11 outdoor terrains, and benchmark it on navigation tasks in three representative outdoor environments. We make the dataset, code, and trained models publicly available.
>
---
#### [new 012] SPRINT: Efficient Spectral Priors for Humanoid Athletic Sprints
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人运动控制任务，旨在解决 humanoid 运动数据不足和稳定性问题。通过引入频率自适应频谱先验，生成高效运动轨迹，实现高速奔跑与平滑步态转换。**

- **链接: [https://arxiv.org/pdf/2605.28549](https://arxiv.org/pdf/2605.28549)**

> **作者:** Yantong Wei; Kaihong Huang; Hainan Pan; Jiawei Luo; Jiawei Zhou; Ziyan Mai; Zhiwen Zeng; Yaonan Wang; Huimin Lu
>
> **摘要:** The pursuit of humanoid athletic sprints is hindered by a scarcity of humanoid-viable kinematic reference data and the inability of existing frameworks to maintain stability during sprints. To overcome these limitations, we introduce SPRINT, a novel framework driven by efficient, frequency-adaptive spectral priors. By characterizing the fundamental periodicity of human locomotion in the frequency domain using a reference library of five discrete motion sequences, these priors generate kinematically feasible joint trajectories across a broad velocity spectrum, successfully extrapolating to speeds that exceed the reference distribution. Guided by these pretrained priors, the SPRINT policy achieves zero-shot sim-to-real transfer in field experiments on the Unitree G1 platform, reaching a peak sprinting velocity of 6 m/s and demonstrating seamless gait transitions while preserving biomimetic naturalness. Ultimately, this work establishes frequency-adaptive spectral priors as a highly data-efficient foundation for humanoid athletic sprints. The project page is available at this https URL.
>
---
#### [new 013] ProgVLA: Progress-Aware Robot Manipulation Skill Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出ProgVLA，解决机器人操作中长序列处理与任务进度感知问题。通过多模态编码和进度头设计，提升复杂任务成功率。**

- **链接: [https://arxiv.org/pdf/2605.28231](https://arxiv.org/pdf/2605.28231)**

> **作者:** Seungsu Kim; Jinyoung Choi; Seungmin Baek; Jean-Michel Renders
>
> **摘要:** We present ProgVLA, a compact vision-language-action (VLA) model designed for reliable robot manipulation under tight compute and memory budgets. The model specifically focuses on efficiently processing long multi-modal sequences by maintaining an explicit representation of task progress over extended horizons. To this end, ProgVLA integrates two key components. First, a multi-modal encoder with a two-stage Perceiver resampling scheme compresses variable-length visual, language, and proprioceptive streams into a fixed set of control-ready context tokens, substantially reducing sequence length while preserving cross-modal grounding. Second, an auxiliary set of progress heads is trained with offline reinforcement learning (RL) objectives to jointly learn critics over normalized remaining-horizon targets. This provides the policy with an internal estimate of task progress and enables advantage- and success-weighted flow-matching imitation learning. On two well-established multi-task robot manipulation benchmarks, a 0.1B-parameter ProgVLA model reaches success rates that are competitive with, and on long-horizon and harder task tiers exceed, substantially larger pretrained baselines. Ablations indicate that the learned context resampler and task-adaptive visual fine-tuning are the largest single contributors, while progress-aware training provides a consistent additional gain that is concentrated on long-horizon and multi-object tasks. We further validate the approach in real-world toy-kitchen environments.
>
---
#### [new 014] HumanoidMimicGen: Data Generation for Loco-Manipulation via Whole-Body Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人模仿学习任务，旨在解决人类机器人行走与操作数据生成难题。通过生成高质量数据提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.27724](https://arxiv.org/pdf/2605.27724)**

> **作者:** Kevin Lin; Ajay Mandlekar; Caelan Reed Garrett; Nikita Chernyadev; Yu Fang; Runyu Ding; Yuqi Xie; Justin Tran; Linxi Fan; Yuke Zhu
>
> **备注:** website: this https URL
>
> **摘要:** Imitation learning is a promising approach for training humanoid robots to both walk and manipulate, but it requires a large number of demonstrations, which are time-intensive and difficult to collect via teleoperation. Existing data-generation algorithms can automatically synthesize demonstrations for manipulators, but they are ineffective on humanoids because their high-dimensional composite action spaces involve arms, legs, and torsos. We present HumanoidMimicGen, a method for generating humanoid legged loco-manipulation data. Our method adapts contact-rich whole-body skills from a handful of source demonstrations to new states, generalizing across changes in object pose. By interleaving these single- and dual-arm skills with whole-body locomotion and manipulation planning, the method generates stable, collision-free data across diverse scenes and layouts. To evaluate our approach, we introduce a new simulated loco-manipulation benchmark containing nine diverse tasks that test humanoid loco-manipulation capabilities. There, we demonstrate that HumanoidMimicGen automatically generates large datasets for imitation learning and enables a systematic study of how data generation and policy learning decisions impact model performance. We show that whole-body visuomotor policies co-trained with data generated by HumanoidMimicGen outperform those trained only on real-world data by 20%.
>
---
#### [new 015] Mag-VLA: Vision-Language-Action Model for Bimanual Magnetically Actuated Microrobot Manipulation
- **分类: cs.RO**

- **简介: 该论文提出Mag-VLA模型，解决双臂磁控微机器人操作问题，通过视觉-语言-动作框架实现精准控制。**

- **链接: [https://arxiv.org/pdf/2605.28486](https://arxiv.org/pdf/2605.28486)**

> **作者:** Yongchen Wang; Kangyi Lu; Lan Wei; Dandan Zhang
>
> **备注:** Accepted by 2026 MARSS
>
> **摘要:** Magnetically actuated microrobots have been used as wireless, non-contact manipulation tools at microscales, making them promising for minimally invasive applications. However, their control remains challenging due to indirect actuation, limited sensing, and nonlinear magnetic interactions. In this work, we propose Mag-VLA, a vision-language-action (VLA) model for dexterous magnetic microrobot manipulation using two robotic arms with mounted magnets for dynamic magnetic-field construction. Bimanual coordination enables capabilities such as microrobot reorientation that are difficult or infeasible with a single arm, but it also introduces coupled control challenges, as the policy must generate coordinated trajectories for both actuators within a shared workspace. Our framework adapts a Qwen2.5-VL-7B backbone using Low-Rank Adaptation (LoRA) to process visual observations and language instructions for action prediction. To capture task progression, we introduce a motion-aware phase classifier and a phase-conditioned Action Chunking Transformer (ACT) decoder for temporally coherent multi-step control. We further construct a teleoperated magnetic microrobot manipulation dataset covering three task configurations. Ablation studies show that the ACT-based decoder substantially outperforms alternative generative action heads. In real-robot experiments, Mag-VLA achieves a 90% approach success rate across all tasks and transport success rates of 80%, 70%, and 50% as task difficulty increases. These results demonstrate that hierarchical VLA modeling provides a promising framework for magnetic microrobot manipulation.
>
---
#### [new 016] Turning Video Models into Generalist Robot Policies
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决跨身体结构的通用机器人策略问题。通过解耦视频规划与逆动力学模型，实现高效、可扩展的零样本控制。**

- **链接: [https://arxiv.org/pdf/2605.27817](https://arxiv.org/pdf/2605.27817)**

> **作者:** Sizhe Lester Li; Evan Kim; Xingjian Bai; Tong Zhao; Tao Pang; Max Simchowitz; Vincent Sitzmann
>
> **备注:** project page: this https URL
>
> **摘要:** Video generative models have emerged as a promising robotics backbone, capable of generating videos that depict the completion of complex tasks across embodiments and environments. Recent work proposes robot foundation models that jointly predict future observations and actions by finetuning video models with action-labeled data. In this paper, we test the limits of an alternative approach: leave the video planner as-is while training an embodiment-specific inverse dynamics model (IDM). This decoupling offers several natural benefits: the video planner remains embodiment-agnostic, different video models can be interchanged easily without re-training the IDM, and the IDM can be independently trained with readily available self-play data. We present a closed-loop, video-to-action policy that combines an action-free video world model with a carefully-designed IDM based on the robot embodiment Jacobian. We demonstrate that our IDM design is both data-efficient and scalable to high-dimensional action spaces. Our policy, which we coin the Video-to-Embodied Robot Action Model (VERA), achieves strong performance across simulated and real-world benchmarks, including zero-shot Panda arm manipulation and 16-DoF Allegro-hand dexterous cube re-orientation. The same video planner can be used across multiple embodiments by pairing it with different embodiment-specific IDMs. Our results show that decoupled video planning plus faithful video-to-action translation is a viable alternative route towards zero-shot, cross-embodiment, and generalizable robot control. More results are available on our project website: this https URL.
>
---
#### [new 017] Whose Is This?: Context-Aware Object Ownership Inference with Uncertainty-Guided Questioning
- **分类: cs.RO**

- **简介: 该论文属于对象所有权推断任务，旨在解决服务机器人在不确定环境下正确识别物品归属的问题。通过结合上下文和不确定性引导的交互方法，提升推断准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.28087](https://arxiv.org/pdf/2605.28087)**

> **作者:** Saki Hashimoto; Akira Taniguchi; Shoichi Hasegawa; Yoshinobu Hagiwara; Tadahiro Taniguchi
>
> **备注:** Under review in Advanced Robotics. Project page is this https URL
>
> **摘要:** Service robots must infer object ownership to correctly interpret instructions such as "bring me my cup." However, ownership is a latent attribute that cannot be directly observed, and existing methods often rely on limited cues such as recent usage, making them unreliable in scenarios such as temporary sharing. We propose a framework for context-aware ownership inference with uncertainty-guided interaction (COIN). The method integrates user background information and object usage history using a large language model (LLM) to estimate ownership scores. To handle uncertainty, we apply conformal prediction to construct a set of plausible owners and selectively generate user queries when the prediction is uncertain. Experiments in a simulated home environment show that the proposed method consistently outperforms baseline approaches, achieving a Subset Accuracy of 0.988 and a Mean Jaccard index of 0.991. The method also maintains high performance in scenarios involving temporary use and shared ownership. The results demonstrate that combining contextual reasoning with uncertainty-aware interaction improves both estimation accuracy and robustness. The project page is available at this https URL.
>
---
#### [new 018] Frequency-Guided Action Diffusion via Sub-Frequency Manifold Traversal
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人视觉-运动策略学习任务，旨在解决模仿学习中因高频率噪声导致的行动不平滑问题。提出FGO方法通过频域引导提升动作平滑性和时间一致性。**

- **链接: [https://arxiv.org/pdf/2605.27919](https://arxiv.org/pdf/2605.27919)**

> **作者:** Junlin Wang
>
> **备注:** A preprint version of FGO
>
> **摘要:** Learning visuomotor policies via behavior cloning typically involves mimicking expert demonstrations collected by human operators. However, natural human demonstrations inherently contain high-frequency noise, such as intermittent jerks, pauses, and action jitter. Training policies to directly imitate these raw trajectories inevitably causes the model to inherit these suboptimal behaviors. This pathology is particularly pronounced in diffusion-based policies, where iterative denoising steps can inadvertently amplify high-frequency artifacts at the expense of meaningful fine-grained details. To address these limitations, we present a novel frequency-based algorithm that enables implicit spectral maneuvering and smooth action generation. Our method, Frequency Guidance Operator (FGO), steers the generation process of diffusion polices by progressively driving the noisy samples through intermediate sub-frequency manifolds with expanding spectral bands. Validated on 15 robotic manipulation tasks from 5 benchmarks, FGO achieves superior performance in enhancing action smoothness and temporal consistency while preserving the details necessary for successful task execution. Project website: this https URL
>
---
#### [new 019] A Digital Twin Framework for Virtual Visuo-Haptic Teleoperation of Complex-Shaped Optical Microrobots
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决复杂形状光学微机器人远程操作中的力反馈问题。通过构建数字孪生框架，实现虚拟视觉-触觉反馈，提升操作精度与成功率。**

- **链接: [https://arxiv.org/pdf/2605.28448](https://arxiv.org/pdf/2605.28448)**

> **作者:** Zongcai Tan; Lan Wei; Dandan Zhang
>
> **备注:** Accepted by 2026 MARSS
>
> **摘要:** Optical tweezers (OT) provide piconewton-scale manipulation for delicate biomedical tasks, where visuo-haptic feedback can improve operator awareness by conveying interaction-force cues and trap-stability information. However, visuo-haptic teleoperation frameworks for complex-shaped optical microrobots remain underdeveloped, particularly in multi-trap manipulation scenarios. This paper presents a digital twin framework for virtual visuo-haptic teleoperation of complex-shaped OT-driven microrobots. The framework integrates a digital twin environment, image-based pose and depth estimation, microrobot motion simulation, and model-based haptic rendering within a Robot Operating System (ROS)-connected bimanual teleoperation system. For force modeling, we combine a Multi-Sphere Distributed Manipulation (MSDM) model with optical-force estimation from the Optical Tweezers Toolbox, enabling simulator-driven visuo-haptic feedback. The framework reproduces representative microrobot motion trends and provides haptic force rendering that is numerically consistent with the fitted optical-force model. In simulated cell-delivery tasks, haptic feedback reduced the standard deviations of the contact-force metric and the microrobot-to-trap-center distance metric by 53.2% and 55.2%, respectively, and improved task success from 30% to 80%. These results demonstrate the framework's effectiveness for evaluating visuo-haptic teleoperation strategies for complex-shaped optical microrobots.
>
---
#### [new 020] SCALE-COMM: Shared, Contrastively-Aligned Latent Embeddings for MARL Communication
- **分类: cs.RO**

- **简介: 该论文属于多智能体强化学习（MARL）任务，旨在解决通信协议不稳定、语义不清和策略优化干扰问题。提出SCALE-COMM框架，通过对比对齐的潜在嵌入实现高效稳定的通信表示。**

- **链接: [https://arxiv.org/pdf/2605.27532](https://arxiv.org/pdf/2605.27532)**

> **作者:** Mahmoud Abouelyazid; Eman Hammad
>
> **备注:** IEEE IV 2026
>
> **摘要:** Emergent communication enables partially observant Autonomous Mobile Robots (AMRs) to coordinate effectively in decentralized multi-agent reinforcement learning (MARL) settings. However, existing approaches often struggle with unstable communication protocols, ungrounded message semantics, and interference between communication learning and policy optimization, leading to degraded coordination over time. We propose SCALE-COMM (Shared, Contrastively-Aligned Latent Embeddings for COMMunication), a self-supervised framework for learning compact, stable, and policy-relevant communication representations. SCALE-COMM decouples communication learning from policy optimization by training low-dimensional latent messages that capture task-relevant planning and traffic information, while enforcing consistency across agents and time. Across standard MARL benchmarks and a realistic warehouse coordination task, SCALE-COMM consistently outperforms existing communication frameworks in both representation quality and task performance. The learned communication space yields improved stability, sample efficiency, and throughput under policy fine-tuning, demonstrating the effectiveness of representation-driven communication for scalable multi-agent coordination.
>
---
#### [new 021] An Operator-Based Approach to STL
- **分类: cs.RO**

- **简介: 该论文属于形式化验证任务，旨在解决STL复杂公式的验证与控制合成问题。提出基于算子的STL方法，构建理论框架并实现在线控制。**

- **链接: [https://arxiv.org/pdf/2605.28092](https://arxiv.org/pdf/2605.28092)**

> **作者:** Panagiotis Rousseas; Dimos V. Dimarogonas
>
> **摘要:** Signal Temporal Logic (STL), has recently seen extensive development, owing to its rich expressivenes for autonomous planning and control. Nevertheless, existing verification and control synthesis methods are limited with respect to the complexity and degree of nesting of the formulae. In this work, we propose a novel approach to STL based on an operator acting on reachability value functions. This constitutes a new theoretical framework for handling complex multi-nested formulae while at the same time providing tools for on-line control synthesis. In contrast to focusing on the design of STL-based reachability (or control barrier) functions, we develop operator-based nesting rules directly. Our method's expressiveness is demonstrated both theoretically, where necessary and sufficient conditions for STL formula satisfaction are extracted, as well as in simulations with complex fragments.
>
---
#### [new 022] POINav: Benchmarking and Enhancing Final-Meters Arrival in Real-World Vision-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，旨在解决真实场景中精准到达POI的“最后米”问题。构建了POINav-Bench基准和POINav-Dataset，并提出脑-行动框架提升导航精度。**

- **链接: [https://arxiv.org/pdf/2605.28237](https://arxiv.org/pdf/2605.28237)**

> **作者:** Ruiyan Gong; Meisheng Zhang; Yuxiang Zhao; Mingchao Sun; Yanfen Shen; Zedong Chu; Zhining Gu; Wei Guo; Xiaolong Cheng; Qiming Li; Kangning Niu; Yanqing Zhu; Xiaolong Wu; Tianlun Li; Mu Xu
>
> **备注:** 25 pages, 9 figures
>
> **摘要:** Real-world navigation is fundamentally driven by Points of Interest (POIs), yet reaching a precise POI remains a critical "final-meters" challenge. Existing Vision-Language Navigation (VLN) benchmarks of POI-goal navigation often suffer from coarse granularity or significant sim-to-real gaps due to generated scene. To bridge this gap, we present POINav-Bench, the first benchmark designed for closed-loop evaluation of real-world POI-goal navigation. It comprises 11 commercial areas reconstructed from real-world captures using 3D Gaussian Splatting (3DGS), covering 126,398 $m^{2}$ in total and spanning 163 distinct POIs. With traversability-aware annotations and reference trajectories, POINav-Bench enables high-fidelity evaluation of navigation agents in realistic, POI-rich real-world environments. Building on this, we propose the POINav Brain-Action Framework where a Brain module performs POI-grounded reasoning to guide an Action module in predicting continuous waypoints for real-world execution. We further curate the POINav-Dataset, containing 70K real-world signage-entrance pairs. Experiments show that our framework provides a viable path toward refining real-world POI-goal navigation.
>
---
#### [new 023] Design of a Real-time Asynchronous Monocular Odometry for Planetary Exploration
- **分类: cs.RO**

- **简介: 该论文属于视觉里程计任务，旨在解决行星探测中实时、异步、高动态光照下的运动估计问题。工作包括设计基于事件相机和ESKF的异步单目里程计系统。**

- **链接: [https://arxiv.org/pdf/2605.27661](https://arxiv.org/pdf/2605.27661)**

> **作者:** Benat Inigo; Florian Steidle; Wolfgang Stuerzl
>
> **摘要:** We describe our preliminary design of a real-time asynchronous event-based monocular odometry for planetary exploration. Operating under strict computational constraints, planetary rovers frequently encounter complex, unpredictable environments that demand high-speed sensing and robustness to high dynamic range (HDR) lighting. Event cameras address these needs by reporting asynchronous, pixel-wise brightness changes with microsecond resolution, significantly reducing data bandwidth while maintaining robustness in extreme lighting conditions. We propose an approach based on an Error-State Kalman Filter (ESKF) that leverages this asynchronous event stream to continuously estimate camera ego-motion. The camera state is updated with every tracked position output generated by RATE, a real-time asynchronous feature tracker.
>
---
#### [new 024] Natural Locomotion: Principle and Method
- **分类: cs.RO; eess.SY; math.DS**

- **简介: 该论文属于机器人学任务，研究如何通过被动动力和环境交互实现高效运动。提出自然运动原理，解决如何设计支持自然运动流形的被动结构问题。**

- **链接: [https://arxiv.org/pdf/2605.28254](https://arxiv.org/pdf/2605.28254)**

> **作者:** Mirado Mortel; Luc Jaulin; Lionel Lapierre; Simon Rohou
>
> **备注:** Preprint. 20 pages, 7 figures
>
> **摘要:** Robotic locomotion can become efficient when mechanisms exploit passive dynamics, compliance, and resonance rather than track prescribed trajectories. This paper formulates natural locomotion as an exchange principle for systems whose motion is mediated by environmental constraints or interactions. A motion is natural when an internal oscillator returns periodically, the body pose drifts, and the mean Propulsion--Oscillator Exchange power (POE power) vanishes over one cycle. The selected family is a Natural Locomotion Manifold (NLM). We develop the conservative realization of this principle for continuous ideal environmental constraints: the constraints do no external work, total mechanical energy is conserved, and zero mean POE power is an internal exchange with the environment-mediated propulsive channel, not external energy input. The method is a closed/open construction. The propulsive channel is first closed to reveal an effective internal oscillator, organized by scalar action-angle structure in one effective degree of freedom or by nonlinear modal sectors in several degrees of freedom. The channel is then reopened, pose is reconstructed, and accepted cycles must preserve internal recurrence and zero mean POE power. We demonstrate the principle on two ideal nonholonomic no-slip systems: a Chaplygin-sleigh / pendulum-driven car and a three-body extension. In the scalar case, POE closure is equivalent to the missing internal return condition, giving a theorem-backed computation of the NLM family. In the multi-degree case, POE closure remains necessary but must be completed by modal identity, internal return, dynamics consistency, same fixed passive architecture, and nonzero displacement. Natural locomotion becomes a design question: which passive architectures support no, one, or several certified NLM families?
>
---
#### [new 025] How Should We Teach Robots? A Comparison of Kinesthetic, Joystick, and Gesture-Based Teaching
- **分类: cs.RO**

- **简介: 该论文属于机器人教学任务，比较了三种教学方式的优劣，旨在提升机器人操作的效率与准确性。通过用户实验评估不同方法在任务中的表现。**

- **链接: [https://arxiv.org/pdf/2605.28033](https://arxiv.org/pdf/2605.28033)**

> **作者:** Petr Vanc; Jan Kristof Behrens; Václav Hlaváč; Karla Stepanova
>
> **备注:** 7 pages, 3 figures, 3 tables, presented at Cognition and Artificial Life (CAL/KUZ) 2026 conference at Chateau Trest
>
> **摘要:** Instructing robots from demonstrations can be done through different teaching modalities, each with different usability and performance trade-offs. This paper compares kinesthetic guidance, joystick teleoperation, and hand gestures in a user study with eight participants. We evaluate replay success, modified NASA-TLX workload, and common teaching errors across three manipulation tasks. Kinesthetic guidance produced the shortest demonstrations, lowest workload, and highest success on the more orientation-sensitive and contact-rich tasks. Joystick teleoperation performed best on simple peg picking. Hand-gesture teaching, although less reliable overall, performed better than expected and in some cases achieved results comparable to kinesthetic guidance.
>
---
#### [new 026] Agentic Language-to-Objective Synthesis for Optofluidic Assembly
- **分类: cs.RO; physics.optics**

- **简介: 该论文属于微装配任务，解决将用户意图转化为可执行目标的问题。通过构建一个基于大语言模型的智能流水线，实现自然语言控制的光流体微尺度组装。**

- **链接: [https://arxiv.org/pdf/2605.27643](https://arxiv.org/pdf/2605.27643)**

> **作者:** Ivan Saraev; Elena Erben; Weida Liao; Fan Nan; Gerhard Neumann; Eric Lauga; Moritz Kreysing
>
> **备注:** 21 pages, 5 figures
>
> **摘要:** Light-based advanced manufacturing increasingly requires programmable, closed-loop tools that translate human design intent into executable operations at small length scales. Yet a key bottleneck persists across robotic and manufacturing modalities: turning user intent into machine-readable objectives that are reliably executable. While micro-robotics offers versatile manipulation via optical actuation of fluids, mathematically tractable goal specification remains manual and hard to reuse. Here, we introduce Speak-to-Objective, a modular agentic pipeline that uses a conditioned Large Language Model (LLM) to translate spoken or written commands into fully differentiable objective functions for assembling microparticles in a constraint-aware inverse solver (SLSQP) and on an experimental optofluidic platform. The approach employs a compact loop - perceive -> compose -> propose -> act -> report & learn - that treats the objective as the interface between intent and actuation, separating what to assemble or pattern from how to actuate, while learning from user feedback. The pipeline composes geometry, spacing, and assignment/topology terms to generate robust descriptive objectives that assemble from partial traces and recover after perturbations, as well as explicit objectives for precise placement, all in an actuator-agnostic fashion. Using laser-induced thermoviscous flows as the physical actuation modality, we demonstrate natural-language-programmable, light-based microscale assembly of particle patterns in a microfluidic environment. Beyond its immediate impact on programmable microassembly, and using laser-induced optofluidic actuation as a reduced-complexity experimental platform, our work points toward self-driving, AI-assisted optical manufacturing platforms in which natural language, differentiable objectives, and laser-based actuation are coupled into a reusable digital workflow.
>
---
#### [new 027] Accelerating Robot Path Planning via Connectivity-Preserving Region Proposal Network
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，旨在解决搜索空间大导致的延迟问题。提出CP-RPN模型，通过分割预测连通区域，压缩搜索空间，提升规划效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.28362](https://arxiv.org/pdf/2605.28362)**

> **作者:** Zhanzheng Ma; Cancan Zhao; Shuai Zhang; Bo Ouyang
>
> **摘要:** Mobile robot path planning methods are often constrained by vast search spaces, resulting in latency in samplingbased algorithms. Learning-based approaches frequently suffer from local region fragmentation and global topological inconsistency. To tackle the problem, we present the Connectivity- Preserving Region Proposal Network (CP-RPN), a segmentationguided model designed to predict compact and topologically connected candidate regions, significantly compressing the search space. Specifically, we design a segmentation model that leverages a Deformable Attention Transformer (DAT) to capture long-range dependencies for global connectivity, with a Deconvolutional decoder to preserve fine-grained spatial details. To guarantee the connectivity of the predicted mask, we design a composite loss function that combines Cross-Entropy loss for pixelwise supervision, a Connectivity-Aware loss to enhance local coherence, and a Topological Continuity loss based on persistent homology to enforce global connectivity. Building on these highconnectivity corridor-like regions, the Voronoi diagram is used to plan the path, backed by a local A* fallback mechanism to ensure robustness. Experimental results demonstrate that CPRPN reduces the candidate region size by over 60.13% compared to the MPT baseline and achieves deterministic low-latency planning (avg. 0.11s) with a 99.60% success rate, outperforming traditional sampling-based algorithms in stability.
>
---
#### [new 028] S-Cheetah: A Novel Quadrupedal Robot with a 3-DOF Active Spine Learning Agile Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人学领域，旨在解决四足机器人敏捷运动能力不足的问题。通过设计具有3-DOF主动脊柱的S-Cheetah，并采用强化学习提升其运动性能。**

- **链接: [https://arxiv.org/pdf/2605.27909](https://arxiv.org/pdf/2605.27909)**

> **作者:** Zimu Li; Weibang Bai
>
> **备注:** Project website: this https URL
>
> **摘要:** The biological spine of quadrupeds enables sagittal flexion/extension, lateral bending, and axial rotation, playing a crucial role in highly agile and dexterous locomotion. While numerous studies have integrated active spinal joints into quadrupedal robots to enhance agility, most designs simplify control complexity by reducing spinal degrees of freedom (DOF), failing to achieve the spatial tri-axial rotation characteristic of biological spines. Consequently, replicating a multi-DOF biomimetic spine and effectively leveraging it to empower the agile locomotion of quadrupedal robots remains a significant research challenge. In this study, we present S-Cheetah, a quadrupedal robot featuring a 3-DOF bio-inspired serial active spine capable of biomimetic spatial tri-axial rotation. To empower the robot to fully utilize this active spine, we developed a specialized reinforcement learning framework to actively promote the engagement of the introduced spine and maximize the robot's locomotive capabilities by integrating an acceleration curriculum learning strategy with tailored reward functions, such as a gallop gait reward, a spine undulation reward, and a spine steering reward. Experimental results demonstrate that S-Cheetah can achieve a peak speed of 6.9 m/s using the rotary G2 gallop gait and an in-place turning rate of 7.2 rad/s. Besides, the system exhibits an emergent, feline-inspired aerial self-righting capability, allowing it to land stably on four feet from arbitrary orientations during free fall. Finally, through extensive evaluations across diverse locomotion tasks, we prove that the introduction of the proposed 3-DOF spine comprehensively enhances the locomotive agility of quadrupedal robots. Project website: this http URL
>
---
#### [new 029] Trinity: Unifying Class-Agnostic Terrain and Semantic Segmentation for Unstructured Outdoor Environments by Leveraging Synthetic Data
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出Trinity模型，解决无结构户外环境中地形理解问题。通过联合进行语义分割和与类别无关的地形分割，提升机器人导航能力。**

- **链接: [https://arxiv.org/pdf/2605.27644](https://arxiv.org/pdf/2605.27644)**

> **作者:** Marcus G Müller; Wout Boerdijk; Maximilian Durner; Riccardo Giubilato; Abel Gawel; Wolfgang Stürzl; Roland Siegwart; Rudolph Triebel
>
> **摘要:** Terrain understanding is fundamental for mobile robots operating in unstructured outdoor environments. Existing vision-based traversability estimation methods rely on robot-specific annotations or semantic class mappings, limiting transferability across platforms and requiring costly re-annotation when robot capabilities change, while standard semantic segmentation methods only focus on specific predefined classes, which do not capture the variety of terrains. In this work, we propose a transformer-based architecture that jointly performs class-specific semantic segmentation and class-agnostic terrain segmentation within a unified network, called Trinity. Terrain regions are segmented based solely on visual appearance, without predefined semantic labels or robot-dependent traversability scores. This formulation enables the learning of robot-agnostic visual terrain priors that can be combined with robot-specific experience for downstream tasks such as traversability estimation, visual odometry, and mission planning. To enable large-scale training with diverse terrain appearances, we extend the OAISYS simulator and introduce RUGDSynth, a synthetic dataset inspired by RUGD with class-agnostic terrain samples. Furthermore, we present the EXTerra Dataset, providing real-world images annotated with both class-specific and class-agnostic terrain labels. Experiments demonstrate the feasibility of the proposed task and the effectiveness of our joint segmentation approach in complex outdoor environments. Code and datasets will be released with this publication (after review).
>
---
#### [new 030] SANTS: A State-Adaptive Scheduler for World Action Models
- **分类: cs.RO**

- **简介: 该论文提出SANTS，用于视频到动作的扩散策略，解决WAMs中固定去噪深度导致的冗余计算和性能下降问题。通过状态自适应选择去噪点，提升任务成功率并降低延迟。**

- **链接: [https://arxiv.org/pdf/2605.27947](https://arxiv.org/pdf/2605.27947)**

> **作者:** Yirui Sun; Guangyu Zhuge; Keliang Liu; Jie Gu; Xinyu Bing; Zhongxue Gan; Chunxu Tian
>
> **备注:** 17 pages, 5 figures, 8 tables. Project page: this https URL
>
> **摘要:** World Action Models (WAMs) improve robot manipulation by using video-based future representations to condition action generation. In pixel-space WAMs, however, the best action condition is not necessarily the fully denoised video. Controlled denoising-depth scans show that video refinement can reduce action error up to a state-dependent point, after which the gain may saturate or even reverse when late predictions become less action-relevant or physically unreliable. This suggests that action generation should use a state-dependent point along the video noise trajectory rather than a fixed terminal denoising depth. We introduce State-Adaptive Noise Trajectory Scheduler (SANTS), a lightweight scheduler for video-to-action diffusion policies. At each video decision point, SANTS reads the current video-state representation and noise level, then jointly predicts a cumulative stopping hazard and a relative noise-progression ratio. SANTS is post-trained with a path-level reward computed after the frozen action branch generates the final action chunk, so the scheduler is optimized for downstream action quality rather than intermediate video fidelity, while redundant video-state updates are explicitly penalized. Experiments show that SANTS reaches \(94.4\%\) overall success on RoboTwin 2.0 and \(73.1\%\) average success across seven real-robot tasks, while reducing latency by \(81.7\%\) and \(79.0\%\) relative to full video denoising, respectively. These results indicate that adaptive selection along the video noise trajectory can preserve the control benefits of WAM-style future reasoning while removing much of its redundant inference cost.
>
---
#### [new 031] Visualizing Latent Phase Structures in Locomotion Policies: A Multi-Environment Study with Temporal Feature Extension
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于强化学习领域，旨在解决运动策略中潜在相位结构可视化的问题。通过扩展特征并改进聚类方法，更准确地识别运动相位结构。**

- **链接: [https://arxiv.org/pdf/2605.28186](https://arxiv.org/pdf/2605.28186)**

> **作者:** Daisuke Yasui; Toshitaka Matuki; Hiroshi Sato
>
> **摘要:** Deep reinforcement learning (DRL) has been shown to achieve high performance on locomotion control tasks in MuJoCo benchmarks such as HalfCheetah, Ant, and Walker2D. However, visualizing the motion structures internally obtained by a trained policy function implemented as a deep neural network remains challenging. It is known from biomechanics and related fields that locomotion control is realized through the repetition of motion phases such as the stance phase and swing phase. In this study, we propose a framework for uncovering latent motion phase structures from trajectories generated by locomotion control policies through interaction with the environment. The proposed method extends the clustering features from state observations alone to augmented features including actions, next states, and next actions, and introduces a method for determining the number of clusters that suppresses self-transitions. Applying the proposed method to three environments -- Ant-v5, HalfCheetah-v5, and Walker2D-v5 -- we successfully identified phase structures with clearer and more regular transition rules than those obtained by the existing method.
>
---
#### [new 032] AURA: Asymptotically Optimal Uncertainty-Robust Replanning Algorithm for Kinodynamic Systems
- **分类: cs.RO**

- **简介: 该论文属于运动规划任务，解决高维系统在不确定性下的轨迹跟踪问题。提出AURA框架，实现在线最优重规划，提升轨迹质量和执行精度。**

- **链接: [https://arxiv.org/pdf/2605.27699](https://arxiv.org/pdf/2605.27699)**

> **作者:** Seyedali Golestaneh; Zhuoyun Zhong; Donghyung Lee; Constantinos Chamzas
>
> **摘要:** Sampling-based motion planners offer a practical and scalable approach to kinodynamic motion planning, notably for high-dimensional, underactuated, or non-holonomic systems. However, these planners are typically used offline, requiring execution to begin only after the trajectory has been computed. In addition, the planned trajectory may not be accurately tracked in the presence of motion uncertainty, leading to deviations from the nominal solution. In this work, these limitations were addressed within a unified framework, \method, an asymptotically-optimal meta-planner framework that improves both path quality and tracking performance during execution. In addition to the main execution thread, this framework comprises a replanning method that continuously explores the state space and refines the trajectory during execution, and an optimization process that refines future control inputs to reduce tracking error. Together, these components enable \method to leverage asymptotically optimal planning online while improving execution accuracy under uncertainty. The proposed approach is evaluated in both simulation and real-world environments across multiple systems, demonstrating consistent improvements in trajectory quality, tracking accuracy, and overall performance compared with baseline methods.
>
---
#### [new 033] What Frozen VLAs Already Know About Success: A Probing Study of Value-Like Structure in Foundation Robot Policies
- **分类: cs.RO**

- **简介: 该论文研究冻结的视觉-语言-动作模型是否包含成功预测信息。任务是识别模型中隐含的成功价值结构。通过线性探测器验证，发现模型特征可有效预测成功，提升行动选择效果。**

- **链接: [https://arxiv.org/pdf/2605.28527](https://arxiv.org/pdf/2605.28527)**

> **作者:** Jiachen Zhang; Junnan Nie; Junyi Lao; Wei Cheng; Chenghao Liu; Jiaxin Jiang; Songfang Huang
>
> **备注:** 14 pages, 1 figure, 11 tables. Equal contribution: Jiachen Zhang, Junnan Nie, and Junyi Lao. Corresponding author: Songfang Huang. Preprint
>
> **摘要:** Vision--language--action (VLA) policies are trained to imitate actions; their loss never asks them to estimate reward, progress, or future success. Their frozen representations nevertheless carry such information, and it can be read out and used to guide action choice without retraining the policy. From mixed successful and failed manipulation trajectories on LIBERO-Goal, we recover Monte-Carlo outcome targets using lightweight linear probes on frozen features. The targets are consistently predictable from OpenVLA, Pi0.5, DINOv2, and CLIP features, and substantially less so from baselines built on progress, time-to-go, task identity, or proprioception. To rule out task and temporal shortcuts, we evaluate the probes under same-task, same-timestep matched comparisons: Pi0.5 probes still reach roughly 92% pairwise ordering accuracy, while label-shuffled controls stay at chance. Used as a test-time selector over sampled Pi0.5 action prefixes, the same probe turns this offline finding into behavior: on push-plate, success rises from 26.7% under greedy decoding to 44.3%, with a second positive case on wine-rack. The gains are not universal and require additional inference compute, but the underlying finding is clean: frozen VLAs already encode information about success that their imitation objective never explicitly demands.
>
---
#### [new 034] PrimitiveVLA: Learning Reusable Motion Primitives for Efficient and Generalizable Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型数据效率低和泛化能力差的问题。通过分解任务为可重用动作基元，提升模型的通用性和效率。**

- **链接: [https://arxiv.org/pdf/2605.28634](https://arxiv.org/pdf/2605.28634)**

> **作者:** Yutai Li; Shaohui Peng; Jiaming Guo; Di Huang; Zihao Zhang; Yuxuan Guo; Yunkai Gao; Siming Lan; Ling Li; Xing Hu; Yunji Chen
>
> **摘要:** Vision-Language-Action (VLA) models offer a promising paradigm for generalist robotic policies, yet their adaptation is hindered by data inefficiency and poor generalization. We argue that these bottlenecks stem from the prevailing Direct Instruction-to-Control Mapping, which forces models to memorize monolithic trajectories rather than reusable motion patterns, i.e., primitives. We propose PrimitiveVLA, a framework that shifts this paradigm toward a Primitive-Centric Disassemble & Assemble paradigm. Supported by a shared Multimodal Canonical Representation (MCR), PrimitiveVLA unifies two phases: (1) Fine-tuning-phase Disassembly, which uses an automated pipeline to disassemble demonstrations into reusable primitives; and (2) Inference-phase Assembly, which employs a VLM-based planner and an LLM-generated switch module for robust closed-loop execution. By disassembling tasks into reusable primitives, PrimitiveVLA enables VLA models to learn invariant motion patterns instead of task-specific trajectories. Extensive experiments show that our framework improves data efficiency and achieves superior zero-shot generalization across unseen and long-horizon tasks.
>
---
#### [new 035] Natural Functional Gradients for Smooth Trajectory Optimization
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决复杂环境中生成无碰撞、平滑轨迹的问题。提出基于自然函数梯度的优化框架，通过几何感知更新提升轨迹可行性与平滑性。**

- **链接: [https://arxiv.org/pdf/2605.28202](https://arxiv.org/pdf/2605.28202)**

> **作者:** Kisang Park; Chanwoo Kim; Kyungjae Lee; Sungjoon Choi
>
> **摘要:** Generating collision-free and smooth motions remains a central challenge in robotic manipulation, particularly in cluttered environments and narrow passages where feasible regions are highly constrained and fragmented. We propose a trajectory optimization framework that performs geometry-aware updates directly in function space using natural functional gradients. The method optimizes a Gaussian-smoothed surrogate objective that regularizes the optimization landscape through smooth trajectory perturbations while preserving trajectory-level structure. Because the updates are defined intrinsically in function space, trajectory regularity can be controlled independently of a particular time discretization. We derive a practical Monte-Carlo estimator of the natural functional gradient that requires only black-box trajectory evaluations, making the method applicable when analytic gradients are unavailable or unreliable due to collision checking and contact-rich simulation. Experiments on constrained robotic manipulation tasks demonstrate that the proposed method improves trajectory feasibility and produces smoother motions than representative planning and trajectory optimization baselines in environments with narrow geometric clearances. Additional results, videos, and implementation details are available at the project page: this https URL
>
---
#### [new 036] ICAN-Deploy: Identity-Stable Canary Deployment for Safety-Critical Embodied Agents
- **分类: cs.RO**

- **简介: 该论文提出ICAN-Deploy，解决安全关键实体代理在金丝雀部署中身份漂移问题，通过分离能力名称与版本确保身份稳定。**

- **链接: [https://arxiv.org/pdf/2605.28097](https://arxiv.org/pdf/2605.28097)**

> **作者:** Xue Qin; Simin Luan; John See; Zeyd Boukhers; Cong Yang; Zhijun Li
>
> **备注:** 14 pages, 6 figures, 4 tables
>
> **摘要:** Canary deployment routes a fraction of traffic to a new software version, monitors metrics, and rolls back on regression. Mainstream controllers (Argo Rollouts, Spinnaker, Flagger) change the deployed system's cryptographic identity during the canary window. The drift is harmless for stateless microservices but breaks the claim that "the agent you certified is still the agent you have" for safety-critical embodied agents, forcing re-certification per canary. We present ICAN-Deploy (Identity-stable CANary Deployment), a middleware construction whose state machine holds the identity hash invariant across the canary window by separating capability names (frozen, hashed) from capability versions (mutable runtime state). We implement ICAN-Deploy inside a runtime governance layer for LLM-driven robots and verify invariance by closed-form proof, AST lint, and TLA+ model-checking, then corroborate over N=100 real canary cycles on a Franka Panda arm in MuJoCo (zero drift; entry latency 95% BCa CI [1.52, 2.01] ms). A feature-flagged strawman that folds versions into the manifest falsifies on the same workload. A system certified once at identity-creation time can then ship arbitrary capability evolution under that same certification, within the version-and-name envelope.
>
---
#### [new 037] Uni-LaViRA: Language-Vision-Robot Actions Translation for Unified Embodied Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Uni-LaViRA，解决跨任务、跨机器人的具身导航问题，通过语言-视觉-机器人动作统一翻译实现零样本泛化。**

- **链接: [https://arxiv.org/pdf/2605.27582](https://arxiv.org/pdf/2605.27582)**

> **作者:** Hongyu Ding; Sizhuo Zhang; Ziming Xu; Jinwen Guo; Hongxiu Liu; Xingzhi Cheng; Zixuan Chen; Haifei Qi; Duo Wang; Hao Xu; Jieqi Shi; Yifan Zhang; Jing Huo; Jian Cheng; Yang Gao; Jiebo Luo
>
> **备注:** Project page: this https URL
>
> **摘要:** Embodied navigation requires an agent to map language and visual observations to a stream of spatial actions that drive a real robot through environments it has never seen. The dominant approach has been to scale vision-language-action (VLA) foundation models on ever-larger collections of robot trajectories. This paper argues that, for navigation specifically, generality can be obtained structurally, not only through data scale. The underlying decision structure of navigation reduces to a single Language-Vision-Robot Actions Translation. The language action emits semantic-level directional command and the vision action emits a pixel-level visual target. Both outputs lie inside the natural output manifold of pretrained multimodal large language models (MLLMs), so the task can be reasoned about by an agent rather than learned from robot data. Therefore, we present Uni-LaViRA, a unified agentic architecture that extends the same insight to four task families (VLN-CE, ObjectNav, EQA, and Aerial-VLN) and to four heterogeneous real robots (Wheeled, Quadruped, Humanoid robot, and a self-built UAV) in a zero-shot manner. Two agent-loop mechanisms make this unification practical. TODO List Memory (TDM) rewrites a structured checklist of pending sub-goals at every step, reciting the unfinished items back into the agent's most recent attention window. Second Chance Backtrack (SCB) rolls the robot back to the pre-error state and conditions the agent's next plan on the failed sub-trajectory, turning single-pass navigation into a self-correcting process. With zero training effort, Uni-LaViRA reaches 60.7% SR on VLN-CE R2R, 51.3% on VLN-CE RxR, 77.7% on HM3D-v2, 60.0% on HM3D-OVON, 54.7% on MP3D-EQA, and 40.0% on OpenUAV, matching or even surpassing recent training navigation foundation models that consume millions of samples and thousands of GPU-hours.
>
---
#### [new 038] Inducing Calmness With Pocket-Sized Robotics: Reducing Movement and Heart Rate in Children through Hand-Held Tactile Interactions
- **分类: cs.RO**

- **简介: 该论文属于儿童行为调控任务，旨在通过手持触觉设备降低儿童的生理唤醒和身体躁动。研究设计触觉游戏，发现能有效减少心率和运动量，促进平静专注状态。**

- **链接: [https://arxiv.org/pdf/2605.27533](https://arxiv.org/pdf/2605.27533)**

> **作者:** Morten Roed Frederiksen; Kasper Støy; Maja Matarić
>
> **备注:** 34 pages, 2 tables, 7 figures
>
> **摘要:** Periods of heightened arousal or restlessness can interfere with children's ability to focus, self-regulation, and physically calm. Technologies that encourage embodied self-regulation through tactile interaction may provide a simple and accessible means of promoting calmness. This paper investigates how interaction with a pocket-sized tactile device influences physiological and behavioral markers of calmness in typically developing children. Building on prior work examining heart rate modulation, we present new findings on how tactile interaction affects full-body movement and postural stability. We employ a device that engages children through a hand-held rhythmic vibration-matching game, designed to focus attention and encourage stillness. Eighteen children participated in a within-subjects study that involved two conditions: with and without tactile interaction with a hand-held device, while having their heart rate and body movement recorded. Results show that the tactile game interaction reduced physiological arousal (heart rate decreased by 3.56 bpm, p < 0.01) and physical restlessness (overall movement decreased by 38%, p < 0.05), with attention-related body regions showing the greatest change toward stillness (45% reduction in movement). These findings demonstrate that brief tactile game-like engagement with a hand-held device can down-regulate physiological activation, promoting the calm and focused states toward sustained attention and behavior regulation.
>
---
#### [new 039] Identifying Explicit Parsimonious Piece-wise Polynomial Relationships in Industrial time-series: Application to manipulator robots
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于工业时间序列建模任务，旨在解决从大量特征中提取简洁的分段多项式关系问题。通过改进算法，得到显式分段多项式模型，并应用于机械臂逆模型识别及泛化能力验证。**

- **链接: [https://arxiv.org/pdf/2605.28320](https://arxiv.org/pdf/2605.28320)**

> **作者:** Mazen Alamir; Sacha Clavel
>
> **摘要:** This paper addresses the problem of identifying parsimonious explicit piece-wise polynomial relationships that might involve a relatively large number of raw features. The algorithm leverages a recently proposed identification algorithm that yields parsimonious implicit relationships enabling to derive normality characterization in the context of anomaly detection and localization. The algorithm proposed in this paper goes a step further by deriving explicit piece-wise representations that are built using the set of polynomials involved in the implicit representations. The framework is illustrated on the problem of identifying parsimonious explicit representations of the inverse model of a 6-axis manipulator robot. Moreover, further experiments on a 4-axis robot are also shown which are designed to investigate the generalization capability of parsimonious models compared to state-of-the-art DNNs structures, when models face unseen contexts of use.
>
---
#### [new 040] Provably Guaranteed Polytopic Uncertainty Quantification for SLAM
- **分类: cs.RO**

- **简介: 该论文属于SLAM任务，解决安全关键场景下的不确定性量化问题。提出保证性的UQ算法，使用多面体表示不确定性，确保姿态和地标被包含在确定性集合中。**

- **链接: [https://arxiv.org/pdf/2605.28172](https://arxiv.org/pdf/2605.28172)**

> **作者:** Guangyang Zeng; Yulong Gao; Yuan Shen; Lingpeng Chen; Haoying Li; Guodong Shi; Junfeng Wu
>
> **备注:** 16 pages, 10 figures; accepted by Robotics: Science and Systems 2026
>
> **摘要:** In safety-critical robotics applications, guaranteed and practical uncertainty quantification (UQ) in perception is vital. Many existing works either offer no formal containment guarantee, rely on restrictive modeling assumptions, or focus only on pose estimation rather than a complete SLAM pipeline. This paper presents provably guaranteed UQ algorithms for 3D-3D landmark-based SLAM. The algorithms consist of three basic UQ modules: forward UQ for mapping, backward UQ for pose tracking, and pose compound. Each module produces a certified uncertainty set; when the input uncertainty bounds are deterministic, the output sets inherit deterministic guarantees, i.e., they provably contain the true poses and landmarks. Specifically, we use polytopes to represent uncertainty sets, enabling tractable computations and a unified treatment of pose uncertainty. To enhance algorithms' practical usability, we incorporate conformal prediction to calibrate measurement uncertainty from data with prescribed probability. Simulations and experiments demonstrate that the proposed algorithms provide both strong theoretical guarantees and practical usability. The code is open-sourced at this https URL.
>
---
#### [new 041] Simultaneous Contact Selection and Planning for Contact-Rich Manipulation with Cascaded Optimization
- **分类: cs.RO**

- **简介: 该论文属于机器人接触丰富操作任务，解决接触位置选择与路径规划问题。提出SCSP框架，通过级联优化实现高效、鲁棒的接触选择与规划。**

- **链接: [https://arxiv.org/pdf/2605.27972](https://arxiv.org/pdf/2605.27972)**

> **作者:** Zhe Zhang; Xingrong Diao; Haoxiang Liang; Han Yang; Bi-Ke Zhu; Dandan Zhang; Jiankun Wang
>
> **备注:** 20 pages, 18 pages
>
> **摘要:** We propose an optimization-based framework for robust contact-rich manipulation. Recent contact-implicit methods enable online hybrid planning across contact modes, allowing closed-loop manipulation for a given target state and contact location sequence of the robot and object. However, most existing approaches lack the ability to autonomously reason and generate diverse contact location sequences and manipulation trajectories, i.e., active contact location selection, which limits their applicability to relatively simple tasks. Active contact location selection is challenging due to complementarity in contact dynamics and the sparse gradients, making the design of a unified framework for contact selection and planning difficult. To address these challenges, we introduce Simultaneous Contact Selection and Planning (SCSP), a cascaded optimization framework comprising Contact Selection Optimization (CSO) and Contact Planning Optimization (CPO). CSO leverages a surrogate contact model and discrete-continuous optimization to efficiently resolve the nonsmoothness and coupling in contact selection, enabling online global searching of optimal contact locations. CPO performs prior-guided contact planning by evaluating the reference contact locations produced by CSO and generating corresponding manipulation trajectories in real time for redundant manipulators. Extensive simulations and real-world experiments demonstrate that SCSP produces diverse manipulation behaviors and robust control under inaccurate dynamics and perceptual noise. We further validate the generalization of the framework on challenging manipulation tasks. Project website: \href{this https URL}{this https URL}.
>
---
#### [new 042] EIT-Pneumatic Hybrid Robotic Skin for Practical and Accurate Force Map Reconstruction
- **分类: cs.RO**

- **简介: 该论文属于触觉感知任务，旨在解决机器人皮肤力场重建的准确性与实用性问题。通过融合EIT与气动传感技术，提升大范围触觉感知的精度与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.28468](https://arxiv.org/pdf/2605.28468)**

> **作者:** Junhwi Cho; Sunggyu Bae; Junghyeon Ma; Hyosang Lee; Jung Kim; Kyungseo Park
>
> **备注:** 8 pages, 8 figures. Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026. J. Cho, S. Bae, J. Ma contributed equally
>
> **摘要:** We present a hybrid robotic skin that combines electrical impedance tomography (EIT) with pneumatic tactile sensing to improve force reconstruction capability. The developed robotic skin is fabricated entirely by 3D printing and spray coating, making it affordable and easy to build. A Tikhonov-regularized inverse reconstruction, paired with per-pad pneumatic calibration, enables accurate large-area tactile sensing with a simple measurement scheme. For validation, we conducted load-cell indentation experiments; the results showed consistent force reconstruction across locations within a pad. Compared with an EIT-only baseline, sensitivity non-uniformity was also reduced, with the coefficient of variation decreasing from 0.31 to 0.14, indicating that the proposed approach addresses a longstanding limitation of EIT. We further demonstrated chest-mounted integration on a humanoid robot and found that the pneumatic signals remained reliable across diverse contact scenarios, including multiple simultaneous contacts on the same sensing pad. These results indicate a practical path toward accurate, scalable whole-body tactile sensing in real robotic systems.
>
---
#### [new 043] Simulation-Informed Diffusion for Decentralized Multi-robot Motion Planning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于多机器人运动规划任务，解决分布式场景下机器人无法全局感知和可靠通信的问题。提出SID框架，利用扩散模型预测邻近机器人轨迹并规划自身安全路径。**

- **链接: [https://arxiv.org/pdf/2605.27697](https://arxiv.org/pdf/2605.27697)**

> **作者:** Jinhao Liang; Sven Koenig; Ferdinando Fioretto
>
> **摘要:** Decentralized multi-robot motion planning requires each robot to generate collision-free trajectories from local observations, without global sensing or reliable communication. However, most existing planners, whether classical or learning-based, generate trajectories from a static snapshot of the local observation, which limits their ability to anticipate the future behavior of neighboring robots. This limitation is critical as the number of robots increases and the environment becomes more cluttered. To overcome this challenge, this paper introduces Simulation-Informed Diffusion (SID), a decentralized framework built on constraint-aware diffusion models (CADM). SID first uses CADM to simulate the future trajectories of neighboring robots from their currently observed states, and then uses the same CADM to plan each robot's own trajectory under safety constraints informed by these simulations. Crucially, the accurate simulation of neighbors enables a minimal communication scheme that triggers coordination only when necessary in highly congested scenarios. Experiments across diverse environments show that SID consistently outperforms baseline methods in terms of planning effectiveness and constraint satisfaction, and scales to scenarios with 108 robots and 160 obstacles.
>
---
#### [new 044] A Factory-Floor Deployment Case Study of VLA Pipelines for Industrial Packaging Task: Workflow, Failures, and Lessons
- **分类: cs.RO**

- **简介: 该论文研究工业包装任务中VLA策略的部署，解决实际应用中的可靠性问题。通过迭代优化，提升机器人在复杂环境下的操作能力。**

- **链接: [https://arxiv.org/pdf/2605.27461](https://arxiv.org/pdf/2605.27461)**

> **作者:** Brian Zhu; Philipp Schmitt; Philine Meister; Lukas Gensler; Momen Khalil; Emmanuele Poggi; Johannes Hechtl; Carsten Braunroth; Kai Wurm; Gokul Narayanan; Eugen Solowjow; Georg von Wichert; Andre Scholz; Felix Albrecht; Maxmillian Metzner
>
> **摘要:** Vision-Language-Action (VLA) policies have shown promising manipulation capabilities, yet their practical impact is often limited by the reliability demands of real-world deployment. We present a deployment study of an industrial packaging task at Siemens Factory (GWE, Erlangen, Germany), where a robot must pick a transparent accessory bag from a cluttered pile, insert it into the remaining cavity of a cardboard package, and ensure that the bag and its contents remain below the closing plane. Our goal is to understand the practical effort required to adapt a pretrained Pi0.5 policy to a single factory-floor task through iterative fine-tuning and deployment-driven refinement. The pipeline consists of repeated loops of data collection, curation, fine-tuning, evaluation, and targeted recovery data collection. We have accumulated 2535 episodes (10 hours) from the on-site factory settings. In this paper, we contribute an empirical account of a factory-floor VLA deployment, highlighting recurring failure modes and lessons that inform how to improve the deployment workflow.
>
---
#### [new 045] Tactile-Proprioceptive Sensor Fusion for Contact Wrench Estimation in Whole-Body Physical Human-Robot Interaction
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于物理人机交互任务，旨在提升接触力估计的准确性与响应性。通过融合触觉与本体感知信息，解决摩擦不确定性问题，实现更安全、自然的人机互动。**

- **链接: [https://arxiv.org/pdf/2605.28412](https://arxiv.org/pdf/2605.28412)**

> **作者:** Junha Min; Junghyeon Ma; Jiwung Kwon; Sunggyu Bae; Joohyung Kim; Kyungseo Park
>
> **备注:** 8 pages, 6 figures. Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Direct physical guidance is a natural means of teaching and interacting with robots, and robotic skins make a key contribution by enabling sensitive contact sensing and localization. This paper presents a tactile-proprioceptive sensor fusion framework for natural physical human-robot interaction. Tactile cues from pneumatic skin pads serve as contact indicators that bypass the ambiguity between frictional residues and applied external forces, enabling highly sensitive contact detection without explicit friction identification. We fuse these cues with motor-current-based proprioception to reconstruct multi-axis contact forces on the robot surface. To maintain accuracy during motion, we employ a temporal convolutional network (TCN) to mitigate friction hysteresis during stick-slip transitions, reducing uncertainty at contact onset and yielding smooth, responsive guidance. We validate the approach on a skin-integrated robot arm: (i) multi-axis forces are reconstructed in stationary contacts, and (ii) simultaneous force estimation and kinesthetic teaching are demonstrated. Results indicate improved sensitivity and responsiveness across diverse contact conditions compared with tactile-only and proprioceptive-only baselines, supporting tactile-proprioceptive fusion as a reliable pathway to safe, intuitive physical human-robot interaction.
>
---
#### [new 046] Beyond Binary: Sim-to-Real Dexterous Manipulation with Physics-Grounded Contact Representation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人触觉控制任务，旨在解决模拟到现实的接触丰富操作问题。提出基于物理的CoP触觉表示，保留密集接触信息并实现零样本迁移。**

- **链接: [https://arxiv.org/pdf/2605.28812](https://arxiv.org/pdf/2605.28812)**

> **作者:** Jiahe Pan; Stelian Coros; Jitendra Malik; Toru Lin
>
> **备注:** Project site: this https URL
>
> **摘要:** A primary bottleneck in contact-rich manipulation is the difficulty of collecting real-world data. Sim-to-real reinforcement learning offers a scalable alternative, but the simulation-reality gap prevents information-dense modalities like touch from being effectively used. Existing sim-to-real methods often mitigate this gap by simplifying tactile data into coarse low-dimensional features -- sacrificing the richness required for complex manipulation. In this work, we introduce Center-of-Pressure (CoP), an effective tactile representation grounded in physical principles that preserves dense contact information while maintaining robustness for sim-to-real transfer. To support this representation, we propose a sensor calibration scheme based on differentiable dynamics, enabling the estimation of taxel orientations without requiring ground-truth force measurements. We evaluate CoP on two blind, challenging contact-rich manipulation tasks: peg-in-hole insertion and ball balancing. Across both tasks, policies conditioned on CoP achieve zero-shot sim-to-real transfer on a multi-fingered hand, and outperform both coarse binary-contact and raw-taxel baselines. Analysis of learned policy states further suggests that CoP-conditioned policies encode task-relevant physical properties, such as object mass, as an emergent byproduct of control.
>
---
#### [new 047] IMU Propagation as Preintegration
- **分类: cs.RO**

- **简介: 该论文属于视觉-惯性导航任务，解决IMU预积分与传播等价性问题，通过统一视角实现代码复用和一致性验证。**

- **链接: [https://arxiv.org/pdf/2605.28279](https://arxiv.org/pdf/2605.28279)**

> **作者:** Jianzhu Huai
>
> **备注:** 6 pages, 2 figures, to present in ISPRS2026 Thematic Session 10 on Radar Perception
>
> **摘要:** IMU preintegration is widely used in factor-graph-based visual--inertial, lidar--inertial, and radar--inertial state estimation, yet it is often treated as a specialized implementation separate from conventional IMU propagation. This note shows that IMU preintegration and propagation are equivalent realizations of the same underlying computation. We present a convention-agnostic view in which the preintegrated measurement, bias Jacobians, and covariance can be obtained by wrapping an existing IMU propagation routine, while a preintegration module can conversely recover state-transition matrices and propagated covariances. This perspective simplifies the reuse of existing propagation code, supports translation across different error-state definitions, and provides practical consistency checks for preintegration implementations. Experiments with random IMU sequences demonstrate close agreement between an RK4-based propagation implementation and GTSAM's tangent and manifold preintegration modules in the recovered Jacobians, covariances, and transition matrices.
>
---
#### [new 048] GE-Sim 2.0: A Roadmap Towards Comprehensive Closed-loop Video World Simulators for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出GE-Sim 2.0，用于机器人操作的闭环视频世界模拟。解决真实环境模拟与策略学习的问题，通过新模块提升模拟精度与效率，支持高效策略训练与评估。**

- **链接: [https://arxiv.org/pdf/2605.27491](https://arxiv.org/pdf/2605.27491)**

> **作者:** Boxiang Qiu; Liliang Chen; Yue Liao; Nan Wang; Lintao Wang; Jiayi Luo; Wenzhi Zhao; Shengcong Chen; Di Chen; Ye Li; Chen Gao; Shuicheng Yan; Si Liu; Maoqing Yao; Guanghui Ren
>
> **摘要:** We introduce GE-Sim 2.0 (Genie Envisioner World Simulator 2.0), a closed-loop video world simulator for robotic manipulation. Building on the action-conditioned video generation framework of Genie Envisioner, GE-Sim 2.0 is re-trained on thousands of hours of real-world robot data spanning teleoperation, contact-rich interaction, and on-robot policy deployment, substantially improving action-following fidelity and trajectory coverage. On top of this foundation, three new modules close the loop from video simulation to policy learning: a state expert that decodes proprioceptive state from video latents to support next-chunk prediction by downstream VLA policies; a world judge that scores generated rollouts against task instructions, yielding machine-verifiable success signals and rewards in place of manual inspection; and an acceleration framework that delivers a 25-frame rollout in 2.3 seconds on a single H100, with up to 4* frame skipping at inference for long-horizon evaluation. GE-Sim 2.0 tops the public WorldArena leaderboard at only 2B parameters, outperforming both dedicated robotic world models and closed-source general video generators, and policies trained against its rollouts and rewards translate into measurable real-world gains, establishing GE-Sim 2.0 as a practical platform for scalable evaluation and closed-loop learning of manipulation policies.
>
---
#### [new 049] Magnet-Based Soft Robotic Skin Using a 3D-Printed Multi-Lattice Structure and CNN-Based Tactile Super-Resolution
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉感知任务，旨在解决大范围、高精度触觉传感问题。通过磁-软皮肤结构与CNN模型，实现接触位置与力的实时估计。**

- **链接: [https://arxiv.org/pdf/2605.28352](https://arxiv.org/pdf/2605.28352)**

> **作者:** Yunseong Bang; Joowon Park; Suan Sim; Youngjun Ryu; Sukho Park; Kyungseo Park
>
> **备注:** 6 pages, 9 figures. Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026. Y. Bang and J. Park contributed equally
>
> **摘要:** This paper presents a magnet-based robotic skin that integrates a multilayer soft lattice with distributed Hall-effect sensor arrays and a tactile super-resolution model. External contact forces are converted to magnetic field changes by embedded permanent magnets, and the lattice spreads these changes across the sensing domain. This gives each sensor a large, overlapping receptive field and enables a large sensing area with minimal blind spots. Lattice parameters are tunable, enabling joint adjustment of mechanical compliance and transduction characteristics. An implicit modeling workflow and selective laser sintering (SLS) 3D printing support rapid fabrication of conformal, high-complexity structures. A convolutional neural network trained on experimental measurements estimates contact location and normal force in real time. Experiments validate localization accuracy and indicate scalability to larger surfaces, suggesting applicability to whole-body robotic skin and safe human-robot interaction.
>
---
#### [new 050] Synthetic Emotions vs. Gamification: Exploring Engagement Strategies for Small Social Robots in Different Age Groups
- **分类: cs.RO**

- **简介: 该论文研究小社交机器人在不同年龄群体中的参与策略，比较合成情感与游戏化机制的效果，旨在提升儿童和成人的互动参与度。**

- **链接: [https://arxiv.org/pdf/2605.27539](https://arxiv.org/pdf/2605.27539)**

> **作者:** Morten Roed Frederiksen; Kasper Støy
>
> **备注:** 7 pages
>
> **摘要:** Many children experience challenges in emotional regulation and social interaction, which can limit their participation in everyday activities and therapeutic programs. For socially assistive robots to be effective in this context, it is essential that children remain consistently and meaningfully engaged. We explore engagement strategies for a tactile robot designed to support children suffering from anxiety disorders through daily interactions. The robot delivers either synthetic emotional feedback or point rewards to encourage user participation. We evaluated these strategies through two studies: a preference assessment with 16 school children aged 6-8 years, and a behavioral study with 14 university students aged 20-27 years in naturalistic environments. The study with school children indicated a preference for emotional engagement over points-based approaches. The follow up study with university students across a full day of interactions revealed contrasting results: points-based systems produced significantly higher task accuracy (p < 0.05) and sustained performance over time. Findings from different user groups suggest that stated preferences and behavioral outcomes can diverge depending on engagement context, highlighting the importance of validating design assumptions through observed interaction. This work contributes insights into age-related differences in engagement strategy effectiveness in human-robot interaction design.
>
---
#### [new 051] Colosseum V2: Benchmarking Generalization for Vision Language Action Models
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决VLA模型在分布变化下的泛化能力问题。提出Colosseum V2基准，评估不同条件下的性能，促进通用机器人策略发展。**

- **链接: [https://arxiv.org/pdf/2605.27759](https://arxiv.org/pdf/2605.27759)**

> **作者:** Jeremy Morgan; Prajwal Vijay; Hyeonho Oh; Jincen Song; Ashvin Arora; Alina Du; Gaurav Sukhatme; Jesse Thomason; Ishika Singh
>
> **摘要:** Vision-Language-Action (VLA) models demonstrate promising generalization in robotic manipulation, driven by advances in large-scale vision and language pre-training. This progress can be misleading. Despite the zero-shot perception and language capabilities of VLAs, their overall task performance often degrades under distribution shifts, revealing gaps in how these systems translate high-level understanding into robust behavior. To systematically study this gap, we introduce Colosseum V2, a large-scale simulation benchmark for evaluating VLA generalization in robot learning across diverse conditions. The benchmark comprises 28 tasks spanning 13 task categories and two robot morphologies, covering a wide range of manipulation primitives and long-horizon behaviors. Built on the ManiSkill simulator, Colosseum V2 enables fast, GPU-parallelized evaluation and supports both in-domain and out-of-domain testing at scale. We evaluate state-of-the-art methods, including Action Chunking Transformers (ACT) and Pi0.5, and reveal limitations in both base performance and generalization. We demonstrate strong correlations between simulation and real-world metrics that support the ecological validity of the benchmark. By standardizing tasks, metrics, and evaluation protocols within a unified benchmark, Colosseum V2 enables reproducible and fair comparisons, reduced evaluation overhead, and accelerated progress toward general-purpose robot policies.
>
---
#### [new 052] EventShiftFlow: Towards Hardware-efficient FPGA-based Flow Estimation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于运动估计任务，旨在解决事件相机在FPGA上高效实现流估计的问题。提出一种基于固定时间窗和1位网格的并行速度估计算法，无需浮点运算和迭代优化。**

- **链接: [https://arxiv.org/pdf/2605.28312](https://arxiv.org/pdf/2605.28312)**

> **作者:** Arianna Alonso Bizzi; Fernando Cladera; C. J. Taylor
>
> **备注:** 10 pages, 5 figures. Accepted to the IEEE ICRA 2026 Workshop on Challenges and Opportunities of Neuromorphic Field Robotics and Automation
>
> **摘要:** Event-based vision sensors offer asynchronous, high-temporal-resolution measurements that are attractive for low-latency robotic perception, but many event-based motion estimation methods are computationally intensive and difficult to map to FPGA hardware. We present a streaming velocity estimator that discretizes asynchronous events into fixed-duration time bins, constructs a 1-bit spatial occupancy grid, and evaluates multiple velocity hypotheses in parallel using only fixed-width integer logic - shift registers, counters, comparators, and small LUT-mapped multiplies - with no dividers and no DSP blocks. It requires no frame reconstruction, no floating-point arithmetic, and no iterative optimization. The method deliberately trades dense sub-pixel optical flow for a sparse, quantized velocity estimate at each active pixel, suited to low-latency tasks such as reactive obstacle avoidance on size-, weight-, and power-constrained platforms. On noisy synthetic data with known ground-truth velocities, the method recovers both magnitude and direction, with magnitude estimates being most challenged when objects of different velocities intersect. On a real event-camera sequence, directional accuracy reaches 99.5% across all four evaluated motion segments, with performance remaining robust across occupancy densities in the 10-40% range. We characterize the algorithm's density-dependent behavior, present a parameter sensitivity analysis, show that the proposed datapath requires less than 2 kB of storage, and implement a single-axis prototype on a low-cost Xilinx Artix-7.
>
---
#### [new 053] How VLAs Fail Differently: Black-Box Action Monitoring Reveals Architecture-Specific Failure Signatures
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于视觉语言动作（VLA）领域，研究不同架构在执行任务时的失败模式。通过分析三种VLA模型，发现其失败特征不同，需选择适配的监控机制以提高安全性。**

- **链接: [https://arxiv.org/pdf/2605.28726](https://arxiv.org/pdf/2605.28726)**

> **作者:** Krishnam Gupta
>
> **备注:** Accepted at IEEE ICRA 2026 Workshop "From Data to Decisions: VLA Pipelines for Real Robots", Vienna, June 2026. Non-archival workshop. 5 pages, 2 figures, 22 references
>
> **摘要:** We discover that VLA architectures fail in fundamentally different, predictable ways at the motor-command level. Running VQ-BeT, Diffusion Policy, and ACT on identical evaluation protocols (n=450 episodes across PushT and ALOHA 14-DOF bimanual manipulation), we find: (1) direction reversal rate is a universal failure predictor across all three architectures (AUROC=0.93, 0.79, 0.91; p<0.001); (2) jerk monitoring is predictive only for discrete-token architectures, following a discrete-to-continuous gradient (0.88, 0.69, 0.41); (3) velocity violations alone are non-predictive everywhere (AUROC 0.41-0.69), yet velocity checking is the most common safety mechanism in VLA deployment code; and (4) for continuous-family VLAs, velocity monitoring provides effectively zero predictive signal (AUROC=0.52 on ACT, 0.41 on Diffusion), proving that architecture-matched monitor selection is essential. These results quantify a monitoring consequence of the well-known discrete/continuous VLA distinction: the two families produce qualitatively different failure signatures that require different monitors. No single monitor works universally; architecture-matched selection is required. This finding was enabled by SafeContract, a training-free, black-box action monitoring toolkit with conformal calibration. Code: this https URL
>
---
#### [new 054] Learning a Kinodynamic Trajectory Manifold for Impact-Aware Compliant Catching of Fast-Moving Objects
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决快速移动物体的精准捕捉问题。通过强化学习构建低维轨迹流形，实现高效、稳定的抓取控制。**

- **链接: [https://arxiv.org/pdf/2605.28462](https://arxiv.org/pdf/2605.28462)**

> **作者:** Guorui Pei; Mengshi Zhang; Xi Chen; Jinsong Wu; Jiaming Qi; Peng Zhou
>
> **摘要:** Fast catching of free-flying objects is difficult because of short reaction time, impact uncertainty, and kinodynamic constraints. We use reinforcement learning in simulation to collect successful catching trajectories and learn a low-dimensional kinodynamic trajectory manifold. At run time, the estimated object initial state is mapped directly to a reference catching trajectory without online nonlinear optimization. The trajectory is tracked with compliant control near contact for improved impact absorption and capture stability.
>
---
#### [new 055] Robo-Blocks: Generative Scaffolding in End-User Design and Programming of Social Robots
- **分类: cs.HC; cs.RO**

- **简介: 论文探讨如何利用大语言模型为新手提供生成式辅助，以支持社交机器人编程。任务是解决新手编程技能不足问题，通过设计Robo-Blocks工具实现结构化叙事引导。**

- **链接: [https://arxiv.org/pdf/2605.28154](https://arxiv.org/pdf/2605.28154)**

> **作者:** Arissa J. Sato; Callie Y. Kim; Nathan Thomas White; Abhinav Maneesh; Yuqing Wang; Hui-Ru Ho; Bilge Mutlu
>
> **摘要:** Programming social robots is challenging for novice robot programmers due to required expertise in planning, interaction design, and programming. While large language models (LLMs) hold significant promise through code generation from natural-language descriptions, they can obscure critical elements of programming and supplant designer intent, eventually resulting in over-reliance instead of developing programming skills. In this paper, we explore how LLM-based social-robot-programming tools can support novice robot programmers through a Research through Design (RtD) process. We designed and prototyped Robo-Blocks, a block-based programming environment that leverages LLMs to offer novice robot programmers generative scaffolding through structured narratives that connect high-level ideas to executable robot behaviors. Through deployment with novices, we discovered emerging user personas and usage patterns for generative scaffolding and showed how this scaffolding shapes end-user design and programming strategies. We present design insights for the effective use of generative scaffolding and its integration into the practice of social-robot programming.
>
---
#### [new 056] Surprising Performances of Students with Autism in Classroom with NAO Robot
- **分类: cs.HC; cs.CY; cs.RO**

- **简介: 论文探讨了NAO机器人在自闭症学生课堂中的应用，旨在提升其学习表现和社交能力。任务属于教育技术领域，解决如何利用社交机器人改善自闭症儿童课堂参与度的问题。研究通过实验验证了NAO机器人的有效性。**

- **链接: [https://arxiv.org/pdf/2407.12014](https://arxiv.org/pdf/2407.12014)**

> **作者:** Qin Yang; Huan Lu; Dandan Liang; Shengrong Gong; Huanghao Feng
>
> **摘要:** Autism is a developmental disorder that manifests in early childhood and persists throughout life, profoundly affecting social behavior and hindering the acquisition of learning and social skills in those diagnosed. As technological advancements progress, an increasing array of technologies is being utilized to support the education of students with Autism Spectrum Disorder (ASD), aiming to improve their educational outcomes and social capabilities. Numerous studies on autism intervention have highlighted the effectiveness of social robots in behavioral treatments. However, research on the integration of social robots into classroom settings for children with autism remains sparse. This paper describes the design and implementation of a group experiment in a collective classroom setting mediated by the NAO robot. The experiment involved special education teachers and the NAO robot collaboratively conducting classroom activities, aiming to foster a dynamic learning environment through interactions among teachers, the robot, and students. Conducted in a special education school, this experiment served as a foundational study in anticipation of extended robot-assisted classroom sessions. Data from the experiment suggest that ASD students in classrooms equipped with the NAO robot exhibited notably better performance compared to those in regular classrooms. The humanoid features and body language of the NAO robot captivated the students' attention, particularly during talent shows and command tasks, where students demonstrated heightened engagement and a decrease in stereotypical repetitive behaviors and irrelevant minor movements commonly observed in regular settings. Our preliminary findings indicate that the NAO robot significantly enhances focus and classroom engagement among students with ASD, potentially improving educational performance and fostering better social behaviors.
>
---
#### [new 057] Teacher-Student Representational Alignment for Reinforcement Learning-Driven Imitation Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于模仿学习任务，旨在解决教师与学生策略间存在的模仿差距问题。通过构建共享嵌入空间，提升学生性能并减少模仿差距。**

- **链接: [https://arxiv.org/pdf/2605.28372](https://arxiv.org/pdf/2605.28372)**

> **作者:** Meraj Mammadov; Pedro Zuidberg Dos Martires; Johannes Andreas Stork
>
> **备注:** 6 pages, 5 figures. Accepted as an oral presentation at the RL4IL Workshop at ICRA 2026
>
> **摘要:** Imitation learning (IL) from a state-based reinforcement learning (RL) policy is a common approach to overcome the curse of dimensionality in complex and high-dimensional observation spaces prevalent in robotics. This paper addresses the irreducible imitation gap that emerges when teacher and student are learned in isolation, and the teacher policy has the liberty to rely on privileged state information that the student cannot infer from its observations. Instead of improving poor student performance with RL finetuning after IL, which often requires a whole new training setup, we propose a novel algorithm which learns a shared embedding space that hides agent-specific observations and thus trains imitable teacher policies by construction. We train the shared embedding space with self-supervised contrastive learning in parallel to the teacher policy and prevent it from extracting private information by limiting its gradients from updating the encoder networks. We perform evaluations on several example domains and compare to state-of-the-art baselines showing that our algorithm enables higher student performance with substantially reduced imitation gap.
>
---
#### [new 058] SAM-Enhanced Segmentation on Road Datasets: Balancing Critical Classes in Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于语义分割任务，解决自动驾驶中缺乏像素级标注的问题。通过SAM生成密集标注，提升模型在极端类别不平衡下的表现。**

- **链接: [https://arxiv.org/pdf/2605.28136](https://arxiv.org/pdf/2605.28136)**

> **作者:** Toomas Tahves; Mauro Bellone; Junyi Gu; Raivo Sell
>
> **摘要:** Dense semantic segmentation is essential for autonomous driving, yet many multi-modal datasets lack pixel-level annotations. The Zenseact Open Dataset (ZOD) provides rich multi-sensor data but only bounding-box labels, limiting its use for segmentation research. Our primary contribution is a Segment Anything Model (SAM)-based annotation pipeline that produces dense, pixel-level annotations for ZOD by converting bounding boxes into semantic masks. In this pilot study, we process over 100,000 frames and manually curate a 2,300-frame subset (36% acceptance rate) to establish a reliable baseline. Using these annotations, we evaluate transformer-based CLFT and CNN-based DeepLabV3+ architectures across diverse weather conditions, achieving up to 48.1% mIoU with CLFT-Hybrid. To address extreme class imbalance, where pedestrians, cyclists, and signs constitute less than 1% of pixels, we explore specialized models targeting rare classes. We further validate the pipeline on the Iseauto autonomous-vehicle platform, achieving 77.5% mIoU, and show that SAM-derived representations transfer effectively across sensor configurations via bidirectional transfer learning. All code and annotations are released to support reproducible research.
>
---
#### [new 059] Con-DSO: Learning Short-Horizon Consistency Priors for RGB-D Direct Sparse Odometry
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉里程计任务，解决RGB-D直接稀疏里程计在复杂环境中的一致性问题。通过预测光流和深度一致性不确定性，提升跟踪鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.27952](https://arxiv.org/pdf/2605.27952)**

> **作者:** Haolan Zhang; Thanh Nguyen Canh; Chenghao Li; Ziyan Gao; Xiongwen Jiang; Nak Young Chong
>
> **备注:** Submitted
>
> **摘要:** Visual odometry (VO) is a fundamental component in robotics and augmented reality. RGB-D direct VO benefits from metric depth measurements, but it can degrade in challenging environments, where dynamic objects, occlusions, illumination changes, and unreliable depth violate the short-horizon photometric and depth-geometric consistency assumptions used by direct alignment. Existing approaches mitigate these issues through semantic filtering, explicit occlusion reasoning, illumination adaptation, or hand-crafted geometric criteria, but often rely on external modules or fixed assumptions tailored to individual failure modes, limiting their flexibility and ability to handle diverse challenges in a unified manner. In this work, we propose Con-DSO, a consistency-aware RGB-D direct sparse odometry framework that predicts dense photometric and depth-geometric consistency uncertainty from temporally adjacent RGB-D frame pairs. The consistency network is trained using flow-guided photometric errors and projective depth-consistency errors, allowing consistency violations to be represented as pixel-level uncertainty. These pairwise uncertainty predictions are converted into a host-side quality prior for keyframe-based tracking. The prior is then applied to VO through quality-aware support-pixel selection and decoupled photometric-geometric weighting during pose estimation, enabling continuous attenuation of unreliable observations rather than hard rejection or threshold-based gating. Experiments on five public RGB-D benchmarks show substantial gains over direct RGB-D VO baselines, with over 20\% absolute trajectory error reduction on ICL-NUIM and 50\%--80\% reductions on RGB-D Scenes V2, TUM/Bonn Dynamic, and OpenLORIS sequences.
>
---
#### [new 060] Differentiable Model Predictive Safety for Heterogeneous Mobility at Urban Intersections
- **分类: cs.MA; cs.RO**

- **简介: 该论文属于多智能体协同控制任务，解决城市交叉口异构交通流的安全协调问题。提出DMPS框架，结合模型预测与强化学习，提升安全性并减少碰撞。**

- **链接: [https://arxiv.org/pdf/2605.27418](https://arxiv.org/pdf/2605.27418)**

> **作者:** Wenzhe Song; Hao Zhang
>
> **备注:** 6 pages. Published in IEEE IARCE 2025
>
> **摘要:** The imminent integration of autonomous vehicles and mobile robots in urban settings presents a critical safety challenge for future intelligent transportation systems. This paper addresses the complex problem of coordinating heterogeneous agents with disparate dynamics at unregulated intersections. We introduce a novel framework, differentiable model predictive safety (DMPS), which embeds the foresight of model-predictive control into a data-driven, end-to-end reinforcement learning architecture. DMPS agents learn a latent dynamics model to predict future trajectories contingent on their actions. A learned, differentiable safety critic then evaluates the risk of these trajectories. Crucially, by leveraging backpropagation through the entire unrolled predictive model, agents can efficiently compute the gradient of future safety with respect to their current action, enabling a minimal and precise online safety correction. Integrated into a multi-agent training scheme, DMPS virtually eliminates collisions to less than 5.6% in high-density, mixed vehicle-robot traffic simulations, demonstrating state-of-the-art safety without compromising energy and traffic efficiency.
>
---
## 更新

#### [replaced 001] Inversely Learning Transferable Rewards via Abstracted States
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于逆强化学习任务，旨在学习可迁移的抽象奖励函数，以在不同但相关的任务中生成有效行为。**

- **链接: [https://arxiv.org/pdf/2501.01669](https://arxiv.org/pdf/2501.01669)**

> **作者:** Yikang Gui; Prashant Doshi
>
> **备注:** Accepted at IJCAI 2026
>
> **摘要:** Inverse reinforcement learning (IRL) has progressed significantly toward accurately learning the underlying rewards in both discrete and continuous domains from behavior data. The next advance is to learn {\em intrinsic} preferences in ways that produce useful behavior in settings or tasks which are different but aligned with the observed ones. In the context of robotic applications, this helps integrate robots into processing lines involving new tasks (with shared intrinsic preferences) without programming from scratch. We introduce a method to inversely learn an abstract reward function from behavior trajectories in two or more differing instances of a domain. The abstract reward function is then used to learn task behavior in another separate instance of the domain. This step offers evidence of its transferability and validates its correctness. We evaluate the method on trajectories in tasks from multiple domains in OpenAI's Gym testbed and AssistiveGym and show that the learned abstract reward functions can successfully learn task behaviors in instances of the respective domains, which have not been seen previously.
>
---
#### [replaced 002] A Survey on Event-based Optical Marker Systems
- **分类: cs.RO; cs.CV**

- **简介: 本文综述事件驱动的光学标记系统，属于机器感知任务，解决传统视觉系统在动态环境中的局限性，通过分析其异步特性与鲁棒性，探讨其在目标跟踪、位姿估计等领域的应用。**

- **链接: [https://arxiv.org/pdf/2504.20736](https://arxiv.org/pdf/2504.20736)**

> **作者:** Nafiseh Jabbari Tofighi; Maxime Robic; Fabio Morbidi; Pascal Vasseur
>
> **备注:** 11 pages, 6 figures, 2 table
>
> **摘要:** The advent of event-based cameras, with their low latency, high dynamic range, and reduced power consumption, marked a turning point in machine perception and robotic vision. In~particular, the combination of these neuromorphic sensors with widely-available passive or active optical markers (e.g. AprilTags, arrays of blinking LEDs), has recently opened up a new field of opportunities. This survey paper provides a comprehensive review of Event-Based Optical Marker Systems (EBOMS). We~analyze the underlying principles and technologies on which these systems are based, with a special focus on their asynchronous operation and robustness against challenging lighting conditions. We also describe the most relevant applications of EBOMS, including object detection and tracking, pose estimation, and optical communication. The article concludes with a discussion of possible future research directions in this rapidly-emerging and multidisciplinary area.
>
---
#### [replaced 003] RCM Constraint-Consistent Dynamic Control in Surgical Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于手术机器人控制任务，解决虚拟RCM约束在扭矩层面的一致性问题。通过建模RCM为时变完整约束，设计了统一的逆动力学控制器，提升控制精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2509.14075](https://arxiv.org/pdf/2509.14075)**

> **作者:** Yu Li; Hamid Sadeghian; Zewen Yang; Valentin Le Mesle; Sami Haddadin
>
> **备注:** Accepted at ICRA 2026
>
> **摘要:** Robotic-assisted minimally invasive surgery (RAMIS) requires accurate enforcement of the remote center of motion (RCM) constraint to ensure safe tool motion through a trocar. Existing virtual RCM controllers are commonly formulated either at the kinematic level or as task-space objectives, which makes torque-level enforcement under trocar motion and physical interaction difficult to formulate consistently. This paper models the RCM as a rheonomic holonomic constraint and incorporates it into a projection-based inverse-dynamics controller with explicit constrained/free-motion torque decomposition. The resulting formulation unifies kinematic RCM enforcement and task-space tracking at the torque level, while preserving a constraint-consistent structure for residual regulation and null-space compliance. The proposed controller is validated in simulation and on a RAMIS training platform against representative projection-based and constrained-dynamics baselines. Across spiral tracking, varying insertion depth, moving trocar conditions, and human interaction, the method achieves lower RCM residuals and smoother torque profiles while maintaining accurate tool-tip tracking. These results support the use of constraint-consistent torque control for reliable virtual RCM enforcement in surgical robotics. The project page is available at this https URL
>
---
#### [replaced 004] Field evaluation and optimization of a lightweight autonomous lidar-based UAV system based on a rigorous experimental setup in boreal forest environments
- **分类: cs.RO**

- **简介: 该论文属于自主无人机森林探测任务，旨在解决现有算法评估标准不统一的问题。提出标准化实验方案，优化轻量级激光雷达无人机系统，并通过实际飞行验证其性能提升。**

- **链接: [https://arxiv.org/pdf/2512.14340](https://arxiv.org/pdf/2512.14340)**

> **作者:** Aleksi Karhunen; Teemu Hakala; Väinö Karjalainen; Eija Honkavaara
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Interest in utilizing autonomous uncrewed aerial vehicles (UAVs) for under-canopy forest remote sensing has increased in recent years, resulting in the publication of numerous autonomous flight algorithms in the scientific literature. To support the selection and development of such algorithms, a reliable comparison of existing approaches based on published studies is essential. However, reliable comparisons are currently challenging due to widely varying experimental setups and incomplete reporting practices. This study proposes a standardized experimental setup for evaluating autonomous under-canopy UAV systems to fill this gap. The proposed setup emphasizes quantitative reporting of forest complexity, visual representation of test environments, execution of multiple repeated flights, and reporting of flight success rates alongside qualitative flight results. In addition, flights at multiple target speeds are encouraged, with reporting of realized flight speed, mission completion time, and point-to-point flight distance. The proposed setup is demonstrated using a lightweight lidar-based quadrotor employing state-of-the-art open-source algorithms, evaluated through extensive experiments in two natural boreal forest environments. Based on a systematic evaluation of the original system, several improvements were introduced. The same experimental protocol was then repeated with the optimized system, resulting in a total of 93 real-world flights. The optimized system achieved success rates of 12/15 and 15/15 at target flight speeds of 1 m/s and 2 m/s, respectively, in a medium-difficulty forest, and 12/15 and 5/15 in a difficult forest. Adoption of the proposed experimental setup would facilitate the literature-based comparison of autonomous under-canopy flight systems and support systematic performance improvement of future UAV-based forest robotics solutions.
>
---
#### [replaced 005] PRISM-SLAM: Probabilistic Ray-Grounded Inference for Scale-aware Metric SLAM
- **分类: cs.RO**

- **简介: 该论文提出PRISM-SLAM，解决单目SLAM的尺度模糊和动态环境跟踪问题，通过融合视觉基础模型先验，实现精准的度量定位与建图。**

- **链接: [https://arxiv.org/pdf/2605.19257](https://arxiv.org/pdf/2605.19257)**

> **作者:** Eunsoo Im; Gyeonggwan Lee; Junghun Suh
>
> **摘要:** Monocular SLAM historically suffers from scale ambiguity and tracking failure in dynamic environments. While recent vision foundation models (VFMs) provide remarkable zero-shot depth priors, naively integrating these deterministic predictions ignores predictive uncertainty and frame-to-frame scale inconsistencies. We propose PRISM-SLAM, a real-time framework that rigorously integrates VFM priors into a structured Bayesian factor graph to achieve scale-aware, metric-consistent localization and mapping. Specifically, we introduce a Plücker Ray-Distance Factor to anchor monocular observations in absolute space within a globally consistent metric coordinate system, mathematically resolving scale drift by making the metric scale Fisher-identifiable. To handle environmental dynamics, we derive an epistemic uncertainty proxy from temporal depth consistency and formulate a Dynamic Scene Uncertainty Gating (DSUG) mechanism. This soft-gating approach probabilistically down-weights dynamic distractors without incurring the heavy computational overhead associated with traditional semantic segmentation masks. By employing a multi-process architecture that asynchronously processes VFM inference and geometric tracking, PRISM-SLAM provides verified metric output at 30 FPS using solely RGB input, bridging the gap between foundation models and real-world robotic applications. Evaluated on the TUM RGB-D and 7-Scenes benchmarks, PRISM-SLAM achieves a metric $SE(3)$ Absolute Trajectory Error (ATE) nearly identical to its oracle-aligned $Sim(3)$ error. This demonstrates that our system can produce deployment-ready metric trajectories by delivering robust metric SLAM solutions without any post-hoc scale correction. Project page: this https URL
>
---
#### [replaced 006] Mind Dreamer: Untethering Imagination via Active Causal Intervention on Latent Manifolds
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出Mind Dreamer框架，解决强化学习中因历史数据依赖导致的想象力受限问题。通过主动因果干预，提升样本效率和稀疏奖励任务表现。**

- **链接: [https://arxiv.org/pdf/2605.16030](https://arxiv.org/pdf/2605.16030)**

> **作者:** Shaojun Xu; Xiaoling Zhou; Yihan Lin; Yapeng Meng; Xinglong Ji; Luping Shi; Rong Zhao
>
> **备注:** 34 pages, 7 figures, ICML 2026 accepted
>
> **摘要:** Model-Based Reinforcement Learning yields sample efficiency via latent imagination, yet remains constrained by Historical Tethering: imagination is typically initialized from observed states. This creates a learning asymmetry, where the world model's manifold discovery outpaces the policy's sparse-reward optimization. We propose Mind Dreamer (MD), a framework that instantiates Active Causal Intervention to transcend Markovian continuity. MD reformulates discovery as the minimization of a global Relay Expected Free Energy. Instead of initializing from historical data, it draws initial states from an adversarial generator $s_0 \sim p_{gen}(\cdot)$, creating non-continuous latent jumps to epistemic blind spots that are physically plausible yet cognitively challenging. We derive Relay Value Function and Relay Uncertainty Function to resolve the credit assignment paradox across these spatial ruptures. Treating synthesized anchors as interventional intermediary states, these potentials propagate pragmatic and epistemic value through Bellman-style backups. Notably, we prove that uncertainty propagation across discontinuities necessitates a quadratic discount $\gamma^2$, establishing a formal epistemic horizon. Theoretically, MD approximates a variance-minimizing importance sampler that expands the manifold's spectral gap, reducing the hitting time to critical bottleneck states. Empirically, MD achieves a 1.67$\times$ average speedup over DreamerV3 on DeepMind Control Suite, reaching 8.8$\times$ in sparse-reward tasks.
>
---
#### [replaced 007] Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决动作预测离散化带来的连续性问题，提出NIAF将动作表示为连续函数，提升控制平滑性和物理合理性。**

- **链接: [https://arxiv.org/pdf/2603.01766](https://arxiv.org/pdf/2603.01766)**

> **作者:** Haoyun Liu; Jianzhuang Zhao; Xinyuan Chang; Tianle Shi; Chuanzhang Meng; Jiayuan Tan; Feng Xiong; Tong Lin; Dongjie Huo; Mu Xu; SongLin Dong; Zhiheng Ma; Yihong Gong; Sheng Zhong
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Despite the rapid progress of vision-language-action (VLA) models, the prevailing practice of predicting action chunks as discrete waypoints remains structurally misaligned with the intrinsic continuity of physical motion. This discretization arises naturally from fixed-rate robot data collection and the token-by-token prediction paradigm of large language models, but ties actions to rigid sampling rates, does not naturally support analytically consistent higher-order derivatives, and introduces quantization artifacts that hinder precise, compliant interaction. We propose Neural Implicit Action Fields (NIAF), which reformulates chunk-level action representation from discrete waypoints to continuous action functions. Using a vision-language model as a hierarchical spectral modulator over a learnable motion prior, NIAF synthesizes continuous-time action manifolds with arbitrary temporal resolution. This formulation enables analytical differentiation, allowing explicit supervision of velocity and regularization of higher-order derivative signals to promote mathematical consistency, physical plausibility, and control smoothness. Our approach achieves strong results on CALVIN and LIBERO across diverse backbones. Real-world experiments further confirm that NIAF supports stable impedance control, bridging policy-side action generation and execution-side smooth control.
>
---
#### [replaced 008] Relational Semantic Reasoning on 3D Scene Graphs for Open World Interactive Object Search
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于开放世界交互式物体搜索任务，解决传统方法在语义关系理解与实时性上的不足。提出SCOUT方法，通过3D场景图和关系启发式进行高效探索，并引入SymSearch评估基准。**

- **链接: [https://arxiv.org/pdf/2603.05642](https://arxiv.org/pdf/2603.05642)**

> **作者:** Imen Mahdi; Matteo Cassinelli; Fabien Despinoy; Tim Welschehold; Abhinav Valada
>
> **摘要:** Open-world interactive object search in household environments requires understanding semantic relationships between objects and their surrounding context to guide exploration efficiently. Prior methods either rely on vision-language embeddings similarity, which does not reliably capture task-relevant relational semantics, or large language models (LLMs), which are too slow and costly for real-time deployment. We introduce SCOUT: Scene Graph-Based Exploration with Learned Utility for Open-World Interactive Object Search, a novel method that searches directly over 3D scene graphs by assigning utility scores to rooms, frontiers, and objects using relational exploration heuristics such as room-object containment and object-object co-occurrence. To make this practical without sacrificing open-vocabulary generalization, we propose an offline procedural distillation framework that extracts structured relational knowledge from LLMs into lightweight models for on-robot inference. Furthermore, we present SymSearch, a scalable symbolic benchmark for evaluating semantic reasoning in interactive object search tasks. Extensive evaluations across symbolic and simulation environments show that SCOUT outperforms embedding similarity-based methods and matches LLM-level performance while remaining computationally efficient. Finally, real-world experiments demonstrate effective transfer to physical environments, enabling open-world interactive object search under realistic sensing and navigation constraints.
>
---
#### [replaced 009] Delay-Aware Reinforcement Learning for Highway On-Ramp Merging under Stochastic Communication Latency
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶控制任务，解决高速公路上匝道并道中因通信延迟导致的感知信息不准确问题。提出DAROM框架，通过延迟感知编码器和物理安全控制器提升控制性能。**

- **链接: [https://arxiv.org/pdf/2403.11852](https://arxiv.org/pdf/2403.11852)**

> **作者:** Amin Tabrizian; Zhitong Huang; Arsyi Aziz; Peng Wei
>
> **摘要:** Delayed and partially observable state information poses significant challenges for reinforcement learning (RL)-based control in real-world autonomous driving. In highway on-ramp merging, a roadside unit (RSU) can sense nearby traffic, perform edge perception, and transmit state estimates to the ego vehicle over vehicle-to-infrastructure (V2I) links. With recent advancements in intelligent transportation infrastructure and edge computing, such RSU-assisted perception is increasingly realistic and already deployed in modern connected roadway systems. However, edge processing time and wireless transmission can introduce stochastic V2I communication delays, violating the Markov assumption and substantially degrading control performance. In this work, we propose DAROM, a Delay-Aware Reinforcement Learning framework for On-ramp Merging that is robust to stochastic delays. We model the problem as a random delay Markov decision process (RDMDP) and develop a unified RL agent for joint longitudinal and lateral control. To recover a Markovian representation under delayed observations, we introduce a Delay-Aware Encoder that conditions on delayed observations, masked action histories, and observed delay magnitude to infer the current latent state. We further integrate a physics-based safety controller to reduce collision risk during merging. Experiments in the Simulation of Urban MObility (SUMO) simulator using real-world traffic data from the Next Generation Simulation (NGSIM) dataset demonstrate that DAROM consistently outperforms standard RL baselines across traffic densities. In particular, the gated recurrent unit (GRU)-based encoder achieves over 99% success in high-density traffic with random V2I delays of up to 2.0 seconds.
>
---
#### [replaced 010] LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出LocateAnything，解决视觉-语言定位与检测任务中的效率与精度问题。通过并行框解码技术提升解码速度和定位准确性，并构建大规模数据集增强模型性能。**

- **链接: [https://arxiv.org/pdf/2605.27365](https://arxiv.org/pdf/2605.27365)**

> **作者:** Shihao Wang; Shilong Liu; Yuanguo Kuang; Xinyu Wei; Yangzhou Liu; Zhiqi Li; Yunze Man; Guo Chen; Andrew Tao; Guilin Liu; Jan Kautz; Lei Zhang; Zhiding Yu
>
> **备注:** fix github link
>
> **摘要:** Vision-language models (VLMs) commonly formulate visual grounding and detection as a coordinate-token generation problem, serializing each 2D box into multiple 1D tokens that are learned and decoded largely independently. This token-by-token decoding mismatches the coupled structure of box geometry and creates a practical inference bottleneck due to strictly sequential generation. We introduce LocateAnything, a unified generative grounding and detection framework based on Parallel Box Decoding (PBD). By decoding geometric elements such as bounding boxes and points as atomic units in a single step, LocateAnything preserves intra-box geometric coherence and unlocks substantial parallelism. We show that PBD improves both decoding throughput and localization accuracy. We further develop a scalable data engine and curate LocateAnything-Data, a large-scale dataset with more than 138 million training samples, substantially increasing data diversity for high-precision localization. Extensive evaluations show that LocateAnything advances the speed-accuracy frontier, achieving significantly higher decoding throughput while improving high-IoU localization quality across diverse benchmarks. The results highlight the complementary benefits of Parallel Box Decoding and large-scale training data in enabling efficient and precise unified visual grounding and detection.
>
---
#### [replaced 011] SPARC: Spatial-Aware Path Planning via Attentive Agent Communication
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多机器人路径规划任务，旨在解决通信效率低下问题。通过引入RMHA机制，提升机器人间通信的时空相关性，提高高密度环境下的成功率。**

- **链接: [https://arxiv.org/pdf/2603.02845](https://arxiv.org/pdf/2603.02845)**

> **作者:** Sayang Mu; Xiangyu Wu; Bo An
>
> **备注:** The manuscript is being withdrawn at the request of the first author for the purpose of revising content and re-uploading a revised version with updated data/figures/text . The revised manuscript will be resubmitted to arXiv promptly with the same author list and research theme
>
> **摘要:** Efficient communication is critical for decentralized Multi-Robot Path Planning (MRPP), yet existing learned communication methods treat all neighboring robots equally regardless of their spatial proximity, leading to diluted attention in congested regions where coordination matters most. We propose Relation enhanced Multi Head Attention (RMHA), a communication mechanism that explicitly embeds pairwise Manhattan distances into the attention weight computation, enabling each robot to dynamically prioritize messages from spatially relevant neighbors. Combined with a distance-constrained attention mask and GRU gated message fusion, RMHA integrates seamlessly with MAPPO for stable end-to-end training. In zero-shot generalization from 8 training robots to 128 test robots on 40x40 grids, RMHA achieves approximately 75 percent success rate at 30 percent obstacle density outperforming the best baseline by over 25 percentage points. Ablation studies confirm that distance-relation encoding is the key contributor to success rate improvement in high-density environments. Index Terms-Multi-robot path planning, graph attention mechanism, multi-head attention, communication optimization, cooperative decision-making
>
---
#### [replaced 012] CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出CogVLA，解决视觉-语言-动作模型的效率与性能问题，通过指令驱动路由和稀疏化提升效果，适用于机器人任务。**

- **链接: [https://arxiv.org/pdf/2508.21046](https://arxiv.org/pdf/2508.21046)**

> **作者:** Wei Li; Renshan Zhang; Rui Shao; Jie He; Liqiang Nie
>
> **备注:** Accepted to NeurIPS 2025, Project Page: this https URL
>
> **摘要:** Recent Vision-Language-Action (VLA) models built on pre-trained Vision-Language Models (VLMs) require extensive post-training, resulting in high computational overhead that limits scalability and this http URL propose CogVLA, a Cognition-Aligned Vision-Language-Action framework that leverages instruction-driven routing and sparsification to improve both efficiency and performance. CogVLA draws inspiration from human multimodal coordination and introduces a 3-stage progressive architecture. 1) Encoder-FiLM based Aggregation Routing (EFA-Routing) injects instruction information into the vision encoder to selectively aggregate and compress dual-stream visual tokens, forming a instruction-aware latent representation. 2) Building upon this compact visual encoding, LLM-FiLM based Pruning Routing (LFP-Routing) introduces action intent into the language model by pruning instruction-irrelevant visually grounded tokens, thereby achieving token-level sparsity. 3) To ensure that compressed perception inputs can still support accurate and coherent action generation, we introduce V-L-A Coupled Attention (CAtten), which combines causal vision-language attention with bidirectional action parallel decoding. Extensive experiments on the LIBERO benchmark and real-world robotic tasks demonstrate that CogVLA achieves state-of-the-art performance with success rates of 97.4% and 70.0%, respectively, while reducing training costs by 2.5-fold and decreasing inference latency by 2.8-fold compared to OpenVLA. CogVLA is open-sourced and publicly available at this https URL.
>
---
#### [replaced 013] Implicit Null-space Manifold Generation for Redundant Robotic Systems
- **分类: cs.RO**

- **简介: 该论文研究冗余机器人系统的解空间表示问题。通过构建隐式标量场，捕捉解流形的几何结构，实现对解空间的连续距离场建模，提升任务规划效率。**

- **链接: [https://arxiv.org/pdf/2605.25770](https://arxiv.org/pdf/2605.25770)**

> **作者:** Taiki Ishigaki; Teresa Vidal-Calleja; Ko Ayusawa; Eiichi Yoshida
>
> **备注:** Corrected author names in references
>
> **摘要:** Robotic systems with redundant degrees of freedom can achieve the same task outcome using multiple configurations, resulting in solution sets that form manifolds in the configuration space. Existing approaches typically exploit such redundancy locally through Jacobian-based techniques to compute individual solutions or trajectories. While effective for solution computation, these methods do not retain a representation of the geometry of the solution set itself. In this work, we adopt a representation-centric approach to estimate the geometric structure of the solution space. We consider solution manifolds induced by general task-defining maps and construct an implicit scalar field over the configuration space, whose zero-level set corresponds to the solution manifold. To this end, we generate samples in the neighborhood of the solution manifold using a Jacobian-guided exploration strategy, which efficiently captures its local and global structure. The resulting implicit representation is defined over the configuration space and naturally induces a continuous, distance field that encodes proximity to the solution manifold. Experiments on a planar three-link robot and a seven-degree-of-freedom Franka manipulator demonstrate the effectiveness of the proposed representation. Furthermore, the framework enables consistent modeling of solution spaces across families of tasks with continuous variation.
>
---
#### [replaced 014] Imitating and Finetuning Model Predictive Control for Robust and Symmetric Quadrupedal Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在提升四足机器人在复杂地形上的运动性能。通过结合模型预测控制与强化学习，提出IFM框架，实现更稳健、对称和高效的步态。**

- **链接: [https://arxiv.org/pdf/2311.02304](https://arxiv.org/pdf/2311.02304)**

> **作者:** Donghoon Youm; Hyunyoung Jung; Hyeongjun Kim; Jemin Hwangbo; Hae-Won Park; Sehoon Ha
>
> **摘要:** Control of legged robots is a challenging problem that has been investigated by different approaches, such as model-based control and learning algorithms. This work proposes a novel Imitating and Finetuning Model Predictive Control (IFM) framework to take the strengths of both approaches. Our framework first develops a conventional model predictive controller (MPC) using Differential Dynamic Programming and Raibert heuristic, which serves as an expert policy. Then we train a clone of the MPC using imitation learning to make the controller learnable. Finally, we leverage deep reinforcement learning with limited exploration for further finetuning the policy on more challenging terrains. By conducting comprehensive simulation and hardware experiments, we demonstrate that the proposed IFM framework can significantly improve the performance of the given MPC controller on rough, slippery, and conveyor terrains that require careful coordination of footsteps. We also showcase that IFM can efficiently produce more symmetric, periodic, and energy-efficient gaits compared to Vanilla RL with a minimal burden of reward shaping.
>
---
#### [replaced 015] From Passive Monitoring to Active Defence: Resilient Control of Manipulators Under Cyberattacks
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究冗余机械臂在隐蔽虚假数据注入攻击下的弹性控制，提出从被动监测到主动防御的架构，有效减少攻击导致的末端偏差。**

- **链接: [https://arxiv.org/pdf/2603.13003](https://arxiv.org/pdf/2603.13003)**

> **作者:** Gabriele Gualandi; Alessandro V. Papadopoulos
>
> **备注:** v2: Accepted at ICRA 2026. Corrected minor typos, grammatical errors, and notation inconsistencies. Corrected the attacker's PD law in Sec. III-C: removed the feedforward acceleration term, viable only when the attacker assumes sufficient tracking precision; the active defence prevents this in our experiments, so only PD terms are used
>
> **摘要:** Cyber-physical robotic systems are vulnerable to false data injection attacks (FDIAs), in which an adversary corrupts sensor signals while evading residual-based passive anomaly detectors such as the chi-squared test. Such stealthy attacks can induce substantial end-effector deviations without triggering alarms. This paper studies the resilience of redundant manipulators to stealthy FDIAs and advances the architecture from passive monitoring to active defence. We formulate a closed-loop model comprising a feedback-linearized manipulator, a steady-state Kalman filter, and a chi-squared-based anomaly detector. Building on this passive monitoring layer, we propose an active control-level defence that attenuates the control input through a monotone function of an anomaly score generated by a novel actuation-projected, measurement-free state predictor. The proposed design provides probabilistic guarantees on nominal actuation loss and preserves closed-loop stability. From the attacker perspective, we derive a convex QCQP for computing one-step optimal stealthy attacks. Simulations on a 6-DOF planar manipulator show that the proposed defence significantly reduces attack-induced end-effector deviation while preserving nominal task performance in the absence of attacks.
>
---
#### [replaced 016] MVP-LAM: Learning Action-Centric Latent Action via Cross-Viewpoint Reconstruction
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MVP-LAM模型，用于学习具有动作信息的潜在动作表示，解决多视角下潜在动作监督不足的问题，提升动作预测与下游任务性能。**

- **链接: [https://arxiv.org/pdf/2602.03668](https://arxiv.org/pdf/2602.03668)**

> **作者:** Jung Min Lee; Dohyeok Lee; Seokhun Ju; Taehyun Cho; Jin Woo Koo; Li Zhao; Sangwoo Hong; Jungwoo Lee
>
> **摘要:** Latent actions learned from diverse human videos serve as pseudo-labels for vision-language-action (VLA) pretraining, but provide effective supervision only if they remain informative about the underlying ground-truth actions. For effective supervision, latent actions should contain information about the underlying actions even though they are inaccessible. We propose Multi-ViewPoint Latent Action Moel (MVP-LAM), which learns latent actions that are highly informative about ground-truth actions from multi-view videos. MVP-LAM trains latent actions with a cross-viewpoint reconstruction objective, so that a latent action from one view must explain the future in another view, reducing reliance on viewpoint-specific cues. On Bridge V2, MVP-LAM produces more action-centric latent actions, achieving higher mutual information with ground-truth actions and improved action prediction, including under out-of-distribution evaluation. Finally, pretraining VLAs with MVP-LAM latent actions improves downstream manipulation performance on various benchmarks. The code and trained checkpoints are available at this https URL.
>
---
#### [replaced 017] Bayesian Optimization Parameter Tuning Framework for a Lyapunov Based Path Following Controller
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于参数调优任务，解决非线性路径跟踪控制器手动调优效率低的问题。通过引入贝叶斯优化框架，高效搜索最优控制参数，提升控制器性能。**

- **链接: [https://arxiv.org/pdf/2512.12649](https://arxiv.org/pdf/2512.12649)**

> **作者:** Zhewen Zheng; Wenjing Cao; Hongkang Yu; Mo Chen; Takashi Suzuki
>
> **备注:** The authors request withdrawal because the current arXiv version does not reflect the complete and finalized authorship record of the manuscript. The author list and contribution record require correction before further public dissemination
>
> **摘要:** Parameter tuning in real-world experiments is constrained by the limited evaluation budget available on hardware. The path-following controller studied in this paper reflects a typical situation in nonlinear geometric controller, where multiple gains influence the dynamics through coupled nonlinear terms. Such interdependence makes manual tuning inefficient and unlikely to yield satisfactory performance within a practical number of trials. To address this challenge, we propose a Bayesian optimization (BO) framework that treats the closed-loop system as a black box and selects controller gains using a Gaussian-process surrogate. BO offers model-free exploration, quantified uncertainty, and data-efficient search, making it well suited for tuning tasks where each evaluation is costly. The framework is implemented on Honda's AI-Formula three-wheeled robot and assessed through repeated full-lap experiments on a fixed test track. The results show that BO improves controller performance within 32 trials, including 15 warm-start initial evaluations, indicating that it can efficiently locate high-performing regions of the parameter space under real-world conditions. These findings demonstrate that BO provides a practical, reliable, and data-efficient tuning approach for nonlinear path-following controllers on real robotic platforms.
>
---
#### [replaced 018] Informative Path Planning with Guaranteed Estimation Uncertainty
- **分类: cs.RO**

- **简介: 该论文属于环境监测任务，解决资源受限下如何高效获取数据的问题。通过结合信息路径规划与不确定性保证，提出方法在减少采样点和路径长度的同时，确保估计精度。**

- **链接: [https://arxiv.org/pdf/2602.05198](https://arxiv.org/pdf/2602.05198)**

> **作者:** Kalvik Jakkala; Saurav Agarwal; Jason O'Kane; Srinivas Akella
>
> **备注:** 15 pages, 11 figures, RSS 2026
>
> **摘要:** Environmental monitoring robots often need to estimate data fields (e.g., salinity, temperature, bathymetry) under tight resource constraints. Classical boustrophedon lawnmower surveys provide geometric coverage guarantees but can waste effort by oversampling predictable regions. In contrast, informative path planning (IPP) methods leverage spatial correlations to reduce oversampling, yet typically offer no guarantees on estimation quality. This paper bridges these approaches by addressing IPP with guaranteed estimation uncertainty in complex environments: computing the shortest path whose measurements ensure that the Gaussian process (GP) posterior variance -- an intrinsic uncertainty measure that lower-bounds the mean-squared prediction error under the GP model -- is upper bounded by a user-specified threshold over the monitoring region. We propose a three-stage approach for efficient environmental monitoring: (i) learning a GP model from prior information; (ii) transforming the GP kernel into binary coverage maps that identify locations where uncertainty can be reduced below a target threshold; and (iii) planning a near-shortest route to satisfy the global uncertainty constraint. Our approach incorporates non-stationary kernels to capture spatially varying correlations in heterogeneous phenomena and accommodates non-convex environments with obstacles. We provide near-optimal approximation guarantees for both sensing-location selection and the joint selection-and-routing problem under a travel budget. Experiments on real-world topographic data demonstrate that our planners achieve uncertainty targets with fewer sensing locations and shorter travel distances than representative baselines. Furthermore, field experiments with autonomous surface and underwater vehicles validate the real-world feasibility of the approach. Our code is available at: this http URL
>
---
#### [replaced 019] TacSE3: Equivariant SE(3) Motion Estimation from Low-Texture Visuotactile Images for In-Gripper Tracking and Compensation
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取中的运动估计任务，解决低纹理触觉图像下物体运动跟踪问题。提出TacSE3方法，通过触觉信号估计SE(3)位姿，提升抓取稳定性。**

- **链接: [https://arxiv.org/pdf/2605.17929](https://arxiv.org/pdf/2605.17929)**

> **作者:** Zhongyuan Liao; Junzhe Wang; Qingyang Liu; Zhenmin Huang; Jun Ma; Yi Cai; Fei Meng; Haobo Liang; Michael Yu Wang
>
> **摘要:** Robotic in-hand manipulation requires reliable object-motion tracking under frequent visual occlusion, yet low-texture visuotactile images provide few stable correspondences for conventional image- or geometry-matching methods. This paper presents TacSE3, a tactile motion-estimation pipeline that converts low-texture visuotactile observations into a decoupled three-dimensional force field and estimates incremental rigid-body motion on SE(3). The method derives planar translation from contact-centroid motion and estimates rotation primarily from shear-related tactile responses, yielding a physically interpretable signal for in-gripper tracking and compensation. Experiments with paired DM-Tac fingertip sensors show that dual-sensor sensing reduces translation-rotation ambiguity, supports rotation tracking across axes and object geometries, and provides a lightweight compensation signal that improves disturbance tolerance in downstream manipulation tasks without retraining the base policy.
>
---
#### [replaced 020] Realizing Robotic Swimming with Unified Fluid-Robot Multiphysics
- **分类: cs.RO; physics.comp-ph; physics.flu-dyn**

- **简介: 该论文属于水下机器人运动控制任务，旨在提升机器人游泳效率与灵活性。通过统一流体-机器人多物理场模拟，解决复杂流体交互难题，并优化仿鳗鱼机器人的运动模式。**

- **链接: [https://arxiv.org/pdf/2506.05012](https://arxiv.org/pdf/2506.05012)**

> **作者:** Jeong Hun Lee; Junzhe Hu; Sofia Kwok; Carmel Majidi; Zachary Manchester
>
> **备注:** 9 pages, 10 figures, accepted to Robotics: Science and Systems 2026
>
> **摘要:** Matching the swimming efficiency and agility of fish has remained an elusive goal in underwater robotics. Such locomotion capabilities rely on complex vortex interactions between the robot's body and the surrounding fluid. However, simulating these dynamics, which are governed by coupled ordinary and partial differential equations, is significantly more difficult than the multi-body dynamics of classical rigid robotic systems. We present a differentiable framework for simulating strongly coupled fluid-robot multiphysics as a unified optimization problem. The coupled manipulator and incompressible Navier-Stokes equations are derived together from a single Lagrangian using the principle of least action. We employ discrete variational mechanics to derive a stable, well-conditioned, and physically accurate scheme for jointly simulating articulated bodies and the surrounding fluid. We leverage the implicit function theorem to compute derivatives of the fully coupled dynamics. Using this simulator and its gradients, we realize undulating swimming gaits and optimize a highly dynamic C-start escape maneuver for a bioinspired eel robot. We validate both gaits on physical hardware, demonstrating successful sim-to-real transfer. Simulation code, hardware data, and schematics for the eel robot can be found here: this https URL
>
---
#### [replaced 021] Degradation-Aware Cooperative Multi-Modal GNSS-Denied Localization Leveraging LiDAR-Based Robot Detections
- **分类: cs.RO**

- **简介: 该论文属于多机器人协同定位任务，旨在解决GNSS拒止环境下的长期定位问题。通过融合异构传感器数据，提升在传感器退化情况下的定位精度。**

- **链接: [https://arxiv.org/pdf/2510.20480](https://arxiv.org/pdf/2510.20480)**

> **作者:** Václav Pritzl; Xianjia Yu; Tomi Westerlund; Petr Štěpán; Martin Saska
>
> **备注:** Preprint version. This work has been submitted to Elsevier for possible publication
>
> **摘要:** Accurate long-term localization using onboard sensors is crucial for robots operating in Global Navigation Satellite System (GNSS)-denied environments. While complementary sensors mitigate individual degradations, carrying all the available sensor types on a single robot significantly increases the size, weight, and power demands. Distributing sensors across multiple robots enhances the deployability but introduces challenges in fusing asynchronous, multi-modal data from independently moving platforms. We propose a novel adaptive multi-modal multi-robot cooperative localization approach using a factor-graph formulation to fuse asynchronous Visual-Inertial Odometry (VIO), LiDAR-Inertial Odometry (LIO), and 3D inter-robot detections from distinct robots in a loosely-coupled fashion. The approach adapts to changing conditions, leveraging reliable data to assist robots affected by sensory degradations. A novel interpolation-based factor enables fusion of the unsynchronized measurements. LIO degradations are evaluated based on the approximate scan-matching Hessian. A novel approach of weighting odometry data proportionally to the Wasserstein distance between the consecutive VIO outputs is proposed. A theoretical analysis is provided, investigating the cooperative localization problem under various conditions, mainly in the presence of sensory degradations. The proposed method has been extensively evaluated on real-world data gathered with heterogeneous teams of an Unmanned Ground Vehicle (UGV) and Unmanned Aerial Vehicles (UAVs), showing that the approach provides significant improvements in localization accuracy in the presence of various sensory degradations.
>
---
#### [replaced 022] ROOM: A Physics-Based Continuum Robot Simulator for Photorealistic Medical Datasets Generation
- **分类: cs.RO**

- **简介: 该论文提出ROOM，一个用于生成医学数据的物理连续机器人模拟器，解决医疗训练数据不足问题。通过CT扫描生成逼真图像和传感器数据，应用于医疗机器人任务如姿态估计和深度估计。**

- **链接: [https://arxiv.org/pdf/2509.13177](https://arxiv.org/pdf/2509.13177)**

> **作者:** Salvatore Esposito; Matías Mattamala; Daniel Rebain; Francis Xiatian Zhang; Kevin Dhaliwal; Mohsen Khadem; Subramanian Ramamoorthy
>
> **摘要:** Continuum robots are advancing bronchoscopy procedures by accessing complex lung airways and enabling targeted interventions. However, their development is limited by the lack of realistic training and test environments: Real data is difficult to collect due to ethical constraints and patient safety concerns, and developing autonomy algorithms requires realistic imaging and physical feedback. We present ROOM (Realistic Optical Observation in Medicine), a comprehensive simulation framework designed for generating photorealistic bronchoscopy training data. By leveraging patient CT scans, our pipeline renders multi-modal sensor data including RGB images with realistic noise and light specularities, metric depth maps, surface normals, optical flow and point clouds at medically relevant scales. We validate the data generated by ROOM in two canonical tasks for medical robotics: multi-view pose estimation and monocular depth estimation, demonstrating diverse challenges that state-of-the-art methods must overcome to transfer to these medical settings. Furthermore, we show that the data produced by ROOM can be used to fine-tune existing depth estimation models to overcome these challenges, also enabling other downstream applications such as navigation. We expect that ROOM will enable large-scale data generation across diverse patient anatomies and procedural scenarios that are challenging to capture in clinical settings. Code and data: this https URL.
>
---
#### [replaced 023] DSSE: a drone swarm search environment
- **分类: cs.LG; cs.AI; cs.RO; eess.SY**

- **简介: 该论文介绍DSSE，一个基于PettingZoo的无人机群搜索环境，用于研究需要动态概率输入的强化学习算法。任务是目标搜索，解决未知目标位置与距离奖励的问题。**

- **链接: [https://arxiv.org/pdf/2307.06240](https://arxiv.org/pdf/2307.06240)**

> **作者:** Manuel Castanares; Luis F. S. Carrete; Enrico F. Damiani; Leonardo D. M. de Abreu; José Fernando B. Brancalion; Fabrício J. Barth
>
> **备注:** 7 pages
>
> **摘要:** The Drone Swarm Search project is an environment, based on \textsc{PettingZoo}, that is to be used in conjunction with multi-agent (or single-agent) reinforcement learning algorithms. It is an environment in which the agents (drones), have to find the targets (shipwrecked people). The agents do not know the position of the target and do not receive rewards related to their own distance to the target(s). However, the agents receive the probabilities of the target(s) being in a certain cell of the map. The aim of this project is to aid in the study of reinforcement learning algorithms that require dynamic probabilities as inputs. A peer-reviewed paper describing version 2 of this software has been published in JOSS: this https URL.
>
---
#### [replaced 024] Emerging Extrinsic Dexterity in Cluttered Scenes via Dynamics-aware Policy Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，解决 cluttered 场景下的 extrinsic dexterity 问题。通过 DAPL 框架学习接触动力学，提升非抓取操作的性能。**

- **链接: [https://arxiv.org/pdf/2603.09882](https://arxiv.org/pdf/2603.09882)**

> **作者:** Yixin Zheng; Jiangran Lyu; Yifan Zhang; Jiayi Chen; Mi Yan; Yuntian Deng; Xuesong Shi; Xiaoguang Zhao; Yizhou Wang; Zhizheng Zhang; He Wang
>
> **备注:** Accepted to Robotics: Science and Systems (RSS) 2026. Project page: this https URL
>
> **摘要:** Extrinsic dexterity leverages environmental contact to overcome the limitations of prehensile manipulation. However, achieving such dexterity in cluttered scenes remains challenging and underexplored, as it requires selectively exploiting contact among multiple interacting objects with inherently coupled dynamics. Existing approaches lack explicit modeling of such complex dynamics and therefore fall short in non-prehensile manipulation in cluttered environments, which in turn limits their practical applicability in real-world environments. In this paper, we introduce a Dynamics-Aware Policy Learning (DAPL) framework that can facilitate policy learning with a learned representation of contact-induced object dynamics in cluttered environments. This representation is learned through explicit world modeling and used to condition reinforcement learning, enabling extrinsic dexterity to emerge without hand-crafted contact heuristics or complex reward shaping. We evaluate our approach in both simulation and the real world. Our method outperforms prehensile manipulation, human teleoperation, and prior representation-based policies by over 25% in success rate on unseen simulated cluttered scenes with varying densities. The real-world success rate reaches around 50% across 10 cluttered scenes, while a practical grocery deployment further demonstrates robust sim-to-real transfer and applicability.
>
---
#### [replaced 025] Rectified Schrödinger Bridge Matching for Few-Step Visual Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉导航任务，解决生成策略需多步积分的问题。提出RSBM框架，通过调节熵正则化参数，在少步数内实现高效稳定导航。**

- **链接: [https://arxiv.org/pdf/2604.05673](https://arxiv.org/pdf/2604.05673)**

> **作者:** Wuyang Luan; Junhui Li; Weiguang Zhao; Wenjian Zhang; Tieru Wu; Rui Ma
>
> **备注:** 18 pages, 7 figures, 10 tables. Code available at this https URL
>
> **摘要:** Visual navigation is a core challenge in Embodied AI, requiring autonomous agents to translate high-dimensional sensory observations into continuous, long-horizon action trajectories. While generative policies based on diffusion models and Schrödinger Bridges (SB) effectively capture multimodal action distributions, they require dozens of integration steps due to high-variance stochastic transport, posing a critical barrier for real-time robotic control. We propose Rectified Schrödinger Bridge Matching (RSBM), a framework that exploits a shared velocity-field structure between standard Schrödinger Bridges ($\varepsilon=1$, maximum-entropy transport) and deterministic Optimal Transport ($\varepsilon\to 0$, as in Conditional Flow Matching), controlled by a single entropic regularization parameter $\varepsilon$. We prove two key results: (1) the conditional velocity field's functional form is invariant across the entire $\varepsilon$-spectrum (Velocity Structure Invariance), enabling a single network to serve all regularization strengths; and (2) reducing $\varepsilon$ linearly decreases the conditional velocity variance, enabling more stable coarse-step ODE integration. Anchored to a learned conditional prior that shortens transport distance, RSBM operates at an intermediate $\varepsilon$ that balances multimodal coverage and path straightness. Empirically, while standard bridges require $\geq 10$ steps to converge, RSBM achieves over 94% cosine similarity and 92% success rate in merely 3 integration steps -- without distillation or multi-stage training -- substantially narrowing the gap between high-fidelity generative policies and the low-latency demands of Embodied AI.
>
---
