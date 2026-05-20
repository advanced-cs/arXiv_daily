# 机器人 cs.RO

- **最新发布 59 篇**

- **更新 32 篇**

## 最新发布

#### [new 001] FlyMirage: A Fully Automated Generation Pipeline for Diverse and Scalable UAV Flight Data via Generative World Model
- **分类: cs.RO**

- **简介: 该论文提出FlyMirage，用于生成多样化、可扩展的无人机飞行数据集，解决航空视觉-语言导航数据不足的问题。通过大语言模型和生成式世界模型自动化构建高真实感场景，生成可行轨迹。**

- **链接: [https://arxiv.org/pdf/2605.19600](https://arxiv.org/pdf/2605.19600)**

> **作者:** Jinhan Li; Xijie Huang; Zhaoqi Wang; Yijin Wang; Weiqi Ge; Qiyi He; Mo Zhu; Fei Gao; Yuze Wu; Xin Zhou
>
> **摘要:** In the field of Vision-Language Navigation (VLN), aerial datasets remain limited in their ability to combine scale, diversity, and realism, often relying on either costly real-world scenes or visually limited simulations. To address these challenges, we introduce FlyMirage, a highly scalable and fully automated data generation pipeline for aerial VLN. Our approach leverages large language models (LLM) as an environment designer to promote scene diversity, paired with a generative world model that instantiates these designs into high-fidelity 3D Gaussian Splatting (3DGS) scenes. To substantially reduce human labor and ensure the feasibility of flight data, FlyMirage automates scene exploration and semantic information acquisition, and further integrates a dynamically feasible planner for uncrewed aerial vehicle (UAV) trajectory generation. Utilizing this toolchain, we generate a large-scale, diverse, and photorealistic aerial VLN dataset, with dynamically feasible flying trajectories, designed to support the development of next-generation embodied navigation models.
>
---
#### [new 002] Adversarial Stress Testing of SPARK Humanoid Safety Filters
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人安全控制任务，旨在评估SPARK humanoid安全滤波器的鲁棒性。通过压力测试和指标分析，揭示其在复杂环境中的失效模式。**

- **链接: [https://arxiv.org/pdf/2605.19009](https://arxiv.org/pdf/2605.19009)**

> **作者:** Saurav Ghosh; Abdou Sow; Luke Zhang
>
> **备注:** 5 pages, 7 figures, 1 table. Code available at this https URL
>
> **摘要:** Humanoid robots are difficult to deploy safely because they have high-dimensional bodies, many collision constraints, and must operate near people and obstacles. Safety filters help by modifying a nominal control action when it may violate collision-avoidance constraints. Still, nominal benchmark scores do not fully show how these filters behave in harder environments. In this work, we study the robustness of SPARK humanoid safety filters through replication and stress testing. We replicate the SPARK benchmark case G1SportMode_D1_WG_SO_v1 in MuJoCo and evaluate RSSA, RSSS, SSA, CBF, PFM, and SMA under controlled random seeds. We also built a post-processing pipeline that converts raw SPARK logs into goal-tracking, minimum-distance, and collision-step metrics. Our results show that some methods track the goal more closely, while others reduce collision steps more effectively. The stress tests further indicate that safety behavior can change under obstacle crowding, noisy distance estimates, and delayed obstacle information. These findings suggest that humanoid autonomy should be evaluated beyond nominal performance, using metrics that expose failure modes before deployment.
>
---
#### [new 003] CLUE: Adaptively Prioritized Contextual Cues by Leveraging a Unified Semantic Map for Effective Zero-Shot Object-Goal Navigation
- **分类: cs.RO**

- **简介: 该论文属于零样本物体目标导航任务，解决导航中上下文线索重要性不确定的问题。通过构建统一语义地图，自适应地平衡房间与物体线索，提升导航效果。**

- **链接: [https://arxiv.org/pdf/2605.19206](https://arxiv.org/pdf/2605.19206)**

> **作者:** Taeyun Kim; Alvin Jinsung Choi; Dasol Hong; Hyun Myung
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Zero-shot object-goal navigation (ZSON) is a challenging problem in robotics that requires a comprehensive understanding of both language and visual observations. Contextual cues from rooms and objects are critical, but their relative importance depends on the target: some objects are strongly tied to specific room types, while others are better predicted by nearby co-located objects. Existing methods overlook this distinction, leading to inefficient and inaccurate exploration. We present CLUE, a novel navigation framework that adaptively balances the use of contextual rooms and objects by leveraging commonsense knowledge extracted from an offline large language model (LLM). By estimating a target's association with room types using LLM, the agent prioritizes room cues for predictable objects and object cues for those with weak room associations. Our framework constructs a unified semantic value map that integrates both types of contextual information, adaptively weighted by the target's ambiguity to guide exploration. Combined with multi-viewpoint verification and an exploration strategy informed by contextual cues, CLUE achieves robust and efficient navigation. Extensive experiments in simulation and real-world deployments show that our method consistently outperforms state-of-the-art baselines in both success rate (SR) and success weighted by path length (SPL), demonstrating its effectiveness and practicality for real-world navigation tasks.
>
---
#### [new 004] Self-assembling Modular Aerial Robot for Versatile Aerial Tasks
- **分类: cs.RO**

- **简介: 该论文提出LEGION模块化空中机器人，解决空中灵巧操作与稳定抓取的矛盾。通过自组装实现灵活飞行与协作操作，完成推、拉、转、抓等任务。**

- **链接: [https://arxiv.org/pdf/2605.19431](https://arxiv.org/pdf/2605.19431)**

> **作者:** Junichiro Sugihara; Masaki Kitagawa; Jinjie Li; Yunong Li; Takuzumi Nishio; Kei Okada; Moju Zhao
>
> **摘要:** Multirotor aerial robots excel at maneuvering in three-dimensional space, and recent advances enable nimble navigation in cluttered and confined environments, especially for small airframes. By contrast, platforms built for high-altitude work tend to be larger to deliver high thrust for stable physical interaction with the environment. However, these conflicting design requirements create a long-standing trade-off between nimble navigation and robust aerial manipulation. Here, we present LEGION units, which are reconfigurable modular aerial robots capable of in-flight self-assembly for cooperative manipulation, drawing inspiration from the self-organized collectives formed by ants. Each unit retains nimble maneuverability while joint-equipped docking interfaces at both ends enable end-to-end self-assembly into a flying manipulator. We show that multiple units autonomously dock in flight; once latched, they maintain a zero-clearance interlock by controlling the contact force and torque, enabling reliable aggregation and articulated motion even outdoors. We further show that self-reconfigurability enables morphological switching between nimble individual flight and collective articulated manipulation, while realizing core in-flight manipulation primitives including pushing, pulling, rotating, grasping, and carrying. LEGION's self-organization enables aerial robots, especially in swarms, to shift from passive observers to active participants in their environment, broadening the scope of aerial physical interaction.
>
---
#### [new 005] MCNav: Memory-Aware Dynamic Cognitive Map for Zero-shot Goal-oriented Navigation
- **分类: cs.RO**

- **简介: 该论文属于零样本目标导航任务，旨在解决导航中因遗漏或误识别目标导致的失败问题。提出MCNav框架，利用动态认知地图和记忆策略提升导航准确性。**

- **链接: [https://arxiv.org/pdf/2605.19594](https://arxiv.org/pdf/2605.19594)**

> **作者:** Jingyu Li; Zhe Liu; Wenxiao Wu; Li Zhang
>
> **摘要:** Navigating to instance-level targets in complex environments is a challenging problem. Many existing zero-shot methods achieve strong performance by modeling the entire environment and leveraging large language models for scene understanding. However, such strategies primarily focus on exploring new regions while lacking a deeper exploitation of information from previously explored areas. Consequently, when targets are missed or misidentified within previously visited regions, navigation failures occur frequently. To address these limitations, we propose MCNav, a memory-aware navigation framework with a dynamic cognitive map. This map stores efficiently queryable information about relevant objects in explored areas. Building on this memory structure, MCNav introduces two memory-aware exploration strategies: goal re-validation, which re-assesses previously seen objects to correct matching failures, and missed goal re-exploration, which estimates the likelihood that a target is present in an explored region from contextual cues. These strategies are further stabilized by a blacklist mechanism to prevent repeated errors and a double-check mechanism for high-confidence confirmation. We evaluate MCNav on the HM3Dv1 and HM3Dv2 datasets across three different tasks, where it achieves state-of-the-art performance, particularly on the instance-level goal navigation task.
>
---
#### [new 006] KIO-planner: Attention-Guided Single-Stage Motion Planning with Dual Mapping for UAV Navigation
- **分类: cs.RO**

- **简介: 该论文属于无人机路径规划任务，解决复杂环境中低延迟、安全导航问题。提出KIO-planner，结合注意力机制和双映射策略，提升规划效率与安全性。**

- **链接: [https://arxiv.org/pdf/2605.19703](https://arxiv.org/pdf/2605.19703)**

> **作者:** Dexing Yao; Haochen Li; Junhao Wei; Yifu Zhao; Yanxiao Li; Jiahui Xu; Jinxuan Hu; Lele Tian; Baili Lu; Zikun Li; Xu Yang; Sio-Kei Im; Dingcheng Yang; Yapeng Wang
>
> **备注:** Accepted by an IEEE Vehicular Technology Conference. 6 pages, 4 figures, 1 table
>
> **摘要:** Autonomous UAV flight in confined, wall-dense environments requires low-latency and reliable motion planning under strict safety constraints. Traditional optimization-based planners suffer from mapping latency and easily fall into local minima when navigating through dense structural obstacles. Meanwhile, existing end-to-end learning methods struggle to extract fine-grained geometric features from raw depth images and lack hard kinodynamic constraints, leading to unpredictable collisions near walls. To address these issues, we propose KIO-planner, an attention-guided single-stage trajectory planning framework. First, we integrate a Convolutional Block Attention Module (CBAM) into the perception backbone to adaptively focus on critical structural edges and traversable space. Second, we introduce a novel Dual Mapping mechanism--comprising physical bounds activation and a deterministic Geometric Safety Shield in the depth-pixel space--to enforce kinodynamic feasibility and collision-free flight without global map fusion. Extensive high-fidelity simulated experiments demonstrate that KIO-planner enables highly agile navigation at speeds up to 3.0 m/s. Compared to the state-of-the-art baseline, KIO-planner achieves lower inference latency (approximately 24 ms) and generates significantly smoother trajectories, reducing control cost by 28.4%. Most notably, our Dual Mapping substantially increases the worst-case safety margin, measured by minimum distance to obstacles, from 0.48 m to 0.76 m, ensuring fast, smooth, and safer navigation in highly constrained environments.
>
---
#### [new 007] PAPO-VLA: Planning-Aware Policy Optimization for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型的优化任务，旨在提升VLA策略的可靠性。针对任务执行中规划动作识别不足的问题，提出PAPO-VLA方法，通过分析动作重要性增强优化效果。**

- **链接: [https://arxiv.org/pdf/2605.19580](https://arxiv.org/pdf/2605.19580)**

> **作者:** Peizheng Guo; Jingyao Wang; Changwen Zheng; Wenwen Qiang
>
> **摘要:** Vision-Language-Action (VLA) models show promising ability in language-guided robotic tasks. However, making VLA policies reliable remains challenging, because a manipulation task is completed through closed-loop interaction, where each action affects subsequent execution. To analyze this problem, we revisit VLA policy during execution and argue that a VLA policy acts both as a planner, which makes task-oriented decisions that change the direction of execution, and as an executor, which realizes these decisions through dense continuous actions. This view suggests that improving VLA reliability requires particular attention to planning actions. Existing optimization methods can imitate actions or improve complete trajectories, but they usually do not explicitly identify planning actions or measure their importance for task success. To address this issue, we propose Planning-Aware Policy Optimization for VLA models (PAPO-VLA). PAPO-VLA first identifies planning actions by jointly considering action variation and trajectory outcome, then estimates their importance through causal sufficiency and causal necessity, and finally incorporates this importance into GRPO advantage estimation. In this way, more important planning actions receive stronger optimization emphasis, while the whole trajectory is still optimized by trajectory-level feedback. Experiments on multiple benchmarks demonstrate the effectiveness of PAPO-VLA.
>
---
#### [new 008] RoVLA: Multi-Consistency Constraints for Robust Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于具身控制任务，旨在解决VLA模型在视觉变化、指令改写等场景下的脆弱性问题。提出RoVLA框架，通过多一致性约束提升模型鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.19678](https://arxiv.org/pdf/2605.19678)**

> **作者:** Jingzhou Luo; Yifan Wen; Yongjie Bai; Xinshuai Song; Yang Liu; Liang Lin
>
> **摘要:** Vision-Language-Action (VLA) models have shown strong performance on embodied manipulation, yet they remain brittle under visual observation changes, paraphrased language instructions, and compounded perturbations. This limitation suggests that existing methods still rely heavily on shallow correlations in the training distribution, rather than learning stable couplings among task semantics, environment states, and action generation. Although recent efforts improve robustness through larger-scale training, post-training adaptation, or enhanced predictive modeling, they rarely enforce invariance-oriented consistency within the end-to-end policy itself. To address this issue, we propose RoVLA, a robust vision-language-action framework with multi-consistency constraints. RoVLA enforces consistency under three complementary transformations: instruction semantics, trajectory evolution, and observation perturbation. Specifically, Instructional Consistency (IC) promotes stable grounding under semantically equivalent instruction rewrites, Evolutionary Consistency (EC) preserves coherent action intent throughout the generation process, and Observational Consistency (OC) improves robustness to visual and proprioceptive perturbations by enforcing consistent predictions before and after targeted disturbances. By explicitly modeling these invariances during training, RoVLA reduces reliance on superficial correlations and improves robustness and generalization. Experiments on LIBERO-Plus, RoboTwin 2.0, and real-world manipulation tasks show that RoVLA consistently outperforms strong baseline methods and exhibits superior robustness under diverse task and observation shifts. These results demonstrate the effectiveness of multi-consistency learning for robust embodied control. Codes will be available at this https URL.
>
---
#### [new 009] HEAT: Heterogeneous End-to-End Autonomous Driving via Trajectory-Guided World Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决多领域泛化问题。通过轨迹驱动学习和世界模型，提升模型在异构环境中的性能。**

- **链接: [https://arxiv.org/pdf/2605.19631](https://arxiv.org/pdf/2605.19631)**

> **作者:** Hoonhee Cho; Giwon Lee; Jae-Young Kang; Hyemin Yang; Heejun Park; Kuk-Jin Yoon
>
> **摘要:** End-to-end autonomous driving has emerged as a compelling alternative to traditional modular pipelines by directly mapping raw sensor data to driving actions. While recent approaches achieve strong performance on single-domain datasets, their performance degrades significantly when trained jointly across multiple heterogeneous domains. In practice, however, autonomous systems must operate across diverse environments with heterogeneous distributions, including different cities, sensor configurations, and traffic patterns, without domain-specific retraining. This gap highlights a key challenge in multi-domain learning: domain-specific variations across heterogeneous domains introduce conflicting learning signals, driving models toward compromised solutions that are suboptimal across domains. To address this, we propose a trajectory-driven learning paradigm that organizes training around planning trajectories, enabling the model to capture domain-invariant representations of driving intent. Furthermore, we incorporate a world model that predicts future latent features conditioned on ego actions, improving feature consistency and mitigating domain-induced biases. We evaluate our approach on three benchmarks, nuScenes, NAVSIM, and the Waymo end-to-end dataset, and show substantial improvements over existing methods across all domains. Our results demonstrate that a single unified model can be trained on heterogeneous datasets while maintaining strong performance within each domain, highlighting a step toward scalable real-world deployment. We will make our code publicly available.
>
---
#### [new 010] DEFLECT: Delay-Robust Execution via Flow-matching Likelihood-Estimated Counterfactual Tuning for VLA Policies
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出DEFLECT方法，解决异步VLA控制中的预测执行错位问题。通过离线优化提升延迟容忍度，无需标签或奖励模型，显著提高任务成功率。**

- **链接: [https://arxiv.org/pdf/2605.19294](https://arxiv.org/pdf/2605.19294)**

> **作者:** Yixiang Zhu; Yonghao Chen; Rui Meng; Jingyu Guo; Jiaxiang Zou; Zijie Yang; Taowen Wang; Xinyu Chen
>
> **摘要:** Vision-Language-Action (VLA) policies are typically deployed with asynchronous inference: the robot executes a previously predicted action chunk while the model computes the next one. This creates a prediction-execution misalignment: the chunk is conditioned on the observation taken before inference began, but executes in a physical state that has already drifted forward by several control steps; naive asynchronous rollover collapses from 89% to under 1% on Kinetix as the inference cycle covers up to seven control steps. We introduce DEFLECT, a fully offline post-training refinement that applies as a near drop-in upgrade to existing async-VLA stacks by converting latency itself into a label-free preference signal: counterfactual fresh/stale action pairs are constructed from a frozen reference policy and scored under the deployment-time conditioning via an implicit flow-matching likelihood-ratio surrogate, with no human labels, reward models, or online rollouts. DEFLECT substantially extends the usable delay envelope of async VLA control, with +6.4 success-rate gain in the high-latency regime (5-7 control steps), +4.6 when transferred to a real-scale VLA at the longest delay, and consistent improvements on two real-robot tasks (a bimanual conveyor pick-and-place and a reactive whack-a-mole).
>
---
#### [new 011] D-CLING: Prior-Preserving Depth-Conditioned Fine-Tuning for Navigation Foundation Models
- **分类: cs.RO**

- **简介: 该论文属于导航任务，针对导航基础模型微调后泛化能力下降的问题，提出D-CLING方法，在保留预训练知识的同时提升新场景下的导航性能。**

- **链接: [https://arxiv.org/pdf/2605.19690](https://arxiv.org/pdf/2605.19690)**

> **作者:** Shintaro Nakaoka; Takayuki Kanai; Kazuhito Tanaka
>
> **备注:** This paper has been accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA 2026), which will be held in Vienna, Austria, from June 1 to 5, 2026
>
> **摘要:** Navigation Foundation Models (NFMs) trained on large cross-embodied datasets have demonstrated powerful generalizability in various scenarios. Adopting in-domain fine-tuning for an NFM efficiently calibrates the visuomotor policy, promising further improvement even in a novel scenario. However, the fine-tuned models still suffer from poor obstacle avoidance or fail to properly reach the provided goals. Furthermore, model updates using a small subset of data typically erode the pre-trained prior, compromising the pre-training generalization. Consequently, fine-tuning deteriorates the capability of the model for robust and accurate navigation. In this work, we present a novel fine-tuning method that leverages large-scale pre-training while efficiently learning in novel setups, such as environments or camera configurations. In particular, inspired by ControlNet, we fine-tune an NFM by attaching a trainable copy of the pre-trained backbone using zero-initialized residual pathways, thereby learning geometric cues. This design enables the model to efficiently acquire in-domain geometry while preserving pre-trained knowledge across various behaviors. Despite its simplicity, our comprehensive evaluation of real-world navigation suggests that our proposal effectively enables robust long-horizon navigation with minimal collisions and human intervention. Additionally, our offline analysis shows that the proposed method maintains or further improves action prediction capabilities beyond the fine-tuned dataset, providing a key insight into continual learning for general navigation. The project page: this https URL
>
---
#### [new 012] COBALT: Crowdsourcing Robot Learning via Cloud-Based Teleoperation with Smartphones
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出COBALT平台，解决机器人学习中高质量示范数据不足的问题。通过云端遥控实现大规模、低成本的机器人学习，支持多用户并发操作，提升数据收集效率与质量。**

- **链接: [https://arxiv.org/pdf/2605.19138](https://arxiv.org/pdf/2605.19138)**

> **作者:** Ayush Agarwal; Ansh Gandhi; Jeremy A. Collins; Omar Rayyan; Aryan Sarswat; Ranjani Koushik; Masoud Moghani; Ajay Mandlekar; Animesh Garg
>
> **摘要:** The scarcity of large-scale, high-quality demonstration data remains a bottleneck in scaling imitation learning for robotic manipulation. We present COBALT, a teleoperation platform designed to democratize robot learning at scale both in simulation and in the real world. By leveraging vectorized environments, our scalable, load-balanced infrastructure supports concurrent teleoperation by multiple users on a single GPU, yielding a significant reduction in teleoperation cost. Operators can connect from nearly anywhere on Earth using commonly available devices, including single or dual smartphones, VR headsets, 3D mice, and keyboards. An inmemory data cache and efficient video streaming keep control and rendering synchronous, sustaining dozens of concurrent users at 20 Hz with sub-100 ms end-to-end latency for up to 8 concurrent users per GPU. We also demonstrate stable operation supporting 256 simulated clients across 8 GPUs, underscoring the system's ability to scale across hardware and within individual servers. We perform a comprehensive user study showing that phone-based teleoperation performs comparably to or better than specialized hardware, enabling faster, more ergonomic data collection. To ensure data quality, COBALT logs a suite of real-time metrics to automatically filter suboptimal demonstrations. We further demonstrate that a structured user training curriculum significantly improves data collection quality. Guided by insights from our user study, we crowdsource the collection of a large-scale, high-quality pilot dataset with 7500+ demonstrations (50+ hours) collected with smartphones across nine countries over five days. We validate the dataset's quality by training state-of-the-art imitation learning algorithms. Please visit \href{this https URL}{this http URL} for more details.
>
---
#### [new 013] Multi-Session Ground Texture SLAM in Low-Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文属于SLAM任务，解决多时段低动态环境下的地面纹理定位与建图问题。通过分析三种技术，发现KL散度在轨迹估计中效果最佳，并发布了一个多时段数据集。**

- **链接: [https://arxiv.org/pdf/2605.19701](https://arxiv.org/pdf/2605.19701)**

> **作者:** Kyle M. Hart; Brendan Englot
>
> **备注:** 8 pages, 9 figures. To appear at the 23rd International Conference on Ubiquitous Robots, Osaka, Japan. Distribution Statement A: Approved for public release; distribution is unlimited, as submitted under NAVAIR Public Release Authorization 2025-0098
>
> **摘要:** The simultaneous localization and mapping community has introduced a growing number of systems adapted for multi-session operations where the operational environment features low-dynamic changes that impact mapping, such as surface wear, weather phenomena, or seasonal change. These systems allow for lifelong operations by a robot within these environments. There is also growing interest in operations in environments where the unique ground texture is the only mapping feature available for use. These ground texture systems are not yet targeted for multi-session low-dynamic-change environments though. This work explores the impact of three different techniques on trajectory estimation accuracy in these multi-session low-dynamic ground texture environments. Of the three, the use of Kullback-Leibler Divergence, as a similarity score and a bias influencing loop closure confidence, is found to have the most success. We show an analysis of all three methods and a deeper exploration of the impact of Kullback-Leibler Divergence. We also introduce a dataset for use by the robotics community that contains multi-session images where the ground changes between sessions and also high-accuracy pose information for use in evaluation.
>
---
#### [new 014] KG-ASG: Collision-Knowledge-Guided Closed-Loop Adversarial Scenario Generation With Primary-Support Attribution
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶安全验证任务，解决多车碰撞场景生成问题。提出KG-ASG框架，通过碰撞知识引导生成可解释、可控的对抗场景。**

- **链接: [https://arxiv.org/pdf/2605.18895](https://arxiv.org/pdf/2605.18895)**

> **作者:** Cheng Wang; Chen Xiong; Ziwen Wang; Yuchen Zhou; Qiang Liu
>
> **摘要:** Safety validation of autonomous driving systems requires high-risk scenario coverage, clear collision semantics, executable trajectories, and attributable multi-vehicle interactions. Existing safety-critical scenario generation methods often rely on low-level trajectory perturbations, collision-proxy optimization, or single-adversary search, which may produce adversarial samples with ambiguous collision causes or uncontrolled multi-vehicle collisions. This paper proposes KG-ASG, a collision-knowledge-guided closed-loop adversarial scenario generation framework with primary-support attribution. KG-ASG constructs a structured collision knowledge base and trains a lightweight Collision Expert to infer the target collision mode, the unique primary adversary, support vehicles, and their interaction roles. Guided by this semantic prior, multi-vehicle adversarial generation is formulated as a primary-support process, where the primary adversary induces the main conflict and support vehicles shape the surrounding risk structure without becoming additional colliders. Rule, physical, interaction-safety, and single-collider constraints are imposed as hard gates to filter non-executable samples. To handle reactive ego behaviors, planner-controller feedback is further used for failure diagnosis, candidate re-ranking, and terminal refinement. Experiments on WOMD scenarios reconstructed in MetaDrive show that KG-ASG achieves strong adversarial effectiveness while improving Valid Primary Attack, reducing multi-collision, and obtaining closed-loop recovery gains under IDM, Cruise, and Expert controllers. These results demonstrate that collision-knowledge guidance and primary-support single-collider reasoning improve adversarial effectiveness, interpretability, and executability for autonomous driving safety validation.
>
---
#### [new 015] Learning-Accelerated Optimization-based Trajectory Planning for Cooperative Aerial-Ground Handover Missions
- **分类: cs.RO; cs.LG; math.OC**

- **简介: 该论文属于多机器人协同任务，解决UAV与UGV交接轨迹规划问题。通过结合神经网络与优化方法，提升轨迹生成速度与成功率。**

- **链接: [https://arxiv.org/pdf/2605.19562](https://arxiv.org/pdf/2605.19562)**

> **作者:** Jingshan Chen; Bochen Yu; Henrik Ebel; Peter Eberhard
>
> **备注:** Preprint of a contribution accepted for publication in the RoManSy 2026 Springer proceedings
>
> **摘要:** This paper presents a learning-augmented trajectory planning framework for cooperative unmanned aerial vehicle (UAV) and unmanned ground vehicle (UGV) handover missions. While centralized trajectory optimization ensures dynamic feasibility and task optimality, its high computational cost limits real-time applicability. We propose a neural surrogate planner utilizing decoupled encoder-decoder long short-term memory (LSTM) networks to generate coordinated handover trajectory predictions from the task specifications. These predictions serve as informed warm starts for the downstream centralized optimizer, thereby accelerating convergence to dynamically feasible solutions. Benchmark evaluations demonstrate that the learning-augmented planning framework achieves more than a threefold speedup and 100% optimization success rate compared to cold start optimization. The results indicate that combining data-driven inference with model-based refinement enables fast and reliable trajectory generation for heterogeneous multi-robot systems.
>
---
#### [new 016] CEER: Compliant End-Effector and Root Control as a Unified Interface for Hierarchical Humanoid Loco-Manipulation
- **分类: cs.RO**

- **简介: 该论文提出CEER，用于解决人形机器人在复杂接触环境中的操作问题。通过统一的末端-根控制接口，实现模块化、可扩展的运动规划与控制。**

- **链接: [https://arxiv.org/pdf/2605.19981](https://arxiv.org/pdf/2605.19981)**

> **作者:** Xinyuan Luo; Xingrui Chen; Xunjian Yin; Hongxuan Wu; Boxi Xia; Zhuoqun Chen; Jinzhou Li; Boyuan Chen; Xianyi Cheng
>
> **备注:** Project page: this https URL. 9 pages, 7 figures
>
> **摘要:** Humanoid robots have achieved impressive locomotion performance, yet contact-rich and long-horizon manipulation remains a major bottleneck. Manipulation is inherently contact-rich and demands compliant whole-body control for stable interaction, while its diversity and long-horizon nature favor modular, planner-compatible interfaces over joint-space tracking. We propose CEER, a compliant end-effector-root (EE-root) control abstraction for modular humanoid loco-manipulation within a hierarchical planning framework. CEER enables compliance-aware whole-body control in an interpretable task space defined by root motion commands and end-effector pose targets, and supports plug-and-play integration with heterogeneous high-level planners. A teacher-student framework is adopted to distill a general motion-tracking controller into a low-level policy that consumes only EE-root commands. We further construct a hierarchical system that integrates heterogeneous planners and task modules through the EE-root interface, enabling diverse manipulation tasks without retraining the underlying whole-body policy. Experiments in simulation and on hardware demonstrate 3.3 cm end-effector tracking accuracy with substantially reduced jerk compared to baselines, stable contact-rich manipulation under teleoperation, and up to 70% success in simulated single-object loco-manipulation tasks within a room-scale environment. These results indicate that compliant EE-root control provides a practical abstraction for humanoid loco-manipulation, enabling modular and scalable integration of diverse skills.
>
---
#### [new 017] Automatically Improving Simulation Physics for Articulated Objects
- **分类: cs.RO**

- **简介: 该论文属于机器人学习中的物理仿真任务，解决人工构建模拟对象耗时的问题。通过多模态方法和模拟器反馈，自动提升对象的物理一致性与交互可靠性。**

- **链接: [https://arxiv.org/pdf/2605.19136](https://arxiv.org/pdf/2605.19136)**

> **作者:** Anh-Quan Pham
>
> **摘要:** Simulation is a central tool for scalable robot learning, but its effectiveness depends on the quality of object assets. While modern 3D datasets provide rich geometric and kinematic representations, they typically lack the physical properties required for stable and realistic interaction, requiring significant manual effort to construct simulation-ready articulated objects. In this thesis, we introduce interaction-readiness, which characterizes whether an object can be reliably simulated under manipulation. We propose a quantitative evaluation framework that decomposes interaction-readiness into measurable components, enabling systematic analysis of object quality and revealing failure modes not captured by conventional evaluation. We further present a multi-modal, simulator-in-the-loop approach for generating interaction-ready articulated objects from incomplete 3D assets. The method integrates geometric, visual, and semantic information to infer physical properties and refines them through iterative simulator feedback to improve physical consistency. Experiments across diverse articulated objects and manipulation tasks show that object quality directly impacts simulation stability, interaction behavior, and policy performance. Objects refined by our method exhibit more stable and realistic dynamics, enabling more reliable downstream learning and evaluation. Overall, this thesis demonstrates the importance of physical realism for articulated objects in simulation and introduces a practical multi-modal refinement approach, guided by simulator feedback, for constructing such objects at scale.
>
---
#### [new 018] CANINE: Coaching Visually Impaired Users for Interactive Navigation with a Robot Guide Dog
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机协作导航任务，旨在解决视觉障碍者与机器人导盲犬协调困难的问题。通过CANINE系统，提供个性化反馈以提升用户导航能力。**

- **链接: [https://arxiv.org/pdf/2605.19501](https://arxiv.org/pdf/2605.19501)**

> **作者:** Cunjun Yu; Zishuo Wang; Anxing Xiao; Linfeng Li; David Hsu
>
> **备注:** Accepted to RSS 2026
>
> **摘要:** Robot guide dogs offer navigation assistance that greatly expands the independent mobility of the visually impaired, but their effective use requires subtle human-robot coordination that is difficult for users to learn from generic verbal instructions. To tackle this challenge, we present CANINE, an automated coaching system that trains users for interactive navigation with a robot guide dog, through personalized, adaptive verbal feedback. CANINE decomposes a complex coordination task into sub-skills and operates at two levels. At the high level, it decides what to train by tracking the learner's proficiency across sub-skills using knowledge tracing and prioritizing training on the weakest areas. At the low level, CANINE decides how to train each sub-skill by observing each human practice episode, using foundation models to infer the underlying causes of errors, and generating targeted verbal corrections adaptively. A controlled study with blindfolded participants, treated as a proxy population for quantitative evaluation, demonstrates that CANINE significantly improves both learning efficiency and final navigation performance compared to generic verbal instructions. We further validate CANINE through a retention study and an exploratory case study. The retention study shows lasting skill improvement after two weeks. The case study confirms CANINE's effectiveness in training a visually impaired user, while revealing additional design considerations for real-world deployment. Both are well aligned with the findings of the controlled study. Project page: this https URL
>
---
#### [new 019] Minimalist Visual Inertial Odometry
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于视觉-惯性里程计任务，解决资源消耗大的问题。通过四点光敏传感器和IMU实现高效精准的平面定位。**

- **链接: [https://arxiv.org/pdf/2605.19990](https://arxiv.org/pdf/2605.19990)**

> **作者:** Francesco Pasti; Jeremy Klotz; Nicola Bellotto; Shree K. Nayar
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Visual-Inertial Odometry(VIO), which is critical to mobile robot navigation, uses cameras with a large number of pixels. Capturing and processing camera images requires significant resources. This work presents a minimalist approach to planar odometry, demonstrating that just four visual measurements and an IMU can provide robust motion estimation for differential-drive robots. Our key insight is that four downward-facing photodiodes that sense the world through optical Gabor masks produce signals that encode speed. Based on this, we jointly optimize the mask parameters alongside a Temporal Convolutional Network (TCN) using a physically-grounded simulator. The resulting model decodes speed from just the four measurements produced by the photodiodes. Pairing these estimates with the angular speed from an IMU yields a continuous planar trajectory. We validate our approach with a prototype sensor mounted on a differential drive robot. Across diverse indoor and outdoor terrains, our system closely tracks the reference ground truth without any real-world fine-tuning. Our work shows that minimalist sensing enables efficient and accurate planar odometry.
>
---
#### [new 020] ARC-RL: A Reinforcement Learning Playground Inspired by ARC Raiders
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出ARC-RL环境，用于研究仿生机器人强化学习。任务是解决不同形态机器人在模拟中的运动控制问题，通过统一框架对比不同算法性能。**

- **链接: [https://arxiv.org/pdf/2605.19503](https://arxiv.org/pdf/2605.19503)**

> **作者:** Carlo Romeo; Andrew D. Bagdanov
>
> **摘要:** Reinforcement learning for legged locomotion has matured into a stack of multi-component reward functions and physics-engine benchmarks whose morphologies are uniformly derived from real commercial hardware. Game NPCs, however, are bound by stylistic constraints absent from sim-to-real robotics and routinely take the form of creatures with no real-robot counterpart. We introduce ARC-RL, a suite of four MuJoCo continuous-control environments featuring robotic morphologies inspired by the bestiary of ARC Raiders: the 18-DoF tall hexapod Queen, the 12-DoF armoured hexapod Bastion, the 18-DoF compact hexapod Tick, and the 12-DoF quadruped Leaper. All four robots share a unified observation template, action convention, simulation cadence, and a single closed-form multi-component reward function whose only per-morphology variation lives in a small set of weights and parameters. The reward fuses a velocity-tracking tent, a healthy survive bonus, a phase-locked gait-compliance bonus/cost pair, action regularisers, three safety penalties, and a posture anchor; no motion-capture data enters the reward at any point. We additionally provide hand-crafted Central Pattern Generator demonstrators per morphology, which serve both as fixed expert references and as sources of prior data for offline-to-online training. On this playground, we conduct a controlled empirical study comparing standard online algorithms (SAC, SPEQ, SOPE-EO) and methods augmented with prior data (SACfD, SPEQ-O2O, SOPE), and characterise how each paradigm copes with the playground's morphological diversity and animation-style stylistic constraints.
>
---
#### [new 021] Neuromorphic Control of a Flapping-Wing Robot on Resource-Constrained Hardware
- **分类: cs.RO**

- **简介: 该论文属于自主飞行控制任务，解决FWMAV在资源受限硬件上的实时控制问题。通过部署轻量SNN实现状态估计与控制，提升效率并降低功耗。**

- **链接: [https://arxiv.org/pdf/2605.19430](https://arxiv.org/pdf/2605.19430)**

> **作者:** Rim El Filali; Chenrui Feng; Chao Gao; Weibin Gu
>
> **摘要:** Flapping-Wing Micro Aerial Vehicles (FWMAVs) provide exceptional maneuverability and aerodynamic efficiency but pose significant challenges for onboard control due to nonlinear dynamics and stringent Size, Weight, and Power (SWaP) constraints, as exemplified by a butterfly-inspired robot less than 30 gram. To this end, we present a hierarchical neuromorphic control framework that enables fully onboard, closed-loop flight on a widely available, resource-constrained ESP32 microcontroller with a unit cost of approximately $5. Specifically, our method deploys two lightweight Spiking Neural Networks (SNNs) onboard: one for state estimation from raw sensory feedback and another for control via modulation of a Central Pattern Generator (CPG) for wing actuation. Trained by imitation learning, the system achieves stable pitch and heading angle tracking during untethered real-world flight. Experimental results further reveal that the SNN-based controller reduces latency by 36% (1059us to 680us) and power by 18% (0.033W to 0.027W) for inference compared to the conventional Artificial Neural Network (ANN) baseline, demonstrating the viability of spike-based computation without specialized hardware. To the best of our knowledge, this work constitutes the first demonstration of fully onboard neuromorphic control for autonomous flight of a FWMAV, highlighting the potential of SNNs to enable energy-efficient autonomy under stringent SWaP constraints. Visual abstract: this http URL
>
---
#### [new 022] A Heuristic Approach for Performance Tuning in RL-based Quadrotor Control via Reward Design and Termination Conditions
- **分类: cs.RO; cs.LG; math.OC**

- **简介: 该论文属于无人机控制任务，解决RL策略性能可调问题。通过设计奖励函数和终止条件，实现精准、可控的飞行性能调整。**

- **链接: [https://arxiv.org/pdf/2605.19166](https://arxiv.org/pdf/2605.19166)**

> **作者:** Fausto Mauricio Lagos Suarez; Akshit Saradagi; Vidya Sumathy; George Nikolakopoulos
>
> **备注:** Accepted in the 34th Mediterranean Conference on Control and Automation
>
> **摘要:** Reinforcement learning (RL)-based quadrotor control policies have achieved impressive performance in tasks such as fast navigation in cluttered environments and drone racing, where the focus is on speed and agility. However, in several applications, such as infrastructure inspection, it is critical to achieve precise, controlled maneuvers with tunable performance. In this article, we present a novel heuristic approach to achieve tunable performance in RL-based Quadrotor control through reward design and termination conditions. We present a novel reward structure containing dual bandwidth exponentials that achieves a baseline critically damped response in setpoint tracking, with low steady-state errors. When trained with a Proximal Policy Optimization (PPO) algorithm, in conjunction with episode truncation conditions, the desired performance is achieved in 6 million time steps in a sample-efficient manner. In order to tune the performance about the baseline behavior, we present intuitive heuristic rules to adjust the reward weights and exponential coefficients to achieve faster (acrobatic-like) and slower (inspection-like) settling time performance, while retaining the baseline critically damped response and approximately 2\% steady-state error. We evaluate the three RL policies (baseline, acrobatic, and inspection) across 100 trials and show accurate and tunable performance in position and yaw tracking from random initial conditions, thereby demonstrating the effectiveness of the proposed heuristic approach.
>
---
#### [new 023] Distributionally Robust Control via Stein Variational Inference for Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操控任务，解决接触丰富环境下的不确定性问题。通过分布鲁棒控制与Stein变分推断，提升模型的不确定性建模能力，增强控制可靠性。**

- **链接: [https://arxiv.org/pdf/2605.19029](https://arxiv.org/pdf/2605.19029)**

> **作者:** Hrishikesh Sathyanarayan; Victor Vantilborgh; Harish Ravichandar; Tom Lefebvre; Ian Abraham
>
> **备注:** In Proceedings of Robotics: Science and Systems, Sydney, Australia, July 2025
>
> **摘要:** Reliable robotic manipulation requires control policies that can accurately represent and adapt to uncertainty arising from contact-rich interactions. Modern data-driven methods mitigate uncertainty through large-scale training and computation, and degrade significantly in performance with limited number of training samples. By contrast, classical model-based controllers are computationally efficient and reliable, but their limited ability to represent task-relevant uncertainty can hinder performance in contact-rich interactions. In this work, we propose to expand the capabilities of model-based manipulation control through more flexible uncertainty modeling that retains performance while exactly adapting to uncertainty. Our approach casts the manipulation problem as a distributionally robust control optimization and proposes a novel deterministic formulation based on Stein variational inference that preserves performance while explicitly modeling task-sensitive parameter uncertainty. As a result, the derived controllers are more aware of task sensitivities to uncertainty, yielding high reliability without compromising performance. Experimental results demonstrate up to 3$\times$ improved robustness across a range of contact-rich manipulation tasks under broad parametric uncertainty, outperforming existing model-based control methods.
>
---
#### [new 024] Trajectory Planning and Control near the Limits: an Open Experimental Benchmark on the RoboRacer Platform
- **分类: cs.RO**

- **简介: 该论文属于自主驾驶轨迹规划与控制任务，旨在提升高加速场景下的性能。通过构建模块化框架，结合模型结构神经网络和在线速度重规划，优化路径跟踪与控制效果。**

- **链接: [https://arxiv.org/pdf/2605.19881](https://arxiv.org/pdf/2605.19881)**

> **作者:** Mattia Piccinini; Patrick Zambiasi; Aniello Mungiello; Mattia Piazza; Felix Jahncke; Johannnes Betz
>
> **备注:** Accepted - 2026 IEEE 29th International Conference on Intelligent Transportation Systems (ITSC)
>
> **摘要:** We present a modular framework to benchmark new and existing methods for trajectory planning and control in high-acceleration maneuvers that push autonomous driving to the limits. Our framework includes time-optimal raceline generation, online time-optimal velocity replanning, geometric path tracking controllers, and a new model-structured neural network (MS-NN) to learn the inverse dynamics for steering control. We deploy our framework on a 1:10-scale RoboRacer platform, using two circuits. Through several ablations with cautious and aggressive racelines, we study the performance of single modules and their combinations. We show that our MS-NN significantly improves tracking accuracy, decreases steering oscillations, and is physically interpretable. Moreover, online velocity replanning improves lap times by compensating for execution errors, and enables the vehicle to safely reach higher speeds and accelerations. To support future research, our code, datasets, videos and results are publicly available at this https URL.
>
---
#### [new 025] Justifying bio-inspired robotics research: A taxonomy of strategies
- **分类: cs.RO**

- **简介: 该论文属于分类任务，旨在解决生物启发机器人研究缺乏系统性解释的问题。提出一种动机分类体系，帮助研究人员和资助者评估不同生物启发方法的价值。**

- **链接: [https://arxiv.org/pdf/2605.19840](https://arxiv.org/pdf/2605.19840)**

> **作者:** Margaret J. Zhang; Justin Ting; Talia Y. Moore
>
> **摘要:** For most of human history, we have not thought systematically about how and why we incorporate aspects of the natural world into our designs. The lack of a systematic approach has resulted in inconsistencies in motivations and methods that make it difficult to predict or evaluate the success of bio-inspired design. This mismatch between expectations and results can lead to disappointment when a reader considers a bio-inspired design to be superficial, weak, or incomplete. This is especially true in the field of Robotics, in which similarity to a biological system might be the driving motivation for construction. In an effort to assist robotics researchers justify their specific bio-inspired approach and to assist funding program managers with discerning the value of different bio-inspired approaches, here we propose a taxonomy of motivations for bio-inspired design and describe the potential significant contributions that are likely to result from different approaches.
>
---
#### [new 026] Beyond Action Residuals: Real-World Robot Policy Steering via Bottleneck Latent Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决预训练策略在真实世界中的适应问题。通过引入瓶颈潜在空间，提升强化学习的在线微调效果，提高成功率和探索稳定性。**

- **链接: [https://arxiv.org/pdf/2605.19919](https://arxiv.org/pdf/2605.19919)**

> **作者:** Dongjie Yu; Kun Lei; Zhennan Jiang; Jia Pan; Huazhe Xu
>
> **摘要:** Pretrained imitation policies have become a strong foundation for robot manipulation, but they often require online improvement to overcome execution errors, limited dataset coverage, and deployment mismatch. A central question is therefore how reinforcement learning (RL) should adapt policies after offline pretraining. Existing lightweight methods commonly apply residual corrections directly in action space, but this often leads to noisy and poorly structured exploration. In this work, we propose Z-Perturbation Reinforcement Learning (ZPRL), an approach that steers pretrained policies through a compact bottleneck latent rather than through policy weights or output actions. During offline training, we augment the policy with a plug-and-play variational information bottleneck (VIB) module to extract a task-relevant latent interface from observation embeddings. During online finetuning, the base policy is frozen and RL learns only a residual perturbation on this latent, whose decoded representation conditions the frozen action generator. We instantiate ZPRL on flow-matching policies and evaluate it on eight simulation tasks and four real-world tasks. Across diverse manipulation settings, ZPRL improves both sample efficiency and final performance over strong post-training baselines. In the real world, ZPRL improves the average success rate on four tasks by 33.7% over imitation base policies while producing smoother exploration behaviors than an action residual counterpart. These results suggest that a compact, task-aligned bottleneck latent provides an effective interface for online RL adaptation. More videos can be found at this https URL.
>
---
#### [new 027] RoHIL: Robust Human-in-the-Loop Robotic Reinforcement Learning Against Illumination Variations
- **分类: cs.RO**

- **简介: 该论文属于机器人强化学习任务，解决光照变化导致的跨工作站性能下降问题。提出RoHIL框架，通过图像重渲染和数据级抗遗忘机制，提升模型在不同光照下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.19924](https://arxiv.org/pdf/2605.19924)**

> **作者:** Shuoqin Zhang; Yixin Xiong; Xiru Gao; Kai Liu; Ke Wang; Xichuan Zhou; Zhe Hu
>
> **摘要:** Human-in-the-loop reinforcement learning systems achieve near-perfect success on the workstation where they are trained, but collapse when the same robot is moved to a workstation a few meters away due to shifts in the visual input distribution caused by new lamp positions and window light. Re-collecting demonstrations and re-running HIL on every workstation is incompatible with deployment, and naively fine-tuning on shifted-light data triggers catastrophic forgetting of the source workstation. To close this cross-domain gap, we present RoHIL, an offline fine-tuning framework that uses no extra real-robot interaction. RoHIL combines (i) a world-model-based image relighter that re-synthesises the visual stream of source-workstation trajectories under multiple virtual HDRI environments, leaving actions and rewards real; (ii) Illumination-Retention Replay (IRR), a data-level anti-forgetting mechanism that interleaves relit adaptation transitions with original-light retention transitions to preserve source-workstation Bellman coverage; and (iii) an anchored Bellman-actor regulariser that constrains representation and policy drift from the original source-workstation policy. Across four real-robot manipulation tasks under significant cross-workstation illumination variations, RoHIL substantially improves shifted-light performance where standard HIL-RL collapses, while preserving source-workstation performance, eliminating the need to re-collect data and retrain for every new workstation and environment. Project page: this https URL
>
---
#### [new 028] TravExplorer: Cross-Floor Embodied Exploration via Traversability-Aware 3-D Planning
- **分类: cs.RO**

- **简介: 该论文提出TravExplorer，解决跨楼层的零样本目标导航问题。通过三维可行驶性规划与语义引导，实现复杂建筑环境中的自主探索。**

- **链接: [https://arxiv.org/pdf/2605.19958](https://arxiv.org/pdf/2605.19958)**

> **作者:** Han Zheng; Zhe Chen; Yudong Huang; Haoran Liu; Jinghao Wang; Ming Yang; Tong Qin
>
> **摘要:** Zero-shot Object Navigation (ZSON) has shown promise for open-vocabulary target search in unseen environments, yet most existing systems remain tied to planar representations and single-floor assumptions. These assumptions become inadequate in real buildings, where navigation involves floors, stairs, landings, and vertically overlapping spaces. This article presents TravExplorer, a cross-floor embodied exploration framework that couples zero-shot semantic guidance with traversability-aware 3-D planning. TravExplorer maintains a unified volumetric map that distinguishes occupied structures from robot-reachable support surfaces and extracts traversable frontiers from connected support surfaces, including floors, stairs, and landings. A FOV-aware active perception strategy further resolves incomplete observations during cross-floor traversal. To reduce semantic-reasoning latency, a lightweight guidance module aligns a probabilistic instance map from online open-vocabulary segmentation with a spatial value map from fast image-to-text matching. Based on these geometric and semantic memories, a hierarchical planner performs target-aware frontier touring over object hypotheses, traversable frontiers, and stair landmarks, and generates executable cross-floor motions through foothold-guided 3-D search and vertically constrained local trajectory optimization. Experiments over 4,195 simulated episodes on HM3D and MP3D demonstrate consistent advantages over representative ObjectNav baselines. Fifty real-world trials on a Unitree Go2 further validate open-vocabulary target search across single-floor and cross-floor indoor environments without prior maps or human intervention. The code will be released at this https URL.
>
---
#### [new 029] RLFTSim: Realistic and Controllable Multi-Agent Traffic Simulation via Reinforcement Learning Fine-Tuning
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文属于交通仿真任务，解决传统方法难以捕捉动态多智能体交互的问题。通过强化学习微调提升仿真真实性和可控性。**

- **链接: [https://arxiv.org/pdf/2605.19033](https://arxiv.org/pdf/2605.19033)**

> **作者:** Ehsan Ahmadi; Hunter Schofield; Behzad Khamidehi; Fazel Arasteh; Jinjun Shan; Lili Mou; Dongfeng Bai; Kasra Rezaee
>
> **备注:** CVPR 2026 Highlight; Project page at this https URL
>
> **摘要:** Supervised open-loop training has been widely adopted for training traffic simulation models; however, it fails to capture the inherently dynamic, multi-agent interactions common in complex driving scenarios. We introduce RLFTSim, a reinforcement-learning-based fine-tuning framework that enhances scenario realism by aligning simulator rollouts with real-world data distributions and provides a method for distilling goal-conditioned controllability in scenario generation. We instantiate RLFTSim on top of a pre-trained simulation model, design a reward that balances fidelity and controllability, and perform comprehensive experiments on the Waymo Open Motion Dataset. Our results show improvements in realism, achieving state-of-the-art performance. Compared with other heuristic search-based fine-tuning methods, RLFTSim requires significantly fewer samples due to a proposed low-variance and dense reward signal, and it directly addresses the realism alignment issue by design. We also demonstrate the effectiveness of our approach for distilling traffic simulation controllability through goal conditioning. The project page is available at this https URL.
>
---
#### [new 030] Geo-Data-Driven HD Map Generation Workflow with Integrated Reference-Free Constraint-Based Verification
- **分类: cs.RO**

- **简介: 该论文属于高精度地图生成任务，旨在解决依赖传感器和参考数据的问题。通过集成约束验证的地理数据驱动流程，实现无需外部参考的高质量地图生成。**

- **链接: [https://arxiv.org/pdf/2605.18921](https://arxiv.org/pdf/2605.18921)**

> **作者:** Ruidi He; Vaibhav Tiwari; Mohanad Al-Ghobari; Meng Zhang; Andreas Rausch
>
> **摘要:** High-definition (HD) maps are core artifacts for automated driving systems, but their generation commonly relies on sensor-intensive mobile mapping campaigns, while quality assessment often depends on high-precision reference data. These dependencies make HD map engineering costly and difficult to apply in settings where specialised measurement data or independently measured reference maps are unavailable. This paper presents an engineering-oriented geo-data-driven workflow for HD map generation with integrated representation-level verification. The workflow uses openly available geo-engineering datasets as the primary input source and transforms them into lane-level HD map representations of existing road environments through explicit intermediate representations and processing stages. To assess the generated representations without external reference maps, the workflow integrates executable constraint-based verification into the engineering process. Selected constraints are derived from specifications relevant to automated driving and road-design guidelines. They are evaluated directly on the generated lanelet-based representation to detect geometric, topological, and elevation-related inconsistencies. The workflow is evaluated using real-world shapefile-based road-network data from four cities in Lower Saxony, Germany, and controlled defect-injection scenarios. The real-world evaluation shows that the generated map representations satisfy the selected constraints in the evaluated scenarios, while the defect-injection study demonstrates complete detection of the considered defect types without observed false positives. The results indicate that geo-data-driven HD map generation with integrated executable verification can provide a modular and inspectable complement to sensor-intensive mapping workflows under reduced sensing and reference-data availability.
>
---
#### [new 031] PRISM-SLAM: Probabilistic Ray-Grounded Inference for Scale-aware Metric SLAM
- **分类: cs.RO**

- **简介: 该论文属于视觉SLAM任务，解决单目SLAM的尺度模糊和动态环境跟踪问题。通过融合视觉基础模型先验，构建贝叶斯因子图，实现尺度感知的精准定位与建图。**

- **链接: [https://arxiv.org/pdf/2605.19257](https://arxiv.org/pdf/2605.19257)**

> **作者:** Eunsoo Im
>
> **摘要:** Monocular SLAM historically suffers from scale ambiguity and tracking failure in dynamic environments. While recent vision foundation models (VFMs) provide remarkable zero-shot depth priors, naively integrating these deterministic predictions ignores predictive uncertainty and frame-to-frame scale inconsistencies. We propose PRISM-SLAM, a real-time framework that rigorously integrates VFM priors into a structured Bayesian factor graph to achieve scale-aware, metric-consistent localization and mapping. Specifically, we introduce a Plücker Ray-Distance Factor to anchor monocular observations in absolute space within a globally consistent metric coordinate system, mathematically resolving scale drift by making the metric scale Fisher-identifiable. To handle environmental dynamics, we derive an epistemic uncertainty proxy from temporal depth consistency and formulate a Dynamic Scene Uncertainty Gating (DSUG) mechanism. This soft-gating approach probabilistically down-weights dynamic distractors without incurring the heavy computational overhead associated with traditional semantic segmentation masks. By employing a multi-process architecture that asynchronously processes VFM inference and geometric tracking, PRISM-SLAM provides verified metric output at 30 FPS using solely RGB input, bridging the gap between foundation models and real-world robotic applications. Evaluated on the TUM RGB-D and 7-Scenes benchmarks, PRISM-SLAM achieves a metric $SE(3)$ Absolute Trajectory Error (ATE) nearly identical to its oracle-aligned $Sim(3)$ error. This demonstrates that our system can produce deployment-ready metric trajectories by delivering robust metric SLAM solutions without any post-hoc scale correction. Project page: this https URL
>
---
#### [new 032] Guiding Neuro-Symbolic Scenario Generation with Spatio-Temporal Logic
- **分类: cs.RO; cs.LG**

- **简介: 论文提出STRELGen框架，用于生成安全关键的自动驾驶场景。解决传统测试方法成本高、效率低的问题，结合扩散模型与时空逻辑，实现高效场景生成。**

- **链接: [https://arxiv.org/pdf/2605.19038](https://arxiv.org/pdf/2605.19038)**

> **作者:** Lorenzo Bonin; Francesco Giacomarra; Luca Bortolussi; Jyotirmoy V. Deshmukh; Francesca Cairoli
>
> **摘要:** The rapid advancement of autonomous driving (AD) technologies has outpaced the development of robust safety evaluation methods. Conventional testing relies on exposing AD systems to vast numbers of real-world traffic scenes -- a brute-force approach that is prohibitively expensive and statistically ineffective at capturing the rare, safety-critical edge cases essential for validating real-world robustness. To address this fundamental limitation, we introduce STRELGen, a scalable framework for the targeted generation of safety-critical driving scenarios. STRELGen synergistically combines a multi-agent trajectory-generation diffusion model (DM) with Spatio-Temporal Logic (STREL) specifications that encode complex safety and realism properties through a highly interpretable formalism. Crucially, monitoring satisfaction levels of these specifications is differentiable, enabling gradient-based search. At inference time, we optimize directly over the DM latent space to maximize STREL formula satisfaction. The result is efficient generation of highly plausible yet safety-critical multi-agent scenarios that lie within the learned data distribution. STRELGen thus provides a flexible, interpretable, and powerful tool for stress-testing autonomous driving systems, moving beyond the limitations of brute-force data collection.
>
---
#### [new 033] Implicit Action Chunking for Smooth Continuous Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于强化学习任务，旨在解决控制信号震荡导致的安全与稳定性问题。提出DWS框架，通过隐式动作分块实现平滑连续控制，提升安全性与性能。**

- **链接: [https://arxiv.org/pdf/2605.19592](https://arxiv.org/pdf/2605.19592)**

> **作者:** Bosun Liang; Shuo Pei; Zirui Chen; Chuanzhi Fan; Chen Sun; Yuankai Wu; Huachun Tan; Yong Wang
>
> **摘要:** Reinforcement learning often produces high-frequency oscillatory control signals that undermine the safety and stability required for physical deployment. Explicit action chunking addresses this by predicting fixed-horizon trajectories but scales the policy output dimension proportionally with the horizon length, leading to optimization difficulties and incompatibility with standard step-wise interaction. To overcome these challenges, this paper proposes Dual-Window Smoothing (DWS), an implicit action chunking framework for smooth continuous control. Unlike explicit methods, DWS enforces temporal coherence without expanding the action space. It uses a dual-window design: an execution window that ensures physical smoothness through deterministic modulation, and a value window that aligns temporal-difference targets over the horizon to correct critic bias caused by open-loop execution. DWS also includes a lightweight actor-side temporal regularizer based on first-order action differences to promote global continuity. This design effectively bridges the gap between temporal abstraction and reactive step-wise control. Experiments on benchmarks including the DeepMind Control Suite and industrial energy management tasks show that DWS outperforms state-of-the-art (SOTA) baselines. In complex vision-based autonomous driving tasks, DWS achieves smoother control, safer behavior with reduced jitter, and attains a 100% success rate.
>
---
#### [new 034] Closed-Loop Hybrid Digital Twin Platform for Connected and Automated Vehicle Validation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶车辆验证任务，旨在解决仿真与实车脱节的问题。提出一种闭环混合数字孪生平台，实现仿真与实车的实时同步和控制。**

- **链接: [https://arxiv.org/pdf/2605.19490](https://arxiv.org/pdf/2605.19490)**

> **作者:** Kanglong Quan; Zhebing Xia; Linfeng Jiang; Hao Yu; Ziheng Qiao; Dapeng Dong; Dongyao Jia
>
> **摘要:** Comprehensive and efficient validation of connected and automated vehicles (CAVs) is critical prior to real-world deployment. While simulation-based testing offers scalability, existing approaches often lack seamless integration with real vehicles and field data, limiting their fidelity in capturing dynamic, real-world interactions. To bridge this gap, this paper proposes a novel real-time hybrid digital twin platform. Its core innovation lies in the tight coupling of a high-fidelity CARLA-SUMO co-simulation with a physical test site and vehicle via a low-latency Vehicle-to-Everything (V2X) communication link. A custom-developed middleware serves as the critical bridge, synchronizing a real CAV's kinematic state as a shadow vehicle in the simulation and translating virtual control commands into chassis-actuating Controller Area Network (CAN) messages for closed-loop control. Detailed implementation includes using photogrammetry for full-scale asset reconstruction and a cloud-edge collaborative architecture for scalable, multi-user operation. Experimental results demonstrate stable synchronization and effective closed-loop control with low latency, confirming the platform's practicality for multi-scenario CAV verification.
>
---
#### [new 035] Beyond Imitation: Learning Safe End-to-End Autonomous Driving from Hard Negatives
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决模仿学习中安全性和轨迹偏差不匹配的问题。通过引入失败样本和新的损失函数，提升驾驶安全性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.19771](https://arxiv.org/pdf/2605.19771)**

> **作者:** Junli Wang; Zhihua Hua; Xueyi Liu; Zebin Xing; Haochen Tian; Kun Ma; Hangjun Ye; Guang Chen; Long Chen; Qichao Zhang
>
> **摘要:** Existing imitation learning methods for end-to-end autonomous driving predominantly learn from successful demonstrations by minimizing geometric deviations from expert trajectories. This paradigm implicitly assumes that spatial proximity implies behavioral safety, leading to a critical objective mismatch: trajectories with nearly identical imitation losses may exhibit drastically different safety outcomes, where one remains recoverable while the other results in collision. To address this limitation, we propose BeyondDrive, a failure-aware imitation learning framework that jointly learns from successful and failed driving behaviors. First, we introduce a flow matching-based negative trajectory generator that synthesizes safety-critical yet expert-proximate trajectories, enabling explicit modeling of safety asymmetry. Second, we develop a diversity-aware sampling strategy that mitigates mode collapse and improves coverage of diverse failure modes during negative trajectory generation. Third, we propose a Repulsive Distance Loss that simultaneously attracts predictions toward expert demonstrations while repelling them from hard negative trajectories, thereby establishing discriminative safety boundaries in trajectory space. Applied to the uni-modal baseline Latent TransFuser, BeyondDrive achieves 89.7 PDMS on the NAVSIMv1 closed-loop benchmark, outperforming prior state-of-the-art methods. Moreover, BeyondDrive generalizes effectively across different autonomous driving architectures, including multi-modal planners, and further demonstrates strong zero-shot transferability on the HUGSIM benchmark.
>
---
#### [new 036] Beyond Waypoints: Dual-Heatmap Grounding for Cross-Embodiment Semantic Navigation
- **分类: cs.RO**

- **简介: 该论文属于语义导航任务，解决机器人将抽象指令转化为可执行目标的问题。提出双热图框架，提升导航可靠性与跨平台适应性。**

- **链接: [https://arxiv.org/pdf/2605.19420](https://arxiv.org/pdf/2605.19420)**

> **作者:** Kaijie Yun; Yue Chen
>
> **摘要:** Grounding open-ended semantic instructions into physically executable local goals is a fundamental challenge in human-robot interaction. While existing navigation frameworks often regress deterministic waypoints, this rigid formulation collapses spatial uncertainty and frequently targets non-traversable object centers, leading to severe execution failures. In this work, we focus on the practical setting of in-FOV semantic navigation, where a robot receives concise, interleaved multimodal (text and image) prompts. To bridge the gap between abstract semantic intent and physical reachability, we propose a unified Vision-Language framework that abandons single-point regression in favor of a Dual-Heatmap representation. Our framework predicts a navigation affordance heatmap that captures continuous reachable regions, coupled with a facing heatmap for orientation constraints. These dense outputs inherently function as a differentiable semantic potential field, integrating seamlessly with downstream local planners. To support this paradigm, we build a fully automated, foundation-model-assisted synthetic data pipeline and establish a comprehensive simulation benchmark. Extensive experiments demonstrate that our framework achieves state-of-the-art performance among comparable 8B baselines. Crucially, a feature-fusion study and simulation studies across diverse robot embodiments (Jetbot, H1, Aliengo) reveal that explicit heatmap prediction drastically improves the Affordance Rate (AR). By placing targets reliably in executable free space, our framework effectively mitigates the brittleness of point regression, offering a transferable path toward safe cross-embodiment semantic navigation.
>
---
#### [new 037] SafeAlign-VLA: A Negative-Enhanced Safe Alignment Framework for Risk-Aware Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决安全关键场景下的风险感知不足问题。通过引入负样本增强安全对齐框架，提升系统对危险行为的理解与应对能力。**

- **链接: [https://arxiv.org/pdf/2605.19524](https://arxiv.org/pdf/2605.19524)**

> **作者:** Kefei Tian; Yuansheng Lian; Kai Yang; Xiangdong Chen; Shen Li
>
> **摘要:** End-to-end autonomous driving systems excel in common scenarios but struggle with safety-critical long-tail cases. Vision-Language-Action (VLA) models are promising due to their strong reasoning capabilities. However, most VLA-based approaches rely on positive expert demonstrations, rarely exploiting negative samples, leading to insufficient understanding of risky behaviors and safety boundaries. To address this limitation, we propose SafeAlign-VLA, a unified negative-enhanced safe alignment framework that incorporates negative data into supervised learning and reinforcement learning. First, we develop a counterfactual safety pairing paradigm to generate structured safety labels and counterfactual positive trajectories from risky scenarios via counterfactual reasoning. Then, a two-stage training strategy is adopted: negative-enhanced supervised fine-tuning for failure feedback and trajectory correction, followed by anchor-based group relative policy optimization that uses positive and negative trajectories as contrastive anchors to steer sampling and penalize high-risk behaviors via group-relative advantages. Experiments on NAVSIM and DeepAccident validate the proposed framework. SafeAlign-VLA achieves 89.1 PDMS on the NAVSIM v1 testset, improving over the baseline without negative data by 1.3%. On DeepAccident, it reduces the collision rate to 3.36%, while achieving 84.2% language accuracy and 85.8% risk prediction accuracy. These results demonstrate the effectiveness of the proposed negative-enhanced safe alignment framework for safe and robust autonomous driving.
>
---
#### [new 038] Topology-Optimized Pneumatic Soft Actuator: Design and Experimental Validation
- **分类: cs.RO**

- **简介: 该论文属于软体致动器设计任务，旨在通过拓扑优化提升其弯曲性能。工作包括扩展优化框架至3D，生成设计方案，并进行实验验证。**

- **链接: [https://arxiv.org/pdf/2605.20101](https://arxiv.org/pdf/2605.20101)**

> **作者:** Sumit Mehta; Konstantinos Poulios
>
> **备注:** 20 pages, 13 figures
>
> **摘要:** This paper demonstrates the computational design of soft elastomeric pneumatic actuators using nonlinear topology optimization. An existing density- and porohyperelasticity-based topology optimization framework was extended from 2D to 3D and used to generate two manufacturable actuator designs, which were then studied numerically and experimentally. For both designs, the objective was to maximize the bending response for a prescribed actuation pressure under two different allowable strain limits. A key advantage of the employed topology optimization framework is that it can consistently, during the optimization, account for the very large deformations induced upon pressurization. The two optimized 3D designs were fabricated using stereolithography and experimentally tested to validate their performance.
>
---
#### [new 039] Hamilton--Jacobi Reachability for Spacecraft Collision Avoidance
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于航天器避撞任务，解决两星在同轨道碰撞问题。通过HJ可达性方法建模相对运动，计算不可避撞集，结合控制逻辑实现安全规避。**

- **链接: [https://arxiv.org/pdf/2605.20138](https://arxiv.org/pdf/2605.20138)**

> **作者:** Larry Hui; Jordan Kam; William Su; Jianshu Zhou
>
> **备注:** Accepted to the 20th IEEE International Conference on Control & Automation (IEEE ICCA 2026). 6 pages, 4 figures
>
> **摘要:** This article presents a Hamilton--Jacobi (HJ) reachability framework for a two--satellite collision avoidance problem operating in the same circular orbit, where relative motion is modeled in the radial--tangential--normal (RTN) frame using planar Hill--Clohessy--Wiltshire (HCW) dynamics. We define the target state space as unsafe relative configurations in the orbit plane corresponding to minimum separation requirements consistent with Federal Communications Commission (FCC) orbital standards. The interaction between spacecraft is formulated as a zero--sum differential game, where Player 1 is the controlled satellite and Player 2 is modeled as a bounded adversarial disturbance with unknown intent. We present the HJ formulation and compute backward reachable sets that characterize relative states from which collision cannot be avoided under worst-case disturbances, while states outside this set admit provably collision-free trajectories. These reachable sets are integrated with supervisory hybrid control logic to determine when evasive maneuvers must be initiated, enabling mathematically grounded safety guarantees for scalability.
>
---
#### [new 040] CosFly: Plan in the Matrix, Fly in the World
- **分类: cs.RO**

- **简介: 该论文提出CosFly系统，用于空中目标跟踪，解决无人机在复杂环境中的导航与感知问题，包含数据集和模拟流程。**

- **链接: [https://arxiv.org/pdf/2605.19120](https://arxiv.org/pdf/2605.19120)**

> **作者:** Hanxuan Chen; Xiangyue Wang; Songsheng Cheng; Ruilong Ren; Jie Zheng; Shuai Yuan; Tianle Zeng; Hanzhong Guo; Binbo Li; Kangli Wang; Ji Pei
>
> **摘要:** We present CosFly, a box-structured planning and multimodal simulation pipeline for aerial tracking, together with CosFly-Track, a large-scale UAV dataset for dynamic target tracking across diverse environments including urban centers, highways, rural landscapes, forests, and coastal towns. In our current implementation on CARLA, CosFly provides a modular 7-step construction pipeline that converts complex 3D worlds into structured obstacle representations for planning, then projects the resulting trajectories back into multi-modal sensor data -- including RGB images, high-precision depth maps, and semantic segmentation masks -- paired with natural language navigation instructions. A key feature is the support for configurable fixed-FOV zoom levels (one FOV setting drawn per trajectory and held constant throughout), enabling simulation of various focal lengths through camera-intrinsic adjustments. The pipeline covers the complete workflow from 3D map export through grid simplification, pedestrian and drone trajectory planning, multi-modal rendering with 6-DOF pose annotations, quality inspection, and teacher-student caption generation. We analyze two trajectory-planning paradigms for aerial target tracking: a conventional two-stage pipeline with front-end candidate generation and backend refinement, and a direct gradient-based formulation that optimizes multiple tracking constraints in a single objective. The public CosFly-Track release contains 250 validated trajectories and approximately 100,000 rendered images with complete 6-DOF drone pose annotations (position x, y, z and orientation yaw, pitch, roll). Together, the pipeline and dataset establish a scalable foundation for aerial-ground collaborative research, supporting dynamic target tracking, UAV navigation, and multi-modal perception across diverse environments.
>
---
#### [new 041] Aerial Inspection Behaviors via RL-based Quadrotor Control for Under-canopy Forest Environments
- **分类: cs.RO; cs.AI; math.OC**

- **简介: 该论文属于无人机自主导航任务，解决森林下层环境下的巡检问题。通过RL控制与路径规划结合，实现安全有效的巡检路径和姿态跟踪。**

- **链接: [https://arxiv.org/pdf/2605.19202](https://arxiv.org/pdf/2605.19202)**

> **作者:** Fausto Mauricio Lagos Suarez; Akshit Saradagi; Vidya Sumathy; Viswa Narayanan Sankaranarayanan; George Nikolakopoulos
>
> **备注:** Submitted to 2026 IEEE 22nd International Conference on Automation Science and Engineering
>
> **摘要:** This paper addresses the problem of using a deep Reinforcement Learning (RL)-based low-level Quadrotor controller within an autonomous Quadrotor navigation stack for aerial inspection missions in under-canopy forest environments. Specifically, the article presents an end-to-end (mapping states to RPMs) Quadrotor control policy that achieves inspection view-pose tracking (simultaneous position and yaw reference tracking), which is crucial for various target inspection behaviors and point-to-point navigation in forests. To ensure safe and reliable deployment of the end-to-end RL controller in long-range missions, this article utilizes a higher navigation guidance layer comprising of a Traveling Salesman Problem planner (TSP) and a Rapidly-exploring Random Tree Star (RRT*) planner. Over a known map of a forest and a set of user-specified inspection regions, the TSP planner finds the optimal visitation sequence. Between two target regions, collision-free paths that respect the tracking limitations of the lower end-to-end RL policy are generated by an RRT* planner. Through five target inspection scenarios, this article demonstrates that an RL-based motor-level stabilizing controller, supported by a navigation guidance layer, can be used effectively as the low-level inspection execution module for under-canopy forest inspection missions.
>
---
#### [new 042] Graph Neural Planning and Predictive Control for Multi-Robot Communication-Constrained Unlabeled Motion Planning
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人运动规划任务，解决通信受限下的协同路径规划问题。提出GATP与NMPC结合的框架，实现安全轨迹生成与高效协作。**

- **链接: [https://arxiv.org/pdf/2605.19209](https://arxiv.org/pdf/2605.19209)**

> **作者:** Manohari Goarin; Yang Zhou; Giuseppe Loianno
>
> **备注:** 8 pages, 6 figures, Accepted at the IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** The multi-robot unlabeled motion planning problem of concurrently assigning robots to goals and generating safe trajectories is central in many collaborative tasks. Recent Graph Neural Network methods offer scalable decentralized solutions but rely on simplified dynamics and simulation environments, overlooking key challenges of real-world deployment such as dynamic feasibility and communication constraints. To address these gaps, we propose a hierarchical framework that combines a Graph ATtention Planner (GATP) with a decentralized Nonlinear Model Predictive Controller (NMPC). GATP provides intermediate subgoals through multi-robot cooperation, and the NMPC enforces safety under nonlinear dynamics and actuation constraints. We evaluate our framework in both simulation and real-world quadrotor experiments. Thanks to attention mechanisms and minimal communication requirements, we demonstrate improved generalization to larger teams, robustness to communication delays up to 200 ms and practical feasibility with decentralized on-board inference.
>
---
#### [new 043] Bilateral Teleoperation with Compliant 6-DOF Pose-and-Force Sensing
- **分类: cs.RO**

- **简介: 该论文属于遥操作任务，解决传统系统成本高、耦合强的问题，提出基于低成本柔性6-DOF传感器的框架，实现稳定远程操作与力控。**

- **链接: [https://arxiv.org/pdf/2605.19255](https://arxiv.org/pdf/2605.19255)**

> **作者:** Yue Feng; Weicheng Huang; I-Ming Chen
>
> **备注:** 8 pages, 16 figures, 2 tables. Preprint
>
> **摘要:** Existing bilateral teleoperation platforms still rely on costly rigid six-axis force/torque sensors, tightly coupled leader-follower hardware, and kilohertz control loops. We present a Cartesian bilateral framework built on the hardware-agnostic WinGs Operating Studio (WOS) middleware, in which a low-cost compliant 6-DOF pose-and-force sensing end-effector, Delta6, is mounted on both sides so that each manipulator behaves as an end-effector 6-DOF series elastic actuator (SEA). The leader runs a damping-only admittance loop with a 6-D biquad notch filter; the follower realizes a stiffness-damping impedance through a position-based outer loop with a PID wrench-to-pose mapping. Three time scales (hardware I/O, mid-rate impedance/admittance, low-rate teleoperation messages) are explicitly decoupled, enabling the same application to drive heterogeneous arms. On a Lite6/FR3 testbed at 150 Hz, the system tracks stably under delays up to $120\pm40$ ms and 1% packet loss, matches the prescribed virtual stiffness in contact, and shows a favorable cumulative energy signature in passivity-style tests.
>
---
#### [new 044] Beyond Binary Success: A Diagnostic Meta-Evaluation Framework for Fine-Grained Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人操作任务，旨在解决现有基准评估过于简化的问题。提出MetaFine框架，通过分解理解、感知和行为三方面诊断模型缺陷，提升细粒度操作能力评估的准确性。**

- **链接: [https://arxiv.org/pdf/2605.19986](https://arxiv.org/pdf/2605.19986)**

> **作者:** He-Yang Xu; Pengyuan Zhang; Zongyuan Ge; Xiaoshuai Hao; Serge Belongie; Xin Geng; Yuxin Peng; Xiu-Shen Wei
>
> **备注:** Project page: this https URL
>
> **摘要:** Fine-grained manipulation marks a regime where global scene context no longer suffices, and success hinges on the tight coupling of local attribute grounding, high-fidelity spatial perception, and constraint-respecting motor execution. However, current embodied AI benchmarks collapse these capacities into binary success rates, systematically inflating reported capabilities by up to 70% and masking the architectural bottlenecks that impede real-world deployment. We introduce MetaFine, a diagnostic meta-evaluation framework that disentangles manipulation competency along three axes: understanding, perception, and controlled behavior. Built on a compositional task graph, MetaFine absorbs heterogeneous external benchmarks and reconstructs them into diagnostic scenarios of varying complexity under a unified protocol. Evaluating state-of-the-art vision-language-action (VLA) models through this lens exposes severe dimension-specific failures invisible to conventional metrics. Through targeted causal intervention, we identify the visual encoder's ability to preserve local spatial structure as a key bottleneck for fine-grained precision: improving it directly unlocks previously inaccessible manipulation capabilities without modifying downstream policies. MetaFine further supports hybrid real-sim validation, using limited paired real-world rollouts to calibrate scalable simulation-based estimates for more stable physical benchmarking. By shifting evaluation from ranking to diagnosis, MetaFine turns benchmarking into an actionable compass for repairing the layered capacities underlying genuine physical dexterity. The MetaFine framework, benchmarks, and supporting resources will be publicly released at our project page: this https URL.
>
---
#### [new 045] Neural Operators for Design-Space Surrogate Modeling of Tendon-Actuated Continuum Robots
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人建模任务，旨在解决连续体机器人设计空间中高效准确的代理模型问题。通过神经算子方法，实现跨设计的快速泛化建模。**

- **链接: [https://arxiv.org/pdf/2605.19104](https://arxiv.org/pdf/2605.19104)**

> **作者:** Branden Frieden; James M. Ferguson; Alan Kuntz; Varun Shankar
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Continuum robots enable dexterous manipulation in constrained environments, but require accurate and efficient models for real-time manipulation and control. Traditional physics-based models can be computationally expensive and may suffer from inaccuracies due to unmodeled effects, while current learning-based methods often generalize poorly beyond the specific robot on which they are trained. We present a formulation of surrogate modeling for tendon-driven continuum robots as an operator learning problem that maps robot design parameters and tendon actuation inputs to resulting configurations. This formulation enables a single trained model to generalize across a large class of robot designs. We develop four novel neural operator architectures--two based on Deep Operator Networks (DeepONets) and two based on Fourier Neural Operators (FNOs)--and train them on simulation data to predict robot configurations. All architectures achieve good accuracy while allowing for fast and accurate generalization across designs. Our results demonstrate that operator learning provides an effective and generalizable surrogate for continuum robot mechanics in the design space, enabling fast modeling for control, planning, and design optimization in surgical and industrial applications.
>
---
#### [new 046] ContextFlow: Hierarchical Task-State Alignment for Long-Horizon Embodied Agents
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于长期任务的具身智能体研究，解决任务状态不一致问题。通过ContextFlow框架，实现任务阶段的显式对齐与可审计管理。**

- **链接: [https://arxiv.org/pdf/2605.19314](https://arxiv.org/pdf/2605.19314)**

> **作者:** Shuhan Guo; Kun Zhang; Haifei Liu; Xingyu Gao; Yongqi Zhang; Yaqing Wang; Quanming Yao
>
> **摘要:** Long-horizon embodied agents increasingly delegate navigation, search, approach, and manipulation to specialist executors. As these executors become stronger, the main bottleneck shifts from local skill execution to maintaining a coherent task frontier across planning, monitoring, memory, and execution. We study task-state misalignment, a task-level consistency failure in which the planner's active stage, runtime evidence, remembered context, and delegated executor no longer justify the same next-step decision. This failure can lead to unsupported handoffs, stage lock, executor-context mismatch, and unnecessary replanning. We propose ContextFlow, an inspectable alignment framework that represents stages as explicit contracts, converts runtime observations into evidence packets, and applies scoped updates including continue, refine, transfer, promote, and repair. ContextFlow keeps specialist executors responsible for local closed-loop control while making task-frontier alignment explicit and auditable. Experiments and demonstration traces on long-horizon embodied tasks illustrate how evidence-grounded scoped updates diagnose and mitigate recurring task-state failures.
>
---
#### [new 047] World-Ego Modeling for Long-Horizon Evolution in Hybrid Embodied Tasks
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于具身智能任务，旨在解决长时序混合任务中世界与自我状态纠缠的问题。提出World-Ego Modeling，分离世界与自我演化，构建WEM模型并建立基准HTEWorld进行评估。**

- **链接: [https://arxiv.org/pdf/2605.19957](https://arxiv.org/pdf/2605.19957)**

> **作者:** Zuyao Lin; Jianhui Zhang; Peidong Jia; Xiaoguang Zhao; Shanghang Zhang; Xingyu Chen
>
> **摘要:** World models are widely explored in embodied intelligence, yet they typically predict distinct evolutions of the world and the ego within a single stream, where the world captures persistent instruction-agnostic scene regularities and the ego captures robot-centric instruction-conditioned dynamics. This world-ego entanglement leads to a degradation in long-horizon embodied scenarios, particularly in hybrid tasks with interleaved navigation and manipulation behaviors. In this paper, we introduce \emph{World-Ego Modeling}, a new conceptual paradigm that decomposes future evolution into world and ego components. We define the world-ego boundary from three perspectives, i.e., motion-, semantic-, and intention-based views, and analyze three disentanglement strategies with post-, pre-, and full disentanglement. Further, we instantiate this paradigm as the World-Ego Model (WEM), a unified embodied world model that couples an implicit separate world-ego planner with a cascade-parallel mixture-of-experts (CP-MoE) diffusion generator. To enable rigorous evaluation, we further construct HTEWorld, the first benchmark for long-horizon world modeling with hybrid navigation-manipulation tasks, providing 125K video clips (over 4.5M frames) with fine-grained action annotations and 300 multi-turn evaluation trajectories (over 2K instructions). Extensive experiments show that WEM achieves state-of-the-art performance on HTEWorld while remaining competitive on existing manipulation-only benchmarks.
>
---
#### [new 048] Probabilistic Recursively Feasible Motion Planning Under Uncertain Environments
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于运动规划任务，解决不确定环境中递归可行性问题。提出PRF-MPC框架，通过概率约束确保规划的递归可行性。**

- **链接: [https://arxiv.org/pdf/2605.19015](https://arxiv.org/pdf/2605.19015)**

> **作者:** Hyeontae Sung; Hyeongchan Ham; Junyoung Park; Kai Ren; Heejin Ahn
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Safe motion planning in uncertain, time-varying environments is challenging because the safe region can change unpredictably across planning steps, often causing a loss of recursive feasibility. In this work, we present a Probabilistic Recursively Feasible Model Predictive Control (PRF-MPC) framework that guarantees recursive feasibility with a specified probability. We introduce properties that an ideal predictor should satisfy to ensure distributional consistency, and use these properties to derive closed-form expressions for the means and covariances of trajectories predicted at future time steps. Building on this analysis, we construct safety constraints that ensure, with high probability, that the current safe set is contained within the safe sets at future time steps, thereby probabilistically guaranteeing recursive feasibility. Simulation results on a lane-change scenario demonstrate that the proposed method significantly improves recursive feasibility.
>
---
#### [new 049] EUPHORIA: Efficient Universal Planning via Hybrid Optimization for Robust Industrial Robotic Assembly
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于工业机器人装配任务，解决传统规划方法适应性差、效率低的问题。提出EUPHORIA框架，融合元学习、物理感知注意力和残差优化，实现高效通用装配规划。**

- **链接: [https://arxiv.org/pdf/2605.18872](https://arxiv.org/pdf/2605.18872)**

> **作者:** Shih-Yu Lai; Chia-Ching Yen; Yang-Ting Shen; Peter Yichen Chen; Yu-Lun Liu; Bing-Yu Chen
>
> **摘要:** Robotic assembly in architectural construction faces a persistent bottleneck: existing planners are either highly specialized, requiring prohibitive retraining for every new geometric design, or operationally inefficient, treating structural sequencing and kinematic motion as disjoint processes. We present EUPHORIA, a unified framework that achieves universal few-shot adaptability and dynamic efficiency through a hybrid optimization strategy. To overcome the retraining bottleneck, we propose a Meta-Geometric Encoder based on Graph Hypernetworks: unlike standard contrastive learning, which performs only feature-level recognition, our hypernetwork dynamically generates policy parameters from a minimal support set, enabling parameter-level adaptation to complex topologies (e.g., domes, arches) without gradient-based retraining. For structural reasoning, we introduce a Physics-Informed Graph Transformer trained via Soft Actor-Critic (SAC), with a Physics-Bias Attention mechanism that modulates attention scores using contact forces from Discrete Element Model (DEM) simulations, guiding the planner toward structurally critical connections. We further ensure operational efficiency through Kinematics-Aware Sequencing, where the SAC objective penalizes high-energy transitions. Finally, we bridge the Sim2Real gap via Residual Stability Correction, a differentiable optimization layer that fine-tunes coarse assembly actions by minimizing a joint energy-stability cost prior to execution. Experiments show that EUPHORIA significantly reduces energy consumption over decoupled baselines and achieves state-of-the-art success rates on unseen, non-standard geometries with minimal few-shot examples, fusing meta-learning, physics-informed attention, and residual optimization into a cohesive, generalized planner.
>
---
#### [new 050] From Prompts to Pavement Through Time: Temporal Grounding in Agentic Scene-to-Plan Reasoning
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于自主车辆场景到计划推理任务，旨在解决时间感知不足导致的推理不一致问题。通过引入具有时间整合的规划器架构，评估其在BDD-X数据集上的表现。**

- **链接: [https://arxiv.org/pdf/2605.19824](https://arxiv.org/pdf/2605.19824)**

> **作者:** Ahmed Y. Gado; Omar Y. Goba; Alaa Hassanein; Catherine M. Elias; Ahmed Hussein
>
> **摘要:** Recent attempts to support high-level scene interpretation and planning in Autonomous Vehicles (AVs) using ensembles of Large Language Models (LLMs) and Large Multimodal Models (LMMs) continue to treat time as a secondary property. This lack of temporal grounding leads to inconsistencies in reasoning about continuous actions, undermining both safety and interpretability. This work explores whether temporal conditioning within inter-agent communication can preserve or enhance coherence without introducing degradation in semantic or logical consistency. To investigate this, we introduce three planner architectures with progressively increasing temporal integration and evaluate them on curated subsets of the BDD-X dataset using semantic, syntactic, and logical metrics. Results show that while temporal conditioning reshapes reasoning style, it yields no statistically significant improvements in standard NLP-based correctness metrics. However, qualitative analysis reveals predictive hazard reasoning, stable corrective behavior, and strategic divergence in the Sentinel. These findings clarify the limits of prompt-based temporal grounding and establish the first empirical benchmark for temporal scene-to-plan reasoning.
>
---
#### [new 051] CADENet: Condition-Adaptive Asynchronous Dual-Stream Enhancement Network for Adverse Weather Perception in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文属于自动驾驶中的恶劣天气目标检测任务，解决增强与检测无法实时协同的问题。提出CADENet，实现无延迟检测与条件自适应增强，提升恶劣天气下感知效果。**

- **链接: [https://arxiv.org/pdf/2605.19837](https://arxiv.org/pdf/2605.19837)**

> **作者:** Sherif Khairy; Catherine M. Elias
>
> **摘要:** Adverse weather (rain, fog, sand, and snow) degrades camera-based object detection in autonomous vehicles. Existing enhancement-then-detect approaches stall the safety-critical perception loop, violating hard real-time requirements. Progress on this problem is also constrained by an under-recognized evaluation ceiling: ground truth annotated on degraded images cannot credit a detector that recovers objects the annotators themselves could not see, so a genuinely useful enhancement can register as a near-flat F1 gain. This paper presents CADENet (Condition-Adaptive Asynchronous Dual-stream Enhancement Network), a training-free three-thread system: Thread S (YOLOv11n) delivers detections at full frame rate with zero added latency; Thread Q applies condition-adaptive enhancement (CAPE) and fuses results via entropy-guided NMS (EG-NMS) without blocking Thread S; Thread E provides CLIP zero-shot weather classification, so new weather categories require only a new text prompt, with no labeled data and no retraining. Evaluated on 1327 DAWN images (YOLOv11m, IoU = 0.5, confidence = 0.25), CADENet achieves Recall = 0.0103 (micro), F1 = 0.0230 on snow, and F1 = 0.0038 on rain. We formalize the annotation completeness bias on DAWN-class data, so the reported F1 values are lower bounds on the true gain; recall is the annotation-gap-immune headline metric. Thread S sustains approximately 44 FPS regardless of enhancement load. No model retraining or additional sensor hardware is required.
>
---
#### [new 052] Sampling-Based Safe Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出SBSRL，解决强化学习中的安全探索问题，通过约束动态样本保证安全性，实现高效安全的策略学习。**

- **链接: [https://arxiv.org/pdf/2605.19469](https://arxiv.org/pdf/2605.19469)**

> **作者:** Luca Vignola; Bruce D. Lee; Manish Prajapat; Manuel Wendl; Melanie Zeilinger; Andreas Krause; Yarden As
>
> **摘要:** Safe exploration remains a fundamental challenge in reinforcement learning (RL), limiting the deployment of RL agents in the real world. We propose Sampling-Based Safe Reinforcement Learning (SBSRL), a model-based RL algorithm that maintains safety throughout the learning process by enforcing constraints jointly across a finite set of dynamics samples. This formulation approximates an intractable worst-case optimization over uncertain dynamics and enables practical safety guarantees in continuous domains. We further introduce an exploration strategy based on constraining epistemic uncertainty, eliminating the need for explicit exploration bonuses. Under regularity conditions, we derive high-probability guarantees of safety throughout learning and a finite-time sample complexity bound for recovering a near-optimal policy. Empirically, SBSRL achieves safe and efficient exploration both in simulation and in real robotic hardware, and readily extends to practical deep-ensemble implementations that scale to high-dimensional continuous control problems.
>
---
#### [new 053] Domain-Adaptive Communication-Rate Optimization for Sim-to-Real Humanoid-Robot Wireless XR Teleoperation
- **分类: cs.IT; cs.LG; cs.RO**

- **简介: 该论文属于人形机器人无线XR遥操作任务，解决高频率运动传输带来的通信能耗问题。通过优化采样率和引入PPO方法，在模拟到现实的分布偏移中提升重建精度与通信效率。**

- **链接: [https://arxiv.org/pdf/2605.19293](https://arxiv.org/pdf/2605.19293)**

> **作者:** Caolu Xu; Zhiyong Chen; Meixia Tao; Li Song; Feng Yang; Wenjun Zhang
>
> **备注:** submitted to IEEE journal
>
> **摘要:** Wireless extended reality (XR) teleoperation provides embodied interaction capability for collecting humanoid robot demonstrations, but the large-scale adoption is restricted by the overhead of high-frequency motion transmission. This paper develops a system framework that integrates sampling, transmission, interpolation, and reconstruction and formulates a communication-rate optimization that aims to minimize the communication energy while maintaining the reconstruction accuracy of robot motion trajectories through dimension-wise sampling-rate control. Since acquiring real-time feedback from physical robots is limited by hardware costs, it is necessary to solve the problem through simulator interaction with offline real-domain data correction. To guide sim-to-real adaptation, we provide a PAC-Bayes generalization characterization that reveals the effects of latent density-ratio estimation, finite-sample deviation, and encoder bias. Building on this analysis, we propose a proximal policy optimization (PPO) method with density-ratio weighting and trust-region regularization. Experiments on public humanoid teleoperation dataset show that the proposed method improves the tradeoff between reconstruction error and communication energy consumption under sim-to-real distribution shift. We further analyze the effectiveness of the proposed algorithm across various wireless channels and dynamic motion trajectories.
>
---
#### [new 054] Robotics-Inspired Guardrails for Foundation Models in Socially Sensitive Domains
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于AI安全任务，旨在解决基础模型在敏感领域中的行为控制问题。通过引入机器人学概念，构建可运行干预的框架，确保交互过程安全。**

- **链接: [https://arxiv.org/pdf/2605.19940](https://arxiv.org/pdf/2605.19940)**

> **作者:** Rebecca Ramnauth; Drazen Brscic; Brian Scassellati
>
> **备注:** Under review at Journal of Artificial Intelligence Research (JAIR)
>
> **摘要:** Foundation models are increasingly deployed in socially sensitive domains such as education, mental health, and caregiving, where failures are often cumulative and context-dependent. Existing guardrail approaches -- ranging from training-time alignment to prompting, decoding constraints, and post-hoc moderation -- primarily provide empirical risk reduction rather than enforceable behavioral guarantees, and largely treat safety as a property of individual outputs rather than interaction trajectories. We reframe guardrails as a problem of runtime behavioral control over interaction trajectories, drawing on robotics to introduce formal constructs for constraint enforcement in uncertain, closed-loop systems. We instantiate these ideas in the Grounded Observer framework and apply it across three real-world deployments: small talk, in-home autism therapy, and behavioral de-escalation in schools. Across settings, the framework enables runtime interventions that mitigate drift into undesirable interaction regimes while adapting to diverse social contexts. We discuss extensions to the framework and propose research directions toward stronger guarantees.
>
---
#### [new 055] Probing Embodied LLMs: When Higher Observation Fidelity Hurts Problem Solving
- **分类: cs.AI; cs.RO**

- **简介: 该论文研究了具身大语言模型在问题解决中的表现，探讨不同感知精度对任务的影响。任务属于机器人认知与语言模型交互领域，旨在解决感知信息与决策之间的关系问题。实验发现高精度感知反而降低性能，而适度噪声提升效果。**

- **链接: [https://arxiv.org/pdf/2605.20072](https://arxiv.org/pdf/2605.20072)**

> **作者:** Oussama Zenkri; Oliver Brock
>
> **备注:** Submitted to From Animals to Animats: The 18th International Conference on the Simulation of Adaptive Behavior (SAB)
>
> **摘要:** Large Language Models are increasingly proposed as cognitive components for robotic systems, yet their opaque decision processes make it difficult to explain success or failure in closed-loop embodied tasks. Following an empirical AI methodology, we study embodied LLM agents behaviorally by varying the information available to the agent and measuring the resulting changes in behavior. Using the Lockbox, a sequential mechanical puzzle with hidden interdependencies, we evaluate LLMs across RGB, RGB-D, and ground-truth symbolic observations in a physical robotic setup and use controlled simulation to probe the resulting behavior. Counterintuitively, agents perform best under raw RGB input and worst under perfect ground-truth observations. In simulation, we probe this effect by randomly flipping perceived action outcomes and find that moderate noise improves performance, peaking at a 40% flip probability with a 2.85-fold success rate increase over the noise-free baseline. Further analysis links this gain to a reduction in repetitive action loops. These findings suggest that success rates alone are insufficient for evaluating LLMs, as measured performance may reflect the interaction between perceptual errors and reasoning failures rather than robust problem solving.
>
---
#### [new 056] DAG-Based QoS-Aware Dynamic Task Placement for Networked Multi-Stage Control Pipelines
- **分类: cs.DC; cs.MA; cs.RO; eess.SY**

- **简介: 该论文属于网络化机器人控制任务，旨在解决多阶段控制流水线的动态任务部署问题。通过DAG建模与QoS优化，实现低延迟、高可靠的任务分配。**

- **链接: [https://arxiv.org/pdf/2605.19887](https://arxiv.org/pdf/2605.19887)**

> **作者:** Thien Tran; Jonathan Kua; Thuong Hoang; Minh Tran; Yuemin Ding; Jiong Jin
>
> **备注:** 4 pages, 1 figure, 1 algorithm, accepted as a Work-in-Progress (WiP) paper, on the 24th IEEE International Conference on Industrial Informatics (INDIN), 26-29 July, 2026, Melbourne, Australia
>
> **摘要:** Current Physical AI (PAI) relies heavily on closed-loop visual-servoing pipelines, whose perception and planning stages may become computationally intensive onboard due to complex models embedded on robots. In practice, offloading the perception task to on-site edges statically is inappropriate for latency-sensitive, precise industrial settings over a standardized industrial network. This emphasizes the importance of Control-Communication-Computing (3C) co-design in industrial automation: monolithic local execution saturates AI-accelerated machine and robot hardware, while static edge offloading exposes the control loop to network jitter. Existing adaptive task placement (ATP) controllers can partially address the gap by relocating a single pipeline stage on binary threshold rules, without a multi-stage model and an explicit cost on placement switching. In this Work-in-Progress (WiP) paper, we propose a directed acyclic graph (DAG) based quality-of-service (QoS)-aware dynamic task placement (DTP) framework for sensing-perception-planning-control pipelines in networked robotics. This pipeline is formalized as a DAG with task-level and node-level attributes for compute cost, communication delay, and feasible placement sets; over a small interpretable candidate set (fully local, static offload, hybrid), a window-based cost function combines tail end-to-end latency, deadline violation rate, hardware utilization, and a Hamming-distance switching penalty, and a DTP algorithm with hysteresis and a minimum dwell-time bounds placement chatter. Our WiP paper presents the theoretical framework, a structured qualitative analysis, and a two-phase simulation plus hardware-in-the-loop validation roadmap.
>
---
#### [new 057] RoboJailBench: Benchmarking Adversarial Attacks and Defenses in Embodied Robotic Agents
- **分类: cs.CR; cs.RO**

- **简介: 该论文属于安全评估任务，旨在解决 embodied AI 系统中对抗攻击与防御的评估问题。构建了 RoboJailBench 基准，涵盖安全分类、数据增强和标准化评估框架。**

- **链接: [https://arxiv.org/pdf/2605.19328](https://arxiv.org/pdf/2605.19328)**

> **作者:** Doguhuan Yeke; Yanming Zhou; Leo Y. Lin; Hongyu Cai; Antonio Bianchi; Z. Berkay Celik
>
> **摘要:** Recent advances in Vision-Language Models (VLMs) facilitate a new class of embodied AI systems, where these models are integrated into physical platforms, e.g. robots and autonomous vehicles, to interpret visual scenes and execute natural language commands in diverse environments. Previous research has introduced jailbreak attacks and defenses for embodied AI. Their evaluations, however, rely on ad-hoc datasets, limited metrics, and emphasize attack success while neglecting the trade-off between security and the ability to follow benign commands. Existing benchmarks and evaluation frameworks either target traditional chat-based models or focus on non-adversarial safety evaluation for embodied AI; neither captures the adversarial risks, inputs, consequences, and evaluation criteria necessary for jailbreak attacks in embodied AI systems. In this paper, we address this gap with RoboJailBench, which consists of three core components. We establish a security taxonomy derived from ISO standards, regulatory rules, and documented incidents. This effort yields 18 categories of security violation consequences for embodied AI. We introduce an intent contrast dataset pipeline that augments existing datasets with paired adversarial and benign goals to measure both security and utility. Lastly, we provide an evolving repository with standardized metrics and a unified process for assessing and integrating new attacks and defenses. With this benchmark, we construct a new taxonomy-balanced dataset and augment five existing datasets. We integrate four attacks and two defenses to evaluate their performance on leading embodied VLMs. This benchmark provides the first standardized evaluation framework for jailbreak attacks in embodied AI and supports future research. We release our code, datasets, and artifacts, and maintain a leaderboard at this https URL.
>
---
#### [new 058] Towards LLM-Assisted Architecture Recovery for Real-World ROS~2 Systems: An Agent-Based Multi-Level Approach to Hierarchical Structural Architecture Reconstruction
- **分类: cs.SE; cs.AI; cs.RO**

- **简介: 该论文属于软件架构恢复任务，旨在解决ROS 2系统中隐式结构难以恢复的问题。通过改进的提示技术和多级中间表示，实现更精确的层次结构重建。**

- **链接: [https://arxiv.org/pdf/2605.20055](https://arxiv.org/pdf/2605.20055)**

> **作者:** Dominique Briechle; Raj Chanchad; Tobias Geger; Ruidi He; Dhruv Jajadiya; Dhruv Kapadiya; Andreas Rausch; Meng Zhang
>
> **摘要:** Explicit software architecture models are essential artifacts for communicating, analyzing, and evolving complex software-intensive systems. In ROS~2-based robotic systems, however, structural (de-)composition and integration semantics are often only implicitly encoded across distributed artifacts such as source code and launch files, making recovery of hierarchical architecture particularly difficult. Existing approaches mainly focus on node-level entities and communication wiring, while providing limited support for recovering hierarchical structural (de-)composition across multiple abstraction levels. In this paper, we extend our previously proposed blueprint-guided LLM-assisted architecture recovery pipeline for ROS~2 systems through two major enhancements: (1) refined prompting to improve the consistency and controllability of architecture synthesis, and (2) a staged recovery strategy based on multi-level intermediate architectural representations that incorporate the atomic ROS node list and launch file dependencies, thereby enabling structurally constrained reconstruction across multiple abstraction levels. The approach is evaluated on a real-world automated product disassembly system based on cooperative robotic arms and heterogeneous ROS~2 artifacts. Compared to our previous work, the considered case study exhibits substantially higher integration complexity and richer functionality. The results demonstrate improved structural consistency, scalability, and robustness of architecture recovery, while also revealing remaining challenges related to dynamic integration semantics in large-scale ROS~2 systems.
>
---
#### [new 059] EgoTraj: Real-World Egocentric Human Trajectory Dataset for Multimodal Prediction
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出EgoTraj数据集，用于解决真实世界第一视角人类轨迹预测问题。通过多模态数据收集与分析，提升AR导航与辅助系统性能。**

- **链接: [https://arxiv.org/pdf/2605.19004](https://arxiv.org/pdf/2605.19004)**

> **作者:** Ahmad Yehia; Abduallah Mohamed; Tianyi Wang; Jiseop Byeon; Kun Qian; Junfeng Jiao; Christian Claudel
>
> **备注:** 21 pages, 14 figures. Project page: this https URL
>
> **摘要:** Accurately forecasting human trajectories from an egocentric perspective plays a central role in applications such as humanoid robotics, wearable sensing systems, and assistive navigation. However, progress in this direction remains limited due to the scarcity of egocentric trajectory datasets collected in real-world environments. Addressing this need, we introduce EgoTraj, an egocentric multimodal open dataset recorded using Meta Quest Pro (MQPro). EgoTraj contains 75 sequences of human navigation collected from multiple MQPro wearers in real-world urban environments. Each recording provides synchronized RGB video along with ground-truth data, including continuous time-synchronized 6-degree-of-freedom head poses, per-frame 3D eye gaze vectors, scene annotations. To the best of our knowledge, EgoTraj differs from typical egocentric trajectory datasets by capturing long-horizon, self-directed navigation across diverse urban routes with broad participant diversity. To demonstrate the potential of the dataset, we benchmark several state-of-the-art methods for egocentric trajectory prediction and conduct ablation studies to analyze the contributions of gaze, scene, and motion cues. The results highlight the utility of EgoTraj for AR-based perception, navigation, and assistive systems. The EgoTraj dataset, code, and EgoViz Dashboard are publicly available at this https URL.
>
---
## 更新

#### [replaced 001] Solving Reach- and Stabilize-Avoid Problems Using Discounted Reachability
- **分类: math.OC; cs.RO; eess.SY**

- **简介: 该论文研究非线性系统的无限时域可达-避免和稳定-避免问题，通过设计值函数方法解决状态集合的可达性和稳定性分析，验证于3D Dubins车模型。**

- **链接: [https://arxiv.org/pdf/2505.09067](https://arxiv.org/pdf/2505.09067)**

> **作者:** Boyang Li; Zheng Gong; Sylvia Herbert
>
> **备注:** 16 pages, 6 figures, 1 table. Accepted to IEEE Transactions on Automatic Control
>
> **摘要:** In this article, we consider the infinite-horizon reach-avoid (RA) and stabilize-avoid (SA) zero-sum game problems for general nonlinear continuous-time systems, where the goal is to find the set of states that can be controlled to reach or stabilize to a target set, without violating constraints even under the worst-case disturbance. Based on the Hamilton-Jacobi reachability method, we address the RA problem by designing a new Lipschitz continuous RA value function, whose zero sublevel set exactly characterizes the RA set. We establish that the associated Bellman backup operator is contractive and that the RA value function is the unique viscosity solution of a Hamilton-Jacobi variational inequality. Finally, we develop a two-step framework for the SA problem by integrating our RA strategies with a recently proposed Robust Control Lyapunov-Value Function, thereby ensuring both target reachability and long-term stability. We numerically verify our RA and SA frameworks on a 3D Dubins car system to demonstrate the efficacy of the proposed approach.
>
---
#### [replaced 002] TwinRL: Digital Twin-Driven Reinforcement Learning for Real-World Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决真实世界中VLA模型训练成本高、探索效率低的问题。提出TwinRL框架，通过数字孪生引导强化学习，提升探索效率与成功率。**

- **链接: [https://arxiv.org/pdf/2602.09023](https://arxiv.org/pdf/2602.09023)**

> **作者:** Qinwen Xu; Jiaming Liu; Rui Zhou; Shaojun Shi; Nuowei Han; Zhuoyang Liu; Chenyang Gu; Shuo Gu; Yang Yue; Gao Huang; Wenzhao Zheng; Sirui Han; Peng Jia; Shanghang Zhang
>
> **摘要:** Despite strong generalization capabilities, Vision-Language-Action (VLA) models remain constrained by the high cost of expert demonstrations and limited real-world interaction. While online reinforcement learning (RL) has shown promise, its application to real-world VLA manipulation is hindered by low exploration efficiency and restricted exploration coverage. Through systematic real-world experiments, we observe that the effective exploration space of online RL is largely constrained by the trajectory distribution induced during supervised fine-tuning (SFT). Motivated by this observation, we propose TwinRL, a digital twin-real-world collaborative post-training framework that expands and guides RL exploration for VLA models through three stages: SFT warm-up, twin RL warm-up, and real-world RL. TwinRL first reconstructs a high-fidelity digital twin from smartphone-captured scenes. During the SFT stage, we introduce an exploration space expansion strategy that expands the support of the trajectory distribution beyond real demonstrations, reshaping the exploration space for more effective RL. Rather than treating the twin as a data augmentation tool, we propose a twin RL warm-up strategy that enables it to act as an exploration guide for real-world RL. Specifically, TwinRL performs efficient parallel RL in the digital twin to generate interactive trajectories that populate the replay buffer and stabilize subsequent real-world RL learning. This process also identifies failure-prone yet informative configurations, enabling targeted human-in-the-loop rollouts to further improve on-robot efficiency. Across four tasks, TwinRL achieves near-100% success in both in-distribution and out-of-distribution regions, delivering over 30% faster convergence than prior real-world RL methods with only 20 minutes of on-robot interaction.
>
---
#### [replaced 003] 3D Modeling and Automated Measurement of Concrete Cracks via Segment Anything Refinement and Visual Inertial LiDAR Fusion
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于混凝土裂缝检测任务，解决传统方法在复杂场景下的适应性差、精度不足问题，通过融合视觉与LiDAR数据实现精准3D裂缝建模与测量。**

- **链接: [https://arxiv.org/pdf/2501.09203](https://arxiv.org/pdf/2501.09203)**

> **作者:** Pengru Deng; Jiapeng Yao; Chun Li; Su Wang; Xinrun Li; Varun Ojha; Xuhui He
>
> **备注:** Title and author list updated
>
> **摘要:** Visual-Spatial Systems has become increasingly essential in concrete crack inspection. However, existing methods often lacks adaptability to diverse scenarios, exhibits limited robustness in image-based approaches, and struggles with curved or complex geometries. To address these limitations, an innovative framework for two-dimensional (2D) crack detection, three-dimensional (3D) reconstruction, and 3D automatic crack measurement was proposed by integrating computer vision technologies and multi-modal Simultaneous localization and mapping (SLAM) in this study. Firstly, building on a base DeepLabv3+ segmentation model, and incorporating specific refinements utilizing foundation model Segment Anything Model (SAM), we developed a crack segmentation method with strong generalization across unfamiliar scenarios, enabling the generation of precise 2D crack masks. To enhance the accuracy and robustness of 3D reconstruction, Light Detection and Ranging (LiDAR) point clouds were utilized together with image data and segmentation masks. By leveraging both image- and LiDAR-SLAM, we developed a multi-frame and multi-modal fusion framework that produces dense, colorized point clouds, effectively capturing crack semantics at a 3D real-world scale. Furthermore, the crack geometric attributions were measured automatically and directly within 3D dense point cloud space, surpassing the limitations of conventional 2D image-based measurements. This advancement makes the method suitable for structural components with curved and complex 3D geometries. Experimental results across various concrete structures highlight the significant improvements and unique advantages of the proposed method, demonstrating its effectiveness, accuracy, and robustness in real-world applications.
>
---
#### [replaced 004] Active Learning of Fractional-Order Viscoelastic Model Parameters for Realistic Haptic Rendering
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于参数优化任务，旨在解决分数阶粘弹性模型参数难以确定的问题，通过主动学习和人类反馈优化参数，提升触觉模拟的真实感。**

- **链接: [https://arxiv.org/pdf/2512.00667](https://arxiv.org/pdf/2512.00667)**

> **作者:** Harun Tolasa; Gorkem Gemalmaz; Volkan Patoglu
>
> **备注:** This work has been submitted to the IEEE Transactions on Haptics for possible publication. 14 pages, 8 figures
>
> **摘要:** Effective medical simulators necessitate realistic haptic rendering of biological tissues that exhibit viscoelastic material properties, such as creep and stress relaxation. Fractional-order models provide an effective means of describing intrinsically time-dependent viscoelastic dynamics with few parameters, as they naturally capture memory effects. However, due to the unintuitive, frequency-dependent coupling among the order of the fractional element and other parameters, determining appropriate parameter values for fractional-order models that yield high perceived realism remains a significant challenge. In this study, we propose a systematic means of determining the parameters of fractional-order viscoelastic models that optimizes the perceived realism of haptic rendering across general populations. First, we demonstrate that the parameters of fractional-order models can be effectively optimized through active learning, using qualitative feedback-based human-in-the-loop (HiL) optimization, to ensure consistently high realism ratings for each individual. Second, we propose a rigorous method to combine HiL optimization results into an aggregate perceptual map trained on the entire dataset, and demonstrate how to select population-level optimal parameters from this representation that are broadly perceived as realistic across general populations. Finally, we provide evidence of the effectiveness of the generalized fractional-order viscoelastic model parameters for three viscoelastic materials by characterizing their perceived realism through human-subject experiments. Overall, generalized fractional-order viscoelastic models established through the proposed HiL optimization and aggregation approach possess the potential to significantly improve the sim-to-real transition performance of medical training simulators.
>
---
#### [replaced 005] A Practical Framework of Key Performance Indicators for Multi-Robot Lunar and Planetary Field Tests
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人行星探测任务，旨在解决不同实验间性能比较困难的问题。通过构建关键绩效指标框架，提升多机器人系统评估的一致性和目标导向性。**

- **链接: [https://arxiv.org/pdf/2601.20529](https://arxiv.org/pdf/2601.20529)**

> **作者:** Julia Richter; David Oberacker; Gabriela Ligeza; Valentin T. Bickel; Philip Arm; William Talbot; Marvin Grosse Besselmann; Florian Kehl; Tristan Schnell; Hendrik Kolvenbach; Rüdiger Dillmann; Arne Roennau; Marco Hutter
>
> **备注:** Presented at ICRA 2026 Workshop on Multi-Agent Robotic Systems: Real-World Collaboration and Interaction
>
> **摘要:** Robotic prospecting for critical resources on the Moon, such as ilmenite, rare earth elements, and water ice, requires robust exploration methods given the diverse terrain and harsh environmental conditions. Although numerous analog field trials address these goals, comparing their results remains challenging because of differences in robot platforms and experimental setups. These missions typically assess performance using selected, scenario-specific engineering metrics that fail to establish a clear link between field performance and science-driven objectives. In this paper, we address this gap by deriving a structured framework of KPI from three realistic multi-robot lunar scenarios reflecting scientific objectives and operational constraints. Our framework emphasizes scenario-dependent priorities in efficiency, robustness, and precision, and is explicitly designed for practical applicability in field deployments. We validated the framework in a multi-robot field test and found it practical and easy to apply for efficiency- and robustness-related KPI, whereas precision-oriented KPI require reliable ground-truth data that is not always feasible to obtain in outdoor analog environments. Overall, we propose this framework as a common evaluation standard enabling consistent, goal-oriented comparison of multi-robot field trials and supporting systematic development of robotic systems for future planetary exploration.
>
---
#### [replaced 006] Neural Configuration-Space Barriers for Manipulation Planning and Control
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于机器人运动规划与控制任务，解决高维机械臂在复杂环境中的安全高效路径规划问题。通过神经网络构建配置空间障碍，提升规划效率与控制鲁棒性。**

- **链接: [https://arxiv.org/pdf/2503.04929](https://arxiv.org/pdf/2503.04929)**

> **作者:** Kehan Long; Ki Myung Brian Lee; Nikola Raicevic; Niyas Attasseri; Melvin Leok; Nikolay Atanasov
>
> **摘要:** Planning and control for high-dimensional robot manipulators in cluttered dynamic environments require computational efficiency and robust safety guarantees. Inspired by recent advances in learning configuration-space distance functions (CDFs) as representations of robot bodies, we propose a unified approach for motion planning and control that formulates safety constraints as CDF barriers. A CDF barrier approximates the local free configuration space, substantially reducing the number of collision-checking operations during motion planning. However, learning a CDF barrier with a neural network and relying on online sensor observations introduces uncertainties that must be considered during control synthesis. To address this, we develop a distributionally robust CDF barrier formulation for control that accounts for modeling errors and sensor noise without assuming a known underlying distribution. Simulations and hardware experiments on a UFactory xArm6 manipulator show that our neural CDF barrier formulation enables efficient planning and robust safe control in cluttered and dynamic environments, relying only on onboard point-cloud observations.
>
---
#### [replaced 007] HoloMotion-1 Technical Report
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出HoloMotion-1，属于人体运动追踪任务，解决零样本全身运动跟踪问题，通过混合数据训练提升模型泛化能力和准确性。**

- **链接: [https://arxiv.org/pdf/2605.15336](https://arxiv.org/pdf/2605.15336)**

> **作者:** Maiyue Chen; Kaihui Wang; Bo Zhang; Xihan Ma; Zhiyuan Yang; Yi Ren; Qijun Huang; Zihao Zhu; Yucheng Wang; Zhizhong Su
>
> **备注:** 20 pages, 4 figures, 6 tables. Technical report
>
> **摘要:** In this report, we present HoloMotion-1, a humanoid motion foundation model for zero-shot whole-body motion tracking. A key innovation of HoloMotion-1 is to scale control-policy training with a large-scale hybrid motion corpus, where video-reconstructed motions from in-the-wild videos provide the dominant source of motion diversity, while curated motion-capture and in-house motion data provide higher-fidelity supervision and deployment-oriented coverage. This data regime enables HoloMotion-1 to move beyond conventional MoCap-only training and exposes the policy to substantially broader behaviors, capture conditions, and motion styles. Learning from such heterogeneous data introduces new challenges, including reconstruction noise, source-domain mismatch, uneven motion quality, and the need for temporal modeling under large behavioral variation. To address these challenges, HoloMotion-1 integrates large-capacity temporal modeling, a sparsely activated Mixture-of-Experts Transformer with KV-cache inference for real-time control, and a sequence-level training strategy that improves learning efficiency on extended motion sequences. Extensive experiments on multiple unseen motion benchmarks show that HoloMotion-1 generalizes robustly across diverse motion types and capture conditions, significantly improves tracking accuracy over prior methods, and transfers directly to a real humanoid robot without task-specific fine-tuning.
>
---
#### [replaced 008] HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks
- **分类: cs.RO**

- **简介: 该论文提出HDFlow框架，用于解决长时域任务的规划问题。通过结合扩散模型与流模型，提升规划效率与效果。**

- **链接: [https://arxiv.org/pdf/2605.04525](https://arxiv.org/pdf/2605.04525)**

> **作者:** Nandiraju Gireesh; Yuanliang Ju; Chaoyi Xu; Weiheng Liu; Yuxuan Wan; He Wang
>
> **备注:** ICML 2026 (Spotlight)
>
> **摘要:** Recent advances in generative models have shown promise in generating behavior plans for long-horizon, sparse reward tasks. While these approaches have achieved promising results, they often lack a principled framework for hierarchical decomposition and struggle with the computational demands of real-time execution, due to their iterative denoising process. In this work, we introduce Hierarchical Diffusion-Flow (HDFlow), a novel hierarchical planning framework that optimally leverages the strengths of diffusion and rectified flow models to overcome the limitations of single-paradigm generative planners. HDFlow employs a high-level diffusion planner to generate sequences of strategic subgoals in a learned latent space, capitalizing on diffusion's powerful exploratory capabilities. These subgoals then guide a low-level rectified flow planner that generates smooth and dense trajectories, exploiting the speed and efficiency of ordinary differential equation (ODE)-based trajectory generation. We evaluate HDFlow on four challenging furniture assembly tasks in both simulation and real-world, where it significantly outperforms state-of-the-art methods. Furthermore, we also showcase our method's generalizability on two long-horizon benchmarks comprising diverse locomotion and manipulation tasks. Project website: this https URL
>
---
#### [replaced 009] Receptogenesis in a Vascularized Robotic Embodiment
- **分类: cs.RO; cond-mat.mtrl-sci**

- **简介: 该论文属于机器人学任务，旨在解决机器人实时物理适应问题。通过流体系统实现材料重组，完成传感器的现场生成，提升机器人环境响应能力。**

- **链接: [https://arxiv.org/pdf/2603.09473](https://arxiv.org/pdf/2603.09473)**

> **作者:** Kadri-Ann Pankratov; Leonid Zinatullin; Hans Priks; Adele Metsniit; Urmas Johanson; Tarmo Tamm; Alvo Aabloo; Edoardo Sinibaldi; Indrek Must
>
> **备注:** Supplementary Files currently unavailable online. Please contact the First Author to request any Supplementary Files Version 2 - revision
>
> **摘要:** Equipping robotic systems with the capacity to generate $\textit{ex novo}$ hardware during operation extends control of physical adaptability. Unlike modular systems that rely on discrete component integration pre- or post-deployment, we envision the possibility that physical adaptation and development emerge from dynamic material restructuring to shape the body's intrinsic functions. Drawing inspiration from circulatory systems that redistribute mass and function in biological organisms, we utilize fluidics to restructure the material interface, a capability currently unpaired in robotics. Here, we realize this synthetic growth capability through a vascularized robotic composite designed for programmable material synthesis, demonstrated via receptogenesis - the on-demand construction of sensors from internal fluid reserves based on environmental cues. By coordinating the fluidic transport of precursors with external localized UV irradiation, we drive an $\textit{in situ}$ photopolymerization that chemically reconstructs the vasculature from the inside out. This reaction converts precursors with photolatent initiator into a solid dispersion of UV-sensitive polypyrrole in PETG, establishing a sensing modality validated by a characteristic decrease in electrical impedance. The newly synthesized sensor closed a local control loop to regulate wing flapping in a moth-inspired robotic demonstrator. This physical update increased the robot's capability in real time. Material-level functional restructuring of the vascularized robot body provides a proof-of-concept materials basis for $\textit{ex novo}$ hardware generation in situated robotic systems - a step toward situated robots in which a reaction to environmental stimuli autonomously produces hardware updates to match new environmental demands.
>
---
#### [replaced 010] Certifiable Alignment of GNSS and Local Frames via Lagrangian Duality
- **分类: cs.RO**

- **简介: 该论文属于GNSS与本地坐标系对齐任务，解决传统方法依赖卫星数量和易陷入局部最优的问题，提出一种可验证全局最优的求解方法。**

- **链接: [https://arxiv.org/pdf/2512.20931](https://arxiv.org/pdf/2512.20931)**

> **作者:** Baoshan Song; Matthew Giamou; Penggao Yan; Chunxi Xia; Li-Ta Hsu
>
> **备注:** Final version in RA-L
>
> **摘要:** Estimating the absolute orientation of a local system relative to a global navigation satellite system (GNSS) reference often suffers from local minima and high dependency on satellite availability. Existing methods for this alignment task rely on abundant satellites unavailable in GNSS-degraded environments, or use local optimization methods which cannot guarantee the optimality of a solution. This work introduces a globally optimal solver that transforms raw pseudo-range or Doppler measurements into a convexly relaxed problem. The proposed method is certifiable, meaning it can numerically verify the correctness of the result, filling a gap where existing local optimizers fail. We first formulate the original frame alignment problem as a nonconvex quadratically constrained quadratic program (QCQP) problem and relax the QCQP problem to a concave Lagrangian dual problem that provides a lower cost bound for the original problem. Then we perform relaxation tightness and observability analysis to derive criteria for certifiable optimality of the solution. Finally, simulation and real world experiments are conducted to evaluate the proposed method. The experiments show that our method provides certifiably optimal solutions even with only 2 satellites with Doppler measurements and 2D vehicle motion, while the traditional velocity-based VOBA method and the advanced GVINS alignment technique may fail or converge to local optima without notice. To support the development of GNSS-based navigation techniques in robotics, all code and data are open-sourced at this https URL.
>
---
#### [replaced 011] Iterative Compositional Data Generation for Robot Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，解决多任务数据生成问题。通过构建可组合的扩散Transformer模型，提升未见任务的泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.10891](https://arxiv.org/pdf/2512.10891)**

> **作者:** Anh-Quan Pham; Marcel Hussing; Shubhankar P. Patankar; Dani S. Bassett; Jorge Mendez-Mendez; Eric Eaton
>
> **摘要:** Collecting robotic manipulation data is expensive, making it impractical to acquire demonstrations for the combinatorially large space of tasks that arise in multi-object, multi-robot, and multi-environment settings. While recent generative models can synthesize useful data for individual tasks, they do not exploit the compositional structure of robotic domains and struggle to generalize to unseen task combinations. We propose a semantic compositional diffusion transformer that factorizes transitions into robot-, object-, obstacle-, and objective-specific components and learns their interactions through attention. Once trained on a limited subset of tasks, we show that our model can zero-shot generate high-quality transitions from which we can learn control policies for unseen task combinations. Then, we introduce an iterative self-improvement procedure in which synthetic data is validated via offline reinforcement learning and incorporated into subsequent training rounds. Our approach substantially improves zero-shot performance over monolithic and hard-coded compositional baselines, ultimately solving nearly all held-out tasks and demonstrating the emergence of meaningful compositional structure in the learned representations.
>
---
#### [replaced 012] Causality-Aware End-to-End Autonomous Driving via Ego-Centric Joint Scene Modeling
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，旨在解决传统方法忽视车辆与周围交通参与者因果关系的问题。提出CaAD框架，通过联合建模和策略对齐提升轨迹预测的可靠性。**

- **链接: [https://arxiv.org/pdf/2605.13646](https://arxiv.org/pdf/2605.13646)**

> **作者:** Seokha Moon; Minseung Lee; Joon Seo; Jinkyu Kim; Jungbeom Lee
>
> **摘要:** End-to-end autonomous driving, which bypasses traditional modular pipelines by directly predicting future trajectories from sensor inputs, has recently achieved substantial progress. However, existing methods often overlook the causal inter-dependencies in ego-vehicle planning, ignoring the reciprocal relations between the ego vehicle and surrounding agents. This causal oversight leads to inconsistent and unreliable trajectory predictions, especially in interaction-critical scenarios where ego decisions and neighboring agent behaviors must be reasoned about jointly. To address this limitation, we propose CaAD, a Causality-aware end-to-end Autonomous Driving framework that captures these dependencies within a shared latent scene representation. First, we propose an ego-centric joint-causal modeling module that builds on the marginal prediction branch, and learns causal dependencies between the ego vehicle and interaction-relevant agents. Second, we employ a causality-aware policy alignment stage implemented with joint-mode embeddings to align the stochastic ego policy with planning-oriented closed-loop feedback computed from surrounding traffic and map context. On the Bench2Drive and NAVSIM benchmarks, CaAD demonstrates strong closed-loop planning performance, achieving a Driving Score of 87.53 and Success Rate of 71.81 on Bench2Drive, and a PDMS of 91.1 on NAVSIM. The project page is available at this https URL.
>
---
#### [replaced 013] RoboMD: Uncovering Robot Vulnerabilities through Semantic Potential Fields
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人安全任务，旨在解决现实环境中机器人操作策略的脆弱性问题。通过虚拟环境学习潜在场，识别并分析漏洞，提升操作性能。**

- **链接: [https://arxiv.org/pdf/2412.02818](https://arxiv.org/pdf/2412.02818)**

> **作者:** Som Sagar; Jiafei Duan; Sreevishakh Vasudevan; Yifan Zhou; Heni Ben Amor; Dieter Fox; Ransalu Senanayake
>
> **备注:** 26 Pages, 20 figures
>
> **摘要:** Robot manipulation policies, while central to the promise of physical AI, are highly vulnerable in the presence of external variations in the real world. Diagnosing these vulnerabilities is hindered by two key challenges: (i) the relevant variations to test against are often unknown, and (ii) direct testing in the real world is costly and unsafe. We introduce a framework that tackles both issues by learning a separate deep reinforcement learning (deep RL) policy for vulnerability prediction through virtual runs on a continuous vision-language embedding trained with limited success-failure data. By treating this embedding space, which is rich in semantic and visual variations, as a potential field, the policy learns to move toward vulnerable regions while being repelled from success regions. This vulnerability prediction policy, trained on virtual rollouts, enables scalable and safe vulnerability analysis without expensive physical trials. By querying this policy, our framework builds a probabilistic vulnerability-likelihood map. Experiments across simulation benchmarks and a physical robot arm show that our framework uncovers up to 23% more unique vulnerabilities than state-of-the-art vision-language baselines, revealing subtle vulnerabilities overlooked by heuristic testing. Additionally, we show that fine-tuning the manipulation policy with the vulnerabilities discovered by our framework improves manipulation performance with much less fine-tuning data.
>
---
#### [replaced 014] Preserving Foundational Capabilities in Flow-Matching VLAs through Conservative SFT
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型的微调任务，旨在解决参数过度更新导致能力退化的问题。提出ConSFT方法，在不依赖先验数据的情况下，有效保持模型原有能力。**

- **链接: [https://arxiv.org/pdf/2605.08879](https://arxiv.org/pdf/2605.08879)**

> **作者:** Tianyi Zhang; Shaopeng Zhai; Haoran Zhang; Fuxian Huang; Qi Zhang
>
> **备注:** 20 pages, 9 figures
>
> **摘要:** Unconstrained fine-tuning of flow-matching Vision-Language-Action (VLA) models drives dense parameter overwrites, degrading pre-trained capabilities. We present Conservative Supervised Fine-Tuning (ConSFT), an optimization objective that adapts to target distributions while mitigating catastrophic forgetting, requiring zero prior data or architectural overhead. By dynamically scaling learning signals based on model confidence, ConSFT suppresses excessive gradients from low-confidence samples to prevent disproportionate parameter updates, thereby bounding the intrinsic parameter disruption risk. Inspired by reinforcement learning's trust-region clipping, this formulation establishes a progressive learning dynamic to secure target convergence and prior capability retention, maintaining sparse parameter updates without relying on the parallel reference networks required by explicit regularization. We evaluate ConSFT on the LIBERO and RoboTwin benchmarks across state-of-the-art flow-matching VLAs ($\pi_0$, $\pi_{0.5}$, and GR00T-N1.6-3B). The method outperforms vanilla SFT in capability retention by an average absolute margin of over 20\%, matching the efficacy of data-heavy Experience Replay in a prior-data-free regime. Real-world robotic deployments confirm that ConSFT precludes spatial overfitting during downstream adaptation, preserving pre-trained physical skills while acquiring sequential target tasks.
>
---
#### [replaced 015] Distributionally Robust Safety Under Arbitrary Uncertainties: A Safety Filtering Approach
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于安全控制任务，解决非线性系统在分布不确定性下的概率安全性问题。通过安全过滤框架，结合Wasserstein模糊集，实现高效的安全认证与切换策略。**

- **链接: [https://arxiv.org/pdf/2605.12974](https://arxiv.org/pdf/2605.12974)**

> **作者:** Daniel M. Cherenson; Haejoon Lee; Taekyung Kim; Dimitra Panagou
>
> **备注:** 10 pages, 4 figures, submitted to IEEE Robotics and Automation Letters (RA-L); Project Page: this https URL
>
> **摘要:** In this work, we study how to ensure probabilistic safety for nonlinear systems under distributional ambiguity. Our approach builds on a backup-based safety filtering framework that switches between a high-performance nominal policy and a certified backup policy to ensure safety. To handle arbitrary uncertainties from ambiguous distributions, i.e., where the distribution is not of specific structure and the true distribution is unknown, we adopt a distributionally robust (DR) formulation using Wasserstein ambiguity sets. Rather than solving a high-dimensional DR trajectory optimization problem online, we exploit the structure of backup-based safety filtering to reduce safety certification to a one-dimensional search over the switching time between nominal and backup policies. We then develop a sampling-based certification procedure with finite-sample guarantees, where empirical failure probabilities are compared against a Wasserstein-inflated threshold. We validate our method through simulations across three systems, from a Dubins vehicle to a high-speed racing car and a fighter jet, demonstrating the broad applicability and computational efficiency.
>
---
#### [replaced 016] HEX: Humanoid-Aligned Experts for Cross-Embodiment Whole-Body Manipulation
- **分类: cs.RO**

- **简介: 该论文属于人形机器人操控任务，解决高自由度控制不稳定问题。提出HEX框架，实现全身协调与长期任务执行。**

- **链接: [https://arxiv.org/pdf/2604.07993](https://arxiv.org/pdf/2604.07993)**

> **作者:** Shuanghao Bai; Meng Li; Xinyuan Lv; Jiawei Wang; Xinhua Wang; Fei Liao; Chengkai Hou; Langzhe Gu; Wanqi Zhou; Kun Wu; Ziluo Ding; Zhiyuan Xu; Lei Sun; Shanghang Zhang; Zhengping Che; Jian Tang; Badong Chen
>
> **备注:** Project page: this https URL
>
> **摘要:** Humans achieve complex manipulation through coordinated whole-body control, whereas most Vision-Language-Action (VLA) models treat robot body parts largely independently, making high-DoF humanoid control challenging and often unstable. We present HEX, a state-centric framework for coordinated manipulation on full-sized bipedal humanoid robots. HEX introduces a humanoid-aligned universal state representation for scalable learning across heterogeneous embodiments, and incorporates a Mixture-of-Experts Unified Proprioceptive Predictor to model whole-body coordination and temporal motion dynamics from large-scale multi-embodiment trajectory data. To efficiently capture temporal visual context, HEX uses lightweight history tokens to summarize past observations, avoiding repeated encoding of historical images during inference. It further employs a residual-gated fusion mechanism with a flow-matching action head to adaptively integrate visual-language cues with proprioceptive dynamics for action generation. Experiments on real-world humanoid manipulation tasks show that HEX achieves state-of-the-art performance in task success rate and generalization, particularly in fast-reaction and long-horizon scenarios.
>
---
#### [replaced 017] Q-learning with Adjoint Matching
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文提出QAM算法，解决连续动作强化学习中策略优化难题，通过邻接匹配技术实现稳定高效的策略更新，提升稀疏奖励任务性能。**

- **链接: [https://arxiv.org/pdf/2601.14234](https://arxiv.org/pdf/2601.14234)**

> **作者:** Qiyang Li; Sergey Levine
>
> **备注:** 32 pages, 8 figures, 7 tables
>
> **摘要:** We propose Q-learning with Adjoint Matching (QAM), a novel TD-based reinforcement learning (RL) algorithm that tackles a long-standing challenge in continuous-action RL: efficient optimization of an expressive diffusion or flow-matching policy with respect to a parameterized Q-function. Effective optimization requires exploiting the first-order information of the critic, but it is challenging to do so for flow or diffusion policies because direct gradient-based optimization via backpropagation through their multi-step denoising process is numerically unstable. Existing methods work around this either by only using the value and discarding the gradient information, or by relying on approximations that sacrifice policy expressivity or bias the learned policy. QAM sidesteps both of these challenges by leveraging adjoint matching, a recently proposed technique in generative modeling, which transforms the critic's action gradient to form a step-wise objective function that is free from unstable backpropagation, while providing an unbiased, expressive policy at the optimum. Combined with temporal-difference backup for critic learning, QAM consistently outperforms prior approaches on hard, sparse reward tasks in both offline and offline-to-online RL.
>
---
#### [replaced 018] Learn2Decompose: Learning Problem Decomposition for Efficient Sequential Multi-object Manipulation Planning
- **分类: cs.RO**

- **简介: 该论文属于多物体操作任务，解决动态环境中顺序操作规划效率低的问题。通过学习问题分解方法提升TAMP求解效率。**

- **链接: [https://arxiv.org/pdf/2408.06843](https://arxiv.org/pdf/2408.06843)**

> **作者:** Yan Zhang; Teng Xue; Amirreza Razmjoo; Sylvain Calinon
>
> **备注:** Extension of RAL version: added PR2 Whole-body kitchen task and detailed discussion on limitations in main text; added pseudocode and robustness analysis of our approach, and formal analysis on why and when task goals are decomposable in appendix
>
> **摘要:** We present an efficient task and motion replanning approach for sequential multi-object manipulation in dynamic environments. Conventional Task And Motion Planning (TAMP) solvers experience an exponential increase in planning time as the planning horizon and number of objects grow, limiting their applicability in real-world scenarios. To address this, we propose learning problem decompositions from demonstrations to accelerate TAMP solvers. Our approach consists of three key components: goal decomposition learning, computational distance learning, and object reduction. Goal decomposition identifies the necessary sequences of states that the system must pass through before reaching the final goal, treating them as subgoal sequences. Computational distance learning predicts the computational complexity between two states, enabling the system to identify the temporally closest subgoal from a disturbed state. Object reduction minimizes the set of active objects considered during replanning, further improving efficiency. We evaluate our approach on three benchmarks, demonstrating its effectiveness in improving replanning efficiency for sequential multi-object manipulation tasks in dynamic environments.
>
---
#### [replaced 019] Hybrid Training for Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文研究视觉-语言-动作模型的混合训练方法，旨在解决长链式思考影响推理效率的问题，通过允许推理时省略思考步骤，提升模型灵活性与实用性。**

- **链接: [https://arxiv.org/pdf/2510.00600](https://arxiv.org/pdf/2510.00600)**

> **作者:** Pietro Mazzaglia; Cansu Sancaktar; Markus Peschl; Daniel Dijkman
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Using Large Language Models to produce intermediate thoughts, a.k.a. Chain-of-thought (CoT), before providing an answer has been a successful recipe for solving complex language tasks. In robotics, similar embodied CoT strategies, generating thoughts before actions, have also been shown to lead to improved performance when using Vision-Language-Action models (VLAs). As these techniques increase the length of the model's generated outputs to include the thoughts, the inference time is negatively affected. Delaying an agent's actions in real-world executions, as in robotic manipulation settings, strongly affects the usability of a method, as tasks require long sequences of actions. However, is the generation of long chains-of-thought a strong prerequisite for achieving performance improvements? In this work, we explore the idea of Hybrid Training (HyT), a framework that enables VLAs to learn from thoughts and benefit from the associated performance gains, while enabling the possibility to leave out CoT generation during inference. Furthermore, by learning to conditionally predict a diverse set of outputs, HyT supports flexibility at inference time, enabling the model to either predict actions directly, generate thoughts or follow instructions. We evaluate the proposed method in a series of simulated benchmarks and real-world experiments.
>
---
#### [replaced 020] Compliant Explicit Reference Governor for Contact Friendly Robotic Manipulators
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出CERG系统，用于机器人安全接触控制。解决机器人与环境交互时的安全与性能平衡问题，通过限制接触能量确保安全，同时不影响无接触时的性能。**

- **链接: [https://arxiv.org/pdf/2504.09188](https://arxiv.org/pdf/2504.09188)**

> **作者:** Yaashia Gautam; Gilberto Briscoe-Martinez; Adhitya Mohan; Nataliya Nechyporenko; Alessandro Roncone; Marco M. Nicotra
>
> **备注:** Updated paper with current contributions and author list , accepted at IFAC World Congress, Busan, 2026
>
> **摘要:** This paper introduces the Compliant Explicit Reference Governor (CERG), a modular reference management system that enables robots to interact physically with their environment under provable guarantees. The CERG is an intermediate layer that can be placed between a high-level planner and a low-level controller: it enforces operational constraints and enables smooth transitions between free-motion and contact operations. The CERG ensures safety by limiting the total energy available to the robotic arm at the time of contact. In the absence of contact, however, the CERG does not penalize the system performance. Simulation and hardware experiments validate the CERG on increasingly complex systems.
>
---
#### [replaced 021] Data-centric Design of Learning-based Surgical Gaze Perception Models in Multi-Task Simulation
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于手术视觉感知任务，旨在解决专家 gaze 数据获取困难的问题。通过构建多任务数据集，比较主动与被动 gaze 的效果，评估其在模型训练中的替代性。**

- **链接: [https://arxiv.org/pdf/2602.09259](https://arxiv.org/pdf/2602.09259)**

> **作者:** Yizhou Li; Shuyuan Yang; Jiaji Su; Zonghe Chua
>
> **备注:** 8 pages, conference pre-print
>
> **摘要:** In robot-assisted minimally invasive surgery (RMIS), reduced haptic feedback and depth cues increase reliance on expert visual perception, motivating gaze-guided training and learning-based surgical perception models. However, operative expert gaze is costly to collect, and it remains unclear how the source of gaze supervision, both expertise level (intermediate vs. novice) and perceptual modality (active execution vs. passive viewing), shapes what attention models learn. We introduce a paired active-passive, multi-task surgical gaze dataset collected on the da Vinci SimNow simulator across four drills. Active gaze was recorded during task execution using a VR headset with eye tracking, and the corresponding videos were reused as stimuli to collect passive gaze from observers, enabling controlled same-video comparisons. We quantify skill- and modality-dependent differences in gaze organization and evaluate the substitutability of passive gaze for operative supervision using fixation density overlap analyses and single-frame saliency modeling. Across settings, MSI-Net produced stable, interpretable predictions, whereas SalGAN was unstable and often poorly aligned with human fixations. Models trained on passive gaze recovered a substantial portion of intermediate active attention, but with predictable degradation, and transfer was asymmetric between active and passive targets. Notably, novice passive labels approximated intermediate-passive targets with limited loss on higher-quality demonstrations, suggesting a practical path for scalable, crowd-sourced gaze supervision in surgical coaching and perception modeling.
>
---
#### [replaced 022] EfficientTDMPC: Improved MPC Objectives for Sample-Efficient Continuous Control
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出EfficientTDMPC，解决连续控制中的样本效率问题。通过集成动态模型和不确定性惩罚，提升策略优化效果，实现更高效的强化学习。**

- **链接: [https://arxiv.org/pdf/2605.16692](https://arxiv.org/pdf/2605.16692)**

> **作者:** Thomas Evers; Cristian Meo; Wendelin Bohmer; Justin Dauwels; Yaniv Oren
>
> **摘要:** We introduce EfficientTDMPC, a sample-efficient model-based reinforcement learning method for continuous control built on the TD-MPC family of algorithms. Central to this family is a planner that aims to find an action sequence that maximizes the estimated return. The return is estimated using a learned model and value networks, each of which can introduce error. EfficientTDMPC proposes to reduce this error in two ways. First, it introduces an ensemble of dynamics models and averages the return estimates across those models and across different rollout depths. Second, it adds the option to apply an uncertainty penalty to the planner objective, yielding a planner that avoids actions with uncertain return estimates. It then adds practical improvements which increase buffer data freshness and reduce compute. Lastly, we find that our contributions enable EfficientTDMPC to benefit more from a higher update-to-data (UTD) ratio, further improving sample efficiency. To the best of our knowledge, in the low data regime of each benchmark, EfficientTDMPC achieves state-of-the-art (SOTA) in terms of sample efficiency on HumanoidBench-Hard and DMC hard, while matching SOTA on DMC easy.
>
---
#### [replaced 023] VECTOR-Drive: Tightly Coupled Vision-Language and Trajectory Expert Routing for End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出VECTOR-DRIVE，解决端到端自动驾驶中语义与轨迹耦合问题。通过共享注意力和专家路由，实现视觉语言与轨迹预测的紧密融合，提升驾驶性能。**

- **链接: [https://arxiv.org/pdf/2605.08830](https://arxiv.org/pdf/2605.08830)**

> **作者:** Rui Zhao; Jianlin Yu; Zhenhai Gao; Jiaqiao Liu; Fei Gao
>
> **摘要:** End-to-end autonomous driving requires models to understand traffic scenes, infer driving intent, and generate executable motion plans. Recent vision-language-action (VLA) models inherit semantic priors from large-scale vision-language pretraining, yet still face a coupling trade-off: fully shared backbones preserve multimodal interaction but may entangle language reasoning and trajectory prediction, whereas decou pled reasoning-action pipelines reduce task conflict but weaken semantic-motion coupling. We propose VECTOR-DRIVE, a tightly coupled VLA framework built on Qwen2.5-VL-3B. VECTOR-DRIVE keeps all tokens coupled through shared self attention and routes feed-forward computation according to token semantics. Vision and language tokens are processed by a Vision-Language Expert to preserve semantic priors, while target-point, ego-state, and noisy action tokens are routed to a Trajectory Expert for motion-specific computation. On the action-token pathway, a flow-matching planner refines noisy action tokens into future waypoints and speed profiles. This design couples semantic reasoning and motion planning within a single multimodal Transformer while separating task-specific FFN computation. On Bench2Drive, VECTOR-DRIVE achieves 88.91 Driving Score and outperforms representative end-to end and VLA-based baselines. Qualitative results and ablations further validate the benefits of shared attention, semantic-aware expert routing, progressive training, and flow-based action de coding.
>
---
#### [replaced 024] Robots that learn to evaluate models of collective behavior
- **分类: cs.RO**

- **简介: 该论文属于行为模型评估任务，旨在解决传统方法无法准确评价动物行为模型的问题。通过机器人与真实鱼群的闭环互动，量化模型与实际行为的差异，提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2604.07303](https://arxiv.org/pdf/2604.07303)**

> **作者:** Mathis Hocke; Andreas Gerken; David Bierbach; Jens Krause; Tim Landgraf
>
> **摘要:** Understanding and modeling animal behavior is essential for studying collective motion, decision-making, and bio-inspired robotics. Yet, evaluating the accuracy of behavioral models still often relies on offline comparisons to static trajectory statistics. Here we introduce a reinforcement-learning-based framework that uses a biomimetic robotic fish (RoboFish) to evaluate computational models of live fish behavior through closed-loop interaction. We trained policies in simulation using four distinct fish models-a simple constant-follow baseline, two rule-based models, and a biologically grounded convolutional neural network model-and transferred these policies to the real RoboFish setup, where they interacted with live fish. Policies were trained to guide a simulated fish to goal locations, enabling us to quantify how the response of real fish differs from the simulated fish's response. We evaluate the fish models by quantifying the sim-to-real gaps, defined as the Wasserstein distance between simulated and real distributions of behavioral metrics such as goal-reaching performance, inter-individual distances, wall interactions, and alignment. The neural network-based fish model exhibited the smallest gap across goal-reaching performance and most other metrics, indicating higher behavioral fidelity than conventional rule-based models under this benchmark. More importantly, this separation shows that the proposed evaluation can quantitatively distinguish candidate models under matched closed-loop conditions. Our work demonstrates how learning-based robotic experiments can uncover deficiencies in behavioral models and provides a general framework for evaluating animal behavior models through embodied interaction.
>
---
#### [replaced 025] Reflection-Based Relative Localization for Cooperative UAV Teams Using Active Markers
- **分类: cs.RO**

- **简介: 该论文属于多无人机相对定位任务，解决环境反射带来的定位模糊问题。通过利用主动标记的反射实现无需先验信息的高精度相对定位。**

- **链接: [https://arxiv.org/pdf/2511.17166](https://arxiv.org/pdf/2511.17166)**

> **作者:** Tim Lakemann; Daniel Bonilla Licea; Viktor Walter; Martin Saska
>
> **摘要:** Reflections of active markers in the environment are a common source of ambiguity in onboard visual relative localization. This work presents a novel approach that exploits these typically unwanted reflections for onboard relative localization in heterogeneous multi-UAV teams. The method operates without prior knowledge of robot size or predefined marker configurations, remains independent of surface properties, and explicitly accounts for uncertainties caused by surface irregularities, including dynamic water surfaces relevant for marine deployments. We validated the approach in both indoor and outdoor experiments, demonstrating reliable operation across varying lighting conditions and achieving greater effective range (above 30 m) and accuracy than state-of-the-art methods. The video is available under the following link: this https URL.
>
---
#### [replaced 026] STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics-Physics Dual System
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于场景生成任务，解决从任务指令生成符合物理规则的桌面场景问题。提出STABLE系统，结合语义与物理修正模块，提升场景的合理性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.16137](https://arxiv.org/pdf/2605.16137)**

> **作者:** Zhen Luo; Yixuan Yang; Xudong Xu; Jinkun Hao; Zhaoyang Lyu; Feng Zheng; Jiangmiao Pang; Yanwei Fu
>
> **备注:** ICML 2026
>
> **摘要:** Generating simulation-ready tabletop scenes from task instructions is an intriguing and promising research direction in the field of Embodied AI. However, existing task-to-scene generation methods rely exclusively on large language models (LLMs) to predict scene layouts, inevitably yielding object collisions or floating due to LLMs' inherent limitations in 3D spatial reasoning. In this paper, we present STABLE, a semantics-physics dual-system tailored for simulation-ready tabletop scene generation. STABLE consists of two complementary modules: (i) a Semantic Reasoner, a fine-tuned LLM trained on a structured tabletop scene dataset to generate coarse layouts from input task instructions, and (ii) a Physics Corrector, a physics-aware flow-based denoising model that outputs pose updates to refine layouts, which ensures the physical plausibility of scenes while preserves semantic alignment with task instructions. STABLE adopts a progressive generation paradigm: by alternating between the Semantic Reasoner and Physics Corrector, it incrementally expands the scene from task-critical objects to background objects. Experiments demonstrate that STABLE successfully generates simulation-ready tabletop scenes that strictly conform to task instructions and significantly enhances the physical validity of scenes over prior art.
>
---
#### [replaced 027] Learning Bilevel Policies over Symbolic World Models for Long-Horizon Planning
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于长期规划任务，旨在解决机器人在复杂环境中执行长周期任务的难题。通过结合低层模仿学习与高层符号抽象，提出双层策略框架，提升规划效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.15975](https://arxiv.org/pdf/2605.15975)**

> **作者:** Dillon Z. Chen; Till Hofmann; Toryn Q. Klassen; Sheila A. McIlraith
>
> **摘要:** We tackle the challenge of building embodied AI agents that can reliably solve long-horizon planning problems. Imitation learning from demonstrations has shown itself to be effective in training robots to solve a diversity of complex tasks requiring fine motor control and manipulation over low-level (LL), continuous environments. Yet, it remains a difficult endeavour to generate long-horizon plans from imitation learning alone. In contrast, high-level (HL), symbolic abstractions facilitate efficient and interpretable long-horizon planning. We propose to combine the strengths of LL imitation learning for manipulation and control, and HL symbolic abstractions for long-horizon planning. We realise this idea via \emph{bilevel policies} of the form $(\pi^{\mathrm{hl}}, \pi^{\mathrm{ll}})$, consisting of a neural policy $\pi^{\mathrm{ll}}$ learned from LL demonstrations, and an HL symbolic policy $\pi^{\mathrm{hl}}$ that is constructed from symbolic abstractions of the LL demonstrations combined with inductive generalisation. We implement these ideas in the BISON system. Experiments on extended MetaWorld benchmarks demonstrate that BISON generalises to long horizons and problems with greater numbers of objects than those solved by VLA and end-to-end methods, and is more time and memory efficient in training and inference. Notably, when ignoring LL execution, BISON's HL policies can solve HL problems with 10,000 relevant objects in under a minute. Project page: this https URL
>
---
#### [replaced 028] Efficient Emotion-Aware Iconic Gesture Prediction for Robot Co-Speech
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于情感驱动的图标手势预测任务，解决机器人共言手势语义不明确的问题。提出轻量级Transformer模型，仅凭文本和情感生成手势，效果优于GPT-4o。**

- **链接: [https://arxiv.org/pdf/2604.11417](https://arxiv.org/pdf/2604.11417)**

> **作者:** Edwin C. Montiel-Vazquez; Christian Arzate Cruz; Stefanos Gkikas; Thomas Kassiotis; Giorgos Giannakakis; Randy Gomez
>
> **摘要:** Co-speech gestures increase engagement and improve speech understanding. Most data-driven robot systems generate rhythmic beat-like motion, yet few integrate semantic emphasis. To address this, we propose a lightweight transformer that derives iconic gesture placement and intensity from text and emotion alone, requiring no audio input at inference time. The model outperforms GPT-4o in both semantic gesture placement classification and intensity regression on the BEAT2 dataset, while remaining computationally compact and suitable for real-time deployment on embodied agents.
>
---
#### [replaced 029] R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出R$^3$L框架，解决3D布局生成中的相对空间推理问题，提升布局的物理可行性和语义一致性。**

- **链接: [https://arxiv.org/pdf/2605.06758](https://arxiv.org/pdf/2605.06758)**

> **作者:** Zhifeng Gu; Yuqi Wang; Bing Wang
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Relative spatial relations provide a compact representation of spatial structure and are fundamental to relative spatial reasoning in 3D layout generation. Recent works leverage Multimodal Large Language Models (MLLMs) to infer such relations, but the inferred relations are often unreliable and are typically handled with post-hoc heuristics. In this paper, we propose R$^3$L, a general framework that improves the reliability and consistency of relative spatial reasoning for 3D layout generation. Our key motivation is that multi-hop reasoning requires repeated reference-frame transformations, which accumulate errors in inferred relations and lead to semantic and metric drift. To mitigate this, we propose invariant spatial decomposition to break coupled relation chains, and consistent spatial imagination to promote self-consistency through an imagine-and-revise loop. We further introduce supportive spatial optimization to ease pose optimization via global-to-local coordinate re-parameterization. Extensive experiments across diverse scene types and instructions demonstrate that R$^3$L produces more physically feasible and semantically consistent layouts. Notably, our analysis shows that resolving frame-induced inconsistencies is crucial for reliable multi-hop relative spatial reasoning. The code is available at this https URL.
>
---
#### [replaced 030] SAMe: A Semantic Anatomy Mapping Engine for Robotic Ultrasound
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SAMe，解决机器人超声扫描初始化问题。通过语义解剖映射，实现自主定位与控制，提升扫描效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.25646](https://arxiv.org/pdf/2604.25646)**

> **作者:** Jing Zhang; Duojie Chen; Wentao Jiang; Zihan Lou; Jianxin Liu; Xinwu Cui; Qinghong Zhao; Bo Du; Christoph F. Dietrich; Dacheng Tao
>
> **备注:** Supplementary information included. Code will be released at this https URL
>
> **摘要:** Robotic ultrasound has advanced local image-driven control, contact regulation, and view optimization, yet current systems lack the anatomical understanding needed to determine what to scan, where to begin, and how to adapt to individual patient anatomy. These gaps make systems still reliant on expert intervention to initiate scanning. Here we present SAMe, a semantic anatomy mapping engine that provides robotic ultrasound with an explicit anatomical prior layer. SAMe addresses scan initiation as a target-to-anatomy-to-action process: it grounds under-specified clinical complaints into structured target organs, instantiates a patient-specific anatomical representation for the grounded targets from a single external body image, and translates this representation into control-facing 6-DoF probe initialization states without any additional registration using preoperative CT or MRI. The anatomical representation maintained by SAMe is explicit, lightweight (single-organ inference in 0.08s), and compatible with downstream control by design. Across semantic grounding, anatomical instantiation, and real-robot evaluation, SAMe shows strong performance across the full initialization pipeline. In real-robot experiments, centroid-based SAMe initialization outperformed the body-keypoint-based heuristic baseline under a budget-matched single-target setting for both liver (86.7% versus 46.7%) and kidney (80.0% versus 73.3%) initialization. Furthermore, The trial-level organ-hit rate reached 97.3% for liver and 83.3% for kidney when multiple candidate targets were available. These results establish an explicit anatomical prior layer that addresses scan initialization and is designed to support broader downstream autonomous scanning pipelines, providing the anatomical foundation for complaint-driven, anatomically informed robotic ultrasonography.
>
---
#### [replaced 031] RE-SAC: Disentangling aleatoric and epistemic risks in bus fleet control: A stable and robust ensemble DRL approach
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，解决公交调度中的不确定性问题。通过分离aleatoric和epistemic风险，提升策略稳定性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.18396](https://arxiv.org/pdf/2603.18396)**

> **作者:** Yifan Zhang; Liang Zheng
>
> **摘要:** Bus holding control is challenging due to stochastic traffic and passenger demand. While deep reinforcement learning (DRL) shows promise, standard actor-critic algorithms suffer from Q-value instability in volatile environments. A key source of this instability is the conflation of two distinct uncertainties: aleatoric uncertainty (irreducible noise) and epistemic uncertainty (data insufficiency). Treating these as a single risk leads to value underestimation in noisy states, causing catastrophic policy collapse. We propose a robust ensemble soft actor-critic (RE-SAC) framework to explicitly disentangle these uncertainties. RE-SAC applies Integral Probability Metric (IPM)-based weight regularization to the critic network to hedge against aleatoric risk, providing a smooth analytical lower bound for the robust Bellman operator without expensive inner-loop perturbations. To address epistemic risk, a diversified Q-ensemble penalizes overconfident value estimates in sparsely covered regions. This dual mechanism prevents the ensemble variance from misidentifying noise as a data gap, a failure mode identified in our ablation study. Experiments in a realistic bidirectional bus corridor simulation demonstrate that RE-SAC achieves the highest cumulative reward (approx. -0.4e6) compared to vanilla SAC (-0.55e6). Mahalanobis rareness analysis confirms that RE-SAC reduces Oracle Q-value estimation error by up to 62% in rare out-of-distribution states (MAE of 1647 vs. 4343), demonstrating superior robustness under high traffic variability.
>
---
#### [replaced 032] COMPASS: Confined-space Manipulation Planning with Active Sensing Strategy
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决受限环境中物体操作难题。提出COMPASS框架，通过主动感知和优化策略提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2509.14787](https://arxiv.org/pdf/2509.14787)**

> **作者:** Qixuan Li; Chen Le; Dongyue Huang; Jincheng Yu; Xinlei Chen
>
> **备注:** Accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** Manipulation in confined and cluttered environments remains a significant challenge due to partial observability and complex configuration spaces. Effective manipulation in such environments requires an intelligent exploration strategy to safely understand the scene and search the target. In this paper, we propose COMPASS, a multi-stage exploration and manipulation framework featuring a manipulation-aware sampling-based planner. First, we reduce collision risks with a near-field awareness scan to build a local collision map. Additionally, we employ a multi-objective utility function to find viewpoints that are both informative and conducive to subsequent manipulation. Moreover, we perform a constrained manipulation optimization strategy to generate manipulation poses that respect obstacle constraints. To systematically evaluate method's performance under these difficulties, we propose a benchmark of confined-space exploration and manipulation containing four level challenging scenarios. Compared to exploration methods designed for other robots and only considering information gain, our framework increases manipulation success rate by 24.25% in simulations. Real-world experiments demonstrate our method's capability for active sensing and manipulation in confined environments.
>
---
