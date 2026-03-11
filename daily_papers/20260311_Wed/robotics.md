# 机器人 cs.RO

- **最新发布 77 篇**

- **更新 40 篇**

## 最新发布

#### [new 001] Caterpillar-Inspired Spring-Based Compressive Continuum Robot for Bristle-based Exploration
- **分类: cs.RO**

- **简介: 该论文属于机器人探索任务，旨在解决受限空间检测难题。设计了一种仿毛虫的柔性机器人，结合肌腱驱动与接触传感器，实现精准定位与表面感知。**

- **链接: [https://arxiv.org/pdf/2603.09745](https://arxiv.org/pdf/2603.09745)**

> **作者:** Zhixian Hu; Yu She; Juan Wachs
>
> **备注:** Accepted by RoboSoft 2026
>
> **摘要:** Exploration of confined spaces, such as pipelines and ducts, remains challenging for conventional rigid robots due to limited space, irregular geometry, and restricted access. Inspired by caterpillar locomotion and sensing, this paper presents a compact spring-based tendon-driven continuum robot that integrates with commercial robotic arms for confined-space inspection. The system combines a mechanically compliant continuum body with a tendon actuation module, enabling coupled bending and axial length change, and uses a constant-curvature kinematic model for positional control. Experiments show a mean position error of 4.32 mm under the proposed model and control pipeline. To extend the system from motion to inspection, we integrate an artificial bristle contact sensor and demonstrate surface perception and confined-space exploration through contact interactions. This compact and compliant design offers a cost-effective upgrade for commercial robots and promises effective exploration in challenging environments.
>
---
#### [new 002] ReTac-ACT: A State-Gated Vision-Tactile Fusion Transformer for Precision Assembly
- **分类: cs.RO**

- **简介: 该论文提出ReTac-ACT，解决精密装配中视觉失效时的定位问题，通过视觉触觉融合提升装配精度。**

- **链接: [https://arxiv.org/pdf/2603.09565](https://arxiv.org/pdf/2603.09565)**

> **作者:** Minchi Ruan; LiangQing Zhou; Hongtong Li; Zongtao Wang; ZhaoMing Lu; Jianwei Zhang; Bin Fang
>
> **摘要:** Precision assembly requires sub-millimeter corrections in contact-rich "last-millimeter" regions where visual feedback fails due to occlusion from the end-effector and workpiece. We present ReTac-ACT (Reconstruction-enhanced Tactile ACT), a vision-tactile imitation learning policy that addresses this challenge through three synergistic mechanisms: (i) bidirectional cross-attention enabling reciprocal visuo-tactile feature enhancement before fusion, (ii) a proprioception-conditioned gating network that dynamically elevates tactile reliance when visual occlusion occurs, and (iii) a tactile reconstruction objective enforcing learning of manipulation-relevant contact information rather than generic visual textures. Evaluated on the standardized NIST Assembly Task Board M1 benchmark, ReTac-ACT achieves 90% peg-in-hole success, substantially outperforming vision-only and generalist baseline methods, and maintains 80% success at industrial-grade 0.1mm clearance. Ablation studies validate that each architectural component is indispensable. The ReTac-ACT codebase and a vision-tactile demonstration dataset covering various clearance levels with both visual and tactile features will be released to support reproducible research.
>
---
#### [new 003] Age-Related Differences in the Perception of Eye-Gaze from a Social Robot
- **分类: cs.RO**

- **简介: 论文研究社交机器人中眼神线索对老年人与年轻人社会感知的影响，属于人机交互任务。旨在解决年龄差异导致的社会感知能力下降问题，通过分析不同年龄群体对机器人眼神的反应，为设计适应性非语言交互提供依据。**

- **链接: [https://arxiv.org/pdf/2603.08810](https://arxiv.org/pdf/2603.08810)**

> **作者:** Lucas Morillo-Mendez; Martien G.S. Schrooten; Oscar Martinez Mozos
>
> **备注:** This is the pre-print version. Final publication available at this https URL
>
> **摘要:** There is an increasing interest in social robots assisting older adults during daily life tasks. In this context, non-verbal cues such as deictic gaze are important in natural communication in human-robot interaction. However, the sensibility to deictic-gaze declines naturally with age and results in a reduction in social perception. Therefore, this work explores the benefits of deictic gaze from social robots assisting older adults during daily life tasks, and how age-related differences may influence their social perception in contrast to younger populations. This may help on the design of adaptive age-related non-verbal cues in the Human-Robot Interaction context.
>
---
#### [new 004] ImpedanceDiffusion: Diffusion-Based Global Path Planning for UAV Swarm Navigation with Generative Impedance Control
- **分类: cs.RO**

- **简介: 该论文提出ImpedanceDiffusion框架，解决无人机群在复杂室内环境中的安全导航问题。通过扩散模型生成路径，结合APF和阻抗控制实现高效避障与自适应运动。**

- **链接: [https://arxiv.org/pdf/2603.09031](https://arxiv.org/pdf/2603.09031)**

> **作者:** Faryal Batool; Yasheerah Yaqoot; Muhammad Ahsan Mustafa; Roohan Ahmed Khan; Aleksey Fedoseev; Dzmitry Tsetserukou
>
> **备注:** This is paper is under review
>
> **摘要:** Safe swarm navigation in cluttered indoor environment requires long-horizon planning, reactive obstacle avoidance, and adaptive compliance. We propose ImpedanceDiffusion, a hierarchical framework that leverages image-conditioned diffusion-based global path planning with Artificial Potential Field (APF) tracking and semantic-aware variable impedance control for aerial drone swarms. The diffusion model generates geometric global trajectories directly from RGB images without explicit map construction. These trajectories are tracked by an APF-based reactive layer, while a VLM-RAG module performs semantic obstacle classification with 90% retrieval accuracy to adapt impedance parameters for mixed obstacle environments during execution. Two diffusion planners are evaluated: (i) a top-view long-horizon planner using single-pass inference and (ii) a first-person-view (FPV) short-horizon planner deployed via a two-stage inference pipeline. Both planners achieve a 100% trajectory generation rate across twenty static and dynamic experimental configurations and are validated via zero-shot sim-to-real deployment on Crazyflie 2.1 drones through the hierarchical APF-impedance control stack. The top-view planner produces smoother trajectories that yield conservative tracking speeds of 1.0-1.2 m/s near hard obstacles and 0.6-1.0 m/s near soft obstacles. In contrast, the FPV planner generates trajectories with greater local clearance and typically higher speeds, reaching 1.4-2.0 m/s near hard obstacles and up to 1.6 m/s near soft obstacles. Across 20 experimental configurations (100 total runs), the framework achieved a 92% success rate while maintaining stable impedance-based formation control with bounded oscillations and no in-flight collisions, demonstrating reliable and adaptive swarm navigation in cluttered indoor environments.
>
---
#### [new 005] 3D UAV Trajectory Estimation and Classification from Internet Videos via Language Model
- **分类: cs.RO**

- **简介: 该论文属于3D UAV轨迹估计任务，解决无标注数据下轨迹和类别信息提取问题。通过语言模型和视觉推理，从网络视频中自动生成轨迹并优化其一致性。**

- **链接: [https://arxiv.org/pdf/2603.09070](https://arxiv.org/pdf/2603.09070)**

> **作者:** Haoxiang Lei; Daotong Wang; Shenghai Yuan; Jianbo Su
>
> **摘要:** Reliable 3D trajectory estimation of unmanned aerial vehicles (UAVs) is a fundamental requirement for anti-UAV systems, yet the acquisition of large-scale and accurately annotated trajectory data remains prohibitively expensive. In this work, we present a novel framework that derives UAV 3D trajectories and category information directly from Internet-scale UAV videos, without relying on manual annotations. First, language-driven data acquisition is employed to autonomously discover and collect UAV-related videos, while vision-language reasoning progressively filters task-relevant segments. Second, a training-free cross-modal label generation module is introduced to infer 3D trajectory hypotheses and UAV type cues. Third, a physics-informed refinement process is designed to impose temporal smoothness and kinematic consistency on the estimated trajectories. The resulting video clips and trajectory annotations can be readily utilized for downstream anti-UAV tasks. To assess effectiveness and generalization, we conduct zero-shot transfer experiments on a public, well-annotated 3D UAV benchmark. Results reveal a clear data scaling behavior: as the amount of online video data increases, zero-shot transfer performance on the target dataset improves consistently, without any target-domain training. The proposed method closely approaches the current state-of-the-art, highlighting its robustness and applicability to real-world anti-UAV scenarios. Code and datasets will be released upon acceptance.
>
---
#### [new 006] SPAN-Nav: Generalized Spatial Awareness for Versatile Vision-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决复杂环境中空间感知不足的问题。提出SPAN-Nav模型，通过RGB视频流实现通用3D空间感知，并利用单个空间标记提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.09163](https://arxiv.org/pdf/2603.09163)**

> **作者:** Jiahang Liu; Tianyu Xu; Jiawei Chen; Lu Yue; Jiazhao Zhang; Zhiyong Wang; Minghan Li; Qisheng Zhao; Anqi Li; Qi Su; Zhizheng Zhang; He Wang
>
> **摘要:** Recent embodied navigation approaches leveraging Vision-Language Models (VLMs) demonstrate strong generalization in versatile Vision-Language Navigation (VLN). However, reliable path planning in complex environments remains challenging due to insufficient spatial awareness. In this work, we introduce SPAN-Nav, an end-to-end foundation model designed to infuse embodied navigation with universal 3D spatial awareness using RGB video streams. SPAN-Nav extracts spatial priors across diverse scenes through an occupancy prediction task on extensive indoor and outdoor environments. To mitigate the computational burden, we introduce a compact representation for spatial priors, finding that a single token is sufficient to encapsulate the coarse-grained cues essential for navigation tasks. Furthermore, inspired by the Chain-of-Thought (CoT) mechanism, SPAN-Nav utilizes this single spatial token to explicitly inject spatial cues into action reasoning through an end-to end framework. Leveraging multi-task co-training, SPAN-Nav captures task-adaptive cues from generalized spatial priors, enabling robust spatial awareness to generalize even to the task lacking explicit spatial supervision. To support comprehensive spatial learning, we present a massive dataset of 4.2 million occupancy annotations that covers both indoor and outdoor scenes across multi-type navigation tasks. SPAN-Nav achieves state-of-the-art performance across three benchmarks spanning diverse scenarios and varied navigation tasks. Finally, real-world experiments validate the robust generalization and practical reliability of our approach across complex physical scenarios.
>
---
#### [new 007] PM-Nav: Priori-Map Guided Embodied Navigation in Functional Buildings
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人导航任务，旨在解决功能建筑中因特征相似导致的导航难题。通过构建先验地图引导的导航框架PM-Nav，提升导航精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.09113](https://arxiv.org/pdf/2603.09113)**

> **作者:** Jiang Gao; Xiangyu Dong; Haozhou Li; Haoran Zhao; Yaoming Zhou; Xiaoguang Ma
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** Existing language-driven embodied navigation paradigms face challenges in functional buildings (FBs) with highly similar features, as they lack the ability to effectively utilize priori spatial knowledge. To tackle this issue, we propose a Priori-Map Guided Embodied Navigation (PM-Nav), wherein environmental maps are transformed into navigation-friendly semantic priori-maps, a hierarchical chain-of-thought prompt template with an annotation priori-map is designed to enable precise path planning, and a multi-model collaborative action output mechanism is built to accomplish positioning decisions and execution control for navigation planning. Comprehensive tests using a home-made FB dataset show that the PM-Nav obtains average improvements of 511\% and 1175\%, and 650\% and 400\% over the SG-Nav and the InstructNav in simulation and real-world, respectively. These tremendous boosts elucidate the great potential of using the PM-Nav as a backbone navigation framework for FBs.
>
---
#### [new 008] Emerging Extrinsic Dexterity in Cluttered Scenes via Dynamics-aware Policy Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，解决 cluttered 场景下的 extrinsic dexterity 问题。通过 DAPL 框架学习接触动力学，提升非抓取操作的性能。**

- **链接: [https://arxiv.org/pdf/2603.09882](https://arxiv.org/pdf/2603.09882)**

> **作者:** Yixin Zheng; Jiangran Lyu; Yifan Zhang; Jiayi Chen; Mi Yan; Yuntian Deng; Xuesong Shi; Xiaoguang Zhao; Yizhou Wang; Zhizheng Zhang; He Wang
>
> **备注:** Project Page: this https URL
>
> **摘要:** Extrinsic dexterity leverages environmental contact to overcome the limitations of prehensile manipulation. However, achieving such dexterity in cluttered scenes remains challenging and underexplored, as it requires selectively exploiting contact among multiple interacting objects with inherently coupled dynamics. Existing approaches lack explicit modeling of such complex dynamics and therefore fall short in non-prehensile manipulation in cluttered environments, which in turn limits their practical applicability in real-world environments. In this paper, we introduce a Dynamics-Aware Policy Learning (DAPL) framework that can facilitate policy learning with a learned representation of contact-induced object dynamics in cluttered environments. This representation is learned through explicit world modeling and used to condition reinforcement learning, enabling extrinsic dexterity to emerge without hand-crafted contact heuristics or complex reward shaping. We evaluate our approach in both simulation and the real world. Our method outperforms prehensile manipulation, human teleoperation, and prior representation-based policies by over 25% in success rate on unseen simulated cluttered scenes with varying densities. The real-world success rate reaches around 50% across 10 cluttered scenes, while a practical grocery deployment further demonstrates robust sim-to-real transfer and applicability.
>
---
#### [new 009] Vision-Augmented On-Track System Identification for Autonomous Racing via Attention-Based Priors and Iterative Neural Correction
- **分类: cs.RO**

- **简介: 该论文属于自主赛车系统辨识任务，旨在解决轮胎动力学实时识别难题。通过融合视觉信息与神经网络，提升参数估计精度与收敛速度。**

- **链接: [https://arxiv.org/pdf/2603.09399](https://arxiv.org/pdf/2603.09399)**

> **作者:** Zhiping Wu; Cheng Hu; Yiqin Wang; Lei Xie; Hongye Su
>
> **摘要:** Operating autonomous vehicles at the absolute limits of handling requires precise, real-time identification of highly non-linear tire dynamics. However, traditional online optimization methods suffer from "cold-start" initialization failures and struggle to model high-frequency transient dynamics. To address these bottlenecks, this paper proposes a novel vision-augmented, iterative system identification framework. First, a lightweight CNN (MobileNetV3) translates visual road textures into a continuous heuristic friction prior, providing a robust "warm-start" for parameter optimization. Next, a S4 model captures complex temporal dynamic residuals, circumventing the memory and latency limitations of traditional MLPs and RNNs. Finally, a derivative-free Nelder-Mead algorithm iteratively extracts physically interpretable Pacejka tire parameters via a hybrid virtual simulation. Co-simulation in CarSim demonstrates that the lightweight vision backbone reduces friction estimation error by 76.1 using 85 fewer FLOPs, accelerating cold-start convergence by 71.4. Furthermore, the S4-augmented framework improves parameter extraction accuracy and decreases lateral force RMSE by over 60 by effectively capturing complex vehicle dynamics, demonstrating superior performance compared to conventional neural architectures.
>
---
#### [new 010] StyleVLA: Driving Style-Aware Vision Language Action Model for Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决生成多样化且物理可行的驾驶行为问题。提出StyleVLA模型，结合视觉语言与物理约束，提升驾驶风格适应性与轨迹可行性。**

- **链接: [https://arxiv.org/pdf/2603.09482](https://arxiv.org/pdf/2603.09482)**

> **作者:** Yuan Gao; Dengyuan Hua; Mattia Piccinini; Finn Rasmus Schäfer; Korbinian Moller; Lin Li; Johannes Betz
>
> **备注:** 8 pages
>
> **摘要:** Vision Language Models (VLMs) bridge visual perception and linguistic reasoning. In Autonomous Driving (AD), this synergy has enabled Vision Language Action (VLA) models, which translate high-level multimodal understanding into driving behaviors, typically represented as future trajectories. However, existing VLA models mainly generate generic collision-free trajectories. Beyond collision avoidance, adapting to diverse driving styles (e.g., sporty, comfortable) is essential for personalized driving. Moreover, many methods treat trajectory generation as naive token prediction, which can produce kinematically infeasible actions. To address these limitations, we present StyleVLA, a physics-informed VLA framework for generating diverse and physically plausible driving behaviors. We introduce a hybrid loss that combines a kinematic consistency constraint with a continuous regression head to improve trajectory feasibility. To train StyleVLA, built on Qwen3-VL-4B, we construct a large-scale instruction dataset with over 1.2k scenarios, 76k Bird's Eye View (BEV) samples, and 42k First Person View (FPV) samples, with ground-truth trajectories for five driving styles and natural-language instructions. Experiments show that our 4B-parameter StyleVLA significantly outperforms proprietary models (e.g., Gemini-3-Pro) and state-of-the-art VLA models. Using a composite driving score measuring success rate, physical feasibility, and style adherence, StyleVLA achieves 0.55 on BEV and 0.51 on FPV, versus 0.32 and 0.35 for Gemini-3-Pro. These results show that a specialized, physics-informed, lightweight model can surpass closed-source models on domain-specific tasks.
>
---
#### [new 011] Robust Cooperative Localization in Featureless Environments: A Comparative Study of DCL, StCL, CCL, CI, and Standard-CL
- **分类: cs.RO**

- **简介: 论文研究多机器人协同定位问题，比较五种CL方法在无特征环境中的性能，分析其准确性、稳定性和一致性，为实际应用提供选择依据。**

- **链接: [https://arxiv.org/pdf/2603.09886](https://arxiv.org/pdf/2603.09886)**

> **作者:** Nivand Khosravi; Meysam Basiri; Rodrigo Ventura
>
> **备注:** Presented at the 2026 12th International Conference on Automation, Robotics and Applications (ICARA); to be published in IEEE conference proceedings
>
> **摘要:** Cooperative localization (CL) enables accurate position estimation in multi-robot systems operating in GPS-denied environments. This paper presents a comparative study of five CL approaches: Centralized Cooperative Localization (CCL), Decentralized Cooperative Localization (DCL), Sequential Cooperative Localization (StCL), Covariance Intersection (CI), and Standard Cooperative Localization (Standard-CL). All methods are implemented in ROS and evaluated through Monte Carlo simulations under two conditions: weak data association and robust detection. Our analysis reveals fundamental trade-offs among the methods. StCL and Standard-CL achieve the lowest position errors but exhibit severe filter inconsistency, making them unsuitable for safety-critical applications. DCL demonstrates remarkable stability under challenging conditions due to its measurement stride mechanism, which provides implicit regularization against outliers. CI emerges as the most balanced approach, achieving near-optimal consistency while maintaining competitive accuracy. CCL provides theoretically optimal estimation but shows sensitivity to measurement outliers. These findings offer practical guidance for selecting CL algorithms based on application requirements.
>
---
#### [new 012] Quality over Quantity: Demonstration Curation via Influence Functions for Data-Centric Robot Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人学习任务，旨在解决示范数据质量低下问题。通过影响函数筛选高质量数据，提升策略性能。**

- **链接: [https://arxiv.org/pdf/2603.09056](https://arxiv.org/pdf/2603.09056)**

> **作者:** Haeone Lee; Taywon Min; Junsu Kim; Sinjae Kang; Fangchen Liu; Lerrel Pinto; Kimin Lee
>
> **备注:** Accepted to ICRA 2026, 8 pages
>
> **摘要:** Learning from demonstrations has emerged as a promising paradigm for end-to-end robot control, particularly when scaled to diverse and large datasets. However, the quality of demonstration data, often collected through human teleoperation, remains a critical bottleneck for effective data-driven robot learning. Human errors, operational constraints, and teleoperator variability introduce noise and suboptimal behaviors, making data curation essential yet largely manual and heuristic-driven. In this work, we propose Quality over Quantity (QoQ), a grounded and systematic approach to identifying high-quality data by defining data quality as the contribution of each training sample to reducing loss on validation demonstrations. To efficiently estimate this contribution, we leverage influence functions, which quantify the impact of individual training samples on model performance. We further introduce two key techniques to adapt influence functions for robot demonstrations: (i) using maximum influence across validation samples to capture the most relevant state-action pairs, and (ii) aggregating influence scores of state-action pairs within the same trajectory to reduce noise and improve data coverage. Experiments in both simulated and real-world settings show that QoQ consistently improves policy performances over prior data selection methods.
>
---
#### [new 013] NS-VLA: Towards Neuro-Symbolic Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出NS-VLA框架，解决机器人操作中指令理解与动作生成的问题，通过符号编码和强化学习提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.09542](https://arxiv.org/pdf/2603.09542)**

> **作者:** Ziyue Zhu; Shangyang Wu; Shuai Zhao; Zhiqiu Zhao; Shengjie Li; Yi Wang; Fang Li; Haoran Luo
>
> **摘要:** Vision-Language-Action (VLA) models are formulated to ground instructions in visual context and generate action sequences for robotic manipulation. Despite recent progress, VLA models still face challenges in learning related and reusable primitives, reducing reliance on large-scale data and complex architectures, and enabling exploration beyond demonstrations. To address these challenges, we propose a novel Neuro-Symbolic Vision-Language-Action (NS-VLA) framework via online reinforcement learning (RL). It introduces a symbolic encoder to embedding vision and language features and extract structured primitives, utilizes a symbolic solver for data-efficient action sequencing, and leverages online RL to optimize generation via expansive exploration. Experiments on robotic manipulation benchmarks demonstrate that NS-VLA outperforms previous methods in both one-shot training and data-perturbed settings, while simultaneously exhibiting superior zero-shot generalizability, high data efficiency and expanded exploration space. Our code is available.
>
---
#### [new 014] CORAL: Scalable Multi-Task Robot Learning via LoRA Experts
- **分类: cs.RO**

- **简介: 该论文提出CORAL框架，解决多任务机器人学习中的任务干扰问题。通过LoRA专家实现参数隔离，提升任务性能与扩展性。属于多任务机器人学习领域。**

- **链接: [https://arxiv.org/pdf/2603.09298](https://arxiv.org/pdf/2603.09298)**

> **作者:** Yuankai Luo; Woping Chen; Tong Liang; Zhenguo Li
>
> **摘要:** Deploying Vision-Language-Action (VLA) models in real-world robotics exposes a core multi-task learning challenge: reconciling task interference in multi-task robotic learning. When multiple tasks are jointly fine-tuned in a single stage, gradients from different tasks can conflict, causing negative transfer and reducing per-task performance. Yet maintaining a separate full checkpoint per task is often storage- and deployment-prohibitive. To address this dilemma, we present CORAL, a backbone- and embodiment-agnostic framework designed primarily to mitigate multi-task interference while remaining naturally extensible to a continuous stream of new tasks. CORAL freezes a single pre-trained VLA backbone and attaches one lightweight Low-Rank Adaptation (LoRA) expert per task; at runtime, a dynamic inference engine (the CORAL Manager) routes language instructions to the appropriate expert and swaps experts on the fly with zero inference overhead. This strict parameter isolation avoids complex gating networks and prevents parameter-level cross-task interference by construction; as an added capability, it also enables sequentially introducing new tasks without parameter overwriting caused by catastrophic forgetting. We validate CORAL on a real-world Galaxea R1 dual-arm mobile manipulator and three simulation benchmarks (LIBERO, WidowX, Google Robot), where CORAL overcomes fine-grained instructional ambiguity and substantially outperforms joint training, yielding a practical and scalable system for lifelong multi-task robot learning. Website: this https URL
>
---
#### [new 015] Robust Spatiotemporal Motion Planning for Multi-Agent Autonomous Racing via Topological Gap Identification and Accelerated MPC
- **分类: cs.RO**

- **简介: 该论文属于多智能体自动驾驶竞速任务，解决高速场景下的鲁棒时空规划问题。通过拓扑间隙识别与加速MPC方法，提升超车效率与计算性能。**

- **链接: [https://arxiv.org/pdf/2603.09188](https://arxiv.org/pdf/2603.09188)**

> **作者:** Mingyi Zhang; Cheng Hu; Yiqin Wang; Haotong Qin; Hongye Su; Lei Xie
>
> **摘要:** High-speed multi-agent autonomous racing demands robust spatiotemporal planning and precise control under strict computational limits. Current methods often oversimplify interactions or abandon strict kinematic constraints. We resolve this by proposing a Topological Gap Identification and Accelerated MPC framework. By predicting opponent behaviors via SGPs, our method constructs dynamic occupancy corridors to robustly select optimal overtaking gaps. We ensure strict kinematic feasibility using a Linear Time-Varying MPC powered by a customized Pseudo-Transient Continuation (PTC) solver for high-frequency execution. Experimental results on the F1TENTH platform show that our method significantly outperforms state-of-the-art baselines: it reduces total maneuver time by 51.6% in sequential scenarios, consistently maintains an overtaking success rate exceeding 81% in dense bottlenecks, and lowers average computational latency by 20.3%, pushing the boundaries of safe and high-speed autonomous racing.
>
---
#### [new 016] Let's Reward Step-by-Step: Step-Aware Contrastive Alignment for Vision-Language Navigation in Continuous Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，解决连续环境中代理的长期决策与错误恢复问题。提出SACA框架，通过步骤感知对比对齐提取密集监督信号，提升训练稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2603.09740](https://arxiv.org/pdf/2603.09740)**

> **作者:** Haoyuan Li; Rui Liu; Hehe Fan; Yi Yang
>
> **备注:** 28 pages, 10 figures
>
> **摘要:** Vision-Language Navigation in Continuous Environments (VLN-CE) requires agents to learn complex reasoning from long-horizon human interactions. While Multi-modal Large Language Models (MLLMs) have driven recent progress, current training paradigms struggle to balance generalization capability, error recovery and training stability. Specifically, (i) policies derived from SFT suffer from compounding errors, struggling to recover from out-of-distribution states, and (ii) Reinforcement Fine-Tuning (RFT) methods e.g. GRPO are bottlenecked by sparse outcome rewards. Their binary feedback fails to assign credit to individual steps, leading to gradient signal collapse in failure dominant batches. To address these challenges, we introduce Step-Aware Contrastive Alignment (SACA), a framework designed to extract dense supervision from imperfect trajectories. At its core, the Perception-Grounded Step-Aware auditor evaluates progress step-by-step, disentangling failed trajectories into valid prefixes and exact divergence points. Leveraging these signals, Scenario-Conditioned Group Construction mechanism dynamically routes batches to specialized resampling and optimization strategies. Extensive experiments on VLN-CE benchmarks demonstrate that SACA achieves state-of-the-art performance.
>
---
#### [new 017] Trajectory Optimization for Self-Wrap-Aware Cable-Towed Planar Object Manipulation under Implicit Tension Constraints
- **分类: cs.RO**

- **简介: 该论文研究电缆牵引平面物体的轨迹优化问题，解决如何在考虑自缠绕和张力约束下的有效力传递问题。通过构建不同松弛模型实现路径与张力的耦合优化。**

- **链接: [https://arxiv.org/pdf/2603.09557](https://arxiv.org/pdf/2603.09557)**

> **作者:** Yu Li; Amin Fakhari; Hamid Sadeghian
>
> **摘要:** Cable/rope elements are pervasive in deformable-object manipulation, often serving as a deformable force-transmission medium whose routing and contact determine how wrenches are delivered. In cable-towed manipulation, transmission is unilateral and hybrid: the tether can pull only when taut and becomes force-free when slack; in practice, the tether may also contact the object boundary and self-wrap around edges, which is not merely collision avoidance but a change of the wrench transmission channel by shifting the effective application point and moment arm, thereby coupling routing geometry with rigid-body motion and tensioning. We formulate self-wrap towing as a routing-aware, tensioning-implicit trajectory optimization (TITO) problem that couples (i) a tensioning-implicit taut/slack constraint and (ii) routing-conditioned transmission maps for effective length and wrench, and we build a relaxation hierarchy from a strict mode-conditioned reference to three tractable relaxations: Full-Mode Relaxation (FMR), Binary-Mode Relaxation (BMR), and Implicit-Mode Relaxation (IMR). Across planar towing tasks, we find that making routing an explicit decision often yields conservative solutions that stay near switching boundaries, whereas IMR induces self-wrap through state evolution and exploits the redirected torque channel whenever turning requires it.
>
---
#### [new 018] Walking on Rough Terrain with Any Number of Legs
- **分类: cs.RO**

- **简介: 该论文属于多足机器人控制任务，旨在解决复杂地形行走问题。提出一种轻量级控制架构，通过分段状态机实现稳定运动，适用于6至16足机器人。**

- **链接: [https://arxiv.org/pdf/2603.09147](https://arxiv.org/pdf/2603.09147)**

> **作者:** Zhuoyang Chen; Xinyuan Wang; Shai Revzen
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Robotics would gain by replicating the remarkable agility of arthropods in navigating complex environments. Here we consider the control of multi-legged systems which have 6 or more legs. Current multi-legged control strategies in robots include large black-box machine learning models, Central Pattern Generator (CPG) networks, and open-loop feed-forward control with stability arising from mechanics. Here we present a multi-legged control architecture for rough terrain using a segmental robot with 3 actuators for every 2 legs, which we validated in simulation for robots with 6 to 16 legs. Segments have identical state machines, and each segment also receives input from the segment in front of it. Our design bridges the gap between WalkNet-like event cascade controllers and CPG-based controllers: it tightly couples to the ground when contact is present, but produces fictive locomotion when ground contact is missing. The approach may be useful as an adaptive and computationally lightweight controller for multi-legged robots, and as a baseline capability for scaffolding the learning of machine learning controllers.
>
---
#### [new 019] Latent World Models for Automated Driving: A Unified Taxonomy, Evaluation Framework, and Open Challenges
- **分类: cs.RO; cs.AI; cs.LG; cs.MA; eess.SY**

- **简介: 该论文属于自动驾驶领域，旨在解决复杂环境下的决策与规划问题。提出统一的潜在空间框架，整合生成模型与视觉-语言-动作系统，提升模拟、预测与决策能力。**

- **链接: [https://arxiv.org/pdf/2603.09086](https://arxiv.org/pdf/2603.09086)**

> **作者:** Rongxiang Zeng; Yongqi Dong
>
> **备注:** 17 pages, 6 figures, under review by IEEE Transactions on Intelligent Transportation Systems (IEEE-T-ITS)
>
> **摘要:** Emerging generative world models and vision-language-action (VLA) systems are rapidly reshaping automated driving by enabling scalable simulation, long-horizon forecasting, and capability-rich decision making. Across these directions, latent representations serve as the central computational substrate: they compress high-dimensional multi-sensor observations, enable temporally coherent rollouts, and provide interfaces for planning, reasoning, and controllable generation. This paper proposes a unifying latent-space framework that synthesizes recent progress in world models for automated driving. The framework organizes the design space by the target and form of latent representations (latent worlds, latent actions, latent generators; continuous states, discrete tokens, and hybrids) and by structural priors for geometry, topology, and semantics. Building on this taxonomy, the paper articulates five cross-cutting internal mechanics (i.e, structural isomorphism, long-horizon temporal stability, semantic and reasoning alignment, value-aligned objectives and post-training, as well as adaptive computation and deliberation) and connects these design choices to robustness, generalization, and deployability. The work also proposes concrete evaluation prescriptions, including a closed-loop metric suite and a resource-aware deliberation cost, designed to reduce the open-loop / closed-loop mismatch. Finally, the paper identifies actionable research directions toward advancing latent world model for decision-ready, verifiable, and resource-efficient automated driving.
>
---
#### [new 020] Beyond Short-Horizon: VQ-Memory for Robust Long-Horizon Manipulation in Non-Markovian Simulation Benchmarks
- **分类: cs.RO**

- **简介: 该论文属于机器人长期操作任务，解决非马尔可夫环境下复杂操作问题。提出RuleSafe基准和VQ-Memory方法，提升长期规划与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.09513](https://arxiv.org/pdf/2603.09513)**

> **作者:** Wang Honghui; Jing Zhi; Ao Jicong; Song Shiji; Li Xuelong; Huang Gao; Bai Chenjia
>
> **备注:** 9 pages
>
> **摘要:** The high cost of collecting real-robot data has made robotic simulation a scalable platform for both evaluation and data generation. Yet most existing benchmarks concentrate on simple manipulation tasks such as pick-and-place, failing to capture the non-Markovian characteristics of real-world tasks and the complexity of articulated object interactions. To address this limitation, we present RuleSafe, a new articulated manipulation benchmark built upon a scalable LLM-aided simulation framework. RuleSafe features safes with diverse unlocking mechanisms, such as key locks, password locks, and logic locks, which require different multi-stage reasoning and manipulation strategies. These LLM-generated rules produce non-Markovian and long-horizon tasks that require temporal modeling and memory-based reasoning. We further propose VQ-Memory, a compact and structured temporal representation that uses vector-quantized variational autoencoders (VQ-VAEs) to encode past proprioceptive states into discrete latent tokens. This representation filters low-level noise while preserving high-level task-phase context, providing lightweight yet robust temporal cues that are compatible with existing Vision-Language-Action models (VLA). Extensive experiments on state-of-the-art VLA models and diffusion policies show that VQ-Memory consistently improves long-horizon planning, enhances generalization to unseen configurations, and enables more efficient manipulation with reduced computational cost. Project page: this http URL
>
---
#### [new 021] APPLV: Adaptive Planner Parameter Learning from Vision-Language-Action Model
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人自主导航任务，解决传统方法需调参与端到端方法控制不足的问题。提出APPLV模型，通过视觉-语言-动作模型预测规划参数，提升导航性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.08862](https://arxiv.org/pdf/2603.08862)**

> **作者:** Yuanjie Lu; Beichen Wang; Zhengqi Wu; Yang Li; Xiaomin Lin; Chengzhi Mao; Xuesu Xiao
>
> **摘要:** Autonomous navigation in highly constrained environments remains challenging for mobile robots. Classical navigation approaches offer safety assurances but require environment-specific parameter tuning; end-to-end learning bypasses parameter tuning but struggles with precise control in constrained spaces. To this end, recent robot learning approaches automate parameter tuning while retaining classical systems' safety, yet still face challenges in generalizing to unseen environments. Recently, Vision-Language-Action (VLA) models have shown promise by leveraging foundation models' scene understanding capabilities, but still struggle with precise control and inference latency in navigation tasks. In this paper, we propose Adaptive Planner Parameter Learning from Vision-Language-Action Model (\textsc{applv}). Unlike traditional VLA models that directly output actions, \textsc{applv} leverages pre-trained vision-language models with a regression head to predict planner parameters that configure classical planners. We develop two training strategies: supervised learning fine-tuning from collected navigation trajectories and reinforcement learning fine-tuning to further optimize navigation performance. We evaluate \textsc{applv} across multiple motion planners on the simulated Benchmark Autonomous Robot Navigation (BARN) dataset and in physical robot experiments. Results demonstrate that \textsc{applv} outperforms existing methods in both navigation performance and generalization to unseen environments.
>
---
#### [new 022] DexHiL: A Human-in-the-Loop Framework for Vision-Language-Action Model Post-Training in Dexterous Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出DexHiL框架，解决灵巧操作中VLA模型的后训练问题。通过人机协同提升模型可靠性与适应性，实现手臂与手指的联合干预。**

- **链接: [https://arxiv.org/pdf/2603.09121](https://arxiv.org/pdf/2603.09121)**

> **作者:** Yifan Han; Zhongxi Chen; Yuxuan Zhao; Congsheng Xu; Yanming Shao; Yichuan Peng; Yao Mu; Wenzhao Lian
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** While Vision-Language-Action (VLA) models have demonstrated promising generalization capabilities in robotic manipulation, deploying them on specific and complex downstream tasks still demands effective post-training. In parallel, Human-in-the-Loop (HiL) learning has proven to be a powerful mechanism for refining robot policies. However, extending this paradigm to dexterous manipulation remains challenging: multi-finger control is high-dimensional, contact-intensive, and exhibits execution distributions that differ markedly from standard arm motions, leaving existing dexterous VLA systems limited in reliability and adaptability. We present DexHiL, the first integrated arm-hand human-in-the-loop framework for dexterous VLA models, enabling coordinated interventions over the arm and the dexterous hand within a single system. DexHiL introduces an intervention-aware data sampling strategy that prioritizes corrective segments for post-training, alongside a lightweight teleoperation interface that supports instantaneous human corrections during execution. Real-robot experiments demonstrate that DexHiL serves as an effective post-training framework, yielding a substantial performance leap, outperforming standard offline-only fine-tuning baselines by an average of 25% in success rates across distinct tasks. Project page: this https URL
>
---
#### [new 023] WESPR: Wind-adaptive Energy-Efficient Safe Perception & Planning for Robust Flight with Quadrotors
- **分类: cs.RO**

- **简介: 该论文提出WESPR框架，解决无人机在复杂风场中的安全导航问题。通过结合环境几何与风况数据，实现快速路径规划与控制调整，提升飞行稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2603.09194](https://arxiv.org/pdf/2603.09194)**

> **作者:** Khuzema Habib; Pranav Deshakulkarni Manjunath; Kasra Torshizi; Troi Williams; Pratap Tokekar
>
> **备注:** 8 pages, 9 Figures
>
> **摘要:** Local wind conditions strongly influence drone performance: headwinds increase flight time, crosswinds and wind shear hinder agility in cluttered spaces, while tailwinds reduce travel time. Although adaptive controllers can mitigate turbulence, they remain unaware of the surrounding geometry that generates it, preventing proactive avoidance. Existing methods that model how wind interacts with the environment typically rely on computationally expensive fluid dynamics simulations, limiting real-time adaptation to new environments and conditions. To bridge this gap, we present WESPR, a fast framework that predicts how environmental geometry affects local wind conditions, enabling proactive path planning and control adaptation. Our lightweight pipeline integrates geometric perception and local weather data to estimate wind fields, compute cost-efficient paths, and adjust control strategies-all within 10 seconds. We validate WESPR on a Crazyflie drone navigating turbulent obstacle courses. Our results show a 12.5-58.7% reduction in maximum trajectory deviation and a 24.6% improvement in stability compared to a wind-agnostic adaptive controller.
>
---
#### [new 024] Improving through Interaction: Searching Behavioral Representation Spaces with CMA-ES-IG
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于人机交互任务，旨在提升机器人对用户偏好的适应能力。针对现有方法忽视用户体验的问题，提出CMA-ES-IG算法，优化行为排序过程，提高学习效率与用户满意度。**

- **链接: [https://arxiv.org/pdf/2603.09011](https://arxiv.org/pdf/2603.09011)**

> **作者:** Nathaniel Dennler; Zhonghao Shi; Yiran Tao; Andreea Bobu; Stefanos Nikolaidis; Maja Matarić
>
> **备注:** Under submission to IJRR
>
> **摘要:** Robots that interact with humans must adapt to individual users' preferences to operate effectively in human-centered environments. An intuitive and effective technique to learn non-expert users' preferences is through rankings of robot behaviors, e.g., trajectories, gestures, or voices. Existing techniques primarily focus on generating queries that optimize preference learning outcomes, such as sample efficiency or final preference estimation accuracy. However, the focus on outcome overlooks key user expectations in the process of providing these rankings, which can negatively impact users' adoption of robotic systems. This work proposes the Covariance Matrix Adaptation Evolution Strategies with Information Gain (CMA-ES-IG) algorithm. CMA-ES-IG explicitly incorporates user experience considerations into the preference learning process by suggesting perceptually distinct and informative trajectories for users to rank. We demonstrate these benefits through both simulated studies and real-robot experiments. CMA-ES-IG, compared to state-of-the-art alternatives, (1) scales more effectively to higher-dimensional preference spaces, (2) maintains computational tractability for high-dimensional problems, (3) is robust to noisy or inconsistent user feedback, and (4) is preferred by non-expert users in identifying their preferred robot behaviors. This project's code is available at this http URL
>
---
#### [new 025] Provably Safe Trajectory Generation for Manipulators Under Motion and Environmental Uncertainties
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决不确定环境下机械臂安全轨迹生成问题。提出融合深度随机Koopman模型与SOS验证的框架，提升碰撞风险认证效率与安全性。**

- **链接: [https://arxiv.org/pdf/2603.09083](https://arxiv.org/pdf/2603.09083)**

> **作者:** Fei Meng; Zijiang Yang; Xinyu Mao; Haobo Liang; Max Q.-H. Meng
>
> **摘要:** Robot manipulators operating in uncertain and non-convex environments present significant challenges for safe and optimal motion planning. Existing methods often struggle to provide efficient and formally certified collision risk guarantees, particularly when dealing with complex geometries and non-Gaussian uncertainties. This article proposes a novel risk-bounded motion planning framework to address this unmet need. Our approach integrates a rigid manipulator deep stochastic Koopman operator (RM-DeSKO) model to robustly predict the robot's state distribution under motion uncertainty. We then introduce an efficient, hierarchical verification method that combines parallelizable physics simulations with sum-of-squares (SOS) programming as a filter for fine-grained, formal certification of collision risk. This method is embedded within a Model Predictive Path Integral (MPPI) controller that uniquely utilizes binary collision information from SOS decomposition to improve its policy. The effectiveness of the proposed framework is validated on two typical robot manipulators through extensive simulations and real-world experiments, including a challenging human-robot collaboration scenario, demonstrating sim-to-real transfer of the learned model and its ability to generate safe and efficient trajectories in complex, uncertain settings.
>
---
#### [new 026] On the Cost of Evolving Task Specialization in Multi-Robot Systems
- **分类: cs.RO**

- **简介: 论文研究多机器人系统中任务专业化带来的成本与效益，针对有限优化预算下的效率问题。通过进化神经网络比较通用行为与专业行为的表现，发现专业行为在合作效率上不如通用行为。**

- **链接: [https://arxiv.org/pdf/2603.09552](https://arxiv.org/pdf/2603.09552)**

> **作者:** Paolo Leopardi; Heiko Hamann; Jonas Kuckling; Tanja Katharina Kaiser
>
> **备注:** Accepted for publication in the proceeding of ANTS 2026 - 15th International Conference on Swarm Intelligence
>
> **摘要:** Task specialization can lead to simpler robot behaviors and higher efficiency in multi-robot systems. Previous works have shown the emergence of task specialization during evolutionary optimization, focusing on feasibility rather than costs. In this study, we take first steps toward a cost-benefit analysis of task specialization in robot swarms using a foraging scenario. We evolve artificial neural networks as generalist behaviors for the entire task and as task-specialist behaviors for subtasks within a limited evaluation budget. We show that generalist behaviors can be successfully optimized while the evolved task-specialist controllers fail to cooperate efficiently, resulting in worse performance than the generalists. Consequently, task specialization does not necessarily improve efficiency when optimization budget is limited.
>
---
#### [new 027] SCDP: Learning Humanoid Locomotion from Partial Observations via Mixed-Observation Distillation
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人运动控制任务，解决从部分观测中学习仿人机器人运动的问题。提出SCDP方法，仅使用机载传感器，无需显式状态估计，实现稳定运动控制。**

- **链接: [https://arxiv.org/pdf/2603.09574](https://arxiv.org/pdf/2603.09574)**

> **作者:** Milo Carroll; Tianhu Peng; Lingfan Bao; Chengxu Zhou; Zhibin Li
>
> **备注:** 6 pages, 8 figures, 5 tables, iRos
>
> **摘要:** Distilling humanoid locomotion control from offline datasets into deployable policies remains a challenge, as existing methods rely on privileged full-body states that require complex and often unreliable state estimation. We present Sensor-Conditioned Diffusion Policies (SCDP) that enables humanoid locomotion using only onboard sensors, eliminating the need for explicit state estimation. SCDP decouples sensing from supervision through mixed-observation training: diffusion model conditions on sensor histories while being supervised to predict privileged future state-action trajectories, enforcing the model to infer the motion dynamics under partial observability. We further develop restricted denoising, context distribution alignment, and context-aware attention masking to encourage implicit state estimation within the model and to prevent train-deploy mismatch. We validate SCDP on velocity-commanded locomotion and motion reference tracking tasks. In simulation, SCDP achieves near-perfect success on velocity control (99-100%) and 93% tracking success in AMASS test set, performing comparable to privileged baselines while using only onboard sensors. Finally, we deploy the trained policy on a real G1 humanoid at 50 Hz, demonstrating robust real robot locomotion without external sensing or state estimation.
>
---
#### [new 028] High-Slip-Ratio Control for Peak Tire-Road Friction Estimation Using Automated Vehicles
- **分类: cs.RO**

- **简介: 该论文属于车辆控制任务，旨在解决恶劣路况下轮胎-路面摩擦系数估计问题。通过高滑移率控制框架，提升摩擦系数估计的准确性与安全性。**

- **链接: [https://arxiv.org/pdf/2603.09073](https://arxiv.org/pdf/2603.09073)**

> **作者:** Zhaohui Liang; Hang Zhou; Heye Huanh; Xiaopeng Li
>
> **摘要:** Accurate estimation of the tire-road friction coefficient (TRFC) is critical for ensuring safe vehicle control, especially under adverse road conditions. However, most existing methods rely on naturalistic driving data from regular vehicles, which typically operate under mild acceleration and braking. As a result, the data provide insufficient slip excitation and offer limited observability of the peak TRFC. This paper presents a high-slip-ratio control framework that enables automated vehicles (AVs) to actively excite the peak friction region during empty-haul operations while maintaining operational safety. A simplified Magic Formula tire model is adopted to represent nonlinear slip-force dynamics and is locally fitted using repeated high-slip measurements. To support safe execution in car-following scenarios, we formulate a constrained optimal control strategy that balances slip excitation, trajectory tracking, and collision avoidance. In parallel, a binning-based statistical projection method is introduced to robustly estimate peak TRFC under noise and local sparsity. The framework is validated through both closed-loop simulations and real-vehicle experiments, demonstrating its accuracy, safety, and feasibility for scalable, cost-effective roadway friction screening.
>
---
#### [new 029] Receptogenesis in a Vascularized Robotic Embodiment
- **分类: cs.RO; cond-mat.mtrl-sci**

- **简介: 该论文属于机器人学任务，旨在解决机器人物理自适应问题。通过流体系统实现材料动态重构，实现在地生成传感器，提升机器人实时环境响应能力。**

- **链接: [https://arxiv.org/pdf/2603.09473](https://arxiv.org/pdf/2603.09473)**

> **作者:** Kadri-Ann Pankratov; Leonid Zinatullin; Hans Priks; Adele Metsniit; Urmas Johanson; Tarmo Tamm; Alvo Aabloo; Edoardo Sinibaldi; Indrek Must
>
> **备注:** Supplementary Files currently unavailable online. Please contact the First Author to request any Supplementary Files
>
> **摘要:** Equipping robotic systems with the capacity to generate $\textit{ex novo}$ hardware during operation extends control of physical adaptability. Unlike modular systems that rely on discrete component integration pre- or post-deployment, we envision the possibility that physical adaptation and development emerge from dynamic material restructuring to shape the body's intrinsic functions. Drawing inspiration from circulatory systems that redistribute mass and function in biological organisms, we utilize fluidics to restructure the material interface, a capability currently unpaired in robotics. Here, we realize this synthetic growth capability through a vascularized robotic composite designed for programmable material synthesis, demonstrated via receptogenesis - the on-demand construction of sensors from internal fluid reserves based on environmental cues. By coordinating the fluidic transport of precursors with external localized UV irradiation, we drive an $\textit{in situ}$ photopolymerization that chemically reconstructs the vasculature from the inside out. This reaction converts precursors with photolatent initiator into a solid dispersion of UV-sensitive polypyrrole, establishing a sensing modality validated by a characteristic decrease in electrical impedance. The newly synthesized sensor closed a control loop to regulate wing flapping in a moth-inspired robotic demonstrator. This physical update increased the robot's capability in real time. This work establishes a materials-based framework for constitutive evolution, enabling robots to physically grow the hardware needed to support emerging behaviors in a complex environment; for example, suggesting a pathway toward autonomous systems capable of generating specialized features, such as neurovascular systems in situated robotics.
>
---
#### [new 030] STONE Dataset: A Scalable Multi-Modal Surround-View 3D Traversability Dataset for Off-Road Robot Navigation
- **分类: cs.RO**

- **简介: 该论文提出STONE数据集，解决非结构化环境下的3D可行驶性预测问题，通过多模态传感器和自动标注方法构建大规模标注数据。**

- **链接: [https://arxiv.org/pdf/2603.09175](https://arxiv.org/pdf/2603.09175)**

> **作者:** Konyul Park; Daehun Kim; Jiyong Oh; Seunghoon Yu; Junseo Park; Jaehyun Park; Hongjae Shin; Hyungchan Cho; Jungho Kim; Jun Won Choi
>
> **摘要:** Reliable off-road navigation requires accurate estimation of traversable regions and robust perception under diverse terrain and sensing conditions. However, existing datasets lack both scalability and multi-modality, which limits progress in 3D traversability prediction. In this work, we introduce STONE, a large-scale multi-modal dataset for off-road navigation. STONE provides (1) trajectory-guided 3D traversability maps generated by a fully automated, annotation-free pipeline, and (2) comprehensive surround-view sensing with synchronized 128-channel LiDAR, six RGB cameras, and three 4D imaging radars. The dataset covers a wide range of environments and conditions, including day and night, grasslands, farmlands, construction sites, and lakes. Our auto-labeling pipeline reconstructs dense terrain surfaces from LiDAR scans, extracts geometric attributes such as slope, elevation, and roughness, and assigns traversability labels beyond the robot's trajectory using a Mahalanobis-distance-based criterion. This design enables scalable, geometry-aware ground-truth construction without manual annotation. Finally, we establish a benchmark for voxel-level 3D traversability prediction and provide strong baselines under both single-modal and multi-modal settings. STONE is available at: this https URL
>
---
#### [new 031] MuxGel: Simultaneous Dual-Modal Visuo-Tactile Sensing via Spatially Multiplexing and Deep Reconstruction
- **分类: cs.RO**

- **简介: 该论文属于多模态感知任务，旨在解决视觉与触觉传感冲突的问题。通过空间复用设计，MuxGel同时获取视觉和触觉信息，并利用深度重建实现高保真信号恢复。**

- **链接: [https://arxiv.org/pdf/2603.09761](https://arxiv.org/pdf/2603.09761)**

> **作者:** Zhixian Hu; Zhengtong Xu; Sheeraz Athar; Juan Wachs; Yu She
>
> **备注:** Submitted to IROS 2026
>
> **摘要:** High-fidelity visuo-tactile sensing is important for precise robotic manipulation. However, most vision-based tactile sensors face a fundamental trade-off: opaque coatings enable tactile sensing but block pre-contact vision. To address this, we propose MuxGel, a spatially multiplexed sensor that captures both external visual information and contact-induced tactile signals through a single camera. By using a checkerboard coating pattern, MuxGel interleaves tactile-sensitive regions with transparent windows for external vision. This design maintains standard form factors, allowing for plug-and-play integration into GelSight-style sensors by simply replacing the gel pad. To recover full-resolution vision and tactile signals from the multiplexed inputs, we develop a U-Net-based reconstruction framework. Leveraging a sim-to-real pipeline, our model effectively decouples and restores high-fidelity tactile and visual fields simultaneously. Experiments on unseen objects demonstrate the framework's generalization and accuracy. Furthermore, we demonstrate MuxGel's utility in grasping tasks, where dual-modality feedback facilitates both pre-contact alignment and post-contact interaction. Results show that MuxGel enhances the perceptual capabilities of existing vision-based tactile sensors while maintaining compatibility with their hardware stacks. Project webpage: this https URL.
>
---
#### [new 032] TIMID: Time-Dependent Mistake Detection in Videos of Robot Executions
- **分类: cs.RO**

- **简介: 该论文提出TIMID框架，用于检测机器人执行高阶任务中的时间依赖性错误。解决视频异常检测中难以识别复杂时间违规的问题，通过弱监督训练和模拟数据提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.09782](https://arxiv.org/pdf/2603.09782)**

> **作者:** Nerea Gallego; Fernando Salanova; Claudio Mannarano; Cristian Mahulea; Eduardo Montijano
>
> **备注:** 8 pages, 5 figures , IROS submission
>
> **摘要:** As robotic systems execute increasingly difficult task sequences, so does the number of ways in which they can fail. Video Anomaly Detection (VAD) frameworks typically focus on singular, low-level kinematic or action failures, struggling to identify more complex temporal or spatial task violations, because they do not necessarily manifest as low-level execution errors. To address this problem, the main contribution of this paper is a new VAD-inspired architecture, TIMID, which is able to detect robot time-dependent mistakes when executing high-level tasks. Our architecture receives as inputs a video and prompts of the task and the potential mistake, and returns a frame-level prediction in the video of whether the mistake is present or not. By adopting a VAD formulation, the model can be trained with weak supervision, requiring only a single label per video. Additionally, to alleviate the problem of data scarcity of incorrect executions, we introduce a multi-robot simulation dataset with controlled temporal errors and real executions for zero-shot sim-to-real evaluation. Our experiments demonstrate that out-of-the-box VLMs lack the explicit temporal reasoning required for this task, whereas our framework successfully detects different types of temporal errors. Project: this https URL
>
---
#### [new 033] Predictive Control with Indirect Adaptive Laws for Payload Transportation by Quadrupedal Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于四足机器人负载运输任务，解决负载未知下的稳定控制问题。通过融合MPC与自适应算法，提升机器人在复杂地形中的负载运输能力。**

- **链接: [https://arxiv.org/pdf/2603.08831](https://arxiv.org/pdf/2603.08831)**

> **作者:** Leila Amanzadeh; Taizoon Chunawala; Randall T. Fawcett; Alexander Leonessa; Kaveh Akbari Hamed
>
> **备注:** 8 pages, 6 figures. Published in IEEE Robotics and Automation Letters
>
> **摘要:** This paper formally develops a novel hierarchical planning and control framework for robust payload transportation by quadrupedal robots, integrating a model predictive control (MPC) algorithm with a gradient-descent-based adaptive updating law. At the framework's high level, an indirect adaptive law estimates the unknown parameters of the reduced-order (template) locomotion model under varying payloads. These estimated parameters feed into an MPC algorithm for real-time trajectory planning, incorporating a convex stability criterion within the MPC constraints to ensure the stability of the template model's estimation error. The optimal reduced-order trajectories generated by the high-level adaptive MPC (AMPC) are then passed to a low-level nonlinear whole-body controller (WBC) for tracking. Extensive numerical investigations validate the framework's capabilities, showcasing the robot's proficiency in transporting unmodeled, unknown static payloads up to 109% in experiments on flat terrains and 91% on rough experimental terrains. The robot also successfully manages dynamic payloads with 73% of its mass on rough terrains. Performance comparisons with a normal MPC and an L1 MPC indicate a significant improvement. Furthermore, comprehensive hardware experiments conducted in indoor and outdoor environments confirm the method's efficacy on rough terrains despite uncertainties such as payload variations, push disturbances, and obstacles.
>
---
#### [new 034] Scale-Plan: Scalable Language-Enabled Task Planning for Heterogeneous Multi-Robot Teams
- **分类: cs.RO; cs.AI; cs.ET; cs.MA**

- **简介: 该论文提出Scale-Plan，解决异构多机器人系统长周期任务规划问题，通过语言引导生成精简任务表示，提升规划效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.08814](https://arxiv.org/pdf/2603.08814)**

> **作者:** Piyush Gupta; Sangjae Bae; Jiachen Li; David Isele
>
> **摘要:** Long-horizon task planning for heterogeneous multi-robot systems is essential for deploying collaborative teams in real-world environments; yet, it remains challenging due to the large volume of perceptual information, much of which is irrelevant to task objectives and burdens planning. Traditional symbolic planners rely on manually constructed problem specifications, limiting scalability and adaptability, while recent large language model (LLM)-based approaches often suffer from hallucinations and weak grounding-i.e., poor alignment between generated plans and actual environmental objects and constraints-in object-rich settings. We present Scale-Plan, a scalable LLM-assisted framework that generates compact, task-relevant problem representations from natural language instructions. Given a PDDL domain specification, Scale-Plan constructs an action graph capturing domain structure and uses shallow LLM reasoning to guide a structured graph search that identifies a minimal subset of relevant actions and objects. By filtering irrelevant information prior to planning, Scale-Plan enables efficient decomposition, allocation, and long-horizon plan generation. We evaluate our approach on complex multi-agent tasks and introduce MAT2-THOR, a cleaned benchmark built on AI2-THOR for reliable evaluation of multi-robot planning systems. Scale-Plan outperforms pure LLM and hybrid LLM-PDDL baselines across all metrics, improving scalability and reliability.
>
---
#### [new 035] BEACON: Language-Conditioned Navigation Affordance Prediction under Occlusion
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于语言引导的导航任务，解决遮挡环境下目标位置预测问题。提出BEACON模型，通过融合视觉和深度信息生成鸟瞰图热力图，提升导航准确性。**

- **链接: [https://arxiv.org/pdf/2603.09961](https://arxiv.org/pdf/2603.09961)**

> **作者:** Xinyu Gao; Gang Chen; Javier Alonso-Mora
>
> **备注:** 8 pages. Project page: this https URL
>
> **摘要:** Language-conditioned local navigation requires a robot to infer a nearby traversable target location from its current observation and an open-vocabulary, relational instruction. Existing vision-language spatial grounding methods usually rely on vision-language models (VLMs) to reason in image space, producing 2D predictions tied to visible pixels. As a result, they struggle to infer target locations in occluded regions, typically caused by furniture or moving humans. To address this issue, we propose BEACON, which predicts an ego-centric Bird's-Eye View (BEV) affordance heatmap over a bounded local region including occluded areas. Given an instruction and surround-view RGB-D observations from four directions around the robot, BEACON predicts the BEV heatmap by injecting spatial cues into a VLM and fusing the VLM's output with depth-derived BEV features. Using an occlusion-aware dataset built in the Habitat simulator, we conduct detailed experimental analysis to validate both our BEV space formulation and the design choices of each module. Our method improves the accuracy averaged across geodesic thresholds by 22.74 percentage points over the state-of-the-art image-space baseline on the validation subset with occluded target locations. Our project page is: this https URL.
>
---
#### [new 036] Kinodynamic Motion Retargeting for Humanoid Locomotion via Multi-Contact Whole-Body Trajectory Optimization
- **分类: cs.RO**

- **简介: 该论文属于人形机器人运动重定向任务，解决传统方法导致的物理不一致问题。通过多接触全身体轨迹优化，提升运动动态可行性与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.09956](https://arxiv.org/pdf/2603.09956)**

> **作者:** Xiaoyu Zhang; Steven Haener; Varun Madabushi; Maegan Tucker
>
> **摘要:** We present the KinoDynamic Motion Retargeting (KDMR) framework, a novel approach for humanoid locomotion that models the retargeting process as a multi-contact, whole-body trajectory optimization problem. Conventional kinematics-based retargeting methods rely solely on spatial motion capture (MoCap) data, inevitably introducing physically inconsistent artifacts, such as foot sliding and ground penetration, that severely degrade the performance of downstream imitation learning policies. To bridge this gap, KDMR extends beyond pure kinematics by explicitly enforcing rigid-body dynamics and contact complementarity constraints. Further, by integrating ground reaction force (GRF) measurements alongside MoCap data, our method automatically detects heel-toe contact events to accurately replicate complex human-like contact patterns. We evaluate KDMR against the state-of-the-art baseline, GMR, across three key dimensions: 1) the dynamic feasibility and smoothness of the retargeted motions, 2) the accuracy of GRF tracking compared to raw source data, and 3) the training efficiency and final performance of downstream control policies trained via the BeyondMimic framework. Experimental results demonstrate that KDMR significantly outperforms purely kinematic methods, yielding dynamically viable reference trajectories that accelerate policy convergence and enhance overall locomotion stability. Our end-to-end pipeline will be open-sourced upon publication.
>
---
#### [new 037] Adaptive SINDy: Residual Force System Identification Based UAV Disturbance Rejection
- **分类: cs.RO**

- **简介: 该论文属于无人机扰动抑制任务，旨在解决复杂环境中风扰动导致的稳定性问题。通过结合SINDy与RLS方法，提升无人机轨迹跟踪精度。**

- **链接: [https://arxiv.org/pdf/2603.08863](https://arxiv.org/pdf/2603.08863)**

> **作者:** Fawad Mehboob; Amir Atef Habel; Roohan Ahmed Khan; Mikhail Derevianchenko; Clement Fortin; Dzmitry Tsetserukou
>
> **摘要:** The stability and control of Unmanned Aerial Vehicles (UAVs) in a turbulent environment is a matter of great concern. Devising a robust control algorithm to reject disturbances is challenging due to the highly nonlinear nature of wind dynamics, and modeling the dynamics using analytical techniques is not straightforward. While traditional techniques using disturbance observers and classical adaptive control have shown some progress, they are mostly limited to relatively non-complex environments. On the other hand, learning based approaches are increasingly being used for modeling of residual forces and disturbance rejection; however, their generalization and interpretability is a factor of concern. To this end, we propose a novel integration of data-driven system identification using Sparse Identification of Non-Linear Dynamics (SINDy) with a Recursive Least Square (RLS) adaptive control to adapt and reject wind disturbances in a turbulent environment. We tested and validated our approach on Gazebo harmonic environment and on real flights with wind speeds of up to 2 m/s from four directions, creating a highly dynamic and turbulent environment. Adaptive SINDy outperformed the baseline PID and INDI controllers on several trajectory tracking error metrics without crashing. A root mean square error (RMSE) of up to 12.2 cm and 17.6 cm, and a mean absolute error (MAE) of 13.7 cm and 10.5 cm were achieved on circular and lemniscate trajectories, respectively. The validation was performed on a very lightweight Crazyflie drone under a highly dynamic environment for complex trajectory tracking.
>
---
#### [new 038] SEA-Nav: Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments
- **分类: cs.RO**

- **简介: 该论文属于四足机器人导航任务，旨在解决复杂环境中安全与敏捷性不足的问题。提出SEA-Nav框架，结合CBF和自适应机制，实现高效安全的导航。**

- **链接: [https://arxiv.org/pdf/2603.09460](https://arxiv.org/pdf/2603.09460)**

> **作者:** Shiyi Chen; Mingye Yang; Haiyan Mao; Jiaqi Zhang; Haiyi Liu; Shuheng He; Debing Zhang; Zihao Qiu; Chun Zhang
>
> **备注:** Project website: this https URL
>
> **摘要:** Efficiently training quadruped robot navigation in densely cluttered environments remains a significant challenge. Existing methods are either limited by a lack of safety and agility in simple obstacle distributions or suffer from slow locomotion in complex environments, often requiring excessively long training phases. To this end, we propose SEA-Nav (Safe, Efficient, and Agile Navigation), a reinforcement learning framework for quadruped navigation. Within diverse and dense obstacle environments, a differentiable control barrier function (CBF)-based shield constraints the navigation policy to output safe velocity commands. An adaptive collision replay mechanism and hazardous exploration rewards are introduced to increase the probability of learning from critical experiences, guiding efficient exploration and exploitation. Finally, kinematic action constraints are incorporated to ensure safe velocity commands, facilitating successful physical deployment. To the best of our knowledge, this is the first approach that achieves highly challenging quadruped navigation in the real world with minute-level training time.
>
---
#### [new 039] Robotic Scene Cloning:Advancing Zero-Shot Robotic Scene Adaptation in Manipulation via Visual Prompt Editing
- **分类: cs.RO**

- **简介: 该论文属于机器人场景适应任务，解决预训练机器人在新场景中零样本泛化能力不足的问题。通过编辑操作轨迹实现场景克隆，提升策略泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.09712](https://arxiv.org/pdf/2603.09712)**

> **作者:** Binyuan Huang; Yuqing Wen; Yucheng Zhao; Yaosi Hu; Tiancai Wang; Chang Wen Chen; Haoqiang Fan; Zhenzhong Chen
>
> **摘要:** Modern robots can perform a wide range of simple tasks and adapt to diverse scenarios in the well-trained environment. However, deploying pre-trained robot models in real-world user scenarios remains challenging due to their limited zero-shot capabilities, often necessitating extensive on-site data collection. To address this issue, we propose Robotic Scene Cloning (RSC), a novel method designed for scene-specific adaptation by editing existing robot operation trajectories. RSC achieves accurate and scene-consistent sample generation by leveraging a visual prompting mechanism and a carefully tuned condition injection module. Not only transferring textures but also performing moderate shape adaptations in response to the visual prompts, RSC demonstrates reliable task performance across a variety of object types. Experiments across various simulated and real-world environments demonstrate that RSC significantly enhances policy generalization in target environments.
>
---
#### [new 040] Fly, Track, Land: Infrastructure-less Magnetic Localization for Heterogeneous UAV-UGV Teaming
- **分类: cs.RO**

- **简介: 该论文属于无人机与地面机器人协同任务，解决无基础设施下的精准定位问题。提出一种磁感应定位系统，实现无人机对移动机器人的厘米级自主跟踪与着陆。**

- **链接: [https://arxiv.org/pdf/2603.08926](https://arxiv.org/pdf/2603.08926)**

> **作者:** Valerio Brunacci; Davide Plozza; Alessio De Angelis; Michele Magno; Tommaso Polonelli
>
> **备注:** Submitted to IEEE Transactions on Robotics (T-RO). Supplementary video available
>
> **摘要:** We present a complete infrastructure-less magneto-inductive (MI) localization system enabling a lightweight UAV to autonomously hover, track, and land with centimeter precision on a mobile quadruped robot acting as a dynamic docking pad. This work advances the vision of heterogeneous robot collaboration, where ultra-lightweight flying robots serve as mobile perception agents for ground-based Unmanned Ground Vehicles (UGVs). By extending the sensing horizon and providing complementary viewpoints, the UAVs enhance exploration efficiency and improve the quality of data collection in large-scale, unknown environments. The proposed system aims to complements traditional localization modalities with a compact, embedded, and infrastructure-less magnetic sensing approach, providing accurate short-range relative positioning to bridge the gap between coarse navigation and precise UAV docking. A single lightweight receive coil and a fully embedded estimation pipeline on the UAV deliver 20 Hz relative pose estimates in the UGV's frame, achieving a 3D position root-mean-square error (RMSE) of 5 cm. The system uses real-time estimation and a warm-started solver to estimate the 3D position, which is then fused with inertial and optical-flow measurements in the onboard extended Kalman filter. Real-world experiments validate the effectiveness of the framework, demonstrating significant improvements in UAV--UGV teaming in infrastructure-less scenarios compared to state-of-the-art methods, requiring no external anchors or global positioning. In dynamic scenarios, the UAV tracks and docks with a moving UGV while maintaining a 7.2 cm RMSE and achieving successful autonomous landings.
>
---
#### [new 041] Towards Terrain-Aware Safe Locomotion for Quadrupedal Robots Using Proprioceptive Sensing
- **分类: cs.RO**

- **简介: 该论文属于四足机器人安全运动任务，解决复杂地形下安全控制问题。通过融合本体感觉信息，构建地形估计框架并集成安全控制，提升定位与接触估计的精度和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.09585](https://arxiv.org/pdf/2603.09585)**

> **作者:** Peiyu Yang; Jiatao Ding; Wei Pan; Claudio Semini; Cosimo Della Santina
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** Achieving safe quadrupedal locomotion in real-world environments has attracted much attention in recent years. When walking over uneven terrain, achieving reliable estimation and realising safety-critical control based on the obtained information is still an open question. To address this challenge, especially for low-cost robots equipped solely with proprioceptive sensors (e.g., IMUs, joint encoders, and contact force sensors), this work first presents an estimation framework that generates a 2.5-D terrain map and extracts support plane parameters, which are then integrated into contact and state estimation. Then, we integrate this estimation framework into a safety-critical control pipeline by formulating control barrier functions that provide rigorous safety guarantees. Experiments demonstrate that the proposed terrain estimation method provides smooth terrain representations. Moreover, the coupled estimation framework of terrain, state, and contact reduces the mean absolute error of base position estimation by 64.8%, decreases the estimation variance by 47.2%, and improves the robustness of contact estimation compared to a decoupled framework. The terrain-informed CBFs integrate historical terrain information and current proprioceptive measurements to ensure global safety by keeping the robot out of hazardous areas and local safety by preventing body-terrain collision, relying solely on proprioceptive sensing.
>
---
#### [new 042] A Generalized Voronoi Graph based Coverage Control Approach for Non-Convex Environment
- **分类: cs.RO**

- **简介: 该论文属于多机器人覆盖控制任务，旨在解决非凸环境中机器人高效覆盖的问题。通过分阶段方法，利用广义Voronoi图进行区域划分与负载均衡，实现有效覆盖。**

- **链接: [https://arxiv.org/pdf/2603.09596](https://arxiv.org/pdf/2603.09596)**

> **作者:** Zuyi Guo; Ronghao Zheng; Meiqin Liu; Senlin Zhang
>
> **备注:** 8 pages, 7 figures, published to ACC 2026
>
> **摘要:** To address the challenge of efficient coverage by multi-robot systems in non-convex regions with multiple obstacles, this paper proposes a coverage control method based on the Generalized Voronoi Graph (GVG), which has two phases: Load-Balancing Algorithm phase and Collaborative Coverage phase. In Load-Balancing Algorithm phase, the non-convex region is partitioned into multiple sub-regions based on GVG. Besides, a weighted load-balancing algorithm is developed, which considers the quality differences among sub-regions. By iteratively optimizing the robot allocation ratio, the number of robots in each sub-region is matched with the sub-region quality to achieve load balance. In Collaborative Coverage phase, each robot is controlled by a new controller to effectively coverage the region. The convergence of the method is proved and its performance is evaluated through simulations.
>
---
#### [new 043] Characterization, Analytical Planning, and Hybrid Force Control for the Inspire RH56DFX Hand
- **分类: cs.RO**

- **简介: 该论文针对Inspire RH56DFX机械手的使用难题，提出硬件校准、仿真模型和混合控制策略，提升其作为研究工具的性能与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.08988](https://arxiv.org/pdf/2603.08988)**

> **作者:** Xuan Tan; William Xie; Nikolaus Correll
>
> **摘要:** Commercially accessible dexterous robot hands are increasingly prevalent, but many remain difficult to use as scientific instruments. For example, the Inspire RH56DFX hand exposes only uncalibrated proprioceptive information and shows unreliable contact behavior at high speed (up to 1618% force limit overshoot). Furthermore, its underactuated, coupled finger linkages make antipodal grasps non-trivial. We contribute three improvements to the Inspire RH56DFX to transform it from a black-box device to a research tool: (1) hardware characterization (force calibration, latency, and overshoot), (2) a sim2real validated MuJoCo model for analytical width-to-grasp planning, and (3) a hybrid, closed-loop speed-force grasp controller. We validate these components on peg-in-hole insertion, achieving 65% success and outperforming a wrist-force-only baseline of 10% and on 300 grasps across 15 physically diverse objects, achieving 87% success and outperforming plan-free grasps and learned grasps. Our approach is modular, designed for compatibility with external object detectors and vision-language models for width & force estimation and high-level planning, and provides an interpretable and immediately deployable interface for dexterous manipulation with the Inspire RH56DFX hand, open-sourced at this website this https URL.
>
---
#### [new 044] Stein Variational Ergodic Surface Coverage with SE(3) Constraints
- **分类: cs.RO**

- **简介: 该论文属于机器人表面操作任务，解决3D点云覆盖问题。提出基于SE(3)约束的Stein变分梯度下降方法，提升轨迹优化效果。**

- **链接: [https://arxiv.org/pdf/2603.09458](https://arxiv.org/pdf/2603.09458)**

> **作者:** Jiayun Li; Yufeng Jin; Sangli Teng; Dejian Gong; Georgia Chalvatzaki
>
> **摘要:** Surface manipulation tasks require robots to generate trajectories that comprehensively cover complex 3D surfaces while maintaining precise end-effector poses. Existing ergodic trajectory optimization (TO) methods demonstrate success in coverage tasks, while struggling with point-cloud targets due to the nonconvex optimization landscapes and the inadequate handling of SE(3) constraints in sampling-as-optimization (SAO) techniques. In this work, we introduce a preconditioned SE(3) Stein Variational Gradient Descent (SVGD) approach for SAO ergodic trajectory generation. Our proposed approach comprises multiple innovations. First, we reformulate point-cloud ergodic coverage as a manifold-aware sampling problem. Second, we derive SE(3)-specific SVGD particle updates, and, third, we develop a preconditioner to accelerate TO convergence. Our sampling-based framework consistently identifies superior local optima compared to strong optimization-based and SAO baselines while preserving the SE(3) geometric structure. Experiments on a 3D point-cloud surface coverage benchmark and robotic surface drawing tasks demonstrate that our method achieves superior coverage quality with tractable computation in our setting relative to existing TO and SAO approaches, and is validated in real-world robot experiments.
>
---
#### [new 045] TRIP-Bag: A Portable Teleoperation System for Plug-and-Play Robotic Arms and Leaders
- **分类: cs.RO**

- **简介: 该论文提出TRIP-Bag系统，用于解决机器人操作数据收集难题。通过便携式遥控操作，减少实体差距，提升数据质量，支持多种环境下的高效数据采集。**

- **链接: [https://arxiv.org/pdf/2603.09226](https://arxiv.org/pdf/2603.09226)**

> **作者:** Noboru Myers; Sankalp Yamsani; Obin Kwon; Joohyung Kim
>
> **摘要:** Large scale, diverse demonstration data for manipulation tasks remains a major challenge in learning-based robot policies. Existing in-the-wild data collection approaches often rely on vision-based pose estimation of hand-held grippers or gloves, which introduces an embodiment gap between the collection platform and the target robot. Teleoperation systems eliminate the embodiment gap, but are typically impractical to deploy outside the laboratory environment. We propose TRIP-Bag (Teleoperation, Recording, Intelligence in a Portable Bag), a portable, puppeteer-style teleoperation system fully contained within a commercial suitcase, as a practical solution for collecting high-fidelity manipulation data across varied settings. With a setup time of under five minutes and direct joint-to-joint teleoperation, TRIP-Bag enables rapid and reliable data collection in any environment. We validated TRIP-Bag's usability through experiments with non-expert users, showing that the system is intuitive and easy to operate. Furthermore, we confirmed the quality of the collected data by training benchmark manipulation policies, demonstrating its value as a practical resource for robot learning.
>
---
#### [new 046] SEP-NMPC: Safety Enhanced Passivity-Based Nonlinear Model Predictive Control for a UAV Slung Payload System
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于无人机运输任务，解决悬吊载荷飞行中的稳定与避障问题。提出SEP-NMPC方法，结合能量稳定性与高阶控制屏障函数，确保安全与实时性。**

- **链接: [https://arxiv.org/pdf/2603.08860](https://arxiv.org/pdf/2603.08860)**

> **作者:** Seyedreza Rezaei; Junjie Kang; Amaldev Haridevan; Jinjun Shan
>
> **备注:** Accepted at ICRA 2026
>
> **摘要:** Model Predictive Control (MPC) is widely adopted for agile multirotor vehicles, yet achieving both stability and obstacle-free flight is particularly challenging when a payload is suspended beneath the airframe. This paper introduces a Safety Enhanced Passivity-Based Nonlinear MPC (SEP-NMPC) that provides formal guarantees of stability and safety for a quadrotor transporting a slung payload through cluttered environments. Stability is enforced by embedding a strict passivity inequality, which is derived from a shaped energy storage function with adaptive damping, directly into the NMPC. This formulation dissipates excess energy and ensures asymptotic convergence despite payload swings. Safety is guaranteed through high-order control barrier functions (HOCBFs) that render user-defined clearance sets forward-invariant, obliging both the quadrotor and the swinging payload to maintain separation while interacting with static and dynamic obstacles. The optimization remains quadratic-program compatible and is solved online at each sampling time without gain scheduling or heuristic switching. Extensive simulations and real-world experiments confirm stable payload transport, collision-free trajectories, and real-time feasibility across all tested scenarios. The SEP-NMPC framework therefore unifies passivity-based closed-loop stability with HOCBF-based safety guarantees for UAV slung-payload transportation.
>
---
#### [new 047] Cutting the Cord: System Architecture for Low-Cost, GPU-Accelerated Bimanual Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文属于移动操作任务，旨在解决低成本、无缆化机器人系统的设计与实现。通过优化机械结构、电源拓扑和嵌入式自主控制，构建了一个具备远程操控和自主导航能力的双臂机器人平台。**

- **链接: [https://arxiv.org/pdf/2603.09051](https://arxiv.org/pdf/2603.09051)**

> **作者:** Artemis Shaw; Chen Liu; Justin Costa; Rane Gray; Alina Skowronek; Kevin Diaz; Nam Bui; Nikolaus Correll
>
> **摘要:** We present a bimanual mobile manipulator built on the open-source XLeRobot with integrated onboard compute for less than \$1300. Key contributions include: (1) optimized mechanical design maximizing stiffness-to-weight ratio, (2) a Tri-Bus power topology isolating compute from motor-induced voltage transients, and (3) embedded autonomy using NVIDIA Jetson Orin Nano for untethered operation. The platform enables teleoperation, autonomous SLAM navigation, and vision-based manipulation without external dependencies, providing a low-cost alternative for research and education in robotics and robot learning.
>
---
#### [new 048] NLiPsCalib: An Efficient Calibration Framework for High-Fidelity 3D Reconstruction of Curved Visuotactile Sensors
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉触觉传感器校准任务，解决非均匀光照导致的重建精度问题。提出NLiPsCalib框架，利用可控光源简化校准过程。**

- **链接: [https://arxiv.org/pdf/2603.09319](https://arxiv.org/pdf/2603.09319)**

> **作者:** Xuhao Qin; Feiyu Zhao; Yatao Leng; Runze Hu; Chenxi Xiao
>
> **备注:** 8 pages, 8 figures, accepted to 2026 IEEE International Conference on Robotics & Automation (ICRA 2026)
>
> **摘要:** Recent advances in visuotactile sensors increasingly employ biomimetic curved surfaces to enhance sensorimotor capabilities. Although such curved visuotactile sensors enable more conformal object contact, their perceptual quality is often degraded by non-uniform illumination, which reduces reconstruction accuracy and typically necessitates calibration. Existing calibration methods commonly rely on customized indenters and specialized devices to collect large-scale photometric data, but these processes are expensive and labor-intensive. To overcome these calibration challenges, we present NLiPsCalib, a physics-consistent and efficient calibration framework for curved visuotactile sensors. NLiPsCalib integrates controllable near-field light sources and leverages Near-Light Photometric Stereo (NLiPs) to estimate contact geometry, simplifying calibration to just a few simple contacts with everyday objects. We further introduce NLiPsTac, a controllable-light-source tactile sensor developed to validate our framework. Experimental results demonstrate that our approach enables high-fidelity 3D reconstruction across diverse curved form factors with a simple calibration procedure. We emphasize that our approach lowers the barrier to developing customized visuotactile sensors of diverse geometries, thereby making visuotactile sensing more accessible to the broader community.
>
---
#### [new 049] Proprioceptive Safe Active Navigation and Exploration for Planetary Environments
- **分类: cs.RO**

- **简介: 该论文属于行星环境导航任务，解决未知松散地形的安全导航问题。通过腿部与地形交互信息构建可 traversability 模型，实现安全探索与路径规划。**

- **链接: [https://arxiv.org/pdf/2603.08905](https://arxiv.org/pdf/2603.08905)**

> **作者:** Matthew Y. Jiang; Feifei Qian; Shipeng Liu
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Deformable granular terrains introduce significant locomotion and immobilization risks in planetary exploration and are difficult to detect via remote sensing (e.g., vision). Legged robots can sense terrain properties through leg-terrain interactions during locomotion, offering a direct means to assess traversability in deformable environments. How to systematically exploit this interaction-derived information for navigation planning, however, remains underexplored. We address this gap by presenting PSANE, a Proprioceptive Safe Active Navigation and Exploration framework that leverages leg-terrain interaction measurements for safe navigation and exploration in unknown deformable environments. PSANE learns a traversability model via Gaussian Process regression to estimate and certify safe regions and identify exploration frontiers online, and integrates these estimates with a reactive controller for real-time navigation. Frontier selection is formulated as a multi-objective optimization that balances safe-set expansion probability and goal-directed cost, with subgoals selected via scalarization over the Pareto-optimal frontier set. PSANE safely explores unknown granular terrain and reaches specified goals using only proprioceptively estimated traversability, while achieving performance improvements over baseline methods.
>
---
#### [new 050] Formation-Aware Adaptive Conformalized Perception for Safe Leader-Follower Multi-Robot Systems
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于多机器人系统安全感知任务，解决分布式视觉跟随中的感知安全性问题。通过自适应共形预测方法，提升 formation 相关的不确定性量化与跟踪性能。**

- **链接: [https://arxiv.org/pdf/2603.08958](https://arxiv.org/pdf/2603.08958)**

> **作者:** Richie R. Suganda; Bin Hu
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** This paper considers the perception safety problem in distributed vision-based leader-follower formations, where each robot uses onboard perception to estimate relative states, track desired setpoints, and keep the leader within its camera field of view (FOV). Safety is challenging due to heteroscedastic perception errors and the coupling between formation maneuvers and visibility constraints. We propose a distributed, formation-aware adaptive conformal prediction method based on Risk-Aware Mondrian CP to produce formation-conditioned uncertainty quantiles. The resulting bounds tighten in high-risk configurations (near FOV limits) and relax in safer regions. We integrate these bounds into a Formation-Aware Conformal CBF-QP with a smooth margin to enforce visibility while maintaining feasibility and tracking performance. Gazebo simulations show improved formation success rates and tracking accuracy over non-adaptive (global) CP baselines that ignore formation-dependent visibility risk, while preserving finite-sample probabilistic safety guarantees. The experimental videos are available on the \href{this https URL}{project website}\footnote{Project Website: this https URL}.
>
---
#### [new 051] DRIFT: Dual-Representation Inter-Fusion Transformer for Automated Driving Perception with 4D Radar Point Clouds
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DRIFT模型，用于自动驾驶中的感知任务，解决4D雷达点云密度低导致的上下文信息不足问题。通过双路径结构融合局部与全局特征，提升目标检测和道路估计性能。**

- **链接: [https://arxiv.org/pdf/2603.09695](https://arxiv.org/pdf/2603.09695)**

> **作者:** Siqi Pei; Andras Palffy; Dariu M. Gavrila
>
> **摘要:** 4D radars, which provide 3D point cloud data along with Doppler velocity, are attractive components of modern automated driving systems due to their low cost and robustness under adverse weather conditions. However, they provide a significantly lower point cloud density than LiDAR sensors. This makes it important to exploit not only local but also global contextual scene information. This paper proposes DRIFT, a model that effectively captures and fuses both local and global contexts through a dual-path architecture. The model incorporates a point path to aggregate fine-grained local features and a pillar path to encode coarse-grained global features. These two parallel paths are intertwined via novel feature-sharing layers at multiple stages, enabling full utilization of both representations. DRIFT is evaluated on the widely used View-of-Delft (VoD) dataset and a proprietary internal dataset. It outperforms the baselines on the tasks of object detection and/or free road estimation. For example, DRIFT achieves a mean average precision (mAP) of 52.6\% (compared to, say, 45.4\% of CenterPoint) on the VoD dataset.
>
---
#### [new 052] MO-Playground: Massively Parallelized Multi-Objective Reinforcement Learning for Robotics
- **分类: cs.RO**

- **简介: 该论文属于多目标强化学习任务，旨在解决传统方法计算效率低的问题。作者提出MORLAX算法和MO-Playground平台，实现快速并行化多目标机器人学习。**

- **链接: [https://arxiv.org/pdf/2603.09237](https://arxiv.org/pdf/2603.09237)**

> **作者:** Neil Janwani; Ellen Novoseller; Vernon J. Lawhern; Maegan Tucker
>
> **备注:** 8 pages, 4 figures, 3 tables
>
> **摘要:** Multi-objective reinforcement learning (MORL) is a powerful tool to learn Pareto-optimal policy families across conflicting objectives. However, unlike traditional RL algorithms, existing MORL algorithms do not effectively leverage large-scale parallelization to concurrently simulate thousands of environments, resulting in vastly increased computation time. Ultimately, this has limited MORL's application towards complex multi-objective robotics problems. To address these challenges, we present 1) MORLAX, a new GPU-native, fast MORL algorithm, and 2) MO-Playground, a pip-installable playground of GPU-accelerated multi-objective environments. Together, MORLAX and MO-Playground approximate Pareto sets within minutes, offering 25-270x speed-ups compared to legacy CPU-based approaches whilst achieving superior Pareto front hypervolumes. We demonstrate the versatility of our approach by implementing a custom BRUCE humanoid robot environment using MO-Playground and learning Pareto-optimal locomotion policies across 6 realistic objectives for BRUCE, such as smoothness, efficiency and arm swinging.
>
---
#### [new 053] SurgCalib: Gaussian Splatting-Based Hand-Eye Calibration for Robot-Assisted Minimally Invasive Surgery
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于手眼标定任务，解决手术机器人中因电缆伸缩导致的定位误差问题。提出SurgCalib框架，实现无需标记的自动标定。**

- **链接: [https://arxiv.org/pdf/2603.08983](https://arxiv.org/pdf/2603.08983)**

> **作者:** Zijian Wu; Shuojue Yang; Yu Chung Lee; Eitan Prisman; Yueming Jin; Septimiu E. Salcudean
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** We present a Gaussian Splatting-based framework for hand-eye calibration of the da Vinci surgical robot. In a vision-guided robotic system, accurate estimation of the rigid transformation between the robot base and the camera frame is essential for reliable closed-loop control. For cable-driven surgical robots, this task faces unique challenges. The encoders of surgical instruments often produce inaccurate proprioceptive measurements due to cable stretch and backlash. Conventional hand-eye calibration approaches typically rely on known fiducial patterns and solve the AX = XB formulation. While effective, introducing additional markers into the operating room (OR) environment can violate sterility protocols and disrupt surgical workflows. In this study, we propose SurgCalib, an automatic, markerless framework that has the potential to be used in the OR. SurgCalib first initializes the pose of the surgical instrument using raw kinematic measurements and subsequently refines this pose through a two-phase optimization procedure under the RCM constraint within a Gaussian Splatting-based differentiable rendering pipeline. We evaluate the proposed method on the public dVRK benchmark, SurgPose. The results demonstrate average 2D tool-tip reprojection errors of 12.24 px (2.06 mm) and 11.33 px (1.9 mm), and 3D tool-tip Euclidean distance errors of 5.98 mm and 4.75 mm, for the left and right instruments, respectively.
>
---
#### [new 054] HMR-1: Hierarchical Massage Robot with Vision-Language-Model for Embodied Healthcare
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决按摩机器人缺乏标准评估和多模态数据的问题。构建了多模态数据集MedMassage-12K，并提出分层控制框架，提升按摩精度与实用性。**

- **链接: [https://arxiv.org/pdf/2603.08817](https://arxiv.org/pdf/2603.08817)**

> **作者:** Rongtao Xu; Mingming Yu; Xiaofeng Han; Yu Zhang; Kaiyi Hu; Zhe Feng; Zenghuang Fu; Changwei Wang; Weiliang Meng; Xiaopeng Zhang
>
> **摘要:** The rapid advancement of Embodied Intelligence has opened transformative opportunities in healthcare, particularly in physical therapy and rehabilitation. However, critical challenges remain in developing robust embodied healthcare solutions, such as the lack of standardized evaluation benchmarks and the scarcity of open-source multimodal acupoint massage datasets. To address these gaps, we construct MedMassage-12K - a multimodal dataset containing 12,190 images with 174,177 QA pairs, covering diverse lighting conditions and backgrounds. Furthermore, we propose a hierarchical embodied massage framework, which includes a high-level acupoint grounding module and a low-level control module. The high-level acupoint grounding module uses multimodal large language models to understand human language and identify acupoint locations, while the low-level control module provides the planned trajectory. Based on this, we evaluate existing MLLMs and establish a benchmark for embodied massage tasks. Additionally, we fine-tune the Qwen-VL model, demonstrating the framework's effectiveness. Physical experiments further confirm the practical applicability of the this http URL dataset and code are publicly available at this https URL.
>
---
#### [new 055] Lightweight 3D LiDAR-Based UAV Tracking: An Adaptive Extended Kalman Filtering Approach
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于无人机相对定位任务，解决小无人机在无GPS环境下精准跟踪的问题。通过设计轻量级LiDAR系统与自适应扩展卡尔曼滤波，提升跟踪精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.09783](https://arxiv.org/pdf/2603.09783)**

> **作者:** Nivand Khosravi; Meysam Basiri; Rodrigo Ventura
>
> **备注:** Presented at the 19th International Conference on Intelligent Autonomous Systems, IAS-19, Genoa, Italy, June 30 to July 4, 2025. To appear in the Springer post-proceedings of the conference
>
> **摘要:** Accurate relative positioning is crucial for swarm aerial robotics, enabling coordinated flight and collision avoidance. Although vision-based tracking has been extensively studied, 3D LiDAR-based methods remain underutilized despite their robustness under varying lighting conditions. Existing systems often rely on bulky, power-intensive sensors, making them impractical for small UAVs with strict payload and energy constraints. This paper presents a lightweight LiDAR-based UAV tracking system incorporating an Adaptive Extended Kalman Filter (AEKF) framework. Our approach effectively addresses the challenges posed by sparse, noisy, and nonuniform point cloud data generated by non-repetitive scanning 3D LiDARs, ensuring reliable tracking while remaining suitable for small drones with strict payload constraints. Unlike conventional filtering techniques, the proposed method dynamically adjusts the noise covariance matrices using innovation and residual statistics, thereby enhancing tracking accuracy under real-world conditions. Additionally, a recovery mechanism ensures continuity of tracking during temporary detection failures caused by scattered LiDAR returns or occlusions. Experimental validation was performed using a Livox Mid-360 LiDAR mounted on a DJI F550 UAV in real-world flight scenarios. The proposed method demonstrated robust UAV tracking performance under sparse LiDAR returns and intermittent detections, consistently outperforming both standard Kalman filtering and particle filtering approaches during aggressive maneuvers. These results confirm that the framework enables reliable relative positioning in GPS-denied environments without the need for multi-sensor arrays or external infrastructure.
>
---
#### [new 056] Beyond Amplitude: Channel State Information Phase-Aware Deep Fusion for Robotic Activity Recognition
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于机器人活动识别任务，旨在解决传统方法仅利用CSI幅度而忽视相位信息的问题。通过提出GF-BiLSTM模型，融合幅度与相位信息，提升识别准确性和跨速度鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.09047](https://arxiv.org/pdf/2603.09047)**

> **作者:** Rojin Zandi; Hojjat Salehinejad; Milad Siami
>
> **备注:** Accepted at 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), May 4--8, 2026, Barcelona, Spain
>
> **摘要:** Wi-Fi Channel State Information (CSI) has emerged as a promising non-line-of-sight sensing modality for human and robotic activity recognition. However, prior work has predominantly relied on CSI amplitude while underutilizing phase information, particularly in robotic arm activity recognition. In this paper, we present GateFusion-Bidirectional Long Short-Term Memory network (GF-BiLSTM) for WiFi sensing in robotic activity recognition. GF-BiLSTM is a two-stream gated fusion network that encodes amplitude and phase separately and adaptively integrates per-time features through a learned gating mechanism. We systematically evaluate state-of-the-art deep learning models under a Leave-One-Velocity-Out (LOVO) protocol across four input configurations: amplitude only, phase only, amplitude + unwrapped phase, and amplitude + sanitized phase. Experimental results demonstrate that incorporating phase alongside amplitude consistently improves recognition accuracy and cross-speed robustness, with GF-BiLSTM achieving the best performance. To the best of our knowledge, this work provides the first systematic exploration of CSI phase for robotic activity recognition, establishing its critical role in Wi-Fi-based sensing.
>
---
#### [new 057] NanoBench: A Multi-Task Benchmark Dataset for Nano-Quadrotor System Identification, Control, and State Estimation
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出NanoBench，一个用于纳米四旋翼系统识别、控制和状态估计的多任务基准数据集，解决小尺寸无人机建模与控制难题。**

- **链接: [https://arxiv.org/pdf/2603.09908](https://arxiv.org/pdf/2603.09908)**

> **作者:** Syed Izzat Ullah; Jose Baca
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Existing aerial-robotics benchmarks target vehicles from hundreds of grams to several kilograms and typically expose only high-level state data. They omit the actuator-level signals required to study nano-scale quadrotors, where low-Reynolds number aerodynamics, coreless DC motor nonlinearities, and severe computational constraints invalidate models and controllers developed for larger vehicles. We introduce NanoBench, an open-source multi-task benchmark collected on the commercially available Crazyflie 2.1 nano-quadrotor (takeoff weight 27 g) in a Vicon motion capture arena. The dataset contains over 170 flight trajectories spanning hover, multi-frequency excitation, standard tracking, and aggressive maneuvers across multiple speed regimes. Each trajectory provides synchronized Vicon ground truth, raw IMU data, onboard extended Kalman filter estimates, PID controller internals, and motor PWM commands at 100 Hz, alongside battery telemetry at 10 Hz, aligned with sub-0.5 ms consistency. NanoBench defines standardized evaluation protocols, train/test splits, and open-source baselines for three tasks: nonlinear system identification, closed-loop controller benchmarking, and onboard state estimation assessment. To our knowledge, it is the first public dataset to jointly provide actuator commands, controller internals, and estimator outputs with millimeter-accurate ground truth on a commercially available nano-scale aerial platform.
>
---
#### [new 058] ZeroWBC: Learning Natural Visuomotor Humanoid Control Directly from Human Egocentric Video
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人形机器人控制任务，旨在解决传统方法依赖昂贵数据和缺乏自然行为的问题。通过直接从人类视角视频学习控制策略，实现更自然的机器人交互。**

- **链接: [https://arxiv.org/pdf/2603.09170](https://arxiv.org/pdf/2603.09170)**

> **作者:** Haoran Yang; Jiacheng Bao; Yucheng Xin; Haoming Song; Yuyang Tian; Bin Zhao; Dong Wang; Xuelong Li
>
> **摘要:** Achieving versatile and naturalistic whole-body control for humanoid robot scene-interaction remains a significant challenge. While some recent works have demonstrated autonomous humanoid interactive control, they are constrained to rigid locomotion patterns and expensive teleoperation data collection, lacking the versatility to execute more human-like natural behaviors such as sitting or kicking. Furthermore, acquiring the necessary real robot teleoperation data is prohibitively expensive and time-consuming. To address these limitations, we introduce ZeroWBC, a novel framework that learns a natural humanoid visuomotor control policy directly from human egocentric videos, eliminating the need for large-scale robot teleoperation data and enabling natural humanoid robot scene-interaction control. Specifically, our approach first fine-tunes a Vision-Language Model (VLM) to predict future whole-body human motions based on text instructions and egocentric visual context, then these generated motions are retargeted to real robot joints and executed via our robust general motion tracking policy for humanoid whole-body control. Extensive experiments on the Unitree G1 humanoid robot demonstrate that our method outperforms baseline approaches in motion naturalness and versatility, successfully establishing a pipeline that eliminates teleoperation data collection overhead for whole-body humanoid control, offering a scalable and efficient paradigm for general humanoid whole-body control.
>
---
#### [new 059] From Flow to One Step: Real-Time Multi-Modal Trajectory Policies via Implicit Maximum Likelihood Estimation-based Distribution Distillation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操控任务，旨在解决生成式策略延迟高和分布崩溃问题。通过IMLE方法将CFM专家模型压缩为单步学生模型，提升控制频率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.09415](https://arxiv.org/pdf/2603.09415)**

> **作者:** Ju Dong; Liding Zhang; Lei Zhang; Yu Fu; Kaixin Bai; Zoltan-Csaba Marton; Zhenshan Bing; Zhaopeng Chen; Alois Christian Knoll; Jianwei Zhang
>
> **备注:** this https URL, 8 pages
>
> **摘要:** Generative policies based on diffusion and flow matching achieve strong performance in robotic manipulation by modeling multi-modal human demonstrations. However, their reliance on iterative Ordinary Differential Equation (ODE) integration introduces substantial latency, limiting high-frequency closed-loop control. Recent single-step acceleration methods alleviate this overhead but often exhibit distributional collapse, producing averaged trajectories that fail to execute coherent manipulation strategies. We propose a framework that distills a Conditional Flow Matching (CFM) expert into a fast single-step student via Implicit Maximum Likelihood Estimation (IMLE). A bi-directional Chamfer distance provides a set-level objective that promotes both mode coverage and fidelity, enabling preservation of the teacher multi-modal action distribution in a single forward pass. A unified perception encoder further integrates multi-view RGB, depth, point clouds, and proprioception into a geometry-aware representation. The resulting high-frequency control supports real-time receding-horizon re-planning and improved robustness under dynamic disturbances.
>
---
#### [new 060] Impact of Different Failures on a Robot's Perceived Reliability
- **分类: cs.RO**

- **简介: 该论文属于人机交互领域，研究不同故障对机器人可信度的影响及恢复机制。通过实验分析故障类型对感知可靠性的作用，探讨无需修复动作的成功执行如何恢复信任。**

- **链接: [https://arxiv.org/pdf/2603.08821](https://arxiv.org/pdf/2603.08821)**

> **作者:** Andrew Violette; Zhanxin Wu; Haruki Nishimura; Masha Itkina; Leticia Priebe Rocha; Mark Zolotas; Guy Hoffman; Hadas Kress-Gazit
>
> **备注:** Accepted to ICRA 2026. 8 pages, 6 figures
>
> **摘要:** Robots fail, potentially leading to a loss in the robot's perceived reliability (PR), a measure correlated with trustworthiness. In this study we examine how various kinds of failures affect the PR of the robot differently, and how this measure recovers without explicit social repair actions by the robot. In a preregistered and controlled online video study, participants were asked to predict a robot's success in a pick-and-place task. We examined manipulation failures (slips), freezing (lapses), and three types of incorrect picked objects or place goals (mistakes). Participants were shown one of 11 videos -- one of five types of failure, one of five types of failure followed by a successful execution in the same video, or a successful execution video. This was followed by two additional successful execution videos. Participants bet money either on the robot or on a coin toss after each video. People's betting patterns along with a qualitative analysis of their survey responses highlight that mistakes are less damaging to PR than slips or lapses, and some mistakes are even perceived as successes. We also see that successes immediately following a failure have the same effect on PR as successes without a preceding failure. Finally, we show that successful executions recover PR after a failure. Our findings highlight which robot failures are in higher need of repair in a human-robot interaction, and how trust could be recovered by robot successes.
>
---
#### [new 061] Embodied Human Simulation for Quantitative Design and Analysis of Interactive Robotics
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于交互机器人设计任务，旨在解决人机互动动态评估难题。通过构建人体肌肉骨骼模型与强化学习控制结合的仿真框架，实现人机系统协同优化。**

- **链接: [https://arxiv.org/pdf/2603.09218](https://arxiv.org/pdf/2603.09218)**

> **作者:** Chenhui Zuo; Jinhao Xu; Michael Qian Vergnolle; Yanan Sui
>
> **摘要:** Physical interactive robotics, ranging from wearable devices to collaborative humanoid robots, require close coordination between mechanical design and control. However, evaluating interactive dynamics is challenging due to complex human biomechanics and motor responses. Traditional experiments rely on indirect metrics without measuring human internal states, such as muscle forces or joint loads. To address this issue, we develop a scalable simulation-based framework for the quantitative analysis of physical human-robot interaction. At its core is a full-body musculoskeletal model serving as a predictive surrogate for the human dynamical system. Driven by a reinforcement learning controller, it generates adaptive, physiologically grounded motor behaviors. We employ a sequential training pipeline where the pre-trained human motion control policy acts as a consistent evaluator, making large-scale design space exploration computationally tractable. By simulating the coupled human-robot system, the framework provides access to internal biomechanical metrics, offering a systematic way to concurrently co-optimize a robot's structural parameters and control policy. We demonstrate its capability in optimizing human-exoskeleton interactions, showing improved joint alignment and reduced contact forces. This work establishes embodied human simulation as a scalable paradigm for interactive robotics design.
>
---
#### [new 062] PlayWorld: Learning Robot World Models from Autonomous Play
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出PlayWorld，用于训练高保真机器人世界模型。任务是提升机器人操作中的物理交互预测能力，解决现有模型在复杂物理互动上的不足。通过自主玩耍数据训练，提高预测准确性和政策性能。**

- **链接: [https://arxiv.org/pdf/2603.09030](https://arxiv.org/pdf/2603.09030)**

> **作者:** Tenny Yin; Zhiting Mei; Zhonghe Zheng; Miyu Yamane; David Wang; Jade Sceats; Samuel M. Bateman; Lihan Zha; Apurva Badithela; Ola Shorinwa; Anirudha Majumdar
>
> **备注:** this https URL
>
> **摘要:** Action-conditioned video models offer a promising path to building general-purpose robot simulators that can improve directly from data. Yet, despite training on large-scale robot datasets, current state-of-the-art video models still struggle to predict physically consistent robot-object interactions that are crucial in robotic manipulation. To close this gap, we present PlayWorld, a simple, scalable, and fully autonomous pipeline for training high-fidelity video world simulators from interaction experience. In contrast to prior approaches that rely on success-biased human demonstrations, PlayWorld is the first system capable of learning entirely from unsupervised robot self-play, enabling naturally scalable data collection while capturing complex, long-tailed physical interactions essential for modeling realistic object dynamics. Experiments across diverse manipulation tasks show that PlayWorld generates high-quality, physically consistent predictions for contact-rich interactions that are not captured by world models trained on human-collected this http URL further demonstrate the versatility of PlayWorld in enabling fine-grained failure prediction and policy evaluation, with up to 40% improvements over human-collected data. Finally, we demonstrate how PlayWorld enables reinforcement learning in the world model, improving policy performance by 65% in success rates when deployed in the real world.
>
---
#### [new 063] FAME: Force-Adaptive RL for Expanding the Manipulation Envelope of a Full-Scale Humanoid
- **分类: cs.RO**

- **简介: 该论文属于人形机器人操作任务，旨在解决外部力作用下保持平衡的问题。提出FAME框架，通过强化学习适应力扰动，提升操作范围和稳定性。**

- **链接: [https://arxiv.org/pdf/2603.08961](https://arxiv.org/pdf/2603.08961)**

> **作者:** Niraj Pudasaini; Yutong Zhang; Jensen Lavering; Alessandro Roncone; Nikolaus Correll
>
> **摘要:** Maintaining balance under external hand forces is critical for humanoid bimanual manipulation, where interaction forces propagate through the kinematic chain and constrain the feasible manipulation envelope. We propose \textbf{FAME}, a force-adaptive reinforcement learning framework that conditions a standing policy on a learned latent context encoding upper-body joint configuration and bimanual interaction forces. During training, we apply diverse, spherically sampled 3D forces on each hand to inject disturbances in simulation together with an upper-body pose curriculum, exposing the policy to manipulation-induced perturbations across continuously varying arm configurations. At deployment, interaction forces are estimated from the robot dynamics and fed to the same encoder, enabling online adaptation without wrist force/torque sensors. In simulation across five fixed arm configurations with randomized hand forces and commanded base heights, FAME improves mean standing success to 73.84%, compared to 51.40% for the curriculum-only baseline and 29.44% for the base policy. We further deploy the learned policy on a full-scale Unitree H12 humanoid and evaluate robustness in representative load-interaction scenarios, including asymmetric single-arm load and symmetric bimanual load. Code and videos are available on this https URL
>
---
#### [new 064] See, Plan, Rewind: Progress-Aware Vision-Language-Action Models for Robust Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，解决任务进度感知问题。提出SPR框架，通过视觉-语言-动作结合，实现动态子目标规划与错误恢复。**

- **链接: [https://arxiv.org/pdf/2603.09292](https://arxiv.org/pdf/2603.09292)**

> **作者:** Tingjun Dai; Mingfei Han; Tingwen Du; Zhiheng Liu; Zhihui Li; Salman Khan; Jun Yu; Xiaojun Chang
>
> **备注:** Suggested to CVPR Findings. this https URL
>
> **摘要:** Measurement of task progress through explicit, actionable milestones is critical for robust robotic manipulation. This progress awareness enables a model to ground its current task status, anticipate verifiable intermediate states, and detect and recover from failures when progress stalls. To embody this capability, we introduce See, Plan, Rewind (SPR), a progress-aware vision-language-action framework that dynamically grounds language instructions into a sequence of spatial subgoals. SPR operates through a continuous core cycle, Seeing the current state and upcoming milestone, Planning a trajectory towards the next 2D waypoint, and Rewinding to a recoverable state upon failure by monitoring progress against the expected sequence. This closed-loop approach enables robust error correction without requiring additional training data or auxiliary models. Extensive experiments demonstrate the framework's effectiveness, generalization and robustness: SPR outperforms the MolmoAct baseline by 5\% on the LIBERO benchmark. On the challenging LIBERO-Plus benchmark with unseen instructions and initial states, SPR achieves state-of-the-art robustness with the smallest performance drop, surpassing OpenVLA-OFT and UniVLA, demonstrating superior out-of-distribution robustness.
>
---
#### [new 065] TiPToP: A Modular Open-Vocabulary Planning System for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出TiPToP，一个模块化系统，结合预训练视觉模型与任务规划器，解决机器人操作任务。旨在简化部署，无需机器人数据，提升多步骤操作能力。**

- **链接: [https://arxiv.org/pdf/2603.09971](https://arxiv.org/pdf/2603.09971)**

> **作者:** William Shen; Nishanth Kumar; Sahit Chintalapudi; Jie Wang; Christopher Watson; Edward Hu; Jing Cao; Dinesh Jayaraman; Leslie Pack Kaelbling; Tomás Lozano-Pérez
>
> **备注:** Project website: this https URL
>
> **摘要:** We present TiPToP, an extensible modular system that combines pretrained vision foundation models with an existing Task and Motion Planner (TAMP) to solve multi-step manipulation tasks directly from input RGB images and natural-language instructions. Our system aims to be simple and easy-to-use: it can be installed and run on a standard DROID setup in under one hour and adapted to new embodiments with minimal effort. We evaluate TiPToP -- which requires zero robot data -- over 28 tabletop manipulation tasks in simulation and the real world and find it matches or outperforms $\pi_{0.5}\text{-DROID}$, a vision-language-action (VLA) model fine-tuned on 350 hours of embodiment-specific demonstrations. TiPToP's modular architecture enables us to analyze the system's failure modes at the component level. We analyze results from an evaluation of 173 trials and identify directions for improvement. We release TiPToP open-source to further research on modular manipulation systems and tighter integration between learning and planning. Project website and code: this https URL
>
---
#### [new 066] OTPL-VIO: Robust Visual-Inertial Odometry with Optimal Transport Line Association and Adaptive Uncertainty
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出OTPL-VIO系统，解决低纹理和光照变化下的视觉惯性里程计问题，通过优化传输线匹配和自适应不确定性提升精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.09653](https://arxiv.org/pdf/2603.09653)**

> **作者:** Zikun Chen; Wentao Zhao; Yihe Niu; Tianchen Deng; Jingchuan Wang
>
> **摘要:** Robust stereo visual-inertial odometry (VIO) remains challenging in low-texture scenes and under abrupt illumination changes, where point features become sparse and unstable, leading to ambiguous association and under-constrained estimation. Line structures offer complementary geometric cues, yet many efficient point-line systems still rely on point-guided line association, which can break down when point support is weak and may lead to biased constraints. We present a stereo point-line VIO system in which line segments are equipped with dedicated deep descriptors and matched using an entropy-regularized optimal transport formulation, enabling globally consistent correspondences under ambiguity, outliers, and partial observations. The proposed descriptor is training-free and is computed by sampling and pooling network feature maps. To improve estimation stability, we analyze the impact of line measurement noise and introduce reliability-adaptive weighting to regulate the influence of line constraints during optimization. Experiments on EuRoC and UMA-VI, together with real-world deployments in low-texture and illumination-challenging environments, demonstrate improved accuracy and robustness over representative baselines while maintaining real-time performance.
>
---
#### [new 067] Implicit Geometry Representations for Vision-and-Language Navigation from Web Videos
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言导航任务，旨在解决传统数据集多样性不足的问题。通过引入网络视频构建大规模框架，并利用隐式几何表示提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.09259](https://arxiv.org/pdf/2603.09259)**

> **作者:** Mingfei Han; Haihong Hao; Liang Ma; Kamila Zhumakhanova; Ekaterina Radionova; Jingyi Zhang; Xiaojun Chang; Xiaodan Liang; Ivan Laptev
>
> **备注:** Extension of CVPR 2025 RoomTour3D with implicit geometric representations
>
> **摘要:** Vision-and-Language Navigation (VLN) has long been constrained by the limited diversity and scalability of simulator-curated datasets, which fail to capture the complexity of real-world environments. To overcome this limitation, we introduce a large-scale video-instruction framework derived from web-based room tour videos, enabling agents to learn from natural human walking demonstrations in diverse, realistic indoor settings. Unlike existing datasets, our framework integrates both open-ended description-enriched trajectories and action-enriched trajectories reconstructed in 3D, providing richer spatial and semantic supervision. A key extension in this work is the incorporation of implicit geometry representations, which extract spatial cues directly from RGB frames without requiring fragile 3D reconstruction. This approach substantially improves data utilization, alleviates reconstruction failures, and unlocks large portions of previously unusable video data. Comprehensive experiments across multiple VLN benchmarks (CVDN, SOON, R2R, and REVERIE) demonstrate that our method not only sets new state-of-the-art performance but also enables the development of robust zero-shot navigation agents. By bridging large-scale web videos with implicit spatial reasoning, this work advances embodied navigation towards more scalable, generalizable, and real-world applicable solutions.
>
---
#### [new 068] Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于文本导航任务，解决3D场景中实例定位问题。通过上下文驱动探索和视角感知的空间推理，提升导航准确性。**

- **链接: [https://arxiv.org/pdf/2603.09506](https://arxiv.org/pdf/2603.09506)**

> **作者:** Won Shik Jang; Ue-Hwan Kim
>
> **备注:** Camera-ready version. Accepted to CVPR 2026
>
> **摘要:** Text-goal instance navigation (TGIN) asks an agent to resolve a single, free-form description into actions that reach the correct object instance among same-category distractors. We present \textit{Context-Nav} that elevates long, contextual captions from a local matching cue to a global exploration prior and verifies candidates through 3D spatial reasoning. First, we compute dense text-image alignments for a value map that ranks frontiers -- guiding exploration toward regions consistent with the entire description rather than early detections. Second, upon observing a candidate, we perform a viewpoint-aware relation check: the agent samples plausible observer poses, aligns local frames, and accepts a target only if the spatial relations can be satisfied from at least one viewpoint. The pipeline requires no task-specific training or fine-tuning; we attain state-of-the-art performance on InstanceNav and CoIN-Bench. Ablations show that (i) encoding full captions into the value map avoids wasted motion and (ii) explicit, viewpoint-aware 3D verification prevents semantically plausible but incorrect stops. This suggests that geometry-grounded spatial reasoning is a scalable alternative to heavy policy training or human-in-the-loop interaction for fine-grained instance disambiguation in cluttered 3D scenes.
>
---
#### [new 069] $M^2$-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于自动驾驶中的3D语义占据预测任务，解决多摄像头输入不完整时的几何与语义一致性问题。提出$M^2$-Occ框架，通过多视角重建和特征记忆模块提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.09737](https://arxiv.org/pdf/2603.09737)**

> **作者:** Kaixin Lin; Kunyu Peng; Di Wen; Yufan Chen; Ruiping Liu; Kailun Yang
>
> **备注:** The source code will be publicly released at this https URL
>
> **摘要:** Semantic occupancy prediction enables dense 3D geometric and semantic understanding for autonomous driving. However, existing camera-based approaches implicitly assume complete surround-view observations, an assumption that rarely holds in real-world deployment due to occlusion, hardware malfunction, or communication failures. We study semantic occupancy prediction under incomplete multi-camera inputs and introduce $M^2$-Occ, a framework designed to preserve geometric structure and semantic coherence when views are missing. $M^2$-Occ addresses two complementary challenges. First, a Multi-view Masked Reconstruction (MMR) module leverages the spatial overlap among neighboring cameras to recover missing-view representations directly in the feature space. Second, a Feature Memory Module (FMM) introduces a learnable memory bank that stores class-level semantic prototypes. By retrieving and integrating these global priors, the FMM refines ambiguous voxel features, ensuring semantic consistency even when observational evidence is incomplete. We introduce a systematic missing-view evaluation protocol on the nuScenes-based SurroundOcc benchmark, encompassing both deterministic single-view failures and stochastic multi-view dropout scenarios. Under the safety-critical missing back-view setting, $M^2$-Occ improves the IoU by 4.93%. As the number of missing cameras increases, the robustness gap further widens; for instance, under the setting with five missing views, our method boosts the IoU by 5.01%. These gains are achieved without compromising full-view performance. The source code will be publicly released at this https URL.
>
---
#### [new 070] Efficient and robust control with spikes that constrain free energy
- **分类: q-bio.NC; cs.RO; eess.SY**

- **简介: 该论文属于控制理论任务，旨在解决高效且鲁棒的控制问题。提出一种基于自由能约束的脉冲控制框架，利用生物特性实现高效、抗扰的控制算法。**

- **链接: [https://arxiv.org/pdf/2603.09729](https://arxiv.org/pdf/2603.09729)**

> **作者:** André Urbano; Pablo Lanillos; Sander Keemink
>
> **摘要:** Animal brains exhibit remarkable efficiency in perception and action, while being robust to both external and internal perturbations. The means by which brains accomplish this remains, for now, poorly understood, hindering our understanding of animal and human cognition, as well as our own implementation of efficient algorithms for control of dynamical systems.A potential candidate for a robust mechanism of state estimation and action computation is the free energy principle, but existing implementations of this principle have largely relied on conventional, biologically implausible approaches without spikes. We propose a novel, efficient, and robust spiking control framework with realistic biological characteristics. The resulting networks function as free energy constrainers, in which neurons only fire if they reduce the free energy of their internal representation. The networks offer efficient operation through highly sparse activity while matching performance with other similar spiking frameworks, and have high resilience against both external (e.g. sensory noise or collisions) and internal perturbations (e.g. synaptic noise and delays or neuron silencing) that such a network would be faced with when deployed by either an organism or an engineer. Overall, our work provides a novel mathematical account for spiking control through constraining free energy, providing both better insight into how brain networks might leverage their spiking substrate and a new route for implementing efficient control algorithms in neuromorphic hardware.
>
---
#### [new 071] Open-World Motion Forecasting
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于运动预测任务，解决真实场景中对象类别动态变化带来的挑战。提出一种端到端增量学习框架，有效应对灾难性遗忘，提升新类别的适应能力。**

- **链接: [https://arxiv.org/pdf/2603.09420](https://arxiv.org/pdf/2603.09420)**

> **作者:** Nicolas Schischka; Nikhil Gosala; B Ravi Kiran; Senthil Yogamani; Abhinav Valada
>
> **摘要:** Motion forecasting aims to predict the future trajectories of dynamic agents in the scene, enabling autonomous vehicles to effectively reason about scene evolution. Existing approaches operate under the closed-world regime and assume fixed object taxonomy as well as access to high-quality perception. Therefore, they struggle in real-world settings where perception is imperfect and object taxonomy evolves over time. In this work, we bridge this fundamental gap by introducing open-world motion forecasting, a novel setting in which new object classes are sequentially introduced over time and future object trajectories are estimated directly from camera images. We tackle this setting by proposing the first end-to-end class-incremental motion forecasting framework to mitigate catastrophic forgetting while simultaneously learning to forecast newly introduced classes. When a new class is introduced, our framework employs a pseudo-labeling strategy to first generate motion forecasting pseudo-labels for all known classes which are then processed by a vision-language model to filter inconsistent and over-confident predictions. Parallelly, our approach further mitigates catastrophic forgetting by using a novel replay sampling strategy that leverages query feature variance to sample previous sequences with informative motion patterns. Extensive evaluation on the nuScenes and Argoverse 2 datasets demonstrates that our approach successfully resists catastrophic forgetting and maintains performance on previously learned classes while improving adaptation to novel ones. Further, we demonstrate that our approach supports zero-shot transfer to real-world driving and naturally extends to end-to-end class-incremental planning, enabling continual adaptation of the full autonomous driving system. We provide the code at this https URL .
>
---
#### [new 072] Why Channel-Centric Models are not Enough to Predict End-to-End Performance in Private 5G: A Measurement Campaign and Case Study
- **分类: cs.NI; cs.LG; cs.RO**

- **简介: 该论文属于通信与机器人规划任务，旨在解决5G网络吞吐量预测不准确的问题。通过测量和模型对比，发现通道级指标不足以预测实际性能，提出数据驱动方法提升预测精度。**

- **链接: [https://arxiv.org/pdf/2603.08865](https://arxiv.org/pdf/2603.08865)**

> **作者:** Nils Jörgensen
>
> **摘要:** Communication-aware robot planning requires accurate predictions of wireless network performance. Current approaches rely on channel-level metrics such as received signal strength and signal-to-noise ratio, assuming these translate reliably into end-to-end throughput. We challenge this assumption through a measurement campaign in a private 5G industrial environment. We evaluate throughput predictions from a commercial ray-tracing simulator as well as data-driven Gaussian process regression models against measurements collected using a mobile robot. The study uses off-the-shelf user equipment in an underground, radio-shielded facility with detailed 3D modeling, representing a best-case scenario for prediction accuracy. The ray-tracing simulator captures the spatial structure of indoor propagation and predicts channel-level metrics with reasonable fidelity. However, it systematically over-predicts throughput, even in line-of-sight regions. The dominant error source is shown to be over-estimation of sustainable MIMO spatial layers: the simulator assumes near-uniform four-layer transmission while measurements reveal substantial adaptation between one and three layers. This mismatch inflates predicted throughput even when channel metrics appear accurate. In contrast, a Gaussian process model with a rational quadratic kernel achieves approximately two-thirds reduction in prediction error with near-zero bias by learning end-to-end throughput directly from measurements. These findings demonstrate that favorable channel conditions do not guarantee high throughput; communication-aware planners relying solely on channel-centric predictions risk overly optimistic trajectories that violate reliability requirements. Accurate throughput prediction for 5G systems requires either extensive calibration of link-layer models or data-driven approaches that capture real system behavior.
>
---
#### [new 073] SPAARS: Safer RL Policy Alignment through Abstract Exploration and Refined Exploitation of Action Space
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决离线到在线策略对齐中的安全探索问题。提出SPAARS框架，通过潜在空间探索和动作空间精炼，提升样本效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.09378](https://arxiv.org/pdf/2603.09378)**

> **作者:** Swaminathan S K; Aritra Hazra
>
> **备注:** 9 pages
>
> **摘要:** Offline-to-online reinforcement learning (RL) offers a promising paradigm for robotics by pre-training policies on safe, offline demonstrations and fine-tuning them via online interaction. However, a fundamental challenge remains: how to safely explore online without deviating from the behavioral support of the offline data? While recent methods leverage conditional variational autoencoders (CVAEs) to bound exploration within a latent space, they inherently suffer from an exploitation gap -- a performance ceiling imposed by the decoder's reconstruction loss. We introduce SPAARS, a curriculum learning framework that initially constrains exploration to the low-dimensional latent manifold for sample-efficient, safe behavioral improvement, then seamlessly transfers control to the raw action space, bypassing the decoder bottleneck. SPAARS has two instantiations: the CVAE-based variant requires only unordered (s,a) pairs and no trajectory segmentation; SPAARS-SUPE pairs SPAARS with OPAL temporal skill pretraining for stronger exploration structure at the cost of requiring trajectory chunks. We prove an upper bound on the exploitation gap using the Performance Difference Lemma, establish that latent-space policy gradients achieve provable variance reduction over raw-space exploration, and show that concurrent behavioral cloning during the latent phase directly controls curriculum transition stability. Empirically, SPAARS-SUPE achieves 0.825 normalized return on kitchen-mixed-v0 versus 0.75 for SUPE, with 5x better sample efficiency; standalone SPAARS achieves 92.7 and 102.9 normalized return on hopper-medium-v2 and walker2d-medium-v2 respectively, surpassing IQL baselines of 66.3 and 78.3 respectively, confirming the utility of the unordered-pair CVAE instantiation.
>
---
#### [new 074] RAE-NWM: Navigation World Model in Dense Visual Representation Space
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉导航任务，旨在解决世界模型在压缩空间中丢失结构信息的问题。提出RAE-NWM，在密集视觉表示空间建模导航动态，提升导航精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.09241](https://arxiv.org/pdf/2603.09241)**

> **作者:** Mingkun Zhang; Wangtian Shen; Fan Zhang; Haijian Qin; Zihao Pei; Ziyang Meng
>
> **备注:** Code is available at: this https URL
>
> **摘要:** Visual navigation requires agents to reach goals in complex environments through perception and planning. World models address this task by simulating action-conditioned state transitions to predict future observations. Current navigation world models typically learn state evolution under actions within the compressed latent space of a Variational Autoencoder, where spatial compression often discards fine-grained structural information and hinders precise control. To better understand the propagation characteristics of different representations, we conduct a linear dynamics probe and observe that dense DINOv2 features exhibit stronger linear predictability for action-conditioned transitions. Motivated by this observation, we propose the Representation Autoencoder-based Navigation World Model (RAE-NWM), which models navigation dynamics in a dense visual representation space. We employ a Conditional Diffusion Transformer with Decoupled Diffusion Transformer head (CDiT-DH) to model continuous transitions, and introduce a separate time-driven gating module for dynamics conditioning to regulate action injection strength during generation. Extensive evaluations show that modeling sequential rollouts in this space improves structural stability and action accuracy, benefiting downstream planning and navigation.
>
---
#### [new 075] PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出PanoAffordanceNet，解决360°室内环境中的整体可操作性定位问题，通过新框架和数据集提升场景感知能力。**

- **链接: [https://arxiv.org/pdf/2603.09760](https://arxiv.org/pdf/2603.09760)**

> **作者:** Guoliang Zhu; Wanjun Jia; Caoyang Shao; Yuheng Zhang; Zhiyong Li; Kailun Yang
>
> **备注:** The source code and benchmark dataset will be made publicly available at this https URL
>
> **摘要:** Global perception is essential for embodied agents in 360° spaces, yet current affordance grounding remains largely object-centric and restricted to perspective views. To bridge this gap, we introduce a novel task: Holistic Affordance Grounding in 360° Indoor Environments. This task faces unique challenges, including severe geometric distortions from Equirectangular Projection (ERP), semantic dispersion, and cross-scale alignment difficulties. We propose PanoAffordanceNet, an end-to-end framework featuring a Distortion-Aware Spectral Modulator (DASM) for latitude-dependent calibration and an Omni-Spherical Densification Head (OSDH) to restore topological continuity from sparse activations. By integrating multi-level constraints comprising pixel-wise, distributional, and region-text contrastive objectives, our framework effectively suppresses semantic drift under low supervision. Furthermore, we construct 360-AGD, the first high-quality panoramic affordance grounding dataset. Extensive experiments demonstrate that PanoAffordanceNet significantly outperforms existing methods, establishing a solid baseline for scene-level perception in embodied intelligence. The source code and benchmark dataset will be made publicly available at this https URL.
>
---
#### [new 076] SPREAD: Subspace Representation Distillation for Lifelong Imitation Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于终身模仿学习任务，旨在解决知识遗忘问题。通过几何保持的子空间对齐和置信度引导的蒸馏策略，提升模型的稳定性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.08763](https://arxiv.org/pdf/2603.08763)**

> **作者:** Kaushik Roy; Giovanni D'urso; Nicholas Lawrance; Brendan Tidd; Peyman Moghadam
>
> **备注:** IEEE International Conference on Robotics & Automation (ICRA) 2026
>
> **摘要:** A key challenge in lifelong imitation learning (LIL) is enabling agents to acquire new skills from expert demonstrations while retaining prior knowledge. This requires preserving the low-dimensional manifolds and geometric structures that underlie task representations across sequential learning. Existing distillation methods, which rely on L2-norm feature matching in raw feature space, are sensitive to noise and high-dimensional variability, often failing to preserve intrinsic task manifolds. To address this, we introduce SPREAD, a geometry-preserving framework that employs singular value decomposition (SVD) to align policy representations across tasks within low-rank subspaces. This alignment maintains the underlying geometry of multimodal features, facilitating stable transfer, robustness, and generalization. Additionally, we propose a confidence-guided distillation strategy that applies a Kullback-Leibler divergence loss restricted to the top-M most confident action samples, emphasizing reliable modes and improving optimization stability. Experiments on the LIBERO, lifelong imitation learning benchmark, show that SPREAD substantially improves knowledge transfer, mitigates catastrophic forgetting, and achieves state-of-the-art performance.
>
---
#### [new 077] GST-VLA: Structured Gaussian Spatial Tokens for 3D Depth-Aware Vision-Language-Action Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出GST-VLA模型，解决3D视觉-语言-动作任务中的几何结构缺失问题，通过引入3D高斯令牌和深度感知推理，提升动作精度。**

- **链接: [https://arxiv.org/pdf/2603.09079](https://arxiv.org/pdf/2603.09079)**

> **作者:** Md Selim Sarowar; Omer Tariq; Sungho Kim
>
> **备注:** The results presented in this paper are preliminary. Please note that the experiments are currently ongoing, and the final data is subject to change upon the completion of the study. All ideas, results, methods, and any content herein are the sole property of the authors
>
> **摘要:** VLA models encode visual observations as 2D patch tokens with no intrinsic geometric structure. We introduce GST-VLA with two contributions. First, the Gaussian Spatial Tokenizer (GST) converts frozen dense depth and frozen semantic patch features into $N_g{=}128$ anisotropic 3D Gaussian primitives, each parameterized by a metric residual mean $\mu \in \mathbb{R}^3$, log-scale covariance $\log \sigma \in \mathbb{R}^3$, and learned opacity $\alpha \in (0,1)$. The covariance eigenstructure encodes local surface orientation, and opacity provides per-primitive geometric confidence, both inaccessible from scalar depth. Spatial attention pooling with learned queries concentrates the fixed token budget on geometrically salient regions rather than distributing uniformly. Second, 3D Depth-Aware Chain-of-Thought (DA-CoT) reasoning supervises four structured intermediate spatial thoughts, covering 3D object grounding, grasp affordance contact geometry, pairwise metric distances, and coarse SE(3) waypoints, as explicit generation targets in the training loss. A cross-attention sublayer at every VLM transformer block provides direct access to the raw 256-primitive Gaussian field during DA-CoT generation. A 300M-parameter flow-matching action expert with mixture-of-experts feedforward sublayers decodes 7-DoF delta action chunks via conditional ODE integration, conditioned on both VLM hidden states and DA-CoT outputs through dual cross-attention. Trained with composite $\mathcal{L}_\mathrm{flow} + \mathcal{L}_\mathrm{CoT} + \mathcal{L}_\mathrm{depth}$ across three progressive stages, GST-VLA achieves 96.4% on LIBERO (+2.0%), and 80.2% on SimplerEnv (+5.4%). Ablations isolate the contribution of each GST component, each DA-CoT thought, and each training stage, confirming independent and synergistic gains concentrated on precision demanding tasks.
>
---
## 更新

#### [replaced 001] Bootstrap Dynamic-Aware 3D Visual Representation for Scalable Robot Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，旨在解决3D视觉预训练在机械操作中表现不佳的问题。提出AFRO框架，通过动态感知的3D表示学习提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2512.00074](https://arxiv.org/pdf/2512.00074)**

> **作者:** Qiwei Liang; Boyang Cai; Minghao Lai; Sitong Zhuang; Tao Lin; Yan Qin; Yixuan Ye; Jiaming Liang; Renjing Xu
>
> **备注:** Project Page: this https URL, accepted by CVPR 2026
>
> **摘要:** Despite strong results on recognition and segmentation, current 3D visual pre-training methods often underperform on robotic manipulation. We attribute this gap to two factors: the lack of state-action-state dynamics modeling and the unnecessary redundancy of explicit geometric reconstruction. We introduce AFRO, a self-supervised framework that learns dynamics-aware 3D representations without action or reconstruction supervision. AFRO casts state prediction as a generative diffusion process and jointly models forward and inverse dynamics in a shared latent space to capture causal transition structure. To prevent feature leakage in action learning, we employ feature differencing and inverse-consistency supervision, improving the quality and stability of visual features. When combined with Diffusion Policy, AFRO substantially increases manipulation success rates across 16 simulated and 4 real-world tasks, outperforming existing pre-training approaches. The framework also scales favorably with data volume and task complexity. Qualitative visualizations indicate that AFRO learns semantically rich, discriminative features, offering an effective pre-training solution for 3D representation learning in robotics. Project page: this https URL
>
---
#### [replaced 002] Automated Coral Spawn Monitoring for Reef Restoration: The Coral Spawn and Larvae Imaging Camera System (CSLICS)
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决珊瑚繁殖监测效率低的问题。通过开发CSLICS系统，实现自动化 spawn 计数，提升珊瑚养殖效率。**

- **链接: [https://arxiv.org/pdf/2509.17299](https://arxiv.org/pdf/2509.17299)**

> **作者:** Dorian Tsai; Christopher A. Brunner; Riki Lamont; F. Mikaela Nordborg; Andrea Severati; Java Terry; Karen Jackel; Matthew Dunbabin; Tobias Fischer; Scarlett Raine
>
> **备注:** 8 pages, 7 figures, accepted for presentation at the IEEE International Conference on Robotics and Automation, 2026
>
> **摘要:** Coral aquaculture for reef restoration requires accurate and continuous spawn counting for resource distribution and larval health monitoring, but current methods are labor-intensive and represent a critical bottleneck in the coral production pipeline. We propose the Coral Spawn and Larvae Imaging Camera System (CSLICS), which uses low cost modular cameras and object detectors trained using human-in-the-loop labeling approaches for automated spawn counting in larval rearing tanks. This paper details the system engineering, dataset collection, and computer vision techniques to detect, classify and count coral spawn. Experimental results from mass spawning events demonstrate an F1 score of 82.4% for surface spawn detection at different embryogenesis stages, 65.3% F1 score for sub-surface spawn detection, and a saving of 5,720 hours of labor per spawning event compared to manual sampling methods at the same frequency. Comparison of manual counts with CSLICS monitoring during a mass coral spawning event on the Great Barrier Reef demonstrates CSLICS' accurate measurement of fertilization success and sub-surface spawn counts. These findings enhance the coral aquaculture process and enable upscaling of coral reef restoration efforts to address climate change threats facing ecosystems like the Great Barrier Reef.
>
---
#### [replaced 003] Magnetically Driven Elastic Microswimmers: Exploiting Hysteretic Collapse for Autonomous Propulsion and Independent Control
- **分类: cond-mat.soft; cs.RO; physics.app-ph; physics.flu-dyn; physics.med-ph**

- **简介: 该论文研究磁驱动弹性微泳器，解决低雷诺数下实现自主推进的问题。通过磁场调控实现非对称运动，实现独立控制与高效游泳。**

- **链接: [https://arxiv.org/pdf/2601.07370](https://arxiv.org/pdf/2601.07370)**

> **作者:** Theo Lequy; Andreas M. Menzel
>
> **备注:** 12 pages, 7 figures, submitted to ACS Nanoscience Au
>
> **摘要:** When swimming at low Reynolds numbers, inertial effects are negligible and reciprocal movements cannot induce net motion. Instead, symmetry breaking is necessary to achieve net propulsion. Directed swimming can be supported by magnetic fields, which simultaneously provide a versatile means of remote actuation. Thus, we analyze the motion of a straight microswimmer composed of three magnetizable beads connected by two elastic links. The swimming mechanism is based on oriented external magnetic fields that oscillate in magnitude. Through induced reversible hysteretic collapse of the two segments of the swimmer, the two pairs of beads jump into contact and separate nonreciprocally. Due to higher-order hydrodynamic interactions, net displacement results after each cycle. Different microswimmers can be tuned to different driving amplitudes and frequencies, allowing for simultaneous independent control by just one external magnetic field. The swimmer geometry and magnetic field shape are optimized for maximum swimming speed using an evolutionary optimization strategy. Thanks to the simple working principle, an experimental realization of such a microrobot seems feasible and may open new approaches for microinvasive medical interventions such as targeted drug delivery.
>
---
#### [replaced 004] Enhancing Heterogeneous Multi-Agent Cooperation in Decentralized MARL via GNN-driven Intrinsic Rewards
- **分类: cs.MA; cs.AI; cs.RO**

- **简介: 该论文属于多智能体强化学习任务，解决分布式环境下异质智能体协作问题，提出CoHet算法，利用GNN驱动的内在奖励提升合作效果。**

- **链接: [https://arxiv.org/pdf/2408.06503](https://arxiv.org/pdf/2408.06503)**

> **作者:** Jahir Sadik Monon; Deeparghya Dutta Barua; Md. Mosaddek Khan
>
> **备注:** Full paper version for AAMAS 2025, 9 pages, 5 figures
>
> **摘要:** Multi-agent Reinforcement Learning (MARL) is emerging as a key framework for various sequential decision-making and control tasks. Unlike their single-agent counterparts, multi-agent systems necessitate successful cooperation among the agents. The deployment of these systems in real-world scenarios often requires decentralized training, a diverse set of agents, and learning from infrequent environmental reward signals. These challenges become more pronounced under partial observability and the lack of prior knowledge about agent heterogeneity. While notable studies use intrinsic motivation (IM) to address reward sparsity or cooperation in decentralized settings, those dealing with heterogeneity typically assume centralized training, parameter sharing, and agent indexing. To overcome these limitations, we propose the CoHet algorithm, which utilizes a novel Graph Neural Network (GNN) based intrinsic motivation to facilitate the learning of heterogeneous agent policies in decentralized settings, under the challenges of partial observability and reward sparsity. Evaluation of CoHet in the Multi-agent Particle Environment (MPE) and Vectorized Multi-Agent Simulator (VMAS) benchmarks demonstrates superior performance compared to the state-of-the-art in a range of cooperative multi-agent scenarios. Our research is supplemented by an analysis of the impact of the agent dynamics model on the intrinsic motivation module, insights into the performance of different CoHet variants, and its robustness to an increasing number of heterogeneous agents.
>
---
#### [replaced 005] You Only Pose Once: A Minimalist's Detection Transformer for Monocular RGB Category-level 9D Multi-Object Pose Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于单目RGB图像的多目标位姿估计任务，旨在准确恢复未见实例的9自由度位姿。提出YOPO方法，通过统一检测与位姿估计，无需额外数据即可实现高精度。**

- **链接: [https://arxiv.org/pdf/2508.14965](https://arxiv.org/pdf/2508.14965)**

> **作者:** Hakjin Lee; Junghoon Seo; Jaehoon Sim
>
> **备注:** This paper has been accepted by IEEE ICRA 2026
>
> **摘要:** Accurately recovering the full 9-DoF pose of unseen instances within specific categories from a single RGB image remains a core challenge for robotics and automation. Most existing solutions still rely on pseudo-depth, CAD models, or multi-stage cascades that separate 2D detection from pose estimation. Motivated by the need for a simpler, RGB-only alternative that learns directly at the category level, we revisit a longstanding question: Can object detection and 9-DoF pose estimation be unified with high performance, without any additional data? We show that they can with our method, YOPO, a single-stage, query-based framework that treats category-level 9-DoF estimation as a natural extension of 2D detection. YOPO augments a transformer detector with a lightweight pose head, a bounding-box-conditioned translation module, and a 6D-aware Hungarian matching cost. The model is trained end-to-end only with RGB images and category-level pose labels. Despite its minimalist design, YOPO sets a new state of the art on three benchmarks. On the REAL275 dataset, it achieves 79.6% $\rm{IoU}_{50}$ and 54.1% under the $10^\circ$$10{\rm{cm}}$ metric, surpassing prior RGB-only methods and closing much of the gap to RGB-D systems. The code, models, and additional qualitative results can be found on this https URL.
>
---
#### [replaced 006] Morphological-Symmetry-Equivariant Heterogeneous Graph Neural Network for Robotic Dynamics Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出MS-HGNN，用于机器人动力学学习，整合运动结构和形态对称性，提升模型泛化与效率，解决多体系统动力学建模问题。**

- **链接: [https://arxiv.org/pdf/2412.01297](https://arxiv.org/pdf/2412.01297)**

> **作者:** Fengze Xie; Sizhe Wei; Yue Song; Yisong Yue; Lu Gan
>
> **摘要:** We present a morphological-symmetry-equivariant heterogeneous graph neural network, namely MS-HGNN, for robotic dynamics learning, that integrates robotic kinematic structures and morphological symmetries into a single graph network. These structural priors are embedded into the learning architecture as constraints, ensuring high generalizability, sample and model efficiency. The proposed MS-HGNN is a versatile and general architecture that is applicable to various multi-body dynamic systems and a wide range of dynamics learning problems. We formally prove the morphological-symmetry-equivariant property of our MS-HGNN and validate its effectiveness across multiple quadruped robot learning problems using both real-world and simulated data. Our code is made publicly available at this https URL.
>
---
#### [replaced 007] Asset-Centric Metric-Semantic Maps of Indoor Environments
- **分类: cs.RO**

- **简介: 该论文属于室内环境建模任务，旨在解决传统地图缺乏语义和细节的问题。提出一种基于物体的度量-语义地图，提升精度与效率，并支持语言模型推理与导航。**

- **链接: [https://arxiv.org/pdf/2510.10778](https://arxiv.org/pdf/2510.10778)**

> **作者:** Christopher D. Hsu; Pratik Chaudhari
>
> **备注:** 9 pages, 8 figures, 3 tables
>
> **摘要:** Large Language Models (LLMs) can help robots reason about abstract task specifications. This requires augmenting classical representations of the environment used by robots, such as point-clouds and meshes, with natural language-based priors. There are a number of approaches to do so in the existing literature. While some navigation frameworks leverage scene-level semantics at the expense of object-level detail, others such as language-guided neural radiance fields (NeRFs) or segment-anything 3D (SAM3D) prioritize object accuracy over global scene context. This paper argues that we can get the best of both worlds. We use a Unitree Go2 quadruped with a RealSense stereo camera (RGB-D data) to build an explicit metric-semantic representation of indoor environments. This is a scene-scale representation with each object (e.g., chairs, couches, doors, of various shapes and sizes) represented by a detailed mesh, its category, and a pose. We show that this representation is more accurate than foundation-model-based maps such as those built by SAM3D, as well as state-of-the-art scene-level robotics mapping pipelines such as Clio (Maggio et al., 2024). Our implementation is about 25$\times$ faster than SAM3D and is about 10$\times$ slower than Clio. We can also adapt our approach to enable open-set scene-level mapping, i.e., when object meshes are not known a priori, by building upon SAM3D to further improve precision and recall. We show how this representation can be readily used with LLMs such as Google's Gemini to demonstrate scene understanding, complex inferences, and planning. We also display the utility of having these representations for semantic navigation in simulated warehouse and hospital settings using Nvidia's Issac Sim.
>
---
#### [replaced 008] Unveiling the Potential of iMarkers: Invisible Fiducial Markers for Advanced Robotics
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉任务，旨在解决传统标记影响美观的问题。提出iMarkers，实现机器人和AR设备专用的不可见标记，提升应用适应性。**

- **链接: [https://arxiv.org/pdf/2501.15505](https://arxiv.org/pdf/2501.15505)**

> **作者:** Ali Tourani; Deniz Isinsu Avsar; Hriday Bavle; Jose Luis Sanchez-Lopez; Jan Lagerwall; Holger Voos
>
> **备注:** 19 pages, 10 figures, 4 tables
>
> **摘要:** Fiducial markers are widely used in robotics for navigation, object recognition, and scene understanding. While offering significant advantages for robots and Augmented Reality (AR) applications, they often disrupt the visual aesthetics of environments, as they are visible to humans, making them unsuitable for many everyday use cases. To address this gap, this paper presents iMarkers, innovative, unobtrusive fiducial markers detectable exclusively by robots and AR devices equipped with adequate sensors and detection algorithms. These markers offer high flexibility in production, allowing customization of their visibility range and encoding algorithms to suit various demands. The paper also introduces the hardware designs and open-sourced software algorithms developed for detecting iMarkers, highlighting their adaptability and robustness in the detection and recognition stages. Numerous evaluations have demonstrated the effectiveness of iMarkers relative to conventional (printed) and blended fiducial markers and have confirmed their applicability across diverse robotics scenarios.
>
---
#### [replaced 009] Open-World Task and Motion Planning via Vision-Language Model Genereated Constraints
- **分类: cs.RO**

- **简介: 该论文提出OWL-TAMP，将视觉语言模型与任务运动规划结合，解决长周期机器人操作问题，通过生成约束实现开放世界推理。**

- **链接: [https://arxiv.org/pdf/2411.08253](https://arxiv.org/pdf/2411.08253)**

> **作者:** Nishanth Kumar; William Shen; Fabio Ramos; Dieter Fox; Tomás Lozano-Pérez; Leslie Pack Kaelbling; Caelan Reed Garrett
>
> **备注:** A version of this paper appears in IEEE Robotics and Automation Letters (RA-L) Volume 11, Issue 3
>
> **摘要:** Foundation models like Vision-Language Models (VLMs) excel at common sense vision and language tasks such as visual question answering. However, they cannot yet directly solve complex, long-horizon robot manipulation problems requiring precise continuous reasoning. Task and Motion Planning (TAMP) systems can handle long-horizon reasoning through discrete-continuous hybrid search over parameterized skills, but rely on detailed environment models and cannot interpret novel human objectives, such as arbitrary natural language goals. We propose integrating VLMs into TAMP systems by having them generate discrete and continuous language-parameterized constraints that enable open-world reasoning. Specifically, we use VLMs to generate discrete action ordering constraints that constrain TAMP search over action sequences, and continuous constraints in the form of code that augments traditional TAMP manipulation constraints. Experiments show that our approach, OWL-TAMP, outperforms baselines relying solely on TAMP or VLMs across several long-horizon manipulation tasks specified directly in natural language. We additionally demonstrate that OWL-TAMP can be deployed with an off-the-shelf TAMP system to solve challenging manipulation tasks on real-world hardware.
>
---
#### [replaced 010] Learning responsibility allocations for multi-agent interactions: A differentiable optimization approach with control barrier functions
- **分类: eess.SY; cs.LG; cs.MA; cs.RO**

- **简介: 该论文属于多智能体交互任务，旨在解决安全与效率的平衡问题。通过责任分配建模，利用控制屏障函数和可微优化方法，从数据中学习智能体的行为调整机制。**

- **链接: [https://arxiv.org/pdf/2410.07409](https://arxiv.org/pdf/2410.07409)**

> **作者:** Isaac Remy; David Fridovich-Keil; Karen Leung
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** From autonomous driving to package delivery, ensuring safe yet efficient multi-agent interaction is challenging as the interaction dynamics are influenced by hard-to-model factors such as social norms and contextual cues. Understanding these influences can aid in the design and evaluation of socially-aware autonomous agents whose behaviors are aligned with human values. In this work, we seek to codify factors governing safe multi-agent interactions via the lens of responsibility, i.e., an agent's willingness to deviate from their desired control to accommodate safe interaction with others. Specifically, we propose a data-driven modeling approach based on control barrier functions and differentiable optimization that efficiently learns agents' responsibility allocation from data. We demonstrate on synthetic and real-world datasets that we can obtain an interpretable and quantitative understanding of how much agents adjust their behavior to ensure the safety of others given their current environment.
>
---
#### [replaced 011] EgoMI: Learning Active Vision and Whole-Body Manipulation from Egocentric Human Demonstrations
- **分类: cs.RO**

- **简介: 该论文属于机器人模仿学习任务，旨在解决人类与机器人在视角和动作协调上的差距。通过捕捉同步的头部和手部运动数据，提出EgoMI框架提升机器人模仿性能。**

- **链接: [https://arxiv.org/pdf/2511.00153](https://arxiv.org/pdf/2511.00153)**

> **作者:** Justin Yu; Yide Shentu; Di Wu; Pieter Abbeel; Ken Goldberg; Philipp Wu
>
> **摘要:** Imitation learning from human demonstrations offers a promising approach for robot skill acquisition, but egocentric human data introduces fundamental challenges due to the embodiment gap. During manipulation, humans actively coordinate head and hand movements, continuously reposition their viewpoint and use pre-action visual fixation search strategies to locate relevant objects. These behaviors create dynamic, task-driven head motions that static robot sensing systems cannot replicate, leading to a significant distribution shift that degrades policy performance. We present EgoMI (Egocentric Manipulation Interface), a framework that captures synchronized end-effector and active head trajectories during manipulation tasks, resulting in data that can be retargeted to compatible semi-humanoid robot embodiments. To handle rapid and wide-spanning head viewpoint changes, we introduce a memory-augmented policy that selectively incorporates historical observations. We evaluate our approach on a bimanual robot equipped with an actuated camera head and find that policies with explicit head-motion modeling consistently outperform baseline methods. Results suggest that coordinated hand-eye learning with EgoMI effectively bridges the human-robot embodiment gap for robust imitation learning on semi-humanoid embodiments. Project page: this https URL
>
---
#### [replaced 012] Reactive Slip Control in Multifingered Grasping: Hybrid Tactile Sensing and Internal-Force Optimization
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人抓取任务，解决多指抓取中的滑动问题。通过融合触觉传感与力优化，实现快速反应的内部力调整，提升抓取稳定性。**

- **链接: [https://arxiv.org/pdf/2602.16127](https://arxiv.org/pdf/2602.16127)**

> **作者:** Théo Ayral; Saifeddine Aloui; Mathieu Grossard
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA), 2026
>
> **摘要:** We present a hybrid learning and model-based approach for reactive internal-force adaptation to halt in-hand slip in a multifingered robotic gripper. A multimodal tactile stack combines piezoelectric (PzE) sensing for fast slip cues with piezoresistive (PzR) arrays for contact localization, enabling online construction of the grasp matrix. Upon slip detection, internal forces are updated in the null space of the grasp through a quadratic program that reinforces normal forces while preserving the object wrench. We demonstrate reactive stabilization of multifingered grasps under external perturbations. Augmenting analytic force control with learned tactile cues enables fast and reliable closed-loop stabilization in the evaluated grasp scenarios. The pipeline yields a theoretical sensing-to-command latency of 35-40 ms, including 5 ms for PzR-based grasp geometry updates and approximately 4 ms for solving the quadratic program. In controlled trials, slip onset is detected after ~ 20 ms. The analysis supports the feasibility of sub-50 ms integrated closed-loop stabilization.
>
---
#### [replaced 013] Vectorized Online POMDP Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人决策任务，解决部分可观测环境下的规划问题。提出VOPP方法，通过向量化计算实现高效并行POMDP求解，克服了传统方法的同步瓶颈。**

- **链接: [https://arxiv.org/pdf/2510.27191](https://arxiv.org/pdf/2510.27191)**

> **作者:** Marcus Hoerger; Muhammad Sudrajat; Hanna Kurniawati
>
> **备注:** 8 pages, 3 figures. Accepted at ICRA 2026
>
> **摘要:** Planning under partial observability is an essential capability of autonomous robots. The Partially Observable Markov Decision Process (POMDP) provides a powerful framework for planning under partial observability problems, capturing the stochastic effects of actions and the limited information available through noisy observations. POMDP solving could benefit tremendously from massive parallelization on today's hardware, but parallelizing POMDP solvers has been challenging. Most solvers rely on interleaving numerical optimization over actions with the estimation of their values, which creates dependencies and synchronization bottlenecks between parallel processes that can offset the benefits of parallelization. In this paper, we propose Vectorized Online POMDP Planner (VOPP), a novel parallel online solver that leverages a recent POMDP formulation which analytically solves part of the optimization component, leaving numerical computations to consist of only estimation of expectations. VOPP represents all data structures related to planning as a collection of tensors, and implements all planning steps as fully vectorized computations over this representation. The result is a massively parallel online solver with no dependencies or synchronization bottlenecks between concurrent processes. Experimental results indicate that VOPP is at least $20\times$ more efficient in computing near-optimal solutions compared to an existing state-of-the-art parallel online solver. Moreover, VOPP outperforms state-of-the-art sequential online solvers, while using a planning budget that is $1000\times$ smaller.
>
---
#### [replaced 014] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在提升扩散或流模型的策略性能。通过测试时的分布组合方法，无需额外训练即可增强策略表现。**

- **链接: [https://arxiv.org/pdf/2510.01068](https://arxiv.org/pdf/2510.01068)**

> **作者:** Jiahang Cao; Yize Huang; Hanzhong Guo; Rui Zhang; Mu Nan; Weijian Mai; Jiaxu Wang; Hao Cheng; Jingkai Sun; Gang Han; Wen Zhao; Qiang Zhang; Yijie Guo; Qihao Zheng; Chunfeng Song; Xiao Li; Ping Luo; Andrew F. Luo
>
> **备注:** Accepted to ICLR 2026. Project Page: this https URL
>
> **摘要:** Diffusion-based models for robotic control, including vision-language-action (VLA) and vision-action (VA) policies, have demonstrated significant capabilities. Yet their advancement is constrained by the high cost of acquiring large-scale interaction datasets. This work introduces an alternative paradigm for enhancing policy performance without additional model training. Perhaps surprisingly, we demonstrate that the composed policies can exceed the performance of either parent policy. Our contribution is threefold. First, we establish a theoretical foundation showing that the convex composition of distributional scores from multiple diffusion models can yield a superior one-step functional objective compared to any individual score. A Grönwall-type bound is then used to show that this single-step improvement propagates through entire generation trajectories, leading to systemic performance gains. Second, motivated by these results, we propose General Policy Composition (GPC), a training-free method that enhances performance by combining the distributional scores of multiple pre-trained policies via a convex combination and test-time search. GPC is versatile, allowing for the plug-and-play composition of heterogeneous policies, including VA and VLA models, as well as those based on diffusion or flow-matching, irrespective of their input visual modalities. Third, we provide extensive empirical validation. Experiments on Robomimic, PushT, and RoboTwin benchmarks, alongside real-world robotic evaluations, confirm that GPC consistently improves performance and adaptability across a diverse set of tasks. Further analysis of alternative composition operators and weighting strategies offers insights into the mechanisms underlying the success of GPC. These results establish GPC as a simple yet effective method for improving control performance by leveraging existing policies.
>
---
#### [replaced 015] RoboRouter: Training-Free Policy Routing for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出RoboRouter，用于机器人操作中的策略路由，解决单一策略泛化能力差的问题。通过智能选择最佳策略，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.07892](https://arxiv.org/pdf/2603.07892)**

> **作者:** Yiteng Chen; Zhe Cao; Hongjia Ren; Chenjie Yang; Wenbo Li; Shiyi Wang; Yemin Wang; Li Zhang; Yanming Shao; Zhenjun Zhao; Huiping Zhuang; Qingyao Wu
>
> **备注:** We need to withdraw the paper as some of the reference papers are incorrect and need to be removed
>
> **摘要:** Research on robotic manipulation has developed a diverse set of policy paradigms, including vision-language-action (VLA) models, vision-action (VA) policies, and code-based compositional approaches. Concrete policies typically attain high success rates on specific task distributions but lim-ited generalization beyond it. Rather than proposing an other monolithic policy, we propose to leverage the complementary strengths of existing approaches through intelligent policy routing. We introduce RoboRouter, a training-free framework that maintains a pool of heterogeneous policies and learns to select the best-performing policy for each task through accumulated execution experience. Given a new task, RoboRouter constructs a semantic task representation, retrieves historical records of similar tasks, predicts the optimal policy choice without requiring trial-and-error, and incorporates structured feedback to refine subsequent routing decisions. Integrating a new policy into the system requires only lightweight evaluation and incurs no training overhead. Across simulation benchmark and real-world evaluations, RoboRouter consistently outperforms than in-dividual policies, improving average success rate by more than 3% in simulation and over 13% in real-world settings, while preserving execution efficiency. Our results demonstrate that intelligent routing across heterogeneous, off-the-shelf policies provides a practical and scalable pathway toward building more capable robotic systems.
>
---
#### [replaced 016] Latent Policy Steering with Embodiment-Agnostic Pretrained World Models
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人视觉运动策略任务，解决数据不足与身体差异问题。通过预训练世界模型并微调，提升行为克隆策略性能。**

- **链接: [https://arxiv.org/pdf/2507.13340](https://arxiv.org/pdf/2507.13340)**

> **作者:** Yiqi Wang; Mrinal Verghese; Jeff Schneider
>
> **摘要:** The performance of learned robot visuomotor policies is heavily dependent on the size and quality of the training dataset. Although large-scale robot and human datasets are increasingly available, embodiment gaps and mismatched action spaces make them difficult to leverage. Our main insight is that skills performed across different embodiments produce visual similarities in motions that can be captured using off-the-shelf action representations such as optical flow. Moreover, World Models (WMs) can leverage sub-optimal data since they focus on modeling dynamics. In this work, we aim to improve visuomotor policies in low-data regimes by first pretraining a WM using optical flow as an embodiment-agnostic action representation to leverage accessible or easily collected data from multiple embodiments (robots, humans). Given a small set of demonstrations on a target embodiment, we finetune the WM on this data to better align the WM predictions, train a base policy, and learn a robust value function. Using our finetuned WM and value function, our approach evaluates action candidates from the base policy and selects the best one to improve performance. Our approach, which we term Latent Policy Steering (LPS), improves behavior-cloned policies by 10.6% on average across four Robomimic tasks, even though most of the pretraining data comes from the real world. In the real-world experiments, LPS achieves larger gains: 70% relative improvement with 30-50 target-embodiment demonstrations, and 44% relative improvement with 60-100 demonstrations, compared to a behavior-cloned baseline.
>
---
#### [replaced 017] RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出RL-100框架，解决现实世界机器人操作问题，通过强化学习实现高效、稳定的操作。**

- **链接: [https://arxiv.org/pdf/2510.14830](https://arxiv.org/pdf/2510.14830)**

> **作者:** Kun Lei; Huanyu Li; Dongjie Yu; Zhenyu Wei; Lingxiao Guo; Zhennan Jiang; Ziyu Wang; Shiyu Liang; Huazhe Xu
>
> **备注:** this https URL
>
> **摘要:** Real-world robotic manipulation in homes and factories demands reliability, efficiency, and robustness that approach or surpass those of skilled human operators. We present RL-100, a real-world reinforcement learning framework built on diffusion visuomotor policies. RL-100 unifies imitation and reinforcement learning under a single clipped PPO surrogate objective applied within the denoising process, yielding conservative and stable improvements across offline and online stages. To meet deployment latency requirements, a lightweight consistency distillation method compresses multi-step diffusion into a one-step controller for high-frequency control. The framework is task-, embodiment-, and representation-agnostic, and supports both single-action and action-chunking control. We evaluate RL-100 on eight diverse real-robot tasks, from dynamic pushing and agile bowling to pouring, cloth folding, unscrewing, multi-stage juicing, and long-horizon box folding. RL-100 attains 100 percent success across evaluated trials, for a total of 1000 out of 1000 episodes, including up to 250 out of 250 consecutive trials on one task. It matches or surpasses expert teleoperators in time to completion. Without retraining, a single policy attains approximately 90 percent zero-shot success under environmental and dynamics shifts, adapts in a few-shot regime to significant task variations (86.7 percent), and remains robust to aggressive human perturbations (about 96 percent). Notably, our juicing robot served random customers continuously for about seven hours without failure when deployed zero-shot in a shopping mall. These results suggest a practical path to deployment-ready robot learning by starting from human priors, aligning training objectives with human-grounded metrics, and reliably extending performance beyond human demonstrations.
>
---
#### [replaced 018] From Demonstrations to Safe Deployment: Path-Consistent Safety Filtering for Diffusion Policies
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人安全控制任务，旨在解决扩散策略在真实环境中安全性不足的问题。通过路径一致性安全过滤（PACS），确保执行安全且保持任务成功率。**

- **链接: [https://arxiv.org/pdf/2511.06385](https://arxiv.org/pdf/2511.06385)**

> **作者:** Ralf Römer; Julian Balletshofer; Jakob Thumm; Marco Pavone; Angela P. Schoellig; Matthias Althoff
>
> **备注:** Accepted to IEEE ICRA 2026. Project page: this https URL. 8 pages, 4 figures
>
> **摘要:** Diffusion policies (DPs) achieve state-of-the-art performance on complex manipulation tasks by learning from large-scale demonstration datasets, often spanning multiple embodiments and environments. However, they cannot guarantee safe behavior, requiring external safety mechanisms. These, however, alter actions in ways unseen during training, causing unpredictable behavior and performance degradation. To address these problems, we propose path-consistent safety filtering (PACS) for DPs. Our approach performs path-consistent braking on a trajectory computed from the sequence of generated actions. In this way, we keep the execution consistent with the training distribution of the policy, maintaining the learned, task-completing behavior. To enable real-time deployment and handle uncertainties, we verify safety using set-based reachability analysis. Our experimental evaluation in simulation and on three challenging real-world human-robot interaction tasks shows that PACS (a) provides formal safety guarantees in dynamic environments, (b) preserves task success rates, and (c) outperforms reactive safety approaches, such as control barrier functions, by up to 68 % in terms of task success. Videos are available at our project website: this https URL.
>
---
#### [replaced 019] Image Compression Using Novel View Synthesis Priors
- **分类: eess.IV; cs.CV; cs.RO**

- **简介: 该论文属于图像压缩任务，旨在解决水下实时图像传输带宽不足的问题。通过结合先验任务信息和深度学习方法，提升压缩效率与质量。**

- **链接: [https://arxiv.org/pdf/2411.13862](https://arxiv.org/pdf/2411.13862)**

> **作者:** Luyuan Peng; Mandar Chitre; Hari Vishnu; Yuen Min Too; Bharath Kalyan; Rajat Mishra; Soo Pieng Tan
>
> **备注:** Preprint submitted to IEEE Journal of Oceanic Engineering (v2.0)
>
> **摘要:** Real-time visual feedback is essential for tetherless control of remotely operated vehicles, particularly during inspection and manipulation tasks. Though acoustic communication is the preferred choice for medium-range communication underwater, its limited bandwidth renders it impractical to transmit images or videos in real-time. To address this, we propose a model-based image compression technique that leverages prior mission information. Our approach employs trained machine-learning based novel view synthesis models, and uses gradient descent optimization to refine latent representations to help generate compressible differences between camera images and rendered images. We evaluate the proposed compression technique using a dataset from an artificial ocean basin, demonstrating superior compression ratios and image quality over existing techniques. Moreover, our method exhibits robustness to introduction of new objects within the scene, highlighting its potential for advancing tetherless remotely operated vehicle operations.
>
---
#### [replaced 020] UniBYD: A Unified Framework for Learning Robotic Manipulation Across Embodiments Beyond Imitation of Human Demonstrations
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决机器人与人类手部形态差异导致的模仿学习难题。提出UniBYD框架，通过动态强化学习和统一形态表示，提升多样机器人形态的操控性能。**

- **链接: [https://arxiv.org/pdf/2512.11609](https://arxiv.org/pdf/2512.11609)**

> **作者:** Tingyu Yuan; Biaoliang Guan; Wen Ye; Ziyan Tian; Yi Yang; Weijie Zhou; Zhaowen Li; Yan Huang; Peng Wang; Chaoyang Zhao; Jinqiao Wang
>
> **摘要:** In embodied intelligence, the embodiment gap between robotic and human hands brings significant challenges for learning from human demonstrations. Although some studies have attempted to bridge this gap using reinforcement learning, they remain confined to merely reproducing human manipulation, resulting in limited task performance. Moreover, current methods struggle to support diverse robotic hand configurations. In this paper, we propose UniBYD, a unified framework that uses a dynamic reinforcement learning algorithm to discover manipulation policies aligned with the robot's physical characteristics. To enable consistent modeling across diverse robotic hand morphologies, UniBYD incorporates a unified morphological representation (UMR). Building on UMR, we design a dynamic PPO with an annealed reward schedule, enabling reinforcement learning to transition from offline-informed imitation of human demonstrations to online-adaptive exploration of policies better adapted to diverse robotic morphologies, thereby going beyond mere imitation of human hands. To address the severe state drift caused by the incapacity of early-stage policies, we design a hybrid Markov-based shadow engine that provides fine-grained guidance to anchor the imitation within the expert's manifold. To evaluate UniBYD, we propose UniManip, the first benchmark for cross-embodiment manipulation spanning diverse robotic morphologies. Experiments demonstrate a 44.08% average improvement in success rate over the current state-of-the-art. Upon acceptance, we will release our code and benchmark.
>
---
#### [replaced 021] From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言-动作（VLA）任务，旨在解决3D空间推理不足的问题。通过引入FALCON模型，注入丰富3D空间令牌，提升模型的泛化与适应能力。**

- **链接: [https://arxiv.org/pdf/2510.17439](https://arxiv.org/pdf/2510.17439)**

> **作者:** Zhengshen Zhang; Hao Li; Yalun Dai; Zhengbang Zhu; Lei Zhou; Chenchen Liu; Dong Wang; Francis E. H. Tay; Sijin Chen; Ziwei Liu; Yuxiao Liu; Xinghang Li; Pan Zhou
>
> **备注:** Accepted at ICLR 2026. Project page: this https URL
>
> **摘要:** Existing vision-language-action (VLA) models act in 3D real-world but are typically built on 2D encoders, leaving a spatial reasoning gap that limits generalization and adaptability. Recent 3D integration techniques for VLAs either require specialized sensors and transfer poorly across modalities, or inject weak cues that lack geometry and degrade vision-language alignment. In this work, we introduce FALCON (From Spatial to Action), a novel paradigm that injects rich 3D spatial tokens into the action head. FALCON leverages spatial foundation models to deliver strong geometric priors from RGB alone, and includes an Embodied Spatial Model that can optionally fuse depth, or pose for higher fidelity when available, without retraining or architectural changes. To preserve language reasoning, spatial tokens are consumed by a Spatial-Enhanced Action Head rather than being concatenated into the vision-language backbone. These designs enable FALCON to address limitations in spatial representation, modality transferability, and alignment. In comprehensive evaluations across three simulation benchmarks and eleven real-world tasks, our proposed FALCON achieves state-of-the-art performance, consistently surpasses competitive baselines, and remains robust under clutter, spatial-prompt conditioning, and variations in object scale and height.
>
---
#### [replaced 022] A 26-Gram Butterfly-Inspired Robot Achieving Autonomous Tailless Flight
- **分类: cs.RO**

- **简介: 该论文属于自主飞行机器人任务，旨在解决尾部缺失微型飞行器的稳定控制问题。研究设计了一款26克蝴蝶仿生机器人，实现自主飞行与稳定操控。**

- **链接: [https://arxiv.org/pdf/2602.06811](https://arxiv.org/pdf/2602.06811)**

> **作者:** Weibin Gu; Chenrui Feng; Lian Liu; Chen Yang; Xingchi Jiao; Yuhe Ding; Xiaofei Shi; Chao Gao; Alessandro Rizzo; Guyue Zhou
>
> **摘要:** The flight of biological butterflies represents a unique aerodynamic regime where high-amplitude, low-frequency wingstrokes induce significant body undulations and inertial fluctuations. While existing tailless flapping-wing micro air vehicles typically employ high-frequency kinematics to minimize such perturbations, the lepidopteran flight envelope remains a challenging and underexplored frontier for autonomous robotics. Here, we present \textit{AirPulse}, a 26-gram butterfly-inspired robot that achieves the first onboard, closed-loop controlled flight for a tailless two-winged platform at this scale. It replicates key biomechanical traits of butterfly flight, utilizing low-aspect-ratio, compliant carbon-fiber-reinforced wings and low-frequency flapping that reproduces characteristic biological body undulations. Leveraging a quantitative mapping of control effectiveness, we introduce a hierarchical control architecture featuring state estimator, attitude controller, and central pattern generator with Stroke Timing Asymmetry Rhythm (STAR), which translates attitude control demands into smooth and stable wingstroke timing and angle-offset modulations. Free-flight experiments demonstrate stable climbing and directed turning maneuvers, proving that autonomous locomotion is achievable even within oscillatory dynamical regimes. By bridging biological morphology with a minimalist control architecture, \textit{AirPulse} serves as both a hardware-validated model for decoding butterfly flight dynamics and a prototype for a new class of collision-resilient aerial robots. Its lightweight and compliant structure offers a non-invasive solution for a wide range of applications, such as ecological monitoring and confined-space inspection, where traditional drones may fall short.
>
---
#### [replaced 023] SPARC: Spatial-Aware Path Planning via Attentive Robot Communication
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多机器人路径规划任务，旨在解决通信效率低下问题。通过引入RMHA机制，增强机器人间通信的时空相关性，提升密集环境下的协作效果。**

- **链接: [https://arxiv.org/pdf/2603.02845](https://arxiv.org/pdf/2603.02845)**

> **作者:** Sayang Mu; Xiangyu Wu; Bo An
>
> **备注:** The manuscript is being withdrawn at the request of the first author for the purpose of revising content and re-uploading a revised version with updated data/figures/text . The revised manuscript will be resubmitted to arXiv promptly with the same author list and research theme
>
> **摘要:** Efficient communication is critical for decentralized Multi-Robot Path Planning (MRPP), yet existing learned communication methods treat all neighboring robots equally regardless of their spatial proximity, leading to diluted attention in congested regions where coordination matters most. We propose Relation enhanced Multi Head Attention (RMHA), a communication mechanism that explicitly embeds pairwise Manhattan distances into the attention weight computation, enabling each robot to dynamically prioritize messages from spatially relevant neighbors. Combined with a distance-constrained attention mask and GRU gated message fusion, RMHA integrates seamlessly with MAPPO for stable end-to-end training. In zero-shot generalization from 8 training robots to 128 test robots on 40x40 grids, RMHA achieves approximately 75 percent success rate at 30 percent obstacle density outperforming the best baseline by over 25 percentage points. Ablation studies confirm that distance-relation encoding is the key contributor to success rate improvement in high-density environments. Index Terms-Multi-robot path planning, graph attention mechanism, multi-head attention, communication optimization, cooperative decision-making
>
---
#### [replaced 024] Robot Control Stack: A Lean Ecosystem for Robot Learning at Scale
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人学习任务，旨在解决传统框架与大规模策略训练之间的瓶颈问题。提出Robot Control Stack（RCS），支持高效仿真到现实的迁移，提升政策性能。**

- **链接: [https://arxiv.org/pdf/2509.14932](https://arxiv.org/pdf/2509.14932)**

> **作者:** Tobias Jülg; Pierre Krack; Seongjin Bien; Yannik Blei; Khaled Gamal; Ken Nakahara; Johannes Hechtl; Roberto Calandra; Wolfram Burgard; Florian Walter
>
> **备注:** Accepted at ICRA 2026
>
> **摘要:** Vision-Language-Action models (VLAs) mark a major shift in robot learning. They replace specialized architectures and task-tailored components of expert policies with large-scale data collection and setup-specific fine-tuning. In this machine learning-focused workflow that is centered around models and scalable training, traditional robotics software frameworks become a bottleneck, while robot simulations offer only limited support for transitioning from and to real-world experiments. In this work, we close this gap by introducing Robot Control Stack (RCS), a lean ecosystem designed from the ground up to support research in robot learning with large-scale generalist policies. At its core, RCS features a modular and easily extensible layered architecture with a unified interface for simulated and physical robots, facilitating sim-to-real transfer. Despite its minimal footprint and dependencies, it offers a complete feature set, enabling both real-world experiments and large-scale training in simulation. Our contribution is twofold: First, we introduce the architecture of RCS and explain its design principles. Second, we evaluate its usability and performance along the development cycle of VLA and RL policies. Our experiments also provide an extensive evaluation of Octo, OpenVLA, and Pi Zero on multiple robots and shed light on how simulation data can improve real-world policy performance. Our code, datasets, weights, and videos are available at: this https URL
>
---
#### [replaced 025] NavSpace: How Navigation Agents Follow Spatial Intelligence Instructions
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出NavSpace基准，评估导航代理的空间智能。针对传统基准忽视空间感知的问题，设计任务集并测试多个模型，提出SNav模型提升导航性能。**

- **链接: [https://arxiv.org/pdf/2510.08173](https://arxiv.org/pdf/2510.08173)**

> **作者:** Haolin Yang; Yuxing Long; Zhuoyuan Yu; Zihan Yang; Minghan Wang; Jiapeng Xu; Yihan Wang; Ziyan Yu; Wenzhe Cai; Lei Kang; Hao Dong
>
> **备注:** ICRA 2026
>
> **摘要:** Instruction-following navigation is a key step toward embodied intelligence. Prior benchmarks mainly focus on semantic understanding but overlook systematically evaluating navigation agents' spatial perception and reasoning capabilities. In this work, we introduce the NavSpace benchmark, which contains six task categories and 1,228 trajectory-instruction pairs designed to probe the spatial intelligence of navigation agents. On this benchmark, we comprehensively evaluate 22 navigation agents, including state-of-the-art navigation models and multimodal large language models. The evaluation results lift the veil on spatial intelligence in embodied navigation. Furthermore, we propose SNav, a new spatially intelligent navigation model. SNav outperforms existing navigation agents on NavSpace and real robot tests, establishing a strong baseline for future work.
>
---
#### [replaced 026] Connectivity Maintenance and Recovery for Multi-Robot Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于多机器人路径规划任务，旨在解决复杂环境中保持连接性与导航能力的平衡问题。提出MPC-CLF-CBF算法，实现轨迹与控制协同生成，提升导航成功率并恢复连接。**

- **链接: [https://arxiv.org/pdf/2510.03504](https://arxiv.org/pdf/2510.03504)**

> **作者:** Yutong Wang; Lishuo Pan; Yichun Qu; Tengxiang Wang; Nora Ayanian
>
> **摘要:** Connectivity is crucial in many multi-robot applications, yet balancing between maintaining it and the fleet's traversability in obstacle-rich environments remains a challenge. Reactive controllers, such as control barrier functions, while providing connectivity guarantees, often struggle to traverse obstacle-rich environments due to deadlocks. We propose a real-time Bézier-based constrained motion planning algorithm, namely, MPC--CLF--CBF, that produces trajectory and control concurrently, under high-order control barrier functions and control Lyapunov functions conditions. Our motion planner significantly improves the navigation success rate of connected fleets in a cluttered workspace and recovers after inevitable connection loss by bypassing obstacles or from an initially disconnected fleet configuration. In addition, our predictive motion planner, owing to its Bézier curve solution, can easily obtain continuous-time arbitrary orders of derivatives, making it suitable for agile differentially flat systems, such as quadrotors. We validate the proposed algorithm through simulations and a physical experiment with $8$ Crazyflie nano-quadrotors.
>
---
#### [replaced 027] NaviGait: Navigating Dynamically Feasible Gait Libraries using Deep Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人步态控制任务，旨在解决RL奖励设计复杂和轨迹优化易受干扰的问题。提出NaviGait框架，结合轨迹优化与强化学习，实现高效、鲁棒的步行控制。**

- **链接: [https://arxiv.org/pdf/2510.11542](https://arxiv.org/pdf/2510.11542)**

> **作者:** Neil Janwani; Varun Madabushi; Maegan Tucker
>
> **备注:** Accepted to the International Conference on Robotics and Automation (2026). 8 pages, 9 figures
>
> **摘要:** Reinforcement learning (RL) has emerged as a powerful method to learn robust control policies for bipedal locomotion. Yet, it can be difficult to tune desired robot behaviors due to unintuitive and complex reward design. In comparison, trajectory optimization-based methods offer more tuneable, interpretable, and mathematically grounded motion plans for high-dimensional legged systems. However, these methods often remain brittle to real-world disturbances like external perturbations. In this work, we present NaviGait, a hierarchical framework that combines the structure of trajectory optimization with the adaptability of RL for robust and intuitive locomotion control. NaviGait leverages RL to synthesize new motions by selecting, minimally morphing, and stabilizing gaits taken from an offline-generated gait library. NaviGait results in walking policies that match the reference motion well while maintaining robustness comparable to other locomotion controllers. Additionally, the structure imposed by NaviGait drastically simplifies the RL reward composition. Our experimental results demonstrate that NaviGait enables faster training compared to conventional and imitation-based RL, and produces motions that remain closest to the original reference. Overall, by decoupling high-level motion generation from low-level correction, NaviGait offers a more scalable and generalizable approach for achieving dynamic and robust locomotion. Videos and the full framework are publicly available at this https URL
>
---
#### [replaced 028] A Distributional Treatment of Real2Sim2Real for Object-Centric Agent Adaptation in Vision-Driven Deformable Linear Object Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究视觉驱动的可变形线性物体操作任务，通过分布处理实现真实到模拟再到真实的迁移，解决现实环境中策略泛化问题。**

- **链接: [https://arxiv.org/pdf/2502.18615](https://arxiv.org/pdf/2502.18615)**

> **作者:** Georgios Kamaras; Subramanian Ramamoorthy
>
> **摘要:** We present an integrated (or end-to-end) framework for the Real2Sim2Real problem of manipulating deformable linear objects (DLOs) based on visual perception. Working with a parameterised set of DLOs, we use likelihood-free inference (LFI) to compute the posterior distributions for the physical parameters using which we can approximately simulate the behaviour of each specific DLO. We use these posteriors for domain randomisation while training, in simulation, object-specific visuomotor policies (i.e. assuming only visual and proprioceptive sensory) for a DLO reaching task, using model-free reinforcement learning. We demonstrate the utility of this approach by deploying sim-trained DLO manipulation policies in the real world in a zero-shot manner, i.e. without any further fine-tuning. In this context, we evaluate the capacity of a prominent LFI method to perform fine classification over the parametric set of DLOs, using only visual and proprioceptive data obtained in a dynamic manipulation trajectory. We then study the implications of the resulting domain distributions in sim-based policy learning and real-world performance.
>
---
#### [replaced 029] Pri4R: Learning World Dynamics for Vision-Language-Action Models with Privileged 4D Representation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出Pri4R，解决VLA模型缺乏物理动态理解的问题。通过引入4D信息，提升模型对场景变化的感知，增强控制精度。**

- **链接: [https://arxiv.org/pdf/2603.01549](https://arxiv.org/pdf/2603.01549)**

> **作者:** Jisoo Kim; Jungbin Cho; Sanghyeok Chu; Ananya Bal; Jinhyung Kim; Gunhee Lee; Sihaeng Lee; Seung Hwan Kim; Bohyung Han; Hyunmin Lee; Laszlo A. Jeni; Seungryong Kim
>
> **摘要:** Humans learn not only how their bodies move, but also how the surrounding world responds to their actions. In contrast, while recent Vision-Language-Action (VLA) models exhibit impressive semantic understanding, they often fail to capture the spatiotemporal dynamics governing physical interaction. In this paper, we introduce Pri4R, a simple yet effective approach that endows VLA models with an implicit understanding of world dynamics by leveraging privileged 4D information during training. Specifically, Pri4R augments VLAs with a lightweight point track head that predicts 3D point tracks. By injecting VLA features into this head to jointly predict future 3D trajectories, the model learns to incorporate evolving scene geometry within its shared representation space, enabling more physically aware context for precise control. Due to its architectural simplicity, Pri4R is compatible with dominant VLA design patterns with minimal changes. During inference, we run the model using the original VLA architecture unchanged; Pri4R adds no extra inputs, outputs, or computational overhead. Across simulation and real-world evaluations, Pri4R significantly improves performance on challenging manipulation tasks, including a +10% gain on LIBERO-Long and a +40% gain on RoboCasa. We further show that 3D point track prediction is an effective supervision target for learning action-world dynamics, and validate our design choices through extensive ablations. Project page: this https URL
>
---
#### [replaced 030] LLM-Advisor: An LLM Benchmark for Cost-efficient Path Planning across Multiple Terrains
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人路径规划任务，解决多地形下成本高效路径规划问题。提出LLM-Advisor框架，利用大语言模型优化路径成本，提升导航效率。**

- **链接: [https://arxiv.org/pdf/2503.01236](https://arxiv.org/pdf/2503.01236)**

> **作者:** Ling Xiao; Toshihiko Yamasaki
>
> **摘要:** Cost-efficient path planning across multiple terrains is a crucial task in robot navigation, requiring the identification of a path from the start to the goal that not only avoids obstacles but also minimizes the overall travel cost. This is especially crucial for real-world applications where robots need to navigate diverse terrains in outdoor environments with limited opportunities for recharging or refueling. Despite its practical importance, cost-efficient path planning across heterogeneous terrains has received relatively limited attention in prior work. In this paper, we propose LLM-Advisor, a prompt-based, planner-agnostic framework that leverages large language models (LLMs) as non-decisive post-processing advisors for cost refinement, without modifying the underlying planner. While we observe that LLMs may occasionally produce implausible suggestions, we introduce two effective hallucination-mitigation strategies. We further introduce two datasets, MultiTerraPath and RUGD_v2, for systematic evaluation of cost-efficient path planning. Extensive experiments reveal that state-of-the-art LLMs, including GPT-4o, GPT-4-turbo, Gemini-2.5-Flash, and Claude-Opus-4, perform poorly in zero-shot terrain-aware path planning, highlighting their limited spatial reasoning capability. In contrast, the proposed LLM-Advisor (with GPT-4o) improves cost efficiency for 72.37% of A*-planned paths, 69.47% of RRT*-planned paths, and 78.70% of LLM-A*-planned paths. On the MultiTerraPath dataset, LLM-Advisor demonstrates stronger performance on the hard subset, further validating its applicability to real-world scenarios.
>
---
#### [replaced 031] Physics-Conditioned Grasping for Stable Tool Use
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决工具使用中因握持不稳定导致失败的问题。通过引入iTuP和SDG-Net，提升抓取稳定性，减少扭矩影响。**

- **链接: [https://arxiv.org/pdf/2505.01399](https://arxiv.org/pdf/2505.01399)**

> **作者:** Noah Trupin; Zixing Wang; Ahmed H. Qureshi
>
> **备注:** In submission and under review
>
> **摘要:** Tool use often fails not because robots misidentify tools, but because grasps cannot withstand task-induced wrench. Existing vision-language manipulation systems ground tools and contact regions from language yet select grasps under quasi-static or geometry-only assumptions. During interaction, inertial impulse and lever-arm amplification generate wrist torque and tangential loads that trigger slip and rotation. We introduce inverse Tool-use Planning (iTuP), which selects grasps by minimizing predicted interaction wrench along a task-conditioned trajectory. From rigid-body mechanics, we derive torque, slip, and alignment penalties, and train a Stable Dynamic Grasp Network (SDG-Net) to approximate these trajectory-conditioned costs for real-time scoring. Across hammering, sweeping, knocking, and reaching in simulation and on hardware, SDG-Net suppresses induced torque up to 17.6%, shifts grasps below empirically observed instability thresholds, and improves real-world success by 17.5% over a compositional baseline. Improvements concentrate where wrench amplification dominates, showing that robot tool use requires wrench-aware grasp selection, not perception alone.
>
---
#### [replaced 032] Exploring Single Domain Generalization of LiDAR-based Semantic Segmentation under Imperfect Labels
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文研究LiDAR语义分割中的域泛化问题，针对标签噪声提出DuNe框架，提升模型在不同域下的性能。**

- **链接: [https://arxiv.org/pdf/2510.09035](https://arxiv.org/pdf/2510.09035)**

> **作者:** Weitong Kong; Zichao Zeng; Di Wen; Jiale Wei; Kunyu Peng; June Moh Goo; Jan Boehm; Rainer Stiefelhagen
>
> **摘要:** Accurate perception is critical for vehicle safety, with LiDAR as a key enabler in autonomous driving. To ensure robust performance across environments, sensor types, and weather conditions without costly re-annotation, domain generalization in LiDAR-based 3D semantic segmentation is essential. However, LiDAR annotations are often noisy due to sensor imperfections, occlusions, and human errors. Such noise degrades segmentation accuracy and is further amplified under domain shifts, threatening system reliability. While noisy-label learning is well-studied in images, its extension to 3D LiDAR segmentation under domain generalization remains largely unexplored, as the sparse and irregular structure of point clouds limits direct use of 2D methods. To address this gap, we introduce the novel task Domain Generalization for LiDAR Semantic Segmentation under Noisy Labels (DGLSS-NL) and establish the first benchmark by adapting three representative noisy-label learning strategies from image classification to 3D segmentation. However, we find that existing noisy-label learning approaches adapt poorly to LiDAR data. We therefore propose DuNe, a dual-view framework with strong and weak branches that enforce feature-level consistency and apply cross-entropy loss based on confidence-aware filtering of predictions. Our approach shows state-of-the-art performance by achieving 56.86% mIoU on SemanticKITTI, 42.28% on nuScenes, and 52.58% on SemanticPOSS under 10% symmetric label noise, with an overall Arithmetic Mean (AM) of 49.57% and Harmonic Mean (HM) of 48.50%, thereby demonstrating robust domain generalization in DGLSS-NL tasks. The code is available on our project page.
>
---
#### [replaced 033] StructBiHOI: Structured Articulation Modeling for Long--Horizon Bimanual Hand--Object Interaction Generation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多模态手-物交互生成任务，旨在解决双臂操作中长期规划不稳定、关节精细控制和跨手协调难题。提出StructBiHOI框架，通过分层结构实现稳定高效的双臂协同与物体交互。**

- **链接: [https://arxiv.org/pdf/2603.08390](https://arxiv.org/pdf/2603.08390)**

> **作者:** Zhi Wang; Liu Liu; Ruonan Liu; Dan Guo; Meng Wang
>
> **摘要:** Recent progress in 3D hand--object interaction (HOI) generation has primarily focused on single--hand grasp synthesis, while bimanual manipulation remains significantly more challenging. Long--horizon planning instability, fine--grained joint articulation, and complex cross--hand coordination make coherent bimanual generation difficult, especially under multimodal conditions. Existing approaches often struggle to simultaneously ensure temporal consistency, physical plausibility, and semantic alignment over extended sequences. We propose StructBiHOI, a Structured articulation modeling framework for long-horizon Bimanual HOI generation. Our key insight is to structurally disentangle temporal joint planning from frame--level manipulation refinement. Specifically, a jointVAE models long-term joint evolution conditioned on object geometry and task semantics, while a maniVAE refines fine-grained hand poses at the single--frame level. To enable stable and efficient long--sequence generation, we incorporate a state--space--inspired diffusion denoiser based on Mamba, which models long--range dependencies with linear complexity. This hierarchical design facilitates coherent dual-hand coordination and articulated object interaction. Extensive experiments on bimanual manipulation and single-hand grasping benchmarks demonstrate that our method achieves superior long--horizon stability, motion realism, and computational efficiency compared to strong baselines.
>
---
#### [replaced 034] Multi-Quadruped Cooperative Object Transport: Learning Decentralized Pinch-Lift-Move
- **分类: cs.RO**

- **简介: 该论文研究多足机器人协作搬运任务，解决无通信、无机械连接的物体协同运输问题。通过设计奖励机制和分层策略，实现机器人间隐式同步与协调。**

- **链接: [https://arxiv.org/pdf/2509.14342](https://arxiv.org/pdf/2509.14342)**

> **作者:** Bikram Pandit; Aayam Kumar Shrestha; Alan Fern
>
> **备注:** Accepted to ICRA 2026. Project page: this https URL
>
> **摘要:** We study decentralized cooperative transport using teams of N-quadruped robots with arm that must pinch, lift, and move ungraspable objects through physical contact alone. Unlike prior work that relies on rigid mechanical coupling between robots and objects, we address the more challenging setting where mechanically independent robots must coordinate through contact forces alone without any communication or centralized control. To this end, we employ a hierarchical policy architecture that separates base locomotion from arm control, and propose a constellation reward formulation that unifies position and orientation tracking to enforce rigid contact behavior. The key insight is encouraging robots to behave as if rigidly connected to the object through careful reward design and training curriculum rather than explicit mechanical constraints. Our approach enables coordination through shared policy parameters and implicit synchronization cues - scaling to arbitrary team sizes without retraining. We show extensive simulation experiments to demonstrate robust transport across 2-10 robots on diverse object geometries and masses, along with sim2real transfer results on lightweight objects.
>
---
#### [replaced 035] Multimodal Adversarial Quality Policy for Safe Grasping
- **分类: cs.RO**

- **简介: 该论文属于安全抓取任务，解决DNN在HRI中的安全风险问题。提出MAQP框架，通过双模态优化和梯度平衡策略提升RGBD模态下的抓取安全性。**

- **链接: [https://arxiv.org/pdf/2603.01479](https://arxiv.org/pdf/2603.01479)**

> **作者:** Kunlin Xie; Chenghao Li; Haolan Zhang; Nak Young Chong
>
> **备注:** submitted
>
> **摘要:** Vision-guided robot grasping based on Deep Neural Networks (DNNs) generalizes well but poses safety risks in the Human-Robot Interaction (HRI). Recent works solved it by designing benign adversarial attacks and patches with RGB modality, yet depth-independent characteristics limit their effectiveness on RGBD modality. In this work, we propose the Multimodal Adversarial Quality Policy (MAQP) to realize multimodal safe grasping. Our framework introduces two key components. First, the Heterogeneous Dual-Patch Optimization Scheme (HDPOS) mitigates the distribution discrepancy between RGB and depth modalities in patch generation by adopting modality-specific initialization strategies, employing a Gaussian distribution for depth patches and a uniform distribution for RGB patches, while jointly optimizing both modalities under a unified objective function. Second, the Gradient-Level Modality Balancing Strategy (GLMBS) is designed to resolve the optimization imbalance from RGB and Depth patches in patch shape adaptation by reweighting gradient contributions based on per-channel sensitivity analysis and applying distance-adaptive perturbation bounds. We conduct extensive experiments on the benchmark datasets and a cobot, showing the effectiveness of MAQP.
>
---
#### [replaced 036] CuriousBot: Interactive Mobile Exploration via Actionable 3D Relational Object Graph
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人移动探索任务，旨在解决传统方法缺乏主动交互的问题。通过构建3D关系物体图，实现更有效的环境探索与交互。**

- **链接: [https://arxiv.org/pdf/2501.13338](https://arxiv.org/pdf/2501.13338)**

> **作者:** Yixuan Wang; Leonor Fermoselle; Tarik Kelestemur; Jiuguang Wang; Yunzhu Li
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L). Project Page: this https URL
>
> **摘要:** Mobile exploration is a longstanding challenge in robotics, yet current methods primarily focus on active perception instead of active interaction, limiting the robot's ability to interact with and fully explore its environment. Existing robotic exploration approaches via active interaction are often restricted to tabletop scenes, neglecting the unique challenges posed by mobile exploration, such as large exploration spaces, complex action spaces, and diverse object relations. In this work, we introduce a 3D relational object graph that encodes diverse object relations and enables exploration through active interaction. We develop a system based on this representation and evaluate it across diverse scenes. Our qualitative and quantitative results demonstrate the system's effectiveness and generalization across object instances, relations, and scenes, outperforming methods solely relying on vision-language models (VLMs).
>
---
#### [replaced 037] Revisiting Replanning from Scratch: Real-Time Incremental Planning with Fast Almost-Surely Asymptotically Optimal Planners
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决动态环境中高效重规划问题。通过使用快速几乎必然渐近最优算法，无需更新旧计划即可生成高质量路径。**

- **链接: [https://arxiv.org/pdf/2510.21074](https://arxiv.org/pdf/2510.21074)**

> **作者:** Mitchell E. C. Sabbadini; Andrew H. Liu; Joseph Ruan; Tyler S. Wilson; Zachary Kingston; Jonathan D. Gammell
>
> **备注:** IEEE International Conference on Robotics and Automation (ICRA) 2026, 8 pages, 5 figures, 1 table. A video of this work can be found at this https URL
>
> **摘要:** Robots operating in changing environments either predict obstacle changes and/or plan quickly enough to react to them. Predictive approaches require a strong prior about the position and motion of obstacles. Reactive approaches require no assumptions about their environment but must replan quickly and find high-quality paths to navigate effectively. Reactive approaches often reuse information between queries to reduce planning cost. These techniques are conceptually sound but updating dense planning graphs when information changes can be computationally prohibitive. It can also require significant effort to detect the changes in some applications. This paper revisits the long-held assumption that reactive replanning requires updating existing plans. It shows that the incremental planning problem can alternatively be solved more efficiently as a series of independent problems using fast almost-surely asymptotically optimal (ASAO) planning algorithms. These ASAO algorithms quickly find an initial solution and converge towards an optimal solution which allows them to find consistent global plans in the presence of changing obstacles without requiring explicit plan reuse. This is demonstrated with simulated experiments where Effort Informed Trees (EIT*) finds shorter median solution paths than the tested reactive planning algorithms and is further validated using Asymptotically Optimal RRT-Connect (AORRTC) on a real-world planning problem on a robot arm.
>
---
#### [replaced 038] SynHLMA:Synthesizing Hand Language Manipulation for Articulated Object with Discrete Human Object Interaction Representation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出SynHLMA框架，解决具身对象手势语言操作生成问题。通过离散表示与语言嵌入结合，实现手势序列生成、预测与插值，提升机器人灵巧抓取性能。**

- **链接: [https://arxiv.org/pdf/2510.25268](https://arxiv.org/pdf/2510.25268)**

> **作者:** Wang zhi; Yuyan Liu; Liu Liu; Li Zhang; Ruixuan Lu; Dan Guo
>
> **摘要:** Generating hand grasps with language instructions is a widely studied topic that benefits from embodied AI and VR/AR applications. While transferring into hand articulatied object interaction (HAOI), the hand grasps synthesis requires not only object functionality but also long-term manipulation sequence along the object deformation. This paper proposes a novel HAOI sequence generation framework SynHLMA, to synthesize hand language manipulation for articulated objects. Given a complete point cloud of an articulated object, we utilize a discrete HAOI representation to model each hand object interaction frame. Along with the natural language embeddings, the representations are trained by an HAOI manipulation language model to align the grasping process with its language description in a shared representation space. A joint-aware loss is employed to ensure hand grasps follow the dynamic variations of articulated object joints. In this way, our SynHLMA achieves three typical hand manipulation tasks for articulated objects of HAOI generation, HAOI prediction and HAOI interpolation. We evaluate SynHLMA on our built HAOI-lang dataset and experimental results demonstrate the superior hand grasp sequence generation performance comparing with state-of-the-art. We also show a robotics grasp application that enables dexterous grasps execution from imitation learning using the manipulation sequence provided by our SynHLMA. Our codes and datasets will be made publicly available.
>
---
#### [replaced 039] VLN-Cache: Enabling Token Caching for VLN Models with Visual/Semantic Dynamics Awareness
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于视觉语言导航任务，解决大模型推理成本高的问题。提出VLN-Cache框架，通过视觉和语义动态感知实现高效token缓存，提升推理速度。**

- **链接: [https://arxiv.org/pdf/2603.07080](https://arxiv.org/pdf/2603.07080)**

> **作者:** Zihao Zheng; Zhihao Mao; Xingyue Zhou; Jiayu Chen; Maoliang Li; Xinhao Sun; Hailong Zou; Zhaobo Zhang; Xuanzhe Liu; Donggang Cao; Hong Mei; Xiang Chen
>
> **摘要:** Vision-and-Language Navigation (VLN) increasingly relies on large vision-language models, but their inference cost conflicts with real-time deployment. Token caching is a promising training-free strategy that avoids redundant computation by reusing stable visual tokens across frames. However, existing methods assume a static camera and fixed semantic focus, assumptions that VLN fundamentally violates. We identify two failure modes: (1) visual dynamics, where viewpoint shift displaces token positions across frames, causing position-wise matching to pair misaligned content; (2) semantic dynamics, where token relevance shifts across task stages as navigation progresses, making cached states stale. We propose VLN-Cache, a visual-dynamic-aware and semantic-dynamic-aware caching framework that introduces view-aligned remapping to recover geometric correspondences and a task-relevance saliency filter to veto reuse at semantic transitions. A layer-adaptive entropy policy further balances the per-layer reuse budget. Experiments on the R2R-CE simulation benchmark show up to 1.52x speedup while maintaining competitive navigation success rates.
>
---
#### [replaced 040] Relative Localization System Design for SnailBot: A Modular Self-reconfigurable Robot
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人定位任务，解决模块化机器人SnailBot的相对定位问题，融合ArUco标记、光流和IMU数据实现精准实时定位。**

- **链接: [https://arxiv.org/pdf/2512.21226](https://arxiv.org/pdf/2512.21226)**

> **作者:** Shuhan Zhang; Tin Lun Lam
>
> **备注:** The paper contains factual error and logic flaws, which needs to be repaired before submitting
>
> **摘要:** This paper presents the design and implementation of a relative localization system for SnailBot, a modular self reconfigurable robot. The system integrates ArUco marker recognition, optical flow analysis, and IMU data processing into a unified fusion framework, enabling robust and accurate relative positioning for collaborative robotic tasks. Experimental validation demonstrates the effectiveness of the system in realtime operation, with a rule based fusion strategy ensuring reliability across dynamic scenarios. The results highlight the potential for scalable deployment in modular robotic systems.
>
---
