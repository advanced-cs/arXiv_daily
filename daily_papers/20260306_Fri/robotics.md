# 机器人 cs.RO

- **最新发布 66 篇**

- **更新 50 篇**

## 最新发布

#### [new 001] U-OBCA: Uncertainty-Aware Optimization-Based Collision Avoidance via Wasserstein Distributionally Robust Chance Constraints
- **分类: cs.RO; eess.SY; math.OC**

- **简介: 该论文属于机器人路径规划任务，解决不确定性下的避障问题。针对现有方法的保守性和对分布假设的依赖，提出U-OBCA方法，通过概率约束优化提升导航效率。**

- **链接: [https://arxiv.org/pdf/2603.04914](https://arxiv.org/pdf/2603.04914)**

> **作者:** Zehao Wang; Yuxuan Tang; Han Zhang; Jingchuan Wang; Weidong Chen
>
> **摘要:** Uncertainties arising from localization error, trajectory prediction errors of the moving obstacles and environmental disturbances pose significant challenges to robot's safe navigation. Existing uncertainty-aware planners often approximate polygon-shaped robots and obstacles using simple geometric primitives such as circles or ellipses. Though computationally convenient, these approximations substantially shrink the feasible space, leading to overly conservative trajectories and even planning failure in narrow environments. In addition, many such methods rely on specific assumptions about noise distributions, which may not hold in practice and thus limit their performance guarantees. To address these limitations, we extend the Optimization-Based Collision Avoidance (OBCA) framework to an uncertainty-aware formulation, termed \emph{U-OBCA}. The proposed method explicitly accounts for the collision risk between polygon-shaped robots and obstacles by formulating OBCA-based chance constraints, and hence avoiding geometric simplifications and reducing unnecessary conservatism. These probabilistic constraints are further tightened into deterministic nonlinear constraints under mild distributional assumptions, which can be solved efficiently by standard numerical optimization solvers. The proposed approach is validated through theoretical analysis, numerical simulations and real-world experiments. The results demonstrate that U-OBCA significantly mitigates the conservatism in trajectory planning and achieves higher navigation efficiency compared to existing baseline methods, particularly in narrow and cluttered environments.
>
---
#### [new 002] GIANT - Global Path Integration and Attentive Graph Networks for Multi-Agent Trajectory Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多智能体轨迹规划任务，旨在解决多机器人碰撞避免问题。通过结合全局路径规划与注意力图神经网络，提升动态环境下的导航性能。**

- **链接: [https://arxiv.org/pdf/2603.04659](https://arxiv.org/pdf/2603.04659)**

> **作者:** Jonas le Fevre Sejersen; Toyotaro Suzumura; Erdal Kayacan
>
> **备注:** Published in: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** This paper presents a novel approach to multi-robot collision avoidance that integrates global path planning with local navigation strategies, utilizing attentive graph neural networks to manage dynamic interactions among agents. We introduce a local navigation model that leverages pre-planned global paths, allowing robots to adhere to optimal routes while dynamically adjusting to environmental changes. The models robustness is enhanced through the introduction of noise during training, resulting in superior performance in complex, dynamic environments. Our approach is evaluated against established baselines, including NH-ORCA, DRL-NAV, and GA3C-CADRL, across various structurally diverse simulated scenarios. The results demonstrate that our model achieves consistently higher success rates, lower collision rates, and more efficient navigation, particularly in challenging scenarios where baseline models struggle. This work offers an advancement in multi-robot navigation, with implications for robust performance in complex, dynamic environments with varying degrees of complexity, such as those encountered in logistics, where adaptability is essential for accommodating unforeseen obstacles and unpredictable changes.
>
---
#### [new 003] Adaptive Policy Switching of Two-Wheeled Differential Robots for Traversing over Diverse Terrains
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人自主导航任务，旨在解决复杂地形下策略适应问题。通过分析机器人姿态数据，实现地形分类，为自适应策略切换提供依据。**

- **链接: [https://arxiv.org/pdf/2603.04761](https://arxiv.org/pdf/2603.04761)**

> **作者:** Haruki Izawa; Takeshi Takai; Shingo Kitano; Mikita Miyaguchi; Hiroaki Kawashima
>
> **备注:** Author's version of the paper presented at AROB-ISBC 2026
>
> **摘要:** Exploring lunar lava tubes requires robots to traverse without human intervention. Because pre-trained policies cannot fully cover all possible terrain conditions, our goal is to enable adaptive policy switching, where the robot selects an appropriate terrain-specialized model based on its current terrain features. This study investigates whether terrain types can be estimated effectively using posture-related observations collected during navigation. We fine-tuned a pre-trained policy using Proximal Policy Optimization (PPO), and then collected the robot's 3D orientation data as it moved across flat and rough terrain in a simulated lava-tube environment. Our analysis revealed that the standard deviation of the robot's pitch data shows a clear difference between these two terrain types. Using Gaussian mixture models (GMM), we evaluated terrain classification across various window sizes. An accuracy of more than 98% was achieved when using a 70-step window. The result suggests that short-term orientation data are sufficient for reliable terrain estimation, providing a foundation for adaptive policy switching.
>
---
#### [new 004] Beyond the Patch: Exploring Vulnerabilities of Visuomotor Policies via Viewpoint-Consistent 3D Adversarial Object
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉控制领域，旨在解决动态视角下2D对抗补丁失效的问题。通过3D对抗纹理优化方法，提升对机器人策略的攻击效果。**

- **链接: [https://arxiv.org/pdf/2603.04913](https://arxiv.org/pdf/2603.04913)**

> **作者:** Chanmi Lee; Minsung Yoon; Woojae Kim; Sebin Lee; Sung-eui Yoon
>
> **备注:** 8 pages, 10 figures, Accepted to ICRA 2026. Project page: this https URL
>
> **摘要:** Neural network-based visuomotor policies enable robots to perform manipulation tasks but remain susceptible to perceptual attacks. For example, conventional 2D adversarial patches are effective under fixed-camera setups, where appearance is relatively consistent; however, their efficacy often diminishes under dynamic viewpoints from moving cameras, such as wrist-mounted setups, due to perspective distortions. To proactively investigate potential vulnerabilities beyond 2D patches, this work proposes a viewpoint-consistent adversarial texture optimization method for 3D objects through differentiable rendering. As optimization strategies, we employ Expectation over Transformation (EOT) with a Coarse-to-Fine (C2F) curriculum, exploiting distance-dependent frequency characteristics to induce textures effective across varying camera-object distances. We further integrate saliency-guided perturbations to redirect policy attention and design a targeted loss that persistently drives robots toward adversarial objects. Our comprehensive experiments show that the proposed method is effective under various environmental conditions, while confirming its black-box transferability and real-world applicability.
>
---
#### [new 005] UltraDexGrasp: Learning Universal Dexterous Grasping for Bimanual Robots with Synthetic Data
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决双臂机器人灵巧抓取的问题。通过合成数据生成高质量抓取策略，提升抓取成功率。**

- **链接: [https://arxiv.org/pdf/2603.05312](https://arxiv.org/pdf/2603.05312)**

> **作者:** Sizhe Yang; Yiman Xie; Zhixuan Liang; Yang Tian; Jia Zeng; Dahua Lin; Jiangmiao Pang
>
> **备注:** Published at International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Grasping is a fundamental capability for robots to interact with the physical world. Humans, equipped with two hands, autonomously select appropriate grasp strategies based on the shape, size, and weight of objects, enabling robust grasping and subsequent manipulation. In contrast, current robotic grasping remains limited, particularly in multi-strategy settings. Although substantial efforts have targeted parallel-gripper and single-hand grasping, dexterous grasping for bimanual robots remains underexplored, with data being a primary bottleneck. Achieving physically plausible and geometrically conforming grasps that can withstand external wrenches poses significant challenges. To address these issues, we introduce UltraDexGrasp, a framework for universal dexterous grasping with bimanual robots. The proposed data-generation pipeline integrates optimization-based grasp synthesis with planning-based demonstration generation, yielding high-quality and diverse trajectories across multiple grasp strategies. With this framework, we curate UltraDexGrasp-20M, a large-scale, multi-strategy grasp dataset comprising 20 million frames across 1,000 objects. Based on UltraDexGrasp-20M, we further develop a simple yet effective grasp policy that takes point clouds as input, aggregates scene features via unidirectional attention, and predicts control commands. Trained exclusively on synthetic data, the policy achieves robust zero-shot sim-to-real transfer and consistently succeeds on novel objects with varied shapes, sizes, and weights, attaining an average success rate of 81.2% in real-world universal dexterous grasping. To facilitate future research on grasping with bimanual robots, we open-source the data generation pipeline at this https URL.
>
---
#### [new 006] CT-Enabled Patient-Specific Simulation and Contact-Aware Robotic Planning for Cochlear Implantation
- **分类: cs.RO**

- **简介: 该论文属于机器人辅助耳科手术任务，旨在解决人工耳蜗植入中的接触力控制问题。通过CT影像构建患者特异性模型，结合力学模拟与规划算法，提升植入精度与安全性。**

- **链接: [https://arxiv.org/pdf/2603.05333](https://arxiv.org/pdf/2603.05333)**

> **作者:** Lingxiao Xun; Gang Zheng; Alexandre Kruszewski; Renato Torres
>
> **摘要:** Robotic cochlear-implant (CI) insertion requires precise prediction and regulation of contact forces to minimize intracochlear trauma and prevent failure modes such as locking and buckling. Aligned with the integration of advanced medical imaging and robotics for autonomous, precision interventions, this paper presents a unified CT-to-simulation pipeline for contact-aware insertion planning and validation. We develop a low-dimensional, differentiable Cosserat-rod model of the electrode array coupled with frictional contact and pseudo-dynamics regularization to ensure continuous stick-slip transitions. Patient-specific cochlear anatomy is reconstructed from CT imaging and encoded via an analytic parametrization of the scala-tympani lumen, enabling efficient and differentiable contact queries through closest-point projection. Based on a differentiated equilibrium-constraint formulation, we derive an online direction-update law under an RCM-like constraint that suppresses lateral insertion forces while maintaining axial advancement. Simulations and benchtop experiments validate deformation and force trends, demonstrating reduced locking/buckling risk and improved insertion depth. The study highlights how CT-based imaging enhances modeling, planning, and safety capabilities in robot-assisted inner-ear procedures.
>
---
#### [new 007] AIM-SLAM: Dense Monocular SLAM via Adaptive and Informative Multi-View Keyframe Prioritization with Foundation Model
- **分类: cs.RO**

- **简介: 该论文提出AIM-SLAM，解决单目SLAM中密集重建的视图选择问题。通过自适应多视角关键帧优先策略，提升定位与重建精度。**

- **链接: [https://arxiv.org/pdf/2603.05097](https://arxiv.org/pdf/2603.05097)**

> **作者:** Jinwoo Jeon; Dong-Uk Seo; Eungchang Mason Lee; Hyun Myung
>
> **备注:** 8 pages
>
> **摘要:** Recent advances in geometric foundation models have emerged as a promising alternative for addressing the challenge of dense reconstruction in monocular visual simultaneous localization and mapping (SLAM). Although geometric foundation models enable SLAM to leverage variable input views, the previous methods remain confined to two-view pairs or fixed-length inputs without sufficient deliberation of geometric context for view selection. To tackle this problem, we propose AIM-SLAM, a dense monocular SLAM framework that exploits an adaptive and informative multi-view keyframe prioritization with dense pointmap predictions from visual geometry grounded transformer (VGGT). Specifically, we introduce the selective information- and geometric-aware multi-view adaptation (SIGMA) module, which employs voxel overlap and information gain to retrieve a candidate set of keyframes and adaptively determine its size. Furthermore, we formulate a joint multi-view Sim(3) optimization that enforces consistent alignment across selected views, substantially improving pose estimation accuracy. The effectiveness of AIM-SLAM is demonstrated on real-world datasets, where it achieves state-of-the-art performance in both pose estimation and dense reconstruction. Our system supports ROS integration, with code is available at this https URL.
>
---
#### [new 008] RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，旨在解决长时序、依赖历史的机械操作问题。提出RoboMME基准，评估和提升VLA模型的记忆能力。**

- **链接: [https://arxiv.org/pdf/2603.04639](https://arxiv.org/pdf/2603.04639)**

> **作者:** Yinpei Dai; Hongze Fu; Jayjun Lee; Yuejiang Liu; Haoran Zhang; Jianing Yang; Chelsea Finn; Nima Fazeli; Joyce Chai
>
> **摘要:** Memory is critical for long-horizon and history-dependent robotic manipulation. Such tasks often involve counting repeated actions or manipulating objects that become temporarily occluded. Recent vision-language-action (VLA) models have begun to incorporate memory mechanisms; however, their evaluations remain confined to narrow, non-standardized settings. This limits their systematic understanding, comparison, and progress measurement. To address these challenges, we introduce RoboMME: a large-scale standardized benchmark for evaluating and advancing VLA models in long-horizon, history-dependent scenarios. Our benchmark comprises 16 manipulation tasks constructed under a carefully designed taxonomy that evaluates temporal, spatial, object, and procedural memory. We further develop a suite of 14 memory-augmented VLA variants built on the {\pi}0.5 backbone to systematically explore different memory representations across multiple integration strategies. Experimental results show that the effectiveness of memory representations is highly task-dependent, with each design offering distinct advantages and limitations across different tasks. Videos and code can be found at our website this https URL.
>
---
#### [new 009] Safe-SAGE: Social-Semantic Adaptive Guidance for Safe Engagement through Laplace-Modulated Poisson Safety Functions
- **分类: cs.RO**

- **简介: 该论文属于机器人安全导航任务，解决传统方法对障碍物语义感知不足的问题。提出Safe-SAGE框架，结合语义信息与安全控制，实现动态环境下的安全路径规划。**

- **链接: [https://arxiv.org/pdf/2603.05497](https://arxiv.org/pdf/2603.05497)**

> **作者:** Lizhi Yang; Ryan M. Bena; Meg Wilkinson; Gilbert Bahati; Andy Navarro Brenes; Ryan K. Cosner; Aaron D. Ames
>
> **摘要:** Traditional safety-critical control methods, such as control barrier functions, suffer from semantic blindness, exhibiting the same behavior around obstacles regardless of contextual significance. This limitation leads to the uniform treatment of all obstacles, despite their differing semantic meanings. We present Safe-SAGE (Social-Semantic Adaptive Guidance for Safe Engagement), a unified framework that bridges the gap between high-level semantic understanding and low-level safety-critical control through a Poisson safety function (PSF) modulated using a Laplace guidance field. Our approach perceives the environment by fusing multi-sensor point clouds with vision-based instance segmentation and persistent object tracking to maintain up-to-date semantics beyond the camera's field of view. A multi-layer safety filter is then used to modulate system inputs to achieve safe navigation using this semantic understanding of the environment. This safety filter consists of both a model predictive control layer and a control barrier function layer. Both layers utilize the PSF and flux modulation of the guidance field to introduce varying levels of conservatism and multi-agent passing norms for different obstacles in the environment. Our framework enables legged robots to navigate semantically rich, dynamic environments with context-dependent safety margins while maintaining rigorous safety guarantees.
>
---
#### [new 010] Residual RL--MPC for Robust Microrobotic Cell Pushing Under Time-Varying Flow
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究微流体环境下微机器人抓取任务，解决时间变化流场中接触不稳定和漂移问题。提出混合控制器，结合MPC与强化学习残差策略，提升跟踪精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.05448](https://arxiv.org/pdf/2603.05448)**

> **作者:** Yanda Yang; Sambeeta Das
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Contact-rich micromanipulation in microfluidic flow is challenging because small disturbances can break pushing contact and induce large lateral drift. We study planar cell pushing with a magnetic rolling microrobot that tracks a waypoint-sampled reference curve under time-varying Poiseuille flow. We propose a hybrid controller that augments a nominal MPC with a learned residual policy trained by SAC. The policy outputs a bounded 2D velocity correction that is contact-gated, so residual actions are applied only during robot--cell contact, preserving reliable approach behavior and stabilizing learning. All methods share the same actuation interface and speed envelope for fair comparisons. Experiments show improved robustness and tracking accuracy over pure MPC and PID under nonstationary flow, with generalization from a clover training curve to unseen circle and square trajectories. A residual-bound sweep identifies an intermediate correction limit as the best trade-off, which we use in all benchmarks.
>
---
#### [new 011] VPWEM: Non-Markovian Visuomotor Policy with Working and Episodic Memory
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出VPWEM，解决机器人控制中非马尔可夫任务的长期记忆问题。通过工作记忆和情景记忆，提升策略性能。**

- **链接: [https://arxiv.org/pdf/2603.04910](https://arxiv.org/pdf/2603.04910)**

> **作者:** Yuheng Lei; Zhixuan Liang; Hongyuan Zhang; Ping Luo
>
> **摘要:** Imitation learning from human demonstrations has achieved significant success in robotic control, yet most visuomotor policies still condition on single-step observations or short-context histories, making them struggle with non-Markovian tasks that require long-term memory. Simply enlarging the context window incurs substantial computational and memory costs and encourages overfitting to spurious correlations, leading to catastrophic failures under distribution shift and violating real-time constraints in robotic systems. By contrast, humans can compress important past experiences into long-term memories and exploit them to solve tasks throughout their lifetime. In this paper, we propose VPWEM, a non-Markovian visuomotor policy equipped with working and episodic memories. VPWEM retains a sliding window of recent observation tokens as short-term working memory, and introduces a Transformer-based contextual memory compressor that recursively converts out-of-window observations into a fixed number of episodic memory tokens. The compressor uses self-attention over a cache of past summary tokens and cross-attention over a cache of historical observations, and is trained jointly with the policy. We instantiate VPWEM on diffusion policies to exploit both short-term and episode-wide information for action generation with nearly constant memory and computation per step. Experiments demonstrate that VPWEM outperforms state-of-the-art baselines including diffusion policies and vision-language-action (VLA) models by more than 20% on the memory-intensive manipulation tasks in MIKASA and achieves an average 5% improvement on the mobile manipulation benchmark MoMaRT. Code is available at this https URL.
>
---
#### [new 012] LLM-Guided Decentralized Exploration with Self-Organizing Robot Teams
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人协作任务，解决无中心控制下的团队自组织与目标探索问题。提出自组织算法和基于LLM的探索策略，提升探索效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.04762](https://arxiv.org/pdf/2603.04762)**

> **作者:** Hiroaki Kawashima; Shun Ikejima; Takeshi Takai; Mikita Miyaguchi; Yasuharu Kunii
>
> **备注:** Author's version of the paper presented at AROB-ISBC 2026
>
> **摘要:** When individual robots have limited sensing capabilities or insufficient fault tolerance, it becomes necessary for multiple robots to form teams during exploration, thereby increasing the collective observation range and reliability. Traditionally, swarm formation has often been managed by a central controller; however, from the perspectives of robustness and flexibility, it is preferable for the swarm to operate autonomously even in the absence of centralized control. In addition, the determination of exploration targets for each team is crucial for efficient exploration in such multi-team exploration scenarios. This study therefore proposes an exploration method that combines (1) an algorithm for self-organization, enabling the autonomous and dynamic formation of multiple teams, and (2) an algorithm that allows each team to autonomously determine its next exploration target (destination). In particular, for (2), this study explores a novel strategy based on large language models (LLMs), while classical frontier-based methods and deep reinforcement learning approaches have been widely studied. The effectiveness of the proposed method was validated through simulations involving tens to hundreds of robots.
>
---
#### [new 013] Hyperbolic Multiview Pretraining for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决传统方法在结构关系建模上的不足。通过引入双曲空间的多视角预训练框架HyperMVP，提升空间感知能力。**

- **链接: [https://arxiv.org/pdf/2603.04848](https://arxiv.org/pdf/2603.04848)**

> **作者:** Jin Yang; Ping Wei; Yixin Chen
>
> **备注:** This paper was submitted to CVPR 2026 and was recommended for Findings, but the authors have withdrawn it and are currently adding more content to submit it elsewhere
>
> **摘要:** 3D-aware visual pretraining has proven effective in improving the performance of downstream robotic manipulation tasks. However, existing methods are constrained to Euclidean embedding spaces, whose flat geometry limits their ability to model structural relations among embeddings. As a result, they struggle to learn structured embeddings that are essential for robust spatial perception in robotic applications. To this end, we propose HyperMVP, a self-supervised framework for \underline{Hyper}bolic \underline{M}ulti\underline{V}iew \underline{P}retraining. Hyperbolic space offers geometric properties well suited for capturing structural relations. Methodologically, we extend the masked autoencoder paradigm and design a GeoLink encoder to learn multiview hyperbolic representations. The pretrained encoder is then finetuned with visuomotor policies on manipulation tasks. In addition, we introduce 3D-MOV, a large-scale dataset comprising multiple types of 3D point clouds to support pretraining. We evaluate HyperMVP on COLOSSEUM, RLBench, and real-world scenarios, where it consistently outperforms strong baselines across diverse tasks and perturbation settings. Our results highlight the potential of 3D-aware pretraining in a non-Euclidean space for learning robust and generalizable robotic manipulation policies.
>
---
#### [new 014] Selecting Spots by Explicitly Predicting Intention from Motion History Improves Performance in Autonomous Parking
- **分类: cs.RO**

- **简介: 该论文研究自动驾驶泊车任务，解决如何通过显式预测其他车辆意图来选择停车位的问题。工作包括构建基于运动历史和概率信念图的泊车选择管道，并在仿真中验证其有效性。**

- **链接: [https://arxiv.org/pdf/2603.04695](https://arxiv.org/pdf/2603.04695)**

> **作者:** Long Kiu Chung; David Isele; Faizan M. Tariq; Sangjae Bae; Shreyas Kousik; Jovin D'sa
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** In many applications of social navigation, existing works have shown that predicting and reasoning about human intentions can help robotic agents make safer and more socially acceptable decisions. In this work, we study this problem for autonomous valet parking (AVP), where an autonomous vehicle ego agent must drop off its passengers, explore the parking lot, find a parking spot, negotiate for the spot with other vehicles, and park in the spot without human supervision. Specifically, we propose an AVP pipeline that selects parking spots by explicitly predicting where other agents are going to park from their motion history using learned models and probabilistic belief maps. To test this pipeline, we build a simulation environment with reactive agents and realistic modeling assumptions on the ego agent, such as occlusion-aware observations, and imperfect trajectory prediction. Simulation experiments show that our proposed method outperforms existing works that infer intentions from future predicted motion or embed them implicitly in end-to-end models, yielding better results in prediction accuracy, social acceptance, and task completion. Our key insight is that, in parking, where driving regulations are more lax, explicit intention prediction is crucial for reasoning about diverse and ambiguous long-term goals, which cannot be reliably inferred from short-term motion prediction alone, but can be effectively learned from motion history.
>
---
#### [new 015] cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决高自由度机器人轨迹生成与碰撞避免问题。提出cuRoboV2框架，集成轨迹优化、感知与计算模块，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.05493](https://arxiv.org/pdf/2603.05493)**

> **作者:** Balakumar Sundaralingam; Adithyavairavan Murali; Stan Birchfield
>
> **备注:** cuRoboV2 Technical Report
>
> **摘要:** Effective robot autonomy requires motion generation that is safe, feasible, and reactive. Current methods are fragmented: fast planners output physically unexecutable trajectories, reactive controllers struggle with high-fidelity perception, and existing solvers fail on high-DoF systems. We present cuRoboV2, a unified framework with three key innovations: (1) B-spline trajectory optimization that enforces smoothness and torque limits; (2) a GPU-native TSDF/ESDF perception pipeline that generates dense signed distance fields covering the full workspace, unlike existing methods that only provide distances within sparsely allocated blocks, up to 10x faster and in 8x less memory than the state-of-the-art at manipulation scale, with up to 99% collision recall; and (3) scalable GPU-native whole-body computation, namely topology-aware kinematics, differentiable inverse dynamics, and map-reduce self-collision, that achieves up to 61x speedup while also extending to high-DoF humanoids (where previous GPU implementations fail). On benchmarks, cuRoboV2 achieves 99.7% success under 3kg payload (where baselines achieve only 72--77%), 99.6% collision-free IK on a 48-DoF humanoid (where prior methods fail entirely), and 89.5% retargeting constraint satisfaction (vs. 61% for PyRoki); these collision-free motions yield locomotion policies with 21% lower tracking error than PyRoki and 12x lower cross-seed variance than mink. A ground-up codebase redesign for discoverability enabled LLM coding assistants to author up to 73% of new modules, including hand-optimized CUDA kernels, demonstrating that well-structured robotics code can unlock productive human--LLM collaboration. Together, these advances provide a unified, dynamics-aware motion generation stack that scales from single-arm manipulators to full humanoids.
>
---
#### [new 016] ELLIPSE: Evidential Learning for Robust Waypoints and Uncertainties
- **分类: cs.RO**

- **简介: 该论文提出ELLIPSE方法，用于机器人在复杂环境中鲁棒地预测路径点及不确定性。解决分布偏移导致的过度自信问题，通过深度证据回归和后校准提升预测可靠性。**

- **链接: [https://arxiv.org/pdf/2603.04585](https://arxiv.org/pdf/2603.04585)**

> **作者:** Zihao Dong; Chanyoung Chung; Dong-Ki Kim; Mukhtar Maulimov; Xiangyun Meng; Harmish Khambhaita; Ali-akbar Agha-mohammadi; Amirreza Shaban
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Robust waypoint prediction is crucial for mobile robots operating in open-world, safety-critical settings. While Imitation Learning (IL) methods have demonstrated great success in practice, they are susceptible to distribution shifts: the policy can become dangerously overconfident in unfamiliar states. In this paper, we present \textit{ELLIPSE}, a method building on multivariate deep evidential regression to output waypoints and multivariate Student-t predictive distributions in a single forward pass. To reduce covariate-shift-induced overconfidence under viewpoint and pose perturbations near expert trajectories, we introduce a lightweight domain augmentation procedure that synthesizes plausible viewpoint/pose variations without collecting additional demonstrations. To improve uncertainty reliability under environment/domain shift (e.g., unseen staircases), we apply a post-hoc isotonic recalibration on probability integral transform (PIT) values so that prediction sets remain plausible during deployment. We ground the discussion and experiments in staircase waypoint prediction, where obtaining robust waypoint and uncertainty is pivotal. Extensive real world evaluations show that \textit{ELLIPSE} improves both task success rate and uncertainty coverage compared to baselines.
>
---
#### [new 017] Python Bindings for a Large C++ Robotics Library: The Case of OMPL
- **分类: cs.RO**

- **简介: 论文探讨使用大语言模型辅助生成Python绑定，以解决C++库与Python集成的难题。任务是提高绑定生成效率与质量，工作包括设计流程、利用LLM生成代码并由专家优化。**

- **链接: [https://arxiv.org/pdf/2603.04668](https://arxiv.org/pdf/2603.04668)**

> **作者:** Weihang Guo; Theodoros Tyrovouzis; Lydia E. Kavraki
>
> **摘要:** Python bindings are a critical bridge between high-performance C++ libraries and the flexibility of Python, enabling rapid prototyping, reproducible experiments, and integration with simulation and learning frameworks in robotics research. Yet, generating bindings for large codebases is a tedious process that creates a heavy burden for a small group of maintainers. In this work, we investigate the use of Large Language Models (LLMs) to assist in generating nanobind wrappers, with human experts kept in the loop. Our workflow mirrors the structure of the C++ codebase, scaffolds empty wrapper files, and employs LLMs to fill in binding definitions. Experts then review and refine the generated code to ensure correctness, compatibility, and performance. Through a case study on a large C++ motion planning library, we document common failure modes, including mismanaging shared pointers, overloads, and trampolines, and show how in-context examples and careful prompt design improve reliability. Experiments demonstrate that the resulting bindings achieve runtime performance comparable to legacy solutions. Beyond this case study, our results provide general lessons for applying LLMs to binding generation in large-scale C++ projects.
>
---
#### [new 018] RoboPocket: Improve Robot Policies Instantly with Your Phone
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出RoboPocket系统，用于提升机器人策略的模仿学习效率。针对数据收集低效和依赖物理机器人的问题，利用AR视觉预测实现远程反馈与在线微调，提高数据利用率和样本效率。**

- **链接: [https://arxiv.org/pdf/2603.05504](https://arxiv.org/pdf/2603.05504)**

> **作者:** Junjie Fang; Wendi Chen; Han Xue; Fangyuan Zhou; Tian Le; Yi Wang; Yuting Zhang; Jun Lv; Chuan Wen; Cewu Lu
>
> **备注:** Project page: this https URL
>
> **摘要:** Scaling imitation learning is fundamentally constrained by the efficiency of data collection. While handheld interfaces have emerged as a scalable solution for in-the-wild data acquisition, they predominantly operate in an open-loop manner: operators blindly collect demonstrations without knowing the underlying policy's weaknesses, leading to inefficient coverage of critical state distributions. Conversely, interactive methods like DAgger effectively address covariate shift but rely on physical robot execution, which is costly and difficult to scale. To reconcile this trade-off, we introduce RoboPocket, a portable system that enables Robot-Free Instant Policy Iteration using single consumer smartphones. Its core innovation is a Remote Inference framework that visualizes the policy's predicted trajectory via Augmented Reality (AR) Visual Foresight. This immersive feedback allows collectors to proactively identify potential failures and focus data collection on the policy's weak regions without requiring a physical robot. Furthermore, we implement an asynchronous Online Finetuning pipeline that continuously updates the policy with incoming data, effectively closing the learning loop in minutes. Extensive experiments demonstrate that RoboPocket adheres to data scaling laws and doubles the data efficiency compared to offline scaling strategies, overcoming their long-standing efficiency bottleneck. Moreover, our instant iteration loop also boosts sample efficiency by up to 2$\times$ in distributed environments a small number of interactive corrections per person. Project page and videos: this https URL.
>
---
#### [new 019] SPIRIT: Perceptive Shared Autonomy for Robust Robotic Manipulation under Deep Learning Uncertainty
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出SPIRIT系统，解决深度学习在机器人操作中的不确定性问题。通过感知共享自主，结合点云配准技术，提升操作性能与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.05111](https://arxiv.org/pdf/2603.05111)**

> **作者:** Jongseok Lee; Ribin Balachandran; Harsimran Singh; Jianxiang Feng; Hrishik Mishra; Marco De Stefano; Rudolph Triebel; Alin Albu-Schaeffer; Konstantin Kondak
>
> **备注:** 19 pages, 14 figures
>
> **摘要:** Deep learning (DL) has enabled impressive advances in robotic perception, yet its limited robustness and lack of interpretability hinder reliable deployment in safety critical applications. We propose a concept termed perceptive shared autonomy, in which uncertainty estimates from DL based perception are used to regulate the level of autonomy. Specifically, when the robot's perception is confident, semi-autonomous manipulation is enabled to improve performance; when uncertainty increases, control transitions to haptic teleoperation for maintaining robustness. In this way, high-performing but uninterpretable DL methods can be integrated safely into robotic systems. A key technical enabler is an uncertainty aware DL based point cloud registration approach based on the so called Neural Tangent Kernels (NTK). We evaluate perceptive shared autonomy on challenging aerial manipulation tasks through a user study of 15 participants and realization of mock-up industrial scenarios, demonstrating reliable robotic manipulation despite failures in DL based perception. The resulting system, named SPIRIT, improves both manipulation performance and system reliability. SPIRIT was selected as a finalist of a major industrial innovation award.
>
---
#### [new 020] Autonomous Aerial Non-Destructive Testing: Ultrasound Inspection with a Commercial Quadrotor in an Unstructured Environment
- **分类: cs.RO**

- **简介: 该论文属于自主飞行检测任务，解决无人机在非结构化环境中进行接触式无损检测的问题。通过开发控制架构，实现商用四旋翼的自主 ultrasound 检测。**

- **链接: [https://arxiv.org/pdf/2603.04642](https://arxiv.org/pdf/2603.04642)**

> **作者:** Ruben Veenstra; Barbara Bazzana; Sander Smits; Antonio Franchi
>
> **摘要:** This work presents an integrated control and software architecture that enables arguably the first fully autonomous, contact-based non-destructive testing (NDT) using a commercial multirotor originally restricted to remotely-piloted operations. To allow autonomous operation with an off-the-shelf platform, we developed a real-time framework that interfaces directly with its onboard sensor suite. The architecture features a multi-rate control scheme: low-level control is executed at 200 Hz, force estimation at 100 Hz, while an admittance filter and trajectory planner operate at 50 Hz, ultimately supplying acceleration and yaw rate commands to the internal flight controller. We validate the system through physical experiments on a Flyability Elios 3 quadrotor equipped with an ultrasound payload. Relying exclusively on onboard sensing, the vehicle successfully performs autonomous NDT measurements within an unstructured, industrial-like environment. This work demonstrates the viability of retrofitting off-the-shelf platforms for autonomous physical interaction, paving the way for safe, contact-based inspection of hazardous and confined infrastructure.
>
---
#### [new 021] Risk-Aware Reinforcement Learning for Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文属于移动操作任务，解决机器人在不确定环境中进行风险感知决策的问题。通过强化学习和模仿学习，构建了具有风险敏感性的视觉运动策略。**

- **链接: [https://arxiv.org/pdf/2603.04579](https://arxiv.org/pdf/2603.04579)**

> **作者:** Michael Groom; James Wilson; Nick Hawes; Lars Kunze
>
> **摘要:** For robots to successfully transition from lab settings to everyday environments, they must begin to reason about the risks associated with their actions and make informed, risk-aware decisions. This is particularly true for robots performing mobile manipulation tasks, which involve both interacting with and navigating within dynamic, unstructured spaces. However, existing whole-body controllers for mobile manipulators typically lack explicit mechanisms for risk-sensitive decision-making under uncertainty. To our knowledge, we are the first to (i) learn risk-aware visuomotor policies for mobile manipulation conditioned on egocentric depth observations with runtime-adjustable risk sensitivity, and (ii) show risk-aware behaviours can be transferred through Imitation Learning (IL) to a visuomotor policy conditioned on egocentric depth observations. Our method achieves this by first training a privileged teacher policy using Distributional Reinforcement Learning (DRL), with a risk-neutral distributional critic. Distortion risk-metrics are then applied to the critic's predicted return distribution to calculate risk-adjusted advantage estimates used in policy updates to achieve a range of risk-aware behaviours. We then distil teacher policies with IL to obtain risk-aware student policies conditioned on egocentric depth observations. We perform extensive evaluations demonstrating that our trained visuomotor policies exhibit risk-aware behaviour (specifically achieving better worst-case performance) while performing reactive whole-body motions in unmapped environments, leveraging live depth observations for perception.
>
---
#### [new 022] From Code to Road: A Vehicle-in-the-Loop and Digital Twin-Based Framework for Central Car Server Testing in Autonomous Driving
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自动驾驶软件测试任务，旨在解决传统仿真无法准确反映现实的问题。提出基于VIL和数字孪生的测试框架，实现虚拟与物理环境的无缝集成，提升测试效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.05279](https://arxiv.org/pdf/2603.05279)**

> **作者:** Chengdong Wu; Sven Kirchner; Nils Purschke; Axel Torschmied; Norbert Kroth; Yinglei Song; André Schamschurko; Erik Leo Haß; Kuo-Yi Chao; Yi Zhang; Nenad Petrovic; Alois C. Knoll
>
> **备注:** 8 pages; Accepted for publication at the 37th IEEE Intelligent Vehicles Symposium (IV), Detroit, MI, United States, June 22-25, 2026
>
> **摘要:** Simulation is one of the most essential parts in the development stage of automotive software. However, purely virtual simulations often struggle to accurately capture all real-world factors due to limitations in modeling. To address this challenge, this work presents a test framework for automotive software on the centralized E/E architecture, which is a central car server in our case, based on Vehicle-in-the-Loop (ViL) and digital twin technology. The framework couples a physical test vehicle on a dynamometer test bench with its synchronized virtual counterpart in a simulation environment. Our approach provides a safe, reproducible, realistic, and cost-effective platform for validating autonomous driving algorithms with a centralized architecture. This test method eliminates the need to test individual physical ECUs and their communication protocols separately. In contrast to traditional ViL methods, the proposed framework runs the full autonomous driving software directly on the vehicle hardware after the simulation process, eliminating flashing and intermediate layers while enabling seamless virtual-physical integration and accurately reflecting centralized E/E behavior. In addition, incorporating mixed testing in both simulated and physical environments reduces the need for full hardware integration during the early stages of automotive development. Experimental case studies demonstrate the effectiveness of the framework in different test scenarios. These findings highlight the potential to reduce development and integration efforts for testing autonomous driving pipelines in the future.
>
---
#### [new 023] Direct Contact-Tolerant Motion Planning With Vision Language Models
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决复杂环境中接触容忍的运动规划问题。通过引入视觉语言模型，直接感知并处理点云，提升导航的鲁棒性和效率。**

- **链接: [https://arxiv.org/pdf/2603.05017](https://arxiv.org/pdf/2603.05017)**

> **作者:** He Li; Jian Sun; Chengyang Li; Guoliang Li; Qiyu Ruan; Shuai Wang; Chengzhong Xu
>
> **摘要:** Navigation in cluttered environments often requires robots to tolerate contact with movable or deformable objects to maintain efficiency. Existing contact-tolerant motion planning (CTMP) methods rely on indirect spatial representations (e.g., prebuilt map, obstacle set), resulting in inaccuracies and a lack of adaptiveness to environmental uncertainties. To address this issue, we propose a direct contact-tolerant (DCT) planner, which integrates vision-language models (VLMs) into direct point perception and navigation, including two key components. The first one is VLM point cloud partitioner (VPP), which performs contact-tolerance reasoning in image space using VLM, caches inference masks, propagates them across frames using odometry, and projects them onto the current scan to generate a contact-aware point cloud. The second innovation is VPP guided navigation (VGN), which formulates CTMP as a perception-to-control optimization problem under direct contact-aware point cloud constraints, which is further solved by a specialized deep neural network (DNN). We implement DCT in Isaac Sim and a real car-like robot, demonstrating that DCT achieves robust and efficient navigation in cluttered environments with movable obstacles, outperforming representative baselines across diverse metrics. The code is available at: this https URL.
>
---
#### [new 024] Curve-Induced Dynamical Systems on Riemannian Manifolds and Lie Groups
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出CDSM框架，用于在流形和李群上构建动态系统，解决机器人行为生成问题，提升轨迹精度与响应速度。**

- **链接: [https://arxiv.org/pdf/2603.05268](https://arxiv.org/pdf/2603.05268)**

> **作者:** Saray Bakker; Martin Schonger; Tobias Löw; Javier Alonso-Mora; Sylvain Calinon
>
> **备注:** Preprint, 14 pages, video linked in the paper, Saray Bakker and Martin Schonger contributed equally as first authors and are listed alphabetically
>
> **摘要:** Deploying robots in household environments requires safe, adaptable, and interpretable behaviors that respect the geometric structure of tasks. Often represented on Lie groups and Riemannian manifolds, this includes poses on SE(3) or symmetric positive definite matrices encoding stiffness or damping matrices. In this context, dynamical system-based approaches offer a natural framework for generating such behavior, providing stability and convergence while remaining responsive to changes in the environment. We introduce Curve-induced Dynamical systems on Smooth Manifolds (CDSM), a real-time framework for constructing dynamical systems directly on Riemannian manifolds and Lie groups. The proposed approach constructs a nominal curve on the manifold, and generates a dynamical system which combines a tangential component that drives motion along the curve and a normal component that attracts the state toward the curve. We provide a stability analysis of the resulting dynamical system and validate the method quantitatively. On an S2 benchmark, CDSM demonstrates improved trajectory accuracy, reduced path deviation, and faster generation and query times compared to state-of-the-art methods. Finally, we demonstrate the practical applicability of the framework on both a robotic manipulator, where poses on SE(3) and damping matrices on SPD(n) are adapted online, and a mobile manipulator.
>
---
#### [new 025] Designing and Validating a Self-Aligning Tool Changer for Modular Reconfigurable Manipulation Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人模块化任务，解决模块交换中的对准误差问题。设计了自对准工具更换系统，采用被动几何结构实现可靠模块交换。**

- **链接: [https://arxiv.org/pdf/2603.04760](https://arxiv.org/pdf/2603.04760)**

> **作者:** Mahfudz Maskur; Takuya Kiyokawa; Kensuke Harada
>
> **备注:** 6 pages, 13 figures
>
> **摘要:** Modular reconfigurable robots require reliable mechanisms for automated module exchange, but conventional rigid active couplings often fail due to inevitable positioning and orientational errors. To address this, we propose a misalignment-tolerant tool-changing system. The hardware features a motor-driven coupling utilizing passive self-alignment geometries, specifically chamfered receptacles and triangular lead-in guides, to robustly compensate for angular and lateral misalignments without complex force sensors. To make this autonomous exchange practically feasible, the mechanism is complemented by a compact rotating tool exchange station for efficient module storage. Real-world autonomous tool-picking experiments validate that the self-aligning features successfully absorb execution errors, enabling highly reliable robotic tool reconfiguration.
>
---
#### [new 026] Distributed State Estimation for Vision-Based Cooperative Slung Load Transportation in GPS-Denied Environments
- **分类: cs.RO**

- **简介: 该论文属于多旋翼协同运输任务，解决GPS缺失环境下的负载状态估计问题。通过视觉传感器和分布式滤波方法实现鲁棒的负载状态估计与控制。**

- **链接: [https://arxiv.org/pdf/2603.04571](https://arxiv.org/pdf/2603.04571)**

> **作者:** Jack R. Pence; Jackson Fezell; Jack W. Langelaan; Junyi Geng
>
> **备注:** In proceedings of the 2026 AIAA SciTech Forum, Session: Intelligent Systems-27
>
> **摘要:** Transporting heavy or oversized slung loads using rotorcraft has traditionally relied on single-aircraft systems, which limits both payload capacity and control authority. Cooperative multilift using teams of rotorcraft offers a scalable and efficient alternative, especially for infrequent but challenging "long-tail" payloads without the need of building larger and larger rotorcraft. Most prior multilift research assumes GPS availability, uses centralized estimation architectures, or relies on controlled laboratory motion-capture setups. As a result, these methods lack robustness to sensor loss and are not viable in GPS-denied or operationally constrained environments. This paper addresses this limitation by presenting a distributed and decentralized payload state estimation framework for vision-based multilift operations. Using onboard monocular cameras, each UAV detects a fiducial marker on the payload and estimates its relative pose. These measurements are fused via a Distributed and Decentralized Extended Information Filter (DDEIF), enabling robust and scalable estimation that is resilient to individual sensor dropouts. This payload state estimate is then used for closed-loop trajectory tracking control. Monte Carlo simulation results in Gazebo show the effectiveness of the proposed approach, including the effect of communication loss during flight.
>
---
#### [new 027] Accelerating Sampling-Based Control via Learned Linear Koopman Dynamics
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于控制任务，旨在提升采样式控制的效率。通过引入学习的线性Koopman模型替代非线性动力学，实现更快的轨迹采样与计算，降低计算成本，提高实时控制性能。**

- **链接: [https://arxiv.org/pdf/2603.05385](https://arxiv.org/pdf/2603.05385)**

> **作者:** Wenjian Hao; Yuxuan Fang; Zehui Lu; Shaoshuai Mou
>
> **摘要:** This paper presents an efficient model predictive path integral (MPPI) control framework for systems with complex nonlinear dynamics. To improve the computational efficiency of classic MPPI while preserving control performance, we replace the nonlinear dynamics used for trajectory propagation with a learned linear deep Koopman operator (DKO) model, enabling faster rollout and more efficient trajectory sampling. The DKO dynamics are learned directly from interaction data, eliminating the need for analytical system models. The resulting controller, termed MPPI-DK, is evaluated in simulation on pendulum balancing and surface vehicle navigation tasks, and validated on hardware through reference-tracking experiments on a quadruped robot. Experimental results demonstrate that MPPI-DK achieves control performance close to MPPI with true dynamics while substantially reducing computational cost, enabling efficient real-time control on robotic platforms.
>
---
#### [new 028] Critic in the Loop: A Tri-System VLA Framework for Robust Long-Horizon Manipulation
- **分类: cs.RO**

- **简介: 该论文属于视觉机器人操作任务，解决高语义推理与低层控制的平衡问题。提出Tri-System框架，结合VLM、VLA和视觉Critic，提升长时程操作的鲁棒性与自主性。**

- **链接: [https://arxiv.org/pdf/2603.05185](https://arxiv.org/pdf/2603.05185)**

> **作者:** Pengfei Yi; Yingjie Ma; Wenjiang Xu; Yanan Hao; Shuai Gan; Wanting Li; Shanlin Zhong
>
> **摘要:** Balancing high-level semantic reasoning with low-level reactive control remains a core challenge in visual robotic manipulation. While Vision-Language Models (VLMs) excel at cognitive planning, their inference latency precludes real-time execution. Conversely, fast Vision-Language-Action (VLA) models often lack the semantic depth required for complex, long-horizon tasks. To bridge this gap, we introduce Critic in the Loop, an adaptive hierarchical framework driven by dynamic VLM-Expert scheduling. At its core is a bionic Tri-System architecture comprising a VLM brain for global reasoning, a VLA cerebellum for reactive execution, and a lightweight visual Critic. By continuously monitoring the workspace, the Critic dynamically routes control authority. It sustains rapid closed-loop execution via the VLA for routine subtasks, and adaptively triggers the VLM for replanning upon detecting execution anomalies such as task stagnation or failures. Furthermore, our architecture seamlessly integrates human-inspired rules to intuitively break infinite retry loops. This visually-grounded scheduling minimizes expensive VLM queries, while substantially enhancing system robustness and autonomy in out-of-distribution (OOD) scenarios. Comprehensive experiments on challenging, long-horizon manipulation benchmarks reveal that our approach achieves state-of-the-art performance.
>
---
#### [new 029] Observer Design for Augmented Reality-based Teleoperation of Soft Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人 teleoperation 任务，旨在解决软体机器人在增强现实环境中的操作问题。提出基于 HoloLens 2 的界面，实现对软体机械臂的精确控制与交互。**

- **链接: [https://arxiv.org/pdf/2603.05015](https://arxiv.org/pdf/2603.05015)**

> **作者:** Jorge Francisco García-Samartín; Iago López Pérez; Emirhan Yolcu; Jaime del Cerro; Antonio Barrientos
>
> **摘要:** Although virtual and augmented reality are gaining traction as teleoperation tools for various types of robots, including manipulators and mobile robots, they are not being used for soft robots. The inherent difficulties of modelling soft robots mean that combining accurate and computationally efficient representations is very challenging. This paper presents an augmented reality interface for teleoperating these devices. The developed system consists of Microsoft HoloLens 2 glasses and a central computer responsible for calculations. Validation is performed on PETER, a highly modular pneumatic manipulator. Using data collected from sensors, the computer estimates the robot's position based on the physics of the virtual reality programme. Errors obtained are on the order of 5% of the robot's length, demonstrating that augmented reality facilitates operator interaction with soft manipulators and can be integrated into the control loop.
>
---
#### [new 030] GaussTwin: Unified Simulation and Correction with Gaussian Splatting for Robotic Digital Twins
- **分类: cs.RO**

- **简介: 该论文提出GaussTwin，解决机器人数字孪生中的真实与仿真一致性问题，结合物理模拟与视觉校正，提升跟踪精度和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.05108](https://arxiv.org/pdf/2603.05108)**

> **作者:** Yichen Cai; Paul Jansonnie; Cristiana de Farias; Oleg Arenz; Jan Peters
>
> **备注:** 8 pages, 4 figures, 3 tables, ICRA 2026
>
> **摘要:** Digital twins promise to enhance robotic manipulation by maintaining a consistent link between real-world perception and simulation. However, most existing systems struggle with the lack of a unified model, complex dynamic interactions, and the real-to-sim gap, which limits downstream applications such as model predictive control. Thus, we propose GaussTwin, a real-time digital twin that combines position-based dynamics with discrete Cosserat rod formulations for physically grounded simulation, and Gaussian splatting for efficient rendering and visual correction. By anchoring Gaussians to physical primitives and enforcing coherent SE(3) updates driven by photometric error and segmentation masks, GaussTwin achieves stable prediction-correction while preserving physical fidelity. Through experiments in both simulation and on a Franka Research 3 platform, we show that GaussTwin consistently improves tracking accuracy and robustness compared to shape-matching and rigid-only baselines, while also enabling downstream tasks such as push-based planning. These results highlight GaussTwin as a step toward unified, physically meaningful digital twins that can support closed-loop robotic interaction and learning.
>
---
#### [new 031] SeedPolicy: Horizon Scaling via Self-Evolving Diffusion Policy for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决长时序模仿学习中的性能下降问题。提出SeedPolicy，通过自进化注意力模块提升时间建模能力，实现高效、可扩展的长时序操作。**

- **链接: [https://arxiv.org/pdf/2603.05117](https://arxiv.org/pdf/2603.05117)**

> **作者:** Youqiang Gui; Yuxuan Zhou; Shen Cheng; Xinyang Yuan; Haoqiang Fan; Peng Cheng; Shuaicheng Liu
>
> **备注:** 16 pages, 13 figures
>
> **摘要:** Imitation Learning (IL) enables robots to acquire manipulation skills from expert demonstrations. Diffusion Policy (DP) models multi-modal expert behaviors but suffers performance degradation as observation horizons increase, limiting long-horizon manipulation. We propose Self-Evolving Gated Attention (SEGA), a temporal module that maintains a time-evolving latent state via gated attention, enabling efficient recurrent updates that compress long-horizon observations into a fixed-size representation while filtering irrelevant temporal information. Integrating SEGA into DP yields Self-Evolving Diffusion Policy (SeedPolicy), which resolves the temporal modeling bottleneck and enables scalable horizon extension with moderate overhead. On the RoboTwin 2.0 benchmark with 50 manipulation tasks, SeedPolicy outperforms DP and other IL baselines. Averaged across both CNN and Transformer backbones, SeedPolicy achieves 36.8% relative improvement in clean settings and 169% relative improvement in randomized challenging settings over the DP. Compared to vision-language-action models such as RDT with 1.2B parameters, SeedPolicy achieves competitive performance with one to two orders of magnitude fewer parameters, demonstrating strong efficiency and scalability. These results establish SeedPolicy as a state-of-the-art imitation learning method for long-horizon robotic manipulation. Code is available at: this https URL.
>
---
#### [new 032] Rethinking the Role of Collaborative Robots in Rehabilitation
- **分类: cs.RO**

- **简介: 该论文属于康复机器人研究任务，旨在拓展协作机器人在物理治疗中的应用，解决治疗资源不足与效率问题，提出机器人在治疗前、中、后全程辅助 therapist 和患者的工作模式。**

- **链接: [https://arxiv.org/pdf/2603.05252](https://arxiv.org/pdf/2603.05252)**

> **作者:** Vivek Gupte; Shalutha Rajapakshe; Emmanuel Senft
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** Current research on collaborative robots (cobots) in physical rehabilitation largely focuses on repeated motion training for people undergoing physical therapy (PuPT), even though these sessions include phases that could benefit from robotic collaboration and assistance. Meanwhile, access to physical therapy remains limited for people with disabilities and chronic illnesses. Cobots could support both PuPT and therapists, and improve access to therapy, yet their broader potential remains underexplored. We propose extending the scope of cobots by imagining their role in assisting therapists and PuPT before, during, and after a therapy session. We discuss how cobot assistance may lift access barriers by promoting ability-based therapy design and helping therapists manage their time and effort. Finally, we highlight challenges to realizing these roles, including advancing user-state understanding, ensuring safety, and integrating cobots into therapists' workflow. This view opens new research questions and opportunities to draw from the HRI community's advances in assistive robotics.
>
---
#### [new 033] Latent Policy Steering through One-Step Flow Policies
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于离线强化学习任务，旨在解决策略优化与行为约束的平衡问题。提出LPS方法，通过直接传递原始动作空间Q梯度实现高保真潜在策略改进，提升性能并减少调参需求。**

- **链接: [https://arxiv.org/pdf/2603.05296](https://arxiv.org/pdf/2603.05296)**

> **作者:** Hokyun Im; Andrey Kolobov; Jianlong Fu; Youngwoon Lee
>
> **备注:** Project Webpage : this https URL
>
> **摘要:** Offline reinforcement learning (RL) allows robots to learn from offline datasets without risky exploration. Yet, offline RL's performance often hinges on a brittle trade-off between (1) return maximization, which can push policies outside the dataset support, and (2) behavioral constraints, which typically require sensitive hyperparameter tuning. Latent steering offers a structural way to stay within the dataset support during RL, but existing offline adaptations commonly approximate action values using latent-space critics learned via indirect distillation, which can lose information and hinder convergence. We propose Latent Policy Steering (LPS), which enables high-fidelity latent policy improvement by backpropagating original-action-space Q-gradients through a differentiable one-step MeanFlow policy to update a latent-action-space actor. By eliminating proxy latent critics, LPS allows an original-action-space critic to guide end-to-end latent-space optimization, while the one-step MeanFlow policy serves as a behavior-constrained generative prior. This decoupling yields a robust method that works out-of-the-box with minimal tuning. Across OGBench and real-world robotic tasks, LPS achieves state-of-the-art performance and consistently outperforms behavioral cloning and strong latent steering baselines.
>
---
#### [new 034] Lifelong Language-Conditioned Robotic Manipulation Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，解决持续学习中技能遗忘问题。提出SkillsCrafter框架，通过知识保留与语义聚合，实现多技能持续学习与泛化。**

- **链接: [https://arxiv.org/pdf/2603.05160](https://arxiv.org/pdf/2603.05160)**

> **作者:** Xudong Wang; Zebin Han; Zhiyu Liu; Gan Li; Jiahua Dong; Baichen Liu; Lianqing Liu; Zhi Han
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Traditional language-conditioned manipulation agent sequential adaptation to new manipulation skills leads to catastrophic forgetting of old skills, limiting dynamic scene practical deployment. In this paper, we propose SkillsCrafter, a novel robotic manipulation framework designed to continually learn multiple skills while reducing catastrophic forgetting of old skills. Specifically, we propose a Manipulation Skills Adaptation to retain the old skills knowledge while inheriting the shared knowledge between new and old skills to facilitate learning of new skills. Meanwhile, we perform the singular value decomposition on the diverse skill instructions to obtain common skill semantic subspace projection matrices, thereby recording the essential semantic space of skills. To achieve forget-less and generalization manipulation, we propose a Skills Specialization Aggregation to compute inter-skills similarity in skill semantic subspaces, achieving aggregation of the previously learned skill knowledge for any new or unknown skill. Extensive experiments demonstrate the effectiveness and superiority of our proposed SkillsCrafter.
>
---
#### [new 035] Many-RRT*: Robust Joint-Space Trajectory Planning for Serial Manipulators
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决高自由度机械臂在关节空间中因正运动学非逆而产生的多解问题。通过扩展RRT*-Connect，同时规划多个目标，提高轨迹质量和成功率。**

- **链接: [https://arxiv.org/pdf/2603.04547](https://arxiv.org/pdf/2603.04547)**

> **作者:** Theodore M. Belmont; Benjamin A. Christie; Anton Netchaev
>
> **摘要:** The rapid advancement of high degree-of-freedom (DoF) serial manipulators necessitates the use of swift, sampling-based motion planners for high-dimensional spaces. While sampling-based planners like the Rapidly-Exploring Random Tree (RRT) are widely used, planning in the manipulator's joint space presents significant challenges due to non-invertible forward kinematics. A single task-space end-effector pose can correspond to multiple configuration-space states, creating a multi-arm bandit problem for the planner. In complex environments, simply choosing the wrong joint space goal can result in suboptimal trajectories or even failure to find a viable plan. To address this planning problem, we propose Many-RRT*: an extension of RRT*-Connect that plans to multiple goals in parallel. By generating multiple IK solutions and growing independent trees from these goal configurations simultaneously alongside a single start tree, Many-RRT* ensures that computational effort is not wasted on suboptimal IK solutions. This approach maintains robust convergence and asymptotic optimality. Experimental evaluations across robot morphologies and diverse obstacle environments demonstrate that Many-RRT* provides higher quality trajectories (44.5% lower cost in the same runtime) with a significantly higher success rate (100% vs. the next best of 1.6%) than previous RRT iterations without compromising on runtime performance.
>
---
#### [new 036] From Local Corrections to Generalized Skills: Improving Neuro-Symbolic Policies with MEMO
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决机器人因缺乏合适技能而无法执行复杂任务的问题。通过整合用户反馈，构建通用技能模板，提升机器人泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.04560](https://arxiv.org/pdf/2603.04560)**

> **作者:** Benjamin A. Christie; Yinlong Dai; Mohammad Bararjanianbahnamiri; Simon Stepputtis; Dylan P. Losey
>
> **摘要:** Recent works use a neuro-symbolic framework for general manipulation policies. The advantage of this framework is that -- by applying off-the-shelf vision and language models -- the robot can break complex tasks down into semantic subtasks. However, the fundamental bottleneck is that the robot needs skills to ground these subtasks into embodied motions. Skills can take many forms (e.g., trajectory snippets, motion primitives, coded functions), but regardless of their form skills act as a constraint. The high-level policy can only ground its language reasoning through the available skills; if the robot cannot generate the right skill for the current task, its policy will fail. We propose to address this limitation -- and dynamically expand the robot's skills -- by leveraging user feedback. When a robot fails, humans can intuitively explain what went wrong (e.g., ``no, go higher''). While a simple approach is to recall this exact text the next time the robot faces a similar situation, we hypothesize that by collecting, clustering, and re-phrasing natural language corrections across multiple users and tasks, we can synthesize more general text guidance and coded skill templates. Applying this hypothesis we develop Memory Enhanced Manipulation (MEMO). MEMO builds and maintains a retrieval-augmented skillbook gathered from human feedback and task successes. At run time, MEMO retrieves relevant text and code from this skillbook, enabling the robot's policy to generate new skills while reasoning over multi-task human feedback. Our experiments demonstrate that using MEMO to aggregate local feedback into general skill templates enables generalization to novel tasks where existing baselines fall short. See supplemental material here: this https URL
>
---
#### [new 037] Iterative On-Policy Refinement of Hierarchical Diffusion Policies for Language-Conditioned Manipulation
- **分类: cs.RO**

- **简介: 该论文属于语言条件操作任务，解决层次化策略中规划器与控制器不匹配的问题。提出HD-ExpIt框架，通过环境反馈迭代优化策略，提升性能。**

- **链接: [https://arxiv.org/pdf/2603.05291](https://arxiv.org/pdf/2603.05291)**

> **作者:** Clemence Grislain; Olivier Sigaud; Mohamed Chetouani
>
> **摘要:** Hierarchical policies for language-conditioned manipulation decompose tasks into subgoals, where a high-level planner guides a low-level controller. However, these hierarchical agents often fail because the planner generates subgoals without considering the actual limitations of the controller. Existing solutions attempt to bridge this gap via intermediate modules or shared representations, but they remain limited by their reliance on fixed offline datasets. We propose HD-ExpIt, a framework for iterative fine-tuning of hierarchical diffusion policies via environment feedback. HD-ExpIt organizes training into a self-reinforcing cycle: it utilizes diffusion-based planning to autonomously discover successful behaviors, which are then distilled back into the hierarchical policy. This loop enables both components to improve while implicitly grounding the planner in the controller's actual capabilities without requiring explicit proxy models. Empirically, HD-ExpIt significantly improves hierarchical policies trained solely on offline data, achieving state-of-the-art performance on the long-horizon CALVIN benchmark among methods trained from scratch.
>
---
#### [new 038] Act-Observe-Rewrite: Multimodal Coding Agents as In-Context Policy Learners for Robot Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出AOR框架，利用多模态语言模型在不依赖示范或奖励的情况下，通过自我反思改进机器人操作策略，解决机器人操控中的政策学习问题。**

- **链接: [https://arxiv.org/pdf/2603.04466](https://arxiv.org/pdf/2603.04466)**

> **作者:** Vaishak Kumar
>
> **摘要:** Can a multimodal language model learn to manipulate physical objects by reasoning about its own failures-without gradient updates, demonstrations, or reward engineering? We argue the answer is yes, under conditions we characterise precisely. We present Act-Observe-Rewrite (AOR), a framework in which an LLM agent improves a robot manipulation policy by synthesising entirely new executable Python controller code between trials, guided by visual observations and structured episode outcomes. Unlike prior work that grounds LLMs in pre-defined skill libraries or uses code generation for one-shot plan synthesis, AOR makes the full low-level motor control implementation the unit of LLM reasoning, enabling the agent to change not just what the robot does, but how it does it. The central claim is that interpretable code as the policy representation creates a qualitatively different kind of in-context learning from opaque neural policies: the agent can diagnose systematic failures and rewrite their causes. We validate this across three robosuite manipulation tasks and report promising results, with the agent achieving high success rates without demonstrations, reward engineering, or gradient updates.
>
---
#### [new 039] PTLD: Sim-to-real Privileged Tactile Latent Distillation for Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉操作任务，旨在解决模拟到现实的触觉操控迁移问题。通过引入PTLD方法，利用真实触觉数据提升仿真中本体感觉策略的性能。**

- **链接: [https://arxiv.org/pdf/2603.04531](https://arxiv.org/pdf/2603.04531)**

> **作者:** Rosy Chen; Mustafa Mukadam; Michael Kaess; Tingfan Wu; Francois R Hogan; Jitendra Malik; Akash Sharma
>
> **摘要:** Tactile dexterous manipulation is essential to automating complex household tasks, yet learning effective control policies remains a challenge. While recent work has relied on imitation learning, obtaining high quality demonstrations for multi-fingered hands via robot teleoperation or kinesthetic teaching is prohibitive. Alternatively, with reinforcement we can learn skills in simulation, but fast and realistic simulation of tactile observations is challenging. To bridge this gap, we introduce PTLD: sim-to-real Privileged Tactile Latent Distillation, a novel approach to learning tactile manipulation skills without requiring tactile simulation. Instead of simulating tactile sensors or relying purely on proprioceptive policies to transfer zero-shot sim-to-real, our key idea is to leverage privileged sensors in the real world to collect real-world tactile policy data. This data is then used to distill a robust state estimator that operates on tactile input. We demonstrate from our experiments that PTLD can be used to improve proprioceptive manipulation policies trained in simulation significantly by incorporating tactile sensing. On the benchmark in-hand rotation task, PTLD achieves a 182% improvement over a proprioception only policy. We also show that PTLD enables learning the challenging task of tactile in-hand reorientation where we see a 57% improvement in the number of goals reached over using proprioception alone. Website: this https URL.
>
---
#### [new 040] PhysiFlow: Physics-Aware Humanoid Whole-Body VLA via Multi-Brain Latent Flow Matching and Robust Tracking
- **分类: cs.RO**

- **简介: 该论文属于人形机器人控制任务，旨在解决VLA与全身控制融合中的语义引导不足和效率低的问题。提出一种物理感知的多脑VLA框架，提升动态协调任务的稳定性。**

- **链接: [https://arxiv.org/pdf/2603.05410](https://arxiv.org/pdf/2603.05410)**

> **作者:** Weikai Qin; Sichen Wu; Ci Chen; Mengfan Liu; Linxi Feng; Xinru Cui; Haoqi Han; Hesheng Wang
>
> **摘要:** In the domain of humanoid robot control, the fusion of Vision-Language-Action (VLA) with whole-body control is essential for semantically guided execution of real-world tasks. However, existing methods encounter challenges in terms of low VLA inference efficiency or an absence of effective semantic guidance for whole-body control, resulting in instability in dynamic limb-coordinated tasks. To bridge this gap, we present a semantic-motion intent guided, physics-aware multi-brain VLA framework for humanoid whole-body control. A series of experiments was conducted to evaluate the performance of the proposed framework. The experimental results demonstrated that the framework enabled reliable vision-language-guided full-body coordination for humanoid robots.
>
---
#### [new 041] On the Strengths and Weaknesses of Data for Open-set Embodied Assistance
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究开放集辅助任务，旨在提升模型对新用户行为和新场景的适应能力。通过合成数据训练模型，探索其在未知类别和配置下的泛化性能。**

- **链接: [https://arxiv.org/pdf/2603.04819](https://arxiv.org/pdf/2603.04819)**

> **作者:** Pradyumna Tambwekar; Andrew Silva; Deepak Gopinath; Jonathan DeCastro; Xiongyi Cui; Guy Rosman
>
> **摘要:** Embodied foundation models are increasingly performant in real-world domains such as robotics or autonomous driving. These models are often deployed in interactive or assistive settings, where it is important that these assistive models generalize to new users and new tasks. Diverse interactive data generation offers a promising avenue for providing data-efficient generalization capabilities for interactive embodied foundation models. In this paper, we investigate the generalization capabilities of a multimodal foundation model fine-tuned on diverse interactive assistance data in a synthetic domain. We explore generalization along two axes: a) assistance with unseen categories of user behavior and b) providing guidance in new configurations not encountered during training. We study a broad capability called \textbf{Open-Set Corrective Assistance}, in which the model needs to inspect lengthy user behavior and provide assistance through either corrective actions or language-based feedback. This task remains unsolved in prior work, which typically assumes closed corrective categories or relies on external planners, making it a challenging testbed for evaluating the limits of assistive data. To support this task, we generate synthetic assistive datasets in Overcooked and fine-tune a LLaMA-based model to evaluate generalization to novel tasks and user behaviors. Our approach provides key insights into the nature of assistive datasets required to enable open-set assistive intelligence. In particular, we show that performant models benefit from datasets that cover different aspects of assistance, including multimodal grounding, defect inference, and exposure to diverse scenarios.
>
---
#### [new 042] LEGS-POMDP: Language and Gesture-Guided Object Search in Partially Observable Environments
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于开放世界物体搜索任务，旨在解决模糊指令下的不确定性问题。通过整合语言、手势和视觉信息，构建LEGS-POMDP系统，提升机器人在部分可观测环境中的搜索能力。**

- **链接: [https://arxiv.org/pdf/2603.04705](https://arxiv.org/pdf/2603.04705)**

> **作者:** Ivy Xiao He; Stefanie Tellex; Jason Xinyu Liu
>
> **备注:** 10 pages, 8 figures, accepted at ACM/IEEE International Conference on Human-Robot Interaction (HRI 2026)
>
> **摘要:** To assist humans in open-world environments, robots must interpret ambiguous instructions to locate desired objects. Foundation model-based approaches excel at multimodal grounding, but they lack a principled mechanism for modeling uncertainty in long-horizon tasks. In contrast, Partially Observable Markov Decision Processes (POMDPs) provide a systematic framework for planning under uncertainty but are often limited in supported modalities and rely on restrictive environment assumptions. We introduce LanguagE and Gesture-Guided Object Search in Partially Observable Environments (LEGS-POMDP), a modular POMDP system that integrates language, gesture, and visual observations for open-world object search. Unlike prior work, LEGS-POMDP explicitly models two sources of partial observability: uncertainty over the target object's identity and its spatial location. In simulation, multimodal fusion significantly outperforms unimodal baselines, achieving an average success rate of 89\% across challenging environments and object categories. Finally, we demonstrate the full system on a quadruped mobile manipulator, where real-world experiments qualitatively validate robust multimodal perception and uncertainty reduction under ambiguous instructions.
>
---
#### [new 043] OpenFrontier: General Navigation with Visual-Language Grounded Frontiers
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人导航任务，解决复杂环境中高效导航问题。提出OpenFrontier框架，无需训练和精细调优，利用视觉语言先验实现高效导航。**

- **链接: [https://arxiv.org/pdf/2603.05377](https://arxiv.org/pdf/2603.05377)**

> **作者:** Esteban Padilla; Boyang Sun; Marc Pollefeys; Hermann Blum
>
> **摘要:** Open-world navigation requires robots to make decisions in complex everyday environments while adapting to flexible task requirements. Conventional navigation approaches often rely on dense 3D reconstruction and hand-crafted goal metrics, which limits their generalization across tasks and environments. Recent advances in vision--language navigation (VLN) and vision--language--action (VLA) models enable end-to-end policies conditioned on natural language, but typically require interactive training, large-scale data collection, or task-specific fine-tuning with a mobile agent. We formulate navigation as a sparse subgoal identification and reaching problem and observe that providing visual anchoring targets for high-level semantic priors enables highly efficient goal-conditioned navigation. Based on this insight, we select navigation frontiers as semantic anchors and propose OpenFrontier, a training-free navigation framework that seamlessly integrates diverse vision--language prior models. OpenFrontier enables efficient navigation with a lightweight system design, without dense 3D mapping, policy training, or model fine-tuning. We evaluate OpenFrontier across multiple navigation benchmarks and demonstrate strong zero-shot performance, as well as effective real-world deployment on a mobile robot.
>
---
#### [new 044] Omni-Manip: Beyond-FOV Large-Workspace Humanoid Manipulation with Omnidirectional 3D Perception
- **分类: cs.RO**

- **简介: 该论文属于人形机器人操作任务，旨在解决大工作空间内感知受限的问题。通过LiDAR驱动的3D视觉策略，实现全方位感知与精准操作。**

- **链接: [https://arxiv.org/pdf/2603.05355](https://arxiv.org/pdf/2603.05355)**

> **作者:** Pei Qu; Zheng Li; Yufei Jia; Ziyun Liu; Liang Zhu; Haoang Li; Jinni Zhou; Jun Ma
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** The deployment of humanoid robots for dexterous manipulation in unstructured environments remains challenging due to perceptual limitations that constrain the effective workspace. In scenarios where physical constraints prevent the robot from repositioning itself, maintaining omnidirectional awareness becomes far more critical than color or semantic information. While recent advances in visuomotor policy learning have improved manipulation capabilities, conventional RGB-D solutions suffer from narrow fields of view (FOV) and self-occlusion, requiring frequent base movements that introduce motion uncertainty and safety risks. Existing approaches to expanding perception, including active vision systems and third-view cameras, introduce mechanical complexity, calibration dependencies, and latency that hinder reliable real-time performance. In this work, We propose Omni-Manip, an end-to-end LiDAR-driven 3D visuomotor policy that enables robust manipulation in large workspaces. Our method processes panoramic point clouds through a Time-Aware Attention Pooling mechanism, efficiently encoding sparse 3D data while capturing temporal dependencies. This 360° perception allows the robot to interact with objects across wide areas without frequent repositioning. To support policy learning, we develop a whole-body teleoperation system for efficient data collection on full-body coordination. Extensive experiments in simulation and real-world environments show that Omni-Manip achieves robust performance in large-workspace and cluttered scenarios, outperforming baselines that rely on egocentric depth cameras.
>
---
#### [new 045] Design, Mapping, and Contact Anticipation with 3D-printed Whole-Body Tactile and Proximity Sensors
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决接触预测问题。提出一种3D打印的全身体感传感器，实现接触检测与提前感知。**

- **链接: [https://arxiv.org/pdf/2603.04714](https://arxiv.org/pdf/2603.04714)**

> **作者:** Carson Kohlbrenner; Anna Soukhovei; Caleb Escobedo; Nataliya Nechyporenko; Alessandro Roncone
>
> **备注:** This work was accepted at the International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Robots operating in dynamic and shared environments benefit from anticipating contact before it occurs. We present GenTact-Prox, a fully 3D-printed artificial skin that integrates tactile and proximity sensing for contact detection and anticipation. The artificial skin platform is modular in design, procedurally generated to fit any robot morphology, and can cover the whole body of a robot. The skin achieved detection ranges of up to 18 cm during evaluation. To characterize how robots perceive nearby space through this skin, we introduce a data-driven framework for mapping the Perisensory Space -- the body-centric volume of space around the robot where sensors provide actionable information for contact anticipation. We demonstrate this approach on a Franka Research 3 robot equipped with five GenTact-Prox units, enabling online object-aware operation and contact prediction.
>
---
#### [new 046] VinePT-Map: Pole-Trunk Semantic Mapping for Resilient Autonomous Robotics in Vineyards
- **分类: cs.RO**

- **简介: 该论文属于农业机器人定位与建图任务，旨在解决 vineyard 中因季节变化导致的定位不稳问题。通过利用葡萄藤和支柱作为语义地标，构建鲁棒的长期定位系统。**

- **链接: [https://arxiv.org/pdf/2603.05070](https://arxiv.org/pdf/2603.05070)**

> **作者:** Giorgio Audrito; Mauro Martini; Alessandro Navone; Giorgia Galluzzo; Marcello Chiaberge
>
> **摘要:** Reliable long-term deployment of autonomous robots in agricultural environments remains challenging due to perceptual aliasing, seasonal variability, and the dynamic nature of crop canopies. Vineyards, characterized by repetitive row structures and significant visual changes across phenological stages, represent a pivotal field challenge, limiting the robustness of conventional feature-based localization and mapping approaches. This paper introduces VinePT-Map, a semantic mapping framework that leverages vine trunks and support poles as persistent structural landmarks to enable season-agnostic and resilient robot localization. The proposed method formulates the mapping problem as a factor graph, integrating GPS, IMU, and RGB-D observations through robust geometrical constraints that exploit vineyard structure. An efficient perception pipeline based on instance segmentation and tracking, combined with a clustering filter for outlier rejection and pose refinement, enables accurate landmark detection using low-cost sensors and onboard computation. To validate the pipeline, we present a multi-season dataset for trunk and pole segmentation and tracking. Extensive field experiments conducted across diverse seasons demonstrate the robustness and accuracy of the proposed approach, highlighting its suitability for long-term autonomous operation in agricultural environments.
>
---
#### [new 047] Loop Closure via Maximal Cliques in 3D LiDAR-Based SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D LiDAR SLAM任务，解决 loop closure 检测问题。提出 CliReg 算法，通过最大团搜索替代 RANSAC，提升鲁棒性与精度。**

- **链接: [https://arxiv.org/pdf/2603.05397](https://arxiv.org/pdf/2603.05397)**

> **作者:** Javier Laserna; Saurabh Gupta; Oscar Martinez Mozos; Cyrill Stachniss; Pablo San Segundo
>
> **备注:** Accepted in the 2025 European Conference on Mobile Robots (ECMR). This is the author's version of the work
>
> **摘要:** Reliable loop closure detection remains a critical challenge in 3D LiDAR-based SLAM, especially under sensor noise, environmental ambiguity, and viewpoint variation conditions. RANSAC is often used in the context of loop closures for geometric model fitting in the presence of outliers. However, this approach may fail, leading to map inconsistency. We introduce a novel deterministic algorithm, CliReg, for loop closure validation that replaces RANSAC verification with a maximal clique search over a compatibility graph of feature correspondences. This formulation avoids random sampling and increases robustness in the presence of noise and outliers. We integrated our approach into a real- time pipeline employing binary 3D descriptors and a Hamming distance embedding binary search tree-based matching. We evaluated it on multiple real-world datasets featuring diverse LiDAR sensors. The results demonstrate that our proposed technique consistently achieves a lower pose error and more reliable loop closures than RANSAC, especially in sparse or ambiguous conditions. Additional experiments on 2D projection-based maps confirm its generality across spatial domains, making our approach a robust and efficient alternative for loop closure detection.
>
---
#### [new 048] Integrated cooperative localization of heterogeneous measurement swarm: A unified data-driven method
- **分类: cs.RO**

- **简介: 该论文属于协同定位任务，解决异构机器人系统中因传感器差异导致的定位难题。通过数据驱动方法实现配对相对定位，并设计分布式定位策略，提升在弱连通拓扑下的定位性能。**

- **链接: [https://arxiv.org/pdf/2603.04932](https://arxiv.org/pdf/2603.04932)**

> **作者:** Kunrui Ze; Wei Wang; Guibin Sun; Jiaqi Yan; Kexin Liu; Jinhu Lü
>
> **摘要:** The cooperative localization (CL) problem in heterogeneous robotic systems with different measurement capabilities is investigated in this work. In practice, heterogeneous sensors lead to directed and sparse measurement topologies, whereas most existing CL approaches rely on multilateral localization with restrictive multi-neighbor geometric requirements. To overcome this limitation, we enable pairwise relative localization (RL) between neighboring robots using only mutual measurement and odometry information. A unified data-driven adaptive RL estimator is first developed to handle heterogeneous and unidirectional measurements. Based on the convergent RL estimates, a distributed pose-coupling CL strategy is then designed, which guarantees CL under a weakly connected directed measurement topology, representing the least restrictive condition among existing results. The proposed method is independent of specific control tasks and is validated through a formation control application and real-world experiments.
>
---
#### [new 049] GAIDE: Graph-based Attention Masking for Spatial- and Embodiment-aware Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于运动规划任务，旨在解决高维空间中采样效率低的问题。提出GAIDE方法，利用图结构和注意力机制提升规划效率与成功率。**

- **链接: [https://arxiv.org/pdf/2603.04463](https://arxiv.org/pdf/2603.04463)**

> **作者:** Davood Soleymanzadeh; Xiao Liang; Minghui Zheng
>
> **摘要:** Sampling-based motion planning algorithms are widely used for motion planning of robotic manipulators, but they often struggle with sample inefficiency in high-dimensional configuration spaces due to their reliance on uniform or hand-crafted informed sampling primitives. Neural informed samplers address this limitation by learning the sampling distribution from prior planning experience to guide the motion planner towards planning goal. However, existing approaches often struggle to encode the spatial structure inherent in motion planning problems. To address this limitation, we introduce Graph-based Attention Masking for Spatial- and Embodiment-aware Motion Planning (GAIDE), a neural informed sampler that leverages both the spatial structure of the planning problem and the robotic manipulator's embodiment to guide the planning algorithm. GAIDE represents these structures as a graph and integrates it into a transformer-based neural sampler through attention masking. We evaluate GAIDE against baseline state-of-the-art sampling-based planners using uniform sampling, hand-crafted informed sampling, and neural informed sampling primitives. Evaluation results demonstrate that GAIDE improves planning efficiency and success rate.
>
---
#### [new 050] Constraint-Free Static Modeling of Continuum Parallel Robot
- **分类: cs.RO**

- **简介: 该论文属于机器人建模任务，解决连续体并联机器人静态建模难题。通过几何方法建立无约束模型，处理大变形和旋转情况，提升仿真与控制精度。**

- **链接: [https://arxiv.org/pdf/2603.05309](https://arxiv.org/pdf/2603.05309)**

> **作者:** Lingxiao Xun; Matyas Diezinger; Azad Artinian; Guillaume Laurent; Brahim Tamadazte
>
> **摘要:** Continuum parallel robots (CPR) combine rigid actuation mechanisms with multiple elastic rods in a closed-loop topology, making forward statics challenging when rigid--continuum junctions are enforced by explicit kinematic constraints. Such constraint-based formulations typically introduce additional algebraic variables and complicate both numerical solution and downstream control. This paper presents a geometric exact, configuration-based and constraint-free static model of CPR that remains valid under geometrically nonlinear, large-deformation and large-rotation conditions. Connectivity constraints are eliminated by kinematic embedding, yielding a reduced unconstrained problem. Each rod of CPR is discretized by nodal poses on SE(3), while the element-wise strain field is reconstructed through a linear strain parameterization. A fourth-order Magnus approximation yields an explicit and geometrically consistent mapping between element end poses and the strain. Rigid attachments at the motor-driven base and the end-effector platforms are handled through kinematic embeddings. Based on total potential energy and virtual work, we derive assembly-ready residuals and explicit Newton tangents, and solve the resulting nonlinear equilibrium equations using a Riemannian Newton iteration on the product manifold. Experiments on a three-servomotor, six-rod prototype validate the model by showing good agreement between simulation and measurements for both unloaded motions and externally loaded cases.
>
---
#### [new 051] Observing and Controlling Features in Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文研究视觉-语言-动作模型的可解释性，解决VLAs内部结构难以理解的问题。通过引入特征可观测性和可控性，实现对模型输出的精准控制。**

- **链接: [https://arxiv.org/pdf/2603.05487](https://arxiv.org/pdf/2603.05487)**

> **作者:** Hugo Buurmeijer; Carmen Amo Alonso; Aiden Swann; Marco Pavone
>
> **摘要:** Vision-Language-Action Models (VLAs) have shown remarkable progress towards embodied intelligence. While their architecture partially resembles that of Large Language Models (LLMs), VLAs exhibit higher complexity due to their multi-modal inputs/outputs and often hybrid nature of transformer and diffusion heads. This is part of the reason why insights from mechanistic interpretability in LLMs, which explain how the internal model representations relate to their output behavior, do not trivially transfer to VLA counterparts. In this work, we propose to close this gap by introducing and analyzing two main concepts: feature-observability and feature-controllability. In particular, we first study features that are linearly encoded in representation space, and show how they can be observed by means of a linear classifier. Then, we use a minimal linear intervention grounded in optimal control to accurately place internal representations and steer the VLA's output towards a desired region. Our results show that targeted, lightweight interventions can reliably steer a robot's behavior while preserving closed-loop capabilities. We demonstrate on different VLA architectures ($\pi_{0.5}$ and OpenVLA) through simulation experiments that VLAs possess interpretable internal structure amenable to online adaptation without fine-tuning, enabling real-time alignment with user preferences and task requirements.
>
---
#### [new 052] ROScopter: A Multirotor Autopilot based on ROSflight 2.0
- **分类: cs.RO**

- **简介: 该论文提出ROScopter，一个基于ROSflight 2.0的多旋翼自动驾驶仪，用于加速研究代码的仿真与硬件测试，解决代码模块化与易修改问题。**

- **链接: [https://arxiv.org/pdf/2603.05404](https://arxiv.org/pdf/2603.05404)**

> **作者:** Jacob Moore; Ian Reid; Phil Tokumaru; Tim McLain
>
> **摘要:** ROScopter is a lean multirotor autopilot built for researchers. ROScopter seeks to accelerate simulation and hardware testing of research code with an architecture that is both easy to understand and simple to modify. ROScopter is designed to interface with ROSflight 2.0 and runs entirely on an onboard flight computer, leveraging the features of ROS 2 to improve modularity. This work describes the architecture of ROScopter and how it can be used to test application code in both simulated and hardware environments. Hardware results of the default ROScopter behavior are presented, showing that ROScopter achieves similar performance to another state-of-the-art autopilot for basic waypoint-following maneuvers, but with a significantly reduced and more modular code-base.
>
---
#### [new 053] Gait Generation Balancing Joint Load and Mobility for Legged Modular Robots with Easily Detachable Joints
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，旨在解决模块化机器人在运动中关节负载过大的问题。通过优化框架降低关节负载，同时保持运动速度与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.04757](https://arxiv.org/pdf/2603.04757)**

> **作者:** Kennosuke Chihara; Takuya Kiyokawa; Kensuke Harada
>
> **备注:** 6 pages, 7 figures
>
> **摘要:** While modular robots offer versatility, excessive joint torque during locomotion poses a significant risk of mechanical failure, especially for detachable joints. To address this, we propose an optimization framework using the NSGA-III algorithm. Unlike conventional approaches that prioritize mobility alone, our method derives Pareto optimal solutions to minimize joint load while maintaining necessary locomotion speed and stability. Simulations and physical experiments demonstrate that our approach successfully generates gait motions for diverse environments, such as slopes and steps, ensuring structural integrity without compromising overall mobility.
>
---
#### [new 054] Data-Driven Control of a Magnetically Actuated Fish-Like Robot
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于控制任务，旨在解决磁驱动仿鱼机器人的精确控制问题。通过数据驱动方法构建模型并优化控制策略，提升其路径跟踪性能。**

- **链接: [https://arxiv.org/pdf/2603.04787](https://arxiv.org/pdf/2603.04787)**

> **作者:** Akiyuki Koyama; Hiroaki Kawashima
>
> **备注:** Author's version of the paper presented at AROB-ISBC 2026
>
> **摘要:** Magnetically actuated fish-like robots offer promising solutions for underwater exploration due to their miniaturization and agility; however, precise control remains a significant challenge because of nonlinear fluid dynamics, flexible fin hysteresis, and the variable-duration control steps inherent to the actuation mechanism. This paper proposes a comprehensive data-driven control framework to address these complexities without relying on analytical modeling. Our methodology comprises three core components: 1) developing a forward dynamics model (FDM) using a neural network trained on real-world experimental data to capture state transitions under varying time steps; 2) integrating this FDM into a gradient-based model predictive control (G-MPC) architecture to optimize control inputs for path following; and 3) applying imitation learning to approximate the G-MPC policy, thereby reducing the computational cost for real-time implementation. We validate the approach through simulations utilizing the identified dynamics model. The results demonstrate that the G-MPC framework achieves accurate path convergence with minimal root mean square error (RMSE), and the imitation learning controller (ILC) effectively replicates this performance. This study highlights the potential of data-driven control strategies for the precise navigation of miniature, fish-like soft robots.
>
---
#### [new 055] Efficient Autonomous Navigation of a Quadruped Robot in Underground Mines on Edge Hardware
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决地下矿井中四足机器人在无GPS、低功耗设备上的高效导航问题。提出一种完全自主的导航系统，实现实时路径规划与避障。**

- **链接: [https://arxiv.org/pdf/2603.04470](https://arxiv.org/pdf/2603.04470)**

> **作者:** Yixiang Gao; Kwame Awuah-Offei
>
> **摘要:** Embodied navigation in underground mines faces significant challenges, including narrow passages, uneven terrain, near-total darkness, GPS-denied conditions, and limited communication infrastructure. While recent learning-based approaches rely on GPU-accelerated inference and extensive training data, we present a fully autonomous navigation stack for a Boston Dynamics Spot quadruped robot that runs entirely on a low-power Intel NUC edge computer with no GPU and no network connectivity requirements. The system integrates LiDAR-inertial odometry, scan-matching localization against a prior map, terrain segmentation, and visibility-graph global planning with a velocity-regulated local path follower, achieving real-time perception-to-action at consistent control rates. After a single mapping pass of the environment, the system handles arbitrary goal locations within the known map without any environment-specific training or learned components. We validate the system through repeated field trials using four target locations of varying traversal difficulty in an experimental underground mine, accumulating over 700 m of fully autonomous traverse with a 100% success rate across all 20 trials (5 repetitions x 4 targets) and an overall Success weighted by Path Length (SPL) of 0.73 \pm 0.09.
>
---
#### [new 056] Task-Relevant and Irrelevant Region-Aware Augmentation for Generalizable Vision-Based Imitation Learning in Agricultural Manipulation
- **分类: cs.RO**

- **简介: 论文提出DRAIL方法，解决农业操作中视觉模仿学习的泛化问题。通过区分任务相关与无关区域进行增强，提升政策对关键特征的依赖，增强鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.04845](https://arxiv.org/pdf/2603.04845)**

> **作者:** Shun Hattori; Hikaru Sasaki; Takumi Hachimine; Yusuke Mizutani; Takamitsu Matsubara
>
> **摘要:** Vision-based imitation learning has shown promise for robotic manipulation; however, its generalization remains limited in practical agricultural tasks. This limitation stems from scarce demonstration data and substantial visual domain gaps caused by i) crop-specific appearance diversity and ii) background variations. To address this limitation, we propose Dual-Region Augmentation for Imitation Learning (DRAIL), a region-aware augmentation framework designed for generalizable vision-based imitation learning in agricultural manipulation. DRAIL explicitly separates visual observations into task-relevant and task-irrelevant regions. The task-relevant region is augmented in a domain-knowledge-driven manner to preserve essential visual characteristics, while the task-irrelevant region is aggressively randomized to suppress spurious background correlations. By jointly handling both sources of visual variation, DRAIL promotes learning policies that rely on task-essential features rather than incidental visual cues. We evaluate DRAIL on diffusion policy-based visuomotor controllers through robot experiments on artificial vegetable harvesting and real lettuce defective leaf picking preparation tasks. The results show consistent improvements in success rates under unseen visual conditions compared to baseline methods. Further attention analysis and representation generalization metrics indicate that the learned policies rely more on task-essential visual features, resulting in enhanced robustness and generalization.
>
---
#### [new 057] Causally Robust Reward Learning from Reason-Augmented Preference Feedback
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于强化学习中的奖励建模任务，解决偏好反馈稀疏导致的因果混淆问题。通过引入自然语言理由，增强奖励模型的因果鲁棒性，提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.04861](https://arxiv.org/pdf/2603.04861)**

> **作者:** Minjune Hwang; Yigit Korkmaz; Daniel Seita; Erdem Bıyık
>
> **备注:** Published in International Conference on Learning Representations (ICLR) 2026
>
> **摘要:** Preference-based reward learning is widely used for shaping agent behavior to match a user's preference, yet its sparse binary feedback makes it especially vulnerable to causal confusion. The learned reward often latches onto spurious features that merely co-occur with preferred trajectories during training, collapsing when those correlations disappear or reverse at test time. We introduce ReCouPLe, a lightweight framework that uses natural language rationales to provide the missing causal signal. Each rationale is treated as a guiding projection axis in an embedding space, training the model to score trajectories based on features aligned with that axis while de-emphasizing context that is unrelated to the stated reason. Because the same rationales (e.g., "avoids collisions", "completes the task faster") can appear across multiple tasks, ReCouPLe naturally reuses the same causal direction whenever tasks share semantics, and transfers preference knowledge to novel tasks without extra data or language-model fine-tuning. Our learned reward model can ground preferences on the articulated reason, aligning better with user intent and generalizing beyond spurious features. ReCouPLe outperforms baselines by up to 1.5x in reward accuracy under distribution shifts, and 2x in downstream policy performance in novel tasks. We have released our code at this https URL
>
---
#### [new 058] Risk-Aware Rulebooks for Multi-Objective Trajectory Evaluation under Uncertainty
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于多目标轨迹评估任务，解决不确定环境下系统轨迹的评价问题。通过构建风险感知的规则体系，明确轨迹与环境的相互影响，提升决策的解释性。**

- **链接: [https://arxiv.org/pdf/2603.04603](https://arxiv.org/pdf/2603.04603)**

> **作者:** Tichakorn Wongpiromsarn
>
> **摘要:** We present a risk-aware formalism for evaluating system trajectories in the presence of uncertain interactions between the system and its environment. The proposed formalism supports reasoning under uncertainty and systematically handles complex relationships among requirements and objectives, including hierarchical priorities and non-comparability. Rather than treating the environment as exogenous noise, we explicitly model how each system trajectory influences the environment and evaluate trajectories under the resulting distribution of environment responses. We prove that the formalism induces a preorder on the set of system trajectories, ensuring consistency and preventing cyclic preferences. Finally, we illustrate the approach with an autonomous driving example that demonstrates how the formalism enhances explainability by clarifying the rationale behind trajectory selection.
>
---
#### [new 059] CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多摄像头3D目标检测任务，解决模型在不同摄像头配置下泛化能力差的问题。提出CoIn3D框架，通过空间感知特征调制和相机感知数据增强提升跨配置性能。**

- **链接: [https://arxiv.org/pdf/2603.05042](https://arxiv.org/pdf/2603.05042)**

> **作者:** Zhaonian Kuang; Rui Ding; Haotian Wang; Xinhu Zheng; Meng Yang; Gang Hua
>
> **备注:** Accepted to CVPR 2026 main track
>
> **摘要:** Multi-camera 3D object detection (MC3D) has attracted increasing attention with the growing deployment of multi-sensor physical agents, such as robots and autonomous vehicles. However, MC3D models still struggle to generalize to unseen platforms with new multi-camera configurations. Current solutions simply employ a meta-camera for unified representation but lack comprehensive consideration. In this paper, we revisit this issue and identify that the devil lies in spatial prior discrepancies across source and target configurations, including different intrinsics, extrinsics, and array layouts. To address this, we propose CoIn3D, a generalizable MC3D framework that enables strong transferability from source configurations to unseen target ones. CoIn3D explicitly incorporates all identified spatial priors into both feature embedding and image observation through spatial-aware feature modulation (SFM) and camera-aware data augmentation (CDA), respectively. SFM enriches feature space by integrating four spatial representations, such as focal length, ground depth, ground gradient, and Plücker coordinate. CDA improves observation diversity under various configurations via a training-free dynamic novel-view image synthesis scheme. Extensive experiments demonstrate that CoIn3D achieves strong cross-configuration performance on landmark datasets such as NuScenes, Waymo, and Lyft, under three dominant MC3D paradigms represented by BEVDepth, BEVFormer, and PETR.
>
---
#### [new 060] Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于强化学习中的决策规划任务，旨在解决世界模型在实时控制中计算成本过高的问题。通过提出一种将观测压缩为8个离散标记的紧凑分词器CompACT，提升规划效率。**

- **链接: [https://arxiv.org/pdf/2603.05438](https://arxiv.org/pdf/2603.05438)**

> **作者:** Dongwon Kim; Gawon Seo; Jinsung Lee; Minsu Cho; Suha Kwak
>
> **备注:** CVPR 2026
>
> **摘要:** World models provide a powerful framework for simulating environment dynamics conditioned on actions or instructions, enabling downstream tasks such as action planning or policy learning. Recent approaches leverage world models as learned simulators, but its application to decision-time planning remains computationally prohibitive for real-time control. A key bottleneck lies in latent representations: conventional tokenizers encode each observation into hundreds of tokens, making planning both slow and resource-intensive. To address this, we propose CompACT, a discrete tokenizer that compresses each observation into as few as 8 tokens, drastically reducing computational cost while preserving essential information for planning. An action-conditioned world model that occupies CompACT tokenizer achieves competitive planning performance with orders-of-magnitude faster planning, offering a practical step toward real-world deployment of world models.
>
---
#### [new 061] Dual-Interaction-Aware Cooperative Control Strategy for Alleviating Mixed Traffic Congestion
- **分类: eess.SY; cs.MA; cs.RO**

- **简介: 该论文属于交通控制任务，旨在解决混合交通中的拥堵问题。通过提出DIACC策略，增强CAV的协同控制能力，提升交通效率与适应性。**

- **链接: [https://arxiv.org/pdf/2603.03848](https://arxiv.org/pdf/2603.03848)**

> **作者:** Zhengxuan Liu; Yuxin Cai; Yijing Wang; Xiangkun He; Chen Lv; Zhiqiang Zuo
>
> **摘要:** As Intelligent Transportation System (ITS) develops, Connected and Automated Vehicles (CAVs) are expected to significantly reduce traffic congestion through cooperative strategies, such as in bottleneck areas. However, the uncertainty and diversity in the behaviors of Human-Driven Vehicles (HDVs) in mixed traffic environments present major challenges for CAV cooperation. This paper proposes a Dual-Interaction-Aware Cooperative Control (DIACC) strategy that enhances both local and global interaction perception within the Multi-Agent Reinforcement Learning (MARL) framework for Connected and Automated Vehicles (CAVs) in mixed traffic bottleneck scenarios. The DIACC strategy consists of three key innovations: 1) A Decentralized Interaction-Adaptive Decision-Making (D-IADM) module that enhances actor's local interaction perception by distinguishing CAV-CAV cooperative interactions from CAV-HDV observational interactions. 2) A Centralized Interaction-Enhanced Critic (C-IEC) that improves critic's global traffic understanding through interaction-aware value estimation, providing more accurate guidance for policy updates. 3) A reward design that employs softmin aggregation with temperature annealing to prioritize interaction-intensive scenarios in mixed traffic. Additionally, a lightweight Proactive Safety-based Action Refinement (PSAR) module applies rule-based corrections to accelerate training convergence. Experimental results demonstrate that DIACC significantly improves traffic efficiency and adaptability compared to rule-based and benchmark MARL models.
>
---
#### [new 062] Act, Think or Abstain: Complexity-Aware Adaptive Inference for Vision-Language-Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决计算复杂度高和推理效率低的问题。通过自适应框架动态选择执行策略，提升效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.05147](https://arxiv.org/pdf/2603.05147)**

> **作者:** Riccardo Andrea Izzo; Gianluca Bardaro; Matteo Matteucci
>
> **摘要:** Current research on Vision-Language-Action (VLA) models predominantly focuses on enhancing generalization through established reasoning techniques. While effective, these improvements invariably increase computational complexity and inference latency. Furthermore, these mechanisms are typically applied indiscriminately, resulting in the inefficient allocation of resources for trivial tasks while simultaneously failing to provide the uncertainty estimation necessary to prevent catastrophic failure on out-of-distribution tasks. Inspired by human cognition, we propose an adaptive framework that dynamically routes VLA execution based on the complexity of the perceived state. Our approach transforms the VLA's vision-language backbone into an active detection tool by projecting latent embeddings into an ensemble of parametric and non-parametric estimators. This allows the system to execute known tasks immediately (Act), reason about ambiguous scenarios (Think), and preemptively halt execution when encountering significant physical or semantic anomalies (Abstain). In our empirical analysis, we observe a phenomenon where visual embeddings alone are superior for inferring task complexity due to the semantic invariance of language. Evaluated on the LIBERO and LIBERO-PRO benchmarks as well as on a real robot, our vision-only configuration achieves 80% F1-Score using as little as 5% of training data, establishing itself as a reliable and efficient task complexity detector.
>
---
#### [new 063] Diffusion Policy through Conditional Proximal Policy Optimization
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习领域，解决扩散策略在在线策略训练中计算动作对数似然困难的问题。提出一种新方法，通过简单高斯概率实现高效训练，并支持熵正则化。**

- **链接: [https://arxiv.org/pdf/2603.04790](https://arxiv.org/pdf/2603.04790)**

> **作者:** Ben Liu; Shunpeng Yang; Hua Chen
>
> **摘要:** Reinforcement learning (RL) has been extensively employed in a wide range of decision-making problems, such as games and robotics. Recently, diffusion policies have shown strong potential in modeling multi-modal behaviors, enabling more diverse and flexible action generation compared to the conventional Gaussian policy. Despite various attempts to combine RL with diffusion, a key challenge is the difficulty of computing action log-likelihood under the diffusion model. This greatly hinders the direct application of diffusion policies in on-policy reinforcement learning. Most existing methods calculate or approximate the log-likelihood through the entire denoising process in the diffusion model, which can be memory- and computationally inefficient. To overcome this challenge, we propose a novel and efficient method to train a diffusion policy in an on-policy setting that requires only evaluating a simple Gaussian probability. This is achieved by aligning the policy iteration with the diffusion process, which is a distinct paradigm compared to previous work. Moreover, our formulation can naturally handle entropy regularization, which is often difficult to incorporate into diffusion policies. Experiments demonstrate that the proposed method produces multimodal policy behaviors and achieves superior performance on a variety of benchmark tasks in both IsaacLab and MuJoCo Playground.
>
---
#### [new 064] Decoupling Task and Behavior: A Two-Stage Reward Curriculum in Reinforcement Learning for Robotics
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习在机器人控制中的应用，旨在解决奖励函数设计困难的问题。通过两阶段奖励课程，分离任务目标与行为因素，提升训练效果与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.05113](https://arxiv.org/pdf/2603.05113)**

> **作者:** Kilian Freitag; Knut Åkesson; Morteza Haghir Chehreghani
>
> **摘要:** Deep Reinforcement Learning is a promising tool for robotic control, yet practical application is often hindered by the difficulty of designing effective reward functions. Real-world tasks typically require optimizing multiple objectives simultaneously, necessitating precise tuning of their weights to learn a policy with the desired characteristics. To address this, we propose a two-stage reward curriculum where we decouple task-specific objectives from behavioral terms. In our method, we first train the agent on a simplified task-only reward function to ensure effective exploration before introducing the full reward that includes auxiliary behavior-related terms such as energy efficiency. Further, we analyze various transition strategies and demonstrate that reusing samples between phases is critical for training stability. We validate our approach on the DeepMind Control Suite, ManiSkill3, and a mobile robot environment, modified to include auxiliary behavioral objectives. Our method proves to be simple yet effective, substantially outperforming baselines trained directly on the full reward while exhibiting higher robustness to specific reward weightings.
>
---
#### [new 065] Digital Twin Driven Textile Classification and Foreign Object Recognition in Automated Sorting Systems
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于纺织品分类与异物检测任务，解决自动化分拣中的变形衣物识别和复杂环境下的异物检测问题。工作包括构建数字孪生系统，融合视觉语言模型与感知技术实现高效分拣。**

- **链接: [https://arxiv.org/pdf/2603.05230](https://arxiv.org/pdf/2603.05230)**

> **作者:** Serkan Ergun; Tobias Mitterer; Hubert Zangl
>
> **备注:** 10 pages,single column, 5 figures, preprint for Photomet Edumet 2026 (Klagenfurt, Austria)
>
> **摘要:** The increasing demand for sustainable textile recycling requires robust automation solutions capable of handling deformable garments and detecting foreign objects in cluttered environments. This work presents a digital twin driven robotic sorting system that integrates grasp prediction, multi modal perception, and semantic reasoning for real world textile classification. A dual arm robotic cell equipped with RGBD sensing, capacitive tactile feedback, and collision-aware motion planning autonomously separates garments from an unsorted basket, transfers them to an inspection zone, and classifies them using state of the art Visual Language Models (VLMs). We benchmark nine VLM s from five model families on a dataset of 223 inspection scenarios comprising shirts, socks, trousers, underwear, foreign objects (including garments outside of the aforementioned classes), and empty scenes. The evaluation assesses per class accuracy, hallucination behavior, and computational performance under practical hardware constraints. Results show that the Qwen model family achieves the highest overall accuracy (up to 87.9 %), with strong foreign object detection performance, while lighter models such as Gemma3 offer competitive speed accuracy trade offs for edge deployment. A digital twin combined with MoveIt enables collision aware path planning and integrates segmented 3D point clouds of inspected garments into the virtual environment for improved manipulation reliability. The presented system demonstrates the feasibility of combining semantic VLM reasoning with conventional grasp detection and digital twin technology for scalable, autonomous textile sorting in realistic industrial settings.
>
---
#### [new 066] Person Detection and Tracking from an Overhead Crane LiDAR
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于目标检测与跟踪任务，解决工业场景下从吊车LiDAR数据中检测和跟踪人员的问题。通过构建数据集并优化检测模型，提升检测精度与实时性。**

- **链接: [https://arxiv.org/pdf/2603.04938](https://arxiv.org/pdf/2603.04938)**

> **作者:** Nilusha Jayawickrama; Henrik Toikka; Risto Ojala
>
> **备注:** 8 pages, 7 figures, 4 tables. Submitted to Ubiquitous Robots (UR) 2026. Code: this https URL
>
> **摘要:** This paper investigates person detection and tracking in an industrial indoor workspace using a LiDAR mounted on an overhead crane. The overhead viewpoint introduces a strong domain shift from common vehicle-centric LiDAR benchmarks, and limited availability of suitable public training data. Henceforth, we curate a site-specific overhead LiDAR dataset with 3D human bounding-box annotations and adapt selected candidate 3D detectors under a unified training and evaluation protocol. We further integrate lightweight tracking-by-detection using AB3DMOT and SimpleTrack to maintain person identities over time. Detection performance is reported with distance-sliced evaluation to quantify the practical operating envelope of the sensing setup. The best adapted detector configurations achieve average precision (AP) up to 0.84 within a 5.0 m horizontal radius, increasing to 0.97 at 1.0 m, with VoxelNeXt and SECOND emerging as the most reliable backbones across this range. The acquired results contribute in bridging the domain gap between standard driving datasets and overhead sensing for person detection and tracking. We also report latency measurements, highlighting practical real-time feasibility. Finally, we release our dataset and implementations in GitHub to support further research
>
---
## 更新

#### [replaced 001] Towards Exploratory and Focused Manipulation with Bimanual Active Perception: A New Problem, Benchmark and Strategy
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Exploratory and Focused Manipulation（EFM）任务，解决视觉遮挡导致的信息缺失问题。构建了EFM-10基准和BAP策略，通过双臂协作提升操作性能。**

- **链接: [https://arxiv.org/pdf/2602.01939](https://arxiv.org/pdf/2602.01939)**

> **作者:** Yuxin He; Ruihao Zhang; Tianao Shen; Cheng Liu; Qiang Nie
>
> **备注:** ICRA 2026
>
> **摘要:** Recently, active vision has reemerged as an important concept for manipulation, since visual occlusion occurs more frequently when main cameras are mounted on the robot heads. We reflect on the visual occlusion issue and identify its essence as the absence of information useful for task completion. Inspired by this, we come up with the more fundamental problem of Exploratory and Focused Manipulation (EFM). The proposed problem is about actively collecting information to complete challenging manipulation tasks that require exploration or focus. As an initial attempt to address this problem, we establish the EFM-10 benchmark that consists of 4 categories of tasks that align with our definition (10 tasks in total). We further come up with a Bimanual Active Perception (BAP) strategy, which leverages one arm to provide active vision and another arm to provide force sensing while manipulating. Based on this idea, we collect a dataset named BAPData for the tasks in EFM-10. With the dataset, we successfully verify the effectiveness of the BAP strategy in an imitation learning manner. We hope that the EFM-10 benchmark along with the BAP strategy can become a cornerstone that facilitates future research towards this direction. Project website: this http URL.
>
---
#### [replaced 002] Conflict-Based Search as a Protocol: A Multi-Agent Motion Planning Protocol for Heterogeneous Agents, Solvers, and Independent Tasks
- **分类: cs.MA; cs.RO**

- **简介: 该论文提出一种基于冲突的搜索协议，用于解决异构机器人在共享环境中的多智能体路径规划问题，通过统一接口实现不同规划算法的协同。**

- **链接: [https://arxiv.org/pdf/2510.00425](https://arxiv.org/pdf/2510.00425)**

> **作者:** Rishi Veerapaneni; Alvin Tang; Haodong He; Sophia Zhao; Viraj Shah; Yidai Cen; Ziteng Ji; Gabriel Olin; Jon Arrizabalaga; Yorai Shaoul; Jiaoyang Li; Maxim Likhachev
>
> **备注:** Published at ICRA 2026, Project webpage: this https URL
>
> **摘要:** Imagine the future construction site, hospital, or office with dozens of robots bought from different manufacturers. How can we enable these different robots to effectively move in a shared environment, given that each robot may have its own independent motion planning system? This work shows how we can get efficient collision-free movements between algorithmically heterogeneous agents by using Conflict-Based Search (Sharon et al. 2015) as a protocol. At its core, the CBS Protocol requires one specific single-agent motion planning API; finding a collision-free path that satisfies certain space-time constraints. Given such an API, CBS uses a central planner to find collision-free paths - independent of how the API is implemented. We demonstrate how this protocol enables multi-agent motion planning for a heterogeneous team of agents completing independent tasks with a variety of single-agent planners including: Heuristic Search (e.g., A*), Sampling Based Search (e.g., RRT), Optimization (e.g., Direct Collocation), Diffusion, and Reinforcement Learning.
>
---
#### [replaced 003] Distributed UAV Formation Control Robust to Relative Pose Measurement Noise
- **分类: cs.RO**

- **简介: 该论文属于无人机编队控制任务，解决相对定位噪声导致的振荡和漂移问题，通过改进梯度下降方法提升编队稳定性。**

- **链接: [https://arxiv.org/pdf/2304.03057](https://arxiv.org/pdf/2304.03057)**

> **作者:** Viktor Walter; Matouš Vrba; Daniel Bonilla Licea; Matej Hilmer; Martin Saska
>
> **备注:** Submitted to Robotics and Autonomous Systems journal on May 10. 2025 (Revision on February 27. 2026)
>
> **摘要:** A technique that allows a Formation-Enforcing Control (FEC) derived from graph rigidity theory to interface with a realistic relative localization system onboard lightweight Unmanned Aerial Vehicles (UAVs) is proposed in this paper. The proposed methodology enables reliable real-world deployment of UAVs in tight formations using relative localization systems burdened by non-negligible sensory noise. Such noise otherwise causes undesirable oscillations and drifts in sensor-based formations, and this effect is not sufficiently addressed in existing FEC algorithms. The proposed solution is based on decomposition of the gradient descent-based FEC command into interpretable elements, and then modifying these individually based on the estimated distribution of sensory noise, such that the resulting action limits the probability of overshooting the desired formation. The behavior of the system was analyzed and the practicality of the proposed solution was compared to pure gradient-descent in real-world experiments where it presented significantly better performance in terms of oscillations, deviation from the desired state
>
---
#### [replaced 004] LAP: Fast LAtent Diffusion Planner for Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶规划任务，解决扩散模型在轨迹生成中的高延迟和低效问题。提出LAP框架，在潜在空间中分离高层意图与低层运动，提升规划效率与质量。**

- **链接: [https://arxiv.org/pdf/2512.00470](https://arxiv.org/pdf/2512.00470)**

> **作者:** Jinhao Zhang; Wenlong Xia; Zhexuan Zhou; Haoming Song; Youmin Gong; Jie Mei
>
> **摘要:** Diffusion models have demonstrated strong capabilities for modeling human-like driving behaviors in autonomous driving, but their iterative sampling process induces substantial latency, and operating directly on raw trajectory points forces the model to spend capacity on low-level kinematics, rather than high-level multi-modal semantics. To address these limitations, we propose LAtent Planner (LAP), a framework that plans in a VAE-learned latent space that disentangles high-level intents from low-level kinematics, enabling our planner to capture rich, multi-modal driving strategies. To bridge the representational gap between the high-level semantic planning space and the vectorized scene context, we introduce an intermediate feature alignment mechanism that facilitates robust information fusion. Notably, LAP can produce high-quality plans in one single denoising step, substantially reducing computational overhead. Through extensive evaluations on the large-scale nuPlan benchmark, LAP achieves state-of-the-art closed-loop performance among learning-based planning methods, while demonstrating an inference speed-up of at most 10x over previous SOTA approaches.
>
---
#### [replaced 005] Responsibility and Engagement -- Evaluating Interactions in Social Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于社会机器人导航任务，旨在评估人机交互中的冲突解决。通过引入责任和参与度指标，量化代理在冲突中的贡献与行为质量。**

- **链接: [https://arxiv.org/pdf/2509.12890](https://arxiv.org/pdf/2509.12890)**

> **作者:** Malte Probst; Raphael Wenzel; Monica Dasi
>
> **备注:** Accepted at the 2026 IEEE International Conference on Robotics & Automation (ICRA)
>
> **摘要:** In Social Robot Navigation (SRN), the availability of meaningful metrics is crucial for evaluating trajectories from human-robot interactions. In the SRN context, such interactions often relate to resolving conflicts between two or more agents. Correspondingly, the shares to which agents contribute to the resolution of such conflicts are important. This paper builds on recent work, which proposed a Responsibility metric capturing such shares. We extend this framework in two directions: First, we model the conflict buildup phase by introducing a time normalization. Second, we propose the related Engagement metric, which captures how the agents' actions intensify a conflict. In a comprehensive series of simulated scenarios with dyadic, group and crowd interactions, we show that the metrics carry meaningful information about the cooperative resolution of conflicts in interactions. They can be used to assess behavior quality and foresightedness. We extensively discuss applicability, design choices and limitations of the proposed metrics.
>
---
#### [replaced 006] DDP-WM: Disentangled Dynamics Prediction for Efficient World Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DDP-WM，解决世界模型计算效率低的问题。通过分解动态预测，提升实时性能，适用于导航与操作任务。**

- **链接: [https://arxiv.org/pdf/2602.01780](https://arxiv.org/pdf/2602.01780)**

> **作者:** Shicheng Yin; Kaixuan Yin; Weixing Chen; Yang Liu; Guanbin Li; Liang Lin
>
> **备注:** Efficient and high-fidelity world model. Code is available at this https URL
>
> **摘要:** World models are essential for autonomous robotic planning. However, the substantial computational overhead of existing dense Transformerbased models significantly hinders real-time deployment. To address this efficiency-performance bottleneck, we introduce DDP-WM, a novel world model centered on the principle of Disentangled Dynamics Prediction (DDP). We hypothesize that latent state evolution in observed scenes is heterogeneous and can be decomposed into sparse primary dynamics driven by physical interactions and secondary context-driven background updates. DDP-WM realizes this decomposition through an architecture that integrates efficient historical processing with dynamic localization to isolate primary dynamics. By employing a crossattention mechanism for background updates, the framework optimizes resource allocation and provides a smooth optimization landscape for planners. Extensive experiments demonstrate that DDP-WM achieves significant efficiency and performance across diverse tasks, including navigation, precise tabletop manipulation, and complex deformable or multi-body interactions. Specifically, on the challenging Push-T task, DDP-WM achieves an approximately 9 times inference speedup and improves the MPC success rate from 90% to98% compared to state-of-the-art dense models. The results establish a promising path for developing efficient, high-fidelity world models. Codes is available at this https URL.
>
---
#### [replaced 007] In-Hand Manipulation of Articulated Tools with Dexterous Robot Hands with Sim-to-Real Transfer
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决关节物体在手中操作的挑战。针对接触动态和模型不准确问题，提出一种结合仿真与真实反馈的控制方法，提升操作鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.23075](https://arxiv.org/pdf/2509.23075)**

> **作者:** Soofiyan Atar; Daniel Huang; Florian Richter; Michael Yip
>
> **摘要:** Reinforcement learning (RL) and sim-to-real transfer have advanced rigid-object manipulation. However, policies remain brittle for articulated mechanisms due to contact-rich dynamics that require both stable grasping and simultaneous free in-hand articulation. Furthermore, articulated objects and robot hands exhibit under-modeled joint phenomena such as friction, stiction, and backlash in real life that can increase the sim-to-real gap, and robot hands still fall short of idealized tactile sensing, both in terms of coverage, sensitivity, and specificity. In this paper, we present an original approach to learning dexterous in-hand manipulation of articulated tools that has reduced articulation and kinematic redundancy relative to the human hand. Our approach augments a simulation-trained base policy with a sensor-driven refinement learned from hardware demonstrations. This refinement conditions on proprioception and target articulation states while fusing whole-hand tactile and force-torque feedback with the policy's action intent through cross-attention. The resulting controller adapts online to instance-specific articulation properties, stabilizes contact interactions, and regulates internal forces under perturbations. We validate our method across diverse real-world tools, including scissors, pliers, minimally invasive surgical instruments, and staplers, demonstrating robust sim-to-real transfer, improved disturbance resilience, and generalization across structurally related articulated tools without precise physical modeling.
>
---
#### [replaced 008] CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 该论文属于强化学习安全控制任务，旨在解决RL训练中安全约束不足的问题。通过引入CBF-RL框架，在训练中嵌入安全约束，提升策略安全性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.14959](https://arxiv.org/pdf/2510.14959)**

> **作者:** Lizhi Yang; Blake Werner; Massimiliano de Sa; Aaron D. Ames
>
> **备注:** 8 pages
>
> **摘要:** Reinforcement learning (RL), while powerful and expressive, can often prioritize performance at the expense of safety. Yet safety violations can lead to catastrophic outcomes in real-world deployments. Control Barrier Functions (CBFs) offer a principled method to enforce dynamic safety -- traditionally deployed online via safety filters. While the result is safe behavior, the fact that the RL policy does not have knowledge of the CBF can lead to conservative behaviors. This paper proposes CBF-RL, a framework for generating safe behaviors with RL by enforcing CBFs in training. CBF-RL has two key attributes: (1) minimally modifying a nominal RL policy to encode safety constraints via a CBF term, (2) and safety filtering of the policy rollouts in training. Theoretically, we prove that continuous-time safety filters can be deployed via closed-form expressions on discrete-time roll-outs. Practically, we demonstrate that CBF-RL internalizes the safety constraints in the learned policy -- both enforcing safer actions and biasing towards safer rewards -- enabling safe deployment without the need for an online safety filter. We validate our framework through ablation studies on navigation tasks and on the Unitree G1 humanoid robot, where CBF-RL enables safer exploration, faster convergence, and robust performance under uncertainty, enabling the humanoid robot to avoid obstacles and climb stairs safely in real-world settings without a runtime safety filter.
>
---
#### [replaced 009] MOSAIC: Modular Scalable Autonomy for Intelligent Coordination of Heterogeneous Robotic Teams
- **分类: cs.RO**

- **简介: 该论文提出MOSAIC框架，解决多机器人协作探索任务中的自主性与操作员干预问题，通过动态任务分配实现高效、可靠的操作。**

- **链接: [https://arxiv.org/pdf/2601.23038](https://arxiv.org/pdf/2601.23038)**

> **作者:** David Oberacker; Julia Richter; Philip Arm; Marvin Grosse Besselmann; Lennart Puck; William Talbot; Maximilian Schik; Sabine Bellmann; Tristan Schnell; Hendrik Kolvenbach; Rüdiger Dillmann; Marco Hutter; Arne Roennau
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Mobile robots have become indispensable for exploring hostile environments, such as in space or disaster relief scenarios, but often remain limited to teleoperation by a human operator. This restricts the deployment scale and requires near-continuous low-latency communication between the operator and the robot. We present MOSAIC: a scalable autonomy framework for multi-robot scientific exploration using a unified mission abstraction based on Points of Interest (POIs) and multiple layers of autonomy, enabling supervision by a single operator. The framework dynamically allocates exploration and measurement tasks based on each robot's capabilities, leveraging team-level redundancy and specialization to enable continuous operation. We validated the framework in a space-analog field experiment emulating a lunar prospecting scenario, involving a heterogeneous team of five robots and a single operator. Despite the complete failure of one robot during the mission, the team completed 82.3% of assigned tasks at an Autonomy Ratio of 86%, while the operator workload remained at only 78.2%. These results demonstrate that the proposed framework enables robust, scalable multi-robot scientific exploration with limited operator intervention. We further derive practical lessons learned in robot interoperability, networking architecture, team composition, and operator workload management to inform future multi-robot exploration missions.
>
---
#### [replaced 010] Observer-Actor: Active Vision Imitation Learning with Sparse-View Gaussian Splatting
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出ObAct框架，用于主动视觉模仿学习，解决机器人视角受限问题。通过动态调整观察者与执行者角色，提升视觉清晰度，增强策略鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.18140](https://arxiv.org/pdf/2511.18140)**

> **作者:** Yilong Wang; Cheng Qian; Ruomeng Fan; Edward Johns
>
> **备注:** Accepted at ICRA 2026. Project Webpage: this https URL
>
> **摘要:** We propose Observer Actor (ObAct), a novel framework for active vision imitation learning in which the observer moves to optimal visual observations for the actor. We study ObAct on a dual-arm robotic system equipped with wrist-mounted cameras. At test time, ObAct dynamically assigns observer and actor roles: the observer arm constructs a 3D Gaussian Splatting (3DGS) representation from three images, virtually explores this to find an optimal camera pose, then moves to this pose; the actor arm then executes a policy using the observer's observations. This formulation enhances the clarity and visibility of both the object and the gripper in the policy's observations. As a result, we enable the training of ambidextrous policies on observations that remain closer to the occlusion-free training distribution, leading to more robust policies. We study this formulation with two existing imitation learning methods -- trajectory transfer and behavior cloning -- and experiments show that ObAct significantly outperforms static-camera setups: trajectory transfer improves by 145% without occlusion and 233% with occlusion, while behavior cloning improves by 75% and 143%, respectively. Videos are available at this https URL.
>
---
#### [replaced 011] ROVER: Regulator-Driven Robust Temporal Verification of Black-Box Robot Policies
- **分类: cs.RO**

- **简介: 该论文属于机器人安全验证任务，解决黑盒策略的时序安全性问题。通过引入监管者机制，使用STL规范评估策略，提升安全性和稳定性。**

- **链接: [https://arxiv.org/pdf/2511.17781](https://arxiv.org/pdf/2511.17781)**

> **作者:** Kristy Sakano; Jianyu An; Dinesh Manocha; Huan Xu
>
> **摘要:** We present a novel, regulator-driven approach for the temporal verification of black-box autonomous robot policies, inspired by real-world certification processes where regulators often evaluate observable behavior without access to model internals. Central to our method is a regulator-in-the-loop approach that evaluates execution traces from black-box policies against temporal safety requirements. These requirements, expressed as prioritized Signal Temporal Logic (STL) specifications, characterize behavior changes over time and encode domain knowledge into the verification process. We use Total Robustness Value (TRV) and Largest Robustness Value (LRV) to quantify average performance and worst-case adherence, and introduce Average Violation Robustness Value (AVRV) to measure average specification violation. Together, these metrics guide targeted retraining and iterative model improvement. Our approach accommodates diverse temporal safety requirements (e.g., lane-keeping, delayed acceleration, and turn smoothness), capturing persistence, sequencing, and response across two distinct domains (virtual racing game and mobile robot navigation). Across six STL specifications in both scenarios, regulator-guided retraining increased satisfaction rates by an average of 43.8%, with consistent improvement in average performance (TRV) and reduced violation severity (LRV) in half of the specifications. Finally, real-world validation on a TurtleBot3 robot demonstrates a 27% improvement in smooth-navigation satisfaction, yielding smoother paths and stronger compliance with STL-defined temporal safety requirements.
>
---
#### [replaced 012] TEMPO-VINE: A Multi-Temporal Sensor Fusion Dataset for Localization and Mapping in Vineyards
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出TEMPO-VINE数据集，用于农业环境下的定位与建图研究，解决多传感器融合及场景识别问题。**

- **链接: [https://arxiv.org/pdf/2512.04772](https://arxiv.org/pdf/2512.04772)**

> **作者:** Mauro Martini; Marco Ambrosio; Judith Vilella-Cantos; Alessandro Navone; Marcello Chiaberge
>
> **摘要:** In recent years, precision agriculture has been introducing groundbreaking innovations in the field, with a strong focus on automation. However, research studies in robotics and autonomous navigation often rely on controlled simulations or isolated field trials. The absence of a realistic common benchmark represents a significant limitation for the diffusion of robust autonomous systems under real complex agricultural conditions. Vineyards pose significant challenges due to their dynamic nature, and they are increasingly drawing attention from both academic and industrial stakeholders interested in automation. In this context, we introduce the TEMPO-VINE dataset, a large-scale multi-temporal dataset specifically designed for evaluating sensor fusion, simultaneous localization and mapping (SLAM), and place recognition techniques within operational vineyard environments. TEMPO-VINE is the first multi-modal public dataset that brings together data from heterogeneous LiDARs of different price levels, AHRS, RTK-GPS, and cameras in real trellis and pergola vineyards, with multiple rows exceeding 100 m in length. In this work, we address a critical gap in the landscape of agricultural datasets by providing researchers with a comprehensive data collection and ground truth trajectories in different seasons, vegetation growth stages, terrain and weather conditions. The sequence paths with multiple runs and revisits will foster the development of sensor fusion, localization, mapping and place recognition solutions for agricultural fields. The dataset, the processing tools and the benchmarking results are available on the webpage.
>
---
#### [replaced 013] 3D Dynamics-Aware Manipulation: Endowing Manipulation Policies with 3D Foresight
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决深度运动下2D建模不足的问题。通过引入3D动态建模与策略学习，提升操作策略的3D预见能力。**

- **链接: [https://arxiv.org/pdf/2502.10028](https://arxiv.org/pdf/2502.10028)**

> **作者:** Yuxin He; Ruihao Zhang; Xianzu Wu; Zhiyuan Zhang; Cheng Ding; Qiang Nie
>
> **备注:** ICRA 2026
>
> **摘要:** The incorporation of world modeling into manipulation policy learning has pushed the boundary of manipulation performance. However, existing efforts simply model the 2D visual dynamics, which is insufficient for robust manipulation when target tasks involve prominent depth-wise movement. To address this, we present a 3D dynamics-aware manipulation framework that seamlessly integrates 3D world modeling and policy learning. Three self-supervised learning tasks (current depth estimation, future RGB-D prediction, 3D flow prediction) are introduced within our framework, which complement each other and endow the policy model with 3D foresight. Extensive experiments on simulation and the real world show that 3D foresight can greatly boost the performance of manipulation policies without sacrificing inference speed. Code is available at this https URL.
>
---
#### [replaced 014] Seeing the Bigger Picture: 3D Latent Mapping for Mobile Manipulation Policy Learning
- **分类: cs.RO**

- **简介: 该论文研究移动操作策略学习任务，旨在解决传统图像依赖策略空间和时间推理不足的问题。提出SBP方法，利用3D潜在地图提升全局推理与长时记忆能力。**

- **链接: [https://arxiv.org/pdf/2510.03885](https://arxiv.org/pdf/2510.03885)**

> **作者:** Sunghwan Kim; Woojeh Chung; Zhirui Dai; Dwait Bhatt; Arth Shukla; Hao Su; Yulun Tian; Nikolay Atanasov
>
> **备注:** ICRA 2026, project page: this https URL
>
> **摘要:** In this paper, we demonstrate that mobile manipulation policies utilizing a 3D latent map achieve stronger spatial and temporal reasoning than policies relying solely on images. We introduce Seeing the Bigger Picture (SBP), an end-to-end policy learning approach that operates directly on a 3D map of latent features. In SBP, the map extends perception beyond the robot's current field of view and aggregates observations over long horizons. Our mapping approach incrementally fuses multiview observations into a grid of scene-specific latent features. A pre-trained, scene-agnostic decoder reconstructs target embeddings from these features and enables online optimization of the map features during task execution. A policy, trainable with behavior cloning or reinforcement learning, treats the latent map as a state variable and uses global context from the map obtained via a 3D feature aggregator. We evaluate SBP on scene-level mobile manipulation and sequential tabletop manipulation tasks. Our experiments demonstrate that SBP (i) reasons globally over the scene, (ii) leverages the map as long-horizon memory, and (iii) outperforms image-based policies in both in-distribution and novel scenes, e.g., improving the success rate by 15% for the sequential manipulation task.
>
---
#### [replaced 015] GUIDE: A Diffusion-Based Autonomous Robot Exploration Framework Using Global Graph Inference
- **分类: cs.RO**

- **简介: 论文提出GUIDE框架，解决室内环境自主探索问题。通过结合全局图推理与扩散决策，提升路径规划效率，减少冗余移动。**

- **链接: [https://arxiv.org/pdf/2509.19916](https://arxiv.org/pdf/2509.19916)**

> **作者:** Zijun Che; Yinghong Zhang; Shengyi Liang; Boyu Zhou; Jun Ma; Jinni Zhou
>
> **摘要:** Autonomous exploration in structured and complex indoor environments remains a challenging task, as existing methods often struggle to appropriately model unobserved space and plan globally efficient paths. To address these limitations, we propose GUIDE, a novel exploration framework that synergistically combines global graph inference with diffusion-based decision-making. We introduce a region-evaluation global graph representation that integrates both observed environmental data and predictions of unexplored areas, enhanced by a region-level evaluation mechanism to prioritize reliable structural inferences while discounting uncertain predictions. Building upon this enriched representation, a diffusion policy network generates stable, foresighted action sequences with significantly reduced denoising steps. Extensive simulations and real-world deployments demonstrate that GUIDE consistently outperforms state-of-the-art methods, achieving up to 18.3% faster coverage completion and a 34.9% reduction in redundant movements.
>
---
#### [replaced 016] EmboTeam: Grounding LLM Reasoning into Reactive Behavior Trees via PDDL for Embodied Multi-Robot Collaboration
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文提出EmboTeam框架，解决多机器人协作中的长周期任务规划问题。通过LLM与经典规划器结合，生成行为树实现动态协调。**

- **链接: [https://arxiv.org/pdf/2601.11063](https://arxiv.org/pdf/2601.11063)**

> **作者:** Haishan Zeng; Mengna Wang; Peng Li
>
> **摘要:** In embodied artificial intelligence, enabling heterogeneous robot teams to execute long-horizon tasks from high-level instructions remains a critical challenge. While large language models (LLMs) show promise in instruction parsing and preliminary planning, they exhibit limitations in long-term reasoning and dynamic multi-robot coordination. We propose EmboTeam, a novel embodied multi-robot task planning framework that addresses these issues through a three-stage cascaded architecture: 1) It leverages an LLM to parse instructions and generate Planning Domain Definition Language (PDDL) problem descriptions, thereby transforming commands into formal planning problems; 2) It combines the semantic reasoning of LLMs with the search capabilities of a classical planner to produce optimized action sequences; 3) It compiles the resulting plan into behavior trees for reactive control. The framework supports dynamically sized heterogeneous robot teams via a shared blackboard mechanism for communication and state synchronization. To validate our approach, we introduce the MACE-THOR benchmark dataset, comprising 42 complex tasks across 8 distinct household layouts. Experiments show EmboTeam improves the task success rate from 12% to 55% and goal condition recall from 32% to 72% over the LaMMA-P baseline.
>
---
#### [replaced 017] RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RoboPARA框架，解决双臂机器人任务并行规划问题，通过两阶段方法提升任务效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2506.06683](https://arxiv.org/pdf/2506.06683)**

> **作者:** Shiying Duan; Pei Ren; Nanxiang Jiang; Zhengping Che; Jian Tang; Zhaoxin Fan; Yifan Sun; Wenjun Wu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Dual-arm robots play a crucial role in improving efficiency and flexibility in complex multitasking this http URL existing methods have achieved promising results in task planning, they often fail to fully optimize task parallelism, limiting the potential of dual-arm this http URL address this issue, we propose RoboPARA, a novel large language model (LLM)-driven framework for dual-arm task parallelism this http URL employs a two-stage process: (1) Dependency Graph-based Planning Candidates Generation, which constructs directed acyclic graphs (DAGs) to model task dependencies and eliminate redundancy, and (2) Graph Re-Traversal-based Dual-Arm Parallel Planning, which optimizes DAG traversal to maximize parallelism while maintaining task this http URL addition, we introduce the Cross-Scenario Dual-Arm Parallel Task dataset (X-DAPT dataset), the first dataset specifically designed to evaluate dual-arm task parallelism across diverse scenarios and difficulty this http URL experiments demonstrate that RoboPARA significantly outperforms existing planning methods, achieving higher efficiency and reliability, particularly in complex task this http URL code is publicly available at this https URL.
>
---
#### [replaced 018] Balancing Progress and Safety: A Novel Risk-Aware Objective for RL in Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于强化学习在自动驾驶中的应用任务，旨在解决奖励函数设计不合理导致的安全问题。通过构建层次化目标和引入风险感知机制，提升驾驶安全性与性能。**

- **链接: [https://arxiv.org/pdf/2505.06737](https://arxiv.org/pdf/2505.06737)**

> **作者:** Ahmed Abouelazm; Jonas Michel; Helen Gremmelmaier; Tim Joseph; Philip Schörner; J. Marius Zöllner
>
> **备注:** Accepted in the 36th IEEE Intelligent vehicles Symposium (IV 2025)
>
> **摘要:** Reinforcement Learning (RL) is a promising approach for achieving autonomous driving due to robust decision-making capabilities. RL learns a driving policy through trial and error in traffic scenarios, guided by a reward function that combines the driving objectives. The design of such reward function has received insufficient attention, yielding ill-defined rewards with various pitfalls. Safety, in particular, has long been regarded only as a penalty for collisions. This leaves the risks associated with actions leading up to a collision unaddressed, limiting the applicability of RL in real-world scenarios. To address these shortcomings, our work focuses on enhancing the reward formulation by defining a set of driving objectives and structuring them hierarchically. Furthermore, we discuss the formulation of these objectives in a normalized manner to transparently determine their contribution to the overall reward. Additionally, we introduce a novel risk-aware objective for various driving interactions based on a two-dimensional ellipsoid function and an extension of Responsibility-Sensitive Safety (RSS) concepts. We evaluate the efficacy of our proposed reward in unsignalized intersection scenarios with varying traffic densities. The approach decreases collision rates by 21\% on average compared to baseline rewards and consistently surpasses them in route progress and cumulative reward, demonstrating its capability to promote safer driving behaviors while maintaining high-performance levels.
>
---
#### [replaced 019] Risk-Aware Autonomous Driving with Linear Temporal Logic Specifications
- **分类: eess.SY; cs.FL; cs.RO**

- **简介: 该论文属于自主驾驶任务，旨在解决如何让自动驾驶系统像人类一样平衡多种驾驶风险。通过扩展风险度量并结合线性时序逻辑（LTL），实现更安全的驾驶决策。**

- **链接: [https://arxiv.org/pdf/2409.09769](https://arxiv.org/pdf/2409.09769)**

> **作者:** Shuhao Qi; Zengjie Zhang; Zhiyong Sun; Sofie Haesaert
>
> **摘要:** Human drivers naturally balance the risks of different concerns while driving, including traffic rule violations, minor accidents, and fatalities. However, achieving the same behavior in autonomous driving systems remains an open problem. This paper extends a risk metric that has been verified in human-like driving studies to encompass more complex driving scenarios specified by linear temporal logic (LTL) that go beyond just collision risks. This extension incorporates the timing and severity of events into LTL specifications, thereby reflecting a human-like risk awareness. Without sacrificing expressivity for traffic rules, we adopt LTL specifications composed of safety and co-safety formulas, allowing the control synthesis problem to be reformulated as a reachability problem. By leveraging occupation measures, we further formulate a linear programming (LP) problem for this LTL-based risk metric. Consequently, the synthesized policy balances different types of driving risks, including both collision risks and traffic rule violations. The effectiveness of the proposed approach is validated by three typical traffic scenarios in Carla simulator.
>
---
#### [replaced 020] PeRoI: A Pedestrian-Robot Interaction Dataset for Learning Avoidance, Neutrality, and Attraction Behaviors in Social Navigation
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于社会导航任务，旨在解决机器人与行人交互行为建模问题。提出PeRoI数据集和NeuRoSFM模型，提升对避让、中立和吸引行为的预测精度。**

- **链接: [https://arxiv.org/pdf/2503.16481](https://arxiv.org/pdf/2503.16481)**

> **作者:** Subham Agrawal; Nico Ostermann-Myrau; Nils Dengler; Maren Bennewitz
>
> **摘要:** Robots are increasingly being deployed in public spaces such as shopping malls, sidewalks, and hospitals, where safe and socially aware navigation depends on anticipating how pedestrians respond to their presence. However, existing datasets rarely capture the full spectrum of robot-induced reactions, e.g., avoidance, neutrality, attraction, which limits progress in modeling these interactions. In this paper, we present the Pedestrian-Robot Interaction~(PeRoI) dataset that captures pedestrian motions categorized into attraction, neutrality, and repulsion across two outdoor sites under three controlled conditions: no robot present, with stationary robot, and with moving robot. This design explicitly reveals how pedestrian behavior varies across robot contexts, and we provide qualitative and quantitative comparisons to established state-of-the-art datasets. Building on these data, we propose the Neural Robot Social Force Model~(NeuRoSFM), an extension of the Social Force Model that integrates neural networks to augment inter-human dynamics with learned components and explicit robot-induced forces to better predict pedestrian motion in vicinity of robots. We evaluate NeuRoSFM by generating trajectories on multiple real-world datasets. The results demonstrate improved modeling of pedestrian-robot interactions, leading to better prediction accuracy, and highlight the value of our dataset and method for advancing socially aware navigation strategies in human-centered environments.
>
---
#### [replaced 021] Efficient Path Generation with Curvature Guarantees by Mollification
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于路径生成任务，解决非光滑路径与控制算法需求之间的差异问题。通过 mollification 方法生成平滑且具有曲率约束的路径，提高计算效率。**

- **链接: [https://arxiv.org/pdf/2512.13183](https://arxiv.org/pdf/2512.13183)**

> **作者:** Alfredo González-Calvin; Juan F.Jiménez; Héctor García de Marina
>
> **摘要:** Path generation, the process of converting high-level mission specifications, such as sequences of waypoints from a path planner, into smooth, executable paths, is a fundamental challenge in mobile robotics. Most path following and trajectory tracking algorithms require the desired path to be defined by at least twice continuously differentiable functions to guarantee key properties such as global convergence, especially for nonholonomic robots like unicycles with speed constraints. Consequently, path generation methods must bridge the gap between convenient but non-differentiable planning outputs, such as piecewise linear segments, and the differentiability requirements imposed by downstream control algorithms. While techniques such as spline interpolation or optimization-based methods are commonly used to smooth non-differentiable paths or create feasible ones from sequences of waypoints, they either produce unnecessarily complex trajectories or are computationally expensive. In this work, we present a method to regularize non-differentiable functions and generate feasible paths through mollification. Specifically, we approximate an arbitrary path with a differentiable function that can converge to it with arbitrary precision. Additionally, we provide a systematic method for bounding the curvature of generated paths, which we demonstrate by applying it to paths resulting from linking a sequence of waypoints with segments. The proposed approach is analytically shown to be computationally more efficient than standard interpolation methods, enabling real-time implementation on microcontrollers, while remaining compatible with standard trajectory tracking and path following algorithms.
>
---
#### [replaced 022] FreeTacMan: Robot-free Visuo-Tactile Data Collection System for Contact-rich Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决接触丰富操作中的数据收集问题。提出FreeTacMan系统，实现高效、准确的视觉触觉数据采集。**

- **链接: [https://arxiv.org/pdf/2506.01941](https://arxiv.org/pdf/2506.01941)**

> **作者:** Longyan Wu; Checheng Yu; Jieji Ren; Li Chen; Yufei Jiang; Ran Huang; Guoying Gu; Hongyang Li
>
> **摘要:** Enabling robots with contact-rich manipulation remains a pivotal challenge in robot learning, which is substantially hindered by the data collection gap, including its inefficiency and limited sensor setup. While prior work has explored handheld paradigms, their rod-based mechanical structures remain rigid and unintuitive, providing limited tactile feedback and posing challenges for operators. Motivated by the dexterity and force feedback of human motion, we propose FreeTacMan, a human-centric and robot-free data collection system for accurate and efficient robot manipulation. Concretely, we design a wearable gripper with visuo-tactile sensors for data collection, which can be worn by human fingers for intuitive control. A high-precision optical tracking system is introduced to capture end-effector poses while synchronizing visual and tactile feedback simultaneously. We leverage FreeTacMan to collect a large-scale multimodal dataset, comprising over 3000k paired visuo-tactile images with end-effector poses, 10k demonstration trajectories across 50 diverse contact-rich manipulation tasks. FreeTacMan achieves multiple improvements in data collection performance over prior works and enables effective policy learning from self-collected datasets. By open-sourcing the hardware and the dataset, we aim to facilitate reproducibility and support research in visuo-tactile manipulation.
>
---
#### [replaced 023] GRAND: Guidance, Rebalancing, and Assignment for Networked Dispatch in Multi-Agent Path Finding
- **分类: cs.RO; cs.LG; cs.MA**

- **简介: 该论文属于多智能体路径规划任务，解决大规模仓储中任务调度问题。提出GRAND方法，结合学习引导与优化，提升调度效率与实时性。**

- **链接: [https://arxiv.org/pdf/2512.03194](https://arxiv.org/pdf/2512.03194)**

> **作者:** Johannes Gaber; Meshal Alharbi; Daniele Gammelli; Gioele Zardini
>
> **摘要:** Large robot fleets are now common in warehouses and other logistics settings, where small control gains translate into large operational impacts. In this article, we address task scheduling for lifelong Multi-Agent Pickup-and-Delivery (MAPD) and propose a hybrid method that couples learning-based global guidance with lightweight optimization. A graph neural network policy trained via reinforcement learning outputs a desired distribution of free agents over an aggregated warehouse graph. This signal is converted into region-to-region rebalancing through a minimum-cost flow, and finalized by small, local assignment problems, preserving accuracy while keeping per-step latency within a 1 s compute budget. We call this approach GRAND: a hierarchical algorithm that relies on Guidance, Rebalancing, and Assignment to explicitly leverage the workspace Network structure and Dispatch agents to tasks. On congested warehouse benchmarks from the League of Robot Runners (LoRR) with up to 500 agents, our approach improves throughput by up to 10% over the 2024 winning scheduler while maintaining real-time execution. The results indicate that coupling graph-structured learned guidance with tractable solvers reduces congestion and yields a practical, scalable blueprint for high-throughput scheduling in large fleets.
>
---
#### [replaced 024] Collaborative Learning of Local 3D Occupancy Prediction and Versatile Global Occupancy Mapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D语义占用预测任务，旨在提升自动驾驶中的环境感知。通过融合全局先验信息，增强局部预测的鲁棒性，并构建大规模全局占用地图。**

- **链接: [https://arxiv.org/pdf/2504.13596](https://arxiv.org/pdf/2504.13596)**

> **作者:** Shanshuai Yuan; Julong Wei; Muer Tie; Xiangyun Ren; Zhongxue Gan; Wenchao Ding
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Vision-based 3D semantic occupancy prediction is vital for autonomous driving, enabling unified modeling of static infrastructure and dynamic agents. Global occupancy maps serve as long-term memory priors, providing valuable historical context that enhances local perception. This is particularly important in challenging scenarios such as occlusion or poor illumination, where current and nearby observations may be unreliable or incomplete. Priors aggregated from previous traversals under better conditions help fill gaps and enhance the robustness of local 3D occupancy prediction. In this paper, we propose Long-term Memory Prior Occupancy (LMPOcc), a plug-and-play framework that incorporates global occupancy priors to boost local prediction and simultaneously updates global maps with new observations. To realize the information gain from global priors, we design an efficient and lightweight Current-Prior Fusion module that adaptively integrates prior and current features. Meanwhile, we introduce a model-agnostic prior format to enable continual updating of global occupancy and ensure compatibility across diverse prediction baselines. LMPOcc achieves state-of-the-art local occupancy prediction performance validated on the Occ3D-nuScenes benchmark, especially on static semantic categories. Furthermore, we verify LMPOcc's capability to build large-scale global occupancy maps through multi-vehicle crowdsourcing, and utilize occupancy-derived dense depth to support the construction of 3D open-vocabulary maps. Our method opens up a new paradigm for continuous global information updating and storage, paving the way towards more comprehensive and scalable scene understanding in large outdoor environments.
>
---
#### [replaced 025] SpikeATac: A Multimodal Tactile Finger with Taxelized Dynamic Sensing for Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文提出SpikeATac触觉手指，用于灵巧操作任务，解决脆弱物体抓取与操控问题。结合动态与静态传感，提升触觉感知精度与响应速度。**

- **链接: [https://arxiv.org/pdf/2510.27048](https://arxiv.org/pdf/2510.27048)**

> **作者:** Eric T. Chang; Peter Ballentine; Zhanpeng He; Do-Gon Kim; Kai Jiang; Hua-Hsuan Liang; Joaquin Palacios; William Wang; Pedro Piacenza; Ioannis Kymissis; Matei Ciocarlie
>
> **备注:** 8 pages, 8 figures, ICRA 2026
>
> **摘要:** In this work, we introduce SpikeATac, a multimodal tactile finger combining a taxelized and highly sensitive dynamic response (PVDF) with a static transduction method (capacitive) for multimodal touch sensing. Named for its `spiky' response, SpikeATac's 16-taxel PVDF film sampled at 4 kHz provides fast, sensitive dynamic signals to the very onset and breaking of contact. We characterize the sensitivity of the different modalities, and show that SpikeATac provides the ability to stop quickly and delicately when grasping fragile, deformable objects. Beyond parallel grasping, we show that SpikeATac can be used in a learning-based framework to achieve new capabilities on a dexterous multifingered robot hand. We use a learning recipe that combines reinforcement learning from human feedback with tactile-based rewards to fine-tune the behavior of a policy to modulate force. Our hardware platform and learning pipeline together enable a difficult dexterous and contact-rich task that has not previously been achieved: in-hand manipulation of fragile objects. Videos are available at this https URL .
>
---
#### [replaced 026] Distant Object Localisation from Noisy Image Segmentation Sequences
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于无人机火灾监测任务，解决远距离目标定位问题。通过多视角三角测量和粒子滤波方法，在计算资源有限情况下实现可靠定位与不确定性估计。**

- **链接: [https://arxiv.org/pdf/2509.20906](https://arxiv.org/pdf/2509.20906)**

> **作者:** Julius Pesonen; Arno Solin; Eija Honkavaara
>
> **摘要:** 3D object localisation based on a sequence of camera measurements is essential for safety-critical surveillance tasks, such as drone-based wildfire monitoring. Localisation of objects detected with a camera can typically be solved with specialised sensor configurations or 3D scene reconstruction. However, in the context of distant objects or tasks limited by the amount of available computational resources, neither solution is feasible. In this paper, we show that the task can be solved with either multi-view triangulation or particle filters, with the latter also providing shape and uncertainty estimates. We studied the solutions using 3D simulation and drone-based image segmentation sequences with global navigation satellite system (GNSS) based camera pose estimates. The results suggest that combining the proposed methods with pre-existing image segmentation models and drone-carried computational resources yields a reliable system for drone-based wildfire monitoring. The proposed solutions are independent of the detection method, also enabling quick adaptation to similar tasks.
>
---
#### [replaced 027] LHM-Humanoid: Learning a Unified Policy for Long-Horizon Humanoid Whole-Body Loco-Manipulation in Diverse Messy Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出LHM-Humanoid任务，解决复杂环境中人形机器人长时序全身操作问题。通过统一策略学习，实现跨场景导航与物体操作，提升鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2508.16943](https://arxiv.org/pdf/2508.16943)**

> **作者:** Haozhuo Zhang; Jingkai Sun; Michele Caprio; Jian Tang; Shanghang Zhang; Qiang Zhang; Wei Pan
>
> **摘要:** We introduce LHM-Humanoid, a benchmark and learning framework for long-horizon whole-body humanoid loco-manipulation in diverse, cluttered scenes. In our setting, multiple objects are displaced from their intended locations and may obstruct navigation; a humanoid agent must repeatedly (i) walk to a target, (ii) pick it up with diverse whole-body postures under balance constraints, (iii) carry it while navigating around obstacles, and (iv) place it at a designated goal -- all within a single continuous episode and without any environment reset. This task simultaneously demands cross-scene generalization and unified one-policy control: layouts, obstacle arrangements, object category/mass/shape/color and object start/goal poses vary substantially even within a room category, requiring a single general policy that directly outputs actions rather than invoking pre-trained skill libraries. Our dataset spans four room types (bedroom, living room, kitchen, and warehouse), comprising 350 diverse scenes/tasks with 79 objects (25 movable targets). Since no scene-specific ground-truth motion sequences are provided, we learn goal-conditioned teacher policies via reinforcement learning and distill them into a single end-to-end student policy using DAgger. We further distill this unified policy into a vision-language-action (VLA) model driven by egocentric RGB observations and natural language. Experiments in Isaac Gym demonstrate that LHM-Humanoid substantially outperforms end-to-end RL baselines and prior humanoid loco-manipulation methods on both seen and unseen scenes, exhibiting strong long-horizon robustness and cross-scene generalization.
>
---
#### [replaced 028] Environment-Aware Learning of Smooth GNSS Covariance Dynamics for Autonomous Racing
- **分类: cs.RO**

- **简介: 该论文属于自主驾驶中的状态估计任务，旨在解决GNSS测量不确定性动态建模问题。通过学习框架LACE，实现环境感知的平滑协方差动态建模，提升定位精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.21366](https://arxiv.org/pdf/2602.21366)**

> **作者:** Y. Deemo Chen; Arion Zimmermann; Thomas A. Berrueta; Soon-Jo Chung
>
> **备注:** 8 pages, Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Ensuring accurate and stable state estimation is a challenging task crucial to safety-critical domains such as high-speed autonomous racing, where measurement uncertainty must be both adaptive to the environment and temporally smooth for control. In this work, we develop a learning-based framework, LACE, capable of directly modeling the temporal dynamics of GNSS measurement covariance. We model the covariance evolution as an exponentially stable dynamical system where a deep neural network (DNN) learns to predict the system's process noise from environmental features through an attention mechanism. By using contraction-based stability and systematically imposing spectral constraints, we formally provide guarantees of exponential stability and smoothness for the resulting covariance dynamics. We validate our approach on an AV-24 autonomous racecar, demonstrating improved localization performance and smoother covariance estimates in challenging, GNSS-degraded environments. Our results highlight the promise of dynamically modeling the perceived uncertainty in state estimation problems that are tightly coupled with control sensitivity.
>
---
#### [replaced 029] Learning Physical Systems: Symplectification via Gauge Fixing in Dirac Structures
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于物理信息深度学习任务，旨在解决耗散与约束系统中对称性退化问题。提出PSN框架，通过Dirac结构恢复非退化解析几何，实现稳定长期预测。**

- **链接: [https://arxiv.org/pdf/2506.18812](https://arxiv.org/pdf/2506.18812)**

> **作者:** Aristotelis Papatheodorou; Pranav Vaidhyanathan; Natalia Ares; Ioannis Havoutis
>
> **备注:** Presented at Equivariant Systems: Theory and Applications in State Estimation, Artificial Intelligence and Control, Robotics: Science and Systems (RSS) 2025 Workshop, 6 Pages, 3 Figures
>
> **摘要:** Physics-informed deep learning has achieved remarkable progress by embedding geometric priors, such as Hamiltonian symmetries and variational principles, into neural networks, enabling structure-preserving models that extrapolate with high accuracy. However, in systems with dissipation and holonomic constraints, ubiquitous in legged locomotion and multibody robotics, the canonical symplectic form becomes degenerate, undermining the very invariants that guarantee stability and long-term prediction. In this work, we tackle this foundational limitation by introducing Presymplectification Networks (PSNs), the first framework to learn the symplectification lift via Dirac structures, restoring a non-degenerate symplectic geometry by embedding constrained systems into a higher-dimensional manifold. Our architecture combines a recurrent encoder with a flow-matching objective to learn the augmented phase-space dynamics end-to-end. We then attach a lightweight Symplectic Network (SympNet) to forecast constrained trajectories while preserving energy, momentum, and constraint satisfaction. We demonstrate our method on the dynamics of the ANYmal quadruped robot, a challenging contact-rich, multibody system. To the best of our knowledge, this is the first framework that effectively bridges the gap between constrained, dissipative mechanical systems and symplectic learning, unlocking a whole new class of geometric machine learning models, grounded in first principles yet adaptable from data.
>
---
#### [replaced 030] Whole-Body Safe Control of Robotic Systems with Koopman Neural Dynamics
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决非线性系统实时安全控制问题。通过学习Koopman模型并结合安全集算法，实现高效、安全的轨迹跟踪与避障。**

- **链接: [https://arxiv.org/pdf/2603.03740](https://arxiv.org/pdf/2603.03740)**

> **作者:** Sebin Jung; Abulikemu Abuduweili; Jiaxing Li; Changliu Liu
>
> **摘要:** Controlling robots with strongly nonlinear, high-dimensional dynamics remains challenging, as direct nonlinear optimization with safety constraints is often intractable in real time. The Koopman operator offers a way to represent nonlinear systems linearly in a lifted space, enabling the use of efficient linear control. We propose a data-driven framework that learns a Koopman embedding and operator from data, and integrates the resulting linear model with the Safe Set Algorithm (SSA). This allows the tracking and safety constraints to be solved in a single quadratic program (QP), ensuring feasibility and optimality without a separate safety filter. We validate the method on a Kinova Gen3 manipulator and a Go2 quadruped, showing accurate tracking and obstacle avoidance.
>
---
#### [replaced 031] Automatic Curriculum Learning for Driving Scenarios: Towards Robust and Efficient Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，旨在解决RL训练泛化与效率问题。提出自动课程学习框架，动态生成适应智能体能力的驾驶场景，提升训练效果与收敛速度。**

- **链接: [https://arxiv.org/pdf/2505.08264](https://arxiv.org/pdf/2505.08264)**

> **作者:** Ahmed Abouelazm; Tim Weinstein; Tim Joseph; Philip Schörner; J. Marius Zöllner
>
> **备注:** Accepted in the 36th IEEE Intelligent Vehicles Symposium (IV 2025)
>
> **摘要:** This paper addresses the challenges of training end-to-end autonomous driving agents using Reinforcement Learning (RL). RL agents are typically trained in a fixed set of scenarios and nominal behavior of surrounding road users in simulations, limiting their generalization and real-life deployment. While domain randomization offers a potential solution by randomly sampling driving scenarios, it frequently results in inefficient training and sub-optimal policies due to the high variance among training scenarios. To address these limitations, we propose an automatic curriculum learning framework that dynamically generates driving scenarios with adaptive complexity based on the agent's evolving capabilities. Unlike manually designed curricula that introduce expert bias and lack scalability, our framework incorporates a ``teacher'' that automatically generates and mutates driving scenarios based on their learning potential -- an agent-centric metric derived from the agent's current policy -- eliminating the need for expert design. The framework enhances training efficiency by excluding scenarios the agent has mastered or finds too challenging. We evaluate our framework in a reinforcement learning setting where the agent learns a driving policy from camera images. Comparative results against baseline methods, including fixed scenario training and domain randomization, demonstrate that our approach leads to enhanced generalization, achieving higher success rates: +9% in low traffic density, +21% in high traffic density, and faster convergence with fewer training steps. Our findings highlight the potential of ACL in improving the robustness and efficiency of RL-based autonomous driving agents.
>
---
#### [replaced 032] Scout-Rover cooperation: online terrain strength mapping and traversal risk estimation for planetary-analog explorations
- **分类: cs.RO**

- **简介: 该论文属于行星探测任务，旨在解决松散地形下的安全导航问题。通过 scout-rover 协作，实现地形强度在线映射和路径风险评估，提升探测器在复杂地形中的作业能力。**

- **链接: [https://arxiv.org/pdf/2602.18688](https://arxiv.org/pdf/2602.18688)**

> **作者:** Shipeng Liu; J. Diego Caporale; Yifeng Zhang; Xingjue Liao; William Hoganson; Wilson Hu; Shivangi Misra; Neha Peddinti; Rachel Holladay; Ethan Fulcher; Akshay Ram Panyam; Andrik Puentes; Jordan M. Bretzfelder; Michael Zanetti; Uland Wong; Daniel E. Koditschek; Mark Yim; Douglas Jerolmack; Cynthia Sung; Feifei Qian
>
> **备注:** 8 figures
>
> **摘要:** Robot-aided exploration of planetary surfaces is essential for understanding geologic processes, yet many scientifically valuable regions, such as Martian dunes and lunar craters, remain hazardous due to loose, deformable regolith. We present a scout-rover cooperation framework that expands safe access to such terrain using a hybrid team of legged and wheeled robots. In our approach, a high-mobility legged robot serves as a mobile scout, using proprioceptive leg-terrain interactions to estimate regolith strength during locomotion and construct spatially resolved terrain maps. These maps are integrated with rover locomotion models to estimate traversal risk and inform path planning. We validate the framework through analogue missions at the NASA Ames Lunar Simulant Testbed and the White Sands Dune Field. Experiments demonstrate (1) online terrain strength mapping from legged locomotion and (2) rover-specific traversal-risk estimation enabling safe navigation to scientific targets. Results show that scout-generated terrain maps reliably capture spatial variability and predict mobility failure modes, allowing risk-aware path planning that avoids hazardous regions. By combining embodied terrain sensing with heterogeneous rover cooperation, this framework enhances operational robustness and expands the reachable science workspace in deformable planetary environments.
>
---
#### [replaced 033] Boundary-Guided Trajectory Prediction for Road Aware and Physically Feasible Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于轨迹预测任务，旨在解决自动驾驶中轨迹的路权合规性和物理可行性问题。通过引入边界约束和加速度预测，提升预测准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2505.06740](https://arxiv.org/pdf/2505.06740)**

> **作者:** Ahmed Abouelazm; Mianzhi Liu; Christian Hubschneider; Yin Wu; Daniel Slieter; J. Marius Zöllner
>
> **备注:** Accepted in the 36th IEEE Intelligent Vehicles Symposium (IV 2025)
>
> **摘要:** Accurate prediction of surrounding road users' trajectories is essential for safe and efficient autonomous driving. While deep learning models have improved performance, challenges remain in preventing off-road predictions and ensuring kinematic feasibility. Existing methods incorporate road-awareness modules and enforce kinematic constraints but lack plausibility guarantees and often introduce trade-offs in complexity and flexibility. This paper proposes a novel framework that formulates trajectory prediction as a constrained regression guided by permissible driving directions and their boundaries. Using the agent's current state and an HD map, our approach defines the valid boundaries and ensures on-road predictions by training the network to learn superimposed paths between left and right boundary polylines. To guarantee feasibility, the model predicts acceleration profiles that determine the vehicle's travel distance along these paths while adhering to kinematic constraints. We evaluate our approach on the Argoverse-2 dataset against the HPTR baseline. Our approach shows a slight decrease in benchmark metrics compared to HPTR but notably improves final displacement error and eliminates infeasible trajectories. Moreover, the proposed approach has superior generalization to less prevalent maneuvers and unseen out-of-distribution scenarios, reducing the off-road rate under adversarial attacks from 66% to just 1%. These results highlight the effectiveness of our approach in generating feasible and robust predictions.
>
---
#### [replaced 034] Diffusion-Based Impedance Learning for Contact-Rich Manipulation Tasks
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人接触操作任务，解决接触-rich场景下运动生成与阻抗控制的结合问题。通过扩散模型生成接触一致轨迹，并在线调整阻抗参数，提升操作精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.19696](https://arxiv.org/pdf/2509.19696)**

> **作者:** Noah Geiger; Tamim Asfour; Neville Hogan; Johannes Lachner
>
> **备注:** 15 pages, 12 figures
>
> **摘要:** Learning-based methods excel at robot motion generation but remain limited in contact-rich physical interaction. Impedance control provides stable and safe contact behavior but requires task-specific tuning of stiffness and damping parameters. We present Diffusion-Based Impedance Learning, a framework that bridges these paradigms by combining generative modeling with energy-consistent impedance control. A Transformer-based Diffusion Model, conditioned via cross-attention on measured external wrenches, reconstructs simulated Zero-Force Trajectories (sZFTs) that represent contact-consistent equilibrium behavior. A SLERP-based quaternion noise scheduler preserves geometric consistency for rotations on the unit sphere. The reconstructed sZFT is used by an energy-based estimator to adapt impedance online through directional stiffness and damping modulation. Trained on parkour and robot-assisted therapy demonstrations collected via Apple Vision Pro teleoperation, the model achieves sub-millimeter positional and sub-degree rotational accuracy using only tens of thousands of samples. Deployed in realtime torque control on a KUKA LBR iiwa, the approach enables smooth obstacle traversal and generalizes to unseen tasks, achieving 100% success in multi-geometry peg-in-hole insertion.
>
---
#### [replaced 035] Runge-Kutta Approximations for Direct Coning Compensation Applying Lie Theory
- **分类: cs.RO**

- **简介: 该论文属于导航系统任务，解决陀螺仪积分中的锥形补偿问题，提出基于Runge-Kutta方法的新型补偿算法。**

- **链接: [https://arxiv.org/pdf/2511.00412](https://arxiv.org/pdf/2511.00412)**

> **作者:** John A. Christian; Michael R. Walker II; Wyatt Bridgman; Michael J. Sparapany
>
> **备注:** Accepted manuscript. AIAA JGCD
>
> **摘要:** The integration of gyroscope measurements is an essential task for most navigation systems. Modern vehicles typically use strapdown systems, such that gyro integration requires coning compensation to account for the sensor's rotation during the integration. Many coning compensation algorithms have been developed and a few are reviewed. This work introduces a new class of coning correction algorithm built directly from the classical Runge-Kutta integration routines. A simple case is shown to collapse to one of the most popular coning algorithms and a clear procedure for generating higher-order algorithms is presented.
>
---
#### [replaced 036] Walk Like Dogs: Learning Steerable Imitation Controllers for Legged Robots from Unlabeled Motion Data
- **分类: cs.RO**

- **简介: 该论文属于机器人模仿学习任务，解决从无标签数据中学习可操控的步态问题。通过自动提取行为模式并映射用户指令，实现风格一致的运动模仿。**

- **链接: [https://arxiv.org/pdf/2507.00677](https://arxiv.org/pdf/2507.00677)**

> **作者:** Dongho Kang; Jin Cheng; Fatemeh Zargarbashi; Taerim Yoon; Sungjoon Choi; Stelian Coros
>
> **备注:** The supplementary video is available at this https URL
>
> **摘要:** We present an imitation learning framework that extracts distinctive legged locomotion behaviors and transitions between them from unlabeled real-world motion data. By automatically discovering behavioral modes and mapping user steering commands to them, the framework enables user-steerable and stylistically consistent motion imitation. Our approach first bridges the morphological and physical gap between the motion source and the robot by transforming raw data into a physically consistent, robot-compatible dataset using a kino-dynamic motion retargeting strategy. This data is used to train a steerable motion synthesis module that generates stylistic, multi-modal kinematic targets from high-level user commands. These targets serve as a reference for a reinforcement learning controller, which reliably executes them on the robot hardware. In our experiments, a controller trained on dog motion data demonstrated distinctive quadrupedal gait patterns and emergent gait transitions in response to varying velocity commands. These behaviors were achieved without manual labeling, predefined mode counts, or explicit switching rules, maintaining the stylistic coherence of the data.
>
---
#### [replaced 037] Viewpoint Matters: Dynamically Optimizing Viewpoints with Masked Autoencoder for Visual Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉任务，旨在解决单相机系统视角固定导致的适应性差问题。通过动态选择最优视角，提升视觉感知效果。**

- **链接: [https://arxiv.org/pdf/2602.04243](https://arxiv.org/pdf/2602.04243)**

> **作者:** Pengfei Yi; Yifan Han; Junyan Li; Litao Liu; Wenzhao Lian
>
> **摘要:** Robotic manipulation continues to be a challenge, and imitation learning (IL) enables robots to learn tasks from expert demonstrations. Current IL methods typically rely on fixed camera setups, where cameras are manually positioned in static locations, imposing significant limitations on adaptability and coverage. Inspired by human active perception, where humans dynamically adjust their viewpoint to capture the most relevant and least noisy information, we propose MAE-Select, a novel framework for active viewpoint selection in single-camera robotic systems. MAE-Select fully leverages pre-trained multi-view masked autoencoder representations and dynamically selects the next most informative viewpoint at each time chunk without requiring labeled viewpoints. Extensive experiments demonstrate that MAE-Select improves the capabilities of single-camera systems and, in some cases, even surpasses multi-camera setups. The project will be available at this https URL.
>
---
#### [replaced 038] MachaGrasp: Morphology-Aware Cross-Embodiment Dexterous Hand Articulation Generation for Grasping
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出MachaGrasp，解决多指灵巧手跨形态抓取生成问题。通过形态嵌入和特征抓取集，结合物体点云与手腕姿态，生成关节运动，提升抓取成功率。**

- **链接: [https://arxiv.org/pdf/2510.06068](https://arxiv.org/pdf/2510.06068)**

> **作者:** Heng Zhang; Kevin Yuchen Ma; Mike Zheng Shou; Weisi Lin; Yan Wu
>
> **摘要:** Dexterous grasping with multi-fingered hands remains challenging due to high-dimensional articulations and the cost of optimization-based pipelines. Existing end-to-end methods require training on large-scale datasets for specific hands, limiting their ability to generalize across different embodiments. We propose MachaGrasp, an eigengrasp-based, end-to-end framework for cross-embodiment grasp generation. From a hand's morphology description, we derive a morphology embedding and an eigengrasp set. Conditioned on these, together with the object point cloud and wrist pose, an amplitude predictor regresses articulation coefficients in a low-dimensional space, which are decoded into full joint articulations. Articulation learning is supervised with a Kinematic-Aware Articulation Loss (KAL) that emphasizes fingertip-relevant motions and injects morphology-specific structure. In simulation on unseen objects across three dexterous hands, MachaGrasp attains a 91.9% average grasp success rate with less than 0.4 seconds inference per grasp. With few-shot adaptation to an unseen hand, it achieves 85.6% success on unseen objects in simulation, and real-world experiments on this few-shot-generalized hand achieve an 87% success rate. The code and additional materials are available on our project website this https URL.
>
---
#### [replaced 039] EgoTraj-Bench: Towards Robust Trajectory Prediction Under Ego-view Noisy Observations
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于轨迹预测任务，解决ego视角下噪声观测导致的预测不鲁棒问题。提出EgoTraj-Bench基准和BiFlow模型，提升预测精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.00405](https://arxiv.org/pdf/2510.00405)**

> **作者:** Jiayi Liu; Jiaming Zhou; Ke Ye; Kun-Yu Lin; Allan Wang; Junwei Liang
>
> **摘要:** Reliable trajectory prediction from an ego-centric perspective is crucial for robotic navigation in human-centric environments. However, existing methods typically assume noiseless observation histories, failing to account for the perceptual artifacts inherent in first-person vision, such as occlusions, ID switches, and tracking drift. This discrepancy between training assumptions and deployment reality severely limits model robustness. To bridge this gap, we introduce EgoTraj-Bench, built upon TBD dataset, which is the first real-world benchmark that aligns noisy, first-person visual histories with clean, bird's-eye-view future trajectories, enabling robust learning under realistic perceptual constraints. Building on this benchmark, we propose BiFlow, a dual-stream flow matching model that concurrently denoises historical observations and forecasts future motion. To better model agent intent, BiFlow incorporates our EgoAnchor mechanism, which conditions the prediction decoder on distilled historical features via feature modulation. Extensive experiments show that BiFlow achieves state-of-the-art performance, reducing minADE and minFDE by 10-15% on average and demonstrating superior robustness. We anticipate that our benchmark and model will provide a critical foundation for robust real-world ego-centric trajectory prediction. The benchmark library is available at: this https URL.
>
---
#### [replaced 040] Interpretable Multimodal Gesture Recognition for Drone and Mobile Robot Teleoperation via Log-Likelihood Ratio Fusion
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于手势识别任务，旨在解决无人机和移动机器人远程操作中的手势识别问题。通过多模态数据融合提升识别性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2602.23694](https://arxiv.org/pdf/2602.23694)**

> **作者:** Seungyeol Baek; Jaspreet Singh; Lala Shakti Swarup Ray; Hymalai Bello; Paul Lukowicz; Sungho Suh
>
> **摘要:** Human operators are still frequently exposed to hazardous environments such as disaster zones and industrial facilities, where intuitive and reliable teleoperation of mobile robots and Unmanned Aerial Vehicles (UAVs) is essential. In this context, hands-free teleoperation enhances operator mobility and situational awareness, thereby improving safety in hazardous environments. While vision-based gesture recognition has been explored as one method for hands-free teleoperation, its performance often deteriorates under occlusions, lighting variations, and cluttered backgrounds, limiting its applicability in real-world operations. To overcome these limitations, we propose a multimodal gesture recognition framework that integrates inertial data (accelerometer, gyroscope, and orientation) from Apple Watches on both wrists with capacitive sensing signals from custom gloves. We design a late fusion strategy based on the log-likelihood ratio (LLR), which not only enhances recognition performance but also provides interpretability by quantifying modality-specific contributions. To support this research, we introduce a new dataset of 20 distinct gestures inspired by aircraft marshalling signals, comprising synchronized RGB video, IMU, and capacitive sensor data. Experimental results demonstrate that our framework achieves performance comparable to a state-of-the-art vision-based baseline while significantly reducing computational cost, model size, and training time, making it well suited for real-time robot control. We therefore underscore the potential of sensor-based multimodal fusion as a robust and interpretable solution for gesture-driven mobile robot and drone teleoperation.
>
---
#### [replaced 041] Learning Agile Gate Traversal via Analytical Optimal Policy Gradient
- **分类: cs.RO**

- **简介: 该论文属于无人机敏捷飞行任务，解决窄门穿越问题。提出混合框架，结合神经网络与模型预测控制，提升穿越效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2508.21592](https://arxiv.org/pdf/2508.21592)**

> **作者:** Tianchen Sun; Bingheng Wang; Nuthasith Gerdpratoom; Longbin Tang; Yichao Gao; Lin Zhao
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Traversing narrow gates presents a significant challenge and has become a standard benchmark for evaluating agile and precise quadrotor flight. Traditional modularized autonomous flight stacks require extensive design and parameter tuning, while end-to-end reinforcement learning (RL) methods often suffer from low sample efficiency, limited interpretability, and degraded disturbance rejection under unseen perturbations. In this work, we present a novel hybrid framework that adaptively fine-tunes model predictive control (MPC) parameters online using outputs from a neural network (NN) trained offline. The NN jointly predicts a reference pose and cost function weights, conditioned on the coordinates of the gate corners and the current drone state. To achieve efficient training, we derive analytical policy gradients not only for the MPC module but also for an optimization-based gate traversal detection module. Hardware experiments demonstrate agile and accurate gate traversal with peak accelerations of $30\ \mathrm{m/s^2}$, as well as recovery within $0.85\ \mathrm{s}$ following body-rate disturbances exceeding $1146\ \mathrm{deg/s}$.
>
---
#### [replaced 042] Quadrotor Navigation using Reinforcement Learning with Privileged Information
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于无人机导航任务，解决大障碍物绕行问题。通过强化学习结合特权信息，提升导航成功率。**

- **链接: [https://arxiv.org/pdf/2509.08177](https://arxiv.org/pdf/2509.08177)**

> **作者:** Jonathan Lee; Abhishek Rathod; Kshitij Goel; John Stecklein; Wennie Tabib
>
> **摘要:** This paper presents a reinforcement learning-based quadrotor navigation method that leverages efficient differentiable simulation, novel loss functions, and privileged information to navigate around large obstacles. Prior learning-based methods perform well in scenes that exhibit narrow obstacles, but struggle when the goal location is blocked by large walls or terrain. In contrast, the proposed method utilizes time-of-arrival (ToA) maps as privileged information and a yaw alignment loss to guide the robot around large obstacles. The policy is evaluated in photo-realistic simulation environments containing large obstacles, sharp corners, and dead-ends. Our approach achieves an 86% success rate and outperforms baseline strategies by 34%. We deploy the policy onboard a custom quadrotor in outdoor cluttered environments both during the day and night. The policy is validated across 20 flights, covering 589 meters without collisions at speeds up to 4 m/s.
>
---
#### [replaced 043] Vision Language Model-based Testing of Industrial Autonomous Mobile Robots
- **分类: cs.SE; cs.RO**

- **简介: 该论文属于工业AMR安全测试任务，旨在解决真实环境测试成本高、风险大的问题。通过VLM生成违反安全要求的人类行为场景，提升测试有效性。**

- **链接: [https://arxiv.org/pdf/2508.02338](https://arxiv.org/pdf/2508.02338)**

> **作者:** Jiahui Wu; Chengjie Lu; Aitor Arrieta; Shaukat Ali; Thomas Peyrucain
>
> **摘要:** PAL Robotics, in Spain, builds a variety of Autonomous Mobile Robots (AMRs), which are deployed in diverse environments (e.g., warehouses, retail spaces, and offices), where they work alongside humans. Given that human behavior can be unpredictable and that AMRs may not have been trained to handle all possible unknown and uncertain behaviors, it is important to test AMRs under a wide range of human interactions to ensure their safe behavior. Moreover, testing in real environments with actual AMRs and humans is often costly, impractical, and potentially hazardous (e.g., it could result in human injury). To this end, we propose a Vision Language Model (VLM)-based testing approach (RVSG) for industrial AMRs developed together with PAL Robotics. Based on the functional and safety requirements, RVSG uses the VLM to generate diverse human behaviors that violate these requirements. We evaluated RVSG with several requirements and navigation routes in a simulator using the latest AMR from PAL Robotics. Our results show that, compared with the baseline, RVSG can effectively generate requirement-violating scenarios. Moreover, RVSG-generated scenarios increase variability in robot behavior, thereby helping reveal their uncertain behaviors.
>
---
#### [replaced 044] Least Restrictive Hyperplane Control Barrier Functions
- **分类: cs.RO**

- **简介: 该论文属于控制安全领域，解决动态系统安全控制问题。针对传统CBF的保守性，提出最小限制超平面CBF，优化超平面方向以提升控制性能。**

- **链接: [https://arxiv.org/pdf/2510.18643](https://arxiv.org/pdf/2510.18643)**

> **作者:** Mattias Trende; Petter Ögren
>
> **摘要:** Control Barrier Functions (CBFs) can provide provable safety guarantees for dynamic systems. However, finding a valid CBF for a system of interest is often non-trivial, especially for systems having low computational resources, higher-order dynamics, and moving close to obstacles of complex shape. A common solution to this problem is to use a purely distance-based CBF. In this paper, we study Hyperplane CBFs (H-CBFs), where a hyperplane separates the agent from the obstacle. First, we note that the common distance-based CBF is a special case of an H-CBF where the hyperplane is a supporting hyperplane of the obstacle that is orthogonal to a line between the agent and the obstacle. Then we show that a less conservative CBF can be found by optimising over the orientation of the supporting hyperplane, in order to find the Least Restrictive Hyperplane CBF. This enables us to maintain the safety guarantees while allowing controls that are closer to the desired ones, especially when moving fast and passing close to obstacles. We illustrate the approach on a double integrator dynamical system with acceleration constraints, moving through a group of arbitrarily shaped static and moving obstacles.
>
---
#### [replaced 045] Infinite-Dimensional Closed-Loop Inverse Kinematics for Soft Robots via Neural Operators
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究软体机器人的逆运动学问题，提出一种基于神经算子的无限维闭环逆运动学方法，解决软体机器人因欠驱动导致的运动控制难题。**

- **链接: [https://arxiv.org/pdf/2602.18655](https://arxiv.org/pdf/2602.18655)**

> **作者:** Carina Veil; Moritz Flaschel; Ellen Kuhl; Cosimo Della Santina
>
> **摘要:** For fully actuated rigid robots, kinematic inversion is a purely geometric problem, efficiently solved by closed-loop inverse kinematics (CLIK) schemes that compute joint configurations to position the robot body in space. For underactuated soft robots, however, not all configurations are attainable through control action, making kinematic inversion extremely challenging. Extensions of CLIK address this by introducing end-to-end mappings from actuation to task space for the controller to operate on, but typically assume finite dimensions of the underlying virtual configuration space. In this work, we formulate CLIK in the infinite-dimensional domain to reason about the entire soft robot shape while solving tasks. We do this by composing an actuation-to-shape map with a shape-to-task map, deriving the differential end-to-end kinematics via an infinite-dimensional chain rule, and thereby obtaining a Jacobian-based CLIK algorithm. Since this actuation-to-shape mapping is rarely available in closed form, we propose to learn it using differentiable neural operator networks. We first present an analytical study on a constant-curvature segment, and then apply the neural version of the algorithm to a three-fiber soft robotic arm whose underlying model relies on morphoelasticity and active filament theory.
>
---
#### [replaced 046] NeuralRemaster: Phase-Preserving Diffusion for Structure-Aligned Generation
- **分类: cs.CV; cs.GR; cs.LG; cs.RO**

- **简介: 该论文提出一种相位保持的扩散模型（φ-PD），解决图像生成中结构不一致问题，适用于需要几何一致性的任务，如重渲染和图像到图像翻译。**

- **链接: [https://arxiv.org/pdf/2512.05106](https://arxiv.org/pdf/2512.05106)**

> **作者:** Yu Zeng; Charles Ochoa; Mingyuan Zhou; Vishal M. Patel; Vitor Guizilini; Rowan McAllister
>
> **摘要:** Standard diffusion corrupts data using Gaussian noise whose Fourier coefficients have random magnitudes and random phases. While effective for unconditional or text-to-image generation, corrupting phase components destroys spatial structure, making it ill-suited for tasks requiring geometric consistency, such as re-rendering, simulation enhancement, and image-to-image translation. We introduce Phase-Preserving Diffusion (\phi-PD), a model-agnostic reformulation of the diffusion process that preserves input phase while randomizing magnitude, enabling structure-aligned generation without architectural changes or additional parameters. We further propose Frequency-Selective Structured (FSS) noise, which provides continuous control over structural rigidity via a single frequency-cutoff parameter. \phi-PD adds no inference-time cost and is compatible with any diffusion model for images or videos. Across photorealistic and stylized re-rendering, as well as sim-to-real enhancement for driving planners, \phi-PD produces controllable, spatially aligned results. When applied to the CARLA simulator, \phi-PD significantly improves sim-to-real planner transfer performance. The method is complementary to existing conditioning approaches and broadly applicable to image-to-image and video-to-video generation. Videos, additional examples, and code are available on our \href{this https URL}{project page}.
>
---
#### [replaced 047] Ask, Reason, Assist: Robot Collaboration via Natural Language and Temporal Logic
- **分类: cs.RO**

- **简介: 该论文属于机器人协作任务，解决异构机器人团队在冲突时的协同问题。通过自然语言和时序逻辑实现机器人间自主请求与提供帮助，提升任务效率。**

- **链接: [https://arxiv.org/pdf/2509.23506](https://arxiv.org/pdf/2509.23506)**

> **作者:** Dan BW Choe; Sundhar Vinodh Sangeetha; Steven Emanuel; Chih-Yuan Chiu; Samuel Coogan; Shreyas Kousik
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2505.13376
>
> **摘要:** Increased robot deployment, such as in warehousing, has revealed a need for collaboration among heterogeneous robot teams to resolve unforeseen conflicts. To this end, we propose a peer-to-peer coordination protocol that enables robots to request and provide help without a central task allocator. The process begins when a robot detects a conflict and uses a Large Language Model (LLM) to decide whether external assistance is required. If so, it crafts and broadcasts a natural language (NL) help request. Potential helper robots reason over the request and respond with offers of assistance, including information about the effect on their ongoing tasks. Helper reasoning is implemented via an LLM grounded in Signal Temporal Logic (STL) using a Backus-Naur Form (BNF) grammar, ensuring syntactically valid NL-to-STL translations, which are then solved as a Mixed Integer Linear Program (MILP). Finally, the requester robot selects a helper by reasoning over the expected increase in system-level total task completion time. We evaluated our framework through experiments comparing different helper-selection strategies and found that considering multiple offers allows the requester to minimize added makespan. Our approach significantly outperforms heuristics such as selecting the nearest available candidate helper robot, and achieves performance comparable to a centralized "Oracle" baseline but without heavy information demands.
>
---
#### [replaced 048] Seeing Through Uncertainty: A Free-Energy Approach for Real-Time Perceptual Adaptation in Robust Visual Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉导航任务，解决传感器噪声下的鲁棒性问题。提出FEP-Nav框架，通过分解自由能实现实时感知适应。**

- **链接: [https://arxiv.org/pdf/2403.01977](https://arxiv.org/pdf/2403.01977)**

> **作者:** Maytus Piriyajitakonkij; Rishabh Dev Yadav; Mingfei Sun; Mengmi Zhang; Wei Pan
>
> **摘要:** Navigation in the natural world is a feat of adaptive inference, where biological organisms maintain goal-directed behaviour despite noisy and incomplete sensory streams. Central to this ability is the Free Energy Principle (FEP), which posits that perception is a generative process where the brain minimises Variational Free Energy (VFE) to maintain accurate internal models of the world. While Deep Neural Networks (DNNs) have served as powerful analogues for biological brains, they typically lack the real-time plasticity required to handle abrupt sensory shifts. We introduce FEP-Nav, a biologically-inspired framework that implements real-time perceptual adaptation for robust visual navigation. By decomposing VFE into its constituent components--prediction error and Bayesian surprise--we propose a dual-mechanism architecture: a Top-down Decoder that provides an internal expectation of uncorrupted sensory input, and Adaptive Normalisation that dynamically aligns shifted feature distributions with prior beliefs. Theoretically, we demonstrate that this integration of reconstruction and normalisation provides a formal mechanism for minimising VFE during inference without the need for gradient-based updates. Evaluations across a diverse suite of simulated and real-world visual corruptions demonstrate that FEP-Nav facilitates a substantial recovery of navigation performance, consistently exceeding the capabilities of both non-adaptive baselines and strong adaptive methods. We show that bridging machine learning with the brain's variational principles offers a robust strategy for autonomous behaviour, enabling robots to remain functional under sensory conditions that typically degrade the performance of standard adaptive models.
>
---
#### [replaced 049] MarketGen: A Scalable Simulation Platform with Auto-Generated Embodied Supermarket Environments
- **分类: cs.RO**

- **简介: 该论文提出MarketGen平台，解决商业环境仿真不足问题，支持自动生成超市场景，提供基准任务以促进具身智能研究。**

- **链接: [https://arxiv.org/pdf/2511.21161](https://arxiv.org/pdf/2511.21161)**

> **作者:** Xu Hu; Yiyang Feng; Junran Peng; Jiawei He; Liyi Chen; Wei Sui; Chuanchen Luo; Xucheng Yin; Qing Li; Zhaoxiang Zhang
>
> **备注:** Project Page: this https URL
>
> **摘要:** The development of embodied agents for complex commercial environments is hindered by a critical gap in existing robotics datasets and benchmarks, which primarily focus on household or tabletop settings with short-horizon tasks. To address this limitation, we introduce MarketGen, a scalable simulation platform with automatic scene generation for complex supermarket environments. MarketGen features a novel agent-based Procedural Content Generation (PCG) framework. It uniquely supports multi-modal inputs (text and reference images) and integrates real-world design principles to automatically generate complete, structured, and realistic supermarkets. We also provide an extensive and diverse 3D asset library with a total of 1100+ supermarket goods and parameterized facilities assets. Building on this generative foundation, we propose a novel benchmark for assessing supermarket agents, featuring two daily tasks in a supermarket: (1) Checkout Unloading: long-horizon tabletop tasks for cashier agents, and (2) In-Aisle Item Collection: complex mobile manipulation tasks for salesperson agents. We validate our platform and benchmark through extensive experiments, including the deployment of a modular agent system and successful sim-to-real transfer. MarketGen provides a comprehensive framework to accelerate research in embodied AI for complex commercial applications.
>
---
#### [replaced 050] Kinodynamic Task and Motion Planning using VLM-guided and Interleaved Sampling
- **分类: cs.RO**

- **简介: 该论文属于任务与运动规划（TAMP）领域，旨在解决长时域规划中运动采样成本高的问题。提出一种基于混合状态树的 kinodynamic TAMP 方法，结合 VLM 引导和回溯，提升成功率并减少规划时间。**

- **链接: [https://arxiv.org/pdf/2510.26139](https://arxiv.org/pdf/2510.26139)**

> **作者:** Minseo Kwon; Young J. Kim
>
> **摘要:** Task and Motion Planning (TAMP) integrates high-level task planning with low-level motion feasibility, but existing methods are costly in long-horizon problems due to excessive motion sampling. While LLMs provide commonsense priors, they lack 3D spatial reasoning and cannot ensure geometric or dynamic feasibility. We propose a kinodynamic TAMP planner based on a hybrid state tree that uniformly represents symbolic and numeric states during planning, enabling task and motion decisions to be jointly decided. Kinodynamic constraints embedded in the TAMP problem are verified by an off-the-shelf motion planner and physics simulator, and a VLM guides exploring a TAMP solution and backtracks the search based on visual rendering of the states. Experiments on the simulated domains and in the real world show 32.14% - 1166.67% increased average success rates compared to traditional and LLM-based TAMP planners and reduced planning time on complex problems, with ablations further highlighting the benefits of VLM backtracking. More details are available at this https URL.
>
---
