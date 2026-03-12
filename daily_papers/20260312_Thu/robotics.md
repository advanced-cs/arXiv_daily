# 机器人 cs.RO

- **最新发布 62 篇**

- **更新 27 篇**

## 最新发布

#### [new 001] Interleaving Scheduling and Motion Planning with Incremental Learning of Symbolic Space-Time Motion Abstractions
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究多目标导航中的调度与运动规划问题，通过交替使用调度器和运动规划器，在增量学习中生成符合时空约束的可行方案。**

- **链接: [https://arxiv.org/pdf/2603.10651](https://arxiv.org/pdf/2603.10651)**

> **作者:** Elisa Tosello; Arthur Bit-Monnot; Davide Lusuardi; Alessandro Valentini; Andrea Micheli
>
> **摘要:** Task and Motion Planning combines high-level task sequencing (what to do) with low-level motion planning (how to do it) to generate feasible, collision-free execution plans. However, in many real-world domains, such as automated warehouses, tasks are predefined, shifting the challenge to if, when, and how to execute them safely and efficiently under resource, time and motion constraints. In this paper, we formalize this as the Scheduling and Motion Planning problem for multi-object navigation in shared workspaces. We propose a novel solution framework that interleaves off-the-shelf schedulers and motion planners in an incremental learning loop. The scheduler generates candidate plans, while the motion planner checks feasibility and returns symbolic feedback, i.e., spatial conflicts and timing adjustments, to guide the scheduler towards motion-feasible solutions. We validate our proposal on logistics and job-shop scheduling benchmarks augmented with motion tasks, using state-of-the-art schedulers and sampling-based motion planners. Our results show the effectiveness of our framework in generating valid plans under complex temporal and spatial constraints, where synchronized motion is critical.
>
---
#### [new 002] AR-VLA: True Autoregressive Action Expert for Vision-Language-Action Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出AR-VLA，解决机器人动作生成中的时序一致性问题，通过自回归方式生成连续动作，提升任务成功率与动作平滑性。**

- **链接: [https://arxiv.org/pdf/2603.10126](https://arxiv.org/pdf/2603.10126)**

> **作者:** Yutong Hu; Jan-Nico Zaech; Nikolay Nikolov; Yuanqi Yao; Sombit Dey; Giuliano Albanese; Renaud Detry; Luc Van Gool; Danda Paudel
>
> **摘要:** We propose a standalone autoregressive (AR) Action Expert that generates actions as a continuous causal sequence while conditioning on refreshable vision-language prefixes. In contrast to existing Vision-Language-Action (VLA) models and diffusion policies that reset temporal context with each new observation and predict actions reactively, our Action Expert maintains its own history through a long-lived memory and is inherently context-aware. This structure addresses the frequency mismatch between fast control and slow reasoning, enabling efficient independent pretraining of kinematic syntax and modular integration with heavy perception backbones, naturally ensuring spatio-temporally consistent action generation across frames. To synchronize these asynchronous hybrid V-L-A modalities, we utilize a re-anchoring mechanism that mathematically accounts for perception staleness during both training and inference. Experiments on simulated and real-robot manipulation tasks demonstrate that the proposed method can effectively replace traditional chunk-based action heads for both specialist and generalist policies. AR-VLA exhibits superior history awareness and substantially smoother action trajectories while maintaining or exceeding the task success rates of state-of-the-art reactive VLAs. Overall, our work introduces a scalable, context-aware action generation schema that provides a robust structural foundation for training effective robotic policies.
>
---
#### [new 003] FAR-Dex: Few-shot Data Augmentation and Adaptive Residual Policy Refinement for Dexterous Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在解决多指手与机械臂协作中的精细操作难题。通过引入少量数据增强和自适应残差优化，提升操作精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.10451](https://arxiv.org/pdf/2603.10451)**

> **作者:** Yushan Bai; Fulin Chen; Hongzheng Sun; Yuchuang Tong; En Li; Zhengtao Zhang
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Achieving human-like dexterous manipulation through the collaboration of multi-fingered hands with robotic arms remains a longstanding challenge in robotics, primarily due to the scarcity of high-quality demonstrations and the complexity of high-dimensional action spaces. To address these challenges, we propose FAR-Dex, a hierarchical framework that integrates few-shot data augmentation with adaptive residual refinement to enable robust and precise arm-hand coordination in dexterous tasks. First, FAR-DexGen leverages the IsaacLab simulator to generate diverse and physically constrained trajectories from a few demonstrations, providing a data foundation for policy training. Second, FAR-DexRes introduces an adaptive residual module that refines policies by combining multi-step trajectory segments with observation features, thereby enhancing accuracy and robustness in manipulation scenarios. Experiments in both simulation and real-world demonstrate that FAR-Dex improves data quality by 13.4% and task success rates by 7% over state-of-the-art methods. It further achieves over 80% success in real-world tasks, enabling fine-grained dexterous manipulation with strong positional generalization.
>
---
#### [new 004] PC-Diffuser: Path-Consistent Capsule CBF Safety Filtering for Diffusion-Based Trajectory Planner
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶轨迹规划任务，旨在解决扩散模型在复杂场景中安全性不足的问题。提出PC-Diffuser框架，通过嵌入安全约束提升轨迹生成的安全性与可行性。**

- **链接: [https://arxiv.org/pdf/2603.10330](https://arxiv.org/pdf/2603.10330)**

> **作者:** Eugene Ku; Yiwei Lyu
>
> **摘要:** Autonomous driving in complex traffic requires planners that generalize beyond hand-crafted rules, motivating data-driven approaches that learn behavior from expert demonstrations. Diffusion-based trajectory planners have recently shown strong closed-loop performance by iteratively denoising a full-horizon plan, but they remain difficult to certify and can fail catastrophically in rare or out-of-distribution scenarios. To address this challenge, we present PC-Diffuser, a safety augmentation framework that embeds a certifiable, path-consistent barrier-function structure directly into the denoising loop of diffusion planning. The key idea is to make safety an intrinsic part of trajectory generation rather than a post-hoc fix: we enforce forward invariance along the rollout while preserving the diffusion model's intended path geometry. Specifically, PC-Diffuser (i) evaluates collision risk using a capsule-distance barrier function that better reflects vehicle geometry and reduces unnecessary conservativeness, (ii) converts denoised waypoints into dynamically feasible motion under a kinematic bicycle model, and (iii) applies a path-consistent safety filter that eliminates residual constraint violations without geometric distortion, so the corrected plan remains close to the learned distribution. By injecting these safety-consistent corrections at every denoising step and feeding the refined trajectory back into the diffusion process, PC-Diffuser enables iterative, context-aware safeguarding instead of post-hoc repair...
>
---
#### [new 005] RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion
- **分类: cs.RO**

- **简介: 该论文提出一种结合强化学习与模型预测控制的架构，用于解决非步态和混合运动的机器人控制问题，通过学习无循环步态降低计算负担。**

- **链接: [https://arxiv.org/pdf/2603.10878](https://arxiv.org/pdf/2603.10878)**

> **作者:** Andrea Patrizi; Carlo Rizzardo; Arturo Laurenzi; Francesco Ruscelli; Luca Rossini; Nikos G. Tsagarakis
>
> **摘要:** We propose a contact-explicit hierarchical architecture coupling Reinforcement Learning (RL) and Model Predictive Control (MPC), where a high-level RL agent provides gait and navigation commands to a low-level locomotion MPC. This offloads the combinatorial burden of contact timing from the MPC by learning acyclic gaits through trial and error in simulation. We show that only a minimal set of rewards and limited tuning are required to obtain effective policies. We validate the architecture in simulation across robotic platforms spanning 50 kg to 120 kg and different MPC implementations, observing the emergence of acyclic gaits and timing adaptations in flat-terrain legged and hybrid locomotion, and further demonstrating extensibility to non-flat terrains. Across all platforms, we achieve zero-shot sim-to-sim transfer without domain randomization, and we further demonstrate zero-shot sim-to-real transfer without domain randomization on Centauro, our 120 kg wheeled-legged humanoid robot. We make our software framework and evaluation results publicly available at this https URL.
>
---
#### [new 006] Cybo-Waiter: A Physical Agentic Framework for Humanoid Whole-Body Locomotion-Manipulation
- **分类: cs.RO**

- **简介: 该论文提出Cybo-Waiter框架，解决人形机器人在部分可观测环境下执行复杂任务的问题。通过视觉语言计划与3D监督实现可靠长时序操作。**

- **链接: [https://arxiv.org/pdf/2603.10675](https://arxiv.org/pdf/2603.10675)**

> **作者:** Peng Ren; Haoyang Ge; Chuan Qi; Cong Huang; Hong Li; Jiang Zhao; Pei Chi; Kai Chen
>
> **摘要:** Robots are increasingly expected to execute open ended natural language requests in human environments, which demands reliable long horizon execution under partial observability. This is especially challenging for humanoids because locomotion and manipulation are tightly coupled through stance, reachability, and balance. We present a humanoid agent framework that turns VLM plans into verifiable task programs and closes the loop with multi object 3D geometric supervision. A VLM planner compiles each instruction into a typed JSON sequence of subtasks with explicit predicate based preconditions and success conditions. Using SAM3 and RGB-D, we ground all task relevant entities in 3D, estimate object centroids and extents, and evaluate predicates over stable frames to obtain condition level diagnostics. The supervisor uses these diagnostics to verify subtask completion and to provide condition-level feedback for progression and replanning. We execute each subtask by coordinating humanoid locomotion and whole-body manipulation, selecting feasible motion primitives under reachability and balance constraints. Experiments on tabletop manipulation and long horizon humanoid loco manipulation tasks show improved robustness from multi object grounding, temporal stability, and recovery driven replanning.
>
---
#### [new 007] GRACE: A Unified 2D Multi-Robot Path Planning Simulator & Benchmark for Grid, Roadmap, And Continuous Environments
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文提出GRACE，一个统一的2D多机器人路径规划模拟器与基准测试平台，解决不同抽象层次下的路径规划比较问题。**

- **链接: [https://arxiv.org/pdf/2603.10858](https://arxiv.org/pdf/2603.10858)**

> **作者:** Chuanlong Zang; Anna Mannucci; Isabelle Barz; Philipp Schillinger; Florian Lier; Wolfgang Hönig
>
> **备注:** ICRA 2026, code will be released soon
>
> **摘要:** Advancing Multi-Agent Pathfinding (MAPF) and Multi-Robot Motion Planning (MRMP) requires platforms that enable transparent, reproducible comparisons across modeling choices. Existing tools either scale under simplifying assumptions (grids, homogeneous agents) or offer higher fidelity with less comparable instrumentation. We present GRACE, a unified 2D simulator+benchmark that instantiates the same task at multiple abstraction levels (grid, roadmap, continuous) via explicit, reproducible operators and a common evaluation protocol. Our empirical results on public maps and representative planners enable commensurate comparisons on a shared instance set. Furthermore, we quantify the expected representation-fidelity trade-offs (MRMP solves instances at higher fidelity but lower speed, while grid/roadmap planners scale farther). By consolidating representation, execution, and evaluation, GRACE thereby aims to make cross-representation studies more comparable and provides a means to advance multi-robot planning research and its translation to practice.
>
---
#### [new 008] Cross-Hand Latent Representation for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决多机械手间泛化能力不足的问题。提出XL-VLA框架，构建统一潜在动作空间，实现跨机械手高效学习与数据复用。**

- **链接: [https://arxiv.org/pdf/2603.10158](https://arxiv.org/pdf/2603.10158)**

> **作者:** Guangqi Jiang; Yutong Liang; Jianglong Ye; Jia-Yang Huang; Changwei Jing; Rocky Duan; Pieter Abbeel; Xiaolong Wang; Xueyan Zou
>
> **备注:** Website: this https URL
>
> **摘要:** Dexterous manipulation is essential for real-world robot autonomy, mirroring the central role of human hand coordination in daily activity. Humans rely on rich multimodal perception--vision, sound, and language-guided intent--to perform dexterous actions, motivating vision-based, language-conditioned manipulation systems for robots. However, training reliable vision-language-action (VLA) models for dexterous manipulation requires large-scale demonstrations across many robotic hands. In addition, as new dexterous embodiments appear rapidly, collecting data for each becomes costly and impractical, creating a need for scalable cross-embodiment learning. We introduce XL-VLA, a vision-language-action framework integrated with a unified latent action space shared across diverse dexterous hands. This embodiment-invariant latent space is directly pluggable into standard VLA architectures, enabling seamless cross-embodiment training and efficient reuse of both existing and newly collected data. Experimental results demonstrate that XL-VLA consistently outperforms baseline VLA models operating in raw joint spaces, establishing it as an effective solution for scalable cross-embodiment dexterous manipulation.
>
---
#### [new 009] Shape Control of a Planar Hyper-Redundant Robot via Hybrid Kinematics-Informed and Learning-based Approach
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决超冗余机器人形状控制的稳定性与精度问题。通过融合运动学与学习方法，提出SpacioCoupledNet框架，提升控制性能。**

- **链接: [https://arxiv.org/pdf/2603.10402](https://arxiv.org/pdf/2603.10402)**

> **作者:** Yuli Song; Wenbo Li; Wenci Xin; Zhiqiang Tang; Daniela Rus; Cecilia Laschi
>
> **摘要:** Hyper-redundant robots offer high dexterity, making them good at operating in confined and unstructured environments. To extend the reachable workspace, we built a multi-segment flexible rack actuated planar robot. However, the compliance of the flexible mechanism introduces instability, rendering it sensitive to external and internal uncertainties. To address these limitations, we propose a hybrid kinematics-informed and learning-based shape control method, named SpatioCoupledNet. The neural network adopts a hierarchical design that explicitly captures bidirectional spatial coupling between segments while modeling local disturbance along the robot body. A confidence-gating mechanism integrates prior kinematic knowledge, allowing the controller to adaptively balance model-based and learned components for improved convergence and fidelity. The framework is validated on a five-segment planar hyper-redundant robot under three representative shape configurations. Experimental results demonstrate that the proposed method consistently outperforms both analytical and purely neural controllers. In complex scenarios, it reduces steady-state error by up to 75.5% against the analytical model, and accelerates convergence by up to 20.5% compared to the data-driven baseline. Furthermore, gating analysis reveals a state-dependent authority fusion, shifting toward data-driven predictions in unstable states, while relying on physical priors in the remaining cases. Finally, we demonstrate robust performance in a dynamic task where the robot maintains a fixed end-effector position while avoiding moving obstacles, achieving a precise tip-positioning accuracy with a mean error of 10.47 mm.
>
---
#### [new 010] Degeneracy-Resilient Teach and Repeat for Geometrically Challenging Environments Using FMCW Lidar
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决GPS缺失环境下路径重复定位问题。提出基于FMCW激光雷达的鲁棒系统，提升复杂地形中的定位可靠性。**

- **链接: [https://arxiv.org/pdf/2603.10248](https://arxiv.org/pdf/2603.10248)**

> **作者:** Katya M. Papais; Wenda Zhao; Timothy D. Barfoot
>
> **摘要:** Teach and Repeat (T&R) topometric navigation enables robots to autonomously repeat previously traversed paths without relying on GPS, making it well suited for operations in GPS-denied environments such as underground mines and lunar navigation. State-of-the-art T&R systems typically rely on iterative closest point (ICP)-based estimation; however, in geometrically degenerate environments with sparsely structured terrain, ICP often becomes ill-conditioned, resulting in degraded localization and unreliable navigation performance. To address this challenge, we present a degeneracy-resilient Frequency-Modulated Continuous-Wave (FMCW) lidar T&R navigation system consisting of Doppler velocity-based odometry and degeneracy-aware scan-to-map localization. Leveraging FMCW lidar, which provides per-point radial velocity measurements via the Doppler effect, we extend a geometry-independent, correspondence-free motion estimation to include principled pose uncertainty estimation that remains stable in degenerate environments. We further propose a degeneracy-aware localization method that incorporates per-point curvature for improved data association, and unifies translational and rotational scales to enable consistent degeneracy detection. Closed-loop field experiments across three environments with varying structural richness demonstrate that the proposed system reliably completes autonomous navigation, including in a challenging flat airport test field where a conventional ICP-based system fails.
>
---
#### [new 011] ScanDP: Generalizable 3D Scanning with Diffusion Policy
- **分类: cs.RO**

- **简介: 该论文属于3D扫描任务，旨在解决传统方法泛化能力差和数据依赖性强的问题。提出基于扩散策略的框架，结合占用网格和路径优化，提升扫描效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.10390](https://arxiv.org/pdf/2603.10390)**

> **作者:** Itsuki Hirako; Ryo Hakoda; Yubin Liu; Matthew Hwang; Yoshihiro Sato; Takeshi Oishi
>
> **备注:** 8 pages, 7 figures, 5 tables. Project Page: this https URL
>
> **摘要:** Learning-based 3D Scanning plays a crucial role in enabling efficient and accurate scanning of target objects. However, recent reinforcement learning-based methods often require large-scale training data and still struggle to generalize to unseen object this http URL this work, we propose a data-efficient 3D scanning framework that uses Diffusion Policy to imitate human-like scanning strategies. To enhance robustness and generalization, we adopt the Occupancy Grid Mapping instead of direct point cloud processing, offering improved noise resilience and handling of diverse object geometries. We also introduce a hybrid approach combining a sphere-based space representation with a path optimization procedure that ensures path safety and scanning efficiency. This approach addresses limitations in conventional imitation learning, such as redundant or unpredictable behavior. We evaluate our method on diverse unseen objects in both shape and scale. Ours achieves higher coverage and shorter paths than baselines, while remaining robust to sensor noise. We further confirm practical feasibility and stable operation in real-world execution.
>
---
#### [new 012] KnowDiffuser: A Knowledge-Guided Diffusion Planner with LM Reasoning and Prior-Informed Trajectory Initialization
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶运动规划任务，旨在解决语义与物理轨迹不匹配的问题。融合语言模型与扩散模型，生成语义合理且物理可行的轨迹。**

- **链接: [https://arxiv.org/pdf/2603.10441](https://arxiv.org/pdf/2603.10441)**

> **作者:** Fan Ding; Xuewen Luo; Fengze Yang; Bo Yu; HwaHui Tew; Ganesh Krishnasamy; Junn Yong Loo
>
> **备注:** 10pages, 1 figure
>
> **摘要:** Recent advancements in Language Models (LMs) have demonstrated strong semantic reasoning capabilities, enabling their application in high-level decision-making for autonomous driving (AD). However, LMs operate over discrete token spaces and lack the ability to generate continuous, physically feasible trajectories required for motion planning. Meanwhile, diffusion models have proven effective at generating reliable and dynamically consistent trajectories, but often lack semantic interpretability and alignment with scene-level understanding. To address these limitations, we propose \textbf{KnowDiffuser}, a knowledge-guided motion planning framework that tightly integrates the semantic understanding of language models with the generative power of diffusion models. The framework employs a language model to infer context-aware meta-actions from structured scene representations, which are then mapped to prior trajectories that anchor the subsequent denoising process. A two-stage truncated denoising mechanism refines these trajectories efficiently, preserving both semantic alignment and physical feasibility. Experiments on the nuPlan benchmark demonstrate that KnowDiffuser significantly outperforms existing planners in both open-loop and closed-loop evaluations, establishing a robust and interpretable framework that effectively bridges the semantic-to-physical gap in AD systems.
>
---
#### [new 013] Parallel-in-Time Nonlinear Optimal Control via GPU-native Sequential Convex Programming
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于实时轨迹优化任务，解决CPU计算效率低的问题，提出一种基于GPU的并行非线性最优控制方法，提升计算速度与能效。**

- **链接: [https://arxiv.org/pdf/2603.10711](https://arxiv.org/pdf/2603.10711)**

> **作者:** Yilin Zou; Zhong Zhang; Fanghua Jiang
>
> **摘要:** Real-time trajectory optimization for nonlinear constrained autonomous systems is critical and typically performed by CPU-based sequential solvers. Specifically, reliance on global sparse linear algebra or the serial nature of dynamic programming algorithms restricts the utilization of massively parallel computing architectures like GPUs. To bridge this gap, we introduce a fully GPU-native trajectory optimization framework that combines sequential convex programming with a consensus-based alternating direction method of multipliers. By applying a temporal splitting strategy, our algorithm decouples the optimization horizon into independent, per-node subproblems that execute massively in parallel. The entire process runs fully on the GPU, eliminating costly memory transfers and large-scale sparse factorizations. This architecture naturally scales to multi-trajectory optimization. We validate the solver on a quadrotor agile flight task and a Mars powered descent problem using an on-board edge computing platform. Benchmarks reveal a sustained 4x throughput speedup and a 51% reduction in energy consumption over a heavily optimized 12-core CPU baseline. Crucially, the framework saturates the hardware, maintaining over 96% active GPU utilization to achieve planning rates exceeding 100 Hz. Furthermore, we demonstrate the solver's extensibility to robust Model Predictive Control by jointly optimizing dynamically coupled scenarios under stochastic disturbances, enabling scalable and safe autonomy.
>
---
#### [new 014] A gripper for flap separation and opening of sealed bags
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于医疗机器人任务，旨在解决手动打开密封袋的重复性难题。设计新型夹爪，实现可靠抓取与密封开启，提升效率并减少人力负担。**

- **链接: [https://arxiv.org/pdf/2603.10890](https://arxiv.org/pdf/2603.10890)**

> **作者:** Sergi Foix; Jaume Oriol; Carme Torras; Júlia Borràs
>
> **备注:** 8 pages, Accepted at the 2026 IEEE International Conference on Robotics & Automation (ICRA2026)
>
> **摘要:** Separating thin, flexible layers that must be individually grasped is a common but challenging manipulation primitive for most off-the-shelf grippers. A prominent example arises in clinical settings: the opening of sterile flat pouches for the preparation of the operating room, where the first step is to separate and grasp the flaps. We present a novel gripper design and opening strategy that enables reliable flap separation and robust seal opening. This capability addresses a high-volume repetitive hospital procedure in which nurses manually open up to 240 bags per shift, a physically demanding task linked to musculoskeletal injuries. Our design combines an active dented-roller fingertip with compliant fingers that exploit environmental constraints to robustly grasp thin flexible flaps. Experiments demonstrate that the proposed gripper reliably grasps and separates sealed bag flaps and other thin-layered materials from the hospital, the most sensitive variable affecting performance being the normal force applied. When two copies of the gripper grasp both flaps, the system withstands the forces needed to open the seals robustly. To our knowledge, this is one of the first demonstrations of robotic assistance to automate this repetitive, low-value, but critical hospital task.
>
---
#### [new 015] TacLoc: Global Tactile Localization on Objects from a Registration Perspective
- **分类: cs.RO**

- **简介: 论文提出TacLoc，用于物体姿态估计任务，解决视觉受阻时的定位问题。通过点云配准方法，无需预训练模型，实现高效准确的触觉定位。**

- **链接: [https://arxiv.org/pdf/2603.10565](https://arxiv.org/pdf/2603.10565)**

> **作者:** Zirui Zhang; Boyang Zhang; Fumin Zhang; Huan Yin
>
> **备注:** 8 pages, 12 figures
>
> **摘要:** Pose estimation is essential for robotic manipulation, particularly when visual perception is occluded during gripper-object interactions. Existing tactile-based methods generally rely on tactile simulation or pre-trained models, which limits their generalizability and efficiency. In this study, we propose TacLoc, a novel tactile localization framework that formulates the problem as a one-shot point cloud registration task. TacLoc introduces a graph-theoretic partial-to-full registration method, leveraging dense point clouds and surface normals from tactile sensing for efficient and accurate pose estimation. Without requiring rendered data or pre-trained models, TacLoc achieves improved performance through normal-guided graph pruning and a hypothesis-and-verification pipeline. TacLoc is evaluated extensively on the YCB dataset. We further demonstrate TacLoc on real-world objects across two different visual-tactile sensors.
>
---
#### [new 016] Learning Adaptive Force Control for Contact-Rich Sample Scraping with Heterogeneous Materials
- **分类: cs.RO**

- **简介: 该论文属于机器人操控任务，旨在解决复杂材料刮取中的力控制问题。通过自适应控制框架和强化学习，实现高效、稳定的样本刮取。**

- **链接: [https://arxiv.org/pdf/2603.10979](https://arxiv.org/pdf/2603.10979)**

> **作者:** Cenk Cetin; Shreyas Pouli; Gabriella Pizzuto
>
> **备注:** 8 pages, 6 figures, 4 tables; Submitted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2026
>
> **摘要:** The increasing demand for accelerated scientific discovery, driven by global challenges, highlights the need for advanced AI-driven robotics. Deploying robotic chemists in human-centric labs is key for the next horizon of autonomous discovery, as complex tasks still demand the dexterity of human scientists. Robotic manipulation in this context is uniquely challenged by handling diverse chemicals (granular, powdery, or viscous liquids), under varying lab conditions. For example, humans use spatulas for scraping materials from vial walls. Automating this process is challenging because it goes beyond simple robotic insertion tasks and traditional lab automation, requiring the execution of fine-granular movements within a constrained environment (the sample vial). Our work proposes an adaptive control framework to address this, relying on a low-level Cartesian impedance controller for stable and compliant physical interaction and a high-level reinforcement learning agent that learns to dynamically adjust interaction forces at the end-effector. The agent is guided by perception feedback, which provides the material's location. We first created a task-representative simulation environment with a Franka Research 3 robot, a scraping tool, and a sample vial containing heterogeneous materials. To facilitate the learning of an adaptive policy and model diverse characteristics, the sample is modelled as a collection of spheres, where each sphere is assigned a unique dislodgement force threshold, which is procedurally generated using Perlin noise. We train an agent to autonomously learn and adapt the optimal contact wrench for a sample scraping task in simulation and then successfully transfer this policy to a real robotic setup. Our method was evaluated across five different material setups, outperforming a fixed-wrench baseline by an average of 10.9%.
>
---
#### [new 017] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于在线高精度地图构建任务，解决标注成本高的问题。通过引入地理空间对比学习，提升特征表示，实现半监督训练，提升地图感知性能。**

- **链接: [https://arxiv.org/pdf/2603.10688](https://arxiv.org/pdf/2603.10688)**

> **作者:** Jonas Merkert; Alexander Blumberg; Jan-Hendrik Pauls; Christoph Stiller
>
> **摘要:** Autonomous vehicles rely on map information to understand the world around them. However, the creation and maintenance of offline high-definition (HD) maps remains costly. A more scalable alternative lies in online HD map construction, which only requires map annotations at training time. To further reduce the need for annotating vast training labels, self-supervised training provides an alternative. This work focuses on improving the latent birds-eye-view (BEV) feature grid representation within a vectorized online HD map construction model by enforcing geospatial consistency between overlapping BEV feature grids as part of a contrastive loss function. To ensure geospatial overlap for contrastive pairs, we introduce an approach to analyze the overlap between traversals within a given dataset and generate subsidiary dataset splits following adjustable multi-traversal requirements. We train the same model supervised using a reduced set of single-traversal labeled data and self-supervised on a broader unlabeled set of data following our multi-traversal requirements, effectively implementing a semi-supervised approach. Our approach outperforms the supervised baseline across the board, both quantitatively in terms of the downstream tasks vectorized map perception performance and qualitatively in terms of segmentation in the principal component analysis (PCA) visualization of the BEV feature space.
>
---
#### [new 018] OnFly: Onboard Zero-Shot Aerial Vision-Language Navigation toward Safety and Efficiency
- **分类: cs.RO**

- **简介: 该论文属于航空视觉-语言导航任务，解决零样本AVLN中的决策不稳定、长距离监控不可靠及安全与效率的平衡问题。提出OnFly框架，通过双代理架构和语义几何验证提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.10682](https://arxiv.org/pdf/2603.10682)**

> **作者:** Guiyong Zheng; Yueting Ban; Mingjie Zhang; Juepeng Zheng; Boyu Zhou
>
> **摘要:** Aerial vision-language navigation (AVLN) enables UAVs to follow natural-language instructions in complex 3D environments. However, existing zero-shot AVLN methods often suffer from unstable single-stream Vision-Language Model decision-making, unreliable long-horizon progress monitoring, and a trade-off between safety and efficiency. We propose OnFly, a fully onboard, real-time framework for zero-shot AVLN. OnFly adopts a shared-perception dual-agent architecture that decouples high-frequency target generation from low-frequency progress monitoring, thereby stabilizing decision-making. It further employs a hybrid keyframe-recent-frame memory to preserve global trajectory context while maintaining KV-cache prefix stability, enabling reliable long-horizon monitoring with termination and recovery signals. In addition, a semantic-geometric verifier refines VLM-predicted targets for instruction consistency and geometric safety using VLM features and depth cues, while a receding-horizon planner generates optimized collision-free trajectories under geometric safety constraints, improving both safety and efficiency. In simulation, OnFly improves task success from 26.4% to 67.8%, compared with the strongest state-of-the-art baseline, while fully onboard real-world flights validate its feasibility for real-time deployment. The code will be released at this https URL
>
---
#### [new 019] Recover to Predict: Progressive Retrospective Learning for Variable-Length Trajectory Prediction
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于轨迹预测任务，解决变量长度观测下的预测问题。提出PRF框架，通过逐步对齐特征提升短轨迹预测精度。**

- **链接: [https://arxiv.org/pdf/2603.10597](https://arxiv.org/pdf/2603.10597)**

> **作者:** Hao Zhou; Lu Qi; Jason Li; Jie Zhang; Yi Liu; Xu Yang; Mingyu Fan; Fei Luo
>
> **备注:** Paper is accepted by CVPR 2026
>
> **摘要:** Trajectory prediction is critical for autonomous driving, enabling safe and efficient planning in dense, dynamic traffic. Most existing methods optimize prediction accuracy under fixed-length observations. However, real-world driving often yields variable-length, incomplete observations, posing a challenge to these methods. A common strategy is to directly map features from incomplete observations to those from complete ones. This one-shot mapping, however, struggles to learn accurate representations for short trajectories due to significant information gaps. To address this issue, we propose a Progressive Retrospective Framework (PRF), which gradually aligns features from incomplete observations with those from complete ones via a cascade of retrospective units. Each unit consists of a Retrospective Distillation Module (RDM) and a Retrospective Prediction Module (RPM), where RDM distills features and RPM recovers previous timesteps using the distilled features. Moreover, we propose a Rolling-Start Training Strategy (RSTS) that enhances data efficiency during PRF training. PRF is plug-and-play with existing methods. Extensive experiments on datasets Argoverse 2 and Argoverse 1 demonstrate the effectiveness of PRF. Code is available at this https URL.
>
---
#### [new 020] DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control
- **分类: cs.RO**

- **简介: 该论文提出DiT4DiT模型，解决机器人控制中动态与动作联合建模问题，通过视频生成提升控制性能与样本效率。**

- **链接: [https://arxiv.org/pdf/2603.10448](https://arxiv.org/pdf/2603.10448)**

> **作者:** Teli Ma; Jia Zheng; Zifan Wang; Chuili Jiang; Andy Cui; Junwei Liang; Shuo Yang
>
> **备注:** this https URL
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising paradigm for robot learning, but their representations are still largely inherited from static image-text pretraining, leaving physical dynamics to be learned from comparatively limited action data. Generative video models, by contrast, encode rich spatiotemporal structure and implicit physics, making them a compelling foundation for robotic manipulation. But their potentials are not fully explored in the literature. To bridge the gap, we introduce DiT4DiT, an end-to-end Video-Action Model that couples a video Diffusion Transformer with an action Diffusion Transformer in a unified cascaded framework. Instead of relying on reconstructed future frames, DiT4DiT extracts intermediate denoising features from the video generation process and uses them as temporally grounded conditions for action prediction. We further propose a dual flow-matching objective with decoupled timesteps and noise scales for video prediction, hidden-state extraction, and action inference, enabling coherent joint training of both modules. Across simulation and real-world benchmarks, DiT4DiT achieves state-of-the-art results, reaching average success rates of 98.6% on LIBERO and 50.8% on RoboCasa GR1 while using substantially less training data. On the Unitree G1 robot, it also delivers superior real-world performance and strong zero-shot generalization. Importantly, DiT4DiT improves sample efficiency by over 10x and speeds up convergence by up to 7x, demonstrating that video generation can serve as an effective scaling proxy for robot policy learning. We release code and models at this https URL.
>
---
#### [new 021] Safety-critical Control Under Partial Observability: Reach-Avoid POMDP meets Belief Space Control
- **分类: cs.RO**

- **简介: 该论文属于机器人决策任务，解决部分可观测环境下的安全控制问题。通过分层控制架构和BCLFs/BCBFs实现目标达成、信息收集与安全的协同。**

- **链接: [https://arxiv.org/pdf/2603.10572](https://arxiv.org/pdf/2603.10572)**

> **作者:** Matti Vahs; Joris Verhagen; Jana Tumova
>
> **摘要:** Partially Observable Markov Decision Processes (POMDPs) provide a principled framework for robot decision-making under uncertainty. Solving reach-avoid POMDPs, however, requires coordinating three distinct behaviors: goal reaching, safety, and active information gathering to reduce uncertainty. Existing online POMDP solvers attempt to address all three within a single belief tree search, but this unified approach struggles with the conflicting time scales inherent to these objectives. We propose a layered, certificate-based control architecture that operates directly in belief space, decoupling goal reaching, information gathering, and safety into modular components. We introduce Belief Control Lyapunov Functions (BCLFs) that formalize information gathering as a Lyapunov convergence problem in belief space, and show how they can be learned via reinforcement learning. For safety, we develop Belief Control Barrier Functions (BCBFs) that leverage conformal prediction to provide probabilistic safety guarantees over finite horizons. The resulting control synthesis reduces to lightweight quadratic programs solvable in real time, even for non-Gaussian belief representations with dimension $>10^4$. Experiments in simulation and on a space-robotics platform demonstrate real-time performance and improved safety and task success compared to state-of-the-art constrained POMDP solvers.
>
---
#### [new 022] COHORT: Hybrid RL for Collaborative Large DNN Inference on Multi-Robot Systems Under Real-Time Constraints
- **分类: cs.RO; cs.DC**

- **简介: 该论文提出COHORT框架，解决多机器人系统在实时约束下协同执行大模型任务的问题。通过混合强化学习策略优化任务分配与资源利用。**

- **链接: [https://arxiv.org/pdf/2603.10436](https://arxiv.org/pdf/2603.10436)**

> **作者:** Mohammad Saeid Anwar; Anuradha Ravi; Indrajeet Ghosh; Gaurav Shinde; Carl Busart; Nirmalya Roy
>
> **备注:** Recently accepted at 27th IEEE International Symposium on a World of Wireless, Mobile and Multimedia Networks ( IEEE WoWMoM 2026)
>
> **摘要:** Large deep neural networks (DNNs), especially transformer-based and multimodal architectures, are computationally demanding and challenging to deploy on resource-constrained edge platforms like field robots. These challenges intensify in mission-critical scenarios (e.g., disaster response), where robots must collaborate under tight constraints on bandwidth, latency, and battery life, often without infrastructure or server support. To address these limitations, we present COHORT, a collaborative DNN inference and task-execution framework for multi-robot systems built on the Robotic Operating System (ROS). COHORT employs a hybrid offline-online reinforcement learning (RL) strategy to dynamically schedule and distribute DNN module execution across robots. Our key contributions are threefold: (a) Offline RL policy learning combined with Advantage-Weighted Regression (AWR), trained on auction-based task allocation data from heterogeneous DNN workloads across distributed robots, (b) Online policy adaptation via Multi-Agent PPO (MAPPO), initialized from the offline policy and fine-tuned in real time, and (c) comprehensive evaluation of COHORT on vision-language model (VLM) inference tasks such as CLIP and SAM, analyzing scalability with increasing robot/workload and robustness under . We benchmark COHORT against genetic algorithms and multiple RL baselines. Experimental results demonstrate that COHORT reduces battery consumption by 15.4% and increases GPU utilization by 51.67%, while satisfying frame-rate and deadline constraints 2.55 times of the time.
>
---
#### [new 023] MAVEN: A Meta-Reinforcement Learning Framework for Varying-Dynamics Expertise in Agile Quadrotor Maneuvers
- **分类: cs.RO**

- **简介: 该论文提出MAVEN框架，解决四旋翼无人机在动态变化下的自适应导航问题。通过元强化学习，实现跨不同动力学特性的敏捷飞行控制。**

- **链接: [https://arxiv.org/pdf/2603.10714](https://arxiv.org/pdf/2603.10714)**

> **作者:** Jin Zhou; Dongcheng Cao; Xian Wang; Shuo Li
>
> **摘要:** Reinforcement learning (RL) has emerged as a powerful paradigm for achieving online agile navigation with quadrotors. Despite this success, policies trained via standard RL typically fail to generalize across significant dynamic variations, exhibiting a critical lack of adaptability. This work introduces MAVEN, a meta-RL framework that enables a single policy to achieve robust end-to-end navigation across a wide range of quadrotor dynamics. Our approach features a novel predictive context encoder, which learns to infer a latent representation of the system dynamics from interaction history. We demonstrate our method in agile waypoint traversal tasks under two challenging scenarios: large variations in quadrotor mass and severe single-rotor thrust loss. We leverage a GPU-vectorized simulator to distribute tasks across thousands of parallel environments, overcoming the long training times of meta-RL to converge in less than an hour. Through extensive experiments in both simulation and the real world, we validate that MAVEN achieves superior adaptation and agility. The policy successfully executes zero-shot sim-to-real transfer, demonstrating robust online adaptation by performing high-speed maneuvers despite mass variations of up to 66.7% and single-rotor thrust losses as severe as 70%.
>
---
#### [new 024] Adaptive Manipulation Potential and Haptic Estimation for Tool-Mediated Interaction
- **分类: cs.RO**

- **简介: 该论文属于工具交互任务，解决接触丰富环境下的操作与触觉估计问题。提出参数化平衡流形和闭环框架，实现动态轨迹规划与自适应阻抗控制。**

- **链接: [https://arxiv.org/pdf/2603.10352](https://arxiv.org/pdf/2603.10352)**

> **作者:** Lin Yang; Anirvan Dutta; Yuan Ji; Yanxin Zhou; Shilin Shan; Lv Chen; Etienne Burdet; Domenico Campolo
>
> **摘要:** Achieving human-level dexterity in contact-rich, tool-mediated manipulation remains a significant challenge due to visual occlusion and the underdetermined nature of haptic sensing. This paper introduces a parameterized Equilibrium Manifold (EM) as a unified representation for tool-mediated interaction, and develops a closed-loop framework that integrates haptic estimation, online planning, and adaptive stiffness control. We establish a physical-geometric duality using an adaptive manipulation potential incorporating a differentiable contact model, which induces the manifold's geometric structure and ensures that complex physical interactions are encapsulated as continuous operations on the EM. Within this framework, we reformulate haptic estimation as a manifold parameter estimation problem. Specifically, a hybrid inference strategy (haptic SLAM) is employed in which discrete object shapes are classified via particle filtering, while the continuous object pose is estimated using analytical gradients for efficient optimization. By continuously updating the parameters of the manipulation potential, the framework dynamically reshapes the induced EM to guide online trajectory replanning and implement uncertainty-aware impedance control, thereby closing the perception-action loop. The system is validated through simulation and over 260 real-world screw-loosening trials. Experimental results demonstrate robust identification and manipulation success in standard scenarios while maintaining accurate tracking. Furthermore, ablation studies confirm that haptic SLAM and uncertainty-aware stiffness modulation outperform fixed impedance baselines, effectively preventing jamming during tight tolerance interactions.
>
---
#### [new 025] Model-Free Co-Optimization of Manufacturable Sensor Layouts and Deformation Proprioception
- **分类: cs.RO**

- **简介: 该论文属于软体机器人感知任务，解决传感器布局与变形预测协同优化问题。通过数据驱动方法联合优化传感器配置和网络参数，提升预测精度并确保可制造性。**

- **链接: [https://arxiv.org/pdf/2603.10059](https://arxiv.org/pdf/2603.10059)**

> **作者:** Yingjun Tian; Guoxin Fang; Aoran Lyu; Xilong Wang; Zikang Shi; Yuhu Guo; Weiming Wang; Charlie C.L. Wang
>
> **摘要:** Flexible sensors are increasingly employed in soft robotics and wearable devices to provide proprioception of freeform this http URL supervised learning can train shape predictors from sensor signals, prediction accuracy strongly depends on sensor layout, which is typically determined heuristically or through trial-and-error. This work introduces a model-free, data-driven computational pipeline that jointly optimizes the number, length, and placement of flexible length-measurement sensors together with the parameters of a shape prediction network for large freeform deformations. Unlike model-based approaches, the proposed method relies solely on datasets of deformed shapes, without requiring physical simulation models, and is therefore broadly applicable to diverse robotic sensing tasks. The pipeline incorporates differentiable loss functions that account for both prediction accuracy and manufacturability constraints. By co-optimizing sensor layouts and network parameters, the method significantly improves deformation prediction accuracy over unoptimized layouts while ensuring practical feasibility. The effectiveness and generality of the approach are validated through numerical and physical experiments on multiple soft robotic and wearable systems.
>
---
#### [new 026] AsyncMDE: Real-Time Monocular Depth Estimation via Asynchronous Spatial Memory
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出AsyncMDE，解决实时单目深度估计问题，通过异步机制降低计算成本，提升边缘部署可行性。**

- **链接: [https://arxiv.org/pdf/2603.10438](https://arxiv.org/pdf/2603.10438)**

> **作者:** Lianjie Ma; Yuquan Li; Bingzheng Jiang; Ziming Zhong; Han Ding; Lijun Zhu
>
> **备注:** 8 pages, 5 figures, 5 tables
>
> **摘要:** Foundation-model-based monocular depth estimation offers a viable alternative to active sensors for robot perception, yet its computational cost often prohibits deployment on edge platforms. Existing methods perform independent per-frame inference, wasting the substantial computational redundancy between adjacent viewpoints in continuous robot operation. This paper presents AsyncMDE, an asynchronous depth perception system consisting of a foundation model and a lightweight model that amortizes the foundation model's computational cost over time. The foundation model produces high-quality spatial features in the background, while the lightweight model runs asynchronously in the foreground, fusing cached memory with current observations through complementary fusion, outputting depth estimates, and autoregressively updating the memory. This enables cross-frame feature reuse with bounded accuracy degradation. At a mere 3.83M parameters, it operates at 237 FPS on an RTX 4090, recovering 77% of the accuracy gap to the foundation model while achieving a 25X parameter reduction. Validated across indoor static, dynamic, and synthetic extreme-motion benchmarks, AsyncMDE degrades gracefully between refreshes and achieves 161FPS on a Jetson AGX Orin with TensorRT, clearly demonstrating its feasibility for real-time edge deployment.
>
---
#### [new 027] Decision-Aware Uncertainty Evaluation of Vision-Language Model-Based Early Action Anticipation for Human-Robot Interaction
- **分类: cs.RO**

- **简介: 该论文属于人机交互中的早期动作预测任务，解决部分观察下模型过自信导致的交互风险问题，通过评估视觉-语言模型的不确定性并提出评估协议与指标。**

- **链接: [https://arxiv.org/pdf/2603.10061](https://arxiv.org/pdf/2603.10061)**

> **作者:** Zhaoda Du; Michael Bowman; Qiaojie Zheng; Xiaoli Zhang
>
> **摘要:** Robots in shared workspaces must interpret human actions from partial, ambiguous observations, where overconfident early predictions can lead to unsafe or disruptive interaction. This challenge is amplified in egocentric views, where viewpoint changes and occlusions increase perceptual noise and ambiguity. As a result, downstream human-robot interaction modules require not only an action hypothesis but also a trustworthy estimate of confidence under partial observation. Recent vision-language model-based approaches have been proposed for short-term action recognition due to their open-vocabulary and context-aware reasoning, but their uncertainty reliability in the temporal-prefix regime is largely uncharacterized. We present the first systematic evaluation of uncertainty in vision-language model-based short-term action recognition for human-robot interaction. We introduce a temporal-prefix evaluation protocol and metrics for calibration and selective prediction. We also characterize miscalibration patterns and failure modes under partial observations. Our study provides the missing reliability evidence needed to use vision-language model predictions in confidence-gated human-robot interaction modules.
>
---
#### [new 028] SUBTA: A Framework for Supported User-Guided Bimanual Teleoperation in Structured Assembly
- **分类: cs.RO**

- **简介: 该论文提出SUBTA框架，解决人机协作中双臂装配的意图识别与辅助问题，通过结合意图估计、任务规划和上下文运动辅助，提升操作精度与用户体验。**

- **链接: [https://arxiv.org/pdf/2603.10459](https://arxiv.org/pdf/2603.10459)**

> **作者:** Xiao Liu; Prakash Baskaran; Songpo Li; Simon Manschitz; Wei Ma; Dirk Ruiken; Soshi Iba
>
> **备注:** 8 pages, 7 figures, accepted at ICRA 2026
>
> **摘要:** In human-robot collaboration, shared autonomy enhances human performance through precise, intuitive support. Effective robotic assistance requires accurately inferring human intentions and understanding task structures to determine optimal support timing and methods. In this paper, we present SUBTA, a supported teleoperation system for bimanual assembly that couples learned intention estimation, scene-graph task planning, and context-dependent motion assists. We validate our approach through a user study (N=12) comparing standard teleoperation, motion-support only, and SUBTA. Linear mixed-effects analysis revealed that SUBTA significantly outperformed standard teleoperation in position accuracy (p<0.001, d=1.18) and orientation accuracy (p<0.001, d=1.75), while reducing mental demand (p=0.002, d=1.34). Post-experiment ratings indicate clearer, more trustworthy visual feedback and predictable interventions in SUBTA. The results demonstrate that SUBTA greatly improves both effectiveness and user experience in teleoperation.
>
---
#### [new 029] Update-Free On-Policy Steering via Verifiers
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决行为克隆策略脆弱、操作精度不足的问题。提出UF-OPS方法，在不改变基础参数的情况下，通过验证器提升动作成功率。**

- **链接: [https://arxiv.org/pdf/2603.10282](https://arxiv.org/pdf/2603.10282)**

> **作者:** Maria Attarian; Ian Vyse; Claas Voelcker; Jasper Gerigk; Evgenii Opryshko; Anas Almasri; Sumeet Singh; Yilun Du; Igor Gilitschenski
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** In recent years, Behavior Cloning (BC) has become one of the most prevalent methods for enabling robots to mimic human demonstrations. However, despite their successes, BC policies are often brittle and struggle with precise manipulation. To overcome these issues, we propose UF-OPS, an Update-Free On-Policy Steering method that enables the robot to predict the success likelihood of its actions and adapt its strategy at execution time. We accomplish this by training verifier functions using policy rollout data obtained during an initial evaluation of the policy. These verifiers are subsequently used to steer the base policy toward actions with a higher likelihood of success. Our method improves the performance of black-box diffusion policy, without changing the base parameters, making it light-weight and flexible. We present results from both simulation and real-world data and achieve an average 49% improvement in success rate over the base policy across 5 real tasks.
>
---
#### [new 030] PPGuide: Steering Diffusion Policies with Performance Predictive Guidance
- **分类: cs.RO**

- **简介: 该论文提出PPGuide，用于改进扩散策略在机器人操作中的可靠性。解决动作序列错误累积问题，通过性能预测引导策略，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.10980](https://arxiv.org/pdf/2603.10980)**

> **作者:** Zixing Wang; Devesh K. Jha; Ahmed H. Qureshi; Diego Romeres
>
> **备注:** Accepted by ICRA'26
>
> **摘要:** Diffusion policies have shown to be very efficient at learning complex, multi-modal behaviors for robotic manipulation. However, errors in generated action sequences can compound over time which can potentially lead to failure. Some approaches mitigate this by augmenting datasets with expert demonstrations or learning predictive world models which might be computationally expensive. We introduce Performance Predictive Guidance (PPGuide), a lightweight, classifier-based framework that steers a pre-trained diffusion policy away from failure modes at inference time. PPGuide makes use of a novel self-supervised process: it uses attention-based multiple instance learning to automatically estimate which observation-action chunks from the policy's rollouts are relevant to success or failure. We then train a performance predictor on this self-labeled data. During inference, this predictor provides a real-time gradient to guide the policy toward more robust actions. We validated our proposed PPGuide across a diverse set of tasks from the Robomimic and MimicGen benchmarks, demonstrating consistent improvements in performance.
>
---
#### [new 031] SteadyTray: Learning Object Balancing Tasks in Humanoid Tray Transport via Residual Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于人形机器人托盘运输任务，旨在解决动态行走中负载不稳定的问题。提出ReST-RL框架，分离运动与稳定控制，提升运输可靠性。**

- **链接: [https://arxiv.org/pdf/2603.10306](https://arxiv.org/pdf/2603.10306)**

> **作者:** Anlun Huang; Zhenyu Wu; Soofiyan Atar; Yuheng Zhi; Michael Yip
>
> **备注:** Project website: this https URL
>
> **摘要:** Stabilizing unsecured payloads against the inherent oscillations of dynamic bipedal locomotion remains a critical engineering bottleneck for humanoids in unstructured environments. To solve this, we introduce ReST-RL, a hierarchical reinforcement learning architecture that explicitly decouples locomotion from payload stabilization, evaluated via the SteadyTray benchmark. Rather than relying on monolithic end-to-end learning, our framework integrates a robust base locomotion policy with a dynamic residual module engineered to actively cancel gait-induced perturbations at the end-effector. This architectural separation ensures steady tray transport without degrading the underlying bipedal stability. In simulation, the residual design significantly outperforms end-to-end baselines in gait smoothness and orientation accuracy, achieving a 96.9% success rate in variable velocity tracking and 74.5% robustness against external force disturbances. Successfully deployed on the Unitree G1 humanoid hardware, this modular approach demonstrates highly reliable zero-shot sim-to-real generalization across various objects and external force disturbances.
>
---
#### [new 032] Hierarchical Task Model Predictive Control for Sequential Mobile Manipulation Tasks
- **分类: cs.RO**

- **简介: 该论文提出一种分层任务模型预测控制方法，解决移动操作机器人执行序列任务的效率与反应问题，通过利用机器人冗余提升性能。**

- **链接: [https://arxiv.org/pdf/2603.10232](https://arxiv.org/pdf/2603.10232)**

> **作者:** Xintong Du; Siqi Zhou; Angela P. Schoellig
>
> **备注:** 8 pages, Published in IEEE Robotics and Automation Letters ( Volume: 9, Issue: 2, February 2024)
>
> **摘要:** Mobile manipulators are envisioned to serve more complex roles in people's everyday lives. With recent breakthroughs in large language models, task planners have become better at translating human verbal instructions into a sequence of tasks. However, there is still a need for a decision-making algorithm that can seamlessly interface with the high-level task planner to carry out the sequence of tasks efficiently. In this work, building on the idea of nonlinear lexicographic optimization, we propose a novel Hierarchical-Task Model Predictive Control framework that is able to complete sequential tasks with improved performance and reactivity by effectively leveraging the robot's redundancy. Compared to the state-of-the-art task-prioritized inverse kinematic control method, our approach has improved hierarchical trajectory tracking performance by 42% on average when facing task changes, robot singularity and reference variations. Compared to a typical single-task architecture, our proposed hierarchical task control architecture enables the robot to traverse a shorter path in task space and achieves an execution time 2.3 times faster when executing a sequence of delivery tasks. We demonstrated the results with real-world experiments on a 9 degrees of freedom mobile manipulator.
>
---
#### [new 033] Few-Shot Adaptation to Non-Stationary Environments via Latent Trend Embedding for Robotics
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人适应任务，解决非平稳环境中概念漂移问题。通过估计低维环境状态（Trend ID）实现少样本适应，无需修改模型参数。**

- **链接: [https://arxiv.org/pdf/2603.10373](https://arxiv.org/pdf/2603.10373)**

> **作者:** Yasuyuki Fujii; Emika Kameda; Hiroki Fukada; Yoshiki Mori; Tadashi Matsuo; Nobutaka Shimada
>
> **摘要:** Robotic systems operating in real-world environments often suffer from concept shift, where the input-output relationship changes due to latent environmental factors that are not directly observable. Conventional adaptation methods update model parameters, which may cause catastrophic forgetting and incur high computational cost. This paper proposes a latent Trend ID-based framework for few-shot adaptation in non-stationary environments. Instead of modifying model weights, a low-dimensional environmental state, referred to as the Trend ID, is estimated via backpropagation while the model parameters remain fixed. To prevent overfitting caused by per-sample latent variables, we introduce temporal regularization and a state transition model that enforces smooth evolution of the latent space. Experiments on a quantitative food grasping task demonstrate that the learned Trend IDs are distributed across distinct regions of the latent space with temporally consistent trajectories, and that few-shot adaptation to unseen environments is achieved without modifying model parameters. The proposed framework provides a scalable and interpretable solution for robotics applications operating across diverse and evolving environments.
>
---
#### [new 034] BinWalker: Development and Field Evaluation of a Quadruped Manipulator Platform for Sustainable Litter Collection
- **分类: cs.RO**

- **简介: 该论文属于环境清理任务，旨在解决难达区域 litter 收集问题。提出一种四足机器人平台，结合感知、移动和抓取功能，实现自主清理。**

- **链接: [https://arxiv.org/pdf/2603.10529](https://arxiv.org/pdf/2603.10529)**

> **作者:** Giulio Turrisi; Angelo Bratta; Giovanni Minelli; Gabriel Fischer Abati; Amir H. Rad; João Carlos Virgolino Soares; Claudio Semini
>
> **摘要:** Litter pollution represents a growing environmental problem affecting natural and urban ecosystems worldwide. Waste discarded in public spaces often accumulates in areas that are difficult to access, such as uneven terrains, coastal environments, parks, and roadside vegetation. Over time, these materials degrade and release harmful substances, including toxic chemicals and microplastics, which can contaminate soil and water and pose serious threats to wildlife and human health. Despite increasing awareness of the problem, litter collection is still largely performed manually by human operators, making large-scale cleanup operations labor-intensive, time-consuming, and costly. Robotic solutions have the potential to support and partially automate environmental cleanup tasks. In this work, we present a quadruped robotic system designed for autonomous litter collection in challenging outdoor scenarios. The robot combines the mobility advantages of legged locomotion with a manipulation system consisting of a robotic arm and an onboard litter container. This configuration enables the robot to detect, grasp, and store litter items while navigating through uneven terrains. The proposed system aims to demonstrate the feasibility of integrating perception, locomotion, and manipulation on a legged robotic platform for environmental cleanup tasks. Experimental evaluations conducted in outdoor scenarios highlight the effectiveness of the approach and its potential for assisting large-scale litter removal operations in environments that are difficult to reach with traditional robotic platforms. The code associated with this work can be found at: this https URL.
>
---
#### [new 035] Safe Probabilistic Planning for Human-Robot Interaction using Conformal Risk Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机交互安全控制任务，旨在解决复杂人类行为下的安全规划问题。通过结合控制屏障函数与符合风险控制，提供形式化安全保证，并动态调整安全边界以减少碰撞风险。**

- **链接: [https://arxiv.org/pdf/2603.10392](https://arxiv.org/pdf/2603.10392)**

> **作者:** Jake Gonzales; Kazuki Mizuta; Karen Leung; Lillian J. Ratliff
>
> **摘要:** In this paper, we present a novel probabilistic safe control framework for human-robot interaction that combines control barrier functions (CBFs) with conformal risk control to provide formal safety guarantees while considering complex human behavior. The approach uses conformal risk control to quantify and control the prediction errors in CBF safety values and establishes formal guarantees on the probability of constraint satisfaction during interaction. We introduce an algorithm that dynamically adjusts the safety margins produced by conformal risk control based on the current interaction context. Through experiments on human-robot navigation scenarios, we demonstrate that our approach significantly reduces collision rates and safety violations as compared to baseline methods while maintaining high success rates in goal-reaching tasks and efficient control. The code, simulations, and other supplementary material can be found on the project website: this https URL.
>
---
#### [new 036] AdaClearGrasp: Learning Adaptive Clearing for Zero-Shot Robust Dexterous Grasping in Densely Cluttered Environments
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决密集杂乱环境中精准抓取问题。提出AdaClearGrasp框架，通过自适应清除或直接抓取实现鲁棒操作。**

- **链接: [https://arxiv.org/pdf/2603.10616](https://arxiv.org/pdf/2603.10616)**

> **作者:** Zixuan Chen; Wenquan Zhang; Jing Fang; Ruiming Zeng; Zhixuan Xu; Yiwen Hou; Xinke Wang; Jieqi Shi; Jing Huo; Yang Gao
>
> **备注:** 12 pages. Under review
>
> **摘要:** In densely cluttered environments, physical interference, visual occlusions, and unstable contacts often cause direct dexterous grasping to fail, while aggressive singulation strategies may compromise safety. Enabling robots to adaptively decide whether to clear surrounding objects or directly grasp the target is therefore crucial for robust manipulation. We propose AdaClearGrasp, a closed-loop decision-execution framework for adaptive clearing and zero-shot dexterous grasping in densely cluttered environments. The framework formulates manipulation as a controllable high-level decision process that determines whether to directly grasp the target or first clear surrounding objects. A pretrained vision-language model (VLM) interprets visual observations and language task descriptions to reason about grasp interference and generate a high-level planning skeleton, which invokes structured atomic skills through a unified action interface. For dexterous grasping, we train a reinforcement learning policy with a relative hand-object distance representation, enabling zero-shot generalization across diverse object geometries and physical properties. During execution, visual feedback monitors outcomes and triggers replanning upon failures, forming a closed-loop correction mechanism. To evaluate language-conditioned dexterous grasping in clutter, we introduce Clutter-Bench, the first simulation benchmark with graded clutter complexity. It includes seven target objects across three clutter levels, yielding 210 task scenarios. We further perform sim-to-real experiments on three objects under three clutter levels (18 scenarios). Results demonstrate that AdaClearGrasp significantly improves grasp success rates in densely cluttered environments. For more videos and code, please visit our project website: this https URL.
>
---
#### [new 037] STM32-Based Smart Waste Bin for Hygienic Disposal Using Embedded Sensing and Automated Control
- **分类: cs.RO**

- **简介: 该论文属于智能硬件设计任务，旨在解决卫生与便捷的垃圾处理问题。通过STM32微控制器、超声波传感器和伺服电机实现自动开盖和垃圾满溢检测，提升卫生水平和使用体验。**

- **链接: [https://arxiv.org/pdf/2603.10660](https://arxiv.org/pdf/2603.10660)**

> **作者:** Mohammed Aman Bhuiyan; Aritra Islam Saswato; Md. Misbah Khan; Anish Paul; Ahmed Faizul Haque Dhrubo; Mohammad Abdul Qayum
>
> **备注:** This paper consists of 6 pages, with 3 figures, 3 tables, and 1 algorithm
>
> **摘要:** The increasing demand for hygienic and contactless solutions in public and private environments has encouraged the development of automated systems for everyday applications. This paper presents the design and implementation of a motion- sensing automatic waste bin using an STM32 microcontroller, ultrasonic sensors, and a servo motor. The system detects user presence through ultrasonic sensing and automatically opens the bin lid using a servo motor controlled by the microcontroller. An additional ultrasonic sensor is used to monitor the internal waste level of the bin, while an OLED display provides real- time feedback regarding system status. The proposed system offers a low-cost, reliable, and easily deployable solution for touch-free waste disposal. Experimental evaluation demonstrates fast response time, stable sensing performance, and smooth mechanical operation. The system can be effectively deployed in homes, educational institutions, hospitals, and public facilities to improve hygiene and user convenience.
>
---
#### [new 038] OmniGuide: Universal Guidance Fields for Enhancing Generalist Robot Policies
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出OMNIGUIDE框架，用于提升通用机器人策略在复杂任务中的表现，通过融合多种引导源增强视觉-语言-动作模型的性能。**

- **链接: [https://arxiv.org/pdf/2603.10052](https://arxiv.org/pdf/2603.10052)**

> **作者:** Yunzhou Song; Long Le; Yong-Hyun Park; Jie Wang; Junyao Shi; Lingjie Liu; Jiatao Gu; Eric Eaton; Dinesh Jayaraman; Kostas Daniilidis
>
> **备注:** Project Page: $\href{this https URL}{this\; url}$
>
> **摘要:** Vision-language-action(VLA) models have shown great promise as generalist policies for a large range of relatively simple tasks. However, they demonstrate limited performance on more complex tasks, such as those requiring complex spatial or semantic understanding, manipulation in clutter, or precise manipulation. We propose OMNIGUIDE, a flexible framework that improves VLA performance on such tasks by leveraging arbitrary sources of guidance, such as 3D foundation models, semantic-reasoning VLMs, and human pose models. We show how many kinds of guidance can be naturally expressed as differentiable energy functions with task-specific attractors and repellers located in 3D space, that influence the sampling of VLA actions. In this way, OMNIGUIDE enables guidance sources with complementary task-relevant strengths to improve a VLA model's performance on challenging tasks. Extensive experiments in both simulation and real-world environments, across diverse sources of guidance, demonstrate that OMNIGUIDE enhances the performance of state-of-the-art generalist policies (e.g., $\pi_{0.5}$, GR00T N1.6) significantly across success and safety rates. Critically, our unified framework matches or surpasses the performance of prior methods designed to incorporate specific sources of guidance into VLA policies. Project Page: $\href{this https URL}{this \; url}$
>
---
#### [new 039] DepthCache: Depth-Guided Training-Free Visual Token Merging for Vision-Language-Action Model Inference
- **分类: cs.RO**

- **简介: 该论文针对视觉-语言-动作模型推理中的高延迟问题，提出DepthCache框架，通过深度引导的视觉令牌压缩提升推理速度，同时保持任务性能。**

- **链接: [https://arxiv.org/pdf/2603.10469](https://arxiv.org/pdf/2603.10469)**

> **作者:** Yuquan Li; Lianjie Ma; Han Ding; Lijun Zhu
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Vision-Language-Action (VLA) models enable generalist robotic manipulation but suffer from high inference latency. This bottleneck stems from the massive number of visual tokens processed by large language backbones. Existing methods either prune or merge tokens uniformly, degrading the spatial reasoning essential for robotic control. We present DepthCache, a training-free framework that leverages depth as a structural prior for visual token compression. It partitions observations into depth-based regions and applies spatially differentiated merge ratios, preserving the near-field workspace while compressing the distant background. To exploit temporal redundancy, DepthCache distributes the merging process across consecutive frames, ensuring consistent representations while reducing per-step computation. A motion-adaptive pipeline further optimizes auxiliary view compression based on end-effector dynamics. The framework requires no model modification, generalizing across diverse VLA architectures. On the LIBERO benchmark, DepthCache achieves up to 1.28x inference speedup with less than 1% average success rate degradation across three VLA models (pi_0.5, OpenVLA, GR00T), whereas pruning and merging baselines incur 4--24% degradation at comparable compression. Real-world experiments on a physical manipulator demonstrate that DepthCache enables faster task throughput and more responsive closed-loop control in latency-sensitive scenarios.
>
---
#### [new 040] Octopus-inspired Distributed Control for Soft Robotic Arms: A Graph Neural Network-Based Attention Policy with Environmental Interaction
- **分类: cs.RO**

- **简介: 该论文提出SoftGM，一种仿章鱼的分布式控制架构，用于软体机械臂在复杂环境中自主抓取。解决多智能体强化学习中的环境交互与协同控制问题，通过图神经网络实现高效信息传递与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.10198](https://arxiv.org/pdf/2603.10198)**

> **作者:** Linxin Hou; Qirui Wu; Zhihang Qin; Yongxin Guo; Cecilia Laschi
>
> **备注:** 9 pages, 6 figures, 2 tables, submitted for IROS 2026
>
> **摘要:** This paper proposes SoftGM, an octopus-inspired distributed control architecture for segmented soft robotic arms that learn to reach targets in contact-rich environments using online obstacle discovery without relying on global obstacle geometry. SoftGM formulates each arm section as a cooperative agent and represents the arm-environment interaction as a graph. SoftGM uses a two-stage graph attention message passing scheme following a Centralised Training Decentralised Execution (CTDE) paradigm with a centralised critic and decentralised actor. We evaluate SoftGM in a Cosserat-rod simulator (PyElastica) across three tasks that increase the complexity of the environment: obstacle-free, structured obstacles, and a wall-with-hole scenario. Compared with six widely used MARL baselines (IDDPG, IPPO, ISAC, MADDPG, MAPPO, MASAC) under identical information content and training conditions, SoftGM matches strong CTDE methods in simpler settings and achieves the best performance in the wall-with-hole task. Robustness tests with observation noise, single-section actuation failure, and transient disturbances show that SoftGM preserves success while keeping control effort bounded, indicating resilient coordination driven by selective contact-relevant information routing.
>
---
#### [new 041] FutureVLA: Joint Visuomotor Prediction for Vision-Language-Action Model
- **分类: cs.RO**

- **简介: 该论文提出FutureVLA，解决视觉-语言-动作模型中的联合视觉运动预测问题。通过解耦视觉与运动信息，提升预测的时序连续性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.10712](https://arxiv.org/pdf/2603.10712)**

> **作者:** Xiaoxu Xu; Hao Li; Jinhui Ye; Yilun Chen; Jia Zeng; Xinyi Chen; Linning Xu; Dahua Lin; Weixin Li; Jiangmiao Pang
>
> **摘要:** Predictive foresight is important to intelligent embodied agents. Since the motor execution of a robot is intrinsically constrained by its visual perception of environmental geometry, effectively anticipating the future requires capturing this tightly coupled visuomotor interplay. While recent vision-language-action models attempt to incorporate future guidance, they struggle with this joint modeling. Existing explicit methods divert capacity to task-irrelevant visual details, whereas implicit methods relying on sparse frame pairs disrupt temporal continuity. By heavily relying on visual reconstruction, these methods become visually dominated, entangling static scene context with dynamic action intent. We argue that effective joint visuomotor predictive modeling requires both temporal continuity and visually-conditioned supervision decoupling. To this end, we propose FutureVLA, featuring a novel Joint Visuomotor Predictive Architecture. FutureVLA is designed to extract joint visuomotor embeddings by first decoupling visual and motor information, and then jointly encoding generalized physical priors. Specifically, in the pretraining stage, we leverage heterogeneous manipulation datasets and introduce a Joint Visuomotor Gating mechanism to structurally separate visual state preservation from temporal action modeling. It allows the motor stream to focus on continuous physical dynamics while explicitly querying visual tokens for environmental constraints, yielding highly generalizable joint visuomotor embeddings. Subsequently, in the post-training stage, we employ a latent embeddings alignment strategy, enabling diverse downstream VLA models to internalize these temporal priors without modifying their inference architectures. Extensive experiments demonstrate that FutureVLA consistently improves VLA frameworks.
>
---
#### [new 042] ASTER: Attitude-aware Suspended-payload Quadrotor Traversal via Efficient Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于无人机控制任务，解决悬吊负载无人机姿态约束下的敏捷飞行问题。提出ASTER框架，通过状态初始化策略实现自主倒飞和精准姿态控制。**

- **链接: [https://arxiv.org/pdf/2603.10715](https://arxiv.org/pdf/2603.10715)**

> **作者:** Dongcheng Cao; Jin Zhou; Shuo Li
>
> **摘要:** Agile maneuvering of the quadrotor cable-suspended system is significantly hindered by its non-smooth hybrid dynamics. While model-free Reinforcement Learning (RL) circumvents explicit differentiation of complex models, achieving attitude-constrained or inverted flight remains an open challenge due to the extreme reward sparsity under strict orientation requirements. This paper presents ASTER, a robust RL framework that achieves, to our knowledge, the first successful autonomous inverted flight for the cable-suspended system. We propose hybrid-dynamics-informed state seeding (HDSS), an initialization strategy that back-propagates target configurations through physics-consistent kinematic inversions across both taut and slack cable phases. HDSS enables the policy to discover aggressive maneuvers that are unreachable via standard exploration. Extensive simulations and real-world experiments demonstrate remarkable agility, precise attitude alignment, and robust zero-shot sim-to-real transfer across complex trajectories.
>
---
#### [new 043] Dynamic Modeling and Attitude Control of a Reaction-Wheel-Based Low-Gravity Bipedal Hopper
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于航天机器人领域，旨在解决低重力环境下跳跃机器人姿态不稳定的问题。通过引入反作用轮控制，实现飞行中姿态稳定与精准着陆。**

- **链接: [https://arxiv.org/pdf/2603.10670](https://arxiv.org/pdf/2603.10670)**

> **作者:** Shriram Hari; M Venkata Sai Nikhil; R Prasanth Kumar
>
> **备注:** Preprint. Under review
>
> **摘要:** Planetary bodies characterized by low gravitational acceleration, such as the Moon and near-Earth asteroids, impose unique locomotion constraints due to diminished contact forces and extended airborne intervals. Among traversal strategies, hopping locomotion offers high energy efficiency but is prone to mid-flight attitude instability caused by asymmetric thrust generation and uneven terrain interactions. This paper presents an underactuated bipedal hopping robot that employs an internal reaction wheel to regulate body posture during the ballistic flight phase. The system is modeled as a gyrostat, enabling analysis of the dynamic coupling between torso rotation and reaction wheel momentum. The locomotion cycle comprises three phases: a leg-driven propulsive jump, mid-air attitude stabilization via an active momentum exchange controller, and a shock-absorbing landing. A reduced-order model is developed to capture the critical coupling between torso rotation and reaction wheel dynamics. The proposed framework is evaluated in MuJoCo-based simulations under lunar gravity conditions (g = 1.625 m/s^2). Results demonstrate that activation of the reaction wheel controller reduces peak mid-air angular deviation by more than 65% and constrains landing attitude error to within 3.5 degrees at touchdown. Additionally, actuator saturation per hop cycle is reduced, ensuring sufficient control authority. Overall, the approach significantly mitigates in-flight attitude excursions and enables consistent upright landings, providing a practical and control-efficient solution for locomotion on irregular extraterrestrial terrains.
>
---
#### [new 044] Learning Bimanual Cloth Manipulation with Vision-based Tactile Sensing via Single Robotic Arm
- **分类: cs.RO**

- **简介: 该论文研究单臂双指布料操作任务，解决布料高维状态、变形及遮挡问题。提出Touch G.O.G.框架，结合新型夹爪、视觉Transformer和合成数据生成，实现高精度布料识别与定位。**

- **链接: [https://arxiv.org/pdf/2603.10609](https://arxiv.org/pdf/2603.10609)**

> **作者:** Dongmyoung Lee; Wei Chen; Xiaoshuai Chen; Rui Zong; Petar Kormushev
>
> **备注:** 11 pages, 13 figures
>
> **摘要:** Robotic cloth manipulation remains challenging due to the high-dimensional state space of fabrics, their deformable nature, and frequent occlusions that limit vision-based sensing. Although dual-arm systems can mitigate some of these issues, they increase hardware and control complexity. This paper presents Touch G.O.G., a compact vision-based tactile gripper and perception/control framework for single-arm bimanual cloth manipulation. The proposed framework combines three key components: (1) a novel gripper design and control strategy for in-gripper cloth sliding with a single robot arm, (2) a Vision Foundation Model-backboned Vision Transformer pipeline for cloth part classification (PC-Net) and edge pose estimation (PE-Net) using real and synthetic tactile images, and (3) an encoder-decoder synthetic data generator (SD-Net) that reduces manual annotation by producing high-fidelity tactile images. Experiments show 96% accuracy in distinguishing edges, corners, interior regions, and grasp failures, together with sub-millimeter edge localization and 4.5° orientation error. Real-world results demonstrate reliable cloth unfolding, even for crumpled fabrics, using only a single robotic arm. These results highlight Touch G.O.G. as a compact and cost-effective solution for deformable object manipulation.
>
---
#### [new 045] Semantic Landmark Particle Filter for Robot Localisation in Vineyards
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人定位任务，解决葡萄园中因行间感知歧义导致的定位问题，通过融合语义地标与LiDAR信息提升定位精度。**

- **链接: [https://arxiv.org/pdf/2603.10847](https://arxiv.org/pdf/2603.10847)**

> **作者:** Rajitha de Silva; Jonathan Cox; James R. Heselden; Marija Popović; Cesar Cadena; Riccardo Polvara
>
> **备注:** Submmitted to IROS 2026
>
> **摘要:** Reliable localisation in vineyards is hindered by row-level perceptual aliasing: parallel crop rows produce nearly identical LiDAR observations, causing geometry-only and vision-based SLAM systems to converge towards incorrect corridors, particularly during headland transitions. We present a Semantic Landmark Particle Filter (SLPF) that integrates trunk and pole landmark detections with 2D LiDAR within a probabilistic localisation framework. Detected trunks are converted into semantic walls, forming structural row boundaries embedded in the measurement model to improve discrimination between adjacent rows. GNSS is incorporated as a lightweight prior that stabilises localisation when semantic observations are sparse. Field experiments in a 10-row vineyard demonstrate consistent improvements over geometry-only (AMCL), vision-based (RTAB-Map), and GNSS baselines. Compared to AMCL, SLPF reduces Absolute Pose Error by 22% and 65% across two traversal directions; relative to a NoisyGNSS baseline, APE decreases by 65% and 61%. Row correctness improves from 0.67 to 0.73, while mean cross-track error decreases from 1.40 m to 1.26 m. These results show that embedding row-level structural semantics within the measurement model enables robust localisation in highly repetitive outdoor agricultural environments.
>
---
#### [new 046] From Prior to Pro: Efficient Skill Mastery via Distribution Contractive RL Finetuning
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出DICE-RL框架，用于高效精调预训练机器人策略。解决的是如何从高维视觉输入中学习复杂操作技能的问题，通过分布收缩提升性能与样本效率。**

- **链接: [https://arxiv.org/pdf/2603.10263](https://arxiv.org/pdf/2603.10263)**

> **作者:** Zhanyi Sun; Shuran Song
>
> **摘要:** We introduce Distribution Contractive Reinforcement Learning (DICE-RL), a framework that uses reinforcement learning (RL) as a "distribution contraction" operator to refine pretrained generative robot policies. DICE-RL turns a pretrained behavior prior into a high-performing "pro" policy by amplifying high-success behaviors from online feedback. We pretrain a diffusion- or flow-based policy for broad behavioral coverage, then finetune it with a stable, sample-efficient residual off-policy RL framework that combines selective behavior regularization with value-guided action selection. Extensive experiments and analyses show that DICE-RL reliably improves performance with strong stability and sample efficiency. It enables mastery of complex long-horizon manipulation skills directly from high-dimensional pixel inputs, both in simulation and on a real robot. Project website: this https URL.
>
---
#### [new 047] Autonomous Search for Sparsely Distributed Visual Phenomena through Environmental Context Modeling
- **分类: cs.RO**

- **简介: 该论文属于目标搜索任务，解决水下AUV在稀疏分布目标中高效定位的问题。通过环境上下文建模，结合单图像检测实现高效自主搜索。**

- **链接: [https://arxiv.org/pdf/2603.10174](https://arxiv.org/pdf/2603.10174)**

> **作者:** Eric Chen; Travis Manderson; Nare Karapetyan; Peter Edmunds; Nicholas Roy; Yogesh Girdhar
>
> **备注:** Accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** Autonomous underwater vehicles (AUVs) are increasingly used to survey coral reefs, yet efficiently locating specific coral species of interest remains difficult: target species are often sparsely distributed across the reef, and an AUV with limited battery life cannot afford to search everywhere. When detections of the target itself are too sparse to provide directional guidance, the robot benefits from an additional signal to decide where to look next. We propose using the visual environmental context -- the habitat features that tend to co-occur with a target species -- as that signal. Because context features are spatially denser and often vary more smoothly than target detections, we hypothesize that a reward function targeted at broader environmental context will enable adaptive planners to make better decisions on where to go next, even in regions where no target has yet been observed. Starting from a single labeled image, our method uses patch-level DINOv2 embeddings to perform one-shot detections of both the target species and its surrounding context online. We validate our approach using real imagery collected by an AUV at two reef sites in St. John, U.S. Virgin Islands, simulating the robot's motion offline. Our results demonstrate that one-shot detection combined with adaptive context modeling enables efficient autonomous surveying, sampling up to 75$\%$ of the target in roughly half the time required by exhaustive coverage when the target is sparsely distributed, and outperforming search strategies that only use target detections.
>
---
#### [new 048] FG-CLTP: Fine-Grained Contrastive Language Tactile Pretraining for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决 tactile-language-action 模型中定量接触状态表示不足的问题。通过构建数据集和提出 FG-CLTP 框架，提升触觉感知精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.10871](https://arxiv.org/pdf/2603.10871)**

> **作者:** Wenxuan Ma; Chaofan Zhang; Yinghao Cai; Guocai Yao; Shaowei Cui; Shuo Wang
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Recent advancements in integrating tactile sensing into vision-language-action (VLA) models have demonstrated transformative potential for robotic perception. However, existing tactile representations predominantly rely on qualitative descriptors (e.g., texture), neglecting quantitative contact states such as force magnitude, contact geometry, and principal axis orientation, which are indispensable for fine-grained manipulation. To bridge this gap, we propose FG-CLTP, a fine-grained contrastive language tactile pretraining framework. We first introduce a novel dataset comprising over 100k tactile 3D point cloud-language pairs that explicitly capture multidimensional contact states from the sensor's perspective. We then implement a discretized numerical tokenization mechanism to achieve quantitative-semantic alignment, effectively injecting explicit physical metrics into the multimodal feature space. The proposed FG-CLTP model yields a 95.9% classification accuracy and reduces the regression error (MAE) by 52.6% compared to state-of-the-art methods. Furthermore, the integration of 3D point cloud representations establishes a sensor-agnostic foundation with a minimal sim-to-real gap of 3.5%. Building upon this fine-grained representation, we develop a 3D tactile-language-action (3D-TLA) architecture driven by a flow matching policy to enable multimodal reasoning and control. Extensive experiments demonstrate that our framework significantly outperforms strong baselines in contact-rich manipulation tasks, providing a robust and generalizable foundation for tactile-language-action models.
>
---
#### [new 049] Design of a Robot-Assisted Chemical Dialysis System
- **分类: cs.RO**

- **简介: 该论文属于机器人辅助实验系统设计任务，旨在解决人工实验操作繁琐的问题。通过用户研究优化系统设计，提升实验效率，减轻科研人员负担。**

- **链接: [https://arxiv.org/pdf/2603.10264](https://arxiv.org/pdf/2603.10264)**

> **作者:** Diane Jung; Caleb Escobedo; Noah Liska; Maitrey Gramopadhye; Daniel Szafir; Alessandro Roncone; Carson Bruns
>
> **备注:** Accepted at ACM/IEEE International Conference on Human-Robot Interaction (HRI'26), Late Breaking Reports 5 pages, 2 figures
>
> **摘要:** Scientists perform diverse manual procedures that are tedious and laborious. Such procedures are considered a bottleneck for modern experimental science, as they consume time and increase burdens in fields including material science and medicine. We employ a user-centered approach to designing a robot-assisted system for dialysis, a common multi-day purification method used in polymer and protein synthesis. Through two usability studies, we obtain participant feedback and revise design requirements to develop the final system that satisfies scientists' needs and has the potential for applications in other experimental workflows. We anticipate that integration of this system into real synthesis procedures in a chemical wet lab will decrease workload on scientists during long experimental procedures and provide an effective approach to designing more systems that have the potential to accelerate scientific discovery and liberate scientists from tedious labor.
>
---
#### [new 050] Rethinking Gaussian Trajectory Predictors: Calibrated Uncertainty for Safe Planning
- **分类: cs.RO**

- **简介: 该论文属于轨迹预测任务，解决Gaussian预测器不确定性校准问题，通过新损失函数提升预测可靠性，增强安全规划性能。**

- **链接: [https://arxiv.org/pdf/2603.10407](https://arxiv.org/pdf/2603.10407)**

> **作者:** Fatemeh Cheraghi Pouria; Mahsa Golchoubian; Katherine Driggs-Campbell
>
> **摘要:** Accurate trajectory prediction is critical for safe autonomous navigation in crowded environments. While many trajectory predictors output Gaussian distributions to represent the multi-modal distribution over future pedestrian positions, the reliability of their confidence levels often remains unaddressed. This limitation can lead to unsafe or overly conservative motion planning when the predictor is integrated with an uncertainty-aware planner. Existing Gaussian trajectory predictors primarily rely on the Negative Log-Likelihood loss, which is prone to predict over- or under-confident distributions, and may compromise downstream planner safety. This paper introduces a novel loss function for calibrating prediction uncertainty which leverages Kernel Density Estimation to estimate the empirical distribution of confidence levels. The proposed formulation enforces consistency with the properties of a Gaussian assumption by explicitly matching the estimated empirical distribution to the Chi-squared distribution. To ensure accurate mean prediction, a Mean Squared Error term is also incorporated in the final loss formulation. Experimental results on real-world trajectory datasets show that our method significantly improves the reliability of confidence levels predicted by different State-Of-The-Art Gaussian trajectory predictors. We also demonstrate the importance of providing planners with reliable probabilistic insights (i.e. calibrated confidence levels) for collision-free navigation in complex scenarios. For this purpose, we integrate Gaussian trajectory predictors trained with our loss function with an uncertainty-aware Model Predictive Control on scenarios extracted from real-world datasets, achieving improved planning performance through calibrated confidence levels.
>
---
#### [new 051] Characterizing Healthy & Post-Stroke Neuromotor Behavior During 6D Upper-Limb Isometric Gaming: Implications for Design of End-Effector Rehabilitation Robot Interfaces
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究康复机器人界面设计，解决如何评估健康与中风后神经运动行为的问题。通过分析力数据和肌电信号，提出新的分类方法，为个性化康复系统设计提供依据。**

- **链接: [https://arxiv.org/pdf/2603.10173](https://arxiv.org/pdf/2603.10173)**

> **作者:** Ajay Anand; Gabriel Parra; Chad A. Berghoff; Laura A. Hallock
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Successful robot-mediated rehabilitation requires designing games and robot interventions that promote healthy motor practice. However, the interplay between a given user's neuromotor behavior, the gaming interface, and the physical robot makes designing system elements -- and even characterizing what behaviors are "healthy" or pathological -- challenging. We leverage our OpenRobotRehab 1.0 open access data set to assess the characteristics of 13 healthy and 2 post-stroke users' force output, muscle activations, and game performance while executing isometric trajectory tracking tasks using an end-effector rehabilitation robot. We present an assessment of how subtle aspects of interface design impact user behavior; an analysis of how pathological neuromotor behaviors are reflected in end-effector force dynamics; and a novel hidden Markov model (HMM)-based neuromotor behavior classification method based on surface electromyography (sEMG) signals during cyclic motions. We demonstrate that task specification (including which axes are constrained and how users interpret tracking instructions) shapes user behavior; that pathology-related features are detectable in 6D end-effector force data during isometric task execution (with significant differences between healthy and post-stroke profiles in force error and average force production at $p=0.05$); and that healthy neuromotor strategies are heterogeneous and inherently difficult to characterize. We also show that our HMM-based models discriminate healthy and post-stroke neuromotor dynamics where synergy-based decompositions reflect no such differentiation. Lastly, we discuss these results' implications for the design of adaptive end-effector rehabilitation robots capable of promoting healthier movement strategies across diverse user populations.
>
---
#### [new 052] Contact Coverage-Guided Exploration for General-Purpose Dexterous Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于具身操作任务，解决缺乏通用奖励机制的问题。提出CCGE方法，通过接触覆盖引导探索，提升操作效率与成功率。**

- **链接: [https://arxiv.org/pdf/2603.10971](https://arxiv.org/pdf/2603.10971)**

> **作者:** Zixuan Liu; Ruoyi Qiao; Chenrui Tie; Xuanwei Liu; Yunfan Lou; Chongkai Gao; Zhixuan Xu; Lin Shao
>
> **备注:** 16 pages
>
> **摘要:** Deep Reinforcement learning (DRL) has achieved remarkable success in domains with well-defined reward structures, such as Atari games and locomotion. In contrast, dexterous manipulation lacks general-purpose reward formulations and typically depends on task-specific, handcrafted priors to guide hand-object interactions. We propose Contact Coverage-Guided Exploration (CCGE), a general exploration method designed for general-purpose dexterous manipulation tasks. CCGE represents contact state as the intersection between object surface points and predefined hand keypoints, encouraging dexterous hands to discover diverse and novel contact patterns, namely which fingers contact which object regions. It maintains a contact counter conditioned on discretized object states obtained via learned hash codes, capturing how frequently each finger interacts with different object regions. This counter is leveraged in two complementary ways: (1) to assign a count-based contact coverage reward that promotes exploration of novel contact patterns, and (2) an energy-based reaching reward that guides the agent toward under-explored contact regions. We evaluate CCGE on a diverse set of dexterous manipulation tasks, including cluttered object singulation, constrained object retrieval, in-hand reorientation, and bimanual manipulation. Experimental results show that CCGE substantially improves training efficiency and success rates over existing exploration methods, and that the contact patterns learned with CCGE transfer robustly to real-world robotic systems. Project page is this https URL.
>
---
#### [new 053] Dance2Hesitate: A Multi-Modal Dataset of Dancer-Taught Hesitancy for Understandable Robot Motion
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机协作任务，旨在解决机器人犹豫动作难以通用的问题。通过构建多模态数据集，记录不同犹豫程度的机器人与人类动作，以支持可理解的机器人运动研究。**

- **链接: [https://arxiv.org/pdf/2603.10166](https://arxiv.org/pdf/2603.10166)**

> **作者:** Srikrishna Bangalore Raghu; Anna Soukhovei; Divya Sai Sindhuja Vankineni; Alexandra Bacula; Alessandro Roncone
>
> **备注:** Accepted to the Designing Transparent and Understandable Robots (D-TUR) Workshop at the ACM/IEEE International Conference on Human-Robot Interaction (HRI) 2026, Edinburgh, UK
>
> **摘要:** In human-robot collaboration, a robot's expression of hesitancy is a critical factor that shapes human coordination strategies, attention allocation, and safety-related judgments. However, designing hesitant robot motion that generalizes is challenging because the observer's inference is highly dependent on embodiment and context. To address these challenges, we introduce and open-source a multi-modal, dancer-generated dataset of hesitant motion where we focus on specific context-embodiment pairs (i.e., manipulator/human upper-limb approaching a Jenga Tower, and anthropomorphic whole body motion in free space). The dataset includes (i) kinesthetic teaching demonstrations on a Franka Emika Panda reaching from a fixed start configuration to a fixed target (a Jenga tower) with three graded hesitancy levels (slight, significant, extreme) and (ii) synchronized RGB-D motion capture of dancers performing the same reaching behavior using their upper limb across three hesitancy levels, plus full human body sequences for extreme hesitancy. We further provide documentation to enable reproducible benchmarking across robot and human modalities. Across all dancers, we obtained 70 unique whole-body trajectories, 84 upper limb trajectories spanning over the three hesitancy levels, and 66 kinesthetic teaching trajectories spanning over the three hesitancy levels. The dataset can be accessed here: this https URL.
>
---
#### [new 054] Perceptive Hierarchical-Task MPC for Sequential Mobile Manipulation in Unstructured Semi-Static Environments
- **分类: cs.RO**

- **简介: 该论文属于移动操作任务，解决长期操作中环境变化带来的挑战。提出一种感知分层任务模型预测控制框架，实现高效顺序操作。**

- **链接: [https://arxiv.org/pdf/2603.10227](https://arxiv.org/pdf/2603.10227)**

> **作者:** Xintong Du; Jingxing Qian; Siqi Zhou; Angela P. Schoellig
>
> **摘要:** As compared to typical mobile manipulation tasks, sequential mobile manipulation poses a unique challenge -- as the robot operates over extended periods, successful task completion is not solely dependent on consistent motion generation but also on the robot's awareness and adaptivity to changes in the operating environment. While existing motion planners can generate whole-body trajectories to complete sequential tasks, they typically assume that the environment remains static and rely on precomputed maps. This assumption often breaks down during long-term operations, where semi-static changes such as object removal, introduction, or shifts are common. In this work, we propose a novel perceptive hierarchical-task model predictive control (HTMPC) framework for efficient sequential mobile manipulation in unstructured, changing environments. To tackle the challenge, we leverage a Bayesian inference framework to explicitly model object-level changes and thereby maintain a temporally accurate representation of the 3D environment; this up-to-date representation is embedded in a lexicographic optimization framework to enable efficient execution of sequential tasks. We validate our perceptive HTMPC approach through both simulated and real-robot experiments. In contrast to baseline methods, our approach systematically accounts for moved and phantom obstacles, successfully completing sequential tasks with higher efficiency and reactivity, without relying on prior maps or external infrastructure.
>
---
#### [new 055] Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于连续模仿学习任务，解决多任务下策略持续优化问题。通过多模态潜在空间存储信息，并引入增量调整机制，提升学习稳定性与任务区分性。**

- **链接: [https://arxiv.org/pdf/2603.10929](https://arxiv.org/pdf/2603.10929)**

> **作者:** Fanqi Yu; Matteo Tiezzi; Tommaso Apicella; Cigdem Beyan; Vittorio Murino
>
> **摘要:** We introduce a lifelong imitation learning framework that enables continual policy refinement across sequential tasks under realistic memory and data constraints. Our approach departs from conventional experience replay by operating entirely in a multimodal latent space, where compact representations of visual, linguistic, and robot's state information are stored and reused to support future learning. To further stabilize adaptation, we introduce an incremental feature adjustment mechanism that regularizes the evolution of task embeddings through an angular margin constraint, preserving inter-task distinctiveness. Our method establishes a new state of the art in the LIBERO benchmarks, achieving 10-17 point gains in AUC and up to 65% less forgetting compared to previous leading methods. Ablation studies confirm the effectiveness of each component, showing consistent gains over alternative strategies. The code is available at: this https URL.
>
---
#### [new 056] Robotic Ultrasound Makes CBCT Alive
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于医学影像任务，旨在解决CBCT静态图像无法实时反映软组织变形的问题。通过结合机器人超声与深度学习，实现动态更新CBCT图像。**

- **链接: [https://arxiv.org/pdf/2603.10220](https://arxiv.org/pdf/2603.10220)**

> **作者:** Feng Li; Ziyuan Li; Zhongliang Jiang; Nassir Navab; Yuan Bi
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Intraoperative Cone Beam Computed Tomography (CBCT) provides a reliable 3D anatomical context essential for interventional planning. However, its static nature fails to provide continuous monitoring of soft-tissue deformations induced by respiration, probe pressure, and surgical manipulation, leading to navigation discrepancies. We propose a deformation-aware CBCT updating framework that leverages robotic ultrasound as a dynamic proxy to infer tissue motion and update static CBCT slices in real time. Starting from calibration-initialized alignment with linear correlation of linear combination (LC2)-based rigid refinement, our method establishes accurate multimodal correspondence. To capture intraoperative dynamics, we introduce the ultrasound correlation UNet (USCorUNet), a lightweight network trained with optical flow-guided supervision to learn deformation-aware correlation representations, enabling accurate, real-time dense deformation field estimation from ultrasound streams. The inferred deformation is spatially regularized and transferred to the CBCT reference to produce deformation-consistent visualizations without repeated radiation exposure. We validate the proposed approach through deformation estimation and ultrasound-guided CBCT updating experiments. Results demonstrate real-time end-to-end CBCT slice updating and physically plausible deformation estimation, enabling dynamic refinement of static CBCT guidance during robotic ultrasound-assisted interventions. The source code is publicly available at this https URL.
>
---
#### [new 057] DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DynVLA模型，解决自动驾驶中的决策问题。通过引入Dynamics CoT，预测世界动态以提升决策质量。**

- **链接: [https://arxiv.org/pdf/2603.11041](https://arxiv.org/pdf/2603.11041)**

> **作者:** Shuyao Shang; Bing Zhan; Yunfei Yan; Yuqi Wang; Yingyan Li; Yasong An; Xiaoman Wang; Jierui Liu; Lu Hou; Lue Fan; Zhaoxiang Zhang; Tieniu Tan
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** We propose DynVLA, a driving VLA model that introduces a new CoT paradigm termed Dynamics CoT. DynVLA forecasts compact world dynamics before action generation, enabling more informed and physically grounded decision-making. To obtain compact dynamics representations, DynVLA introduces a Dynamics Tokenizer that compresses future evolution into a small set of dynamics tokens. Considering the rich environment dynamics in interaction-intensive driving scenarios, DynVLA decouples ego-centric and environment-centric dynamics, yielding more accurate world dynamics modeling. We then train DynVLA to generate dynamics tokens before actions through SFT and RFT, improving decision quality while maintaining latency-efficient inference. Compared to Textual CoT, which lacks fine-grained spatiotemporal understanding, and Visual CoT, which introduces substantial redundancy due to dense image prediction, Dynamics CoT captures the evolution of the world in a compact, interpretable, and efficient form. Extensive experiments on NAVSIM, Bench2Drive, and a large-scale in-house dataset demonstrate that DynVLA consistently outperforms Textual CoT and Visual CoT methods, validating the effectiveness and practical value of Dynamics CoT.
>
---
#### [new 058] Muscle Synergy Priors Enhance Biomechanical Fidelity in Predictive Musculoskeletal Locomotion Simulation
- **分类: cs.LG; cs.NE; cs.RO**

- **简介: 该论文属于人体运动模拟任务，旨在提升预测性骨骼肌运动模拟的生物力学精度。通过引入肌肉协同机制，约束强化学习控制，改善了运动仿真效果。**

- **链接: [https://arxiv.org/pdf/2603.10474](https://arxiv.org/pdf/2603.10474)**

> **作者:** Ilseung Park; Eunsik Choi; Jangwhan Ahn; Jooeun Ahn
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Human locomotion emerges from high-dimensional neuromuscular control, making predictive musculoskeletal simulation challenging. We present a physiology-informed reinforcement-learning framework that constrains control using muscle synergies. We extracted a low-dimensional synergy basis from inverse musculoskeletal analyses of a small set of overground walking trials and used it as the action space for a muscle-driven three-dimensional model trained across variable speeds, slopes and uneven terrain. The resulting controller generated stable gait from 0.7-1.8 m/s and on $\pm$ 6$^{\circ}$ grades and reproduced condition-dependent modulation of joint angles, joint moments and ground reaction forces. Compared with an unconstrained controller, synergy-constrained control reduced non-physiological knee kinematics and kept knee moment profiles within the experimental envelope. Across conditions, simulated vertical ground reaction forces correlated strongly with human measurements, and muscle-activation timing largely fell within inter-subject variability. These results show that embedding neurophysiological structure into reinforcement learning can improve biomechanical fidelity and generalization in predictive human locomotion simulation with limited experimental data.
>
---
#### [new 059] STADA: Specification-based Testing for Autonomous Driving Agents
- **分类: cs.SE; cs.RO**

- **简介: 该论文提出STADA，一种基于规范的自动驾驶测试框架，解决传统测试方法在覆盖安全要求上的不足。通过形式化规范生成测试场景，提升测试覆盖率。**

- **链接: [https://arxiv.org/pdf/2603.10940](https://arxiv.org/pdf/2603.10940)**

> **作者:** Joy Saha; Trey Woodlief; Sebastian Elbaum; Matthew B. Dwyer
>
> **摘要:** Simulation-based testing has become a standard approach to validating autonomous driving agents prior to real-world deployment. A high-quality validation campaign will exercise an agent in diverse contexts comprised of varying static environments, e.g., lanes, intersections, signage, and dynamic elements, e.g., vehicles and pedestrians. To achieve this, existing test generation techniques rely on template-based, manually constructed, or random scenario generation. When applied to validate formally specified safety requirements, such methods either require significant human effort or run the risk of missing important behavior related to the requirement. To address this gap, we present STADA, a Specification-based Test generation framework for Autonomous Driving Agents that systematically generates the space of scenarios defined by a formal specification expressed in temporal logic (LTLf). Given a specification, STADA constructs all distinct initial scenes, a diverse space of continuations of those scenes, and simulations that reflect the behaviors of the specification. Evaluation of STADA on a variety of LTLf specifications formalized in SCENEFLOW using three complementary coverage criteria demonstrates that STADA yields more than 2x higher coverage than the best baseline on the finest criteria and a 75% increase for the coarsest criteria. Moreover, it matches the coverage of the best baseline with 6 times fewer simulations. While set in the context of autonomous driving, the approach is applicable to other domains with rich simulation environments.
>
---
#### [new 060] Need for Speed: Zero-Shot Depth Completion with Single-Step Diffusion
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Marigold-SSD，解决深度补全任务中的效率问题。通过单步扩散方法，提升推理速度并减少训练成本，实现高效且泛化的3D感知。**

- **链接: [https://arxiv.org/pdf/2603.10584](https://arxiv.org/pdf/2603.10584)**

> **作者:** Jakub Gregorek; Paraskevas Pegios; Nando Metzger; Konrad Schindler; Theodora Kontogianni; Lazaros Nalpantidis
>
> **摘要:** We introduce Marigold-SSD, a single-step, late-fusion depth completion framework that leverages strong diffusion priors while eliminating the costly test-time optimization typically associated with diffusion-based methods. By shifting computational burden from inference to finetuning, our approach enables efficient and robust 3D perception under real-world latency constraints. Marigold-SSD achieves significantly faster inference with a training cost of only 4.5 GPU days. We evaluate our method across four indoor and two outdoor benchmarks, demonstrating strong cross-domain generalization and zero-shot performance compared to existing depth completion approaches. Our approach significantly narrows the efficiency gap between diffusion-based and discriminative models. Finally, we challenge common evaluation protocols by analyzing performance under varying input sparsity levels. Page: this https URL
>
---
#### [new 061] Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation
- **分类: cs.CV; cs.AI; cs.RO; eess.SY**

- **简介: 该论文属于视觉-语言-动作模型任务，解决 clutter 环境下的精度-推理差距问题。提出 CGVD 方法，通过概念门控和傅里叶修复提升操作准确性。**

- **链接: [https://arxiv.org/pdf/2603.10340](https://arxiv.org/pdf/2603.10340)**

> **作者:** Sangmim Song; Sarath Kodagoda; Marc Carmichael; Karthick Thiyagarajan
>
> **备注:** 7 pages, 4 figures, 3 tables
>
> **摘要:** Vision-Language-Action (VLA) models demonstrate impressive zero-shot generalization but frequently suffer from a "Precision-Reasoning Gap" in cluttered environments. This failure is driven by background-induced feature dilution, where high-frequency semantic noise corrupts the geometric grounding required for precise manipulation. To bridge this gap, we propose Concept-Gated Visual Distillation (CGVD), a training-free, model-agnostic inference framework that stabilizes VLA policies. CGVD operates by parsing instructions into safe and distractor sets, utilizing a two-layer target refinement process--combining cross-validation and spatial disambiguation--to explicitly penalize false positives and isolate genuine manipulation targets. We then process the scene via Fourier-based inpainting, generating a clean observation that actively suppresses semantic distractors while preserving critical spatial geometry and visual proprioception. Extensive evaluations in highly cluttered manipulation tasks demonstrate that CGVD prevents performance collapse. In environments with dense semantic distractors, our method significantly outperforms state-of-the-art baselines, achieving a 77.5% success rate compared to the baseline's 43.0%. By enforcing strict attribute adherence, CGVD establishes inference-time visual distillation as a critical prerequisite for robust robotic manipulation in the clutter.
>
---
#### [new 062] Sublinear-Time Reconfiguration of Programmable Matter with Joint Movements
- **分类: cs.DS; cs.CG; cs.RO**

- **简介: 该论文研究可编程物质的快速重配置问题，解决几何 amoebot 结构在不依赖辅助假设下的子线性时间重配置。通过集中式算法实现结构到线段的高效转换。**

- **链接: [https://arxiv.org/pdf/2603.10720](https://arxiv.org/pdf/2603.10720)**

> **作者:** Manish Kumar; Othon Michail; Andreas Padalkin; Christian Scheideler
>
> **摘要:** We study centralized reconfiguration problems for geometric amoebot structures. A set of $n$ amoebots occupy nodes on the triangular grid and can reconfigure via expansion and contraction operations. We focus on the joint movement extension, where amoebots may expand and contract in parallel, enabling coordinated motion of larger substructures. Prior work introduced this extension and analyzed reconfiguration under additional assumptions such as metamodules. In contrast, we investigate the intrinsic dynamics of reconfiguration without such assumptions by restricting attention to centralized algorithms, leaving distributed solutions for future work. We study the reconfiguration problem between two classes of amoebot structures $A$ and $B$: For every structure $S\in A$, the goal is to compute a schedule that reconfigures $S$ into some structure $S'\in B$. Our focus is on sublinear-time algorithms. We affirmatively answer the open problem by Padalkin et al. (Auton. Robots, 2025) whether a within-the-model sublinear-time universal reconfiguration algorithm is possible, by proving that any structure can be reconfigured into a canonical line-segment structure in $O(\sqrt{n}\log n)$ rounds. Additionally, we give a constant-time algorithm for reconfiguring any spiral structure into a line segment. These results are enabled by new constant-time primitives that facilitate efficient parallel movement. Our findings demonstrate that the joint movement model supports sublinear reconfiguration without auxiliary assumptions. A central open question is whether universal reconfiguration within this model can be achieved in polylogarithmic or even constant time.
>
---
## 更新

#### [replaced 001] Open-World Task and Motion Planning via Vision-Language Model Generated Constraints
- **分类: cs.RO**

- **简介: 该论文提出OWL-TAMP，将视觉语言模型融入任务与运动规划系统，解决长时序机器人操作中自然语言目标理解与开放世界推理问题。**

- **链接: [https://arxiv.org/pdf/2411.08253](https://arxiv.org/pdf/2411.08253)**

> **作者:** Nishanth Kumar; William Shen; Fabio Ramos; Dieter Fox; Tomás Lozano-Pérez; Leslie Pack Kaelbling; Caelan Reed Garrett
>
> **备注:** A version of this paper appears in IEEE Robotics and Automation Letters (RA-L) Volume 11, Issue 3
>
> **摘要:** Foundation models like Vision-Language Models (VLMs) excel at common sense vision and language tasks such as visual question answering. However, they cannot yet directly solve complex, long-horizon robot manipulation problems requiring precise continuous reasoning. Task and Motion Planning (TAMP) systems can handle long-horizon reasoning through discrete-continuous hybrid search over parameterized skills, but rely on detailed environment models and cannot interpret novel human objectives, such as arbitrary natural language goals. We propose integrating VLMs into TAMP systems by having them generate discrete and continuous language-parameterized constraints that enable open-world reasoning. Specifically, we use VLMs to generate discrete action ordering constraints that constrain TAMP search over action sequences, and continuous constraints in the form of code that augments traditional TAMP manipulation constraints. Experiments show that our approach, OWL-TAMP, outperforms baselines relying solely on TAMP or VLMs across several long-horizon manipulation tasks specified directly in natural language. We additionally demonstrate that OWL-TAMP can be deployed with an off-the-shelf TAMP system to solve challenging manipulation tasks on real-world hardware.
>
---
#### [replaced 002] CompassNav: Steering From Path Imitation To Decision Understanding In Navigation
- **分类: cs.RO**

- **简介: 该论文属于导航任务，旨在解决传统路径模仿方法限制智能体理解能力的问题。通过引入新数据集和奖励函数，提升智能体的决策理解与探索能力。**

- **链接: [https://arxiv.org/pdf/2510.10154](https://arxiv.org/pdf/2510.10154)**

> **作者:** LinFeng Li; Jian Zhao; Yuan Xie; Xin Tan; Xuelong Li
>
> **摘要:** The dominant paradigm for training Large Vision-Language Models (LVLMs) in navigation relies on imitating expert trajectories. This approach reduces the complex navigation task to a sequence-to-sequence replication of a single correct path, fundamentally limiting the agent's ability to explore and generalize. In this work, we argue for and introduce a new paradigm: a shift from Path Imitation to Decision Understanding. The goal of this paradigm is to build agents that do not just follow, but truly understand how to navigate. We materialize this through two core contributions: first, we introduce Compass-Data-22k, a novel 22k-trajectory dataset. Its Reinforcement Fine-Tuning (RFT) subset provides a panoramic view of the decision landscape by annotating all feasible actions with A* geodesic distances. Second, we design a novel gap-aware hybrid reward function that dynamically adapts its feedback to decision certainty, shifting between decisive signals for optimal actions and nuanced scores to encourage exploration. Integrated into an SFT-then-RFT recipe, our CompassNav agent is trained not to memorize static routes, but to develop an internal compass that constantly intuits the direction to the goal by evaluating the relative quality of all possible moves. This approach enables our 7B agent to set a new state-of-the-art on Goal navigation benchmarks, outperforming even larger proprietary models, and achieve robust real-world goal navigation on a physical robot.
>
---
#### [replaced 003] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于机器人任务规划领域，解决模糊人类指令影响规划性能的问题。通过构建REI-Bench基准并提出上下文认知方法，提升非专家用户指令的处理效果。**

- **链接: [https://arxiv.org/pdf/2505.10872](https://arxiv.org/pdf/2505.10872)**

> **作者:** Chenxi Jiang; Chuhao Zhou; Jianfei Yang
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Robot task planning decomposes human instructions into executable action sequences that enable robots to complete a series of complex tasks. Although recent large language model (LLM)-based task planners achieve amazing performance, they assume that human instructions are clear and straightforward. However, real-world users are not experts, and their instructions to robots often contain significant vagueness. Linguists suggest that such vagueness frequently arises from referring expressions (REs), whose meanings depend heavily on dialogue context and environment. This vagueness is even more prevalent among the elderly and children, who are the groups that robots should serve more. This paper studies how such vagueness in REs within human instructions affects LLM-based robot task planning and how to overcome this issue. To this end, we propose the first robot task planning benchmark that systematically models vague REs grounded in pragmatic theory (REI-Bench), where we discover that the vagueness of REs can severely degrade robot planning performance, leading to success rate drops of up to 36.9%. We also observe that most failure cases stem from missing objects in planners. To mitigate the REs issue, we propose a simple yet effective approach: task-oriented context cognition, which generates clear instructions for robots, achieving state-of-the-art performance compared to aware prompts, chains of thought, and in-context learning. By tackling the overlooked issue of vagueness, this work contributes to the research community by advancing real-world task planning and making robots more accessible to non-expert users, e.g., the elderly and children.
>
---
#### [replaced 004] Score Matching Diffusion Based Feedback Control and Planning of Nonlinear Systems
- **分类: math.OC; cs.LG; cs.RO; eess.SY**

- **简介: 该论文属于非线性系统控制任务，解决如何通过概率密度控制实现系统状态的精确引导。工作包括提出基于扩散的确定性反馈控制框架，设计反向过程以实现目标分布的跟踪。**

- **链接: [https://arxiv.org/pdf/2504.09836](https://arxiv.org/pdf/2504.09836)**

> **作者:** Karthik Elamvazhuthi; Darshan Gadginmath; Fabio Pasqualetti
>
> **摘要:** In this paper, we propose a deterministic diffusion-based framework for controlling the probability density of nonlinear control-affine systems, with theoretical guarantees for drift-free and linear time-invariant (LTI) dynamics. The central idea is to first excite the system with white noise so that a forward diffusion process explores the reachable regions of state space, and then to design a deterministic feedback law that acts as a denoising mechanism driving the system back toward a desired target distribution supported on the target set. This denoising phase provides a feedback controller that steers the control system to the target set. In this framework, control synthesis reduces to constructing a deterministic reverse process that reproduces the desired evolution of state densities. We derive existence conditions ensuring such deterministic realizations of time-reversals for controllable drift-free and LTI systems, and show that the resulting feedback laws provide a tractable alternative to nonlinear control by viewing density control as a relaxation of controlling a system to target sets. Numerical studies on a unicycle model with obstacles, a five-dimensional driftless system, and a four-dimensional LTI system demonstrate reliable diffusion-inspired density control.
>
---
#### [replaced 005] Scalable Multi-Task Learning through Spiking Neural Networks with Adaptive Task-Switching Policy for Intelligent Autonomous Agents
- **分类: cs.NE; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于多任务学习领域，解决自主代理在资源受限下的多任务协同问题。提出SwitchMT方法，通过自适应任务切换策略提升性能与可扩展性。**

- **链接: [https://arxiv.org/pdf/2504.13541](https://arxiv.org/pdf/2504.13541)**

> **作者:** Rachmad Vidya Wicaksana Putra; Avaneesh Devkota; Muhammad Shafique
>
> **备注:** Accepted at the 63rd ACM/IEEE Design Automation Conference (DAC), July 26-29, 2026 in Long Beach, CA, USA Codes: this https URL
>
> **摘要:** Training resource-constrained autonomous agents on multiple tasks simultaneously is crucial for adapting to diverse real-world environments. Recent works employ reinforcement learning (RL) approach, but they still suffer from sub-optimal multi-task performance due to task interference. State-of-the-art works employ Spiking Neural Networks (SNNs) to improve RL-based multi-task learning and enable low-power/energy operations through network enhancements and spike-driven data stream processing. However, they rely on fixed task-switching intervals during its training, thus limiting its performance and scalability. To address this, we propose SwitchMT, a novel methodology that employs adaptive task-switching for effective, scalable, and simultaneous multi-task learning. SwitchMT employs the following key ideas: (1) leveraging a Deep Spiking Q-Network with active dendrites and dueling structure, that utilizes task-specific context signals to create specialized sub-networks; and (2) devising an adaptive task-switching policy that leverages both rewards and internal dynamics of the network parameters. Experimental results demonstrate that SwitchMT achieves competitive scores in multiple Atari games (i.e., Pong: -8.8, Breakout: 5.6, and Enduro: 355.2) and longer game episodes as compared to the state-of-the-art. These results also highlight the effectiveness of SwitchMT methodology in addressing task interference without increasing the network complexity, enabling intelligent autonomous agents with scalable multi-task learning capabilities.
>
---
#### [replaced 006] Safe and Optimal Learning from Preferences via Weighted Temporal Logic with Applications in Robotics and Formula 1
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于偏好学习任务，旨在安全地从人类反馈中学习最优行为。通过WSTL方法解决多线性约束问题，转化为MILP求解，确保安全性与效率。**

- **链接: [https://arxiv.org/pdf/2511.08502](https://arxiv.org/pdf/2511.08502)**

> **作者:** Ruya Karagulle; Cristian-Ioan Vasile; Necmiye Ozay
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** Autonomous systems increasingly rely on human feedback to align their behavior, expressed as pairwise comparisons, rankings, or demonstrations. While existing methods can adapt behaviors, they often fail to guarantee safety in safety-critical domains. We propose a safety-guaranteed, optimal, and efficient approach for solving the learning problem from preferences, rankings, or demonstrations using Weighted Signal Temporal Logic (WSTL). WSTL learning problems, when implemented naively, lead to multi-linear constraints in the weights to be learned. By introducing structural pruning and log-transform procedures, we reduce the problem size and recast it as a Mixed-Integer Linear Program while preserving safety guarantees. Experiments on robotic navigation and real-world Formula 1 data demonstrate that the method captures nuanced preferences and models complex task objectives.
>
---
#### [replaced 007] Cross-embodied Co-design for Dexterous Hands
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人操控任务，旨在解决手部设计与控制协同优化问题。通过共设计框架，同时学习手部形态和控制策略，实现高效 Dexterous 操作。**

- **链接: [https://arxiv.org/pdf/2512.03743](https://arxiv.org/pdf/2512.03743)**

> **作者:** Kehlani Fay; Darin Anthony Djapri; Anya Zorin; James Clinton; Ali El Lahib; Hao Su; Michael T. Tolley; Sha Yi; Xiaolong Wang
>
> **摘要:** Dexterous manipulation is limited by both control and design, without consensus as to what makes manipulators best for performing dexterous tasks. This raises a fundamental challenge: how should we design and control robot manipulators that are optimized for dexterity? We present a co-design framework that learns task-specific hand morphology and complementary dexterous control policies. The framework supports 1) an expansive morphology search space including joint, finger, and palm generation, 2) scalable evaluation across the wide design space via morphology-conditioned cross-embodied control, and 3) real-world fabrication with accessible components. We evaluate the approach across multiple dexterous tasks, including in-hand rotation with simulation and real deployment. Our framework enables an end-to-end pipeline that can design, train, fabricate, and deploy a new robotic hand in under 24 hours. The full framework will be open-sourced and available on our website: this https URL .
>
---
#### [replaced 008] PlayWorld: Learning Robot World Models from Autonomous Play
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出PlayWorld，用于训练高保真机器人世界模型。任务是提升机器人操作中的物理一致性预测。通过自主玩耍数据训练，解决传统方法依赖人类示范的不足。**

- **链接: [https://arxiv.org/pdf/2603.09030](https://arxiv.org/pdf/2603.09030)**

> **作者:** Tenny Yin; Zhiting Mei; Zhonghe Zheng; Miyu Yamane; David Wang; Jade Sceats; Samuel M. Bateman; Lihan Zha; Apurva Badithela; Ola Shorinwa; Anirudha Majumdar
>
> **备注:** this https URL
>
> **摘要:** Action-conditioned video models offer a promising path to building general-purpose robot simulators that can improve directly from data. Yet, despite training on large-scale robot datasets, current state-of-the-art video models still struggle to predict physically consistent robot-object interactions that are crucial in robotic manipulation. To close this gap, we present PlayWorld, a simple, scalable, and fully autonomous pipeline for training high-fidelity video world simulators from interaction experience. In contrast to prior approaches that rely on success-biased human demonstrations, PlayWorld is the first system capable of learning entirely from unsupervised robot self-play, enabling naturally scalable data collection while capturing complex, long-tailed physical interactions essential for modeling realistic object dynamics. Experiments across diverse manipulation tasks show that PlayWorld generates high-quality, physically consistent predictions for contact-rich interactions that are not captured by world models trained on human-collected data. We further demonstrate the versatility of PlayWorld in enabling fine-grained failure prediction and policy evaluation, with up to 40% improvements over human-collected data. Finally, we demonstrate how PlayWorld enables reinforcement learning in the world model, improving policy performance by 65% in success rates when deployed in the real world.
>
---
#### [replaced 009] Moving On, Even When You're Broken: Fail-Active Trajectory Generation via Diffusion Policies Conditioned on Embodiment and Task
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机器人在故障情况下的自主操作问题，提出DEFT方法，通过扩散策略生成轨迹，实现故障时的任务完成。属于机器人运动规划任务。**

- **链接: [https://arxiv.org/pdf/2602.02895](https://arxiv.org/pdf/2602.02895)**

> **作者:** Gilberto G. Briscoe-Martinez; Yaashia Gautam; Rahul Shetty; Anuj Pasricha; Marco M. Nicotra; Alessandro Roncone
>
> **备注:** To be published in the 2026 IEEE International Conference on Robotics & Automation
>
> **摘要:** Robot failure is detrimental and disruptive, often requiring human intervention to recover. Our vision is 'fail-active' operation, allowing robots to safely complete their tasks even when damaged. Focusing on 'actuation failures', we introduce DEFT, a diffusion-based trajectory generator conditioned on the robot's current embodiment and task constraints. DEFT generalizes across failure types, supports constrained and unconstrained motions, and enables task completion under arbitrary failure. We evaluate DEFT in both simulation and real-world scenarios using a 7-DoF robotic arm. DEFT outperforms its baselines over thousands of failure conditions, achieving a 99.5% success rate for unconstrained motions versus RRT's 42.4%, and 46.4% for constrained motions versus differential IK's 30.9%. Furthermore, DEFT demonstrates robust zero-shot generalization by maintaining performance on failure conditions unseen during training. Finally, we perform real-world evaluations on two multi-step tasks, drawer manipulation and whiteboard erasing. These experiments demonstrate DEFT succeeding on tasks where classical methods fail. Our results show that DEFT achieves fail-active manipulation across arbitrary failure configurations and real-world deployments.
>
---
#### [replaced 010] SPAARS: Safer RL Policy Alignment through Abstract Exploration and Refined Exploitation of Action Space
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决离线到在线策略对齐中的安全探索问题。提出SPAARS框架，通过潜空间约束探索并逐步转移至原始动作空间，提升样本效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.09378](https://arxiv.org/pdf/2603.09378)**

> **作者:** Swaminathan S K; Aritra Hazra
>
> **备注:** 9 pages
>
> **摘要:** Offline-to-online reinforcement learning (RL) offers a promising paradigm for robotics by pre-training policies on safe, offline demonstrations and fine-tuning them via online interaction. However, a fundamental challenge remains: how to safely explore online without deviating from the behavioral support of the offline data? While recent methods leverage conditional variational autoencoders (CVAEs) to bound exploration within a latent space, they inherently suffer from an exploitation gap -- a performance ceiling imposed by the decoder's reconstruction loss. We introduce SPAARS, a curriculum learning framework that initially constrains exploration to the low-dimensional latent manifold for sample-efficient, safe behavioral improvement, then seamlessly transfers control to the raw action space, bypassing the decoder bottleneck. SPAARS has two instantiations: the CVAE-based variant requires only unordered (s,a) pairs and no trajectory segmentation; SPAARS-SUPE pairs SPAARS with OPAL temporal skill pretraining for stronger exploration structure at the cost of requiring trajectory chunks. We prove an upper bound on the exploitation gap using the Performance Difference Lemma, establish that latent-space policy gradients achieve provable variance reduction over raw-space exploration, and show that concurrent behavioral cloning during the latent phase directly controls curriculum transition stability. Empirically, SPAARS-SUPE achieves 0.825 normalized return on kitchen-mixed-v0 versus 0.75 for SUPE, with 5x better sample efficiency; standalone SPAARS achieves 92.7 and 102.9 normalized return on hopper-medium-v2 and walker2d-medium-v2 respectively, surpassing IQL baselines of 66.3 and 78.3 respectively, confirming the utility of the unordered-pair CVAE instantiation.
>
---
#### [replaced 011] vS-Graphs: Tightly Coupling Visual SLAM and 3D Scene Graphs Exploiting Hierarchical Scene Understanding
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，旨在解决传统系统地图语义不足的问题。通过融合视觉场景理解和3D场景图，提升地图的语义丰富性和可理解性。**

- **链接: [https://arxiv.org/pdf/2503.01783](https://arxiv.org/pdf/2503.01783)**

> **作者:** Ali Tourani; Saad Ejaz; Hriday Bavle; Miguel Fernandez-Cortizas; David Morilla-Cabello; Jose Luis Sanchez-Lopez; Holger Voos
>
> **备注:** 20 pages, 10 figures, 5 tables
>
> **摘要:** Current Visual Simultaneous Localization and Mapping (VSLAM) systems often struggle to create maps that are both semantically rich and easily interpretable. While incorporating semantic scene knowledge aids in building richer maps with contextual associations among mapped objects, representing them in structured formats, such as scene graphs, has not been widely addressed, resulting in complex map comprehension and limited scalability. This paper introduces vS-Graphs, a novel real-time VSLAM framework that integrates vision-based scene understanding with map reconstruction and comprehensible graph-based representation. The framework infers structural elements (i.e., rooms and floors) from detected building components (i.e., walls and ground surfaces) and incorporates them into optimizable 3D scene graphs. This solution enhances the reconstructed map's semantic richness, comprehensibility, and localization accuracy. Extensive experiments on standard benchmarks and real-world datasets demonstrate that vS-Graphs achieves an average of 15.22% accuracy gain across all tested datasets compared to state-of-the-art VSLAM methods. Furthermore, the proposed framework achieves environment-driven semantic entity detection accuracy comparable to that of precise LiDAR-based frameworks, using only visual features. The code is publicly available at this https URL and is actively being improved. Moreover, a web page containing more media and evaluation outcomes is available on this https URL.
>
---
#### [replaced 012] Dull, Dirty, Dangerous: Understanding the Past, Present, and Future of a Key Motivation for Robotics
- **分类: cs.RO**

- **简介: 该论文属于机器人学研究，旨在解决DDD概念在机器人应用中的模糊性。通过分析文献，提出框架以明确DDD定义和应用场景。**

- **链接: [https://arxiv.org/pdf/2602.04746](https://arxiv.org/pdf/2602.04746)**

> **作者:** Nozomi Nakajima; Pedro Reynolds-Cuéllar; Caitrin Lynch; Kate Darling
>
> **摘要:** In robotics, the concept of "dull, dirty, and dangerous" (DDD) work has been used to motivate where robots might be useful. In this paper, we conduct an empirical analysis of robotics publications between 1980 and 2024 that mention DDD, and find that only 2.7% of publications define DDD and 8.7% of publications provide concrete examples of tasks or jobs that are DDD. We then review the social science literature on "dull," "dirty," and "dangerous" work to provide definitions and guidance on how to conceptualize DDD for robotics. Finally, we propose a framework that helps the robotics community consider the job context for our technology, encouraging a more informed perspective on how robotics may impact human labor.
>
---
#### [replaced 013] Pixel Motion Diffusion is What We Need for Robot Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DAWN框架，用于机器人控制任务，解决如何将高层指令转化为低层动作的问题。通过扩散模型和像素运动表示，实现端到端学习与可靠现实迁移。**

- **链接: [https://arxiv.org/pdf/2509.22652](https://arxiv.org/pdf/2509.22652)**

> **作者:** E-Ro Nguyen; Yichi Zhang; Kanchana Ranasinghe; Xiang Li; Michael S. Ryoo
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** We present DAWN (Diffusion is All We Need for robot control), a unified diffusion-based framework for language-conditioned robotic manipulation that bridges high-level motion intent and low-level robot action via structured pixel motion representation. In DAWN, both the high-level and low-level controllers are modeled as diffusion processes, yielding a fully trainable, end-to-end system with interpretable intermediate motion abstractions. DAWN achieves state-of-the-art results on the challenging CALVIN benchmark, demonstrating strong multi-task performance, and further validates its effectiveness on MetaWorld. Despite the substantial domain gap between simulation and reality and limited real-world data, we demonstrate reliable real-world transfer with only minimal finetuning, illustrating the practical viability of diffusion-based motion abstractions for robotic control. Our results show the effectiveness of combining diffusion modeling with motion-centric representations as a strong baseline for scalable and robust robot learning. Project page: this https URL
>
---
#### [replaced 014] RACAS: Controlling Diverse Robots With a Single Agentic System
- **分类: cs.RO; cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出RACAS系统，解决跨平台机器人控制问题。通过自然语言交互的模块实现通用控制，无需修改代码或模型。**

- **链接: [https://arxiv.org/pdf/2603.05621](https://arxiv.org/pdf/2603.05621)**

> **作者:** Dylan R. Ashley; Jan Przepióra; Yimeng Chen; Ali Abualsaud; Nurzhan Yesmagambet; Shinkyu Park; Eric Feron; Jürgen Schmidhuber
>
> **备注:** 7 pages in main text + 1 page of appendices + 1 page of references, 5 figures in main text + 1 figure in appendices, 2 tables in main text; source code available at this https URL
>
> **摘要:** Many robotic platforms expose an API through which external software can command their actuators and read their sensors. However, transitioning from these low-level interfaces to high-level autonomous behaviour requires a complicated pipeline, whose components demand distinct areas of expertise. Existing approaches to bridging this gap either require retraining for every new embodiment or have only been validated across structurally similar platforms. We introduce RACAS (Robot-Agnostic Control via Agentic Systems), a cooperative agentic architecture in which three LLM/VLM-based modules (Monitors, a Controller, and a Memory Curator) communicate exclusively through natural language to provide closed-loop robot control. RACAS requires only a natural language description of the robot, a definition of available actions, and a task specification; no source code, model weights, or reward functions need to be modified to move between platforms. We evaluate RACAS on several tasks using a wheeled ground robot, a recently published novel multi-jointed robotic limb, and an underwater vehicle. RACAS consistently solved all assigned tasks across these radically different platforms, demonstrating the potential of agentic AI to substantially reduce the barrier to prototyping robotic solutions.
>
---
#### [replaced 015] A Chain-Driven, Sandwich-Legged Quadruped Robot: Design and Experimental Analysis
- **分类: cs.RO**

- **简介: 该论文属于机器人设计任务，旨在开发一种低成本、高可靠性的四足机器人。通过优化腿部结构和驱动方式，提升运动性能与安全性，实现稳定行走。**

- **链接: [https://arxiv.org/pdf/2503.14255](https://arxiv.org/pdf/2503.14255)**

> **作者:** Aman Singh; Bhavya Giri Goswami; Ketan Nehete; Shishir N. Y. Kolathaya
>
> **备注:** 6 pages, 9 figures
>
> **摘要:** This paper introduces a chain-driven, sandwich-legged mid-size quadruped robot designed as an accessible research platform. The design prioritizes enhanced locomotion, improved actuation reliability and safety, and simplified, cost-effective manufacturing. Locomotion performance is improved through a sandwiched leg architecture and dual-motor configuration, reducing leg inertia for agile motion. Reliability and safety are enhanced using robust cable strain reliefs, motor heat sinks for thermal management, and mechanical limits to restrict leg motion. The design incorporates quasi-direct-drive (QDD) actuators and low-cost fabrication methods such as laser cutting and 3D printing for rapid prototyping. The $25\,\mathrm{kg}$ robot is built under \$8000, providing an affordable quadruped research platform. Experiments demonstrate trot and crawl gaits on flat terrain and slopes. We also open-source the mechanical designs. VIDEO: this https URL CADs: this https URL
>
---
#### [replaced 016] Cosmos-H-Surgical: Learning Surgical Robot Policies from Videos via World Modeling
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于手术机器人领域，旨在解决数据稀缺问题。通过构建世界模型和合成数据，提升手术机器人自主学习能力。**

- **链接: [https://arxiv.org/pdf/2512.23162](https://arxiv.org/pdf/2512.23162)**

> **作者:** Yufan He; Pengfei Guo; Mengya Xu; Zhaoshuo Li; Andriy Myronenko; Dillan Imans; Bingjie Liu; Dongren Yang; Mingxue Gu; Yongnan Ji; Yueming Jin; Ren Zhao; Baiyong Shen; Daguang Xu
>
> **摘要:** Data scarcity remains a fundamental barrier to achieving fully autonomous surgical robots. While large scale vision language action (VLA) models have shown impressive generalization in household and industrial manipulation by leveraging paired video action data from diverse domains, surgical robotics suffers from the paucity of datasets that include both visual observations and accurate robot kinematics. In contrast, vast corpora of surgical videos exist, but they lack corresponding action labels, preventing direct application of imitation learning or VLA training. In this work, we aim to alleviate this problem by learning policy models from Cosmos-H-Surgical, a world model designed for surgical physical AI. We curated the Surgical Action Text Alignment (SATA) dataset with detailed action description specifically for surgical robots. Then we built Cosmos-H-Surgical based on the most advanced physical AI world model and SATA. It's able to generate diverse, generalizable and realistic surgery videos. We are also the first to use an inverse dynamics model to infer pseudokinematics from synthetic surgical videos, producing synthetic paired video action data. We demonstrate that a surgical VLA policy trained with these augmented data significantly outperforms models trained only on real demonstrations on a real surgical robot platform. Our approach offers a scalable path toward autonomous surgical skill acquisition by leveraging the abundance of unlabeled surgical video and generative world modeling, thus opening the door to generalizable and data efficient surgical robot policies.
>
---
#### [replaced 017] Global End-Effector Pose Control of an Underactuated Aerial Manipulator via Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文研究基于强化学习的无人机机械臂末端位姿控制任务，解决轻量化设计下的欠驱动与扰动敏感问题，通过PPO算法生成控制指令，实现高精度操控。**

- **链接: [https://arxiv.org/pdf/2512.21085](https://arxiv.org/pdf/2512.21085)**

> **作者:** Shlok Deshmukh; Javier Alonso-Mora; Sihao Sun
>
> **备注:** 8 pages, 6 figures, accepted by IEEE ICRA 2026
>
> **摘要:** Aerial manipulators, which combine robotic arms with multi-rotor drones, face strict constraints on arm weight and mechanical complexity. In this work, we study a lightweight 2-degree-of-freedom (DoF) arm mounted on a quadrotor via a differential mechanism, capable of full six-DoF end-effector pose control. While the minimal design enables simplicity and reduced payload, it also introduces challenges such as underactuation and sensitivity to external disturbances. To address these, we employ reinforcement learning, training a Proximal Policy Optimization (PPO) agent in simulation to generate feedforward commands for quadrotor acceleration and body rates, along with joint angle targets. These commands are tracked by an incremental nonlinear dynamic inversion (INDI) attitude controller and a PID joint controller, respectively. Flight experiments demonstrate centimeter-level position accuracy and degree-level orientation precision, with robust performance under external force disturbances, including manipulation of heavy loads and pushing tasks. The results highlight the potential of learning-based control strategies for enabling contact-rich aerial manipulation using simple, lightweight platforms. Videos of the experiment and the method are summarized in this https URL.
>
---
#### [replaced 018] Self-Improving Loops for Visual Robotic Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出SILVR方法，用于视觉机器人规划任务，解决未知任务泛化问题。通过自收集数据迭代优化模型，提升性能与样本效率。**

- **链接: [https://arxiv.org/pdf/2506.06658](https://arxiv.org/pdf/2506.06658)**

> **作者:** Calvin Luo; Zilai Zeng; Mingxi Jia; Yilun Du; Chen Sun
>
> **备注:** ICLR 2026. Project Page: this https URL
>
> **摘要:** Video generative models trained on expert demonstrations have been utilized as performant text-conditioned visual planners for solving robotic tasks. However, generalization to unseen tasks remains a challenge. Whereas improved generalization may be facilitated by leveraging learned prior knowledge from additional pre-collected offline data sources, such as web-scale video datasets, in the era of experience we aim to design agents that can continuously improve in an online manner from self-collected behaviors. In this work we thus propose the Self-Improving Loops for Visual Robotic Planning (SILVR), where an in-domain video model iteratively updates itself on self-produced trajectories, and steadily improves its performance for a specified task of interest. We apply SILVR to a diverse suite of MetaWorld tasks, as well as two manipulation tasks on a real robot arm, and find that performance improvements continuously emerge over multiple iterations for novel tasks unseen during initial in-domain video model training. We demonstrate that SILVR is robust in the absence of human-provided ground-truth reward functions or expert-quality demonstrations, and is preferable to alternate approaches that utilize online experience in terms of performance and sample efficiency.
>
---
#### [replaced 019] Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，解决对称性破坏环境下的价值估计误差问题。提出PI-MDP框架，结合对称性和鲁棒性，提升样本效率和泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.00915](https://arxiv.org/pdf/2512.00915)**

> **作者:** Junwoo Chang; Minwoo Park; Joohwan Seo; Roberto Horowitz; Jongmin Lee; Jongeun Choi
>
> **备注:** ICLR 2026
>
> **摘要:** Group symmetries provide a powerful inductive bias for reinforcement learning (RL), enabling efficient generalization across symmetric states and actions via group-invariant Markov Decision Processes (MDPs). However, real-world environments almost never realize fully group-invariant MDPs; dynamics, actuation limits, and reward design usually break symmetries, often only locally. Under group-invariant Bellman backups for such cases, local symmetry-breaking introduces errors that propagate across the entire state-action space, resulting in global value estimation errors. To address this, we introduce Partially group-Invariant MDP (PI-MDP), which selectively applies group-invariant or standard Bellman backups depending on where symmetry holds. This framework mitigates error propagation from locally broken symmetries while maintaining the benefits of equivariance, thereby enhancing sample efficiency and generalizability. Building on this framework, we present practical RL algorithms -- Partially Equivariant (PE)-DQN for discrete control and PE-SAC for continuous control -- that combine the benefits of equivariance with robustness to symmetry-breaking. Experiments across Grid-World, locomotion, and manipulation benchmarks demonstrate that PE-DQN and PE-SAC significantly outperform baselines, highlighting the importance of selective symmetry exploitation for robust and sample-efficient RL. Project page: this https URL
>
---
#### [replaced 020] PvP: Data-Efficient Humanoid Robot Learning with Proprioceptive-Privileged Contrastive Representations
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于人形机器人控制任务，旨在解决强化学习样本效率低的问题。提出PvP框架，利用本体感知与特权状态的互补性，提升学习效率和性能。**

- **链接: [https://arxiv.org/pdf/2512.13093](https://arxiv.org/pdf/2512.13093)**

> **作者:** Mingqi Yuan; Tao Yu; Haolin Song; Bo Li; Xin Jin; Hua Chen; Wenjun Zeng
>
> **备注:** 15 pages, 17 figures
>
> **摘要:** Achieving efficient and robust whole-body control (WBC) is essential for enabling humanoid robots to perform complex tasks in dynamic environments. Despite the success of reinforcement learning (RL) in this domain, its sample inefficiency remains a significant challenge due to the intricate dynamics and partial observability of humanoid robots. To address this limitation, we propose PvP, a Proprioceptive-Privileged contrastive learning framework that leverages the intrinsic complementarity between proprioceptive and privileged states. PvP learns compact and task-relevant latent representations without requiring hand-crafted data augmentations, enabling faster and more stable policy learning. To support systematic evaluation, we develop SRL4Humanoid, the first unified and modular framework that provides high-quality implementations of representative state representation learning (SRL) methods for humanoid robot learning. Extensive experiments on the LimX Oli robot across velocity tracking and motion imitation tasks demonstrate that PvP significantly improves sample efficiency and final performance compared to baseline SRL methods. Our study further provides practical insights into integrating SRL with RL for humanoid WBC, offering valuable guidance for data-efficient humanoid robot learning.
>
---
#### [replaced 021] Automated Layout and Control Co-Design of Robust Multi-UAV Transportation Systems
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于多无人机协同运输系统设计任务，旨在优化物理布局与控制器以提高飞行精度和鲁棒性。通过联合优化无人机布局和控制算法，提升系统抗干扰能力。**

- **链接: [https://arxiv.org/pdf/2310.07649](https://arxiv.org/pdf/2310.07649)**

> **作者:** Carlo Bosio; Mark W. Mueller
>
> **备注:** 7 pages, 7 figures, journal paper (IEEE RA-L)
>
> **摘要:** The joint optimization of physical parameters and controllers in robotic systems is challenging. This is due to the difficulties of predicting the effect that changes in physical parameters have on final performances. At the same time, physical and morphological modifications can improve robot capabilities, perhaps completely unlocking new skills and tasks. We present a novel approach to co-optimize the physical layout and the control of a cooperative aerial transportation system. The goal is to achieve the most precise and robust flight when carrying a payload. We assume the agents are connected to the payload through rigid attachments, essentially transforming the whole system into a larger flying object with ``thrust modules" at the attachment locations of the quadcopters. We investigate the optimal arrangement of the thrust modules around the payload, so that the resulting system achieves the best disturbance rejection capabilities. We propose a novel metric of robustness inspired by H2 control, and propose an algorithm to optimize the layout of the vehicles around the object and their controller altogether. We experimentally validate the effectiveness of our approach using fleets of three and four quadcopters and payloads of diverse shapes.
>
---
#### [replaced 022] Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Context-Nav，解决文本引导的实例导航任务，通过上下文驱动探索和3D空间推理，提升在复杂场景中精准定位目标实例的能力。**

- **链接: [https://arxiv.org/pdf/2603.09506](https://arxiv.org/pdf/2603.09506)**

> **作者:** Won Shik Jang; Ue-Hwan Kim
>
> **备注:** Camera-ready version. Accepted to CVPR 2026
>
> **摘要:** Text-goal instance navigation (TGIN) asks an agent to resolve a single, free-form description into actions that reach the correct object instance among same-category distractors. We present \textit{Context-Nav} that elevates long, contextual captions from a local matching cue to a global exploration prior and verifies candidates through 3D spatial reasoning. First, we compute dense text-image alignments for a value map that ranks frontiers -- guiding exploration toward regions consistent with the entire description rather than early detections. Second, upon observing a candidate, we perform a viewpoint-aware relation check: the agent samples plausible observer poses, aligns local frames, and accepts a target only if the spatial relations can be satisfied from at least one viewpoint. The pipeline requires no task-specific training or fine-tuning; we attain state-of-the-art performance on InstanceNav and CoIN-Bench. Ablations show that (i) encoding full captions into the value map avoids wasted motion and (ii) explicit, viewpoint-aware 3D verification prevents semantically plausible but incorrect stops. This suggests that geometry-grounded spatial reasoning is a scalable alternative to heavy policy training or human-in-the-loop interaction for fine-grained instance disambiguation in cluttered 3D scenes.
>
---
#### [replaced 023] CostNav: A Navigation Benchmark for Real-World Economic-Cost Evaluation of Physical AI Agents
- **分类: cs.AI; cs.CE; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出CostNav，一个用于评估物理AI代理经济成本的导航基准，解决传统导航任务忽略经济约束的问题。通过整合真实商业数据，评估方法的经济可行性。**

- **链接: [https://arxiv.org/pdf/2511.20216](https://arxiv.org/pdf/2511.20216)**

> **作者:** Haebin Seong; Sungmin Kim; Yongjun Cho; Myunchul Joe; Geunwoo Kim; Yubeen Park; Sunhoo Kim; Yoonshik Kim; Suhwan Choi; Jaeyoon Jung; Jiyong Youn; Jinmyung Kwak; Sunghee Ahn; Jaemin Lee; Younggil Do; Seungyeop Yi; Woojin Cheong; Minhyeok Oh; Minchan Kim; Seongjae Kang; Samwoo Seong; Youngjae Yu; Yunsung Lee
>
> **摘要:** While current navigation benchmarks prioritize task success in simplified settings, they neglect the multidimensional economic constraints essential for the real-world commercialization of autonomous delivery systems. We introduce CostNav, an Economic Navigation Benchmark that evaluates physical AI agents through comprehensive economic cost-revenue analysis aligned with real-world business operations. By integrating industry-standard data--such as Securities and Exchange Commission (SEC) filings and Abbreviated Injury Scale (AIS) injury reports--with Isaac Sim's detailed collision and cargo dynamics, CostNav transcends simple task completion to accurately evaluate business value in complex, real-world scenarios. To our knowledge, CostNav is the first physics-grounded economic benchmark that uses industry-standard regulatory and financial data to quantitatively expose the gap between navigation research metrics and commercial viability, revealing that optimizing for task success on a simplified task fundamentally differs from optimizing for real-world economic deployment. Evaluating seven baselines--two rule-based and five imitation learning--we find that no current method is economically viable, all yielding negative contribution margins. The best-performing method, CANVAS (-27.36\$/run), equipped with only an RGB camera and GPS, outperforms LiDAR-equipped Nav2 w/ GPS (-35.46\$/run). We challenge the community to develop navigation policies that achieve economic viability on CostNav. We remain method-agnostic, evaluating success solely on cost rather than the underlying architecture. All resources are available at this https URL.
>
---
#### [replaced 024] MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent
- **分类: cs.RO**

- **简介: 该论文提出MergeVLA，解决VLA模型多技能融合难题，通过设计可合并架构实现跨任务、跨环境的强泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.18810](https://arxiv.org/pdf/2511.18810)**

> **作者:** Yuxia Fu; Zhizhen Zhang; Yuqi Zhang; Zijian Wang; Zi Huang; Yadan Luo
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Recent Vision-Language-Action (VLA) models reformulate vision-language models by tuning them with millions of robotic demonstrations. While they perform well when fine-tuned for a single embodiment or task family, extending them to multi-skill settings remains challenging: directly merging VLA experts trained on different tasks results in near-zero success rates. This raises a fundamental question: what prevents VLAs from mastering multiple skills within one model? With an empirical decomposition of learnable parameters during VLA fine-tuning, we identify two key sources of non-mergeability: (1) Finetuning drives LoRA adapters in the VLM backbone toward divergent, task-specific directions beyond the capacity of existing merging methods to unify. (2) Action experts develop inter-block dependencies through self-attention feedback, causing task information to spread across layers and preventing modular recombination. To address these challenges, we present MergeVLA, a merging-oriented VLA architecture that preserves mergeability by design. MergeVLA introduces sparsely activated LoRA adapters via task masks to retain consistent parameters and reduce irreconcilable conflicts in the VLM. Its action expert replaces self-attention with cross-attention-only blocks to keep specialization localized and composable. When the task is unknown, it uses a test-time task router to adaptively select the appropriate task mask and expert head from the initial observation, enabling unsupervised task inference. Across LIBERO, LIBERO-Plus, RoboTwin, and multi-task experiments on the real SO101 robotic arm, MergeVLA achieves performance comparable to or even exceeding individually finetuned experts, demonstrating robust generalization across tasks, embodiments, and environments. Project page: this https URL
>
---
#### [replaced 025] Symskill: Symbol and Skill Co-Invention for Data-Efficient and Reactive Long-Horizon Manipulation
- **分类: cs.RO**

- **简介: 该论文提出SymSkill，解决动态环境中多步骤操作的组合泛化与实时恢复问题。通过联合学习符号抽象和技能，实现数据高效、反应式的长周期操作。**

- **链接: [https://arxiv.org/pdf/2510.01661](https://arxiv.org/pdf/2510.01661)**

> **作者:** Yifei Simon Shao; Yuchen Zheng; Sunan Sun; Pratik Chaudhari; Vijay Kumar; Nadia Figueroa
>
> **备注:** ICRA 2026; CoRL 2025 Learning Effective Abstractions for Planning (LEAP) Workshop Best Paper Award (this https URL)
>
> **摘要:** Multi-step manipulation in dynamic environments remains challenging. Imitation learning (IL) is reactive but lacks compositional generalization, since monolithic policies do not decide which skill to reuse when scenes change. Classical task-and-motion planning (TAMP) offers compositionality, but its high planning latency prevents real-time failure recovery. We introduce SymSkill, a unified framework that jointly learns predicates, operators, and skills from unlabeled, unsegmented demonstrations, combining compositional generalization with real-time recovery. Offline, SymSkill learns symbolic abstractions and goal-oriented skills directly from demonstrations. Online, given a conjunction of learned predicates, it uses a symbolic planner to compose and reorder skills to achieve symbolic goals while recovering from failures at both the motion and symbolic levels in real time. Coupled with a compliant controller, SymSkill supports safe execution under human and environmental disturbances. In RoboCasa simulation, SymSkill executes 12 single-step tasks with 85% success and composes them into multi-step plans without additional data. On a real Franka robot, it learns from 5 minutes of play data and performs 12-step tasks from goal specifications. Code and additional analysis are available at this https URL.
>
---
#### [replaced 026] Robust Cooperative Localization in Featureless Environments: A Comparative Study of DCL, StCL, CCL, CI, and Standard-CL
- **分类: cs.RO**

- **简介: 论文研究多机器人协同定位问题，比较了五种CL方法在无特征环境中的性能，分析其准确性与稳定性，为应用选择提供依据。**

- **链接: [https://arxiv.org/pdf/2603.09886](https://arxiv.org/pdf/2603.09886)**

> **作者:** Nivand Khosravi; Rodrigo Ventura; Meysam Basiri
>
> **备注:** Presented at the 2026 12th International Conference on Automation, Robotics and Applications (ICARA); to be published in IEEE conference proceedings
>
> **摘要:** Cooperative localization (CL) enables accurate position estimation in multi-robot systems operating in GPS-denied environments. This paper presents a comparative study of five CL approaches: Centralized Cooperative Localization (CCL), Decentralized Cooperative Localization (DCL), Sequential Cooperative Localization (StCL), Covariance Intersection (CI), and Standard Cooperative Localization (Standard-CL). All methods are implemented in ROS and evaluated through Monte Carlo simulations under two conditions: weak data association and robust detection. Our analysis reveals fundamental trade-offs among the methods. StCL and Standard-CL achieve the lowest position errors but exhibit severe filter inconsistency, making them unsuitable for safety-critical applications. DCL demonstrates remarkable stability under challenging conditions due to its measurement stride mechanism, which provides implicit regularization against outliers. CI emerges as the most balanced approach, achieving near-optimal consistency while maintaining competitive accuracy. CCL provides theoretically optimal estimation but shows sensitivity to measurement outliers. These findings offer practical guidance for selecting CL algorithms based on application requirements.
>
---
#### [replaced 027] World Models That Know When They Don't Know - Controllable Video Generation with Calibrated Uncertainty
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视频生成任务，旨在解决可控视频模型 hallucinate 的问题。通过引入不确定性量化方法，提升模型对不确定性的估计能力，增强可靠性。**

- **链接: [https://arxiv.org/pdf/2512.05927](https://arxiv.org/pdf/2512.05927)**

> **作者:** Zhiting Mei; Tenny Yin; Micah Baker; Ola Shorinwa; Anirudha Majumdar
>
> **摘要:** Recent advances in generative video models have led to significant breakthroughs in high-fidelity video synthesis, specifically in controllable video generation where the generated video is conditioned on text and action inputs, e.g., in instruction-guided video editing and world modeling in robotics. Despite these exceptional capabilities, controllable video models often hallucinate - generating future video frames that are misaligned with physical reality - which raises serious concerns in many tasks such as robot policy evaluation and planning. However, state-of-the-art video models lack the ability to assess and express their confidence, impeding hallucination mitigation. To rigorously address this challenge, we propose C3, an uncertainty quantification (UQ) method for training continuous-scale calibrated controllable video models for dense confidence estimation at the subpatch level, precisely localizing the uncertainty in each generated video frame. Our UQ method introduces three core innovations to empower video models to estimate their uncertainty. First, our method develops a novel framework that trains video models for correctness and calibration via strictly proper scoring rules. Second, we estimate the video model's uncertainty in latent space, avoiding training instability and prohibitive training costs associated with pixel-space approaches. Third, we map the dense latent-space uncertainty to interpretable pixel-level uncertainty in the RGB space for intuitive visualization, providing high-resolution uncertainty heatmaps that identify untrustworthy regions. Through extensive experiments on large-scale robot learning datasets (Bridge and DROID) and real-world evaluations, we demonstrate that our method not only provides calibrated uncertainty estimates within the training distribution, but also enables effective out-of-distribution detection.
>
---
