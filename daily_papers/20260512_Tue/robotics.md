# 机器人 cs.RO

- **最新发布 113 篇**

- **更新 61 篇**

## 最新发布

#### [new 001] MVB-Grasp: Minimum-Volume-Box Filtering of Diffusion-based Grasps for Frontal Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机械臂抓取任务，解决低成本机械臂在受限空间中抓取失败率高的问题。通过引入最小体积包围盒过滤，提升抓取成功率。**

- **链接: [https://arxiv.org/pdf/2605.09672](https://arxiv.org/pdf/2605.09672)**

> **作者:** Bibek Poudel; Abdul Basit; Muhammad Shafique
>
> **备注:** 8 pages, 12 figures, accepted to IJCNN 2026
>
> **摘要:** State-of-the-art 6-DoF grasp generators excel on tabletop benchmarks with overhead cameras but struggle in frontal grasping scenarios on low-cost manipulators with constrained workspaces, where kinematic limits and approach-direction constraints cause high failure rates. We address this challenge for the Unitree Z1 arm by proposing MVB-Grasp, a novel grasping stack that injects a Minimum Volume Bounding Box (MVBB) geometric prior into diffusion-based grasp generation to dramatically improve success rates in frontal, workspace-constrained settings. Our key scientific contributions are threefold: (i) an MVBB-based geometric filter that exploits oriented bounding-box face normals to reject grasps approaching through the table or misaligned with accessible object faces in O(N) time; (ii) a combined re-scoring function that blends learned discriminator scores with face-alignment geometry {\alpha}=0.85, specifically calibrated for the Z1's frontal workspace and kinematic constraints; and (iii) a systematic MuJoCo evaluation protocol measuring grasp success across object types, distances, lateral positions, and pitch orientations to validate embodiment-specific performance. We implement MVB-Grasp on a Unitree Z1 arm with an Intel RealSense D405 camera, integrating YOLOv8 object detection, GraspGen for candidate generation, Principal Component Analysis (PCA)-based MVBB fitting, and inverse-kinematics trajectory planning. Experiments across 81 MuJoCo episodes (cylinder, asymmetric box, waterbottle) demonstrate that MVB-Grasp achieves 59.3% success versus 24.7% for vanilla GraspGen, a 2.4x improvement, by filtering geometrically infeasible candidates and prioritizing face-aligned grasps suited to the Z1's frontal approach constraints. Real-world trials confirm that the MVBB prior substantially improves grasp reliability on constrained, low-cost manipulators without requiring model retraining.
>
---
#### [new 002] SHIELD: Scalable Optimal Control with Certification using Duality and Convexity
- **分类: cs.RO**

- **简介: 本文提出SHIELD算法，用于解决复杂交通场景下的模型预测控制问题。通过降维和约束削减，实现高效安全的控制决策。**

- **链接: [https://arxiv.org/pdf/2605.09171](https://arxiv.org/pdf/2605.09171)**

> **作者:** Hansung Kim; Siddharth H. Nair; Francesco Borrelli
>
> **摘要:** We present SHIELD, a hierarchical algorithm that reduces both the decision-variable dimension and the constraint set in $\ell_1$-regularized convex programs. From strong convexity and Lagrangian duality, we derive certificates that \emph{safely} discard constraints and decision variables while guaranteeing that all removed constraints remain satisfied and all removed variables are null. To further accelerate the proposed algorithm, we propose a transformer-based deep neural network to guide the dual certificate inference. We validate SHIELD on stochastic model predictive control (SMPC) in complex, multi-modal traffic scenarios, comparing against a full-dimensional SMPC policy. Numerical simulations demonstrate order-of-magnitude computational speedups while preserving feasibility and closed-loop safety, highlighting the practicality of certifiably safe, lightweight MPC in complex driving scenes.
>
---
#### [new 003] A cell-decomposition based path planner for 3D navigation in constrained workspaces
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决3D受限空间中的路径可行性问题。提出基于单元分解的算法，结合优化方法实现高效路径搜索。**

- **链接: [https://arxiv.org/pdf/2605.10086](https://arxiv.org/pdf/2605.10086)**

> **作者:** João P. L. Morais; Luciano C. A. Pimenta; Marcelo A. Santos; Guilherme V. Raffo
>
> **备注:** Accepted for publication at the 23rd IFAC World Congress (Busan, Korea)
>
> **摘要:** This paper proposes a cell decomposition algorithm for binary occupancy grids that ensures mutual complete visibility from each cell to at least one adjacent cell. This decomposition establishes a simplified framework for verifying path feasibility that can be easily embedded in optimization problems. To illustrate its utility, we formulate both second-order cone programs (SOCP) and their mixed-integer variant (MISOCP) within the proposed framework. Furthermore, we propose the KSP-SOCP method, which combines Yen's k-shortest path algorithm with the SOCP, achieving improved solutions compared to a standard SOCP approach while avoiding the computational burden of MISOCP. The cell decomposition algorithm, KSP-SOCP, and MISOCP approaches were evaluated in 9 city-like workspaces. The decomposition efficiently partitioned each map, enabling both optimization methods to compute feasible paths. The proposed KSP-SOCP achieved time performance comparable to the MISOCP while requiring less memory, making it highly suitable for large-scale problems.
>
---
#### [new 004] Beyond Self-Play and Scale: A Behavior Benchmark for Generalization in Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶领域，旨在解决强化学习策略泛化能力不足的问题。通过构建BehaviorBench基准，评估并提升策略在复杂、多样化交通环境中的表现。**

- **链接: [https://arxiv.org/pdf/2605.10034](https://arxiv.org/pdf/2605.10034)**

> **作者:** Aron Distelzweig; Faris Janjoš; Andreas Look; Anna Rothenhäusler; Daniel Jost; Oliver Scheel; Raghu Rajan; Daphne Cornelisse; Eugene Vinitsky; Joschka Boedecker
>
> **摘要:** Recent Autonomous Driving (AD) works such as GigaFlow and PufferDrive have unlocked Reinforcement Learning (RL) at scale as a training strategy for driving policies. Yet such policies remain disconnected from established benchmarks, leaving the performance of large-scale RL for driving on standardized evaluations unknown. We present BehaviorBench -- a comprehensive test suite that closes this gap along three axes: Evaluation, Complexity, and Behavior Diversity. In terms of Evaluation, we provide an interface connecting PufferDrive to nuPlan, which, for the first time, enables policies trained via RL at scale to be evaluated on an established planning benchmark for autonomous driving. Complementarily, we offer an evaluation framework that allows planners to be benchmarked directly inside the PufferDrive simulation, at a fraction of the time. Regarding Complexity, we observe that today's standardized benchmarks are so simple that near-perfect scores are achievable by straight lane following with collision checking. We extract a meaningful, interaction-rich split from the Waymo Open Motion Dataset (WOMD) on which strong performance is impossible without multi-agent reasoning. Lastly, we address Behavior Diversity. Existing benchmarks commonly evaluate planners against a single rule-based traffic model, the Intelligent Driver Model (IDM). We provide a diverse suite of interactive traffic agents to stress-test policies under heterogeneous behaviors, beyond just using IDM. Overall, our benchmarking analysis uncovers the following insight: despite learning interactive behaviors in an emergent manner, policies trained via pure self-play under standard reward functions overfit to their training opponents and fail to generalize to other traffic agent behaviors. Building on this observation, we propose a hybrid planner that combines a PPO policy with a rule-based planner.
>
---
#### [new 005] ORICF -- Open Robotics Inference and Control Framework
- **分类: cs.RO**

- **简介: 该论文提出ORICF框架，解决机器人推理与控制的计算开销问题。通过模块化设计和边缘计算，降低能耗与延迟，提升效率。**

- **链接: [https://arxiv.org/pdf/2605.09656](https://arxiv.org/pdf/2605.09656)**

> **作者:** Andrés Meseguer Valenzuela; Luís Miguel Bartolín Arnau
>
> **备注:** Accepted in ICRA26 Workshop: 8th International Workshop on Robotics Software Engineering (RoSE 26)
>
> **摘要:** Recent advances in artificial intelligence (AI) have enabled effective perception and language models for robots, but their deployment remains computationally expensive, increasing latency and energy use. This work presents the Open Robotics Inference and Control Framework (ORICF), a modular, declarative, and model-agnostic platform for composing multimodal robotic inference pipelines. ORICF integrates input/output (I/O) adapters, pluggable inference back ends, and post-processing logic, while lightweight YAML specifications allow models, hardware targets, and data channels to be changed without code modification. The framework also supports edge offloading, i.e., executing inference on nearby external computers instead of onboard the robot. ORICF is evaluated on a mobile robot that answers spoken queries about people detected in its camera stream by combining automatic speech recognition (ASR), a large language model (LLM), and a convolutional neural network (CNN) detector through Robot Operating System 2 (ROS2). Compared with onboard execution, ORICF-based edge deployment reduces robot-side compute utilization by up to 83.16% and estimated energy consumption by 65.8%, while preserving modularity and reproducibility.
>
---
#### [new 006] A low-cost mockup to simulate robotic laser cutting in nuclear decommissioning
- **分类: cs.RO**

- **简介: 该论文属于机器人激光切割任务，旨在解决核退役中精准切割的问题。构建低成本模拟系统，采用自适应控制器提高轨迹跟踪精度，无需校准且能避障。**

- **链接: [https://arxiv.org/pdf/2605.08947](https://arxiv.org/pdf/2605.08947)**

> **作者:** Frederico Fernandes Afonso Silva; Murilo Marques Marinho; Bruno Vilhena Adorno
>
> **备注:** 7 pages, 8 figures, 2 tables. Under Review for TAROS 2026 (Towards Autonomous Robotic Systems)
>
> **摘要:** This paper introduces a low-cost experimental mockup to simulate the laser cutting process of containers in nuclear decommissioning. It is composed of a three-axis table supporting a cuboid container with ultraviolet-sensitive faces, a six-degree-of-freedom serial manipulator holding an ultraviolet torch that simulates the laser, and a visual system based on cameras and fiducial markers. The system employs a constrained task-space adaptive motion controller that compensates for inaccurate parameters and eliminates the need to calibrate the system. Furthermore, as the motion controller explicitly accounts for geometric constraints, the robot reactively avoids collisions with obstacles while handling the ultraviolet torch. To enhance tracking of the laser-cutting path, we control the ultraviolet beam, which requires only four degrees of freedom, instead of the full end-effector pose. Experiments show that, despite an initially uncalibrated system, the overall system is capable of tracking different trajectories with an overall mean accuracy of 3.9 (sd 2.5) mm when the end-effector pose is controlled and 2.4 (sd 1.3) mm when the ultraviolet beam is controlled.
>
---
#### [new 007] Safe Aerial 3D Path Planning for Autonomous UAVs using Magnetic Potential Fields
- **分类: cs.RO**

- **简介: 该论文属于无人机路径规划任务，解决城市环境中安全自主导航问题。提出3DMaxConvNet方法，利用磁势场实现高效避障路径规划。**

- **链接: [https://arxiv.org/pdf/2605.10880](https://arxiv.org/pdf/2605.10880)**

> **作者:** Haechan Mark Bong; Giovanni Beltrame
>
> **摘要:** Safe autonomous Uncrewed Aerial Vehicle (UAV) navigation in urban environments requires real-time path planning that avoids obstacles. MaxConvNet is a potential-field planner that leverages properties of Maxwell's equations to generate a path to the goal without local minima. We extend the 2D MaxConvNet magnetic field planner to 3D, using a convolutional autoencoder to predict obstacle-aware potential fields from LiDAR-derived 101^3 voxel grids. Evaluation across 100 randomized closed-loop trials in two distinct Cosys-AirSim urban environments, a dense night-time cityscape and a suburban district shows a 100% path planning success rate on both maps without retraining. In offline path planning, 3DMaxConvNet produces path lengths comparable to A* on unseen maps while reducing runtime from 0.155--0.17s to 0.087--0.089s, or about 1.7--1.95 times faster than A*. Against RRT*(3k), 3DMaxConvNet achieves similar path quality while reducing planning runtime from 17.2--17.5s to about 0.09s, which is roughly 193--201 times faster than RRT*(3k).
>
---
#### [new 008] ElasticFlow: One-Step Physics-Consistent Policy with Elastic Time Horizons for Language-Guided Manipulation
- **分类: cs.RO**

- **简介: 该论文提出ElasticFlow，解决语言引导操作中的高延迟与物理一致性问题，通过单步策略和弹性时间范围机制实现高效、语义对齐的控制。**

- **链接: [https://arxiv.org/pdf/2605.08799](https://arxiv.org/pdf/2605.08799)**

> **作者:** Kewei Chen; Yayu Long; Shuai Li; Mingsheng Shang
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Diffusion policies have demonstrated exceptional performance in embodied AI. However, their iterative denoising process results in high latency, and existing acceleration methods often sacrifice physical consistency. To address this, we propose ElasticFlow, a distillation-free, physics-consistent one-step policy framework. We reconstruct the Mean Field Theory by directly modeling the average velocity field, enabling a direct single-step mapping from noise to action. Addressing the Temporal Heterogeneity of robotic tasks, we introduce the Elastic Time Horizons mechanism. This mechanism effectively overcomes Spectral Bias by explicitly encoding control granularity, achieving efficient alignment between semantic instructions and physical execution horizons. Experiments on benchmarks such as LIBERO, CALVIN, and RoboTwin demonstrate that ElasticFlow achieves efficient 1-NFE inference (approximately 71Hz). Furthermore, it outperforms state-of-the-art methods, including OpenVLA and $\pi_0$, on long-horizon tasks, highlighting its potential for efficient, robust, and semantically aligned control.
>
---
#### [new 009] REAP: Reinforcement-Learning End-to-End Autonomous Parking with Gaussian Splatting Simulator for Real2Sim2Real Transfer
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主泊车任务，旨在解决极端场景下泊车失败问题。提出REAP方法，结合强化学习与端到端模型，提升泊车性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.08713](https://arxiv.org/pdf/2605.08713)**

> **作者:** Changze Li; Zhe Chen; Shaoyu Chen; Lisen Mu; Yijian Li; Yuelong Yu; Qian Zhang; Qing Su; Ming Yang; Tong Qin
>
> **摘要:** In recent years, autonomous parking has made significant advances, yet parking tasks still face challenges in extreme scenarios such as mechanical and dead-end parking slots, often resulting in failures. This is mainly due to traditional parking methods adopting a multistage approach, lacking the ability to optimize the parking problem as a whole. End-to-end methods enable joint optimization across perception and planning modules to eliminate the accumulation of errors, enhancing algorithm performance in extreme scenarios. Although several end-to-end parking methods use imitation or reinforcement learning, the former is limited by data cost and distribution coverage, while the latter suffers from inefficient exploration. To address these challenges, we propose a Reinforcement learning End-to-end Autonomous Parking method (REAP). REAP employs Soft Actor-Critic (SAC) within an asymmetric reinforcement learning framework to improve training efficiency and inference performance. To accelerate model convergence, we distill the capabilities of a rule-based planner into the end-to-end network through behavior cloning. We further introduce a soft predictive collision penalty mechanism to reduce collision rates by penalizing obstacle-approaching actions. To ensure that the trained reinforcement learning network can directly transfer to real-world scenarios, we have established a Real2Sim2Real simulator. In the Real2Sim step, we use 3D Gaussian Splatting (3DGS) to transform real-world scenes into digital scenes. In the Sim2Real step, we deploy the end-to-end model onto the vehicle to bridge the Sim2Real gap. Trained in the 3DGS simulator and deployed on physical vehicles, REAP successfully parks in various types of parking spaces, especially demonstrating the feasibility of end-to-end RL parking in extremely narrow mechanical slots.
>
---
#### [new 010] PECMAN: Perception-enabled Collaborative Multi-Agent Navigation in Unknown Environments
- **分类: cs.RO; cs.MA**

- **简介: 该论文提出PECMAN，用于未知环境中多智能体协同导航任务。解决动态环境下路径规划效率与协作问题，通过分布式树重构和共享感知策略提升导航效率与成功率。**

- **链接: [https://arxiv.org/pdf/2605.09344](https://arxiv.org/pdf/2605.09344)**

> **作者:** Tianchonghui Fang; Shaunak Roy; Shalabh Gupta
>
> **摘要:** Most path planners assume fully known, static environments, assumptions that fail when robots navigate in dynamic and partially observable environments. SMART-3D addresses these issues by real-time replanning, where it morphs the underlying RRT* tree whenever new obstacles or structures are discovered in the environment. Instead of rebuilding the tree entirely from scratch, SMART-3D prunes invalid nodes and edges and subsequently repairs the disjoint subtrees at hot-nodes to find a new path, thus providing high computational efficiency for real-time adaptability. We extend SMART-3D to perception-enabled collaborative multi-agent navigation (PECMAN) in unknown environments. PECMAN is built upon distributed tree morphing and shared perception strategies, where each agent reacts to environmental changes and morphs its respective tree to replan its path, while simultaneously broadcasting newly discovered structures to other agents, thus enabling them to proactively replan even in areas that have not yet been explored by them. This approach reduces redundant reactions and unnecessary replannings of the agents due to improved situational awareness. The performance of PECMAN was evaluated by 28,000 multi-agent simulations on seven 2D scenarios with different case studies. The results show that PECMAN achieves up to 52% reduction in the team-completion time, while maintaining near 100% success rates. Finally, PECMAN was tested by real experiments on two autonomous robots in a building environment.
>
---
#### [new 011] Zero-Shot Sim-to-Real Robot Learning: A Dexterous Manipulation Study on Reactive Catching
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决物理敏感的灵巧操作中的sim-to-real迁移难题。通过提出DRIS方法，提升策略鲁棒性，实现无需现实微调的零样本迁移。**

- **链接: [https://arxiv.org/pdf/2605.09789](https://arxiv.org/pdf/2605.09789)**

> **作者:** Kejia Ren; Gaotian Wang; Andrew S. Morgan; Kaiyu Hang
>
> **摘要:** Dexterous manipulation is physics-intensive and highly sensitive to modeling errors and perception noise, making sim-to-real transfer prohibitively challenging. Domain randomization (DR) is commonly used to improve the robustness of learned policies for such tasks, but conventional DR randomizes one instance per episode, offering very limited exposure to the variability of real-world dynamics. To this end, we propose Domain-Randomized Instance Set (DRIS), which represents and propagates a set of randomized instances simultaneously, providing richer approximation of uncertain dynamics and enabling policies to learn actions that account for multiple possible outcomes. Supported by theoretical analysis, we show that DRIS yields more robust policies and alleviates the need for real-world fine-tuning, even with a modest number of instances (e.g., 10). We demonstrate this on a challenging reactive catching task. Unlike traditional catching setups that use end-effectors designed to mechanically stabilize the object (e.g., curved or enclosing surfaces), our system uses a flat plate that offers no passive stabilization, making the task highly sensitive to noise and requiring rapid reactive motions. The learned policies exhibit strong robustness to uncertainties and achieve reliable zero-shot sim-to-real transfer.
>
---
#### [new 012] Automated Robotic Moisture Monitoring in Agricultural Fields
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于农业监测任务，旨在解决大田水分检测效率低的问题。通过机器人与传感器结合，实现自动化的水分监测与路径规划。**

- **链接: [https://arxiv.org/pdf/2605.09050](https://arxiv.org/pdf/2605.09050)**

> **作者:** Senthil Palanisamy; Akila I.S
>
> **备注:** 2018 International Seminar on Intelligent Technology and Its Applications (ISITIA)
>
> **摘要:** Monitoring moisture level of land in a large-scale plantation is tedious. The main objective of this project is to use a robotic kit in collaboration with the on-field moisture sensor circuits, thereby creating an efficient and economical moisture monitoring system. A large agriculture field is divided into smaller grids. Each grid is placed with a moisture sensor. Whenever a sensor reports the soil to be dry, the robot goes to the concerned field for inspection. The path to the concerned field is found by applying Dijkstra's shortest path algorithm on the aerial image of the field. Then the total moisture content of the field is calculated by the robot using suitable image processing algorithms and reported accordingly. For developing and testing this work, a small study field was set up above which a camera was mounted at an appropriate height to capture its aerial view. Thus a prototype for an automated system of monitoring agricultural fields' moisture has been developed through this work.
>
---
#### [new 013] Guided Streaming Stochastic Interpolant Policy
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，解决实时动态目标引导问题。提出SSIP框架，通过优化指导项实现快速反应控制，提升环境适应能力。**

- **链接: [https://arxiv.org/pdf/2605.10051](https://arxiv.org/pdf/2605.10051)**

> **作者:** Puming Jiang; Meiyi Wang; Kelvin Lin; Ce Hao; Harold Soh
>
> **备注:** Accepted to Robotics: Science and Systems (RSS) 2026. The first two authors contributed equally
>
> **摘要:** Inference-time guidance is essential for steering generative robot policies toward dynamic objectives without retraining, yet existing methods are largely confined to chunk-based architectures that exhibit high latency and lack the reactivity needed for test-time preference alignment or obstacle avoidance. In this work, we formally derive the optimal guidance term for Stochastic Interpolants (SI) by analyzing the value function's time evolution via the Backward Kolmogorov Equation, establishing a modified drift that theoretically guarantees sampling from a target distribution. We apply this framework to real-time control through the Streaming Stochastic Interpolant Policy (SSIP), which generalizes the deterministic Streaming Flow Policy (SFP). Unifying this guidance law with the streaming architecture enables fast and reactive control. To support diverse deployment needs, we propose two complementary mechanisms: training-free Stochastic Trajectory Ensemble Guidance (STEG) that computes gradients on-the-fly for zero-shot adaptation, and training-based Conditional Critic Guidance (CCG) for amortized inference. Empirical evaluations demonstrate that our guided streaming approach significantly outperforms conventional chunk-based policies in reactivity and provides superior, physically valid guidance for dynamic, unstructured environments.
>
---
#### [new 014] Data-Asymmetric Latent Imagination and Reranking for 3D Robotic Imitation Learning
- **分类: cs.RO**

- **简介: 该论文属于3D机器人模仿学习任务，解决低质量轨迹数据利用问题。提出DALI-R框架，通过潜在世界模型和重排序机制提升决策效果，无需额外高质量数据。**

- **链接: [https://arxiv.org/pdf/2605.10166](https://arxiv.org/pdf/2605.10166)**

> **作者:** Lianghao Luo; Xizhou Bu; Ruyan Liu; Qingqiu Huang; Chufeng Tang; Xiaoshuai Hao; Hongbo Wang; Wei Li
>
> **摘要:** Robotic imitation learning typically assumes access to optimal demonstrations, yet real-world data collection often yields suboptimal, exploratory, or even failed trajectories. Discarding such data wastes valuable information about environment dynamics and failure modes, which can instead be leveraged to improve decision-making. While 3D policies reduce reliance on high-quality demonstrations through strong spatial generalization, they still require large-scale data to achieve high task success. To address this, we propose DALI-R, a Data-Asymmetric Latent Imagination and Reranking framework for 3D robotic imitation learning from mixed-quality trajectories. It learns a Latent World Model over 3D point clouds for imagined rollouts and a Task Completion Scorer that reranks candidate action chunks, improving decision-making without additional high-quality demonstrations. We instantiate DALI-R with both diffusion and efficient flow-matching policies and evaluate it on Adroit and MetaWorld benchmarks. Across the two evaluated 3D base policies, DALI-R achieves an average $6.8$\% improvement in success rate while incurring less than $0.7\times$ additional inference overhead.
>
---
#### [new 015] ObjView-Bench: Rethinking Difficulty and Deployment for Object-Centric View Planning
- **分类: cs.RO**

- **简介: 该论文属于3D重建任务，解决对象中心视角规划的评估问题。通过分离难度因素、设计部署评估协议，提升视角规划方法的可靠性与性能分析。**

- **链接: [https://arxiv.org/pdf/2605.10707](https://arxiv.org/pdf/2605.10707)**

> **作者:** Sicong Pan; Hao Hu; Xuying Huang; Benno Wingender; Maren Bennewitz
>
> **摘要:** Object-centric view planning is a core component of active geometric 3D reconstruction in robotics, yet existing evaluations often conflate object complexity, planning difficulty, budget assumptions, and physical reachability constraints. As a result, conclusions drawn from idealized view-planning evaluations may not reliably predict performance under realistic reconstruction settings. We introduce ObjView-Bench, an evaluation framework for rethinking difficulty and deployment in object-centric view planning. First, we disentangle three quantities underlying view-planning evaluation: omnidirectional self-occlusion as an object-side attribute, observation saturation difficulty, and protocol-dependent planning difficulty defined through a set-cover formulation. This separation supports controlled dataset construction, analysis of slow-saturation objects, and a case study showing that planning difficulty-aware sampling can improve learned view planners. Second, we design deployment-oriented evaluation protocols that reveal how budget regimes and reachable-view constraints alter method behavior. Across classical, learned, and hybrid planners, ObjView-Bench shows that difficulty, budget, and reachability constraints substantially change method rankings and failure modes.
>
---
#### [new 016] Beyond Isolation: A Unified Benchmark for General-Purpose Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决现有评估协议碎片化的问题。提出OmniNavBench基准，支持多技能协作和多形态机器人泛化，提升导航系统的真实场景适应能力。**

- **链接: [https://arxiv.org/pdf/2605.09441](https://arxiv.org/pdf/2605.09441)**

> **作者:** Samson Sun; Tianyi Yang; Tengyue Wang; Yikai Xue; Zhengjie Xu; Lingming Zhang; Qichen Zhang; Chao Liang; Zhipeng Zhang
>
> **备注:** Accepted at RSS 2026
>
> **摘要:** The pursuit of general-purpose embodied agents is hindered by fragmented evaluation protocols that isolate navigation skills and fixate on specific robot morphologies, failing to reflect real-world scenarios where agents must orchestrate diverse behaviors across varying embodiments. To bridge this gap, we introduce OmniNavBench, a benchmark for cross-skill coordination and cross-embodiment generalization. OmniNavBench introduces three paradigm shifts: (1) Compositional Complexity. We propose composite instructions that interleave sub-tasks from 6 categories (PointNav, VLN, ObjectNav, SocialNav, Human Following and EQA), compelling agents to transition between exploration, interaction, and social compliance within a single episode. (2) Morphological Universality and Sensor Flexibility. We present a simulation platform that breaks the reliance on single-morphology evaluation, enabling generalization tests across humanoid, quadrupedal, and wheeled robots, with a modular sensor interface and 170 environments blending synthetic assets with real-world scans. (3) Demonstrations Quality. Moving beyond shortest-path algorithms, we curate 1779 expert trajectories via human teleoperation, capturing behavioral nuances such as exploratory glance and anticipatory avoidance. Extensive evaluations demonstrate that current methods, despite their claimed unified design, struggle with the complex, interleaved nature of general-purpose navigation. This exposes a critical disparity between existing capabilities and real-world deployment demands, underscoring OmniNavBench as a testbed for the next generation of generalist navigators. Dataset, code, and leaderboard are available at this http URL.
>
---
#### [new 017] Learning Point Cloud Geometry as a Statistical Manifold: Theory and Practice
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决点云几何表示问题。通过将局部几何建模为统计流形，提出POLI模型，实现无需标注数据的自监督几何估计。**

- **链接: [https://arxiv.org/pdf/2605.10456](https://arxiv.org/pdf/2605.10456)**

> **作者:** Jinwoo Lee; Jiwoo Kim; Woojae Shin; Giseop Kim; Hyondong Oh
>
> **摘要:** Point clouds are a fundamental representation for robotic perception tasks such as localization, mapping, and object pose estimation. However, LiDAR-acquired point clouds are inherently sparse and non-uniform, providing incomplete observations of the underlying scene geometry. This makes reliable geometric reasoning challenging and degrades downstream perception performance. Existing approaches attempt to compensate for these limitations by estimating local geometry, but often rely on hand-crafted statistics or end-to-end supervised learning, which can suffer from limited scalability or require large amounts of accurately labeled data. To address these challenges, we explicitly model point cloud geometry under a principled mathematical formulation. We represent local geometry as a statistical manifold induced by a family of Gaussian distributions, where each point is associated with a Gaussian capturing its local geometric structure. Based on this formulation, we introduce Point-to-Ellipsoid (POLI), a deep neural estimator that predicts per-point Gaussian geometry. POLI learns a mapping from point cloud observations to their underlying geometry in a self-supervised manner, removing the need for labeled data while preserving strong geometric inductive biases. The resulting representation integrates seamlessly into existing robotic perception pipelines without architectural modifications. Extensive experiments show that POLI enables accurate and robust geometry estimation and consistently improves performance across diverse robotic perception tasks.
>
---
#### [new 018] BEACON: Cross-Domain Co-Training of Generative Robot Policies via Best-Effort Adaptation
- **分类: cs.RO**

- **简介: 该论文提出BEACON框架，解决跨域机器人策略训练问题，通过重要性加权和差异估计，提升策略的泛化与数据效率。**

- **链接: [https://arxiv.org/pdf/2605.08571](https://arxiv.org/pdf/2605.08571)**

> **作者:** Antong Zhang; Han Qi; Heng Yang
>
> **摘要:** We introduce BEACON--Best-Effort Adaptation for Cross-Domain Co-Training--a theory-driven framework for training generative robot policies with abundant source demonstrations and limited target demonstrations. BEACON casts cross-domain co-training as a discrepancy-aware importance-reweighting problem, jointly learning a diffusion-based visuomotor policy and per-sample source weights that minimize an objective informed by target-domain generalization guarantees. To make best-effort adaptation practical for high-dimensional sequence policies, we develop scalable instance-level discrepancy estimators, stochastic alternating updates for policy and weights, and a multi-source extension that balances heterogeneous source domains. Across sim-to-sim, sim-to-real, and multi-source manipulation settings, BEACON improves robustness and data efficiency over target-only, fixed-ratio co-training, and feature-alignment baselines. Importantly, even without an explicit alignment objective, BEACON achieves feature alignment as an implicit result of discrepancy-aware cross-domain co-training.
>
---
#### [new 019] IMPACT: An Implicit Active-Set Augmented Lagrangian for Fast Contact-Implicit Trajectory Optimization
- **分类: cs.RO**

- **简介: 该论文提出IMPACT方法，用于解决接触隐式轨迹优化（CITO）中的数值难题，提升求解效率与控制质量，适用于复杂机器人任务。**

- **链接: [https://arxiv.org/pdf/2605.09127](https://arxiv.org/pdf/2605.09127)**

> **作者:** Jiayun Li; Dejian Gong; Georgia Chalvatzaki
>
> **备注:** Accepted to Robotics: Science and Systems (RSS), 2026
>
> **摘要:** Contact-implicit trajectory optimization (CITO) has attracted growing attention as a unified framework for planning and control in contact-rich robotic tasks. Recent approaches have demonstrated promising results in manipulation and locomotion without requiring a prescribed contact-mode schedule. It is well known that the underlying mathematical programs with complementarity constraints (MPCCs) remain numerically ill-conditioned, and systematic, scalable solution strategies for CITO remain an active area of research. More efficient and principled solvers that can handle contact constraints are therefore essential to broaden the applicability of CITO. In this work, we develop an augmented-Lagrangian approach to CITO for solving MPCC-based CITO with stationarity guarantees. The method can be interpreted as identifying the implicit contact-mode branches on the fly during the trajectory optimization (TO) iterations; we call this approach IMPACT (IMPlicit contact ACtive-set Trajectory optimization). We provide an efficient C++ implementation tailored to trajectory-optimization workloads and evaluate it on the open-source CITO and contact-implicit model predictive control (CI-MPC) benchmarks. On CITO, IMPACT achieves 2.9x-70x speedups over strong baselines (geometric mean 13.8x). On CI-MPC, we show improved control quality for contact-rich trajectories on dexterous manipulation tasks in simulation. Finally, we demonstrate the proposed method on real robotic hardware on a T-shaped object pushing task.
>
---
#### [new 020] Efficient Multi-Robot Motion Planning with Precomputed Translation-Invariant Edge Bundles
- **分类: cs.RO**

- **简介: 该论文属于多机器人运动规划任务，旨在解决多机器人路径规划中的碰撞避免与效率问题。提出KiTE-Extend方法，利用预计算轨迹段提升规划效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2605.09801](https://arxiv.org/pdf/2605.09801)**

> **作者:** Himanshu Gupta; Paul Motter; Aritra Chakrabarty; Rishabh Sodani; Srikrishna Bangalore Raghu; Alessandro Roncone; Bradley Hayes; Zachary Sunberg
>
> **摘要:** Solving multi-robot motion planning (MRMP) requires generating collision-free kinodynamically feasible trajectories for multiple interacting robots. We introduce Kinodynamic Translation-Invariant Edge Bundles or KiTE-Extend, a planner-agnostic action selection mechanism for sampling-based kinodynamic motion planning. KiTE-Extend uses a library of trajectory segments computed offline to guide action selection during online planning, improving the ability of existing planners to identify feasible motion segments without altering state propagation, collision checking, or cost evaluation, and without changing their theoretical guarantees. While KiTE-Extend can modestly improve single-agent planners, its benefits are most clear in the multi-agent setting, where it is able to explore more effectively and significantly improve planning through the dense spatiotemporal constraints introduced by robot-robot interaction. Through experiments on multiple kinodynamic systems and environments, we show that KiTE-Extend reduces planning time and improves scalability across the three most common MRMP paradigms: centralized, prioritized, and conflict-based.
>
---
#### [new 021] ConsistNav: Closing the Action Consistency Gap in Zero-Shot Object Navigation with Semantic Executive Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于零样本目标导航任务，解决导航过程中因动作不一致导致的失败问题。提出ConsistNav框架，通过语义执行机制提升导航稳定性与成功率。**

- **链接: [https://arxiv.org/pdf/2605.09869](https://arxiv.org/pdf/2605.09869)**

> **作者:** Haosen Wang; Zhenyang Li; Yinqiang Zhang; Zongqi He; Lutao Jiang; Kai Li; Yizhou Zhao; Liaoyuan Fan; Wenjian Hou; Tingbang Liang; Yibin Wen; Defeng Gu
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Zero-shot object navigation has advanced rapidly with open-vocabulary detectors, image--text models, and language-guided exploration. However, even after current methods detect a plausible target hypothesis, the agent may still oscillate between exploration and pursuit, or abandon the object near success. We identify this failure mode as an action consistency gap: semantic evidence is repeatedly reinterpreted at each step without persistent commitment across the episode. We introduce ConsistNav, a training-free zero-shot ObjectNav framework built around a semantic executive composed of three coordinated modules: Finite-State Executive Controller stages target pursuit through guarded semantic phases; Persistent Candidate Memory accumulates cross-frame target evidence into stable object hypotheses; and Stability-Aware Action Control suppresses rotational stagnation, ineffective pursuit, and unverified stopping. This design changes neither the detector nor the low-level planner; instead, it controls when semantic evidence should influence navigation and when it should be suppressed or revisited. We conduct extensive experiments on HM3D and MP3D, where ConsistNav achieves state-of-the-art results among compared zero-shot ObjectNav methods and improves SR by 11.4% and SPL by 7.9% over the controlled baseline on MP3D. Ablation studies and real-world deployment experiments further demonstrate the effectiveness and robustness of the proposed executive mechanism.
>
---
#### [new 022] SABER: A Scalable Action-Based Embodied Dataset for Real-World VLA Adaptation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SABER数据集，解决零售场景机器人任务中数据不足的问题。通过真实环境采集多模态动作数据，提升机器人在复杂任务中的表现。**

- **链接: [https://arxiv.org/pdf/2605.09613](https://arxiv.org/pdf/2605.09613)**

> **作者:** Narsimha Menga; Parikshit Sakurikar; Amirreza Rouhi; Satya Sai Reddy; Anirudh Govil; Sri Harsha Chittajallu; Rajat Aggarwal; Anoop Namboodiri; Sashi Reddi
>
> **摘要:** Robotic deployment in real-world environments depends on rich, domain-specific action data as much as on strong model architecture. General-purpose robot foundation models show modest performance in complex unseen tasks such as manipulation in a retail domain when applied out of the box. The root cause is a data gap: retail environments are structurally absent from general robot pretraining distributions, and the path to filling that gap through teleoperation is prohibitively expensive, logistically constrained, and difficult to scale. We introduce SABER, a high-fidelity retail robotics action dataset built from over 100 hours of natural in-store capture across multiple real grocery environments. Egocentric footage from head-mounted cameras records fine-grained hand activity at the point of interaction, while exocentric 360-degree scene footage from DreamVu's ALIA camera simultaneously observes all actors and activities across the entire space. This combination yields a uniquely complete picture of human retail behavior: dexterous hand activity, whole-body motion, and scene dynamics, all captured without staging, scripting, or teleoperation overhead. The SABER corpus contains 44.8K training samples across three action representation streams: 25K latent action sequences via LAPA-style encoding, 18.6K dexterous hand-pose trajectories retargeted to robot joint space, and 1.2K whole-body synchronized motion sequences retargeted to a humanoid embodiment. When applied to GR00T N1.6 via a shared-backbone multi-task post-training recipe, SABER yields a mean success rate of 29.3% across ten retail manipulation tasks -- more than 2.19x over fine-tuning baselines (13.4%). SABER demonstrates that the path to capable retail robots runs through better data, which can be collected today, at scale, without a robot in the loop. The dataset and code are available at this https URL
>
---
#### [new 023] LE-PAVD: Learning-Enhanced Physics-Aware Vehicle Dynamics for High-Speed Autonomous Navigation
- **分类: cs.RO**

- **简介: 该论文属于高精度车辆动力学建模任务，旨在解决高速自动驾驶中的动态预测问题。通过融合物理先验与学习模型，提出LE-PAVD方法，提升预测精度与计算效率。**

- **链接: [https://arxiv.org/pdf/2605.08489](https://arxiv.org/pdf/2605.08489)**

> **作者:** Musabbir Ahmed Arrafi; Malik Ali; Nicholas M. Stiffler; Krishna Bhavithavya Kidambi
>
> **摘要:** Accurate modeling of nonlinear vehicle dynamics is essential for high-speed autonomous racing, where controllers operate at the handling limits. Model-based methods are interpretable but rely on simplifying assumptions, while purely learned models capture nonlinearities yet often lack physical consistency and generalization. We propose LE-PAVD (Learning-Enhanced Physics-Aware Vehicle Dynamics), a hybrid model that integrates physics priors with learned components. Our architecture adds four components: load-sensitive Pacejka tire forces, longitudinal load transfer, lateral tire-force effects, and rate-limited actuator inputs. Trained end-to-end on simulation and real-world telemetry, LE-PAVD enforces physical consistency while improving state prediction accuracy. On an unseen track, LE-PAVD reduces average displacement error (ADE) by 16.1$\%$, final displacement error (FDE) by 20.6$\%$, and lowers yaw-rate root mean squared error (RMSE) by 91.3$\%$ versus a deep dynamics baseline, while using 21.6$\%$ fewer FLOPs and achieving approximately 1.50$\times$ faster inference. In closed-loop simulations, LE-PAVD consistently outperforms the baseline by achieving faster lap times by 17.4$\%$ on a training track and 9.5$\%$ on a test track, without any track boundary violations. Overall, LE-PAVD offers a compact, physics-grounded dynamics backbone that improves predictive fidelity and closed-loop performance while reducing inference cost.
>
---
#### [new 024] Above and Below: Heterogeneous Multi-robot SLAM Across Surface and Underwater Domains
- **分类: cs.RO**

- **简介: 该论文属于多机器人SLAM任务，旨在解决USV与AUV间定位与地图构建问题。通过检测环闭合，实现跨水面与水下机器人的协同定位。**

- **链接: [https://arxiv.org/pdf/2605.09811](https://arxiv.org/pdf/2605.09811)**

> **作者:** John McConnell; Armon Shariati; Paul Szenher; Yaxuan Li
>
> **摘要:** Multi-robot simultaneous localization and mapping (SLAM) is a fundamental task in multi-robot operations. Robots must have a common understanding of their location and that of their team members to complete coordinated actions. However, multi-robot SLAM between Uncrewed Surface Vessels (USVs) and Autonomous Underwater Vehicles (AUVs) has primarily been achieved through acoustic pinging between robots to retrieve range measurements; a measurement technique requires that robots to be in similar locations simultaneously, have an uninterrupted path for signal propagation, and may necessitate synchronized clocks. This is especially challenging in complex, cluttered maritime environments, where structures may impede signals. However, these same structures may be observable above and below the water's surface, presenting an opportunity for inter-robot SLAM loop closure between USV and AUV data streams. This work builds upon recent research on inter-robot SLAM loop closure between USV and AUV data, extending it to propose a centralized multi-robot SLAM system. Each robot performs its state estimation, and we detect loop closures between each AUV and the USV data. These inter-robot loop closures are used to merge each robot's state estimate into a centralized graph, yielding estimates for the whole time history of the USV and all AUVs in the system. Validation is performed using real-world perceptual data in three different environments. Results show improved errors for AUVs in the multi-robot SLAM system compared to single-robot SLAM over the same trajectories. To our knowledge, this is the first instance of a multi-robot SLAM system with AUVs and USVs built on loop closures rather than acoustic distance measurements.
>
---
#### [new 025] Hierarchical Prompting with Dual LLM Modules for Robotic Task and Motion Planning
- **分类: cs.RO**

- **简介: 该论文提出一种分层语言驱动框架，用于机器人任务与运动规划，解决人机交互中的自然指令理解与精准操作问题。通过两个LLM模块实现高、低层任务处理。**

- **链接: [https://arxiv.org/pdf/2605.08330](https://arxiv.org/pdf/2605.08330)**

> **作者:** Karolina Źróbek; Tessa Pulli; Paweł Gajewski; Antonio Galiza Cerdeira Gonzalez; Bipin Indurkhya
>
> **摘要:** We present a hierarchical language-driven framework for robotic task and motion planning to improve natural, intuitive human-robot interaction in service and assistance scenarios. The proposed system employs two large language model (LLM) modules: a high-level planning agent and a low-level spatial reasoning sub-module. The primary agent processes natural language commands and generates action sequences using a ReAct-style prompt, interacting with tools for object perception and manipulation (e.g., pick, place, release). For precise spatial placement, such as interpreting "place the mug next to the plate", a separate sub-prompting module handles 3D reasoning based on object geometry and scene layout. The system integrates YOLOX-GDRNet for object detection and pose estimation, along with a motion execution stub. We evaluated the system in 24 test scenarios, ranging from simple spatial commands to high-level instructions and infeasible requests. The system achieved an overall task success rate of 86%.
>
---
#### [new 026] VEGA: Visual Encoder Grounding Alignment for Spatially-Aware Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出VEGA框架，解决VLA模型空间感知不足的问题。通过直接对齐视觉编码器输出与三维特征，提升空间推理能力，适用于机器人操作任务。**

- **链接: [https://arxiv.org/pdf/2605.10485](https://arxiv.org/pdf/2605.10485)**

> **作者:** Hao Wang; Xiaobao Wei; Jingyang He; Chengyu Bai; Chun-Kai Fan; Jiajun Cao; Jintao Chen; Ying Li; Shanyu Rong; Ming Lu; Xiaozhu Ju; Jian Tang; Shanghang Zhang
>
> **摘要:** Precise spatial reasoning is fundamental to robotic manipulation, yet the visual backbones of current vision-language-action (VLA) models are predominantly pretrained on 2D image data without explicit 3D geometric supervision, resulting in representations that lack accurate spatial awareness. Existing implicit spatial grounding methods partially address this by aligning VLA features with those of 3D-aware foundation models, but they rely on empirical layer search and perform alignment on LLM-level visual tokens where spatial structure has already been entangled with linguistic semantics, limiting both generalizability and geometric interpretability. We propose VEGA (Visual Encoder Grounding Alignment), a simple yet effective framework that directly aligns the output of the VLA's visual encoder with spatially-aware features from DINOv2-FiT3D, a DINOv2 model fine-tuned with multi-view consistent 3D Gaussian Splatting supervision. By performing alignment at the visual encoder output level, VEGA grounds spatial awareness before any linguistic entanglement occurs, offering a more interpretable and principled alignment target. The alignment is implemented via a lightweight projector trained with a cosine similarity loss alongside the standard action prediction objective, and is discarded at inference time, introducing no additional computational overhead. Extensive experiments on simulation benchmark and real-world manipulation tasks demonstrate that VEGA consistently outperforms existing implicit spatial grounding baselines, establishing a new state-of-the-art among implicit spatial grounding methods for VLA models.
>
---
#### [new 027] Constraint-Aware Diffusion Priors for High-Fidelity and Versatile Quadruped Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，解决多源数据下的运动生成与安全部署问题。提出Diff-CAST框架，利用扩散模型提升运动多样性与稳定性，确保真实硬件安全运行。**

- **链接: [https://arxiv.org/pdf/2605.08804](https://arxiv.org/pdf/2605.08804)**

> **作者:** Jianhui Chen; Ruixin Zhan; Liu Liu; Yang Cai; Ziqiao Li
>
> **摘要:** Reinforcement learning combined with imitation learning has significantly advanced biomimetic quadrupedal locomotion. However, scaling these frameworks to massive, multi-source datasets exposes fundamental bottlenecks. First, traditional GAN-based discriminators are prone to mode collapse, struggling to capture diverse motion distributions from uncurated datasets. Second, existing kinematic priors suffer from out-of-distribution (OOD) tracking conflicts, leading to severe unintended heading drifts during complex maneuvers. Furthermore, deploying unconstrained priors to physical hardware poses critical safety risks by disregarding actuator dynamics. To overcome these challenges, we propose Diff-CAST (Diffusion-guided Constraint-Aware Symmetric Tracking), a novel motion prior framework leveraging the multi-modal distribution modeling capabilities of diffusion models for stylistic rewards. Diff-CAST effectively replaces traditional GAN discriminators, unlocking robust data scaling on heterogeneous collections. To ensure high-fidelity intent execution and reliable real-world deployment, we introduce a comprehensive Sim2Re architecture integrating Symmetric Augmented Command Conditioning (SACC) for drift-free tracking, and Constrained RL for hardware safety. Experiments on a quadruped demonstrate that Diff-CAST mitigates mode collapse, enables seamless transitions between diverse skills, and ensures robust, hardware-compliant locomotion.
>
---
#### [new 028] Neural Distance-Guided Path Integral Control for Tractor-Trailer Navigation
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，旨在解决拖拉机-挂车系统在复杂农业环境中的安全路径规划问题。通过神经网络与MPPI控制器结合，实现实时、地图无关的几何推理与控制。**

- **链接: [https://arxiv.org/pdf/2605.09939](https://arxiv.org/pdf/2605.09939)**

> **作者:** Peng Wei; Chen Peng; Stavros Vougioukas
>
> **摘要:** Autonomous and safe navigation of tractor-trailer systems requires accurate, real-time collision avoidance and dynamically feasible control, particularly in cluttered and complex agricultural environments. This is challenging due to their articulated, deformable geometries and nonlinear dynamics. Traditional methods oversimplify vehicle geometry or rely on precomputed distance fields that assume a known map, limiting their applicability in dynamic, partially unknown environments. To address these limitations, we propose a geometric neural encoder that provides fast and accurate distance estimates between the full tractor-trailer body and raw LiDAR perception, enabling real-time, map-free geometric reasoning. These learned distances are integrated into a Model Predictive Path Integral (MPPI) controller, allowing the system to incorporate true articulated geometry directly into its cost evaluation and enabling more responsive navigation in challenging agricultural settings. Simulation results demonstrate that the proposed framework generates dynamically feasible and safe trajectories for navigating tractor-trailer systems in cluttered and complex environments.
>
---
#### [new 029] Muninn: Your Trajectory Diffusion Model But Faster
- **分类: cs.RO; cs.PF; eess.SY**

- **简介: 该论文提出Muninn，解决扩散轨迹规划器速度慢的问题，通过缓存和不确定性预算提升效率，保持轨迹质量与安全。**

- **链接: [https://arxiv.org/pdf/2605.09999](https://arxiv.org/pdf/2605.09999)**

> **作者:** Gokul Puthumanaillam; Hao Jiang; Ruben Hernandez; Jose Fuentes; Paulo Padrao; Leonardo Bobadilla; Melkior Ornik
>
> **备注:** Accepted to Robotics: Science and Systems 2026
>
> **摘要:** Diffusion-based trajectory planners can synthesize rich, multimodal robot motions, but their iterative denoising makes online planning and control prohibitively slow. Existing accelerations either modify the sampler or compress the network--sacrificing plan quality or requiring retraining without accounting for downstream control risk. We address the problem of making diffusion-based trajectory planners fast enough for real-time robot use without retraining the model or sacrificing trajectory quality, and in a way that works across diverse state-space diffusion architectures. Our key insight is that diffusion trajectory planners expose two signals we can exploit: a cheap probe of how their internal trajectory representation changes across steps, and analytic coefficients that describe how denoiser errors affect the sampler's state update. By calibrating the first signal against the second on offline runs, we obtain a per-step score that upper-bounds how far the final trajectory can deviate when we reuse a cached denoiser output, and we treat this bound as an uncertainty budget that we can spend over the denoising process. Building on this insight, we present Muninn, a training-free caching wrapper that tracks this uncertainty budget during sampling and, at each diffusion step, chooses between reusing a cached denoiser output when the predicted deviation is small and recomputing the denoiser when it is not. Across standard benchmarks Muninn delivers up to 4.6x wall-clock speedups across several trajectory diffusion models by reducing denoiser evaluations, while preserving task performance and safety metrics. Muninn further certifies that cached rollouts remain within a specified distance of their full-compute counterparts, and we validate these gains in real-time closed-loop navigation and manipulation hardware deployments. Project page: this https URL.
>
---
#### [new 030] Minimizing Worst-Case Weighted Latency for Multi-Robot Persistent Monitoring: Theory and RL-Based Solutions
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究多机器人持续监控任务，旨在最小化节点加权最坏情况延迟。提出新的性能指标和强化学习方法，优化机器人轨迹。**

- **链接: [https://arxiv.org/pdf/2605.09633](https://arxiv.org/pdf/2605.09633)**

> **作者:** Weizhen Wang; Ziheng Wang; Jianping He; Xinping Guan; Xiaoming Duan
>
> **摘要:** We study multi-robot persistent monitoring on weighted graphs, where node weights encode monitoring priorities and edge weights encode travel distances. The goal is to design joint robot trajectories that minimize the worst-case weighted latency across all nodes over an infinite time horizon. The widely adopted worst-case latency objective evaluates team performance over the entire time horizon and therefore may fail to distinguish strategies with poor transient behavior but strong asymptotic performance. To address this limitation, we propose a family of tail-performance objectives that generalize the standard objective and study the resulting functional optimization problems. We establish several key theoretical properties, including the existence of optimal strategies, relationships among the proposed objectives and their corresponding optimization problems, approximation by periodic solutions to arbitrary accuracy, and reductions to event-driven decision models with discretized waiting times. Building on these results, we construct an equivalent event-driven Markov decision process (MDP), called the Tail Worst-case Latency-Optimizing Markov Decision Process (TWLO-MDP), which reformulates the tail-performance objective as a standard average-reward criterion. We then develop reinforcement-learning-based solution methods for the TWLO-MDP and introduce the multi-robot monitoring benchmark (M2Bench), a unified platform that supports the evaluation and comparison of heuristic and learning-based monitoring algorithms. Experiments on synthetic and realistic monitoring scenarios show that our methods effectively reduce the worst-case weighted latency and outperform representative baselines.
>
---
#### [new 031] RePO-VLA: Recovery-Driven Policy Optimization for Vision-Language-Action Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RePO-VLA，解决视觉-语言-动作模型在长周期操作中的鲁棒性问题，通过恢复驱动策略优化提升成功率。**

- **链接: [https://arxiv.org/pdf/2605.09410](https://arxiv.org/pdf/2605.09410)**

> **作者:** Weijia Liufu; Xiaoyu Guo; Ruiyi Chen; Jingzhi Liu; Kaidong Zhang; Xiwen Liang; Jianqi Lin; Dawei Sun; Yuze Wang; Rongtao Xu; Bingqian Lin; Bowen Yang; Tongtong Cao; Bowen Peng; Dongyu Zhang; Guangrun Wang; Min Wang; Liang Lin; Xiaodan Liang
>
> **摘要:** Vision-Language-Action (VLA) models remain brittle in long-horizon, contact-rich manipulation because success-only imitation provides little supervision for execution drift, while failed rollouts are often discarded. We introduce RePO-VLA, a recovery-driven policy optimization framework that assigns distinct roles to success, recovery, and failure trajectories. RePO-VLA first applies Recovery-Aware Initialization (RAI), slicing recovery segments and resetting history so corrective actions depend on the current adverse state rather than the preceding failure. It then learns a Progress-Aware Semantic Value Function (PAS-VF), aligning spatiotemporal trajectory features with instructions and successful references. The resulting labels salvage useful failure prefixes via reliability decay, while low-value labels mark drift and terminal breakdowns, teaching differences among nominal, failed, and corrective actions. The data engine turns adverse states into planner-generated or human-collected corrective rollouts, teaching recovery to the success manifold. Value-Conditioned Refinement (VCR) trains the policy to prefer high-progress actions. At deployment, a fixed high value ($v=1.0$) biases actions toward the learned success manifold without online failure detectors or heuristic retries. We introduce FRBench, with standardized error injection and recovery-focused evaluation. Across simulated and real-world bimanual tasks, RePO-VLA improves robustness, raising adversarial success from 20% to 75% on average and up to 80% in scaled real-world trials.
>
---
#### [new 032] Beyond Self-Play: Hierarchical Reasoning for Continuous Motion in Closed-Loop Traffic Simulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于交通仿真任务，旨在解决自博弈方法在行为真实性上的不足。提出分层架构，结合多智能体推理与连续运动生成，提升控制平滑性和安全性。**

- **链接: [https://arxiv.org/pdf/2605.09153](https://arxiv.org/pdf/2605.09153)**

> **作者:** Weifan Zhang; Xiaofeng Zhao; Adel Bazzi; Mingrui Li; Yifan Wei; Dengfeng Sun
>
> **备注:** Submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Closed-loop traffic simulation requires agents that are both scalable and behaviorally realistic. Recent self-play reinforcement learning approaches demonstrate strong scalability, but their equilibrium strategies fail to capture the socially aware behaviors of real human drivers. We propose a hierarchical architecture that goes beyond self-play by combining high-level multi-agent interaction reasoning with low-level continuous trajectory realization. Specifically, a Stackelberg-style Multi-Agent Reinforcement Learning (MARL) module generates interaction-aware intention commands. These commands condition a low-level continuous motion module, translating the strategic intent into physically consistent, scene-responsive control sequences. To mitigate distribution shift in closed-loop deployment, we introduce a hybrid co-training scheme combining MARL with auxiliary recovery supervision. Experiments on a SUMO-based urban network demonstrate that the proposed framework achieves superior control smoothness and safety compared to self-play and passive imitation baselines, while maintaining competitive traffic efficiency.
>
---
#### [new 033] Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning
- **分类: cs.RO**

- **简介: 该论文属于机器人长期任务规划领域，解决指令漂移问题。通过引入CAPS框架和SNR机制，提升模型在长时序任务中的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.09537](https://arxiv.org/pdf/2605.09537)**

> **作者:** Kewei Chen; Yayu Long; Mingsheng Shang
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Despite rapid progress in Vision-Language-Action (VLA) models for robotic control, instruction drift remains a persistent failure mode in long-horizon tasks. This paper reconceptualizes this phenomenon, positing that instruction drift is fundamentally a systematic sampling error: local greedy sampling is prone to collapsing into "Negative Pivotal Windows"--irreversible local optima with high local probability that sever global success pathways. To address this, we propose Context-Aware Power Sampling (CAPS), a training-free inference-time computation framework. CAPS leverages power distributions to sharpen global trajectory probabilities, enabling lookahead search over the model's conditional generative trajectory distribution. Furthermore, we introduce a metacognitive control mechanism based on Signal-to-Noise Ratio (SNR). This mechanism triggers adaptive MCMC search solely when drift risk is detected, enabling a dynamic transition from "intuitive fast thinking" to "rational slow search." Experiments on RoboTwin, Simpler-WindowX, and Libero-long benchmarks show that CAPS achieves substantial improvements over strong baselines, including OpenVLA and TACO, without parameter updates. These results support the effectiveness of adaptive inference-time computation for improving long-horizon robustness in embodied control.
>
---
#### [new 034] Model-Reference Adaptive Flight Control of the 95-mg Bee++
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于飞行控制任务，旨在解决微小型无人机的高精度位置跟踪问题。通过设计MRAC架构，提升Bee++的控制性能。**

- **链接: [https://arxiv.org/pdf/2605.08525](https://arxiv.org/pdf/2605.08525)**

> **作者:** Francisco M. F. R. Gonçalves; Conor K. Trygstad; Néstor O. Pérez-Arancibia
>
> **备注:** Extended abstract to appear in the proceedings of the LSU Symposium on Control, Learning, and Intelligent Systems
>
> **摘要:** We introduce a model-reference adaptive control (MRAC) architecture for high-performance positional tracking of the Bee++, a 95-mg insect-scale flapping-wing aerial vehicle. The suitability, functionality, and high performance of the proposed approach are demonstrated using data from real-time flight experiments.
>
---
#### [new 035] PriorVLA: Prior-Preserving Adaptation for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决VLA模型适应下游任务时先验信息丢失的问题。提出PriorVLA框架，保留预训练先验并有效适应下游任务。**

- **链接: [https://arxiv.org/pdf/2605.10925](https://arxiv.org/pdf/2605.10925)**

> **作者:** Xinyu Guo; Bin Xie; Wei Chai; Xianchi Deng; Tiancai Wang; Zhengxing Wu; Xingyu Chen
>
> **备注:** 32 pages. Project page: this https URL
>
> **摘要:** Large-scale pretraining has made Vision-Language-Action (VLA) models promising foundations for generalist robot manipulation, yet adapting them to downstream tasks remains necessary. However, the common practice of full fine-tuning treats pretraining as initialization and can shift broad priors toward narrow training-distribution patterns. We propose PriorVLA, a novel framework that preserves pretrained priors and learns to leverage them for effective adaptation. PriorVLA keeps a frozen Prior Expert as a read-only prior source and trains an Adaptation Expert for downstream specialization. Expert Queries capture scene priors from the pretrained VLM and motor priors from the Prior Expert, integrating both into the Adaptation Expert to guide adaptation. Together, PriorVLA updates only 25% of the parameters updated by full fine-tuning. Across RoboTwin 2.0, LIBERO, and real-world tasks, PriorVLA achieves stronger overall performance than full fine-tuning and state-of-the-art VLA baselines, with the largest gains under out-of-distribution (OOD) and few-shot settings. PriorVLA improves over pi0.5 by 11 points on RoboTwin 2.0-Hard and achieves 99.1% average success on LIBERO. Across eight real-world tasks and two embodiments, PriorVLA reaches 81% in-distribution (ID) and 57% OOD success with standard data. With only 10 demonstrations per task, PriorVLA reaches 48% ID and 32% OOD success, surpassing pi0.5 by 24 and 22 points, respectively.
>
---
#### [new 036] Preserving Foundational Capabilities in Flow-Matching VLAs through Conservative SFT
- **分类: cs.RO**

- **简介: 该论文针对视觉-语言-动作模型的微调问题，提出ConSFT方法，解决参数过度更新导致能力退化的问题，通过动态调整学习信号保持模型原有能力。**

- **链接: [https://arxiv.org/pdf/2605.08879](https://arxiv.org/pdf/2605.08879)**

> **作者:** Tianyi Zhang; Shaopeng Zhai; Haoran Zhang; Fuxian Huang; Qi Zhang
>
> **备注:** 20 pages, 9 figures
>
> **摘要:** Unconstrained fine-tuning of flow-matching Vision-Language-Action (VLA) models drives dense parameter overwrites, degrading pre-trained capabilities. We present Conservative Supervised Fine-Tuning (ConSFT), an optimization objective that adapts to target distributions while mitigating catastrophic forgetting, requiring zero prior data or architectural overhead. By dynamically scaling learning signals based on model confidence, ConSFT suppresses excessive gradients from low-confidence samples to prevent disproportionate parameter updates, thereby bounding the intrinsic parameter disruption risk. Inspired by reinforcement learning's trust-region clipping, this formulation establishes a progressive learning dynamic to secure target convergence and prior capability retention, maintaining sparse parameter updates without relying on the parallel reference networks required by explicit regularization. We evaluate ConSFT on the LIBERO and RoboTwin benchmarks across state-of-the-art flow-matching VLAs ($\pi_0$, $\pi_{0.5}$, and GR00T-N1.6-3B). The method outperforms vanilla SFT in capability retention by an average absolute margin of over 20\%, matching the efficacy of data-heavy Experience Replay in a prior-data-free regime. Real-world robotic deployments confirm that ConSFT precludes spatial overfitting during downstream adaptation, preserving pre-trained physical skills while acquiring sequential target tasks.
>
---
#### [new 037] Mismatch-Aware Adaptive Constraint Tightening for Bicycle-Model Trajectory Optimization
- **分类: cs.RO**

- **简介: 该论文属于自主车辆轨迹优化任务，解决模型不匹配导致的安全约束失效问题。通过分析动态特性，提出MACT方法，实现自适应约束收紧，提升安全性与效率。**

- **链接: [https://arxiv.org/pdf/2605.09376](https://arxiv.org/pdf/2605.09376)**

> **作者:** Lingxue Lyu; Zihui Liu
>
> **摘要:** Trajectory optimization for autonomous vehicles usually relies on the kinematic bicycle model because of its computational simplicity. However, when the planned trajectory is executed under the true vehicle dynamics, which include lateral slip, tire stiffness and yaw-lateral coupling, safety constraints can be violated owing to the model mismatch. In this paper, we make three theoretical contributions. First, we derive a characteristic speed $v_c=\sqrt{C_\alpha L/M}$ which separates two different mismatch regimes: below $v_c$ the dynamic bicycle initially oversteers inward (safe); above $v_c$ it understeers outward (safety-critical). Second, we prove that the peak outward deviation $\varepsilon^*$ follows a $T^2$ horizon scaling whose coefficient transitions between a transient bound $\frac{1}{2}(v^2-v_c^2)\kappa$ and a steady-state bound. Third, we obtain a simulation-free analytical coefficient $a_2^{\mathrm{anal}}=\frac{1}{2}(1-v_c^2/v_{\max}^2)T^2$ that is computable from vehicle parameters and the planning horizon alone. Putting these together, we propose Mismatch-Aware Adaptive Constraint Tightening (MACT), $\epsilon(v,\kappa)=a_2 v^2|\kappa|$, which replaces a fixed worst-case margin by a state-dependent one that is large at high speed/curvature but nearly zero on gentle paths. Eight numerical experiments confirm the scaling laws. MACT reaches 100% safety with 84% less wasted margin than a fixed-margin baseline on the 2-DOF vehicle, extends to a nonlinear leaning bicycle, and in a closed-loop direct-shooting MPC comparison it cuts the applied margin by 34% compared with tube MPC while keeping the same safety.
>
---
#### [new 038] HyDRA Scorpion: A Cost-effective and Modular ROV for Real-Time Underwater Inspection, Intervention, and Object Detection
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于水下机器人任务，旨在解决高成本和智能化不足的问题。设计了低成本、模块化的HyDRA Scorpion ROV，集成AI感知与测量功能，提升水下检测与操作能力。**

- **链接: [https://arxiv.org/pdf/2605.09093](https://arxiv.org/pdf/2605.09093)**

> **作者:** Anika Tabassum Orchi; Md Farhan Zaman; Md Darain Khan; Md Alamgir; Mahbubul Islam; Md. Jobayer Rahman; A. M. Zayed Abdullah; Md Mehrab Hossain Khan; Md. Kutub Al Baki; Iftekharul Islam; Shakil Ahmed; Md Sadique Hossain; Md Muzahidul Islam; Shah Mohammad Seaman; Nusrat Jahan Piyal; Shekh Md. Saifur Rahman; Fahim Hafiz; A.K.M. Muzahidul Islam; M. Rezwan Khan
>
> **备注:** 9 Pages, 11 figures, Research Paper by UIU Mariner Team
>
> **摘要:** A Remotely Operated Vehicle (ROV) is a tethered underwater robot used for tasks like inspection and intervention. While essential tools for underwater science, the high cost of commercial ROVs and a persistent gap between mechanically capable platforms and those with integrated intelligence create a significant barrier to access. HyDRA Scorpion differs from conventional systems by addressing these challenges, integrating an advanced, AI-driven perception stack with in-situ measurement capabilities onto a low-cost, locally manufacturable platform. The system combines 4-DoF maneuverability, dual manipulators, and a custom pressure-tested housing. Experimental results validate the system's robustness and performance. Leak-free operation was confirmed through prolonged pressure testing of the electronics housing to 4 bar, equivalent to the pressure of a 304.8-meter water depth approximately in a simulated environment, with no moisture ingress detected. The vehicle also demonstrated stable station-keeping, maintaining its position within a tight tolerance of $\(\pm\)0.15$ meters under external disturbances. The onboard AI module achieved underwater object detection mean Average Precision (mAP) of 0.89 with real-time inference, length and 3D-mapping based distance measurement. Also, 4-DoF manipulator arm can grip and maintain dual-function manipulator feature which support 360 degree tangle-free rotation.
>
---
#### [new 039] Geometry Guided Self-Consistency for Physical AI
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于物理AI任务，解决动作生成中因随机性导致的稳定性问题。通过并行生成多个动作轨迹，聚类选择最优解，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2605.08638](https://arxiv.org/pdf/2605.08638)**

> **作者:** Yinwei Dai; Zhuofu Chen; Lijie Yang; Ravi Netravali
>
> **摘要:** State-of-the-art physical AI models generate a chunk of actions per inference through diffusion or flow matching, iteratively refining an initial noise sample into an action trajectory. Because this inference process is inherently stochastic, committing to a single trajectory per round is brittle, and this brittleness compounds across the many sequential rounds that comprise a complete episode. We introduce KeyStone, an inference-time self-consistency method for diffusion-based action generation that draws $K$ candidate action chunks in parallel from a shared model context, clusters them in continuous action space, and returns the medoid of the largest cluster -- no additional model required. Two properties make this practical. First, the compact nature of action trajectories makes diffusion inference memory-bandwidth bound, leaving spare compute capacity to run $K$ chains in parallel with no additional wall-clock latency. Second, unlike token or pixel spaces where distance carries no semantic meaning and selection requires a learned judge, action chunks are geometrically structured such that Euclidean distance directly reflects physical similarity, making selection principled and judge-free. Across diverse vision-language-action models (VLAs) and world-action models (WAMs), KeyStone improves task success rates by up to \textbf{13.3\%} over single-trajectory sampling with negligible latency overhead, while having on par accuracy with model-based selectors at no training cost. We open source KeyStone at this https URL.
>
---
#### [new 040] Safety-Critical LiDAR-Inertial Odometry with On-Manifold Deterministic Protection Level
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决机器人在线安全评估问题。提出一种基于LiDAR与惯性测量的里程计系统，提供确定性保护水平，确保安全操作。**

- **链接: [https://arxiv.org/pdf/2605.09383](https://arxiv.org/pdf/2605.09383)**

> **作者:** Yueqi Zhu; Yan Pan; Chufan Rui; Jiasheng Luo; Shihua Li; Bo Zhou
>
> **摘要:** In safety-critical scenarios, the protection level of the autonomous navigation system is crucial for enabling mobile robots to perform safe tasks. However, existing studies on probabilistic navigation systems for robots usually perform offline accuracy evaluations using limited datasets and assume that the results can be applied to unknown real-world environments. As a result, current autonomous mobile robots often lack protection levels for online safety assessment. To fill this gap, we propose a safety-critical LiDAR-inertial odometry (LIO) that provides deterministic protection levels based on on-manifold deterministic state estimation. By adopting the unknown but bounded assumption, we derive a neat closed-form relationship between point cloud noise and the uncertainty of the estimation from the iterated closest point algorithm. Using this relationship, we design an on-manifold ellipsoidal set-membership filter and implement it within the LIO system. Leveraging the properties of the set-membership filter, our system offers the feasible sets of the estimated locations as the deterministic protection levels, serving as safety references for the robots' downstream autonomous operations. The experimental results show that our system can provide effective deterministic online safety references for diverse robots in various environments.
>
---
#### [new 041] Omni-scale Learning-based Sequential Decision Framework for Order Fulfillment of Tote-handling Robotic Systems
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于订单履行任务，解决托盘式机器人系统的顺序决策问题。提出一种融合优化与强化学习的通用框架，提升效率并降低能耗。**

- **链接: [https://arxiv.org/pdf/2605.08758](https://arxiv.org/pdf/2605.08758)**

> **作者:** Jiaxin Liu; Peng Yang; Yuping Li; Xinyue Xie
>
> **备注:** 35 pages, 5 figures
>
> **摘要:** Driven by the rapid expansion of e-commerce and small-batch production, the size of the intralogistics load unit of finished goods, semi-finished goods and raw materials is steadily shrinking. Totes are gradually replacing pallets as the primary handling and storage container. This shift has propelled tote-handling robotic systems to the forefront of automation order fulfillment centers. The order-fulfillment decisions of tote-handling robotic systems share a common order-tote-robot sequential decision-making nature. Existing studies primarily focus on decision mechanisms tailored to particular systems, making it difficult to generalize or transfer them to other contexts. We propose an Omni-scale Learning-based Sequential Decision Framework for Order Fulfillment of Tote-handling Robotic Systems (OLSF-TRS), a generalized and scalable sequential decision framework that combines structured combinatorial optimization with multi-agent reinforcement learning to coordinate order,tote, and robot decisions. On small-scale tote-handling robotic systems, OLSF-TRS achieves near-optimal performance with average optimality gaps below 3.5% across two distinct system configurations. In large-scale scenarios, OLSF-TRS consistently outperforms heuristic baselines across two different system types, reducing total tote movements by 8-12% and over 30% compared to SOTA rule-based approaches, while maintaining real-time responsiveness. These improvements translate into tangible operational benefits, including cost reduction, lower energy consumption, and enhanced throughput stability. The proposed framework delivers an efficient and unified order fulfillment decision-making framework for widely deployed tote-handling robotic systems,supporting high-quality order fulfillment in both e-commerce and industrial logistics sectors.
>
---
#### [new 042] ASACK : Adaptive Safe Active Continual Koopman Learning for Uncertain Systems with Contractive Guarantees
- **分类: cs.RO**

- **简介: 该论文属于控制与学习任务，解决不确定系统中Koopman模型的在线适应问题。提出ASACK框架，实现安全、高效的模型持续优化与控制。**

- **链接: [https://arxiv.org/pdf/2605.09659](https://arxiv.org/pdf/2605.09659)**

> **作者:** Chandan Kumar Sah; Rajpal Singh; Jishnu Keshavan
>
> **摘要:** Koopman operator theory provides a powerful framework for representing nonlinear dynamics through a linear operator acting on lifted observables, enabling the use of linear control techniques for nonlinear systems. However, Koopman models are typically learned from data and often degrade in performance under model uncertainty and distributional shifts between training and deployment. Although several works have explored online adaptation to address this issue, many rely on neural network-based updates that introduce significant computational overhead and lack formal safety guarantees, limiting their suitability for real-time and safety-critical robotic applications. In this work, we propose a unified framework for continual adaptive Koopman learning that enables safe and efficient online refinement of learned models during task execution. An autoencoder-based Koopman model is first learned offline and subsequently refined online through a contractive adaptation law, which provides theoretical convergence guarantees under distributional shifts and model uncertainty. To improve data efficiency and accelerate model refinement, the adaptation mechanism is integrated with an active learning strategy that drives the system to collect informative data while accomplishing task objectives. The resulting control problem is formulated as a nonconvex optimization problem incorporating both active learning objectives and safety constraints. We further derive theoretical bounds on model approximation error and show how these bounds can be incorporated within a robust Model Predictive Control (MPC) framework to provide formal safety guarantees. The proposed approach unifies learning, excitation, and safety within a single control framework without sacrificing real-time feasibility. Extensive simulation and experimental studies demonstrate superior performance compared to state-of-the-art baselines.
>
---
#### [new 043] A Visuo-Tactile Data Collection System with Haptic Feedback for Coarse-to-Fine Imitation Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人模仿学习任务，旨在解决传统系统缺乏触觉反馈的问题。通过设计触觉数据采集系统，实现精细力控演示的收集，支持粗到细的学习算法。**

- **链接: [https://arxiv.org/pdf/2605.08757](https://arxiv.org/pdf/2605.08757)**

> **作者:** Yeseung Kim; Nayoung Oh; Jun Park; Teetat Thamronglak; Daehyung Park
>
> **摘要:** We present a visuo-tactile data-collection system that generates temporally structured, contact-rich demonstrations for imitation learning. Conventional systems often decouple the operator from contact forces, which hinders the demonstration of subtle force modulation. Our system introduces a direct-drive gripper that the operator actuates with the fingers, preserving natural haptic feedback. Integrated visual sensors and custom tactile arrays capture image streams and contact geometry. A handle-mounted push button enables the operator to annotate the task's temporal structure in real time by marking task-critical regions. By fusing in-hand force perception with in-situ temporal annotation, the system produces multimodal datasets designed for coarse-to-fine learning algorithms that exploit structural task knowledge, enabling the development of high-quality manipulation policies.
>
---
#### [new 044] EFGCL: Learning Dynamic Motion through Spotting-Inspired External Force Guided Curriculum Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人运动学习任务，旨在解决腿式机器人动态运动学习效率低的问题。提出EFGCL方法，通过物理引导提高探索效率，实现复杂动作的快速学习与部署。**

- **链接: [https://arxiv.org/pdf/2605.10063](https://arxiv.org/pdf/2605.10063)**

> **作者:** Keita Yoneda; Kento Kawaharazuka; Kei Okada
>
> **备注:** Accepted at RA-L 2026, website - this https URL, YouTube - this https URL
>
> **摘要:** Learning dynamic whole-body motions for legged robots through reinforcement learning (RL) remains challenging due to the high risk of failure, which makes efficient exploration difficult and often leads to unstable learning. In this paper, we propose External Force Guided Curriculum Learning (EFGCL), a guided RL approach based on the principle of physical guidance, in which external assistive forces are introduced during training. Inspired by spotting in artistic gymnastics, EFGCL enables agents to physically experience successful motion executions without relying on task-specific reward shaping or reference trajectories. Experiments on a quadrupedal robot performing Jump, Backflip, and Lateral-Flip tasks demonstrate that EFGCL accelerates learning of the Jump task by approximately a factor of two and enables the acquisition of complex whole body motions that conventional RL methods fail to learn. We further show that the learned policies can be deployed on real robot, reproducing motions consistent with those observed in simulation. These results indicate that physically guided exploration, which allows agents to experience success early in training, is an effective and general strategy for improving learning efficiency in dynamic whole-body motion tasks.
>
---
#### [new 045] Nano-U: Efficient Terrain Segmentation for Tiny Robot Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人导航任务，解决微控制器上地形分割的高效实现问题。提出Nano-U网络，结合量化感知蒸馏训练，在低功耗设备上实现快速准确的二值地形分割。**

- **链接: [https://arxiv.org/pdf/2605.10210](https://arxiv.org/pdf/2605.10210)**

> **作者:** Federico Pizzolato; Francesco Pasti; Nicola Bellotto
>
> **备注:** Code repository: this https URL
>
> **摘要:** Terrain segmentation is a fundamental capability for autonomous mobile robots operating in unstructured outdoor environments. However, state-of-the-art models are incompatible with the memory and compute constraints typical of microcontrollers, limiting scalable deployment in small robotics platforms. To address this gap, we develop a complete framework for robust binary terrain segmentation on a low-cost microcontroller. At the core of our approach we design Nano-U, a highly compact binary segmentation network with a few thousand parameters. To compensate for the network's minimal capacity, we train Nano-U via Quantization-Aware Distillation (QAD), combining knowledge distillation and quantization-aware training. This allows the final quantized model to achieve excellent results on the Botanic Garden dataset and to perform very well on TinyAgri, a custom agricultural field dataset with more challenging scenes. We deploy the quantized Nano-U on a commodity microcontroller by extending MicroFlow, a compiler-based inference engine for TinyML implemented in Rust. By eliminating interpreter overhead and dynamic memory allocation, the quantized model executes on an ESP32-S3 with a minimal memory footprint and low latency. This compiler-based execution demonstrates a viable and energy-efficient solution for perception on low-cost robotic platforms.
>
---
#### [new 046] HeteroGenManip: Generalizable Manipulation For Heterogeneous Object Interactions
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出HeteroGenManip，解决机器人对异质物体交互的泛化操作问题。通过分阶段框架，提升抓取精度和交互效果，显著提高任务成功率。**

- **链接: [https://arxiv.org/pdf/2605.10201](https://arxiv.org/pdf/2605.10201)**

> **作者:** Zhenhao Shen; Zeming Yang; Yue Chen; Yuran Wang; Shengqiang Xu; Mingleyang Li; Hao Dong; Ruihai Wu
>
> **摘要:** Generalizable manipulation involving cross-type object interactions is a critical yet challenging capability in robotics. To reliably accomplish such tasks, robots must address two fundamental challenges: ``where to manipulate'' (contact point localization) and ``how to manipulate'' (subsequent interaction trajectory planning). Existing foundation-model-based approaches often adopt end-to-end learning that obscures the distinction between these stages, exacerbating error accumulation in long-horizon tasks. Furthermore, they typically rely on a single uniform model, which fails to capture the diverse, category-specific features required for heterogeneous objects. To overcome these limitations, we propose HeteroGenManip, a task-conditioned, two-stage framework designed to decouple initial grasp from complex interaction execution. First, Foundation-Correspondence-Guided Grasp module leverages structural priors to align the initial contact state, thereby significantly reducing the pose uncertainty of grasping. Subsequently, Multi-Foundation-Model Diffusion Policy (MFMDP) routes objects to category-specialized foundation models, integrating fine-grained geometric information with highly-variable part features via a dual-stream cross-attention mechanism. Experimental evaluations demonstrate that HeteroGenManip achieves robust intra-category shape and pose generalization. The framework achieves an average 31\% performance improvement in simulation tasks with broad type setting, alongside a 36.7\% gain across four real-world tasks with different interaction types.
>
---
#### [new 047] HiDrive: A Closed-Loop Benchmark for High-Level Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决现有基准测试不足的问题。提出HiDrive基准，增强长尾场景和高级驾驶能力评估。**

- **链接: [https://arxiv.org/pdf/2605.09972](https://arxiv.org/pdf/2605.09972)**

> **作者:** Zhongyu Xia; Guanyu Zhu; Guo Tang; Wenhao Chen; Yongtao Wang
>
> **摘要:** End-to-end autonomous driving has witnessed rapid progress, yet existing benchmarks are increasingly saturated, with state-of-the-art models achieving near-perfect scores on widely used open-loop and closed-loop benchmarks. This saturation does not mean that the problem has been solved; instead, it reveals that current benchmarks remain limited in scenario diversity, object variety, and the breadth of driving capabilities they evaluate. In particular, they lack sufficient long-tail scenarios involving rare but safety-critical objects and fail to assess advanced decision-making such as legal compliance, ethical reasoning, and emergency response. To address these gaps, we propose HiDrive, a new closed-loop benchmark for end-to-end autonomous driving that emphasizes long-tail scenarios and a richer evaluation of driving capabilities. HiDrive introduces a diverse set of rare objects and uncommon traffic situations, and expands evaluation from basic driving skills to more advanced capabilities, including rule compliance, moral reasoning, and context-dependent emergency maneuvers. Correspondingly, we extend previous collision-avoidance-centered metrics into a comprehensive evaluation system that encompasses collision and braking, traffic-rule compliance, and moral-reasoning indicators. Built on a more advanced physics engine, HiDrive provides physically realistic lighting and high-fidelity visual rendering, offering a more challenging and realistic testbed for assessing whether autonomous driving systems can handle the complexity of real-world deployment. The HiDrive software, source code, digital assets, and documentation are available at this https URL.
>
---
#### [new 048] Towards Generative Predictive Display for Vision-Based Teleoperation: A Zero-Shot Benchmark of Off-the-Shelf Video Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉遥操作任务，旨在解决通信延迟导致的感知与控制问题。通过评估现有生成视频模型在预测显示中的表现，发现其无法满足实时性与准确性要求。**

- **链接: [https://arxiv.org/pdf/2605.09670](https://arxiv.org/pdf/2605.09670)**

> **作者:** Aws Khalil; Jaerock Kwon
>
> **摘要:** Teleoperation systems are fundamentally limited by communication latency, which degrades situational awareness and control performance. Predictive display aims to mitigate this limitation by presenting an estimate of the current visual state rather than delayed observations. While recent advances in generative video models enable high-quality video synthesis, their suitability for latency-sensitive predictive display remains unclear. This paper presents a zero-shot benchmark of off-the-shelf generative video models for short-horizon predictive display, without task-specific fine-tuning. We formulate the problem as rollout-based future frame prediction and develop a unified benchmarking pipeline using simulated driving data from the CARLA simulator. Five publicly released video models spanning transformer-based and diffusion-based families are evaluated across two resolutions and two conditioning regimes (multi-frame and single-frame). Performance is assessed using prediction accuracy (mean absolute difference), per-rollout latency, peak GPU memory usage, and temporal error evolution across the prediction horizon. On this zero-shot benchmark, no tested model simultaneously achieves low rollout error, non-divergent per-step error behavior, and real-time inference at the source frame rate. Increasing model scale or resolution yields limited and, in some cases, inverted improvements. These findings highlight a gap between general-purpose generative video synthesis and the requirements of predictive display in teleoperation, suggesting that practical deployment will require either explicit short-horizon temporal supervision, in-domain adaptation, or aggressive inference optimization rather than direct application of off-the-shelf models. Code, configurations, and qualitative results are released on the project page: this https URL
>
---
#### [new 049] ALAM: Algebraically Consistent Latent Transitions for Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出ALAM模型，解决VLA模型中因缺乏动作标注数据导致的难题。通过视频中的时间关系学习结构化潜在转移，提升策略生成效果。**

- **链接: [https://arxiv.org/pdf/2605.10819](https://arxiv.org/pdf/2605.10819)**

> **作者:** Zuojin Tang; Haoyun Liu; Xinyuan Chang; Changjie Wu; Dongjie Huo; Yandan Yang; Bin Liu; Zhejia Cai; Feng Xiong; Mu Xu; jiachen Luo; De Ma; Zhiheng Ma; Gang Pan
>
> **摘要:** Vision-language-action (VLA) models remain constrained by the scarcity of action-labeled robot data, whereas action-free videos provide abundant evidence of how the physical world changes. Latent action models offer a promising way to extract such priors from videos, but reconstruction-trained latent codes are not necessarily suitable for policy generation: they may predict future observations while lacking the structure needed to be reused or generated coherently with robot actions. We introduce ALAM (Algebraic Latent Action Model), an Algebraically Consistent Latent Action Model that turns temporal relations in action-free video into structural supervision. Given frame triplets, ALAM learns latent transitions that are grounded by reconstruction while being regularized by composition and reversal consistency, encouraging a locally additive transition space. For downstream VLA learning, we freeze the pretrained encoder and use its latent transition sequences as auxiliary generative targets, co-generated with robot actions under a joint flow-matching objective. This couples structured latent transitions with flow-based policy generation, allowing the policy to exploit ALAM's locally consistent transition geometry without requiring latent-to-action decoding. Representation probes show that ALAM reduces additivity and reversibility errors by 25-85 times over unstructured latent-action baselines and improves long-horizon cumulative reconstruction. When transferred to VLA policies, ALAM raises the average success rate from 47.9% to 85.0% on MetaWorld MT50 and from 94.1% to 98.1% on LIBERO, with consistent gains on real-world manipulation tasks. Ablations further confirm that the strongest improvements arise from the synergy between algebraically structured latent transitions and joint flow matching.
>
---
#### [new 050] Towards Backdoor-Based Ownership Verification for Vision-Language-Action Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于模型所有权验证任务，旨在解决VLAs模型被非法使用的安全问题。工作是提出GuardVLA框架，通过嵌入后门水印实现可靠验证。**

- **链接: [https://arxiv.org/pdf/2605.09005](https://arxiv.org/pdf/2605.09005)**

> **作者:** Ming Sun; Rui Wang; Xingrui Yu; Lihua Jing; Hangyu Du; Zhenglin Wan; Xu Pan; Ivor Tsang
>
> **摘要:** Vision-Language-Action models (VLAs) support generalist robotic control by enabling end-to-end decision policies directly from multi-modal inputs. As trained VLAs are increasingly shared and adapted, protecting model ownership becomes essential for secure deployment and responsible open-source usage. In this paper, we present GuardVLA, the first backdoor-based ownership verification framework specifically designed for VLAs. GuardVLA embeds a stealthy and harmless backdoor watermark into the protected model during training by injecting secret messages into embodied visual data. For post-release verification, we propose a swap-and-detect mechanism, in which the trigger projector and an external classifier head are used to activate and detect the embedded backdoor based on prediction probabilities. Extensive experiments across multiple datasets, model architectures, and adaptation settings demonstrate that GuardVLA enables reliable ownership verification while preserving benign task performance. Further results show that the embedded watermark remains detectable under post-release model adaptation.
>
---
#### [new 051] LASSA Architecture-Based Autonomous Fault-Tolerant Control of Unmanned Underwater Vehicles
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主控制任务，旨在解决UUV在故障下的容错控制问题。提出LASSA架构，结合LLM与约束验证，实现自主故障检测与路径重规划，提升控制可靠性与实时性。**

- **链接: [https://arxiv.org/pdf/2605.09494](https://arxiv.org/pdf/2605.09494)**

> **作者:** Hong Chen; Zixiang Tang; Yuanbao Chen; Yu Liu
>
> **摘要:** Unmanned underwater vehicles (UUVs) operate persistently in communication-constrained environments, thus requiring high-level autonomous fault-tolerant control under faulty operating conditions. Existing approaches rely heavily on predefined hard-coded rules and struggle to achieve effective fault-tolerant control against unforeseen faults. Although large language models (LLMs) possess powerful cognitive and reasoning capabilities, their inherent hallucinations remain a major obstacle to their application in UUV control systems. This paper proposes an intelligent control method based on the LASSA (LLM-based Agent with Solver, Sensor and Actuator) architecture. Within this architecture, an LLM identifies unknown faults and accomplishes task replanning via autonomous reasoning without hard-coded rules; the intelligent agent undertakes perception, scheduling and decision evaluation; the solver verifies physical boundary feasibility constraints prior to command transmission to the actuators. This architecture suppresses physically infeasible LLM hallucinations and ensures interpretable, verifiable decision-making. Moreover, it enables fast-slow dual closed-loop collaborative control, where the slow loop undertakes high-level dynamic decision-making and the fast loop guarantees high-frequency real-time control, simultaneously balancing decision intelligence and control timeliness. Lake experiments under normal and lower-rudder-fault conditions show that the framework detects trajectory tracking abnormalities, replans the route by adjusting the turning radius from 4m to 12m and reducing speed from 2kn to 1kn, passes all three solver constraints on the first invocation, and guides the UUV to complete the full mission; under normal conditions no false fault alarms are raised throughout the run.
>
---
#### [new 052] Network-Efficient World Model Token Streaming
- **分类: cs.RO**

- **简介: 该论文研究车联网中高效传输离散世界模型状态的问题，提出一种自适应的键帧-差分协议，提升压缩传输效果和下游任务性能。**

- **链接: [https://arxiv.org/pdf/2605.09886](https://arxiv.org/pdf/2605.09886)**

> **作者:** Shatadal Mishra; Ahmadreza Moradipari; Nejib Ammar
>
> **备注:** Accepted at IEEE VNC 2026
>
> **摘要:** Generative driving world models rely on compact latent state representations that must be efficiently transmitted and synchronized across distributed compute and connected vehicles. We study network-efficient streaming of a discrete world model state, where a stride-16 VQ-U-Net tokenizer (codebook size 8,192) maps each 288x512 frame to an 18x32 grid of token IDs (576 tokens/frame), equivalent to 936 bytes/frame under fixed-length coding. We consider a keyframe--delta protocol under strict per-message payload budgets and packet loss, and propose a fully online, label-free algorithm that prioritizes delta updates via cosine distance in codebook embedding space and triggers keyframes adaptively using a Hamming-drift threshold. The adaptive algorithm consistently improves the rate distortion frontier over periodic keyframes at matched bitrates: at 0.024 Mb/s (200-byte budget) dynamic-only embedding distortion drops from 0.0712 to 0.0661 (7.2\%), and at 0.036 Mb/s (400-byte budget) from 0.0427 to 0.0407 (4.8\%). Under 10\% delta packet loss at 200 bytes, dynamic-only distortion is 0.0757 versus 0.0789 for a matched periodic baseline. To connect state fidelity to world model usefulness, we train a lightweight next-token predictor and evaluate perplexity conditioned on streamed receiver states: at 0.024 Mb/s, dynamic-position perplexity improves from 206.0 to 193.1 (6.3\%), and at 0.036 Mb/s from 158.9 to 155.6 (2.1\%). These results support discrete token-state streaming as a practical systems layer for bandwidth-aware synchronization and improved downstream token-dynamics utility under vehicular networking constraints.
>
---
#### [new 053] Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation
- **分类: cs.RO**

- **简介: 该论文属于具身导航任务，旨在解决视觉语言模型在开放世界中导航效果差的问题。通过构建物理基础的抽象环境，提升导航策略的泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.10118](https://arxiv.org/pdf/2605.10118)**

> **作者:** Zhixuan Shen; Jiawei Du; Ziyu Guo; Han Luo; Lilan Peng; Joey Tianyi Zhou; Haonan Luo; Tianrui Li
>
> **备注:** 28 pages, 15 figures, Extended Version of accepted ICML 2026 Paper
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated exceptional general reasoning capabilities. However, their performance in embodied navigation remains hindered by a scarcity of aligned open-world vision and robot control data. Despite simulators providing a cost-effective alternative for data collection, the inherent reliance on photorealistic simulations often limits the transferability of learned policies. To this end, we propose \textit{\textbf{S}andbox-\textbf{A}bstracted \textbf{G}rounded \textbf{E}xperience} (\textbf{\textit{SAGE}}), a framework that enables agents to learn within a physics-grounded semantic abstraction rather than a photorealistic simulation, mimicking the human capacity for mental simulation where plans are rehearsed in simplified physics abstractions before execution. \textit{SAGE} system operates via three synergistic phases: (1) \textit{Genesis}: constructing diverse, physics-constrained semantic environments to bootstrap experience; (2) \textit{Evolution}: distilling experiences through Reinforcement Learning (RL), utilizing a novel asymmetric adaptive clipping mechanism to stabilize updates; (3) \textit{Navigation}: bridging the abstract policy to open-world control. We demonstrate that \textit{SAGE} significantly improves planner-assisted embodied navigation, achieving a 53.21\% LLM-Match Success Rate on A-EQA (+9.7\% over baseline), while showing encouraging transfer to physical indoor robot deployment.
>
---
#### [new 054] Smoothing Out the Edges: Continuous-Time Estimation with Gaussian Process Motion Priors on Factor Graphs
- **分类: cs.RO**

- **简介: 该论文属于连续时间状态估计任务，旨在解决非参数方法应用不足的问题。通过因子图框架简化高斯过程方法，提供三个实现示例以促进应用。**

- **链接: [https://arxiv.org/pdf/2605.09073](https://arxiv.org/pdf/2605.09073)**

> **作者:** Connor Holmes; Sven Lilge; Zi Cong Guo; Frank Dellaert; Timothy D. Barfoot
>
> **摘要:** Continuous-time state estimation is gaining in popularity due to its abilities to provide smooth solutions, handle asynchronous sensors, and interpolate between data points. While there are two main paradigms, parametric (e.g., temporal basis functions, splines) and nonparametric (Gaussian processes), the latter has seen less adoption despite its technical advantages and relative ease of implementation. In this article, we seek to rectify this situation by providing a new simplified explanation of GP continuous-time estimation rooted in the language of factor graphs, which have become the de facto estimation paradigm in much of robotics. To simplify onboarding, we also provide three working examples implemented in the popular GTSAM estimation framework.
>
---
#### [new 055] Understanding Asynchronous Inference Methods for Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决视觉-语言-动作模型异步推理中的延迟问题。通过系统比较四种方法，评估其在不同延迟下的性能。**

- **链接: [https://arxiv.org/pdf/2605.08168](https://arxiv.org/pdf/2605.08168)**

> **作者:** Ayoub Agouzoul
>
> **摘要:** Vision-Language-Action (VLA) models offer a promising path to generalist robot control, but their inference latency causes observation staleness when generated actions are executed asynchronously. Several methods have been proposed concurrently to mitigate this problem: inference-time inpainting (IT-RTC), training-time delay simulation (TT-RTC), future-state-aware conditioning (VLASH), and lightweight residual correction (A2C2). Each takes a fundamentally different approach, but they have so far been evaluated independently with different codebases, base policies, and protocols. We present a systematic comparison of these four methods under controlled conditions. We develop two unified codebases that integrate all methods with harmonized library and dataset versions, and we benchmark them on the Kinetix suite with MLPMixer policies and on the LIBERO manipulation benchmark with SmolVLA, sweeping inference delays up to $d=20$ control steps. A2C2's per-step residual correction is the most effective method on Kinetix, holding above 90% solve rate up to $d=8$, and also leads on LIBERO from $d=4$ onwards. IT-RTC is competitive at low delays but degrades sharply under long chunks ($H=30$) and high delays. TT-RTC is the most robust training-based method: stable across $d_\max$ choices, generalizes beyond its training delay distribution, and adds zero inference overhead. VLASH exhibits a clear low-delay vs. high-delay trade-off governed by the fine-tuning delay range $[0,d_\max]$. Code is available at this https URL
>
---
#### [new 056] StereoPolicy: Improving Robotic Manipulation Policies via Stereo Perception
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操控任务，旨在解决单目视觉缺乏深度信息的问题。通过引入立体视觉的StereoPolicy框架，提升机器人在复杂场景中的操作精度。**

- **链接: [https://arxiv.org/pdf/2605.09989](https://arxiv.org/pdf/2605.09989)**

> **作者:** Evans Han; Yunfan Jiang; Yingke Wang; Haoyue Xiao; Huang Huang; Jianwen Xie; Jiajun Wu; Li Fei-Fei; Ruohan Zhang
>
> **摘要:** Recent advances in robot imitation learning have yielded powerful visuomotor policies capable of manipulating a wide variety of objects directly from monocular visual inputs. However, monocular observations inherently lack reliable depth cues and spatial awareness, which are critical for precise manipulation in cluttered or geometrically complex scenes. To address this limitation, we introduce StereoPolicy, a new visuomotor policy learning framework that directly leverages synchronized stereo image pairs to strengthen geometric reasoning, without requiring explicit 3D reconstruction or camera calibration. StereoPolicy employs pretrained 2D vision encoders to process each image independently and fuses the resulting representations through a Stereo Transformer. This design implicitly captures spatial correspondence and disparity cues. The framework integrates seamlessly with diffusion-based and pretrained vision-language-action (VLA) policies, delivering consistent improvements over RGB, RGB-D, point cloud, and multi-view baselines across three simulation benchmarks: RoboMimic, RoboCasa, and OmniGibson. We further validate StereoPolicy on real-robot experiments spanning both tabletop and bimanual mobile manipulation settings. Our results underscore stereo vision as a scalable and robust modality that bridges 2D pretrained representations with 3D geometric understanding for robotic manipulation.
>
---
#### [new 057] Explicit Stair Geometry Conditioning for Robust Humanoid Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在解决人形机器人爬楼梯时的鲁棒性问题。通过显式建模楼梯几何参数，提升对不同楼梯结构的适应能力。**

- **链接: [https://arxiv.org/pdf/2605.09944](https://arxiv.org/pdf/2605.09944)**

> **作者:** Jianguo Zhang; Wentai Xu; Shusheng Ye; Yuxiang He; Weimin Qi; Qinbo Sun; Ning Ding; Liguang Zhou
>
> **备注:** 8 pages, 7 figures, 4 tables
>
> **摘要:** Robust humanoid stair climbing remains challenging due to geometric discontinuities, sensitivity to step height variations, and perception uncertainty in real-world environments. Existing learning-based locomotion policies often rely on implicit terrain representations or blind proprioceptive feedback, limiting their ability to generalize across varying stair geometries and to anticipate required gait adjustments. This paper proposes an explicit stair geometry conditioning framework for robust humanoid stair climbing. Instead of encoding terrain as high-dimensional latent features, we extract a compact set of interpretable geometric parameters, including step height, step depth, and current yaw angle relative to the robot heading. These explicit stair parameters directly condition a Proximal Policy Optimization (PPO)-based locomotion policy, enabling proactive modulation of swing-foot clearance and stride characteristics according to stair structure. Simulation experiments demonstrate improved generalization across unseen stair heights beyond the training distribution. Real-world experiments on the Unitree G1 humanoid validate reliable indoor and outdoor stair traversal. In challenging outdoor scenarios, the robot successfully ascends 33 consecutive steps without failure, demonstrating robustness and practical deployability.
>
---
#### [new 058] MDrive: Benchmarking Closed-Loop Cooperative Driving for End-to-End Multi-agent Systems
- **分类: cs.RO**

- **简介: 该论文属于多智能体协同驾驶任务，旨在解决现有基准评估不足的问题。提出MDrive基准，包含225个场景，验证多智能体系统优势及挑战。**

- **链接: [https://arxiv.org/pdf/2605.10904](https://arxiv.org/pdf/2605.10904)**

> **作者:** Marco Coscoy; Zewei Zhou; Seth Z. Zhao; Henry Wei; Angela Magtoto; Johnson Liu; Rui Song; Walter Zimmer; Zhiyu Huang; Chen Tang; Bolei Zhou; Jiaqi Ma
>
> **备注:** website:this https URL
>
> **摘要:** Vehicle-to-Everything (V2X) communication has emerged as a promising paradigm for autonomous driving, enabling connected agents to share complementary perception information and negotiate with each other to benefit the final planning. Existing V2X benchmarks, however, fall short in two ways: (i) open-loop evaluations fail to capture the inherently closed-loop nature of driving, leading to evaluation gaps, and (ii) current closed-loop evaluations lack behavioral and interactive diversity to reflect real-world driving. Thus, it is still unclear the extent of benefits of multi-agent systems for closed-loop driving. In this paper, we introduce MDrive, a closed-loop cooperative driving benchmark comprising 225 scenarios grounded in both NHTSA pre-crash typologies and real-world V2X datasets. Our benchmark results demonstrate that multi-agent systems are generally better than single-agent counterparts. However, current multi-agent systems still face two important challenges: (i) perception sharing enhances perceptions, but doesn't always translate to better planning; (ii) negotiation improves planning performance but harms it in complex and dense traffic scenarios. MDrive further provides an open-source toolbox for scenario generation, Real2Sim conversion, and human-in-the-loop simulation. Together, MDrive establishes a reproducible foundation for evaluating and improving the generalization and robustness of cooperative driving systems.
>
---
#### [new 059] AssemPlanner: A Multi-Agent Based Task Planning Framework for Flexible Assembly System
- **分类: cs.RO**

- **简介: 该论文提出AssemPlanner，解决柔性装配系统中任务规划效率低的问题。通过多智能体框架，实现自然语言到生产操作的转换与自适应调整。**

- **链接: [https://arxiv.org/pdf/2605.08831](https://arxiv.org/pdf/2605.08831)**

> **作者:** Chenhao Zhang; Chaoran Zhang; Zhaobo Xu; Yongbo Yang; Pingfa Feng; Long Zeng
>
> **摘要:** In flexible assembly systems, existing task planning methods require a time-consuming configuration process by multiple experts to establish a production line for a new product. To address this challenge, we propose a multi-agent based task planning framework for flexible assembly systems, denoted as AssemPlanner. It takes tasks described in natural language as input, which are then converted into actionable sequential production operations. It comprises several specialized agents, including SchedAgent , KnowledgeAgent, LineBalanceAgent, and a scene graph. Within the proposed framework, SchedAgent serves as the central reasoning engine. Departing from traditional static pipelines, AssemPlanner utilizes a ReAct-based SchedAgent to adaptively adjust actions via multi-agent feedback. By observing the feedback from KnowledgeAgent, LineBalanceAgent, and the scene graph, it autonomously resolves complex industrial process constraints. To facilitate reproducibility, all code and datasets are released at this https URL.
>
---
#### [new 060] High Precision Hydraulic Excavator Control for Heavy-Duty Grading
- **分类: cs.RO**

- **简介: 该论文属于自主控制任务，旨在解决重型挖掘机高精度整平问题。通过分层控制器提升精度与效率，优于现有商业方案。**

- **链接: [https://arxiv.org/pdf/2605.09465](https://arxiv.org/pdf/2605.09465)**

> **作者:** Lennart Werner; Pol Eyschen; Sean Costello; Andrei Cramariuc; Marco Hutter
>
> **备注:** 12 pages 19 figures, RSS 2026
>
> **摘要:** High-precision heavy-duty grading is a common step in earthworks, traditionally carried out manually by skilled operators. Removing a significant amount of material while achieving a high-precision surface requires substantial machine-specific experience. Different hydraulic architectures react differently to operator inputs and soil interaction forces, which makes generalizable controllers challenging. In this paper, we present an autonomous controller that achieves high-precision grading at expert-operator speed on Load Sensing and Negative Flow Control machines alike. We split our controller into two parts: (1) a hydraulic-aware low-level loop that is hydraulic architecture-specific and (2) a path-tracking layer that coordinates joint motions and responses. Through a calibration process, our technique is applicable to load-sensing and negative-flow-control machinery. To showcase its versatility, we benchmark our approach on two excavators with different hydraulics and compare it against a commercial state-of-the-art solution. Our technique (RMSE 1.8~cm) outperforms the commercial solution (RMSE 4.7~cm) in precision by a factor of 2.6 and improves machine usage by leveraging the maximum function pressure, as opposed to commercial solutions that stall prematurely.
>
---
#### [new 061] RoboMemArena: A Comprehensive and Challenging Robotic Memory Benchmark
- **分类: cs.RO**

- **简介: 该论文提出RoboMemArena基准，解决机器人记忆评估不足的问题，通过多模态任务和真实环境测试，提升记忆系统性能。**

- **链接: [https://arxiv.org/pdf/2605.10921](https://arxiv.org/pdf/2605.10921)**

> **作者:** Huashuo Lei; Wenxuan Song; Huarui Zhang; Jieyuan Pei; Jiayi Chen; Haodong Yan; Han Zhao; Pengxiang Ding; Zhipeng Zhang; Lida Huang; Donglin Wang; Yan Wang; Haoang Li
>
> **备注:** Project website: this https URL
>
> **摘要:** Memory is a critical component of robotic intelligence, as robots must rely on past observations and actions to accomplish long-horizon tasks in partially observable environments. However, existing robotic memory benchmarks still lack multimodal annotations for memory formation, provide limited task coverage and structural complexity, and remain restricted to simulation without real-world evaluation. We address this gap with RoboMemArena, a large-scale benchmark of 26 tasks, with average trajectory lengths exceeding 1,000 steps per task and 68.9% of subtasks being memory-dependent. The generation pipeline leverages a vision-language model (VLM) to design and compose subtasks, generates full trajectories through atomic functions, and provides memory-related annotations, including subtask instructions and native keyframe annotations, while paired real-world memory tasks support physical evaluation. We further design PrediMem, a dual-system VLA in which a high-level VLM planner manages a memory bank with recent and keyframe buffers and uses a predictive coding head to improve sensitivity to task dynamics. Extensive experiments on RoboMemArena show that PrediMem outperforms all baselines and provides insights into memory management, model architecture, and scaling laws for complex memory systems.
>
---
#### [new 062] Embodied AI in Action: Insights from SAE World Congress 2026 on Safety, Trust, Robotics, and Real-World Deployment
- **分类: cs.RO**

- **简介: 论文探讨了具身人工智能在现实系统中的应用与挑战，属于技术评估任务，旨在解决安全、信任及部署问题，提出系统工程与标准建设的实践建议。**

- **链接: [https://arxiv.org/pdf/2605.10653](https://arxiv.org/pdf/2605.10653)**

> **作者:** Jan-Mou Li; Paul Schmitt; Wei Tong; Majed Mohammed; Akshay Chalana; Arpan Kusari; Edward Griffor
>
> **摘要:** Embodied artificial intelligence is rapidly moving from research into real-world systems such as autonomous vehicles, mobile robots, and industrial machines. As these systems become more capable of perceiving, deciding, and acting in dynamic environments, they also introduce new challenges in safety, trust, governance, and operational reliability. This white paper summarizes key insights from the SAE World Congress 2026 panel session \textit{Embodied AI in Action}, which brought together experts from automotive, robotics, artificial intelligence, and safety engineering. The discussion highlighted the need to treat embodied AI as a systems challenge requiring engineering rigor, lifecycle governance, human-centered design, and evolving standards. The paper provides practical perspectives for executives, policymakers, and technical leaders seeking to adopt embodied AI responsibly. The panel reached broad agreement that long-term success will depend not only on advances in AI capability, but equally on safe and trustworthy deployment.
>
---
#### [new 063] Raymoval: Raycasting-based Dynamic Object Removal for Static 3D Mapping
- **分类: cs.RO**

- **简介: 该论文属于静态3D地图构建任务，旨在解决动态物体残留影响地图一致性的问题。通过射线投射方法识别并移除动态点，提升地图质量。**

- **链接: [https://arxiv.org/pdf/2605.08937](https://arxiv.org/pdf/2605.08937)**

> **作者:** Daebeom Kim; Seungjae Lee; Seoyeon Jang; Kevin Christiansen Marsim; Hyun Myung
>
> **备注:** 12 pages, 5 figures, 3 tables, Presented at RiTA 2025
>
> **摘要:** Static mapping is fundamental to robot navigation, providing a persistent geometric prior and a consistent reference for long-term autonomy. However, dynamic objects leave residual traces and cause surface loss, which reduces map consistency. We propose a raycasting-based module for dynamic object removal in static 3D mapping. Each scan is projected onto an azimuth-elevation grid, and for every viewing direction we compare the bin-wise minimum range with the map's first-hit distance computed by raycasting. Furthermore, we apply a raycast consistency test that separates dynamic from static points. Finally, a spatial consistency validation step refines labels, producing static maps with lower residual dynamics and reduced over-removal. We evaluate our approach quantitatively and qualitatively on SemanticKITTI and a challenging custom dataset, and show consistent static mapping results.
>
---
#### [new 064] Continuum Robot Modeling with Action Conditioned Flow Matching
- **分类: cs.RO**

- **简介: 该论文属于机器人建模任务，解决TDCR稳态形状预测问题。通过数据驱动方法，建立动作条件的点云流匹配模型，提升形状预测精度。**

- **链接: [https://arxiv.org/pdf/2605.09216](https://arxiv.org/pdf/2605.09216)**

> **作者:** Jiong Lin; Jinchen Ruan; Hod Lipson
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** Predicting the shape of tendon driven continuum robots (TDCRs) at steady state from actuation remains challenging due to continuous deformation, complex tendon routing, compliance, friction, and fabrication variability. In this paper, we address this problem as kinematic self modeling conditioned on action. We present a lightweight 3D printed TDCR hardware platform and an RGB-D data collection pipeline with multiple cameras, and we learn a point cloud flow matching model that maps motor actuation states to the robot's settled 3D geometry. The model is trained from randomly sampled quasi static configurations and evaluated on test motor commands within the same TDCR design family and actuation range. We compare against prior 3D deformable object and robot self modeling approaches in both MuJoCo simulation and real hardware experiments. Experiments on simulated 2-, 3-, and 5-module TDCRs and real 2- and 3-module robots show improved shape prediction accuracy under CD and EMD metrics. We further show in simulation that the same conditional formulation generalizes to tip payload as a conditioning input, enabling payload conditioned steady-state shape prediction. These results demonstrate a data driven self modeling framework for quasi static TDCR geometry prediction.
>
---
#### [new 065] Octopus Protocol: One-Shot Hardware Discovery and Control for AI Agents via Infrastructure-as-Prompts
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文提出Octopus Protocol，解决AI代理控制硬件的工程成本问题。通过单命令实现硬件发现与控制，降低开发难度。属于智能系统与自动化任务。**

- **链接: [https://arxiv.org/pdf/2605.09055](https://arxiv.org/pdf/2605.09055)**

> **作者:** Quilee Simeon; Justin M. Wei; Yile Fan
>
> **摘要:** Recent agentic-robotics systems, from Code-asPolicies to modern vision-language-action (VLA) foundation models, presuppose that drivers, SDKs, or ROS-style primitives for the target hardware already exist. Writing those primitives is the dominant engineering cost of bringing up new hardware for agent control. We present Octopus Protocol, a system that collapses that cost to a single shell command. Given only raw OS access and a language-model API key, a coding agent executes a five-stage pipeline--PROBE, IDENTIFY, INTERFACE, SERVE, DEPLOY--to discover connected devices, infer their capabilities, generate a Model Context Protocol (MCP) server with typed tools, and deploy it as a live HTTP endpoint. A persistent daemon then monitors the system, heals broken code, and perceives physical state through the camera tools it generated for itself. Two architectural principles make this work: protocols are prompts, not code, and the coding agent is the runtime. We validate the system on three heterogeneous platforms (PC/WSL, Apple Silicon macOS, Raspberry Pi 4) and on a commercial 6-DOF robotic arm with USB camera feedback. One command onboards the hardware in ~10-15 minutes and exposes up to 30 MCP tools; an MCP-compliant client then performs closed-loop visual-motor control through tools no human wrote.
>
---
#### [new 066] ProcVLM: Learning Procedure-Grounded Progress Rewards for Robotic Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出ProcVLM，用于机器人操作中的密集进度奖励学习。解决长期任务中进度估计不准确的问题，通过程序结构和视觉变化进行监督，提升任务推理与轨迹评估效果。**

- **链接: [https://arxiv.org/pdf/2605.08774](https://arxiv.org/pdf/2605.08774)**

> **作者:** Youhe Feng; Hansen Shi; Haoyang Li; Xinlei Guo; Yang Wang; Chengyang Zhang; Jinkai Zhang; Xiaohan Zhang; Jie Tang; Jing Zhang
>
> **摘要:** Long-horizon robotic manipulation requires dense feedback that reflects how a task advances through its procedural stages, not merely whether the final outcome is successful. Existing reward models often rely on trajectory-level success labels or time-based interpolation, which can conflate elapsed time with true task progress and therefore fail to capture unfinished steps, stagnation, and failure states. We present ProcVLM, a progress-aware vision-language model that learns procedure-grounded progress as a dense reward signal for manipulation. Rather than deriving progress from terminal outcomes or temporal proxies, ProcVLM grounds progress estimation in procedural structure and intra-stage visual change, and further adopts a reasoning-before-estimation paradigm that infers the remaining atomic actions before estimating task progress. Specifically, we construct this supervision by synthesizing frame-level subtask-semantic annotations, assigning progress budgets according to subtask structure, and distributing each budget based on intra-subtask visual change. To train ProcVLM at scale, we build a standardized procedural supervision synthesis pipeline and construct ProcCorpus-60M from 30 embodied datasets with 60M annotated frames, from which we derive ProcVQA for procedure-aware pretraining, with progress estimation as the central task alongside action segmentation and future planning. Experiments on ProcVQA and reward-model benchmarks show that ProcVLM improves embodied procedural reasoning and yields more discriminative trajectory-internal progress estimates than representative baselines, supporting its use as a dense reward model for downstream reward-guided policy optimization. Project page: this https URL
>
---
#### [new 067] VRA: Grounding Discrete-Time Joint Acceleration in Voltage-Constrained Actuation
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决电压约束下关节加速度不可执行的问题。提出VRA方法，将加速度限制在电压可行范围内，提升执行一致性。**

- **链接: [https://arxiv.org/pdf/2605.10696](https://arxiv.org/pdf/2605.10696)**

> **作者:** Lingwei Zhang; Jiaming Wang; Tianlin Zhang; Zhitao Song; Xuanqi Zeng; Weipeng Xia; Zhongyu Li; Yun-hui Liu
>
> **备注:** 10 pages, Accepted by RSS 2026
>
> **摘要:** Discrete-time joint acceleration constraints are widely used to enforce position and velocity limits. However, under voltage-constrained electric actuators, kinematically admissible accelerations may be physically unrealizable, exposing a missing execution-level abstraction. We propose Voltage-Realizable Acceleration (VRA), a joint-level acceleration interface that grounds kinematic acceleration in voltage-constrained actuator physics by restricting commanded accelerations to voltage-realizable constraints. Hardware experiments on electric actuators and a wheel-legged quadruped show that VRA removes unrealizable accelerations, restores consistent near-constraint execution, and reduces constraint-induced oscillations.
>
---
#### [new 068] Retrieve-then-Steer: Online Success Memory for Test-Time Adaptation of Generative VLAs
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在提升生成式VLA模型在长期部署中的可靠性。通过在线成功记忆机制，利用过往成功经验优化动作生成，实现无需参数更新的测试时自适应。**

- **链接: [https://arxiv.org/pdf/2605.10094](https://arxiv.org/pdf/2605.10094)**

> **作者:** Jianchao Zhao; Huoren Yang; Hu Yusong; Yuyang Gao; Qiguan Ou; Cong Wan; SongLin Dong; Zhiheng Ma; Yihong Gong
>
> **摘要:** Vision-Language-Action (VLA) models show strong potential for general-purpose robotic manipulation, yet their closed-loop reliability often degrades under local deployment conditions. Existing evaluations typically treat test episodes as independent zero-shot trials. However, real robots often operate repeatedly in the same or slowly changing environments, where successful executions provide environment-verified evidence of reliable behavior patterns. We study this persistent-deployment setting, asking whether a partially competent frozen VLA can improve its reliability by reusing its successful test-time experience. We propose an online success-memory guided test-time adaptation framework for generative VLAs. During deployment, the robot stores progress-calibrated successful observation-action segments in a long-term memory. At inference, it retrieves state-relevant action chunks, filters inconsistent candidates via trajectory-level consistency, and aggregates them into an elite action prior. To incorporate this prior into action generation, we introduce confidence-adaptive prior guidance, which injects the elite prior into an intermediate state of the flow-matching action sampler and adjusts the guidance strength based on retrieval confidence. This design allows the frozen VLA to exploit environment-specific successful experience while preserving observation-conditioned generative refinement. This retrieve-then-steer mechanism enables lightweight, non-parametric test-time adaptation without requiring parameter updates. Simulation and real-world experiments show improved task success and closed-loop stability, especially in long-horizon and multi-stage tasks.
>
---
#### [new 069] MAGS-SLAM: Monocular Multi-Agent Gaussian Splatting SLAM for Geometrically and Photometrically Consistent Reconstruction
- **分类: cs.RO**

- **简介: 该论文提出MAGS-SLAM，解决多智能体RGB-only 3D重建问题，通过局部高斯子图和紧凑通信实现无深度传感器的协同重建。**

- **链接: [https://arxiv.org/pdf/2605.10760](https://arxiv.org/pdf/2605.10760)**

> **作者:** Zhihao Cao; Qi Shao; Shuhao Zhai; Jing Zhang; Anh Nguyen; Baoru Huang
>
> **摘要:** Collaborative photorealistic 3D reconstruction from multiple agents enables rapid large-scale scene capture for virtual production and cooperative multi-robot exploration. While recent 3D Gaussian Splatting (3DGS) SLAM algorithms can generate high-fidelity real-time mapping, most of the existing multi-agent Gaussian SLAM methods still rely on RGB-D sensors to obtain metric depth and simplify cross-agent alignment, which limits the deployment on lightweight, low-cost, or power-constrained robotic platforms. To address this challenge, we propose MAGS-SLAM, the first RGB-only multi-agent 3DGS SLAM framework for collaborative scene reconstruction. Each agent independently builds local monocular Gaussian submaps and transmits compact submap summaries rather than raw observations or dense maps. To facilitate robust collaboration in the presence of monocular scale ambiguity, our framework integrates compact submap communication, geometry- and appearance-aware loop verification, and occupancy-aware Gaussian fusion, enabling coherent global reconstruction without active depth sensors. We further introduce ReplicaMultiagent Plus benchmark for evaluating collaborative Gaussian SLAM. Intensive experiments on synthetic and real-world datasets show that MAGS-SLAM achieves competitive tracking accuracy and comparable or superior rendering quality to state-of-the-art RGB-D collaborative Gaussian SLAM methods while relying only RGB images.
>
---
#### [new 070] JODA: Composable Joint Dynamics for Articulated Objects
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出JODA框架，用于生成关节动力学，解决模拟中缺乏精细动态效果的问题。通过结构化三通道场建模，实现可编辑和优化的关节行为。**

- **链接: [https://arxiv.org/pdf/2605.09954](https://arxiv.org/pdf/2605.09954)**

> **作者:** Tianhong Gao; Cheng Yu; Yinghao Xu; Mengyu Chu
>
> **摘要:** Articulated objects used in simulation and embodied AI are typically specified by geometry and kinematic structure, but lack the fine-grained dynamical effects that govern realistic mechanical behavior, such as frictional holding, detents, soft closing, and snap latching. Existing approaches either ignore the detailed structure of dynamics entirely, or use simple models with limited expressiveness. We introduce JODA, a framework for generating joint-level dynamics as a structured three-channel field over the joint degree of freedom, capturing conservative forces, dry friction, and damping. Instantiated using shape-constrained piecewise cubic interpolation (PCHIP), this formulation defines a compact and expressive function space that is both interpretable and compatible with differentiable simulation. Building on this representation, we develop methods for inferring and refining joint dynamics from multimodal inputs. Given visual observations and joint context, a vision-language model proposes structured dynamical primitives, which are composed into a unified dynamics field. The resulting representation supports both direct manipulation and gradient-based refinement. We demonstrate that JODA enables plausible and controllable modeling of diverse joint behaviors, providing a unified interface for inference, editing, and optimization. Code and example assets with their generated profiles will be released upon publication.
>
---
#### [new 071] HULK: Large-scale Hierarchical Coordination under Continual and Uncertain Temporal Tasks
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究多智能体系统的协调问题，针对持续且不确定的时序任务，提出HULK框架，实现分层高效协作。**

- **链接: [https://arxiv.org/pdf/2605.08722](https://arxiv.org/pdf/2605.08722)**

> **作者:** Qingyuan Luo; Jie Li; Meng Guo
>
> **备注:** Accepted to the IEEE International Conference on Robotics and Automation. 7 pages, 4 figures
>
> **摘要:** Multi-agent systems can be extremely efficient when working concurrently and collaboratively, e.g., for delivery, surveillance, search and rescue. Coordination of such teams often involves two aspects: selecting appropriate subteams for different tasks in various areas, and coordinating agents in the subteams to execute the associated subtasks. Existing work often assumes that the tasks are static and known beforehand, where an integer program can be formulated and solved offline. However, in many applications, the team-wise tasks are generated online continually by external requests, and the amount of subtasks within each task is uncertain, e.g., the number of packages to deliver or victims to rescue. The aforementioned offline solution becomes inadequate as it would require constant re-computation for the whole team and global communication to broadcast the results. Thus, this work tackles the large-scale coordination problem under continual and uncertain temporal tasks, specified as temporal logic formulas over collaborative actions. The proposed hierarchical framework, HULK, consists of two interleaved layers: the rolling assignment of currently known tasks to subteams within a certain horizon, and the dynamic coordination within a subteam given the detected subtasks during online execution. Thus, coordination is performed hierarchically at different granularities and triggering conditions, improving computational efficiency and robustness. The method is validated rigorously over large-scale heterogeneous systems under various temporal tasks and environment uncertainties.
>
---
#### [new 072] Unified Noise Steering for Efficient Human-Guided VLA Adaptation
- **分类: cs.RO**

- **简介: 该论文属于机器人操控任务，解决VLA模型在真实环境中的高效适应问题。通过结合人类纠正与噪声空间强化学习，提出UniSteer框架，提升适应效率。**

- **链接: [https://arxiv.org/pdf/2605.10821](https://arxiv.org/pdf/2605.10821)**

> **作者:** Junjie Lu; Xinyao Qin; Yuhua Jiang; Kaixin Wang; Chuheng Zhang; Bin Liang; Jun Yang; Min Xu; Li Zhao
>
> **摘要:** Diffusion-based vision-language-action (VLA) models have emerged as strong priors for robotic manipulation, yet adapting them to real-world distributions remains challenging. In particular, on-robot reinforcement learning (RL) is expensive and time-consuming, so effective adaptation depends on efficient policy improvement within a limited budget of real-world interactions. Noise-space RL lowers the cost by keeping the pretrained VLA fixed as a denoising generator while updating only a lightweight actor that predicts the noise. However, its performance is still limited due to inefficient autonomous exploration. Human corrective interventions can reduce this exploration burden, but they are naturally provided in action space, whereas noise-space finetuning requires supervision over noise variables. To address these challenges, we propose UniSteer, a Unified Noise Steering framework that combines human corrective guidance with noise-space RL through approximate action-to-noise inversion. Given a human corrective action, UniSteer inverts the frozen flow-matching decoder to recover a noise target, which provides supervised guidance for the same noise actor that is simultaneously optimized via reinforcement learning. Real-world experiments on diverse manipulation tasks show that UniSteer adapts more efficiently than strong noise-space RL and action-space human-in-the-loop baselines, improving the success rate from 20% to 90% in 66 minutes on average across four real-world adaptation tasks.
>
---
#### [new 073] Failing Forward: Adaptive Failure-Informed Learning for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型因仅依赖成功数据而易失效的问题。提出AFIL框架，利用失败轨迹提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.08434](https://arxiv.org/pdf/2605.08434)**

> **作者:** Meng Zheng; Samhita Marri; Anwesa Choudhuri; Benjamin Planche; Zhongpai Gao; Van Nguyen Nguyen; Terrence Chen; Girish Chowdhary; Ziyan Wu
>
> **摘要:** Vision-language-action (VLA) models provide a promising paradigm for scalable robotic manipulation, yet their reliance on success-only behavioral cloning leaves them brittle; lacking corrective training signals, minor execution errors rapidly compound into unrecoverable, out-of-distribution failures. To address this limitation, we propose Adaptive Failure-Informed Learning (AFIL), an end-to-end framework that leverages failure trajectories as adaptive negative guidance for diffusion- and flow-based VLA policies. AFIL uses a pretrained VLA to generate failure rollouts online, avoiding the need for handcrafted failure-mode design or human-in-the-loop recovery. It then jointly trains Dual Action Generators (DAGs) for successful and failed behaviors while sharing a common vision-language backbone, enabling efficient failure-aware policy learning with limited parameter overhead. During sampling, the failure generator adaptively steers action generation away from failure-prone regions and toward more reliable success modes, with guidance strength determined by the per-diffusion-step distance between success and failure distributions. Experiments across in-domain and out-of-domain robotic manipulation tasks, covering both short- and long-horizon settings, show that AFIL consistently improves task success rates and robustness over existing VLA baselines, demonstrating its effectiveness, efficiency, and generality.
>
---
#### [new 074] From Ontology Conformance to Admissible Reconfiguration: A RoSO/SMGI Adequacy Argument for Robotic Service Governance
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于服务机器人领域，解决服务配置在修改后如何保持合法的问题。工作是将RoSO嵌入SMGI，建立动态治理机制。**

- **链接: [https://arxiv.org/pdf/2605.08185](https://arxiv.org/pdf/2605.08185)**

> **作者:** Aomar Osmani
>
> **备注:** 26 pages
>
> **摘要:** The Robotic Service Ontology (RoSO) gives service robotics a typed semantic vocabulary for services, functions, interactions, and deployment-sensitive constraints. Its public revision trail makes visible a harder question than ontology conformance alone can settle: once a service is rebound, recomposed, repaired, or redeployed, under what conditions does the resulting configuration remain an admissible realization of the same protected service? This article argues that the Structural Model of General Intelligence (SMGI) is relevant exactly at that level \citep{osmani2026smgi}. SMGI adds not only a structural interface $\theta$, but an induced behavioral semantics $T_\theta$ and a governance discipline for norm-respecting change. We show that RoSO can be embedded into SMGI as a typed semantic layer, so that service descriptions become dynamically governable rather than merely well formed. This yields a RoSO-to-SMGI adequacy theorem, identity-preserving reconfiguration criteria, and compositional conditions under which locally acceptable updates remain globally admissible. The resulting claim is not that SMGI replaces RoSO, but that it provides a formal account of what admissible runtime change requires once service semantics must survive revision.
>
---
#### [new 075] Anatomical Landmark-Guided Deep Reinforcement Learning for Autonomous Gastric Navigation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自主胃部导航任务，旨在解决无线胶囊内镜覆盖不足和导航方法泛化性差的问题。通过结合解剖标志与深度强化学习，提升导航效果与效率。**

- **链接: [https://arxiv.org/pdf/2605.08269](https://arxiv.org/pdf/2605.08269)**

> **作者:** Haoxuan Wu; Sishen Yuan; Haitao Gao; Zhen Li; Xiuli Zuo; Hongliang Ren
>
> **摘要:** Wireless capsule endoscopy (WCE) enables painless visualization of the gastrointestinal tract, but its diagnostic potential is limited by incomplete mucosal coverage and poor transferability of existing navigation methods across patient anatomies. We propose a transferable, anatomical landmarkguided deep reinforcement learning (AL-DRL) framework for autonomous gastric navigation. Leveraging a lightweight edgecontour-depth fusion module, our policy operates on stable, lowdimensional landmark coordinates rather than high-dimensional video streams, effectively bridging the sim-to-real gap. In simulations across eight patient-derived models, the method achieves over 97% coverage within 50 seconds, significantly outperforming vanilla PPO, SAC, and DQN agents. A two-stage sim-to-real pipeline with an adaptive dynamic programming controller actively mitigates physical disturbances. Ex-vivo experiments demonstrate a mean coverage of 87% and a 53% reduction in procedure time compared with expert manual control.
>
---
#### [new 076] Terminal Matters: Kinodynamic Planning with a Terminal Cost and Learned Uncertainty in Belief State-Cost Space
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决在不确定性下可靠到达目标的问题。通过引入终端成本和学习的不确定性模型，提升规划的可靠性与目标达成率。**

- **链接: [https://arxiv.org/pdf/2605.09046](https://arxiv.org/pdf/2605.09046)**

> **作者:** Zhuoyun Zhong; Seyedali Golestaneh; Constantinos Chamzas
>
> **摘要:** In many real-world robotic tasks, robots must generate dynamically feasible motions that reliably reach desired goals even under uncertainty. Yet existing sampling-based kinodynamic planners typically optimize accumulated trajectory costs and treat goal reaching as a feasibility check, rather than explicitly optimizing terminal-state quality, such as goal preference or goal-reaching reliability. In this work, we introduce a terminal-cost formulation for kinodynamic planning that allows terminal-state quality to be optimized alongside accumulated trajectory cost. We prove that AO-RRT, an asymptotically optimal kinodynamic planner, preserves its asymptotic optimality under this augmented objective. We further extend the formulation to belief space and prove that minimizing the Wasserstein distance between the terminal belief and the goal improves a lower bound on the probability of reaching the goal region. The resulting planner, KiTe, uses this terminal-cost objective to encode goal preferences and improve reliability under uncertainty. To support systems without analytical uncertainty models, we learn dynamics and process uncertainty directly from data and integrate the learned belief dynamics into planning. Experiments on Flappy Bird, Car Parking, and Planar Pushing show that KiTe consistently improves goal-reaching success under uncertainty. Real-world Planar Pushing experiments further demonstrate that KiTe can plan effectively with learned dynamics and uncertainty. Source code is available at this https URL.
>
---
#### [new 077] Trajectory-Consistent Flow Matching for Robust Visuomotor Policy Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉-运动策略学习任务，解决轨迹误差累积问题。通过轨迹一致性流匹配、多步监督和RK4积分等方法提升策略鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.08511](https://arxiv.org/pdf/2605.08511)**

> **作者:** Riad Ahmed; Sujosh Nag; Moniruzzaman Akash; Mostafa Hussein; Momotaz Begum
>
> **摘要:** Flow matching policies learn continuous velocity fields that transport noise to actions, enabling fast deterministic inference for robot manipulation. However, standard training optimizes a pointwise velocity objective while inference requires numerical integration of that field -- a mismatch that causes compounding trajectory errors. We propose four complementary remedies: (1) auxiliary rectified flow velocity regression that provides uniform temporal supervision across the full time interval; (2) multi-step trajectory consistency training that supervises the integrated displacement of the velocity field over trajectory segments, directly closing the train-inference gap; (3) velocity field regularization that enforces temporal smoothness, preventing oscillations that destabilize integration; and (4) fourth-order Runge-Kutta (RK4) inference that reduces global discretization error by orders of magnitude over Euler methods. Critically, these components are not independently sufficient -- RK4 without a smooth velocity field fails, and smoothness without trajectory-level supervision still drifts, as our ablation study confirms. We further pair these with a dual-view 3D point cloud encoder using two independent PointNet encoders for complementary spatial perception. On four real-robot tasks across a Franka arm and a Boston Dynamics Spot, our method achieves 70% and 60% overall success on two long-horizon multi-phase tasks where both baselines score 0%, and reaches 100% on precision tool placement. Three MetaWorld simulation tasks confirm consistent improvements, validating that trajectory-level supervision is essential for reliable policy execution.
>
---
#### [new 078] ATAAT: Adaptive Threat-Aware Adversarial Tuning Framework against Backdoor Attacks on Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于安全防护任务，针对视觉-语言-动作模型的后门攻击问题，提出ATAAT框架，解决梯度干扰问题，实现高效隐匿攻击。**

- **链接: [https://arxiv.org/pdf/2605.08612](https://arxiv.org/pdf/2605.08612)**

> **作者:** Kewei Chen; Yayu Long; Shuai Li; Mingsheng Shang
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Addressing the escalating security vulnerabilities in Vision-Language-Action (VLA) models, this study investigates backdoor attacks targeting the visual pathway. We identify a core obstacle causing the failure of traditional attack paradigms: "Gradient Interference." This phenomenon represents an optimization failure triggered by conflicting strategies during end-to-end training. To resolve this, we propose an Adaptive Threat-Aware Adversarial Tuning (ATAAT) framework. Through its core "Threat-Method Adaptive Mapping" mechanism, ATAAT intelligently selects the optimal gradient decoupling strategy based on the adversary's capabilities. Extensive experiments demonstrate that ATAAT exhibits significant advantages, achieving a highly robust Targeted Attack Success Rate (TASR > 80%) while maintaining extreme stealthiness with merely a 5% poisoning rate. It efficiently handles complex semantic-level triggers and achieves implicit decoupled attacks in data poisoning scenarios for the first time. This work reveals a critical security vulnerability in VLAs and provides theoretical and methodological support for future defense architectures.
>
---
#### [new 079] HarmoWAM: Harmonizing Generalizable and Precise Manipulation via Adaptive World Action Models
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决传统WAM模型在泛化性和操作精度间的权衡问题。提出HarmoWAM，结合预测与反应专家，提升任务执行的准确性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.10942](https://arxiv.org/pdf/2605.10942)**

> **作者:** Qiuxuan Feng; Jiale Yu; Jiaming Liu; Yueru Jia; Zhuangzhe Wu; Hao Chen; Zezhong Qian; Shuo Gu; Peng Jia; Siwei Ma; Shanghang Zhang
>
> **摘要:** World Action Models (WAMs) have emerged as a promising paradigm for robot control by modeling physical dynamics. Current WAMs generally follow two paradigms: the "Imagine-then-Execute" approach, which uses video prediction to infer actions via inverse dynamics, and the "Joint Modeling" approach, which jointly models actions and video representations. Based on systematic experiments, we observe a fundamental trade-off between these paradigms: the former explicitly leverages world models for generalizable transit but lacks interaction precision, whereas the latter enables fine-grained, temporally coherent action generation but is constrained by the exploration space of the training distribution. Motivated by these findings, we propose HarmoWAM, an end-to-end WAM that fully leverages a world model to unify predictive and reactive control, enabling both generalizable transit and precise manipulation. Specifically, the world model provides spatio-temporal physical priors that condition two complementary action experts: a predictive expert that leverages latent dynamics for iterative action generation, and a reactive expert that directly infers actions from predicted visual evolution. To enable adaptive coordination, a Process-Adaptive Gating Mechanism is proposed to automatically determine the timing and location of switching between them. This allows the world model to drive the reactive expert to expand the exploration space and the predictive expert to perform precise interactions across different stages of a task. For evaluation, we construct three training-unseen test environments across six real-world robotic tasks, covering variations in background, position, and object semantics. Notably, HarmoWAM achieves strong zero-shot generalization across these scenarios, significantly outperforming prior state-of-the-art VLA models and WAMs by margins of 33% and 29%, respectively.
>
---
#### [new 080] Latent Geometry Beyond Search: Amortizing Planning in World Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决世界模型中快速目标导向规划的问题。通过构建潜在逆动力学模型，将规划过程从在线搜索转移到学习推理，提升效率。**

- **链接: [https://arxiv.org/pdf/2605.08732](https://arxiv.org/pdf/2605.08732)**

> **作者:** Hoang Nguyen; Xiaohao Xu; Xiaonan Huang
>
> **备注:** 31 pages
>
> **摘要:** Modern vision-based world models can represent observations as compact yet expressive latent manifolds, but fast goal-oriented planning in these spaces remains challenging. This raises a central question: when does a learned representation simplify control, rather than merely enabling prediction? We study this question in a pretrained LeWorldModel, whose latent geometry is regularized for smoothness and uniformity. Our key insight is that, under such geometry, planning can be amortized into a latent inverse-dynamics mapping instead of requiring online search. We therefore replace iterative planning with a lightweight Goal-Conditioned Inverse Dynamics Model (GC-IDM) that maps the current latent state, goal latent state, and remaining horizon directly to the next action. Empirically, across four benchmark environments spanning navigation, contact-rich manipulation, and continuous control, our controller matches or exceeds CEM in seven of eight environment-protocol settings while reducing per-decision cost by 100-130x. A broader sweep over test-time planners (CEM, MPPI, iCEM, and gradient-based methods) shows that this result is not specific to a particular optimizer. These findings suggest that much of the structure recovered by test-time planning is already locally encoded in the latent representation. More broadly, our results indicate that sufficiently structured latent spaces can shift part of the planning burden from online optimization to learned inference.
>
---
#### [new 081] ReasonSTL: Bridging Natural Language and Signal Temporal Logic via Tool-Augmented Process-Rewarded Learning
- **分类: cs.AI; cs.RO; eess.SY**

- **简介: 该论文属于自然语言到信号时序逻辑（STL）的翻译任务，旨在解决手动编写STL公式成本高、隐私风险等问题。论文提出ReasonSTL框架，通过分解翻译过程并引入过程奖励训练，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2605.06483](https://arxiv.org/pdf/2605.06483)**

> **作者:** Bowen Ye; Zhijian Li; Junyue Huang; Junkai Ma; Xiang Yin
>
> **摘要:** Signal Temporal Logic (STL) is an expressive formal language for specifying spatio-temporal requirements over real-valued, real-time signals. It has been widely used for the verification and synthesis of autonomous systems and cyber-physical systems. In practice, however, users often express their requirements in natural language rather than in structured STL formulas, making natural-language-to-STL translation a critical yet challenging task. Manual specification requires temporal-logic expertise and cannot scale, while prompting commercial LLM APIs incurs substantial token costs and may expose sensitive system requirements to third-party services, raising privacy concerns for industrial deployment. To address these challenges, we present \textsc{ReasonSTL}, a tool-augmented framework that adapts local open-source language models for natural-language-to-STL generation. \textsc{ReasonSTL} decomposes the translation process into explicit reasoning, deterministic tool calls, and structured formula construction. We further introduce process-rewarded training to supervise both tool-use trajectories and final formulas, together with \textsc{STL-Bench}, a bilingual, computation-aware benchmark grounded in real-world signals. Experiments show that a 4B model trained with \textsc{ReasonSTL} achieves state-of-the-art performance in both automatic metrics and human evaluations, demonstrating that \textsc{ReasonSTL} provides a transparent, low-cost, and privacy-preserving alternative for formal specification drafting.
>
---
#### [new 082] Test-Time Training for Visual Foresight Vision-Language-Action Models
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于视觉语言动作模型任务，旨在解决VF-VLA在分布外数据下的脆弱性问题。通过引入测试时训练和自适应更新过滤机制，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.08215](https://arxiv.org/pdf/2605.08215)**

> **作者:** Sangwu Park; Wonjoong Kim; Yeonjun In; Sein Kim; Hongseok Kang; Chanyoung Park
>
> **备注:** Preprint. Under review
>
> **摘要:** Visual Foresight VLA (VF-VLA) has become a prominent architectural choice in the recent VLA due to its impressive performance. Nevertheless, the inherent design of VF-VLA makes it particularly vulnerable to out-of-distribution (OOD) shifts. Because the quality of action directly depends on the accuracy of the predicted future visual information, OOD conditions affect both stages at once. To address this vulnerability, we propose Test-Time Training Visual Foresight VLA ($T^3$VF), a test-time training approach motivated by the observation that the predicted future image and its subsequent observation form a natural supervision pair. To further address the practical challenges that arise from indiscriminate test-time updates, we introduce an adaptive update filtering mechanism. Empirically, $T^3$VF mitigates the OOD vulnerability of VF-VLA at a modest additional inference cost, without requiring any architectural modification or auxiliary modules.
>
---
#### [new 083] Variational Inference for Lévy Process-Driven SDEs via Neural Tilting
- **分类: cs.LG; cs.AI; cs.CV; cs.RO; stat.ML**

- **简介: 该论文属于贝叶斯推断任务，解决Lévy过程驱动SDE的后验估计问题。提出神经指数倾斜方法，提升推断效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.10934](https://arxiv.org/pdf/2605.10934)**

> **作者:** Yaman Kindap; Manfred Opper; Benjamin Dupuis; Umut Simsekli; Tolga Birdal
>
> **备注:** The associated project page which contains the official implementation can be found in this https URL
>
> **摘要:** Modelling extreme events and heavy-tailed phenomena is central to building reliable predictive systems in domains such as finance, climate science, and safety-critical AI. While Lévy processes provide a natural mathematical framework for capturing jumps and heavy tails, Bayesian inference for Lévy-driven stochastic differential equations (SDEs) remains intractable with existing methods: Monte Carlo approaches are rigorous but lack scalability, whereas neural variational inference methods are efficient but rely on Gaussian assumptions that fail to capture discontinuities. We address this tension by introducing a neural exponential tilting framework for variational inference in Lévy-driven SDEs. Our approach constructs a flexible variational family by exponentially reweighting the Lévy measure using neural networks. This parametrization preserves the jump structure of the underlying process while remaining computationally tractable. To enable efficient inference, we develop a quadratic neural parametrization that yields closed-form normalization of the tilted measure, a conditional Gaussian representation for stable processes that facilitates simulation, and symmetry-aware Monte Carlo estimators for scalable optimization. Empirically, we demonstrate that the method accurately captures jump dynamics and yields reliable posterior inference in regimes where Gaussian-based variational approaches fail, on both synthetic and real-world datasets.
>
---
#### [new 084] VISTA: A Benchmark for Real-Time Video Streaming under Network Impairments in Surgical Teleoperation
- **分类: eess.IV; cs.RO**

- **简介: 该论文属于网络性能评估任务，旨在研究网络干扰对远程手术视频流的影响。通过构建VISTA基准，模拟多种网络环境，分析其对视频质量、时间连续性及操作成功率的影响。**

- **链接: [https://arxiv.org/pdf/2605.08886](https://arxiv.org/pdf/2605.08886)**

> **作者:** Zexin Deng; Zhenhui Yuan; Tian Lu; Gaofeng Li; Meipeng Huang; Longhao Zou
>
> **备注:** Oral presentation at the Connected Autonomous Robotic Systems Workshop, ICRA 2026
>
> **摘要:** Real-time video streaming is crucial in surgical teleoperation, yet reproducible evaluation under realistic network impairments remains limited. This paper presents VISTA, a benchmark designed to study how impairments along the forward video path affect received video quality, temporal continuity, and human task performance. VISTA employs Linux Traffic Control with NetEm and a Gilbert-Elliott loss model to emulate five network conditions: Hospital LAN, 5G Urban, 4G Rural, LEO Satellite, and GEO Satellite. The benchmark integrates a standardised peg transfer task with synchronized measurements of network quality of service (QoS), objective video quality (PSNR, SSIM, and VMAF), and temporal continuity through freeze rate, while maintaining a stable reverse control channel. Across 375 experimental trials, network degradation substantially reduced teleoperation performance: success rate decreased from 97% in Hospital LAN to 79% in 5G Urban, 35% in 4G Rural, 71% in LEO Satellite, and 12% in GEO Satellite, while mean task completion time for successful trials increased from 80 s in Hospital LAN to 117 s in 5G Urban, 211 s in 4G Rural, 152 s in LEO Satellite, and 255 s in GEO Satellite. These findings show that network impairments have a direct impact on task completion and success in surgical teleoperation, and provide a reproducible basis for evaluating teleoperation video under realistic network constraints. Source code available at this https URL.
>
---
#### [new 085] Increasing the Efficiency of DETR for Maritime High-Resolution Images
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于目标检测任务，旨在提升海面高分辨率图像中物体检测的效率与准确性。针对实时检测难题，提出基于ViM的改进方法，优化计算效率与内存占用。**

- **链接: [https://arxiv.org/pdf/2605.10269](https://arxiv.org/pdf/2605.10269)**

> **作者:** Tinsae Yehuala; Hao Cheng; Ville Lehtola
>
> **备注:** Accepted to IEEE ITSC 2026. Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses. DOI to be added upon publication
>
> **摘要:** Maritime object detection is critical for the safe navigation of unmanned surface vessels (USVs), requiring accurate recognition of obstacles from small buoys to large vessels. Real-time detection is challenging due to long distances, small object sizes, large-scale variations, edge computing limitations, and the high memory demands of high-resolution imagery. Existing solutions, such as downsampling or image splitting, often reduce accuracy or require additional processing, while memory-efficient models typically handle only limited resolutions. To overcome these limitations, we leverage Vision Mamba (ViM) backbones, which build on State Space Models (SSMs) to capture long-range dependencies while scaling linearly with sequence length. Images are tokenized into sequences for efficient high-resolution processing. For further computational efficiency, we design a tailored Feature Pyramid Network with successive downsampling and SSM layers, as well as token pruning to reduce unnecessary computation on background regions. Compared to state-of-the-art methods like RT-DETR with ResNet50 backbone, our approach achieves a better balance between performance and computational efficiency in maritime object detection.
>
---
#### [new 086] QueST: Persistent Queries as Semantic Monitors for Drift Suppression in Long-Horizon Tracking
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于长时序跟踪任务，解决视频中目标点因误差累积导致的语义漂移问题。提出QueST框架，通过语义查询和物理约束实现稳定跟踪。**

- **链接: [https://arxiv.org/pdf/2605.09513](https://arxiv.org/pdf/2605.09513)**

> **作者:** Mayank Anand; Mohammad Saqlain; Kyan Mahajan; Priya Shukla; Gora Chand Nandi; Andrew Melnik
>
> **摘要:** Tracking points in videos is typically formulated as frame-to-frame correspondence, where each point is matched locally to the next frame. While this works over short horizons, errors accumulate under articulation, occlusion, and viewpoint change, leading to silent semantic drift that existing trackers cannot detect or correct. In this work, we revisit long-horizon tracking from a monitoring perspective and introduce QueST, a monitoring-by-design framework that treats interaction-relevant entities as persistent semantic queries rather than transient point tracks. Instead of local propagation, each query attends globally over spatio-temporal video features at every time-step, providing a stable semantic anchor across time. We further constrain query trajectories with lightweight 3D physical grounding, using geometric plausibility to suppress unbounded drift under occlusion. We evaluate QueST on long-horizon articulated sequences from PartNet-Mobility in SAPIEN and compare against RAFT-3D, CoTracker, and TAP-Net. QueST substantially reduces terminal drift achieving a 67.7% Absolute Point Error (APE) improvement over TAP-Net while better preserving identity over extended horizons. Our results show that embedding semantic monitoring directly into perception enables more reliable long-horizon tracking under distribution shift.
>
---
#### [new 087] Neuromorphic Reinforcement Learning for Quadruped Locomotion Control on Uneven Terrain
- **分类: cs.NE; cs.RO**

- **简介: 该论文属于四足机器人运动控制任务，旨在解决复杂地形下控制器适应性差的问题。通过结合EP-PPO框架与生物启发策略，实现低功耗在线学习与优化。**

- **链接: [https://arxiv.org/pdf/2605.09595](https://arxiv.org/pdf/2605.09595)**

> **作者:** Zhuangyu Han; Abhronil Sengupta
>
> **摘要:** Reinforcement learning (RL) has enabled robust quadruped locomotion over complex terrain, but most learned controllers are trained offline with backpropagation in massively parallel simulation and deployed as fixed policies, limiting adaptation to terrain variation, payload changes, actuator wear, and other real-world conditions under onboard power constraints. Local learning provides a potential path toward energy-aware on-robot adaptation by replacing global backpropagation graphs with updates driven by local neural states, making the learning rule more compatible with neuromorphic and in-memory computing substrates. This work proposes an equilibrium-propagation (EP)-based proximal policy optimization (PPO) framework for uneven-terrain quadruped locomotion. The controller combines a bio-inspired central pattern generator (CPG) policy with a residual postural adjustment policy, while replacing conventional backpropagation-trained policy and value networks with EP-enabled local learning. To train stochastic continuous-control policies with EP, we derive an EP-compatible PPO output-nudging signal and introduce a two-sided ratio clipping mechanism that stabilizes policy updates during relaxation. Experiments on a 12-DoF A1 quadruped show that the proposed controller achieves stable policy convergence in a two-stage uneven terrain locomotion task. Its locomotion performance is comparable to a backpropagation-trained PPO baseline in success rate, velocity tracking, actuator power, and body stability, while improving GPU memory efficiency by 4.3\(\times\) compared with backpropagation through time (BPTT). These results suggest that local equilibrium-based learning can support high-dimensional embodied locomotion and provide an algorithmic foundation for low-power on-robot adaptation and fine-tuning.
>
---
#### [new 088] DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决长时序世界建模问题。通过预测未来帧的潜在语义特征和引入文本推理机制，提升驾驶决策的准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.10564](https://arxiv.org/pdf/2605.10564)**

> **作者:** Lingjun Zhang; Changjie Wu; Linzhe Shi; Jiangyang Li; Jiaxin Liu; Lei Yang; Hang Zhang; Mu Xu; Hong Wang
>
> **备注:** ICML 2026
>
> **摘要:** End-to-end autonomous driving systems are increasingly integrating Vision-Language Model (VLM) architectures, incorporating text reasoning or visual reasoning to enhance the robustness and accuracy of driving decisions. However, the reasoning mechanisms employed in most methods are direct adaptations from general domains, lacking in-depth exploration tailored to autonomous driving scenarios, particularly within visual reasoning modules. In this paper, we propose a driving world model that performs parallel prediction of latent semantic features for consecutive future frames in the bird's-eye-view (BEV) space, thereby enabling long-horizon modeling of future world states. We also introduce an efficient and adaptive text reasoning mechanism that utilizes additional social knowledge and reasoning capabilities to further improve driving performance in challenging long-tail scenarios. We present a novel, efficient, and effective approach that achieves state-of-the-art (SOTA) results on the closed-loop Bench2drive benchmark. Codes are available at: this https URL.
>
---
#### [new 089] Safe Exploration for Nonlinear Processes Using Online Gaussian Process Learning
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文属于安全控制任务，解决非线性系统在模型不确定下的安全探索问题。通过在线高斯过程学习，确保系统稳定与约束满足。**

- **链接: [https://arxiv.org/pdf/2605.09772](https://arxiv.org/pdf/2605.09772)**

> **作者:** Stefano Tonini; Soroush Rastegarpour; Hamid Reza Feyzmahdavian; Nicola Bastianello; Karl Henrik Johansson
>
> **备注:** Accepted in 23rd IFAC World Congress
>
> **摘要:** This paper proposes a safe data-driven control framework for nonlinear systems with partially known dynamics. The method ensures stability and constraint satisfaction during online learning, assuming only a stabilizable linear approximation of the process is available. Unmodeled nonlinear dynamics are captured by a Gaussian process residual learned in real time. Safety is enforced through a probabilistic control-invariant set derived from Lyapunov theory, guaranteeing high-probability stability. A convex quadratic program computes control inputs that maximize information gain while respecting probabilistic safety constraints. The framework provides finite-sample safety guarantees and allows adaptive expansion of the invariant set as uncertainty decreases. Numerical results validate the approach, demonstrating safe and informative exploration under model uncertainty: the safe set expands by about 30% while the Gaussian process root-mean-square error drops from 1.11 to 0.03.
>
---
#### [new 090] MAG-VLAQ: Multi-modal Aerial-Ground Query Aggregation for Cross-View Place Recognition
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于跨视角场景识别任务，旨在解决地面与航拍图像间的视角、模态和结构差异问题。通过融合多模态数据，提出MAG-VLAQ框架提升匹配性能。**

- **链接: [https://arxiv.org/pdf/2605.09418](https://arxiv.org/pdf/2605.09418)**

> **作者:** Zhengyi Xu; Yuhang Ming; Zhihao Zhan; Hanyu Zhu; Javier Civera; Wanzeng Kong
>
> **备注:** 16 pages, 4 figures, 3 tables
>
> **摘要:** Multi-modal cross-view place recognition remains a fundamental challenge in computer vision and robotics due to the severe viewpoint, modality, and spatial-structure discrepancies between ground observations and aerial references. To address this challenge, we present MAG-VLAQ, a foundation-model-enhanced query aggregation framework for multi-modal aerial-ground cross-view place recognition. Specifically, our approach leverages pre-trained foundation models to extract dense visual tokens from both ground and aerial images, as well as expressive geometric tokens from ground LiDAR observations. These heterogeneous tokens are then projected into a shared embedding space for cross-modal alignment and fusion. As our main contribution, we propose ODE-conditioned VLAQ, which tightly couples neural ordinary differential equations (ODE)-based RGB-LiDAR fusion with vectors of locally aggregated queries (VLAQ). In this design, the VLAQ query centers are dynamically adapted according to the fused multi-modal state. This mechanism allows the final global descriptor to preserve globally learned retrieval prototypes while remaining responsive to scene-specific visual and geometric evidence, significantly improving aerial-ground matching. Extensive experiments on KITTI360-AG and nuScenes-AG validate the effectiveness of our proposed MAG-VLAQ. Notably, on KITTI360-AG, our MAG-VLAQ nearly doubles the state-of-the-art performance, achieving 61.1 Recall@1 in the satellite setting, compared with 34.5 from the closest competing approach.
>
---
#### [new 091] Geometrically Approximated Modeling for Emitter-Centric Ray-Triangle Filtering in Arbitrarily Dynamic LiDAR Simulation
- **分类: cs.GR; cs.PF; cs.RO**

- **简介: 该论文属于实时LiDAR模拟任务，解决动态场景中快速找到射线与三角形交点的问题。提出GRCA算法，通过发射器中心视角优化射线-三角形过滤，提升模拟效率。**

- **链接: [https://arxiv.org/pdf/2605.10457](https://arxiv.org/pdf/2605.10457)**

> **作者:** Rabin Gajmer; Joonas Haapala; Zoltan Beck
>
> **备注:** 21 pages, 20 figures
>
> **摘要:** Real-time Light Detection And Ranging (LiDAR) simulation must find, per emitted ray, the closest intersecting triangle even in dynamic scenes containing large numbers of moving and deformable objects. Dominant acceleration-structure approaches require rebuilding each frame for dynamic geometry -- a cost that compounds directly with scene dynamics and cannot be amortized regardless of how little actually changed. This paper presents the Gajmer Ray-Casting Algorithm (GRCA), which inverts the question: instead of asking what does each ray hit? it asks which rays can each triangle possibly hit? GRCA geometrically models spinning LiDAR emitters as rotation-traced cones or planes and uses each triangle's emitter-centric apparent area to cull, per triangle, which channels and the rays within those channels can possibly reach it -- without any acceleration structure. GRCA is compute-based and vendor-agnostic by design, targeting highly dynamic, high-resolution simultaneous multi-sensor simulation. At its core, GRCA is a general-purpose ray-casting algorithm: the emitter-centric inversion applies to any setting where rays originate from a known position, not only LiDAR. Benchmarks evaluate 2-8 simultaneous 128x4096-ray LiDARs (360deg/180deg) over complex dynamic scenes -- with just two sensors casting ~1M rays per frame. With range culling inactive, GRCA reaches up to 7.97x over hardware-accelerated OptiX (GPU) and 14.55x over Embree (CPU). Two independent extensions further boost performance even in the most complex scene (~22M triangles, ~9M of which are dynamic, 8 LiDARs): range culling at realistic deployment ranges (10-100m) reaches up to 7.02x GPU and 9.33x CPU; a hybrid pipeline -- GRCA for dynamic geometry, OptiX/Embree for static -- reaches up to 10.5x GPU and 19.2x CPU.
>
---
#### [new 092] OpenSGA: Efficient 3D Scene Graph Alignment in the Open World
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D场景图对齐任务，解决机器人在开放世界中高效场景理解与重定位问题。提出OpenSGA框架，融合多模态特征提升对齐精度，并构建大规模数据集ScanNet-SG。**

- **链接: [https://arxiv.org/pdf/2605.10484](https://arxiv.org/pdf/2605.10484)**

> **作者:** Gang Chen; Sebastián Barbas Laina; Stefan Leutenegger; Javier Alonso-Mora
>
> **备注:** 13 figures
>
> **摘要:** Scene graph alignment establishes object correspondences between two 3D scene graphs constructed from partially overlapping observations. This enables efficient scene understanding and object-level relocalization when a robot revisits a place, as well as global map fusion across multiple agents. Such capabilities are essential for robots that require long-term memory for long-horizon tasks involving interactions with the environment. Existing approaches mainly focus on subscan-to-subscan (S2S) alignment and depend heavily on geometric point-cloud features, leaving frame-to-scan (F2S) alignment and open-set vision-language features underexplored. In addition, existing datasets for scene graph alignment remain small-scale with limited object diversity, constraining systematic training and evaluation. We present a unified and efficient scene graph alignment framework that predicts object correspondences by fusing vision-language, textual, and geometric features with spatial context. The framework comprises modules such as a distance-gated spatial attention encoder, a minimum-cost-flow-based allocator, and a global scene embedding generator to achieve accurate alignment even under large coordinate discrepancies. We further introduce ScanNet-SG, a large-scale dataset generated via an automated annotation pipeline with over 700k samples, covering 509 object categories from ScanNet labels and over 3k categories from GPT-4o-based tagging. Experiments show that our method achieves the best overall performance on both F2S and S2S tasks, substantially outperforming existing scene graph alignment methods. Our code and dataset are released at: this https URL.
>
---
#### [new 093] Priority-Driven Control and Communication in Decentralized Multi-Agent Systems via Reinforcement Learning
- **分类: eess.SY; cs.LG; cs.RO**

- **简介: 该论文属于多智能体系统控制任务，解决通信带宽受限问题。通过强化学习联合学习通信优先级与控制策略，避免依赖系统模型。**

- **链接: [https://arxiv.org/pdf/2605.10482](https://arxiv.org/pdf/2605.10482)**

> **作者:** Qingyun Guo; Junyi Shi; Tomasz Piotr Kucner; Dominik Baumann
>
> **备注:** Accepted to the 23rd IFAC World Congress
>
> **摘要:** Event-triggered control provides a mechanism for avoiding excessive use of constrained communication bandwidth in networked multi-agent systems. However, most existing methods rely on accurate system models, which may be unavailable in practice. In this work, we propose a model-free, priority-driven reinforcement learning algorithm that learns communication priorities and control policies jointly from data in decentralized multi-agent systems. By learning communication priorities, we circumvent the hybrid action space typical in event-triggered control with binary communication decisions. We evaluate our algorithm on benchmark tasks and demonstrate that it outperforms the baseline method.
>
---
#### [new 094] Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges
- **分类: cs.LG; cs.MA; cs.RO**

- **简介: 该论文研究多智能体路径规划（MAPF）问题，通过多边缘最优传输和薛定谔桥方法将其转化为可扩展的线性规划问题，实现高效、无冲突的路径规划。**

- **链接: [https://arxiv.org/pdf/2605.10917](https://arxiv.org/pdf/2605.10917)**

> **作者:** Usman A. Khan; Joseph W. Durham
>
> **备注:** Accepted in ICML 2026 as a spotlight paper
>
> **摘要:** We consider anonymous multi-agent path finding (MAPF) where a set of robots is tasked to travel to a set of targets on a finite, connected graph. We show that MAPF can be cast as a special class of multi-marginal optimal transport (MMOT) problems with an underlying Markovian structure, under which the exponentially large MMOT collapses to a linear program (LP) polynomial in size. Focusing on the anonymous setting, we establish conditions under which the corresponding LP is feasible, totally unimodular, and consequently, yields min-cost, integral $(\{0,1\})$ transports that do not overlap in both space and time. To adapt the approach to large-scale problems, we cast the MAPF-MMOT in a probabilistic framework via Schrödinger bridges. Under standard assumptions, we show that the Schrödinger bridge formulation reduces to an entropic regularization of the corresponding MMOT that admits an iterative Sinkhorn-type solution. The Schrödinger bridge, being a probabilistic framework, provides a shadow (fractional) transport that we use as a template to solve a reduced LP and demonstrate that it results in near-optimal, integral transports at a significant reduction in complexity. Extensive experiments highlight the optimality and scalability of the proposed approaches.
>
---
#### [new 095] LoopVLA: Learning Sufficiency in Recurrent Refinement for Vision-Language-Action Models
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出LoopVLA，解决VLA模型在动作预测中过度抽象的问题，通过迭代优化和 Sufficiency 估计提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.09948](https://arxiv.org/pdf/2605.09948)**

> **作者:** Boyang Shen; Kaixiang Yang; Hao Wang; Qiuyu Yu; Qiang Xie; Qiang Li; Zhiwei Wang
>
> **摘要:** Current Vision-Language-Action (VLA) models typically treat the deepest representation of a vision-language backbone as universally optimal for action prediction. However, robotic manipulation is composed of many frequent closed-loop spatial adjustments, for which excessive abstraction may waste computation and weaken low-level geometric cues essential for precise control. Existing early-exit strategies attempt to reduce computation by stopping at predefined layers or applying heuristic rules such as action consistency, but they do not directly answer when a representation is actually sufficient for action. In this paper, we present LoopVLA, a recurrent VLA architecture that jointly learns representation refinement, action prediction, and sufficiency estimation. LoopVLA iteratively applies a shared Transformer block to refine multimodal tokens, and at each iteration produces both a candidate action and a sufficiency score that estimates whether further refinement is necessary. By sharing parameters across iterations, LoopVLA decouples refinement from absolute layer indices and grounds sufficiency estimation in the evolving representation itself. Since sufficiency has no direct supervision, we introduce a self-supervised distribution alignment objective, where intermediate confidence scores are trained to match the relative action quality across refinement steps, thereby linking sufficiency learning to policy optimization signals. Experiments on LIBERO, LIBERO-Plus, and VLA-Arena show that LoopVLA pushes the efficiency-performance frontier of VLA policies, reducing parameters by 45% and improving inference throughput by up to 1.7 times while matching or outperforming strong baselines in task success.
>
---
#### [new 096] MTA-RL: Robust Urban Driving via Multi-modal Transformer-based 3D Affordances and Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决城市驾驶中的场景理解与决策稳定性问题。提出MTA-RL框架，结合多模态Transformer和强化学习，提升驾驶性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.10177](https://arxiv.org/pdf/2605.10177)**

> **作者:** Guangli Chen; Dianzhao Li; Wenjian Zhong; Bangquan Xie; Ostap Okhrin
>
> **摘要:** Robust urban autonomous driving requires reliable 3D scene understanding and stable decision-making under dense interactions. However, existing end-to-end models lack interpretability, while modular pipelines suffer from error propagation across brittle interfaces. This paper proposes MTA-RL, the first framework that bridges perception and control through Multi-modal Transformer-based 3D Affordances and Reinforcement Learning (RL). Unlike previous fusion models that directly regress actions, RGB images and LiDAR point clouds are fused using a transformer architecture to predict explicit, geometry-aware affordance representations. These structured representations serve as a compact observation space, enabling the RL policy to operate purely on predicted driving semantics, which significantly improves sample efficiency and stability. Extensive evaluations in CARLA Town01-03 across varying densities (20-60 background vehicles) show that MTA-RL consistently outperforms state-of-the-art baselines. Trained solely on Town03, our method demonstrates superior zero-shot generalization in unseen towns, achieving up to a 9.0% increase in Route Completion, an 11.0% increase in Total Distance, and an 83.7% improvement in Distance Per Violation. Furthermore, ablation studies confirm that our multi-modal fusion and reward shaping are critical, significantly outperforming image-only and unshaped variants, demonstrating the effectiveness of MTA-RL for robust urban autonomous driving.
>
---
#### [new 097] Temporal Sampling Frequency Matters: A Capacity-Aware Study of End-to-End Driving Trajectory Prediction
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究端到端自动驾驶轨迹预测任务，探讨时间采样频率对模型性能的影响。通过不同频率的采样实验，发现采样频率需根据模型容量调整，而非固定为最高值。**

- **链接: [https://arxiv.org/pdf/2605.10388](https://arxiv.org/pdf/2605.10388)**

> **作者:** Yumao Liu; Tao Liu; Xiangyu Li; Jiaxiang Li; Ke Ma
>
> **摘要:** End to end (E2E) autonomous driving trajectory prediction is often trained with camera frames sampled at the highest available temporal frequency, assuming that denser sampling improves performance. We question this assumption by treating temporal sampling frequency as an explicit training set design variable. Starting from high frequency E2E driving datasets, we construct frequency sweep training sets by temporally subsampling camera frames along each trajectory. For each model dataset pair, we train and evaluate the same model under a fixed protocol, so the frequency response reflects how prediction performance changes with sampling frequency. We analyze this response from a capacity aware perspective. Sparse sampling may miss driving relevant cues, while dense sampling may add redundant visual content and off manifold noise. For finite capacity models, this can create a driving irrelevant capacity burden. We evaluate three smaller E2E models and a larger VLA style AutoVLA model on Waymo, nuScenes, and PAVE. Results show model and dataset dependent frequency responses. Smaller E2E models often show non monotonic or near plateau trends and achieve their best 3 second ADE at lower or intermediate frequencies. In contrast, AutoVLA achieves its best 3 second ADE and FDE at the highest evaluated frequency on all three datasets. Iteration matched controls suggest that the advantage of lower or intermediate frequencies for smaller models is not explained only by unequal training update counts. These findings show that temporal sampling frequency should be reported and tuned, rather than fixed to the highest available value.
>
---
#### [new 098] Benchmarking ResNet Backbones in RT-DETR: Impact of Depth and Regularization under environmental conditions
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于目标检测任务，研究环境变化对RT-DETR模型性能的影响，比较不同ResNet主干网络在光照和背景变化下的表现。**

- **链接: [https://arxiv.org/pdf/2605.08136](https://arxiv.org/pdf/2605.08136)**

> **作者:** Pamela Barboza; Víctor Castelli; Belén Pereira; Ricardo Grando; Bruna de Vargas; Augusto Calfani
>
> **备注:** Accepted at the International Conference on Data Science, Technology and Applications (DATA) 2026
>
> **摘要:** Visual perception plays a central role in competitive robotics, where environmental variations can directly affect real-time detection performance. The related literature on transformer-based detectors lack information regarding the impact of backbone scale and environmental settings on model performance. This work presents a comparative evaluation of RT-DETR for detecting round objects under environmental and hyperparameter variations relevant to competitive robotics. Four ResNet backbones (ResNet18, ResNet34, ResNet50, and ResNet101) were compared using dropout rates, analyzing their effect on confidence and accuracy. All models were trained under the same configuration and evaluated under changes in lighting and background contrast. Environmental conditions primarily impact prediction confidence, while inference latency remains largely unaffected and classification accuracy stays consistently high, approaching or above 1.00 in most cases. Two distinct behaviors were observed. Under illumination variation, ResNet50 achieves the best trade-off, combining near-perfect accuracy, confidence values up to approximately 0.869 and latency around 0.058-0.059 ms. Under background variation, ResNet34 provides the most balanced performance, reaching near-perfect accuracy and higher confidence values up to approximately 0.887. These results indicate that the optimal architecture depends on the type of environmental variation, with intermediate-depth models offering the best balance between performance and efficiency.
>
---
#### [new 099] Decentralized Contingency MPC based on Safe Sets for Nonlinear Multi-agent Collision Avoidance
- **分类: math.OC; cs.MA; cs.RO; eess.SY**

- **简介: 该论文属于多智能体碰撞避免任务，解决非通信环境下分布式避障问题。提出一种基于安全集的应急MPC框架，实现无通信的协同避障与收敛性保证。**

- **链接: [https://arxiv.org/pdf/2605.10738](https://arxiv.org/pdf/2605.10738)**

> **作者:** Max Studt; Georg Schildbach
>
> **摘要:** Decentralized collision avoidance remains challenging, particularly when agents do not communicate any information related to planned trajectories. Most existing approaches either rely on conservative coordination mechanisms or provide limited guarantees on recursive feasibility and convergence. This paper develops a decentralized contingency MPC framework for multi-agent systems with nonlinear dynamics that achieves collision-free motion under a state-only information pattern. Each agent follows the same consensual rule set, enabling safe decentralized planning without communication. Each agent solves a local optimization problem that couples a nominal trajectory with a contingency certificate ensuring a feasible backup maneuver under receding-horizon operation. A novel geometric and decentralized safe-set update mechanism prevents feasibility loss between consecutive time steps. The resulting scheme guarantees recursive feasibility, including collision avoidance, and establishes a Lyapunov-type convergence result to an admissible safe equilibrium. Simulation results demonstrate performance in both sparse and dense multi-agent environments, including cluttered bottleneck scenarios and under plug-and-play operation.
>
---
#### [new 100] SceneFactory: GPU-Accelerated Multi-Agent Driving Simulation with Physics-Based Vehicle Dynamics
- **分类: cs.MA; cs.RO**

- **简介: 该论文提出SceneFactory，一个基于GPU的多智能体驾驶仿真平台，解决物理真实与并行效率的平衡问题，实现高吞吐量、物理逼真的自动驾驶训练。**

- **链接: [https://arxiv.org/pdf/2605.08528](https://arxiv.org/pdf/2605.08528)**

> **作者:** Yicheng Zhu; Yang Chen; Tao Li; Zilin Bian
>
> **摘要:** Autonomous-driving simulators typically trade physical fidelity for scalable parallelism. Physics-based platforms such as CARLA and MetaDrive provide articulated vehicle dynamics and contact, but their non-vectorized interfaces make batched training difficult. GPU-batched systems such as Waymax and GPUDrive scale to hundreds of scenarios by replacing rigid-body physics with simplified kinematic models, omitting tire--road interaction, suspension, contact dynamics, and road-condition-dependent friction. We introduce SceneFactory, a GPU-vectorized platform for procedural scene construction, physics-based multi-agent simulation, and RL in autonomous-driving environments. Built on NVIDIA Isaac Sim + Isaac Lab, SceneFactory represents worlds and agents as batched tensors: control, observations, rewards, resets, and policy inference run as GPU tensor operations over the Isaac Lab tensor API. SceneFactory converts Waymo Open Motion Dataset road topologies into simulation-ready USD worlds, runs many worlds concurrently on one GPU, populates each with multiple articulated PhysX vehicles, and maps precipitation and road-surface type to PhysX material friction coefficients. With GPU vectorization, SceneFactory achieves up to 127$\times$ higher throughput than a non-vectorized PhysX baseline on the same GPU and physics solver, reaching 19,250 controlled-agent simulation steps per second at 256 worlds $\times$ 16 agents. Cross-simulator transfer reveals an asymmetric dynamics gap: physics-grounded RL policies transfer to a simplified kinematic bicycle model with 99.5% success, whereas reverse transfer drops to 47.3%. Under wet-road friction, friction-aware policies reduce mean peak DRAC from 58.7 to 27.8,m/s$^2$ without sacrificing goal reach. SceneFactory shows that scalable autonomous-driving training need not discard articulated rigid-body dynamics or physically grounded road-condition variation.
>
---
#### [new 101] C-CoT: Counterfactual Chain-of-Thought with Vision-Language Models for Safe Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶安全决策任务，旨在解决复杂场景下的风险预测与决策问题。提出C-CoT框架，结合视觉语言模型，提升安全性和可解释性。**

- **链接: [https://arxiv.org/pdf/2605.10744](https://arxiv.org/pdf/2605.10744)**

> **作者:** Kefei Tian; Yuansheng Lian; Kai Yang; Xiangdong Chen; Shen Li
>
> **摘要:** Safety-critical planning in complex environments, particularly at urban intersections, remains a fundamental challenge for autonomous driving. Existing methods, whether rule-based or data-driven, frequently struggle to capture complex scene semantics, infer potential risks, and make reliable decisions in rare, high-risk situations. While vision-language models (VLMs) offer promising approaches for safe decision-making in these environments, most current approaches lack reflective and causal reasoning, thereby limiting their overall robustness. To address this, we propose a counterfactual chain-of-thought (C-CoT) framework that leverages VLMs to decompose driving decisions into five sequential stages: scene description, critical object identification, risk prediction, counterfactual risk reasoning, and final action planning. Within the counterfactual reasoning stage, we introduce a structured meta-action evaluation tree to explicitly assess the potential consequences of alternative action combinations. This self-reflective reasoning establishes causal links between action choices and safety outcomes, improving robustness in long-tail and out-of-distribution scenarios. To validate our approach, we construct the DeepAccident-CCoT dataset based on the DeepAccident benchmark and fine-tune a Qwen2.5-VL (7B) model using low-rank adaptation. Our model achieves a risk prediction recall of 81.9%, reduces the collision rate to 3.52%, and lowers L2 error to 1.98 m. Ablation studies further confirm the critical role of counterfactual reasoning and the meta-action evaluation tree in enhancing safety and interpretability.
>
---
#### [new 102] Quantile-Coupled Flow Matching for Distributional Reinforcement Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于分布强化学习任务，旨在解决CFM批评者与Wasserstein距离不匹配的问题。提出FlowIQN，通过量化对齐实现Wasserstein对齐的流匹配，提升分布准确性。**

- **链接: [https://arxiv.org/pdf/2605.08515](https://arxiv.org/pdf/2605.08515)**

> **作者:** Michael Groom; Victor-Alexandru Darvariu; Lars Kunze; James Wilson; Nick Hawes
>
> **摘要:** Unlike standard expected-return Reinforcement Learning (RL), Distributional RL (DRL) models the full return distribution, making it better-suited for uncertainty-aware and risk-sensitive decision-making. Conditional Flow Matching (CFM) critics have recently attracted attention for modelling continuous, multi-modal return distributions. Despite this interest, there remains a substantial metric mismatch: DRL theory relies on the distributional Bellman operator being contractive in the $p$-Wasserstein distance, yet existing CFM critics are trained with arbitrary source-target couplings, so their flow-matching losses are not Wasserstein-aligned surrogates for matching Bellman target return distributions. In this work, we address this mismatch by proposing FlowIQN, a CFM critic that sorts source and Bellman target samples within each mini-batch to approximate the monotone optimal transport coupling, replacing arbitrary pairings with quantile-aligned flow paths. We prove that the loss of our quantile-coupled CFM critic yields a Wasserstein-aligned approximate projection compatible with the foundations of DRL. To our knowledge, FlowIQN is the first flow-matching distributional critic with an explicit Wasserstein-aligned projection guarantee. We further extend FlowIQN with shortcut models for efficient inference. Empirical results show that FlowIQN improves Wasserstein return-distribution accuracy over other CFM critics. It also yields competitive performance on offline RL benchmarks across multiple policy extraction methods, providing a theoretically grounded CFM critic that is readily compatible with DRL pipelines. Code: this https URL.
>
---
#### [new 103] Flame3D: Zero-shot Compositional Reasoning of 3D Scenes with Agentic Language Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出Flame3D，解决3D场景理解任务中的组合推理问题。无需训练，通过可编辑的视觉文本记忆和空间工具实现开放推理。**

- **链接: [https://arxiv.org/pdf/2605.09218](https://arxiv.org/pdf/2605.09218)**

> **作者:** Sagar Bharadwaj; Ziyong Ma; Anurag Ghosh; Srinivasan Seshan; Anthony Rowe
>
> **摘要:** 3D scene understanding spans reasoning about free space, object grounding, hypothetical object insertions, complex geometric relationships, and integrating all of these with external tools and data sources. Existing 3D understanding methods typically rely on large-scale 3D-language training or focus on object grounding and simple spatial relationships. We argue that the broad generalization that motivates 3D-language training can be achieved at inference time, without 3D-specific training. We propose Flame3D, a training-free framework that represents scenes as editable visual-textual 3D memories and exposes them to an off-the-shelf MLLM through composable spatial tools. Flame3D also lets the agent synthesize custom spatial programs at inference time, enabling open-ended reasoning over layouts, empty space, and objects not yet present in the scene. External data and corrections can be added to the memory without retraining. In addition to showing competitive performance to finetuned 3D-LMM methods on ScanQA, we study multi-hop 3D reasoning capabilities of Flame3D by evaluating it on a curated compositional spatial-reasoning benchmark, Compose3D. We find that fixed tools fall short and that the agent's ability to synthesize spatial operations at inference time is essential. These results invite the question: should future progress in 3D scene understanding focus on richer scene memories and expressive compositional abstractions?
>
---
#### [new 104] Is Your Driving World Model an All-Around Player?
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于驾驶世界模型评估任务，旨在解决现有模型在物理和行为真实性上的不足。提出WorldLens基准，涵盖多维度评估，并构建人类偏好数据集与评估代理。**

- **链接: [https://arxiv.org/pdf/2605.10858](https://arxiv.org/pdf/2605.10858)**

> **作者:** Lingdong Kong; Ao Liang; Tianyi Yan; Hongsi Liu; Wesley Yang; Ziqi Huang; Xian Sun; Wei Yin; Jialong Zuo; Yixuan Hu; Dekai Zhu; Dongyue Lu; Youquan Liu; Guangfeng Jiang; Linfeng Li; Xiangtai Li; Long Zhuo; Lai Xing Ng; Benoit R. Cottereau; Changxin Gao; Liang Pan; Wei Tsang Ooi; Ziwei Liu
>
> **备注:** CVPR 2026 VideoWorldModel Workshop; Project Page at this https URL GitHub at this https URL
>
> **摘要:** Today's driving world models can generate remarkably realistic dash-cam videos, yet no single model excels universally. Some generate photorealistic textures but violate basic physics; others maintain geometric consistency but fail when subjected to closed-loop planning. This disconnect exposes a critical gap: the field evaluates how real generated worlds appear, but rarely whether they behave realistically. We introduce WorldLens, a unified benchmark that measures world-model fidelity across the full spectrum, from pixel quality and 4D geometry to closed-loop driving and human perceptual alignment, through five complementary aspects and 24 standardized dimensions. Our evaluation of six representative models reveals that no existing approach dominates across all axes: texture-rich models violate geometry, geometry-aware models lack behavioral fidelity, and even the strongest performers achieve only 2-3 out of 10 on human realism ratings. To bridge algorithmic metrics with human perception, we further contribute WorldLens-26K, a 26,808-entry human-annotated preference dataset pairing numerical scores with textual rationales, and WorldLens-Agent, a vision-language evaluator distilled from these judgments that enables scalable, explainable auto-assessment. Together, the benchmark, dataset, and agent form a unified ecosystem for assessing generated worlds not merely by visual appeal, but by physical and behavioral fidelity.
>
---
#### [new 105] NEXUS: Continual Learning of Symbolic Constraints for Safe and Robust Embodied Planning
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决LLM在物理世界中安全与确定性不足的问题。提出NEXUS框架，实现持续学习与符号约束的结合，提升规划安全性和效率。**

- **链接: [https://arxiv.org/pdf/2605.09387](https://arxiv.org/pdf/2605.09387)**

> **作者:** Tiehan Cui; Peipei Liu; Yanxu Mao; Congying Liu; Mingzhe Xing; Datao You
>
> **摘要:** While Large Language Models (LLMs) have catalyzed progress in embodied intelligence, a fundamental gap between their inherent probabilistic uncertainty and the strict determinism and verifiable safety required in the physical world. To mitigate this gap, this paper introduces NEXUS, a modular framework designed for continual learning in embodied agents. Different from prior works that treat symbolic artifacts merely as static interfaces, NEXUS leverages them for symbolic grounding and knowledge evolution. The framework explicitly decouples physical feasibility from safety specifications: capability of agents is improved through closed-loop execution feedback, while probabilistic risk assessments are grounded into deterministic hard constraints to establish a rigorous pre-action defense. Experiments on SafeAgentBench demonstrate that NEXUS achieves superior task success rates while effectively refusing unsafe instructions, exhibiting robust defense against adversarial attacks, and progressively improving planning efficiency through knowledge accumulation.
>
---
#### [new 106] VECTOR-Drive: Tightly Coupled Vision-Language and Trajectory Expert Routing for End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出VECTOR-DRIVE，解决端到端自动驾驶中多模态耦合与任务冲突问题。通过共享注意力和语义路由，实现视觉语言与轨迹预测的紧密协同。**

- **链接: [https://arxiv.org/pdf/2605.08830](https://arxiv.org/pdf/2605.08830)**

> **作者:** Rui Zhao; Jianlin Yu; Zhenhai Gao; Jiaqiao Liu; Fei Gao
>
> **摘要:** End-to-end autonomous driving requires models to understand traffic scenes, infer driving intent, and generate executable motion plans. Recent vision-language-action (VLA) models inherit semantic priors from large-scale vision-language pretraining, yet still face a coupling trade-off: fully shared backbones preserve multimodal interaction but may entangle language reasoning and trajectory prediction, whereas decou pled reasoning-action pipelines reduce task conflict but weaken semantic-motion coupling. We propose VECTOR-DRIVE, a tightly coupled VLA framework built on Qwen2.5-VL-3B. VECTOR-DRIVE keeps all tokens coupled through shared self attention and routes feed-forward computation according to token semantics. Vision and language tokens are processed by a Vision-Language Expert to preserve semantic priors, while target-point, ego-state, and noisy action tokens are routed to a Trajectory Expert for motion-specific computation. On the action-token pathway, a flow-matching planner refines noisy action tokens into future waypoints and speed profiles. This design couples semantic reasoning and motion planning within a single multimodal Transformer while separating task-specific FFN computation. On Bench2Drive, VECTOR-DRIVE achieves 88.91 Driving Score and outperforms representative end-to end and VLA-based baselines. Qualitative results and ablations further validate the benefits of shared attention, semantic-aware expert routing, progressive training, and flow-based action de coding.
>
---
#### [new 107] VISOR: A Vision-Language Model-based Test Oracle for Testing Robot
- **分类: cs.SE; cs.RO**

- **简介: 该论文提出VISOR，一种基于视觉-语言模型的测试Oracle方法，用于自动化评估机器人任务的正确性和质量，解决传统测试中依赖人工或特定符号Oracle的问题。**

- **链接: [https://arxiv.org/pdf/2605.10408](https://arxiv.org/pdf/2605.10408)**

> **作者:** Prasun Saurabh; Pablo Valle; Aitor Arrieta; Shaukat Ali; Paolo Arcaini
>
> **摘要:** Testing robots requires assessing whether they perform their intended tasks correctly, dependably, and with high quality, a challenge known as the test oracle problem in software testing. Traditionally, this assessment relies on task-specific symbolic oracles for task correctness and on human manual evaluation of robot behavior, which is time-consuming, subjective, and error-prone. To address this, we propose VISOR, a Vision-Language Model (VLM)-based approach for automated test oracle assessment that eliminates the need of expensive human evaluations. VISOR performs automated evaluation of task correctness and quality, addressing the limitations of existing symbolic test oracles, which are task-specific and provide pass/fail judgments without explicitly quantifying task quality. Given the inherent uncertainty in VLMs, VISOR also explicitly quantifies its own uncertainty during test assessments. We evaluated VISOR using two VLMs, i.e., GPT and Gemini, across four robotic tasks on over 1,000 videos. Results show that Gemini achieves higher recall while GPT achieves higher precision. However, both models show low correlation between uncertainty and correctness, which prevents using uncertainty as a correctness predictor.
>
---
#### [new 108] DeformMaster: An Interactive Physics-Neural World Model for Deformable Objects from Videos
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DeformMaster，属于物理-神经世界模型任务，解决从视频中学习可变形物体的物理动态与外观问题，通过神经残差补偿未建模效应，实现高保真动态渲染与交互。**

- **链接: [https://arxiv.org/pdf/2605.09586](https://arxiv.org/pdf/2605.09586)**

> **作者:** Can Li; Zhoujian Li; Ren Li; Jie Gu; Lei Lei; Jingmin Chen; Lei Sun
>
> **摘要:** World models for deformable objects should recover not only geometry and appearance, but also underlying physical dynamics, interaction grounding, and material behavior. Learning such a model from real videos is challenging because deformable linear, planar, and volumetric objects evolve under high-dimensional deformation, noisy interactions, and complex material response. The model must therefore infer a physical state from visual observations, roll it forward under new interactions, and render the resulting dynamics with high visual fidelity. We present DeformMaster, a video-derived interactive physics--neural world model that turns real interaction videos into an online interactive model of deformable objects within a unified dynamics-and-appearance framework. DeformMaster preserves structured physical rollout while using a neural residual to compensate for unmodeled effects, grounds sparse hand motion as distributed compliant actuator for hand--continuum interaction, represents material response with spatially varying constitutive experts, and drives high-fidelity 4D appearance from the predicted physical evolution. Experiments on real-world deformable-object sequences demonstrate DeformMaster's ability to roll out future dynamics and render dynamic appearance, outperforming state-of-the-art baselines while supporting novel action rollout, material-parameter variation, and dynamic novel-view synthesis.
>
---
#### [new 109] CapVector: Learning Transferable Capability Vectors in Parametric Space for Vision-Language-Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型的微调任务，旨在解决预训练模型在微调时效果不佳、计算成本高的问题。通过分离参数空间中的两个目标，生成能力向量以提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.10903](https://arxiv.org/pdf/2605.10903)**

> **作者:** Wenxuan Song; Han Zhao; Fuhao Li; Ziyang Zhou; Xi Wang; Jing Lyu; Pengxiang Ding; Yan Wang; Donglin Wang; Haoang Li
>
> **摘要:** This paper proposes a novel approach to address the challenge that pretrained VLA models often fail to effectively improve performance and reduce adaptation costs during standard supervised finetuning (SFT). Some advanced finetuning methods with auxiliary training objectives can improve performance and reduce the number of convergence steps. However, they typically incur significant computational overhead due to the additional losses from auxiliary objectives. To simultaneously achieve the enhanced capabilities of auxiliary training with the simplicity of standard SFT, we decouple the two objectives of auxiliary-objective SFT within the parameter space, namely, enhancing general capabilities and fitting task-specific action distributions. To deliver the goal, we only need to train the model to converge on a small-scale task set using two distinct training strategies, resulting in two finetuned models. The parameters' difference between the two models can then be interpreted as capability vectors provided by auxiliary objectives. These vectors are then merged with pretrained parameters to form a capability-enhanced meta model. Moreover, when standard SFT is augmented with a lightweight orthogonal regularization loss, the merged model attains performance comparable to auxiliary finetuned baselines with reduced computational overhead. Internal and external experiments demonstrate that our capability vectors (1) are effective and versatile across diverse models, (2) can generalize to novel environments and embodiments out of the box.
>
---
#### [new 110] xApp Empowered Resource Management for Non-Terrestrial Users in 5G O-RAN Networks
- **分类: eess.SP; cs.RO**

- **简介: 该论文属于5G O-RAN网络中的资源管理任务，旨在解决无人机移动性管理问题。通过DDQN强化学习优化切换决策，提升连接可靠性并减少切换频率。**

- **链接: [https://arxiv.org/pdf/2605.10704](https://arxiv.org/pdf/2605.10704)**

> **作者:** Mohammed M.H. Qazzaz; Syed Ali Zaidi; Aubida A. Al-Hameed; Abdelaziz Salama; Des Mclernon
>
> **摘要:** This paper introduces a proactive Unmanned Aerial Vehicle (UAV) mobility management xApp for Open Radio Access Network (O-RAN) Near Real-Time Radio Intelligent Controller (Near-RT RIC) environments, employing Double Deep Q-Network (DDQN) reinforcement learning (RL) enhanced with transfer learning to optimise handover decisions for UAVs operating along predetermined flight trajectories. Unlike reactive approaches that respond to signal degradation, the proposed framework anticipates network conditions and minimises both outage probability and handover frequency through predictive optimisation. The system leverages centralised weight averaging to consolidate knowledge from multiple flight scenarios into a global model capable of generalising to previously unseen operational environments without extensive retraining. A comprehensive evaluation demonstrates that the proposed framework achieves a favourable trade-off between handover frequency and connectivity reliability, reducing handover events by up to 54.6% compared to greedy approaches while maintaining outage probability at practically negligible levels. The results validate the effectiveness of intelligent learning-based approaches for UAV mobility management in next-generation O-RAN architectures, thereby contributing to seamless integration of aerial user equipment into cellular networks.
>
---
#### [new 111] PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于手-物体交互重建任务，旨在解决非刚性物体与手部的3D重建问题。通过物理模拟实现手与变形物体的协同重建与优化。**

- **链接: [https://arxiv.org/pdf/2605.09538](https://arxiv.org/pdf/2605.09538)**

> **作者:** Jihyun Lee; Changmin Lee; Donghwan Kim; Tae-Kyun Kim
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** While existing methods for reconstructing hand-object interactions have made impressive progress, they either focus on rigid or part-wise rigid objects-limiting their ability to model real-world objects (e.g., cloth, stuffed animals) that exhibit highly non-rigid deformations-or model deformable objects without full 3D hand reconstruction. To bridge this gap, we present PhysHanDI (Physics-based Reconstruction of Hand and Deformable Object Interactions), a framework that enables full 3D reconstruction of both interacting hands and non-rigid objects. Our key idea is to physically simulate object deformations driven by forces induced from densely reconstructed 3D hand motions, ensuring that the reconstructed object dynamics are both physically plausible and coherent with the interacting hand movements. Furthermore, we demonstrate that such simulation of object deformations can, in turn, refine and improve hand reconstruction via inverse physics. In experiments, PhysHanDI outperforms the state-of-the-art baseline across reconstruction and future prediction.
>
---
#### [new 112] PaMoSplat: Part-Aware Motion-Guided Gaussian Splatting for Dynamic Scene Reconstruction
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出PaMoSplat，用于动态场景重建，解决复杂运动下高保真渲染与精准跟踪问题，通过引入部件感知和运动引导实现更优效果。**

- **链接: [https://arxiv.org/pdf/2605.10307](https://arxiv.org/pdf/2605.10307)**

> **作者:** Yinan Deng; Jianyu Dou; Jiahui Wang; Jingyu Zhao; Yi Yang; Yufeng Yue
>
> **备注:** Accepted by TCSVT. Project Url: this https URL
>
> **摘要:** Dynamic scene reconstruction represents a fundamental yet demanding challenge in computer vision and robotics. While recent progress in 3DGS-based methods has advanced dynamic scene modeling, obtaining high-fidelity rendering and accurate tracking in scenarios with substantial, intricate motions remains significantly challenging. To address these challenges, we propose PaMoSplat, a novel dynamic Gaussian splatting framework incorporating part awareness and motion priors. Our approach is grounded in two key observations: 1) Parts serve as primitives for scene deformation, and 2) Motion cues from optical flow can effectively guide part motion. Specifically, PaMoSplat initializes by lifting multi-view segmentation masks into 3D space via graph clustering, establishing coherent Gaussian parts. For subsequent timestamps, we leverage a differential evolutionary algorithm to estimate the rigid motion of these parts using multi-view optical flow cues, providing a robust warm-start for further optimization. Additionally, PaMoSplat introduces an adaptive iteration count mechanism, internal learnable rigidity, and flow-supervised rendering loss to accelerate and optimize the training process. Comprehensive evaluations across diverse scenes, including real-world environments, demonstrate that PaMoSplat delivers superior rendering quality, improved tracking precision, and faster convergence compared to existing methods. Furthermore, it enables multiple part-level downstream applications, such as 4D scene editing.
>
---
#### [new 113] RigidFormer: Learning Rigid Dynamics using Transformers
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.RO**

- **简介: 该论文提出RigidFormer，用于学习无网格的刚体动力学模拟。解决多物体接触和长时序误差累积问题，通过Transformer模型实现高效、可扩展的刚体模拟。**

- **链接: [https://arxiv.org/pdf/2605.09196](https://arxiv.org/pdf/2605.09196)**

> **作者:** Zhiyang Dou; Minghao Guo; Haixu Wu; Doug Roble; Tuur Stuyck; Wojciech Matusik
>
> **备注:** Project Page: this https URL
>
> **摘要:** Learning-based simulation of multi-object rigid-body dynamics remains difficult because contact is discontinuous and errors compound over long horizons. Most existing methods remain tied to mesh connectivity and vertex-level message passing, which limits their applicability to mesh-free inputs such as point clouds and leads to high computational cost. Efficiently modeling high-fidelity rigid-body dynamics from mesh-free representations, therefore, remains challenging. We introduce RigidFormer, an object-centric Transformer-based model that learns mesh-free rigid-body dynamics with controllable integration step sizes. RigidFormer reasons at the object level and advances each object through compact anchors; Anchor-Vertex Pooling enriches these anchors with local vertex features, retaining contact-relevant geometry without dense vertex-level interaction. We propose Anchor-based RoPE to inject anchor geometry into attention while respecting the unordered nature of objects and anchors: object-token processing is permutation-equivariant, and the mean-pooled anchor descriptor is invariant to anchor reindexing while preserving shape extent. RigidFormer further enforces rigidity by projecting updates onto the rigid-body manifold using differentiable Kabsch alignment. On standard benchmarks, RigidFormer outperforms or matches mesh-based baselines using point inputs, runs faster, generalizes to unseen point resolutions and across datasets, and scales to 200+ objects; we also show a preliminary extension to command-conditioned articulated bodies by treating body parts as interacting object-level components.
>
---
## 更新

#### [replaced 001] Learning When to Jump for Off-road Navigation
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决越野行驶中速度与安全的平衡问题。提出MAT表示法，建模地形成本与速度的关系，提升导航效率与安全性。**

- **链接: [https://arxiv.org/pdf/2602.00877](https://arxiv.org/pdf/2602.00877)**

> **作者:** Zhipeng Zhao; Taimeng Fu; Shaoshu Su; Qiwei Du; Ehsan Tarkesh Esfahani; Karthik Dantu; Souma Chowdhury; Chen Wang
>
> **摘要:** Low speed does not always guarantee safety in off-road driving. For instance, crossing a ditch may be risky at a low speed due to the risk of getting stuck, yet safe at a higher speed with a controlled, accelerated jump. Achieving such behavior requires path planning that explicitly models complex motion dynamics, whereas existing methods often neglect this aspect and plan solely based on positions or a fixed velocity. To address this gap, we introduce Motion-aware Traversability (MAT) representation to explicitly model terrain cost conditioned on actual robot motion. Instead of assigning a single scalar score for traversability, MAT models each terrain region as a Gaussian function of velocity. During online planning, we decompose the terrain cost computation into two stages: (1) predict terrain-dependent Gaussian parameters from perception in a single forward pass, (2) efficiently update terrain costs for new velocities inferred from current dynamics by evaluating these functions without repeated inference. We develop a system that integrates MAT to enable agile off-road navigation and evaluate it in both simulated and real-world environments with various obstacles. Results show that MAT achieves real-time efficiency and enhances the performance of off-road navigation, reducing path detours by 75% while maintaining safety across challenging terrains.
>
---
#### [replaced 002] Constraint-Aware Reinforcement Learning via Adaptive Action Scaling
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于安全强化学习任务，旨在解决探索过程中约束违规问题。提出一种模块化调节器，通过自适应动作缩放减少违规，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2510.11491](https://arxiv.org/pdf/2510.11491)**

> **作者:** Murad Dawood; Usama Ahmed Siddiquie; Shahram Khorshidi; Maren Bennewitz
>
> **备注:** Accepted in 8th Annual Learning for Dynamics & Control Conference (L4DC)
>
> **摘要:** Safe reinforcement learning (RL) seeks to mitigate unsafe behaviors that arise from exploration during training by reducing constraint violations while maintaining task performance. Existing approaches typically rely on a single policy to jointly optimize reward and safety, which can cause instability due to conflicting objectives, or they use external safety filters that override actions and require prior system knowledge. In this paper, we propose a modular cost-aware regulator that scales the agent's actions based on predicted constraint violations, preserving exploration through smooth action modulation rather than overriding the policy. The regulator is trained to minimize constraint violations while avoiding degenerate suppression of actions. Our approach integrates seamlessly with off-policy RL methods such as SAC and TD3, and achieves state-of-the-art return-to-cost ratios on Safety Gym locomotion tasks with sparse costs, reducing constraint violations by up to 126 times while increasing returns by over an order of magnitude compared to prior methods.
>
---
#### [replaced 003] Visibility-Aware Mobile Grasping in Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文属于移动抓取任务，解决动态环境中机器人在有限视野下安全导航与抓取的问题。提出统一系统，结合低层规划与高层行为树，提升抓取成功率与安全性。**

- **链接: [https://arxiv.org/pdf/2605.02487](https://arxiv.org/pdf/2605.02487)**

> **作者:** Tianrun Hu; Anxing Xiao; David Hsu; Hanbo Zhang
>
> **摘要:** This paper addresses the problem of mobile grasping in dynamic, unknown environments where a robot must operate under a limited field-of-view. The fundamental challenge is the inherent trade-off between ``seeing'' around to reduce environmental uncertainty and ``moving'' the body to achieve task progress in a high-dimensional configuration space, subject to visibility constraints. Previous approaches often assume known or static environments and decouple these objectives, failing to guarantee safety when unobserved dynamic obstacles intersect the robot's path during manipulation. In this paper, we propose a unified mobile grasping system comprising two core components: (1) an iterative low-level whole-body planner coupled with velocity-aware active perception to navigate dynamic environments safely; and (2) a hierarchical high-level planner based on behavior trees that adaptively generates subgoals to guide the robot through exploration and runtime failures. We provide experimental results across 400 randomized simulation scenarios and real-world deployment on a Fetch mobile manipulator. Results show that our system achieves a success rate of 68.8\% and 58.0\% in unknown static and dynamic environments, respectively, significantly boosting success rates by 22.8\% and 18.0\% over the \nam approach in both unknown static and dynamic environments, with improved collision safety.
>
---
#### [replaced 004] MOBIUS: A Multi-Modal Bipedal Robot that can Walk, Crawl, Climb, and Roll
- **分类: cs.RO; eess.SY**

- **简介: 该论文介绍MOBIUS机器人，解决多模式移动与操作问题，通过混合控制和高阶规划实现行走、攀爬等动作，提升机器人交互能力。**

- **链接: [https://arxiv.org/pdf/2511.01774](https://arxiv.org/pdf/2511.01774)**

> **作者:** Alexander Schperberg; Yusuke Tanaka; Stefano Di Cairano; Dennis Hong
>
> **备注:** Paper is accepted at the Robotics: Science and Systems conference, held in Sydney, Australia, July 13th-17th, 2026. Alexander Schperberg and Yusuke Tanaka are co-first authors. Both were at the Robotics and Mechanisms Laboratory (RoMeLa) at UCLA when the work started, and are now with Mitsubishi Electric Research Laboratories and ETH Zurich (RSL) respectively
>
> **摘要:** This paper presents the MOBIUS platform, a bipedal robot capable of walking, crawling, climbing, and rolling. MOBIUS features four limbs, two 6-DoF arms with two-finger grippers for manipulation and climbing, and two 4-DoF legs for locomotion--enabling smooth transitions across diverse terrains without reconfiguration. A hybrid control architecture combines reinforcement learning for locomotion and force control for compliant contact interactions during manipulation. A high-level MIQCP planner autonomously selects locomotion modes to balance stability and energy efficiency. Hardware experiments demonstrate robust gait transitions, dynamic climbing, and full-body load support via pinch grasp. Overall, MOBIUS demonstrates the importance of tight integration between morphology, high-level planning, and control to enable mobile loco-manipulation and grasping, substantially expanding its interaction capabilities, workspace, and traversability.
>
---
#### [replaced 005] Xiaomi OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文提出OneVL，解决VLA自动驾驶中轨迹预测的延迟问题。通过融合视觉-语言解释与世界模型，实现高效且准确的潜在推理与规划。**

- **链接: [https://arxiv.org/pdf/2604.18486](https://arxiv.org/pdf/2604.18486)**

> **作者:** Jinghui Lu; Jiayi Guan; Zhijian Huang; Jinlong Li; Guang Li; Lingdong Kong; Yingyan Li; Han Wang; Shaoqing Xu; Yuechen Luo; Fang Li; Chenxu Dang; Junli Wang; Tao Xu; Jing Wu; Jianhua Wu; Xiaoshuai Hao; Wen Zhang; Tianyi Jiang; Lingfeng Zhang; Lei Zhou; Yingbo Tang; Jie Wang; Yinfeng Gao; Xizhou Bu; Haochen Tian; Yihang Qiu; Feiyang Jia; Lin Liu; Yigu Ge; Hanbing Li; Yuannan Shen; Jianwei Cui; Hongwei Xie; Bing Wang; Haiyang Sun; Jingwei Zhao; Jiahui Huang; Pei Liu; Zeyu Zhu; Yuncheng Jiang; Zibin Guo; Chuhong Gong; Hanchao Leng; Kun Ma; Naiyan Wang; Guang Chen; Kuiyuan Yang; Hangjun Ye; Long Chen
>
> **备注:** Technical Report; 49 pages, 22 figures, 10 tables; Project Page at this https URL GitHub at this https URL
>
> **摘要:** Chain-of-Thought (CoT) reasoning has become a powerful driver of trajectory prediction in VLA-based autonomous driving, yet its autoregressive nature imposes a latency cost that is prohibitive for real-time deployment. Latent CoT methods attempt to close this gap by compressing reasoning into continuous hidden states, but consistently fall short of their explicit counterparts. We suggest that this is due to purely linguistic latent representations compressing a symbolic abstraction of the world, rather than the causal dynamics that actually govern driving. Thus, we present OneVL (One-step latent reasoning and planning with Vision-Language explanations), a unified VLA and World Model framework that routes reasoning through compact latent tokens supervised by dual auxiliary decoders. Alongside a language decoder that reconstructs text CoT, we introduce a visual world model decoder that predicts future-frame tokens, forcing the latent space to internalize the causal dynamics of road geometry, agent motion, and environmental change. A three-stage training pipeline progressively aligns these latents with trajectory, language, and visual objectives, ensuring stable joint optimization. In inference, the auxiliary decoders are discarded, and all latent tokens are prefilled in a single parallel pass, matching the speed of answer-only prediction. Across four benchmarks, OneVL becomes the first latent CoT method to surpass explicit CoT, delivering superior accuracy at answer-only latency. These results show that with world model supervision, latent CoT produces more generalizable representations than verbose token-by-token reasoning. Code has been open-sourced to the community. Project Page: this https URL
>
---
#### [replaced 006] Integrated Hierarchical Decision-Making in Inverse Kinematic Planning and Control
- **分类: cs.RO**

- **简介: 该论文属于机器人逆运动学规划与控制任务，解决复杂非线性层次决策问题。提出一种高效、灵活的稀疏分层非线性规划方法，提升末端执行器位置选择与双臂抓取的精度与效率。**

- **链接: [https://arxiv.org/pdf/2412.01324](https://arxiv.org/pdf/2412.01324)**

> **作者:** Kai Pfeiffer; Quan Zhang; Yuqing Chen; Gordon Boateng; Yuquan Wang; Vincent Bonnet; Aberrahmane Kheddar
>
> **摘要:** This work presents a novel and efficient nonlinear programming framework that tightly integrates hierarchical decision-making with whole-body inverse kinematic planning and control. Decision-making plays a central role in many aspects of robotics, from sparse inverse kinematic control with a minimal number of joints, to inverse kinematic planning while simultaneously selecting a discrete end-effector location from multiple candidates. Current approaches often rely on heavy computations using mixed-integer nonlinear programming, separate decision-making from inverse kinematics (some times approximated by reachability methods), or employ efficient but less versatile $\ell_1$-norm formulations of linear sparse programming, without addressing the underlying nonlinear problem formulations. In contrast, the proposed sparse hierarchical nonlinear programming solver is efficient, versatile, and accurate by exploiting sparse hierarchical structure and leveraging the $\ell_0$-norm which is rarely used in robotics. The solver efficiently tackles complex nonlinear hierarchical decision-making problems previously unaddressed in the literature, such as inverse kinematic planning with simultaneous prioritized selection of end-effector locations from a large set of candidates, or inverse kinematic control with simultaneous selection of bi-manual grasp locations on a randomly rotated box.
>
---
#### [replaced 007] Morphology-Aware Graph Reinforcement Learning for Tensegrity Robot Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决张力结构机器人运动控制难题。通过融合图神经网络与强化学习，提升其运动性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2510.26067](https://arxiv.org/pdf/2510.26067)**

> **作者:** Chi Zhang; Mingrui Li; Wenzhe Tong; Xiaonan Huang
>
> **备注:** 8 pages, 10 figures. Project page: this https URL
>
> **摘要:** Tensegrity robots combine rigid rods and elastic cables, offering high resilience and deployability but at the same time posing major challenges for locomotion control due to their underactuated and highly coupled dynamics. This paper introduces a morphology-aware reinforcement learning framework that integrates a graph neural network (GNN) into the Soft Actor-Critic (SAC) algorithm. By representing the robot's physical topology as a graph, the proposed GNN-based policy captures coupling among components, enabling faster and more stable learning than conventional multilayer perceptron (MLP) policies. The method is validated on a physical 3-bar tensegrity robot across three locomotion primitives, including straight-line tracking and bidirectional turning. It shows superior sample efficiency, robustness to noise and stiffness variations, and improved trajectory accuracy. Additionally, the learned policies transfer directly from simulation to hardware without fine-tuning, achieving stable real-world locomotion. These results demonstrate the advantages of incorporating structural priors into reinforcement learning for tensegrity robot control.
>
---
#### [replaced 008] Force Policy: Learning Hybrid Force-Position Control Policy under Interaction Frame for Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文属于接触丰富操作任务，解决感知与力反馈融合不足的问题。提出Force Policy，结合全局视觉与局部力控，提升接触稳定性与执行质量。**

- **链接: [https://arxiv.org/pdf/2602.22088](https://arxiv.org/pdf/2602.22088)**

> **作者:** Hongjie Fang; Shirun Tang; Mingyu Mei; Haoxiang Qin; Zihao He; Jingjing Chen; Ying Feng; Chenxi Wang; Wanxi Liu; Zaixing He; Cewu Lu; Shiquan Wang
>
> **备注:** accepted by RSS 2026
>
> **摘要:** Contact-rich manipulation demands human-like integration of perception and force feedback: vision should guide task progress, while high-frequency interaction control must stabilize contact under uncertainty. Existing learning-based policies often entangle these roles in a monolithic network, trading off global generalization against stable local refinement, while control-centric approaches typically assume a known task structure or learn only controller parameters rather than the structure itself. In this paper, we formalize a physically grounded interaction frame, an instantaneous local basis that decouples force regulation from motion execution, and propose a method to recover it from demonstrations. Based on this, we address both issues by proposing Force Policy, a global-local vision-force policy in which a global policy guides free-space actions using vision, and upon contact, a high-frequency local policy with force feedback estimates the interaction frame and executes hybrid force-position control for stable interaction. Real-world experiments across diverse contact-rich tasks show consistent gains over strong baselines, with more robust contact establishment, more accurate force regulation, and reliable generalization to novel objects with varied geometries and physical properties, ultimately improving both contact stability and execution quality. Project page: this https URL
>
---
#### [replaced 009] DexWrist: A Robotic Wrist for Constrained and Dynamic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出DexWrist，一种用于复杂环境的高精度机械腕部，解决传统腕部刚性大、控制难的问题。通过优化设计提升操作灵活性与动态接触稳定性。**

- **链接: [https://arxiv.org/pdf/2507.01008](https://arxiv.org/pdf/2507.01008)**

> **作者:** Martin Peticco; Gabriella Ulloa; John Marangola; Nitish Dashora; Pulkit Agrawal
>
> **备注:** 9 pages, 8 figures. Submitted to RA-L 2026
>
> **摘要:** Development of dexterous manipulation hardware has primarily focused on hands and grippers. However, these end-effectors are often paired with bulky and highly stiff wrists that limit performance in human environments. More designs have adopted backdrivable actuation, but are still difficult to model and control due to coupled kinematics or high mechanical inertia from heavy links. We present DexWrist, a robotic wrist that advances manipulation in highly constrained environments and enables dynamic, contact-rich tasks. We achieve this by combining quasi-direct drive actuation with a decoupled parallel kinematic mechanism in a compact design. It delivers 3.75 +/- 0.05 Nm rated torque, 0.33 +/- 0.06 Nm backdrive torque, 10.15 +/- 1.34 Hz torque bandwidth, +/- 40 degrees ROM in both DOFs, and a one-to-one motor-to-DOF mapping in a 0.97 kg package. In practice, these properties increase workspace in cluttered environments and stabilize contact without the need for finely tuned admittance control. We evaluate DexWrist as a drop-in wrist upgrade in simulation and on two robot arms performing representative constrained and contact-rich tasks. In learned policy evaluations, DexWrist achieved 50-76% relative improvements in success rate, and reduced autonomous task completion times by 3-5x. More details about DexWrist can be found at this https URL.
>
---
#### [replaced 010] Featurized Occupation Measures for Structured Global Search in Numerical Optimal Control
- **分类: math.OC; cs.RO; eess.SY**

- **简介: 该论文属于数值最优控制任务，解决全局搜索与局部优化的矛盾，提出Featurized Occupation Measures (FOM)框架，实现全局与局部方法的耦合。**

- **链接: [https://arxiv.org/pdf/2603.16231](https://arxiv.org/pdf/2603.16231)**

> **作者:** Qi Wei; Jianfeng Tao; Haoyang Tan; Hongyu Nie
>
> **摘要:** Numerical optimal control has long been split between globally structured but dimensionally intractable Hamilton--Jacobi--Bellman (HJB) methods and scalable but local trajectory optimization. We introduce Featurized Occupation Measures (FOM), a finite-dimensional primal--dual interface for coupling numerical optimal control solvers with explicit HJB subsolutions: the certificate guides the primal search, while primal residuals tighten the certificate in a primal-dual language. Two realizations are developed. The explicit realization uses finite weak-form Liouville tests, and the implicit realization couples rollout-based search with sampled primal--dual residuals. Both are proved asymptotically consistent with the exact occupation-measure linear program under refinement, separating primal expressiveness from dual accuracy in the limit. The framework also gives structural conditions under which HJB-type certificates avoid full state-space representation. For factor graphs induced by compatible passivity-based interconnections, blockwise HJB inequalities assemble into globally feasible OM-dual certificates, and the decomposition is preserved under blockwise approximation. The curse of dimensionality is then shifted from state space to interconnection topology. Approximate certificates remain reusable under time shifts and bounded model perturbations, with explicit degradation bounds. On a static obstacle-avoidance benchmark, certificates of increasing tightness guide a sample-based optimizer toward global optima, confirming that even a coarse certificate carries useful global information.
>
---
#### [replaced 011] Q-learning with Adjoint Matching
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文提出QAM算法，解决连续动作强化学习中高效优化表达性策略的问题。通过邻接匹配技术，避免数值不稳定，提升策略性能。**

- **链接: [https://arxiv.org/pdf/2601.14234](https://arxiv.org/pdf/2601.14234)**

> **作者:** Qiyang Li; Sergey Levine
>
> **备注:** 32 pages, 8 figures, 7 tables
>
> **摘要:** We propose Q-learning with Adjoint Matching (QAM), a novel TD-based reinforcement learning (RL) algorithm that tackles a long-standing challenge in continuous-action RL: efficient optimization of an expressive diffusion or flow-matching policy with respect to a parameterized Q-function. Effective optimization requires exploiting the first-order information of the critic, but it is challenging to do so for flow or diffusion policies because direct gradient-based optimization via backpropagation through their multi-step denoising process is numerically unstable. Existing methods work around this either by only using the value and discarding the gradient information, or by relying on approximations that sacrifice policy expressivity or bias the learned policy. QAM sidesteps both of these challenges by leveraging adjoint matching, a recently proposed technique in generative modeling, which transforms the critic's action gradient to form a step-wise objective function that is free from unstable backpropagation, while providing an unbiased, expressive policy at the optimum. Combined with temporal-difference backup for critic learning, QAM consistently outperforms prior approaches on hard, sparse reward tasks in both offline and offline-to-online RL.
>
---
#### [replaced 012] When a Robot is More Capable than a Human: Learning from Constrained Demonstrators
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人学习任务，解决受限专家示范导致策略次优的问题。通过推断状态奖励并探索更优轨迹，提升政策性能。**

- **链接: [https://arxiv.org/pdf/2510.09096](https://arxiv.org/pdf/2510.09096)**

> **作者:** Xinhu Li; Ayush Jain; Zhaojing Yang; Yigit Korkmaz; Erdem Bıyık
>
> **摘要:** Learning from demonstrations enables experts to teach robots complex tasks using interfaces such as kinesthetic teaching, joystick control, and sim-to-real transfer. However, these interfaces often constrain the expert's ability to demonstrate optimal behavior due to indirect control, setup restrictions, and hardware safety. For example, a joystick can move a robotic arm only in a 2D plane, even though the robot operates in a higher-dimensional space. As a result, the demonstrations collected by constrained experts lead to suboptimal performance of the learned policies. This raises a key question: Can a robot learn a better policy than the one demonstrated by a constrained expert? We address this by allowing the agent to go beyond direct imitation of expert actions and explore shorter and more efficient trajectories. We use the demonstrations to infer a state-only reward signal that measures task progress, and self-label reward for unknown states using temporal interpolation. Our approach outperforms common imitation learning in both sample efficiency and task completion time. On a real WidowX robotic arm, it completes the task in 12 seconds, 10x faster than behavioral cloning, as shown in real-robot videos on this https URL .
>
---
#### [replaced 013] AlignDrive: Aligned Lateral-Longitudinal Planning for End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决纵向与横向决策耦合不足的问题。提出AlignDrive框架，通过路径条件化预测提升协调与安全性。**

- **链接: [https://arxiv.org/pdf/2601.01762](https://arxiv.org/pdf/2601.01762)**

> **作者:** Yanhao Wu; Haoyang Zhang; Fei He; Rui Wu; Yanhu Shan; Congpei Qiu; Liang Gao; Wei Ke; Tong Zhang
>
> **备注:** underreview
>
> **摘要:** Practical autonomous driving requires models that generalize by reasoning through spatial-temporal possibilities to exclude unsafe outcomes. While state-of-the-art (SOTA) methods use parallel planning architectures, they fail to explicitly couple speed decisions with agent behavior along the driving path, leading to suboptimal coordination. To address this, we propose a cascaded framework that transforms longitudinal planning from an independent prediction task into a path-conditioned reasoning process. On the model side, we introduce an anchor-based regression design that conditions longitudinal prediction on the lateral drive path, and reformulate longitudinal planning as 1D displacement prediction along the path. This reduces geometric uncertainty and sharpens the model's focus on interaction-driven dynamics. On the data side, we introduce a planning-oriented data augmentation strategy that simulates rare safety-critical events by programmatically inserting agents and relabeling longitudinal targets to enforce collision avoidance. Evaluated on the challenging Bench2Drive benchmark, our method achieves SOTA performance with a driving score of 89.07 and a success rate of 73.18%, demonstrating significantly improved coordination and safety. Further evaluation on Fail2Drive confirms strong generalization to rare edge cases where parallel formulations typically fail. Project page:this https URL.
>
---
#### [replaced 014] Continuum Robot Localization using Distributed Time-of-Flight Sensors
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，解决软体连续机器人在非结构化环境中的定位问题。通过分布式低分辨率ToF传感器和形状先验信息融合，实现高精度定位。**

- **链接: [https://arxiv.org/pdf/2602.07209](https://arxiv.org/pdf/2602.07209)**

> **作者:** Spencer Teetaert; Giammarco Caroleo; Marco Pontin; Sven Lilge; Jessica Burgner-Kahrs; Timothy D. Barfoot; Perla Maiolino
>
> **备注:** Print version, to be published at Robotics: Science and Systems (RSS) 2026
>
> **摘要:** Localization and mapping of an environment are crucial tasks for any robot operating in unstructured environments. Time-of-flight (ToF) sensors (e.g.,~lidar) have proven useful in mobile robotics, where high-resolution sensors can be used for simultaneous localization and mapping. In soft and continuum robotics, however, these high-resolution sensors are too large for practical use. This, combined with the deformable nature of such robots, has resulted in continuum robot (CR) localization and mapping in unstructured environments being a largely untouched area. In this work, we present a localization technique for CRs that relies on small, low-resolution ToF sensors distributed along the length of the robot. By fusing measurement information with a robot shape prior, we show that accurate localization is possible despite each sensor experiencing frequent degenerate scenarios. We achieve an average localization error of 2.5cm in position and 7.2° in rotation across all experimental conditions with a 53cm long robot. We demonstrate that the results are repeated across multiple environments, in both simulation and real-world experiments, and study robustness in the estimation to deviations in the prior map.
>
---
#### [replaced 015] A Nonasymptotic Theory of Gain-Dependent Error Dynamics in Behavior Cloning
- **分类: cs.RO; cs.AI; math.OC**

- **简介: 该论文研究行为克隆中的增益依赖误差动态问题，分析控制器增益对任务失败的影响，提出非渐近理论框架以解释误差传播机制。**

- **链接: [https://arxiv.org/pdf/2604.14484](https://arxiv.org/pdf/2604.14484)**

> **作者:** Junghoon Seo
>
> **摘要:** Behavior cloning (BC) policies on position-controlled robots inherit the closed-loop response of the underlying PD controller, yet the nonasymptotic finite-horizon consequences of controller gains for BC failure remain open. We show that independent sub-Gaussian action errors propagate through the gain-dependent closed-loop dynamics to yield sub-Gaussian position errors whose proxy matrix $X_\infty(K)$ governs the failure tail. The probability of horizon-$T$ task failure factorizes into a gain-dependent amplification index $\Gamma_T(K)$ and the validation loss plus a generalization slack, so training loss alone cannot predict closed-loop performance. Under shape-preserving upper-bound structural assumptions, the proxy admits the scalar bound $X_\infty(K)\preceq\Psi(K)\bar X$, with $\Psi(K)$ decomposed into label difficulty, injection strength, and contraction. This ranks the four canonical regimes with compliant-overdamped (CO) tightest, stiff-underdamped (SU) loosest, and the stiff-overdamped versus compliant-underdamped ordering system-dependent. For the canonical scalar second-order PD system, the closed-form continuous-time stationary variance $X_\infty^{\mathrm{c}}(\alpha,\beta)=\sigma^2\alpha/(2\beta)$ is strictly monotone in stiffness and damping over the entire stable orthant, covering both underdamped and overdamped regimes, and the exact zero-order-hold (ZOH) discretization inherits this monotonicity. The analysis gives a nonasymptotic finite-horizon extension of the gain-dependent error-attenuation explanation of Bronars et al.
>
---
#### [replaced 016] Now You See That: Learning End-to-End Humanoid Locomotion from Raw Pixels
- **分类: cs.RO**

- **简介: 该论文属于视觉驱动的人形机器人行走任务，解决sim-to-real感知噪声和地形适应问题。提出高保真深度模拟和视觉感知行为蒸馏，结合地形奖励 shaping 实现鲁棒运动控制。**

- **链接: [https://arxiv.org/pdf/2602.06382](https://arxiv.org/pdf/2602.06382)**

> **作者:** Wandong Sun; Yongbo Su; Leoric Huang; Alex Zhang; Dwyane Wei; Mu San; Daniel Tian; Ellie Cao; Baoshi Cao; Yang Liu; Finn Yan; Ethan Xie; Zongwu Xie
>
> **摘要:** Achieving robust vision-based humanoid locomotion remains challenging due to two fundamental issues: the sim-to-real gap introduces significant perception noise that degrades performance on fine-grained tasks, and training a unified policy across diverse terrains is hindered by conflicting learning objectives. To address these challenges, we present an end-to-end framework for vision-driven humanoid locomotion. For robust sim-to-real transfer, we develop a high-fidelity depth sensor simulation that captures stereo matching artifacts and calibration uncertainties inherent in real-world sensing. We further propose a vision-aware behavior distillation approach that combines latent space alignment with noise-invariant auxiliary tasks, enabling effective knowledge transfer from privileged height maps to noisy depth observations. For versatile terrain adaptation, we introduce terrain-specific reward shaping integrated with multi-critic and multi-discriminator learning, where dedicated networks capture the distinct dynamics and motion priors of each terrain type. We validate our approach on two humanoid platforms equipped with different stereo depth cameras. The resulting policy demonstrates robust performance across diverse environments, seamlessly handling extreme challenges such as high platforms and wide gaps, as well as fine-grained tasks including bidirectional long-term staircase traversal.
>
---
#### [replaced 017] Safe and Real-Time Consistent Planning for Autonomous Vehicles in Partially Observed Environments via Parallel Consensus Optimization
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自动驾驶任务，解决部分观测环境下车辆的安全与一致性规划问题。提出CPTO方法，通过并行优化确保轨迹安全与一致。**

- **链接: [https://arxiv.org/pdf/2409.10310](https://arxiv.org/pdf/2409.10310)**

> **作者:** Lei Zheng; Rui Yang; Minzhe Zheng; Michael Yu Wang; Jun Ma
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** Ensuring safety and driving consistency is a significant challenge for autonomous vehicles operating in partially observed environments. This work introduces a consistent parallel trajectory optimization (CPTO) approach to enable safe and consistent driving in dense obstacle environments with perception uncertainties. Utilizing discrete-time barrier function theory, we develop a consensus safety barrier module that ensures reliable safety coverage within the spatiotemporal trajectory space across potential obstacle configurations. Following this, a bi-convex parallel trajectory optimization problem is derived that facilitates decomposition into a series of low-dimensional quadratic programming problems to accelerate computation. By leveraging the consensus alternating direction method of multipliers (ADMM) for parallel optimization, each generated candidate trajectory corresponds to a possible environment configuration while sharing a common consensus trajectory segment. This ensures driving safety and consistency when executing the consensus trajectory segment for the ego vehicle in real time. We validate our CPTO framework through extensive comparisons with state-of-the-art baselines across multiple driving tasks in partially observable environments. Our results demonstrate improved safety and consistency using both synthetic and real-world traffic datasets.
>
---
#### [replaced 018] VP-VLA: Visual Prompting as an Interface for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出VP-VLA，属于视觉-语言-动作模型任务，解决传统模型在空间精度和泛化性上的不足，通过双系统框架分离高阶推理与低阶执行。**

- **链接: [https://arxiv.org/pdf/2603.22003](https://arxiv.org/pdf/2603.22003)**

> **作者:** Zixuan Wang; Yuxin Chen; Yuqi Liu; Jinhui Ye; Pengguang Chen; Changsheng Lu; Shu Liu; Bei Yu; Jiaya Jia
>
> **备注:** Project page: this https URL
>
> **摘要:** Vision-Language-Action (VLA) models typically map visual observations and linguistic instructions directly to control signals. This "black-box" mapping forces a single forward pass to simultaneously handle instruction interpretation, spatial grounding, and low-level control, often leading to poor spatial precision and limited robustness in out-of-distribution scenarios. To address these limitations, we propose VP-VLA, a dual-system framework that decouples high-level reasoning and low-level execution via a structured visual prompting interface. Specifically, a "System 2 Planner" decomposes complex instructions into sub-tasks and identifies relevant target objects and goal locations. These spatial anchors are rendered directly within the native RGB observation space as modality-consistent visual prompts, such as crosshairs and bounding boxes. This avoids the modality mismatch introduced by dense masks, affordance maps, or additional control-specific representations. Guided by these prompts and enhanced by a novel auxiliary visual grounding objective during training, a "System 1 Controller" reliably generates precise low-level execution motions. Extensive experiments in simulation and real world demonstrate that VP-VLA surpasses state-of-the-art end-to-end baselines including QwenOFT and GR00T-N1.6. Project page: this https URL
>
---
#### [replaced 019] Hydra-DP3: Frequency-Aware Right-Sizing of 3D Diffusion Policies for Visuomotor Control
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉控制任务，解决3D扩散策略的效率问题。通过频率分析，提出轻量级扩散模型HDP3，实现高效动作生成。**

- **链接: [https://arxiv.org/pdf/2605.01581](https://arxiv.org/pdf/2605.01581)**

> **作者:** Jinhao Zhang; Zhexuan Zhou; Huizhe Li; Yichen Lai; Wenlong Xia; Haoming Song; Youmin Gong; Jie Mei
>
> **摘要:** Diffusion-based visuomotor policies perform well in robotic manipulation, yet current methods still inherit image-generation-style decoders and multi-step sampling. We revisit this design from a frequency-domain perspective. Robot action trajectories are highly smooth, with most energy concentrated in a few low-frequency discrete cosine transform modes. Under this structure, we show that the error of the optimal denoiser is bounded by the low-frequency subspace dimension and residual high-frequency energy, implying that denoising error saturates after very few reverse steps. This also suggests that action denoising requires a much simpler denoising model than image generation. Motivated by this insight, we propose Hydra-DP3 (HDP3), a pocket-scale 3D diffusion policy with a lightweight Diffusion Mixer decoder that supports two-step DDIM inference. Our synthetic experiments validate the theory and support the sufficiency of two-step denoising. Futhermore, across RoboTwin2.0, Adroit, MetaWorld, and real-world tasks, HDP3 achieves state-of-the-art performance with fewer than 1% of the parameters of prior 3D diffusion-based policies and substantially lower inference latency.
>
---
#### [replaced 020] MapNav: A Novel Memory Representation via Annotated Semantic Maps for Vision-and-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决传统方法依赖历史观测导致的存储与计算开销问题。提出MapNav模型，利用带语义标注的地图替代历史帧，提升导航效率与准确性。**

- **链接: [https://arxiv.org/pdf/2502.13451](https://arxiv.org/pdf/2502.13451)**

> **作者:** Lingfeng Zhang; Xiaoshuai Hao; Qinwen Xu; Qiang Zhang; Xinyao Zhang; Pengwei Wang; Jing Zhang; Zhongyuan Wang; Shanghang Zhang; Renjing Xu
>
> **摘要:** Vision-and-language navigation (VLN) is a key task in Embodied AI, requiring agents to navigate diverse and unseen environments while following natural language instructions. Traditional approaches rely heavily on historical observations as spatio-temporal contexts for decision making, leading to significant storage and computational overhead. In this paper, we introduce MapNav, a novel end-to-end VLN model that leverages Annotated Semantic Map (ASM) to replace historical frames. Specifically, our approach constructs a top-down semantic map at the start of each episode and update it at each timestep, allowing for precise object mapping and structured navigation information. Then, we enhance this map with explicit textual labels for key regions, transforming abstract semantics into clear navigation cues and generate our ASM. MapNav agent using the constructed ASM as input, and use the powerful end-to-end capabilities of VLM to empower VLN. Extensive experiments demonstrate that MapNav achieves state-of-the-art (SOTA) performance in both simulated and real-world environments, validating the effectiveness of our method. Moreover, we will release our ASM generation source code and dataset to ensure reproducibility, contributing valuable resources to the field. We believe that our proposed MapNav can be used as a new memory representation method in VLN, paving the way for future research in this field.
>
---
#### [replaced 021] Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文研究 embodied LLMs 的任务规划问题，旨在提升机器人在部署中的反思与学习能力。通过引入反射式测试时规划，增强错误纠正与经验积累，提升长期任务表现。**

- **链接: [https://arxiv.org/pdf/2602.21198](https://arxiv.org/pdf/2602.21198)**

> **作者:** Yining Hong; Huang Huang; Manling Li; Li Fei-Fei; Leonidas Guibas; Jiajun Wu; Yejin Choi
>
> **摘要:** Embodied LLMs endow robots with high-level task reasoning, but they cannot reflect on what went wrong or why, turning deployment into a sequence of independent trials where mistakes repeat rather than accumulate into experience. Drawing upon human reflective practitioners, we introduce Reflective Test-Time Planning, which integrates two modes of reflection: \textit{reflection-in-action}, where the agent uses test-time scaling to generate and score multiple candidate actions using internal reflections before execution; and \textit{reflection-on-action}, which uses test-time training to update both its internal reflection model and its action policy based on external reflections after execution. We also include retrospective reflection, allowing the agent to re-evaluate earlier decisions and perform model updates with hindsight for proper long-horizon credit assignment. Experiments on our newly-designed Long-Horizon Household benchmark and MuJoCo Cupboard Fitting benchmark show significant gains over baseline models, with zero-shot generalization to photorealistic HM3D environments and real-robot experiments on a Franka Panda arm. Ablations confirm that reflection-in-action and reflection-on-action are mutually dependent, and that retrospective reflection achieves better credit assignment than step-wise external feedback at lower computational overhead. Qualitative analyses further highlight behavioral correction through reflection.
>
---
#### [replaced 022] Supervised Mixture-of-Experts for Surgical Grasping and Retraction
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于手术机器人操作任务，解决数据稀缺和高安全要求下的机械臂抓取与牵开问题。提出监督混合专家架构，提升ACT模型在有限演示下的性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.21971](https://arxiv.org/pdf/2601.21971)**

> **作者:** Lorenzo Mazza; Ariel Rodriguez; Rayan Younis; Martin Lelis; Ortrun Hellig; Chenpan Li; Sebastian Bodenstedt; Martin Wagner; Stefanie Speidel
>
> **备注:** Accepted at Robotics:Science and Systems 2026
>
> **摘要:** Imitation learning has achieved remarkable success in robotic manipulation, yet its application to surgical robotics remains challenging due to data scarcity, constrained workspaces, and the need for an exceptional level of safety and predictability. We present a supervised Mixture-of-Experts (MoE) architecture designed for phase-structured surgical manipulation tasks, which can be added on top of any autonomous policy. Unlike prior surgical robot learning approaches that rely on multi-camera setups or thousands of demonstrations, we show that a lightweight action decoder policy like Action Chunking Transformer (ACT) can learn complex, long-horizon manipulation from less than 150 demonstrations using solely stereo endoscopic images, when equipped with our architecture. We evaluate our approach on the collaborative surgical task of bowel grasping and retraction, where a robot assistant interprets visual cues from a human surgeon, executes targeted grasping on deformable tissue, and performs sustained retraction. Our results show that generalist Vision Language Action models fail to acquire the task entirely, even under standard in-distribution conditions. Furthermore, while standard ACT achieves moderate success in-distribution, adopting a supervised MoE architecture significantly boosts its performance, yielding higher success rates in-distribution and demonstrating superior robustness in out-of-distribution scenarios, including novel grasp locations, reduced illumination, and partial occlusions. Notably, it generalizes to unseen testing viewpoints and also transfers zero-shot to ex vivo porcine tissue without additional training, offering a promising pathway toward in vivo deployment. To support this statement, we present qualitative preliminary results of policy roll-outs during in vivo porcine surgery.
>
---
#### [replaced 023] Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion
- **分类: cs.RO**

- **简介: 该论文属于四足机器人运动控制任务，解决sim-to-real迁移性和奖励过拟合问题。提出MoE策略与RoboGauge评估框架，提升复杂地形下的运动鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.00678](https://arxiv.org/pdf/2602.00678)**

> **作者:** Tianyang Wu; Hanwei Guo; Yuhang Wang; Junshu Yang; Xinyang Sui; Jiayi Xie; Xingyu Chen; Zeyang Liu; Xuguang Lan
>
> **备注:** Accepted at Robotics Science and Systems (RSS), 2026. Project Page: this https URL
>
> **摘要:** Reinforcement learning has shown strong promise for quadrupedal agile locomotion, even with proprioception-only sensing. In practice, however, sim-to-real gap and reward overfitting in complex terrains can produce policies that fail to transfer, while physical validation remains risky and inefficient. To address these challenges, we introduce a unified framework encompassing a Mixture-of-Experts (MoE) locomotion policy for robust multi-terrain representation with RoboGauge, a predictive assessment suite that quantifies sim-to-real transferability. The MoE policy employs a gated set of specialist experts to decompose latent terrain and command modeling, achieving superior deployment robustness and generalization via proprioception alone. RoboGauge further provides multi-dimensional proprioception-based metrics via sim-to-sim tests over terrains, difficulty levels, and domain randomizations, enabling reliable MoE policy selection without extensive physical trials. Experiments on a Unitree Go2 demonstrate robust locomotion on unseen challenging terrains, including snow, sand, stairs, slopes, and 30 cm obstacles. In dedicated high-speed tests, the robot reaches 4 m/s and exhibits an emergent narrow-width gait associated with improved stability at high velocity.
>
---
#### [replaced 024] Equivariant Volumetric Grasping
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机械臂抓取任务，旨在提升抓取采样效率。通过引入旋转等变的三平面特征表示，设计了更高效的抓取模型。**

- **链接: [https://arxiv.org/pdf/2507.18847](https://arxiv.org/pdf/2507.18847)**

> **作者:** Pinhao Song; Yutong Hu; Pengteng Li; Renaud Detry
>
> **备注:** 21 pages
>
> **摘要:** We propose a new volumetric grasp model that is equivariant to rotations around the vertical axis, leading to a significant improvement in sampling efficiency. Our model employs a tri-plane volumetric feature representation -- i.e., the projection of 3D features onto three canonical planes. We introduce a novel tri-plane feature design in which features on the horizontal plane are \emph{equivariant} to $90^\circ$ rotations, while the \emph{sum} of features from the other two planes remains \emph{invariant} to reflections induced by the same transformations. We further develop equivariant adaptations of two state-of-the-art volumetric grasp planners, GIGA and IGD. Specifically, we derive a new equivariant formulation of IGD's deformable attention mechanism and propose an equivariant generative model of grasp orientations based on flow matching. We provide a detailed analytical justification of the proposed equivariance properties and validate our approach through extensive simulated and real-world experiments. Our results demonstrate that the proposed projection-based design reduces both computational and memory costs. Moreover, the equivariant grasp models built on top of our tri-plane features consistently outperform their non-equivariant counterparts, achieving higher performance within a real-time cost constraint. Video and code can be viewed in: this https URL
>
---
#### [replaced 025] Recovering Hidden Reward in Diffusion-Based Policies
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决从专家演示中恢复隐藏奖励的问题。通过能量场框架，结合生成动作建模与逆强化学习，实现无需对抗训练的奖励提取与政策泛化。**

- **链接: [https://arxiv.org/pdf/2605.00623](https://arxiv.org/pdf/2605.00623)**

> **作者:** Yanbiao Ji; Qiuchang Li; Yuting Hu; Shaokai Wu; Wenyuan Xie; Guodong Zhang; Qicheng He; Deyi Ji; Yue Ding; Hongtao Lu
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** This paper introduces EnergyFlow, a framework that unifies generative action modeling with inverse reinforcement learning by parameterizing a scalar energy function whose gradient is the denoising field. We establish that under maximum-entropy optimality, the score function learned via denoising score matching recovers the gradient of the expert's soft Q-function, enabling reward extraction without adversarial training. Formally, we prove that constraining the learned field to be conservative reduces hypothesis complexity and tightens out-of-distribution generalization bounds. We further characterize the identifiability of recovered rewards and bound how score estimation errors propagate to action preferences. Empirically, EnergyFlow achieves state-of-the-art imitation performance on various manipulation tasks while providing an effective reward signal for downstream reinforcement learning that outperforms both adversarial IRL methods and likelihood-based alternatives. These results show that the structural constraints required for valid reward extraction simultaneously serve as beneficial inductive biases for policy generalization. The code is available at this https URL.
>
---
#### [replaced 026] UniUncer: Unified Dynamic Static Uncertainty for End to End Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出UniUncer，解决端到端驾驶中的不确定性问题，通过统一建模静态与动态元素的不确定性，提升规划可靠性。**

- **链接: [https://arxiv.org/pdf/2603.07686](https://arxiv.org/pdf/2603.07686)**

> **作者:** Yu Gao; Jijun Wang; Zongzheng Zhang; Anqing Jiang; Yiru Wang; Yuwen Heng; Shuo Wang; Hao Sun; Zhangfeng Hu; Hao Zhao
>
> **备注:** Accepted ICRA 2026
>
> **摘要:** End-to-end (E2E) driving has become a cornerstone of both industry deployment and academic research, offering a single learnable pipeline that maps multi-sensor inputs to actions while avoiding hand-engineered modules. However, the reliability of such pipelines strongly depends on how well they handle uncertainty: sensors are noisy, semantics can be ambiguous, and interaction with other road users is inherently stochastic. Uncertainty also appears in multiple forms: classification vs. localization, and, crucially, in both static map elements and dynamic agents. Existing E2E approaches model only static-map uncertainty, leaving planning vulnerable to overconfident and unreliable inputs. We present UniUncer, the first lightweight, unified uncertainty framework that jointly estimates and uses uncertainty for both static and dynamic scene elements inside an E2E planner. Concretely: (1) we convert deterministic heads to probabilistic Laplace regressors that output per-vertex location and scale for vectorized static and dynamic entities; (2) we introduce an uncertainty-fusion module that encodes these parameters and injects them into object/map queries to form uncertainty-aware queries; and (3) we design an uncertainty-aware gate that adaptively modulates reliance on historical inputs (ego status or temporal perception queries) based on current uncertainty levels. The design adds minimal overhead and drops throughput by only $\sim$0.5 FPS while remaining plug-and-play for common E2E backbones. On nuScenes (open-loop), UniUncer reduces average L2 trajectory error by 7\%. On NavsimV2 (pseudo closed-loop), it improves overall EPDMS by 10.8\% and notable stage two gains in challenging, interaction-heavy scenes. Ablations confirm that dynamic-agent uncertainty and the uncertainty-aware gate are both necessary.
>
---
#### [replaced 027] When to Trust Imagination: Adaptive Action Execution for World Action Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，解决WAM执行中预测与现实不一致的问题。提出FFDC验证器和Mixture-of-Horizon训练，实现自适应动作执行，提升效率与成功率。**

- **链接: [https://arxiv.org/pdf/2605.06222](https://arxiv.org/pdf/2605.06222)**

> **作者:** Rui Wang; Yue Zhang; Jiehong Lin; Kuncheng Luo; Jianan Wang; Zhongrui Wang; Xiaojuan Qi
>
> **摘要:** World Action Models (WAMs) have recently emerged as a promising paradigm for robotic manipulation by jointly predicting future visual observations and future actions. However, current WAMs typically execute a fixed number of predicted actions after each model inference, leaving the robot blind to whether the imagined future remains consistent with the actual physical rollout. In this work, we formulate adaptive WAM execution as a future-reality verification problem: the robot should execute longer when the WAM-predicted future remains reliable, and replan earlier when reality deviates from imagination. To this end, we propose Future Forward Dynamics Causal Attention (FFDC), a lightweight verifier that jointly reasons over predicted future actions, predicted visual dynamics, real observations, and language instructions to estimate whether the remaining action rollout can still be trusted. FFDC enables adaptive action chunk sizes as an emergent consequence of prediction-observation consistency, preserving the efficiency of long-horizon execution while restoring responsiveness in contact-rich or difficult phases. We further introduce Mixture-of-Horizon Training to improve long-horizon trajectory coverage for adaptive execution. Experiments on the RoboTwin benchmark and in the real world demonstrate that our method achieves a strong robustness-efficiency trade-off: on RoboTwin, it reduces WAM forward passes by 69.10% and execution time by 34.02%, while improving success rate by 2.54% over the short-chunk baseline; in real-world experiments, it improves success rate by 35%.
>
---
#### [replaced 028] Commanding Humanoid by Free-form Language: A Large Language Action Model with Unified Motion Vocabulary
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机交互任务，旨在让类人机器人根据自然语言指令执行复杂动作。解决语言指令与运动控制匹配及物理稳定性问题，提出Humanoid-LLA模型实现高效、多样且真实的运动生成。**

- **链接: [https://arxiv.org/pdf/2511.22963](https://arxiv.org/pdf/2511.22963)**

> **作者:** Zhirui Liu; Kaiyang Ji; Ke Yang; Yahao Fan; Jingyi Yu; Ye Shi; Jingya Wang
>
> **备注:** Project page: this https URL
>
> **摘要:** Enabling humanoid robots to follow free-form natural language commands is a critical step toward seamless human-robot interaction and general-purpose embodied AI. However, existing methods remain limited, often constrained to simple instructions or forced to sacrifice motion diversity for physical plausibility. To address this gap, we present Humanoid-LLA, a Large Language Action model that translates unconstrained natural language directly into executable whole-body motions for humanoid robots. Our approach tackles two core challenges: paired language-humanoid motion data scarcity and physical instability. First, we bridge high-level language semantics with physically-grounded control by learning a unified human-humanoid motion vocabulary. Second, we introduce a novel two-stage fine-tuning framework that begins with supervised motion Chain-of-Thought learning, followed by reinforcement learning refined with physical feedback to ensure robustness and stability. Extensive evaluation in simulation and real-world cross-embodiment experiments demonstrates that Humanoid-LLA achieves superior generalization to novel language commands and diverse motion generation while maintaining high physical fidelity.
>
---
#### [replaced 029] SCORP: Scene-Consistent Multi-agent Diffusion Planning with Stable Online Reinforcement Post-Training for Cooperative Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对协同驾驶任务，解决多智能体轨迹一致性与闭环优化难题。提出SCORP模型，结合场景一致的扩散规划与稳定在线强化学习后训练，提升安全与效率。**

- **链接: [https://arxiv.org/pdf/2604.11734](https://arxiv.org/pdf/2604.11734)**

> **作者:** Haojie Bai; Aimin Li; Ruoyu Yao; Xiongwei Zhao; Tingting Zhang; Xing Zhang; Lin Gao; and Jun Ma
>
> **摘要:** Cooperative driving is a safety- and efficiency-critical task that requires the coordination of diverse, interaction-realistic multi-agent trajectories. Although existing diffusion-based methods can capture multimodal behaviors from demonstrations, they often exhibit weak scene consistency and poor alignment with closed-loop cooperative objectives. This makes post-training necessary for further improvement, yet achieving stable online post-training in reactive multi-agent environments remains challenging. In this paper, we propose SCORP, a scene-consistent multi-agent diffusion planner with stable online reinforcement learning (RL) post-training for cooperative driving. For pre-training, we develop a scene-conditioned multi-agent denoising architecture that couples inter-agent self-attention with a dual-path conditioning mechanism: cross-attention provides direct scene-information injection, while AdaLN-Zero enables additional flexible and stable conditional modulation, thereby improving the scene consistency and road adherence of joint trajectories. For post-training, we formulate a two-layer Markov decision process (MDP) that explicitly integrates the reverse denoising chain with policy-environment interaction. We further co-design dense, well-shaped planning rewards and variance-gated group-relative policy optimization (VG-GRPO) to mitigate advantage collapse and gradient instability during closed-loop training. Extensive experiments show that SCORP outperforms strong open-source baselines on WOMD, with 10.47%-28.26% and 1.70%-7.22% improvements in core safety and efficiency metrics, respectively. Moreover, compared with alternative post-training methods, SCORP delivers significant and consistent gains in both driving safety and traffic efficiency, highlighting stable and sustained advances in closed-loop cooperative driving.
>
---
#### [replaced 030] Learning Agile Striker Skills for Humanoid Soccer Robots from Noisy Sensory Input
- **分类: cs.RO**

- **简介: 该论文属于机器人足球任务，旨在解决 humanoid 机器人在噪声感知下学习稳定踢球技能的问题。通过强化学习框架，提升机器人踢球的准确性与适应性。**

- **链接: [https://arxiv.org/pdf/2512.06571](https://arxiv.org/pdf/2512.06571)**

> **作者:** Zifan Xu; Myoungkyu Seo; Dongmyeong Lee; Hao Fu; Jiaheng Hu; Jiaxun Cui; Yuqian Jiang; Zhihan Wang; Anastasiia Brund; Joydeep Biswas; Peter Stone
>
> **摘要:** Learning fast and robust ball-kicking skills is a critical capability for humanoid soccer robots, yet it remains a challenging problem due to the need for rapid leg swings, postural stability on a single support foot, and robustness under noisy sensory input and external perturbations (e.g., opponents). This paper presents a reinforcement learning (RL)-based system that enables humanoid robots to execute robust continual ball-kicking with adaptability to different ball-goal configurations. The system extends a typical teacher-student training framework -- in which a "teacher" policy is trained with ground truth state information and the "student" learns to mimic it with noisy, imperfect sensing -- by including four training stages: (1) long-distance ball chasing (teacher); (2) directional kicking (teacher); (3) teacher policy distillation (student); and (4) student adaptation and refinement (student). Key design elements -- including tailored reward functions, realistic noise modeling, and online constrained RL for adaptation and refinement -- are critical for closing the sim-to-real gap and sustaining performance under perceptual uncertainty. Extensive evaluations in both simulation and on a real robot demonstrate strong kicking accuracy and goal-scoring success across diverse ball-goal configurations. Ablation studies further highlight the necessity of the constrained RL, noise modeling, and the adaptation stage. This work presents a system for learning robust continual humanoid ball-kicking under imperfect perception, establishing a benchmark task for visuomotor skill learning in humanoid whole-body control.
>
---
#### [replaced 031] AffordSim: A Scalable Data Generator and Benchmark for Affordance-Aware Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出AffordSim，用于生成和基准测试感知 affordance 的机器人操作数据。解决传统方法在任务语义和泛化性上的不足，通过集成3D affordance预测提升模拟轨迹生成效果。**

- **链接: [https://arxiv.org/pdf/2604.11674](https://arxiv.org/pdf/2604.11674)**

> **作者:** Mingyang Li; Haofan Xu; Haowen Sun; Xinzhe Chen; Sihua Ren; Liqi Huang; Xinyang Sui; Chenyang Miao; Jiawei Ye; Qiongjie Cui; Zeyang Liu; Xingyu Chen; Xuguang Lan
>
> **摘要:** Many everyday robot manipulation skills are affordance-dependent, with success determined by whether the robot contacts the functional object region required by the subsequent action. Current simulation data generators obtain contacts from generic grasp estimators or per-object manual contact annotations, but generic estimators rank stable grasps without task semantics and often select contacts that are misaligned with the downstream action, while manual contact annotations must be rewritten for each new object and task. To solve these challenges, we introduce AffordSim, a scalable data generator and benchmark that integrates open-vocabulary 3D affordance prediction into simulation-based trajectory generation. Given a natural-language task description, AffordSim synthesizes a task-relevant scene, emits affordance queries, grounds them on object surfaces, samples region-conditioned grasps, and selects executable candidates with motion planning. It further randomizes object pose, texture, lighting, image noise, and cross-viewpoint backgrounds for sim-to-real transfer. We instantiate AffordSim as a 50-task benchmark across diverse manipulation skills, five robot embodiments, and 500+ rigid and articulated objects. AffordSim achieves 93% of the trajectory collection success rate of manual contact annotations on affordance-critical tasks and 89% on hard composite tasks. Vision-language-action policies trained on AffordSim data transfer zero-shot to a real Franka FR3, reaching 24% average success.
>
---
#### [replaced 032] Tempered Sequential Monte Carlo for Trajectory and Policy Optimization with Differentiable Dynamics
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出一种基于采样的轨迹与策略优化框架，解决不同微分动力系统下的控制问题。通过将控制器设计转化为推理任务，利用TSMC方法高效采样最优策略。**

- **链接: [https://arxiv.org/pdf/2604.21456](https://arxiv.org/pdf/2604.21456)**

> **作者:** Heng Yang
>
> **备注:** Robotics: Science and Systems 2026
>
> **摘要:** We propose a sampling-based framework for finite-horizon trajectory and policy optimization under differentiable dynamics by casting controller design as inference. Specifically, we minimize a KL-regularized expected trajectory cost, which yields an optimal "Boltzmann-tilted" distribution over controller parameters that concentrates on low-cost solutions as temperature decreases. To sample efficiently from this sharp, potentially multimodal target, we introduce tempered sequential Monte Carlo (TSMC): an annealing scheme that adaptively reweights and resamples particles along a tempering path from a prior to the target distribution, while using Hamiltonian Monte Carlo rejuvenation to maintain diversity and exploit exact gradients obtained by differentiating through trajectory rollouts. For policy optimization, we extend TSMC via (i) a deterministic empirical approximation of the initial-state distribution and (ii) an extended-space construction that treats rollout randomness as auxiliary variables. Experiments across trajectory- and policy-optimization benchmarks show that TSMC is broadly applicable and compares favorably to state-of-the-art baselines.
>
---
#### [replaced 033] Information Filtering via Variational Regularization for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决视觉-运动策略中中间特征的噪声问题。通过引入变分正则化模块，提升模型性能并取得新基准结果。**

- **链接: [https://arxiv.org/pdf/2601.21926](https://arxiv.org/pdf/2601.21926)**

> **作者:** Jinhao Zhang; Wenlong Xia; Yaojia Wang; Zhexuan Zhou; Huizhe Li; Yichen Lai; Haoming Song; Youmin Gong; Jie Mei
>
> **摘要:** Diffusion-based visuomotor policies built on 3D visual representations have achieved strong performance in learning complex robotic skills. However, most existing methods employ an oversized denoising decoder. While increasing model capacity can improve denoising, empirical evidence suggests that it also introduces redundancy and noise in intermediate feature blocks. Crucially, we find that randomly masking backbone features in U-Net or skipping intermediate layers in DiT at inference time (without changing training) can improve performance, confirming the presence of task-irrelevant noise in intermediate features. To this end, we propose Variational Regularization (VR), a plug-and-play module that imposes a context-conditioned Gaussian over the noisy features and applies a KL-divergence regularizer, forming an adaptive information bottleneck. Extensive experiments on three simulation benchmarks, RoboTwin2.0, Adroit, and MetaWorld, show that our approach consistently improves task success rates over the baseline for both DP3-UNet and DP3-DiT, achieving new state-of-the-art results. Real-world experiments further demonstrate that our method performs well in practical deployments.
>
---
#### [replaced 034] EROAS: 3D Efficient Reactive Obstacle Avoidance System for Autonomous Underwater Vehicles using 2.5D Forward-Looking Sonar
- **分类: cs.RO**

- **简介: 该论文属于水下机器人避障任务，解决复杂环境中安全高效导航问题。提出EROAS系统，融合2.5D声呐与三个模块，提升避障性能。**

- **链接: [https://arxiv.org/pdf/2411.05516](https://arxiv.org/pdf/2411.05516)**

> **作者:** Pruthviraj Mane; Allen Jacob George; Rajini Makam; Subhash Gurikar; Rudrashis Majumder; Suresh Sundaram
>
> **备注:** Accepted for publication as a Technical Communication, Special Issue on AUV Symposium in the IEEE Journal of Oceanic Engineering (JOE)
>
> **摘要:** Autonomous Underwater Vehicles (AUVs) have advanced significantly in obstacle detection and path planning through sonar, cameras, and learning-based methods. However, safe and efficient navigation in cluttered environments remains challenging due to partial observability, turbidity, the limited field-of-view of forward-looking sonar (FLS), and occlusions that obscure obstacle geometry. To address these issues, we propose the Efficient Reactive Obstacle Avoidance Strategy (EROAS), a lightweight framework that augments a standard 2D FLS with a pivoting mechanism, effectively transforming it into a cost-efficient \emph{2.5D sonar}. This design provides vertical information on demand, extending situational awareness while minimizing computational overhead. EROAS integrates three complementary modules: first, Sonar Profile-guided Directional Decision Control (SPD2C) for rapid gap detection and generation of reference commands in both horizontal and vertical planes. Secondly, the Spatial Context Generator (SCG), which maintains a short-term obstacle memory of the past to mitigate partial observability, and finally, a Spatio-Temporal Control Barrier Function (ST-CBF) that enforces forward-invariance of safety constraints by filtering nominal references. Together, these components enable robust, reactive avoidance of obstacles in uncertain and cluttered 3D underwater settings. Simulation and hardware-in-the-loop (HIL) experiments validate the efficacy of the proposed EROAS algorithm, demonstrating improved trajectory efficiency, reduced travel time, and enhanced safety compared to conventional methods such as the Dynamic Window Approach (DWA) and Artificial Potential Fields (APF). this https URL
>
---
#### [replaced 035] TEACar: An Open-Source Autonomous Driving Platform
- **分类: cs.RO; eess.SY**

- **简介: 该论文介绍TEACAR，一个用于智能交通系统研究的开源自动驾驶平台。针对现有平台模块化不足的问题，设计了模块化机械结构和ROS 2软件架构，提升可扩展性和实验效率。**

- **链接: [https://arxiv.org/pdf/2604.24934](https://arxiv.org/pdf/2604.24934)**

> **作者:** Zhongzheng Zhang; Maxwell Ruyle; Andrew Kappes; Tyler Ruble; William Shaoul; Dana Moreno; Jack Penn; Ivan Ruchkin
>
> **摘要:** Intelligent Transportation Systems (ITS) increasingly rely on vision-based perception and learning-based control, necessitating experimental platforms that support realistic hardware-in-the-loop validation. Small-scale platforms for autonomous racing offer a practical path to hardware validation, but often suffer from limited modularity, high integration complexity, or restricted extensibility. This paper presents TEACAR, a 1/14- to 1/16-scale autonomous driving platform designed with modular mechanical architecture, hardware abstraction, and ROS 2-based software. The system adopts a four-layer deck structure that physically decouples sensing, computation, actuation, and power subsystems, improving structural rigidity while simplifying reconfiguration. We constructed and comprehensively evaluated the prototype of TEACAR. Its mechanical stability, structural characteristics, and software performance were quantified based on three CNN-based steering controllers. Inference latency, power consumption, and system operating time were measured to evaluate computational capability and robustness. Our experiments demonstrated that TEACAR offers a scalable, modular, and cost-effective testbed for ITS research, education, and development. Our project repository is available on GitHub.
>
---
#### [replaced 036] 3DRO: Lidar-level SE(3) Direct Radar Odometry Using a 2D Imaging Radar and a Gyroscope
- **分类: cs.RO**

- **简介: 该论文属于机器人状态估计任务，解决2D雷达数据中SE(3)位姿估计问题。通过融合陀螺仪数据，扩展原有SE(2)方法至SE(3)，实现高精度三维运动估计。**

- **链接: [https://arxiv.org/pdf/2604.12027](https://arxiv.org/pdf/2604.12027)**

> **作者:** Cedric Le Gentil; Daniil Lisus; Timothy D. Barfoot
>
> **备注:** Accepted for presentation at the ICRA 2026 Workshop on Radar in Robotics (poster: this https URL )
>
> **摘要:** Recently, the robotics community has regained interest in radar-based perception and state estimation. A 2D imaging radar provides dense 360deg information about the environment. Despite the radar antenna's cone of emission and reception, the collected data is generally assumed to be limited to the plane orthogonal to the radar's spinning axis. Accordingly, most methods based on 2D imaging radars only perform SE(2) state estimation. This paper presents 3DRO, an extension of the SE(2) Direct Radar Odometry (DRO) framework to perform state estimation in SE(3). While still assuming planarity of the data through DRO's 2D velocity estimates, it integrates 3D gyroscope measurements over SO(3) to estimate SE(3) ego motion. While simple, this approach provides lidar-level odometry accuracy as demonstrated using 643km of data from the Boreas-RT dataset.
>
---
#### [replaced 037] Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Uni-Hand框架，解决egocentric视角下手部运动预测问题，通过多模态输入和多任务预测提升预测精度与应用效果。**

- **链接: [https://arxiv.org/pdf/2511.12878](https://arxiv.org/pdf/2511.12878)**

> **作者:** Junyi Ma; Wentao Bao; Jingyi Xu; Guanzhong Sun; Yu Zheng; Erhang Zhang; Xieyuanli Chen; Hesheng Wang
>
> **备注:** Accepted by T-PAMI 2026. Code and data: this https URL
>
> **摘要:** Forecasting how human hands move in egocentric views is critical for applications like augmented reality and human-robot policy transfer. Recently, several hand trajectory prediction (HTP) methods have been developed to generate future possible hand waypoints, which still suffer from insufficient prediction targets, inherent modality gaps, entangled hand-head motion, and limited validation in downstream tasks. To address these limitations, we present a universal hand motion forecasting framework considering multi-modal input, multi-dimensional and multi-target prediction patterns, and multi-task affordances for downstream applications. We harmonize multiple modalities by vision-language fusion, global context incorporation, and task-aware text embedding injection, to forecast hand waypoints in both 2D and 3D spaces. A novel dual-branch diffusion is proposed to concurrently predict human head and hand movements, capturing their motion synergy in egocentric vision. By introducing target indicators, the prediction model can forecast the specific joint waypoints of the wrist or the fingers, besides the widely studied hand center points. In addition, we enable Uni-Hand to additionally predict hand-object interaction states (contact/separation) to facilitate downstream tasks better. As the first work to incorporate downstream task evaluation in the literature, we build novel benchmarks to assess the real-world applicability of hand motion forecasting algorithms. The experimental results on multiple publicly available datasets and our newly proposed benchmarks demonstrate that Uni-Hand achieves the state-of-the-art performance in multi-dimensional and multi-target hand motion forecasting. Extensive validation in multiple downstream tasks also presents its impressive human-robot policy transfer to enable robotic manipulation, and effective feature enhancement for action anticipation/recognition.
>
---
#### [replaced 038] Operating Within the Operational Design Domain: Zero-Shot Perception with Vision-Language Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文研究视觉语言模型在自动驾驶中的零样本ODD感知任务，旨在提升安全关键场景下的适应性与透明度。工作包括实验分析、优化策略评估及提示模板设计。**

- **链接: [https://arxiv.org/pdf/2605.07649](https://arxiv.org/pdf/2605.07649)**

> **作者:** Berkehan Ünal; Hauke Dierend; Dren Fazlija; Christopher Plachetka
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Over the last few years, research on autonomous systems has matured to such a degree that the field is increasingly well-positioned to translate research into practical, stakeholder-driven use cases across well-defined domains. However, for a wide-scale practical adoption of autonomous systems, adherence to safety regulations is crucial. Many regulations are influenced by the Operational Design Domain (ODD), which defines the specific conditions in which an autonomous agent can function. This is especially relevant for Automated Driving Systems (ADS), as a dependable perception of ODD elements is essential for safe implementation and auditing. Vision-language models (VLMs) integrate visual recognition and language reasoning, functioning without task-specific training data, which makes them suitable for adaptable ODD perception. To assess whether VLMs can function as zero-shot "ODD sensors" that adapt to evolving definitions, we contribute (i) an empirical study of zero-shot ODD classification and detection using four VLMs on a custom dataset and Mapillary Vistas, along with failure analyses; (ii) an ablation of zero-shot optimization strategies with a cost-performance overview; and (iii) a suite of reusable prompting templates with guidance for adaptation. Our findings indicate that definition-anchored chain-of-thought prompting with persona decomposition performs best, while other methods may result in reduced recall. Overall, our results pave the way for transparent and effective ODD-based perception in safety-critical applications.
>
---
#### [replaced 039] AR-VLA: True Autoregressive Action Expert for Vision-Language-Action Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出AR-VLA，解决机器人动作生成中的时序一致性问题，通过自回归机制实现上下文感知的动作序列生成。**

- **链接: [https://arxiv.org/pdf/2603.10126](https://arxiv.org/pdf/2603.10126)**

> **作者:** Yutong Hu; Jan-Nico Zaech; Nikolay Nikolov; Yuanqi Yao; Sombit Dey; Giuliano Albanese; Renaud Detry; Luc Van Gool; Danda Paudel
>
> **备注:** RSS 2026 accepted
>
> **摘要:** We propose a standalone autoregressive (AR) Action Expert that generates actions as a continuous causal sequence while conditioning on refreshable vision-language prefixes. In contrast to existing Vision-Language-Action (VLA) models and diffusion policies that reset temporal context with each new observation and predict actions reactively, our Action Expert maintains its own history through a long-lived memory and is inherently context-aware. This structure addresses the frequency mismatch between fast control and slow reasoning, enabling efficient independent pretraining of kinematic syntax and modular integration with heavy perception backbones, naturally ensuring spatio-temporally consistent action generation across frames. To synchronize these asynchronous hybrid V-L-A modalities, we utilize a re-anchoring mechanism that mathematically accounts for perception staleness during both training and inference. Experiments on simulated and real-robot manipulation tasks demonstrate that the proposed method can effectively replace traditional chunk-based action heads for both specialist and generalist policies. AR-VLA exhibits superior history awareness and substantially smoother action trajectories while maintaining or exceeding the task success rates of state-of-the-art reactive VLAs. Overall, our work introduces a scalable, context-aware action generation schema that provides a robust structural foundation for training effective robotic policies. Code and Videos available at this https URL
>
---
#### [replaced 040] A Radius of Robust Feasibility Approach to Directional Sensors in Uncertain Terrain
- **分类: math.OC; cs.RO**

- **简介: 该论文属于传感器网络任务，解决不确定地形中方向性传感器的覆盖优化问题。提出一种基于鲁棒可行半径的方法，通过分布式贪心算法提升覆盖率和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.19407](https://arxiv.org/pdf/2510.19407)**

> **作者:** Vanshika Datta; C. Nahak
>
> **摘要:** A sensor has the ability to probe its surroundings. However, uncertainties in its exact location can significantly compromise its sensing performance. The radius of robust feasibility defines the maximum range within which robust feasibility is ensured. This work introduces a novel approach integrating it with the directional sensor networks to enhance coverage using a distributed greedy algorithm. In particular, we provide an exact formula for the radius of robust feasibility of sensors in a directional sensor network. The proposed model strategically orients the sensors in regions with high coverage potential, accounting for robustness in the face of uncertainty. We analyze the algorithm's adaptability in dynamic environments, demonstrating its ability to enhance efficiency and robustness. Experimental results validate its efficacy in maximizing coverage and optimizing sensor orientations, highlighting its practical advantages for real-world scenarios.
>
---
#### [replaced 041] NoTVLA: Semantics-Preserving Robot Adaptation via Narrative Action Interfaces
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，旨在解决VLA模型的灾难性遗忘问题。通过引入NoTVLA框架，采用稀疏轨迹训练，提升模型泛化能力和效率。**

- **链接: [https://arxiv.org/pdf/2510.03895](https://arxiv.org/pdf/2510.03895)**

> **作者:** Zheng Huang; Mingyu Liu; Xiaoyi Lin; Muzhi Zhu; Canyu Zhao; Zongze Du; Ye Lin; Xiaoman Li; Yiduo Jia; Hao Zhong; Hao Chen; Chunhua Shen
>
> **摘要:** Vision-Language-Action (VLA) models represent a pivotal advance in embodied intelligence, yet they confront critical barriers to real-world deployment, most notably catastrophic forgetting. This issue stems from their overreliance on continuous action sequences or action chunks, which inadvertently create isolated data silos that disrupt knowledge retention across tasks. To tackle these challenges, we propose the Narrowing of Trajectory VLA (NoTVLA) framework: a novel approach that narrows its focus to sparse trajectories, thereby avoiding the catastrophic forgetting associated with dense trajectory fine-tuning. A key innovation of NoTVLA lies in its trajectory planning strategy: instead of centering on the target object's trajectory, it leverages temporal compression and spatial reasoning pruning specifically for the robot end effector's trajectory. Furthermore, training is conducted using these sparse trajectories rather than dense action trajectories, an optimization that delivers remarkable practical advantages with better performance in zero-shot. In multi-task evaluation scenarios, NoTVLA achieves superior performance and generalization compared to pi0 while operating under two critical constraints: it uses over an order of magnitude less computing power than pi0 and requires no wrist-mounted camera. This design ensures that NoTVLA's operational accuracy closely approximates that of single-task expert models. Crucially, it also preserves the model's inherent language capabilities, enabling zero-shot generalization in specific scenarios, supporting unified model deployment across multiple robot platforms, and fostering a degree of generalization even when perceiving tasks from novel perspectives.
>
---
#### [replaced 042] CoLA-Flow Policy: Temporally Coherent Imitation Learning via Continuous Latent Action Flow Matching for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出CoLA-Flow Policy，解决长时程机器人操作中的行为建模与稳定执行问题。通过在连续潜在动作空间中进行流匹配，提升轨迹平滑度与任务成功率。**

- **链接: [https://arxiv.org/pdf/2601.23087](https://arxiv.org/pdf/2601.23087)**

> **作者:** Wu Songwei; Jiang Zhiduo; Sun Wandong; Xie Guanghu; Zhao Rui; Liu Hong; Liu Yang
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** Learning long-horizon robotic manipulation requires jointly achieving expressive behavior modeling, real-time inference, and stable execution, which remains challenging for existing generative policies. Diffusion-based approaches offer strong modeling capacity but incur high inference latency, while flow matching enables fast, near-single-step generation yet often suffers from unstable execution when operating directly in the raw action space. We propose Continuous Latent Action Flow Policy (CoLA-Flow Policy), a trajectory-level imitation learning framework that performs flow matching in a continuous latent action space. By encoding action sequences into temporally coherent latent trajectories and learning an explicit latent-space flow, CoLA-Flow Policy decouples global motion structure from low-level control noise, enabling smooth and reliable long-horizon execution. The framework further integrates geometry-aware point cloud conditioning and execution-time multimodal modulation, using visual cues as a representative modality to enhance real-world robustness. Experiments in simulation and on real robots show that CoLA-Flow Policy achieves near-single-step inference, improves trajectory smoothness by up to 93.7% and task success by up to 25 percentage points over raw action-space flow baselines, while remaining significantly faster than diffusion-based policies.
>
---
#### [replaced 043] Language Conditioned Multi-Finger Dexterous Manipulation Enabled by Physical Compliance and Switching of Controllers
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在解决高阶推理与精细控制结合的问题。通过切换控制器和硬件柔性设计，提升机械手的灵巧性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2410.14022](https://arxiv.org/pdf/2410.14022)**

> **作者:** Cheng Pan; Kai Junge; Benhui Dai; Qinghua Guan; Josie Hughes
>
> **摘要:** Human dexterity arises from combining high-level task reasoning with finger-level dexterity control and physical compliance at the muscle and skin layers. In robotics, large Vision-Language-Action (VLA) models demonstrate text-conditioned high-level planning across diverse manipulation tasks, typically using pincher grippers. Smaller imitation-learning policies, conversely, show success in dexterous tasks using higher degree-of-freedom (DoF) grippers, but only for limited-scope tasks. However, few approaches combine high-level reasoning with dexterous, robust low-level control, which requires both intelligent control and compliant robot design. We propose a method inspired by the two-channel hypothesis of human motor control that combines these capabilities using a switching controller integrating high-level VLAs and smaller control models. Coordination between the two channels is managed through an event-driven switching mechanism that monitors subtask progression and completion, requiring minimal demonstration data by fine-tuning the VLA to predict event signals and training lightweight subtask-level dexterous policies. This approach is applied to our custom compliant 13-DoF anthropomorphic robotic hand, where compliance can be modulated to evaluate its impact on dexterity and robustness when combined with an autonomous policy. We show that hardware-level compliance in robotic fingers enables passive adaptation to disturbances and improves contact stability. The methodology is validated across a range of language-conditioned dexterous tasks. To demonstrate modularity, we show that adaptation to additional dexterous skills and different compliant hands can be achieved without retraining the VLA model. This provides an efficient, scalable, cross-embodiment approach to dexterity that leverages compliance while retaining the advantages of large AI models.
>
---
#### [replaced 044] False Feasibility in Variable Impedance MPC for Legged Locomotion
- **分类: cs.RO**

- **简介: 该论文研究变阻抗MPC在腿部运动中的可行性问题，指出参数可行集与实际可行集的差异，通过分析参数α解决控制指令不可实现的问题。**

- **链接: [https://arxiv.org/pdf/2604.22251](https://arxiv.org/pdf/2604.22251)**

> **作者:** Vishal Ramesh
>
> **备注:** Paper withdrawn to make some revisions in the discussion and experiments sections
>
> **摘要:** Variable impedance model predictive control (MPC) formulations often treat joint stiffness as an instantaneous decision variable. The resulting feasible set strictly contains the physically realizable set under first-order actuator dynamics. We identify this as a formulation error rather than a modeling approximation, formalize the distinction between the parameter-based feasible set F_param and the realizable set F_real, and characterize the regime of mismatch via the dimensionless parameter {\alpha} = {\omega}sT (actuator bandwidth times task timescale). For the 1D hopping monoped, we prove that below an analytical threshold {\alpha}_crit derived in closed form from task physics, no admissible stiffness command realizes the parameter-based prediction. Numerical validation in 1D shows monotonic deviation growth as {\alpha} decreases, with the predicted scaling holding across ten parameter combinations (log-log R2 = 0.986). Mechanism transfer to planar spring-loaded inverted pendulum dynamics confirms center-of-mass and stance-timing deviation as the primary consequence, with regime-dependent friction effects as a tertiary observable. A second threshold {\alpha}_infeas < {\alpha}_crit establishes a floor below which restricting the admissible stiffness range cannot repair realizability, closing the conservative-tuning objection. Augmenting the prediction state with stiffness closes the mismatch by construction.
>
---
#### [replaced 045] Cyclic Nullspace Coordination: Perpetual Flight of Aerial Carriers for Static Suspension
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究多无人机协同悬挂负载的持续飞行问题，提出一种基于循环零空间协调的算法，实现负载静止下的无人机永续飞行。**

- **链接: [https://arxiv.org/pdf/2503.03481](https://arxiv.org/pdf/2503.03481)**

> **作者:** Chiara Gabellieri; Yaolei Shen; Martina Paolucci; Antonio Franchi
>
> **备注:** Accepted for publications on the IEEE Transactions on Control Systems Technology
>
> **摘要:** This work demonstrates that the non-stop flights of three or more carriers are compatible with holding a constant pose of a cable-suspended load. It also presents an algorithm for generating the carriers' coordinated non-stop trajectories. The proposed method builds upon two pillars: (1) the choice of n special linearly independent directions of internal forces within the 3n-6-dimensional nullspace of the grasp matrix of the load, chosen as the edges of a Hamiltonian cycle on the graph that connects the cable attachment points on the load. Adjacent pairs of directions are used to generate n forces evolving on distinct 2D affine subspaces, despite the attachment points being generically in 3D; (2) the construction of elliptical trajectories within these subspaces by mapping, through appropriate graph coloring, each edge of the Hamiltonian cycle to a periodic coordinate while ensuring that no adjacent coordinates exhibit simultaneous zero derivatives. Combined with conditions for load statics and attachment point positions, these choices ensure that each of the n force trajectories projects onto the corresponding cable constraint sphere with non-zero tangential velocity, enabling perpetual motion of the carriers while the load is still. The work provides a scalable constructive design for any n greater than or equal to 3 with tuning guidelines, quantifies sensitivity and single-carrier failures, and provides a fixed-wing-compatible planner that preserves load statics under speed/bank/flight-path constraints. The theoretical findings are validated through simulations and laboratory experiments with quadrotor UAVs.
>
---
#### [replaced 046] SegSTRONG-C: Segmenting Surgical Tools Robustly On Non-adversarial Generated Corruptions -- An EndoVis'24 Challenge
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于手术工具分割任务，旨在解决模型在非对抗性噪声下的鲁棒性问题。通过构建数据集并评估不同方法的性能，探索提升模型鲁棒性的有效策略。**

- **链接: [https://arxiv.org/pdf/2407.11906](https://arxiv.org/pdf/2407.11906)**

> **作者:** Hao Ding; Yuqian Zhang; Tuxun Lu; Ruixing Liang; Hongchao Shu; Lalithkumar Seenivasan; Yonghao Long; Qi Dou; Cong Gao; Yicheng Leng; Seok Bong Yoo; Eung-Joo Lee; Negin Ghamsarian; Klaus Schoeffmann; Raphael Sznitman; Zijian Wu; Yuxin Chen; Septimiu E. Salcudean; Samra Irshad; Shadi Albarqouni; Seong Tae Kim; Yueyi Sun; An Wang; Long Bai; Hongliang Ren; Ihsan Ullah; Ho-Gun Ha; Attaullah Khan; Hyunki Lee; Satoshi Kondo; Satoshi Kasai; Kousuke Hirasawa; Sita Tailor; Ricardo Sanchez-Matilla; Imanol Luengo; Tianhao Fu; Jun Ma; Bo Wang; Marcos Fernández-Rodríguez; Estevao Lima; João L. Vilaça; Mathias Unberath
>
> **摘要:** Surgical data science has seen rapid advancement with the excellent performance of end-to-end deep neural networks (DNNs). Despite their successes, DNNs have been proven susceptible to minor "corruptions," introducing a major concern for the translation of cutting-edge technology, especially in high-stakes scenarios. We introduce the SegSTRONG-C challenge dedicated to better understanding model deterioration under unforeseen but plausible non-adversarial "corruption" and the capabilities of contemporary methods that seek to improve it. Built on a dataset generated through counterfactual robotic replay, SegSTRONG-C provides paired clean and "corrupted" samples, enabling reproducible evaluation of model robustness. Participants are challenged to train tool segmentation algorithms on "uncorrupted" data and evaluate them on "corrupted" test domains for the binary robot tool segmentation task. Through comprehensive baseline experiments and participating submissions from widespread community engagement, SegSTRONG-C reveals key themes for model failure and identifies promising directions for improving robustness. The performance of challenge winners, achieving an average 0.9394 DSC and 0.9301 NSD across the unreleased test sets with "corruption" types: bleeding, smoke, and low brightness. This highlights how prior knowledge, customized training strategies, and architectural choice can be leveraged to improve robustness. In conclusion, the SegSTRONG-C challenge has identified practical approaches for enhancing model robustness. However, most approaches rely on conventional techniques that have known limitations. Looking ahead, we advocate for expanding intellectual diversity and creativity in non-adversarial robustness beyond data augmentation, calling for new paradigms that enhance universal robustness to unforeseen "corruptions" to facilitate richer applications in surgical data science.
>
---
#### [replaced 047] Semantic-Aware UAV Command and Control for Efficient IoT Data Collection
- **分类: cs.RO**

- **简介: 该论文属于无人机协同任务，旨在解决物联网数据高效采集问题。通过语义通信与无人机控制结合，提升图像重建质量。**

- **链接: [https://arxiv.org/pdf/2604.08153](https://arxiv.org/pdf/2604.08153)**

> **作者:** Assane Sankara; Daniel Bonilla Licea; Hajar El Hammouti
>
> **备注:** Accepted for publication at the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP). v2: added clarification on the DDQN implementation and TSP algorithm
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) have emerged as a key enabler technology for data collection from Internet of Things (IoT) devices. However, effective data collection is challenged by resource constraints and the need for real-time decision-making. In this work, we propose a novel framework that integrates semantic communication with UAV command-and-control (C&C) to enable efficient image data collection from IoT devices. Each device uses Deep Joint Source-Channel Coding (DeepJSCC) to generate a compact semantic latent representation of its image to enable image reconstruction even under partial transmission. A base station (BS) controls the UAV's trajectory by transmitting acceleration commands. The objective is to maximize the average quality of reconstructed images by maintaining proximity to each device for a sufficient duration within a fixed time horizon. To address the challenging trade-off and account for delayed C&C signals, we model the problem as a Markov Decision Process and propose a Double Deep Q-Learning (DDQN)-based adaptive flight policy. Simulation results show that our approach outperforms baseline methods such as greedy and traveling salesman algorithms, in both device coverage and semantic reconstruction quality.
>
---
#### [replaced 048] Good in Bad (GiB): Sifting Through End-user Demonstrations for Learning a Better Policy
- **分类: cs.RO**

- **简介: 该论文属于模仿学习任务，旨在解决从低质量人类示范中学习安全有效策略的问题。提出GiB算法，自动识别并剔除错误子任务，提升策略性能。**

- **链接: [https://arxiv.org/pdf/2605.01529](https://arxiv.org/pdf/2605.01529)**

> **作者:** Noushad Sojib; Ola Ghattas; Momotaz Begum
>
> **摘要:** Imitation learning offers a promising framework for enabling robots to acquire diverse skills from human users. However, most imitation learning algorithms assume access to high-quality demonstrations an unrealistic expectation when collecting data from non-expert users, whose demonstrations often contain inadvertent errors. Naively learning from such demonstrations can result in unsafe policy behavior, while discarding entire demonstrations due to occasional mistakes wastes valuable data, especially in low-data settings. In this work, we introduce GiB (Good-in-Bad), an algorithm that automatically identifies and discards erroneous subtasks within demonstrations while preserving high-quality subtasks. The filtered data can then be used by any policy learning algorithm to train more robust policies. GiB first trains a self-supervised model to learn latent features and assigns binary weights to label each demonstration as good or bad. It then models the latent feature distribution of high-quality segments and uses the Mahalanobis distance to detect and evaluate poor-quality subtasks. We validate GiB on the Franka robot in both simulated and real-world multi-step tasks, demonstrating improved policy performance when learning from mixed-quality human demonstrations.
>
---
#### [replaced 049] Reinforcement Learning with Action Chunking
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文属于强化学习任务，旨在解决长时序、稀疏奖励环境下的探索与样本效率问题。通过引入动作分块技术，提升在线学习效果。**

- **链接: [https://arxiv.org/pdf/2507.07969](https://arxiv.org/pdf/2507.07969)**

> **作者:** Qiyang Li; Zhiyuan Zhou; Sergey Levine
>
> **备注:** The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025); 29 pages, 17 figures
>
> **摘要:** We present Q-chunking, a simple yet effective recipe for improving reinforcement learning (RL) algorithms for long-horizon, sparse-reward tasks. Our recipe is designed for the offline-to-online RL setting, where the goal is to leverage an offline prior dataset to maximize the sample-efficiency of online learning. Effective exploration and sample-efficient learning remain central challenges in this setting, as it is not obvious how the offline data should be utilized to acquire a good exploratory policy. Our key insight is that action chunking, a technique popularized in imitation learning where sequences of future actions are predicted rather than a single action at each timestep, can be applied to temporal difference (TD)-based RL methods to mitigate the exploration challenge. Q-chunking adopts action chunking by directly running RL in a 'chunked' action space, enabling the agent to (1) leverage temporally consistent behaviors from offline data for more effective online exploration and (2) use unbiased $n$-step backups for more stable and efficient TD learning. Our experimental results demonstrate that Q-chunking exhibits strong offline performance and online sample efficiency, outperforming prior best offline-to-online methods on a range of long-horizon, sparse-reward manipulation tasks.
>
---
#### [replaced 050] Scalable and Efficient Continual Learning from Demonstration via a Hypernetwork-generated Stable Dynamics Model
- **分类: cs.RO**

- **简介: 该论文属于机器人持续学习任务，解决多技能稳定记忆问题。提出超网络生成稳定动力学模型，提升持续学习性能并减少训练时间。**

- **链接: [https://arxiv.org/pdf/2311.03600](https://arxiv.org/pdf/2311.03600)**

> **作者:** Sayantan Auddy; Jakob Hollenstein; Matteo Saveriano; Antonio Rodríguez-Sánchez; Justus Piater
>
> **备注:** To appear in IEEE Transactions on Cognitive and Developmental Systems
>
> **摘要:** Robots capable of learning from demonstration (LfD) must exhibit stability while executing learned motion skills. To be effective in the real world, they should also remember multiple skills over time -- a capability lacking in current stable-LfD methods. We propose an approach to stable, continual LfD, and highlight the role of stability in improving continual learning. Our proposed hypernetwork generates the parameters of two neural networks: a trajectory learning dynamics model, and a trajectory-stabilizing Lyapunov function. These generated networks form a clock-augmented stable neural ODE solver (sNODE), a stable dynamics model that offers a superior stability-accuracy trade-off compared to the state-of-the-art. We further propose stochastic hypernetwork regularization with a single, uniformly-sampled task embedding, reducing the cumulative training time for $N$ tasks from O($N^2$) to O($N$) without degrading performance on real-world tasks. We introduce high-dimensional variants of the popular LASA dataset to assess scalability and extend a dataset of robotic LfD tasks to assess real-world performance. We empirically evaluate our approach on multiple LfD datasets of varying complexity, including sequences of 7--26 tasks, trajectories of 2--32 dimensions, and real-world tasks involving position and orientation. Our thorough evaluation on multiple LfD datasets demonstrates that our approach sequentially learns and retains multiple motion skills without retraining on past demonstrations, and outperforms other relevant baselines in terms of trajectory errors, continual learning scores, and stability metrics. Notably, we show that stability greatly enhances continual learning performance, particularly in size-efficient chunked hypernetworks. Our code is available at this https URL.
>
---
#### [replaced 051] Towards Robust Surgical Automation via Digital Twin Representations from Foundation Models
- **分类: cs.RO**

- **简介: 该论文属于手术自动化任务，旨在解决手术场景感知与规划问题。通过结合数字孪生和大语言模型，提升系统在复杂环境中的鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2409.13107](https://arxiv.org/pdf/2409.13107)**

> **作者:** Hao Ding; Lalithkumar Seenivasan; Hongchao Shu; Grayson Byrd; Han Zhang; Pu Xiao; Juan Antonio Barragan; Russell H. Taylor; Peter Kazanzides; Mathias Unberath
>
> **摘要:** Large language model-based (LLM) agents are emerging as a powerful enabler of robust embodied intelligence due to their capability of planning complex action sequences. Sound planning ability is necessary for robust automation in many task domains, but especially in surgical automation. These agents rely on a highly detailed natural language representation of the scene. Thus, to leverage the emergent capabilities of LLM agents for surgical task planning, developing similarly powerful and robust perception algorithms is necessary to derive a detailed scene representation of the environment from visual input. Previous research has focused primarily on enabling LLM-based task planning while adopting simple yet severely limited perception solutions to meet the needs for bench-top experiments, but lacks the critical flexibility to scale to less constrained settings. In this work, we propose an alternate perception approach -- a digital twin (DT)-based machine perception approach that capitalizes on the convincing performance and out-of-the-box generalization of recent vision foundation models. Integrating our DT representation and LLM agent for planning with the dVRK platform, we develop an embodied intelligence system and evaluate its robustness in performing peg transfer and gauze retrieval tasks. Our approach shows strong task performance and generalizability to varied environmental settings. Despite a convincing performance, this work is merely a first step towards the integration of DT representations. Future studies are necessary for the realization of a comprehensive DT framework to improve the interpretability and generalizability of embodied intelligence in surgery.
>
---
#### [replaced 052] GameChat: Multi-LLM Dialogue for Safe, Agile, and Socially Optimal Multi-Agent Navigation in Constrained Environments
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多智能体导航任务，解决约束环境中安全、敏捷和社交合规的导航问题。提出GameChat方法，通过自然语言沟通实现冲突自解，提升导航效率与优先级处理。**

- **链接: [https://arxiv.org/pdf/2503.12333](https://arxiv.org/pdf/2503.12333)**

> **作者:** Vagul Mahadevan; Shangtong Zhang; Rohan Chandra
>
> **摘要:** Safe, agile, and socially compliant multi-robot navigation in cluttered and constrained environments remains a critical challenge. This is especially difficult with self-interested agents with unique, unknown priorities in decentralized settings, where there is no central authority to resolve conflicts induced by spatial symmetry. We address this challenge by proposing an intuitive, but very effective approach, GameChat, which facilitates safe, agile, and deadlock-free navigation for both cooperative and self-interested agents in cluttered environments. Key to our approach is the idea that agents should resolve conflicts on their own using natural language to communicate, much like humans. We evaluate GameChat in simulated environments with doorways and intersections. The results show that even in the worst case, GameChat reduces the time for all agents to reach their goals by over 35% from a naive baseline and by over 20% from a state of the art baseline in the intersection scenario, while doubling the rate of ensuring the agent with a higher priority task reaches the goal first, from 50% (equivalent to random chance) to 100%. We also demonstrate how GameChat can be extended to more than two agents.
>
---
#### [replaced 053] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?
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
#### [replaced 054] Scalable Inspection Planning via Flow-based Mixed Integer Linear Programming
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决大规模点检测路径优化问题。通过流式混合整数线性规划方法，提升求解效率与质量。**

- **链接: [https://arxiv.org/pdf/2603.16593](https://arxiv.org/pdf/2603.16593)**

> **作者:** Adir Morgan; Kiril Solovey; Oren Salzman
>
> **摘要:** Inspection planning is concerned with computing the shortest robot path to inspect a given set of points of interest (POIs) using the robot's sensors. This problem arises in a wide range of applications from manufacturing to medical robotics. To alleviate the problem's complexity, recent methods rely on sampling-based methods to obtain a more manageable (discrete) graph inspection planning (GIP) problem. Unfortunately, GIP still remains highly difficult to solve at scale as it requires simultaneously satisfying POI-coverage and path-connectivity constraints, giving rise to a challenging optimization problem, particularly at scales encountered in real-world scenarios. In this work, we present highly scalable Mixed Integer Linear Programming (MILP) solutions for GIP that significantly advance the state-of-the-art in both runtime and solution quality. Our key insight is a reformulation of the problem's core constraints as a network flow, which enables effective MILP models and a specialized Branch-and-Cut solver that exploits the combinatorial structure of flows. We evaluate our approach on medical and infrastructure benchmarks alongside large-scale synthetic instances. Across all scenarios, our method produces substantially tighter lower bounds than existing formulations, reducing optimality gaps by 30-50% on large instances. Furthermore, our solver demonstrates unprecedented scalability: it provides non-trivial solutions for problems with up to 15,000 vertices and thousands of POIs, where prior state-of-the-art methods typically exhaust memory or fail to provide any meaningful optimality guarantees.
>
---
#### [replaced 055] Informative Path Planning with Guaranteed Estimation Uncertainty
- **分类: cs.RO**

- **简介: 该论文属于环境监测任务，解决资源受限下如何高效获取数据的问题。通过结合信息路径规划与不确定性保证，提出一种确保估计精度的路径规划方法。**

- **链接: [https://arxiv.org/pdf/2602.05198](https://arxiv.org/pdf/2602.05198)**

> **作者:** Kalvik Jakkala; Saurav Agarwal; Jason O'Kane; Srinivas Akella
>
> **备注:** 15 pages, 11 figures, RSS 2026
>
> **摘要:** Environmental monitoring robots often need to estimate data fields (e.g., salinity, temperature, bathymetry) under tight resource constraints. Classical boustrophedon lawnmower surveys provide geometric coverage guarantees but can waste effort by oversampling predictable regions. In contrast, informative path planning (IPP) methods leverage spatial correlations to reduce oversampling, yet typically offer no guarantees on estimation quality. This paper bridges these approaches by addressing IPP with guaranteed estimation uncertainty in complex environments: computing the shortest path whose measurements ensure that the Gaussian process (GP) posterior variance -- an intrinsic uncertainty measure that lower-bounds the mean-squared prediction error under the GP model -- is upper bounded by a user-specified threshold over the monitoring region. We propose a three-stage approach for efficient environmental monitoring: (i) learning a GP model from prior information; (ii) transforming the GP kernel into binary coverage maps that identify locations where uncertainty can be reduced below a target threshold; and (iii) planning a near-shortest route to satisfy the global uncertainty constraint. Our approach incorporates non-stationary kernels to capture spatially varying correlations in heterogeneous phenomena and accommodates non-convex environments with obstacles. We provide near-optimal approximation guarantees for both sensing-location selection and the joint selection-and-routing problem under a travel budget. Experiments on real-world topographic data demonstrate that our planners achieve uncertainty targets with fewer sensing locations and shorter travel distances than representative baselines. Furthermore, field experiments with autonomous surface and underwater vehicles validate the real-world feasibility of the approach. Our code is available at: this http URL
>
---
#### [replaced 056] Explicit Bounds on the Hausdorff Distance for Truncated mRPI Sets via Norm-Dependent Contraction Rates
- **分类: cs.RO; eess.SY; math.DS**

- **简介: 该论文属于控制理论领域，解决如何精确估计截断mRPI集与无限集之间的Hausdorff距离问题。通过推导显式上界，提供了一种非迭代的时域选择方法。**

- **链接: [https://arxiv.org/pdf/2511.18374](https://arxiv.org/pdf/2511.18374)**

> **作者:** Jiaxun Sun; Hengyu Xue; Yuyang Zhao
>
> **备注:** 6 pages, 5 figures. Accepted at the 2026 IEEE Conference on Control Technology and Applications (CCTA), Vancouver, BC, Canada, August 12-14, 2026
>
> **摘要:** We derive a computable closed-form upper bound on the Hausdorff distance between a truncated minimal robust positively invariant (mRPI) set and its infinite-horizon limit. The bound depends only on a disturbance-set size measure and an induced-norm contraction factor of the system matrix, and it yields an explicit, fully analytic horizon-selection rule that guarantees a prescribed approximation tolerance without iterative set computations. The choice of vector norm enters as a design lever: norm shaping -- through diagonal or Lyapunov-based weighting -- tightens both the contraction factor and the resulting certificate, with direct consequences for robust invariant-set approximation and tube-based model predictive control (MPC) constraint tightening. Numerical examples illustrate the accuracy, scalability, and practical impact of the proposed bound.
>
---
#### [replaced 057] Learning Tactile-Aware Quadrupedal Loco-Manipulation Policies
- **分类: cs.RO**

- **简介: 该论文研究四足机器人在接触丰富的环境中进行操作的任务。针对视觉和本体感知不足的问题，提出一种结合触觉的分层策略，提升操作性能。**

- **链接: [https://arxiv.org/pdf/2604.27224](https://arxiv.org/pdf/2604.27224)**

> **作者:** Pokuang Zhou; Yuhao Zhou; Quan Khanh Luu; Seungho Han; Heng Zhang; Binghao Huang; Yunzhu Li; Arash Ajoudani; Zhengtong Xu; Yu She
>
> **摘要:** Quadrupedal loco-manipulation is commonly built on visual perception and proprioception. Yet reliable contact-rich manipulation remains difficult: vision and proprioception alone cannot resolve uncertain, evolving interactions with the environment. Tactile sensing offers direct contact observability, but scalable tactile-aware learning framework for quadrupedal loco-manipulation is still underexplored. In this paper, we present a tactile-aware loco-manipulation policy learning pipeline with a hierarchical structure. Our approach has two key components. First, we leverage real-world human demonstrations to train a tactile-conditioned visuotactile high-level policy. This policy predicts not only end-effector trajectories for manipulation, but also the evolving tactile interaction cues that characterize how contact should develop over time. Second, we perform large-scale reinforcement learning in simulation to learn a tactile-aware whole-body control policy that tracks diverse commanded trajectories and tactile interaction cues, and transfers zero-shot to the real world. Together, these components enable coordinated locomotion and manipulation under contact-rich scenarios. We evaluate the system on real-world contact-rich tasks, including in-hand reorientation with insertion, valve tightening, and delicate object manipulation. Compared to vision-only and visuotactile baselines, our method improves performance by 28.54% on average across these tasks.
>
---
#### [replaced 058] Accurate Trajectory Tracking with MPCC for Flapping-Wing MAVs
- **分类: cs.RO**

- **简介: 该论文属于自主飞行控制任务，解决鸟型扑翼微型飞行器的精确轨迹跟踪问题。提出MPCC方法，实现在线优化和实时控制，提升跟踪精度。**

- **链接: [https://arxiv.org/pdf/2605.06042](https://arxiv.org/pdf/2605.06042)**

> **作者:** Charbel Toumieh; Jack Zeng; Niel Mistry; Dario Floreano
>
> **备注:** 7 pages, 6 figures
>
> **摘要:** Flapping-wing micro aerial vehicles offer quieter and safer operation than rotary-wing drones, yet achieving precise autonomous control of bird-scale ornithopters remains challenging: lift, airspeed, and turning authority are tightly coupled and governed by only a few control inputs. Conventional cascaded controllers treat altitude, speed, and heading independently, producing persistent tracking errors during complex maneuvers, while time-parameterized trajectory tracking requires predefined speed profiles that existing methods cannot robustly produce for these coupled dynamics. We address both limitations simultaneously with a Model Predictive Contouring Control (MPCC) approach that tracks arc-length-parameterized trajectories while optimizing progress online, eliminating the need for predefined timing. However, MPCC requires a dynamical model that captures the coupled aerodynamics without exceeding the computational budget of real-time nonlinear optimization. Here, we propose a compact, continuously differentiable model that captures the dominant couplings of bird-scale ornithopters, enabling real-time predictive control. We validated the method with the XFly ornithopter flying along circular and three-dimensional racing trajectories and achieved a mean deviation from the reference trajectory between 6.5 and 9 cm at speeds up to 3 m/s, which represents an almost 10-fold improvement over prior ornithopter control methods.
>
---
#### [replaced 059] Decentralized Heterogeneous Multi-Robot Collaborative Exploration for Indoor and Outdoor 3D Environments
- **分类: cs.RO**

- **简介: 该论文属于多机器人协同探索任务，旨在解决异构机器人在复杂3D环境中高效协作的问题。通过构建感知地图、优化任务分配与路径规划，提升探索效率和通信效率。**

- **链接: [https://arxiv.org/pdf/2604.23693](https://arxiv.org/pdf/2604.23693)**

> **作者:** Yuxiang Li; Kun Chen; Jiancheng Wang; Shihao Fang; Haoyao Chen; Yunhui Liu
>
> **摘要:** Heterogeneous multi-robot systems feature significant adaptability for complex environments. However, effective collaboration that fully exploits the robots' potential remains a core challenge. This paper proposes a decentralized collaborative framework for heterogeneous multi-robot systems to autonomously explore indoor and outdoor 3D environments. First, a basic perception map that integrates terrain and observation metrics is designed. Improved supervoxel segmentation is developed to simplify the map structure and form a high-level representation that supports lightweight communication. Second, the traversal and observation capabilities of heterogeneous robots are modeled to evaluate the requirements of task views derived from incomplete supervoxels. These task views are grouped by requirements and clustered to streamline assignment. Subsequently, the view-cluster assignment is formulated as a heterogeneous multi-depot multi-traveling salesman problem (HMDMTSP) that incorporates constraints between view-cluster requirements and robot capabilities. An improved genetic algorithm is developed to efficiently solve this problem while ensuring global consistency. Based on the assignments, redundant views within clusters are eliminated to refine exploration routes. Finally, conflicts between robots' motion paths are resolved. Simulations and field experiments in cluttered indoor and outdoor environments demonstrate that our approach effectively coordinates exploration tasks among heterogeneous robots, achieving superior exploration efficiency and communication savings compared to state-of-the-art approaches.
>
---
#### [replaced 060] Wavelet Policy: Imitation Learning in the Scale Domain with World Prior Memory
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉运动模仿学习任务，旨在解决长时序操作中的场景感知与记忆问题。提出Wavelet Policy框架，结合多尺度动作建模与世界先验记忆，提升长期任务性能。**

- **链接: [https://arxiv.org/pdf/2504.04991](https://arxiv.org/pdf/2504.04991)**

> **作者:** Changchuan Yang; Yuhang Dong; Guanzhong Tian; Haizhou Ge; Hongrui Zhu
>
> **摘要:** Conventional visuomotor imitation learning usually predicts future robot actions directly in the time domain. Such formulations often have limited physical scene awareness and weak long-horizon memory. In contrast, world-model-based perception and memory-augmented policies can improve world awareness with substantial computation overhead. In this work, we propose Wavelet Policy, a lightweight imitation learning framework that combines World Prior Memory (WPM) with wavelet-based multi-scale action modeling. Our key idea is to encode persistent physical scene structure from static background images into compact memory tokens, which are fused into world-prior tokens and injected into the encoder during forward propagation. Based on this memory-conditioned representation, We further perform wavelet-domain decomposition over horizon-aligned latent action tokens and adopt a Single-Encoder Multiple-Decoder (SE2MD) architecture to model latent components at different temporal scales. The resulting latent subbands are reconstructed through inverse wavelet transform and finally projected into executable action chunks. To facilitate efficient world prior learning, we introduce a world-prior adaptation loss, encouraging the background encoder to retain persistent scene knowledge while remaining lightweight and stable. Extensive experiments on four simulated and six real-world robotic manipulation tasks show that Wavelet Policy consistently outperforms strong baselines. These results demonstrate that combining scale-domain action modeling with world-prior memory provides an effective and efficient solution for long-horizon embodied manipulation. We release the source code, data and model checkpoint of simulation task at this https URL.
>
---
#### [replaced 061] HiVLA: A Visual-Grounded-Centric Hierarchical Embodied Manipulation System
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出HiVLA系统，解决机器人操作中视觉-语言-动作模型的推理与控制分离问题，通过分层架构提升长任务和精细操作性能。**

- **链接: [https://arxiv.org/pdf/2604.14125](https://arxiv.org/pdf/2604.14125)**

> **作者:** Tianshuo Yang; Guanyu Chen; Yutian Chen; Zhixuan Liang; Yitian Liu; Zanxin Chen; Chunpu Xu; Haotian Liang; Jiangmiao Pang; Yao Mu; Ping Luo
>
> **备注:** Project Page: this https URL
>
> **摘要:** While end-to-end Vision-Language-Action (VLA) models offer a promising paradigm for robotic manipulation, fine-tuning them on narrow control data often compromises the profound reasoning capabilities inherited from their base Vision-Language Models (VLMs). To resolve this fundamental trade-off, we propose HiVLA, a visual-grounded-centric hierarchical framework that explicitly decouples high-level semantic planning from low-level motor control. In high-level part, a VLM planner first performs task decomposition and visual grounding to generate structured plans, comprising a subtask instruction and a precise target bounding box. Then, to translate this plan into physical actions, we introduce a flow-matching Diffusion Transformer (DiT) action expert in low-level part equipped with a novel cascaded cross-attention mechanism. This design sequentially fuses global context, high-resolution object-centric crops and skill semantics, enabling the DiT to focus purely on robust execution. Our decoupled architecture preserves the VLM's zero-shot reasoning while allowing independent improvement of both components. Extensive experiments in simulation and the real world demonstrate that HiVLA significantly outperforms state-of-the-art end-to-end baselines, particularly excelling in long-horizon skill composition and the fine-grained manipulation of small objects in cluttered scenes.
>
---
