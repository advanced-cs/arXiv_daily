# 机器人 cs.RO

- **最新发布 34 篇**

- **更新 20 篇**

## 最新发布

#### [new 001] Instrumentation for Imitation Learning: Enhancing Training Datasets for Clothes Hanger Insertion
- **分类: cs.RO**

- **简介: 该论文研究机器人抓取任务，解决数据需求高导致模仿学习效果受限的问题。通过传感器集成提供状态信息，提升模仿学习性能。**

- **链接: [https://arxiv.org/pdf/2605.23847](https://arxiv.org/pdf/2605.23847)**

> **作者:** Remko Proesmans; Thomas Lips; Francis wyffels
>
> **备注:** Accepted for presentation at ICRA2026
>
> **摘要:** Large behaviour models have transformed the field of robotic manipulation, but prohibitive data requirements have thus far prevented a revolution similar to vision language models. We believe that instrumentation, i.e. sensor integration in objects, can provide invaluable state information and enable efficient learning for robotic manipulation. In this paper, we present instrumented imitation learning of clothes hanger insertion. Using 180 teleoperated demonstrations, we train diffusion policies with and without access to instrumentation data. Results show that policies leveraging instrumentation outperform vision-only counterparts by 14-25 %pt and exhibit greater task awareness. Crucially, a black-box imitation learning policy learns to prioritise instrumentation signals without explicit guidance. In addition, enhancing the teleoperation dataset with rollouts from an instrumented expert policy, enables a vision-only student policy to achieve performance comparable to the instrumented expert, thereby surpassing the original vision-only policy. These findings establish instrumentation as a promising strategy to enhance imitation learning for robotic manipulation. Datasets are available on Zenodo.
>
---
#### [new 002] Four Simple Proprioceptive Estimators for Legged Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人状态估计任务，解决腿式机器人IMU漂移问题。通过利用足部接触信息，提出四种改进的估计算法，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2605.23100](https://arxiv.org/pdf/2605.23100)**

> **作者:** Frank Dellaert; Chiyun Noh; Varun Agrawal; Ayoung Kim
>
> **摘要:** Legged robots carry an IMU, but the inertial solution drifts because consumer-grade IMUs are noisy. However, the feet create intermittent contacts with the environment that can be used to mitigate that drift. This report develops a sequence of increasingly expressive legged robot state estimators that leverage this. In all cases, the floating-base state comprises attitude, position, velocity, and IMU biases. To model foot contacts, we start from the contact-aided invariant EKF of Hartley et al., albeit at a reduced contact update rate. This is then augmented by replacing the measurement update by a small factor graph. Finally, we turn the same factors into a fixed-lag smoother with contact-episode footholds, with and without an evolving IMU bias. To facilitate reproducibility and further research in proprioceptive legged odometry, all four variants are available in GTSAM (Dellaert et. al), and we additionally provide a ROS2-compatible implementation.
>
---
#### [new 003] Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Any2Any，解决人形机器人全身跟踪模型跨平台迁移问题，通过运动学对齐和动态微调实现高效迁移，显著降低训练成本。**

- **链接: [https://arxiv.org/pdf/2605.23733](https://arxiv.org/pdf/2605.23733)**

> **作者:** Ming Yang; Tao Yu; Feng Li; Hua Chen
>
> **摘要:** Whole-body tracking (WBT) models have become a key foundation for humanoid robots, enabling them to imitate diverse motions with high fidelity. Training such models from scratch requires large-scale data and computation, making rapid deployment on new humanoid platforms costly. This raises a natural question: Can pretrained WBT models transfer across embodiments with minimal adaptation? To answer this question, we propose Any2Any, a paradigm that efficiently transfers an existing WBT specialist to a new humanoid embodiment with only a small amount of data and compute. Any2Any first performs kinematic alignment between source and target humanoids, aligning their input and output spaces so that the pretrained source policy can be meaningfully reused on the target embodiment.Any2Any then performs dynamics adaptation by applying lightweight parameter-efficient fine-tuning (PEFT) components to selected dynamics-sensitive modules, preserving useful behavioral priors while enabling targeted adaptation to the target robot. Extensive experiments on multiple humanoid platforms and pretrained backbones show that Any2Any substantially accelerates convergence and reduces training cost compared with training from scratch, while achieving competitive or superior tracking performance. Notably, using only 1% of the compute and data required for full training, Any2Any successfully transfers Sonic models pre-trained on Unitree G1 to LimX Oli and LimX Luna. These results suggest that pretrained WBT specialists can be efficiently reused across embodiments, providing a scalable path toward deploying humanoid whole-body control on new robots.
>
---
#### [new 004] Signal Temporal Logic Motion Planning via Graphs of Convex Sets
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于运动规划任务，解决在STL约束下生成满足逻辑与时间要求的平滑轨迹问题，通过结合定时自动机与凸集图实现高效求解。**

- **链接: [https://arxiv.org/pdf/2605.23240](https://arxiv.org/pdf/2605.23240)**

> **作者:** Yu Chen; Ancheng Hou; Mingyang Feng; Xiao Yu; Xiang Yin
>
> **摘要:** This paper investigates continuous-time motion planning under Signal Temporal Logic (STL) specifications. The goal is to generate smooth robot trajectories that satisfy high-level logical and timing requirements while respecting low-level motion constraints. To this end, we propose an efficient framework that combines timed-automata reasoning with graphs of convex sets (GCS). An STL specification is first represented by a timed automaton, which is then coupled with a convex decomposition of the configuration space to form a joint transition system encoding both task progress and region occupancy. Based on this joint transition system, the STL motion-planning problem is reformulated as a shortest-path problem over a GCS, whose solution induces a smooth Bézier-spline trajectory satisfying the STL specification, smoothness requirements, and velocity bounds. We establish the soundness of the proposed formulation and analyze its computational complexity, showing that, once the timed automaton and convex decomposition are fixed, the convex relaxation scales polynomially with the configuration-space dimension and the Bézier degree. We further develop a compact timed-automaton construction for an expressive STL fragment using dedicated templates and Boolean composition. Numerical experiments on low-dimensional benchmarks, a $3$-D quadrotor, a $30$-DoF humanoid, and a hardware experiment on a UR-3 robot arm demonstrate that the proposed method efficiently solves complex STL motion-planning problems and produces smooth executable trajectories.
>
---
#### [new 005] UfM*: Uncertainty from Motion* for DNN Depth Estimation Using Gaussians
- **分类: cs.RO**

- **简介: 该论文属于单目深度估计任务，旨在解决安全关键系统中不确定性估计的问题。提出UfM*算法，通过高斯混合高效计算多视角不一致，提升不确定性估计效率。**

- **链接: [https://arxiv.org/pdf/2605.23098](https://arxiv.org/pdf/2605.23098)**

> **作者:** Soumya Sudhakar; Sertac Karaman; Vivienne Sze
>
> **备注:** 18 pages, 15 figures
>
> **摘要:** Reliable uncertainty estimation is critical for deploying monocular depth deep neural networks (DNNs) in safety-critical robotic systems. Conventional uncertainty methods such as ensembles and sampling-based approaches require multiple inferences per image, incurring substantial compute and memory overhead. Moreover, uncertainty predicted from a single image misses out on measuring disagreement between predictions across views of the same region. We propose Uncertainty from Motion* (UfM*), an uncertainty estimation algorithm that measures multiview disagreement efficiently by comparing previous and current views using a compact Gaussian mixture, requiring only a single DNN inference per image. Using Gaussians to compute multiview disagreement is not only more compute- and memory-efficient than a prior approach using a point cloud, but also improves uncertainty by measuring disagreement across regions of 3D space. UfM* paired with aleatoric uncertainty improves expected calibration error by 24-28% compared to an ensemble, while requiring only 3% of the energy and 0.02% of the memory on 100 out-of-distribution ScanNet sequences. We demonstrate UfM* consumes only 63 mJ per 224x224 image while running real-time at 30 FPS on an Arm Cortex-A76 CPU onboard a miniature energy-constrained robot, highlighting that measuring multiview disagreement using Gaussians enables efficient uncertainty for resource-constrained robotic systems.
>
---
#### [new 006] Sparse Compositional Flow Matching by geometric assembly from motion primitives
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人轨迹生成任务，旨在解决传统方法对轨迹结构建模不足的问题。通过构建可组合的运动基元和物理空间流匹配，提升轨迹生成精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.23341](https://arxiv.org/pdf/2605.23341)**

> **作者:** Yan Tang; Yuanbo Tang; Tingyu Cao; Shaolun Huang; Yang Li
>
> **摘要:** Embodied trajectories, such as the executable motion sequences of robotic manipulators, underwater vehicles, and mobile robots, are a fundamental output of embodied AI. Modern generative models often treat them as a dense, monolithic signal generated point by point, fitting an intricate high-dimensional posterior while leaving the data's latent structure unmodeled, the same sample inefficiency long identified by the structured generative model literature. We argue that a compositional latent structure is a natural choice: many embodied tasks share recurring motion fragments that can be made explicit as a finite repertoire of reusable motion primitives, and compositional units naturally align with subtask boundaries to support task decomposition. Existing compositional generators, however, compose in a latent space and rely on post-hoc decoding to relate sampled units to actual trajectory segments. We instead compose directly in the physical trajectory space through a flow-matching framework with two coupled designs. Motion-Primitive Dictionary Learning equips each atom with a learnable length mask and binary starting indicators so the atom itself is the primitive, reused verbatim wherever it is placed. Structural Sparse Flow Matching with Geometric Constraints then generates a binary placement matrix using duration-aware tokenization and a differentiable geometric loss that enforces spatial continuity and temporal contiguity where adjacent primitives meet. On Open X-Embodiment and 3DMoTraj, the framework attains state-of-the-art accuracy and reduces the FDE/ADE ratio from 1.8 to 1.07, improving ADE by 19.2% and FDE by 21.0% over the strongest baseline.
>
---
#### [new 007] Turning Adaptation into Assets: Cross-Domain Bridging for Online Vision-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，解决在线适应中的灾难性遗忘和负迁移问题。提出IDEA框架，通过积累和组合资产实现跨域桥梁构建。**

- **链接: [https://arxiv.org/pdf/2605.23257](https://arxiv.org/pdf/2605.23257)**

> **作者:** Zixuan Hu; Xuantuo Huang; Yancheng Li; Yichun Hu; Shengyong Xu; Ling-Yu Duan
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Navigating under non-stationary environment shifts poses a critical challenge for a Vision-and-Language Navigation (VLN) agent deployed in the wild. Yet, existing Test-Time Adaptation (TTA) methods for VLN largely treat online adaptation as transient, isolated updates, leading to catastrophic forgetting and negative transfer. To overcome these issues, we propose Inter-Domain BridgE with Historical Assets (IDEA), a novel TTA framework that transforms adaptation into the accumulation and composition of assets. Specifically, IDEA introduces soft prompts optimized via a Fisher-guided weighting scheme to capture the transferable knowledge. These optimized prompts are then augmented with domain coordinates to form a dynamic asset library. Leveraging this library, IDEA constructs a cross-domain bridge by projecting the target domain onto the convex hull of historical knowledge. These designs form a complementary loop: the evolving library underpins bridge construction, while the bridge provides superior initialization to accelerate asset optimization. Extensive experiments across REVERIE, R2R, and R2R-CE benchmarks demonstrate the consistent superiority of IDEA over existing methods, showcasing its ability to enable training-free adaptation via asset sharing.
>
---
#### [new 008] SFG-ROS: A Resource-Aware Framework for Dense Multi-Agent Perception
- **分类: cs.RO**

- **简介: 该论文属于多智能体感知任务，旨在解决ROS 2在密集传感器流传输中的网络拥堵与计算开销问题。提出SFG-ROS框架，通过流量路由、解码管道和硬件适配优化性能。**

- **链接: [https://arxiv.org/pdf/2605.23832](https://arxiv.org/pdf/2605.23832)**

> **作者:** Constantin Blessing; Elias Geiger; Jakob Häringer; Dennis Grewe; Markus Enzweiler
>
> **摘要:** Deploying heterogeneous multi-agent robot fleets for collaborative perception requires robust data exchange and scalable software architectures. However, standard ROS 2 implementations often suffer from network saturation, namespace collisions, and severe computational overhead when distributing dense sensor streams across devices. To address these bottlenecks, we present SFG-ROS, a resource-aware multi-agent software framework designed for dynamic fleet deployments. SFG-ROS addresses these challenges through three primary contributions. First, schema-driven traffic routing isolates high-frequency intra-agent traffic from the global network using a programmatic fully qualified name schema and targeted Fast DDS routing. Second, an on-demand centralized decoding pipeline automatically offloads high-bandwidth sensor data decompression, eliminating redundant processing across local consumer nodes. Finally, a hardware-agnostic container pipeline dynamically adapts to heterogeneous accelerators, seamlessly bridging development environments with zero-touch, field-ready execution. We evaluate the framework using a fleet of wheeled and legged robots equipped with LiDAR and stereo depth cameras. Experimental results show SFG-ROS bounds network traffic to $\mathcal{O}(1)$ and, by replacing redundant decompression with lightweight IPC, reduces the per-subscriber CPU scaling penalty by 72.3\% versus standard ROS 2, all while maintaining low latency. Finally, we publish SFG-ROS under a permissive license, available via \href{this https URL}{this http URL}.
>
---
#### [new 009] Vision-Based Agile Landing on Turbulent Waters
- **分类: cs.RO**

- **简介: 该论文属于自主飞行器自主降落任务，解决在湍流海面中无人旋翼机精准着陆的问题。通过强化学习方法，利用视觉特征和飞行器状态信息进行姿态与推力控制，实现无需平台状态信息的自主降落。**

- **链接: [https://arxiv.org/pdf/2605.23717](https://arxiv.org/pdf/2605.23717)**

> **作者:** Dimosthenis Angelis; Leonard Bauersfeld; Davide Scaramuzza; Evangelos Boukas
>
> **摘要:** Autonomous landing of Unmanned Aerial Vehicles on maritime vessels is challenging due to the coupled motion of the vehicle and landing platform in open-sea conditions. This paper presents a reinforcement-learning-based approach for autonomous multirotor landing on moving maritime platforms without requiring explicit platform-state information. The proposed method uses multirotor state measurements together with local visual features, consisting of keypoints and associated descriptors extracted from the landing surface, to predict attitude and thrust commands. These commands are tracked by a conventional low-level controller. The policy is trained in simulation using synthetic keypoints with randomly generated normalized descriptors, enabling zero-shot deployment with different local feature extractors onboard the UAV. We evaluate the method in a realistic simulator and show that it outperforms a state-of-the-art Model Predictive Control baseline under platform motions corresponding to ``Very Rough'' sea conditions. Finally, we perform extensive real-world experiments, demonstrating autonomous onboard landing using two different local feature extractors. To the best of our knowledge, this is the first approach for agile multirotor landing on maritime platforms in turbulent waters that does not rely on an explicit platform-state representation.
>
---
#### [new 010] Verified Task-Space Motion Planning Under Joint-Space Constraints
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决关节空间约束下任务空间规划的步长适应问题。通过计算可证可达的笛卡尔超矩形，确保规划过程不违反关节限制，提升规划成功率。**

- **链接: [https://arxiv.org/pdf/2605.22991](https://arxiv.org/pdf/2605.22991)**

> **作者:** Hanjiang Hu; Changliu Liu; Yebin Wang
>
> **摘要:** Reactive task-space planners such as Bug2 operate with fixed Cartesian step sizes and are unaware of the manipulator's joint-angle limits. When the Jacobian is poorly conditioned, even small Cartesian steps can demand joint changes that exceed admissible bounds; clipping the joints to their limits causes tracking drift and can prevent goal reaching entirely. We address this by computing, at each planning step, the largest Cartesian hyperrectangle that is \emph{certifiably reachable} under joint displacement bounds. Using a second-order polynomial approximation of the inverse kinematics and the S-procedure, we formulate a small semidefinite program whose solution yields the certified half-width~$\lambda^\star$. An equivalent bisection procedure exploiting the quadratic structure solves the certification in sub-millisecond time. Integrating this certificate with Bug2 yields a planner whose step size adapts to local kinematic conditioning. In a statistical evaluation over 94 adversarial scenarios spanning six joint-limit settings, the SOS-verified planner achieves \emph{zero} joint-limit violations with a 100\% goal-reaching rate, whereas a standard Bug2 planner violates joint limits in 6--11\% of steps and fails to reach the goal in up to 18\% of scenarios.
>
---
#### [new 011] $π_0$-EqM: Equilibrium Matching for Closed-Loop Vision-Language-Action Control
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决VLA模型中动作生成的计算效率与成功率问题。提出$\pi_0$-EqM方法，通过等平衡解码提升控制性能。**

- **链接: [https://arxiv.org/pdf/2605.23128](https://arxiv.org/pdf/2605.23128)**

> **作者:** Huanming Liu; Congsheng Xu; Jianmin Ji; Yao Mu
>
> **备注:** Preprint. 5 pages, 3 figures
>
> **摘要:** Currently, Vision-Language-Action (VLA) models have become the most adopted paradigm for robotic manipulation for its great potential for task generalization. While most generative flow-matching action decoders for VLA control are often deployed with fixed sampling horizons, limiting state-dependent compute and temporal reuse across control cycles. We present $\pi_0$-EqM, which replaces the flow-matching expert in $\pi_0$ with an Equilibrium Matching (EqM) decoder while leaving the upstream VLA stack unchanged. Under a matched 300-step budget, $\pi_0$-EqM improves RoboTwin average success from 40.4% to 50.2% across 19 tasks and remains competitive on LIBERO, with its clearest gain on LIBERO-10 (87.0%). Two threshold scans reveal a task-dependent non-monotonic relation between residual and success, which we term the stationarity--executability gap. The results suggest that inference depth in iterative VLA control is part of policy design and introduce an energy-based VLA perspective that may inform future work on composable action generation across tasks and embodiments.
>
---
#### [new 012] How Many Training Samples Are Needed for the Inverse Kinematics Solutions by Artificial Neural Networks
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人逆运动学任务，研究如何确定足够训练样本数量以提高神经网络的精度与效率，通过实验分析数据量对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2605.23583](https://arxiv.org/pdf/2605.23583)**

> **作者:** Dong-Won Lim
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** Inverse Kinematics (IK) plays a critical role in robotic motion planning and control. The IK solutions of a robot manipulator could be done by conventional ways such as geometric, algebraic, or Jacobian methods, which have drawbacks. The Artificial Neural Networks (ANNs) have become a promising alternative for approximating IK solutions due to their generalization ability and computational efficiency. This approach basically trains only a few samples of the end effector that are recorded for the solution of the IK problem. However, a fundamental question remains: how many training samples are sufficient to achieve reliable and accurate IK predictions? This study investigates the mathematical framework of relating the size of training datasets and the accuracy of ANN-based IK solvers. Using an articulated robotic manipulator, we generate varying amounts of joint-position pairs to train feedforward neural networks and assess their accuracy, convergence, and generalization capability. The results reveal more training samples than 125 did not contribute to the improvement of the model efficiency that the comparable measure dealing with the approximation accuracy over the sampling size, offering valuable insight into data efficiency. This work provides practical guidance for optimizing the data sizing of ANN solutions, balancing computational cost and model accuracy for real-world robotic applications.
>
---
#### [new 013] Point Tracking Improves World Action Models
- **分类: cs.RO**

- **简介: 该论文属于机器人策略学习任务，解决环境动态建模中像素预测受干扰的问题。提出JOPAT模型，结合像素与点轨迹预测，提升长时序任务性能。**

- **链接: [https://arxiv.org/pdf/2605.23856](https://arxiv.org/pdf/2605.23856)**

> **作者:** Jiarui Guan; Wenshuai Zhao; Yue Pei; Ziliang Chen; Arno Solin; Juho Kannala
>
> **摘要:** Robot policy learning benefits from world-action models that capture environment dynamics, but pixel-level prediction entangles dynamics with nuisance factors such as lighting and texture, making learned representations vulnerable to task-irrelevant visual variation. We propose JOPAT, a JOint Pixel-And-Track World-Action Model that predicts latent visual observations, 2D point tracks with visibility, and actions in a single denoising diffusion transformer. The key insight is that tracks provide an explicit representation of motion that captures long-horizon dynamics and remains robust under occlusion or partial out-of-frame motion, offering greater utility than modeling pixel appearance alone. On LIBERO and real-world LeRobot tasks, JOPAT improves over pixel-based baselines, with the largest gains on long-horizon tasks involving occlusion, object interaction, and off-screen motion.
>
---
#### [new 014] Multi-Floor Exploration for Ground Robots via an Incremental Reachable Graph and Structural Priors
- **分类: cs.RO**

- **简介: 该论文属于多楼层机器人探索任务，解决传统二维地图无法表示多层可行走区域的问题。通过构建增量可达图和结构先验，提升探索效率与地图完整性。**

- **链接: [https://arxiv.org/pdf/2605.23350](https://arxiv.org/pdf/2605.23350)**

> **作者:** Zhiwen Zhu; Jiaqi Chen; Xiangyi Huang; Meiqi Hu; Boyu Zhou
>
> **摘要:** Autonomous exploration of multi-floor buildings remains challenging for ground robots because conventional 2D and 2.5D maps cannot represent overlapping traversable surfaces such as stairs, ramps, and multiple reachable elevations. This letter presents a multi-floor exploration framework based on an incremental reachable graph. Built as a sparse graph over reachable support surfaces, the graph preserves potentially valid connectivity through tentative graph elements under sparse observations and enables stable, physically reachable frontier detection. To guide exploration beyond the currently mapped floor, we project task-zone priors from an explored floor to initialize a hypothetical graph on the target floor and reconcile it incrementally with incoming observations. A hierarchical planner then jointly reasons over confirmed and hypothetical structures for global guidance. In simulation, the proposed method demonstrates improved exploration efficiency and mapping completeness compared to evaluated baselines. Furthermore, onboard real-world experiments validate its practical feasibility and real-time performance.
>
---
#### [new 015] PIMbot: A Self-Adaptive Attack Framework for Adversarial Manipulation of Multi-Robot Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文提出PIMbot框架，用于操控多机器人强化学习中的社会困境问题。通过奖励和策略操纵实现有效干预，解决合作任务中的脆弱性暴露问题。**

- **链接: [https://arxiv.org/pdf/2605.23027](https://arxiv.org/pdf/2605.23027)**

> **作者:** Zexin Li; Ziliang Zhang; Hyoseung Kim; Cong Liu
>
> **备注:** Extension version of IROS'23
>
> **摘要:** Recent research has demonstrated the potential of reinforcement learning in effective multi-robot collaboration, particularly in social dilemmas where robots face a trade-off between self-interest and collective benefits. However, environmental factors such as miscommunication and adversarial robots can impact cooperation, making it crucial to explore how multi-robot communication can be manipulated to achieve different outcomes. This paper presents PIMbot, a framework that manipulates outcomes via two complementary levers: (i) incentive manipulation of the reward channel and (ii) policy manipulation of an agent's own actions. An adaptive multi-objective controller balances these levers in an online manner. Our work introduces a novel approach to manipulation in recent multi-agent RL social dilemmas that utilize a unique reward function for incentivization. By utilizing our proposed PIMbot mechanisms, a robot is able to manipulate the social dilemma environment effectively. Comprehensive experimental results demonstrate the effectiveness of our proposed methods in the Gazebo-simulated multi-robot environment. Moreover, a real embedded device case study on NVIDIA Jetson Orin Nano quantifies system cost and validates PIMbot's effectiveness on realistic autonomous embedded systems scenarios beyond simulation. Together, these results position PIMbot as a rigorous stress-test tool exposing critical vulnerabilities in multi-robot cooperative tasks.
>
---
#### [new 016] Robotic Strawberry Harvesting with Robust Vision and Deep Reinforcement Learning based Sim-to-Real Control
- **分类: cs.RO**

- **简介: 该论文属于农业机器人任务，旨在解决草莓采摘中的感知与控制问题。通过改进的视觉模型和模拟训练的深度强化学习控制，实现高效可靠的采摘。**

- **链接: [https://arxiv.org/pdf/2605.23863](https://arxiv.org/pdf/2605.23863)**

> **作者:** Al Bashir; Shao-Yang Chang; Partho Ghose; Prem Raj; Chen-Kang Huang; Azlan Zahid
>
> **摘要:** This study presents a closed-loop robotic strawberry harvesting system that combines a robust vision module, simulation-trained deep reinforcement learning (DRL) control, and ROS-based realrobot execution. For perception, we propose HRAttnEdge-YOLO26-seg, a modified YOLO26-seg architecture that incorporates a high-resolution P2 branch, segmentation-path attention, and edgesupervised prototype learning to improve instance segmentation in cluttered scenes. For control, we train a target-conditioned Proximal Policy Optimization (PPO) policy in Isaac Lab to produce smooth joint-position commands for a UR10e manipulator and deploy it on a UR10e robot for targetfruit reaching and harvesting. This simulation-based approach reduces hardware dependency, lowers development cost, and allows scalable policy training without exhaustive physical trials before real deployment. The proposed vision model demonstrated the highest overall performance among the evaluated methods. On both self-collected and public datasets, the model showed a 10 to 14% improvement in segmentation performance. In controlled in-house tests, the PPO controller produced stable and dynamically smoother motion than a inverse kinematics (IK)-based MoveIt baseline. In greenhouse trials, the proposed integrated system harvested 281 strawberries, achieving 96.6% reaching success, 91.3% grasp-and-pull success, and 84.3% overall harvesting success. These results illustrate that task-specific perception combined with simulation-trained PPO can serve as a practical and resource-efficient alternative to conventional planner-dependent reaching in manipulation, enabling reliable closed-loop robotic harvesting in complex agricultural environments.
>
---
#### [new 017] Agentic-VLA: Efficient Online Adaptation for Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出Agentic-VLA，解决VLA模型在新环境中的泛化能力差和训练效率低的问题，通过三项创新提升在线适应效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.22896](https://arxiv.org/pdf/2605.22896)**

> **作者:** Ruofan Jin; Zaixi Zhang
>
> **备注:** Total 15 pages
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising paradigm for robotic manipulation by leveraging pre-trained vision-language representations. However, current VLA training methods suffer from two critical limitations: poor generalization to novel environments and low training efficiency requiring extensive demonstrations. We introduce Agentic-VLA, an agentic training framework that enables VLAs to efficiently adapt online through three key innovations: (1) Adaptive Reward Synthesis, which dynamically generates and adjusts reward functions based on the VLA's current capabilities and task complexity, decomposing complex tasks into learnable sub-goals for curriculum learning; (2) Language-Guided Exploration, where a critic model provides structured guidance for systematic exploration rather than random sampling; and (3) Experience Memory,which stores and retrieves task-relevant policy weights for warm-starting adaptation to similar tasks. We evaluate Agentic-VLA on the LIBERO benchmark, achieving substantial improvements: +12.3% on long-horizon tasks, +28.5% in 1-shot learning, and enabling cross-task transfer from 0% to 31.2% without task-specific demonstrations. Our framework also demonstrates 2.4x faster convergence compared to existing online adaptation methods. Beyond LIBERO, Agentic-VLA retains its advantage on the dual-arm RoboTwin 2.0 benchmark, including under its randomized Hard setting. These results establish Agentic-VLA as a significant step toward truly adaptive VLA systems capable of continuous learning in deployment.
>
---
#### [new 018] Robots That Know What to Ask: Recovering Misaligned Rewards through Targeted Explanations
- **分类: cs.RO; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于强化学习中的奖励函数学习任务，旨在解决演示数据不足导致的奖励不匹配问题。通过检测未充分说明的特征并主动请求针对性演示，提升奖励恢复效果。**

- **链接: [https://arxiv.org/pdf/2605.22986](https://arxiv.org/pdf/2605.22986)**

> **作者:** Helena Merker; Nick Walker; Andreea Bobu
>
> **摘要:** Learning reward functions from demonstrations assumes that demonstrations provide adequate supervision over all features -- or task-relevant aspects of behavior. In practice, demonstrations are often imperfect: humans may under-emphasize certain features due to cognitive load or physical difficulty, or the training regime may fail to sufficiently cover all relevant situations. In either case, important features may be underspecified, leading to ambiguity in the learned reward function and misaligned behavior at deployment. We propose a framework that detects such underspecified features and actively solicits targeted corrective demonstrations. Our key insight is that demonstrations implicitly reveal which features are well specified: features that are consistently optimized show little variation across demonstrations, while features that are underspecified vary widely. We leverage this statistical signal to infer which features may have been insufficiently demonstrated. The robot then explains which features it is uncertain about in natural language and queries for demonstrations that explicitly address the identified gaps. We evaluate our approach in a simulated tabletop manipulation domain and in a user study with a real Franka robot. Targeted, explanation-guided queries significantly improve reward recovery compared to random querying and passive data collection, reducing ambiguity that would otherwise persist in learning from imperfect demonstrations.
>
---
#### [new 019] Semantic-Aware Guided Drone Exploration for Language-Conditioned 3D Indoor Mapping
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SAGE系统，用于3D室内环境的语义引导探索，解决未知环境中高效对象发现与覆盖的问题。通过融合CLIP模型提升探索效果。**

- **链接: [https://arxiv.org/pdf/2605.23160](https://arxiv.org/pdf/2605.23160)**

> **作者:** Nitin Vegesna; Avideh Zakhor
>
> **备注:** 10 pages, 6 figures, 4 tables. To be presented at the 2nd 3D-LLM/VLA Workshop at CVPR 2026 (non-archival workshop)
>
> **摘要:** We present Semantic-Aware Guided Exploration, SAGE, a system for open-vocabulary exploration in unknown 3D indoor environments that preserves coverage-oriented behavior while allowing semantic cues to reprioritize frontier selection. Building on the FALCON volumetric explorer, SAGE integrates Contrastive Language-Image Pre-training (CLIP) via four key components: object-centric embedding storage, a temporal cache that projects recent observations onto the free-unknown boundary, object frontiers for high-similarity detections, and a unified semantic-geometric planning cost. This cost function bounds semantic reweighting influence, ensuring frontiers are prioritized without sacrificing total coverage. In Matterport3D-based simulations, SAGE outperforms FALCON and a semantic-only ablation in object discovery across map-query pairs. Compared to Finding Things in the Unknown (FTU), SAGE completes exploration 9.0 to 25.9 times faster across the nine shared map-query pairs, achieving a mean speedup of 13.7. Furthermore, SAGE achieves substantially higher volumetric throughput than FTU. Finally, we deploy SAGE in five real-world flights in two environments on a Modal AI Starling 2 quadrotor with onboard sensing and planning, and offboard CLIP inference. Comparing SAGE and FALCON, we find that while FALCON results in faster exploration and shorter mapping trajectories, SAGE outperforms FALCON in terms of object discovery.
>
---
#### [new 020] Autonomous Frontier-Based Exploration with VLM Guidance
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于自主机器人探索任务，旨在提升未知危险环境中的探索效率。通过引入VLM进行高层决策，替代传统几何启发式方法，实现更优的路径选择与地图覆盖。**

- **链接: [https://arxiv.org/pdf/2605.23165](https://arxiv.org/pdf/2605.23165)**

> **作者:** Aarush Aitha; Avideh Zakhor
>
> **备注:** 8 pages, 10 figures, CVPR 2026: 2nd Workshop on 3D-LLM/VLA: Bridging Language, Vision and Action in 3D Environments
>
> **摘要:** Autonomous robotic exploration of unknown and hazardous environments, a long-standing challenge, can be significantly improved by leveraging the advanced reasoning of Vision-Language Models (VLMs). We introduce a novel exploration pipeline where a VLM performs high-level strategic decision-making, guiding a conventional low-level robotics control stack. At decision points, the robot generates a multimodal prompt with its current map and visual imagery of potential paths, or frontiers. The VLM analyzes this prompt to select the most promising frontier, replacing simple geometric heuristics with contextual spatial reasoning. This approach, validated in simulation across six indoor environments, improves map coverage by up to 24\% over existing methods. Our pipeline is lightweight, training-free, and easily transferable to any robot with standard sensors and an internet connection.
>
---
#### [new 021] Semantically Structured Mixture-of-Experts for Compositional Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决扩散策略在多任务环境中的可扩展性问题。提出SMoDP框架，通过语义结构化的专家混合实现高效、可迁移的控制。**

- **链接: [https://arxiv.org/pdf/2605.23477](https://arxiv.org/pdf/2605.23477)**

> **作者:** Chengyu Deng; Guanqi Chen; Yizhou Chen; Zejia Liu; Zhiwen Ruan; Guanhua Chen; Jia Pan
>
> **备注:** Accepted to Robotics: Science and Systems (RSS) 2026
>
> **摘要:** Diffusion-based policies have established a new standard for precise robotic manipulation but face a critical scalability bottleneck: high-performance models are computationally expensive, while lightweight alternatives often fail to generalize across diverse multi-task environments. Mixture-of-Experts (MoE) architectures offer a promising path to efficiency by activating only a subset of parameters. However, existing MoE routing mechanisms typically rely on low-level noise or latent statistics, ignoring the compositional nature of manipulation tasks. This can fragment reusable behaviors across experts, limiting interpretability and transferability. We introduce Semantically Structured Mixture-of-Experts Diffusion Policy (SMoDP) for compositional robotic manipulation, a framework that grounds expert specialization in semantic task structure. SMoDP leverages a lightweight, inference-time skill predictor, supervised by offline annotations from Vision-Language Models (VLMs), to route action chunks to experts specialized for specific behavioral phases. To ensure robust assignment, we propose a dual contrastive alignment strategy that grounds multi-modal observations in language-defined skill semantics (Inter-modal) while enforcing routing consistency across visually distinct but functionally related behaviors (Intra-modal). Our approach outperforms representative diffusion and MoE-based baselines on multi-task benchmarks with significantly improved parameter efficiency and demonstrates effective compositional transfer to novel tasks through parameter-efficient fine-tuning. Project website: this https URL
>
---
#### [new 022] Droneulator: A Portable UAV Simulator for Agricultural Workflows with RotorPy and Godot 4
- **分类: cs.RO**

- **简介: 该论文提出Droneulator，一个用于农业无人机的便携式模拟器，解决真实场景模拟与控制集成问题。融合RotorPy与Godot 4，支持多种农业任务，如图像采集、路径规划和强化学习。**

- **链接: [https://arxiv.org/pdf/2605.23386](https://arxiv.org/pdf/2605.23386)**

> **作者:** Jacob Swindell; Michael Lowen; Marija Popovic; Riccardo Polvara
>
> **摘要:** Agricultural UAV research requires simulators that integrate realistic 3D scenes, high-fidelity vehicle dynamics, and robotics middleware, while remaining practical to deploy across heterogeneous development machines. We present Droneulator, a portable UAV simulator architecture that combines RotorPy for multirotor dynamics with Godot 4 for rendering and sensor generation. Droneulator exposes both PX4-based control and a lightweight WebSocket command path, and publishes synchronised visual and state streams through a Zenoh-based ROS~2-compatible pipeline. This integration enables a single stack to support inspection-oriented data capture, ROS~2/PX4 local planning, and reinforcement learning experiments without modifying the simulator infrastructure. We present quantified validation of the current system across three agricultural UAV workflows: tree-scale image collection for 3D reconstruction with COLMAP, local planning around canopy obstacles using EGO-Planner, and closed-loop reinforcement learning through a custom Gymnasium environment. In the reported setup, the results show that the simulator can sustain low-latency sensing, support reconstruction-oriented data collection under varying capture density, execute collision-free local planning around canopy obstacles, and support stable depth-sensing-based policy training for obstacle-aware navigation. Together, these results show the potential of Droneulator for agricultural UAV inspection, planning, and learning within one deployable stack.
>
---
#### [new 023] 6G Communication Networks Enabling Embodied Agents: Architecture and Prototype
- **分类: cs.RO; cs.AI; eess.SP; eess.SY**

- **简介: 该论文属于6G通信任务，旨在解决 embodied agents 的高要求通信问题。研究提出了一种分层架构，并通过原型验证其可行性。**

- **链接: [https://arxiv.org/pdf/2605.23263](https://arxiv.org/pdf/2605.23263)**

> **作者:** Lipeng Dai; Luping Xiang; Kun Yang
>
> **摘要:** Embodied agents, which couple intelligent decision-making with physical actuation in the real world, impose far more stringent and heterogeneous communication requirements than purely software-based agents. While 6G promises sub-millisecond latency, ultra-high reliability, native intelligence, and integrated sensing, systematic studies on how to exploit these capabilities for embodied agent communication remain limited. This article investigates 6G-enabled communication systems for embodied agents from both conceptual and engineering perspectives. First, we review the concept, embodiment value of embodied agents, and clarify their distinctions from disembodied agents. Then, we analyse the symbiotic relationship between embodied agents and 6G networks. We highlight how key 6G enablers can support the stringent requirements of human-robot interaction. Furthermore, we demonstrate the proactive role of embodied agents in bolstering communication networks through coverage extension, environmental sensing, and physical world understanding. Building on these insights, we propose a hierarchical communication architecture for human-robot remote interaction, comprising a human-intent perception layer, an open radio access network (O-RAN)-based transport layer, an intelligent intermediary layer, and an embodiment layer. To validate its feasibility, we implement an end-to-end prototype that integrates a haptic device, an industrial robotic arm, an intermediary platform, and a 5G O-RAN testbed. Experimental results demonstrate millisecond-level latency and stable closed-loop operation, confirming the practicality of the proposed architecture and providing a reference for future 6G-embodied agent research and industrial deployments.
>
---
#### [new 024] TactileReflex: Noise-Statistics-Driven Vision-Tactile Reflex Control for Force-Sensitive Manipulation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于力敏感操作任务，旨在解决脆弱容器的实时握力控制问题。通过视觉触觉反馈和噪声统计方法，设计了TactileReflex控制器，实现防滑、自适应释放和力保护。**

- **链接: [https://arxiv.org/pdf/2605.23568](https://arxiv.org/pdf/2605.23568)**

> **作者:** Ziyan Feng; Yulong Fu; Zheng Li; Yuxin He; Jieji Ren; Lujia Wang; Jinni Zhou; Yudong Zhong; Qiang Nie
>
> **备注:** 8 pages, 4 figures, 6 tables
>
> **摘要:** Manipulating fragile deformable containers, such as disposable plastic cups filled with liquid, demands real-time grip-force adaptation within an extremely narrow force margin: insufficient force causes slip, while excessive force irreversibly deforms the thin wall. Existing approaches struggle to achieve such force-sensitive manipulation tasks. We propose a noise-statistics-based calibration-driven reflex control paradigm with vision-based tactile sensing: by analyzing the sensor's intrinsic noise characteristics (via a brief static-hold-and-unload protocol), we directly derive all controller thresholds, eliminating external force calibration, trial-and-error manual tuning, or material-specific physical models. Instantiating this paradigm, we present TactileReflex, a three-channel closed-loop controller that extracts three image-level proxies, shear intensity ($S_y$), contact intensity ($F_n$), and center of pressure ($C$), from dual visuo-tactile sensors and drives prioritized reflex channels at ~12 Hz for slip suppression, weight-adaptive release, and force protection. Each channel closes the loop directly on its proxy via noise-derived thresholds. Ablation demonstrates that only the full three-channel system is able to prevent irreversible container deformation (5/5 success vs. at most 1/5 for partial configurations). In a dynamic pouring task, fixed-effort baselines fail in all 10 attempts due to pose drift, while TactileReflex achieves 9/10 success across two water volumes. As a self-contained and interpretable controller, TactileReflex can serve as a plug-and-play safety layer beneath high-level manipulation pipelines, including haptic-free VR teleoperation and vision-language-action (VLA) policies.
>
---
#### [new 025] Remote Teleoperation of Endovascular Intervention Robots: A Systematic Review
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人领域，旨在解决远程血管介入手术的技术与应用问题。通过系统综述，分析了远程操作系统的可行性、通信基础设施及临床效果，指出了研究空白并提出未来方向。**

- **链接: [https://arxiv.org/pdf/2605.22889](https://arxiv.org/pdf/2605.22889)**

> **作者:** Xingyu Chen; Yinchao Yang; Nikola Fischer; Harry Robertshaw; Benjamin Jackson; Mohammad Shikh-Bahaei; Christos Bergeles; Thomas C Booth
>
> **备注:** The manuscript has been submitted to IEEE Transaction on Medical Robotic and Bionics
>
> **摘要:** Remote robotic-assisted endovascular intervention offers a promising approach to reduce clinician radiation exposure and physical strain, while extending specialized vascular care to geographically distant regions. Despite advancements, teleoperated endovascular intervention remains underexplored, especially for time-sensitive interventions like mechanical thrombectomy for acute stroke. The aim of the current review was to determine the evidence regarding teleoperated endovascular robotic systems, covering technical feasibility, communication infrastructure, and clinical outcomes. The review further identified research gaps and future directions. Following PRISMA guidelines, 16 studies were included that met the inclusion criteria out of 2501 initial search results. We found that teleoperated catheters and guidewires, driven by mechanical or electromagnetic systems, can be navigated across distances up to 7000 km. With robust communication infrastructure, network latency remained within clinically acceptable limits (30-163 ms). Although initial outcomes highlighted 100% procedural success in small-scale human trials, most evidence stemmed from animal or phantom models. Overall, the findings suggest that teleoperated endovascular intervention can reduce occupational hazards, expand patient access to urgent procedures, and optimize resource allocation. Future research should be conducted in low and middle income countries to demonstrate broader geographical access. Ultimately, multi-center clinical trials are required to validate the safety, efficacy, and generalization in diverse clinical settings.
>
---
#### [new 026] Direct Dynamic Retargeting for Humanoid Imitation Learning from Videos
- **分类: cs.RO**

- **简介: 该论文属于机器人模仿学习任务，解决人体动作到人形机器人的动态迁移问题。提出直接动态迁移方法，避免几何偏差，生成高保真轨迹。**

- **链接: [https://arxiv.org/pdf/2605.23762](https://arxiv.org/pdf/2605.23762)**

> **作者:** Constant Roux; Ludovic De Matteïs; Armand Jordana; Valentin Guillet; Nicolas Mansard; Olivier Stasse; Philippe Souères
>
> **摘要:** Imitation Learning from monocular video demonstrations provides a scalable approach for teaching complex skills to humanoid robots. However, translating human motion to humanoids requires overcoming significant morphological mismatches. Standard approaches rely on Geometric Retargeting or Indirect Dynamic Retargeting pipelines. We identify that these intermediate kinematic projections introduce a geometric bias, restricting the search space and yielding suboptimal dynamic behaviors. In this paper, we propose Direct Dynamic Retargeting (DDR), a novel single-stage framework that generates high-fidelity, dynamically feasible trajectories directly from expert videos. By formulating the problem in the task space and leveraging a sampling-based Model Predictive Control solver within a physics simulator, DDR natively optimizes over complex contact sequences while mitigating input drift. Our experiments demonstrate that bypassing the geometric bias allows DDR to outperform state-of-the-art baselines in demonstration tracking accuracy. Furthermore, we establish that providing such physically viable references to RL agents accelerates training convergence and enhances the final execution of agile and balancing behaviors. Source code will be made publicly available.
>
---
#### [new 027] Extending Deep Event Visual Odometry with Sparse Point-Cloud Export
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉里程计任务，旨在提升DEVO系统以输出稀疏点云。通过暴露内部3D结构并转换为点云，实现可视化与后续处理。**

- **链接: [https://arxiv.org/pdf/2605.22890](https://arxiv.org/pdf/2605.22890)**

> **作者:** Alireza Safdari; Sajad Ashraf
>
> **备注:** 9 Pages, 4 figures, 5 tabel
>
> **摘要:** Event cameras are well suited for visual odometry under high-speed motion and challenging lighting conditions due to their low latency, high temporal resolution, and high dynamic range. Deep Event Visual Odometry (DEVO) demonstrated that monocular event-only odometry can achieve strong performance by combining sparse patch tracking, learned patch selection, recurrent correspondence refinement, and differentiable bundle adjustment. In this project, we extend DEVO with a sparse point-cloud export pipeline. Rather than modifying the core odometry formulation, our approach exposes the internal 3D structure already estimated by DEVO and converts it into an explicit point-cloud representation for visualization and further processing. In addition, we implement a practical workflow for data export, format conversion, and point-cloud cleanup. The resulting system preserves the original visual odometry pipeline while enabling sparse geometric scene output. Experiments on the BOARD SLOW sequence show that the exported sparse cloud is locally consistent with EMVS reconstructions, achieving high precision at a 5 cm threshold, while also highlighting the expected limitations in density, completeness, and sensitivity to accumulated odometry noise.
>
---
#### [new 028] Active Sensing Subserves Task-Level Control
- **分类: q-bio.NC; cs.LG; cs.RO; eess.SY**

- **简介: 该论文探讨主动感知在任务控制中的作用，旨在解决生物与工程系统在感知与控制上的差异。研究提出主动感知是任务控制的必要手段，通过模式切换实现高效反馈控制。**

- **链接: [https://arxiv.org/pdf/2605.22988](https://arxiv.org/pdf/2605.22988)**

> **作者:** Andrew Lamperski; Debojyoti Biswas; Eric S. Fortune; John Guckenheimer; Kathleen Hoffman; Noah J. Cowan
>
> **摘要:** Active sensing is traditionally defined as the expenditure of energy, typically in the form of movement, for obtaining information. Here, we propose that the combination of reliance on adaptive sensors, the linkage between movement and sensing, and task-level control inevitably gives rise to the emergence of active sensing movements. In this way, active sensing is not driven by sensory goals, such as minimizing uncertainty about the state, but rather is necessary for task-level control. This hypothesis, that active sensing subserves control, is supported by both empirical data from organisms and mathematical theory. Interestingly, active sensing behaviors often occur in discrete epochs, interspersed with goal-oriented behavior. This suggests that animals switch between two behavioral modes with distinct control policies, an `explore' mode in which animals produce dynamic movements to shape sensory feedback, and an `exploit' mode in which animals produce slower compensatory movements that are directly related to achieving task goals. This strategy for feedback control that relies on adaptive sensors, active sensing, and mode switching is not commonly used in engineered systems despite being ubiquitous in biology. Engineered systems comprising state-of-the-art sensors, actuators, and mechanical designs can outperform animals with respect to ``cost functions'' such as maximum force generation, precision, and speed. Nevertheless, animals routinely achieve robust, graceful behaviors that are currently unmatched by engineered systems, suggesting that current control systems are insufficient. These insights, expressed in the language of control theory, may be critical for improving robotic sensing and control.
>
---
#### [new 029] ChainFlow-VLA: Causal Flow Planning with Vision-Language Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出ChainFlow-VLA，解决自动驾驶中因果推理与全局优化不统一的问题，通过结合自回归生成和扩散模型，提升轨迹规划的准确性与安全性。**

- **链接: [https://arxiv.org/pdf/2605.23270](https://arxiv.org/pdf/2605.23270)**

> **作者:** Xiyang Wang; Xinlin Wang; Tingguang Zhou; Gong Chen; Xingtai Gui; Zhi Xu; Xiaolei Wu; Feiyang Tan; Hangning Zhou; Mu Yang
>
> **摘要:** Current end-to-end autonomous driving systems are fundamentally limited by a mismatch between temporal causal reasoning and global trajectory consistency. Autoregressive (AR) models capture interaction-aware temporal dependencies via causal factorization, but their step-wise decoding leads to error accumulation and suboptimal global structure. In contrast, diffusion models optimize trajectories globally but lack explicit causal constraints, making them unreliable in interactive and safety-critical scenarios. This dichotomy reveals a deeper issue: existing methods treat causal modeling and global optimization as separate paradigms, without a principled way to unify them within a single trajectory distribution. To address this, we propose ChainFlow-VLA, which unifies causal generation and global refinement within a unified probabilistic framework. We formulate planning as a mixture over AR-induced modes and learn Vision-Language Model (VLM)-conditioned residual distributions over these modes. An autoregressive generator (Chain) produces a discrete set of causal trajectory modes, followed by a diffusion-based refiner (Flow) that leverages VLM hidden states as semantic priors to perform mode-conditioned correction in residual space while preserving causal structure. This straightforward conditioning seamlessly injects high-level scene understanding into fine-grained trajectory adjustments. Experiments demonstrate that ChainFlow-VLA achieves robust planning in ambiguous and long-tail scenarios, achieving a state-of-the-art score of 94.85 on the NAVSIM v1 leaderboard, matching human-level performance (94.8). Code will be available at this https URL.
>
---
#### [new 030] Good Token Hunting: A Hitchhiker's Guide to Token Selection for Visual Geometry Transformers
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.RO**

- **简介: 该论文针对视觉几何Transformer的计算效率问题，提出一种分阶段的token选择策略，以降低计算成本并提升性能。**

- **链接: [https://arxiv.org/pdf/2605.23892](https://arxiv.org/pdf/2605.23892)**

> **作者:** Shuhong Zheng; Michael Oechsle; Erik Sandström; Marie-Julie Rakotosaona; Federico Tombari; Igor Gilitschenski
>
> **备注:** Project Page: this https URL, Code: this https URL
>
> **摘要:** Visual geometry transformers have become powerful architectures for multi-view 3D reconstruction, enabling joint prediction of multiple 3D attributes in a feed-forward manner. However, their computational cost grows quadratically with the input sequence length due to the global attention layers inside these models. This limits both their scalability and efficiency. In this work, we address this challenge with a simple yet general strategy: restricting the number of key/value tokens that each query interacts with during global attention. To achieve effective token selection, we introduce a two-stage framework. First, an inter-frame selection step operates at the frame level to identify frames that should be preserved. Second, an intra-frame selection step further discards more redundant tokens within the selected frames. Our analysis highlights the advantage of a diversity-based strategy for inter-frame selection, which ensures broad coverage of the scene. For intra-frame selection, we show that layer-aware sparsification is necessary, with the selection process guided by the entropy of the global attention pattern. Our approach offers a superior speed-accuracy trade-off compared to existing solutions. Extensive experiments show that it accelerates visual geometry transformers by over 85% for scenes with 500 images while maintaining, or even improving, baseline performance, which hints that how our token selection strategy can play a crucial role in future applications of visual geometry transformers. Our project website is available at this https URL.
>
---
#### [new 031] Lipschitz Optimization for Formal Verification of Homographies
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于形式化验证任务，解决视觉神经网络在相机运动下的鲁棒性问题。通过Lipschitz优化，建立相机位姿与像素值的映射，实现对投影几何变换的正式验证。**

- **链接: [https://arxiv.org/pdf/2605.23203](https://arxiv.org/pdf/2605.23203)**

> **作者:** Jean-Guillaume Durand; Panagiotis Kouvaros; Maxime Gariel; Alessio Lomuscio
>
> **备注:** 18 pages, 13 figures, 6 tables, to be published at CVPR 2026
>
> **摘要:** The adoption of vision neural networks in regulated industries requires formal robustness guarantees, especially in safety-critical domains such as healthcare, autonomous vehicles, and aerospace. However, current approaches are confined to incomplete statistical verification or robustness to $\ell_p$-norm and affine transforms, which cover only a narrow subset of perturbations to the image formation process. In particular, robustness to camera motion remains an open problem despite being key to deploy many vision applications. We present a formal verification approach that targets robustness against 3D motion perturbations of the capturing camera. We first establish a closed-form mapping from camera pose to pixel values. By analyzing the continuity properties of the resulting homographies, we show that recent work on Lipschitz optimization and piecewise continuity can be extended to derive tight linear bounds on perturbed pixel values. Our approach applies to scenes with predominantly planar structure, such as ground planes in augmented reality, road markings and traffic signs in autonomous driving, or planar workspaces in robotic manipulation. This enables the first formal verification of projective geometry transforms, without complex simulation, surrogate networks, or explicit image-formation models. We validate our implementation and show up to 89% speedup and 7% tighter bounds over prior work. We then evaluate our method on the VNN-COMP benchmark and reveal systematic weaknesses to projective perturbations. Finally, we demonstrate a real-world case study on a safety-critical runway classifier, highlighting practical vulnerabilities to camera motion, and addressing a key challenge in the certification of learned models. Data and code are publicly available at this https URL .
>
---
#### [new 032] GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GEM-4D，解决视频生成中点级运动不一致问题，提升机器人操作的物理可靠性。属于视频世界模型任务，通过几何监督和逆动力学模块实现更准确的视频预测与轨迹生成。**

- **链接: [https://arxiv.org/pdf/2605.22882](https://arxiv.org/pdf/2605.22882)**

> **作者:** Kaichen Zhou; Yuzhen Chen; Fangneng Zhan; Hang Hua; Grace Chen; Xinhai Chang; Ao Qu; Yilun Du; Zhuang Liu; Paul Pu Liang; Mengyu Wang
>
> **备注:** Robotic World Model, Video Generative Model
>
> **摘要:** Video world models can generate realistic futures from a single instruction, but they often fail to preserve consistent point-level motion over time. As a result, the generated videos appear plausible, yet lack the physical grounding required for reliable action execution, such as robot manipulation. We present GEM-4D, a geometry-grounded video world model that resolves this limitation by injecting dense 4D correspondence supervision, distilled from a pretrained geometry foundation model, into the video generative backbone during training. This supervision enables the model to jointly capture appearance and geometric structure while retaining a single-stream architecture with no additional inference cost. We further introduce an inverse dynamics module that converts correspondence-consistent video rollouts into executable robot trajectories, enabling direct deployment in both real-world and simulated manipulation. GEM-4D achieves state-of-the-art performance on both video prediction and geometric consistency across simulation and realistic scenarios and improves real-world manipulation success from 61% to 81%. Additional results are available at the project page: this https URL.
>
---
#### [new 033] SCRIPT: Scalable Diffusion Policy with Multi-stage Training for Language-driven Physics-Based Humanoid Control
- **分类: cs.GR; cs.LG; cs.RO**

- **简介: 该论文属于语言驱动的物理人形控制任务，解决语义表达与物理可行性之间的矛盾。提出SCRIPT框架，结合多阶段训练和扩散模型，提升指令遵循、运动质量和控制稳定性。**

- **链接: [https://arxiv.org/pdf/2605.22894](https://arxiv.org/pdf/2605.22894)**

> **作者:** Jingyan Zhang; Han Liang; Ruichi Zhang; Bin Li; Juze Zhang; Xin Chen; Jingya Wang; Lan Xu; Jingyi Yu
>
> **备注:** Project page: this https URL
>
> **摘要:** Controlling physics-based humanoids from natural-language instructions is a critical step toward general-purpose embodied agents. However, existing methods remain constrained by a tension between semantic expressiveness and physical feasibility, often failing to jointly achieve faithful instruction following, high-quality motion, and stable long-horizon control. We propose SCRIPT, a scalable diffusion policy with a multi-stage training framework for language-driven physics-based humanoid control. The core of SCRIPT is a Joint Action-State-Text Diffusion Transformer (JAST-DiT), which represents actions, physical states, and text as dedicated token streams and couples them through joint attention, enabling direct interaction between language semantics and control dynamics. To stabilize autoregressive control, we introduce a nonlinear history conditioning mechanism, which preserves the dense recent context and samples increasingly sparse cues from long-term history. Beyond supervised imitation pre-training, we propose a post-training stage, further improving the performance using Reinforcement Learning with Hybrid Rewards (RLHR). By injecting learnable noise into the flow-sampling process, RLHR effectively improves motion quality and instruction following within closed-loop simulations using hybrid physical feedback and text rewards. Quantitative evaluations demonstrate that SCRIPT outperforms prior state-of-the-art methods, with gains across text alignment, motion quality, and physical realism metrics. Furthermore, scaling studies on the 1200-hour MotionMillion dataset demonstrate consistent performance gains with model scaling, highlighting SCRIPT's robust scalability for large-scale pre-training. Our code will be publicly available for future research.
>
---
#### [new 034] IntentionNav: A Benchmark for Intent-Driven Object Navigation from Implicit Human Instruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出IntentionNav基准，解决从隐式指令中进行意图驱动的物体导航问题，通过分析语言理解、目标推理和定位效果，评估模型在复杂场景中的表现。**

- **链接: [https://arxiv.org/pdf/2605.23187](https://arxiv.org/pdf/2605.23187)**

> **作者:** Lin Qian; Shijie Li; Sihao Lin; Xuan Zhang; Bangya Liu; Yanran Li; Hujun Yin
>
> **备注:** preprint
>
> **摘要:** Existing object navigation benchmarks usually tell an embodied agent which object category to find, such as microwave or chair. Human-facing embodied AI is often asked something less direct: "I need something to warm this food" or "the room feels stuffy." The agent must infer the object that can satisfy the need, find a scene-grounded instance, and decide whether the goal has been reached. We study this setting as intent-driven object navigation and introduce IntentionNav, a diagnostic benchmark for active object search from implicit human instructions. Each episode provides a free-text intent, RGB-D observations, and pose, but withholds the target object name. IntentionNav contains 500 intents over 176 Isaac Sim scenes and 64 target categories. Each intent is rewritten in four controlled instruction styles and annotated with one of four intent modes, separating surface phrasing from semantic cue type under matched geometry. This paired design supports analysis of target inference, language robustness, neighborhood reachability, and terminal success rather than only aggregate success. We evaluated three VLMs using a fixed active-navigation agent. Models identify the intended target in 48.3 percent of episodes and enter its 2 m neighborhood in 68.7 percent, but terminate successfully in only 24.9 percent and achieve grounded 1 m success in 5.5 percent. Success is highest for event-script intents (28.7 percent) and lower for physical-state and affordance intents (19.2 percent and 18.5 percent), showing that indirect human intent remains a bottleneck for target selection, visual verification, and terminal localization in active embodied search.
>
---
## 更新

#### [replaced 001] Data-driven Spatial Classification using Multi-Arm Bandits for Monitoring with Energy-Constrained Mobile Robots
- **分类: cs.RO**

- **简介: 该论文研究基于多臂老虎机的数据驱动空间分类任务，解决能源受限移动机器人监测中的区域快速分类问题，提出双层策略优化路径规划与区域选择。**

- **链接: [https://arxiv.org/pdf/2501.08222](https://arxiv.org/pdf/2501.08222)**

> **作者:** Xiaoshan Lin; Siddharth Nayak; Stefano Di Cairano; Abraham P. Vinod
>
> **备注:** 8 pages, 6 figures. See this https URL for an overview of the approach along with videos of the hardware experiments
>
> **摘要:** We consider the spatial classification problem for monitoring using data collected by a coordinated team of mobile robots. Such classification problems arise in several applications including search-and-rescue and precision agriculture. Specifically, we want to classify the regions of a search environment into interesting and uninteresting as quickly as possible using a team of mobile sensors and mobile charging stations. We develop a data-driven strategy that accommodates the noise in sensed data and the limited energy capacity of the sensors, and generates collision-free motion plans for the team. We propose a bi-level approach, where a high-level planner leverages a multi-armed bandit framework to determine the potential regions of interest for the drones to visit next based on the data collected online. Then, a low-level path planner based on integer programming coordinates the paths for the team to visit the determined regions subject to the physical constraints. We characterize several theoretical properties of the proposed approach, including anytime guarantees and task completion time. We show the efficacy of our approach in simulation, and further validate these observations in physical experiments using mobile robots.
>
---
#### [replaced 002] Safe and Energy-Aware Multi-Robot Density Control via PDE-Constrained Optimization for Long-Duration Autonomy
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于多机器人系统控制任务，解决长期自主下的密度控制问题。通过PDE约束优化，实现安全和节能的密度跟踪与避障。**

- **链接: [https://arxiv.org/pdf/2604.15524](https://arxiv.org/pdf/2604.15524)**

> **作者:** Longchen Niu; Andrew Nasif; Gennaro Notomista
>
> **摘要:** This paper presents a novel density control framework for multi-robot systems with spatial safety and energy sustainability guarantees. Stochastic robot motion is encoded through the Fokker-Planck Partial Differential Equation (PDE) at the density level. Control Lyapunov and control barrier functions are integrated with PDEs to enforce target density tracking, obstacle region avoidance, and energy sufficiency over multiple charging cycles. The resulting quadratic program enables fast in-the-loop implementation that adjusts commands in real-time. Multi-robot experiment and extensive simulations were conducted to demonstrate the effectiveness of the controller under localization and motion uncertainties.
>
---
#### [replaced 003] Towards Trustworthy and Explainable AI for Perception Models: From Concept to Prototype Vehicle Deployment
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶感知任务，旨在解决深度神经网络的不透明性问题。提出一个可信赖的AI感知模块，集成可解释性和不确定性估计，并在原型车中部署验证。**

- **链接: [https://arxiv.org/pdf/2605.16087](https://arxiv.org/pdf/2605.16087)**

> **作者:** Till Beemelmanns; Shayan Sharifi; Manas Mehrotra; Ayushman Choudhuri; Lutz Eckstein
>
> **备注:** Accepted for publication at IEEE ITSC 2026
>
> **摘要:** Deep Neural Networks have become the dominant solution for Autonomous Driving perception, but their opacity conflicts with emerging Trustworthy AI guidelines and complicates safety assurance, debugging, and human oversight. While theoretical frameworks for safe and Explainable AI (XAI) exist, concrete implementations of Trustworthy AI for 3D scene understanding remain scarce. We address this gap by proposing a Trustworthy AI perception module that is remarkably robust, integrates faithful explainability, and calibrated uncertainty estimates. Building on a transformer-based detector, we derive explanation from the attention mechanism at inference time and validate their faithfulness using perturbation-based consistency tests. We further integrate an uncertainty estimation and calibration module, and apply robustness-enhancing training methods. Experiments show faithful saliency behavior, improved robustness, and well-calibrated uncertainty estimates. Finally, we deploy these Trustworthy AI elements in a prototype vehicle and provide an XAI Interface that visualizes documentation artifacts, model uncertainty state, and saliency maps, demonstrating the feasibility of trustworthy perception monitoring in real time. Supplementary materials are available at this https URL .
>
---
#### [replaced 004] Encirclement Guaranteed Finite-Time Capture against Unknown Evader Strategies
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于多智能体协同控制任务，解决在未知逃逸策略下实现有限时间围捕问题。提出策略确保围捕同时保持逃逸者被包围，计算捕获时间上限。**

- **链接: [https://arxiv.org/pdf/2603.15278](https://arxiv.org/pdf/2603.15278)**

> **作者:** Dinesh Patra; Prajakta Surve; Ashish R. Hota; Shaunak D. Bopardikar
>
> **摘要:** We consider a pursuit-evasion scenario involving a group of pursuers and a single evader in a two-dimensional unbounded environment. The pursuers aim to capture the evader in finite time while ensuring the evader remains enclosed within the convex hull of their positions until capture, without knowledge of the evader's heading angle. Prior works have addressed the problem of encirclement and capture separately in different contexts. In this paper, we present a class of strategies for the pursuers that guarantee capture in finite time while maintaining encirclement, irrespective of the evader's strategy. Furthermore, we derive an upper bound on the time to capture. Numerical results highlight the effectiveness of the proposed framework against a range of evader strategies.
>
---
#### [replaced 005] Optimal Solutions for the Moving Target Vehicle Routing Problem with Obstacles via Lazy Branch and Price
- **分类: cs.RO**

- **简介: 该论文研究MT-VRP-O任务，解决多代理在障碍物环境中最优路径规划问题。提出Lazy BPRC方法，通过延迟成本计算提升求解效率。**

- **链接: [https://arxiv.org/pdf/2603.21880](https://arxiv.org/pdf/2603.21880)**

> **作者:** Anoop Bhat; Geordan Gutow; Surya Singh; Zhongqiang Ren; Sivakumar Rathinam; Howie Choset
>
> **摘要:** The Moving Target Vehicle Routing Problem with Obstacles (MT-VRP-O) seeks trajectories for several agents that collectively intercept a set of moving targets. Each target has one or more time windows where it must be visited, and the agents must avoid static obstacles and satisfy speed and capacity constraints. We introduce Lazy Branch-and-Price with Relaxed Continuity (Lazy BPRC), which finds optimal solutions for the MT-VRP-O. Lazy BPRC applies the branch-and-price framework for VRPs, which alternates between a restricted master problem (RMP) and a pricing problem. The RMP aims to select a sequence of target-time window pairings (called a tour) for each agent to follow, from a limited subset of tours. The pricing problem adds tours to the limited subset. Conventionally, solving the RMP requires computing the cost for an agent to follow each tour in the limited subset. Computing these costs in the MT-VRP-O is computationally intensive, since it requires collision-free motion planning between moving targets. Lazy BPRC defers cost computations by solving the RMP using lower bounds on the costs of each tour, computed via motion planning with relaxed continuity constraints. We lazily evaluate the true costs of tours as-needed. We compute a tour's cost by searching for a shortest path on a Graph of Convex Sets (GCS), and we accelerate this search using our continuity relaxation method. We demonstrate that Lazy BPRC runs up to an order of magnitude faster than two ablations.
>
---
#### [replaced 006] CarlaNCAP: A Framework for Quantifying the Safety of Vulnerable Road Users in Infrastructure-Assisted Collective Perception Using EuroNCAP Scenarios
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶安全评估任务，旨在解决VRUs在城市环境中的安全问题。通过构建CarlaNCAP数据集，评估基础设施辅助集体感知对提升VRU安全的效果。**

- **链接: [https://arxiv.org/pdf/2512.11551](https://arxiv.org/pdf/2512.11551)**

> **作者:** Jörg Gamerdinger; Sven Teufel; Simon Roller; Oliver Bringmann
>
> **摘要:** The growing number of road users has significantly increased the risk of accidents in recent years. Vulnerable Road Users (VRUs) are particularly at risk, especially in urban environments where they are often occluded by parked vehicles or buildings. Autonomous Driving (AD) and Collective Perception (CP) are promising solutions to mitigate these risks. In particular, infrastructure-assisted CP, where sensor units are mounted on infrastructure elements such as traffic lights or lamp posts, can help overcome perceptual limitations by providing enhanced points of view, which significantly reduces occlusions. To encourage decision makers to adopt this technology, comprehensive studies and datasets demonstrating safety improvements for VRUs are essential. In this paper, we propose a framework for evaluating the safety improvement by infrastructure-based CP specifically targeted at VRUs including a dataset with safety-critical EuroNCAP scenarios (CarlaNCAP) with 11k frames. Using this dataset, we conduct an in-depth simulation study and demonstrate that infrastructure-assisted CP can significantly reduce accident rates in safety-critical scenarios, achieving up to 100% accident avoidance compared to a vehicle equipped with sensors with only 33%. Code is available at this https URL
>
---
#### [replaced 007] Dream-MPC: Gradient-Based Model Predictive Control with Latent Imagination
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出Dream-MPC，解决连续控制任务中的高效策略优化问题。通过结合模型预测控制与梯度上升，提升策略性能并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2605.04568](https://arxiv.org/pdf/2605.04568)**

> **作者:** Jonathan Spieler; Sven Behnke
>
> **备注:** Accepted for International Conference on Machine Learning (ICML) 2026
>
> **摘要:** State-of-the-art model-based Reinforcement Learning (RL) approaches either use gradient-free, population-based methods for planning, learned policy networks, or a combination of policy networks and planning. Hybrid approaches that combine Model Predictive Control (MPC) with a learned model and a policy prior to leverage the advantages of both paradigms have shown promising results. However, these approaches typically rely on gradient-free optimization methods, which can be computationally expensive for high-dimensional control tasks. While gradient-based methods are a promising alternative, recent works have empirically shown that gradient-based methods often perform worse than their gradient-free counterparts. We propose Dream-MPC, a novel approach that generates few candidate trajectories from a rolled-out policy and optimizes each trajectory by gradient ascent using a learned world model, uncertainty regularization and amortization of optimization iterations over time by reusing previously optimized actions. Our results on 24 continuous control tasks show that Dream-MPC can significantly improve the performance of the underlying policy and can outperform gradient-free MPC and state-of-the-art baselines. Code and videos are available at this https URL.
>
---
#### [replaced 008] Adapting Dijkstra for Buffers and Unlimited Transfers
- **分类: cs.DS; cs.AI; cs.RO**

- **简介: 该论文属于公共交通路径规划任务，解决无限换乘下的最优路径问题。通过改进Dijkstra算法，提出TAD方法，在考虑缓冲时间的情况下提升效率和准确性。**

- **链接: [https://arxiv.org/pdf/2603.11729](https://arxiv.org/pdf/2603.11729)**

> **作者:** Denys Katkalo; Andrii Rohovyi; Toby Walsh
>
> **备注:** v4: clarified RAPTOR description in the Background section
>
> **摘要:** In recent years, RAPTOR based algorithms have been considered the state-of-the-art for path-finding with unlimited transfers without preprocessing. However, this status largely stems from the evolution of routing research, where Dijkstra-based solutions were superseded by timetable-based algorithms without a systematic comparison. In this work, we revisit classical Dijkstra-based approaches for public transit routing with unlimited transfers and demonstrate that Time-Dependent Dijkstra (TD-Dijkstra) outperforms MR. However, efficient TD-Dijkstra implementations rely on filtering dominated connections during preprocessing, which assumes passengers can always switch to a faster connection. We show that this filtering is unsound when stops have buffer times, as it cannot distinguish between seated passengers who may continue without waiting and transferring passengers who must respect the buffer. To address this limitation, we introduce Transfer Aware Dijkstra (TAD), a modification that scans entire trip sequences rather than individual edges, correctly handling buffer times while maintaining performance advantages over MR. Our experiments on London and Switzerland networks show that we can achieve a greater than two time speed-up over MR while producing optimal results on both networks with and without buffer times.
>
---
#### [replaced 009] GAF: Gaussian Action Field as a 4D Representation for Dynamic World Modeling in Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决动态场景建模与动作精准性问题。提出GAF方法，通过4D高斯动作场实现更准确的场景重建和动作预测。**

- **链接: [https://arxiv.org/pdf/2506.14135](https://arxiv.org/pdf/2506.14135)**

> **作者:** Ying Chai; Litao Deng; Ruizhi Shao; Jiajun Zhang; Kangchen Lv; Liangjun Xing; Xiang Li; Hongwen Zhang; Yebin Liu
>
> **备注:** this https URL
>
> **摘要:** Accurate scene perception is critical for vision-based robotic manipulation. Existing approaches typically follow either a Vision-to-Action (V-A) paradigm, predicting actions directly from visual inputs, or a Vision-to-3D-to-Action (V-3D-A) paradigm, leveraging intermediate 3D representations. However, these methods often struggle with action inaccuracies due to the complexity and dynamic nature of manipulation scenes. In this paper, we adopt a V-4D-A framework that enables direct action reasoning from motion-aware 4D representations via a Gaussian Action Field (GAF). GAF extends 3D Gaussian Splatting (3DGS) by incorporating learnable motion attributes, allowing 4D modeling of dynamic scenes and manipulation actions. To learn time-varying scene geometry and action-aware robot motion, GAF provides three interrelated outputs: reconstruction of the current scene, prediction of future frames, and estimation of init action via Gaussian motion. Furthermore, we employ an action-vision-aligned denoising framework, conditioned on a unified representation that combines the init action and the Gaussian perception, both generated by the GAF, to further obtain more precise actions. Extensive experiments demonstrate significant improvements, with GAF achieving +11.5385 dB PSNR, +0.3864 SSIM and -0.5574 LPIPS improvements in reconstruction quality, while boosting the average +7.3% success rate in robotic manipulation tasks over state-of-the-art methods.
>
---
#### [replaced 010] V-VLAPS: Value-Guided Planning for Vision-Language-Action Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，解决VLA模型在长序列任务中因策略偏差导致的规划问题。通过引入价值引导的树搜索方法，提升规划效果。**

- **链接: [https://arxiv.org/pdf/2601.00969](https://arxiv.org/pdf/2601.00969)**

> **作者:** Ke Ren; Ali Salamatian; Kieran Pattison; Cyrus Neary
>
> **摘要:** Vision-language-action (VLA) models provide strong action priors for robotic manipulation, but their reactive behavior can fail under distribution shift and long-horizon task structure. Recent VLA-guided planning methods improve execution by using pretrained policies to guide tree search, yet node selection still depends heavily on policy priors and visit-count exploration. Consequently, when the policy favors poor actions, the planner lacks a learned value signal to correct this bias. Prior work has shown that VLA representations encode rollout success and failure information, suggesting that they may also support value estimation during planning. We introduce Value-Guided Vision-Language-Action Planning and Search (V-VLAPS), which augments VLA-guided planning with a lightweight value head trained on offline VLA rollouts to predict Monte Carlo returns. These predictions guide Monte Carlo Tree Search toward higher-value branches. Across five LIBERO suites, V-VLAPS matches value-free planning baseline at the default search budget in aggregate, and analysis shows that many hard failures are root-level timeouts where predicted values are weakly separated. With a larger search budget, V-VLAPS improves over the baseline in all task suites with +6 percentage points on LIBERO-Object and +4 percentage points on LIBERO-10. Our results suggest that VLA representations can support not only failure prediction, but also value-guided planning when search reaches branches where value-based ranking matters.
>
---
#### [replaced 011] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于在线高精地图构建任务，旨在减少标注依赖。通过地理对比学习增强特征表示，提升地图感知性能。**

- **链接: [https://arxiv.org/pdf/2603.10688](https://arxiv.org/pdf/2603.10688)**

> **作者:** Jonas Merkert; Alexander Blumberg; Jan-Hendrik Pauls; Christoph Stiller
>
> **摘要:** Autonomous vehicles rely on map information to understand the world around them. However, the creation and maintenance of offline high-definition (HD) maps remains costly. A more scalable alternative lies in online HD map construction, which only requires map annotations at training time. To further reduce the need for annotating vast training labels, self-supervised training provides an alternative. This work focuses on improving the latent birds-eye-view (BEV) feature grid representation within a vectorized online HD map construction model by enforcing geospatial consistency between overlapping BEV feature grids as part of a contrastive loss function. To ensure geospatial overlap for contrastive pairs, we introduce an approach to analyze the overlap between traversals within a given dataset and generate subsidiary dataset splits following adjustable multi-traversal requirements. We train the same model supervised using a reduced set of single-traversal labeled data and self-supervised on a broader unlabeled set of data following our multi-traversal requirements, effectively implementing a semi-supervised approach. Our approach outperforms the supervised baseline across the board, both quantitatively in terms of the downstream tasks vectorized map perception performance and qualitatively in terms of segmentation in the principal component analysis (PCA) visualization of the BEV feature space.
>
---
#### [replaced 012] Neural Configuration-Space Barriers for Manipulation Planning and Control
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于机器人运动规划与控制任务，解决高维机械臂在复杂动态环境中的安全高效路径规划问题。通过神经网络构建配置空间障碍，提升规划效率与控制鲁棒性。**

- **链接: [https://arxiv.org/pdf/2503.04929](https://arxiv.org/pdf/2503.04929)**

> **作者:** Kehan Long; Ki Myung Brian Lee; Nikola Raicevic; Niyas Attasseri; Melvin Leok; Nikolay Atanasov
>
> **摘要:** Planning and control for high-dimensional robot manipulators in cluttered dynamic environments require computational efficiency and robust safety guarantees. Inspired by recent advances in learning configuration-space distance functions (CDFs) as representations of robot bodies, we propose a unified approach for motion planning and control that formulates safety constraints as CDF barriers. A CDF barrier approximates the local free configuration space, substantially reducing the number of collision-checking operations during motion planning. However, learning a CDF barrier with a neural network and relying on online sensor observations introduces uncertainties that must be considered during control synthesis. To address this, we develop a distributionally robust CDF barrier formulation for control that accounts for modeling errors and sensor noise without assuming a known underlying distribution. Simulations and hardware experiments on a UFactory xArm6 manipulator show that our neural CDF barrier formulation enables efficient planning and robust safe control in cluttered and dynamic environments, relying only on onboard point-cloud observations.
>
---
#### [replaced 013] X-TRACK: Physics-Aware xLSTM for Realistic Vehicle Trajectory Prediction
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于车辆轨迹预测任务，旨在解决传统LSTM模型在处理长期依赖和物理可行性上的不足。提出X-TRACK模型，融合运动学约束，提升轨迹预测的准确性与合理性。**

- **链接: [https://arxiv.org/pdf/2511.00266](https://arxiv.org/pdf/2511.00266)**

> **作者:** Aanchal Rajesh Chugh; Marion Neumeier; Sebastian Dorn
>
> **摘要:** Accurate trajectory prediction is crucial for safe and reliable autonomous driving systems, requiring models that capture long-term temporal dependencies while accounting for social interactions among neighboring vehicles in highway driving scenarios. While Long Short Term Memory (LSTM) networks have been widely used in the domain of trajectory prediction, they have limitations such as limited memory capacity and scalar cell state. The recently introduced Extended Long Short Term Memory (xLSTM) addresses these limitations of traditional LSTMs by introducing exponential gating and enhanced memory structures, making them better suited for modeling long-term temporal dependencies. Despite their potential, xLSTM-based models remain underexplored in the context of vehicle trajectory prediction. This paper introduces a novel xLSTM-based highway trajectory prediction framework, X-TRAJ, as the first application of xLSTM, and its physics-aware variant, X-TRACK (eXtended LSTM for TRAjectory prediction Constraint by Kinematics), which explicitly integrates vehicle motion kinematics into the model learning process. By introducing physical constraints, the proposed model generates realistic and feasible highway trajectories. A comprehensive evaluation on the publicly available highway datasets, highD and NGSIM, demonstrates that X-TRACK outperforms state-of-the-art baselines on highD and is among the state-of-the-art models on the NGSIM dataset.
>
---
#### [replaced 014] LACY: A Vision-Language Model-based Language-Action Cycle for Self-Improving Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出LACY框架，解决机器人操作中语言与动作的双向映射问题，提升任务成功率和泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.02239](https://arxiv.org/pdf/2511.02239)**

> **作者:** Youngjin Hong; Houjian Yu; Mingen Li; Changhyun Choi
>
> **备注:** Accepted to ICRA 2026. Project page: this https URL
>
> **摘要:** Learning generalizable policies for robotic manipulation increasingly relies on large-scale models that map language instructions to actions (L2A). However, this one-way paradigm often produces policies that execute tasks without deeper contextual understanding, limiting their ability to generalize or explain their behavior. We argue that the complementary skill of mapping actions back to language (A2L) is essential for developing more holistic grounding. An agent capable of both acting and explaining its actions can form richer internal representations and unlock new paradigms for self-supervised learning. We introduce LACY (Language-Action Cycle), a unified framework that learns such bidirectional mappings within a single vision-language model. LACY is jointly trained on three synergistic tasks: generating parameterized actions from language (L2A), explaining observed actions in language (A2L), and verifying semantic consistency between two language descriptions (L2C). This enables a self-improving cycle that autonomously generates and filters new training data through an active augmentation strategy targeting low-confidence cases, thereby improving the model without additional human labels. Experiments on pick-and-place tasks in both simulation and the real world show that LACY improves task success rates by 56.46% on average and yields more robust language-action grounding for robotic manipulation. Project page: this https URL
>
---
#### [replaced 015] Imagine2Real: Towards Zero-shot Humanoid-Object Interaction via Video Generative Priors
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人形机器人与物体交互任务，解决3D数据稀缺和迁移复杂问题。提出Imagine2Real框架，通过4D点轨迹和关键点追踪实现零样本物理部署。**

- **链接: [https://arxiv.org/pdf/2605.22272](https://arxiv.org/pdf/2605.22272)**

> **作者:** Jiahe Chen; ZiRui Wang; Feiyu Jia; Xiao Chen; Xiaojie Niu; Weishuai Zeng; Tianfan Xue; Xiaowei Zhou; Jiangmiao Pang; Jingbo Wang
>
> **摘要:** Whole-body Humanoid-Object Interaction (HOI) is bottlenecked by the scarcity of high-fidelity 3D data. While video generative priors offer a promising alternative, existing methods suffer from \textit{Representation Misalignment} due to their reliance on geometric priors (e.g., explicit CAD models), and \textit{Retargeting Complexity} arising from intensive morphing and morphological mismatch. We propose Imagine2Real, a zero-shot HOI framework for flexible, geometry-free interaction. To resolve misalignment, we formulate robot and object motions as unified 4D point trajectories. To overcome retargeting complexity, our Keypoints Tracker tracks only sparse critical points (base, hands, and object), entirely bypassing the error-amplifying retargeting process. To maintain natural gaits despite these sparse signals, we utilize the latent space of a Behavior Foundation Model (BFM) as the tracker's search domain. Using a progressive training strategy, Imagine2Real learns robust behaviors with simple tracking rewards, enabling zero-shot physical deployment within a motion capture(mocap) system.
>
---
#### [replaced 016] Using Ensemble Diffusion to Estimate Uncertainty for End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决轨迹规划中的不确定性问题。提出EnDfuser系统，利用扩散模型生成多候选轨迹，提升决策的不确定性感知能力。**

- **链接: [https://arxiv.org/pdf/2506.00560](https://arxiv.org/pdf/2506.00560)**

> **作者:** Florian Wintel; Sigmund H. Høeg; Gabriel Kiss; Frank Lindseth
>
> **备注:** Accepted at NLDL 2026
>
> **摘要:** End-to-end planning systems for autonomous driving are rapidly improving, especially in closed-loop simulation environments like CARLA. Many such driving systems either do not consider uncertainty as part of the plan itself or obtain it by using specialized representations that do not generalize. In this paper, we propose EnDfuser, an end-to-end driving system that uses a diffusion model as the trajectory planner. EnDfuser effectively leverages complex perception information like fused camera and LiDAR features, through combining attention pooling and trajectory planning into a single diffusion transformer module. Instead of committing to a single plan, EnDfuser produces a distribution of candidate trajectories (128 for our case) from a single perception frame through ensemble diffusion. By observing the full set of candidate trajectories, EnDfuser provides interpretability for uncertain, multimodal future trajectory spaces. Using this information we design a simplistic safety-rule that improves the system's driving score by 1.7% on the LAV benchmark. Our findings suggest that ensemble diffusion, used as a drop-in replacement for traditional point-estimate trajectory planning modules, can contribute to an uncertainty-aware decision making process in End-to-End driving policies by modeling the uncertainty of the posterior trajectory distribution.
>
---
#### [replaced 017] SEG-JPEG: Simple Visual Semantic Communications for Remote Operation of Automated Vehicles over Unreliable Wireless Networks
- **分类: cs.RO**

- **简介: 该论文属于视觉语义通信任务，旨在解决远程操控自动驾驶车辆时网络不稳定导致的图像传输问题。通过语义分割压缩，降低数据率并保持清晰度，提升远程操作效率。**

- **链接: [https://arxiv.org/pdf/2602.15258](https://arxiv.org/pdf/2602.15258)**

> **作者:** Sebastian Donnelly; Ruth Anderson; George Economides; James Broughton; Peter Ball; Alexander Rast; Andrew Bradley
>
> **备注:** 7 pages, 9 figures. Under minor revision for CSNDSP 2026
>
> **摘要:** Remote Operation is touted as being key to the rapid deployment of automated vehicles. Streaming imagery to control connected vehicles remotely currently requires a reliable, high throughput network connection, which can be limited in real-world remote operation deployments relying on public network infrastructure. This paper investigates how the application of computer vision assisted semantic communication can be used to circumvent data loss and corruption associated with traditional image compression techniques. By encoding the segmentations of detected road users into colour coded highlights within low resolution greyscale imagery, the required data rate can be reduced by 50% compared with conventional techniques, while maintaining visual clarity. This enables a median glass-to-glass latency of below 200 ms even when the network data rate is below 500 kbit/s, while clearly outlining salient road users to enhance situational awareness of the remote operator. The approach is demonstrated in an area of variable 4G mobile connectivity using an automated last-mile delivery vehicle. Results indicate that large-scale deployment of remotely operated automated vehicles could be possible even on the often constrained public 4G/5G mobile network, providing the potential to expedite the nationwide roll-out of automated vehicles.
>
---
#### [replaced 018] Investigating Robot Control Policy Learning for Autonomous X-ray-guided Spine Procedures
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于机器人控制策略学习任务，旨在解决X射线引导下脊柱手术的自主操作问题。通过仿真与数据训练，探索模仿学习在稀疏输入场景中的应用与挑战。**

- **链接: [https://arxiv.org/pdf/2511.03882](https://arxiv.org/pdf/2511.03882)**

> **作者:** Florence Klitzner; Blanca Inigo; Benjamin D. Killeen; Lalithkumar Seenivasan; Michelle Song; Axel Krieger; Mathias Unberath
>
> **摘要:** Imitation learning-based robot control policies are enjoying renewed interest in video-based robotics. However, it remains unclear whether this approach applies to X-ray-guided procedures, such as spine instrumentation, with sparse inputs. We examine the feasibility, opportunities and challenges for imitation policy learning in bi-plane-guided cannula insertion. We develop an in silico sandbox for scalable, automated simulation of X-ray-guided spine procedures with a high degree of realism. We curate a dataset of correct trajectories and corresponding bi-planar X-ray sequences that emulate the stepwise alignment of providers. We then train imitation learning policies for planning and open-loop control that iteratively align a cannula in a vertebroplasty setting solely based on visual information. This precisely controlled setup offers insights into limitations and capabilities of this method. Our policy succeeded on the first attempt in 68.5% of cases, maintaining safe intra-pedicular trajectories across diverse vertebral levels. The policy transferred to complex anatomy, including fractures, as well as varied anatomies and initializations. Rollouts on real X-ray indicate that partial sim-to-real transfer with plausible trajectories is possible. While these preliminary results are promising, we also identify limitations, especially in entry point precision. The current results present a clear benchmark for future efforts, while with more robust priors and domain knowledge, such models may provide a foundation for future efforts toward lightweight and CT-free robotic intra-operative spinal navigation.
>
---
#### [replaced 019] USIM and U0: A Vision-Language-Action Dataset and Model for General Underwater Robots
- **分类: cs.RO**

- **简介: 该论文属于水下机器人任务，旨在解决多任务通用智能不足的问题。构建了USIM数据集并提出U0模型，提升水下机器人的感知与执行能力。**

- **链接: [https://arxiv.org/pdf/2510.07869](https://arxiv.org/pdf/2510.07869)**

> **作者:** Junwen Gu; Zhiheng Wu; Pengxuan Si; Shuang Qiu; Zhentao Zhang; Yukai Feng; Luoyang Sun; Laien Luo; Lianyi Yu; Jian Wang; Zhengxing Wu
>
> **备注:** Project Page: this https URL
>
> **摘要:** Underwater environments pose unique challenges for robotic navigation and manipulation. While existing research has primarily focused on task-specific methods, studies on general-purpose intelligence for multi-task execution remain scarce. To address this gap, we propose a unified framework for general-purpose underwater robots that integrates perception and action driven by language instructions. First, we develop a data synthesis pipeline to construct USIM, a simulation-based dataset which comprises over 905K frames from 2275 trajectories, totaling approximately 25 hours of BlueROV2 interactions. Furthermore, we propose U0, a vision-language-action (VLA) model capable of executing various tasks from obstacle-avoidance navigation to three-dimensional mobile manipulation. The model features a convolution-attention-based perception (CAP) module, which incorporates target pose estimation as an auxiliary task to explicitly bolster the model's spatial awareness. For evaluation, we establish a systematic assessment framework and an automated pipeline encompassing both offline metrics and online task execution. Experimental results demonstrate that the USIM dataset significantly empowers existing VLA models to adapt to underwater scenarios. Notably, our U0 model achieves state-of-the-art performance: it reduces the offline mean action prediction error to 0.0359 and achieves an overall online success rate of 43.1%, marking a 5.5% improvement over existing competitive baselines (below 37.6%), with navigation tasks reaching as high as 87.5%. These results validate the feasibility of general-purpose intelligence in underwater robotics, providing a foundation for scalable dataset synthesis and aquatic embodied agents.
>
---
#### [replaced 020] Modeling and Control of a Pneumatic Morphing Soft Quadrotor based on the SOFA Framework for Dynamic Soft Robotic Simulation
- **分类: cs.RO**

- **简介: 该论文属于软体机器人控制任务，旨在解决气动软体四旋翼机的建模与动态控制问题。通过SOFA框架实现软体结构仿真与控制设计。**

- **链接: [https://arxiv.org/pdf/2605.21031](https://arxiv.org/pdf/2605.21031)**

> **作者:** F. Labra Caso; V. Sumathy; P. Ferrentino; B. Vanderborght; J. Haluska; G. Nikolakopoulos
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** This article presents a novel SOFA based finite element method for the soft body modeling and the corresponding dynamic simulation and control of a pneumatic morphing soft quadrotor. The proposed modeling preserves the physical interpretability and control structure of traditional quadrotor dynamics, while capturing the complex, time-varying behavior of pneumatically actuated soft arms. In SOFA, the soft pneumatically actuated arms are discretized as a tetrahedral mesh following an elastic material law that produces internal forces adequate to the real dynamic behavior of the body. Pneumatic actuation governed by both periodic and error-based control signals is applied within the internal cavities to analyze the morphing capability. Finally, a proportional-integral controller is proposed to study the controlled dynamic behavior and morphing capabilities of the pneumatic arm, wherein the pneumatic actuation to the soft arm is controlled to achieve the desired target position. The simulation results show the effectiveness of the proposed novel modeling framework and the related controller design.
>
---
