# 机器人 cs.RO

- **最新发布 69 篇**

- **更新 37 篇**

## 最新发布

#### [new 001] OGScene3D: Incremental Open-Vocabulary 3D Gaussian Scene Graph Mapping for Scene Understanding
- **分类: cs.RO**

- **简介: 该论文属于开放词汇场景理解任务，解决机器人在动态环境中构建3D语义图的问题。提出OGScene3D系统，实现增量式3D语义映射与场景图构建。**

- **链接: [https://arxiv.org/pdf/2603.16301](https://arxiv.org/pdf/2603.16301)**

> **作者:** Siting Zhu; Ziyun Lu; Guangming Wang; Chenguang Huang; Yongbo Chen; I-Ming Chen; Wolfram Burgard; Hesheng Wang
>
> **摘要:** Open-vocabulary scene understanding is crucial for robotic applications, enabling robots to comprehend complex 3D environmental contexts and supporting various downstream tasks such as navigation and manipulation. However, existing methods require pre-built complete 3D semantic maps to construct scene graphs for scene understanding, which limits their applicability in robotic scenarios where environments are explored incrementally. To address this challenge, we propose OGScene3D, an open-vocabulary scene understanding system that achieves accurate 3D semantic mapping and scene graph construction incrementally. Our system employs a confidence-based Gaussian semantic representation that jointly models semantic predictions and their reliability, enabling robust scene modeling. Building on this representation, we introduce a hierarchical 3D semantic optimization strategy that achieves semantic consistency through local correspondence establishment and global refinement, thereby constructing globally consistent semantic maps. Moreover, we design a long-term global optimization method that leverages temporal memory of historical observations to enhance semantic predictions. By integrating 2D-3D semantic consistency with Gaussian rendering contribution, this method continuously refines the semantic understanding of the entire this http URL, we develop a progressive graph construction approach that dynamically creates and updates both nodes and semantic relationships, allowing continuous updating of the 3D scene graphs. Extensive experiments on widely used datasets and real-world scenes demonstrate the effectiveness of our OGScene3D on open-vocabulary scene understanding.
>
---
#### [new 002] MG-Grasp: Metric-Scale Geometric 6-DoF Grasping Framework with Sparse RGB Observations
- **分类: cs.RO**

- **简介: 该论文属于6-DoF抓取任务，解决RGB图像下几何表示不准确的问题，提出MG-Grasp框架，通过多视角重建高精度点云实现可靠抓取。**

- **链接: [https://arxiv.org/pdf/2603.16270](https://arxiv.org/pdf/2603.16270)**

> **作者:** Kangxu Wang; Siang Chen; Chenxing Jiang; Shaojie Shen; Yixiang Dai; Guijin Wang
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Single-view RGB-D grasp detection remains a com- mon choice in 6-DoF robotic grasping systems, which typically requires a depth sensor. While RGB-only 6-DoF grasp methods has been studied recently, their inaccurate geometric repre- sentation is not directly suitable for physically reliable robotic manipulation, thereby hindering reliable grasp generation. To address these limitations, we propose MG-Grasp, a novel depth- free 6-DoF grasping framework that achieves high-quality object grasping. Leveraging two-view 3D foundation model with camera intrinsic/extrinsic, our method reconstructs metric- scale and multi-view consistent dense point clouds from sparse RGB images and generates stable 6-DoF grasp. Experiments on GraspNet-1Billion dataset and real world demonstrate that MG-Grasp achieves state-of-the-art (SOTA) grasp performance among RGB-based 6-DoF grasping methods.
>
---
#### [new 003] Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决复杂抓取与操控问题。提出一种无需人工干预的强化学习框架，通过多样化重置提升探索效率，实现更鲁棒的策略学习。**

- **链接: [https://arxiv.org/pdf/2603.15789](https://arxiv.org/pdf/2603.15789)**

> **作者:** Patrick Yin; Tyler Westenbroek; Zhengyu Zhang; Joshua Tran; Ignacio Dagnino; Eeshani Shilamkar; Numfor Mbiziwo-Tiapo; Simran Bagaria; Xinlei Liu; Galen Mullins; Andrey Kolobov; Abhishek Gupta
>
> **摘要:** Reinforcement learning in massively parallel physics simulations has driven major progress in sim-to-real robot learning. However, current approaches remain brittle and task-specific, relying on extensive per-task engineering to design rewards, curricula, and demonstrations. Even with this engineering, they often fail on long-horizon, contact-rich manipulation tasks and do not meaningfully scale with compute, as performance quickly saturates when training revisits the same narrow regions of state space. We introduce \Method, a simple and scalable framework that enables on-policy reinforcement learning to robustly solve a broad class of dexterous manipulation tasks using a single reward function, fixed algorithm hyperparameters, no curricula, and no human demonstrations. Our key insight is that long-horizon exploration can be dramatically simplified by using simulator resets to systematically expose the RL algorithm to the diverse set of robot-object interactions which underlie dexterous manipulation. \Method\ programmatically generates such resets with minimal human input, converting additional compute directly into broader behavioral coverage and continued performance gains. We show that \Method\ gracefully scales to long-horizon dexterous manipulation tasks beyond the capabilities of existing approaches and is able to learn robust policies over significantly wider ranges of initial conditions than baselines. Finally, we distill \Method \ into visuomotor policies which display robust retrying behavior and substantially higher success rates than baselines when transferred to the real world zero-shot. Project webpage: this https URL
>
---
#### [new 004] Reconciling distributed compliance with high-performance control in continuum soft robotics
- **分类: cs.RO**

- **简介: 该论文属于软体机器人控制任务，旨在解决分布式柔顺与高精度控制难以共存的问题。通过创新设计实现软体机械臂的快速精准控制。**

- **链接: [https://arxiv.org/pdf/2603.16630](https://arxiv.org/pdf/2603.16630)**

> **作者:** Vito Daniele Perfetta; Daniel Feliu Talegon; Ebrahim Shahabi; Cosimo Della Santina
>
> **摘要:** High-performance closed-loop control of truly soft continuum manipulators has remained elusive. Experimental demonstrations have largely relied on sufficiently stiff, piecewise architectures in which each actuated segment behaves as a distributed yet effectively rigid element, while deformation modes beyond simple bending are suppressed. This strategy simplifies modeling and control, but sidesteps the intrinsic complexity of a fully compliant body and makes the system behave as a serial kinematic chain, much like a conventional articulated robot. An implicit conclusion has consequently emerged within the community: distributed softness and dynamic precision are incompatible. Here we show this trade-off is not fundamental. We present a highly compliant, fully continuum robotic arm - without hardware discretization or stiffness-based mode suppression - that achieves fast, precise task-space convergence under dynamic conditions. The platform integrates direct-drive actuation, a tendon routing scheme enabling coupled bending and twisting, and a structured nonlinear control architecture grounded in reduced-order strain modeling of underactuated systems. Modeling, actuation, and control are co-designed to preserve essential mechanical complexity while enabling high-bandwidth loop closure. Experiments demonstrate accurate, repeatable execution of dynamic Cartesian tasks, including fast positioning and interaction. The proposed system achieves the fastest reported task-execution speed among soft robots. At millimetric precision, execution speed increases nearly fourfold compared with prior approaches, while operating on a fully compliant continuum body. These results show that distributed compliance and high-performance dynamic control can coexist, opening a path toward truly soft manipulators approaching the operational capabilities of rigid robots without sacrificing morphological richness.
>
---
#### [new 005] Robust Dynamic Object Detection in Cluttered Indoor Scenes via Learned Spatiotemporal Cues
- **分类: cs.RO**

- **简介: 该论文属于动态目标检测任务，解决杂乱室内环境中动态障碍物检测问题。通过融合时序占用网格与学习的BEV动态先验，提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.15826](https://arxiv.org/pdf/2603.15826)**

> **作者:** Juan Rached; Yixuan Jia; Kota Kondo; Jonathan P. How
>
> **摘要:** Reliable dynamic object detection in cluttered environments remains a critical challenge for autonomous navigation. Purely geometric LiDAR pipelines that rely on clustering and heuristic filtering can miss dynamic obstacles when they move in close proximity to static structure or are only partially observed. Vision-augmented approaches can provide additional semantic cues, but are often limited by closed-set detectors and camera field-of-view constraints, reducing robustness to novel obstacles and out-of-frustum events. In this work, we present a LiDAR-only framework that fuses temporal occupancy-grid-based motion segmentation with a learned bird's-eye-view (BEV) dynamic prior. A fusion module prioritizes 3D detections when available, while using the learned dynamic grid to recover detections that would otherwise be lost due to proximity-induced false negatives. Experiments with motion-capture ground truth show our method achieves 28.67% higher recall and 18.50% higher F1 score than the state-of-the-art in substantially cluttered environments while maintaining comparable precision and position error.
>
---
#### [new 006] Toward Deep Representation Learning for Event-Enhanced Visual Autonomous Perception: the eAP Dataset
- **分类: cs.RO**

- **简介: 该论文提出eAP数据集，解决自动驾驶中光照挑战下的视觉感知问题。通过事件相机与深度学习结合，提升3D目标检测和TTC估计性能。**

- **链接: [https://arxiv.org/pdf/2603.16303](https://arxiv.org/pdf/2603.16303)**

> **作者:** Jinghang Li; Shichao Li; Qing Lian; Peiliang Li; Xiaozhi Chen; Yi Zhou
>
> **摘要:** Recent visual autonomous perception systems achieve remarkable performances with deep representation learning. However, they fail in scenarios with challenging this http URL event cameras can mitigate this problem, there is a lack of a large-scale dataset to develop event-enhanced deep visual perception models in autonomous driving scenes. To address the gap, we present the eAP (event-enhanced Autonomous Perception) dataset, the largest dataset with event cameras for autonomous perception. We demonstrate how eAP can facilitate the study of different autonomous perception tasks, including 3D vehicle detection and object time-to-contact (TTC) estimation, through deep representation learning. Based on eAP, we demonstrate the ffrst successful use of events to improve a popular 3D vehicle detection network in challenging illumination scenarios. eAP also enables a devoted study of the representation learning problem of object TTC estimation. We show how a geometryaware representation learning framework leads to the best eventbased object TTC estimation network that operates at 200 FPS. The dataset, code, and pre-trained models will be made publicly available for future research.
>
---
#### [new 007] Ultrafast Sampling-based Kinodynamic Planning via Differential Flatness
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决动态约束下的轨迹生成问题。通过微分平坦性，将问题转换到平坦输出空间，实现快速、精确的轨迹规划。**

- **链接: [https://arxiv.org/pdf/2603.16059](https://arxiv.org/pdf/2603.16059)**

> **作者:** Thai Duong; Clayton W. Ramsey; Zachary Kingston; Wil Thomason; Lydia E. Kavraki
>
> **备注:** 16 pages, 9 figures, under review
>
> **摘要:** Motion planning under dynamics constraints, i.e., kinodynamic planning, enables safe robot operation by generating dynamically feasible trajectories that the robot can accurately track. For high-\dof robots such as manipulators, sampling-based motion planners are commonly used, especially for complex tasks in cluttered environments. However, enforcing constraints on robot dynamics in such planners requires solving either challenging two-point boundary value problems (BVPs) or propagating robot dynamics over time, both of which are computational bottlenecks that drastically increase planning times. Meanwhile, recent efforts have shown that sampling-based motion planners can generate plans in microseconds using parallelization, but are limited to geometric paths. This paper develops AkinoPDF, a fast parallelized sampling-based kinodynamic motion planning technique for a broad class of differentially flat robot systems, including manipulators, ground and aerial vehicles, and more. Differential flatness allows us to transform the motion planning problem from the original state space to a flat output space, where an analytical time-parameterized solution of the BVP and dynamics integration can be obtained. A trajectory in the flat output space is then converted back to a closed-form dynamically feasible trajectory in the original state space, enabling fast validation via ``single instruction, multiple data" parallelism. Our method is fast, exact, and compatible with any sampling-based motion planner. We extensively verify the effectiveness of our approach in both simulated benchmarks and real experiments with cluttered and dynamic environments, requiring mere microseconds to milliseconds of planning time.
>
---
#### [new 008] Faulty Coffees: Barriers to Adoption of an In-the-wild Robo-Barista
- **分类: cs.RO**

- **简介: 该论文属于人机交互研究，探讨任务叙事对服务机器人长期使用的影响。通过部署罗伯咖啡师实验，发现用户参与度低、技术故障和社交障碍等问题，提出对长期人机交互研究的反思与改进方向。**

- **链接: [https://arxiv.org/pdf/2603.16336](https://arxiv.org/pdf/2603.16336)**

> **作者:** Bruce W. Wilson; David A. Robb; Mei Yii Lim; Helen Hastie; Matthew Peter Aylett; Theodoros Georgiou
>
> **备注:** Accepted for publication in Failing Forward, Design and Deployment Lessons from Real-World Human-Robot Interaction Workshop at HRI 2026, March 16, 2026, Edinburgh, Scotland
>
> **摘要:** We set out to study whether task-based narratives could influence long-term engagement with a service robot. To do so, we deployed a Robo-Barista for five weeks in an over-50's housing complex in Stockton, England. Residents received a free daily coffee by interacting with a Furhat robot assigned to either a narrative or non-narrative dialogue condition. Despite designing for sustained engagement, repeat interaction was low, and we encountered curiosity trials without retention, technical breakdowns, accessibility barriers, and the social dynamics of a housing complex setting. Rather than treating these as peripheral issues, we foreground them in this paper. We reflect on the in-the-wild realities of our experiment and offer lessons for conducting longitudinal Human-Robot Interaction research when studies unravel in practice.
>
---
#### [new 009] ADAPT: Adaptive Dual-projection Architecture for Perceptive Traversal
- **分类: cs.RO**

- **简介: 该论文提出ADAPT架构，用于复杂3D环境中的仿人机器人运动控制，解决感知与计算效率的平衡问题，通过自适应投影实现高效环境感知和路径规划。**

- **链接: [https://arxiv.org/pdf/2603.16328](https://arxiv.org/pdf/2603.16328)**

> **作者:** Shuo Shao; Tianchen Huang; Wei Gao; Shiwu Zhang
>
> **摘要:** Agile humanoid locomotion in complex 3D en- vironments requires balancing perceptual fidelity with com- putational efficiency, yet existing methods typically rely on rigid sensing configurations. We propose ADAPT (Adaptive dual-projection architecture for perceptive traversal), which represents the environment using a horizontal elevation map for terrain geometry and a vertical distance map for traversable- space constraints. ADAPT further treats its spatial sensing range as a learnable action, enabling the policy to expand its perceptual horizon during fast motion and contract it in cluttered scenes for finer local resolution. Compared with voxel-based baselines, ADAPT drastically reduces observation dimensionality and computational overhead while substantially accelerating training. Experimentally, it achieves successful zero-shot transfer to a Unitree G1 Humanoid and signifi- cantly outperforms fixed-range baselines, yielding highly robust traversal across diverse 3D environtmental challenges.
>
---
#### [new 010] vAccSOL: Efficient and Transparent AI Vision Offloading for Mobile Robots
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出vAccSOL框架，解决移动机器人视觉任务的计算资源不足问题，通过优化推理和边缘卸载提升效率与续航。**

- **链接: [https://arxiv.org/pdf/2603.16685](https://arxiv.org/pdf/2603.16685)**

> **作者:** Adam Zahir; Michele Gucciardom Falk Selker; Anastasios Nanos; Kostis Papazafeiropoulos; Carlos J. Bernardos; Nicolas Weber; Roberto Gonzalez
>
> **摘要:** Mobile robots are increasingly deployed for inspection, patrol, and search-and-rescue operations, relying on computer vision for perception, navigation, and autonomous decision-making. However, executing modern vision workloads onboard is challenging due to limited compute resources and strict energy constraints. While some platforms include embedded accelerators, these are typically tied to proprietary software stacks, leaving user-defined workloads to run on resource-constrained companion computers. We present vAccSOL, a framework for efficient and transparent execution of AI-based vision workloads across heterogeneous robotic and edge platforms. vAccSOL integrates two components: SOL, a neural network compiler that generates optimized inference libraries with minimal runtime dependencies, and vAccel, a lightweight execution framework that transparently dispatches inference locally on the robot or to nearby edge infrastructure. This combination enables hardware-optimized inference and flexible execution placement without requiring modifications to robot applications. We evaluate vAccSOL on a real-world testbed with a commercial quadruped robot and twelve deep learning models covering image classification, video classification, and semantic segmentation. Compared to a PyTorch compiler baseline, SOL achieves comparable or better inference performance. With edge offloading, vAccSOL reduces robot-side power consumption by up to 80% and edge-side power by up to 60% compared to PyTorch, while increasing vision pipeline frame rate by up to 24x, extending the operating lifetime of battery-powered robots.
>
---
#### [new 011] Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人学中的模拟到现实迁移任务，旨在解决仿真与真实环境差异导致的适应问题。通过模拟蒸馏方法，将仿真先验知识迁移到世界模型，实现快速现实适应。**

- **链接: [https://arxiv.org/pdf/2603.15759](https://arxiv.org/pdf/2603.15759)**

> **作者:** Jacob Levy; Tyler Westenbroek; Kevin Huang; Fernando Palafox; Patrick Yin; Shayegan Omidshafiei; Dong-Ki Kim; Abhishek Gupta; David Fridovich-Keil
>
> **备注:** Project website: this https URL
>
> **摘要:** Simulation-to-real transfer remains a central challenge in robotics, as mismatches between simulated and real-world dynamics often lead to failures. While reinforcement learning offers a principled mechanism for adaptation, existing sim-to-real finetuning methods struggle with exploration and long-horizon credit assignment in the low-data regimes typical of real-world robotics. We introduce Simulation Distillation (SimDist), a sim-to-real framework that distills structural priors from a simulator into a latent world model and enables rapid real-world adaptation via online planning and supervised dynamics finetuning. By transferring reward and value models directly from simulation, SimDist provides dense planning signals from raw perception without requiring value learning during deployment. As a result, real-world adaptation reduces to short-horizon system identification, avoiding long-horizon credit assignment and enabling fast, stable improvement. Across precise manipulation and quadruped locomotion tasks, SimDist substantially outperforms prior methods in data efficiency, stability, and final performance. Project website and code: this https URL
>
---
#### [new 012] BrickSim: A Physics-Based Simulator for Manipulating Interlocking Brick Assemblies
- **分类: cs.RO; cs.GR**

- **简介: 该论文提出BrickSim，用于模拟积木装配的物理交互，解决传统仿真无法准确捕捉卡扣机制的问题。属于机器人操作任务，通过力模型和优化算法实现高保真实时仿真。**

- **链接: [https://arxiv.org/pdf/2603.16853](https://arxiv.org/pdf/2603.16853)**

> **作者:** Haowei Wen; Ruixuan Liu; Weiyi Piao; Siyu Li; Changliu Liu
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** Interlocking brick assemblies provide a standardized yet challenging testbed for contact-rich and long-horizon robotic manipulation, but existing rigid-body simulators do not faithfully capture snap-fit mechanics. We present BrickSim, the first real-time physics-based simulator for interlocking brick assemblies. BrickSim introduces a compact force-based mechanics model for snap-fit connections and solves the resulting internal force distribution using a structured convex quadratic program. Combined with a hybrid architecture that delegates rigid-body dynamics to the underlying physics engine while handling snap-fit mechanics separately, BrickSim enables real-time, high-fidelity simulation of assembly, disassembly, and structural collapse. On 150 real-world assemblies, BrickSim achieves 100% accuracy in static stability prediction with an average solve time of 5 ms. In dynamic drop tests, it also faithfully reproduces real-world structural collapse, precisely mirroring both the occurrence of breakage and the specific breakage locations. Built on Isaac Sim, BrickSim further supports seamless integration with a wide variety of robots and existing pipelines. We demonstrate robotic construction of brick assemblies using BrickSim, highlighting its potential as a foundation for research in dexterous, long-horizon robotic manipulation. BrickSim is open-source, and the code is available at this https URL.
>
---
#### [new 013] Beyond Cybathlon: On-demand Quadrupedal Assistance for People with Limited Mobility
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于服务机器人任务，旨在为行动受限人群提供自主协助。提出一种可随需应变的四足机器人系统，结合半自主与远程操控，解决移动操作难题，提升用户独立性。**

- **链接: [https://arxiv.org/pdf/2603.16772](https://arxiv.org/pdf/2603.16772)**

> **作者:** Carmen Scheidemann; Andrei Cramariuc; Changan Chen; Jia-Ruei Chiu; Marco Hutter
>
> **摘要:** Background: Assistance robots have the potential to increase the independence of people who need daily care due to limited mobility or being wheelchair-bound. Current solutions of attaching robotic arms to motorized wheelchairs offer limited additional mobility at the cost of increased size and reduced wheelchair maneuverability. Methods: We present an on-demand quadrupedal assistance robot system controlled via a shared autonomy approach, which combines semi-autonomous task execution with human teleoperation. Due to the mobile nature of the system it can assist the operator whenever needed and perform autonomous tasks independently, without otherwise restricting their mobility. We automate pick-and-place tasks, as well as robot movement through the environment with semantic, collision-aware navigation. For teleoperation, we present a mouth-level joystick interface that enables an operator with reduced mobility to control the robot's end effector for precision manipulation. Results: We showcase our system in the \textit{Cybathlon 2024 Assistance Robot Race}, and validate it in an at-home experimental setup, where we measure task completion times and user satisfaction. We find our system capable of assisting in a broad variety of tasks, including those that require dexterous manipulation. The user study confirms the intuition that increased robot autonomy alleviates the operator's mental load. Conclusions: We present a flexible system that has the potential to help people in wheelchairs maintain independence in everyday life by enabling them to solve mobile manipulation problems without external support. We achieve results comparable to previous state-of-the-art on subjective metrics while allowing for more autonomy of the operator and greater agility for manipulation.
>
---
#### [new 014] Learning Whole-Body Control for a Salamander Robot
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决仿鳗机器人在复杂环境中实现稳定全身控制的问题。通过强化学习方法，设计可迁移的关节级控制器，并成功实现在真实地形和水陆转换中的协调运动。**

- **链接: [https://arxiv.org/pdf/2603.16683](https://arxiv.org/pdf/2603.16683)**

> **作者:** Mengze Tian; Qiyuan Fu; Chuanfang Ning; Javier Jia Jie Pey; Auke Ijspeert
>
> **摘要:** Amphibious legged robots inspired by salamanders are promising in applications in complex amphibious environments. However, despite the significant success of training controllers that achieve diverse locomotion behaviors in conventional quadrupedal robots, most salamander robots relied on central-pattern-generator (CPG)-based and model-based coordination strategies for locomotion control. Learning unified joint-level whole-body control that reliably transfers from simulation to highly articulated physical salamander robots remains relatively underexplored. In addition, few legged robots have tried learning-based controllers in amphibious environments. In this work, we employ Reinforcement Learning to map proprioceptive observations and commanded velocities to joint-level actions, allowing coordinated locomotor behaviors to emerge. To deploy these policies on hardware, we adopt a system-level real-to-sim matching and sim-to-real transfer strategy. The learned controller achieves stable and coordinated walking on both flat and uneven terrains in the real world. Beyond terrestrial locomotion, the framework enables transitions between walking and swimming in simulation, highlighting a phenomenon of interest for understanding locomotion across distinct physical modes.
>
---
#### [new 015] Industrial cuVSLAM Benchmark & Integration
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，旨在提升移动机器人在真实环境中的定位精度。通过对比不同VO/VSLAM方法，提出融合cuVSLAM的集成方案，优化定位性能。**

- **链接: [https://arxiv.org/pdf/2603.16240](https://arxiv.org/pdf/2603.16240)**

> **作者:** Charbel Abi Hana; Kameel Amareen; Mohamad Mostafa; Dmitry Slepichev; Hesam Rabeti; Zheng Wang; Mihir Acharya; Anthony Rizk
>
> **摘要:** This work presents a comprehensive benchmark evaluation of visual odometry (VO) and visual SLAM (VSLAM) systems for mobile robot navigation in real-world logistical environments. We compare multiple visual odometry approaches across controlled trajectories covering translational, rotational, and mixed motion patterns, as well as a large-scale production facility dataset spanning approximately 1.7 km. Performance is evaluated using Absolute Pose Error (APE) against ground truth from a Vicon motion capture system and a LiDAR-based SLAM reference. Our results show that a hybrid stack combining the cuVSLAM front-end with a custom SLAM back-end achieves the strongest mapping accuracy, motivating a deeper integration of cuVSLAM as the core VO component in our robotics stack. We further validate this integration by deploying and testing the cuVSLAM-based VO stack on an NVIDIA Jetson platform.
>
---
#### [new 016] PA-LVIO: Real-Time LiDAR-Visual-Inertial Odometry and Mapping with Pose-Only Bundle Adjustment
- **分类: cs.RO**

- **简介: 该论文属于实时LiDAR-视觉-惯性里程计与建图任务，解决导航与建图中的精度和实时性问题，提出PA-LVIO方法提升定位与地图质量。**

- **链接: [https://arxiv.org/pdf/2603.16228](https://arxiv.org/pdf/2603.16228)**

> **作者:** Hailiang Tang; Tisheng Zhang; Liqiang Wang; Xin Ding; Man Yuan; Xiaoji Niu
>
> **备注:** 14 pages, 10 figures
>
> **摘要:** Real-time LiDAR-visual-inertial odometry and mapping is crucial for navigation and planning tasks in intelligent transportation systems. This study presents a pose-only bundle adjustment (PA) LiDAR-visual-inertial odometry (LVIO), named PA-LVIO, to meet the urgent need for real-time navigation and mapping. The proposed PA framework for LiDAR and visual measurements is highly accurate and efficient, and it can derive reliable frame-to-frame constraints within multiple frames. A marginalization-free and frame-to-map (F2M) LiDAR measurement model is integrated into the state estimator to eliminate odometry drifts. Meanwhile, an IMU-centric online spatial-temporal calibration is employed to obtain a pixel-wise LiDAR-camera alignment. With accurate estimated odometry and extrinsics, a high-quality and RGB-rendered point-cloud map can be built. Comprehensive experiments are conducted on both public and private datasets collected by wheeled robot, unmanned aerial vehicle (UAV), and handheld devices with 28 sequences and more than 50 km trajectories. Sufficient results demonstrate that the proposed PA-LVIO yields superior or comparable performance to state-of-the-art LVIO methods, in terms of the odometry accuracy and mapping quality. Besides, PA-LVIO can run in real-time on both the desktop PC and the onboard ARM computer.
>
---
#### [new 017] Early-Terminable Energy-Safe Iterative Coupling for Parallel Simulation of Port-Hamiltonian Systems
- **分类: cs.RO; eess.SY; math.NA**

- **简介: 该论文属于并行仿真任务，解决子系统耦合中能量不守恒问题。提出一种能量安全的迭代耦合方法，确保并行仿真中的能量一致性。**

- **链接: [https://arxiv.org/pdf/2603.16424](https://arxiv.org/pdf/2603.16424)**

> **作者:** Qi Wei; Jianfeng Tao; Hongyu Nie; Wangtao Tan
>
> **摘要:** Parallel simulation and control of large-scale robotic systems often rely on partitioned time stepping, yet finite-iteration coupling can inject spurious energy by violating power consistency--even when each subsystem is passive. This letter proposes a novel energy-safe, early-terminable iterative coupling for port-Hamiltonian subsystems by embedding a Douglas--Rachford (DR) splitting scheme in scattering (wave) coordinates. The lossless interconnection is enforced as an orthogonal constraint in the wave domain, while each subsystem contributes a discrete-time scattering port map induced by its one-step integrator. Under a discrete passivity condition on the subsystem time steps and a mild impedance-tuning condition, we prove an augmented-storage inequality certifying discrete passivity of the coupled macro-step for any finite inner-iteration budget, with the remaining mismatch captured by an explicit residual. As the inner budget increases, the partitioned update converges to the monolithic discrete-time update induced by the same integrators, yielding a principled, adaptive accuracy--compute trade-off, supporting energy-consistent real-time parallel simulation under varying computational budgets. Experiments on a coupled-oscillator benchmark validate the passivity certificates at numerical roundoff (on the order of 10e-14 in double precision) and show that the reported RMS state error decays monotonically with increasing inner-iteration budgets, consistent with the hard-coupling limit.
>
---
#### [new 018] MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决模拟到现实的零样本迁移问题。通过大规模仿真数据训练，实现无需真实数据微调的高效操作。**

- **链接: [https://arxiv.org/pdf/2603.16861](https://arxiv.org/pdf/2603.16861)**

> **作者:** Abhay Deshpande; Maya Guru; Rose Hendrix; Snehal Jauhri; Ainaz Eftekhar; Rohun Tripathi; Max Argus; Jordi Salvador; Haoquan Fang; Matthew Wallingford; Wilbert Pumacay; Yejin Kim; Quinn Pfeifer; Ying-Chun Lee; Piper Wolters; Omar Rayyan; Mingtong Zhang; Jiafei Duan; Karen Farley; Winson Han; Eli Vanderbilt; Dieter Fox; Ali Farhadi; Georgia Chalvatzaki; Dhruv Shah; Ranjay Krishna
>
> **摘要:** A prevailing view in robot learning is that simulation alone is not enough; effective sim-to-real transfer is widely believed to require at least some real-world data collection or task-specific fine-tuning to bridge the gap between simulated and physical environments. We challenge that assumption. With sufficiently large-scale and diverse simulated synthetic training data, we show that zero-shot transfer to the real world is not only possible, but effective for both static and mobile manipulation. We introduce MolmoBot-Engine, a fully open-source pipeline for procedural data generation across robots, tasks, and diverse simulated environments in MolmoSpaces. With it, we release MolmoBot-Data, a dataset of 1.8 million expert trajectories for articulated object manipulation and pick-and-place tasks. We train three policy classes: MolmoBot, a Molmo2-based multi-frame vision-language model with a flow-matching action head; MolmoBot-Pi0, which replicates the $\pi_0$ architecture to enable direct comparison; and MolmoBot-SPOC, a lightweight policy suitable for edge deployment and amenable to RL fine-tuning. We evaluate on two robotic platforms: the Franka FR3 for tabletop manipulation tasks and the Rainbow Robotics RB-Y1 mobile manipulator for door opening, drawer manipulation, cabinet interaction, and mobile pick-and-place. Without any real-world fine-tuning, our policies achieve zero-shot transfer to unseen objects and environments. On tabletop pick-and-place, MolmoBot achieves a success rate of 79.2% in real world evaluations across 4 settings, outperforming $\pi_{0.5}$ at 39.2%. Our results demonstrate that procedural environment generation combined with diverse articulated assets can produce robust manipulation policies that generalize broadly to the real world. Technical Blog: this https URL
>
---
#### [new 019] Enabling Dynamic Tracking in Vision-Language-Action Models via Time-Discrete and Time-Continuous Velocity Feedforward
- **分类: cs.RO**

- **简介: 该论文属于机器人操控任务，解决VLA模型在工业机器人上部署时的响应与柔顺性矛盾。通过引入速度前馈项提升控制性能。**

- **链接: [https://arxiv.org/pdf/2603.16218](https://arxiv.org/pdf/2603.16218)**

> **作者:** Johannes Hechtl; Philipp Schmitt; Georg von Wichert; Wolfram Burgard
>
> **摘要:** While vision-language-action (VLA) models have shown great promise for robot manipulation, their deployment on rigid industrial robots remains challenging due to the inherent trade-off between compliance and responsiveness. Standard Behavior Cloning (BC) approaches predict discrete poses at low frequencies, omitting the velocity and acceleration feedforward terms typically used by low-level compliant controllers. This requires to rely on high stiffness for accurate tracking, thereby sacrificing safe contact dynamics. In this paper, we demonstrate the importance of integrating velocity feedforward terms into VLA policies to resolve this trade-off. We propose two methods for extracting velocity targets from VLAs: a time-discrete finite-difference approximation that serves as a highly effective bridge for existing models, and a continuous Cubic B-Spline action space that natively yields $C^2$ continuous trajectories for high-frequency control. Crucially, both approaches are strictly model-agnostic and compatible with any standard action-chunking architecture, requiring modifications only to teleoperation, data processing, and the low-level controller. We fine-tune the $\pi_{0.5}$ model and evaluate both of our approaches on a demanding, contact-rich cube-in-hole task. Our results indicate that incorporating the velocity feedforward term via finite differences significantly improves task execution speed, while the continuous B-Spline approach maintains high overall success rates and provides a foundation for smoother higher-order derivatives without compromising compliance.
>
---
#### [new 020] Dexterous grasp data augmentation based on grasp synthesis with fingertip workspace cloud and contact-aware sampling
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决抓取数据生成效率低的问题。通过引入FSG和AutoWS，实现高效、通用的抓取数据增强。**

- **链接: [https://arxiv.org/pdf/2603.16609](https://arxiv.org/pdf/2603.16609)**

> **作者:** Liqi Wu; Haoyu Jia; Kento Kawaharazuka; Hirokazu Ishida; Kei Okada
>
> **备注:** Accepted to Advanced Robotics, GitHub: this https URL, YouTube: this https URL
>
> **摘要:** Robotic grasping is a fundamental yet crucial component of robotic applications, as effective grasping often serves as the starting point for various tasks. With the rapid advancement of neural networks, data-driven approaches for robotic grasping have become mainstream. However, efficiently generating grasp datasets for training remains a bottleneck. This is compounded by the diverse structures of robotic hands, making the design of generalizable grasp generation methods even more complex. In this work, we propose a teleoperation-based framework to collect a small set of grasp pose demonstrations, which are augmented using FSG--a Fingertip-contact-aware Sampling-based Grasp generator. Based on the demonstrated grasp poses, we propose AutoWS, which automatically generates structured workspace clouds of robotic fingertips, embedding the hand structure information directly into the clouds to eliminate the need for inverse kinematics calculations. Experiments on grasping the YCB objects show that our method significantly outperforms existing approaches in both speed and valid pose generation rate. Our framework enables real-time grasp generation for hands with arbitrary structures and produces human-like grasps when combined with demonstrations, providing an efficient and robust data augmentation tool for data-driven grasp training.
>
---
#### [new 021] ManiTwin: Scaling Data-Generation-Ready Digital Object Dataset to 100K
- **分类: cs.RO; cs.AI; cs.GR; cs.LG; cs.SE**

- **简介: 该论文提出ManiTwin，解决机器人操作数据生成中数字资产不足的问题，通过自动化流程生成100K高质量3D资产，用于仿真数据合成与策略学习。**

- **链接: [https://arxiv.org/pdf/2603.16866](https://arxiv.org/pdf/2603.16866)**

> **作者:** Kaixuan Wang; Tianxing Chen; Jiawei Liu; Honghao Su; Shaolong Zhu; Minxuan Wang; Zixuan Li; Yue Chen; Huan-ang Gao; Yusen Qin; Jiawei Wang; Qixuan Zhang; Lan Xu; Jingyi Yu; Yao Mu; Ping Luo
>
> **备注:** Website: this https URL
>
> **摘要:** Learning in simulation provides a useful foundation for scaling robotic manipulation capabilities. However, this paradigm often suffers from a lack of data-generation-ready digital assets, in both scale and diversity. In this work, we present ManiTwin, an automated and efficient pipeline for generating data-generation-ready digital object twins. Our pipeline transforms a single image into simulation-ready and semantically annotated 3D asset, enabling large-scale robotic manipulation data generation. Using this pipeline, we construct ManiTwin-100K, a dataset containing 100K high-quality annotated 3D assets. Each asset is equipped with physical properties, language descriptions, functional annotations, and verified manipulation proposals. Experiments demonstrate that ManiTwin provides an efficient asset synthesis and annotation workflow, and that ManiTwin-100K offers high-quality and diverse assets for manipulation data generation, random scene synthesis, and VQA data generation, establishing a strong foundation for scalable simulation data synthesis and policy learning. Our webpage is available at this https URL.
>
---
#### [new 022] Coverage First Next Best View for Inspection of Cluttered Pipe Networks Using Mobile Manipulators
- **分类: cs.RO**

- **简介: 该论文属于机器人自主导航任务，解决在复杂管道中自主巡检与覆盖路径规划问题。提出基于信息增益的下一视角规划方法，结合概率约束实现避障与环境重建。**

- **链接: [https://arxiv.org/pdf/2603.16471](https://arxiv.org/pdf/2603.16471)**

> **作者:** Joshua Raymond Bettles; Jiaxu Wu; Bruno Vilhena Adorno; Joaquin Carrasco; Atsushi Yamashita
>
> **备注:** 8 pages, 9 figures, 1 table. Submitted to IEEE/RSJ International Conference on Intelligent Robots and Systems 2026
>
> **摘要:** Robotic inspection of radioactive areas enables operators to be removed from hazardous environments; however, planning and operating in confined, cluttered environments remain challenging. These systems must autonomously reconstruct the unknown environment and cover its surfaces, whilst estimating and avoiding collisions with objects in the environment. In this paper, we propose a new planning approach based on next-best-view that enables simultaneous exploration and exploitation of the environment by reformulating the coverage path planning problem in terms of information gain. To handle obstacle avoidance under uncertainty, we extend the vector-field-inequalities framework to explicitly account for stochastic measurements of geometric primitives in the environment via chance constraints in a constrained optimal control law. The stochastic constraints were evaluated experimentally alongside the planner on a mobile manipulator in a confined environment to inspect a pipe network. These experiments demonstrate that the system can autonomously plan and execute inspection and coverage paths to reconstruct and fully cover the simplified pipe network. Moreover, the system successfully estimated geometric primitives online and avoided collisions during motion between viewpoints.
>
---
#### [new 023] SE(3)-LIO: Smooth IMU Propagation With Jointly Distributed Poses on SE(3) Manifold for Accurate and Robust LiDAR-Inertial Odometry
- **分类: cs.RO**

- **简介: 该论文属于LiDAR-Inertial Odometry任务，旨在解决IMU传播中的运动预测与补偿问题，提出SE(3)-LIO方法，实现更准确的位姿估计。**

- **链接: [https://arxiv.org/pdf/2603.16118](https://arxiv.org/pdf/2603.16118)**

> **作者:** Gunhee Shin; Seungjae Lee; Jei Kong; Youngwoo Seo; Hyun Myung
>
> **摘要:** In estimating odometry accurately, an inertial measurement unit (IMU) is widely used owing to its high-rate measurements, which can be utilized to obtain motion information through IMU propagation. In this paper, we address the limitations of existing IMU propagation methods in terms of motion prediction and motion compensation. In motion prediction, the existing methods typically represent a 6-DoF pose by separating rotation and translation and propagate them on their respective manifold, so that the rotational variation is not effectively incorporated into translation propagation. During motion compensation, the relative transformation between predicted poses is used to compensate motion-induced distortion in other measurements, while inherent errors in the predicted poses introduce uncertainty in the relative transformation. To tackle these challenges, we represent and propagate the pose on SE(3) manifold, where propagated translation properly accounts for rotational variation. Furthermore, we precisely characterize the relative transformation uncertainty by considering the correlation between predicted poses, and incorporate this uncertainty into the measurement noise during motion compensation. To this end, we propose a LiDAR-inertial odometry (LIO), referred to as SE(3)-LIO, that integrates the proposed IMU propagation and uncertainty-aware motion compensation (UAMC). We validate the effectiveness of SE(3)-LIO on diverse datasets. Our source code and additional material are available at: this https URL.
>
---
#### [new 024] Onboard MuJoCo-based Model Predictive Control for Shipboard Crane with Double-Pendulum Sway Suppression
- **分类: cs.RO**

- **简介: 该论文属于船舶起重机控制任务，旨在解决双摆晃动问题。通过基于MuJoCo的模型预测控制方法，实现实时有效抑制晃动，提升操作安全性与效率。**

- **链接: [https://arxiv.org/pdf/2603.16407](https://arxiv.org/pdf/2603.16407)**

> **作者:** Oscar Pang; Lisa Coiffard; Paul Templier; Luke Beddow; Kamil Dreczkowski; Antoine Cully
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Transferring heavy payloads in maritime settings relies on efficient crane operation, limited by hazardous double-pendulum payload sway. This sway motion is further exacerbated in offshore environments by external perturbations from wind and ocean waves. Manual suppression of these oscillations on an underactuated crane system by human operators is challenging. Existing control methods struggle in such settings, often relying on simplified analytical models, while deep reinforcement learning (RL) approaches tend to generalise poorly to unseen conditions. Deploying a predictive controller onto compute-constrained, highly non-linear physical systems without relying on extensive offline training or complex analytical models remains a significant challenge. Here we show a complete real-time control pipeline centered on the MuJoCo MPC framework that leverages a cross-entropy method planner to evaluate candidate action sequences directly within a physics simulator. By using simulated rollouts, this sampling-based approach successfully reconciles the conflicting objectives of dynamic target tracking and sway damping without relying on complex analytical models. We demonstrate that the controller can run effectively on a resource-constrained embedded hardware, while outperforming traditional PID and RL baselines in counteracting external base perturbations. Furthermore, our system demonstrates robustness even when subjected to unmodeled physical discrepancies like the introduction of a second payload.
>
---
#### [new 025] GenZ-LIO: Generalizable LiDAR-Inertial Odometry Beyond Indoor--Outdoor Boundaries
- **分类: cs.RO**

- **简介: 该论文提出GenZ-LIO，解决LiDAR-Inertial Odometry在室内外场景转换中的尺度敏感问题，通过自适应降采样、混合度量更新和体素剪枝策略提升鲁棒性和效率。**

- **链接: [https://arxiv.org/pdf/2603.16273](https://arxiv.org/pdf/2603.16273)**

> **作者:** Daehan Lee; Hyungtae Lim; Seongjun Kim; Soonbin Rho; Changhyeon Lee; Sanghyun Park; Junwoo Hong; Eunseon Choi; Hyunyoung Jo; Soohee Han
>
> **备注:** 19 pages, 11 figures
>
> **摘要:** Light detection and ranging (LiDAR)-inertial odometry (LIO) enables accurate localization and mapping for autonomous navigation in various scenes. However, its performance remains sensitive to variations in spatial scale, which refers to the spatial extent of the scene reflected in the distribution of point ranges in a LiDAR scan. Transitions between confined indoor and expansive outdoor spaces induce substantial variations in point density, which may reduce robustness and computational efficiency. To address this issue, we propose GenZ-LIO, a LIO framework generalizable across both indoor and outdoor environments. GenZ-LIO comprises three key components. First, inspired by the principle of the proportional-integral-derivative (PID) controller, it adaptively regulates the voxel size for downsampling via feedback control, driving the voxelized point count toward a scale-informed setpoint while enabling stable and efficient processing across varying scene scales. Second, we formulate a hybrid-metric state update that jointly leverages point-to-plane and point-to-point residuals to mitigate LiDAR degeneracy arising from directionally insufficient geometric constraints. Third, to alleviate the computational burden introduced by point-to-point matching, we introduce a voxel-pruned correspondence search strategy that discards non-promising voxel candidates and reduces unnecessary computations. Experimental results demonstrate that GenZ-LIO achieves robust odometry estimation and improved computational efficiency across confined indoor, open outdoor, and transitional environments. Our code will be made publicly available upon publication.
>
---
#### [new 026] When Rolling Gets Weird: A Curved-Link Tensegrity Robot for Non-Intuitive Behavior
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人设计任务，旨在解决传统张力结构机器人在滚动与稳定性间的平衡问题。通过设计弯曲链接的张力机器人，实现高效且稳定的移动。**

- **链接: [https://arxiv.org/pdf/2603.16503](https://arxiv.org/pdf/2603.16503)**

> **作者:** Lauren Ervin; Harish Bezawada; Vishesh Vikas
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Conventional mobile tensegrity robots constructed with straight links offer mobility at the cost of locomotion speed. While spherical robots provide highly effective rolling behavior, they often lack the stability required for navigating unstructured terrain common in many space exploration environments. This research presents a solution with a semi-circular, curved-link tensegrity robot that strikes a balance between efficient rolling locomotion and controlled stability, enabled by discontinuities present at the arc endpoints. Building upon an existing geometric static modeling framework [1], this work presents the system design of an improved Tensegrity eXploratory Robot 2 (TeXploR2). Internal shifting masses instantaneously roll along each curved-link, dynamically altering the two points of contact with the ground plane. Simulations of quasistatic, piecewise continuous locomotion sequences reveal new insights into the positional displacement between inertial and body frames. Non-intuitive rolling behaviors are identified and experimentally validated using a tetherless prototype, demonstrating successful dynamic locomotion. A preliminary impact test highlights the tensegrity structure's inherent shock absorption capabilities and conformability. Future work will focus on finalizing a dynamic model that is experimentally validated with extended testing in real-world environments as well as further refinement of the prototype to incorporate additional curved-links and subsequent ground contact points for increased controllability.
>
---
#### [new 027] ExpertGen: Scalable Sim-to-Real Expert Policy Learning from Imperfect Behavior Priors
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ExpertGen，解决机器人行为克隆中数据获取成本高的问题。通过模拟学习专家策略，实现高效、可靠的现实迁移。**

- **链接: [https://arxiv.org/pdf/2603.15956](https://arxiv.org/pdf/2603.15956)**

> **作者:** Zifan Xu; Ran Gong; Maria Vittoria Minniti; Ahmet Salih Gundogdu; Eric Rosen; Kausik Sivakumar; Riedana Yan; Zixing Wang; Di Deng; Peter Stone; Xiaohan Zhang; Karl Schmeckpeper
>
> **摘要:** Learning generalizable and robust behavior cloning policies requires large volumes of high-quality robotics data. While human demonstrations (e.g., through teleoperation) serve as the standard source for expert behaviors, acquiring such data at scale in the real world is prohibitively expensive. This paper introduces ExpertGen, a framework that automates expert policy learning in simulation to enable scalable sim-to-real transfer. ExpertGen first initializes a behavior prior using a diffusion policy trained on imperfect demonstrations, which may be synthesized by large language models or provided by humans. Reinforcement learning is then used to steer this prior toward high task success by optimizing the diffusion model's initial noise while keep original policy frozen. By keeping the pretrained diffusion policy frozen, ExpertGen regularizes exploration to remain within safe, human-like behavior manifolds, while also enabling effective learning with only sparse rewards. Empirical evaluations on challenging manipulation benchmarks demonstrate that ExpertGen reliably produces high-quality expert policies with no reward engineering. On industrial assembly tasks, ExpertGen achieves a 90.5% overall success rate, while on long-horizon manipulation tasks it attains 85% overall success, outperforming all baseline methods. The resulting policies exhibit dexterous control and remain robust across diverse initial configurations and failure states. To validate sim-to-real transfer, the learned state-based expert policies are further distilled into visuomotor policies via DAgger and successfully deployed on real robotic hardware.
>
---
#### [new 028] Routing and Control for Marine Oil-Spill Cleanup with a Boom-Towing Vessel Fleet
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于海洋油污清理任务，解决多艘自主水面航行器协同作业问题。提出一种集成框架，优化路径并设计控制器，实现高效、风险感知的油污处理。**

- **链接: [https://arxiv.org/pdf/2603.16626](https://arxiv.org/pdf/2603.16626)**

> **作者:** Snir Carmeli; Adir Morgan; Kiril Solovey
>
> **摘要:** Marine oil spills damage ecosystems, contaminate coastlines, and disrupt food webs, while imposing substantial economic losses on fisheries and coastal communities. Prior work has demonstrated the feasibility of containing and cleaning individual spills using a duo of autonomous surface vehicles (ASVs) equipped with a towed boom and skimmers. However, existing algorithmic approaches primarily address isolated slicks and individual ASV duos, lacking scalable methods for coordinating large robotic fleets across multiple spills representative of realistic oil-spill incidents. In this work, we propose an integrated multi-robot framework for coordinated oil-spill confinement and cleanup using autonomous ASV duos. We formulate multi-spill response as a risk-weighted minimum-latency problem, where spill-specific risk factors and service times jointly determine cumulative environmental damage. To solve this problem, we develop a hybrid optimization approach combining mixed-integer linear programming, and a tailored warm-start heuristic, enabling near-optimal routing plans for scenarios with tens of spills within minutes on commodity hardware. For physical execution, we design and analyze two tracking controllers for boom-towing ASV duos: a feedback-linearization controller with proven asymptotic stability, and a baseline PID controller. Simulation results under coupled vessel-boom dynamics demonstrate accurate path tracking for both controllers. Together, these components provide a scalable, holistic framework for rapid, risk-aware multi-robot response to large-scale oil spill disasters.
>
---
#### [new 029] Conservative Offline Robot Policy Learning via Posterior-Transition Reweighting
- **分类: cs.RO**

- **简介: 该论文属于机器人策略学习任务，解决离线数据中异质性带来的适应问题。提出PTR方法，通过重加权提升对高质量样本的依赖，增强策略的保守性与适应性。**

- **链接: [https://arxiv.org/pdf/2603.16542](https://arxiv.org/pdf/2603.16542)**

> **作者:** Wanpeng Zhang; Hao Luo; Sipeng Zheng; Yicheng Feng; Haiweng Xu; Ziheng Xi; Chaoyi Xu; Haoqi Yuan; Zongqing Lu
>
> **摘要:** Offline post-training adapts a pretrained robot policy to a target dataset by supervised regression on recorded actions. In practice, robot datasets are heterogeneous: they mix embodiments, camera setups, and demonstrations of varying quality, so many trajectories reflect recovery behavior, inconsistent operator skill, or weakly informative supervision. Uniform post-training gives equal credit to all samples and can therefore average over conflicting or low-attribution data. We propose Posterior-Transition Reweighting (PTR), a reward-free and conservative post-training method that decides how much each training sample should influence the supervised update. For each sample, PTR encodes the observed post-action consequence as a latent target, inserts it into a candidate pool of mismatched targets, and uses a separate transition scorer to estimate a softmax identification posterior over target indices. The posterior-to-uniform ratio defines the PTR score, which is converted into a clipped-and-mixed weight and applied to the original action objective through self-normalized weighted regression. This construction requires no tractable policy likelihood and is compatible with both diffusion and flow-matching action heads. Rather than uniformly trusting all recorded supervision, PTR reallocates credit according to how attributable each sample's post-action consequence is under the current representation, improving conservative offline adaptation to heterogeneous robot data.
>
---
#### [new 030] Agile Interception of a Flying Target using Competitive Reinforcement Learning
- **分类: cs.RO; stat.ML**

- **简介: 该论文属于无人机拦截任务，解决如何用敏捷无人机拦截另一架敏捷无人机的问题。通过竞争强化学习训练策略，在仿真和真实环境中实现高效拦截。**

- **链接: [https://arxiv.org/pdf/2603.16279](https://arxiv.org/pdf/2603.16279)**

> **作者:** Timothée Gavin; Simon Lacroix; Murat Bronz
>
> **摘要:** This article presents a solution to intercept an agile drone by another agile drone carrying a catching net. We formulate the interception as a Competitive Reinforcement Learning problem, where the interceptor and the target drone are controlled by separate policies trained with Proximal Policy Optimization (PPO). We introduce a high-fidelity simulation environment that integrates a realistic quadrotor dynamics model and a low-level control architecture implemented in JAX, which allows for fast parallelized execution on GPUs. We train the agents using low-level control, collective thrust and body rates, to achieve agile flights both for the interceptor and the target. We compare the performance of the trained policies in terms of catch rate, time to catch, and crash rate, against common heuristic baselines and show that our solution outperforms these baselines for interception of agile targets. Finally, we demonstrate the performance of the trained policies in a scaled real-world scenario using agile drones inside an indoor flight arena.
>
---
#### [new 031] Kamino: GPU-based Massively Parallel Simulation of Multi-Body Systems with Challenging Topologies
- **分类: cs.RO**

- **简介: 该论文提出Kamino，一个基于GPU的多体系统并行仿真工具，解决复杂机械系统中耦合拓扑的仿真难题，支持高通量、高保真模拟。**

- **链接: [https://arxiv.org/pdf/2603.16536](https://arxiv.org/pdf/2603.16536)**

> **作者:** Vassilios Tsounis; Guirec Maloisel; Christian Schumacher; Ruben Grandia; Agon Serifi; David Müller; Chris Amevor; Tobias Widmer; Moritz Bächer
>
> **摘要:** We present Kamino, a GPU-based physics solver for massively parallel simulations of heterogeneous highly-coupled mechanical systems. Implemented in Python using NVIDIA Warp and integrated into the Newton framework, it enables the application of data-driven methods, such as large-scale reinforcement learning, to complex robotic systems that exhibit strongly coupled kinematic and dynamic constraints such as kinematic loops. The latter are often circumvented by practitioners; approximating the system topology as a kinematic tree and incorporating explicit loop-closure constraints or so-called mimic joints. Kamino aims at alleviating this burden by natively supporting these types of coupling. This capability facilitates high-throughput parallelized simulations that capture the true nature of mechanical systems that exploit closed kinematic chains for mechanical advantage. Moreover, Kamino supports heterogeneous worlds, allowing for batched simulation of structurally diverse robots on a single GPU. At its core lies a state-of-the-art constrained optimization algorithm that computes constraint forces by solving the constrained rigid multi-body forward dynamics transcribed as a nonlinear complementarity problem. This leads to high-fidelity simulations that can resolve contact dynamics without resorting to approximate models that simplify and/or convexify the problem. We demonstrate RL policy training on DR Legs, a biped with six nested kinematic loops, generating a feasible walking policy while simulating 4096 parallel environments on a single GPU.
>
---
#### [new 032] PanguMotion: Continuous Driving Motion Forecasting with Pangu Transformers
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶中的运动预测任务，旨在解决传统方法忽略时间连续性和历史关联的问题。通过引入Pangu-1B的Transformer模块，构建连续驾驶场景的预测框架。**

- **链接: [https://arxiv.org/pdf/2603.16196](https://arxiv.org/pdf/2603.16196)**

> **作者:** Quanhao Ren; Yicheng Li; Nan Song
>
> **摘要:** Motion forecasting is a core task in autonomous driving systems, aiming to accurately predict the future trajectories of surrounding agents to ensure driving safety. Existing methods typically process discrete driving scenes independently, neglecting the temporal continuity and historical context correlations inherent in real-world driving environments. This paper proposes PanguMotion, a motion forecasting framework for continuous driving scenarios that integrates Transformer blocks from the Pangu-1B large language model as feature enhancement modules into autonomous driving motion prediction architectures. We conduct experiments on the Argoverse 2 datasets processed by the RealMotion data reorganization strategy, transforming each independent scene into a continuous sequence to mimic real-world driving scenarios.
>
---
#### [new 033] Geometry-Aligned LLM Fine-Tuning for Sequential Narrow-Opening Planning
- **分类: cs.RO**

- **简介: 该论文研究机器人通过多个狭窄开口的运动规划问题，提出一种几何对齐的LLM微调框架，生成符合几何约束的路径序列，提升长时序规划能力。**

- **链接: [https://arxiv.org/pdf/2603.16028](https://arxiv.org/pdf/2603.16028)**

> **作者:** Al Jaber Mahmud; Xuan Wang
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** We study rigid-body motion planning through multiple sequential narrow openings, which requires long-horizon geometric reasoning because the configuration used to traverse an early opening constrains the set of reachable configurations for subsequent ones. To achieve this, we propose a geometry-aligned large language model (LLM) fine-tuning framework that generates fixed-length, machine-readable waypoint sequences that are both geometrically feasible and coordinated across openings. Our approach uses a bi-level training pipeline. First, we perform failure-driven LoRA supervised fine-tuning (SFT) on human demonstrations, which incorporates structured failure feedback to teach the model common failure modes and enforce the output format. Second, we refine the same LoRA adapters using Group Relative Policy Optimization (GRPO) with geometric verification: each sampled waypoint sequence is densified by a model-based planner and scored with a deterministic geometry-derived reward to achieve continuous-motion feasibility. To validate the effectiveness of our proposed method, we provide both quantitative and qualitative results from simulations. Our method achieves the highest success rate in both in-distribution and out-of-distribution environments and qualitatively exhibits long-horizon geometric reasoning by selecting exit poses that facilitate entry into subsequent openings.
>
---
#### [new 034] Scalable Inspection Planning via Flow-based Mixed Integer Linear Programming
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决大规模点检测路径优化问题。通过流式混合整数线性规划方法，提升求解效率与质量，实现高效、可扩展的检测路径规划。**

- **链接: [https://arxiv.org/pdf/2603.16593](https://arxiv.org/pdf/2603.16593)**

> **作者:** Adir Morgan; Kiril Solovey; Oren Salzman
>
> **摘要:** Inspection planning is concerned with computing the shortest robot path to inspect a given set of points of interest (POIs) using the robot's sensors. This problem arises in a wide range of applications from manufacturing to medical robotics. To alleviate the problem's complexity, recent methods rely on sampling-based methods to obtain a more manageable (discrete) graph inspection planning (GIP) problem. Unfortunately, GIP still remains highly difficult to solve at scale as it requires simultaneously satisfying POI-coverage and path-connectivity constraints, giving rise to a challenging optimization problem, particularly at scales encountered in real-world scenarios. In this work, we present highly scalable Mixed Integer Linear Programming (MILP) solutions for GIP that significantly advance the state-of-the-art in both runtime and solution quality. Our key insight is a reformulation of the problem's core constraints as a network flow, which enables effective MILP models and a specialized Branch-and-Cut solver that exploits the combinatorial structure of flows. We evaluate our approach on medical and infrastructure benchmarks alongside large-scale synthetic instances. Across all scenarios, our method produces substantially tighter lower bounds than existing formulations, reducing optimality gaps by 30-50% on large instances. Furthermore, our solver demonstrates unprecedented scalability: it provides non-trivial solutions for problems with up to 15,000 vertices and thousands of POIs, where prior state-of-the-art methods typically exhaust memory or fail to provide any meaningful optimality guarantees.
>
---
#### [new 035] Compact Optical Single-axis Joint Torque Sensor Using Redundant Photo-Reflectors and Quadratic-Programming Calibration
- **分类: cs.RO**

- **简介: 该论文属于机器人扭矩传感任务，旨在解决协作机器人低扭矩测量精度不足的问题。通过光学传感器和二次规划校准方法，提高扭矩检测的精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.16040](https://arxiv.org/pdf/2603.16040)**

> **作者:** Hyun-Bin Kim; Byeong-Il Ham; Kyung-Soo Kim
>
> **备注:** 10 pages
>
> **摘要:** This study proposes a non-contact photo-reflector-based joint torque sensor for precise joint-level torque control and safe physical interaction. Current-sensor-based torque estimation in many collaborative robots suffers from poor low-torque accuracy due to gearbox stiction/friction and current-torque nonlinearity, especially near static conditions. The proposed sensor optically measures micro-deformation of an elastic structure and employs a redundant array of photo-reflectors arranged in four directions to improve sensitivity and signal-to-noise ratio. We further present a quadratic-programming-based calibration method that exploits redundancy to suppress noise and enhance resolution compared to least-squares calibration. The sensor is implemented in a compact form factor (96 mm diameter, 12 mm thickness). Experiments demonstrate a maximum error of 0.083%FS and an RMS error of 0.0266 Nm for z-axis torque measurement. Calibration tests show that the proposed calibration achieves a 3 sigma resolution of 0.0224 Nm at 1 kHz without filtering, corresponding to a 2.14 times improvement over the least-squares baseline. Temperature chamber characterization and rational fitting based compensation mitigate zero drift induced by MCU self heating and motor heat. Motor-level validation via torque control and admittance control confirms improved low torque tracking and disturbance robustness relative to current-sensor-based control.
>
---
#### [new 036] A Pin-Array Structured Climbing Robot for Stable Locomotion on Steep Rocky Terrain
- **分类: cs.RO; cs.AR**

- **简介: 该论文属于爬壁机器人任务，旨在解决复杂地形稳定攀爬问题。设计了具有柔性针阵列的抓取机构，实现被动适应表面不规则，提升抓握稳定性。**

- **链接: [https://arxiv.org/pdf/2603.16543](https://arxiv.org/pdf/2603.16543)**

> **作者:** Keita Nagaoka; Kentaro Uno; Kazuya Yoshida
>
> **备注:** Author's version of a manuscript accepted at the 2026 IEEE International Conference on Robotics and Automation (ICRA). (c) IEEE
>
> **摘要:** Climbing robots face significant challenges when navigating unstructured environments, where reliable attachment to irregular surfaces is critical. We present a novel mobile climbing robot equipped with compliant pin-array structured grippers that passively conform to surface irregularities, ensuring stable ground gripping without the need for complicated sensing or control. Each pin features a vertically split design, combining an elastic element with a metal spine to enable mechanical interlocking with microscale surface features. Statistical modeling and experimental validation indicate that variability in individual pin forces and contact numbers are the primary sources of grasping uncertainty. The robot demonstrated robust and stable locomotion in indoor tests on inclined walls (10-30 degrees) and in outdoor tests on natural rocky terrain. This work highlights that a design emphasizing passive compliance and mechanical redundancy provides a practical and robust solution for real-world climbing robots while minimizing control complexity.
>
---
#### [new 037] DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人抓取任务，解决跨硬件灵巧抓取问题。通过构建形态对齐图网络，实现无需重新训练的零样本迁移抓取。**

- **链接: [https://arxiv.org/pdf/2603.16806](https://arxiv.org/pdf/2603.16806)**

> **作者:** Yuliang Wu; Yanhan Lin; WengKit Lao; Yuhao Lin; Yi-Lin Wei; Wei-Shi Zheng; Ancong Wu
>
> **摘要:** To meet the demands of increasingly diverse dexterous hand hardware, it is crucial to develop a policy that enables zero-shot cross-embodiment grasping without redundant re-learning. Cross-embodiment alignment is challenging due to heterogeneous hand kinematics and physical constraints. Existing approaches typically predict intermediate motion targets and retarget them to each embodiment, which may introduce errors and violate embodiment-specific limits, hindering transfer across diverse hands. To overcome these limitations, we propose \textit{DexGrasp-Zero}, a policy that learns universal grasping skills from diverse embodiments, enabling zero-shot transfer to unseen hands. We first introduce a morphology-aligned graph representation that maps each hand's kinematic keypoints to anatomically grounded nodes and equips each node with tri-axial orthogonal motion primitives, enabling structural and semantic alignment across different morphologies. Relying on this graph-based representation, we design a \textit{Morphology-Aligned Graph Convolutional Network} (MAGCN) to encode the graph for policy learning. MAGCN incorporates a \textit{Physical Property Injection} mechanism that fuses hand-specific physical constraints into the graph features, enabling adaptive compensation for varying link lengths and actuation limits for precise and stable grasping. Our extensive simulation evaluations on the YCB dataset demonstrate that our policy, jointly trained on four heterogeneous hands (Allegro, Shadow, Schunk, Ability), achieves an 85\% zero-shot success rate on unseen hardware (LEAP, Inspire), outperforming the state-of-the-art method by 59.5\%. Real-world experiments further evaluate our policy on three robot platforms (LEAP, Inspire, Revo2), achieving an 82\% average success rate on unseen objects.
>
---
#### [new 038] SignNav: Leveraging Signage for Semantic Visual Navigation in Large-Scale Indoor Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SignNav任务，解决大尺度室内环境中利用标识进行语义导航的问题。构建了LSI-Dataset，并提出START模型提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.16166](https://arxiv.org/pdf/2603.16166)**

> **作者:** Jian Sun; Yuming Huang; He Li; Shuqi Xiao; Shenyan Guo; Maani Ghaffari; Qingbiao Li; Chengzhong Xu; Hui Kong
>
> **摘要:** Humans routinely leverage semantic hints provided by signage to navigate to destinations within novel Large-Scale Indoor (LSI) environments, such as hospitals and airport terminals. However, this capability remains underexplored within the field of embodied navigation. This paper introduces a novel embodied navigation task, SignNav, which requires the agent to interpret semantic hint from signage and reason about the subsequent action based on current observation. To facilitate research in this domain, we construct the LSI-Dataset for the training and evaluation of various SignNav agents. Dynamically changing semantic hints and sparse placement of signage in LSI environments present significant challenges to the SignNav task. To address these challenges, we propose the Spatial-Temporal Aware Transformer (START) model for end-to-end decision-making. The spatial-aware module grounds the semantic hint of signage into physical world, while the temporal-aware module captures long-range dependencies between historical states and current observation. Leveraging a two-stage training strategy with Dataset Aggregation (DAgger), our approach achieves state-of-the-art performance, recording an 80% Success Rate (SR) and 0.74 NDTW on val-unseen split. Real-world deployment further demonstrates the practicality of our method in physical environment without pre-built map.
>
---
#### [new 039] Development of Low-Cost and Bidirectional Syringe Pumps for Soft Robotics Applications
- **分类: cs.RO**

- **简介: 该论文属于软体机器人技术领域，旨在解决传统气动驱动成本高、复杂度大的问题。研究开发了一种低成本、可双向泵气的模块化注射泵系统，提升软体机器人的控制精度与灵活性。**

- **链接: [https://arxiv.org/pdf/2603.16803](https://arxiv.org/pdf/2603.16803)**

> **作者:** Krishamsu Subedi Chhetri; Aryan Mayor; Elise Corbin; Logan Walker; John Rieffel
>
> **摘要:** Soft robotics leverages deformable materials to develop robots capable of navigating unstructured and dynamic environments. Silicone Voxel-Based Soft Robots (Silibots) are a type of pneumatically actuated soft robots that rely on the inflation and deflation of their voxels for shape-shifting behaviors. However, traditional pneumatic actuation methods (high pressure solenoids, medical diaphragm pumps, micro compressors, compressed fluid) pose significant challenges due to their limited efficacy, cost, complexity, or lack of precision. This work introduces a low cost and modular syringe pump system, constructed with off the shelf and 3D printed parts, designed to overcome these limitations. The syringe pump system also enhances actuation with the unique ability to pull a vacuum as well pump air into the soft robot. Furthermore, the syringe pump features modular hardware and customizable software, allowing for researchers to tailor the syringe pump to their requirements or operate multiple pumps simultaneously with unique pump parameters. This flexibility makes the syringe pump an accessible and scalable tool that paves the way for broader adoption of soft robotic technologies in research and education.
>
---
#### [new 040] ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace
- **分类: cs.RO**

- **简介: 该论文属于航空轨迹预测任务，旨在提升非塔台终端空域的通用航空安全。提出ASCENT模型，结合Transformer实现高效、准确的三维轨迹预测。**

- **链接: [https://arxiv.org/pdf/2603.16550](https://arxiv.org/pdf/2603.16550)**

> **作者:** Alexander Prutsch; David Schinagl; Horst Possegger
>
> **备注:** ICRA 2026. Project Page at this https URL
>
> **摘要:** Accurate trajectory prediction can improve General Aviation safety in non-towered terminal airspace, where high traffic density increases accident risk. We present ASCENT, a lightweight transformer-based model for multi-modal 3D aircraft trajectory forecasting, which integrates domain-aware 3D coordinate normalization and parameterized predictions. ASCENT employs a transformer-based motion encoder and a query-based decoder, enabling the generation of diverse maneuver hypotheses with low latency. Experiments on the TrajAir and TartanAviation datasets demonstrate that our model outperforms prior baselines, as the encoder effectively captures motion dynamics and the decoder aligns with structured aircraft traffic patterns. Furthermore, ablation studies confirm the contributions of the decoder design, coordinate-frame modeling, and parameterized outputs. These results establish ASCENT as an effective approach for real-time aircraft trajectory prediction in non-towered terminal airspace.
>
---
#### [new 041] The Era of End-to-End Autonomy: Transitioning from Rule-Based Driving to Large Driving Models
- **分类: cs.RO; cs.CV; eess.IV**

- **简介: 论文探讨自动驾驶从规则系统向端到端学习模型的转变，分析架构设计与安全问题，旨在推动更高效的自动驾驶技术发展。**

- **链接: [https://arxiv.org/pdf/2603.16050](https://arxiv.org/pdf/2603.16050)**

> **作者:** Eduardo Nebot; Julie Stephany Berrio Perez
>
> **摘要:** Autonomous driving is undergoing a shift from modular rule based pipelines toward end to end (E2E) learning systems. This paper examines this transition by tracing the evolution from classical sense perceive plan control architectures to large driving models (LDMs) capable of mapping raw sensor input directly to driving actions. We analyze recent developments including Tesla's Full Self Driving (FSD) V12 V14, Rivian's Unified Intelligence platform, NVIDIA Cosmos, and emerging commercial robotaxi deployments, focusing on architectural design, deployment strategies, safety considerations and industry implications. A key emerging product category is supervised E2E driving, often referred to as FSD (Supervised) or L2 plus plus, which several manufacturers plan to deploy from 2026 onwards. These systems can perform most of the Dynamic Driving Task (DDT) in complex environments while requiring human supervision, shifting the driver's role to safety oversight. Early operational evidence suggests E2E learning handles the long tail distribution of real world driving scenarios and is becoming a dominant commercial strategy. We also discuss how similar architectural advances may extend beyond autonomous vehicles (AV) to other embodied AI systems, including humanoid robotics.
>
---
#### [new 042] CABTO: Context-Aware Behavior Tree Grounding for Robot Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，解决BT接地问题，即自动构建完整一致的行为树系统。提出CABTO框架，利用预训练模型高效生成动作模型与控制策略。**

- **链接: [https://arxiv.org/pdf/2603.16809](https://arxiv.org/pdf/2603.16809)**

> **作者:** Yishuai Cai; Xinglin Chen; Yunxin Mao; Kun Hu; Minglong Li; Yaodong Yang; Yuanpei Chen
>
> **摘要:** Behavior Trees (BTs) offer a powerful paradigm for designing modular and reactive robot controllers. BT planning, an emerging field, provides theoretical guarantees for the automated generation of reliable BTs. However, BT planning typically assumes that a well-designed BT system is already grounded -- comprising high-level action models and low-level control policies -- which often requires extensive expert knowledge and manual effort. In this paper, we formalize the BT Grounding problem: the automated construction of a complete and consistent BT system. We analyze its complexity and introduce CABTO (Context-Aware Behavior Tree grOunding), the first framework to efficiently solve this challenge. CABTO leverages pre-trained Large Models (LMs) to heuristically search the space of action models and control policies, guided by contextual feedback from BT planners and environmental observations. Experiments spanning seven task sets across three distinct robotic manipulation scenarios demonstrate CABTO's effectiveness and efficiency in generating complete and consistent behavior tree systems.
>
---
#### [new 043] DreamPlan: Efficient Reinforcement Fine-Tuning of Vision-Language Planners via Video World Models
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言规划任务，旨在解决VLM在真实环境中物理理解不足导致的失败问题。通过视频世界模型和强化学习，无需大量真实数据即可提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2603.16860](https://arxiv.org/pdf/2603.16860)**

> **作者:** Emily Yue-Ting Jia; Weiduo Yuan; Tianheng Shi; Vitor Guizilini; Jiageng Mao; Yue Wang
>
> **摘要:** Robotic manipulation requires sophisticated commonsense reasoning, a capability naturally possessed by large-scale Vision-Language Models (VLMs). While VLMs show promise as zero-shot planners, their lack of grounded physical understanding often leads to compounding errors and low success rates when deployed in complex real-world environments, particularly for challenging tasks like deformable object manipulation. Although Reinforcement Learning (RL) can adapt these planners to specific task dynamics, directly fine-tuning VLMs via real-world interaction is prohibitively expensive, unsafe, and sample-inefficient. To overcome this bottleneck, we introduce DreamPlan, a novel framework for the reinforcement fine-tuning of VLM planners via video world models. Instead of relying on costly physical rollouts, DreamPlan first leverages the zero-shot VLM to collect exploratory interaction data. We demonstrate that this sub-optimal data is sufficient to train an action-conditioned video generation model, which implicitly captures complex real-world physics. Subsequently, the VLM planner is fine-tuned entirely within the "imagination" of this video world model using Odds Ratio Policy Optimization (ORPO). By utilizing these virtual rollouts, physical and task-specific knowledge is efficiently injected into the VLM. Our results indicate that DreamPlan bridges the gap between semantic reasoning and physical grounding, significantly improving manipulation success rates without the need for large-scale real-world data collection. Our project page is this https URL.
>
---
#### [new 044] Controlling Fish Schools via Reinforcement Learning of Virtual Fish Movement
- **分类: cs.RO; cs.LG; q-bio.PE**

- **简介: 该论文属于控制动物群体行为的任务，旨在通过虚拟鱼的强化学习引导真实鱼群。工作包括构建虚拟鱼模型、训练有效策略，并验证其在现实中的有效性。**

- **链接: [https://arxiv.org/pdf/2603.16384](https://arxiv.org/pdf/2603.16384)**

> **作者:** Yusuke Nishii; Hiroaki Kawashima
>
> **备注:** English translation of the author's 2018 bachelor's thesis. Keywords: fish schooling, reinforcement learning, collective behavior, artificial agents, swarm-machine interaction
>
> **摘要:** This study investigates a method to guide and control fish schools using virtual fish trained with reinforcement learning. We utilize 2D virtual fish displayed on a screen to overcome technical challenges such as durability and movement constraints inherent in physical robotic agents. To address the lack of detailed behavioral models for real fish, we adopt a model-free reinforcement learning approach. First, simulation results show that reinforcement learning can acquire effective movement policies even when simulated real fish frequently ignore the virtual stimulus. Second, real-world experiments with live fish confirm that the learned policy successfully guides fish schools toward specified target directions. Statistical analysis reveals that the proposed method significantly outperforms baseline conditions, including the absence of stimulus and a heuristic "stay-at-edge" strategy. This study provides an early demonstration of how reinforcement learning can be used to influence collective animal behavior through artificial agents.
>
---
#### [new 045] Safety Case Patterns for VLA-based driving systems: Insights from SimLingo
- **分类: cs.RO; cs.SE**

- **简介: 该论文属于自动驾驶安全领域，旨在解决VLA系统因引入自然语言输入而产生的安全隐患。提出RAISE方法构建安全案例，确保系统安全可靠。**

- **链接: [https://arxiv.org/pdf/2603.16013](https://arxiv.org/pdf/2603.16013)**

> **作者:** Gerhard Yu; Fuyuki Ishikawa; Oluwafemi Odu; Alvine Boaye Belle
>
> **摘要:** Vision-Language-Action (VLA)-based driving systems represent a significant paradigm shift in autonomous driving since, by combining traffic scene understanding, linguistic interpretation, and action generation, these systems enable more flexible, adaptive, and instruction-responsive driving behaviors. However, despite their growing adoption and potential to support socially responsible autonomous driving while understanding high-level human instructions, VLA-based driving systems may exhibit new types of hazardous behaviors. Such as the addition of natural language inputs (e.g., user or navigation instructions) into the multimodal control loop, which may lead to unpredictable and unsafe behaviors that could endanger vehicle occupants and pedestrians. Hence, assuring the safety of these systems is crucial to help build trust in their operations. To support this, we propose a novel safety case design approach called RAISE. Our approach introduces novel patterns tailored to instruction-based driving systems such as VLA-based driving systems, an extension of Hazard Analysis and Risk Assessment (HARA) detailing safe scenarios and their outcomes, and a design technique to create the safety cases of VLA-based driving systems. A case study on SimLingo illustrates how our approach can be used to construct rigorous, evidence-based safety claims for this emerging class of autonomous driving systems.
>
---
#### [new 046] Encoding Predictability and Legibility for Style-Conditioned Diffusion Policy
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于人机协作任务，解决运动效率与可读性之间的平衡问题。提出SCDP框架，根据环境配置调整轨迹生成，提升模糊场景的可读性，同时保持高效路径。**

- **链接: [https://arxiv.org/pdf/2603.16368](https://arxiv.org/pdf/2603.16368)**

> **作者:** Adrien Jacquet Crétides; Mouad Abrini; Hamed Rahimi; Mohamed Chetouani
>
> **备注:** Submitted to the 18th International Conference on Social Robotics (ICSR 2026)
>
> **摘要:** Striking a balance between efficiency and transparent motion is a core challenge in human-robot collaboration, as highly expressive movements often incur unnecessary time and energy costs. In collaborative environments, legibility allows a human observer a better understanding of the robot's actions, increasing safety and trust. However, these behaviors result in sub-optimal and exaggerated trajectories that are redundant in low-ambiguity scenarios where the robot's goal is already obvious. To address this trade-off, we propose Style-Conditioned Diffusion Policy (SCDP), a modular framework that constrains the trajectory generation of a pre-trained diffusion model toward either legibility or efficiency based on the environment's configuration. Our method utilizes a post-training pipeline that freezes the base policy and trains a lightweight scene encoder and conditioning predictor to modulate the diffusion process. At inference time, an ambiguity detection module activates the appropriate conditioning, prioritizing expressive motion only for ambiguous goals and reverting to efficient paths otherwise. We evaluate SCDP on manipulation and navigation tasks, and results show that it enhances legibility in ambiguous settings while preserving optimal efficiency when legibility is unnecessary, all without retraining the base policy.
>
---
#### [new 047] Gaze-Aware Task Progression Detection Framework for Human-Robot Interaction Using RGB Cameras
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决无需专用设备的 gaze 检测问题。通过 RGB 相机实现任务进展检测，提升交互自然性与舒适度。**

- **链接: [https://arxiv.org/pdf/2603.15951](https://arxiv.org/pdf/2603.15951)**

> **作者:** Linlin Cheng; Koen Hindriks; Artem V. Belopolsky
>
> **备注:** 9 pages, 7 figures. This article has been accepted for publication in IEEE Robotics and Automation Letters
>
> **摘要:** In human-robot interaction (HRI), detecting a human's gaze helps robots interpret user attention and intent. However, most gaze detection approaches rely on specialized eye-tracking hardware, limiting deployment in everyday settings. Appearance-based gaze estimation methods remove this dependency by using standard RGB cameras, but their practicality in HRI remains underexplored. We present a calibration-free framework for detecting task progression when information is conveyed via integrated display interfaces. The framework uses only the robot's built-in monocular RGB camera (640x480 resolution) and state-of-the-art gaze estimation to monitor attention patterns. It leverages natural behavior, where users shift focus from task interfaces to the robot's face to signal task completion, formalized through three Areas of Interest (AOI): tablet, robot face, and elsewhere. Systematic parameter optimization identifies configurations that balance detection accuracy and interaction latency. We validate our framework in a "First Day at Work" scenario, comparing it to button-based interaction. Results show a task completion detection accuracy of 77.6%. Compared to button-based interaction, the proposed system exhibits slightly higher response latency but preserves information retention and significantly improves comfort, social presence, and perceived naturalness. Notably, most participants reported that they did not consciously use eye movements to guide the interaction, underscoring the intuitive role of gaze as a communicative cue. This work demonstrates the feasibility of intuitive, low-cost, RGB-only gaze-based HRI for natural and engaging interactions.
>
---
#### [new 048] Real-Time Decoding of Movement Onset and Offset for Brain-Controlled Rehabilitation Exoskeleton
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于脑机接口任务，旨在解决康复外骨骼的实时运动启停控制问题。通过EEG实现运动意图的准确识别，提升康复训练的精准性和有效性。**

- **链接: [https://arxiv.org/pdf/2603.16825](https://arxiv.org/pdf/2603.16825)**

> **作者:** Kanishka Mitra; Satyam Kumar; Frigyes Samuel Racz; Deland Liu; Ashish D. Deshpande; José del R. Millán
>
> **备注:** Accepted to ICRA 2026. 8 pages, 5 figures. Project page available at this https URL
>
> **摘要:** Robot-assisted therapy can deliver high-dose, task-specific training after neurologic injury, but most systems act primarily at the limb level-engaging the impaired neural circuits only indirectly-which remains a key barrier to truly contingent, neuroplasticity-targeted rehabilitation. We address this gap by implementing online, dual-state motor imagery control of an upper-limb exoskeleton, enabling goal-directed reaches to be both initiated and terminated directly from non-invasive EEG. Eight participants used EEG to initiate assistance and then volitionally halt the robot mid-trajectory. Across two online sessions, group-mean hit rates were 61.5% for onset and 64.5% for offset, demonstrating reliable start-stop command delivery despite instrumental noise and passive arm motion. Methodologically, we reveal a systematic, class-driven bias induced by common task-based recentering using an asymmetric margin diagnostic, and we introduce a class-agnostic fixation-based recentering method that tracks drift without sampling command classes while preserving class geometry. This substantially improves threshold-free separability (AUC gains: onset +56%, p = 0.0117; offset +34%, p = 0.0251) and reduces bias within and across days. Together, these results help bridge offline decoding and practical, intention-driven start-stop control of a rehabilitation exoskeleton, enabling precisely timed, contingent assistance aligned with neuroplasticity goals while supporting future clinical translation.
>
---
#### [new 049] LIMBERO: A Limbed Climbing Exploration Robot Toward Traveling on Rocky Cliffs
- **分类: cs.RO**

- **简介: 该论文属于机器人攀爬任务，旨在解决在崎岖岩石悬崖上稳定移动的问题。研究提出LIMBERO机器人，采用新型抓取器和算法，实现高效攀爬。**

- **链接: [https://arxiv.org/pdf/2603.16531](https://arxiv.org/pdf/2603.16531)**

> **作者:** Kentaro Uno; Masazumi Imai; Kazuki Takada; Teruhiro Kataonami; Yudai Matsuura; Antonin Ringeval-Meusnier; Keita Nagaoka; Mikio Eguchi; Ryo Nishibe; Kazuya Yoshida
>
> **备注:** Author's version of a manuscript accepted at the 2026 IEEE International Conference on Robotics and Automation (ICRA). (c) IEEE
>
> **摘要:** In lunar and planetary exploration, legged robots have attracted significant attention as an alternative to conventional wheeled robots, which struggle to traverse rough and uneven terrain. To enable locomotion over highly irregular and steeply inclined surfaces, limbed climbing robots equipped with grippers on their feet have emerged as a promising solution. In this paper, we present LIMBERO, a 10 kg-class quadrupedal climbing robot that employs spine-type grippers for stable locomotion and climbing on rugged and steep terrain. We first introduce a novel gripper design featuring coupled finger-closing and spine-hooking motions, tightly actuated by a single motor, which achieves exceptional grasping performance (>150 N) despite its lightweight design (525 g). Furthermore, we develop an efficient algorithm to visualize a geometry-based graspability index on continuous rough terrain. Finally, we integrate these components into LIMBERO and demonstrate its ability to ascend steep rocky surfaces under a 1 G gravity condition, a performance not previously achieved yet for limbed climbing robots of this scale.
>
---
#### [new 050] When Should a Robot Think? Resource-Aware Reasoning via Reinforcement Learning for Embodied Robotic Decision-Making
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人决策任务，解决 embodied 代理在推理与行动间的资源分配问题。提出 RARRL 框架，通过强化学习实现自适应推理控制，提升任务成功率与系统效率。**

- **链接: [https://arxiv.org/pdf/2603.16673](https://arxiv.org/pdf/2603.16673)**

> **作者:** Jun Liu; Pu Zhao; Zhenglun Kong; Xuan Shen; Peiyan Dong; Fan Yang; Lin Cui; Hao Tang; Geng Yuan; Wei Niu; Wenbin Zhang; Xue Lin; Gaowen Liu; Yanzhi Wang; Dong Huang
>
> **摘要:** Embodied robotic systems increasingly rely on large language model (LLM)-based agents to support high-level reasoning, planning, and decision-making during interactions with the environment. However, invoking LLM reasoning introduces substantial computational latency and resource overhead, which can interrupt action execution and reduce system reliability. Excessive reasoning may delay actions, while insufficient reasoning often leads to incorrect decisions and task failures. This raises a fundamental question for embodied agents: when should the agent reason, and when should it act? In this work, we propose RARRL (Resource-Aware Reasoning via Reinforcement Learning), a hierarchical framework for resource-aware orchestration of embodied agents. Rather than learning low-level control policies, RARRL learns a high-level orchestration policy that operates at the agent's decision-making layer. This policy enables the agent to adaptively determine whether to invoke reasoning, which reasoning role to employ, and how much computational budget to allocate based on current observations, execution history, and remaining resources. Extensive experiments, including evaluations with empirical latency profiles derived from the ALFRED benchmark, show that RARRL consistently improves task success rates while reducing execution latency and enhancing robustness compared with fixed or heuristic reasoning strategies. These results demonstrate that adaptive reasoning control is essential for building reliable and efficient embodied robotic agents.
>
---
#### [new 051] CorrectionPlanner: Self-Correction Planner with Reinforcement Learning in Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主驾驶任务，解决学习型规划器缺乏自我纠正的问题。提出CorrectionPlanner，通过自回归生成和强化学习实现安全路径规划。**

- **链接: [https://arxiv.org/pdf/2603.15771](https://arxiv.org/pdf/2603.15771)**

> **作者:** Yihong Guo; Dongqiangzi Ye; Sijia Chen; Anqi Liu; Xianming Liu
>
> **摘要:** Autonomous driving requires safe planning, but most learning-based planners lack explicit self-correction ability: once an unsafe action is proposed, there is no mechanism to correct it. Thus, we propose CorrectionPlanner, an autoregressive planner with self-correction that models planning as motion-token generation within a propose, evaluate, and correct loop. At each planning step, the policy proposes an action, namely a motion token, and a learned collision critic predicts whether it will induce a collision within a short horizon. If the critic predicts a collision, we retain the sequence of historical unsafe motion tokens as a self-correction trace, generate the next motion token conditioned on it, and repeat this process until a safe motion token is proposed or the safety criterion is met. This self-correction trace, consisting of all unsafe motion tokens, represents the planner's correction process in motion-token space, analogous to a reasoning trace in language models. We train the planner with imitation learning followed by model-based reinforcement learning using rollouts from a pretrained world model that realistically models agents' reactive behaviors. Closed-loop evaluations show that CorrectionPlanner reduces collision rate by over 20% on Waymax and achieves state-of-the-art planning scores on nuPlan.
>
---
#### [new 052] Enforcing Task-Specified Compliance Bounds for Humanoids via Anisotropic Lipschitz-Constrained Policies
- **分类: cs.RO**

- **简介: 该论文属于人形机器人控制任务，旨在解决RL中难以实现任务指定的柔顺性问题。提出ALCP方法，通过状态依赖的Lipschitz约束实现方向性柔顺控制，提升运动稳定性与抗冲击能力。**

- **链接: [https://arxiv.org/pdf/2603.16180](https://arxiv.org/pdf/2603.16180)**

> **作者:** Zewen He; Yoshihiko Nakamura
>
> **备注:** Submitted to IEEE for possible publication, under review
>
> **摘要:** Reinforcement learning (RL) has demonstrated substantial potential for humanoid bipedal locomotion and the control of complex motions. To cope with oscillations and impacts induced by environmental interactions, compliant control is widely regarded as an effective remedy. However, the model-free nature of RL makes it difficult to impose task-specified and quantitatively verifiable compliance objectives, and classical model-based stiffness designs are not directly applicable. Lipschitz-Constrained Policies (LCP), which regularize the local sensitivity of a policy via gradient penalties, have recently been used to smooth humanoid motions. Nevertheless, existing LCP-based methods typically employ a single scalar Lipschitz budget and lack an explicit connection to physically meaningful compliance specifications in real-world systems. In this study, we propose an anisotropic Lipschitz-constrained policy (ALCP) that maps a task-space stiffness upper bound to a state-dependent Lipschitz-style constraint on the policy Jacobian. The resulting constraint is enforced during RL training via a hinge-squared spectral-norm penalty, preserving physical interpretability while enabling direction-dependent compliance. Experiments on humanoid robots show that ALCP improves locomotion stability and impact robustness, while reducing oscillations and energy usage.
>
---
#### [new 053] You've Got a Golden Ticket: Improving Generative Robot Policies With A Single Noise Vector
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在提升预训练生成式机器人策略的性能。通过引入固定的初始噪声向量（黄金票），在不改变策略的情况下优化任务表现。**

- **链接: [https://arxiv.org/pdf/2603.15757](https://arxiv.org/pdf/2603.15757)**

> **作者:** Omkar Patil; Ondrej Biza; Thomas Weng; Karl Schmeckpeper; Wil Thomason; Xiaohan Zhang; Robin Walters; Nakul Gopalan; Sebastian Castro; Eric Rosen
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** What happens when a pretrained generative robot policy is provided a constant initial noise as input, rather than repeatedly sampling it from a Gaussian? We demonstrate that the performance of a pretrained, frozen diffusion or flow matching policy can be improved with respect to a downstream reward by swapping the sampling of initial noise from the prior distribution (typically isotropic Gaussian) with a well-chosen, constant initial noise input -- a golden ticket. We propose a search method to find golden tickets using Monte-Carlo policy evaluation that keeps the pretrained policy frozen, does not train any new networks, and is applicable to all diffusion/flow matching policies (and therefore many VLAs). Our approach to policy improvement makes no assumptions beyond being able to inject initial noise into the policy and calculate (sparse) task rewards of episode rollouts, making it deployable with no additional infrastructure or models. Our method improves the performance of policies in 38 out of 43 tasks across simulated and real-world robot manipulation benchmarks, with relative improvements in success rate by up to 58% for some simulated tasks, and 60% within 50 search episodes for real-world tasks. We also show unique benefits of golden tickets for multi-task settings: the diversity of behaviors from different tickets naturally defines a Pareto frontier for balancing different objectives (e.g., speed, success rates); in VLAs, we find that a golden ticket optimized for one task can also boost performance in other related tasks. We release a codebase with pretrained policies and golden tickets for simulation benchmarks using VLAs, diffusion policies, and flow matching policies.
>
---
#### [new 054] Towards the Vision-Sound-Language-Action Paradigm: The HEAR Framework for Sound-Centric Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.SD**

- **简介: 该论文提出HEAR框架，解决实时声音驱动的交互任务，通过整合视觉、声音、语言和本体感知，实现连续音频感知与动作生成。**

- **链接: [https://arxiv.org/pdf/2603.16086](https://arxiv.org/pdf/2603.16086)**

> **作者:** Chang Nie; Tianchen Deng; Guangming Wang; Zhe Liu; Hesheng Wang
>
> **摘要:** While recent Vision-Language-Action (VLA) models have begun to incorporate audio, they typically treat sound as static pre-execution prompts or focus exclusively on human speech. This leaves a significant gap in real-time, sound-centric manipulation where fleeting environmental acoustics provide critical state verification during task execution. Consequently, key sounds are easily missed due to low-frequency updates or system latency. This problem is exacerbated by action chunking with open-loop execution, which creates a Blind Execution Interval where acoustic events are lost between discrete audio observation windows. Recognizing the necessity of continuous auditory awareness, we formalize Vision-Sound-Language-Action (VSLA) as a continuous control paradigm conditioned on vision, streaming audio, language, and proprioception under delayed decision loops. As an instantiation, we introduce HEAR, a VSLA framework integrating four components: (i) a streaming Historizer to maintain a compact, causal audio context across execution gaps; (ii) an Envisioner adapted from omni foundation models to reason over multi-sensory inputs; (iii) an Advancer, formulated as an audio world model, to learn temporal dynamics by predicting near-future audio codes; and (iv) a flow-matching Realizer policy to generate smooth action chunks. To address the scarcity of pretraining data and evaluations for VSLA, we construct OpenX-Sound for pretraining, alongside HEAR-Bench, the first sound-centric manipulation benchmark with strict causal timing rules. Our results suggest that robust sound-centric manipulation necessitates causal persistence and explicit temporal learning. This framework provides a practical step toward multi-sensory foundation models for embodied agents, enabling robots to perceive and interact with dynamic environments. Code and videos are available at this https URL.
>
---
#### [new 055] Large Reward Models: Generalizable Online Robot Reward Generation with Vision-Language Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人强化学习任务，解决奖励函数设计困难的问题。通过视觉语言模型生成在线奖励信号，提升机器人操作的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2603.16065](https://arxiv.org/pdf/2603.16065)**

> **作者:** Yanru Wu; Weiduo Yuan; Ang Qi; Vitor Guizilini; Jiageng Mao; Yue Wang
>
> **摘要:** Reinforcement Learning (RL) has shown great potential in refining robotic manipulation policies, yet its efficacy remains strongly bottlenecked by the difficulty of designing generalizable reward functions. In this paper, we propose a framework for online policy refinement by adapting foundation VLMs into online reward generators. We develop a robust, scalable reward model based on a state-of-the-art VLM, trained on a large-scale, multi-source dataset encompassing real-world robot trajectories, human-object interactions, and diverse simulated environments. Unlike prior approaches that evaluate entire trajectories post-hoc, our method leverages the VLM to formulate a multifaceted reward signal comprising process, completion, and temporal contrastive rewards based on current visual observations. Initializing with a base policy trained via Imitation Learning (IL), we employ these VLM rewards to guide the model to correct sub-optimal behaviors in a closed-loop manner. We evaluate our framework on challenging long-horizon manipulation benchmarks requiring sequential execution and precise control. Crucially, our reward model operates in a purely zero-shot manner within these test environments. Experimental results demonstrate that our method significantly improves the success rate of the initial IL policy within just 30 RL iterations, demonstrating remarkable sample efficiency. This empirical evidence highlights that VLM-generated signals can provide reliable feedback to resolve execution errors, effectively eliminating the need for manual reward engineering and facilitating efficient online refinement for robot learning.
>
---
#### [new 056] Kinema4D: Kinematic 4D World Modeling for Spatiotemporal Embodied Simulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Kinema4D，解决机器人与环境交互的4D建模问题，通过精确控制和生成环境反应，实现物理真实、几何一致的模拟。**

- **链接: [https://arxiv.org/pdf/2603.16669](https://arxiv.org/pdf/2603.16669)**

> **作者:** Mutian Xu; Tianbao Zhang; Tianqi Liu; Zhaoxi Chen; Xiaoguang Han; Ziwei Liu
>
> **备注:** Project page: this https URL
>
> **摘要:** Simulating robot-world interactions is a cornerstone of Embodied AI. Recently, a few works have shown promise in leveraging video generations to transcend the rigid visual/physical constraints of traditional simulators. However, they primarily operate in 2D space or are guided by static environmental cues, ignoring the fundamental reality that robot-world interactions are inherently 4D spatiotemporal events that require precise interactive modeling. To restore this 4D essence while ensuring the precise robot control, we introduce Kinema4D, a new action-conditioned 4D generative robotic simulator that disentangles the robot-world interaction into: i) Precise 4D representation of robot controls: we drive a URDF-based 3D robot via kinematics, producing a precise 4D robot control trajectory. ii) Generative 4D modeling of environmental reactions: we project the 4D robot trajectory into a pointmap as a spatiotemporal visual signal, controlling the generative model to synthesize complex environments' reactive dynamics into synchronized RGB/pointmap sequences. To facilitate training, we curated a large-scale dataset called Robo4D-200k, comprising 201,426 robot interaction episodes with high-quality 4D annotations. Extensive experiments demonstrate that our method effectively simulates physically-plausible, geometry-consistent, and embodiment-agnostic interactions that faithfully mirror diverse real-world dynamics. For the first time, it shows potential zero-shot transfer capability, providing a high-fidelity foundation for advancing next-generation embodied simulation.
>
---
#### [new 057] S-VAM: Shortcut Video-Action Model by Self-Distilling Geometric and Semantic Foresight
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视频动作建模任务，旨在解决传统方法无法同时实现实时推理与高保真预测的问题。提出S-VAM模型，通过单次前向传递和自蒸馏策略提升效率与精度。**

- **链接: [https://arxiv.org/pdf/2603.16195](https://arxiv.org/pdf/2603.16195)**

> **作者:** Haodong Yan; Zhide Zhong; Jiaguan Zhu; Junjie He; Weilin Yuan; Wenxuan Song; Xin Gong; Yingjie Cai; Guanyi Zhao; Xu Yan; Bingbing Liu; Ying-Cong Chen; Haoang Li
>
> **摘要:** Video action models (VAMs) have emerged as a promising paradigm for robot learning, owing to their powerful visual foresight for complex manipulation tasks. However, current VAMs, typically relying on either slow multi-step video generation or noisy one-step feature extraction, cannot simultaneously guarantee real-time inference and high-fidelity foresight. To address this limitation, we propose S-VAM, a shortcut video-action model that foresees coherent geometric and semantic representations via a single forward pass. Serving as a stable blueprint, these foreseen representations significantly simplify the action prediction. To enable this efficient shortcut, we introduce a novel self-distillation strategy that condenses structured generative priors of multi-step denoising into one-step inference. Specifically, vision foundation model (VFM) representations extracted from the diffusion model's own multi-step generated videos provide teacher targets. Lightweight decouplers, as students, learn to directly map noisy one-step features to these targets. Extensive experiments in simulation and the real world demonstrate that our S-VAM outperforms state-of-the-art methods, enabling efficient and precise manipulation in complex environments. Our project page is this https URL
>
---
#### [new 058] FEEL (Force-Enhanced Egocentric Learning): A Dataset for Physical Action Understanding
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出FEEL数据集，结合力传感器与第一视角视频，解决物理动作理解问题。通过力信息提升接触理解和动作表征学习，无需人工标注即可取得优异效果。**

- **链接: [https://arxiv.org/pdf/2603.15847](https://arxiv.org/pdf/2603.15847)**

> **作者:** Eadom Dessalene; Botao He; Michael Maynord; Yonatan Tussa; Pavan Mantripragada; Yianni Karabati; Nirupam Roy; Yiannis Aloimonos
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** We introduce FEEL (Force-Enhanced Egocentric Learning), the first large-scale dataset pairing force measurements gathered from custom piezoresistive gloves with egocentric video. Our gloves enable scalable data collection, and FEEL contains approximately 3 million force-synchronized frames of natural unscripted manipulation in kitchen environments, with 45% of frames involving hand-object contact. Because force is the underlying cause that drives physical interaction, it is a critical primitive for physical action understanding. We demonstrate the utility of force for physical action understanding through application of FEEL to two families of tasks: (1) contact understanding, where we jointly perform temporal contact segmentation and pixel-level contacted object segmentation; and, (2) action representation learning, where force prediction serves as a self-supervised pretraining objective for video backbones. We achieve state-of-the-art temporal contact segmentation results and competitive pixel-level segmentation results without any need for manual contacted object segmentation annotations. Furthermore we demonstrate that action representation learning with FEEL improves transfer performance on action understanding tasks without any manual labels over EPIC-Kitchens, SomethingSomething-V2, EgoExo4D and Meccano.
>
---
#### [new 059] Resilience Meets Autonomy: Governing Embodied AI in Critical Infrastructure
- **分类: cs.AI; cs.RO**

- **简介: 论文探讨如何在关键基础设施中有效治理具身AI，解决其在危机中的脆弱性问题。提出基于任务复杂度的混合治理架构，结合机器能力与人类判断。**

- **链接: [https://arxiv.org/pdf/2603.15885](https://arxiv.org/pdf/2603.15885)**

> **作者:** Puneet Sharma; Christer Henrik Pursiainen
>
> **备注:** 6 pages
>
> **摘要:** Critical infrastructure increasingly incorporates embodied AI for monitoring, predictive maintenance, and decision support. However, AI systems designed to handle statistically representable uncertainty struggle with cascading failures and crisis dynamics that exceed their training assumptions. This paper argues that Embodied AIs resilience depends on bounded autonomy within a hybrid governance architecture. We outline four oversight modes and map them to critical infrastructure sectors based on task complexity, risk level, and consequence severity. Drawing on the EU AI Act, ISO safety standards, and crisis management research, we argue that effective governance requires a structured allocation of machine capability and human judgement.
>
---
#### [new 060] FlatLands: Generative Floormap Completion From a Single Egocentric View
- **分类: cs.CV; cs.AI; cs.RO; eess.IV**

- **简介: 该论文提出FlatLands任务，解决单视角下室内地面完整地图生成问题，通过数据集和基准测试评估不同方法，实现端到端的RGB到地面图转换。**

- **链接: [https://arxiv.org/pdf/2603.16016](https://arxiv.org/pdf/2603.16016)**

> **作者:** Subhransu S. Bhattacharjee; Dylan Campbell; Rahul Shome
>
> **备注:** Under review
>
> **摘要:** A single egocentric image typically captures only a small portion of the floor, yet a complete metric traversability map of the surroundings would better serve applications such as indoor navigation. We introduce FlatLands, a dataset and benchmark for single-view bird's-eye view (BEV) floor completion. The dataset contains 270,575 observations from 17,656 real metric indoor scenes drawn from six existing datasets, with aligned observation, visibility, validity, and ground-truth BEV maps, and the benchmark includes both in- and out-of-distribution evaluation protocols. We compare training-free approaches, deterministic models, ensembles, and stochastic generative models. Finally, we instantiate the task as an end-to-end monocular RGB-to-floormaps pipeline. FlatLands provides a rigorous testbed for uncertainty-aware indoor mapping and generative completion for embodied navigation.
>
---
#### [new 061] Designing for Disagreement: Front-End Guardrails for Assistance Allocation in LLM-Enabled Robots
- **分类: cs.AI; cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，解决LLM机器人在多人协助分配中的价值冲突与不确定性问题。提出一种带有可争议性的前端校准机制，确保优先级决策透明且可调整。**

- **链接: [https://arxiv.org/pdf/2603.16537](https://arxiv.org/pdf/2603.16537)**

> **作者:** Carmen Ng
>
> **备注:** Accepted at the Proceedings of the CHI 2026 Workshop: Ethics at the Front-End
>
> **摘要:** LLM-enabled robots prioritizing scarce assistance in social settings face pluralistic values and LLM behavioral variability: reasonable people can disagree about who is helped first, while LLM-mediated interaction policies vary across prompts, contexts, and groups in ways that are difficult to anticipate or verify at contact point. Yet user-facing guardrails for real-time, multi-user assistance allocation remain under-specified. We propose bounded calibration with contestability, a procedural front-end pattern that (i) constrains prioritization to a governance-approved menu of admissible modes, (ii) keeps the active mode legible in interaction-relevant terms at the point of deferral, and (iii) provides an outcome-specific contest pathway without renegotiating the global rule. Treating pluralism and LLM uncertainty as standing conditions, the pattern avoids both silent defaults that hide implicit value skews and wide-open user-configurable "value settings" that shift burden under time pressure. We illustrate the pattern with a public-concourse robot vignette and outline an evaluation agenda centered on legibility, procedural legitimacy, and actionability, including risks of automation bias and uneven usability of contest channels.
>
---
#### [new 062] MessyKitchens: Contact-rich object-level 3D scene reconstruction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于3D场景重建任务，旨在解决复杂场景中物体的精确重建与物理合理性问题。提出MessyKitchens数据集和多物体解码器MOD，提升物体间接触与穿透的准确性。**

- **链接: [https://arxiv.org/pdf/2603.16868](https://arxiv.org/pdf/2603.16868)**

> **作者:** Junaid Ahmed Ansari; Ran Ding; Fabio Pizzati; Ivan Laptev
>
> **摘要:** Monocular 3D scene reconstruction has recently seen significant progress. Powered by the modern neural architectures and large-scale data, recent methods achieve high performance in depth estimation from a single image. Meanwhile, reconstructing and decomposing common scenes into individual 3D objects remains a hard challenge due to the large variety of objects, frequent occlusions and complex object relations. Notably, beyond shape and pose estimation of individual objects, applications in robotics and animation require physically-plausible scene reconstruction where objects obey physical principles of non-penetration and realistic contacts. In this work we advance object-level scene reconstruction along two directions. First, we introduceMessyKitchens, a new dataset with real-world scenes featuring cluttered environments and providing high-fidelity object-level ground truth in terms of 3D object shapes, poses and accurate object contacts. Second, we build on the recent SAM 3D approach for single-object reconstruction and extend it with Multi-Object Decoder (MOD) for joint object-level scene reconstruction. To validate our contributions, we demonstrate MessyKitchens to significantly improve previous datasets in registration accuracy and inter-object penetration. We also compare our multi-object reconstruction approach on three datasets and demonstrate consistent and significant improvements of MOD over the state of the art. Our new benchmark, code and pre-trained models will become publicly available on our project website: this https URL.
>
---
#### [new 063] Featurized Occupation Measures for Structured Global Search in Numerical Optimal Control
- **分类: math.OC; cs.RO; eess.SY**

- **简介: 该论文属于数值最优控制领域，解决全局搜索与局部优化的矛盾。提出Featurized Occupation Measure（FOM），统一轨迹搜索与全局HJB验证，实现高效且可靠的控制优化。**

- **链接: [https://arxiv.org/pdf/2603.16231](https://arxiv.org/pdf/2603.16231)**

> **作者:** Qi Wei; Jianfeng Tao; Haoyang Tan; Hongyu Nie
>
> **摘要:** Numerical optimal control is commonly divided between globally structured but dimensionally intractable Hamilton-Jacobi-Bellman (HJB) methods and scalable but local trajectory optimization. We introduce the Featurized Occupation Measure (FOM), a finite-dimensional primal-dual interface for the occupation-measure formulation that unifies trajectory search and global HJB-type certification. FOM is broad yet numerically tractable, covering both explicit weak-form schemes and implicit simulator- or rollout-based sampling methods. Within this framework, approximate HJB subsolutions serve as intrinsic numerical certificates to directly evaluate and guide the primal search. We prove asymptotic consistency with the exact infinite-dimensional occupation-measure problem, and show that for block-organized feasible certificates, finite-dimensional approximation preserves certified lower bounds with blockwise error and complexity control. We also establish persistence of these lower bounds under time shifts and bounded model perturbations. Consequently, these structural properties render global certificates into flexible, reusable computational objects, establishing a systematic basis for certificate-guided optimization in nonlinear control.
>
---
#### [new 064] Regularized Latent Dynamics Prediction is a Strong Baseline For Behavioral Foundation Models
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于强化学习领域，解决零样本RL中状态特征表达问题。通过引入正则化机制，提升特征多样性，实现更优的模型表现。**

- **链接: [https://arxiv.org/pdf/2603.15857](https://arxiv.org/pdf/2603.15857)**

> **作者:** Pranaya Jajoo; Harshit Sikchi; Siddhant Agarwal; Amy Zhang; Scott Niekum; Martha White
>
> **备注:** ICLR 2026
>
> **摘要:** Behavioral Foundation Models (BFMs) produce agents with the capability to adapt to any unknown reward or task. These methods, however, are only able to produce near-optimal policies for the reward functions that are in the span of some pre-existing state features, making the choice of state features crucial to the expressivity of the BFM. As a result, BFMs are trained using a variety of complex objectives and require sufficient dataset coverage, to train task-useful spanning features. In this work, we examine the question: are these complex representation learning objectives necessary for zero-shot RL? Specifically, we revisit the objective of self-supervised next-state prediction in latent space for state feature learning, but observe that such an objective alone is prone to increasing state-feature similarity, and subsequently reducing span. We propose an approach, Regularized Latent Dynamics Prediction (RLDP), that adds a simple orthogonality regularization to maintain feature diversity and can match or surpass state-of-the-art complex representation learning methods for zero-shot RL. Furthermore, we empirically show that prior approaches perform poorly in low-coverage scenarios where RLDP still succeeds.
>
---
#### [new 065] Exploring the Use of VLMs for Navigation Assistance for People with Blindness and Low Vision
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于辅助导航任务，旨在探索视觉语言模型（VLMs）在帮助视障人士导航中的应用。工作包括评估多种VLMs的视觉技能和导航表现，分析其优缺点，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2603.15624](https://arxiv.org/pdf/2603.15624)**

> **作者:** Yu Li; Yuchen Zheng; Giles Hamilton-Fletcher; Marco Mezzavilla; Yao Wang; Sundeep Rangan; Maurizio Porfiri; Zhou Yu; John-Ross Rizzo
>
> **摘要:** This paper investigates the potential of vision-language models (VLMs) to assist people with blindness and low vision (pBLV) in navigation tasks. We evaluate state-of-the-art closed-source models, including GPT-4V, GPT-4o, Gemini-1.5-Pro, and Claude-3.5-Sonnet, alongside open-source models, such as Llava-v1.6-mistral and Llava-onevision-qwen, to analyze their capabilities in foundational visual skills: counting ambient obstacles, relative spatial reasoning, and common-sense wayfinding-pertinent scene understanding. We further assess their performance in navigation scenarios, using pBLV-specific prompts designed to simulate real-world assistance tasks. Our findings reveal notable performance disparities between these models: GPT-4o consistently outperforms others across all tasks, particularly in spatial reasoning and scene understanding. In contrast, open-source models struggle with nuanced reasoning and adaptability in complex environments. Common challenges include difficulties in accurately counting objects in cluttered settings, biases in spatial reasoning, and a tendency to prioritize object details over spatial feedback, limiting their usability for pBLV in navigation tasks. Despite these limitations, VLMs show promise for wayfinding assistance when better aligned with human feedback and equipped with improved spatial reasoning. This research provides actionable insights into the strengths and limitations of current VLMs, guiding developers on effectively integrating VLMs into assistive technologies while addressing key limitations for enhanced usability.
>
---
#### [new 066] Ground Reaction Inertial Poser: Physics-based Human Motion Capture from Sparse IMUs and Insole Pressure Sensors
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GRIP方法，用于从稀疏IMU和足底压力传感器数据中重建物理合理的运动，解决人体运动捕捉任务中的动态与地面交互问题。**

- **链接: [https://arxiv.org/pdf/2603.16233](https://arxiv.org/pdf/2603.16233)**

> **作者:** Ryosuke Hori; Jyun-Ting Song; Zhengyi Luo; Jinkun Cao; Soyong Shin; Hideo Saito; Kris Kitani
>
> **摘要:** We propose Ground Reaction Inertial Poser (GRIP), a method that reconstructs physically plausible human motion using four wearable devices. Unlike conventional IMU-only approaches, GRIP combines IMU signals with foot pressure data to capture both body dynamics and ground interactions. Furthermore, rather than relying solely on kinematic estimation, GRIP uses a digital twin of a person, in the form of a synthetic humanoid in a physics simulator, to reconstruct realistic and physically plausible motion. At its core, GRIP consists of two modules: KinematicsNet, which estimates body poses and velocities from sensor data, and DynamicsNet, which controls the humanoid in the simulator using the residual between the KinematicsNet prediction and the simulated humanoid state. To enable robust training and fair evaluation, we introduce a large-scale dataset, Pressure and Inertial Sensing for Human Motion and Interaction (PRISM), that captures diverse human motions with synchronized IMUs and insole pressure sensors. Experimental results show that GRIP outperforms existing IMU-only and IMU-pressure fusion methods across all evaluated datasets, achieving higher global pose accuracy and improved physical consistency.
>
---
#### [new 067] AsgardBench - Evaluating Visually Grounded Interactive Planning Under Minimal Feedback
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出AsgardBench，用于评估视觉引导的交互式规划能力。任务是解决计划在执行中根据视觉反馈调整的问题，通过控制环境变化测试模型的适应性。**

- **链接: [https://arxiv.org/pdf/2603.15888](https://arxiv.org/pdf/2603.15888)**

> **作者:** Andrea Tupini; Lars Liden; Reuben Tan; Yu Wang; Jianfeng Gao
>
> **备注:** 19 figures, 6 tables, including appendix
>
> **摘要:** With AsgardBench we aim to evaluate visually grounded, high-level action sequence generation and interactive planning, focusing specifically on plan adaptation during execution based on visual observations rather than navigation or low-level manipulation. In the landscape of embodied AI benchmarks, AsgardBench targets the capability category of interactive planning, which is more sophisticated than offline high-level planning as it requires agents to revise plans in response to environmental feedback, yet remains distinct from low-level execution. Unlike prior embodied AI benchmarks that conflate reasoning with navigation or provide rich corrective feedback that substitutes for perception, AsgardBench restricts agent input to images, action history, and lightweight success/failure signals, isolating interactive planning in a controlled simulator without low-level control noise. The benchmark contains 108 task instances spanning 12 task types, each systematically varied through object state, placement, and scene configuration. These controlled variations create conditional branches in which a single instruction can require different action sequences depending on what the agent observes, emphasizing conditional branching and plan repair during execution. Our evaluations of leading vision language models show that performance drops sharply without visual input, revealing weaknesses in visual grounding and state tracking that ultimately undermine interactive planning. Our benchmark zeroes in on a narrower question: can a model actually use what it sees to adapt a plan when things do not go as expected?
>
---
#### [new 068] S2Act: Simple Spiking Actor
- **分类: cs.MA; cs.ET; cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决SNN在复杂环境中的性能与部署问题。提出S2Act框架，通过简化设计提升SNN的效率与稳定性，适用于多机器人系统。**

- **链接: [https://arxiv.org/pdf/2603.15725](https://arxiv.org/pdf/2603.15725)**

> **作者:** Ugur Akcal; Seung Hyun Kim; Mikihisa Yuasa; Hamid Osooli; Jiarui Sun; Ribhav Sahu; Mattia Gazzola; Huy T. Tran; Girish Chowdhary
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Spiking neural networks (SNNs) and biologically-inspired learning mechanisms are attractive in mobile robotics, where the size and performance of onboard neural network policies are constrained by power and computational budgets. Existing SNN approaches, such as population coding, reward modulation, and hybrid artificial neural network (ANN)-SNN architectures, have shown promising results; however, they face challenges in complex, highly stochastic environments due to SNN sensitivity to hyperparameters and inconsistent gradient signals. To address these challenges, we propose simple spiking actor (S2Act), a computationally lightweight framework that deploys an RL policy using an SNN in three steps: (1) architect an actor-critic model based on an approximated network of rate-based spiking neurons, (2) train the network with gradients using compatible activation functions, and (3) transfer the trained weights into physical parameters of rate-based leaky integrate-and-fire (LIF) neurons for inference and deployment. By globally shaping LIF neuron parameters such that their rate-based responses approximate ReLU activations, S2Act effectively mitigates the vanishing gradient problem, while pre-constraining LIF response curves reduces reliance on complex SNN-specific hyperparameter tuning. We demonstrate our method in two multi-agent stochastic environments (capture-the-flag and parking) that capture the complexity of multi-robot interactions, and deploy our trained policies on physical TurtleBot platforms using Intel's Loihi neuromorphic hardware. Our experimental results show that S2Act outperforms relevant baselines in task performance and real-time inference in nearly all considered scenarios, highlighting its potential for rapid prototyping and efficient real-world deployment of SNN-based RL policies.
>
---
#### [new 069] Thermopneumatic Pixels for Fast, Localized, Low-Voltage Touch Feedback
- **分类: cs.HC; cs.ET; cs.RO**

- **简介: 该论文属于触觉反馈技术领域，旨在解决低电压、快速触觉反馈的实现问题。研究提出热气动像素（TPPs），通过低电压驱动产生显著触觉效果，适用于可穿戴和交互设备。**

- **链接: [https://arxiv.org/pdf/2603.16750](https://arxiv.org/pdf/2603.16750)**

> **作者:** Max Linnander; Yon Visell
>
> **摘要:** We present thermopneumatic pixels (TPPs), which are tactile actuators designed for rapid fabrication and straightforward integration into compact wearable and surface-based haptic systems. Each TPP converts low-voltage ($\sim$10 V) electrical pulses into transient pressure increases within a sealed cavity, producing out-of-plane forces and displacements suitable for tactile stimulation. The architecture enables scalable fabrication and spatially distributed actuation while maintaining simple electrical interfacing. The TPPs are constructed from inexpensive, readily available materials using straightforward layer-based assembly, facilitating rapid prototyping and integration into interactive devices. Mechanical characterization demonstrates peak forces exceeding 1 N and millimeter displacements. We further present driving electronics for operating multiple TPP modules concurrently and report perceptual study results demonstrating the effectiveness of the resulting tactile feedback. Together, these results establish low-voltage thermopneumatic actuation as an accessible and high-performance approach for embedding tactile feedback into experimental and consumer-facing interfaces.
>
---
## 更新

#### [replaced 001] $χ_{0}$: Resource-Aware Robust Manipulation via Taming Distributional Inconsistencies
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人长期操作任务，解决分布不一致导致的可靠性问题。提出$\chi_{0}$框架，通过模型算术、阶段优势和训练部署对齐提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.09021](https://arxiv.org/pdf/2602.09021)**

> **作者:** Checheng Yu; Chonghao Sima; Gangcheng Jiang; Hai Zhang; Haoguang Mai; Hongyang Li; Huijie Wang; Jin Chen; Kaiyang Wu; Li Chen; Lirui Zhao; Modi Shi; Ping Luo; Qingwen Bu; Shijia Peng; Tianyu Li; Yibo Yuan
>
> **摘要:** High-reliability long-horizon robotic manipulation has traditionally relied on large-scale data and compute to understand complex real-world dynamics. However, we identify that the primary bottleneck to real-world robustness is not resource scale alone, but the distributional shift among the human demonstration distribution, the inductive bias learned by the policy, and the test-time execution distribution -- a systematic inconsistency that causes compounding errors in multi-stage tasks. To mitigate these inconsistencies, we propose $\chi_{0}$, a resource-efficient framework with effective modules designated to achieve production-level robustness in robotic manipulation. Our approach builds off three technical pillars: (i) Model Arithmetic, a weight-space merging strategy that efficiently soaks up diverse distributions of different demonstrations, varying from object appearance to state variations; (ii) Stage Advantage, a stage-aware advantage estimator that provides stable, dense progress signals, overcoming the numerical instability of prior non-stage approaches; and (iii) Train-Deploy Alignment, which bridges the distribution gap via spatio-temporal augmentation, heuristic DAgger corrections, and temporal chunk-wise smoothing. $\chi_{0}$ enables two sets of dual-arm robots to collaboratively orchestrate long-horizon garment manipulation, spanning tasks from flattening, folding, to hanging different clothes. Our method exhibits high-reliability autonomy; we are able to run the system from arbitrary initial state for consecutive 24 hours non-stop. Experiments validate that $\chi_{0}$ surpasses the state-of-the-art $\pi_{0.5}$ in success rate by nearly 250%, with only 20-hour data and 8 A100 GPUs. Code, data and models will be released to facilitate the community.
>
---
#### [replaced 002] CLAIM: Camera-LiDAR Alignment with Intensity and Monodepth
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于相机与LiDAR标定任务，解决二者数据对齐问题。提出CLAIM方法，通过结构和纹理损失优化变换矩阵，实现高效精准对齐。**

- **链接: [https://arxiv.org/pdf/2512.14001](https://arxiv.org/pdf/2512.14001)**

> **作者:** Zhuo Zhang; Yonghui Liu; Meijie Zhang; Feiyang Tan; Yikang Ding
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** In this paper, we unleash the potential of the powerful monodepth model in camera-LiDAR calibration and propose CLAIM, a novel method of aligning data from the camera and LiDAR. Given the initial guess and pairs of images and LiDAR point clouds, CLAIM utilizes a coarse-to-fine searching method to find the optimal transformation minimizing a patched Pearson correlation-based structure loss and a mutual information-based texture loss. These two losses serve as good metrics for camera-LiDAR alignment results and require no complicated steps of data processing, feature extraction, or feature matching like most methods, rendering our method simple and adaptive to most scenes. We validate CLAIM on public KITTI, Waymo, and MIAS-LCEC datasets, and the experimental results demonstrate its superior performance compared with the state-of-the-art methods. The code is available at this https URL.
>
---
#### [replaced 003] Real-Time Quasi-Static Modeling of UAV Tether Aerodynamics
- **分类: cs.RO**

- **简介: 论文研究无人机系绳动力学建模问题，提出两种实时准静态方法解决系绳受力分析，提升长时续航与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.22588](https://arxiv.org/pdf/2512.22588)**

> **作者:** Max Beffert; Andreas Zell
>
> **摘要:** One of the main limitations of multirotor UAVs is their short flight time due to battery constraints. A practical solution for continuous operation is to power the drone from the ground via a tether. While this approach has been demonstrated for stationary systems, scenarios with a fast-moving base vehicle or strong wind conditions require modeling the tether forces, including aerodynamic effects. In this work, we propose two complementary approaches for real-time quasi-static tether modeling with aerodynamics. The first is an analytical method based on catenary theory with a uniform drag assumption, achieving very fast solve times below 1~ms. The second is a numerical method that discretizes the tether into segments and lumped masses, solving the equilibrium equations using CasADi and IPOPT. By leveraging initialization strategies, such as warm starting and analytical initialization, real-time performance was achieved with a solve time of 5~ms, while allowing for flexible force formulations. Both approaches were validated in real-world tests using a load cell to measure the tether force. The results show that the analytical method provides sufficient accuracy for most tethered UAV applications with minimal computational cost, while the numerical method offers higher flexibility and physical accuracy when required. These approaches form a lightweight and extensible framework for real-time tether simulation, applicable to both offline optimization and online tasks such as simulation, control, and trajectory planning.
>
---
#### [replaced 004] DySL-VLA: Efficient Vision-Language-Action Model Inference via Dynamic-Static Layer-Skipping for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文提出DySL-VLA，用于机器人操作的视觉-语言-动作模型推理。解决高计算成本问题，通过动态跳过非关键层提升效率，同时保持精度。**

- **链接: [https://arxiv.org/pdf/2602.22896](https://arxiv.org/pdf/2602.22896)**

> **作者:** Zebin Yang; Yijiahao Qi; Tong Xie; Bo Yu; Shaoshan Liu; Meng Li
>
> **备注:** DAC 2026
>
> **摘要:** Vision-Language-Action (VLA) models have shown remarkable success in robotic tasks like manipulation by fusing a language model's reasoning with a vision model's 3D understanding. However, their high computational cost remains a major obstacle for real-world applications that require real-time performance. We observe that the actions within a task have varying levels of importance: critical steps demand high precision, while less important ones can tolerate more variance. Leveraging this insight, we propose DySL-VLA, a novel framework that addresses computational cost by dynamically skipping VLA layers based on each action's importance. DySL-VLA categorizes its layers into two types: informative layers, which are consistently executed, and incremental layers, which can be selectively skipped. To intelligently skip layers without sacrificing accuracy, we invent a prior-post skipping guidance mechanism to determine when to initiate layer-skipping. We also propose a skip-aware two-stage knowledge distillation algorithm to efficiently train a standard VLA into a DySL-VLA. Our experiments indicate that DySL-VLA achieves 2.1% improvement in success length over Deer-VLA on the Calvin dataset, while simultaneously reducing trainable parameters by a factor of 85.7 and providing a 3.75x speedup relative to the RoboFlamingo baseline at iso-accuracy. Our code is available on this https URL.
>
---
#### [replaced 005] Real-World Deployment of Cloud-based Autonomous Mobility Systems for Outdoor and Indoor Environments
- **分类: cs.RO; cs.NI**

- **简介: 该论文属于自主移动系统任务，解决复杂环境下的感知与协调问题。通过云架构整合多传感器数据，提升系统感知鲁棒性与安全性。**

- **链接: [https://arxiv.org/pdf/2505.21676](https://arxiv.org/pdf/2505.21676)**

> **作者:** Yufeng Yang; Minghao Ning; Keqi Shu; Aladdin Saleh; Ehsan Hashemi; Amir Khajepour
>
> **备注:** This paper has been submitted to IEEE Robotics and Automation Magazine
>
> **摘要:** Autonomous mobility systems increasingly operate in dense and dynamic environments where perception occlusions, limited sensing coverage, and multi-agent interactions pose major challenges. While onboard sensors provide essential local perception, they often struggle to maintain reliable situational awareness in crowded urban or indoor settings. This article presents the Cloud-based Autonomous Mobility (CAM) framework, a generalized architecture that integrates infrastructure-based intelligent sensing with cloud-level coordination to enhance autonomous operations. The system deploys distributed Intelligent Sensor Nodes (ISNs) equipped with cameras, LiDAR, and edge computing to perform multi-modal perception and transmit structured information to a cloud platform via high-speed wireless communication. The cloud aggregates observations from multiple nodes to generate a global scene representation for other autonomous modules, such as decision making, motion planning, etc. Real-world deployments in an urban roundabout and a hospital-like indoor environment demonstrate improved perception robustness, safety, and coordination for future intelligent mobility systems.
>
---
#### [replaced 006] GeoFIK: A Fast and Reliable Geometric Solver for the IK of the Franka Arm based on Screw Theory Enabling Multiple Redundancy Parameters
- **分类: cs.RO**

- **简介: 该论文属于机器人逆运动学（IK）任务，旨在解决Franka机械臂的快速、可靠求解问题。提出GeoFIK方法，利用螺旋理论处理冗余和奇异情况，提升求解性能。**

- **链接: [https://arxiv.org/pdf/2503.03992](https://arxiv.org/pdf/2503.03992)**

> **作者:** Pablo C. Lopez-Custodio; Yuhe Gong; Luis F.C. Figueredo
>
> **摘要:** Modern robotics applications require an inverse kinematics (IK) solver that is fast, robust and consistent, and that provides all possible solutions. Currently, the Franka robot arm is the most widely used manipulator in robotics research. With 7 DOFs, the IK of this robot is not only complex due to its 1-DOF redundancy, but also due to the link offsets at the wrist and elbow. Due to this complexity, none of the Franka IK solvers available in the literature provide satisfactory results when used in real-world applications. Therefore, in this paper we introduce GeoFIK (Geometric Franka IK), an analytical IK solver that allows the use of different joint variables to resolve the redundancy. The approach uses screw theory to describe the entire geometry of the robot, allowing the computation of the Jacobian matrix prior to computation of joint angles. All singularities are identified and handled. As an example of how the geometric elements obtained by the IK can be exploited, a solver with the swivel angle as the free variable is provided. Several experiments are carried out to validate the speed, robustness and reliability of the GeoFIK against two state-of-the-art solvers.
>
---
#### [replaced 007] WorldVLM: Combining World Model Forecasting and Vision-Language Reasoning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶领域，旨在解决环境理解与动态预测问题。提出WorldVLM融合视觉语言模型与世界模型，提升决策与预测能力。**

- **链接: [https://arxiv.org/pdf/2603.14497](https://arxiv.org/pdf/2603.14497)**

> **作者:** Stefan Englmeier; Katharina Winter; Fabian B. Flohr
>
> **备注:** 8 pages, 6 figures, 5 tables; submitted to IEEE
>
> **摘要:** Autonomous driving systems depend on on models that can reason about high-level scene contexts and accurately predict the dynamics of their surrounding environment. Vision- Language Models (VLMs) have recently emerged as promising tools for decision-making and scene understanding, offering strong capabilities in contextual reasoning. However, their limited spatial comprehension constrains their effectiveness as end-to-end driving models. World Models (WM) internalize environmental dynamics to predict future scene evolution. Recently explored as ego-motion predictors and foundation models for autonomous driving, they represent a promising direction for addressing key challenges in the field, particularly enhancing generalization while maintaining dynamic prediction. To leverage the complementary strengths of context-based decision making and prediction, we propose WorldVLM: A hybrid architecture that unifies VLMs and WMs. In our design, the high-level VLM generates behavior commands to guide the driving WM, enabling interpretable and context-aware actions. We evaluate conditioning strategies and provide insights into the hybrid design challenges.
>
---
#### [replaced 008] MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人导航任务，解决零样本导航中的视觉信息丢失与词汇受限问题。提出M3DSG和MSGNav系统，提升导航性能与开放词汇支持。**

- **链接: [https://arxiv.org/pdf/2511.10376](https://arxiv.org/pdf/2511.10376)**

> **作者:** Xun Huang; Shijia Zhao; Yunxiang Wang; Xin Lu; Wanfa Zhang; Rongsheng Qu; Weixin Li; Yunhong Wang; Chenglu Wen
>
> **备注:** 18 pages, Accepted by CVPR 2026
>
> **摘要:** Embodied navigation is a fundamental capability for robotic agents operating. Real-world deployment requires open vocabulary generalization and low training overhead, motivating zero-shot methods rather than task-specific RL training. However, existing zero-shot methods that build explicit 3D scene graphs often compress rich visual observations into text-only relations, leading to high construction cost, irreversible loss of visual evidence, and constrained vocabularies. To address these limitations, we introduce the Multi-modal 3D Scene Graph (M3DSG), which preserves visual cues by replacing textual relational edges with dynamically assigned images. Built on M3DSG, we propose MSGNav, a zero-shot navigation system that includes a Key Subgraph Selection module for efficient reasoning, an Adaptive Vocabulary Update module for open vocabulary support, and a Closed-Loop Reasoning module for accurate exploration reasoning. Additionally, we further identify the last mile problem in zero-shot navigation determining the feasible target location with a suitable final viewpoint, and propose a Visibility-based Viewpoint Decision module to explicitly resolve it. Comprehensive experimental results demonstrate that MSGNav achieves state-of-the-art performance on the challenging GOAT-Bench and HM3D-ObjNav benchmark. The code will be publicly available at this https URL.
>
---
#### [replaced 009] Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于立体匹配任务，旨在解决实时应用中零样本泛化与计算效率的矛盾。提出Fast-FoundationStereo，通过知识蒸馏、架构搜索和结构化剪枝实现高速高精度匹配。**

- **链接: [https://arxiv.org/pdf/2512.11130](https://arxiv.org/pdf/2512.11130)**

> **作者:** Bowen Wen; Shaurya Dewan; Stan Birchfield
>
> **摘要:** Stereo foundation models achieve strong zero-shot generalization but remain computationally prohibitive for real-time applications. Efficient stereo architectures, on the other hand, sacrifice robustness for speed and require costly per-domain fine-tuning. To bridge this gap, we present Fast-FoundationStereo, a family of architectures that achieve, for the first time, strong zero-shot generalization at real-time frame rate. We employ a divide-and-conquer acceleration strategy with three components: (1) knowledge distillation to compress the hybrid backbone into a single efficient student; (2) blockwise neural architecture search for automatically discovering optimal cost filtering designs under latency budgets, reducing search complexity exponentially; and (3) structured pruning for eliminating redundancy in the iterative refinement module. Furthermore, we introduce an automatic pseudo-labeling pipeline used to curate 1.4M in-the-wild stereo pairs to supplement synthetic training data and facilitate knowledge distillation. The resulting model can run over 10x faster than FoundationStereo while closely matching its zero-shot accuracy, thus establishing a new state-of-the-art among real-time methods. Project page: this https URL
>
---
#### [replaced 010] UGotMe: An Embodied System for Affective Human-Robot Interaction
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于情感人机交互任务，旨在解决多人群聊中视觉噪声和实时响应问题。提出UGotMe系统，采用去噪策略提升情感识别效果，并优化数据传输以实现实时交互。**

- **链接: [https://arxiv.org/pdf/2410.18373](https://arxiv.org/pdf/2410.18373)**

> **作者:** Peizhen Li; Longbing Cao; Xiao-Ming Wu; Xiaohan Yu; Runze Yang
>
> **备注:** Accepted to the 2025 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Equipping humanoid robots with the capability to understand emotional states of human interactants and express emotions appropriately according to situations is essential for affective human-robot interaction. However, enabling current vision-aware multimodal emotion recognition models for affective human-robot interaction in the real-world raises embodiment challenges: addressing the environmental noise issue and meeting real-time requirements. First, in multiparty conversation scenarios, the noises inherited in the visual observation of the robot, which may come from either 1) distracting objects in the scene or 2) inactive speakers appearing in the field of view of the robot, hinder the models from extracting emotional cues from vision inputs. Secondly, realtime response, a desired feature for an interactive system, is also challenging to achieve. To tackle both challenges, we introduce an affective human-robot interaction system called UGotMe designed specifically for multiparty conversations. Two denoising strategies are proposed and incorporated into the system to solve the first issue. Specifically, to filter out distracting objects in the scene, we propose extracting face images of the speakers from the raw images and introduce a customized active face extraction strategy to rule out inactive speakers. As for the second issue, we employ efficient data transmission from the robot to the local server to improve realtime response capability. We deploy UGotMe on a human robot named Ameca to validate its real-time inference capabilities in practical scenarios. Videos demonstrating real-world deployment are available at this https URL.
>
---
#### [replaced 011] No More Blind Spots: Learning Vision-Based Omnidirectional Bipedal Locomotion for Challenging Terrain
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人运动控制任务，解决复杂地形下双足机器人全方位运动问题。通过视觉感知和强化学习，实现高效、自适应的全方位行走。**

- **链接: [https://arxiv.org/pdf/2508.11929](https://arxiv.org/pdf/2508.11929)**

> **作者:** Mohitvishnu S. Gadde; Pranay Dugar; Ashish Malik; Alan Fern
>
> **摘要:** Effective bipedal locomotion in dynamic environments, such as cluttered indoor spaces or uneven terrain, requires agile and adaptive movement in all directions. This necessitates omnidirectional terrain sensing and a controller capable of processing such input. We present a learning framework for vision-based omnidirectional bipedal locomotion, enabling seamless movement using depth images. A key challenge is the high computational cost of rendering omnidirectional depth images in simulation, making traditional sim-to-real reinforcement learning (RL) impractical. Our method combines a robust blind controller with a teacher policy that supervises a vision-based student policy, trained on noise-augmented terrain data to avoid rendering costs during RL and ensure robustness. We also introduce a data augmentation technique for supervised student training, accelerating training by up to 10 times compared to conventional methods. Our framework is validated through simulation and real-world tests, demonstrating effective omnidirectional locomotion with minimal reliance on expensive rendering. This is, to the best of our knowledge, the first demonstration of vision-based omnidirectional bipedal locomotion, showcasing its adaptability to diverse terrains.
>
---
#### [replaced 012] System Design of the Ultra Mobility Vehicle: A Driving, Balancing, and Jumping Bicycle Robot
- **分类: cs.RO**

- **简介: 该论文属于机器人设计任务，旨在开发一种具备动态移动能力的自行车机器人UMV，解决单轮平衡与跳跃问题，通过仿真优化和强化学习实现多种运动行为。**

- **链接: [https://arxiv.org/pdf/2602.22118](https://arxiv.org/pdf/2602.22118)**

> **作者:** Benjamin Bokser; Daniel Gonzalez; Aaron Preston; Alex Bahner; Annika Wollschläger; Arianna Ilvonen; Asa Eckert-Erdheim; Ashwin Khadke; Bilal Hammoud; Dean Molinaro; Fabian Jenelten; Henry Mayne; Howie Choset; Igor Bogoslavskyi; Itic Tinman; James Tigue; Jan Preisig; Kaiyu Zheng; Kenny Sharma; Kim Ang; Laura Lee; Liana Margolese; Nicole Lin; Oscar Frias; Paul Drews; Ravi Boggavarapu; Rick Burnham; Samuel Zapolsky; Sangbae Kim; Scott Biddlestone; Sean Mayorga; Shamel Fahmi; Surya Singh; Tyler McCollum; Velin Dimitrov; William Moyne; Yu-Ming Chen; Farbod Farshidian; Marco Hutter; David Perry; Al Rizzi; Gabe Nelson
>
> **备注:** 17 Pages, 11 figures, 3 movies, 2 tables
>
> **摘要:** Trials cyclists and mountain bike riders can hop, jump, balance, and drive on one or both wheels. This versatility allows them to achieve speed and energy-efficiency on smooth terrain and agility over rough terrain. Inspired by these athletes, we present the design and control of a robotic platform, Ultra Mobility Vehicle (UMV), which combines a bicycle and a reaction mass to move dynamically with minimal actuated degrees of freedom. We employ a simulation-driven design optimization process to synthesize a spatial linkage topology with a focus on vertical jump height and momentum-based balancing on a single wheel contact. Using a constrained Reinforcement Learning (RL) framework, we demonstrate zero-shot transfer of diverse athletic behaviors, including track-stands, jumps, wheelies, rear wheel hopping, and front flips. This 23.5 kg robot is capable of high speeds (8 m/s) and jumping on and over large obstacles (1 m tall, or 130% of the robot's nominal height).
>
---
#### [replaced 013] Trust in Autonomous Human--Robot Collaboration: Effects of Responsive Interaction Policies
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机协作研究，探讨自主交互中信任的形成机制。通过对比不同交互策略，分析其对信任的影响，旨在提升自主机器人系统的交互设计。**

- **链接: [https://arxiv.org/pdf/2603.00154](https://arxiv.org/pdf/2603.00154)**

> **作者:** Shauna Heron; Meng Cheng Lau
>
> **摘要:** Trust plays a central role in human--robot collaboration, yet its formation is rarely examined under the constraints of fully autonomous interaction. This pilot study investigated how interaction policy influences trust during in-person collaboration with a social robot operating without Wizard-of-Oz control or scripted repair. Participants completed a multi-stage collaborative task with a mobile robot that autonomously managed spoken-language dialogue, affect inference, and task progression. Two interaction policies were compared: a responsive policy, in which the robot proactively adapted its dialogue and assistance based on inferred interaction state, and a neutral, reactive policy, in which the robot provided only direct, task-relevant responses when prompted. Responsive interaction was associated with significantly higher post-interaction trust under viable communication conditions, despite no reliable differences in overall task accuracy. Sensitivity analyses indicated that affective and experiential components of trust were more sensitive to communication breakdown than evaluative judgments of reliability, and that as language-mediated interaction degraded, the trust advantage associated with responsiveness attenuated and ratings became less clearly interpretable as calibrated evaluations of collaborative competence. These findings suggest that trust in autonomous human--robot interaction emerges from process-level interaction dynamics and operates within constraints imposed by communication viability, highlighting the importance of evaluating trust under real autonomy conditions when designing interactive robotic systems.
>
---
#### [replaced 014] Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry
- **分类: cs.RO**

- **简介: 该论文属于视觉-惯性里程计任务，解决传统方法在精度与效率间的权衡问题。通过双智能体强化学习，优化视觉前端触发与融合策略，提升系统效率与内存使用。**

- **链接: [https://arxiv.org/pdf/2511.21083](https://arxiv.org/pdf/2511.21083)**

> **作者:** Feiyang Pan; Shenghe Zheng; Chunyan Yin; Guangbin Dou
>
> **备注:** Accepted to the CVPR 2026 Main Track
>
> **摘要:** Visual-Inertial Odometry (VIO) is a critical component for robust ego-motion estimation, enabling foundational capabilities such as autonomous navigation in robotics and real-time 6-DoF tracking for augmented reality. Existing methods face a well-known trade-off: filter-based approaches are efficient but prone to drift, while optimization-based methods, though accurate, rely on computationally prohibitive Visual-Inertial Bundle Adjustment (VIBA) that is difficult to run on resource-constrained platforms. Rather than removing VIBA altogether, we aim to reduce how often and how heavily it must be invoked. To this end, we cast two key design choices in modern VIO, when to run the visual frontend and how strongly to trust its output, as sequential decision problems, and solve them with lightweight reinforcement learning (RL) agents. Our framework introduces a lightweight, dual-pronged RL policy that serves as our core contribution: (1) a Select Agent intelligently gates the entire VO pipeline based only on high-frequency IMU data; and (2) a composite Fusion Agent that first estimates a robust velocity state via a supervised network, before an RL policy adaptively fuses the full (p, v, q) state. Experiments on the EuRoC MAV and TUM-VI datasets show that, in our unified evaluation, the proposed method achieves a more favorable accuracy-efficiency-memory trade-off than prior GPU-based VO/VIO systems: it attains the best average ATE while running up to 1.77 times faster and using less GPU memory. Compared to classical optimization-based VIO systems, our approach maintains competitive trajectory accuracy while substantially reducing computational load.
>
---
#### [replaced 015] Stein Variational Ergodic Surface Coverage with SE(3) Constraints
- **分类: cs.RO**

- **简介: 该论文属于机器人表面操作任务，解决3D点云覆盖问题。提出基于SE(3)约束的Stein变分梯度下降方法，提升轨迹优化效率与精度。**

- **链接: [https://arxiv.org/pdf/2603.09458](https://arxiv.org/pdf/2603.09458)**

> **作者:** Jiayun Li; Yufeng Jin; Sangli Teng; Dejian Gong; Georgia Chalvatzaki
>
> **摘要:** Surface manipulation tasks require robots to generate trajectories that comprehensively cover complex 3D surfaces while maintaining precise end-effector poses. Existing ergodic trajectory optimization (TO) methods demonstrate success in coverage tasks, while struggling with point-cloud targets due to the nonconvex optimization landscapes and the inadequate handling of SE(3) constraints in sampling-as-optimization (SAO) techniques. In this work, we introduce a preconditioned SE(3) Stein Variational Gradient Descent (SVGD) approach for SAO ergodic trajectory generation. Our proposed approach comprises multiple innovations. First, we reformulate point-cloud ergodic coverage as a manifold-aware sampling problem. Second, we derive SE(3)-specific SVGD particle updates, and, third, we develop a preconditioner to accelerate TO convergence. Our sampling-based framework consistently identifies superior local optima compared to strong optimization-based and SAO baselines while preserving the SE(3) geometric structure. Experiments on a 3D point-cloud surface coverage benchmark and robotic surface drawing tasks demonstrate that our method achieves superior coverage quality with tractable computation in our setting relative to existing TO and SAO approaches, and is validated in real-world robot experiments.
>
---
#### [replaced 016] When a Robot is More Capable than a Human: Learning from Constrained Demonstrators
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人学习任务，解决受限专家示范导致策略次优的问题。通过推断状态奖励并探索更优轨迹，提升政策性能。**

- **链接: [https://arxiv.org/pdf/2510.09096](https://arxiv.org/pdf/2510.09096)**

> **作者:** Xinhu Li; Ayush Jain; Zhaojing Yang; Yigit Korkmaz; Erdem Bıyık
>
> **摘要:** Learning from demonstrations enables experts to teach robots complex tasks using interfaces such as kinesthetic teaching, joystick control, and sim-to-real transfer. However, these interfaces often constrain the expert's ability to demonstrate optimal behavior due to indirect control, setup restrictions, and hardware safety. For example, a joystick can move a robotic arm only in a 2D plane, even though the robot operates in a higher-dimensional space. As a result, the demonstrations collected by constrained experts lead to suboptimal performance of the learned policies. This raises a key question: Can a robot learn a better policy than the one demonstrated by a constrained expert? We address this by allowing the agent to go beyond direct imitation of expert actions and explore shorter and more efficient trajectories. We use the demonstrations to infer a state-only reward signal that measures task progress, and self-label reward for unknown states using temporal interpolation. Our approach outperforms common imitation learning in both sample efficiency and task completion time. On a real WidowX robotic arm, it completes the task in 12 seconds, 10x faster than behavioral cloning, as shown in real-robot videos on this https URL .
>
---
#### [replaced 017] Optimal Solutions for the Moving Target Vehicle Routing Problem via Branch-and-Price with Relaxed Continuity
- **分类: cs.RO**

- **简介: 该论文研究MT-VRP任务，解决多代理拦截移动目标的路径规划问题。提出BPRC算法，通过改进的标签算法高效求解，提升求解速度。**

- **链接: [https://arxiv.org/pdf/2603.00663](https://arxiv.org/pdf/2603.00663)**

> **作者:** Anoop Bhat; Geordan Gutow; Zhongqiang Ren; Sivakumar Rathinam; Howie Choset
>
> **备注:** Accepted to ICAPS 2026
>
> **摘要:** The Moving Target Vehicle Routing Problem (MT-VRP) seeks trajectories for several agents that intercept a set of moving targets, subject to speed, time window, and capacity constraints. We introduce an exact algorithm, Branch-and-Price with Relaxed Continuity (BPRC), for the MT-VRP. The main challenge in a branch-and-price approach for the MT-VRP is the pricing subproblem, which is complicated by moving targets and time-dependent travel costs between targets. Our key contribution is a new labeling algorithm that solves this subproblem by means of a novel dominance criterion tailored for problems with moving targets. Numerical results on instances with up to 25 targets show that our algorithm finds optimal solutions more than an order of magnitude faster than a baseline based on previous work, showing particular strength in scenarios with limited agent capacities.
>
---
#### [replaced 018] Minimal Intervention Shared Control with Guaranteed Safety under Non-Convex Constraints
- **分类: cs.RO; cs.HC; eess.SY**

- **简介: 该论文属于共享控制任务，旨在解决非凸约束下的安全性和最小干预问题。提出一种约束感知控制器，确保可行性与约束满足，验证结果表明有效提升任务表现与用户信任。**

- **链接: [https://arxiv.org/pdf/2507.02438](https://arxiv.org/pdf/2507.02438)**

> **作者:** Shivam Chaubey; Francesco Verdoja; Shankar Deka; Ville Kyrki
>
> **备注:** Accepted for publication at the 2026 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Shared control combines human intention with autonomous decision-making. At the low level, the primary goal is to maintain safety regardless of the user's input to the system. However, existing shared control methods-based on, e.g., Model Predictive Control, Control Barrier Functions, or learning-based control-often face challenges with feasibility, scalability, and mixed constraints. To address these challenges, we propose a Constraint-Aware Assistive Controller that computes control actions online while ensuring recursive feasibility, strict constraint satisfaction, and minimal deviation from the user's intent. It also accommodates a structured class of non-convex constraints common in real-world settings. We leverage Robust Controlled Invariant Sets for recursive feasibility and a Mixed-Integer Quadratic Programming formulation to handle non-convex constraints. We validate the approach through a large-scale user study with 66 participants-one of the most extensive in shared control research-using a simulated environment to assess task load, trust, and perceived control, in addition to performance. The results show consistent improvements across all these aspects without compromising safety and user intent. Additionally, a real-world experiment on a robotic manipulator demonstrates the framework's applicability under bounded disturbances, ensuring safety and collision-free operation.
>
---
#### [replaced 019] CloSE: A Geometric Shape-Agnostic Cloth State Representation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决衣物变形状态表示问题。提出CloSE表示方法，通过几何特征抽象实现对不同形状衣物的通用描述，用于预测折痕和辅助规划。**

- **链接: [https://arxiv.org/pdf/2504.05033](https://arxiv.org/pdf/2504.05033)**

> **作者:** Jay Kamat; Júlia Borràs; Carme Torras
>
> **备注:** Accepted at ICRA 2026 (8 pages, 11 figures, 1 table). Project page: this https URL
>
> **摘要:** Cloth manipulation is a difficult problem mainly because of the non-rigid nature of cloth, which makes a good representation of deformation essential. We present a new representation for the deformation-state of clothes. First, we propose the dGLI disk representation based on topological indices computed for edge segments of the cloth border that are arranged on a circular grid. The heat-map of the dGLI disk uncovers patterns that correspond to features of the cloth state that are consistent for different shapes, sizes or orientation of the cloth. We then abstract these important features from the dGLI disk into a circle, calling it the Cloth StatE representation (CloSE). This representation is compact, continuous, and general for different shapes. We show that this representation is able to accurately predict the fold locations for several simulation clothing datasets. Finally, we also show the strengths of this representation in two relevant applications: semantic labeling and high- and low-level planning. The code and the dataset can be accessed from: this https URL
>
---
#### [replaced 020] An Intention-driven Lane Change Framework Considering Heterogeneous Dynamic Cooperation in Mixed-traffic Environment
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，解决混合交通中自主车辆的变道问题。通过融合驾驶风格识别与合作决策，提升变道安全与效率。**

- **链接: [https://arxiv.org/pdf/2509.22550](https://arxiv.org/pdf/2509.22550)**

> **作者:** Xiaoyun Qiu; Haichao Liu; Yue Pan; Jun Ma; Xinhu Zheng
>
> **摘要:** In mixed-traffic environments, autonomous vehicles (AVs) must interact with heterogeneous human-driven vehicles (HVs) whose intentions and driving styles vary across individuals and scenarios. Such variability introduces uncertainty into lane change interactions, where safety and efficiency critically depend on accurately anticipating surrounding drivers' cooperative responses. Existing methods often oversimplify these interactions by assuming uniform or fixed behavioral patterns. To address this limitation, we propose an intention-driven lane change framework that integrates driving-style recognition with cooperation-aware decision-making and motion-planning. A deep learning-based classifier identifies distinct human driving styles in real time. We then introduce a dual-perspective cooperation score composed of intrinsic style-dependent tendencies and interactive dynamic components, enabling interpretable and adaptive intention prediction and quantitative inference. A decision-making module combines behavior cloning (BC) and inverse reinforcement learning (IRL) to determine lane change feasibility. Later, a coordinated motion-planning architecture integrating IRL-based intention inference with model predictive control (MPC) is established to generate collision-free and socially compliant trajectories. Experiments on the NGSIM dataset show that the proposed decision-making model outperforms representative rule-based and learning-based baselines, achieving 96.98% accuracy in lane change classification. Motion-planning evaluations further demonstrate improved maneuver success and execution stability in mixed-traffic environments. These results validate the effectiveness of structured cooperation modeling for intention-driven autonomous lane changes.
>
---
#### [replaced 021] Metamorphic Testing of Vision-Language Action-Enabled Robots
- **分类: cs.RO; cs.SE**

- **简介: 该论文研究VLA模型的测试问题，针对测试用例难以定义的难题，提出基于元测试的方法，通过设计元关系模式自动检测故障，提升测试效率与通用性。**

- **链接: [https://arxiv.org/pdf/2602.22579](https://arxiv.org/pdf/2602.22579)**

> **作者:** Pablo Valle; Sergio Segura; Shaukat Ali; Aitor Arrieta
>
> **摘要:** Vision-Language-Action (VLA) models are multimodal robotic task controllers that, given an instruction and visual inputs, produce a sequence of low-level control actions (or motor commands) enabling a robot to execute the requested task in the physical environment. These systems face the test oracle problem from multiple perspectives. On the one hand, a test oracle must be defined for each instruction prompt, which is a complex and non-generalizable approach. On the other hand, current state-of-the-art oracles typically capture symbolic representations of the world (e.g., robot and object states), enabling the correctness evaluation of a task, but fail to assess other critical aspects, such as the quality with which VLA-enabled robots perform a task. In this paper, we explore whether Metamorphic Testing (MT) can alleviate the test oracle problem in this context. To do so, we propose two metamorphic relation patterns and five metamorphic relations to assess whether changes to the test inputs impact the original trajectory of the VLA-enabled robots. An empirical study involving five VLA models, two simulated robots, and four robotic tasks shows that MT can effectively alleviate the test oracle problem by automatically detecting diverse types of failures, including, but not limited to, uncompleted tasks. More importantly, the proposed MRs are generalizable, making the proposed approach applicable across different VLA models, robots, and tasks, even in the absence of test oracles.
>
---
#### [replaced 022] Traj2Action: A Co-Denoising Framework for Trajectory-Guided Human-to-Robot Skill Transfer
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Traj2Action框架，解决人类到机器人技能迁移问题。通过3D轨迹作为中间表示，提升机器人操作性能，实验证明效果显著。**

- **链接: [https://arxiv.org/pdf/2510.00491](https://arxiv.org/pdf/2510.00491)**

> **作者:** Han Zhou; Jinjin Cao; Liyuan Ma; Xueji Fang; Guo-jun Qi
>
> **摘要:** Learning diverse manipulation skills for real-world robots is severely bottlenecked by the reliance on costly and hard-to-scale teleoperated demonstrations. While human videos offer a scalable alternative, effectively transferring manipulation knowledge is fundamentally hindered by the significant morphological gap between human and robotic embodiments. To address this challenge and facilitate skill transfer from human to robot, we introduce Traj2Action, a novel framework that bridges this embodiment gap by using the 3D trajectory of the operational endpoint as a unified intermediate representation, and then transfers the manipulation knowledge embedded in this trajectory to the robot's actions. Our policy first learns to generate a coarse trajectory, which forms a high-level motion plan by leveraging both human and robot data. This plan then conditions the synthesis of precise, robot-specific actions (e.g., orientation and gripper state) within a co-denoising framework. Our work centers on two core objectives: first, the systematic verification of the Traj2Action framework's effectiveness-spanning architectural design, cross-task generalization, and data efficiency and second, the revelation of key laws that govern robot policy learning during the integration of human hand demonstration data. This research focus enables us to provide a scalable paradigm tailored to address human-to-robot skill transfer across morphological gaps. Extensive real-world experiments on a Franka robot demonstrate that Traj2Action boosts the performance by up to 27% and 22.25% over $\pi_0$ baseline on short- and long-horizon real-world tasks, and achieves significant gains as human data scales in robot policy learning.
>
---
#### [replaced 023] Haptic Light-Emitting Diodes: Miniature, Luminous Tactile Actuators
- **分类: cs.HC; cs.RO**

- **简介: 该论文提出HLED，一种将光直接转换为机械力的微型触觉执行器，解决人机交互中的触觉反馈问题。通过光热效应产生位移，兼具发光与驱动功能。**

- **链接: [https://arxiv.org/pdf/2601.11043](https://arxiv.org/pdf/2601.11043)**

> **作者:** Max Linnander; Yon Visell
>
> **摘要:** We present Haptic Light-Emitting Diodes (HLEDs), luminous thermopneumatic actuators that directly convert pulsed light into mechanical forces and displacements. Each device packages a miniature surface-mount LED in a gas-filled cavity that contains a low-inertia graphite photoabsorber. The cavity is sealed by an elastic membrane, which functions as a working diaphragm. Brief optical pulses heat the photoabsorber, which heats the gas. The resulting rapid pressure increases generate forces and displacements at the working diaphragm. Millimeter-scale HLEDs produce forces exceeding 0.4 N and displacements of 0.9 mm at low voltages, with 5 to 100 ms response times, making them attractive as actuators providing tactile feedback in human-machine interfaces. Unusually, these actuators are also light-emitting, as a fraction of optical energy is transmitted through the membrane. These photomechanical actuators have many potential applications in tactile displays, human interface engineering, wearable computing, and other areas.
>
---
#### [replaced 024] CompliantVLA-adaptor: VLM-Guided Variable Impedance Action for Safe Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决接触密集场景下的安全控制问题。通过引入VLM指导的变阻抗控制，提升操作安全性与成功率。**

- **链接: [https://arxiv.org/pdf/2601.15541](https://arxiv.org/pdf/2601.15541)**

> **作者:** Heng Zhang; Wei-Hsing Huang; Qiyi Tong; Gokhan Solak; Puze Liu; Kaidi Zhang; Sheng Liu; Jan Peters; Yu She; Arash Ajoudani
>
> **备注:** under review
>
> **摘要:** We propose a CompliantVLA-adaptor that augments the state-of-the-art Vision-Language-Action (VLA) models with vision-language model (VLM)-informed context-aware variable impedance control (VIC) to improve the safety and effectiveness of contact-rich robotic manipulation tasks. Existing VLA systems (e.g., RDT, Pi0.5, OpenVLA-oft) typically output position, but lack force-aware adaptation, leading to unsafe or failed interactions in physical tasks involving contact, compliance, or uncertainty. In the proposed CompliantVLA-adaptor, a VLM interprets task context from images and natural language to adapt the stiffness and damping parameters of a VIC controller. These parameters are further regulated using real-time force/torque feedback to ensure interaction forces remain within safe thresholds. We demonstrate that our method outperforms the VLA baselines on a suite of complex contact-rich tasks, both in simulation and the real world, with improved success rates and reduced force violations. This work presents a promising path towards a safe foundation model for physical contact-rich manipulation. We release our code, prompts, and force-torque-impedance-scenario context datasets at this https URL.
>
---
#### [replaced 025] BiGraspFormer: End-to-End Bimanual Grasp Transformer
- **分类: cs.RO**

- **简介: 该论文属于双臂抓取任务，旨在解决传统方法协调性差的问题。提出BiGraspFormer框架，直接生成协同双臂抓取方案，提升抓取效率与安全性。**

- **链接: [https://arxiv.org/pdf/2509.19142](https://arxiv.org/pdf/2509.19142)**

> **作者:** Kangmin Kim; Seunghyeok Back; Geonhyup Lee; Sangbeom Lee; Sangjun Noh; Kyoobin Lee
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Bimanual grasping is essential for robots to handle large and complex objects. However, existing methods either focus solely on single-arm grasping or employ separate grasp generation and bimanual evaluation stages, leading to coordination problems including collision risks and unbalanced force distribution. To address these limitations, we propose BiGraspFormer, a unified end-to-end transformer framework that directly generates coordinated bimanual grasps from object point clouds. Our key idea is the Single-Guided Bimanual (SGB) strategy, which first generates diverse single grasp candidates using a transformer decoder, then leverages their learned features through specialized attention mechanisms to jointly predict bimanual poses and quality scores. This conditioning strategy reduces the complexity of the 12-DoF search space while ensuring coordinated bimanual manipulation. Comprehensive simulation experiments and real-world validation demonstrate that BiGraspFormer consistently outperforms existing methods while maintaining efficient inference speed (<0.05s), confirming the effectiveness of our framework. Code and supplementary materials are available at this https URL
>
---
#### [replaced 026] KEEP: A KV-Cache-Centric Memory Management System for Efficient Embodied Planning
- **分类: cs.RO; cs.AI; cs.SE**

- **简介: 该论文属于 embodied planning 任务，旨在解决 LLMs 中记忆管理效率低的问题。提出 KEEP 系统，通过优化 KV 缓存管理提升推理速度与效果。**

- **链接: [https://arxiv.org/pdf/2602.23592](https://arxiv.org/pdf/2602.23592)**

> **作者:** Zebin Yang; Tong Xie; Baotong Lu; Shaoshan Liu; Bo Yu; Meng Li
>
> **备注:** DAC 2026
>
> **摘要:** Memory-augmented Large Language Models (LLMs) have demonstrated remarkable capability for complex and long-horizon embodied planning. By keeping track of past experiences and environmental states, memory enables LLMs to maintain a global view, thereby avoiding repetitive exploration. However, existing approaches often store the memory as raw text, leading to excessively long prompts and high prefill latency. While it is possible to store and reuse the KV caches, the efficiency benefits are greatly undermined due to frequent KV cache updates. In this paper, we propose KEEP, a KV-cache-centric memory management system for efficient embodied planning. KEEP features 3 key innovations: (1) a Static-Dynamic Memory Construction algorithm that reduces KV cache recomputation by mixed-granularity memory group; (2) a Multi-hop Memory Re-computation algorithm that dynamically identifies important cross-attention among different memory groups and reconstructs memory interactions iteratively; (3) a Layer-balanced Memory Loading that eliminates unbalanced KV cache loading and cross-attention computation across different layers. Extensive experimental results have demonstrated that KEEP achieves 2.68x speedup with negligible accuracy loss compared with text-based memory methods on ALFRED dataset. Compared with the KV re-computation method CacheBlend (EuroSys'25), KEEP shows 4.13% success rate improvement and 1.90x time-to-first-token (TTFT) reduction. Our code is available on this https URL.
>
---
#### [replaced 027] SHaRe-RL: Structured, Interactive Reinforcement Learning for Contact-Rich Industrial Assembly Tasks
- **分类: cs.RO**

- **简介: 该论文针对高混小批量工业装配任务，解决机器人在接触密集环境中的学习效率与安全性问题。提出SHaRe-RL框架，结合先验知识与人类示范，实现高效安全的在线学习。**

- **链接: [https://arxiv.org/pdf/2509.13949](https://arxiv.org/pdf/2509.13949)**

> **作者:** Jannick Stranghöner; Philipp Hartmann; Marco Braun; Sebastian Wrede; Klaus Neumann
>
> **备注:** 8 pages, 8 figures, accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** High-mix low-volume (HMLV) industrial assembly, common in small and medium-sized enterprises (SMEs), requires the same precision, safety, and reliability as high-volume automation while remaining flexible to product variation and environmental uncertainty. Current robotic systems struggle to meet these demands. Manual programming is brittle and costly to adapt, while learning-based methods suffer from poor sample efficiency and unsafe exploration in contact-rich tasks. To address this, we present SHaRe-RL, a reinforcement learning framework that leverages multiple sources of prior knowledge. By (i) structuring skills into manipulation primitives, (ii) incorporating human demonstrations and online corrections, and (iii) bounding interaction forces with per-axis compliance, SHaRe-RL enables efficient and safe online learning for long-horizon, contact-rich industrial assembly tasks. Experiments on the insertion of industrial Harting connector modules with 0.2-0.4 mm clearance demonstrate that SHaRe-RL achieves reliable performance within practical time budgets. Our results show that process expertise, without requiring robotics or RL knowledge, can meaningfully contribute to learning, enabling safer, more robust, and more economically viable deployment of RL for industrial assembly.
>
---
#### [replaced 028] One-Shot Badminton Shuttle Detection for Mobile Robots
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于目标检测任务，解决移动机器人对羽毛球的实时检测问题。构建了数据集并优化YOLOv8模型，提升动态视角下的检测效果。**

- **链接: [https://arxiv.org/pdf/2603.06691](https://arxiv.org/pdf/2603.06691)**

> **作者:** Florentin Dipner; William Talbot; Turcan Tuna; Andrei Cramariuc; Marco Hutter
>
> **摘要:** This paper presents a robust one-shot badminton shuttlecock detection framework for non-stationary robots. To address the lack of egocentric shuttlecock detection datasets, we introduce a dataset of 20,510 semi-automatically annotated frames captured across 11 distinct backgrounds in diverse indoor and outdoor environments, and categorize each frame into one of three difficulty levels. For labeling, we present a novel semi-automatic annotation pipeline, that enables efficient labeling from stationary camera footage. We propose a metric suited to our downstream use case and fine-tune a YOLOv8 network optimized for real-time shuttlecock detection, achieving an F1-score of 0.86 under our metric in test environments similar to training, and 0.70 in entirely unseen environments. Our analysis reveals that detection performance is critically dependent on shuttlecock size and background texture complexity. Qualitative experiments confirm their applicability to robots with moving cameras. Unlike prior work with stationary camera setups, our detector is specifically designed for the egocentric, dynamic viewpoints of mobile robots, providing a foundational building block for downstream tasks, including tracking, trajectory estimation, and system (re)-initialization.
>
---
#### [replaced 029] Crowd-FM: Learned Optimal Selection of Conditional Flow Matching-generated Trajectories for Crowd Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决密集人群中的安全导航与人类行为相似性问题。提出Crowd-FM方法，结合条件流匹配与人类轨迹评分，提升导航成功率与自然度。**

- **链接: [https://arxiv.org/pdf/2602.06698](https://arxiv.org/pdf/2602.06698)**

> **作者:** Antareep Singha; Laksh Nanwani; Mathai Mathew P.; Samkit Jain; Phani Teja Singamaneni; Arun Kumar Singh; K. Madhava Krishna
>
> **备注:** Accepted at IEEE ICRA 2026. Authors Antareep Singha and Laksh Nanwani have equal contributions
>
> **摘要:** Safe and computationally efficient local planning for mobile robots in dense, unstructured human crowds remains a fundamental challenge. Moreover, ensuring that robot trajectories are similar to how a human moves will increase the acceptance of the robot in human environments. In this paper, we present Crowd-FM, a learning-based approach to address both safety and human-likeness challenges. Our approach has two novel components. First, we train a Conditional Flow-Matching (CFM) policy over a dataset of optimally controlled trajectories to learn a set of collision-free primitives that a robot can choose at any given scenario. The chosen optimal control solver can generate multi-modal collision-free trajectories, allowing the CFM policy to learn a diverse set of maneuvers. Secondly, we learn a score function over a dataset of human demonstration trajectories that provides a human-likeness score for the flow primitives. At inference time, computing the optimal trajectory requires selecting the one with the highest score. Our approach improves the state-of-the-art by showing that our CFM policy alone can produce collision-free navigation with a higher success rate than existing learning-based baselines. Furthermore, when augmented with inference-time refinement, our approach can outperform even expensive optimisation-based planning approaches. Finally, we validate that our scoring network can select trajectories closer to the expert data than a manually designed cost function.
>
---
#### [replaced 030] Closed-Loop Action Chunks with Dynamic Corrections for Training-Free Diffusion Policy
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，解决动态场景下策略适应性差的问题。提出DCDP框架，结合动作块与实时修正，提升系统响应能力和适应性。**

- **链接: [https://arxiv.org/pdf/2603.01953](https://arxiv.org/pdf/2603.01953)**

> **作者:** Pengyuan Wu; Pingrui Zhang; Zhigang Wang; Dong Wang; Bin Zhao; Xuelong Li
>
> **备注:** Accepted by ICRA2026
>
> **摘要:** Diffusion-based policies have achieved remarkable results in robotic manipulation but often struggle to adapt rapidly in dynamic scenarios, leading to delayed responses or task failures. We present DCDP, a Dynamic Closed-Loop Diffusion Policy framework that integrates chunk-based action generation with real-time correction. DCDP integrates a self-supervised dynamic feature encoder, cross-attention fusion, and an asymmetric action encoder-decoder to inject environmental dynamics before action execution, achieving real-time closed-loop action correction and enhancing the system's adaptability in dynamic scenarios. In dynamic PushT simulations, DCDP improves adaptability by 19\% without retraining while requiring only 5\% additional computation. Its modular design enables plug-and-play integration, achieving both temporal coherence and real-time responsiveness in dynamic robotic scenarios, including real-world manipulation tasks. The project page is at: this https URL
>
---
#### [replaced 031] Real-time Capable Learning-based Visual Tool Pose Correction via Differentiable Simulation
- **分类: cs.RO**

- **简介: 该论文属于视觉姿态估计任务，旨在解决机器人手术中因传感器误差导致的位姿不准确问题。通过基于视觉Transformer的方法，结合可微分仿真实现实时、高精度的位姿校正。**

- **链接: [https://arxiv.org/pdf/2505.08875](https://arxiv.org/pdf/2505.08875)**

> **作者:** Shuyuan Yang; Zonghe Chua
>
> **摘要:** Autonomy in robot-assisted minimally invasive surgery has the potential to reduce surgeon cognitive and task load, thereby increasing procedural efficiency. However, implementing accurate autonomous control can be difficult due to poor end-effector proprioception. Joint encoder readings are typically inaccurate due to kinematic non-idealities in their cable-driven transmissions. Vision-based pose estimation approaches are highly effective, but lack real-time capability, generalizability, or can be hard to train. In this work, we demonstrate a real-time capable, Vision Transformer-based pose estimation approach that is trained using end-to-end differentiable kinematics and rendering. We demonstrate the potential of this approach to correct for noisy pose estimates through a real robot dataset and the potential real-time processing ability. Our approach is able to reduce more than 50% of hand-eye translation errors in the dataset, reaching the same performance level as an existing optimization-based method. Our approach is four times faster, and capable of near real-time inference at 22 Hz. A zero-shot prediction on an unseen dataset shows good generalization ability, and can be further finetuned for increased performance without human labeling.
>
---
#### [replaced 032] Ontological foundations for contrastive explanatory narration of robot plans
- **分类: cs.RO; cs.AI; cs.IR; cs.LO**

- **简介: 该论文属于人机交互任务，旨在解决机器人决策解释问题。通过构建本体模型和新算法，比较不同计划并生成对比性解释，提升解释效果。**

- **链接: [https://arxiv.org/pdf/2509.22493](https://arxiv.org/pdf/2509.22493)**

> **作者:** Alberto Olivares-Alarcos; Sergi Foix; Júlia Borràs; Gerard Canal; Guillem Alenyà
>
> **摘要:** Mutual understanding of artificial agents' decisions is key to ensuring a trustworthy and successful human-robot interaction. Hence, robots are expected to make reasonable decisions and communicate them to humans when needed. In this article, the focus is on an approach to modeling and reasoning about the comparison of two competing plans, so that robots can later explain the divergent result. First, a novel ontological model is proposed to formalize and reason about the differences between competing plans, enabling the classification of the most appropriate one (e.g., the shortest, the safest, the closest to human preferences, etc.). This work also investigates the limitations of a baseline algorithm for ontology-based explanatory narration. To address these limitations, a novel algorithm is presented, leveraging divergent knowledge between plans and facilitating the construction of contrastive narratives. Through empirical evaluation, it is observed that the explanations excel beyond the baseline method.
>
---
#### [replaced 033] DreamFlow: Local Navigation Beyond Observation via Conditional Flow Matching in the Latent Space
- **分类: cs.RO**

- **简介: 该论文属于机器人局部导航任务，解决复杂环境中感知不足导致的导航失败问题。通过条件流匹配方法扩展感知范围，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.02976](https://arxiv.org/pdf/2603.02976)**

> **作者:** Jiwon Park; Dongkyu Lee; I Made Aswin Nahrendra; Jaeyoung Lim; Hyun Myung
>
> **摘要:** Local navigation in cluttered environments often suffers from dense obstacles and frequent local minima. Conventional local planners rely on heuristics and are prone to failure, while deep reinforcement learning(DRL)based approaches provide adaptability but are constrained by limited onboard sensing. These limitations lead to navigation failures because the robot cannot perceive structures outside its field of view. In this paper, we propose DreamFlow, a DRL-based local navigation framework that extends the robot's perceptual horizon through conditional flow matching(CFM). The proposed CFM based prediction module learns probabilistic mapping between local height map latent representation and broader spatial representation conditioned on navigation context. This enables the navigation policy to predict unobserved environmental features and proactively avoid potential local minima. Experimental results demonstrate that DreamFlow outperforms existing methods in terms of latent prediction accuracy and navigation performance in simulation. The proposed method was further validated in cluttered real world environments with a quadrupedal robot. The project page is available at this https URL.
>
---
#### [replaced 034] EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于对象目标导航任务，解决小模型在本地设备上导航成功率低和延迟高的问题。提出EfficientNav，通过语义检索和缓存优化提升性能。**

- **链接: [https://arxiv.org/pdf/2510.18546](https://arxiv.org/pdf/2510.18546)**

> **作者:** Zebin Yang; Sunjian Zheng; Tong Xie; Tianshi Xu; Bo Yu; Fan Wang; Jie Tang; Shaoshan Liu; Meng Li
>
> **备注:** NeurIPS 2025
>
> **摘要:** Object-goal navigation (ObjNav) tasks an agent with navigating to the location of a specific object in an unseen environment. Embodied agents equipped with large language models (LLMs) and online constructed navigation maps can perform ObjNav in a zero-shot manner. However, existing agents heavily rely on giant LLMs on the cloud, e.g., GPT-4, while directly switching to small LLMs, e.g., LLaMA3.2-11b, suffer from significant success rate drops due to limited model capacity for understanding complex navigation maps, which prevents deploying ObjNav on local devices. At the same time, the long prompt introduced by the navigation map description will cause high planning latency on local devices. In this paper, we propose EfficientNav to enable on-device efficient LLM-based zero-shot ObjNav. To help the smaller LLMs better understand the environment, we propose semantics-aware memory retrieval to prune redundant information in navigation maps. To reduce planning latency, we propose discrete memory caching and attention-based memory clustering to efficiently save and re-use the KV cache. Extensive experimental results demonstrate that EfficientNav achieves 11.1% improvement in success rate on HM3D benchmark over GPT-4-based baselines, and demonstrates 6.7x real-time latency reduction and 4.7x end-to-end latency reduction over GPT-4 planner. Our code is available on this https URL.
>
---
#### [replaced 035] DefVINS: Visual-Inertial Odometry for Deformable Scenes
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-惯性里程计任务，解决变形场景下的定位问题。提出DefVINS方法，通过分解状态处理刚性和非刚性运动，并构建基准测试验证效果。**

- **链接: [https://arxiv.org/pdf/2601.00702](https://arxiv.org/pdf/2601.00702)**

> **作者:** Samuel Cerezo; Javier Civera
>
> **备注:** 4 figures, 2 tables. Submitted to RA-L
>
> **摘要:** Deformable scenes violate the rigidity assumptions underpinning classical visual--inertial odometry (VIO), often leading to over-fitting to local non-rigid motion or to severe camera pose drift when deformation dominates visual parallax. In this paper, we introduce DefVINS, the first visual-inertial odometry pipeline designed to operate in deformable environments. Our approach models the odometry state by decomposing it into a rigid, IMU-anchored component and a non-rigid scene warp represented by an embedded deformation graph. As a second contribution, we present VIMandala, the first benchmark containing real images and ground-truth camera poses for visual-inertial odometry in deformable scenes. In addition, we augment the synthetic Drunkard's benchmark with simulated inertial measurements to further evaluate our pipeline under controlled conditions. We also provide an observability analysis of the visual-inertial deformable odometry problem, characterizing how inertial measurements constrain camera motion and render otherwise unobservable modes identifiable in the presence of deformation. This analysis motivates the use of IMU anchoring and leads to a conditioning-based activation strategy that avoids ill-posed updates under poor excitation. Experimental results on both the synthetic Drunkard's and our real VIMandala benchmarks show that DefVINS outperforms rigid visual--inertial and non-rigid visual odometry baselines. Our source code and data will be released upon acceptance.
>
---
#### [replaced 036] World Models for Learning Dexterous Hand-Object Interactions from Human Videos
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，解决手-物体交互建模问题。通过引入DexWM模型，利用视频中的手指关键点进行训练，提升对精细动作的预测与迁移能力。**

- **链接: [https://arxiv.org/pdf/2512.13644](https://arxiv.org/pdf/2512.13644)**

> **作者:** Raktim Gautam Goswami; Amir Bar; David Fan; Tsung-Yen Yang; Gaoyue Zhou; Prashanth Krishnamurthy; Michael Rabbat; Farshad Khorrami; Yann LeCun
>
> **摘要:** Modeling dexterous hand-object interactions is challenging as it requires understanding how subtle finger motions influence the environment through contact with objects. While recent world models address interaction modeling, they typically rely on coarse action spaces that fail to capture fine-grained dexterity. We, therefore, introduce DexWM, a Dexterous Interaction World Model that predicts future latent states of the environment conditioned on past states and dexterous actions. To overcome the scarcity of finely annotated dexterous datasets, DexWM represents actions using finger keypoints extracted from egocentric videos, enabling training on over 900 hours of human and non-dexterous robot data. Further, to accurately model dexterity, we find that predicting visual features alone is insufficient; therefore, we incorporate an auxiliary hand consistency loss that enforces accurate hand configurations. DexWM outperforms prior world models conditioned on text, navigation, or full-body actions in future-state prediction and demonstrates strong zero-shot transfer to unseen skills on a Franka Panda arm with an Allegro gripper, surpassing Diffusion Policy by over 50% on average across grasping, placing, and reaching tasks.
>
---
#### [replaced 037] Vision-Language Models for Infrared Industrial Sensing in Additive Manufacturing Scene Description
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言模型在工业红外感知中的应用任务，旨在解决低光环境下无标签数据的工件检测问题。通过预处理红外图像并适配CLIP模型，实现无需重新训练的零样本检测。**

- **链接: [https://arxiv.org/pdf/2512.11098](https://arxiv.org/pdf/2512.11098)**

> **作者:** Nazanin Mahjourian; Vinh Nguyen
>
> **摘要:** Many manufacturing environments operate in low-light conditions or within enclosed machines where conventional vision systems struggle. Infrared cameras provide complementary advantages in such environments. Simultaneously, supervised AI systems require large labeled datasets, which makes zero-shot learning frameworks more practical for applications including infrared cameras. Recent advances in vision-language foundation models (VLMs) offer a new path in zero-shot predictions from paired image-text representations. However, current VLMs cannot understand infrared camera data since they are trained on RGB data. This work introduces VLM-IRIS (Vision-Language Models for InfraRed Industrial Sensing), a zero-shot framework that adapts VLMs to infrared data by preprocessing infrared images captured by a FLIR Boson sensor into RGB-compatible inputs suitable for CLIP-based encoders. We demonstrate zero-shot workpiece presence detection on a 3D printer bed where temperature differences between the build plate and workpieces make the task well-suited for thermal imaging. VLM-IRIS converts the infrared images to magma representation and applies centroid prompt ensembling with a CLIP ViT-B/32 encoder to achieve high accuracy on infrared images without any model retraining. These findings demonstrate that the proposed improvements to VLMs can be effectively extended to thermal applications for label-free monitoring.
>
---
