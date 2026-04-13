# 机器人 cs.RO

- **最新发布 37 篇**

- **更新 22 篇**

## 最新发布

#### [new 001] Musculoskeletal Motion Imitation for Learning Personalized Exoskeleton Control Policy in Impaired Gait
- **分类: cs.RO**

- **简介: 该论文属于个性化外骨骼控制策略学习任务，旨在解决传统方法依赖大量数据或迭代优化的问题。通过结合生理肌肉骨骼模拟与强化学习，生成高效且个性化的辅助控制策略。**

- **链接: [https://arxiv.org/pdf/2604.09431](https://arxiv.org/pdf/2604.09431)**

> **作者:** Itak Choi; Ilseung Park; Eni Halilaj; Inseung Kang
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Designing generalizable control policies for lower-limb exoskeletons remains fundamentally constrained by exhaustive data collection or iterative optimization procedures, which limit accessibility to clinical populations. To address this challenge, we introduce a device-agnostic framework that combines physiologically plausible musculoskeletal simulation with reinforcement learning to enable scalable personalized exoskeleton assistance for both able-bodied and clinical populations. Our control policies not only generate physiologically plausible locomotion dynamics but also capture clinically observed compensatory strategies under targeted muscular deficits, providing a unified computational model of both healthy and pathological gait. Without task-specific tuning, the resulting exoskeleton control policies produce assistive torque profiles at the hip and ankle that align with state-of-the-art profiles validated in human experiments, while consistently reducing metabolic cost across walking speeds. For simulated impaired-gait models, the learned control policies yield asymmetric, deficit-specific exoskeleton assistance that improves both energetic efficiency and bilateral kinematic symmetry without explicit prescription of the target gait pattern. These results demonstrate that physiologically plausible musculoskeletal simulation via reinforcement learning can serve as a scalable foundation for personalized exoskeleton control across both able-bodied and clinical populations, eliminating the need for extensive physical trials.
>
---
#### [new 002] Towards Lifelong Aerial Autonomy: Geometric Memory Management for Continual Visual Place Recognition in Dynamic Environments
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于视觉定位任务，解决动态环境中持续学习的灾难性遗忘问题。提出一种异构记忆框架，结合静态地理锚点与动态经验回放，提升长期空中自主能力。**

- **链接: [https://arxiv.org/pdf/2604.09038](https://arxiv.org/pdf/2604.09038)**

> **作者:** Xingyu Shao; Zhiqiang Yan; Liangzheng Sun; Mengfan He; Chao Chen; Jinhui Zhang; Chunyu Li; Ziyang Meng
>
> **摘要:** Robust geo-localization in changing environmental conditions is critical for long-term aerial autonomy. While visual place recognition (VPR) models perform well when airborne views match the training domain, adapting them to shifting distributions during sequential missions triggers catastrophic forgetting. Existing continual learning (CL) methods often fail here because geographic features exhibit severe intra-class variations. In this work, we formulate aerial VPR as a mission-based domain-incremental learning (DIL) problem and propose a novel heterogeneous memory framework. To respect strict onboard storage constraints, our "Learn-and-Dispose" pipeline decouples geographic knowledge into static satellite anchors (preserving global geometric priors) and a dynamic experience replay buffer (retaining domain-specific features). We introduce a spatially-constrained allocation strategy that optimizes buffer selection based on sample difficulty or feature space diversity. To facilitate systematic assessment, we provide three evaluation criteria and a comprehensive benchmark derived from 21 diverse mission sequences. Extensive experiments demonstrate that our architecture significantly boosts spatial generalization; our diversity-driven buffer selection outperforms the random baseline by 7.8% in knowledge retention. Unlike class-mean preservation methods that fail in unstructured environments, maximizing structural diversity achieves a superior plasticity-stability balance and ensures order-agnostic robustness across randomized sequences. These results prove that maintaining structural feature coverage is more critical than sample difficulty for resolving catastrophic forgetting in lifelong aerial autonomy.
>
---
#### [new 003] Adaptor: Advancing Assistive Teleoperation with Few-Shot Learning and Cross-Operator Generalization
- **分类: cs.RO**

- **简介: 该论文属于辅助远程操作任务，解决因操作者差异导致的意图识别不稳定问题。提出Adaptor框架，通过预处理和策略学习实现跨操作者鲁棒识别。**

- **链接: [https://arxiv.org/pdf/2604.09462](https://arxiv.org/pdf/2604.09462)**

> **作者:** Yu Liu; Yihang Yin; Tianlv Huang; Fei Yan; Yuan Xu; Weinan Hong; Wei Han; Yue Cao; Xiangyu Chen; Zipei Fan; Xuan Song
>
> **备注:** Accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** Assistive teleoperation enhances efficiency via shared control, yet inter-operator variability, stemming from diverse habits and expertise, induces highly heterogeneous trajectory distributions that undermine intent recognition stability. We present Adaptor, a few-shot framework for robust cross-operator intent recognition. The Adaptor bridges the domain gap through two stages: (i) preprocessing, which models intent uncertainty by synthesizing trajectory perturbations via noise injection and performs geometry-aware keyframe extraction; and (ii) policy learning, which encodes the processed trajectories with an Intention Expert and fuses them with the pre-trained vision-language model context to condition an Action Expert for action generation. Experiments on real-world and simulated benchmarks demonstrate that Adaptor achieves state-of-the-art performance, improving success rates and efficiency over baselines. Moreover, the method exhibits low variance across operators with varying expertise, demonstrating robust cross-operator generalization.
>
---
#### [new 004] One Interface, Many Robots: Unified Real-Time Low-Level Motion Planning for Collaborative Arms
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，旨在解决协作机械臂在异构平台上的实时低层轨迹控制问题。通过统一接口和多项式插值与二次规划结合的方法，实现平滑精确的轨迹生成。**

- **链接: [https://arxiv.org/pdf/2604.08787](https://arxiv.org/pdf/2604.08787)**

> **作者:** Yue Feng; Weicheng Huang; I-Ming Chen
>
> **摘要:** This paper proposes a common interface for real-time low-level motion planning of collaborative robotic arms, aimed at enabling broader applicability and improved portability across heterogeneous hardware platforms. In previous work, we introduced WinGs Operating Studio (WOS), a middleware solution that abstracts diverse robotic components into uniform software resources and provides a broad suite of language-agnostic APIs. This paper specifically focuses on its minimal yet flexible interface for real-time end-effector trajectory control. By employing an n-degree polynomial interpolator in conjunction with a quadratic programming solver, the proposed method generates smooth, continuously differentiable trajectories with precise position, velocity, and acceleration profiles. We validate our approach in three distinct scenarios. First, in an offline demonstration, a collaborative arm accurately draws various geometric shapes on paper. Second, in an interruptible, low-frequency re-planning setting, a robotic manipulator grasps a dynamic object placed on a moving mobile robot. Finally, we conducted a teleoperation experiment in which one robotic arm controlled another to perform a series of dexterous manipulations, confirming the proposed method's reliability, versatility, and ease of use.
>
---
#### [new 005] VAG: Dual-Stream Video-Action Generation for Embodied Data Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出VAG框架，解决机器人合成数据生成中视频与动作对齐问题，通过双流结构联合生成视频和动作，提升跨模态一致性。**

- **链接: [https://arxiv.org/pdf/2604.09330](https://arxiv.org/pdf/2604.09330)**

> **作者:** Xiaolei Lang; Yang Wang; Yukun Zhou; Chaojun Ni; Kerui Li; Jiagang Zhu; Tianze Liu; Jiajun Lv; Xingxing Zuo; Yun Ye; Guan Huang; Xiaofeng Wang; Zheng Zhu
>
> **摘要:** Recent advances in robot foundation models trained on large-scale human teleoperation data have enabled robots to perform increasingly complex real-world tasks. However, scaling these systems remains difficult because collecting task-specific demonstrations is expensive and labor-intensive. Synthetic data, especially generated videos, offer a promising direction, but existing World Models (WMs) are not directly suitable for policy learning since they do not provide paired action trajectories. World-Action (WA) models partially address this by predicting actions with visual outputs, yet often lack strong video-action alignment, while two-stage pipelines that generate video first and then infer actions introduce inefficiency and error accumulation. To address these limitations, we propose VAG, a unified flow-matching-based dual-stream framework that jointly generates video and action under visual and language conditioning. By synchronizing denoising in both branches and using an adaptive 3D pooling mechanism to transfer compact global video context to the action branch, VAG improves cross-modal consistency during generation. Across both simulated and real-world settings, VAG produces aligned video-action pairs with competitive prediction quality, supports executable trajectory replay, and provides useful synthetic pretraining data that improves downstream policy generalization, indicating its potential as a practical world-action model for embodied data synthesis.
>
---
#### [new 006] On the Terminology and Geometric Aspects of Redundant Parallel Manipulators
- **分类: cs.RO; math.DS**

- **简介: 该论文属于机器人学领域，旨在解决冗余并联机械臂的术语和几何问题。通过建立统一术语和模型，分析冗余驱动对奇异性的影响，提出DOA概念进行分类。**

- **链接: [https://arxiv.org/pdf/2604.09156](https://arxiv.org/pdf/2604.09156)**

> **作者:** Andreas Mueller
>
> **摘要:** Parallel kinematics machines (PKM) can exhibit kinematic as well as actuation redundancy. While the meaning of kinematic redundancy has been clarified already for serial manipulators, actuation redundancy, that is only possible in PKM, is differently classified in the literature. In this paper a consistent terminology for general redundant PKM is proposed. A kinematic model is introduced with the configuration space (c-space) as central part. The notion of kinematic redundancy is recalled for PKM. C-space, output, and input singularities are distinguished. The significance of the c-space geometry is emphasized, and it is pointed out geometrically that input singularities can be avoided by redundant actuation schemes. In order to distinguish different actuation schemes of PKM a non-linear control system is introduced whose dynamics evolves on the c-space. The degree of actuation (DOA) is introduced as the number of independent control vector fields, and PKM are classified as full-actuated and underactuated. Relating this DOA to the degree of freedom (DOF) allows to classify the actuation redundancy.
>
---
#### [new 007] Simulation of Adaptive Running with Flexible Sports Prosthesis using Reinforcement Learning of Hybrid-link System
- **分类: cs.RO**

- **简介: 该论文属于运动仿真任务，旨在解决假肢设计与性能优化问题。通过强化学习与混合链系统结合，模拟截肢者跑步动作，分析不同刚度对代谢成本的影响。**

- **链接: [https://arxiv.org/pdf/2604.08882](https://arxiv.org/pdf/2604.08882)**

> **作者:** Yuta Shimane; Ko Yamamoto
>
> **摘要:** This study proposes a reinforcement learning-based adaptive running motion simulation for a unilateral transtibial amputee with the flexibility of a leaf-spring-type sports prosthesis using hybrid-link system. The design and selection of sports prostheses often rely on trial and error. A comprehensive whole-body dynamics analysis that considers the interaction between human motion and prosthetic deformation could provide valuable insights for user-specific design and selection. The hybrid-link system facilitates whole-body dynamics analysis by incorporating the Piece-wise Constant Strain model to represent the flexible deformation of the prosthesis. Based on this system, the simulation methodology generates whole-body dynamic motions of a unilateral transtibial amputee through a reinforcement learning-based approach, which combines imitation learning from motion capture data with accurate prosthetic dynamics computation. We simulated running motions under different virtual prosthetic stiffness conditions and analyzed the metabolic cost of transport obtained from the simulations, suggesting that variations in stiffness influence running performance. Our findings demonstrate the potential of this approach for simulation and analysis under virtual conditions that differ from real conditions.
>
---
#### [new 008] LEGO: Latent-space Exploration for Geometry-aware Optimization of Humanoid Kinematic Design
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人设计任务，旨在解决传统设计依赖直觉、缺乏系统方法的问题。通过学习现有设计构建潜在空间，利用人类运动数据优化机器人结构，实现自动化设计。**

- **链接: [https://arxiv.org/pdf/2604.08636](https://arxiv.org/pdf/2604.08636)**

> **作者:** Jihwan Yoon; Taemoon Jeong; Jeongeun Park; Chanwoo Kim; Jaewoon Kwon; Yonghyeon Lee; Kyungjae Lee; Sungjoon Choi
>
> **备注:** Aceepted in ICRA 2026
>
> **摘要:** Designing robot morphologies and kinematics has traditionally relied on human intuition, with little systematic foundation. Motion-design co-optimization offers a promising path toward automation, but two major challenges remain: (i) the vast, unstructured design space and (ii) the difficulty of constructing task-specific loss functions. We propose a new paradigm that minimizes human involvement by (i) learning the design search space from existing mechanical designs, rather than hand-crafting it, and (ii) defining the loss directly from human motion data via motion retargeting and Procrustes analysis. Using screw-theory-based joint axis representation and isometric manifold learning, we construct a compact, geometry-preserving latent space of humanoid upper body designs in which optimization is tractable. We then solve design optimization in this latent space using gradient-free optimization. Our approach establishes a principled framework for data-driven robot design and demonstrates that leveraging existing designs and human motion can effectively guide the automated discovery of novel robot design.
>
---
#### [new 009] Generative Simulation for Policy Learning in Physical Human-Robot Interaction
- **分类: cs.RO**

- **简介: 该论文属于物理人机交互任务，旨在解决训练数据稀缺问题。通过生成式模拟框架自动合成多样化场景，训练视觉仿生策略，实现零样本仿真到现实的迁移。**

- **链接: [https://arxiv.org/pdf/2604.08664](https://arxiv.org/pdf/2604.08664)**

> **作者:** Junxiang Wang; Xinwen Xu; Tiancheng Wu; Julian Millan; Nir Pechuk; Zackory Erickson
>
> **备注:** 9 pages, 3 figures, 2 tables
>
> **摘要:** Developing autonomous physical human-robot interaction (pHRI) systems is limited by the scarcity of large-scale training data to learn robust robot behaviors for real-world applications. In this paper, we introduce a zero-shot "text2sim2real" generative simulation framework that automatically synthesizes diverse pHRI scenarios from high-level natural-language prompts. Leveraging Large Language Models (LLMs) and Vision-Language Models (VLMs), our pipeline procedurally generates soft-body human models, scene layouts, and robot motion trajectories for assistive tasks. We utilize this framework to autonomously collect large-scale synthetic demonstration datasets and then train vision-based imitation learning policies operating on segmented point clouds. We evaluate our approach through a user study on two physically assistive tasks: scratching and bathing. Our learned policies successfully achieve zero-shot sim-to-real transfer, attaining success rates exceeding 80% and demonstrating resilience to unscripted human motion. Overall, we introduce the first generative simulation pipeline for pHRI applications, automating simulation environment synthesis, data collection, and policy learning. Additional information may be found on our project website: this https URL
>
---
#### [new 010] Physics-Informed Reinforcement Learning of Spatial Density Velocity Potentials for Map-Free Racing
- **分类: cs.RO**

- **简介: 该论文属于自主赛车任务，解决无地图环境下的实时控制问题。通过物理感知强化学习，提升车辆在不同赛道的性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.09499](https://arxiv.org/pdf/2604.09499)**

> **作者:** Shathushan Sivashangaran; Apoorva Khairnar; Sepideh Gohari; Vihaan Dutta; Azim Eskandarian
>
> **摘要:** Autonomous racing without prebuilt maps is a grand challenge for embedded robotics that requires kinodynamic planning from instantaneous sensor data at the acceleration and tire friction limits. Out-Of-Distribution (OOD) generalization to various racetrack configurations utilizes Machine Learning (ML) to encode the mathematical relation between sensor data and vehicle actuation for end-to-end control, with implicit localization. These comprise Behavioral Cloning (BC) that is capped to human reaction times and Deep Reinforcement Learning (DRL) which requires large-scale collisions for comprehensive training that can be infeasible without simulation but is arduous to transfer to reality, thus exhibiting greater performance than BC in simulation, but actuation instability on hardware. This paper presents a DRL method that parameterizes nonlinear vehicle dynamics from the spectral distribution of depth measurements with a non-geometric, physics-informed reward, to infer vehicle time-optimal and overtaking racing controls with an Artificial Neural Network (ANN) that utilizes less than 1% of the computation of BC and model-based DRL. Slaloming from simulation to reality transfer and variance-induced conservatism are eliminated with the combination of a physics engine exploit-aware reward and the replacement of an explicit collision penalty with an implicit truncation of the value horizon. The policy outperforms human demonstrations by 12% in OOD tracks on proportionally scaled hardware, by maximizing the friction circle with tire dynamics that resemble an empirical Pacejka tire model. System identification illuminates a functional bifurcation where the first layer compresses spatial observations to extract digitized track features with higher resolution in corner apexes, and the second encodes nonlinear dynamics.
>
---
#### [new 011] Task-Aware Bimanual Affordance Prediction via VLM-Guided Semantic-Geometric Reasoning
- **分类: cs.RO**

- **简介: 该论文研究双臂操作中的任务感知抓取预测问题，旨在解决几何与语义结合的交互区域识别和手臂分配难题。通过融合多视角RGB-D数据并利用视觉语言模型进行语义推理，提升复杂环境下的操作成功率。**

- **链接: [https://arxiv.org/pdf/2604.08726](https://arxiv.org/pdf/2604.08726)**

> **作者:** Fabian Hahne; Vignesh Prasad; Georgia Chalvatzaki; Jan Peters; Alap Kshirsagar
>
> **摘要:** Bimanual manipulation requires reasoning about where to interact with an object and which arm should perform each action, a joint affordance localization and arm allocation problem that geometry-only planners cannot resolve without semantic understanding of task intent. Existing approaches either treat affordance prediction as coarse part segmentation or rely on geometric heuristics for arm assignment, failing to jointly reason about task-relevant contact regions and arm allocation. We reframe bimanual manipulation as a joint affordance localization and arm allocation problem and propose a hierarchical framework for task-aware bimanual affordance prediction that leverages a Vision-Language Model (VLM) to generalize across object categories and task descriptions without requiring category-specific training. Our approach fuses multi-view RGB-D observations into a consistent 3D scene representation and generates global 6-DoF grasp candidates, which are then spatially and semantically filtered by querying the VLM for task-relevant affordance regions on each object, as well as for arm allocation to the individual objects, thereby ensuring geometric validity while respecting task semantics. We evaluate our method on a dual-arm platform across nine real-world manipulation tasks spanning four categories: parallel manipulation, coordinated stabilization, tool use, and human handover. Our approach achieves consistently higher task success rates than geometric and semantic baselines for task-oriented grasping, demonstrating that explicit semantic reasoning over affordances and arm allocation helps enable reliable bimanual manipulation in unstructured environments.
>
---
#### [new 012] Soft Electroadhesive Feet for Micro Aerial Robots Perching on Smooth and Curved Surfaces
- **分类: cs.RO**

- **简介: 该论文属于微飞行器着陆任务，解决其在光滑曲面稳定附着的问题。通过设计软电粘附足，实现可靠附着与快速脱离。**

- **链接: [https://arxiv.org/pdf/2604.09270](https://arxiv.org/pdf/2604.09270)**

> **作者:** Chen Liu; Sonu Feroz; Ketao Zhang
>
> **摘要:** Electroadhesion (EA) provides electrically switchable adhesion and is a promising mechanism for perching micro aerial robots on smooth surfaces. However, practical implementations of soft and stretchable EA pads for aerial perching remain limited. This work presents (i) an efficient workflow for fabricating soft, stretchable electroadhesive pads with sinusoidal wave and concentric-circle electrodes in multiple sizes, (ii) a controlled experimental comparison of normal and shear adhesion under inactive (0 kV) and active (4.8 kV) conditions using an Instron-based setup, and (iii) a perching demonstration using a Crazyflie quadrotor equipped with electroadhesive feet on flat and curved substrates. Experimental results show that shear adhesion dominates, reaching forces on the order of 3 N with partial pad contact, while normal adhesion is comparatively small and strongly dependent on substrate properties. The Crazyflie prototype demonstrates repeatable attachment on smooth plastic surfaces, including curved geometries, as well as rapid detachment when the voltage is removed. These results highlight the potential of soft electroadhesive feet for lightweight and reliable perching in micro aerial vehicles (MAVs).
>
---
#### [new 013] V-CAGE: Vision-Closed-Loop Agentic Generation Engine for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出V-CAGE，解决机器人操作数据生成中场景不真实、目标不可达的问题。通过语义布局与视觉验证，实现高质量数据集的自动化合成。**

- **链接: [https://arxiv.org/pdf/2604.09036](https://arxiv.org/pdf/2604.09036)**

> **作者:** Yaru Liu; Ao-bo Wang; Nanyang Ye
>
> **摘要:** Scaling Vision-Language-Action (VLA) models requires massive datasets that are both semantically coherent and physically feasible. However, existing scene generation methods often lack context-awareness, making it difficult to synthesize high-fidelity environments embedded with rich semantic information, frequently resulting in unreachable target positions that cause tasks to fail prematurely. We present V-CAGE (Vision-Closed-loop Agentic Generation Engine), an agentic framework for autonomous robotic data synthesis. Unlike traditional scripted pipelines, V-CAGE operates as an embodied agentic system, leveraging foundation models to bridge high-level semantic reasoning with low-level physical interaction. Specifically, we introduce Inpainting-Guided Scene Construction to systematically arrange context-aware layouts, ensuring that the generated scenes are both semantically structured and kinematically reachable. To ensure trajectory correctness, we integrate functional metadata with a Vision-Language Model based closed-loop verification mechanism, acting as a visual critic to rigorously filter out silent failures and sever the error propagation chain. Finally, to overcome the storage bottleneck of massive video datasets, we implement a perceptually-driven compression algorithm that achieves over 90\% filesize reduction without compromising downstream VLA training efficacy. By centralizing semantic layout planning and visual self-verification, V-CAGE automates the end-to-end pipeline, enabling the highly scalable synthesis of diverse, high-quality robotic manipulation datasets.
>
---
#### [new 014] Characterizing Lidar Range-Measurement Ambiguity due to Multiple Returns
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于激光雷达感知任务，旨在解决多回波导致的距离测量模糊问题，通过分析数据集并提出累积分布函数来评估其对定位的影响。**

- **链接: [https://arxiv.org/pdf/2604.09282](https://arxiv.org/pdf/2604.09282)**

> **作者:** Jason H. Rife; Yifan Li
>
> **备注:** Proceedings of the 38th International Technical Meeting of the Satellite Division of The Institute of Navigation (ION GNSS+ 2025), Baltimore, Maryland, September 2025, pp. 1949-1963
>
> **摘要:** Reliable position and attitude sensing is critical for highly automated vehicles that operate on conventional roadways. Lidar sensors are increasingly incorporated into pose-estimation systems. Despite its great utility, lidar is a complex sensor, and its performance in roadway environments is not yet well understood. For instance, it is often assumed in lidar-localization algorithms that a lidar will always identify a unique surface along a given raypath. However, this assumption is not always true, as ample prior evidence exists to suggest that lidar units may generate measurements probabilistically when more than one scattering surface appears within the lidar's conical beam. In this paper, we analyze lidar datasets to characterize cases with probabilistic returns along particular raypaths. Our contribution is to present representative cumulative distribution functions (CDFs) for raypaths observed by two different mechanically rotating lidar units with stationary bases. In subsequent discussion, we outline a qualitative methodology to assess the effect of probabilistic multi-return cases on lidar-based localization.
>
---
#### [new 015] Robust Adaptive Backstepping Impedance Control of Robots in Unknown Environments
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决未知环境中机器人轨迹跟踪与力控制问题。提出RABIC方法，结合自适应阻抗控制与反步法，提升系统稳定性与安全性。**

- **链接: [https://arxiv.org/pdf/2604.09323](https://arxiv.org/pdf/2604.09323)**

> **作者:** Reza Nazmara; Alap Kshirsagar; Jan Peters; A. Pedro Aguiar
>
> **备注:** 8
>
> **摘要:** This paper presents a Robust Adaptive Backstepping Impedance Control (RABIC) strategy for robots operating in contact-rich and uncertain environments. The proposed control strategy considers the complete coupled dynamics of the system and explicitly accounts for key sources of uncertainty, including external disturbances and unmodeled dynamics, while not requiring the robot's dynamic parameters in implementation. We propose a backstepping-based adaptive impedance control scheme for the inner loop to track the reference impedance model. To handle uncertainties, we employ a Taylor series-based estimator for system dynamics and an adaptive estimator for determining the upper bound of external forces. Stability analysis demonstrates the semi-global practical finite-time stability of the overall system. To demonstrate the effectiveness of the proposed method, a simulated mobile manipulator scenario and experimental evaluations on a real Franka Emika Panda robot were conducted. The proposed approach exhibits safer performance compared to PD control while ensuring trajectory tracking and force monitoring. Overall, the RABIC framework provides a solid basis for future research on adaptive and learning-based impedance control for coupled mobile and fixed serially linked manipulators.
>
---
#### [new 016] Sim-to-Real Transfer for Muscle-Actuated Robots via Generalized Actuator Networks
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，解决肌肉驱动机器人从仿真到现实的迁移问题。通过提出GeAN模型，实现无需力矩传感器的精确控制，成功部署了模拟训练的策略。**

- **链接: [https://arxiv.org/pdf/2604.09487](https://arxiv.org/pdf/2604.09487)**

> **作者:** Jan Schneider; Mridul Mahajan; Le Chen; Simon Guist; Bernhard Schölkopf; Ingmar Posner; Dieter Büchler
>
> **摘要:** Tendon drives paired with soft muscle actuation enable faster and safer robots while potentially accelerating skill acquisition. Still, these systems are rarely used in practice due to inherent nonlinearities, friction, and hysteresis, which complicate modeling and control. So far, these challenges have hindered policy transfer from simulation to real systems. To bridge this gap, we propose a sim-to-real pipeline that learns a neural network model of this complex actuation and leverages established rigid body simulation for the arm dynamics and interactions with the environment. Our method, called Generalized Actuator Network (GeAN), enables actuation model identification across a wide range of robots by learning directly from joint position trajectories rather than requiring torque sensors. Using GeAN on PAMY2, a tendon-driven robot powered by pneumatic artificial muscles, we successfully deploy precise goal-reaching and dynamic ball-in-a-cup policies trained entirely in simulation. To the best of our knowledge, this result constitutes the first successful sim-to-real transfer for a four-degrees-of-freedom muscle-actuated robot arm.
>
---
#### [new 017] A Benchmark of Dexterity for Anthropomorphic Robotic Hands
- **分类: cs.RO**

- **简介: 该论文提出POMDAR基准，用于评估仿人机械手的灵巧性。解决灵巧性定义不统一的问题，通过任务性能进行客观评价。**

- **链接: [https://arxiv.org/pdf/2604.09294](https://arxiv.org/pdf/2604.09294)**

> **作者:** Davide Liconti; Yuning Zhou; Yasunori Toshimitsu; Ronan Hinchet; Robert K. Katzschmann
>
> **摘要:** Dexterity is a central yet ambiguously defined concept in the design and evaluation of anthropomorphic robotic hands. In practice, the term is often used inconsistently, with different systems evaluated under disparate criteria, making meaningful comparisons across designs difficult. This highlights the need for a unified, performance-based definition of dexterity grounded in measurable outcomes rather than proxy metrics. In this work, we introduce POMDAR, a comprehensive dexterity benchmark that formalizes dexterity as task performance across a structured set of manipulation and grasping motions. The benchmark was systematically derived from established taxonomies in human motor control. It is implemented in both real-world and simulation and includes four manipulation configurations: vertical and horizontal configurations, continuous rotation, and pure grasping. The task designs contain mechanical scaffolding to constrain task motion, suppress compensatory strategies, and enable metrics to be measured unambiguously. We define a quantitative scoring metric combining task correctness and execution speed, effectively measuring dexterity as throughput. This enables objective, reproducible, and interpretable evaluation across different hand designs. POMDAR provides an open-source, standardized, and taxonomy-grounded benchmark for consistent comparison and evaluation of anthropomorphic robot hands to facilitate a systematic advancement of dexterous manipulation platforms. CAD, simulation files, and evaluation videos are publicly available at this https URL.
>
---
#### [new 018] Online Intention Prediction via Control-Informed Learning
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于在线意图预测任务，旨在实时估计自主系统的目标状态，解决动态环境下意图变化及参数未知的问题。通过逆最优控制方法和在线学习策略实现高效、自适应的意图预测。**

- **链接: [https://arxiv.org/pdf/2604.09303](https://arxiv.org/pdf/2604.09303)**

> **作者:** Tianyu Zhou; Zihao Liang; Zehui Lu; Shaoshuai Mou
>
> **摘要:** This paper presents an online intention prediction framework for estimating the goal state of autonomous systems in real time, even when intention is time-varying, and system dynamics or objectives include unknown parameters. The problem is formulated as an inverse optimal control / inverse reinforcement learning task, with the intention treated as a parameter in the objective. A shifting horizon strategy discounts outdated information, while online control-informed learning enables efficient gradient computation and online parameter updates. Simulations under varying noise levels and hardware experiments on a quadrotor drone demonstrate that the proposed approach achieves accurate, adaptive intention prediction in complex environments.
>
---
#### [new 019] Toward Hardware-Agnostic Quadrupedal World Models via Morphology Conditioning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人学任务，旨在解决硬件特定的世界模型泛化问题。通过显式条件化机器人形态，构建可跨形态通用的四足世界模型，实现零样本控制。**

- **链接: [https://arxiv.org/pdf/2604.08780](https://arxiv.org/pdf/2604.08780)**

> **作者:** Mohamad H. Danesh; Chenhao Li; Amin Abyaneh; Anas Houssaini; Kirsty Ellis; Glen Berseth; Marco Hutter; Hsiu-Chin Lin
>
> **摘要:** World models promise a paradigm shift in robotics, where an agent learns the underlying physics of its environment once to enable efficient planning and behavior learning. However, current world models are often hardware-locked specialists: a model trained on a Boston Dynamics Spot robot fails catastrophically on a Unitree Go1 due to the mismatch in kinematic and dynamic properties, as the model overfits to specific embodiment constraints rather than capturing the universal locomotion dynamics. Consequently, a slight change in actuator dynamics or limb length necessitates training a new model from scratch. In this work, we take a step towards a framework for training a generalizable Quadrupedal World Model (QWM) that disentangles environmental dynamics from robot morphology. We address the limitations of implicit system identification, where treating static physical properties (like mass or limb length) as latent variables to be inferred from motion history creates an adaptation lag that can compromise zero-shot safety and efficiency. Instead, we explicitly condition the generative dynamics on the robot's engineering specifications. By integrating a physical morphology encoder and a reward normalizer, we enable the model to serve as a neural simulator capable of generalizing across morphologies. This capability unlocks zero-shot control across a range of embodiments. We introduce, for the first time, a world model that enables zero-shot generalization to new morphologies for locomotion. While we carefully study the limitations of our method, QWM operates as a distribution-bounded interpolator within the quadrupedal morphology family rather than a universal physics engine, this work represents a significant step toward morphology-conditioned world models for legged locomotion.
>
---
#### [new 020] SafeMind: A Risk-Aware Differentiable Control Framework for Adaptive and Safe Quadruped Locomotion
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决四足机器人在不确定环境中的安全问题。提出SafeMind框架，结合概率控制屏障函数与语义理解，提升安全性与适应性。**

- **链接: [https://arxiv.org/pdf/2604.09474](https://arxiv.org/pdf/2604.09474)**

> **作者:** Zukun Zhang; Kai Shu; Mingqiao Mo
>
> **摘要:** Learning-based quadruped controllers achieve impressive agility but typically lack formal safety guarantees under model uncertainty, perception noise, and unstructured contact conditions. We introduce SafeMind, a differentiable stochastic safety-control framework that unifies probabilistic Control Barrier Functions with semantic context understanding and meta-adaptive risk calibration. SafeMind explicitly models epistemic and aleatoric uncertainty through a variance-aware barrier constraint embedded in a differentiable quadratic program, thereby preserving gradient flow for end-to-end training. A semantics-to-constraint encoder modulates safety margins using perceptual or language cues, while a meta-adaptive learner continuously adjusts risk sensitivity across environments. We provide theoretical conditions for probabilistic forward invariance, feasibility, and stability under stochastic dynamics. SafeMind is deployed on Unitree A1 and ANYmal C at 200~Hz and validated across 12 terrain types, dynamic obstacles, morphology perturbations, and semantically defined tasks. Experiments show that SafeMind reduces safety violations by 3--10x and energy consumption by 10--15% relative to state-of-the-art CBF, MPC, and hybrid RL baselines, while maintaining real-time control performance.
>
---
#### [new 021] Multimodal Anomaly Detection for Human-Robot Interaction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人机交互中的异常检测任务，旨在提升机器人对异常事件的识别能力。通过多模态特征重构，融合视觉与传感器数据，提高检测效果。**

- **链接: [https://arxiv.org/pdf/2604.09326](https://arxiv.org/pdf/2604.09326)**

> **作者:** Guilherme Ribeiro; Iordanis Antypas; Leonardo Bizzaro; João Bimbo; Nuno Cruz Garcia
>
> **摘要:** Ensuring safety and reliability in human-robot interaction (HRI) requires the timely detection of unexpected events that could lead to system failures or unsafe behaviours. Anomaly detection thus plays a critical role in enabling robots to recognize and respond to deviations from normal operation during collaborative tasks. While reconstruction models have been actively explored in HRI, approaches that operate directly on feature vectors remain largely unexplored. In this work, we propose MADRI, a framework that first transforms video streams into semantically meaningful feature vectors before performing reconstruction-based anomaly detection. Additionally, we augment these visual feature vectors with the robot's internal sensors' readings and a Scene Graph, enabling the model to capture both external anomalies in the visual environment and internal failures within the robot itself. To evaluate our approach, we collected a custom dataset consisting of a simple pick-and-place robotic task under normal and anomalous conditions. Experimental results demonstrate that reconstruction on vision-based feature vectors alone is effective for detecting anomalies, while incorporating other modalities further improves detection performance, highlighting the benefits of multimodal feature reconstruction for robust anomaly detection in human-robot collaboration.
>
---
#### [new 022] {\sf TriDeliver}: Cooperative Air-Ground Instant Delivery with UAVs, Couriers, and Crowdsourced Ground Vehicles
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于即时配送任务，解决单一配送方式效率低的问题。通过整合无人机、快递员和众包车辆，提出TriDeliver框架，提升配送效率并降低成本。**

- **链接: [https://arxiv.org/pdf/2604.09049](https://arxiv.org/pdf/2604.09049)**

> **作者:** Junhui Gao; Yan Pan; Qianru Wang; Wenzhe Hou; Yiqin Deng; Liangliang Jiang; Yuguang Fang
>
> **摘要:** Instant delivery, shipping items before critical deadlines, is essential in daily life. While multiple delivery agents, such as couriers, Unmanned Aerial Vehicles (UAVs), and crowdsourced agents, have been widely employed, each of them faces inherent limitations (e.g., low efficiency/labor shortages, flight control, and dynamic capabilities, respectively), preventing them from meeting the surging demands alone. This paper proposes {\sf TriDeliver}, the first hierarchical cooperative framework, integrating human couriers, UAVs, and crowdsourced ground vehicles (GVs) for efficient instant delivery. To obtain the initial scheduling knowledge for GVs and UAVs as well as improve the cooperative delivery performance, we design a Transfer Learning (TL)-based algorithm to extract delivery knowledge from couriers' behavioral history and transfer their knowledge to UAVs and GVs with fine-tunings, which is then used to dispatch parcels for efficient delivery. Evaluated on one-month real-world trajectory and delivery datasets, it has been demonstrated that 1) by integrating couriers, UAVs, and crowdsourced GVs, {\sf TriDeliver} reduces the delivery cost by $65.8\%$ versus state-of-the-art cooperative delivery by UAVs and couriers; 2) {\sf TriDeliver} achieves further improvements in terms of delivery time ($-17.7\%$), delivery cost ($-9.8\%$), and impacts on original tasks of crowdsourced GVs ($-43.6\%$), even with the representation of the transferred knowledge by simple neural networks, respectively.
>
---
#### [new 023] HTNav: A Hybrid Navigation Framework with Tiered Structure for Urban Aerial Vision-and-Language Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于城市空中视觉语言导航任务，解决复杂环境中导航泛化差、路径规划不佳和空间连续性理解不足的问题。提出HTNav框架，结合模仿学习与强化学习，提升导航精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.08883](https://arxiv.org/pdf/2604.08883)**

> **作者:** Chengjie Fan; Cong Pan; Zijian Liu; Ningzhong Liu; Jie Qin
>
> **摘要:** Inspired by the general Vision-and-Language Navigation (VLN) task, aerial VLN has attracted widespread attention, owing to its significant practical value in applications such as logistics delivery and urban inspection. However, existing methods face several challenges in complex urban environments, including insufficient generalization to unseen scenes, suboptimal performance in long-range path planning, and inadequate understanding of spatial continuity. To address these challenges, we propose HTNav, a new collaborative navigation framework that integrates Imitation Learning (IL) and Reinforcement Learning (RL) within a hybrid IL-RL framework. This framework adopts a staged training mechanism to ensure the stability of the basic navigation strategy while enhancing its environmental exploration capability. By integrating a tiered decision-making mechanism, it achieves collaborative interaction between macro-level path planning and fine-grained action control. Furthermore, a map representation learning module is introduced to deepen its understanding of spatial continuity in open domains. On the CityNav benchmark, our method achieves state-of-the-art performance across all scene levels and task difficulties. Experimental results demonstrate that this framework significantly improves navigation precision and robustness in complex urban environments.
>
---
#### [new 024] AssemLM: Spatial Reasoning Multimodal Large Language Models for Robotic Assembly
- **分类: cs.RO**

- **简介: 该论文属于机器人装配任务，解决3D几何推理不足的问题。提出AssemLM模型，结合多模态数据进行6D姿态预测，并构建了AssemBench数据集。**

- **链接: [https://arxiv.org/pdf/2604.08983](https://arxiv.org/pdf/2604.08983)**

> **作者:** Zhi Jing; Jinbin Qiao; Ouyang Lu; Jicong Ao; Shuang Qiu; Yu-Gang Jiang; Chenjia Bai
>
> **备注:** Project Page: this https URL
>
> **摘要:** Spatial reasoning is a fundamental capability for embodied intelligence, especially for fine-grained manipulation tasks such as robotic assembly. While recent vision-language models (VLMs) exhibit preliminary spatial awareness, they largely rely on coarse 2D perception and lack the ability to perform accurate reasoning over 3D geometry, which is crucial for precise assembly operations. To address this limitation, we propose AssemLM, a spatial multimodal large language model tailored for robotic assembly. AssemLM integrates assembly manuals, point clouds, and textual instructions to reason about and predict task-critical 6D assembly poses, enabling explicit geometric understanding throughout the assembly process. To effectively bridge raw 3D perception and high-level reasoning, we adopt a specialized point cloud encoder to capture fine-grained geometric and rotational features, which are then integrated into the multimodal language model to support accurate 3D spatial reasoning for assembly tasks. In addition, we construct AssemBench, a large-scale dataset and benchmark for assembly-oriented spatial reasoning, comprising over 900K multimodal samples with precise 6D pose annotations. AssemBench extends spatial reasoning evaluation beyond 2D and grounding tasks into full 3D geometric inference, filling a critical gap in existing embodied AI benchmarks. Extensive experiments demonstrate that AssemLM achieves state-of-the-art performance in 6D pose reasoning across diverse assembly scenarios. Furthermore, real-robot evaluations show that our model can support fine-grained and multi-step assembly execution in real-world settings, demonstrating its potential for robotic assembly applications.
>
---
#### [new 025] WOMBET: World Model-based Experience Transfer for Robust and Sample-efficient Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出WOMBET框架，解决机器人强化学习中的经验迁移问题，通过联合生成与利用先验数据提升样本效率和性能。**

- **链接: [https://arxiv.org/pdf/2604.08958](https://arxiv.org/pdf/2604.08958)**

> **作者:** Mintae Kim; Koushil Sreenath
>
> **备注:** 13 pages, 6 figures, 8th Annual Learning for Dynamics & Control Conference (L4DC)
>
> **摘要:** Reinforcement learning (RL) in robotics is often limited by the cost and risk of data collection, motivating experience transfer from a source task to a target task. Offline-to-online RL leverages prior data but typically assumes a given fixed dataset and does not address how to generate reliable data for transfer. We propose \textit{World Model-based Experience Transfer} (WOMBET), a framework that jointly generates and utilizes prior data. WOMBET learns a world model in the source task and generates offline data via uncertainty-penalized planning, followed by filtering trajectories with high return and low epistemic uncertainty. It then performs online fine-tuning in the target task using adaptive sampling between offline and online data, enabling a stable transition from prior-driven initialization to task-specific adaptation. We show that the uncertainty-penalized objective provides a lower bound on the true return and derive a finite-sample error decomposition capturing distribution mismatch and approximation error. Empirically, WOMBET improves sample efficiency and final performance over strong baselines on continuous control benchmarks, demonstrating the benefit of jointly optimizing data generation and transfer.
>
---
#### [new 026] TouchAnything: Diffusion-Guided 3D Reconstruction from Sparse Robot Touches
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D重建任务，解决从稀疏触觉数据中准确重建物体几何的问题。通过迁移视觉扩散模型的先验知识，提升触觉重建精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.08945](https://arxiv.org/pdf/2604.08945)**

> **作者:** Langzhe Gu; Hung-Jui Huang; Mohamad Qadri; Michael Kaess; Wenzhen Yuan
>
> **备注:** Project Page: this https URL
>
> **摘要:** Accurate object geometry estimation is essential for many downstream tasks, including robotic manipulation and physical interaction. Although vision is the dominant modality for shape perception, it becomes unreliable under occlusions or challenging lighting conditions. In such scenarios, tactile sensing provides direct geometric information through physical contact. However, reconstructing global 3D geometry from sparse local touches alone is fundamentally underconstrained. We present TouchAnything, a framework that leverages a pretrained large-scale 2D vision diffusion model as a semantic and geometric prior for 3D reconstruction from sparse tactile measurements. Unlike prior work that trains category-specific reconstruction networks or learns diffusion models directly from tactile data, we transfer the geometric knowledge encoded in pretrained visual diffusion models to the tactile domain. Given sparse contact constraints and a coarse class-level description of the object, we formulate reconstruction as an optimization problem that enforces tactile consistency while guiding solutions toward shapes consistent with the diffusion prior. Our method reconstructs accurate geometries from only a few touches, outperforms existing baselines, and enables open-world 3D reconstruction of previously unseen object instances. Our project page is this https URL .
>
---
#### [new 027] PhysInOne: Visual Physics Learning and Reasoning in One Suite
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出PhysInOne，一个大规模合成数据集，解决AI缺乏物理基础训练数据的问题。用于物理学习与推理任务，涵盖多种物理现象，提升模型的物理合理性。**

- **链接: [https://arxiv.org/pdf/2604.09415](https://arxiv.org/pdf/2604.09415)**

> **作者:** Siyuan Zhou; Hejun Wang; Hu Cheng; Jinxi Li; Dongsheng Wang; Junwei Jiang; Yixiao Jin; Jiayue Huang; Shiwei Mao; Shangjia Liu; Yafei Yang; Hongkang Song; Shenxing Wei; Zihui Zhang; Peng Huang; Shijie Liu; Zhengli Hao; Hao Li; Yitian Li; Wenqi Zhou; Zhihan Zhao; Zongqi He; Hongtao Wen; Shouwang Huang; Peng Yun; Bowen Cheng; Pok Kazaf Fu; Wai Kit Lai; Jiahao Chen; Kaiyuan Wang; Zhixuan Sun; Ziqi Li; Haochen Hu; Di Zhang; Chun Ho Yuen; Bing Wang; Zhihua Wang; Chuhang Zou; Bo Yang
>
> **备注:** CVPR 2026. Siyuan, Hejun, Hu, Jinxi, Dongsheng, Junwei, Yixiao, Jiayue, and Shiwei are co-first authors. Project page: this https URL
>
> **摘要:** We present PhysInOne, a large-scale synthetic dataset addressing the critical scarcity of physically-grounded training data for AI systems. Unlike existing datasets limited to merely hundreds or thousands of examples, PhysInOne provides 2 million videos across 153,810 dynamic 3D scenes, covering 71 basic physical phenomena in mechanics, optics, fluid dynamics, and magnetism. Distinct from previous works, our scenes feature multiobject interactions against complex backgrounds, with comprehensive ground-truth annotations including 3D geometry, semantics, dynamic motion, physical properties, and text descriptions. We demonstrate PhysInOne's efficacy across four emerging applications: physics-aware video generation, long-/short-term future frame prediction, physical property estimation, and motion transfer. Experiments show that fine-tuning foundation models on PhysInOne significantly enhances physical plausibility, while also exposing critical gaps in modeling complex physical dynamics and estimating intrinsic properties. As the largest dataset of its kind, orders of magnitude beyond prior works, PhysInOne establishes a new benchmark for advancing physics-grounded world models in generation, simulation, and embodied AI.
>
---
#### [new 028] 3D-VCD: Hallucination Mitigation in 3D-LLM Embodied Agents through Visual Contrastive Decoding
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于多模态推理任务，旨在解决3D大模型在具身代理中的幻觉问题。通过构建扭曲的3D场景图进行对比解码，提升 grounded 推理可靠性。**

- **链接: [https://arxiv.org/pdf/2604.08645](https://arxiv.org/pdf/2604.08645)**

> **作者:** Makanjuola Ogunleye; Eman Abdelrahman; Ismini Lourentzou
>
> **备注:** 8 pages, 6 figures, Accepted at IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Large multimodal models are increasingly used as the reasoning core of embodied agents operating in 3D environments, yet they remain prone to hallucinations that can produce unsafe and ungrounded decisions. Existing inference-time hallucination mitigation methods largely target 2D vision-language settings and do not transfer to embodied 3D reasoning, where failures arise from object presence, spatial layout, and geometric grounding rather than pixel-level inconsistencies. We introduce 3D-VCD, the first inference-time visual contrastive decoding framework for hallucination mitigation in 3D embodied agents. 3D-VCD constructs a distorted 3D scene graph by applying semantic and geometric perturbations to object-centric representations, such as category substitutions and coordinate or extent corruption. By contrasting predictions under the original and distorted 3D contexts, our method suppresses tokens that are insensitive to grounded scene evidence and are therefore likely driven by language priors. We evaluate 3D-VCD on the 3D-POPE and HEAL benchmarks and show that it consistently improves grounded reasoning without any retraining, establishing inference-time contrastive decoding over structured 3D representations as an effective and practical route to more reliable embodied intelligence.
>
---
#### [new 029] "Take Me Home, Wi-Fi Drone": A Drone-based Wireless System for Wilderness Search and Rescue
- **分类: cs.NI; cs.RO**

- **简介: 论文提出Wi2SAR系统，用于野外搜救任务。通过无人机模拟Wi-Fi网络，定位携带移动设备的遇险者，解决传统搜救中依赖基础设施的问题。**

- **链接: [https://arxiv.org/pdf/2604.09115](https://arxiv.org/pdf/2604.09115)**

> **作者:** Weiying Hou; Luca Jiang-Tao Yu; Chenshu Wu
>
> **备注:** 16 pages, 12 figures, 1 table. Project page: this https URL
>
> **摘要:** Wilderness Search and Rescue (WiSAR) represents a longstanding and critical societal challenge, demanding innovative and automatic technological solutions. In this paper, we introduce Wi2SAR, a novel autonomous drone-based wireless system for long-range, through-occlusion WiSAR operations, without relying on existing infrastructure. Our basic insight is to leverage the automatic reconnection behavior of modern Wi-Fi devices to known networks. By mimicking these networks via on-drone Wi-Fi, Wi2SAR uniquely facilitates the discovery and localization of victims through their accompanying mobile devices. Translating this simple idea into a practical system poses substantial technical challenges. Wi2SAR overcomes these challenges via three distinct innovations: (1) a rapid and energy-efficient device discovery mechanism to discover and identify the target victim, (2) a novel RSS-only, long-range direction finding approach using a 3D-printed Luneburg Lens, amplifying the directional signal strength differences and significantly extending the operational range, and (3) an adaptive drone navigation scheme that guides the drone toward the target efficiently. We implement an end-to-end prototype and evaluate Wi2SAR across various mobile devices and real-world wilderness scenarios. Experimental results demonstrate Wi2SAR's high performance, efficiency, and practicality, highlighting its potential to advance autonomous WiSAR solutions. Wi2SAR is open-sourced at this https URL to facilitate further research and real-world deployment.
>
---
#### [new 030] Fine-Grained Action Segmentation for Renorrhaphy in Robot-Assisted Partial Nephrectomy
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于动作分割任务，解决机器人辅助部分肾切除术中精细动作识别问题。通过对比不同模型在基准数据集上的性能，提升动作识别的准确性。**

- **链接: [https://arxiv.org/pdf/2604.09051](https://arxiv.org/pdf/2604.09051)**

> **作者:** Jiaheng Dai; Huanrong Liu; Tailai Zhou; Tongyu Jia; Qin Liu; Yutong Ban; Zeju Li; Yu Gao; Xin Ma; Qingbiao Li
>
> **摘要:** Fine-grained action segmentation during renorrhaphy in robot-assisted partial nephrectomy requires frame-level recognition of visually similar suturing gestures with variable duration and substantial class imbalance. The SIA-RAPN benchmark defines this problem on 50 clinical videos acquired with the da Vinci Xi system and annotated with 12 frame-level labels. The benchmark compares four temporal models built on I3D features: MS-TCN++, AsFormer, TUT, and DiffAct. Evaluation uses balanced accuracy, edit score, segmental F1 at overlap thresholds of 10, 25, and 50, frame-wise accuracy, and frame-wise mean average precision. In addition to the primary evaluation across five released split configurations on SIA-RAPN, the benchmark reports cross-domain results on a separate single-port RAPN dataset. Across the strongest reported values over those five runs on the primary dataset, DiffAct achieves the highest F1, frame-wise accuracy, edit score, and frame mAP, while MS-TCN++ attains the highest balanced accuracy.
>
---
#### [new 031] Decentralized Opinion-Integrated Decision making at Unsignalized Intersections via Signed Networks
- **分类: eess.SY; cs.MA; cs.RO**

- **简介: 该论文属于自动驾驶车辆协同决策任务，解决无信号交叉口的去中心化协调问题。通过双符号网络实现车辆间意图交换与决策同步，确保安全高效通行。**

- **链接: [https://arxiv.org/pdf/2604.09351](https://arxiv.org/pdf/2604.09351)**

> **作者:** Bhaskar Varma; Ying Shuai Quan; Karl D. von Ellenrieder; Paolo Falcone
>
> **备注:** Submitted to CDC 2026 with L-CSS Parallel option
>
> **摘要:** In this letter, we consider the problem of decentralized decision making among connected autonomous vehicles at unsignalized intersections, where existing centralized approaches do not scale gracefully under mixed maneuver intentions and coordinator failure. We propose a closed-loop opinion-dynamic decision model for intersection coordination, where vehicles exchange intent through dual signed networks: a conflict topology based communication network and a commitment-driven belief network that enable cooperation without a centralized coordinator. Continuous opinion states modulate velocity optimizer weights prior to commitment; a closed-form predictive feasibility gate then freezes each vehicle's decision into a GO or YIELD commitment, which propagates back through the belief network to pre-condition neighbor behavior ahead of physical conflicts. Crossing order emerges from geometric feasibility and arrival priority without the use of joint optimization or a solver. The approach is validated across three scenarios spanning fully competitive, merge, and mixed conflict topologies. The results demonstrate collision-free coordination and lower last-vehicle exit times compared to first come first served (FCFS) in all conflict non-trivial configurations.
>
---
#### [new 032] Accelerating Transformer-Based Monocular SLAM via Geometric Utility Scoring
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于单目SLAM任务，旨在解决GFM部署中的计算冗余问题。提出LeanGate网络，在特征提取前预测几何价值，减少冗余帧处理，提升效率并保持精度。**

- **链接: [https://arxiv.org/pdf/2604.08718](https://arxiv.org/pdf/2604.08718)**

> **作者:** Xinmiao Xiong; Bangya Liu; Hao Wang; Dayou Li; Nuo Chen; Andrew Feng; Mingyu Ding; Suman Banerjee; Yang Zhou; Zhiwen Fan
>
> **摘要:** Geometric Foundation Models (GFMs) have recently advanced monocular SLAM by providing robust, calibration-free 3D priors. However, deploying these models on dense video streams introduces significant computational redundancy. Current GFM-based SLAM systems typically rely on post hoc keyframe selection. Because of this, they must perform expensive dense geometric decoding simply to determine whether a frame contains novel geometry, resulting in late rejection and wasted computation. To mitigate this inefficiency, we propose LeanGate, a lightweight feed-forward frame-gating network. LeanGate predicts a geometric utility score to assess a frame's mapping value prior to the heavy GFM feature extraction and matching stages. As a predictive plug-and-play module, our approach bypasses over 90% of redundant frames. Evaluations on standard SLAM benchmarks demonstrate that LeanGate reduces tracking FLOPs by more than 85% and achieves a 5x end-to-end throughput speedup. Furthermore, it maintains the tracking and mapping accuracy of dense baselines.
>
---
#### [new 033] Incremental Semantics-Aided Meshing from LiDAR-Inertial Odometry and RGB Direct Label Transfer
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于三维重建任务，解决室内大场景中点云稀疏导致的几何失真问题。通过融合RGB语义与LiDAR数据，提升网格重建质量。**

- **链接: [https://arxiv.org/pdf/2604.09478](https://arxiv.org/pdf/2604.09478)**

> **作者:** Muhammad Affan; Ville Lehtola; George Vosselman
>
> **备注:** 8 pages, 5 figures, 2 tables. Accepted in ISPRS Archives 2026
>
> **摘要:** Geometric high-fidelity mesh reconstruction from LiDAR-inertial scans remains challenging in large, complex indoor environments -- such as cultural buildings -- where point cloud sparsity, geometric drift, and fixed fusion parameters produce holes, over-smoothing, and spurious surfaces at structural boundaries. We propose a modular, incremental RGB+LiDAR pipeline that generates incremental semantics-aided high-quality meshes from indoor scans through scan frame-based direct label transfer. A vision foundation model labels each incoming RGB frame; labels are incrementally projected and fused onto a LiDAR-inertial odometry map; and an incremental semantics-aware Truncated Signed Distance Function (TSDF) fusion step produces the final mesh via marching cubes. This frame-level fusion strategy preserves the geometric fidelity of LiDAR while leveraging rich visual semantics to resolve geometric ambiguities at reconstruction boundaries caused by LiDAR point-cloud sparsity and geometric drift. We demonstrate that semantic guidance improves geometric reconstruction quality; quantitative evaluation is therefore performed using geometric metrics on the Oxford Spires dataset, while results from the NTU VIRAL dataset are analyzed qualitatively. The proposed method outperforms state-of-the-art geometric baselines ImMesh and Voxblox, demonstrating the benefit of semantics-aided fusion for geometric mesh quality. The resulting semantically labelled meshes are of value when reconstructing Universal Scene Description (USD) assets, offering a path from indoor LiDAR scanning to XR and digital modeling.
>
---
#### [new 034] Physically Grounded 3D Generative Reconstruction under Hand Occlusion using Proprioception and Multi-Contact Touch
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D物体重建任务，解决手部遮挡下的物体结构与位姿估计问题。通过融合本体感觉和多点触觉信息，提升遮挡区域的重建精度与物理合理性。**

- **链接: [https://arxiv.org/pdf/2604.09100](https://arxiv.org/pdf/2604.09100)**

> **作者:** Gabriele Mario Caddeo; Pasquale Marra; Lorenzo Natale
>
> **备注:** 27 pages, 10 figures, under review
>
> **摘要:** We propose a multimodal, physically grounded approach for metric-scale amodal object reconstruction and pose estimation under severe hand occlusion. Unlike prior occlusion-aware 3D generation methods that rely only on vision, we leverage physical interaction signals: proprioception provides the posed hand geometry, and multi-contact touch constrains where the object surface must lie, reducing ambiguity in occluded regions. We represent object structure as a pose-aware, camera-aligned signed distance field (SDF) and learn a compact latent space with a Structure-VAE. In this latent space, we train a conditional flow-matching diffusion model, pretraining on vision-only images and finetuning on occluded manipulation scenes while conditioning on visible RGB evidence, occluder/visibility masks, the hand latent representation, and tactile information. Crucially, we incorporate physics-based objectives and differentiable decoder-guidance during finetuning and inference to reduce hand--object interpenetration and to align the reconstructed surface with contact observations. Because our method produces a metric, physically consistent structure estimate, it integrates naturally into existing two-stage reconstruction pipelines, where a downstream module refines geometry and predicts appearance. Experiments in simulation show that adding proprioception and touch substantially improves completion under occlusion and yields physically plausible reconstructions at correct real-world scale compared to vision-only baselines; we further validate transfer by deploying the model on a real humanoid robot with an end-effector different from those used during training.
>
---
#### [new 035] LMGenDrive: Bridging Multimodal Understanding and Generative World Modeling for End-to-End Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决长尾和开放场景下的泛化问题。通过融合多模态理解和生成模型，提出LMGenDrive框架，实现端到端驾驶决策。**

- **链接: [https://arxiv.org/pdf/2604.08719](https://arxiv.org/pdf/2604.08719)**

> **作者:** Hao Shao; Letian Wang; Yang Zhou; Yuxuan Hu; Zhuofan Zong; Steven L. Waslander; Wei Zhan; Hongsheng Li
>
> **摘要:** Recent years have seen remarkable progress in autonomous driving, yet generalization to long-tail and open-world scenarios remains a major bottleneck for large-scale deployment. To address this challenge, some works use LLMs and VLMs for vision-language understanding and reasoning, enabling vehicles to interpret rare and safety-critical situations when generating actions. Others study generative world models to capture the spatio-temporal evolution of driving scenes, allowing agents to imagine possible futures before acting. Inspired by human intelligence, which unifies understanding and imagination, we explore a unified model for autonomous driving. We present LMGenDrive, the first framework that combines LLM-based multimodal understanding with generative world models for end-to-end closed-loop driving. Given multi-view camera inputs and natural-language instructions, LMGenDrive generates both future driving videos and control signals. This design provides complementary benefits: video prediction improves spatio-temporal scene modeling, while the LLM contributes strong semantic priors and instruction grounding from large-scale pretraining. We further propose a progressive three-stage training strategy, from vision pretraining to multi-step long-horizon driving, to improve stability and performance. LMGenDrive supports both low-latency online planning and autoregressive offline video generation. Experiments show that it significantly outperforms prior methods on challenging closed-loop benchmarks, with clear gains in instruction following, spatio-temporal understanding, and robustness to rare scenarios. These results suggest that unifying multimodal understanding and generation is a promising direction for more generalizable and robust embodied decision-making systems.
>
---
#### [new 036] MASS: Mesh-inellipse Aligned Deformable Surfel Splatting for Hand Reconstruction and Rendering from Egocentric Monocular Video
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于手部重建与渲染任务，解决单目视频中高精度手部建模和实时渲染问题。提出MASS方法，通过可变形高斯表面点实现高效、逼真的手部重建。**

- **链接: [https://arxiv.org/pdf/2604.08943](https://arxiv.org/pdf/2604.08943)**

> **作者:** Haoyu Zhu; Yi Zhang; Lei Yao; Lap-pui Chau; Yi Wang
>
> **备注:** This paper has been accepted to CVM 2026 Journal Track and is under consideration for publication in IEEE TVCG
>
> **摘要:** Reconstructing high-fidelity 3D hands from egocentric monocular videos remains a challenge due to the limitations in capturing high-resolution geometry, hand-object interactions, and complex objects on hands. Additionally, existing methods often incur high computational costs, making them impractical for real-time applications. In this work, we propose Mesh-inellipse Aligned deformable Surfel Splatting (MASS) to address these challenges by leveraging a deformable 2D Gaussian Surfel representation. We introduce the mesh-aligned Steiner Inellipse and fractal densification for mesh-to-surfel conversion that initiates high-resolution 2D Gaussian surfels from coarse parametric hand meshes, providing surface representation with photorealistic rendering potential. Second, we propose Gaussian Surfel Deformation, which enables efficient modeling of hand deformations and personalized features by predicting residual updates to surfel attributes and introducing an opacity mask to refine geometry and texture without adaptive density control. In addition, we propose a two-stage training strategy and a novel binding loss to improve the optimization robustness and reconstruction quality. Extensive experiments on the ARCTIC dataset, the Hand Appearance dataset, and the Interhand2.6M dataset demonstrate that our model achieves superior reconstruction performance compared to state-of-the-art methods.
>
---
#### [new 037] 2D or 3D: Who Governs Salience in VLA Models? -- Tri-Stage Token Pruning Framework with Modality Salience Awareness
- **分类: cs.MM; cs.CV; cs.RO**

- **简介: 该论文针对多模态视觉-语言-动作模型中的token剪枝问题，提出一种考虑2D/3D模态显著性的三阶段剪枝框架，以提升推理速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2604.09244](https://arxiv.org/pdf/2604.09244)**

> **作者:** Zihao Zheng; Sicheng Tian; Zhihao Mao; Lingyue Zhang; Chenyue Li; Ziyun Zhang; Hong Gao; Yuchen Huang; Yutong Xu; Guojie Luo; Xiang Chen
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as the mainstream of embodied intelligence. Recent VLA models have expanded their input modalities from 2D-only to 2D+3D paradigms, forming multi-visual-modal VLA (MVLA) models. Despite achieving improved spatial perception, MVLA faces a greater acceleration demand due to the increased number of input tokens caused by modal expansion. Token pruning is an effective optimization methods tailored to MVLA models. However, existing token pruning schemes are designed for 2D-only VLA models, ignoring 2D/3D modality salience differences. In this paper, we follow the application process of multi-modal data in MVLA models and develop a tri-stage analysis to capture the discrepancy and dynamics of 2D/3D modality salience. Based on these, we propose a corresponding tri-stage token pruning framework for MVLA models to achieve optimal 2D/3D token selection and efficient pruning. Experiments show that our framework achieves up to a 2.55x inference speedup with minimal accuracy loss, while only costing 5.8% overhead. Our Code is coming soon.
>
---
## 更新

#### [replaced 001] RESample: A Robust Data Augmentation Framework via Exploratory Sampling for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人操作任务，旨在解决模仿学习中数据集分布有限导致的泛化能力不足问题。提出RESample框架，通过探索性采样增强数据分布覆盖。**

- **链接: [https://arxiv.org/pdf/2510.17640](https://arxiv.org/pdf/2510.17640)**

> **作者:** Yuquan Xue; Guanxing Lu; Zhenyu Wu; Chuanrui Zhang; Bofang Jia; Zhengyi Gu; Ziwei Wang
>
> **备注:** 8 pages, submitted to IROS2026
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated remarkable performance on complex tasks through imitation learning in recent robotic manipulation works. Based on large-scale and high-quality demonstration datasets, existing imitation learning method arms VLA models acquired with strong capabilities. However, these datasets that predominantly consist of successful trajectories, are costly to collect and often limited in distribution, leading to capability bottlenecks when faced with out-of-distribution (OOD) scenarios during deployment while unable to recover. To address this issue, we propose an automated data augmentation framework named RESample that effectively improves the distribution coverage of VLA training datasets through the well-designed exploratory sampling mechanism. Specifically, the exploratory sampling mechanism identifies the potential coverage gaps during the policy rollout and actively samples exploratory actions to extend the coverage of training data with high sample efficiency. Furthermore, to effectively reflect the distribution of the training dataset, we propose a lightweight Coverage Function that indicates the coverage density of states in the training dataset, which further guides the exploratory sampling process to focus on low-coverage regions. To validate the effectiveness of our method, we conduct extensive experiments on the LIBERO benchmark as well as a series of real-world robotic tasks, demonstrating a significant performance gain of 12% of our proposed RESample over baselines, with only 10-20% additional samples compared to original training data.
>
---
#### [replaced 002] Commanding Humanoid by Free-form Language: A Large Language Action Model with Unified Motion Vocabulary
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人形机器人语言控制任务，旨在解决如何让机器人根据自然语言执行复杂动作的问题。工作包括构建统一运动词汇、设计物理可行控制器及优化训练方法。**

- **链接: [https://arxiv.org/pdf/2511.22963](https://arxiv.org/pdf/2511.22963)**

> **作者:** Zhirui Liu; Kaiyang Ji; Ke Yang; Jingyi Yu; Ye Shi; Jingya Wang
>
> **备注:** Project page: this https URL
>
> **摘要:** Enabling humanoid robots to follow free-form language commands is critical for seamless human-robot interaction, collaborative task execution, and general-purpose embodied intelligence. While recent advances have improved low-level humanoid locomotion and robot manipulation, language-conditioned whole-body control remains a significant challenge. Existing methods are often limited to simple instructions and sacrifice either motion diversity or physical plausibility. To address this, we introduce Humanoid-LLA, a Large Language Action Model that maps expressive language commands to physically executable whole-body actions for humanoid robots. Our approach integrates three core components: a unified motion vocabulary that aligns human and humanoid motion primitives into a shared discrete space; a vocabulary-directed controller distilled from a privileged policy to ensure physical feasibility; and a physics-informed fine-tuning stage using reinforcement learning with dynamics-aware rewards to enhance robustness and stability. Extensive evaluations in simulation and on real-world Unitree G1 and Booster T1 humanoids show that Humanoid-LLA delivers strong language generalization while maintaining high physical fidelity, outperforming existing language-conditioned controllers in motion naturalness, stability, and execution success rate.
>
---
#### [replaced 003] Dejavu: Towards Experience Feedback Learning for Embodied Intelligence
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出Dejavu框架，解决部署后智能体无法持续学习的问题。通过经验反馈网络，增强冻结策略，提升任务适应性和成功率。属于机器人学习任务。**

- **链接: [https://arxiv.org/pdf/2510.10181](https://arxiv.org/pdf/2510.10181)**

> **作者:** Shaokai Wu; Yanbiao Ji; Qiuchang Li; Zhiyi Zhang; Qichen He; Wenyuan Xie; Guodong Zhang; Bayram Bayramli; Yue Ding; Hongtao Lu
>
> **摘要:** Embodied agents face a fundamental limitation: once deployed in real-world environments, they cannot easily acquire new knowledge to improve task performance. In this paper, we propose Dejavu, a general post-deployment learning framework that augments a frozen Vision-Language-Action (VLA) policy with retrieved execution memories through an Experience Feedback Network (EFN). EFN identifies contextually relevant prior action experiences and conditions action prediction on the retrieved guidance. We train EFN with reinforcement learning and semantic similarity rewards, encouraging the predicted actions to align with past behaviors under the current observation. During deployment, EFN continually expands its memory with new trajectories, enabling the agent to exhibit ``learning from experience.'' Experiments across diverse embodied tasks show that EFN improves adaptability, robustness, and success rates over frozen baselines. Our Project Page is this https URL.
>
---
#### [replaced 004] Temporal Transfer Learning for Traffic Optimization with Coarse-grained Advisory Autonomy
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 该论文研究交通优化中的粗粒度建议自主性问题，通过时间迁移学习提升深度强化学习的泛化能力，以提高交通流效率。**

- **链接: [https://arxiv.org/pdf/2312.09436](https://arxiv.org/pdf/2312.09436)**

> **作者:** Jung-Hoon Cho; Sirui Li; Jeongyun Kim; Cathy Wu
>
> **备注:** 18 pages, 12 figures
>
> **摘要:** The recent development of connected and automated vehicle (CAV) technologies has spurred investigations to optimize dense urban traffic to maximize vehicle speed and throughput. This paper explores advisory autonomy, in which real-time driving advisories are issued to the human drivers, thus achieving near-term performance of automated vehicles. Due to the complexity of traffic systems, recent studies of coordinating CAVs have resorted to leveraging deep reinforcement learning (RL). Coarse-grained advisory is formalized as zero-order holds, and we consider a range of hold duration from 0.1 to 40 seconds. However, despite the similarity of the higher frequency tasks on CAVs, a direct application of deep RL fails to be generalized to advisory autonomy tasks. To overcome this, we utilize zero-shot transfer, training policies on a set of source tasks--specific traffic scenarios with designated hold durations--and then evaluating the efficacy of these policies on different target tasks. We introduce Temporal Transfer Learning (TTL) algorithms to select source tasks for zero-shot transfer, systematically leveraging the temporal structure to solve the full range of tasks. TTL selects the most suitable source tasks to maximize the performance of the range of tasks. We validate our algorithms on diverse mixed-traffic scenarios, demonstrating that TTL more reliably solves the tasks than baselines. This paper underscores the potential of coarse-grained advisory autonomy with TTL in traffic flow optimization.
>
---
#### [replaced 005] Traj2Action: A Co-Denoising Framework for Trajectory-Guided Human-to-Robot Skill Transfer
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人到机器人技能迁移任务，解决人类与机器人形态差异导致的知识转移难题。提出Traj2Action框架，通过3D轨迹实现技能迁移，提升机器人性能。**

- **链接: [https://arxiv.org/pdf/2510.00491](https://arxiv.org/pdf/2510.00491)**

> **作者:** Han Zhou; Jinjin Cao; Liyuan Ma; Xueji Fang; Guo-jun Qi
>
> **摘要:** Learning diverse manipulation skills for real-world robots is severely bottlenecked by the reliance on costly and hard-to-scale teleoperated demonstrations. While human videos offer a scalable alternative, effectively transferring manipulation knowledge is fundamentally hindered by the significant morphological gap between human and robotic embodiments. To address this challenge and facilitate skill transfer from human to robot, we introduce Traj2Action, a novel framework that bridges this embodiment gap by using the 3D trajectory of the operational endpoint as a unified intermediate representation, and then transfers the manipulation knowledge embedded in this trajectory to the robot's actions. Our policy first learns to generate a coarse trajectory, which forms a high-level motion plan by leveraging both human and robot data. This plan then conditions the synthesis of precise, robot-specific actions (e.g., orientation and gripper state) within a co-denoising framework. Our work centers on two core objectives: first, the systematic verification of the Traj2Action framework's effectiveness-spanning architectural design, cross-task generalization, and data efficiency and second, the revelation of key laws that govern robot policy learning during the integration of human hand demonstration data. This research focus enables us to provide a scalable paradigm tailored to address human-to-robot skill transfer across morphological gaps. Extensive real-world experiments on a Franka robot demonstrate that Traj2Action boosts the performance by up to 27% and 22.25% over $\pi_0$ baseline on short- and long-horizon real-world tasks, and achieves significant gains as human data scales in robot policy learning.
>
---
#### [replaced 006] SIM1: Physics-Aligned Simulator as Zero-Shot Data Scaler in Deformable Worlds
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文聚焦于柔性物体的机器人操作任务，解决仿真与现实数据不匹配的问题。提出SIM1系统，通过物理对齐实现高效数据生成与策略学习。**

- **链接: [https://arxiv.org/pdf/2604.08544](https://arxiv.org/pdf/2604.08544)**

> **作者:** Yunsong Zhou; Hangxu Liu; Xuekun Jiang; Xing Shen; Yuanzhen Zhou; Hui Wang; Baole Fang; Yang Tian; Mulin Yu; Qiaojun Yu; Li Ma; Hengjie Li; Hanqing Wang; Jia Zeng; Jiangmiao Pang
>
> **备注:** Website: this https URL
>
> **摘要:** Robotic manipulation with deformable objects represents a data-intensive regime in embodied learning, where shape, contact, and topology co-evolve in ways that far exceed the variability of rigids. Although simulation promises relief from the cost of real-world data acquisition, prevailing sim-to-real pipelines remain rooted in rigid-body abstractions, producing mismatched geometry, fragile soft dynamics, and motion primitives poorly suited for cloth interaction. We posit that simulation fails not for being synthetic, but for being ungrounded. To address this, we introduce SIM1, a physics-aligned real-to-sim-to-real data engine that grounds simulation in the physical world. Given limited demonstrations, the system digitizes scenes into metric-consistent twins, calibrates deformable dynamics through elastic modeling, and expands behaviors via diffusion-based trajectory generation with quality filtering. This pipeline transforms sparse observations into scaled synthetic supervision with near-demonstration fidelity. Experiments show that policies trained on purely synthetic data achieve parity with real-data baselines at a 1:15 equivalence ratio, while delivering 90% zero-shot success and 50% generalization gains in real-world deployment. These results validate physics-aligned simulation as scalable supervision for deformable manipulation and a practical pathway for data-efficient policy learning.
>
---
#### [replaced 007] Fine-tuning is Not Enough: A Parallel Framework for Collaborative Imitation and Reinforcement Learning in End-to-end Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，旨在解决传统模仿学习性能受限的问题。通过提出PaIR-Drive框架，实现模仿学习与强化学习的并行协作，提升驾驶性能。**

- **链接: [https://arxiv.org/pdf/2603.13842](https://arxiv.org/pdf/2603.13842)**

> **作者:** Zhexi Lian; Haoran Wang; Xuerun Yan; Weimeng Lin; Xianhong Zhang; Yongyu Chen; Jia Hu
>
> **备注:** 11 pages, 7 figures, 6 tables
>
> **摘要:** End-to-end autonomous driving is typically built upon imitation learning (IL), yet its performance is constrained by the quality of human demonstrations. To overcome this limitation, recent methods incorporate reinforcement learning (RL) through sequential fine-tuning. However, such a paradigm remains suboptimal: sequential RL fine-tuning can introduce policy drift and often leads to a performance ceiling due to its dependence on the pretrained IL policy. To address these issues, we propose PaIR-Drive, a general Parallel framework for collaborative Imitation and Reinforcement learning in end-to-end autonomous driving. During training, PaIR-Drive separates IL and RL into two parallel branches with conflict-free training objectives, enabling fully collaborative optimization. This design eliminates the need to retrain RL when applying a new IL policy. During inference, RL leverages the IL policy to further optimize the final plan, allowing performance beyond prior knowledge of IL. Furthermore, we introduce a tree-structured trajectory neural sampler to group relative policy optimization (GRPO) in the RL branch, which enhances exploration capability. Extensive analysis on NAVSIMv1 and v2 benchmark demonstrates that PaIR-Drive achieves Competitive performance of 91.2 PDMS and 87.9 EPDMS, building upon Transfuser and DiffusionDrive IL baselines. PaIR-Drive consistently outperforms existing RL fine-tuning methods, and could even correct human experts' suboptimal behaviors. Qualitative results further confirm that PaIR-Drive can effectively explore and generate high-quality trajectories.
>
---
#### [replaced 008] You've Got a Golden Ticket: Improving Generative Robot Policies With A Single Noise Vector
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，解决预训练生成式策略性能提升问题。通过引入固定噪声向量（黄金票）替代随机噪声，提升策略在多种任务中的表现。**

- **链接: [https://arxiv.org/pdf/2603.15757](https://arxiv.org/pdf/2603.15757)**

> **作者:** Omkar Patil; Ondrej Biza; Thomas Weng; Karl Schmeckpeper; Wil Thomason; Xiaohan Zhang; Robin Walters; Nakul Gopalan; Sebastian Castro; Eric Rosen
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** What happens when a pretrained generative robot policy is provided a constant initial noise as input, rather than repeatedly sampling it from a Gaussian? We demonstrate that the performance of a pretrained, frozen diffusion or flow matching policy can be improved with respect to a downstream reward by swapping the sampling of initial noise from the prior distribution (typically isotropic Gaussian) with a well-chosen, constant initial noise input -- a golden ticket. We propose a search method to find golden tickets using Monte-Carlo policy evaluation that keeps the pretrained policy frozen, does not train any new networks, and is applicable to all diffusion/flow matching policies (and therefore many VLAs). Our approach to policy improvement makes no assumptions beyond being able to inject initial noise into the policy and calculate (sparse) task rewards of episode rollouts, making it deployable with no additional infrastructure or models. Our method improves the performance of policies in 38 out of 43 tasks across simulated and real-world robot manipulation benchmarks, with relative improvements in success rate by up to 58% for some simulated tasks, and 60% within 50 search episodes for real-world tasks. We also show unique benefits of golden tickets for multi-task settings: the diversity of behaviors from different tickets naturally defines a Pareto frontier for balancing different objectives (e.g., speed, success rates); in VLAs, we find that a golden ticket optimized for one task can also boost performance in other related tasks. We release a codebase with pretrained policies and golden tickets for simulation benchmarks using VLAs, diffusion policies, and flow matching policies.
>
---
#### [replaced 009] Koopman Operator Framework for Modeling and Control of Off-Road Vehicle on Deformable Terrain
- **分类: eess.SY; cs.RO; math.DS**

- **简介: 该论文属于自主越野车控制任务，旨在解决变形地形下车辆建模与控制问题。通过构建融合物理模型与数据驱动的Koopman框架，提升控制精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.28965](https://arxiv.org/pdf/2603.28965)**

> **作者:** Kartik Loya; Phanindra Tallapragada
>
> **备注:** 11 pages, 14 figures, 4 tables. Submitted to ASME Journal of Autonomous Vehicles (JAVS-26-1012)
>
> **摘要:** This work presents a hybrid physics-informed and data-driven modeling framework for predictive control of autonomous off-road vehicles operating on deformable terrain. Traditional high-fidelity terramechanics models are often too computationally demanding to be directly used in control design. Modern Koopman operator methods can be used to represent the complex terramechanics and vehicle dynamics in a linear form. We develop a framework whereby a Koopman linear system can be constructed using data from simulations of a vehicle moving on deformable terrain. For vehicle simulations, the deformable-terrain terramechanics are modeled using Bekker-Wong theory, and the vehicle is represented as a simplified five-degree-of-freedom (5-DOF) system. The Koopman operators are identified from large simulation datasets for sandy loam and clay using a recursive subspace identification method, where Grassmannian distance is used to prioritize informative data segments during training. The advantage of this approach is that the Koopman operator learned from simulations can be updated with data from the physical system in a seamless manner, making this a hybrid physics-informed and data-driven approach. Prediction results demonstrate stable short-horizon accuracy and robustness under mild terrain-height variations. When embedded in a constrained MPC, the learned predictor enables stable closed-loop tracking of aggressive maneuvers while satisfying steering and torque limits.
>
---
#### [replaced 010] Dream to Fly: Model-Based Reinforcement Learning for Vision-Based Drone Flight
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，解决无人机自主飞行问题。通过模型基于强化学习，仅用像素训练出高效飞行策略，实现高速自主穿越赛道。**

- **链接: [https://arxiv.org/pdf/2501.14377](https://arxiv.org/pdf/2501.14377)**

> **作者:** Angel Romero; Ashwin Shenai; Ismail Geles; Elie Aljalbout; Davide Scaramuzza
>
> **备注:** 8 pages, 6 Figures, accepted to IEEE ICRA 2026
>
> **摘要:** Autonomous drone racing has risen as a challenging robotic benchmark for testing the limits of learning, perception, planning, and control. Expert human pilots are able to fly a drone through a race track by mapping pixels from a single camera directly to control commands. Recent works in autonomous drone racing attempting direct pixel-to-commands control policies have relied on either intermediate representations that simplify the observation space or performed extensive bootstrapping using Imitation Learning (IL). This paper leverages DreamerV3 to train visuomotor policies capable of agile flight through a racetrack using only pixels as observations. In contrast to model-free methods like PPO or SAC, which are sample-inefficient and struggle in this setting, our approach acquires drone racing skills from pixels. Notably, a perception-aware behaviour of actively steering the camera toward texture-rich gate regions emerges without the need of handcrafted reward terms for the viewing direction. Our experiments show in both, simulation and real-world flight using a hardware-in-the-loop setup with rendered image observations, how the proposed approach can be deployed on real quadrotors at speeds of up to 9 m/s. These results advance the state of pixel-based autonomous flight and demonstrate that MBRL offers a promising path for real-world robotics research.
>
---
#### [replaced 011] REACT3D: Recovering Articulations for Interactive Physical 3D Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出REACT3D，解决静态3D场景交互化问题，通过检测、分割、估计关节等步骤生成可模拟的互动场景，提升场景理解研究效率。**

- **链接: [https://arxiv.org/pdf/2510.11340](https://arxiv.org/pdf/2510.11340)**

> **作者:** Zhao Huang; Boyang Sun; Alexandros Delitzas; Jiaqi Chen; Marc Pollefeys
>
> **备注:** Accepted at IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Interactive 3D scenes are increasingly vital for embodied intelligence, yet existing datasets remain limited due to the labor-intensive process of annotating part segmentation, kinematic types, and motion trajectories. We present REACT3D, a scalable zero-shot framework that converts static 3D scenes into simulation-ready interactive replicas with consistent geometry, enabling direct use in diverse downstream tasks. Our contributions include: (i) openable-object detection and segmentation to extract candidate movable parts from static scenes, (ii) articulation estimation that infers joint types and motion parameters, (iii) hidden-geometry completion followed by interactive object assembly, and (iv) interactive scene integration in widely supported formats to ensure compatibility with standard simulation platforms. We achieve state-of-the-art performance on detection/segmentation and articulation metrics across diverse indoor scenes, demonstrating the effectiveness of our framework and providing a practical foundation for scalable interactive scene generation, thereby lowering the barrier to large-scale research on articulated scene understanding. Our project page is this https URL
>
---
#### [replaced 012] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文属于机器人视觉-语言-动作任务，解决传统方法忽略历史信息的问题。提出AVA-VLA框架，通过递归状态和主动视觉注意力提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.18960](https://arxiv.org/pdf/2511.18960)**

> **作者:** Lei Xiao; Jifeng Li; Juntao Gao; Feiyang Ye; Yan Jin; Jingjing Qian; Jing Zhang; Yong Wu; Xiaoyuan Yu
>
> **备注:** Accepted at CVPR 2026 (Highlight)
>
> **摘要:** Vision-Language-Action (VLA) models have shown remarkable progress in embodied tasks recently, but most methods process visual observations independently at each timestep. This history-agnostic design treats robot manipulation as a Markov Decision Process, even though real-world robotic control is inherently partially observable and requires reasoning over past interactions. To address this mismatch, we reformulate VLA policy learning from a Partially Observable Markov Decision Process perspective and propose AVA-VLA, a framework that conditions action generation on a recurrent state that serves as a neural approximation to the agent's belief over task history. Built on this recurrent state, we introduce Active Visual Attention (AVA), which dynamically reweights visual tokens in the current observation to focus on regions most relevant given both the instruction and execution history. Extensive experiments show that AVA-VLA achieves state-of-the-art performance on standard robotic benchmarks, including LIBERO and CALVIN, and transfers effectively to real-world dual-arm manipulation tasks. These results demonstrate the effectiveness of temporally grounded active visual processing for improving VLA performance in robotic sequential decision-making. The project page is available at this https URL.
>
---
#### [replaced 013] CaRLi-V: Camera-RADAR-LiDAR Point-Wise 3D Velocity Estimation
- **分类: cs.RO**

- **简介: 该论文属于3D速度估计任务，旨在解决动态环境中物体点级速度精确估计的问题。通过融合雷达、激光雷达和相机数据，提出CaRLi-V方法，实现高精度的三维速度估计。**

- **链接: [https://arxiv.org/pdf/2511.01383](https://arxiv.org/pdf/2511.01383)**

> **作者:** Landson Guo; Andres M. Diaz Aguilar; William Talbot; Turcan Tuna; Marco Hutter; Cesar Cadena
>
> **摘要:** Accurate point-wise velocity estimation in 3D is crucial for robot interaction with non-rigid dynamic agents, enabling robust performance in path planning, collision avoidance, and object manipulation in dynamic environments. To this end, this paper proposes a novel RADAR, LiDAR, and camera fusion pipeline for point-wise 3D velocity estimation named CaRLi-V. This pipeline leverages raw RADAR measurements to create a novel RADAR representation, the velocity cube, which densely encodes RADAR radial velocities. By combining the velocity cube for radial velocity extraction, optical flow for tangential velocity estimation, and LiDAR for point-wise range measurements through a closed-form solution, our approach can produce 3D velocity estimates for a dense array of points. Developed as an open-source ROS2 package, CaRLi-V has been field-tested on a custom dataset and achieves low velocity error metrics relative to ground truth while outperforming state-of-the-art scene flow methods.
>
---
#### [replaced 014] Harnessing Embodied Agents: Runtime Governance for Policy-Constrained Execution
- **分类: cs.RO**

- **简介: 该论文属于智能体系统任务，解决运行时治理问题。提出分离认知与监管的框架，实现安全执行控制，提升系统可靠性。**

- **链接: [https://arxiv.org/pdf/2604.07833](https://arxiv.org/pdf/2604.07833)**

> **作者:** Xue Qin; Simin Luan; John See; Cong Yang; Zhijun Li
>
> **备注:** 36 pages, 3 figures, 10 tables
>
> **摘要:** Embodied agents are evolving from passive reasoning systems into active executors that interact with tools, robots, and physical environments. Once granted execution authority, the central challenge becomes how to keep actions governable at runtime. Existing approaches embed safety and recovery logic inside the agent loop, making execution control difficult to standardize, audit, and adapt. This paper argues that embodied intelligence requires not only stronger agents, but stronger runtime governance. We propose a framework for policy-constrained execution that separates agent cognition from execution oversight. Governance is externalized into a dedicated runtime layer performing policy checking, capability admission, execution monitoring, rollback handling, and human override. We formalize the control boundary among the embodied agent, Embodied Capability Modules (ECMs), and runtime governance layer, and validate through 1000 randomized simulation trials across three governance dimensions. Results show 96.2% interception of unauthorized actions, reduction of unsafe continuation from 100% to 22.2% under runtime drift, and 91.4% recovery success with full policy compliance, substantially outperforming all baselines (p<0.001). By reframing runtime governance as a first-class systems problem, this paper positions policy-constrained execution as a key design principle for embodied agent systems.
>
---
#### [replaced 015] Informed Hybrid Zonotope-based Motion Planning Algorithm
- **分类: cs.RO**

- **简介: 该论文属于运动规划任务，解决非凸自由空间中的路径规划问题。提出HZ-MP算法，通过分解空间和引导采样提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2507.09309](https://arxiv.org/pdf/2507.09309)**

> **作者:** Peng Xie; Johannes Betz; Amr Alanwar
>
> **摘要:** Optimal path planning in nonconvex free spaces poses substantial computational challenges. A common approach formulates such problems as mixed-integer linear programs (MILPs); however, solving general MILPs is computationally intractable and severely limits scalability. To address these limitations, we propose HZ-MP, an informed Hybrid Zonotope-based Motion Planner, which decomposes the obstacle-free space and performs low-dimensional face sampling guided by an ellipsotope heuristic, thereby concentrating exploration on promising transition regions. This structured exploration mitigates the excessive wasted sampling that degrades existing informed planners in narrow-passage or enclosed-goal scenarios. We prove that HZ-MP is probabilistically complete and asymptotically optimal, and demonstrate empirically that it converges to high-quality trajectories within a small number of iterations.
>
---
#### [replaced 016] CrashSight: A Phase-Aware, Infrastructure-Centric Video Benchmark for Traffic Crash Scene Understanding and Reasoning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出CrashSight，一个用于交通碰撞场景理解的视觉语言基准，解决自动驾驶中基础设施视角下的事故分析问题。通过真实路侧视频数据，评估模型在时间与因果推理上的能力。**

- **链接: [https://arxiv.org/pdf/2604.08457](https://arxiv.org/pdf/2604.08457)**

> **作者:** Rui Gan; Junyi Ma; Pei Li; Xingyou Yang; Kai Chen; Sikai Chen; Bin Ran
>
> **摘要:** Cooperative autonomous driving requires traffic scene understanding from both vehicle and infrastructure perspectives. While vision-language models (VLMs) show strong general reasoning capabilities, their performance in safety-critical traffic scenarios remains insufficiently evaluated due to the ego-vehicle focus of existing benchmarks. To bridge this gap, we present \textbf{CrashSight}, a large-scale vision-language benchmark for roadway crash understanding using real-world roadside camera data. The dataset comprises 250 crash videos, annotated with 13K multiple-choice question-answer pairs organized under a two-tier taxonomy. Tier 1 evaluates the visual grounding of scene context and involved parties, while Tier 2 probes higher-level reasoning, including crash mechanics, causal attribution, temporal progression, and post-crash outcomes. We benchmark 8 state-of-the-art VLMs and show that, despite strong scene description capabilities, current models struggle with temporal and causal reasoning in safety-critical scenarios. We provide a detailed analysis of failure scenarios and discuss directions for improving VLM crash understanding. The benchmark provides a standardized evaluation framework for infrastructure-assisted perception in cooperative autonomous driving. The CrashSight benchmark, including the full dataset and code, is accessible at this https URL.
>
---
#### [replaced 017] The Impact of Gait Pattern Personalization on the Perception of Rigid Robotic Guidance: A Pilot User Experience Evaluation
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，研究个性化步态对用户感知的影响。通过实验比较不同步态模式，探讨个性化设计对用户体验的作用。**

- **链接: [https://arxiv.org/pdf/2512.17425](https://arxiv.org/pdf/2512.17425)**

> **作者:** Beatrice Luciani; Katherine Lin Poggensee; Heike Vallery; Alex van den Berg; Severin David Woernle; Mostafa Mogharabi; Stefano Dalla Gasperina; Laura Marchal-Crespo
>
> **摘要:** Exoskeletons modulate human movement across diverse applications, from performance augmentation to daily-life assistance. These systems often enforce specific kinematic patterns to mitigate injury risks and motivate users to keep moving despite diminished capacity. However, little is known about users' perception of such robot-imposed guidance, especially when personalized to the uniqueness of individual human walk. Given the usually substantial computational cost for personalization, understanding its subjective impact is essential to justify its implementation over standard patterns. Ten unimpaired participants completed a within-subject experiment in a multi-planar treadmill-based exoskeleton that enforced three different gait patterns: personalized, standard, and a randomly selected pattern from a publicly available database. Personalization was achieved using a data-driven framework that predicts hip, knee, and pelvis trajectories from walking speed, anthropometric, and demographic data. The standard pattern was obtained by averaging gait patterns from the aforementioned database. After each condition, participants rated enjoyment, comfort, and perceived naturalness. Knee joint interaction forces were also recorded. Subjective ratings revealed no significant differences among patterns, despite all trajectories being executed with high accuracy. However, gait patterns experienced last were rated as significantly more comfortable and natural, indicating adaptation to the system. Higher interaction forces were observed only for the random vs. standard pattern. Personalizing gait kinematics had minimal short-term influence on user experience relative to the dominant effect of adaptation to the exoskeleton. These findings highlight the importance of integrating subjective feedback and accounting for user adaptation when designing personalized robot controllers.
>
---
#### [replaced 018] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决动作分块大小固定导致的响应与一致性失衡问题，提出自适应动作分块策略，提升机器人操作性能。**

- **链接: [https://arxiv.org/pdf/2604.04161](https://arxiv.org/pdf/2604.04161)**

> **作者:** Yuanchang Liang; Xiaobo Wang; Kai Wang; Shuo Wang; Xiaojiang Peng; Haoyu Chen; David Kim Huat Chua; Prahlad Vadakkepat
>
> **备注:** accepted by CVPR 2026
>
> **摘要:** In Vision-Language-Action (VLA) models, action chunking (i.e., executing a sequence of actions without intermediate replanning) is a key technique to improve robotic manipulation abilities. However, a large chunk size reduces the model's responsiveness to new information, while a small one increases the likelihood of mode-jumping, jerky behavior resulting from discontinuities between chunks. Therefore, selecting the optimal chunk size is an urgent demand to balance the model's reactivity and consistency. Unfortunately, a dominant trend in current VLA models is an empirical fixed chunk length at inference-time, hindering their superiority and scalability across diverse manipulation tasks. To address this issue, we propose a novel Adaptive Action Chunking (AAC) strategy, which exploits action entropy as the cue to adaptively determine the chunk size based on current predictions. Extensive experiments on a wide range of simulated and real-world robotic manipulation tasks have demonstrated that our approach substantially improves performance over the state-of-the-art alternatives. The videos and source code are publicly available at this https URL.
>
---
#### [replaced 019] Governed Capability Evolution for Embodied Agents: Safe Upgrade, Compatibility Checking, and Runtime Rollback for Embodied Capability Modules
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人系统任务，解决 embodied agents 能力模块升级中的安全与兼容性问题。提出生命周期感知的升级框架，包含多项兼容性检查和回滚机制，确保升级过程安全可靠。**

- **链接: [https://arxiv.org/pdf/2604.08059](https://arxiv.org/pdf/2604.08059)**

> **作者:** Xue Qin; Simin Luan; John See; Cong Yang; Zhijun Li
>
> **备注:** 46 pages, 3 figures, 10 tables, 7 appendices
>
> **摘要:** Embodied agents are increasingly expected to improve over time by updating their executable capabilities rather than rewriting the agent itself. Prior work has separately studied modular capability packaging, capability evolution, and runtime governance. However, a key systems problem remains underexplored: once an embodied capability module evolves into a new version, how can the hosting system deploy it safely without breaking policy constraints, execution assumptions, or recovery guarantees? We formulate governed capability evolution as a first-class systems problem for embodied agents. We propose a lifecycle-aware upgrade framework in which every new capability version is treated as a governed deployment candidate rather than an immediately executable replacement. The framework introduces four upgrade compatibility checks -- interface, policy, behavioral, and recovery -- and organizes them into a staged runtime pipeline comprising candidate validation, sandbox evaluation, shadow deployment, gated activation, online monitoring, and rollback. We evaluate over 6 rounds of capability upgrade with 15 random seeds. Naive upgrade achieves 72.9% task success but drives unsafe activation to 60% by the final round; governed upgrade retains comparable success (67.4%) while maintaining zero unsafe activations across all rounds (Wilcoxon p=0.003). Shadow deployment reveals 40% of regressions invisible to sandbox evaluation alone, and rollback succeeds in 79.8% of post-activation drift scenarios.
>
---
#### [replaced 020] SimScale: Learning to Drive via Real-World Simulation at Scale
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决真实数据不足问题。通过SimScale框架生成大量模拟数据，并结合真实数据训练，提升模型鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.23369](https://arxiv.org/pdf/2511.23369)**

> **作者:** Haochen Tian; Tianyu Li; Haochen Liu; Jiazhi Yang; Yihang Qiu; Guang Li; Junli Wang; Yinfeng Gao; Zhang Zhang; Liang Wang; Hangjun Ye; Tieniu Tan; Long Chen; Hongyang Li
>
> **备注:** CVPR 2026 Oral. Project page: this https URL
>
> **摘要:** Achieving fully autonomous driving systems requires learning rational decisions in a wide span of scenarios, including safety-critical and out-of-distribution ones. However, such cases are underrepresented in real-world corpus collected by human experts. To complement for the lack of data diversity, we introduce a novel and scalable simulation framework capable of synthesizing massive unseen states upon existing driving logs. Our pipeline utilizes advanced neural rendering with a reactive environment to generate high-fidelity multi-view observations controlled by the perturbed ego trajectory. Furthermore, we develop a pseudo-expert trajectory generation mechanism for these newly simulated states to provide action supervision. Upon the synthesized data, we find that a simple co-training strategy on both real-world and simulated samples can lead to significant improvements in both robustness and generalization for various planning methods on challenging real-world benchmarks, up to +8.6 EPDMS on navhard and +2.9 on navtest. More importantly, such policy improvement scales smoothly by increasing simulation data only, even without extra real-world data streaming in. We further reveal several crucial findings of such a sim-real learning system, which we term SimScale, including the design of pseudo-experts and the scaling properties for different policy architectures. Simulation data and code have been released at this https URL.
>
---
#### [replaced 021] UAV-Track VLA: Embodied Aerial Tracking via Vision-Language-Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于无人机视觉跟踪任务，解决动态环境中目标跟踪的精度与效率问题。提出UAV-Track VLA模型，提升长距离跟踪性能和实时性。**

- **链接: [https://arxiv.org/pdf/2604.02241](https://arxiv.org/pdf/2604.02241)**

> **作者:** Qiyao Zhang; Shuhua Zheng; Jianli Sun; Chengxiang Li; Xianke Wu; Zihan Song; Zhiyong Cui; Yisheng Lv; Yonglin Tian
>
> **摘要:** Embodied visual tracking is crucial for Unmanned Aerial Vehicles (UAVs) executing complex real-world tasks. In dynamic urban scenarios with complex semantic requirements, Vision-Language-Action (VLA) models show great promise due to their cross-modal fusion and continuous action generation capabilities. To benchmark multimodal tracking in such environments, we construct a dedicated evaluation benchmark and a large-scale dataset encompassing over 890K frames, 176 tasks, and 85 diverse objects. Furthermore, to address temporal feature redundancy and the lack of spatial geometric priors in existing VLA models, we propose an improved VLA tracking model, UAV-Track VLA. Built upon the $\pi_{0.5}$ architecture, our model introduces a temporal compression net to efficiently capture inter-frame dynamics. Additionally, a parallel dual-branch decoder comprising a spatial-aware auxiliary grounding head and a flow matching action expert is designed to decouple cross-modal features and generate fine-grained continuous actions. Systematic experiments in the CARLA simulator validate the superior end-to-end performance of our method. Notably, in challenging long-distance pedestrian tracking tasks, UAV-Track VLA achieves a 61.76\% success rate and 269.65 average tracking frames, significantly outperforming existing baselines. Furthermore, it demonstrates robust zero-shot generalization in unseen environments and reduces single-step inference latency by 33.4\% (to 0.0571s) compared to the original $\pi_{0.5}$, enabling highly efficient, real-time UAV control. Data samples and demonstration videos are available at: this https URL.
>
---
#### [replaced 022] Allocation for Omnidirectional Aerial Robots: Incorporating Power Dynamics
- **分类: cs.RO**

- **简介: 该论文研究倾斜旋翼飞行器的执行器分配问题，旨在解决动态协调与动力学建模难题。提出三种新方法，提升系统性能与灵活性。**

- **链接: [https://arxiv.org/pdf/2412.16107](https://arxiv.org/pdf/2412.16107)**

> **作者:** Eugenio Cuniato; Mike Allenspach; Thomas Stastny; Helen Oleynikova; Roland Siegwart; Michael Pantic
>
> **摘要:** Tilt-rotor aerial robots are more dynamic and versatile than fixed-rotor platforms, since the thrust vector and body orientation are decoupled. However, the coordination of servos and propellers (the allocation problem) is not trivial, especially accounting for overactuation and actuator dynamics. We incrementally build and present three novel allocation methods for tilt-rotor aerial robots, comparing them to state-of-the-art methods on a real system performing dynamic maneuvers. We extend the state-of-the-art geometric allocation into a differential allocation, which uses the platform's redundancy and does not suffer from singularities. We expand it by incorporating actuator dynamics and propeller power dynamics. These allow us to model dynamic propeller acceleration limits, bringing two main advantages: balancing propeller speed without the need for nullspace goals and allowing the platform to selectively turn off propellers during flight, opening the door to new manipulation possibilities. We also use actuator dynamics and limits to normalize the allocation problem, making it easier to tune and allowing it to track 70% faster trajectories than a geometric allocation.
>
---
