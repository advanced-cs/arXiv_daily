# 机器人 cs.RO

- **最新发布 129 篇**

- **更新 70 篇**

## 最新发布

#### [new 001] JuggleRL: Mastering Ball Juggling with a Quadrotor via Deep Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文研究空中机器人进行球类杂耍的任务，解决精确控制与动态交互问题。提出JuggleRL系统，通过深度强化学习实现高效球类杂耍。**

- **链接: [http://arxiv.org/pdf/2509.24892v1](http://arxiv.org/pdf/2509.24892v1)**

> **作者:** Shilong Ji; Yinuo Chen; Chuqi Wang; Jiayu Chen; Ruize Zhang; Feng Gao; Wenhao Tang; Shu'ang Yu; Sirui Xiang; Xinlei Chen; Chao Yu; Yu Wang
>
> **摘要:** Aerial robots interacting with objects must perform precise, contact-rich maneuvers under uncertainty. In this paper, we study the problem of aerial ball juggling using a quadrotor equipped with a racket, a task that demands accurate timing, stable control, and continuous adaptation. We propose JuggleRL, the first reinforcement learning-based system for aerial juggling. It learns closed-loop policies in large-scale simulation using systematic calibration of quadrotor and ball dynamics to reduce the sim-to-real gap. The training incorporates reward shaping to encourage racket-centered hits and sustained juggling, as well as domain randomization over ball position and coefficient of restitution to enhance robustness and transferability. The learned policy outputs mid-level commands executed by a low-level controller and is deployed zero-shot on real hardware, where an enhanced perception module with a lightweight communication protocol reduces delays in high-frequency state estimation and ensures real-time control. Experiments show that JuggleRL achieves an average of $311$ hits over $10$ consecutive trials in the real world, with a maximum of $462$ hits observed, far exceeding a model-based baseline that reaches at most $14$ hits with an average of $3.1$. Moreover, the policy generalizes to unseen conditions, successfully juggling a lighter $5$ g ball with an average of $145.9$ hits. This work demonstrates that reinforcement learning can empower aerial robots with robust and stable control in dynamic interaction tasks.
>
---
#### [new 002] LLM-Handover:Exploiting LLMs for Task-Oriented Robot-Human Handovers
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，解决机器人交接物体时忽视人类后续动作的问题。通过结合大语言模型与部件分割，实现更智能的抓取选择与执行。**

- **链接: [http://arxiv.org/pdf/2509.24706v1](http://arxiv.org/pdf/2509.24706v1)**

> **作者:** Andreea Tulbure; Rene Zurbruegg; Timm Grigat; Marco Hutter
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Effective human-robot collaboration depends on task-oriented handovers, where robots present objects in ways that support the partners intended use. However, many existing approaches neglect the humans post-handover action, relying on assumptions that limit generalizability. To address this gap, we propose LLM-Handover, a novel framework that integrates large language model (LLM)-based reasoning with part segmentation to enable context-aware grasp selection and execution. Given an RGB-D image and a task description, our system infers relevant object parts and selects grasps that optimize post-handover usability. To support evaluation, we introduce a new dataset of 60 household objects spanning 12 categories, each annotated with detailed part labels. We first demonstrate that our approach improves the performance of the used state-of-the-art part segmentation method, in the context of robot-human handovers. Next, we show that LLM-Handover achieves higher grasp success rates and adapts better to post-handover task constraints. During hardware experiments, we achieve a success rate of 83% in a zero-shot setting over conventional and unconventional post-handover tasks. Finally, our user study underlines that our method enables more intuitive, context-aware handovers, with participants preferring it in 86% of cases.
>
---
#### [new 003] High-Precision Climbing Robot Localization Using Planar Array UWB/GPS/IMU/Barometer Integration
- **分类: cs.RO**

- **简介: 该论文属于高精度定位任务，解决复杂高海拔环境下爬墙机器人定位问题，通过多传感器融合提升定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.23801v1](http://arxiv.org/pdf/2509.23801v1)**

> **作者:** Shuning Zhang; Renjing Xu; Zhanchen Zhu; Xiangyu Chen; Yunheng Wang; Xu Jiang; Peibo Duan
>
> **摘要:** To address the need for high-precision localization of climbing robots in complex high-altitude environments, this paper proposes a multi-sensor fusion system that overcomes the limitations of single-sensor approaches. Firstly, the localization scenarios and the problem model are analyzed. An integrated architecture of Attention Mechanism-based Fusion Algorithm (AMFA) incorporating planar array Ultra-Wideband (UWB), GPS, Inertial Measurement Unit (IMU), and barometer is designed to handle challenges such as GPS occlusion and UWB Non-Line-of-Sight (NLOS) problem. Then, End-to-end neural network inference models for UWB and barometer are developed, along with a multimodal attention mechanism for adaptive data fusion. An Unscented Kalman Filter (UKF) is applied to refine the trajectory, improving accuracy and robustness. Finally, real-world experiments show that the method achieves 0.48 m localization accuracy and lower MAX error of 1.50 m, outperforming baseline algorithms such as GPS/INS-EKF and demonstrating stronger robustness.
>
---
#### [new 004] Robot Learning from Any Images
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出RoLA框架，解决从单张图像生成物理交互机器人环境的问题。通过场景恢复与视觉融合，实现快速数据生成与机器人学习。**

- **链接: [http://arxiv.org/pdf/2509.22970v1](http://arxiv.org/pdf/2509.22970v1)**

> **作者:** Siheng Zhao; Jiageng Mao; Wei Chow; Zeyu Shangguan; Tianheng Shi; Rong Xue; Yuxi Zheng; Yijia Weng; Yang You; Daniel Seita; Leonidas Guibas; Sergey Zakharov; Vitor Guizilini; Yue Wang
>
> **备注:** CoRL 2025 camera ready
>
> **摘要:** We introduce RoLA, a framework that transforms any in-the-wild image into an interactive, physics-enabled robotic environment. Unlike previous methods, RoLA operates directly on a single image without requiring additional hardware or digital assets. Our framework democratizes robotic data generation by producing massive visuomotor robotic demonstrations within minutes from a wide range of image sources, including camera captures, robotic datasets, and Internet images. At its core, our approach combines a novel method for single-view physical scene recovery with an efficient visual blending strategy for photorealistic data collection. We demonstrate RoLA's versatility across applications like scalable robotic data generation and augmentation, robot learning from Internet images, and single-image real-to-sim-to-real systems for manipulators and humanoids. Video results are available at https://sihengz02.github.io/RoLA .
>
---
#### [new 005] AIRoA MoMa Dataset: A Large-Scale Hierarchical Dataset for Mobile Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于移动操作任务，旨在解决机器人在复杂环境中执行长时序、接触丰富的操作问题。工作包括构建包含多模态数据和层次标注的大型数据集AIRoA MoMa。**

- **链接: [http://arxiv.org/pdf/2509.25032v1](http://arxiv.org/pdf/2509.25032v1)**

> **作者:** Ryosuke Takanami; Petr Khrapchenkov; Shu Morikuni; Jumpei Arima; Yuta Takaba; Shunsuke Maeda; Takuya Okubo; Genki Sano; Satoshi Sekioka; Aoi Kadoya; Motonari Kambara; Naoya Nishiura; Haruto Suzuki; Takanori Yoshimoto; Koya Sakamoto; Shinnosuke Ono; Hu Yang; Daichi Yashima; Aoi Horo; Tomohiro Motoda; Kensuke Chiyoma; Hiroshi Ito; Koki Fukuda; Akihito Goto; Kazumi Morinaga; Yuya Ikeda; Riko Kawada; Masaki Yoshikawa; Norio Kosuge; Yuki Noguchi; Kei Ota; Tatsuya Matsushima; Yusuke Iwasawa; Yutaka Matsuo; Tetsuya Ogata
>
> **摘要:** As robots transition from controlled settings to unstructured human environments, building generalist agents that can reliably follow natural language instructions remains a central challenge. Progress in robust mobile manipulation requires large-scale multimodal datasets that capture contact-rich and long-horizon tasks, yet existing resources lack synchronized force-torque sensing, hierarchical annotations, and explicit failure cases. We address this gap with the AIRoA MoMa Dataset, a large-scale real-world multimodal dataset for mobile manipulation. It includes synchronized RGB images, joint states, six-axis wrist force-torque signals, and internal robot states, together with a novel two-layer annotation schema of sub-goals and primitive actions for hierarchical learning and error analysis. The initial dataset comprises 25,469 episodes (approx. 94 hours) collected with the Human Support Robot (HSR) and is fully standardized in the LeRobot v2.1 format. By uniquely integrating mobile manipulation, contact-rich interaction, and long-horizon structure, AIRoA MoMa provides a critical benchmark for advancing the next generation of Vision-Language-Action models. The first version of our dataset is now available at https://huggingface.co/datasets/airoa-org/airoa-moma .
>
---
#### [new 006] IA-VLA: Input Augmentation for Vision-Language-Action models in settings with semantically complex tasks
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决VLAs在复杂语言指令下的理解不足问题。通过引入IA-VLA框架，利用大语言模型增强输入，提升VLA性能。**

- **链接: [http://arxiv.org/pdf/2509.24768v1](http://arxiv.org/pdf/2509.24768v1)**

> **作者:** Eric Hannus; Miika Malin; Tran Nguyen Le; Ville Kyrki
>
> **备注:** Under review for ICRA 2026
>
> **摘要:** Vision-language-action models (VLAs) have become an increasingly popular approach for addressing robot manipulation problems in recent years. However, such models need to output actions at a rate suitable for robot control, which limits the size of the language model they can be based on, and consequently, their language understanding capabilities. Manipulation tasks may require complex language instructions, such as identifying target objects by their relative positions, to specify human intention. Therefore, we introduce IA-VLA, a framework that utilizes the extensive language understanding of a large vision language model as a pre-processing stage to generate improved context to augment the input of a VLA. We evaluate the framework on a set of semantically complex tasks which have been underexplored in VLA literature, namely tasks involving visual duplicates, i.e., visually indistinguishable objects. A dataset of three types of scenes with duplicate objects is used to compare a baseline VLA against two augmented variants. The experiments show that the VLA benefits from the augmentation scheme, especially when faced with language instructions that require the VLA to extrapolate from concepts it has seen in the demonstrations. For the code, dataset, and videos, see https://sites.google.com/view/ia-vla.
>
---
#### [new 007] Real-time Recognition of Human Interactions from a Single RGB-D Camera for Socially-Aware Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于社交机器人导航任务，旨在识别人类互动以提升人机交互自然性。通过RGB-D相机和PCA、鞋带公式等方法，实时检测群体互动并计算参与区域。**

- **链接: [http://arxiv.org/pdf/2509.24907v1](http://arxiv.org/pdf/2509.24907v1)**

> **作者:** Thanh Long Nguyen; Duc Phu Nguyen; Thanh Thao Ton Nu; Quan Le; Thuan Hoang Tran; Manh Duong Phung
>
> **摘要:** {Recognizing human interactions is essential for social robots as it enables them to navigate safely and naturally in shared environments. Conventional robotic systems however often focus on obstacle avoidance, neglecting social cues necessary for seamless human-robot interaction. To address this gap, we propose a framework to recognize human group interactions for socially aware navigation. Our method utilizes color and depth frames from a monocular RGB-D camera to estimate 3D human keypoints and positions. Principal component analysis (PCA) is then used to determine dominant interaction directions. The shoelace formula is finally applied to compute interest points and engagement areas. Extensive experiments have been conducted to evaluate the validity of the proposed method. The results show that our method is capable of recognizing group interactions across different scenarios with varying numbers of individuals. It also achieves high-speed performance, processing each frame in approximately 4 ms on a single-board computer used in robotic systems. The method is implemented as a ROS 2 package making it simple to integrate into existing navigation systems. Source code is available at https://github.com/thanhlong103/social-interaction-detector
>
---
#### [new 008] Nonlinear Model Predictive Control with Single-Shooting Method for Autonomous Personal Mobility Vehicle
- **分类: cs.RO**

- **简介: 该论文属于自主车辆控制任务，解决路径规划与避障问题，采用NMPC单次射击方法实现SEATER的精准控制与约束满足。**

- **链接: [http://arxiv.org/pdf/2509.22694v1](http://arxiv.org/pdf/2509.22694v1)**

> **作者:** Rakha Rahmadani Pratama; Catur Hilman A. H. B. Baskoro; Joga Dharma Setiawan; Dyah Kusuma Dewi; P Paryanto; Mochammad Ariyanto; Roni Permana Saputra
>
> **备注:** 15 pages, 3 figures, 4 tables
>
> **摘要:** This paper introduces a proposed control method for autonomous personal mobility vehicles, specifically the Single-passenger Electric Autonomous Transporter (SEATER), using Nonlinear Model Predictive Control (NMPC). The proposed method leverages a single-shooting approach to solve the optimal control problem (OCP) via non-linear programming (NLP). The proposed NMPC is implemented to a non-holonomic vehicle with a differential drive system, using odometry data as localization feedback to guide the vehicle towards its target pose while achieving objectives and adhering to constraints, such as obstacle avoidance. To evaluate the performance of the proposed method, a number of simulations have been conducted in both obstacle-free and static obstacle environments. The SEATER model and testing environment have been developed in the Gazebo Simulation and the NMPC are implemented within the Robot Operating System (ROS) framework. The simulation results demonstrate that the NMPC-based approach successfully controls the vehicle to reach the desired target location while satisfying the imposed constraints. Furthermore, this study highlights the robustness and real-time effectiveness of NMPC with a single-shooting approach for autonomous vehicle control in the evaluated scenarios.
>
---
#### [new 009] MAD-PINN: A Decentralized Physics-Informed Machine Learning Framework for Safe and Optimal Multi-Agent Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多智能体控制任务，旨在解决大规模系统中安全与性能的协同优化问题。提出MAD-PINN框架，结合物理信息神经网络和分布式决策，提升安全性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2509.23960v1](http://arxiv.org/pdf/2509.23960v1)**

> **作者:** Manan Tayal; Aditya Singh; Shishir Kolathaya; Somil Bansal
>
> **备注:** 9 Pages, 4 Figures, 3 Tables. First two authors have contributed equally
>
> **摘要:** Co-optimizing safety and performance in large-scale multi-agent systems remains a fundamental challenge. Existing approaches based on multi-agent reinforcement learning (MARL), safety filtering, or Model Predictive Control (MPC) either lack strict safety guarantees, suffer from conservatism, or fail to scale effectively. We propose MAD-PINN, a decentralized physics-informed machine learning framework for solving the multi-agent state-constrained optimal control problem (MASC-OCP). Our method leverages an epigraph-based reformulation of SC-OCP to simultaneously capture performance and safety, and approximates its solution via a physics-informed neural network. Scalability is achieved by training the SC-OCP value function on reduced-agent systems and deploying them in a decentralized fashion, where each agent relies only on local observations of its neighbours for decision-making. To further enhance safety and efficiency, we introduce an Hamilton-Jacobi (HJ) reachability-based neighbour selection strategy to prioritize safety-critical interactions, and a receding-horizon policy execution scheme that adapts to dynamic interactions while reducing computational burden. Experiments on multi-agent navigation tasks demonstrate that MAD-PINN achieves superior safety-performance trade-offs, maintains scalability as the number of agents grows, and consistently outperforms state-of-the-art baselines.
>
---
#### [new 010] Path Diffuser: Diffusion Model for Data-Driven Traffic Simulator
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于交通场景模拟任务，旨在解决传统方法依赖历史数据、生成场景有限的问题。提出Path Diffuser模型，无需历史轨迹即可生成多样化且符合道路规则的交通场景。**

- **链接: [http://arxiv.org/pdf/2509.24995v1](http://arxiv.org/pdf/2509.24995v1)**

> **作者:** Da Saem Lee; Akash Karthikeyan; Yash Vardhan Pant; Sebastian Fischmeister
>
> **摘要:** Simulating diverse and realistic traffic scenarios is critical for developing and testing autonomous planning. Traditional rule-based planners lack diversity and realism, while learning-based simulators often replay, forecast, or edit scenarios using historical agent trajectories. However, they struggle to generate new scenarios, limiting scalability and diversity due to their reliance on fully annotated logs and historical data. Thus, a key challenge for a learning-based simulator's performance is that it requires agents' past trajectories and pose information in addition to map data, which might not be available for all agents on the road.Without which, generated scenarios often produce unrealistic trajectories that deviate from drivable areas, particularly under out-of-distribution (OOD) map scenes (e.g., curved roads). To address this, we propose Path Diffuser (PD): a two-stage, diffusion model for generating agent pose initializations and their corresponding trajectories conditioned on the map, free of any historical context of agents' trajectories. Furthermore, PD incorporates a motion primitive-based prior, leveraging Frenet frame candidate trajectories to enhance diversity while ensuring road-compliant trajectory generation. We also explore various design choices for modeling complex multi-agent interactions. We demonstrate the effectiveness of our method through extensive experiments on the Argoverse2 Dataset and additionally evaluate the generalizability of the approach on OOD map variants. Notably, Path Diffuser outperforms the baseline methods by 1.92x on distribution metrics, 1.14x on common-sense metrics, and 1.62x on road compliance from adversarial benchmarks.
>
---
#### [new 011] Mobile Robot Localization via Indoor Positioning System and Odometry Fusion
- **分类: cs.RO**

- **简介: 该论文属于移动机器人定位任务，解决室内定位精度问题，通过融合超声波定位系统与轮式里程计数据，提升定位准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.22693v1](http://arxiv.org/pdf/2509.22693v1)**

> **作者:** Muhammad Hafil Nugraha; Fauzi Abdul; Lastiko Bramantyo; Estiko Rijanto; Roni Permana Saputra; Oka Mahendra
>
> **备注:** 6 pages, 7 figures,
>
> **摘要:** Accurate localization is crucial for effectively operating mobile robots in indoor environments. This paper presents a comprehensive approach to mobile robot localization by integrating an ultrasound-based indoor positioning system (IPS) with wheel odometry data via sensor fusion techniques. The fusion methodology leverages the strengths of both IPS and wheel odometry, compensating for the individual limitations of each method. The Extended Kalman Filter (EKF) fusion method combines the data from the IPS sensors and the robot's wheel odometry, providing a robust and reliable localization solution. Extensive experiments in a controlled indoor environment reveal that the fusion-based localization system significantly enhances accuracy and precision compared to standalone systems. The results demonstrate significant improvements in trajectory tracking, with the EKF-based approach reducing errors associated with wheel slippage and sensor noise.
>
---
#### [new 012] Simulated Annealing for Multi-Robot Ergodic Information Acquisition Using Graph-Based Discretization
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于多机器人信息采集任务，旨在解决区域不确定性不均问题。通过模拟退火算法优化采样分布，提升采集效率与一致性。**

- **链接: [http://arxiv.org/pdf/2509.23214v1](http://arxiv.org/pdf/2509.23214v1)**

> **作者:** Benjamin Wong; Aaron Weber; Mohamed M. Safwat; Santosh Devasia; Ashis G. Banerjee
>
> **摘要:** One of the goals of active information acquisition using multi-robot teams is to keep the relative uncertainty in each region at the same level to maintain identical acquisition quality (e.g., consistent target detection) in all the regions. To achieve this goal, ergodic coverage can be used to assign the number of samples according to the quality of observation, i.e., sampling noise levels. However, the noise levels are unknown to the robots. Although this noise can be estimated from samples, the estimates are unreliable at first and can generate fluctuating values. The main contribution of this paper is to use simulated annealing to generate the target sampling distribution, starting from uniform and gradually shifting to an estimated optimal distribution, by varying the coldness parameter of a Boltzmann distribution with the estimated sampling entropy as energy. Simulation results show a substantial improvement of both transient and asymptotic entropy compared to both uniform and direct-ergodic searches. Finally, a demonstration is performed with a TurtleBot swarm system to validate the physical applicability of the algorithm.
>
---
#### [new 013] GUARD: Toward a Compromise between Traditional Control and Learning for Safe Robot Systems
- **分类: cs.RO**

- **简介: 该论文属于机器人安全控制任务，旨在解决传统控制与学习方法之间的平衡问题，通过融合不确定性感知与实时优化提升碰撞避免能力。**

- **链接: [http://arxiv.org/pdf/2509.23312v1](http://arxiv.org/pdf/2509.23312v1)**

> **作者:** Johannes A. Gaus; Junheon Yoon; Woo-Jeong Baek; Seungwon Choi; Suhan Park; Jaeheung Park
>
> **备注:** Submitted as workshop paper to IEEE IROS 2025
>
> **摘要:** This paper presents the framework \textbf{GUARD} (\textbf{G}uided robot control via \textbf{U}ncertainty attribution and prob\textbf{A}bilistic kernel optimization for \textbf{R}isk-aware \textbf{D}ecision making) that combines traditional control with an uncertainty-aware perception technique using active learning with real-time capability for safe robot collision avoidance. By doing so, this manuscript addresses the central challenge in robotics of finding a reasonable compromise between traditional methods and learning algorithms to foster the development of safe, yet efficient and flexible applications. By unifying a reactive model predictive countouring control (RMPCC) with an Iterative Closest Point (ICP) algorithm that enables the attribution of uncertainty sources online using active learning with real-time capability via a probabilistic kernel optimization technique, \emph{GUARD} inherently handles the existing ambiguity of the term \textit{safety} that exists in robotics literature. Experimental studies indicate the high performance of \emph{GUARD}, thereby highlighting the relevance and need to broaden its applicability in future.
>
---
#### [new 014] Finding an Initial Probe Pose in Teleoperated Robotic Echocardiography via 2D LiDAR-Based 3D Reconstruction
- **分类: cs.RO**

- **简介: 该论文属于机器人超声心动图任务，旨在解决探头初始位置估计问题。通过2D LiDAR进行3D重建，自动确定探头起始位置，提高操作效率。**

- **链接: [http://arxiv.org/pdf/2509.24867v1](http://arxiv.org/pdf/2509.24867v1)**

> **作者:** Mariadas Capsran Roshan; Edgar M Hidalgo; Mats Isaksson; Michelle Dunn; Jagannatha Charjee Pyaraka
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Echocardiography is a key imaging modality for cardiac assessment but remains highly operator-dependent, and access to trained sonographers is limited in underserved settings. Teleoperated robotic echocardiography has been proposed as a solution; however, clinical studies report longer examination times than manual procedures, increasing diagnostic delays and operator workload. Automating non-expert tasks, such as automatically moving the probe to an ideal starting pose, offers a pathway to reduce this burden. Prior vision- and depth-based approaches to estimate an initial probe pose are sensitive to lighting, texture, and anatomical variability. We propose a robot-mounted 2D LiDAR-based approach that reconstructs the chest surface in 3D and estimates the initial probe pose automatically. To the best of our knowledge, this is the first demonstration of robot-mounted 2D LiDAR used for 3D reconstruction of a human body surface. Through plane-based extrinsic calibration, the transformation between the LiDAR and robot base frames was estimated with an overall root mean square (RMS) residual of 1.8 mm and rotational uncertainty below 0.2{\deg}. The chest front surface, reconstructed from two linear LiDAR sweeps, was aligned with non-rigid templates to identify an initial probe pose. A mannequin-based study assessing reconstruction accuracy showed mean surface errors of 2.78 +/- 0.21 mm. Human trials (N=5) evaluating the proposed approach found probe initial points typically 20-30 mm from the clinically defined initial point, while the variation across repeated trials on the same subject was less than 4 mm.
>
---
#### [new 015] RAVEN: Resilient Aerial Navigation via Open-Set Semantic Memory and Behavior Adaptation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于空中语义导航任务，解决长距离户外目标搜索问题。提出RAVEN框架，结合3D记忆与行为树，提升导航鲁棒性与适应性。**

- **链接: [http://arxiv.org/pdf/2509.23563v1](http://arxiv.org/pdf/2509.23563v1)**

> **作者:** Seungchan Kim; Omar Alama; Dmytro Kurdydyk; John Keller; Nikhil Keetha; Wenshan Wang; Yonatan Bisk; Sebastian Scherer
>
> **摘要:** Aerial outdoor semantic navigation requires robots to explore large, unstructured environments to locate target objects. Recent advances in semantic navigation have demonstrated open-set object-goal navigation in indoor settings, but these methods remain limited by constrained spatial ranges and structured layouts, making them unsuitable for long-range outdoor search. While outdoor semantic navigation approaches exist, they either rely on reactive policies based on current observations, which tend to produce short-sighted behaviors, or precompute scene graphs offline for navigation, limiting adaptability to online deployment. We present RAVEN, a 3D memory-based, behavior tree framework for aerial semantic navigation in unstructured outdoor environments. It (1) uses a spatially consistent semantic voxel-ray map as persistent memory, enabling long-horizon planning and avoiding purely reactive behaviors, (2) combines short-range voxel search and long-range ray search to scale to large environments, (3) leverages a large vision-language model to suggest auxiliary cues, mitigating sparsity of outdoor targets. These components are coordinated by a behavior tree, which adaptively switches behaviors for robust operation. We evaluate RAVEN in 10 photorealistic outdoor simulation environments over 100 semantic tasks, encompassing single-object search, multi-class, multi-instance navigation and sequential task changes. Results show RAVEN outperforms baselines by 85.25% in simulation and demonstrate its real-world applicability through deployment on an aerial robot in outdoor field tests.
>
---
#### [new 016] SONAR: Semantic-Object Navigation with Aggregated Reasoning through a Cross-Modal Inference Paradigm
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言导航任务，旨在解决机器人在未知环境中理解指令并导航的问题。提出SONAR方法，结合语义地图与视觉-语言模型，提升导航鲁棒性与适应性。**

- **链接: [http://arxiv.org/pdf/2509.24321v1](http://arxiv.org/pdf/2509.24321v1)**

> **作者:** Yao Wang; Zhirui Sun; Wenzheng Chi; Baozhi Jia; Wenjun Xu; Jiankun Wang
>
> **摘要:** Understanding human instructions and accomplishing Vision-Language Navigation tasks in unknown environments is essential for robots. However, existing modular approaches heavily rely on the quality of training data and often exhibit poor generalization. Vision-Language Model based methods, while demonstrating strong generalization capabilities, tend to perform unsatisfactorily when semantic cues are weak. To address these issues, this paper proposes SONAR, an aggregated reasoning approach through a cross modal paradigm. The proposed method integrates a semantic map based target prediction module with a Vision-Language Model based value map module, enabling more robust navigation in unknown environments with varying levels of semantic cues, and effectively balancing generalization ability with scene adaptability. In terms of target localization, we propose a strategy that integrates multi-scale semantic maps with confidence maps, aiming to mitigate false detections of target objects. We conducted an evaluation of the SONAR within the Gazebo simulator, leveraging the most challenging Matterport 3D (MP3D) dataset as the experimental benchmark. Experimental results demonstrate that SONAR achieves a success rate of 38.4% and an SPL of 17.7%.
>
---
#### [new 017] AdaNav: Adaptive Reasoning with Uncertainty for Vision-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决长期序列中推理不准确和计算冗余问题。提出AdaNav框架，通过动态触发推理提升导航性能。**

- **链接: [http://arxiv.org/pdf/2509.24387v1](http://arxiv.org/pdf/2509.24387v1)**

> **作者:** Xin Ding; Jianyu Wei; Yifan Yang; Shiqi Jiang; Qianxi Zhang; Hao Wu; Fucheng Jia; Liang Mi; Yuxuan Yan; Weijun Wang; Yunxin Liu; Zhibo Chen; Ting Cao
>
> **摘要:** Vision Language Navigation (VLN) requires agents to follow natural language instructions by grounding them in sequential visual observations over long horizons. Explicit reasoning could enhance temporal consistency and perception action alignment, but reasoning at fixed steps often leads to suboptimal performance and unnecessary computation. To address this, we propose AdaNav, an uncertainty-based adaptive reasoning framework for VLN. At its core is the Uncertainty Adaptive Reasoning Block (UAR), a lightweight plugin that dynamically triggers reasoning. We introduce Action Entropy as a policy prior for UAR and progressively refine it through a Heuristics to RL training method, enabling agents to learn difficulty aware reasoning policies under the strict data limitations of embodied tasks. Results show that with only 6K training samples, AdaNav achieves substantial gains over closed source models trained on million scale data, improving success rate by 20% on R2R val-unseen, 11.7% on RxR-CE, and 11.4% in real world scenes. The code is available at https://github.com/xinding-sys/AdaNav.
>
---
#### [new 018] FTACT: Force Torque aware Action Chunking Transformer for Pick-and-Reorient Bottle Task
- **分类: cs.RO**

- **简介: 该论文针对零售环境中的抓取与翻转瓶类任务，解决视觉不足时的接触感知问题，提出融合力扭矩信息的模仿学习方法，提升操作成功率。**

- **链接: [http://arxiv.org/pdf/2509.23112v1](http://arxiv.org/pdf/2509.23112v1)**

> **作者:** Ryo Watanabe; Maxime Alvarez; Pablo Ferreiro; Pavel Savkin; Genki Sano
>
> **摘要:** Manipulator robots are increasingly being deployed in retail environments, yet contact rich edge cases still trigger costly human teleoperation. A prominent example is upright lying beverage bottles, where purely visual cues are often insufficient to resolve subtle contact events required for precise manipulation. We present a multimodal Imitation Learning policy that augments the Action Chunking Transformer with force and torque sensing, enabling end-to-end learning over images, joint states, and forces and torques. Deployed on Ghost, single-arm platform by Telexistence Inc, our approach improves Pick-and-Reorient bottle task by detecting and exploiting contact transitions during pressing and placement. Hardware experiments demonstrate greater task success compared to baseline matching the observation space of ACT as an ablation and experiments indicate that force and torque signals are beneficial in the press and place phases where visual observability is limited, supporting the use of interaction forces as a complementary modality for contact rich skills. The results suggest a practical path to scaling retail manipulation by combining modern imitation learning architectures with lightweight force and torque sensing.
>
---
#### [new 019] Prompting Robot Teams with Natural Language
- **分类: cs.RO; cs.LG; cs.MA**

- **简介: 该论文属于多机器人协作任务，旨在通过自然语言提示实现机器人团队的高效协同。工作包括构建基于DFA和RNN的任务表示框架，并训练GNN控制策略以执行任务。**

- **链接: [http://arxiv.org/pdf/2509.24575v1](http://arxiv.org/pdf/2509.24575v1)**

> **作者:** Nicolas Pfitzer; Eduardo Sebastián; Ajay Shankar; Amanda Prorok
>
> **摘要:** This paper presents a framework towards prompting multi-robot teams with high-level tasks using natural language expressions. Our objective is to use the reasoning capabilities demonstrated by recent language models in understanding and decomposing human expressions of intent, and repurpose these for multi-robot collaboration and decision-making. The key challenge is that an individual's behavior in a collective can be hard to specify and interpret, and must continuously adapt to actions from others. This necessitates a framework that possesses the representational capacity required by the logic and semantics of a task, and yet supports decentralized and interactive real-time operation. We solve this dilemma by recognizing that a task can be represented as a deterministic finite automaton (DFA), and that recurrent neural networks (RNNs) can encode numerous automata. This allows us to distill the logic and sequential decompositions of sub-tasks obtained from a language model into an RNN, and align its internal states with the semantics of a given task. By training a graph neural network (GNN) control policy that is conditioned on the hidden states of the RNN and the language embeddings, our method enables robots to execute task-relevant actions in a decentralized manner. We present evaluations of this single light-weight interpretable model on various simulated and real-world multi-robot tasks that require sequential and collaborative behavior by the team -- sites.google.com/view/prompting-teams.
>
---
#### [new 020] Distributed Multi-Robot Multi-Target Simultaneous Search and Tracking in an Unknown Non-convex Environment
- **分类: cs.RO**

- **简介: 该论文属于多机器人协同搜索与跟踪任务，旨在解决未知非凸环境中同时探索、搜索和跟踪目标的问题。通过整合三种控制策略，实现高效覆盖与精准跟踪。**

- **链接: [http://arxiv.org/pdf/2509.23308v1](http://arxiv.org/pdf/2509.23308v1)**

> **作者:** Jun Chen; Jiaqing Ma; Philip Dames
>
> **摘要:** In unknown non-convex environments, such as indoor and underground spaces, deploying a fleet of robots to explore the surroundings while simultaneously searching for and tracking targets of interest to maintain high-precision data collection represents a fundamental challenge that urgently requires resolution in applications such as environmental monitoring and rescue operations. Current research has made significant progress in addressing environmental exploration, information search, and target tracking problems, but has yet to establish a framework for simultaneously optimizing these tasks in complex environments. In this paper, we propose a novel motion planning algorithm framework that integrates three control strategies: a frontier-based exploration strategy, a guaranteed coverage strategy based on Lloyd's algorithm, and a sensor-based multi-target tracking strategy. By incorporating these three strategies, the proposed algorithm balances coverage search and high-precision active tracking during exploration. Our approach is validated through a series of MATLAB simulations, demonstrating validity and superiority over standard approaches.
>
---
#### [new 021] GES-UniGrasp: A Two-Stage Dexterous Grasping Strategy With Geometry-Based Expert Selection
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决通用物体自然抓取问题。通过构建数据集并采用几何聚类方法，提出两阶段专家选择框架，提升抓取适应性和成功率。**

- **链接: [http://arxiv.org/pdf/2509.23567v1](http://arxiv.org/pdf/2509.23567v1)**

> **作者:** Fangting Xu; Jilin Zhu; Xiaoming Gu; Jianzhong Tang
>
> **摘要:** Robust and human-like dexterous grasping of general objects is a critical capability for advancing intelligent robotic manipulation in real-world scenarios. However, existing reinforcement learning methods guided by grasp priors often result in unnatural behaviors. In this work, we present \textit{ContactGrasp}, a robotic dexterous pre-grasp and grasp dataset that explicitly accounts for task-relevant wrist orientation and thumb-index pinching coordination. The dataset covers 773 objects in 82 categories, providing a rich foundation for training human-like grasp strategies. Building upon this dataset, we perform geometry-based clustering to group objects by shape, enabling a two-stage Geometry-based Expert Selection (GES) framework that selects among specialized experts for grasping diverse object geometries, thereby enhancing adaptability to diverse shapes and generalization across categories. Our approach demonstrates natural grasp postures and achieves high success rates of 99.4\% and 96.3\% on the train and test sets, respectively, showcasing strong generalization and high-quality grasp execution.
>
---
#### [new 022] ARMimic: Learning Robotic Manipulation from Passive Human Demonstrations in Augmented Reality
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决传统示教方法 cumbersome 的问题。提出ARMimic框架，通过AR和摄像头实现轻量级数据收集与模仿学习。**

- **链接: [http://arxiv.org/pdf/2509.22914v1](http://arxiv.org/pdf/2509.22914v1)**

> **作者:** Rohan Walia; Yusheng Wang; Ralf Römer; Masahiro Nishio; Angela P. Schoellig; Jun Ota
>
> **摘要:** Imitation learning is a powerful paradigm for robot skill acquisition, yet conventional demonstration methods--such as kinesthetic teaching and teleoperation--are cumbersome, hardware-heavy, and disruptive to workflows. Recently, passive observation using extended reality (XR) headsets has shown promise for egocentric demonstration collection, yet current approaches require additional hardware, complex calibration, or constrained recording conditions that limit scalability and usability. We present ARMimic, a novel framework that overcomes these limitations with a lightweight and hardware-minimal setup for scalable, robot-free data collection using only a consumer XR headset and a stationary workplace camera. ARMimic integrates egocentric hand tracking, augmented reality (AR) robot overlays, and real-time depth sensing to ensure collision-aware, kinematically feasible demonstrations. A unified imitation learning pipeline is at the core of our method, treating both human and virtual robot trajectories as interchangeable, which enables policies that generalize across different embodiments and environments. We validate ARMimic on two manipulation tasks, including challenging long-horizon bowl stacking. In our experiments, ARMimic reduces demonstration time by 50% compared to teleoperation and improves task success by 11% over ACT, a state-of-the-art baseline trained on teleoperated data. Our results demonstrate that ARMimic enables safe, seamless, and in-the-wild data collection, offering great potential for scalable robot learning in diverse real-world settings.
>
---
#### [new 023] DexFlyWheel: A Scalable and Self-improving Data Generation Framework for Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决数据稀缺与多样性不足的问题。提出DexFlyWheel框架，通过自增强循环生成高质量、多样化的操作数据。**

- **链接: [http://arxiv.org/pdf/2509.23829v1](http://arxiv.org/pdf/2509.23829v1)**

> **作者:** Kefei Zhu; Fengshuo Bai; YuanHao Xiang; Yishuai Cai; Xinglin Chen; Ruochong Li; Xingtao Wang; Hao Dong; Yaodong Yang; Xiaopeng Fan; Yuanpei Chen
>
> **备注:** NeurIPS 2025, Spotlight
>
> **摘要:** Dexterous manipulation is critical for advancing robot capabilities in real-world applications, yet diverse and high-quality datasets remain scarce. Existing data collection methods either rely on human teleoperation or require significant human engineering, or generate data with limited diversity, which restricts their scalability and generalization. In this paper, we introduce DexFlyWheel, a scalable data generation framework that employs a self-improving cycle to continuously enrich data diversity. Starting from efficient seed demonstrations warmup, DexFlyWheel expands the dataset through iterative cycles. Each cycle follows a closed-loop pipeline that integrates Imitation Learning (IL), residual Reinforcement Learning (RL), rollout trajectory collection, and data augmentation. Specifically, IL extracts human-like behaviors from demonstrations, and residual RL enhances policy generalization. The learned policy is then used to generate trajectories in simulation, which are further augmented across diverse environments and spatial configurations before being fed back into the next cycle. Over successive iterations, a self-improving data flywheel effect emerges, producing datasets that cover diverse scenarios and thereby scaling policy performance. Experimental results demonstrate that DexFlyWheel generates over 2,000 diverse demonstrations across four challenging tasks. Policies trained on our dataset achieve an average success rate of 81.9\% on the challenge test sets and successfully transfer to the real world through digital twin, achieving a 78.3\% success rate on dual-arm lift tasks.
>
---
#### [new 024] Control Your Robot: A Unified System for Robot Control and Policy Deployment
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决跨平台控制碎片化问题。提出统一框架，实现数据收集与策略部署的标准化和模块化。**

- **链接: [http://arxiv.org/pdf/2509.23823v1](http://arxiv.org/pdf/2509.23823v1)**

> **作者:** Tian Nian; Weijie Ke; Yao Mu; Tianxing Chen; Shaolong Zhu; Bingshan Hu
>
> **备注:** Code: https://github.com/Tian-Nian/control_your_robot
>
> **摘要:** Cross-platform robot control remains difficult because hardware interfaces, data formats, and control paradigms vary widely, which fragments toolchains and slows deployment. To address this, we present Control Your Robot, a modular, general-purpose framework that unifies data collection and policy deployment across diverse platforms. The system reduces fragmentation through a standardized workflow with modular design, unified APIs, and a closed-loop architecture. It supports flexible robot registration, dual-mode control with teleoperation and trajectory playback, and seamless integration from multimodal data acquisition to inference. Experiments on single-arm and dual-arm systems show efficient, low-latency data collection and effective support for policy learning with imitation learning and vision-language-action models. Policies trained on data gathered by Control Your Robot match expert demonstrations closely, indicating that the framework enables scalable and reproducible robot learning across platforms.
>
---
#### [new 025] RAISE: A Robot-Assisted Selective Disassembly and Sorting System for End-of-Life Phones
- **分类: cs.RO**

- **简介: 该论文属于EoL手机拆解任务，解决人工拆解效率低、成本高的问题。提出一种自动化拆解与分拣系统，提升处理效率和经济效益。**

- **链接: [http://arxiv.org/pdf/2509.23048v1](http://arxiv.org/pdf/2509.23048v1)**

> **作者:** Chang Liu; Badrinath Balasubramaniam; Neal Yancey; Michael Severson; Adam Shine; Philip Bove; Beiwen Li; Xiao Liang; Minghui Zheng
>
> **摘要:** End-of-Life (EoL) phones significantly exacerbate global e-waste challenges due to their high production volumes and short lifecycles. Disassembly is among the most critical processes in EoL phone recycling. However, it relies heavily on human labor due to product variability. Consequently, the manual process is both labor-intensive and time-consuming. In this paper, we propose a low-cost, easily deployable automated and selective disassembly and sorting system for EoL phones, consisting of three subsystems: an adaptive cutting system, a vision-based robotic sorting system, and a battery removal system. The system can process over 120 phones per hour with an average disassembly success rate of 98.9%, efficiently delivering selected high-value components to downstream processing. It provides a reliable and scalable automated solution to the pressing challenge of EoL phone disassembly. Additionally, the automated system can enhance disassembly economics, converting a previously unprofitable process into one that yields a net profit per unit weight of EoL phones.
>
---
#### [new 026] In-Hand Manipulation of Articulated Tools with Dexterous Robot Hands with Sim-to-Real Transfer
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决关节机构在真实环境中的操控问题。通过强化学习与仿真到现实的迁移，结合传感器反馈提升操控稳定性与适应性。**

- **链接: [http://arxiv.org/pdf/2509.23075v1](http://arxiv.org/pdf/2509.23075v1)**

> **作者:** Soofiyan Atar; Daniel Huang; Florian Richter; Michael Yip
>
> **摘要:** Reinforcement learning (RL) and sim-to-real transfer have advanced robotic manipulation of rigid objects. Yet, policies remain brittle when applied to articulated mechanisms due to contact-rich dynamics and under-modeled joint phenomena such as friction, stiction, backlash, and clearances. We address this challenge through dexterous in-hand manipulation of articulated tools using a robotic hand with reduced articulation and kinematic redundancy relative to the human hand. Our controller augments a simulation-trained base policy with a sensor-driven refinement learned from hardware demonstrations, conditioning on proprioception and target articulation states while fusing whole-hand tactile and force feedback with the policy's internal action intent via cross-attention-based integration. This design enables online adaptation to instance-specific articulation properties, stabilizes contact interactions, regulates internal forces, and coordinates coupled-link motion under perturbations. We validate our approach across a diversity of real-world examples, including scissors, pliers, minimally invasive surgical tools, and staplers. We achieve robust transfer from simulation to hardware, improved disturbance resilience, and generalization to previously unseen articulated tools, thereby reducing reliance on precise physical modeling in contact-rich settings.
>
---
#### [new 027] Towards Developing Standards and Guidelines for Robot Grasping and Manipulation Pipelines in the COMPARE Ecosystem
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取与操作任务，旨在解决组件模块化与管道集成问题，通过构建标准、分析现有管道和开发新管道来提升兼容性与效率。**

- **链接: [http://arxiv.org/pdf/2509.22801v1](http://arxiv.org/pdf/2509.22801v1)**

> **作者:** Huajing Zhao; Brian Flynn; Adam Norton; Holly Yanco
>
> **备注:** The 3rd AAAI Fall Symposium on Unifying Representations for Robot Application Development (UR-RAD), Arlington, VA, USA, November 2025
>
> **摘要:** The COMPARE Ecosystem aims to improve the compatibility and benchmarking of open-source products for robot manipulation through a series of activities. One such activity is the development of standards and guidelines to specify modularization practices at the component-level for individual modules (e.g., perception, grasp planning, motion planning) and integrations of components that form robot manipulation capabilities at the pipeline-level. This paper briefly reviews our work-in-progress to date to (1) build repositories of open-source products to identify common characteristics of each component in the pipeline, (2) investigate existing modular pipelines to glean best practices, and (3) develop new modular pipelines that advance prior work while abiding by the proposed standards and guidelines.
>
---
#### [new 028] AgriCruiser: An Open Source Agriculture Robot for Over-the-row Navigation
- **分类: cs.RO**

- **简介: 该论文属于农业机器人任务，旨在解决低成本、高效 Weed 管理问题。设计并测试了 AgriCruiser 机器人，实现精准喷洒与灵活移动。**

- **链接: [http://arxiv.org/pdf/2509.25056v1](http://arxiv.org/pdf/2509.25056v1)**

> **作者:** Kenny Truong; Yongkyu Lee; Jason Irie; Shivam Kumar Panda; Shahab Ahmad; Md. Mukhlesur Rahman; M. Khalid Jawed
>
> **备注:** GitHub: https://github.com/structuresComp/agri-cruiser
>
> **摘要:** We present the AgriCruiser, an open-source over-the-row agricultural robot developed for low-cost deployment and rapid adaptation across diverse crops and row layouts. The chassis provides an adjustable track width of 1.42 m to 1.57 m, along with a ground clearance of 0.94 m. The AgriCruiser achieves compact pivot turns with radii of 0.71 m to 0.79 m, enabling efficient headland maneuvers. The platform is designed for the integration of the other subsystems, and in this study, a precision spraying system was implemented to assess its effectiveness in weed management. In twelve flax plots, a single robotic spray pass reduced total weed populations (pigweed and Venice mallow) by 24- to 42-fold compared to manual weeding in four flax plots, while also causing less crop damage. Mobility experiments conducted on concrete, asphalt, gravel, grass, and both wet and dry soil confirmed reliable traversal consistent with torque sizing. The complete chassis can be constructed from commodity T-slot extrusion with minimal machining, resulting in a bill of materials costing approximately $5,000 - $6,000, which enables replication and customization. The mentioned results demonstrate that low-cost, reconfigurable over-the-row robots can achieve effective weed management with reduced crop damage and labor requirements, while providing a versatile foundation for phenotyping, sensing, and other agriculture applications. Design files and implementation details are released to accelerate research and adoption of modular agricultural robotics.
>
---
#### [new 029] Good Weights: Proactive, Adaptive Dead Reckoning Fusion for Continuous and Robust Visual SLAM
- **分类: cs.RO**

- **简介: 该论文属于视觉SLAM任务，解决纹理缺失环境下的定位问题。通过自适应融合死区导航，提升SLAM的连续性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.22910v1](http://arxiv.org/pdf/2509.22910v1)**

> **作者:** Yanwei Du; Jing-Chen Peng; Patricio A. Vela
>
> **备注:** 8 pages, 9 figures, 1 table. Submitted to IEEE Conference
>
> **摘要:** Given that Visual SLAM relies on appearance cues for localization and scene understanding, texture-less or visually degraded environments (e.g., plain walls or low lighting) lead to poor pose estimation and track loss. However, robots are typically equipped with sensors that provide some form of dead reckoning odometry with reasonable short-time performance but unreliable long-time performance. The Good Weights (GW) algorithm described here provides a framework to adaptively integrate dead reckoning (DR) with passive visual SLAM for continuous and accurate frame-level pose estimation. Importantly, it describes how all modules in a comprehensive SLAM system must be modified to incorporate DR into its design. Adaptive weighting increases DR influence when visual tracking is unreliable and reduces when visual feature information is strong, maintaining pose track without overreliance on DR. Good Weights yields a practical solution for mobile navigation that improves visual SLAM performance and robustness. Experiments on collected datasets and in real-world deployment demonstrate the benefits of Good Weights.
>
---
#### [new 030] Ancestry Tree Clustering for Particle Filter Diversity Maintenance
- **分类: cs.RO; cs.AI; cs.LG; F.2.2; G.3; I.5.3; F.2.2; I.2.9; G.3; I.5.3**

- **简介: 该论文属于粒子滤波任务，旨在解决多模态环境下的多样性维持问题。通过祖先树聚类方法提升算法鲁棒性与估计紧凑性。**

- **链接: [http://arxiv.org/pdf/2509.24124v1](http://arxiv.org/pdf/2509.24124v1)**

> **作者:** Ilari Vallivaara; Bingnan Duan; Yinhuan Dong; Tughrul Arslan
>
> **备注:** 15th International Conference on Indoor Positioning and Indoor Navigation, 15-18 September 2025, Tampere, Finland Originally 8 pages. The online version with appendices is 14 pages
>
> **摘要:** We propose a method for linear-time diversity maintenance in particle filtering. It clusters particles based on ancestry tree topology: closely related particles in sufficiently large subtrees are grouped together. The main idea is that the tree structure implicitly encodes similarity without the need for spatial or other domain-specific metrics. This approach, when combined with intra-cluster fitness sharing and the protection of particles not included in a cluster, effectively prevents premature convergence in multimodal environments while maintaining estimate compactness. We validate our approach in a multimodal robotics simulation and a real-world multimodal indoor environment. We compare the performance to several diversity maintenance algorithms from the literature, including Deterministic Resampling and Particle Gaussian Mixtures. Our algorithm achieves high success rates with little to no negative effect on compactness, showing particular robustness to different domains and challenging initial conditions.
>
---
#### [new 031] Trajectory Prediction via Bayesian Intention Inference under Unknown Goals and Kinematics
- **分类: cs.RO**

- **简介: 该论文属于轨迹预测任务，解决未知目标意图与运动特性下的实时预测问题。提出一种自适应贝叶斯算法，同时估计意图和运动参数，提升预测准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.24928v1](http://arxiv.org/pdf/2509.24928v1)**

> **作者:** Shunan Yin; Zehui Lu; Shaoshuai Mou
>
> **摘要:** This work introduces an adaptive Bayesian algorithm for real-time trajectory prediction via intention inference, where a target's intentions and motion characteristics are unknown and subject to change. The method concurrently estimates two critical variables: the target's current intention, modeled as a Markovian latent state, and an intention parameter that describes the target's adherence to a shortest-path policy. By integrating this joint update technique, the algorithm maintains robustness against abrupt intention shifts and unknown motion dynamics. A sampling-based trajectory prediction mechanism then exploits these adaptive estimates to generate probabilistic forecasts with quantified uncertainty. We validate the framework through numerical experiments: Ablation studies of two cases, and a 500-trial Monte Carlo analysis; Hardware demonstrations on quadrotor and quadrupedal platforms. Experimental results demonstrate that the proposed approach significantly outperforms non-adaptive and partially adaptive methods. The method operates in real time around 270 Hz without requiring training or detailed prior knowledge of target behavior, showcasing its applicability in various robotic systems.
>
---
#### [new 032] CineWild: Balancing Art and Robotics for Ethical Wildlife Documentary Filmmaking
- **分类: cs.RO; cs.MM**

- **简介: 该论文属于无人机与伦理结合的野生动物纪录片任务，解决 drones 干扰动物的问题，提出 CineWild 系统实现安全拍摄。**

- **链接: [http://arxiv.org/pdf/2509.24921v1](http://arxiv.org/pdf/2509.24921v1)**

> **作者:** Pablo Pueyo; Fernando Caballero; Ana Cristina Murillo; Eduardo Montijano
>
> **摘要:** Drones, or unmanned aerial vehicles (UAVs), have become powerful tools across domains-from industry to the arts. In documentary filmmaking, they offer dynamic, otherwise unreachable perspectives, transforming how stories are told. Wildlife documentaries especially benefit, yet drones also raise ethical concerns: the risk of disturbing the animals they aim to capture. This paper introduces CineWild, an autonomous UAV framework that combines robotics, cinematography, and ethics. Built on model predictive control, CineWild dynamically adjusts flight paths and camera settings to balance cinematic quality with animal welfare. Key features include adaptive zoom for filming from acoustic and visual safe distances, path-planning that avoids an animal's field of view, and smooth, low-noise maneuvers. CineWild exemplifies interdisciplinary innovation-bridging engineering, visual storytelling, and environmental ethics. We validate the system through simulation studies and will release the code upon acceptance.
>
---
#### [new 033] Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决泛化能力不足的问题。通过引入语言对齐的3D关键点，提升策略在新指令和环境中的表现。**

- **链接: [http://arxiv.org/pdf/2509.23575v1](http://arxiv.org/pdf/2509.23575v1)**

> **作者:** Jianshu Hu; Lidi Wang; Shujia Li; Yunpeng Jiang; Xiao Li; Paul Weng; Yutong Ban
>
> **摘要:** Hierarchical coarse-to-fine policy, where a coarse branch predicts a region of interest to guide a fine-grained action predictor, has demonstrated significant potential in robotic 3D manipulation tasks by especially enhancing sample efficiency and enabling more precise manipulation. However, even augmented with pre-trained models, these hierarchical policies still suffer from generalization issues. To enhance generalization to novel instructions and environment variations, we propose Coarse-to-fine Language-Aligned manipulation Policy (CLAP), a framework that integrates three key components: 1) task decomposition, 2) VLM fine-tuning for 3D keypoint prediction, and 3) 3D-aware representation. Through comprehensive experiments in simulation and on a real robot, we demonstrate its superior generalization capability. Specifically, on GemBench, a benchmark designed for evaluating generalization, our approach achieves a 12\% higher average success rate than the SOTA method while using only 1/5 of the training trajectories. In real-world experiments, our policy, trained on only 10 demonstrations, successfully generalizes to novel instructions and environments.
>
---
#### [new 034] SSR-ZSON: Zero-Shot Object Navigation via Spatial-Semantic Relations within a Hierarchical Exploration Framework
- **分类: cs.RO**

- **简介: 该论文属于零样本目标导航任务，解决未知环境中探索效率低和局部被困问题，提出SSR-ZSON方法结合语义与空间信息提升导航性能。**

- **链接: [http://arxiv.org/pdf/2509.24763v1](http://arxiv.org/pdf/2509.24763v1)**

> **作者:** Xiangyi Meng; Delun Li; Zihao Mao; Yi Yang; Wenjie Song
>
> **摘要:** Zero-shot object navigation in unknown environments presents significant challenges, mainly due to two key limitations: insufficient semantic guidance leads to inefficient exploration, while limited spatial memory resulting from environmental structure causes entrapment in local regions. To address these issues, we propose SSR-ZSON, a spatial-semantic relative zero-shot object navigation method based on the TARE hierarchical exploration framework, integrating a viewpoint generation strategy balancing spatial coverage and semantic density with an LLM-based global guidance mechanism. The performance improvement of the proposed method is due to two key innovations. First, the viewpoint generation strategy prioritizes areas of high semantic density within traversable sub-regions to maximize spatial coverage and minimize invalid exploration. Second, coupled with an LLM-based global guidance mechanism, it assesses semantic associations to direct navigation toward high-value spaces, preventing local entrapment and ensuring efficient exploration. Deployed on hybrid Habitat-Gazebo simulations and physical platforms, SSR-ZSON achieves real-time operation and superior performance. On Matterport3D and Habitat-Matterport3D datasets, it improves the Success Rate(SR) by 18.5\% and 11.2\%, and the Success weighted by Path Length(SPL) by 0.181 and 0.140, respectively, over state-of-the-art methods.
>
---
#### [new 035] Sequence Pathfinder for Multi-Agent Pickup and Delivery in the Warehouse
- **分类: cs.RO; cs.AI; cs.LG; cs.MA**

- **简介: 该论文研究多智能体搬运任务（MAPD），解决仓库环境中路径规划效率低的问题。提出SePar模型，利用序列建模和Transformer实现高效全局决策。**

- **链接: [http://arxiv.org/pdf/2509.23778v1](http://arxiv.org/pdf/2509.23778v1)**

> **作者:** Zeyuan Zhang; Chaoran Li; Shao Zhang; Ying Wen
>
> **备注:** Preprint Under Review
>
> **摘要:** Multi-Agent Pickup and Delivery (MAPD) is a challenging extension of Multi-Agent Path Finding (MAPF), where agents are required to sequentially complete tasks with fixed-location pickup and delivery demands. Although learning-based methods have made progress in MAPD, they often perform poorly in warehouse-like environments with narrow pathways and long corridors when relying only on local observations for distributed decision-making. Communication learning can alleviate the lack of global information but introduce high computational complexity due to point-to-point communication. To address this challenge, we formulate MAPF as a sequence modeling problem and prove that path-finding policies under sequence modeling possess order-invariant optimality, ensuring its effectiveness in MAPD. Building on this, we propose the Sequential Pathfinder (SePar), which leverages the Transformer paradigm to achieve implicit information exchange, reducing decision-making complexity from exponential to linear while maintaining efficiency and global awareness. Experiments demonstrate that SePar consistently outperforms existing learning-based methods across various MAPF tasks and their variants, and generalizes well to unseen environments. Furthermore, we highlight the necessity of integrating imitation learning in complex maps like warehouses.
>
---
#### [new 036] High Torque Density PCB Axial Flux Permanent Magnet Motor for Micro Robots
- **分类: cs.RO**

- **简介: 该论文属于微机电系统任务，旨在解决微型机器人高扭矩密度电机设计问题。通过PCB绕组技术提升铜填充率，实现薄型高扭矩电机。**

- **链接: [http://arxiv.org/pdf/2509.23561v1](http://arxiv.org/pdf/2509.23561v1)**

> **作者:** Jianren Wang; Jie Han; Abhinav Gupta; Deepak Pathak; Yang Zhang
>
> **摘要:** Quasi-direct-drive (QDD) actuation is transforming legged and manipulator robots by eliminating high-ratio gearboxes, yet it demands motors that deliver very high torque at low speed within a thin, disc-shaped joint envelope. Axial-flux permanent-magnet (AFPM) machines meet these geometric and torque requirements, but scaling them below a 20mm outer diameter is hampered by poor copper fill in conventional wound stators, inflating resistance and throttling continuous torque. This paper introduces a micro-scale AFPM motor that overcomes these limitations through printed-circuit-board (PCB) windings fabricated with advanced IC-substrate high-density interconnect (HDI) technology. The resulting 48-layer stator-formed by stacking four 12-layer HDI modules-achieves a record 45\% copper fill in a package only 5mm thick and 19mm in diameter. We perform comprehensive electromagnetic and thermal analyses to inform the motor design, then fabricate a prototype whose performance characteristics are experimentally verified.
>
---
#### [new 037] Persistent Autoregressive Mapping with Traffic Rules for Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，解决持久交通规则与地图融合问题。提出PAMR框架，实现车道与规则的联合构建与持续一致性维护。**

- **链接: [http://arxiv.org/pdf/2509.22756v1](http://arxiv.org/pdf/2509.22756v1)**

> **作者:** Shiyi Liang; Xinyuan Chang; Changjie Wu; Huiyuan Yan; Yifan Bai; Xinran Liu; Hang Zhang; Yujian Yuan; Shuang Zeng; Mu Xu; Xing Wei
>
> **摘要:** Safe autonomous driving requires both accurate HD map construction and persistent awareness of traffic rules, even when their associated signs are no longer visible. However, existing methods either focus solely on geometric elements or treat rules as temporary classifications, failing to capture their persistent effectiveness across extended driving sequences. In this paper, we present PAMR (Persistent Autoregressive Mapping with Traffic Rules), a novel framework that performs autoregressive co-construction of lane vectors and traffic rules from visual observations. Our approach introduces two key mechanisms: Map-Rule Co-Construction for processing driving scenes in temporal segments, and Map-Rule Cache for maintaining rule consistency across these segments. To properly evaluate continuous and consistent map generation, we develop MapDRv2, featuring improved lane geometry annotations. Extensive experiments demonstrate that PAMR achieves superior performance in joint vector-rule mapping tasks, while maintaining persistent rule effectiveness throughout extended driving sequences.
>
---
#### [new 038] Liaohe-CobotMagic-PnP: an Imitation Learning Dataset of Intelligent Robot for Industrial Applications
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于工业机器人领域，旨在解决动态环境下的感知与控制问题。通过构建多模态干扰数据集，提升机器人在复杂环境中的稳定性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.23111v1](http://arxiv.org/pdf/2509.23111v1)**

> **作者:** Chen Yizhe; Wang Qi; Hu Dongxiao; Jingzhe Fang; Liu Sichao; Zixin An; Hongliang Niu; Haoran Liu; Li Dong; Chuanfen Feng; Lan Dapeng; Liu Yu; Zhibo Pang
>
> **备注:** Accepted to IAI 2025 (International Conference on Industrial Artificial Intelligence), Shenyang, China, Aug 21 - 24, 2025. Preprint (before IEEE copyright transfer)
>
> **摘要:** In Industry 4.0 applications, dynamic environmental interference induces highly nonlinear and strongly coupled interactions between the environmental state and robotic behavior. Effectively representing dynamic environmental states through multimodal sensor data fusion remains a critical challenge in current robotic datasets. To address this, an industrial-grade multimodal interference dataset is presented, designed for robotic perception and control under complex conditions. The dataset integrates multi-dimensional interference features including size, color, and lighting variations, and employs high-precision sensors to synchronously collect visual, torque, and joint-state measurements. Scenarios with geometric similarity exceeding 85\% and standardized lighting gradients are included to ensure real-world representativeness. Microsecond-level time-synchronization and vibration-resistant data acquisition protocols, implemented via the Robot Operating System (ROS), guarantee temporal and operational fidelity. Experimental results demonstrate that the dataset enhances model validation robustness and improves robotic operational stability in dynamic, interference-rich environments. The dataset is publicly available at:https://modelscope.cn/datasets/Liaoh_LAB/Liaohe-CobotMagic-PnP.
>
---
#### [new 039] Unlocking the Potential of Soft Actor-Critic for Imitation Learning
- **分类: cs.RO**

- **简介: 该论文属于模仿学习任务，旨在解决传统方法样本效率低和泛化能力差的问题。工作是将AMP与SAC结合，提升数据效率和任务执行效果。**

- **链接: [http://arxiv.org/pdf/2509.24539v1](http://arxiv.org/pdf/2509.24539v1)**

> **作者:** Nayari Marie Lessa; Melya Boukheddimi; Frank Kirchner
>
> **摘要:** Learning-based methods have enabled robots to acquire bio-inspired movements with increasing levels of naturalness and adaptability. Among these, Imitation Learning (IL) has proven effective in transferring complex motion patterns from animals to robotic systems. However, current state-of-the-art frameworks predominantly rely on Proximal Policy Optimization (PPO), an on-policy algorithm that prioritizes stability over sample efficiency and policy generalization. This paper proposes a novel IL framework that combines Adversarial Motion Priors (AMP) with the off-policy Soft Actor-Critic (SAC) algorithm to overcome these limitations. This integration leverages replay-driven learning and entropy-regularized exploration, enabling naturalistic behavior and task execution, improving data efficiency and robustness. We evaluate the proposed approach (AMP+SAC) on quadruped gaits involving multiple reference motions and diverse terrains. Experimental results demonstrate that the proposed framework not only maintains stable task execution but also achieves higher imitation rewards compared to the widely used AMP+PPO method. These findings highlight the potential of an off-policy IL formulation for advancing motion generation in robotics.
>
---
#### [new 040] Teleoperator-Aware and Safety-Critical Adaptive Nonlinear MPC for Shared Autonomy in Obstacle Avoidance of Legged Robots
- **分类: cs.RO; math.OC**

- **简介: 该论文属于机器人共享自主任务，解决人机协作中的安全避障问题。提出一种自适应非线性MPC框架，实现安全高效的遥控操作。**

- **链接: [http://arxiv.org/pdf/2509.22815v1](http://arxiv.org/pdf/2509.22815v1)**

> **作者:** Ruturaj Sambhus; Muneeb Ahmad; Basit Muhammad Imran; Sujith Vijayan; Dylan P. Losey; Kaveh Akbari Hamed
>
> **摘要:** Ensuring safe and effective collaboration between humans and autonomous legged robots is a fundamental challenge in shared autonomy, particularly for teleoperated systems navigating cluttered environments. Conventional shared-control approaches often rely on fixed blending strategies that fail to capture the dynamics of legged locomotion and may compromise safety. This paper presents a teleoperator-aware, safety-critical, adaptive nonlinear model predictive control (ANMPC) framework for shared autonomy of quadrupedal robots in obstacle-avoidance tasks. The framework employs a fixed arbitration weight between human and robot actions but enhances this scheme by modeling the human input with a noisily rational Boltzmann model, whose parameters are adapted online using a projected gradient descent (PGD) law from observed joystick commands. Safety is enforced through control barrier function (CBF) constraints integrated into a computationally efficient NMPC, ensuring forward invariance of safe sets despite uncertainty in human behavior. The control architecture is hierarchical: a high-level CBF-based ANMPC (10 Hz) generates blended human-robot velocity references, a mid-level dynamics-aware NMPC (60 Hz) enforces reduced-order single rigid body (SRB) dynamics to track these references, and a low-level nonlinear whole-body controller (500 Hz) imposes the full-order dynamics via quadratic programming to track the mid-level trajectories. Extensive numerical and hardware experiments, together with a user study, on a Unitree Go2 quadrupedal robot validate the framework, demonstrating real-time obstacle avoidance, online learning of human intent parameters, and safe teleoperator collaboration.
>
---
#### [new 041] Self-driving cars: Are we there yet?
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在评估运动规划算法。通过对比CARLA、nuPlan和Waymo数据集，分析现有方法的优缺点，推动领域发展。**

- **链接: [http://arxiv.org/pdf/2509.22754v1](http://arxiv.org/pdf/2509.22754v1)**

> **作者:** Merve Atasever; Zhuochen Liu; Qingpei Li; Akshay Hitendra Shah; Hans Walker; Jyotirmoy V. Deshmukh; Rahul Jain
>
> **摘要:** Autonomous driving remains a highly active research domain that seeks to enable vehicles to perceive dynamic environments, predict the future trajectories of traffic agents such as vehicles, pedestrians, and cyclists and plan safe and efficient future motions. To advance the field, several competitive platforms and benchmarks have been established to provide standardized datasets and evaluation protocols. Among these, leaderboards by the CARLA organization and nuPlan and the Waymo Open Dataset have become leading benchmarks for assessing motion planning algorithms. Each offers a unique dataset and challenging planning problems spanning a wide range of driving scenarios and conditions. In this study, we present a comprehensive comparative analysis of the motion planning methods featured on these three leaderboards. To ensure a fair and unified evaluation, we adopt CARLA leaderboard v2.0 as our common evaluation platform and modify the selected models for compatibility. By highlighting the strengths and weaknesses of current approaches, we identify prevailing trends, common challenges, and suggest potential directions for advancing motion planning research.
>
---
#### [new 042] ViReSkill: Vision-Grounded Replanning with Skill Memory for LLM-Based Planning in Lifelong Robot Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人持续学习任务，解决LLM在运动规划中缺乏场景感知和执行不稳定的问题。提出ViReSkill框架，结合视觉重规划与技能记忆，提升任务成功率和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.24219v1](http://arxiv.org/pdf/2509.24219v1)**

> **作者:** Tomoyuki Kagaya; Subramanian Lakshmi; Anbang Ye; Thong Jing Yuan; Jayashree Karlekar; Sugiri Pranata; Natsuki Murakami; Akira Kinose; Yang You
>
> **摘要:** Robots trained via Reinforcement Learning (RL) or Imitation Learning (IL) often adapt slowly to new tasks, whereas recent Large Language Models (LLMs) and Vision-Language Models (VLMs) promise knowledge-rich planning from minimal data. Deploying LLMs/VLMs for motion planning, however, faces two key obstacles: (i) symbolic plans are rarely grounded in scene geometry and object physics, and (ii) model outputs can vary for identical prompts, undermining execution reliability. We propose ViReSkill, a framework that pairs vision-grounded replanning with a skill memory for accumulation and reuse. When a failure occurs, the replanner generates a new action sequence conditioned on the current scene, tailored to the observed state. On success, the executed plan is stored as a reusable skill and replayed in future encounters without additional calls to LLMs/VLMs. This feedback loop enables autonomous continual learning: each attempt immediately expands the skill set and stabilizes subsequent executions. We evaluate ViReSkill on simulators such as LIBERO and RLBench as well as on a physical robot. Across all settings, it consistently outperforms conventional baselines in task success rate, demonstrating robust sim-to-real generalization.
>
---
#### [new 043] KiVi: Kinesthetic-Visuospatial Integration for Dynamic and Safe Egocentric Legged Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人动态行走任务，旨在解决视觉感知不稳定问题。通过融合本体感觉与视觉信息，提出KiVi框架，提升机器人在复杂环境中的行走稳定性与适应性。**

- **链接: [http://arxiv.org/pdf/2509.23650v1](http://arxiv.org/pdf/2509.23650v1)**

> **作者:** Peizhuo Li; Hongyi Li; Yuxuan Ma; Linnan Chang; Xinrong Yang; Ruiqi Yu; Yifeng Zhang; Yuhong Cao; Qiuguo Zhu; Guillaume Sartoretti
>
> **摘要:** Vision-based locomotion has shown great promise in enabling legged robots to perceive and adapt to complex environments. However, visual information is inherently fragile, being vulnerable to occlusions, reflections, and lighting changes, which often cause instability in locomotion. Inspired by animal sensorimotor integration, we propose KiVi, a Kinesthetic-Visuospatial integration framework, where kinesthetics encodes proprioceptive sensing of body motion and visuospatial reasoning captures visual perception of surrounding terrain. Specifically, KiVi separates these pathways, leveraging proprioception as a stable backbone while selectively incorporating vision for terrain awareness and obstacle avoidance. This modality-balanced, yet integrative design, combined with memory-enhanced attention, allows the robot to robustly interpret visual cues while maintaining fallback stability through proprioception. Extensive experiments show that our method enables quadruped robots to stably traverse diverse terrains and operate reliably in unstructured outdoor environments, remaining robust to out-of-distribution (OOD) visual noise and occlusion unseen during training, thereby highlighting its effectiveness and applicability to real-world legged locomotion.
>
---
#### [new 044] UniPrototype: Humn-Robot Skill Learning with Uniform Prototypes
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，解决数据稀缺问题，通过共享运动基元实现人类到机器人的技能迁移，提升学习效率与任务表现。**

- **链接: [http://arxiv.org/pdf/2509.23021v1](http://arxiv.org/pdf/2509.23021v1)**

> **作者:** Xiao Hu; Qi Yin; Yangming Shi; Yang Ye
>
> **摘要:** Data scarcity remains a fundamental challenge in robot learning. While human demonstrations benefit from abundant motion capture data and vast internet resources, robotic manipulation suffers from limited training examples. To bridge this gap between human and robot manipulation capabilities, we propose UniPrototype, a novel framework that enables effective knowledge transfer from human to robot domains via shared motion primitives. ur approach makes three key contributions: (1) We introduce a compositional prototype discovery mechanism with soft assignments, enabling multiple primitives to co-activate and thus capture blended and hierarchical skills; (2) We propose an adaptive prototype selection strategy that automatically adjusts the number of prototypes to match task complexity, ensuring scalable and efficient representation; (3) We demonstrate the effectiveness of our method through extensive experiments in both simulation environments and real-world robotic systems. Our results show that UniPrototype successfully transfers human manipulation knowledge to robots, significantly improving learning efficiency and task performance compared to existing approaches.The code and dataset will be released upon acceptance at an anonymous repository.
>
---
#### [new 045] LocoFormer: Generalist Locomotion via Long-context Adaptation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人运动控制任务，旨在解决通用运动控制器的适应性问题。通过长上下文强化学习，使模型能控制不同形态的机器人并自适应变化。**

- **链接: [http://arxiv.org/pdf/2509.23745v1](http://arxiv.org/pdf/2509.23745v1)**

> **作者:** Min Liu; Deepak Pathak; Ananye Agarwal
>
> **备注:** Accepted to CoRL 2025
>
> **摘要:** Modern locomotion controllers are manually tuned for specific embodiments. We present LocoFormer, a generalist omni-bodied locomotion model that can control previously unseen legged and wheeled robots, even without precise knowledge of their kinematics. LocoFormer is able to adapt to changes in morphology and dynamics at test time. We find that two key choices enable adaptation. First, we train massive scale RL on procedurally generated robots with aggressive domain randomization. Second, in contrast to previous policies that are myopic with short context lengths, we extend context by orders of magnitude to span episode boundaries. We deploy the same LocoFormer to varied robots and show robust control even with large disturbances such as weight change and motor failures. In extreme scenarios, we see emergent adaptation across episodes, LocoFormer learns from falls in early episodes to improve control strategies in later ones. We believe that this simple, yet general recipe can be used to train foundation models for other robotic skills in the future. Videos at generalist-locomotion.github.io.
>
---
#### [new 046] Safe Planning in Unknown Environments using Conformalized Semantic Maps
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决未知环境中语义避障问题。通过构建动态语义地图并利用置信预测，确保在感知不确定下完成指定任务的概率。**

- **链接: [http://arxiv.org/pdf/2509.25124v1](http://arxiv.org/pdf/2509.25124v1)**

> **作者:** David Smith Sundarsingh; Yifei Li; Tianji Tang; George J. Pappas; Nikolay Atanasov; Yiannis Kantaros
>
> **备注:** 8 pages, 5 figures, 2 algorithms, 1 table
>
> **摘要:** This paper addresses semantic planning problems in unknown environments under perceptual uncertainty. The environment contains multiple unknown semantically labeled regions or objects, and the robot must reach desired locations while maintaining class-dependent distances from them. We aim to compute robot paths that complete such semantic reach-avoid tasks with user-defined probability despite uncertain perception. Existing planning algorithms either ignore perceptual uncertainty - thus lacking correctness guarantees - or assume known sensor models and noise characteristics. In contrast, we present the first planner for semantic reach-avoid tasks that achieves user-specified mission completion rates without requiring any knowledge of sensor models or noise. This is enabled by quantifying uncertainty in semantic maps - constructed on-the-fly from perceptual measurements - using conformal prediction in a model- and distribution-free manner. We validate our approach and the theoretical mission completion rates through extensive experiments, showing that it consistently outperforms baselines in mission success rates.
>
---
#### [new 047] CE-Nav: Flow-Guided Reinforcement Refinement for Cross-Embodiment Local Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人局部导航任务，解决跨形态导航策略泛化问题。通过两阶段框架CE-Nav，分离几何推理与动态适配，提升导航系统通用性与效率。**

- **链接: [http://arxiv.org/pdf/2509.23203v1](http://arxiv.org/pdf/2509.23203v1)**

> **作者:** Kai Yang; Tianlin Zhang; Zhengbo Wang; Zedong Chu; Xiaolong Wu; Yang Cai; Mu Xu
>
> **摘要:** Generalizing local navigation policies across diverse robot morphologies is a critical challenge. Progress is often hindered by the need for costly and embodiment-specific data, the tight coupling of planning and control, and the "disastrous averaging" problem where deterministic models fail to capture multi-modal decisions (e.g., turning left or right). We introduce CE-Nav, a novel two-stage (IL-then-RL) framework that systematically decouples universal geometric reasoning from embodiment-specific dynamic adaptation. First, we train an embodiment-agnostic General Expert offline using imitation learning. This expert, a conditional normalizing flow model named VelFlow, learns the full distribution of kinematically-sound actions from a large-scale dataset generated by a classical planner, completely avoiding real robot data and resolving the multi-modality issue. Second, for a new robot, we freeze the expert and use it as a guiding prior to train a lightweight, Dynamics-Aware Refiner via online reinforcement learning. This refiner rapidly learns to compensate for the target robot's specific dynamics and controller imperfections with minimal environmental interaction. Extensive experiments on quadrupeds, bipeds, and quadrotors show that CE-Nav achieves state-of-the-art performance while drastically reducing adaptation cost. Successful real-world deployments further validate our approach as an efficient and scalable solution for building generalizable navigation systems.
>
---
#### [new 048] Leave No Observation Behind: Real-time Correction for VLA Action Chunks
- **分类: cs.RO; cs.AI; cs.CV; cs.SY; eess.SY**

- **简介: 该论文属于视觉-语言-动作任务，解决VLA模型在延迟和长时序下的反应性问题，提出A2C2模块实现实时动作块修正。**

- **链接: [http://arxiv.org/pdf/2509.23224v1](http://arxiv.org/pdf/2509.23224v1)**

> **作者:** Kohei Sendai; Maxime Alvarez; Tatsuya Matsushima; Yutaka Matsuo; Yusuke Iwasawa
>
> **摘要:** To improve efficiency and temporal coherence, Vision-Language-Action (VLA) models often predict action chunks; however, this action chunking harms reactivity under inference delay and long horizons. We introduce Asynchronous Action Chunk Correction (A2C2), which is a lightweight real-time chunk correction head that runs every control step and adds a time-aware correction to any off-the-shelf VLA's action chunk. The module combines the latest observation, the predicted action from VLA (base action), a positional feature that encodes the index of the base action within the chunk, and some features from the base policy, then outputs a per-step correction. This preserves the base model's competence while restoring closed-loop responsiveness. The approach requires no retraining of the base policy and is orthogonal to asynchronous execution schemes such as Real Time Chunking (RTC). On the dynamic Kinetix task suite (12 tasks) and LIBERO Spatial, our method yields consistent success rate improvements across increasing delays and execution horizons (+23% point and +7% point respectively, compared to RTC), and also improves robustness for long horizons even with zero injected delay. Since the correction head is small and fast, there is minimal overhead compared to the inference of large VLA models. These results indicate that A2C2 is an effective, plug-in mechanism for deploying high-capacity chunking policies in real-time control.
>
---
#### [new 049] Memory Transfer Planning: LLM-driven Context-Aware Code Adaptation for Robot Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人操作任务，旨在解决LMM在新环境中的适应性问题。通过引入MTP框架，利用历史代码示例进行上下文适配，提升规划效果与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.24160v1](http://arxiv.org/pdf/2509.24160v1)**

> **作者:** Tomoyuki Kagaya; Subramanian Lakshmi; Yuxuan Lou; Thong Jing Yuan; Jayashree Karlekar; Sugiri Pranata; Natsuki Murakami; Akira Kinose; Yang You
>
> **摘要:** Large language models (LLMs) are increasingly explored in robot manipulation, but many existing methods struggle to adapt to new environments. Many systems require either environment-specific policy training or depend on fixed prompts and single-shot code generation, leading to limited transferability and manual re-tuning. We introduce Memory Transfer Planning (MTP), a framework that leverages successful control-code examples from different environments as procedural knowledge, using them as in-context guidance for LLM-driven planning. Specifically, MTP (i) generates an initial plan and code using LLMs, (ii) retrieves relevant successful examples from a code memory, and (iii) contextually adapts the retrieved code to the target setting for re-planning without updating model parameters. We evaluate MTP on RLBench, CALVIN, and a physical robot, demonstrating effectiveness beyond simulation. Across these settings, MTP consistently improved success rate and adaptability compared with fixed-prompt code generation, naive retrieval, and memory-free re-planning. Furthermore, in hardware experiments, leveraging a memory constructed in simulation proved effective. MTP provides a practical approach that exploits procedural knowledge to realize robust LLM-based planning across diverse robotic manipulation scenarios, enhancing adaptability to novel environments and bridging simulation and real-world deployment.
>
---
#### [new 050] Stabilizing Humanoid Robot Trajectory Generation via Physics-Informed Learning and Control-Informed Steering
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **简介: 该论文属于人形机器人轨迹生成任务，旨在解决数据不足和物理约束违反问题。通过融合物理知识和控制策略，提升轨迹的稳定性和真实性。**

- **链接: [http://arxiv.org/pdf/2509.24697v1](http://arxiv.org/pdf/2509.24697v1)**

> **作者:** Evelyn D'Elia; Paolo Maria Viceconte; Lorenzo Rapetti; Diego Ferigo; Giulio Romualdi; Giuseppe L'Erario; Raffaello Camoriano; Daniele Pucci
>
> **备注:** This paper has been accepted for publication at the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Hangzhou, China, 2025
>
> **摘要:** Recent trends in humanoid robot control have successfully employed imitation learning to enable the learned generation of smooth, human-like trajectories from human data. While these approaches make more realistic motions possible, they are limited by the amount of available motion data, and do not incorporate prior knowledge about the physical laws governing the system and its interactions with the environment. Thus they may violate such laws, leading to divergent trajectories and sliding contacts which limit real-world stability. We address such limitations via a two-pronged learning strategy which leverages the known physics of the system and fundamental control principles. First, we encode physics priors during supervised imitation learning to promote trajectory feasibility. Second, we minimize drift at inference time by applying a proportional-integral controller directly to the generated output state. We validate our method on various locomotion behaviors for the ergoCub humanoid robot, where a physics-informed loss encourages zero contact foot velocity. Our experiments demonstrate that the proposed approach is compatible with multiple controllers on a real robot and significantly improves the accuracy and physical constraint conformity of generated trajectories.
>
---
#### [new 051] Towards Modular and Accessible AUV Systems
- **分类: cs.RO**

- **简介: 该论文属于AUV系统开发任务，旨在解决模块化与可访问性问题。提出MVP框架，集成软硬件设计，提升定制化与兼容性。**

- **链接: [http://arxiv.org/pdf/2509.24864v1](http://arxiv.org/pdf/2509.24864v1)**

> **作者:** Mingxi Zhou; Farhang Naderi; Yuewei Fu; Tony Jacob; Lin Zhao; Manavi Panjnani; Chengzhi Yuan; William McConnell; Emir Cem Gezer
>
> **备注:** 6 pages, accepted by 2024 IEEE/OES Autonomous Underwater Vehicles Symposium (AUV)
>
> **摘要:** This paper reports the development of a new open- access modular framework, called Marine Vehicle Packages (MVP), for Autonomous Underwater Vehicles. The framework consists of both software and hardware designs allowing easy construction of AUV for research with increased customizability and sufficient payload capacity. This paper will present the scalable hardware system design and the modular software design architecture. New features, such as articulated thruster integra- tion and high-level Graphic User Interface will be discussed. Both simulation and field experiments results are shown to highlight the performance and compatibility of the MVP.
>
---
#### [new 052] Safe Task Space Synchronization with Time-Delayed Information
- **分类: cs.RO; cs.SY; eess.SP; eess.SY**

- **简介: 该论文属于人机协作任务，解决机器人与人类轨迹同步问题。通过设计自适应控制器，利用延迟信息实现安全同步。**

- **链接: [http://arxiv.org/pdf/2509.22976v1](http://arxiv.org/pdf/2509.22976v1)**

> **作者:** Rounak Bhattacharya; Vrithik R. Guthikonda; Ashwin P. Dani
>
> **摘要:** In this paper, an adaptive controller is designed for the synchronization of the trajectory of a robot with unknown kinematics and dynamics to that of the current human trajectory in the task space using the delayed human trajectory information. The communication time delay may be a result of various factors that arise in human-robot collaboration tasks, such as sensor processing or fusion to estimate trajectory/intent, network delays, or computational limitations. The developed adaptive controller uses Barrier Lyapunov Function (BLF) to constrain the Cartesian coordinates of the robot to ensure safety, an ICL-based adaptive law to account for the unknown kinematics, and a gradient-based adaptive law to estimate unknown dynamics. Barrier Lyapunov-Krasovskii (LK) functionals are used for the stability analysis to show that the synchronization and parameter estimation errors remain semi-globally uniformly ultimately bounded (SGUUB). The simulation results based on a human-robot synchronization scenario with time delay are provided to demonstrate the effectiveness of the designed synchronization controller with safety constraints.
>
---
#### [new 053] Online Dynamic Goal Recognition in Gym Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于目标识别任务，旨在解决GR研究中基准不统一的问题。作者提出了两个开源框架，支持GR算法的开发与评估。**

- **链接: [http://arxiv.org/pdf/2509.23244v1](http://arxiv.org/pdf/2509.23244v1)**

> **作者:** Shamir Matan; Elhadad Osher; Nageris Ben; Mirsky Reuth
>
> **摘要:** Goal Recognition (GR) is the task of inferring an agent's intended goal from partial observations of its behavior, typically in an online and one-shot setting. Despite recent advances in model-free GR, particularly in applications such as human-robot interaction, surveillance, and assistive systems, the field remains fragmented due to inconsistencies in benchmarks, domains, and evaluation protocols. To address this, we introduce gr-libs (https://github.com/MatanShamir1/gr_libs) and gr-envs (https://github.com/MatanShamir1/gr_envs), two complementary open-source frameworks that support the development, evaluation, and comparison of GR algorithms in Gym-compatible environments. gr-libs includes modular implementations of MDP-based GR baselines, diagnostic tools, and evaluation utilities. gr-envs provides a curated suite of environments adapted for dynamic and goal-directed behavior, along with wrappers that ensure compatibility with standard reinforcement learning toolkits. Together, these libraries offer a standardized, extensible, and reproducible platform for advancing GR research. Both packages are open-source and available on GitHub and PyPI.
>
---
#### [new 054] LAGEA: Language Guided Embodied Agents for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决机器人缺乏从自身错误中学习的问题。通过语言反馈引导机器人自我反思并改进决策。**

- **链接: [http://arxiv.org/pdf/2509.23155v1](http://arxiv.org/pdf/2509.23155v1)**

> **作者:** Abdul Monaf Chowdhury; Akm Moshiur Rahman Mazumder; Rabeya Akter; Safaeid Hossain Arib
>
> **摘要:** Robotic manipulation benefits from foundation models that describe goals, but today's agents still lack a principled way to learn from their own mistakes. We ask whether natural language can serve as feedback, an error reasoning signal that helps embodied agents diagnose what went wrong and correct course. We introduce LAGEA (Language Guided Embodied Agents), a framework that turns episodic, schema-constrained reflections from a vision language model (VLM) into temporally grounded guidance for reinforcement learning. LAGEA summarizes each attempt in concise language, localizes the decisive moments in the trajectory, aligns feedback with visual state in a shared representation, and converts goal progress and feedback agreement into bounded, step-wise shaping rewardswhose influence is modulated by an adaptive, failure-aware coefficient. This design yields dense signals early when exploration needs direction and gracefully recedes as competence grows. On the Meta-World MT10 embodied manipulation benchmark, LAGEA improves average success over the state-of-the-art (SOTA) methods by 9.0% on random goals and 5.3% on fixed goals, while converging faster. These results support our hypothesis: language, when structured and grounded in time, is an effective mechanism for teaching robots to self-reflect on mistakes and make better choices. Code will be released soon.
>
---
#### [new 055] Learning to Sample: Reinforcement Learning-Guided Sampling for Autonomous Vehicle Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于自主车辆运动规划任务，解决复杂城市场景中采样效率低的问题。通过强化学习引导采样，提升轨迹可行性，减少样本数量和运行时间。**

- **链接: [http://arxiv.org/pdf/2509.24313v1](http://arxiv.org/pdf/2509.24313v1)**

> **作者:** Korbinian Moller; Roland Stroop; Mattia Piccinini; Alexander Langmann; Johannes Betz
>
> **备注:** 8 pages, submitted to the IEEE ICRA 2026, Vienna, Austria
>
> **摘要:** Sampling-based motion planning is a well-established approach in autonomous driving, valued for its modularity and analytical tractability. In complex urban scenarios, however, uniform or heuristic sampling often produces many infeasible or irrelevant trajectories. We address this limitation with a hybrid framework that learns where to sample while keeping trajectory generation and evaluation fully analytical and verifiable. A reinforcement learning (RL) agent guides the sampling process toward regions of the action space likely to yield feasible trajectories, while evaluation and final selection remains governed by deterministic feasibility checks and cost functions. We couple the RL sampler with a world model (WM) based on a decodable deep set encoder, enabling both variable numbers of traffic participants and reconstructable latent representations. The approach is evaluated in the CommonRoad simulation environment, showing up to 99% fewer required samples and a runtime reduction of up to 84% while maintaining planning quality in terms of success and collision-free rates. These improvements lead to faster, more reliable decision-making for autonomous vehicles in urban environments, achieving safer and more responsive navigation under real-world constraints. Code and trained artifacts are publicly available at: https://github.com/TUM-AVS/Learning-to-Sample
>
---
#### [new 056] MSG: Multi-Stream Generative Policies for Sample-Efficient Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人操作任务，解决生成式策略样本效率低的问题。提出MSG框架，通过多流策略组合提升泛化能力和效率。**

- **链接: [http://arxiv.org/pdf/2509.24956v1](http://arxiv.org/pdf/2509.24956v1)**

> **作者:** Jan Ole von Hartz; Lukas Schweizer; Joschka Boedecker; Abhinav Valada
>
> **摘要:** Generative robot policies such as Flow Matching offer flexible, multi-modal policy learning but are sample-inefficient. Although object-centric policies improve sample efficiency, it does not resolve this limitation. In this work, we propose Multi-Stream Generative Policy (MSG), an inference-time composition framework that trains multiple object-centric policies and combines them at inference to improve generalization and sample efficiency. MSG is model-agnostic and inference-only, hence widely applicable to various generative policies and training paradigms. We perform extensive experiments both in simulation and on a real robot, demonstrating that our approach learns high-quality generative policies from as few as five demonstrations, resulting in a 95% reduction in demonstrations, and improves policy performance by 89 percent compared to single-stream approaches. Furthermore, we present comprehensive ablation studies on various composition strategies and provide practical recommendations for deployment. Finally, MSG enables zero-shot object instance transfer. We make our code publicly available at https://msg.cs.uni-freiburg.de.
>
---
#### [new 057] Contextual Neural Moving Horizon Estimation for Robust Quadrotor Control in Varying Conditions
- **分类: cs.RO**

- **简介: 该论文属于无人机控制任务，解决复杂环境下扰动估计问题。通过结合神经网络与贝叶斯优化，动态调整参数以提升鲁棒性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.24281v1](http://arxiv.org/pdf/2509.24281v1)**

> **作者:** Kasra Torshizi; Chak Lam Shek; Khuzema Habib; Guangyao Shi; Pratap Tokekar; Troi Williams
>
> **备注:** 9 pages, 7 Figures, Kasra Torshizi and Chak Lam Shek contributed equally
>
> **摘要:** Adaptive controllers on quadrotors typically rely on estimation of disturbances to ensure robust trajectory tracking. Estimating disturbances across diverse environmental contexts is challenging due to the inherent variability and uncertainty in the real world. Such estimators require extensive fine-tuning for a specific scenario, which makes them inflexible and brittle to changing conditions. Machine-learning approaches, such as training a neural network to tune the estimator's parameters, are promising. However, collecting data across all possible environmental contexts is impossible. It is also inefficient as the same estimator parameters could work for "nearby" contexts. In this paper, we present a sequential decision making strategy that decides which environmental contexts, using Bayesian Optimization with a Gaussian Process, to collect data from in order to ensure robust performance across a wide range of contexts. Our method, Contextual NeuroMHE, eliminates the need for exhaustive training across all environments while maintaining robust performance under different conditions. By enabling the neural network to adapt its parameters dynamically, our method improves both efficiency and generalization. Experimental results in various real-world settings demonstrate that our approach outperforms the prior work by 20.3\% in terms of maximum absolute position error and can capture the variations in the environment with a few carefully chosen contexts.
>
---
#### [new 058] From Code to Action: Hierarchical Learning of Diffusion-VLM Policies
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人操作任务，解决模仿学习中的泛化与数据不足问题。通过结合代码生成的视觉语言模型和扩散策略，实现分层控制与任务分解。**

- **链接: [http://arxiv.org/pdf/2509.24917v1](http://arxiv.org/pdf/2509.24917v1)**

> **作者:** Markus Peschl; Pietro Mazzaglia; Daniel Dijkman
>
> **备注:** 19 pages including references, 6 figures. Accepted to CoRL LEAP 2025
>
> **摘要:** Imitation learning for robotic manipulation often suffers from limited generalization and data scarcity, especially in complex, long-horizon tasks. In this work, we introduce a hierarchical framework that leverages code-generating vision-language models (VLMs) in combination with low-level diffusion policies to effectively imitate and generalize robotic behavior. Our key insight is to treat open-source robotic APIs not only as execution interfaces but also as sources of structured supervision: the associated subtask functions - when exposed - can serve as modular, semantically meaningful labels. We train a VLM to decompose task descriptions into executable subroutines, which are then grounded through a diffusion policy trained to imitate the corresponding robot behavior. To handle the non-Markovian nature of both code execution and certain real-world tasks, such as object swapping, our architecture incorporates a memory mechanism that maintains subtask context across time. We find that this design enables interpretable policy decomposition, improves generalization when compared to flat policies and enables separate evaluation of high-level planning and low-level control.
>
---
#### [new 059] U-DiT Policy: U-shaped Diffusion Transformers for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决扩散策略的全局上下文建模不足和过平滑问题。提出U-DiT框架，结合U-Net与Transformer优势，提升性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.24579v1](http://arxiv.org/pdf/2509.24579v1)**

> **作者:** Linzhi Wu; Aoran Mei; Xiyue Wang; Guo-Niu Zhu; Zhongxue Gan
>
> **摘要:** Diffusion-based methods have been acknowledged as a powerful paradigm for end-to-end visuomotor control in robotics. Most existing approaches adopt a Diffusion Policy in U-Net architecture (DP-U), which, while effective, suffers from limited global context modeling and over-smoothing artifacts. To address these issues, we propose U-DiT Policy, a novel U-shaped Diffusion Transformer framework. U-DiT preserves the multi-scale feature fusion advantages of U-Net while integrating the global context modeling capability of Transformers, thereby enhancing representational power and policy expressiveness. We evaluate U-DiT extensively across both simulation and real-world robotic manipulation tasks. In simulation, U-DiT achieves an average performance gain of 10\% over baseline methods and surpasses Transformer-based diffusion policies (DP-T) that use AdaLN blocks by 6\% under comparable parameter budgets. On real-world robotic tasks, U-DiT demonstrates superior generalization and robustness, achieving an average improvement of 22.5\% over DP-U. In addition, robustness and generalization experiments under distractor and lighting variations further highlight the advantages of U-DiT. These results highlight the effectiveness and practical potential of U-DiT Policy as a new foundation for diffusion-based robotic manipulation.
>
---
#### [new 060] CEDex: Cross-Embodiment Dexterous Grasp Generation at Scale from Human-like Contact Representations
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人抓取任务，解决跨形态灵巧抓取问题。通过生成类人接触表示并对其对齐优化，实现多种机械手的高质量抓取生成。**

- **链接: [http://arxiv.org/pdf/2509.24661v1](http://arxiv.org/pdf/2509.24661v1)**

> **作者:** Zhiyuan Wu; Rolandos Alexandros Potamias; Xuyang Zhang; Zhongqun Zhang; Jiankang Deng; Shan Luo
>
> **摘要:** Cross-embodiment dexterous grasp synthesis refers to adaptively generating and optimizing grasps for various robotic hands with different morphologies. This capability is crucial for achieving versatile robotic manipulation in diverse environments and requires substantial amounts of reliable and diverse grasp data for effective model training and robust generalization. However, existing approaches either rely on physics-based optimization that lacks human-like kinematic understanding or require extensive manual data collection processes that are limited to anthropomorphic structures. In this paper, we propose CEDex, a novel cross-embodiment dexterous grasp synthesis method at scale that bridges human grasping kinematics and robot kinematics by aligning robot kinematic models with generated human-like contact representations. Given an object's point cloud and an arbitrary robotic hand model, CEDex first generates human-like contact representations using a Conditional Variational Auto-encoder pretrained on human contact data. It then performs kinematic human contact alignment through topological merging to consolidate multiple human hand parts into unified robot components, followed by a signed distance field-based grasp optimization with physics-aware constraints. Using CEDex, we construct the largest cross-embodiment grasp dataset to date, comprising 500K objects across four gripper types with 20M total grasps. Extensive experiments show that CEDex outperforms state-of-the-art approaches and our dataset benefits cross-embodiment grasp learning with high-quality diverse grasps.
>
---
#### [new 061] APREBot: Active Perception System for Reflexive Evasion Robot
- **分类: cs.RO**

- **简介: 该论文属于机器人自主导航任务，解决动态环境中障碍物快速避让问题。通过融合LiDAR与相机的主动感知系统，提升机器人环境感知能力。**

- **链接: [http://arxiv.org/pdf/2509.24733v1](http://arxiv.org/pdf/2509.24733v1)**

> **作者:** Zihao Xu; Kuankuan Sima; Junhao Deng; Zixuan Zhuang; Chunzheng Wang; Ce Hao; Jin Song Dong
>
> **摘要:** Reliable onboard perception is critical for quadruped robots navigating dynamic environments, where obstacles can emerge from any direction under strict reaction-time constraints. Single-sensor systems face inherent limitations: LiDAR provides omnidirectional coverage but lacks rich texture information, while cameras capture high-resolution detail but suffer from restricted field of view. We introduce APREBot (Active Perception System for Reflexive Evasion Robot), a novel framework that integrates reflexive evasion with active hierarchical perception. APREBot strategically combines LiDAR-based omnidirectional scanning with camera-based active focusing, achieving comprehensive environmental awareness essential for agile obstacle avoidance in quadruped robots. We validate APREBot through extensive sim-to-real experiments on a quadruped platform, evaluating diverse obstacle types, trajectories, and approach directions. Our results demonstrate substantial improvements over state-of-the-art baselines in both safety metrics and operational efficiency, highlighting APREBot's potential for dependable autonomy in safety-critical scenarios. Videos are available at https://sites.google.com/view/aprebot/
>
---
#### [new 062] SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人路径规划任务，解决安全性和实时性问题。通过结合流匹配与控制屏障函数，提出SafeFlowMatcher框架，确保路径安全且高效。**

- **链接: [http://arxiv.org/pdf/2509.24243v1](http://arxiv.org/pdf/2509.24243v1)**

> **作者:** Jeongyong Yang; Seunghwan Jang; Soojean Han
>
> **备注:** 10 pages, 7 figures, 4 tables
>
> **摘要:** Generative planners based on flow matching (FM) can produce high-quality paths in one or a few ODE steps, but their sampling dynamics offer no formal safety guarantees and can yield incomplete paths near constraints. We present SafeFlowMatcher, a planning framework that couples FM with control barrier functions (CBFs) to achieve both real-time efficiency and certified safety. SafeFlowMatcher uses a two-phase prediction-correction (PC) integrator: (i) a prediction phase integrates the learned FM once (or a few steps) to obtain a candidate path without intervention; (ii) a correction phase refines this path with a vanishing time-scaled vector field and a CBF-based quadratic program that minimally perturbs the vector field. We prove a barrier certificate for the resulting flow system, establishing forward invariance of a robust safe set and finite-time convergence to the safe set. By enforcing safety only on the executed path (rather than on all intermediate latent paths), SafeFlowMatcher avoids distributional drift and mitigates local trap problems. Across maze navigation and locomotion benchmarks, SafeFlowMatcher attains faster, smoother, and safer paths than diffusion- and FM-based baselines. Extensive ablations corroborate the contributions of the PC integrator and the barrier certificate.
>
---
#### [new 063] A Novel Model for 3D Motion Planning for a Generalized Dubins Vehicle with Pitch and Yaw Rate Constraints
- **分类: cs.RO; math.OC**

- **简介: 该论文属于3D路径规划任务，解决固定翼无人机在姿态约束下的最短路径问题。通过引入完整姿态模型和双控制输入，提出新算法生成更优路径。**

- **链接: [http://arxiv.org/pdf/2509.24143v1](http://arxiv.org/pdf/2509.24143v1)**

> **作者:** Deepak Prakash Kumar; Swaroop Darbha; Satyanarayana Gupta Manyam; David Casbeer
>
> **备注:** The code for this paper is available at https://github.com/DeepakPrakashKumar/3D-Motion-Planning-for-Generalized-Dubins-with-Pitch-Yaw-constraints
>
> **摘要:** In this paper, we propose a new modeling approach and a fast algorithm for 3D motion planning, applicable for fixed-wing unmanned aerial vehicles. The goal is to construct the shortest path connecting given initial and final configurations subject to motion constraints. Our work differs from existing literature in two ways. First, we consider full vehicle orientation using a body-attached frame, which includes roll, pitch, and yaw angles. However, existing work uses only pitch and/or heading angle, which is insufficient to uniquely determine orientation. Second, we use two control inputs to represent bounded pitch and yaw rates, reflecting control by two separate actuators. In contrast, most previous methods rely on a single input, such as path curvature, which is insufficient for accurately modeling the vehicle's kinematics in 3D. We use a rotation minimizing frame to describe the vehicle's configuration and its evolution, and construct paths by concatenating optimal Dubins paths on spherical, cylindrical, or planar surfaces. Numerical simulations show our approach generates feasible paths within 10 seconds on average and yields shorter paths than existing methods in most cases.
>
---
#### [new 064] Zero-shot Whole-Body Manipulation with a Large-Scale Soft Robotic Torso via Guided Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决软体机器人在复杂环境中进行全身操作的问题。通过引导强化学习，实现零样本仿真到现实的高效操控。**

- **链接: [http://arxiv.org/pdf/2509.23556v1](http://arxiv.org/pdf/2509.23556v1)**

> **作者:** Curtis C. Johnson; Carlo Alessi; Egidio Falotico; Marc D. Killpack
>
> **备注:** Submitted to IEEE Transactions on Robotics for review
>
> **摘要:** Whole-body manipulation is a powerful yet underexplored approach that enables robots to interact with large, heavy, or awkward objects using more than just their end-effectors. Soft robots, with their inherent passive compliance, are particularly well-suited for such contact-rich manipulation tasks, but their uncertainties in kinematics and dynamics pose significant challenges for simulation and control. In this work, we address this challenge with a simulation that can run up to 350x real time on a single thread in MuJoCo and provide a detailed analysis of the critical tradeoffs between speed and accuracy for this simulation. Using this framework, we demonstrate a successful zero-shot sim-to-real transfer of a learned whole-body manipulation policy, achieving an 88% success rate on the Baloo hardware platform. We show that guiding RL with a simple motion primitive is critical to this success where standard reward shaping methods struggled to produce a stable and successful policy for whole-body manipulation. Furthermore, our analysis reveals that the learned policy does not simply mimic the motion primitive. It exhibits beneficial reactive behavior, such as re-grasping and perturbation recovery. We analyze and contrast this learned policy against an open-loop baseline to show that the policy can also exhibit aggressive over-corrections under perturbation. To our knowledge, this is the first demonstration of forceful, six-DoF whole-body manipulation using two continuum soft arms on a large-scale platform (10 kg payloads), with zero-shot policy transfer.
>
---
#### [new 065] GLUE: Global-Local Unified Encoding for Imitation Learning via Key-Patch Tracking
- **分类: cs.RO**

- **简介: 该论文属于机器人模仿学习任务，解决复杂场景下视觉表示不稳定的问题。通过全局-局部统一编码框架GLUE，提升策略鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.23220v1](http://arxiv.org/pdf/2509.23220v1)**

> **作者:** Ye Chen; Zichen Zhou; Jianyu Dou; Te Cui; Yi Yang; Yufeng Yue
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** In recent years, visual representation learning has gained widespread attention in robotic imitation learning. However, in complex Out-of-Distribution(OOD) settings characterized by clutter and occlusion, the attention of global visual representations can be diluted or interfered, leading to degraded policy performance. The invariance of local representations for task-relevant objects offers a solution. By efficiently utilizing these local representations, training and testing data can be mapped to a more similar feature space, thereby mitigating the covariate shift problem. Accordingly, we propose GLUE, a global-local unified encoding framework for imitation learning based on key-patch tracking. GLUE selects and tracks key-patches as critical local representations by employing a text-guided mechanism. It features a novel fusion framework where global patch features query local patches to distill essential information, yielding fine-grained local features with low heterogeneity relative to the global context. This fused representation steers the robot's visual attention toward task-relevant objects and preserves precise global context, which together align the training and testing distributions into a similar and task-informative feature space, ultimately enhancing the robustness of the imitation learning policy. Experiments demonstrate that GLUE achieves strong performance across diverse tasks in both simulation and real-world settings, outperforming the strongest baseline by 17.6% in simulation, 36.3% in real-world environments, and 58.3% on real-world generalization settings. The project website of GLUE is available at https://GLUE666.github.io/.
>
---
#### [new 066] DynaMIC: Dynamic Multimodal In-Context Learning Enabled Embodied Robot Counterfactual Resistance Ability
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于机器人控制任务，旨在解决机器人因遵循错误指令导致的安全问题。提出DynaMIC框架，用于识别误导性指令并主动反馈，提升执行可靠性。**

- **链接: [http://arxiv.org/pdf/2509.24413v1](http://arxiv.org/pdf/2509.24413v1)**

> **作者:** Tianqiang Yan; Ziqiao Lin; Sicheng Wang; Tianwei Zhang; Zhenglong Sun
>
> **摘要:** The emergence of large pre-trained models based on natural language has breathed new life into robotics development. Extensive research has integrated large models with robots, utilizing the powerful semantic understanding and generation capabilities of large models to facilitate robot control through natural language instructions gradually. However, we found that robots that strictly adhere to human instructions, especially those containing misleading information, may encounter errors during task execution, potentially leading to safety hazards. This resembles the concept of counterfactuals in natural language processing (NLP), which has not yet attracted much attention in robotic research. In an effort to highlight this issue for future studies, this paper introduced directive counterfactuals (DCFs) arising from misleading human directives. We present DynaMIC, a framework for generating robot task flows to identify DCFs and relay feedback to humans proactively. This capability can help robots be sensitive to potential DCFs within a task, thus enhancing the reliability of the execution process. We conducted semantic-level experiments and ablation studies, showcasing the effectiveness of this framework.
>
---
#### [new 067] Open-Vocabulary Spatio-Temporal Scene Graph for Robot Perception and Teleoperation Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人感知与远程操作任务，解决通信延迟导致的指令误解问题。提出ST-OVSG模型，融合时空信息和开放词汇，提升规划鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.23107v1](http://arxiv.org/pdf/2509.23107v1)**

> **作者:** Yi Wang; Zeyu Xue; Mujie Liu; Tongqin Zhang; Yan Hu; Zhou Zhao; Chenguang Yang; Zhenyu Lu
>
> **摘要:** Teleoperation via natural-language reduces operator workload and enhances safety in high-risk or remote settings. However, in dynamic remote scenes, transmission latency during bidirectional communication creates gaps between remote perceived states and operator intent, leading to command misunderstanding and incorrect execution. To mitigate this, we introduce the Spatio-Temporal Open-Vocabulary Scene Graph (ST-OVSG), a representation that enriches open-vocabulary perception with temporal dynamics and lightweight latency annotations. ST-OVSG leverages LVLMs to construct open-vocabulary 3D object representations, and extends them into the temporal domain via Hungarian assignment with our temporal matching cost, yielding a unified spatio-temporal scene graph. A latency tag is embedded to enable LVLM planners to retrospectively query past scene states, thereby resolving local-remote state mismatches caused by transmission delays. To further reduce redundancy and highlight task-relevant cues, we propose a task-oriented subgraph filtering strategy that produces compact inputs for the planner. ST-OVSG generalizes to novel categories and enhances planning robustness against transmission latency without requiring fine-tuning. Experiments show that our method achieves 74 percent node accuracy on the Replica benchmark, outperforming ConceptGraph. Notably, in the latency-robustness experiment, the LVLM planner assisted by ST-OVSG achieved a planning success rate of 70.5 percent.
>
---
#### [new 068] Fidelity-Aware Data Composition for Robust Robot Generalization
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人学习任务，解决OOP泛化问题。通过引入CIFT框架，实现高保真数据组合，提升机器人在分布外场景的表现。**

- **链接: [http://arxiv.org/pdf/2509.24797v1](http://arxiv.org/pdf/2509.24797v1)**

> **作者:** Zizhao Tong; Di Chen; Sicheng Hu; Hongwei Fan; Liliang Chen; Guanghui Ren; Hao Tang; Hao Dong; Ling Shao
>
> **备注:** 33 pages
>
> **摘要:** Generalist robot policies trained on large-scale, visually homogeneous datasets can be susceptible to shortcut learning, which impairs their out-of-distribution (OOD) generalization. While generative data augmentation is a common approach to introduce diversity, it presents a subtle challenge: data composition. Naively mixing real and synthetic data can corrupt the learning signal, as this process often prioritizes visual diversity at the expense of information fidelity. This paper suggests that robust generalization depends on principled, fidelity-aware data composition. We introduce Coherent Information Fidelity Tuning (CIFT), a framework that treats data composition as an optimization problem. CIFT uses a practical proxy for Information Fidelity based on the feature-space geometry of a dataset. This enables the identification of a phase transition, termed the Decoherence Point, where training stability degrades. The framework includes a generative engine, Multi-View Video Augmentation (MVAug), to synthesize a causally disentangled data spectrum for this tuning process. Applying CIFT to policy architectures such as $\pi_0$ and Diffusion Policy improves OOD success rates by over 54\%. These results indicate that fidelity-aware composition, beyond data synthesis alone, is an important component for developing robust, general-purpose robots.
>
---
#### [new 069] Hierarchical Control Design for Space Robots with Application to In-Orbit Servicing Missions
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于空间机器人控制任务，旨在解决在轨服务中捕获失控目标的问题。通过设计分层控制框架，提升捕获的鲁棒性与适应性。**

- **链接: [http://arxiv.org/pdf/2509.22955v1](http://arxiv.org/pdf/2509.22955v1)**

> **作者:** Pietro Bruschi
>
> **摘要:** In-Orbit Servicing and Active Debris Removal require advanced robotic capabilities for capturing and detumbling uncooperative targets. This work presents a hierarchical control framework for autonomous robotic capture of tumbling objects in space. A simulation environment is developed, incorporating sloshing dynamics of the chaser, a rarely studied effect in space robotics. The proposed controller combines an inner Lyapunov-based robust control loop for multi-body dynamics with an outer loop addressing an extended inverse kinematics problem. Simulation results show improved robustness and adaptability compared to existing control schemes.
>
---
#### [new 070] Preventing Robotic Jailbreaking via Multimodal Domain Adaptation
- **分类: cs.RO; I.2.6; I.2.9**

- **简介: 该论文属于机器人安全任务，解决VLMs易受 jailbreaking 攻击的问题。通过多模态域适应方法J-DAPT提升检测准确性至近100%。**

- **链接: [http://arxiv.org/pdf/2509.23281v1](http://arxiv.org/pdf/2509.23281v1)**

> **作者:** Francesco Marchiori; Rohan Sinha; Christopher Agia; Alexander Robey; George J. Pappas; Mauro Conti; Marco Pavone
>
> **备注:** Project page: https://j-dapt.github.io/. 9 pages, 6 figures
>
> **摘要:** Large Language Models (LLMs) and Vision-Language Models (VLMs) are increasingly deployed in robotic environments but remain vulnerable to jailbreaking attacks that bypass safety mechanisms and drive unsafe or physically harmful behaviors in the real world. Data-driven defenses such as jailbreak classifiers show promise, yet they struggle to generalize in domains where specialized datasets are scarce, limiting their effectiveness in robotics and other safety-critical contexts. To address this gap, we introduce J-DAPT, a lightweight framework for multimodal jailbreak detection through attention-based fusion and domain adaptation. J-DAPT integrates textual and visual embeddings to capture both semantic intent and environmental grounding, while aligning general-purpose jailbreak datasets with domain-specific reference data. Evaluations across autonomous driving, maritime robotics, and quadruped navigation show that J-DAPT boosts detection accuracy to nearly 100% with minimal overhead. These results demonstrate that J-DAPT provides a practical defense for securing VLMs in robotic applications. Additional materials are made available at: https://j-dapt.github.io.
>
---
#### [new 071] HeLoM: Hierarchical Learning for Whole-Body Loco-Manipulation in Hexapod Robot
- **分类: cs.RO**

- **简介: 该论文属于六足机器人全身操控任务，解决重物推动稳定性问题。提出HeLoM框架，通过多肢体协同实现稳定推动物体。**

- **链接: [http://arxiv.org/pdf/2509.23651v1](http://arxiv.org/pdf/2509.23651v1)**

> **作者:** Xinrong Yang; Peizhuo Li; Hongyi Li; Junkai Lu; Linnan Chang; Yuhong Cao; Yifeng Zhang; Ge Sun; Guillaume Sartoretti
>
> **摘要:** Robots in real-world environments are often required to move/manipulate objects comparable in weight to their own bodies. Compared to grasping and carrying, pushing provides a more straightforward and efficient non-prehensile manipulation strategy, avoiding complex grasp design while leveraging direct contact to regulate an object's pose. Achieving effective pushing, however, demands both sufficient manipulation forces and the ability to maintain stability, which is particularly challenging when dealing with heavy or irregular objects. To address these challenges, we propose HeLoM, a learning-based hierarchical whole-body manipulation framework for a hexapod robot that exploits coordinated multi-limb control. Inspired by the cooperative strategies of multi-legged insects, our framework leverages redundant contact points and high degrees of freedom to enable dynamic redistribution of contact forces. HeLoM's high-level planner plans pushing behaviors and target object poses, while its low-level controller maintains locomotion stability and generates dynamically consistent joint actions. Our policies trained in simulation are directly deployed on real robots without additional fine-tuning. This design allows the robot to maintain balance while exerting continuous and controllable pushing forces through coordinated foreleg interaction and supportive hind-leg propulsion. We validate the effectiveness of HeLoM through both simulation and real-world experiments. Results show that our framework can stably push boxes of varying sizes and unknown physical properties to designated goal poses in the real world.
>
---
#### [new 072] PoseDiff: A Unified Diffusion Model Bridging Robot Pose Estimation and Video-to-Action Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人姿态估计与视频到动作控制任务，旨在统一感知与控制流程。提出PoseDiff模型，通过单图像估计姿态并生成连续动作序列，提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.24591v1](http://arxiv.org/pdf/2509.24591v1)**

> **作者:** Haozhuo Zhang; Michele Caprio; Jing Shao; Qiang Zhang; Jian Tang; Shanghang Zhang; Wei Pan
>
> **摘要:** We present PoseDiff, a conditional diffusion model that unifies robot state estimation and control within a single framework. At its core, PoseDiff maps raw visual observations into structured robot states-such as 3D keypoints or joint angles-from a single RGB image, eliminating the need for multi-stage pipelines or auxiliary modalities. Building upon this foundation, PoseDiff extends naturally to video-to-action inverse dynamics: by conditioning on sparse video keyframes generated by world models, it produces smooth and continuous long-horizon action sequences through an overlap-averaging strategy. This unified design enables scalable and efficient integration of perception and control. On the DREAM dataset, PoseDiff achieves state-of-the-art accuracy and real-time performance for pose estimation. On Libero-Object manipulation tasks, it substantially improves success rates over existing inverse dynamics modules, even under strict offline settings. Together, these results show that PoseDiff provides a scalable, accurate, and efficient bridge between perception, planning, and control in embodied AI. The video visualization results can be found on the project page: https://haozhuo-zhang.github.io/PoseDiff-project-page/.
>
---
#### [new 073] Towards Tighter Convex Relaxation of Mixed-integer Programs: Leveraging Logic Network Flow for Task and Motion Planning
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人任务与运动规划领域，解决如何更高效地整合时序逻辑与混合整数规划的问题。提出“逻辑网络流”框架，通过网络流模型提升凸松弛的紧致性，实现更快的计算和实时重规划。**

- **链接: [http://arxiv.org/pdf/2509.24235v1](http://arxiv.org/pdf/2509.24235v1)**

> **作者:** Xuan Lin; Jiming Ren; Yandong Luo; Weijun Xie; Ye Zhao
>
> **备注:** 35 pages, 17 figures, 7 tables
>
> **摘要:** This paper proposes an optimization-based task and motion planning framework, named "Logic Network Flow", that integrates temporal logic specifications into mixed-integer programs for efficient robot planning. Inspired by the Graph-of-Convex-Sets formulation, temporal predicates are encoded as polyhedron constraints on each edge of a network flow model, instead of as constraints between nodes in traditional Logic Tree formulations. We further propose a network-flow-based Fourier-Motzkin elimination procedure that removes continuous flow variables while preserving convex relaxation tightness, leading to provably tighter convex relaxations and fewer constraints than Logic Tree formulations. For temporal logic motion planning with piecewise-affine dynamic systems, comprehensive experiments across vehicle routing, multi-robot coordination, and temporal logic control on dynamical systems using point mass and linear inverted pendulum models demonstrate computational speedups of up to several orders of magnitude. Hardware demonstrations with quadrupedal robots validate real-time replanning capabilities under dynamically changing environmental conditions. The project website is at https://logicnetworkflow.github.io/.
>
---
#### [new 074] Curriculum Imitation Learning of Distributed Multi-Robot Policies
- **分类: cs.RO; cs.LG; cs.MA**

- **简介: 该论文属于多机器人系统控制任务，解决长期协调与数据获取难题。通过课程学习和感知估计方法，从全局演示中学习鲁棒的分布式策略。**

- **链接: [http://arxiv.org/pdf/2509.25097v1](http://arxiv.org/pdf/2509.25097v1)**

> **作者:** Jesús Roche; Eduardo Sebastián; Eduardo Montijano
>
> **备注:** Accepted and presented at the Eight Iberian Robotics Conference, 2025
>
> **摘要:** Learning control policies for multi-robot systems (MRS) remains a major challenge due to long-term coordination and the difficulty of obtaining realistic training data. In this work, we address both limitations within an imitation learning framework. First, we shift the typical role of Curriculum Learning in MRS, from scalability with the number of robots, to focus on improving long-term coordination. We propose a curriculum strategy that gradually increases the length of expert trajectories during training, stabilizing learning and enhancing the accuracy of long-term behaviors. Second, we introduce a method to approximate the egocentric perception of each robot using only third-person global state demonstrations. Our approach transforms idealized trajectories into locally available observations by filtering neighbors, converting reference frames, and simulating onboard sensor variability. Both contributions are integrated into a physics-informed technique to produce scalable, distributed policies from observations. We conduct experiments across two tasks with varying team sizes and noise levels. Results show that our curriculum improves long-term accuracy, while our perceptual estimation method yields policies that are robust to realistic uncertainty. Together, these strategies enable the learning of robust, distributed controllers from global demonstrations, even in the absence of expert actions or onboard measurements.
>
---
#### [new 075] Certifiably Optimal State Estimation and Robot Calibration Using Trace-Constrained SDP
- **分类: cs.RO**

- **简介: 该论文属于机器人状态估计与标定任务，解决非凸问题转为凸优化后的秩-1矩阵恢复问题，通过trace-constrained SDP和梯度优化提升求解精度与可靠性。**

- **链接: [http://arxiv.org/pdf/2509.23656v1](http://arxiv.org/pdf/2509.23656v1)**

> **作者:** Liangting Wu; Roberto Tron
>
> **摘要:** Many nonconvex problems in robotics can be relaxed into convex formulations via semidefinite programming (SDP), which offers the advantage of global optimality. The practical quality of these solutions, however, critically depends on achieving rank-1 matrices, a condition that typically requires additional tightening. In this work, we focus on trace-constrained SDPs, where the decision variables are positive semidefinite (PSD) matrices with fixed trace values. These additional constraints not only capture important structural properties but also facilitate first-order methods for recovering rank-1 solutions. We introduce customized fixed-trace variables and constraints to represent common robotic quantities such as rotations and translations, which can be exactly recovered when the corresponding variables are rank-1. To further improve practical performance, we develop a gradient-based refinement procedure that projects relaxed SDP solutions toward rank-1, low-cost candidates, which can then be certified for global optimality via the dual problem. We demonstrate that many robotics tasks can be expressed within this trace-constrained SDP framework, and showcase its effectiveness through simulations in perspective-n-point (PnP) estimation, hand-eye calibration, and dual-robot system calibration. To support broader use, we also introduce a modular ``virtual robot'' abstraction that simplifies modeling across different problem settings.
>
---
#### [new 076] Game Theory to Study Cooperation in Human-Robot Mixed Groups: Exploring the Potential of the Public Good Game
- **分类: cs.RO**

- **简介: 该论文属于人机协作研究，旨在探索机器人在混合群体中促进合作与信任的潜力。通过改进的公共物品博弈实验，分析不同机器人策略对人类参与者合作意愿的影响。**

- **链接: [http://arxiv.org/pdf/2509.24530v1](http://arxiv.org/pdf/2509.24530v1)**

> **作者:** Giulia Pusceddu; Sara Mongile; Francesco Rea; Alessandra Sciutti
>
> **备注:** Work presented at the workshop BAILAR in conjunction with IEEE RO-MAN 2023. Peer reviewed
>
> **摘要:** In this study, we explore the potential of Game Theory as a means to investigate cooperation and trust in human-robot mixed groups. Particularly, we introduce the Public Good Game (PGG), a model highlighting the tension between individual self-interest and collective well-being. In this work, we present a modified version of the PGG, where three human participants engage in the game with the humanoid robot iCub to assess whether various robot game strategies (e.g., always cooperate, always free ride, and tit-for-tat) can influence the participants' inclination to cooperate. We test our setup during a pilot study with nineteen participants. A preliminary analysis indicates that participants prefer not to invest their money in the common pool, despite they perceive the robot as generous. By conducting this research, we seek to gain valuable insights into the role that robots can play in promoting trust and cohesion during human-robot interactions within group contexts. The results of this study may hold considerable potential for developing social robots capable of fostering trust and cooperation within mixed human-robot groups.
>
---
#### [new 077] Annotation-Free One-Shot Imitation Learning for Multi-Step Manipulation Tasks
- **分类: cs.RO**

- **简介: 该论文属于多步骤操作任务的模仿学习，解决单次演示下长序列任务的泛化问题。提出无需额外训练或标注的方法，实现高成功率的技能迁移。**

- **链接: [http://arxiv.org/pdf/2509.24972v1](http://arxiv.org/pdf/2509.24972v1)**

> **作者:** Vijja Wichitwechkarn; Emlyn Williams; Charles Fox; Ruchi Choudhary
>
> **摘要:** Recent advances in one-shot imitation learning have enabled robots to acquire new manipulation skills from a single human demonstration. While existing methods achieve strong performance on single-step tasks, they remain limited in their ability to handle long-horizon, multi-step tasks without additional model training or manual annotation. We propose a method that can be applied to this setting provided a single demonstration without additional model training or manual annotation. We evaluated our method on multi-step and single-step manipulation tasks where our method achieves an average success rate of 82.5% and 90%, respectively. Our method matches and exceeds the performance of the baselines in both these cases. We also compare the performance and computational efficiency of alternative pre-trained feature extractors within our framework.
>
---
#### [new 078] Empart: Interactive Convex Decomposition for Converting Meshes to Parts
- **分类: cs.RO**

- **简介: 该论文属于3D模型简化任务，解决统一简化误差导致的性能与精度不平衡问题。通过交互式区域约束，提升机器人应用中的模拟效率。**

- **链接: [http://arxiv.org/pdf/2509.22847v1](http://arxiv.org/pdf/2509.22847v1)**

> **作者:** Brandon Vu; Shameek Ganguly; Pushkar Joshi
>
> **摘要:** Simplifying complex 3D meshes is a crucial step in robotics applications to enable efficient motion planning and physics simulation. Common methods, such as approximate convex decomposition, represent a mesh as a collection of simple parts, which are computationally inexpensive to simulate. However, existing approaches apply a uniform error tolerance across the entire mesh, which can result in a sub-optimal trade-off between accuracy and performance. For instance, a robot grasping an object needs high-fidelity geometry in the vicinity of the contact surfaces but can tolerate a coarser simplification elsewhere. A uniform tolerance can lead to excessive detail in non-critical areas or insufficient detail where it's needed most. To address this limitation, we introduce Empart, an interactive tool that allows users to specify different simplification tolerances for selected regions of a mesh. Our method leverages existing convex decomposition algorithms as a sub-routine but uses a novel, parallelized framework to handle region-specific constraints efficiently. Empart provides a user-friendly interface with visual feedback on approximation error and simulation performance, enabling designers to iteratively refine their decomposition. We demonstrate that our approach significantly reduces the number of convex parts compared to a state-of-the-art method (V-HACD) at a fixed error threshold, leading to substantial speedups in simulation performance. For a robotic pick-and-place task, Empart-generated collision meshes reduced the overall simulation time by 69% compared to a uniform decomposition, highlighting the value of interactive, region-specific simplification for performant robotics applications.
>
---
#### [new 079] Ask, Reason, Assist: Decentralized Robot Collaboration via Language and Logic
- **分类: cs.RO**

- **简介: 该论文属于多机器人协作任务，解决异构机器人在冲突时的协同问题。通过语言与逻辑结合的框架，实现去中心化帮助请求与选择，提升任务效率。**

- **链接: [http://arxiv.org/pdf/2509.23506v1](http://arxiv.org/pdf/2509.23506v1)**

> **作者:** Dan BW Choe; Sundhar Vinodh Sangeetha; Steven Emanuel; Chih-Yuan Chiu; Samuel Coogan; Shreyas Kousik
>
> **摘要:** Increased robot deployment, such as in warehousing, has revealed a need for seamless collaboration among heterogeneous robot teams to resolve unforeseen conflicts. To address this challenge, we propose a novel decentralized framework that enables robots to request and provide help. The process begins when a robot detects a conflict and uses a Large Language Model (LLM) to decide whether external assistance is required. If so, it crafts and broadcasts a natural language (NL) help request. Potential helper robots reason over the request and respond with offers of assistance, including information about the effect on their ongoing tasks. Helper reasoning is implemented via an LLM grounded in Signal Temporal Logic (STL) using a Backus-Naur Form (BNF) grammar, ensuring syntactically valid NL-to-STL translations, which are then solved as a Mixed Integer Linear Program (MILP). Finally, the requester robot selects a helper by reasoning over the expected increase in system-level total task completion time. We evaluated our framework through experiments comparing different helper-selection strategies and found that considering multiple offers allows the requester to minimize added makespan. Our approach significantly outperforms heuristics such as selecting the nearest available candidate helper robot, and achieves performance comparable to a centralized "Oracle" baseline but without heavy information demands.
>
---
#### [new 080] Advancing Audio-Visual Navigation Through Multi-Agent Collaboration in 3D Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多智能体协同导航任务，解决动态3D环境中高效协作定位与导航问题。提出MASTAVN框架，通过通信与融合机制提升导航效率与成功率。**

- **链接: [http://arxiv.org/pdf/2509.22698v1](http://arxiv.org/pdf/2509.22698v1)**

> **作者:** Hailong Zhang; Yinfeng Yu; Liejun Wang; Fuchun Sun; Wendong Zheng
>
> **备注:** Main paper (15 pages). Accepted for publication by ICONIP( International Conference on Neural Information Processing) 2025
>
> **摘要:** Intelligent agents often require collaborative strategies to achieve complex tasks beyond individual capabilities in real-world scenarios. While existing audio-visual navigation (AVN) research mainly focuses on single-agent systems, their limitations emerge in dynamic 3D environments where rapid multi-agent coordination is critical, especially for time-sensitive applications like emergency response. This paper introduces MASTAVN (Multi-Agent Scalable Transformer Audio-Visual Navigation), a scalable framework enabling two agents to collaboratively localize and navigate toward an audio target in shared 3D environments. By integrating cross-agent communication protocols and joint audio-visual fusion mechanisms, MASTAVN enhances spatial reasoning and temporal synchronization. Through rigorous evaluation in photorealistic 3D simulators (Replica and Matterport3D), MASTAVN achieves significant reductions in task completion time and notable improvements in navigation success rates compared to single-agent and non-collaborative baselines. This highlights the essential role of spatiotemporal coordination in multi-agent systems. Our findings validate MASTAVN's effectiveness in time-sensitive emergency scenarios and establish a paradigm for advancing scalable multi-agent embodied intelligence in complex 3D environments.
>
---
#### [new 081] ReSeFlow: Rectifying SE(3)-Equivariant Policy Learning Flows
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决生成高效、鲁棒轨迹策略的问题。通过结合SE(3)等变性和修正流，提出ReSeFlow模型，提升政策生成效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.22695v1](http://arxiv.org/pdf/2509.22695v1)**

> **作者:** Zhitao Wang; Yanke Wang; Jiangtao Wen; Roberto Horowitz; Yuxing Han
>
> **备注:** This work was submitted to 2026 IEEE International Conference on Robotics & Automation
>
> **摘要:** Robotic manipulation in unstructured environments requires the generation of robust and long-horizon trajectory-level policy with conditions of perceptual observations and benefits from the advantages of SE(3)-equivariant diffusion models that are data-efficient. However, these models suffer from the inference time costs. Inspired by the inference efficiency of rectified flows, we introduce the rectification to the SE(3)-diffusion models and propose the ReSeFlow, i.e., Rectifying SE(3)-Equivariant Policy Learning Flows, providing fast, geodesic-consistent, least-computational policy generation. Crucially, both components employ SE(3)-equivariant networks to preserve rotational and translational symmetry, enabling robust generalization under rigid-body motions. With the verification on the simulated benchmarks, we find that the proposed ReSeFlow with only one inference step can achieve better performance with lower geodesic distance than the baseline methods, achieving up to a 48.5% error reduction on the painting task and a 21.9% reduction on the rotating triangle task compared to the baseline's 100-step inference. This method takes advantages of both SE(3) equivariance and rectified flow and puts it forward for the real-world application of generative policy learning models with the data and inference efficiency.
>
---
#### [new 082] Focusing on What Matters: Object-Agent-centric Tokenization for Vision Language Action models
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于视觉语言动作模型任务，旨在降低计算成本。针对视觉输入的分词方案问题，提出Oat-VLA方法，通过对象-代理中心化分词减少视觉标记数量，提升训练效率与性能。**

- **链接: [http://arxiv.org/pdf/2509.23655v1](http://arxiv.org/pdf/2509.23655v1)**

> **作者:** Rokas Bendikas; Daniel Dijkman; Markus Peschl; Sanjay Haresh; Pietro Mazzaglia
>
> **备注:** Presented at 9th Conference on Robot Learning (CoRL 2025), Seoul, Korea
>
> **摘要:** Vision-Language-Action (VLA) models offer a pivotal approach to learning robotic manipulation at scale by repurposing large pre-trained Vision-Language-Models (VLM) to output robotic actions. However, adapting VLMs for robotic domains comes with an unnecessarily high computational cost, which we attribute to the tokenization scheme of visual inputs. In this work, we aim to enable efficient VLA training by proposing Oat-VLA, an Object-Agent-centric Tokenization for VLAs. Building on the insights of object-centric representation learning, our method introduces an inductive bias towards scene objects and the agent's own visual information. As a result, we find that Oat-VLA can drastically reduce the number of visual tokens to just a few tokens without sacrificing performance. We reveal that Oat-VLA converges at least twice as fast as OpenVLA on the LIBERO suite, as well as outperform OpenVLA in diverse real-world pick and place tasks.
>
---
#### [new 083] MDCPP: Multi-robot Dynamic Coverage Path Planning for Workload Adaptation
- **分类: cs.RO**

- **简介: 该论文属于多机器人路径规划任务，解决传统方法在动态工作负载下的效率问题。通过动态估计工作量并优化区域分配，提升覆盖效率。**

- **链接: [http://arxiv.org/pdf/2509.23705v1](http://arxiv.org/pdf/2509.23705v1)**

> **作者:** Jun Chen; Mingjia Chen; Shinkyu Park
>
> **摘要:** Multi-robot Coverage Path Planning (MCPP) addresses the problem of computing paths for multiple robots to effectively cover a large area of interest. Conventional approaches to MCPP typically assume that robots move at fixed velocities, which is often unrealistic in real-world applications where robots must adapt their speeds based on the specific coverage tasks assigned to them.Consequently, conventional approaches often lead to imbalanced workload distribution among robots and increased completion time for coverage tasks. To address this, we introduce a novel Multi-robot Dynamic Coverage Path Planning (MDCPP) algorithm for complete coverage in two-dimensional environments. MDCPP dynamically estimates each robot's remaining workload by approximating the target distribution with Gaussian mixture models, and assigns coverage regions using a capacity-constrained Voronoi diagram. We further develop a distributed implementation of MDCPP for range-constrained robotic networks. Simulation results validate the efficacy of MDCPP, showing qualitative improvements and superior performance compared to an existing sweeping algorithm, and a quantifiable impact of communication range on coverage efficiency.
>
---
#### [new 084] Multi-Modal Manipulation via Multi-Modal Policy Consensus
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人操作任务，解决多模态感知融合问题。通过分解策略为专用扩散模型并使用路由网络自适应整合，提升多模态推理能力与系统鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.23468v1](http://arxiv.org/pdf/2509.23468v1)**

> **作者:** Haonan Chen; Jiaming Xu; Hongyu Chen; Kaiwen Hong; Binghao Huang; Chaoqi Liu; Jiayuan Mao; Yunzhu Li; Yilun Du; Katherine Driggs-Campbell
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Effectively integrating diverse sensory modalities is crucial for robotic manipulation. However, the typical approach of feature concatenation is often suboptimal: dominant modalities such as vision can overwhelm sparse but critical signals like touch in contact-rich tasks, and monolithic architectures cannot flexibly incorporate new or missing modalities without retraining. Our method factorizes the policy into a set of diffusion models, each specialized for a single representation (e.g., vision or touch), and employs a router network that learns consensus weights to adaptively combine their contributions, enabling incremental of new representations. We evaluate our approach on simulated manipulation tasks in {RLBench}, as well as real-world tasks such as occluded object picking, in-hand spoon reorientation, and puzzle insertion, where it significantly outperforms feature-concatenation baselines on scenarios requiring multimodal reasoning. Our policy further demonstrates robustness to physical perturbations and sensor corruption. We further conduct perturbation-based importance analysis, which reveals adaptive shifts between modalities.
>
---
#### [new 085] EKF-Based Fusion of Wi-Fi/LiDAR/IMU for Indoor Localization and Navigation
- **分类: cs.RO; cs.LG; cs.NI**

- **简介: 该论文属于室内定位与导航任务，解决传统Wi-Fi和LiDAR/IMU定位精度低、不稳定的问题，通过EKF融合多传感器数据提升定位性能。**

- **链接: [http://arxiv.org/pdf/2509.23118v1](http://arxiv.org/pdf/2509.23118v1)**

> **作者:** Zeyi Li; Zhe Tang; Kyeong Soo Kim; Sihao Li; Jeremy S. Smith
>
> **备注:** 8 pages, 7 figures, 3 tables, and submitted for presentation at a conference
>
> **摘要:** Conventional Wi-Fi received signal strength indicator (RSSI) fingerprinting cannot meet the growing demand for accurate indoor localization and navigation due to its lower accuracy, while solutions based on light detection and ranging (LiDAR) can provide better localization performance but is limited by their higher deployment cost and complexity. To address these issues, we propose a novel indoor localization and navigation framework integrating Wi-Fi RSSI fingerprinting, LiDAR-based simultaneous localization and mapping (SLAM), and inertial measurement unit (IMU) navigation based on an extended Kalman filter (EKF). Specifically, coarse localization by deep neural network (DNN)-based Wi-Fi RSSI fingerprinting is refined by IMU-based dynamic positioning using a Gmapping-based SLAM to generate an occupancy grid map and output high-frequency attitude estimates, which is followed by EKF prediction-update integrating sensor information while effectively suppressing Wi-Fi-induced noise and IMU drift errors. Multi-group real-world experiments conducted on the IR building at Xi'an Jiaotong-Liverpool University demonstrates that the proposed multi-sensor fusion framework suppresses the instability caused by individual approaches and thereby provides stable accuracy across all path configurations with mean two-dimensional (2D) errors ranging from 0.2449 m to 0.3781 m. In contrast, the mean 2D errors of Wi-Fi RSSI fingerprinting reach up to 1.3404 m in areas with severe signal interference, and those of LiDAR/IMU localization are between 0.6233 m and 2.8803 m due to cumulative drift.
>
---
#### [new 086] Physically-Feasible Reactive Synthesis for Terrain-Adaptive Locomotion
- **分类: cs.RO**

- **简介: 该论文属于四足机器人运动规划任务，解决动态地形下足点选择与轨迹优化问题。通过结合符号合成与混合整数凸规划，实现安全、自适应的 locomotion 控制。**

- **链接: [http://arxiv.org/pdf/2509.23185v1](http://arxiv.org/pdf/2509.23185v1)**

> **作者:** Ziyi Zhou; Qian Meng; Hadas Kress-Gazit; Ye Zhao
>
> **摘要:** We present an integrated planning framework for quadrupedal locomotion over dynamically changing, unforeseen terrains. Existing methods often depend on heuristics for real-time foothold selection-limiting robustness and adaptability-or rely on computationally intensive trajectory optimization across complex terrains and long horizons. In contrast, our approach combines reactive synthesis for generating correct-by-construction symbolic-level controllers with mixed-integer convex programming (MICP) for dynamic and physically feasible footstep planning during each symbolic transition. To reduce the reliance on costly MICP solves and accommodate specifications that may be violated due to physical infeasibility, we adopt a symbolic repair mechanism that selectively generates only the required symbolic transitions. During execution, real-time MICP replanning based on actual terrain data, combined with runtime symbolic repair and delay-aware coordination, enables seamless bridging between offline synthesis and online operation. Through extensive simulation and hardware experiments, we validate the framework's ability to identify missing locomotion skills and respond effectively in safety-critical environments, including scattered stepping stones and rebar scenarios.
>
---
#### [new 087] DA-MMP: Learning Coordinated and Accurate Throwing with Dynamics-Aware Motion Manifold Primitives
- **分类: cs.RO**

- **简介: 该论文属于动态操作任务，旨在解决轨迹规划与执行间的动态差异问题。提出DA-MMP框架，通过学习运动流形生成协调准确的投掷轨迹。**

- **链接: [http://arxiv.org/pdf/2509.23721v1](http://arxiv.org/pdf/2509.23721v1)**

> **作者:** Chi Chu; Huazhe Xu
>
> **摘要:** Dynamic manipulation is a key capability for advancing robot performance, enabling skills such as tossing. While recent learning-based approaches have pushed the field forward, most methods still rely on manually designed action parameterizations, limiting their ability to produce the highly coordinated motions required in complex tasks. Motion planning can generate feasible trajectories, but the dynamics gap-stemming from control inaccuracies, contact uncertainties, and aerodynamic effects-often causes large deviations between planned and executed trajectories. In this work, we propose Dynamics-Aware Motion Manifold Primitives (DA-MMP), a motion generation framework for goal-conditioned dynamic manipulation, and instantiate it on a challenging real-world ring-tossing task. Our approach extends motion manifold primitives to variable-length trajectories through a compact parametrization and learns a high-quality manifold from a large-scale dataset of planned motions. Building on this manifold, a conditional flow matching model is trained in the latent space with a small set of real-world trials, enabling the generation of throwing trajectories that account for execution dynamics. Experiments show that our method can generate coordinated and smooth motion trajectories for the ring-tossing task. In real-world evaluations, it achieves high success rates and even surpasses the performance of trained human experts. Moreover, it generalizes to novel targets beyond the training range, indicating that it successfully learns the underlying trajectory-dynamics mapping.
>
---
#### [new 088] PhysiAgent: An Embodied Agent Framework in Physical World
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文属于机器人任务执行领域，解决VLA与VLM协作不足的问题。提出PhysiAgent框架，通过自监督机制提升任务执行效果。**

- **链接: [http://arxiv.org/pdf/2509.24524v1](http://arxiv.org/pdf/2509.24524v1)**

> **作者:** Zhihao Wang; Jianxiong Li; Jinliang Zheng; Wencong Zhang; Dongxiu Liu; Yinan Zheng; Haoyi Niu; Junzhi Yu; Xianyuan Zhan
>
> **摘要:** Vision-Language-Action (VLA) models have achieved notable success but often struggle with limited generalizations. To address this, integrating generalized Vision-Language Models (VLMs) as assistants to VLAs has emerged as a popular solution. However, current approaches often combine these models in rigid, sequential structures: using VLMs primarily for high-level scene understanding and task planning, and VLAs merely as executors of lower-level actions, leading to ineffective collaboration and poor grounding challenges. In this paper, we propose an embodied agent framework, PhysiAgent, tailored to operate effectively in physical environments. By incorporating monitor, memory, self-reflection mechanisms, and lightweight off-the-shelf toolboxes, PhysiAgent offers an autonomous scaffolding framework to prompt VLMs to organize different components based on real-time proficiency feedback from VLAs to maximally exploit VLAs' capabilities. Experimental results demonstrate significant improvements in task-solving performance on complex real-world robotic tasks, showcasing effective self-regulation of VLMs, coherent tool collaboration, and adaptive evolution of the framework during execution. PhysiAgent makes practical and pioneering efforts to integrate VLMs and VLAs, effectively grounding embodied agent frameworks in real-world settings.
>
---
#### [new 089] BOSfM: A View Planning Framework for Optimal 3D Reconstruction of Agricultural Scenes
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于农业场景3D重建任务，解决如何通过优化相机位置以最少图像实现高质量重建的问题。提出BOSfM框架，利用贝叶斯优化应对噪声和环境泛化挑战。**

- **链接: [http://arxiv.org/pdf/2509.24126v1](http://arxiv.org/pdf/2509.24126v1)**

> **作者:** Athanasios Bacharis; Konstantinos D. Polyzos; Georgios B. Giannakis; Nikolaos Papanikolopoulos
>
> **摘要:** Active vision (AV) has been in the spotlight of robotics research due to its emergence in numerous applications including agricultural tasks such as precision crop monitoring and autonomous harvesting to list a few. A major AV problem that gained popularity is the 3D reconstruction of targeted environments using 2D images from diverse viewpoints. While collecting and processing a large number of arbitrarily captured 2D images can be arduous in many practical scenarios, a more efficient solution involves optimizing the placement of available cameras in 3D space to capture fewer, yet more informative, images that provide sufficient visual information for effective reconstruction of the environment of interest. This process termed as view planning (VP), can be markedly challenged (i) by noise emerging in the location of the cameras and/or in the extracted images, and (ii) by the need to generalize well in other unknown similar agricultural environments without need for re-optimizing or re-training. To cope with these challenges, the present work presents a novel VP framework that considers a reconstruction quality-based optimization formulation that relies on the notion of `structure-from-motion' to reconstruct the 3D structure of the sought environment from the selected 2D images. With no analytic expression of the optimization function and with costly function evaluations, a Bayesian optimization approach is proposed to efficiently carry out the VP process using only a few function evaluations, while accounting for different noise cases. Numerical tests on both simulated and real agricultural settings signify the benefits of the advocated VP approach in efficiently estimating the optimal camera placement to accurately reconstruct 3D environments of interest, and generalize well on similar unknown environments.
>
---
#### [new 090] Crop Spirals: Re-thinking the field layout for future robotic agriculture
- **分类: cs.RO**

- **简介: 该论文属于自主农业任务，旨在解决机器人在传统农田布局中导航效率低的问题。提出螺旋布局与新型导航算法，提升路径效率和多机器人协作性能。**

- **链接: [http://arxiv.org/pdf/2509.25091v1](http://arxiv.org/pdf/2509.25091v1)**

> **作者:** Lakshan Lavan; Lanojithan Thiyagarasa; Udara Muthugala; Rajitha de Silva
>
> **备注:** Submitted to Computers and Electronics in Agriculture
>
> **摘要:** Conventional linear crop layouts, optimised for tractors, hinder robotic navigation with tight turns, long travel distances, and perceptual aliasing. We propose a robot-centric square spiral layout with a central tramline, enabling simpler motion and more efficient coverage. To exploit this geometry, we develop a navigation stack combining DH-ResNet18 waypoint regression, pixel-to-odometry mapping, A* planning, and model predictive control (MPC). In simulations, the spiral layout yields up to 28% shorter paths and about 25% faster execution for waypoint-based tasks across 500 waypoints than linear layouts, while full-field coverage performance is comparable to an optimised linear U-turn strategy. Multi-robot studies demonstrate efficient coordination on the spirals rule-constrained graph, with a greedy allocator achieving 33-37% lower batch completion times than a Hungarian assignment under our setup. These results highlight the potential of redesigning field geometry to better suit autonomous agriculture.
>
---
#### [new 091] Robust Orientation Estimation with TRIAD-aided Manifold EKF
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于姿态估计任务，旨在解决磁力计受干扰的问题。通过将TRIAD算法融入Manifold EKF，提升姿态估计的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.23456v1](http://arxiv.org/pdf/2509.23456v1)**

> **作者:** Arjun Sadananda; Ravi Banavar; Kavi Arya
>
> **摘要:** The manifold extended Kalman filter (Manifold EKF) has found extensive application for attitude determination. Magnetometers employed as sensors for such attitude determination are easily prone to disturbances by their sensitivity to calibration and external magnetic fields. The TRIAD (Tri-Axial Attitude Determination) algorithm is well known as a sub-optimal attitude estimator. In this article, we incorporate this sub-optimal feature of the TRIAD in mitigating the influence of the magnetometer reading in the pitch and roll axis determination in the Manifold EKF algorithm. We substantiate our results with experiments.
>
---
#### [new 092] Encoding Material Safety using Control Barrier Functions for Soft Actuator Control
- **分类: cs.RO**

- **简介: 该论文属于软体机器人控制任务，旨在解决材料安全性问题。通过控制屏障函数确保软致动器在操作中的安全。**

- **链接: [http://arxiv.org/pdf/2509.23623v1](http://arxiv.org/pdf/2509.23623v1)**

> **作者:** Nicholas Pagliocca; Behrad Koohbor; Mitja Trkov
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Until recently, the concept of soft robot safety was an informal notion, often attributed solely to the fact that soft robots are less likely to damage their operating environment than rigid robots. As the field moves toward feedback control for practical applications, it becomes increasingly important to define what safety means and to characterize how soft robots can become unsafe. The unifying theme of soft robotics is to achieve useful functionality through deformation. Consequently, limitations in constitutive model accuracy and risks of material failure are inherent to all soft robots and pose a key challenge in designing provably safe controllers. This work introduces a formal definition of material safety based on strain energy functions and provides a controller that enforces it. We characterize safe and unsafe sets of an incompressible hyperelastic material and demonstrate that safety can be enforced using a high-order control barrier function (HOCBF) with quadratic program-based feedback control. As a case study, we consider a pressurized hyperelastic tube with inertial effects, first-order viscous effects, and full-state feedback. Simulation results verify that the proposed methodology can enforce the material safety specification.
>
---
#### [new 093] Prepare for Warp Speed: Sub-millisecond Visual Place Recognition Using Event Cameras
- **分类: cs.RO**

- **简介: 该论文属于视觉定位任务，解决传统VPR响应慢的问题。提出Flash系统，利用亚毫秒级事件数据实现快速位置识别，提升召回率并降低定位延迟。**

- **链接: [http://arxiv.org/pdf/2509.24094v1](http://arxiv.org/pdf/2509.24094v1)**

> **作者:** Vignesh Ramanathan; Michael Milford; Tobias Fischer
>
> **摘要:** Visual Place Recognition (VPR) enables systems to identify previously visited locations within a map, a fundamental task for autonomous navigation. Prior works have developed VPR solutions using event cameras, which asynchronously measure per-pixel brightness changes with microsecond temporal resolution. However, these approaches rely on dense representations of the inherently sparse camera output and require tens to hundreds of milliseconds of event data to predict a place. Here, we break this paradigm with Flash, a lightweight VPR system that predicts places using sub-millisecond slices of event data. Our method is based on the observation that active pixel locations provide strong discriminative features for VPR. Flash encodes these active pixel locations using efficient binary frames and computes similarities via fast bitwise operations, which are then normalized based on the relative event activity in the query and reference frames. Flash improves Recall@1 for sub-millisecond VPR over existing baselines by 11.33x on the indoor QCR-Event-Dataset and 5.92x on the 8 km Brisbane-Event-VPR dataset. Moreover, our approach reduces the duration for which the robot must operate without awareness of its position, as evidenced by a localization latency metric we term Time to Correct Match (TCM). To the best of our knowledge, this is the first work to demonstrate sub-millisecond VPR using event cameras.
>
---
#### [new 094] SAC-Loco: Safe and Adjustable Compliant Quadrupedal Locomotion
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于四足机器人运动控制任务，旨在解决机器人在外部扰动下缺乏适应性与安全性的问题。通过引入可调节合规策略和安全策略，提升机器人的稳定性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.23223v1](http://arxiv.org/pdf/2509.23223v1)**

> **作者:** Aoqian Zhang; Zixuan Zhuang; Chunzheng Wang; Shuzhi Sam Ge; Fan Shi; Cheng Xiang
>
> **摘要:** Quadruped robots are designed to achieve agile locomotion by mimicking legged animals. However, existing control methods for quadrupeds often lack one of the key capabilities observed in animals: adaptive and adjustable compliance in response to external disturbances. Most locomotion controllers do not provide tunable compliance and tend to fail under large perturbations. In this work, we propose a switched policy framework for compliant and safe quadruped locomotion. First, we train a force compliant policy with adjustable compliance levels using a teacher student reinforcement learning framework, eliminating the need for explicit force sensing. Next, we develop a safe policy based on the capture point concept to stabilize the robot when the compliant policy fails. Finally, we introduce a recoverability network that predicts the likelihood of failure and switches between the compliant and safe policies. Together, this framework enables quadruped robots to achieve both force compliance and robust safety when subjected to severe external disturbances.
>
---
#### [new 095] World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）模型的后训练任务，旨在解决数据稀缺、安全性和执行效率问题。提出World-Env框架，利用虚拟环境进行强化学习，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2509.24948v1](http://arxiv.org/pdf/2509.24948v1)**

> **作者:** Junjin Xiao; Yandan Yang; Xinyuan Chang; Ronghan Chen; Feng Xiong; Mu Xu; Wei-Shi Zheng; Qing Zhang
>
> **摘要:** Vision-Language-Action (VLA) models trained via imitation learning suffer from significant performance degradation in data-scarce scenarios due to their reliance on large-scale demonstration datasets. Although reinforcement learning (RL)-based post-training has proven effective in addressing data scarcity, its application to VLA models is hindered by the non-resettable nature of real-world environments. This limitation is particularly critical in high-risk domains such as industrial automation, where interactions often induce state changes that are costly or infeasible to revert. Furthermore, existing VLA approaches lack a reliable mechanism for detecting task completion, leading to redundant actions that reduce overall task success rates. To address these challenges, we propose World-Env, an RL-based post-training framework that replaces physical interaction with a low-cost, world model-based virtual simulator. World-Env consists of two key components: (1) a video-based world simulator that generates temporally consistent future visual observations, and (2) a vision-language model (VLM)-guided instant reflector that provides continuous reward signals and predicts action termination. This simulated environment enables VLA models to safely explore and generalize beyond their initial imitation learning distribution. Our method achieves notable performance gains with as few as five expert demonstrations per task. Experiments on complex robotic manipulation tasks demonstrate that World-Env effectively overcomes the data inefficiency, safety constraints, and inefficient execution of conventional VLA models that rely on real-world interaction, offering a practical and scalable solution for post-training in resource-constrained settings.
>
---
#### [new 096] Very High Frequency Interpolation for Direct Torque Control
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决扭矩控制稳定性问题，通过高频线性反馈实现高精度扭矩控制。**

- **链接: [http://arxiv.org/pdf/2509.24175v1](http://arxiv.org/pdf/2509.24175v1)**

> **作者:** Rafael Kourdis; Maciej Stępień; Jérôme Manhes; Nicolas Mansard; Steve Tonneau; Philippe Souères; Thomas Flayols
>
> **摘要:** Torque control enables agile and robust robot motion, but deployment is often hindered by instability and hardware limits. Here, we present a novel solution to execute whole-body linear feedback at up to 40 kHz on open-source hardware. We use this to interpolate non-linear schemes during real-world execution, such as inverse dynamics and learned torque policies. Our results show that by stabilizing torque controllers, high-frequency linear feedback could be an effective route towards unlocking the potential of torque-controlled robotics.
>
---
#### [new 097] Mash, Spread, Slice! Learning to Manipulate Object States via Visual Spatial Progress
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，解决物体状态变化问题。提出SPARTA框架，通过视觉空间进展识别物体状态变化，提升操控精度与效率。**

- **链接: [http://arxiv.org/pdf/2509.24129v1](http://arxiv.org/pdf/2509.24129v1)**

> **作者:** Priyanka Mandikal; Jiaheng Hu; Shivin Dass; Sagnik Majumder; Roberto Martín-Martín; Kristen Grauman
>
> **摘要:** Most robot manipulation focuses on changing the kinematic state of objects: picking, placing, opening, or rotating them. However, a wide range of real-world manipulation tasks involve a different class of object state change--such as mashing, spreading, or slicing--where the object's physical and visual state evolve progressively without necessarily changing its position. We present SPARTA, the first unified framework for the family of object state change manipulation tasks. Our key insight is that these tasks share a common structural pattern: they involve spatially-progressing, object-centric changes that can be represented as regions transitioning from an actionable to a transformed state. Building on this insight, SPARTA integrates spatially progressing object change segmentation maps, a visual skill to perceive actionable vs. transformed regions for specific object state change tasks, to generate a) structured policy observations that strip away appearance variability, and b) dense rewards that capture incremental progress over time. These are leveraged in two SPARTA policy variants: reinforcement learning for fine-grained control without demonstrations or simulation; and greedy control for fast, lightweight deployment. We validate SPARTA on a real robot for three challenging tasks across 10 diverse real-world objects, achieving significant improvements in training time and accuracy over sparse rewards and visual goal-conditioned baselines. Our results highlight progress-aware visual representations as a versatile foundation for the broader family of object state manipulation tasks. Project website: https://vision.cs.utexas.edu/projects/sparta-robot
>
---
#### [new 098] DRCP: Diffusion on Reinforced Cooperative Perception for Perceiving Beyond Limits
- **分类: cs.RO; cs.CV; eess.IV**

- **简介: 该论文属于自动驾驶中的协同感知任务，旨在解决动态环境中检测精度不足的问题。提出DRCP框架，结合跨模态融合与轻量扩散模块，提升感知鲁棒性与实时性。**

- **链接: [http://arxiv.org/pdf/2509.24903v1](http://arxiv.org/pdf/2509.24903v1)**

> **作者:** Lantao Li; Kang Yang; Rui Song; Chen Sun
>
> **摘要:** Cooperative perception enabled by Vehicle-to-Everything communication has shown great promise in enhancing situational awareness for autonomous vehicles and other mobile robotic platforms. Despite recent advances in perception backbones and multi-agent fusion, real-world deployments remain challenged by hard detection cases, exemplified by partial detections and noise accumulation which limit downstream detection accuracy. This work presents Diffusion on Reinforced Cooperative Perception (DRCP), a real-time deployable framework designed to address aforementioned issues in dynamic driving environments. DRCP integrates two key components: (1) Precise-Pyramid-Cross-Modality-Cross-Agent, a cross-modal cooperative perception module that leverages camera-intrinsic-aware angular partitioning for attention-based fusion and adaptive convolution to better exploit external features; and (2) Mask-Diffusion-Mask-Aggregation, a novel lightweight diffusion-based refinement module that encourages robustness against feature perturbations and aligns bird's-eye-view features closer to the task-optimal manifold. The proposed system achieves real-time performance on mobile platforms while significantly improving robustness under challenging conditions. Code will be released in late 2025.
>
---
#### [new 099] Fostering Robots: A Governance-First Conceptual Framework for Domestic, Curriculum-Based Trajectory Collection
- **分类: cs.RO**

- **简介: 该论文属于机器人部署任务，旨在解决家庭机器人长期交互轨迹的治理问题，提出一种以治理为核心的框架，确保轨迹质量符合欧盟标准。**

- **链接: [http://arxiv.org/pdf/2509.23821v1](http://arxiv.org/pdf/2509.23821v1)**

> **作者:** Federico Pablo-Marti; Carlos Mir Fernandez
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** We propose a conceptual, empirically testable framework for Robot Fostering, -a curriculum-driven, governance-first approach to domestic robot deployments, emphasizing long-term, curated interaction trajectories. We formalize trajectory quality with quantifiable metrics and evaluation protocols aligned with EU-grade governance standards, delineating a low-resource empirical roadmap to enable rigorous validation through future pilot studies.
>
---
#### [new 100] Space Robotics Bench: Robot Learning Beyond Earth
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于空间机器人学习任务，旨在解决太空环境中自主系统训练数据不足与成本高的问题。作者提出了Space Robotics Bench仿真框架，支持大规模多样化训练与性能评估。**

- **链接: [http://arxiv.org/pdf/2509.23328v1](http://arxiv.org/pdf/2509.23328v1)**

> **作者:** Andrej Orsula; Matthieu Geist; Miguel Olivares-Mendez; Carol Martinez
>
> **备注:** The source code is available at https://github.com/AndrejOrsula/space_robotics_bench
>
> **摘要:** The growing ambition for space exploration demands robust autonomous systems that can operate in unstructured environments under extreme extraterrestrial conditions. The adoption of robot learning in this domain is severely hindered by the prohibitive cost of technology demonstrations and the limited availability of data. To bridge this gap, we introduce the Space Robotics Bench, an open-source simulation framework for robot learning in space. It offers a modular architecture that integrates on-demand procedural generation with massively parallel simulation environments to support the creation of vast and diverse training distributions for learning-based agents. To ground research and enable direct comparison, the framework includes a comprehensive suite of benchmark tasks that span a wide range of mission-relevant scenarios. We establish performance baselines using standard reinforcement learning algorithms and present a series of experimental case studies that investigate key challenges in generalization, end-to-end learning, adaptive control, and sim-to-real transfer. Our results reveal insights into the limitations of current methods and demonstrate the utility of the framework in producing policies capable of real-world operation. These contributions establish the Space Robotics Bench as a valuable resource for developing, benchmarking, and deploying the robust autonomous systems required for the final frontier.
>
---
#### [new 101] PROFusion: Robust and Accurate Dense Reconstruction via Camera Pose Regression and Optimization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，解决不稳定运动下密集重建精度不足的问题。通过结合学习初始化与优化精修，提升重建鲁棒性与准确性。**

- **链接: [http://arxiv.org/pdf/2509.24236v1](http://arxiv.org/pdf/2509.24236v1)**

> **作者:** Siyan Dong; Zijun Wang; Lulu Cai; Yi Ma; Yanchao Yang
>
> **摘要:** Real-time dense scene reconstruction during unstable camera motions is crucial for robotics, yet current RGB-D SLAM systems fail when cameras experience large viewpoint changes, fast motions, or sudden shaking. Classical optimization-based methods deliver high accuracy but fail with poor initialization during large motions, while learning-based approaches provide robustness but lack sufficient accuracy for dense reconstruction. We address this challenge through a combination of learning-based initialization with optimization-based refinement. Our method employs a camera pose regression network to predict metric-aware relative poses from consecutive RGB-D frames, which serve as reliable starting points for a randomized optimization algorithm that further aligns depth images with the scene geometry. Extensive experiments demonstrate promising results: our approach outperforms the best competitor on challenging benchmarks, while maintaining comparable accuracy on stable motion sequences. The system operates in real-time, showcasing that combining simple and principled techniques can achieve both robustness for unstable motions and accuracy for dense reconstruction. Project page: https://github.com/siyandong/PROFusion.
>
---
#### [new 102] DBF-MA: A Differential Bayesian Filtering Planner for Multi-Agent Autonomous Racing Overtakes
- **分类: cs.RO**

- **简介: 该论文属于多智能体自主赛车任务，解决复杂赛道上的超车问题。提出DBF-MA方法，基于贝叶斯滤波框架生成无碰撞轨迹，无需简化假设，有效提升超车成功率。**

- **链接: [http://arxiv.org/pdf/2509.22937v1](http://arxiv.org/pdf/2509.22937v1)**

> **作者:** Trent Weiss; Amar Kulkarni; Madhur Behl
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** A significant challenge in autonomous racing is to generate overtaking maneuvers. Racing agents must execute these maneuvers on complex racetracks with little room for error. Optimization techniques and graph-based methods have been proposed, but these methods often rely on oversimplified assumptions for collision-avoidance and dynamic constraints. In this work, we present an approach to trajectory synthesis based on an extension of the Differential Bayesian Filtering framework. Our approach for collision-free trajectory synthesis frames the problem as one of Bayesian Inference over the space of Composite Bezier Curves. Our method is derivative-free, does not require a spherical approximation of the vehicle footprint, linearization of constraints, or simplifying upper bounds on collision avoidance. We conduct a closed-loop analysis of DBF-MA and find it successfully overtakes an opponent in 87% of tested scenarios, outperforming existing methods in autonomous overtaking.
>
---
#### [new 103] Large Language Models for 3D IC Space Planning
- **分类: cs.RO**

- **简介: 该论文属于3D IC布局规划任务，旨在解决空间利用率与布局合法性问题。通过LLM和后序分割树方法，实现高效、无死区的布局设计。**

- **链接: [http://arxiv.org/pdf/2509.22716v1](http://arxiv.org/pdf/2509.22716v1)**

> **作者:** Hung-Ying Chu; Guan-Wei Chen; Shao-Yu Wei; Yu-Cheng Lin
>
> **备注:** Accepted at AICCC 2025
>
> **摘要:** Three-dimensional integrated circuits (3D ICs) have emerged as a promising solution to the scaling limits of two-dimensional designs, offering higher integration density, shorter interconnects, and improved performance. As design complexity increases, effective space planning becomes essential to reduce dead space and ensure layout quality. This study investigates the use of large language models (LLMs) for 3D IC space planning through a post-order slicing tree representation, which guarantees legal space plans while aiming to minimize dead space. Open-source LLMs were fine-tuned on large-scale synthetic datasets and further evaluated on MCNC-derived 3D benchmarks. Experimental results indicate that the proposed framework achieves a favorable balance between runtime efficiency, legality, and dead-space reduction, with zero-dead-space layouts obtained in a significant portion of test cases under practical runtime budgets. Beyond synthetic benchmarks, the method generalizes to MCNC cases such as ami33 and ami49, though larger and irregular instances remain challenging. The approach also shows potential for cross-domain applications, including logistics and 3D object placement, where spatial efficiency is critical. Overall, the results suggest that LLM-based space planning can serve as a data-driven complement to traditional electronic design automation (EDA) methods, providing new insights for scalable 3D layout generation.
>
---
#### [new 104] Dynamic Buffers: Cost-Efficient Planning for Tabletop Rearrangement with Stacking
- **分类: cs.RO; cs.AI; I.2.9; I.2.8**

- **简介: 该论文属于机器人操作任务，解决杂乱桌面上物体重排的问题。通过引入动态缓冲机制，提升重排效率与可行性。**

- **链接: [http://arxiv.org/pdf/2509.22828v1](http://arxiv.org/pdf/2509.22828v1)**

> **作者:** Arman Barghi; Hamed Hosseini; Seraj Ghasemi; Mehdi Tale Masouleh; Ahmad Kalhor
>
> **摘要:** Rearranging objects in cluttered tabletop environments remains a long-standing challenge in robotics. Classical planners often generate inefficient, high-cost plans by shuffling objects individually and using fixed buffers--temporary spaces such as empty table regions or static stacks--to resolve conflicts. When only free table locations are used as buffers, dense scenes become inefficient, since placing an object can restrict others from reaching their goals and complicate planning. Allowing stacking provides extra buffer capacity, but conventional stacking is static: once an object supports another, the base cannot be moved, which limits efficiency. To overcome these issues, a novel planning primitive called the Dynamic Buffer is introduced. Inspired by human grouping strategies, it enables robots to form temporary, movable stacks that can be transported as a unit. This improves both feasibility and efficiency in dense layouts, and it also reduces travel in large-scale settings where space is abundant. Compared with a state-of-the-art rearrangement planner, the approach reduces manipulator travel cost by 11.89% in dense scenarios with a stationary robot and by 5.69% in large, low-density settings with a mobile manipulator. Practicality is validated through experiments on a Delta parallel robot with a two-finger gripper. These findings establish dynamic buffering as a key primitive for cost-efficient and robust rearrangement planning.
>
---
#### [new 105] Preference-Based Long-Horizon Robotic Stacking with Multimodal Large Language Models
- **分类: cs.RO**

- **简介: 该论文属于长期机器人堆叠任务，解决物理属性推理不足的问题。通过多模态LLM和定制数据集优化堆叠顺序，提升堆叠效果。**

- **链接: [http://arxiv.org/pdf/2509.24163v1](http://arxiv.org/pdf/2509.24163v1)**

> **作者:** Wanming Yu; Adrian Röfer; Abhinav Valada; Sethu Vijayakumar
>
> **摘要:** Pretrained large language models (LLMs) can work as high-level robotic planners by reasoning over abstract task descriptions and natural language instructions, etc. However, they have shown a lack of knowledge and effectiveness in planning long-horizon robotic manipulation tasks where the physical properties of the objects are essential. An example is the stacking of containers with hidden objects inside, which involves reasoning over hidden physics properties such as weight and stability. To this end, this paper proposes to use multimodal LLMs as high-level planners for such long-horizon robotic stacking tasks. The LLM takes multimodal inputs for each object to stack and infers the current best stacking sequence by reasoning over stacking preferences. Furthermore, in order to enable the LLM to reason over multiple preferences at the same time without giving explicit instructions, we propose to create a custom dataset considering stacking preferences including weight, stability, size, and footprint, to fine-tune the LLM. Compared to the pretrained LLM with prompt tuning, we demonstrate the improved stacking completion of the LLM fine-tuned with our custom dataset via large-scale simulation evaluation. Furthermore, we showcase the effectiveness of the proposed framework for the long-horizon stacking task on a real humanoid robot in an online manner.
>
---
#### [new 106] A Novel Narrow Region Detector for Sampling-Based Planners' Efficiency: Match Based Passage Identifier
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决窄通道环境下采样效率低的问题。提出一种基于占用网格的新型采样器，提升窄区采样精度与效率。**

- **链接: [http://arxiv.org/pdf/2509.23288v1](http://arxiv.org/pdf/2509.23288v1)**

> **作者:** Yafes Enes Şahiner; Esat Yusuf Gündoğdu; Volkan Sezer
>
> **摘要:** Autonomous technology, which has become widespread today, appears in many different configurations such as mobile robots, manipulators, and drones. One of the most important tasks of these vehicles during autonomous operations is path planning. In the literature, path planners are generally divided into two categories: probabilistic and deterministic methods. In the analysis of probabilistic methods, the common problem of almost all methods is observed in narrow passage environments. In this paper, a novel sampler is proposed that deterministically identifies narrow passage environments using occupancy grid maps and accordingly increases the amount of sampling in these regions. The codes of the algorithm is provided as open source. To evaluate the performance of the algorithm, benchmark studies are conducted in three distinct categories: specific and random simulation environments, and a real-world environment. As a result, it is observed that our algorithm provides higher performance in planning time and number of milestones compared to the baseline samplers.
>
---
#### [new 107] Parameter Identification of a Differentiable Human Arm Musculoskeletal Model without Deep Muscle EMG Reconstruction
- **分类: cs.RO**

- **简介: 该论文属于人体肌骨模型参数识别任务，旨在无需深部肌肉EMG重建的情况下准确识别模型参数，通过优化方法提升方法的可靠性和实用性。**

- **链接: [http://arxiv.org/pdf/2509.22825v1](http://arxiv.org/pdf/2509.22825v1)**

> **作者:** Philip Sanderink; Yingfan Zhou; Shuzhen Luo; Cheng Fang
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Accurate parameter identification of a subject-specific human musculoskeletal model is crucial to the development of safe and reliable physically collaborative robotic systems, for instance, assistive exoskeletons. Electromyography (EMG)-based parameter identification methods have demonstrated promising performance for personalized musculoskeletal modeling, whereas their applicability is limited by the difficulty of measuring deep muscle EMGs invasively. Although several strategies have been proposed to reconstruct deep muscle EMGs or activations for parameter identification, their reliability and robustness are limited by assumptions about the deep muscle behavior. In this work, we proposed an approach to simultaneously identify the bone and superficial muscle parameters of a human arm musculoskeletal model without reconstructing the deep muscle EMGs. This is achieved by only using the least-squares solution of the deep muscle forces to calculate a loss gradient with respect to the model parameters for identifying them in a framework of differentiable optimization. The results of extensive comparative simulations manifested that our proposed method can achieve comparable estimation accuracy compared to a similar method, but with all the muscle EMGs available.
>
---
#### [new 108] Multi-Robot Allocation for Information Gathering in Non-Uniform Spatiotemporal Environments
- **分类: cs.RO**

- **简介: 该论文属于多机器人信息采集任务，解决非均匀时空环境中传感器部署问题。通过两阶段框架优化机器人分配，提升场估计精度。**

- **链接: [http://arxiv.org/pdf/2509.22883v1](http://arxiv.org/pdf/2509.22883v1)**

> **作者:** Kaleb Ben Naveed; Haejoon Lee; Dimitra Panagou
>
> **备注:** Submitted to American Control Conference (ACC) 2026
>
> **摘要:** Autonomous robots are increasingly deployed to estimate spatiotemporal fields (e.g., wind, temperature, gas concentration) that vary across space and time. We consider environments divided into non-overlapping regions with distinct spatial and temporal dynamics, termed non-uniform spatiotemporal environments. Gaussian Processes (GPs) can be used to estimate these fields. The GP model depends on a kernel that encodes how the field co-varies in space and time, with its spatial and temporal lengthscales defining the correlation. Hence, when these lengthscales are incorrect or do not correspond to the actual field, the estimates of uncertainty can be highly inaccurate. Existing GP methods often assume one global lengthscale or update only periodically; some allow spatial variation but ignore temporal changes. To address these limitations, we propose a two-phase framework for multi-robot field estimation. Phase 1 uses a variogram-driven planner to learn region-specific spatial lengthscales. Phase 2 employs an allocation strategy that reassigns robots based on the current uncertainty, and updates sampling as temporal lengthscales are refined. For encoding uncertainty, we utilize clarity, an information metric from our earlier work. We evaluate the proposed method across diverse environments and provide convergence analysis for spatial lengthscale estimation, along with dynamic regret bounds quantifying the gap to the oracle's allocation sequence.
>
---
#### [new 109] Evaluation of Polarimetric Fusion for Semantic Segmentation in Aquatic Environments
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于语义分割任务，旨在解决水体表面反光导致的漂浮物分割困难问题。通过极化图像融合提升分割精度，但增加计算负担。**

- **链接: [http://arxiv.org/pdf/2509.24731v1](http://arxiv.org/pdf/2509.24731v1)**

> **作者:** Luis F. W. Batista; Tom Bourbon; Cedric Pradalier
>
> **备注:** Accepted to VCIP 2025
>
> **摘要:** Accurate segmentation of floating debris on water is often compromised by surface glare and changing outdoor illumination. Polarimetric imaging offers a single-sensor route to mitigate water-surface glare that disrupts semantic segmentation of floating objects. We benchmark state-of-the-art fusion networks on PoTATO, a public dataset of polarimetric images of plastic bottles in inland waterways, and compare their performance with single-image baselines using traditional models. Our results indicate that polarimetric cues help recover low-contrast objects and suppress reflection-induced false positives, raising mean IoU and lowering contour error relative to RGB inputs. These sharper masks come at a cost: the additional channels enlarge the models increasing the computational load and introducing the risk of new false positives. By providing a reproducible, diagnostic benchmark and publicly available code, we hope to help researchers choose if polarized cameras are suitable for their applications and to accelerate related research.
>
---
#### [new 110] FreeAction: Training-Free Techniques for Enhanced Fidelity of Trajectory-to-Video Generation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于轨迹到视频生成任务，旨在提升生成视频的准确性和质量。通过两种无需训练的推理技术，主动利用动作参数优化生成过程。**

- **链接: [http://arxiv.org/pdf/2509.24241v1](http://arxiv.org/pdf/2509.24241v1)**

> **作者:** Seungwook Kim; Seunghyeon Lee; Minsu Cho
>
> **备注:** 8 pages, 4 figures, accepted to CoRL 2025 LSRW workshop
>
> **摘要:** Generating realistic robot videos from explicit action trajectories is a critical step toward building effective world models and robotics foundation models. We introduce two training-free, inference-time techniques that fully exploit explicit action parameters in diffusion-based robot video generation. Instead of treating action vectors as passive conditioning signals, our methods actively incorporate them to guide both the classifier-free guidance process and the initialization of Gaussian latents. First, action-scaled classifier-free guidance dynamically modulates guidance strength in proportion to action magnitude, enhancing controllability over motion intensity. Second, action-scaled noise truncation adjusts the distribution of initially sampled noise to better align with the desired motion dynamics. Experiments on real robot manipulation datasets demonstrate that these techniques significantly improve action coherence and visual quality across diverse robot environments.
>
---
#### [new 111] FastViDAR: Real-Time Omnidirectional Depth Estimation via Alternative Hierarchical Attention
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于全景深度估计任务，解决多视角深度融合问题。提出AHA机制和ERP融合方法，实现高效实时360°深度图生成。**

- **链接: [http://arxiv.org/pdf/2509.23733v1](http://arxiv.org/pdf/2509.23733v1)**

> **作者:** Hangtian Zhao; Xiang Chen; Yizhe Li; Qianhao Wang; Haibo Lu; Fei Gao
>
> **摘要:** In this paper we propose FastViDAR, a novel framework that takes four fisheye camera inputs and produces a full $360^\circ$ depth map along with per-camera depth, fusion depth, and confidence estimates. Our main contributions are: (1) We introduce Alternative Hierarchical Attention (AHA) mechanism that efficiently fuses features across views through separate intra-frame and inter-frame windowed self-attention, achieving cross-view feature mixing with reduced overhead. (2) We propose a novel ERP fusion approach that projects multi-view depth estimates to a shared equirectangular coordinate system to obtain the final fusion depth. (3) We generate ERP image-depth pairs using HM3D and 2D3D-S datasets for comprehensive evaluation, demonstrating competitive zero-shot performance on real datasets while achieving up to 20 FPS on NVIDIA Orin NX embedded hardware. Project page: \href{https://3f7dfc.github.io/FastVidar/}{https://3f7dfc.github.io/FastVidar/}
>
---
#### [new 112] Clebsch-Gordan Transformer: Fast and Global Equivariant Attention
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文属于机器学习任务，解决equivariant transformer计算效率低的问题，提出Clebsch-Gordan Transformer实现全局注意力与高阶等变特征建模。**

- **链接: [http://arxiv.org/pdf/2509.24093v1](http://arxiv.org/pdf/2509.24093v1)**

> **作者:** Owen Lewis Howell; Linfeng Zhao; Xupeng Zhu; Yaoyao Qian; Haojie Huang; Lingfeng Sun; Wil Thomason; Robert Platt; Robin Walters
>
> **摘要:** The global attention mechanism is one of the keys to the success of transformer architecture, but it incurs quadratic computational costs in relation to the number of tokens. On the other hand, equivariant models, which leverage the underlying geometric structures of problem instance, often achieve superior accuracy in physical, biochemical, computer vision, and robotic tasks, at the cost of additional compute requirements. As a result, existing equivariant transformers only support low-order equivariant features and local context windows, limiting their expressiveness and performance. This work proposes Clebsch-Gordan Transformer, achieving efficient global attention by a novel Clebsch-Gordon Convolution on $\SO(3)$ irreducible representations. Our method enables equivariant modeling of features at all orders while achieving ${O}(N \log N)$ input token complexity. Additionally, the proposed method scales well with high-order irreducible features, by exploiting the sparsity of the Clebsch-Gordon matrix. Lastly, we also incorporate optional token permutation equivariance through either weight sharing or data augmentation. We benchmark our method on a diverse set of benchmarks including n-body simulation, QM9, ModelNet point cloud classification and a robotic grasping dataset, showing clear gains over existing equivariant transformers in GPU memory size, speed, and accuracy.
>
---
#### [new 113] Gaze Estimation for Human-Robot Interaction: Analysis Using the NICO Platform
- **分类: cs.CV; cs.RO; I.4.9**

- **简介: 该论文属于人机交互中的注视估计任务，旨在评估现有方法在共享工作空间中的性能，并通过新数据集验证其实际局限性。**

- **链接: [http://arxiv.org/pdf/2509.24001v1](http://arxiv.org/pdf/2509.24001v1)**

> **作者:** Matej Palider; Omar Eldardeer; Viktor Kocur
>
> **备注:** Code available at http://github.com/kocurvik/nico_gaze
>
> **摘要:** This paper evaluates the current gaze estimation methods within an HRI context of a shared workspace scenario. We introduce a new, annotated dataset collected with the NICO robotic platform. We evaluate four state-of-the-art gaze estimation models. The evaluation shows that the angular errors are close to those reported on general-purpose benchmarks. However, when expressed in terms of distance in the shared workspace the best median error is 16.48 cm quantifying the practical limitations of current methods. We conclude by discussing these limitations and offering recommendations on how to best integrate gaze estimation as a modality in HRI systems.
>
---
#### [new 114] Motion Informed Needle Segmentation in Ultrasound Images
- **分类: eess.IV; cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于医学图像分割任务，解决超声图像中运动针的分割问题。通过结合卡尔曼滤波与深度学习，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2312.01239v3](http://arxiv.org/pdf/2312.01239v3)**

> **作者:** Raghavv Goel; Cecilia Morales; Manpreet Singh; Artur Dubrawski; John Galeotti; Howie Choset
>
> **备注:** 7 pages, 4 figures, accepted at ISBI 2024
>
> **摘要:** Segmenting a moving needle in ultrasound images is challenging due to the presence of artifacts, noise, and needle occlusion. This task becomes even more demanding in scenarios where data availability is limited. In this paper, we present a novel approach for needle segmentation for 2D ultrasound that combines classical Kalman Filter (KF) techniques with data-driven learning, incorporating both needle features and needle motion. Our method offers three key contributions. First, we propose a compatible framework that seamlessly integrates into commonly used encoder-decoder style architectures. Second, we demonstrate superior performance compared to recent state-of-the-art needle segmentation models using our novel convolutional neural network (CNN) based KF-inspired block, achieving a 15\% reduction in pixel-wise needle tip error and an 8\% reduction in length error. Third, to our knowledge we are the first to implement a learnable filter to incorporate non-linear needle motion for improving needle segmentation.
>
---
#### [new 115] From Static to Dynamic: a Survey of Topology-Aware Perception in Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶感知任务，解决静态地图局限问题，通过动态感知技术提升系统适应性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2509.23641v1](http://arxiv.org/pdf/2509.23641v1)**

> **作者:** Yixiao Chen; Ruining Yang; Xin Chen; Jia He; Dongliang Xu; Yue Yao
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** The key to achieving autonomous driving lies in topology-aware perception, the structured understanding of the driving environment with an emphasis on lane topology and road semantics. This survey systematically reviews four core research directions under this theme: vectorized map construction, topological structure modeling, prior knowledge fusion, and language model-based perception. Across these directions, we observe a unifying trend: a paradigm shift from static, pre-built maps to dynamic, sensor-driven perception. Specifically, traditional static maps have provided semantic context for autonomous systems. However, they are costly to construct, difficult to update in real time, and lack generalization across regions, limiting their scalability. In contrast, dynamic representations leverage on-board sensor data for real-time map construction and topology reasoning. Each of the four research directions contributes to this shift through compact spatial modeling, semantic relational reasoning, robust domain knowledge integration, and multimodal scene understanding powered by pre-trained language models. Together, they pave the way for more adaptive, scalable, and explainable autonomous driving systems.
>
---
#### [new 116] GRS-SLAM3R: Real-Time Dense SLAM with Gated Recurrent State
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于实时密集SLAM任务，解决点云全局一致性问题。提出GRS-SLAM3R框架，通过空间记忆和分块对齐实现高精度实时重建。**

- **链接: [http://arxiv.org/pdf/2509.23737v1](http://arxiv.org/pdf/2509.23737v1)**

> **作者:** Guole Shen; Tianchen Deng; Yanbo Wang; Yongtao Chen; Yilin Shen; Jiuming Liu; Jingchuan Wang
>
> **摘要:** DUSt3R-based end-to-end scene reconstruction has recently shown promising results in dense visual SLAM. However, most existing methods only use image pairs to estimate pointmaps, overlooking spatial memory and global consistency.To this end, we introduce GRS-SLAM3R, an end-to-end SLAM framework for dense scene reconstruction and pose estimation from RGB images without any prior knowledge of the scene or camera parameters. Unlike existing DUSt3R-based frameworks, which operate on all image pairs and predict per-pair point maps in local coordinate frames, our method supports sequentialized input and incrementally estimates metric-scale point clouds in the global coordinate. In order to improve consistent spatial correlation, we use a latent state for spatial memory and design a transformer-based gated update module to reset and update the spatial memory that continuously aggregates and tracks relevant 3D information across frames. Furthermore, we partition the scene into submaps, apply local alignment within each submap, and register all submaps into a common world frame using relative constraints, producing a globally consistent map. Experiments on various datasets show that our framework achieves superior reconstruction accuracy while maintaining real-time performance.
>
---
#### [new 117] When Autonomous Vehicle Meets V2X Cooperative Perception: How Far Are We?
- **分类: cs.AI; cs.RO; cs.SE**

- **简介: 该论文属于自动驾驶感知任务，旨在解决V2X协同感知系统的性能与可靠性问题。通过分析误差模式和关键组件，评估其对驾驶安全的影响。**

- **链接: [http://arxiv.org/pdf/2509.24927v1](http://arxiv.org/pdf/2509.24927v1)**

> **作者:** An Guo; Shuoxiao Zhang; Enyi Tang; Xinyu Gao; Haomin Pang; Haoxiang Tian; Yanzhou Mu; Wu Wen; Chunrong Fang; Zhenyu Chen
>
> **备注:** The paper has been accepted by the 40th IEEE/ACM International Conference on Automated Software Engineering, ASE 2025
>
> **摘要:** With the tremendous advancement of deep learning and communication technology, Vehicle-to-Everything (V2X) cooperative perception has the potential to address limitations in sensing distant objects and occlusion for a single-agent perception system. V2X cooperative perception systems are software systems characterized by diverse sensor types and cooperative agents, varying fusion schemes, and operation under different communication conditions. Therefore, their complex composition gives rise to numerous operational challenges. Furthermore, when cooperative perception systems produce erroneous predictions, the types of errors and their underlying causes remain insufficiently explored. To bridge this gap, we take an initial step by conducting an empirical study of V2X cooperative perception. To systematically evaluate the impact of cooperative perception on the ego vehicle's perception performance, we identify and analyze six prevalent error patterns in cooperative perception systems. We further conduct a systematic evaluation of the critical components of these systems through our large-scale study and identify the following key findings: (1) The LiDAR-based cooperation configuration exhibits the highest perception performance; (2) Vehicle-to-infrastructure (V2I) and vehicle-to-vehicle (V2V) communication exhibit distinct cooperative perception performance under different fusion schemes; (3) Increased cooperative perception errors may result in a higher frequency of driving violations; (4) Cooperative perception systems are not robust against communication interference when running online. Our results reveal potential risks and vulnerabilities in critical components of cooperative perception systems. We hope that our findings can better promote the design and repair of cooperative perception systems.
>
---
#### [new 118] ThermalGen: Style-Disentangled Flow-Based Generative Models for RGB-to-Thermal Image Translation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于RGB到热成像图像翻译任务，旨在解决缺乏同步RGB-热数据的问题。提出ThermalGen模型，结合风格解耦机制，实现高质量热图生成。**

- **链接: [http://arxiv.org/pdf/2509.24878v1](http://arxiv.org/pdf/2509.24878v1)**

> **作者:** Jiuhong Xiao; Roshan Nayak; Ning Zhang; Daniel Tortei; Giuseppe Loianno
>
> **备注:** 23 pages including the checklist and appendix. Accepted at NeurIPS 2025
>
> **摘要:** Paired RGB-thermal data is crucial for visual-thermal sensor fusion and cross-modality tasks, including important applications such as multi-modal image alignment and retrieval. However, the scarcity of synchronized and calibrated RGB-thermal image pairs presents a major obstacle to progress in these areas. To overcome this challenge, RGB-to-Thermal (RGB-T) image translation has emerged as a promising solution, enabling the synthesis of thermal images from abundant RGB datasets for training purposes. In this study, we propose ThermalGen, an adaptive flow-based generative model for RGB-T image translation, incorporating an RGB image conditioning architecture and a style-disentangled mechanism. To support large-scale training, we curated eight public satellite-aerial, aerial, and ground RGB-T paired datasets, and introduced three new large-scale satellite-aerial RGB-T datasets--DJI-day, Bosonplus-day, and Bosonplus-night--captured across diverse times, sensor types, and geographic regions. Extensive evaluations across multiple RGB-T benchmarks demonstrate that ThermalGen achieves comparable or superior translation performance compared to existing GAN-based and diffusion-based methods. To our knowledge, ThermalGen is the first RGB-T image translation model capable of synthesizing thermal images that reflect significant variations in viewpoints, sensor characteristics, and environmental conditions. Project page: http://xjh19971.github.io/ThermalGen
>
---
#### [new 119] Fast Feature Field ($\text{F}^3$): A Predictive Representation of Events
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出$\text{F}^3$，一种用于事件相机数据的预测表示方法，解决场景结构与运动信息的高效表征问题。通过预测未来事件，实现高帧率的实时处理，并在多个任务中取得最佳效果。**

- **链接: [http://arxiv.org/pdf/2509.25146v1](http://arxiv.org/pdf/2509.25146v1)**

> **作者:** Richeek Das; Kostas Daniilidis; Pratik Chaudhari
>
> **备注:** 39 pages, 9 figures
>
> **摘要:** This paper develops a mathematical argument and algorithms for building representations of data from event-based cameras, that we call Fast Feature Field ($\text{F}^3$). We learn this representation by predicting future events from past events and show that it preserves scene structure and motion information. $\text{F}^3$ exploits the sparsity of event data and is robust to noise and variations in event rates. It can be computed efficiently using ideas from multi-resolution hash encoding and deep sets - achieving 120 Hz at HD and 440 Hz at VGA resolutions. $\text{F}^3$ represents events within a contiguous spatiotemporal volume as a multi-channel image, enabling a range of downstream tasks. We obtain state-of-the-art performance on optical flow estimation, semantic segmentation, and monocular metric depth estimation, on data from three robotic platforms (a car, a quadruped robot and a flying platform), across different lighting conditions (daytime, nighttime), environments (indoors, outdoors, urban, as well as off-road) and dynamic vision sensors (resolutions and event rates). Our implementations can predict these tasks at 25-75 Hz at HD resolution.
>
---
#### [new 120] SCOPE: Semantic Conditioning for Sim2Real Category-Level Object Pose Estimation in Robotics
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人领域中的对象位姿估计任务，旨在解决未知物体的准确位姿估计问题。通过引入语义先验和扩散模型，提升模型泛化能力，实现跨类别的高精度位姿估计。**

- **链接: [http://arxiv.org/pdf/2509.24572v1](http://arxiv.org/pdf/2509.24572v1)**

> **作者:** Peter Hönig; Stefan Thalhammer; Jean-Baptiste Weibel; Matthias Hirschmanner; Markus Vincze
>
> **摘要:** Object manipulation requires accurate object pose estimation. In open environments, robots encounter unknown objects, which requires semantic understanding in order to generalize both to known categories and beyond. To resolve this challenge, we present SCOPE, a diffusion-based category-level object pose estimation model that eliminates the need for discrete category labels by leveraging DINOv2 features as continuous semantic priors. By combining these DINOv2 features with photorealistic training data and a noise model for point normals, we reduce the Sim2Real gap in category-level object pose estimation. Furthermore, injecting the continuous semantic priors via cross-attention enables SCOPE to learn canonicalized object coordinate systems across object instances beyond the distribution of known categories. SCOPE outperforms the current state of the art in synthetically trained category-level object pose estimation, achieving a relative improvement of 31.9\% on the 5$^\circ$5cm metric. Additional experiments on two instance-level datasets demonstrate generalization beyond known object categories, enabling grasping of unseen objects from unknown categories with a success rate of up to 100\%. Code available: https://github.com/hoenigpeter/scope.
>
---
#### [new 121] Discrete Variational Autoencoding via Policy Search
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于生成模型任务，解决离散变分自编码器训练难题。通过政策搜索方法提升训练效果，实现高效高维数据重建。**

- **链接: [http://arxiv.org/pdf/2509.24716v1](http://arxiv.org/pdf/2509.24716v1)**

> **作者:** Michael Drolet; Firas Al-Hafez; Aditya Bhatt; Jan Peters; Oleg Arenz
>
> **摘要:** Discrete latent bottlenecks in variational autoencoders (VAEs) offer high bit efficiency and can be modeled with autoregressive discrete distributions, enabling parameter-efficient multimodal search with transformers. However, discrete random variables do not allow for exact differentiable parameterization; therefore, discrete VAEs typically rely on approximations, such as Gumbel-Softmax reparameterization or straight-through gradient estimates, or employ high-variance gradient-free methods such as REINFORCE that have had limited success on high-dimensional tasks such as image reconstruction. Inspired by popular techniques in policy search, we propose a training framework for discrete VAEs that leverages the natural gradient of a non-parametric encoder to update the parametric encoder without requiring reparameterization. Our method, combined with automatic step size adaptation and a transformer-based encoder, scales to challenging datasets such as ImageNet and outperforms both approximate reparameterization methods and quantization-based discrete autoencoders in reconstructing high-dimensional data from compact latent spaces, achieving a 20% improvement on FID Score for ImageNet 256.
>
---
#### [new 122] Systematic Alias Sampling: an efficient and low-variance way to sample from a discrete distribution
- **分类: cs.DS; cs.MS; cs.RO; F.2.2; G.3; G.4; I.2.9; I.6.6**

- **简介: 该论文属于随机采样任务，旨在提高离散分布采样的效率与方差控制。通过结合别名方法与系统抽样，提出一种更快速且低方差的采样方法。**

- **链接: [http://arxiv.org/pdf/2509.24089v1](http://arxiv.org/pdf/2509.24089v1)**

> **作者:** Ilari Vallivaara; Katja Poikselkä; Pauli Rikula; Juha Röning
>
> **摘要:** In this paper we combine the Alias method with the concept of systematic sampling, a method commonly used in particle filters for efficient low-variance resampling. The proposed method allows very fast sampling from a discrete distribution: drawing k samples is up to an order of magnitude faster than binary search from the cumulative distribution function (cdf) or inversion methods used in many libraries. The produced empirical distribution function is evaluated using a modified Cram\'er-Von Mises goodness-of-fit statistic, showing that the method compares very favourably to multinomial sampling. As continuous distributions can often be approximated with discrete ones, the proposed method can be used as a very general way to efficiently produce random samples for particle filter proposal distributions, e.g. for motion models in robotics.
>
---
#### [new 123] Safety-Critical Input-Constrained Nonlinear Intercept Guidance in Multiple Engagement Zones
- **分类: eess.SY; cs.MA; cs.RO; cs.SY; math.DS**

- **简介: 该论文属于导弹拦截任务，解决多防御者环境下安全拦截问题。通过几何方法定义威胁区域，设计非线性制导律实现安全拦截。**

- **链接: [http://arxiv.org/pdf/2509.25053v1](http://arxiv.org/pdf/2509.25053v1)**

> **作者:** Praveen Kumar Ranjan; Abhinav Sinha; Yongcan Cao
>
> **摘要:** This paper presents an input-constrained nonlinear guidance law to address the problem of intercepting a stationary target in contested environments with multiple defending agents. Contrary to prior approaches that rely on explicit knowledge of defender strategies or utilize conservative safety conditions based on a defender's range, our work characterizes defender threats geometrically through engagement zones that delineate inevitable interception regions. Outside these engagement zones, the interceptor remains invulnerable. The proposed guidance law switches between a repulsive safety maneuver near these zones and a pursuit maneuver outside their influence. To deal with multiple engagement zones, we employ a smooth minimum function (log-sum-exponent approximation) that aggregates threats from all the zones while prioritizing the most critical threats. Input saturation is modeled and embedded in the non-holonomic vehicle dynamics so the controller respects actuator limits while maintaining stability. Numerical simulations with several defenders demonstrate the proposed method's ability to avoid engagement zones and achieve interception across diverse initial conditions.
>
---
#### [new 124] DriveE2E: Closed-Loop Benchmark for End-to-End Autonomous Driving through Real-to-Simulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决闭环评估中场景真实性不足的问题。通过将真实交通场景引入CARLA模拟器，构建更真实的驾驶评估环境。**

- **链接: [http://arxiv.org/pdf/2509.23922v1](http://arxiv.org/pdf/2509.23922v1)**

> **作者:** Haibao Yu; Wenxian Yang; Ruiyang Hao; Chuanye Wang; Jiaru Zhong; Ping Luo; Zaiqing Nie
>
> **备注:** End-to-End Autonomous Driving Simulation and Benchmark
>
> **摘要:** Closed-loop evaluation is increasingly critical for end-to-end autonomous driving. Current closed-loop benchmarks using the CARLA simulator rely on manually configured traffic scenarios, which can diverge from real-world conditions, limiting their ability to reflect actual driving performance. To address these limitations, we introduce a simple yet challenging closed-loop evaluation framework that closely integrates real-world driving scenarios into the CARLA simulator with infrastructure cooperation. Our approach involves extracting 800 dynamic traffic scenarios selected from a comprehensive 100-hour video dataset captured by high-mounted infrastructure sensors, and creating static digital twin assets for 15 real-world intersections with consistent visual appearance. These digital twins accurately replicate the traffic and environmental characteristics of their real-world counterparts, enabling more realistic simulations in CARLA. This evaluation is challenging due to the diversity of driving behaviors, locations, weather conditions, and times of day at complex urban intersections. In addition, we provide a comprehensive closed-loop benchmark for evaluating end-to-end autonomous driving models. Project URL: \href{https://github.com/AIR-THU/DriveE2E}{https://github.com/AIR-THU/DriveE2E}.
>
---
#### [new 125] Training Agents Inside of Scalable World Models
- **分类: cs.AI; cs.LG; cs.RO; stat.ML**

- **简介: 该论文属于强化学习任务，旨在解决复杂环境中智能体行为训练问题。通过构建可扩展的世界模型，Dreamer 4在Minecraft中实现无需环境交互的钻石获取。**

- **链接: [http://arxiv.org/pdf/2509.24527v1](http://arxiv.org/pdf/2509.24527v1)**

> **作者:** Danijar Hafner; Wilson Yan; Timothy Lillicrap
>
> **备注:** Website: https://danijar.com/dreamer4/
>
> **摘要:** World models learn general knowledge from videos and simulate experience for training behaviors in imagination, offering a path towards intelligent agents. However, previous world models have been unable to accurately predict object interactions in complex environments. We introduce Dreamer 4, a scalable agent that learns to solve control tasks by reinforcement learning inside of a fast and accurate world model. In the complex video game Minecraft, the world model accurately predicts object interactions and game mechanics, outperforming previous world models by a large margin. The world model achieves real-time interactive inference on a single GPU through a shortcut forcing objective and an efficient transformer architecture. Moreover, the world model learns general action conditioning from only a small amount of data, allowing it to extract the majority of its knowledge from diverse unlabeled videos. We propose the challenge of obtaining diamonds in Minecraft from only offline data, aligning with practical applications such as robotics where learning from environment interaction can be unsafe and slow. This task requires choosing sequences of over 20,000 mouse and keyboard actions from raw pixels. By learning behaviors in imagination, Dreamer 4 is the first agent to obtain diamonds in Minecraft purely from offline data, without environment interaction. Our work provides a scalable recipe for imagination training, marking a step towards intelligent agents.
>
---
#### [new 126] SIG-Chat: Spatial Intent-Guided Conversational Gesture Generation Involving How, When and Where
- **分类: cs.GR; cs.MM; cs.RO**

- **简介: 该论文属于对话手势生成任务，解决交互时间与空间意图不足的问题。通过多模态数据融合与评估，提升手势生成的准确性与互动性。**

- **链接: [http://arxiv.org/pdf/2509.23852v1](http://arxiv.org/pdf/2509.23852v1)**

> **作者:** Yiheng Huang; Junran Peng; Silei Shen; Jingwei Yang; ZeJi Wei; ChenCheng Bai; Yonghao He; Wei Sui; Muyi Sun; Yan Liu; Xu-Cheng Yin; Man Zhang; Zhaoxiang Zhang; Chuanchen Luo
>
> **摘要:** The accompanying actions and gestures in dialogue are often closely linked to interactions with the environment, such as looking toward the interlocutor or using gestures to point to the described target at appropriate moments. Speech and semantics guide the production of gestures by determining their timing (WHEN) and style (HOW), while the spatial locations of interactive objects dictate their directional execution (WHERE). Existing approaches either rely solely on descriptive language to generate motions or utilize audio to produce non-interactive gestures, thereby lacking the characterization of interactive timing and spatial intent. This significantly limits the applicability of conversational gesture generation, whether in robotics or in the fields of game and animation production. To address this gap, we present a full-stack solution. We first established a unique data collection method to simultaneously capture high-precision human motion and spatial intent. We then developed a generation model driven by audio, language, and spatial data, alongside dedicated metrics for evaluating interaction timing and spatial accuracy. Finally, we deployed the solution on a humanoid robot, enabling rich, context-aware physical interactions.
>
---
#### [new 127] Color-Pair Guided Robust Zero-Shot 6D Pose Estimation and Tracking of Cluttered Objects on Edge Devices
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于6D姿态估计任务，解决复杂光照下新物体的鲁棒姿态估计与跟踪问题。提出一种融合颜色对特征的统一框架，实现高效边缘设备部署。**

- **链接: [http://arxiv.org/pdf/2509.23647v1](http://arxiv.org/pdf/2509.23647v1)**

> **作者:** Xingjian Yang; Ashis G. Banerjee
>
> **摘要:** Robust 6D pose estimation of novel objects under challenging illumination remains a significant challenge, often requiring a trade-off between accurate initial pose estimation and efficient real-time tracking. We present a unified framework explicitly designed for efficient execution on edge devices, which synergizes a robust initial estimation module with a fast motion-based tracker. The key to our approach is a shared, lighting-invariant color-pair feature representation that forms a consistent foundation for both stages. For initial estimation, this feature facilitates robust registration between the live RGB-D view and the object's 3D mesh. For tracking, the same feature logic validates temporal correspondences, enabling a lightweight model to reliably regress the object's motion. Extensive experiments on benchmark datasets demonstrate that our integrated approach is both effective and robust, providing competitive pose estimation accuracy while maintaining high-fidelity tracking even through abrupt pose changes.
>
---
#### [new 128] Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多智能体交通模拟任务，旨在解决训练与测试分布不一致导致的模型泛化问题。通过引入R1风格的强化微调方法，提升模拟行为与人类偏好的一致性。**

- **链接: [http://arxiv.org/pdf/2509.23993v1](http://arxiv.org/pdf/2509.23993v1)**

> **作者:** Muleilan Pei; Shaoshuai Shi; Shaojie Shen
>
> **摘要:** Scalable and realistic simulation of multi-agent traffic behavior is critical for advancing autonomous driving technologies. Although existing data-driven simulators have made significant strides in this domain, they predominantly rely on supervised learning to align simulated distributions with real-world driving scenarios. A persistent challenge, however, lies in the distributional shift that arises between training and testing, which often undermines model generalization in unseen environments. To address this limitation, we propose SMART-R1, a novel R1-style reinforcement fine-tuning paradigm tailored for next-token prediction models to better align agent behavior with human preferences and evaluation metrics. Our approach introduces a metric-oriented policy optimization algorithm to improve distribution alignment and an iterative "SFT-RFT-SFT" training strategy that alternates between Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT) to maximize performance gains. Extensive experiments on the large-scale Waymo Open Motion Dataset (WOMD) validate the effectiveness of this simple yet powerful R1-style training framework in enhancing foundation models. The results on the Waymo Open Sim Agents Challenge (WOSAC) showcase that SMART-R1 achieves state-of-the-art performance with an overall realism meta score of 0.7858, ranking first on the leaderboard at the time of submission.
>
---
#### [new 129] ELHPlan: Efficient Long-Horizon Task Planning for Multi-Agent Collaboration
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于多智能体协作任务，解决LLM在动态环境中的适应性与计算效率问题。提出ELHPlan框架，通过动作链实现高效长周期规划。**

- **链接: [http://arxiv.org/pdf/2509.24230v1](http://arxiv.org/pdf/2509.24230v1)**

> **作者:** Shaobin Ling; Yun Wang; Chenyou Fan; Tin Lun Lam; Junjie Hu
>
> **摘要:** Large Language Models (LLMs) enable intelligent multi-robot collaboration but face fundamental trade-offs: declarative methods lack adaptability in dynamic environments, while iterative methods incur prohibitive computational costs that scale poorly with team size and task complexity. In this paper, we propose ELHPlan, a novel framework that introduces Action Chains--sequences of actions explicitly bound to sub-goal intentions--as the fundamental planning primitive. ELHPlan operates via a cyclical process: 1) constructing intention-bound action sequences, 2) proactively validating for conflicts and feasibility, 3) refining issues through targeted mechanisms, and 4) executing validated actions. This design balances adaptability and efficiency by providing sufficient planning horizons while avoiding expensive full re-planning. We further propose comprehensive efficiency metrics, including token consumption and planning time, to more holistically evaluate multi-agent collaboration. Our experiments on benchmark TDW-MAT and C-WAH demonstrate that ELHPlan achieves comparable task success rates while consuming only 24% of the tokens required by state-of-the-art methods. Our research establishes a new efficiency-effectiveness frontier for LLM-based multi-agent planning systems.
>
---
## 更新

#### [replaced 001] Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.RO**

- **链接: [http://arxiv.org/pdf/2405.14314v4](http://arxiv.org/pdf/2405.14314v4)**

> **作者:** Yang Zhang; Shixin Yang; Chenjia Bai; Fei Wu; Xiu Li; Zhen Wang; Xuelong Li
>
> **备注:** accepted by ACL'2025
>
> **摘要:** Grounding the reasoning ability of large language models (LLMs) for embodied tasks is challenging due to the complexity of the physical world. Especially, LLM planning for multi-agent collaboration requires communication of agents or credit assignment as the feedback to re-adjust the proposed plans and achieve effective coordination. However, existing methods that overly rely on physical verification or self-reflection suffer from excessive and inefficient querying of LLMs. In this paper, we propose a novel framework for multi-agent collaboration that introduces Reinforced Advantage feedback (ReAd) for efficient self-refinement of plans. Specifically, we perform critic regression to learn a sequential advantage function from LLM-planned data, and then treat the LLM planner as an optimizer to generate actions that maximize the advantage function. It endows the LLM with the foresight to discern whether the action contributes to accomplishing the final task. We provide theoretical analysis by extending advantage-weighted regression in reinforcement learning to multi-agent systems. Experiments on Overcooked-AI and a difficult variant of RoCoBench show that ReAd surpasses baselines in success rate, and also significantly decreases the interaction steps of agents and query rounds of LLMs, demonstrating its high efficiency for grounding LLMs. More results are given at https://embodied-read.github.io
>
---
#### [replaced 002] Learning More With Less: Sample Efficient Model-Based RL for Loco-Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2501.10499v3](http://arxiv.org/pdf/2501.10499v3)**

> **作者:** Benjamin Hoffman; Jin Cheng; Chenhao Li; Stelian Coros
>
> **摘要:** By combining the agility of legged locomotion with the capabilities of manipulation, loco-manipulation platforms have the potential to perform complex tasks in real-world applications. To this end, state-of-the-art quadrupeds with manipulators, such as the Boston Dynamics Spot, have emerged to provide a capable and robust platform. However, the complexity of loco-manipulation control, as well as the black-box nature of commercial platforms, pose challenges for deriving accurate dynamics models and robust control policies. To address these challenges, we turn to model-based reinforcement learning (RL). We develop a hand-crafted kinematic model of a quadruped-with-arm platform which - employing recent advances in Bayesian Neural Network (BNN)-based learning - we use as a physical prior to efficiently learn an accurate dynamics model from limited data. We then leverage our learned model to derive control policies for loco-manipulation via RL. We demonstrate the effectiveness of our approach on state-of-the-art hardware using the Boston Dynamics Spot, accurately performing dynamic end-effector trajectory tracking even in low data regimes. Project website and videos: https://sites.google.com/view/learning-more-with-less.
>
---
#### [replaced 003] Hierarchical Task Environments as the Next Frontier for Embodied World Models in Robot Soccer
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.RO; 68T05, 90C40, 91A26, 68T42, 93E35; I.2.11; I.2.6; I.2.8; I.2.9; I.2.7**

- **链接: [http://arxiv.org/pdf/2509.04731v2](http://arxiv.org/pdf/2509.04731v2)**

> **作者:** Brennen Hill
>
> **备注:** In the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Embodied World Models for Decision Making (EWM)
>
> **摘要:** Recent advances in agent development have focused on scaling model size and raw interaction data, mirroring the successes seen in large language models. However, for complex, long-horizon multi-agent tasks such as robotic soccer, this end-to-end approach often fails due to intractable exploration spaces and sparse rewards. This position paper argues that the next frontier in developing embodied world models is not merely increasing the fidelity or size of environments, but scaling their structural complexity through explicit hierarchical scaffolding. We posit that an effective world model for decision-making must model not only the world's physics but also its task semantics. Drawing from a systematic review of 2024 research in low-resource multi-agent soccer, we identify a clear trend towards integrating symbolic and hierarchical methods, such as Hierarchical Task Networks (HTNs) and Bayesian Strategy Networks (BSNs), with multi-agent reinforcement learning (MARL). These methods decompose complex goals into manageable subgoals, creating an intrinsic curriculum that shapes agent learning. We propose that such structured environments are essential for bridging the gap between simple, reactive behaviors and sophisticated, strategic team play. We further extend this principle, proposing that this scaffolding can be generalized to other complex domains and dynamically generated by Large Language Models (LLMs), which act as generative world models of tasks. By building environments with explicit, composable task layers, we can guide agent exploration more efficiently, generate meaningful learning signals, and ultimately train more capable and general-purpose agents with fewer resources than purely end-to-end approaches.
>
---
#### [replaced 004] Kineto-Dynamical Planning and Accurate Execution of Minimum-Time Maneuvers on Three-Dimensional Circuits
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.03454v2](http://arxiv.org/pdf/2502.03454v2)**

> **作者:** Mattia Piccinini; Sebastiano Taddei; Johannes Betz; Francesco Biral
>
> **备注:** 2025 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Online planning and execution of minimum-time maneuvers on three-dimensional (3D) circuits is an open challenge in autonomous vehicle racing. In this paper, we present an artificial race driver (ARD) to learn the vehicle dynamics, plan and execute minimum-time maneuvers on a 3D track. ARD integrates a novel kineto-dynamical (KD) vehicle model for trajectory planning with economic nonlinear model predictive control (E-NMPC). We use a high-fidelity vehicle simulator (VS) to compare the closed-loop ARD results with a minimum-lap-time optimal control problem (MLT-VS), solved offline with the same VS. Our ARD sets lap times close to the MLT-VS, and the new KD model outperforms a literature benchmark. Finally, we study the vehicle trajectories, to assess the re-planning capabilities of ARD under execution errors. A video with the main results is available as supplementary material.
>
---
#### [replaced 005] Grasping a Handful: Sequential Multi-Object Dexterous Grasp Generation
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.22370v4](http://arxiv.org/pdf/2503.22370v4)**

> **作者:** Haofei Lu; Yifei Dong; Zehang Weng; Florian T. Pokorny; Jens Lundell; Danica Kragic
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** We introduce the sequential multi-object robotic grasp sampling algorithm SeqGrasp that can robustly synthesize stable grasps on diverse objects using the robotic hand's partial Degrees of Freedom (DoF). We use SeqGrasp to construct the large-scale Allegro Hand sequential grasping dataset SeqDataset and use it for training the diffusion-based sequential grasp generator SeqDiffuser. We experimentally evaluate SeqGrasp and SeqDiffuser against the state-of-the-art non-sequential multi-object grasp generation method MultiGrasp in simulation and on a real robot. The experimental results demonstrate that SeqGrasp and SeqDiffuser reach an 8.71%-43.33% higher grasp success rate than MultiGrasp. Furthermore, SeqDiffuser is approximately 1000 times faster at generating grasps than SeqGrasp and MultiGrasp. Project page: https://yulihn.github.io/SeqGrasp/.
>
---
#### [replaced 006] SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.18648v2](http://arxiv.org/pdf/2509.18648v2)**

> **作者:** Yarden As; Chengrui Qu; Benjamin Unger; Dongho Kang; Max van der Hart; Laixi Shi; Stelian Coros; Adam Wierman; Andreas Krause
>
> **摘要:** Deploying reinforcement learning (RL) safely in the real world is challenging, as policies trained in simulators must face the inevitable sim-to-real gap. Robust safe RL techniques are provably safe, however difficult to scale, while domain randomization is more practical yet prone to unsafe behaviors. We address this gap by proposing SPiDR, short for Sim-to-real via Pessimistic Domain Randomization -- a scalable algorithm with provable guarantees for safe sim-to-real transfer. SPiDR uses domain randomization to incorporate the uncertainty about the sim-to-real gap into the safety constraints, making it versatile and highly compatible with existing training pipelines. Through extensive experiments on sim-to-sim benchmarks and two distinct real-world robotic platforms, we demonstrate that SPiDR effectively ensures safety despite the sim-to-real gap while maintaining strong performance.
>
---
#### [replaced 007] Leveraging Temporally Extended Behavior Sharing for Multi-task Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.20766v2](http://arxiv.org/pdf/2509.20766v2)**

> **作者:** Gawon Lee; Daesol Cho; H. Jin Kim
>
> **备注:** Accepted for publication in the proceedings of the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Multi-task reinforcement learning (MTRL) offers a promising approach to improve sample efficiency and generalization by training agents across multiple tasks, enabling knowledge sharing between them. However, applying MTRL to robotics remains challenging due to the high cost of collecting diverse task data. To address this, we propose MT-L\'evy, a novel exploration strategy that enhances sample efficiency in MTRL environments by combining behavior sharing across tasks with temporally extended exploration inspired by L\'evy flight. MT-L\'evy leverages policies trained on related tasks to guide exploration towards key states, while dynamically adjusting exploration levels based on task success ratios. This approach enables more efficient state-space coverage, even in complex robotics environments. Empirical results demonstrate that MT-L\'evy significantly improves exploration and sample efficiency, supported by quantitative and qualitative analyses. Ablation studies further highlight the contribution of each component, showing that combining behavior sharing with adaptive exploration strategies can significantly improve the practicality of MTRL in robotics applications.
>
---
#### [replaced 008] Symbolic Imitation Learning: From Black-Box to Explainable Driving Policies
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2309.16025v2](http://arxiv.org/pdf/2309.16025v2)**

> **作者:** Iman Sharifi; Mustafa Yildirim; Saber Fallah
>
> **备注:** 24 pages, 4 figures, 4 tables
>
> **摘要:** Current imitation learning approaches, predominantly based on deep neural networks (DNNs), offer efficient mechanisms for learning driving policies from real-world datasets. However, they suffer from inherent limitations in interpretability and generalizability--issues of critical importance in safety-critical domains such as autonomous driving. In this paper, we introduce Symbolic Imitation Learning (SIL), a novel framework that leverages Inductive Logic Programming (ILP) to derive explainable and generalizable driving policies from synthetic datasets. We evaluate SIL on real-world HighD and NGSim datasets, comparing its performance with state-of-the-art neural imitation learning methods using metrics such as collision rate, lane change efficiency, and average speed. The results indicate that SIL significantly enhances policy transparency while maintaining strong performance across varied driving conditions. These findings highlight the potential of integrating ILP into imitation learning to promote safer and more reliable autonomous systems.
>
---
#### [replaced 009] Whole-Body Integrated Motion Planning for Aerial Manipulators
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2501.06493v2](http://arxiv.org/pdf/2501.06493v2)**

> **作者:** Weiliang Deng; Hongming Chen; Biyu Ye; Haoran Chen; Ziliang Li; Ximin Lyu
>
> **备注:** 20 pages, 15 figures
>
> **摘要:** Expressive motion planning for Aerial Manipulators (AMs) is essential for tackling complex manipulation tasks, yet achieving coupled trajectory planning adaptive to various tasks remains challenging, especially for those requiring aggressive maneuvers. In this work, we propose a novel whole-body integrated motion planning framework for quadrotor-based AMs that leverages flexible waypoint constraints to achieve versatile manipulation capabilities. These waypoint constraints enable the specification of individual position requirements for either the quadrotor or end-effector, while also accommodating higher-order velocity and orientation constraints for complex manipulation tasks. To implement our framework, we exploit spatio-temporal trajectory characteristics and formulate an optimization problem to generate feasible trajectories for both the quadrotor and manipulator while ensuring collision avoidance considering varying robot configurations, dynamic feasibility, and kinematic feasibility. Furthermore, to enhance the maneuverability for specific tasks, we employ Imitation Learning (IL) to facilitate the optimization process to avoid poor local optima. The effectiveness of our framework is validated through comprehensive simulations and real-world experiments, where we successfully demonstrate nine fundamental manipulation skills across various environments.
>
---
#### [replaced 010] A Multi-Modality Evaluation of the Reality Gap in Autonomous Driving Systems
- **分类: cs.SE; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.22379v2](http://arxiv.org/pdf/2509.22379v2)**

> **作者:** Stefano Carlo Lambertenghi; Mirena Flores Valdez; Andrea Stocco
>
> **备注:** In proceedings of the 40th IEEE/ACM International Conference on Automated Software Engineering (ASE '25)
>
> **摘要:** Simulation-based testing is a cornerstone of Autonomous Driving System (ADS) development, offering safe and scalable evaluation across diverse driving scenarios. However, discrepancies between simulated and real-world behavior, known as the reality gap, challenge the transferability of test results to deployed systems. In this paper, we present a comprehensive empirical study comparing four representative testing modalities: Software-in-the-Loop (SiL), Vehicle-in-the-Loop (ViL), Mixed-Reality (MR), and full real-world testing. Using a small-scale physical vehicle equipped with real sensors (camera and LiDAR) and its digital twin, we implement each setup and evaluate two ADS architectures (modular and end-to-end) across diverse indoor driving scenarios involving real obstacles, road topologies, and indoor environments. We systematically assess the impact of each testing modality along three dimensions of the reality gap: actuation, perception, and behavioral fidelity. Our results show that while SiL and ViL setups simplify critical aspects of real-world dynamics and sensing, MR testing improves perceptual realism without compromising safety or control. Importantly, we identify the conditions under which failures do not transfer across testing modalities and isolate the underlying dimensions of the gap responsible for these discrepancies. Our findings offer actionable insights into the respective strengths and limitations of each modality and outline a path toward more robust and transferable validation of autonomous driving systems.
>
---
#### [replaced 011] Blast Hole Seeking and Dipping -- The Navigation and Perception Framework in a Mine Site Inspection Robot
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.13785v2](http://arxiv.org/pdf/2508.13785v2)**

> **作者:** Liyang Liu; Ehsan Mihankhah; Nathan Wallace; Javier Martinez; Andrew J. Hill
>
> **摘要:** In open-pit mining, holes are drilled into the surface of the excavation site and detonated with explosives to facilitate digging. These blast holes need to be inspected internally to assess subsurface material types and drill quality, in order to significantly reduce downstream material handling costs. Manual hole inspection is slow and expensive, limited in its ability to capture the geometric and geological characteristics of holes. This has been the motivation for the development of our autonomous mine-site inspection robot - "DIPPeR". In this paper, the automation aspect of the project is explained. We present a robust perception and navigation framework that provides streamlined blasthole seeking, tracking and accurate down-hole sensor positioning. To address challenges in the surface mining environment, where GPS and odometry data are noisy without RTK correction, we adopt a proximity-based adaptive navigation approach, enabling the vehicle to dynamically adjust its operations based on detected target availability and localisation accuracy. For perception, we process LiDAR data to extract the cone-shaped volume of drill-waste above ground, then project the 3D cone points into a virtual depth image to form accurate 2D segmentation of hole regions. To ensure continuous target-tracking as the robot approaches the goal, our system automatically adjusts projection parameters to preserve consistent hole image appearance. At the vicinity of the hole, we apply least squares circle fitting with non-maximum candidate suppression to achieve accurate hole detection and collision-free down-hole sensor placement. We demonstrate the effectiveness of our navigation and perception system in both high-fidelity simulation environments and on-site field trials. A demonstration video is available at https://www.youtube.com/watch?v=fRNbcBcaSqE.
>
---
#### [replaced 012] Constrained Decoding for Robotics Foundation Models
- **分类: cs.RO; cs.LG; cs.LO**

- **链接: [http://arxiv.org/pdf/2509.01728v3](http://arxiv.org/pdf/2509.01728v3)**

> **作者:** Parv Kapoor; Akila Ganlath; Michael Clifford; Changliu Liu; Sebastian Scherer; Eunsuk Kang
>
> **摘要:** Recent advances in the development of robotic foundation models have led to promising end-to-end and general-purpose capabilities in robotic systems. Trained on vast datasets of simulated and real-world trajectories, these models map multimodal observations directly to action sequences for physical execution. Despite promising real-world capabilities, these models are still data-driven and, therefore, lack explicit notions of behavioral correctness. We address this gap by introducing SafeDec, a constrained decoding framework for autoregressive, robot foundation models that enforces invariant safety specifications on candidate action trajectories. Task-specific safety rules are expressed as Signal Temporal Logic (STL) formulas and are enforced at inference time with minimal overhead. Our method ensures that generated actions provably satisfy STL specifications under assumed dynamics at runtime without retraining , while remaining agnostic of the underlying policy. We evaluate SafeDec on tasks from the CHORES benchmark for state-of-the-art generalist policies (e.g., SPOC, Flare, PoliFormer) across hundreds of procedurally generated environments and show that our decoding-time interventions are useful not only for filtering unsafe actions but also for conditional action generation. Videos are available at constrained-robot-fms.github.io.
>
---
#### [replaced 013] Track Any Motions under Any Disturbances
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.13833v2](http://arxiv.org/pdf/2509.13833v2)**

> **作者:** Zhikai Zhang; Jun Guo; Chao Chen; Jilong Wang; Chenghuai Lin; Yunrui Lian; Han Xue; Zhenrong Wang; Maoqi Liu; Huaping Liu; He Wang; Li Yi
>
> **摘要:** A foundational humanoid motion tracker is expected to be able to track diverse, highly dynamic, and contact-rich motions. More importantly, it needs to operate stably in real-world scenarios against various dynamics disturbances, including terrains, external forces, and physical property changes for general practical use. To achieve this goal, we propose Any2Track (Track Any motions under Any disturbances), a two-stage RL framework to track various motions under multiple disturbances in the real world. Any2Track reformulates dynamics adaptability as an additional capability on top of basic action execution and consists of two key components: AnyTracker and AnyAdapter. AnyTracker is a general motion tracker with a series of careful designs to track various motions within a single policy. AnyAdapter is a history-informed adaptation module that endows the tracker with online dynamics adaptability to overcome the sim2real gap and multiple real-world disturbances. We deploy Any2Track on Unitree G1 hardware and achieve a successful sim2real transfer in a zero-shot manner. Any2Track performs exceptionally well in tracking various motions under multiple real-world disturbances.
>
---
#### [replaced 014] Hybrid Many-Objective Optimization in Probabilistic Mission Design for Compliant and Effective UAV Routing
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.18514v2](http://arxiv.org/pdf/2412.18514v2)**

> **作者:** Simon Kohaut; Nikolas Hohmann; Sebastian Brulin; Benedict Flade; Julian Eggert; Markus Olhofer; Jürgen Adamy; Devendra Singh Dhami; Kristian Kersting
>
> **摘要:** Advanced Aerial Mobility encompasses many outstanding applications that promise to revolutionize modern logistics and pave the way for various public services and industry uses. However, throughout its history, the development of such systems has been impeded by the complexity of legal restrictions and physical constraints. While airspaces are often tightly shaped by various legal requirements, Unmanned Aerial Vehicles (UAV) must simultaneously consider, among others, energy demands, signal quality, and noise pollution. In this work, we address this challenge by presenting a novel architecture that integrates methods of Probabilistic Mission Design (ProMis) and Many-Objective Optimization for UAV routing. Hereby, our framework is able to comply with legal requirements under uncertainty while producing effective paths that minimize various physical costs a UAV needs to consider when traversing human-inhabited spaces. To this end, we combine hybrid probabilistic first-order logic for spatial reasoning with mixed deterministic-stochastic route optimization, incorporating physical objectives such as energy consumption and radio interference with a logical, probabilistic model of legal requirements. We demonstrate the versatility and advantages of our system in a large-scale empirical evaluation over real-world, crowd-sourced data from a map extract from the city of Paris, France, showing how a network of effective and compliant paths can be formed.
>
---
#### [replaced 015] WorldGym: World Model as An Environment for Policy Evaluation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.00613v2](http://arxiv.org/pdf/2506.00613v2)**

> **作者:** Julian Quevedo; Ansh Kumar Sharma; Yixiang Sun; Varad Suryavanshi; Percy Liang; Sherry Yang
>
> **备注:** https://world-model-eval.github.io
>
> **摘要:** Evaluating robot control policies is difficult: real-world testing is costly, and handcrafted simulators require manual effort to improve in realism and generality. We propose a world-model-based policy evaluation environment (WorldGym), an autoregressive, action-conditioned video generation model which serves as a proxy to real world environments. Policies are evaluated via Monte Carlo rollouts in the world model, with a vision-language model providing rewards. We evaluate a set of VLA-based real-robot policies in the world model using only initial frames from real robots, and show that policy success rates within the world model highly correlate with real-world success rates. Moreoever, we show that WorldGym is able to preserve relative policy rankings across different policy versions, sizes, and training checkpoints. Due to requiring only a single start frame as input, the world model further enables efficient evaluation of robot policies' generalization ability on novel tasks and environments. We find that modern VLA-based robot policies still struggle to distinguish object shapes and can become distracted by adversarial facades of objects. While generating highly realistic object interaction remains challenging, WorldGym faithfully emulates robot motions and offers a practical starting point for safe and reproducible policy evaluation before deployment.
>
---
#### [replaced 016] Vidar: Embodied Video Diffusion Model for Generalist Manipulation
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.12898v3](http://arxiv.org/pdf/2507.12898v3)**

> **作者:** Yao Feng; Hengkai Tan; Xinyi Mao; Chendong Xiang; Guodong Liu; Shuhe Huang; Hang Su; Jun Zhu
>
> **摘要:** Scaling general-purpose manipulation to new robot embodiments remains challenging: each platform typically needs large, homogeneous demonstrations, and pixel-to-action VLA pipelines typically degenerate under background and viewpoint shifts. In this paper, we present Vidar, a prior-driven, low-shot adaptation paradigm that replaces most embodiment-specific data with transferable video priors. Vidar consists of an embodied video diffusion model as the generalizable prior and a masked inverse dynamics model (MIDM) adapter based on a key decoupling of the policy. The embodied diffusion model is pre-trained on Internet-scale videos and then domain-adapted to 750K multi-view trajectories from three real-world robot platforms using a unified observation space encoding robot, camera, task, and scene contexts. The MIDM module learns action-relevant pixel masks without dense labels, grounding the prior into the target embodiment's action space while suppressing distractors. Crucially, the generative video prior models the distribution of plausible, temporally coherent interactions, implicitly capturing affordances, contact dynamics, and physical consistency from massive unlabeled video. This shifts the challenge from collecting large amounts of new robot data to efficiently aligning a rich prior with a new embodiment. With only 20 minutes of human demonstrations on an unseen robot (1% of typical data), Vidar outperforms state-of-the-art VLA baselines and generalizes to unseen tasks, backgrounds, and camera layouts. Our results suggest a scalable recipe for "one prior, many embodiments": strong, inexpensive video priors + minimal on-robot alignment.
>
---
#### [replaced 017] Hierarchical Intention-Aware Expressive Motion Generation for Humanoid Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.01563v4](http://arxiv.org/pdf/2506.01563v4)**

> **作者:** Lingfan Bao; Yan Pan; Tianhu Peng; Dimitrios Kanoulas; Chengxu Zhou
>
> **备注:** 7 pages, 2 figures, IEEE conference paper
>
> **摘要:** Effective human-robot interaction requires robots to identify human intentions and generate expressive, socially appropriate motions in real-time. Existing approaches often rely on fixed motion libraries or computationally expensive generative models. We propose a hierarchical framework that combines intention-aware reasoning via in-context learning (ICL) with real-time motion generation using diffusion models. Our system introduces structured prompting with confidence scoring, fallback behaviors, and social context awareness to enable intention refinement and adaptive response. Leveraging large-scale motion datasets and efficient latent-space denoising, the framework generates diverse, physically plausible gestures suitable for dynamic humanoid interactions. Experimental validation on a physical platform demonstrates the robustness and social alignment of our method in realistic scenarios.
>
---
#### [replaced 018] Neural-Augmented Kelvinlet for Real-Time Soft Tissue Deformation Modeling
- **分类: cs.GR; cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08043v2](http://arxiv.org/pdf/2506.08043v2)**

> **作者:** Ashkan Shahbazi; Kyvia Pereira; Jon S. Heiselman; Elaheh Akbari; Annie C. Benson; Sepehr Seifi; Xinyuan Liu; Garrison L. Johnston; Jie Ying Wu; Nabil Simaan; Michael L. Miga; Soheil Kolouri
>
> **摘要:** Accurate and efficient modeling of soft-tissue interactions is fundamental for advancing surgical simulation, surgical robotics, and model-based surgical automation. To achieve real-time latency, classical Finite Element Method (FEM) solvers are often replaced with neural approximations; however, naively training such models in a fully data-driven manner without incorporating physical priors frequently leads to poor generalization and physically implausible predictions. We present a novel physics-informed neural simulation framework that enables real-time prediction of soft-tissue deformations under complex single- and multi-grasper interactions. Our approach integrates Kelvinlet-based analytical priors with large-scale FEM data, capturing both linear and nonlinear tissue responses. This hybrid design improves predictive accuracy and physical plausibility across diverse neural architectures while maintaining the low-latency performance required for interactive applications. We validate our method on challenging surgical manipulation tasks involving standard laparoscopic grasping tools, demonstrating substantial improvements in deformation fidelity and temporal stability over existing baselines. These results establish Kelvinlet-augmented learning as a principled and computationally efficient paradigm for real-time, physics-aware soft-tissue simulation in surgical AI.
>
---
#### [replaced 019] Delta-Triplane Transformers as Occupancy World Models
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.07338v3](http://arxiv.org/pdf/2503.07338v3)**

> **作者:** Haoran Xu; Peixi Peng; Guang Tan; Yiqian Chang; Yisen Zhao; Yonghong Tian
>
> **摘要:** Occupancy World Models (OWMs) aim to predict future scenes via 3D voxelized representations of the environment to support intelligent motion planning. Existing approaches typically generate full future occupancy states from VAE-style latent encodings, which can be computationally expensive and redundant. We propose Delta-Triplane Transformers (DTT), a novel 4D OWM for autonomous driving, that introduces two key innovations: (1) a triplane based representation that encodes 3D occupancy more compactly than previous approaches, and (2) an incremental prediction strategy for OWM that models {\em changes} in occupancy rather than dealing with full states. The core insight is that changes in the compact 3D latent space are naturally sparser and easier to model, enabling higher accuracy with a lighter-weight architecture. Building on this representation, DTT extracts multi-scale motion features from historical data and iteratively predict future triplane deltas. These deltas are combined with past states to decode future occupancy and ego-motion trajectories. Extensive experiments demonstrate that DTT delivers a 1.44$\times$ speedup (26 FPS) over the state of the art, improves mean IoU to 30.85, and reduces the mean absolute planning error to 1.0 meters. Demo videos are provided in the supplementary material.
>
---
#### [replaced 020] Loosely coupled 4D-Radar-Inertial Odometry for Ground Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2411.17289v2](http://arxiv.org/pdf/2411.17289v2)**

> **作者:** Lucia Coto Elena; Fernando Caballero; Luis Merino
>
> **备注:** 24 pages, 6 figures, 4 tables, 33 references
>
> **摘要:** Accurate robot odometry is essential for autonomous navigation. While numerous techniques have been developed based on various sensor suites, odometry estimation using only radar and IMU remains an underexplored area. Radar proves particularly valuable in environments where traditional sensors, like cameras or LiDAR, may struggle, especially in low-light conditions or when faced with environmental challenges like fog, rain or smoke. However, despite its robustness, radar data is noisier and more prone to outliers, requiring specialized processing approaches. In this paper, we propose a graph-based optimization approach using a sliding window for radar-based odometry, designed to maintain robust relationships between poses by forming a network of connections, while keeping computational costs fixed (specially beneficial in long trajectories). Additionally, we introduce an enhancement in the ego-velocity estimation specifically for ground vehicles, both holonomic and non-holonomic, which subsequently improves the direct odometry input required by the optimizer. Finally, we present a comparative study of our approach against existing algorithms, showing how our pure odometry approach inproves the state of art in most trajectories of the NTU4DRadLM dataset, achieving promising results when evaluating key performance metrics.
>
---
#### [replaced 021] VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.21723v2](http://arxiv.org/pdf/2509.21723v2)**

> **作者:** Huayi Zhou; Kui Jia
>
> **备注:** under review
>
> **摘要:** Achieving generalizable bimanual manipulation requires systems that can learn efficiently from minimal human input while adapting to real-world uncertainties and diverse embodiments. Existing approaches face a dilemma: imitation policy learning demands extensive demonstrations to cover task variations, while modular methods often lack flexibility in dynamic scenes. We introduce VLBiMan, a framework that derives reusable skills from a single human example through task-aware decomposition, preserving invariant primitives as anchors while dynamically adapting adjustable components via vision-language grounding. This adaptation mechanism resolves scene ambiguities caused by background changes, object repositioning, or visual clutter without policy retraining, leveraging semantic parsing and geometric feasibility constraints. Moreover, the system inherits human-like hybrid control capabilities, enabling mixed synchronous and asynchronous use of both arms. Extensive experiments validate VLBiMan across tool-use and multi-object tasks, demonstrating: (1) a drastic reduction in demonstration requirements compared to imitation baselines, (2) compositional generalization through atomic skill splicing for long-horizon tasks, (3) robustness to novel but semantically similar objects and external disturbances, and (4) strong cross-embodiment transfer, showing that skills learned from human demonstrations can be instantiated on different robotic platforms without retraining. By bridging human priors with vision-language anchored adaptation, our work takes a step toward practical and versatile dual-arm manipulation in unstructured settings.
>
---
#### [replaced 022] InSpire: Vision-Language-Action Models with Intrinsic Spatial Reasoning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.13888v3](http://arxiv.org/pdf/2505.13888v3)**

> **作者:** Ji Zhang; Shihan Wu; Xu Luo; Hao Wu; Lianli Gao; Heng Tao Shen; Jingkuan Song
>
> **摘要:** Leveraging pretrained Vision-Language Models (VLMs) to map language instruction and visual observations to raw low-level actions, Vision-Language-Action models (VLAs) hold great promise for achieving general-purpose robotic systems. Despite their advancements, existing VLAs tend to spuriously correlate task-irrelevant visual features with actions, limiting their generalization capacity beyond the training data. To tackle this challenge, we propose Intrinsic Spatial Reasoning (InSpire), a simple yet effective approach that mitigates the adverse effects of spurious correlations by boosting the spatial reasoning ability of VLAs. Specifically, InSpire redirects the VLA's attention to task-relevant factors by prepending the question "In which direction is the [object] relative to the robot?" to the language instruction and aligning the answer "right/left/up/down/front/back/grasped" and predicted actions with ground-truth. Notably, InSpire can be used as a plugin to enhance existing autoregressive VLAs, requiring no extra training data or interaction with other large models. Extensive experimental results in both simulation and real-world environments demonstrate the effectiveness and flexibility of our approach.
>
---
#### [replaced 023] ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08052v2](http://arxiv.org/pdf/2506.08052v2)**

> **作者:** Yongkang Li; Kaixin Xiong; Xiangyu Guo; Fang Li; Sixu Yan; Gangwei Xu; Lijun Zhou; Long Chen; Haiyang Sun; Bing Wang; Kun Ma; Guang Chen; Hangjun Ye; Wenyu Liu; Xinggang Wang
>
> **摘要:** Recent studies have explored leveraging the world knowledge and cognitive capabilities of Vision-Language Models (VLMs) to address the long-tail problem in end-to-end autonomous driving. However, existing methods typically formulate trajectory planning as a language modeling task, where physical actions are output in the language space, potentially leading to issues such as format-violating outputs, infeasible actions, and slow inference speeds. In this paper, we propose ReCogDrive, a novel Reinforced Cognitive framework for end-to-end autonomous Driving, unifying driving understanding and planning by integrating an autoregressive model with a diffusion planner. First, to instill human driving cognition into the VLM, we introduce a hierarchical data pipeline that mimics the sequential cognitive process of human drivers through three stages: generation, refinement, and quality control. Building on this cognitive foundation, we then address the language-action mismatch by injecting the VLM's learned driving priors into a diffusion planner to efficiently generate continuous and stable trajectories. Furthermore, to enhance driving safety and reduce collisions, we introduce a Diffusion Group Relative Policy Optimization (DiffGRPO) stage, reinforcing the planner for enhanced safety and comfort. Extensive experiments on the NAVSIM and Bench2Drive benchmarks demonstrate that ReCogDrive achieves state-of-the-art performance. Additionally, qualitative results across diverse driving scenarios and DriveBench highlight the model's scene comprehension. All code, model weights, and datasets will be made publicly available to facilitate subsequent research.
>
---
#### [replaced 024] MASt3R-Fusion: Integrating Feed-Forward Visual Model with IMU, GNSS for High-Functionality SLAM
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.20757v2](http://arxiv.org/pdf/2509.20757v2)**

> **作者:** Yuxuan Zhou; Xingxing Li; Shengyu Li; Zhuohao Yan; Chunxi Xia; Shaoquan Feng
>
> **摘要:** Visual SLAM is a cornerstone technique in robotics, autonomous driving and extended reality (XR), yet classical systems often struggle with low-texture environments, scale ambiguity, and degraded performance under challenging visual conditions. Recent advancements in feed-forward neural network-based pointmap regression have demonstrated the potential to recover high-fidelity 3D scene geometry directly from images, leveraging learned spatial priors to overcome limitations of traditional multi-view geometry methods. However, the widely validated advantages of probabilistic multi-sensor information fusion are often discarded in these pipelines. In this work, we propose MASt3R-Fusion,a multi-sensor-assisted visual SLAM framework that tightly integrates feed-forward pointmap regression with complementary sensor information, including inertial measurements and GNSS data. The system introduces Sim(3)-based visualalignment constraints (in the Hessian form) into a universal metric-scale SE(3) factor graph for effective information fusion. A hierarchical factor graph design is developed, which allows both real-time sliding-window optimization and global optimization with aggressive loop closures, enabling real-time pose tracking, metric-scale structure perception and globally consistent mapping. We evaluate our approach on both public benchmarks and self-collected datasets, demonstrating substantial improvements in accuracy and robustness over existing visual-centered multi-sensor SLAM systems. The code will be released open-source to support reproducibility and further research (https://github.com/GREAT-WHU/MASt3R-Fusion).
>
---
#### [replaced 025] Autonomous Close-Proximity Photovoltaic Panel Coating Using a Quadcopter
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2509.10979v2](http://arxiv.org/pdf/2509.10979v2)**

> **作者:** Dimitri Jacquemont; Carlo Bosio; Teaya Yang; Ruiqi Zhang; Ozgur Orun; Shuai Li; Reza Alam; Thomas M. Schutzius; Simo A. Makiharju; Mark W. Mueller
>
> **备注:** 7 pages, 10 figures. Submitted to IEEE RA-L
>
> **摘要:** Photovoltaic (PV) panels are becoming increasingly widespread in the domain of renewable energy, and thus, small efficiency gains can have massive effects. Anti-reflective and self-cleaning coatings enhance panel performance but degrade over time, requiring periodic reapplication. Uncrewed Aerial Vehicles (UAVs) offer a flexible and autonomous way to apply protective coatings more often and at lower cost compared to traditional manual coating methods. In this letter, we propose a quadcopter-based system, equipped with a liquid dispersion mechanism, designed to automate such tasks. The localization stack only uses onboard sensors, relying on visual-inertial odometry and the relative position of the PV panel detected with respect to the quadcopter. The control relies on a model-based controller that accounts for the ground effect and the mass decrease of the quadcopter during liquid dispersion. We validate the autonomy capabilities of our system through extensive indoor and outdoor experiments.
>
---
#### [replaced 026] Self-Supervised Geometry-Guided Initialization for Robust Monocular Visual Odometry
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2406.00929v2](http://arxiv.org/pdf/2406.00929v2)**

> **作者:** Takayuki Kanai; Igor Vasiljevic; Vitor Guizilini; Kazuhiro Shintani
>
> **备注:** Project page: https://toyotafrc.github.io/SGInit-Proj/
>
> **摘要:** Monocular visual odometry is a key technology in various autonomous systems. Traditional feature-based methods suffer from failures due to poor lighting, insufficient texture, and large motions. In contrast, recent learning-based dense SLAM methods exploit iterative dense bundle adjustment to address such failure cases, and achieve robust and accurate localization in a wide variety of real environments, without depending on domain-specific supervision. However, despite its potential, the methods still struggle with scenarios involving large motion and object dynamics. In this study, we diagnose key weaknesses in a popular learning-based dense SLAM model (DROID-SLAM) by analyzing major failure cases on outdoor benchmarks and exposing various shortcomings of its optimization process. We then propose the use of self-supervised priors leveraging a frozen large-scale pre-trained monocular depth estimator to initialize the dense bundle adjustment process, leading to robust visual odometry without the need to fine-tune the SLAM backbone. Despite its simplicity, the proposed method demonstrates significant improvements on KITTI odometry, as well as the challenging DDAD benchmark.
>
---
#### [replaced 027] MimicDreamer: Aligning Human and Robot Demonstrations for Scalable VLA Training
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.22199v2](http://arxiv.org/pdf/2509.22199v2)**

> **作者:** Haoyun Li; Ivan Zhang; Runqi Ouyang; Xiaofeng Wang; Zheng Zhu; Zhiqin Yang; Zhentao Zhang; Boyuan Wang; Chaojun Ni; Wenkang Qin; Xinze Chen; Yun Ye; Guan Huang; Zhenbo Song; Xingang Wang
>
> **摘要:** Vision Language Action (VLA) models derive their generalization capability from diverse training data, yet collecting embodied robot interaction data remains prohibitively expensive. In contrast, human demonstration videos are far more scalable and cost-efficient to collect, and recent studies confirm their effectiveness in training VLA models. However, a significant domain gap persists between human videos and robot-executed videos, including unstable camera viewpoints, visual discrepancies between human hands and robotic arms, and differences in motion dynamics. To bridge this gap, we propose MimicDreamer, a framework that turns fast, low-cost human demonstrations into robot-usable supervision by jointly aligning vision, viewpoint, and actions to directly support policy training. For visual alignment, we propose H2R Aligner, a video diffusion model that generates high-fidelity robot demonstration videos by transferring motion from human manipulation footage. For viewpoint stabilization, EgoStabilizer is proposed, which canonicalizes egocentric videos via homography and inpaints occlusions and distortions caused by warping. For action alignment, we map human hand trajectories to the robot frame and apply a constrained inverse kinematics solver to produce feasible, low-jitter joint commands with accurate pose tracking. Empirically, VLA models trained purely on our synthesized human-to-robot videos achieve few-shot execution on real robots. Moreover, scaling training with human data significantly boosts performance compared to models trained solely on real robot data; our approach improves the average success rate by 14.7\% across six representative manipulation tasks.
>
---
#### [replaced 028] HUNT: High-Speed UAV Navigation and Tracking in Unstructured Environments via Instantaneous Relative Frames
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.19452v3](http://arxiv.org/pdf/2509.19452v3)**

> **作者:** Alessandro Saviolo; Jeffrey Mao; Giuseppe Loianno
>
> **摘要:** Search and rescue operations require unmanned aerial vehicles to both traverse unknown unstructured environments at high speed and track targets once detected. Achieving both capabilities under degraded sensing and without global localization remains an open challenge. Recent works on relative navigation have shown robust tracking by anchoring planning and control to a visible detected object, but cannot address navigation when no target is in the field of view. We present HUNT (High-speed UAV Navigation and Tracking), a real-time framework that unifies traversal, acquisition, and tracking within a single relative formulation. HUNT defines navigation objectives directly from onboard instantaneous observables such as attitude, altitude, and velocity, enabling reactive high-speed flight during search. Once a target is detected, the same perception-control pipeline transitions seamlessly to tracking. Outdoor experiments in dense forests, container compounds, and search-and-rescue operations with vehicles and mannequins demonstrate robust autonomy where global methods fail.
>
---
#### [replaced 029] Guided Reinforcement Learning for Omnidirectional 3D Jumping in Quadruped Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.16481v2](http://arxiv.org/pdf/2507.16481v2)**

> **作者:** Riccardo Bussola; Michele Focchi; Giulio Turrisi; Claudio Semini; Luigi Palopoli
>
> **摘要:** Jumping poses a significant challenge for quadruped robots, despite being crucial for many operational scenarios. While optimisation methods exist for controlling such motions, they are often time-consuming and demand extensive knowledge of robot and terrain parameters, making them less robust in real-world scenarios. Reinforcement learning (RL) is emerging as a viable alternative, yet conventional end-to-end approaches lack efficiency in terms of sample complexity, requiring extensive training in simulations, and predictability of the final motion, which makes it difficult to certify the safety of the final motion. To overcome these limitations, this paper introduces a novel guided reinforcement learning approach that leverages physical intuition for efficient and explainable jumping, by combining B\'ezier curves with a Uniformly Accelerated Rectilinear Motion (UARM) model. Extensive simulation and experimental results clearly demonstrate the advantages of our approach over existing alternatives.
>
---
#### [replaced 030] EvoAgent: Self-evolving Agent with Continual World Model for Long-Horizon Tasks
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.05907v2](http://arxiv.org/pdf/2502.05907v2)**

> **作者:** Tongtong Feng; Xin Wang; Zekai Zhou; Ren Wang; Yuwei Zhan; Guangyao Li; Qing Li; Wenwu Zhu
>
> **摘要:** Completing Long-Horizon (LH) tasks in open-ended worlds is an important yet difficult problem for embodied agents. Existing approaches suffer from two key challenges: (1) they heavily rely on experiences obtained from human-created data or curricula, failing to autonomously update and select multimodal experiences, and (2) they may encounter catastrophic forgetting issues when faced with new tasks, failing to autonomously update world knowledge. To solve these challenges, this paper presents {\it EvoAgent}, a self-evolving agent with a continual World Model (WM), which can autonomously complete various LH tasks across environments through self-planning, self-control, and self-reflection, without human intervention. Our proposed EvoAgent contains three modules, i.e., i) the memory-driven planner which uses an LLM along with the WM and interaction memory, to convert LH tasks into executable sub-tasks; ii) the WM-guided action controller which leverages WM to generate low-level actions and incorporates a self-verification mechanism to update multimodal experiences; iii) the experience-inspired reflector which implements a two-stage curriculum learning algorithm to select experiences for task-adaptive WM updates. Moreover, we develop a continual World Model for EvoAgent, which can autonomously update the multimodal experience pool and world knowledge through closed-loop dynamics. We conducted extensive experiments on Minecraft and Atair, compared with existing methods, EvoAgent can achieve an average success rate improvement of 105% and reduce ineffective actions by more than 6x.
>
---
#### [replaced 031] An Empirical Study on the Computation Budget of Co-Optimization of Robot Design and Control in Simulation
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.08621v2](http://arxiv.org/pdf/2409.08621v2)**

> **作者:** Etor Arza; Frank Veenstra; Tønnes F. Nygaard; Kyrre Glette
>
> **摘要:** The design (shape) of a robot is usually decided before the control is implemented. This might limit how well the design is adapted to a task, as the suitability of the design is given by how well the robot performs in the task, which requires both a design and a controller. The co-optimization or simultaneous optimization of the design and control of robots addresses this limitation by producing a design and control that are both adapted to the task. This paper investigates some of the challenges inherent in the co-optimization of design and control in simulation. The results show that reducing how well the controllers are trained during the co-optimization process significantly improves the robot's performance when considering a second phase in which the controller for the best design is retrained with additional resources. In addition, the results demonstrate that the computation budget allocated to training the controller for each design influences design complexity, with simpler designs associated with lower training budgets. This paper experimentally studies key questions discussed in other works in the literature on the co-optimization of design and control of robots in simulation in four different co-optimization problems.
>
---
#### [replaced 032] Prior Reinforce: Mastering Agile Tasks with Limited Trials
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.21916v2](http://arxiv.org/pdf/2505.21916v2)**

> **作者:** Yihang Hu; Pingyue Sheng; Yuyang Liu; Shengjie Wang; Yang Gao
>
> **摘要:** Embodied robots nowadays can already handle many real-world manipulation tasks. However, certain other real-world tasks involving dynamic processes (e.g., shooting a basketball into a hoop) are highly agile and impose high precision requirements on the outcomes, presenting additional challenges for methods primarily designed for quasi-static manipulations. This leads to increased efforts in costly data collection, laborious reward design, or complex motion planning. Such tasks, however, are far less challenging for humans. Say a novice basketball player typically needs only about 10 attempts to make their first successful shot, by roughly imitating some motion priors and then iteratively adjusting their motion based on the past outcomes. Inspired by this human learning paradigm, we propose Prior Reinforce(P.R.), a simple and scalable approach which first learns a motion pattern from very few demonstrations, then iteratively refines its generated motions based on feedback of a few real-world trials, until reaching a specific goal. Experiments demonstrated that Prior Reinforce can learn and accomplish a wide range of goal-conditioned agile dynamic tasks with human-level precision and efficiency directly in real-world, such as throwing a basketball into the hoop in fewer than 10 trials. Project website:https://adap-robotics.github.io/.
>
---
#### [replaced 033] Diffusion-Based Impedance Learning for Contact-Rich Manipulation Tasks
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.19696v2](http://arxiv.org/pdf/2509.19696v2)**

> **作者:** Noah Geiger; Tamim Asfour; Neville Hogan; Johannes Lachner
>
> **备注:** 15 pages, 12 figures
>
> **摘要:** Learning methods excel at motion generation in the information domain but are not primarily designed for physical interaction in the energy domain. Impedance Control shapes physical interaction but requires task-aware tuning by selecting feasible impedance parameters. We present Diffusion-Based Impedance Learning, a framework that combines both domains. A Transformer-based Diffusion Model with cross-attention to external wrenches reconstructs a simulated Zero-Force Trajectory (sZFT). This captures both translational and rotational task-space behavior. For rotations, we introduce a novel SLERP-based quaternion noise scheduler that ensures geometric consistency. The reconstructed sZFT is then passed to an energy-based estimator that updates stiffness and damping parameters. A directional rule is applied that reduces impedance along non task axes while preserving rigidity along task directions. Training data were collected for a parkour scenario and robotic-assisted therapy tasks using teleoperation with Apple Vision Pro. With only tens of thousands of samples, the model achieved sub-millimeter positional accuracy and sub-degree rotational accuracy. Its compact model size enabled real-time torque control and autonomous stiffness adaptation on a KUKA LBR iiwa robot. The controller achieved smooth parkour traversal within force and velocity limits and 30/30 success rates for cylindrical, square, and star peg insertions without any peg-specific demonstrations in the training data set. All code for the Transformer-based Diffusion Model, the robot controller, and the Apple Vision Pro telemanipulation framework is publicly available. These results mark an important step towards Physical AI, fusing model-based control for physical interaction with learning-based methods for trajectory generation.
>
---
#### [replaced 034] CoT-TL: Low-Resource Temporal Knowledge Representation of Planning Instructions Using Chain-of-Thought Reasoning
- **分类: cs.RO; cs.CL; cs.FL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.16207v2](http://arxiv.org/pdf/2410.16207v2)**

> **作者:** Kumar Manas; Stefan Zwicklbauer; Adrian Paschke
>
> **备注:** Proceedings of the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024), Abu Dhabi 14-18 October 2024
>
> **摘要:** Autonomous agents often face the challenge of interpreting uncertain natural language instructions for planning tasks. Representing these instructions as Linear Temporal Logic (LTL) enables planners to synthesize actionable plans. We introduce CoT-TL, a data-efficient in-context learning framework for translating natural language specifications into LTL representations. CoT-TL addresses the limitations of large language models, which typically rely on extensive fine-tuning data, by extending chain-of-thought reasoning and semantic roles to align with the requirements of formal logic creation. This approach enhances the transparency and rationale behind LTL generation, fostering user trust. CoT-TL achieves state-of-the-art accuracy across three diverse datasets in low-data scenarios, outperforming existing methods without fine-tuning or intermediate translations. To improve reliability and minimize hallucinations, we incorporate model checking to validate the syntax of the generated LTL output. We further demonstrate CoT-TL's effectiveness through ablation studies and evaluations on unseen LTL structures and formulas in a new dataset. Finally, we validate CoT-TL's practicality by integrating it into a QuadCopter for multi-step drone planning based on natural language instructions. Project details: \href{https://github.com/kumarmanas/TAMP\_COT\_TL}{https://github.com/kumarmanas/TAMP\_COT\_TL}
>
---
#### [replaced 035] Diffusion-Based mmWave Radar Point Cloud Enhancement Driven by Range Images
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.02300v2](http://arxiv.org/pdf/2503.02300v2)**

> **作者:** Ruixin Wu; Zihan Li; Jin Wang; Xiangyu Xu; Zhi Zheng; Kaixiang Huang; Guodong Lu
>
> **备注:** 8 pages, 7 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Millimeter-wave (mmWave) radar has attracted significant attention in robotics and autonomous driving. However, despite the perception stability in harsh environments, the point cloud generated by mmWave radar is relatively sparse while containing significant noise, which limits its further development. Traditional mmWave radar enhancement approaches often struggle to leverage the effectiveness of diffusion models in super-resolution, largely due to the unnatural range-azimuth heatmap (RAH) or bird's eye view (BEV) representation. To overcome this limitation, we propose a novel method that pioneers the application of fusing range images with image diffusion models, achieving accurate and dense mmWave radar point clouds that are similar to LiDAR. Benefitting from the projection that aligns with human observation, the range image representation of mmWave radar is close to natural images, allowing the knowledge from pre-trained image diffusion models to be effectively transferred, significantly improving the overall performance. Extensive evaluations on both public datasets and self-constructed datasets demonstrate that our approach provides substantial improvements, establishing a new state-of-the-art performance in generating truly three-dimensional LiDAR-like point clouds via mmWave radar. Code will be released after publication.
>
---
#### [replaced 036] TReF-6: Inferring Task-Relevant Frames from a Single Demonstration for One-Shot Skill Generalization
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.00310v2](http://arxiv.org/pdf/2509.00310v2)**

> **作者:** Yuxuan Ding; Shuangge Wang; Tesca Fitzgerald
>
> **摘要:** Robots often struggle to generalize from a single demonstration due to the lack of a transferable and interpretable spatial representation. In this work, we introduce TReF-6, a method that infers a simplified, abstracted 6DoF Task-Relevant Frame from a single trajectory. Our approach identifies an influence point purely from the trajectory geometry to define the origin for a local frame, which serves as a reference for parameterizing a Dynamic Movement Primitive (DMP). This influence point captures the task's spatial structure, extending the standard DMP formulation beyond start-goal imitation. The inferred frame is semantically grounded via a vision-language model and localized in novel scenes by Grounded-SAM, enabling functionally consistent skill generalization. We validate TReF-6 in simulation and demonstrate robustness to trajectory noise. We further deploy an end-to-end pipeline on real-world manipulation tasks, showing that TReF-6 supports one-shot imitation learning that preserves task intent across diverse object configurations.
>
---
#### [replaced 037] FindingDory: A Benchmark to Evaluate Memory in Embodied Agents
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.15635v2](http://arxiv.org/pdf/2506.15635v2)**

> **作者:** Karmesh Yadav; Yusuf Ali; Gunshi Gupta; Yarin Gal; Zsolt Kira
>
> **备注:** Our dataset and code can be found at: https://findingdory-benchmark.github.io/
>
> **摘要:** Large vision-language models have recently demonstrated impressive performance in planning and control tasks, driving interest in their application to real-world robotics. However, deploying these models for reasoning in embodied contexts is limited by their ability to incorporate long-term experience collected across multiple days and represented by vast collections of images. Current VLMs typically struggle to process more than a few hundred images concurrently, highlighting the need for more efficient mechanisms to handle long-term memory in embodied settings. To effectively evaluate these models for long-horizon control, a benchmark must specifically target scenarios where memory is crucial for success. Existing long-video QA benchmarks overlook embodied challenges like object manipulation and navigation, which demand low-level skills and fine-grained reasoning over past interactions. Moreover, effective memory integration in embodied agents involves both recalling relevant historical information and executing actions based on that information, making it essential to study these aspects together rather than in isolation. In this work, we introduce a new benchmark for long-range embodied tasks in the Habitat simulator. This benchmark evaluates memory-based capabilities across 60 tasks requiring sustained engagement and contextual awareness in an environment. The tasks can also be procedurally extended to longer and more challenging versions, enabling scalable evaluation of memory and reasoning. We also present baselines that integrate state-of-the-art VLMs with low level navigation policies, assessing their performance on these memory-intensive tasks and highlight areas for improvement.
>
---
#### [replaced 038] S2S-Net: Addressing the Domain Gap of Heterogeneous Sensor Systems in LiDAR-Based Collective Perception
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.17399v2](http://arxiv.org/pdf/2504.17399v2)**

> **作者:** Sven Teufel; Jörg Gamerdinger; Oliver Bringmann
>
> **摘要:** Collective Perception (CP) has emerged as a promising approach to overcome the limitations of individual perception in the context of autonomous driving. Various approaches have been proposed to realize collective perception; however, the Sensor2Sensor domain gap that arises from the utilization of different sensor systems in Connected and Automated Vehicles (CAVs) remains mostly unaddressed. This is primarily due to the paucity of datasets containing heterogeneous sensor setups among the CAVs. The recently released SCOPE datasets address this issue by providing data from three different LiDAR sensors for each CAV. This study is the first to address the Sensor2Sensor domain gap in vehicle-to-vehicle (V2V) collective perception. First, we present our sensor-domain robust architecture S2S-Net. Then an in-depth analysis of the Sensor2Sensor domain adaptation capabilities of state-of-the-art CP methods and S2S-Net is conducted on the SCOPE dataset. This study shows that, all evaluated state-of-the-art mehtods for collective perception highly suffer from the Sensor2Sensor domain gap, while S2S-Net demonstrates the capability to maintain very high performance in unseen sensor domains and outperforms the evaluated state-of-the-art methods by up to 44 percentage points.
>
---
#### [replaced 039] TGRPO :Fine-tuning Vision-Language-Action Model via Trajectory-wise Group Relative Policy Optimization
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.08440v3](http://arxiv.org/pdf/2506.08440v3)**

> **作者:** Zengjue Chen; Runliang Niu; He Kong; Qi Wang; Qianli Xing; Zipei Fan
>
> **摘要:** Visual-Language-Action (VLA) models have demonstrated strong cross-scenario generalization capabilities in various robotic tasks through large-scale pre-training and task-specific fine-tuning. However, their training paradigm mainly relies on manually collected successful demonstrations, making it difficult to adapt to complex environments when encountering out-of-distribution (OOD) scenarios or execution biases. While Reinforcement Learning (RL) provides a closed-loop optimization framework via active trial-and-error mechanism, it suffers from sparse rewards, high variance, and unstable optimization in long-horizon robotic tasks. To address these limitations, we propose Trajectory-based Group Relative Policy Optimization (TGRPO), an online RL-based training framework for VLA models. TGRPO leverages task analysis generated by a large language model to automatically construct dense reward functions, providing fine-grained feedback to accelerate convergence and improve credit assignment. The core of our method is a group-based strategy that samples and normalizes multiple trajectories in parallel, reducing variance through relative comparison. By integrating trajectory-level and step-level advantage estimation, TGRPO captures both global and local optimization signals without relying on a value network. Experiments on four task categories of the LIBERO benchmark demonstrate that TGRPO achieves an average success rate of 80.7\%, which is 4.2\% higher than that of Supervised Fine-Tuning (SFT) and outperforms other representative RL-based post-training methods.
>
---
#### [replaced 040] What Do You Need for Diverse Trajectory Composition in Diffusion Planning?
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.18083v2](http://arxiv.org/pdf/2505.18083v2)**

> **作者:** Quentin Clark; Florian Shkurti
>
> **备注:** 9 Pages
>
> **摘要:** In planning, stitching is an ability of algorithms to piece together sub-trajectories of data they are trained on to generate new and diverse behaviours. While stitching is historically a strength of offline reinforcement learning, recent generative behavioural cloning (BC) methods have also shown proficiency at stitching. However, the main factors behind this are poorly understood, hindering the development of new algorithms that can reliably stitch. Focusing on diffusion planners trained via BC, we find two properties are needed to compose: \emph{positional equivariance} and \emph{local receptiveness}. We use these two properties to explain architecture, data, and inference choices in existing generative BC methods based on diffusion planning, including replanning frequency, data augmentation, and data scaling. Experimental comparisions show that (1) while locality is more important than positional equivariance in creating a diffusion planner capable of composition, both are crucial (2) enabling these properties through relatively simple architecture choices can be competitive with more computationally expensive methods such as replanning or scaling data, and (3) simple inpainting-based guidance can guide architecturally compositional models to enable generalization in goal-conditioned settings.
>
---
#### [replaced 041] Follow-Bench: A Unified Motion Planning Benchmark for Socially-Aware Robot Person Following
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.10796v2](http://arxiv.org/pdf/2509.10796v2)**

> **作者:** Hanjing Ye; Weixi Situ; Jianwei Peng; Yu Zhan; Bingyi Xia; Kuanqi Cai; Hong Zhang
>
> **备注:** Project page: https://follow-bench.github.io/
>
> **摘要:** Robot person following (RPF) -- mobile robots that follow and assist a specific person -- has emerging applications in personal assistance, security patrols, eldercare, and logistics. To be effective, such robots must follow the target while ensuring safety and comfort for both the target and surrounding people. In this work, we present the first end-to-end study of RPF, which (i) surveys representative scenarios, motion-planning methods, and evaluation metrics with a focus on safety and comfort; (ii) introduces Follow-Bench, a unified benchmark simulating diverse scenarios, including various target trajectory patterns, dynamic-crowd flows, and environmental layouts; and (iii) re-implements six popular RPF planners, ensuring that both safety and comfort are systematically considered. Moreover, we evaluate the two highest-performing planners from our benchmark on a differential-drive robot to provide insights into real-world deployment. Extensive simulation and real-world experiments provide quantitative insights into the safety-comfort trade-offs of existing planners, while revealing open challenges and future research directions.
>
---
#### [replaced 042] A Hybrid TDMA/CSMA Protocol for Time-Sensitive Traffic in Robot Applications
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2509.06119v2](http://arxiv.org/pdf/2509.06119v2)**

> **作者:** Shiqi Xu; Lihao Zhang; Yuyang Du; Qun Yang; Soung Chang Liew
>
> **摘要:** Recent progress in robotics has underscored the demand for real-time control in applications such as manufacturing, healthcare, and autonomous systems, where the timely delivery of mission-critical commands under heterogeneous robotic traffic is paramount for operational efficacy and safety. In these scenarios, mission-critical traffic follows a strict deadline-constrained communication pattern: commands must arrive within defined QoS deadlines, otherwise late arrivals can degrade performance or destabilize control loops.In this work, we demonstrate on a real-time SDR platform that CSMA, widely adopted in robotic communications,suffers severe degradation under high robot traffic loads, with contention-induced collisions and delays disrupting the on-time arrival of mission-critical packets. To address this problem, we propose an IEEE 802.11-compatible hybrid TDMA/CSMA protocol that combines TDMA's deterministic slot scheduling with CSMA's adaptability for heterogeneous robot traffic.The protocol achieves collision-free, low-latency mission-critical command delivery and IEEE 802.11 compatibility through the synergistic integration of sub-microsecond PTP-based slot synchronization-essential for establishing precise timing for TDMA, a three-session superframe with dynamic TDMA allocation for structured and adaptable traffic management,and beacon-NAV protection to preemptively secure these critical communication sessions from interference. Emulation experiments on real-time SDR testbed and Robot Operating System (ROS) simulation show that the proposed protocol reduces missed-deadline errors by 93% compared to the CSMA baseline. In high-speed robot path-tracking ROS simulations, the protocol lowers Root Mean Square (RMS) trajectory error by up to 90% compared with a CSMA baseline, all while maintaining throughput for non-critical traffic within +-2%.
>
---
#### [replaced 043] GLEAM: Learning Generalizable Exploration Policy for Active Mapping in Complex 3D Indoor Scenes
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.20294v2](http://arxiv.org/pdf/2505.20294v2)**

> **作者:** Xiao Chen; Tai Wang; Quanyi Li; Tao Huang; Jiangmiao Pang; Tianfan Xue
>
> **备注:** Accepted by ICCV 2025. Project page: https://xiao-chen.tech/gleam/
>
> **摘要:** Generalizable active mapping in complex unknown environments remains a critical challenge for mobile robots. Existing methods, constrained by insufficient training data and conservative exploration strategies, exhibit limited generalizability across scenes with diverse layouts and complex connectivity. To enable scalable training and reliable evaluation, we introduce GLEAM-Bench, the first large-scale benchmark designed for generalizable active mapping with 1,152 diverse 3D scenes from synthetic and real-scan datasets. Building upon this foundation, we propose GLEAM, a unified generalizable exploration policy for active mapping. Its superior generalizability comes mainly from our semantic representations, long-term navigable goals, and randomized strategies. It significantly outperforms state-of-the-art methods, achieving 66.50% coverage (+9.49%) with efficient trajectories and improved mapping accuracy on 128 unseen complex scenes. Project page: https://xiao-chen.tech/gleam/.
>
---
#### [replaced 044] Automotive-ENV: Benchmarking Multimodal Agents in Vehicle Interface Systems
- **分类: cs.RO; cs.CL; F.2.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2509.21143v2](http://arxiv.org/pdf/2509.21143v2)**

> **作者:** Junfeng Yan; Biao Wu; Meng Fang; Ling Chen
>
> **备注:** 10 pages, 5 figures,
>
> **摘要:** Multimodal agents have demonstrated strong performance in general GUI interactions, but their application in automotive systems has been largely unexplored. In-vehicle GUIs present distinct challenges: drivers' limited attention, strict safety requirements, and complex location-based interaction patterns. To address these challenges, we introduce Automotive-ENV, the first high-fidelity benchmark and interaction environment tailored for vehicle GUIs. This platform defines 185 parameterized tasks spanning explicit control, implicit intent understanding, and safety-aware tasks, and provides structured multimodal observations with precise programmatic checks for reproducible evaluation. Building on this benchmark, we propose ASURADA, a geo-aware multimodal agent that integrates GPS-informed context to dynamically adjust actions based on location, environmental conditions, and regional driving norms. Experiments show that geo-aware information significantly improves success on safety-aware tasks, highlighting the importance of location-based context in automotive environments. We will release Automotive-ENV, complete with all tasks and benchmarking tools, to further the development of safe and adaptive in-vehicle agents.
>
---
#### [replaced 045] Vintix: Action Model via In-Context Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.19400v2](http://arxiv.org/pdf/2501.19400v2)**

> **作者:** Andrey Polubarov; Nikita Lyubaykin; Alexander Derevyagin; Ilya Zisman; Denis Tarasov; Alexander Nikulin; Vladislav Kurenkov
>
> **备注:** ICML 2025, Poster
>
> **摘要:** In-Context Reinforcement Learning (ICRL) represents a promising paradigm for developing generalist agents that learn at inference time through trial-and-error interactions, analogous to how large language models adapt contextually, but with a focus on reward maximization. However, the scalability of ICRL beyond toy tasks and single-domain settings remains an open challenge. In this work, we present the first steps toward scaling ICRL by introducing a fixed, cross-domain model capable of learning behaviors through in-context reinforcement learning. Our results demonstrate that Algorithm Distillation, a framework designed to facilitate ICRL, offers a compelling and competitive alternative to expert distillation to construct versatile action models. These findings highlight the potential of ICRL as a scalable approach for generalist decision-making systems. Code released at https://github.com/dunnolab/vintix
>
---
#### [replaced 046] Hypo-paradoxical Linkages: Linkages That Should Move-But Don't
- **分类: physics.soc-ph; cs.RO; 70B15**

- **链接: [http://arxiv.org/pdf/2507.20371v2](http://arxiv.org/pdf/2507.20371v2)**

> **作者:** Nir Shvalb; Oded Medina
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** While paradoxical linkages famously violate the Chebyshev-Grubler-Kutzbach criterion by exhibiting unexpected mobility, we identify an opposing phenomenon: a class of linkages that appear mobile according to the same criterion, yet are in fact rigid. We refer to these as hypo-paradoxical linkages, and proceed to analyze and illustrate their behavior. We use the same tools to further explain the unexpected positive mobility of Bennet mechanism.
>
---
#### [replaced 047] Learning Smooth State-Dependent Traversability from Dense Point Clouds
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04362v2](http://arxiv.org/pdf/2506.04362v2)**

> **作者:** Zihao Dong; Alan Papalia; Leonard Jung; Alenna Spiro; Philip R. Osteen; Christa S. Robison; Michael Everett
>
> **备注:** 18 pages, 13 figures
>
> **摘要:** A key open challenge in off-road autonomy is that the traversability of terrain often depends on the vehicle's state. In particular, some obstacles are only traversable from some orientations. However, learning this interaction by encoding the angle of approach as a model input demands a large and diverse training dataset and is computationally inefficient during planning due to repeated model inference. To address these challenges, we present SPARTA, a method for estimating approach angle conditioned traversability from point clouds. Specifically, we impose geometric structure into our network by outputting a smooth analytical function over the 1-Sphere that predicts risk distribution for any angle of approach with minimal overhead and can be reused for subsequent queries. The function is composed of Fourier basis functions, which has important advantages for generalization due to their periodic nature and smoothness. We demonstrate SPARTA both in a high-fidelity simulation platform, where our model achieves a 91\% success rate crossing a 40m boulder field (compared to 73\% for the baseline), and on hardware, illustrating the generalization ability of the model to real-world settings. Our code will be available at https://github.com/neu-autonomy/SPARTA.
>
---
#### [replaced 048] LaVA-Man: Learning Visual Action Representations for Robot Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.19391v2](http://arxiv.org/pdf/2508.19391v2)**

> **作者:** Chaoran Zhu; Hengyi Wang; Yik Lung Pang; Changjae Oh
>
> **摘要:** Visual-textual understanding is essential for language-guided robot manipulation. Recent works leverage pre-trained vision-language models to measure the similarity between encoded visual observations and textual instructions, and then train a model to map this similarity to robot actions. However, this two-step approach limits the model to capture the relationship between visual observations and textual instructions, leading to reduced precision in manipulation tasks. We propose to learn visual-textual associations through a self-supervised pretext task: reconstructing a masked goal image conditioned on an input image and textual instructions. This formulation allows the model to learn visual-action representations without robot action supervision. The learned representations can then be fine-tuned for manipulation tasks with only a few demonstrations. We also introduce the \textit{Omni-Object Pick-and-Place} dataset, which consists of annotated robot tabletop manipulation episodes, including 180 object classes and 3,200 instances with corresponding textual instructions. This dataset enables the model to acquire diverse object priors and allows for a more comprehensive evaluation of its generalisation capability across object instances. Experimental results on the five benchmarks, including both simulated and real-robot validations, demonstrate that our method outperforms prior art.
>
---
#### [replaced 049] Learning Probabilistic Obstacle Spaces from Data-driven Uncertainty using Neural Networks
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2411.14356v2](http://arxiv.org/pdf/2411.14356v2)**

> **作者:** Jun Xiang; Jun Chen
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** Identifying the obstacle space is crucial for path planning. However, generating an accurate obstacle space remains a significant challenge due to various sources of uncertainty, including motion, behavior, and perception limitations. Even though an autonomous system can operate with an inaccurate obstacle space by being over-conservative and using redundant sensors, a more accurate obstacle space generator can reduce both path planning costs and hardware costs. Existing generation methods that generate high-quality output are all computationally expensive. Traditional methods, such as filtering, sensor fusion and data-driven estimators, face significant computational challenges or require large amounts of data, which limits their applicability in realistic scenarios. In this paper, we propose leveraging neural networks, commonly used in imitation learning, to mimic expert methods for modeling uncertainty and generating confidence regions for obstacle positions, which we refer to as the probabilistic obstacle space. The network is trained using a multi-label, supervised learning approach. We adopt a fine-tuned convex approximation method as the expert to construct training datasets. After training, given only a small number of samples, the neural network can accurately replicate the probabilistic obstacle space while achieving substantially faster generation speed. Moreover, the resulting obstacle space is convex, making it more convenient for subsequent path planning.
>
---
#### [replaced 050] AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15298v4](http://arxiv.org/pdf/2505.15298v4)**

> **作者:** Kangan Qian; Sicong Jiang; Yang Zhong; Ziang Luo; Zilin Huang; Tianze Zhu; Kun Jiang; Mengmeng Yang; Zheng Fu; Jinyu Miao; Yining Shi; He Zhe Lim; Li Liu; Tianbao Zhou; Huang Yu; Yifei Hu; Guang Li; Guang Chen; Hao Ye; Lijun Sun; Diange Yang
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Vision-Language Models (VLMs) show promise for autonomous driving, yet their struggle with hallucinations, inefficient reasoning, and limited real-world validation hinders accurate perception and robust step-by-step reasoning. To overcome this, we introduce \textbf{AgentThink}, a pioneering unified framework that integrates Chain-of-Thought (CoT) reasoning with dynamic, agent-style tool invocation for autonomous driving tasks. AgentThink's core innovations include: \textbf{(i) Structured Data Generation}, which establishes an autonomous driving tool library to automatically construct structured, self-verified reasoning data explicitly incorporating tool usage for diverse driving scenarios; \textbf{(ii) A Two-stage Training Pipeline}, employing Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO) to equip VLMs with the capability for autonomous tool invocation; and \textbf{(iii) Agent-style Tool-Usage Evaluation}, introducing a novel multi-tool assessment protocol to rigorously evaluate the model's tool invocation and utilization. Experiments on the DriveLMM-o1 benchmark demonstrate that AgentThink significantly boosts overall reasoning scores by \textbf{53.91%} and enhances answer accuracy by \textbf{33.54%}, while markedly improving reasoning quality and consistency. Furthermore, ablation studies and robust zero-shot/few-shot generalization experiments across various benchmarks underscore its powerful capabilities. These findings highlight a promising trajectory for developing trustworthy and tool-aware autonomous driving models. Code is available at https://github.com/curryqka/AgentThink.
>
---
#### [replaced 051] MARG: MAstering Risky Gap Terrains for Legged Robots with Elevation Mapping
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.20036v2](http://arxiv.org/pdf/2509.20036v2)**

> **作者:** Yinzhao Dong; Ji Ma; Liu Zhao; Wanyue Li; Peng Lu
>
> **摘要:** Deep Reinforcement Learning (DRL) controllers for quadrupedal locomotion have demonstrated impressive performance on challenging terrains, allowing robots to execute complex skills such as climbing, running, and jumping. However, existing blind locomotion controllers often struggle to ensure safety and efficient traversal through risky gap terrains, which are typically highly complex, requiring robots to perceive terrain information and select appropriate footholds during locomotion accurately. Meanwhile, existing perception-based controllers still present several practical limitations, including a complex multi-sensor deployment system and expensive computing resource requirements. This paper proposes a DRL controller named MAstering Risky Gap Terrains (MARG), which integrates terrain maps and proprioception to dynamically adjust the action and enhance the robot's stability in these tasks. During the training phase, our controller accelerates policy optimization by selectively incorporating privileged information (e.g., center of mass, friction coefficients) that are available in simulation but unmeasurable directly in real-world deployments due to sensor limitations. We also designed three foot-related rewards to encourage the robot to explore safe footholds. More importantly, a terrain map generation (TMG) model is proposed to reduce the drift existing in mapping and provide accurate terrain maps using only one LiDAR, providing a foundation for zero-shot transfer of the learned policy. The experimental results indicate that MARG maintains stability in various risky terrain tasks.
>
---
#### [replaced 052] Efficient Iterative Proximal Variational Inference Motion Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2411.03416v4](http://arxiv.org/pdf/2411.03416v4)**

> **作者:** Zinuo Chang; Hongzhe Yu; Patricio Vela; Yongxin Chen
>
> **备注:** 13 pages
>
> **摘要:** We cast motion planning under uncertainty as a stochastic optimal control problem, where the optimal posterior distribution has an explicit form. To approximate this posterior, this work frames an optimization problem in the space of Gaussian distributions by solving a Variational Inference (VI) in the path distribution space. For linear-Gaussian stochastic dynamics, a proximal algorithm is proposed to solve for an optimal Gaussian proposal iteratively. The computational bottleneck is evaluating the gradients with respect to the proposal over a dense trajectory. To tackle this issue, the sparse planning factor graph and Gaussian Belief Propagation (GBP) are exploited, allowing for parallel computation of these gradients on Graphics Processing Units (GPUs). We term the novel paradigm the \textit{Parallel Gaussian Variational Inference Motion Planning (P-GVIMP)}. Building on the efficient algorithm for linear Gaussian systems, we then propose an iterative paradigm based on Statistical Linear Regression (SLR) techniques to solve planning problems for nonlinear stochastic systems, where the P-GVIMP serves as a sub-routine for the linearized time-varying system at each iteration. The proposed framework is validated on various robotic systems, demonstrating significant speed acceleration achieved by leveraging parallel computation and successful planning solutions for nonlinear systems under uncertainty. An open-sourced implementation is presented at \href{https://github.com/hzyu17/VIMP}{https://github.com/hzyu17/VIMP}.
>
---
#### [replaced 053] Beyond Human Demonstrations: Diffusion-Based Reinforcement Learning to Generate Data for VLA Training
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.19752v2](http://arxiv.org/pdf/2509.19752v2)**

> **作者:** Rushuai Yang; Hangxing Wei; Ran Zhang; Zhiyuan Feng; Xiaoyu Chen; Tong Li; Chuheng Zhang; Li Zhao; Jiang Bian; Xiu Su; Yi Chen
>
> **摘要:** Vision-language-action (VLA) models have shown strong generalization across tasks and embodiments; however, their reliance on large-scale human demonstrations limits their scalability owing to the cost and effort of manual data collection. Reinforcement learning (RL) offers a potential alternative to generate demonstrations autonomously, yet conventional RL algorithms often struggle on long-horizon manipulation tasks with sparse rewards. In this paper, we propose a modified diffusion policy optimization algorithm to generate high-quality and low-variance trajectories, which contributes to a diffusion RL-powered VLA training pipeline. Our algorithm benefits from not only the high expressiveness of diffusion models to explore complex and diverse behaviors but also the implicit regularization of the iterative denoising process to yield smooth and consistent demonstrations. We evaluate our approach on the LIBERO benchmark, which includes 130 long-horizon manipulation tasks, and show that the generated trajectories are smoother and more consistent than both human demonstrations and those from standard Gaussian RL policies. Further, training a VLA model exclusively on the diffusion RL-generated data achieves an average success rate of 81.9%, which outperforms the model trained on human data by +5.3% and that on Gaussian RL-generated data by +12.6%. The results highlight our diffusion RL as an effective alternative for generating abundant, high-quality, and low-variance demonstrations for VLA models.
>
---
#### [replaced 054] Deep Reinforcement Learning for Bipedal Locomotion: A Brief Survey
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2404.17070v5](http://arxiv.org/pdf/2404.17070v5)**

> **作者:** Lingfan Bao; Joseph Humphreys; Tianhu Peng; Chengxu Zhou
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Bipedal robots are gaining global recognition due to their potential applications and advancements in artificial intelligence, particularly through Deep Reinforcement Learning (DRL). While DRL has significantly advanced bipedal locomotion, the development of a unified framework capable of handling a wide range of tasks remains an ongoing challenge. This survey systematically categorises, compares, and analyses existing DRL frameworks for bipedal locomotion, organising them into end-to-end and hierarchical control schemes. End-to-end frameworks are evaluated based on their learning approaches, while hierarchical frameworks are examined in terms of layered structures that integrate learning-based or traditional model-based methods. We provide a detailed evaluation of the composition, strengths, limitations, and capabilities of each framework. Additionally, this survey identifies key research gaps and proposes future directions aimed at creating a more integrated and efficient framework for bipedal locomotion, with wide-ranging applications in real-world environments.
>
---
#### [replaced 055] Rapidly Converging Time-Discounted Ergodicity on Graphs for Active Inspection of Confined Spaces
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.10853v2](http://arxiv.org/pdf/2503.10853v2)**

> **作者:** Benjamin Wong; Ryan H. Lee; Tyler M. Paine; Santosh Devasia; Ashis G. Banerjee
>
> **摘要:** Ergodic exploration has spawned a lot of interest in mobile robotics due to its ability to design time trajectories that match desired spatial coverage statistics. However, current ergodic approaches are for continuous spaces, which require detailed sensory information at each point and can lead to fractal-like trajectories that cannot be tracked easily. This paper presents a new ergodic approach for graph-based discretization of continuous spaces. It also introduces a new time-discounted ergodicity metric, wherein early visitations of information-rich nodes are weighted more than late visitations. A Markov chain synthesized using a convex program is shown to converge more rapidly to time-discounted ergodicity than the traditional fastest mixing Markov chain. The resultant ergodic traversal method is used within a hierarchical framework for active inspection of confined spaces with the goal of detecting anomalies robustly using SLAM-driven Bayesian hypothesis testing. Experiments on a ground robot show the advantages of this framework over three continuous space ergodic planners as well as greedy and random exploration methods for left-behind foreign object debris detection in a ballast tank.
>
---
#### [replaced 056] HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.16757v3](http://arxiv.org/pdf/2509.16757v3)**

> **作者:** Haoyang Weng; Yitang Li; Nikhil Sobanbabu; Zihan Wang; Zhengyi Luo; Tairan He; Deva Ramanan; Guanya Shi
>
> **备注:** website: hdmi-humanoid.github.io
>
> **摘要:** Enabling robust whole-body humanoid-object interaction (HOI) remains challenging due to motion data scarcity and the contact-rich nature. We present HDMI (HumanoiD iMitation for Interaction), a simple and general framework that learns whole-body humanoid-object interaction skills directly from monocular RGB videos. Our pipeline (i) extracts and retargets human and object trajectories from unconstrained videos to build structured motion datasets, (ii) trains a reinforcement learning (RL) policy to co-track robot and object states with three key designs: a unified object representation, a residual action space, and a general interaction reward, and (iii) zero-shot deploys the RL policies on real humanoid robots. Extensive sim-to-real experiments on a Unitree G1 humanoid demonstrate the robustness and generality of our approach: HDMI achieves 67 consecutive door traversals and successfully performs 6 distinct loco-manipulation tasks in the real world and 14 tasks in simulation. Our results establish HDMI as a simple and general framework for acquiring interactive humanoid skills from human videos.
>
---
#### [replaced 057] GLIDE: A Coordinated Aerial-Ground Framework for Search and Rescue in Unknown Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.14210v2](http://arxiv.org/pdf/2509.14210v2)**

> **作者:** Seth Farrell; Chenghao Li; Hongzhan Yu; Hesam Mojtahedi; Sicun Gao; Henrik I. Christensen
>
> **摘要:** We present a cooperative aerial-ground search-and-rescue (SAR) framework that pairs two unmanned aerial vehicles (UAVs) with an unmanned ground vehicle (UGV) to achieve rapid victim localization and obstacle-aware navigation in unknown environments. We dub this framework Guided Long-horizon Integrated Drone Escort (GLIDE), highlighting the UGV's reliance on UAV guidance for long-horizon planning. In our framework, a goal-searching UAV executes real-time onboard victim detection and georeferencing to nominate goals for the ground platform, while a terrain-scouting UAV flies ahead of the UGV's planned route to provide mid-level traversability updates. The UGV fuses aerial cues with local sensing to perform time-efficient A* planning and continuous replanning as information arrives. Additionally, we present a hardware demonstration (using a GEM e6 golf cart as the UGV and two X500 UAVs) to evaluate end-to-end SAR mission performance and include simulation ablations to assess the planning stack in isolation from detection. Empirical results demonstrate that explicit role separation across UAVs, coupled with terrain scouting and guided planning, improves reach time and navigation safety in time-critical SAR missions.
>
---
#### [replaced 058] Towards agile multi-robot systems in the real world: Fast onboard tracking of active blinking markers for relative localization
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.01172v2](http://arxiv.org/pdf/2502.01172v2)**

> **作者:** Tim Felix Lakemann; Daniel Bonilla Licea; Viktor Walter; Tomáš Báča; Martin Saska
>
> **摘要:** A novel onboard tracking approach enabling vision-based relative localization and communication using Active blinking Marker Tracking (AMT) is introduced in this article. Active blinking markers on multi-robot team members improve the robustness of relative localization for aerial vehicles in tightly coupled multi-robot systems during real-world deployments, while also serving as a resilient communication system. Traditional tracking algorithms struggle with fast-moving blinking markers due to their intermittent appearance in camera frames and the complexity of associating multiple of these markers across consecutive frames. AMT addresses this by using weighted polynomial regression to predict the future appearance of active blinking markers while accounting for uncertainty in the prediction. In outdoor experiments, the AMT approach outperformed state-of-the-art methods in tracking density, accuracy, and complexity. The experimental validation of this novel tracking approach for relative localization and optical communication involved testing motion patterns motivated by our research on agile multi-robot deployment.
>
---
#### [replaced 059] TreeIRL: Safe Urban Driving with Tree Search and Inverse Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.13579v2](http://arxiv.org/pdf/2509.13579v2)**

> **作者:** Momchil S. Tomov; Sang Uk Lee; Hansford Hendrago; Jinwook Huh; Teawon Han; Forbes Howington; Rafael da Silva; Gianmarco Bernasconi; Marc Heim; Samuel Findler; Xiaonan Ji; Alexander Boule; Michael Napoli; Kuo Chen; Jesse Miller; Boaz Floor; Yunqing Hu
>
> **摘要:** We present TreeIRL, a novel planner for autonomous driving that combines Monte Carlo tree search (MCTS) and inverse reinforcement learning (IRL) to achieve state-of-the-art performance in simulation and in real-world driving. The core idea is to use MCTS to find a promising set of safe candidate trajectories and a deep IRL scoring function to select the most human-like among them. We evaluate TreeIRL against both classical and state-of-the-art planners in large-scale simulations and on 500+ miles of real-world autonomous driving in the Las Vegas metropolitan area. Test scenarios include dense urban traffic, adaptive cruise control, cut-ins, and traffic lights. TreeIRL achieves the best overall performance, striking a balance between safety, progress, comfort, and human-likeness. To our knowledge, our work is the first demonstration of MCTS-based planning on public roads and underscores the importance of evaluating planners across a diverse set of metrics and in real-world environments. TreeIRL is highly extensible and could be further improved with reinforcement learning and imitation learning, providing a framework for exploring different combinations of classical and learning-based approaches to solve the planning bottleneck in autonomous driving.
>
---
#### [replaced 060] Is FISHER All You Need in The Multi-AUV Underwater Target Tracking Task?
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2412.03959v2](http://arxiv.org/pdf/2412.03959v2)**

> **作者:** Guanwen Xie; Jingzehua Xu; Ziqi Zhang; Xiangwang Hou; Dongfang Ma; Shuai Zhang; Yong Ren; Dusit Niyato
>
> **备注:** This paper has been accepted by IEEE Transactions on Mobile Computing. Besides, Guanwen Xie and Jingzehua Xu contributed equally to this work
>
> **摘要:** It is significant to employ multiple autonomous underwater vehicles (AUVs) to execute the underwater target tracking task collaboratively. However, it's pretty challenging to meet various prerequisites utilizing traditional control methods. Therefore, we propose an effective two-stage learning from demonstrations training framework, FISHER, to highlight the adaptability of reinforcement learning (RL) methods in the multi-AUV underwater target tracking task, while addressing its limitations such as extensive requirements for environmental interactions and the challenges in designing reward functions. The first stage utilizes imitation learning (IL) to realize policy improvement and generate offline datasets. To be specific, we introduce multi-agent discriminator-actor-critic based on improvements of the generative adversarial IL algorithm and multi-agent IL optimization objective derived from the Nash equilibrium condition. Then in the second stage, we develop multi-agent independent generalized decision transformer, which analyzes the latent representation to match the future states of high-quality samples rather than reward function, attaining further enhanced policies capable of handling various scenarios. Besides, we propose a simulation to simulation demonstration generation procedure to facilitate the generation of expert demonstrations in underwater environments, which capitalizes on traditional control methods and can easily accomplish the domain transfer to obtain demonstrations. Extensive simulation experiments from multiple scenarios showcase that FISHER possesses strong stability, multi-task performance and capability of generalization.
>
---
#### [replaced 061] When Engineering Outruns Intelligence: Rethinking Instruction-Guided Navigation
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.20021v2](http://arxiv.org/pdf/2507.20021v2)**

> **作者:** Matin Aghaei; Lingfeng Zhang; Mohammad Ali Alomrani; Mahdi Biparva; Yingxue Zhang
>
> **备注:** Preprint; under peer review
>
> **摘要:** Recent ObjectNav systems credit large language models (LLMs) for sizable zero-shot gains, yet it remains unclear how much comes from language versus geometry. We revisit this question by re-evaluating an instruction-guided pipeline, InstructNav, under a detector-controlled setting and introducing two training-free variants that only alter the action value map: a geometry-only Frontier Proximity Explorer (FPE) and a lightweight Semantic-Heuristic Frontier (SHF) that polls the LLM with simple frontier votes. Across HM3D and MP3D, FPE matches or exceeds the detector-controlled instruction follower while using no API calls and running faster; SHF attains comparable accuracy with a smaller, localized language prior. These results suggest that carefully engineered frontier geometry accounts for much of the reported progress, and that language is most reliable as a light heuristic rather than an end-to-end planner.
>
---
#### [replaced 062] RISE: Robust Imitation through Stochastic Encoding
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.12243v2](http://arxiv.org/pdf/2503.12243v2)**

> **作者:** Mumuksh Tayal; Manan Tayal; Ravi Prakash
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Ensuring safety in robotic systems remains a fundamental challenge, especially when deploying offline policy-learning methods such as imitation learning in dynamic environments. Traditional behavior cloning (BC) often fails to generalize when deployed without fine-tuning as it does not account for disturbances in observations that arises in real-world, changing environments. To address this limitation, we propose RISE (Robust Imitation through Stochastic Encodings), a novel imitation-learning framework that explicitly addresses erroneous measurements of environment parameters into policy learning via a variational latent representation. Our framework encodes parameters such as obstacle state, orientation, and velocity into a smooth variational latent space to improve test time generalization. This enables an offline-trained policy to produce actions that are more robust to perceptual noise and environment uncertainty. We validate our approach on two robotic platforms, an autonomous ground vehicle and a Franka Emika Panda manipulator and demonstrate improved safety robustness while maintaining goal-reaching performance compared to baseline methods.
>
---
#### [replaced 063] Decoupling Geometry from Optimization in 2D Irregular Cutting and Packing Problems: an Open-Source Collision Detection Engine
- **分类: cs.CG; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.08341v3](http://arxiv.org/pdf/2508.08341v3)**

> **作者:** Jeroen Gardeyn; Greet Vanden Berghe; Tony Wauters
>
> **备注:** 25 pages, 16 figures
>
> **摘要:** Addressing irregular cutting and packing (C&P) optimization problems poses two distinct challenges: the geometric challenge of determining whether or not an item can be placed feasibly at a certain position, and the optimization challenge of finding a good solution according to some objective function. Until now, those tackling such problems have had to address both challenges simultaneously, requiring two distinct sets of expertise and a lot of research & development effort. One way to lower this barrier is to decouple the two challenges. In this paper we introduce a powerful collision detection engine (CDE) for 2D irregular C&P problems which assumes full responsibility for the geometric challenge. The CDE (i) allows users to focus with full confidence on their optimization challenge by abstracting geometry away and (ii) enables independent advances to propagate to all optimization algorithms built atop it. We present a set of core principles and design philosophies to model a general and adaptable CDE focused on maximizing performance, accuracy and robustness. These principles are accompanied by a concrete open-source implementation called $\texttt{jagua-rs}$. This paper together with its implementation serves as a catalyst for future advances in irregular C&P problems by providing a solid foundation which can either be used as it currently exists or be further improved upon.
>
---
#### [replaced 064] Autonomous Vehicle Controllers From End-to-End Differentiable Simulation
- **分类: cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.07965v2](http://arxiv.org/pdf/2409.07965v2)**

> **作者:** Asen Nachkov; Danda Pani Paudel; Luc Van Gool
>
> **备注:** Polished and accepted at IROS 2025
>
> **摘要:** Current methods to learn controllers for autonomous vehicles (AVs) focus on behavioural cloning. Being trained only on exact historic data, the resulting agents often generalize poorly to novel scenarios. Simulators provide the opportunity to go beyond offline datasets, but they are still treated as complicated black boxes, only used to update the global simulation state. As a result, these RL algorithms are slow, sample-inefficient, and prior-agnostic. In this work, we leverage a differentiable simulator and design an analytic policy gradients (APG) approach to training AV controllers on the large-scale Waymo Open Motion Dataset. Our proposed framework brings the differentiable simulator into an end-to-end training loop, where gradients of the environment dynamics serve as a useful prior to help the agent learn a more grounded policy. We combine this setup with a recurrent architecture that can efficiently propagate temporal information across long simulated trajectories. This APG method allows us to learn robust, accurate, and fast policies, while only requiring widely-available expert trajectories, instead of scarce expert actions. We compare to behavioural cloning and find significant improvements in performance and robustness to noise in the dynamics, as well as overall more intuitive human-like handling.
>
---
#### [replaced 065] Training Tactile Sensors to Learn Force Sensing from Each Other
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.01058v3](http://arxiv.org/pdf/2503.01058v3)**

> **作者:** Zhuo Chen; Ni Ou; Xuyang Zhang; Zhiyuan Wu; Yongqiang Zhao; Yupeng Wang; Emmanouil Spyrakos Papastavridis; Nathan Lepora; Lorenzo Jamone; Jiankang Deng; Shan Luo
>
> **摘要:** Humans achieve stable and dexterous object manipulation by coordinating grasp forces across multiple fingers and palms, facilitated by a unified tactile memory system in the somatosensory cortex. This system encodes and stores tactile experiences across skin regions, enabling the flexible reuse and transfer of touch information. Inspired by this biological capability, we present GenForce, the first framework that enables transferable force sensing across tactile sensors in robotic hands. GenForce unifies tactile signals into shared marker representations, analogous to cortical sensory encoding, allowing force prediction models trained on one sensor to be transferred to others without the need for exhaustive force data collection. We demonstrate that GenForce generalizes across both homogeneous sensors with varying configurations and heterogeneous sensors with distinct sensing modalities and material properties. This transferable force sensing is also demonstrated with high performance in robot force control including daily object grasping, slip detection and avoidance. Our results highlight a scalable paradigm for cross-sensor robotic tactile learning, offering new pathways toward adaptable and tactile memory-driven manipulation in unstructured environments.
>
---
#### [replaced 066] Robot Navigation with Entity-Based Collision Avoidance using Deep Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.14183v2](http://arxiv.org/pdf/2408.14183v2)**

> **作者:** Yury Kolomeytsev; Dmitry Golembiovsky
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Efficient navigation in dynamic environments is crucial for autonomous robots interacting with moving agents and static obstacles. We present a novel deep reinforcement learning approach that improves robot navigation and interaction with different types of agents and obstacles based on specific safety requirements. Our approach uses information about the entity types, improving collision avoidance and ensuring safer navigation. We introduce a new reward function that penalizes the robot for being close to or colliding with different entities such as adults, bicyclists, children, and static obstacles, while also encouraging the robot's progress toward the goal. We propose an optimized algorithm that significantly accelerates the training, validation, and testing phases, enabling efficient learning in complex environments. Comprehensive experiments demonstrate that our approach consistently outperforms state-of-the-art navigation and collision avoidance methods.
>
---
#### [replaced 067] Open-Vocabulary Online Semantic Mapping for SLAM
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.15043v3](http://arxiv.org/pdf/2411.15043v3)**

> **作者:** Tomas Berriel Martins; Martin R. Oswald; Javier Civera
>
> **备注:** Accepted for IEEE Robotics and Automation Letters
>
> **摘要:** This paper presents an Open-Vocabulary Online 3D semantic mapping pipeline, that we denote by its acronym OVO. Given a sequence of posed RGB-D frames, we detect and track 3D segments, which we describe using CLIP vectors. These are computed from the viewpoints where they are observed by a novel CLIP merging method. Notably, our OVO has a significantly lower computational and memory footprint than offline baselines, while also showing better segmentation metrics than offline and online ones. Along with superior segmentation performance, we also show experimental results of our mapping contributions integrated with two different full SLAM backbones (Gaussian-SLAM and ORB-SLAM2), being the first ones using a neural network to merge CLIP descriptors and demonstrating end-to-end open-vocabulary online 3D mapping with loop closure.
>
---
#### [replaced 068] Communication-Efficient Desire Alignment for Embodied Agent-Human Adaptation
- **分类: cs.RO; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2505.22503v2](http://arxiv.org/pdf/2505.22503v2)**

> **作者:** Yuanfei Wang; Xinju Huang; Fangwei Zhong; Yaodong Yang; Yizhou Wang; Yuanpei Chen; Hao Dong
>
> **摘要:** While embodied agents have made significant progress in performing complex physical tasks, real-world applications demand more than pure task execution. The agents must collaborate with unfamiliar agents and human users, whose goals are often vague and implicit. In such settings, interpreting ambiguous instructions and uncovering underlying desires is essential for effective assistance. Therefore, fast and accurate desire alignment becomes a critical capability for embodied agents. In this work, we first develop a home assistance simulation environment HA-Desire that integrates an LLM-driven proxy human user exhibiting realistic value-driven goal selection and communication. The ego agent must interact with this proxy user to infer and adapt to the user's latent desires. To achieve this, we present a novel framework FAMER for fast desire alignment, which introduces a desire-based mental reasoning mechanism to identify user intent and filter desire-irrelevant actions. We further design a reflection-based communication module that reduces redundant inquiries, and incorporate goal-relevant information extraction with memory persistence to improve information reuse and reduce unnecessary exploration. Extensive experiments demonstrate that our framework significantly enhances both task execution and communication efficiency, enabling embodied agents to quickly adapt to user-specific desires in complex embodied environments.
>
---
#### [replaced 069] Learning to Segment for Vehicle Routing Problems
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.01037v2](http://arxiv.org/pdf/2507.01037v2)**

> **作者:** Wenbin Ouyang; Sirui Li; Yining Ma; Cathy Wu
>
> **摘要:** Iterative heuristics are widely recognized as state-of-the-art for Vehicle Routing Problems (VRPs). In this work, we exploit a critical observation: a large portion of the solution remains stable, i.e., unchanged across search iterations, causing redundant computations, especially for large-scale VRPs with long subtours. To address this, we pioneer the formal study of the First-Segment-Then-Aggregate (FSTA) decomposition technique to accelerate iterative solvers. FSTA preserves stable solution segments during the search, aggregates nodes within each segment into fixed hypernodes, and focuses the search only on unstable portions. Yet, a key challenge lies in identifying which segments should be aggregated. To this end, we introduce Learning-to-Segment (L2Seg), a novel neural framework to intelligently differentiate potentially stable and unstable portions for FSTA decomposition. We present three L2Seg variants: non-autoregressive (globally comprehensive but locally indiscriminate), autoregressive (locally refined but globally deficient), and their synergy. Empirical results on CVRP and VRPTW show that L2Seg accelerates state-of-the-art solvers by 2x to 7x. We further provide in-depth analysis showing why synergy achieves the best performance. Notably, L2Seg is compatible with traditional, learning-based, and hybrid solvers, while supporting various VRPs.
>
---
#### [replaced 070] InterKey: Cross-modal Intersection Keypoints for Global Localization on OpenStreetMap
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.13857v2](http://arxiv.org/pdf/2509.13857v2)**

> **作者:** Nguyen Hoang Khoi Tran; Julie Stephany Berrio; Mao Shan; Stewart Worrall
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Reliable global localization is critical for autonomous vehicles, especially in environments where GNSS is degraded or unavailable, such as urban canyons and tunnels. Although high-definition (HD) maps provide accurate priors, the cost of data collection, map construction, and maintenance limits scalability. OpenStreetMap (OSM) offers a free and globally available alternative, but its coarse abstraction poses challenges for matching with sensor data. We propose InterKey, a cross-modal framework that leverages road intersections as distinctive landmarks for global localization. Our method constructs compact binary descriptors by jointly encoding road and building imprints from point clouds and OSM. To bridge modality gaps, we introduce discrepancy mitigation, orientation determination, and area-equalized sampling strategies, enabling robust cross-modal matching. Experiments on the KITTI dataset demonstrate that InterKey achieves state-of-the-art accuracy, outperforming recent baselines by a large margin. The framework generalizes to sensors that can produce dense structural point clouds, offering a scalable and cost-effective solution for robust vehicle localization.
>
---
