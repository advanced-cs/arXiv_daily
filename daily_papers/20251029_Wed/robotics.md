# 机器人 cs.RO

- **最新发布 44 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] Multi-Agent Scenario Generation in Roundabouts with a Transformer-enhanced Conditional Variational Autoencoder
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对智能驾驶测试中多智能体交通场景生成难题，提出基于Transformer增强的条件变分自编码器（CVAE-T），用于生成复杂环岛场景下的真实、多样合成数据。模型能准确重建与生成场景，实现对交互行为的有效评估，并揭示潜在变量的可解释性，助力智能驾驶功能验证与数据增强。**

- **链接: [http://arxiv.org/pdf/2510.24671v1](http://arxiv.org/pdf/2510.24671v1)**

> **作者:** Li Li; Tobias Brinkmann; Till Temmen; Markus Eisenbarth; Jakob Andert
>
> **摘要:** With the increasing integration of intelligent driving functions into serial-produced vehicles, ensuring their functionality and robustness poses greater challenges. Compared to traditional road testing, scenario-based virtual testing offers significant advantages in terms of time and cost efficiency, reproducibility, and exploration of edge cases. We propose a Transformer-enhanced Conditional Variational Autoencoder (CVAE-T) model for generating multi-agent traffic scenarios in roundabouts, which are characterized by high vehicle dynamics and complex layouts, yet remain relatively underexplored in current research. The results show that the proposed model can accurately reconstruct original scenarios and generate realistic, diverse synthetic scenarios. Besides, two Key-Performance-Indicators (KPIs) are employed to evaluate the interactive behavior in the generated scenarios. Analysis of the latent space reveals partial disentanglement, with several latent dimensions exhibiting distinct and interpretable effects on scenario attributes such as vehicle entry timing, exit timing, and velocity profiles. The results demonstrate the model's capability to generate scenarios for the validation of intelligent driving functions involving multi-agent interactions, as well as to augment data for their development and iterative improvement.
>
---
#### [new 002] Manipulate as Human: Learning Task-oriented Manipulation Skills by Adversarial Motion Priors
- **分类: cs.RO**

- **简介: 该论文聚焦于机器人任务导向操作技能学习，旨在让机器人像人一样自然地操纵工具。针对现有方法难以模拟人类动作的问题，提出基于对抗性运动先验的HMAMP框架，结合真实与仿真数据训练策略，生成符合人类运动统计特性的轨迹。在锤击任务中验证了其有效性，并成功应用于真实机械臂。**

- **链接: [http://arxiv.org/pdf/2510.24257v1](http://arxiv.org/pdf/2510.24257v1)**

> **作者:** Ziqi Ma; Changda Tian; Yue Gao
>
> **摘要:** In recent years, there has been growing interest in developing robots and autonomous systems that can interact with human in a more natural and intuitive way. One of the key challenges in achieving this goal is to enable these systems to manipulate objects and tools in a manner that is similar to that of humans. In this paper, we propose a novel approach for learning human-style manipulation skills by using adversarial motion priors, which we name HMAMP. The approach leverages adversarial networks to model the complex dynamics of tool and object manipulation, as well as the aim of the manipulation task. The discriminator is trained using a combination of real-world data and simulation data executed by the agent, which is designed to train a policy that generates realistic motion trajectories that match the statistical properties of human motion. We evaluated HMAMP on one challenging manipulation task: hammering, and the results indicate that HMAMP is capable of learning human-style manipulation skills that outperform current baseline methods. Additionally, we demonstrate that HMAMP has potential for real-world applications by performing real robot arm hammering tasks. In general, HMAMP represents a significant step towards developing robots and autonomous systems that can interact with humans in a more natural and intuitive way, by learning to manipulate tools and objects in a manner similar to how humans do.
>
---
#### [new 003] A Comprehensive General Model of Tendon-Actuated Concentric Tube Robots with Multiple Tubes and Tendons
- **分类: cs.RO**

- **简介: 该论文针对腱驱动同轴管机器人的建模难题，提出基于Cosserat杆的通用力学模型，可处理多管多腱配置。模型支持管体扭转与伸缩，共享弯曲中心线，实现高精度形状预测（误差<4%）。通过实验验证了模型在不同构型下的有效性与通用性，为机器人精确控制提供理论基础。**

- **链接: [http://arxiv.org/pdf/2510.23954v1](http://arxiv.org/pdf/2510.23954v1)**

> **作者:** Pejman Kheradmand; Behnam Moradkhani; Raghavasimhan Sankaranarayanan; Kent K. Yamamoto; Tanner J. Zachem; Patrick J. Codd; Yash Chitalia; Pierre E. Dupont
>
> **摘要:** Tendon-actuated concentric tube mechanisms combine the advantages of tendon-driven continuum robots and concentric tube robots while addressing their respective limitations. They overcome the restricted degrees of freedom often seen in tendon-driven designs, and mitigate issues such as snapping instability associated with concentric tube robots. However, a complete and general mechanical model for these systems remains an open problem. In this work, we propose a Cosserat rod-based framework for modeling the general case of $n$ concentric tubes, each actuated by $m_i$ tendons, where $i = \{1, \ldots, n\}$. The model allows each tube to twist and elongate while enforcing a shared centerline for bending. We validate the proposed framework through experiments with two-tube and three tube assemblies under various tendon routing configurations, achieving tip prediction errors $<4\%$ of the robot's total length. We further demonstrate the model's generality by applying it to existing robots in the field, where maximum tip deviations remain around $5\%$ of the total length. This model provides a foundation for accurate shape estimation and control of advanced tendon-actuated concentric tube robots.
>
---
#### [new 004] VOCALoco: Viability-Optimized Cost-aware Adaptive Locomotion
- **分类: cs.RO; I.2.9**

- **简介: 该论文提出VOCALoco框架，解决腿式机器人在复杂地形中安全与能效平衡问题。通过模块化技能选择，动态评估预训练策略的安全性与能耗，实现对楼梯等场景的自适应高效行走，显著提升真实机器人在攀爬任务中的鲁棒性与安全性。**

- **链接: [http://arxiv.org/pdf/2510.23997v1](http://arxiv.org/pdf/2510.23997v1)**

> **作者:** Stanley Wu; Mohamad H. Danesh; Simon Li; Hanna Yurchyk; Amin Abyaneh; Anas El Houssaini; David Meger; Hsiu-Chin Lin
>
> **备注:** Accepted in IEEE Robotics and Automation Letters (RAL), 2025. 8 pages, 9 figures
>
> **摘要:** Recent advancements in legged robot locomotion have facilitated traversal over increasingly complex terrains. Despite this progress, many existing approaches rely on end-to-end deep reinforcement learning (DRL), which poses limitations in terms of safety and interpretability, especially when generalizing to novel terrains. To overcome these challenges, we introduce VOCALoco, a modular skill-selection framework that dynamically adapts locomotion strategies based on perceptual input. Given a set of pre-trained locomotion policies, VOCALoco evaluates their viability and energy-consumption by predicting both the safety of execution and the anticipated cost of transport over a fixed planning horizon. This joint assessment enables the selection of policies that are both safe and energy-efficient, given the observed local terrain. We evaluate our approach on staircase locomotion tasks, demonstrating its performance in both simulated and real-world scenarios using a quadrupedal robot. Empirical results show that VOCALoco achieves improved robustness and safety during stair ascent and descent compared to a conventional end-to-end DRL policy
>
---
#### [new 005] Feature Matching-Based Gait Phase Prediction for Obstacle Crossing Control of Powered Transfemoral Prosthesis
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对截肢者使用动力假肢跨越障碍的难题，提出基于惯性传感器与遗传算法优化神经网络的步态相位预测方法，实现对大腿和膝关节角度的精准预测，有效提升假肢在复杂地形中的适应能力。**

- **链接: [http://arxiv.org/pdf/2510.24676v1](http://arxiv.org/pdf/2510.24676v1)**

> **作者:** Jiaxuan Zhang; Yuquan Leng; Yixuan Guo; Chenglong Fu
>
> **备注:** 6 pages, conference
>
> **摘要:** For amputees with powered transfemoral prosthetics, navigating obstacles or complex terrain remains challenging. This study addresses this issue by using an inertial sensor on the sound ankle to guide obstacle-crossing movements. A genetic algorithm computes the optimal neural network structure to predict the required angles of the thigh and knee joints. A gait progression prediction algorithm determines the actuation angle index for the prosthetic knee motor, ultimately defining the necessary thigh and knee angles and gait progression. Results show that when the standard deviation of Gaussian noise added to the thigh angle data is less than 1, the method can effectively eliminate noise interference, achieving 100\% accuracy in gait phase estimation under 150 Hz, with thigh angle prediction error being 8.71\% and knee angle prediction error being 6.78\%. These findings demonstrate the method's ability to accurately predict gait progression and joint angles, offering significant practical value for obstacle negotiation in powered transfemoral prosthetics.
>
---
#### [new 006] Improved Accuracy of Robot Localization Using 3-D LiDAR in a Hippocampus-Inspired Model
- **分类: cs.RO; cs.AI; cs.LG; cs.SY; eess.SY; q-bio.NC; I.2.9; I.2.6**

- **简介: 该论文属于机器人定位任务，旨在解决二维环境模型在三维复杂空间中因水平对称性导致的空间混淆问题。通过引入垂直角度敏感性，构建基于3D LiDAR的海马体启发式边界向量细胞模型，提升机器人在真实三维环境中的定位精度与地图构建能力。**

- **链接: [http://arxiv.org/pdf/2510.24029v1](http://arxiv.org/pdf/2510.24029v1)**

> **作者:** Andrew Gerstenslager; Bekarys Dukenbaev; Ali A. Minai
>
> **备注:** 8 pages, 9 figures, Presented at the 2025 International Joint Conference on Neural Networks, Rome, July 2025
>
> **摘要:** Boundary Vector Cells (BVCs) are a class of neurons in the brains of vertebrates that encode environmental boundaries at specific distances and allocentric directions, playing a central role in forming place fields in the hippocampus. Most computational BVC models are restricted to two-dimensional (2D) environments, making them prone to spatial ambiguities in the presence of horizontal symmetries in the environment. To address this limitation, we incorporate vertical angular sensitivity into the BVC framework, thereby enabling robust boundary detection in three dimensions, and leading to significantly more accurate spatial localization in a biologically-inspired robot model. The proposed model processes LiDAR data to capture vertical contours, thereby disambiguating locations that would be indistinguishable under a purely 2D representation. Experimental results show that in environments with minimal vertical variation, the proposed 3D model matches the performance of a 2D baseline; yet, as 3D complexity increases, it yields substantially more distinct place fields and markedly reduces spatial aliasing. These findings show that adding a vertical dimension to BVC-based localization can significantly enhance navigation and mapping in real-world 3D spaces while retaining performance parity in simpler, near-planar scenarios.
>
---
#### [new 007] Flatness-based trajectory planning for 3D overhead cranes with friction compensation and collision avoidance
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对3D桥式起重机的轨迹规划任务，解决含摩擦与防碰撞约束下的快速安全运动问题。基于微分平坦性理论，构建考虑非线性摩擦和碰撞避免的最优轨迹生成方法，实现仅末端限幅的激进运动，仿真验证了摩擦建模对避免执行器饱和与碰撞的关键作用。**

- **链接: [http://arxiv.org/pdf/2510.24457v1](http://arxiv.org/pdf/2510.24457v1)**

> **作者:** Jorge Vicente-Martinez; Edgar Ramirez-Laboreo
>
> **备注:** 8 pages, 11 figures
>
> **摘要:** This paper presents an optimal trajectory generation method for 3D overhead cranes by leveraging differential flatness. This framework enables the direct inclusion of complex physical and dynamic constraints, such as nonlinear friction and collision avoidance for both payload and rope. Our approach allows for aggressive movements by constraining payload swing only at the final point. A comparative simulation study validates our approach, demonstrating that neglecting dry friction leads to actuator saturation and collisions. The results show that friction modeling is a fundamental requirement for fast and safe crane trajectories.
>
---
#### [new 008] NVSim: Novel View Synthesis Simulator for Large Scale Indoor Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出NVSim，用于从图像序列自动生成大规模可导航室内模拟环境。针对传统3D扫描成本高、扩展性差的问题，采用改进的3D高斯泼溅技术，结合地面感知与无网格可通行性分析，构建清晰的可行走平面与拓扑导航图，实现高效、真实的室内场景重建与导航模拟。**

- **链接: [http://arxiv.org/pdf/2510.24335v1](http://arxiv.org/pdf/2510.24335v1)**

> **作者:** Mingyu Jeong; Eunsung Kim; Sehun Park; Andrew Jaeyong Choi
>
> **备注:** 9 pages, 10 figures
>
> **摘要:** We present NVSim, a framework that automatically constructs large-scale, navigable indoor simulators from only common image sequences, overcoming the cost and scalability limitations of traditional 3D scanning. Our approach adapts 3D Gaussian Splatting to address visual artifacts on sparsely observed floors a common issue in robotic traversal data. We introduce Floor-Aware Gaussian Splatting to ensure a clean, navigable ground plane, and a novel mesh-free traversability checking algorithm that constructs a topological graph by directly analyzing rendered views. We demonstrate our system's ability to generate valid, large-scale navigation graphs from real-world data. A video demonstration is avilable at https://youtu.be/tTiIQt6nXC8
>
---
#### [new 009] Balanced Collaborative Exploration via Distributed Topological Graph Voronoi Partition
- **分类: cs.RO**

- **简介: 该论文针对多机器人协同探索任务，解决动态非凸环境下探索区域划分与任务分配不均衡问题。提出一种分布式拓扑图Voronoi划分方法，构建增量更新的拓扑地图，实现全局覆盖引导下的均衡分区与高效路径规划，保障探索效率、完整性和负载平衡。**

- **链接: [http://arxiv.org/pdf/2510.24067v1](http://arxiv.org/pdf/2510.24067v1)**

> **作者:** Tianyi Ding; Ronghao Zheng; Senlin Zhang; Meiqin Liu
>
> **摘要:** This work addresses the collaborative multi-robot autonomous online exploration problem, particularly focusing on distributed exploration planning for dynamically balanced exploration area partition and task allocation among a team of mobile robots operating in obstacle-dense non-convex environments. We present a novel topological map structure that simultaneously characterizes both spatial connectivity and global exploration completeness of the environment. The topological map is updated incrementally to utilize known spatial information for updating reachable spaces, while exploration targets are planned in a receding horizon fashion under global coverage guidance. A distributed weighted topological graph Voronoi algorithm is introduced implementing balanced graph space partitions of the fused topological maps. Theoretical guarantees are provided for distributed consensus convergence and equitable graph space partitions with constant bounds. A local planner optimizes the visitation sequence of exploration targets within the balanced partitioned graph space to minimize travel distance, while generating safe, smooth, and dynamically feasible motion trajectories. Comprehensive benchmarking against state-of-the-art methods demonstrates significant improvements in exploration efficiency, completeness, and workload balance across the robot team.
>
---
#### [new 010] Adaptive-twist Soft Finger Mechanism for Grasping by Wrapping
- **分类: cs.RO**

- **简介: 该论文针对柔性机械手在密集物体中抓取难题，提出一种可自适应扭转的软指机构。通过单驱动实现平面内外的自适应变形与变刚度控制，支持包裹式抓取。结合有限元分析优化设计，实验验证了其对多种物体的有效抓取能力。**

- **链接: [http://arxiv.org/pdf/2510.23963v1](http://arxiv.org/pdf/2510.23963v1)**

> **作者:** Hiroki Ishikawa; Kyosuke Ishibashi; Ko Yamamoto
>
> **摘要:** This paper presents a soft robot finger capable of adaptive-twist deformation to grasp objects by wrapping them. For a soft hand to grasp and pick-up one object from densely contained multiple objects, a soft finger requires the adaptive-twist deformation function in both in-plane and out-of-plane directions. The function allows the finger to be inserted deeply into a limited gap among objects. Once inserted, the soft finger requires appropriate control of grasping force normal to contact surface, thereby maintaining the twisted deformation. In this paper, we refer to this type of grasping as grasping by wrapping. To achieve these two functions by a single actuation source, we propose a variable stiffness mechanism that can adaptively change the stiffness as the pressure is higher. We conduct a finite element analysis (FEA) on the proposed mechanism and determine its design parameter based on the FEA result. Using the developed soft finger, we report basic experimental results and demonstrations on grasping various objects.
>
---
#### [new 011] GroundLoc: Efficient Large-Scale Outdoor LiDAR-Only Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出GroundLoc，一种基于LiDAR的室外大场景定位方法。针对高精度、高效定位难题，利用BEV投影与R2D2/SIFT关键点匹配实现地图注册，支持多传感器，地图仅需4MB/km²，实现在多个数据集上低于50cm的轨迹误差，满足实时性要求。**

- **链接: [http://arxiv.org/pdf/2510.24623v1](http://arxiv.org/pdf/2510.24623v1)**

> **作者:** Nicolai Steinke; Daniel Goehring
>
> **摘要:** In this letter, we introduce GroundLoc, a LiDAR-only localization pipeline designed to localize a mobile robot in large-scale outdoor environments using prior maps. GroundLoc employs a Bird's-Eye View (BEV) image projection focusing on the perceived ground area and utilizes the place recognition network R2D2, or alternatively, the non-learning approach Scale-Invariant Feature Transform (SIFT), to identify and select keypoints for BEV image map registration. Our results demonstrate that GroundLoc outperforms state-of-the-art methods on the SemanticKITTI and HeLiPR datasets across various sensors. In the multi-session localization evaluation, GroundLoc reaches an Average Trajectory Error (ATE) well below 50 cm on all Ouster OS2 128 sequences while meeting online runtime requirements. The system supports various sensor models, as evidenced by evaluations conducted with Velodyne HDL-64E, Ouster OS2 128, Aeva Aeries II, and Livox Avia sensors. The prior maps are stored as 2D raster image maps, which can be created from a single drive and require only 4 MB of storage per square kilometer. The source code is available at https://github.com/dcmlr/groundloc.
>
---
#### [new 012] Adaptive Keyframe Selection for Scalable 3D Scene Reconstruction in Dynamic Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对动态环境中3D场景重建的实时性与精度问题，提出一种自适应关键帧选择方法。通过结合基于误差的筛选与动态阈值调整机制，优化数据流压缩，提升重建质量。在Spann3r和CUT3R上验证，显著优于传统静态策略。**

- **链接: [http://arxiv.org/pdf/2510.23928v1](http://arxiv.org/pdf/2510.23928v1)**

> **作者:** Raman Jha; Yang Zhou; Giuseppe Loianno
>
> **备注:** Under Review for ROBOVIS 2026
>
> **摘要:** In this paper, we propose an adaptive keyframe selection method for improved 3D scene reconstruction in dynamic environments. The proposed method integrates two complementary modules: an error-based selection module utilizing photometric and structural similarity (SSIM) errors, and a momentum-based update module that dynamically adjusts keyframe selection thresholds according to scene motion dynamics. By dynamically curating the most informative frames, our approach addresses a key data bottleneck in real-time perception. This allows for the creation of high-quality 3D world representations from a compressed data stream, a critical step towards scalable robot learning and deployment in complex, dynamic environments. Experimental results demonstrate significant improvements over traditional static keyframe selection strategies, such as fixed temporal intervals or uniform frame skipping. These findings highlight a meaningful advancement toward adaptive perception systems that can dynamically respond to complex and evolving visual scenes. We evaluate our proposed adaptive keyframe selection module on two recent state-of-the-art 3D reconstruction networks, Spann3r and CUT3R, and observe consistent improvements in reconstruction quality across both frameworks. Furthermore, an extensive ablation study confirms the effectiveness of each individual component in our method, underlining their contribution to the overall performance gains.
>
---
#### [new 013] Fare: Failure Resilience in Learned Visual Navigation Control
- **分类: cs.RO**

- **简介: 该论文针对视觉导航中模仿学习策略在分布外场景易失效的问题，提出Fare框架，实现失败检测与自动恢复。通过隐式识别失败原因并结合恢复启发式策略，提升策略鲁棒性，支持复杂环境中长距离可靠导航。**

- **链接: [http://arxiv.org/pdf/2510.24680v1](http://arxiv.org/pdf/2510.24680v1)**

> **作者:** Zishuo Wang; Joel Loo; David Hsu
>
> **摘要:** While imitation learning (IL) enables effective visual navigation, IL policies are prone to unpredictable failures in out-of-distribution (OOD) scenarios. We advance the notion of failure-resilient policies, which not only detect failures but also recover from them automatically. Failure recognition that identifies the factors causing failure is key to informing recovery: e.g. pinpointing image regions triggering failure detections can provide cues to guide recovery. We present Fare, a framework to construct failure-resilient IL policies, embedding OOD-detection and recognition in them without using explicit failure data, and pairing them with recovery heuristics. Real-world experiments show that Fare enables failure recovery across two different policy architectures, enabling robust long-range navigation in complex environments.
>
---
#### [new 014] A Survey on Collaborative SLAM with 3D Gaussian Splatting
- **分类: cs.RO**

- **简介: 该论文聚焦多机器人协同SLAM任务，针对3D高斯溅射在多机系统中面临的全局一致性、通信效率与异构数据融合难题，系统梳理了集中式与分布式架构下的关键技术，包括姿态优化、语义蒸馏与实时扩展，并总结了评估数据集与指标，指出了长期建图、语义关联等未来方向。**

- **链接: [http://arxiv.org/pdf/2510.23988v1](http://arxiv.org/pdf/2510.23988v1)**

> **作者:** Phuc Nguyen Xuan; Thanh Nguyen Canh; Huu-Hung Nguyen; Nak Young Chong; Xiem HoangVan
>
> **摘要:** This survey comprehensively reviews the evolving field of multi-robot collaborative Simultaneous Localization and Mapping (SLAM) using 3D Gaussian Splatting (3DGS). As an explicit scene representation, 3DGS has enabled unprecedented real-time, high-fidelity render- ing, ideal for robotics. However, its use in multi-robot systems introduces significant challenges in maintaining global consistency, managing communication, and fusing data from heterogeneous sources. We systematically categorize approaches by their architecture-centralized, distributed- and analyze core components like multi-agent consistency and alignment, communication- efficient, Gaussian representation, semantic distillation, fusion and pose optimization, and real- time scalability. In addition, a summary of critical datasets and evaluation metrics is provided to contextualize performance. Finally, we identify key open challenges and chart future research directions, including lifelong mapping, semantic association and mapping, multi-model for robustness, and bridging the Sim2Real gap.
>
---
#### [new 015] Motivating Students' Self-study with Goal Reminder and Emotional Support
- **分类: cs.RO**

- **简介: 该论文研究社交机器人在大学生自主学习中的作用，旨在提升学习专注度与成效。通过对照实验，验证了目标提醒与情感支持对学习体验的积极影响，发现功能与情感双重支持能增强学生使用意愿与目标达成率，揭示了社交机器人在自学习场景中的潜在价值。**

- **链接: [http://arxiv.org/pdf/2510.23860v1](http://arxiv.org/pdf/2510.23860v1)**

> **作者:** Hyung Chan Cho; Go-Eum Cha; Yanfu Liu; Sooyeon Jeong
>
> **备注:** RO-MAN 2025 accepted paper
>
> **摘要:** While the efficacy of social robots in supporting people in learning tasks has been extensively investigated, their potential impact in assisting students in self-studying contexts has not been investigated much. This study explores how a social robot can act as a peer study companion for college students during self-study tasks by delivering task-oriented goal reminder and positive emotional support. We conducted an exploratory Wizard-of-Oz study to explore how these robotic support behaviors impacted students' perceived focus, productivity, and engagement in comparison to a robot that only provided physical presence (control). Our study results suggest that participants in the goal reminder and the emotional support conditions reported greater ease of use, with the goal reminder condition additionally showing a higher willingness to use the robot in future study sessions. Participants' satisfaction with the robot was correlated with their perception of the robot as a social other, and this perception was found to be a predictor for their level of goal achievement in the self-study task. These findings highlight the potential of socially assistive robots to support self-study through both functional and emotional engagement.
>
---
#### [new 016] Towards Quadrupedal Jumping and Walking for Dynamic Locomotion using Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文研究四足机器人动态运动中的跳跃与行走任务，旨在解决跳跃精度与训练效率问题。通过基于课程的强化学习框架，设计垂直与水平跳跃策略，利用抛体运动优化奖励稀疏性，结合参考状态初始化加速探索，并实现高精度跳跃（最高1.25m水平、1.0m垂直）及复杂地形行走，有效跨越仿真到现实的差距。**

- **链接: [http://arxiv.org/pdf/2510.24584v1](http://arxiv.org/pdf/2510.24584v1)**

> **作者:** Jørgen Anker Olsen; Lars Rønhaug Pettersen; Kostas Alexis
>
> **备注:** 8 pages
>
> **摘要:** This paper presents a curriculum-based reinforcement learning framework for training precise and high-performance jumping policies for the robot `Olympus'. Separate policies are developed for vertical and horizontal jumps, leveraging a simple yet effective strategy. First, we densify the inherently sparse jumping reward using the laws of projectile motion. Next, a reference state initialization scheme is employed to accelerate the exploration of dynamic jumping behaviors without reliance on reference trajectories. We also present a walking policy that, when combined with the jumping policies, unlocks versatile and dynamic locomotion capabilities. Comprehensive testing validates walking on varied terrain surfaces and jumping performance that exceeds previous works, effectively crossing the Sim2Real gap. Experimental validation demonstrates horizontal jumps up to 1.25 m with centimeter accuracy and vertical jumps up to 1.0 m. Additionally, we show that with only minor modifications, the proposed method can be used to learn omnidirectional jumping.
>
---
#### [new 017] An Adaptive Inspection Planning Approach Towards Routine Monitoring in Uncertain Environments
- **分类: cs.RO**

- **简介: 该论文针对不确定环境下机器人巡检任务，提出分层自适应规划框架。通过历史地图生成全局路径，并基于实时环境信息局部重规划，以应对地形变化与障碍物。工作聚焦于提升巡检自主性与鲁棒性，已在真实地下矿井中验证。**

- **链接: [http://arxiv.org/pdf/2510.24554v1](http://arxiv.org/pdf/2510.24554v1)**

> **作者:** Vignesh Kottayam Viswanathan; Yifan Bai; Scott Fredriksson; Sumeet Satpute; Christoforos Kanellakis; George Nikolakopoulos
>
> **备注:** Submitted for ICRA 2026
>
> **摘要:** In this work, we present a hierarchical framework designed to support robotic inspection under environment uncertainty. By leveraging a known environment model, existing methods plan and safely track inspection routes to visit points of interest. However, discrepancies between the model and actual site conditions, caused by either natural or human activities, can alter the surface morphology or introduce path obstructions. To address this challenge, the proposed framework divides the inspection task into: (a) generating the initial global view-plan for region of interests based on a historical map and (b) local view replanning to adapt to the current morphology of the inspection scene. The proposed hierarchy preserves global coverage objectives while enabling reactive adaptation to the local surface morphology. This enables the local autonomy to remain robust against environment uncertainty and complete the inspection tasks. We validate the approach through deployments in real-world subterranean mines using quadrupedal robot.
>
---
#### [new 018] Supervisory Measurement-Guided Noise Covariance Estimation
- **分类: cs.RO**

- **简介: 该论文针对状态估计中传感器噪声协方差难以准确识别的问题，提出一种监督测量引导的协方差估计算法。通过贝叶斯分解构建双层优化框架，利用扩展卡尔曼滤波与梯度并行计算，高效协同优化协方差参数，提升估计精度与效率。**

- **链接: [http://arxiv.org/pdf/2510.24508v1](http://arxiv.org/pdf/2510.24508v1)**

> **作者:** Haoying Li; Yifan Peng; Junfeng Wu
>
> **摘要:** Reliable state estimation hinges on accurate specification of sensor noise covariances, which weigh heterogeneous measurements. In practice, these covariances are difficult to identify due to environmental variability, front-end preprocessing, and other reasons. We address this by formulating noise covariance estimation as a bilevel optimization that, from a Bayesian perspective, factorizes the joint likelihood of so-called odometry and supervisory measurements, thereby balancing information utilization with computational efficiency. The factorization converts the nested Bayesian dependency into a chain structure, enabling efficient parallel computation: at the lower level, an invariant extended Kalman filter with state augmentation estimates trajectories, while a derivative filter computes analytical gradients in parallel for upper-level gradient updates. The upper level refines the covariance to guide the lower-level estimation. Experiments on synthetic and real-world datasets show that our method achieves higher efficiency over existing baselines.
>
---
#### [new 019] Stochastic Prize-Collecting Games: Strategic Planning in Multi-Robot Systems
- **分类: cs.RO**

- **简介: 该论文研究多机器人系统中的竞争性路径规划问题，针对奖励稀缺环境下的自利机器人，提出随机奖赏收集博弈（SPCG）模型。通过理论分析与算法设计（ORS、FORL），实现高效策略学习，使机器人在能量约束和随机转移下达成近似最优解，显著提升大规模团队与不均衡奖励场景下的可扩展性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.24515v1](http://arxiv.org/pdf/2510.24515v1)**

> **作者:** Malintha Fernando; Petter Ögren; Silun Zhang
>
> **备注:** Submitted to IEEE Robotics and Automation Letters
>
> **摘要:** The Team Orienteering Problem (TOP) generalizes many real-world multi-robot scheduling and routing tasks that occur in autonomous mobility, aerial logistics, and surveillance applications. While many flavors of the TOP exist for planning in multi-robot systems, they assume that all the robots cooperate toward a single objective; thus, they do not extend to settings where the robots compete in reward-scarce environments. We propose Stochastic Prize-Collecting Games (SPCG) as an extension of the TOP to plan in the presence of self-interested robots operating on a graph, under energy constraints and stochastic transitions. A theoretical study on complete and star graphs establishes that there is a unique pure Nash equilibrium in SPCGs that coincides with the optimal routing solution of an equivalent TOP given a rank-based conflict resolution rule. This work proposes two algorithms: Ordinal Rank Search (ORS) to obtain the ''ordinal rank'' --one's effective rank in temporarily-formed local neighborhoods during the games' stages, and Fictitious Ordinal Response Learning (FORL) to obtain best-response policies against one's senior-rank opponents. Empirical evaluations conducted on road networks and synthetic graphs under both dynamic and stationary prize distributions show that 1) the state-aliasing induced by OR-conditioning enables learning policies that scale more efficiently to large team sizes than those trained with the global index, and 2) Policies trained with FORL generalize better to imbalanced prize distributions than those with other multi-agent training methods. Finally, the learned policies in the SPCG achieved between 87% and 95% optimality compared to an equivalent TOP solution obtained by mixed-integer linear programming.
>
---
#### [new 020] DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出DynaRend，一种用于机器人操作的3D动态表征学习框架。针对真实数据稀缺导致策略泛化难的问题，通过掩码未来渲染与可微体素渲染，联合学习几何、语义与动态信息。在多视图RGB-D视频上预训练，提升策略成功率与环境扰动下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.24261v1](http://arxiv.org/pdf/2510.24261v1)**

> **作者:** Jingyi Tian; Le Wang; Sanping Zhou; Sen Wang; Jiayi Li; Gang Hua
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Learning generalizable robotic manipulation policies remains a key challenge due to the scarcity of diverse real-world training data. While recent approaches have attempted to mitigate this through self-supervised representation learning, most either rely on 2D vision pretraining paradigms such as masked image modeling, which primarily focus on static semantics or scene geometry, or utilize large-scale video prediction models that emphasize 2D dynamics, thus failing to jointly learn the geometry, semantics, and dynamics required for effective manipulation. In this paper, we present DynaRend, a representation learning framework that learns 3D-aware and dynamics-informed triplane features via masked reconstruction and future prediction using differentiable volumetric rendering. By pretraining on multi-view RGB-D video data, DynaRend jointly captures spatial geometry, future dynamics, and task semantics in a unified triplane representation. The learned representations can be effectively transferred to downstream robotic manipulation tasks via action value map prediction. We evaluate DynaRend on two challenging benchmarks, RLBench and Colosseum, as well as in real-world robotic experiments, demonstrating substantial improvements in policy success rate, generalization to environmental perturbations, and real-world applicability across diverse manipulation tasks.
>
---
#### [new 021] SynAD: Enhancing Real-World End-to-End Autonomous Driving Models through Synthetic Data Integration
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对端到端自动驾驶模型因依赖真实数据导致场景多样性不足的问题，提出SynAD框架。通过在多智能体合成场景中指定信息最全的代理为自车，利用地图投影与新设计的Map-to-BEV网络生成无传感器输入的鸟瞰特征，并融合真实与合成数据进行训练，显著提升模型安全性能。**

- **链接: [http://arxiv.org/pdf/2510.24052v1](http://arxiv.org/pdf/2510.24052v1)**

> **作者:** Jongsuk Kim; Jaeyoung Lee; Gyojin Han; Dongjae Lee; Minki Jeong; Junmo Kim
>
> **摘要:** Recent advancements in deep learning and the availability of high-quality real-world driving datasets have propelled end-to-end autonomous driving. Despite this progress, relying solely on real-world data limits the variety of driving scenarios for training. Synthetic scenario generation has emerged as a promising solution to enrich the diversity of training data; however, its application within E2E AD models remains largely unexplored. This is primarily due to the absence of a designated ego vehicle and the associated sensor inputs, such as camera or LiDAR, typically provided in real-world scenarios. To address this gap, we introduce SynAD, the first framework designed to enhance real-world E2E AD models using synthetic data. Our method designates the agent with the most comprehensive driving information as the ego vehicle in a multi-agent synthetic scenario. We further project path-level scenarios onto maps and employ a newly developed Map-to-BEV Network to derive bird's-eye-view features without relying on sensor inputs. Finally, we devise a training strategy that effectively integrates these map-based synthetic data with real driving data. Experimental results demonstrate that SynAD effectively integrates all components and notably enhances safety performance. By bridging synthetic scenario generation and E2E AD, SynAD paves the way for more comprehensive and robust autonomous driving models.
>
---
#### [new 022] LagMemo: Language 3D Gaussian Splatting Memory for Multi-modal Open-vocabulary Multi-goal Visual Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出LagMemo系统，解决多模态、开放词汇、多目标视觉导航问题。通过构建语言3D高斯点云记忆，实现跨模态目标定位与动态验证，支持复杂场景下的灵活导航。基于新构建的GOAT-Core数据集，实验证明其显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24118v1](http://arxiv.org/pdf/2510.24118v1)**

> **作者:** Haotian Zhou; Xiaole Wang; He Li; Fusheng Sun; Shengyu Guo; Guolei Qi; Jianghuan Xu; Huijing Zhao
>
> **摘要:** Navigating to a designated goal using visual information is a fundamental capability for intelligent robots. Most classical visual navigation methods are restricted to single-goal, single-modality, and closed set goal settings. To address the practical demands of multi-modal, open-vocabulary goal queries and multi-goal visual navigation, we propose LagMemo, a navigation system that leverages a language 3D Gaussian Splatting memory. During exploration, LagMemo constructs a unified 3D language memory. With incoming task goals, the system queries the memory, predicts candidate goal locations, and integrates a local perception-based verification mechanism to dynamically match and validate goals during navigation. For fair and rigorous evaluation, we curate GOAT-Core, a high-quality core split distilled from GOAT-Bench tailored to multi-modal open-vocabulary multi-goal visual navigation. Experimental results show that LagMemo's memory module enables effective multi-modal open-vocabulary goal localization, and that LagMemo outperforms state-of-the-art methods in multi-goal visual navigation. Project page: https://weekgoodday.github.io/lagmemo
>
---
#### [new 023] A Framework for the Systematic Evaluation of Obstacle Avoidance and Object-Aware Controllers
- **分类: cs.RO**

- **简介: 该论文针对机器人动态避障任务，提出一种系统评估对象感知控制器的框架。聚焦运动学、运动轨迹与虚拟约束三方面，通过基础实验验证行为可靠性，对比三种控制器，发现现有方法在运动连续性与稳定性上存在不足，为未来避障算法的设计与评测提供依据。**

- **链接: [http://arxiv.org/pdf/2510.24683v1](http://arxiv.org/pdf/2510.24683v1)**

> **作者:** Caleb Escobedo; Nataliya Nechyporenko; Shreyas Kadekodi; Alessandro Roncone
>
> **摘要:** Real-time control is an essential aspect of safe robot operation in the real world with dynamic objects. We present a framework for the analysis of object-aware controllers, methods for altering a robot's motion to anticipate and avoid possible collisions. This framework is focused on three design considerations: kinematics, motion profiles, and virtual constraints. Additionally, the analysis in this work relies on verification of robot behaviors using fundamental robot-obstacle experimental scenarios. To showcase the effectiveness of our method we compare three representative object-aware controllers. The comparison uses metrics originating from the design considerations. From the analysis, we find that the design of object-aware controllers often lacks kinematic considerations, continuity of control points, and stability in movement profiles. We conclude that this framework can be used in the future to design, compare, and benchmark obstacle avoidance methods.
>
---
#### [new 024] Dynamically-Consistent Trajectory Optimization for Legged Robots via Contact Point Decomposition
- **分类: cs.RO**

- **简介: 该论文针对足式机器人运动规划任务，解决轨迹优化中动力学一致性与摩擦约束满足问题。通过接触点分解与贝塞尔多项式分析，实现平动动力学与摩擦锥约束的精确建模，支持多种步态生成，提升运动可靠性。**

- **链接: [http://arxiv.org/pdf/2510.24069v1](http://arxiv.org/pdf/2510.24069v1)**

> **作者:** Sangmin Kim; Hajun Kim; Gijeong Kim; Min-Gyu Kim; Hae-Won Park
>
> **备注:** 8 pages, 4 figures, IEEE ROBOTICS AND AUTOMATION LETTERS. PREPRINT VERSION. ACCEPTED OCTOBER, 2025
>
> **摘要:** To generate reliable motion for legged robots through trajectory optimization, it is crucial to simultaneously compute the robot's path and contact sequence, as well as accurately consider the dynamics in the problem formulation. In this paper, we present a phase-based trajectory optimization that ensures the feasibility of translational dynamics and friction cone constraints throughout the entire trajectory. Specifically, our approach leverages the superposition properties of linear differential equations to decouple the translational dynamics for each contact point, which operates under different phase sequences. Furthermore, we utilize the differentiation matrix of B{\'e}zier polynomials to derive an analytical relationship between the robot's position and force, thereby ensuring the consistent satisfaction of translational dynamics. Additionally, by exploiting the convex closure property of B{\'e}zier polynomials, our method ensures compliance with friction cone constraints. Using the aforementioned approach, the proposed trajectory optimization framework can generate dynamically reliable motions with various gait sequences for legged robots. We validate our framework using a quadruped robot model, focusing on the feasibility of dynamics and motion generation.
>
---
#### [new 025] Global-State-Free Obstacle Avoidance for Quadrotor Control in Air-Ground Cooperation
- **分类: cs.RO**

- **简介: 该论文针对无人机与地面机器人协同任务中的障碍物避让问题，提出无需全局状态估计的CoNi-OA算法。利用单帧激光雷达数据生成速度调制矩阵，在非惯性框架下实时生成安全轨迹，实现高效、低延迟的避障，适用于动态与未知环境。**

- **链接: [http://arxiv.org/pdf/2510.24315v1](http://arxiv.org/pdf/2510.24315v1)**

> **作者:** Baozhe Zhang; Xinwei Chen; Qingcheng Chen; Chao Xu; Fei Gao; Yanjun Cao
>
> **摘要:** CoNi-MPC provides an efficient framework for UAV control in air-ground cooperative tasks by relying exclusively on relative states, eliminating the need for global state estimation. However, its lack of environmental information poses significant challenges for obstacle avoidance. To address this issue, we propose a novel obstacle avoidance algorithm, Cooperative Non-inertial frame-based Obstacle Avoidance (CoNi-OA), designed explicitly for UAV-UGV cooperative scenarios without reliance on global state estimation or obstacle prediction. CoNi-OA uniquely utilizes a single frame of raw LiDAR data from the UAV to generate a modulation matrix, which directly adjusts the quadrotor's velocity to achieve obstacle avoidance. This modulation-based method enables real-time generation of collision-free trajectories within the UGV's non-inertial frame, significantly reducing computational demands (less than 5 ms per iteration) while maintaining safety in dynamic and unpredictable environments. The key contributions of this work include: (1) a modulation-based obstacle avoidance algorithm specifically tailored for UAV-UGV cooperation in non-inertial frames without global states; (2) rapid, real-time trajectory generation based solely on single-frame LiDAR data, removing the need for obstacle modeling or prediction; and (3) adaptability to both static and dynamic environments, thus extending applicability to featureless or unknown scenarios.
>
---
#### [new 026] Stand, Walk, Navigate: Recovery-Aware Visual Navigation on a Low-Cost Wheeled Quadruped
- **分类: cs.RO**

- **简介: 该论文针对低成本轮足机器人在复杂地形中自主导航与跌倒恢复难题，提出一种基于视觉-惯性感知和深度强化学习的恢复感知导航系统。通过低代价传感器与智能控制策略，实现高效移动、自主避障及多场景下可靠跌倒恢复，推动预算受限平台的智能化部署。**

- **链接: [http://arxiv.org/pdf/2510.23902v1](http://arxiv.org/pdf/2510.23902v1)**

> **作者:** Jans Solano; Diego Quiroz
>
> **备注:** Accepted at the IROS 2025 Workshop on Wheeled-Legged Robots
>
> **摘要:** Wheeled-legged robots combine the efficiency of wheels with the obstacle negotiation of legs, yet many state-of-the-art systems rely on costly actuators and sensors, and fall-recovery is seldom integrated, especially for wheeled-legged morphologies. This work presents a recovery-aware visual-inertial navigation system on a low-cost wheeled quadruped. The proposed system leverages vision-based perception from a depth camera and deep reinforcement learning policies for robust locomotion and autonomous recovery from falls across diverse terrains. Simulation experiments show agile mobility with low-torque actuators over irregular terrain and reliably recover from external perturbations and self-induced failures. We further show goal directed navigation in structured indoor spaces with low-cost perception. Overall, this approach lowers the barrier to deploying autonomous navigation and robust locomotion policies in budget-constrained robotic platforms.
>
---
#### [new 027] RoboOmni: Proactive Robot Manipulation in Omni-modal Context
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出RoboOmni框架，面向多模态情境下的主动机器人操作任务。针对现实交互中用户不常下达明确指令的问题，通过融合语音、视觉与环境声音，实现意图的主动推断与协作。构建了包含140k场景的OmniAction数据集，推动了无需显式命令的机器人智能发展。**

- **链接: [http://arxiv.org/pdf/2510.23763v1](http://arxiv.org/pdf/2510.23763v1)**

> **作者:** Siyin Wang; Jinlan Fu; Feihong Liu; Xinzhe He; Huangxuan Wu; Junhao Shi; Kexin Huang; Zhaoye Fei; Jingjing Gong; Zuxuan Wu; Yugang Jiang; See-Kiong Ng; Tat-Seng Chua; Xipeng Qiu
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have driven rapid progress in Vision-Language-Action (VLA) models for robotic manipulation. Although effective in many scenarios, current approaches largely rely on explicit instructions, whereas in real-world interactions, humans rarely issue instructions directly. Effective collaboration requires robots to infer user intentions proactively. In this work, we introduce cross-modal contextual instructions, a new setting where intent is derived from spoken dialogue, environmental sounds, and visual cues rather than explicit commands. To address this new setting, we present RoboOmni, a Perceiver-Thinker-Talker-Executor framework based on end-to-end omni-modal LLMs that unifies intention recognition, interaction confirmation, and action execution. RoboOmni fuses auditory and visual signals spatiotemporally for robust intention recognition, while supporting direct speech interaction. To address the absence of training data for proactive intention recognition in robotic manipulation, we build OmniAction, comprising 140k episodes, 5k+ speakers, 2.4k event sounds, 640 backgrounds, and six contextual instruction types. Experiments in simulation and real-world settings show that RoboOmni surpasses text- and ASR-based baselines in success rate, inference speed, intention recognition, and proactive assistance.
>
---
#### [new 028] ZTRS: Zero-Imitation End-to-end Autonomous Driving with Trajectory Scoring
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出ZTRS框架，解决端到端自动驾驶中依赖模仿学习导致的性能瓶颈问题。通过纯强化学习结合轨迹评分机制，直接从高维传感器数据学习驾驶策略，无需专家示范，实现零模仿、全端到端的鲁棒规划，在多个基准上达到领先性能。**

- **链接: [http://arxiv.org/pdf/2510.24108v1](http://arxiv.org/pdf/2510.24108v1)**

> **作者:** Zhenxin Li; Wenhao Yao; Zi Wang; Xinglong Sun; Jingde Chen; Nadine Chang; Maying Shen; Jingyu Song; Zuxuan Wu; Shiyi Lan; Jose M. Alvarez
>
> **摘要:** End-to-end autonomous driving maps raw sensor inputs directly into ego-vehicle trajectories to avoid cascading errors from perception modules and to leverage rich semantic cues. Existing frameworks largely rely on Imitation Learning (IL), which can be limited by sub-optimal expert demonstrations and covariate shift during deployment. On the other hand, Reinforcement Learning (RL) has recently shown potential in scaling up with simulations, but is typically confined to low-dimensional symbolic inputs (e.g. 3D objects and maps), falling short of full end-to-end learning from raw sensor data. We introduce ZTRS (Zero-Imitation End-to-End Autonomous Driving with Trajectory Scoring), a framework that combines the strengths of both worlds: sensor inputs without losing information and RL training for robust planning. To the best of our knowledge, ZTRS is the first framework that eliminates IL entirely by only learning from rewards while operating directly on high-dimensional sensor data. ZTRS utilizes offline reinforcement learning with our proposed Exhaustive Policy Optimization (EPO), a variant of policy gradient tailored for enumerable actions and rewards. ZTRS demonstrates strong performance across three benchmarks: Navtest (generic real-world open-loop planning), Navhard (open-loop planning in challenging real-world and synthetic scenarios), and HUGSIM (simulated closed-loop driving). Specifically, ZTRS achieves the state-of-the-art result on Navhard and outperforms IL-based baselines on HUGSIM. Code will be available at https://github.com/woxihuanjiangguo/ZTRS.
>
---
#### [new 029] Spatiotemporal Calibration of Doppler Velocity Logs for Underwater Robots
- **分类: cs.RO**

- **简介: 该论文针对水下机器人中多传感器（如DVL）的时空标定问题，提出统一迭代校准（UIC）框架，联合估计外参与时钟偏移。通过高斯过程运动先验实现精确运动插值，结合最大后验估计与可证明一致的初始化策略，支持多种传感器协同标定，验证了其在仿真与实测中的有效性。**

- **链接: [http://arxiv.org/pdf/2510.24571v1](http://arxiv.org/pdf/2510.24571v1)**

> **作者:** Hongxu Zhao; Guangyang Zeng; Yunling Shao; Tengfei Zhang; Junfeng Wu
>
> **摘要:** The calibration of extrinsic parameters and clock offsets between sensors for high-accuracy performance in underwater SLAM systems remains insufficiently explored. Existing methods for Doppler Velocity Log (DVL) calibration are either constrained to specific sensor configurations or rely on oversimplified assumptions, and none jointly estimate translational extrinsics and time offsets. We propose a Unified Iterative Calibration (UIC) framework for general DVL sensor setups, formulated as a Maximum A Posteriori (MAP) estimation with a Gaussian Process (GP) motion prior for high-fidelity motion interpolation. UIC alternates between efficient GP-based motion state updates and gradient-based calibration variable updates, supported by a provably statistically consistent sequential initialization scheme. The proposed UIC can be applied to IMU, cameras and other modalities as co-sensors. We release an open-source DVL-camera calibration toolbox. Beyond underwater applications, several aspects of UIC-such as the integration of GP priors for MAP-based calibration and the design of provably reliable initialization procedures-are broadly applicable to other multi-sensor calibration problems. Finally, simulations and real-world tests validate our approach.
>
---
#### [new 030] PFEA: An LLM-based High-Level Natural Language Planning and Feedback Embodied Agent for Human-Centered AI
- **分类: cs.RO**

- **简介: 该论文提出PFEA框架，面向人本AI的机器人智能体，解决复杂自然语言指令下机器人在线规划与执行难题。通过融合视觉语言模型，实现语音交互、任务规划、指令转换与反馈评估，显著提升任务成功率。**

- **链接: [http://arxiv.org/pdf/2510.24109v1](http://arxiv.org/pdf/2510.24109v1)**

> **作者:** Wenbin Ding; Jun Chen; Mingjia Chen; Fei Xie; Qi Mao; Philip Dames
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has marked a significant breakthrough in Artificial Intelligence (AI), ushering in a new era of Human-centered Artificial Intelligence (HAI). HAI aims to better serve human welfare and needs, thereby placing higher demands on the intelligence level of robots, particularly in aspects such as natural language interaction, complex task planning, and execution. Intelligent agents powered by LLMs have opened up new pathways for realizing HAI. However, existing LLM-based embodied agents often lack the ability to plan and execute complex natural language control tasks online. This paper explores the implementation of intelligent robotic manipulating agents based on Vision-Language Models (VLMs) in the physical world. We propose a novel embodied agent framework for robots, which comprises a human-robot voice interaction module, a vision-language agent module and an action execution module. The vision-language agent itself includes a vision-based task planner, a natural language instruction converter, and a task performance feedback evaluator. Experimental results demonstrate that our agent achieves a 28\% higher average task success rate in both simulated and real environments compared to approaches relying solely on LLM+CLIP, significantly improving the execution success rate of high-level natural language instruction tasks.
>
---
#### [new 031] GeVI-SLAM: Gravity-Enhanced Stereo Visua Inertial SLAM for Underwater Robots
- **分类: cs.RO**

- **简介: 该论文提出GeVI-SLAM，一种用于水下机器人的增强重力立体视觉惯性SLAM系统。针对视觉退化与惯性信号激励不足问题，利用立体相机深度估计消除尺度不确定性，通过重力初始化解耦姿态并采用4-DOF PnP求解，提升精度与稳定性，实现高效鲁棒的定位与建图。**

- **链接: [http://arxiv.org/pdf/2510.24533v1](http://arxiv.org/pdf/2510.24533v1)**

> **作者:** Yuan Shen; Yuze Hong; Guangyang Zeng; Tengfei Zhang; Pui Yi Chui; Ziyang Hong; Junfeng Wu
>
> **摘要:** Accurate visual inertial simultaneous localization and mapping (VI SLAM) for underwater robots remains a significant challenge due to frequent visual degeneracy and insufficient inertial measurement unit (IMU) motion excitation. In this paper, we present GeVI-SLAM, a gravity-enhanced stereo VI SLAM system designed to address these issues. By leveraging the stereo camera's direct depth estimation ability, we eliminate the need to estimate scale during IMU initialization, enabling stable operation even under low acceleration dynamics. With precise gravity initialization, we decouple the pitch and roll from the pose estimation and solve a 4 degrees of freedom (DOF) Perspective-n-Point (PnP) problem for pose tracking. This allows the use of a minimal 3-point solver, which significantly reduces computational time to reject outliers within a Random Sample Consensus framework. We further propose a bias-eliminated 4-DOF PnP estimator with provable consistency, ensuring the relative pose converges to the true value as the feature number increases. To handle dynamic motion, we refine the full 6-DOF pose while jointly estimating the IMU covariance, enabling adaptive weighting of the gravity prior. Extensive experiments on simulated and real-world data demonstrate that GeVI-SLAM achieves higher accuracy and greater stability compared to state-of-the-art methods.
>
---
#### [new 032] Language-Conditioned Representations and Mixture-of-Experts Policy for Robust Multi-Task Robotic Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对多任务机器人操作中的感知模糊与任务冲突问题，提出结合语言条件视觉表征（LCVR）与语言条件专家混合密度策略（LMoE-DP）的框架。通过语言引导视觉特征与专家分工机制，提升任务区分与动作决策能力，在真实机器人上实现79%平均成功率，显著优于基线。**

- **链接: [http://arxiv.org/pdf/2510.24055v1](http://arxiv.org/pdf/2510.24055v1)**

> **作者:** Xiucheng Zhang; Yang Jiang; Hongwei Qing; Jiashuo Bai
>
> **备注:** 8 pages
>
> **摘要:** Perceptual ambiguity and task conflict limit multitask robotic manipulation via imitation learning. We propose a framework combining a Language-Conditioned Visual Representation (LCVR) module and a Language-conditioned Mixture-ofExperts Density Policy (LMoE-DP). LCVR resolves perceptual ambiguities by grounding visual features with language instructions, enabling differentiation between visually similar tasks. To mitigate task conflict, LMoE-DP uses a sparse expert architecture to specialize in distinct, multimodal action distributions, stabilized by gradient modulation. On real-robot benchmarks, LCVR boosts Action Chunking with Transformers (ACT) and Diffusion Policy (DP) success rates by 33.75% and 25%, respectively. The full framework achieves a 79% average success, outperforming the advanced baseline by 21%. Our work shows that combining semantic grounding and expert specialization enables robust, efficient multi-task manipulation
>
---
#### [new 033] Embodying Physical Computing into Soft Robots
- **分类: cs.RO**

- **简介: 该论文探讨将物理计算融入软体机器人，解决其智能化与鲁棒性难题。提出通过模拟振荡器、物理储层计算和物理算法计算三种策略，实现无需传统电子芯片的复杂行为控制，如避障、分类与逻辑编程，推动软体机器人向更智能、自适应方向发展。**

- **链接: [http://arxiv.org/pdf/2510.24692v1](http://arxiv.org/pdf/2510.24692v1)**

> **作者:** Jun Wang; Ziyang Zhou; Ardalan Kahak; Suyi Li
>
> **摘要:** Softening and onboarding computers and controllers is one of the final frontiers in soft robotics towards their robustness and intelligence for everyday use. In this regard, embodying soft and physical computing presents exciting potential. Physical computing seeks to encode inputs into a mechanical computing kernel and leverage the internal interactions among this kernel's constituent elements to compute the output. Moreover, such input-to-output evolution can be re-programmable. This perspective paper proposes a framework for embodying physical computing into soft robots and discusses three unique strategies in the literature: analog oscillators, physical reservoir computing, and physical algorithmic computing. These embodied computers enable the soft robot to perform complex behaviors that would otherwise require CMOS-based electronics -- including coordinated locomotion with obstacle avoidance, payload weight and orientation classification, and programmable operation based on logical rules. This paper will detail the working principles of these embodied physical computing methods, survey the current state-of-the-art, and present a perspective for future development.
>
---
#### [new 034] Blindfolded Experts Generalize Better: Insights from Robotic Manipulation and Videogames
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究行为克隆中的泛化问题，提出让专家“盲视”部分任务信息以促进探索。通过实验证明，克隆盲视专家在机器人插销和视频游戏任务中泛化能力更强，理论分析也证实其误差随信息量减小而降低。**

- **链接: [http://arxiv.org/pdf/2510.24194v1](http://arxiv.org/pdf/2510.24194v1)**

> **作者:** Ev Zisselman; Mirco Mutti; Shelly Francis-Meretzki; Elisei Shafer; Aviv Tamar
>
> **摘要:** Behavioral cloning is a simple yet effective technique for learning sequential decision-making from demonstrations. Recently, it has gained prominence as the core of foundation models for the physical world, where achieving generalization requires countless demonstrations of a multitude of tasks. Typically, a human expert with full information on the task demonstrates a (nearly) optimal behavior. In this paper, we propose to hide some of the task's information from the demonstrator. This ``blindfolded'' expert is compelled to employ non-trivial exploration to solve the task. We show that cloning the blindfolded expert generalizes better to unseen tasks than its fully-informed counterpart. We conduct experiments of real-world robot peg insertion tasks with (limited) human demonstrations, alongside videogames from the Procgen benchmark. Additionally, we support our findings with theoretical analysis, which confirms that the generalization error scales with $\sqrt{I/m}$, where $I$ measures the amount of task information available to the demonstrator, and $m$ is the number of demonstrated tasks. Both theory and practice indicate that cloning blindfolded experts generalizes better with fewer demonstrated tasks. Project page with videos and code: https://sites.google.com/view/blindfoldedexperts/home
>
---
#### [new 035] A Hybrid Approach for Visual Multi-Object Tracking
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种混合视觉多目标跟踪方法，融合随机与确定性机制，解决未知且动态变化目标数下的身份一致性问题。通过粒子滤波结合粒子群优化处理非线性动态与噪声，利用时空一致性和外观相似性提升跟踪精度；通过确定性关联与轨迹平滑更新策略，增强弱跟踪鲁棒性，适用于视频与实时流。**

- **链接: [http://arxiv.org/pdf/2510.24410v1](http://arxiv.org/pdf/2510.24410v1)**

> **作者:** Toan Van Nguyen; Rasmus G. K. Christiansen; Dirk Kraft; Leon Bodenhagen
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper proposes a visual multi-object tracking method that jointly employs stochastic and deterministic mechanisms to ensure identifier consistency for unknown and time-varying target numbers under nonlinear dynamics. A stochastic particle filter addresses nonlinear dynamics and non-Gaussian noise, with support from particle swarm optimization (PSO) to guide particles toward state distribution modes and mitigate divergence through proposed fitness measures incorporating motion consistency, appearance similarity, and social-interaction cues with neighboring targets. Deterministic association further enforces identifier consistency via a proposed cost matrix incorporating spatial consistency between particles and current detections, detection confidences, and track penalties. Subsequently, a novel scheme is proposed for the smooth updating of target states while preserving their identities, particularly for weak tracks during interactions with other targets and prolonged occlusions. Moreover, velocity regression over past states provides trend-seed velocities, enhancing particle sampling and state updates. The proposed tracker is designed to operate flexibly for both pre-recorded videos and camera live streams, where future frames are unavailable. Experimental results confirm superior performance compared to state-of-the-art trackers. The source-code reference implementations of both the proposed method and compared-trackers are provided on GitHub: https://github.com/SDU-VelKoTek/GenTrack2
>
---
#### [new 036] GenTrack: A New Generation of Multi-Object Tracking
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GenTrack，一种新型多目标跟踪方法，旨在解决目标数量未知、动态变化及遮挡下的身份一致性问题。通过融合随机与确定性追踪，引入粒子群优化与社会交互建模，提升弱检测条件下的跟踪鲁棒性，并构建了包含三种变体的开源基准实现，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24399v1](http://arxiv.org/pdf/2510.24399v1)**

> **作者:** Toan Van Nguyen; Rasmus G. K. Christiansen; Dirk Kraft; Leon Bodenhagen
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper introduces a novel multi-object tracking (MOT) method, dubbed GenTrack, whose main contributions include: a hybrid tracking approach employing both stochastic and deterministic manners to robustly handle unknown and time-varying numbers of targets, particularly in maintaining target identity (ID) consistency and managing nonlinear dynamics, leveraging particle swarm optimization (PSO) with some proposed fitness measures to guide stochastic particles toward their target distribution modes, enabling effective tracking even with weak and noisy object detectors, integration of social interactions among targets to enhance PSO-guided particles as well as improve continuous updates of both strong (matched) and weak (unmatched) tracks, thereby reducing ID switches and track loss, especially during occlusions, a GenTrack-based redefined visual MOT baseline incorporating a comprehensive state and observation model based on space consistency, appearance, detection confidence, track penalties, and social scores for systematic and efficient target updates, and the first-ever publicly available source-code reference implementation with minimal dependencies, featuring three variants, including GenTrack Basic, PSO, and PSO-Social, facilitating flexible reimplementation. Experimental results have shown that GenTrack provides superior performance on standard benchmarks and real-world scenarios compared to state-of-the-art trackers, with integrated implementations of baselines for fair comparison. Potential directions for future work are also discussed. The source-code reference implementations of both the proposed method and compared-trackers are provided on GitHub: https://github.com/SDU-VelKoTek/GenTrack
>
---
#### [new 037] Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks
- **分类: cs.AI; cs.RO**

- **简介: 该论文针对脉冲神经网络（SNN）在强化学习中的训练难题，提出自适应代理梯度与引导策略结合的方法。解决非可微性与序列训练短时困境，显著提升无人机控制性能，实现在真实场景中平均收益达400点，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24461v1](http://arxiv.org/pdf/2510.24461v1)**

> **作者:** Korneel Van den Berghe; Stein Stroobants; Vijay Janapa Reddi; G. C. H. E. de Croon
>
> **摘要:** Neuromorphic computing systems are set to revolutionize energy-constrained robotics by achieving orders-of-magnitude efficiency gains, while enabling native temporal processing. Spiking Neural Networks (SNNs) represent a promising algorithmic approach for these systems, yet their application to complex control tasks faces two critical challenges: (1) the non-differentiable nature of spiking neurons necessitates surrogate gradients with unclear optimization properties, and (2) the stateful dynamics of SNNs require training on sequences, which in reinforcement learning (RL) is hindered by limited sequence lengths during early training, preventing the network from bridging its warm-up period. We address these challenges by systematically analyzing surrogate gradient slope settings, showing that shallower slopes increase gradient magnitude in deeper layers but reduce alignment with true gradients. In supervised learning, we find no clear preference for fixed or scheduled slopes. The effect is much more pronounced in RL settings, where shallower slopes or scheduled slopes lead to a 2.1x improvement in both training and final deployed performance. Next, we propose a novel training approach that leverages a privileged guiding policy to bootstrap the learning process, while still exploiting online environment interactions with the spiking policy. Combining our method with an adaptive slope schedule for a real-world drone position control task, we achieve an average return of 400 points, substantially outperforming prior techniques, including Behavioral Cloning and TD3BC, which achieve at most --200 points under the same conditions. This work advances both the theoretical understanding of surrogate gradient learning in SNNs and practical training methodologies for neuromorphic controllers demonstrated in real-world robotic systems.
>
---
#### [new 038] Learning Parameterized Skills from Demonstrations
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出DEPS算法，用于从专家示范中学习可参数化的技能。针对传统方法在潜变量模型中易出现退化的问题，结合时序变分推断与信息理论正则化，联合学习技能策略与元策略，实现对时序延续、语义明确且可适应的参数化技能的发现。实验表明，该方法在多任务场景下显著提升泛化能力，并能提取可解释技能（如抓取位置可调的抓取技能）。**

- **链接: [http://arxiv.org/pdf/2510.24095v1](http://arxiv.org/pdf/2510.24095v1)**

> **作者:** Vedant Gupta; Haotian Fu; Calvin Luo; Yiding Jiang; George Konidaris
>
> **备注:** Neurips 2025
>
> **摘要:** We present DEPS, an end-to-end algorithm for discovering parameterized skills from expert demonstrations. Our method learns parameterized skill policies jointly with a meta-policy that selects the appropriate discrete skill and continuous parameters at each timestep. Using a combination of temporal variational inference and information-theoretic regularization methods, we address the challenge of degeneracy common in latent variable models, ensuring that the learned skills are temporally extended, semantically meaningful, and adaptable. We empirically show that learning parameterized skills from multitask expert demonstrations significantly improves generalization to unseen tasks. Our method outperforms multitask as well as skill learning baselines on both LIBERO and MetaWorld benchmarks. We also demonstrate that DEPS discovers interpretable parameterized skills, such as an object grasping skill whose continuous arguments define the grasp location.
>
---
#### [new 039] Sample-efficient and Scalable Exploration in Continuous-Time RL
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文研究连续时间强化学习问题，针对真实系统连续动态与传统离散时间算法不匹配的挑战，提出COMBRL算法。利用高斯过程等概率模型建模未知微分方程，通过平衡奖励与不确定性实现高效探索。证明了其在奖励驱动和无监督设置下的理论性能，并在实验中展示其样本效率与可扩展性优于基线方法。**

- **链接: [http://arxiv.org/pdf/2510.24482v1](http://arxiv.org/pdf/2510.24482v1)**

> **作者:** Klemens Iten; Lenart Treven; Bhavya Sukhija; Florian Dörfler; Andreas Krause
>
> **备注:** 26 pages, 6 figures, 6 tables
>
> **摘要:** Reinforcement learning algorithms are typically designed for discrete-time dynamics, even though the underlying real-world control systems are often continuous in time. In this paper, we study the problem of continuous-time reinforcement learning, where the unknown system dynamics are represented using nonlinear ordinary differential equations (ODEs). We leverage probabilistic models, such as Gaussian processes and Bayesian neural networks, to learn an uncertainty-aware model of the underlying ODE. Our algorithm, COMBRL, greedily maximizes a weighted sum of the extrinsic reward and model epistemic uncertainty. This yields a scalable and sample-efficient approach to continuous-time model-based RL. We show that COMBRL achieves sublinear regret in the reward-driven setting, and in the unsupervised RL setting (i.e., without extrinsic rewards), we provide a sample complexity bound. In our experiments, we evaluate COMBRL in both standard and unsupervised RL settings and demonstrate that it scales better, is more sample-efficient than prior methods, and outperforms baselines across several deep RL tasks.
>
---
#### [new 040] Modeling and Scheduling of Fusion Patterns in Autonomous Driving Systems (Extended Version)
- **分类: eess.SY; cs.OS; cs.RO; cs.SY**

- **简介: 该论文针对自动驾驶系统中数据融合任务的调度问题，提出一种基于ILP的框架，建模三种融合模式（定时触发、等待全部、即时融合），优化反应时间、信息年龄等实时性能指标，生成可直接部署的确定性调度方案。**

- **链接: [http://arxiv.org/pdf/2510.23895v1](http://arxiv.org/pdf/2510.23895v1)**

> **作者:** Hoora Sobhani; Hyoseung Kim
>
> **摘要:** In Autonomous Driving Systems (ADS), Directed Acyclic Graphs (DAGs) are widely used to model complex data dependencies and inter-task communication. However, existing DAG scheduling approaches oversimplify data fusion tasks by assuming fixed triggering mechanisms, failing to capture the diverse fusion patterns found in real-world ADS software stacks. In this paper, we propose a systematic framework for analyzing various fusion patterns and their performance implications in ADS. Our framework models three distinct fusion task types: timer-triggered, wait-for-all, and immediate fusion, which comprehensively represent real-world fusion behaviors. Our Integer Linear Programming (ILP)-based approach enables an optimization of multiple real-time performance metrics, including reaction time, time disparity, age of information, and response time, while generating deterministic offline schedules directly applicable to real platforms. Evaluation using real-world ADS case studies, Raspberry Pi implementation, and randomly generated DAGs demonstrates that our framework handles diverse fusion patterns beyond the scope of existing work, and achieves substantial performance improvements in comparable scenarios.
>
---
#### [new 041] Can LLMs Translate Human Instructions into a Reinforcement Learning Agent's Internal Emergent Symbolic Representation?
- **分类: cs.CL; cs.RO**

- **简介: 该论文研究大语言模型（LLMs）将人类自然语言指令翻译为强化学习智能体内部涌现符号表征的能力。针对符号表征与语言之间的对齐问题，作者在蚁迷宫和蚁坠落环境中评估GPT、Claude等模型的翻译性能，发现其效果受划分粒度和任务复杂度影响显著，揭示了当前LLMs在表示对齐上的局限性。**

- **链接: [http://arxiv.org/pdf/2510.24259v1](http://arxiv.org/pdf/2510.24259v1)**

> **作者:** Ziqi Ma; Sao Mai Nguyen; Philippe Xu
>
> **摘要:** Emergent symbolic representations are critical for enabling developmental learning agents to plan and generalize across tasks. In this work, we investigate whether large language models (LLMs) can translate human natural language instructions into the internal symbolic representations that emerge during hierarchical reinforcement learning. We apply a structured evaluation framework to measure the translation performance of commonly seen LLMs -- GPT, Claude, Deepseek and Grok -- across different internal symbolic partitions generated by a hierarchical reinforcement learning algorithm in the Ant Maze and Ant Fall environments. Our findings reveal that although LLMs demonstrate some ability to translate natural language into a symbolic representation of the environment dynamics, their performance is highly sensitive to partition granularity and task complexity. The results expose limitations in current LLMs capacity for representation alignment, highlighting the need for further research on robust alignment between language and internal agent representations.
>
---
#### [new 042] BLM$_1$: A Boundless Large Model for Cross-Space, Cross-Task, and Cross-Embodiment Learning
- **分类: cs.AI; cs.MM; cs.RO**

- **简介: 该论文提出BLM₁，一种跨空间、跨任务、跨具身性的通用大模型。针对现有模型在数字与物理世界间迁移能力弱、具身推理不足的问题，通过两阶段训练整合具身知识与高层语义控制，实现统一建模。在多机器人、多任务场景下显著优于现有模型。**

- **链接: [http://arxiv.org/pdf/2510.24161v1](http://arxiv.org/pdf/2510.24161v1)**

> **作者:** Wentao Tan; Bowen Wang; Heng Zhi; Chenyu Liu; Zhe Li; Jian Liu; Zengrong Lin; Yukun Dai; Yipeng Chen; Wenjie Yang; Enci Xie; Hao Xue; Baixu Ji; Chen Xu; Zhibin Wang; Tianshi Wang; Lei Zhu; Heng Tao Shen
>
> **摘要:** Multimodal large language models (MLLMs) have advanced vision-language reasoning and are increasingly deployed in embodied agents. However, significant limitations remain: MLLMs generalize poorly across digital-physical spaces and embodiments; vision-language-action models (VLAs) produce low-level actions yet lack robust high-level embodied reasoning; and most embodied large language models (ELLMs) are constrained to digital-space with poor generalization to the physical world. Thus, unified models that operate seamlessly across digital and physical spaces while generalizing across embodiments and tasks remain absent. We introduce the \textbf{Boundless Large Model (BLM$_1$)}, a multimodal spatial foundation model that preserves instruction following and reasoning, incorporates embodied knowledge, and supports robust cross-embodiment control. BLM$_1$ integrates three key capabilities -- \textit{cross-space transfer, cross-task learning, and cross-embodiment generalization} -- via a two-stage training paradigm. Stage I injects embodied knowledge into the MLLM through curated digital corpora while maintaining language competence. Stage II trains a policy module through an intent-bridging interface that extracts high-level semantics from the MLLM to guide control, without fine-tuning the MLLM backbone. This process is supported by a self-collected cross-embodiment demonstration suite spanning four robot embodiments and six progressively challenging tasks. Evaluations across digital and physical benchmarks show that a single BLM$_1$ instance outperforms four model families -- MLLMs, ELLMs, VLAs, and GMLMs -- achieving $\sim\!\textbf{6%}$ gains in digital tasks and $\sim\!\textbf{3%}$ in physical tasks.
>
---
#### [new 043] Logic-based Task Representation and Reward Shaping in Multiagent Reinforcement Learning
- **分类: cs.MA; cs.RO**

- **简介: 该论文研究多智能体强化学习中的任务规划问题，针对基于线性时序逻辑（LTL）的任务规范，提出基于逻辑的任務表示与奖励塑形方法。通过构建产品半马尔可夫决策过程，结合选项机制与奖励塑形，显著降低样本复杂度，加速最优策略学习，并在网格世界中验证了有效性。**

- **链接: [http://arxiv.org/pdf/2510.23615v1](http://arxiv.org/pdf/2510.23615v1)**

> **作者:** Nishant Doshi
>
> **摘要:** This paper presents an approach for accelerated learning of optimal plans for a given task represented using Linear Temporal Logic (LTL) in multi-agent systems. Given a set of options (temporally abstract actions) available to each agent, we convert the task specification into the corresponding Buchi Automaton and proceed with a model-free approach which collects transition samples and constructs a product Semi Markov Decision Process (SMDP) on-the-fly. Value-based Reinforcement Learning algorithms can then be used to synthesize a correct-by-design controller without learning the underlying transition model of the multi-agent system. The exponential sample complexity due to multiple agents is dealt with using a novel reward shaping approach. We test the proposed algorithm in a deterministic gridworld simulation for different tasks and find that the reward shaping results in significant reduction in convergence times. We also infer that using options becomes increasing more relevant as the state and action space increases in multi-agent systems.
>
---
#### [new 044] Coordinated Autonomous Drones for Human-Centered Fire Evacuation in Partially Observable Urban Environments
- **分类: cs.MA; cs.RO**

- **简介: 该论文研究多无人机协同救援任务，针对火灾中人类因恐慌偏离路线、环境信息不全等问题，提出基于POMDP的双无人机协作框架。通过融合心理学驱动的人类行为模型与强化学习算法，实现对受困者的实时定位、拦截与引导，显著提升疏散效率。**

- **链接: [http://arxiv.org/pdf/2510.23899v1](http://arxiv.org/pdf/2510.23899v1)**

> **作者:** Maria G. Mendoza; Addison Kalanther; Daniel Bostwick; Emma Stephan; Chinmay Maheshwari; Shankar Sastry
>
> **备注:** Accepted to IEEE Global Humanitarian Technology Conference (GHTC 2025). 8 pages, 4 figures
>
> **摘要:** Autonomous drone technology holds significant promise for enhancing search and rescue operations during evacuations by guiding humans toward safety and supporting broader emergency response efforts. However, their application in dynamic, real-time evacuation support remains limited. Existing models often overlook the psychological and emotional complexity of human behavior under extreme stress. In real-world fire scenarios, evacuees frequently deviate from designated safe routes due to panic and uncertainty. To address these challenges, this paper presents a multi-agent coordination framework in which autonomous Unmanned Aerial Vehicles (UAVs) assist human evacuees in real-time by locating, intercepting, and guiding them to safety under uncertain conditions. We model the problem as a Partially Observable Markov Decision Process (POMDP), where two heterogeneous UAV agents, a high-level rescuer (HLR) and a low-level rescuer (LLR), coordinate through shared observations and complementary capabilities. Human behavior is captured using an agent-based model grounded in empirical psychology, where panic dynamically affects decision-making and movement in response to environmental stimuli. The environment features stochastic fire spread, unknown evacuee locations, and limited visibility, requiring UAVs to plan over long horizons to search for humans and adapt in real-time. Our framework employs the Proximal Policy Optimization (PPO) algorithm with recurrent policies to enable robust decision-making in partially observable settings. Simulation results demonstrate that the UAV team can rapidly locate and intercept evacuees, significantly reducing the time required for them to reach safety compared to scenarios without UAV assistance.
>
---
## 更新

#### [replaced 001] Learning to See and Act: Task-Aware View Planning for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.05186v3](http://arxiv.org/pdf/2508.05186v3)**

> **作者:** Yongjie Bai; Zhouxia Wang; Yang Liu; Weixing Chen; Ziliang Chen; Mingtong Dai; Yongsen Zheng; Lingbo Liu; Guanbin Li; Liang Lin
>
> **备注:** 14 pages, 8 figures, project page: https://hcplab-sysu.github.io/TAVP
>
> **摘要:** Recent vision-language-action (VLA) models for multi-task robotic manipulation commonly rely on static viewpoints and shared visual encoders, which limit 3D perception and cause task interference, hindering robustness and generalization. In this work, we propose Task-Aware View Planning (TAVP), a framework designed to overcome these challenges by integrating active view planning with task-specific representation learning. TAVP employs an efficient exploration policy, accelerated by a novel pseudo-environment, to actively acquire informative views. Furthermore, we introduce a Mixture-of-Experts (MoE) visual encoder to disentangle features across different tasks, boosting both representation fidelity and task generalization. By learning to see the world in a task-aware way, TAVP generates more complete and discriminative visual representations, demonstrating significantly enhanced action prediction across a wide array of manipulation challenges. Extensive experiments on RLBench tasks show that our proposed TAVP model achieves superior performance over state-of-the-art fixed-view approaches. Visual results and code are provided at: https://hcplab-sysu.github.io/TAVP.
>
---
#### [replaced 002] Robust Point Cloud Reinforcement Learning via PCA-Based Canonicalization
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.20974v2](http://arxiv.org/pdf/2510.20974v2)**

> **作者:** Michael Bezick; Vittorio Giammarino; Ahmed H. Qureshi
>
> **摘要:** Reinforcement Learning (RL) from raw visual input has achieved impressive successes in recent years, yet it remains fragile to out-of-distribution variations such as changes in lighting, color, and viewpoint. Point Cloud Reinforcement Learning (PC-RL) offers a promising alternative by mitigating appearance-based brittleness, but its sensitivity to camera pose mismatches continues to undermine reliability in realistic settings. To address this challenge, we propose PCA Point Cloud (PPC), a canonicalization framework specifically tailored for downstream robotic control. PPC maps point clouds under arbitrary rigid-body transformations to a unique canonical pose, aligning observations to a consistent frame, thereby substantially decreasing viewpoint-induced inconsistencies. In our experiments, we show that PPC improves robustness to unseen camera poses across challenging robotic tasks, providing a principled alternative to domain randomization.
>
---
#### [replaced 003] Robot Cell Modeling via Exploratory Robot Motions: A Novel and Accessible Data-Driven Approach
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.01484v2](http://arxiv.org/pdf/2502.01484v2)**

> **作者:** Gaetano Meli; Niels Dehio
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** Generating a collision-free robot motion is crucial for safe applications in real-world settings. This requires an accurate model of all obstacle shapes within the constrained robot cell, which is particularly challenging and time-consuming. The difficulty is heightened in flexible production lines, where the environment model must be updated each time the robot cell is modified. Furthermore, sensor-based methods often necessitate costly hardware and calibration procedures and can be influenced by environmental factors (e.g., light conditions or reflections). To address these challenges, we present a novel data-driven approach to modeling a cluttered workspace, leveraging solely the robot internal joint encoders to capture exploratory motions. By computing the corresponding swept volume (SV), we generate a (conservative) mesh of the environment that is subsequently used for collision checking within established path planning and control methods. Our method significantly reduces the complexity and cost of classical environment modeling by removing the need for computer-aided design (CAD) files and external sensors. We validate the approach with the KUKA LBR iisy collaborative robot in a pick-and-place scenario. In less than three minutes of exploratory robot motions and less than four additional minutes of computation time, we obtain an accurate model that enables collision-free motions. Our approach is intuitive and easy to use, making it accessible to users without specialized technical knowledge. It is applicable to all types of industrial robots or cobots.
>
---
#### [replaced 004] Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.20072v2](http://arxiv.org/pdf/2508.20072v2)**

> **作者:** Zhixuan Liang; Yizhuo Li; Tianshuo Yang; Chengyue Wu; Sitong Mao; Tian Nian; Liuao Pei; Shunbo Zhou; Xiaokang Yang; Jiangmiao Pang; Yao Mu; Ping Luo
>
> **备注:** 16 pages
>
> **摘要:** Vision-Language-Action (VLA) models adapt large vision-language backbones to map images and instructions into robot actions. However, prevailing VLAs either generate actions auto-regressively in a fixed left-to-right order or attach separate MLP or diffusion heads outside the backbone, leading to fragmented information pathways and specialized training requirements that hinder a unified, scalable architecture. We present Discrete Diffusion VLA, a unified-transformer policy that models discretized action chunks with discrete diffusion. The design retains diffusion's progressive refinement paradigm while remaining natively compatible with the discrete token interface of VLMs. Our method achieves an adaptive decoding order that resolves easy action elements before harder ones and uses secondary re-masking to revisit uncertain predictions across refinement rounds, which improves consistency and enables robust error correction. This unified decoder preserves pre-trained vision-language priors, supports parallel decoding, breaks the autoregressive bottleneck, and reduces the number of function evaluations. Discrete Diffusion VLA achieves 96.3% avg. success rates on LIBERO, 71.2% visual matching on SimplerEnv-Fractal and 54.2% overall on SimplerEnv-Bridge, improving over autoregressive, MLP decoder and continuous diffusion baselines. These findings indicate that discrete-diffusion VLA supports precise action modeling and consistent training, laying groundwork for scaling VLA to larger models and datasets. Our project page is https://github.com/Liang-ZX/DiscreteDiffusionVLA
>
---
#### [replaced 005] Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views
- **分类: cs.CV; cs.CL; cs.RO; I.2.10; I.2.9; I.2.7; H.5.2**

- **链接: [http://arxiv.org/pdf/2510.22672v2](http://arxiv.org/pdf/2510.22672v2)**

> **作者:** Anna Deichler; Jonas Beskow
>
> **备注:** 10 pages, 6 figures, 2 tables. Accepted to the NeurIPS 2025 Workshop on SPACE in Vision, Language, and Embodied AI (SpaVLE). Dataset: https://huggingface.co/datasets/annadeichler/KTH-ARIA-referential
>
> **摘要:** We introduce Look and Tell, a multimodal dataset for studying referential communication across egocentric and exocentric perspectives. Using Meta Project Aria smart glasses and stationary cameras, we recorded synchronized gaze, speech, and video as 25 participants instructed a partner to identify ingredients in a kitchen. Combined with 3D scene reconstructions, this setup provides a benchmark for evaluating how different spatial representations (2D vs. 3D; ego vs. exo) affect multimodal grounding. The dataset contains 3.67 hours of recordings, including 2,707 richly annotated referential expressions, and is designed to advance the development of embodied agents that can understand and engage in situated dialogue.
>
---
#### [replaced 006] Acoustic Neural 3D Reconstruction Under Pose Drift
- **分类: eess.SP; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.08930v2](http://arxiv.org/pdf/2503.08930v2)**

> **作者:** Tianxiang Lin; Mohamad Qadri; Kevin Zhang; Adithya Pediredla; Christopher A. Metzler; Michael Kaess
>
> **备注:** 8 pages, 8 figures. This paper is accepted by 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** We consider the problem of optimizing neural implicit surfaces for 3D reconstruction using acoustic images collected with drifting sensor poses. The accuracy of current state-of-the-art 3D acoustic modeling algorithms is highly dependent on accurate pose estimation; small errors in sensor pose can lead to severe reconstruction artifacts. In this paper, we propose an algorithm that jointly optimizes the neural scene representation and sonar poses. Our algorithm does so by parameterizing the 6DoF poses as learnable parameters and backpropagating gradients through the neural renderer and implicit representation. We validated our algorithm on both real and simulated datasets. It produces high-fidelity 3D reconstructions even under significant pose drift.
>
---
#### [replaced 007] Two-Stage Learning of Stabilizing Neural Controllers via Zubov Sampling and Iterative Domain Expansion
- **分类: cs.LG; cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.01356v2](http://arxiv.org/pdf/2506.01356v2)**

> **作者:** Haoyu Li; Xiangru Zhong; Bin Hu; Huan Zhang
>
> **备注:** NeurIPS 2025
>
> **摘要:** Learning-based neural network (NN) control policies have shown impressive empirical performance. However, obtaining stability guarantees and estimates of the region of attraction of these learned neural controllers is challenging due to the lack of stable and scalable training and verification algorithms. Although previous works in this area have achieved great success, much conservatism remains in their frameworks. In this work, we propose a novel two-stage training framework to jointly synthesize a controller and a Lyapunov function for continuous-time systems. By leveraging a Zubov-inspired region of attraction characterization to directly estimate stability boundaries, we propose a novel training-data sampling strategy and a domain-updating mechanism that significantly reduces the conservatism in training. Moreover, unlike existing works on continuous-time systems that rely on an SMT solver to formally verify the Lyapunov condition, we extend state-of-the-art neural network verifier $\alpha,\!\beta$-CROWN with the capability of performing automatic bound propagation through the Jacobian of dynamical systems and a novel verification scheme that avoids expensive bisection. To demonstrate the effectiveness of our approach, we conduct numerical experiments by synthesizing and verifying controllers on several challenging nonlinear systems across multiple dimensions. We show that our training can yield region of attractions with volume $5 - 1.5\cdot 10^{5}$ times larger compared to the baselines, and our verification on continuous systems can be up to $40-10{,}000$ times faster compared to the traditional SMT solver dReal. Our code is available at https://github.com/Verified-Intelligence/Two-Stage_Neural_Controller_Training.
>
---
#### [replaced 008] Procedural Generation of Articulated Simulation-Ready Assets
- **分类: cs.RO; cs.GR**

- **链接: [http://arxiv.org/pdf/2505.10755v3](http://arxiv.org/pdf/2505.10755v3)**

> **作者:** Abhishek Joshi; Beining Han; Jack Nugent; Max Gonzalez Saez-Diez; Yiming Zuo; Jonathan Liu; Hongyu Wen; Stamatis Alexandropoulos; Karhan Kayan; Anna Calveri; Tao Sun; Gaowen Liu; Yi Shao; Alexander Raistrick; Jia Deng
>
> **备注:** Updated to include information on newly implemented assets, new experimental results (both simulation and real world), and additional features including material and dynamics parameters
>
> **摘要:** We introduce Infinigen-Articulated, a toolkit for generating realistic, procedurally generated articulated assets for robotics simulation. We include procedural generators for 18 common articulated object categories along with high-level utilities for use creating custom articulated assets in Blender. We also provide an export pipeline to integrate the resulting assets along with their physical properties into common robotics simulators. Experiments demonstrate that assets sampled from these generators are effective for movable object segmentation, training generalizable reinforcement learning policies, and sim-to-real transfer of imitation learning policies.
>
---
#### [replaced 009] GaussianFusion: Gaussian-Based Multi-Sensor Fusion for End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.00034v2](http://arxiv.org/pdf/2506.00034v2)**

> **作者:** Shuai Liu; Quanmin Liang; Zefeng Li; Boyang Li; Kai Huang
>
> **备注:** Accepted at NeurIPS2025 (Spotlight)
>
> **摘要:** Multi-sensor fusion is crucial for improving the performance and robustness of end-to-end autonomous driving systems. Existing methods predominantly adopt either attention-based flatten fusion or bird's eye view fusion through geometric transformations. However, these approaches often suffer from limited interpretability or dense computational overhead. In this paper, we introduce GaussianFusion, a Gaussian-based multi-sensor fusion framework for end-to-end autonomous driving. Our method employs intuitive and compact Gaussian representations as intermediate carriers to aggregate information from diverse sensors. Specifically, we initialize a set of 2D Gaussians uniformly across the driving scene, where each Gaussian is parameterized by physical attributes and equipped with explicit and implicit features. These Gaussians are progressively refined by integrating multi-modal features. The explicit features capture rich semantic and spatial information about the traffic scene, while the implicit features provide complementary cues beneficial for trajectory planning. To fully exploit rich spatial and semantic information in Gaussians, we design a cascade planning head that iteratively refines trajectory predictions through interactions with Gaussians. Extensive experiments on the NAVSIM and Bench2Drive benchmarks demonstrate the effectiveness and robustness of the proposed GaussianFusion framework. The source code will be released at https://github.com/Say2L/GaussianFusion.
>
---
#### [replaced 010] SimpleVSF: VLM-Scoring Fusion for Trajectory Prediction of End-to-End Autonomous Driving
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.17191v2](http://arxiv.org/pdf/2510.17191v2)**

> **作者:** Peiru Zheng; Yun Zhao; Zhan Gong; Hong Zhu; Shaohua Wu
>
> **摘要:** End-to-end autonomous driving has emerged as a promising paradigm for achieving robust and intelligent driving policies. However, existing end-to-end methods still face significant challenges, such as suboptimal decision-making in complex scenarios. In this paper,we propose SimpleVSF (Simple VLM-Scoring Fusion), a novel framework that enhances end-to-end planning by leveraging the cognitive capabilities of Vision-Language Models (VLMs) and advanced trajectory fusion techniques. We utilize the conventional scorers and the novel VLM-enhanced scorers. And we leverage a robust weight fusioner for quantitative aggregation and a powerful VLM-based fusioner for qualitative, context-aware decision-making. As the leading approach in the ICCV 2025 NAVSIM v2 End-to-End Driving Challenge, our SimpleVSF framework demonstrates state-of-the-art performance, achieving a superior balance between safety, comfort, and efficiency.
>
---
#### [replaced 011] Taxonomy and Trends in Reinforcement Learning for Robotics and Control Systems: A Structured Review
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.21758v2](http://arxiv.org/pdf/2510.21758v2)**

> **作者:** Kumater Ter; Ore-Ofe Ajayi; Daniel Udekwe
>
> **摘要:** Reinforcement learning (RL) has become a foundational approach for enabling intelligent robotic behavior in dynamic and uncertain environments. This work presents an in-depth review of RL principles, advanced deep reinforcement learning (DRL) algorithms, and their integration into robotic and control systems. Beginning with the formalism of Markov Decision Processes (MDPs), the study outlines essential elements of the agent-environment interaction and explores core algorithmic strategies including actor-critic methods, value-based learning, and policy gradients. Emphasis is placed on modern DRL techniques such as DDPG, TD3, PPO, and SAC, which have shown promise in solving high-dimensional, continuous control tasks. A structured taxonomy is introduced to categorize RL applications across domains such as locomotion, manipulation, multi-agent coordination, and human-robot interaction, along with training methodologies and deployment readiness levels. The review synthesizes recent research efforts, highlighting technical trends, design patterns, and the growing maturity of RL in real-world robotics. Overall, this work aims to bridge theoretical advances with practical implementations, providing a consolidated perspective on the evolving role of RL in autonomous robotic systems.
>
---
#### [replaced 012] FieldGen: From Teleoperated Pre-Manipulation Trajectories to Field-Guided Data Generation
- **分类: cs.RO; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2510.20774v2](http://arxiv.org/pdf/2510.20774v2)**

> **作者:** Wenhao Wang; Kehe Ye; Xinyu Zhou; Tianxing Chen; Cao Min; Qiaoming Zhu; Xiaokang Yang; Ping Luo; Yongjian Shen; Yang Yang; Maoqing Yao; Yao Mu
>
> **备注:** Webpage: https://fieldgen.github.io/
>
> **摘要:** Large-scale and diverse datasets are vital for training robust robotic manipulation policies, yet existing data collection methods struggle to balance scale, diversity, and quality. Simulation offers scalability but suffers from sim-to-real gaps, while teleoperation yields high-quality demonstrations with limited diversity and high labor cost. We introduce FieldGen, a field-guided data generation framework that enables scalable, diverse, and high-quality real-world data collection with minimal human supervision. FieldGen decomposes manipulation into two stages: a pre-manipulation phase, allowing trajectory diversity, and a fine manipulation phase requiring expert precision. Human demonstrations capture key contact and pose information, after which an attraction field automatically generates diverse trajectories converging to successful configurations. This decoupled design combines scalable trajectory diversity with precise supervision. Moreover, FieldGen-Reward augments generated data with reward annotations to further enhance policy learning. Experiments demonstrate that policies trained with FieldGen achieve higher success rates and improved stability compared to teleoperation-based baselines, while significantly reducing human effort in long-term real-world data collection. Webpage is available at https://fieldgen.github.io/.
>
---
#### [replaced 013] Performance evaluation of a ROS2 based Automated Driving System
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2411.11607v3](http://arxiv.org/pdf/2411.11607v3)**

> **作者:** Jorin Kouril; Bernd Schäufele; Ilja Radusch; Bettina Schnor
>
> **备注:** Published and presented at VEHITS 2024, Proceedings of the 10th International Conference on Vehicle Technology and Intelligent Transport Systems - VEHITS; 2024
>
> **摘要:** Automated driving is currently a prominent area of scientific work. In the future, highly automated driving and new Advanced Driver Assistance Systems will become reality. While Advanced Driver Assistance Systems and automated driving functions for certain domains are already commercially available, ubiquitous automated driving in complex scenarios remains a subject of ongoing research. Contrarily to single-purpose Electronic Control Units, the software for automated driving is often executed on high performance PCs. The Robot Operating System 2 (ROS2) is commonly used to connect components in an automated driving system. Due to the time critical nature of automated driving systems, the performance of the framework is especially important. In this paper, a thorough performance evaluation of ROS2 is conducted, both in terms of timeliness and error rate. The results show that ROS2 is a suitable framework for automated driving systems.
>
---
#### [replaced 014] Concurrent-Allocation Task Execution for Multi-Robot Path-Crossing-Minimal Navigation in Obstacle Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.09230v2](http://arxiv.org/pdf/2504.09230v2)**

> **作者:** Bin-Bin Hu; Weijia Yao; Yanxin Zhou; Henglai Wei; Chen Lv
>
> **备注:** Accepted in IEEE Transactions on Robotics
>
> **摘要:** Reducing undesirable path crossings among trajectories of different robots is vital in multi-robot navigation missions, which not only reduces detours and conflict scenarios, but also enhances navigation efficiency and boosts productivity. Despite recent progress in multi-robot path-crossing-minimal (MPCM) navigation, the majority of approaches depend on the minimal squared-distance reassignment of suitable desired points to robots directly. However, if obstacles occupy the passing space, calculating the actual robot-point distances becomes complex or intractable, which may render the MPCM navigation in obstacle environments inefficient or even infeasible. In this paper, the concurrent-allocation task execution (CATE) algorithm is presented to address this problem (i.e., MPCM navigation in obstacle environments). First, the path-crossing-related elements in terms of (i) robot allocation, (ii) desired-point convergence, and (iii) collision and obstacle avoidance are encoded into integer and control barrier function (CBF) constraints. Then, the proposed constraints are used in an online constrained optimization framework, which implicitly yet effectively minimizes the possible path crossings and trajectory length in obstacle environments by minimizing the desired point allocation cost and slack variables in CBF constraints simultaneously. In this way, the MPCM navigation in obstacle environments can be achieved with flexible spatial orderings. Note that the feasibility of solutions and the asymptotic convergence property of the proposed CATE algorithm in obstacle environments are both guaranteed, and the calculation burden is also reduced by concurrently calculating the optimal allocation and the control input directly without the path planning process.
>
---
#### [replaced 015] HyPerNav: Hybrid Perception for Object-Oriented Navigation in Unknown Environment
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.22917v2](http://arxiv.org/pdf/2510.22917v2)**

> **作者:** Zecheng Yin; Hao Zhao; Zhen Li
>
> **备注:** under review
>
> **摘要:** Objective-oriented navigation(ObjNav) enables robot to navigate to target object directly and autonomously in an unknown environment. Effective perception in navigation in unknown environment is critical for autonomous robots. While egocentric observations from RGB-D sensors provide abundant local information, real-time top-down maps offer valuable global context for ObjNav. Nevertheless, the majority of existing studies focus on a single source, seldom integrating these two complementary perceptual modalities, despite the fact that humans naturally attend to both. With the rapid advancement of Vision-Language Models(VLMs), we propose Hybrid Perception Navigation (HyPerNav), leveraging VLMs' strong reasoning and vision-language understanding capabilities to jointly perceive both local and global information to enhance the effectiveness and intelligence of navigation in unknown environments. In both massive simulation evaluation and real-world validation, our methods achieved state-of-the-art performance against popular baselines. Benefiting from hybrid perception approach, our method captures richer cues and finds the objects more effectively, by simultaneously leveraging information understanding from egocentric observations and the top-down map. Our ablation study further proved that either of the hybrid perception contributes to the navigation performance.
>
---
#### [replaced 016] Autonomous Horizon-based Asteroid Navigation With Observability-constrained Maneuvers
- **分类: cs.RO; math.OC**

- **链接: [http://arxiv.org/pdf/2501.15806v2](http://arxiv.org/pdf/2501.15806v2)**

> **作者:** Aditya Arjun Anibha; Kenshiro Oguri
>
> **备注:** 52 pages, 18 figures, published in the Journal of the Astronautical Sciences
>
> **摘要:** Small body exploration is a pertinent challenge due to low gravity environments and strong sensitivity to perturbations like Solar Radiation Pressure (SRP). Thus, autonomous methods are being developed to enable safe navigation and control around small bodies. These methods often involve using Optical Navigation (OpNav) to determine the spacecraft's location. Ensuring OpNav reliability would allow the spacecraft to maintain an accurate state estimate throughout its mission. This research presents an observability-constrained Lyapunov controller that steers a spacecraft to a desired target orbit while guaranteeing continuous OpNav observability. We design observability path constraints to avoid regions where horizon-based OpNav methods exhibit poor performance, ensuring control input that maintains good observability. This controller is implemented with a framework that simulates small body dynamics, synthetic image generation, edge detection, horizon-based OpNav, and filtering. We evaluate the approach in two representative scenarios, orbit maintenance and approach with circularization, around spherical and ellipsoidal target bodies. In Monte Carlo simulations, the proposed approach improves the rate of attaining target orbits without observability violations by up to 94% compared to an unconstrained Lyapunov baseline, demonstrating improved robustness over conventional methods.
>
---
#### [replaced 017] COMPASS: Cross-embodiment Mobility Policy via Residual RL and Skill Synthesis
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.16372v3](http://arxiv.org/pdf/2502.16372v3)**

> **作者:** Wei Liu; Huihua Zhao; Chenran Li; Yuchen Deng; Joydeep Biswas; Soha Pouya; Yan Chang
>
> **摘要:** As robots are increasingly deployed in diverse application domains, enabling robust mobility across different embodiments has become a critical challenge. Classical mobility stacks, though effective on specific platforms, require extensive per-robot tuning and do not scale easily to new embodiments. Learning-based approaches, such as imitation learning (IL), offer alternatives, but face significant limitations on the need for high-quality demonstrations for each embodiment. To address these challenges, we introduce COMPASS, a unified framework that enables scalable cross-embodiment mobility using expert demonstrations from only a single embodiment. We first pre-train a mobility policy on a single robot using IL, combining a world model with a policy model. We then apply residual reinforcement learning (RL) to efficiently adapt this policy to diverse embodiments through corrective refinements. Finally, we distill specialist policies into a single generalist policy conditioned on an embodiment embedding vector. This design significantly reduces the burden of collecting data while enabling robust generalization across a wide range of robot designs. Our experiments demonstrate that COMPASS scales effectively across diverse robot platforms while maintaining adaptability to various environment configurations, achieving a generalist policy with a success rate approximately 5X higher than the pre-trained IL policy on unseen embodiments, and further demonstrates zero-shot sim-to-real transfer.
>
---
#### [replaced 018] DynaFlow: Dynamics-embedded Flow Matching for Physically Consistent Motion Generation from State-only Demonstrations
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.19804v2](http://arxiv.org/pdf/2509.19804v2)**

> **作者:** Sowoo Lee; Dongyun Kang; Jaehyun Park; Hae-Won Park
>
> **备注:** 8 pages
>
> **摘要:** This paper introduces DynaFlow, a novel framework that embeds a differentiable simulator directly into a flow matching model. By generating trajectories in the action space and mapping them to dynamically feasible state trajectories via the simulator, DynaFlow ensures all outputs are physically consistent by construction. This end-to-end differentiable architecture enables training on state-only demonstrations, allowing the model to simultaneously generate physically consistent state trajectories while inferring the underlying action sequences required to produce them. We demonstrate the effectiveness of our approach through quantitative evaluations and showcase its real-world applicability by deploying the generated actions onto a physical Go1 quadruped robot. The robot successfully reproduces diverse gait present in the dataset, executes long-horizon motions in open-loop control and translates infeasible kinematic demonstrations into dynamically executable, stylistic behaviors. These hardware experiments validate that DynaFlow produces deployable, highly effective motions on real-world hardware from state-only demonstrations, effectively bridging the gap between kinematic data and real-world execution.
>
---
#### [replaced 019] Boosting Omnidirectional Stereo Matching with a Pre-trained Depth Foundation Model
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.23502v3](http://arxiv.org/pdf/2503.23502v3)**

> **作者:** Jannik Endres; Oliver Hahn; Charles Corbière; Simone Schaub-Meyer; Stefan Roth; Alexandre Alahi
>
> **备注:** Accepted at IROS 2025. Project page: https://vita-epfl.github.io/DFI-OmniStereo-website/
>
> **摘要:** Omnidirectional depth perception is essential for mobile robotics applications that require scene understanding across a full 360{\deg} field of view. Camera-based setups offer a cost-effective option by using stereo depth estimation to generate dense, high-resolution depth maps without relying on expensive active sensing. However, existing omnidirectional stereo matching approaches achieve only limited depth accuracy across diverse environments, depth ranges, and lighting conditions, due to the scarcity of real-world data. We present DFI-OmniStereo, a novel omnidirectional stereo matching method that leverages a large-scale pre-trained foundation model for relative monocular depth estimation within an iterative optimization-based stereo matching architecture. We introduce a dedicated two-stage training strategy to utilize the relative monocular depth features for our omnidirectional stereo matching before scale-invariant fine-tuning. DFI-OmniStereo achieves state-of-the-art results on the real-world Helvipad dataset, reducing disparity MAE by approximately 16% compared to the previous best omnidirectional stereo method.
>
---
#### [replaced 020] GRS: Generating Robotic Simulation Tasks from Real-World Images
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.15536v3](http://arxiv.org/pdf/2410.15536v3)**

> **作者:** Alex Zook; Fan-Yun Sun; Josef Spjut; Valts Blukis; Stan Birchfield; Jonathan Tremblay
>
> **摘要:** We introduce GRS (Generating Robotic Simulation tasks), a system addressing real-to-sim for robotic simulations. GRS creates digital twin simulations from single RGB-D observations with solvable tasks for virtual agent training. Using vision-language models (VLMs), our pipeline operates in three stages: 1) scene comprehension with SAM2 for segmentation and object description, 2) matching objects with simulation-ready assets, and 3) generating appropriate tasks. We ensure simulation-task alignment through generated test suites and introduce a router that iteratively refines both simulation and test code. Experiments demonstrate our system's effectiveness in object correspondence and task environment generation through our novel router mechanism.
>
---
#### [replaced 021] Radar and Event Camera Fusion for Agile Robot Ego-Motion Estimation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18443v2](http://arxiv.org/pdf/2506.18443v2)**

> **作者:** Yang Lyu; Zhenghao Zou; Yanfeng Li; Xiaohu Guo; Chunhui Zhao; Quan Pan
>
> **备注:** 2025.10.28 version v2 for TwistEstimator
>
> **摘要:** Achieving reliable ego motion estimation for agile robots, e.g., aerobatic aircraft, remains challenging because most robot sensors fail to respond timely and clearly to highly dynamic robot motions, often resulting in measurement blurring, distortion, and delays. In this paper, we propose an IMU-free and feature-association-free framework to achieve aggressive ego-motion velocity estimation of a robot platform in highly dynamic scenarios by combining two types of exteroceptive sensors, an event camera and a millimeter wave radar, First, we used instantaneous raw events and Doppler measurements to derive rotational and translational velocities directly. Without a sophisticated association process between measurement frames, the proposed method is more robust in texture-less and structureless environments and is more computationally efficient for edge computing devices. Then, in the back-end, we propose a continuous-time state-space model to fuse the hybrid time-based and event-based measurements to estimate the ego-motion velocity in a fixed-lagged smoother fashion. In the end, we validate our velometer framework extensively in self-collected experiment datasets. The results indicate that our IMU-free and association-free ego motion estimation framework can achieve reliable and efficient velocity output in challenging environments. The source code, illustrative video and dataset are available at https://github.com/ZzhYgwh/TwistEstimator.
>
---
