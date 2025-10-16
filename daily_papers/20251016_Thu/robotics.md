# 机器人 cs.RO

- **最新发布 41 篇**

- **更新 26 篇**

## 最新发布

#### [new 001] Characterizing Lidar Point-Cloud Adversities Using a Vector Field Visualization
- **分类: cs.RO**

- **简介: 该论文提出一种面向离线分析的矢量场可视化方法，用于辅助人工识别影响激光雷达配准的 adversity 模式。通过可视化配准点云间的局部差异，揭示原始数据中难以察觉的问题模式，并在仿真与实测数据中验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2510.13619v1](http://arxiv.org/pdf/2510.13619v1)**

> **作者:** Daniel Choate; Jason Rife
>
> **备注:** This is the preprint version of the paper published in: Proceedings of the 37th International Technical Meeting of the Satellite Division of The Institute of Navigation (ION GNSS+ 2024), September 2024 The final version is available at https://doi.org/10.33012/2024.19864
>
> **摘要:** In this paper we introduce a visualization methodology to aid a human analyst in classifying adversity modes that impact lidar scan matching. Our methodology is intended for offline rather than real-time analysis. The method generates a vector-field plot that characterizes local discrepancies between a pair of registered point clouds. The vector field plot reveals patterns that would be difficult for the analyst to extract from raw point-cloud data. After introducing our methodology, we apply the process to two proof-of-concept examples: one a simulation study and the other a field experiment. For both data sets, a human analyst was able to reason about a series of adversity mechanisms and iteratively remove those mechanisms from the raw data, to help focus attention on progressively smaller discrepancies.
>
---
#### [new 002] Adversarial Fine-tuning in Offline-to-Online Reinforcement Learning for Robust Robot Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究离线到在线强化学习中的机器人鲁棒控制，旨在解决离线策略在动作扰动下脆弱的问题。提出对抗性微调框架，结合性能感知课程调整扰动概率，提升策略抗干扰能力，兼顾鲁棒性与稳定性。**

- **链接: [http://arxiv.org/pdf/2510.13358v1](http://arxiv.org/pdf/2510.13358v1)**

> **作者:** Shingo Ayabe; Hiroshi Kera; Kazuhiko Kawamoto
>
> **备注:** 16 pages, 8 figures
>
> **摘要:** Offline reinforcement learning enables sample-efficient policy acquisition without risky online interaction, yet policies trained on static datasets remain brittle under action-space perturbations such as actuator faults. This study introduces an offline-to-online framework that trains policies on clean data and then performs adversarial fine-tuning, where perturbations are injected into executed actions to induce compensatory behavior and improve resilience. A performance-aware curriculum further adjusts the perturbation probability during training via an exponential-moving-average signal, balancing robustness and stability throughout the learning process. Experiments on continuous-control locomotion tasks demonstrate that the proposed method consistently improves robustness over offline-only baselines and converges faster than training from scratch. Matching the fine-tuning and evaluation conditions yields the strongest robustness to action-space perturbations, while the adaptive curriculum strategy mitigates the degradation of nominal performance observed with the linear curriculum strategy. Overall, the results show that adversarial fine-tuning enables adaptive and robust control under uncertain environments, bridging the gap between offline efficiency and online adaptability.
>
---
#### [new 003] The Omega Turn: A General Turning Template for Elongate Robots
- **分类: cs.RO**

- **简介: 该论文研究细长无肢机器人的转向控制问题，受线虫Omega转弯启发，提出一种基于双行波叠加的通用转向模板，实现了机器人在复杂环境中的鲁棒转向，并可推广至多足细长机器人。**

- **链接: [http://arxiv.org/pdf/2510.12970v1](http://arxiv.org/pdf/2510.12970v1)**

> **作者:** Baxi Chong; Tianyu Wang; Kelimar Diaz; Christopher J. Pierce; Eva Erickson; Julian Whitman; Yuelin Deng; Esteban Flores; Ruijie Fu; Juntao He; Jianfeng Lin; Hang Lu; Guillaume Sartoretti; Howie Choset; Daniel I. Goldman
>
> **摘要:** Elongate limbless robots have the potential to locomote through tightly packed spaces for applications such as search-and-rescue and industrial inspections. The capability to effectively and robustly maneuver elongate limbless robots is crucial to realize such potential. However, there has been limited research on turning strategies for such systems. To achieve effective and robust turning performance in cluttered spaces, we take inspiration from a microscopic nematode, C. elegans, which exhibits remarkable maneuverability in rheologically complex environments partially because of its ability to perform omega turns. Despite recent efforts to analyze omega turn kinematics, it remains unknown if there exists a wave equation sufficient to prescribe an omega turn, let alone its reconstruction on robot platforms. Here, using a comparative theory-biology approach, we prescribe the omega turn as a superposition of two traveling waves. With wave equations as a guideline, we design a controller for limbless robots enabling robust and effective turning behaviors in lab and cluttered field environments. Finally, we show that such omega turn controllers can also generalize to elongate multi-legged robots, demonstrating an alternative effective body-driven turning strategy for elongate robots, with and without limbs.
>
---
#### [new 004] Bridge the Gap: Enhancing Quadruped Locomotion with Vertical Ground Perturbations
- **分类: cs.RO**

- **简介: 该论文研究四足机器人在垂直地面扰动下的行走鲁棒性，提出通过模拟振荡桥梁环境，结合强化学习与域随机化训练多种步态策略，实现无需先验知识的稳定通行，提升机器人在动态地形中的适应能力。**

- **链接: [http://arxiv.org/pdf/2510.13488v1](http://arxiv.org/pdf/2510.13488v1)**

> **作者:** Maximilian Stasica; Arne Bick; Nico Bohlinger; Omid Mohseni; Max Johannes Alois Fritzsche; Clemens Hübler; Jan Peters; André Seyfarth
>
> **摘要:** Legged robots, particularly quadrupeds, excel at navigating rough terrains, yet their performance under vertical ground perturbations, such as those from oscillating surfaces, remains underexplored. This study introduces a novel approach to enhance quadruped locomotion robustness by training the Unitree Go2 robot on an oscillating bridge - a 13.24-meter steel-and-concrete structure with a 2.0 Hz eigenfrequency designed to perturb locomotion. Using Reinforcement Learning (RL) with the Proximal Policy Optimization (PPO) algorithm in a MuJoCo simulation, we trained 15 distinct locomotion policies, combining five gaits (trot, pace, bound, free, default) with three training conditions: rigid bridge and two oscillating bridge setups with differing height regulation strategies (relative to bridge surface or ground). Domain randomization ensured zero-shot transfer to the real-world bridge. Our results demonstrate that policies trained on the oscillating bridge exhibit superior stability and adaptability compared to those trained on rigid surfaces. Our framework enables robust gait patterns even without prior bridge exposure. These findings highlight the potential of simulation-based RL to improve quadruped locomotion during dynamic ground perturbations, offering insights for designing robots capable of traversing vibrating environments.
>
---
#### [new 005] InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出InternVLA-M1，面向通用机器人策略，解决指令跟随中的空间定位与动作生成问题。通过空间引导的视觉-语言-动作训练框架，实现“在哪做”和“怎么做”的解耦，提升机器人在多任务、新物体和复杂场景下的泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.13778v1](http://arxiv.org/pdf/2510.13778v1)**

> **作者:** Xinyi Chen; Yilun Chen; Yanwei Fu; Ning Gao; Jiaya Jia; Weiyang Jin; Hao Li; Yao Mu; Jiangmiao Pang; Yu Qiao; Yang Tian; Bin Wang; Bolun Wang; Fangjing Wang; Hanqing Wang; Tai Wang; Ziqin Wang; Xueyuan Wei; Chao Wu; Shuai Yang; Jinhui Ye; Junqiu Yu; Jia Zeng; Jingjing Zhang; Jinyu Zhang; Shi Zhang; Feng Zheng; Bowen Zhou; Yangkun Zhu
>
> **备注:** Technical report
>
> **摘要:** We introduce InternVLA-M1, a unified framework for spatial grounding and robot control that advances instruction-following robots toward scalable, general-purpose intelligence. Its core idea is spatially guided vision-language-action training, where spatial grounding serves as the critical link between instructions and robot actions. InternVLA-M1 employs a two-stage pipeline: (i) spatial grounding pre-training on over 2.3M spatial reasoning data to determine ``where to act'' by aligning instructions with visual, embodiment-agnostic positions, and (ii) spatially guided action post-training to decide ``how to act'' by generating embodiment-aware actions through plug-and-play spatial prompting. This spatially guided training recipe yields consistent gains: InternVLA-M1 outperforms its variant without spatial guidance by +14.6% on SimplerEnv Google Robot, +17% on WidowX, and +4.3% on LIBERO Franka, while demonstrating stronger spatial reasoning capability in box, point, and trace prediction. To further scale instruction following, we built a simulation engine to collect 244K generalizable pick-and-place episodes, enabling a 6.2% average improvement across 200 tasks and 3K+ objects. In real-world clustered pick-and-place, InternVLA-M1 improved by 7.3%, and with synthetic co-training, achieved +20.6% on unseen objects and novel configurations. Moreover, in long-horizon reasoning-intensive scenarios, it surpassed existing works by over 10%. These results highlight spatially guided training as a unifying principle for scalable and resilient generalist robots. Code and models are available at https://github.com/InternRobotics/InternVLA-M1.
>
---
#### [new 006] Real-Time Knee Angle Prediction Using EMG and Kinematic Data with an Attention-Based CNN-LSTM Network and Transfer Learning Across Multiple Datasets
- **分类: cs.RO**

- **简介: 该论文研究基于EMG和运动学数据的实时膝关节角度预测，旨在解决模型泛化性差、数据需求大及实时性不足的问题。作者提出一种带注意力机制的CNN-LSTM模型，结合跨数据集迁移学习，在少量新用户数据下实现高精度短长期预测。**

- **链接: [http://arxiv.org/pdf/2510.13443v1](http://arxiv.org/pdf/2510.13443v1)**

> **作者:** Mojtaba Mollahossein; Gholamreza Vossoughi; Mohammad Hossein Rohban
>
> **摘要:** Electromyography (EMG) signals are widely used for predicting body joint angles through machine learning (ML) and deep learning (DL) methods. However, these approaches often face challenges such as limited real-time applicability, non-representative test conditions, and the need for large datasets to achieve optimal performance. This paper presents a transfer-learning framework for knee joint angle prediction that requires only a few gait cycles from new subjects. Three datasets - Georgia Tech, the University of California Irvine (UCI), and the Sharif Mechatronic Lab Exoskeleton (SMLE) - containing four EMG channels relevant to knee motion were utilized. A lightweight attention-based CNN-LSTM model was developed and pre-trained on the Georgia Tech dataset, then transferred to the UCI and SMLE datasets. The proposed model achieved Normalized Mean Absolute Errors (NMAE) of 6.8 percent and 13.7 percent for one-step and 50-step predictions on abnormal subjects using EMG inputs alone. Incorporating historical knee angles reduced the NMAE to 3.1 percent and 3.5 percent for normal subjects, and to 2.8 percent and 7.5 percent for abnormal subjects. When further adapted to the SMLE exoskeleton with EMG, kinematic, and interaction force inputs, the model achieved 1.09 percent and 3.1 percent NMAE for one- and 50-step predictions, respectively. These results demonstrate robust performance and strong generalization for both short- and long-term rehabilitation scenarios.
>
---
#### [new 007] PlanarMesh: Building Compact 3D Meshes from LiDAR using Incremental Adaptive Resolution Reconstruction
- **分类: cs.RO**

- **简介: 该论文提出PlanarMesh，用于实时构建紧凑且精细的3D LiDAR地图。针对计算效率与存储开销问题，设计增量式平面-网格混合表示，结合曲率与空闲空间自适应调整分辨率，实现高精度、小体积、实时性兼顾的在线重建。**

- **链接: [http://arxiv.org/pdf/2510.13599v1](http://arxiv.org/pdf/2510.13599v1)**

> **作者:** Jiahao Wang; Nived Chebrolu; Yifu Tao; Lintong Zhang; Ayoung Kim; Maurice Fallon
>
> **摘要:** Building an online 3D LiDAR mapping system that produces a detailed surface reconstruction while remaining computationally efficient is a challenging task. In this paper, we present PlanarMesh, a novel incremental, mesh-based LiDAR reconstruction system that adaptively adjusts mesh resolution to achieve compact, detailed reconstructions in real-time. It introduces a new representation, planar-mesh, which combines plane modeling and meshing to capture both large surfaces and detailed geometry. The planar-mesh can be incrementally updated considering both local surface curvature and free-space information from sensor measurements. We employ a multi-threaded architecture with a Bounding Volume Hierarchy (BVH) for efficient data storage and fast search operations, enabling real-time performance. Experimental results show that our method achieves reconstruction accuracy on par with, or exceeding, state-of-the-art techniques-including truncated signed distance functions, occupancy mapping, and voxel-based meshing-while producing smaller output file sizes (10 times smaller than raw input and more than 5 times smaller than mesh-based methods) and maintaining real-time performance (around 2 Hz for a 64-beam sensor).
>
---
#### [new 008] RoboHiMan: A Hierarchical Evaluation Paradigm for Compositional Generalization in Long-Horizon Manipulation
- **分类: cs.RO**

- **简介: 该论文聚焦机器人长视野操作中的组合泛化问题，提出RoboHiMan评估范式，包含新基准HiMan-Bench和三种评估模式，系统分析分层架构在技能组合与抗扰能力上的瓶颈，推动面向真实场景的机器人学习模型发展。**

- **链接: [http://arxiv.org/pdf/2510.13149v1](http://arxiv.org/pdf/2510.13149v1)**

> **作者:** Yangtao Chen; Zixuan Chen; Nga Teng Chan; Junting Chen; Junhui Yin; Jieqi Shi; Yang Gao; Yong-Lu Li; Jing Huo
>
> **备注:** Under review. These first two authors contributed equally to this work
>
> **摘要:** Enabling robots to flexibly schedule and compose learned skills for novel long-horizon manipulation under diverse perturbations remains a core challenge. Early explorations with end-to-end VLA models show limited success, as these models struggle to generalize beyond the training distribution. Hierarchical approaches, where high-level planners generate subgoals for low-level policies, bring certain improvements but still suffer under complex perturbations, revealing limited capability in skill composition. However, existing benchmarks primarily emphasize task completion in long-horizon settings, offering little insight into compositional generalization, robustness, and the interplay between planning and execution. To systematically investigate these gaps, we propose RoboHiMan, a hierarchical evaluation paradigm for compositional generalization in long-horizon manipulation. RoboHiMan introduces HiMan-Bench, a benchmark of atomic and compositional tasks under diverse perturbations, supported by a multi-level training dataset for analyzing progressive data scaling, and proposes three evaluation paradigms (vanilla, decoupled, coupled) that probe the necessity of skill composition and reveal bottlenecks in hierarchical architectures. Experiments highlight clear capability gaps across representative models and architectures, pointing to directions for advancing models better suited to real-world long-horizon manipulation tasks. Videos and open-source code can be found on our project website: https://chenyt31.github.io/robo-himan.github.io/.
>
---
#### [new 009] Active Tactile Exploration for Rigid Body Pose and Shape Estimation
- **分类: cs.RO**

- **简介: 该论文研究机器人通过主动触觉探索，估计未知刚体的位姿与形状。针对触觉数据稀疏及接触导致物体移动的问题，提出一种学习框架，结合物理约束损失函数与信息增益驱动探索，仅用少量触觉数据即可高效实现形状与位姿联合估计。**

- **链接: [http://arxiv.org/pdf/2510.13595v1](http://arxiv.org/pdf/2510.13595v1)**

> **作者:** Ethan K. Gordon; Bruke Baraki; Hien Bui; Michael Posa
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** General robot manipulation requires the handling of previously unseen objects. Learning a physically accurate model at test time can provide significant benefits in data efficiency, predictability, and reuse between tasks. Tactile sensing can compliment vision with its robustness to occlusion, but its temporal sparsity necessitates careful online exploration to maintain data efficiency. Direct contact can also cause an unrestrained object to move, requiring both shape and location estimation. In this work, we propose a learning and exploration framework that uses only tactile data to simultaneously determine the shape and location of rigid objects with minimal robot motion. We build on recent advances in contact-rich system identification to formulate a loss function that penalizes physical constraint violation without introducing the numerical stiffness inherent in rigid-body contact. Optimizing this loss, we can learn cuboid and convex polyhedral geometries with less than 10s of randomly collected data after first contact. Our exploration scheme seeks to maximize Expected Information Gain and results in significantly faster learning in both simulated and real-robot experiments. More information can be found at https://dairlab.github.io/activetactile
>
---
#### [new 010] Geometric Model Predictive Path Integral for Agile UAV Control with Online Collision Avoidance
- **分类: cs.RO**

- **简介: 该论文研究无人机敏捷控制与实时避障任务，提出几何模型预测路径积分（GMPPI）方法，结合SE(3)几何控制与在线深度感知，提升轨迹跟踪精度和避障能力，实现在复杂环境中高速飞行。**

- **链接: [http://arxiv.org/pdf/2510.12924v1](http://arxiv.org/pdf/2510.12924v1)**

> **作者:** Pavel Pochobradský; Ondřej Procházka; Robert Pěnička; Vojtěch Vonásek; Martin Saska
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** In this letter, we introduce Geometric Model Predictive Path Integral (GMPPI), a sampling-based controller capable of tracking agile trajectories while avoiding obstacles. In each iteration, GMPPI generates a large number of candidate rollout trajectories and then averages them to create a nominal control to be followed by the Unmanned Aerial Vehicle (UAV). We propose using geometric SE(3) control to generate part of the rollout trajectories, significantly increasing precision in agile flight. Furthermore, we introduce varying rollout simulation time step length and dynamic cost and noise parameters, vastly improving tracking performance of smooth and low-speed trajectories over an existing Model Predictive Path Integral (MPPI) implementation. Finally, we propose an integration of GMPPI with a stereo depth camera, enabling online obstacle avoidance at high speeds, a crucial step towards autonomous UAV flights in complex environments. The proposed controller can track simulated agile reference trajectories with position error similar to the geometric SE(3) controller. However, the same configuration of the proposed controller can avoid obstacles in a simulated forest environment at speeds of up to 13m/s, surpassing the performance of a state-of-the-art obstacle-aware planner. In real-world experiments, GMPPI retains the capability to track agile trajectories and avoids obstacles at speeds of up to 10m/s.
>
---
#### [new 011] On Your Own: Pro-level Autonomous Drone Racing in Uninstrumented Arenas
- **分类: cs.RO**

- **简介: 该论文研究自主无人机竞速任务，旨在解决现有系统依赖受控环境的问题。作者提出可在无外部感知设备的复杂环境中运行的自主飞行方法，并证明其性能媲美职业飞手，同时公开了真实飞行数据。**

- **链接: [http://arxiv.org/pdf/2510.13644v1](http://arxiv.org/pdf/2510.13644v1)**

> **作者:** Michael Bosello; Flavio Pinzarrone; Sara Kiade; Davide Aguiari; Yvo Keuter; Aaesha AlShehhi; Gyordan Caminati; Kei Long Wong; Ka Seng Chou; Junaid Halepota; Fares Alneyadi; Jacopo Panerati; Giovanni Pau
>
> **摘要:** Drone technology is proliferating in many industries, including agriculture, logistics, defense, infrastructure, and environmental monitoring. Vision-based autonomy is one of its key enablers, particularly for real-world applications. This is essential for operating in novel, unstructured environments where traditional navigation methods may be unavailable. Autonomous drone racing has become the de facto benchmark for such systems. State-of-the-art research has shown that autonomous systems can surpass human-level performance in racing arenas. However, direct applicability to commercial and field operations is still limited as current systems are often trained and evaluated in highly controlled environments. In our contribution, the system's capabilities are analyzed within a controlled environment -- where external tracking is available for ground-truth comparison -- but also demonstrated in a challenging, uninstrumented environment -- where ground-truth measurements were never available. We show that our approach can match the performance of professional human pilots in both scenarios. We also publicly release the data from the flights carried out by our approach and a world-class human pilot.
>
---
#### [new 012] LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在机器人操作中鲁棒性不足的问题，系统分析了七类扰动下的模型表现。研究发现，尽管基准成绩高，但模型对视角、初始状态等敏感，且常忽略语言指令，揭示了当前评估方式的局限性。**

- **链接: [http://arxiv.org/pdf/2510.13626v1](http://arxiv.org/pdf/2510.13626v1)**

> **作者:** Senyu Fei; Siyin Wang; Junhao Shi; Zihao Dai; Jikun Cai; Pengfang Qian; Li Ji; Xinzhe He; Shiduo Zhang; Zhaoye Fei; Jinlan Fu; Jingjing Gong; Xipeng Qiu
>
> **摘要:** Visual-Language-Action (VLA) models report impressive success rates on robotic manipulation benchmarks, yet these results may mask fundamental weaknesses in robustness. We perform a systematic vulnerability analysis by introducing controlled perturbations across seven dimensions: objects layout, camera viewpoints, robot initial states, language instructions, light conditions, background textures and sensor noise. We comprehensively analyzed multiple state-of-the-art models and revealed consistent brittleness beneath apparent competence. Our analysis exposes critical weaknesses: models exhibit extreme sensitivity to perturbation factors, including camera viewpoints and robot initial states, with performance dropping from 95% to below 30% under modest perturbations. Surprisingly, models are largely insensitive to language variations, with further experiments revealing that models tend to ignore language instructions completely. Our findings challenge the assumption that high benchmark scores equate to true competency and highlight the need for evaluation practices that assess reliability under realistic variation.
>
---
#### [new 013] DAMM-LOAM: Degeneracy Aware Multi-Metric LiDAR Odometry and Mapping
- **分类: cs.RO**

- **简介: 该论文属LiDAR SLAM任务，旨在解决稀疏特征、重复结构等导致的位姿估计退化问题。提出DAMM-LOAM，通过点云分类与退化感知加权ICP提升建图与定位精度，并结合Scan Context实现鲁棒闭环检测。**

- **链接: [http://arxiv.org/pdf/2510.13287v1](http://arxiv.org/pdf/2510.13287v1)**

> **作者:** Nishant Chandna; Akshat Kaushal
>
> **备注:** Accepted at IROS Active Perception Workshop
>
> **摘要:** LiDAR Simultaneous Localization and Mapping (SLAM) systems are essential for enabling precise navigation and environmental reconstruction across various applications. Although current point-to-plane ICP algorithms perform effec- tively in structured, feature-rich environments, they struggle in scenarios with sparse features, repetitive geometric structures, and high-frequency motion. This leads to degeneracy in 6- DOF pose estimation. Most state-of-the-art algorithms address these challenges by incorporating additional sensing modalities, but LiDAR-only solutions continue to face limitations under such conditions. To address these issues, we propose a novel Degeneracy-Aware Multi-Metric LiDAR Odometry and Map- ping (DAMM-LOAM) module. Our system improves mapping accuracy through point cloud classification based on surface normals and neighborhood analysis. Points are classified into ground, walls, roof, edges, and non-planar points, enabling accurate correspondences. A Degeneracy-based weighted least squares-based ICP algorithm is then applied for accurate odom- etry estimation. Additionally, a Scan Context based back-end is implemented to support robust loop closures. DAMM-LOAM demonstrates significant improvements in odometry accuracy, especially in indoor environments such as long corridors
>
---
#### [new 014] Learning to Grasp Anything by Playing with Random Toys
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究机器人抓取的泛化能力，提出通过在由四种基本形状组成的随机“玩具”上训练，使机器人零样本迁移至真实物体。其核心是基于检测池化的对象中心表征，实现了在YCB数据集上67%的成功率，超越依赖更多真实数据的现有方法。**

- **链接: [http://arxiv.org/pdf/2510.12866v1](http://arxiv.org/pdf/2510.12866v1)**

> **作者:** Dantong Niu; Yuvan Sharma; Baifeng Shi; Rachel Ding; Matteo Gioia; Haoru Xue; Henry Tsai; Konstantinos Kallidromitis; Anirudh Pai; Shankar Shastry; Trevor Darrell; Jitendra Malik; Roei Herzig
>
> **摘要:** Robotic manipulation policies often struggle to generalize to novel objects, limiting their real-world utility. In contrast, cognitive science suggests that children develop generalizable dexterous manipulation skills by mastering a small set of simple toys and then applying that knowledge to more complex items. Inspired by this, we study if similar generalization capabilities can also be achieved by robots. Our results indicate robots can learn generalizable grasping using randomly assembled objects that are composed from just four shape primitives: spheres, cuboids, cylinders, and rings. We show that training on these "toys" enables robust generalization to real-world objects, yielding strong zero-shot performance. Crucially, we find the key to this generalization is an object-centric visual representation induced by our proposed detection pooling mechanism. Evaluated in both simulation and on physical robots, our model achieves a 67% real-world grasping success rate on the YCB dataset, outperforming state-of-the-art approaches that rely on substantially more in-domain data. We further study how zero-shot generalization performance scales by varying the number and diversity of training toys and the demonstrations per toy. We believe this work offers a promising path to scalable and generalizable learning in robotic manipulation. Demonstration videos, code, checkpoints and our dataset are available on our project page: https://lego-grasp.github.io/ .
>
---
#### [new 015] Hoecken-D Hand: A Novel Robotic Hand for Linear Parallel Pinching and Self-Adaptive Grasping
- **分类: cs.RO**

- **简介: 该论文提出一种新型欠驱动机械手Hoecken-D Hand，旨在实现线性平行夹持与自适应包络抓取。通过改进Hoecken连杆与差动弹簧机构，单驱动即可完成两种模式切换。工作包括设计、建模、仿真与3D打印验证，适用于非结构化环境中的灵巧抓取。**

- **链接: [http://arxiv.org/pdf/2510.13553v1](http://arxiv.org/pdf/2510.13553v1)**

> **作者:** Wentao Guo; Wenzeng Zhang
>
> **备注:** Accepted by IEEE International Conference on Robotics and Biomimetics (IROS) 2025, Hangzhou, China. This version includes updated contact information
>
> **摘要:** This paper presents the Hoecken-D Hand, an underactuated robotic gripper that combines a modified Hoecken linkage with a differential spring mechanism to achieve both linear parallel pinching and a mid-stroke transition to adaptive envelope. The original Hoecken linkage is reconfigured by replacing one member with differential links, preserving straight-line guidance while enabling contact-triggered reconfiguration without additional actuators. A double-parallelogram arrangement maintains fingertip parallelism during conventional pinching, whereas the differential mechanism allows one finger to wrap inward upon encountering an obstacle, improving stability on irregular or thin objects. The mechanism can be driven by a single linear actuator, minimizing complexity and cost; in our prototype, each finger is driven by its own linear actuator for simplicity. We perform kinematic modeling and force analysis to characterize grasp performance, including simulated grasping forces and spring-opening behavior under varying geometric parameters. The design was prototyped using PLA-based 3D printing, achieving a linear pinching span of approximately 200 mm. Preliminary tests demonstrate reliable grasping in both modes across a wide range of object geometries, highlighting the Hoecken-D Hand as a compact, adaptable, and cost-effective solution for manipulation in unstructured environments.
>
---
#### [new 016] Gaussian Process Implicit Surfaces as Control Barrier Functions for Safe Robot Navigation
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究安全机器人导航，提出将高斯过程隐式曲面（GPIS）用作控制屏障函数（CBF），通过传感器数据构建安全边界，结合不确定性提供鲁棒性，并设计稀疏方法提升计算效率，实现碰撞规避。**

- **链接: [http://arxiv.org/pdf/2510.12919v1](http://arxiv.org/pdf/2510.12919v1)**

> **作者:** Mouhyemen Khan; Tatsuya Ibuki; Abhijit Chatterjee
>
> **备注:** 8 pages, 7 figures, under review
>
> **摘要:** Level set methods underpin modern safety techniques such as control barrier functions (CBFs), while also serving as implicit surface representations for geometric shapes via distance fields. Inspired by these two paradigms, we propose a unified framework where the implicit surface itself acts as a CBF. We leverage Gaussian process (GP) implicit surface (GPIS) to represent the safety boundaries, using safety samples which are derived from sensor measurements to condition the GP. The GP posterior mean defines the implicit safety surface (safety belief), while the posterior variance provides a robust safety margin. Although GPs have favorable properties such as uncertainty estimation and analytical tractability, they scale cubically with data. To alleviate this issue, we develop a sparse solution called sparse Gaussian CBFs. To the best of our knowledge, GPIS have not been explicitly used to synthesize CBFs. We validate the approach on collision avoidance tasks in two settings: a simulated 7-DOF manipulator operating around the Stanford bunny, and a quadrotor navigating in 3D around a physical chair. In both cases, Gaussian CBFs (with and without sparsity) enable safe interaction and collision-free execution of trajectories that would otherwise intersect the objects.
>
---
#### [new 017] UNCAP: Uncertainty-Guided Planning Using Natural Language Communication for Cooperative Autonomous Vehicles
- **分类: cs.RO; cs.CL; cs.CV; cs.MA**

- **简介: 该论文研究多车协同规划任务，解决通信效率与安全性问题。提出UNCAP方法，利用自然语言传递带不确定性感知的信息，通过选择性融合高价值消息，降低带宽消耗，提升决策安全性和可靠性。**

- **链接: [http://arxiv.org/pdf/2510.12992v1](http://arxiv.org/pdf/2510.12992v1)**

> **作者:** Neel P. Bhatt; Po-han Li; Kushagra Gupta; Rohan Siva; Daniel Milan; Alexander T. Hogue; Sandeep P. Chinchali; David Fridovich-Keil; Zhangyang Wang; Ufuk Topcu
>
> **摘要:** Safe large-scale coordination of multiple cooperative connected autonomous vehicles (CAVs) hinges on communication that is both efficient and interpretable. Existing approaches either rely on transmitting high-bandwidth raw sensor data streams or neglect perception and planning uncertainties inherent in shared data, resulting in systems that are neither scalable nor safe. To address these limitations, we propose Uncertainty-Guided Natural Language Cooperative Autonomous Planning (UNCAP), a vision-language model-based planning approach that enables CAVs to communicate via lightweight natural language messages while explicitly accounting for perception uncertainty in decision-making. UNCAP features a two-stage communication protocol: (i) an ego CAV first identifies the subset of vehicles most relevant for information exchange, and (ii) the selected CAVs then transmit messages that quantitatively express their perception uncertainty. By selectively fusing messages that maximize mutual information, this strategy allows the ego vehicle to integrate only the most relevant signals into its decision-making, improving both the scalability and reliability of cooperative planning. Experiments across diverse driving scenarios show a 63% reduction in communication bandwidth with a 31% increase in driving safety score, a 61% reduction in decision uncertainty, and a four-fold increase in collision distance margin during near-miss events. Project website: https://uncap-project.github.io/
>
---
#### [new 018] Development of an Intuitive GUI for Non-Expert Teleoperation of Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文致力于开发面向非专家用户的直观图形界面（GUI），解决人形机器人遥操作中界面复杂、难以上手的问题。通过结合用户界面设计与人机交互理论，构建可扩展、易用的控制系统，使非专业人员能顺利完成FIRA障碍任务。**

- **链接: [http://arxiv.org/pdf/2510.13594v1](http://arxiv.org/pdf/2510.13594v1)**

> **作者:** Austin Barret; Meng Cheng Lau
>
> **备注:** 9 Figure. Presented at FIRA Summit 2025, Daegu, S. Korea
>
> **摘要:** The operation of humanoid robotics is an essential field of research with many practical and competitive applications. Many of these systems, however, do not invest heavily in developing a non-expert-centered graphical user interface (GUI) for operation. The focus of this research is to develop a scalable GUI that is tailored to be simple and intuitive so non-expert operators can control the robot through a FIRA-regulated obstacle course. Using common practices from user interface development (UI) and understanding concepts described in human-robot interaction (HRI) and other related concepts, we will develop a new interface with the goal of a non-expert teleoperation system.
>
---
#### [new 019] Efficient Force and Stiffness Prediction in Robotic Produce Handling with a Piezoresistive Pressure Sensor
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究机器人采摘中易损农产品的柔顺抓取问题，提出一种低成本可穿戴压阻压力传感器，集成于刚性和软体夹爪，结合快速稳态估计算法，实现对未知物的力与刚度实时感知，用于抓取反馈、成熟度判断与品质分选。**

- **链接: [http://arxiv.org/pdf/2510.13616v1](http://arxiv.org/pdf/2510.13616v1)**

> **作者:** Preston Fairchild; Claudia Chen; Xiaobo Tan
>
> **备注:** For supplementary videos, see https://drive.google.com/drive/folders/1jol-_z6gaUfjpL1Qi7EG420usTbVSodv?usp=sharing
>
> **摘要:** Properly handling delicate produce with robotic manipulators is a major part of the future role of automation in agricultural harvesting and processing. Grasping with the correct amount of force is crucial in not only ensuring proper grip on the object, but also to avoid damaging or bruising the product. In this work, a flexible pressure sensor that is both low cost and easy to fabricate is integrated with robotic grippers for working with produce of varying shapes, sizes, and stiffnesses. The sensor is successfully integrated with both a rigid robotic gripper, as well as a pneumatically actuated soft finger. Furthermore, an algorithm is proposed for accelerated estimation of the steady-state value of the sensor output based on the transient response data, to enable real-time applications. The sensor is shown to be effective in incorporating feedback to correctly grasp objects of unknown sizes and stiffnesses. At the same time, the sensor provides estimates for these values which can be utilized for identification of qualities such as ripeness levels and bruising. It is also shown to be able to provide force feedback for objects of variable stiffnesses. This enables future use not only for produce identification, but also for tasks such as quality control and selective distribution based on ripeness levels.
>
---
#### [new 020] VLA-0: Building State-of-the-Art VLAs with Zero Modification
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究视觉-语言-动作（VLA）模型，旨在简化机器人操控模型构建。作者提出VLA-0，将动作直接表示为文本，无需修改视觉语言模型结构。实验表明，该方法在多个基准上超越复杂模型，验证了其高效性与实用性。**

- **链接: [http://arxiv.org/pdf/2510.13054v1](http://arxiv.org/pdf/2510.13054v1)**

> **作者:** Ankit Goyal; Hugo Hadfield; Xuning Yang; Valts Blukis; Fabio Ramos
>
> **摘要:** Vision-Language-Action models (VLAs) hold immense promise for enabling generalist robot manipulation. However, the best way to build them remains an open question. Current approaches often add complexity, such as modifying the existing vocabulary of a Vision-Language Model (VLM) with action tokens or introducing special action heads. Curiously, the simplest strategy of representing actions directly as text has remained largely unexplored. This work introduces VLA-0 to investigate this idea. We find that VLA-0 is not only effective; it is surprisingly powerful. With the right design, VLA-0 outperforms more involved models. On LIBERO, a popular benchmark for evaluating VLAs, VLA-0 outperforms all existing methods trained on the same robotic data, including $\pi_0.5$-KI, OpenVLA-OFT and SmolVLA. Furthermore, without large-scale robotics-specific training, it outperforms methods trained on large-scale robotic data, like $\pi_0.5$-KI, $\pi_0$, GR00T-N1 and MolmoAct. These findings also translate to the real world, where VLA-0 outperforms SmolVLA, a VLA model pre-trained on large-scale real data. This paper summarizes our unexpected findings and spells out the specific techniques required to unlock the high performance of this simple yet potent VLA design. Visual results, code, and trained models are provided here: https://vla0.github.io/.
>
---
#### [new 021] Hierarchical Discrete Lattice Assembly: An Approach for the Digital Fabrication of Scalable Macroscale Structures
- **分类: cs.RO**

- **简介: 该论文提出一种基于分层离散晶格组装的大型结构数字制造方法，旨在解决大尺度制造复杂、昂贵的问题。通过小尺度复杂构件的分组装配与移动机器人协同，结合数字孪生系统实现米级结构的高效、可扩展建造。**

- **链接: [http://arxiv.org/pdf/2510.13686v1](http://arxiv.org/pdf/2510.13686v1)**

> **作者:** Miana Smith; Paul Arthur Richard; Alexander Htet Kyaw; Neil Gershenfeld
>
> **备注:** In ACM Symposium on Computational Fabrication (SCF '25), November 20-21, 2025, Cambridge, MA, USA. ACM, New York, NY, USA, 15 pages
>
> **摘要:** Although digital fabrication processes at the desktop scale have become proficient and prolific, systems aimed at producing larger-scale structures are still typically complex, expensive, and unreliable. In this work, we present an approach for the fabrication of scalable macroscale structures using simple robots and interlocking lattice building blocks. A target structure is first voxelized so that it can be populated with an architected lattice. These voxels are then grouped into larger interconnected blocks, which are produced using standard digital fabrication processes, leveraging their capability to produce highly complex geometries at a small scale. These blocks, on the size scale of tens of centimeters, are then fed to mobile relative robots that are able to traverse over the structure and place new blocks to form structures on the meter scale. To facilitate the assembly of large structures, we introduce a live digital twin simulation tool for controlling and coordinating assembly robots that enables both global planning for a target structure and live user design, interaction, or intervention. To improve assembly throughput, we introduce a new modular assembly robot, designed for hierarchical voxel handling. We validate this system by demonstrating the voxelization, hierarchical blocking, path planning, and robotic fabrication of a set of meter-scale objects.
>
---
#### [new 022] MODUR: A Modular Dual-reconfigurable Robot
- **分类: cs.RO**

- **简介: 该论文提出一种新型模块化自重构机器人MODUR，旨在提升机器人的适应性与鲁棒性。通过设计具双层重构能力的模块，实现模块间拓扑重组与单个模块形态变化，结合紧凑连接器与剪叉机构，完成运动解耦与位移迁移，并验证了其运动能力。**

- **链接: [http://arxiv.org/pdf/2510.13356v1](http://arxiv.org/pdf/2510.13356v1)**

> **作者:** Jie Gu; Tin Lun Lam; Chunxu Tian; Zhihao Xia; Yongheng Xing; Dan Zhang
>
> **摘要:** Modular Self-Reconfigurable Robot (MSRR) systems are a class of robots capable of forming higher-level robotic systems by altering the topological relationships between modules, offering enhanced adaptability and robustness in various environments. This paper presents a novel MSRR called MODUR, featuring dual-level reconfiguration capabilities designed to integrate reconfigurable mechanisms into MSRR. Specifically, MODUR can perform high-level self-reconfiguration among modules to create different configurations, while each module is also able to change its shape to execute basic motions. The design of MODUR primarily includes a compact connector and scissor linkage groups that provide actuation, forming a parallel mechanism capable of achieving both connector motion decoupling and adjacent position migration capabilities. Furthermore, the workspace, considering the interdependent connectors, is comprehensively analyzed, laying a theoretical foundation for the design of the module's basic motion. Finally, the motion of MODUR is validated through a series of experiments.
>
---
#### [new 023] A Modular Object Detection System for Humanoid Robots Using YOLO
- **分类: cs.RO**

- **简介: 该论文针对人形机器人视觉系统效率低的问题，提出基于YOLOv9的模块化目标检测方法。在ROS1中构建虚拟环境实现兼容，应用于FIRA Hurocup场景，实验表明其精度相当但计算成本较高，鲁棒性更优。**

- **链接: [http://arxiv.org/pdf/2510.13625v1](http://arxiv.org/pdf/2510.13625v1)**

> **作者:** Nicolas Pottier; Meng Cheng Lau
>
> **备注:** 7 Figures, 5 tables. This article was presented at FIRA Summit 2025. It will be updated for journal submission
>
> **摘要:** Within the field of robotics, computer vision remains a significant barrier to progress, with many tasks hindered by inefficient vision systems. This research proposes a generalized vision module leveraging YOLOv9, a state-of-the-art framework optimized for computationally constrained environments like robots. The model is trained on a dataset tailored to the FIRA robotics Hurocup. A new vision module is implemented in ROS1 using a virtual environment to enable YOLO compatibility. Performance is evaluated using metrics such as frames per second (FPS) and Mean Average Precision (mAP). Performance is then compared to the existing geometric framework in static and dynamic contexts. The YOLO model achieved comparable precision at a higher computational cost then the geometric model, while providing improved robustness.
>
---
#### [new 024] Development of a Linear Guide-Rail Testbed for Physically Emulating ISAM Operations
- **分类: cs.RO**

- **简介: 该论文设计并开发了一种线性导轨硬件在环测试平台，用于模拟在轨服务、组装与制造（ISAM）操作。旨在解决自由漂浮卫星上机械臂运动控制难以实验验证的问题，通过6-DOF机械臂与1-DOF导轨系统实现空间运动特性模拟，支持动力学模型与接触力学的实验研究。**

- **链接: [http://arxiv.org/pdf/2510.13005v1](http://arxiv.org/pdf/2510.13005v1)**

> **作者:** Robert Muldrow; Channing Ludden; Christopher Petersen
>
> **备注:** 12 pages, 4 figures, AAS/AIAA Space Flight Mechanics
>
> **摘要:** In-Space Servicing, Assembly, and Manufacturing (ISAM) is a set of emerging operations that provides several benefits to improve the longevity, capacity, mo- bility, and expandability of existing and future space assets. Serial robotic ma- nipulators are particularly vital in accomplishing ISAM operations, however, the complex perturbation forces and motions associated with movement of a robotic arm on a free-flying satellite presents a complex controls problem requiring addi- tional study. While many dynamical models are developed, experimentally test- ing and validating these models is challenging given that the models operate in space, where satellites have six-degrees-of-freedom (6-DOF). This paper attempts to resolve those challenges by presenting the design and development of a new hardware-in-the-loop (HIL) experimental testbed utilized to emulate ISAM. This emulation will be accomplished by means of a 6-DOF UR3e robotic arm attached to a satellite bus. This satellite bus is mounted to a 1-DOF guide-rail system, en- abling the satellite bus and robotic arm to move freely in one linear direction. This experimental ISAM emulation system will explore and validate models for space motion, serial robot manipulation, and contact mechanics.
>
---
#### [new 025] Actron3D: Learning Actionable Neural Functions from Videos for Transferable Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出Actron3D，旨在从少量单目人类操作视频中学习可迁移的6自由度机器人操控技能。其核心是神经可供性函数，融合几何、外观与功能线索，实现跨任务技能迁移，在仿真与真实场景中均显著提升成功率。**

- **链接: [http://arxiv.org/pdf/2510.12971v1](http://arxiv.org/pdf/2510.12971v1)**

> **作者:** Anran Zhang; Hanzhi Chen; Yannick Burkhardt; Yao Zhong; Johannes Betz; Helen Oleynikova; Stefan Leutenegger
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** We present Actron3D, a framework that enables robots to acquire transferable 6-DoF manipulation skills from just a few monocular, uncalibrated, RGB-only human videos. At its core lies the Neural Affordance Function, a compact object-centric representation that distills actionable cues from diverse uncalibrated videos-geometry, visual appearance, and affordance-into a lightweight neural network, forming a memory bank of manipulation skills. During deployment, we adopt a pipeline that retrieves relevant affordance functions and transfers precise 6-DoF manipulation policies via coarse-to-fine optimization, enabled by continuous queries to the multimodal features encoded in the neural functions. Experiments in both simulation and the real world demonstrate that Actron3D significantly outperforms prior methods, achieving a 14.9 percentage point improvement in average success rate across 13 tasks while requiring only 2-3 demonstration videos per task.
>
---
#### [new 026] Kinematic Kitbashing for Modeling Functional Articulated Objects
- **分类: cs.RO; cs.GR**

- **简介: 该论文提出“运动学套件拼接”框架，旨在通过重用现有模型部件自动生成可动功能物体。解决部件组装中运动连贯性与功能目标满足的问题，结合运动感知的几何匹配与功能驱动优化，实现高质量功能性铰接物体建模。**

- **链接: [http://arxiv.org/pdf/2510.13048v1](http://arxiv.org/pdf/2510.13048v1)**

> **作者:** Minghao Guo; Victor Zordan; Sheldon Andrews; Wojciech Matusik; Maneesh Agrawala; Hsueh-Ti Derek Liu
>
> **摘要:** We introduce Kinematic Kitbashing, an automatic framework that synthesizes functionality-aware articulated objects by reusing parts from existing models. Given a kinematic graph with a small collection of articulated parts, our optimizer jointly solves for the spatial placement of every part so that (i) attachments remain geometrically sound over the entire range of motion and (ii) the assembled object satisfies user-specified functional goals such as collision-free actuation, reachability, or trajectory following. At its core is a kinematics-aware attachment energy that aligns vector distance function features sampled across multiple articulation snapshots. We embed this attachment term within an annealed Riemannian Langevin dynamics sampler that treats functionality objectives as additional energies, enabling robust global exploration while accommodating non-differentiable functionality objectives and constraints. Our framework produces a wide spectrum of assembled articulated shapes, from trash-can wheels grafted onto car bodies to multi-segment lamps, gear-driven paddlers, and reconfigurable furniture, and delivers strong quantitative improvements over state-of-the-art baselines across geometric, kinematic, and functional metrics. By tightly coupling articulation-aware geometry matching with functionality-driven optimization, Kinematic Kitbashing bridges part-based shape modeling and functional assembly design, empowering rapid creation of interactive articulated assets.
>
---
#### [new 027] Tactile-Conditioned Diffusion Policy for Force-Aware Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文研究力感知机器人操作任务，解决传统模仿学习中触觉反馈利用不足、抓握力控制不精确的问题。提出FARM框架，结合高维触觉数据与力感知动作空间，通过扩散策略实现对抓取姿态、宽度和力度的联合预测，提升复杂操作中的力控制性能。**

- **链接: [http://arxiv.org/pdf/2510.13324v1](http://arxiv.org/pdf/2510.13324v1)**

> **作者:** Erik Helmut; Niklas Funk; Tim Schneider; Cristiana de Farias; Jan Peters
>
> **摘要:** Contact-rich manipulation depends on applying the correct grasp forces throughout the manipulation task, especially when handling fragile or deformable objects. Most existing imitation learning approaches often treat visuotactile feedback only as an additional observation, leaving applied forces as an uncontrolled consequence of gripper commands. In this work, we present Force-Aware Robotic Manipulation (FARM), an imitation learning framework that integrates high-dimensional tactile data to infer tactile-conditioned force signals, which in turn define a matching force-based action space. We collect human demonstrations using a modified version of the handheld Universal Manipulation Interface (UMI) gripper that integrates a GelSight Mini visual tactile sensor. For deploying the learned policies, we developed an actuated variant of the UMI gripper with geometry matching our handheld version. During policy rollouts, the proposed FARM diffusion policy jointly predicts robot pose, grip width, and grip force. FARM outperforms several baselines across three tasks with distinct force requirements -- high-force, low-force, and dynamic force adaptation -- demonstrating the advantages of its two key components: leveraging force-grounded, high-dimensional tactile observations and a force-based control space. The codebase and design files are open-sourced and available at https://tactile-farm.github.io .
>
---
#### [new 028] ALOHA2 Robot Kitchen Application Scenario Reproduction Report
- **分类: cs.RO**

- **简介: 该论文介绍ALOHA2机器人厨房应用场景的复现。属于机器人系统设计任务，旨在提升双臂遥操作机器人的性能与人体工学设计。工作包括构建高鲁棒性、多视角视觉反馈的机器人平台，并实现主从控制与数据采集。**

- **链接: [http://arxiv.org/pdf/2510.13284v1](http://arxiv.org/pdf/2510.13284v1)**

> **作者:** Haoyang Wu; Siheng Wu; William X. Liu; Fangui Zeng
>
> **摘要:** ALOHA2 is an enhanced version of the dual-arm teleoperated robot ALOHA, featuring higher performance and robustness compared to the original design, while also being more ergonomic. Like ALOHA, ALOHA2 consists of two grippers and two ViperX 6-DoF arms, as well as two smaller WidowX arms. Users control the follower mechanical arms by operating the leader mechanical arms through back-driving. The device also includes cameras that generate images from multiple viewpoints, allowing for RGB data collection during teleoperation. The robot is mounted on a 48-inch x 30-inch table, equipped with an aluminum frame that provides additional mounting points for cameras and gravity compensation systems.
>
---
#### [new 029] Enhancing Sampling-based Planning with a Library of Paths
- **分类: cs.RO**

- **简介: 该论文针对3D物体在狭窄通道中的路径规划难题，提出利用历史路径库复用经验。通过检索相似物体的已有路径并调整使用，指导采样过程，显著提升采样效率，在复杂场景中实现更快求解，尤其在传统方法失败时仍可成功。**

- **链接: [http://arxiv.org/pdf/2510.12962v1](http://arxiv.org/pdf/2510.12962v1)**

> **作者:** Michal Minařík; Vojtěch Vonásek; Robert Pěnička
>
> **摘要:** Path planning for 3D solid objects is a challenging problem, requiring a search in a six-dimensional configuration space, which is, nevertheless, essential in many robotic applications such as bin-picking and assembly. The commonly used sampling-based planners, such as Rapidly-exploring Random Trees, struggle with narrow passages where the sampling probability is low, increasing the time needed to find a solution. In scenarios like robotic bin-picking, various objects must be transported through the same environment. However, traditional planners start from scratch each time, losing valuable information gained during the planning process. We address this by using a library of past solutions, allowing the reuse of previous experiences even when planning for a new, previously unseen object. Paths for a set of objects are stored, and when planning for a new object, we find the most similar one in the library and use its paths as approximate solutions, adjusting for possible mutual transformations. The configuration space is then sampled along the approximate paths. Our method is tested in various narrow passage scenarios and compared with state-of-the-art methods from the OMPL library. Results show significant speed improvements (up to 85% decrease in the required time) of our method, often finding a solution in cases where the other planners fail. Our implementation of the proposed method is released as an open-source package.
>
---
#### [new 030] A Novel Robot Hand with Hoeckens Linkages and Soft Phalanges for Scooping and Self-Adaptive Grasping in Environmental Constraints
- **分类: cs.RO**

- **简介: 该论文提出一种新型欠驱动机器人手Hockens-A Hand，旨在解决复杂环境中自适应抓取问题。通过集成Hoeckens机构、双平行四边形和四杆联动结构，实现三种抓取模式，结合软性指节提升包裹能力，并经仿真与实验证明其有效性。**

- **链接: [http://arxiv.org/pdf/2510.13535v1](http://arxiv.org/pdf/2510.13535v1)**

> **作者:** Wentao Guo; Yizhou Wang; Wenzeng Zhang
>
> **备注:** Accepted by IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025, Hangzhou. This version includes updated contact information
>
> **摘要:** This paper presents a novel underactuated adaptive robotic hand, Hockens-A Hand, which integrates the Hoeckens mechanism, a double-parallelogram linkage, and a specialized four-bar linkage to achieve three adaptive grasping modes: parallel pinching, asymmetric scooping, and enveloping grasping. Hockens-A Hand requires only a single linear actuator, leveraging passive mechanical intelligence to ensure adaptability and compliance in unstructured environments. Specifically, the vertical motion of the Hoeckens mechanism introduces compliance, the double-parallelogram linkage ensures line contact at the fingertip, and the four-bar amplification system enables natural transitions between different grasping modes. Additionally, the inclusion of a mesh-textured silicone phalanx further enhances the ability to envelop objects of various shapes and sizes. This study employs detailed kinematic analysis to optimize the push angle and design the linkage lengths for optimal performance. Simulations validated the design by analyzing the fingertip motion and ensuring smooth transitions between grasping modes. Furthermore, the grasping force was analyzed using power equations to enhance the understanding of the system's performance.Experimental validation using a 3D-printed prototype demonstrates the three grasping modes of the hand in various scenarios under environmental constraints, verifying its grasping stability and broad applicability.
>
---
#### [new 031] Comparison of Forced and Unforced Rendezvous, Proximity Operations, and Docking Under Model Mismatch
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文研究交会对接中强制与非强制运动的燃料消耗差异，解决模型失配对轨迹预测的影响问题。通过比较CW模型与高保真模型下自然与强制绕飞任务的控制脉冲需求，表明非强制运动未必更省燃料，为高效轨道操作提供依据。**

- **链接: [http://arxiv.org/pdf/2510.13004v1](http://arxiv.org/pdf/2510.13004v1)**

> **作者:** Robert Muldrow; Channing Ludden; Christopher Petersen
>
> **备注:** 12 pages, 4 figures, AAS/AIAA Space Flight Mechanics
>
> **摘要:** This paper compares the required fuel usage for forced and unforced motion of a chaser satellite engaged in Rendezvous, Proximity Operations, and Docking (RPOD) maneuvers. Improved RPOD models are vital, particularly as the space industry expands and demands for improved fuel efficiency, cost effectiveness, and mission life span increase. This paper specifically examines the Clohessy- Wiltshire (CW) Equations and the extent of model mismatch by comparing pre- dicted trajectories from this model with a more computationally complex, higher fidelity RPOD model. This paper assesses several test cases of similar mission parameters, in each case comparing natural motion circumnavigation (NMC) with comparable forced motion circumnavigation. The Guidance, Navigation, and Con- trol (GNC) impulse maneuvers required to maintain the supposedly zero fuel CW trajectories is representative of the extent of CW model mismatch. This paper demonstrates that unforced motions are not inherently more fuel efficient than forced motions, thus permitting extended orbital operations given the higher fuel efficiency.
>
---
#### [new 032] DriveCritic: Towards Context-Aware, Human-Aligned Evaluation for Autonomous Driving with Vision-Language Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对自动驾驶评测中缺乏情境感知与人类判断对齐的问题，提出DriveCritic框架，包含含人类偏好标注的挑战性数据集和基于视觉语言模型的评估器，通过两阶段训练实现更贴近人类判断的上下文感知轨迹评价。**

- **链接: [http://arxiv.org/pdf/2510.13108v1](http://arxiv.org/pdf/2510.13108v1)**

> **作者:** Jingyu Song; Zhenxin Li; Shiyi Lan; Xinglong Sun; Nadine Chang; Maying Shen; Joshua Chen; Katherine A. Skinner; Jose M. Alvarez
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Benchmarking autonomous driving planners to align with human judgment remains a critical challenge, as state-of-the-art metrics like the Extended Predictive Driver Model Score (EPDMS) lack context awareness in nuanced scenarios. To address this, we introduce DriveCritic, a novel framework featuring two key contributions: the DriveCritic dataset, a curated collection of challenging scenarios where context is critical for correct judgment and annotated with pairwise human preferences, and the DriveCritic model, a Vision-Language Model (VLM) based evaluator. Fine-tuned using a two-stage supervised and reinforcement learning pipeline, the DriveCritic model learns to adjudicate between trajectory pairs by integrating visual and symbolic context. Experiments show DriveCritic significantly outperforms existing metrics and baselines in matching human preferences and demonstrates strong context awareness. Overall, our work provides a more reliable, human-aligned foundation to evaluating autonomous driving systems.
>
---
#### [new 033] MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control
- **分类: cs.GR; cs.LG; cs.RO**

- **简介: 该论文提出MimicKit，一个用于动作模仿与控制的强化学习开源框架。旨在统一动作模仿研究的训练流程，解决不同方法间环境、数据结构不一致的问题，提供模块化、易扩展的代码实现，支持图形学与机器人领域的研究应用。**

- **链接: [http://arxiv.org/pdf/2510.13794v1](http://arxiv.org/pdf/2510.13794v1)**

> **作者:** Xue Bin Peng
>
> **摘要:** MimicKit is an open-source framework for training motion controllers using motion imitation and reinforcement learning. The codebase provides implementations of commonly-used motion-imitation techniques and RL algorithms. This framework is intended to support research and applications in computer graphics and robotics by providing a unified training framework, along with standardized environment, agent, and data structures. The codebase is designed to be modular and easily configurable, enabling convenient modification and extension to new characters and tasks. The open-source codebase is available at: https://github.com/xbpeng/MimicKit.
>
---
#### [new 034] Simplicial Embeddings Improve Sample Efficiency in Actor-Critic Agents
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在提升actor-critic算法的样本效率。针对环境交互次数多的问题，提出simplicial embeddings，通过几何结构约束增强表示学习，改善策略梯度和值估计稳定性，在不增加计算开销的前提下提升了性能。**

- **链接: [http://arxiv.org/pdf/2510.13704v1](http://arxiv.org/pdf/2510.13704v1)**

> **作者:** Johan Obando-Ceron; Walter Mayor; Samuel Lavoie; Scott Fujimoto; Aaron Courville; Pablo Samuel Castro
>
> **摘要:** Recent works have proposed accelerating the wall-clock training time of actor-critic methods via the use of large-scale environment parallelization; unfortunately, these can sometimes still require large number of environment interactions to achieve a desired level of performance. Noting that well-structured representations can improve the generalization and sample efficiency of deep reinforcement learning (RL) agents, we propose the use of simplicial embeddings: lightweight representation layers that constrain embeddings to simplicial structures. This geometric inductive bias results in sparse and discrete features that stabilize critic bootstrapping and strengthen policy gradients. When applied to FastTD3, FastSAC, and PPO, simplicial embeddings consistently improve sample efficiency and final performance across a variety of continuous- and discrete-control environments, without any loss in runtime speed.
>
---
#### [new 035] Physics-Informed Neural Network Modeling of Vehicle Collision Dynamics in Precision Immobilization Technique Maneuvers
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文研究车辆碰撞动力学建模，旨在提升预测精度与计算效率。提出双物理信息神经网络框架，结合物理约束与数据驱动方法，实现高精度、实时的碰撞力与运动轨迹预测，适用于精准拦截等安全控制场景。**

- **链接: [http://arxiv.org/pdf/2510.13461v1](http://arxiv.org/pdf/2510.13461v1)**

> **作者:** Yangye Jiang; Jiachen Wang; Daofei Li
>
> **摘要:** Accurate prediction of vehicle collision dynamics is crucial for advanced safety systems and post-impact control applications, yet existing methods face inherent trade-offs among computational efficiency, prediction accuracy, and data requirements. This paper proposes a dual Physics-Informed Neural Network framework addressing these challenges through two complementary networks. The first network integrates Gaussian Mixture Models with PINN architecture to learn impact force distributions from finite element analysis data while enforcing momentum conservation and energy consistency constraints. The second network employs an adaptive PINN with dynamic constraint weighting to predict post-collision vehicle dynamics, featuring an adaptive physics guard layer that prevents unrealistic predictions whil e preserving data-driven learning capabilities. The framework incorporates uncertainty quantification through time-varying parameters and enables rapid adaptation via fine-tuning strategies. Validation demonstrates significant improvements: the impact force model achieves relative errors below 15.0% for force prediction on finite element analysis (FEA) datasets, while the vehicle dynamics model reduces average trajectory prediction error by 63.6% compared to traditional four-degree-of-freedom models in scaled vehicle experiments. The integrated system maintains millisecond-level computational efficiency suitable for real-time applications while providing probabilistic confidence bounds essential for safety-critical control. Comprehensive validation through FEA simulation, dynamic modeling, and scaled vehicle experiments confirms the framework's effectiveness for Precision Immobilization Technique scenarios and general collision dynamics prediction.
>
---
#### [new 036] VLURes: Benchmarking VLM Visual and Linguistic Understanding in Low-Resource Languages
- **分类: cs.CL; cs.AI; cs.CV; cs.RO**

- **简介: 该论文聚焦多模态模型在低资源语言中的视觉与语言理解问题，提出新基准VLURes，包含八项任务和长文本多语言数据，评估十种VLM在英语、日语、斯瓦希里语和乌尔都语下的细粒度理解能力，揭示语言间性能差距。**

- **链接: [http://arxiv.org/pdf/2510.12845v1](http://arxiv.org/pdf/2510.12845v1)**

> **作者:** Jesse Atuhurra; Iqra Ali; Tomoya Iwakura; Hidetaka Kamigaito; Tatsuya Hiraoka
>
> **摘要:** Vision Language Models (VLMs) are pivotal for advancing perception in intelligent agents. Yet, evaluation of VLMs remains limited to predominantly English-centric benchmarks in which the image-text pairs comprise short texts. To evaluate VLM fine-grained abilities, in four languages under long-text settings, we introduce a novel multilingual benchmark VLURes featuring eight vision-and-language tasks, and a pioneering unrelatedness task, to probe the fine-grained Visual and Linguistic Understanding capabilities of VLMs across English, Japanese, and low-resource languages, Swahili, and Urdu. Our datasets, curated from web resources in the target language, encompass ten diverse image categories and rich textual context, introducing valuable vision-language resources for Swahili and Urdu. By prompting VLMs to generate responses and rationales, evaluated automatically and by native speakers, we uncover performance disparities across languages and tasks critical to intelligent agents, such as object recognition, scene understanding, and relationship understanding. We conducted evaluations of ten VLMs with VLURes. The best performing model, GPT-4o, achieves an overall accuracy of 90.8% and lags human performance by 6.7%, though the gap is larger for open-source models. The gap highlights VLURes' critical role in developing intelligent agents to tackle multi-modal visual reasoning.
>
---
#### [new 037] Accelerated Feature Detectors for Visual SLAM: A Comparative Study of FPGA vs GPU
- **分类: cs.CV; cs.ET; cs.PF; cs.RO; C.3; C.4; I.4.6**

- **简介: 该论文研究视觉SLAM中特征检测的硬件加速，比较FPGA与GPU在Jetson Orin和AMD Versal上的性能。针对FAST、Harris和SuperPoint检测器，分析其在运行效率、能耗和精度方面的表现，探讨硬件选择对V-SLAM系统的影响。**

- **链接: [http://arxiv.org/pdf/2510.13546v1](http://arxiv.org/pdf/2510.13546v1)**

> **作者:** Ruiqi Ye; Mikel Luján
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Feature detection is a common yet time-consuming module in Simultaneous Localization and Mapping (SLAM) implementations, which are increasingly deployed on power-constrained platforms, such as drones. Graphics Processing Units (GPUs) have been a popular accelerator for computer vision in general, and feature detection and SLAM in particular. On the other hand, System-on-Chips (SoCs) with integrated Field Programmable Gate Array (FPGA) are also widely available. This paper presents the first study of hardware-accelerated feature detectors considering a Visual SLAM (V-SLAM) pipeline. We offer new insights by comparing the best GPU-accelerated FAST, Harris, and SuperPoint implementations against the FPGA-accelerated counterparts on modern SoCs (Nvidia Jetson Orin and AMD Versal). The evaluation shows that when using a non-learning-based feature detector such as FAST and Harris, their GPU implementations, and the GPU-accelerated V-SLAM can achieve better run-time performance and energy efficiency than the FAST and Harris FPGA implementations as well as the FPGA-accelerated V-SLAM. However, when considering a learning-based detector such as SuperPoint, its FPGA implementation can achieve better run-time performance and energy efficiency (up to 3.1$\times$ and 1.4$\times$ improvements, respectively) than the GPU implementation. The FPGA-accelerated V-SLAM can also achieve comparable run-time performance compared to the GPU-accelerated V-SLAM, with better FPS in 2 out of 5 dataset sequences. When considering the accuracy, the results show that the GPU-accelerated V-SLAM is more accurate than the FPGA-accelerated V-SLAM in general. Last but not least, the use of hardware acceleration for feature detection could further improve the performance of the V-SLAM pipeline by having the global bundle adjustment module invoked less frequently without sacrificing accuracy.
>
---
#### [new 038] A New Perspective on Transformers in Online Reinforcement Learning for Continuous Control
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文研究在线无模型强化学习中的连续控制任务，解决Transformer因训练敏感性导致应用受限的问题。通过探索输入条件、组件共享和序列切片等设计，提出稳定架构与训练策略，使Transformer在多种设置下表现优异。**

- **链接: [http://arxiv.org/pdf/2510.13367v1](http://arxiv.org/pdf/2510.13367v1)**

> **作者:** Nikita Kachaev; Daniil Zelezetsky; Egor Cherepanov; Alexey K. Kovelev; Aleksandr I. Panov
>
> **摘要:** Despite their effectiveness and popularity in offline or model-based reinforcement learning (RL), transformers remain underexplored in online model-free RL due to their sensitivity to training setups and model design decisions such as how to structure the policy and value networks, share components, or handle temporal information. In this paper, we show that transformers can be strong baselines for continuous control in online model-free RL. We investigate key design questions: how to condition inputs, share components between actor and critic, and slice sequential data for training. Our experiments reveal stable architectural and training strategies enabling competitive performance across fully and partially observable tasks, and in both vector- and image-based settings. These findings offer practical guidance for applying transformers in online RL.
>
---
#### [new 039] Through the Lens of Doubt: Robust and Efficient Uncertainty Estimation for Visual Place Recognition
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对视觉位置识别（VPR）中缺乏可靠不确定性估计的问题，提出三种无需训练的不确定性度量方法，通过分析相似性得分的统计特性来评估匹配置信度，提升VPR在复杂环境下的鲁棒性和精度，适用于实时机器人应用。**

- **链接: [http://arxiv.org/pdf/2510.13464v1](http://arxiv.org/pdf/2510.13464v1)**

> **作者:** Emily Miller; Michael Milford; Muhammad Burhan Hafez; SD Ramchurn; Shoaib Ehsan
>
> **摘要:** Visual Place Recognition (VPR) enables robots and autonomous vehicles to identify previously visited locations by matching current observations against a database of known places. However, VPR systems face significant challenges when deployed across varying visual environments, lighting conditions, seasonal changes, and viewpoints changes. Failure-critical VPR applications, such as loop closure detection in simultaneous localization and mapping (SLAM) pipelines, require robust estimation of place matching uncertainty. We propose three training-free uncertainty metrics that estimate prediction confidence by analyzing inherent statistical patterns in similarity scores from any existing VPR method. Similarity Distribution (SD) quantifies match distinctiveness by measuring score separation between candidates; Ratio Spread (RS) evaluates competitive ambiguity among top-scoring locations; and Statistical Uncertainty (SU) is a combination of SD and RS that provides a unified metric that generalizes across datasets and VPR methods without requiring validation data to select the optimal metric. All three metrics operate without additional model training, architectural modifications, or computationally expensive geometric verification. Comprehensive evaluation across nine state-of-the-art VPR methods and six benchmark datasets confirms that our metrics excel at discriminating between correct and incorrect VPR matches, and consistently outperform existing approaches while maintaining negligible computational overhead, making it deployable for real-time robotic applications across varied environmental conditions with improved precision-recall performance.
>
---
#### [new 040] SimULi: Real-Time LiDAR and Camera Simulation with Unscented Transforms
- **分类: cs.CV; cs.GR; cs.LG; cs.RO**

- **简介: 该论文提出SimULi，旨在解决多传感器（LiDAR与相机）实时高保真模拟问题。现有方法速度慢、仅支持针孔相机且存在跨传感器不一致。SimULi基于3DGUT扩展，支持任意相机模型与旋转式LiDAR，通过因子化高斯表示和锚定策略提升一致性，实现高效渲染。**

- **链接: [http://arxiv.org/pdf/2510.12901v1](http://arxiv.org/pdf/2510.12901v1)**

> **作者:** Haithem Turki; Qi Wu; Xin Kang; Janick Martinez Esturo; Shengyu Huang; Ruilong Li; Zan Gojcic; Riccardo de Lutio
>
> **备注:** Project page: https://research.nvidia.com/labs/sil/projects/simuli
>
> **摘要:** Rigorous testing of autonomous robots, such as self-driving vehicles, is essential to ensure their safety in real-world deployments. This requires building high-fidelity simulators to test scenarios beyond those that can be safely or exhaustively collected in the real-world. Existing neural rendering methods based on NeRF and 3DGS hold promise but suffer from low rendering speeds or can only render pinhole camera models, hindering their suitability to applications that commonly require high-distortion lenses and LiDAR data. Multi-sensor simulation poses additional challenges as existing methods handle cross-sensor inconsistencies by favoring the quality of one modality at the expense of others. To overcome these limitations, we propose SimULi, the first method capable of rendering arbitrary camera models and LiDAR data in real-time. Our method extends 3DGUT, which natively supports complex camera models, with LiDAR support, via an automated tiling strategy for arbitrary spinning LiDAR models and ray-based culling. To address cross-sensor inconsistencies, we design a factorized 3D Gaussian representation and anchoring strategy that reduces mean camera and depth error by up to 40% compared to existing methods. SimULi renders 10-20x faster than ray tracing approaches and 1.5-10x faster than prior rasterization-based work (and handles a wider range of camera models). When evaluated on two widely benchmarked autonomous driving datasets, SimULi matches or exceeds the fidelity of existing state-of-the-art methods across numerous camera and LiDAR metrics.
>
---
#### [new 041] Safe Driving in Occluded Environments
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文针对遮挡环境下的自动驾驶安全控制问题，提出基于概率不变性的安全证书方法，以应对不可观测风险。通过构建线性动作约束，降低潜在碰撞风险，兼顾安全性与控制灵活性，并在CARLA中验证了有效性。**

- **链接: [http://arxiv.org/pdf/2510.13114v1](http://arxiv.org/pdf/2510.13114v1)**

> **作者:** Zhuoyuan Wang; Tongyao Jia; Pharuj Rajborirug; Neeraj Ramesh; Hiroyuki Okuda; Tatsuya Suzuki; Soummya Kar; Yorie Nakahira
>
> **摘要:** Ensuring safe autonomous driving in the presence of occlusions poses a significant challenge in its policy design. While existing model-driven control techniques based on set invariance can handle visible risks, occlusions create latent risks in which safety-critical states are not observable. Data-driven techniques also struggle to handle latent risks because direct mappings from risk-critical objects in sensor inputs to safe actions cannot be learned without visible risk-critical objects. Motivated by these challenges, in this paper, we propose a probabilistic safety certificate for latent risk. Our key technical enabler is the application of probabilistic invariance: It relaxes the strict observability requirements imposed by set-invariance methods that demand the knowledge of risk-critical states. The proposed techniques provide linear action constraints that confine the latent risk probability within tolerance. Such constraints can be integrated into model predictive controllers or embedded in data-driven policies to mitigate latent risks. The proposed method is tested using the CARLA simulator and compared with a few existing techniques. The theoretical and empirical analysis jointly demonstrate that the proposed methods assure long-term safety in real-time control in occluded environments without being overly conservative and with transparency to exposed risks.
>
---
## 更新

#### [replaced 001] Behavior Trees and State Machines in Robotics Applications
- **分类: cs.RO; cs.SE; D.0; D.2.13; D.2.2**

- **链接: [http://arxiv.org/pdf/2208.04211v3](http://arxiv.org/pdf/2208.04211v3)**

> **作者:** Razan Ghzouli; Thorsten Berger; Einar Broch Johnsen; Andrzej Wasowski; Swaib Dragule
>
> **备注:** Published at IEEE TSE as a journal extension of a preceding SLE paper (available as arXiv:2010.06256). arXiv admin note: text overlap with arXiv:2010.06256
>
> **摘要:** Autonomous robots combine skills to form increasingly complex behaviors, called missions. While skills are often programmed at a relatively low abstraction level, their coordination is architecturally separated and often expressed in higher-level languages or frameworks. State machines have been the go-to language to model behavior for decades, but recently, behavior trees have gained attention among roboticists. Although several implementations of behavior trees are in use, little is known about their usage and scope in the real world.How do concepts offered by behavior trees relate to traditional languages, such as state machines? How are concepts in behavior trees and state machines used in actual applications? This paper is a study of the key language concepts in behavior trees as realized in domain-specific languages (DSLs), internal and external DSLs offered as libraries, and their use in open-source robotic applications supported by the Robot Operating System (ROS). We analyze behavior-tree DSLs and compare them to the standard language for behavior models in robotics:state machines. We identify DSLs for both behavior-modeling languages, and we analyze five in-depth.We mine open-source repositories for robotic applications that use the analyzed DSLs and analyze their usage. We identify similarities between behavior trees and state machines in terms of language design and the concepts offered to accommodate the needs of the robotics domain. We observed that the usage of behavior-tree DSLs in open-source projects is increasing rapidly. We observed similar usage patterns at model structure and at code reuse in the behavior-tree and state-machine models within the mined open-source projects. We contribute all extracted models as a dataset, hoping to inspire the community to use and further develop behavior trees, associated tools, and analysis techniques.
>
---
#### [replaced 002] Hi-Drive: Hierarchical POMDP Planning for Safe Autonomous Driving in Diverse Urban Environments
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.18411v2](http://arxiv.org/pdf/2409.18411v2)**

> **作者:** Xuanjin Jin; Chendong Zeng; Shengfa Zhu; Chunxiao Liu; Panpan Cai
>
> **摘要:** Uncertainties in dynamic road environments pose significant challenges for behavior and trajectory planning in autonomous driving. This paper introduces Hi-Drive, a hierarchical planning algorithm addressing uncertainties at both behavior and trajectory levels using a hierarchical Partially Observable Markov Decision Process (POMDP) formulation. Hi-Drive employs driver models to represent uncertain behavioral intentions of other vehicles and uses their parameters to infer hidden driving styles. By treating driver models as high-level decision-making actions, our approach effectively manages the exponential complexity inherent in POMDPs. To further enhance safety and robustness, Hi-Drive integrates a trajectory optimization based on importance sampling, refining trajectories using a comprehensive analysis of critical agents. Evaluations on real-world urban driving datasets demonstrate that Hi-Drive significantly outperforms state-of-the-art planning-based and learning-based methods across diverse urban driving situations in real-world benchmarks.
>
---
#### [replaced 003] Extended Friction Models for the Physics Simulation of Servo Actuators
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.08650v3](http://arxiv.org/pdf/2410.08650v3)**

> **作者:** Marc Duclusaud; Grégoire Passault; Vincent Padois; Olivier Ly
>
> **摘要:** Accurate physical simulation is crucial for the development and validation of control algorithms in robotic systems. Recent works in Reinforcement Learning (RL) take notably advantage of extensive simulations to produce efficient robot control. State-of-the-art servo actuator models generally fail at capturing the complex friction dynamics of these systems. This limits the transferability of simulated behaviors to real-world applications. In this work, we present extended friction models that allow to more accurately simulate servo actuator dynamics. We propose a comprehensive analysis of various friction models, present a method for identifying model parameters using recorded trajectories from a pendulum test bench, and demonstrate how these models can be integrated into physics engines. The proposed friction models are validated on four distinct servo actuators and tested on 2R manipulators, showing significant improvements in accuracy over the standard Coulomb-Viscous model. Our results highlight the importance of considering advanced friction effects in the simulation of servo actuators to enhance the realism and reliability of robotic simulations.
>
---
#### [replaced 004] Geometric Backstepping Control of Omnidirectional Tiltrotors Incorporating Servo-Rotor Dynamics for Robustness against Sudden Disturbances
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2510.01675v2](http://arxiv.org/pdf/2510.01675v2)**

> **作者:** Jaewoo Lee; Dongjae Lee; Jinwoo Lee; Hyungyu Lee; Yeonjoon Kim; H. Jin Kim
>
> **摘要:** This work presents a geometric backstepping controller for a variable-tilt omnidirectional multirotor that explicitly accounts for both servo and rotor dynamics. Considering actuator dynamics is essential for more effective and reliable operation, particularly during aggressive flight maneuvers or recovery from sudden disturbances. While prior studies have investigated actuator-aware control for conventional and fixed-tilt multirotors, these approaches rely on linear relationships between actuator input and wrench, which cannot capture the nonlinearities induced by variable tilt angles. In this work, we exploit the cascade structure between the rigid-body dynamics of the multirotor and its nonlinear actuator dynamics to design the proposed backstepping controller and establish exponential stability of the overall system. Furthermore, we reveal parametric uncertainty in the actuator model through experiments, and we demonstrate that the proposed controller remains robust against such uncertainty. The controller was compared against a baseline that does not account for actuator dynamics across three experimental scenarios: fast translational tracking, rapid rotational tracking, and recovery from sudden disturbance. The proposed method consistently achieved better tracking performance, and notably, while the baseline diverged and crashed during the fastest translational trajectory tracking and the recovery experiment, the proposed controller maintained stability and successfully completed the tasks, thereby demonstrating its effectiveness.
>
---
#### [replaced 005] A Hierarchical Bin Packing Framework with Dual Manipulators via Heuristic Search and Deep Reinforcement Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.01628v2](http://arxiv.org/pdf/2506.01628v2)**

> **作者:** Beomjoon Lee; Changjoo Nam
>
> **摘要:** We address the bin packing problem (BPP), which aims to maximize bin utilization when packing a variety of items. The offline problem, where the complete information about the item set and their sizes is known in advance, is proven to be NP-hard. The semi-online and online variants are even more challenging, as full information about incoming items is unavailable. While existing methods have tackled both 2D and 3D BPPs, the 2D BPP remains underexplored in terms of fully maximizing utilization. We propose a hierarchical approach for solving the 2D online and semi-online BPP by combining deep reinforcement learning (RL) with heuristic search. The heuristic search selects which item to pack or unpack, determines the packing order, and chooses the orientation of each item, while the RL agent decides the precise position within the bin. Our method is capable of handling diverse scenarios, including repacking, varying levels of item information, differing numbers of accessible items, and coordination of dual manipulators. Experimental results demonstrate that our approach achieves near-optimal utilization across various practical scenarios, largely due to its repacking capability. In addition, the algorithm is evaluated in a physics-based simulation environment, where execution time is measured to assess its real-world performance.
>
---
#### [replaced 006] EReLiFM: Evidential Reliability-Aware Residual Flow Meta-Learning for Open-Set Domain Generalization under Noisy Labels
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2510.12687v2](http://arxiv.org/pdf/2510.12687v2)**

> **作者:** Kunyu Peng; Di Wen; Kailun Yang; Jia Fu; Yufan Chen; Ruiping Liu; Jiamin Wu; Junwei Zheng; M. Saquib Sarfraz; Luc Van Gool; Danda Pani Paudel; Rainer Stiefelhagen
>
> **备注:** The source code is available at https://github.com/KPeng9510/ERELIFM
>
> **摘要:** Open-Set Domain Generalization (OSDG) aims to enable deep learning models to recognize unseen categories in new domains, which is crucial for real-world applications. Label noise hinders open-set domain generalization by corrupting source-domain knowledge, making it harder to recognize known classes and reject unseen ones. While existing methods address OSDG under Noisy Labels (OSDG-NL) using hyperbolic prototype-guided meta-learning, they struggle to bridge domain gaps, especially with limited clean labeled data. In this paper, we propose Evidential Reliability-Aware Residual Flow Meta-Learning (EReLiFM). We first introduce an unsupervised two-stage evidential loss clustering method to promote label reliability awareness. Then, we propose a residual flow matching mechanism that models structured domain- and category-conditioned residuals, enabling diverse and uncertainty-aware transfer paths beyond interpolation-based augmentation. During this meta-learning process, the model is optimized such that the update direction on the clean set maximizes the loss decrease on the noisy set, using pseudo labels derived from the most confident predicted class for supervision. Experimental results show that EReLiFM outperforms existing methods on OSDG-NL, achieving state-of-the-art performance. The source code is available at https://github.com/KPeng9510/ERELIFM.
>
---
#### [replaced 007] Ctrl-World: A Controllable Generative World Model for Robot Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.10125v2](http://arxiv.org/pdf/2510.10125v2)**

> **作者:** Yanjiang Guo; Lucy Xiaoyang Shi; Jianyu Chen; Chelsea Finn
>
> **备注:** 17 pages
>
> **摘要:** Generalist robot policies can now perform a wide range of manipulation skills, but evaluating and improving their ability with unfamiliar objects and instructions remains a significant challenge. Rigorous evaluation requires a large number of real-world rollouts, while systematic improvement demands additional corrective data with expert labels. Both of these processes are slow, costly, and difficult to scale. World models offer a promising, scalable alternative by enabling policies to rollout within imagination space. However, a key challenge is building a controllable world model that can handle multi-step interactions with generalist robot policies. This requires a world model compatible with modern generalist policies by supporting multi-view prediction, fine-grained action control, and consistent long-horizon interactions, which is not achieved by previous works. In this paper, we make a step forward by introducing a controllable multi-view world model that can be used to evaluate and improve the instruction-following ability of generalist robot policies. Our model maintains long-horizon consistency with a pose-conditioned memory retrieval mechanism and achieves precise action control through frame-level action conditioning. Trained on the DROID dataset (95k trajectories, 564 scenes), our model generates spatially and temporally consistent trajectories under novel scenarios and new camera placements for over 20 seconds. We show that our method can accurately rank policy performance without real-world robot rollouts. Moreover, by synthesizing successful trajectories in imagination and using them for supervised fine-tuning, our approach can improve policy success by 44.7\%.
>
---
#### [replaced 008] Robust Statistics vs. Machine Learning vs. Bayesian Inference: Insights into Handling Faulty GNSS Measurements in Field Robotics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.06015v2](http://arxiv.org/pdf/2504.06015v2)**

> **作者:** Haoming Zhang
>
> **备注:** Accepted to the 2nd Workshop on Safety of Intelligent and Autonomous Vehicles: Formal Methods vs. Machine Learning approaches for reliable navigation (SIAV-FM2L) at IEEE IROS2025
>
> **摘要:** This paper presents research findings on handling faulty measurements (i.e., outliers) of global navigation satellite systems (GNSS) for vehicle localization under adverse signal conditions in field applications, where raw GNSS data are frequently corrupted due to environmental interference such as multipath, signal blockage, or non-line-of-sight conditions. In this context, we investigate three strategies applied specifically to GNSS pseudorange observations: robust statistics for error mitigation, machine learning for faulty measurement prediction, and Bayesian inference for noise distribution approximation. Since previous studies have provided limited insight into the theoretical foundations and practical evaluations of these three methodologies within a unified problem statement (i.e., state estimation using ranging sensors), we conduct extensive experiments using real-world sensor data collected in diverse urban environments. Our goal is to examine both established techniques and newly proposed methods, thereby advancing the understanding of how to handle faulty range measurements, such as GNSS, for robust, long-term vehicle localization. In addition to presenting successful results, this work highlights critical observations and open questions to motivate future research in robust state estimation.
>
---
#### [replaced 009] Flattening Hierarchies with Policy Bootstrapping
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.14975v2](http://arxiv.org/pdf/2505.14975v2)**

> **作者:** John L. Zhou; Jonathan C. Kao
>
> **备注:** NeurIPS 2025 (Spotlight, top 3.2%)
>
> **摘要:** Offline goal-conditioned reinforcement learning (GCRL) is a promising approach for pretraining generalist policies on large datasets of reward-free trajectories, akin to the self-supervised objectives used to train foundation models for computer vision and natural language processing. However, scaling GCRL to longer horizons remains challenging due to the combination of sparse rewards and discounting, which obscures the comparative advantages of primitive actions with respect to distant goals. Hierarchical RL methods achieve strong empirical results on long-horizon goal-reaching tasks, but their reliance on modular, timescale-specific policies and subgoal generation introduces significant additional complexity and hinders scaling to high-dimensional goal spaces. In this work, we introduce an algorithm to train a flat (non-hierarchical) goal-conditioned policy by bootstrapping on subgoal-conditioned policies with advantage-weighted importance sampling. Our approach eliminates the need for a generative model over the (sub)goal space, which we find is key for scaling to high-dimensional control in large state spaces. We further show that existing hierarchical and bootstrapping-based approaches correspond to specific design choices within our derivation. Across a comprehensive suite of state- and pixel-based locomotion and manipulation benchmarks, our method matches or surpasses state-of-the-art offline GCRL algorithms and scales to complex, long-horizon tasks where prior approaches fail. Project page: https://johnlyzhou.github.io/saw/
>
---
#### [replaced 010] Product Digital Twin Supporting End-of-life Phase of Electric Vehicle Batteries Utilizing Product-Process-Resource Asset Network
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2510.02167v3](http://arxiv.org/pdf/2510.02167v3)**

> **作者:** Sara Strakosova; Petr Novak; Petr Kadera
>
> **备注:** \copyright 2024 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** In a circular economy, products in their end-of-life phase should be either remanufactured or recycled. Both of these processes are crucial for sustainability and environmental conservation. However, manufacturers frequently do not support these processes enough in terms of not sharing relevant data about the products nor their (re-)manufacturing processes. This paper proposes to accompany each product with a digital twin technology, specifically the Product Digital Twin (PDT), which can carry information for facilitating and optimizing production and remanufacturing processes. This paper introduces a knowledge representation called Bi-Flow Product-Process-Resource Asset Network (Bi-PAN). Bi-PAN extends a well-proven Product-Process-Resource Asset Network (PAN) paradigm by integrating both assembly and disassembly workflows into a single information model. Such networks enable capturing relevant relationships across products, production resources, manufacturing processes, and specific production operations that have to be done in the manufacturing phase of a product. The proposed approach is demonstrated in a use-case of disassembling electric vehicle (EV) batteries. By utilizing PDTs with Bi-PAN knowledge models, challenges associated with disassembling of EV batteries can be solved flexibly and efficiently for various battery types, enhancing the sustainability of the EV battery life-cycle management.
>
---
#### [replaced 011] RealEngine: Simulating Autonomous Driving in Realistic Context
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.16902v2](http://arxiv.org/pdf/2505.16902v2)**

> **作者:** Junzhe Jiang; Nan Song; Jingyu Li; Xiatian Zhu; Li Zhang
>
> **摘要:** Driving simulation plays a crucial role in developing reliable driving agents by providing controlled, evaluative environments. To enable meaningful assessments, a high-quality driving simulator must satisfy several key requirements: multi-modal sensing capabilities (e.g., camera and LiDAR) with realistic scene rendering to minimize observational discrepancies; closed-loop evaluation to support free-form trajectory behaviors; highly diverse traffic scenarios for thorough evaluation; multi-agent cooperation to capture interaction dynamics; and high computational efficiency to ensure affordability and scalability. However, existing simulators and benchmarks fail to comprehensively meet these fundamental criteria. To bridge this gap, this paper introduces RealEngine, a novel driving simulation framework that holistically integrates 3D scene reconstruction and novel view synthesis techniques to achieve realistic and flexible closed-loop simulation in the driving context. By leveraging real-world multi-modal sensor data, RealEngine reconstructs background scenes and foreground traffic participants separately, allowing for highly diverse and realistic traffic scenarios through flexible scene composition. This synergistic fusion of scene reconstruction and view synthesis enables photorealistic rendering across multiple sensor modalities, ensuring both perceptual fidelity and geometric accuracy. Building upon this environment, RealEngine supports three essential driving simulation categories: non-reactive simulation, safety testing, and multi-agent interaction, collectively forming a reliable and comprehensive benchmark for evaluating the real-world performance of driving agents.
>
---
#### [replaced 012] QuaDreamer: Controllable Panoramic Video Generation for Quadruped Robots
- **分类: cs.RO; cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2508.02512v3](http://arxiv.org/pdf/2508.02512v3)**

> **作者:** Sheng Wu; Fei Teng; Hao Shi; Qi Jiang; Kai Luo; Kaiwei Wang; Kailun Yang
>
> **备注:** Accepted to CoRL 2025. The source code and model weights will be publicly available at https://github.com/losehu/QuaDreamer
>
> **摘要:** Panoramic cameras, capturing comprehensive 360-degree environmental data, are suitable for quadruped robots in surrounding perception and interaction with complex environments. However, the scarcity of high-quality panoramic training data-caused by inherent kinematic constraints and complex sensor calibration challenges-fundamentally limits the development of robust perception systems tailored to these embodied platforms. To address this issue, we propose QuaDreamer-the first panoramic data generation engine specifically designed for quadruped robots. QuaDreamer focuses on mimicking the motion paradigm of quadruped robots to generate highly controllable, realistic panoramic videos, providing a data source for downstream tasks. Specifically, to effectively capture the unique vertical vibration characteristics exhibited during quadruped locomotion, we introduce Vertical Jitter Encoding (VJE). VJE extracts controllable vertical signals through frequency-domain feature filtering and provides high-quality prompts. To facilitate high-quality panoramic video generation under jitter signal control, we propose a Scene-Object Controller (SOC) that effectively manages object motion and boosts background jitter control through the attention mechanism. To address panoramic distortions in wide-FoV video generation, we propose the Panoramic Enhancer (PE)-a dual-stream architecture that synergizes frequency-texture refinement for local detail enhancement with spatial-structure correction for global geometric consistency. We further demonstrate that the generated video sequences can serve as training data for the quadruped robot's panoramic visual perception model, enhancing the performance of multi-object tracking in 360-degree scenes. The source code and model weights will be publicly available at https://github.com/losehu/QuaDreamer.
>
---
#### [replaced 013] Multimodal Fusion and Vision-Language Models: A Survey for Robot Vision
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02477v3](http://arxiv.org/pdf/2504.02477v3)**

> **作者:** Xiaofeng Han; Shunpeng Chen; Zenghuang Fu; Zhe Feng; Lue Fan; Dong An; Changwei Wang; Li Guo; Weiliang Meng; Xiaopeng Zhang; Rongtao Xu; Shibiao Xu
>
> **备注:** 27 pages, 11 figures. Accepted to Information Fusion. Final journal version: volume 126 (Part B), February 2026
>
> **摘要:** Robot vision has greatly benefited from advancements in multimodal fusion techniques and vision-language models (VLMs). We adopt a task-oriented perspective to systematically review the applications and advancements of multimodal fusion methods and VLMs in the field of robot vision. For semantic scene understanding tasks, we categorize fusion approaches into encoder-decoder frameworks, attention-based architectures, and graph neural networks. Meanwhile, we also analyze the architectural characteristics and practical implementations of these fusion strategies in key tasks such as simultaneous localization and mapping (SLAM), 3D object detection, navigation, and manipulation. We compare the evolutionary paths and applicability of VLMs based on large language models (LLMs) with traditional multimodal fusion methods.Additionally, we conduct an in-depth analysis of commonly used datasets, evaluating their applicability and challenges in real-world robotic scenarios. Building on this analysis, we identify key challenges in current research, including cross-modal alignment, efficient fusion, real-time deployment, and domain adaptation. We propose future directions such as self-supervised learning for robust multimodal representations, structured spatial memory and environment modeling to enhance spatial intelligence, and the integration of adversarial robustness and human feedback mechanisms to enable ethically aligned system deployment. Through a comprehensive review, comparative analysis, and forward-looking discussion, we provide a valuable reference for advancing multimodal perception and interaction in robotic vision. A comprehensive list of studies in this survey is available at https://github.com/Xiaofeng-Han-Res/MF-RV.
>
---
#### [replaced 014] A Verification Methodology for Safety Assurance of Robotic Autonomous Systems
- **分类: cs.RO; cs.FL; cs.SE; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.19622v2](http://arxiv.org/pdf/2506.19622v2)**

> **作者:** Mustafa Adam; David A. Anisi; Pedro Ribeiro
>
> **备注:** In Proc. of the 26th TAROS (Towards Autonomous Robotic Systems) Conference, York, UK, August, 2025
>
> **摘要:** Autonomous robots deployed in shared human environments, such as agricultural settings, require rigorous safety assurance to meet both functional reliability and regulatory compliance. These systems must operate in dynamic, unstructured environments, interact safely with humans, and respond effectively to a wide range of potential hazards. This paper presents a verification workflow for the safety assurance of an autonomous agricultural robot, covering the entire development life-cycle, from concept study and design to runtime verification. The outlined methodology begins with a systematic hazard analysis and risk assessment to identify potential risks and derive corresponding safety requirements. A formal model of the safety controller is then developed to capture its behaviour and verify that the controller satisfies the specified safety properties with respect to these requirements. The proposed approach is demonstrated on a field robot operating in an agricultural setting. The results show that the methodology can be effectively used to verify safety-critical properties and facilitate the early identification of design issues, contributing to the development of safer robots and autonomous systems.
>
---
#### [replaced 015] EO-1: Interleaved Vision-Text-Action Pretraining for General Robot Control
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.21112v4](http://arxiv.org/pdf/2508.21112v4)**

> **作者:** Delin Qu; Haoming Song; Qizhi Chen; Zhaoqing Chen; Xianqiang Gao; Xinyi Ye; Qi Lv; Modi Shi; Guanghui Ren; Cheng Ruan; Maoqing Yao; Haoran Yang; Jiacheng Bao; Bin Zhao; Dong Wang
>
> **摘要:** The human ability to seamlessly perform multimodal reasoning and physical interaction in the open world is a core goal for general-purpose embodied intelligent systems. Recent vision-language-action (VLA) models, which are co-trained on large-scale robot and visual-text data, have demonstrated notable progress in general robot control. However, they still fail to achieve human-level flexibility in interleaved reasoning and interaction. In this work, introduce EO-Robotics, consists of EO-1 model and EO-Data1.5M dataset. EO-1 is a unified embodied foundation model that achieves superior performance in multimodal embodied reasoning and robot control through interleaved vision-text-action pre-training. The development of EO-1 is based on two key pillars: (i) a unified architecture that processes multimodal inputs indiscriminately (image, text, video, and action), and (ii) a massive, high-quality multimodal embodied reasoning dataset, EO-Data1.5M, which contains over 1.5 million samples with emphasis on interleaved vision-text-action comprehension. EO-1 is trained through synergies between auto-regressive decoding and flow matching denoising on EO-Data1.5M, enabling seamless robot action generation and multimodal embodied reasoning. Extensive experiments demonstrate the effectiveness of interleaved vision-text-action learning for open-world understanding and generalization, validated through a variety of long-horizon, dexterous manipulation tasks across multiple embodiments. This paper details the architecture of EO-1, the data construction strategy of EO-Data1.5M, and the training methodology, offering valuable insights for developing advanced embodied foundation models.
>
---
#### [replaced 016] Dual-Regularized Riccati Recursions for Interior-Point Optimal Control
- **分类: math.OC; cs.MS; cs.RO; cs.SY; eess.SY; 49M37, 90C51, 93B45; G.1.6**

- **链接: [http://arxiv.org/pdf/2509.16370v2](http://arxiv.org/pdf/2509.16370v2)**

> **作者:** João Sousa-Pinto; Dominique Orban
>
> **摘要:** We derive closed-form extensions of Riccati's recursions (both sequential and parallel) for solving dual-regularized LQR problems. We show how these methods can be used to solve general constrained, non-convex, discrete-time optimal control problems via a regularized interior point method, while guaranteeing that each step is a descent direction of an Augmented Barrier-Lagrangian merit function. We provide MIT-licensed implementations of our methods in C++ and JAX.
>
---
#### [replaced 017] Inland-LOAM: Voxel-Based Structural Semantic LiDAR Odometry and Mapping for Inland Waterway Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.03672v2](http://arxiv.org/pdf/2508.03672v2)**

> **作者:** Zhongbi Luo; Yunjia Wang; Jan Swevers; Peter Slaets; Herman Bruyninckx
>
> **摘要:** Accurate geospatial information is crucial for safe, autonomous Inland Waterway Transport (IWT), as existing charts (IENC) lack real-time detail and conventional LiDAR SLAM fails in waterway environments. These challenges lead to vertical drift and non-semantic maps, hindering autonomous navigation. This paper introduces Inland-LOAM, a LiDAR SLAM framework for waterways. It uses an improved feature extraction and a water surface planar constraint to mitigate vertical drift. A novel pipeline transforms 3D point clouds into structured 2D semantic maps using voxel-based geometric analysis, enabling real-time computation of navigational parameters like bridge clearances. An automated module extracts shorelines and exports them into a lightweight, IENC-compatible format. Evaluations on a real-world dataset show Inland-LOAM achieves superior localization accuracy over state-of-the-art methods. The generated semantic maps and shorelines align with real-world conditions, providing reliable data for enhanced situational awareness. The code and dataset will be publicly available
>
---
#### [replaced 018] LLM-Enabled In-Context Learning for Data Collection Scheduling in UAV-assisted Sensor Networks
- **分类: cs.AI; cs.ET; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.14556v2](http://arxiv.org/pdf/2504.14556v2)**

> **作者:** Yousef Emami; Hao Zhou; SeyedSina Nabavirazani; Luis Almeida
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are increasingly being utilized in various private and commercial applications, e.g., traffic control, parcel delivery, and Search and Rescue (SAR) missions. Machine Learning (ML) methods used in UAV-Assisted Sensor Networks (UASNETs) and, especially, in Deep Reinforcement Learning (DRL) face challenges such as complex and lengthy model training, gaps between simulation and reality, and low sampling efficiency, which conflict with the urgency of emergencies, such as SAR missions. In this paper, an In-Context Learning (ICL)-Data Collection Scheduling (ICLDC) system is proposed as an alternative to DRL in emergencies. The UAV collects sensory data and transmits it to a Large Language Model (LLM), which creates a task description in natural language. From this description, the UAV receives a data collection schedule that must be executed. A verifier ensures safe UAV operations by evaluating the schedules generated by the LLM and overriding unsafe schedules based on predefined rules. The system continuously adapts by incorporating feedback into the task descriptions and using this for future decisions. This method is tested against jailbreaking attacks, where the task description is manipulated to undermine network performance, highlighting the vulnerability of LLMs to such attacks. The proposed ICLDC significantly reduces cumulative packet loss compared to both the DQN and Maximum Channel Gain baselines. ICLDC presents a promising direction for intelligent scheduling and control in UASNETs.
>
---
#### [replaced 019] GARField: Addressing the visual Sim-to-Real gap in garment manipulation with mesh-attached radiance fields
- **分类: cs.RO; cs.GR**

- **链接: [http://arxiv.org/pdf/2410.05038v3](http://arxiv.org/pdf/2410.05038v3)**

> **作者:** Donatien Delehelle; Darwin G. Caldwell; Fei Chen
>
> **备注:** Project site: https://ddonatien.github.io/garfield-website/
>
> **摘要:** While humans intuitively manipulate garments and other textile items swiftly and accurately, it is a significant challenge for robots. A factor crucial to human performance is the ability to imagine, a priori, the intended result of the manipulation intents and hence develop predictions on the garment pose. That ability allows us to plan from highly obstructed states, adapt our plans as we collect more information and react swiftly to unforeseen circumstances. Conversely, robots struggle to establish such intuitions and form tight links between plans and observations. We can partly attribute this to the high cost of obtaining densely labelled data for textile manipulation, both in quality and quantity. The problem of data collection is a long-standing issue in data-based approaches to garment manipulation. As of today, generating high-quality and labelled garment manipulation data is mainly attempted through advanced data capture procedures that create simplified state estimations from real-world observations. However, this work proposes a novel approach to the problem by generating real-world observations from object states. To achieve this, we present GARField (Garment Attached Radiance Field), the first differentiable rendering architecture, to our knowledge, for data generation from simulated states stored as triangle meshes. Code is available on https://ddonatien.github.io/garfield-website/
>
---
#### [replaced 020] Tiny Learning-Based MPC for Multirotors: Solver-Aware Learning for Efficient Embedded Predictive Control
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2410.23634v3](http://arxiv.org/pdf/2410.23634v3)**

> **作者:** Babak Akbari; Justin Frank; Melissa Greeff
>
> **摘要:** Tiny aerial robots hold great promise for applications such as environmental monitoring and search-and-rescue, yet face significant control challenges due to limited onboard computing power and nonlinear dynamics. Model Predictive Control (MPC) enables agile trajectory tracking and constraint handling but depends on an accurate dynamics model. While existing Learning-Based (LB) MPC methods, such as Gaussian Process (GP) MPC, enhance performance by learning residual dynamics, their high computational cost restricts onboard deployment on tiny robots. This paper introduces Tiny LB MPC, a co-designed MPC framework and optimization solver for resource-constrained micro multirotor platforms. The proposed approach achieves 100 Hz control on a Crazyflie 2.1 equipped with a Teensy 4.0 microcontroller, demonstrating a 43% average improvement in tracking performance over existing embedded MPC methods under model uncertainty, and achieving the first onboard implementation of LB MPC on a 53 g multirotor.
>
---
#### [replaced 021] Hybrid Terrain-Aware Path Planning: Integrating VD-RRT* Exploration and VD-D* Lite Repair
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2510.12169v2](http://arxiv.org/pdf/2510.12169v2)**

> **作者:** Akshay Naik; William R. Norris; Dustin Nottage; Ahmet Soylemezoglu
>
> **摘要:** Autonomous ground vehicles operating off-road must plan curvature-feasible paths while accounting for spatially varying soil strength and slope hazards in real time. We present a continuous state--cost metric that combines a Bekker pressure--sinkage model with elevation-derived slope and attitude penalties. The resulting terrain cost field is analytic, bounded, and monotonic in soil modulus and slope, ensuring well-posed discretization and stable updates under sensor noise. This metric is evaluated on a lattice with exact steering primitives: Dubins and Reeds--Shepp motions for differential drive and time-parameterized bicycle arcs for Ackermann steering. Global exploration is performed using Vehicle-Dynamics RRT\(^{*}\), while local repair is managed by Vehicle-Dynamics D\(^{*}\) Lite, enabling millisecond-scale replanning without heuristic smoothing. By separating the terrain--vehicle model from the planner, the framework provides a reusable basis for deterministic, sampling-based, or learning-driven planning in deformable terrain. Hardware trials on an off-road platform demonstrate real-time navigation across soft soil and slope transitions, supporting reliable autonomy in unstructured environments.
>
---
#### [replaced 022] More than A Point: Capturing Uncertainty with Adaptive Affordance Heatmaps for Spatial Grounding in Robotic Tasks
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.10912v2](http://arxiv.org/pdf/2510.10912v2)**

> **作者:** Xinyu Shao; Yanzhe Tang; Pengwei Xie; Kaiwen Zhou; Yuzheng Zhuang; Xingyue Quan; Jianye Hao; Long Zeng; Xiu Li
>
> **备注:** More details and videos can be found at https://robo-map.github.io
>
> **摘要:** Many language-guided robotic systems rely on collapsing spatial reasoning into discrete points, making them brittle to perceptual noise and semantic ambiguity. To address this challenge, we propose RoboMAP, a framework that represents spatial targets as continuous, adaptive affordance heatmaps. This dense representation captures the uncertainty in spatial grounding and provides richer information for downstream policies, thereby significantly enhancing task success and interpretability. RoboMAP surpasses the previous state-of-the-art on a majority of grounding benchmarks with up to a 50x speed improvement, and achieves an 82\% success rate in real-world manipulation. Across extensive simulated and physical experiments, it demonstrates robust performance and shows strong zero-shot generalization to navigation. More details and videos can be found at https://robo-map.github.io.
>
---
#### [replaced 023] EmbodiedCoder: Parameterized Embodied Mobile Manipulation via Modern Coding Model
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.06207v2](http://arxiv.org/pdf/2510.06207v2)**

> **作者:** Zefu Lin; Rongxu Cui; Chen Hanning; Xiangyu Wang; Junjia Xu; Xiaojuan Jin; Chen Wenbo; Hui Zhou; Lue Fan; Wenling Li; Zhaoxiang Zhang
>
> **备注:** Demo Page: https://embodiedcoder.github.io/EmbodiedCoder/
>
> **摘要:** Recent advances in control robot methods, from end-to-end vision-language-action frameworks to modular systems with predefined primitives, have advanced robots' ability to follow natural language instructions. Nonetheless, many approaches still struggle to scale to diverse environments, as they often rely on large annotated datasets and offer limited interpretability.In this work, we introduce EmbodiedCoder, a training-free framework for open-world mobile robot manipulation that leverages coding models to directly generate executable robot trajectories. By grounding high-level instructions in code, EmbodiedCoder enables flexible object geometry parameterization and manipulation trajectory synthesis without additional data collection or fine-tuning.This coding-based paradigm provides a transparent and generalizable way to connect perception with manipulation. Experiments on real mobile robots show that EmbodiedCoder achieves robust performance across diverse long-term tasks and generalizes effectively to novel objects and environments.Our results demonstrate an interpretable approach for bridging high-level reasoning and low-level control, moving beyond fixed primitives toward versatile robot intelligence. See the project page at: https://embodiedcoder.github.io/EmbodiedCoder/
>
---
#### [replaced 024] Towards Proprioception-Aware Embodied Planning for Dual-Arm Humanoid Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.07882v2](http://arxiv.org/pdf/2510.07882v2)**

> **作者:** Boyu Li; Siyuan He; Hang Xu; Haoqi Yuan; Xinrun Xu; Yu Zang; Liwei Hu; Junpeng Yue; Zhenxiong Jiang; Pengbo Hu; Börje F. Karlsson; Yehui Tang; Zongqing Lu
>
> **摘要:** In recent years, Multimodal Large Language Models (MLLMs) have demonstrated the ability to serve as high-level planners, enabling robots to follow complex human instructions. However, their effectiveness, especially in long-horizon tasks involving dual-arm humanoid robots, remains limited. This limitation arises from two main challenges: (i) the absence of simulation platforms that systematically support task evaluation and data collection for humanoid robots, and (ii) the insufficient embodiment awareness of current MLLMs, which hinders reasoning about dual-arm selection logic and body positions during planning. To address these issues, we present DualTHOR, a new dual-arm humanoid simulator, with continuous transition and a contingency mechanism. Building on this platform, we propose Proprio-MLLM, a model that enhances embodiment awareness by incorporating proprioceptive information with motion-based position embedding and a cross-spatial encoder. Experiments show that, while existing MLLMs struggle in this environment, Proprio-MLLM achieves an average improvement of 19.75% in planning performance. Our work provides both an essential simulation platform and an effective model to advance embodied intelligence in humanoid robotics. The code is available at https://anonymous.4open.science/r/DualTHOR-5F3B.
>
---
#### [replaced 025] USIM and U0: A Vision-Language-Action Dataset and Model for General Underwater Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.07869v3](http://arxiv.org/pdf/2510.07869v3)**

> **作者:** Junwen Gu; Zhiheng Wu; Pengxuan Si; Shuang Qiu; Yukai Feng; Luoyang Sun; Laien Luo; Lianyi Yu; Jian Wang; Zhengxing Wu
>
> **备注:** Project Page: https://vincentgu2000.github.io/u0project/
>
> **摘要:** Underwater environments present unique challenges for robotic operation, including complex hydrodynamics, limited visibility, and constrained communication. Although data-driven approaches have advanced embodied intelligence in terrestrial robots and enabled task-specific autonomous underwater robots, developing underwater intelligence capable of autonomously performing multiple tasks remains highly challenging, as large-scale, high-quality underwater datasets are still scarce. To address these limitations, we introduce USIM, a simulation-based multi-task Vision-Language-Action (VLA) dataset for underwater robots. USIM comprises over 561K frames from 1,852 trajectories, totaling approximately 15.6 hours of BlueROV2 interactions across 20 tasks in 9 diverse scenarios, ranging from visual navigation to mobile manipulation. Building upon this dataset, we propose U0, a VLA model for general underwater robots, which integrates binocular vision and other sensor modalities through multimodal fusion, and further incorporates a convolution-attention-based perception focus enhancement module (CAP) to improve spatial understanding and mobile manipulation. Across tasks such as inspection, obstacle avoidance, scanning, and dynamic tracking, the framework achieves a success rate of 80%, while in challenging mobile manipulation tasks, it reduces the distance to the target by 21.2% compared with baseline methods, demonstrating its effectiveness. USIM and U0 show that VLA models can be effectively applied to underwater robotic applications, providing a foundation for scalable dataset construction, improved task autonomy, and the practical realization of intelligent general underwater robots.
>
---
#### [replaced 026] MTIL: Encoding Full History with Mamba for Temporal Imitation Learning
- **分类: cs.RO; I.2.9**

- **链接: [http://arxiv.org/pdf/2505.12410v3](http://arxiv.org/pdf/2505.12410v3)**

> **作者:** Yulin Zhou; Yuankai Lin; Fanzhe Peng; Jiahui Chen; Kaiji Huang; Hua Yang; Zhouping Yin
>
> **备注:** Published in IEEE Robotics and Automation Letters (RA-L), 2025. 8 pages, 5 figures
>
> **摘要:** Standard imitation learning (IL) methods have achieved considerable success in robotics, yet often rely on the Markov assumption, which falters in long-horizon tasks where history is crucial for resolving perceptual ambiguity. This limitation stems not only from a conceptual gap but also from a fundamental computational barrier: prevailing architectures like Transformers are often constrained by quadratic complexity, rendering the processing of long, high-dimensional observation sequences infeasible. To overcome this dual challenge, we introduce Mamba Temporal Imitation Learning (MTIL). Our approach represents a new paradigm for robotic learning, which we frame as a practical synthesis of World Model and Dynamical System concepts. By leveraging the linear-time recurrent dynamics of State Space Models (SSMs), MTIL learns an implicit, action-oriented world model that efficiently encodes the entire trajectory history into a compressed, evolving state. This allows the policy to be conditioned on a comprehensive temporal context, transcending the confines of Markovian approaches. Through extensive experiments on simulated benchmarks (ACT, Robomimic, LIBERO) and on challenging real-world tasks, MTIL demonstrates superior performance against SOTA methods like ACT and Diffusion Policy, particularly in resolving long-term temporal ambiguities. Our findings not only affirm the necessity of full temporal context but also validate MTIL as a powerful and a computationally feasible approach for learning long-horizon, non-Markovian behaviors from high-dimensional observations.
>
---
