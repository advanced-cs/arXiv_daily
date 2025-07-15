# 机器人 cs.RO

- **最新发布 61 篇**

- **更新 34 篇**

## 最新发布

#### [new 001] Probabilistic Human Intent Prediction for Mobile Manipulation: An Evaluation with Human-Inspired Constraints
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 论文提出GUIDER框架，用于移动操作中的人机协作，解决人类意图预测问题。通过双阶段（导航与操作）概率模型，结合多源信息实时估计用户意图。实验验证其在导航与操作任务中的稳定性与提前预测能力均优于基线方法。**

- **链接: [http://arxiv.org/pdf/2507.10131v1](http://arxiv.org/pdf/2507.10131v1)**

> **作者:** Cesar Alan Contreras; Manolis Chiou; Alireza Rastegarpanah; Michal Szulik; Rustam Stolkin
>
> **备注:** Submitted to Journal of Intelligent & Robotic Systems (Under Review)
>
> **摘要:** Accurate inference of human intent enables human-robot collaboration without constraining human control or causing conflicts between humans and robots. We present GUIDER (Global User Intent Dual-phase Estimation for Robots), a probabilistic framework that enables a robot to estimate the intent of human operators. GUIDER maintains two coupled belief layers, one tracking navigation goals and the other manipulation goals. In the Navigation phase, a Synergy Map blends controller velocity with an occupancy grid to rank interaction areas. Upon arrival at a goal, an autonomous multi-view scan builds a local 3D cloud. The Manipulation phase combines U2Net saliency, FastSAM instance saliency, and three geometric grasp-feasibility tests, with an end-effector kinematics-aware update rule that evolves object probabilities in real-time. GUIDER can recognize areas and objects of intent without predefined goals. We evaluated GUIDER on 25 trials (five participants x five task variants) in Isaac Sim, and compared it with two baselines, one for navigation and one for manipulation. Across the 25 trials, GUIDER achieved a median stability of 93-100% during navigation, compared with 60-100% for the BOIR baseline, with an improvement of 39.5% in a redirection scenario (T5). During manipulation, stability reached 94-100% (versus 69-100% for Trajectron), with a 31.4% difference in a redirection task (T3). In geometry-constrained trials (manipulation), GUIDER recognized the object intent three times earlier than Trajectron (median remaining time to confident prediction 23.6 s vs 7.8 s). These results validate our dual-phase framework and show improvements in intent inference in both phases of mobile manipulation tasks.
>
---
#### [new 002] mmE-Loc: Facilitating Accurate Drone Landing with Ultra-High-Frequency Localization
- **分类: cs.RO**

- **简介: 该论文属于无人机定位任务，旨在解决传统相机与毫米波雷达采样频率不匹配导致的定位瓶颈问题。工作提出mmE-Loc系统，结合事件相机与毫米波雷达，并设计两个模块提升定位精度与效率，实现更准确、低延迟的无人机着陆。**

- **链接: [http://arxiv.org/pdf/2507.09469v1](http://arxiv.org/pdf/2507.09469v1)**

> **作者:** Haoyang Wang; Jingao Xu; Xinyu Luo; Ting Zhang; Xuecheng Chen; Ruiyang Duan; Jialong Chen; Yunhao Liu; Jianfeng Zheng; Weijie Hong; Xinlei Chen
>
> **备注:** 17 pages, 34 figures. arXiv admin note: substantial text overlap with arXiv:2502.14992
>
> **摘要:** For precise, efficient, and safe drone landings, ground platforms should real-time, accurately locate descending drones and guide them to designated spots. While mmWave sensing combined with cameras improves localization accuracy, lower sampling frequency of traditional frame cameras compared to mmWave radar creates bottlenecks in system throughput. In this work, we upgrade traditional frame camera with event camera, a novel sensor that harmonizes in sampling frequency with mmWave radar within ground platform setup, and introduce mmE-Loc, a high-precision, low-latency ground localization system designed for precise drone landings. To fully exploit the \textit{temporal consistency} and \textit{spatial complementarity} between these two modalities, we propose two innovative modules: \textit{(i)} the Consistency-instructed Collaborative Tracking module, which further leverages the drone's physical knowledge of periodic micro-motions and structure for accurate measurements extraction, and \textit{(ii)} the Graph-informed Adaptive Joint Optimization module, which integrates drone motion information for efficient sensor fusion and drone localization. Real-world experiments conducted in landing scenarios with a drone delivery company demonstrate that mmE-Loc significantly outperforms state-of-the-art methods in both accuracy and latency.
>
---
#### [new 003] Self-supervised Pretraining for Integrated Prediction and Planning of Automated Vehicles
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决预测周围智能体行为并规划安全路径的问题。作者提出了Plan-MAE预训练框架，通过重建道路网络、轨迹、导航路径及局部子规划任务，提升场景理解与规划能力，实现更精准的预测与规划。**

- **链接: [http://arxiv.org/pdf/2507.09537v1](http://arxiv.org/pdf/2507.09537v1)**

> **作者:** Yangang Ren; Guojian Zhan; Chen Lv; Jun Li; Fenghua Liang; Keqiang Li
>
> **摘要:** Predicting the future of surrounding agents and accordingly planning a safe, goal-directed trajectory are crucial for automated vehicles. Current methods typically rely on imitation learning to optimize metrics against the ground truth, often overlooking how scene understanding could enable more holistic trajectories. In this paper, we propose Plan-MAE, a unified pretraining framework for prediction and planning that capitalizes on masked autoencoders. Plan-MAE fuses critical contextual understanding via three dedicated tasks: reconstructing masked road networks to learn spatial correlations, agent trajectories to model social interactions, and navigation routes to capture destination intents. To further align vehicle dynamics and safety constraints, we incorporate a local sub-planning task predicting the ego-vehicle's near-term trajectory segment conditioned on earlier segment. This pretrained model is subsequently fine-tuned on downstream tasks to jointly generate the prediction and planning trajectories. Experiments on large-scale datasets demonstrate that Plan-MAE outperforms current methods on the planning metrics by a large margin and can serve as an important pre-training step for learning-based motion planner.
>
---
#### [new 004] Customize Harmonic Potential Fields via Hybrid Optimization over Homotopic Paths
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，旨在解决如何定制基于调和势场的导航路径拓扑特性的问题。现有方法无法灵活调整调和势场及路径结构。作者提出一种混合优化方法，在复杂环境中自动搜索同伦类路径，并通过梯度下降优化参数，实现对路径拓扑的定制化设计。**

- **链接: [http://arxiv.org/pdf/2507.09858v1](http://arxiv.org/pdf/2507.09858v1)**

> **作者:** Shuaikang Wang; Tiecheng Guo; Meng Guo
>
> **备注:** accepted to IEEE RA-L
>
> **摘要:** Safe navigation within a workspace is a fundamental skill for autonomous robots to accomplish more complex tasks. Harmonic potentials are artificial potential fields that are analytical, globally convergent and provably free of local minima. Thus, it has been widely used for generating safe and reliable robot navigation control policies. However, most existing methods do not allow customization of the harmonic potential fields nor the resulting paths, particularly regarding their topological properties. In this paper, we propose a novel method that automatically finds homotopy classes of paths that can be generated by valid harmonic potential fields. The considered complex workspaces can be as general as forest worlds consisting of numerous overlapping star-obstacles. The method is based on a hybrid optimization algorithm that searches over homotopy classes, selects the structure of each tree-of-stars within the forest, and optimizes over the continuous weight parameters for each purged tree via the projected gradient descent. The key insight is to transform the forest world to the unbounded point world via proper diffeomorphic transformations. It not only facilitates a simpler design of the multi-directional D-signature between non-homotopic paths, but also retain the safety and convergence properties. Extensive simulations and hardware experiments are conducted for non-trivial scenarios, where the navigation potentials are customized for desired homotopic properties. Project page: https://shuaikang-wang.github.io/CustFields.
>
---
#### [new 005] Online 3D Bin Packing with Fast Stability Validation and Stable Rearrangement Planning
- **分类: cs.RO**

- **简介: 该论文属于在线三维装箱任务，旨在解决物品实时装箱时的结构稳定性和重排规划问题。作者提出了基于负载可承载凸多边形（LBCP）的稳定性验证方法和稳定重排规划（SRP）模块，实现了高效稳定的自动装箱策略。**

- **链接: [http://arxiv.org/pdf/2507.09123v1](http://arxiv.org/pdf/2507.09123v1)**

> **作者:** Ziyan Gao; Lijun Wang; Yuntao Kong; Nak Young Chong
>
> **摘要:** The Online Bin Packing Problem (OBPP) is a sequential decision-making task in which each item must be placed immediately upon arrival, with no knowledge of future arrivals. Although recent deep-reinforcement-learning methods achieve superior volume utilization compared with classical heuristics, the learned policies cannot ensure the structural stability of the bin and lack mechanisms for safely reconfiguring the bin when a new item cannot be placed directly. In this work, we propose a novel framework that integrates packing policy with structural stability validation and heuristic planning to overcome these limitations. Specifically, we introduce the concept of Load Bearable Convex Polygon (LBCP), which provides a computationally efficient way to identify stable loading positions that guarantee no bin collapse. Additionally, we present Stable Rearrangement Planning (SRP), a module that rearranges existing items to accommodate new ones while maintaining overall stability. Extensive experiments on standard OBPP benchmarks demonstrate the efficiency and generalizability of our LBCP-based stability validation, as well as the superiority of SRP in finding the effort-saving rearrangement plans. Our method offers a robust and practical solution for automated packing in real-world industrial and logistics applications.
>
---
#### [new 006] MP-RBFN: Learning-based Vehicle Motion Primitives using Radial Basis Function Networks
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶中的运动规划任务，旨在解决传统方法在计算成本与轨迹形状约束上的不足。作者提出了MP-RBFN，结合采样法与径向基函数网络，高效学习高质量运动基元。实验表明其精度提升七倍，并实现快速推理，已开源应用。**

- **链接: [http://arxiv.org/pdf/2507.10047v1](http://arxiv.org/pdf/2507.10047v1)**

> **作者:** Marc Kaufeld; Mattia Piccinini; Johannes Betz
>
> **备注:** 8 pages, Submitted to the IEEE International Conference on Intelligent Transportation Systems (ITSC 2025), Australia
>
> **摘要:** This research introduces MP-RBFN, a novel formulation leveraging Radial Basis Function Networks for efficiently learning Motion Primitives derived from optimal control problems for autonomous driving. While traditional motion planning approaches based on optimization are highly accurate, they are often computationally prohibitive. In contrast, sampling-based methods demonstrate high performance but impose constraints on the geometric shape of trajectories. MP-RBFN combines the strengths of both by coupling the high-fidelity trajectory generation of sampling-based methods with an accurate description of vehicle dynamics. Empirical results show compelling performance compared to previous methods, achieving a precise description of motion primitives at low inference times. MP-RBFN yields a seven times higher accuracy in generating optimized motion primitives compared to existing semi-analytic approaches. We demonstrate the practical applicability of MP-RBFN for motion planning by integrating the method into a sampling-based trajectory planner. MP-RBFN is available as open-source software at https://github.com/TUM-AVS/RBFN-Motion-Primitives.
>
---
#### [new 007] MP1: Mean Flow Tames Policy Learning in 1-step for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务中的策略学习。旨在解决生成模型在扩散模型与流方法间的效率和约束权衡问题。提出MP1方法，结合MeanFlow范式与点云输入，实现单步生成动作轨迹，提升推理速度与精度，并通过CFG与轻量损失优化可控性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.10543v1](http://arxiv.org/pdf/2507.10543v1)**

> **作者:** Juyi Sheng; Ziyi Wang; Peiming Li; Mengyuan Liu
>
> **摘要:** In robot manipulation, robot learning has become a prevailing approach. However, generative models within this field face a fundamental trade-off between the slow, iterative sampling of diffusion models and the architectural constraints of faster Flow-based methods, which often rely on explicit consistency losses. To address these limitations, we introduce MP1, which pairs 3D point-cloud inputs with the MeanFlow paradigm to generate action trajectories in one network function evaluation (1-NFE). By directly learning the interval-averaged velocity via the MeanFlow Identity, our policy avoids any additional consistency constraints. This formulation eliminates numerical ODE-solver errors during inference, yielding more precise trajectories. MP1 further incorporates CFG for improved trajectory controllability while retaining 1-NFE inference without reintroducing structural constraints. Because subtle scene-context variations are critical for robot learning, especially in few-shot learning, we introduce a lightweight Dispersive Loss that repels state embeddings during training, boosting generalization without slowing inference. We validate our method on the Adroit and Meta-World benchmarks, as well as in real-world scenarios. Experimental results show MP1 achieves superior average task success rates, outperforming DP3 by 10.2% and FlowPolicy by 7.3%. Its average inference time is only 6.8 ms-19x faster than DP3 and nearly 2x faster than FlowPolicy. Our code is available at https://mp1-2254.github.io/.
>
---
#### [new 008] Informed Hybrid Zonotope-based Motion Planning Algorithm
- **分类: cs.RO**

- **简介: 该论文属于运动规划任务，旨在解决非凸自由空间中的最优路径规划难题。现有方法在狭窄区域或复杂场景中采样效率低，作者提出HZ-MP算法，结合混合zonotope表示与启发式采样策略，提升搜索效率和路径质量，并证明其完备性和收敛性。**

- **链接: [http://arxiv.org/pdf/2507.09309v1](http://arxiv.org/pdf/2507.09309v1)**

> **作者:** Peng Xie; Johannes Betz; Amr Alanwar
>
> **摘要:** Optimal path planning in nonconvex free spaces is notoriously challenging, as formulating such problems as mixed-integer linear programs (MILPs) is NP-hard. We propose HZ-MP, an informed Hybrid Zonotope-based Motion Planner, as an alternative approach that decomposes the obstacle-free space and performs low-dimensional face sampling guided by an ellipsotope heuristic, enabling focused exploration along promising transit regions. This structured exploration eliminates the excessive, unreachable sampling that degrades existing informed planners such as AIT* and EIT* in narrow gaps or boxed-goal scenarios. We prove that HZ-MP is probabilistically complete and asymptotically optimal. It converges to near-optimal trajectories in finite time and scales to high-dimensional cluttered scenes.
>
---
#### [new 009] End-to-End Generation of City-Scale Vectorized Maps by Crowdsourced Vehicles
- **分类: cs.RO**

- **简介: 论文任务为城市级高精度矢量地图生成，旨在解决传统方法成本高、效率低及单车辆感知精度不足的问题。工作提出EGC-VMAP框架，通过多车多时段数据融合与新型Transformer架构，实现端到端地图生成，提升准确性和鲁棒性，降低标注成本。**

- **链接: [http://arxiv.org/pdf/2507.08901v1](http://arxiv.org/pdf/2507.08901v1)**

> **作者:** Zebang Feng; Miao Fan; Bao Liu; Shengtong Xu; Haoyi Xiong
>
> **备注:** Accepted by ITSC'25
>
> **摘要:** High-precision vectorized maps are indispensable for autonomous driving, yet traditional LiDAR-based creation is costly and slow, while single-vehicle perception methods lack accuracy and robustness, particularly in adverse conditions. This paper introduces EGC-VMAP, an end-to-end framework that overcomes these limitations by generating accurate, city-scale vectorized maps through the aggregation of data from crowdsourced vehicles. Unlike prior approaches, EGC-VMAP directly fuses multi-vehicle, multi-temporal map elements perceived onboard vehicles using a novel Trip-Aware Transformer architecture within a unified learning process. Combined with hierarchical matching for efficient training and a multi-objective loss, our method significantly enhances map accuracy and structural robustness compared to single-vehicle baselines. Validated on a large-scale, multi-city real-world dataset, EGC-VMAP demonstrates superior performance, enabling a scalable, cost-effective solution for city-wide mapping with a reported 90\% reduction in manual annotation costs.
>
---
#### [new 010] IteraOptiRacing: A Unified Planning-Control Framework for Real-time Autonomous Racing for Iterative Optimal Performance
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于自动驾驶赛车任务，旨在解决多车竞速环境下实时路径规划与控制问题。作者提出IteraOptiRacing框架，基于i²LQR算法，统一处理避障与时间优化，实现低计算负担、可并行的实时轨迹生成，验证显示其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.09714v1](http://arxiv.org/pdf/2507.09714v1)**

> **作者:** Yifan Zeng; Yihan Li; Suiyi He; Koushil Sreenath; Jun Zeng
>
> **摘要:** This paper presents a unified planning-control strategy for competing with other racing cars called IteraOptiRacing in autonomous racing environments. This unified strategy is proposed based on Iterative Linear Quadratic Regulator for Iterative Tasks (i2LQR), which can improve lap time performance in the presence of surrounding racing obstacles. By iteratively using the ego car's historical data, both obstacle avoidance for multiple moving cars and time cost optimization are considered in this unified strategy, resulting in collision-free and time-optimal generated trajectories. The algorithm's constant low computation burden and suitability for parallel computing enable real-time operation in competitive racing scenarios. To validate its performance, simulations in a high-fidelity simulator are conducted with multiple randomly generated dynamic agents on the track. Results show that the proposed strategy outperforms existing methods across all randomly generated autonomous racing scenarios, enabling enhanced maneuvering for the ego racing car.
>
---
#### [new 011] Visual Homing in Outdoor Robots Using Mushroom Body Circuits and Learning Walks
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人视觉导航任务，旨在解决户外环境下自主机器人高效视觉归巢问题。受蚂蚁归巢机制启发，作者基于蘑菇体神经回路设计了一种轻量级模型，结合路径积分与学习行走策略，在真实环境中实现精准归巢。**

- **链接: [http://arxiv.org/pdf/2507.09725v1](http://arxiv.org/pdf/2507.09725v1)**

> **作者:** Gabriel G. Gattaux; Julien R. Serres; Franck Ruffier; Antoine Wystrach
>
> **备注:** Published by Springer Nature with the 14th bioinspired and biohybrid systems conference in Sheffield, and presented at the conference in July 2025
>
> **摘要:** Ants achieve robust visual homing with minimal sensory input and only a few learning walks, inspiring biomimetic solutions for autonomous navigation. While Mushroom Body (MB) models have been used in robotic route following, they have not yet been applied to visual homing. We present the first real-world implementation of a lateralized MB architecture for visual homing onboard a compact autonomous car-like robot. We test whether the sign of the angular path integration (PI) signal can categorize panoramic views, acquired during learning walks and encoded in the MB, into "goal on the left" and "goal on the right" memory banks, enabling robust homing in natural outdoor settings. We validate this approach through four incremental experiments: (1) simulation showing attractor-like nest dynamics; (2) real-world homing after decoupled learning walks, producing nest search behavior; (3) homing after random walks using noisy PI emulated with GPS-RTK; and (4) precise stopping-at-the-goal behavior enabled by a fifth MB Output Neuron (MBON) encoding goal-views to control velocity. This mimics the accurate homing behavior of ants and functionally resembles waypoint-based position control in robotics, despite relying solely on visual input. Operating at 8 Hz on a Raspberry Pi 4 with 32x32 pixel views and a memory footprint under 9 kB, our system offers a biologically grounded, resource-efficient solution for autonomous visual homing.
>
---
#### [new 012] Tactile-VLA: Unlocking Vision-Language-Action Model's Physical Knowledge for Tactile Generalization
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人与视觉语言动作（VLA）模型结合的任务，旨在解决接触丰富场景中机器人泛化能力差的问题。作者提出Tactile-VLA框架，融合视觉、语言、动作与触觉感知，通过混合控制策略和推理模块，实现基于触觉反馈的自适应操作，激活VLA模型中隐含的物理交互知识，完成零样本泛化任务。**

- **链接: [http://arxiv.org/pdf/2507.09160v1](http://arxiv.org/pdf/2507.09160v1)**

> **作者:** Jialei Huang; Shuo Wang; Fanqi Lin; Yihang Hu; Chuan Wen; Yang Gao
>
> **摘要:** Vision-Language-Action (VLA) models have shown remarkable achievements, driven by the rich implicit knowledge of their vision-language components. However, achieving generalist robotic agents demands precise grounding into physical interactions, especially in contact-rich scenarios where fine-grained force control is essential. We advance VLAs' implicit knowledge beyond identifying what to do, towards guiding how to physically interact with real world. This paper introduces Tactile-VLA, a novel framework that deeply fuses vision, language, action, and tactile sensing. This framework incorporates a hybrid position-force controller to translate the model's intentions into precise physical actions and a reasoning module that allows the robot to adapt its strategy based on tactile feedback. Experiments demonstrate Tactile-VLA's effectiveness and generalizability in three key aspects: (1) enabling tactile-aware instruction following, (2) utilizing tactile-relevant commonsense, and (3) facilitating adaptive tactile-involved reasoning. A key finding is that the VLM's prior knowledge already contains semantic understanding of physical interaction; by connecting it to the robot's tactile sensors with only a few demonstrations, we can activate this prior knowledge to achieve zero-shot generalization in contact-rich tasks.
>
---
#### [new 013] Robust RL Control for Bipedal Locomotion with Closed Kinematic Chains
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决双足机器人闭环运动链建模不准确导致的控制鲁棒性差问题。作者提出了一种考虑闭环运动链动力学的强化学习框架，并结合对称性损失、对抗训练和网络正则化提升策略鲁棒性，最终在多种地形上实现了稳定行走。**

- **链接: [http://arxiv.org/pdf/2507.10164v1](http://arxiv.org/pdf/2507.10164v1)**

> **作者:** Egor Maslennikov; Eduard Zaliaev; Nikita Dudorov; Oleg Shamanin; Karanov Dmitry; Gleb Afanasev; Alexey Burkov; Egor Lygin; Simeon Nedelchev; Evgeny Ponomarev
>
> **摘要:** Developing robust locomotion controllers for bipedal robots with closed kinematic chains presents unique challenges, particularly since most reinforcement learning (RL) approaches simplify these parallel mechanisms into serial models during training. We demonstrate that this simplification significantly impairs sim-to-real transfer by failing to capture essential aspects such as joint coupling, friction dynamics, and motor-space control characteristics. In this work, we present an RL framework that explicitly incorporates closed-chain dynamics and validate it on our custom-built robot TopA. Our approach enhances policy robustness through symmetry-aware loss functions, adversarial training, and targeted network regularization. Experimental results demonstrate that our integrated approach achieves stable locomotion across diverse terrains, significantly outperforming methods based on simplified kinematic models.
>
---
#### [new 014] Prompt Informed Reinforcement Learning for Visual Coverage Path Planning
- **分类: cs.RO; cs.MA**

- **简介: 论文提出Prompt-Informed Reinforcement Learning（PIRL），用于视觉覆盖路径规划任务，旨在通过结合大语言模型与强化学习，解决无人机在复杂环境中实现高效视觉覆盖、减少冗余和提升电池效率的问题。**

- **链接: [http://arxiv.org/pdf/2507.10284v1](http://arxiv.org/pdf/2507.10284v1)**

> **作者:** Venkat Margapuri
>
> **摘要:** Visual coverage path planning with unmanned aerial vehicles (UAVs) requires agents to strategically coordinate UAV motion and camera control to maximize coverage, minimize redundancy, and maintain battery efficiency. Traditional reinforcement learning (RL) methods rely on environment-specific reward formulations that lack semantic adaptability. This study proposes Prompt-Informed Reinforcement Learning (PIRL), a novel approach that integrates the zero-shot reasoning ability and in-context learning capability of large language models with curiosity-driven RL. PIRL leverages semantic feedback from an LLM, GPT-3.5, to dynamically shape the reward function of the Proximal Policy Optimization (PPO) RL policy guiding the agent in position and camera adjustments for optimal visual coverage. The PIRL agent is trained using OpenAI Gym and evaluated in various environments. Furthermore, the sim-to-real-like ability and zero-shot generalization of the agent are tested by operating the agent in Webots simulator which introduces realistic physical dynamics. Results show that PIRL outperforms multiple learning-based baselines such as PPO with static rewards, PPO with exploratory weight initialization, imitation learning, and an LLM-only controller. Across different environments, PIRL outperforms the best-performing baseline by achieving up to 14% higher visual coverage in OpenAI Gym and 27% higher in Webots, up to 25% higher battery efficiency, and up to 18\% lower redundancy, depending on the environment. The results highlight the effectiveness of LLM-guided reward shaping in complex spatial exploration tasks and suggest a promising direction for integrating natural language priors into RL for robotics.
>
---
#### [new 015] Unmanned Aerial Vehicle (UAV) Data-Driven Modeling Software with Integrated 9-Axis IMUGPS Sensor Fusion and Data Filtering Algorithm
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出了一种基于低成本传感器的无人机数据驱动建模软件，属无人机状态估计任务。为解决传感器数据精度低与更新频率不匹配问题，融合9轴IMU与GPS数据，采用四元数表示避免云台锁定，并结合滤波算法提升定位与姿态估计精度，实现无人机飞行状态的高精度实时可视化。**

- **链接: [http://arxiv.org/pdf/2507.09464v1](http://arxiv.org/pdf/2507.09464v1)**

> **作者:** Azfar Azdi Arfakhsyad; Aufa Nasywa Rahman; Larasati Kinanti; Ahmad Ataka Awwalur Rizqi; Hannan Nur Muhammad
>
> **备注:** 7 pages, 13 figures. Accepted to IEEE ICITEE 2023
>
> **摘要:** Unmanned Aerial Vehicles (UAV) have emerged as versatile platforms, driving the demand for accurate modeling to support developmental testing. This paper proposes data-driven modeling software for UAV. Emphasizes the utilization of cost-effective sensors to obtain orientation and location data subsequently processed through the application of data filtering algorithms and sensor fusion techniques to improve the data quality to make a precise model visualization on the software. UAV's orientation is obtained using processed Inertial Measurement Unit (IMU) data and represented using Quaternion Representation to avoid the gimbal lock problem. The UAV's location is determined by combining data from the Global Positioning System (GPS), which provides stable geographic coordinates but slower data update frequency, and the accelerometer, which has higher data update frequency but integrating it to get position data is unstable due to its accumulative error. By combining data from these two sensors, the software is able to calculate and continuously update the UAV's real-time position during its flight operations. The result shows that the software effectively renders UAV orientation and position with high degree of accuracy and fluidity
>
---
#### [new 016] TOP: Trajectory Optimization via Parallel Optimization towards Constant Time Complexity
- **分类: cs.RO**

- **简介: 该论文属于运动规划任务，旨在解决大规模轨迹优化效率低的问题。通过提出基于CADMM的并行优化框架，将轨迹分段并行处理，降低时间复杂度至O(1)。结合闭式解和数值解方法，提升优化速度与平滑性，实验证明在GPU上可高效处理上千段轨迹。**

- **链接: [http://arxiv.org/pdf/2507.10290v1](http://arxiv.org/pdf/2507.10290v1)**

> **作者:** Jiajun Yu; Nanhe Chen; Guodong Liu; Chao Xu; Fei Gao; Yanjun Cao
>
> **备注:** 8 pages, submitted to RA-L
>
> **摘要:** Optimization has been widely used to generate smooth trajectories for motion planning. However, existing trajectory optimization methods show weakness when dealing with large-scale long trajectories. Recent advances in parallel computing have accelerated optimization in some fields, but how to efficiently solve trajectory optimization via parallelism remains an open question. In this paper, we propose a novel trajectory optimization framework based on the Consensus Alternating Direction Method of Multipliers (CADMM) algorithm, which decomposes the trajectory into multiple segments and solves the subproblems in parallel. The proposed framework reduces the time complexity to O(1) per iteration to the number of segments, compared to O(N) of the state-of-the-art (SOTA) approaches. Furthermore, we introduce a closed-form solution that integrates convex linear and quadratic constraints to speed up the optimization, and we also present numerical solutions for general inequality constraints. A series of simulations and experiments demonstrate that our approach outperforms the SOTA approach in terms of efficiency and smoothness. Especially for a large-scale trajectory, with one hundred segments, achieving over a tenfold speedup. To fully explore the potential of our algorithm on modern parallel computing architectures, we deploy our framework on a GPU and show high performance with thousands of segments.
>
---
#### [new 017] Physics-Informed Neural Networks with Unscented Kalman Filter for Sensorless Joint Torque Estimation in Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决无关节扭矩传感器的人形机器人全身扭矩控制问题。通过结合物理信息神经网络（PINNs）进行摩擦建模与无迹卡尔曼滤波（UKF）进行扭矩估计，提升扭矩跟踪精度和系统鲁棒性。实验验证了方法在动态平衡任务中的有效性，并展示了其跨平台适应能力。**

- **链接: [http://arxiv.org/pdf/2507.10105v1](http://arxiv.org/pdf/2507.10105v1)**

> **作者:** Ines Sorrentino; Giulio Romualdi; Lorenzo Moretti; Silvio Traversaro; Daniele Pucci
>
> **摘要:** This paper presents a novel framework for whole-body torque control of humanoid robots without joint torque sensors, designed for systems with electric motors and high-ratio harmonic drives. The approach integrates Physics-Informed Neural Networks (PINNs) for friction modeling and Unscented Kalman Filtering (UKF) for joint torque estimation, within a real-time torque control architecture. PINNs estimate nonlinear static and dynamic friction from joint and motor velocity readings, capturing effects like motor actuation without joint movement. The UKF utilizes PINN-based friction estimates as direct measurement inputs, improving torque estimation robustness. Experimental validation on the ergoCub humanoid robot demonstrates improved torque tracking accuracy, enhanced energy efficiency, and superior disturbance rejection compared to the state-of-the-art Recursive Newton-Euler Algorithm (RNEA), using a dynamic balancing experiment. The framework's scalability is shown by consistent performance across robots with similar hardware but different friction characteristics, without re-identification. Furthermore, a comparative analysis with position control highlights the advantages of the proposed torque control approach. The results establish the method as a scalable and practical solution for sensorless torque control in humanoid robots, ensuring torque tracking, adaptability, and stability in dynamic environments.
>
---
#### [new 018] TruckV2X: A Truck-Centered Perception Dataset
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶感知任务，旨在解决卡车因体积大、拖车动态复杂导致的感知难题。作者构建了首个以卡车为中心的多模态、多智能体协同感知数据集TruckV2X，包含卡车、拖车、自动驾驶车辆和路侧单元，用于提升遮挡场景下的感知性能，并推动自动驾驶卡车系统落地。**

- **链接: [http://arxiv.org/pdf/2507.09505v1](http://arxiv.org/pdf/2507.09505v1)**

> **作者:** Tenghui Xie; Zhiying Song; Fuxi Wen; Jun Li; Guangzhao Liu; Zijian Zhao
>
> **摘要:** Autonomous trucking offers significant benefits, such as improved safety and reduced costs, but faces unique perception challenges due to trucks' large size and dynamic trailer movements. These challenges include extensive blind spots and occlusions that hinder the truck's perception and the capabilities of other road users. To address these limitations, cooperative perception emerges as a promising solution. However, existing datasets predominantly feature light vehicle interactions or lack multi-agent configurations for heavy-duty vehicle scenarios. To bridge this gap, we introduce TruckV2X, the first large-scale truck-centered cooperative perception dataset featuring multi-modal sensing (LiDAR and cameras) and multi-agent cooperation (tractors, trailers, CAVs, and RSUs). We further investigate how trucks influence collaborative perception needs, establishing performance benchmarks while suggesting research priorities for heavy vehicle perception. The dataset provides a foundation for developing cooperative perception systems with enhanced occlusion handling capabilities, and accelerates the deployment of multi-agent autonomous trucking systems. The TruckV2X dataset is available at https://huggingface.co/datasets/XieTenghu1/TruckV2X.
>
---
#### [new 019] PRAG: Procedural Action Generator
- **分类: cs.RO**

- **简介: 论文提出PRAG方法，用于生成多步骤、接触丰富的机器人操作任务。它通过符号和物理验证确保任务可解，输出可用于训练的高质量任务集，解决任务生成与验证问题。**

- **链接: [http://arxiv.org/pdf/2507.09167v1](http://arxiv.org/pdf/2507.09167v1)**

> **作者:** Michal Vavrecka; Radoslav Skoviera; Gabriela Sejnova; Karla Stepanova
>
> **摘要:** We present a novel approach for the procedural construction of multi-step contact-rich manipulation tasks in robotics. Our generator takes as input user-defined sets of atomic actions, objects, and spatial predicates and outputs solvable tasks of a given length for the selected robotic environment. The generator produces solvable tasks by constraining all possible (nonsolvable) combinations by symbolic and physical validation. The symbolic validation checks each generated sequence for logical and operational consistency, and also the suitability of object-predicate relations. Physical validation checks whether tasks can be solved in the selected robotic environment. Only the tasks that passed both validators are retained. The output from the generator can be directly interfaced with any existing framework for training robotic manipulation tasks, or it can be stored as a dataset of curated robotic tasks with detailed information about each task. This is beneficial for RL training as there are dense reward functions and initial and goal states paired with each subgoal. It allows the user to measure the semantic similarity of all generated tasks. We tested our generator on sequences of up to 15 actions resulting in millions of unique solvable multi-step tasks.
>
---
#### [new 020] Active Probing with Multimodal Predictions for Motion Planning
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于自动驾驶任务，旨在解决动态环境中因他车行为不确定性导致的决策难题。作者提出一种融合多模态预测与主动探测的统一框架，通过新型风险度量和主动策略减少预测歧义，提升路径规划安全性，并在仿真中验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.09822v1](http://arxiv.org/pdf/2507.09822v1)**

> **作者:** Darshan Gadginmath; Farhad Nawaz; Minjun Sung; Faizan M Tariq; Sangjae Bae; David Isele; Fabio Pasqualetti; Jovin Dsa
>
> **备注:** To appear at IROS '25. 8 pages. 3 tables. 6 figures
>
> **摘要:** Navigation in dynamic environments requires autonomous systems to reason about uncertainties in the behavior of other agents. In this paper, we introduce a unified framework that combines trajectory planning with multimodal predictions and active probing to enhance decision-making under uncertainty. We develop a novel risk metric that seamlessly integrates multimodal prediction uncertainties through mixture models. When these uncertainties follow a Gaussian mixture distribution, we prove that our risk metric admits a closed-form solution, and is always finite, thus ensuring analytical tractability. To reduce prediction ambiguity, we incorporate an active probing mechanism that strategically selects actions to improve its estimates of behavioral parameters of other agents, while simultaneously handling multimodal uncertainties. We extensively evaluate our framework in autonomous navigation scenarios using the MetaDrive simulation environment. Results demonstrate that our active probing approach successfully navigates complex traffic scenarios with uncertain predictions. Additionally, our framework shows robust performance across diverse traffic agent behavior models, indicating its broad applicability to real-world autonomous navigation challenges. Code and videos are available at https://darshangm.github.io/papers/active-probing-multimodal-predictions/.
>
---
#### [new 021] TGLD: A Trust-Aware Game-Theoretic Lane-Changing Decision Framework for Automated Vehicles in Heterogeneous Traffic
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文属于自动驾驶决策任务，旨在解决自动驾驶车辆（AV）在混杂交通中与人工驾驶车辆（HV）有效协作的问题。通过构建信任感知的博弈模型，动态评估HV信任水平，并优化AV变道策略，提升交互效率与安全性。**

- **链接: [http://arxiv.org/pdf/2507.10075v1](http://arxiv.org/pdf/2507.10075v1)**

> **作者:** Jie Pan; Tianyi Wang; Yangyang Wang; Junfeng Jiao; Christian Claudel
>
> **备注:** 6 pages, 7 figures, accepted for IEEE International Conference on Intelligent Transportation Systems (ITSC) 2025
>
> **摘要:** Automated vehicles (AVs) face a critical need to adopt socially compatible behaviors and cooperate effectively with human-driven vehicles (HVs) in heterogeneous traffic environment. However, most existing lane-changing frameworks overlook HVs' dynamic trust levels, limiting their ability to accurately predict human driver behaviors. To address this gap, this study proposes a trust-aware game-theoretic lane-changing decision (TGLD) framework. First, we formulate a multi-vehicle coalition game, incorporating fully cooperative interactions among AVs and partially cooperative behaviors from HVs informed by real-time trust evaluations. Second, we develop an online trust evaluation method to dynamically estimate HVs' trust levels during lane-changing interactions, guiding AVs to select context-appropriate cooperative maneuvers. Lastly, social compatibility objectives are considered by minimizing disruption to surrounding vehicles and enhancing the predictability of AV behaviors, thereby ensuring human-friendly and context-adaptive lane-changing strategies. A human-in-the-loop experiment conducted in a highway on-ramp merging scenario validates our TGLD approach. Results show that AVs can effectively adjust strategies according to different HVs' trust levels and driving styles. Moreover, incorporating a trust mechanism significantly improves lane-changing efficiency, maintains safety, and contributes to transparent and adaptive AV-HV interactions.
>
---
#### [new 022] Raci-Net: Ego-vehicle Odometry Estimation in Adverse Weather Conditions
- **分类: cs.RO; I.2**

- **简介: 该论文属于自动驾驶中的定位任务，旨在解决恶劣天气下传感器失效导致的定位不准问题。论文提出了Raci-Net模型，融合视觉、惯性与毫米波雷达数据，动态调整传感器贡献，提升定位鲁棒性与精度，并在Boreas数据集上验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.10376v1](http://arxiv.org/pdf/2507.10376v1)**

> **作者:** Mohammadhossein Talebi; Pragyan Dahal; Davide Possenti; Stefano Arrigoni; Francesco Braghin
>
> **备注:** 8 pages
>
> **摘要:** Autonomous driving systems are highly dependent on sensors like cameras, LiDAR, and inertial measurement units (IMU) to perceive the environment and estimate their motion. Among these sensors, perception-based sensors are not protected from harsh weather and technical failures. Although existing methods show robustness against common technical issues like rotational misalignment and disconnection, they often degrade when faced with dynamic environmental factors like weather conditions. To address these problems, this research introduces a novel deep learning-based motion estimator that integrates visual, inertial, and millimeter-wave radar data, utilizing each sensor strengths to improve odometry estimation accuracy and reliability under adverse environmental conditions such as snow, rain, and varying light. The proposed model uses advanced sensor fusion techniques that dynamically adjust the contributions of each sensor based on the current environmental condition, with radar compensating for visual sensor limitations in poor visibility. This work explores recent advancements in radar-based odometry and highlights that radar robustness in different weather conditions makes it a valuable component for pose estimation systems, specifically when visual sensors are degraded. Experimental results, conducted on the Boreas dataset, showcase the robustness and effectiveness of the model in both clear and degraded environments.
>
---
#### [new 023] Hand Gesture Recognition for Collaborative Robots Using Lightweight Deep Learning in Real-Time Robotic Systems
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决无需额外设备的自然机器人控制问题。作者设计了一个轻量级深度学习模型，可实时识别8种手势，用于控制协作机器人。模型仅1,103参数，经优化后大小仅7 KB，成功部署于UR5机器人，实现在ROS2框架下的高效交互。**

- **链接: [http://arxiv.org/pdf/2507.10055v1](http://arxiv.org/pdf/2507.10055v1)**

> **作者:** Muhtadin; I Wayan Agus Darmawan; Muhammad Hilmi Rusydiansyah; I Ketut Eddy Purnama; Chastine Fatichah; Mauridhi Hery Purnomo
>
> **摘要:** Direct and natural interaction is essential for intuitive human-robot collaboration, eliminating the need for additional devices such as joysticks, tablets, or wearable sensors. In this paper, we present a lightweight deep learning-based hand gesture recognition system that enables humans to control collaborative robots naturally and efficiently. This model recognizes eight distinct hand gestures with only 1,103 parameters and a compact size of 22 KB, achieving an accuracy of 93.5%. To further optimize the model for real-world deployment on edge devices, we applied quantization and pruning using TensorFlow Lite, reducing the final model size to just 7 KB. The system was successfully implemented and tested on a Universal Robot UR5 collaborative robot within a real-time robotic framework based on ROS2. The results demonstrate that even extremely lightweight models can deliver accurate and responsive hand gesture-based control for collaborative robots, opening new possibilities for natural human-robot interaction in constrained environments.
>
---
#### [new 024] Unified Linear Parametric Map Modeling and Perception-aware Trajectory Planning for Mobile Robotics
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于移动机器人自主导航任务，旨在解决复杂环境中感知与规划的高效性及安全性问题。论文提出了RMRP建模方法和RPATR规划框架，统一地图表示并提升轨迹规划的鲁棒性与感知意识，适用于无人机和地面机器人。**

- **链接: [http://arxiv.org/pdf/2507.09340v1](http://arxiv.org/pdf/2507.09340v1)**

> **作者:** Hongyu Nie; Xingyu Li; Xu Liu; Zhaotong Tan; Sen Mei; Wenbo Su
>
> **备注:** Submitted to IEEE Transactions on Robotics (TRO) in July 2025
>
> **摘要:** Autonomous navigation in mobile robots, reliant on perception and planning, faces major hurdles in large-scale, complex environments. These include heavy computational burdens for mapping, sensor occlusion failures for UAVs, and traversal challenges on irregular terrain for UGVs, all compounded by a lack of perception-aware strategies. To address these challenges, we introduce Random Mapping and Random Projection (RMRP). This method constructs a lightweight linear parametric map by first mapping data to a high-dimensional space, followed by a sparse random projection for dimensionality reduction. Our novel Residual Energy Preservation Theorem provides theoretical guarantees for this process, ensuring critical geometric properties are preserved. Based on this map, we propose the RPATR (Robust Perception-Aware Trajectory Planner) framework. For UAVs, our method unifies grid and Euclidean Signed Distance Field (ESDF) maps. The front-end uses an analytical occupancy gradient to refine initial paths for safety and smoothness, while the back-end uses a closed-form ESDF for trajectory optimization. Leveraging the trained RMRP model's generalization, the planner predicts unobserved areas for proactive navigation. For UGVs, the model characterizes terrain and provides closed-form gradients, enabling online planning to circumvent large holes. Validated in diverse scenarios, our framework demonstrates superior mapping performance in time, memory, and accuracy, and enables computationally efficient, safe navigation for high-speed UAVs and UGVs. The code will be released to foster community collaboration.
>
---
#### [new 025] Simulations and experiments with assemblies of fiber-reinforced soft actuators
- **分类: cs.RO**

- **简介: 该论文旨在解决软连续臂（SCAs）实际控制困难的问题。任务是开发一种仿真框架，用于模拟由纤维增强弹性封装（FREEs）模块组装的SCAs，并结合视频追踪系统进行实验验证与控制设计，以提升其在实际应用中的可行性。**

- **链接: [http://arxiv.org/pdf/2507.10121v1](http://arxiv.org/pdf/2507.10121v1)**

> **作者:** Seung Hyun Kim; Jiamiao Guo; Arman Tekinalp; Heng-Sheng Chang; Ugur Akcal; Tixian Wang; Darren Biskup; Benjamin Walt; Girish Chowdhary; Girish Krishnan; Prashant G. Mehta; Mattia Gazzola
>
> **备注:** 8 pages, 4 figures This work has been submitted to the IEEE for possible publication
>
> **摘要:** Soft continuum arms (SCAs) promise versatile manipulation through mechanical compliance, for assistive devices, agriculture, search applications, or surgery. However, SCAs' real-world use is challenging, partly due to their hard-to-control non-linear behavior. Here, a simulation framework for SCAs modularly assembled out of fiber reinforced elastomeric enclosures (FREEs) is developed and integrated with a video-tracking system for experimental testing and control design.
>
---
#### [new 026] Demonstrating the Octopi-1.5 Visual-Tactile-Language Model
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人感知任务，旨在解决多模态（视觉-触觉-语言）融合与推理问题。作者提出了Octopi-1.5模型，增强其对多部位触觉信号处理能力，并引入RAG模块以实时学习新物体。通过TMI手持触觉接口实现无机器人交互演示，展示了其在物体识别、操作建议及在线学习方面的能力。**

- **链接: [http://arxiv.org/pdf/2507.09985v1](http://arxiv.org/pdf/2507.09985v1)**

> **作者:** Samson Yu; Kelvin Lin; Harold Soh
>
> **备注:** Published at R:SS 2025
>
> **摘要:** Touch is recognized as a vital sense for humans and an equally important modality for robots, especially for dexterous manipulation, material identification, and scenarios involving visual occlusion. Building upon very recent work in touch foundation models, this demonstration will feature Octopi-1.5, our latest visual-tactile-language model. Compared to its predecessor, Octopi-1.5 introduces the ability to process tactile signals from multiple object parts and employs a simple retrieval-augmented generation (RAG) module to improve performance on tasks and potentially learn new objects on-the-fly. The system can be experienced live through a new handheld tactile-enabled interface, the TMI, equipped with GelSight and TAC-02 tactile sensors. This convenient and accessible setup allows users to interact with Octopi-1.5 without requiring a robot. During the demonstration, we will showcase Octopi-1.5 solving tactile inference tasks by leveraging tactile inputs and commonsense knowledge. For example, in a Guessing Game, Octopi-1.5 will identify objects being grasped and respond to follow-up queries about how to handle it (e.g., recommending careful handling for soft fruits). We also plan to demonstrate Octopi-1.5's RAG capabilities by teaching it new items. With live interactions, this demonstration aims to highlight both the progress and limitations of VTLMs such as Octopi-1.5 and to foster further interest in this exciting field. Code for Octopi-1.5 and design files for the TMI gripper are available at https://github.com/clear-nus/octopi-1.5.
>
---
#### [new 027] Finetuning Deep Reinforcement Learning Policies with Evolutionary Strategies for Control of Underactuated Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在提升欠驱动机器人控制策略的性能与鲁棒性。论文提出先用SAC算法训练初始策略，再通过SNES进化策略进行微调，以逼近复杂评分目标。实验表明该方法在IROS 2024比赛中表现优异，超越基线模型。**

- **链接: [http://arxiv.org/pdf/2507.10030v1](http://arxiv.org/pdf/2507.10030v1)**

> **作者:** Marco Calì; Alberto Sinigaglia; Niccolò Turcato; Ruggero Carli; Gian Antonio Susto
>
> **摘要:** Deep Reinforcement Learning (RL) has emerged as a powerful method for addressing complex control problems, particularly those involving underactuated robotic systems. However, in some cases, policies may require refinement to achieve optimal performance and robustness aligned with specific task objectives. In this paper, we propose an approach for fine-tuning Deep RL policies using Evolutionary Strategies (ES) to enhance control performance for underactuated robots. Our method involves initially training an RL agent with Soft-Actor Critic (SAC) using a surrogate reward function designed to approximate complex specific scoring metrics. We subsequently refine this learned policy through a zero-order optimization step employing the Separable Natural Evolution Strategy (SNES), directly targeting the original score. Experimental evaluations conducted in the context of the 2nd AI Olympics with RealAIGym at IROS 2024 demonstrate that our evolutionary fine-tuning significantly improves agent performance while maintaining high robustness. The resulting controllers outperform established baselines, achieving competitive scores for the competition tasks.
>
---
#### [new 028] Ariel Explores: Vision-based underwater exploration and inspection via generalist drone-level autonomy
- **分类: cs.RO**

- **简介: 论文提出了一种基于视觉的水下探索与检测自主解决方案，集成于定制水下机器人Ariel中。该方案通过多相机和IMU实现状态估计，并结合学习方法提升鲁棒性，旨在解决水下环境中的自主导航与检查问题。**

- **链接: [http://arxiv.org/pdf/2507.10003v1](http://arxiv.org/pdf/2507.10003v1)**

> **作者:** Mohit Singh; Mihir Dharmadhikari; Kostas Alexis
>
> **备注:** Presented at the 2025 IEEE ICRA Workshop on Field Robotics
>
> **摘要:** This work presents a vision-based underwater exploration and inspection autonomy solution integrated into Ariel, a custom vision-driven underwater robot. Ariel carries a $5$ camera and IMU based sensing suite, enabling a refraction-aware multi-camera visual-inertial state estimation method aided by a learning-based proprioceptive robot velocity prediction method that enhances robustness against visual degradation. Furthermore, our previously developed and extensively field-verified autonomous exploration and general visual inspection solution is integrated on Ariel, providing aerial drone-level autonomy underwater. The proposed system is field-tested in a submarine dry dock in Trondheim under challenging visual conditions. The field demonstration shows the robustness of the state estimation solution and the generalizability of the path planning techniques across robot embodiments.
>
---
#### [new 029] REACT: Real-time Entanglement-Aware Coverage Path Planning for Tethered Underwater Vehicles
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于水下机器人路径规划任务，旨在解决系缆水下机器人在复杂结构中检测时易发生缆绳缠绕的问题。论文提出了REACT框架，利用几何模型和实时重规划策略避免缠绕，实现了安全高效的全覆盖检测。**

- **链接: [http://arxiv.org/pdf/2507.10204v1](http://arxiv.org/pdf/2507.10204v1)**

> **作者:** Abdelhakim Amer; Mohit Mehindratta; Yury Brodskiy; Bilal Wehbe; Erdal Kayacan
>
> **摘要:** Inspection of complex underwater structures with tethered underwater vehicles is often hindered by the risk of tether entanglement. We propose REACT (real-time entanglement-aware coverage path planning for tethered underwater vehicles), a framework designed to overcome this limitation. REACT comprises a fast geometry-based tether model using the signed distance field (SDF) map for accurate, real-time simulation of taut tether configurations around arbitrary structures in 3D. This model enables an efficient online replanning strategy by enforcing a maximum tether length constraint, thereby actively preventing entanglement. By integrating REACT into a coverage path planning framework, we achieve safe and optimal inspection paths, previously challenging due to tether constraints. The complete REACT framework's efficacy is validated in a pipe inspection scenario, demonstrating safe, entanglement-free navigation and full-coverage inspection. Simulation results show that REACT achieves complete coverage while maintaining tether constraints and completing the total mission 20% faster than conventional planners, despite a longer inspection time due to proactive avoidance of entanglement that eliminates extensive post-mission disentanglement. Real-world experiments confirm these benefits, where REACT completes the full mission, while the baseline planner fails due to physical tether entanglement.
>
---
#### [new 030] OTAS: Open-vocabulary Token Alignment for Outdoor Segmentation
- **分类: cs.RO**

- **简介: 论文提出OTAS，用于开放词汇户外语义分割。该方法通过聚类预训练视觉模型输出的语义结构，并与语言对齐，实现无需微调的零样本分割。解决了户外场景中语义模糊、类别边界不清晰的问题，支持多视角几何一致的2D和3D分割，适用于机器人应用。**

- **链接: [http://arxiv.org/pdf/2507.08851v1](http://arxiv.org/pdf/2507.08851v1)**

> **作者:** Simon Schwaiger; Stefan Thalhammer; Wilfried Wöber; Gerald Steinbauer-Wagner
>
> **摘要:** Understanding open-world semantics is critical for robotic planning and control, particularly in unstructured outdoor environments. Current vision-language mapping approaches rely on object-centric segmentation priors, which often fail outdoors due to semantic ambiguities and indistinct semantic class boundaries. We propose OTAS - an Open-vocabulary Token Alignment method for Outdoor Segmentation. OTAS overcomes the limitations of open-vocabulary segmentation models by extracting semantic structure directly from the output tokens of pretrained vision models. By clustering semantically similar structures across single and multiple views and grounding them in language, OTAS reconstructs a geometrically consistent feature field that supports open-vocabulary segmentation queries. Our method operates zero-shot, without scene-specific fine-tuning, and runs at up to ~17 fps. OTAS provides a minor IoU improvement over fine-tuned and open-vocabulary 2D segmentation methods on the Off-Road Freespace Detection dataset. Our model achieves up to a 151% IoU improvement over open-vocabulary mapping methods in 3D segmentation on TartanAir. Real-world reconstructions demonstrate OTAS' applicability to robotic applications. The code and ROS node will be made publicly available upon paper acceptance.
>
---
#### [new 031] C-ZUPT: Stationarity-Aided Aerial Hovering
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出C-ZUPT方法，用于空中悬停的导航与控制任务，旨在解决惯性传感器导致的状态估计漂移问题。通过定义不确定性阈值，在准静态平衡时提供速度修正，减少漂移和控制能耗，提升飞行稳定性和续航能力。**

- **链接: [http://arxiv.org/pdf/2507.09344v1](http://arxiv.org/pdf/2507.09344v1)**

> **作者:** Daniel Engelsman; Itzik Klein
>
> **备注:** 14 Pages, 16 Figures, 9 Tables
>
> **摘要:** Autonomous systems across diverse domains have underscored the need for drift-resilient state estimation. Although satellite-based positioning and cameras are widely used, they often suffer from limited availability in many environments. As a result, positioning must rely solely on inertial sensors, leading to rapid accuracy degradation over time due to sensor biases and noise. To counteract this, alternative update sources-referred to as information aiding-serve as anchors of certainty. Among these, the zero-velocity update (ZUPT) is particularly effective in providing accurate corrections during stationary intervals, though it is restricted to surface-bound platforms. This work introduces a controlled ZUPT (C-ZUPT) approach for aerial navigation and control, independent of surface contact. By defining an uncertainty threshold, C-ZUPT identifies quasi-static equilibria to deliver precise velocity updates to the estimation filter. Extensive validation confirms that these opportunistic, high-quality updates significantly reduce inertial drift and control effort. As a result, C-ZUPT mitigates filter divergence and enhances navigation stability, enabling more energy-efficient hovering and substantially extending sustained flight-key advantages for resource-constrained aerial systems.
>
---
#### [new 032] Towards Human-level Dexterity via Robot Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，旨在解决实现人类级别灵巧操作的问题。论文工作包括开发基于强化学习和模仿学习的方法，结合结构化探索与视觉-触觉演示，提升多指机械手的灵巧操作能力。**

- **链接: [http://arxiv.org/pdf/2507.09117v1](http://arxiv.org/pdf/2507.09117v1)**

> **作者:** Gagan Khandate
>
> **备注:** PhD thesis
>
> **摘要:** Dexterous intelligence -- the ability to perform complex interactions with multi-fingered hands -- is a pinnacle of human physical intelligence and emergent higher-order cognitive skills. However, contrary to Moravec's paradox, dexterous intelligence in humans appears simple only superficially. Many million years were spent co-evolving the human brain and hands including rich tactile sensing. Achieving human-level dexterity with robotic hands has long been a fundamental goal in robotics and represents a critical milestone toward general embodied intelligence. In this pursuit, computational sensorimotor learning has made significant progress, enabling feats such as arbitrary in-hand object reorientation. However, we observe that achieving higher levels of dexterity requires overcoming very fundamental limitations of computational sensorimotor learning. I develop robot learning methods for highly dexterous multi-fingered manipulation by directly addressing these limitations at their root cause. Chiefly, through key studies, this disseration progressively builds an effective framework for reinforcement learning of dexterous multi-fingered manipulation skills. These methods adopt structured exploration, effectively overcoming the limitations of random exploration in reinforcement learning. The insights gained culminate in a highly effective reinforcement learning that incorporates sampling-based planning for direct exploration. Additionally, this thesis explores a new paradigm of using visuo-tactile human demonstrations for dexterity, introducing corresponding imitation learning techniques.
>
---
#### [new 033] Foundation Model Driven Robotics: A Comprehensive Review
- **分类: cs.RO**

- **简介: 该论文综述了基础模型（如大语言模型和视觉-语言模型）在机器人领域的应用，探讨其在感知、规划、控制和人机交互方面的潜力。论文旨在分析当前技术的优势与瓶颈，如多模态推理、场景生成与现实迁移等问题，提出未来研究方向，以实现更可靠、可解释和具身化的机器人系统。**

- **链接: [http://arxiv.org/pdf/2507.10087v1](http://arxiv.org/pdf/2507.10087v1)**

> **作者:** Muhammad Tayyab Khan; Ammar Waheed
>
> **摘要:** The rapid emergence of foundation models, particularly Large Language Models (LLMs) and Vision-Language Models (VLMs), has introduced a transformative paradigm in robotics. These models offer powerful capabilities in semantic understanding, high-level reasoning, and cross-modal generalization, enabling significant advances in perception, planning, control, and human-robot interaction. This critical review provides a structured synthesis of recent developments, categorizing applications across simulation-driven design, open-world execution, sim-to-real transfer, and adaptable robotics. Unlike existing surveys that emphasize isolated capabilities, this work highlights integrated, system-level strategies and evaluates their practical feasibility in real-world environments. Key enabling trends such as procedural scene generation, policy generalization, and multimodal reasoning are discussed alongside core bottlenecks, including limited embodiment, lack of multimodal data, safety risks, and computational constraints. Through this lens, this paper identifies both the architectural strengths and critical limitations of foundation model-based robotics, highlighting open challenges in real-time operation, grounding, resilience, and trust. The review concludes with a roadmap for future research aimed at bridging semantic reasoning and physical intelligence through more robust, interpretable, and embodied models.
>
---
#### [new 034] Polygonal Obstacle Avoidance Combining Model Predictive Control and Fuzzy Logic
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文属于移动机器人导航任务，旨在解决在离散成本地图环境中实现障碍规避的模型预测控制（MPC）兼容性问题。工作内容是将多边形障碍表示为半空间交集，利用模糊逻辑将逻辑运算符转化为连续可微约束，从而嵌入MPC框架，实现平滑避障轨迹规划。**

- **链接: [http://arxiv.org/pdf/2507.10310v1](http://arxiv.org/pdf/2507.10310v1)**

> **作者:** Michael Schröder; Eric Schöneberg; Daniel Görges; Hans D. Schotten
>
> **摘要:** In practice, navigation of mobile robots in confined environments is often done using a spatially discrete cost-map to represent obstacles. Path following is a typical use case for model predictive control (MPC), but formulating constraints for obstacle avoidance is challenging in this case. Typically the cost and constraints of an MPC problem are defined as closed-form functions and typical solvers work best with continuously differentiable functions. This is contrary to spatially discrete occupancy grid maps, in which a grid's value defines the cost associated with occupancy. This paper presents a way to overcome this compatibility issue by re-formulating occupancy grid maps to continuously differentiable functions to be embedded into the MPC scheme as constraints. Each obstacle is defined as a polygon -- an intersection of half-spaces. Any half-space is a linear inequality representing one edge of a polygon. Using AND and OR operators, the combined set of all obstacles and therefore the obstacle avoidance constraints can be described. The key contribution of this paper is the use of fuzzy logic to re-formulate such constraints that include logical operators as inequality constraints which are compatible with standard MPC formulation. The resulting MPC-based trajectory planner is successfully tested in simulation. This concept is also applicable outside of navigation tasks to implement logical or verbal constraints in MPC.
>
---
#### [new 035] AdvGrasp: Adversarial Attacks on Robotic Grasping from a Physical Perspective
- **分类: cs.RO; cs.CR**

- **简介: 该论文属于机器人抓取任务，旨在评估和提升抓取系统的鲁棒性。针对现有研究忽略物理原理的问题，论文提出了AdvGrasp框架，通过形状变形攻击抓取的两个核心指标——举升能力和抓取稳定性，从而生成对抗物体以验证系统性能。**

- **链接: [http://arxiv.org/pdf/2507.09857v1](http://arxiv.org/pdf/2507.09857v1)**

> **作者:** Xiaofei Wang; Mingliang Han; Tianyu Hao; Cegang Li; Yunbo Zhao; Keke Tang
>
> **备注:** IJCAI'2025
>
> **摘要:** Adversarial attacks on robotic grasping provide valuable insights into evaluating and improving the robustness of these systems. Unlike studies that focus solely on neural network predictions while overlooking the physical principles of grasping, this paper introduces AdvGrasp, a framework for adversarial attacks on robotic grasping from a physical perspective. Specifically, AdvGrasp targets two core aspects: lift capability, which evaluates the ability to lift objects against gravity, and grasp stability, which assesses resistance to external disturbances. By deforming the object's shape to increase gravitational torque and reduce stability margin in the wrench space, our method systematically degrades these two key grasping metrics, generating adversarial objects that compromise grasp performance. Extensive experiments across diverse scenarios validate the effectiveness of AdvGrasp, while real-world validations demonstrate its robustness and practical applicability
>
---
#### [new 036] Constrained Style Learning from Imperfect Demonstrations under Task Optimality
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决从不完美示范中学习风格化动作的问题。现有方法在示范不完整时易牺牲任务性能来提升风格，本文提出将问题建模为约束马尔可夫决策过程，通过自适应调整拉格朗日乘子，在保持任务性能的同时学习风格化运动，实验证明了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.09371v1](http://arxiv.org/pdf/2507.09371v1)**

> **作者:** Kehan Wen; Chenhao Li; Junzhe He; Marco Hutter
>
> **备注:** This paper is under review
>
> **摘要:** Learning from demonstration has proven effective in robotics for acquiring natural behaviors, such as stylistic motions and lifelike agility, particularly when explicitly defining style-oriented reward functions is challenging. Synthesizing stylistic motions for real-world tasks usually requires balancing task performance and imitation quality. Existing methods generally depend on expert demonstrations closely aligned with task objectives. However, practical demonstrations are often incomplete or unrealistic, causing current methods to boost style at the expense of task performance. To address this issue, we propose formulating the problem as a constrained Markov Decision Process (CMDP). Specifically, we optimize a style-imitation objective with constraints to maintain near-optimal task performance. We introduce an adaptively adjustable Lagrangian multiplier to guide the agent to imitate demonstrations selectively, capturing stylistic nuances without compromising task performance. We validate our approach across multiple robotic platforms and tasks, demonstrating both robust task performance and high-fidelity style learning. On ANYmal-D hardware we show a 14.5% drop in mechanical energy and a more agile gait pattern, showcasing real-world benefits.
>
---
#### [new 037] Influence of Static and Dynamic Downwash Interactions on Multi-Quadrotor Systems
- **分类: cs.RO**

- **简介: 该论文研究多旋翼无人机间的下洗气流相互作用，旨在解决因下洗效应导致的系统不稳定和性能下降问题。通过力和扭矩测量及粒子图像测速技术，分析单个与多个无人机配置下的下洗特性，为优化编队和提升控制鲁棒性提供物理依据。**

- **链接: [http://arxiv.org/pdf/2507.09463v1](http://arxiv.org/pdf/2507.09463v1)**

> **作者:** Anoop Kiran; Nora Ayanian; Kenneth Breuer
>
> **备注:** Accepted for publication in Robotics: Science and Systems (RSS) 2025, 12 pages, 16 figures
>
> **摘要:** Flying multiple quadrotors in close proximity presents a significant challenge due to complex aerodynamic interactions, particularly downwash effects that are known to destabilize vehicles and degrade performance. Traditionally, multi-quadrotor systems rely on conservative strategies, such as collision avoidance zones around the robot volume, to circumvent this effect. This restricts their capabilities by requiring a large volume for the operation of a multi-quadrotor system, limiting their applicability in dense environments. This work provides a comprehensive, data-driven analysis of the downwash effect, with a focus on characterizing, analyzing, and understanding forces, moments, and velocities in both single and multi-quadrotor configurations. We use measurements of forces and torques to characterize vehicle interactions, and particle image velocimetry (PIV) to quantify the spatial features of the downwash wake for a single quadrotor and an interacting pair of quadrotors. This data can be used to inform physics-based strategies for coordination, leverage downwash for optimized formations, expand the envelope of operation, and improve the robustness of multi-quadrotor control.
>
---
#### [new 038] Scene-Aware Conversational ADAS with Generative AI for Real-Time Driver Assistance
- **分类: cs.RO; cs.AI; cs.CV; cs.HC**

- **简介: 论文提出SC-ADAS框架，将生成式AI与ADAS结合，实现基于场景感知的自然语言交互。解决当前ADAS缺乏语境理解和对话能力的问题，支持多轮对话与实时驾驶辅助决策，提升系统灵活性与适应性。**

- **链接: [http://arxiv.org/pdf/2507.10500v1](http://arxiv.org/pdf/2507.10500v1)**

> **作者:** Kyungtae Han; Yitao Chen; Rohit Gupta; Onur Altintas
>
> **摘要:** While autonomous driving technologies continue to advance, current Advanced Driver Assistance Systems (ADAS) remain limited in their ability to interpret scene context or engage with drivers through natural language. These systems typically rely on predefined logic and lack support for dialogue-based interaction, making them inflexible in dynamic environments or when adapting to driver intent. This paper presents Scene-Aware Conversational ADAS (SC-ADAS), a modular framework that integrates Generative AI components including large language models, vision-to-text interpretation, and structured function calling to enable real-time, interpretable, and adaptive driver assistance. SC-ADAS supports multi-turn dialogue grounded in visual and sensor context, allowing natural language recommendations and driver-confirmed ADAS control. Implemented in the CARLA simulator with cloud-based Generative AI, the system executes confirmed user intents as structured ADAS commands without requiring model fine-tuning. We evaluate SC-ADAS across scene-aware, conversational, and revisited multi-turn interactions, highlighting trade-offs such as increased latency from vision-based context retrieval and token growth from accumulated dialogue history. These results demonstrate the feasibility of combining conversational reasoning, scene perception, and modular ADAS control to support the next generation of intelligent driver assistance.
>
---
#### [new 039] Real-Time Adaptive Motion Planning via Point Cloud-Guided, Energy-Based Diffusion and Potential Fields
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人运动规划任务，旨在解决复杂环境中实时轨迹生成与避障问题。针对追逃场景中障碍物部分可观的情况，提出结合能量扩散模型与势场方法的框架，直接处理点云数据，实现高效、鲁棒的实时运动规划。**

- **链接: [http://arxiv.org/pdf/2507.09383v1](http://arxiv.org/pdf/2507.09383v1)**

> **作者:** Wondmgezahu Teshome; Kian Behzad; Octavia Camps; Michael Everett; Milad Siami; Mario Sznaier
>
> **备注:** Accepted to IEEE RA-L 2025
>
> **摘要:** Motivated by the problem of pursuit-evasion, we present a motion planning framework that combines energy-based diffusion models with artificial potential fields for robust real time trajectory generation in complex environments. Our approach processes obstacle information directly from point clouds, enabling efficient planning without requiring complete geometric representations. The framework employs classifier-free guidance training and integrates local potential fields during sampling to enhance obstacle avoidance. In dynamic scenarios, the system generates initial trajectories using the diffusion model and continuously refines them through potential field-based adaptation, demonstrating effective performance in pursuit-evasion scenarios with partial pursuer observability.
>
---
#### [new 040] Unscented Kalman Filter with a Nonlinear Propagation Model for Navigation Applications
- **分类: cs.RO; eess.SP**

- **简介: 论文研究任务为导航中的状态估计问题，旨在提升非线性滤波精度。针对传统方法在预测均值和协方差时的不足，提出一种基于非线性传播模型的无迹卡尔曼滤波方法，并通过自主水下机器人实测数据验证其有效性。**

- **链接: [http://arxiv.org/pdf/2507.10082v1](http://arxiv.org/pdf/2507.10082v1)**

> **作者:** Amit Levy; Itzik Klein
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** The unscented Kalman filter is a nonlinear estimation algorithm commonly used in navigation applications. The prediction of the mean and covariance matrix is crucial to the stable behavior of the filter. This prediction is done by propagating the sigma points according to the dynamic model at hand. In this paper, we introduce an innovative method to propagate the sigma points according to the nonlinear dynamic model of the navigation error state vector. This improves the filter accuracy and navigation performance. We demonstrate the benefits of our proposed approach using real sensor data recorded by an autonomous underwater vehicle during several scenarios.
>
---
#### [new 041] Multi-residual Mixture of Experts Learning for Cooperative Control in Multi-vehicle Systems
- **分类: cs.RO; cs.AI; cs.LG; cs.MA; cs.SY; eess.SY**

- **简介: 该论文属于交通控制任务，旨在解决自动驾驶车辆在复杂多样交通场景中协同控制策略泛化性差的问题。作者提出了MRMEL框架，结合残差强化学习与专家混合模型，动态选择并优化控制策略，提升了多车系统在不同城市交叉口生态驾驶的性能，有效降低了车辆排放。**

- **链接: [http://arxiv.org/pdf/2507.09836v1](http://arxiv.org/pdf/2507.09836v1)**

> **作者:** Vindula Jayawardana; Sirui Li; Yashar Farid; Cathy Wu
>
> **摘要:** Autonomous vehicles (AVs) are becoming increasingly popular, with their applications now extending beyond just a mode of transportation to serving as mobile actuators of a traffic flow to control flow dynamics. This contrasts with traditional fixed-location actuators, such as traffic signals, and is referred to as Lagrangian traffic control. However, designing effective Lagrangian traffic control policies for AVs that generalize across traffic scenarios introduces a major challenge. Real-world traffic environments are highly diverse, and developing policies that perform robustly across such diverse traffic scenarios is challenging. It is further compounded by the joint complexity of the multi-agent nature of traffic systems, mixed motives among participants, and conflicting optimization objectives subject to strict physical and external constraints. To address these challenges, we introduce Multi-Residual Mixture of Expert Learning (MRMEL), a novel framework for Lagrangian traffic control that augments a given suboptimal nominal policy with a learned residual while explicitly accounting for the structure of the traffic scenario space. In particular, taking inspiration from residual reinforcement learning, MRMEL augments a suboptimal nominal AV control policy by learning a residual correction, but at the same time dynamically selects the most suitable nominal policy from a pool of nominal policies conditioned on the traffic scenarios and modeled as a mixture of experts. We validate MRMEL using a case study in cooperative eco-driving at signalized intersections in Atlanta, Dallas Fort Worth, and Salt Lake City, with real-world data-driven traffic scenarios. The results show that MRMEL consistently yields superior performance-achieving an additional 4%-9% reduction in aggregate vehicle emissions relative to the strongest baseline in each setting.
>
---
#### [new 042] AirScape: An Aerial Generative World Model with Motion Controllability
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人空间预测任务，旨在解决飞行器如何基于视觉输入和运动意图预测三维空间中动作结果的问题。作者构建了包含11k视频-意图对的数据集，并提出AirScape模型，通过两阶段训练实现具有运动可控性的空中世界建模。**

- **链接: [http://arxiv.org/pdf/2507.08885v1](http://arxiv.org/pdf/2507.08885v1)**

> **作者:** Baining Zhao; Rongze Tang; Mingyuan Jia; Ziyou Wang; Fanghang Man; Xin Zhang; Yu Shang; Weichen Zhang; Chen Gao; Wei Wu; Xin Wang; Xinlei Chen; Yong Li
>
> **摘要:** How to enable robots to predict the outcomes of their own motion intentions in three-dimensional space has been a fundamental problem in embodied intelligence. To explore more general spatial imagination capabilities, here we present AirScape, the first world model designed for six-degree-of-freedom aerial agents. AirScape predicts future observation sequences based on current visual inputs and motion intentions. Specifically, we construct an dataset for aerial world model training and testing, which consists of 11k video-intention pairs. This dataset includes first-person-view videos capturing diverse drone actions across a wide range of scenarios, with over 1,000 hours spent annotating the corresponding motion intentions. Then we develop a two-phase training schedule to train a foundation model -- initially devoid of embodied spatial knowledge -- into a world model that is controllable by motion intentions and adheres to physical spatio-temporal constraints.
>
---
#### [new 043] On the Importance of Neural Membrane Potential Leakage for LIDAR-based Robot Obstacle Avoidance using Spiking Neural Networks
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究基于激光雷达（LIDAR）数据的机器人避障任务，探索脉冲神经网络（SNN）中神经元膜电位泄漏对性能的影响。通过调整泄漏常数，提升SNN在避障中的控制精度，达到与传统卷积神经网络相当的水平，并发布了一个相关数据集。**

- **链接: [http://arxiv.org/pdf/2507.09538v1](http://arxiv.org/pdf/2507.09538v1)**

> **作者:** Zainab Ali; Lujayn Al-Amir; Ali Safa
>
> **摘要:** Using neuromorphic computing for robotics applications has gained much attention in recent year due to the remarkable ability of Spiking Neural Networks (SNNs) for high-precision yet low memory and compute complexity inference when implemented in neuromorphic hardware. This ability makes SNNs well-suited for autonomous robot applications (such as in drones and rovers) where battery resources and payload are typically limited. Within this context, this paper studies the use of SNNs for performing direct robot navigation and obstacle avoidance from LIDAR data. A custom robot platform equipped with a LIDAR is set up for collecting a labeled dataset of LIDAR sensing data together with the human-operated robot control commands used for obstacle avoidance. Crucially, this paper provides what is, to the best of our knowledge, a first focused study about the importance of neuron membrane leakage on the SNN precision when processing LIDAR data for obstacle avoidance. It is shown that by carefully tuning the membrane potential leakage constant of the spiking Leaky Integrate-and-Fire (LIF) neurons used within our SNN, it is possible to achieve on-par robot control precision compared to the use of a non-spiking Convolutional Neural Network (CNN). Finally, the LIDAR dataset collected during this work is released as open-source with the hope of benefiting future research.
>
---
#### [new 044] Multimodal HD Mapping for Intersections by Intelligent Roadside Units
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于高精地图构建任务，旨在解决传统车载方法在复杂路口因遮挡和视角受限导致的地图构建难题。论文提出了基于路侧智能单元的相机与激光雷达融合框架，并构建了包含多模态数据的RS-seq数据集，以提升语义分割精度，推动车路协同自动驾驶研究。**

- **链接: [http://arxiv.org/pdf/2507.08903v1](http://arxiv.org/pdf/2507.08903v1)**

> **作者:** Zhongzhang Chen; Miao Fan; Shengtong Xu; Mengmeng Yang; Kun Jiang; Xiangzeng Liu; Haoyi Xiong
>
> **备注:** Accepted by ITSC'25
>
> **摘要:** High-definition (HD) semantic mapping of complex intersections poses significant challenges for traditional vehicle-based approaches due to occlusions and limited perspectives. This paper introduces a novel camera-LiDAR fusion framework that leverages elevated intelligent roadside units (IRUs). Additionally, we present RS-seq, a comprehensive dataset developed through the systematic enhancement and annotation of the V2X-Seq dataset. RS-seq includes precisely labelled camera imagery and LiDAR point clouds collected from roadside installations, along with vectorized maps for seven intersections annotated with detailed features such as lane dividers, pedestrian crossings, and stop lines. This dataset facilitates the systematic investigation of cross-modal complementarity for HD map generation using IRU data. The proposed fusion framework employs a two-stage process that integrates modality-specific feature extraction and cross-modal semantic integration, capitalizing on camera high-resolution texture and precise geometric data from LiDAR. Quantitative evaluations using the RS-seq dataset demonstrate that our multimodal approach consistently surpasses unimodal methods. Specifically, compared to unimodal baselines evaluated on the RS-seq dataset, the multimodal approach improves the mean Intersection-over-Union (mIoU) for semantic segmentation by 4\% over the image-only results and 18\% over the point cloud-only results. This study establishes a baseline methodology for IRU-based HD semantic mapping and provides a valuable dataset for future research in infrastructure-assisted autonomous driving systems.
>
---
#### [new 045] DLBAcalib: Robust Extrinsic Calibration for Non-Overlapping LiDARs Based on Dual LBA
- **分类: cs.RO; cs.CL**

- **简介: 该论文属于多激光雷达（LiDAR）系统外参标定任务，旨在解决非重叠视场下多LiDAR系统的高精度、鲁棒标定问题。作者提出DLBAcalib方法，结合LBA优化与迭代精炼，实现无需目标和初始参数的标定，具有高精度和强鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.09176v1](http://arxiv.org/pdf/2507.09176v1)**

> **作者:** Han Ye; Yuqiang Jin; Jinyuan Liu; Tao Li; Wen-An Zhang; Minglei Fu
>
> **备注:** 9 pages,14 figures
>
> **摘要:** Accurate extrinsic calibration of multiple LiDARs is crucial for improving the foundational performance of three-dimensional (3D) map reconstruction systems. This paper presents a novel targetless extrinsic calibration framework for multi-LiDAR systems that does not rely on overlapping fields of view or precise initial parameter estimates. Unlike conventional calibration methods that require manual annotations or specific reference patterns, our approach introduces a unified optimization framework by integrating LiDAR bundle adjustment (LBA) optimization with robust iterative refinement. The proposed method constructs an accurate reference point cloud map via continuous scanning from the target LiDAR and sliding-window LiDAR bundle adjustment, while formulating extrinsic calibration as a joint LBA optimization problem. This method effectively mitigates cumulative mapping errors and achieves outlier-resistant parameter estimation through an adaptive weighting mechanism. Extensive evaluations in both the CARLA simulation environment and real-world scenarios demonstrate that our method outperforms state-of-the-art calibration techniques in both accuracy and robustness. Experimental results show that for non-overlapping sensor configurations, our framework achieves an average translational error of 5 mm and a rotational error of 0.2{\deg}, with an initial error tolerance of up to 0.4 m/30{\deg}. Moreover, the calibration process operates without specialized infrastructure or manual parameter tuning. The code is open source and available on GitHub (\underline{https://github.com/Silentbarber/DLBAcalib})
>
---
#### [new 046] MTF-Grasp: A Multi-tier Federated Learning Approach for Robotic Grasping
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决联邦学习中数据非独立同分布和样本量低导致的性能下降问题。论文提出了MTF-Grasp方法，通过选择数据质量高、样本多的“顶层”机器人训练初始模型，并将其分发给其他机器人，从而提升整体性能。**

- **链接: [http://arxiv.org/pdf/2507.10158v1](http://arxiv.org/pdf/2507.10158v1)**

> **作者:** Obaidullah Zaland; Erik Elmroth; Monowar Bhuyan
>
> **备注:** The work is accepted for presentation at IEEE SMC 2025
>
> **摘要:** Federated Learning (FL) is a promising machine learning paradigm that enables participating devices to train privacy-preserved and collaborative models. FL has proven its benefits for robotic manipulation tasks. However, grasping tasks lack exploration in such settings where robots train a global model without moving data and ensuring data privacy. The main challenge is that each robot learns from data that is nonindependent and identically distributed (non-IID) and of low quantity. This exhibits performance degradation, particularly in robotic grasping. Thus, in this work, we propose MTF-Grasp, a multi-tier FL approach for robotic grasping, acknowledging the unique challenges posed by the non-IID data distribution across robots, including quantitative skewness. MTF-Grasp harnesses data quality and quantity across robots to select a set of "top-level" robots with better data distribution and higher sample count. It then utilizes top-level robots to train initial seed models and distribute them to the remaining "low-level" robots, reducing the risk of model performance degradation in low-level robots. Our approach outperforms the conventional FL setup by up to 8% on the quantity-skewed Cornell and Jacquard grasping datasets.
>
---
#### [new 047] View Invariant Learning for Vision-Language Navigation in Continuous Environments
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言导航任务，旨在解决视角变化影响导航策略的问题。作者提出了VIL方法，通过对比学习和师生框架提升模型对视角变化的鲁棒性。实验表明其在R2R-CE和RxR-CE数据集上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.08831v1](http://arxiv.org/pdf/2507.08831v1)**

> **作者:** Josh Qixuan Sun; Xiaoying Xing; Huaiyuan Weng; Chul Min Yeum; Mark Crowley
>
> **备注:** Under review
>
> **摘要:** Vision-Language Navigation in Continuous Environments (VLNCE), where an agent follows instructions and moves freely to reach a destination, is a key research problem in embodied AI. However, most navigation policies are sensitive to viewpoint changes, i.e., variations in camera height and viewing angle that alter the agent's observation. In this paper, we introduce a generalized scenario, V2-VLNCE (VLNCE with Varied Viewpoints), and propose VIL (View Invariant Learning), a view-invariant post-training strategy that enhances the robustness of existing navigation policies to changes in camera viewpoint. VIL employs a contrastive learning framework to learn sparse and view-invariant features. Additionally, we introduce a teacher-student framework for the Waypoint Predictor Module, a core component of most VLNCE baselines, where a view-dependent teacher model distills knowledge into a view-invariant student model. We employ an end-to-end training paradigm to jointly optimize these components, thus eliminating the cost for individual module training. Empirical results show that our method outperforms state-of-the-art approaches on V2-VLNCE by 8-15% measured on Success Rate for two standard benchmark datasets R2R-CE and RxR-CE. Furthermore, we evaluate VIL under the standard VLNCE setting and find that, despite being trained for varied viewpoints, it often still improves performance. On the more challenging RxR-CE dataset, our method also achieved state-of-the-art performance across all metrics when compared to other map-free methods. This suggests that adding VIL does not diminish the standard viewpoint performance and can serve as a plug-and-play post-training method.
>
---
#### [new 048] RoHOI: Robustness Benchmark for Human-Object Interaction Detection
- **分类: cs.CV; cs.HC; cs.RO; eess.IV**

- **简介: 该论文属于人类-物体交互（HOI）检测任务，旨在解决模型在真实场景中因环境干扰导致的性能下降问题。作者构建了首个鲁棒性基准RoHOI，包含20种数据干扰类型，并提出SAMPL学习策略提升模型鲁棒性，实验表明其方法优于现有技术。**

- **链接: [http://arxiv.org/pdf/2507.09111v1](http://arxiv.org/pdf/2507.09111v1)**

> **作者:** Di Wen; Kunyu Peng; Kailun Yang; Yufan Chen; Ruiping Liu; Junwei Zheng; Alina Roitberg; Rainer Stiefelhagen
>
> **备注:** Benchmarks, datasets, and code will be made publicly available at https://github.com/Kratos-Wen/RoHOI
>
> **摘要:** Human-Object Interaction (HOI) detection is crucial for robot-human assistance, enabling context-aware support. However, models trained on clean datasets degrade in real-world conditions due to unforeseen corruptions, leading to inaccurate prediction. To address this, we introduce the first robustness benchmark for HOI detection, evaluating model resilience under diverse challenges. Despite advances, current models struggle with environmental variability, occlusion, and noise. Our benchmark, RoHOI, includes 20 corruption types based on HICO-DET and V-COCO datasets and a new robustness-focused metric. We systematically analyze existing models in the related field, revealing significant performance drops under corruptions. To improve robustness, we propose a Semantic-Aware Masking-based Progressive Learning (SAMPL) strategy to guide the model to be optimized based on holistic and partial cues, dynamically adjusting the model's optimization to enhance robust feature learning. Extensive experiments show our approach outperforms state-of-the-art methods, setting a new standard for robust HOI detection. Benchmarks, datasets, and code will be made publicly available at https://github.com/Kratos-Wen/RoHOI.
>
---
#### [new 049] Geo-RepNet: Geometry-Aware Representation Learning for Surgical Phase Recognition in Endoscopic Submucosal Dissection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于手术阶段识别任务，旨在解决微创手术中视觉相似性高和结构线索不足导致的识别难题。论文提出了Geo-RepNet模型，融合RGB图像与深度信息，引入几何感知模块，提升复杂手术场景下的识别性能。实验表明其在自建ESD数据集上达到最优效果。**

- **链接: [http://arxiv.org/pdf/2507.09294v1](http://arxiv.org/pdf/2507.09294v1)**

> **作者:** Rui Tang; Haochen Yin; Guankun Wang; Long Bai; An Wang; Huxin Gao; Jiazheng Wang; Hongliang Ren
>
> **备注:** IEEE ICIA 2025
>
> **摘要:** Surgical phase recognition plays a critical role in developing intelligent assistance systems for minimally invasive procedures such as Endoscopic Submucosal Dissection (ESD). However, the high visual similarity across different phases and the lack of structural cues in RGB images pose significant challenges. Depth information offers valuable geometric cues that can complement appearance features by providing insights into spatial relationships and anatomical structures. In this paper, we pioneer the use of depth information for surgical phase recognition and propose Geo-RepNet, a geometry-aware convolutional framework that integrates RGB image and depth information to enhance recognition performance in complex surgical scenes. Built upon a re-parameterizable RepVGG backbone, Geo-RepNet incorporates the Depth-Guided Geometric Prior Generation (DGPG) module that extracts geometry priors from raw depth maps, and the Geometry-Enhanced Multi-scale Attention (GEMA) to inject spatial guidance through geometry-aware cross-attention and efficient multi-scale aggregation. To evaluate the effectiveness of our approach, we construct a nine-phase ESD dataset with dense frame-level annotations from real-world ESD videos. Extensive experiments on the proposed dataset demonstrate that Geo-RepNet achieves state-of-the-art performance while maintaining robustness and high computational efficiency under complex and low-texture surgical environments.
>
---
#### [new 050] Learning and Transferring Better with Depth Information in Visual Reinforcement Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉强化学习任务，旨在提升智能体在不同环境中的泛化能力。通过融合RGB与深度信息，采用基于视觉Transformer的网络结构，并引入对比无监督学习策略以提高样本效率，同时设计课程学习方案促进仿真到现实的迁移。**

- **链接: [http://arxiv.org/pdf/2507.09180v1](http://arxiv.org/pdf/2507.09180v1)**

> **作者:** Zichun Xu; Yuntao Li; Zhaomin Wang; Lei Zhuang; Guocai Yang; Jingdong Zhao
>
> **摘要:** Depth information is robust to scene appearance variations and inherently carries 3D spatial details. In this paper, a visual backbone based on the vision transformer is proposed to fuse RGB and depth modalities for enhancing generalization. Different modalities are first processed by separate CNN stems, and the combined convolutional features are delivered to the scalable vision transformer to obtain visual representations. Moreover, a contrastive unsupervised learning scheme is designed with masked and unmasked tokens to accelerate the sample efficiency during the reinforcement learning progress. For sim2real transfer, a flexible curriculum learning schedule is developed to deploy domain randomization over training processes.
>
---
#### [new 051] GenAI-based Multi-Agent Reinforcement Learning towards Distributed Agent Intelligence: A Generative-RL Agent Perspective
- **分类: cs.AI; cs.ET; cs.HC; cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于多智能体强化学习任务，旨在解决传统方法在联合动作空间、非平稳环境和部分可观测性方面的局限性。论文提出基于生成式AI的强化学习范式，使智能体具备预测环境演化、协调行动和战略推理能力，实现从反应式到前瞻性智能的转变，推动分布式协作智能的发展。**

- **链接: [http://arxiv.org/pdf/2507.09495v1](http://arxiv.org/pdf/2507.09495v1)**

> **作者:** Hang Wang; Junshan Zhang
>
> **备注:** Position paper
>
> **摘要:** Multi-agent reinforcement learning faces fundamental challenges that conventional approaches have failed to overcome: exponentially growing joint action spaces, non-stationary environments where simultaneous learning creates moving targets, and partial observability that constrains coordination. Current methods remain reactive, employing stimulus-response mechanisms that fail when facing novel scenarios. We argue for a transformative paradigm shift from reactive to proactive multi-agent intelligence through generative AI-based reinforcement learning. This position advocates reconceptualizing agents not as isolated policy optimizers, but as sophisticated generative models capable of synthesizing complex multi-agent dynamics and making anticipatory decisions based on predictive understanding of future interactions. Rather than responding to immediate observations, generative-RL agents can model environment evolution, predict other agents' behaviors, generate coordinated action sequences, and engage in strategic reasoning accounting for long-term dynamics. This approach leverages pattern recognition and generation capabilities of generative AI to enable proactive decision-making, seamless coordination through enhanced communication, and dynamic adaptation to evolving scenarios. We envision this paradigm shift will unlock unprecedented possibilities for distributed intelligence, moving beyond individual optimization toward emergent collective behaviors representing genuine collaborative intelligence. The implications extend across autonomous systems, robotics, and human-AI collaboration, promising solutions to coordination challenges intractable under traditional reactive frameworks.
>
---
#### [new 052] Behavioral Exploration: Learning to Explore via In-Context Adaptation
- **分类: cs.LG; cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人与机器学习任务，旨在解决自主智能体快速在线探索与适应的问题。现有方法依赖随机探索和缓慢更新行为策略，而人类能通过少量交互快速适应新环境。为此，作者提出“行为探索”方法，通过大规模专家示范数据训练长上下文生成模型，使其根据历史交互情境选择不同专家行为，从而实现快速在线适应与目标性探索。实验验证了方法在模拟与真实机械操作中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.09041v1](http://arxiv.org/pdf/2507.09041v1)**

> **作者:** Andrew Wagenmaker; Zhiyuan Zhou; Sergey Levine
>
> **摘要:** Developing autonomous agents that quickly explore an environment and adapt their behavior online is a canonical challenge in robotics and machine learning. While humans are able to achieve such fast online exploration and adaptation, often acquiring new information and skills in only a handful of interactions, existing algorithmic approaches tend to rely on random exploration and slow, gradient-based behavior updates. How can we endow autonomous agents with such capabilities on par with humans? Taking inspiration from recent progress on both in-context learning and large-scale behavioral cloning, in this work we propose behavioral exploration: training agents to internalize what it means to explore and adapt in-context over the space of ``expert'' behaviors. To achieve this, given access to a dataset of expert demonstrations, we train a long-context generative model to predict expert actions conditioned on a context of past observations and a measure of how ``exploratory'' the expert's behaviors are relative to this context. This enables the model to not only mimic the behavior of an expert, but also, by feeding its past history of interactions into its context, to select different expert behaviors than what have been previously selected, thereby allowing for fast online adaptation and targeted, ``expert-like'' exploration. We demonstrate the effectiveness of our method in both simulated locomotion and manipulation settings, as well as on real-world robotic manipulation tasks, illustrating its ability to learn adaptive, exploratory behavior.
>
---
#### [new 053] LifelongPR: Lifelong knowledge fusion for point cloud place recognition based on replay and prompt learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于点云场景识别任务，旨在解决模型在持续学习新环境时遗忘旧知识的问题。作者提出了LifelongPR框架，结合回放样本选择和提示学习，提升模型的跨域适应性与知识保留能力，从而增强实际应用中的可扩展性与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.10034v1](http://arxiv.org/pdf/2507.10034v1)**

> **作者:** Xianghong Zou; Jianping Li; Zhe Chen; Zhen Cao; Zhen Dong; Qiegen Liu; Bisheng Yang
>
> **摘要:** Point cloud place recognition (PCPR) plays a crucial role in photogrammetry and robotics applications such as autonomous driving, intelligent transportation, and augmented reality. In real-world large-scale deployments of a positioning system, PCPR models must continuously acquire, update, and accumulate knowledge to adapt to diverse and dynamic environments, i.e., the ability known as continual learning (CL). However, existing PCPR models often suffer from catastrophic forgetting, leading to significant performance degradation in previously learned scenes when adapting to new environments or sensor types. This results in poor model scalability, increased maintenance costs, and system deployment difficulties, undermining the practicality of PCPR. To address these issues, we propose LifelongPR, a novel continual learning framework for PCPR, which effectively extracts and fuses knowledge from sequential point cloud data. First, to alleviate the knowledge loss, we propose a replay sample selection method that dynamically allocates sample sizes according to each dataset's information quantity and selects spatially diverse samples for maximal representativeness. Second, to handle domain shifts, we design a prompt learning-based CL framework with a lightweight prompt module and a two-stage training strategy, enabling domain-specific feature adaptation while minimizing forgetting. Comprehensive experiments on large-scale public and self-collected datasets are conducted to validate the effectiveness of the proposed method. Compared with state-of-the-art (SOTA) methods, our method achieves 6.50% improvement in mIR@1, 7.96% improvement in mR@1, and an 8.95% reduction in F. The code and pre-trained models are publicly available at https://github.com/zouxianghong/LifelongPR.
>
---
#### [new 054] Domain Adaptation and Multi-view Attention for Learnable Landmark Tracking with Sparse Data
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于地表特征跟踪任务，旨在解决稀疏数据下自主航天器地形导航中地标检测与跟踪难题。论文提出轻量级神经网络架构，结合领域自适应与多视角注意力机制，实现高效实时地标检测与描述，提升了现有方法的泛化能力与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.09420v1](http://arxiv.org/pdf/2507.09420v1)**

> **作者:** Timothy Chase Jr; Karthik Dantu
>
> **备注:** Presented at the RSS Space Robotics Workshop 2025. Poster available online at https://tjchase34.github.io/assets/pdfs/rss_poster.pdf
>
> **摘要:** The detection and tracking of celestial surface terrain features are crucial for autonomous spaceflight applications, including Terrain Relative Navigation (TRN), Entry, Descent, and Landing (EDL), hazard analysis, and scientific data collection. Traditional photoclinometry-based pipelines often rely on extensive a priori imaging and offline processing, constrained by the computational limitations of radiation-hardened systems. While historically effective, these approaches typically increase mission costs and duration, operate at low processing rates, and have limited generalization. Recently, learning-based computer vision has gained popularity to enhance spacecraft autonomy and overcome these limitations. While promising, emerging techniques frequently impose computational demands exceeding the capabilities of typical spacecraft hardware for real-time operation and are further challenged by the scarcity of labeled training data for diverse extraterrestrial environments. In this work, we present novel formulations for in-situ landmark tracking via detection and description. We utilize lightweight, computationally efficient neural network architectures designed for real-time execution on current-generation spacecraft flight processors. For landmark detection, we propose improved domain adaptation methods that enable the identification of celestial terrain features with distinct, cheaply acquired training data. Concurrently, for landmark description, we introduce a novel attention alignment formulation that learns robust feature representations that maintain correspondence despite significant landmark viewpoint variations. Together, these contributions form a unified system for landmark tracking that demonstrates superior performance compared to existing state-of-the-art techniques.
>
---
#### [new 055] Towards Emotion Co-regulation with LLM-powered Socially Assistive Robots: Integrating LLM Prompts and Robotic Behaviors to Support Parent-Neurodivergent Child Dyads
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互与情感计算任务，旨在解决神经多样性儿童与家长间情感共调节的问题。研究通过结合大语言模型与社交辅助机器人（MiRo-E），设计可调控对话与行为的干预系统，探索其对亲子互动及情绪调节的效果，并提出优化设计方向。**

- **链接: [http://arxiv.org/pdf/2507.10427v1](http://arxiv.org/pdf/2507.10427v1)**

> **作者:** Jing Li; Felix Schijve; Sheng Li; Yuye Yang; Jun Hu; Emilia Barakova
>
> **备注:** Submission for the IROS 2025 conference
>
> **摘要:** Socially Assistive Robotics (SAR) has shown promise in supporting emotion regulation for neurodivergent children. Recently, there has been increasing interest in leveraging advanced technologies to assist parents in co-regulating emotions with their children. However, limited research has explored the integration of large language models (LLMs) with SAR to facilitate emotion co-regulation between parents and children with neurodevelopmental disorders. To address this gap, we developed an LLM-powered social robot by deploying a speech communication module on the MiRo-E robotic platform. This supervised autonomous system integrates LLM prompts and robotic behaviors to deliver tailored interventions for both parents and neurodivergent children. Pilot tests were conducted with two parent-child dyads, followed by a qualitative analysis. The findings reveal MiRo-E's positive impacts on interaction dynamics and its potential to facilitate emotion regulation, along with identified design and technical challenges. Based on these insights, we provide design implications to advance the future development of LLM-powered SAR for mental health applications.
>
---
#### [new 056] Assuring the Safety of Reinforcement Learning Components: AMLAS-RL
- **分类: cs.LG; cs.AI; cs.RO; cs.SE**

- **简介: 该论文属于安全验证任务，旨在解决强化学习（RL）在安全关键型系统中的安全保障问题。作者基于AMLAS方法，提出适用于RL系统的AMLAS-RL框架，通过迭代过程生成安全论证，并以轮式车辆避障为例进行演示。**

- **链接: [http://arxiv.org/pdf/2507.08848v1](http://arxiv.org/pdf/2507.08848v1)**

> **作者:** Calum Corrie Imrie; Ioannis Stefanakos; Sepeedeh Shahbeigi; Richard Hawkins; Simon Burton
>
> **摘要:** The rapid advancement of machine learning (ML) has led to its increasing integration into cyber-physical systems (CPS) across diverse domains. While CPS offer powerful capabilities, incorporating ML components introduces significant safety and assurance challenges. Among ML techniques, reinforcement learning (RL) is particularly suited for CPS due to its capacity to handle complex, dynamic environments where explicit models of interaction between system and environment are unavailable or difficult to construct. However, in safety-critical applications, this learning process must not only be effective but demonstrably safe. Safe-RL methods aim to address this by incorporating safety constraints during learning, yet they fall short in providing systematic assurance across the RL lifecycle. The AMLAS methodology offers structured guidance for assuring the safety of supervised learning components, but it does not directly apply to the unique challenges posed by RL. In this paper, we adapt AMLAS to provide a framework for generating assurance arguments for an RL-enabled system through an iterative process; AMLAS-RL. We demonstrate AMLAS-RL using a running example of a wheeled vehicle tasked with reaching a target goal without collision.
>
---
#### [new 057] Consistency Trajectory Planning: High-Quality and Efficient Trajectory Optimization for Offline Model-Based Reinforcement Learning
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 论文提出了一种名为Consistency Trajectory Planning（CTP）的离线模型强化学习方法，用于高效轨迹优化。它基于新引入的Consistency Trajectory Model（CTM），旨在解决扩散模型在规划中计算成本高的问题，实现快速单步生成高质量轨迹，适用于长视野、目标条件任务。**

- **链接: [http://arxiv.org/pdf/2507.09534v1](http://arxiv.org/pdf/2507.09534v1)**

> **作者:** Guanquan Wang; Takuya Hiraoka; Yoshimasa Tsuruoka
>
> **摘要:** This paper introduces Consistency Trajectory Planning (CTP), a novel offline model-based reinforcement learning method that leverages the recently proposed Consistency Trajectory Model (CTM) for efficient trajectory optimization. While prior work applying diffusion models to planning has demonstrated strong performance, it often suffers from high computational costs due to iterative sampling procedures. CTP supports fast, single-step trajectory generation without significant degradation in policy quality. We evaluate CTP on the D4RL benchmark and show that it consistently outperforms existing diffusion-based planning methods in long-horizon, goal-conditioned tasks. Notably, CTP achieves higher normalized returns while using significantly fewer denoising steps. In particular, CTP achieves comparable performance with over $120\times$ speedup in inference time, demonstrating its practicality and effectiveness for high-performance, low-latency offline planning.
>
---
#### [new 058] Privacy-Preserving Multi-Stage Fall Detection Framework with Semi-supervised Federated Learning and Robotic Vision Confirmation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于智能医疗任务，旨在解决老年人跌倒检测中的隐私与准确率问题。作者提出了一种结合半监督联邦学习（SF2D）、室内定位导航和机器人视觉确认的多阶段跌倒检测框架。通过可穿戴设备初步识别跌倒，再由机器人导航至事发地点并用视觉系统确认，最终实现高精度（99.99%）且保护隐私的跌倒检测。**

- **链接: [http://arxiv.org/pdf/2507.10474v1](http://arxiv.org/pdf/2507.10474v1)**

> **作者:** Seyed Alireza Rahimi Azghadi; Truong-Thanh-Hung Nguyen; Helene Fournier; Monica Wachowicz; Rene Richard; Francis Palma; Hung Cao
>
> **摘要:** The aging population is growing rapidly, and so is the danger of falls in older adults. A major cause of injury is falling, and detection in time can greatly save medical expenses and recovery time. However, to provide timely intervention and avoid unnecessary alarms, detection systems must be effective and reliable while addressing privacy concerns regarding the user. In this work, we propose a framework for detecting falls using several complementary systems: a semi-supervised federated learning-based fall detection system (SF2D), an indoor localization and navigation system, and a vision-based human fall recognition system. A wearable device and an edge device identify a fall scenario in the first system. On top of that, the second system uses an indoor localization technique first to localize the fall location and then navigate a robot to inspect the scenario. A vision-based detection system running on an edge device with a mounted camera on a robot is used to recognize fallen people. Each of the systems of this proposed framework achieves different accuracy rates. Specifically, the SF2D has a 0.81% failure rate equivalent to 99.19% accuracy, while the vision-based fallen people detection achieves 96.3% accuracy. However, when we combine the accuracy of these two systems with the accuracy of the navigation system (95% success rate), our proposed framework creates a highly reliable performance for fall detection, with an overall accuracy of 99.99%. Not only is the proposed framework safe for older adults, but it is also a privacy-preserving solution for detecting falls.
>
---
#### [new 059] Learning to Control Dynamical Agents via Spiking Neural Networks and Metropolis-Hastings Sampling
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决脉冲神经网络（SNN）在无梯度情况下训练困难的问题。作者提出一种基于Metropolis-Hastings采样的新方法，用于训练SNN完成动态智能体控制，不依赖反向传播。实验表明其性能优于传统深度Q学习和已有SNN强化学习方法。**

- **链接: [http://arxiv.org/pdf/2507.09540v1](http://arxiv.org/pdf/2507.09540v1)**

> **作者:** Ali Safa; Farida Mohsen; Ali Al-Zawqari
>
> **摘要:** Spiking Neural Networks (SNNs) offer biologically inspired, energy-efficient alternatives to traditional Deep Neural Networks (DNNs) for real-time control systems. However, their training presents several challenges, particularly for reinforcement learning (RL) tasks, due to the non-differentiable nature of spike-based communication. In this work, we introduce what is, to our knowledge, the first framework that employs Metropolis-Hastings (MH) sampling, a Bayesian inference technique, to train SNNs for dynamical agent control in RL environments without relying on gradient-based methods. Our approach iteratively proposes and probabilistically accepts network parameter updates based on accumulated reward signals, effectively circumventing the limitations of backpropagation while enabling direct optimization on neuromorphic platforms. We evaluated this framework on two standard control benchmarks: AcroBot and CartPole. The results demonstrate that our MH-based approach outperforms conventional Deep Q-Learning (DQL) baselines and prior SNN-based RL approaches in terms of maximizing the accumulated reward while minimizing network resources and training episodes.
>
---
#### [new 060] SegVec3D: A Method for Vector Embedding of 3D Objects Oriented Towards Robot manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D点云实例分割与多模态理解任务，旨在提升机器人操作中对3D物体的识别与语义理解能力。论文提出SegVec3D框架，融合注意力机制、嵌入学习和跨模态对齐，实现无监督实例分割与零样本检索，解决了现有方法在监督依赖和多模态融合上的不足。**

- **链接: [http://arxiv.org/pdf/2507.09459v1](http://arxiv.org/pdf/2507.09459v1)**

> **作者:** Zhihan Kang; Boyu Wang
>
> **备注:** Undergraduate Theis; 12 pages, 6 figures
>
> **摘要:** We propose SegVec3D, a novel framework for 3D point cloud instance segmentation that integrates attention mechanisms, embedding learning, and cross-modal alignment. The approach builds a hierarchical feature extractor to enhance geometric structure modeling and enables unsupervised instance segmentation via contrastive clustering. It further aligns 3D data with natural language queries in a shared semantic space, supporting zero-shot retrieval. Compared to recent methods like Mask3D and ULIP, our method uniquely unifies instance segmentation and multimodal understanding with minimal supervision and practical deployability.
>
---
#### [new 061] Bridging Bots: from Perception to Action via Multimodal-LMs and Knowledge Graphs
- **分类: cs.AI; cs.RO**

- **简介: 该论文旨在解决服务机器人在复杂环境中从感知到行动的决策问题，属于人工智能与机器人交互任务。为实现跨平台互操作性，作者提出结合多模态语言模型和知识图谱的神经符号框架，通过生成符合本体的知识图谱指导机器人行为。实验评估了多个模型的效果，并强调集成策略的重要性。**

- **链接: [http://arxiv.org/pdf/2507.09617v1](http://arxiv.org/pdf/2507.09617v1)**

> **作者:** Margherita Martorana; Francesca Urgese; Mark Adamik; Ilaria Tiddi
>
> **摘要:** Personal service robots are deployed to support daily living in domestic environments, particularly for elderly and individuals requiring assistance. These robots must perceive complex and dynamic surroundings, understand tasks, and execute context-appropriate actions. However, current systems rely on proprietary, hard-coded solutions tied to specific hardware and software, resulting in siloed implementations that are difficult to adapt and scale across platforms. Ontologies and Knowledge Graphs (KGs) offer a solution to enable interoperability across systems, through structured and standardized representations of knowledge and reasoning. However, symbolic systems such as KGs and ontologies struggle with raw and noisy sensory input. In contrast, multimodal language models are well suited for interpreting input such as images and natural language, but often lack transparency, consistency, and knowledge grounding. In this work, we propose a neurosymbolic framework that combines the perceptual strengths of multimodal language models with the structured representations provided by KGs and ontologies, with the aim of supporting interoperability in robotic applications. Our approach generates ontology-compliant KGs that can inform robot behavior in a platform-independent manner. We evaluated this framework by integrating robot perception data, ontologies, and five multimodal models (three LLaMA and two GPT models), using different modes of neural-symbolic interaction. We assess the consistency and effectiveness of the generated KGs across multiple runs and configurations, and perform statistical analyzes to evaluate performance. Results show that GPT-o1 and LLaMA 4 Maverick consistently outperform other models. However, our findings also indicate that newer models do not guarantee better results, highlighting the critical role of the integration strategy in generating ontology-compliant KGs.
>
---
## 更新

#### [replaced 001] XMoP: Whole-Body Control Policy for Zero-shot Cross-Embodiment Neural Motion Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.15585v2](http://arxiv.org/pdf/2409.15585v2)**

> **作者:** Prabin Kumar Rath; Nakul Gopalan
>
> **备注:** Website at https://prabinrath.github.io/xmop Paper has 17 pages, 13 figures, 5 tables
>
> **摘要:** Classical manipulator motion planners work across different robot embodiments. However they plan on a pre-specified static environment representation, and are not scalable to unseen dynamic environments. Neural Motion Planners (NMPs) are an appealing alternative to conventional planners as they incorporate different environmental constraints to learn motion policies directly from raw sensor observations. Contemporary state-of-the-art NMPs can successfully plan across different environments. However none of the existing NMPs generalize across robot embodiments. In this paper we propose Cross-Embodiment Motion Policy (XMoP), a neural policy for learning to plan over a distribution of manipulators. XMoP implicitly learns to satisfy kinematic constraints for a distribution of robots and $\textit{zero-shot}$ transfers the planning behavior to unseen robotic manipulators within this distribution. We achieve this generalization by formulating a whole-body control policy that is trained on planning demonstrations from over three million procedurally sampled robotic manipulators in different simulated environments. Despite being completely trained on synthetic embodiments and environments, our policy exhibits strong sim-to-real generalization across manipulators with different kinematic variations and degrees of freedom with a single set of frozen policy parameters. We evaluate XMoP on $7$ commercial manipulators and show successful cross-embodiment motion planning, achieving an average $70\%$ success rate on baseline benchmarks. Furthermore, we demonstrate our policy sim-to-real on two unseen manipulators solving novel planning problems across three real-world domains even with dynamic obstacles.
>
---
#### [replaced 002] Spatial-Temporal Aware Visuomotor Diffusion Policy Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.06710v2](http://arxiv.org/pdf/2507.06710v2)**

> **作者:** Zhenyang Liu; Yikai Wang; Kuanning Wang; Longfei Liang; Xiangyang Xue; Yanwei Fu
>
> **摘要:** Visual imitation learning is effective for robots to learn versatile tasks. However, many existing methods rely on behavior cloning with supervised historical trajectories, limiting their 3D spatial and 4D spatiotemporal awareness. Consequently, these methods struggle to capture the 3D structures and 4D spatiotemporal relationships necessary for real-world deployment. In this work, we propose 4D Diffusion Policy (DP4), a novel visual imitation learning method that incorporates spatiotemporal awareness into diffusion-based policies. Unlike traditional approaches that rely on trajectory cloning, DP4 leverages a dynamic Gaussian world model to guide the learning of 3D spatial and 4D spatiotemporal perceptions from interactive environments. Our method constructs the current 3D scene from a single-view RGB-D observation and predicts the future 3D scene, optimizing trajectory generation by explicitly modeling both spatial and temporal dependencies. Extensive experiments across 17 simulation tasks with 173 variants and 3 real-world robotic tasks demonstrate that the 4D Diffusion Policy (DP4) outperforms baseline methods, improving the average simulation task success rate by 16.4% (Adroit), 14% (DexArt), and 6.45% (RLBench), and the average real-world robotic task success rate by 8.6%.
>
---
#### [replaced 003] Self-Supervised Monocular 4D Scene Reconstruction for Egocentric Videos
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.09145v4](http://arxiv.org/pdf/2411.09145v4)**

> **作者:** Chengbo Yuan; Geng Chen; Li Yi; Yang Gao
>
> **摘要:** Egocentric videos provide valuable insights into human interactions with the physical world, which has sparked growing interest in the computer vision and robotics communities. A critical challenge in fully understanding the geometry and dynamics of egocentric videos is dense scene reconstruction. However, the lack of high-quality labeled datasets in this field has hindered the effectiveness of current supervised learning methods. In this work, we aim to address this issue by exploring an self-supervised dynamic scene reconstruction approach. We introduce EgoMono4D, a novel model that unifies the estimation of multiple variables necessary for Egocentric Monocular 4D reconstruction, including camera intrinsic, camera poses, and video depth, all within a fast feed-forward framework. Starting from pretrained single-frame depth and intrinsic estimation model, we extend it with camera poses estimation and align multi-frame results on large-scale unlabeled egocentric videos. We evaluate EgoMono4D in both in-domain and zero-shot generalization settings, achieving superior performance in dense pointclouds sequence reconstruction compared to all baselines. EgoMono4D represents the first attempt to apply self-supervised learning for pointclouds sequence reconstruction to the label-scarce egocentric field, enabling fast, dense, and generalizable reconstruction. The interactable visualization, code and trained models are released https://egomono4d.github.io/
>
---
#### [replaced 004] Uncertainty-Aware Safety-Critical Decision and Control for Autonomous Vehicles at Unsignalized Intersections
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.19939v2](http://arxiv.org/pdf/2505.19939v2)**

> **作者:** Ran Yu; Zhuoren Li; Lu Xiong; Wei Han; Bo Leng
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Reinforcement learning (RL) has demonstrated potential in autonomous driving (AD) decision tasks. However, applying RL to urban AD, particularly in intersection scenarios, still faces significant challenges. The lack of safety constraints makes RL vulnerable to risks. Additionally, cognitive limitations and environmental randomness can lead to unreliable decisions in safety-critical scenarios. Therefore, it is essential to quantify confidence in RL decisions to improve safety. This paper proposes an Uncertainty-aware Safety-Critical Decision and Control (USDC) framework, which generates a risk-averse policy by constructing a risk-aware ensemble distributional RL, while estimating uncertainty to quantify the policy's reliability. Subsequently, a high-order control barrier function (HOCBF) is employed as a safety filter to minimize intervention policy while dynamically enhancing constraints based on uncertainty. The ensemble critics evaluate both HOCBF and RL policies, embedding uncertainty to achieve dynamic switching between safe and flexible strategies, thereby balancing safety and efficiency. Simulation tests on unsignalized intersections in multiple tasks indicate that USDC can improve safety while maintaining traffic efficiency compared to baselines.
>
---
#### [replaced 005] SF-TIM: A Simple Framework for Enhancing Quadrupedal Robot Jumping Agility by Combining Terrain Imagination and Measurement
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2408.00486v2](http://arxiv.org/pdf/2408.00486v2)**

> **作者:** Ze Wang; Yang Li; Long Xu; Hao Shi; Zunwang Ma; Zhen Chu; Chao Li; Fei Gao; Kailun Yang; Kaiwei Wang
>
> **备注:** Accepted to IROS 2025. A demo video has been made available at https://flysoaryun.github.io/SF-TIM
>
> **摘要:** Dynamic jumping on high platforms and over gaps differentiates legged robots from wheeled counterparts. Dynamic locomotion on abrupt surfaces, as opposed to walking on rough terrains, demands the integration of proprioceptive and exteroceptive perception to enable explosive movements. In this paper, we propose SF-TIM (Simple Framework combining Terrain Imagination and Measurement), a single-policy method that enhances quadrupedal robot jumping agility, while preserving their fundamental blind walking capabilities. In addition, we introduce a terrain-guided reward design specifically to assist quadrupedal robots in high jumping, improving their performance in this task. To narrow the simulation-to-reality gap in quadrupedal robot learning, we introduce a stable and high-speed elevation map generation framework, enabling zero-shot simulation-to-reality transfer of locomotion ability. Our algorithm has been deployed and validated on both the small-/large-size quadrupedal robots, demonstrating its effectiveness in real-world applications: the robot has successfully traversed various high platforms and gaps, showing the robustness of our proposed approach. A demo video has been made available at https://flysoaryun.github.io/SF-TIM.
>
---
#### [replaced 006] Class-Aware PillarMix: Can Mixed Sample Data Augmentation Enhance 3D Object Detection with Radar Point Clouds?
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.02687v2](http://arxiv.org/pdf/2503.02687v2)**

> **作者:** Miao Zhang; Sherif Abdulatif; Benedikt Loesch; Marco Altmann; Bin Yang
>
> **备注:** 8 pages, 6 figures, 4 tables, accepted to 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Due to the significant effort required for data collection and annotation in 3D perception tasks, mixed sample data augmentation (MSDA) has been widely studied to generate diverse training samples by mixing existing data. Recently, many MSDA techniques have been developed for point clouds, but they mainly target LiDAR data, leaving their application to radar point clouds largely unexplored. In this paper, we examine the feasibility of applying existing MSDA methods to radar point clouds and identify several challenges in adapting these techniques. These obstacles stem from the radar's irregular angular distribution, deviations from a single-sensor polar layout in multi-radar setups, and point sparsity. To address these issues, we propose Class-Aware PillarMix (CAPMix), a novel MSDA approach that applies MixUp at the pillar level in 3D point clouds, guided by class labels. Unlike methods that rely a single mix ratio to the entire sample, CAPMix assigns an independent ratio to each pillar, boosting sample diversity. To account for the density of different classes, we use class-specific distributions: for dense objects (e.g., large vehicles), we skew ratios to favor points from another sample, while for sparse objects (e.g., pedestrians), we sample more points from the original. This class-aware mixing retains critical details and enriches each sample with new information, ultimately generating more diverse training data. Experimental results demonstrate that our method not only significantly boosts performance but also outperforms existing MSDA approaches across two datasets (Bosch Street and K-Radar). We believe that this straightforward yet effective approach will spark further investigation into MSDA techniques for radar data.
>
---
#### [replaced 007] Learning Free Terminal Time Optimal Closed-loop Control of Manipulators
- **分类: math.OC; cs.RO**

- **链接: [http://arxiv.org/pdf/2311.17749v2](http://arxiv.org/pdf/2311.17749v2)**

> **作者:** Wei Hu; Yue Zhao; Weinan E; Jiequn Han; Jihao Long
>
> **备注:** Accepted for presentation at the American Control Conference (ACC) 2025
>
> **摘要:** This paper presents a novel approach to learning free terminal time closed-loop control for robotic manipulation tasks, enabling dynamic adjustment of task duration and control inputs to enhance performance. We extend the supervised learning approach, namely solving selected optimal open-loop problems and utilizing them as training data for a policy network, to the free terminal time scenario. Three main challenges are addressed in this extension. First, we introduce a marching scheme that enhances the solution quality and increases the success rate of the open-loop solver by gradually refining time discretization. Second, we extend the QRnet in Nakamura-Zimmerer et al. (2021b) to the free terminal time setting to address discontinuity and improve stability at the terminal state. Third, we present a more automated version of the initial value problem (IVP) enhanced sampling method from previous work (Zhang et al., 2022) to adaptively update the training dataset, significantly improving its quality. By integrating these techniques, we develop a closed-loop policy that operates effectively over a broad domain with varying optimal time durations, achieving near globally optimal total costs.
>
---
#### [replaced 008] Informed, Constrained, Aligned: A Field Analysis on Degeneracy-aware Point Cloud Registration in the Wild
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2408.11809v3](http://arxiv.org/pdf/2408.11809v3)**

> **作者:** Turcan Tuna; Julian Nubert; Patrick Pfreundschuh; Cesar Cadena; Shehryar Khattak; Marco Hutter
>
> **备注:** Accepted to IEEE Transactions on Field Robotics
>
> **摘要:** The ICP registration algorithm has been a preferred method for LiDAR-based robot localization for nearly a decade. However, even in modern SLAM solutions, ICP can degrade and become unreliable in geometrically ill-conditioned environments. Current solutions primarily focus on utilizing additional sources of information, such as external odometry, to either replace the degenerate directions of the optimization solution or add additional constraints in a sensor-fusion setup afterward. In response, this work investigates and compares new and existing degeneracy mitigation methods for robust LiDAR-based localization and analyzes the efficacy of these approaches in degenerate environments for the first time in the literature at this scale. Specifically, this work investigates i) the effect of using active or passive degeneracy mitigation methods for the problem of ill-conditioned ICP in LiDAR degenerate environments, ii) the evaluation of TSVD, inequality constraints, and linear/non-linear Tikhonov regularization for the application of degenerate point cloud registration for the first time. Furthermore, a sensitivity analysis for least-squares minimization step of the ICP problem is carried out to better understand how each method affects the optimization and what to expect from each method. The results of the analysis are validated through multiple real-world robotic field and simulated experiments. The analysis demonstrates that active optimization degeneracy mitigation is necessary and advantageous in the absence of reliable external estimate assistance for LiDAR-SLAM, and soft-constrained methods can provide better results in complex ill-conditioned scenarios with heuristic fine-tuned parameters.
>
---
#### [replaced 009] GO-VMP: Global Optimization for View Motion Planning in Fruit Mapping
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.03912v2](http://arxiv.org/pdf/2503.03912v2)**

> **作者:** Allen Isaac Jose; Sicong Pan; Tobias Zaenker; Rohit Menon; Sebastian Houben; Maren Bennewitz
>
> **备注:** Allen Isaac Jose and Sicong Pan have equal contribution. Publication to appear in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025
>
> **摘要:** Automating labor-intensive tasks such as crop monitoring with robots is essential for enhancing production and conserving resources. However, autonomously monitoring horticulture crops remains challenging due to their complex structures, which often result in fruit occlusions. Existing view planning methods attempt to reduce occlusions but either struggle to achieve adequate coverage or incur high robot motion costs. We introduce a global optimization approach for view motion planning that aims to minimize robot motion costs while maximizing fruit coverage. To this end, we leverage coverage constraints derived from the set covering problem (SCP) within a shortest Hamiltonian path problem (SHPP) formulation. While both SCP and SHPP are well-established, their tailored integration enables a unified framework that computes a global view path with minimized motion while ensuring full coverage of selected targets. Given the NP-hard nature of the problem, we employ a region-prior-based selection of coverage targets and a sparse graph structure to achieve effective optimization outcomes within a limited time. Experiments in simulation demonstrate that our method detects more fruits, enhances surface coverage, and achieves higher volume accuracy than the motion-efficient baseline with a moderate increase in motion cost, while significantly reducing motion costs compared to the coverage-focused baseline. Real-world experiments further confirm the practical applicability of our approach.
>
---
#### [replaced 010] Ark: An Open-source Python-based Framework for Robot Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.21628v2](http://arxiv.org/pdf/2506.21628v2)**

> **作者:** Magnus Dierking; Christopher E. Mower; Sarthak Das; Huang Helong; Jiacheng Qiu; Cody Reading; Wei Chen; Huidong Liang; Huang Guowei; Jan Peters; Quan Xingyue; Jun Wang; Haitham Bou-Ammar
>
> **摘要:** Robotics has made remarkable hardware strides-from DARPA's Urban and Robotics Challenges to the first humanoid-robot kickboxing tournament-yet commercial autonomy still lags behind progress in machine learning. A major bottleneck is software: current robot stacks demand steep learning curves, low-level C/C++ expertise, fragmented tooling, and intricate hardware integration, in stark contrast to the Python-centric, well-documented ecosystems that propelled modern AI. We introduce ARK, an open-source, Python-first robotics framework designed to close that gap. ARK presents a Gym-style environment interface that allows users to collect data, preprocess it, and train policies using state-of-the-art imitation-learning algorithms (e.g., ACT, Diffusion Policy) while seamlessly toggling between high-fidelity simulation and physical robots. A lightweight client-server architecture provides networked publisher-subscriber communication, and optional C/C++ bindings ensure real-time performance when needed. ARK ships with reusable modules for control, SLAM, motion planning, system identification, and visualization, along with native ROS interoperability. Comprehensive documentation and case studies-from manipulation to mobile navigation-demonstrate rapid prototyping, effortless hardware swapping, and end-to-end pipelines that rival the convenience of mainstream machine-learning workflows. By unifying robotics and AI practices under a common Python umbrella, ARK lowers entry barriers and accelerates research and commercial deployment of autonomous robots.
>
---
#### [replaced 011] CSC-MPPI: A Novel Constrained MPPI Framework with DBSCAN for Reliable Obstacle Avoidance
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.16386v2](http://arxiv.org/pdf/2506.16386v2)**

> **作者:** Leesai Park; Keunwoo Jang; Sanghyun Kim
>
> **摘要:** This paper proposes Constrained Sampling Cluster Model Predictive Path Integral (CSC-MPPI), a novel constrained formulation of MPPI designed to enhance trajectory optimization while enforcing strict constraints on system states and control inputs. Traditional MPPI, which relies on a probabilistic sampling process, often struggles with constraint satisfaction and generates suboptimal trajectories due to the weighted averaging of sampled trajectories. To address these limitations, the proposed framework integrates a primal-dual gradient-based approach and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to steer sampled input trajectories into feasible regions while mitigating risks associated with weighted averaging. First, to ensure that sampled trajectories remain within the feasible region, the primal-dual gradient method is applied to iteratively shift sampled inputs while enforcing state and control constraints. Then, DBSCAN groups the sampled trajectories, enabling the selection of representative control inputs within each cluster. Finally, among the representative control inputs, the one with the lowest cost is chosen as the optimal action. As a result, CSC-MPPI guarantees constraint satisfaction, improves trajectory selection, and enhances robustness in complex environments. Simulation and real-world experiments demonstrate that CSC-MPPI outperforms traditional MPPI in obstacle avoidance, achieving improved reliability and efficiency. The experimental videos are available at https://cscmppi.github.io
>
---
#### [replaced 012] CoDe: A Cooperative and Decentralized Collision Avoidance Algorithm for Small-Scale UAV Swarms Considering Energy Efficiency
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2204.08594v2](http://arxiv.org/pdf/2204.08594v2)**

> **作者:** Shuangyao Huang; Haibo Zhang; Zhiyi Huang
>
> **备注:** Accepted at IROS 2024
>
> **摘要:** This paper introduces a cooperative and decentralized collision avoidance algorithm (CoDe) for small-scale UAV swarms consisting of up to three UAVs. CoDe improves energy efficiency of UAVs by achieving effective cooperation among UAVs. Moreover, CoDe is specifically tailored for UAV's operations by addressing the challenges faced by existing schemes, such as ineffectiveness in selecting actions from continuous action spaces and high computational complexity. CoDe is based on Multi-Agent Reinforcement Learning (MARL), and finds cooperative policies by incorporating a novel credit assignment scheme. The novel credit assignment scheme estimates the contribution of an individual by subtracting a baseline from the joint action value for the swarm. The credit assignment scheme in CoDe outperforms other benchmarks as the baseline takes into account not only the importance of a UAV's action but also the interrelation between UAVs. Furthermore, extensive experiments are conducted against existing MARL-based and conventional heuristic-based algorithms to demonstrate the advantages of the proposed algorithm.
>
---
#### [replaced 013] An Adaptive Sliding Window Estimator for Positioning of Unmanned Aerial Vehicle Using a Single Anchor
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.06501v4](http://arxiv.org/pdf/2409.06501v4)**

> **作者:** Kaiwen Xiong; Sijia Chen; Wei Dong
>
> **备注:** This work has been accepted by the IEEE Sensors Journal
>
> **摘要:** Localization using a single range anchor combined with onboard optical-inertial odometry offers a lightweight solution that provides multidimensional measurements for the positioning of unmanned aerial vehicles. Unfortunately, the performance of such lightweight sensors varies with the dynamic environment, and the fidelity of the dynamic model is also severely affected by environmental aerial flow. To address this challenge, we propose an adaptive sliding window estimator equipped with an estimation reliability evaluator, where the states, noise covariance matrices and aerial drag are estimated simultaneously. The aerial drag effects are first evaluated based on posterior states and covariance. Then, an augmented Kalman filter is designed to pre-process multidimensional measurements and inherit historical information. Subsequently, an inverse-Wishart smoother is employed to estimate posterior states and covariance matrices. To further suppress potential divergence, a reliability evaluator is devised to infer estimation errors. We further determine the fidelity of each sensor based on the error propagation. Extensive experiments are conducted in both standard and harsh environments, demonstrating the adaptability and robustness of the proposed method. The root mean square error reaches 0.15 m, outperforming the state-of-the-art approach. Real-world close-loop control experiments are additionally performed to verify the estimator's competence in practical application.
>
---
#### [replaced 014] Is Intermediate Fusion All You Need for UAV-based Collaborative Perception?
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.21774v2](http://arxiv.org/pdf/2504.21774v2)**

> **作者:** Jiuwu Hao; Liguo Sun; Yuting Wan; Yueyang Wu; Ti Xiang; Haolin Song; Pin Lv
>
> **备注:** Accepted by ITSC 2025
>
> **摘要:** Collaborative perception enhances environmental awareness through inter-agent communication and is regarded as a promising solution to intelligent transportation systems. However, existing collaborative methods for Unmanned Aerial Vehicles (UAVs) overlook the unique characteristics of the UAV perspective, resulting in substantial communication overhead. To address this issue, we propose a novel communication-efficient collaborative perception framework based on late-intermediate fusion, dubbed LIF. The core concept is to exchange informative and compact detection results and shift the fusion stage to the feature representation level. In particular, we leverage vision-guided positional embedding (VPE) and box-based virtual augmented feature (BoBEV) to effectively integrate complementary information from various agents. Additionally, we innovatively introduce an uncertainty-driven communication mechanism that uses uncertainty evaluation to select high-quality and reliable shared areas. Experimental results demonstrate that our LIF achieves superior performance with minimal communication bandwidth, proving its effectiveness and practicality. Code and models are available at https://github.com/uestchjw/LIF.
>
---
#### [replaced 015] SEAL: Towards Safe Autonomous Driving via Skill-Enabled Adversary Learning for Closed-Loop Scenario Generation
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.10320v3](http://arxiv.org/pdf/2409.10320v3)**

> **作者:** Benjamin Stoler; Ingrid Navarro; Jonathan Francis; Jean Oh
>
> **备注:** Accepted to the IEEE Robotics and Automation Letters (RA-L) on June 28, 2025
>
> **摘要:** Verification and validation of autonomous driving (AD) systems and components is of increasing importance, as such technology increases in real-world prevalence. Safety-critical scenario generation is a key approach to robustify AD policies through closed-loop training. However, existing approaches for scenario generation rely on simplistic objectives, resulting in overly-aggressive or non-reactive adversarial behaviors. To generate diverse adversarial yet realistic scenarios, we propose SEAL, a scenario perturbation approach which leverages learned objective functions and adversarial, human-like skills. SEAL-perturbed scenarios are more realistic than SOTA baselines, leading to improved ego task success across real-world, in-distribution, and out-of-distribution scenarios, of more than 20%. To facilitate future research, we release our code and tools: https://github.com/cmubig/SEAL
>
---
#### [replaced 016] A Multi-Simulation Approach with Model Predictive Control for Anafi Drones
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.10218v2](http://arxiv.org/pdf/2502.10218v2)**

> **作者:** Pascal Goldschmid; Aamir Ahmad
>
> **备注:** 8 pages, 7 figures, accepted for publication at ECMR 2025
>
> **摘要:** Simulation frameworks are essential for the safe development of robotic applications. However, different components of a robotic system are often best simulated in different environments, making full integration challenging. This is particularly true for partially-open or closed-source simulators, which commonly suffer from two limitations: (i) lack of runtime control over scene actors via interfaces like ROS, and (ii) restricted access to real-time state data (e.g., pose, velocity) of scene objects. In the first part of this work, we address these issues by integrating aerial drones simulated in Parrot's Sphinx environment (used for Anafi drones) into the Gazebo simulator. Our approach uses a mirrored drone instance embedded within Gazebo environments to bridge the two simulators. One key application is aerial target tracking, a common task in multi-robot systems. However, Parrot's default PID-based controller lacks the agility needed for tracking fast-moving targets. To overcome this, in the second part of this work we develop a model predictive controller (MPC) that leverages cumulative error states to improve tracking accuracy. Our MPC significantly outperforms the built-in PID controller in dynamic scenarios, increasing the effectiveness of the overall system. We validate our integrated framework by incorporating the Anafi drone into an existing Gazebo-based airship simulation and rigorously test the MPC against a custom PID baseline in both simulated and real-world experiments.
>
---
#### [replaced 017] SAM2Act: Integrating Visual Foundation Model with A Memory Architecture for Robotic Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2501.18564v4](http://arxiv.org/pdf/2501.18564v4)**

> **作者:** Haoquan Fang; Markus Grotz; Wilbert Pumacay; Yi Ru Wang; Dieter Fox; Ranjay Krishna; Jiafei Duan
>
> **备注:** Including Appendix, Project Page: https://sam2act.github.io
>
> **摘要:** Robotic manipulation systems operating in diverse, dynamic environments must exhibit three critical abilities: multitask interaction, generalization to unseen scenarios, and spatial memory. While significant progress has been made in robotic manipulation, existing approaches often fall short in generalization to complex environmental variations and addressing memory-dependent tasks. To bridge this gap, we introduce SAM2Act, a multi-view robotic transformer-based policy that leverages multi-resolution upsampling with visual representations from large-scale foundation model. SAM2Act achieves a state-of-the-art average success rate of 86.8% across 18 tasks in the RLBench benchmark, and demonstrates robust generalization on The Colosseum benchmark, with only a 4.3% performance gap under diverse environmental perturbations. Building on this foundation, we propose SAM2Act+, a memory-based architecture inspired by SAM2, which incorporates a memory bank, an encoder, and an attention mechanism to enhance spatial memory. To address the need for evaluating memory-dependent tasks, we introduce MemoryBench, a novel benchmark designed to assess spatial memory and action recall in robotic manipulation. SAM2Act+ achieves an average success rate of 94.3% on memory-based tasks in MemoryBench, significantly outperforming existing approaches and pushing the boundaries of memory-based robotic systems. Project page: sam2act.github.io.
>
---
#### [replaced 018] DriveMRP: Enhancing Vision-Language Models with Synthetic Motion Data for Motion Risk Prediction
- **分类: cs.CV; cs.AI; cs.RO; I.4.8; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2507.02948v3](http://arxiv.org/pdf/2507.02948v3)**

> **作者:** Zhiyi Hou; Enhui Ma; Fang Li; Zhiyi Lai; Kalok Ho; Zhanqian Wu; Lijun Zhou; Long Chen; Chitian Sun; Haiyang Sun; Bing Wang; Guang Chen; Hangjun Ye; Kaicheng Yu
>
> **备注:** 12 pages, 4 figures. Code available at https://github.com/hzy138/DriveMRP
>
> **摘要:** Autonomous driving has seen significant progress, driven by extensive real-world data. However, in long-tail scenarios, accurately predicting the safety of the ego vehicle's future motion remains a major challenge due to uncertainties in dynamic environments and limitations in data coverage. In this work, we aim to explore whether it is possible to enhance the motion risk prediction capabilities of Vision-Language Models (VLM) by synthesizing high-risk motion data. Specifically, we introduce a Bird's-Eye View (BEV) based motion simulation method to model risks from three aspects: the ego-vehicle, other vehicles, and the environment. This allows us to synthesize plug-and-play, high-risk motion data suitable for VLM training, which we call DriveMRP-10K. Furthermore, we design a VLM-agnostic motion risk estimation framework, named DriveMRP-Agent. This framework incorporates a novel information injection strategy for global context, ego-vehicle perspective, and trajectory projection, enabling VLMs to effectively reason about the spatial relationships between motion waypoints and the environment. Extensive experiments demonstrate that by fine-tuning with DriveMRP-10K, our DriveMRP-Agent framework can significantly improve the motion risk prediction performance of multiple VLM baselines, with the accident recognition accuracy soaring from 27.13% to 88.03%. Moreover, when tested via zero-shot evaluation on an in-house real-world high-risk motion dataset, DriveMRP-Agent achieves a significant performance leap, boosting the accuracy from base_model's 29.42% to 68.50%, which showcases the strong generalization capabilities of our method in real-world scenarios.
>
---
#### [replaced 019] Smart Ankleband for Plug-and-Play Hand-Prosthetic Control
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.17846v2](http://arxiv.org/pdf/2503.17846v2)**

> **作者:** Dean Zadok; Oren Salzman; Alon Wolf; Alex M. Bronstein
>
> **摘要:** Building robotic prostheses requires a sensor-based interface designed to provide the robotic hand with the control required to perform hand gestures. Traditional Electromyography (EMG) based prosthetics and emerging alternatives often face limitations such as muscle-activation limitations, high cost, and complex calibrations. In this paper, we present a low-cost robotic system composed of a smart ankleband for intuitive, calibration-free control of a robotic hand, and a robotic prosthetic hand that executes actions corresponding to leg gestures. The ankleband integrates an Inertial Measurement Unit (IMU) sensor with a lightweight neural network to infer user-intended leg gestures from motion data. Our system represents a significant step towards higher adoption rates of robotic prostheses among arm amputees, as it enables one to operate a prosthetic hand using a low-cost, low-power, and calibration-free solution. To evaluate our work, we collected data from 10 subjects and tested our prototype ankleband with a robotic hand on an individual with an upper-limb amputation. Our results demonstrate that this system empowers users to perform daily tasks more efficiently, requiring few compensatory movements.
>
---
#### [replaced 020] Safe Navigation in Uncertain Crowded Environments Using Risk Adaptive CVaR Barrier Functions
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.06513v3](http://arxiv.org/pdf/2504.06513v3)**

> **作者:** Xinyi Wang; Taekyung Kim; Bardh Hoxha; Georgios Fainekos; Dimitra Panagou
>
> **摘要:** Robot navigation in dynamic, crowded environments poses a significant challenge due to the inherent uncertainties in the obstacle model. In this work, we propose a risk-adaptive approach based on the Conditional Value-at-Risk Barrier Function (CVaR-BF), where the risk level is automatically adjusted to accept the minimum necessary risk, achieving a good performance in terms of safety and optimization feasibility under uncertainty. Additionally, we introduce a dynamic zone-based barrier function which characterizes the collision likelihood by evaluating the relative state between the robot and the obstacle. By integrating risk adaptation with this new function, our approach adaptively expands the safety margin, enabling the robot to proactively avoid obstacles in highly dynamic environments. Comparisons and ablation studies demonstrate that our method outperforms existing social navigation approaches, and validate the effectiveness of our proposed framework.
>
---
#### [replaced 021] Video Individual Counting for Moving Drones
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.10701v2](http://arxiv.org/pdf/2503.10701v2)**

> **作者:** Yaowu Fan; Jia Wan; Tao Han; Antoni B. Chan; Andy J. Ma
>
> **备注:** This work has been accepted to ICCV 2025
>
> **摘要:** Video Individual Counting (VIC) has received increasing attention for its importance in intelligent video surveillance. Existing works are limited in two aspects, i.e., dataset and method. Previous datasets are captured with fixed or rarely moving cameras with relatively sparse individuals, restricting evaluation for a highly varying view and time in crowded scenes. Existing methods rely on localization followed by association or classification, which struggle under dense and dynamic conditions due to inaccurate localization of small targets. To address these issues, we introduce the MovingDroneCrowd Dataset, featuring videos captured by fast-moving drones in crowded scenes under diverse illuminations, shooting heights and angles. We further propose a Shared Density map-guided Network (SDNet) using a Depth-wise Cross-Frame Attention (DCFA) module to directly estimate shared density maps between consecutive frames, from which the inflow and outflow density maps are derived by subtracting the shared density maps from the global density maps. The inflow density maps across frames are summed up to obtain the number of unique pedestrians in a video. Experiments on our datasets and publicly available ones show the superiority of our method over the state of the arts in highly dynamic and complex crowded scenes. Our dataset and codes have been released publicly.
>
---
#### [replaced 022] Riemannian Time Warping: Multiple Sequence Alignment in Curved Spaces
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.01635v3](http://arxiv.org/pdf/2506.01635v3)**

> **作者:** Julian Richter; Christopher A. Erdös; Christian Scheurer; Jochen J. Steil; Niels Dehio
>
> **摘要:** Temporal alignment of multiple signals through time warping is crucial in many fields, such as classification within speech recognition or robot motion learning. Almost all related works are limited to data in Euclidean space. Although an attempt was made in 2011 to adapt this concept to unit quaternions, a general extension to Riemannian manifolds remains absent. Given its importance for numerous applications in robotics and beyond, we introduce Riemannian Time Warping (RTW). This novel approach efficiently aligns multiple signals by considering the geometric structure of the Riemannian manifold in which the data is embedded. Extensive experiments on synthetic and real-world data, including tests with an LBR iiwa robot, demonstrate that RTW consistently outperforms state-of-the-art baselines in both averaging and classification tasks.
>
---
#### [replaced 023] A Review of Reward Functions for Reinforcement Learning in the context of Autonomous Driving
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.01440v2](http://arxiv.org/pdf/2405.01440v2)**

> **作者:** Ahmed Abouelazm; Jonas Michel; J. Marius Zoellner
>
> **备注:** Accepted at the 35th IEEE Intelligent Vehicles Symposium (IV 2024)
>
> **摘要:** Reinforcement learning has emerged as an important approach for autonomous driving. A reward function is used in reinforcement learning to establish the learned skill objectives and guide the agent toward the optimal policy. Since autonomous driving is a complex domain with partly conflicting objectives with varying degrees of priority, developing a suitable reward function represents a fundamental challenge. This paper aims to highlight the gap in such function design by assessing different proposed formulations in the literature and dividing individual objectives into Safety, Comfort, Progress, and Traffic Rules compliance categories. Additionally, the limitations of the reviewed reward functions are discussed, such as objectives aggregation and indifference to driving context. Furthermore, the reward categories are frequently inadequately formulated and lack standardization. This paper concludes by proposing future research that potentially addresses the observed shortcomings in rewards, including a reward validation framework and structured rewards that are context-aware and able to resolve conflicts.
>
---
#### [replaced 024] Stream Function-Based Navigation for Complex Quadcopter Obstacle Avoidance
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.06787v2](http://arxiv.org/pdf/2507.06787v2)**

> **作者:** Sean Smith; Emmanuel Witrant; Ya-Jun Pan
>
> **摘要:** This article presents a novel stream function-based navigational control system for obstacle avoidance, where obstacles are represented as two-dimensional (2D) rigid surfaces in inviscid, incompressible flows. The approach leverages the vortex panel method (VPM) and incorporates safety margins to control the stream function and flow properties around virtual surfaces, enabling navigation in complex, partially observed environments using real-time sensing. To address the limitations of the VPM in managing relative distance and avoiding rapidly accelerating obstacles at close proximity, the system integrates a model predictive controller (MPC) based on higher-order control barrier functions (HOCBF). This integration incorporates VPM trajectory generation, state estimation, and constraint handling into a receding-horizon optimization problem. The 2D rigid surfaces are enclosed using minimum bounding ellipses (MBEs), while an adaptive Kalman filter (AKF) captures and predicts obstacle dynamics, propagating these estimates into the MPC-HOCBF for rapid avoidance maneuvers. Evaluation is conducted using a PX4-powered Clover drone Gazebo simulator and real-time experiments involving a COEX Clover quadcopter equipped with a 360 degree LiDAR sensor.
>
---
#### [replaced 025] RoboEngine: Plug-and-Play Robot Data Augmentation with Semantic Robot Segmentation and Background Generation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.18738v2](http://arxiv.org/pdf/2503.18738v2)**

> **作者:** Chengbo Yuan; Suraj Joshi; Shaoting Zhu; Hang Su; Hang Zhao; Yang Gao
>
> **备注:** Project Page: https://roboengine.github.io/
>
> **摘要:** Visual augmentation has become a crucial technique for enhancing the visual robustness of imitation learning. However, existing methods are often limited by prerequisites such as camera calibration or the need for controlled environments (e.g., green screen setups). In this work, we introduce RoboEngine, the first plug-and-play visual robot data augmentation toolkit. For the first time, users can effortlessly generate physics- and task-aware robot scenes with just a few lines of code. To achieve this, we present a novel robot scene segmentation dataset, a generalizable high-quality robot segmentation model, and a fine-tuned background generation model, which together form the core components of the out-of-the-box toolkit. Using RoboEngine, we demonstrate the ability to generalize robot manipulation tasks across six entirely new scenes, based solely on demonstrations collected from a single scene, achieving a more than 200% performance improvement compared to the no-augmentation baseline. All datasets, model weights, and the toolkit are released https://roboengine.github.io/
>
---
#### [replaced 026] Large Language Model-Driven Closed-Loop UAV Operation with Semantic Observations
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.01930v3](http://arxiv.org/pdf/2507.01930v3)**

> **作者:** Wenhao Wang; Yanyan Li; Long Jiao; Jiawei Yuan
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Recent advances in large Language Models (LLMs) have revolutionized mobile robots, including unmanned aerial vehicles (UAVs), enabling their intelligent operation within Internet of Things (IoT) ecosystems. However, LLMs still face challenges from logical reasoning and complex decision-making, leading to concerns about the reliability of LLM-driven UAV operations in IoT applications. In this paper, we propose a LLM-driven closed-loop control framework that enables reliable UAV operations powered by effective feedback and refinement using two LLM modules, i.e., a Code Generator and an Evaluator. Our framework transforms numerical state observations from UAV operations into natural language trajectory descriptions to enhance the evaluator LLM's understanding of UAV dynamics for precise feedback generation. Our framework also enables a simulation-based refinement process, and hence eliminates the risks to physical UAVs caused by incorrect code execution during the refinement. Extensive experiments on UAV control tasks with different complexities are conducted. The experimental results show that our framework can achieve reliable UAV operations using LLMs, which significantly outperforms baseline approaches in terms of success rate and completeness with the increase of task complexity.
>
---
#### [replaced 027] Accurate Simulation and Parameter Identification of Deformable Linear Objects using Discrete Elastic Rods in Generalized Coordinates
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2310.00911v3](http://arxiv.org/pdf/2310.00911v3)**

> **作者:** Qi Jing Chen; Timothy Bretl; Quang-Cuong Pham
>
> **备注:** 7 pages, 6 figures
>
> **摘要:** This paper presents a fast and accurate model of a deformable linear object (DLO) -- e.g., a rope, wire, or cable -- integrated into an established robot physics simulator, MuJoCo. Most accurate DLO models with low computational times exist in standalone numerical simulators, which are unable or require tedious work to handle external objects. Based on an existing state-of-the-art DLO model -- Discrete Elastic Rods (DER) -- our implementation provides an improvement in accuracy over MuJoCo's own native cable model. To minimize computational load, our model utilizes force-lever analysis to adapt the Cartesian stiffness forces of the DER into its generalized coordinates. As a key contribution, we introduce a novel parameter identification pipeline designed for both simplicity and accuracy, which we utilize to determine the bending and twisting stiffness of three distinct DLOs. We then evaluate the performance of each model by simulating the DLOs and comparing them to their real-world counterparts and against theoretically proven validation tests.
>
---
#### [replaced 028] Fast Bilateral Teleoperation and Imitation Learning Using Sensorless Force Control via Accurate Dynamics Model
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.06174v2](http://arxiv.org/pdf/2507.06174v2)**

> **作者:** Koki Yamane; Yunhan Li; Masashi Konosu; Koki Inami; Junji Oaki; Sho Sakaino; Toshiaki Tsuji
>
> **备注:** 20 pages, 9 figures, Submitted to CoRL 2025
>
> **摘要:** In recent years, the advancement of imitation learning has led to increased interest in teleoperating low-cost manipulators to collect demonstration data. However, most existing systems rely on unilateral control, which only transmits target position values. While this approach is easy to implement and suitable for slow, non-contact tasks, it struggles with fast or contact-rich operations due to the absence of force feedback. This work demonstrates that fast teleoperation with force feedback is feasible even with force-sensorless, low-cost manipulators by leveraging 4-channel bilateral control. Based on accurately identified manipulator dynamics, our method integrates nonlinear terms compensation, velocity and external force estimation, and variable gain corresponding to inertial variation. Furthermore, using data collected by 4-channel bilateral control, we show that incorporating force information into both the input and output of learned policies improves performance in imitation learning. These results highlight the practical effectiveness of our system for high-fidelity teleoperation and data collection on affordable hardware.
>
---
#### [replaced 029] Uncertainty-Informed Active Perception for Open Vocabulary Object Goal Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.13367v2](http://arxiv.org/pdf/2506.13367v2)**

> **作者:** Utkarsh Bajpai; Julius Rückin; Cyrill Stachniss; Marija Popović
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Mobile robots exploring indoor environments increasingly rely on vision-language models to perceive high-level semantic cues in camera images, such as object categories. Such models offer the potential to substantially advance robot behaviour for tasks such as object-goal navigation (ObjectNav), where the robot must locate objects specified in natural language by exploring the environment. Current ObjectNav methods heavily depend on prompt engineering for perception and do not address the semantic uncertainty induced by variations in prompt phrasing. Ignoring semantic uncertainty can lead to suboptimal exploration, which in turn limits performance. Hence, we propose a semantic uncertainty-informed active perception pipeline for ObjectNav in indoor environments. We introduce a novel probabilistic sensor model for quantifying semantic uncertainty in vision-language models and incorporate it into a probabilistic geometric-semantic map to enhance spatial understanding. Based on this map, we develop a frontier exploration planner with an uncertainty-informed multi-armed bandit objective to guide efficient object search. Experimental results demonstrate that our method achieves ObjectNav success rates comparable to those of state-of-the-art approaches, without requiring extensive prompt engineering.
>
---
#### [replaced 030] Diffusion Models for Robotic Manipulation: A Survey
- **分类: cs.RO; stat.ML**

- **链接: [http://arxiv.org/pdf/2504.08438v3](http://arxiv.org/pdf/2504.08438v3)**

> **作者:** Rosa Wolf; Yitian Shi; Sheng Liu; Rania Rayyes
>
> **备注:** 26 pages, 2 figure, 9 tables
>
> **摘要:** Diffusion generative models have demonstrated remarkable success in visual domains such as image and video generation. They have also recently emerged as a promising approach in robotics, especially in robot manipulations. Diffusion models leverage a probabilistic framework, and they stand out with their ability to model multi-modal distributions and their robustness to high-dimensional input and output spaces. This survey provides a comprehensive review of state-of-the-art diffusion models in robotic manipulation, including grasp learning, trajectory planning, and data augmentation. Diffusion models for scene and image augmentation lie at the intersection of robotics and computer vision for vision-based tasks to enhance generalizability and data scarcity. This paper also presents the two main frameworks of diffusion models and their integration with imitation learning and reinforcement learning. In addition, it discusses the common architectures and benchmarks and points out the challenges and advantages of current state-of-the-art diffusion-based methods.
>
---
#### [replaced 031] A Noise-Robust Turn-Taking System for Real-World Dialogue Robots: A Field Experiment
- **分类: cs.RO; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.06241v2](http://arxiv.org/pdf/2503.06241v2)**

> **作者:** Koji Inoue; Yuki Okafuji; Jun Baba; Yoshiki Ohira; Katsuya Hyodo; Tatsuya Kawahara
>
> **备注:** This paper has been accepted for presentation at IEEE/RSJ International Conference on Intelligent Robots and Systems 2025 (IROS 2025) and represents the author's version of the work
>
> **摘要:** Turn-taking is a crucial aspect of human-robot interaction, directly influencing conversational fluidity and user engagement. While previous research has explored turn-taking models in controlled environments, their robustness in real-world settings remains underexplored. In this study, we propose a noise-robust voice activity projection (VAP) model, based on a Transformer architecture, to enhance real-time turn-taking in dialogue robots. To evaluate the effectiveness of the proposed system, we conducted a field experiment in a shopping mall, comparing the VAP system with a conventional cloud-based speech recognition system. Our analysis covered both subjective user evaluations and objective behavioral analysis. The results showed that the proposed system significantly reduced response latency, leading to a more natural conversation where both the robot and users responded faster. The subjective evaluations suggested that faster responses contribute to a better interaction experience.
>
---
#### [replaced 032] DexSim2Real$^{2}$: Building Explicit World Model for Precise Articulated Object Dexterous Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.08750v2](http://arxiv.org/pdf/2409.08750v2)**

> **作者:** Taoran Jiang; Yixuan Guan; Liqian Ma; Jing Xu; Jiaojiao Meng; Weihang Chen; Zecui Zeng; Lusong Li; Dan Wu; Rui Chen
>
> **备注:** Project Webpage: https://jiangtaoran.github.io/dexsim2real2web/ . arXiv admin note: text overlap with arXiv:2302.10693
>
> **摘要:** Articulated objects are ubiquitous in daily life. In this paper, we present DexSim2Real$^{2}$, a novel framework for goal-conditioned articulated object manipulation. The core of our framework is constructing an explicit world model of unseen articulated objects through active interactions, which enables sampling-based model predictive control to plan trajectories achieving different goals without requiring demonstrations or RL. It first predicts an interaction using an affordance network trained on self-supervised interaction data or videos of human manipulation. After executing the interactions on the real robot to move the object parts, we propose a novel modeling pipeline based on 3D AIGC to build a digital twin of the object in simulation from multiple frames of observations. For dexterous hands, we utilize eigengrasp to reduce the action dimension, enabling more efficient trajectory searching. Experiments validate the framework's effectiveness for precise manipulation using a suction gripper, a two-finger gripper and two dexterous hand. The generalizability of the explicit world model also enables advanced manipulation strategies like manipulating with tools.
>
---
#### [replaced 033] Learning Decentralized Multi-Biped Control for Payload Transport
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.17279v2](http://arxiv.org/pdf/2406.17279v2)**

> **作者:** Bikram Pandit; Ashutosh Gupta; Mohitvishnu S. Gadde; Addison Johnson; Aayam Kumar Shrestha; Helei Duan; Jeremy Dao; Alan Fern
>
> **备注:** Submitted to CoRL 2024, Project website: decmbc.github.io
>
> **摘要:** Payload transport over flat terrain via multi-wheel robot carriers is well-understood, highly effective, and configurable. In this paper, our goal is to provide similar effectiveness and configurability for transport over rough terrain that is more suitable for legs rather than wheels. For this purpose, we consider multi-biped robot carriers, where wheels are replaced by multiple bipedal robots attached to the carrier. Our main contribution is to design a decentralized controller for such systems that can be effectively applied to varying numbers and configurations of rigidly attached bipedal robots without retraining. We present a reinforcement learning approach for training the controller in simulation that supports transfer to the real world. Our experiments in simulation provide quantitative metrics showing the effectiveness of the approach over a wide variety of simulated transport scenarios. In addition, we demonstrate the controller in the real-world for systems composed of two and three Cassie robots. To our knowledge, this is the first example of a scalable multi-biped payload transport system.
>
---
#### [replaced 034] LSTP-Nav: Lightweight Spatiotemporal Policy for Map-free Multi-agent Navigation with LiDAR
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2408.16370v4](http://arxiv.org/pdf/2408.16370v4)**

> **作者:** Xingrong Diao; Zhirui Sun; Jianwei Peng; Jiankun Wang
>
> **摘要:** Safe and efficient multi-agent navigation in dynamic environments remains inherently challenging, particularly when real-time decision-making is required on resource-constrained platforms. Ensuring collision-free trajectories while adapting to uncertainties without relying on pre-built maps further complicates real-world deployment. To address these challenges, we propose LSTP-Nav, a lightweight end-to-end policy for multi-agent navigation that enables map-free collision avoidance in complex environments by directly mapping raw LiDAR point clouds to motion commands. At the core of this framework lies LSTP-Net, an efficient network that processes raw LiDAR data using a GRU architecture, enhanced with attention mechanisms to dynamically focus on critical environmental features while minimizing computational overhead. Additionally, a novel HS reward optimizes collision avoidance by incorporating angular velocity, prioritizing obstacles along the predicted heading, and enhancing training stability. To narrow the sim-to-real gap, we develop PhysReplay-Simlab, a physics-realistic multi-agent simulator, employs localized replay to mine near-failure experiences. Relying solely on LiDA, LSTP-Nav achieves efficient zero-shot sim-to-real transfer on a CPU-only robotic platform, enabling robust navigation in dynamic environments while maintaining computation frequencies above 40 Hz. Extensive experiments demonstrate that LSTP-Nav outperforms baselines with a 9.58\% higher success rate and a 12.30\% lower collision rate, underscoring its practicality and robustness for real-world applications.
>
---
