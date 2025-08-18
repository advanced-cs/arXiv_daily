# 机器人 cs.RO

- **最新发布 36 篇**

- **更新 20 篇**

## 最新发布

#### [new 001] Open, Reproducible and Trustworthy Robot-Based Experiments with Virtual Labs and Digital-Twin-Based Execution Tracing
- **分类: cs.RO; cs.AI; 68T40; I.2.9**

- **简介: 该论文属于机器人科学实验任务，旨在解决实验可重复性与透明度问题。提出执行追踪框架和虚拟实验室平台，实现机器人实验的开放、可信与可复现。**

- **链接: [http://arxiv.org/pdf/2508.11406v1](http://arxiv.org/pdf/2508.11406v1)**

> **作者:** Benjamin Alt; Mareike Picklum; Sorin Arion; Franklin Kenghagho Kenfack; Michael Beetz
>
> **备注:** 8 pages, 6 figures, submitted to the 1st IROS Workshop on Embodied AI and Robotics for Future Scientific Discovery
>
> **摘要:** We envision a future in which autonomous robots conduct scientific experiments in ways that are not only precise and repeatable, but also open, trustworthy, and transparent. To realize this vision, we present two key contributions: a semantic execution tracing framework that logs sensor data together with semantically annotated robot belief states, ensuring that automated experimentation is transparent and replicable; and the AICOR Virtual Research Building (VRB), a cloud-based platform for sharing, replicating, and validating robot task executions at scale. Together, these tools enable reproducible, robot-driven science by integrating deterministic execution, semantic memory, and open knowledge representation, laying the foundation for autonomous systems to participate in scientific discovery.
>
---
#### [new 002] A Recursive Total Least Squares Solution for Bearing-Only Target Motion Analysis and Circumnavigation
- **分类: cs.RO**

- **简介: 该论文属于目标运动分析任务，解决无距离信息的方位跟踪问题。提出递归总最小二乘方法和环绕控制器，提升定位精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2508.11289v1](http://arxiv.org/pdf/2508.11289v1)**

> **作者:** Lin Li; Xueming Liu; Zhoujingzi Qiu; Tianjiang Hu; Qingrui Zhang
>
> **备注:** Accepted by 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 6 Pages
>
> **摘要:** Bearing-only Target Motion Analysis (TMA) is a promising technique for passive tracking in various applications as a bearing angle is easy to measure. Despite its advantages, bearing-only TMA is challenging due to the nonlinearity of the bearing measurement model and the lack of range information, which impairs observability and estimator convergence. This paper addresses these issues by proposing a Recursive Total Least Squares (RTLS) method for online target localization and tracking using mobile observers. The RTLS approach, inspired by previous results on Total Least Squares (TLS), mitigates biases in position estimation and improves computational efficiency compared to pseudo-linear Kalman filter (PLKF) methods. Additionally, we propose a circumnavigation controller to enhance system observability and estimator convergence by guiding the mobile observer in orbit around the target. Extensive simulations and experiments are performed to demonstrate the effectiveness and robustness of the proposed method. The proposed algorithm is also compared with the state-of-the-art approaches, which confirms its superior performance in terms of both accuracy and stability.
>
---
#### [new 003] Tactile Robotics: An Outlook
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.11261v1](http://arxiv.org/pdf/2508.11261v1)**

> **作者:** Shan Luo; Nathan F. Lepora; Wenzhen Yuan; Kaspar Althoefer; Gordon Cheng; Ravinder Dahiya
>
> **备注:** 20 pages, 2 figures, accepted to IEEE Transactions on Robotics
>
> **摘要:** Robotics research has long sought to give robots the ability to perceive the physical world through touch in an analogous manner to many biological systems. Developing such tactile capabilities is important for numerous emerging applications that require robots to co-exist and interact closely with humans. Consequently, there has been growing interest in tactile sensing, leading to the development of various technologies, including piezoresistive and piezoelectric sensors, capacitive sensors, magnetic sensors, and optical tactile sensors. These diverse approaches utilise different transduction methods and materials to equip robots with distributed sensing capabilities, enabling more effective physical interactions. These advances have been supported in recent years by simulation tools that generate large-scale tactile datasets to support sensor designs and algorithms to interpret and improve the utility of tactile data. The integration of tactile sensing with other modalities, such as vision, as well as with action strategies for active tactile perception highlights the growing scope of this field. To further the transformative progress in tactile robotics, a holistic approach is essential. In this outlook article, we examine several challenges associated with the current state of the art in tactile robotics and explore potential solutions to inspire innovations across multiple domains, including manufacturing, healthcare, recycling and agriculture.
>
---
#### [new 004] GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉强化学习任务，旨在解决生成数据质量依赖和精细操作难题。通过引入生成式对象中心流，提取低维特征以学习泛化性强的策略。**

- **链接: [http://arxiv.org/pdf/2508.11049v1](http://arxiv.org/pdf/2508.11049v1)**

> **作者:** Kelin Yu; Sheng Zhang; Harshit Soora; Furong Huang; Heng Huang; Pratap Tokekar; Ruohan Gao
>
> **备注:** Published at ICCV 2025
>
> **摘要:** Recent advances have shown that video generation models can enhance robot learning by deriving effective robot actions through inverse dynamics. However, these methods heavily depend on the quality of generated data and struggle with fine-grained manipulation due to the lack of environment feedback. While video-based reinforcement learning improves policy robustness, it remains constrained by the uncertainty of video generation and the challenges of collecting large-scale robot datasets for training diffusion models. To address these limitations, we propose GenFlowRL, which derives shaped rewards from generated flow trained from diverse cross-embodiment datasets. This enables learning generalizable and robust policies from diverse demonstrations using low-dimensional, object-centric features. Experiments on 10 manipulation tasks, both in simulation and real-world cross-embodiment evaluations, demonstrate that GenFlowRL effectively leverages manipulation features extracted from generated object-centric flow, consistently achieving superior performance across diverse and challenging scenarios. Our Project Page: https://colinyu1.github.io/genflowrl
>
---
#### [new 005] Sim2Dust: Mastering Dynamic Waypoint Tracking on Granular Media
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于自主导航任务，解决行星表面动态路径跟踪问题。通过强化学习与仿真训练，提升机器人在松散地形上的导航能力。**

- **链接: [http://arxiv.org/pdf/2508.11503v1](http://arxiv.org/pdf/2508.11503v1)**

> **作者:** Andrej Orsula; Matthieu Geist; Miguel Olivares-Mendez; Carol Martinez
>
> **备注:** The source code is available at https://github.com/AndrejOrsula/space_robotics_bench
>
> **摘要:** Reliable autonomous navigation across the unstructured terrains of distant planetary surfaces is a critical enabler for future space exploration. However, the deployment of learning-based controllers is hindered by the inherent sim-to-real gap, particularly for the complex dynamics of wheel interactions with granular media. This work presents a complete sim-to-real framework for developing and validating robust control policies for dynamic waypoint tracking on such challenging surfaces. We leverage massively parallel simulation to train reinforcement learning agents across a vast distribution of procedurally generated environments with randomized physics. These policies are then transferred zero-shot to a physical wheeled rover operating in a lunar-analogue facility. Our experiments systematically compare multiple reinforcement learning algorithms and action smoothing filters to identify the most effective combinations for real-world deployment. Crucially, we provide strong empirical evidence that agents trained with procedural diversity achieve superior zero-shot performance compared to those trained on static scenarios. We also analyze the trade-offs of fine-tuning with high-fidelity particle physics, which offers minor gains in low-speed precision at a significant computational cost. Together, these contributions establish a validated workflow for creating reliable learning-based navigation systems, marking a critical step towards deploying autonomous robots in the final frontier.
>
---
#### [new 006] Visuomotor Grasping with World Models for Surgical Robots
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于手术机器人视觉抓取任务，旨在解决泛化性差、依赖人工特征和环境复杂等问题。提出GASv2框架，实现无需重训练的通用抓取。**

- **链接: [http://arxiv.org/pdf/2508.11200v1](http://arxiv.org/pdf/2508.11200v1)**

> **作者:** Hongbin Lin; Bin Li; Kwok Wai Samuel Au
>
> **摘要:** Grasping is a fundamental task in robot-assisted surgery (RAS), and automating it can reduce surgeon workload while enhancing efficiency, safety, and consistency beyond teleoperated systems. Most prior approaches rely on explicit object pose tracking or handcrafted visual features, limiting their generalization to novel objects, robustness to visual disturbances, and the ability to handle deformable objects. Visuomotor learning offers a promising alternative, but deploying it in RAS presents unique challenges, such as low signal-to-noise ratio in visual observations, demands for high safety and millimeter-level precision, as well as the complex surgical environment. This paper addresses three key challenges: (i) sim-to-real transfer of visuomotor policies to ex vivo surgical scenes, (ii) visuomotor learning using only a single stereo camera pair -- the standard RAS setup, and (iii) object-agnostic grasping with a single policy that generalizes to diverse, unseen surgical objects without retraining or task-specific models. We introduce Grasp Anything for Surgery V2 (GASv2), a visuomotor learning framework for surgical grasping. GASv2 leverages a world-model-based architecture and a surgical perception pipeline for visual observations, combined with a hybrid control system for safe execution. We train the policy in simulation using domain randomization for sim-to-real transfer and deploy it on a real robot in both phantom-based and ex vivo surgical settings, using only a single pair of endoscopic cameras. Extensive experiments show our policy achieves a 65% success rate in both settings, generalizes to unseen objects and grippers, and adapts to diverse disturbances, demonstrating strong performance, generality, and robustness.
>
---
#### [new 007] Geometry-Aware Predictive Safety Filters on Humanoids: From Poisson Safety Functions to CBF Constrained MPC
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人安全导航任务，解决动态环境中足式机器人轨迹规划问题。通过结合泊松安全函数与控制屏障函数，提出一种在线预测安全过滤器。**

- **链接: [http://arxiv.org/pdf/2508.11129v1](http://arxiv.org/pdf/2508.11129v1)**

> **作者:** Ryan M. Bena; Gilbert Bahati; Blake Werner; Ryan K. Cosner; Lizhi Yang; Aaron D. Ames
>
> **备注:** 2025 IEEE-RAS 24th International Conference on Humanoid Robots
>
> **摘要:** Autonomous navigation through unstructured and dynamically-changing environments is a complex task that continues to present many challenges for modern roboticists. In particular, legged robots typically possess manipulable asymmetric geometries which must be considered during safety-critical trajectory planning. This work proposes a predictive safety filter: a nonlinear model predictive control (MPC) algorithm for online trajectory generation with geometry-aware safety constraints based on control barrier functions (CBFs). Critically, our method leverages Poisson safety functions to numerically synthesize CBF constraints directly from perception data. We extend the theoretical framework for Poisson safety functions to incorporate temporal changes in the domain by reformulating the static Dirichlet problem for Poisson's equation as a parameterized moving boundary value problem. Furthermore, we employ Minkowski set operations to lift the domain into a configuration space that accounts for robot geometry. Finally, we implement our real-time predictive safety filter on humanoid and quadruped robots in various safety-critical scenarios. The results highlight the versatility of Poisson safety functions, as well as the benefit of CBF constrained model predictive safety-critical controllers.
>
---
#### [new 008] EvoPSF: Online Evolution of Autonomous Driving Models via Planning-State Feedback
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，解决模型在部署后无法适应新环境的问题。通过在线进化框架EvoPSF，利用规划不确定性触发模型更新，提升运动预测和路径规划的准确性与稳定性。**

- **链接: [http://arxiv.org/pdf/2508.11453v1](http://arxiv.org/pdf/2508.11453v1)**

> **作者:** Jiayue Jin; Lang Qian; Jingyu Zhang; Chuanyu Ju; Liang Song
>
> **摘要:** Recent years have witnessed remarkable progress in autonomous driving, with systems evolving from modular pipelines to end-to-end architectures. However, most existing methods are trained offline and lack mechanisms to adapt to new environments during deployment. As a result, their generalization ability diminishes when faced with unseen variations in real-world driving scenarios. In this paper, we break away from the conventional "train once, deploy forever" paradigm and propose EvoPSF, a novel online Evolution framework for autonomous driving based on Planning-State Feedback. We argue that planning failures are primarily caused by inaccurate object-level motion predictions, and such failures are often reflected in the form of increased planner uncertainty. To address this, we treat planner uncertainty as a trigger for online evolution, using it as a diagnostic signal to initiate targeted model updates. Rather than performing blind updates, we leverage the planner's agent-agent attention to identify the specific objects that the ego vehicle attends to most, which are primarily responsible for the planning failures. For these critical objects, we compute a targeted self-supervised loss by comparing their predicted waypoints from the prediction module with their actual future positions, selected from the perception module's outputs with high confidence scores. This loss is then backpropagated to adapt the model online. As a result, our method improves the model's robustness to environmental changes, leads to more precise motion predictions, and therefore enables more accurate and stable planning behaviors. Experiments on both cross-region and corrupted variants of the nuScenes dataset demonstrate that EvoPSF consistently improves planning performance under challenging conditions.
>
---
#### [new 009] OVSegDT: Segmenting Transformer for Open-Vocabulary Object Goal Navigation
- **分类: cs.RO**

- **简介: 该论文属于开放词汇目标导航任务，解决代理在未见类别上泛化差和碰撞多的问题。提出OVSegDT模型，结合语义分支和熵自适应损失，提升导航性能与安全性。**

- **链接: [http://arxiv.org/pdf/2508.11479v1](http://arxiv.org/pdf/2508.11479v1)**

> **作者:** Tatiana Zemskova; Aleksei Staroverov; Dmitry Yudin; Aleksandr Panov
>
> **摘要:** Open-vocabulary Object Goal Navigation requires an embodied agent to reach objects described by free-form language, including categories never seen during training. Existing end-to-end policies overfit small simulator datasets, achieving high success on training scenes but failing to generalize and exhibiting unsafe behaviour (frequent collisions). We introduce OVSegDT, a lightweight transformer policy that tackles these issues with two synergistic components. The first component is the semantic branch, which includes an encoder for the target binary mask and an auxiliary segmentation loss function, grounding the textual goal and providing precise spatial cues. The second component consists of a proposed Entropy-Adaptive Loss Modulation, a per-sample scheduler that continuously balances imitation and reinforcement signals according to the policy entropy, eliminating brittle manual phase switches. These additions cut the sample complexity of training by 33%, and reduce collision count in two times while keeping inference cost low (130M parameters, RGB-only input). On HM3D-OVON, our model matches the performance on unseen categories to that on seen ones and establishes state-of-the-art results (40.1% SR, 20.9% SPL on val unseen) without depth, odometry, or large vision-language models. Code is available at https://github.com/CognitiveAISystems/OVSegDT.
>
---
#### [new 010] Pedestrian Dead Reckoning using Invariant Extended Kalman Filter
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，解决GPS缺失环境下的行人导航问题。提出一种基于不变扩展卡尔曼滤波的惯性导航方法，通过站立脚伪测量提高定位精度。**

- **链接: [http://arxiv.org/pdf/2508.11396v1](http://arxiv.org/pdf/2508.11396v1)**

> **作者:** Jingran Zhang; Zhengzhang Yan; Yiming Chen; Zeqiang He; Jiahao Chen
>
> **摘要:** This paper presents a cost-effective inertial pedestrian dead reckoning method for the bipedal robot in the GPS-denied environment. Each time when the inertial measurement unit (IMU) is on the stance foot, a stationary pseudo-measurement can be executed to provide innovation to the IMU measurement based prediction. The matrix Lie group based theoretical development of the adopted invariant extended Kalman filter (InEKF) is set forth for tutorial purpose. Three experiments are conducted to compare between InEKF and standard EKF, including motion capture benchmark experiment, large-scale multi-floor walking experiment, and bipedal robot experiment, as an effort to show our method's feasibility in real-world robot system. In addition, a sensitivity analysis is included to show that InEKF is much easier to tune than EKF.
>
---
#### [new 011] 3D FlowMatch Actor: Unified 3D Policy for Single- and Dual-Arm Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决单臂和双臂操作的策略学习问题。提出3DFA模型，结合流匹配与3D视觉表示，提升训练效率与性能。**

- **链接: [http://arxiv.org/pdf/2508.11002v1](http://arxiv.org/pdf/2508.11002v1)**

> **作者:** Nikolaos Gkanatsios; Jiahe Xu; Matthew Bronars; Arsalan Mousavian; Tsung-Wei Ke; Katerina Fragkiadaki
>
> **摘要:** We present 3D FlowMatch Actor (3DFA), a 3D policy architecture for robot manipulation that combines flow matching for trajectory prediction with 3D pretrained visual scene representations for learning from demonstration. 3DFA leverages 3D relative attention between action and visual tokens during action denoising, building on prior work in 3D diffusion-based single-arm policy learning. Through a combination of flow matching and targeted system-level and architectural optimizations, 3DFA achieves over 30x faster training and inference than previous 3D diffusion-based policies, without sacrificing performance. On the bimanual PerAct2 benchmark, it establishes a new state of the art, outperforming the next-best method by an absolute margin of 41.4%. In extensive real-world evaluations, it surpasses strong baselines with up to 1000x more parameters and significantly more pretraining. In unimanual settings, it sets a new state of the art on 74 RLBench tasks by directly predicting dense end-effector trajectories, eliminating the need for motion planning. Comprehensive ablation studies underscore the importance of our design choices for both policy effectiveness and efficiency.
>
---
#### [new 012] MultiPark: Multimodal Parking Transformer with Next-Segment Prediction
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶中的停车任务，解决复杂空间停车的多模态行为生成问题。提出MultiPark模型，通过Transformer结构和多模态策略提升停车的准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.11537v1](http://arxiv.org/pdf/2508.11537v1)**

> **作者:** Han Zheng; Zikang Zhou; Guli Zhang; Zhepei Wang; Kaixuan Wang; Peiliang Li; Shaojie Shen; Ming Yang; Tong Qin
>
> **摘要:** Parking accurately and safely in highly constrained spaces remains a critical challenge. Unlike structured driving environments, parking requires executing complex maneuvers such as frequent gear shifts and steering saturation. Recent attempts to employ imitation learning (IL) for parking have achieved promising results. However, existing works ignore the multimodal nature of parking behavior in lane-free open space, failing to derive multiple plausible solutions under the same situation. Notably, IL-based methods encompass inherent causal confusion, so enabling a neural network to generalize across diverse parking scenarios is particularly difficult. To address these challenges, we propose MultiPark, an autoregressive transformer for multimodal parking. To handle paths filled with abrupt turning points, we introduce a data-efficient next-segment prediction paradigm, enabling spatial generalization and temporal extrapolation. Furthermore, we design learnable parking queries factorized into gear, longitudinal, and lateral components, parallelly decoding diverse parking behaviors. To mitigate causal confusion in IL, our method employs target-centric pose and ego-centric collision as outcome-oriented loss across all modalities beyond pure imitation loss. Evaluations on real-world datasets demonstrate that MultiPark achieves state-of-the-art performance across various scenarios. We deploy MultiPark on a production vehicle, further confirming our approach's robustness in real-world parking environments.
>
---
#### [new 013] A Comparative Study of Floating-Base Space Parameterizations for Agile Whole-Body Motion Planning
- **分类: cs.RO; cs.SY; eess.SY; math.OC**

- **简介: 该论文属于机器人运动规划任务，研究不同浮基参数化对敏捷全身运动的影响，旨在提升轨迹优化性能。**

- **链接: [http://arxiv.org/pdf/2508.11520v1](http://arxiv.org/pdf/2508.11520v1)**

> **作者:** Evangelos Tsiatsianas; Chairi Kiourt; Konstantinos Chatzilygeroudis
>
> **备注:** 8 pages, 2 figures, 4 tables, Accepted at Humanoids 2025
>
> **摘要:** Automatically generating agile whole-body motions for legged and humanoid robots remains a fundamental challenge in robotics. While numerous trajectory optimization approaches have been proposed, there is no clear guideline on how the choice of floating-base space parameterization affects performance, especially for agile behaviors involving complex contact dynamics. In this paper, we present a comparative study of different parameterizations for direct transcription-based trajectory optimization of agile motions in legged systems. We systematically evaluate several common choices under identical optimization settings to ensure a fair comparison. Furthermore, we introduce a novel formulation based on the tangent space of SE(3) for representing the robot's floating-base pose, which, to our knowledge, has not received attention from the literature. This approach enables the use of mature off-the-shelf numerical solvers without requiring specialized manifold optimization techniques. We hope that our experiments and analysis will provide meaningful insights for selecting the appropriate floating-based representation for agile whole-body motion generation.
>
---
#### [new 014] Learning Differentiable Reachability Maps for Optimization-based Humanoid Motion Generation
- **分类: cs.RO**

- **简介: 该论文属于人形机器人运动规划任务，旨在降低计算成本。通过学习可微可达性地图，将机器人末端可达区域转化为连续优化约束，提升运动生成效率。**

- **链接: [http://arxiv.org/pdf/2508.11275v1](http://arxiv.org/pdf/2508.11275v1)**

> **作者:** Masaki Murooka; Iori Kumagai; Mitsuharu Morisawa; Fumio Kanehiro
>
> **摘要:** To reduce the computational cost of humanoid motion generation, we introduce a new approach to representing robot kinematic reachability: the differentiable reachability map. This map is a scalar-valued function defined in the task space that takes positive values only in regions reachable by the robot's end-effector. A key feature of this representation is that it is continuous and differentiable with respect to task-space coordinates, enabling its direct use as constraints in continuous optimization for humanoid motion planning. We describe a method to learn such differentiable reachability maps from a set of end-effector poses generated using a robot's kinematic model, using either a neural network or a support vector machine as the learning model. By incorporating the learned reachability map as a constraint, we formulate humanoid motion generation as a continuous optimization problem. We demonstrate that the proposed approach efficiently solves various motion planning problems, including footstep planning, multi-contact motion planning, and loco-manipulation planning for humanoid robots.
>
---
#### [new 015] Relative Position Matters: Trajectory Prediction and Planning with Polar Representation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11492v1](http://arxiv.org/pdf/2508.11492v1)**

> **作者:** Bozhou Zhang; Nan Song; Bingzhao Gao; Li Zhang
>
> **摘要:** Trajectory prediction and planning in autonomous driving are highly challenging due to the complexity of predicting surrounding agents' movements and planning the ego agent's actions in dynamic environments. Existing methods encode map and agent positions and decode future trajectories in Cartesian coordinates. However, modeling the relationships between the ego vehicle and surrounding traffic elements in Cartesian space can be suboptimal, as it does not naturally capture the varying influence of different elements based on their relative distances and directions. To address this limitation, we adopt the Polar coordinate system, where positions are represented by radius and angle. This representation provides a more intuitive and effective way to model spatial changes and relative relationships, especially in terms of distance and directional influence. Based on this insight, we propose Polaris, a novel method that operates entirely in Polar coordinates, distinguishing itself from conventional Cartesian-based approaches. By leveraging the Polar representation, this method explicitly models distance and direction variations and captures relative relationships through dedicated encoding and refinement modules, enabling more structured and spatially aware trajectory prediction and planning. Extensive experiments on the challenging prediction (Argoverse 2) and planning benchmarks (nuPlan) demonstrate that Polaris achieves state-of-the-art performance.
>
---
#### [new 016] Robust Online Calibration for UWB-Aided Visual-Inertial Navigation with Bias Correction
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于视觉-惯性导航系统的校准任务，解决UWB锚点定位不准确问题，通过引入鲁棒初始化和在线优化方法提升校准精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2508.10999v1](http://arxiv.org/pdf/2508.10999v1)**

> **作者:** Yizhi Zhou; Jie Xu; Jiawei Xia; Zechen Hu; Weizi Li; Xuan Wang
>
> **摘要:** This paper presents a novel robust online calibration framework for Ultra-Wideband (UWB) anchors in UWB-aided Visual-Inertial Navigation Systems (VINS). Accurate anchor positioning, a process known as calibration, is crucial for integrating UWB ranging measurements into state estimation. While several prior works have demonstrated satisfactory results by using robot-aided systems to autonomously calibrate UWB systems, there are still some limitations: 1) these approaches assume accurate robot localization during the initialization step, ignoring localization errors that can compromise calibration robustness, and 2) the calibration results are highly sensitive to the initial guess of the UWB anchors' positions, reducing the practical applicability of these methods in real-world scenarios. Our approach addresses these challenges by explicitly incorporating the impact of robot localization uncertainties into the calibration process, ensuring robust initialization. To further enhance the robustness of the calibration results against initialization errors, we propose a tightly-coupled Schmidt Kalman Filter (SKF)-based online refinement method, making the system suitable for practical applications. Simulations and real-world experiments validate the improved accuracy and robustness of our approach.
>
---
#### [new 017] Scene Graph-Guided Proactive Replanning for Failure-Resilient Embodied Agent
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人自主导航任务，解决环境变化导致的执行失败问题。通过场景图对比和轻量推理模块实现主动重规划，提升任务成功率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.11286v1](http://arxiv.org/pdf/2508.11286v1)**

> **作者:** Che Rin Yu; Daewon Chae; Dabin Seo; Sangwon Lee; Hyeongwoo Im; Jinkyu Kim
>
> **摘要:** When humans perform everyday tasks, we naturally adjust our actions based on the current state of the environment. For instance, if we intend to put something into a drawer but notice it is closed, we open it first. However, many autonomous robots lack this adaptive awareness. They often follow pre-planned actions that may overlook subtle yet critical changes in the scene, which can result in actions being executed under outdated assumptions and eventual failure. While replanning is critical for robust autonomy, most existing methods respond only after failures occur, when recovery may be inefficient or infeasible. While proactive replanning holds promise for preventing failures in advance, current solutions often rely on manually designed rules and extensive supervision. In this work, we present a proactive replanning framework that detects and corrects failures at subtask boundaries by comparing scene graphs constructed from current RGB-D observations against reference graphs extracted from successful demonstrations. When the current scene fails to align with reference trajectories, a lightweight reasoning module is activated to diagnose the mismatch and adjust the plan. Experiments in the AI2-THOR simulator demonstrate that our approach detects semantic and spatial mismatches before execution failures occur, significantly improving task success and robustness.
>
---
#### [new 018] An Exploratory Study on Crack Detection in Concrete through Human-Robot Collaboration
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于结构检测任务，旨在解决传统人工检测效率低、风险高的问题。通过人机协作与AI视觉技术提升混凝土裂缝检测的准确性与安全性。**

- **链接: [http://arxiv.org/pdf/2508.11404v1](http://arxiv.org/pdf/2508.11404v1)**

> **作者:** Junyeon Kim; Tianshu Ruan; Cesar Alan Contreras; Manolis Chiou
>
> **摘要:** Structural inspection in nuclear facilities is vital for maintaining operational safety and integrity. Traditional methods of manual inspection pose significant challenges, including safety risks, high cognitive demands, and potential inaccuracies due to human limitations. Recent advancements in Artificial Intelligence (AI) and robotic technologies have opened new possibilities for safer, more efficient, and accurate inspection methodologies. Specifically, Human-Robot Collaboration (HRC), leveraging robotic platforms equipped with advanced detection algorithms, promises significant improvements in inspection outcomes and reductions in human workload. This study explores the effectiveness of AI-assisted visual crack detection integrated into a mobile Jackal robot platform. The experiment results indicate that HRC enhances inspection accuracy and reduces operator workload, resulting in potential superior performance outcomes compared to traditional manual methods.
>
---
#### [new 019] Visual Perception Engine: Fast and Flexible Multi-Head Inference for Robotic Vision Tasks
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对机器人视觉任务中的计算冗余和内存占用问题，提出VPEngine框架，通过共享基础模型和并行任务头实现高效多任务处理。**

- **链接: [http://arxiv.org/pdf/2508.11584v1](http://arxiv.org/pdf/2508.11584v1)**

> **作者:** Jakub Łucki; Jonathan Becktor; Georgios Georgakis; Robert Royce; Shehryar Khattak
>
> **备注:** 6 pages, 6 figures, 2 tables
>
> **摘要:** Deploying multiple machine learning models on resource-constrained robotic platforms for different perception tasks often results in redundant computations, large memory footprints, and complex integration challenges. In response, this work presents Visual Perception Engine (VPEngine), a modular framework designed to enable efficient GPU usage for visual multitasking while maintaining extensibility and developer accessibility. Our framework architecture leverages a shared foundation model backbone that extracts image representations, which are efficiently shared, without any unnecessary GPU-CPU memory transfers, across multiple specialized task-specific model heads running in parallel. This design eliminates the computational redundancy inherent in feature extraction component when deploying traditional sequential models while enabling dynamic task prioritization based on application demands. We demonstrate our framework's capabilities through an example implementation using DINOv2 as the foundation model with multiple task (depth, object detection and semantic segmentation) heads, achieving up to 3x speedup compared to sequential execution. Building on CUDA Multi-Process Service (MPS), VPEngine offers efficient GPU utilization and maintains a constant memory footprint while allowing per-task inference frequencies to be adjusted dynamically during runtime. The framework is written in Python and is open source with ROS2 C++ (Humble) bindings for ease of use by the robotics community across diverse robotic platforms. Our example implementation demonstrates end-to-end real-time performance at $\geq$50 Hz on NVIDIA Jetson Orin AGX for TensorRT optimized models.
>
---
#### [new 020] Nominal Evaluation Of Automatic Multi-Sections Control Potential In Comparison To A Simpler One- Or Two-Sections Alternative With Predictive Spray Switching
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于农业自动化任务，旨在比较多段自动喷洒控制与简化方案的性能，解决喷洒重叠和效率问题。通过实验评估不同控制方法的效果。**

- **链接: [http://arxiv.org/pdf/2508.11573v1](http://arxiv.org/pdf/2508.11573v1)**

> **作者:** Mogens Plessen
>
> **备注:** 14 pages plus 7 pages appendix with additional figures, 18 main figures, 3 tables
>
> **摘要:** Automatic Section Control (ASC) is a long-standing trend for spraying in agriculture. It promises to minimise spray overlap areas. The core idea is to (i) switch off spray nozzles on areas that have already been sprayed, and (ii) to dynamically adjust nozzle flow rates along the boom bar that holds the spray nozzles when velocities of boom sections vary during turn maneuvers. ASC is not possible without sensors, in particular for accurate positioning data. Spraying and the movement of modern wide boom bars are highly dynamic processes. In addition, many uncertainty factors have an effect such as cross wind drift, boom height, nozzle clogging in open-field conditions, and so forth. In view of this complexity, the natural question arises if a simpler alternative exist. Therefore, an Automatic Multi-Sections Control method is compared to a proposed simpler one- or two-sections alternative that uses predictive spray switching. The comparison is provided under nominal conditions. Agricultural spraying is intrinsically linked to area coverage path planning and spray switching logic. Combinations of two area coverage path planning and switching logics as well as three sections-setups are compared. The three sections-setups differ by controlling 48 sections, 2 sections or controlling all nozzles uniformly with the same control signal as one single section. Methods are evaluated on 10 diverse real-world field examples, including non-convex field contours, freeform mainfield lanes and multiple obstacle areas. A preferred method is suggested that (i) minimises area coverage pathlength, (ii) offers intermediate overlap, (iii) is suitable for manual driving by following a pre-planned predictive spray switching logic for an area coverage path plan, and (iv) and in contrast to ASC can be implemented sensor-free and therefore at low cost.
>
---
#### [new 021] Utilizing Vision-Language Models as Action Models for Intent Recognition and Assistance
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于人机协作任务，旨在提升机器人意图识别与辅助能力。通过融合视觉语言模型和文本模型，增强导航与操作意图的准确性与适应性。**

- **链接: [http://arxiv.org/pdf/2508.11093v1](http://arxiv.org/pdf/2508.11093v1)**

> **作者:** Cesar Alan Contreras; Manolis Chiou; Alireza Rastegarpanah; Michal Szulik; Rustam Stolkin
>
> **备注:** Accepted at Human-Centered Robot Autonomy for Human-Robot Teams (HuRoboT) at IEEE RO-MAN 2025, Eindhoven, the Netherlands
>
> **摘要:** Human-robot collaboration requires robots to quickly infer user intent, provide transparent reasoning, and assist users in achieving their goals. Our recent work introduced GUIDER, our framework for inferring navigation and manipulation intents. We propose augmenting GUIDER with a vision-language model (VLM) and a text-only language model (LLM) to form a semantic prior that filters objects and locations based on the mission prompt. A vision pipeline (YOLO for object detection and the Segment Anything Model for instance segmentation) feeds candidate object crops into the VLM, which scores their relevance given an operator prompt; in addition, the list of detected object labels is ranked by a text-only LLM. These scores weight the existing navigation and manipulation layers of GUIDER, selecting context-relevant targets while suppressing unrelated objects. Once the combined belief exceeds a threshold, autonomy changes occur, enabling the robot to navigate to the desired area and retrieve the desired object, while adapting to any changes in the operator's intent. Future work will evaluate the system on Isaac Sim using a Franka Emika arm on a Ridgeback base, with a focus on real-time assistance.
>
---
#### [new 022] i2Nav-Robot: A Large-Scale Indoor-Outdoor Robot Dataset for Multi-Sensor Fusion Navigation and Mapping
- **分类: cs.RO**

- **简介: 该论文属于机器人导航与定位任务，旨在解决现有数据集不足的问题。通过构建i2Nav-Robot数据集，集成多传感器并提供高精度地面真值，支持多传感器融合导航与建图研究。**

- **链接: [http://arxiv.org/pdf/2508.11485v1](http://arxiv.org/pdf/2508.11485v1)**

> **作者:** Hailiang Tang; Tisheng Zhang; Liqiang Wang; Xin Ding; Man Yuan; Zhiyu Xiang; Jujin Chen; Yuhan Bian; Shuangyan Liu; Yuqing Wang; Guan Wang; Xiaoji Niu
>
> **备注:** 10 pages, 12 figures
>
> **摘要:** Accurate and reliable navigation is crucial for autonomous unmanned ground vehicle (UGV). However, current UGV datasets fall short in meeting the demands for advancing navigation and mapping techniques due to limitations in sensor configuration, time synchronization, ground truth, and scenario diversity. To address these challenges, we present i2Nav-Robot, a large-scale dataset designed for multi-sensor fusion navigation and mapping in indoor-outdoor environments. We integrate multi-modal sensors, including the newest front-view and 360-degree solid-state LiDARs, 4-dimensional (4D) radar, stereo cameras, odometer, global navigation satellite system (GNSS) receiver, and inertial measurement units (IMU) on an omnidirectional wheeled robot. Accurate timestamps are obtained through both online hardware synchronization and offline calibration for all sensors. The dataset comprises ten larger-scale sequences covering diverse UGV operating scenarios, such as outdoor streets, and indoor parking lots, with a total length of about 17060 meters. High-frequency ground truth, with centimeter-level accuracy for position, is derived from post-processing integrated navigation methods using a navigation-grade IMU. The proposed i2Nav-Robot dataset is evaluated by more than ten open-sourced multi-sensor fusion systems, and it has proven to have superior data quality.
>
---
#### [new 023] Embodied Edge Intelligence Meets Near Field Communication: Concept, Design, and Verification
- **分类: cs.RO; cs.NI**

- **简介: 该论文属于边缘智能与通信融合任务，解决大模型实时推理与通信效率问题，提出NEEI框架并设计优化方案提升性能。**

- **链接: [http://arxiv.org/pdf/2508.11232v1](http://arxiv.org/pdf/2508.11232v1)**

> **作者:** Guoliang Li; Xibin Jin; Yujie Wan; Chenxuan Liu; Tong Zhang; Shuai Wang; Chengzhong Xu
>
> **备注:** 9 pages, 6 figures, to appear in IEEE Network
>
> **摘要:** Realizing embodied artificial intelligence is challenging due to the huge computation demands of large models (LMs). To support LMs while ensuring real-time inference, embodied edge intelligence (EEI) is a promising paradigm, which leverages an LM edge to provide computing powers in close proximity to embodied robots. Due to embodied data exchange, EEI requires higher spectral efficiency, enhanced communication security, and reduced inter-user interference. To meet these requirements, near-field communication (NFC), which leverages extremely large antenna arrays as its hardware foundation, is an ideal solution. Therefore, this paper advocates the integration of EEI and NFC, resulting in a near-field EEI (NEEI) paradigm. However, NEEI also introduces new challenges that cannot be adequately addressed by isolated EEI or NFC designs, creating research opportunities for joint optimization of both functionalities. To this end, we propose radio-friendly embodied planning for EEI-assisted NFC scenarios and view-guided beam-focusing for NFC-assisted EEI scenarios. We also elaborate how to realize resource-efficient NEEI through opportunistic collaborative navigation. Experimental results are provided to confirm the superiority of the proposed techniques compared with various benchmarks.
>
---
#### [new 024] Swarm-in-Blocks: Simplifying Drone Swarm Programming with Block-Based Language
- **分类: cs.RO**

- **简介: 该论文属于无人机编队编程任务，旨在简化无人机群管理。通过块状编程语言降低使用门槛，解决初学者难以操作的问题。**

- **链接: [http://arxiv.org/pdf/2508.11498v1](http://arxiv.org/pdf/2508.11498v1)**

> **作者:** Agnes Bressan de Almeida; Joao Aires Correa Fernandes Marsicano
>
> **摘要:** Swarm in Blocks, originally developed for CopterHack 2022, is a high-level interface that simplifies drone swarm programming using a block-based language. Building on the Clover platform, this tool enables users to create functionalities like loops and conditional structures by assembling code blocks. In 2023, we introduced Swarm in Blocks 2.0, further refining the platform to address the complexities of swarm management in a user-friendly way. As drone swarm applications grow in areas like delivery, agriculture, and surveillance, the challenge of managing them, especially for beginners, has also increased. The Atena team developed this interface to make swarm handling accessible without requiring extensive knowledge of ROS or programming. The block-based approach not only simplifies swarm control but also expands educational opportunities in programming.
>
---
#### [new 025] Multi-Group Equivariant Augmentation for Reinforcement Learning in Robot Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作中的强化学习任务，旨在提升采样效率。针对任务对称性限制，提出非等距对称增强方法MEA，结合离线RL与体素表示，有效改善学习效率。**

- **链接: [http://arxiv.org/pdf/2508.11204v1](http://arxiv.org/pdf/2508.11204v1)**

> **作者:** Hongbin Lin; Juan Rojas; Kwok Wai Samuel Au
>
> **摘要:** Sampling efficiency is critical for deploying visuomotor learning in real-world robotic manipulation. While task symmetry has emerged as a promising inductive bias to improve efficiency, most prior work is limited to isometric symmetries -- applying the same group transformation to all task objects across all timesteps. In this work, we explore non-isometric symmetries, applying multiple independent group transformations across spatial and temporal dimensions to relax these constraints. We introduce a novel formulation of the partially observable Markov decision process (POMDP) that incorporates the non-isometric symmetry structures, and propose a simple yet effective data augmentation method, Multi-Group Equivariance Augmentation (MEA). We integrate MEA with offline reinforcement learning to enhance sampling efficiency, and introduce a voxel-based visual representation that preserves translational equivariance. Extensive simulation and real-robot experiments across two manipulation domains demonstrate the effectiveness of our approach.
>
---
#### [new 026] Investigating Sensors and Methods in Grasp State Classification in Agricultural Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于农业抓取状态分类任务，旨在提高果实采摘的准确性和可靠性。通过集成多种传感器和分类模型，实现对抓取状态的实时识别与纠正。**

- **链接: [http://arxiv.org/pdf/2508.11588v1](http://arxiv.org/pdf/2508.11588v1)**

> **作者:** Benjamin Walt; Jordan Westphal; Girish Krishnan
>
> **摘要:** Effective and efficient agricultural manipulation and harvesting depend on accurately understanding the current state of the grasp. The agricultural environment presents unique challenges due to its complexity, clutter, and occlusion. Additionally, fruit is physically attached to the plant, requiring precise separation during harvesting. Selecting appropriate sensors and modeling techniques is critical for obtaining reliable feedback and correctly identifying grasp states. This work investigates a set of key sensors, namely inertial measurement units (IMUs), infrared (IR) reflectance, tension, tactile sensors, and RGB cameras, integrated into a compliant gripper to classify grasp states. We evaluate the individual contribution of each sensor and compare the performance of two widely used classification models: Random Forest and Long Short-Term Memory (LSTM) networks. Our results demonstrate that a Random Forest classifier, trained in a controlled lab environment and tested on real cherry tomato plants, achieved 100% accuracy in identifying slip, grasp failure, and successful picks, marking a substantial improvement over baseline performance. Furthermore, we identify a minimal viable sensor combination, namely IMU and tension sensors that effectively classifies grasp states. This classifier enables the planning of corrective actions based on real-time feedback, thereby enhancing the efficiency and reliability of fruit harvesting operations.
>
---
#### [new 027] Robot Policy Evaluation for Sim-to-Real Transfer: A Benchmarking Perspective
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决模拟到现实的策略迁移问题。通过高保真仿真、任务复杂度提升和性能对齐评估，提升策略在真实环境中的表现。**

- **链接: [http://arxiv.org/pdf/2508.11117v1](http://arxiv.org/pdf/2508.11117v1)**

> **作者:** Xuning Yang; Clemens Eppner; Jonathan Tremblay; Dieter Fox; Stan Birchfield; Fabio Ramos
>
> **备注:** 2025 Robot: Science and Systems (RSS) Workshop on Robot Evaluation for the Real World
>
> **摘要:** Current vision-based robotics simulation benchmarks have significantly advanced robotic manipulation research. However, robotics is fundamentally a real-world problem, and evaluation for real-world applications has lagged behind in evaluating generalist policies. In this paper, we discuss challenges and desiderata in designing benchmarks for generalist robotic manipulation policies for the goal of sim-to-real policy transfer. We propose 1) utilizing high visual-fidelity simulation for improved sim-to-real transfer, 2) evaluating policies by systematically increasing task complexity and scenario perturbation to assess robustness, and 3) quantifying performance alignment between real-world performance and its simulation counterparts.
>
---
#### [new 028] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于长期视觉机器人操作任务，解决稀疏奖励下连续动作块学习的问题。提出AC3框架，通过稳定策略和价值学习提升数据效率与成功率。**

- **链接: [http://arxiv.org/pdf/2508.11143v1](http://arxiv.org/pdf/2508.11143v1)**

> **作者:** Jiarui Yang; Bin Zhu; Jingjing Chen; Yu-Gang Jiang
>
> **摘要:** Existing reinforcement learning (RL) methods struggle with long-horizon robotic manipulation tasks, particularly those involving sparse rewards. While action chunking is a promising paradigm for robotic manipulation, using RL to directly learn continuous action chunks in a stable and data-efficient manner remains a critical challenge. This paper introduces AC3 (Actor-Critic for Continuous Chunks), a novel RL framework that learns to generate high-dimensional, continuous action sequences. To make this learning process stable and data-efficient, AC3 incorporates targeted stabilization mechanisms for both the actor and the critic. First, to ensure reliable policy improvement, the actor is trained with an asymmetric update rule, learning exclusively from successful trajectories. Second, to enable effective value learning despite sparse rewards, the critic's update is stabilized using intra-chunk $n$-step returns and further enriched by a self-supervised module providing intrinsic rewards at anchor points aligned with each action chunk. We conducted extensive experiments on 25 tasks from the BiGym and RLBench benchmarks. Results show that by using only a few demonstrations and a simple model architecture, AC3 achieves superior success rates on most tasks, validating its effective design.
>
---
#### [new 029] Towards Fully Onboard State Estimation and Trajectory Tracking for UAVs with Suspended Payloads
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于无人机悬吊载荷的轨迹跟踪任务，解决如何在无额外硬件条件下精确估计和控制载荷位置的问题。通过融合GPS和IMU数据，设计了状态估计与控制框架，实现高效可靠控制。**

- **链接: [http://arxiv.org/pdf/2508.11547v1](http://arxiv.org/pdf/2508.11547v1)**

> **作者:** Martin Jiroušek; Tomáš Báča; Martin Saska
>
> **摘要:** This paper addresses the problem of tracking the position of a cable-suspended payload carried by an unmanned aerial vehicle, with a focus on real-world deployment and minimal hardware requirements. In contrast to many existing approaches that rely on motion-capture systems, additional onboard cameras, or instrumented payloads, we propose a framework that uses only standard onboard sensors--specifically, real-time kinematic global navigation satellite system measurements and data from the onboard inertial measurement unit--to estimate and control the payload's position. The system models the full coupled dynamics of the aerial vehicle and payload, and integrates a linear Kalman filter for state estimation, a model predictive contouring control planner, and an incremental model predictive controller. The control architecture is designed to remain effective despite sensing limitations and estimation uncertainty. Extensive simulations demonstrate that the proposed system achieves performance comparable to control based on ground-truth measurements, with only minor degradation (< 6%). The system also shows strong robustness to variations in payload parameters. Field experiments further validate the framework, confirming its practical applicability and reliable performance in outdoor environments using only off-the-shelf aerial vehicle hardware.
>
---
#### [new 030] Developing and Validating a High-Throughput Robotic System for the Accelerated Development of Porous Membranes
- **分类: cs.RO; cond-mat.mtrl-sci**

- **简介: 该论文属于材料开发任务，旨在解决传统膜制备过程耗时且重复的问题。通过构建自动化系统，实现高效、可控的膜制备与表征。**

- **链接: [http://arxiv.org/pdf/2508.10973v1](http://arxiv.org/pdf/2508.10973v1)**

> **作者:** Hongchen Wang; Sima Zeinali Danalou; Jiahao Zhu; Kenneth Sulimro; Chaewon Lim; Smita Basak; Aimee Tai; Usan Siriwardana; Jason Hattrick-Simpers; Jay Werber
>
> **摘要:** The development of porous polymeric membranes remains a labor-intensive process, often requiring extensive trial and error to identify optimal fabrication parameters. In this study, we present a fully automated platform for membrane fabrication and characterization via nonsolvent-induced phase separation (NIPS). The system integrates automated solution preparation, blade casting, controlled immersion, and compression testing, allowing precise control over fabrication parameters such as polymer concentration and ambient humidity. The modular design allows parallel processing and reproducible handling of samples, reducing experimental time and increasing consistency. Compression testing is introduced as a sensitive mechanical characterization method for estimating membrane stiffness and as a proxy to infer porosity and intra-sample uniformity through automated analysis of stress-strain curves. As a proof of concept to demonstrate the effectiveness of the system, NIPS was carried out with polysulfone, the green solvent PolarClean, and water as the polymer, solvent, and nonsolvent, respectively. Experiments conducted with the automated system reproduced expected effects of polymer concentration and ambient humidity on membrane properties, namely increased stiffness and uniformity with increasing polymer concentration and humidity variations in pore morphology and mechanical response. The developed automated platform supports high-throughput experimentation and is well-suited for integration into self-driving laboratory workflows, offering a scalable and reproducible foundation for data-driven optimization of porous polymeric membranes through NIPS.
>
---
#### [new 031] HQ-OV3D: A High Box Quality Open-World 3D Detection Framework based on Diffision Model
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于3D目标检测任务，解决开放世界场景下伪标签质量低的问题。提出HQ-OV3D框架，通过生成和优化高质量伪标签提升检测性能。**

- **链接: [http://arxiv.org/pdf/2508.10935v1](http://arxiv.org/pdf/2508.10935v1)**

> **作者:** Qi Liu; Yabei Li; Hongsong Wang; Lei He
>
> **摘要:** Traditional closed-set 3D detection frameworks fail to meet the demands of open-world applications like autonomous driving. Existing open-vocabulary 3D detection methods typically adopt a two-stage pipeline consisting of pseudo-label generation followed by semantic alignment. While vision-language models (VLMs) recently have dramatically improved the semantic accuracy of pseudo-labels, their geometric quality, particularly bounding box precision, remains commonly neglected.To address this issue, we propose a High Box Quality Open-Vocabulary 3D Detection (HQ-OV3D) framework, dedicated to generate and refine high-quality pseudo-labels for open-vocabulary classes. The framework comprises two key components: an Intra-Modality Cross-Validated (IMCV) Proposal Generator that utilizes cross-modality geometric consistency to generate high-quality initial 3D proposals, and an Annotated-Class Assisted (ACA) Denoiser that progressively refines 3D proposals by leveraging geometric priors from annotated categories through a DDIM-based denoising mechanism.Compared to the state-of-the-art method, training with pseudo-labels generated by our approach achieves a 7.37% improvement in mAP on novel classes, demonstrating the superior quality of the pseudo-labels produced by our framework. HQ-OV3D can serve not only as a strong standalone open-vocabulary 3D detector but also as a plug-in high-quality pseudo-label generator for existing open-vocabulary detection or annotation pipelines.
>
---
#### [new 032] ViPE: Video Pose Engine for 3D Geometric Perception
- **分类: cs.CV; cs.GR; cs.RO; eess.IV**

- **简介: 该论文属于3D几何感知任务，旨在解决从视频中准确估计相机参数和深度的问题。提出ViPE工具，高效处理多种视频场景并生成精确标注数据。**

- **链接: [http://arxiv.org/pdf/2508.10934v1](http://arxiv.org/pdf/2508.10934v1)**

> **作者:** Jiahui Huang; Qunjie Zhou; Hesam Rabeti; Aleksandr Korovko; Huan Ling; Xuanchi Ren; Tianchang Shen; Jun Gao; Dmitry Slepichev; Chen-Hsuan Lin; Jiawei Ren; Kevin Xie; Joydeep Biswas; Laura Leal-Taixe; Sanja Fidler
>
> **备注:** Paper website: https://research.nvidia.com/labs/toronto-ai/vipe/
>
> **摘要:** Accurate 3D geometric perception is an important prerequisite for a wide range of spatial AI systems. While state-of-the-art methods depend on large-scale training data, acquiring consistent and precise 3D annotations from in-the-wild videos remains a key challenge. In this work, we introduce ViPE, a handy and versatile video processing engine designed to bridge this gap. ViPE efficiently estimates camera intrinsics, camera motion, and dense, near-metric depth maps from unconstrained raw videos. It is robust to diverse scenarios, including dynamic selfie videos, cinematic shots, or dashcams, and supports various camera models such as pinhole, wide-angle, and 360{\deg} panoramas. We have benchmarked ViPE on multiple benchmarks. Notably, it outperforms existing uncalibrated pose estimation baselines by 18%/50% on TUM/KITTI sequences, and runs at 3-5FPS on a single GPU for standard input resolutions. We use ViPE to annotate a large-scale collection of videos. This collection includes around 100K real-world internet videos, 1M high-quality AI-generated videos, and 2K panoramic videos, totaling approximately 96M frames -- all annotated with accurate camera poses and dense depth maps. We open-source ViPE and the annotated dataset with the hope of accelerating the development of spatial AI systems.
>
---
#### [new 033] Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D语义占用预测任务，解决协作场景下通信成本高和依赖深度监督的问题。提出基于稀疏3D语义高斯点云的协作方法，提升性能并减少通信量。**

- **链接: [http://arxiv.org/pdf/2508.10936v1](http://arxiv.org/pdf/2508.10936v1)**

> **作者:** Cheng Chen; Hao Huang; Saurabh Bagchi
>
> **摘要:** Collaborative perception enables connected vehicles to share information, overcoming occlusions and extending the limited sensing range inherent in single-agent (non-collaborative) systems. Existing vision-only methods for 3D semantic occupancy prediction commonly rely on dense 3D voxels, which incur high communication costs, or 2D planar features, which require accurate depth estimation or additional supervision, limiting their applicability to collaborative scenarios. To address these challenges, we propose the first approach leveraging sparse 3D semantic Gaussian splatting for collaborative 3D semantic occupancy prediction. By sharing and fusing intermediate Gaussian primitives, our method provides three benefits: a neighborhood-based cross-agent fusion that removes duplicates and suppresses noisy or inconsistent Gaussians; a joint encoding of geometry and semantics in each primitive, which reduces reliance on depth supervision and allows simple rigid alignment; and sparse, object-centric messages that preserve structural information while reducing communication volume. Extensive experiments demonstrate that our approach outperforms single-agent perception and baseline collaborative methods by +8.42 and +3.28 points in mIoU, and +5.11 and +22.41 points in IoU, respectively. When further reducing the number of transmitted Gaussians, our method still achieves a +1.9 improvement in mIoU, using only 34.6% communication volume, highlighting robust performance under limited communication budgets.
>
---
#### [new 034] Optimizing ROS 2 Communication for Wireless Robotic Systems
- **分类: cs.NI; cs.RO**

- **简介: 该论文属于无线机器人通信优化任务，解决ROS 2在无线环境下传输大负载时的性能问题，通过分析并改进DDS通信参数提升可靠性与效率。**

- **链接: [http://arxiv.org/pdf/2508.11366v1](http://arxiv.org/pdf/2508.11366v1)**

> **作者:** Sanghoon Lee; Taehun Kim; Jiyeong Chae; Kyung-Joon Park
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Wireless transmission of large payloads, such as high-resolution images and LiDAR point clouds, is a major bottleneck in ROS 2, the leading open-source robotics middleware. The default Data Distribution Service (DDS) communication stack in ROS 2 exhibits significant performance degradation over lossy wireless links. Despite the widespread use of ROS 2, the underlying causes of these wireless communication challenges remain unexplored. In this paper, we present the first in-depth network-layer analysis of ROS 2's DDS stack under wireless conditions with large payloads. We identify the following three key issues: excessive IP fragmentation, inefficient retransmission timing, and congestive buffer bursts. To address these issues, we propose a lightweight and fully compatible DDS optimization framework that tunes communication parameters based on link and payload characteristics. Our solution can be seamlessly applied through the standard ROS 2 application interface via simple XML-based QoS configuration, requiring no protocol modifications, no additional components, and virtually no integration efforts. Extensive experiments across various wireless scenarios demonstrate that our framework successfully delivers large payloads in conditions where existing DDS modes fail, while maintaining low end-to-end latency.
>
---
#### [new 035] GhostObjects: Instructing Robots by Manipulating Spatially Aligned Virtual Twins in Augmented Reality
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决如何通过增强现实指导机器人操作的问题。工作是提出GhostObjects，让用户通过操控虚拟物体来精确指定物理目标和空间参数。**

- **链接: [http://arxiv.org/pdf/2508.11022v1](http://arxiv.org/pdf/2508.11022v1)**

> **作者:** Lauren W. Wang; Parastoo Abtahi
>
> **摘要:** Robots are increasingly capable of autonomous operations, yet human interaction remains essential for issuing personalized instructions. Instead of directly controlling robots through Programming by Demonstration (PbD) or teleoperation, we propose giving instructions by interacting with GhostObjects-world-aligned, life-size virtual twins of physical objects-in augmented reality (AR). By direct manipulation of GhostObjects, users can precisely specify physical goals and spatial parameters, with features including real-world lasso selection of multiple objects and snapping back to default positions, enabling tasks beyond simple pick-and-place.
>
---
#### [new 036] ReachVox: Clutter-free Reachability Visualization for Robot Motion Planning in Virtual Reality
- **分类: cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.11426v1](http://arxiv.org/pdf/2508.11426v1)**

> **作者:** Steffen Hauck; Diar Abdlkarim; John Dudley; Per Ola Kristensson; Eyal Ofek; Jens Grubert
>
> **备注:** To appear in Proceedings of IEEE ISMAR 2025
>
> **摘要:** Human-Robot-Collaboration can enhance workflows by leveraging the mutual strengths of human operators and robots. Planning and understanding robot movements remain major challenges in this domain. This problem is prevalent in dynamic environments that might need constant robot motion path adaptation. In this paper, we investigate whether a minimalistic encoding of the reachability of a point near an object of interest, which we call ReachVox, can aid the collaboration between a remote operator and a robotic arm in VR. Through a user study (n=20), we indicate the strength of the visualization relative to a point-based reachability check-up.
>
---
## 更新

#### [replaced 001] An Open-Source User-Friendly Interface for Simulating Magnetic Soft Robots using Simulation Open Framework Architecture (SOFA)
- **分类: cs.RO; cond-mat.soft**

- **链接: [http://arxiv.org/pdf/2508.10686v2](http://arxiv.org/pdf/2508.10686v2)**

> **作者:** Carla Wehner; Finn Schubert; Heiko Hellkamp; Julius Hahnewald; Kilian Schaefer; Muhammad Bilal Khan; Oliver Gutfleisch
>
> **摘要:** Soft robots, particularly magnetic soft robots, require specialized simulation tools to accurately model their deformation under external magnetic fields. However, existing platforms often lack dedicated support for magnetic materials, making them difficult to use for researchers at different expertise levels. This work introduces an open-source, user-friendly simulation interface using the Simulation Open Framework Architecture (SOFA), specifically designed to model magnetic soft robots. The tool enables users to define material properties, apply magnetic fields, and observe resulting deformations in real time. By integrating intuitive controls and stress analysis capabilities, it aims to bridge the gap between theoretical modeling and practical design. Four benchmark models -- a beam, three- and four-finger grippers, and a butterfly -- demonstrate its functionality. The software's ease of use makes it accessible to both beginners and advanced researchers. Future improvements will refine accuracy through experimental validation and comparison with industry-standard finite element solvers, ensuring realistic and predictive simulations of magnetic soft robots.
>
---
#### [replaced 002] KDPE: A Kernel Density Estimation Strategy for Diffusion Policy Trajectory Selection
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.10511v2](http://arxiv.org/pdf/2508.10511v2)**

> **作者:** Andrea Rosasco; Federico Ceola; Giulia Pasquale; Lorenzo Natale
>
> **备注:** 9th Conference on Robot Learning (CoRL 2025), Seoul, Korea
>
> **摘要:** Learning robot policies that capture multimodality in the training data has been a long-standing open challenge for behavior cloning. Recent approaches tackle the problem by modeling the conditional action distribution with generative models. One of these approaches is Diffusion Policy, which relies on a diffusion model to denoise random points into robot action trajectories. While achieving state-of-the-art performance, it has two main drawbacks that may lead the robot out of the data distribution during policy execution. First, the stochasticity of the denoising process can highly impact on the quality of generated trajectory of actions. Second, being a supervised learning approach, it can learn data outliers from the dataset used for training. Recent work focuses on mitigating these limitations by combining Diffusion Policy either with large-scale training or with classical behavior cloning algorithms. Instead, we propose KDPE, a Kernel Density Estimation-based strategy that filters out potentially harmful trajectories output of Diffusion Policy while keeping a low test-time computational overhead. For Kernel Density Estimation, we propose a manifold-aware kernel to model a probability density function for actions composed of end-effector Cartesian position, orientation, and gripper state. KDPE overall achieves better performance than Diffusion Policy on simulated single-arm tasks and real robot experiments. Additional material and code are available on our project page at https://hsp-iit.github.io/KDPE/.
>
---
#### [replaced 003] Tool-Planner: Task Planning with Clusters across Multiple Tools
- **分类: cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2406.03807v4](http://arxiv.org/pdf/2406.03807v4)**

> **作者:** Yanming Liu; Xinyue Peng; Jiannan Cao; Yuwei Zhang; Xuhong Zhang; Sheng Cheng; Xun Wang; Jianwei Yin; Tianyu Du
>
> **备注:** ICLR 2025 Camera Ready version
>
> **摘要:** Large language models (LLMs) have demonstrated exceptional reasoning capabilities, enabling them to solve various complex problems. Recently, this ability has been applied to the paradigm of tool learning. Tool learning involves providing examples of tool usage and their corresponding functions, allowing LLMs to formulate plans and demonstrate the process of invoking and executing each tool. LLMs can address tasks that they cannot complete independently, thereby enhancing their potential across different tasks. However, this approach faces two key challenges. First, redundant error correction leads to unstable planning and long execution time. Additionally, designing a correct plan among multiple tools is also a challenge in tool learning. To address these issues, we propose Tool-Planner, a task-processing framework based on toolkits. Tool-Planner groups tools based on the API functions with the same function into a toolkit and allows LLMs to implement planning across the various toolkits. When a tool error occurs, the language model can reselect and adjust tools based on the toolkit. Experiments show that our approach demonstrates a high pass and win rate across different datasets and optimizes the planning scheme for tool learning in models such as GPT-4 and Claude 3, showcasing the potential of our method. Our code is public at https://github.com/OceannTwT/Tool-Planner
>
---
#### [replaced 004] Towards Physically Realizable Adversarial Attacks in Embodied Vision Navigation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.10071v5](http://arxiv.org/pdf/2409.10071v5)**

> **作者:** Meng Chen; Jiawei Tu; Chao Qi; Yonghao Dang; Feng Zhou; Wei Wei; Jianqin Yin
>
> **备注:** 7 pages, 7 figures, Accept by IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** The significant advancements in embodied vision navigation have raised concerns about its susceptibility to adversarial attacks exploiting deep neural networks. Investigating the adversarial robustness of embodied vision navigation is crucial, especially given the threat of 3D physical attacks that could pose risks to human safety. However, existing attack methods for embodied vision navigation often lack physical feasibility due to challenges in transferring digital perturbations into the physical world. Moreover, current physical attacks for object detection struggle to achieve both multi-view effectiveness and visual naturalness in navigation scenarios. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches to objects, where both opacity and textures are learnable. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which optimizes the patch's texture based on feedback from the vision-based perception model used in navigation. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, in which opacity is fine-tuned after texture optimization. Experimental results demonstrate that our adversarial patches decrease the navigation success rate by an average of 22.39%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: https://github.com/chen37058/Physical-Attacks-in-Embodied-Nav
>
---
#### [replaced 005] A Computationally Efficient Maximum A Posteriori Sequence Estimation via Stein Variational Inference
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2312.08684v3](http://arxiv.org/pdf/2312.08684v3)**

> **作者:** Min-Won Seo; Solmaz S. Kia
>
> **备注:** 14 pages
>
> **摘要:** State estimation in robotic systems presents significant challenges, particularly due to the prevalence of multimodal posterior distributions in real-world scenarios. One effective strategy for handling such complexity is to compute maximum a posteriori (MAP) sequences over a discretized or sampled state space, which enables a concise representation of the most likely state trajectory. However, this approach often incurs substantial computational costs, especially in high-dimensional settings. In this article, we propose a novel MAP sequence estimation method, \textsf{Stein-MAP-Seq}, which effectively addresses multimodality while substantially reducing computational and memory overhead. Our key contribution is a sequential variational inference framework that captures temporal dependencies in dynamical system models and integrates Stein variational gradient descent (SVGD) into a Viterbi-style dynamic programming algorithm, enabling computationally efficient MAP sequence estimation. \textsf{Stein-MAP-Seq} achieves a computational complexity of $\mathcal{O}(M^2)$, where $M$ is the number of particles, in contrast to the $\mathcal{O}(N^2)$ complexity of conventional MAP sequence estimators, with $N \gg M$. Furthermore, the method inherits SVGD's parallelism, enabling efficient computation for real-time deployment on GPU-equipped autonomous systems. We validate the proposed method in various multimodal scenarios, including those arising from nonlinear dynamics with ambiguous observations, unknown data associations, and temporary unobservability, demonstrating substantial improvements in estimation accuracy and robustness to multimodality over existing approaches.
>
---
#### [replaced 006] MARS-FTCP: Robust Fault-Tolerant Control and Agile Trajectory Planning for Modular Aerial Robot Systems
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.09351v2](http://arxiv.org/pdf/2503.09351v2)**

> **作者:** Rui Huang; Zhenyu Zhang; Siyu Tang; Zhiqian Cai; Lin Zhao
>
> **摘要:** Modular Aerial Robot Systems (MARS) consist of multiple drone units that can self-reconfigure to adapt to various mission requirements and fault conditions. However, existing fault-tolerant control methods exhibit significant oscillations during docking and separation, impacting system stability. To address this issue, we propose a novel fault-tolerant control reallocation method that adapts to an arbitrary number of modular robots and their assembly formations. The algorithm redistributes the expected collective force and torque required for MARS to individual units according to their moment arm relative to the center of MARS mass. Furthermore, we propose an agile trajectory planning method for MARS of arbitrary configurations, which is collision-avoiding and dynamically feasible. Our work represents the first comprehensive approach to enable fault-tolerant and collision avoidance flight for MARS. We validate our method through extensive simulations, demonstrating improved fault tolerance, enhanced trajectory tracking accuracy, and greater robustness in cluttered environments. The videos and source code of this work are available at https://github.com/RuiHuangNUS/MARS-FTCP/
>
---
#### [replaced 007] Why Report Failed Interactions With Robots?! Towards Vignette-based Interaction Quality
- **分类: cs.RO; cs.HC**

- **链接: [http://arxiv.org/pdf/2508.10603v2](http://arxiv.org/pdf/2508.10603v2)**

> **作者:** Agnes Axelsson; Merle Reimann; Ronald Cumbal; Hannah Pelikan; Divesh Lala
>
> **备注:** Accepted at the workshop on Real-World HRI in Public and Private Spaces: Successes, Failures, and Lessons Learned (PubRob-Fails), held at the IEEE RO-MAN Conference, 2025. 6 pages
>
> **摘要:** Although the quality of human-robot interactions has improved with the advent of LLMs, there are still various factors that cause systems to be sub-optimal when compared to human-human interactions. The nature and criticality of failures are often dependent on the context of the interaction and so cannot be generalized across the wide range of scenarios and experiments which have been implemented in HRI research. In this work we propose the use of a technique overlooked in the field of HRI, ethnographic vignettes, to clearly highlight these failures, particularly those that are rarely documented. We describe the methodology behind the process of writing vignettes and create our own based on our personal experiences with failures in HRI systems. We emphasize the strength of vignettes as the ability to communicate failures from a multi-disciplinary perspective, promote transparency about the capabilities of robots, and document unexpected behaviours which would otherwise be omitted from research reports. We encourage the use of vignettes to augment existing interaction evaluation methods.
>
---
#### [replaced 008] Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.08896v3](http://arxiv.org/pdf/2508.08896v3)**

> **作者:** Haoyu Zhao; Linghao Zhuang; Xingyue Zhao; Cheng Zeng; Haoran Xu; Yuming Jiang; Jun Cen; Kexiang Wang; Jiayan Guo; Siteng Huang; Xin Li; Deli Zhao; Hua Zou
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** A dexterous hand capable of generalizable grasping objects is fundamental for the development of general-purpose embodied AI. However, previous methods focus narrowly on low-level grasp stability metrics, neglecting affordance-aware positioning and human-like poses which are crucial for downstream manipulation. To address these limitations, we propose AffordDex, a novel framework with two-stage training that learns a universal grasping policy with an inherent understanding of both motion priors and object affordances. In the first stage, a trajectory imitator is pre-trained on a large corpus of human hand motions to instill a strong prior for natural movement. In the second stage, a residual module is trained to adapt these general human-like motions to specific object instances. This refinement is critically guided by two components: our Negative Affordance-aware Segmentation (NAA) module, which identifies functionally inappropriate contact regions, and a privileged teacher-student distillation process that ensures the final vision-based policy is highly successful. Extensive experiments demonstrate that AffordDex not only achieves universal dexterous grasping but also remains remarkably human-like in posture and functionally appropriate in contact location. As a result, AffordDex significantly outperforms state-of-the-art baselines across seen objects, unseen instances, and even entirely novel categories.
>
---
#### [replaced 009] Safe Multi-Robotic Arm Interaction via 3D Convex Shapes
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.11791v2](http://arxiv.org/pdf/2503.11791v2)**

> **作者:** Ali Umut Kaypak; Shiqing Wei; Prashanth Krishnamurthy; Farshad Khorrami
>
> **摘要:** Inter-robot collisions pose a significant safety risk when multiple robotic arms operate in close proximity. We present an online collision avoidance methodology leveraging 3D convex shape-based High-Order Control Barrier Functions (HOCBFs) to address this issue. While prior works focused on using Control Barrier Functions (CBFs) for human-robotic arm and single-arm collision avoidance, we explore the problem of collision avoidance between multiple robotic arms operating in a shared space. In our methodology, we utilize the proposed HOCBFs as centralized and decentralized safety filters. These safety filters are compatible with many nominal controllers and ensure safety without significantly restricting the robots' workspace. A key challenge in implementing these filters is the computational overhead caused by the large number of safety constraints and the computation of a Hessian matrix per constraint. We address this challenge by employing numerical differentiation methods to approximate computationally intensive terms. The effectiveness of our method is demonstrated through extensive simulation studies and real-world experiments with Franka Research 3 robotic arms. The project video is available at this link.
>
---
#### [replaced 010] EmbodiedAgent: A Scalable Hierarchical Approach to Overcome Practical Challenge in Multi-Robot Control
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.10030v2](http://arxiv.org/pdf/2504.10030v2)**

> **作者:** Hanwen Wan; Yifei Chen; Yixuan Deng; Zeyu Wei; Dongrui Li; Zexin Lin; Donghao Wu; Jiu Cheng; Xiaoqiang Ji
>
> **摘要:** This paper introduces EmbodiedAgent, a hierarchical framework for heterogeneous multi-robot control. EmbodiedAgent addresses critical limitations of hallucination in impractical tasks. Our approach integrates a next-action prediction paradigm with a structured memory system to decompose tasks into executable robot skills while dynamically validating actions against environmental constraints. We present MultiPlan+, a dataset of more than 18,000 annotated planning instances spanning 100 scenarios, including a subset of impractical cases to mitigate hallucination. To evaluate performance, we propose the Robot Planning Assessment Schema (RPAS), combining automated metrics with LLM-aided expert grading. Experiments demonstrate EmbodiedAgent's superiority over state-of-the-art models, achieving 71.85% RPAS score. Real-world validation in an office service task highlights its ability to coordinate heterogeneous robots for long-horizon objectives.
>
---
#### [replaced 011] Large-Scale Multi-Robot Assembly Planning for Autonomous Manufacturing
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2311.00192v2](http://arxiv.org/pdf/2311.00192v2)**

> **作者:** Kyle Brown; Dylan M. Asmar; Mac Schwager; Mykel J. Kochenderfer
>
> **备注:** Repository: https://github.com/sisl/ConstructionBots.jl. Under review
>
> **摘要:** Mobile autonomous robots have the potential to revolutionize manufacturing processes. However, employing large robot fleets in manufacturing requires addressing challenges including collision-free movement in a shared workspace, effective multi-robot collaboration to manipulate and transport large payloads, complex task allocation due to coupled manufacturing processes, and spatial planning for parallel assembly and transportation of nested subassemblies. We propose a full algorithmic stack for large-scale multi-robot assembly planning that addresses these challenges and can synthesize construction plans for complex assemblies with thousands of parts in a matter of minutes. Our approach takes in a CAD-like product specification and automatically plans a full-stack assembly procedure for a group of robots to manufacture the product. We propose an algorithmic stack that comprises: (i) an iterative radial layout optimization procedure to define a global staging layout for the manufacturing facility, (ii) a graph-repair mixed-integer program formulation and a modified greedy task allocation algorithm to optimally allocate robots and robot sub-teams to assembly and transport tasks, (iii) a geometric heuristic and a hill-climbing algorithm to plan collaborative carrying configurations of robot sub-teams, and (iv) a distributed control policy that enables robots to execute the assembly motion plan collision-free. We also present an open-source multi-robot manufacturing simulator implemented in Julia as a resource to the research community, to test our algorithms and to facilitate multi-robot manufacturing research more broadly. Our empirical results demonstrate the scalability and effectiveness of our approach by generating plans to manufacture a LEGO model of a Saturn V launch vehicle with 1845 parts, 306 subassemblies, and 250 robots in under three minutes on a standard laptop computer.
>
---
#### [replaced 012] UniTracker: Learning Universal Whole-Body Motion Tracker for Humanoid Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.07356v2](http://arxiv.org/pdf/2507.07356v2)**

> **作者:** Kangning Yin; Weishuai Zeng; Ke Fan; Zirui Wang; Qiang Zhang; Zheng Tian; Jingbo Wang; Jiangmiao Pang; Weinan Zhang
>
> **备注:** three-stage universal motion tracker for humanoid robots
>
> **摘要:** Achieving expressive and generalizable whole-body motion control is essential for deploying humanoid robots in real-world environments. In this work, we propose UniTracker, a three-stage training framework that enables robust and scalable motion tracking across a wide range of human behaviors. In the first stage, we train a teacher policy with privileged observations to generate high-quality actions. In the second stage, we introduce a Conditional Variational Autoencoder (CVAE) to model a universal student policy that can be deployed directly on real hardware. The CVAE structure allows the policy to learn a global latent representation of motion, enhancing generalization to unseen behaviors and addressing the limitations of standard MLP-based policies under partial observations. Unlike pure MLPs that suffer from drift in global attributes like orientation, our CVAE-student policy incorporates global intent during training by aligning a partial-observation prior to the full-observation encoder. In the third stage, we introduce a fast adaptation module that fine-tunes the universal policy on harder motion sequences that are difficult to track directly. This adaptation can be performed both for single sequences and in batch mode, further showcasing the flexibility and scalability of our approach. We evaluate UniTracker in both simulation and real-world settings using a Unitree G1 humanoid, demonstrating strong performance in motion diversity, tracking accuracy, and deployment robustness.
>
---
#### [replaced 013] Propeller Motion of a Devil-Stick using Normal Forcing
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2501.17789v2](http://arxiv.org/pdf/2501.17789v2)**

> **作者:** Aakash Khandelwal; Ranjan Mukherjee
>
> **备注:** 6 pages, 5 figures. This work has been accepted for publication in the proceedings of the 2025 IEEE Conference on Control Technology and Applications (CCTA)
>
> **摘要:** The problem of realizing rotary propeller motion of a devil-stick in the vertical plane using forces purely normal to the stick is considered. This problem represents a nonprehensile manipulation task of an underactuated system. In contrast with previous approaches, the devil-stick is manipulated by controlling the normal force and its point of application. Virtual holonomic constraints are used to design the trajectory of the center-of-mass of the devil-stick in terms of its orientation angle, and conditions for stable propeller motion are derived. Intermittent large-amplitude forces are used to asymptotically stabilize a desired propeller motion. Simulations demonstrate the efficacy of the approach in realizing stable propeller motion without loss of contact between the actuator and devil-stick.
>
---
#### [replaced 014] From Autonomy to Agency: Agentic Vehicles for Human-Centered Mobility Systems
- **分类: cs.CY; cs.CE; cs.CL; cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.04996v3](http://arxiv.org/pdf/2507.04996v3)**

> **作者:** Jiangbo Yu
>
> **摘要:** Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity to operate according to internal rules without external control. Accordingly, autonomous vehicles (AuVs) are viewed as vehicular systems capable of perceiving their environment and executing pre-programmed tasks independently of external input. However, both research and real-world deployments increasingly showcase vehicles that demonstrate behaviors beyond this definition (including the SAE levels 0 to 5); Examples of this outpace include the interaction with humans with natural language, goal adaptation, contextual reasoning, external tool use, and unseen ethical dilemma handling, largely empowered by multi-modal large language models (LLMs). These developments reveal a conceptual gap between technical autonomy and the broader cognitive and social capabilities needed for future human-centered mobility systems. To address this gap, this paper introduces the concept of agentic vehicles (AgVs), referring to vehicles that integrate agentic AI systems to reason, adapt, and interact within complex environments. This paper proposes the term AgVs and their distinguishing characteristics from conventional AuVs. It synthesizes relevant advances in integrating LLMs and AuVs and highlights how AgVs might transform future mobility systems and ensure the systems are human-centered. The paper concludes by identifying key challenges in the development and governance of AgVs, and how they can play a significant role in future agentic transportation systems.
>
---
#### [replaced 015] A Segmented Robot Grasping Perception Neural Network for Edge AI
- **分类: cs.RO; cs.AI; I.2; I.2.9; I.2.10**

- **链接: [http://arxiv.org/pdf/2507.13970v3](http://arxiv.org/pdf/2507.13970v3)**

> **作者:** Casper Bröcheler; Thomas Vroom; Derrick Timmermans; Alan van den Akker; Guangzhi Tang; Charalampos S. Kouzinopoulos; Rico Möckel
>
> **备注:** Accepted by SMC 2025
>
> **摘要:** Robotic grasping, the ability of robots to reliably secure and manipulate objects of varying shapes, sizes and orientations, is a complex task that requires precise perception and control. Deep neural networks have shown remarkable success in grasp synthesis by learning rich and abstract representations of objects. When deployed at the edge, these models can enable low-latency, low-power inference, making real-time grasping feasible in resource-constrained environments. This work implements Heatmap-Guided Grasp Detection, an end-to-end framework for the detection of 6-Dof grasp poses, on the GAP9 RISC-V System-on-Chip. The model is optimised using hardware-aware techniques, including input dimensionality reduction, model partitioning, and quantisation. Experimental evaluation on the GraspNet-1Billion benchmark validates the feasibility of fully on-chip inference, highlighting the potential of low-power MCUs for real-time, autonomous manipulation.
>
---
#### [replaced 016] IRL-VLA: Training an Vision-Language-Action Policy via Reward World Model
- **分类: cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.06571v3](http://arxiv.org/pdf/2508.06571v3)**

> **作者:** Anqing Jiang; Yu Gao; Yiru Wang; Zhigang Sun; Shuo Wang; Yuwen Heng; Hao Sun; Shichen Tang; Lijuan Zhu; Jinhao Chai; Jijun Wang; Zichong Gu; Hao Jiang; Li Sun
>
> **备注:** 9 pagres, 2 figures
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated potential in autonomous driving. However, two critical challenges hinder their development: (1) Existing VLA architectures are typically based on imitation learning in open-loop setup which tends to capture the recorded behaviors in the dataset, leading to suboptimal and constrained performance, (2) Close-loop training relies heavily on high-fidelity sensor simulation, where domain gaps and computational inefficiencies pose significant barriers. In this paper, we introduce IRL-VLA, a novel close-loop Reinforcement Learning via \textbf{I}nverse \textbf{R}einforcement \textbf{L}earning reward world model with a self-built VLA approach. Our framework proceeds in a three-stage paradigm: In the first stage, we propose a VLA architecture and pretrain the VLA policy via imitation learning. In the second stage, we construct a lightweight reward world model via inverse reinforcement learning to enable efficient close-loop reward computation. To further enhance planning performance, finally, we design specialized reward world model guidence reinforcement learning via PPO(Proximal Policy Optimization) to effectively balance the safety incidents, comfortable driving, and traffic efficiency. Our approach achieves state-of-the-art performance in NAVSIM v2 end-to-end driving benchmark, 1st runner up in CVPR2025 Autonomous Grand Challenge. We hope that our framework will accelerate VLA research in close-loop autonomous driving.
>
---
#### [replaced 017] SORT3D: Spatial Object-centric Reasoning Toolbox for Zero-Shot 3D Grounding Using Large Language Models
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.18684v2](http://arxiv.org/pdf/2504.18684v2)**

> **作者:** Nader Zantout; Haochen Zhang; Pujith Kachana; Jinkai Qiu; Guofei Chen; Ji Zhang; Wenshan Wang
>
> **备注:** 8 pages, 6 figures, published in IROS 2025
>
> **摘要:** Interpreting object-referential language and grounding objects in 3D with spatial relations and attributes is essential for robots operating alongside humans. However, this task is often challenging due to the diversity of scenes, large number of fine-grained objects, and complex free-form nature of language references. Furthermore, in the 3D domain, obtaining large amounts of natural language training data is difficult. Thus, it is important for methods to learn from little data and zero-shot generalize to new environments. To address these challenges, we propose SORT3D, an approach that utilizes rich object attributes from 2D data and merges a heuristics-based spatial reasoning toolbox with the ability of large language models (LLMs) to perform sequential reasoning. Importantly, our method does not require text-to-3D data for training and can be applied zero-shot to unseen environments. We show that SORT3D achieves state-of-the-art zero-shot performance on complex view-dependent grounding tasks on two benchmarks. We also implement the pipeline to run real-time on two autonomous vehicles and demonstrate that our approach can be used for object-goal navigation on previously unseen real-world environments. All source code for the system pipeline is publicly released at https://github.com/nzantout/SORT3D.
>
---
#### [replaced 018] Dancing with REEM-C: A robot-to-human physical-social communication study
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2408.05301v2](http://arxiv.org/pdf/2408.05301v2)**

> **作者:** Marie Charbonneau; Francisco Javier Andrade Chavez; Katja Mombaur
>
> **备注:** 21 pages, 16 figures
>
> **摘要:** Humans often work closely together and relay a wealth of information through physical interaction. Robots, on the other hand, are not yet able to work similarly closely with humans and to effectively convey information when engaging in physical-social human-robot interaction (psHRI). This currently limits the potential of human-robot collaboration to solve real-world problems. This paper investigates how to establish clear and intuitive robot-to-human communication, while considering human comfort during psHRI. We approach this question from the perspective of a leader-follower dancing scenario, in which a full-body humanoid robot leads a human by signaling the next steps through a choice of communication modalities including haptic, visual, and audio signals. This is achieved through the development of a split whole-body control framework combining admittance and impedance control on the upper body, with position control on the lower body for balancing and stepping. Robot-led psHRI participant experiments allowed us to verify controller performance, as well as to build an understanding of what types of communication work better from the perspective of human partners, particularly in terms of perceived effectiveness and comfort.
>
---
#### [replaced 019] Optimal Planning and Machine Learning for Responsive Tracking and Enhanced Forecasting of Wildfires using a Spacecraft Constellation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.06687v2](http://arxiv.org/pdf/2508.06687v2)**

> **作者:** Sreeja Roy-Singh; Vinay Ravindra; Richard Levinson; Mahta Moghaddam; Jan Mandel; Adam Kochanski; Angel Farguell Caus; Kurtis Nelson; Samira Alkaee Taleghan; Archana Kannan; Amer Melebari
>
> **摘要:** We propose a novel concept of operations using optimal planning methods and machine learning (ML) to collect spaceborne data that is unprecedented for monitoring wildfires, process it to create new or enhanced products in the context of wildfire danger or spread monitoring, and assimilate them to improve existing, wildfire decision support tools delivered to firefighters within latency appropriate for time-critical applications. The concept is studied with respect to NASA's CYGNSS Mission, a constellation of passive microwave receivers that measure specular GNSS-R reflections despite clouds and smoke. Our planner uses a Mixed Integer Program formulation to schedule joint observation data collection and downlink for all satellites. Optimal solutions are found quickly that collect 98-100% of available observation opportunities. ML-based fire predictions that drive the planner objective are greater than 40% more correlated with ground truth than existing state-of-art. The presented case study on the TX Smokehouse Creek fire in 2024 and LA fires in 2025 represents the first high-resolution data collected by CYGNSS of active fires. Creation of Burnt Area Maps (BAM) using ML on data from active fires and BAM assimilation into NASA's Weather Research and Forecasting Model using neural nets to broadcast fire spread are novel outcomes. BAM and CYGNSS obtained soil moisture are integrated for the first time into USGS fire danger maps. Inclusion of CYGNSS data in ML-based burn predictions boosts accuracy by 13%, and inclusion of high-resolution data boosts ML recall by another 15%. The proposed workflow has an expected latency of 6-30h, improving on the current delivery time of multiple days. All components in the proposed concept are shown to be computationally scalable and globally generalizable, with sustainability considerations such as edge efficiency and low latency on small devices.
>
---
#### [replaced 020] CogDDN: A Cognitive Demand-Driven Navigation with Decision Optimization and Dual-Process Thinking
- **分类: cs.AI; cs.RO; I.2.9**

- **链接: [http://arxiv.org/pdf/2507.11334v2](http://arxiv.org/pdf/2507.11334v2)**

> **作者:** Yuehao Huang; Liang Liu; Shuangming Lei; Yukai Ma; Hao Su; Jianbiao Mei; Pengxiang Zhao; Yaqing Gu; Yong Liu; Jiajun Lv
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Mobile robots are increasingly required to navigate and interact within unknown and unstructured environments to meet human demands. Demand-driven navigation (DDN) enables robots to identify and locate objects based on implicit human intent, even when object locations are unknown. However, traditional data-driven DDN methods rely on pre-collected data for model training and decision-making, limiting their generalization capability in unseen scenarios. In this paper, we propose CogDDN, a VLM-based framework that emulates the human cognitive and learning mechanisms by integrating fast and slow thinking systems and selectively identifying key objects essential to fulfilling user demands. CogDDN identifies appropriate target objects by semantically aligning detected objects with the given instructions. Furthermore, it incorporates a dual-process decision-making module, comprising a Heuristic Process for rapid, efficient decisions and an Analytic Process that analyzes past errors, accumulates them in a knowledge base, and continuously improves performance. Chain of Thought (CoT) reasoning strengthens the decision-making process. Extensive closed-loop evaluations on the AI2Thor simulator with the ProcThor dataset show that CogDDN outperforms single-view camera-only methods by 15\%, demonstrating significant improvements in navigation accuracy and adaptability. The project page is available at https://yuehaohuang.github.io/CogDDN/.
>
---
