# 机器人 cs.RO

- **最新发布 56 篇**

- **更新 40 篇**

## 最新发布

#### [new 001] PROD: Palpative Reconstruction of Deformable Objects through Elastostatic Signed Distance Functions
- **分类: cs.RO; cs.CV**

- **简介: 论文提出PROD方法，通过力控接触测量重建软体物体的形状与力学特性。解决传统方法依赖视觉数据、难以估计材料刚度的问题。工作包括构建弹性静力学SDF模型，从稀疏力和位姿数据中恢复未变形形状并估计刚度，具备鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.12554v1](http://arxiv.org/pdf/2508.12554v1)**

> **作者:** Hamza El-Kebir
>
> **备注:** Accepted for presentation at the 2025 IEEE Conference on Decision and Control (CDC)
>
> **摘要:** We introduce PROD (Palpative Reconstruction of Deformables), a novel method for reconstructing the shape and mechanical properties of deformable objects using elastostatic signed distance functions (SDFs). Unlike traditional approaches that rely on purely geometric or visual data, PROD integrates palpative interaction -- measured through force-controlled surface probing -- to estimate both the static and dynamic response of soft materials. We model the deformation of an object as an elastostatic process and derive a governing Poisson equation for estimating its SDF from a sparse set of pose and force measurements. By incorporating steady-state elastodynamic assumptions, we show that the undeformed SDF can be recovered from deformed observations with provable convergence. Our approach also enables the estimation of material stiffness by analyzing displacement responses to varying force inputs. We demonstrate the robustness of PROD in handling pose errors, non-normal force application, and curvature errors in simulated soft body interactions. These capabilities make PROD a powerful tool for reconstructing deformable objects in applications ranging from robotic manipulation to medical imaging and haptic feedback systems.
>
---
#### [new 002] SIGN: Safety-Aware Image-Goal Navigation for Autonomous Drones via Reinforcement Learning
- **分类: cs.RO**

- **简介: 论文提出基于强化学习的无人机图像目标导航方法，解决未知环境中自主探索、避障与图像目标定位问题。通过视觉辅助训练和深度安全模块，实现端到端直接速度控制，无需外部定位或全局地图。**

- **链接: [http://arxiv.org/pdf/2508.12394v1](http://arxiv.org/pdf/2508.12394v1)**

> **作者:** Zichen Yan; Rui Huang; Lei He; Shao Guo; Lin Zhao
>
> **摘要:** Image-goal navigation (ImageNav) tasks a robot with autonomously exploring an unknown environment and reaching a location that visually matches a given target image. While prior works primarily study ImageNav for ground robots, enabling this capability for autonomous drones is substantially more challenging due to their need for high-frequency feedback control and global localization for stable flight. In this paper, we propose a novel sim-to-real framework that leverages visual reinforcement learning (RL) to achieve ImageNav for drones. To enhance visual representation ability, our approach trains the vision backbone with auxiliary tasks, including image perturbations and future transition prediction, which results in more effective policy training. The proposed algorithm enables end-to-end ImageNav with direct velocity control, eliminating the need for external localization. Furthermore, we integrate a depth-based safety module for real-time obstacle avoidance, allowing the drone to safely navigate in cluttered environments. Unlike most existing drone navigation methods that focus solely on reference tracking or obstacle avoidance, our framework supports comprehensive navigation behaviors--autonomous exploration, obstacle avoidance, and image-goal seeking--without requiring explicit global mapping. Code and model checkpoints will be released upon acceptance.
>
---
#### [new 003] Fully Spiking Actor-Critic Neural Network for Robotic Manipulation
- **分类: cs.RO**

- **简介: 论文提出基于全脉冲神经网络的强化学习框架，用于9自由度机械臂的目标抓取任务。为降低复杂度与延迟，简化网络结构并结合课程学习策略，提升学习效率与能效，在仿真中验证了方法在动态任务中的优越性。**

- **链接: [http://arxiv.org/pdf/2508.12038v1](http://arxiv.org/pdf/2508.12038v1)**

> **作者:** Liwen Zhang; Heng Deng; Guanghui Sun
>
> **摘要:** This study proposes a hybrid curriculum reinforcement learning (CRL) framework based on a fully spiking neural network (SNN) for 9-degree-of-freedom robotic arms performing target reaching and grasping tasks. To reduce network complexity and inference latency, the SNN architecture is simplified to include only an input and an output layer, which shows strong potential for resource-constrained environments. Building on the advantages of SNNs-high inference speed, low energy consumption, and spike-based biological plausibility, a temporal progress-partitioned curriculum strategy is integrated with the Proximal Policy Optimization (PPO) algorithm. Meanwhile, an energy consumption modeling framework is introduced to quantitatively compare the theoretical energy consumption between SNNs and conventional Artificial Neural Networks (ANNs). A dynamic two-stage reward adjustment mechanism and optimized observation space further improve learning efficiency and policy accuracy. Experiments on the Isaac Gym simulation platform demonstrate that the proposed method achieves superior performance under realistic physical constraints. Comparative evaluations with conventional PPO and ANN baselines validate the scalability and energy efficiency of the proposed approach in dynamic robotic manipulation tasks.
>
---
#### [new 004] Semi-Infinite Programming for Collision-Avoidance in Optimal and Model Predictive Control
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出基于半无限规划的碰撞避障方法，用于最优控制和模型预测控制，通过点云表示环境与机器人，解决无穷多约束下的高效避障问题，并实现高频率实时控制与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.12335v1](http://arxiv.org/pdf/2508.12335v1)**

> **作者:** Yunfan Gao; Florian Messerer; Niels van Duijkeren; Rashmi Dabir; Moritz Diehl
>
> **备注:** 21 pages, 15 figures
>
> **摘要:** This paper presents a novel approach for collision avoidance in optimal and model predictive control, in which the environment is represented by a large number of points and the robot as a union of padded polygons. The conditions that none of the points shall collide with the robot can be written in terms of an infinite number of constraints per obstacle point. We show that the resulting semi-infinite programming (SIP) optimal control problem (OCP) can be efficiently tackled through a combination of two methods: local reduction and an external active-set method. Specifically, this involves iteratively identifying the closest point obstacles, determining the lower-level distance minimizer among all feasible robot shape parameters, and solving the upper-level finitely-constrained subproblems. In addition, this paper addresses robust collision avoidance in the presence of ellipsoidal state uncertainties. Enforcing constraint satisfaction over all possible uncertainty realizations extends the dimension of constraint infiniteness. The infinitely many constraints arising from translational uncertainty are handled by local reduction together with the robot shape parameterization, while rotational uncertainty is addressed via a backoff reformulation. A controller implemented based on the proposed method is demonstrated on a real-world robot running at 20Hz, enabling fast and collision-free navigation in tight spaces. An application to 3D collision avoidance is also demonstrated in simulation.
>
---
#### [new 005] A robust and compliant robotic assembly control strategy for batch precision assembly task with uncertain fit types and fit amounts
- **分类: cs.RO**

- **简介: 论文针对不确定 fit 类型和 fit 量的批量高精度装配任务，提出基于力-视觉融合与多任务强化学习的鲁棒柔顺控制策略，通过分解子任务、多策略训练与知识蒸馏，提升装配成功率与效率。**

- **链接: [http://arxiv.org/pdf/2508.12296v1](http://arxiv.org/pdf/2508.12296v1)**

> **作者:** Bin Wang; Jiwen Zhang; Song Wang; Dan Wu
>
> **摘要:** In some high-precision industrial applications, robots are deployed to perform precision assembly tasks on mass batches of manufactured pegs and holes. If the peg and hole are designed with transition fit, machining errors may lead to either a clearance or an interference fit for a specific pair of components, with uncertain fit amounts. This paper focuses on the robotic batch precision assembly task involving components with uncertain fit types and fit amounts, and proposes an efficient methodology to construct the robust and compliant assembly control strategy. Specifically, the batch precision assembly task is decomposed into multiple deterministic subtasks, and a force-vision fusion controller-driven reinforcement learning method and a multi-task reinforcement learning training method (FVFC-MTRL) are proposed to jointly learn multiple compliance control strategies for these subtasks. Subsequently, the multi-teacher policy distillation approach is designed to integrate multiple trained strategies into a unified student network, thereby establishing a robust control strategy. Real-world experiments demonstrate that the proposed method successfully constructs the robust control strategy for high-precision assembly task with different fit types and fit amounts. Moreover, the MTRL framework significantly improves training efficiency, and the final developed control strategy achieves superior force compliance and higher success rate compared with many existing methods.
>
---
#### [new 006] Temporal and Rotational Calibration for Event-Centric Multi-Sensor Systems
- **分类: cs.RO; cs.CV; I.2.9**

- **简介: 论文提出一种无需标定物的事件相机多传感器时空与旋转标定方法，通过运动信息估计时间偏移和旋转外参，结合CCA初始化与非线性优化提升精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2508.12564v1](http://arxiv.org/pdf/2508.12564v1)**

> **作者:** Jiayao Mai; Xiuyuan Lu; Kuan Dai; Shaojie Shen; Yi Zhou
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Event cameras generate asynchronous signals in response to pixel-level brightness changes, offering a sensing paradigm with theoretically microsecond-scale latency that can significantly enhance the performance of multi-sensor systems. Extrinsic calibration is a critical prerequisite for effective sensor fusion; however, the configuration that involves event cameras remains an understudied topic. In this paper, we propose a motion-based temporal and rotational calibration framework tailored for event-centric multi-sensor systems, eliminating the need for dedicated calibration targets. Our method uses as input the rotational motion estimates obtained from event cameras and other heterogeneous sensors, respectively. Different from conventional approaches that rely on event-to-frame conversion, our method efficiently estimates angular velocity from normal flow observations, which are derived from the spatio-temporal profile of event data. The overall calibration pipeline adopts a two-step approach: it first initializes the temporal offset and rotational extrinsics by exploiting kinematic correlations in the spirit of Canonical Correlation Analysis (CCA), and then refines both temporal and rotational parameters through a joint non-linear optimization using a continuous-time parametrization in SO(3). Extensive evaluations on both publicly available and self-collected datasets validate that the proposed method achieves calibration accuracy comparable to target-based methods, while exhibiting superior stability over purely CCA-based methods, and highlighting its precision, robustness and flexibility. To facilitate future research, our implementation will be made open-source. Code: https://github.com/NAIL-HNU/EvMultiCalib.
>
---
#### [new 007] Energy Efficiency in Robotics Software: A Systematic Literature Review (2020-2024)
- **分类: cs.RO**

- **简介: 该论文属于系统性文献综述任务，旨在解决机器人软件能效研究方法与成果不统一的问题。作者通过自动化与人工结合的筛选流程，分析了2020–2024年79篇论文，识别出主流技术、能量消耗源和评估方式，并提出标准化报告建议与未来方向。**

- **链接: [http://arxiv.org/pdf/2508.12170v1](http://arxiv.org/pdf/2508.12170v1)**

> **作者:** Aryan Gupta
>
> **摘要:** This study presents a systematic literature review of software-level approaches to energy efficiency in robotics published from 2020 through 2024, updating and extending pre-2020 evidence. An automated-but-audited pipeline combined Google Scholar seeding, backward/forward snowballing, and large-language-model (LLM) assistance for screening and data extraction, with ~10% human audits at each automated step and consensus-with-tie-breaks for full-text decisions. The final corpus comprises 79 peer-reviewed studies analyzed across application domain, metrics, evaluation type, energy models, major energy consumers, software technique families, and energy-quality trade-offs. Industrial settings dominate (31.6%) followed by exploration (25.3%). Motors/actuators are identified as the primary consumer in 68.4% of studies, with computing/controllers a distant second (13.9%). Simulation-only evaluations remain most common (51.9%), though hybrid evaluations are frequent (25.3%). Representational (physics-grounded) energy models predominate (87.3%). Motion and trajectory optimization is the leading technique family (69.6%), often paired with learning/prediction (40.5%) and computation allocation/scheduling (26.6%); power management/idle control (11.4%) and communication/data efficiency (3.8%) are comparatively underexplored. Reporting is heterogeneous: composite objectives that include energy are most common, while task-normalized and performance-per-energy metrics appear less often, limiting cross-paper comparability. The review offers a minimal reporting checklist (e.g., total energy and average power plus a task-normalized metric and clear baselines) and highlights opportunities in cross-layer designs and in quantifying non-performance trade-offs (accuracy, stability). A replication package with code, prompts, and frozen datasets accompanies the review.
>
---
#### [new 008] Autonomous Oil Spill Response Through Liquid Neural Trajectory Modeling and Coordinated Marine Robotics
- **分类: cs.RO; 68T07, 93C85, 86A05; I.2.6; I.2.9; J.2**

- **简介: 该论文属于环境监测与应急响应任务，旨在解决油污扩散预测难、响应慢的问题。通过融合液态时间常数神经网络与多机器人协同系统，实现高精度实时轨迹预测与自主响应，提升应对效率与可靠性。**

- **链接: [http://arxiv.org/pdf/2508.12456v1](http://arxiv.org/pdf/2508.12456v1)**

> **作者:** Hadas C. Kuzmenko; David Ehevich; Oren Gal
>
> **备注:** 30 pages, 40 figures. Framework combining Liquid Time-Constant Neural Networks with autonomous marine robotics for oil spill trajectory prediction and response coordination
>
> **摘要:** Marine oil spills pose grave environmental and economic risks, threatening marine ecosystems, coastlines, and dependent industries. Predicting and managing oil spill trajectories is highly complex, due to the interplay of physical, chemical, and environmental factors such as wind, currents, and temperature, which makes timely and effective response challenging. Accurate real-time trajectory forecasting and coordinated mitigation are vital for minimizing the impact of these disasters. This study introduces an integrated framework combining a multi-agent swarm robotics system built on the MOOS-IvP platform with Liquid Time-Constant Neural Networks (LTCNs). The proposed system fuses adaptive machine learning with autonomous marine robotics, enabling real-time prediction, dynamic tracking, and rapid response to evolving oil spills. By leveraging LTCNs--well-suited for modeling complex, time-dependent processes--the framework achieves real-time, high-accuracy forecasts of spill movement. Swarm intelligence enables decentralized, scalable, and resilient decision-making among robot agents, enhancing collective monitoring and containment efforts. Our approach was validated using data from the Deepwater Horizon spill, where the LTC-RK4 model achieved 0.96 spatial accuracy, surpassing LSTM approaches by 23%. The integration of advanced neural modeling with autonomous, coordinated robotics demonstrates substantial improvements in prediction precision, flexibility, and operational scalability. Ultimately, this research advances the state-of-the-art for sustainable, autonomous oil spill management and environmental protection by enhancing both trajectory prediction and response coordination.
>
---
#### [new 009] Saliency-Based Attention Shifting: A Framework for Improving Driver Situational Awareness of Out-of-Label Hazards
- **分类: cs.RO; cs.HC**

- **简介: 论文提出基于显著性注意力转移的框架，解决半自动驾驶中驾驶员对未标注危险物警觉性不足的问题。通过实时眼动追踪、显著性分析与视听提示协同，提升驾驶员情境感知能力，确保接管时安全。**

- **链接: [http://arxiv.org/pdf/2508.11887v1](http://arxiv.org/pdf/2508.11887v1)**

> **作者:** Yousra Shleibik; Jordan Sinclair; Kerstin Haring
>
> **摘要:** The advent of autonomous driving systems promises to transform transportation by enhancing safety, efficiency, and comfort. As these technologies evolve toward higher levels of autonomy, the need for integrated systems that seamlessly support human involvement in decision-making becomes increasingly critical. Certain scenarios necessitate human involvement, including those where the vehicle is unable to identify an object or element in the scene, and as such cannot take independent action. Therefore, situational awareness is essential to mitigate potential risks during a takeover, where a driver must assume control and autonomy from the vehicle. The need for driver attention is important to avoid collisions with external agents and ensure a smooth transition during takeover operations. This paper explores the integration of attention redirection techniques, such as gaze manipulation through targeted visual and auditory cues, to help drivers maintain focus on emerging hazards and reduce target fixation in semi-autonomous driving scenarios. We propose a conceptual framework that combines real-time gaze tracking, context-aware saliency analysis, and synchronized visual and auditory alerts to enhance situational awareness, proactively address potential hazards, and foster effective collaboration between humans and autonomous systems.
>
---
#### [new 010] RoboRetriever: Single-Camera Robot Object Retrieval via Active and Interactive Perception with Dynamic Scene Graph
- **分类: cs.RO**

- **简介: 论文提出RoboRetriever框架，解决单摄像头下复杂场景中的物体检索问题。通过动态场景图和视觉提示实现主动感知与交互，结合自然语言指令完成精准抓取。**

- **链接: [http://arxiv.org/pdf/2508.12916v1](http://arxiv.org/pdf/2508.12916v1)**

> **作者:** Hecheng Wang; Jiankun Ren; Jia Yu; Lizhe Qi; Yunquan Sun
>
> **摘要:** Humans effortlessly retrieve objects in cluttered, partially observable environments by combining visual reasoning, active viewpoint adjustment, and physical interaction-with only a single pair of eyes. In contrast, most existing robotic systems rely on carefully positioned fixed or multi-camera setups with complete scene visibility, which limits adaptability and incurs high hardware costs. We present \textbf{RoboRetriever}, a novel framework for real-world object retrieval that operates using only a \textbf{single} wrist-mounted RGB-D camera and free-form natural language instructions. RoboRetriever grounds visual observations to build and update a \textbf{dynamic hierarchical scene graph} that encodes object semantics, geometry, and inter-object relations over time. The supervisor module reasons over this memory and task instruction to infer the target object and coordinate an integrated action module combining \textbf{active perception}, \textbf{interactive perception}, and \textbf{manipulation}. To enable task-aware scene-grounded active perception, we introduce a novel visual prompting scheme that leverages large reasoning vision-language models to determine 6-DoF camera poses aligned with the semantic task goal and geometry scene context. We evaluate RoboRetriever on diverse real-world object retrieval tasks, including scenarios with human intervention, demonstrating strong adaptability and robustness in cluttered scenes with only one RGB-D camera.
>
---
#### [new 011] Humanoid Motion Scripting with Postural Synergies
- **分类: cs.RO**

- **简介: 论文提出SynSculptor框架，利用后姿势协同机制实现无需训练的人形机器人动作脚本生成。解决人类动作采集、合成与映射难题，通过PCA提取运动协同，构建风格条件库，并用Transformer实现动作执行时的姿态自适应调整。**

- **链接: [http://arxiv.org/pdf/2508.12184v1](http://arxiv.org/pdf/2508.12184v1)**

> **作者:** Rhea Malhotra; William Chong; Catie Cuan; Oussama Khatib
>
> **摘要:** Generating sequences of human-like motions for humanoid robots presents challenges in collecting and analyzing reference human motions, synthesizing new motions based on these reference motions, and mapping the generated motion onto humanoid robots. To address these issues, we introduce SynSculptor, a humanoid motion analysis and editing framework that leverages postural synergies for training-free human-like motion scripting. To analyze human motion, we collect 3+ hours of motion capture data across 20 individuals where a real-time operational space controller mimics human motion on a simulated humanoid robot. The major postural synergies are extracted using principal component analysis (PCA) for velocity trajectories segmented by changes in robot momentum, constructing a style-conditioned synergy library for free-space motion generation. To evaluate generated motions using the synergy library, the foot-sliding ratio and proposed metrics for motion smoothness involving total momentum and kinetic energy deviations are computed for each generated motion, and compared with reference motions. Finally, we leverage the synergies with a motion-language transformer, where the humanoid, during execution of motion tasks with its end-effectors, adapts its posture based on the chosen synergy. Supplementary material, code, and videos are available at https://rhea-mal.github.io/humanoidsynergies.io.
>
---
#### [new 012] Control of Legged Robots using Model Predictive Optimized Path Integral
- **分类: cs.RO**

- **简介: 论文提出MPOPI方法，结合MPPI与CE、CMA优化，提升腿式机器人在复杂地形中的实时运动控制效率，以更少样本实现更好步态性能。**

- **链接: [http://arxiv.org/pdf/2508.11917v1](http://arxiv.org/pdf/2508.11917v1)**

> **作者:** Hossein Keshavarz; Alejandro Ramirez-Serrano; Majid Khadiv
>
> **备注:** 8 pages, 13 figures, Humanoid conference
>
> **摘要:** Legged robots possess a unique ability to traverse rough terrains and navigate cluttered environments, making them well-suited for complex, real-world unstructured scenarios. However, such robots have not yet achieved the same level as seen in natural systems. Recently, sampling-based predictive controllers have demonstrated particularly promising results. This paper investigates a sampling-based model predictive strategy combining model predictive path integral (MPPI) with cross-entropy (CE) and covariance matrix adaptation (CMA) methods to generate real-time whole-body motions for legged robots across multiple scenarios. The results show that combining the benefits of MPPI, CE and CMA, namely using model predictive optimized path integral (MPOPI), demonstrates greater sample efficiency, enabling robots to attain superior locomotion results using fewer samples when compared to typical MPPI algorithms. Extensive simulation experiments in multiple scenarios on a quadruped robot show that MPOPI can be used as an anytime control strategy, increasing locomotion capabilities at each iteration.
>
---
#### [new 013] Bimanual Robot-Assisted Dressing: A Spherical Coordinate-Based Strategy for Tight-Fitting Garments
- **分类: cs.RO**

- **简介: 论文研究机器人辅助穿紧身衣物任务，针对单臂易卡住的问题，提出基于球坐标系的双臂协作策略，利用GMM/GMR学习适应不同手臂姿态的穿衣轨迹，提升成功率。**

- **链接: [http://arxiv.org/pdf/2508.12274v1](http://arxiv.org/pdf/2508.12274v1)**

> **作者:** Jian Zhao; Yunlong Lian; Andy M Tyrrell; Michael Gienger; Jihong Zhu
>
> **备注:** 8 pages, 41 figures
>
> **摘要:** Robot-assisted dressing is a popular but challenging topic in the field of robotic manipulation, offering significant potential to improve the quality of life for individuals with mobility limitations. Currently, the majority of research on robot-assisted dressing focuses on how to put on loose-fitting clothing, with little attention paid to tight garments. For the former, since the armscye is larger, a single robotic arm can usually complete the dressing task successfully. However, for the latter, dressing with a single robotic arm often fails due to the narrower armscye and the property of diminishing rigidity in the armscye, which eventually causes the armscye to get stuck. This paper proposes a bimanual dressing strategy suitable for dressing tight-fitting clothing. To facilitate the encoding of dressing trajectories that adapt to different human arm postures, a spherical coordinate system for dressing is established. We uses the azimuthal angle of the spherical coordinate system as a task-relevant feature for bimanual manipulation. Based on this new coordinate, we employ Gaussian Mixture Model (GMM) and Gaussian Mixture Regression (GMR) for imitation learning of bimanual dressing trajectories, generating dressing strategies that adapt to different human arm postures. The effectiveness of the proposed method is validated through various experiments.
>
---
#### [new 014] Mechanical Automation with Vision: A Design for Rubik's Cube Solver
- **分类: cs.RO; cs.CV**

- **简介: 论文提出了一种基于视觉的鲁比克魔方自动求解系统，通过YOLOv8模型识别魔方状态，利用Kociemba算法计算解法，并由三轴步进电机执行物理操作，平均解题时间约2.2分钟。**

- **链接: [http://arxiv.org/pdf/2508.12469v1](http://arxiv.org/pdf/2508.12469v1)**

> **作者:** Abhinav Chalise; Nimesh Gopal Pradhan; Nishan Khanal; Prashant Raj Bista; Dinesh Baniya Kshatri
>
> **备注:** Presented at the 15th IOE Graduate Conference, Tribhuvan University, May 2024. Original paper available at https://conference.ioe.edu.np/publications/ioegc15/IOEGC-15-023-C1-2-42.pdf
>
> **摘要:** The core mechanical system is built around three stepper motors for physical manipulation, a microcontroller for hardware control, a camera and YOLO detection model for real-time cube state detection. A significant software component is the development of a user-friendly graphical user interface (GUI) designed in Unity. The initial state after detection from real-time YOLOv8 model (Precision 0.98443, Recall 0.98419, Box Loss 0.42051, Class Loss 0.2611) is virtualized on GUI. To get the solution, the system employs the Kociemba's algorithm while physical manipulation with a single degree of freedom is done by combination of stepper motors' interaction with the cube achieving the average solving time of ~2.2 minutes.
>
---
#### [new 015] Self-Guided Action Diffusion
- **分类: cs.RO; cs.AI**

- **简介: 论文提出自引导动作扩散方法，用于提升扩散模型生成机器人策略的效率与性能。针对现有双向解码计算成本高的问题，通过引入前序决策引导每步提议分布，实现近最优效果且推理开销极低，在动态任务中显著提高成功率。**

- **链接: [http://arxiv.org/pdf/2508.12189v1](http://arxiv.org/pdf/2508.12189v1)**

> **作者:** Rhea Malhotra; Yuejiang Liu; Chelsea Finn
>
> **摘要:** Recent works have shown the promise of inference-time search over action samples for improving generative robot policies. In particular, optimizing cross-chunk coherence via bidirectional decoding has proven effective in boosting the consistency and reactivity of diffusion policies. However, this approach remains computationally expensive as the diversity of sampled actions grows. In this paper, we introduce self-guided action diffusion, a more efficient variant of bidirectional decoding tailored for diffusion-based policies. At the core of our method is to guide the proposal distribution at each diffusion step based on the prior decision. Experiments in simulation tasks show that the proposed self-guidance enables near-optimal performance at negligible inference cost. Notably, under a tight sampling budget, our method achieves up to 70% higher success rates than existing counterparts on challenging dynamic tasks. See project website at https://rhea-mal.github.io/selfgad.github.io.
>
---
#### [new 016] Improving Pre-Trained Vision-Language-Action Policies with Model-Based Search
- **分类: cs.RO; cs.AI**

- **简介: 论文提出VLAPS框架，将模型搜索嵌入预训练视觉-语言-动作（VLA）策略推理中，解决其在分布外场景下行为脆弱的问题。通过结合VLA动作先验与环境模型的蒙特卡洛树搜索，显著提升任务成功率，最高达67个百分点。**

- **链接: [http://arxiv.org/pdf/2508.12211v1](http://arxiv.org/pdf/2508.12211v1)**

> **作者:** Cyrus Neary; Omar G. Younis; Artur Kuramshin; Ozgur Aslan; Glen Berseth
>
> **摘要:** Pre-trained vision-language-action (VLA) models offer a promising foundation for generalist robot policies, but often produce brittle behaviours or unsafe failures when deployed zero-shot in out-of-distribution scenarios. We present Vision-Language-Action Planning & Search (VLAPS) -- a novel framework and accompanying algorithms that embed model-based search into the inference procedure of pre-trained VLA policies to improve their performance on robotic tasks. Specifically, our method biases a modified Monte Carlo Tree Search (MCTS) algorithm -- run using a model of the target environment -- using action priors defined by the VLA policy. By using VLA-derived abstractions and priors in model-based search, VLAPS efficiently explores language-conditioned robotics tasks whose search spaces would otherwise be intractably large. Conversely, by integrating model-based search with the VLA policy's inference procedure, VLAPS yields behaviours that are more performant than those obtained by directly following the VLA policy's action predictions. VLAPS offers a principled framework to: i) control test-time compute in VLA models, ii) leverage a priori knowledge of the robotic environment, and iii) integrate established planning and reinforcement learning techniques into the VLA inference process. Across all experiments, VLAPS significantly outperforms VLA-only baselines on language-specified tasks that would otherwise be intractable for uninformed search algorithms, increasing success rates by as much as 67 percentage points.
>
---
#### [new 017] ExploreVLM: Closed-Loop Robot Exploration Task Planning with Vision-Language Models
- **分类: cs.RO**

- **简介: 论文提出ExploreVLM框架，用于机器人闭环任务规划，解决VLM在交互探索、感知准确性和实时调整上的不足。通过双阶段反思式规划和结构化场景表示，提升机器人在动态环境中的任务执行能力。**

- **链接: [http://arxiv.org/pdf/2508.11918v1](http://arxiv.org/pdf/2508.11918v1)**

> **作者:** Zhichen Lou; Kechun Xu; Zhongxiang Zhou; Rong Xiong
>
> **摘要:** The advancement of embodied intelligence is accelerating the integration of robots into daily life as human assistants. This evolution requires robots to not only interpret high-level instructions and plan tasks but also perceive and adapt within dynamic environments. Vision-Language Models (VLMs) present a promising solution by combining visual understanding and language reasoning. However, existing VLM-based methods struggle with interactive exploration, accurate perception, and real-time plan adaptation. To address these challenges, we propose ExploreVLM, a novel closed-loop task planning framework powered by Vision-Language Models (VLMs). The framework is built around a step-wise feedback mechanism that enables real-time plan adjustment and supports interactive exploration. At its core is a dual-stage task planner with self-reflection, enhanced by an object-centric spatial relation graph that provides structured, language-grounded scene representations to guide perception and planning. An execution validator supports the closed loop by verifying each action and triggering re-planning. Extensive real-world experiments demonstrate that ExploreVLM significantly outperforms state-of-the-art baselines, particularly in exploration-centric tasks. Ablation studies further validate the critical role of the reflective planner and structured perception in achieving robust and efficient task execution.
>
---
#### [new 018] Into the Wild: When Robots Are Not Welcome
- **分类: cs.RO; cs.HC**

- **简介: 论文探讨社会机器人在公共空间部署中的挑战，聚焦于用户和利益相关者的接受度问题。通过两个实际场景（学生服务中心与难民服务点），研究者最终赢得工作人员信任，实现机器人部署与研究开展。任务为提升机器人在真实环境中的可接受性与实用性。**

- **链接: [http://arxiv.org/pdf/2508.12075v1](http://arxiv.org/pdf/2508.12075v1)**

> **作者:** Shaul Ashkenazi; Gabriel Skantze; Jane Stuart-Smith; Mary Ellen Foster
>
> **备注:** Accepted at the workshop on Real-World HRI in Public and Private Spaces: Successes, Failures, and Lessons Learned (PubRob-Fails), held at the IEEE RO-MAN Conference, 2025 (paper PubRob-Fails/2025/4)
>
> **摘要:** Social robots are increasingly being deployed in public spaces, where they face not only technological difficulties and unexpected user utterances, but also objections from stakeholders who may not be comfortable with introducing a robot into those spaces. We describe our difficulties with deploying a social robot in two different public settings: 1) Student services center; 2) Refugees and asylum seekers drop-in service. Although this is a failure report, in each use case we eventually managed to earn the trust of the staff and form a relationship with them, allowing us to deploy our robot and conduct our studies.
>
---
#### [new 019] Using Natural Language for Human-Robot Collaboration in the Real World
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 论文探讨如何利用大语言模型（LLM）提升机器人在现实世界中理解自然语言的能力，以实现与人类更有效的协作。针对语言理解挑战，提出基于认知代理的AI系统架构，并通过实验验证其可行性，旨在构建集成语言能力的机器人助手。**

- **链接: [http://arxiv.org/pdf/2508.11759v1](http://arxiv.org/pdf/2508.11759v1)**

> **作者:** Peter Lindes; Kaoutar Skiker
>
> **备注:** 34 pages, 11 figures, 5 tables. Submitted for publication (2026) in W.F. Lawless, Ranjeev Mittu, Shannon P. McGrarry, & Marco Brambilla (Eds.), Generative AI Risks and Benefits within Human-Machine Teams, Elsevier, Chapter 6
>
> **摘要:** We have a vision of a day when autonomous robots can collaborate with humans as assistants in performing complex tasks in the physical world. This vision includes that the robots will have the ability to communicate with their human collaborators using language that is natural to the humans. Traditional Interactive Task Learning (ITL) systems have some of this ability, but the language they can understand is very limited. The advent of large language models (LLMs) provides an opportunity to greatly improve the language understanding of robots, yet integrating the language abilities of LLMs with robots that operate in the real physical world is a challenging problem. In this chapter we first review briefly a few commercial robot products that work closely with humans, and discuss how they could be much better collaborators with robust language abilities. We then explore how an AI system with a cognitive agent that controls a physical robot at its core, interacts with both a human and an LLM, and accumulates situational knowledge through its experiences, can be a possible approach to reach that vision. We focus on three specific challenges of having the robot understand natural language, and present a simple proof-of-concept experiment using ChatGPT for each. Finally, we discuss what it will take to turn these simple experiments into an operational system where LLM-assisted language understanding is a part of an integrated robotic assistant that uses language to collaborate with humans.
>
---
#### [new 020] Manipulate-to-Navigate: Reinforcement Learning with Visual Affordances and Manipulability Priors
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出一种基于强化学习的“操纵引导导航”方法，解决动态环境中障碍物阻碍路径的问题。通过结合可操作性先验和视觉 affordance 图，引导机器人先操纵障碍物再导航，提升策略学习效率。在模拟和真实机器人上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2508.13151v1](http://arxiv.org/pdf/2508.13151v1)**

> **作者:** Yuying Zhang; Joni Pajarinen
>
> **摘要:** Mobile manipulation in dynamic environments is challenging due to movable obstacles blocking the robot's path. Traditional methods, which treat navigation and manipulation as separate tasks, often fail in such 'manipulate-to-navigate' scenarios, as obstacles must be removed before navigation. In these cases, active interaction with the environment is required to clear obstacles while ensuring sufficient space for movement. To address the manipulate-to-navigate problem, we propose a reinforcement learning-based approach for learning manipulation actions that facilitate subsequent navigation. Our method combines manipulability priors to focus the robot on high manipulability body positions with affordance maps for selecting high-quality manipulation actions. By focusing on feasible and meaningful actions, our approach reduces unnecessary exploration and allows the robot to learn manipulation strategies more effectively. We present two new manipulate-to-navigate simulation tasks called Reach and Door with the Boston Dynamics Spot robot. The first task tests whether the robot can select a good hand position in the target area such that the robot base can move effectively forward while keeping the end effector position fixed. The second task requires the robot to move a door aside in order to clear the navigation path. Both of these tasks need first manipulation and then navigating the base forward. Results show that our method allows a robot to effectively interact with and traverse dynamic environments. Finally, we transfer the learned policy to a real Boston Dynamics Spot robot, which successfully performs the Reach task.
>
---
#### [new 021] Integrating Symbolic RL Planning into a BDI-based Autonomous UAV Framework: System Integration and SIL Validation
- **分类: cs.RO; cs.AI**

- **简介: 论文提出AMAD-SRL框架，将符号强化学习融入BDI架构，提升无人机动态任务规划能力。解决传统规则系统在复杂环境中适应性差的问题，通过SIL验证实现高效路径规划，任务效率提升75%。**

- **链接: [http://arxiv.org/pdf/2508.11890v1](http://arxiv.org/pdf/2508.11890v1)**

> **作者:** Sangwoo Jeon; Juchul Shin; YeonJe Cho; Gyeong-Tae Kim; Seongwoo Kim
>
> **摘要:** Modern autonomous drone missions increasingly require software frameworks capable of seamlessly integrating structured symbolic planning with adaptive reinforcement learning (RL). Although traditional rule-based architectures offer robust structured reasoning for drone autonomy, their capabilities fall short in dynamically complex operational environments that require adaptive symbolic planning. Symbolic RL (SRL), using the Planning Domain Definition Language (PDDL), explicitly integrates domain-specific knowledge and operational constraints, significantly improving the reliability and safety of unmanned aerial vehicle (UAV) decision making. In this study, we propose the AMAD-SRL framework, an extended and refined version of the Autonomous Mission Agents for Drones (AMAD) cognitive multi-agent architecture, enhanced with symbolic reinforcement learning for dynamic mission planning and execution. We validated our framework in a Software-in-the-Loop (SIL) environment structured identically to an intended Hardware-In-the-Loop Simulation (HILS) platform, ensuring seamless transition to real hardware. Experimental results demonstrate stable integration and interoperability of modules, successful transitions between BDI-driven and symbolic RL-driven planning phases, and consistent mission performance. Specifically, we evaluate a target acquisition scenario in which the UAV plans a surveillance path followed by a dynamic reentry path to secure the target while avoiding threat zones. In this SIL evaluation, mission efficiency improved by approximately 75% over a coverage-based baseline, measured by travel distance reduction. This study establishes a robust foundation for handling complex UAV missions and discusses directions for further enhancement and validation.
>
---
#### [new 022] Adaptive Model-Predictive Control of a Soft Continuum Robot Using a Physics-Informed Neural Network Based on Cosserat Rod Theory
- **分类: cs.RO; cs.LG**

- **简介: 论文提出一种基于物理信息神经网络的自适应模型预测控制方法，用于软连续机器人实时动态控制。解决高计算成本与模型精度不足问题，通过DD-PINN加速动力学建模并估计状态，实现高精度轨迹跟踪与实测控制。**

- **链接: [http://arxiv.org/pdf/2508.12681v1](http://arxiv.org/pdf/2508.12681v1)**

> **作者:** Johann Licher; Max Bartholdt; Henrik Krauss; Tim-Lukas Habich; Thomas Seel; Moritz Schappler
>
> **备注:** 20 pages, 15 figures
>
> **摘要:** Dynamic control of soft continuum robots (SCRs) holds great potential for expanding their applications, but remains a challenging problem due to the high computational demands of accurate dynamic models. While data-driven approaches like Koopman-operator-based methods have been proposed, they typically lack adaptability and cannot capture the full robot shape, limiting their applicability. This work introduces a real-time-capable nonlinear model-predictive control (MPC) framework for SCRs based on a domain-decoupled physics-informed neural network (DD-PINN) with adaptable bending stiffness. The DD-PINN serves as a surrogate for the dynamic Cosserat rod model with a speed-up factor of 44000. It is also used within an unscented Kalman filter for estimating the model states and bending compliance from end-effector position measurements. We implement a nonlinear evolutionary MPC running at 70 Hz on the GPU. In simulation, it demonstrates accurate tracking of dynamic trajectories and setpoint control with end-effector position errors below 3 mm (2.3% of the actuator's length). In real-world experiments, the controller achieves similar accuracy and accelerations up to 3.55 m/s2.
>
---
#### [new 023] Tactile Gesture Recognition with Built-in Joint Sensors for Industrial Robots
- **分类: cs.RO; cs.AI**

- **简介: 论文研究工业机器人中仅用内置关节传感器进行触觉手势识别的任务，解决外部传感器依赖问题。通过收集数据集并测试CNN模型，发现谱图表示显著提升准确率，且在新姿态下泛化性更好，实现超95%的识别精度。**

- **链接: [http://arxiv.org/pdf/2508.12435v1](http://arxiv.org/pdf/2508.12435v1)**

> **作者:** Deqing Song; Weimin Yang; Maryam Rezayati; Hans Wernher van de Venn
>
> **摘要:** While gesture recognition using vision or robot skins is an active research area in Human-Robot Collaboration (HRC), this paper explores deep learning methods relying solely on a robot's built-in joint sensors, eliminating the need for external sensors. We evaluated various convolutional neural network (CNN) architectures and collected two datasets to study the impact of data representation and model architecture on the recognition accuracy. Our results show that spectrogram-based representations significantly improve accuracy, while model architecture plays a smaller role. We also tested generalization to new robot poses, where spectrogram-based models performed better. Implemented on a Franka Emika Research robot, two of our methods, STFT2DCNN and STT3DCNN, achieved over 95% accuracy in contact detection and gesture classification. These findings demonstrate the feasibility of external-sensor-free tactile recognition and promote further research toward cost-effective, scalable solutions for HRC.
>
---
#### [new 024] Insights from Interviews with Teachers and Students on the Use of a Social Robot in Computer Science Class in Sixth Grade
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究社会机器人在六年级计算机科学课堂中的应用，通过访谈教师和学生了解需求与潜在用途。任务是探索机器人教学的可行性与设计挑战，解决如何满足不同用户群体的需求问题。**

- **链接: [http://arxiv.org/pdf/2508.12946v1](http://arxiv.org/pdf/2508.12946v1)**

> **作者:** Ann-Sophie Schenk; Stefan Schiffer; Heqiu Song
>
> **摘要:** In this paper we report on first insights from interviews with teachers and students on using social robots in computer science class in sixth grade. Our focus is on learning about requirements and potential applications. We are particularly interested in getting both perspectives, the teachers' and the learners' view on how robots could be used and what features they should or should not have. Results show that teachers as well as students are very open to robots in the classroom. However, requirements are partially quite heterogeneous among the groups. This leads to complex design challenges which we discuss at the end of this paper.
>
---
#### [new 025] Deformation of the panoramic sphere into an ellipsoid to induce self-motion in telepresence users
- **分类: cs.RO; cs.HC**

- **简介: 论文研究如何通过变形全景球为椭球来制造自运动幻觉，缓解远程操控机器人时的延迟问题。针对高延迟导致控制困难的任务，提出利用光流生成虚拟运动感，但实验表明该方法在500ms延迟下未提升控制性能，反而可能加剧晕动症。**

- **链接: [http://arxiv.org/pdf/2508.12925v1](http://arxiv.org/pdf/2508.12925v1)**

> **作者:** Eetu Laukka; Evan G. Center; Timo Ojala; Steven M. LaValle; Matti Pouke
>
> **备注:** 2025 IEEE Conference on Telepresence
>
> **摘要:** Mobile telepresence robots allow users to feel present and explore remote environments using technology. Traditionally, these systems are implemented using a camera onboard a mobile robot that can be controlled. Although high-immersion technologies, such as 360-degree cameras, can increase situational awareness and presence, they also introduce significant challenges. Additional processing and bandwidth requirements often result in latencies of up to seconds. The current delay with a 360-degree camera streaming over the internet makes real-time control of these systems difficult. Working with high-latency systems requires some form of assistance to the users. This study presents a novel way to utilize optical flow to create an illusion of self-motion to the user during the latency period between user sending motion commands to the robot and seeing the actual motion through the 360-camera stream. We find no significant benefit of using the self-motion illusion to performance or accuracy of controlling a telepresence robot with a latency of 500 ms, as measured by the task completion time and collisions into objects. Some evidence is shown that the method might increase virtual reality (VR) sickness, as measured by the simulator sickness questionnaire (SSQ). We conclude that further adjustments are necessary in order to render the method viable.
>
---
#### [new 026] LocoMamba: Vision-Driven Locomotion via End-to-End Deep Reinforcement Learning with Mamba
- **分类: cs.RO**

- **简介: 论文提出LocoMamba，基于Mamba的视觉驱动运动控制框架，解决复杂环境中机器人高效、鲁棒行走问题。通过端到端深度强化学习，融合本体感知与深度图像，实现长序列建模与快速训练，提升泛化能力与安全性。**

- **链接: [http://arxiv.org/pdf/2508.11849v1](http://arxiv.org/pdf/2508.11849v1)**

> **作者:** Allen Wang; Gavin Tao
>
> **摘要:** We introduce LocoMamba, a vision-driven cross-modal DRL framework built on selective state-space models, specifically leveraging Mamba, that achieves near-linear-time sequence modeling, effectively captures long-range dependencies, and enables efficient training with longer sequences. First, we embed proprioceptive states with a multilayer perceptron and patchify depth images with a lightweight convolutional neural network, producing compact tokens that improve state representation. Second, stacked Mamba layers fuse these tokens via near-linear-time selective scanning, reducing latency and memory footprint, remaining robust to token length and image resolution, and providing an inductive bias that mitigates overfitting. Third, we train the policy end-to-end with Proximal Policy Optimization under terrain and appearance randomization and an obstacle-density curriculum, using a compact state-centric reward that balances progress, smoothness, and safety. We evaluate our method in challenging simulated environments with static and moving obstacles as well as uneven terrain. Compared with state-of-the-art baselines, our method achieves higher returns and success rates with fewer collisions, exhibits stronger generalization to unseen terrains and obstacle densities, and improves training efficiency by converging in fewer updates under the same compute budget.
>
---
#### [new 027] Contact-Rich and Deformable Foot Modeling for Locomotion Control of the Human Musculoskeletal System
- **分类: cs.RO**

- **简介: 论文提出一种接触丰富且可变形的足部模型，用于改进人体运动系统中的步态控制。解决现有模型简化足地接触导致仿真不准确的问题。通过两阶段策略训练学习自然行走模式，并验证了其在运动学、动力学和步态稳定性上的优越性。**

- **链接: [http://arxiv.org/pdf/2508.11885v1](http://arxiv.org/pdf/2508.11885v1)**

> **作者:** Haixin Gong; Chen Zhang; Yanan Sui
>
> **备注:** IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids 2025)
>
> **摘要:** The human foot serves as the critical interface between the body and environment during locomotion. Existing musculoskeletal models typically oversimplify foot-ground contact mechanics, limiting their ability to accurately simulate human gait dynamics. We developed a novel contact-rich and deformable model of the human foot integrated within a complete musculoskeletal system that captures the complex biomechanical interactions during walking. To overcome the control challenges inherent in modeling multi-point contacts and deformable material, we developed a two-stage policy training strategy to learn natural walking patterns for this interface-enhanced model. Comparative analysis between our approach and conventional rigid musculoskeletal models demonstrated improvements in kinematic, kinetic, and gait stability metrics. Validation against human subject data confirmed that our simulation closely reproduced real-world biomechanical measurements. This work advances contact-rich interface modeling for human musculoskeletal systems and establishes a robust framework that can be extended to humanoid robotics applications requiring precise foot-ground interaction control.
>
---
#### [new 028] Geodesic Tracing-Based Kinematic Integration of Rolling and Sliding Contact on Manifold Meshes for Dexterous In-Hand Manipulation
- **分类: cs.RO**

- **简介: 论文研究机器人手指在手中操作中滚动与滑动接触的建模问题，提出基于测地线追踪的积分方法，实现对网格表面接触的高精度模拟，提升操作稳定性与准确性。**

- **链接: [http://arxiv.org/pdf/2508.12439v1](http://arxiv.org/pdf/2508.12439v1)**

> **作者:** Sunyu Wang; Arjun S. Lakshmipathy; Jean Oh; Nancy S. Pollard
>
> **摘要:** Reasoning about rolling and sliding contact, or roll-slide contact for short, is critical for dexterous manipulation tasks that involve intricate geometries. But existing works on roll-slide contact mostly focus on continuous shapes with differentiable parametrizations. This work extends roll-slide contact modeling to manifold meshes. Specifically, we present an integration scheme based on geodesic tracing to first-order time-integrate roll-slide contact directly on meshes, enabling dexterous manipulation to reason over high-fidelity discrete representations of an object's true geometry. Using our method, we planned dexterous motions of a multi-finger robotic hand manipulating five objects in-hand in simulation. The planning was achieved with a least-squares optimizer that strives to maintain the most stable instantaneous grasp by minimizing contact sliding and spinning. Then, we evaluated our method against a baseline using collision detection and a baseline using primitive shapes. The results show that our method performed the best in accuracy and precision, even for coarse meshes. We conclude with a future work discussion on incorporating multiple contacts and contact forces to achieve accurate and robust mesh-based surface contact modeling.
>
---
#### [new 029] PUB: A Plasma-Propelled Ultra-Quiet Blimp with Two-DOF Vector Thrusting
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出一种等离子体推进的超静音飞艇（PUB），解决传统飞艇噪声大、操控性差的问题。通过四层不对称电容产生离子风推力，实现无机械部件的安静飞行与二维矢量控制，验证了其全包线飞行能力。**

- **链接: [http://arxiv.org/pdf/2508.12395v1](http://arxiv.org/pdf/2508.12395v1)**

> **作者:** Zihan Wang
>
> **摘要:** This study presents the design and control of a Plasma-propelled Ultra-silence Blimp (PUB), a novel aerial robot employing plasma vector propulsion for ultra-quiet flight without mechanical propellers. The system utilizes a helium-lift platform for extended endurance and a four-layer ring asymmetric capacitor to generate ionic wind thrust. The modular propulsion units allow flexible configuration to meet mission-specific requirements, while a two-degree-of-freedom (DOF) head enables thrust vector control. A closed-loop slip control scheme is implemented for stable maneuvering. Flight experiments demonstrate full-envelope capability, including take-off, climb, hover, descent, and smooth landing, confirming the feasibility of plasma vector propulsion, the effectiveness of DOF vector control, and the stability of the control system. Owing to its low acoustic signature, structural simplicity, and high maneuverability, PUB is well suited for noise-sensitive, enclosed, and near-space applications.
>
---
#### [new 030] OASIS: Real-Time Opti-Acoustic Sensing for Intervention Systems in Unstructured Environments
- **分类: cs.RO**

- **简介: 论文提出OASIS方法，用于水下非结构化环境中的实时三维重建任务。针对现有方法多为离线处理的问题，该工作结合光学相机与声呐数据，利用机械臂实现多视角快速融合，提升水下操作的实时空间感知能力。**

- **链接: [http://arxiv.org/pdf/2508.12071v1](http://arxiv.org/pdf/2508.12071v1)**

> **作者:** Amy Phung; Richard Camilli
>
> **备注:** This paper has been accepted for publication in IROS 2025. Copyright IEEE
>
> **摘要:** High resolution underwater 3D scene reconstruction is crucial for various applications, including construction, infrastructure maintenance, monitoring, exploration, and scientific investigation. Prior work has leveraged the complementary sensing modalities of imaging sonars and optical cameras for opti-acoustic 3D scene reconstruction, demonstrating improved results over methods which rely solely on either sensor. However, while most existing approaches focus on offline reconstruction, real-time spatial awareness is essential for both autonomous and piloted underwater vehicle operations. This paper presents OASIS, an opti-acoustic fusion method that integrates data from optical images with voxel carving techniques to achieve real-time 3D reconstruction unstructured underwater workspaces. Our approach utilizes an "eye-in-hand" configuration, which leverages the dexterity of robotic manipulator arms to capture multiple workspace views across a short baseline. We validate OASIS through tank-based experiments and present qualitative and quantitative results that highlight its utility for underwater manipulation tasks.
>
---
#### [new 031] Anticipatory and Adaptive Footstep Streaming for Teleoperated Bipedal Robots
- **分类: cs.RO**

- **简介: 论文提出了一种用于遥控双足机器人的步态传输方法，解决用户与机器人运动不同步的问题。通过预测和自适应调整步态位置，使机器人在复杂地形中保持稳定，实现更自然的实时操作。**

- **链接: [http://arxiv.org/pdf/2508.11802v1](http://arxiv.org/pdf/2508.11802v1)**

> **作者:** Luigi Penco; Beomyeong Park; Stefan Fasano; Nehar Poddar; Stephen McCrory; Nicholas Kitchel; Tomasz Bialek; Dexton Anderson; Duncan Calvert; Robert Griffin
>
> **备注:** 2025 IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids)
>
> **摘要:** Achieving seamless synchronization between user and robot motion in teleoperation, particularly during high-speed tasks, remains a significant challenge. In this work, we propose a novel approach for transferring stepping motions from the user to the robot in real-time. Instead of directly replicating user foot poses, we retarget user steps to robot footstep locations, allowing the robot to utilize its own dynamics for locomotion, ensuring better balance and stability. Our method anticipates user footsteps to minimize delays between when the user initiates and completes a step and when the robot does it. The step estimates are continuously adapted to converge with the measured user references. Additionally, the system autonomously adjusts the robot's steps to account for its surrounding terrain, overcoming challenges posed by environmental mismatches between the user's flat-ground setup and the robot's uneven terrain. Experimental results on the humanoid robot Nadia demonstrate the effectiveness of the proposed system.
>
---
#### [new 032] No More Blind Spots: Learning Vision-Based Omnidirectional Bipedal Locomotion for Challenging Terrain
- **分类: cs.RO; cs.AI**

- **简介: 论文提出了一种基于视觉的全向双足行走控制框架，解决动态环境中地形感知与高效运动难题。通过引入教师-学生策略和噪声增强数据训练，避免昂贵的全向深度图渲染，实现仿真到现实的快速迁移，首次展示视觉驱动的全向双足行走能力。**

- **链接: [http://arxiv.org/pdf/2508.11929v1](http://arxiv.org/pdf/2508.11929v1)**

> **作者:** Mohitvishnu S. Gadde; Pranay Dugar; Ashish Malik; Alan Fern
>
> **摘要:** Effective bipedal locomotion in dynamic environments, such as cluttered indoor spaces or uneven terrain, requires agile and adaptive movement in all directions. This necessitates omnidirectional terrain sensing and a controller capable of processing such input. We present a learning framework for vision-based omnidirectional bipedal locomotion, enabling seamless movement using depth images. A key challenge is the high computational cost of rendering omnidirectional depth images in simulation, making traditional sim-to-real reinforcement learning (RL) impractical. Our method combines a robust blind controller with a teacher policy that supervises a vision-based student policy, trained on noise-augmented terrain data to avoid rendering costs during RL and ensure robustness. We also introduce a data augmentation technique for supervised student training, accelerating training by up to 10 times compared to conventional methods. Our framework is validated through simulation and real-world tests, demonstrating effective omnidirectional locomotion with minimal reliance on expensive rendering. This is, to the best of our knowledge, the first demonstration of vision-based omnidirectional bipedal locomotion, showcasing its adaptability to diverse terrains.
>
---
#### [new 033] Grounding Actions in Camera Space: Observation-Centric Vision-Language-Action Policy
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出OC-VLA框架，解决VLA模型在真实环境中因观察与动作空间不一致导致的泛化问题。通过将动作预测锚定在相机空间，实现跨视角鲁棒性提升，兼容现有架构且无需修改。**

- **链接: [http://arxiv.org/pdf/2508.13103v1](http://arxiv.org/pdf/2508.13103v1)**

> **作者:** Tianyi Zhang; Haonan Duan; Haoran Hao; Yu Qiao; Jifeng Dai; Zhi Hou
>
> **摘要:** Vision-Language-Action (VLA) models frequently encounter challenges in generalizing to real-world environments due to inherent discrepancies between observation and action spaces. Although training data are collected from diverse camera perspectives, the models typically predict end-effector poses within the robot base coordinate frame, resulting in spatial inconsistencies. To mitigate this limitation, we introduce the Observation-Centric VLA (OC-VLA) framework, which grounds action predictions directly in the camera observation space. Leveraging the camera's extrinsic calibration matrix, OC-VLA transforms end-effector poses from the robot base coordinate system into the camera coordinate system, thereby unifying prediction targets across heterogeneous viewpoints. This lightweight, plug-and-play strategy ensures robust alignment between perception and action, substantially improving model resilience to camera viewpoint variations. The proposed approach is readily compatible with existing VLA architectures, requiring no substantial modifications. Comprehensive evaluations on both simulated and real-world robotic manipulation tasks demonstrate that OC-VLA accelerates convergence, enhances task success rates, and improves cross-view generalization. The code will be publicly available.
>
---
#### [new 034] Belief-Conditioned One-Step Diffusion: Real-Time Trajectory Planning with Just-Enough Sensing
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **简介: 论文提出Belief-Conditioned One-Step Diffusion（B-COD），解决机器人在部分可观测环境中如何最小化传感器使用同时保证定位精度的问题。通过扩散模型条件化于位姿信念和传感器掩码，实现快速轨迹规划与局部误差代理，支持在线传感器选择，显著降低能耗并保持任务性能。**

- **链接: [http://arxiv.org/pdf/2508.12166v1](http://arxiv.org/pdf/2508.12166v1)**

> **作者:** Gokul Puthumanaillam; Aditya Penumarti; Manav Vora; Paulo Padrao; Jose Fuentes; Leonardo Bobadilla; Jane Shin; Melkior Ornik
>
> **备注:** Accepted to CoRL 2025 (Conference on Robot Learning)
>
> **摘要:** Robots equipped with rich sensor suites can localize reliably in partially-observable environments, but powering every sensor continuously is wasteful and often infeasible. Belief-space planners address this by propagating pose-belief covariance through analytic models and switching sensors heuristically--a brittle, runtime-expensive approach. Data-driven approaches--including diffusion models--learn multi-modal trajectories from demonstrations, but presuppose an accurate, always-on state estimate. We address the largely open problem: for a given task in a mapped environment, which \textit{minimal sensor subset} must be active at each location to maintain state uncertainty \textit{just low enough} to complete the task? Our key insight is that when a diffusion planner is explicitly conditioned on a pose-belief raster and a sensor mask, the spread of its denoising trajectories yields a calibrated, differentiable proxy for the expected localisation error. Building on this insight, we present Belief-Conditioned One-Step Diffusion (B-COD), the first planner that, in a 10 ms forward pass, returns a short-horizon trajectory, per-waypoint aleatoric variances, and a proxy for localisation error--eliminating external covariance rollouts. We show that this single proxy suffices for a soft-actor-critic to choose sensors online, optimising energy while bounding pose-covariance growth. We deploy B-COD in real-time marine trials on an unmanned surface vehicle and show that it reduces sensing energy consumption while matching the goal-reach performance of an always-on baseline.
>
---
#### [new 035] MCTR: Midpoint Corrected Triangulation for Autonomous Racing via Digital Twin Simulation in CARLA
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出MCTR算法，用于自主赛车中的路径规划任务，解决DTR算法路径不平滑及仿真环境缺乏3D LiDAR支持的问题。通过曲率校正平均法提升轨迹平滑性，并在CARLA数字孪生环境中验证算法有效性。**

- **链接: [http://arxiv.org/pdf/2508.12729v1](http://arxiv.org/pdf/2508.12729v1)**

> **作者:** Junhao Ye; Cheng Hu; Yiqin Wang; Weizhan Huang; Nicolas Baumann; Jie He; Meixun Qu; Lei Xie; Hongye Su
>
> **摘要:** In autonomous racing, reactive controllers eliminate the computational burden of the full See-Think-Act autonomy stack by directly mapping sensor inputs to control actions. This bypasses the need for explicit localization and trajectory planning. A widely adopted baseline in this category is the Follow-The-Gap method, which performs trajectory planning using LiDAR data. Building on FTG, the Delaunay Triangulation-based Racing algorithm introduces further enhancements. However, DTR's use of circumcircles for trajectory generation often results in insufficiently smooth paths, ultimately degrading performance. Additionally, the commonly used F1TENTH-simulator for autonomous racing competitions lacks support for 3D LiDAR perception, limiting its effectiveness in realistic testing. To address these challenges, this work proposes the MCTR algorithm. MCTR improves trajectory smoothness through the use of Curvature Corrected Moving Average and implements a digital twin system within the CARLA simulator to validate the algorithm's robustness under 3D LiDAR perception. The proposed algorithm has been thoroughly validated through both simulation and real-world vehicle experiments.
>
---
#### [new 036] Robot Trains Robot: Automatic Real-World Policy Adaptation and Learning for Humanoids
- **分类: cs.RO**

- **简介: 论文提出RTR框架，让机械臂指导人形机器人在真实世界中自主学习与适应。解决仿真到现实的迁移难题，通过教师引导实现安全、高效的强化学习，验证了行走速度控制和从零开始学swing-up任务的有效性。**

- **链接: [http://arxiv.org/pdf/2508.12252v1](http://arxiv.org/pdf/2508.12252v1)**

> **作者:** Kaizhe Hu; Haochen Shi; Yao He; Weizhuo Wang; C. Karen Liu; Shuran Song
>
> **备注:** Accepted to The Conference on Robot Learning (CoRL) 2025
>
> **摘要:** Simulation-based reinforcement learning (RL) has significantly advanced humanoid locomotion tasks, yet direct real-world RL from scratch or adapting from pretrained policies remains rare, limiting the full potential of humanoid robots. Real-world learning, despite being crucial for overcoming the sim-to-real gap, faces substantial challenges related to safety, reward design, and learning efficiency. To address these limitations, we propose Robot-Trains-Robot (RTR), a novel framework where a robotic arm teacher actively supports and guides a humanoid robot student. The RTR system provides protection, learning schedule, reward, perturbation, failure detection, and automatic resets. It enables efficient long-term real-world humanoid training with minimal human intervention. Furthermore, we propose a novel RL pipeline that facilitates and stabilizes sim-to-real transfer by optimizing a single dynamics-encoded latent variable in the real world. We validate our method through two challenging real-world humanoid tasks: fine-tuning a walking policy for precise speed tracking and learning a humanoid swing-up task from scratch, illustrating the promising capabilities of real-world humanoid learning realized by RTR-style systems. See https://robot-trains-robot.github.io/ for more info.
>
---
#### [new 037] Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决传统方法在非结构化环境中难以扩展和泛化的问题。作者系统梳理了基于大视觉语言模型的视觉-语言-动作模型，提出分类体系并总结关键技术与未来方向。**

- **链接: [http://arxiv.org/pdf/2508.13073v1](http://arxiv.org/pdf/2508.13073v1)**

> **作者:** Rui Shao; Wei Li; Lingsen Zhang; Renshan Zhang; Zhiyang Liu; Ran Chen; Liqiang Nie
>
> **备注:** Project Page: https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation
>
> **摘要:** Robotic manipulation, a key frontier in robotics and embodied AI, requires precise motor control and multimodal understanding, yet traditional rule-based methods fail to scale or generalize in unstructured, novel environments. In recent years, Vision-Language-Action (VLA) models, built upon Large Vision-Language Models (VLMs) pretrained on vast image-text datasets, have emerged as a transformative paradigm. This survey provides the first systematic, taxonomy-oriented review of large VLM-based VLA models for robotic manipulation. We begin by clearly defining large VLM-based VLA models and delineating two principal architectural paradigms: (1) monolithic models, encompassing single-system and dual-system designs with differing levels of integration; and (2) hierarchical models, which explicitly decouple planning from execution via interpretable intermediate representations. Building on this foundation, we present an in-depth examination of large VLM-based VLA models: (1) integration with advanced domains, including reinforcement learning, training-free optimization, learning from human videos, and world model integration; (2) synthesis of distinctive characteristics, consolidating architectural traits, operational strengths, and the datasets and benchmarks that support their development; (3) identification of promising directions, including memory mechanisms, 4D perception, efficient adaptation, multi-agent cooperation, and other emerging capabilities. This survey consolidates recent advances to resolve inconsistencies in existing taxonomies, mitigate research fragmentation, and fill a critical gap through the systematic integration of studies at the intersection of large VLMs and robotic manipulation. We provide a regularly updated project page to document ongoing progress: https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation.
>
---
#### [new 038] Data Shift of Object Detection in Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文研究自动驾驶中目标检测的数据分布偏移问题，提出结合CycleGAN数据增强与YOLOv5的优化模型，在BDD100K数据集上提升检测性能。**

- **链接: [http://arxiv.org/pdf/2508.11868v1](http://arxiv.org/pdf/2508.11868v1)**

> **作者:** Lida Xu
>
> **摘要:** With the widespread adoption of machine learning technologies in autonomous driving systems, their role in addressing complex environmental perception challenges has become increasingly crucial. However, existing machine learning models exhibit significant vulnerability, as their performance critically depends on the fundamental assumption that training and testing data satisfy the independent and identically distributed condition, which is difficult to guarantee in real-world applications. Dynamic variations in data distribution caused by seasonal changes, weather fluctuations lead to data shift problems in autonomous driving systems. This study investigates the data shift problem in autonomous driving object detection tasks, systematically analyzing its complexity and diverse manifestations. We conduct a comprehensive review of data shift detection methods and employ shift detection analysis techniques to perform dataset categorization and balancing. Building upon this foundation, we construct an object detection model. To validate our approach, we optimize the model by integrating CycleGAN-based data augmentation techniques with the YOLOv5 framework. Experimental results demonstrate that our method achieves superior performance compared to baseline models on the BDD100K dataset.
>
---
#### [new 039] Bioinspired underwater soft robots: from biology to robotics and back
- **分类: cs.RO**

- **简介: 论文提出双向融合的软体机器人框架，将生物原理用于机器人设计，并用机器人验证生物学假设，解决传统单向仿生局限。工作包括构建可测试演化假说的软机器人系统，推动海洋探索与科学发现。**

- **链接: [http://arxiv.org/pdf/2508.11883v1](http://arxiv.org/pdf/2508.11883v1)**

> **作者:** Lei Li; Boyang Qin; Wenzhuo Gao; Yanyu Li; Yiyuan Zhang; Bo Wang; Shihan Kong; Jian Wang; Dekui He; Junzhi Yu
>
> **摘要:** The ocean vast unexplored regions and diverse soft-bodied marine organisms have spurred interest in bio-inspired underwater soft robotics. Recent advances have enabled new capabilities in underwater movement, sensing, and interaction. However, these efforts are largely unidirectional, with biology guiding robotics while insights from robotics rarely feed back into biology. Here we propose a holistic, bidirectional framework that integrates biological principles, robotic implementation, and biological validation. We show that soft robots can serve as experimental tools to probe biological functions and even test evolutionary hypotheses. Their inherent compliance also allows them to outperform rigid systems in unstructured environments, supporting applications in marine exploration, manipulation, and medicine. Looking forward, we introduce bio-universal-inspired robotics, a paradigm that transcends species-specific mimicry by identifying convergent principles across species to inspire more adaptable designs. Despite rapid progress, challenges persist in material robustness, actuation efficiency, autonomy, and intelligence. By uniting biology and engineering, soft robots can advance ocean exploration and deepen scientific discovery.
>
---
#### [new 040] OmniD: Generalizable Robot Manipulation Policy via Image-Based BEV Representation
- **分类: cs.RO**

- **简介: 论文提出OmniD，一种基于图像BEV表示的通用机器人操作策略，解决视觉策略过拟合和多视角信息融合难的问题。通过可变形注意力机制提取任务相关特征，提升分布内、分布外及少样本场景下的泛化性能。**

- **链接: [http://arxiv.org/pdf/2508.11898v1](http://arxiv.org/pdf/2508.11898v1)**

> **作者:** Jilei Mao; Jiarui Guan; Yingjuan Tang; Qirui Hu; Zhihang Li; Junjie Yu; Yongjie Mao; Yunzhe Sun; Shuang Liu; Xiaozhu Ju
>
> **摘要:** The visuomotor policy can easily overfit to its training datasets, such as fixed camera positions and backgrounds. This overfitting makes the policy perform well in the in-distribution scenarios but underperform in the out-of-distribution generalization. Additionally, the existing methods also have difficulty fusing multi-view information to generate an effective 3D representation. To tackle these issues, we propose Omni-Vision Diffusion Policy (OmniD), a multi-view fusion framework that synthesizes image observations into a unified bird's-eye view (BEV) representation. We introduce a deformable attention-based Omni-Feature Generator (OFG) to selectively abstract task-relevant features while suppressing view-specific noise and background distractions. OmniD achieves 11\%, 17\%, and 84\% average improvement over the best baseline model for in-distribution, out-of-distribution, and few-shot experiments, respectively. Training code and simulation benchmark are available: https://github.com/1mather/omnid.git
>
---
#### [new 041] Scaling Whole-body Multi-contact Manipulation with Contact Optimization
- **分类: cs.RO**

- **简介: 论文研究 humanoid 机器人全身体接触操作任务，解决现有方法因离散采样难以扩展的问题。提出基于闭式计算的表面表示与成本设计，实现高效规划，提升77%效率，并在真实机器人上验证。**

- **链接: [http://arxiv.org/pdf/2508.12980v1](http://arxiv.org/pdf/2508.12980v1)**

> **作者:** Victor Levé; João Moura; Sachiya Fujita; Tamon Miyake; Steve Tonneau; Sethu Vijayakumar
>
> **备注:** This work has been accepted for publication in IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids 2025). Copyrights to IEEE
>
> **摘要:** Daily tasks require us to use our whole body to manipulate objects, for instance when our hands are unavailable. We consider the issue of providing humanoid robots with the ability to autonomously perform similar whole-body manipulation tasks. In this context, the infinite possibilities for where and how contact can occur on the robot and object surfaces hinder the scalability of existing planning methods, which predominantly rely on discrete sampling. Given the continuous nature of contact surfaces, gradient-based optimization offers a more suitable approach for finding solutions. However, a key remaining challenge is the lack of an efficient representation of robot surfaces. In this work, we propose (i) a representation of robot and object surfaces that enables closed-form computation of proximity points, and (ii) a cost design that effectively guides whole-body manipulation planning. Our experiments demonstrate that the proposed framework can solve problems unaddressed by existing methods, and achieves a 77% improvement in planning time over the state of the art. We also validate the suitability of our approach on real hardware through the whole-body manipulation of boxes by a humanoid robot.
>
---
#### [new 042] Talk Less, Fly Lighter: Autonomous Semantic Compression for UAV Swarm Communication via LLMs
- **分类: cs.RO**

- **简介: 论文研究无人机 swarm 在带宽受限下的语义压缩通信问题，提出基于大语言模型的自主语义压缩方法，通过构建仿真场景和通信执行流程，验证了其在多跳链路下高效协作的可行性。**

- **链接: [http://arxiv.org/pdf/2508.12043v1](http://arxiv.org/pdf/2508.12043v1)**

> **作者:** Fei Lin; Tengchao Zhang; Qinghua Ni; Jun Huang; Siji Ma; Yonglin Tian; Yisheng Lv; Naiqi Wu
>
> **摘要:** The rapid adoption of Large Language Models (LLMs) in unmanned systems has significantly enhanced the semantic understanding and autonomous task execution capabilities of Unmanned Aerial Vehicle (UAV) swarms. However, limited communication bandwidth and the need for high-frequency interactions pose severe challenges to semantic information transmission within the swarm. This paper explores the feasibility of LLM-driven UAV swarms for autonomous semantic compression communication, aiming to reduce communication load while preserving critical task semantics. To this end, we construct four types of 2D simulation scenarios with different levels of environmental complexity and design a communication-execution pipeline that integrates system prompts with task instruction prompts. On this basis, we systematically evaluate the semantic compression performance of nine mainstream LLMs in different scenarios and analyze their adaptability and stability through ablation studies on environmental complexity and swarm size. Experimental results demonstrate that LLM-based UAV swarms have the potential to achieve efficient collaborative communication under bandwidth-constrained and multi-hop link conditions.
>
---
#### [new 043] Implementation and evaluation of a prediction algorithm for an autonomous vehicle
- **分类: cs.RO**

- **简介: 论文提出并评估了一种自动驾驶车辆轨迹预测算法，解决高精度实时轨迹估计问题。通过对比动力学与运动学模型，引入新型光学测量法确定轮胎参数，并基于扩展卡尔曼滤波实现高效预测，精度提升达82.6%。**

- **链接: [http://arxiv.org/pdf/2508.12312v1](http://arxiv.org/pdf/2508.12312v1)**

> **作者:** Marco Leon Rapp
>
> **备注:** 7 pages, 7 figures
>
> **摘要:** This paper presents a prediction algorithm that estimates the vehicle trajectory every five milliseconds for an autonomous vehicle. A kinematic and a dynamic bicycle model are compared, with the dynamic model exhibiting superior accuracy at higher speeds. Vehicle parameters such as mass, center of gravity, moment of inertia, and cornering stiffness are determined experimentally. For cornering stiffness, a novel measurement procedure using optical position tracking is introduced. The model is incorporated into an extended Kalman filter and implemented in a ROS node in C++. The algorithm achieves a positional deviation of only 1.25 cm per meter over the entire test drive and is up to 82.6% more precise than the kinematic model.
>
---
#### [new 044] Toward General Physical Intelligence for Resilient Agile Manufacturing Automation
- **分类: cs.RO**

- **简介: 论文探讨通用物理智能（GPI）在敏捷制造自动化中的应用，旨在解决机器人在非结构化环境中安全交互与情境推理问题。作者系统回顾VLA模型进展，分析五大主题，评估工业部署潜力，并提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2508.11960v1](http://arxiv.org/pdf/2508.11960v1)**

> **作者:** Sandeep Kanta; Mehrdad Tavassoli; Varun Teja Chirkuri; Venkata Akhil Kumar; Santhi Bharath Punati; Praveen Damacharla; Sunny Katyara
>
> **备注:** Advanced Engineering Informatics
>
> **摘要:** Agile and human-centric manufacturing stipulates resilient robotic solutions capable of contextual reasoning and safe interaction in unstructured environments. Foundation models particularly the Vision Language Action (VLA) models have emerged to fuse multimodal perception, reasoning and physically grounded action across varied embodiments into unified representation, termed as General Physical Intelligence (GPI). While GPI has already been described in the literature but its practical application and evolving role in contemporary agile manufacturing processes have yet to be duly explored. To bridge this gap, this practical review systematically surveys recent advancements in VLA models within GPI context, performs comprehensive comparative analysis of leading implementations and evaluates their readiness for industrial deployment through structured ablation study. Our analysis has organized state-of-the-art into five thematic pillars including multisensory representation learning, sim2real transfer, planning and control, uncertainty and safety measures and benchmarking. Finally, we articulate open research challenges and propose directions to better integrate GPI into next-generation industrial ecosystems in line with Industry 5.0.
>
---
#### [new 045] From Screen to Stage: Kid Cosmo, A Life-Like, Torque-Controlled Humanoid for Entertainment Robotics
- **分类: cs.RO**

- **简介: 论文提出Kid Cosmo，一个类人娱乐机器人，解决娱乐场景中机器人动作自然性与功能性难以兼顾的问题。通过28自由度扭矩控制设计，实现拟人运动与角色形象融合，验证了表演导向人形机器人的可行性。**

- **链接: [http://arxiv.org/pdf/2508.11884v1](http://arxiv.org/pdf/2508.11884v1)**

> **作者:** Havel Liu; Mingzhang Zhu; Arturo Moises Flores Alvarez; Yuan Hung Lo; Conrad Ku; Federico Parres; Justin Quan; Colin Togashi; Aditya Navghare; Quanyou Wang; Dennis W. Hong
>
> **备注:** 8 pages, 14 figures, accepted by IEEE Humanoids 2025
>
> **摘要:** Humanoid robots represent the cutting edge of robotics research, yet their potential in entertainment remains largely unexplored. Entertainment as a field prioritizes visuals and form, a principle that contrasts with the purely functional designs of most contemporary humanoid robots. Designing entertainment humanoid robots capable of fluid movement presents a number of unique challenges. In this paper, we present Kid Cosmo, a research platform designed for robust locomotion and life-like motion generation while imitating the look and mannerisms of its namesake character from Netflix's movie The Electric State. Kid Cosmo is a child-sized humanoid robot, standing 1.45 m tall and weighing 25 kg. It contains 28 degrees of freedom and primarily uses proprioceptive actuators, enabling torque-control walking and lifelike motion generation. Following worldwide showcases as part of the movie's press tour, we present the system architecture, challenges of a functional entertainment robot and unique solutions, and our initial findings on stability during simultaneous upper and lower body movement. We demonstrate the viability of performance-oriented humanoid robots that prioritize both character embodiment and technical functionality.
>
---
#### [new 046] BOW: Bayesian Optimization over Windows for Motion Planning in Complex Environments
- **分类: cs.RO**

- **简介: 论文提出BOW Planner，用于复杂环境中的机器人运动规划任务，解决传统方法在处理速度、加速度等动力学约束时效率低的问题。通过基于约束贝叶斯优化的窗口采样策略，实现高效、安全、快速的轨迹生成。**

- **链接: [http://arxiv.org/pdf/2508.13052v1](http://arxiv.org/pdf/2508.13052v1)**

> **作者:** Sourav Raxit; Abdullah Al Redwan Newaz; Paulo Padrao; Jose Fuentes; Leonardo Bobadilla
>
> **摘要:** This paper introduces the BOW Planner, a scalable motion planning algorithm designed to navigate robots through complex environments using constrained Bayesian optimization (CBO). Unlike traditional methods, which often struggle with kinodynamic constraints such as velocity and acceleration limits, the BOW Planner excels by concentrating on a planning window of reachable velocities and employing CBO to sample control inputs efficiently. This approach enables the planner to manage high-dimensional objective functions and stringent safety constraints with minimal sampling, ensuring rapid and secure trajectory generation. Theoretical analysis confirms the algorithm's asymptotic convergence to near-optimal solutions, while extensive evaluations in cluttered and constrained settings reveal substantial improvements in computation times, trajectory lengths, and solution times compared to existing techniques. Successfully deployed across various real-world robotic systems, the BOW Planner demonstrates its practical significance through exceptional sample efficiency, safety-aware optimization, and rapid planning capabilities, making it a valuable tool for advancing robotic applications. The BOW Planner is released as an open-source package and videos of real-world and simulated experiments are available at https://bow-web.github.io.
>
---
#### [new 047] Simultaneous Contact Sequence and Patch Planning for Dynamic Locomotion
- **分类: cs.RO**

- **简介: 论文提出基于蒙特卡洛树搜索与全身体轨迹优化的联合规划方法，解决腿式机器人在复杂环境中同时规划接触序列与接触面的选择问题，实现动态行走和复杂非循环多接触动作的高效生成与实机验证。**

- **链接: [http://arxiv.org/pdf/2508.12928v1](http://arxiv.org/pdf/2508.12928v1)**

> **作者:** Victor Dhédin; Haizhou Zhao; Majid Khadiv
>
> **摘要:** Legged robots have the potential to traverse highly constrained environments with agile maneuvers. However, planning such motions requires solving a highly challenging optimization problem with a mixture of continuous and discrete decision variables. In this paper, we present a full pipeline based on Monte-Carlo tree search (MCTS) and whole-body trajectory optimization (TO) to perform simultaneous contact sequence and patch selection on highly challenging environments. Through extensive simulation experiments, we show that our framework can quickly find a diverse set of dynamically consistent plans. We experimentally show that these plans are transferable to a real quadruped robot. We further show that the same framework can find highly complex acyclic humanoid maneuvers. To the best of our knowledge, this is the first demonstration of simultaneous contact sequence and patch selection for acyclic multi-contact locomotion using the whole-body dynamics of a quadruped.
>
---
#### [new 048] Has GPT-5 Achieved Spatial Intelligence? An Empirical Study
- **分类: cs.CV; cs.CL; cs.LG; cs.MM; cs.RO**

- **简介: 该论文研究多模态模型的空间智能，旨在评估GPT-5在空间理解与推理上的能力。通过构建任务分类体系并测试8个基准，发现GPT-5虽强但仍未达人类水平，且高端模型在难题上无明显优势。**

- **链接: [http://arxiv.org/pdf/2508.13142v1](http://arxiv.org/pdf/2508.13142v1)**

> **作者:** Zhongang Cai; Yubo Wang; Qingping Sun; Ruisi Wang; Chenyang Gu; Wanqi Yin; Zhiqian Lin; Zhitao Yang; Chen Wei; Xuanke Shi; Kewang Deng; Xiaoyang Han; Zukai Chen; Jiaqi Li; Xiangyu Fan; Hanming Deng; Lewei Lu; Bo Li; Ziwei Liu; Quan Wang; Dahua Lin; Lei Yang
>
> **摘要:** Multi-modal models have achieved remarkable progress in recent years. Nevertheless, they continue to exhibit notable limitations in spatial understanding and reasoning, which are fundamental capabilities to achieving artificial general intelligence. With the recent release of GPT-5, allegedly the most powerful AI model to date, it is timely to examine where the leading models stand on the path toward spatial intelligence. First, we propose a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and discuss the challenges in ensuring fair evaluation. We then evaluate state-of-the-art proprietary and open-source models on eight key benchmarks, at a cost exceeding one billion total tokens. Our empirical study reveals that (1) GPT-5 demonstrates unprecedented strength in spatial intelligence, yet (2) still falls short of human performance across a broad spectrum of tasks. Moreover, we (3) identify the more challenging spatial intelligence problems for multi-modal models, and (4) proprietary models do not exhibit a decisive advantage when facing the most difficult problems. In addition, we conduct a qualitative evaluation across a diverse set of scenarios that are intuitive for humans yet fail even the most advanced multi-modal models.
>
---
#### [new 049] Precise Action-to-Video Generation Through Visual Action Prompts
- **分类: cs.CV; cs.RO**

- **简介: 论文提出视觉动作提示（VAP），用于动作到视频生成任务，解决精度与跨域迁移之间的权衡问题。通过将动作渲染为通用的视觉骨架提示，实现复杂交互的精准控制并保持跨域动态一致性。**

- **链接: [http://arxiv.org/pdf/2508.13104v1](http://arxiv.org/pdf/2508.13104v1)**

> **作者:** Yuang Wang; Chao Wen; Haoyu Guo; Sida Peng; Minghan Qin; Hujun Bao; Xiaowei Zhou; Ruizhen Hu
>
> **备注:** Accepted to ICCV 2025. Project page: https://zju3dv.github.io/VAP/
>
> **摘要:** We present visual action prompts, a unified action representation for action-to-video generation of complex high-DoF interactions while maintaining transferable visual dynamics across domains. Action-driven video generation faces a precision-generality trade-off: existing methods using text, primitive actions, or coarse masks offer generality but lack precision, while agent-centric action signals provide precision at the cost of cross-domain transferability. To balance action precision and dynamic transferability, we propose to "render" actions into precise visual prompts as domain-agnostic representations that preserve both geometric precision and cross-domain adaptability for complex actions; specifically, we choose visual skeletons for their generality and accessibility. We propose robust pipelines to construct skeletons from two interaction-rich data sources - human-object interactions (HOI) and dexterous robotic manipulation - enabling cross-domain training of action-driven generative models. By integrating visual skeletons into pretrained video generation models via lightweight fine-tuning, we enable precise action control of complex interaction while preserving the learning of cross-domain dynamics. Experiments on EgoVid, RT-1 and DROID demonstrate the effectiveness of our proposed approach. Project page: https://zju3dv.github.io/VAP/.
>
---
#### [new 050] Control of a commercial vehicle by a tetraplegic human using a bimanual brain-computer interface
- **分类: eess.SY; cs.NE; cs.RO; cs.SY**

- **简介: 论文提出一种双侧脑机接口系统，使高位截瘫患者通过脑信号控制车辆行驶与制动。解决了BCI在真实场景中应用有限的问题，实现了远程驾驶和安全操控，验证了其可行性和潜力。**

- **链接: [http://arxiv.org/pdf/2508.11805v1](http://arxiv.org/pdf/2508.11805v1)**

> **作者:** Xinyun Zou; Jorge Gamez; Meghna Menon; Phillip Ring; Chadwick Boulay; Likhith Chitneni; Jackson Brennecke; Shana R. Melby; Gracy Kureel; Kelsie Pejsa; Emily R. Rosario; Ausaf A. Bari; Aniruddh Ravindran; Tyson Aflalo; Spencer S. Kellis; Dimitar Filev; Florian Solzbacher; Richard A. Andersen
>
> **备注:** 41 pages, 7 figures, 1 table. 22 supplementary pages, 6 supplementary figures, 11 supplementary tables, 9 supplementary movies available as ancillary files
>
> **摘要:** Brain-computer interfaces (BCIs) read neural signals directly from the brain to infer motor planning and execution. However, the implementation of this technology has been largely limited to laboratory settings, with few real-world applications. We developed a bimanual BCI system to drive a vehicle in both simulated and real-world environments. We demonstrate that an individual with tetraplegia, implanted with intracortical BCI electrodes in the posterior parietal cortex (PPC) and the hand knob region of the motor cortex (MC), reacts at least as fast and precisely as motor intact participants, and drives a simulated vehicle as proficiently as the same control group. This BCI participant, living in California, could also remotely drive a Ford Mustang Mach-E vehicle in Michigan. Our first teledriving task relied on cursor control for speed and steering in a closed urban test facility. However, the final BCI system added click control for full-stop braking and thus enabled bimanual cursor-and-click control for both simulated driving through a virtual town with traffic and teledriving through an obstacle course without traffic in the real world. We also demonstrate the safety and feasibility of BCI-controlled driving. This first-of-its-kind implantable BCI application not only highlights the versatility and innovative potentials of BCIs but also illuminates the promising future for the development of life-changing solutions to restore independence to those who suffer catastrophic neurological injury.
>
---
#### [new 051] On the complexity of constrained reconfiguration and motion planning
- **分类: cs.CC; cs.DM; cs.DS; cs.RO; math.CO**

- **简介: 论文研究多智能体在约束环境中的运动规划与重配置问题，提出k-兼容排序模型，证明其NP完全性，并给出特定情况下的多项式时间算法，适用于机器人臂旋转、调度等场景。**

- **链接: [http://arxiv.org/pdf/2508.13032v1](http://arxiv.org/pdf/2508.13032v1)**

> **作者:** Nicolas Bousquet; Remy El Sabeh; Amer E. Mouawad; Naomi Nishimura
>
> **摘要:** Coordinating the motion of multiple agents in constrained environments is a fundamental challenge in robotics, motion planning, and scheduling. A motivating example involves $n$ robotic arms, each represented as a line segment. The objective is to rotate each arm to its vertical orientation, one at a time (clockwise or counterclockwise), without collisions nor rotating any arm more than once. This scenario is an example of the more general $k$-Compatible Ordering problem, where $n$ agents, each capable of $k$ state-changing actions, must transition to specific target states under constraints encoded as a set $\mathcal{G}$ of $k$ pairs of directed graphs. We show that $k$-Compatible Ordering is $\mathsf{NP}$-complete, even when $\mathcal{G}$ is planar, degenerate, or acyclic. On the positive side, we provide polynomial-time algorithms for cases such as when $k = 1$ or $\mathcal{G}$ has bounded treewidth. We also introduce generalized variants supporting multiple state-changing actions per agent, broadening the applicability of our framework. These results extend to a wide range of scheduling, reconfiguration, and motion planning applications in constrained environments.
>
---
#### [new 052] Adjustable AprilTags For Identity Secured Tasks
- **分类: cs.CR; cs.RO**

- **简介: 论文提出使用可调整的AprilTags解决开放环境中身份安全问题，防止对抗攻击。属于计算机视觉中的身份识别与安全防护任务。**

- **链接: [http://arxiv.org/pdf/2508.12304v1](http://arxiv.org/pdf/2508.12304v1)**

> **作者:** Hao Li
>
> **摘要:** Special tags such as AprilTags that facilitate image processing and pattern recognition are useful in practical applications. In close and private environments, identity security is unlikely to be an issue because all involved AprilTags can be completely regulated. However, in open and public environments, identity security is no longer an issue that can be neglected. To handle potential harm caused by adversarial attacks, this note advocates utilization of adjustable AprilTags instead of fixed ones.
>
---
#### [new 053] Recent Advances in Transformer and Large Language Models for UAV Applications
- **分类: cs.CV; cs.AI; cs.RO; cs.SY; eess.IV; eess.SY**

- **简介: 论文综述基于Transformer和大语言模型在无人机领域的应用，解决感知、决策与自主性提升问题。工作包括分类模型、分析性能、梳理数据集与挑战，并提出未来方向。**

- **链接: [http://arxiv.org/pdf/2508.11834v1](http://arxiv.org/pdf/2508.11834v1)**

> **作者:** Hamza Kheddar; Yassine Habchi; Mohamed Chahine Ghanem; Mustapha Hemis; Dusit Niyato
>
> **摘要:** The rapid advancement of Transformer-based models has reshaped the landscape of uncrewed aerial vehicle (UAV) systems by enhancing perception, decision-making, and autonomy. This review paper systematically categorizes and evaluates recent developments in Transformer architectures applied to UAVs, including attention mechanisms, CNN-Transformer hybrids, reinforcement learning Transformers, and large language models (LLMs). Unlike previous surveys, this work presents a unified taxonomy of Transformer-based UAV models, highlights emerging applications such as precision agriculture and autonomous navigation, and provides comparative analyses through structured tables and performance benchmarks. The paper also reviews key datasets, simulators, and evaluation metrics used in the field. Furthermore, it identifies existing gaps in the literature, outlines critical challenges in computational efficiency and real-time deployment, and offers future research directions. This comprehensive synthesis aims to guide researchers and practitioners in understanding and advancing Transformer-driven UAV technologies.
>
---
#### [new 054] Lifelong Learner: Discovering Versatile Neural Solvers for Vehicle Routing Problems
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 论文提出终身学习框架LL，用于解决不同场景下的车辆路径问题（VRP）。针对现有神经解法在单一场景下训练导致泛化能力差的问题，LL通过Transformer结构和跨情境注意力机制，逐步学习多种VRP情境，并利用动态调度器增强知识迁移，显著提升解的质量与适用性。**

- **链接: [http://arxiv.org/pdf/2508.11679v1](http://arxiv.org/pdf/2508.11679v1)**

> **作者:** Shaodi Feng; Zhuoyi Lin; Jianan Zhou; Cong Zhang; Jingwen Li; Kuan-Wen Chen; Senthilnath Jayavelu; Yew-Soon Ong
>
> **摘要:** Deep learning has been extensively explored to solve vehicle routing problems (VRPs), which yields a range of data-driven neural solvers with promising outcomes. However, most neural solvers are trained to tackle VRP instances in a relatively monotonous context, e.g., simplifying VRPs by using Euclidean distance between nodes and adhering to a single problem size, which harms their off-the-shelf application in different scenarios. To enhance their versatility, this paper presents a novel lifelong learning framework that incrementally trains a neural solver to manage VRPs in distinct contexts. Specifically, we propose a lifelong learner (LL), exploiting a Transformer network as the backbone, to solve a series of VRPs. The inter-context self-attention mechanism is proposed within LL to transfer the knowledge obtained from solving preceding VRPs into the succeeding ones. On top of that, we develop a dynamic context scheduler (DCS), employing the cross-context experience replay to further facilitate LL looking back on the attained policies of solving preceding VRPs. Extensive results on synthetic and benchmark instances (problem sizes up to 18k) show that our LL is capable of discovering effective policies for tackling generic VRPs in varying contexts, which outperforms other neural solvers and achieves the best performance for most VRPs.
>
---
#### [new 055] Scaling Robust Optimization for Swarms: A Distributed Perspective
- **分类: math.OC; cs.RO**

- **简介: 论文提出一种分布式鲁棒优化框架，用于多智能体系统在不确定环境下的安全控制。针对传统方法难以处理确定性或缺乏概率数据的不确定性问题，引入鲁棒约束和机会约束，并通过ADMM实现高效分布式求解，显著提升计算效率与可扩展性。**

- **链接: [http://arxiv.org/pdf/2508.11799v1](http://arxiv.org/pdf/2508.11799v1)**

> **作者:** Arshiya Taj Abdul; Augustinos D. Saravanos; Evangelos A. Theodorou
>
> **摘要:** This article introduces a decentralized robust optimization framework for safe multi-agent control under uncertainty. Although stochastic noise has been the primary form of modeling uncertainty in such systems, these formulations might fall short in addressing uncertainties that are deterministic in nature or simply lack probabilistic data. To ensure safety under such scenarios, we employ the concept of robust constraints that must hold for all possible uncertainty realizations lying inside a bounded set. Nevertheless, standard robust optimization approaches become intractable due to the large number or non-convexity of the constraints involved in safe multi-agent control. To address this, we introduce novel robust reformulations that significantly reduce complexity without compromising safety. The applicability of the framework is further broadened to address both deterministic and stochastic uncertainties by incorporating robust chance constraints and distribution steering techniques. To achieve scalability, we derive a distributed approach based on the Alternating Direction Method of Multipliers (ADMM), supported by a convergence study that accounts for the underlying non-convexity. In addition, computational complexity bounds highlighting the efficiency of the proposed frameworks against standard approaches are presented. Finally, the robustness and scalability of the framework is demonstrated through extensive simulation results across diverse scenarios, including environments with nonconvex obstacles and up to 246 agents.
>
---
#### [new 056] DynamicPose: Real-time and Robust 6D Object Pose Tracking for Fast-Moving Cameras and Objects
- **分类: cs.CV; cs.RO**

- **简介: 论文提出DynamicPose框架，解决快速移动相机和物体下的6D姿态跟踪问题。通过视觉惯性里程计、深度引导2D跟踪和VIO引导卡尔曼滤波实现鲁棒实时跟踪。**

- **链接: [http://arxiv.org/pdf/2508.11950v1](http://arxiv.org/pdf/2508.11950v1)**

> **作者:** Tingbang Liang; Yixin Zeng; Jiatong Xie; Boyu Zhou
>
> **摘要:** We present DynamicPose, a retraining-free 6D pose tracking framework that improves tracking robustness in fast-moving camera and object scenarios. Previous work is mainly applicable to static or quasi-static scenes, and its performance significantly deteriorates when both the object and the camera move rapidly. To overcome these challenges, we propose three synergistic components: (1) A visual-inertial odometry compensates for the shift in the Region of Interest (ROI) caused by camera motion; (2) A depth-informed 2D tracker corrects ROI deviations caused by large object translation; (3) A VIO-guided Kalman filter predicts object rotation, generates multiple candidate poses, and then obtains the final pose by hierarchical refinement. The 6D pose tracking results guide subsequent 2D tracking and Kalman filter updates, forming a closed-loop system that ensures accurate pose initialization and precise pose tracking. Simulation and real-world experiments demonstrate the effectiveness of our method, achieving real-time and robust 6D pose tracking for fast-moving cameras and objects.
>
---
## 更新

#### [replaced 001] HCOA*: Hierarchical Class-ordered A* for Navigation in Semantic Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.03128v2](http://arxiv.org/pdf/2505.03128v2)**

> **作者:** Evangelos Psomiadis; Panagiotis Tsiotras
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** This paper addresses the problem of robot navigation in mixed geometric/semantic 3D environments. Given a hierarchical representation of the environment, the objective is to navigate from a start position to a goal, while satisfying task-specific safety constraints and minimizing computational cost. We introduce Hierarchical Class-ordered A* (HCOA*), an algorithm that leverages the environment's hierarchy for efficient and safe path-planning in mixed geometric/semantic graphs. We use a total order over the semantic classes and prove theoretical performance guarantees for the algorithm. We propose three approaches for higher-layer node classification based on the semantics of the lowest layer: a Graph Neural Network method, a k-Nearest Neighbors method, and a Majority-Class method. We evaluate HCOA* in simulations on two 3D Scene Graphs, comparing it to the state-of-the-art and assessing the performance of each classification approach. Results show that HCOA* reduces the computational time of navigation by up to 50%, while maintaining near-optimal performance across a wide range of scenarios.
>
---
#### [replaced 002] HuB: Learning Extreme Humanoid Balance
- **分类: cs.RO; cs.AI; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2505.07294v2](http://arxiv.org/pdf/2505.07294v2)**

> **作者:** Tong Zhang; Boyuan Zheng; Ruiqian Nai; Yingdong Hu; Yen-Jen Wang; Geng Chen; Fanqi Lin; Jiongye Li; Chuye Hong; Koushil Sreenath; Yang Gao
>
> **备注:** CoRL 2025 (Oral Presentation). Project website: https://hub-robot.github.io
>
> **摘要:** The human body demonstrates exceptional motor capabilities-such as standing steadily on one foot or performing a high kick with the leg raised over 1.5 meters-both requiring precise balance control. While recent research on humanoid control has leveraged reinforcement learning to track human motions for skill acquisition, applying this paradigm to balance-intensive tasks remains challenging. In this work, we identify three key obstacles: instability from reference motion errors, learning difficulties due to morphological mismatch, and the sim-to-real gap caused by sensor noise and unmodeled dynamics. To address these challenges, we propose HuB (Humanoid Balance), a unified framework that integrates reference motion refinement, balance-aware policy learning, and sim-to-real robustness training, with each component targeting a specific challenge. We validate our approach on the Unitree G1 humanoid robot across challenging quasi-static balance tasks, including extreme single-legged poses such as Swallow Balance and Bruce Lee's Kick. Our policy remains stable even under strong physical disturbances-such as a forceful soccer strike-while baseline methods consistently fail to complete these tasks. Project website: https://hub-robot.github.io
>
---
#### [replaced 003] Towards Multimodal Social Conversations with Robots: Using Vision-Language Models
- **分类: cs.RO; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.19196v2](http://arxiv.org/pdf/2507.19196v2)**

> **作者:** Ruben Janssens; Tony Belpaeme
>
> **备注:** Accepted at the workshop "Human - Foundation Models Interaction: A Focus On Multimodal Information" (FoMo-HRI) at IEEE RO-MAN 2025 (Camera-ready version)
>
> **摘要:** Large language models have given social robots the ability to autonomously engage in open-domain conversations. However, they are still missing a fundamental social skill: making use of the multiple modalities that carry social interactions. While previous work has focused on task-oriented interactions that require referencing the environment or specific phenomena in social interactions such as dialogue breakdowns, we outline the overall needs of a multimodal system for social conversations with robots. We then argue that vision-language models are able to process this wide range of visual information in a sufficiently general manner for autonomous social robots. We describe how to adapt them to this setting, which technical challenges remain, and briefly discuss evaluation practices.
>
---
#### [replaced 004] Unravelling Responsibility for AI
- **分类: cs.AI; cs.CY; cs.RO**

- **链接: [http://arxiv.org/pdf/2308.02608v4](http://arxiv.org/pdf/2308.02608v4)**

> **作者:** Zoe Porter; Philippa Ryan; Phillip Morgan; Joanna Al-Qaddoumi; Bernard Twomey; Paul Noordhof; John McDermid; Ibrahim Habli
>
> **摘要:** It is widely acknowledged that we need to establish where responsibility lies for the outputs and impacts of AI-enabled systems. This is important to achieve justice and compensation for victims of AI harms, and to inform policy and engineering practice. But without a clear, thorough understanding of what "responsibility" means, deliberations about where responsibility lies will be, at best, unfocused and incomplete and, at worst, misguided. Furthermore, AI-enabled systems exist within a wider ecosystem of actors, decisions, and governance structures, giving rise to complex networks of responsibility relations. To address these issues, this paper presents a conceptual framework of responsibility, accompanied with a graphical notation and general methodology for visualising these responsibility networks and for tracing different responsibility attributions for AI. Taking the three-part formulation "Actor A is responsible for Occurrence O," the framework unravels the concept of responsibility to clarify that there are different possibilities of who is responsible for AI, senses in which they are responsible, and aspects of events they are responsible for. The notation allows these permutations to be represented graphically. The methodology enables users to apply the framework to specific scenarios. The aim is to offer a foundation to support stakeholders from diverse disciplinary backgrounds to discuss and address complex responsibility questions in hypothesised and real-world cases involving AI. The work is illustrated by application to a fictitious scenario of a fatal collision between a crewless, AI-enabled maritime vessel in autonomous mode and a traditional, crewed vessel at sea.
>
---
#### [replaced 005] FSDP: Fast and Safe Data-Driven Overtaking Trajectory Planning for Head-to-Head Autonomous Racing Competitions
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.06075v2](http://arxiv.org/pdf/2503.06075v2)**

> **作者:** Cheng Hu; Jihao Huang; Wule Mao; Yonghao Fu; Xuemin Chi; Haotong Qin; Nicolas Baumann; Zhitao Liu; Michele Magno; Lei Xie
>
> **备注:** accepted by IROS 2025
>
> **摘要:** Generating overtaking trajectories in autonomous racing is a challenging task, as the trajectory must satisfy the vehicle's dynamics and ensure safety and real-time performance running on resource-constrained hardware. This work proposes the Fast and Safe Data-Driven Planner to address this challenge. Sparse Gaussian predictions are introduced to improve both the computational efficiency and accuracy of opponent predictions. Furthermore, the proposed approach employs a bi-level quadratic programming framework to generate an overtaking trajectory leveraging the opponent predictions. The first level uses polynomial fitting to generate a rough trajectory, from which reference states and control inputs are derived for the second level. The second level formulates a model predictive control optimization problem in the Frenet frame, generating a trajectory that satisfies both kinematic feasibility and safety. Experimental results on the F1TENTH platform show that our method outperforms the State-of-the-Art, achieving an 8.93% higher overtaking success rate, allowing the maximum opponent speed, ensuring a smoother ego trajectory, and reducing 74.04% computational time compared to the Predictive Spliner method. The code is available at: https://github.com/ZJU-DDRX/FSDP.
>
---
#### [replaced 006] Embodied Long Horizon Manipulation with Closed-loop Code Generation and Incremental Few-shot Adaptation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.21969v2](http://arxiv.org/pdf/2503.21969v2)**

> **作者:** Yuan Meng; Xiangtong Yao; Haihui Ye; Yirui Zhou; Shengqiang Zhang; Zhenguo Sun; Zhenshan Bing; Alois Knoll
>
> **备注:** update ICRA 6 page
>
> **摘要:** Embodied long-horizon manipulation requires robotic systems to process multimodal inputs-such as vision and natural language-and translate them into executable actions. However, existing learning-based approaches often depend on large, task-specific datasets and struggle to generalize to unseen scenarios. Recent methods have explored using large language models (LLMs) as high-level planners that decompose tasks into subtasks using natural language and guide pretrained low-level controllers. Yet, these approaches assume perfect execution from low-level policies, which is unrealistic in real-world environments with noise or suboptimal behaviors. To overcome this, we fully discard the pretrained low-level policy and instead use the LLM to directly generate executable code plans within a closed-loop framework. Our planner employs chain-of-thought (CoT)-guided few-shot learning with incrementally structured examples to produce robust and generalizable task plans. Complementing this, a reporter evaluates outcomes using RGB-D and delivers structured feedback, enabling recovery from misalignment and replanning under partial observability. This design eliminates per-step inference, reduces computational overhead, and limits error accumulation that was observed in previous methods. Our framework achieves state-of-the-art performance on 30+ diverse seen and unseen long-horizon tasks across LoHoRavens, CALVIN, Franka Kitchen, and cluttered real-world settings.
>
---
#### [replaced 007] Novel Object 6D Pose Estimation with a Single Reference View
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.05578v2](http://arxiv.org/pdf/2503.05578v2)**

> **作者:** Jian Liu; Wei Sun; Kai Zeng; Jin Zheng; Hui Yang; Hossein Rahmani; Ajmal Mian; Lin Wang
>
> **备注:** 17 pages, 12 figures (including supplementary material)
>
> **摘要:** Existing novel object 6D pose estimation methods typically rely on CAD models or dense reference views, which are both difficult to acquire. Using only a single reference view is more scalable, but challenging due to large pose discrepancies and limited geometric and spatial information. To address these issues, we propose a Single-Reference-based novel object 6D (SinRef-6D) pose estimation method. Our key idea is to iteratively establish point-wise alignment in a common coordinate system based on state space models (SSMs). Specifically, iterative object-space point-wise alignment can effectively handle large pose discrepancies, while our proposed RGB and Points SSMs can capture long-range dependencies and spatial information from a single view, offering linear complexity and superior spatial modeling capability. Once pre-trained on synthetic data, SinRef-6D can estimate the 6D pose of a novel object using only a single reference view, without requiring retraining or a CAD model. Extensive experiments on six popular datasets and real-world robotic scenes demonstrate that we achieve on-par performance with CAD-based and dense reference view-based methods, despite operating in the more challenging single reference setting. Code will be released at https://github.com/CNJianLiu/SinRef-6D.
>
---
#### [replaced 008] FDSPC: Fast and Direct Smooth Path Planning via Continuous Curvature Integration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2405.03281v2](http://arxiv.org/pdf/2405.03281v2)**

> **作者:** Zong Chen; Yiqun Li
>
> **摘要:** In recent decades, global path planning of robot has seen significant advancements. Both heuristic search-based methods and probability sampling-based methods have shown capabilities to find feasible solutions in complex scenarios. However, mainstream global path planning algorithms often produce paths with bends, requiring additional smoothing post-processing. In this work, we propose a fast and direct path planning method based on continuous curvature integration. This method ensures path feasibility while directly generating global smooth paths with constant velocity, thus eliminating the need for post-path-smoothing. Furthermore, we compare the proposed method with existing approaches in terms of solution time, path length, memory usage, and smoothness under multiple scenarios. The proposed method is vastly superior to the average performance of state-of-the-art (SOTA) methods, especially in terms of the self-defined $\mathcal{S}_2 $ smoothness (mean angle of steering). These results demonstrate the effectiveness and superiority of our approach in several representative environments.
>
---
#### [replaced 009] Multi-agent Task-Driven Exploration via Intelligent Map Compression and Sharing
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2403.14780v2](http://arxiv.org/pdf/2403.14780v2)**

> **作者:** Evangelos Psomiadis; Dipankar Maity; Panagiotis Tsiotras
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** This paper investigates the task-driven exploration of unknown environments with mobile sensors communicating compressed measurements. The sensors explore the area and transmit their compressed data to another robot, assisting it to reach its goal location. We propose a novel communication framework and a tractable multi-agent exploration algorithm to select the sensors' actions. The algorithm uses a task-driven measure of uncertainty, resulting from map compression, as a reward function. We validate the efficacy of our algorithm through numerical simulations conducted on a realistic map and compare it with alternative approaches. The results indicate that the proposed algorithm effectively decreases the time required for the robot to reach its target without causing excessive load on the communication network.
>
---
#### [replaced 010] Solving Stochastic Orienteering Problems with Chance Constraints Using a GNN Powered Monte Carlo Tree Search
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.04653v2](http://arxiv.org/pdf/2409.04653v2)**

> **作者:** Marcos Abel Zuzuárregui; Stefano Carpin
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Leveraging the power of a graph neural network (GNN) with message passing, we present a Monte Carlo Tree Search (MCTS) method to solve stochastic orienteering problems with chance constraints. While adhering to an assigned travel budget the algorithm seeks to maximize collected reward while incurring stochastic travel costs. In this context, the acceptable probability of exceeding the assigned budget is expressed as a chance constraint. Our MCTS solution is an online and anytime algorithm alternating planning and execution that determines the next vertex to visit by continuously monitoring the remaining travel budget. The novelty of our work is that the rollout phase in the MCTS framework is implemented using a message passing GNN, predicting both the utility and failure probability of each available action. This allows to enormously expedite the search process. Our experimental evaluation shows that with the proposed method and architecture we manage to efficiently solve complex problem instances while incurring in moderate losses in terms of collected reward. Moreover, we demonstrate how the approach is capable of generalizing beyond the characteristics of the training dataset. The paper's website, open-source code, and supplementary documentation can be found at ucmercedrobotics.github.io/gnn-sop.
>
---
#### [replaced 011] Formal Verification and Control with Conformal Prediction
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2409.00536v3](http://arxiv.org/pdf/2409.00536v3)**

> **作者:** Lars Lindemann; Yiqi Zhao; Xinyi Yu; George J. Pappas; Jyotirmoy V. Deshmukh
>
> **摘要:** We present recent advances in formal verification and control for autonomous systems with practical safety guarantees enabled by conformal prediction (CP), a statistical tool for uncertainty quantification. This survey is particularly motivated by learning-enabled autonomous systems (LEASs), where the complexity of learning-enabled components (LECs) poses a major bottleneck for applying traditional model-based verification and control techniques. To address this challenge, we advocate for CP as a lightweight alternative and demonstrate its use in formal verification, systems and control, and robotics. CP is appealing due to its simplicity (easy to understand, implement, and adapt), generality (requires no assumptions on learned models and underlying data distributions), and efficiency (real-time capable and accurate). This survey provides an accessible introduction to CP for non-experts interested in applying CP to autonomy problems. We particularly show how CP can be used for formal verification of LECs and the design of safe control as well as offline and online verification algorithms for LEASs. We present these techniques within a unifying framework that addresses the complexity of LEASs. Our exposition spans simple specifications, such as robot navigation tasks, to complex mission requirements expressed in temporal logic. Throughout the survey, we contrast CP with other statistical techniques, including scenario optimization and PAC-Bayes theory, highlighting advantages and limitations for verification and control. Finally, we outline open problems and promising directions for future research.
>
---
#### [replaced 012] LGR2: Language Guided Reward Relabeling for Accelerating Hierarchical Reinforcement Learning
- **分类: cs.LG; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2406.05881v4](http://arxiv.org/pdf/2406.05881v4)**

> **作者:** Utsav Singh; Pramit Bhattacharyya; Vinay P. Namboodiri
>
> **摘要:** Large language models (LLMs) have shown remarkable abilities in logical reasoning, in-context learning, and code generation. However, translating natural language instructions into effective robotic control policies remains a significant challenge, especially for tasks requiring long-horizon planning and operating under sparse reward conditions. Hierarchical Reinforcement Learning (HRL) provides a natural framework to address this challenge in robotics; however, it typically suffers from non-stationarity caused by the changing behavior of the lower-level policy during training, destabilizing higher-level policy learning. We introduce LGR2, a novel HRL framework that leverages LLMs to generate language-guided reward functions for the higher-level policy. By decoupling high-level reward generation from low-level policy changes, LGR2 fundamentally mitigates the non-stationarity problem in off-policy HRL, enabling stable and efficient learning. To further enhance sample efficiency in sparse environments, we integrate goal-conditioned hindsight experience relabeling. Extensive experiments across simulated and real-world robotic navigation and manipulation tasks demonstrate LGR2 outperforms both hierarchical and non-hierarchical baselines, achieving over 55% success rates on challenging tasks and robust transfer to real robots, without additional fine-tuning.
>
---
#### [replaced 013] Vibration-Based Energy Metric for Restoring Needle Alignment in Autonomous Robotic Ultrasound
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.06921v2](http://arxiv.org/pdf/2508.06921v2)**

> **作者:** Zhongyu Chen; Chenyang Li; Xuesong Li; Dianye Huang; Zhongliang Jiang; Stefanie Speidel; Xiangyu Chu; K. W. Samuel Au
>
> **备注:** Accepted by IROS2025
>
> **摘要:** Precise needle alignment is essential for percutaneous needle insertion in robotic ultrasound-guided procedures. However, inherent challenges such as speckle noise, needle-like artifacts, and low image resolution make robust needle detection difficult, particularly when visibility is reduced or lost. In this paper, we propose a method to restore needle alignment when the ultrasound imaging plane and the needle insertion plane are misaligned. Unlike many existing approaches that rely heavily on needle visibility in ultrasound images, our method uses a more robust feature by periodically vibrating the needle using a mechanical system. Specifically, we propose a vibration-based energy metric that remains effective even when the needle is fully out of plane. Using this metric, we develop a control strategy to reposition the ultrasound probe in response to misalignments between the imaging plane and the needle insertion plane in both translation and rotation. Experiments conducted on ex-vivo porcine tissue samples using a dual-arm robotic ultrasound-guided needle insertion system demonstrate the effectiveness of the proposed approach. The experimental results show the translational error of 0.41$\pm$0.27 mm and the rotational error of 0.51$\pm$0.19 degrees.
>
---
#### [replaced 014] RNBF: Real-Time RGB-D Based Neural Barrier Functions for Safe Robotic Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.02294v2](http://arxiv.org/pdf/2505.02294v2)**

> **作者:** Satyajeet Das; Yifan Xue; Haoming Li; Nadia Figueroa
>
> **摘要:** Autonomous safe navigation in unstructured and novel environments poses significant challenges, especially when environment information can only be provided through low-cost vision sensors. Although safe reactive approaches have been proposed to ensure robot safety in complex environments, many base their theory off the assumption that the robot has prior knowledge on obstacle locations and geometries. In this paper, we present a real-time, vision-based framework that constructs continuous, first-order differentiable Signed Distance Fields (SDFs) of unknown environments entirely online, without any pre-training, and is fully compatible with established SDF-based reactive controllers. To achieve robust performance under practical sensing conditions, our approach explicitly accounts for noise in affordable RGB-D cameras, refining the neural SDF representation online for smoother geometry and stable gradient estimates. We validate the proposed method in simulation and real-world experiments using a Fetch robot.
>
---
#### [replaced 015] NarraGuide: an LLM-based Narrative Mobile Robot for Remote Place Exploration
- **分类: cs.HC; cs.RO; 68**

- **链接: [http://arxiv.org/pdf/2508.01235v2](http://arxiv.org/pdf/2508.01235v2)**

> **作者:** Yaxin Hu; Arissa J. Sato; Jingxin Du; Chenming Ye; Anjun Zhu; Pragathi Praveena; Bilge Mutlu
>
> **摘要:** Robotic telepresence enables users to navigate and experience remote environments. However, effective navigation and situational awareness depend on users' prior knowledge of the environment, limiting the usefulness of these systems for exploring unfamiliar places. We explore how integrating location-aware LLM-based narrative capabilities into a mobile robot can support remote exploration. We developed a prototype system, called NarraGuide, that provides narrative guidance for users to explore and learn about a remote place through a dialogue-based interface. We deployed our prototype in a geology museum, where remote participants (n=20) used the robot to tour the museum. Our findings reveal how users perceived the robot's role, engaged in dialogue in the tour, and expressed preferences for bystander encountering. Our work demonstrates the potential of LLM-enabled robotic capabilities to deliver location-aware narrative guidance and enrich the experience of exploring remote environments.
>
---
#### [replaced 016] RIFT: Closed-Loop RL Fine-Tuning for Realistic and Controllable Traffic Simulation
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.03344v2](http://arxiv.org/pdf/2505.03344v2)**

> **作者:** Keyu Chen; Wenchao Sun; Hao Cheng; Sifa Zheng
>
> **摘要:** Achieving both realism and controllability in closed-loop traffic simulation remains a key challenge in autonomous driving. Dataset-based methods reproduce realistic trajectories but suffer from covariate shift in closed-loop deployment, compounded by simplified dynamics models that further reduce reliability. Conversely, physics-based simulation methods enhance reliable and controllable closed-loop interactions but often lack expert demonstrations, compromising realism. To address these challenges, we introduce a dual-stage AV-centric simulation framework that conducts open-loop imitation learning pre-training in a data-driven simulator to capture trajectory-level realism and route-level controllability, followed by closed-loop reinforcement learning fine-tuning in a physics-based simulator to enhance style-level controllability and mitigate covariate shift. In the fine-tuning stage, we propose RIFT, a novel RL fine-tuning strategy that evaluates all candidate modalities through group-relative optimization with a dual-clip surrogate objective, enhancing style-level controllability and mitigating covariate shift, while preserving the trajectory-level realism and route-level controllability inherited from IL pre-training. Extensive experiments demonstrate that RIFT improves realism and controllability in traffic simulation while simultaneously exposing the limitations of modern AV systems in closed-loop evaluation. Project Page: https://currychen77.github.io/RIFT/
>
---
#### [replaced 017] LD-Scene: LLM-Guided Diffusion for Controllable Generation of Adversarial Safety-Critical Driving Scenarios
- **分类: cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.11247v2](http://arxiv.org/pdf/2505.11247v2)**

> **作者:** Mingxing Peng; Yuting Xie; Xusen Guo; Ruoyu Yao; Hai Yang; Jun Ma
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Ensuring the safety and robustness of autonomous driving systems necessitates a comprehensive evaluation in safety-critical scenarios. However, these safety-critical scenarios are rare and difficult to collect from real-world driving data, posing significant challenges to effectively assessing the performance of autonomous vehicles. Typical existing methods often suffer from limited controllability and lack user-friendliness, as extensive expert knowledge is essentially required. To address these challenges, we propose LD-Scene, a novel framework that integrates Large Language Models (LLMs) with Latent Diffusion Models (LDMs) for user-controllable adversarial scenario generation through natural language. Our approach comprises an LDM that captures realistic driving trajectory distributions and an LLM-based guidance module that translates user queries into adversarial loss functions, facilitating the generation of scenarios aligned with user queries. The guidance module integrates an LLM-based Chain-of-Thought (CoT) code generator and an LLM-based code debugger, enhancing the controllability and robustness in generating guidance functions. Extensive experiments conducted on the nuScenes dataset demonstrate that LD-Scene achieves state-of-the-art performance in generating realistic, diverse, and effective adversarial scenarios. Furthermore, our framework provides fine-grained control over adversarial behaviors, thereby facilitating more effective testing tailored to specific driving scenarios.
>
---
#### [replaced 018] LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.11528v4](http://arxiv.org/pdf/2505.11528v4)**

> **作者:** Yuhang Huang; Jiazhao Zhang; Shilong Zou; Xinwang Liu; Ruizhen Hu; Kai Xu
>
> **备注:** CoRL 2025
>
> **摘要:** Predictive manipulation has recently gained considerable attention in the Embodied AI community due to its potential to improve robot policy performance by leveraging predicted states. However, generating accurate future visual states of robot-object interactions from world models remains a well-known challenge, particularly in achieving high-quality pixel-level representations. To this end, we propose LaDi-WM, a world model that predicts the latent space of future states using diffusion modeling. Specifically, LaDi-WM leverages the well-established latent space aligned with pre-trained Visual Foundation Models (VFMs), which comprises both geometric features (DINO-based) and semantic features (CLIP-based). We find that predicting the evolution of the latent space is easier to learn and more generalizable than directly predicting pixel-level images. Building on LaDi-WM, we design a diffusion policy that iteratively refines output actions by incorporating forecasted states, thereby generating more consistent and accurate results. Extensive experiments on both synthetic and real-world benchmarks demonstrate that LaDi-WM significantly enhances policy performance by 27.9\% on the LIBERO-LONG benchmark and 20\% on the real-world scenario. Furthermore, our world model and policies achieve impressive generalizability in real-world experiments.
>
---
#### [replaced 019] Towards Safe Autonomous Driving Policies using a Neuro-Symbolic Deep Reinforcement Learning Approach
- **分类: cs.RO; cs.AI; cs.LG; cs.LO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2307.01316v3](http://arxiv.org/pdf/2307.01316v3)**

> **作者:** Iman Sharifi; Mustafa Yildirim; Saber Fallah
>
> **备注:** 15 pages, 9 figures, 1 table, 1 algorithm
>
> **摘要:** The dynamic nature of driving environments and the presence of diverse road users pose significant challenges for decision-making in autonomous driving. Deep reinforcement learning (DRL) has emerged as a popular approach to tackle this problem. However, the application of existing DRL solutions is mainly confined to simulated environments due to safety concerns, impeding their deployment in real-world. To overcome this limitation, this paper introduces a novel neuro-symbolic model-free DRL approach, called DRL with Symbolic Logic (DRLSL) that combines the strengths of DRL (learning from experience) and symbolic first-order logic (knowledge-driven reasoning) to enable safe learning in real-time interactions of autonomous driving within real environments. This innovative approach provides a means to learn autonomous driving policies by actively engaging with the physical environment while ensuring safety. We have implemented the DRLSL framework in a highway driving scenario using the HighD dataset and demonstrated that our method successfully avoids unsafe actions during both the training and testing phases. Furthermore, our results indicate that DRLSL achieves faster convergence during training and exhibits better generalizability to new highway driving scenarios compared to traditional DRL methods.
>
---
#### [replaced 020] Mapping the Unseen: Unified Promptable Panoptic Mapping with Dynamic Labeling using Foundation Models
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2405.02162v4](http://arxiv.org/pdf/2405.02162v4)**

> **作者:** Mohamad Al Mdfaa; Raghad Salameh; Geesara Kulathunga; Sergey Zagoruyko; Gonzalo Ferrer
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** In robotics and computer vision, semantic mapping remains a critical challenge for machines to comprehend complex environments. Traditional panoptic mapping approaches are constrained by fixed labels, limiting their ability to handle novel objects. We present Unified Promptable Panoptic Mapping (UPPM), which leverages foundation models for dynamic labeling without additional training. UPPM is evaluated across three comprehensive levels: Segmentation-to-Map, Map-to-Map, and Segmentation-to-Segmentation. Results demonstrate UPPM attains exceptional geometry reconstruction accuracy (0.61cm on the Flat dataset), the highest panoptic quality (0.414), and better performance compared to state-of-the-art segmentation methods. Furthermore, ablation studies validate the contributions of unified semantics, custom NMS, and blurry frame filtering, with the custom NMS improving the completion ratio by 8.27% on the Flat dataset. UPPM demonstrates effective scene reconstruction with rich semantic labeling across diverse datasets.
>
---
#### [replaced 021] CaRL: Learning Scalable Planning Policies with Simple Rewards
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.17838v2](http://arxiv.org/pdf/2504.17838v2)**

> **作者:** Bernhard Jaeger; Daniel Dauner; Jens Beißwenger; Simon Gerstenecker; Kashyap Chitta; Andreas Geiger
>
> **备注:** Accepted at the Conference on Robot Learning 2025
>
> **摘要:** We investigate reinforcement learning (RL) for privileged planning in autonomous driving. State-of-the-art approaches for this task are rule-based, but these methods do not scale to the long tail. RL, on the other hand, is scalable and does not suffer from compounding errors like imitation learning. Contemporary RL approaches for driving use complex shaped rewards that sum multiple individual rewards, \eg~progress, position, or orientation rewards. We show that PPO fails to optimize a popular version of these rewards when the mini-batch size is increased, which limits the scalability of these approaches. Instead, we propose a new reward design based primarily on optimizing a single intuitive reward term: route completion. Infractions are penalized by terminating the episode or multiplicatively reducing route completion. We find that PPO scales well with higher mini-batch sizes when trained with our simple reward, even improving performance. Training with large mini-batch sizes enables efficient scaling via distributed data parallelism. We scale PPO to 300M samples in CARLA and 500M samples in nuPlan with a single 8-GPU node. The resulting model achieves 64 DS on the CARLA longest6 v2 benchmark, outperforming other RL methods with more complex rewards by a large margin. Requiring only minimal adaptations from its use in CARLA, the same method is the best learning-based approach on nuPlan. It scores 91.3 in non-reactive and 90.6 in reactive traffic on the Val14 benchmark while being an order of magnitude faster than prior work.
>
---
#### [replaced 022] STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2412.15182v2](http://arxiv.org/pdf/2412.15182v2)**

> **作者:** Marius Memmel; Jacob Berg; Bingqing Chen; Abhishek Gupta; Jonathan Francis
>
> **备注:** Project website at https://weirdlabuw.github.io/strap/
>
> **摘要:** Robot learning is witnessing a significant increase in the size, diversity, and complexity of pre-collected datasets, mirroring trends in domains such as natural language processing and computer vision. Many robot learning methods treat such datasets as multi-task expert data and learn a multi-task, generalist policy by training broadly across them. Notably, while these generalist policies can improve the average performance across many tasks, the performance of generalist policies on any one task is often suboptimal due to negative transfer between partitions of the data, compared to task-specific specialist policies. In this work, we argue for the paradigm of training policies during deployment given the scenarios they encounter: rather than deploying pre-trained policies to unseen problems in a zero-shot manner, we non-parametrically retrieve and train models directly on relevant data at test time. Furthermore, we show that many robotics tasks share considerable amounts of low-level behaviors and that retrieval at the "sub"-trajectory granularity enables significantly improved data utilization, generalization, and robustness in adapting policies to novel problems. In contrast, existing full-trajectory retrieval methods tend to underutilize the data and miss out on shared cross-task content. This work proposes STRAP, a technique for leveraging pre-trained vision foundation models and dynamic time warping to retrieve sub-sequences of trajectories from large training corpora in a robust fashion. STRAP outperforms both prior retrieval algorithms and multi-task learning methods in simulated and real experiments, showing the ability to scale to much larger offline datasets in the real world as well as the ability to learn robust control policies with just a handful of real-world demonstrations.
>
---
#### [replaced 023] D-CODA: Diffusion for Coordinated Dual-Arm Data Augmentation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.04860v2](http://arxiv.org/pdf/2505.04860v2)**

> **作者:** I-Chun Arthur Liu; Jason Chen; Gaurav Sukhatme; Daniel Seita
>
> **备注:** Accepted to the Conference on Robot Learning (CoRL) 2025
>
> **摘要:** Learning bimanual manipulation is challenging due to its high dimensionality and tight coordination required between two arms. Eye-in-hand imitation learning, which uses wrist-mounted cameras, simplifies perception by focusing on task-relevant views. However, collecting diverse demonstrations remains costly, motivating the need for scalable data augmentation. While prior work has explored visual augmentation in single-arm settings, extending these approaches to bimanual manipulation requires generating viewpoint-consistent observations across both arms and producing corresponding action labels that are both valid and feasible. In this work, we propose Diffusion for COordinated Dual-arm Data Augmentation (D-CODA), a method for offline data augmentation tailored to eye-in-hand bimanual imitation learning that trains a diffusion model to synthesize novel, viewpoint-consistent wrist-camera images for both arms while simultaneously generating joint-space action labels. It employs constrained optimization to ensure that augmented states involving gripper-to-object contacts adhere to constraints suitable for bimanual coordination. We evaluate D-CODA on 5 simulated and 3 real-world tasks. Our results across 2250 simulation trials and 300 real-world trials demonstrate that it outperforms baselines and ablations, showing its potential for scalable data augmentation in eye-in-hand bimanual manipulation. Our project website is at: https://dcodaaug.github.io/D-CODA/.
>
---
#### [replaced 024] HQ-OV3D: A High Box Quality Open-World 3D Detection Framework based on Diffision Model
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.10935v2](http://arxiv.org/pdf/2508.10935v2)**

> **作者:** Qi Liu; Yabei Li; Hongsong Wang; Lei He
>
> **摘要:** Traditional closed-set 3D detection frameworks fail to meet the demands of open-world applications like autonomous driving. Existing open-vocabulary 3D detection methods typically adopt a two-stage pipeline consisting of pseudo-label generation followed by semantic alignment. While vision-language models (VLMs) recently have dramatically improved the semantic accuracy of pseudo-labels, their geometric quality, particularly bounding box precision, remains commonly neglected. To address this issue, we propose a High Box Quality Open-Vocabulary 3D Detection (HQ-OV3D) framework, dedicated to generate and refine high-quality pseudo-labels for open-vocabulary classes. The framework comprises two key components: an Intra-Modality Cross-Validated (IMCV) Proposal Generator that utilizes cross-modality geometric consistency to generate high-quality initial 3D proposals, and an Annotated-Class Assisted (ACA) Denoiser that progressively refines 3D proposals by leveraging geometric priors from annotated categories through a DDIM-based denoising mechanism. Compared to the state-of-the-art method, training with pseudo-labels generated by our approach achieves a 7.37% improvement in mAP on novel classes, demonstrating the superior quality of the pseudo-labels produced by our framework. HQ-OV3D can serve not only as a strong standalone open-vocabulary 3D detector but also as a plug-in high-quality pseudo-label generator for existing open-vocabulary detection or annotation pipelines.
>
---
#### [replaced 025] Reasoning and Learning a Perceptual Metric for Self-Training of Reflective Objects in Bin-Picking with a Low-cost Camera
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.20207v2](http://arxiv.org/pdf/2503.20207v2)**

> **作者:** Peiyuan Ni; Chee Meng Chew; Marcelo H. Ang Jr.; Gregory S. Chirikjian
>
> **备注:** 8 pages, 10 figures; Accepted by IEEE RAL, presentation at ICRA 2026
>
> **摘要:** Bin-picking of metal objects using low-cost RGB-D cameras often suffers from sparse depth information and reflective surface textures, leading to errors and the need for manual labeling. To reduce human intervention, we propose a two-stage framework consisting of a metric learning stage and a self-training stage. Specifically, to automatically process data captured by a low-cost camera (LC), we introduce a Multi-object Pose Reasoning (MoPR) algorithm that optimizes pose hypotheses under depth, collision, and boundary constraints. To further refine pose candidates, we adopt a Symmetry-aware Lie-group based Bayesian Gaussian Mixture Model (SaL-BGMM), integrated with the Expectation-Maximization (EM) algorithm, for symmetry-aware filtering. Additionally, we propose a Weighted Ranking Information Noise Contrastive Estimation (WR-InfoNCE) loss to enable the LC to learn a perceptual metric from reconstructed data, supporting self-training on untrained or even unseen objects. Experimental results show that our approach outperforms several state-of-the-art methods on both the ROBI dataset and our newly introduced Self-ROBI dataset.
>
---
#### [replaced 026] RT-Cache: Training-Free Retrieval for Real-Time Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.09040v2](http://arxiv.org/pdf/2505.09040v2)**

> **作者:** Owen Kwon; Abraham George; Alison Bartsch; Amir Barati Farimani
>
> **备注:** 8 pages, 6 figures. Accepted to the 2025 IEEE-RAS 24th International Conference on Humanoid Robots
>
> **摘要:** Real robots are expected to repeat the same behavior in new environments with very little new data, yet modern controllers either incur heavy per-step inference or require deployment-time fine-tuning. We propose RT-Cache, a training-free retrieval-as-control pipeline that caches diverse image action trajectories in a unified vector memory and, at test time, embeds the current frame to retrieve and replay multi-step snippets, replacing per-step model calls. A hierarchical search keeps lookups sub-second at million scale, shifting cost from compute to storage and enabling real-time control on modest GPUs. Across real-robot tasks and large open logs, RT-Cache achieves higher success and lower completion time than strong retrieval baselines (approximately x2 higher success and ~30% faster in our settings), and a single-episode anchoring study shows immediate adaptation to a more complex, contact-rich task without fine-tuning. RT-Cache turns experience into an append-only memory, offering a simple, scalable path to few-shot deployment today and a foundation for multimodal keys and optional integration with high-level policies. Project page: https://rt-cache.github.io/.
>
---
#### [replaced 027] A flexible framework for accurate LiDAR odometry, map manipulation, and localization
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2407.20465v3](http://arxiv.org/pdf/2407.20465v3)**

> **作者:** José Luis Blanco-Claraco
>
> **备注:** 44 pages, 35 figures
>
> **摘要:** LiDAR-based SLAM is a core technology for autonomous vehicles and robots. One key contribution of this work to 3D LiDAR SLAM and localization is a fierce defense of view-based maps (pose graphs with time-stamped sensor readings) as the fundamental representation of maps. As will be shown, they allow for the greatest flexibility, enabling the posterior generation of arbitrary metric maps optimized for particular tasks, e.g. obstacle avoidance, real-time localization. Moreover, this work introduces a new framework in which mapping pipelines can be defined without coding, defining the connections of a network of reusable blocks much like deep-learning networks are designed by connecting layers of standardized elements. We also introduce tightly-coupled estimation of linear and angular velocity vectors within the Iterative Closest Point (ICP)-like optimizer, leading to superior robustness against aggressive motion profiles without the need for an IMU. Extensive experimental validation reveals that the proposal compares well to, or improves, former state-of-the-art (SOTA) LiDAR odometry systems, while also successfully mapping some hard sequences where others diverge. A proposed self-adaptive configuration has been used, without parameter changes, for all 3D LiDAR datasets with sensors between 16 and 128 rings, and has been extensively tested on 83 sequences over more than 250~km of automotive, hand-held, airborne, and quadruped LiDAR datasets, both indoors and outdoors. The system flexibility is demonstrated with additional configurations for 2D LiDARs and for building 3D NDT-like maps. The framework is open-sourced online: https://github.com/MOLAorg/mola
>
---
#### [replaced 028] Affordance-R1: Reinforcement Learning for Generalizable Affordance Reasoning in Multimodal Large Language Model
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.06206v3](http://arxiv.org/pdf/2508.06206v3)**

> **作者:** Hanqing Wang; Shaoyang Wang; Yiming Zhong; Zemin Yang; Jiamin Wang; Zhiqing Cui; Jiahao Yuan; Yifan Han; Mingyu Liu; Yuexin Ma
>
> **摘要:** Affordance grounding focuses on predicting the specific regions of objects that are associated with the actions to be performed by robots. It plays a vital role in the fields of human-robot interaction, human-object interaction, embodied manipulation, and embodied perception. Existing models often neglect the affordance shared among different objects because they lack the Chain-of-Thought(CoT) reasoning abilities, limiting their out-of-domain (OOD) generalization and explicit reasoning capabilities. To address these challenges, we propose Affordance-R1, the first unified affordance grounding framework that integrates cognitive CoT guided Group Relative Policy Optimization (GRPO) within a reinforcement learning paradigm. Specifically, we designed a sophisticated affordance function, which contains format, perception, and cognition rewards to effectively guide optimization directions. Furthermore, we constructed a high-quality affordance-centric reasoning dataset, ReasonAff, to support training. Trained exclusively via reinforcement learning with GRPO and without explicit reasoning data, Affordance-R1 achieves robust zero-shot generalization and exhibits emergent test-time reasoning capabilities. Comprehensive experiments demonstrate that our model outperforms well-established methods and exhibits open-world generalization. To the best of our knowledge, Affordance-R1 is the first to integrate GRPO-based RL with reasoning into affordance reasoning. The code of our method and our dataset is released on https://github.com/hq-King/Affordance-R1.
>
---
#### [replaced 029] The Foundational Pose as a Selection Mechanism for the Design of Tool-Wielding Multi-Finger Robotic Hands
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.14158v2](http://arxiv.org/pdf/2409.14158v2)**

> **作者:** Sunyu Wang; Jean Oh; Nancy S. Pollard
>
> **摘要:** To wield an object means to hold and move it in a way that exploits its functions. When humans wield tools -- such as writing with a pen or cutting with scissors -- our hands would reach very specific poses, often drastically different from how we pick up the same objects just to transport them. In this work, we investigate the design of tool-wielding multi-finger robotic hand through a hypothesis: If a hand can kinematically reach a foundational pose (FP) with a tool, then it can wield the tool from that FP. We interpret FPs as snapshots that capture the workings of underlying parallel mechanisms formed by the tool and the hand, and one hand can form multiple mechanisms with the same tool. We tested our hypothesis in a hand design experiment, where we developed a sampling-based multi-objective design optimization framework that uses three FPs to computationally generate many different hand designs and evaluate them. The results show that 10,785 out of the 100,480 hand designs we sampled reached the FPs; more than 99\% of the 10,785 hands that reached the FPs successfully wielded tools, supporting our hypothesis. Meanwhile, our methods provide insights into the non-convex, multi-objective hand design optimization problem -- such as clustering and the Pareto front -- that could be hard to unveil with methods that return a single ``optimal" design. Lastly, we demonstrate our methods' real-world feasibility and potential with a hardware prototype equipped with rigid endoskeleton and soft skin.
>
---
#### [replaced 030] Towards Infant Sleep-Optimized Driving: Synergizing Wearable and Vehicle Sensing in Intelligent Cruise Control
- **分类: cs.LG; cs.ET; cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.06459v3](http://arxiv.org/pdf/2506.06459v3)**

> **作者:** Ruitao Chen; Mozhang Guo; Jinge Li
>
> **摘要:** Automated driving (AD) has substantially improved vehicle safety and driving comfort, but their impact on passenger well-being, particularly infant sleep, is not sufficiently studied. Sudden acceleration, abrupt braking, and sharp maneuvers can disrupt infant sleep, compromising both passenger comfort and parental convenience. To solve this problem, this paper explores the integration of reinforcement learning (RL) within AD to personalize driving behavior and optimally balance occupant comfort and travel efficiency. In particular, we propose an intelligent cruise control framework that adapts to varying driving conditions to enhance infant sleep quality by effectively synergizing wearable sensing and vehicle data. Long short-term memory (LSTM) and transformer-based neural networks are integrated with RL to model the relationship between driving behavior and infant sleep quality under diverse traffic and road conditions. Based on the sleep quality indicators from the wearable sensors, driving action data from vehicle controllers, and map data from map applications, the model dynamically computes the optimal driving aggressiveness level, which is subsequently translated into specific AD control strategies, e.g., the magnitude and frequency of acceleration, lane change, and overtaking. Simulation experiments conducted in the CARLA environment indicate that the proposed solution significantly improves infant sleep quality compared to baseline methods, while preserving desirable travel efficiency.
>
---
#### [replaced 031] SLAC: Simulation-Pretrained Latent Action Space for Whole-Body Real-World RL
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.04147v4](http://arxiv.org/pdf/2506.04147v4)**

> **作者:** Jiaheng Hu; Peter Stone; Roberto Martín-Martín
>
> **备注:** CoRL 2025
>
> **摘要:** Building capable household and industrial robots requires mastering the control of versatile, high-degree-of-freedom (DoF) systems such as mobile manipulators. While reinforcement learning (RL) holds promise for autonomously acquiring robot control policies, scaling it to high-DoF embodiments remains challenging. Direct RL in the real world demands both safe exploration and high sample efficiency, which are difficult to achieve in practice. Sim-to-real RL, on the other hand, is often brittle due to the reality gap. This paper introduces SLAC, a method that renders real-world RL feasible for complex embodiments by leveraging a low-fidelity simulator to pretrain a task-agnostic latent action space. SLAC trains this latent action space via a customized unsupervised skill discovery method designed to promote temporal abstraction, disentanglement, and safety, thereby facilitating efficient downstream learning. Once a latent action space is learned, SLAC uses it as the action interface for a novel off-policy RL algorithm to autonomously learn downstream tasks through real-world interactions. We evaluate SLAC against existing methods on a suite of bimanual mobile manipulation tasks, where it achieves state-of-the-art performance. Notably, SLAC learns contact-rich whole-body tasks in under an hour of real-world interactions, without relying on any demonstrations or hand-crafted behavior priors. More information and robot videos at robo-rl.github.io
>
---
#### [replaced 032] Visual Perception Engine: Fast and Flexible Multi-Head Inference for Robotic Vision Tasks
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.11584v2](http://arxiv.org/pdf/2508.11584v2)**

> **作者:** Jakub Łucki; Jonathan Becktor; Georgios Georgakis; Rob Royce; Shehryar Khattak
>
> **备注:** 8 pages, 6 figures, 2 tables
>
> **摘要:** Deploying multiple machine learning models on resource-constrained robotic platforms for different perception tasks often results in redundant computations, large memory footprints, and complex integration challenges. In response, this work presents Visual Perception Engine (VPEngine), a modular framework designed to enable efficient GPU usage for visual multitasking while maintaining extensibility and developer accessibility. Our framework architecture leverages a shared foundation model backbone that extracts image representations, which are efficiently shared, without any unnecessary GPU-CPU memory transfers, across multiple specialized task-specific model heads running in parallel. This design eliminates the computational redundancy inherent in feature extraction component when deploying traditional sequential models while enabling dynamic task prioritization based on application demands. We demonstrate our framework's capabilities through an example implementation using DINOv2 as the foundation model with multiple task (depth, object detection and semantic segmentation) heads, achieving up to 3x speedup compared to sequential execution. Building on CUDA Multi-Process Service (MPS), VPEngine offers efficient GPU utilization and maintains a constant memory footprint while allowing per-task inference frequencies to be adjusted dynamically during runtime. The framework is written in Python and is open source with ROS2 C++ (Humble) bindings for ease of use by the robotics community across diverse robotic platforms. Our example implementation demonstrates end-to-end real-time performance at $\geq$50 Hz on NVIDIA Jetson Orin AGX for TensorRT optimized models.
>
---
#### [replaced 033] Hierarchical Multi-Agent Reinforcement Learning with Control Barrier Functions for Safety-Critical Autonomous Systems
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.14850v2](http://arxiv.org/pdf/2507.14850v2)**

> **作者:** H. M. Sabbir Ahmad; Ehsan Sabouni; Alexander Wasilkoff; Param Budhraja; Zijian Guo; Songyuan Zhang; Chuchu Fan; Christos Cassandras; Wenchao Li
>
> **摘要:** We address the problem of safe policy learning in multi-agent safety-critical autonomous systems. In such systems, it is necessary for each agent to meet the safety requirements at all times while also cooperating with other agents to accomplish the task. Toward this end, we propose a safe Hierarchical Multi-Agent Reinforcement Learning (HMARL) approach based on Control Barrier Functions (CBFs). Our proposed hierarchical approach decomposes the overall reinforcement learning problem into two levels learning joint cooperative behavior at the higher level and learning safe individual behavior at the lower or agent level conditioned on the high-level policy. Specifically, we propose a skill-based HMARL-CBF algorithm in which the higher level problem involves learning a joint policy over the skills for all the agents and the lower-level problem involves learning policies to execute the skills safely with CBFs. We validate our approach on challenging environment scenarios whereby a large number of agents have to safely navigate through conflicting road networks. Compared with existing state of the art methods, our approach significantly improves the safety achieving near perfect (within 5%) success/safety rate while also improving performance across all the environments.
>
---
#### [replaced 034] Self-Tuning PID Control via a Hybrid Actor-Critic-Based Neural Structure for Quadcopter Control
- **分类: eess.SY; cs.AI; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2307.01312v2](http://arxiv.org/pdf/2307.01312v2)**

> **作者:** Iman Sharifi; Aria Alasty
>
> **备注:** 7 pages, 18 figures, The 30th Annual International Conference of Iranian Society of Mechanical Engineers
>
> **摘要:** Proportional-Integrator-Derivative (PID) controller is used in a wide range of industrial and experimental processes. There are a couple of offline methods for tuning PID gains. However, due to the uncertainty of model parameters and external disturbances, real systems such as Quadrotors need more robust and reliable PID controllers. In this research, a self-tuning PID controller using a Reinforcement-Learning-based Neural Network for attitude and altitude control of a Quadrotor has been investigated. An Incremental PID, which contains static and dynamic gains, has been considered and only the variable gains have been tuned. To tune dynamic gains, a model-free actor-critic-based hybrid neural structure was used that was able to properly tune PID gains, and also has done the best as an identifier. In both tunning and identification tasks, a Neural Network with two hidden layers and sigmoid activation functions has been learned using Adaptive Momentum (ADAM) optimizer and Back-Propagation (BP) algorithm. This method is online, able to tackle disturbance, and fast in training. In addition to robustness to mass uncertainty and wind gust disturbance, results showed that the proposed method had a better performance when compared to a PID controller with constant gains.
>
---
#### [replaced 035] SLAG: Scalable Language-Augmented Gaussian Splatting
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.08124v2](http://arxiv.org/pdf/2505.08124v2)**

> **作者:** Laszlo Szilagyi; Francis Engelmann; Jeannette Bohg
>
> **摘要:** Language-augmented scene representations hold great promise for large-scale robotics applications such as search-and-rescue, smart cities, and mining. Many of these scenarios are time-sensitive, requiring rapid scene encoding while also being data-intensive, necessitating scalable solutions. Deploying these representations on robots with limited computational resources further adds to the challenge. To address this, we introduce SLAG, a multi-GPU framework for language-augmented Gaussian splatting that enhances the speed and scalability of embedding large scenes. Our method integrates 2D visual-language model features into 3D scenes using SAM and CLIP. Unlike prior approaches, SLAG eliminates the need for a loss function to compute per-Gaussian language embeddings. Instead, it derives embeddings from 3D Gaussian scene parameters via a normalized weighted average, enabling highly parallelized scene encoding. Additionally, we introduce a vector database for efficient embedding storage and retrieval. Our experiments show that SLAG achieves an 18 times speedup in embedding computation on a 16-GPU setup compared to OpenGaussian, while preserving embedding quality on the ScanNet and LERF datasets. For more details, visit our project website: https://slag-project.github.io/.
>
---
#### [replaced 036] Safe and Efficient Robot Action Planning in the Presence of Unconcerned Humans
- **分类: cs.RO; math.OC**

- **链接: [http://arxiv.org/pdf/2501.13203v2](http://arxiv.org/pdf/2501.13203v2)**

> **作者:** Mohsen Amiri; Mehdi Hosseinzadeh
>
> **摘要:** This paper proposes a robot action planning scheme that provides an efficient and probabilistically safe plan for a robot interacting with an unconcerned human -- someone who is either unaware of the robot's presence or unwilling to engage in ensuring safety. The proposed scheme is predictive, meaning that the robot is required to predict human actions over a finite future horizon; such predictions are often inaccurate in real-world scenarios. One possible approach to reduce the uncertainties is to provide the robot with the capability of reasoning about the human's awareness of potential dangers. This paper discusses that by using a binary variable, so-called danger awareness coefficient, it is possible to differentiate between concerned and unconcerned humans, and provides a learning algorithm to determine this coefficient by observing human actions. Moreover, this paper argues how humans rely on predictions of other agents' future actions (including those of robots in human-robot interaction) in their decision-making. It also shows that ignoring this aspect in predicting human's future actions can significantly degrade the efficiency of the interaction, causing agents to deviate from their optimal paths. The proposed robot action planning scheme is verified and validated via extensive simulation and experimental studies on a LoCoBot WidowX-250.
>
---
#### [replaced 037] Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.12609v3](http://arxiv.org/pdf/2504.12609v3)**

> **作者:** Tyler Ga Wei Lum; Olivia Y. Lee; C. Karen Liu; Jeannette Bohg
>
> **摘要:** Teaching robots dexterous manipulation skills often requires collecting hundreds of demonstrations using wearables or teleoperation, a process that is challenging to scale. Videos of human-object interactions are easier to collect and scale, but leveraging them directly for robot learning is difficult due to the lack of explicit action labels and human-robot embodiment differences. We propose Human2Sim2Robot, a novel real-to-sim-to-real framework for training dexterous manipulation policies using only one RGB-D video of a human demonstrating a task. Our method utilizes reinforcement learning (RL) in simulation to cross the embodiment gap without relying on wearables, teleoperation, or large-scale data collection. From the video, we extract: (1) the object pose trajectory to define an object-centric, embodiment-agnostic reward, and (2) the pre-manipulation hand pose to initialize and guide exploration during RL training. These components enable effective policy learning without any task-specific reward tuning. In the single human demo regime, Human2Sim2Robot outperforms object-aware replay by over 55% and imitation learning by over 68% on grasping, non-prehensile manipulation, and multi-step tasks. Website: https://human2sim2robot.github.io
>
---
#### [replaced 038] Measuring and Minimizing Disturbance of Marine Animals to Underwater Vehicles
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.11335v2](http://arxiv.org/pdf/2506.11335v2)**

> **作者:** Levi Cai; Youenn Jézéquel; T. Aran Mooney; Yogesh Girdhar
>
> **备注:** ISER 2025 proceedings
>
> **摘要:** Do fish respond to the presence of underwater vehicles, potentially biasing our estimates about them? If so, are there strategies to measure and mitigate this response? This work provides a theoretical and practical framework towards bias-free estimation of animal behavior from underwater vehicle observations. We also provide preliminary results from the field in coral reef environments to address these questions.
>
---
#### [replaced 039] Tracking Control of Euler-Lagrangian Systems with Prescribed State, Input, and Temporal Constraints
- **分类: eess.SY; cs.RO; cs.SY; math.OC**

- **链接: [http://arxiv.org/pdf/2503.01866v3](http://arxiv.org/pdf/2503.01866v3)**

> **作者:** Chidre Shravista Kashyap; Pushpak Jagtap; Jishnu Keshavan
>
> **摘要:** The synthesis of a smooth tracking control for Euler-Lagrangian (EL) systems under stringent state, input, and temporal (SIT) constraints is challenging. In contrast to existing methods that utilize prior knowledge of EL model parameters and uncertainty bounds, this study proposes an approximation-free adaptive barrier function-based control policy to ensure local prescribed time convergence of tracking error under state and input constraints. The proposed approach uses smooth time-based generator functions embedded in the filtered tracking error, which is combined with a saturation function that limits control action and confines states within the prescribed limits by enforcing the time-varying bounds on the filtered tracking error. Importantly, corresponding feasibility conditions are derived pertaining to the minimum control authority, the maximum disturbance rejection capability of the control policy, and the viable set of initial conditions, illuminating the narrow operating domain of EL systems arising from the interplay of SIT constraints. Finally, the efficacy of the proposed approach is demonstrated using experimental and comparison studies.
>
---
#### [replaced 040] Research Challenges and Progress in the End-to-End V2X Cooperative Autonomous Driving Competition
- **分类: cs.RO; cs.CV; I.4.9**

- **链接: [http://arxiv.org/pdf/2507.21610v2](http://arxiv.org/pdf/2507.21610v2)**

> **作者:** Ruiyang Hao; Haibao Yu; Jiaru Zhong; Chuanye Wang; Jiahao Wang; Yiming Kan; Wenxian Yang; Siqi Fan; Huilin Yin; Jianing Qiu; Yao Mu; Jiankai Sun; Li Chen; Walter Zimmer; Dandan Zhang; Shanghang Zhang; Mac Schwager; Ping Luo; Zaiqing Nie
>
> **备注:** 10 pages, 4 figures, accepted by ICCVW Author list updated to match the camera-ready version, in compliance with conference policy
>
> **摘要:** With the rapid advancement of autonomous driving technology, vehicle-to-everything (V2X) communication has emerged as a key enabler for extending perception range and enhancing driving safety by providing visibility beyond the line of sight. However, integrating multi-source sensor data from both ego-vehicles and infrastructure under real-world constraints, such as limited communication bandwidth and dynamic environments, presents significant technical challenges. To facilitate research in this area, we organized the End-to-End Autonomous Driving through V2X Cooperation Challenge, which features two tracks: cooperative temporal perception and cooperative end-to-end planning. Built on the UniV2X framework and the V2X-Seq-SPD dataset, the challenge attracted participation from over 30 teams worldwide and established a unified benchmark for evaluating cooperative driving systems. This paper describes the design and outcomes of the challenge, highlights key research problems including bandwidth-aware fusion, robust multi-agent planning, and heterogeneous sensor integration, and analyzes emerging technical trends among top-performing solutions. By addressing practical constraints in communication and data fusion, the challenge contributes to the development of scalable and reliable V2X-cooperative autonomous driving systems.
>
---
