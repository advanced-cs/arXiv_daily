# 机器人 cs.RO

- **最新发布 42 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] Bayesian Optimization for Automatic Tuning of Torque-Level Nonlinear Model Predictive Control
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文针对机器人扭矩型非线性模型预测控制（nMPC）的参数调优难题，提出基于数字孪生与高维贝叶斯优化（SAASBO）的自动调参框架。通过仿真高效搜索参数空间，优化成本权重与控制器增益，显著提升轨迹跟踪精度与求解效率，并在UR10e机械臂上实现性能验证。**

- **链接: [https://arxiv.org/pdf/2512.03772v1](https://arxiv.org/pdf/2512.03772v1)**

> **作者:** Gabriele Fadini; Deepak Ingole; Tong Duy Son; Alisa Rupenyan
>
> **备注:** 6 pages, 7 figures, 3 tables
>
> **摘要:** This paper presents an auto-tuning framework for torque-based Nonlinear Model Predictive Control (nMPC), where the MPC serves as a real-time controller for optimal joint torque commands. The MPC parameters, including cost function weights and low-level controller gains, are optimized using high-dimensional Bayesian Optimization (BO) techniques, specifically Sparse Axis-Aligned Subspace (SAASBO) with a digital twin (DT) to achieve precise end-effector trajectory real-time tracking on an UR10e robot arm. The simulation model allows efficient exploration of the high-dimensional parameter space, and it ensures safe transfer to hardware. Our simulation results demonstrate significant improvements in tracking performance (+41.9%) and reduction in solve times (-2.5%) compared to manually-tuned parameters. Moreover, experimental validation on the real robot follows the trend (with a +25.8% improvement), emphasizing the importance of digital twin-enabled automated parameter optimization for robotic operations.
>
---
#### [new 002] PerFACT: Motion Policy with LLM-Powered Dataset Synthesis and Fusion Action-Chunking Transformers
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对机器人运动规划中数据集小、泛化能力差及模型编码效率低的问题，提出PerFACT框架。通过大语言模型生成多样化工作空间，构建350万条轨迹数据；设计融合动作分块的Transformer网络，提升多模态信息建模与规划速度，显著增强泛化性与效率。**

- **链接: [https://arxiv.org/pdf/2512.03444v1](https://arxiv.org/pdf/2512.03444v1)**

> **作者:** Davood Soleymanzadeh; Xiao Liang; Minghui Zheng
>
> **摘要:** Deep learning methods have significantly enhanced motion planning for robotic manipulators by leveraging prior experiences within planning datasets. However, state-of-the-art neural motion planners are primarily trained on small datasets collected in manually generated workspaces, limiting their generalizability to out-of-distribution scenarios. Additionally, these planners often rely on monolithic network architectures that struggle to encode critical planning information. To address these challenges, we introduce Motion Policy with Dataset Synthesis powered by large language models (LLMs) and Fusion Action-Chunking Transformers (PerFACT), which incorporates two key components. Firstly, a novel LLM-powered workspace generation method, MotionGeneralizer, enables large-scale planning data collection by producing a diverse set of semantically feasible workspaces. Secondly, we introduce Fusion Motion Policy Networks (MpiNetsFusion), a generalist neural motion planner that uses a fusion action-chunking transformer to better encode planning signals and attend to multiple feature modalities. Leveraging MotionGeneralizer, we collect 3.5M trajectories to train and evaluate MpiNetsFusion against state-of-the-art planners, which shows that the proposed MpiNetsFusion can plan several times faster on the evaluated tasks.
>
---
#### [new 003] A Novel Approach to Tomato Harvesting Using a Hybrid Gripper with Semantic Segmentation and Keypoint Detection
- **分类: cs.RO**

- **简介: 该论文针对智能番茄采摘任务，解决复杂环境下精准、无损采摘难题。提出一种融合软硬结构的混合夹爪，结合语义分割与关键点检测实现果实定位，基于虚拟功原理建模并采用力反馈闭环控制，配合粒子群优化轨迹规划，实现高效低损伤采摘。**

- **链接: [https://arxiv.org/pdf/2512.03684v1](https://arxiv.org/pdf/2512.03684v1)**

> **作者:** Shahid Ansari; Mahendra Kumar Gohil; Yusuke Maeda; Bishakh Bhattacharya
>
> **摘要:** This paper presents an autonomous tomato-harvesting system built around a hybrid robotic gripper that combines six soft auxetic fingers with a rigid exoskeleton and a latex basket to achieve gentle, cage-like grasping. The gripper is driven by a servo-actuated Scotch--yoke mechanism, and includes separator leaves that form a conical frustum for fruit isolation, with an integrated micro-servo cutter for pedicel cutting. For perception, an RGB--D camera and a Detectron2-based pipeline perform semantic segmentation of ripe/unripe tomatoes and keypoint localization of the pedicel and fruit center under occlusion and variable illumination. An analytical model derived using the principle of virtual work relates servo torque to grasp force, enabling design-level reasoning about actuation requirements. During execution, closed-loop grasp-force regulation is achieved using a proportional--integral--derivative controller with feedback from force-sensitive resistors mounted on selected fingers to prevent slip and bruising. Motion execution is supported by Particle Swarm Optimization (PSO)--based trajectory planning for a 5-DOF manipulator. Experiments demonstrate complete picking cycles (approach, separation, cutting, grasping, transport, release) with an average cycle time of 24.34~s and an overall success rate of approximately 80\%, while maintaining low grasp forces (0.20--0.50~N). These results validate the proposed hybrid gripper and integrated vision--control pipeline for reliable harvesting in cluttered environments.
>
---
#### [new 004] GOMP: Grasped Object Manifold Projection for Multimodal Imitation Learning of Manipulation
- **分类: cs.RO**

- **简介: 该论文针对模仿学习在精密装配任务中因误差累积导致轨迹精度不足的问题，提出GOMP方法。通过将非刚性抓握物体约束于低维流形，并利用触觉反馈与强化学习优化策略，提升任务精度。方法基于同一专家数据集训练，无需额外标注，且对模态不敏感。**

- **链接: [https://arxiv.org/pdf/2512.03347v1](https://arxiv.org/pdf/2512.03347v1)**

> **作者:** William van den Bogert; Gregory Linkowski; Nima Fazeli
>
> **备注:** 8 pages, 8 figures, 2 tables
>
> **摘要:** Imitation Learning (IL) holds great potential for learning repetitive manipulation tasks, such as those in industrial assembly. However, its effectiveness is often limited by insufficient trajectory precision due to compounding errors. In this paper, we introduce Grasped Object Manifold Projection (GOMP), an interactive method that mitigates these errors by constraining a non-rigidly grasped object to a lower-dimensional manifold. GOMP assumes a precise task in which a manipulator holds an object that may shift within the grasp in an observable manner and must be mated with a grounded part. Crucially, all GOMP enhancements are learned from the same expert dataset used to train the base IL policy, and are adjusted with an n-arm bandit-based interactive component. We propose a theoretical basis for GOMP's improvement upon the well-known compounding error bound in IL literature. We demonstrate the framework on four precise assembly tasks using tactile feedback, and note that the approach remains modality-agnostic. Data and videos are available at williamvdb.github.io/GOMPsite.
>
---
#### [new 005] Digital Twin-based Control Co-Design of Full Vehicle Active Suspensions via Deep Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对主动悬架系统在动态环境下性能受限的问题，提出基于数字孪生与深度强化学习的控制协同设计框架。通过联合优化硬件结构与控制策略，实现对不同驾驶风格的个性化调节，显著降低控制能耗并提升舒适性与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.03891v1](https://arxiv.org/pdf/2512.03891v1)**

> **作者:** Ying-Kuan Tsai; Yi-Ping Chen; Vispi Karkaria; Wei Chen
>
> **备注:** 28 pages, 17 figures
>
> **摘要:** Active suspension systems are critical for enhancing vehicle comfort, safety, and stability, yet their performance is often limited by fixed hardware designs and control strategies that cannot adapt to uncertain and dynamic operating conditions. Recent advances in digital twins (DTs) and deep reinforcement learning (DRL) offer new opportunities for real-time, data-driven optimization across a vehicle's lifecycle. However, integrating these technologies into a unified framework remains an open challenge. This work presents a DT-based control co-design (CCD) framework for full-vehicle active suspensions using multi-generation design concepts. By integrating automatic differentiation into DRL, we jointly optimize physical suspension components and control policies under varying driver behaviors and environmental uncertainties. DRL also addresses the challenge of partial observability, where only limited states can be sensed and fed back to the controller, by learning optimal control actions directly from available sensor information. The framework incorporates model updating with quantile learning to capture data uncertainty, enabling real-time decision-making and adaptive learning from digital-physical interactions. The approach demonstrates personalized optimization of suspension systems under two distinct driving settings (mild and aggressive). Results show that the optimized systems achieve smoother trajectories and reduce control efforts by approximately 43% and 52% for mild and aggressive, respectively, while maintaining ride comfort and stability. Contributions include: developing a DT-enabled CCD framework integrating DRL and uncertainty-aware model updating for full-vehicle active suspensions, introducing a multi-generation design strategy for self-improving systems, and demonstrating personalized optimization of active suspension systems for distinct driver types.
>
---
#### [new 006] Hierarchical Vision Language Action Model Using Success and Failure Demonstrations
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对视觉-语言-动作模型因忽略失败数据导致鲁棒性不足的问题，提出VINE模型。通过分层架构分离高层推理与低层控制，利用成功与失败演示联合训练，使系统在规划时识别并规避脆弱路径，提升任务成功率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.03913v1](https://arxiv.org/pdf/2512.03913v1)**

> **作者:** Jeongeun Park; Jihwan Yoon; Byungwoo Jeon; Juhan Park; Jinwoo Shin; Namhoon Cho; Kyungjae Lee; Sangdoo Yun; Sungjoon Choi
>
> **备注:** https://vine-vla.github.io/
>
> **摘要:** Prior Vision-Language-Action (VLA) models are typically trained on teleoperated successful demonstrations, while discarding numerous failed attempts that occur naturally during data collection. However, these failures encode where and how policies can be fragile, information that can be exploited to improve robustness. We address this problem by leveraging mixed-quality datasets to learn failure-aware reasoning at planning time. We introduce VINE, a hierarchical vision-language-action model that separates high-level reasoning (System 2) from low-level control (System 1) under a hierarchical reinforcement learning formalism, making failures usable as a structured learning signal rather than noisy supervision. System 2 performs feasibility-guided tree search over a 2D scene-graph abstraction: it proposes subgoal transitions, predicts success probabilities from both successes and failures, and prunes brittle branches before execution, effectively casting plan evaluation as feasibility scoring. The selected subgoal sequence is then passed to System 1, which executes low-level actions without modifying the agent's core skills. Trained entirely from offline teleoperation data, VINE integrates negative experience directly into the decision loop. Across challenging manipulation tasks, this approach consistently improves success rates and robustness, demonstrating that failure data is an essential resource for converting the broad competence of VLAs into robust execution.
>
---
#### [new 007] What Is The Best 3D Scene Representation for Robotics? From Geometric to Foundation Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文聚焦机器人3D场景表示任务，旨在回答“何种表示最佳”。系统综述点云、体素、NeRF、3DGS及基础模型等方法，分析其在感知、定位、导航等模块的优劣，探讨基础模型作为统一解决方案的潜力与挑战，为未来研究提供参考。**

- **链接: [https://arxiv.org/pdf/2512.03422v1](https://arxiv.org/pdf/2512.03422v1)**

> **作者:** Tianchen Deng; Yue Pan; Shenghai Yuan; Dong Li; Chen Wang; Mingrui Li; Long Chen; Lihua Xie; Danwei Wang; Jingchuan Wang; Javier Civera; Hesheng Wang; Weidong Chen
>
> **摘要:** In this paper, we provide a comprehensive overview of existing scene representation methods for robotics, covering traditional representations such as point clouds, voxels, signed distance functions (SDF), and scene graphs, as well as more recent neural representations like Neural Radiance Fields (NeRF), 3D Gaussian Splatting (3DGS), and the emerging Foundation Models. While current SLAM and localization systems predominantly rely on sparse representations like point clouds and voxels, dense scene representations are expected to play a critical role in downstream tasks such as navigation and obstacle avoidance. Moreover, neural representations such as NeRF, 3DGS, and foundation models are well-suited for integrating high-level semantic features and language-based priors, enabling more comprehensive 3D scene understanding and embodied intelligence. In this paper, we categorized the core modules of robotics into five parts (Perception, Mapping, Localization, Navigation, Manipulation). We start by presenting the standard formulation of different scene representation methods and comparing the advantages and disadvantages of scene representation across different modules. This survey is centered around the question: What is the best 3D scene representation for robotics? We then discuss the future development trends of 3D scene representations, with a particular focus on how the 3D Foundation Model could replace current methods as the unified solution for future robotic applications. The remaining challenges in fully realizing this model are also explored. We aim to offer a valuable resource for both newcomers and experienced researchers to explore the future of 3D scene representations and their application in robotics. We have published an open-source project on GitHub and will continue to add new works and technologies to this project.
>
---
#### [new 008] Crossing the Sim2Real Gap Between Simulation and Ground Testing to Space Deployment of Autonomous Free-flyer Control
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文解决空间机器人自主控制的仿真到现实（Sim2Real）迁移难题。针对地面训练与在轨部署间的差距，利用NVIDIA Omniverse仿真环境与课程学习，训练深度神经网络实现NASA Astrobee机器人的自主导航。首次在国际空间站成功演示基于强化学习的自由飞行控制，验证了地面训练政策可有效迁移到太空，为在轨服务、装配与制造（ISAM）提供关键技术支撑。**

- **链接: [https://arxiv.org/pdf/2512.03736v1](https://arxiv.org/pdf/2512.03736v1)**

> **作者:** Kenneth Stewart; Samantha Chapin; Roxana Leontie; Carl Glen Henshaw
>
> **备注:** published at iSpaRo 2025
>
> **摘要:** Reinforcement learning (RL) offers transformative potential for robotic control in space. We present the first on-orbit demonstration of RL-based autonomous control of a free-flying robot, the NASA Astrobee, aboard the International Space Station (ISS). Using NVIDIA's Omniverse physics simulator and curriculum learning, we trained a deep neural network to replace Astrobee's standard attitude and translation control, enabling it to navigate in microgravity. Our results validate a novel training pipeline that bridges the simulation-to-reality (Sim2Real) gap, utilizing a GPU-accelerated, scientific-grade simulation environment for efficient Monte Carlo RL training. This successful deployment demonstrates the feasibility of training RL policies terrestrially and transferring them to space-based applications. This paves the way for future work in In-Space Servicing, Assembly, and Manufacturing (ISAM), enabling rapid on-orbit adaptation to dynamic mission requirements.
>
---
#### [new 009] Cross-embodied Co-design for Dexterous Hands
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对机器人灵巧手设计与控制不协同的问题，提出跨体态协同设计框架。通过联合优化手部形态与控制策略，实现任务驱动的端到端设计、训练、制造与部署，可在24小时内完成新灵巧手的开发，显著提升操作灵活性。**

- **链接: [https://arxiv.org/pdf/2512.03743v1](https://arxiv.org/pdf/2512.03743v1)**

> **作者:** Kehlani Fay; Darin Anthony Djapri; Anya Zorin; James Clinton; Ali El Lahib; Hao Su; Michael T. Tolley; Sha Yi; Xiaolong Wang
>
> **摘要:** Dexterous manipulation is limited by both control and design, without consensus as to what makes manipulators best for performing dexterous tasks. This raises a fundamental challenge: how should we design and control robot manipulators that are optimized for dexterity? We present a co-design framework that learns task-specific hand morphology and complementary dexterous control policies. The framework supports 1) an expansive morphology search space including joint, finger, and palm generation, 2) scalable evaluation across the wide design space via morphology-conditioned cross-embodied control, and 3) real-world fabrication with accessible components. We evaluate the approach across multiple dexterous tasks, including in-hand rotation with simulation and real deployment. Our framework enables an end-to-end pipeline that can design, train, fabricate, and deploy a new robotic hand in under 24 hours. The full framework will be open-sourced and available on our website.
>
---
#### [new 010] Autonomous Reinforcement Learning Robot Control with Intel's Loihi 2 Neuromorphic Hardware
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究机器人自主控制任务，旨在解决传统深度神经网络在机器人实时控制中能耗高、延迟大的问题。工作包括将仿真训练的ReLU神经网络转化为适配英特尔Loihi 2神经形态芯片的脉冲Sigma-Delta神经网络，并成功部署于Astrobee机器人控制中，实现低延迟、低功耗推理，验证了神经形态硬件在机器人控制中的可行性。**

- **链接: [https://arxiv.org/pdf/2512.03911v1](https://arxiv.org/pdf/2512.03911v1)**

> **作者:** Kenneth Stewart; Roxana Leontie; Samantha Chapin; Joe Hays; Sumit Bam Shrestha; Carl Glen Henshaw
>
> **备注:** Submitted for review at NICE 2026 (Neuro-Inspired Computational Elements) conference
>
> **摘要:** We present an end-to-end pipeline for deploying reinforcement learning (RL) trained Artificial Neural Networks (ANNs) on neuromorphic hardware by converting them into spiking Sigma-Delta Neural Networks (SDNNs). We demonstrate that an ANN policy trained entirely in simulation can be transformed into an SDNN compatible with Intel's Loihi 2 architecture, enabling low-latency and energy-efficient inference. As a test case, we use an RL policy for controlling the Astrobee free-flying robot, similar to a previously hardware in space-validated controller. The policy, trained with Rectified Linear Units (ReLUs), is converted to an SDNN and deployed on Intel's Loihi 2, then evaluated in NVIDIA's Omniverse Isaac Lab simulation environment for closed-loop control of Astrobee's motion. We compare execution performance between GPU and Loihi 2. The results highlight the feasibility of using neuromorphic platforms for robotic control and establish a pathway toward energy-efficient, real-time neuromorphic computation in future space and terrestrial robotics applications.
>
---
#### [new 011] IM HERE: Interaction Model for Human Effort Based Robot Engagement
- **分类: cs.RO**

- **简介: 该论文提出IM HERE框架，旨在建模人机交互中的参与度。针对现有模型泛化性差、定义模糊的问题，通过努力导向的双边关系分析，将互动简化为焦点定位与四种状态，实现对社会行为的自动化分析与描述，推动自主系统在遵循社会规范下实现有效社交融合。**

- **链接: [https://arxiv.org/pdf/2512.03828v1](https://arxiv.org/pdf/2512.03828v1)**

> **作者:** Dominykas Strazdas; Magnus Jung; Jan Marquenie; Ingo Siegert; Ayoub Al-Hamadi
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** The effectiveness of human-robot interaction often hinges on the ability to cultivate engagement - a dynamic process of cognitive involvement that supports meaningful exchanges. Many existing definitions and models of engagement are either too vague or lack the ability to generalize across different contexts. We introduce IM HERE, a novel framework that models engagement effectively in human-human, human-robot, and robot-robot interactions. By employing an effort-based description of bilateral relationships between entities, we provide an accurate breakdown of relationship patterns, simplifying them to focus placement and four key states. This framework captures mutual relationships, group behaviors, and actions conforming to social norms, translating them into specific directives for autonomous systems. By integrating both subjective perceptions and objective states, the model precisely identifies and describes miscommunication. The primary objective of this paper is to automate the analysis, modeling, and description of social behavior, and to determine how autonomous systems can behave in accordance with social norms for full social integration while simultaneously pursuing their own social goals.
>
---
#### [new 012] Context-Triggered Contingency Games for Strategic Multi-Agent Interaction
- **分类: cs.RO**

- **简介: 该论文针对自主多智能体系统中长期战略与短期动态适应的平衡问题，提出上下文触发的应急博弈框架。通过两层架构结合时序逻辑策略模板与因子图求解器，实现安全、高效、实时的多智能体交互。在自动驾驶与机器人导航中验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2512.03639v1](https://arxiv.org/pdf/2512.03639v1)**

> **作者:** Kilian Schweppe; Anne-Kathrin Schmuck
>
> **摘要:** We address the challenge of reliable and efficient interaction in autonomous multi-agent systems, where agents must balance long-term strategic objectives with short-term dynamic adaptation. We propose context-triggered contingency games, a novel integration of strategic games derived from temporal logic specifications with dynamic contingency games solved in real time. Our two-layered architecture leverages strategy templates to guarantee satisfaction of high-level objectives, while a new factor-graph-based solver enables scalable, real-time model predictive control of dynamic interactions. The resulting framework ensures both safety and progress in uncertain, interactive environments. We validate our approach through simulations and hardware experiments in autonomous driving and robotic navigation, demonstrating efficient, reliable, and adaptive multi-agent interaction.
>
---
#### [new 013] Safety Reinforced Model Predictive Control (SRMPC): Improving MPC with Reinforcement Learning for Motion Planning in Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文针对自动驾驶中模型预测控制（MPC）因凸近似导致解局限于局部最优的问题，提出安全强化学习增强的MPC（SRMPC）。通过约束强化学习与状态相关拉格朗日乘子，实现安全轨迹探索，提升全局优化能力。实验表明，该方法在高速公路场景中优于传统MPC和SRL，兼顾安全性与性能。**

- **链接: [https://arxiv.org/pdf/2512.03774v1](https://arxiv.org/pdf/2512.03774v1)**

> **作者:** Johannes Fischer; Marlon Steiner; Ömer Sahin Tas; Christoph Stiller
>
> **摘要:** Model predictive control (MPC) is widely used for motion planning, particularly in autonomous driving. Real-time capability of the planner requires utilizing convex approximation of optimal control problems (OCPs) for the planner. However, such approximations confine the solution to a subspace, which might not contain the global optimum. To address this, we propose using safe reinforcement learning (SRL) to obtain a new and safe reference trajectory within MPC. By employing a learning-based approach, the MPC can explore solutions beyond the close neighborhood of the previous one, potentially finding global optima. We incorporate constrained reinforcement learning (CRL) to ensure safety in automated driving, using a handcrafted energy function-based safety index as the constraint objective to model safe and unsafe regions. Our approach utilizes a state-dependent Lagrangian multiplier, learned concurrently with the safe policy, to solve the CRL problem. Through experimentation in a highway scenario, we demonstrate the superiority of our approach over both MPC and SRL in terms of safety and performance measures.
>
---
#### [new 014] World Models for Autonomous Navigation of Terrestrial Robots from LIDAR Observations
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究基于LIDAR的自主导航任务，针对模型无关强化学习在高维传感器数据下的样本效率低问题，提出一种基于DreamerV3的模型基强化学习框架。通过MLP-VAE编码器将全分辨率LIDAR数据压缩为紧凑潜在表示，结合动态预测模型实现高效策略优化，在仿真环境中实现100%成功率，显著优于传统方法。**

- **链接: [https://arxiv.org/pdf/2512.03429v1](https://arxiv.org/pdf/2512.03429v1)**

> **作者:** Raul Steinmetz; Fabio Demo Rosa; Victor Augusto Kich; Jair Augusto Bottega; Ricardo Bedin Grando; Daniel Fernando Tello Gamarra
>
> **备注:** Accepted for publication in the Journal of Intelligent and Fuzzy Systems
>
> **摘要:** Autonomous navigation of terrestrial robots using Reinforcement Learning (RL) from LIDAR observations remains challenging due to the high dimensionality of sensor data and the sample inefficiency of model-free approaches. Conventional policy networks struggle to process full-resolution LIDAR inputs, forcing prior works to rely on simplified observations that reduce spatial awareness and navigation robustness. This paper presents a novel model-based RL framework built on top of the DreamerV3 algorithm, integrating a Multi-Layer Perceptron Variational Autoencoder (MLP-VAE) within a world model to encode high-dimensional LIDAR readings into compact latent representations. These latent features, combined with a learned dynamics predictor, enable efficient imagination-based policy optimization. Experiments on simulated TurtleBot3 navigation tasks demonstrate that the proposed architecture achieves faster convergence and higher success rate compared to model-free baselines such as SAC, DDPG, and TD3. It is worth emphasizing that the DreamerV3-based agent attains a 100% success rate across all evaluated environments when using the full dataset of the Turtlebot3 LIDAR (360 readings), while model-free methods plateaued below 85%. These findings demonstrate that integrating predictive world models with learned latent representations enables more efficient and robust navigation from high-dimensional sensory data.
>
---
#### [new 015] ContactRL: Safe Reinforcement Learning based Motion Planning for Contact based Human Robot Collaboration
- **分类: cs.RO**

- **简介: 该论文针对人机协作中的安全接触问题，提出ContactRL框架，通过力反馈将接触安全融入强化学习奖励函数，实现低接触力的自适应运动规划。结合能量型控制屏障函数保障部署安全，在仿真与真实UR3e机器人手递手任务中验证了高效性与安全性。**

- **链接: [https://arxiv.org/pdf/2512.03707v1](https://arxiv.org/pdf/2512.03707v1)**

> **作者:** Sundas Rafat Mulkana; Ronyu Yu; Tanaya Guha; Emma Li
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** In collaborative human-robot tasks, safety requires not only avoiding collisions but also ensuring safe, intentional physical contact. We present ContactRL, a reinforcement learning (RL) based framework that directly incorporates contact safety into the reward function through force feedback. This enables a robot to learn adaptive motion profiles that minimize human-robot contact forces while maintaining task efficiency. In simulation, ContactRL achieves a low safety violation rate of 0.2\% with a high task success rate of 87.7\%, outperforming state-of-the-art constrained RL baselines. In order to guarantee deployment safety, we augment the learned policy with a kinetic energy based Control Barrier Function (eCBF) shield. Real-world experiments on an UR3e robotic platform performing small object handovers from a human hand across 360 trials confirm safe contact, with measured normal forces consistently below 10N. These results demonstrate that ContactRL enables safe and efficient physical collaboration, thereby advancing the deployment of collaborative robots in contact-rich tasks.
>
---
#### [new 016] Multi-Agent Reinforcement Learning and Real-Time Decision-Making in Robotic Soccer for Virtual Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究多智能体强化学习在虚拟机器人足球中的实时决策问题。针对任务复杂性与可扩展性挑战，提出分层强化学习结合平均场理论的框架，实现高效协作与策略优化，在4v4仿真中显著提升进球数、控球率与传球准确率。**

- **链接: [https://arxiv.org/pdf/2512.03166v1](https://arxiv.org/pdf/2512.03166v1)**

> **作者:** Aya Taourirte; Md Sohag Mia
>
> **摘要:** The deployment of multi-agent systems in dynamic, adversarial environments like robotic soccer necessitates real-time decision-making, sophisticated cooperation, and scalable algorithms to avoid the curse of dimensionality. While Reinforcement Learning (RL) offers a promising framework, existing methods often struggle with the multi-granularity of tasks (long-term strategy vs. instant actions) and the complexity of large-scale agent interactions. This paper presents a unified Multi-Agent Reinforcement Learning (MARL) framework that addresses these challenges. First, we establish a baseline using Proximal Policy Optimization (PPO) within a client-server architecture for real-time action scheduling, with PPO demonstrating superior performance (4.32 avg. goals, 82.9% ball control). Second, we introduce a Hierarchical RL (HRL) structure based on the options framework to decompose the problem into a high-level trajectory planning layer (modeled as a Semi-Markov Decision Process) and a low-level action execution layer, improving global strategy (avg. goals increased to 5.26). Finally, to ensure scalability, we integrate mean-field theory into the HRL framework, simplifying many-agent interactions into a single agent vs. the population average. Our mean-field actor-critic method achieves a significant performance boost (5.93 avg. goals, 89.1% ball control, 92.3% passing accuracy) and enhanced training stability. Extensive simulations of 4v4 matches in the Webots environment validate our approach, demonstrating its potential for robust, scalable, and cooperative behavior in complex multi-agent domains.
>
---
#### [new 017] Driving is a Game: Combining Planning and Prediction with Bayesian Iterative Best Response
- **分类: cs.RO**

- **简介: 该论文针对自动驾驶中复杂交通场景下的规划与预测协同问题，提出贝叶斯迭代最优响应（BIBeR）框架。通过将先进预测模型融入迭代最优响应机制，实现车辆与周围代理的双向交互建模，提升在密集城市交通中的决策能力。**

- **链接: [https://arxiv.org/pdf/2512.03936v1](https://arxiv.org/pdf/2512.03936v1)**

> **作者:** Aron Distelzweig; Yiwei Wang; Faris Janjoš; Marcel Hallgarten; Mihai Dobre; Alexander Langmann; Joschka Boedecker; Johannes Betz
>
> **摘要:** Autonomous driving planning systems perform nearly perfectly in routine scenarios using lightweight, rule-based methods but still struggle in dense urban traffic, where lane changes and merges require anticipating and influencing other agents. Modern motion predictors offer highly accurate forecasts, yet their integration into planning is mostly rudimental: discarding unsafe plans. Similarly, end-to-end models offer a one-way integration that avoids the challenges of joint prediction and planning modeling under uncertainty. In contrast, game-theoretic formulations offer a principled alternative but have seen limited adoption in autonomous driving. We present Bayesian Iterative Best Response (BIBeR), a framework that unifies motion prediction and game-theoretic planning into a single interaction-aware process. BIBeR is the first to integrate a state-of-the-art predictor into an Iterative Best Response (IBR) loop, repeatedly refining the strategies of the ego vehicle and surrounding agents. This repeated best-response process approximates a Nash equilibrium, enabling bidirectional adaptation where the ego both reacts to and shapes the behavior of others. In addition, our proposed Bayesian confidence estimation quantifies prediction reliability and modulates update strength, more conservative under low confidence and more decisive under high confidence. BIBeR is compatible with modern predictors and planners, combining the transparency of structured planning with the flexibility of learned models. Experiments show that BIBeR achieves an 11% improvement over state-of-the-art planners on highly interactive interPlan lane-change scenarios, while also outperforming existing approaches on standard nuPlan benchmarks.
>
---
#### [new 018] MDE-AgriVLN: Agricultural Vision-and-Language Navigation with Monocular Depth Estimation
- **分类: cs.RO**

- **简介: 该论文聚焦农业视觉语言导航任务，解决单目视觉下农业机器人空间感知不足的问题。提出MDE-AgriVLN方法，通过单目深度估计模块生成深度特征，增强决策推理能力，在A2A基准上提升成功率至0.32，降低导航误差至4.08m，实现农业VLN领域领先性能。**

- **链接: [https://arxiv.org/pdf/2512.03958v1](https://arxiv.org/pdf/2512.03958v1)**

> **作者:** Xiaobei Zhao; Xingqi Lyu; Xiang Li
>
> **摘要:** Agricultural robots are serving as powerful assistants across a wide range of agricultural tasks, nevertheless, still heavily relying on manual operations or railway systems for movement. The AgriVLN method and the A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the agricultural domain, enabling a robot to navigate to a target position following a natural language instruction. Unlike human binocular vision, most agricultural robots are only given a single camera for monocular vision, which results in limited spatial perception. To bridge this gap, we present the method of Agricultural Vision-and-Language Navigation with Monocular Depth Estimation (MDE-AgriVLN), in which we propose the MDE module generating depth features from RGB images, to assist the decision-maker on reasoning. When evaluated on the A2A benchmark, our MDE-AgriVLN method successfully increases Success Rate from 0.23 to 0.32 and decreases Navigation Error from 4.43m to 4.08m, demonstrating the state-of-the-art performance in the agricultural VLN domain. Code: https://github.com/AlexTraveling/MDE-AgriVLN.
>
---
#### [new 019] Surfel-LIO: Fast LiDAR-Inertial Odometry with Pre-computed Surfels and Hierarchical Z-order Voxel Hashing
- **分类: cs.RO**

- **简介: 该论文针对LiDAR-Inertial Odometry（LIO）中的效率问题，提出Surfel-LIO方法。通过预计算的surfel与分层Z-order体素哈希结构，实现O(1)对应点快速检索，避免重复平面拟合，显著提升处理速度，同时保持高精度。**

- **链接: [https://arxiv.org/pdf/2512.03397v1](https://arxiv.org/pdf/2512.03397v1)**

> **作者:** Seungwon Choi; Dong-Gyu Park; Seo-Yeon Hwang; Tae-Wan Kim
>
> **摘要:** LiDAR-inertial odometry (LIO) is an active research area, as it enables accurate real-time state estimation in GPS-denied environments. Recent advances in map data structures and spatial indexing have significantly improved the efficiency of LIO systems. Nevertheless, we observe that two aspects may still leave room for improvement: (1) nearest neighbor search often requires examining multiple spatial units to gather sufficient points for plane fitting, and (2) plane parameters are typically recomputed at every iteration despite unchanged map geometry. Motivated by these observations, we propose Surfel-LIO, which employs a hierarchical voxel structure (hVox) with pre-computed surfel representation. This design enables O(1) correspondence retrieval without runtime neighbor enumeration or plane fitting, combined with Z-order curve encoding for cache-friendly spatial indexing. Experimental results on the M3DGR dataset demonstrate that our method achieves significantly faster processing speed compared to recent state-of-the-art methods while maintaining comparable state estimation accuracy. Our implementation is publicly available at https://github.com/93won/lidar_inertial_odometry.
>
---
#### [new 020] A Learning-based Control Methodology for Transitioning VTOL UAVs
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对VTOL无人机过渡阶段因旋翼倾转导致质心与推力方向变化带来的控制难题，提出一种基于强化学习的耦合控制方法。通过将巡航模式视为悬停的特例，实现位置与姿态的协同控制，有效减少振动，提升轨迹跟踪精度与系统适应性。**

- **链接: [https://arxiv.org/pdf/2512.03548v1](https://arxiv.org/pdf/2512.03548v1)**

> **作者:** Zexin Lin; Yebin Zhong; Hanwen Wan; Jiu Cheng; Zhenglong Sun; Xiaoqiang Ji
>
> **摘要:** Transition control poses a critical challenge in Vertical Take-Off and Landing Unmanned Aerial Vehicle (VTOL UAV) development due to the tilting rotor mechanism, which shifts the center of gravity and thrust direction during transitions. Current control methods' decoupled control of altitude and position leads to significant vibration, and limits interaction consideration and adaptability. In this study, we propose a novel coupled transition control methodology based on reinforcement learning (RL) driven controller. Besides, contrasting to the conventional phase-transition approach, the ST3M method demonstrates a new perspective by treating cruise mode as a special case of hover. We validate the feasibility of applying our method in simulation and real-world environments, demonstrating efficient controller development and migration while accurately controlling UAV position and attitude, exhibiting outstanding trajectory tracking and reduced vibrations during the transition process.
>
---
#### [new 021] GRAND: Guidance, Rebalancing, and Assignment for Networked Dispatch in Multi-Agent Path Finding
- **分类: cs.RO; cs.LG; cs.MA**

- **简介: 该论文针对大规模机器人仓库中的长期多智能体拾取配送（MAPD）任务，提出GRAND方法。通过图神经网络提供全局引导，结合最小费用流与局部分配优化，实现高效调度。在LRR基准上，相较2024年冠军方案提升10%吞吐量，保持实时性，有效缓解拥堵，为大规模调度提供可扩展解决方案。**

- **链接: [https://arxiv.org/pdf/2512.03194v1](https://arxiv.org/pdf/2512.03194v1)**

> **作者:** Johannes Gaber; Meshal Alharbi; Daniele Gammelli; Gioele Zardini
>
> **摘要:** Large robot fleets are now common in warehouses and other logistics settings, where small control gains translate into large operational impacts. In this article, we address task scheduling for lifelong Multi-Agent Pickup-and-Delivery (MAPD) and propose a hybrid method that couples learning-based global guidance with lightweight optimization. A graph neural network policy trained via reinforcement learning outputs a desired distribution of free agents over an aggregated warehouse graph. This signal is converted into region-to-region rebalancing through a minimum-cost flow, and finalized by small, local assignment problems, preserving accuracy while keeping per-step latency within a 1 s compute budget. On congested warehouse benchmarks from the League of Robot Runners (LRR) with up to 500 agents, our approach improves throughput by up to 10% over the 2024 winning scheduler while maintaining real-time execution. The results indicate that coupling graph-structured learned guidance with tractable solvers reduces congestion and yields a practical, scalable blueprint for high-throughput scheduling in large fleets.
>
---
#### [new 022] Autonomous Planning In-space Assembly Reinforcement-learning free-flYer (APIARY) International Space Station Astrobee Testing
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文研究基于强化学习的太空自由飞行机器人自主控制任务，旨在解决零重力环境下机器人精准、自适应控制难题。团队在地面模拟与国际空间站Astrobee机器人上验证了采用PPO算法训练的6自由度控制策略，实现了首次在轨强化学习控制，推动了航天机器人快速部署与智能自主发展。**

- **链接: [https://arxiv.org/pdf/2512.03729v1](https://arxiv.org/pdf/2512.03729v1)**

> **作者:** Samantha Chapin; Kenneth Stewart; Roxana Leontie; Carl Glen Henshaw
>
> **备注:** iSpaRo 2025, Best Paper Award in Orbital Robotics
>
> **摘要:** The US Naval Research Laboratory's (NRL's) Autonomous Planning In-space Assembly Reinforcement-learning free-flYer (APIARY) experiment pioneers the use of reinforcement learning (RL) for control of free-flying robots in the zero-gravity (zero-G) environment of space. On Tuesday, May 27th 2025 the APIARY team conducted the first ever, to our knowledge, RL control of a free-flyer in space using the NASA Astrobee robot on-board the International Space Station (ISS). A robust 6-degrees of freedom (DOF) control policy was trained using an actor-critic Proximal Policy Optimization (PPO) network within the NVIDIA Isaac Lab simulation environment, randomizing over goal poses and mass distributions to enhance robustness. This paper details the simulation testing, ground testing, and flight validation of this experiment. This on-orbit demonstration validates the transformative potential of RL for improving robotic autonomy, enabling rapid development and deployment (in minutes to hours) of tailored behaviors for space exploration, logistics, and real-time mission needs.
>
---
#### [new 023] RoboScape-R: Unified Reward-Observation World Models for Generalizable Robotics Training via RL
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人泛化能力不足的问题，提出RoboScape-R框架，利用世界模型生成内生奖励信号，构建通用训练环境。通过统一的奖励-观测世界模型，克服传统RL依赖人工奖励的局限，显著提升策略在跨场景下的泛化性能，实验显示平均性能提升37.5%。**

- **链接: [https://arxiv.org/pdf/2512.03556v1](https://arxiv.org/pdf/2512.03556v1)**

> **作者:** Yinzhou Tang; Yu Shang; Yinuo Chen; Bingwen Wei; Xin Zhang; Shu'ang Yu; Liangzhi Shi; Chao Yu; Chen Gao; Wei Wu; Yong Li
>
> **摘要:** Achieving generalizable embodied policies remains a key challenge. Traditional policy learning paradigms, including both Imitation Learning (IL) and Reinforcement Learning (RL), struggle to cultivate generalizability across diverse scenarios. While IL policies often overfit to specific expert trajectories, RL suffers from the inherent lack of a unified and general reward signal necessary for effective multi-scene generalization. We posit that the world model is uniquely capable of serving as a universal environment proxy to address this limitation. However, current world models primarily focus on their ability to predict observations and still rely on task-specific, handcrafted reward functions, thereby failing to provide a truly general training environment. Toward this problem, we propose RoboScape-R, a framework leveraging the world model to serve as a versatile, general-purpose proxy for the embodied environment within the RL paradigm. We introduce a novel world model-based general reward mechanism that generates ''endogenous'' rewards derived from the model's intrinsic understanding of real-world state transition dynamics. Extensive experiments demonstrate that RoboScape-R effectively addresses the limitations of traditional RL methods by providing an efficient and general training environment that substantially enhances the generalization capability of embodied policies. Our approach offers critical insights into utilizing the world model as an online training strategy and achieves an average 37.5% performance improvement over baselines under out-of-domain scenarios.
>
---
#### [new 024] OmniDexVLG: Learning Dexterous Grasp Generation from Vision Language Model-Guided Grasp Semantics, Taxonomy and Functional Affordance
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对灵巧抓取生成任务，解决多语义维度（抓取分类、接触语义、功能可及性）统一建模难题。提出OmniDexVLG框架，通过语义丰富的数据生成与多模态语义推理，实现语言与视觉联合引导下的多样化、语义一致的灵巧抓取生成，显著提升抓取多样性与语义一致性。**

- **链接: [https://arxiv.org/pdf/2512.03874v1](https://arxiv.org/pdf/2512.03874v1)**

> **作者:** Lei Zhang; Diwen Zheng; Kaixin Bai; Zhenshan Bing; Zoltan-Csaba Marton; Zhaopeng Chen; Alois Christian Knoll; Jianwei Zhang
>
> **备注:** Project Website: https://sites.google.com/view/omnidexvlg, 16 pages
>
> **摘要:** Dexterous grasp generation aims to produce grasp poses that align with task requirements and human interpretable grasp semantics. However, achieving semantically controllable dexterous grasp synthesis remains highly challenging due to the lack of unified modeling of multiple semantic dimensions, including grasp taxonomy, contact semantics, and functional affordance. To address these limitations, we present OmniDexVLG, a multimodal, semantics aware grasp generation framework capable of producing structurally diverse and semantically coherent dexterous grasps under joint language and visual guidance. Our approach begins with OmniDexDataGen, a semantic rich dexterous grasp dataset generation pipeline that integrates grasp taxonomy guided configuration sampling, functional affordance contact point sampling, taxonomy aware differential force closure grasp sampling, and physics based optimization and validation, enabling systematic coverage of diverse grasp types. We further introduce OmniDexReasoner, a multimodal grasp type semantic reasoning module that leverages multi agent collaboration, retrieval augmented generation, and chain of thought reasoning to infer grasp related semantics and generate high quality annotations that align language instructions with task specific grasp intent. Building upon these components, we develop a unified Vision Language Grasping generation model that explicitly incorporates grasp taxonomy, contact structure, and functional affordance semantics, enabling fine grained control over grasp synthesis from natural language instructions. Extensive experiments in simulation and real world object grasping and ablation studies demonstrate that our method substantially outperforms state of the art approaches in terms of grasp diversity, contact semantic diversity, functional affordance diversity, and semantic consistency.
>
---
#### [new 025] Prediction-Driven Motion Planning: Route Integration Strategies in Attention-Based Prediction Models
- **分类: cs.RO**

- **简介: 该论文研究自动驾驶中预测驱动的运动规划任务，旨在解决多智能体预测与导航目标不一致的问题。通过在注意力预测模型中融合车辆路线与目标位姿，提出多种导航信息集成策略，提升预测与规划的协同性与可行性。**

- **链接: [https://arxiv.org/pdf/2512.03756v1](https://arxiv.org/pdf/2512.03756v1)**

> **作者:** Marlon Steiner; Royden Wagner; Ömer Sahin Tas; Christoph Stiller
>
> **备注:** In Proceedings of the IEEE International Conference on Intelligent Transportation Systems (ITSC), Gold Coast, AUSTRALIA, 18-21 November 2025
>
> **摘要:** Combining motion prediction and motion planning offers a promising framework for enhancing interactions between automated vehicles and other traffic participants. However, this introduces challenges in conditioning predictions on navigation goals and ensuring stable, kinematically feasible trajectories. Addressing the former challenge, this paper investigates the extension of attention-based motion prediction models with navigation information. By integrating the ego vehicle's intended route and goal pose into the model architecture, we bridge the gap between multi-agent motion prediction and goal-based motion planning. We propose and evaluate several architectural navigation integration strategies to our model on the nuPlan dataset. Our results demonstrate the potential of prediction-driven motion planning, highlighting how navigation information can enhance both prediction and planning tasks. Our implementation is at: https://github.com/KIT-MRT/future-motion.
>
---
#### [new 026] AdaPower: Specializing World Foundation Models for Predictive Manipulation
- **分类: cs.RO**

- **简介: 该论文针对世界基础模型（WFM）在机器人精准控制中生成现实与控制精度不匹配的问题，提出AdaPower框架。通过时空间测试时训练与记忆持久化机制，轻量级适配WFM，提升长时序一致性，在不重训策略下使预训练视觉语言动作模型任务成功率提升41%，显著改善控制精度与效率。**

- **链接: [https://arxiv.org/pdf/2512.03538v1](https://arxiv.org/pdf/2512.03538v1)**

> **作者:** Yuhang Huang; Shilong Zou; Jiazhao Zhang; Xinwang Liu; Ruizhen Hu; Kai Xu
>
> **摘要:** World Foundation Models (WFMs) offer remarkable visual dynamics simulation capabilities, yet their application to precise robotic control remains limited by the gap between generative realism and control-oriented precision. While existing approaches use WFMs as synthetic data generators, they suffer from high computational costs and underutilization of pre-trained VLA policies. We introduce \textbf{AdaPower} (\textbf{Ada}pt and Em\textbf{power}), a lightweight adaptation framework that transforms general-purpose WFMs into specialist world models through two novel components: Temporal-Spatial Test-Time Training (TS-TTT) for inference-time adaptation and Memory Persistence (MP) for long-horizon consistency. Integrated within a Model Predictive Control framework, our adapted world model empowers pre-trained VLAs, achieving over 41\% improvement in task success rates on LIBERO benchmarks without policy retraining, while preserving computational efficiency and generalist capabilities.
>
---
#### [new 027] A Modular Architecture Design for Autonomous Driving Racing in Controlled Environments
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对封闭环境下的自动驾驶竞速任务，提出一种模块化架构。通过独立运行的感知、定位、路径规划与控制子系统，构建数据协同的流水线结构，实现高精度实时自主导航，解决系统集成与实时性难题。**

- **链接: [https://arxiv.org/pdf/2512.03886v1](https://arxiv.org/pdf/2512.03886v1)**

> **作者:** Brais Fontan-Costas; M. Diaz-Cacho; Ruben Fernandez-Boullon; Manuel Alonso-Carracedo; Javier Perez-Robles
>
> **摘要:** This paper presents an Autonomous System (AS) architecture for vehicles in a closed circuit. The AS performs precision tasks including computer vision for environment perception, positioning and mapping for accurate localization, path planning for optimal trajectory generation, and control for precise vehicle actuation. Each subsystem operates independently while connecting data through a cohesive pipeline architecture. The system implements a modular design that combines state-of-the-art technologies for real-time autonomous navigation in controlled environments.
>
---
#### [new 028] KALIKO: Kalman-Implicit Koopman Operator Learning For Prediction of Nonlinear Dynamical Systems
- **分类: cs.RO**

- **简介: 该论文针对非线性动力系统长时程预测难题，提出KALIKO方法。通过隐式利用卡尔曼滤波学习潜在状态表示，避免显式编码，实现全局线性化建模。在高维波动数据与闭环控制任务中均优于现有方法，提升了预测精度与控制稳定性。**

- **链接: [https://arxiv.org/pdf/2512.03256v1](https://arxiv.org/pdf/2512.03256v1)**

> **作者:** Albert H. Li; Ivan Dario Jimenez Rodriguez; Joel W. Burdick; Yisong Yue; Aaron D. Ames
>
> **摘要:** Long-horizon dynamical prediction is fundamental in robotics and control, underpinning canonical methods like model predictive control. Yet, many systems and disturbance phenomena are difficult to model due to effects like nonlinearity, chaos, and high-dimensionality. Koopman theory addresses this by modeling the linear evolution of embeddings of the state under an infinite-dimensional linear operator that can be approximated with a suitable finite basis of embedding functions, effectively trading model nonlinearity for representational complexity. However, explicitly computing a good choice of basis is nontrivial, and poor choices may cause inaccurate forecasts or overfitting. To address this, we present Kalman-Implicit Koopman Operator (KALIKO) Learning, a method that leverages the Kalman filter to implicitly learn embeddings corresponding to latent dynamics without requiring an explicit encoder. KALIKO produces interpretable representations consistent with both theory and prior works, yielding high-quality reconstructions and inducing a globally linear latent dynamics. Evaluated on wave data generated by a high-dimensional PDE, KALIKO surpasses several baselines in open-loop prediction and in a demanding closed-loop simulated control task: stabilizing an underactuated manipulator's payload by predicting and compensating for strong wave disturbances.
>
---
#### [new 029] Multimodal Control of Manipulators: Coupling Kinematics and Vision for Self-Driving Laboratory Operations
- **分类: cs.RO**

- **简介: 该论文研究冗余机械臂在自驱动实验室操作中的多模态运动规划，旨在解决轨迹规划与逆运动学求解的协同问题。通过RRT*算法生成轨迹，结合螺旋理论求解正运动学，并采用三种雅可比方法（JT、PI、DLS）进行逆解计算，分析平滑性、误差及运动特性，评估不同方法的适用性。**

- **链接: [https://arxiv.org/pdf/2512.03630v1](https://arxiv.org/pdf/2512.03630v1)**

> **作者:** Shifa Sulaiman; Amarnath H; Simon Bogh; Naresh Marturi
>
> **摘要:** Motion planning schemes are used for planning motions of a manipulator from an initial pose to a final pose during a task execution. A motion planning scheme generally comprises of a trajectory planning method and an inverse kinematic solver to determine trajectories and joints solutions respectively. In this paper, 3 motion planning schemes developed based on Jacobian methods are implemented to traverse a redundant manipulator with a coupled finger gripper through given trajectories. RRT* algorithm is used for planning trajectories and screw theory based forward kinematic equations are solved for determining joint solutions of the manipulator and gripper. Inverse solutions are computed separately using 3 Jacobian based methods such as Jacobian Transpose (JT), Pseudo Inverse (PI), and Damped Least Square (DLS) methods. Space Jacobian and manipulability measurements of the manipulator and gripper are obtained using screw theory formulations. Smoothness and RMSE error of generated trajectories and velocity continuity, acceleration profile, jerk, and snap values of joint motions are analysed for determining an efficient motion planning method for a given task. Advantages and disadvantages of the proposed motion planning schemes mentioned above are analysed using simulation studies to determine a suitable inverse solution technique for the tasks.
>
---
#### [new 030] Artificial Microsaccade Compensation: Stable Vision for an Ornithopter
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对尾翼式扑翼机因高频振动（12–20 Hz）导致视频抖动的问题，提出“人工微跳变补偿”方法。通过优化SO(3)空间的三维旋转，实时稳定视频，消除畸变，提升视觉质量。相比商用软件Adobe Premier Pro，本方法效果更优且可实时运行。**

- **链接: [https://arxiv.org/pdf/2512.03995v1](https://arxiv.org/pdf/2512.03995v1)**

> **作者:** Levi Burner; Guido de Croon; Yiannis Aloimonos
>
> **备注:** 29 pages, 5 figures, 2 tables, under review
>
> **摘要:** Animals with foveated vision, including humans, experience microsaccades, small, rapid eye movements that they are not aware of. Inspired by this phenomenon, we develop a method for "Artificial Microsaccade Compensation". It can stabilize video captured by a tailless ornithopter that has resisted attempts to use camera-based sensing because it shakes at 12-20 Hz. Our approach minimizes changes in image intensity by optimizing over 3D rotation represented in SO(3). This results in a stabilized video, computed in real time, suitable for human viewing, and free from distortion. When adapted to hold a fixed viewing orientation, up to occasional saccades, it can dramatically reduce inter-frame motion while also benefiting from an efficient recursive update. When compared to Adobe Premier Pro's warp stabilizer, which is widely regarded as the best commercial video stabilization software available, our method achieves higher quality results while also running in real time.
>
---
#### [new 031] MPCFormer: A physics-informed data-driven approach for explainable socially-aware autonomous driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对自动驾驶在复杂交互场景中缺乏类人行为的问题，提出MPCFormer模型。通过物理信息引导的Transformer架构学习多车社交交互动态，结合MPC框架实现可解释、安全的决策。在NGSIM数据集上验证，显著提升轨迹预测精度与规划成功率，降低碰撞率。**

- **链接: [https://arxiv.org/pdf/2512.03795v1](https://arxiv.org/pdf/2512.03795v1)**

> **作者:** Jia Hu; Zhexi Lian; Xuerun Yan; Ruiang Bi; Dou Shen; Yu Ruan; Haoran Wang
>
> **备注:** 17 pages, 18 figures
>
> **摘要:** Autonomous Driving (AD) vehicles still struggle to exhibit human-like behavior in highly dynamic and interactive traffic scenarios. The key challenge lies in AD's limited ability to interact with surrounding vehicles, largely due to a lack of understanding the underlying mechanisms of social interaction. To address this issue, we introduce MPCFormer, an explainable socially-aware autonomous driving approach with physics-informed and data-driven coupled social interaction dynamics. In this model, the dynamics are formulated into a discrete space-state representation, which embeds physics priors to enhance modeling explainability. The dynamics coefficients are learned from naturalistic driving data via a Transformer-based encoder-decoder architecture. To the best of our knowledge, MPCFormer is the first approach to explicitly model the dynamics of multi-vehicle social interactions. The learned social interaction dynamics enable the planner to generate manifold, human-like behaviors when interacting with surrounding traffic. By leveraging the MPC framework, the approach mitigates the potential safety risks typically associated with purely learning-based methods. Open-looped evaluation on NGSIM dataset demonstrates that MPCFormer achieves superior social interaction awareness, yielding the lowest trajectory prediction errors compared with other state-of-the-art approach. The prediction achieves an ADE as low as 0.86 m over a long prediction horizon of 5 seconds. Close-looped experiments in highly intense interaction scenarios, where consecutive lane changes are required to exit an off-ramp, further validate the effectiveness of MPCFormer. Results show that MPCFormer achieves the highest planning success rate of 94.67%, improves driving efficiency by 15.75%, and reduces the collision rate from 21.25% to 0.5%, outperforming a frontier Reinforcement Learning (RL) based planner.
>
---
#### [new 032] MSG-Loc: Multi-Label Likelihood-based Semantic Graph Matching for Object-Level Global Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对未知物体类别与语义模糊下的全局定位问题，提出基于多标签似然的语义图匹配方法。通过多标签图表示捕捉语义上下文，利用上下文感知的似然传播增强对应关系，提升定位精度与鲁棒性，在真实与合成环境中验证了其有效性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.03522v1](https://arxiv.org/pdf/2512.03522v1)**

> **作者:** Gihyeon Lee; Jungwoo Lee; Juwon Kim; Young-Sik Shin; Younggun Cho
>
> **备注:** Accepted in IEEE Robotics and Automation Letters (2025)
>
> **摘要:** Robots are often required to localize in environments with unknown object classes and semantic ambiguity. However, when performing global localization using semantic objects, high semantic ambiguity intensifies object misclassification and increases the likelihood of incorrect associations, which in turn can cause significant errors in the estimated pose. Thus, in this letter, we propose a multi-label likelihood-based semantic graph matching framework for object-level global localization. The key idea is to exploit multi-label graph representations, rather than single-label alternatives, to capture and leverage the inherent semantic context of object observations. Based on these representations, our approach enhances semantic correspondence across graphs by combining the likelihood of each node with the maximum likelihood of its neighbors via context-aware likelihood propagation. For rigorous validation, data association and pose estimation performance are evaluated under both closed-set and open-set detection configurations. In addition, we demonstrate the scalability of our approach to large-vocabulary object categories in both real-world indoor scenes and synthetic environments.
>
---
#### [new 033] SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对视觉语言模型在空间推理中缺乏度量精度的问题，提出SpaceTools框架，通过双阶段交互式强化学习（DIRL）实现多工具协同。解决了传统方法依赖手工提示或固定工具链的局限，使模型能自主发现最优工具使用策略，在多个基准上取得领先性能，并成功应用于7-DOF机器人真实操作。**

- **链接: [https://arxiv.org/pdf/2512.04069v1](https://arxiv.org/pdf/2512.04069v1)**

> **作者:** Siyi Chen; Mikaela Angelina Uy; Chan Hee Song; Faisal Ladhak; Adithyavairavan Murali; Qing Qu; Stan Birchfield; Valts Blukis; Jonathan Tremblay
>
> **摘要:** Vision Language Models (VLMs) demonstrate strong qualitative visual understanding, but struggle with metrically precise spatial reasoning required for embodied applications. The agentic paradigm promises that VLMs can use a wide variety of tools that could augment these capabilities, such as depth estimators, segmentation models, and pose estimators. Yet it remains an open challenge how to realize this vision without solely relying on handcrafted prompting strategies or enforcing fixed, predefined tool pipelines that limit VLMs' ability to discover optimal tool-use patterns. Reinforcement Learning could overcome this gap, but has so far been limited to reasoning with a single visual tool due to the large search space in multi-tool reasoning. We introduce Double Interactive Reinforcement Learning (DIRL), a two-phase training framework where VLMs learn to coordinate multiple tools through interactive exploration and feedback. In the teaching phase, we combine demonstrations from a single tool specialist trained via interactive RL with traces from a frontier model using all tools. In the exploration phase, the model further refines multi-tool coordination through continued RL. Our model, SpaceTools, with tool-augmented spatial reasoning ability, achieves state-of-the-art performance on spatial understanding benchmarks (RoboSpatial-Home, BLINK, BOP-ASK) and demonstrates reliable real-world manipulation using a 7-DOF robot as a tool. DIRL provides substantial improvements over the vanilla SFT (+12% on RoboSpatial) and RL (+16% on RoboSpatial) baselines. Project page: https://spacetools.github.io/.
>
---
#### [new 034] Mobility Induced Sensitivity of UAV based Nodes to Jamming in Private 5G Airfield Networks An Experimental Study
- **分类: cs.NI; cs.CR; cs.RO**

- **简介: 该论文研究无人机在私有5G机场网络中受定向干扰时的抗干扰能力。针对无人机移动性引发的信号不稳定问题，通过实验分析速度、高度和飞行模式对CQI、SINR、RLF等指标的影响，评估链路稳定性与服务连续性，旨在提升无人机在复杂环境下的通信可靠性。**

- **链接: [https://arxiv.org/pdf/2512.03536v1](https://arxiv.org/pdf/2512.03536v1)**

> **作者:** Pavlo Mykytyn; Ronald Chitauro; Onur Yener; Peter Langendoerfer
>
> **备注:** 4 pages, 4 figures
>
> **摘要:** This work presents an experimental performance evaluation of a private 5G airfield network under controlled directional SDR jamming attacks targeting UAV-based UE nodes. Using a QualiPoc Android UE, mounted as a payload on a quadcopter UAV, we conducted a series of experiments to evaluate signal degradation, handover performance, and ser-vice stability in the presence of constant directional jamming. The conducted experiments aimed to examine the effects of varying travel speeds, altitudes, and moving patterns of a UAV-based UE to record and analyze the key physical-layer and network-layer metrics such as CQI, MCS, RSRP, SINR, BLER, Net PDSCH Throughput and RLF. The re-sults of this work describe the link stability and signal degradation dependencies, caused by the level of mobility of the UAV-based UE nodes during autonomous and automatic operation in private 5G Airfield networks
>
---
#### [new 035] Flux4D: Flow-based Unsupervised 4D Reconstruction
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出Flux4D，一种无监督的4D动态场景重建方法。针对现有方法在大规模动态场景重建中依赖标注、可扩展性差的问题，Flux4D通过光度损失与“尽可能静态”正则化，直接从原始数据预测3D高斯及其运动，实现高效、可扩展、泛化性强的无监督重建。**

- **链接: [https://arxiv.org/pdf/2512.03210v1](https://arxiv.org/pdf/2512.03210v1)**

> **作者:** Jingkang Wang; Henry Che; Yun Chen; Ze Yang; Lily Goli; Sivabalan Manivasagam; Raquel Urtasun
>
> **备注:** NeurIPS 2025. Project page: https://waabi.ai/flux4d/
>
> **摘要:** Reconstructing large-scale dynamic scenes from visual observations is a fundamental challenge in computer vision, with critical implications for robotics and autonomous systems. While recent differentiable rendering methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have achieved impressive photorealistic reconstruction, they suffer from scalability limitations and require annotations to decouple actor motion. Existing self-supervised methods attempt to eliminate explicit annotations by leveraging motion cues and geometric priors, yet they remain constrained by per-scene optimization and sensitivity to hyperparameter tuning. In this paper, we introduce Flux4D, a simple and scalable framework for 4D reconstruction of large-scale dynamic scenes. Flux4D directly predicts 3D Gaussians and their motion dynamics to reconstruct sensor observations in a fully unsupervised manner. By adopting only photometric losses and enforcing an "as static as possible" regularization, Flux4D learns to decompose dynamic elements directly from raw data without requiring pre-trained supervised models or foundational priors simply by training across many scenes. Our approach enables efficient reconstruction of dynamic scenes within seconds, scales effectively to large datasets, and generalizes well to unseen environments, including rare and unknown objects. Experiments on outdoor driving datasets show Flux4D significantly outperforms existing methods in scalability, generalization, and reconstruction quality.
>
---
#### [new 036] CSMapping: Scalable Crowdsourced Semantic Mapping and Topology Inference for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出CSMapping系统，解决低质量众包数据下自动驾驶地图构建的精度与可扩展性问题。通过训练隐空间扩散模型建立地图结构先验，结合约束优化实现鲁棒语义映射与拓扑中心线生成，显著提升地图质量并随数据量增长而持续改进。**

- **链接: [https://arxiv.org/pdf/2512.03510v1](https://arxiv.org/pdf/2512.03510v1)**

> **作者:** Zhijian Qiao; Zehuan Yu; Tong Li; Chih-Chung Chou; Wenchao Ding; Shaojie Shen
>
> **摘要:** Crowdsourcing enables scalable autonomous driving map construction, but low-cost sensor noise hinders quality from improving with data volume. We propose CSMapping, a system that produces accurate semantic maps and topological road centerlines whose quality consistently increases with more crowdsourced data. For semantic mapping, we train a latent diffusion model on HD maps (optionally conditioned on SD maps) to learn a generative prior of real-world map structure, without requiring paired crowdsourced/HD-map supervision. This prior is incorporated via constrained MAP optimization in latent space, ensuring robustness to severe noise and plausible completion in unobserved areas. Initialization uses a robust vectorized mapping module followed by diffusion inversion; optimization employs efficient Gaussian-basis reparameterization, projected gradient descent zobracket multi-start, and latent-space factor-graph for global consistency. For topological mapping, we apply confidence-weighted k-medoids clustering and kinematic refinement to trajectories, yielding smooth, human-like centerlines robust to trajectory variation. Experiments on nuScenes, Argoverse 2, and a large proprietary dataset achieve state-of-the-art semantic and topological mapping performance, with thorough ablation and scalability studies.
>
---
#### [new 037] When to Say "Hi" - Learn to Open a Conversation with an in-the-wild Dataset
- **分类: cs.HC; cs.RO**

- **简介: 该论文研究社交交互代理（SIA）的对话开启时机识别任务，旨在通过用户身体语言预测何时启动对话。针对服务场景中自然对话起始困难的问题，作者构建并验证了互动开启系统（IIS），基于201次真实场景数据训练模型，实现对对话开启者和时机的准确判断。**

- **链接: [https://arxiv.org/pdf/2512.03991v1](https://arxiv.org/pdf/2512.03991v1)**

> **作者:** Michael Schiffmann; Felix Struth; Sabina Jeschke; Anja Richert
>
> **备注:** 6 pages, 3 figures, 5 tables. This paper has been accepted for publication at IEEE ROMAN 2025
>
> **摘要:** The social capabilities of socially interactive agents (SIA) are a key to successful and smooth interactions between the user and the SIA. A successful start of the interaction is one of the essential factors for satisfying SIA interactions. For a service and information task in which the SIA helps with information, e.g. about the location, it is an important skill to master the opening of the conversation and to recognize which interlocutor opens the conversation and when. We are therefore investigating the extent to which the opening of the conversation can be trained using the user's body language as an input for machine learning to ensure smooth conversation starts for the interaction. In this paper we propose the Interaction Initiation System (IIS) which we developed, trained and validated using an in-the-wild data set. In a field test at the Deutsches Museum Bonn, a Furhat robot from Furhat Robotics was used as a service and information point. Over the period of use we collected the data of \textit{N} = 201 single user interactions for the training of the algorithms. We can show that the IIS, achieves a performance that allows the conclusion that this system is able to determine the greeting period and the opener of the interaction.
>
---
#### [new 038] Classification of User Satisfaction in HRI with Social Signals in the Wild
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互中的用户满意度分类任务，旨在自动评估用户对社交机器人服务的满意程度。针对传统依赖问卷或系统指标的局限性，研究基于真实场景下采集的视频与行为数据，利用身体姿态、面部表情和物理距离等社会信号，通过时间序列分析与机器学习模型实现用户满意度的自动分类，有效识别低满意度交互，无需人工标注。**

- **链接: [https://arxiv.org/pdf/2512.03945v1](https://arxiv.org/pdf/2512.03945v1)**

> **作者:** Michael Schiffmann; Sabina Jeschke; Anja Richert
>
> **备注:** 15 pages, 3 figures. This paper has been accepted for publication at ICSR+AI 2025
>
> **摘要:** Socially interactive agents (SIAs) are being used in various scenarios and are nearing productive deployment. Evaluating user satisfaction with SIAs' performance is a key factor in designing the interaction between the user and SIA. Currently, subjective user satisfaction is primarily assessed manually through questionnaires or indirectly via system metrics. This study examines the automatic classification of user satisfaction through analysis of social signals, aiming to enhance both manual and autonomous evaluation methods for SIAs. During a field trial at the Deutsches Museum Bonn, a Furhat Robotics head was employed as a service and information hub, collecting an "in-the-wild" dataset. This dataset comprises 46 single-user interactions, including questionnaire responses and video data. Our method focuses on automatically classifying user satisfaction based on time series classification. We use time series of social signal metrics derived from the body pose, time series of facial expressions, and physical distance. This study compares three feature engineering approaches on different machine learning models. The results confirm the method's effectiveness in reliably identifying interactions with low user satisfaction without the need for manually annotated datasets. This approach offers significant potential for enhancing SIA performance and user experience through automated feedback mechanisms.
>
---
#### [new 039] MUT3R: Motion-aware Updating Transformer for Dynamic 3D Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对动态3D重建中运动引起的伪影问题，提出MUT3R框架。通过分析自注意力图发现预训练模型隐含运动线索，设计无训练的注意力级门控模块，在早期抑制动态区域影响，提升时序一致性与相机位姿鲁棒性，实现无需微调的运动感知重建。**

- **链接: [https://arxiv.org/pdf/2512.03939v1](https://arxiv.org/pdf/2512.03939v1)**

> **作者:** Guole Shen; Tianchen Deng; Xingrui Qin; Nailin Wang; Jianyu Wang; Yanbo Wang; Yongtao Chen; Hesheng Wang; Jingchuan Wang
>
> **摘要:** Recent stateful recurrent neural networks have achieved remarkable progress on static 3D reconstruction but remain vulnerable to motion-induced artifacts, where non-rigid regions corrupt attention propagation between the spatial memory and image feature. By analyzing the internal behaviors of the state and image token updating mechanism, we find that aggregating self-attention maps across layers reveals a consistent pattern: dynamic regions are naturally down-weighted, exposing an implicit motion cue that the pretrained transformer already encodes but never explicitly uses. Motivated by this observation, we introduce MUT3R, a training-free framework that applies the attention-derived motion cue to suppress dynamic content in the early layers of the transformer during inference. Our attention-level gating module suppresses the influence of dynamic regions before their artifacts propagate through the feature hierarchy. Notably, we do not retrain or fine-tune the model; we let the pretrained transformer diagnose its own motion cues and correct itself. This early regulation stabilizes geometric reasoning in streaming scenarios and leads to improvements in temporal consistency and camera pose robustness across multiple dynamic benchmarks, offering a simple and training-free pathway toward motion-aware streaming reconstruction.
>
---
#### [new 040] NavMapFusion: Diffusion-based Fusion of Navigation Maps for Online Vectorized HD Map Construction
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出NavMapFusion，一种基于扩散模型的在线高精地图构建方法。针对传统导航地图分辨率低、更新滞后的问题，利用低精度先验地图与高精度传感器数据融合，通过迭代去噪实现地图更新。有效提升地图准确性与实时性，推动自动驾驶环境感知的可靠性。**

- **链接: [https://arxiv.org/pdf/2512.03317v1](https://arxiv.org/pdf/2512.03317v1)**

> **作者:** Thomas Monninger; Zihan Zhang; Steffen Staab; Sihao Ding
>
> **备注:** Accepted to 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2026)
>
> **摘要:** Accurate environmental representations are essential for autonomous driving, providing the foundation for safe and efficient navigation. Traditionally, high-definition (HD) maps are providing this representation of the static road infrastructure to the autonomous system a priori. However, because the real world is constantly changing, such maps must be constructed online from on-board sensor data. Navigation-grade standard-definition (SD) maps are widely available, but their resolution is insufficient for direct deployment. Instead, they can be used as coarse prior to guide the online map construction process. We propose NavMapFusion, a diffusion-based framework that performs iterative denoising conditioned on high-fidelity sensor data and on low-fidelity navigation maps. This paper strives to answer: (1) How can coarse, potentially outdated navigation maps guide online map construction? (2) What advantages do diffusion models offer for map fusion? We demonstrate that diffusion-based map construction provides a robust framework for map fusion. Our key insight is that discrepancies between the prior map and online perception naturally correspond to noise within the diffusion process; consistent regions reinforce the map construction, whereas outdated segments are suppressed. On the nuScenes benchmark, NavMapFusion conditioned on coarse road lines from OpenStreetMap data reaches a 21.4% relative improvement on 100 m, and even stronger improvements on larger perception ranges, while maintaining real-time capabilities. By fusing low-fidelity priors with high-fidelity sensor data, the proposed method generates accurate and up-to-date environment representations, guiding towards safer and more reliable autonomous driving. The code is available at https://github.com/tmonnin/navmapfusion
>
---
#### [new 041] PosA-VLA: Enhancing Action Generation via Pose-Conditioned Anchor Attention
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在具身任务中动作冗余、不精确的问题，提出PosA-VLA框架。通过姿态条件化的锚定注意力机制，引导模型聚焦任务相关区域，提升动作生成的精度与效率。无需额外感知模块，架构轻量，实现在复杂环境中的高效精准操作。**

- **链接: [https://arxiv.org/pdf/2512.03724v1](https://arxiv.org/pdf/2512.03724v1)**

> **作者:** Ziwen Li; Xin Wang; Hanlue Zhang; Runnan Chen; Runqi Lin; Xiao He; Han Huang; Yandong Guo; Fakhri Karray; Tongliang Liu; Mingming Gong
>
> **摘要:** The Vision-Language-Action (VLA) models have demonstrated remarkable performance on embodied tasks and shown promising potential for real-world applications. However, current VLAs still struggle to produce consistent and precise target-oriented actions, as they often generate redundant or unstable motions along trajectories, limiting their applicability in time-sensitive scenarios.In this work, we attribute these redundant actions to the spatially uniform perception field of existing VLAs, which causes them to be distracted by target-irrelevant objects, especially in complex environments.To address this issue, we propose an efficient PosA-VLA framework that anchors visual attention via pose-conditioned supervision, consistently guiding the model's perception toward task-relevant regions. The pose-conditioned anchor attention mechanism enables the model to better align instruction semantics with actionable visual cues, thereby improving action generation precision and efficiency. Moreover, our framework adopts a lightweight architecture and requires no auxiliary perception modules (e.g., segmentation or grounding networks), ensuring efficient inference. Extensive experiments verify that our method executes embodied tasks with precise and time-efficient behavior across diverse robotic manipulation benchmarks and shows robust generalization in a variety of challenging environments.
>
---
#### [new 042] Variable-Impedance Muscle Coordination under Slow-Rate Control Frequencies and Limited Observation Conditions Evaluated through Legged Locomotion
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究仿人单足行走中的肌肉协调机制，旨在解决高阶控制器在低频控制与有限感知条件下的稳定性问题。通过构建分层控制器，利用可变阻抗肌肉协调模型，实现低频控制下稳定运动，验证了形态计算可降低高层控制器的反馈依赖，为运动控制设计提供新范式。**

- **链接: [https://arxiv.org/pdf/2512.03459v1](https://arxiv.org/pdf/2512.03459v1)**

> **作者:** Hidaka Asai; Tomoyuki Noda; Jun Morimoto
>
> **备注:** 12 pages, 11 figures. Submitted to IEEE Transactions on Systems, Man, and Cybernetics: Systems
>
> **摘要:** Human motor control remains agile and robust despite limited sensory information for feedback, a property attributed to the body's ability to perform morphological computation through muscle coordination with variable impedance. However, it remains unclear how such low-level mechanical computation reduces the control requirements of the high-level controller. In this study, we implement a hierarchical controller consisting of a high-level neural network trained by reinforcement learning and a low-level variable-impedance muscle coor dination model with mono- and biarticular muscles in monoped locomotion task. We systematically restrict the high-level controller by varying the control frequency and by introducing biologically inspired observation conditions: delayed, partial, and substituted observation. Under these conditions, we evaluate how the low-level variable-impedance muscle coordination contributes to learning process of high-level neural network. The results show that variable-impedance muscle coordination enables stable locomotion even under slow-rate control frequency and limited observation conditions. These findings demonstrate that the morphological computation of muscle coordination effectively offloads high-frequency feedback of the high-level controller and provide a design principle for the controller in motor control.
>
---
## 更新

#### [replaced 001] Magnetic Tactile-Driven Soft Actuator for Intelligent Grasping and Firmness Evaluation
- **分类: cs.RO**

- **简介: 该论文针对软体机器人缺乏集成触觉传感及形变干扰信号的问题，提出SoftMag磁致触觉驱动器。通过共用结构实现传感与驱动一体化，结合多物理场仿真与神经网络解耦算法，提升感知精度。构建双指抓取系统，实现力、位姿实时预测与物体硬度非破坏性评估，推动智能材料感知软体机器人发展。**

- **链接: [https://arxiv.org/pdf/2512.00907v2](https://arxiv.org/pdf/2512.00907v2)**

> **作者:** Chengjin Du; Federico Bernabei; Zhengyin Du; Sergio Decherchi; Matteo Lo Preti; Lucia Beccai
>
> **备注:** 25 pages, 24 figures
>
> **摘要:** Soft robots are powerful tools for manipulating delicate objects, yet their adoption is hindered by two gaps: the lack of integrated tactile sensing and sensor signal distortion caused by actuator deformations. This paper addresses these challenges by introducing the SoftMag actuator: a magnetic tactile-sensorized soft actuator. Unlike systems relying on attached sensors or treating sensing and actuation separately, SoftMag unifies them through a shared architecture while confronting the mechanical parasitic effect, where deformations corrupt tactile signals. A multiphysics simulation framework models this coupling, and a neural-network-based decoupling strategy removes the parasitic component, restoring sensing fidelity. Experiments including indentation, quasi-static and step actuation, and fatigue tests validate the actuator's performance and decoupling effectiveness. Building upon this foundation, the system is extended into a two-finger SoftMag gripper, where a multi-task neural network enables real-time prediction of tri-axial contact forces and position. Furthermore, a probing-based strategy estimates object firmness during grasping. Validation on apricots shows a strong correlation (Pearson r over 0.8) between gripper-estimated firmness and reference measurements, confirming the system's capability for non-destructive quality assessment. Results demonstrate that combining integrated magnetic sensing, learning-based correction, and real-time inference enables a soft robotic platform that adapts its grasp and quantifies material properties. The framework offers an approach for advancing sensorized soft actuators toward intelligent, material-aware robotics.
>
---
#### [replaced 002] LargeAD: Large-Scale Cross-Sensor Data Pretraining for Autonomous Driving
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文针对自动驾驶中3D场景理解不足的问题，提出LargeAD框架，通过视觉基础模型生成语义超像素并与LiDAR点云对齐，实现跨模态预训练。工作包括超像素生成、对比学习、时序一致性保持及多源数据训练，显著提升分割与检测性能。**

- **链接: [https://arxiv.org/pdf/2501.04005v3](https://arxiv.org/pdf/2501.04005v3)**

> **作者:** Lingdong Kong; Xiang Xu; Youquan Liu; Jun Cen; Runnan Chen; Wenwei Zhang; Liang Pan; Kai Chen; Ziwei Liu
>
> **备注:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
>
> **摘要:** Recent advancements in vision foundation models (VFMs) have revolutionized visual perception in 2D, yet their potential for 3D scene understanding, particularly in autonomous driving applications, remains underexplored. In this paper, we introduce LargeAD, a versatile and scalable framework designed for large-scale 3D pretraining across diverse real-world driving datasets. Our framework leverages VFMs to extract semantically rich superpixels from 2D images, which are aligned with LiDAR point clouds to generate high-quality contrastive samples. This alignment facilitates cross-modal representation learning, enhancing the semantic consistency between 2D and 3D data. We introduce several key innovations: (i) VFM-driven superpixel generation for detailed semantic representation, (ii) a VFM-assisted contrastive learning strategy to align multimodal features, (iii) superpoint temporal consistency to maintain stable representations across time, and (iv) multi-source data pretraining to generalize across various LiDAR configurations. Our approach achieves substantial gains over state-of-the-art methods in linear probing and fine-tuning for LiDAR-based segmentation and object detection. Extensive experiments on 11 large-scale multi-sensor datasets highlight our superior performance, demonstrating adaptability, efficiency, and robustness in real-world autonomous driving scenarios.
>
---
#### [replaced 003] FPC-VLA: A Vision-Language-Action Framework with a Supervisor for Failure Prediction and Correction
- **分类: cs.RO**

- **简介: 该论文针对机器人操作中任务灵活性差与失败无法预测的问题，提出FPC-VLA框架，通过引入视觉-语言-动作模型与监督模块，实现对动作可行性的动态评估与失败纠正。利用双流融合机制提升决策鲁棒性，在仿真与真实机器人上验证了其在零样本和微调场景下的优越性能，显著增强系统可靠性。**

- **链接: [https://arxiv.org/pdf/2509.04018v2](https://arxiv.org/pdf/2509.04018v2)**

> **作者:** Yifan Yang; Zhixiang Duan; Tianshi Xie; Fuyu Cao; Pinxi Shen; Peili Song; Piaopiao Jin; Guokang Sun; Shaoqing Xu; Yangwei You; Jingtai Liu
>
> **摘要:** Robotic manipulation is a fundamental component of automation. However, traditional perception-planning pipelines often fall short in open-ended tasks due to limited flexibility, while the architecture of a single end-to-end Vision-Language-Action (VLA) offers promising capabilities but lacks crucial mechanisms for anticipating and recovering from failure. To address these challenges, we propose FPC-VLA, a dual-model framework that integrates VLA with a supervisor for failure prediction and correction. The supervisor evaluates action viability through vision-language queries and generates corrective strategies when risks arise, trained efficiently without manual labeling. A dual-stream fusion module further refines actions by leveraging past predictions. Evaluation results on multiple simulation platforms (SIMPLER and LIBERO) and robot embodiments (WidowX, Google Robot, Franka) show that FPC-VLA outperforms state-of-the-art models in both zero-shot and fine-tuned settings. Successful real-world deployments on diverse, long-horizon tasks confirm FPC-VLA's strong generalization and practical utility for building more reliable autonomous systems.
>
---
#### [replaced 004] Anti-bullying Adaptive Cruise Control: A proactive right-of-way protection approach
- **分类: eess.SY; cs.RO**

- **简介: 该论文针对自适应巡航控制（ACC）在近距离切入场景下易受“路霸”行为影响的问题，提出抗路霸自适应巡航（AACC）方法。通过在线识别驾驶风格，构建基于斯塔克尔伯格博弈的交互式规划框架，实现对不同切入车辆行为的预判与主动保护，显著提升安全、舒适与效率，且满足实时性要求。**

- **链接: [https://arxiv.org/pdf/2412.12197v2](https://arxiv.org/pdf/2412.12197v2)**

> **作者:** Jia Hu; Zhexi Lian; Haoran Wang; Zihan Zhang; Ruoxi Qian; Duo Li; Jaehyun; So; Junnian Zheng
>
> **备注:** 15 pages, 19 figures
>
> **摘要:** Adaptive Cruise Control (ACC) systems have been widely commercialized in recent years. However, existing ACC systems remain vulnerable to close-range cut-ins, a behavior that resembles "road bullying". To address this issue, this research proposes an Anti-bullying Adaptive Cruise Control (AACC) approach, which is capable of proactively protecting right-of-way against such "road bullying" cut-ins. To handle diverse "road bullying" cut-in scenarios smoothly, the proposed approach first leverages an online Inverse Optimal Control (IOC) based algorithm for individual driving style identification. Then, based on Stackelberg competition, a game-theoretic-based motion planning framework is presented in which the identified individual driving styles are utilized to formulate cut-in vehicles' reaction functions. By integrating such reaction functions into the ego vehicle's motion planning, the ego vehicle could consider cut-in vehicles' all possible reactions to find its optimal right-of-way protection maneuver. To the best of our knowledge, this research is the first to model vehicles' interaction dynamics and develop an interactive planner that adapts cut-in vehicle's various driving styles. Simulation results show that the proposed approach can prevent "road bullying" cut-ins and be adaptive to different cut-in vehicles' driving styles. It can improve safety and comfort by up to 79.8% and 20.4%. The driving efficiency has benefits by up to 19.33% in traffic flow. The proposed approach can also adopt more flexible driving strategies. Furthermore, the proposed approach can support real-time field implementation by ensuring less than 50 milliseconds computation time.
>
---
#### [replaced 005] Nonlinear Oscillatory Response of Automated Vehicle Car-following: Theoretical Analysis with Traffic State and Control Input Limits
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究自动驾驶汽车跟驰系统的非线性振荡响应，针对传统方法忽略交通状态与控制输入饱和的问题，提出基于描述函数与增量描述函数的理论分析框架，通过分解轨迹并分析频率响应，实现对非线性系统振荡特性的准确建模与稳定性评估。**

- **链接: [https://arxiv.org/pdf/2505.24029v2](https://arxiv.org/pdf/2505.24029v2)**

> **作者:** Sixu Li; Yang Zhou
>
> **摘要:** This paper presents a framework grounded in the theory of describing function (DF) and incremental-input DF to theoretically analyze the nonlinear oscillatory response of automated vehicles (AVs) car-following (CF) amidst traffic oscillations, considering the limits of traffic state and control input. While prevailing approaches largely ignore these limits (i.e., saturation of acceleration/deceleration and speed) and focus on linear string stability analysis, this framework establishes a basis for theoretically analyzing the frequency response of AV systems with nonlinearities imposed by these limits. To this end, trajectories of CF pairs are decomposed into nominal and oscillatory trajectories, subsequently, the controlled AV system is repositioned within the oscillatory trajectory coordinates. Built on this base, DFs are employed to approximate the frequency responses of nonlinear saturation components by using their first harmonic output, thereby capturing the associated amplification ratio and phase shift. Considering the closed-loop nature of AV control systems, where system states and control input mutually influence each other, amplification ratios and phase shifts are balanced within the loop to ensure consistency. This balancing process may render multiple solutions, hence the incremental-input DF is further applied to identify the reasonable ones. The proposed method is validated by estimations from Simulink, and further comparisons with prevailing methods are conducted. Results confirm the alignment of our framework with Simulink results and exhibit its superior accuracy in analysis compared to the prevailing methods. Furthermore, the framework proves valuable in string stability analysis, especially when conventional linear methods offer misleading insights.
>
---
#### [replaced 006] Quaternion-Based Sliding Mode Control for Six Degrees of Freedom Flight Control of Quadrotors
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对四旋翼飞行器六自由度飞行控制问题，提出一种基于四元数的滑模控制方法。针对传统方法存在的姿态奇点、解缠绕问题及稳定性不足等问题，设计了内外环结构：外环位置控制器采用无坐标系方法生成期望轨迹，内环四元数姿态控制器利用四元数超球面特性，实现全局稳定且结构简洁，有效避免解缠绕，提升控制性能与效率。**

- **链接: [https://arxiv.org/pdf/2403.10934v3](https://arxiv.org/pdf/2403.10934v3)**

> **作者:** Amin Yazdanshenas; Reza Faieghi
>
> **摘要:** Despite extensive research on sliding mode control (SMC) design for quadrotors, the existing approaches suffer from certain limitations. Euler angle-based SMC formulations suffer from poor performance in high-pitch or -roll maneuvers. Quaternion-based SMC approaches have unwinding issues and complex architecture. Coordinate-free methods are slow and only almost globally stable. This paper presents a new six degrees of freedom SMC flight controller to address the above limitations. We use a cascaded architecture with a position controller in the outer loop and a quaternion-based attitude controller in the inner loop. The position controller generates the desired trajectory for the attitude controller using a coordinate-free approach. The quaternion-based attitude controller uses the natural characteristics of the quaternion hypersphere, featuring a simple structure while providing global stability and avoiding unwinding issues. We compare our controller with three other common control methods conducting challenging maneuvers like flip-over and high-speed trajectory tracking in the presence of model uncertainties and disturbances. Our controller consistently outperforms the benchmark approaches with less control effort and actuator saturation, offering highly effective and efficient flight control.
>
---
#### [replaced 007] DynamicCity: Large-Scale 4D Occupancy Generation from Dynamic Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DynamicCity，解决城市场景生成中动态性与大规模4D语义建模问题。通过可变自编码器与DiT扩散模型，构建高效HexPlane表示，实现高精度、大尺度动态4D场景生成，支持多种条件驱动应用，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2410.18084v3](https://arxiv.org/pdf/2410.18084v3)**

> **作者:** Hengwei Bian; Lingdong Kong; Haozhe Xie; Liang Pan; Yu Qiao; Ziwei Liu
>
> **备注:** ICLR 2025 Spotlight; 35 pages, 18 figures, 15 tables; Project Page at https://dynamic-city.github.io/
>
> **摘要:** Urban scene generation has been developing rapidly recently. However, existing methods primarily focus on generating static and single-frame scenes, overlooking the inherently dynamic nature of real-world driving environments. In this work, we introduce DynamicCity, a novel 4D occupancy generation framework capable of generating large-scale, high-quality dynamic 4D scenes with semantics. DynamicCity mainly consists of two key models. 1) A VAE model for learning HexPlane as the compact 4D representation. Instead of using naive averaging operations, DynamicCity employs a novel Projection Module to effectively compress 4D features into six 2D feature maps for HexPlane construction, which significantly enhances HexPlane fitting quality (up to 12.56 mIoU gain). Furthermore, we utilize an Expansion & Squeeze Strategy to reconstruct 3D feature volumes in parallel, which improves both network training efficiency and reconstruction accuracy than naively querying each 3D point (up to 7.05 mIoU gain, 2.06x training speedup, and 70.84% memory reduction). 2) A DiT-based diffusion model for HexPlane generation. To make HexPlane feasible for DiT generation, a Padded Rollout Operation is proposed to reorganize all six feature planes of the HexPlane as a squared 2D feature map. In particular, various conditions could be introduced in the diffusion or sampling process, supporting versatile 4D generation applications, such as trajectory- and command-driven generation, inpainting, and layout-conditioned generation. Extensive experiments on the CarlaSC and Waymo datasets demonstrate that DynamicCity significantly outperforms existing state-of-the-art 4D occupancy generation methods across multiple metrics. The code and models have been released to facilitate future research.
>
---
#### [replaced 008] Are you a robot? Detecting Autonomous Vehicles from Behavior Analysis
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出一种基于行为分析的自动驾驶车辆检测框架，旨在区分自动驾驶与人工驾驶车辆。通过摄像头图像和车辆状态信息，利用机器学习模型实现无须主动标识的自动识别。构建了NexusStreet数据集，实验表明视频分析准确率达80%，结合状态信息可提升至93%。**

- **链接: [https://arxiv.org/pdf/2403.09571v2](https://arxiv.org/pdf/2403.09571v2)**

> **作者:** Fabio Maresca; Filippo Grazioli; Antonio Albanese; Vincenzo Sciancalepore; Gianpiero Negri; Xavier Costa-Perez
>
> **摘要:** The tremendous hype around autonomous driving is eagerly calling for emerging and novel technologies to support advanced mobility use cases. As car manufactures keep developing SAE level 3+ systems to improve the safety and comfort of passengers, traffic authorities need to establish new procedures to manage the transition from human-driven to fully-autonomous vehicles while providing a feedback-loop mechanism to fine-tune envisioned autonomous systems. Thus, a way to automatically profile autonomous vehicles and differentiate those from human-driven ones is a must. In this paper, we present a fully-fledged framework that monitors active vehicles using camera images and state information in order to determine whether vehicles are autonomous, without requiring any active notification from the vehicles themselves. Essentially, it builds on the cooperation among vehicles, which share their data acquired on the road feeding a machine learning model to identify autonomous cars. We extensively tested our solution and created the NexusStreet dataset, by means of the CARLA simulator, employing an autonomous driving control agent and a steering wheel maneuvered by licensed drivers. Experiments show it is possible to discriminate the two behaviors by analyzing video clips with an accuracy of 80%, which improves up to 93% when the target state information is available. Lastly, we deliberately degraded the state to observe how the framework performs under non-ideal data collection conditions.
>
---
#### [replaced 009] DGFusion: Depth-Guided Sensor Fusion for Robust Semantic Perception
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文针对自动驾驶中多传感器语义感知的鲁棒性问题，提出深度引导的融合方法DGFusion。通过引入激光雷达提供的深度信息，构建深度感知特征，动态调整跨模态融合策略，提升在复杂场景下的分割性能。**

- **链接: [https://arxiv.org/pdf/2509.09828v2](https://arxiv.org/pdf/2509.09828v2)**

> **作者:** Tim Broedermannn; Christos Sakaridis; Luigi Piccinelli; Wim Abbeloos; Luc Van Gool
>
> **备注:** Code and models will be available at https://github.com/timbroed/DGFusion
>
> **摘要:** Robust semantic perception for autonomous vehicles relies on effectively combining multiple sensors with complementary strengths and weaknesses. State-of-the-art sensor fusion approaches to semantic perception often treat sensor data uniformly across the spatial extent of the input, which hinders performance when faced with challenging conditions. By contrast, we propose a novel depth-guided multimodal fusion method that upgrades condition-aware fusion by integrating depth information. Our network, DGFusion, poses multimodal segmentation as a multi-task problem, utilizing the lidar measurements, which are typically available in outdoor sensor suites, both as one of the model's inputs and as ground truth for learning depth. Our corresponding auxiliary depth head helps to learn depth-aware features, which are encoded into spatially varying local depth tokens that condition our attentive cross-modal fusion. Together with a global condition token, these local depth tokens dynamically adapt sensor fusion to the spatially varying reliability of each sensor across the scene, which largely depends on depth. In addition, we propose a robust loss for our depth, which is essential for learning from lidar inputs that are typically sparse and noisy in adverse conditions. Our method achieves state-of-the-art panoptic and semantic segmentation performance on the challenging MUSES and DeLiVER datasets. Code and models will be available at https://github.com/timbroed/DGFusion
>
---
#### [replaced 010] SMP: Reusable Score-Matching Motion Priors for Physics-Based Character Control
- **分类: cs.GR; cs.AI; cs.CV; cs.RO**

- **简介: 该论文针对物理角色控制中运动先验复用性差的问题，提出可重用的Score-Matching Motion Priors（SMP）。通过预训练运动扩散模型与分数蒸馏采样，构建任务无关的通用运动先验，可冻结复用于多种下游任务，实现风格迁移与组合，生成高质量自然动作。**

- **链接: [https://arxiv.org/pdf/2512.03028v2](https://arxiv.org/pdf/2512.03028v2)**

> **作者:** Yuxuan Mu; Ziyu Zhang; Yi Shi; Minami Matsumoto; Kotaro Imamura; Guy Tevet; Chuan Guo; Michael Taylor; Chang Shu; Pengcheng Xi; Xue Bin Peng
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** Data-driven motion priors that can guide agents toward producing naturalistic behaviors play a pivotal role in creating life-like virtual characters. Adversarial imitation learning has been a highly effective method for learning motion priors from reference motion data. However, adversarial priors, with few exceptions, need to be retrained for each new controller, thereby limiting their reusability and necessitating the retention of the reference motion data when training on downstream tasks. In this work, we present Score-Matching Motion Priors (SMP), which leverages pre-trained motion diffusion models and score distillation sampling (SDS) to create reusable task-agnostic motion priors. SMPs can be pre-trained on a motion dataset, independent of any control policy or task. Once trained, SMPs can be kept frozen and reused as general-purpose reward functions to train policies to produce naturalistic behaviors for downstream tasks. We show that a general motion prior trained on large-scale datasets can be repurposed into a variety of style-specific priors. Furthermore SMP can compose different styles to synthesize new styles not present in the original dataset. Our method produces high-quality motion comparable to state-of-the-art adversarial imitation learning methods through reusable and modular motion priors. We demonstrate the effectiveness of SMP across a diverse suite of control tasks with physically simulated humanoid characters. Video demo available at https://youtu.be/ravlZJteS20
>
---
#### [replaced 011] MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文针对机器人抓取任务中的策略学习问题，提出MP1方法。通过3D点云输入与均值流（MeanFlow）结合，实现单次网络评估生成动作轨迹，避免一致性约束和数值误差，提升精度与速度。引入轻量级分散损失增强泛化能力。在多个基准和真实场景中表现优异，推理速度远超现有方法。**

- **链接: [https://arxiv.org/pdf/2507.10543v5](https://arxiv.org/pdf/2507.10543v5)**

> **作者:** Juyi Sheng; Ziyi Wang; Peiming Li; Mengyuan Liu
>
> **备注:** This paper has been accepted by AAAI 2026
>
> **摘要:** In robot manipulation, robot learning has become a prevailing approach. However, generative models within this field face a fundamental trade-off between the slow, iterative sampling of diffusion models and the architectural constraints of faster Flow-based methods, which often rely on explicit consistency losses. To address these limitations, we introduce MP1, which pairs 3D point-cloud inputs with the MeanFlow paradigm to generate action trajectories in one network function evaluation (1-NFE). By directly learning the interval-averaged velocity via the "MeanFlow Identity", our policy avoids any additional consistency constraints. This formulation eliminates numerical ODE-solver errors during inference, yielding more precise trajectories. MP1 further incorporates CFG for improved trajectory controllability while retaining 1-NFE inference without reintroducing structural constraints. Because subtle scene-context variations are critical for robot learning, especially in few-shot learning, we introduce a lightweight Dispersive Loss that repels state embeddings during training, boosting generalization without slowing inference. We validate our method on the Adroit and Meta-World benchmarks, as well as in real-world scenarios. Experimental results show MP1 achieves superior average task success rates, outperforming DP3 by 10.2% and FlowPolicy by 7.3%. Its average inference time is only 6.8 ms-19x faster than DP3 and nearly 2x faster than FlowPolicy. Our project page is available at https://mp1-2254.github.io/, and the code can be accessed at https://github.com/LogSSim/MP1.
>
---
#### [replaced 012] SwarmDiffusion: End-To-End Traversability-Guided Diffusion for Embodiment-Agnostic Navigation of Heterogeneous Robots
- **分类: cs.RO**

- **简介: 该论文针对异构机器人自主导航中视觉可通行性估计与轨迹生成分离、依赖人工提示和外部规划器的问题，提出SwarmDiffusion模型。它通过无提示的端到端扩散模型，直接从单张图像预测可通行性并生成可行轨迹，利用随机采样与贝塞尔平滑构建无规划器训练数据，实现跨平台泛化与快速推理。**

- **链接: [https://arxiv.org/pdf/2512.02851v2](https://arxiv.org/pdf/2512.02851v2)**

> **作者:** Iana Zhura; Sausar Karaf; Faryal Batool; Nipun Dhananjaya Weerakkodi Mudalige; Valerii Serpiva; Ali Alridha Abdulkarim; Aleksey Fedoseev; Didar Seyidov; Hajira Amjad; Dzmitry Tsetserukou
>
> **备注:** This work has been submitted for publication and is currently under review
>
> **摘要:** Visual traversability estimation is critical for autonomous navigation, but existing VLM-based methods rely on hand-crafted prompts, generalize poorly across embodiments, and output only traversability maps, leaving trajectory generation to slow external planners. We propose SwarmDiffusion, a lightweight end-to-end diffusion model that jointly predicts traversability and generates a feasible trajectory from a single RGB image. To remove the need for annotated or planner-produced paths, we introduce a planner-free trajectory construction pipeline based on randomized waypoint sampling, Bezier smoothing, and regularization enforcing connectivity, safety, directionality, and path thinness. This enables learning stable motion priors without demonstrations. SwarmDiffusion leverages VLM-derived supervision without prompt engineering and conditions the diffusion process on a compact embodiment state, producing physically consistent, traversable paths that transfer across different robot platforms. Across indoor environments and two embodiments (quadruped and aerial), the method achieves 80-100% navigation success and 0.09s inference, and adapts to a new robot using only-500 additional visual samples. It generalizes reliably to unseen environments in simulation and real-world trials, offering a scalable, prompt-free approach to unified traversability reasoning and trajectory generation.
>
---
#### [replaced 013] Supercomputing for High-speed Avoidance and Reactive Planning in Robots
- **分类: cs.RO; cs.DC**

- **简介: 该论文针对机器人在动态环境中实时避障的需求，提出SHARP系统，利用高性能计算（HPC）实现毫秒级响应。通过将多目标A*搜索并行化并部署于本地与远程HPC集群，验证了HPC离线规划在低延迟网络下的可行性，证明其可支持低于人类反应时间的避障控制，推动了混合控制架构的发展。**

- **链接: [https://arxiv.org/pdf/2509.19486v2](https://arxiv.org/pdf/2509.19486v2)**

> **作者:** Kieran S. Lachmansingh; José R. González-Estrada; Ryan E. Grant; Matthew K. X. J. Pan
>
> **备注:** Error in the graph calculation
>
> **摘要:** This paper presents SHARP (Supercomputing for High-speed Avoidance and Reactive Planning), a proof-of-concept study demonstrating how high-performance computing (HPC) can enable millisecond-scale responsiveness in robotic control. While modern robots face increasing demands for reactivity in human--robot shared workspaces, onboard processors are constrained by size, power, and cost. Offloading to HPC offers massive parallelism for trajectory planning, but its feasibility for real-time robotics remains uncertain due to network latency and jitter. We evaluate SHARP in a stress-test scenario where a 7-DOF manipulator must dodge high-speed foam projectiles. Using a parallelized multi-goal A* search implemented with MPI on both local and remote HPC clusters, the system achieves mean planning latencies of 22.9 ms (local) and 30.0 ms (remote, ~300 km away), with avoidance success rates of 84% and 88%, respectively. These results show that when round-trip latency remains within the tens-of-milliseconds regime, HPC-side computation is no longer the bottleneck, enabling avoidance well below human reaction times. The SHARP results motivate hybrid control architectures: low-level reflexes remain onboard for safety, while bursty, high-throughput planning tasks are offloaded to HPC for scalability. By reporting per-stage timing and success rates, this study provides a reproducible template for assessing real-time feasibility of HPC-driven robotics. Collectively, SHARP reframes HPC offloading as a viable pathway toward dependable, reactive robots in dynamic environments.
>
---
#### [replaced 014] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中失败诊断与学习的难题，提出ViFailback框架，通过视觉符号提升故障诊断效率。构建了包含58,126个VQA对的真实世界失败数据集，并设立评估基准ViFailback-Bench。基于此，训练出ViFailback-8B模型，可生成可视化的纠正指导，实验证明其能有效帮助机器人从失败中恢复。**

- **链接: [https://arxiv.org/pdf/2512.02787v2](https://arxiv.org/pdf/2512.02787v2)**

> **作者:** Xianchao Zeng; Xinyu Zhou; Youcheng Li; Jiayou Shi; Tianle Li; Liangming Chen; Lei Ren; Yong-Lu Li
>
> **摘要:** Vision-Language-Action (VLA) models have recently achieved remarkable progress in robotic manipulation, yet they remain limited in failure diagnosis and learning from failures. Additionally, existing failure datasets are mostly generated programmatically in simulation, which limits their generalization to the real world. In light of these, we introduce ViFailback, a framework designed to diagnose robotic manipulation failures and provide both textual and visual correction guidance. Our framework utilizes explicit visual symbols to enhance annotation efficiency. We further release the ViFailback dataset, a large-scale collection of 58,126 Visual Question Answering (VQA) pairs along with their corresponding 5,202 real-world manipulation trajectories. Based on the dataset, we establish ViFailback-Bench, a benchmark of 11 fine-grained VQA tasks designed to assess the failure diagnosis and correction abilities of Vision-Language Models (VLMs), featuring ViFailback-Bench Lite for closed-ended and ViFailback-Bench Hard for open-ended evaluation. To demonstrate the effectiveness of our framework, we built the ViFailback-8B VLM, which not only achieves significant overall performance improvement on ViFailback-Bench but also generates visual symbols for corrective action guidance. Finally, by integrating ViFailback-8B with a VLA model, we conduct real-world robotic experiments demonstrating its ability to assist the VLA model in recovering from failures. Project Website: https://x1nyuzhou.github.io/vifailback.github.io/
>
---
#### [replaced 015] 3D and 4D World Modeling: A Survey
- **分类: cs.CV; cs.RO**

- **简介: 该论文聚焦3D/4D世界建模任务，针对现有研究忽视原生3D/4D数据及缺乏统一定义与分类的问题，提出涵盖视频、占据网格和LiDAR的系统性分类，梳理数据集、评估指标与应用，总结挑战与方向，为该领域提供首个全面综述。**

- **链接: [https://arxiv.org/pdf/2509.07996v3](https://arxiv.org/pdf/2509.07996v3)**

> **作者:** Lingdong Kong; Wesley Yang; Jianbiao Mei; Youquan Liu; Ao Liang; Dekai Zhu; Dongyue Lu; Wei Yin; Xiaotao Hu; Mingkai Jia; Junyuan Deng; Kaiwen Zhang; Yang Wu; Tianyi Yan; Shenyuan Gao; Song Wang; Linfeng Li; Liang Pan; Yong Liu; Jianke Zhu; Wei Tsang Ooi; Steven C. H. Hoi; Ziwei Liu
>
> **备注:** Survey; 50 pages, 10 figures, 14 tables; GitHub Repo at https://github.com/worldbench/awesome-3d-4d-world-models
>
> **摘要:** World modeling has become a cornerstone in AI research, enabling agents to understand, represent, and predict the dynamic environments they inhabit. While prior work largely emphasizes generative methods for 2D image and video data, they overlook the rapidly growing body of work that leverages native 3D and 4D representations such as RGB-D imagery, occupancy grids, and LiDAR point clouds for large-scale scene modeling. At the same time, the absence of a standardized definition and taxonomy for ``world models'' has led to fragmented and sometimes inconsistent claims in the literature. This survey addresses these gaps by presenting the first comprehensive review explicitly dedicated to 3D and 4D world modeling and generation. We establish precise definitions, introduce a structured taxonomy spanning video-based (VideoGen), occupancy-based (OccGen), and LiDAR-based (LiDARGen) approaches, and systematically summarize datasets and evaluation metrics tailored to 3D/4D settings. We further discuss practical applications, identify open challenges, and highlight promising research directions, aiming to provide a coherent and foundational reference for advancing the field. A systematic summary of existing literature is available at https://github.com/worldbench/awesome-3d-4d-world-models
>
---
#### [replaced 016] AugMapNet: Improving Spatial Latent Structure via BEV Grid Augmentation for Enhanced Vectorized Online HD Map Construction
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文针对自动驾驶中矢量高精地图的实时构建任务，提出AugMapNet模型。通过引入BEV特征网格增强，提升隐空间结构化程度，融合向量解码与密集空间监督，显著改善地图预测精度，尤其在大范围场景下表现优异。**

- **链接: [https://arxiv.org/pdf/2503.13430v2](https://arxiv.org/pdf/2503.13430v2)**

> **作者:** Thomas Monninger; Md Zafar Anwar; Stanislaw Antol; Steffen Staab; Sihao Ding
>
> **备注:** Accepted to 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2026)
>
> **摘要:** Autonomous driving requires understanding infrastructure elements, such as lanes and crosswalks. To navigate safely, this understanding must be derived from sensor data in real-time and needs to be represented in vectorized form. Learned Bird's-Eye View (BEV) encoders are commonly used to combine a set of camera images from multiple views into one joint latent BEV grid. Traditionally, from this latent space, an intermediate raster map is predicted, providing dense spatial supervision but requiring post-processing into the desired vectorized form. More recent models directly derive infrastructure elements as polylines using vectorized map decoders, providing instance-level information. Our approach, Augmentation Map Network (AugMapNet), proposes latent BEV feature grid augmentation, a novel technique that significantly enhances the latent BEV representation. AugMapNet combines vector decoding and dense spatial supervision more effectively than existing architectures while remaining easy to integrate compared to other hybrid approaches. It additionally benefits from extra processing on its latent BEV features. Experiments on nuScenes and Argoverse2 datasets demonstrate significant improvements on vectorized map prediction of up to 13.3% over the StreamMapNet baseline on 60 m range and greater improvements on larger ranges. We confirm transferability by applying our method to another baseline, SQD-MapNet, and find similar improvements. A detailed analysis of the latent BEV grid confirms a more structured latent space of AugMapNet and shows the value of our novel concept beyond pure performance improvement. The code can be found at https://github.com/tmonnin/augmapnet
>
---
#### [replaced 017] HybridWorldSim: A Scalable and Controllable High-fidelity Simulator for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶仿真中视点变化大时视觉失真、几何不一致的问题，提出HybridWorldSim框架，融合神经重建与生成模型，实现高保真、可控的动态场景仿真。构建MIRROR数据集用于基准测试，实验表明其显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.22187v3](https://arxiv.org/pdf/2511.22187v3)**

> **作者:** Qiang Li; Yingwenqi Jiang; Tuoxi Li; Duyu Chen; Xiang Feng; Yucheng Ao; Shangyue Liu; Xingchen Yu; Youcheng Cai; Yumeng Liu; Yuexin Ma; Xin Hu; Li Liu; Yu Zhang; Linkun Xu; Bingtao Gao; Xueyuan Wang; Shuchang Zhou; Xianming Liu; Ligang Liu
>
> **备注:** Project page: https://hybridworldsim.github.io/
>
> **摘要:** Realistic and controllable simulation is critical for advancing end-to-end autonomous driving, yet existing approaches often struggle to support novel view synthesis under large viewpoint changes or to ensure geometric consistency. We introduce HybridWorldSim, a hybrid simulation framework that integrates multi-traversal neural reconstruction for static backgrounds with generative modeling for dynamic agents. This unified design addresses key limitations of previous methods, enabling the creation of diverse and high-fidelity driving scenarios with reliable visual and spatial consistency. To facilitate robust benchmarking, we further release a new multi-traversal dataset MIRROR that captures a wide range of routes and environmental conditions across different cities. Extensive experiments demonstrate that HybridWorldSim surpasses previous state-of-the-art methods, providing a practical and scalable solution for high-fidelity simulation and a valuable resource for research and development in autonomous driving.
>
---
