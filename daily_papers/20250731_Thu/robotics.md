# 机器人 cs.RO

- **最新发布 18 篇**

- **更新 29 篇**

## 最新发布

#### [new 001] Improving Generalization Ability of Robotic Imitation Learning by Resolving Causal Confusion in Observations
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人模仿学习任务，旨在解决模仿学习中因果混淆导致的泛化能力差问题。作者提出一种因果结构学习框架，无需解耦特征表示，可嵌入现有架构如Action Chunking Transformer。实验表明该方法有效提升在模拟环境中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.22380v1](http://arxiv.org/pdf/2507.22380v1)**

> **作者:** Yifei Chen; Yuzhe Zhang; Giovanni D'urso; Nicholas Lawrance; Brendan Tidd
>
> **备注:** 13 pages
>
> **摘要:** Recent developments in imitation learning have considerably advanced robotic manipulation. However, current techniques in imitation learning can suffer from poor generalization, limiting performance even under relatively minor domain shifts. In this work, we aim to enhance the generalization capabilities of complex imitation learning algorithms to handle unpredictable changes from the training environments to deployment environments. To avoid confusion caused by observations that are not relevant to the target task, we propose to explicitly learn the causal relationship between observation components and expert actions, employing a framework similar to [6], where a causal structural function is learned by intervention on the imitation learning policy. Disentangling the feature representation from image input as in [6] is hard to satisfy in complex imitation learning process in robotic manipulation, we theoretically clarify that this requirement is not necessary in causal relationship learning. Therefore, we propose a simple causal structure learning framework that can be easily embedded in recent imitation learning architectures, such as the Action Chunking Transformer [31]. We demonstrate our approach using a simulation of the ALOHA [31] bimanual robot arms in Mujoco, and show that the method can considerably mitigate the generalization problem of existing complex imitation learning algorithms.
>
---
#### [new 002] Explainable Deep Anomaly Detection with Sequential Hypothesis Testing for Robotic Sewer Inspection
- **分类: cs.RO**

- **简介: 该论文属于异常检测任务，旨在解决传统污水管道故障检测效率低、易出错的问题。工作内容是提出一种结合可解释深度学习与序贯假设检验的时空分析系统，实现鲁棒的自动化污水管道异常检测。**

- **链接: [http://arxiv.org/pdf/2507.22546v1](http://arxiv.org/pdf/2507.22546v1)**

> **作者:** Alex George; Will Shepherd; Simon Tait; Lyudmila Mihaylova; Sean R. Anderson
>
> **摘要:** Sewer pipe faults, such as leaks and blockages, can lead to severe consequences including groundwater contamination, property damage, and service disruption. Traditional inspection methods rely heavily on the manual review of CCTV footage collected by mobile robots, which is inefficient and susceptible to human error. To automate this process, we propose a novel system incorporating explainable deep learning anomaly detection combined with sequential probability ratio testing (SPRT). The anomaly detector processes single image frames, providing interpretable spatial localisation of anomalies, whilst the SPRT introduces temporal evidence aggregation, enhancing robustness against noise over sequences of image frames. Experimental results demonstrate improved anomaly detection performance, highlighting the benefits of the combined spatiotemporal analysis system for reliable and robust sewer inspection.
>
---
#### [new 003] Bayesian Optimization applied for accelerated Virtual Validation of the Autonomous Driving Function
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文属于自动驾驶验证任务，旨在解决传统方法验证耗时长、计算资源消耗大的问题。工作提出基于贝叶斯优化的框架，加速发现自动驾驶功能中的关键危险场景，有效减少仿真次数，并验证其在高维参数空间中的可扩展性。**

- **链接: [http://arxiv.org/pdf/2507.22769v1](http://arxiv.org/pdf/2507.22769v1)**

> **作者:** Satyesh Shanker Awasthi; Mohammed Irshadh Ismaaeel Sathyamangalam Imran; Stefano Arrigoni; Francesco Braghin
>
> **摘要:** Rigorous Verification and Validation (V&V) of Autonomous Driving Functions (ADFs) is paramount for ensuring the safety and public acceptance of Autonomous Vehicles (AVs). Current validation relies heavily on simulation to achieve sufficient test coverage within the Operational Design Domain (ODD) of a vehicle, but exhaustively exploring the vast parameter space of possible scenarios is computationally expensive and time-consuming. This work introduces a framework based on Bayesian Optimization (BO) to accelerate the discovery of critical scenarios. We demonstrate the effectiveness of the framework on an Model Predictive Controller (MPC)-based motion planner, showing that it identifies hazardous situations, such as off-road events, using orders of magnitude fewer simulations than brute-force Design of Experiments (DoE) methods. Furthermore, this study investigates the scalability of the framework in higher-dimensional parameter spaces and its ability to identify multiple, distinct critical regions within the ODD of the motion planner used as the case study .
>
---
#### [new 004] A Two-Stage Lightweight Framework for Efficient Land-Air Bimodal Robot Autonomous Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人自主导航任务，旨在解决陆空双模机器人导航中轨迹不优和计算量大的问题。提出了一个两阶段轻量框架，结合全局关键点预测与局部轨迹优化，实现高效、可实现的导航路径，减少了参数和能耗，支持实时运行与仿真到现实的零样本迁移。**

- **链接: [http://arxiv.org/pdf/2507.22473v1](http://arxiv.org/pdf/2507.22473v1)**

> **作者:** Yongjie Li; Zhou Liu; Wenshuai Yu; Zhangji Lu; Chenyang Wang; Fei Yu; Qingquan Li
>
> **备注:** IROS2025
>
> **摘要:** Land-air bimodal robots (LABR) are gaining attention for autonomous navigation, combining high mobility from aerial vehicles with long endurance from ground vehicles. However, existing LABR navigation methods are limited by suboptimal trajectories from mapping-based approaches and the excessive computational demands of learning-based methods. To address this, we propose a two-stage lightweight framework that integrates global key points prediction with local trajectory refinement to generate efficient and reachable trajectories. In the first stage, the Global Key points Prediction Network (GKPN) was used to generate a hybrid land-air keypoint path. The GKPN includes a Sobel Perception Network (SPN) for improved obstacle detection and a Lightweight Attention Planning Network (LAPN) to improves predictive ability by capturing contextual information. In the second stage, the global path is segmented based on predicted key points and refined using a mapping-based planner to create smooth, collision-free trajectories. Experiments conducted on our LABR platform show that our framework reduces network parameters by 14\% and energy consumption during land-air transitions by 35\% compared to existing approaches. The framework achieves real-time navigation without GPU acceleration and enables zero-shot transfer from simulation to reality during
>
---
#### [new 005] UniLegs: Universal Multi-Legged Robot Control through Morphology-Agnostic Policy Distillation
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决不同多足机器人形态下的通用控制问题。通过教师-学生框架，先训练特定形态的教师策略，再将其知识蒸馏至一个基于Transformer的学生策略，实现对多种腿部结构机器人的有效控制。**

- **链接: [http://arxiv.org/pdf/2507.22653v1](http://arxiv.org/pdf/2507.22653v1)**

> **作者:** Weijie Xi; Zhanxiang Cao; Chenlin Ming; Jianying Zheng; Guyue Zhou
>
> **备注:** 6 pages, 3 figures, IROS 2025
>
> **摘要:** Developing controllers that generalize across diverse robot morphologies remains a significant challenge in legged locomotion. Traditional approaches either create specialized controllers for each morphology or compromise performance for generality. This paper introduces a two-stage teacher-student framework that bridges this gap through policy distillation. First, we train specialized teacher policies optimized for individual morphologies, capturing the unique optimal control strategies for each robot design. Then, we distill this specialized expertise into a single Transformer-based student policy capable of controlling robots with varying leg configurations. Our experiments across five distinct legged morphologies demonstrate that our approach preserves morphology-specific optimal behaviors, with the Transformer architecture achieving 94.47\% of teacher performance on training morphologies and 72.64\% on unseen robot designs. Comparative analysis reveals that Transformer-based architectures consistently outperform MLP baselines by leveraging attention mechanisms to effectively model joint relationships across different kinematic structures. We validate our approach through successful deployment on a physical quadruped robot, demonstrating the practical viability of our morphology-agnostic control framework. This work presents a scalable solution for developing universal legged robot controllers that maintain near-optimal performance while generalizing across diverse morphologies.
>
---
#### [new 006] Operationalization of Scenario-Based Safety Assessment of Automated Driving Systems
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶安全评估任务，旨在解决如何实际操作基于场景数据库的安全评估流程，以完善NATM方法。论文探讨了如何利用场景数据库进行安全评估，并结合Horizon Europe项目的方法，推动NATM的全面实施。**

- **链接: [http://arxiv.org/pdf/2507.22433v1](http://arxiv.org/pdf/2507.22433v1)**

> **作者:** Olaf Op den Camp; Erwin de Gelder
>
> **备注:** Accepted for publication in proceedings of the 2025 IEEE International Automated Vehicle Validation Conference
>
> **摘要:** Before introducing an Automated Driving System (ADS) on the road at scale, the manufacturer must conduct some sort of safety assurance. To structure and harmonize the safety assurance process, the UNECE WP.29 Working Party on Automated/Autonomous and Connected Vehicles (GRVA) is developing the New Assessment/Test Method (NATM) that indicates what steps need to be taken for safety assessment of an ADS. In this paper, we will show how to practically conduct safety assessment making use of a scenario database, and what additional steps must be taken to fully operationalize the NATM. In addition, we will elaborate on how the use of scenario databases fits with methods developed in the Horizon Europe projects that focus on safety assessment following the NATM approach.
>
---
#### [new 007] Safety Evaluation of Motion Plans Using Trajectory Predictors as Forward Reachable Set Estimators
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于自动驾驶安全验证任务，旨在解决如何高效准确评估自动驾驶车辆运动规划的安全性问题。论文利用轨迹预测模型估计周围交通参与者前向可达集（FRS），通过凸优化和共形预测校准FRS，确保安全性判断的完备性和合理性，提升系统在真实场景中的可靠性。**

- **链接: [http://arxiv.org/pdf/2507.22389v1](http://arxiv.org/pdf/2507.22389v1)**

> **作者:** Kaustav Chakraborty; Zeyuan Feng; Sushant Veer; Apoorva Sharma; Wenhao Ding; Sever Topan; Boris Ivanovic; Marco Pavone; Somil Bansal
>
> **摘要:** The advent of end-to-end autonomy stacks - often lacking interpretable intermediate modules - has placed an increased burden on ensuring that the final output, i.e., the motion plan, is safe in order to validate the safety of the entire stack. This requires a safety monitor that is both complete (able to detect all unsafe plans) and sound (does not flag safe plans). In this work, we propose a principled safety monitor that leverages modern multi-modal trajectory predictors to approximate forward reachable sets (FRS) of surrounding agents. By formulating a convex program, we efficiently extract these data-driven FRSs directly from the predicted state distributions, conditioned on scene context such as lane topology and agent history. To ensure completeness, we leverage conformal prediction to calibrate the FRS and guarantee coverage of ground-truth trajectories with high probability. To preserve soundness in out-of-distribution (OOD) scenarios or under predictor failure, we introduce a Bayesian filter that dynamically adjusts the FRS conservativeness based on the predictor's observed performance. We then assess the safety of the ego vehicle's motion plan by checking for intersections with these calibrated FRSs, ensuring the plan remains collision-free under plausible future behaviors of others. Extensive experiments on the nuScenes dataset show our approach significantly improves soundness while maintaining completeness, offering a practical and reliable safety monitor for learned autonomy stacks.
>
---
#### [new 008] In-Situ Soil-Property Estimation and Bayesian Mapping with a Simulated Compact Track Loader
- **分类: cs.RO; J.2**

- **简介: 该论文属于机器人自主作业任务，旨在解决复杂地形中土壤属性未知和变化导致的自动化难题。论文提出了一种土壤属性在线估计与贝叶斯绘图系统，结合仿真数据训练物理信息神经网络，实现对土壤特性的实时预测与不确定性评估，支持更智能的自主地形作业。**

- **链接: [http://arxiv.org/pdf/2507.22356v1](http://arxiv.org/pdf/2507.22356v1)**

> **作者:** W. Jacob Wagner; Ahmet Soylemezoglu; Katherine Driggs-Campbell
>
> **备注:** 29 pages, 12 figures, 5 algorithms, ISTVS 2025
>
> **摘要:** Existing earthmoving autonomy is largely confined to highly controlled and well-characterized environments due to the complexity of vehicle-terrain interaction dynamics and the partial observability of the terrain resulting from unknown and spatially varying soil conditions. In this chapter, a a soil-property mapping system is proposed to extend the environmental state, in order to overcome these restrictions and facilitate development of more robust autonomous earthmoving. A GPU accelerated elevation mapping system is extended to incorporate a blind mapping component which traces the movement of the blade through the terrain to displace and erode intersected soil, enabling separately tracking undisturbed and disturbed soil. Each interaction is approximated as a flat blade moving through a locally homogeneous soil, enabling modeling of cutting forces using the fundamental equation of earthmoving (FEE). Building upon our prior work on in situ soil-property estimation, a method is devised to extract approximate geometric parameters of the model given the uneven terrain, and an improved physics infused neural network (PINN) model is developed to predict soil properties and uncertainties of these estimates. A simulation of a compact track loader (CTL) with a blade attachment is used to collect data to train the PINN model. Post-training, the model is leveraged online by the mapping system to track soil property estimates spatially as separate layers in the map, with updates being performed in a Bayesian manner. Initial experiments show that the system accurately highlights regions requiring higher relative interaction forces, indicating the promise of this approach in enabling soil-aware planning for autonomous terrain shaping.
>
---
#### [new 009] Comparing Normalizing Flows with Kernel Density Estimation in Estimating Risk of Automated Driving Systems
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于安全验证任务，旨在解决自动驾驶系统（ADS）风险评估中概率密度函数（PDF）估计不准确的问题。论文比较了归一化流（NF）与核密度估计（KDE）在高维场景参数建模中的效果，结果显示NF在处理高维数据和风险不确定性方面优于KDE，有助于提升ADS安全性评估的精确度。**

- **链接: [http://arxiv.org/pdf/2507.22429v1](http://arxiv.org/pdf/2507.22429v1)**

> **作者:** Erwin de Gelder; Maren Buermann; Olaf Op den Camp
>
> **备注:** Accepted for publication in proceedings of the 2025 IEEE International Automated Vehicle Validation Conference
>
> **摘要:** The development of safety validation methods is essential for the safe deployment and operation of Automated Driving Systems (ADSs). One of the goals of safety validation is to prospectively evaluate the risk of an ADS dealing with real-world traffic. Scenario-based assessment is a widely-used approach, where test cases are derived from real-world driving data. To allow for a quantitative analysis of the system performance, the exposure of the scenarios must be accurately estimated. The exposure of scenarios at parameter level is expressed using a Probability Density Function (PDF). However, assumptions about the PDF, such as parameter independence, can introduce errors, while avoiding assumptions often leads to oversimplified models with limited parameters to mitigate the curse of dimensionality. This paper considers the use of Normalizing Flows (NF) for estimating the PDF of the parameters. NF are a class of generative models that transform a simple base distribution into a complex one using a sequence of invertible and differentiable mappings, enabling flexible, high-dimensional density estimation without restrictive assumptions on the PDF's shape. We demonstrate the effectiveness of NF in quantifying risk and risk uncertainty of an ADS, comparing its performance with Kernel Density Estimation (KDE), a traditional method for non-parametric PDF estimation. While NF require more computational resources compared to KDE, NF is less sensitive to the curse of dimensionality. As a result, NF can improve risk uncertainty estimation, offering a more precise assessment of an ADS's safety. This work illustrates the potential of NF in scenario-based safety. Future work involves experimenting more with using NF for scenario generation and optimizing the NF architecture, transformation types, and training hyperparameters to further enhance their applicability.
>
---
#### [new 010] FLORES: A Reconfigured Wheel-Legged Robot for Enhanced Steering and Adaptability
- **分类: cs.RO**

- **简介: 该论文属于机器人设计任务，旨在解决轮腿机器人在不同地形中灵活性与效率不足的问题。论文提出了FLORES机器人新结构，通过改进前腿自由度提升转向与适应能力，并结合强化学习控制器实现多模式运动，优化复杂环境下的性能。**

- **链接: [http://arxiv.org/pdf/2507.22345v1](http://arxiv.org/pdf/2507.22345v1)**

> **作者:** Zhicheng Song; Jinglan Xu; Chunxin Zheng; Yulin Li; Zhihai Bi; Jun Ma
>
> **摘要:** Wheel-legged robots integrate the agility of legs for navigating rough terrains while harnessing the efficiency of wheels for smooth surfaces. However, most existing designs do not fully capitalize on the benefits of both legged and wheeled structures, which limits overall system flexibility and efficiency. We present FLORES (reconfigured wheel-legged robot for enhanced steering and adaptability), a novel wheel-legged robot design featuring a distinctive front-leg configuration that sets it beyond standard design approaches. Specifically, FLORES replaces the conventional hip-roll degree of freedom (DoF) of the front leg with hip-yaw DoFs, and this allows for efficient movement on flat surfaces while ensuring adaptability when navigating complex terrains. This innovative design facilitates seamless transitions between different locomotion modes (i.e., legged locomotion and wheeled locomotion) and optimizes the performance across varied environments. To fully exploit FLORES's mechanical capabilities, we develop a tailored reinforcement learning (RL) controller that adapts the Hybrid Internal Model (HIM) with a customized reward structure optimized for our unique mechanical configuration. This framework enables the generation of adaptive, multi-modal locomotion strategies that facilitate smooth transitions between wheeled and legged movements. Furthermore, our distinctive joint design enables the robot to exhibit novel and highly efficient locomotion gaits that capitalize on the synergistic advantages of both locomotion modes. Through comprehensive experiments, we demonstrate FLORES's enhanced steering capabilities, improved navigation efficiency, and versatile locomotion across various terrains. The open-source project can be found at https://github.com/ZhichengSong6/FLORES-A-Reconfigured-Wheel-Legged-Robot-for-Enhanced-Steering-and-Adaptability.git.
>
---
#### [new 011] Deployment of Objects with a Soft Everting Robot
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究软体翻转机器人内部运输和部署较大载荷的能力。任务是分析机器人如何携带不同形状、大小和重量的物体通过复杂地形。论文提出了预测载荷滑移的模型，实验验证了其运输1.5kg以内物体的能力，以及通过狭窄孔洞、转弯和跨越1.15米间隙的性能。**

- **链接: [http://arxiv.org/pdf/2507.22188v1](http://arxiv.org/pdf/2507.22188v1)**

> **作者:** Ethan DeVries; Jack Ferlazzo; Mustafa Ugur; Laura H. Blumenschein
>
> **备注:** 9 pages, 10 figures, This work has been submitted to the IEEE for possible publication
>
> **摘要:** Soft everting robots present significant advantages over traditional rigid robots, including enhanced dexterity, improved environmental interaction, and safe navigation in unpredictable environments. While soft everting robots have been widely demonstrated for exploration type tasks, their potential to move and deploy payloads in such tasks has been less investigated, with previous work focusing on sensors and tools for the robot. Leveraging the navigation capabilities, and deployed body, of the soft everting robot to deliver payloads in hazardous areas, e.g. carrying a water bottle to a person stuck under debris, would represent a significant capability in many applications. In this work, we present an analysis of how soft everting robots can be used to deploy larger, heavier payloads through the inside of the robot. We analyze both what objects can be deployed and what terrain features they can be carried through. Building on existing models, we present methods to quantify the effects of payloads on robot growth and self-support, and develop a model to predict payload slip. We then experimentally quantify payload transport using soft everting robot with a variety of payload shapes, sizes, and weights and though a series of tasks: steering, vertical transport, movement through holes, and movement across gaps. Overall, the results show that we can transport payloads in a variety of shapes and up to 1.5kg in weight and that we can move through circular apertures with as little as 0.01cm clearance around payloads, carry out discrete turns up to 135 degrees, and move across unsupported gaps of 1.15m in length.
>
---
#### [new 012] Modified Smith predictor for unstable linear systems
- **分类: math.OC; cs.RO; cs.SY; eess.SY; math.DS**

- **简介: 论文属于控制算法设计任务，旨在解决输入延迟的不稳定线性系统的控制问题。作者改进了Smith预测器，提出一种结构简单、无需复杂积分的控制方法，实现系统稳定化，确保状态变量有界且平衡点指数稳定。**

- **链接: [http://arxiv.org/pdf/2507.22243v1](http://arxiv.org/pdf/2507.22243v1)**

> **作者:** Anton Pyrkin; Konstantin Kalinin
>
> **备注:** in Russian language
>
> **摘要:** The paper presents a new control algorithm for unstable linear systems with input delay. In comparison with known analogues, the control law has been designed, which is a modification of the Smith predictor, and is the simplest one to implement without requiring complex integration methods. At the same time, the problem of stabilization of a closed system is effectively solved, ensuring the boundedness of all state variables and the exponential stability of the equilibrium point.
>
---
#### [new 013] Temporally Consistent Unsupervised Segmentation for Mobile Robot Perception
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于移动机器人感知任务，旨在解决无监督分割在非结构化环境中缺乏时间一致性的问题。作者提出了Frontier-Seg方法，利用DINOv2特征进行超像素聚类，并在视频帧间保持时间一致性，实现无需人工标注的地形分割。**

- **链接: [http://arxiv.org/pdf/2507.22194v1](http://arxiv.org/pdf/2507.22194v1)**

> **作者:** Christian Ellis; Maggie Wigness; Craig Lennon; Lance Fiondella
>
> **摘要:** Rapid progress in terrain-aware autonomous ground navigation has been driven by advances in supervised semantic segmentation. However, these methods rely on costly data collection and labor-intensive ground truth labeling to train deep models. Furthermore, autonomous systems are increasingly deployed in unrehearsed, unstructured environments where no labeled data exists and semantic categories may be ambiguous or domain-specific. Recent zero-shot approaches to unsupervised segmentation have shown promise in such settings but typically operate on individual frames, lacking temporal consistency-a critical property for robust perception in unstructured environments. To address this gap we introduce Frontier-Seg, a method for temporally consistent unsupervised segmentation of terrain from mobile robot video streams. Frontier-Seg clusters superpixel-level features extracted from foundation model backbones-specifically DINOv2-and enforces temporal consistency across frames to identify persistent terrain boundaries or frontiers without human supervision. We evaluate Frontier-Seg on a diverse set of benchmark datasets-including RUGD and RELLIS-3D-demonstrating its ability to perform unsupervised segmentation across unstructured off-road environments.
>
---
#### [new 014] Recognizing Actions from Robotic View for Natural Human-Robot Interaction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自然人机交互中的动作识别任务，旨在解决机器人在复杂场景中远距离识别人类动作的问题。论文构建了大规模数据集ACTIVE，并提出ACTIVE-PC方法，有效提升了移动机器人视角下远距离动作识别的准确性。**

- **链接: [http://arxiv.org/pdf/2507.22522v1](http://arxiv.org/pdf/2507.22522v1)**

> **作者:** Ziyi Wang; Peiming Li; Hong Liu; Zhichao Deng; Can Wang; Jun Liu; Junsong Yuan; Mengyuan Liu
>
> **备注:** 8 pages, 4 figures, Accepted to ICCV2025
>
> **摘要:** Natural Human-Robot Interaction (N-HRI) requires robots to recognize human actions at varying distances and states, regardless of whether the robot itself is in motion or stationary. This setup is more flexible and practical than conventional human action recognition tasks. However, existing benchmarks designed for traditional action recognition fail to address the unique complexities in N-HRI due to limited data, modalities, task categories, and diversity of subjects and environments. To address these challenges, we introduce ACTIVE (Action from Robotic View), a large-scale dataset tailored specifically for perception-centric robotic views prevalent in mobile service robots. ACTIVE comprises 30 composite action categories, 80 participants, and 46,868 annotated video instances, covering both RGB and point cloud modalities. Participants performed various human actions in diverse environments at distances ranging from 3m to 50m, while the camera platform was also mobile, simulating real-world scenarios of robot perception with varying camera heights due to uneven ground. This comprehensive and challenging benchmark aims to advance action and attribute recognition research in N-HRI. Furthermore, we propose ACTIVE-PC, a method that accurately perceives human actions at long distances using Multilevel Neighborhood Sampling, Layered Recognizers, Elastic Ellipse Query, and precise decoupling of kinematic interference from human actions. Experimental results demonstrate the effectiveness of ACTIVE-PC. Our code is available at: https://github.com/wangzy01/ACTIVE-Action-from-Robotic-View.
>
---
#### [new 015] Emergent interactions lead to collective frustration in robotic matter
- **分类: cond-mat.soft; cs.RO**

- **简介: 该论文研究了机器人物质中的集体行为，旨在探索多智能体系统是否能涌现出复杂现象。作者通过一个一维随机多粒子模型，发现系统能产生学习机制转变、粒子种类形成、挫败感等现象，并存在密度依赖的相变。工作结合了深度学习与主动物质理论，揭示了自组织和新兴相互作用如何驱动这些集体行为。**

- **链接: [http://arxiv.org/pdf/2507.22148v1](http://arxiv.org/pdf/2507.22148v1)**

> **作者:** Onurcan Bektas; Adolfo Alsina; Steffen Rulands
>
> **摘要:** Current artificial intelligence systems show near-human-level capabilities when deployed in isolation. Systems of a few collaborating intelligent agents are being engineered to perform tasks collectively. This raises the question of whether robotic matter, where many learning and intelligent agents interact, shows emergence of collective behaviour. And if so, which kind of phenomena would such systems exhibit? Here, we study a paradigmatic model for robotic matter: a stochastic many-particle system in which each particle is endowed with a deep neural network that predicts its transitions based on the particles' environments. For a one-dimensional model, we show that robotic matter exhibits complex emergent phenomena, including transitions between long-lived learning regimes, the emergence of particle species, and frustration. We also find a density-dependent phase transition with signatures of criticality. Using active matter theory, we show that this phase transition is a consequence of self-organisation mediated by emergent inter-particle interactions. Our simple model captures key features of more complex forms of robotic systems.
>
---
#### [new 016] Multi-Agent Path Finding Among Dynamic Uncontrollable Agents with Statistical Safety Guarantees
- **分类: cs.MA; cs.RO**

- **简介: 该论文属于多智能体路径规划任务，旨在解决动态环境中存在不可控智能体时的路径规划问题。通过改进冲突搜索算法，结合学习预测、预测误差量化与统计安全保证，实现安全、高效的路径规划，减少碰撞并保持系统吞吐量。**

- **链接: [http://arxiv.org/pdf/2507.22282v1](http://arxiv.org/pdf/2507.22282v1)**

> **作者:** Kegan J. Strawn; Thomy Phan; Eric Wang; Nora Ayanian; Sven Koenig; Lars Lindemann
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Existing multi-agent path finding (MAPF) solvers do not account for uncertain behavior of uncontrollable agents. We present a novel variant of Enhanced Conflict-Based Search (ECBS), for both one-shot and lifelong MAPF in dynamic environments with uncontrollable agents. Our method consists of (1) training a learned predictor for the movement of uncontrollable agents, (2) quantifying the prediction error using conformal prediction (CP), a tool for statistical uncertainty quantification, and (3) integrating these uncertainty intervals into our modified ECBS solver. Our method can account for uncertain agent behavior, comes with statistical guarantees on collision-free paths for one-shot missions, and scales to lifelong missions with a receding horizon sequence of one-shot instances. We run our algorithm, CP-Solver, across warehouse and game maps, with competitive throughput and reduced collisions.
>
---
#### [new 017] Viser: Imperative, Web-based 3D Visualization in Python
- **分类: cs.CV; cs.RO**

- **简介: 论文提出Viser，一个面向Python的3D可视化库，用于计算机视觉与机器人任务。要解决的问题是现有工具复杂、难扩展。Viser提供易用、可组合的3D与2D界面元素，支持快速搭建可视化应用。核心工作包括设计命令式API与基于网页的查看器，提升兼容性与交互体验。**

- **链接: [http://arxiv.org/pdf/2507.22885v1](http://arxiv.org/pdf/2507.22885v1)**

> **作者:** Brent Yi; Chung Min Kim; Justin Kerr; Gina Wu; Rebecca Feng; Anthony Zhang; Jonas Kulhanek; Hongsuk Choi; Yi Ma; Matthew Tancik; Angjoo Kanazawa
>
> **备注:** Code and docs: https://viser.studio
>
> **摘要:** We present Viser, a 3D visualization library for computer vision and robotics. Viser aims to bring easy and extensible 3D visualization to Python: we provide a comprehensive set of 3D scene and 2D GUI primitives, which can be used independently with minimal setup or composed to build specialized interfaces. This technical report describes Viser's features, interface, and implementation. Key design choices include an imperative-style API and a web-based viewer, which improve compatibility with modern programming patterns and workflows.
>
---
#### [new 018] Toward Trusted Onboard AI: Advancing Small Satellite Operations using Reinforcement Learning
- **分类: eess.SY; cs.RO; cs.SY; 68T05; I.2.9**

- **简介: 论文研究在小型卫星上部署强化学习（RL）算法，实现自主指令控制。旨在解决卫星操作中依赖地面控制、响应速度慢的问题。通过构建数字孪生训练RL代理执行宏观控制动作，并在轨验证其决策能力，推动可信AI在航天领域的应用。**

- **链接: [http://arxiv.org/pdf/2507.22198v1](http://arxiv.org/pdf/2507.22198v1)**

> **作者:** Cannon Whitney; Joseph Melville
>
> **备注:** 11 pages, 2 figures, 2 tables, accepted to the 39th Small Satellite Conference
>
> **摘要:** A RL (Reinforcement Learning) algorithm was developed for command automation onboard a 3U CubeSat. This effort focused on the implementation of macro control action RL, a technique in which an onboard agent is provided with compiled information based on live telemetry as its observation. The agent uses this information to produce high-level actions, such as adjusting attitude to solar pointing, which are then translated into control algorithms and executed through lower-level instructions. Once trust in the onboard agent is established, real-time environmental information can be leveraged for faster response times and reduced reliance on ground control. The approach not only focuses on developing an RL algorithm for a specific satellite but also sets a precedent for integrating trusted AI into onboard systems. This research builds on previous work in three areas: (1) RL algorithms for issuing high-level commands that are translated into low-level executable instructions; (2) the deployment of AI inference models interfaced with live operational systems, particularly onboard spacecraft; and (3) strategies for building trust in AI systems, especially for remote and autonomous applications. Existing RL research for satellite control is largely limited to simulation-based experiments; in this work, these techniques are tailored by constructing a digital twin of a specific spacecraft and training the RL agent to issue macro actions in this simulated environment. The policy of the trained agent is copied to an isolated environment, where it is fed compiled information about the satellite to make inference predictions, thereby demonstrating the RL algorithm's validity on orbit without granting it command authority. This process enables safe comparison of the algorithm's predictions against actual satellite behavior and ensures operation within expected parameters.
>
---
## 更新

#### [replaced 001] FloPE: Flower Pose Estimation for Precision Pollination
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11692v2](http://arxiv.org/pdf/2503.11692v2)**

> **作者:** Rashik Shrestha; Madhav Rijal; Trevor Smith; Yu Gu
>
> **备注:** Accepted to IROS 2025. Project page: https://wvu-irl.github.io/flope-irl/
>
> **摘要:** This study presents Flower Pose Estimation (FloPE), a real-time flower pose estimation framework for computationally constrained robotic pollination systems. Robotic pollination has been proposed to supplement natural pollination to ensure global food security due to the decreased population of natural pollinators. However, flower pose estimation for pollination is challenging due to natural variability, flower clusters, and high accuracy demands due to the flowers' fragility when pollinating. This method leverages 3D Gaussian Splatting to generate photorealistic synthetic datasets with precise pose annotations, enabling effective knowledge distillation from a high-capacity teacher model to a lightweight student model for efficient inference. The approach was evaluated on both single and multi-arm robotic platforms, achieving a mean pose estimation error of 0.6 cm and 19.14 degrees within a low computational cost. Our experiments validate the effectiveness of FloPE, achieving up to 78.75% pollination success rate and outperforming prior robotic pollination techniques.
>
---
#### [replaced 002] Aerial Grasping via Maximizing Delta-Arm Workspace Utilization
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.15539v2](http://arxiv.org/pdf/2506.15539v2)**

> **作者:** Haoran Chen; Weiliang Deng; Biyu Ye; Yifan Xiong; Zongliang Pan; Ximin Lyu
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** The workspace limits the operational capabilities and range of motion for the systems with robotic arms. Maximizing workspace utilization has the potential to provide more optimal solutions for aerial manipulation tasks, increasing the system's flexibility and operational efficiency. In this paper, we introduce a novel planning framework for aerial grasping that maximizes workspace utilization. We formulate an optimization problem to optimize the aerial manipulator's trajectory, incorporating task constraints to achieve efficient manipulation. To address the challenge of incorporating the delta arm's non-convex workspace into optimization constraints, we leverage a Multilayer Perceptron (MLP) to map position points to feasibility probabilities.Furthermore, we employ Reversible Residual Networks (RevNet) to approximate the complex forward kinematics of the delta arm, utilizing efficient model gradients to eliminate workspace constraints. We validate our methods in simulations and real-world experiments to demonstrate their effectiveness.
>
---
#### [replaced 003] FOCI: Trajectory Optimization on Gaussian Splats
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.08510v2](http://arxiv.org/pdf/2505.08510v2)**

> **作者:** Mario Gomez Andreu; Maximum Wilder-Smith; Victor Klemm; Vaishakh Patil; Jesus Tordesillas; Marco Hutter
>
> **备注:** 8 pages, 8 figures, Mario Gomez Andreu and Maximum Wilder-Smith contributed equally
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently gained popularity as a faster alternative to Neural Radiance Fields (NeRFs) in 3D reconstruction and view synthesis methods. Leveraging the spatial information encoded in 3DGS, this work proposes FOCI (Field Overlap Collision Integral), an algorithm that is able to optimize trajectories directly on the Gaussians themselves. FOCI leverages a novel and interpretable collision formulation for 3DGS using the notion of the overlap integral between Gaussians. Contrary to other approaches, which represent the robot with conservative bounding boxes that underestimate the traversability of the environment, we propose to represent the environment and the robot as Gaussian Splats. This not only has desirable computational properties, but also allows for orientation-aware planning, allowing the robot to pass through very tight and narrow spaces. We extensively test our algorithm in both synthetic and real Gaussian Splats, showcasing that collision-free trajectories for the ANYmal legged robot that can be computed in a few seconds, even with hundreds of thousands of Gaussians making up the environment. The project page and code are available at https://rffr.leggedrobotics.com/works/foci/
>
---
#### [replaced 004] TartanGround: A Large-Scale Dataset for Ground Robot Perception and Navigation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10696v2](http://arxiv.org/pdf/2505.10696v2)**

> **作者:** Manthan Patel; Fan Yang; Yuheng Qiu; Cesar Cadena; Sebastian Scherer; Marco Hutter; Wenshan Wang
>
> **备注:** Accepted for publication to IEEE/RSJ IROS 2025
>
> **摘要:** We present TartanGround, a large-scale, multi-modal dataset to advance the perception and autonomy of ground robots operating in diverse environments. This dataset, collected in various photorealistic simulation environments includes multiple RGB stereo cameras for 360-degree coverage, along with depth, optical flow, stereo disparity, LiDAR point clouds, ground truth poses, semantic segmented images, and occupancy maps with semantic labels. Data is collected using an integrated automatic pipeline, which generates trajectories mimicking the motion patterns of various ground robot platforms, including wheeled and legged robots. We collect 910 trajectories across 70 environments, resulting in 1.5 million samples. Evaluations on occupancy prediction and SLAM tasks reveal that state-of-the-art methods trained on existing datasets struggle to generalize across diverse scenes. TartanGround can serve as a testbed for training and evaluation of a broad range of learning-based tasks, including occupancy prediction, SLAM, neural scene representation, perception-based navigation, and more, enabling advancements in robotic perception and autonomy towards achieving robust models generalizable to more diverse scenarios. The dataset and codebase are available on the webpage: https://tartanair.org/tartanground
>
---
#### [replaced 005] An Actionable Hierarchical Scene Representation Enhancing Autonomous Inspection Missions in Unknown Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.19582v3](http://arxiv.org/pdf/2412.19582v3)**

> **作者:** Vignesh Kottayam Viswanathan; Mario Alberto Valdes Saucedo; Sumeet Gajanan Satpute; Christoforos Kanellakis; George Nikolakopoulos
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** In this article, we present the Layered Semantic Graphs (LSG), a novel actionable hierarchical scene graph, fully integrated with a multi-modal mission planner, the FLIE: A First-Look based Inspection and Exploration planner. The novelty of this work stems from aiming to address the task of maintaining an intuitive and multi-resolution scene representation, while simultaneously offering a tractable foundation for planning and scene understanding during an ongoing inspection mission of apriori unknown targets-of-interest in an unknown environment. The proposed LSG scheme is composed of locally nested hierarchical graphs, at multiple layers of abstraction, with the abstract concepts grounded on the functionality of the integrated FLIE planner. Furthermore, LSG encapsulates real-time semantic segmentation models that offer extraction and localization of desired semantic elements within the hierarchical representation. This extends the capability of the inspection planner, which can then leverage LSG to make an informed decision to inspect a particular semantic of interest. We also emphasize the hierarchical and semantic path-planning capabilities of LSG, which could extend inspection missions by improving situational awareness for human operators in an unknown environment. The validity of the proposed scheme is proven through extensive evaluations of the proposed architecture in simulations, as well as experimental field deployments on a Boston Dynamics Spot quadruped robot in urban outdoor environment settings.
>
---
#### [replaced 006] Trajectory First: A Curriculum for Discovering Diverse Policies
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.01568v2](http://arxiv.org/pdf/2506.01568v2)**

> **作者:** Cornelius V. Braun; Sayantan Auddy; Marc Toussaint
>
> **备注:** Accepted into the Inductive Biases in Reinforcement Learning Workshop at RLC 2025
>
> **摘要:** Being able to solve a task in diverse ways makes agents more robust to task variations and less prone to local optima. In this context, constrained diversity optimization has emerged as a powerful reinforcement learning (RL) framework to train a diverse set of agents in parallel. However, existing constrained-diversity RL methods often under-explore in complex tasks such as robotic manipulation, leading to a lack in policy diversity. To improve diversity optimization in RL, we therefore propose a curriculum that first explores at the trajectory level before learning step-based policies. In our empirical evaluation, we provide novel insights into the shortcoming of skill-based diversity optimization, and demonstrate empirically that our curriculum improves the diversity of the learned skills.
>
---
#### [replaced 007] SPADE: Towards Scalable Path Planning Architecture on Actionable Multi-Domain 3D Scene Graphs
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.19098v2](http://arxiv.org/pdf/2505.19098v2)**

> **作者:** Vignesh Kottayam Viswanathan; Akash Patel; Mario Alberto Valdes Saucedo; Sumeet Satpute; Christoforos Kanellakis; George Nikolakopoulos
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** In this work, we introduce SPADE, a path planning framework designed for autonomous navigation in dynamic environments using 3D scene graphs. SPADE combines hierarchical path planning with local geometric awareness to enable collision-free movement in dynamic scenes. The framework bifurcates the planning problem into two: (a) solving the sparse abstract global layer plan and (b) iterative path refinement across denser lower local layers in step with local geometric scene navigation. To ensure efficient extraction of a feasible route in a dense multi-task domain scene graphs, the framework enforces informed sampling of traversable edges prior to path-planning. This removes extraneous information not relevant to path-planning and reduces the overall planning complexity over a graph. Existing approaches address the problem of path planning over scene graphs by decoupling hierarchical and geometric path evaluation processes. Specifically, this results in an inefficient replanning over the entire scene graph when encountering path obstructions blocking the original route. In contrast, SPADE prioritizes local layer planning coupled with local geometric scene navigation, enabling navigation through dynamic scenes while maintaining efficiency in computing a traversable route. We validate SPADE through extensive simulation experiments and real-world deployment on a quadrupedal robot, demonstrating its efficacy in handling complex and dynamic scenarios.
>
---
#### [replaced 008] Spatial Language Likelihood Grounding Network for Bayesian Fusion of Human-Robot Observations
- **分类: cs.RO; cs.CL; cs.IT; cs.LG; cs.SY; eess.SY; math.IT**

- **链接: [http://arxiv.org/pdf/2507.19947v2](http://arxiv.org/pdf/2507.19947v2)**

> **作者:** Supawich Sitdhipol; Waritwong Sukprasongdee; Ekapol Chuangsuwanich; Rina Tse
>
> **备注:** Accepted to the 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC); Supplementary video: https://cu-asl.github.io/fp-lgn/
>
> **摘要:** Fusing information from human observations can help robots overcome sensing limitations in collaborative tasks. However, an uncertainty-aware fusion framework requires a grounded likelihood representing the uncertainty of human inputs. This paper presents a Feature Pyramid Likelihood Grounding Network (FP-LGN) that grounds spatial language by learning relevant map image features and their relationships with spatial relation semantics. The model is trained as a probability estimator to capture aleatoric uncertainty in human language using three-stage curriculum learning. Results showed that FP-LGN matched expert-designed rules in mean Negative Log-Likelihood (NLL) and demonstrated greater robustness with lower standard deviation. Collaborative sensing results demonstrated that the grounded likelihood successfully enabled uncertainty-aware fusion of heterogeneous human language observations and robot sensor measurements, achieving significant improvements in human-robot collaborative task performance.
>
---
#### [replaced 009] Multi-robot LiDAR SLAM: a practical case study in underground tunnel environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.21553v2](http://arxiv.org/pdf/2507.21553v2)**

> **作者:** Federica Di Lauro; Domenico G. Sorrenti; Miguel Angel Sotelo
>
> **备注:** 14 pages, 14 figures
>
> **摘要:** Multi-robot SLAM aims at localizing and building a map with multiple robots, interacting with each other. In the work described in this article, we analyze the pipeline of a decentralized LiDAR SLAM system to study the current limitations of the state of the art, and we discover a significant source of failures, i.e., that the loop detection is the source of too many false positives. We therefore develop and propose a new heuristic to overcome these limitations. The environment taken as reference in this work is the highly challenging case of underground tunnels. We also highlight potential new research areas still under-explored.
>
---
#### [replaced 010] Design, Dynamic Modeling and Control of a 2-DOF Robotic Wrist Actuated by Twisted and Coiled Actuators
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.05508v2](http://arxiv.org/pdf/2503.05508v2)**

> **作者:** Yunsong Zhang; Xinyu Zhou; Feitian Zhang
>
> **摘要:** Artificial muscle-driven modular soft robots exhibit significant potential for executing complex tasks. However, their broader applicability remains constrained by the lack of dynamic model-based control strategies tailored for multi-degree-of-freedom (DOF) configurations. This paper presents a novel design of a 2-DOF robotic wrist, envisioned as a fundamental building block for such advanced robotic systems. The wrist module is actuated by twisted and coiled actuators (TCAs) and utilizes a compact 3RRRR parallel mechanism to achieve a lightweight structure with enhanced motion capability. A comprehensive Lagrangian dynamic model is developed to capture the module's complex nonlinear behavior. Leveraging this model, a nonlinear model predictive controller (NMPC) is designed to ensure accurate trajectory tracking. A physical prototype of the robotic wrist is fabricated, and extensive experiments are performed to validate its motion performance and the fidelity of the proposed dynamic model. Subsequently, comparative evaluations between the NMPC and a conventional PID controller are conducted under various operating conditions. Experimental results demonstrate the effectiveness and robustness of the dynamic model-based control approach in managing the motion of TCA-driven robotic wrists. Finally, to illustrate its practical utility and integrability, the wrist module is incorporated into a multi-segment soft robotic arm, where it successfully executes a trajectory tracking task.
>
---
#### [replaced 011] UAV See, UGV Do: Aerial Imagery and Virtual Teach Enabling Zero-Shot Ground Vehicle Repeat
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.16912v2](http://arxiv.org/pdf/2505.16912v2)**

> **作者:** Desiree Fisker; Alexander Krawciw; Sven Lilge; Melissa Greeff; Timothy D. Barfoot
>
> **备注:** 8 pages, 8 figures, accepted to IROS 2025
>
> **摘要:** This paper presents Virtual Teach and Repeat (VirT&R): an extension of the Teach and Repeat (T&R) framework that enables GPS-denied, zero-shot autonomous ground vehicle navigation in untraversed environments. VirT&R leverages aerial imagery captured for a target environment to train a Neural Radiance Field (NeRF) model so that dense point clouds and photo-textured meshes can be extracted. The NeRF mesh is used to create a high-fidelity simulation of the environment for piloting an unmanned ground vehicle (UGV) to virtually define a desired path. The mission can then be executed in the actual target environment by using NeRF-generated point cloud submaps associated along the path and an existing LiDAR Teach and Repeat (LT&R) framework. We benchmark the repeatability of VirT&R on over 12 km of autonomous driving data using physical markings that allow a sim-to-real lateral path-tracking error to be obtained and compared with LT&R. VirT&R achieved measured root mean squared errors (RMSE) of 19.5 cm and 18.4 cm in two different environments, which are slightly less than one tire width (24 cm) on the robot used for testing, and respective maximum errors were 39.4 cm and 47.6 cm. This was done using only the NeRF-derived teach map, demonstrating that VirT&R has similar closed-loop path-tracking performance to LT&R but does not require a human to manually teach the path to the UGV in the actual environment.
>
---
#### [replaced 012] Distance and Collision Probability Estimation from Gaussian Surface Models
- **分类: cs.RO; cs.CG; cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2402.00186v3](http://arxiv.org/pdf/2402.00186v3)**

> **作者:** Kshitij Goel; Wennie Tabib
>
> **备注:** Accepted at IROS 2025
>
> **摘要:** This paper describes continuous-space methodologies to estimate the collision probability, Euclidean distance and gradient between an ellipsoidal robot model and an environment surface modeled as a set of Gaussian distributions. Continuous-space collision probability estimation is critical for uncertainty-aware motion planning. Most collision detection and avoidance approaches assume the robot is modeled as a sphere, but ellipsoidal representations provide tighter approximations and enable navigation in cluttered and narrow spaces. State-of-the-art methods derive the Euclidean distance and gradient by processing raw point clouds, which is computationally expensive for large workspaces. Recent advances in Gaussian surface modeling (e.g. mixture models, splatting) enable compressed and high-fidelity surface representations. Few methods exist to estimate continuous-space occupancy from such models. They require Gaussians to model free space and are unable to estimate the collision probability, Euclidean distance and gradient for an ellipsoidal robot. The proposed methods bridge this gap by extending prior work in ellipsoid-to-ellipsoid Euclidean distance and collision probability estimation to Gaussian surface models. A geometric blending approach is also proposed to improve collision probability estimation. The approaches are evaluated with numerical 2D and 3D experiments using real-world point cloud data. Methods for efficient calculation of these quantities are demonstrated to execute within a few microseconds per ellipsoid pair using a single-thread on low-power CPUs of modern embedded computers
>
---
#### [replaced 013] SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.20104v5](http://arxiv.org/pdf/2412.20104v5)**

> **作者:** Wenkun He; Yun Liu; Ruitao Liu; Li Yi
>
> **备注:** 27 pages, 10 figures, 20 tables. Accepted by ICCV 2025
>
> **摘要:** Synthesizing realistic human-object interaction motions is a critical problem in VR/AR and human animation. Unlike the commonly studied scenarios involving a single human or hand interacting with one object, we address a more generic multi-body setting with arbitrary numbers of humans, hands, and objects. This complexity introduces significant challenges in synchronizing motions due to the high correlations and mutual influences among bodies. To address these challenges, we introduce SyncDiff, a novel method for multi-body interaction synthesis using a synchronized motion diffusion strategy. SyncDiff employs a single diffusion model to capture the joint distribution of multi-body motions. To enhance motion fidelity, we propose a frequency-domain motion decomposition scheme. Additionally, we introduce a new set of alignment scores to emphasize the synchronization of different body motions. SyncDiff jointly optimizes both data sample likelihood and alignment likelihood through an explicit synchronization strategy. Extensive experiments across four datasets with various multi-body configurations demonstrate the superiority of SyncDiff over existing state-of-the-art motion synthesis methods.
>
---
#### [replaced 014] GRaD-Nav: Efficiently Learning Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.03984v3](http://arxiv.org/pdf/2503.03984v3)**

> **作者:** Qianzhong Chen; Jiankai Sun; Naixiang Gao; JunEn Low; Timothy Chen; Mac Schwager
>
> **摘要:** Autonomous visual navigation is an essential element in robot autonomy. Reinforcement learning (RL) offers a promising policy training paradigm. However existing RL methods suffer from high sample complexity, poor sim-to-real transfer, and limited runtime adaptability to navigation scenarios not seen during training. These problems are particularly challenging for drones, with complex nonlinear and unstable dynamics, and strong dynamic coupling between control and perception. In this paper, we propose a novel framework that integrates 3D Gaussian Splatting (3DGS) with differentiable deep reinforcement learning (DDRL) to train vision-based drone navigation policies. By leveraging high-fidelity 3D scene representations and differentiable simulation, our method improves sample efficiency and sim-to-real transfer. Additionally, we incorporate a Context-aided Estimator Network (CENet) to adapt to environmental variations at runtime. Moreover, by curriculum training in a mixture of different surrounding environments, we achieve in-task generalization, the ability to solve new instances of a task not seen during training. Drone hardware experiments demonstrate our method's high training efficiency compared to state-of-the-art RL methods, zero shot sim-to-real transfer for real robot deployment without fine tuning, and ability to adapt to new instances within the same task class (e.g. to fly through a gate at different locations with different distractors in the environment). Our simulator and training framework are open-sourced at: https://github.com/Qianzhong-Chen/grad_nav.
>
---
#### [replaced 015] NeurIT: Pushing the Limit of Neural Inertial Tracking for Indoor Robotic IoT
- **分类: cs.RO; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2404.08939v2](http://arxiv.org/pdf/2404.08939v2)**

> **作者:** Xinzhe Zheng; Sijie Ji; Yipeng Pan; Kaiwen Zhang; Chenshu Wu
>
> **摘要:** Inertial tracking is vital for robotic IoT and has gained popularity thanks to the ubiquity of low-cost inertial measurement units and deep learning-powered tracking algorithms. Existing works, however, have not fully utilized IMU measurements, particularly magnetometers, nor have they maximized the potential of deep learning to achieve the desired accuracy. To address these limitations, we introduce NeurIT, which elevates tracking accuracy to a new level. NeurIT employs a Time-Frequency Block-recurrent Transformer (TF-BRT) at its core, combining both RNN and Transformer to learn representative features in both time and frequency domains. To fully utilize IMU information, we strategically employ body-frame differentiation of magnetometers, considerably reducing the tracking error. We implement NeurIT on a customized robotic platform and conduct evaluation in various indoor environments. Experimental results demonstrate that NeurIT achieves a mere 1-meter tracking error over a 300-meter distance. Notably, it significantly outperforms state-of-the-art baselines by 48.21% on unseen data. Moreover, NeurIT demonstrates robustness in large urban complexes and performs comparably to the visual-inertial approach (Tango Phone) in vision-favored conditions while surpassing it in feature-sparse settings. We believe NeurIT takes an important step forward toward practical neural inertial tracking for ubiquitous and scalable tracking of robotic things. NeurIT is open-sourced here: https://github.com/aiot-lab/NeurIT.
>
---
#### [replaced 016] KIX: A Knowledge and Interaction-Centric Metacognitive Framework for Task Generalization
- **分类: cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2402.05346v3](http://arxiv.org/pdf/2402.05346v3)**

> **作者:** Arun Kumar; Paul Schrater
>
> **摘要:** People aptly exhibit general intelligence behaviors through flexible problem-solving and the ability to adapt to novel situations by reusing and applying high-level knowledge acquired over time. In contrast, artificial agents tend to be specialists, lacking such generalist behaviors. To bridge this gap, artificial agents will require understanding and exploiting critical structured knowledge representations. We introduce a metacognitive reasoning framework, Knowledge-Interaction-eXecution (KIX), and argue that interactions with objects, by leveraging a type space, facilitate the learning of transferable interaction concepts and promote generalization. This framework offers a principled approach for integrating knowledge into reinforcement learning and holds promise as an enabler for generalist behaviors in artificial intelligence, robotics, and autonomous systems.
>
---
#### [replaced 017] Swing Leg Motion Strategy for Heavy-load Legged Robot Based on Force Sensing
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2309.01112v2](http://arxiv.org/pdf/2309.01112v2)**

> **作者:** Ze Fu; Yinghui Li; Weizhong Guo
>
> **备注:** The manuscript is withdrawn due to ongoing major revisions and improvements to the methodology and experimental validation
>
> **摘要:** The heavy-load legged robot has strong load carrying capacity and can adapt to various unstructured terrains. But the large weight results in higher requirements for motion stability and environmental perception ability. In order to utilize force sensing information to improve its motion performance, in this paper, we propose a finite state machine model for the swing leg in the static gait by imitating the movement of the elephant. Based on the presence or absence of additional terrain information, different trajectory planning strategies are provided for the swing leg to enhance the success rate of stepping and save energy. The experimental results on a novel quadruped robot show that our method has strong robustness and can enable heavy-load legged robots to pass through various complex terrains autonomously and smoothly.
>
---
#### [replaced 018] $S^2M^2$: Scalable Stereo Matching Model for Reliable Depth Estimation
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.13229v3](http://arxiv.org/pdf/2507.13229v3)**

> **作者:** Junhong Min; Youngpil Jeon; Jimin Kim; Minyong Choi
>
> **备注:** 8 pages, 5 figures, ICCV accepted paper
>
> **摘要:** The pursuit of a generalizable stereo matching model, capable of performing well across varying resolutions and disparity ranges without dataset-specific fine-tuning, has revealed a fundamental trade-off. Iterative local search methods achieve high scores on constrained benchmarks, but their core mechanism inherently limits the global consistency required for true generalization. However, global matching architectures, while theoretically more robust, have historically been rendered infeasible by prohibitive computational and memory costs. We resolve this dilemma with $S^2M^2$: a global matching architecture that achieves state-of-the-art accuracy and high efficiency without relying on cost volume filtering or deep refinement stacks. Our design integrates a multi-resolution transformer for robust long-range correspondence, trained with a novel loss function that concentrates probability on feasible matches. This approach enables a more robust joint estimation of disparity, occlusion, and confidence. $S^2M^2$ establishes a new state of the art on Middlebury v3 and ETH3D benchmarks, significantly outperforming prior methods in most metrics while reconstructing high-quality details with competitive efficiency.
>
---
#### [replaced 019] Decision Transformer-Based Drone Trajectory Planning with Dynamic Safety-Efficiency Trade-Offs
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.21506v2](http://arxiv.org/pdf/2507.21506v2)**

> **作者:** Chang-Hun Ji; SiWoon Song; Youn-Hee Han; SungTae Moon
>
> **备注:** Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025. Copyright 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses
>
> **摘要:** A drone trajectory planner should be able to dynamically adjust the safety-efficiency trade-off according to varying mission requirements in unknown environments. Although traditional polynomial-based planners offer computational efficiency and smooth trajectory generation, they require expert knowledge to tune multiple parameters to adjust this trade-off. Moreover, even with careful tuning, the resulting adjustment may fail to achieve the desired trade-off. Similarly, although reinforcement learning-based planners are adaptable in unknown environments, they do not explicitly address the safety-efficiency trade-off. To overcome this limitation, we introduce a Decision Transformer-based trajectory planner that leverages a single parameter, Return-to-Go (RTG), as a \emph{temperature parameter} to dynamically adjust the safety-efficiency trade-off. In our framework, since RTG intuitively measures the safety and efficiency of a trajectory, RTG tuning does not require expert knowledge. We validate our approach using Gazebo simulations in both structured grid and unstructured random environments. The experimental results demonstrate that our planner can dynamically adjust the safety-efficiency trade-off by simply tuning the RTG parameter. Furthermore, our planner outperforms existing baseline methods across various RTG settings, generating safer trajectories when tuned for safety and more efficient trajectories when tuned for efficiency. Real-world experiments further confirm the reliability and practicality of our proposed planner.
>
---
#### [replaced 020] I Know You're Listening: Adaptive Voice for HRI
- **分类: cs.RO; cs.HC; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.15107v2](http://arxiv.org/pdf/2506.15107v2)**

> **作者:** Paige Tuttösí
>
> **备注:** PhD Thesis Simon Fraser University https://summit.sfu.ca/item/39353 Read the Room: IROS 2023, Mmm whatcha say?: INTERSPEECH 2024, Emojivoice: RO-MAN 2025, You sound a little tense: SSW 2025. Thesis presentation here: https://www.youtube.com/watch?v=9BcEwqYOMYI
>
> **摘要:** While the use of social robots for language teaching has been explored, there remains limited work on a task-specific synthesized voices for language teaching robots. Given that language is a verbal task, this gap may have severe consequences for the effectiveness of robots for language teaching tasks. We address this lack of L2 teaching robot voices through three contributions: 1. We address the need for a lightweight and expressive robot voice. Using a fine-tuned version of Matcha-TTS, we use emoji prompting to create an expressive voice that shows a range of expressivity over time. The voice can run in real time with limited compute resources. Through case studies, we found this voice more expressive, socially appropriate, and suitable for long periods of expressive speech, such as storytelling. 2. We explore how to adapt a robot's voice to physical and social ambient environments to deploy our voices in various locations. We found that increasing pitch and pitch rate in noisy and high-energy environments makes the robot's voice appear more appropriate and makes it seem more aware of its current environment. 3. We create an English TTS system with improved clarity for L2 listeners using known linguistic properties of vowels that are difficult for these listeners. We used a data-driven, perception-based approach to understand how L2 speakers use duration cues to interpret challenging words with minimal tense (long) and lax (short) vowels in English. We found that the duration of vowels strongly influences the perception for L2 listeners and created an "L2 clarity mode" for Matcha-TTS that applies a lengthening to tense vowels while leaving lax vowels unchanged. Our clarity mode was found to be more respectful, intelligible, and encouraging than base Matcha-TTS while reducing transcription errors in these challenging tense/lax minimal pairs.
>
---
#### [replaced 021] AffordDexGrasp: Open-set Language-guided Dexterous Grasp with Generalizable-Instructive Affordance
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.07360v2](http://arxiv.org/pdf/2503.07360v2)**

> **作者:** Yi-Lin Wei; Mu Lin; Yuhao Lin; Jian-Jian Jiang; Xiao-Ming Wu; Ling-An Zeng; Wei-Shi Zheng
>
> **备注:** Accepted by ICCV 2025.Project page: https://isee-laboratory.github.io/AffordDexGrasp/
>
> **摘要:** Language-guided robot dexterous generation enables robots to grasp and manipulate objects based on human commands. However, previous data-driven methods are hard to understand intention and execute grasping with unseen categories in the open set. In this work, we explore a new task, Open-set Language-guided Dexterous Grasp, and find that the main challenge is the huge gap between high-level human language semantics and low-level robot actions. To solve this problem, we propose an Affordance Dexterous Grasp (AffordDexGrasp) framework, with the insight of bridging the gap with a new generalizable-instructive affordance representation. This affordance can generalize to unseen categories by leveraging the object's local structure and category-agnostic semantic attributes, thereby effectively guiding dexterous grasp generation. Built upon the affordance, our framework introduces Affordance Flow Matching (AFM) for affordance generation with language as input, and Grasp Flow Matching (GFM) for generating dexterous grasp with affordance as input. To evaluate our framework, we build an open-set table-top language-guided dexterous grasp dataset. Extensive experiments in the simulation and real worlds show that our framework surpasses all previous methods in open-set generalization.
>
---
#### [replaced 022] Application of Vision-Language Model to Pedestrians Behavior and Scene Understanding in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.06680v2](http://arxiv.org/pdf/2501.06680v2)**

> **作者:** Haoxiang Gao; Li Zhang; Yu Zhao; Zhou Yang; Jinghan Cao
>
> **摘要:** Vision-language models (VLMs) have become a promising approach to enhancing perception and decision-making in autonomous driving. The gap remains in applying VLMs to understand complex scenarios interacting with pedestrians and efficient vehicle deployment. In this paper, we propose a knowledge distillation method that transfers knowledge from large-scale vision-language foundation models to efficient vision networks, and we apply it to pedestrian behavior prediction and scene understanding tasks, achieving promising results in generating more diverse and comprehensive semantic attributes. We also utilize multiple pre-trained models and ensemble techniques to boost the model's performance. We further examined the effectiveness of the model after knowledge distillation; the results show significant metric improvements in open-vocabulary perception and trajectory prediction tasks, which can potentially enhance the end-to-end performance of autonomous driving.
>
---
#### [replaced 023] Resilient Multi-Robot Target Tracking with Sensing and Communication Danger Zones
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.11230v2](http://arxiv.org/pdf/2409.11230v2)**

> **作者:** Peihan Li; Yuwei Wu; Jiazhen Liu; Gaurav S. Sukhatme; Vijay Kumar; Lifeng Zhou
>
> **摘要:** Multi-robot collaboration for target tracking in adversarial environments poses significant challenges, including system failures, dynamic priority shifts, and other unpredictable factors. These challenges become even more pronounced when the environment is unknown. In this paper, we propose a resilient coordination framework for multi-robot, multi-target tracking in environments with unknown sensing and communication danger zones. We consider scenarios where failures caused by these danger zones are probabilistic and temporary, allowing robots to escape from danger zones to minimize the risk of future failures. We formulate this problem as a nonlinear optimization with soft chance constraints, enabling real-time adjustments to robot behaviors based on varying types of dangers and failures. This approach dynamically balances target tracking performance and resilience, adapting to evolving sensing and communication conditions in real-time. To validate the effectiveness of the proposed method, we assess its performance across various tracking scenarios, benchmark it against methods without resilient adaptation and collaboration, and conduct several real-world experiments.
>
---
#### [replaced 024] Free-Gate: Planning, Control And Policy Composition via Free Energy Gating
- **分类: math.OC; cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2412.06636v3](http://arxiv.org/pdf/2412.06636v3)**

> **作者:** Francesca Rossi; Émiland Garrabé; Giovanni Russo
>
> **备注:** 15 pages, 2 figures
>
> **摘要:** We consider the problem of optimally composing a set of primitives to tackle planning and control tasks. To address this problem, we introduce a free energy computational model for planning and control via policy composition: Free-Gate. Within Free-Gate, control primitives are combined via a gating mechanism that minimizes variational free energy. This composition problem is formulated as a finite-horizon optimal control problem, which we prove remains convex even when the cost is not convex in states/actions and the environment is nonlinear, stochastic and non-stationary. We develop an algorithm that computes the optimal primitives composition and demonstrate its effectiveness via in-silico and hardware experiments on an application involving robot navigation in an environment with obstacles. The experiments highlight that Free-Gate enables the robot to navigate to the destination despite only having available simple motor primitives that, individually, could not fulfill the task.
>
---
#### [replaced 025] SKiD-SLAM: Robust, Lightweight, and Distributed Multi-Robot LiDAR SLAM in Resource-Constrained Field Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.08230v3](http://arxiv.org/pdf/2505.08230v3)**

> **作者:** Hogyun Kim; Jiwon Choi; Juwon Kim; Geonmo Yang; Dongjin Cho; Hyungtae Lim; Younggun Cho
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** Distributed LiDAR SLAM is crucial for achieving efficient robot autonomy and improving the scalability of mapping. However, two issues need to be considered when applying it in field environments: one is resource limitation, and the other is inter/intra-robot association. The resource limitation issue arises when the data size exceeds the processing capacity of the network or memory, especially when utilizing communication systems or onboard computers in the field. The inter/intra-robot association issue occurs due to the narrow convergence region of ICP under large viewpoint differences, triggering many false positive loops and ultimately resulting in an inconsistent global map for multi-robot systems. To tackle these problems, we propose a distributed LiDAR SLAM framework designed for versatile field applications, called SKiD-SLAM. Extending our previous work that solely focused on lightweight place recognition and fast and robust global registration, we present a multi-robot mapping framework that focuses on robust and lightweight inter-robot loop closure in distributed LiDAR SLAM. Through various environmental experiments, we demonstrate that our method is more robust and lightweight compared to other state-of-the-art distributed SLAM approaches, overcoming resource limitation and inter/intra-robot association issues. Also, we validated the field applicability of our approach through mapping experiments in real-world planetary emulation terrain and cave environments, which are in-house datasets. Our code will be available at https://sparolab.github.io/research/skid_slam/.
>
---
#### [replaced 026] Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.01139v2](http://arxiv.org/pdf/2409.01139v2)**

> **作者:** Erwin de Gelder; Maren Buermann; Olaf Op den Camp
>
> **备注:** Accepted for the 2024 IEEE International Automated Vehicle Validation (IAVVC 2024) Conference
>
> **摘要:** Automated Driving Systems (ADSs) have the potential to make mobility services available and safe for all. A multi-pillar Safety Assessment Framework (SAF) has been proposed for the type-approval process of ADSs. The SAF requires that the test scenarios for the ADS adequately covers the Operational Design Domain (ODD) of the ADS. A common method for generating test scenarios involves basing them on scenarios identified and characterized from driving data. This work addresses two questions when collecting scenarios from driving data. First, do the collected scenarios cover all relevant aspects of the ADS' ODD? Second, do the collected scenarios cover all relevant aspects that are in the driving data, such that no potentially important situations are missed? This work proposes coverage metrics that provide a quantitative answer to these questions. The proposed coverage metrics are illustrated by means of an experiment in which over 200000 scenarios from 10 different scenario categories are collected from the HighD data set. The experiment demonstrates that a coverage of 100 % can be achieved under certain conditions, and it also identifies which data and scenarios could be added to enhance the coverage outcomes in case a 100 % coverage has not been achieved. Whereas this work presents metrics for the quantification of the coverage of driving data and the identified scenarios, this paper concludes with future research directions, including the quantification of the completeness of driving data and the identified scenarios.
>
---
#### [replaced 027] EmojiVoice: Towards long-term controllable expressivity in robot speech
- **分类: cs.RO; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.15085v2](http://arxiv.org/pdf/2506.15085v2)**

> **作者:** Paige Tuttösí; Shivam Mehta; Zachary Syvenky; Bermet Burkanova; Gustav Eje Henter; Angelica Lim
>
> **备注:** Accepted to RO-MAN 2025, Demo at HRI 2025 : https://dl.acm.org/doi/10.5555/3721488.3721774 Project webpage here: https://rosielab.github.io/emojivoice/ Toolbox here: https://github.com/rosielab/emojivoice
>
> **摘要:** Humans vary their expressivity when speaking for extended periods to maintain engagement with their listener. Although social robots tend to be deployed with ``expressive'' joyful voices, they lack this long-term variation found in human speech. Foundation model text-to-speech systems are beginning to mimic the expressivity in human speech, but they are difficult to deploy offline on robots. We present EmojiVoice, a free, customizable text-to-speech (TTS) toolkit that allows social roboticists to build temporally variable, expressive speech on social robots. We introduce emoji-prompting to allow fine-grained control of expressivity on a phase level and use the lightweight Matcha-TTS backbone to generate speech in real-time. We explore three case studies: (1) a scripted conversation with a robot assistant, (2) a storytelling robot, and (3) an autonomous speech-to-speech interactive agent. We found that using varied emoji prompting improved the perception and expressivity of speech over a long period in a storytelling task, but expressive voice was not preferred in the assistant use case.
>
---
#### [replaced 028] Perception-aware Planning for Quadrotor Flight in Unknown and Feature-limited Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.15273v2](http://arxiv.org/pdf/2503.15273v2)**

> **作者:** Chenxin Yu; Zihong Lu; Jie Mei; Boyu Zhou
>
> **备注:** Accepted by IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Various studies on perception-aware planning have been proposed to enhance the state estimation accuracy of quadrotors in visually degraded environments. However, many existing methods heavily rely on prior environmental knowledge and face significant limitations in previously unknown environments with sparse localization features, which greatly limits their practical application. In this paper, we present a perception-aware planning method for quadrotor flight in unknown and feature-limited environments that properly allocates perception resources among environmental information during navigation. We introduce a viewpoint transition graph that allows for the adaptive selection of local target viewpoints, which guide the quadrotor to efficiently navigate to the goal while maintaining sufficient localizability and without being trapped in feature-limited regions. During the local planning, a novel yaw trajectory generation method that simultaneously considers exploration capability and localizability is presented. It constructs a localizable corridor via feature co-visibility evaluation to ensure localization robustness in a computationally efficient way. Through validations conducted in both simulation and real-world experiments, we demonstrate the feasibility and real-time performance of the proposed method. The source code will be released to benefit the community.
>
---
#### [replaced 029] Embodied Web Agents: Bridging Physical-Digital Realms for Integrated Agent Intelligence
- **分类: cs.AI; cs.CL; cs.CV; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.15677v3](http://arxiv.org/pdf/2506.15677v3)**

> **作者:** Yining Hong; Rui Sun; Bingxuan Li; Xingcheng Yao; Maxine Wu; Alexander Chien; Da Yin; Ying Nian Wu; Zhecan James Wang; Kai-Wei Chang
>
> **摘要:** AI agents today are mostly siloed - they either retrieve and reason over vast amount of digital information and knowledge obtained online; or interact with the physical world through embodied perception, planning and action - but rarely both. This separation limits their ability to solve tasks that require integrated physical and digital intelligence, such as cooking from online recipes, navigating with dynamic map data, or interpreting real-world landmarks using web knowledge. We introduce Embodied Web Agents, a novel paradigm for AI agents that fluidly bridge embodiment and web-scale reasoning. To operationalize this concept, we first develop the Embodied Web Agents task environments, a unified simulation platform that tightly integrates realistic 3D indoor and outdoor environments with functional web interfaces. Building upon this platform, we construct and release the Embodied Web Agents Benchmark, which encompasses a diverse suite of tasks including cooking, navigation, shopping, tourism, and geolocation - all requiring coordinated reasoning across physical and digital realms for systematic assessment of cross-domain intelligence. Experimental results reveal significant performance gaps between state-of-the-art AI systems and human capabilities, establishing both challenges and opportunities at the intersection of embodied cognition and web-scale knowledge access. All datasets, codes and websites are publicly available at our project page https://embodied-web-agent.github.io/.
>
---
