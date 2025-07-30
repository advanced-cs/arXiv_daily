# 机器人 cs.RO

- **最新发布 52 篇**

- **更新 40 篇**

## 最新发布

#### [new 001] When Engineering Outruns Intelligence: A Re-evaluation of Instruction-Guided Navigation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于目标导向导航任务，旨在分析大语言模型（LLM）在导航中的实际作用。作者移除了LLM相关组件，仅使用几何启发式方法DWFE，显著提升了成功率和路径质量。加入轻量语言先验SHF后效果进一步提升。研究发现，导航性能提升主要来自几何策略，而非LLM推理能力。**

- **链接: [http://arxiv.org/pdf/2507.20021v1](http://arxiv.org/pdf/2507.20021v1)**

> **作者:** Matin Aghaei; Mohammad Ali Alomrani; Yingxue Zhang; Mahdi Biparva
>
> **摘要:** Large language models (LLMs) are often credited with recent leaps in ObjectGoal Navigation, yet the extent to which they improve planning remains unclear. We revisit this question on the HM3D-v1 validation split. First, we strip InstructNav of its Dynamic Chain-of-Navigation prompt, open-vocabulary GLEE detector and Intuition saliency map, and replace them with a simple Distance-Weighted Frontier Explorer (DWFE). This geometry-only heuristic raises Success from 58.0% to 61.1% and lifts SPL from 20.9% to 36.0% over 2 000 validation episodes, outperforming all previous training-free baselines. Second, we add a lightweight language prior (SHF); on a 200-episode subset this yields a further +2% Success and +0.9% SPL while shortening paths by five steps on average. Qualitative trajectories confirm the trend: InstructNav back-tracks and times-out, DWFE reaches the goal after a few islands, and SHF follows an almost straight route. Our results indicate that frontier geometry, not emergent LLM reasoning, drives most reported gains, and suggest that metric-aware prompts or offline semantic graphs are necessary before attributing navigation success to "LLM intelligence."
>
---
#### [new 002] LanternNet: A Novel Hub-and-Spoke System to Seek and Suppress Spotted Lanternfly Populations
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文提出LanternNet，一种基于机器人与AI的中心辐射式系统，用于检测与控制入侵物种斑翅 lanternfly（SLF）种群。该系统通过计算机视觉识别SLF，并利用多个机器人执行灭虫、环境监测等任务，显著减少SLF数量并改善树木健康，属于生态治理与自动化控制任务。**

- **链接: [http://arxiv.org/pdf/2507.20800v1](http://arxiv.org/pdf/2507.20800v1)**

> **作者:** Vinil Polepalli
>
> **摘要:** The invasive spotted lanternfly (SLF) poses a significant threat to agriculture and ecosystems, causing widespread damage. Current control methods, such as egg scraping, pesticides, and quarantines, prove labor-intensive, environmentally hazardous, and inadequate for long-term SLF suppression. This research introduces LanternNet, a novel autonomous robotic Hub-and-Spoke system designed for scalable detection and suppression of SLF populations. A central, tree-mimicking hub utilizes a YOLOv8 computer vision model for precise SLF identification. Three specialized robotic spokes perform targeted tasks: pest neutralization, environmental monitoring, and navigation/mapping. Field deployment across multiple infested sites over 5 weeks demonstrated LanternNet's efficacy. Quantitative analysis revealed significant reductions (p < 0.01, paired t-tests) in SLF populations and corresponding improvements in tree health indicators across the majority of test sites. Compared to conventional methods, LanternNet offers substantial cost advantages and improved scalability. Furthermore, the system's adaptability for enhanced autonomy and targeting of other invasive species presents significant potential for broader ecological impact. LanternNet demonstrates the transformative potential of integrating robotics and AI for advanced invasive species management and improved environmental outcomes.
>
---
#### [new 003] Advancing Shared and Multi-Agent Autonomy in Underwater Missions: Integrating Knowledge Graphs and Retrieval-Augmented Generation
- **分类: cs.RO**

- **简介: 该论文属于水下机器人任务，旨在解决复杂水下环境中多机器人自主协作与人机协同问题。作者结合知识图谱与检索增强生成技术，提升机器人对环境的理解与决策能力，实现100%任务验证与行为完整性，并通过消融实验验证结构化知识对决策质量的重要性。**

- **链接: [http://arxiv.org/pdf/2507.20370v1](http://arxiv.org/pdf/2507.20370v1)**

> **作者:** Michele Grimaldi; Carlo Cernicchiaro; Sebastian Realpe Rua; Alaaeddine El-Masri-El-Chaarani; Markus Buchholz; Loizos Michael; Pere Ridao Rodriguez; Ignacio Carlucho; Yvan R. Petillot
>
> **摘要:** Robotic platforms have become essential for marine operations by providing regular and continuous access to offshore assets, such as underwater infrastructure inspection, environmental monitoring, and resource exploration. However, the complex and dynamic nature of underwater environments, characterized by limited visibility, unpredictable currents, and communication constraints, presents significant challenges that demand advanced autonomy while ensuring operator trust and oversight. Central to addressing these challenges are knowledge representation and reasoning techniques, particularly knowledge graphs and retrieval-augmented generation (RAG) systems, that enable robots to efficiently structure, retrieve, and interpret complex environmental data. These capabilities empower robotic agents to reason, adapt, and respond effectively to changing conditions. The primary goal of this work is to demonstrate both multi-agent autonomy and shared autonomy, where multiple robotic agents operate independently while remaining connected to a human supervisor. We show how a RAG-powered large language model, augmented with knowledge graph data and domain taxonomy, enables autonomous multi-agent decision-making and facilitates seamless human-robot interaction, resulting in 100\% mission validation and behavior completeness. Finally, ablation studies reveal that without structured knowledge from the graph and/or taxonomy, the LLM is prone to hallucinations, which can compromise decision quality.
>
---
#### [new 004] GABRIL: Gaze-Based Regularization for Mitigating Causal Confusion in Imitation Learning
- **分类: cs.RO**

- **简介: 该论文属于模仿学习任务，旨在解决智能体在模仿人类专家演示时出现的因果混淆问题。通过引入基于人类注视数据的正则化方法（GABRIL），引导模型关注因果特征，减少对虚假相关性的依赖。实验验证了该方法在Atari和CARLA环境中的有效性，相较基线方法性能显著提升，并增强了模型可解释性。**

- **链接: [http://arxiv.org/pdf/2507.19647v1](http://arxiv.org/pdf/2507.19647v1)**

> **作者:** Amin Banayeeanzade; Fatemeh Bahrani; Yutai Zhou; Erdem Bıyık
>
> **备注:** IROS 2025 camera-ready version. First two authors contributed equally
>
> **摘要:** Imitation Learning (IL) is a widely adopted approach which enables agents to learn from human expert demonstrations by framing the task as a supervised learning problem. However, IL often suffers from causal confusion, where agents misinterpret spurious correlations as causal relationships, leading to poor performance in testing environments with distribution shift. To address this issue, we introduce GAze-Based Regularization in Imitation Learning (GABRIL), a novel method that leverages the human gaze data gathered during the data collection phase to guide the representation learning in IL. GABRIL utilizes a regularization loss which encourages the model to focus on causally relevant features identified through expert gaze and consequently mitigates the effects of confounding variables. We validate our approach in Atari environments and the Bench2Drive benchmark in CARLA by collecting human gaze datasets and applying our method in both domains. Experimental results show that the improvement of GABRIL over behavior cloning is around 179% more than the same number for other baselines in the Atari and 76% in the CARLA setup. Finally, we show that our method provides extra explainability when compared to regular IL agents.
>
---
#### [new 005] A 4D Radar Camera Extrinsic Calibration Tool Based on 3D Uncertainty Perspective N Points
- **分类: cs.RO**

- **简介: 论文任务是4D雷达与相机的外参标定。要解决雷达与相机间精确标定困难的问题。工作提出了一种考虑3D不确定性传播的PnP算法（3DUPnP），有效建模雷达噪声并提升标定精度，实验验证了其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.19829v1](http://arxiv.org/pdf/2507.19829v1)**

> **作者:** Chuan Cao; Xiaoning Wang; Wenqian Xi; Han Zhang; Weidong Chen; Jingchuan Wang
>
> **摘要:** 4D imaging radar is a type of low-cost millimeter-wave radar(costing merely 10-20$\%$ of lidar systems) capable of providing range, azimuth, elevation, and Doppler velocity information. Accurate extrinsic calibration between millimeter-wave radar and camera systems is critical for robust multimodal perception in robotics, yet remains challenging due to inherent sensor noise characteristics and complex error propagation. This paper presents a systematic calibration framework to address critical challenges through a spatial 3d uncertainty-aware PnP algorithm (3DUPnP) that explicitly models spherical coordinate noise propagation in radar measurements, then compensating for non-zero error expectations during coordinate transformations. Finally, experimental validation demonstrates significant performance improvements over state-of-the-art CPnP baseline, including improved consistency in simulations and enhanced precision in physical experiments. This study provides a robust calibration solution for robotic systems equipped with millimeter-wave radar and cameras, tailored specifically for autonomous driving and robotic perception applications.
>
---
#### [new 006] Humanoid Occupancy: Enabling A Generalized Multimodal Occupancy Perception System on Humanoid Robots
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于环境感知任务，旨在解决人形机器人在复杂环境中实现全面环境理解的问题。论文提出了Humanoid Occupancy系统，结合多模态融合技术与传感器布局策略，生成语义与几何信息融合的网格化占据输出，并构建了全景占据数据集，为人形机器人环境感知提供技术基础与数据支持。**

- **链接: [http://arxiv.org/pdf/2507.20217v1](http://arxiv.org/pdf/2507.20217v1)**

> **作者:** Wei Cui; Haoyu Wang; Wenkang Qin; Yijie Guo; Gang Han; Wen Zhao; Jiahang Cao; Zhang Zhang; Jiaru Zhong; Jingkai Sun; Pihai Sun; Shuai Shi; Botuo Jiang; Jiahao Ma; Jiaxu Wang; Hao Cheng; Zhichao Liu; Yang Wang; Zheng Zhu; Guan Huang; Jian Tang; Qiang Zhang
>
> **备注:** Tech Report
>
> **摘要:** Humanoid robot technology is advancing rapidly, with manufacturers introducing diverse heterogeneous visual perception modules tailored to specific scenarios. Among various perception paradigms, occupancy-based representation has become widely recognized as particularly suitable for humanoid robots, as it provides both rich semantic and 3D geometric information essential for comprehensive environmental understanding. In this work, we present Humanoid Occupancy, a generalized multimodal occupancy perception system that integrates hardware and software components, data acquisition devices, and a dedicated annotation pipeline. Our framework employs advanced multi-modal fusion techniques to generate grid-based occupancy outputs encoding both occupancy status and semantic labels, thereby enabling holistic environmental understanding for downstream tasks such as task planning and navigation. To address the unique challenges of humanoid robots, we overcome issues such as kinematic interference and occlusion, and establish an effective sensor layout strategy. Furthermore, we have developed the first panoramic occupancy dataset specifically for humanoid robots, offering a valuable benchmark and resource for future research and development in this domain. The network architecture incorporates multi-modal feature fusion and temporal information integration to ensure robust perception. Overall, Humanoid Occupancy delivers effective environmental perception for humanoid robots and establishes a technical foundation for standardizing universal visual modules, paving the way for the widespread deployment of humanoid robots in complex real-world scenarios.
>
---
#### [new 007] Feeling the Force: A Nuanced Physics-based Traversability Sensor for Navigation in Unstructured Vegetation
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人导航任务，旨在解决机器人在复杂植被环境中安全通行的问题。通过设计一种可测量植被反作用力的传感器，实现对植被可穿越性的量化评估，为导航决策提供依据。**

- **链接: [http://arxiv.org/pdf/2507.19831v1](http://arxiv.org/pdf/2507.19831v1)**

> **作者:** Zaar Khizar; Johann Laconte; Roland Lenain; Romuald Aufrere
>
> **摘要:** In many applications, robots are increasingly deployed in unstructured and natural environments where they encounter various types of vegetation. Vegetation presents unique challenges as a traversable obstacle, where the mechanical properties of the plants can influence whether a robot can safely collide with and overcome the obstacle. A more nuanced approach is required to assess the safety and traversability of these obstacles, as collisions can sometimes be safe and necessary for navigating through dense or unavoidable vegetation. This paper introduces a novel sensor designed to directly measure the applied forces exerted by vegetation on a robot: by directly capturing the push-back forces, our sensor provides a detailed understanding of the interactions between the robot and its surroundings. We demonstrate the sensor's effectiveness through experimental validations, showcasing its ability to measure subtle force variations. This force-based approach provides a quantifiable metric that can inform navigation decisions and serve as a foundation for developing future learning algorithms.
>
---
#### [new 008] Think, Act, Learn: A Framework for Autonomous Robotic Agents using Closed-Loop Large Language Models
- **分类: cs.RO; cs.HC; 68T05, 68T07, 68T40; I.2.6; I.2.9; I.2.7; I.2.10; H.5.2**

- **简介: 论文提出“Think, Act, Learn”（T-A-L）框架，属于机器人自主学习任务，旨在解决当前大语言模型（LLM）在机器人控制中开环操作导致的适应性差问题。通过闭环系统实现任务规划、执行与反馈学习，使机器人能自主反思并改进策略，显著提升复杂任务成功率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.19854v1](http://arxiv.org/pdf/2507.19854v1)**

> **作者:** Anjali R. Menon; Rohit K. Sharma; Priya Singh; Chengyu Wang; Aurora M. Ferreira; Mateja Novak
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** The integration of Large Language Models (LLMs) into robotics has unlocked unprecedented capabilities in high-level task planning. However, most current systems operate in an open-loop fashion, where LLMs act as one-shot planners, rendering them brittle and unable to adapt to unforeseen circumstances in dynamic physical environments. To overcome this limitation, this paper introduces the "Think, Act, Learn" (T-A-L) framework, a novel architecture that enables an embodied agent to autonomously learn and refine its policies through continuous interaction. Our framework establishes a closed-loop cycle where an LLM first "thinks" by decomposing high-level commands into actionable plans. The robot then "acts" by executing these plans while gathering rich, multimodal sensory feedback. Critically, the "learn" module processes this feedback to facilitate LLM-driven self-reflection, allowing the agent to perform causal analysis on its failures and generate corrective strategies. These insights are stored in an experiential memory to guide future planning cycles. We demonstrate through extensive experiments in both simulation and the real world that our T-A-L agent significantly outperforms baseline methods, including open-loop LLMs, Behavioral Cloning, and traditional Reinforcement Learning. Our framework achieves over a 97% success rate on complex, long-horizon tasks, converges to a stable policy in an average of just 9 trials, and exhibits remarkable generalization to unseen tasks. This work presents a significant step towards developing more robust, adaptive, and truly autonomous robotic agents.
>
---
#### [new 009] Ag2x2: Robust Agent-Agnostic Visual Representations for Zero-Shot Bimanual Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人双臂操作任务，旨在解决无需专家演示或设计奖励函数的零样本学习问题。作者提出了Ag2x2框架，通过协调感知的视觉表征，同时编码物体状态与手部运动模式，实现高效双臂操作。实验表明其在多种任务中表现优异，支持模仿学习，提升了复杂技能的可扩展性学习能力。**

- **链接: [http://arxiv.org/pdf/2507.19817v1](http://arxiv.org/pdf/2507.19817v1)**

> **作者:** Ziyin Xiong; Yinghan Chen; Puhao Li; Yixin Zhu; Tengyu Liu; Siyuan Huang
>
> **备注:** Accepted to IROS 2025, oral presentation. Project page link: https://ziyin-xiong.github.io/ag2x2.github.io/
>
> **摘要:** Bimanual manipulation, fundamental to human daily activities, remains a challenging task due to its inherent complexity of coordinated control. Recent advances have enabled zero-shot learning of single-arm manipulation skills through agent-agnostic visual representations derived from human videos; however, these methods overlook crucial agent-specific information necessary for bimanual coordination, such as end-effector positions. We propose Ag2x2, a computational framework for bimanual manipulation through coordination-aware visual representations that jointly encode object states and hand motion patterns while maintaining agent-agnosticism. Extensive experiments demonstrate that Ag2x2 achieves a 73.5% success rate across 13 diverse bimanual tasks from Bi-DexHands and PerAct2, including challenging scenarios with deformable objects like ropes. This performance outperforms baseline methods and even surpasses the success rate of policies trained with expert-engineered rewards. Furthermore, we show that representations learned through Ag2x2 can be effectively leveraged for imitation learning, establishing a scalable pipeline for skill acquisition without expert supervision. By maintaining robust performance across diverse tasks without human demonstrations or engineered rewards, Ag2x2 represents a step toward scalable learning of complex bimanual robotic skills.
>
---
#### [new 010] A Strawberry Harvesting Tool with Minimal Footprint
- **分类: cs.RO**

- **简介: 论文设计了一种用于采摘立体栽培草莓的轻型工具，通过激光精准切割果茎，减少接触面积。该工具可高温杀菌，延缓果实脱水，提升保鲜期。研究优化了激光参数，实现平均切割时间2.88秒，验证了可行性。**

- **链接: [http://arxiv.org/pdf/2507.20784v1](http://arxiv.org/pdf/2507.20784v1)**

> **作者:** Mohamed Sorour; Mohamed Heshmat; Khaled Elgeneidy; Pål Johan From
>
> **摘要:** In this paper, a novel prototype for harvesting table-top grown strawberries is presented, that is minimalist in its footprint interacting with the fruit. In our methodology, a smooth trapper manipulates the stem into a precise groove location at which a distant laser beam is focused. The tool reaches temperatures as high as 188{\deg} Celsius and as such killing germs and preventing the spread of local plant diseases. The burnt stem wound preserves water content and in turn the fruit shelf life. Cycle and cut times achieved are 5.56 and 2.88 seconds respectively in successful in-door harvesting demonstration. Extensive experiments are performed to optimize the laser spot diameter and lateral speed against the cutting time.
>
---
#### [new 011] Uncertainty-aware Planning with Inaccurate Models for Robotized Liquid Handling
- **分类: cs.RO**

- **简介: 该论文属于机器人液体操作任务，旨在解决模型不准确导致的规划不可靠问题。论文提出了一种考虑不确定性的蒙特卡洛树搜索（MCTS）算法，通过估计模型不确定性并将其纳入规划过程，提高了在未知或变化环境下的任务成功率。**

- **链接: [http://arxiv.org/pdf/2507.20861v1](http://arxiv.org/pdf/2507.20861v1)**

> **作者:** Marco Faroni; Carlo Odesco; Andrea Zanchettin; Paolo Rocco
>
> **备注:** Accepted at IEEE/RSJ IROS 2025
>
> **摘要:** Physics-based simulations and learning-based models are vital for complex robotics tasks like deformable object manipulation and liquid handling. However, these models often struggle with accuracy due to epistemic uncertainty or the sim-to-real gap. For instance, accurately pouring liquid from one container to another poses challenges, particularly when models are trained on limited demonstrations and may perform poorly in novel situations. This paper proposes an uncertainty-aware Monte Carlo Tree Search (MCTS) algorithm designed to mitigate these inaccuracies. By incorporating estimates of model uncertainty, the proposed MCTS strategy biases the search towards actions with lower predicted uncertainty. This approach enhances the reliability of planning under uncertain conditions. Applied to a liquid pouring task, our method demonstrates improved success rates even with models trained on minimal data, outperforming traditional methods and showcasing its potential for robust decision-making in robotics.
>
---
#### [new 012] Skin-Machine Interface with Multimodal Contact Motion Classifier
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决如何利用皮肤传感器实现对复杂机器人操作控制的问题。论文提出了一种基于多模态触觉信息和循环神经网络的接触动作分类框架，并通过硬件设计提升分类准确率，最终实现机器人多样化动作控制。**

- **链接: [http://arxiv.org/pdf/2507.19760v1](http://arxiv.org/pdf/2507.19760v1)**

> **作者:** Alberto Confente; Takanori Jin; Taisuke Kobayashi; Julio Rogelio Guadarrama-Olvera; Gordon Cheng
>
> **备注:** 8 pages, 8 figures (accepted in Humanoids2025)
>
> **摘要:** This paper proposes a novel framework for utilizing skin sensors as a new operation interface of complex robots. The skin sensors employed in this study possess the capability to quantify multimodal tactile information at multiple contact points. The time-series data generated from these sensors is anticipated to facilitate the classification of diverse contact motions exhibited by an operator. By mapping the classification results with robot motion primitives, a diverse range of robot motions can be generated by altering the manner in which the skin sensors are interacted with. In this paper, we focus on a learning-based contact motion classifier employing recurrent neural networks. This classifier is a pivotal factor in the success of this framework. Furthermore, we elucidate the requisite conditions for software-hardware designs. Firstly, multimodal sensing and its comprehensive encoding significantly contribute to the enhancement of classification accuracy and learning stability. Utilizing all modalities simultaneously as inputs to the classifier proves to be an effective approach. Secondly, it is essential to mount the skin sensors on a flexible and compliant support to enable the activation of three-axis accelerometers. These accelerometers are capable of measuring horizontal tactile information, thereby enhancing the correlation with other modalities. Furthermore, they serve to absorb the noises generated by the robot's movements during deployment. Through these discoveries, the accuracy of the developed classifier surpassed 95 %, enabling the dual-arm mobile manipulator to execute a diverse range of tasks via the Skin-Machine Interface. https://youtu.be/UjUXT4Z4BC8
>
---
#### [new 013] Learning Physical Interaction Skills from Human Demonstrations
- **分类: cs.RO**

- **简介: 该论文属于机器人学习与模仿任务，旨在解决不同形态代理间物理交互技能学习的问题。通过提取交互动力学的紧凑表示（EIG），使多样化形态的智能体能从人类演示中学习并复现互动行为。**

- **链接: [http://arxiv.org/pdf/2507.20445v1](http://arxiv.org/pdf/2507.20445v1)**

> **作者:** Tianyu Li; Hengbo Ma; Sehoon Ha; Kwonjoon Lee
>
> **摘要:** Learning physical interaction skills, such as dancing, handshaking, or sparring, remains a fundamental challenge for agents operating in human environments, particularly when the agent's morphology differs significantly from that of the demonstrator. Existing approaches often rely on handcrafted objectives or morphological similarity, limiting their capacity for generalization. Here, we introduce a framework that enables agents with diverse embodiments to learn wholebbody interaction behaviors directly from human demonstrations. The framework extracts a compact, transferable representation of interaction dynamics, called the Embedded Interaction Graph (EIG), which captures key spatiotemporal relationships between the interacting agents. This graph is then used as an imitation objective to train control policies in physics-based simulations, allowing the agent to generate motions that are both semantically meaningful and physically feasible. We demonstrate BuddyImitation on multiple agents, such as humans, quadrupedal robots with manipulators, or mobile manipulators and various interaction scenarios, including sparring, handshaking, rock-paper-scissors, or dancing. Our results demonstrate a promising path toward coordinated behaviors across morphologically distinct characters via cross embodiment interaction learning.
>
---
#### [new 014] Model-Structured Neural Networks to Control the Steering Dynamics of Autonomous Race Cars
- **分类: cs.RO; J.2; I.2; I.6**

- **简介: 该论文属于自动驾驶控制任务，旨在提升自主赛车转向控制的安全性与鲁棒性。论文提出了一种融合车辆动力学知识的神经网络模型MS-NN-steer，解决了传统黑箱模型在小数据集上泛化能力差和对初始化敏感的问题，并在真实赛事数据中验证了其优越性能。**

- **链接: [http://arxiv.org/pdf/2507.20427v1](http://arxiv.org/pdf/2507.20427v1)**

> **作者:** Mattia Piccinini; Aniello Mungiello; Georg Jank; Gastone Pietro Rosati Papini; Francesco Biral; Johannes Betz
>
> **备注:** Accepted at the 2025 IEEE International Conference on Intelligent Transportation Systems (ITSC)
>
> **摘要:** Autonomous racing has gained increasing attention in recent years, as a safe environment to accelerate the development of motion planning and control methods for autonomous driving. Deep learning models, predominantly based on neural networks (NNs), have demonstrated significant potential in modeling the vehicle dynamics and in performing various tasks in autonomous driving. However, their black-box nature is critical in the context of autonomous racing, where safety and robustness demand a thorough understanding of the decision-making algorithms. To address this challenge, this paper proposes MS-NN-steer, a new Model-Structured Neural Network for vehicle steering control, integrating the prior knowledge of the nonlinear vehicle dynamics into the neural architecture. The proposed controller is validated using real-world data from the Abu Dhabi Autonomous Racing League (A2RL) competition, with full-scale autonomous race cars. In comparison with general-purpose NNs, MS-NN-steer is shown to achieve better accuracy and generalization with small training datasets, while being less sensitive to the weights' initialization. Also, MS-NN-steer outperforms the steering controller used by the A2RL winning team. Our implementation is available open-source in a GitHub repository.
>
---
#### [new 015] PhysVarMix: Physics-Informed Variational Mixture Model for Multi-Modal Trajectory Prediction
- **分类: cs.RO; stat.ML**

- **简介: 该论文属于轨迹预测任务，旨在解决复杂城市环境中未来路径多模态预测的问题。作者提出PhysVarMix方法，结合学习与物理约束，通过变分贝叶斯混合模型生成多样且物理合理的预测轨迹，提升自动驾驶系统的决策能力。**

- **链接: [http://arxiv.org/pdf/2507.19701v1](http://arxiv.org/pdf/2507.19701v1)**

> **作者:** Haichuan Li; Tomi Westerlund
>
> **摘要:** Accurate prediction of future agent trajectories is a critical challenge for ensuring safe and efficient autonomous navigation, particularly in complex urban environments characterized by multiple plausible future scenarios. In this paper, we present a novel hybrid approach that integrates learning-based with physics-based constraints to address the multi-modality inherent in trajectory prediction. Our method employs a variational Bayesian mixture model to effectively capture the diverse range of potential future behaviors, moving beyond traditional unimodal assumptions. Unlike prior approaches that predominantly treat trajectory prediction as a data-driven regression task, our framework incorporates physical realism through sector-specific boundary conditions and Model Predictive Control (MPC)-based smoothing. These constraints ensure that predicted trajectories are not only data-consistent but also physically plausible, adhering to kinematic and dynamic principles. Furthermore, our method produces interpretable and diverse trajectory predictions, enabling enhanced downstream decision-making and planning in autonomous driving systems. We evaluate our approach on two benchmark datasets, demonstrating superior performance compared to existing methods. Comprehensive ablation studies validate the contributions of each component and highlight their synergistic impact on prediction accuracy and reliability. By balancing data-driven insights with physics-informed constraints, our approach offers a robust and scalable solution for navigating the uncertainties of real-world urban environments.
>
---
#### [new 016] Tactile-Guided Robotic Ultrasound: Mapping Preplanned Scan Paths for Intercostal Imaging
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决超声成像中肋骨间隙扫描路径生成困难的问题。通过利用触觉信号构建肋骨表面点云，结合插值与配准技术，实现个体化扫描路径规划，并提出自动调整倾角方法以提高成像完整性。**

- **链接: [http://arxiv.org/pdf/2507.20282v1](http://arxiv.org/pdf/2507.20282v1)**

> **作者:** Yifan Zhang; Dianye Huang; Nassir Navab; Zhongliang Jiang
>
> **备注:** Accepted by IROS2025, video link: https://youtu.be/SBwpFVzEhAg
>
> **摘要:** Medical ultrasound (US) imaging is widely used in clinical examinations due to its portability, real-time capability, and radiation-free nature. To address inter- and intra-operator variability, robotic ultrasound systems have gained increasing attention. However, their application in challenging intercostal imaging remains limited due to the lack of an effective scan path generation method within the constrained acoustic window. To overcome this challenge, we explore the potential of tactile cues for characterizing subcutaneous rib structures as an alternative signal for ultrasound segmentation-free bone surface point cloud extraction. Compared to 2D US images, 1D tactile-related signals offer higher processing efficiency and are less susceptible to acoustic noise and artifacts. By leveraging robotic tracking data, a sparse tactile point cloud is generated through a few scans along the rib, mimicking human palpation. To robustly map the scanning trajectory into the intercostal space, the sparse tactile bone location point cloud is first interpolated to form a denser representation. This refined point cloud is then registered to an image-based dense bone surface point cloud, enabling accurate scan path mapping for individual patients. Additionally, to ensure full coverage of the object of interest, we introduce an automated tilt angle adjustment method to visualize structures beneath the bone. To validate the proposed method, we conducted comprehensive experiments on four distinct phantoms. The final scanning waypoint mapping achieved Mean Nearest Neighbor Distance (MNND) and Hausdorff distance (HD) errors of 3.41 mm and 3.65 mm, respectively, while the reconstructed object beneath the bone had errors of 0.69 mm and 2.2 mm compared to the CT ground truth.
>
---
#### [new 017] Homotopy-aware Multi-agent Navigation via Distributed Model Predictive Control
- **分类: cs.RO; cs.MA; cs.SY; eess.SY**

- **简介: 该论文属于多智能体导航任务，旨在解决高密度环境中多智能体路径规划中的死锁问题。论文提出了一种结合同伦感知路径规划与分布式模型预测控制的分层框架，通过全局路径选择与局部轨迹优化协同，有效提升导航效率与成功率。**

- **链接: [http://arxiv.org/pdf/2507.19860v1](http://arxiv.org/pdf/2507.19860v1)**

> **作者:** Haoze Dong; Meng Guo; Chengyi He; Zhongkui Li
>
> **摘要:** Multi-agent trajectory planning requires ensuring both safety and efficiency, yet deadlocks remain a significant challenge, especially in obstacle-dense environments. Such deadlocks frequently occur when multiple agents attempt to traverse the same long and narrow corridor simultaneously. To address this, we propose a novel distributed trajectory planning framework that bridges the gap between global path and local trajectory cooperation. At the global level, a homotopy-aware optimal path planning algorithm is proposed, which fully leverages the topological structure of the environment. A reference path is chosen from distinct homotopy classes by considering both its spatial and temporal properties, leading to improved coordination among agents globally. At the local level, a model predictive control-based trajectory optimization method is used to generate dynamically feasible and collision-free trajectories. Additionally, an online replanning strategy ensures its adaptability to dynamic environments. Simulations and experiments validate the effectiveness of our approach in mitigating deadlocks. Ablation studies demonstrate that by incorporating time-aware homotopic properties into the underlying global paths, our method can significantly reduce deadlocks and improve the average success rate from 4%-13% to over 90% in randomly generated dense scenarios.
>
---
#### [new 018] High-Speed Event Vision-Based Tactile Roller Sensor for Large Surface Measurements
- **分类: cs.RO**

- **简介: 该论文属于工业检测任务，旨在解决大型表面（如飞机机身）高精度、高速度的3D几何测量问题。传统视觉触觉传感器受限于速度与连续性，本文提出结合神经形态相机与滚动机构的新型触觉传感器，实现高速连续高分辨率扫描，并通过新算法提升精度与特征识别能力。**

- **链接: [http://arxiv.org/pdf/2507.19914v1](http://arxiv.org/pdf/2507.19914v1)**

> **作者:** Akram Khairi; Hussain Sajwani; Abdallah Mohammad Alkilany; Laith AbuAssi; Mohamad Halwani; Islam Mohamed Zaid; Ahmed Awadalla; Dewald Swart; Abdulla Ayyad; Yahya Zweiri
>
> **备注:** 14 pages, 11 figures
>
> **摘要:** Inspecting large-scale industrial surfaces like aircraft fuselages for quality control requires capturing their precise 3D surface geometry at high resolution. Vision-based tactile sensors (VBTSs) offer high local resolution but require slow 'press-and-lift' measurements stitched for large areas. Approaches with sliding or roller/belt VBTS designs provide measurements continuity. However, they face significant challenges respectively: sliding struggles with friction/wear and both approaches are speed-limited by conventional camera frame rates and motion blur, making large-area scanning time consuming. Thus, a rapid, continuous, high-resolution method is needed. We introduce a novel tactile sensor integrating a neuromorphic camera in a rolling mechanism to achieve this. Leveraging its high temporal resolution and robustness to motion blur, our system uses a modified event-based multi-view stereo approach for 3D reconstruction. We demonstrate state-of-the-art scanning speeds up to 0.5 m/s, achieving Mean Absolute Error below 100 microns -- 11 times faster than prior continuous tactile sensing methods. A multi-reference Bayesian fusion strategy enhances accuracy (reducing MAE by 25.2\% compared to EMVS) and mitigates curvature errors. We also validate high-speed feature recognition via Braille reading 2.6 times faster than previous approaches.
>
---
#### [new 019] Large-Scale LiDAR-Inertial Dataset for Degradation-Robust High-Precision Mapping
- **分类: cs.RO; 68T40; I.2.9**

- **简介: 该论文属于定位与建图任务，旨在解决现有LIO系统在复杂真实场景中验证不足的问题。作者构建了一个大规模、高精度的LiDAR-惯性里程计数据集，涵盖多样环境和长轨迹，并通过融合SLAM与RTK-GNSS提供高精度真值，用于评估LIO系统在实际高精度地图构建中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.20516v1](http://arxiv.org/pdf/2507.20516v1)**

> **作者:** Xiaofeng Jin; Ningbo Bu; Shijie Wang; Jianfei Ge; Jiangjian Xiao; Matteo Matteucci
>
> **备注:** 9 pages,7 figures, 6 tables
>
> **摘要:** This paper introduces a large-scale, high-precision LiDAR-Inertial Odometry (LIO) dataset, aiming to address the insufficient validation of LIO systems in complex real-world scenarios in existing research. The dataset covers four diverse real-world environments spanning 60,000 to 750,000 square meters, collected using a custom backpack-mounted platform equipped with multi-beam LiDAR, an industrial-grade IMU, and RTK-GNSS modules. The dataset includes long trajectories, complex scenes, and high-precision ground truth, generated by fusing SLAM-based optimization with RTK-GNSS anchoring, and validated for trajectory accuracy through the integration of oblique photogrammetry and RTK-GNSS. This dataset provides a comprehensive benchmark for evaluating the generalization ability of LIO systems in practical high-precision mapping scenarios.
>
---
#### [new 020] LLMs-guided adaptive compensator: Bringing Adaptivity to Automatic Control Systems with Large Language Models
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文属于自动控制任务，旨在解决传统自适应控制器设计复杂、适应性差的问题。作者提出了一种基于大语言模型（LLM）的自适应补偿器框架，通过引导LLM利用系统误差设计补偿器，提升控制系统的适应性和泛化能力。实验表明其性能优于传统方法。**

- **链接: [http://arxiv.org/pdf/2507.20509v1](http://arxiv.org/pdf/2507.20509v1)**

> **作者:** Zhongchao Zhou; Yuxi Lu; Yaonan Zhu; Yifan Zhao; Bin He; Liang He; Wenwen Yu; Yusuke Iwasawa
>
> **摘要:** With rapid advances in code generation, reasoning, and problem-solving, Large Language Models (LLMs) are increasingly applied in robotics. Most existing work focuses on high-level tasks such as task decomposition. A few studies have explored the use of LLMs in feedback controller design; however, these efforts are restricted to overly simplified systems, fixed-structure gain tuning, and lack real-world validation. To further investigate LLMs in automatic control, this work targets a key subfield: adaptive control. Inspired by the framework of model reference adaptive control (MRAC), we propose an LLM-guided adaptive compensator framework that avoids designing controllers from scratch. Instead, the LLMs are prompted using the discrepancies between an unknown system and a reference system to design a compensator that aligns the response of the unknown system with that of the reference, thereby achieving adaptivity. Experiments evaluate five methods: LLM-guided adaptive compensator, LLM-guided adaptive controller, indirect adaptive control, learning-based adaptive control, and MRAC, on soft and humanoid robots in both simulated and real-world environments. Results show that the LLM-guided adaptive compensator outperforms traditional adaptive controllers and significantly reduces reasoning complexity compared to the LLM-guided adaptive controller. The Lyapunov-based analysis and reasoning-path inspection demonstrate that the LLM-guided adaptive compensator enables a more structured design process by transforming mathematical derivation into a reasoning task, while exhibiting strong generalizability, adaptability, and robustness. This study opens a new direction for applying LLMs in the field of automatic control, offering greater deployability and practicality compared to vision-language models.
>
---
#### [new 021] Hanging Around: Cognitive Inspired Reasoning for Reactive Robotics
- **分类: cs.RO**

- **简介: 该论文属于机器人任务规划与认知推理领域，旨在解决机器人在动态环境中对物体部分的识别与知识扩展问题。研究提出了一种神经符号系统，结合图像处理与本体推理，使机器人通过观察自主学习物体部分（如“把手”），并用于任务规划。**

- **链接: [http://arxiv.org/pdf/2507.20832v1](http://arxiv.org/pdf/2507.20832v1)**

> **作者:** Mihai Pomarlan; Stefano De Giorgis; Rachel Ringe; Maria M. Hedblom; Nikolaos Tsiogkas
>
> **备注:** This article is published online with Open Access by IOS Press and distributed under the terms of the Creative Commons Attribution Non-Commercial License 4.0 (CC BY-NC 4.0)
>
> **摘要:** Situationally-aware artificial agents operating with competence in natural environments face several challenges: spatial awareness, object affordance detection, dynamic changes and unpredictability. A critical challenge is the agent's ability to identify and monitor environmental elements pertinent to its objectives. Our research introduces a neurosymbolic modular architecture for reactive robotics. Our system combines a neural component performing object recognition over the environment and image processing techniques such as optical flow, with symbolic representation and reasoning. The reasoning system is grounded in the embodied cognition paradigm, via integrating image schematic knowledge in an ontological structure. The ontology is operatively used to create queries for the perception system, decide on actions, and infer entities' capabilities derived from perceptual data. The combination of reasoning and image processing allows the agent to focus its perception for normal operation as well as discover new concepts for parts of objects involved in particular interactions. The discovered concepts allow the robot to autonomously acquire training data and adjust its subsymbolic perception to recognize the parts, as well as making planning for more complex tasks feasible by focusing search on those relevant object parts. We demonstrate our approach in a simulated world, in which an agent learns to recognize parts of objects involved in support relations. While the agent has no concept of handle initially, by observing examples of supported objects hanging from a hook it learns to recognize the parts involved in establishing support and becomes able to plan the establishment/destruction of the support relation. This underscores the agent's capability to expand its knowledge through observation in a systematic way, and illustrates the potential of combining deep reasoning [...].
>
---
#### [new 022] Free Energy-Inspired Cognitive Risk Integration for AV Navigation in Pedestrian-Rich Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶行为规划任务，旨在解决复杂行人环境中自动驾驶车辆（AV）的交互决策问题。论文提出了一种基于自由能原理的认知风险建模框架，分别用于行人轨迹预测和AV决策，提升了AV在多行人环境中的安全性、效率与行驶平顺性。**

- **链接: [http://arxiv.org/pdf/2507.20850v1](http://arxiv.org/pdf/2507.20850v1)**

> **作者:** Meiting Dang; Yanping Wu; Yafei Wang; Dezong Zhao; David Flynn; Chongfeng Wei
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** Recent advances in autonomous vehicle (AV) behavior planning have shown impressive social interaction capabilities when interacting with other road users. However, achieving human-like prediction and decision-making in interactions with vulnerable road users remains a key challenge in complex multi-agent interactive environments. Existing research focuses primarily on crowd navigation for small mobile robots, which cannot be directly applied to AVs due to inherent differences in their decision-making strategies and dynamic boundaries. Moreover, pedestrians in these multi-agent simulations follow fixed behavior patterns that cannot dynamically respond to AV actions. To overcome these limitations, this paper proposes a novel framework for modeling interactions between the AV and multiple pedestrians. In this framework, a cognitive process modeling approach inspired by the Free Energy Principle is integrated into both the AV and pedestrian models to simulate more realistic interaction dynamics. Specifically, the proposed pedestrian Cognitive-Risk Social Force Model adjusts goal-directed and repulsive forces using a fused measure of cognitive uncertainty and physical risk to produce human-like trajectories. Meanwhile, the AV leverages this fused risk to construct a dynamic, risk-aware adjacency matrix for a Graph Convolutional Network within a Soft Actor-Critic architecture, allowing it to make more reasonable and informed decisions. Simulation results indicate that our proposed framework effectively improves safety, efficiency, and smoothness of AV navigation compared to the state-of-the-art method.
>
---
#### [new 023] Bipedalism for Quadrupedal Robots: Versatile Loco-Manipulation through Risk-Adaptive Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决四足机器人在操作时牺牲运动能力的问题。通过引入双足站立方式，使前腿可自由操作环境。论文提出一种基于风险自适应强化学习的控制框架，提升机器人在不稳定站立状态下的操作性能与鲁棒性，并在仿真和实际机器人上验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.20382v1](http://arxiv.org/pdf/2507.20382v1)**

> **作者:** Yuyou Zhang; Radu Corcodel; Ding Zhao
>
> **备注:** Humanoids 2025
>
> **摘要:** Loco-manipulation of quadrupedal robots has broadened robotic applications, but using legs as manipulators often compromises locomotion, while mounting arms complicates the system. To mitigate this issue, we introduce bipedalism for quadrupedal robots, thus freeing the front legs for versatile interactions with the environment. We propose a risk-adaptive distributional Reinforcement Learning (RL) framework designed for quadrupedal robots walking on their hind legs, balancing worst-case conservativeness with optimal performance in this inherently unstable task. During training, the adaptive risk preference is dynamically adjusted based on the uncertainty of the return, measured by the coefficient of variation of the estimated return distribution. Extensive experiments in simulation show our method's superior performance over baselines. Real-world deployment on a Unitree Go2 robot further demonstrates the versatility of our policy, enabling tasks like cart pushing, obstacle probing, and payload transport, while showcasing robustness against challenging dynamics and external disturbances.
>
---
#### [new 024] Digital and Robotic Twinning for Validation of Proximity Operations and Formation Flying
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于航天器导航与控制验证任务，旨在解决RPO和FF任务中GNC系统难以验证的问题。工作内容是构建数字与机器人双胞胎框架，结合多个测试平台，实现软硬件在环测试，验证系统性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.20034v1](http://arxiv.org/pdf/2507.20034v1)**

> **作者:** Aviad Golan; Gregory Zin; Zahra Ahmed; Emily Bates; Toby Bell; Pol Francesch Huc; Samuel Y. W. Low; Juergen Bosse; Simone D'Amico
>
> **备注:** 23 pages, 12 figures. 2025 Astrodynamics Specialist Conference
>
> **摘要:** In spacecraft Rendezvous, Proximity Operations (RPO), and Formation Flying (FF), the Guidance Navigation and Control (GNC) system is safety-critical and must meet strict performance requirements. However, validating such systems is challenging due to the complexity of the space environment, necessitating a verification and validation (V&V) process that bridges simulation and real-world behavior. The key contribution of this paper is a unified, end-to-end digital and robotic twinning framework that enables software- and hardware-in-the-loop testing for multi-modal GNC systems. The robotic twin includes three testbeds at Stanford's Space Rendezvous Laboratory (SLAB): the GNSS and Radiofrequency Autonomous Navigation Testbed for Distributed Space Systems (GRAND) to validate RF-based navigation techniques, and the Testbed for Rendezvous and Optical Navigation (TRON) and Optical Stimulator (OS) to validate vision-based methods. The test article for this work is an integrated multi-modal GNC software stack for RPO and FF developed at SLAB. This paper introduces the hybrid framework and summarizes calibration and error characterization for the robotic twin. Then, the GNC stack's performance and robustness is characterized using the integrated digital and robotic twinning pipeline for a full-range RPO mission scenario in Low-Earth Orbit (LEO). The results shown in the paper demonstrate consistency between digital and robotic twins, validating the hybrid twinning pipeline as a reliable framework for realistic assessment and verification of GNC systems.
>
---
#### [new 025] DOA: A Degeneracy Optimization Agent with Adaptive Pose Compensation Capability based on Deep Reinforcement Learning
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **简介: 论文研究2D-SLAM在室内定位中的退化问题，提出基于深度强化学习的自适应退化优化代理DOA。通过PPO训练，解决数据获取瓶颈、样本质量下降和标注协议模糊三大挑战，设计专用奖励函数与线性插值方法，动态调整传感器贡献，结合迁移学习提升跨环境泛化能力，有效优化退化场景下的姿态估计。**

- **链接: [http://arxiv.org/pdf/2507.19742v1](http://arxiv.org/pdf/2507.19742v1)**

> **作者:** Yanbin Li; Canran Xiao; Hongyang He; Shenghai Yuan; Zong Ke; Jiajie Yu; Zixiong Qin; Zhiguo Zhang; Wenzheng Chi; Wei Zhang
>
> **备注:** 10 pages,9 figures
>
> **摘要:** Particle filter-based 2D-SLAM is widely used in indoor localization tasks due to its efficiency. However, indoor environments such as long straight corridors can cause severe degeneracy problems in SLAM. In this paper, we use Proximal Policy Optimization (PPO) to train an adaptive degeneracy optimization agent (DOA) to address degeneracy problem. We propose a systematic methodology to address three critical challenges in traditional supervised learning frameworks: (1) data acquisition bottlenecks in degenerate dataset, (2) inherent quality deterioration of training samples, and (3) ambiguity in annotation protocol design. We design a specialized reward function to guide the agent in developing perception capabilities for degenerate environments. Using the output degeneracy factor as a reference weight, the agent can dynamically adjust the contribution of different sensors to pose optimization. Specifically, the observation distribution is shifted towards the motion model distribution, with the step size determined by a linear interpolation formula related to the degeneracy factor. In addition, we employ a transfer learning module to endow the agent with generalization capabilities across different environments and address the inefficiency of training in degenerate environments. Finally, we conduct ablation studies to demonstrate the rationality of our model design and the role of transfer learning. We also compare the proposed DOA with SOTA methods to prove its superior degeneracy detection and optimization capabilities across various environments.
>
---
#### [new 026] FMimic: Foundation Models are Fine-grained Action Learners from Human Videos
- **分类: cs.RO**

- **简介: 该论文属于视觉模仿学习任务，旨在解决机器人从人类视频中学习精细动作的问题。现有方法依赖预定义动作，效果受限。论文提出FMimic方法，利用基础模型直接从少量视频中学习可泛化的精细动作技能，显著提升了任务表现。**

- **链接: [http://arxiv.org/pdf/2507.20622v1](http://arxiv.org/pdf/2507.20622v1)**

> **作者:** Guangyan Chen; Meiling Wang; Te Cui; Yao Mu; Haoyang Lu; Zicai Peng; Mengxiao Hu; Tianxing Zhou; Mengyin Fu; Yi Yang; Yufeng Yue
>
> **备注:** accepted to International Journal of Robotics Research(IJRR)
>
> **摘要:** Visual imitation learning (VIL) provides an efficient and intuitive strategy for robotic systems to acquire novel skills. Recent advancements in foundation models, particularly Vision Language Models (VLMs), have demonstrated remarkable capabilities in visual and linguistic reasoning for VIL tasks. Despite this progress, existing approaches primarily utilize these models for learning high-level plans from human demonstrations, relying on pre-defined motion primitives for executing physical interactions, which remains a major bottleneck for robotic systems. In this work, we present FMimic, a novel paradigm that harnesses foundation models to directly learn generalizable skills at even fine-grained action levels, using only a limited number of human videos. Extensive experiments demonstrate that our FMimic delivers strong performance with a single human video, and significantly outperforms all other methods with five videos. Furthermore, our method exhibits significant improvements of over 39% and 29% in RLBench multi-task experiments and real-world manipulation tasks, respectively, and exceeds baselines by more than 34% in high-precision tasks and 47% in long-horizon tasks.
>
---
#### [new 027] SuperMag: Vision-based Tactile Data Guided High-resolution Tactile Shape Reconstruction for Magnetic Tactile Sensors
- **分类: cs.RO**

- **简介: 该论文属于 tactile shape reconstruction 任务，旨在解决磁性触觉传感器（MBTS）因感测单元稀疏导致的空间分辨率低问题。作者提出 SuperMag 方法，利用高分辨率视觉触觉传感器（VBTS）数据指导 MBTS 的超分辨率重建，通过同步采集和条件生成模型实现快速高精度形状恢复。**

- **链接: [http://arxiv.org/pdf/2507.20002v1](http://arxiv.org/pdf/2507.20002v1)**

> **作者:** Peiyao Hou; Danning Sun; Meng Wang; Yuzhe Huang; Zeyu Zhang; Hangxin Liu; Wanlin Li; Ziyuan Jiao
>
> **备注:** 7 pages, 7 figures; accepted by IROS 2025
>
> **摘要:** Magnetic-based tactile sensors (MBTS) combine the advantages of compact design and high-frequency operation but suffer from limited spatial resolution due to their sparse taxel arrays. This paper proposes SuperMag, a tactile shape reconstruction method that addresses this limitation by leveraging high-resolution vision-based tactile sensor (VBTS) data to supervise MBTS super-resolution. Co-designed, open-source VBTS and MBTS with identical contact modules enable synchronized data collection of high-resolution shapes and magnetic signals via a symmetric calibration setup. We frame tactile shape reconstruction as a conditional generative problem, employing a conditional variational auto-encoder to infer high-resolution shapes from low-resolution MBTS inputs. The MBTS achieves a sampling frequency of 125 Hz, whereas the shape reconstruction sustains an inference time within 2.5 ms. This cross-modality synergy advances tactile perception of the MBTS, potentially unlocking its new capabilities in high-precision robotic tasks.
>
---
#### [new 028] Methods for the Segmentation of Reticular Structures Using 3D LiDAR Data: A Comparative Evaluation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于点云分割任务，旨在解决攀爬机器人在网状结构中自主导航时的可导航面检测问题。论文提出了基于特征分解的分析算法和多种深度学习模型（如PointNet、PointTransformerV3）进行二值分割，并在3D点云数据上进行比较评估，结果显示PointTransformerV3分割精度最高，而分析算法在调参和效率上更具优势。**

- **链接: [http://arxiv.org/pdf/2507.20589v1](http://arxiv.org/pdf/2507.20589v1)**

> **作者:** Francisco J. Soler Mora; Adrián Peidró Vidal; Marc Fabregat-Jaén; Luis Payá Castelló; Óscar Reinoso García
>
> **摘要:** Reticular structures form the backbone of major infrastructure like bridges, pylons, and airports, but their inspection and maintenance are costly and hazardous, often requiring human intervention. While prior research has focused on fault detection via images or robotic platform design, the autonomous navigation of robots within these structures is less explored. This study addresses that gap by proposing methods to detect navigable surfaces in truss structures, enhancing the autonomy of climbing robots. The paper introduces several approaches for binary segmentation of navigable surfaces versus background from 3D point clouds of metallic trusses. These methods fall into two categories: analytical algorithms and deep learning models. The analytical approach features a custom algorithm that segments structures by analyzing the eigendecomposition of planar patches in the point cloud. In parallel, advanced deep learning models PointNet, PointNet++, MinkUNet34C, and PointTransformerV3 are trained and evaluated for the same task. Comparative analysis shows that the analytical algorithm offers easier parameter tuning and performance comparable to deep learning models, which, while more computationally intensive, excel in segmentation accuracy. Notably, PointTransformerV3 achieves a Mean Intersection Over Union (mIoU) of about 97%. The study demonstrates the promise of both analytical and deep learning methods for improving autonomous navigation in complex truss environments. The results highlight the trade-offs between computational efficiency and segmentation performance, providing valuable guidance for future research and practical applications in autonomous infrastructure inspection and maintenance.
>
---
#### [new 029] A real-time full-chain wearable sensor-based musculoskeletal simulation: an OpenSim-ROS Integration
- **分类: cs.RO**

- **简介: 该论文旨在解决传统肌肉骨骼建模与仿真系统受限于高成本、实验室环境及复杂软件集成的问题。任务是实现一个基于可穿戴传感器的实时完整链路肌肉骨骼仿真框架。作者通过集成OpenSimRT、ROS和可穿戴传感器，验证了其在步态分析和肌肉激活估计中的有效性，为康复、机器人及外骨骼设计提供新工具。**

- **链接: [http://arxiv.org/pdf/2507.20049v1](http://arxiv.org/pdf/2507.20049v1)**

> **作者:** Frederico Belmonte Klein; Zhaoyuan Wan; Huawei Wang; Ruoli Wang
>
> **备注:** 11 pages, 10 figures
>
> **摘要:** Musculoskeletal modeling and simulations enable the accurate description and analysis of the movement of biological systems with applications such as rehabilitation assessment, prosthesis, and exoskeleton design. However, the widespread usage of these techniques is limited by costly sensors, laboratory-based setups, computationally demanding processes, and the use of diverse software tools that often lack seamless integration. In this work, we address these limitations by proposing an integrated, real-time framework for musculoskeletal modeling and simulations that leverages OpenSimRT, the robotics operating system (ROS), and wearable sensors. As a proof-of-concept, we demonstrate that this framework can reasonably well describe inverse kinematics of both lower and upper body using either inertial measurement units or fiducial markers. Additionally, we show that it can effectively estimate inverse dynamics of the ankle joint and muscle activations of major lower limb muscles during daily activities, including walking, squatting and sit to stand, stand to sit when combined with pressure insoles. We believe this work lays the groundwork for further studies with more complex real-time and wearable sensor-based human movement analysis systems and holds potential to advance technologies in rehabilitation, robotics and exoskeleton designs.
>
---
#### [new 030] RAKOMO: Reachability-Aware K-Order Markov Path Optimization for Quadrupedal Loco-Manipulation
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出RAKOMO方法，用于四足机器人操作任务的路径优化。该论文属于运动规划任务，旨在解决四足机器人在执行操作任务时因运动学约束和接触切换导致的规划困难问题。论文通过结合K-Order Markov Optimization与基于可达区域的优化准则，并利用神经网络预测可达性，实现了更高效和安全的轨迹优化。**

- **链接: [http://arxiv.org/pdf/2507.19652v1](http://arxiv.org/pdf/2507.19652v1)**

> **作者:** Mattia Risiglione; Abdelrahman Abdalla; Victor Barasuol; Kim Tien Ly; Ioannis Havoutis; Claudio Semini
>
> **摘要:** Legged manipulators, such as quadrupeds equipped with robotic arms, require motion planning techniques that account for their complex kinematic constraints in order to perform manipulation tasks both safely and effectively. However, trajectory optimization methods often face challenges due to the hybrid dynamics introduced by contact discontinuities, and tend to neglect leg limitations during planning for computational reasons. In this work, we propose RAKOMO, a path optimization technique that integrates the strengths of K-Order Markov Optimization (KOMO) with a kinematically-aware criterion based on the reachable region defined as reachability margin. We leverage a neural-network to predict the margin and optimize it by incorporating it in the standard KOMO formulation. This approach enables rapid convergence of gradient-based motion planning -- commonly tailored for continuous systems -- while adapting it effectively to legged manipulators, successfully executing loco-manipulation tasks. We benchmark RAKOMO against a baseline KOMO approach through a set of simulations for pick-and-place tasks with the HyQReal quadruped robot equipped with a Kinova Gen3 robotic arm.
>
---
#### [new 031] Spatial Language Likelihood Grounding Network for Bayesian Fusion of Human-Robot Observations
- **分类: cs.RO; cs.CL; cs.IT; cs.LG; cs.SY; eess.SY; math.IT**

- **简介: 该论文属于人机协作任务中的信息融合研究，旨在解决机器人如何有效融合人类观察与传感器数据的问题。论文提出了一种基于特征金字塔的空间语言似然网络（FP-LGN），通过学习地图图像特征与空间语言的关系，建立人类输入的不确定性模型。实验表明该方法在不确定性感知融合中表现优异，提升了协作任务性能。**

- **链接: [http://arxiv.org/pdf/2507.19947v1](http://arxiv.org/pdf/2507.19947v1)**

> **作者:** Supawich Sitdhipol; Waritwong Sukprasongdee; Ekapol Chuangsuwanich; Rina Tse
>
> **备注:** Accepted to the 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC)
>
> **摘要:** Fusing information from human observations can help robots overcome sensing limitations in collaborative tasks. However, an uncertainty-aware fusion framework requires a grounded likelihood representing the uncertainty of human inputs. This paper presents a Feature Pyramid Likelihood Grounding Network (FP-LGN) that grounds spatial language by learning relevant map image features and their relationships with spatial relation semantics. The model is trained as a probability estimator to capture aleatoric uncertainty in human language using three-stage curriculum learning. Results showed that FP-LGN matched expert-designed rules in mean Negative Log-Likelihood (NLL) and demonstrated greater robustness with lower standard deviation. Collaborative sensing results demonstrated that the grounded likelihood successfully enabled uncertainty-aware fusion of heterogeneous human language observations and robot sensor measurements, achieving significant improvements in human-robot collaborative task performance.
>
---
#### [new 032] CLASP: General-Purpose Clothes Manipulation with Semantic Keypoints
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人衣物操作任务，旨在解决不同衣物类型和任务的通用操作问题。提出CLASP方法，利用语义关键点（如“左袖子”、“右肩”）作为感知与动作的中间表示，结合视觉语言模型与操作技能库，实现高效任务规划与执行。实验验证了其在多种衣物操作任务中的有效性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.19983v1](http://arxiv.org/pdf/2507.19983v1)**

> **作者:** Yuhong Deng; Chao Tang; Cunjun Yu; Linfeng Li; David Hsu
>
> **摘要:** Clothes manipulation, such as folding or hanging, is a critical capability for home service robots. Despite recent advances, most existing methods remain limited to specific tasks and clothes types, due to the complex, high-dimensional geometry of clothes. This paper presents CLothes mAnipulation with Semantic keyPoints (CLASP), which aims at general-purpose clothes manipulation over different clothes types, T-shirts, shorts, skirts, long dresses, ... , as well as different tasks, folding, flattening, hanging, ... . The core idea of CLASP is semantic keypoints -- e.g., ''left sleeve'', ''right shoulder'', etc. -- a sparse spatial-semantic representation that is salient for both perception and action. Semantic keypoints of clothes can be reliably extracted from RGB-D images and provide an effective intermediate representation of clothes manipulation policies. CLASP uses semantic keypoints to bridge high-level task planning and low-level action execution. At the high level, it exploits vision language models (VLMs) to predict task plans over the semantic keypoints. At the low level, it executes the plans with the help of a simple pre-built manipulation skill library. Extensive simulation experiments show that CLASP outperforms state-of-the-art baseline methods on multiple tasks across diverse clothes types, demonstrating strong performance and generalization. Further experiments with a Franka dual-arm system on four distinct tasks -- folding, flattening, hanging, and placing -- confirm CLASP's performance on a real robot.
>
---
#### [new 033] Bridging Simulation and Usability: A User-Friendly Framework for Scenario Generation in CARLA
- **分类: cs.RO**

- **简介: 论文任务是自动驾驶仿真测试中的场景生成。论文旨在解决现有场景生成工具需编程知识、使用门槛高的问题。工作提出了一种无需编程的交互式框架，通过图形界面支持用户便捷创建、修改和执行仿真场景，降低使用难度，提升仿真验证的可及性与效率。**

- **链接: [http://arxiv.org/pdf/2507.19883v1](http://arxiv.org/pdf/2507.19883v1)**

> **作者:** Ahmed Abouelazm; Mohammad Mahmoud; Conrad Walter; Oleksandr Shchetsura; Erne Hussong; Helen Gremmelmaier; J. Marius Zöllner
>
> **备注:** Paper is accepted in IEEE International Automated Vehicle Validation Conference (IAVVC 2025)
>
> **摘要:** Autonomous driving promises safer roads, reduced congestion, and improved mobility, yet validating these systems across diverse conditions remains a major challenge. Real-world testing is expensive, time-consuming, and sometimes unsafe, making large-scale validation impractical. In contrast, simulation environments offer a scalable and cost-effective alternative for rigorous verification and validation. A critical component of the validation process is scenario generation, which involves designing and configuring traffic scenarios to evaluate autonomous systems' responses to various events and uncertainties. However, existing scenario generation tools often require programming knowledge, limiting accessibility for non-technical users. To address this limitation, we present an interactive, no-code framework for scenario generation. Our framework features a graphical interface that enables users to create, modify, save, load, and execute scenarios without needing coding expertise or detailed simulation knowledge. Unlike script-based tools such as Scenic or ScenarioRunner, our approach lowers the barrier to entry and supports a broader user base. Central to our framework is a graph-based scenario representation that facilitates structured management, supports both manual and automated generation, and enables integration with deep learning-based scenario and behavior generation methods. In automated mode, the framework can randomly sample parameters such as actor types, behaviors, and environmental conditions, allowing the generation of diverse and realistic test datasets. By simplifying the scenario generation process, this framework supports more efficient testing workflows and increases the accessibility of simulation-based validation for researchers, engineers, and policymakers.
>
---
#### [new 034] Reward-Augmented Reinforcement Learning for Continuous Control in Precision Autonomous Parking via Policy Optimization Methods
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于自动驾驶任务中的自主泊车问题，旨在解决传统方法在复杂环境中的适应性和泛化能力不足的问题。作者提出了RARLAP框架，通过设计三种奖励机制并结合策略优化方法，在高保真仿真环境中训练和评估模型。实验表明，所提出的Milestone-augmented Reward方法在性能和稳定性上优于其他方法。**

- **链接: [http://arxiv.org/pdf/2507.19642v1](http://arxiv.org/pdf/2507.19642v1)**

> **作者:** Ahmad Suleman; Misha Urooj Khan; Zeeshan Kaleem; Ali H. Alenezi; Iqra Shabbir Sinem Coleri; Chau Yuen
>
> **摘要:** Autonomous parking (AP) represents a critical yet complex subset of intelligent vehicle automation, characterized by tight spatial constraints, frequent close-range obstacle interactions, and stringent safety margins. However, conventional rule-based and model-predictive methods often lack the adaptability and generalization needed to handle the nonlinear and environment-dependent complexities of AP. To address these limitations, we propose a reward-augmented learning framework for AP (RARLAP), that mitigates the inherent complexities of continuous-domain control by leveraging structured reward design to induce smooth and adaptable policy behavior, trained entirely within a high-fidelity Unity-based custom 3D simulation environment. We systematically design and assess three structured reward strategies: goal-only reward (GOR), dense proximity reward (DPR), and milestone-augmented reward (MAR), each integrated with both on-policy and off-policy optimization paradigms. Empirical evaluations demonstrate that the on-policy MAR achieves a 91\% success rate, yielding smoother trajectories and more robust behavior, while GOR and DPR fail to guide effective learning. Convergence and trajectory analyses demonstrate that the proposed framework enhances policy adaptability, accelerates training, and improves safety in continuous control. Overall, RARLAP establishes that reward augmentation effectively addresses complex autonomous parking challenges, enabling scalable and efficient policy optimization with both on- and off-policy methods. To support reproducibility, the code accompanying this paper is publicly available.
>
---
#### [new 035] PlaneHEC: Efficient Hand-Eye Calibration for Multi-view Robotic Arm via Any Point Cloud Plane Detection
- **分类: cs.RO**

- **简介: 该论文属于视觉引导机器人系统的手眼标定任务，旨在解决多视角机械臂手眼坐标系转换矩阵的高效准确标定问题。现有方法依赖复杂模型或人工辅助，效率低且泛化性差。论文提出PlaneHEC方法，利用任意平面点云进行标定，无需复杂模型，通过平面约束建立可解释性强的方程，并结合闭式解与迭代优化提升精度，实现了快速、通用的标定效果。**

- **链接: [http://arxiv.org/pdf/2507.19851v1](http://arxiv.org/pdf/2507.19851v1)**

> **作者:** Ye Wang; Haodong Jing; Yang Liao; Yongqiang Ma; Nanning Zheng
>
> **备注:** Accepted by 2025 IEEE International Conference on Robotics & Automation (ICRA)
>
> **摘要:** Hand-eye calibration is an important task in vision-guided robotic systems and is crucial for determining the transformation matrix between the camera coordinate system and the robot end-effector. Existing methods, for multi-view robotic systems, usually rely on accurate geometric models or manual assistance, generalize poorly, and can be very complicated and inefficient. Therefore, in this study, we propose PlaneHEC, a generalized hand-eye calibration method that does not require complex models and can be accomplished using only depth cameras, which achieves the optimal and fastest calibration results using arbitrary planar surfaces like walls and tables. PlaneHEC introduces hand-eye calibration equations based on planar constraints, which makes it strongly interpretable and generalizable. PlaneHEC also uses a comprehensive solution that starts with a closed-form solution and improves it withiterative optimization, which greatly improves accuracy. We comprehensively evaluated the performance of PlaneHEC in both simulated and real-world environments and compared the results with other point-cloud-based calibration methods, proving its superiority. Our approach achieves universal and fast calibration with an innovative design of computational models, providing a strong contribution to the development of multi-agent systems and embodied intelligence.
>
---
#### [new 036] A Human-in-the-loop Approach to Robot Action Replanning through LLM Common-Sense Reasoning
- **分类: cs.RO**

- **简介: 该论文属于机器人任务规划与人机交互领域，旨在解决仅依赖视觉输入进行机器人动作规划时的不可靠性与扩展性差问题。通过结合人类自然语言输入与大语言模型（LLM）的常识推理能力，提出一种人在环中的方法，自动修正基于单个RGB视频生成的执行计划，以预防潜在失败并适应新指令，提升机器人任务执行的鲁棒性与适应性。**

- **链接: [http://arxiv.org/pdf/2507.20870v1](http://arxiv.org/pdf/2507.20870v1)**

> **作者:** Elena Merlo; Marta Lagomarsino; Arash Ajoudani
>
> **摘要:** To facilitate the wider adoption of robotics, accessible programming tools are required for non-experts. Observational learning enables intuitive human skills transfer through hands-on demonstrations, but relying solely on visual input can be inefficient in terms of scalability and failure mitigation, especially when based on a single demonstration. This paper presents a human-in-the-loop method for enhancing the robot execution plan, automatically generated based on a single RGB video, with natural language input to a Large Language Model (LLM). By including user-specified goals or critical task aspects and exploiting the LLM common-sense reasoning, the system adjusts the vision-based plan to prevent potential failures and adapts it based on the received instructions. Experiments demonstrated the framework intuitiveness and effectiveness in correcting vision-derived errors and adapting plans without requiring additional demonstrations. Moreover, interactive plan refinement and hallucination corrections promoted system robustness.
>
---
#### [new 037] A roadmap for AI in robotics
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文探讨人工智能在机器人领域的应用，任务是分析AI技术如何突破机器人实际应用中的障碍。论文总结了自1990年代以来的成果，提出短期与中期研究路线图，涵盖数据集构建、算法设计、人机协作、可解释性等挑战，并展望长期目标，如持续学习、安全性与计算效率。**

- **链接: [http://arxiv.org/pdf/2507.19975v1](http://arxiv.org/pdf/2507.19975v1)**

> **作者:** Aude Billard; Alin Albu-Schaeffer; Michael Beetz; Wolfram Burgard; Peter Corke; Matei Ciocarlie; Ravinder Dahiya; Danica Kragic; Ken Goldberg; Yukie Nagai; Davide Scaramuzza
>
> **摘要:** AI technologies, including deep learning, large-language models have gone from one breakthrough to the other. As a result, we are witnessing growing excitement in robotics at the prospect of leveraging the potential of AI to tackle some of the outstanding barriers to the full deployment of robots in our daily lives. However, action and sensing in the physical world pose greater and different challenges than analysing data in isolation. As the development and application of AI in robotic products advances, it is important to reflect on which technologies, among the vast array of network architectures and learning models now available in the AI field, are most likely to be successfully applied to robots; how they can be adapted to specific robot designs, tasks, environments; which challenges must be overcome. This article offers an assessment of what AI for robotics has achieved since the 1990s and proposes a short- and medium-term research roadmap listing challenges and promises. These range from keeping up-to-date large datasets, representatives of a diversity of tasks robots may have to perform, and of environments they may encounter, to designing AI algorithms tailored specifically to robotics problems but generic enough to apply to a wide range of applications and transfer easily to a variety of robotic platforms. For robots to collaborate effectively with humans, they must predict human behavior without relying on bias-based profiling. Explainability and transparency in AI-driven robot control are not optional but essential for building trust, preventing misuse, and attributing responsibility in accidents. We close on what we view as the primary long-term challenges, that is, to design robots capable of lifelong learning, while guaranteeing safe deployment and usage, and sustainable computational costs.
>
---
#### [new 038] PixelNav: Towards Model-based Vision-Only Navigation with Topological Graphs
- **分类: cs.RO**

- **简介: 论文提出PixelNav，一种基于视觉的移动机器人导航方法，属于机器人视觉导航任务。旨在解决纯数据驱动模型依赖大量训练数据且可解释性差的问题。工作结合深度学习与传统模型预测控制、可通行性估计、视觉定位等，利用拓扑图提升可解释性与实用性。**

- **链接: [http://arxiv.org/pdf/2507.20892v1](http://arxiv.org/pdf/2507.20892v1)**

> **作者:** Sergey Bakulin; Timur Akhtyamov; Denis Fatykhov; German Devchich; Gonzalo Ferrer
>
> **摘要:** This work proposes a novel hybrid approach for vision-only navigation of mobile robots, which combines advances of both deep learning approaches and classical model-based planning algorithms. Today, purely data-driven end-to-end models are dominant solutions to this problem. Despite advantages such as flexibility and adaptability, the requirement of a large amount of training data and limited interpretability are the main bottlenecks for their practical applications. To address these limitations, we propose a hierarchical system that utilizes recent advances in model predictive control, traversability estimation, visual place recognition, and pose estimation, employing topological graphs as a representation of the target environment. Using such a combination, we provide a scalable system with a higher level of interpretability compared to end-to-end approaches. Extensive real-world experiments show the efficiency of the proposed method.
>
---
#### [new 039] Robot Excavation and Manipulation of Geometrically Cohesive Granular Media
- **分类: cs.RO**

- **简介: 该论文研究机器人如何利用几何黏性颗粒材料进行无预先设计的随机结构建造。任务是探索机器人如何自主操控软体颗粒材料进行施工。工作包括开发机器人物理模型、测试不同材料状态下的性能，并分析材料特性对挖掘和建造的影响。**

- **链接: [http://arxiv.org/pdf/2507.19999v1](http://arxiv.org/pdf/2507.19999v1)**

> **作者:** Laura Treers; Daniel Soto; Joonha Hwang; Michael A. D. Goodisman; Daniel I. Goldman
>
> **摘要:** Construction throughout history typically assumes that its blueprints and building blocks are pre-determined. However, recent work suggests that alternative approaches can enable new paradigms for structure formation. Aleatory architectures, or those which rely on the properties of their granular building blocks rather than pre-planned design or computation, have thus far relied on human intervention for their creation. We imagine that robotic swarms could be valuable to create such aleatory structures by manipulating and forming structures from entangled granular materials. To discover principles by which robotic systems can effectively manipulate soft matter, we develop a robophysical model for interaction with geometrically cohesive granular media composed of u-shape particles. This robotic platform uses environmental signals to autonomously coordinate excavation, transport, and deposition of material. We test the effect of substrate initial conditions by characterizing robot performance in two different material compaction states and observe as much as a 75% change in transported mass depending on initial substrate compressive loading. These discrepancies suggest the functional role that material properties such as packing and cohesion/entanglement play in excavation and construction. To better understand these material properties, we develop an apparatus for tensile testing of the geometrically cohesive substrates, which reveals how entangled material strength responds strongly to initial compressive loading. These results explain the variation observed in robotic performance and point to future directions for better understanding robotic interaction mechanics with entangled materials.
>
---
#### [new 040] Decentralized Uncertainty-Aware Multi-Agent Collision Avoidance With Model Predictive Path Integral
- **分类: cs.RO**

- **简介: 论文研究多智能体在不确定环境下的分布式避障导航任务，解决传感器和动作噪声影响下的避障与路径规划问题。作者提出一种结合模型预测路径积分（MPPI）与概率避障策略的新方法，并通过锥规划融入安全约束，实现高效安全的多机器人避障。**

- **链接: [http://arxiv.org/pdf/2507.20293v1](http://arxiv.org/pdf/2507.20293v1)**

> **作者:** Stepan Dergachev; Konstantin Yakovlev
>
> **备注:** This is a pre-print of the paper accepted to IROS2025. It contains 8 pages, 4 figures and 1 table. The supplementary video available at https://youtu.be/_D4zDYJ4KCk
>
> **摘要:** Decentralized multi-agent navigation under uncertainty is a complex task that arises in numerous robotic applications. It requires collision avoidance strategies that account for both kinematic constraints, sensing and action execution noise. In this paper, we propose a novel approach that integrates the Model Predictive Path Integral (MPPI) with a probabilistic adaptation of Optimal Reciprocal Collision Avoidance. Our method ensures safe and efficient multi-agent navigation by incorporating probabilistic safety constraints directly into the MPPI sampling process via a Second-Order Cone Programming formulation. This approach enables agents to operate independently using local noisy observations while maintaining safety guarantees. We validate our algorithm through extensive simulations with differential-drive robots and benchmark it against state-of-the-art methods, including ORCA-DD and B-UAVC. Results demonstrate that our approach outperforms them while achieving high success rates, even in densely populated environments. Additionally, validation in the Gazebo simulator confirms its practical applicability to robotic platforms.
>
---
#### [new 041] Uni-Mapper: Unified Mapping Framework for Multi-modal LiDARs in Complex and Dynamic Environments
- **分类: cs.RO**

- **简介: 论文提出Uni-Mapper，用于多模态LiDAR在复杂动态环境中的统一地图构建框架。任务是解决跨传感器模态与动态环境的地图融合问题。工作包括动态物体去除、动态感知回环检测与多模态地图融合，提升多地图对齐与全局一致性。**

- **链接: [http://arxiv.org/pdf/2507.20538v1](http://arxiv.org/pdf/2507.20538v1)**

> **作者:** Gilhwan Kang; Hogyun Kim; Byunghee Choi; Seokhwan Jeong; Young-Sik Shin; Younggun Cho
>
> **备注:** 18 pages, 14 figures
>
> **摘要:** The unification of disparate maps is crucial for enabling scalable robot operation across multiple sessions and collaborative multi-robot scenarios. However, achieving a unified map robust to sensor modalities and dynamic environments remains a challenging problem. Variations in LiDAR types and dynamic elements lead to differences in point cloud distribution and scene consistency, hindering reliable descriptor generation and loop closure detection essential for accurate map alignment. To address these challenges, this paper presents Uni-Mapper, a dynamic-aware 3D point cloud map merging framework for multi-modal LiDAR systems. It comprises dynamic object removal, dynamic-aware loop closure, and multi-modal LiDAR map merging modules. A voxel-wise free space hash map is built in a coarse-to-fine manner to identify and reject dynamic objects via temporal occupancy inconsistencies. The removal module is integrated with a LiDAR global descriptor, which encodes preserved static local features to ensure robust place recognition in dynamic environments. In the final stage, multiple pose graph optimizations are conducted for both intra-session and inter-map loop closures. We adopt a centralized anchor-node strategy to mitigate intra-session drift errors during map merging. In the final stage, centralized anchor-node-based pose graph optimization is performed to address intra- and inter-map loop closures for globally consistent map merging. Our framework is evaluated on diverse real-world datasets with dynamic objects and heterogeneous LiDARs, showing superior performance in loop detection across sensor modalities, robust mapping in dynamic environments, and accurate multi-map alignment over existing methods. Project Page: https://sparolab.github.io/research/uni_mapper.
>
---
#### [new 042] Extending Group Relative Policy Optimization to Continuous Control: A Theoretical Framework for Robotic Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于强化学习任务，旨在解决连续控制环境中策略优化依赖值函数的问题。作者扩展了GRPO算法，引入轨迹聚类、状态感知优势估计和正则化更新，以适应机器人高维动作、稀疏奖励等挑战，并提供理论分析，为后续实验打下基础。**

- **链接: [http://arxiv.org/pdf/2507.19555v1](http://arxiv.org/pdf/2507.19555v1)**

> **作者:** Rajat Khanda; Mohammad Baqar; Sambuddha Chakrabarti; Satyasaran Changdar
>
> **备注:** 13 pages, 2 figures
>
> **摘要:** Group Relative Policy Optimization (GRPO) has shown promise in discrete action spaces by eliminating value function dependencies through group-based advantage estimation. However, its application to continuous control remains unexplored, limiting its utility in robotics where continuous actions are essential. This paper presents a theoretical framework extending GRPO to continuous control environments, addressing challenges in high-dimensional action spaces, sparse rewards, and temporal dynamics. Our approach introduces trajectory-based policy clustering, state-aware advantage estimation, and regularized policy updates designed for robotic applications. We provide theoretical analysis of convergence properties and computational complexity, establishing a foundation for future empirical validation in robotic systems including locomotion and manipulation tasks.
>
---
#### [new 043] Hypo-paradoxical Linkages: Linkages That Should Move-But Don't
- **分类: physics.soc-ph; cs.RO; 70B15**

- **简介: 论文研究了一类看似可动但实际刚性的连杆机构，称为“低悖连杆”（hypo-paradoxical linkages），与传统悖论连杆（如Bennet机构）相反。该研究旨在揭示这些连杆的异常行为，分析其运动性，并重新探讨Chebyshev-Grubler-Kutzbach准则在判定连杆自由度时的局限性。**

- **链接: [http://arxiv.org/pdf/2507.20371v1](http://arxiv.org/pdf/2507.20371v1)**

> **作者:** Nir Shvalb; Oded Medina
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** While paradoxical linkages famously violate the Chebyshev-Grubler-Kutzbach criterion by exhibiting unexpected mobility, we identify an opposing phenomenon: a class of linkages that appear mobile according to the same criterion, yet are in fact rigid. We refer to these as hypo-paradoxical linkages, and proceed to analyze and illustrate their behavior. We use the same tools to further explain the unexpected positive mobility of Bennet mechanism.
>
---
#### [new 044] Optimizing Spreading Factor Selection for Mobile LoRa Gateways Using Single-Channel Hardware
- **分类: cs.NI; cs.RO**

- **简介: 该论文旨在解决低成本单信道LoRa网关在移动部署中的可靠通信问题。通过静态选择最优扩频因子（SF），提出两阶段算法，结合规则排除与加权评分模型，优化时间、能耗、速率和链路鲁棒性。验证表明方法有效，适用于农业等成本敏感场景。**

- **链接: [http://arxiv.org/pdf/2507.19938v1](http://arxiv.org/pdf/2507.19938v1)**

> **作者:** W. A. Sasindu Wijesuriya
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** The deployment of mobile LoRa gateways using low-cost single-channel hardware presents a significant challenge in maintaining reliable communication due to the lack of dynamic configuration support. In traditional LoRaWAN networks, Adaptive Data Rate (ADR) mechanisms optimize communication parameters in real time. However, such features are typically supported only by expensive multi-channel gateways. This study proposes a cost-effective and energy-efficient solution by statically selecting the optimal Spreading Factor (SF) using a two-phase algorithm. The method first applies rule-based exclusion to eliminate SFs that violate constraints related to distance, data rate, link margin, and regulatory limits. Remaining candidates are then evaluated using a weighted scoring model incorporating Time-on-Air, energy consumption, data rate, and link robustness. The proposed algorithm was validated through extensive field tests and NS-3 simulations under line-of-sight conditions. Results demonstrate that the selected SF matched the optimal SF in over 92% of cases across 672 simulated scenarios, confirming the algorithm's effectiveness. This approach offers a scalable alternative to dynamic protocols, enabling reliable mobile LoRa deployments in cost-sensitive environments such as agriculture and rural sensing applications.
>
---
#### [new 045] Flow Matching Policy Gradients
- **分类: cs.LG; cs.RO**

- **简介: 论文提出Flow Policy Optimization（FPO），将流匹配引入策略梯度框架，用于连续控制任务。旨在解决基于扩散模型的强化学习依赖特定采样方法的问题，实现兼容PPO-clip的训练，无需精确似然计算，并在多种任务中从零训练扩散风格策略。**

- **链接: [http://arxiv.org/pdf/2507.21053v1](http://arxiv.org/pdf/2507.21053v1)**

> **作者:** David McAllister; Songwei Ge; Brent Yi; Chung Min Kim; Ethan Weber; Hongsuk Choi; Haiwen Feng; Angjoo Kanazawa
>
> **备注:** See our blog post: https://flowreinforce.github.io
>
> **摘要:** Flow-based generative models, including diffusion models, excel at modeling continuous distributions in high-dimensional spaces. In this work, we introduce Flow Policy Optimization (FPO), a simple on-policy reinforcement learning algorithm that brings flow matching into the policy gradient framework. FPO casts policy optimization as maximizing an advantage-weighted ratio computed from the conditional flow matching loss, in a manner compatible with the popular PPO-clip framework. It sidesteps the need for exact likelihood computation while preserving the generative capabilities of flow-based models. Unlike prior approaches for diffusion-based reinforcement learning that bind training to a specific sampling method, FPO is agnostic to the choice of diffusion or flow integration at both training and inference time. We show that FPO can train diffusion-style policies from scratch in a variety of continuous control tasks. We find that flow-based models can capture multimodal action distributions and achieve higher performance than Gaussian policies, particularly in under-conditioned settings.
>
---
#### [new 046] VLMPlanner: Integrating Visual Language Models with Motion Planning
- **分类: cs.AI; cs.RO**

- **简介: 论文提出VLMPlanner，将视觉语言模型与运动规划结合，解决自动驾驶中复杂场景决策问题。利用视觉语言模型处理多视角图像，捕捉细粒度道路信息，指导实时规划器生成安全轨迹，并设计CAI-Gate机制动态调整推理频率，平衡性能与效率。**

- **链接: [http://arxiv.org/pdf/2507.20342v1](http://arxiv.org/pdf/2507.20342v1)**

> **作者:** Zhipeng Tang; Sha Zhang; Jiajun Deng; Chenjie Wang; Guoliang You; Yuting Huang; Xinrui Lin; Yanyong Zhang
>
> **备注:** 8 pages, 3 figures, this paper has been accepted by ACM MM 2025
>
> **摘要:** Integrating large language models (LLMs) into autonomous driving motion planning has recently emerged as a promising direction, offering enhanced interpretability, better controllability, and improved generalization in rare and long-tail scenarios. However, existing methods often rely on abstracted perception or map-based inputs, missing crucial visual context, such as fine-grained road cues, accident aftermath, or unexpected obstacles, which are essential for robust decision-making in complex driving environments. To bridge this gap, we propose VLMPlanner, a hybrid framework that combines a learning-based real-time planner with a vision-language model (VLM) capable of reasoning over raw images. The VLM processes multi-view images to capture rich, detailed visual information and leverages its common-sense reasoning capabilities to guide the real-time planner in generating robust and safe trajectories. Furthermore, we develop the Context-Adaptive Inference Gate (CAI-Gate) mechanism that enables the VLM to mimic human driving behavior by dynamically adjusting its inference frequency based on scene complexity, thereby achieving an optimal balance between planning performance and computational efficiency. We evaluate our approach on the large-scale, challenging nuPlan benchmark, with comprehensive experimental results demonstrating superior planning performance in scenarios with intricate road conditions and dynamic elements. Code will be available.
>
---
#### [new 047] Efficient Self-Supervised Neuro-Analytic Visual Servoing for Real-time Quadrotor Control
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于无人机视觉控制任务，旨在解决传统视觉伺服控制在计算效率和数值稳定性方面的不足。论文提出了一种自监督的神经解析控制方法，通过知识蒸馏将解析模型的能力转移到轻量级神经网络，实现高效实时的四旋翼无人机控制，无需显式几何模型或标记。**

- **链接: [http://arxiv.org/pdf/2507.19878v1](http://arxiv.org/pdf/2507.19878v1)**

> **作者:** Sebastian Mocanu; Sebastian-Ion Nae; Mihai-Eugen Barbu; Marius Leordeanu
>
> **备注:** Accepted at the International Conference on Computer Vision Workshops 2025
>
> **摘要:** This work introduces a self-supervised neuro-analytical, cost efficient, model for visual-based quadrotor control in which a small 1.7M parameters student ConvNet learns automatically from an analytical teacher, an improved image-based visual servoing (IBVS) controller. Our IBVS system solves numerical instabilities by reducing the classical visual servoing equations and enabling efficient stable image feature detection. Through knowledge distillation, the student model achieves 11x faster inference compared to the teacher IBVS pipeline, while demonstrating similar control accuracy at a significantly lower computational and memory cost. Our vision-only self-supervised neuro-analytic control, enables quadrotor orientation and movement without requiring explicit geometric models or fiducial markers. The proposed methodology leverages simulation-to-reality transfer learning and is validated on a small drone platform in GPS-denied indoor environments. Our key contributions include: (1) an analytical IBVS teacher that solves numerical instabilities inherent in classical approaches, (2) a two-stage segmentation pipeline combining YOLOv11 with a U-Net-based mask splitter for robust anterior-posterior vehicle segmentation to correctly estimate the orientation of the target, and (3) an efficient knowledge distillation dual-path system, which transfers geometric visual servoing capabilities from the analytical IBVS teacher to a compact and small student neural network that outperforms the teacher, while being suitable for real-time onboard deployment.
>
---
#### [new 048] Co-Win: Joint Object Detection and Instance Segmentation in LiDAR Point Clouds via Collaborative Window Processing
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的环境感知任务，旨在解决复杂城市环境中激光雷达点云的多目标检测与实例分割问题。论文提出了Co-Win框架，结合鸟瞰图感知、窗口特征提取与变分掩码分割，实现对场景的精细理解和实例识别，提升自动驾驶系统的决策能力。**

- **链接: [http://arxiv.org/pdf/2507.19691v1](http://arxiv.org/pdf/2507.19691v1)**

> **作者:** Haichuan Li; Tomi Westerlund
>
> **摘要:** Accurate perception and scene understanding in complex urban environments is a critical challenge for ensuring safe and efficient autonomous navigation. In this paper, we present Co-Win, a novel bird's eye view (BEV) perception framework that integrates point cloud encoding with efficient parallel window-based feature extraction to address the multi-modality inherent in environmental understanding. Our method employs a hierarchical architecture comprising a specialized encoder, a window-based backbone, and a query-based decoder head to effectively capture diverse spatial features and object relationships. Unlike prior approaches that treat perception as a simple regression task, our framework incorporates a variational approach with mask-based instance segmentation, enabling fine-grained scene decomposition and understanding. The Co-Win architecture processes point cloud data through progressive feature extraction stages, ensuring that predicted masks are both data-consistent and contextually relevant. Furthermore, our method produces interpretable and diverse instance predictions, enabling enhanced downstream decision-making and planning in autonomous driving systems.
>
---
#### [new 049] Partially Observable Monte-Carlo Graph Search
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于人工智能中的强化学习任务，旨在解决大规模部分可观测马尔可夫决策过程（POMDPs）的离线规划问题。现有在线方法无法满足时间或能量受限的应用需求，而传统离线方法又难以扩展到大型问题。为此，论文提出了一种新的采样算法——部分可观测蒙特卡洛图搜索（POMCGS），通过将搜索树折叠为策略图，显著减少计算量，并支持策略分析与验证。此外，结合动作渐进扩展和观测聚类方法，POMCGS还能处理某些连续POMDP问题。实验表明，该方法在最具挑战性的POMDP问题上生成的策略优于现有离线算法，且与最先进的在线算法相当。**

- **链接: [http://arxiv.org/pdf/2507.20951v1](http://arxiv.org/pdf/2507.20951v1)**

> **作者:** Yang You; Vincent Thomas; Alex Schutz; Robert Skilton; Nick Hawes; Olivier Buffet
>
> **备注:** To be published in Proceedings of ICAPS 2025
>
> **摘要:** Currently, large partially observable Markov decision processes (POMDPs) are often solved by sampling-based online methods which interleave planning and execution phases. However, a pre-computed offline policy is more desirable in POMDP applications with time or energy constraints. But previous offline algorithms are not able to scale up to large POMDPs. In this article, we propose a new sampling-based algorithm, the partially observable Monte-Carlo graph search (POMCGS) to solve large POMDPs offline. Different from many online POMDP methods, which progressively develop a tree while performing (Monte-Carlo) simulations, POMCGS folds this search tree on the fly to construct a policy graph, so that computations can be drastically reduced, and users can analyze and validate the policy prior to embedding and executing it. Moreover, POMCGS, together with action progressive widening and observation clustering methods provided in this article, is able to address certain continuous POMDPs. Through experiments, we demonstrate that POMCGS can generate policies on the most challenging POMDPs, which cannot be computed by previous offline algorithms, and these policies' values are competitive compared with the state-of-the-art online POMDP algorithms.
>
---
#### [new 050] ACCESS-AV: Adaptive Communication-Computation Codesign for Sustainable Autonomous Vehicle Localization in Smart Factories
- **分类: eess.SY; cs.AR; cs.NI; cs.RO; cs.SY; eess.SP**

- **简介: 论文提出ACCESS-AV框架，用于智能工厂中自动驾驶配送车辆的节能定位。该任务属于车辆定位优化。为解决能耗高与基础设施成本问题，利用5G同步信号块实现自适应通信计算协同设计，动态平衡能效与精度。实验表明其节能达43.09%，定位精度高，降低成本，适用于可持续智能工厂。**

- **链接: [http://arxiv.org/pdf/2507.20399v1](http://arxiv.org/pdf/2507.20399v1)**

> **作者:** Rajat Bhattacharjya; Arnab Sarkar; Ish Kool; Sabur Baidya; Nikil Dutt
>
> **备注:** 28 pages, 9 figures
>
> **摘要:** Autonomous Delivery Vehicles (ADVs) are increasingly used for transporting goods in 5G network-enabled smart factories, with the compute-intensive localization module presenting a significant opportunity for optimization. We propose ACCESS-AV, an energy-efficient Vehicle-to-Infrastructure (V2I) localization framework that leverages existing 5G infrastructure in smart factory environments. By opportunistically accessing the periodically broadcast 5G Synchronization Signal Blocks (SSBs) for localization, ACCESS-AV obviates the need for dedicated Roadside Units (RSUs) or additional onboard sensors to achieve energy efficiency as well as cost reduction. We implement an Angle-of-Arrival (AoA)-based estimation method using the Multiple Signal Classification (MUSIC) algorithm, optimized for resource-constrained ADV platforms through an adaptive communication-computation strategy that dynamically balances energy consumption with localization accuracy based on environmental conditions such as Signal-to-Noise Ratio (SNR) and vehicle velocity. Experimental results demonstrate that ACCESS-AV achieves an average energy reduction of 43.09% compared to non-adaptive systems employing AoA algorithms such as vanilla MUSIC, ESPRIT, and Root-MUSIC. It maintains sub-30 cm localization accuracy while also delivering substantial reductions in infrastructure and operational costs, establishing its viability for sustainable smart factory environments.
>
---
#### [new 051] Beyond Line-of-Sight: Cooperative Localization Using Vision and V2X Communication
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于自动驾驶定位任务，旨在解决城市复杂环境中GNSS信号不可靠时的车辆定位问题。论文提出了一种基于视觉和V2X通信的分布式定位算法，利用车载摄像头和车与环境通信，使车辆在遮挡严重场景下也能估计自身位置和方向。通过理论证明和实验验证了算法的有效性与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.20772v1](http://arxiv.org/pdf/2507.20772v1)**

> **作者:** Annika Wong; Zhiqi Tang; Frank J. Jiang; Karl H. Johansson; Jonas Mårtensson
>
> **备注:** Accepted at the 2025 IEEE 28th International Conference on Intelligent Transportation Systems (ITSC 2025)
>
> **摘要:** Accurate and robust localization is critical for the safe operation of Connected and Automated Vehicles (CAVs), especially in complex urban environments where Global Navigation Satellite System (GNSS) signals are unreliable. This paper presents a novel vision-based cooperative localization algorithm that leverages onboard cameras and Vehicle-to-Everything (V2X) communication to enable CAVs to estimate their poses, even in occlusion-heavy scenarios such as busy intersections. In particular, we propose a novel decentralized observer for a group of connected agents that includes landmark agents (static or moving) in the environment with known positions and vehicle agents that need to estimate their poses (both positions and orientations). Assuming that (i) there are at least three landmark agents in the environment, (ii) each vehicle agent can measure its own angular and translational velocities as well as relative bearings to at least three neighboring landmarks or vehicles, and (iii) neighboring vehicles can communicate their pose estimates, each vehicle can estimate its own pose using the proposed decentralized observer. We prove that the origin of the estimation error is locally exponentially stable under the proposed observer, provided that the minimal observability conditions are satisfied. Moreover, we evaluate the proposed approach through experiments with real 1/10th-scale connected vehicles and large-scale simulations, demonstrating its scalability and validating the theoretical guarantees in practical scenarios.
>
---
#### [new 052] AQUA: A Large Language Model for Aquaculture & Fisheries
- **分类: cs.CL; cs.AI; cs.CE; cs.LG; cs.RO**

- **简介: 该论文提出AQUA，首个专为水产养殖设计的大语言模型，旨在解决行业面临的疾病、效率、成本等复杂问题。通过AQUADAPT框架生成高质量合成数据，支持研究与决策，推动AI在水产养殖的应用。**

- **链接: [http://arxiv.org/pdf/2507.20520v1](http://arxiv.org/pdf/2507.20520v1)**

> **作者:** Praneeth Narisetty; Uday Kumar Reddy Kattamanchi; Lohit Akshant Nimma; Sri Ram Kaushik Karnati; Shiva Nagendra Babu Kore; Mounika Golamari; Tejashree Nageshreddy
>
> **摘要:** Aquaculture plays a vital role in global food security and coastal economies by providing sustainable protein sources. As the industry expands to meet rising demand, it faces growing challenges such as disease outbreaks, inefficient feeding practices, rising labor costs, logistical inefficiencies, and critical hatchery issues, including high mortality rates and poor water quality control. Although artificial intelligence has made significant progress, existing machine learning methods fall short of addressing the domain-specific complexities of aquaculture. To bridge this gap, we introduce AQUA, the first large language model (LLM) tailored for aquaculture, designed to support farmers, researchers, and industry practitioners. Central to this effort is AQUADAPT (Data Acquisition, Processing and Tuning), an Agentic Framework for generating and refining high-quality synthetic data using a combination of expert knowledge, largescale language models, and automated evaluation techniques. Our work lays the foundation for LLM-driven innovations in aquaculture research, advisory systems, and decision-making tools.
>
---
## 更新

#### [replaced 001] REGRACE: A Robust and Efficient Graph-based Re-localization Algorithm using Consistency Evaluation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.03599v2](http://arxiv.org/pdf/2503.03599v2)**

> **作者:** Débora N. P. Oliveira; Joshua Knights; Sebastián Barbas Laina; Simon Boche; Wolfram Burgard; Stefan Leutenegger
>
> **备注:** Accepted to IROS2025
>
> **摘要:** Loop closures are essential for correcting odometry drift and creating consistent maps, especially in the context of large-scale navigation. Current methods using dense point clouds for accurate place recognition do not scale well due to computationally expensive scan-to-scan comparisons. Alternative object-centric approaches are more efficient but often struggle with sensitivity to viewpoint variation. In this work, we introduce REGRACE, a novel approach that addresses these challenges of scalability and perspective difference in re-localization by using LiDAR-based submaps. We introduce rotation-invariant features for each labeled object and enhance them with neighborhood context through a graph neural network. To identify potential revisits, we employ a scalable bag-of-words approach, pooling one learned global feature per submap. Additionally, we define a revisit with geometrical consistency cues rather than embedding distance, allowing us to recognize far-away loop closures. Our evaluations demonstrate that REGRACE achieves similar results compared to state-of-the-art place recognition and registration baselines while being twice as fast. Code and models are publicly available.
>
---
#### [replaced 002] Robotic Visual Instruction
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.00693v3](http://arxiv.org/pdf/2505.00693v3)**

> **作者:** Yanbang Li; Ziyang Gong; Haoyang Li; Xiaoqi Huang; Haolan Kang; Guangping Bai; Xianzheng Ma
>
> **备注:** Project website: https://robotic-visual-instruction.github.io/
>
> **摘要:** Recently, natural language has been the primary medium for human-robot interaction. However, its inherent lack of spatial precision introduces challenges for robotic task definition such as ambiguity and verbosity. Moreover, in some public settings where quiet is required, such as libraries or hospitals, verbal communication with robots is inappropriate. To address these limitations, we introduce the Robotic Visual Instruction (RoVI), a novel paradigm to guide robotic tasks through an object-centric, hand-drawn symbolic representation. RoVI effectively encodes spatial-temporal information into human-interpretable visual instructions through 2D sketches, utilizing arrows, circles, colors, and numbers to direct 3D robotic manipulation. To enable robots to understand RoVI better and generate precise actions based on RoVI, we present Visual Instruction Embodied Workflow (VIEW), a pipeline formulated for RoVI-conditioned policies. This approach leverages Vision-Language Models (VLMs) to interpret RoVI inputs, decode spatial and temporal constraints from 2D pixel space via keypoint extraction, and then transform them into executable 3D action sequences. We additionally curate a specialized dataset of 15K instances to fine-tune small VLMs for edge deployment,enabling them to effectively learn RoVI capabilities. Our approach is rigorously validated across 11 novel tasks in both real and simulated environments, demonstrating significant generalization capability. Notably, VIEW achieves an 87.5% success rate in real-world scenarios involving unseen tasks that feature multi-step actions, with disturbances, and trajectory-following requirements. Project website: https://robotic-visual-instruction.github.io/
>
---
#### [replaced 003] Safe Expeditious Whole-Body Control of Mobile Manipulators for Collision Avoidance
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.14775v3](http://arxiv.org/pdf/2409.14775v3)**

> **作者:** Bingjie Chen; Yancong Wei; Rihao Liu; Chenxi Han; Houde Liu; Chongkun Xia; Liang Han; Bin Liang
>
> **摘要:** Whole-body reactive obstacle avoidance for mobile manipulators (MM) remains an open research problem. Control Barrier Functions (CBF), combined with Quadratic Programming (QP), have become a popular approach for reactive control with safety guarantees. However, traditional CBF methods often face issues such as pseudo-equilibrium problems (PEP) and are ineffective in handling dynamic obstacles. To overcome these challenges, we introduce the Adaptive Cyclic Inequality (ACI) method. ACI takes into account both the obstacle's velocity and the robot's nominal control to define a directional safety constraint. When added to the CBF-QP, ACI helps avoid PEP and enables reliable collision avoidance in dynamic environments. We validate our approach on a MM that includes a low-dimensional mobile base and a high-dimensional manipulator, demonstrating the generality of the framework. In addition, we integrate a simple yet effective method for avoiding self-collisions, allowing the robot enabling comprehensive whole-body collision-free operation. Extensive benchmark comparisons and experiments demonstrate that our method performs well in unknown and dynamic scenarios, including difficult tasks like avoiding sticks swung by humans and rapidly thrown objects.
>
---
#### [replaced 004] LATMOS: Latent Automaton Task Model from Observation Sequences
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.08090v2](http://arxiv.org/pdf/2503.08090v2)**

> **作者:** Weixiao Zhan; Qiyue Dong; Eduardo Sebastián; Nikolay Atanasov
>
> **备注:** IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Robot task planning from high-level instructions is an important step towards deploying fully autonomous robot systems in the service sector. Three key aspects of robot task planning present challenges yet to be resolved simultaneously, namely, (i) factorization of complex tasks specifications into simpler executable subtasks, (ii) understanding of the current task state from raw observations, and (iii) planning and verification of task executions. To address these challenges, we propose LATMOS, an automata-inspired task model that, given observations from correct task executions, is able to factorize the task, while supporting verification and planning operations. LATMOS combines an observation encoder to extract the features from potentially high-dimensional observations with automata theory to learn a sequential model that encapsulates an automaton with symbols in the latent feature space. We conduct extensive evaluations in three task model learning setups: (i) abstract tasks described by logical formulas, (ii) real-world human tasks described by videos and natural language prompts and (iii) a robot task described by image and state observations. The results demonstrate the improved plan generation and verification capabilities of LATMOS across observation modalities and tasks.
>
---
#### [replaced 005] Unlocking Constraints: Source-Free Occlusion-Aware Seamless Segmentation
- **分类: cs.CV; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.21198v2](http://arxiv.org/pdf/2506.21198v2)**

> **作者:** Yihong Cao; Jiaming Zhang; Xu Zheng; Hao Shi; Kunyu Peng; Hang Liu; Kailun Yang; Hui Zhang
>
> **备注:** Accepted to ICCV 2025. All data and code will be made publicly available at https://github.com/yihong-97/UNLOCK
>
> **摘要:** Panoramic image processing is essential for omni-context perception, yet faces constraints like distortions, perspective occlusions, and limited annotations. Previous unsupervised domain adaptation methods transfer knowledge from labeled pinhole data to unlabeled panoramic images, but they require access to source pinhole data. To address these, we introduce a more practical task, i.e., Source-Free Occlusion-Aware Seamless Segmentation (SFOASS), and propose its first solution, called UNconstrained Learning Omni-Context Knowledge (UNLOCK). Specifically, UNLOCK includes two key modules: Omni Pseudo-Labeling Learning and Amodal-Driven Context Learning. While adapting without relying on source data or target labels, this framework enhances models to achieve segmentation with 360{\deg} viewpoint coverage and occlusion-aware reasoning. Furthermore, we benchmark the proposed SFOASS task through both real-to-real and synthetic-to-real adaptation settings. Experimental results show that our source-free method achieves performance comparable to source-dependent methods, yielding state-of-the-art scores of 10.9 in mAAP and 11.6 in mAP, along with an absolute improvement of +4.3 in mAPQ over the source-only method. All data and code will be made publicly available at https://github.com/yihong-97/UNLOCK.
>
---
#### [replaced 006] FlowNav: Combining Flow Matching and Depth Priors for Efficient Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2411.09524v3](http://arxiv.org/pdf/2411.09524v3)**

> **作者:** Samiran Gode; Abhijeet Nayak; Débora N. P. Oliveira; Michael Krawez; Cordelia Schmid; Wolfram Burgard
>
> **备注:** Accepted to IROS'25. Previous version accepted at CoRL 2024 workshop on Learning Effective Abstractions for Planning (LEAP) and workshop on Differentiable Optimization Everywhere: Simulation, Estimation, Learning, and Control
>
> **摘要:** Effective robot navigation in unseen environments is a challenging task that requires precise control actions at high frequencies. Recent advances have framed it as an image-goal-conditioned control problem, where the robot generates navigation actions using frontal RGB images. Current state-of-the-art methods in this area use diffusion policies to generate these control actions. Despite their promising results, these models are computationally expensive and suffer from weak perception. To address these limitations, we present FlowNav, a novel approach that uses a combination of CFM and depth priors from off-the-shelf foundation models to learn action policies for robot navigation. FlowNav is significantly more accurate and faster at navigation and exploration than state-of-the-art methods. We validate our contributions using real robot experiments in multiple environments, demonstrating improved navigation reliability and accuracy. Code and trained models are publicly available.
>
---
#### [replaced 007] Vidar: Embodied Video Diffusion Model for Generalist Bimanual Manipulation
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.12898v2](http://arxiv.org/pdf/2507.12898v2)**

> **作者:** Yao Feng; Hengkai Tan; Xinyi Mao; Guodong Liu; Shuhe Huang; Chendong Xiang; Hang Su; Jun Zhu
>
> **摘要:** Bimanual robotic manipulation, which involves the coordinated control of two robotic arms, is foundational for solving challenging tasks. Despite recent progress in general-purpose manipulation, data scarcity and embodiment heterogeneity remain serious obstacles to further scaling up in bimanual settings. In this paper, we introduce Video Diffusion for Action Reasoning (Vidar), a two-stage framework that leverages large-scale, diffusion-based video pre-training and a novel masked inverse dynamics model for action prediction. We pre-train the video diffusion model on 750K multi-view videos from three real-world bimanual robot platforms, utilizing a unified observation space that encodes robot, camera, task, and scene contexts. Our masked inverse dynamics model learns masks to extract action-relevant information from generated trajectories without requiring pixel-level labels, and the masks can effectively generalize to unseen backgrounds. Our experiments demonstrate that with only 20 minutes of human demonstrations on an unseen robot platform (only 1% of typical data requirements), Vidar generalizes to unseen tasks and backgrounds with strong semantic understanding, surpassing state-of-the-art methods. Our findings highlight the potential of video foundation models, coupled with masked action prediction, to enable scalable and generalizable robotic manipulation in diverse real-world settings.
>
---
#### [replaced 008] Recasting Classical Motion Planning for Contact-Rich Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.00351v2](http://arxiv.org/pdf/2506.00351v2)**

> **作者:** Lin Yang; Huu-Thiet Nguyen; Chen Lv; Domenico Campolo
>
> **摘要:** In this work, we explore how conventional motion planning algorithms can be reapplied to contact-rich manipulation tasks. Rather than focusing solely on efficiency, we investigate how manipulation aspects can be recast in terms of conventional motion-planning algorithms. Conventional motion planners, such as Rapidly-Exploring Random Trees (RRT), typically compute collision-free paths in configuration space. However, in many manipulation tasks, contact is either unavoidable or essential for task success, such as for creating space or maintaining physical equilibrium. As such, we presents Haptic Rapidly-Exploring Random Trees (HapticRRT), a planning algorithm that incorporates a recently proposed optimality measure in the context of \textit{quasi-static} manipulation, based on the (squared) Hessian of manipulation potential. The key contributions are i) adapting classical RRT to operate on the quasi-static equilibrium manifold, while deepening the interpretation of haptic obstacles and metrics; ii) discovering multiple manipulation strategies, corresponding to branches of the equilibrium manifold. iii) validating the generality of our method across three diverse manipulation tasks, each requiring only a single manipulation potential expression. The video can be found at https://youtu.be/R8aBCnCCL40.
>
---
#### [replaced 009] Cooperative Payload Estimation by a Team of Mocobots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.04600v2](http://arxiv.org/pdf/2502.04600v2)**

> **作者:** Haoxuan Zhang; C. Lin Liu; Matthew L. Elwin; Randy A. Freeman; Kevin M. Lynch
>
> **备注:** 8 pages, 6 figures. Submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** For high-performance autonomous manipulation of a payload by a mobile manipulator team, or for collaborative manipulation with the human, robots should be able to discover where other robots are attached to the payload, as well as the payload's mass and inertial properties. In this paper, we describe a method for the robots to autonomously discover this information. The robots cooperatively manipulate the payload, and the twist, twist derivative, and wrench data at their grasp frames are used to estimate the transformation matrices between the grasp frames, the location of the payload's center of mass, and the payload's inertia matrix. The method is validated experimentally with a team of three mobile cobots, or mocobots.
>
---
#### [replaced 010] Context-Aware Deep Lagrangian Networks for Model Predictive Control
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.15249v3](http://arxiv.org/pdf/2506.15249v3)**

> **作者:** Lucas Schulze; Jan Peters; Oleg Arenz
>
> **备注:** Accepted to the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Controlling a robot based on physics-consistent dynamic models, such as Deep Lagrangian Networks (DeLaN), can improve the generalizability and interpretability of the resulting behavior. However, in complex environments, the number of objects to potentially interact with is vast, and their physical properties are often uncertain. This complexity makes it infeasible to employ a single global model. Therefore, we need to resort to online system identification of context-aware models that capture only the currently relevant aspects of the environment. While physical principles such as the conservation of energy may not hold across varying contexts, ensuring physical plausibility for any individual context-aware model can still be highly desirable, particularly when using it for receding horizon control methods such as model predictive control (MPC). Hence, in this work, we extend DeLaN to make it context-aware, combine it with a recurrent network for online system identification, and integrate it with an MPC for adaptive, physics-consistent control. We also combine DeLaN with a residual dynamics model to leverage the fact that a nominal model of the robot is typically available. We evaluate our method on a 7-DOF robot arm for trajectory tracking under varying loads. Our method reduces the end-effector tracking error by 39%, compared to a 21% improvement achieved by a baseline that uses an extended Kalman filter.
>
---
#### [replaced 011] TPK: Trustworthy Trajectory Prediction Integrating Prior Knowledge For Interpretability and Kinematic Feasibility
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.06743v3](http://arxiv.org/pdf/2505.06743v3)**

> **作者:** Marius Baden; Ahmed Abouelazm; Christian Hubschneider; Yin Wu; Daniel Slieter; J. Marius Zöllner
>
> **备注:** First and Second authors contributed equally; Accepted in the 36th IEEE Intelligent Vehicles Symposium (IV 2025) for oral presentation; Winner of the best paper award
>
> **摘要:** Trajectory prediction is crucial for autonomous driving, enabling vehicles to navigate safely by anticipating the movements of surrounding road users. However, current deep learning models often lack trustworthiness as their predictions can be physically infeasible and illogical to humans. To make predictions more trustworthy, recent research has incorporated prior knowledge, like the social force model for modeling interactions and kinematic models for physical realism. However, these approaches focus on priors that suit either vehicles or pedestrians and do not generalize to traffic with mixed agent classes. We propose incorporating interaction and kinematic priors of all agent classes--vehicles, pedestrians, and cyclists with class-specific interaction layers to capture agent behavioral differences. To improve the interpretability of the agent interactions, we introduce DG-SFM, a rule-based interaction importance score that guides the interaction layer. To ensure physically feasible predictions, we proposed suitable kinematic models for all agent classes with a novel pedestrian kinematic model. We benchmark our approach on the Argoverse 2 dataset, using the state-of-the-art transformer HPTR as our baseline. Experiments demonstrate that our method improves interaction interpretability, revealing a correlation between incorrect predictions and divergence from our interaction prior. Even though incorporating the kinematic models causes a slight decrease in accuracy, they eliminate infeasible trajectories found in the dataset and the baseline model. Thus, our approach fosters trust in trajectory prediction as its interaction reasoning is interpretable, and its predictions adhere to physics.
>
---
#### [replaced 012] Interleaved Multitask Learning with Energy Modulated Learning Progress
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.00707v2](http://arxiv.org/pdf/2504.00707v2)**

> **作者:** Hanne Say; Suzan Ece Ada; Emre Ugur; Minoru Asada; Erhan Oztop
>
> **备注:** submitted to Neural Networks Journal (under review), 48 pages, 11 figures
>
> **摘要:** As humans learn new skills and apply their existing knowledge while maintaining previously learned information, "continual learning" in machine learning aims to incorporate new data while retaining and utilizing past knowledge. However, existing machine learning methods often does not mimic human learning where tasks are intermixed due to individual preferences and environmental conditions. Humans typically switch between tasks instead of completely mastering one task before proceeding to the next. To explore how human-like task switching can enhance learning efficiency, we propose a multi task learning architecture that alternates tasks based on task-agnostic measures such as "learning progress" and "neural computational energy expenditure". To evaluate the efficacy of our method, we run several systematic experiments by using a set of effect-prediction tasks executed by a simulated manipulator robot. The experiments show that our approach surpasses random interleaved and sequential task learning in terms of average learning accuracy. Moreover, by including energy expenditure in the task switching logic, our approach can still perform favorably while reducing neural energy expenditure.
>
---
#### [replaced 013] Free-form language-based robotic reasoning and grasping
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13082v2](http://arxiv.org/pdf/2503.13082v2)**

> **作者:** Runyu Jiao; Alice Fasoli; Francesco Giuliari; Matteo Bortolon; Sergio Povoli; Guofeng Mei; Yiming Wang; Fabio Poiesi
>
> **备注:** Accepted to IROS 2025. Project website: https://tev-fbk.github.io/FreeGrasp/
>
> **摘要:** Performing robotic grasping from a cluttered bin based on human instructions is a challenging task, as it requires understanding both the nuances of free-form language and the spatial relationships between objects. Vision-Language Models (VLMs) trained on web-scale data, such as GPT-4o, have demonstrated remarkable reasoning capabilities across both text and images. But can they truly be used for this task in a zero-shot setting? And what are their limitations? In this paper, we explore these research questions via the free-form language-based robotic grasping task, and propose a novel method, FreeGrasp, leveraging the pre-trained VLMs' world knowledge to reason about human instructions and object spatial arrangements. Our method detects all objects as keypoints and uses these keypoints to annotate marks on images, aiming to facilitate GPT-4o's zero-shot spatial reasoning. This allows our method to determine whether a requested object is directly graspable or if other objects must be grasped and removed first. Since no existing dataset is specifically designed for this task, we introduce a synthetic dataset FreeGraspData by extending the MetaGraspNetV2 dataset with human-annotated instructions and ground-truth grasping sequences. We conduct extensive analyses with both FreeGraspData and real-world validation with a gripper-equipped robotic arm, demonstrating state-of-the-art performance in grasp reasoning and execution. Project website: https://tev-fbk.github.io/FreeGrasp/.
>
---
#### [replaced 014] Modeling the Dynamics of Sub-Millisecond Electroadhesive Engagement and Release Times
- **分类: cs.RO; cs.HC; physics.app-ph**

- **链接: [http://arxiv.org/pdf/2412.16803v2](http://arxiv.org/pdf/2412.16803v2)**

> **作者:** Ahad M. Rauf; Sean Follmer
>
> **备注:** This work has been published in Extreme Mechanics Letters
>
> **摘要:** Electroadhesive clutches are electrically controllable switchable adhesives commonly used in soft robots and haptic user interfaces. They can form strong bonds to a wide variety of surfaces at low power consumption. However, electroadhesive clutches in the literature engage to and release from substrates several orders of magnitude slower than a traditional electrostatic model would predict. Large release times, in particular, can limit electroadhesion's usefulness in high-bandwidth applications. We develop a novel electromechanical model for electroadhesion, factoring in polarization dynamics, the drive circuitry's rise and fall times, and contact mechanics between the dielectric and substrate. We show in simulation and experimentally how different design parameters affect the engagement and release times of centimeter-scale electroadhesive clutches to metallic substrates, and we find that the model accurately captures the magnitude and trends of our experimental results. In particular, we find that higher drive frequencies, narrower substrate aspect ratios, and faster drive circuitry output stages enable significantly faster release times. The fastest clutches have engagement times less than 15 us and release times less than 875 us, which are 10x and 17.1x faster, respectively, than the best times found in prior literature on centimeter-scale electroadhesive clutches.
>
---
#### [replaced 015] Hydra-NeXt: Robust Closed-Loop Driving with Open-Loop Training
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.12030v2](http://arxiv.org/pdf/2503.12030v2)**

> **作者:** Zhenxin Li; Shihao Wang; Shiyi Lan; Zhiding Yu; Zuxuan Wu; Jose M. Alvarez
>
> **摘要:** End-to-end autonomous driving research currently faces a critical challenge in bridging the gap between open-loop training and closed-loop deployment. Current approaches are trained to predict trajectories in an open-loop environment, which struggle with quick reactions to other agents in closed-loop environments and risk generating kinematically infeasible plans due to the gap between open-loop training and closed-loop driving. In this paper, we introduce Hydra-NeXt, a novel multi-branch planning framework that unifies trajectory prediction, control prediction, and a trajectory refinement network in one model. Unlike current open-loop trajectory prediction models that only handle general-case planning, Hydra-NeXt further utilizes a control decoder to focus on short-term actions, which enables faster responses to dynamic situations and reactive agents. Moreover, we propose the Trajectory Refinement module to augment and refine the planning decisions by effectively adhering to kinematic constraints in closed-loop environments. This unified approach bridges the gap between open-loop training and closed-loop driving, demonstrating superior performance of 65.89 Driving Score (DS) and 48.20% Success Rate (SR) on the Bench2Drive dataset without relying on external experts for data collection. Hydra-NeXt surpasses the previous state-of-the-art by 22.98 DS and 17.49 SR, marking a significant advancement in autonomous driving. Code will be available at https://github.com/woxihuanjiangguo/Hydra-NeXt.
>
---
#### [replaced 016] Perpetua: Multi-Hypothesis Persistence Modeling for Semi-Static Environments
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.18808v2](http://arxiv.org/pdf/2507.18808v2)**

> **作者:** Miguel Saavedra-Ruiz; Samer B. Nashed; Charlie Gauthier; Liam Paull
>
> **备注:** Accepted to the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025) Code available at https://github.com/montrealrobotics/perpetua-code. Webpage and additional videos at https://montrealrobotics.ca/perpetua/
>
> **摘要:** Many robotic systems require extended deployments in complex, dynamic environments. In such deployments, parts of the environment may change between subsequent robot observations. Most robotic mapping or environment modeling algorithms are incapable of representing dynamic features in a way that enables predicting their future state. Instead, they opt to filter certain state observations, either by removing them or some form of weighted averaging. This paper introduces Perpetua, a method for modeling the dynamics of semi-static features. Perpetua is able to: incorporate prior knowledge about the dynamics of the feature if it exists, track multiple hypotheses, and adapt over time to enable predicting of future feature states. Specifically, we chain together mixtures of "persistence" and "emergence" filters to model the probability that features will disappear or reappear in a formal Bayesian framework. The approach is an efficient, scalable, general, and robust method for estimating the states of features in an environment, both in the present as well as at arbitrary future times. Through experiments on simulated and real-world data, we find that Perpetua yields better accuracy than similar approaches while also being online adaptable and robust to missing observations.
>
---
#### [replaced 017] Investigation of the Challenges of Underwater-Visual-Monocular-SLAM
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2306.08738v2](http://arxiv.org/pdf/2306.08738v2)**

> **作者:** Michele Grimaldi; David Nakath; Mengkun She; Kevin Köser
>
> **摘要:** In this paper, we present a comprehensive investigation of the challenges of Monocular Visual Simultaneous Localization and Mapping (vSLAM) methods for underwater robots. While significant progress has been made in state estimation methods that utilize visual data in the past decade, most evaluations have been limited to controlled indoor and urban environments, where impressive performance was demonstrated. However, these techniques have not been extensively tested in extremely challenging conditions, such as underwater scenarios where factors such as water and light conditions, robot path, and depth can greatly impact algorithm performance. Hence, our evaluation is conducted in real-world AUV scenarios as well as laboratory settings which provide precise external reference. A focus is laid on understanding the impact of environmental conditions, such as optical properties of the water and illumination scenarios, on the performance of monocular vSLAM methods. To this end, we first show that all methods perform very well in in-air settings and subsequently show the degradation of their performance in challenging underwater environments. The final goal of this study is to identify techniques that can improve accuracy and robustness of SLAM methods in such conditions. To achieve this goal, we investigate the potential of image enhancement techniques to improve the quality of input images used by the SLAM methods, specifically in low visibility and extreme lighting scenarios in scattering media. We present a first evaluation on calibration maneuvers and simple image restoration techniques to determine their ability to enable or enhance the performance of monocular SLAM methods in underwater environments.
>
---
#### [replaced 018] SparseLoc: Sparse Open-Set Landmark-based Global Localization for Autonomous Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.23465v2](http://arxiv.org/pdf/2503.23465v2)**

> **作者:** Pranjal Paul; Vineeth Bhat; Tejas Salian; Mohammad Omama; Krishna Murthy Jatavallabhula; Naveen Arulselvan; K. Madhava Krishna
>
> **摘要:** Global localization is a critical problem in autonomous navigation, enabling precise positioning without reliance on GPS. Modern global localization techniques often depend on dense LiDAR maps, which, while precise, require extensive storage and computational resources. Recent approaches have explored alternative methods, such as sparse maps and learned features, but they suffer from poor robustness and generalization. We propose SparseLoc, a global localization framework that leverages vision-language foundation models to generate sparse, semantic-topometric maps in a zero-shot manner. It combines this map representation with a Monte Carlo localization scheme enhanced by a novel late optimization strategy, ensuring improved pose estimation. By constructing compact yet highly discriminative maps and refining localization through a carefully designed optimization schedule, SparseLoc overcomes the limitations of existing techniques, offering a more efficient and robust solution for global localization. Our system achieves over a 5X improvement in localization accuracy compared to existing sparse mapping techniques. Despite utilizing only 1/500th of the points of dense mapping methods, it achieves comparable performance, maintaining an average global localization error below 5m and 2 degrees on KITTI sequences.
>
---
#### [replaced 019] Safe and Real-Time Consistent Planning for Autonomous Vehicles in Partially Observed Environments via Parallel Consensus Optimization
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2409.10310v2](http://arxiv.org/pdf/2409.10310v2)**

> **作者:** Lei Zheng; Rui Yang; Minzhe Zheng; Michael Yu Wang; Jun Ma
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** Ensuring safety and driving consistency is a significant challenge for autonomous vehicles operating in partially observed environments. This work introduces a consistent parallel trajectory optimization (CPTO) approach to enable safe and consistent driving in dense obstacle environments with perception uncertainties. Utilizing discrete-time barrier function theory, we develop a consensus safety barrier module that ensures reliable safety coverage within the spatiotemporal trajectory space across potential obstacle configurations. Following this, a bi-convex parallel trajectory optimization problem is derived that facilitates decomposition into a series of low-dimensional quadratic programming problems to accelerate computation. By leveraging the consensus alternating direction method of multipliers (ADMM) for parallel optimization, each generated candidate trajectory corresponds to a possible environment configuration while sharing a common consensus trajectory segment. This ensures driving safety and consistency when executing the consensus trajectory segment for the ego vehicle in real time. We validate our CPTO framework through extensive comparisons with state-of-the-art baselines across multiple driving tasks in partially observable environments. Our results demonstrate improved safety and consistency using both synthetic and real-world traffic datasets.
>
---
#### [replaced 020] Trends in Motion Prediction Toward Deployable and Generalizable Autonomy: A Revisit and Perspectives
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.09074v2](http://arxiv.org/pdf/2505.09074v2)**

> **作者:** Letian Wang; Marc-Antoine Lavoie; Sandro Papais; Barza Nisar; Yuxiao Chen; Wenhao Ding; Boris Ivanovic; Hao Shao; Abulikemu Abuduweili; Evan Cook; Yang Zhou; Peter Karkus; Jiachen Li; Changliu Liu; Marco Pavone; Steven Waslander
>
> **备注:** Book Published by Foundation and Trends in Robotics. 162 pages, 40 figures, 13 tables
>
> **摘要:** Motion prediction, the anticipation of future agent states or scene evolution, is rooted in human cognition, bridging perception and decision-making. It enables intelligent systems, such as robots and self-driving cars, to act safely in dynamic, human-involved environments, and informs broader time-series reasoning challenges. With advances in methods, representations, and datasets, the field has seen rapid progress, reflected in quickly evolving benchmark results. Yet, when state-of-the-art methods are deployed in the real world, they often struggle to generalize to open-world conditions and fall short of deployment standards. This reveals a gap between research benchmarks, which are often idealized or ill-posed, and real-world complexity. To address this gap, this survey revisits the generalization and deployability of motion prediction models, with an emphasis on the applications of robotics, autonomous driving, and human motion. We first offer a comprehensive taxonomy of motion prediction methods, covering representations, modeling strategies, application domains, and evaluation protocols. We then study two key challenges: (1) how to push motion prediction models to be deployable to realistic deployment standards, where motion prediction does not act in a vacuum, but functions as one module of closed-loop autonomy stacks - it takes input from the localization and perception, and informs downstream planning and control. 2) how to generalize motion prediction models from limited seen scenarios/datasets to the open-world settings. Throughout the paper, we highlight critical open challenges to guide future work, aiming to recalibrate the community's efforts, fostering progress that is not only measurable but also meaningful for real-world applications. The project webpage corresponding to this paper can be found here https://trends-in-motion-prediction- 2025.github.io/.
>
---
#### [replaced 021] A Step-by-step Guide on Nonlinear Model Predictive Control for Safe Mobile Robot Navigation
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.17856v2](http://arxiv.org/pdf/2507.17856v2)**

> **作者:** Dennis Benders; Laura Ferranti; Johannes Köhler
>
> **备注:** 51 pages, 3 figures
>
> **摘要:** Designing a Model Predictive Control (MPC) scheme that enables a mobile robot to safely navigate through an obstacle-filled environment is a complicated yet essential task in robotics. In this technical report, safety refers to ensuring that the robot respects state and input constraints while avoiding collisions with obstacles despite the presence of disturbances and measurement noise. This report offers a step-by-step approach to implementing Nonlinear Model Predictive Control (NMPC) schemes addressing these safety requirements. Numerous books and survey papers provide comprehensive overviews of linear MPC (LMPC), NMPC, and their applications in various domains, including robotics. This report does not aim to replicate those exhaustive reviews. Instead, it focuses specifically on NMPC as a foundation for safe mobile robot navigation. The goal is to provide a practical and accessible path from theoretical concepts to mathematical proofs and implementation, emphasizing safety and performance guarantees. It is intended for researchers, robotics engineers, and practitioners seeking to bridge the gap between theoretical NMPC formulations and real-world robotic applications. This report is not necessarily meant to remain fixed over time. If someone finds an error in the presented theory, please reach out via the given email addresses. We are happy to update the document if necessary.
>
---
#### [replaced 022] Bi-LAT: Bilateral Control-Based Imitation Learning via Natural Language and Action Chunking with Transformers
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.01301v2](http://arxiv.org/pdf/2504.01301v2)**

> **作者:** Takumi Kobayashi; Masato Kobayashi; Thanpimon Buamanee; Yuki Uranishi
>
> **摘要:** We present Bi-LAT, a novel imitation learning framework that unifies bilateral control with natural language processing to achieve precise force modulation in robotic manipulation. Bi-LAT leverages joint position, velocity, and torque data from leader-follower teleoperation while also integrating visual and linguistic cues to dynamically adjust applied force. By encoding human instructions such as "softly grasp the cup" or "strongly twist the sponge" through a multimodal Transformer-based model, Bi-LAT learns to distinguish nuanced force requirements in real-world tasks. We demonstrate Bi-LAT's performance in (1) unimanual cup-stacking scenario where the robot accurately modulates grasp force based on language commands, and (2) bimanual sponge-twisting task that requires coordinated force control. Experimental results show that Bi-LAT effectively reproduces the instructed force levels, particularly when incorporating SigLIP among tested language encoders. Our findings demonstrate the potential of integrating natural language cues into imitation learning, paving the way for more intuitive and adaptive human-robot interaction. For additional material, please visit: https://mertcookimg.github.io/bi-lat/
>
---
#### [replaced 023] ADA-DPM: A Neural Descriptors-based Adaptive Noise Filtering Strategy for SLAM
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18016v2](http://arxiv.org/pdf/2506.18016v2)**

> **作者:** Yongxin Shao; Aihong Tan; Binrui Wang; Yinlian Jin; Licong Guan; Peng Liao
>
> **摘要:** Lidar SLAM plays a significant role in mobile robot navigation and high-definition map construction. However, existing methods often face a trade-off between localization accuracy and system robustness in scenarios with a high proportion of dynamic objects, point cloud distortion, and unstructured environments. To address this issue, we propose a neural descriptors-based adaptive noise filtering strategy for SLAM, named ADA-DPM, which improves the performance of localization and mapping tasks through three key technical innovations. Firstly, to tackle dynamic object interference, we design the Dynamic Segmentation Head to predict and filter out dynamic feature points, eliminating the ego-motion interference caused by dynamic objects. Secondly, to mitigate the impact of noise and unstructured feature points, we propose the Global Importance Scoring Head that adaptively selects high-contribution feature points while suppressing the influence of noise and unstructured feature points. Moreover, we introduce the Cross-Layer Graph Convolution Module (GLI-GCN) to construct multi-scale neighborhood graphs, fusing local structural information across different scales and improving the discriminative power of overlapping features. Finally, experimental validations on multiple public datasets confirm the effectiveness of ADA-DPM.
>
---
#### [replaced 024] Learning Local Heuristics for Search-Based Navigation Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2303.09477v2](http://arxiv.org/pdf/2303.09477v2)**

> **作者:** Rishi Veerapaneni; Muhammad Suhail Saleem; Maxim Likhachev
>
> **备注:** Published at the International Conference on Automated Planning and Scheduling 2023 (ICAPS 2023)
>
> **摘要:** Graph search planning algorithms for navigation typically rely heavily on heuristics to efficiently plan paths. As a result, while such approaches require no training phase and can directly plan long horizon paths, they often require careful hand designing of informative heuristic functions. Recent works have started bypassing hand designed heuristics by using machine learning to learn heuristic functions that guide the search algorithm. While these methods can learn complex heuristic functions from raw input, they i) require a significant training phase and ii) do not generalize well to new maps and longer horizon paths. Our contribution is showing that instead of learning a global heuristic estimate, we can define and learn local heuristics which results in a significantly smaller learning problem and improves generalization. We show that using such local heuristics can reduce node expansions by 2-20x while maintaining bounded suboptimality, are easy to train, and generalize to new maps & long horizon plans.
>
---
#### [replaced 025] Competency-Aware Planning for Probabilistically Safe Navigation Under Perception Uncertainty
- **分类: cs.RO; cs.AI; cs.CV; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2409.06111v5](http://arxiv.org/pdf/2409.06111v5)**

> **作者:** Sara Pohland; Claire Tomlin
>
> **摘要:** Perception-based navigation systems are useful for unmanned ground vehicle (UGV) navigation in complex terrains, where traditional depth-based navigation schemes are insufficient. However, these data-driven methods are highly dependent on their training data and can fail in surprising and dramatic ways with little warning. To ensure the safety of the vehicle and the surrounding environment, it is imperative that the navigation system is able to recognize the predictive uncertainty of the perception model and respond safely and effectively in the face of uncertainty. In an effort to enable safe navigation under perception uncertainty, we develop a probabilistic and reconstruction-based competency estimation (PaRCE) method to estimate the model's level of familiarity with an input image as a whole and with specific regions in the image. We find that the overall competency score can correctly predict correctly classified, misclassified, and out-of-distribution (OOD) samples. We also confirm that the regional competency maps can accurately distinguish between familiar and unfamiliar regions across images. We then use this competency information to develop a planning and control scheme that enables effective navigation while maintaining a low probability of error. We find that the competency-aware scheme greatly reduces the number of collisions with unfamiliar obstacles, compared to a baseline controller with no competency awareness. Furthermore, the regional competency information is very valuable in enabling efficient navigation.
>
---
#### [replaced 026] DiffOG: Differentiable Policy Trajectory Optimization with Generalizability
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.13807v4](http://arxiv.org/pdf/2504.13807v4)**

> **作者:** Zhengtong Xu; Zichen Miao; Qiang Qiu; Zhe Zhang; Yu She
>
> **摘要:** Imitation learning-based visuomotor policies excel at manipulation tasks but often produce suboptimal action trajectories compared to model-based methods. Directly mapping camera data to actions via neural networks can result in jerky motions and difficulties in meeting critical constraints, compromising safety and robustness in real-world deployment. For tasks that require high robustness or strict adherence to constraints, ensuring trajectory quality is crucial. However, the lack of interpretability in neural networks makes it challenging to generate constraint-compliant actions in a controlled manner. This paper introduces differentiable policy trajectory optimization with generalizability (DiffOG), a learning-based trajectory optimization framework designed to enhance visuomotor policies. By leveraging the proposed differentiable formulation of trajectory optimization with transformer, DiffOG seamlessly integrates policies with a generalizable optimization layer. DiffOG refines action trajectories to be smoother and more constraint-compliant while maintaining alignment with the original demonstration distribution, thus avoiding degradation in policy performance. We evaluated DiffOG across 11 simulated tasks and 2 real-world tasks. The results demonstrate that DiffOG significantly enhances the trajectory quality of visuomotor policies while having minimal impact on policy performance, outperforming trajectory processing baselines such as greedy constraint clipping and penalty-based trajectory optimization. Furthermore, DiffOG achieves superior performance compared to existing constrained visuomotor policy. For more details, please visit the project website: https://zhengtongxu.github.io/diffog-website/.
>
---
#### [replaced 027] SHINE: Social Homology Identification for Navigation in Crowded Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2404.16705v2](http://arxiv.org/pdf/2404.16705v2)**

> **作者:** Diego Martinez-Baselga; Oscar de Groot; Luzia Knoedler; Luis Riazuelo; Javier Alonso-Mora; Luis Montano
>
> **备注:** This paper has been accepted for publication at The International Journal of Robotics Research. Please, when citing the paper, refer to the official manuscript with the following DOI: 10.1177/02783649251344639
>
> **摘要:** Navigating mobile robots in social environments remains a challenging task due to the intricacies of human-robot interactions. Most of the motion planners designed for crowded and dynamic environments focus on choosing the best velocity to reach the goal while avoiding collisions, but do not explicitly consider the high-level navigation behavior (avoiding through the left or right side, letting others pass or passing before others, etc.). In this work, we present a novel motion planner that incorporates topology distinct paths representing diverse navigation strategies around humans. The planner selects the topology class that imitates human behavior the best using a deep neural network model trained on real-world human motion data, ensuring socially intelligent and contextually aware navigation. Our system refines the chosen path through an optimization-based local planner in real time, ensuring seamless adherence to desired social behaviors. In this way, we decouple perception and local planning from the decision-making process. We evaluate the prediction accuracy of the network with real-world data. In addition, we assess the navigation capabilities in both simulation and a real-world platform, comparing it with other state-of-the-art planners. We demonstrate that our planner exhibits socially desirable behaviors and shows a smooth and remarkable performance.
>
---
#### [replaced 028] RESC: A Reinforcement Learning Based Search-to-Control Framework for Quadrotor Local Planning in Dense Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2408.00275v5](http://arxiv.org/pdf/2408.00275v5)**

> **作者:** Zhaohong Liu; Wenxuan Gao; Yinshuai Sun; Peng Dong
>
> **备注:** This paper has been accepted for publication in IEEE Robotics and Automation Letters (RAL), 2025. The final authenticated version is available online at IEEE Xplore
>
> **摘要:** Agile flight in complex environments poses significant challenges to current motion planning methods, as they often fail to fully leverage the quadrotor dynamic potential, leading to performance failures and reduced efficiency during aggressive maneuvers.Existing approaches frequently decouple trajectory optimization from control generation and neglect the dynamics, further limiting their ability to generate aggressive and feasible motions.To address these challenges, we introduce an enhanced Search-to-Control planning framework that integrates visibility path searching with reinforcement learning (RL) control generation, directly accounting for dynamics and bridging the gap between planning and control.Our method first extracts control points from collision-free paths using a proposed heuristic search, which are then refined by an RL policy to generate low-level control commands for the quadrotor controller, utilizing reduced-dimensional obstacle observations for efficient inference with lightweight neural networks.We validate the framework through simulations and real-world experiments, demonstrating improved time efficiency and dynamic maneuverability compared to existing methods, while confirming its robustness and applicability.
>
---
#### [replaced 029] GSplatVNM: Point-of-View Synthesis for Visual Navigation Models Using Gaussian Splatting
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.05152v3](http://arxiv.org/pdf/2503.05152v3)**

> **作者:** Kohei Honda; Takeshi Ishita; Yasuhiro Yoshimura; Ryo Yonetani
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** This paper presents a novel approach to image-goal navigation by integrating 3D Gaussian Splatting (3DGS) with Visual Navigation Models (VNMs), a method we refer to as GSplatVNM. VNMs offer a promising paradigm for image-goal navigation by guiding a robot through a sequence of point-of-view images without requiring metrical localization or environment-specific training. However, constructing a dense and traversable sequence of target viewpoints from start to goal remains a central challenge, particularly when the available image database is sparse. To address these challenges, we propose a 3DGS-based viewpoint synthesis framework for VNMs that synthesizes intermediate viewpoints to seamlessly bridge gaps in sparse data while significantly reducing storage overhead. Experimental results in a photorealistic simulator demonstrate that our approach not only enhances navigation efficiency but also exhibits robustness under varying levels of image database sparsity.
>
---
#### [replaced 030] Critiques of World Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.05169v3](http://arxiv.org/pdf/2507.05169v3)**

> **作者:** Eric Xing; Mingkai Deng; Jinyu Hou; Zhiting Hu
>
> **摘要:** World Model, the supposed algorithmic surrogate of the real-world environment which biological agents experience with and act upon, has been an emerging topic in recent years because of the rising needs to develop virtual agents with artificial (general) intelligence. There has been much debate on what a world model really is, how to build it, how to use it, and how to evaluate it. In this essay, starting from the imagination in the famed Sci-Fi classic Dune, and drawing inspiration from the concept of "hypothetical thinking" in psychology literature, we offer critiques of several schools of thoughts on world modeling, and argue the primary goal of a world model to be simulating all actionable possibilities of the real world for purposeful reasoning and acting. Building on the critiques, we propose a new architecture for a general-purpose world model, based on hierarchical, multi-level, and mixed continuous/discrete representations, and a generative and self-supervision learning framework, with an outlook of a Physical, Agentic, and Nested (PAN) AGI system enabled by such a model.
>
---
#### [replaced 031] MRHaD: Mixed Reality-based Hand-Drawn Map Editing Interface for Mobile Robot Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.00580v2](http://arxiv.org/pdf/2504.00580v2)**

> **作者:** Takumi Taki; Masato Kobayashi; Eduardo Iglesius; Naoya Chiba; Shizuka Shirai; Yuki Uranishi
>
> **摘要:** Mobile robot navigation systems are increasingly relied upon in dynamic and complex environments, yet they often struggle with map inaccuracies and the resulting inefficient path planning. This paper presents MRHaD, a Mixed Reality-based Hand-drawn Map Editing Interface that enables intuitive, real-time map modifications through natural hand gestures. By integrating the MR head-mounted display with the robotic navigation system, operators can directly create hand-drawn restricted zones (HRZ), thereby bridging the gap between 2D map representations and the real-world environment. Comparative experiments against conventional 2D editing methods demonstrate that MRHaD significantly improves editing efficiency, map accuracy, and overall usability, contributing to safer and more efficient mobile robot operations. The proposed approach provides a robust technical foundation for advancing human-robot collaboration and establishing innovative interaction models that enhance the hybrid future of robotics and human society. For additional material, please check: https://mertcookimg.github.io/mrhad/
>
---
#### [replaced 032] ViewActive: Active viewpoint optimization from a single image
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.09997v5](http://arxiv.org/pdf/2409.09997v5)**

> **作者:** Jiayi Wu; Xiaomin Lin; Botao He; Cornelia Fermuller; Yiannis Aloimonos
>
> **摘要:** When observing objects, humans benefit from their spatial visualization and mental rotation ability to envision potential optimal viewpoints based on the current observation. This capability is crucial for enabling robots to achieve efficient and robust scene perception during operation, as optimal viewpoints provide essential and informative features for accurately representing scenes in 2D images, thereby enhancing downstream tasks. To endow robots with this human-like active viewpoint optimization capability, we propose ViewActive, a modernized machine learning approach drawing inspiration from aspect graph, which provides viewpoint optimization guidance based solely on the current 2D image input. Specifically, we introduce the 3D Viewpoint Quality Field (VQF), a compact and consistent representation of viewpoint quality distribution similar to an aspect graph, composed of three general-purpose viewpoint quality metrics: self-occlusion ratio, occupancy-aware surface normal entropy, and visual entropy. We utilize pre-trained image encoders to extract robust visual and semantic features, which are then decoded into the 3D VQF, allowing our model to generalize effectively across diverse objects, including unseen categories. The lightweight ViewActive network (72 FPS on a single GPU) significantly enhances the performance of state-of-the-art object recognition pipelines and can be integrated into real-time motion planning for robotic applications. Our code and dataset are available here: https://github.com/jiayi-wu-umd/ViewActive.
>
---
#### [replaced 033] Eyes Will Shut: A Vision-Based Next GPS Location Prediction Model by Reinforcement Learning from Visual Map Feed Back
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.18661v2](http://arxiv.org/pdf/2507.18661v2)**

> **作者:** Ruixing Zhang; Yang Zhang; Tongyu Zhu; Leilei Sun; Weifeng Lv
>
> **摘要:** Next Location Prediction is a fundamental task in the study of human mobility, with wide-ranging applications in transportation planning, urban governance, and epidemic forecasting. In practice, when humans attempt to predict the next location in a trajectory, they often visualize the trajectory on a map and reason based on road connectivity and movement trends. However, the vast majority of existing next-location prediction models do not reason over maps \textbf{in the way that humans do}. Fortunately, the recent development of Vision-Language Models (VLMs) has demonstrated strong capabilities in visual perception and even visual reasoning. This opens up a new possibility: by rendering both the road network and trajectory onto an image and leveraging the reasoning abilities of VLMs, we can enable models to perform trajectory inference in a human-like manner. To explore this idea, we first propose a method called Vision-Guided Location Search (VGLS), which evaluates whether a general-purpose VLM is capable of trajectory-based reasoning without modifying any of its internal parameters. Based on insights from the VGLS results, we further propose our main approach: VLMLocPredictor, which is composed of two stages: In the first stage, we design two Supervised Fine-Tuning (SFT) tasks that help the VLM understand road network and trajectory structures and acquire basic reasoning ability on such visual inputs. In the second stage, we introduce Reinforcement Learning from Visual Map Feedback, enabling the model to self-improve its next-location prediction ability through interaction with the environment. Experiments conducted on datasets from four different cities show that our method achieves state-of-the-art (SOTA) performance and exhibits superior cross-city generalization compared to other LLM-based approaches.
>
---
#### [replaced 034] DG16M: A Large-Scale Dataset for Dual-Arm Grasping with Force-Optimized Grasps
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.08358v3](http://arxiv.org/pdf/2503.08358v3)**

> **作者:** Md Faizal Karim; Mohammed Saad Hashmi; Shreya Bollimuntha; Mahesh Reddy Tapeti; Gaurav Singh; Nagamanikandan Govindan; K Madhava Krishna
>
> **摘要:** Dual-arm robotic grasping is crucial for handling large objects that require stable and coordinated manipulation. While single-arm grasping has been extensively studied, datasets tailored for dual-arm settings remain scarce. We introduce a large-scale dataset of 16 million dual-arm grasps, evaluated under improved force-closure constraints. Additionally, we develop a benchmark dataset containing 300 objects with approximately 30,000 grasps, evaluated in a physics simulation environment, providing a better grasp quality assessment for dual-arm grasp synthesis methods. Finally, we demonstrate the effectiveness of our dataset by training a Dual-Arm Grasp Classifier network that outperforms the state-of-the-art methods by 15\%, achieving higher grasp success rates and improved generalization across objects.
>
---
#### [replaced 035] DogLegs: Robust Proprioceptive State Estimation for Legged Robots Using Multiple Leg-Mounted IMUs
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.04580v2](http://arxiv.org/pdf/2503.04580v2)**

> **作者:** Yibin Wu; Jian Kuang; Shahram Khorshidi; Xiaoji Niu; Lasse Klingbeil; Maren Bennewitz; Heiner Kuhlmann
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Robust and accurate proprioceptive state estimation of the main body is crucial for legged robots to execute tasks in extreme environments where exteroceptive sensors, such as LiDARs and cameras, may become unreliable. In this paper, we propose DogLegs, a state estimation system for legged robots that fuses the measurements from a body-mounted inertial measurement unit (Body-IMU), joint encoders, and multiple leg-mounted IMUs (Leg-IMU) using an extended Kalman filter (EKF). The filter system contains the error states of all IMU frames. The Leg-IMUs are used to detect foot contact, thereby providing zero-velocity measurements to update the state of the Leg-IMU frames. Additionally, we compute the relative position constraints between the Body-IMU and Leg-IMUs by the leg kinematics and use them to update the main body state and reduce the error drift of the individual IMU frames. Field experimental results have shown that our proposed DogLegs system achieves better state estimation accuracy compared to the traditional leg odometry method (using only Body-IMU and joint encoders) across various terrains. We make our datasets publicly available to benefit the research community (https://github.com/YibinWu/leg-odometry).
>
---
#### [replaced 036] Critical Anatomy-Preserving & Terrain-Augmenting Navigation (CAPTAiN): Application to Laminectomy Surgical Education
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.20496v2](http://arxiv.org/pdf/2506.20496v2)**

> **作者:** Jonathan Wang; Hisashi Ishida; David Usevitch; Kesavan Venkatesh; Yi Wang; Mehran Armand; Rachel Bronheim; Amit Jain; Adnan Munawar
>
> **摘要:** Surgical training remains a crucial milestone in modern medicine, with procedures such as laminectomy exemplifying the high risks involved. Laminectomy drilling requires precise manual control to mill bony tissue while preserving spinal segment integrity and avoiding breaches in the dura: the protective membrane surrounding the spinal cord. Despite unintended tears occurring in up to 11.3% of cases, no assistive tools are currently utilized to reduce this risk. Variability in patient anatomy further complicates learning for novice surgeons. This study introduces CAPTAiN, a critical anatomy-preserving and terrain-augmenting navigation system that provides layered, color-coded voxel guidance to enhance anatomical awareness during spinal drilling. CAPTAiN was evaluated against a standard non-navigated approach through 110 virtual laminectomies performed by 11 orthopedic residents and medical students. CAPTAiN significantly improved surgical completion rates of target anatomy (87.99% vs. 74.42%) and reduced cognitive load across multiple NASA-TLX domains. It also minimized performance gaps across experience levels, enabling novices to perform on par with advanced trainees. These findings highlight CAPTAiN's potential to optimize surgical execution and support skill development across experience levels. Beyond laminectomy, it demonstrates potential for broader applications across various surgical and drilling procedures, including those in neurosurgery, otolaryngology, and other medical fields.
>
---
#### [replaced 037] Physical simulation of Marsupial UAV-UGV Systems Connected by a Variable-Length Hanging Tether
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.12776v3](http://arxiv.org/pdf/2412.12776v3)**

> **作者:** Jose Enrique Maese; Fernando Caballero; Luis Merino
>
> **摘要:** This paper presents a simulation framework able of modeling the dynamics of a hanging tether with adjustable length, connecting a UAV to a UGV. The model incorporates the interaction between the UAV, UGV, and a winch, allowing for dynamic tether adjustments based on the relative motion of the robots. The accuracy and reliability of the simulator are assessed through extensive experiments, including comparisons with real-world experiment, to evaluate its ability to reproduce the complex tether dynamics observed in physical deployments. The results demonstrate that the simulation closely aligns with real-world behavior, particularly in constrained environments where tether effects are significant. This work provides a validated tool for studying tethered robotic systems, offering valuable insights into their motion dynamics and control strategies.
>
---
#### [replaced 038] Aether: Geometric-Aware Unified World Modeling
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.18945v3](http://arxiv.org/pdf/2503.18945v3)**

> **作者:** Aether Team; Haoyi Zhu; Yifan Wang; Jianjun Zhou; Wenzheng Chang; Yang Zhou; Zizun Li; Junyi Chen; Chunhua Shen; Jiangmiao Pang; Tong He
>
> **备注:** Project Page: https://aether-world.github.io/
>
> **摘要:** The integration of geometric reconstruction and generative modeling remains a critical challenge in developing AI systems capable of human-like spatial reasoning. This paper proposes Aether, a unified framework that enables geometry-aware reasoning in world models by jointly optimizing three core capabilities: (1) 4D dynamic reconstruction, (2) action-conditioned video prediction, and (3) goal-conditioned visual planning. Through task-interleaved feature learning, Aether achieves synergistic knowledge sharing across reconstruction, prediction, and planning objectives. Building upon video generation models, our framework demonstrates zero-shot synthetic-to-real generalization despite never observing real-world data during training. Furthermore, our approach achieves zero-shot generalization in both action following and reconstruction tasks, thanks to its intrinsic geometric modeling. Notably, even without real-world data, its reconstruction performance is comparable with or even better than that of domain-specific models. Additionally, Aether employs camera trajectories as geometry-informed action spaces, enabling effective action-conditioned prediction and visual planning. We hope our work inspires the community to explore new frontiers in physically-reasonable world modeling and its applications.
>
---
#### [replaced 039] Leveraging Analytic Gradients in Provably Safe Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.01665v2](http://arxiv.org/pdf/2506.01665v2)**

> **作者:** Tim Walter; Hannah Markgraf; Jonathan Külz; Matthias Althoff
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** The deployment of autonomous robots in safety-critical applications requires safety guarantees. Provably safe reinforcement learning is an active field of research that aims to provide such guarantees using safeguards. These safeguards should be integrated during training to reduce the sim-to-real gap. While there are several approaches for safeguarding sampling-based reinforcement learning, analytic gradient-based reinforcement learning often achieves superior performance from fewer environment interactions. However, there is no safeguarding approach for this learning paradigm yet. Our work addresses this gap by developing the first effective safeguard for analytic gradient-based reinforcement learning. We analyse existing, differentiable safeguards, adapt them through modified mappings and gradient formulations, and integrate them with a state-of-the-art learning algorithm and a differentiable simulation. Using numerical experiments on three control tasks, we evaluate how different safeguards affect learning. The results demonstrate safeguarded training without compromising performance.
>
---
#### [replaced 040] Real-Time LaCAM for Real-Time MAPF
- **分类: cs.MA; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.06091v2](http://arxiv.org/pdf/2504.06091v2)**

> **作者:** Runzhe Liang; Rishi Veerapaneni; Daniel Harabor; Jiaoyang Li; Maxim Likhachev
>
> **备注:** Published at the International Symposium on Combinatorial Search 2025 (SoCS 2025)
>
> **摘要:** The vast majority of Multi-Agent Path Finding (MAPF) methods with completeness guarantees require planning full-horizon paths. However, planning full-horizon paths can take too long and be impractical in real-world applications. Instead, real-time planning and execution, which only allows the planner a finite amount of time before executing and replanning, is more practical for real-world multi-agent systems. Several methods utilize real-time planning schemes but none are provably complete, which leads to livelock or deadlock. Our main contribution is Real-Time LaCAM, the first Real-Time MAPF method with provable completeness guarantees. We do this by leveraging LaCAM (Okumura 2023) in an incremental fashion. Our results show how we can iteratively plan for congested environments with a cutoff time of milliseconds while still maintaining the same success rate as full-horizon LaCAM. We also show how it can be used with a single-step learned MAPF policy.
>
---
