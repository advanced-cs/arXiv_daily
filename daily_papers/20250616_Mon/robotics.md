# 机器人 cs.RO

- **最新发布 25 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] Dynamic Collaborative Material Distribution System for Intelligent Robots In Smart Manufacturing
- **分类: cs.RO**

- **简介: 该论文属于多机器人协同任务，解决智能工厂中实时物料配送问题。提出轻量级深度强化学习方法，提升计算效率与部署可行性。**

- **链接: [http://arxiv.org/pdf/2506.11723v1](http://arxiv.org/pdf/2506.11723v1)**

> **作者:** Ziren Xiao; Ruxin Xiao; Chang Liu; Xinheng Wang
>
> **摘要:** The collaboration and interaction of multiple robots have become integral aspects of smart manufacturing. Effective planning and management play a crucial role in achieving energy savings and minimising overall costs. This paper addresses the real-time Dynamic Multiple Sources to Single Destination (DMS-SD) navigation problem, particularly with a material distribution case for multiple intelligent robots in smart manufacturing. Enumerated solutions, such as in \cite{xiao2022efficient}, tackle the problem by generating as many optimal or near-optimal solutions as possible but do not learn patterns from the previous experience, whereas the method in \cite{xiao2023collaborative} only uses limited information from the earlier trajectories. Consequently, these methods may take a considerable amount of time to compute results on large maps, rendering real-time operations impractical. To overcome this challenge, we propose a lightweight Deep Reinforcement Learning (DRL) method to address the DMS-SD problem. The proposed DRL method can be efficiently trained and rapidly converges to the optimal solution using the designed target-guided reward function. A well-trained DRL model significantly reduces the computation time for the next movement to a millisecond level, which improves the time up to 100 times in our experiments compared to the enumerated solutions. Moreover, the trained DRL model can be easily deployed on lightweight devices in smart manufacturing, such as Internet of Things devices and mobile phones, which only require limited computational resources.
>
---
#### [new 002] ExoStart: Efficient learning for dexterous manipulation with sensorized exoskeleton demonstrations
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决机器人手高灵巧操作难题。通过人体穿戴式外骨骼采集高质量示范数据，并结合仿真与强化学习生成有效控制策略。**

- **链接: [http://arxiv.org/pdf/2506.11775v1](http://arxiv.org/pdf/2506.11775v1)**

> **作者:** Zilin Si; Jose Enrique Chen; M. Emre Karagozler; Antonia Bronars; Jonathan Hutchinson; Thomas Lampe; Nimrod Gileadi; Taylor Howell; Stefano Saliceti; Lukasz Barczyk; Ilan Olivarez Correa; Tom Erez; Mohit Shridhar; Murilo Fernandes Martins; Konstantinos Bousmalis; Nicolas Heess; Francesco Nori; Maria Bauza Villalonga
>
> **摘要:** Recent advancements in teleoperation systems have enabled high-quality data collection for robotic manipulators, showing impressive results in learning manipulation at scale. This progress suggests that extending these capabilities to robotic hands could unlock an even broader range of manipulation skills, especially if we could achieve the same level of dexterity that human hands exhibit. However, teleoperating robotic hands is far from a solved problem, as it presents a significant challenge due to the high degrees of freedom of robotic hands and the complex dynamics occurring during contact-rich settings. In this work, we present ExoStart, a general and scalable learning framework that leverages human dexterity to improve robotic hand control. In particular, we obtain high-quality data by collecting direct demonstrations without a robot in the loop using a sensorized low-cost wearable exoskeleton, capturing the rich behaviors that humans can demonstrate with their own hands. We also propose a simulation-based dynamics filter that generates dynamically feasible trajectories from the collected demonstrations and use the generated trajectories to bootstrap an auto-curriculum reinforcement learning method that relies only on simple sparse rewards. The ExoStart pipeline is generalizable and yields robust policies that transfer zero-shot to the real robot. Our results demonstrate that ExoStart can generate dexterous real-world hand skills, achieving a success rate above 50% on a wide range of complex tasks such as opening an AirPods case or inserting and turning a key in a lock. More details and videos can be found in https://sites.google.com/view/exostart.
>
---
#### [new 003] Multi-Loco: Unifying Multi-Embodiment Legged Locomotion via Reinforcement Learning Augmented Diffusion
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在解决不同形态腿式机器人泛化 locomotion 策略的问题。通过结合扩散模型与强化学习策略，提升运动性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.11470v1](http://arxiv.org/pdf/2506.11470v1)**

> **作者:** Shunpeng Yang; Zhen Fu; Zhefeng Cao; Guo Junde; Patrick Wensing; Wei Zhang; Hua Chen
>
> **备注:** 19 pages
>
> **摘要:** Generalizing locomotion policies across diverse legged robots with varying morphologies is a key challenge due to differences in observation/action dimensions and system dynamics. In this work, we propose Multi-Loco, a novel unified framework combining a morphology-agnostic generative diffusion model with a lightweight residual policy optimized via reinforcement learning (RL). The diffusion model captures morphology-invariant locomotion patterns from diverse cross-embodiment datasets, improving generalization and robustness. The residual policy is shared across all embodiments and refines the actions generated by the diffusion model, enhancing task-aware performance and robustness for real-world deployment. We evaluated our method with a rich library of four legged robots in both simulation and real-world experiments. Compared to a standard RL framework with PPO, our approach -- replacing the Gaussian policy with a diffusion model and residual term -- achieves a 10.35% average return improvement, with gains up to 13.57% in wheeled-biped locomotion tasks. These results highlight the benefits of cross-embodiment data and composite generative architectures in learning robust, generalized locomotion skills.
>
---
#### [new 004] Construction of a Multiple-DOF Under-actuated Gripper with Force-Sensing via Deep Learning
- **分类: cs.RO**

- **简介: 该论文属于机械控制任务，旨在解决无传感器力反馈抓取问题。设计了一种多自由度欠驱动夹爪，利用LSTM模型实现力控，提升抓取性能与适应性。**

- **链接: [http://arxiv.org/pdf/2506.11570v1](http://arxiv.org/pdf/2506.11570v1)**

> **作者:** Jihao Li; Keqi Zhu; Guodong Lu; I-Ming Chen; Huixu Dong
>
> **摘要:** We present a novel under-actuated gripper with two 3-joint fingers, which realizes force feedback control by the deep learning technique- Long Short-Term Memory (LSTM) model, without any force sensor. First, a five-linkage mechanism stacked by double four-linkages is designed as a finger to automatically achieve the transformation between parallel and enveloping grasping modes. This enables the creation of a low-cost under-actuated gripper comprising a single actuator and two 3-phalange fingers. Second, we devise theoretical models of kinematics and power transmission based on the proposed gripper, accurately obtaining fingertip positions and contact forces. Through coupling and decoupling of five-linkage mechanisms, the proposed gripper offers the expected capabilities of grasping payload/force/stability and objects with large dimension ranges. Third, to realize the force control, an LSTM model is proposed to determine the grasping mode for synthesizing force-feedback control policies that exploit contact sensing after outlining the uncertainty of currents using a statistical method. Finally, a series of experiments are implemented to measure quantitative indicators, such as the payload, grasping force, force sensing, grasping stability and the dimension ranges of objects to be grasped. Additionally, the grasping performance of the proposed gripper is verified experimentally to guarantee the high versatility and robustness of the proposed gripper.
>
---
#### [new 005] Your Ride, Your Rules: Psychology and Cognition Enabled Automated Driving Systems
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决AV与乘客沟通不足的问题。提出PACE-ADS框架，通过感知和响应乘客状态提升驾驶体验与安全性。**

- **链接: [http://arxiv.org/pdf/2506.11842v1](http://arxiv.org/pdf/2506.11842v1)**

> **作者:** Zhipeng Bao; Qianwen Li
>
> **备注:** 10 figures,29 pages, one colummn
>
> **摘要:** Despite rapid advances in autonomous driving, current autonomous vehicles (AVs) lack effective bidirectional communication with occupants, limiting personalization and recovery from immobilization. This reduces comfort and trust, potentially slowing broader AV adoption. We propose PACE-ADS (Psychology and Cognition Enabled Automated Driving Systems), a human-centered autonomy framework that enables AVs to sense, interpret, and respond to both external traffic and internal occupant states. PACE-ADS comprises three foundation model-based agents: a Driver Agent that analyzes the driving context, a Psychologist Agent that interprets occupant psychological signals (e.g., EEG, heart rate, facial expressions) and cognitive commands (e.g., speech), and a Coordinator Agent that integrates these inputs to produce high-level behavior decisions and operational parameters. Rather than replacing existing AV modules, PACE-ADS complements them by operating at the behavioral level, delegating low-level control to native AV systems. This separation enables closed-loop adaptation and supports integration across diverse platforms. We evaluate PACE-ADS in simulation across varied scenarios involving traffic lights, pedestrians, work zones, and car following. Results show that PACE-ADS adapts driving styles to occupant states, improves ride comfort, and enables safe recovery from immobilization via autonomous reasoning or human guidance. Our findings highlight the promise of LLM-based frameworks for bridging the gap between machine autonomy and human-centered driving.
>
---
#### [new 006] Measuring and Minimizing Disturbance of Marine Animals to Underwater Vehicles
- **分类: cs.RO**

- **简介: 该论文属于水下动物行为研究任务，旨在解决水下车辆干扰动物行为的问题，通过理论与实验方法减少观测偏差。**

- **链接: [http://arxiv.org/pdf/2506.11335v1](http://arxiv.org/pdf/2506.11335v1)**

> **作者:** Levi Cai; Youenn Jézéquel; T. Aran Mooney; Yogesh Girdhar
>
> **备注:** Accepted to ISER 2025
>
> **摘要:** Do fish respond to the presence of underwater vehicles, potentially biasing our estimates about them? If so, are there strategies to measure and mitigate this response? This work provides a theoretical and practical framework towards bias-free estimation of animal behavior from underwater vehicle observations. We also provide preliminary results from the field in coral reef environments to address these questions.
>
---
#### [new 007] Control Architecture and Design for a Multi-robotic Visual Servoing System in Automated Manufacturing Environment
- **分类: cs.RO; cs.CV; cs.SY; eess.SY; 93C85 (Primary), 93B52 (Secondary)**

- **简介: 该论文属于多机器人视觉伺服控制任务，旨在解决制造环境中的不确定性问题，提出多机器人控制架构和相机位置优化算法以提高精度。**

- **链接: [http://arxiv.org/pdf/2506.11387v1](http://arxiv.org/pdf/2506.11387v1)**

> **作者:** Rongfei Li
>
> **备注:** 272 pages, 171 figures, PhD dissertation, University of California, Davis, 2025. To be published in ProQuest ETD
>
> **摘要:** The use of robotic technology has drastically increased in manufacturing in the 21st century. But by utilizing their sensory cues, humans still outperform machines, especially in micro scale manufacturing, which requires high-precision robot manipulators. These sensory cues naturally compensate for high levels of uncertainties that exist in the manufacturing environment. Uncertainties in performing manufacturing tasks may come from measurement noise, model inaccuracy, joint compliance (e.g., elasticity), etc. Although advanced metrology sensors and high precision microprocessors, which are utilized in modern robots, have compensated for many structural and dynamic errors in robot positioning, a well-designed control algorithm still works as a comparable and cheaper alternative to reduce uncertainties in automated manufacturing. Our work illustrates that a multi-robot control system that simulates the positioning process for fastening and unfastening applications can reduce various uncertainties, which may occur in this process, to a great extent. In addition, most research papers in visual servoing mainly focus on developing control and observation architectures in various scenarios, but few have discussed the importance of the camera's location in the configuration. In a manufacturing environment, the quality of camera estimations may vary significantly from one observation location to another, as the combined effects of environmental conditions result in different noise levels of a single image shot at different locations. Therefore, in this paper, we also propose a novel algorithm for the camera's moving policy so that it explores the camera workspace and searches for the optimal location where the image noise level is minimized.
>
---
#### [new 008] Poutine: Vision-Language-Trajectory Pre-Training and Reinforcement Learning Post-Training Enable Robust End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Poutine模型，用于端到端自动驾驶任务，解决长尾场景下的驾驶问题。通过视觉-语言-轨迹预训练和强化学习微调，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11234v1](http://arxiv.org/pdf/2506.11234v1)**

> **作者:** Luke Rowe; Rodrigue de Schaetzen; Roger Girgis; Christopher Pal; Liam Paull
>
> **摘要:** We present Poutine, a 3B-parameter vision-language model (VLM) tailored for end-to-end autonomous driving in long-tail driving scenarios. Poutine is trained in two stages. To obtain strong base driving capabilities, we train Poutine-Base in a self-supervised vision-language-trajectory (VLT) next-token prediction fashion on 83 hours of CoVLA nominal driving and 11 hours of Waymo long-tail driving. Accompanying language annotations are auto-generated with a 72B-parameter VLM. Poutine is obtained by fine-tuning Poutine-Base with Group Relative Policy Optimization (GRPO) using less than 500 preference-labeled frames from the Waymo validation set. We show that both VLT pretraining and RL fine-tuning are critical to attain strong driving performance in the long-tail. Poutine-Base achieves a rater-feedback score (RFS) of 8.12 on the validation set, nearly matching Waymo's expert ground-truth RFS. The final Poutine model achieves an RFS of 7.99 on the official Waymo test set, placing 1st in the 2025 Waymo Vision-Based End-to-End Driving Challenge by a significant margin. These results highlight the promise of scalable VLT pre-training and lightweight RL fine-tuning to enable robust and generalizable autonomy.
>
---
#### [new 009] The Space Between Us: A Methodological Framework for Researching Bonding and Proxemics in Situated Group-Agent Interactions
- **分类: cs.RO; cs.HC; stat.ME**

- **简介: 该论文属于人机交互领域，旨在解决群体与智能体互动中的空间和社会动态问题。通过结合主观报告和空间追踪，提出了一种多方法框架，并开发了开源工具包。**

- **链接: [http://arxiv.org/pdf/2506.11829v1](http://arxiv.org/pdf/2506.11829v1)**

> **作者:** Ana Müller; Anja Richert
>
> **备注:** Accepted for presentation at the Workshop on Advancing Group Understanding and Robots' Adaptive Behavior (GROUND), held at the Intelligent Autonomous Systems (IAS) Conference 2025, Genoa, Italy
>
> **摘要:** This paper introduces a multimethod framework for studying spatial and social dynamics in real-world group-agent interactions with socially interactive agents. Drawing on proxemics and bonding theories, the method combines subjective self-reports and objective spatial tracking. Applied in two field studies in a museum (N = 187) with a robot and a virtual agent, the paper addresses the challenges in aligning human perception and behavior. We focus on presenting an open source, scalable, and field-tested toolkit for future studies.
>
---
#### [new 010] Robot Context Protocol (RCP): A Runtime-Agnostic Interface for Agent-Aware Robot Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RCP协议，解决机器人与多智能体系统交互问题，设计轻量、跨平台通信接口，提升系统兼容性与安全性。**

- **链接: [http://arxiv.org/pdf/2506.11650v1](http://arxiv.org/pdf/2506.11650v1)**

> **作者:** Lambert Lee; Joshua Lau
>
> **摘要:** The Robot Context Protocol (RCP) is a lightweight, middleware-agnostic communication protocol designed to simplify the complexity of robotic systems and enable seamless interaction between robots, users, and autonomous agents. RCP provides a unified and semantically meaningful interface that decouples client-facing operations from backend implementations, supporting a wide range of deployment environments including physical robots, cloud-based orchestrators, and simulated platforms. Built on HTTP and WebSocket transport layers, the protocol defines a schema-driven message format with structured operations such as read, write, execute, and subscribe. It integrates features such as runtime introspection, asynchronous feedback, multi-tenant namespace isolation, and strict type validation to ensure robustness, scalability, and security. The architecture, message structure, interface model, and adapter-based backend integration strategy of RCP are described, along with deployment practices and applicability across industries including manufacturing, logistics, and healthcare. RCP enables intelligent, resilient, and safe robotic operations in complex, multi-agent ecosystems.
>
---
#### [new 011] Auditory-Tactile Congruence for Synthesis of Adaptive Pain Expressions in RoboPatients
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于医疗仿真任务，旨在提升机器人患者在触觉与听觉反馈中对疼痛表达的准确性。通过分析触觉与声音的匹配度，优化临床培训效果。**

- **链接: [http://arxiv.org/pdf/2506.11827v1](http://arxiv.org/pdf/2506.11827v1)**

> **作者:** Saitarun Nadipineni; Chapa Sirithunge; Yue Xie; Fumiya Iida; Thilina Dulantha Lalitharatne
>
> **备注:** 17 pages, 9 figures, journal
>
> **摘要:** Misdiagnosis can lead to delayed treatments and harm. Robotic patients offer a controlled way to train and evaluate clinicians in rare, subtle, or complex cases, reducing diagnostic errors. We present RoboPatient, a medical robotic simulator aimed at multimodal pain synthesis based on haptic and auditory feedback during palpation-based training scenarios. The robopatient functions as an adaptive intermediary, capable of synthesizing plausible pain expressions vocal and facial in response to tactile stimuli generated during palpation. Using an abdominal phantom, robopatient captures and processes haptic input via an internal palpation-to-pain mapping model. To evaluate perceptual congruence between palpation and the corresponding auditory output, we conducted a study involving 7680 trials across 20 participants, where they evaluated pain intensity through sound. Results show that amplitude and pitch significantly influence agreement with the robot's pain expressions, irrespective of pain sounds. Stronger palpation forces elicited stronger agreement, aligning with psychophysical patterns. The study revealed two key dimensions: pitch and amplitude are central to how people perceive pain sounds, with pitch being the most influential cue. These acoustic features shape how well the sound matches the applied force during palpation, impacting perceived realism. This approach lays the groundwork for high-fidelity robotic patients in clinical education and diagnostic simulation.
>
---
#### [new 012] mimic-one: a Scalable Model Recipe for General Purpose Robot Dexterity
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在提升机器人手的灵巧性。通过设计硬件和学习算法，实现高效、自纠正的控制，解决复杂操作中的精度与适应性问题。**

- **链接: [http://arxiv.org/pdf/2506.11916v1](http://arxiv.org/pdf/2506.11916v1)**

> **作者:** Elvis Nava; Victoriano Montesinos; Erik Bauer; Benedek Forrai; Jonas Pai; Stefan Weirich; Stephan-Daniel Gravert; Philipp Wand; Stephan Polinski; Benjamin F. Grewe; Robert K. Katzschmann
>
> **摘要:** We present a diffusion-based model recipe for real-world control of a highly dexterous humanoid robotic hand, designed for sample-efficient learning and smooth fine-motor action inference. Our system features a newly designed 16-DoF tendon-driven hand, equipped with wide angle wrist cameras and mounted on a Franka Emika Panda arm. We develop a versatile teleoperation pipeline and data collection protocol using both glove-based and VR interfaces, enabling high-quality data collection across diverse tasks such as pick and place, item sorting and assembly insertion. Leveraging high-frequency generative control, we train end-to-end policies from raw sensory inputs, enabling smooth, self-correcting motions in complex manipulation scenarios. Real-world evaluations demonstrate up to 93.3% out of distribution success rates, with up to a +33.3% performance boost due to emergent self-correcting behaviors, while also revealing scaling trends in policy performance. Our results advance the state-of-the-art in dexterous robotic manipulation through a fully integrated, practical approach to hardware, learning, and real-world deployment.
>
---
#### [new 013] Robust Optimal Task Planning to Maximize Battery Life
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人任务规划任务，旨在延长电池寿命并确保任务完成。通过优化方法解决双线性问题，提升系统鲁棒性与效率。**

- **链接: [http://arxiv.org/pdf/2506.11264v1](http://arxiv.org/pdf/2506.11264v1)**

> **作者:** Jiachen Li; Chu Jian; Feiyang Zhao; Shihao Li; Wei Li; Dongmei Chen
>
> **摘要:** This paper proposes a control-oriented optimization platform for autonomous mobile robots (AMRs), focusing on extending battery life while ensuring task completion. The requirement of fast AMR task planning while maintaining minimum battery state of charge, thus maximizing the battery life, renders a bilinear optimization problem. McCormick envelop technique is proposed to linearize the bilinear term. A novel planning algorithm with relaxed constraints is also developed to handle parameter uncertainties robustly with high efficiency ensured. Simulation results are provided to demonstrate the utility of the proposed methods in reducing battery degradation while satisfying task completion requirements.
>
---
#### [new 014] Foundation Models in Autonomous Driving: A Survey on Scenario Generation and Scenario Analysis
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶领域，解决场景生成与分析问题，探讨基础模型在其中的应用，提出分类体系并评估相关技术。**

- **链接: [http://arxiv.org/pdf/2506.11526v1](http://arxiv.org/pdf/2506.11526v1)**

> **作者:** Yuan Gao; Mattia Piccinini; Yuchen Zhang; Dingrui Wang; Korbinian Moller; Roberto Brusnicki; Baha Zarrouki; Alessio Gambi; Jan Frederik Totz; Kai Storms; Steven Peters; Andrea Stocco; Bassam Alrifaee; Marco Pavone; Johannes Betz
>
> **摘要:** For autonomous vehicles, safe navigation in complex environments depends on handling a broad range of diverse and rare driving scenarios. Simulation- and scenario-based testing have emerged as key approaches to development and validation of autonomous driving systems. Traditional scenario generation relies on rule-based systems, knowledge-driven models, and data-driven synthesis, often producing limited diversity and unrealistic safety-critical cases. With the emergence of foundation models, which represent a new generation of pre-trained, general-purpose AI models, developers can process heterogeneous inputs (e.g., natural language, sensor data, HD maps, and control actions), enabling the synthesis and interpretation of complex driving scenarios. In this paper, we conduct a survey about the application of foundation models for scenario generation and scenario analysis in autonomous driving (as of May 2025). Our survey presents a unified taxonomy that includes large language models, vision-language models, multimodal large language models, diffusion models, and world models for the generation and analysis of autonomous driving scenarios. In addition, we review the methodologies, open-source datasets, simulation platforms, and benchmark challenges, and we examine the evaluation metrics tailored explicitly to scenario generation and analysis. Finally, the survey concludes by highlighting the open challenges and research questions, and outlining promising future research directions. All reviewed papers are listed in a continuously maintained repository, which contains supplementary materials and is available at https://github.com/TUM-AVS/FM-for-Scenario-Generation-Analysis.
>
---
#### [new 015] Demonstration Sidetracks: Categorizing Systematic Non-Optimality in Human Demonstrations
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人学习任务，研究人类示范中的系统性非最优行为，提出“示范偏轨”概念，并通过实验分析其类型与分布，以提升LfD算法性能。**

- **链接: [http://arxiv.org/pdf/2506.11262v1](http://arxiv.org/pdf/2506.11262v1)**

> **作者:** Shijie Fang; Hang Yu; Qidi Fang; Reuben M. Aronson; Elaine S. Short
>
> **摘要:** Learning from Demonstration (LfD) is a popular approach for robots to acquire new skills, but most LfD methods suffer from imperfections in human demonstrations. Prior work typically treats these suboptimalities as random noise. In this paper we study non-optimal behaviors in non-expert demonstrations and show that they are systematic, forming what we call demonstration sidetracks. Using a public space study with 40 participants performing a long-horizon robot task, we recreated the setup in simulation and annotated all demonstrations. We identify four types of sidetracks (Exploration, Mistake, Alignment, Pause) and one control pattern (one-dimension control). Sidetracks appear frequently across participants, and their temporal and spatial distribution is tied to task context. We also find that users' control patterns depend on the control interface. These insights point to the need for better models of suboptimal demonstrations to improve LfD algorithms and bridge the gap between lab training and real-world deployment. All demonstrations, infrastructure, and annotations are available at https://github.com/AABL-Lab/Human-Demonstration-Sidetracks.
>
---
#### [new 016] Sensor Model Identification via Simultaneous Model Selection and State Variable Determination
- **分类: cs.RO; cs.IT; cs.NA; cs.SY; eess.SY; math.IT; math.NA**

- **简介: 该论文属于传感器模型识别任务，旨在解决未知测量数据的传感器模型选择问题，通过模型选择与状态变量确定实现可靠集成。**

- **链接: [http://arxiv.org/pdf/2506.11263v1](http://arxiv.org/pdf/2506.11263v1)**

> **作者:** Christian Brommer; Alessandro Fornasier; Jan Steinbrener; Stephan Weiss
>
> **摘要:** We present a method for the unattended gray-box identification of sensor models commonly used by localization algorithms in the field of robotics. The objective is to determine the most likely sensor model for a time series of unknown measurement data, given an extendable catalog of predefined sensor models. Sensor model definitions may require states for rigid-body calibrations and dedicated reference frames to replicate a measurement based on the robot's localization state. A health metric is introduced, which verifies the outcome of the selection process in order to detect false positives and facilitate reliable decision-making. In a second stage, an initial guess for identified calibration states is generated, and the necessity of sensor world reference frames is evaluated. The identified sensor model with its parameter information is then used to parameterize and initialize a state estimation application, thus ensuring a more accurate and robust integration of new sensor elements. This method is helpful for inexperienced users who want to identify the source and type of a measurement, sensor calibrations, or sensor reference frames. It will also be important in the field of modular multi-agent scenarios and modularized robotic platforms that are augmented by sensor modalities during runtime. Overall, this work aims to provide a simplified integration of sensor modalities to downstream applications and circumvent common pitfalls in the usage and development of localization approaches.
>
---
#### [new 017] Palpation Alters Auditory Pain Expressions with Gender-Specific Variations in Robopatients
- **分类: cs.RO**

- **简介: 该论文属于医疗仿真任务，旨在解决robopatients在触诊时生成准确听觉疼痛反应的问题。通过强化学习优化触诊力度与疼痛声音的关联，提升训练效果。**

- **链接: [http://arxiv.org/pdf/2506.11906v1](http://arxiv.org/pdf/2506.11906v1)**

> **作者:** Chapa Sirithunge; Yue Xie; Saitarun Nadipineni; Fumiya Iida; Thilina Dulantha Lalitharatne
>
> **备注:** 11 pages, 9 figures, journal
>
> **摘要:** Diagnostic errors remain a major cause of preventable deaths, particularly in resource-limited regions. Medical training simulators, including robopatients, play a vital role in reducing these errors by mimicking real patients for procedural training such as palpation. However, generating multimodal feedback, especially auditory pain expressions, remains challenging due to the complex relationship between palpation behavior and sound. The high-dimensional nature of pain sounds makes exploration challenging with conventional methods. This study introduces a novel experimental paradigm for pain expressivity in robopatients where they dynamically generate auditory pain expressions in response to palpation force, by co-optimizing human feedback using machine learning. Using Proximal Policy Optimization (PPO), a reinforcement learning (RL) technique optimized for continuous adaptation, our robot iteratively refines pain sounds based on real-time human feedback. This robot initializes randomized pain responses to palpation forces, and the RL agent learns to adjust these sounds to align with human preferences. The results demonstrated that the system adapts to an individual's palpation forces and sound preferences and captures a broad spectrum of pain intensity, from mild discomfort to acute distress, through RL-guided exploration of the auditory pain space. The study further showed that pain sound perception exhibits saturation at lower forces with gender specific thresholds. These findings highlight the system's potential to enhance abdominal palpation training by offering a controllable and immersive simulation platform.
>
---
#### [new 018] Robotic System for Chemical Experiment Automation with Dual Demonstration of End-effector and Jig Operations
- **分类: cs.RO**

- **简介: 该论文属于化学实验自动化任务，旨在解决机器人与实验夹具同步控制问题。通过双示教方式实现机器人与夹具操作的协同，提升实验可重复性和效率。**

- **链接: [http://arxiv.org/pdf/2506.11384v1](http://arxiv.org/pdf/2506.11384v1)**

> **作者:** Hikaru Sasaki; Naoto Komeno; Takumi Hachimine; Kei Takahashi; Yu-ya Ohnishi; Tetsunori Sugawara; Araki Wakiuchi; Miho Hatanaka; Tomoyuki Miyao; Hiroharu Ajiro; Mikiya Fujii; Takamitsu Matsubara
>
> **摘要:** While robotic automation has demonstrated remarkable performance, such as executing hundreds of experiments continuously over several days, it is challenging to design a program that synchronizes the robot's movements with the experimental jigs to conduct an experiment. We propose a concept that enables the automation of experiments by utilizing dual demonstrations of robot motions and jig operations by chemists in an experimental environment constructed to be controlled by a robot. To verify this concept, we developed a chemical-experiment-automation system consisting of jigs to assist the robot in experiments, a motion-demonstration interface, a jig-control interface, and a mobile manipulator. We validate the concept through polymer-synthesis experiments, focusing on critical liquid-handling tasks such as pipetting and dilution. The experimental results indicate high reproducibility of the demonstrated motions and robust task-success rates. This comprehensive concept not only simplifies the robot programming process for chemists but also provides a flexible and efficient solution to accommodate a wide range of experimental conditions, contributing significantly to the field of chemical experiment automation.
>
---
#### [new 019] CIRO7.2: A Material Network with Circularity of -7.2 and Reinforcement-Learning-Controlled Robotic Disassembler
- **分类: cs.RO; cs.CY**

- **简介: 该论文属于循环经济任务，旨在解决废弃物管理问题。通过构建材料网络和强化学习控制的拆解机器人，提升资源循环利用率。**

- **链接: [http://arxiv.org/pdf/2506.11748v1](http://arxiv.org/pdf/2506.11748v1)**

> **作者:** Federico Zocco; Monica Malvezzi
>
> **备注:** To be submitted
>
> **摘要:** The competition over natural reserves of minerals is expected to increase in part because of the linear-economy paradigm based on take-make-dispose. Simultaneously, the linear economy considers end-of-use products as waste rather than as a resource, which results in large volumes of waste whose management remains an unsolved problem. Since a transition to a circular economy can mitigate these open issues, in this paper we begin by enhancing the notion of circularity based on compartmental dynamical thermodynamics, namely, $\lambda$, and then, we model a thermodynamical material network processing a batch of 2 solid materials of criticality coefficients of 0.1 and 0.95, with a robotic disassembler compartment controlled via reinforcement learning (RL), and processing 2-7 kg of materials. Subsequently, we focused on the design of the robotic disassembler compartment using state-of-the-art RL algorithms and assessing the algorithm performance with respect to $\lambda$ (Fig. 1). The highest circularity is -2.1 achieved in the case of disassembling 2 parts of 1 kg each, whereas it reduces to -7.2 in the case of disassembling 4 parts of 1 kg each contained inside a chassis of 3 kg. Finally, a sensitivity analysis highlighted that the impact on $\lambda$ of the performance of an RL controller has a positive correlation with the quantity and the criticality of the materials to be disassembled. This work also gives the principles of the emerging research fields indicated as circular intelligence and robotics (CIRO). Source code is publicly available.
>
---
#### [new 020] SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人模仿学习任务，解决政策执行速度受限问题，提出SAIL系统实现更快执行。**

- **链接: [http://arxiv.org/pdf/2506.11948v1](http://arxiv.org/pdf/2506.11948v1)**

> **作者:** Nadun Ranawaka Arachchige; Zhenyang Chen; Wonsuhk Jung; Woo Chul Shin; Rohan Bansal; Pierre Barroso; Yu Hang He; Yingyang Celine Lin; Benjamin Joffe; Shreyas Kousik; Danfei Xu
>
> **备注:** The first two authors contributed equally
>
> **摘要:** Offline Imitation Learning (IL) methods such as Behavior Cloning are effective at acquiring complex robotic manipulation skills. However, existing IL-trained policies are confined to executing the task at the same speed as shown in demonstration data. This limits the task throughput of a robotic system, a critical requirement for applications such as industrial automation. In this paper, we introduce and formalize the novel problem of enabling faster-than-demonstration execution of visuomotor policies and identify fundamental challenges in robot dynamics and state-action distribution shifts. We instantiate the key insights as SAIL (Speed Adaptation for Imitation Learning), a full-stack system integrating four tightly-connected components: (1) a consistency-preserving action inference algorithm for smooth motion at high speed, (2) high-fidelity tracking of controller-invariant motion targets, (3) adaptive speed modulation that dynamically adjusts execution speed based on motion complexity, and (4) action scheduling to handle real-world system latencies. Experiments on 12 tasks across simulation and two real, distinct robot platforms show that SAIL achieves up to a 4x speedup over demonstration speed in simulation and up to 3.2x speedup in the real world. Additional detail is available at https://nadunranawaka1.github.io/sail-policy
>
---
#### [new 021] Gondola: Grounded Vision Language Planning for Generalizable Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决语言指令下物体和环境泛化问题。提出Gondola模型，结合多视角图像与历史计划生成精准动作规划。**

- **链接: [http://arxiv.org/pdf/2506.11261v1](http://arxiv.org/pdf/2506.11261v1)**

> **作者:** Shizhe Chen; Ricardo Garcia; Paul Pacaud; Cordelia Schmid
>
> **摘要:** Robotic manipulation faces a significant challenge in generalizing across unseen objects, environments and tasks specified by diverse language instructions. To improve generalization capabilities, recent research has incorporated large language models (LLMs) for planning and action execution. While promising, these methods often fall short in generating grounded plans in visual environments. Although efforts have been made to perform visual instructional tuning on LLMs for robotic manipulation, existing methods are typically constrained by single-view image input and struggle with precise object grounding. In this work, we introduce Gondola, a novel grounded vision-language planning model based on LLMs for generalizable robotic manipulation. Gondola takes multi-view images and history plans to produce the next action plan with interleaved texts and segmentation masks of target objects and locations. To support the training of Gondola, we construct three types of datasets using the RLBench simulator, namely robot grounded planning, multi-view referring expression and pseudo long-horizon task datasets. Gondola outperforms the state-of-the-art LLM-based method across all four generalization levels of the GemBench dataset, including novel placements, rigid objects, articulated objects and long-horizon tasks.
>
---
#### [new 022] Scheduling Agile Earth Observation Satellites with Onboard Processing and Real-Time Monitoring
- **分类: cs.NI; cs.RO**

- **简介: 该论文属于卫星调度任务，解决AEOSs的观测顺序优化问题，通过引入优先级指标和局部搜索算法提升观测效率与信息实时性。**

- **链接: [http://arxiv.org/pdf/2506.11556v1](http://arxiv.org/pdf/2506.11556v1)**

> **作者:** Antonio M. Mercado-Martínez; Beatriz Soret; Antonio Jurado-Navas
>
> **备注:** This paper has been submitted to GLOBECOM 2025
>
> **摘要:** The emergence of Agile Earth Observation Satellites (AEOSs) has marked a significant turning point in the field of Earth Observation (EO), offering enhanced flexibility in data acquisition. Concurrently, advancements in onboard satellite computing and communication technologies have greatly enhanced data compression efficiency, reducing network latency and congestion while supporting near real-time information delivery. In this paper, we address the Agile Earth Observation Satellite Scheduling Problem (AEOSSP), which involves determining the optimal sequence of target observations to maximize overall observation profit. Our approach integrates onboard data processing for real-time remote monitoring into the multi-satellite optimization problem. To this end, we define a set of priority indicators and develop a constructive heuristic method, further enhanced with a Local Search (LS) strategy. The results show that the proposed algorithm provides high-quality information by increasing the resolution of the collected frames by up to 10% on average, while reducing the variance in the monitoring frequency of the targets within the instance by up to 83%, ensuring more up-to-date information across the entire set compared to a First-In First-Out (FIFO) method.
>
---
#### [new 023] A Step-by-Step Guide to Creating a Robust Autonomous Drone Testing Pipeline
- **分类: cs.SE; cs.RO**

- **简介: 该论文属于无人机测试任务，旨在解决自主无人机系统验证问题，通过构建测试流程实现安全可靠部署。**

- **链接: [http://arxiv.org/pdf/2506.11400v1](http://arxiv.org/pdf/2506.11400v1)**

> **作者:** Yupeng Jiang; Yao Deng; Sebastian Schroder; Linfeng Liang; Suhaas Gambhir; Alice James; Avishkar Seth; James Pirrie; Yihao Zhang; Xi Zheng
>
> **摘要:** Autonomous drones are rapidly reshaping industries ranging from aerial delivery and infrastructure inspection to environmental monitoring and disaster response. Ensuring the safety, reliability, and efficiency of these systems is paramount as they transition from research prototypes to mission-critical platforms. This paper presents a step-by-step guide to establishing a robust autonomous drone testing pipeline, covering each critical stage: Software-in-the-Loop (SIL) Simulation Testing, Hardware-in-the-Loop (HIL) Testing, Controlled Real-World Testing, and In-Field Testing. Using practical examples, including the marker-based autonomous landing system, we demonstrate how to systematically verify drone system behaviors, identify integration issues, and optimize performance. Furthermore, we highlight emerging trends shaping the future of drone testing, including the integration of Neurosymbolic and LLMs, creating co-simulation environments, and Digital Twin-enabled simulation-based testing techniques. By following this pipeline, developers and researchers can achieve comprehensive validation, minimize deployment risks, and prepare autonomous drones for safe and reliable real-world operations.
>
---
#### [new 024] FocalAD: Local Motion Planning for End-to-End Autonomous Driving
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决局部交互忽略导致的规划不可靠问题。提出FocalAD框架，通过关注关键局部邻居提升规划准确性。**

- **链接: [http://arxiv.org/pdf/2506.11419v1](http://arxiv.org/pdf/2506.11419v1)**

> **作者:** Bin Sun; Boao Zhang; Jiayi Lu; Xinjie Feng; Jiachen Shang; Rui Cao; Mengchao Zheng; Chuanye Wang; Shichun Yang; Yaoguang Cao; Ziying Song
>
> **摘要:** In end-to-end autonomous driving,the motion prediction plays a pivotal role in ego-vehicle planning. However, existing methods often rely on globally aggregated motion features, ignoring the fact that planning decisions are primarily influenced by a small number of locally interacting agents. Failing to attend to these critical local interactions can obscure potential risks and undermine planning reliability. In this work, we propose FocalAD, a novel end-to-end autonomous driving framework that focuses on critical local neighbors and refines planning by enhancing local motion representations. Specifically, FocalAD comprises two core modules: the Ego-Local-Agents Interactor (ELAI) and the Focal-Local-Agents Loss (FLA Loss). ELAI conducts a graph-based ego-centric interaction representation that captures motion dynamics with local neighbors to enhance both ego planning and agent motion queries. FLA Loss increases the weights of decision-critical neighboring agents, guiding the model to prioritize those more relevant to planning. Extensive experiments show that FocalAD outperforms existing state-of-the-art methods on the open-loop nuScenes datasets and closed-loop Bench2Drive benchmark. Notably, on the robustness-focused Adv-nuScenes dataset, FocalAD achieves even greater improvements, reducing the average colilision rate by 41.9% compared to DiffusionDrive and by 15.6% compared to SparseDrive.
>
---
#### [new 025] Linearly Solving Robust Rotation Estimation
- **分类: cs.CV; cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于旋转估计任务，解决非线性优化难题，通过线性方法和投票机制实现鲁棒、快速的旋转估计。**

- **链接: [http://arxiv.org/pdf/2506.11547v1](http://arxiv.org/pdf/2506.11547v1)**

> **作者:** Yinlong Liu; Tianyu Huang; Zhi-Xin Yang
>
> **备注:** 23 pages, 18 figures
>
> **摘要:** Rotation estimation plays a fundamental role in computer vision and robot tasks, and extremely robust rotation estimation is significantly useful for safety-critical applications. Typically, estimating a rotation is considered a non-linear and non-convex optimization problem that requires careful design. However, in this paper, we provide some new perspectives that solving a rotation estimation problem can be reformulated as solving a linear model fitting problem without dropping any constraints and without introducing any singularities. In addition, we explore the dual structure of a rotation motion, revealing that it can be represented as a great circle on a quaternion sphere surface. Accordingly, we propose an easily understandable voting-based method to solve rotation estimation. The proposed method exhibits exceptional robustness to noise and outliers and can be computed in parallel with graphics processing units (GPUs) effortlessly. Particularly, leveraging the power of GPUs, the proposed method can obtain a satisfactory rotation solution for large-scale($10^6$) and severely corrupted (99$\%$ outlier ratio) rotation estimation problems under 0.5 seconds. Furthermore, to validate our theoretical framework and demonstrate the superiority of our proposed method, we conduct controlled experiments and real-world dataset experiments. These experiments provide compelling evidence supporting the effectiveness and robustness of our approach in solving rotation estimation problems.
>
---
## 更新

#### [replaced 001] Control Industrial Automation System with Large Language Model Agents
- **分类: eess.SY; cs.AI; cs.HC; cs.MA; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2409.18009v2](http://arxiv.org/pdf/2409.18009v2)**

> **作者:** Yuchen Xia; Nasser Jazdi; Jize Zhang; Chaitanya Shah; Michael Weyrich
>
> **备注:** Pre-print accepted at 30th IEEE ETFA 2025
>
> **摘要:** Traditional industrial automation systems require specialized expertise to operate and complex reprogramming to adapt to new processes. Large language models offer the intelligence to make them more flexible and easier to use. However, LLMs' application in industrial settings is underexplored. This paper introduces a framework for integrating LLMs to achieve end-to-end control of industrial automation systems. At the core of the framework are an agent system designed for industrial tasks, a structured prompting method, and an event-driven information modeling mechanism that provides real-time data for LLM inference. The framework supplies LLMs with real-time events on different context semantic levels, allowing them to interpret the information, generate production plans, and control operations on the automation system. It also supports structured dataset creation for fine-tuning on this downstream application of LLMs. Our contribution includes a formal system design, proof-of-concept implementation, and a method for generating task-specific datasets for LLM fine-tuning and testing. This approach enables a more adaptive automation system that can respond to spontaneous events, while allowing easier operation and configuration through natural language for more intuitive human-machine interaction. We provide demo videos and detailed data on GitHub: https://github.com/YuchenXia/LLM4IAS.
>
---
#### [replaced 002] ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.22094v4](http://arxiv.org/pdf/2505.22094v4)**

> **作者:** Tonghe Zhang; Chao Yu; Sichang Su; Yu Wang
>
> **备注:** 30 pages, 13 figures, 10 tables
>
> **摘要:** We propose ReinFlow, a simple yet effective online reinforcement learning (RL) framework that fine-tunes a family of flow matching policies for continuous robotic control. Derived from rigorous RL theory, ReinFlow injects learnable noise into a flow policy's deterministic path, converting the flow into a discrete-time Markov Process for exact and straightforward likelihood computation. This conversion facilitates exploration and ensures training stability, enabling ReinFlow to fine-tune diverse flow model variants, including Rectified Flow [35] and Shortcut Models [19], particularly at very few or even one denoising step. We benchmark ReinFlow in representative locomotion and manipulation tasks, including long-horizon planning with visual input and sparse reward. The episode reward of Rectified Flow policies obtained an average net growth of 135.36% after fine-tuning in challenging legged locomotion tasks while saving denoising steps and 82.63% of wall time compared to state-of-the-art diffusion RL fine-tuning method DPPO [43]. The success rate of the Shortcut Model policies in state and visual manipulation tasks achieved an average net increase of 40.34% after fine-tuning with ReinFlow at four or even one denoising step, whose performance is comparable to fine-tuned DDIM policies while saving computation time for an average of 23.20%. Project webpage: https://reinflow.github.io/
>
---
#### [replaced 003] A Soft Robotic Module with Pneumatic Actuation and Enhanced Controllability Using a Shape Memory Alloy Wire
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.05741v2](http://arxiv.org/pdf/2506.05741v2)**

> **作者:** Mohammadnavid Golchin
>
> **摘要:** In this paper, a compressed air-actuated soft robotic module was developed by incorporating a shape memory alloy (SMA) wire into its structure to achieve the desired bending angle with greater precision. First, a fiber-reinforced bending module with a strain-limiting layer made of polypropylene was fabricated. The SMA wire was then placed in a silicon matrix, which was used as a new strain-limiting layer. A simple closed-loop control algorithm was used to regulate the bending angle of the soft robot within its workspace. A camera was utilized to measure the angular changes in the vertical plane. Different angles, ranging from 0 to 65 degrees, were covered to evaluate the performance of the module and the bending angle control algorithm. The experimental tests demonstrate that using the SMA wire results in more precise control of bending in the vertical plane. In addition, it is possible to bend more with less working pressure. The error range was reduced from an average of 5 degrees to 2 degrees, and the rise time was reduced from an average of 19 seconds to 3 seconds.
>
---
#### [replaced 004] RationalVLA: A Rational Vision-Language-Action Model with Dual System
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.10826v2](http://arxiv.org/pdf/2506.10826v2)**

> **作者:** Wenxuan Song; Jiayi Chen; Wenxue Li; Xu He; Han Zhao; Can Cui; Pengxiang Ding Shiyan Su; Feilong Tang; Xuelian Cheng; Donglin Wang; Zongyuan Ge; Xinhu Zheng; Zhe Liu; Hesheng Wang; Haoang Li
>
> **备注:** 14 pages
>
> **摘要:** A fundamental requirement for real-world robotic deployment is the ability to understand and respond to natural language instructions. Existing language-conditioned manipulation tasks typically assume that instructions are perfectly aligned with the environment. This assumption limits robustness and generalization in realistic scenarios where instructions may be ambiguous, irrelevant, or infeasible. To address this problem, we introduce RAtional MAnipulation (RAMA), a new benchmark that challenges models with both unseen executable instructions and defective ones that should be rejected. In RAMA, we construct a dataset with over 14,000 samples, including diverse defective instructions spanning six dimensions: visual, physical, semantic, motion, safety, and out-of-context. We further propose the Rational Vision-Language-Action model (RationalVLA). It is a dual system for robotic arms that integrates the high-level vision-language model with the low-level manipulation policy by introducing learnable latent space embeddings. This design enables RationalVLA to reason over instructions, reject infeasible commands, and execute manipulation effectively. Experiments demonstrate that RationalVLA outperforms state-of-the-art baselines on RAMA by a 14.5% higher success rate and 0.94 average task length, while maintaining competitive performance on standard manipulation tasks. Real-world trials further validate its effectiveness and robustness in practical applications. Our project page is https://irpn-eai.github.io/RationalVLA.
>
---
#### [replaced 005] Real-time Seafloor Segmentation and Mapping
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.10750v2](http://arxiv.org/pdf/2504.10750v2)**

> **作者:** Michele Grimaldi; Nouf Alkaabi; Francesco Ruscio; Sebastian Realpe Rua; Rafael Garcia; Nuno Gracias
>
> **摘要:** Posidonia oceanica meadows are a species of seagrass highly dependent on rocks for their survival and conservation. In recent years, there has been a concerning global decline in this species, emphasizing the critical need for efficient monitoring and assessment tools. While deep learning-based semantic segmentation and visual automated monitoring systems have shown promise in a variety of applications, their performance in underwater environments remains challenging due to complex water conditions and limited datasets. This paper introduces a framework that combines machine learning and computer vision techniques to enable an autonomous underwater vehicle (AUV) to inspect the boundaries of Posidonia oceanica meadows autonomously. The framework incorporates an image segmentation module using an existing Mask R-CNN model and a strategy for Posidonia oceanica meadow boundary tracking. Furthermore, a new class dedicated to rocks is introduced to enhance the existing model, aiming to contribute to a comprehensive monitoring approach and provide a deeper understanding of the intricate interactions between the meadow and its surrounding environment. The image segmentation model is validated using real underwater images, while the overall inspection framework is evaluated in a realistic simulation environment, replicating actual monitoring scenarios with real underwater images. The results demonstrate that the proposed framework enables the AUV to autonomously accomplish the main tasks of underwater inspection and segmentation of rocks. Consequently, this work holds significant potential for the conservation and protection of marine environments, providing valuable insights into the status of Posidonia oceanica meadows and supporting targeted preservation efforts
>
---
#### [replaced 006] Interior Point Differential Dynamic Programming, Redux
- **分类: math.OC; cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.08278v3](http://arxiv.org/pdf/2504.08278v3)**

> **作者:** Ming Xu; Stephen Gould; Iman Shames
>
> **摘要:** We present IPDDP2, a structure-exploiting algorithm for solving discrete-time, finite-horizon optimal control problems (OCPs) with nonlinear constraints. Inequality constraints are handled using a primal-dual interior point formulation and step acceptance for equality constraints follows a line-search filter approach. The iterates of the algorithm are derived under the Differential Dynamic Programming (DDP) framework. A proof of local quadratic convergence of the IPDDP2 iterates is provided. Our numerical experiments evaluate IPDDP2 on over 500 OCPs derived from five different classes of robotic motion planning problems, three of which are contact-implicit trajectory optimisation problems. IPDDP2 demonstrates improvements in robustness against existing constrained DDP algorithms for contact-implicit planning, while being significantly faster than general-purpose solver IPOPT. We provide a full implementation of IPDDP2 in the Julia programming language.
>
---
#### [replaced 007] DiffTORI: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2402.05421v5](http://arxiv.org/pdf/2402.05421v5)**

> **作者:** Weikang Wan; Ziyu Wang; Yufei Wang; Zackory Erickson; David Held
>
> **备注:** NeurIPS 2024 (Spotlight)
>
> **摘要:** This paper introduces DiffTORI, which utilizes Differentiable Trajectory Optimization as the policy representation to generate actions for deep Reinforcement and Imitation learning. Trajectory optimization is a powerful and widely used algorithm in control, parameterized by a cost and a dynamics function. The key to our approach is to leverage the recent progress in differentiable trajectory optimization, which enables computing the gradients of the loss with respect to the parameters of trajectory optimization. As a result, the cost and dynamics functions of trajectory optimization can be learned end-to-end. DiffTORI addresses the ``objective mismatch'' issue of prior model-based RL algorithms, as the dynamics model in DiffTORI is learned to directly maximize task performance by differentiating the policy gradient loss through the trajectory optimization process. We further benchmark DiffTORI for imitation learning on standard robotic manipulation task suites with high-dimensional sensory observations and compare our method to feed-forward policy classes as well as Energy-Based Models (EBM) and Diffusion. Across 15 model-based RL tasks and 35 imitation learning tasks with high-dimensional image and point cloud inputs, DiffTORI outperforms prior state-of-the-art methods in both domains. Our code is available at https://github.com/wkwan7/DiffTORI.
>
---
#### [replaced 008] Extended Hybrid Zero Dynamics for Bipedal Walking of the Knee-less Robot SLIDER
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.01165v2](http://arxiv.org/pdf/2504.01165v2)**

> **作者:** Rui Zong; Martin Liang; Yuntian Fang; Ke Wang; Xiaoshuai Chen; Wei Chen; Petar Kormushev
>
> **备注:** accepted by CLAWAR 2025
>
> **摘要:** Knee-less bipedal robots like SLIDER have the advantage of ultra-lightweight legs and improved walking energy efficiency compared to traditional humanoid robots. In this paper, we firstly introduce an improved hardware design of the SLIDER bipedal robot with new line-feet and more optimized mass distribution that enables higher locomotion speeds. Secondly, we propose an extended Hybrid Zero Dynamics (eHZD) method, which can be applied to prismatic joint robots like SLIDER. The eHZD method is then used to generate a library of gaits with varying reference velocities in an offline way. Thirdly, a Guided Deep Reinforcement Learning (DRL) algorithm is proposed to use the pre-generated library to create walking control policies in real-time. This approach allows us to combine the advantages of both HZD (for generating stable gaits with a full-dynamics model) and DRL (for real-time adaptive gait generation). The experimental results show that this approach achieves 150% higher walking velocity than the previous MPC-based approach.
>
---
#### [replaced 009] DURA-CPS: A Multi-Role Orchestrator for Dependability Assurance in LLM-Enabled Cyber-Physical Systems
- **分类: cs.RO; cs.AI; cs.ET; cs.HC; cs.MA; C.3; C.4; D.2.4; D.4.6; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.06381v2](http://arxiv.org/pdf/2506.06381v2)**

> **作者:** Trisanth Srinivasan; Santosh Patapati; Himani Musku; Idhant Gode; Aditya Arora; Samvit Bhattacharya; Abubakr Nazriev; Sanika Hirave; Zaryab Kanjiani; Srinjoy Ghose
>
> **备注:** Accepted to the 55th Annual IEEE/IFIP International Conference on Dependable Systems and Networks Workshops (DSN-W)
>
> **摘要:** Cyber-Physical Systems (CPS) increasingly depend on advanced AI techniques to operate in critical applications. However, traditional verification and validation methods often struggle to handle the unpredictable and dynamic nature of AI components. In this paper, we introduce DURA-CPS, a novel framework that employs multi-role orchestration to automate the iterative assurance process for AI-powered CPS. By assigning specialized roles (e.g., safety monitoring, security assessment, fault injection, and recovery planning) to dedicated agents within a simulated environment, DURA-CPS continuously evaluates and refines AI behavior against a range of dependability requirements. We demonstrate the framework through a case study involving an autonomous vehicle navigating an intersection with an AI-based planner. Our results show that DURA-CPS effectively detects vulnerabilities, manages performance impacts, and supports adaptive recovery strategies, thereby offering a structured and extensible solution for rigorous V&V in safety- and security-critical systems.
>
---
#### [replaced 010] PhysNav-DG: A Novel Adaptive Framework for Robust VLM-Sensor Fusion in Navigation Applications
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.01881v3](http://arxiv.org/pdf/2505.01881v3)**

> **作者:** Trisanth Srinivasan; Santosh Patapati
>
> **备注:** Accepted at IEEE/CVF Computer Society Conference on Computer Vision and Pattern Recognition Workshops 2025 (CVPRW)
>
> **摘要:** Robust navigation in diverse environments and domains requires both accurate state estimation and transparent decision making. We present PhysNav-DG, a novel framework that integrates classical sensor fusion with the semantic power of vision-language models. Our dual-branch architecture predicts navigation actions from multi-sensor inputs while simultaneously generating detailed chain-of-thought explanations. A modified Adaptive Kalman Filter dynamically adjusts its noise parameters based on environmental context. It leverages several streams of raw sensor data along with semantic insights from models such as LLaMA 3.2 11B and BLIP-2. To evaluate our approach, we introduce the MD-NEX Benchmark, a novel multi-domain dataset that unifies indoor navigation, autonomous driving, and social navigation tasks with ground-truth actions and human-validated explanations. Extensive experiments and ablations show that PhysNav-DG improves navigation success rates by over 20% and achieves high efficiency, with explanations that are both highly grounded and clear. This work connects high-level semantic reasoning and geometric planning for safer and more trustworthy autonomous systems.
>
---
#### [replaced 011] Autonomous Robotic Radio Source Localization via a Novel Gaussian Mixture Filtering Approach
- **分类: cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2503.10349v3](http://arxiv.org/pdf/2503.10349v3)**

> **作者:** Sukkeun Kim; Sangwoo Moon; Ivan Petrunin; Hyo-Sang Shin; Shehryar Khattak
>
> **摘要:** This study proposes a new Gaussian Mixture Filter (GMF) to improve the estimation performance for the autonomous robotic radio signal source search and localization problem in unknown environments. The proposed filter is first tested with a benchmark numerical problem to validate the performance with other state-of-the-practice approaches such as Particle Filter (PF) and Particle Gaussian Mixture (PGM) filters. Then the proposed approach is tested and compared against PF and PGM filters in real-world robotic field experiments to validate its impact for real-world applications. The considered real-world scenarios have partial observability with the range-only measurement and uncertainty with the measurement model. The results show that the proposed filter can handle this partial observability effectively whilst showing improved performance compared to PF, reducing the computation requirements while demonstrating improved robustness over compared techniques.
>
---
#### [replaced 012] Graph-Based Floor Separation Using Node Embeddings and Clustering of WiFi Trajectories
- **分类: cs.NI; cs.AI; cs.CR; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.08088v2](http://arxiv.org/pdf/2505.08088v2)**

> **作者:** Rabia Yasa Kostas; Kahraman Kostas
>
> **摘要:** Indoor positioning systems (IPSs) are increasingly vital for location-based services in complex multi-storey environments. This study proposes a novel graph-based approach for floor separation using Wi-Fi fingerprint trajectories, addressing the challenge of vertical localization in indoor settings. We construct a graph where nodes represent Wi-Fi fingerprints, and edges are weighted by signal similarity and contextual transitions. Node2Vec is employed to generate low-dimensional embeddings, which are subsequently clustered using K-means to identify distinct floors. Evaluated on the Huawei University Challenge 2021 dataset, our method outperforms traditional community detection algorithms, achieving an accuracy of 68.97\%, an F1-score of 61.99\%, and an Adjusted Rand Index of 57.19\%. By publicly releasing the preprocessed dataset and implementation code, this work contributes to advancing research in indoor positioning. The proposed approach demonstrates robustness to signal noise and architectural complexities, offering a scalable solution for floor-level localization.
>
---
#### [replaced 013] Learning Multimodal Latent Dynamics for Human-Robot Interaction
- **分类: cs.RO; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2311.16380v2](http://arxiv.org/pdf/2311.16380v2)**

> **作者:** Vignesh Prasad; Lea Heitlinger; Dorothea Koert; Ruth Stock-Homburg; Jan Peters; Georgia Chalvatzaki
>
> **备注:** Preprint version of paper accepted at IEEE T-RO. Project website: https://sites.google.com/view/mild-hri
>
> **摘要:** This article presents a method for learning well-coordinated Human-Robot Interaction (HRI) from Human-Human Interactions (HHI). We devise a hybrid approach using Hidden Markov Models (HMMs) as the latent space priors for a Variational Autoencoder to model a joint distribution over the interacting agents. We leverage the interaction dynamics learned from HHI to learn HRI and incorporate the conditional generation of robot motions from human observations into the training, thereby predicting more accurate robot trajectories. The generated robot motions are further adapted with Inverse Kinematics to ensure the desired physical proximity with a human, combining the ease of joint space learning and accurate task space reachability. For contact-rich interactions, we modulate the robot's stiffness using HMM segmentation for a compliant interaction. We verify the effectiveness of our approach deployed on a Humanoid robot via a user study. Our method generalizes well to various humans despite being trained on data from just two humans. We find that users perceive our method as more human-like, timely, and accurate and rank our method with a higher degree of preference over other baselines. We additionally show the ability of our approach to generate successful interactions in a more complex scenario of Bimanual Robot-to-Human Handovers.
>
---
#### [replaced 014] V-Max: A Reinforcement Learning Framework for Autonomous Driving
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.08388v2](http://arxiv.org/pdf/2503.08388v2)**

> **作者:** Valentin Charraut; Thomas Tournaire; Waël Doulazmi; Thibault Buhet
>
> **备注:** Accepted to RLC 25
>
> **摘要:** Learning-based decision-making has the potential to enable generalizable Autonomous Driving (AD) policies, reducing the engineering overhead of rule-based approaches. Imitation Learning (IL) remains the dominant paradigm, benefiting from large-scale human demonstration datasets, but it suffers from inherent limitations such as distribution shift and imitation gaps. Reinforcement Learning (RL) presents a promising alternative, yet its adoption in AD remains limited due to the lack of standardized and efficient research frameworks. To this end, we introduce V-Max, an open research framework providing all the necessary tools to make RL practical for AD. V-Max is built on Waymax, a hardware-accelerated AD simulator designed for large-scale experimentation. We extend it using ScenarioNet's approach, enabling the fast simulation of diverse AD datasets.
>
---
