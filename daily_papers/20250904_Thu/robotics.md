# 机器人 cs.RO

- **最新发布 34 篇**

- **更新 20 篇**

## 最新发布

#### [new 001] Forbal: Force Balanced 2-5 Degree of Freedom Robot Manipulator Built from a Five Bar Linkage
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文设计力平衡五连杆机械臂，通过几何与动力学优化减少关节扭矩和反应力矩，提升精度，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.03119v1](http://arxiv.org/pdf/2509.03119v1)**

> **作者:** Yash Vyas; Matteo Bottin
>
> **摘要:** A force balanced manipulator design based on the closed chain planar five bar linkage is developed and experimentally validated. We present 2 variants as a modular design: Forbal-2, a planar 2-DOF manipulator, and its extension to 5-DOF spatial motion called Forbal-5. The design considerations in terms of geometric, kinematic, and dynamic design that fulfill the force balance conditions while maximizing workspace are discussed. Then, the inverse kinematics of both variants are derived from geometric principles. We validate the improvements from force balancing the manipulator through comparative experiments with counter mass balanced and unbalanced configurations. The results show how the balanced configuration yields a reduction in the average reaction moments of up to 66\%, a reduction of average joint torques of up to 79\%, as well as a noticeable reduction in position error for Forbal-2. For Forbal-5, which has a higher end effector payload mass, the joint torques are reduced up to 84\% for the balanced configuration. Experimental results validate that the balanced manipulator design is suitable for applications where the reduction of joint torques and reaction forces/moments helps achieve millimeter level precision.
>
---
#### [new 002] CTBC: Contact-Triggered Blind Climbing for Wheeled Bipedal Robots with Instruction Learning and Reinforcement Learning
- **分类: cs.RO**

- **简介: 论文提出CTBC框架，解决轮式双足机器人在复杂地形攀爬能力不足的问题，结合指令学习与强化学习，通过接触触发腿部动作，利用前馈轨迹提升性能，并在Tron1机器人上验证。**

- **链接: [http://arxiv.org/pdf/2509.02986v1](http://arxiv.org/pdf/2509.02986v1)**

> **作者:** Rankun Li; Hao Wang; Qi Li; Zhuo Han; Yifei Chu; Linqi Ye; Wende Xie; Wenlong Liao
>
> **摘要:** In recent years, wheeled bipedal robots have gained increasing attention due to their advantages in mobility, such as high-speed locomotion on flat terrain. However, their performance on complex environments (e.g., staircases) remains inferior to that of traditional legged robots. To overcome this limitation, we propose a general contact-triggered blind climbing (CTBC) framework for wheeled bipedal robots. Upon detecting wheel-obstacle contact, the robot triggers a leg-lifting motion to overcome the obstacle. By leveraging a strongly-guided feedforward trajectory, our method enables the robot to rapidly acquire agile leg-lifting skills, significantly enhancing its capability to traverse unstructured terrains. The approach has been experimentally validated and successfully deployed on LimX Dynamics' wheeled bipedal robot, Tron1. Real-world tests demonstrate that Tron1 can reliably climb obstacles well beyond its wheel radius using only proprioceptive feedback.
>
---
#### [new 003] A Digital Twin for Robotic Post Mortem Tissue Sampling using Virtual Reality
- **分类: cs.RO**

- **简介: 该论文提出基于虚拟现实的数字孪生系统，用于机器人尸检组织采样，解决传统方法破坏性大、感染风险高的问题。通过可用性测试与临床实验验证系统精确性，实现远程规划与控制，提升安全性与效率。**

- **链接: [http://arxiv.org/pdf/2509.02760v1](http://arxiv.org/pdf/2509.02760v1)**

> **作者:** Maximilian Neidhardt; Ludwig Bosse; Vidas Raudonis; Kristina Allgoewer; Axel Heinemann; Benjamin Ondruschka; Alexander Schlaefer
>
> **摘要:** Studying tissue samples obtained during autopsies is the gold standard when diagnosing the cause of death and for understanding disease pathophysiology. Recently, the interest in post mortem minimally invasive biopsies has grown which is a less destructive approach in comparison to an open autopsy and reduces the risk of infection. While manual biopsies under ultrasound guidance are more widely performed, robotic post mortem biopsies have been recently proposed. This approach can further reduce the risk of infection for physicians. However, planning of the procedure and control of the robot need to be efficient and usable. We explore a virtual reality setup with a digital twin to realize fully remote planning and control of robotic post mortem biopsies. The setup is evaluated with forensic pathologists in a usability study for three interaction methods. Furthermore, we evaluate clinical feasibility and evaluate the system with three human cadavers. Overall, 132 needle insertions were performed with an off-axis needle placement error of 5.30+-3.25 mm. Tissue samples were successfully biopsied and histopathologically verified. Users reported a very intuitive needle placement approach, indicating that the system is a promising, precise, and low-risk alternative to conventional approaches.
>
---
#### [new 004] DUViN: Diffusion-Based Underwater Visual Navigation via Knowledge-Transferred Depth Features
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DUViN方法，解决水下自主导航中无地图、障碍避障及地形高度保持问题。通过知识迁移的深度特征与扩散模型，实现端到端4-DoF控制，分阶段训练应对域迁移挑战，验证了模拟与真实环境下的有效性。**

- **链接: [http://arxiv.org/pdf/2509.02983v1](http://arxiv.org/pdf/2509.02983v1)**

> **作者:** Jinghe Yang; Minh-Quan Le; Mingming Gong; Ye Pu
>
> **摘要:** Autonomous underwater navigation remains a challenging problem due to limited sensing capabilities and the difficulty of constructing accurate maps in underwater environments. In this paper, we propose a Diffusion-based Underwater Visual Navigation policy via knowledge-transferred depth features, named DUViN, which enables vision-based end-to-end 4-DoF motion control for underwater vehicles in unknown environments. DUViN guides the vehicle to avoid obstacles and maintain a safe and perception awareness altitude relative to the terrain without relying on pre-built maps. To address the difficulty of collecting large-scale underwater navigation datasets, we propose a method that ensures robust generalization under domain shifts from in-air to underwater environments by leveraging depth features and introducing a novel model transfer strategy. Specifically, our training framework consists of two phases: we first train the diffusion-based visual navigation policy on in-air datasets using a pre-trained depth feature extractor. Secondly, we retrain the extractor on an underwater depth estimation task and integrate the adapted extractor into the trained navigation policy from the first step. Experiments in both simulated and real-world underwater environments demonstrate the effectiveness and generalization of our approach. The experimental videos are available at https://www.youtube.com/playlist?list=PLqt2s-RyCf1gfXJgFzKjmwIqYhrP4I-7Y.
>
---
#### [new 005] The Impact of Adaptive Emotional Alignment on Mental State Attribution and User Empathy in HRI
- **分类: cs.RO**

- **简介: 该论文通过实验研究适应性情绪同步对HRI中用户心理状态归因和共情的影响，比较中性与情绪适应性对话条件，发现其显著影响归因和共情，但不影响沟通风格和说服力。**

- **链接: [http://arxiv.org/pdf/2509.02749v1](http://arxiv.org/pdf/2509.02749v1)**

> **作者:** Giorgia Buracchio; Ariele Callegari; Massimo Donini; Cristina Gena; Antonio Lieto; Alberto Lillo; Claudio Mattutino; Alessandro Mazzei; Linda Pigureddu; Manuel Striani; Fabiana Vernero
>
> **备注:** autohor copy of the paper accepted at ROMAN2025
>
> **摘要:** The paper presents an experiment on the effects of adaptive emotional alignment between agents, considered a prerequisite for empathic communication, in Human-Robot Interaction (HRI). Using the NAO robot, we investigate the impact of an emotionally aligned, empathic, dialogue on these aspects: (i) the robot's persuasive effectiveness, (ii) the user's communication style, and (iii) the attribution of mental states and empathy to the robot. In an experiment with 42 participants, two conditions were compared: one with neutral communication and another where the robot provided responses adapted to the emotions expressed by the users. The results show that emotional alignment does not influence users' communication styles or have a persuasive effect. However, it significantly influences attribution of mental states to the robot and its perceived empathy
>
---
#### [new 006] Generalizable Skill Learning for Construction Robots with Crowdsourced Natural Language Instructions, Composable Skills Standardization, and Large Language Model
- **分类: cs.RO**

- **简介: 该论文提出基于众包自然语言指令和大语言模型的通用技能学习框架，解决建筑机器人跨领域任务迁移难题，通过标准化分层建模和实验验证实现高效多任务重编程。**

- **链接: [http://arxiv.org/pdf/2509.02876v1](http://arxiv.org/pdf/2509.02876v1)**

> **作者:** Hongrui Yu; Vineet R. Kamat; Carol C. Menassa
>
> **备注:** Under review for ASCE OPEN: Multidisciplinary Journal of Civil Engineering
>
> **摘要:** The quasi-repetitive nature of construction work and the resulting lack of generalizability in programming construction robots presents persistent challenges to the broad adoption of robots in the construction industry. Robots cannot achieve generalist capabilities as skills learnt from one domain cannot readily transfer to another work domain or be directly used to perform a different set of tasks. Human workers have to arduously reprogram their scene-understanding, path-planning, and manipulation components to enable the robots to perform alternate work tasks. The methods presented in this paper resolve a significant proportion of such reprogramming workload by proposing a generalizable learning architecture that directly teaches robots versatile task-performance skills through crowdsourced online natural language instructions. A Large Language Model (LLM), a standardized and modularized hierarchical modeling approach, and Building Information Modeling-Robot sematic data pipeline are developed to address the multi-task skill transfer problem. The proposed skill standardization scheme and LLM-based hierarchical skill learning framework were tested with a long-horizon drywall installation experiment using a full-scale industrial robotic manipulator. The resulting robot task learning scheme achieves multi-task reprogramming with minimal effort and high quality.
>
---
#### [new 007] Multi-Embodiment Locomotion at Scale with extreme Embodiment Randomization
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出通用运动策略，通过改进架构与课程学习，在50种腿部机器人上训练，实现极端形态随机化，零样本迁移至真实双足和四足机器人。**

- **链接: [http://arxiv.org/pdf/2509.02815v1](http://arxiv.org/pdf/2509.02815v1)**

> **作者:** Nico Bohlinger; Jan Peters
>
> **摘要:** We present a single, general locomotion policy trained on a diverse collection of 50 legged robots. By combining an improved embodiment-aware architecture (URMAv2) with a performance-based curriculum for extreme Embodiment Randomization, our policy learns to control millions of morphological variations. Our policy achieves zero-shot transfer to unseen real-world humanoid and quadruped robots.
>
---
#### [new 008] Uncertainty-aware Test-Time Training (UT$^3$) for Efficient On-the-fly Domain Adaptive Dense Regression
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出UT³框架，用于领域自适应的密集回归（如单目深度估计），通过不确定性感知的自监督减少推理延迟，提升实时应用效率。**

- **链接: [http://arxiv.org/pdf/2509.03012v1](http://arxiv.org/pdf/2509.03012v1)**

> **作者:** Uddeshya Upadhyay
>
> **摘要:** Deep neural networks (DNNs) are increasingly being used in autonomous systems. However, DNNs do not generalize well to domain shift. Adapting to a continuously evolving environment is a safety-critical challenge inevitably faced by all autonomous systems deployed to the real world. Recent work on test-time training proposes methods that adapt to a new test distribution on the fly by optimizing the DNN model for each test input using self-supervision. However, these techniques result in a sharp increase in inference time as multiple forward and backward passes are required for a single test sample (for test-time training) before finally making the prediction based on the fine-tuned features. This is undesirable for real-world robotics applications where these models may be deployed to resource constraint hardware with strong latency requirements. In this work, we propose a new framework (called UT$^3$) that leverages test-time training for improved performance in the presence of continuous domain shift while also decreasing the inference time, making it suitable for real-world applications. Our method proposes an uncertainty-aware self-supervision task for efficient test-time training that leverages the quantified uncertainty to selectively apply the training leading to sharp improvements in the inference time while performing comparably to standard test-time training protocol. Our proposed protocol offers a continuous setting to identify the selected keyframes, allowing the end-user to control how often to apply test-time training. We demonstrate the efficacy of our method on a dense regression task - monocular depth estimation.
>
---
#### [new 009] The Role of Embodiment in Intuitive Whole-Body Teleoperation for Mobile Manipulation
- **分类: cs.RO; cs.HC; cs.LG**

- **简介: 该论文研究移动机器人全身体操作的遥操作系统，比较耦合与解耦控制及VR与屏幕反馈，旨在提升数据质量和操作效率。结果发现耦合控制与解耦工作量相当，VR增加认知负荷，但耦合数据在模仿学习中表现更好。**

- **链接: [http://arxiv.org/pdf/2509.03222v1](http://arxiv.org/pdf/2509.03222v1)**

> **作者:** Sophia Bianchi Moyen; Rickmer Krohn; Sophie Lueth; Kay Pompetzki; Jan Peters; Vignesh Prasad; Georgia Chalvatzaki
>
> **备注:** 8 pages, 8 figures, Accepted at the IEEE-RAS International Conference on Humanoid Robots (Humanoids) 2025
>
> **摘要:** Intuitive Teleoperation interfaces are essential for mobile manipulation robots to ensure high quality data collection while reducing operator workload. A strong sense of embodiment combined with minimal physical and cognitive demands not only enhances the user experience during large-scale data collection, but also helps maintain data quality over extended periods. This becomes especially crucial for challenging long-horizon mobile manipulation tasks that require whole-body coordination. We compare two distinct robot control paradigms: a coupled embodiment integrating arm manipulation and base navigation functions, and a decoupled embodiment treating these systems as separate control entities. Additionally, we evaluate two visual feedback mechanisms: immersive virtual reality and conventional screen-based visualization of the robot's field of view. These configurations were systematically assessed across a complex, multi-stage task sequence requiring integrated planning and execution. Our results show that the use of VR as a feedback modality increases task completion time, cognitive workload, and perceived effort of the teleoperator. Coupling manipulation and navigation leads to a comparable workload on the user as decoupling the embodiments, while preliminary experiments suggest that data acquired by coupled teleoperation leads to better imitation learning performance. Our holistic view on intuitive teleoperation interfaces provides valuable insight into collecting high-quality, high-dimensional mobile manipulation data at scale with the human operator in mind. Project website:https://sophiamoyen.github.io/role-embodiment-wbc-moma-teleop/
>
---
#### [new 010] Robotic 3D Flower Pose Estimation for Small-Scale Urban Farms
- **分类: cs.RO**

- **简介: 该论文提出一种基于点云的三维花朵姿态估计方法，用于小型都市农场的机器人授粉。通过定制机器人采集点云，转换为多视角图像并结合2D检测与3D形状拟合，实现花朵位置与方向的高精度估计，准确率达80%，误差7.7度。**

- **链接: [http://arxiv.org/pdf/2509.02870v1](http://arxiv.org/pdf/2509.02870v1)**

> **作者:** Harsh Muriki; Hong Ray Teo; Ved Sengupta; Ai-Ping Hu
>
> **备注:** 7 pages, 7 figures
>
> **摘要:** The small scale of urban farms and the commercial availability of low-cost robots (such as the FarmBot) that automate simple tending tasks enable an accessible platform for plant phenotyping. We have used a FarmBot with a custom camera end-effector to estimate strawberry plant flower pose (for robotic pollination) from acquired 3D point cloud models. We describe a novel algorithm that translates individual occupancy grids along orthogonal axes of a point cloud to obtain 2D images corresponding to the six viewpoints. For each image, 2D object detection models for flowers are used to identify 2D bounding boxes which can be converted into the 3D space to extract flower point clouds. Pose estimation is performed by fitting three shapes (superellipsoids, paraboloids and planes) to the flower point clouds and compared with manually labeled ground truth. Our method successfully finds approximately 80% of flowers scanned using our customized FarmBot platform and has a mean flower pose error of 7.7 degrees, which is sufficient for robotic pollination and rivals previous results. All code will be made available at https://github.com/harshmuriki/flowerPose.git.
>
---
#### [new 011] Vibration Damping in Underactuated Cable-suspended Artwork -- Flying Belt Motion Control
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对缆绳悬挂艺术装置的振动问题，通过硬件升级与控制算法优化，解决因振荡动态导致的运动迟滞与交互响应不足。研究提出数学模型与输入整形方法，实现振动抑制，提升系统性能与艺术交互体验。**

- **链接: [http://arxiv.org/pdf/2509.03238v1](http://arxiv.org/pdf/2509.03238v1)**

> **作者:** Martin Goubej; Lauria Clarke; Martin Hrabačka; David Tolar
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** This paper presents a comprehensive refurbishment of the interactive robotic art installation Standards and Double Standards by Rafael Lozano-Hemmer. The installation features an array of belts suspended from the ceiling, each actuated by stepper motors and dynamically oriented by a vision-based tracking system that follows the movements of exhibition visitors. The original system was limited by oscillatory dynamics, resulting in torsional and pendulum-like vibrations that constrained rotational speed and reduced interactive responsiveness. To address these challenges, the refurbishment involved significant upgrades to both hardware and motion control algorithms. A detailed mathematical model of the flying belt system was developed to accurately capture its dynamic behavior, providing a foundation for advanced control design. An input shaping method, formulated as a convex optimization problem, was implemented to effectively suppress vibrations, enabling smoother and faster belt movements. Experimental results demonstrate substantial improvements in system performance and audience interaction. This work exemplifies the integration of robotics, control engineering, and interactive art, offering new solutions to technical challenges in real-time motion control and vibration damping for large-scale kinetic installations.
>
---
#### [new 012] Real-Time Instrument Planning and Perception for Novel Measurements of Dynamic Phenomena
- **分类: cs.RO; cs.AI**

- **简介: 论文提出自动化流程，结合动态事件检测与自主轨迹规划，用于高分辨率传感器获取动态现象（如火山羽流）的精准测量。分析分类方法并整合轨迹规划算法，通过模拟提升效用。**

- **链接: [http://arxiv.org/pdf/2509.03500v1](http://arxiv.org/pdf/2509.03500v1)**

> **作者:** Itai Zilberstein; Alberto Candela; Steve Chien
>
> **备注:** Appears in Proceedings of 18th Symposium on Advanced Space Technologies in Robotics and Automation
>
> **摘要:** Advancements in onboard computing mean remote sensing agents can employ state-of-the-art computer vision and machine learning at the edge. These capabilities can be leveraged to unlock new rare, transient, and pinpoint measurements of dynamic science phenomena. In this paper, we present an automated workflow that synthesizes the detection of these dynamic events in look-ahead satellite imagery with autonomous trajectory planning for a follow-up high-resolution sensor to obtain pinpoint measurements. We apply this workflow to the use case of observing volcanic plumes. We analyze classification approaches including traditional machine learning algorithms and convolutional neural networks. We present several trajectory planning algorithms that track the morphological features of a plume and integrate these algorithms with the classifiers. We show through simulation an order of magnitude increase in the utility return of the high-resolution instrument compared to baselines while maintaining efficient runtimes.
>
---
#### [new 013] Cost-Optimized Systems Engineering for IoT-Enabled Robot Nurse in Infectious Pandemic Management
- **分类: cs.RO; cs.HC; cs.SY; eess.SY; I.2.9; C.3; J.3**

- **简介: 该论文设计物联网赋能的机器人护士系统，解决疫情期间医疗自动化与感染控制问题。通过开发测试案例，评估其药物管理、健康监测及生命周期性能，实现成本优化的系统工程。**

- **链接: [http://arxiv.org/pdf/2509.03436v1](http://arxiv.org/pdf/2509.03436v1)**

> **作者:** Md Mhamud Hussen Sifat; Md Maruf; Md Rokunuzzaman
>
> **备注:** 11 pages, 10 figures, 4 tables, 1 algorithm. Corresponding author: Md Maruf (maruf.mte.17@gmail.com)
>
> **摘要:** The utilization of robotic technology has gained traction in healthcare facilities due to progress in the field that enables time and cost savings, minimizes waste, and improves patient care. Digital healthcare technologies that leverage automation, such as robotics and artificial intelligence, have the potential to enhance the sustainability and profitability of healthcare systems in the long run. However, the recent COVID-19 pandemic has amplified the need for cyber-physical robots to automate check-ups and medication administration. A robot nurse is controlled by the Internet of Things (IoT) and can serve as an automated medical assistant while also allowing supervisory control based on custom commands. This system helps reduce infection risk and improves outcomes in pandemic settings. This research presents a test case with a nurse robot that can assess a patient's health status and take action accordingly. We also evaluate the system's performance in medication administration, health-status monitoring, and life-cycle considerations.
>
---
#### [new 014] Can the Waymo Open Motion Dataset Support Realistic Behavioral Modeling? A Validation Study with Naturalistic Trajectories
- **分类: cs.RO; cs.AI; cs.LG; cs.SY; eess.SY; stat.AP**

- **简介: 该论文验证Waymo数据集是否支持真实自动驾驶行为建模，通过对比自然驾驶数据，发现其低估短头距和急刹车等复杂行为，警示需结合独立数据验证模型可靠性。**

- **链接: [http://arxiv.org/pdf/2509.03515v1](http://arxiv.org/pdf/2509.03515v1)**

> **作者:** Yanlin Zhang; Sungyong Chung; Nachuan Li; Dana Monzer; Hani S. Mahmassani; Samer H. Hamdar; Alireza Talebpour
>
> **摘要:** The Waymo Open Motion Dataset (WOMD) has become a popular resource for data-driven modeling of autonomous vehicles (AVs) behavior. However, its validity for behavioral analysis remains uncertain due to proprietary post-processing, the absence of error quantification, and the segmentation of trajectories into 20-second clips. This study examines whether WOMD accurately captures the dynamics and interactions observed in real-world AV operations. Leveraging an independently collected naturalistic dataset from Level 4 AV operations in Phoenix, Arizona (PHX), we perform comparative analyses across three representative urban driving scenarios: discharging at signalized intersections, car-following, and lane-changing behaviors. For the discharging analysis, headways are manually extracted from aerial video to ensure negligible measurement error. For the car-following and lane-changing cases, we apply the Simulation-Extrapolation (SIMEX) method to account for empirically estimated error in the PHX data and use Dynamic Time Warping (DTW) distances to quantify behavioral differences. Results across all scenarios consistently show that behavior in PHX falls outside the behavioral envelope of WOMD. Notably, WOMD underrepresents short headways and abrupt decelerations. These findings suggest that behavioral models calibrated solely on WOMD may systematically underestimate the variability, risk, and complexity of naturalistic driving. Caution is therefore warranted when using WOMD for behavior modeling without proper validation against independently collected data.
>
---
#### [new 015] Acrobotics: A Generalist Approahc To Quadrupedal Robots' Parkour
- **分类: cs.RO**

- **简介: 该论文提出通用强化学习算法，解决四足机器人动态运动控制问题，减少训练代理数量，达到与专家策略相当的性能。**

- **链接: [http://arxiv.org/pdf/2509.02727v1](http://arxiv.org/pdf/2509.02727v1)**

> **作者:** Guillaume Gagné-Labelle; Vassil Atanassov; Ioannis Havoutis
>
> **备注:** Supplementary material can be found here: https://drive.google.com/drive/folders/18h25azbCFfPF4fhSsRfxKrnZo3dPKs_j?usp=sharing
>
> **摘要:** Climbing, crouching, bridging gaps, and walking up stairs are just a few of the advantages that quadruped robots have over wheeled robots, making them more suitable for navigating rough and unstructured terrain. However, executing such manoeuvres requires precise temporal coordination and complex agent-environment interactions. Moreover, legged locomotion is inherently more prone to slippage and tripping, and the classical approach of modeling such cases to design a robust controller thus quickly becomes impractical. In contrast, reinforcement learning offers a compelling solution by enabling optimal control through trial and error. We present a generalist reinforcement learning algorithm for quadrupedal agents in dynamic motion scenarios. The learned policy rivals state-of-the-art specialist policies trained using a mixture of experts approach, while using only 25% as many agents during training. Our experiments also highlight the key components of the generalist locomotion policy and the primary factors contributing to its success.
>
---
#### [new 016] Efficient Active Training for Deep LiDAR Odometry
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出主动训练框架，用于提升LiDAR里程计的效率与泛化能力。针对传统方法需大量多样数据的问题，通过ITSS和AIS策略选择性提取训练样本，以52%数据量达到全数据集效果，优化训练过程并增强模型对复杂环境的适应性。**

- **链接: [http://arxiv.org/pdf/2509.03211v1](http://arxiv.org/pdf/2509.03211v1)**

> **作者:** Beibei Zhou; Zhiyuan Zhang; Zhenbo Song; Jianhui Guo; Hui Kong
>
> **摘要:** Robust and efficient deep LiDAR odometry models are crucial for accurate localization and 3D reconstruction, but typically require extensive and diverse training data to adapt to diverse environments, leading to inefficiencies. To tackle this, we introduce an active training framework designed to selectively extract training data from diverse environments, thereby reducing the training load and enhancing model generalization. Our framework is based on two key strategies: Initial Training Set Selection (ITSS) and Active Incremental Selection (AIS). ITSS begins by breaking down motion sequences from general weather into nodes and edges for detailed trajectory analysis, prioritizing diverse sequences to form a rich initial training dataset for training the base model. For complex sequences that are difficult to analyze, especially under challenging snowy weather conditions, AIS uses scene reconstruction and prediction inconsistency to iteratively select training samples, refining the model to handle a wide range of real-world scenarios. Experiments across datasets and weather conditions validate our approach's effectiveness. Notably, our method matches the performance of full-dataset training with just 52\% of the sequence volume, demonstrating the training efficiency and robustness of our active training paradigm. By optimizing the training process, our approach sets the stage for more agile and reliable LiDAR odometry systems, capable of navigating diverse environmental conditions with greater precision.
>
---
#### [new 017] Exploring persuasive Interactions with generative social robots: An experimental framework
- **分类: cs.RO**

- **简介: 该论文研究生成式社交机器人的说服性互动，设计实验框架测试外观与自我知识对说服效果的影响，分析互动质量与策略，提出优化建议。**

- **链接: [http://arxiv.org/pdf/2509.03231v1](http://arxiv.org/pdf/2509.03231v1)**

> **作者:** Stephan Vonschallen; Larissa Julia Corina Finsler; Theresa Schmiedel; Friederike Eyssel
>
> **备注:** A shortened version of this paper was accepted as poster for the Thirteenth International Conference on Human-Agent Interaction (HAI2025)
>
> **摘要:** Integrating generative AI such as large language models into social robots has improved their ability to engage in natural, human-like communication. This study presents a method to examine their persuasive capabilities. We designed an experimental framework focused on decision making and tested it in a pilot that varied robot appearance and self-knowledge. Using qualitative analysis, we evaluated interaction quality, persuasion effectiveness, and the robot's communicative strategies. Participants generally experienced the interaction positively, describing the robot as competent, friendly, and supportive, while noting practical limits such as delayed responses and occasional speech-recognition errors. Persuasiveness was highly context dependent and shaped by robot behavior: participants responded well to polite, reasoned suggestions and expressive gestures, but emphasized the need for more personalized, context-aware arguments and clearer social roles. These findings suggest that generative social robots can influence user decisions, but their effectiveness depends on communicative nuance and contextual relevance. We propose refinements to the framework to further study persuasive dynamics between robots and human users.
>
---
#### [new 018] Parallel-Constraint Model Predictive Control: Exploiting Parallel Computation for Improving Safety
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出并行约束模型预测控制方法，通过并行计算同时求解多个MPC问题，利用安全集约束提升机器人系统的安全性和性能。**

- **链接: [http://arxiv.org/pdf/2509.03261v1](http://arxiv.org/pdf/2509.03261v1)**

> **作者:** Elias Fontanari; Gianni Lunardi; Matteo Saveriano; Andrea Del Prete
>
> **摘要:** Ensuring constraint satisfaction is a key requirement for safety-critical systems, which include most robotic platforms. For example, constraints can be used for modeling joint position/velocity/torque limits and collision avoidance. Constrained systems are often controlled using Model Predictive Control, because of its ability to naturally handle constraints, relying on numerical optimization. However, ensuring constraint satisfaction is challenging for nonlinear systems/constraints. A well-known tool to make controllers safe is the so-called control-invariant set (a.k.a. safe set). In our previous work, we have shown that safety can be improved by letting the safe-set constraint recede along the MPC horizon. In this paper, we push that idea further by exploiting parallel computation to improve safety. We solve several MPC problems at the same time, where each problem instantiates the safe-set constraint at a different time step along the horizon. Finally, the controller can select the best solution according to some user-defined criteria. We validated this idea through extensive simulations with a 3-joint robotic arm, showing that significant improvements can be achieved in terms of safety and performance, even using as little as 4 computational cores.
>
---
#### [new 019] IL-SLAM: Intelligent Line-assisted SLAM Based on Feature Awareness for Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文提出IL-SLAM，针对动态环境中的视觉SLAM问题，解决传统方法因动态特征去除导致点特征不足及冗余特征引入带来的计算负担和性能下降。通过特征感知机制动态决策是否引入线特征，优化计算效率与定位精度。**

- **链接: [http://arxiv.org/pdf/2509.02972v1](http://arxiv.org/pdf/2509.02972v1)**

> **作者:** Haolan Zhang; Thanh Nguyen Canh; Chenghao Li; Ruidong Yang; Yonghoon Ji; Nak Young Chong
>
> **备注:** submitted to International Conference on Robotic Computing and Communication(IEEE IRC)
>
> **摘要:** Visual Simultaneous Localization and Mapping (SLAM) plays a crucial role in autonomous systems. Traditional SLAM methods, based on static environment assumptions, struggle to handle complex dynamic environments. Recent dynamic SLAM systems employ geometric constraints and deep learning to remove dynamic features, yet this creates a new challenge: insufficient remaining point features for subsequent SLAM processes. Existing solutions address this by continuously introducing additional line and plane features to supplement point features, achieving robust tracking and pose estimation. However, current methods continuously introduce additional features regardless of necessity, causing two problems: unnecessary computational overhead and potential performance degradation from accumulated low-quality additional features and noise. To address these issues, this paper proposes a feature-aware mechanism that evaluates whether current features are adequate to determine if line feature support should be activated. This decision mechanism enables the system to introduce line features only when necessary, significantly reducing computational complexity of additional features while minimizing the introduction of low-quality features and noise. In subsequent processing, the introduced line features assist in obtaining better initial camera poses through tracking, local mapping, and loop closure, but are excluded from global optimization to avoid potential negative impacts from low-quality additional features in long-term process. Extensive experiments on TUM datasets demonstrate substantial improvements in both ATE and RPE metrics compared to ORB-SLAM3 baseline and superior performance over other dynamic SLAM and multi-feature methods.
>
---
#### [new 020] Improving the Resilience of Quadrotors in Underground Environments by Combining Learning-based and Safety Controllers
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文针对四旋翼地下自主导航任务，解决学习控制器泛化不足问题，通过结合归一化流分布监测与安全控制器，实现高效安全的点对点导航。**

- **链接: [http://arxiv.org/pdf/2509.02808v1](http://arxiv.org/pdf/2509.02808v1)**

> **作者:** Isaac Ronald Ward; Mark Paral; Kristopher Riordan; Mykel J. Kochenderfer
>
> **备注:** Accepted and awarded best paper at the 11th International Conference on Control, Decision and Information Technologies (CoDIT 2025 - https://codit2025.org/)
>
> **摘要:** Autonomously controlling quadrotors in large-scale subterranean environments is applicable to many areas such as environmental surveying, mining operations, and search and rescue. Learning-based controllers represent an appealing approach to autonomy, but are known to not generalize well to `out-of-distribution' environments not encountered during training. In this work, we train a normalizing flow-based prior over the environment, which provides a measure of how far out-of-distribution the quadrotor is at any given time. We use this measure as a runtime monitor, allowing us to switch between a learning-based controller and a safe controller when we are sufficiently out-of-distribution. Our methods are benchmarked on a point-to-point navigation task in a simulated 3D cave environment based on real-world point cloud data from the DARPA Subterranean Challenge Final Event Dataset. Our experimental results show that our combined controller simultaneously possesses the liveness of the learning-based controller (completing the task quickly) and the safety of the safety controller (avoiding collision).
>
---
#### [new 021] Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey
- **分类: cs.RO; cs.CV**

- **简介: 该论文系统综述基于大VLM的VLA模型在机器人操作中的应用，解决传统方法在复杂环境中的不足，通过分类架构、分析集成领域及提出未来方向，整合现有研究并填补关键空白。**

- **链接: [http://arxiv.org/pdf/2508.13073v2](http://arxiv.org/pdf/2508.13073v2)**

> **作者:** Rui Shao; Wei Li; Lingsen Zhang; Renshan Zhang; Zhiyang Liu; Ran Chen; Liqiang Nie
>
> **备注:** Project Page: https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation
>
> **摘要:** Robotic manipulation, a key frontier in robotics and embodied AI, requires precise motor control and multimodal understanding, yet traditional rule-based methods fail to scale or generalize in unstructured, novel environments. In recent years, Vision-Language-Action (VLA) models, built upon Large Vision-Language Models (VLMs) pretrained on vast image-text datasets, have emerged as a transformative paradigm. This survey provides the first systematic, taxonomy-oriented review of large VLM-based VLA models for robotic manipulation. We begin by clearly defining large VLM-based VLA models and delineating two principal architectural paradigms: (1) monolithic models, encompassing single-system and dual-system designs with differing levels of integration; and (2) hierarchical models, which explicitly decouple planning from execution via interpretable intermediate representations. Building on this foundation, we present an in-depth examination of large VLM-based VLA models: (1) integration with advanced domains, including reinforcement learning, training-free optimization, learning from human videos, and world model integration; (2) synthesis of distinctive characteristics, consolidating architectural traits, operational strengths, and the datasets and benchmarks that support their development; (3) identification of promising directions, including memory mechanisms, 4D perception, efficient adaptation, multi-agent cooperation, and other emerging capabilities. This survey consolidates recent advances to resolve inconsistencies in existing taxonomies, mitigate research fragmentation, and fill a critical gap through the systematic integration of studies at the intersection of large VLMs and robotic manipulation. We provide a regularly updated project page to document ongoing progress: https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation
>
---
#### [new 022] Population-aware Online Mirror Descent for Mean-Field Games with Common Noise by Deep Reinforcement Learning
- **分类: cs.LG; cs.MA; cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出一种基于深度强化学习的算法，用于解决存在共同噪声和未知初始分布的Mean-Field Games纳什均衡学习问题。通过结合Munchausen RL与在线镜像下降，无需平均或历史采样，实现高效适应性策略，实验验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.03030v1](http://arxiv.org/pdf/2509.03030v1)**

> **作者:** Zida Wu; Mathieu Lauriere; Matthieu Geist; Olivier Pietquin; Ankur Mehta
>
> **备注:** 2025 IEEE 64rd Conference on Decision and Control (CDC)
>
> **摘要:** Mean Field Games (MFGs) offer a powerful framework for studying large-scale multi-agent systems. Yet, learning Nash equilibria in MFGs remains a challenging problem, particularly when the initial distribution is unknown or when the population is subject to common noise. In this paper, we introduce an efficient deep reinforcement learning (DRL) algorithm designed to achieve population-dependent Nash equilibria without relying on averaging or historical sampling, inspired by Munchausen RL and Online Mirror Descent. The resulting policy is adaptable to various initial distributions and sources of common noise. Through numerical experiments on seven canonical examples, we demonstrate that our algorithm exhibits superior convergence properties compared to state-of-the-art algorithms, particularly a DRL version of Fictitious Play for population-dependent policies. The performance in the presence of common noise underscores the robustness and adaptability of our approach.
>
---
#### [new 023] Decentralised self-organisation of pivoting cube ensembles using geometric deep learning
- **分类: cs.NE; cs.AI; cs.RO**

- **简介: 论文提出基于几何深度学习的去中心化模型，用于二维模块化立方体机器人自主重构。通过局部神经网络与多跳信息传递，实现高效目标形状重构，验证了局部控制的有效性，适用于空间系统如CubeSat群。**

- **链接: [http://arxiv.org/pdf/2509.03140v1](http://arxiv.org/pdf/2509.03140v1)**

> **作者:** Nadezhda Dobreva; Emmanuel Blazquez; Jai Grover; Dario Izzo; Yuzhen Qin; Dominik Dold
>
> **摘要:** We present a decentralized model for autonomous reconfiguration of homogeneous pivoting cube modular robots in two dimensions. Each cube in the ensemble is controlled by a neural network that only gains information from other cubes in its local neighborhood, trained using reinforcement learning. Furthermore, using geometric deep learning, we include the grid symmetries of the cube ensemble in the neural network architecture. We find that even the most localized versions succeed in reconfiguring to the target shape, although reconfiguration happens faster the more information about the whole ensemble is available to individual cubes. Near-optimal reconfiguration is achieved with only nearest neighbor interactions by using multiple information passing between cubes, allowing them to accumulate more global information about the ensemble. Compared to standard neural network architectures, using geometric deep learning approaches provided only minor benefits. Overall, we successfully demonstrate mostly local control of a modular self-assembling system, which is transferable to other space-relevant systems with different action spaces, such as sliding cube modular robots and CubeSat swarms.
>
---
#### [new 024] Approximate constrained stochastic optimal control via parameterized input inference
- **分类: eess.SY; cs.MA; cs.RO; cs.SY; math.OC**

- **简介: 该论文提出基于EM算法和障碍函数的参数化输入推断方法，解决约束随机最优控制问题，应用于无人车避障和四旋翼导航，并通过实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.02922v1](http://arxiv.org/pdf/2509.02922v1)**

> **作者:** Shahbaz P Qadri Syed; He Bai
>
> **摘要:** Approximate methods to solve stochastic optimal control (SOC) problems have received significant interest from researchers in the past decade. Probabilistic inference approaches to SOC have been developed to solve nonlinear quadratic Gaussian problems. In this work, we propose an Expectation-Maximization (EM) based inference procedure to generate state-feedback controls for constrained SOC problems. We consider the inequality constraints for the state and controls and also the structural constraints for the controls. We employ barrier functions to address state and control constraints. We show that the expectation step leads to smoothing of the state-control pair while the the maximization step on the non-zero subsets of the control parameters allows inference of structured stochastic optimal controllers. We demonstrate the effectiveness of the algorithm on unicycle obstacle avoidance, four-unicycle formation control, and quadcopter navigation in windy environment examples. In these examples, we perform an empirical study on the parametric effect of barrier functions on the state constraint satisfaction. We also present a comparative study of smoothing algorithms on the performance of the proposed approach.
>
---
#### [new 025] EclipseTouch: Touch Segmentation on Ad Hoc Surfaces using Worn Infrared Shadow Casting
- **分类: cs.HC; cs.CV; cs.GR; cs.RO**

- **简介: 论文提出EclipseTouch技术，通过红外阴影投射与摄像头结合，实现对非专用表面的高精度触摸检测，解决混合现实系统中未仪器化表面的触控难题。**

- **链接: [http://arxiv.org/pdf/2509.03430v1](http://arxiv.org/pdf/2509.03430v1)**

> **作者:** Vimal Mollyn; Nathan DeVrio; Chris Harrison
>
> **备注:** Accepted to UIST 2025
>
> **摘要:** The ability to detect touch events on uninstrumented, everyday surfaces has been a long-standing goal for mixed reality systems. Prior work has shown that virtual interfaces bound to physical surfaces offer performance and ergonomic benefits over tapping at interfaces floating in the air. A wide variety of approaches have been previously developed, to which we contribute a new headset-integrated technique called \systemname. We use a combination of a computer-triggered camera and one or more infrared emitters to create structured shadows, from which we can accurately estimate hover distance (mean error of 6.9~mm) and touch contact (98.0\% accuracy). We discuss how our technique works across a range of conditions, including surface material, interaction orientation, and environmental lighting.
>
---
#### [new 026] ANNIE: Be Careful of Your Robots
- **分类: cs.AI; cs.RO**

- **简介: 该论文针对具身AI系统的对抗性安全攻击问题，提出安全违规分类、ANNIEBench基准及任务感知攻击框架，评估攻击成功率超50%，揭示物理AI系统的安全漏洞，强调需强化防御机制。**

- **链接: [http://arxiv.org/pdf/2509.03383v1](http://arxiv.org/pdf/2509.03383v1)**

> **作者:** Yiyang Huang; Zixuan Wang; Zishen Wan; Yapeng Tian; Haobo Xu; Yinhe Han; Yiming Gan
>
> **摘要:** The integration of vision-language-action (VLA) models into embodied AI (EAI) robots is rapidly advancing their ability to perform complex, long-horizon tasks in humancentric environments. However, EAI systems introduce critical security risks: a compromised VLA model can directly translate adversarial perturbations on sensory input into unsafe physical actions. Traditional safety definitions and methodologies from the machine learning community are no longer sufficient. EAI systems raise new questions, such as what constitutes safety, how to measure it, and how to design effective attack and defense mechanisms in physically grounded, interactive settings. In this work, we present the first systematic study of adversarial safety attacks on embodied AI systems, grounded in ISO standards for human-robot interactions. We (1) formalize a principled taxonomy of safety violations (critical, dangerous, risky) based on physical constraints such as separation distance, velocity, and collision boundaries; (2) introduce ANNIEBench, a benchmark of nine safety-critical scenarios with 2,400 video-action sequences for evaluating embodied safety; and (3) ANNIE-Attack, a task-aware adversarial framework with an attack leader model that decomposes long-horizon goals into frame-level perturbations. Our evaluation across representative EAI models shows attack success rates exceeding 50% across all safety categories. We further demonstrate sparse and adaptive attack strategies and validate the real-world impact through physical robot experiments. These results expose a previously underexplored but highly consequential attack surface in embodied AI systems, highlighting the urgent need for security-driven defenses in the physical AI era. Code is available at https://github.com/RLCLab/Annie.
>
---
#### [new 027] 2nd Place Solution for CVPR2024 E2E Challenge: End-to-End Autonomous Driving Using Vision Language Model
- **分类: cs.CV; cs.RO**

- **简介: 论文提出结合端到端架构与视觉语言模型（VLM）的自动驾驶方法，仅用单目摄像头在CVPR2024挑战中取得第二名，验证了视觉驱动方案的有效性。**

- **链接: [http://arxiv.org/pdf/2509.02659v1](http://arxiv.org/pdf/2509.02659v1)**

> **作者:** Zilong Guo; Yi Luo; Long Sha; Dongxu Wang; Panqu Wang; Chenyang Xu; Yi Yang
>
> **备注:** 2nd place in CVPR 2024 End-to-End Driving at Scale Challenge
>
> **摘要:** End-to-end autonomous driving has drawn tremendous attention recently. Many works focus on using modular deep neural networks to construct the end-to-end archi-tecture. However, whether using powerful large language models (LLM), especially multi-modality Vision Language Models (VLM) could benefit the end-to-end driving tasks remain a question. In our work, we demonstrate that combining end-to-end architectural design and knowledgeable VLMs yield impressive performance on the driving tasks. It is worth noting that our method only uses a single camera and is the best camera-only solution across the leaderboard, demonstrating the effectiveness of vision-based driving approach and the potential for end-to-end driving tasks.
>
---
#### [new 028] sam-llm: interpretable lane change trajectoryprediction via parametric finetuning
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出SAM-LLM模型，通过参数化微调结合LLM与物理模型，解决车道变更轨迹预测的可解释性与计算效率问题，实现高精度且连续的轨迹生成。**

- **链接: [http://arxiv.org/pdf/2509.03462v1](http://arxiv.org/pdf/2509.03462v1)**

> **作者:** Zhuo Cao; Yunxiao Shi; Min Xu
>
> **备注:** 5 pages
>
> **摘要:** This work introduces SAM-LLM, a novel hybrid architecture that bridges the gap between the contextual reasoning of Large Language Models (LLMs) and the physical precision of kinematic lane change models for autonomous driving. The system is designed for interpretable lane change trajectory prediction by finetuning an LLM to output the core physical parameters of a trajectory model instead of raw coordinates. For lane-keeping scenarios, the model predicts discrete coordinates, but for lane change maneuvers, it generates the parameters for an enhanced Sinusoidal Acceleration Model (SAM), including lateral displacement, maneuver duration, initial lateral velocity, and longitudinal velocity change. This parametric approach yields a complete, continuous, and physically plausible trajectory model that is inherently interpretable and computationally efficient, achieving an 80% reduction in output size compared to coordinate-based methods. The SAM-LLM achieves a state-of-the-art overall intention prediction accuracy of 98.73%, demonstrating performance equivalent to traditional LLM predictors while offering significant advantages in explainability and resource efficiency.
>
---
#### [new 029] VendiRL: A Framework for Self-Supervised Reinforcement Learning of Diversely Diverse Skills
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出VendiRL框架，解决自监督强化学习中技能多样性评估与可扩展性问题。通过借鉴生态学的Vendi Score指标，支持多种相似度函数定义的技能多样性学习，实现统一的多样化技能预训练。**

- **链接: [http://arxiv.org/pdf/2509.02930v1](http://arxiv.org/pdf/2509.02930v1)**

> **作者:** Erik M. Lintunen
>
> **备注:** 17 pages including appendices
>
> **摘要:** In self-supervised reinforcement learning (RL), one of the key challenges is learning a diverse set of skills to prepare agents for unknown future tasks. Despite impressive advances, scalability and evaluation remain prevalent issues. Regarding scalability, the search for meaningful skills can be obscured by high-dimensional feature spaces, where relevant features may vary across downstream task domains. For evaluating skill diversity, defining what constitutes "diversity" typically requires a hard commitment to a specific notion of what it means for skills to be diverse, potentially leading to inconsistencies in how skill diversity is understood, making results across different approaches hard to compare, and leaving many forms of diversity unexplored. To address these issues, we adopt a measure of sample diversity that translates ideas from ecology to machine learning -- the Vendi Score -- allowing the user to specify and evaluate any desired form of diversity. We demonstrate how this metric facilitates skill evaluation and introduce VendiRL, a unified framework for learning diversely diverse sets of skills. Given distinct similarity functions, VendiRL motivates distinct forms of diversity, which could support skill-diversity pretraining in new and richly interactive environments where optimising for various forms of diversity may be desirable.
>
---
#### [new 030] Dependency Chain Analysis of ROS 2 DDS QoS Policies: From Lifecycle Tutorial to Static Verification
- **分类: cs.NI; cs.RO**

- **简介: 该论文针对ROS 2 DDS QoS策略配置问题，通过生命周期分析构建依赖链模型，分类41条违规规则，并开发QoS Guard工具实现静态验证，提升系统可靠性与资源效率。**

- **链接: [http://arxiv.org/pdf/2509.03381v1](http://arxiv.org/pdf/2509.03381v1)**

> **作者:** Sanghoon Lee; Junha Kang; Kyung-Joon Park
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** Robot Operating System 2 (ROS 2) relies on the Data Distribution Service (DDS), which offers more than 20 Quality of Service (QoS) policies governing availability, reliability, and resource usage. Yet ROS 2 users lack clear guidance on safe policy combinations and validation processes prior to deployment, which often leads to trial-and-error tuning and unexpected runtime failures. To address these challenges, we analyze DDS Publisher-Subscriber communication over a life cycle divided into Discovery, Data Exchange, and Disassociation, and provide a user oriented tutorial explaining how 16 QoS policies operate in each phase. Building on this analysis, we derive a QoS dependency chain that formalizes inter-policy relationships and classifies 41 dependency violation rules, capturing constraints that commonly cause communication failures in practice. Finally, we introduce QoS Guard, a ROS 2 package that statically validates DDS XML profiles offline, flags conflicts, and enables safe, predeployment tuning without establishing a live ROS 2 session. Together, these contributions give ROS 2 users both conceptual insight and a concrete tool that enables early detection of misconfigurations, improving the reliability and resource efficiency of ROS 2 based robotic systems.
>
---
#### [new 031] Separation of Three or More Autonomous Mobile Models under Hierarchical Schedulers
- **分类: cs.DC; cs.RO**

- **简介: 该论文研究多自主移动模型在分层调度器下的分离问题，通过分析同步性、可观测性与内存交互，揭示不同模型计算能力差异，扩展了14个经典模型的分离图，提出新不可能条件。**

- **链接: [http://arxiv.org/pdf/2508.19805v1](http://arxiv.org/pdf/2508.19805v1)**

> **作者:** Shota Naito; Tsukasa Ninomiya; Koichi Wada
>
> **摘要:** Understanding the computational power of mobile robot systems is a fundamental challenge in distributed computing. While prior work has focused on pairwise separations between models, we explore how robot capabilities, light observability, and scheduler synchrony interact in more complex ways. We first show that the Exponential Times Expansion (ETE) problem is solvable only in the strongest model -- fully-synchronous robots with full mutual lights ($\mathcal{LUMT}^F$). We then introduce the Hexagonal Edge Traversal (HET) and TAR(d)* problems to demonstrate how internal memory and lights interact with synchrony: under weak synchrony, internal memory alone is insufficient, while full synchrony can substitute for both lights and memory. In the asynchronous setting, we classify problems such as LP-MLCv, VEC, and ZCC to show fine-grained separations between $\mathcal{FSTA}$ and $\mathcal{FCOM}$ robots. We also analyze Vertex Traversal Rendezvous (VTR) and Leave Place Convergence (LP-Cv), illustrating the limitations of internal memory in symmetric settings. These results extend the known separation map of 14 canonical robot models, revealing structural phenomena only visible through higher-order comparisons. Our work provides new impossibility criteria and deepens the understanding of how observability, memory, and synchrony collectively shape the computational power of mobile robots.
>
---
#### [new 032] Who Owns The Robot?: Four Ethical and Socio-technical Questions about Wellbeing Robots in the Real World through Community Engagement
- **分类: cs.CY; cs.AI; cs.HC; cs.RO; I.2.9; K.4.2; K.4.1**

- **简介: 该论文通过社区参与研究，探讨福祉机器人在现实应用中的伦理和社会技术问题，提出四个关键问题框架，以指导开发者考虑公共利益与社区意见。**

- **链接: [http://arxiv.org/pdf/2509.02624v1](http://arxiv.org/pdf/2509.02624v1)**

> **作者:** Minja Axelsson; Jiaee Cheong; Rune Nyrup; Hatice Gunes
>
> **备注:** Accepted at the 8th AAAI/ACM Conference on AI, Ethics, and Society. 23 pages, 1 figure
>
> **摘要:** Recent studies indicate that robotic coaches can play a crucial role in promoting wellbeing. However, the real-world deployment of wellbeing robots raises numerous ethical and socio-technical questions and concerns. To explore these questions, we undertake a community-centered investigation to examine three different communities' perspectives on using robotic wellbeing coaches in real-world environments. We frame our work as an anticipatory ethical investigation, which we undertake to better inform the development of robotic technologies with communities' opinions, with the ultimate goal of aligning robot development with public interest. We conducted workshops with three communities who are under-represented in robotics development: 1) members of the public at a science festival, 2) women computer scientists at a conference, and 3) humanities researchers interested in history and philosophy of science. In the workshops, we collected qualitative data using the Social Robot Co-Design Canvas on Ethics. We analysed the collected qualitative data with Thematic Analysis, informed by notes taken during workshops. Through our analysis, we identify four themes regarding key ethical and socio-technical questions about the real-world use of wellbeing robots. We group participants' insights and discussions around these broad thematic questions, discuss them in light of state-of-the-art literature, and highlight areas for future investigation. Finally, we provide the four questions as a broad framework that roboticists can and should use during robotic development and deployment, in order to reflect on the ethics and socio-technical dimensions of their robotic applications, and to engage in dialogue with communities of robot users. The four questions are: 1) Is the robot safe and how can we know that?, 2) Who is the robot built for and with?, 3) Who owns the robot and the data?, and 4) Why a robot?.
>
---
#### [new 033] AI Safety Assurance in Electric Vehicles: A Case Study on AI-Driven SOC Estimation
- **分类: cs.SE; cs.RO**

- **简介: 该论文研究AI在电动车中的安全保证，解决传统方法无法评估AI功能的问题，通过结合ISO 26262与ISO/PAS 8800标准，对AI驱动的SOC估计组件进行独立评估，并通过故障注入测试验证其鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.03270v1](http://arxiv.org/pdf/2509.03270v1)**

> **作者:** Martin Skoglund; Fredrik Warg; Aria Mirzai; Anders Thorsen; Karl Lundgren; Peter Folkesson; Bastian Havers-zulka
>
> **备注:** 12 pages, 9 figures, EVS38, https://evs38-program.org/en/evs-38-proceedings/all
>
> **摘要:** Integrating Artificial Intelligence (AI) technology in electric vehicles (EV) introduces unique challenges for safety assurance, particularly within the framework of ISO 26262, which governs functional safety in the automotive domain. Traditional assessment methodologies are not geared toward evaluating AI-based functions and require evolving standards and practices. This paper explores how an independent assessment of an AI component in an EV can be achieved when combining ISO 26262 with the recently released ISO/PAS 8800, whose scope is AI safety for road vehicles. The AI-driven State of Charge (SOC) battery estimation exemplifies the process. Key features relevant to the independent assessment of this extended evaluation approach are identified. As part of the evaluation, robustness testing of the AI component is conducted using fault injection experiments, wherein perturbed sensor inputs are systematically introduced to assess the component's resilience to input variance.
>
---
#### [new 034] SmartPoser: Arm Pose Estimation with a Smartphone and Smartwatch Using UWB and IMU Data
- **分类: cs.HC; cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出一种基于智能手机与智能手表的臂部姿态估计方法，利用UWB测距与IMU惯性数据互补，解决现有方案的隐私问题及设备复杂性，实现无需训练数据的高精度姿态追踪（中位误差11cm）。**

- **链接: [http://arxiv.org/pdf/2509.03451v1](http://arxiv.org/pdf/2509.03451v1)**

> **作者:** Nathan DeVrio; Vimal Mollyn; Chris Harrison
>
> **备注:** The first two listed authors contributed equally. Published at UIST 2023
>
> **摘要:** The ability to track a user's arm pose could be valuable in a wide range of applications, including fitness, rehabilitation, augmented reality input, life logging, and context-aware assistants. Unfortunately, this capability is not readily available to consumers. Systems either require cameras, which carry privacy issues, or utilize multiple worn IMUs or markers. In this work, we describe how an off-the-shelf smartphone and smartwatch can work together to accurately estimate arm pose. Moving beyond prior work, we take advantage of more recent ultra-wideband (UWB) functionality on these devices to capture absolute distance between the two devices. This measurement is the perfect complement to inertial data, which is relative and suffers from drift. We quantify the performance of our software-only approach using off-the-shelf devices, showing it can estimate the wrist and elbow joints with a \hl{median positional error of 11.0~cm}, without the user having to provide training data.
>
---
## 更新

#### [replaced 001] HDVIO2.0: Wind and Disturbance Estimation with Hybrid Dynamics VIO
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.00969v3](http://arxiv.org/pdf/2504.00969v3)**

> **作者:** Giovanni Cioffi; Leonard Bauersfeld; Davide Scaramuzza
>
> **备注:** Transactions on Robotics (T-RO) 2025
>
> **摘要:** Visual-inertial odometry (VIO) is widely used for state estimation in autonomous micro aerial vehicles using onboard sensors. Current methods improve VIO by incorporating a model of the translational vehicle dynamics, yet their performance degrades when faced with low-accuracy vehicle models or continuous external disturbances, like wind. Additionally, incorporating rotational dynamics in these models is computationally intractable when they are deployed in online applications, e.g., in a closed-loop control system. We present HDVIO2.0, which models full 6-DoF, translational and rotational, vehicle dynamics and tightly incorporates them into a VIO with minimal impact on the runtime. HDVIO2.0 builds upon the previous work, HDVIO, and addresses these challenges through a hybrid dynamics model combining a point-mass vehicle model with a learning-based component, with access to control commands and IMU history, to capture complex aerodynamic effects. The key idea behind modeling the rotational dynamics is to represent them with continuous-time functions. HDVIO2.0 leverages the divergence between the actual motion and the predicted motion from the hybrid dynamics model to estimate external forces as well as the robot state. Our system surpasses the performance of state-of-the-art methods in experiments using public and new drone dynamics datasets, as well as real-world flights in winds up to 25 km/h. Unlike existing approaches, we also show that accurate vehicle dynamics predictions are achievable without precise knowledge of the full vehicle state.
>
---
#### [replaced 002] Sem-RaDiff: Diffusion-Based 3D Radar Semantic Perception in Cluttered Agricultural Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.02283v2](http://arxiv.org/pdf/2509.02283v2)**

> **作者:** Ruibin Zhang; Fei Gao
>
> **摘要:** Accurate and robust environmental perception is crucial for robot autonomous navigation. While current methods typically adopt optical sensors (e.g., camera, LiDAR) as primary sensing modalities, their susceptibility to visual occlusion often leads to degraded performance or complete system failure. In this paper, we focus on agricultural scenarios where robots are exposed to the risk of onboard sensor contamination. Leveraging radar's strong penetration capability, we introduce a radar-based 3D environmental perception framework as a viable alternative. It comprises three core modules designed for dense and accurate semantic perception: 1) Parallel frame accumulation to enhance signal-to-noise ratio of radar raw data. 2) A diffusion model-based hierarchical learning framework that first filters radar sidelobe artifacts then generates fine-grained 3D semantic point clouds. 3) A specifically designed sparse 3D network optimized for processing large-scale radar raw data. We conducted extensive benchmark comparisons and experimental evaluations on a self-built dataset collected in real-world agricultural field scenes. Results demonstrate that our method achieves superior structural and semantic prediction performance compared to existing methods, while simultaneously reducing computational and memory costs by 51.3% and 27.5%, respectively. Furthermore, our approach achieves complete reconstruction and accurate classification of thin structures such as poles and wires-which existing methods struggle to perceive-highlighting its potential for dense and accurate 3D radar perception.
>
---
#### [replaced 003] PPF: Pre-training and Preservative Fine-tuning of Humanoid Locomotion via Model-Assumption-based Regularization
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.09833v2](http://arxiv.org/pdf/2504.09833v2)**

> **作者:** Hyunyoung Jung; Zhaoyuan Gu; Ye Zhao; Hae-Won Park; Sehoon Ha
>
> **摘要:** Humanoid locomotion is a challenging task due to its inherent complexity and high-dimensional dynamics, as well as the need to adapt to diverse and unpredictable environments. In this work, we introduce a novel learning framework for effectively training a humanoid locomotion policy that imitates the behavior of a model-based controller while extending its capabilities to handle more complex locomotion tasks, such as more challenging terrain and higher velocity commands. Our framework consists of three key components: pre-training through imitation of the model-based controller, fine-tuning via reinforcement learning, and model-assumption-based regularization (MAR) during fine-tuning. In particular, MAR aligns the policy with actions from the model-based controller only in states where the model assumption holds to prevent catastrophic forgetting. We evaluate the proposed framework through comprehensive simulation tests and hardware experiments on a full-size humanoid robot, Digit, demonstrating a forward speed of 1.5 m/s and robust locomotion across diverse terrains, including slippery, sloped, uneven, and sandy terrains.
>
---
#### [replaced 004] Communication Efficient Robotic Mixed Reality with Gaussian Splatting Cross-Layer Optimization
- **分类: cs.RO; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2508.08624v2](http://arxiv.org/pdf/2508.08624v2)**

> **作者:** Chenxuan Liu; He Li; Zongze Li; Shuai Wang; Wei Xu; Kejiang Ye; Derrick Wing Kwan Ng; Chengzhong Xu
>
> **备注:** 14 pages, 18 figures, to appear in IEEE Transactions on Cognitive Communications and Networking
>
> **摘要:** Realizing low-cost communication in robotic mixed reality (RoboMR) systems presents a challenge, due to the necessity of uploading high-resolution images through wireless channels. This paper proposes Gaussian splatting (GS) RoboMR (GSMR), which enables the simulator to opportunistically render a photo-realistic view from the robot's pose by calling ``memory'' from a GS model, thus reducing the need for excessive image uploads. However, the GS model may involve discrepancies compared to the actual environments. To this end, a GS cross-layer optimization (GSCLO) framework is further proposed, which jointly optimizes content switching (i.e., deciding whether to upload image or not) and power allocation (i.e., adjusting to content profiles) across different frames by minimizing a newly derived GSMR loss function. The GSCLO problem is addressed by an accelerated penalty optimization (APO) algorithm that reduces computational complexity by over $10$x compared to traditional branch-and-bound and search algorithms. Moreover, variants of GSCLO are presented to achieve robust, low-power, and multi-robot GSMR. Extensive experiments demonstrate that the proposed GSMR paradigm and GSCLO method achieve significant improvements over existing benchmarks on both wheeled and legged robots in terms of diverse metrics in various scenarios. For the first time, it is found that RoboMR can be achieved with ultra-low communication costs, and mixture of data is useful for enhancing GS performance in dynamic scenarios.
>
---
#### [replaced 005] Embodied AI: Emerging Risks and Opportunities for Policy Action
- **分类: cs.CY; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.00117v2](http://arxiv.org/pdf/2509.00117v2)**

> **作者:** Jared Perlo; Alexander Robey; Fazl Barez; Luciano Floridi; Jakob Mökander
>
> **摘要:** The field of embodied AI (EAI) is rapidly advancing. Unlike virtual AI, EAI systems can exist in, learn from, reason about, and act in the physical world. With recent advances in AI models and hardware, EAI systems are becoming increasingly capable across wider operational domains. While EAI systems can offer many benefits, they also pose significant risks, including physical harm from malicious use, mass surveillance, as well as economic and societal disruption. These risks require urgent attention from policymakers, as existing policies governing industrial robots and autonomous vehicles are insufficient to address the full range of concerns EAI systems present. To help address this issue, this paper makes three contributions. First, we provide a taxonomy of the physical, informational, economic, and social risks EAI systems pose. Second, we analyze policies in the US, EU, and UK to assess how existing frameworks address these risks and to identify critical gaps. We conclude by offering policy recommendations for the safe and beneficial deployment of EAI systems, such as mandatory testing and certification schemes, clarified liability frameworks, and strategies to manage EAI's potentially transformative economic and societal impacts.
>
---
#### [replaced 006] A Coarse-to-Fine Approach to Multi-Modality 3D Occupancy Grounding
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.01197v2](http://arxiv.org/pdf/2508.01197v2)**

> **作者:** Zhan Shi; Song Wang; Junbo Chen; Jianke Zhu
>
> **摘要:** Visual grounding aims to identify objects or regions in a scene based on natural language descriptions, essential for spatially aware perception in autonomous driving. However, existing visual grounding tasks typically depend on bounding boxes that often fail to capture fine-grained details. Not all voxels within a bounding box are occupied, resulting in inaccurate object representations. To address this, we introduce a benchmark for 3D occupancy grounding in challenging outdoor scenes. Built on the nuScenes dataset, it integrates natural language with voxel-level occupancy annotations, offering more precise object perception compared to the traditional grounding task. Moreover, we propose GroundingOcc, an end-to-end model designed for 3D occupancy grounding through multi-modal learning. It combines visual, textual, and point cloud features to predict object location and occupancy information from coarse to fine. Specifically, GroundingOcc comprises a multimodal encoder for feature extraction, an occupancy head for voxel-wise predictions, and a grounding head to refine localization. Additionally, a 2D grounding module and a depth estimation module enhance geometric understanding, thereby boosting model performance. Extensive experiments on the benchmark demonstrate that our method outperforms existing baselines on 3D occupancy grounding. The dataset is available at https://github.com/RONINGOD/GroundingOcc.
>
---
#### [replaced 007] RMMI: Reactive Mobile Manipulation using an Implicit Neural Map
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2408.16206v2](http://arxiv.org/pdf/2408.16206v2)**

> **作者:** Nicolas Marticorena; Tobias Fischer; Jesse Haviland; Niko Suenderhauf
>
> **备注:** 8 pages, 6 figures, accepted to the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Mobile manipulator robots operating in complex domestic and industrial environments must effectively coordinate their base and arm motions while avoiding obstacles. While current reactive control methods gracefully achieve this coordination, they rely on simplified and idealised geometric representations of the environment to avoid collisions. This limits their performance in cluttered environments. To address this problem, we introduce RMMI, a reactive control framework that leverages the ability of neural Signed Distance Fields (SDFs) to provide a continuous and differentiable representation of the environment's geometry. RMMI formulates a quadratic program that optimises jointly for robot base and arm motion, maximises the manipulability, and avoids collisions through a set of inequality constraints. These constraints are constructed by querying the SDF for the distance and direction to the closest obstacle for a large number of sampling points on the robot. We evaluate RMMI both in simulation and in a set of real-world experiments. For reaching in cluttered environments, we observe a 25% increase in success rate. For additional details, code, and experiment videos, please visit https://rmmi.github.io/.
>
---
#### [replaced 008] Point Cloud Recombination: Systematic Real Data Augmentation Using Robotic Targets for LiDAR Perception Validation
- **分类: cs.RO; cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.02476v2](http://arxiv.org/pdf/2505.02476v2)**

> **作者:** Hubert Padusinski; Christian Steinhauser; Christian Scherl; Julian Gaal; Jacob Langner
>
> **备注:** Pre-print for IEEE IAVVC 2025
>
> **摘要:** The validation of LiDAR-based perception of intelligent mobile systems operating in open-world applications remains a challenge due to the variability of real environmental conditions. Virtual simulations allow the generation of arbitrary scenes under controlled conditions but lack physical sensor characteristics, such as intensity responses or material-dependent effects. In contrast, real-world data offers true sensor realism but provides less control over influencing factors, hindering sufficient validation. Existing approaches address this problem with augmentation of real-world point cloud data by transferring objects between scenes. However, these methods do not consider validation and remain limited in controllability because they rely on empirical data. We solve these limitations by proposing Point Cloud Recombination, which systematically augments captured point cloud scenes by integrating point clouds acquired from physical target objects measured in controlled laboratory environments. Thus enabling the creation of vast amounts and varieties of repeatable, physically accurate test scenes with respect to phenomena-aware occlusions with registered 3D meshes. Using the Ouster OS1-128 Rev7 sensor, we demonstrate the augmentation of real-world urban and rural scenes with humanoid targets featuring varied clothing and poses, for repeatable positioning. We show that the recombined scenes closely match real sensor outputs, enabling targeted testing, scalable failure analysis, and improved system safety. By providing controlled yet sensor-realistic data, our method enables trustworthy conclusions about the limitations of specific sensors in compound with their algorithms, e.g., object detection.
>
---
#### [replaced 009] Open-Set LiDAR Panoptic Segmentation Guided by Uncertainty-Aware Learning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.13265v3](http://arxiv.org/pdf/2506.13265v3)**

> **作者:** Rohit Mohan; Julia Hindel; Florian Drews; Claudius Gläser; Daniele Cattaneo; Abhinav Valada
>
> **摘要:** Autonomous vehicles that navigate in open-world environments may encounter previously unseen object classes. However, most existing LiDAR panoptic segmentation models rely on closed-set assumptions, failing to detect unknown object instances. In this work, we propose ULOPS, an uncertainty-guided open-set panoptic segmentation framework that leverages Dirichlet-based evidential learning to model predictive uncertainty. Our architecture incorporates separate decoders for semantic segmentation with uncertainty estimation, embedding with prototype association, and instance center prediction. During inference, we leverage uncertainty estimates to identify and segment unknown instances. To strengthen the model's ability to differentiate between known and unknown objects, we introduce three uncertainty-driven loss functions. Uniform Evidence Loss to encourage high uncertainty in unknown regions. Adaptive Uncertainty Separation Loss ensures a consistent difference in uncertainty estimates between known and unknown objects at a global scale. Contrastive Uncertainty Loss refines this separation at the fine-grained level. To evaluate open-set performance, we extend benchmark settings on KITTI-360 and introduce a new open-set evaluation for nuScenes. Extensive experiments demonstrate that ULOPS consistently outperforms existing open-set LiDAR panoptic segmentation methods.
>
---
#### [replaced 010] Distributed Lloyd-Based Algorithm for Uncertainty-Aware Multi-Robot Under-Canopy Flocking
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.18840v2](http://arxiv.org/pdf/2504.18840v2)**

> **作者:** Manuel Boldrer; Vit Kratky; Viktor Walter; Martin Saska
>
> **摘要:** In this letter, we present a distributed algorithm for flocking in complex environments that operates at constant altitude, without explicit communication, no a priori information about the environment, and by using only on-board sensing and computation capabilities. We provide sufficient conditions to guarantee that each robot reaches its goal region in a finite time, avoiding collisions with obstacles and other robots without exceeding a desired maximum distance from a predefined set of neighbors (flocking or proximity constraint). The proposed approach allows to operate in crowded scenarios and to deal with tracking errors and on-board sensing errors, without violating safety and proximity constraints. The algorithm was verified through simulations with varying number of UAVs and also through numerous real-world experiments in a dense forest involving up to four UAVs.
>
---
#### [replaced 011] A Survey: Learning Embodied Intelligence from Physical Simulators and World Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.00917v3](http://arxiv.org/pdf/2507.00917v3)**

> **作者:** Xiaoxiao Long; Qingrui Zhao; Kaiwen Zhang; Zihao Zhang; Dingrui Wang; Yumeng Liu; Zhengjie Shu; Yi Lu; Shouzheng Wang; Xinzhe Wei; Wei Li; Wei Yin; Yao Yao; Jia Pan; Qiu Shen; Ruigang Yang; Xun Cao; Qionghai Dai
>
> **备注:** Update with recent progresses. 49pages, 25figures, 6tables, github repository avalible in https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey
>
> **摘要:** The pursuit of artificial general intelligence (AGI) has placed embodied intelligence at the forefront of robotics research. Embodied intelligence focuses on agents capable of perceiving, reasoning, and acting within the physical world. Achieving robust embodied intelligence requires not only advanced perception and control, but also the ability to ground abstract cognition in real-world interactions. Two foundational technologies, physical simulators and world models, have emerged as critical enablers in this quest. Physical simulators provide controlled, high-fidelity environments for training and evaluating robotic agents, allowing safe and efficient development of complex behaviors. In contrast, world models empower robots with internal representations of their surroundings, enabling predictive planning and adaptive decision-making beyond direct sensory input. This survey systematically reviews recent advances in learning embodied AI through the integration of physical simulators and world models. We analyze their complementary roles in enhancing autonomy, adaptability, and generalization in intelligent robots, and discuss the interplay between external simulation and internal modeling in bridging the gap between simulated training and real-world deployment. By synthesizing current progress and identifying open challenges, this survey aims to provide a comprehensive perspective on the path toward more capable and generalizable embodied AI systems. We also maintain an active repository that contains up-to-date literature and open-source projects at https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey.
>
---
#### [replaced 012] RoboMemory: A Brain-inspired Multi-memory Agentic Framework for Lifelong Learning in Physical Embodied Systems
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.01415v3](http://arxiv.org/pdf/2508.01415v3)**

> **作者:** Mingcong Lei; Honghao Cai; Binbin Que; Zezhou Cui; Liangchen Tan; Junkun Hong; Gehan Hu; Shuangyu Zhu; Yimou Wu; Shaohan Jiang; Ge Wang; Zhen Li; Shuguang Cui; Yiming Zhao; Yatong Han
>
> **摘要:** We present RoboMemory, a brain-inspired multi-memory framework for lifelong learning in physical embodied systems, addressing critical challenges in real-world environments: continuous learning, multi-module memory latency, task correlation capture, and infinite-loop mitigation in closed-loop planning. Grounded in cognitive neuroscience, it integrates four core modules: the Information Preprocessor (thalamus-like), the Lifelong Embodied Memory System (hippocampus-like), the Closed-Loop Planning Module (prefrontal lobe-like), and the Low-Level Executer (cerebellum-like) to enable long-term planning and cumulative learning. The Lifelong Embodied Memory System, central to the framework, alleviates inference speed issues in complex memory frameworks via parallelized updates/retrieval across Spatial, Temporal, Episodic, and Semantic submodules. It incorporates a dynamic Knowledge Graph (KG) and consistent architectural design to enhance memory consistency and scalability. Evaluations on EmbodiedBench show RoboMemory outperforms the open-source baseline (Qwen2.5-VL-72B-Ins) by 25% in average success rate and surpasses the closed-source State-of-the-Art (SOTA) (Claude3.5-Sonnet) by 5%, establishing new SOTA. Ablation studies validate key components (critic, spatial memory, long-term memory), while real-world deployment confirms its lifelong learning capability with significantly improved success rates across repeated tasks. RoboMemory alleviates high latency challenges with scalability, serving as a foundational reference for integrating multi-modal memory systems in physical robots.
>
---
#### [replaced 013] Controlling Deformable Objects with Non-negligible Dynamics: a Shape-Regulation Approach to End-Point Positioning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2402.16114v2](http://arxiv.org/pdf/2402.16114v2)**

> **作者:** Sebastien Tiburzio; Tomás Coleman; Daniel Feliu-Talegon; Cosimo Della Santina
>
> **备注:** 15 pages, 18 figures. Accepted for publication as a Regular Paper in the IEEE Transactions on Robotics (T-RO)
>
> **摘要:** Model-based manipulation of deformable objects has traditionally dealt with objects while neglecting their dynamics, thus mostly focusing on very lightweight objects at steady state. At the same time, soft robotic research has made considerable strides toward general modeling and control, despite soft robots and deformable objects being very similar from a mechanical standpoint. In this work, we leverage these recent results to develop a control-oriented, fully dynamic framework of slender deformable objects grasped at one end by a robotic manipulator. We introduce a dynamic model of this system using functional strain parameterizations and describe the manipulation challenge as a regulation control problem. This enables us to define a fully model-based control architecture, for which we can prove analytically closed-loop stability and provide sufficient conditions for steady state convergence to the desired state. The nature of this work is intended to be markedly experimental. We provide an extensive experimental validation of the proposed ideas, tasking a robot arm with controlling the distal end of six different cables, in a given planar position and orientation in space.
>
---
#### [replaced 014] JARVIS: A Neuro-Symbolic Commonsense Reasoning Framework for Conversational Embodied Agents
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2208.13266v4](http://arxiv.org/pdf/2208.13266v4)**

> **作者:** Kaizhi Zheng; Kaiwen Zhou; Jing Gu; Yue Fan; Jialu Wang; Zonglin Di; Xuehai He; Xin Eric Wang
>
> **备注:** 19th International Conference on Neurosymbolic Learning and Reasoning
>
> **摘要:** Building a conversational embodied agent to execute real-life tasks has been a long-standing yet quite challenging research goal, as it requires effective human-agent communication, multi-modal understanding, long-range sequential decision making, etc. Traditional symbolic methods have scaling and generalization issues, while end-to-end deep learning models suffer from data scarcity and high task complexity, and are often hard to explain. To benefit from both worlds, we propose JARVIS, a neuro-symbolic commonsense reasoning framework for modular, generalizable, and interpretable conversational embodied agents. First, it acquires symbolic representations by prompting large language models (LLMs) for language understanding and sub-goal planning, and by constructing semantic maps from visual observations. Then the symbolic module reasons for sub-goal planning and action generation based on task- and action-level common sense. Extensive experiments on the TEACh dataset validate the efficacy and efficiency of our JARVIS framework, which achieves state-of-the-art (SOTA) results on all three dialog-based embodied tasks, including Execution from Dialog History (EDH), Trajectory from Dialog (TfD), and Two-Agent Task Completion (TATC) (e.g., our method boosts the unseen Success Rate on EDH from 6.1\% to 15.8\%). Moreover, we systematically analyze the essential factors that affect the task performance and also demonstrate the superiority of our method in few-shot settings. Our JARVIS model ranks first in the Alexa Prize SimBot Public Benchmark Challenge.
>
---
#### [replaced 015] Integration of Computer Vision with Adaptive Control for Autonomous Driving Using ADORE
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.17985v2](http://arxiv.org/pdf/2508.17985v2)**

> **作者:** Abu Shad Ahammed; Md Shahi Amran Hossain; Sayeri Mukherjee; Roman Obermaisser; Md. Ziaur Rahman
>
> **摘要:** Ensuring safety in autonomous driving requires a seamless integration of perception and decision making under uncertain conditions. Although computer vision (CV) models such as YOLO achieve high accuracy in detecting traffic signs and obstacles, their performance degrades in drift scenarios caused by weather variations or unseen objects. This work presents a simulated autonomous driving system that combines a context aware CV model with adaptive control using the ADORE framework. The CARLA simulator was integrated with ADORE via the ROS bridge, allowing real-time communication between perception, decision, and control modules. A simulated test case was designed in both clear and drift weather conditions to demonstrate the robust detection performance of the perception model while ADORE successfully adapted vehicle behavior to speed limits and obstacles with low response latency. The findings highlight the potential of coupling deep learning-based perception with rule-based adaptive decision making to improve automotive safety critical system.
>
---
#### [replaced 016] LanternNet: A Hub-and-Spoke System to Seek and Suppress Spotted Lanternfly Populations
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20800v3](http://arxiv.org/pdf/2507.20800v3)**

> **作者:** Vinil Polepalli
>
> **摘要:** The invasive spotted lanternfly (SLF) poses a significant threat to agriculture and ecosystems, causing widespread damage. Current control methods, such as egg scraping, pesticides, and quarantines, prove labor-intensive, environmentally hazardous, and inadequate for long-term SLF suppression. This research introduces LanternNet, a novel autonomous robotic Hub-and-Spoke system designed for scalable detection and suppression of SLF populations. A central, tree-mimicking hub utilizes a YOLOv8 computer vision model for precise SLF identification. Three specialized robotic spokes perform targeted tasks: pest neutralization, environmental monitoring, and navigation/mapping. Field deployment across multiple infested sites over 5 weeks demonstrated LanternNet's efficacy. Quantitative analysis revealed significant reductions (p < 0.01, paired t-tests) in SLF populations and corresponding improvements in tree health indicators across the majority of test sites. Compared to conventional methods, LanternNet offers substantial cost advantages and improved scalability. Furthermore, the system's adaptability for enhanced autonomy and targeting of other invasive species presents significant potential for broader ecological impact. LanternNet demonstrates the transformative potential of integrating robotics and AI for advanced invasive species management and improved environmental outcomes.
>
---
#### [replaced 017] Hey, Teacher, (Don't) Leave Those Kids Alone: Standardizing HRI Education
- **分类: cs.RO; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2404.00024v2](http://arxiv.org/pdf/2404.00024v2)**

> **作者:** Alexis E. Block
>
> **备注:** Presented at the Designing an Intro to HRI Course Workshop at HRI 2024 (arXiv:2403.05588)
>
> **摘要:** Creating a standardized introduction course becomes more critical as the field of human-robot interaction (HRI) becomes more established. This paper outlines the key components necessary to provide an undergraduate with a sufficient foundational understanding of the interdisciplinary nature of this field and provides proposed course content. It emphasizes the importance of creating a course with theoretical and experimental components to accommodate all different learning preferences. This manuscript also advocates creating or adopting a universal platform to standardize the hands-on component of introductory HRI courses, regardless of university funding or size. Next, it recommends formal training in how to read scientific articles and staying up-to-date with the latest relevant papers. Finally, it provides detailed lecture content and project milestones for a 15-week semester. By creating a standardized course, researchers can ensure consistency and quality are maintained across institutions, which will help students as well as industrial and academic employers understand what foundational knowledge is expected.
>
---
#### [replaced 018] BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.16490v3](http://arxiv.org/pdf/2412.16490v3)**

> **作者:** Jiayi Chen; Yubin Ke; He Wang
>
> **备注:** ICRA 2025
>
> **摘要:** Robotic dexterous grasping is important for interacting with the environment. To unleash the potential of data-driven models for dexterous grasping, a large-scale, high-quality dataset is essential. While gradient-based optimization offers a promising way for constructing such datasets, previous works suffer from limitations, such as inefficiency, strong assumptions in the grasp quality energy, or limited object sets for experiments. Moreover, the lack of a standard benchmark for comparing different methods and datasets hinders progress in this field. To address these challenges, we develop a highly efficient synthesis system and a comprehensive benchmark with MuJoCo for dexterous grasping. We formulate grasp synthesis as a bilevel optimization problem, combining a novel lower-level quadratic programming (QP) with an upper-level gradient descent process. By leveraging recent advances in CUDA-accelerated robotic libraries and GPU-based QP solvers, our system can parallelize thousands of grasps and synthesize over 49 grasps per second on a single 3090 GPU. Our synthesized grasps for Shadow, Allegro, and Leap hands all achieve a success rate above 75% in simulation, with a penetration depth under 1 mm, outperforming existing baselines on nearly all metrics. Compared to the previous large-scale dataset, DexGraspNet, our dataset significantly improves the performance of learning models, with a success rate from around 40% to 80% in simulation. Real-world testing of the trained model on the Shadow Hand achieves an 81% success rate across 20 diverse objects. The codes and datasets are released on our project page: https://pku-epic.github.io/BODex.
>
---
#### [replaced 019] Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.18829v2](http://arxiv.org/pdf/2504.18829v2)**

> **作者:** Jiayi Chen; Yubin Ke; Lin Peng; He Wang
>
> **备注:** Accepted by Robotics: Science and Systems (RSS 2025)
>
> **摘要:** Generalizable dexterous grasping with suitable grasp types is a fundamental skill for intelligent robots. Developing such skills requires a large-scale and high-quality dataset that covers numerous grasp types (i.e., at least those categorized by the GRASP taxonomy), but collecting such data is extremely challenging. Existing automatic grasp synthesis methods are often limited to specific grasp types or object categories, hindering scalability. This work proposes an efficient pipeline capable of synthesizing contact-rich, penetration-free, and physically plausible grasps for any grasp type, object, and articulated hand. Starting from a single human-annotated template for each hand and grasp type, our pipeline tackles the complicated synthesis problem with two stages: optimize the object to fit the hand template first, and then locally refine the hand to fit the object in simulation. To validate the synthesized grasps, we introduce a contact-aware control strategy that allows the hand to apply the appropriate force at each contact point to the object. Those validated grasps can also be used as new grasp templates to facilitate future synthesis. Experiments show that our method significantly outperforms previous type-unaware grasp synthesis baselines in simulation. Using our algorithm, we construct a dataset containing 10.7k objects and 9.5M grasps, covering 31 grasp types in the GRASP taxonomy. Finally, we train a type-conditional generative model that successfully performs the desired grasp type from single-view object point clouds, achieving an 82.3% success rate in real-world experiments. Project page: https://pku-epic.github.io/Dexonomy.
>
---
#### [replaced 020] Stretchable Electrohydraulic Artificial Muscle for Full Motion Ranges in Musculoskeletal Antagonistic Joints
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.11017v3](http://arxiv.org/pdf/2409.11017v3)**

> **作者:** Amirhossein Kazemipour; Ronan Hinchet; Robert K. Katzschmann
>
> **备注:** This paper has been accepted to the IEEE International Conference on Robotics and Automation (ICRA) 2025
>
> **摘要:** Artificial muscles play a crucial role in musculoskeletal robotics and prosthetics to approximate the force-generating functionality of biological muscle. However, current artificial muscle systems are typically limited to either contraction or extension, not both. This limitation hinders the development of fully functional artificial musculoskeletal systems. We address this challenge by introducing an artificial antagonistic muscle system capable of both contraction and extension. Our design integrates non-stretchable electrohydraulic soft actuators (HASELs) with electrostatic clutches within an antagonistic musculoskeletal framework. This configuration enables an antagonistic joint to achieve a full range of motion without displacement loss due to tendon slack. We implement a synchronization method to coordinate muscle and clutch units, ensuring smooth motion profiles and speeds. This approach facilitates seamless transitions between antagonistic muscles at operational frequencies of up to 3.2 Hz. While our prototype utilizes electrohydraulic actuators, this muscle-clutch concept is adaptable to other non-stretchable artificial muscles, such as McKibben actuators, expanding their capability for extension and full range of motion in antagonistic setups. Our design represents a significant advancement in the development of fundamental components for more functional and efficient artificial musculoskeletal systems, bringing their capabilities closer to those of their biological counterparts.
>
---
