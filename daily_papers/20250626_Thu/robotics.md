# 机器人 cs.RO

- **最新发布 31 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] Building Forest Inventories with Autonomous Legged Robots -- System, Lessons, and Challenges Ahead
- **分类: cs.RO**

- **简介: 该论文属于森林资源调查任务，旨在解决腿式机器人在非结构化森林环境中的自主导航与树木检测问题。工作包括设计系统架构、实现自主导航与地图构建，并验证其在实际环境中的性能。**

- **链接: [http://arxiv.org/pdf/2506.20315v1](http://arxiv.org/pdf/2506.20315v1)**

> **作者:** Matías Mattamala; Nived Chebrolu; Jonas Frey; Leonard Freißmuth; Haedam Oh; Benoit Casseau; Marco Hutter; Maurice Fallon
>
> **备注:** 20 pages, 13 figures. Pre-print version of the accepted paper for IEEE Transactions on Field Robotics (T-FR)
>
> **摘要:** Legged robots are increasingly being adopted in industries such as oil, gas, mining, nuclear, and agriculture. However, new challenges exist when moving into natural, less-structured environments, such as forestry applications. This paper presents a prototype system for autonomous, under-canopy forest inventory with legged platforms. Motivated by the robustness and mobility of modern legged robots, we introduce a system architecture which enabled a quadruped platform to autonomously navigate and map forest plots. Our solution involves a complete navigation stack for state estimation, mission planning, and tree detection and trait estimation. We report the performance of the system from trials executed over one and a half years in forests in three European countries. Our results with the ANYmal robot demonstrate that we can survey plots up to 1 ha plot under 30 min, while also identifying trees with typical DBH accuracy of 2cm. The findings of this project are presented as five lessons and challenges. Particularly, we discuss the maturity of hardware development, state estimation limitations, open problems in forest navigation, future avenues for robotic forest inventory, and more general challenges to assess autonomous systems. By sharing these lessons and challenges, we offer insight and new directions for future research on legged robots, navigation systems, and applications in natural environments. Additional videos can be found in https://dynamic.robots.ox.ac.uk/projects/legged-robots
>
---
#### [new 002] Leveraging Correlation Across Test Platforms for Variance-Reduced Metric Estimation
- **分类: cs.RO**

- **简介: 该论文属于机器人系统验证任务，旨在减少真实世界测试成本。通过利用仿真与真实数据的相关性，降低估计方差，提升样本效率。**

- **链接: [http://arxiv.org/pdf/2506.20553v1](http://arxiv.org/pdf/2506.20553v1)**

> **作者:** Rachel Luo; Heng Yang; Michael Watson; Apoorva Sharma; Sushant Veer; Edward Schmerling; Marco Pavone
>
> **摘要:** Learning-based robotic systems demand rigorous validation to assure reliable performance, but extensive real-world testing is often prohibitively expensive, and if conducted may still yield insufficient data for high-confidence guarantees. In this work, we introduce a general estimation framework that leverages paired data across test platforms, e.g., paired simulation and real-world observations, to achieve better estimates of real-world metrics via the method of control variates. By incorporating cheap and abundant auxiliary measurements (for example, simulator outputs) as control variates for costly real-world samples, our method provably reduces the variance of Monte Carlo estimates and thus requires significantly fewer real-world samples to attain a specified confidence bound on the mean performance. We provide theoretical analysis characterizing the variance and sample-efficiency improvement, and demonstrate empirically in autonomous driving and quadruped robotics settings that our approach achieves high-probability bounds with markedly improved sample efficiency. Our technique can lower the real-world testing burden for validating the performance of the stack, thereby enabling more efficient and cost-effective experimental evaluation of robotic systems.
>
---
#### [new 003] Critical Anatomy-Preserving & Terrain-Augmenting Navigation (CAPTAiN): Application to Laminectomy Surgical Education
- **分类: cs.RO**

- **简介: 该论文属于手术导航任务，旨在解决 laminectomy 中的精准操作与风险控制问题。通过引入 CAPTAiN 系统，提升手术准确性与安全性。**

- **链接: [http://arxiv.org/pdf/2506.20496v1](http://arxiv.org/pdf/2506.20496v1)**

> **作者:** Jonathan Wang; Hisashi Ishida; David Usevitch; Kesavan Venkatesh; Yi Wang; Mehran Armand; Rachel Bronheim; Amit Jain; Adnan Munawar
>
> **摘要:** Surgical training remains a crucial milestone in modern medicine, with procedures such as laminectomy exemplifying the high risks involved. Laminectomy drilling requires precise manual control to mill bony tissue while preserving spinal segment integrity and avoiding breaches in the dura: the protective membrane surrounding the spinal cord. Despite unintended tears occurring in up to 11.3% of cases, no assistive tools are currently utilized to reduce this risk. Variability in patient anatomy further complicates learning for novice surgeons. This study introduces CAPTAiN, a critical anatomy-preserving and terrain-augmenting navigation system that provides layered, color-coded voxel guidance to enhance anatomical awareness during spinal drilling. CAPTAiN was evaluated against a standard non-navigated approach through 110 virtual laminectomies performed by 11 orthopedic residents and medical students. CAPTAiN significantly improved surgical completion rates of target anatomy (87.99% vs. 74.42%) and reduced cognitive load across multiple NASA-TLX domains. It also minimized performance gaps across experience levels, enabling novices to perform on par with advanced trainees. These findings highlight CAPTAiN's potential to optimize surgical execution and support skill development across experience levels. Beyond laminectomy, it demonstrates potential for broader applications across various surgical and drilling procedures, including those in neurosurgery, otolaryngology, and other medical fields.
>
---
#### [new 004] Evolutionary Gait Reconfiguration in Damaged Legged Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决腿式机器人受损后运动恢复问题。通过生成新步态并优化配置，实现高效稳定移动。**

- **链接: [http://arxiv.org/pdf/2506.19968v1](http://arxiv.org/pdf/2506.19968v1)**

> **作者:** Sahand Farghdani; Robin Chhabra
>
> **摘要:** Multi-legged robots deployed in complex missions are susceptible to physical damage in their legs, impairing task performance and potentially compromising mission success. This letter presents a rapid, training-free damage recovery algorithm for legged robots subject to partial or complete loss of functional legs. The proposed method first stabilizes locomotion by generating a new gait sequence and subsequently optimally reconfigures leg gaits via a developed differential evolution algorithm to maximize forward progression while minimizing body rotation and lateral drift. The algorithm successfully restores locomotion in a 24-degree-of-freedom hexapod within one hour, demonstrating both high efficiency and robustness to structural damage.
>
---
#### [new 005] Finding the Easy Way Through -- the Probabilistic Gap Planner for Social Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于社会机器人导航任务，解决复杂人群中的路径规划问题。提出PGP算法，结合冲突避让与协作避障，提升导航性能。**

- **链接: [http://arxiv.org/pdf/2506.20320v1](http://arxiv.org/pdf/2506.20320v1)**

> **作者:** Malte Probst; Raphael Wenzel; Tim Puphal; Monica Dasi; Nico A. Steinhardt; Sango Matsuzaki; Misa Komuro
>
> **摘要:** In Social Robot Navigation, autonomous agents need to resolve many sequential interactions with other agents. State-of-the art planners can efficiently resolve the next, imminent interaction cooperatively and do not focus on longer planning horizons. This makes it hard to maneuver scenarios where the agent needs to select a good strategy to find gaps or channels in the crowd. We propose to decompose trajectory planning into two separate steps: Conflict avoidance for finding good, macroscopic trajectories, and cooperative collision avoidance (CCA) for resolving the next interaction optimally. We propose the Probabilistic Gap Planner (PGP) as a conflict avoidance planner. PGP modifies an established probabilistic collision risk model to include a general assumption of cooperativity. PGP biases the short-term CCA planner to head towards gaps in the crowd. In extensive simulations with crowds of varying density, we show that using PGP in addition to state-of-the-art CCA planners improves the agents' performance: On average, agents keep more space to others, create less tension, and cause fewer collisions. This typically comes at the expense of slightly longer paths. PGP runs in real-time on WaPOCHI mobile robot by Honda R&D.
>
---
#### [new 006] EANS: Reducing Energy Consumption for UAV with an Environmental Adaptive Navigation Strategy
- **分类: cs.RO**

- **简介: 该论文属于无人机导航任务，旨在解决动态环境下能量消耗过高的问题。通过动态调整导航策略，降低能耗并提升效率。**

- **链接: [http://arxiv.org/pdf/2506.20485v1](http://arxiv.org/pdf/2506.20485v1)**

> **作者:** Tian Liu; Han Liu; Boyang Li; Long Chen; Kai Huang
>
> **摘要:** Unmanned Aerial Vehicles (UAVS) are limited by the onboard energy. Refinement of the navigation strategy directly affects both the flight velocity and the trajectory based on the adjustment of key parameters in the UAVS pipeline, thus reducing energy consumption. However, existing techniques tend to adopt static and conservative strategies in dynamic scenarios, leading to inefficient energy reduction. Dynamically adjusting the navigation strategy requires overcoming the challenges including the task pipeline interdependencies, the environmental-strategy correlations, and the selecting parameters. To solve the aforementioned problems, this paper proposes a method to dynamically adjust the navigation strategy of the UAVS by analyzing its dynamic characteristics and the temporal characteristics of the autonomous navigation pipeline, thereby reducing UAVS energy consumption in response to environmental changes. We compare our method with the baseline through hardware-in-the-loop (HIL) simulation and real-world experiments, showing our method 3.2X and 2.6X improvements in mission time, 2.4X and 1.6X improvements in energy, respectively.
>
---
#### [new 007] CARMA: Context-Aware Situational Grounding of Human-Robot Group Interactions by Combining Vision-Language Models with Object and Action Recognition
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文提出CARMA系统，解决人机群体交互中的情境定位问题。通过结合视觉语言模型与物体动作识别，实现对参与者、物体及动作的准确跟踪与关联。**

- **链接: [http://arxiv.org/pdf/2506.20373v1](http://arxiv.org/pdf/2506.20373v1)**

> **作者:** Joerg Deigmoeller; Stephan Hasler; Nakul Agarwal; Daniel Tanneberg; Anna Belardinelli; Reza Ghoddoosian; Chao Wang; Felix Ocker; Fan Zhang; Behzad Dariush; Michael Gienger
>
> **摘要:** We introduce CARMA, a system for situational grounding in human-robot group interactions. Effective collaboration in such group settings requires situational awareness based on a consistent representation of present persons and objects coupled with an episodic abstraction of events regarding actors and manipulated objects. This calls for a clear and consistent assignment of instances, ensuring that robots correctly recognize and track actors, objects, and their interactions over time. To achieve this, CARMA uniquely identifies physical instances of such entities in the real world and organizes them into grounded triplets of actors, objects, and actions. To validate our approach, we conducted three experiments, where multiple humans and a robot interact: collaborative pouring, handovers, and sorting. These scenarios allow the assessment of the system's capabilities as to role distinction, multi-actor awareness, and consistent instance identification. Our experiments demonstrate that the system can reliably generate accurate actor-action-object triplets, providing a structured and robust foundation for applications requiring spatiotemporal reasoning and situated decision-making in collaborative settings.
>
---
#### [new 008] Communication-Aware Map Compression for Online Path-Planning: A Rate-Distortion Approach
- **分类: cs.RO**

- **简介: 该论文属于协同导航任务，解决在带宽受限下如何压缩地图以支持路径规划的问题。通过率失真优化方法确定压缩策略，实现高效通信与实时决策。**

- **链接: [http://arxiv.org/pdf/2506.20579v1](http://arxiv.org/pdf/2506.20579v1)**

> **作者:** Ali Reza Pedram; Evangelos Psomiadis; Dipankar Maity; Panagiotis Tsiotras
>
> **摘要:** This paper addresses the problem of collaborative navigation in an unknown environment, where two robots, referred to in the sequel as the Seeker and the Supporter, traverse the space simultaneously. The Supporter assists the Seeker by transmitting a compressed representation of its local map under bandwidth constraints to support the Seeker's path-planning task. We introduce a bit-rate metric based on the expected binary codeword length to quantify communication cost. Using this metric, we formulate the compression design problem as a rate-distortion optimization problem that determines when to communicate, which regions of the map should be included in the compressed representation, and at what resolution (i.e., quantization level) they should be encoded. Our formulation allows different map regions to be encoded at varying quantization levels based on their relevance to the Seeker's path-planning task. We demonstrate that the resulting optimization problem is convex, and admits a closed-form solution known in the information theory literature as reverse water-filling, enabling efficient, low-computation, and real-time implementation. Additionally, we show that the Seeker can infer the compression decisions of the Supporter independently, requiring only the encoded map content and not the encoding policy itself to be transmitted, thereby reducing communication overhead. Simulation results indicate that our method effectively constructs compressed, task-relevant map representations, both in content and resolution, that guide the Seeker's planning decisions even under tight bandwidth limitations.
>
---
#### [new 009] Robust Robotic Exploration and Mapping Using Generative Occupancy Map Synthesis
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人探索与建图任务，旨在提升地图质量和可 traversability。通过生成式占用图合成方法，实时融合预测结果，增强地图性能。**

- **链接: [http://arxiv.org/pdf/2506.20049v1](http://arxiv.org/pdf/2506.20049v1)**

> **作者:** Lorin Achey; Alec Reed; Brendan Crowe; Bradley Hayes; Christoffer Heckman
>
> **备注:** arXiv admin note: text overlap with arXiv:2409.10681
>
> **摘要:** We present a novel approach for enhancing robotic exploration by using generative occupancy mapping. We introduce SceneSense, a diffusion model designed and trained for predicting 3D occupancy maps given partial observations. Our proposed approach probabilistically fuses these predictions into a running occupancy map in real-time, resulting in significant improvements in map quality and traversability. We implement SceneSense onboard a quadruped robot and validate its performance with real-world experiments to demonstrate the effectiveness of the model. In these experiments, we show that occupancy maps enhanced with SceneSense predictions better represent our fully observed ground truth data (24.44% FID improvement around the robot and 75.59% improvement at range). We additionally show that integrating SceneSense-enhanced maps into our robotic exploration stack as a "drop-in" map improvement, utilizing an existing off-the-shelf planner, results in improvements in robustness and traversability time. Finally we show results of full exploration evaluations with our proposed system in two dissimilar environments and find that locally enhanced maps provide more consistent exploration results than maps constructed only from direct sensor measurements.
>
---
#### [new 010] DemoDiffusion: One-Shot Human Imitation using pre-trained Diffusion Policy
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人模仿学习任务，解决如何通过单次人类示范让机器人完成操作任务的问题。工作包括利用运动轨迹转换和预训练扩散策略优化动作，提升任务成功率。**

- **链接: [http://arxiv.org/pdf/2506.20668v1](http://arxiv.org/pdf/2506.20668v1)**

> **作者:** Sungjae Park; Homanga Bharadhwaj; Shubham Tulsiani
>
> **备注:** Preprint(17 pages). Under Review
>
> **摘要:** We propose DemoDiffusion, a simple and scalable method for enabling robots to perform manipulation tasks in natural environments by imitating a single human demonstration. Our approach is based on two key insights. First, the hand motion in a human demonstration provides a useful prior for the robot's end-effector trajectory, which we can convert into a rough open-loop robot motion trajectory via kinematic retargeting. Second, while this retargeted motion captures the overall structure of the task, it may not align well with plausible robot actions in-context. To address this, we leverage a pre-trained generalist diffusion policy to modify the trajectory, ensuring it both follows the human motion and remains within the distribution of plausible robot actions. Our approach avoids the need for online reinforcement learning or paired human-robot data, enabling robust adaptation to new tasks and scenes with minimal manual effort. Experiments in both simulation and real-world settings show that DemoDiffusion outperforms both the base policy and the retargeted trajectory, enabling the robot to succeed even on tasks where the pre-trained generalist policy fails entirely. Project page: https://demodiffusion.github.io/
>
---
#### [new 011] Consensus-Driven Uncertainty for Robotic Grasping based on RGB Perception
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人抓取任务，旨在解决深度姿态估计过自信的问题。通过训练轻量网络预测抓取成功率，提升抓取可靠性。**

- **链接: [http://arxiv.org/pdf/2506.20045v1](http://arxiv.org/pdf/2506.20045v1)**

> **作者:** Eric C. Joyce; Qianwen Zhao; Nathaniel Burgdorfer; Long Wang; Philippos Mordohai
>
> **摘要:** Deep object pose estimators are notoriously overconfident. A grasping agent that both estimates the 6-DoF pose of a target object and predicts the uncertainty of its own estimate could avoid task failure by choosing not to act under high uncertainty. Even though object pose estimation improves and uncertainty quantification research continues to make strides, few studies have connected them to the downstream task of robotic grasping. We propose a method for training lightweight, deep networks to predict whether a grasp guided by an image-based pose estimate will succeed before that grasp is attempted. We generate training data for our networks via object pose estimation on real images and simulated grasping. We also find that, despite high object variability in grasping trials, networks benefit from training on all objects jointly, suggesting that a diverse variety of objects can nevertheless contribute to the same goal.
>
---
#### [new 012] Multimodal Behaviour Trees for Robotic Laboratory Task Automation
- **分类: cs.RO**

- **简介: 该论文属于实验室机器人任务自动化，解决机器人执行任务的可靠性和安全性问题，通过多模态行为树方法提升任务执行的准确性和错误检测能力。**

- **链接: [http://arxiv.org/pdf/2506.20399v1](http://arxiv.org/pdf/2506.20399v1)**

> **作者:** Hatem Fakhruldeen; Arvind Raveendran Nambiar; Satheeshkumar Veeramani; Bonilkumar Vijaykumar Tailor; Hadi Beyzaee Juneghani; Gabriella Pizzuto; Andrew Ian Cooper
>
> **备注:** 7 pages, 5 figures, accepted and presented in ICRA 2025
>
> **摘要:** Laboratory robotics offer the capability to conduct experiments with a high degree of precision and reproducibility, with the potential to transform scientific research. Trivial and repeatable tasks; e.g., sample transportation for analysis and vial capping are well-suited for robots; if done successfully and reliably, chemists could contribute their efforts towards more critical research activities. Currently, robots can perform these tasks faster than chemists, but how reliable are they? Improper capping could result in human exposure to toxic chemicals which could be fatal. To ensure that robots perform these tasks as accurately as humans, sensory feedback is required to assess the progress of task execution. To address this, we propose a novel methodology based on behaviour trees with multimodal perception. Along with automating robotic tasks, this methodology also verifies the successful execution of the task, a fundamental requirement in safety-critical environments. The experimental evaluation was conducted on two lab tasks: sample vial capping and laboratory rack insertion. The results show high success rate, i.e., 88% for capping and 92% for insertion, along with strong error detection capabilities. This ultimately proves the robustness and reliability of our approach and that using multimodal behaviour trees should pave the way towards the next generation of robotic chemists.
>
---
#### [new 013] Personalized Mental State Evaluation in Human-Robot Interaction using Federated Learning
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互中的个性化心理状态评估任务，旨在通过联邦学习保护隐私的同时提升机器人对操作者压力水平的识别与适应能力。**

- **链接: [http://arxiv.org/pdf/2506.20212v1](http://arxiv.org/pdf/2506.20212v1)**

> **作者:** Andrea Bussolan; Oliver Avram; Andrea Pignata; Gianvito Urgese; Stefano Baraldo; Anna Valente
>
> **摘要:** With the advent of Industry 5.0, manufacturers are increasingly prioritizing worker well-being alongside mass customization. Stress-aware Human-Robot Collaboration (HRC) plays a crucial role in this paradigm, where robots must adapt their behavior to human mental states to improve collaboration fluency and safety. This paper presents a novel framework that integrates Federated Learning (FL) to enable personalized mental state evaluation while preserving user privacy. By leveraging physiological signals, including EEG, ECG, EDA, EMG, and respiration, a multimodal model predicts an operator's stress level, facilitating real-time robot adaptation. The FL-based approach allows distributed on-device training, ensuring data confidentiality while improving model generalization and individual customization. Results demonstrate that the deployment of an FL approach results in a global model with performance in stress prediction accuracy comparable to a centralized training approach. Moreover, FL allows for enhancing personalization, thereby optimizing human-robot interaction in industrial settings, while preserving data privacy. The proposed framework advances privacy-preserving, adaptive robotics to enhance workforce well-being in smart manufacturing.
>
---
#### [new 014] Hierarchical Reinforcement Learning and Value Optimization for Challenging Quadruped Locomotion
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于四足机器人运动控制任务，旨在解决复杂地形下的运动问题。通过分层强化学习框架提升运动性能与安全性。**

- **链接: [http://arxiv.org/pdf/2506.20036v1](http://arxiv.org/pdf/2506.20036v1)**

> **作者:** Jeremiah Coholich; Muhammad Ali Murtaza; Seth Hutchinson; Zsolt Kira
>
> **摘要:** We propose a novel hierarchical reinforcement learning framework for quadruped locomotion over challenging terrain. Our approach incorporates a two-layer hierarchy in which a high-level policy (HLP) selects optimal goals for a low-level policy (LLP). The LLP is trained using an on-policy actor-critic RL algorithm and is given footstep placements as goals. We propose an HLP that does not require any additional training or environment samples and instead operates via an online optimization process over the learned value function of the LLP. We demonstrate the benefits of this framework by comparing it with an end-to-end reinforcement learning (RL) approach. We observe improvements in its ability to achieve higher rewards with fewer collisions across an array of different terrains, including terrains more difficult than any encountered during training.
>
---
#### [new 015] Enhanced Robotic Navigation in Deformable Environments using Learning from Demonstration and Dynamic Modulation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决在可变形环境中高效安全导航的问题。通过融合示教学习与动态系统，实现对软硬障碍物的实时区分与路径调整。**

- **链接: [http://arxiv.org/pdf/2506.20376v1](http://arxiv.org/pdf/2506.20376v1)**

> **作者:** Lingyun Chen; Xinrui Zhao; Marcos P. S. Campanha; Alexander Wegener; Abdeldjallil Naceri; Abdalla Swikir; Sami Haddadin
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** This paper presents a novel approach for robot navigation in environments containing deformable obstacles. By integrating Learning from Demonstration (LfD) with Dynamical Systems (DS), we enable adaptive and efficient navigation in complex environments where obstacles consist of both soft and hard regions. We introduce a dynamic modulation matrix within the DS framework, allowing the system to distinguish between traversable soft regions and impassable hard areas in real-time, ensuring safe and flexible trajectory planning. We validate our method through extensive simulations and robot experiments, demonstrating its ability to navigate deformable environments. Additionally, the approach provides control over both trajectory and velocity when interacting with deformable objects, including at intersections, while maintaining adherence to the original DS trajectory and dynamically adapting to obstacles for smooth and reliable navigation.
>
---
#### [new 016] Real-Time Obstacle Avoidance Algorithms for Unmanned Aerial and Ground Vehicles
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于无人机与地面车的实时避障任务，旨在解决复杂环境中自主导航问题，提出2D/3D导航策略及协同控制方法。**

- **链接: [http://arxiv.org/pdf/2506.20311v1](http://arxiv.org/pdf/2506.20311v1)**

> **作者:** Jingwen Wei
>
> **摘要:** The growing use of mobile robots in sectors such as automotive, agriculture, and rescue operations reflects progress in robotics and autonomy. In unmanned aerial vehicles (UAVs), most research emphasizes visual SLAM, sensor fusion, and path planning. However, applying UAVs to search and rescue missions in disaster zones remains underexplored, especially for autonomous navigation. This report develops methods for real-time and secure UAV maneuvering in complex 3D environments, crucial during forest fires. Building upon past research, it focuses on designing navigation algorithms for unfamiliar and hazardous environments, aiming to improve rescue efficiency and safety through UAV-based early warning and rapid response. The work unfolds in phases. First, a 2D fusion navigation strategy is explored, initially for mobile robots, enabling safe movement in dynamic settings. This sets the stage for advanced features such as adaptive obstacle handling and decision-making enhancements. Next, a novel 3D reactive navigation strategy is introduced for collision-free movement in forest fire simulations, addressing the unique challenges of UAV operations in such scenarios. Finally, the report proposes a unified control approach that integrates UAVs and unmanned ground vehicles (UGVs) for coordinated rescue missions in forest environments. Each phase presents challenges, proposes control models, and validates them with mathematical and simulation-based evidence. The study offers practical value and academic insights for improving the role of UAVs in natural disaster rescue operations.
>
---
#### [new 017] Generating and Customizing Robotic Arm Trajectories using Neural Networks
- **分类: cs.RO; cs.AI; 68T40, 93C85, 70E60; I.2.9**

- **简介: 该论文属于机器人轨迹生成任务，旨在解决精准控制机械臂运动的问题。通过神经网络计算正向运动学并生成可定制的轨迹。**

- **链接: [http://arxiv.org/pdf/2506.20259v1](http://arxiv.org/pdf/2506.20259v1)**

> **作者:** Andrej Lúčny; Matilde Antonj; Carlo Mazzola; Hana Hornáčková; Igor Farkaš
>
> **备注:** The code is released at https://github.com/andylucny/nico2/tree/main/generate
>
> **摘要:** We introduce a neural network approach for generating and customizing the trajectory of a robotic arm, that guarantees precision and repeatability. To highlight the potential of this novel method, we describe the design and implementation of the technique and show its application in an experimental setting of cognitive robotics. In this scenario, the NICO robot was characterized by the ability to point to specific points in space with precise linear movements, increasing the predictability of the robotic action during its interaction with humans. To achieve this goal, the neural network computes the forward kinematics of the robot arm. By integrating it with a generator of joint angles, another neural network was developed and trained on an artificial dataset created from suitable start and end poses of the robotic arm. Through the computation of angular velocities, the robot was characterized by its ability to perform the movement, and the quality of its action was evaluated in terms of shape and accuracy. Thanks to its broad applicability, our approach successfully generates precise trajectories that could be customized in their shape and adapted to different settings.
>
---
#### [new 018] Learn to Position -- A Novel Meta Method for Robotic Positioning
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，旨在解决定位误差问题。提出一种无需视觉的元方法，通过交互反馈提升定位精度并实现自适应学习。**

- **链接: [http://arxiv.org/pdf/2506.20445v1](http://arxiv.org/pdf/2506.20445v1)**

> **作者:** Dongkun Wang; Junkai Zhao; Yunfei Teng; Jieyang Peng; Wenjing Xue; Xiaoming Tao
>
> **摘要:** Absolute positioning accuracy is a vital specification for robots. Achieving high position precision can be challenging due to the presence of various sources of errors. Meanwhile, accurately depicting these errors is difficult due to their stochastic nature. Vision-based methods are commonly integrated to guide robotic positioning, but their performance can be highly impacted by inevitable occlusions or adverse lighting conditions. Drawing on the aforementioned considerations, a vision-free, model-agnostic meta-method for compensating robotic position errors is proposed, which maximizes the probability of accurate robotic position via interactive feedback. Meanwhile, the proposed method endows the robot with the capability to learn and adapt to various position errors, which is inspired by the human's instinct for grasping under uncertainties. Furthermore, it is a self-learning and self-adaptive method able to accelerate the robotic positioning process as more examples are incorporated and learned. Empirical studies validate the effectiveness of the proposed method. As of the writing of this paper, the proposed meta search method has already been implemented in a robotic-based assembly line for odd-form electronic components.
>
---
#### [new 019] Why Robots Are Bad at Detecting Their Mistakes: Limitations of Miscommunication Detection in Human-Robot Dialogue
- **分类: cs.RO; cs.CL; cs.HC**

- **简介: 该论文属于人机对话任务，旨在解决机器人检测沟通失误的问题。研究通过分析多模态数据，评估机器学习模型在识别对话错误中的表现，发现机器人和人类均难以准确检测沟通失败。**

- **链接: [http://arxiv.org/pdf/2506.20268v1](http://arxiv.org/pdf/2506.20268v1)**

> **作者:** Ruben Janssens; Jens De Bock; Sofie Labat; Eva Verhelst; Veronique Hoste; Tony Belpaeme
>
> **备注:** Accepted at the 34th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN 2025)
>
> **摘要:** Detecting miscommunication in human-robot interaction is a critical function for maintaining user engagement and trust. While humans effortlessly detect communication errors in conversations through both verbal and non-verbal cues, robots face significant challenges in interpreting non-verbal feedback, despite advances in computer vision for recognizing affective expressions. This research evaluates the effectiveness of machine learning models in detecting miscommunications in robot dialogue. Using a multi-modal dataset of 240 human-robot conversations, where four distinct types of conversational failures were systematically introduced, we assess the performance of state-of-the-art computer vision models. After each conversational turn, users provided feedback on whether they perceived an error, enabling an analysis of the models' ability to accurately detect robot mistakes. Despite using state-of-the-art models, the performance barely exceeds random chance in identifying miscommunication, while on a dataset with more expressive emotional content, they successfully identified confused states. To explore the underlying cause, we asked human raters to do the same. They could also only identify around half of the induced miscommunications, similarly to our model. These results uncover a fundamental limitation in identifying robot miscommunications in dialogue: even when users perceive the induced miscommunication as such, they often do not communicate this to their robotic conversation partner. This knowledge can shape expectations of the performance of computer vision models and can help researchers to design better human-robot conversations by deliberately eliciting feedback where needed.
>
---
#### [new 020] A Review of Personalisation in Human-Robot Collaboration and Future Perspectives Towards Industry 5.0
- **分类: cs.RO**

- **简介: 本文综述了人机协作中的个性化研究，探讨其在工业5.0中的应用与挑战，旨在推动更人性化、适应性的协作系统发展。**

- **链接: [http://arxiv.org/pdf/2506.20447v1](http://arxiv.org/pdf/2506.20447v1)**

> **作者:** James Fant-Male; Roel Pieters
>
> **备注:** Accepted by the 2025 34th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN)
>
> **摘要:** The shift in research focus from Industry 4.0 to Industry 5.0 (I5.0) promises a human-centric workplace, with social and well-being values at the centre of technological implementation. Human-Robot Collaboration (HRC) is a core aspect of I5.0 development, with an increase in adaptive and personalised interactions and behaviours. This review investigates recent advancements towards personalised HRC, where user-centric adaption is key. There is a growing trend for adaptable HRC research, however there lacks a consistent and unified approach. The review highlights key research trends on which personal factors are considered, workcell and interaction design, and adaptive task completion. This raises various key considerations for future developments, particularly around the ethical and regulatory development of personalised systems, which are discussed in detail.
>
---
#### [new 021] PIMBS: Efficient Body Schema Learning for Musculoskeletal Humanoids with Physics-Informed Neural Networks
- **分类: cs.RO**

- **简介: 该论文属于机器人身体图式学习任务，旨在解决 musculoskeletal humanoids 身体结构复杂、数据获取困难的问题。通过引入物理信息神经网络，提升学习效率与精度。**

- **链接: [http://arxiv.org/pdf/2506.20343v1](http://arxiv.org/pdf/2506.20343v1)**

> **作者:** Kento Kawaharazuka; Takahiro Hattori; Keita Yoneda; Kei Okada
>
> **备注:** Accepted at IEEE Robotics and Automation Letters
>
> **摘要:** Musculoskeletal humanoids are robots that closely mimic the human musculoskeletal system, offering various advantages such as variable stiffness control, redundancy, and flexibility. However, their body structure is complex, and muscle paths often significantly deviate from geometric models. To address this, numerous studies have been conducted to learn body schema, particularly the relationships among joint angles, muscle tension, and muscle length. These studies typically rely solely on data collected from the actual robot, but this data collection process is labor-intensive, and learning becomes difficult when the amount of data is limited. Therefore, in this study, we propose a method that applies the concept of Physics-Informed Neural Networks (PINNs) to the learning of body schema in musculoskeletal humanoids, enabling high-accuracy learning even with a small amount of data. By utilizing not only data obtained from the actual robot but also the physical laws governing the relationship between torque and muscle tension under the assumption of correct joint structure, more efficient learning becomes possible. We apply the proposed method to both simulation and an actual musculoskeletal humanoid and discuss its effectiveness and characteristics.
>
---
#### [new 022] PSALM-V: Automating Symbolic Planning in Interactive Visual Environments with Large Language Models
- **分类: cs.RO; cs.CL**

- **简介: 该论文提出PSALM-V，解决视觉环境中符号动作语义自动学习问题，通过LLM生成计划和语义，提升任务完成率与效率。**

- **链接: [http://arxiv.org/pdf/2506.20097v1](http://arxiv.org/pdf/2506.20097v1)**

> **作者:** Wang Bill Zhu; Miaosen Chai; Ishika Singh; Robin Jia; Jesse Thomason
>
> **摘要:** We propose PSALM-V, the first autonomous neuro-symbolic learning system able to induce symbolic action semantics (i.e., pre- and post-conditions) in visual environments through interaction. PSALM-V bootstraps reliable symbolic planning without expert action definitions, using LLMs to generate heuristic plans and candidate symbolic semantics. Previous work has explored using large language models to generate action semantics for Planning Domain Definition Language (PDDL)-based symbolic planners. However, these approaches have primarily focused on text-based domains or relied on unrealistic assumptions, such as access to a predefined problem file, full observability, or explicit error messages. By contrast, PSALM-V dynamically infers PDDL problem files and domain action semantics by analyzing execution outcomes and synthesizing possible error explanations. The system iteratively generates and executes plans while maintaining a tree-structured belief over possible action semantics for each action, iteratively refining these beliefs until a goal state is reached. Simulated experiments of task completion in ALFRED demonstrate that PSALM-V increases the plan success rate from 37% (Claude-3.7) to 74% in partially observed setups. Results on two 2D game environments, RTFM and Overcooked-AI, show that PSALM-V improves step efficiency and succeeds in domain induction in multi-agent settings. PSALM-V correctly induces PDDL pre- and post-conditions for real-world robot BlocksWorld tasks, despite low-level manipulation failures from the robot.
>
---
#### [new 023] Behavior Foundation Model: Towards Next-Generation Whole-Body Control System of Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文属于人形机器人控制任务，旨在解决全肢体控制的挑战。通过行为基础模型实现技能学习与快速适应，提升机器人智能水平。**

- **链接: [http://arxiv.org/pdf/2506.20487v1](http://arxiv.org/pdf/2506.20487v1)**

> **作者:** Mingqi Yuan; Tao Yu; Wenqi Ge; Xiuyong Yao; Dapeng Li; Huijiang Wang; Jiayu Chen; Xin Jin; Bo Li; Hua Chen; Wei Zhang; Wenjun Zeng
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Humanoid robots are drawing significant attention as versatile platforms for complex motor control, human-robot interaction, and general-purpose physical intelligence. However, achieving efficient whole-body control (WBC) in humanoids remains a fundamental challenge due to sophisticated dynamics, underactuation, and diverse task requirements. While learning-based controllers have shown promise for complex tasks, their reliance on labor-intensive and costly retraining for new scenarios limits real-world applicability. To address these limitations, behavior(al) foundation models (BFMs) have emerged as a new paradigm that leverages large-scale pretraining to learn reusable primitive skills and behavioral priors, enabling zero-shot or rapid adaptation to a wide range of downstream tasks. In this paper, we present a comprehensive overview of BFMs for humanoid WBC, tracing their development across diverse pre-training pipelines. Furthermore, we discuss real-world applications, current limitations, urgent challenges, and future opportunities, positioning BFMs as a key approach toward scalable and general-purpose humanoid intelligence. Finally, we provide a curated and long-term list of BFM papers and projects to facilitate more subsequent research, which is available at https://github.com/yuanmingqi/awesome-bfm-papers.
>
---
#### [new 024] HRIBench: Benchmarking Vision-Language Models for Real-Time Human Perception in Human-Robot Interaction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人机交互中的视觉-语言模型评估任务，旨在解决实时人类感知的性能与延迟问题。研究构建了HRIBench基准，评估模型在五类关键任务中的表现。**

- **链接: [http://arxiv.org/pdf/2506.20566v1](http://arxiv.org/pdf/2506.20566v1)**

> **作者:** Zhonghao Shi; Enyu Zhao; Nathaniel Dennler; Jingzhen Wang; Xinyang Xu; Kaleen Shrestha; Mengxue Fu; Daniel Seita; Maja Matarić
>
> **备注:** Accepted to the 19th International Symposium on Experimental Robotics (ISER 2025)
>
> **摘要:** Real-time human perception is crucial for effective human-robot interaction (HRI). Large vision-language models (VLMs) offer promising generalizable perceptual capabilities but often suffer from high latency, which negatively impacts user experience and limits VLM applicability in real-world scenarios. To systematically study VLM capabilities in human perception for HRI and performance-latency trade-offs, we introduce HRIBench, a visual question-answering (VQA) benchmark designed to evaluate VLMs across a diverse set of human perceptual tasks critical for HRI. HRIBench covers five key domains: (1) non-verbal cue understanding, (2) verbal instruction understanding, (3) human-robot object relationship understanding, (4) social navigation, and (5) person identification. To construct HRIBench, we collected data from real-world HRI environments to curate questions for non-verbal cue understanding, and leveraged publicly available datasets for the remaining four domains. We curated 200 VQA questions for each domain, resulting in a total of 1000 questions for HRIBench. We then conducted a comprehensive evaluation of both state-of-the-art closed-source and open-source VLMs (N=11) on HRIBench. Our results show that, despite their generalizability, current VLMs still struggle with core perceptual capabilities essential for HRI. Moreover, none of the models within our experiments demonstrated a satisfactory performance-latency trade-off suitable for real-time deployment, underscoring the need for future research on developing smaller, low-latency VLMs with improved human perception capabilities. HRIBench and our results can be found in this Github repository: https://github.com/interaction-lab/HRIBench.
>
---
#### [new 025] A Computationally Aware Multi Objective Framework for Camera LiDAR Calibration
- **分类: cs.RO**

- **简介: 该论文属于传感器标定任务，解决LiDAR与相机的外参校准问题，通过多目标优化框架降低误差和计算成本。**

- **链接: [http://arxiv.org/pdf/2506.20636v1](http://arxiv.org/pdf/2506.20636v1)**

> **作者:** Venkat Karramreddy; Rangarajan Ramanujam
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** Accurate extrinsic calibration between LiDAR and camera sensors is important for reliable perception in autonomous systems. In this paper, we present a novel multi-objective optimization framework that jointly minimizes the geometric alignment error and computational cost associated with camera-LiDAR calibration. We optimize two objectives: (1) error between projected LiDAR points and ground-truth image edges, and (2) a composite metric for computational cost reflecting runtime and resource usage. Using the NSGA-II \cite{deb2002nsga2} evolutionary algorithm, we explore the parameter space defined by 6-DoF transformations and point sampling rates, yielding a well-characterized Pareto frontier that exposes trade-offs between calibration fidelity and resource efficiency. Evaluations are conducted on the KITTI dataset using its ground-truth extrinsic parameters for validation, with results verified through both multi-objective and constrained single-objective baselines. Compared to existing gradient-based and learned calibration methods, our approach demonstrates interpretable, tunable performance with lower deployment overhead. Pareto-optimal configurations are further analyzed for parameter sensitivity and innovation insights. A preference-based decision-making strategy selects solutions from the Pareto knee region to suit the constraints of the embedded system. The robustness of calibration is tested across variable edge-intensity weighting schemes, highlighting optimal balance points. Although real-time deployment on embedded platforms is deferred to future work, this framework establishes a scalable and transparent method for calibration under realistic misalignment and resource-limited conditions, critical for long-term autonomy, particularly in SAE L3+ vehicles receiving OTA updates.
>
---
#### [new 026] SPARK: Graph-Based Online Semantic Integration System for Robot Task Planning
- **分类: cs.RO**

- **简介: 该论文属于机器人任务规划领域，解决动态环境中语义信息在线更新问题。提出SPARK系统，通过图表示增强机器人任务执行能力。**

- **链接: [http://arxiv.org/pdf/2506.20394v1](http://arxiv.org/pdf/2506.20394v1)**

> **作者:** Mimo Shirasaka; Yuya Ikeda; Tatsuya Matsushima; Yutaka Matsuo; Yusuke Iwasawa
>
> **摘要:** The ability to update information acquired through various means online during task execution is crucial for a general-purpose service robot. This information includes geometric and semantic data. While SLAM handles geometric updates on 2D maps or 3D point clouds, online updates of semantic information remain unexplored. We attribute the challenge to the online scene graph representation, for its utility and scalability. Building on previous works regarding offline scene graph representations, we study online graph representations of semantic information in this work. We introduce SPARK: Spatial Perception and Robot Knowledge Integration. This framework extracts semantic information from environment-embedded cues and updates the scene graph accordingly, which is then used for subsequent task planning. We demonstrate that graph representations of spatial relationships enhance the robot system's ability to perform tasks in dynamic environments and adapt to unconventional spatial cues, like gestures.
>
---
#### [new 027] Near Time-Optimal Hybrid Motion Planning for Timber Cranes
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决液压吊车在避障和时间最优路径规划中的问题，提出改进的随机轨迹优化算法与局部规划器结合的方法。**

- **链接: [http://arxiv.org/pdf/2506.20314v1](http://arxiv.org/pdf/2506.20314v1)**

> **作者:** Marc-Philip Ecker; Bernhard Bischof; Minh Nhat Vu; Christoph Fröhlich; Tobias Glück; Wolfgang Kemmetmüller
>
> **备注:** Accepted at ICRA 2025
>
> **摘要:** Efficient, collision-free motion planning is essential for automating large-scale manipulators like timber cranes. They come with unique challenges such as hydraulic actuation constraints and passive joints-factors that are seldom addressed by current motion planning methods. This paper introduces a novel approach for time-optimal, collision-free hybrid motion planning for a hydraulically actuated timber crane with passive joints. We enhance the via-point-based stochastic trajectory optimization (VP-STO) algorithm to include pump flow rate constraints and develop a novel collision cost formulation to improve robustness. The effectiveness of the enhanced VP-STO as an optimal single-query global planner is validated by comparison with an informed RRT* algorithm using a time-optimal path parameterization (TOPP). The overall hybrid motion planning is formed by combination with a gradient-based local planner that is designed to follow the global planner's reference and to systematically consider the passive joint dynamics for both collision avoidance and sway damping.
>
---
#### [new 028] Robust Embodied Self-Identification of Morphology in Damaged Multi-Legged Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人自适应任务，解决多足机器人腿部损伤后的自我建模问题。通过IMU数据和FFT滤波，实现损伤检测与模型更新。**

- **链接: [http://arxiv.org/pdf/2506.19984v1](http://arxiv.org/pdf/2506.19984v1)**

> **作者:** Sahand Farghdani; Mili Patel; Robin Chhabra
>
> **摘要:** Multi-legged robots (MLRs) are vulnerable to leg damage during complex missions, which can impair their performance. This paper presents a self-modeling and damage identification algorithm that enables autonomous adaptation to partial or complete leg loss using only data from a low-cost IMU. A novel FFT-based filter is introduced to address time-inconsistent signals, improving damage detection by comparing body orientation between the robot and its model. The proposed method identifies damaged legs and updates the robot's model for integration into its control system. Experiments on uneven terrain validate its robustness and computational efficiency.
>
---
#### [new 029] Lightweight Multi-Frame Integration for Robust YOLO Object Detection in Videos
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视频目标检测任务，旨在解决单帧检测在动态场景中的鲁棒性问题。通过多帧融合提升检测效果，同时保持模型轻量化和实时性。**

- **链接: [http://arxiv.org/pdf/2506.20550v1](http://arxiv.org/pdf/2506.20550v1)**

> **作者:** Yitong Quan; Benjamin Kiefer; Martin Messmer; Andreas Zell
>
> **备注:** Submitted to ECMR 2025
>
> **摘要:** Modern image-based object detection models, such as YOLOv7, primarily process individual frames independently, thus ignoring valuable temporal context naturally present in videos. Meanwhile, existing video-based detection methods often introduce complex temporal modules, significantly increasing model size and computational complexity. In practical applications such as surveillance and autonomous driving, transient challenges including motion blur, occlusions, and abrupt appearance changes can severely degrade single-frame detection performance. To address these issues, we propose a straightforward yet highly effective strategy: stacking multiple consecutive frames as input to a YOLO-based detector while supervising only the output corresponding to a single target frame. This approach leverages temporal information with minimal modifications to existing architectures, preserving simplicity, computational efficiency, and real-time inference capability. Extensive experiments on the challenging MOT20Det and our BOAT360 datasets demonstrate that our method improves detection robustness, especially for lightweight models, effectively narrowing the gap between compact and heavy detection networks. Additionally, we contribute the BOAT360 benchmark dataset, comprising annotated fisheye video sequences captured from a boat, to support future research in multi-frame video object detection in challenging real-world scenarios.
>
---
#### [new 030] Task Allocation of UAVs for Monitoring Missions via Hardware-in-the-Loop Simulation and Experimental Validation
- **分类: eess.SY; cs.MA; cs.RO; cs.SY**

- **简介: 该论文研究无人机任务分配问题，通过遗传算法与2-Opt优化方法提升工业监测效率，并通过HIL仿真验证方案有效性。**

- **链接: [http://arxiv.org/pdf/2506.20626v1](http://arxiv.org/pdf/2506.20626v1)**

> **作者:** Hamza Chakraa; François Guérin; Edouard Leclercq; Dimitri Lefebvre
>
> **摘要:** This study addresses the optimisation of task allocation for Unmanned Aerial Vehicles (UAVs) within industrial monitoring missions. The proposed methodology integrates a Genetic Algorithms (GA) with a 2-Opt local search technique to obtain a high-quality solution. Our approach was experimentally validated in an industrial zone to demonstrate its efficacy in real-world scenarios. Also, a Hardware-in-the-loop (HIL) simulator for the UAVs team is introduced. Moreover, insights about the correlation between the theoretical cost function and the actual battery consumption and time of flight are deeply analysed. Results show that the considered costs for the optimisation part of the problem closely correlate with real-world data, confirming the practicality of the proposed approach.
>
---
#### [new 031] Learning-Based Distance Estimation for 360° Single-Sensor Setups
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于单目距离估计任务，旨在解决360°摄像头在复杂环境中的距离估算问题。通过神经网络方法，直接从原始图像学习距离信息，提升准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.20586v1](http://arxiv.org/pdf/2506.20586v1)**

> **作者:** Yitong Quan; Benjamin Kiefer; Martin Messmer; Andreas Zell
>
> **备注:** Submitted to ECMR 2025
>
> **摘要:** Accurate distance estimation is a fundamental challenge in robotic perception, particularly in omnidirectional imaging, where traditional geometric methods struggle with lens distortions and environmental variability. In this work, we propose a neural network-based approach for monocular distance estimation using a single 360{\deg} fisheye lens camera. Unlike classical trigonometric techniques that rely on precise lens calibration, our method directly learns and infers the distance of objects from raw omnidirectional inputs, offering greater robustness and adaptability across diverse conditions. We evaluate our approach on three 360{\deg} datasets (LOAF, ULM360, and a newly captured dataset Boat360), each representing distinct environmental and sensor setups. Our experimental results demonstrate that the proposed learning-based model outperforms traditional geometry-based methods and other learning baselines in both accuracy and robustness. These findings highlight the potential of deep learning for real-time omnidirectional distance estimation, making our approach particularly well-suited for low-cost applications in robotics, autonomous navigation, and surveillance.
>
---
## 更新

#### [replaced 001] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.12636v4](http://arxiv.org/pdf/2504.12636v4)**

> **作者:** Rongtao Xu; Jian Zhang; Minghao Guo; Youpeng Wen; Haoting Yang; Min Lin; Jianzheng Huang; Zhe Li; Kaidong Zhang; Liqiong Wang; Yuxuan Kuang; Meng Cao; Feng Zheng; Xiaodan Liang
>
> **摘要:** Robotic manipulation faces critical challenges in understanding spatial affordances--the "where" and "how" of object interactions--essential for complex manipulation tasks like wiping a board or stacking objects. Existing methods, including modular-based and end-to-end approaches, often lack robust spatial reasoning capabilities. Unlike recent point-based and flow-based affordance methods that focus on dense spatial representations or trajectory modeling, we propose A0, a hierarchical affordance-aware diffusion model that decomposes manipulation tasks into high-level spatial affordance understanding and low-level action execution. A0 leverages the Embodiment-Agnostic Affordance Representation, which captures object-centric spatial affordances by predicting contact points and post-contact trajectories. A0 is pre-trained on 1 million contact points data and fine-tuned on annotated trajectories, enabling generalization across platforms. Key components include Position Offset Attention for motion-aware feature extraction and a Spatial Information Aggregation Layer for precise coordinate mapping. The model's output is executed by the action execution module. Experiments on multiple robotic systems (Franka, Kinova, Realman, and Dobot) demonstrate A0's superior performance in complex tasks, showcasing its efficiency, flexibility, and real-world applicability.
>
---
#### [replaced 002] FGS-SLAM: Fourier-based Gaussian Splatting for Real-time SLAM with Sparse and Dense Map Fusion
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.01109v2](http://arxiv.org/pdf/2503.01109v2)**

> **作者:** Yansong Xu; Junlin Li; Wei Zhang; Siyu Chen; Shengyong Zhang; Yuquan Leng; Weijia Zhou
>
> **摘要:** 3D gaussian splatting has advanced simultaneous localization and mapping (SLAM) technology by enabling real-time positioning and the construction of high-fidelity maps. However, the uncertainty in gaussian position and initialization parameters introduces challenges, often requiring extensive iterative convergence and resulting in redundant or insufficient gaussian representations. To address this, we introduce a novel adaptive densification method based on Fourier frequency domain analysis to establish gaussian priors for rapid convergence. Additionally, we propose constructing independent and unified sparse and dense maps, where a sparse map supports efficient tracking via Generalized Iterative Closest Point (GICP) and a dense map creates high-fidelity visual representations. This is the first SLAM system leveraging frequency domain analysis to achieve high-quality gaussian mapping in real-time. Experimental results demonstrate an average frame rate of 36 FPS on Replica and TUM RGB-D datasets, achieving competitive accuracy in both localization and mapping.
>
---
#### [replaced 003] BEVPlace++: Fast, Robust, and Lightweight LiDAR Global Localization for Unmanned Ground Vehicles
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2408.01841v3](http://arxiv.org/pdf/2408.01841v3)**

> **作者:** Lun Luo; Si-Yuan Cao; Xiaorui Li; Jintao Xu; Rui Ai; Zhu Yu; Xieyuanli Chen
>
> **备注:** Accepted to IEEE Transactions on Robotics
>
> **摘要:** This article introduces BEVPlace++, a novel, fast, and robust LiDAR global localization method for unmanned ground vehicles. It uses lightweight convolutional neural networks (CNNs) on Bird's Eye View (BEV) image-like representations of LiDAR data to achieve accurate global localization through place recognition, followed by 3-DoF pose estimation. Our detailed analyses reveal an interesting fact that CNNs are inherently effective at extracting distinctive features from LiDAR BEV images. Remarkably, keypoints of two BEV images with large translations can be effectively matched using CNN-extracted features. Building on this insight, we design a Rotation Equivariant Module (REM) to obtain distinctive features while enhancing robustness to rotational changes. A Rotation Equivariant and Invariant Network (REIN) is then developed by cascading REM and a descriptor generator, NetVLAD, to sequentially generate rotation equivariant local features and rotation invariant global descriptors. The global descriptors are used first to achieve robust place recognition, and then local features are used for accurate pose estimation. \revise{Experimental results on seven public datasets and our UGV platform demonstrate that BEVPlace++, even when trained on a small dataset (3000 frames of KITTI) only with place labels, generalizes well to unseen environments, performs consistently across different days and years, and adapts to various types of LiDAR scanners.} BEVPlace++ achieves state-of-the-art performance in multiple tasks, including place recognition, loop closure detection, and global localization. Additionally, BEVPlace++ is lightweight, runs in real-time, and does not require accurate pose supervision, making it highly convenient for deployment. \revise{The source codes are publicly available at https://github.com/zjuluolun/BEVPlace2.
>
---
#### [replaced 004] COBRA-PPM: A Causal Bayesian Reasoning Architecture Using Probabilistic Programming for Robot Manipulation Under Uncertainty
- **分类: cs.RO; cs.AI; cs.LG; stat.AP; I.2.9; I.2.8; I.2.3; G.3; I.2.6; I.6.8; I.2.4; I.2.10**

- **链接: [http://arxiv.org/pdf/2403.14488v3](http://arxiv.org/pdf/2403.14488v3)**

> **作者:** Ricardo Cannizzaro; Michael Groom; Jonathan Routley; Robert Osazuwa Ness; Lars Kunze
>
> **备注:** 8 pages, 7 figures, accepted to the 2025 IEEE European Conference on Mobile Robots (ECMR 2025)
>
> **摘要:** Manipulation tasks require robots to reason about cause and effect when interacting with objects. Yet, many data-driven approaches lack causal semantics and thus only consider correlations. We introduce COBRA-PPM, a novel causal Bayesian reasoning architecture that combines causal Bayesian networks and probabilistic programming to perform interventional inference for robot manipulation under uncertainty. We demonstrate its capabilities through high-fidelity Gazebo-based experiments on an exemplar block stacking task, where it predicts manipulation outcomes with high accuracy (Pred Acc: 88.6%) and performs greedy next-best action selection with a 94.2% task success rate. We further demonstrate sim2real transfer on a domestic robot, showing effectiveness in handling real-world uncertainty from sensor noise and stochastic actions. Our generalised and extensible framework supports a wide range of manipulation scenarios and lays a foundation for future work at the intersection of robotics and causality.
>
---
#### [replaced 005] Physics-informed Imitative Reinforcement Learning for Real-world Driving
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.02508v3](http://arxiv.org/pdf/2407.02508v3)**

> **作者:** Hang Zhou; Yihao Qin; Dan Xu; Yiding Ji
>
> **摘要:** Recent advances in imitative reinforcement learning (IRL) have considerably enhanced the ability of autonomous agents to assimilate expert demonstrations, leading to rapid skill acquisition in a range of demanding tasks. However, such learning-based agents face significant challenges when transferring knowledge to highly dynamic closed-loop environments. Their performance is significantly impacted by the conflicting optimization objectives of imitation learning (IL) and reinforcement learning (RL), sample inefficiency, and the complexity of uncovering the hidden world model and physics. To address this challenge, we propose a physics-informed IRL that is entirely data-driven. It leverages both expert demonstration data and exploratory data with a joint optimization objective, allowing the underlying physical principles of vehicle dynamics to emerge naturally from the training process. The performance is evaluated through empirical experiments and results exceed popular IL, RL and IRL algorithms in closed-loop settings on Waymax benchmark. Our approach exhibits 37.8% reduction in collision rate and 22.2% reduction in off-road rate compared to the baseline method.
>
---
#### [replaced 006] AnchorDP3: 3D Affordance Guided Sparse Diffusion Policy for Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.19269v2](http://arxiv.org/pdf/2506.19269v2)**

> **作者:** Ziyan Zhao; Ke Fan; He-Yang Xu; Ning Qiao; Bo Peng; Wenlong Gao; Dongjiang Li; Hui Shen
>
> **摘要:** We present AnchorDP3, a diffusion policy framework for dual-arm robotic manipulation that achieves state-of-the-art performance in highly randomized environments. AnchorDP3 integrates three key innovations: (1) Simulator-Supervised Semantic Segmentation, using rendered ground truth to explicitly segment task-critical objects within the point cloud, which provides strong affordance priors; (2) Task-Conditioned Feature Encoders, lightweight modules processing augmented point clouds per task, enabling efficient multi-task learning through a shared diffusion-based action expert; (3) Affordance-Anchored Keypose Diffusion with Full State Supervision, replacing dense trajectory prediction with sparse, geometrically meaningful action anchors, i.e., keyposes such as pre-grasp pose, grasp pose directly anchored to affordances, drastically simplifying the prediction space; the action expert is forced to predict both robot joint angles and end-effector poses simultaneously, which exploits geometric consistency to accelerate convergence and boost accuracy. Trained on large-scale, procedurally generated simulation data, AnchorDP3 achieves a 98.7% average success rate in the RoboTwin benchmark across diverse tasks under extreme randomization of objects, clutter, table height, lighting, and backgrounds. This framework, when integrated with the RoboTwin real-to-sim pipeline, has the potential to enable fully autonomous generation of deployable visuomotor policies from only scene and instruction, totally eliminating human demonstrations from learning manipulation skills.
>
---
#### [replaced 007] IKDiffuser: A Generative Inverse Kinematics Solver for Multi-arm Robots via Diffusion Model
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.13087v3](http://arxiv.org/pdf/2506.13087v3)**

> **作者:** Zeyu Zhang; Ziyuan Jiao
>
> **备注:** under review
>
> **摘要:** Solving Inverse Kinematics (IK) problems is fundamental to robotics, but has primarily been successful with single serial manipulators. For multi-arm robotic systems, IK remains challenging due to complex self-collisions, coupled joints, and high-dimensional redundancy. These complexities make traditional IK solvers slow, prone to failure, and lacking in solution diversity. In this paper, we present IKDiffuser, a diffusion-based model designed for fast and diverse IK solution generation for multi-arm robotic systems. IKDiffuser learns the joint distribution over the configuration space, capturing complex dependencies and enabling seamless generalization to multi-arm robotic systems of different structures. In addition, IKDiffuser can incorporate additional objectives during inference without retraining, offering versatility and adaptability for task-specific requirements. In experiments on 6 different multi-arm systems, the proposed IKDiffuser achieves superior solution accuracy, precision, diversity, and computational efficiency compared to existing solvers. The proposed IKDiffuser framework offers a scalable, unified approach to solving multi-arm IK problems, facilitating the potential of multi-arm robotic systems in real-time manipulation tasks.
>
---
#### [replaced 008] FORTE: Tactile Force and Slip Sensing on Compliant Fingers for Delicate Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.18960v2](http://arxiv.org/pdf/2506.18960v2)**

> **作者:** Siqi Shang; Mingyo Seo; Yuke Zhu; Lillian Chin
>
> **摘要:** Handling delicate and fragile objects remains a major challenge for robotic manipulation, especially for rigid parallel grippers. While the simplicity and versatility of parallel grippers have led to widespread adoption, these grippers are limited by their heavy reliance on visual feedback. Tactile sensing and soft robotics can add responsiveness and compliance. However, existing methods typically involve high integration complexity or suffer from slow response times. In this work, we introduce FORTE, a tactile sensing system embedded in compliant gripper fingers. FORTE uses 3D-printed fin-ray grippers with internal air channels to provide low-latency force and slip feedback. FORTE applies just enough force to grasp objects without damaging them, while remaining easy to fabricate and integrate. We find that FORTE can accurately estimate grasping forces from 0-8 N with an average error of 0.2 N, and detect slip events within 100 ms of occurring. We demonstrate FORTE's ability to grasp a wide range of slippery, fragile, and deformable objects. In particular, FORTE grasps fragile objects like raspberries and potato chips with a 98.6% success rate, and achieves 93% accuracy in detecting slip events. These results highlight FORTE's potential as a robust and practical solution for enabling delicate robotic manipulation. Project page: https://merge-lab.github.io/FORTE
>
---
#### [replaced 009] Neural Graph Map: Dense Mapping with Efficient Loop Closure Integration
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2405.03633v2](http://arxiv.org/pdf/2405.03633v2)**

> **作者:** Leonard Bruns; Jun Zhang; Patric Jensfelt
>
> **备注:** WACV 2025, Project page: https://kth-rpl.github.io/neural_graph_mapping/
>
> **摘要:** Neural field-based SLAM methods typically employ a single, monolithic field as their scene representation. This prevents efficient incorporation of loop closure constraints and limits scalability. To address these shortcomings, we propose a novel RGB-D neural mapping framework in which the scene is represented by a collection of lightweight neural fields which are dynamically anchored to the pose graph of a sparse visual SLAM system. Our approach shows the ability to integrate large-scale loop closures, while requiring only minimal reintegration. Furthermore, we verify the scalability of our approach by demonstrating successful building-scale mapping taking multiple loop closures into account during the optimization, and show that our method outperforms existing state-of-the-art approaches on large scenes in terms of quality and runtime. Our code is available open-source at https://github.com/KTH-RPL/neural_graph_mapping.
>
---
#### [replaced 010] Teacher Motion Priors: Enhancing Robot Locomotion over Challenging Terrain
- **分类: cs.RO; cs.AI; 68T40**

- **链接: [http://arxiv.org/pdf/2504.10390v2](http://arxiv.org/pdf/2504.10390v2)**

> **作者:** Fangcheng Jin; Yuqi Wang; Peixin Ma; Guodong Yang; Pan Zhao; En Li; Zhengtao Zhang
>
> **备注:** 8 pages, 6 figures, 6 tables, IROS 2025
>
> **摘要:** Achieving robust locomotion on complex terrains remains a challenge due to high dimensional control and environmental uncertainties. This paper introduces a teacher prior framework based on the teacher student paradigm, integrating imitation and auxiliary task learning to improve learning efficiency and generalization. Unlike traditional paradigms that strongly rely on encoder-based state embeddings, our framework decouples the network design, simplifying the policy network and deployment. A high performance teacher policy is first trained using privileged information to acquire generalizable motion skills. The teacher's motion distribution is transferred to the student policy, which relies only on noisy proprioceptive data, via a generative adversarial mechanism to mitigate performance degradation caused by distributional shifts. Additionally, auxiliary task learning enhances the student policy's feature representation, speeding up convergence and improving adaptability to varying terrains. The framework is validated on a humanoid robot, showing a great improvement in locomotion stability on dynamic terrains and significant reductions in development costs. This work provides a practical solution for deploying robust locomotion strategies in humanoid robots.
>
---
#### [replaced 011] Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.07744v2](http://arxiv.org/pdf/2506.07744v2)**

> **作者:** Seungho Baek; Taegeon Park; Jongchan Park; Seungjun Oh; Yusung Kim
>
> **备注:** ICML 2025
>
> **摘要:** Existing offline hierarchical reinforcement learning methods rely on high-level policy learning to generate subgoal sequences. However, their efficiency degrades as task horizons increase, and they lack effective strategies for stitching useful state transitions across different trajectories. We propose Graph-Assisted Stitching (GAS), a novel framework that formulates subgoal selection as a graph search problem rather than learning an explicit high-level policy. By embedding states into a Temporal Distance Representation (TDR) space, GAS clusters semantically similar states from different trajectories into unified graph nodes, enabling efficient transition stitching. A shortest-path algorithm is then applied to select subgoal sequences within the graph, while a low-level policy learns to reach the subgoals. To improve graph quality, we introduce the Temporal Efficiency (TE) metric, which filters out noisy or inefficient transition states, significantly enhancing task performance. GAS outperforms prior offline HRL methods across locomotion, navigation, and manipulation tasks. Notably, in the most stitching-critical task, it achieves a score of 88.3, dramatically surpassing the previous state-of-the-art score of 1.0. Our source code is available at: https://github.com/qortmdgh4141/GAS.
>
---
#### [replaced 012] Proximal Control of UAVs with Federated Learning for Human-Robot Collaborative Domains
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.02863v2](http://arxiv.org/pdf/2412.02863v2)**

> **作者:** Lucas Nogueira Nobrega; Ewerton de Oliveira; Martin Saska; Tiago Nascimento
>
> **备注:** version 2
>
> **摘要:** The human-robot interaction (HRI) is a growing area of research. In HRI, complex command (action) classification is still an open problem that usually prevents the real applicability of such a technique. The literature presents some works that use neural networks to detect these actions. However, occlusion is still a major issue in HRI, especially when using uncrewed aerial vehicles (UAVs), since, during the robot's movement, the human operator is often out of the robot's field of view. Furthermore, in multi-robot scenarios, distributed training is also an open problem. In this sense, this work proposes an action recognition and control approach based on Long Short-Term Memory (LSTM) Deep Neural Networks with two layers in association with three densely connected layers and Federated Learning (FL) embedded in multiple drones. The FL enabled our approach to be trained in a distributed fashion, i.e., access to data without the need for cloud or other repositories, which facilitates the multi-robot system's learning. Furthermore, our multi-robot approach results also prevented occlusion situations, with experiments with real robots achieving an accuracy greater than 96%.
>
---
#### [replaced 013] Mamba Policy: Towards Efficient 3D Diffusion Policy with Hybrid Selective State Models
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.07163v2](http://arxiv.org/pdf/2409.07163v2)**

> **作者:** Jiahang Cao; Qiang Zhang; Jingkai Sun; Jiaxu Wang; Hao Cheng; Yulin Li; Jun Ma; Kun Wu; Zhiyuan Xu; Yecheng Shao; Wen Zhao; Gang Han; Yijie Guo; Renjing Xu
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Diffusion models have been widely employed in the field of 3D manipulation due to their efficient capability to learn distributions, allowing for precise prediction of action trajectories. However, diffusion models typically rely on large parameter UNet backbones as policy networks, which can be challenging to deploy on resource-constrained devices. Recently, the Mamba model has emerged as a promising solution for efficient modeling, offering low computational complexity and strong performance in sequence modeling. In this work, we propose the Mamba Policy, a lighter but stronger policy that reduces the parameter count by over 80% compared to the original policy network while achieving superior performance. Specifically, we introduce the XMamba Block, which effectively integrates input information with conditional features and leverages a combination of Mamba and Attention mechanisms for deep feature extraction. Extensive experiments demonstrate that the Mamba Policy excels on the Adroit, Dexart, and MetaWorld datasets, requiring significantly fewer computational resources. Additionally, we highlight the Mamba Policy's enhanced robustness in long-horizon scenarios compared to baseline methods and explore the performance of various Mamba variants within the Mamba Policy framework. Real-world experiments are also conducted to further validate its effectiveness. Our open-source project page can be found at https://andycao1125.github.io/mamba_policy/.
>
---
#### [replaced 014] EvDetMAV: Generalized MAV Detection from Moving Event Cameras
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.19416v2](http://arxiv.org/pdf/2506.19416v2)**

> **作者:** Yin Zhang; Zian Ning; Xiaoyu Zhang; Shiliang Guo; Peidong Liu; Shiyu Zhao
>
> **备注:** 8 pages, 7 figures. This paper is accepted by IEEE Robotics and Automation Letters
>
> **摘要:** Existing micro aerial vehicle (MAV) detection methods mainly rely on the target's appearance features in RGB images, whose diversity makes it difficult to achieve generalized MAV detection. We notice that different types of MAVs share the same distinctive features in event streams due to their high-speed rotating propellers, which are hard to see in RGB images. This paper studies how to detect different types of MAVs from an event camera by fully exploiting the features of propellers in the original event stream. The proposed method consists of three modules to extract the salient and spatio-temporal features of the propellers while filtering out noise from background objects and camera motion. Since there are no existing event-based MAV datasets, we introduce a novel MAV dataset for the community. This is the first event-based MAV dataset comprising multiple scenarios and different types of MAVs. Without training, our method significantly outperforms state-of-the-art methods and can deal with challenging scenarios, achieving a precision rate of 83.0\% (+30.3\%) and a recall rate of 81.5\% (+36.4\%) on the proposed testing dataset. The dataset and code are available at: https://github.com/WindyLab/EvDetMAV.
>
---
