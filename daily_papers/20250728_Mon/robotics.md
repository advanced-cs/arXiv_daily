# 机器人 cs.RO

- **最新发布 25 篇**

- **更新 23 篇**

## 最新发布

#### [new 001] A Fast and Light-weight Non-Iterative Visual Odometry with RGB-D Cameras
- **分类: cs.RO**

- **简介: 该论文属于视觉里程计任务，旨在解决RGB-D相机在低纹理环境下的实时位姿估计问题。现有方法依赖特征提取和迭代优化，计算量大。论文提出一种非迭代、轻量级方法，通过分离旋转和平移估计，利用平面特征和核互相关器提升效率，实现了71Hz的实时性能。**

- **链接: [http://arxiv.org/pdf/2507.18886v1](http://arxiv.org/pdf/2507.18886v1)**

> **作者:** Zheng Yang; Kuan Xu; Shenghai Yuan; Lihua Xie
>
> **摘要:** In this paper, we introduce a novel approach for efficiently estimating the 6-Degree-of-Freedom (DoF) robot pose with a decoupled, non-iterative method that capitalizes on overlapping planar elements. Conventional RGB-D visual odometry(RGBD-VO) often relies on iterative optimization solvers to estimate pose and involves a process of feature extraction and matching. This results in significant computational burden and time delays. To address this, our innovative method for RGBD-VO separates the estimation of rotation and translation. Initially, we exploit the overlaid planar characteristics within the scene to calculate the rotation matrix. Following this, we utilize a kernel cross-correlator (KCC) to ascertain the translation. By sidestepping the resource-intensive iterative optimization and feature extraction and alignment procedures, our methodology offers improved computational efficacy, achieving a performance of 71Hz on a lower-end i5 CPU. When the RGBD-VO does not rely on feature points, our technique exhibits enhanced performance in low-texture degenerative environments compared to state-of-the-art methods.
>
---
#### [new 002] Towards Multimodal Social Conversations with Robots: Using Vision-Language Models
- **分类: cs.RO; cs.CL; cs.HC**

- **简介: 该论文探讨如何使社交机器人通过视觉-语言模型实现多模态社交对话。任务是提升机器人在开放域对话中的社交能力，解决其缺乏多模态交互能力的问题。工作包括分析多模态系统需求、提出使用视觉-语言模型的方案、讨论技术挑战与评估方法。**

- **链接: [http://arxiv.org/pdf/2507.19196v1](http://arxiv.org/pdf/2507.19196v1)**

> **作者:** Ruben Janssens; Tony Belpaeme
>
> **备注:** Submitted to the workshop "Human - Foundation Models Interaction: A Focus On Multimodal Information" (FoMo-HRI) at IEEE RO-MAN 2025
>
> **摘要:** Large language models have given social robots the ability to autonomously engage in open-domain conversations. However, they are still missing a fundamental social skill: making use of the multiple modalities that carry social interactions. While previous work has focused on task-oriented interactions that require referencing the environment or specific phenomena in social interactions such as dialogue breakdowns, we outline the overall needs of a multimodal system for social conversations with robots. We then argue that vision-language models are able to process this wide range of visual information in a sufficiently general manner for autonomous social robots. We describe how to adapt them to this setting, which technical challenges remain, and briefly discuss evaluation practices.
>
---
#### [new 003] Foundation Model-Driven Grasping of Unknown Objects via Center of Gravity Estimation
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决质量分布不均的未知物体抓取不稳定问题。作者利用扩散模型构建数据集并提出基于基础模型的视觉框架，实现对物体质心的精准定位与抓取，提升了抓取成功率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.19242v1](http://arxiv.org/pdf/2507.19242v1)**

> **作者:** Kang Xiangli; Yage He; Xianwu Gong; Zehan Liu; Yuru Bai
>
> **摘要:** This study presents a grasping method for objects with uneven mass distribution by leveraging diffusion models to localize the center of gravity (CoG) on unknown objects. In robotic grasping, CoG deviation often leads to postural instability, where existing keypoint-based or affordance-driven methods exhibit limitations. We constructed a dataset of 790 images featuring unevenly distributed objects with keypoint annotations for CoG localization. A vision-driven framework based on foundation models was developed to achieve CoG-aware grasping. Experimental evaluations across real-world scenarios demonstrate that our method achieves a 49\% higher success rate compared to conventional keypoint-based approaches and an 11\% improvement over state-of-the-art affordance-driven methods. The system exhibits strong generalization with a 76\% CoG localization accuracy on unseen objects, providing a novel solution for precise and stable grasping tasks.
>
---
#### [new 004] Frequency Response Data-Driven Disturbance Observer Design for Flexible Joint Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决柔性关节机器人因关节弹性和系统参数变化导致的控制鲁棒性不足问题。论文提出了一种基于频率响应函数的优化方法，用于设计扰动观测器，以提升控制带宽、抑制振动，并通过实验验证了其有效性与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.18979v1](http://arxiv.org/pdf/2507.18979v1)**

> **作者:** Deokjin Lee; Junho Song; Alireza Karimi; Sehoon Oh
>
> **摘要:** Motion control of flexible joint robots (FJR) is challenged by inherent flexibility and configuration-dependent variations in system dynamics. While disturbance observers (DOB) can enhance system robustness, their performance is often limited by the elasticity of the joints and the variations in system parameters, which leads to a conservative design of the DOB. This paper presents a novel frequency response function (FRF)-based optimization method aimed at improving DOB performance, even in the presence of flexibility and system variability. The proposed method maximizes control bandwidth and effectively suppresses vibrations, thus enhancing overall system performance. Closed-loop stability is rigorously proven using the Nyquist stability criterion. Experimental validation on a FJR demonstrates that the proposed approach significantly improves robustness and motion performance, even under conditions of joint flexibility and system variation.
>
---
#### [new 005] GEAR: Gaze-Enabled Human-Robot Collaborative Assembly
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，旨在解决复杂装配任务中机器人辅助效率与交互问题。作者设计了GEAR系统，通过注视交互提升机器人响应能力，并与触屏交互对比。实验表明GEAR降低了操作负担，提升复杂任务表现与用户体验。**

- **链接: [http://arxiv.org/pdf/2507.18947v1](http://arxiv.org/pdf/2507.18947v1)**

> **作者:** Asad Ali Shahid; Angelo Moroncelli; Drazen Brscic; Takayuki Kanda; Loris Roveda
>
> **备注:** Accepted for publication at 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Recent progress in robot autonomy and safety has significantly improved human-robot interactions, enabling robots to work alongside humans on various tasks. However, complex assembly tasks still present significant challenges due to inherent task variability and the need for precise operations. This work explores deploying robots in an assistive role for such tasks, where the robot assists by fetching parts while the skilled worker provides high-level guidance and performs the assembly. We introduce GEAR, a gaze-enabled system designed to enhance human-robot collaboration by allowing robots to respond to the user's gaze. We evaluate GEAR against a touch-based interface where users interact with the robot through a touchscreen. The experimental study involved 30 participants working on two distinct assembly scenarios of varying complexity. Results demonstrated that GEAR enabled participants to accomplish the assembly with reduced physical demand and effort compared to the touchscreen interface, especially for complex tasks, maintaining great performance, and receiving objects effectively. Participants also reported enhanced user experience while performing assembly tasks. Project page: sites.google.com/view/gear-hri
>
---
#### [new 006] Perpetua: Multi-Hypothesis Persistence Modeling for Semi-Static Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于环境建模任务，旨在解决动态环境中半静态特征的状态预测问题。作者提出了Perpetua方法，通过贝叶斯框架结合“持续”和“出现”滤波器链，实现对环境特征多假设建模，能适应时间变化并预测未来状态，提升了准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.18808v1](http://arxiv.org/pdf/2507.18808v1)**

> **作者:** Miguel Saavedra-Ruiz; Samer B. Nashed; Charlie Gauthier; Liam Paull
>
> **备注:** Accepted to the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025) Code available at https://github.com/montrealrobotics/perpetua-code. Webpage and additional videos at https://montrealrobotics.ca/perpetua/
>
> **摘要:** Many robotic systems require extended deployments in complex, dynamic environments. In such deployments, parts of the environment may change between subsequent robot observations. Most robotic mapping or environment modeling algorithms are incapable of representing dynamic features in a way that enables predicting their future state. Instead, they opt to filter certain state observations, either by removing them or some form of weighted averaging. This paper introduces Perpetua, a method for modeling the dynamics of semi-static features. Perpetua is able to: incorporate prior knowledge about the dynamics of the feature if it exists, track multiple hypotheses, and adapt over time to enable predicting of future feature states. Specifically, we chain together mixtures of "persistence" and "emergence" filters to model the probability that features will disappear or reappear in a formal Bayesian framework. The approach is an efficient, scalable, general, and robust method for estimating the states of features in an environment, both in the present as well as at arbitrary future times. Through experiments on simulated and real-world data, we find that Perpetua yields better accuracy than similar approaches while also being online adaptable and robust to missing observations.
>
---
#### [new 007] Monocular Vision-Based Swarm Robot Localization Using Equilateral Triangular Formations
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人定位任务，旨在解决无外部定位支持的开放环境中，低成本单目视觉传感器的群机器人精确定位问题。提出了基于等边三角形结构的定位方法，通过机器人间一维侧向距离信息，实现二维位置估计。实验表明，其定位误差随时间增长明显小于传统航位推算系统。**

- **链接: [http://arxiv.org/pdf/2507.19100v1](http://arxiv.org/pdf/2507.19100v1)**

> **作者:** Taewon Kang; Ji-Wook Kwon; Il Bae; Jin Hyo Kim
>
> **摘要:** Localization of mobile robots is crucial for deploying robots in real-world applications such as search and rescue missions. This work aims to develop an accurate localization system applicable to swarm robots equipped only with low-cost monocular vision sensors and visual markers. The system is designed to operate in fully open spaces, without landmarks or support from positioning infrastructures. To achieve this, we propose a localization method based on equilateral triangular formations. By leveraging the geometric properties of equilateral triangles, the accurate two-dimensional position of each participating robot is estimated using one-dimensional lateral distance information between robots, which can be reliably and accurately obtained with a low-cost monocular vision sensor. Experimental and simulation results demonstrate that, as travel time increases, the positioning error of the proposed method becomes significantly smaller than that of a conventional dead-reckoning system, another low-cost localization approach applicable to open environments.
>
---
#### [new 008] Bot Appétit! Exploring how Robot Morphology Shapes Perceived Affordances via a Mise en Place Scenario in a VR Kitchen
- **分类: cs.RO**

- **简介: 论文研究机器人形态如何影响人类在虚拟现实厨房协作烹饪场景中的任务分配与摆放决策，属于人机交互任务。通过VR实验与多模态数据分析，提出关于人类偏好生物形态机器人、对其感知能力与协作策略的假设，为后续验证提供基础。**

- **链接: [http://arxiv.org/pdf/2507.19082v1](http://arxiv.org/pdf/2507.19082v1)**

> **作者:** Rachel Ringe; Leandra Thiele; Mihai Pomarlan; Nima Zargham; Robin Nolte; Lars Hurrelbrink; Rainer Malaka
>
> **备注:** Copyright 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** This study explores which factors of the visual design of a robot may influence how humans would place it in a collaborative cooking scenario and how these features may influence task delegation. Human participants were placed in a Virtual Reality (VR) environment and asked to set up a kitchen for cooking alongside a robot companion while considering the robot's morphology. We collected multimodal data for the arrangements created by the participants, transcripts of their think-aloud as they were performing the task, and transcripts of their answers to structured post-task questionnaires. Based on analyzing this data, we formulate several hypotheses: humans prefer to collaborate with biomorphic robots; human beliefs about the sensory capabilities of robots are less influenced by the morphology of the robot than beliefs about action capabilities; and humans will implement fewer avoidance strategies when sharing space with gracile robots. We intend to verify these hypotheses in follow-up studies.
>
---
#### [new 009] SmartPNT-MSF: A Multi-Sensor Fusion Dataset for Positioning and Navigation Research
- **分类: cs.RO**

- **简介: 该论文属于多传感器融合定位与导航任务，旨在解决现有数据集传感器种类少、环境覆盖不足的问题。作者构建了包含GNSS、IMU、相机和LiDAR的高质量公开数据集SmartPNT-MSF，涵盖多种真实场景，并提供标准化采集与处理流程及SLAM算法验证结果。**

- **链接: [http://arxiv.org/pdf/2507.19079v1](http://arxiv.org/pdf/2507.19079v1)**

> **作者:** Feng Zhu; Zihang Zhang; Kangcheng Teng; Abduhelil Yakup; Xiaohong Zhang
>
> **摘要:** High-precision navigation and positioning systems are critical for applications in autonomous vehicles and mobile mapping, where robust and continuous localization is essential. To test and enhance the performance of algorithms, some research institutions and companies have successively constructed and publicly released datasets. However, existing datasets still suffer from limitations in sensor diversity and environmental coverage. To address these shortcomings and advance development in related fields, the SmartPNT Multisource Integrated Navigation, Positioning, and Attitude Dataset has been developed. This dataset integrates data from multiple sensors, including Global Navigation Satellite Systems (GNSS), Inertial Measurement Units (IMU), optical cameras, and LiDAR, to provide a rich and versatile resource for research in multi-sensor fusion and high-precision navigation. The dataset construction process is thoroughly documented, encompassing sensor configurations, coordinate system definitions, and calibration procedures for both cameras and LiDAR. A standardized framework for data collection and processing ensures consistency and scalability, enabling large-scale analysis. Validation using state-of-the-art Simultaneous Localization and Mapping (SLAM) algorithms, such as VINS-Mono and LIO-SAM, demonstrates the dataset's applicability for advanced navigation research. Covering a wide range of real-world scenarios, including urban areas, campuses, tunnels, and suburban environments, the dataset offers a valuable tool for advancing navigation technologies and addressing challenges in complex environments. By providing a publicly accessible, high-quality dataset, this work aims to bridge gaps in sensor diversity, data accessibility, and environmental representation, fostering further innovation in the field.
>
---
#### [new 010] ReCoDe: Reinforcement Learning-based Dynamic Constraint Design for Multi-Agent Coordination
- **分类: cs.RO; cs.AI; cs.LG; cs.MA; I.2.9**

- **简介: 该论文属于多智能体协调任务，旨在解决手工设计约束在复杂场景中效果不佳的问题。作者提出ReCoDe框架，结合基于优化的控制器与多智能体强化学习，通过动态学习额外约束提升协调能力。实验表明其在导航任务中优于传统方法，并能高效利用已有控制器。**

- **链接: [http://arxiv.org/pdf/2507.19151v1](http://arxiv.org/pdf/2507.19151v1)**

> **作者:** Michael Amir; Guang Yang; Zhan Gao; Keisuke Okumura; Heedo Woo; Amanda Prorok
>
> **摘要:** Constraint-based optimization is a cornerstone of robotics, enabling the design of controllers that reliably encode task and safety requirements such as collision avoidance or formation adherence. However, handcrafted constraints can fail in multi-agent settings that demand complex coordination. We introduce ReCoDe--Reinforcement-based Constraint Design--a decentralized, hybrid framework that merges the reliability of optimization-based controllers with the adaptability of multi-agent reinforcement learning. Rather than discarding expert controllers, ReCoDe improves them by learning additional, dynamic constraints that capture subtler behaviors, for example, by constraining agent movements to prevent congestion in cluttered scenarios. Through local communication, agents collectively constrain their allowed actions to coordinate more effectively under changing conditions. In this work, we focus on applications of ReCoDe to multi-agent navigation tasks requiring intricate, context-based movements and consensus, where we show that it outperforms purely handcrafted controllers, other hybrid approaches, and standard MARL baselines. We give empirical (real robot) and theoretical evidence that retaining a user-defined controller, even when it is imperfect, is more efficient than learning from scratch, especially because ReCoDe can dynamically change the degree to which it relies on this controller.
>
---
#### [new 011] Equivariant Volumetric Grasping
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人抓取任务，旨在提升抓取模型的采样效率与性能。论文提出了一种具有旋转等变性的体素抓取模型，采用三平面特征表示和可变形可导向卷积，设计了等变的抓取规划方法，减少了计算和内存成本，并在模拟和真实场景中验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.18847v1](http://arxiv.org/pdf/2507.18847v1)**

> **作者:** Pinhao Song; Yutong Hu; Pengteng Li; Renaud Detry
>
> **备注:** 19 pages
>
> **摘要:** We propose a new volumetric grasp model that is equivariant to rotations around the vertical axis, leading to a significant improvement in sample efficiency. Our model employs a tri-plane volumetric feature representation -- i.e., the projection of 3D features onto three canonical planes. We introduce a novel tri-plane feature design in which features on the horizontal plane are equivariant to 90{\deg} rotations, while the sum of features from the other two planes remains invariant to the same transformations. This design is enabled by a new deformable steerable convolution, which combines the adaptability of deformable convolutions with the rotational equivariance of steerable ones. This allows the receptive field to adapt to local object geometry while preserving equivariance properties. We further develop equivariant adaptations of two state-of-the-art volumetric grasp planners, GIGA and IGD. Specifically, we derive a new equivariant formulation of IGD's deformable attention mechanism and propose an equivariant generative model of grasp orientations based on flow matching. We provide a detailed analytical justification of the proposed equivariance properties and validate our approach through extensive simulated and real-world experiments. Our results demonstrate that the proposed projection-based design significantly reduces both computational and memory costs. Moreover, the equivariant grasp models built on top of our tri-plane features consistently outperform their non-equivariant counterparts, achieving higher performance with only a modest computational overhead. Video and code can be viewed in: https://mousecpn.github.io/evg-page/
>
---
#### [new 012] Probabilistic Collision Risk Estimation through Gauss-Legendre Cubature and Non-Homogeneous Poisson Processes
- **分类: cs.RO**

- **简介: 论文属于自主驾驶运动规划任务，旨在解决高速超车场景中碰撞风险估计不准确的问题。作者提出了GLR算法，结合高斯-勒让德积分与非齐次泊松过程，精确评估车辆几何与轨迹不确定性下的碰撞风险，实验证明其在误差与速度上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.18819v1](http://arxiv.org/pdf/2507.18819v1)**

> **作者:** Trent Weiss; Madhur Behl
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Overtaking in high-speed autonomous racing demands precise, real-time estimation of collision risk; particularly in wheel-to-wheel scenarios where safety margins are minimal. Existing methods for collision risk estimation either rely on simplified geometric approximations, like bounding circles, or perform Monte Carlo sampling which leads to overly conservative motion planning behavior at racing speeds. We introduce the Gauss-Legendre Rectangle (GLR) algorithm, a principled two-stage integration method that estimates collision risk by combining Gauss-Legendre with a non-homogeneous Poisson process over time. GLR produces accurate risk estimates that account for vehicle geometry and trajectory uncertainty. In experiments across 446 overtaking scenarios in a high-fidelity Formula One racing simulation, GLR outperforms five state-of-the-art baselines achieving an average error reduction of 77% and surpassing the next-best method by 52%, all while running at 1000 Hz. The framework is general and applicable to broader motion planning contexts beyond autonomous racing.
>
---
#### [new 013] Diverse and Adaptive Behavior Curriculum for Autonomous Driving: A Student-Teacher Framework with Multi-Agent RL
- **分类: cs.RO**

- **简介: 论文提出一种基于学生-教师框架的多智能体强化学习方法，用于自动驾驶的课程学习。任务是提升自动驾驶在复杂交通中的决策能力。解决现有方法依赖手工设计场景、缺乏行为多样性问题，实现了自动生成多样化交通行为，并根据学生表现动态调整训练难度，最终提升了驾驶策略的鲁棒性和适应性。**

- **链接: [http://arxiv.org/pdf/2507.19146v1](http://arxiv.org/pdf/2507.19146v1)**

> **作者:** Ahmed Abouelazm; Johannes Ratz; Philip Schörner; J. Marius Zöllner
>
> **备注:** Paper accepted in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Autonomous driving faces challenges in navigating complex real-world traffic, requiring safe handling of both common and critical scenarios. Reinforcement learning (RL), a prominent method in end-to-end driving, enables agents to learn through trial and error in simulation. However, RL training often relies on rule-based traffic scenarios, limiting generalization. Additionally, current scenario generation methods focus heavily on critical scenarios, neglecting a balance with routine driving behaviors. Curriculum learning, which progressively trains agents on increasingly complex tasks, is a promising approach to improving the robustness and coverage of RL driving policies. However, existing research mainly emphasizes manually designed curricula, focusing on scenery and actor placement rather than traffic behavior dynamics. This work introduces a novel student-teacher framework for automatic curriculum learning. The teacher, a graph-based multi-agent RL component, adaptively generates traffic behaviors across diverse difficulty levels. An adaptive mechanism adjusts task difficulty based on student performance, ensuring exposure to behaviors ranging from common to critical. The student, though exchangeable, is realized as a deep RL agent with partial observability, reflecting real-world perception constraints. Results demonstrate the teacher's ability to generate diverse traffic behaviors. The student, trained with automatic curricula, outperformed agents trained on rule-based traffic, achieving higher rewards and exhibiting balanced, assertive driving.
>
---
#### [new 014] MetaMorph -- A Metamodelling Approach For Robot Morphology
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于机器人形态分类任务，旨在解决现有分类方法局限于拟人特征、难以涵盖多类型机器人的问题。作者提出MetaMorph框架，基于222个机器人数据，用元建模方法实现对机器人外观特征的结构化比较与设计优化。**

- **链接: [http://arxiv.org/pdf/2507.18820v1](http://arxiv.org/pdf/2507.18820v1)**

> **作者:** Rachel Ringe; Robin Nolte; Nima Zargham; Robert Porzel; Rainer Malaka
>
> **备注:** Copyright 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Robot appearance crucially shapes Human-Robot Interaction (HRI) but is typically described via broad categories like anthropomorphic, zoomorphic, or technical. More precise approaches focus almost exclusively on anthropomorphic features, which fail to classify robots across all types, limiting the ability to draw meaningful connections between robot design and its effect on interaction. In response, we present MetaMorph, a comprehensive framework for classifying robot morphology. Using a metamodeling approach, MetaMorph was synthesized from 222 robots in the IEEE Robots Guide, offering a structured method for comparing visual features. This model allows researchers to assess the visual distances between robot models and explore optimal design traits tailored to different tasks and contexts.
>
---
#### [new 015] How Age Influences the Interpretation of Emotional Body Language in Humanoid Robots -- long paper version
- **分类: cs.RO**

- **简介: 该论文研究不同年龄段人群对人形机器人情感体态语的解读差异，属于人机交互任务。旨在解决机器人情感表达有效性问题，通过分析儿童、年轻人和老年人的数据，发现年龄影响情感解读方式。**

- **链接: [http://arxiv.org/pdf/2507.19335v1](http://arxiv.org/pdf/2507.19335v1)**

> **作者:** Ilaria Consoli; Claudio Mattutino; Cristina Gena; Berardina de Carolis; Giuseppe Palestra
>
> **摘要:** This paper presents an empirical study investigating how individuals across different age groups, children, young and older adults, interpret emotional body language expressed by the humanoid robot NAO. The aim is to offer insights into how users perceive and respond to emotional cues from robotic agents, through an empirical evaluation of the robot's effectiveness in conveying emotions to different groups of users. By analyzing data collected from elderly participants and comparing these findings with previously gathered data from young adults and children, the study highlights similarities and differences between the groups, with younger and older users more similar but different from young adults.
>
---
#### [new 016] Success in Humanoid Reinforcement Learning under Partial Observation
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决在部分可观测环境下人形机器人行走控制的问题。通过设计一种新型历史编码器，处理过往观测信息，成功实现稳定训练与高性能行走，适应身体参数变化，为部分可观测环境下的人形机器人强化学习提供了新方法。**

- **链接: [http://arxiv.org/pdf/2507.18883v1](http://arxiv.org/pdf/2507.18883v1)**

> **作者:** Wuhao Wang; Zhiyong Chen
>
> **备注:** 11 pages, 3 figures, and 4 tables. Not published anywhere else
>
> **摘要:** Reinforcement learning has been widely applied to robotic control, but effective policy learning under partial observability remains a major challenge, especially in high-dimensional tasks like humanoid locomotion. To date, no prior work has demonstrated stable training of humanoid policies with incomplete state information in the benchmark Gymnasium Humanoid-v4 environment. The objective in this environment is to walk forward as fast as possible without falling, with rewards provided for staying upright and moving forward, and penalties incurred for excessive actions and external contact forces. This research presents the first successful instance of learning under partial observability in this environment. The learned policy achieves performance comparable to state-of-the-art results with full state access, despite using only one-third to two-thirds of the original states. Moreover, the policy exhibits adaptability to robot properties, such as variations in body part masses. The key to this success is a novel history encoder that processes a fixed-length sequence of past observations in parallel. Integrated into a standard model-free algorithm, the encoder enables performance on par with fully observed baselines. We hypothesize that it reconstructs essential contextual information from recent observations, thereby enabling robust decision-making.
>
---
#### [new 017] SaLF: Sparse Local Fields for Multi-Sensor Rendering in Real-Time
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶传感器模拟任务，旨在解决现有方法（如NeRF和3DGS）在训练渲染速度、多传感器支持及表示与渲染耦合方面的问题。论文提出SaLF，一种支持光栅化与光线追踪的稀疏局部场表示方法，实现高效、高质量的多传感器实时渲染。**

- **链接: [http://arxiv.org/pdf/2507.18713v1](http://arxiv.org/pdf/2507.18713v1)**

> **作者:** Yun Chen; Matthew Haines; Jingkang Wang; Krzysztof Baron-Lis; Sivabalan Manivasagam; Ze Yang; Raquel Urtasun
>
> **摘要:** High-fidelity sensor simulation of light-based sensors such as cameras and LiDARs is critical for safe and accurate autonomy testing. Neural radiance field (NeRF)-based methods that reconstruct sensor observations via ray-casting of implicit representations have demonstrated accurate simulation of driving scenes, but are slow to train and render, hampering scale. 3D Gaussian Splatting (3DGS) has demonstrated faster training and rendering times through rasterization, but is primarily restricted to pinhole camera sensors, preventing usage for realistic multi-sensor autonomy evaluation. Moreover, both NeRF and 3DGS couple the representation with the rendering procedure (implicit networks for ray-based evaluation, particles for rasterization), preventing interoperability, which is key for general usage. In this work, we present Sparse Local Fields (SaLF), a novel volumetric representation that supports rasterization and raytracing. SaLF represents volumes as a sparse set of 3D voxel primitives, where each voxel is a local implicit field. SaLF has fast training (<30 min) and rendering capabilities (50+ FPS for camera and 600+ FPS LiDAR), has adaptive pruning and densification to easily handle large scenes, and can support non-pinhole cameras and spinning LiDARs. We demonstrate that SaLF has similar realism as existing self-driving sensor simulation methods while improving efficiency and enhancing capabilities, enabling more scalable simulation. https://waabi.ai/salf/
>
---
#### [new 018] EffiComm: Bandwidth Efficient Multi Agent Communication
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶中车际通信任务，旨在解决传输原始感知数据导致的通信负载过高问题。论文提出EffiComm框架，通过选择性传输和自适应网格减少策略，降低传输数据量，同时保持高精度3D目标检测，实现了高效可扩展的通信。**

- **链接: [http://arxiv.org/pdf/2507.19354v1](http://arxiv.org/pdf/2507.19354v1)**

> **作者:** Melih Yazgan; Allen Xavier Arasan; J. Marius Zöllner
>
> **备注:** Accepted for publication at ITSC 2025
>
> **摘要:** Collaborative perception allows connected vehicles to exchange sensor information and overcome each vehicle's blind spots. Yet transmitting raw point clouds or full feature maps overwhelms Vehicle-to-Vehicle (V2V) communications, causing latency and scalability problems. We introduce EffiComm, an end-to-end framework that transmits less than 40% of the data required by prior art while maintaining state-of-the-art 3D object detection accuracy. EffiComm operates on Bird's-Eye-View (BEV) feature maps from any modality and applies a two-stage reduction pipeline: (1) Selective Transmission (ST) prunes low-utility regions with a confidence mask; (2) Adaptive Grid Reduction (AGR) uses a Graph Neural Network (GNN) to assign vehicle-specific keep ratios according to role and network load. The remaining features are fused with a soft-gated Mixture-of-Experts (MoE) attention layer, offering greater capacity and specialization for effective feature integration. On the OPV2V benchmark, EffiComm reaches 0.84 mAP@0.7 while sending only an average of approximately 1.5 MB per frame, outperforming previous methods on the accuracy-per-bit curve. These results highlight the value of adaptive, learned communication for scalable Vehicle-to-Everything (V2X) perception.
>
---
#### [new 019] Efficient Lines Detection for Robot Soccer
- **分类: cs.CV; cs.RO**

- **简介: 论文属于机器人视觉任务，旨在解决足球场上线条检测问题。为实现机器人自定位，作者改进ELSED算法，加入RGB颜色过渡分类步骤，并采用PSO优化阈值，提升检测效率与准确率，适用于低功耗平台实时应用。**

- **链接: [http://arxiv.org/pdf/2507.19469v1](http://arxiv.org/pdf/2507.19469v1)**

> **作者:** João G. Melo; João P. Mafaldo; Edna Barros
>
> **备注:** 12 pages, 8 figures, RoboCup Symposium 2025
>
> **摘要:** Self-localization is essential in robot soccer, where accurate detection of visual field features, such as lines and boundaries, is critical for reliable pose estimation. This paper presents a lightweight and efficient method for detecting soccer field lines using the ELSED algorithm, extended with a classification step that analyzes RGB color transitions to identify lines belonging to the field. We introduce a pipeline based on Particle Swarm Optimization (PSO) for threshold calibration to optimize detection performance, requiring only a small number of annotated samples. Our approach achieves accuracy comparable to a state-of-the-art deep learning model while offering higher processing speed, making it well-suited for real-time applications on low-power robotic platforms.
>
---
#### [new 020] Eyes Will Shut: A Vision-Based Next GPS Location Prediction Model by Reinforcement Learning from Visual Map Feed Back
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于轨迹预测任务，旨在解决基于地图的下一个位置预测问题。现有方法缺乏人类般的地图推理能力。作者提出VLMLocPredictor，结合视觉语言模型与强化学习，通过视觉地图反馈进行自我改进，实现更准确的预测。**

- **链接: [http://arxiv.org/pdf/2507.18661v1](http://arxiv.org/pdf/2507.18661v1)**

> **作者:** Ruixing Zhang; Yang Zhang; Tongyu Zhu; Leilei Sun; Weifeng Lv
>
> **摘要:** Next Location Prediction is a fundamental task in the study of human mobility, with wide-ranging applications in transportation planning, urban governance, and epidemic forecasting. In practice, when humans attempt to predict the next location in a trajectory, they often visualize the trajectory on a map and reason based on road connectivity and movement trends. However, the vast majority of existing next-location prediction models do not reason over maps \textbf{in the way that humans do}. Fortunately, the recent development of Vision-Language Models (VLMs) has demonstrated strong capabilities in visual perception and even visual reasoning. This opens up a new possibility: by rendering both the road network and trajectory onto an image and leveraging the reasoning abilities of VLMs, we can enable models to perform trajectory inference in a human-like manner. To explore this idea, we first propose a method called Vision-Guided Location Search (VGLS), which evaluates whether a general-purpose VLM is capable of trajectory-based reasoning without modifying any of its internal parameters. Based on insights from the VGLS results, we further propose our main approach: VLMLocPredictor, which is composed of two stages: In the first stage, we design two Supervised Fine-Tuning (SFT) tasks that help the VLM understand road network and trajectory structures and acquire basic reasoning ability on such visual inputs. In the second stage, we introduce Reinforcement Learning from Visual Map Feedback, enabling the model to self-improve its next-location prediction ability through interaction with the environment. Experiments conducted on datasets from four different cities show that our method achieves state-of-the-art (SOTA) performance and exhibits superior cross-city generalization compared to other LLM-based approaches.
>
---
#### [new 021] Assessing the Reliability and Validity of a Balance Mat for Measuring Postural Stability: A Combined Robot-Human Approach
- **分类: eess.SP; cs.RO**

- **简介: 该论文评估了一种新型便携式平衡垫（BM）测量姿势稳定性的信度和效度，旨在解决传统力板（FP）便携性差、操作复杂的问题。研究通过机器人模拟和人体实验验证BM性能，结果显示其在校准后具有良好的一致性和准确性。**

- **链接: [http://arxiv.org/pdf/2507.18943v1](http://arxiv.org/pdf/2507.18943v1)**

> **作者:** Abishek Shrestha; Damith Herath; Angie Fearon; Maryam Ghahramani
>
> **摘要:** Postural sway assessment is important for detecting balance problems and identifying people at risk of falls. Force plates (FP) are considered the gold standard postural sway assessment method in laboratory conditions, but their lack of portability and requirement of high-level expertise limit their widespread usage. This study evaluates the reliability and validity of a novel Balance Mat (BM) device, a low-cost portable alternative that uses optical fibre technology. The research includes two studies: a robot study and a human study. In the robot study, a UR10 robotic arm was used to obtain controlled sway patterns to assess the reliability and sensitivity of the BM. In the human study, 51 healthy young participants performed balance tasks on the BM in combination with an FP to evaluate the BM's validity. Sway metrics such as sway mean, sway absolute mean, sway root mean square (RMS), sway path, sway range, and sway velocity were calculated from both BM and FP and compared. Reliability was evaluated using the intra-class correlation coefficient (ICC), where values greater than 0.9 were considered excellent and values between 0.75 and 0.9 were considered good. Results from the robot study demonstrated good to excellent ICC values in both single and double-leg stances. The human study showed moderate to strong correlations for sway path and range. Using Bland-Altman plots for agreement analysis revealed proportional bias between the BM and the FP where the BM overestimated sway metrics compared to the FP. Calibration was used to improve the agreement between the devices. The device demonstrated consistent sway measurement across varied stance conditions, establishing both reliability and validity following appropriate calibration.
>
---
#### [new 022] GMM-Based Time-Varying Coverage Control
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于多机器人覆盖控制任务，旨在解决时间变化密度函数下的覆盖控制问题。通过将密度函数建模为随时间变化的高斯混合模型（GMM），并利用其结构设计高效的时变覆盖控制器，实现了对动态环境（如化学羽流）的分布式跟踪。论文通过仿真和实地实验验证了方法的有效性与实用性。**

- **链接: [http://arxiv.org/pdf/2507.18938v1](http://arxiv.org/pdf/2507.18938v1)**

> **作者:** Behzad Zamani; James Kennedy; Airlie Chapman; Peter Dower; Chris Manzie; Simon Crase
>
> **备注:** Submitted to CDC 2025
>
> **摘要:** In coverage control problems that involve time-varying density functions, the coverage control law depends on spatial integrals of the time evolution of the density function. The latter is often neglected, replaced with an upper bound or calculated as a numerical approximation of the spatial integrals involved. In this paper, we consider a special case of time-varying density functions modeled as Gaussian Mixture Models (GMMs) that evolve with time via a set of time-varying sources (with known corresponding velocities). By imposing this structure, we obtain an efficient time-varying coverage controller that fully incorporates the time evolution of the density function. We show that the induced trajectories under our control law minimise the overall coverage cost. We elicit the structure of the proposed controller and compare it with a classical time-varying coverage controller, against which we benchmark the coverage performance in simulation. Furthermore, we highlight that the computationally efficient and distributed nature of the proposed control law makes it ideal for multi-vehicle robotic applications involving time-varying coverage control problems. We employ our method in plume monitoring using a swarm of drones. In an experimental field trial we show that drones guided by the proposed controller are able to track a simulated time-varying chemical plume in a distributed manner.
>
---
#### [new 023] Fast Learning of Non-Cooperative Spacecraft 3D Models through Primitive Initialization
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于3D建模与计算机视觉任务，旨在解决从单目图像快速学习非合作航天器高精度3D模型的问题。论文提出了一种基于CNN的原始初始化器，结合3D高斯泼溅技术，减少了训练所需迭代次数和图像数量，并能在姿态估计不精确的情况下仍保持高性能，提升了空间应用的可行性。**

- **链接: [http://arxiv.org/pdf/2507.19459v1](http://arxiv.org/pdf/2507.19459v1)**

> **作者:** Pol Francesch Huc; Emily Bates; Simone D'Amico
>
> **摘要:** The advent of novel view synthesis techniques such as NeRF and 3D Gaussian Splatting (3DGS) has enabled learning precise 3D models only from posed monocular images. Although these methods are attractive, they hold two major limitations that prevent their use in space applications: they require poses during training, and have high computational cost at training and inference. To address these limitations, this work contributes: (1) a Convolutional Neural Network (CNN) based primitive initializer for 3DGS using monocular images; (2) a pipeline capable of training with noisy or implicit pose estimates; and (3) and analysis of initialization variants that reduce the training cost of precise 3D models. A CNN takes a single image as input and outputs a coarse 3D model represented as an assembly of primitives, along with the target's pose relative to the camera. This assembly of primitives is then used to initialize 3DGS, significantly reducing the number of training iterations and input images needed -- by at least an order of magnitude. For additional flexibility, the CNN component has multiple variants with different pose estimation techniques. This work performs a comparison between these variants, evaluating their effectiveness for downstream 3DGS training under noisy or implicit pose estimates. The results demonstrate that even with imperfect pose supervision, the pipeline is able to learn high-fidelity 3D representations, opening the door for the use of novel view synthesis in space applications.
>
---
#### [new 024] Perspective from a Higher Dimension: Can 3D Geometric Priors Help Visual Floorplan Localization?
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，旨在解决视觉定位与平面图之间的模态与几何差异问题。通过引入3D几何先验，利用多视角约束和场景表面重建，提升定位准确性。方法无需额外标注，采用自监督对比学习，有效桥接模态差异，显著提高定位效果。**

- **链接: [http://arxiv.org/pdf/2507.18881v1](http://arxiv.org/pdf/2507.18881v1)**

> **作者:** Bolei Chen; Jiaxu Kang; Haonan Yang; Ping Zhong; Jianxin Wang
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Since a building's floorplans are easily accessible, consistent over time, and inherently robust to changes in visual appearance, self-localization within the floorplan has attracted researchers' interest. However, since floorplans are minimalist representations of a building's structure, modal and geometric differences between visual perceptions and floorplans pose challenges to this task. While existing methods cleverly utilize 2D geometric features and pose filters to achieve promising performance, they fail to address the localization errors caused by frequent visual changes and view occlusions due to variously shaped 3D objects. To tackle these issues, this paper views the 2D Floorplan Localization (FLoc) problem from a higher dimension by injecting 3D geometric priors into the visual FLoc algorithm. For the 3D geometric prior modeling, we first model geometrically aware view invariance using multi-view constraints, i.e., leveraging imaging geometric principles to provide matching constraints between multiple images that see the same points. Then, we further model the view-scene aligned geometric priors, enhancing the cross-modal geometry-color correspondences by associating the scene's surface reconstruction with the RGB frames of the sequence. Both 3D priors are modeled through self-supervised contrastive learning, thus no additional geometric or semantic annotations are required. These 3D priors summarized in extensive realistic scenes bridge the modal gap while improving localization success without increasing the computational burden on the FLoc algorithm. Sufficient comparative studies demonstrate that our method significantly outperforms state-of-the-art methods and substantially boosts the FLoc accuracy. All data and code will be released after the anonymous review.
>
---
#### [new 025] Diffusion-FS: Multimodal Free-Space Prediction via Diffusion for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的可行驶区域预测任务，旨在解决现有方法难以准确估计可导航通道的问题。论文提出Diffusion-FS，通过扩散模型结合轮廓点去噪（ContourDiff），实现基于单目图像的自由空间走廊预测，提升了预测的准确性与结构化程度。**

- **链接: [http://arxiv.org/pdf/2507.18763v1](http://arxiv.org/pdf/2507.18763v1)**

> **作者:** Keshav Gupta; Tejas S. Stanley; Pranjal Paul; Arun K. Singh; K. Madhava Krishna
>
> **备注:** 8 pages, 7 figures, IROS 2025
>
> **摘要:** Drivable Free-space prediction is a fundamental and crucial problem in autonomous driving. Recent works have addressed the problem by representing the entire non-obstacle road regions as the free-space. In contrast our aim is to estimate the driving corridors that are a navigable subset of the entire road region. Unfortunately, existing corridor estimation methods directly assume a BEV-centric representation, which is hard to obtain. In contrast, we frame drivable free-space corridor prediction as a pure image perception task, using only monocular camera input. However such a formulation poses several challenges as one doesn't have the corresponding data for such free-space corridor segments in the image. Consequently, we develop a novel self-supervised approach for free-space sample generation by leveraging future ego trajectories and front-view camera images, making the process of visual corridor estimation dependent on the ego trajectory. We then employ a diffusion process to model the distribution of such segments in the image. However, the existing binary mask-based representation for a segment poses many limitations. Therefore, we introduce ContourDiff, a specialized diffusion-based architecture that denoises over contour points rather than relying on binary mask representations, enabling structured and interpretable free-space predictions. We evaluate our approach qualitatively and quantitatively on both nuScenes and CARLA, demonstrating its effectiveness in accurately predicting safe multimodal navigable corridors in the image.
>
---
## 更新

#### [replaced 001] Motion Synthesis with Sparse and Flexible Keyjoint Control
- **分类: cs.GR; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.15557v2](http://arxiv.org/pdf/2503.15557v2)**

> **作者:** Inwoo Hwang; Jinseok Bae; Donggeun Lim; Young Min Kim
>
> **备注:** Accepted to ICCV 2025. Project Page: http://inwoohwang.me/SFControl
>
> **摘要:** Creating expressive character animations is labor-intensive, requiring intricate manual adjustment of animators across space and time. Previous works on controllable motion generation often rely on a predefined set of dense spatio-temporal specifications (e.g., dense pelvis trajectories with exact per-frame timing), limiting practicality for animators. To process high-level intent and intuitive control in diverse scenarios, we propose a practical controllable motions synthesis framework that respects sparse and flexible keyjoint signals. Our approach employs a decomposed diffusion-based motion synthesis framework that first synthesizes keyjoint movements from sparse input control signals and then synthesizes full-body motion based on the completed keyjoint trajectories. The low-dimensional keyjoint movements can easily adapt to various control signal types, such as end-effector position for diverse goal-driven motion synthesis, or incorporate functional constraints on a subset of keyjoints. Additionally, we introduce a time-agnostic control formulation, eliminating the need for frame-specific timing annotations and enhancing control flexibility. Then, the shared second stage can synthesize a natural whole-body motion that precisely satisfies the task requirement from dense keyjoint movements. We demonstrate the effectiveness of sparse and flexible keyjoint control through comprehensive experiments on diverse datasets and scenarios.
>
---
#### [replaced 002] RoboCar: A Rapidly Deployable Open-Source Platform for Autonomous Driving Research
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2405.03572v3](http://arxiv.org/pdf/2405.03572v3)**

> **作者:** Mehdi Testouri; Gamal Elghazaly; Raphael Frank
>
> **摘要:** This paper introduces RoboCar, an open-source research platform for autonomous driving developed at the University of Luxembourg. RoboCar provides a modular, cost-effective framework for the development of experimental Autonomous Driving Systems (ADS), utilizing the 2018 KIA Soul EV. The platform integrates a robust hardware and software architecture that aligns with the vehicle's existing systems, minimizing the need for extensive modifications. It supports various autonomous driving functions and has undergone real-world testing on public roads in Luxembourg City. This paper outlines the platform's architecture, integration challenges, and initial test results, offering insights into its application in advancing autonomous driving research. RoboCar is available to anyone at https://github.com/sntubix/robocar and is released under an open-source MIT license.
>
---
#### [replaced 003] DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.16806v2](http://arxiv.org/pdf/2503.16806v2)**

> **作者:** Jiangran Lyu; Ziming Li; Xuesong Shi; Chaoyi Xu; Yizhou Wang; He Wang
>
> **备注:** Project Page:https://pku-epic.github.io/DyWA/
>
> **摘要:** Nonprehensile manipulation is crucial for handling objects that are too thin, large, or otherwise ungraspable in unstructured environments. While conventional planning-based approaches struggle with complex contact modeling, learning-based methods have recently emerged as a promising alternative. However, existing learning-based approaches face two major limitations: they heavily rely on multi-view cameras and precise pose tracking, and they fail to generalize across varying physical conditions, such as changes in object mass and table friction. To address these challenges, we propose the Dynamics-Adaptive World Action Model (DyWA), a novel framework that enhances action learning by jointly predicting future states while adapting to dynamics variations based on historical trajectories. By unifying the modeling of geometry, state, physics, and robot actions, DyWA enables more robust policy learning under partial observability. Compared to baselines, our method improves the success rate by 31.5% using only single-view point cloud observations in the simulation. Furthermore, DyWA achieves an average success rate of 68% in real-world experiments, demonstrating its ability to generalize across diverse object geometries, adapt to varying table friction, and robustness in challenging scenarios such as half-filled water bottles and slippery surfaces.
>
---
#### [replaced 004] Exploring 6G Potential for Industrial Digital Twinning and Swarm Intelligence in Obstacle-Rich Environments
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2406.19930v3](http://arxiv.org/pdf/2406.19930v3)**

> **作者:** Siyu Yuan; Khurshid Alam; Bin Han; Dennis Krummacker; Hans D. Schotten
>
> **备注:** Submitted to IEEE VTM
>
> **摘要:** With the advent of Sixth Generation (6G) technology, the demand for efficient and intelligent systems in industrial applications has surged, driving the need for advanced solutions in target localization. Utilizing swarm robots to locate unknown targets involves navigating increasingly complex environments. digital twin (DT) offers a robust solution by creating a virtual replica of the physical world, which enhances the swarm's navigation capabilities. Our framework leverages DT and integrates swarm intelligence (SI) to store physical map information in the cloud, enabling robots to efficiently locate unknown targets. The simulation results demonstrate that the DT framework, augmented by SI, significantly improves target location efficiency in obstacle-rich environments compared to traditional methods. This research underscores the potential of combining DT and swarm intelligence to advance the field of robotic navigation and target localization in complex industrial settings.
>
---
#### [replaced 005] TrafficMCTS: A Closed-Loop Traffic Flow Generation Framework with Group-Based Monte Carlo Tree Search
- **分类: cs.RO; cs.MA; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2308.12797v3](http://arxiv.org/pdf/2308.12797v3)**

> **作者:** Ze Fu; Licheng Wen; Pinlong Cai; Daocheng Fu; Song Mao; Botian Shi
>
> **备注:** Published in IEEE Transactions on Intelligent Transportation Systems
>
> **摘要:** Traffic flow simulation within the domain of intelligent transportation systems is garnering significant attention, and generating realistic, diverse, and human-like traffic patterns presents critical challenges that must be addressed. Current approaches often hinge on predefined driver models, objective optimization, or reliance on pre-recorded driving datasets, imposing limitations on their scalability, versatility, and adaptability. In this paper, we introduce TrafficMCTS, an innovative framework that harnesses the synergy of group-based Monte Carlo tree search (MCTS) and Social Value Orientation (SVO) to engender a multifaceted traffic flow with varying driving styles and cooperative tendencies. Anchored by a closed-loop architecture, our framework enables vehicles to dynamically adapt to their environment in real time, and ensure feasible collision-free trajectories. Through comprehensive comparisons with state-of-the-art methods, we illuminate the advantages of our approach in terms of computational efficiency, planning success rate, intention completion time, and diversity metrics. Besides, we simulate multiple scenarios to illustrate the effectiveness of the proposed framework and highlight its ability to induce diverse social behaviors within the traffic flow. Finally, we validate the scalability of TrafficMCTS by demonstrating its capability to efficiently simulate diverse traffic scenarios involving numerous interacting vehicles within a complex road network, capturing the intricate dynamics of human-like driving behaviors.
>
---
#### [replaced 006] Fast-Revisit Coverage Path Planning for Autonomous Mobile Patrol Robots Using Long-Range Sensor Information
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2501.07343v2](http://arxiv.org/pdf/2501.07343v2)**

> **作者:** Srinivas Kachavarapu; Tobias Doernbach; Reinhard Gerndt
>
> **备注:** accepted for presentation at the International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** The utilization of Unmanned Ground Vehicles (UGVs) for patrolling industrial sites has expanded significantly. These UGVs typically are equipped with perception systems, e.g., computer vision, with limited range due to sensor limitations or site topology. High-level control of the UGVs requires Coverage Path Planning (CPP) algorithms that navigate all relevant waypoints and promptly start the next cycle. In this paper, we propose the novel Fast-Revisit Coverage Path Planning (FaRe-CPP) algorithm using a greedy heuristic approach to propose waypoints for maximum coverage area and a random search-based path optimization technique to obtain a path along the proposed waypoints with minimum revisit time. We evaluated the algorithm in a simulated environment using Gazebo and a camera-equipped TurtleBot3 against a number of existing algorithms. Compared to their average path lengths and revisit times, our FaRe-CPP algorithm showed a reduction of at least 21% and 33%, respectively, in these highly relevant performance indicators.
>
---
#### [replaced 007] A Systematic Digital Engineering Approach to Verification & Validation of Autonomous Ground Vehicles in Off-Road Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.13787v2](http://arxiv.org/pdf/2503.13787v2)**

> **作者:** Tanmay Vilas Samak; Chinmay Vilas Samak; Julia Brault; Cori Harber; Kirsten McCane; Jonathon Smereka; Mark Brudnak; David Gorsich; Venkat Krovi
>
> **备注:** Accepted at Modeling, Estimation and Control Conference (MECC) 2025. DISTRIBUTION STATEMENT A. Approved for public release; distribution is unlimited. OPSEC9523
>
> **摘要:** The engineering community currently encounters significant challenges in the systematic development and validation of autonomy algorithms for off-road ground vehicles. These challenges are posed by unusually high test parameters and algorithmic variants. In order to address these pain points, this work presents an optimized digital engineering framework that tightly couples digital twin simulations with model-based systems engineering (MBSE) and model-based design (MBD) workflows. The efficacy of the proposed framework is demonstrated through an end-to-end case study of an autonomous light tactical vehicle (LTV) performing visual servoing to drive along a dirt road and reacting to any obstacles or environmental changes. The presented methodology allows for traceable requirements engineering, efficient variant management, granular parameter sweep setup, systematic test-case definition, and automated execution of the simulations. The candidate off-road autonomy algorithm is evaluated for satisfying requirements against a battery of 128 test cases, which is procedurally generated based on the test parameters (times of the day and weather conditions) and algorithmic variants (perception, planning, and control sub-systems). Finally, the test results and key performance indicators are logged, and the test report is generated automatically. This then allows for manual as well as automated data analysis with traceability and tractability across the digital thread.
>
---
#### [replaced 008] Anti-Degeneracy Scheme for Lidar SLAM based on Particle Filter in Geometry Feature-Less Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.11486v2](http://arxiv.org/pdf/2502.11486v2)**

> **作者:** Yanbin Li; Wei Zhang; Zhiguo Zhang; Xiaogang Shi; Ziruo Li; Mingming Zhang; Hongping Xie; Wenzheng Chi
>
> **备注:** 8 pages, 9 figures, IEEE Robotics and Automation Letters
>
> **摘要:** Simultaneous localization and mapping (SLAM) based on particle filtering has been extensively employed in indoor scenarios due to its high efficiency. However, in geometry feature-less scenes, the accuracy is severely reduced due to lack of constraints. In this article, we propose an anti-degeneracy system based on deep learning. Firstly, we design a scale-invariant linear mapping to convert coordinates in continuous space into discrete indexes, in which a data augmentation method based on Gaussian model is proposed to ensure the model performance by effectively mitigating the impact of changes in the number of particles on the feature distribution. Secondly, we develop a degeneracy detection model using residual neural networks (ResNet) and transformer which is able to identify degeneracy by scrutinizing the distribution of the particle population. Thirdly, an adaptive anti-degeneracy strategy is designed, which first performs fusion and perturbation on the resample process to provide rich and accurate initial values for the pose optimization, and use a hierarchical pose optimization combining coarse and fine matching, which is able to adaptively adjust the optimization frequency and the sensor trustworthiness according to the degree of degeneracy, in order to enhance the ability of searching the global optimal pose. Finally, we demonstrate the optimality of the model, as well as the improvement of the image matrix method and GPU on the computation time through ablation experiments, and verify the performance of the anti-degeneracy system in different scenarios through simulation experiments and real experiments. This work has been submitted to IEEE for publication. Copyright may be transferred without notice, after which this version may no longer be available.
>
---
#### [replaced 009] Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.04790v3](http://arxiv.org/pdf/2507.04790v3)**

> **作者:** Giwon Lee; Wooseong Jeong; Daehee Park; Jaewoo Jeong; Kuk-Jin Yoon
>
> **备注:** Accepted at ICCV 2025 (Highlight)
>
> **摘要:** Motion planning is a crucial component of autonomous robot driving. While various trajectory datasets exist, effectively utilizing them for a target domain remains challenging due to differences in agent interactions and environmental characteristics. Conventional approaches, such as domain adaptation or ensemble learning, leverage multiple source datasets but suffer from domain imbalance, catastrophic forgetting, and high computational costs. To address these challenges, we propose Interaction-Merged Motion Planning (IMMP), a novel approach that leverages parameter checkpoints trained on different domains during adaptation to the target domain. IMMP follows a two-step process: pre-merging to capture agent behaviors and interactions, sufficiently extracting diverse information from the source domain, followed by merging to construct an adaptable model that efficiently transfers diverse interactions to the target domain. Our method is evaluated on various planning benchmarks and models, demonstrating superior performance compared to conventional approaches.
>
---
#### [replaced 010] Towards Generalized Range-View LiDAR Segmentation in Adverse Weather
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08979v3](http://arxiv.org/pdf/2506.08979v3)**

> **作者:** Longyu Yang; Lu Zhang; Jun Liu; Yap-Peng Tan; Heng Tao Shen; Xiaofeng Zhu; Ping Hu
>
> **摘要:** LiDAR segmentation has emerged as an important task to enrich scene perception and understanding. Range-view-based methods have gained popularity due to their high computational efficiency and compatibility with real-time deployment. However, their generalized performance under adverse weather conditions remains underexplored, limiting their reliability in real-world environments. In this work, we identify and analyze the unique challenges that affect the generalization of range-view LiDAR segmentation in severe weather. To address these challenges, we propose a modular and lightweight framework that enhances robustness without altering the core architecture of existing models. Our method reformulates the initial stem block of standard range-view networks into two branches to process geometric attributes and reflectance intensity separately. Specifically, a Geometric Abnormality Suppression (GAS) module reduces the influence of weather-induced spatial noise, and a Reflectance Distortion Calibration (RDC) module corrects reflectance distortions through memory-guided adaptive instance normalization. The processed features are then fused and passed to the original segmentation pipeline. Extensive experiments on different benchmarks and baseline models demonstrate that our approach significantly improves generalization to adverse weather with minimal inference overhead, offering a practical and effective solution for real-world LiDAR segmentation.
>
---
#### [replaced 011] ReSem3D: Refinable 3D Spatial Constraints via Fine-Grained Semantic Grounding for Generalizable Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.18262v2](http://arxiv.org/pdf/2507.18262v2)**

> **作者:** Chenyu Su; Weiwei Shang; Chen Qian; Fei Zhang; Shuang Cong
>
> **备注:** 12 pages,9 figures
>
> **摘要:** Semantics-driven 3D spatial constraints align highlevel semantic representations with low-level action spaces, facilitating the unification of task understanding and execution in robotic manipulation. The synergistic reasoning of Multimodal Large Language Models (MLLMs) and Vision Foundation Models (VFMs) enables cross-modal 3D spatial constraint construction. Nevertheless, existing methods have three key limitations: (1) coarse semantic granularity in constraint modeling, (2) lack of real-time closed-loop planning, (3) compromised robustness in semantically diverse environments. To address these challenges, we propose ReSem3D, a unified manipulation framework for semantically diverse environments, leveraging the synergy between VFMs and MLLMs to achieve fine-grained visual grounding and dynamically constructs hierarchical 3D spatial constraints for real-time manipulation. Specifically, the framework is driven by hierarchical recursive reasoning in MLLMs, which interact with VFMs to automatically construct 3D spatial constraints from natural language instructions and RGB-D observations in two stages: part-level extraction and region-level refinement. Subsequently, these constraints are encoded as real-time optimization objectives in joint space, enabling reactive behavior to dynamic disturbances. Extensive simulation and real-world experiments are conducted in semantically rich household and sparse chemical lab environments. The results demonstrate that ReSem3D performs diverse manipulation tasks under zero-shot conditions, exhibiting strong adaptability and generalization. Code and videos are available at https://github.com/scy-v/ReSem3D and https://resem3d.github.io.
>
---
#### [replaced 012] MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.10543v2](http://arxiv.org/pdf/2507.10543v2)**

> **作者:** Juyi Sheng; Ziyi Wang; Peiming Li; Yong Liu; Mengyuan Liu
>
> **摘要:** In robot manipulation, robot learning has become a prevailing approach. However, generative models within this field face a fundamental trade-off between the slow, iterative sampling of diffusion models and the architectural constraints of faster Flow-based methods, which often rely on explicit consistency losses. To address these limitations, we introduce MP1, which pairs 3D point-cloud inputs with the MeanFlow paradigm to generate action trajectories in one network function evaluation (1-NFE). By directly learning the interval-averaged velocity via the "MeanFlow Identity", our policy avoids any additional consistency constraints. This formulation eliminates numerical ODE-solver errors during inference, yielding more precise trajectories. MP1 further incorporates CFG for improved trajectory controllability while retaining 1-NFE inference without reintroducing structural constraints. Because subtle scene-context variations are critical for robot learning, especially in few-shot learning, we introduce a lightweight Dispersive Loss that repels state embeddings during training, boosting generalization without slowing inference. We validate our method on the Adroit and Meta-World benchmarks, as well as in real-world scenarios. Experimental results show MP1 achieves superior average task success rates, outperforming DP3 by 10.2% and FlowPolicy by 7.3%. Its average inference time is only 6.8 ms-19x faster than DP3 and nearly 2x faster than FlowPolicy. Our code is available at https://github.com/LogSSim/MP1.git.
>
---
#### [replaced 013] $S^2M^2$: Scalable Stereo Matching Model for Reliable Depth Estimation
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.13229v2](http://arxiv.org/pdf/2507.13229v2)**

> **作者:** Junhong Min; Youngpil Jeon; Jimin Kim; Minyong Choi
>
> **备注:** 8 pages, 5 figures, ICCV accepted paper
>
> **摘要:** The pursuit of a generalizable stereo matching model, capable of performing across varying resolutions and disparity ranges without dataset-specific fine-tuning, has revealed a fundamental trade-off. Iterative local search methods achieve high scores on constrained benchmarks, but their core mechanism inherently limits the global consistency required for true generalization. On the other hand, global matching architectures, while theoretically more robust, have been historically rendered infeasible by prohibitive computational and memory costs. We resolve this dilemma with $S^2M^2$: a global matching architecture that achieves both state-of-the-art accuracy and high efficiency without relying on cost volume filtering or deep refinement stacks. Our design integrates a multi-resolution transformer for robust long-range correspondence, trained with a novel loss function that concentrates probability on feasible matches. This approach enables a more robust joint estimation of disparity, occlusion, and confidence. $S^2M^2$ establishes a new state of the art on the Middlebury v3 and ETH3D benchmarks, significantly outperforming prior methods across most metrics while reconstructing high-quality details with competitive efficiency.
>
---
#### [replaced 014] Signal Temporal Logic Compliant Co-design of Planning and Control
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.13225v2](http://arxiv.org/pdf/2507.13225v2)**

> **作者:** Manas Sashank Juvvi; Tushar Dilip Kurne; Vaishnavi J; Shishir Kolathaya; Pushpak Jagtap
>
> **摘要:** This work presents a novel co-design strategy that integrates trajectory planning and control to handle STL-based tasks in autonomous robots. The method consists of two phases: $(i)$ learning spatio-temporal motion primitives to encapsulate the inherent robot-specific constraints and $(ii)$ constructing an STL-compliant motion plan from these primitives. Initially, we employ reinforcement learning to construct a library of control policies that perform trajectories described by the motion primitives. Then, we map motion primitives to spatio-temporal characteristics. Subsequently, we present a sampling-based STL-compliant motion planning strategy tailored to meet the STL specification. The proposed model-free approach, which generates feasible STL-compliant motion plans across various environments, is validated on differential-drive and quadruped robots across various STL specifications. Demonstration videos are available at https://tinyurl.com/m6zp7rsm.
>
---
#### [replaced 015] Collision-free Control Barrier Functions for General Ellipsoids via Separating Hyperplane
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2505.20847v2](http://arxiv.org/pdf/2505.20847v2)**

> **作者:** Zeming Wu; Lu Liu
>
> **摘要:** This paper presents a novel collision avoidance method for general ellipsoids based on control barrier functions (CBFs) and separating hyperplanes. First, collision-free conditions for general ellipsoids are analytically derived using the concept of dual cones. These conditions are incorporated into the CBF framework by extending the system dynamics of controlled objects with separating hyperplanes, enabling efficient and reliable collision avoidance. The validity of the proposed collision-free CBFs is rigorously proven, ensuring their effectiveness in enforcing safety constraints. The proposed method requires only single-level optimization, significantly reducing computational time compared to state-of-the-art methods. Numerical simulations and real-world experiments demonstrate the effectiveness and practicality of the proposed algorithm.
>
---
#### [replaced 016] RAMBO: RL-augmented Model-based Whole-body Control for Loco-manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.06662v3](http://arxiv.org/pdf/2504.06662v3)**

> **作者:** Jin Cheng; Dongho Kang; Gabriele Fadini; Guanya Shi; Stelian Coros
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Loco-manipulation, physical interaction of various objects that is concurrently coordinated with locomotion, remains a major challenge for legged robots due to the need for both precise end-effector control and robustness to unmodeled dynamics. While model-based controllers provide precise planning via online optimization, they are limited by model inaccuracies. In contrast, learning-based methods offer robustness, but they struggle with precise modulation of interaction forces. We introduce RAMBO, a hybrid framework that integrates model-based whole-body control within a feedback policy trained with reinforcement learning. The model-based module generates feedforward torques by solving a quadratic program, while the policy provides feedback corrective terms to enhance robustness. We validate our framework on a quadruped robot across a diverse set of real-world loco-manipulation tasks, such as pushing a shopping cart, balancing a plate, and holding soft objects, in both quadrupedal and bipedal walking. Our experiments demonstrate that RAMBO enables precise manipulation capabilities while achieving robust and dynamic locomotion.
>
---
#### [replaced 017] Incremental Learning for Robot Shared Autonomy
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.06315v4](http://arxiv.org/pdf/2410.06315v4)**

> **作者:** Yiran Tao; Guixiu Qiao; Dan Ding; Zackory Erickson
>
> **摘要:** Shared autonomy holds promise for improving the usability and accessibility of assistive robotic arms, but current methods often rely on costly expert demonstrations and remain static after pretraining, limiting their ability to handle real-world variations. Even with extensive training data, unforeseen challenges--especially those that fundamentally alter task dynamics, such as unexpected obstacles or spatial constraints--can cause assistive policies to break down, leading to ineffective or unreliable assistance. To address this, we propose ILSA, an Incrementally Learned Shared Autonomy framework that continuously refines its assistive policy through user interactions, adapting to real-world challenges beyond the scope of pre-collected data. At the core of ILSA is a structured fine-tuning mechanism that enables continual improvement with each interaction by effectively integrating limited new interaction data while preserving prior knowledge, ensuring a balance between adaptation and generalization. A user study with 20 participants demonstrates ILSA's effectiveness, showing faster task completion and improved user experience compared to static alternatives. Code and videos are available at https://ilsa-robo.github.io/.
>
---
#### [replaced 018] HuNavSim 2.0: An Enhanced Human Navigation Simulator for Human-Aware Robot Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.17317v2](http://arxiv.org/pdf/2507.17317v2)**

> **作者:** Miguel Escudero-Jiménez; Noé Pérez-Higueras; Andrés Martínez-Silva; Fernando Caballero; Luis Merino
>
> **备注:** Preprint submitted to the 8th Iberian Robotics Conference (ROBOT 2025)
>
> **摘要:** This work presents a new iteration of the Human Navigation Simulator (HuNavSim), a novel open-source tool for the simulation of different human-agent navigation behaviors in scenarios with mobile robots. The tool, programmed under the ROS 2 framework, can be used together with different well-known robotics simulators such as Gazebo or NVidia Isaac Sim. The main goal is to facilitate the development and evaluation of human-aware robot navigation systems in simulation. In this new version, several features have been improved and new ones added, such as the extended set of actions and conditions that can be combined in Behavior Trees to compound complex and realistic human behaviors.
>
---
#### [replaced 019] Integration of a Graph-Based Path Planner and Mixed-Integer MPC for Robot Navigation in Cluttered Environments
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2504.13372v2](http://arxiv.org/pdf/2504.13372v2)**

> **作者:** Joshua A. Robbins; Stephen J. Harnett; Andrew F. Thompson; Sean Brennan; Herschel C. Pangborn
>
> **摘要:** The ability to update a path plan is a required capability for autonomous mobile robots navigating through uncertain environments. This paper proposes a re-planning strategy using a multilayer planning and control framework for cases where the robot's environment is partially known. A medial axis graph-based planner defines a global path plan based on known obstacles, where each edge in the graph corresponds to a unique corridor. A mixed-integer model predictive control (MPC) method detects if a terminal constraint derived from the global plan is infeasible, subject to a non-convex description of the local environment. Infeasibility detection is used to trigger efficient global re-planning via medial axis graph edge deletion. The proposed re-planning strategy is demonstrated experimentally.
>
---
#### [replaced 020] SE-VLN: A Self-Evolving Vision-Language Navigation Framework Based on Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.13152v2](http://arxiv.org/pdf/2507.13152v2)**

> **作者:** Xiangyu Dong; Haoran Zhao; Jiang Gao; Haozhou Li; Xiaoguang Ma; Yaoming Zhou; Fuhai Chen; Juan Liu
>
> **摘要:** Recent advances in vision-language navigation (VLN) were mainly attributed to emerging large language models (LLMs). These methods exhibited excellent generalization capabilities in instruction understanding and task reasoning. However, they were constrained by the fixed knowledge bases and reasoning abilities of LLMs, preventing fully incorporating experiential knowledge and thus resulting in a lack of efficient evolutionary capacity. To address this, we drew inspiration from the evolution capabilities of natural agents, and proposed a self-evolving VLN framework (SE-VLN) to endow VLN agents with the ability to continuously evolve during testing. To the best of our knowledge, it was the first time that an multimodal LLM-powered self-evolving VLN framework was proposed. Specifically, SE-VLN comprised three core modules, i.e., a hierarchical memory module to transfer successful and failure cases into reusable knowledge, a retrieval-augmented thought-based reasoning module to retrieve experience and enable multi-step decision-making, and a reflection module to realize continual evolution. Comprehensive tests illustrated that the SE-VLN achieved navigation success rates of 57% and 35.2% in unseen environments, representing absolute performance improvements of 23.9% and 15.0% over current state-of-the-art methods on R2R and REVERSE datasets, respectively. Moreover, the SE-VLN showed performance improvement with increasing experience repository, elucidating its great potential as a self-evolving agent framework for VLN.
>
---
#### [replaced 021] Prolonging Tool Life: Learning Skillful Use of General-purpose Tools through Lifespan-guided Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.17275v2](http://arxiv.org/pdf/2507.17275v2)**

> **作者:** Po-Yen Wu; Cheng-Yu Kuo; Yuki Kadokawa; Takamitsu Matsubara
>
> **备注:** Under review
>
> **摘要:** In inaccessible environments with uncertain task demands, robots often rely on general-purpose tools that lack predefined usage strategies. These tools are not tailored for particular operations, making their longevity highly sensitive to how they are used. This creates a fundamental challenge: how can a robot learn a tool-use policy that both completes the task and prolongs the tool's lifespan? In this work, we address this challenge by introducing a reinforcement learning (RL) framework that incorporates tool lifespan as a factor during policy optimization. Our framework leverages Finite Element Analysis (FEA) and Miner's Rule to estimate Remaining Useful Life (RUL) based on accumulated stress, and integrates the RUL into the RL reward to guide policy learning toward lifespan-guided behavior. To handle the fact that RUL can only be estimated after task execution, we introduce an Adaptive Reward Normalization (ARN) mechanism that dynamically adjusts reward scaling based on estimated RULs, ensuring stable learning signals. We validate our method across simulated and real-world tool use tasks, including Object-Moving and Door-Opening with multiple general-purpose tools. The learned policies consistently prolong tool lifespan (up to 8.01x in simulation) and transfer effectively to real-world settings, demonstrating the practical value of learning lifespan-guided tool use strategies.
>
---
#### [replaced 022] Semi-autonomous Prosthesis Control Using Minimal Depth Information and Vibrotactile Feedback
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2210.00541v2](http://arxiv.org/pdf/2210.00541v2)**

> **作者:** Miguel Nobre Castro; Strahinja Dosen
>
> **摘要:** Semi-autonomous prosthesis controllers based on computer vision improve performance while reducing cognitive effort. However, controllers relying on full-depth data face challenges in being deployed as embedded prosthesis controllers due to the computational demands of processing point clouds. To address this, the present study proposes a method to reconstruct the shape of various daily objects from minimal depth data. This is achieved using four concurrent laser scanner lines instead of a full point cloud. These lines represent the partial contours of an object's cross-section, enabling its dimensions and orientation to be reconstructed using simple geometry. A control prototype was implemented using a depth sensor with four laser scanners. Vibrotactile feedback was also designed to help users to correctly aim the sensor at target objects. Ten able-bodied volunteers used a prosthesis equipped with the novel controller to grasp ten objects of varying shapes, sizes, and orientations. For comparison, they also tested an existing benchmark controller that used full-depth information. The results showed that the novel controller handled all objects and, while performance improved with training, it remained slightly below that of the benchmark. This marks an important step towards a compact vision-based system for embedded depth sensing in prosthesis grasping.
>
---
#### [replaced 023] Cuddle-Fish: Exploring a Soft Floating Robot with Flapping Wings for Physical Interactions
- **分类: cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.01293v2](http://arxiv.org/pdf/2504.01293v2)**

> **作者:** Mingyang Xu; Jiayi Shao; Yulan Ju; Ximing Shen; Qingyuan Gao; Weijen Chen; Qing Zhang; Yun Suen Pai; Giulia Barbareschi; Matthias Hoppe; Kouta Minamizawa; Kai Kunze
>
> **备注:** Augmented Humans International Conference 2025 (AHs '25)
>
> **摘要:** Flying robots, such as quadrotor drones, offer new possibilities for human-robot interaction but often pose safety risks due to fast-spinning propellers, rigid structures, and noise. In contrast, lighter-than-air flapping-wing robots, inspired by animal movement, offer a soft, quiet, and touch-safe alternative. Building on these advantages, we present Cuddle-Fish, a soft flapping-wing floating robot designed for close-proximity interactions in indoor spaces. Through a user study with 24 participants, we explored their perceptions of the robot and experiences during a series of co-located demonstrations in which the robot moved near them. Results showed that participants felt safe, willingly engaged in touch-based interactions with the robot, and exhibited spontaneous affective behaviours, such as patting, stroking, hugging, and cheek-touching, without external prompting. They also reported positive emotional responses towards the robot. These findings suggest that the soft floating robot with flapping wings can serve as a novel and socially acceptable alternative to traditional rigid flying robots, opening new potential for applications in companionship, affective interaction, and play in everyday indoor environments.
>
---
