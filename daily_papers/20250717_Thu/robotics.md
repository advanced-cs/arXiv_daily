# 机器人 cs.RO

- **最新发布 33 篇**

- **更新 22 篇**

## 最新发布

#### [new 001] Towards Autonomous Riding: A Review of Perception, Planning, and Control in Intelligent Two-Wheelers
- **分类: cs.RO; cs.CV; 93C85; F.2.2; I.2.7**

- **简介: 该论文属于自主骑行研究，旨在解决两轮车自动驾驶的安全与稳定性问题，综述了感知、规划与控制技术，分析了现有研究的不足并提出未来方向。**

- **链接: [http://arxiv.org/pdf/2507.11852v1](http://arxiv.org/pdf/2507.11852v1)**

> **作者:** Mohammed Hassanin; Mohammad Abu Alsheikh; Carlos C. N. Kuhn; Damith Herath; Dinh Thai Hoang; Ibrahim Radwan
>
> **备注:** 17 pages
>
> **摘要:** The rapid adoption of micromobility solutions, particularly two-wheeled vehicles like e-scooters and e-bikes, has created an urgent need for reliable autonomous riding (AR) technologies. While autonomous driving (AD) systems have matured significantly, AR presents unique challenges due to the inherent instability of two-wheeled platforms, limited size, limited power, and unpredictable environments, which pose very serious concerns about road users' safety. This review provides a comprehensive analysis of AR systems by systematically examining their core components, perception, planning, and control, through the lens of AD technologies. We identify critical gaps in current AR research, including a lack of comprehensive perception systems for various AR tasks, limited industry and government support for such developments, and insufficient attention from the research community. The review analyses the gaps of AR from the perspective of AD to highlight promising research directions, such as multimodal sensor techniques for lightweight platforms and edge deep learning architectures. By synthesising insights from AD research with the specific requirements of AR, this review aims to accelerate the development of safe, efficient, and scalable autonomous riding systems for future urban mobility.
>
---
#### [new 002] A Multi-Level Similarity Approach for Single-View Object Grasping: Matching, Planning, and Fine-Tuning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于单视角物体抓取任务，解决未知物体抓取的鲁棒性问题。通过多层级相似性匹配与优化，提升抓取准确性。**

- **链接: [http://arxiv.org/pdf/2507.11938v1](http://arxiv.org/pdf/2507.11938v1)**

> **作者:** Hao Chen; Takuya Kiyokawa; Zhengtao Hu; Weiwei Wan; Kensuke Harada
>
> **备注:** Accepted by IEEE T-RO
>
> **摘要:** Grasping unknown objects from a single view has remained a challenging topic in robotics due to the uncertainty of partial observation. Recent advances in large-scale models have led to benchmark solutions such as GraspNet-1Billion. However, such learning-based approaches still face a critical limitation in performance robustness for their sensitivity to sensing noise and environmental changes. To address this bottleneck in achieving highly generalized grasping, we abandon the traditional learning framework and introduce a new perspective: similarity matching, where similar known objects are utilized to guide the grasping of unknown target objects. We newly propose a method that robustly achieves unknown-object grasping from a single viewpoint through three key steps: 1) Leverage the visual features of the observed object to perform similarity matching with an existing database containing various object models, identifying potential candidates with high similarity; 2) Use the candidate models with pre-existing grasping knowledge to plan imitative grasps for the unknown target object; 3) Optimize the grasp quality through a local fine-tuning process. To address the uncertainty caused by partial and noisy observation, we propose a multi-level similarity matching framework that integrates semantic, geometric, and dimensional features for comprehensive evaluation. Especially, we introduce a novel point cloud geometric descriptor, the C-FPFH descriptor, which facilitates accurate similarity assessment between partial point clouds of observed objects and complete point clouds of database models. In addition, we incorporate the use of large language models, introduce the semi-oriented bounding box, and develop a novel point cloud registration approach based on plane detection to enhance matching accuracy under single-view conditions. Videos are available at https://youtu.be/qQDIELMhQmk.
>
---
#### [new 003] Generating Actionable Robot Knowledge Bases by Combining 3D Scene Graphs with Robot Ontologies
- **分类: cs.RO**

- **简介: 该论文属于机器人知识表示任务，解决环境数据格式不统一问题，通过构建统一场景图并结合机器人本体论，实现环境数据到可操作知识的转换。**

- **链接: [http://arxiv.org/pdf/2507.11770v1](http://arxiv.org/pdf/2507.11770v1)**

> **作者:** Giang Nguyen; Mihai Pomarlan; Sascha Jongebloed; Nils Leusmann; Minh Nhat Vu; Michael Beetz
>
> **备注:** 8 pages, 7 figures, IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS2025)
>
> **摘要:** In robotics, the effective integration of environmental data into actionable knowledge remains a significant challenge due to the variety and incompatibility of data formats commonly used in scene descriptions, such as MJCF, URDF, and SDF. This paper presents a novel approach that addresses these challenges by developing a unified scene graph model that standardizes these varied formats into the Universal Scene Description (USD) format. This standardization facilitates the integration of these scene graphs with robot ontologies through semantic reporting, enabling the translation of complex environmental data into actionable knowledge essential for cognitive robotic control. We evaluated our approach by converting procedural 3D environments into USD format, which is then annotated semantically and translated into a knowledge graph to effectively answer competency questions, demonstrating its utility for real-time robotic decision-making. Additionally, we developed a web-based visualization tool to support the semantic mapping process, providing users with an intuitive interface to manage the 3D environment.
>
---
#### [new 004] Robust Route Planning for Sidewalk Delivery Robots
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决 sidewalk 机器人在不确定环境下的可靠路线问题。通过优化与仿真结合，比较多种方法提升导航效率与可靠性。**

- **链接: [http://arxiv.org/pdf/2507.12067v1](http://arxiv.org/pdf/2507.12067v1)**

> **作者:** Xing Tong; Michele D. Simoni
>
> **摘要:** Sidewalk delivery robots are a promising solution for urban freight distribution, reducing congestion compared to trucks and providing a safer, higher-capacity alternative to drones. However, unreliable travel times on sidewalks due to pedestrian density, obstacles, and varying infrastructure conditions can significantly affect their efficiency. This study addresses the robust route planning problem for sidewalk robots, explicitly accounting for travel time uncertainty due to varying sidewalk conditions. Optimization is integrated with simulation to reproduce the effect of obstacles and pedestrian flows and generate realistic travel times. The study investigates three different approaches to derive uncertainty sets, including budgeted, ellipsoidal, and support vector clustering (SVC)-based methods, along with a distributionally robust method to solve the shortest path (SP) problem. A realistic case study reproducing pedestrian patterns in Stockholm's city center is used to evaluate the efficiency of robust routing across various robot designs and environmental conditions. The results show that, when compared to a conventional SP, robust routing significantly enhances operational reliability under variable sidewalk conditions. The Ellipsoidal and DRSP approaches outperform the other methods, yielding the most efficient paths in terms of average and worst-case delay. Sensitivity analyses reveal that robust approaches consistently outperform the conventional SP, particularly for sidewalk delivery robots that are wider, slower, and have more conservative navigation behaviors. These benefits are even more pronounced in adverse weather conditions and high pedestrian congestion scenarios.
>
---
#### [new 005] HCOMC: A Hierarchical Cooperative On-Ramp Merging Control Framework in Mixed Traffic Environment on Two-Lane Highways
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文属于交通控制任务，解决混合交通环境下高速公路匝道合并问题。通过构建HCOMC框架，提升合并安全性与效率。**

- **链接: [http://arxiv.org/pdf/2507.11621v1](http://arxiv.org/pdf/2507.11621v1)**

> **作者:** Tianyi Wang; Yangyang Wang; Jie Pan; Junfeng Jiao; Christian Claudel
>
> **备注:** 7 pages, 2 figures, 3 tables, accepted for IEEE International Conference on Intelligent Transportation Systems (ITSC) 2025
>
> **摘要:** Highway on-ramp merging areas are common bottlenecks to traffic congestion and accidents. Currently, a cooperative control strategy based on connected and automated vehicles (CAVs) is a fundamental solution to this problem. While CAVs are not fully widespread, it is necessary to propose a hierarchical cooperative on-ramp merging control (HCOMC) framework for heterogeneous traffic flow on two-lane highways to address this gap. This paper extends longitudinal car-following models based on the intelligent driver model and lateral lane-changing models using the quintic polynomial curve to account for human-driven vehicles (HDVs) and CAVs, comprehensively considering human factors and cooperative adaptive cruise control. Besides, this paper proposes a HCOMC framework, consisting of a hierarchical cooperative planning model based on the modified virtual vehicle model, a discretionary lane-changing model based on game theory, and a multi-objective optimization model using the elitist non-dominated sorting genetic algorithm to ensure the safe, smooth, and efficient merging process. Then, the performance of our HCOMC is analyzed under different traffic densities and CAV penetration rates through simulation. The findings underscore our HCOMC's pronounced comprehensive advantages in enhancing the safety of group vehicles, stabilizing and expediting merging process, optimizing traffic efficiency, and economizing fuel consumption compared with benchmarks.
>
---
#### [new 006] Assessing the Value of Visual Input: A Benchmark of Multimodal Large Language Models for Robotic Path Planning
- **分类: cs.RO**

- **简介: 该论文研究多模态大语言模型在机器人路径规划中的应用，评估视觉输入的有效性，解决如何提升路径规划性能的问题。**

- **链接: [http://arxiv.org/pdf/2507.12391v1](http://arxiv.org/pdf/2507.12391v1)**

> **作者:** Jacinto Colan; Ana Davila; Yasuhisa Hasegawa
>
> **备注:** Accepted at the 2025 SICE Festival with Annual Conference (SICE FES)
>
> **摘要:** Large Language Models (LLMs) show potential for enhancing robotic path planning. This paper assesses visual input's utility for multimodal LLMs in such tasks via a comprehensive benchmark. We evaluated 15 multimodal LLMs on generating valid and optimal paths in 2D grid environments, simulating simplified robotic planning, comparing text-only versus text-plus-visual inputs across varying model sizes and grid complexities. Our results indicate moderate success rates on simpler small grids, where visual input or few-shot text prompting offered some benefits. However, performance significantly degraded on larger grids, highlighting a scalability challenge. While larger models generally achieved higher average success, the visual modality was not universally dominant over well-structured text for these multimodal systems, and successful paths on simpler grids were generally of high quality. These results indicate current limitations in robust spatial reasoning, constraint adherence, and scalable multimodal integration, identifying areas for future LLM development in robotic path planning.
>
---
#### [new 007] Design and Development of an Automated Contact Angle Tester (ACAT) for Surface Wettability Measurement
- **分类: cs.RO; physics.ins-det**

- **简介: 该论文属于自动化检测任务，旨在解决手动接触角测试的不足，通过设计ACAT系统实现3D打印材料表面润湿性的自动化测量。**

- **链接: [http://arxiv.org/pdf/2507.12431v1](http://arxiv.org/pdf/2507.12431v1)**

> **作者:** Connor Burgess; Kyle Douin; Amir Kordijazi
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** The Automated Contact Angle Tester (ACAT) is a fully integrated robotic work cell developed to automate the measurement of surface wettability on 3D-printed materials. Designed for precision, repeatability, and safety, ACAT addresses the limitations of manual contact angle testing by combining programmable robotics, precise liquid dispensing, and a modular software-hardware architecture. The system is composed of three core subsystems: (1) an electrical system including power, control, and safety circuits compliant with industrial standards such as NEC 70, NFPA 79, and UL 508A; (2) a software control system based on a Raspberry Pi and Python, featuring fault detection, GPIO logic, and operator interfaces; and (3) a mechanical system that includes a 3-axis Cartesian robot, pneumatic actuation, and a precision liquid dispenser enclosed within a safety-certified frame. The ACAT enables high-throughput, automated surface characterization and provides a robust platform for future integration into smart manufacturing and materials discovery workflows. This paper details the design methodology, implementation strategies, and system integration required to develop the ACAT platform.
>
---
#### [new 008] Probabilistic Safety Verification for an Autonomous Ground Vehicle: A Situation Coverage Grid Approach
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶安全验证任务，旨在解决复杂环境下的安全问题。通过构建情境覆盖网格和概率模型，验证系统安全性并提供量化保障。**

- **链接: [http://arxiv.org/pdf/2507.12158v1](http://arxiv.org/pdf/2507.12158v1)**

> **作者:** Nawshin Mannan Proma; Gricel Vázquez; Sepeedeh Shahbeigi; Arjun Badyal; Victoria Hodge
>
> **备注:** 6 pages, 6 figures
>
> **摘要:** As industrial autonomous ground vehicles are increasingly deployed in safety-critical environments, ensuring their safe operation under diverse conditions is paramount. This paper presents a novel approach for their safety verification based on systematic situation extraction, probabilistic modelling and verification. We build upon the concept of a situation coverage grid, which exhaustively enumerates environmental configurations relevant to the vehicle's operation. This grid is augmented with quantitative probabilistic data collected from situation-based system testing, capturing probabilistic transitions between situations. We then generate a probabilistic model that encodes the dynamics of both normal and unsafe system behaviour. Safety properties extracted from hazard analysis and formalised in temporal logic are verified through probabilistic model checking against this model. The results demonstrate that our approach effectively identifies high-risk situations, provides quantitative safety guarantees, and supports compliance with regulatory standards, thereby contributing to the robust deployment of autonomous systems.
>
---
#### [new 009] Tree-SLAM: semantic object SLAM for efficient mapping of individual trees in orchards
- **分类: cs.RO**

- **简介: 该论文属于农业机器人领域的定位与建图任务，旨在解决果园中树木精准定位问题。通过语义SLAM技术，结合RGB-D图像和数据关联算法，实现高精度树位地图构建。**

- **链接: [http://arxiv.org/pdf/2507.12093v1](http://arxiv.org/pdf/2507.12093v1)**

> **作者:** David Rapado-Rincon; Gert Kootstra
>
> **备注:** Paper submitted to Smart Agricultural Technology
>
> **摘要:** Accurate mapping of individual trees is an important component for precision agriculture in orchards, as it allows autonomous robots to perform tasks like targeted operations or individual tree monitoring. However, creating these maps is challenging because GPS signals are often unreliable under dense tree canopies. Furthermore, standard Simultaneous Localization and Mapping (SLAM) approaches struggle in orchards because the repetitive appearance of trees can confuse the system, leading to mapping errors. To address this, we introduce Tree-SLAM, a semantic SLAM approach tailored for creating maps of individual trees in orchards. Utilizing RGB-D images, our method detects tree trunks with an instance segmentation model, estimates their location and re-identifies them using a cascade-graph-based data association algorithm. These re-identified trunks serve as landmarks in a factor graph framework that integrates noisy GPS signals, odometry, and trunk observations. The system produces maps of individual trees with a geo-localization error as low as 18 cm, which is less than 20\% of the planting distance. The proposed method was validated on diverse datasets from apple and pear orchards across different seasons, demonstrating high mapping accuracy and robustness in scenarios with unreliable GPS signals.
>
---
#### [new 010] IANN-MPPI: Interaction-Aware Neural Network-Enhanced Model Predictive Path Integral Approach for Autonomous Driving
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于自动驾驶任务，解决密集交通中路径规划问题，提出IANN-MPPI方法，通过交互感知预测周围车辆反应，提升变道效率。**

- **链接: [http://arxiv.org/pdf/2507.11940v1](http://arxiv.org/pdf/2507.11940v1)**

> **作者:** Kanghyun Ryu; Minjun Sung; Piyush Gupta; Jovin D'sa; Faizan M. Tariq; David Isele; Sangjae Bae
>
> **备注:** To be published in The IEEE International Conference on Intelligent Transportation Systems (ITSC) 2025
>
> **摘要:** Motion planning for autonomous vehicles (AVs) in dense traffic is challenging, often leading to overly conservative behavior and unmet planning objectives. This challenge stems from the AVs' limited ability to anticipate and respond to the interactive behavior of surrounding agents. Traditional decoupled prediction and planning pipelines rely on non-interactive predictions that overlook the fact that agents often adapt their behavior in response to the AV's actions. To address this, we propose Interaction-Aware Neural Network-Enhanced Model Predictive Path Integral (IANN-MPPI) control, which enables interactive trajectory planning by predicting how surrounding agents may react to each control sequence sampled by MPPI. To improve performance in structured lane environments, we introduce a spline-based prior for the MPPI sampling distribution, enabling efficient lane-changing behavior. We evaluate IANN-MPPI in a dense traffic merging scenario, demonstrating its ability to perform efficient merging maneuvers. Our project website is available at https://sites.google.com/berkeley.edu/iann-mppi
>
---
#### [new 011] Hybrid Conformal Prediction-based Risk-Aware Model Predictive Planning in Dense, Uncertain Environments
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于路径规划任务，解决密集不确定环境中实时路径规划问题。提出HyPRAP框架，结合多模型预测与风险评估，提升安全性和计算效率。**

- **链接: [http://arxiv.org/pdf/2507.11920v1](http://arxiv.org/pdf/2507.11920v1)**

> **作者:** Jeongyong Yang; KwangBin Lee; SooJean Han
>
> **摘要:** Real-time path planning in dense, uncertain environments remains a challenging problem, as predicting the future motions of numerous dynamic obstacles is computationally burdensome and unrealistic. To address this, we introduce Hybrid Prediction-based Risk-Aware Planning (HyPRAP), a prediction-based risk-aware path-planning framework which uses a hybrid combination of models to predict local obstacle movement. HyPRAP uses a novel Prediction-based Collision Risk Index (P-CRI) to evaluate the risk posed by each obstacle, enabling the selective use of predictors based on whether the agent prioritizes high predictive accuracy or low computational prediction overhead. This selective routing enables the agent to focus on high-risk obstacles while ignoring or simplifying low-risk ones, making it suitable for environments with a large number of obstacles. Moreover, HyPRAP incorporates uncertainty quantification through hybrid conformal prediction by deriving confidence bounds simultaneously achieved by multiple predictions across different models. Theoretical analysis demonstrates that HyPRAP effectively balances safety and computational efficiency by leveraging the diversity of prediction models. Extensive simulations validate these insights for more general settings, confirming that HyPRAP performs better compared to single predictor methods, and P-CRI performs better over naive proximity-based risk assessment.
>
---
#### [new 012] Robust Planning for Autonomous Vehicles with Diffusion-Based Failure Samplers
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，解决高风险区域碰撞问题。通过生成对抗模型生成故障样本，提升车辆规划安全性与可靠性。**

- **链接: [http://arxiv.org/pdf/2507.11991v1](http://arxiv.org/pdf/2507.11991v1)**

> **作者:** Juanran Wang; Marc R. Schlichting; Mykel J. Kochenderfer
>
> **摘要:** High-risk traffic zones such as intersections are a major cause of collisions. This study leverages deep generative models to enhance the safety of autonomous vehicles in an intersection context. We train a 1000-step denoising diffusion probabilistic model to generate collision-causing sensor noise sequences for an autonomous vehicle navigating a four-way intersection based on the current relative position and velocity of an intruder. Using the generative adversarial architecture, the 1000-step model is distilled into a single-step denoising diffusion model which demonstrates fast inference speed while maintaining similar sampling quality. We demonstrate one possible application of the single-step model in building a robust planner for the autonomous vehicle. The planner uses the single-step model to efficiently sample potential failure cases based on the currently measured traffic state to inform its decision-making. Through simulation experiments, the robust planner demonstrates significantly lower failure rate and delay rate compared with the baseline Intelligent Driver Model controller.
>
---
#### [new 013] A Roadmap for Climate-Relevant Robotics Research
- **分类: cs.RO; cs.AI; cs.LG; cs.SY; eess.SY**

- **简介: 该论文属于跨学科研究任务，旨在通过机器人技术解决气候变化相关问题，如能源优化、农业和环境监测等，提出合作方向与具体应用方案。**

- **链接: [http://arxiv.org/pdf/2507.11623v1](http://arxiv.org/pdf/2507.11623v1)**

> **作者:** Alan Papalia; Charles Dawson; Laurentiu L. Anton; Norhan Magdy Bayomi; Bianca Champenois; Jung-Hoon Cho; Levi Cai; Joseph DelPreto; Kristen Edwards; Bilha-Catherine Githinji; Cameron Hickert; Vindula Jayawardana; Matthew Kramer; Shreyaa Raghavan; David Russell; Shide Salimi; Jingnan Shi; Soumya Sudhakar; Yanwei Wang; Shouyi Wang; Luca Carlone; Vijay Kumar; Daniela Rus; John E. Fernandez; Cathy Wu; George Kantor; Derek Young; Hanumant Singh
>
> **摘要:** Climate change is one of the defining challenges of the 21st century, and many in the robotics community are looking for ways to contribute. This paper presents a roadmap for climate-relevant robotics research, identifying high-impact opportunities for collaboration between roboticists and experts across climate domains such as energy, the built environment, transportation, industry, land use, and Earth sciences. These applications include problems such as energy systems optimization, construction, precision agriculture, building envelope retrofits, autonomous trucking, and large-scale environmental monitoring. Critically, we include opportunities to apply not only physical robots but also the broader robotics toolkit - including planning, perception, control, and estimation algorithms - to climate-relevant problems. A central goal of this roadmap is to inspire new research directions and collaboration by highlighting specific, actionable problems at the intersection of robotics and climate. This work represents a collaboration between robotics researchers and domain experts in various climate disciplines, and it serves as an invitation to the robotics community to bring their expertise to bear on urgent climate priorities.
>
---
#### [new 014] A Fast Method for Planning All Optimal Homotopic Configurations for Tethered Robots and Its Extended Applications
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决 tethered 机器人在路径规划中的拓扑约束问题，提出 CDT-TCS 算法及三个应用算法，提升规划效率与可行性。**

- **链接: [http://arxiv.org/pdf/2507.11880v1](http://arxiv.org/pdf/2507.11880v1)**

> **作者:** Jinyuan Liu; Minglei Fu; Ling Shi; Chenguang Yang; Wenan Zhang
>
> **备注:** 37 pages, 33 figures
>
> **摘要:** Tethered robots play a pivotal role in specialized environments such as disaster response and underground exploration, where their stable power supply and reliable communication offer unparalleled advantages. However, their motion planning is severely constrained by tether length limitations and entanglement risks, posing significant challenges to achieving optimal path planning. To address these challenges, this study introduces CDT-TCS (Convex Dissection Topology-based Tethered Configuration Search), a novel algorithm that leverages CDT Encoding as a homotopy invariant to represent topological states of paths. By integrating algebraic topology with geometric optimization, CDT-TCS efficiently computes the complete set of optimal feasible configurations for tethered robots at all positions in 2D environments through a single computation. Building on this foundation, we further propose three application-specific algorithms: i) CDT-TPP for optimal tethered path planning, ii) CDT-TMV for multi-goal visiting with tether constraints, iii) CDT-UTPP for distance-optimal path planning of untethered robots. All theoretical results and propositions underlying these algorithms are rigorously proven and thoroughly discussed in this paper. Extensive simulations demonstrate that the proposed algorithms significantly outperform state-of-the-art methods in their respective problem domains. Furthermore, real-world experiments on robotic platforms validate the practicality and engineering value of the proposed framework.
>
---
#### [new 015] UniLGL: Learning Uniform Place Recognition for FOV-limited/Panoramic LiDAR Global Localization
- **分类: cs.RO**

- **简介: 该论文属于LiDAR全局定位任务，解决异构传感器下定位不统一问题，提出UniLGL方法实现空间、材料和传感器类型统一的定位。**

- **链接: [http://arxiv.org/pdf/2507.12194v1](http://arxiv.org/pdf/2507.12194v1)**

> **作者:** Hongming Shen; Xun Chen; Yulin Hui; Zhenyu Wu; Wei Wang; Qiyang Lyu; Tianchen Deng; Danwei Wang
>
> **摘要:** Existing LGL methods typically consider only partial information (e.g., geometric features) from LiDAR observations or are designed for homogeneous LiDAR sensors, overlooking the uniformity in LGL. In this work, a uniform LGL method is proposed, termed UniLGL, which simultaneously achieves spatial and material uniformity, as well as sensor-type uniformity. The key idea of the proposed method is to encode the complete point cloud, which contains both geometric and material information, into a pair of BEV images (i.e., a spatial BEV image and an intensity BEV image). An end-to-end multi-BEV fusion network is designed to extract uniform features, equipping UniLGL with spatial and material uniformity. To ensure robust LGL across heterogeneous LiDAR sensors, a viewpoint invariance hypothesis is introduced, which replaces the conventional translation equivariance assumption commonly used in existing LPR networks and supervises UniLGL to achieve sensor-type uniformity in both global descriptors and local feature representations. Finally, based on the mapping between local features on the 2D BEV image and the point cloud, a robust global pose estimator is derived that determines the global minimum of the global pose on SE(3) without requiring additional registration. To validate the effectiveness of the proposed uniform LGL, extensive benchmarks are conducted in real-world environments, and the results show that the proposed UniLGL is demonstratively competitive compared to other State-of-the-Art LGL methods. Furthermore, UniLGL has been deployed on diverse platforms, including full-size trucks and agile Micro Aerial Vehicles (MAVs), to enable high-precision localization and mapping as well as multi-MAV collaborative exploration in port and forest environments, demonstrating the applicability of UniLGL in industrial and field scenarios.
>
---
#### [new 016] The Developments and Challenges towards Dexterous and Embodied Robotic Manipulation: A Survey
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决人类级灵巧操作难题。通过分析数据收集与学习框架，探讨提升机器人操作能力的关键挑战。**

- **链接: [http://arxiv.org/pdf/2507.11840v1](http://arxiv.org/pdf/2507.11840v1)**

> **作者:** Gaofeng Li; Ruize Wang; Peisen Xu; Qi Ye; Jiming Chen
>
> **摘要:** Achieving human-like dexterous robotic manipulation remains a central goal and a pivotal challenge in robotics. The development of Artificial Intelligence (AI) has allowed rapid progress in robotic manipulation. This survey summarizes the evolution of robotic manipulation from mechanical programming to embodied intelligence, alongside the transition from simple grippers to multi-fingered dexterous hands, outlining key characteristics and main challenges. Focusing on the current stage of embodied dexterous manipulation, we highlight recent advances in two critical areas: dexterous manipulation data collection (via simulation, human demonstrations, and teleoperation) and skill-learning frameworks (imitation and reinforcement learning). Then, based on the overview of the existing data collection paradigm and learning framework, three key challenges restricting the development of dexterous robotic manipulation are summarized and discussed.
>
---
#### [new 017] Leveraging Sidewalk Robots for Walkability-Related Analyses
- **分类: cs.RO**

- **简介: 该论文属于城市规划任务，旨在解决传统数据收集方法成本高、难以扩展的问题。通过部署带传感器的送餐机器人，实时采集人行道数据，分析其对步行性的影响。**

- **链接: [http://arxiv.org/pdf/2507.12148v1](http://arxiv.org/pdf/2507.12148v1)**

> **作者:** Xing Tong; Michele D. Simoni; Kaj Munhoz Arfvidsson; Jonas Mårtensson
>
> **摘要:** Walkability is a key component of sustainable urban development, while collecting detailed data on its related features remains challenging due to the high costs and limited scalability of traditional methods. Sidewalk delivery robots, increasingly deployed in urban environments, offer a promising solution to these limitations. This paper explores how these robots can serve as mobile data collection platforms, capturing sidewalk-level features related to walkability in a scalable, automated, and real-time manner. A sensor-equipped robot was deployed on a sidewalk network at KTH in Stockholm, completing 101 trips covering 900 segments. From the collected data, different typologies of features are derived, including robot trip characteristics (e.g., speed, duration), sidewalk conditions (e.g., width, surface unevenness), and sidewalk utilization (e.g., pedestrian density). Their walkability-related implications were investigated with a series of analyses. The results demonstrate that pedestrian movement patterns are strongly influenced by sidewalk characteristics, with higher density, reduced width, and surface irregularity associated with slower and more variable trajectories. Notably, robot speed closely mirrors pedestrian behavior, highlighting its potential as a proxy for assessing pedestrian dynamics. The proposed framework enables continuous monitoring of sidewalk conditions and pedestrian behavior, contributing to the development of more walkable, inclusive, and responsive urban environments.
>
---
#### [new 018] CoNav Chair: Development and Evaluation of a Shared Control based Wheelchair for the Built Environment
- **分类: cs.RO**

- **简介: 该论文属于智能轮椅导航任务，旨在解决传统轮椅在复杂环境中的操控难题。提出CoNav Chair系统，采用共享控制策略提升安全性与效率，并通过实验验证其性能。**

- **链接: [http://arxiv.org/pdf/2507.11716v1](http://arxiv.org/pdf/2507.11716v1)**

> **作者:** Yifan Xu; Qianwei Wang; Jordan Lillie; Vineet Kamat; Carol Menassa; Clive D'Souza
>
> **备注:** 13 pages, 10 figures
>
> **摘要:** As the global population of people with disabilities (PWD) continues to grow, so will the need for mobility solutions that promote independent living and social integration. Wheelchairs are vital for the mobility of PWD in both indoor and outdoor environments. The current SOTA in powered wheelchairs is based on either manually controlled or fully autonomous modes of operation, offering limited flexibility and often proving difficult to navigate in spatially constrained environments. Moreover, research on robotic wheelchairs has focused predominantly on complete autonomy or improved manual control; approaches that can compromise efficiency and user trust. To overcome these challenges, this paper introduces the CoNav Chair, a smart wheelchair based on the Robot Operating System (ROS) and featuring shared control navigation and obstacle avoidance capabilities that are intended to enhance navigational efficiency, safety, and ease of use for the user. The paper outlines the CoNav Chair's design and presents a preliminary usability evaluation comparing three distinct navigation modes, namely, manual, shared, and fully autonomous, conducted with 21 healthy, unimpaired participants traversing an indoor building environment. Study findings indicated that the shared control navigation framework had significantly fewer collisions and performed comparably, if not superior to the autonomous and manual modes, on task completion time, trajectory length, and smoothness; and was perceived as being safer and more efficient based on user reported subjective assessments of usability. Overall, the CoNav system demonstrated acceptable safety and performance, laying the foundation for subsequent usability testing with end users, namely, PWDs who rely on a powered wheelchair for mobility.
>
---
#### [new 019] Fast and Scalable Game-Theoretic Trajectory Planning with Intentional Uncertainties
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多智能体轨迹规划任务，解决意图不确定性下的高效可扩展问题。提出基于贝叶斯博弈的方法，并设计分布式算法提升计算效率与规模。**

- **链接: [http://arxiv.org/pdf/2507.12174v1](http://arxiv.org/pdf/2507.12174v1)**

> **作者:** Zhenmin Huang; Yusen Xie; Benshan Ma; Shaojie Shen; Jun Ma
>
> **摘要:** Trajectory planning involving multi-agent interactions has been a long-standing challenge in the field of robotics, primarily burdened by the inherent yet intricate interactions among agents. While game-theoretic methods are widely acknowledged for their effectiveness in managing multi-agent interactions, significant impediments persist when it comes to accommodating the intentional uncertainties of agents. In the context of intentional uncertainties, the heavy computational burdens associated with existing game-theoretic methods are induced, leading to inefficiencies and poor scalability. In this paper, we propose a novel game-theoretic interactive trajectory planning method to effectively address the intentional uncertainties of agents, and it demonstrates both high efficiency and enhanced scalability. As the underpinning basis, we model the interactions between agents under intentional uncertainties as a general Bayesian game, and we show that its agent-form equivalence can be represented as a potential game under certain minor assumptions. The existence and attainability of the optimal interactive trajectories are illustrated, as the corresponding Bayesian Nash equilibrium can be attained by optimizing a unified optimization problem. Additionally, we present a distributed algorithm based on the dual consensus alternating direction method of multipliers (ADMM) tailored to the parallel solving of the problem, thereby significantly improving the scalability. The attendant outcomes from simulations and experiments demonstrate that the proposed method is effective across a range of scenarios characterized by general forms of intentional uncertainties. Its scalability surpasses that of existing centralized and decentralized baselines, allowing for real-time interactive trajectory planning in uncertain game settings.
>
---
#### [new 020] A Review of Generative AI in Aquaculture: Foundations, Applications, and Future Directions for Smart and Sustainable Farming
- **分类: cs.RO**

- **简介: 该论文属于人工智能在水产养殖中的应用研究，旨在探讨生成式AI如何推动智能可持续养殖，解决数据整合与决策优化问题，分析其技术架构与实际应用。**

- **链接: [http://arxiv.org/pdf/2507.11974v1](http://arxiv.org/pdf/2507.11974v1)**

> **作者:** Waseem Akram; Muhayy Ud Din; Lyes Saad Soud; Irfan Hussain
>
> **摘要:** Generative Artificial Intelligence (GAI) has rapidly emerged as a transformative force in aquaculture, enabling intelligent synthesis of multimodal data, including text, images, audio, and simulation outputs for smarter, more adaptive decision-making. As the aquaculture industry shifts toward data-driven, automation and digital integration operations under the Aquaculture 4.0 paradigm, GAI models offer novel opportunities across environmental monitoring, robotics, disease diagnostics, infrastructure planning, reporting, and market analysis. This review presents the first comprehensive synthesis of GAI applications in aquaculture, encompassing foundational architectures (e.g., diffusion models, transformers, and retrieval augmented generation), experimental systems, pilot deployments, and real-world use cases. We highlight GAI's growing role in enabling underwater perception, digital twin modeling, and autonomous planning for remotely operated vehicle (ROV) missions. We also provide an updated application taxonomy that spans sensing, control, optimization, communication, and regulatory compliance. Beyond technical capabilities, we analyze key limitations, including limited data availability, real-time performance constraints, trust and explainability, environmental costs, and regulatory uncertainty. This review positions GAI not merely as a tool but as a critical enabler of smart, resilient, and environmentally aligned aquaculture systems.
>
---
#### [new 021] Regrasp Maps for Sequential Manipulation Planning
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决复杂环境中多次抓取规划问题。通过构建重组地图加速优化求解，提升抓取策略的效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.12407v1](http://arxiv.org/pdf/2507.12407v1)**

> **作者:** Svetlana Levit; Marc Toussaint
>
> **摘要:** We consider manipulation problems in constrained and cluttered settings, which require several regrasps at unknown locations. We propose to inform an optimization-based task and motion planning (TAMP) solver with possible regrasp areas and grasp sequences to speed up the search. Our main idea is to use a state space abstraction, a regrasp map, capturing the combinations of available grasps in different parts of the configuration space, and allowing us to provide the solver with guesses for the mode switches and additional constraints for the object placements. By interleaving the creation of regrasp maps, their adaptation based on failed refinements, and solving TAMP (sub)problems, we are able to provide a robust search method for challenging regrasp manipulation problems.
>
---
#### [new 022] Next-Gen Museum Guides: Autonomous Navigation and Visitor Interaction with an Agentic Robot
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在提升博物馆参观体验。通过设计自主导航与互动机器人，解决游客信息获取和导览问题。**

- **链接: [http://arxiv.org/pdf/2507.12273v1](http://arxiv.org/pdf/2507.12273v1)**

> **作者:** Luca Garello; Francesca Cocchella; Alessandra Sciutti; Manuel Catalano; Francesco Rea
>
> **摘要:** Autonomous robots are increasingly being tested into public spaces to enhance user experiences, particularly in cultural and educational settings. This paper presents the design, implementation, and evaluation of the autonomous museum guide robot Alter-Ego equipped with advanced navigation and interactive capabilities. The robot leverages state-of-the-art Large Language Models (LLMs) to provide real-time, context aware question-and-answer (Q&A) interactions, allowing visitors to engage in conversations about exhibits. It also employs robust simultaneous localization and mapping (SLAM) techniques, enabling seamless navigation through museum spaces and route adaptation based on user requests. The system was tested in a real museum environment with 34 participants, combining qualitative analysis of visitor-robot conversations and quantitative analysis of pre and post interaction surveys. Results showed that the robot was generally well-received and contributed to an engaging museum experience, despite some limitations in comprehension and responsiveness. This study sheds light on HRI in cultural spaces, highlighting not only the potential of AI-driven robotics to support accessibility and knowledge acquisition, but also the current limitations and challenges of deploying such technologies in complex, real-world environments.
>
---
#### [new 023] EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人学习任务，旨在解决数据规模受限问题。通过人类第一视角视频训练视觉-语言-动作模型，将人类动作转换为机器人动作，提升机器人操作能力。**

- **链接: [http://arxiv.org/pdf/2507.12440v1](http://arxiv.org/pdf/2507.12440v1)**

> **作者:** Ruihan Yang; Qinxi Yu; Yecheng Wu; Rui Yan; Borui Li; An-Chieh Cheng; Xueyan Zou; Yunhao Fang; Hongxu Yin; Sifei Liu; Song Han; Yao Lu; Xiaolong Wang
>
> **备注:** More videos can be found on our website: https://rchalyang.github.io/EgoVLA
>
> **摘要:** Real robot data collection for imitation learning has led to significant advancements in robotic manipulation. However, the requirement for robot hardware in the process fundamentally constrains the scale of the data. In this paper, we explore training Vision-Language-Action (VLA) models using egocentric human videos. The benefit of using human videos is not only for their scale but more importantly for the richness of scenes and tasks. With a VLA trained on human video that predicts human wrist and hand actions, we can perform Inverse Kinematics and retargeting to convert the human actions to robot actions. We fine-tune the model using a few robot manipulation demonstrations to obtain the robot policy, namely EgoVLA. We propose a simulation benchmark called Isaac Humanoid Manipulation Benchmark, where we design diverse bimanual manipulation tasks with demonstrations. We fine-tune and evaluate EgoVLA with Isaac Humanoid Manipulation Benchmark and show significant improvements over baselines and ablate the importance of human data. Videos can be found on our website: https://rchalyang.github.io/EgoVLA
>
---
#### [new 024] NemeSys: An Online Underwater Explorer with Goal-Driven Adaptive Autonomy
- **分类: cs.RO**

- **简介: 该论文属于水下机器人任务，解决AUV在无GPS环境下的动态任务适应问题。提出NemeSys系统，实现在线任务重构与低带宽通信交互。**

- **链接: [http://arxiv.org/pdf/2507.11889v1](http://arxiv.org/pdf/2507.11889v1)**

> **作者:** Adnan Abdullah; Alankrit Gupta; Vaishnav Ramesh; Shivali Patel; Md Jahidul Islam
>
> **备注:** 10 pages, V1
>
> **摘要:** Adaptive mission control and dynamic parameter reconfiguration are essential for autonomous underwater vehicles (AUVs) operating in GPS-denied, communication-limited marine environments. However, most current AUV platforms execute static, pre-programmed missions or rely on tethered connections and high-latency acoustic channels for mid-mission updates, significantly limiting their adaptability and responsiveness. In this paper, we introduce NemeSys, a novel AUV system designed to support real-time mission reconfiguration through compact optical and magnetoelectric (OME) signaling facilitated by floating buoys. We present the full system design, control architecture, and a semantic mission encoding framework that enables interactive exploration and task adaptation via low-bandwidth communication. The proposed system is validated through analytical modeling, controlled experimental evaluations, and open-water trials. Results confirm the feasibility of online mission adaptation and semantic task updates, highlighting NemeSys as an online AUV platform for goal-driven adaptive autonomy in dynamic and uncertain underwater environments.
>
---
#### [new 025] SGLoc: Semantic Localization System for Camera Pose Estimation from 3D Gaussian Splatting Representation
- **分类: cs.CV; cs.RO; I.4.8; I.2.9**

- **简介: 该论文属于视觉定位任务，解决无初始位姿的相机位姿估计问题。通过3D高斯泼溅表示和语义信息，直接回归相机位姿，提升定位精度。**

- **链接: [http://arxiv.org/pdf/2507.12027v1](http://arxiv.org/pdf/2507.12027v1)**

> **作者:** Beining Xu; Siting Zhu; Hesheng Wang
>
> **备注:** 8 pages, 2 figures, IROS 2025
>
> **摘要:** We propose SGLoc, a novel localization system that directly regresses camera poses from 3D Gaussian Splatting (3DGS) representation by leveraging semantic information. Our method utilizes the semantic relationship between 2D image and 3D scene representation to estimate the 6DoF pose without prior pose information. In this system, we introduce a multi-level pose regression strategy that progressively estimates and refines the pose of query image from the global 3DGS map, without requiring initial pose priors. Moreover, we introduce a semantic-based global retrieval algorithm that establishes correspondences between 2D (image) and 3D (3DGS map). By matching the extracted scene semantic descriptors of 2D query image and 3DGS semantic representation, we align the image with the local region of the global 3DGS map, thereby obtaining a coarse pose estimation. Subsequently, we refine the coarse pose by iteratively optimizing the difference between the query image and the rendered image from 3DGS. Our SGLoc demonstrates superior performance over baselines on 12scenes and 7scenes datasets, showing excellent capabilities in global localization without initial pose prior. Code will be available at https://github.com/IRMVLab/SGLoc.
>
---
#### [new 026] Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于交通场景轨迹预测任务，旨在提升自动驾驶系统的安全性。通过引入基于奖励的意图推理机制，结合逆强化学习与策略滚动，生成更准确的未来轨迹及概率。**

- **链接: [http://arxiv.org/pdf/2507.12083v1](http://arxiv.org/pdf/2507.12083v1)**

> **作者:** Muleilan Pei; Shaoshuai Shi; Xuesong Chen; Xu Liu; Shaojie Shen
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Motion forecasting for on-road traffic agents presents both a significant challenge and a critical necessity for ensuring safety in autonomous driving systems. In contrast to most existing data-driven approaches that directly predict future trajectories, we rethink this task from a planning perspective, advocating a "First Reasoning, Then Forecasting" strategy that explicitly incorporates behavior intentions as spatial guidance for trajectory prediction. To achieve this, we introduce an interpretable, reward-driven intention reasoner grounded in a novel query-centric Inverse Reinforcement Learning (IRL) scheme. Our method first encodes traffic agents and scene elements into a unified vectorized representation, then aggregates contextual features through a query-centric paradigm. This enables the derivation of a reward distribution, a compact yet informative representation of the target agent's behavior within the given scene context via IRL. Guided by this reward heuristic, we perform policy rollouts to reason about multiple plausible intentions, providing valuable priors for subsequent trajectory generation. Finally, we develop a hierarchical DETR-like decoder integrated with bidirectional selective state space models to produce accurate future trajectories along with their associated probabilities. Extensive experiments on the large-scale Argoverse and nuScenes motion forecasting datasets demonstrate that our approach significantly enhances trajectory prediction confidence, achieving highly competitive performance relative to state-of-the-art methods.
>
---
#### [new 027] MOSPA: Human Motion Generation Driven by Spatial Audio
- **分类: cs.GR; cs.CV; cs.RO**

- **简介: 该论文属于人机交互任务，解决虚拟人类对空间音频的动态响应问题。构建了SAM数据集，并提出MOSPA模型，实现基于空间音频的人体运动生成。**

- **链接: [http://arxiv.org/pdf/2507.11949v1](http://arxiv.org/pdf/2507.11949v1)**

> **作者:** Shuyang Xu; Zhiyang Dou; Mingyi Shi; Liang Pan; Leo Ho; Jingbo Wang; Yuan Liu; Cheng Lin; Yuexin Ma; Wenping Wang; Taku Komura
>
> **摘要:** Enabling virtual humans to dynamically and realistically respond to diverse auditory stimuli remains a key challenge in character animation, demanding the integration of perceptual modeling and motion synthesis. Despite its significance, this task remains largely unexplored. Most previous works have primarily focused on mapping modalities like speech, audio, and music to generate human motion. As of yet, these models typically overlook the impact of spatial features encoded in spatial audio signals on human motion. To bridge this gap and enable high-quality modeling of human movements in response to spatial audio, we introduce the first comprehensive Spatial Audio-Driven Human Motion (SAM) dataset, which contains diverse and high-quality spatial audio and motion data. For benchmarking, we develop a simple yet effective diffusion-based generative framework for human MOtion generation driven by SPatial Audio, termed MOSPA, which faithfully captures the relationship between body motion and spatial audio through an effective fusion mechanism. Once trained, MOSPA could generate diverse realistic human motions conditioned on varying spatial audio inputs. We perform a thorough investigation of the proposed dataset and conduct extensive experiments for benchmarking, where our method achieves state-of-the-art performance on this task. Our model and dataset will be open-sourced upon acceptance. Please refer to our supplementary video for more details.
>
---
#### [new 028] Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.RO**

- **简介: 该论文属于AI评估任务，解决MLLMs在评估过程中存在的同意偏差问题。通过提出SGV方法，提升模型的验证能力，显著提高任务完成率和准确性。**

- **链接: [http://arxiv.org/pdf/2507.11662v1](http://arxiv.org/pdf/2507.11662v1)**

> **作者:** Moises Andrade; Joonhyuk Cha; Brandon Ho; Vriksha Srihari; Karmesh Yadav; Zsolt Kira
>
> **备注:** Our code and data are publicly available at https://github.com/mshalimay/mllm-verifiers-abias-sgv
>
> **摘要:** Verifiers -- functions assigning rewards to agent behavior -- have been key for AI progress in domains like math and board games. However, extending these gains to domains without clear-cut success criteria (e.g.,computer use) remains a challenge: while humans can recognize suitable outcomes, translating this intuition into scalable rules is non-trivial. Multimodal Large Language Models(MLLMs) emerge as a promising solution, given their world knowledge, human-preference alignment, and reasoning skills. We evaluate MLLMs as verifiers of agent trajectories across web navigation, computer use, and robotic manipulation, and identify a critical limitation: agreement bias, a strong tendency for MLLMs to favor information in their context window, often generating chains of thought to rationalize flawed behavior. This bias is pervasive across models, resilient to test-time scaling, and can impact several methods using MLLMs as evaluators (e.g.,data filtering). Notably, it occurs despite MLLMs showing strong, human-aligned priors on desired behavior. To address this, we propose Self-Grounded Verification (SGV), a lightweight method that enables more effective use of MLLMs' knowledge and reasoning by harnessing their own sampling mechanisms via unconditional and conditional generation. SGV operates in two steps: first, the MLLM is elicited to retrieve broad priors about task completion, independent of the data under evaluation. Then, conditioned on self-generated priors, it reasons over and evaluates a candidate trajectory. Enhanced with SGV, MLLM verifiers show gains of up to 20 points in accuracy and failure detection rates, and can perform real-time supervision of heterogeneous agents, boosting task completion of a GUI specialist in OSWorld, a diffusion policy in robomimic, and a ReAct agent in VisualWebArena -- setting a new state of the art on the benchmark, surpassing the previous best by 48%.
>
---
#### [new 029] AutoVDC: Automated Vision Data Cleaning Using Vision-Language Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于数据清洗任务，旨在解决自动驾驶数据集中标注错误的问题。通过引入VLMs自动检测并清理错误标注，提升数据质量。**

- **链接: [http://arxiv.org/pdf/2507.12414v1](http://arxiv.org/pdf/2507.12414v1)**

> **作者:** Santosh Vasa; Aditi Ramadwar; Jnana Rama Krishna Darabattula; Md Zafar Anwar; Stanislaw Antol; Andrei Vatavu; Thomas Monninger; Sihao Ding
>
> **摘要:** Training of autonomous driving systems requires extensive datasets with precise annotations to attain robust performance. Human annotations suffer from imperfections, and multiple iterations are often needed to produce high-quality datasets. However, manually reviewing large datasets is laborious and expensive. In this paper, we introduce AutoVDC (Automated Vision Data Cleaning) framework and investigate the utilization of Vision-Language Models (VLMs) to automatically identify erroneous annotations in vision datasets, thereby enabling users to eliminate these errors and enhance data quality. We validate our approach using the KITTI and nuImages datasets, which contain object detection benchmarks for autonomous driving. To test the effectiveness of AutoVDC, we create dataset variants with intentionally injected erroneous annotations and observe the error detection rate of our approach. Additionally, we compare the detection rates using different VLMs and explore the impact of VLM fine-tuning on our pipeline. The results demonstrate our method's high performance in error detection and data cleaning experiments, indicating its potential to significantly improve the reliability and accuracy of large-scale production datasets in autonomous driving.
>
---
#### [new 030] Emergent Heterogeneous Swarm Control Through Hebbian Learning
- **分类: cs.NE; cs.AI; cs.RO**

- **简介: 该论文属于 swarm robotics 任务，旨在解决异构控制的难题。通过 Hebbian 学习实现群体自主异构性，提升群体行为能力与适应性。**

- **链接: [http://arxiv.org/pdf/2507.11566v1](http://arxiv.org/pdf/2507.11566v1)**

> **作者:** Fuda van Diggelen; Tugay Alperen Karagüzel; Andres Garcia Rincon; A. E. Eiben; Dario Floreano; Eliseo Ferrante
>
> **摘要:** In this paper, we introduce Hebbian learning as a novel method for swarm robotics, enabling the automatic emergence of heterogeneity. Hebbian learning presents a biologically inspired form of neural adaptation that solely relies on local information. By doing so, we resolve several major challenges for learning heterogeneous control: 1) Hebbian learning removes the complexity of attributing emergent phenomena to single agents through local learning rules, thus circumventing the micro-macro problem; 2) uniform Hebbian learning rules across all swarm members limit the number of parameters needed, mitigating the curse of dimensionality with scaling swarm sizes; and 3) evolving Hebbian learning rules based on swarm-level behaviour minimises the need for extensive prior knowledge typically required for optimising heterogeneous swarms. This work demonstrates that with Hebbian learning heterogeneity naturally emerges, resulting in swarm-level behavioural switching and in significantly improved swarm capabilities. It also demonstrates how the evolution of Hebbian learning rules can be a valid alternative to Multi Agent Reinforcement Learning in standard benchmarking tasks.
>
---
#### [new 031] VISTA: Monocular Segmentation-Based Mapping for Appearance and View-Invariant Global Localization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于全球定位任务，解决多视角和季节变化下的定位问题。提出VISTA框架，结合分割与子图匹配实现高效、鲁棒的定位。**

- **链接: [http://arxiv.org/pdf/2507.11653v1](http://arxiv.org/pdf/2507.11653v1)**

> **作者:** Hannah Shafferman; Annika Thomas; Jouko Kinnari; Michael Ricard; Jose Nino; Jonathan How
>
> **备注:** 9 pages, 6 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Global localization is critical for autonomous navigation, particularly in scenarios where an agent must localize within a map generated in a different session or by another agent, as agents often have no prior knowledge about the correlation between reference frames. However, this task remains challenging in unstructured environments due to appearance changes induced by viewpoint variation, seasonal changes, spatial aliasing, and occlusions -- known failure modes for traditional place recognition methods. To address these challenges, we propose VISTA (View-Invariant Segmentation-Based Tracking for Frame Alignment), a novel open-set, monocular global localization framework that combines: 1) a front-end, object-based, segmentation and tracking pipeline, followed by 2) a submap correspondence search, which exploits geometric consistencies between environment maps to align vehicle reference frames. VISTA enables consistent localization across diverse camera viewpoints and seasonal changes, without requiring any domain-specific training or finetuning. We evaluate VISTA on seasonal and oblique-angle aerial datasets, achieving up to a 69% improvement in recall over baseline methods. Furthermore, we maintain a compact object-based map that is only 0.6% the size of the most memory-conservative baseline, making our approach capable of real-time implementation on resource-constrained platforms.
>
---
#### [new 032] Online Training and Pruning of Deep Reinforcement Learning Networks
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，解决深度网络训练与剪枝的平衡问题。通过联合优化网络权重和随机变量参数，实现高效剪枝，提升RL性能。**

- **链接: [http://arxiv.org/pdf/2507.11975v1](http://arxiv.org/pdf/2507.11975v1)**

> **作者:** Valentin Frank Ingmar Guenter; Athanasios Sideris
>
> **备注:** 25 pages, 5 figures, 4 tables
>
> **摘要:** Scaling deep neural networks (NN) of reinforcement learning (RL) algorithms has been shown to enhance performance when feature extraction networks are used but the gained performance comes at the significant expense of increased computational and memory complexity. Neural network pruning methods have successfully addressed this challenge in supervised learning. However, their application to RL is underexplored. We propose an approach to integrate simultaneous training and pruning within advanced RL methods, in particular to RL algorithms enhanced by the Online Feature Extractor Network (OFENet). Our networks (XiNet) are trained to solve stochastic optimization problems over the RL networks' weights and the parameters of variational Bernoulli distributions for 0/1 Random Variables $\xi$ scaling each unit in the networks. The stochastic problem formulation induces regularization terms that promote convergence of the variational parameters to 0 when a unit contributes little to the performance. In this case, the corresponding structure is rendered permanently inactive and pruned from its network. We propose a cost-aware, sparsity-promoting regularization scheme, tailored to the DenseNet architecture of OFENets expressing the parameter complexity of involved networks in terms of the parameters of the RVs in these networks. Then, when matching this cost with the regularization terms, the many hyperparameters associated with them are automatically selected, effectively combining the RL objectives and network compression. We evaluate our method on continuous control benchmarks (MuJoCo) and the Soft Actor-Critic RL agent, demonstrating that OFENets can be pruned considerably with minimal loss in performance. Furthermore, our results confirm that pruning large networks during training produces more efficient and higher performing RL agents rather than training smaller networks from scratch.
>
---
#### [new 033] Risk in Stochastic and Robust Model Predictive Path-Following Control for Vehicular Motion Planning
- **分类: math.OC; cs.RO**

- **简介: 该论文属于自动驾驶路径跟踪控制任务，旨在解决如何定义和最小化风险的问题。通过引入随机和鲁棒模型预测控制器，比较其安全性和跟踪性能。**

- **链接: [http://arxiv.org/pdf/2304.12063v1](http://arxiv.org/pdf/2304.12063v1)**

> **作者:** Leon Tolksdorf; Arturo Tejada; Nathan van de Wouw; Christian Birkner
>
> **备注:** Accepted for the 2023 Intelligent Vehicles Symposium, 8 pages
>
> **摘要:** In automated driving, risk describes potential harm to passengers of an autonomous vehicle (AV) and other road users. Recent studies suggest that human-like driving behavior emerges from embedding risk in AV motion planning algorithms. Additionally, providing evidence that risk is minimized during the AV operation is essential to vehicle safety certification. However, there has yet to be a consensus on how to define and operationalize risk in motion planning or how to bound or minimize it during operation. In this paper, we define a stochastic risk measure and introduce it as a constraint into both robust and stochastic nonlinear model predictive path-following controllers (RMPC and SMPC respectively). We compare the vehicle's behavior arising from employing SMPC and RMPC with respect to safety and path-following performance. Further, the implementation of an automated driving example is provided, showcasing the effects of different risk tolerances and uncertainty growths in predictions of other road users for both cases. We find that the RMPC is significantly more conservative than the SMPC, while also displaying greater following errors towards references. Further, the RMPCs behavior cannot be considered as human-like. Moreover, unlike SMPC, the RMPC cannot account for different risk tolerances. The RMPC generates undesired driving behavior for even moderate uncertainties, which are handled better by the SMPC.
>
---
## 更新

#### [replaced 001] Incremental Joint Learning of Depth, Pose and Implicit Scene Representation on Monocular Camera in Large-scale Scenes
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2404.06050v3](http://arxiv.org/pdf/2404.06050v3)**

> **作者:** Tianchen Deng; Nailin Wang; Chongdi Wang; Shenghai Yuan; Jingchuan Wang; Hesheng Wang; Danwei Wang; Weidong Chen
>
> **摘要:** Dense scene reconstruction for photo-realistic view synthesis has various applications, such as VR/AR, autonomous vehicles. However, most existing methods have difficulties in large-scale scenes due to three core challenges: \textit{(a) inaccurate depth input.} Accurate depth input is impossible to get in real-world large-scale scenes. \textit{(b) inaccurate pose estimation.} Most existing approaches rely on accurate pre-estimated camera poses. \textit{(c) insufficient scene representation capability.} A single global radiance field lacks the capacity to effectively scale to large-scale scenes. To this end, we propose an incremental joint learning framework, which can achieve accurate depth, pose estimation, and large-scale scene reconstruction. A vision transformer-based network is adopted as the backbone to enhance performance in scale information estimation. For pose estimation, a feature-metric bundle adjustment (FBA) method is designed for accurate and robust camera tracking in large-scale scenes. In terms of implicit scene representation, we propose an incremental scene representation method to construct the entire large-scale scene as multiple local radiance fields to enhance the scalability of 3D scene representation. Extended experiments have been conducted to demonstrate the effectiveness and accuracy of our method in depth estimation, pose estimation, and large-scale scene reconstruction.
>
---
#### [replaced 002] MapEx: Indoor Structure Exploration with Probabilistic Information Gain from Global Map Predictions
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.15590v3](http://arxiv.org/pdf/2409.15590v3)**

> **作者:** Cherie Ho; Seungchan Kim; Brady Moon; Aditya Parandekar; Narek Harutyunyan; Chen Wang; Katia Sycara; Graeme Best; Sebastian Scherer
>
> **备注:** 7 pages
>
> **摘要:** Exploration is a critical challenge in robotics, centered on understanding unknown environments. In this work, we focus on robots exploring structured indoor environments which are often predictable and composed of repeating patterns. Most existing approaches, such as conventional frontier approaches, have difficulty leveraging the predictability and explore with simple heuristics such as `closest first'. Recent works use deep learning techniques to predict unknown regions of the map, using these predictions for information gain calculation. However, these approaches are often sensitive to the predicted map quality or do not reason over sensor coverage. To overcome these issues, our key insight is to jointly reason over what the robot can observe and its uncertainty to calculate probabilistic information gain. We introduce MapEx, a new exploration framework that uses predicted maps to form probabilistic sensor model for information gain estimation. MapEx generates multiple predicted maps based on observed information, and takes into consideration both the computed variances of predicted maps and estimated visible area to estimate the information gain of a given viewpoint. Experiments on the real-world KTH dataset showed on average 12.4% improvement than representative map-prediction based exploration and 25.4% improvement than nearest frontier approach. Website: mapex-explorer.github.io
>
---
#### [replaced 003] Robot Drummer: Learning Rhythmic Skills for Humanoid Drumming
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.11498v2](http://arxiv.org/pdf/2507.11498v2)**

> **作者:** Asad Ali Shahid; Francesco Braghin; Loris Roveda
>
> **摘要:** Humanoid robots have seen remarkable advances in dexterity, balance, and locomotion, yet their role in expressive domains such as music performance remains largely unexplored. Musical tasks, like drumming, present unique challenges, including split-second timing, rapid contacts, and multi-limb coordination over performances lasting minutes. In this paper, we introduce Robot Drummer, a humanoid capable of expressive, high-precision drumming across a diverse repertoire of songs. We formulate humanoid drumming as sequential fulfillment of timed contacts and transform drum scores into a Rhythmic Contact Chain. To handle the long-horizon nature of musical performance, we decompose each piece into fixed-length segments and train a single policy across all segments in parallel using reinforcement learning. Through extensive experiments on over thirty popular rock, metal, and jazz tracks, our results demonstrate that Robot Drummer consistently achieves high F1 scores. The learned behaviors exhibit emergent human-like drumming strategies, such as cross-arm strikes, and adaptive stick assignments, demonstrating the potential of reinforcement learning to bring humanoid robots into the domain of creative musical performance. Project page: robotdrummer.github.io
>
---
#### [replaced 004] On the Need for a Statistical Foundation in Scenario-Based Testing of Autonomous Vehicles
- **分类: cs.SE; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.02274v2](http://arxiv.org/pdf/2505.02274v2)**

> **作者:** Xingyu Zhao; Robab Aghazadeh-Chakherlou; Chih-Hong Cheng; Peter Popov; Lorenzo Strigini
>
> **备注:** Accepted by ITSC 2025
>
> **摘要:** Scenario-based testing has emerged as a common method for autonomous vehicles (AVs) safety assessment, offering a more efficient alternative to mile-based testing by focusing on high-risk scenarios. However, fundamental questions persist regarding its stopping rules, residual risk estimation, debug effectiveness, and the impact of simulation fidelity on safety claims. This paper argues that a rigorous statistical foundation is essential to address these challenges and enable rigorous safety assurance. By drawing parallels between AV testing and established software testing methods, we identify shared research gaps and reusable solutions. We propose proof-of-concept models to quantify the probability of failure per scenario (\textit{pfs}) and evaluate testing effectiveness under varying conditions. Our analysis reveals that neither scenario-based nor mile-based testing universally outperforms the other. Furthermore, we give an example of formal reasoning about alignment of synthetic and real-world testing outcomes, a first step towards supporting statistically defensible simulation-based safety claims.
>
---
#### [replaced 005] Reinforced Imitative Trajectory Planning for Urban Automated Driving
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.15607v2](http://arxiv.org/pdf/2410.15607v2)**

> **作者:** Di Zeng; Ling Zheng; Xiantong Yang; Yinong Li
>
> **备注:** 21 pages, 9 figures
>
> **摘要:** Reinforcement learning (RL) faces challenges in trajectory planning for urban automated driving due to the poor convergence of RL and the difficulty in designing reward functions. Consequently, few RL-based trajectory planning methods can achieve performance comparable to that of imitation learning-based methods. The convergence problem is alleviated by combining RL with supervised learning. However, most existing approaches only reason one step ahead and lack the capability to plan for multiple future steps. Besides, although inverse reinforcement learning holds promise for solving the reward function design issue, existing methods for automated driving impose a linear structure assumption on reward functions, making them difficult to apply to urban automated driving. In light of these challenges, this paper proposes a novel RL-based trajectory planning method that integrates RL with imitation learning to enable multi-step planning. Furthermore, a transformer-based Bayesian reward function is developed, providing effective reward signals for RL in urban scenarios. Moreover, a hybrid-driven trajectory planning framework is proposed to enhance safety and interpretability. The proposed methods were validated on the large-scale real-world urban automated driving nuPlan dataset. Evaluated using closed-loop metrics, the results demonstrated that the proposed method significantly outperformed the baseline employing the identical policy model structure and achieved competitive performance compared to the state-of-the-art method. The code is available at https://github.com/Zigned/nuplan_zigned.
>
---
#### [replaced 006] Reconfigurable legged metamachines that run on autonomous modular legs
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.00784v2](http://arxiv.org/pdf/2505.00784v2)**

> **作者:** Chen Yu; David Matthews; Jingxian Wang; Jing Gu; Douglas Blackiston; Michael Rubenstein; Sam Kriegman
>
> **摘要:** Legged machines are becoming increasingly agile and adaptive but they have so far lacked the morphological diversity of legged animals, which have been rearranged and reshaped to fill millions of niches. Unlike their biological counterparts, legged machines have largely converged over the past decade to canonical quadrupedal and bipedal architectures that cannot be easily reconfigured to meet new tasks or recover from injury. Here we introduce autonomous modular legs: agile yet minimal, single-degree-of-freedom jointed links that can learn complex dynamic behaviors and may be freely attached to form legged metamachines at the meter scale. This enables rapid repair, redesign, and recombination of highly-dynamic modular agents that move quickly and acrobatically (non-quasistatically) through unstructured environments. Because each module is itself a complete agent, legged metamachines are able to sustain deep structural damage that would completely disable other legged robots. We also show how to encode the vast space of possible body configurations into a compact latent design genome that can be efficiently explored, revealing a wide diversity of novel legged forms.
>
---
#### [replaced 007] LiDPM: Rethinking Point Diffusion for Lidar Scene Completion
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.17791v2](http://arxiv.org/pdf/2504.17791v2)**

> **作者:** Tetiana Martyniuk; Gilles Puy; Alexandre Boulch; Renaud Marlet; Raoul de Charette
>
> **备注:** Accepted to IEEE IV 2025 (Oral); v2 - updated quantitative results based on the metrics (Voxel IoU) calculation code corrections
>
> **摘要:** Training diffusion models that work directly on lidar points at the scale of outdoor scenes is challenging due to the difficulty of generating fine-grained details from white noise over a broad field of view. The latest works addressing scene completion with diffusion models tackle this problem by reformulating the original DDPM as a local diffusion process. It contrasts with the common practice of operating at the level of objects, where vanilla DDPMs are currently used. In this work, we close the gap between these two lines of work. We identify approximations in the local diffusion formulation, show that they are not required to operate at the scene level, and that a vanilla DDPM with a well-chosen starting point is enough for completion. Finally, we demonstrate that our method, LiDPM, leads to better results in scene completion on SemanticKITTI. The project page is https://astra-vision.github.io/LiDPM .
>
---
#### [replaced 008] System 0/1/2/3: Quad-process theory for multi-timescale embodied collective cognitive systems
- **分类: cs.AI; cs.RO; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2503.06138v3](http://arxiv.org/pdf/2503.06138v3)**

> **作者:** Tadahiro Taniguchi; Yasushi Hirai; Masahiro Suzuki; Shingo Murata; Takato Horii; Kazutoshi Tanaka
>
> **备注:** Under review
>
> **摘要:** This paper introduces the System 0/1/2/3 framework as an extension of dual-process theory, employing a quad-process model of cognition. Expanding upon System 1 (fast, intuitive thinking) and System 2 (slow, deliberative thinking), we incorporate System 0, which represents pre-cognitive embodied processes, and System 3, which encompasses collective intelligence and symbol emergence. We contextualize this model within Bergson's philosophy by adopting multi-scale time theory to unify the diverse temporal dynamics of cognition. System 0 emphasizes morphological computation and passive dynamics, illustrating how physical embodiment enables adaptive behavior without explicit neural processing. Systems 1 and 2 are explained from a constructive perspective, incorporating neurodynamical and AI viewpoints. In System 3, we introduce collective predictive coding to explain how societal-level adaptation and symbol emergence operate over extended timescales. This comprehensive framework ranges from rapid embodied reactions to slow-evolving collective intelligence, offering a unified perspective on cognition across multiple timescales, levels of abstraction, and forms of human intelligence. The System 0/1/2/3 model provides a novel theoretical foundation for understanding the interplay between adaptive and cognitive processes, thereby opening new avenues for research in cognitive science, AI, robotics, and collective intelligence.
>
---
#### [replaced 009] STEP Planner: Constructing cross-hierarchical subgoal tree as an embodied long-horizon task planner
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.21030v2](http://arxiv.org/pdf/2506.21030v2)**

> **作者:** Tianxing Zhou; Zhirui Wang; Haojia Ao; Guangyan Chen; Boyang Xing; Jingwen Cheng; Yi Yang; Yufeng Yue
>
> **摘要:** The ability to perform reliable long-horizon task planning is crucial for deploying robots in real-world environments. However, directly employing Large Language Models (LLMs) as action sequence generators often results in low success rates due to their limited reasoning ability for long-horizon embodied tasks. In the STEP framework, we construct a subgoal tree through a pair of closed-loop models: a subgoal decomposition model and a leaf node termination model. Within this framework, we develop a hierarchical tree structure that spans from coarse to fine resolutions. The subgoal decomposition model leverages a foundation LLM to break down complex goals into manageable subgoals, thereby spanning the subgoal tree. The leaf node termination model provides real-time feedback based on environmental states, determining when to terminate the tree spanning and ensuring each leaf node can be directly converted into a primitive action. Experiments conducted in both the VirtualHome WAH-NL benchmark and on real robots demonstrate that STEP achieves long-horizon embodied task completion with success rates up to 34% (WAH-NL) and 25% (real robot) outperforming SOTA methods.
>
---
#### [replaced 010] Robot Metabolism: Towards machines that can grow by consuming other machines
- **分类: cs.RO; cs.MA; cs.SY; eess.SY; 70-01, 68-02; I.6; H.4; H.m; I.m; B.m**

- **链接: [http://arxiv.org/pdf/2411.11192v2](http://arxiv.org/pdf/2411.11192v2)**

> **作者:** Philippe Martin Wyder; Riyaan Bakhda; Meiqi Zhao; Quinn A. Booth; Matthew E. Modi; Andrew Song; Simon Kang; Jiahao Wu; Priya Patel; Robert T. Kasumi; David Yi; Nihar Niraj Garg; Pranav Jhunjhunwala; Siddharth Bhutoria; Evan H. Tong; Yuhang Hu; Judah Goldfeder; Omer Mustel; Donghan Kim; Hod Lipson
>
> **备注:** Manuscript combined with Supplementary Materials File for arXiv submission
>
> **摘要:** Biological lifeforms can heal, grow, adapt, and reproduce -- abilities essential for sustained survival and development. In contrast, robots today are primarily monolithic machines with limited ability to self-repair, physically develop, or incorporate material from their environments. While robot minds rapidly evolve new behaviors through AI, their bodies remain closed systems, unable to systematically integrate material to grow or heal. We argue that open-ended physical adaptation is only possible when robots are designed using a small repertoire of simple modules. This allows machines to mechanically adapt by consuming parts from other machines or their surroundings and shed broken components. We demonstrate this principle on a truss modular robot platform. We show how robots can grow bigger, faster, and more capable by consuming materials from their environment and other robots. We suggest that machine metabolic processes like those demonstrated here will be an essential part of any sustained future robot ecology.
>
---
#### [replaced 011] Haptic-Informed ACT with a Soft Gripper and Recovery-Informed Training for Pseudo Oocyte Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.18212v3](http://arxiv.org/pdf/2506.18212v3)**

> **作者:** Pedro Miguel Uriguen Eljuri; Hironobu Shibata; Maeyama Katsuyoshi; Yuanyuan Jia; Tadahiro Taniguchi
>
> **备注:** Accepted at IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS2025) Project website https://tanichu-laboratory.github.io/pedro_haptic_act_iros2025/
>
> **摘要:** In this paper, we introduce Haptic-Informed ACT, an advanced robotic system for pseudo oocyte manipulation, integrating multimodal information and Action Chunking with Transformers (ACT). Traditional automation methods for oocyte transfer rely heavily on visual perception, often requiring human supervision due to biological variability and environmental disturbances. Haptic-Informed ACT enhances ACT by incorporating haptic feedback, enabling real-time grasp failure detection and adaptive correction. Additionally, we introduce a 3D-printed TPU soft gripper to facilitate delicate manipulations. Experimental results demonstrate that Haptic-Informed ACT improves the task success rate, robustness, and adaptability compared to conventional ACT, particularly in dynamic environments. These findings highlight the potential of multimodal learning in robotics for biomedical automation.
>
---
#### [replaced 012] Geometric Formulation of Unified Force-Impedance Control on SE(3) for Robotic Manipulators
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.17080v2](http://arxiv.org/pdf/2504.17080v2)**

> **作者:** Joohwan Seo; Nikhil Potu Surya Prakash; Soomi Lee; Arvind Kruthiventy; Megan Teng; Jongeun Choi; Roberto Horowitz
>
> **摘要:** In this paper, we present an impedance control framework on the SE(3) manifold, which enables force tracking while guaranteeing passivity. Building upon the unified force-impedance control (UFIC) and our previous work on geometric impedance control (GIC), we develop the geometric unified force impedance control (GUFIC) to account for the SE(3) manifold structure in the controller formulation using a differential geometric perspective. As in the case of the UFIC, the GUFIC utilizes energy tank augmentation for both force-tracking and impedance control to guarantee the manipulator's passivity relative to external forces. This ensures that the end effector maintains safe contact interaction with uncertain environments and tracks a desired interaction force. Moreover, we resolve a non-causal implementation problem in the UFIC formulation by introducing velocity and force fields. Due to its formulation on SE(3), the proposed GUFIC inherits the desirable SE(3) invariance and equivariance properties of the GIC, which helps increase sample efficiency in machine learning applications where a learning algorithm is incorporated into the control law. The proposed control law is validated in a simulation environment under scenarios requiring tracking an SE(3) trajectory, incorporating both position and orientation, while exerting a force on a surface. The codes are available at https://github.com/Joohwan-Seo/GUFIC_mujoco.
>
---
#### [replaced 013] Enhancing Trust in Autonomous Agents: An Architecture for Accountability and Explainability through Blockchain and Large Language Models
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.09567v4](http://arxiv.org/pdf/2403.09567v4)**

> **作者:** Laura Fernández-Becerra; Miguel Ángel González-Santamarta; Ángel Manuel Guerrero-Higueras; Francisco Javier Rodríguez-Lera; Vicente Matellán Olivera
>
> **摘要:** The deployment of autonomous agents in environments involving human interaction has increasingly raised security concerns. Consequently, understanding the circumstances behind an event becomes critical, requiring the development of capabilities to justify their behaviors to non-expert users. Such explanations are essential in enhancing trustworthiness and safety, acting as a preventive measure against failures, errors, and misunderstandings. Additionally, they contribute to improving communication, bridging the gap between the agent and the user, thereby improving the effectiveness of their interactions. This work presents an accountability and explainability architecture implemented for ROS-based mobile robots. The proposed solution consists of two main components. Firstly, a black box-like element to provide accountability, featuring anti-tampering properties achieved through blockchain technology. Secondly, a component in charge of generating natural language explanations by harnessing the capabilities of Large Language Models (LLMs) over the data contained within the previously mentioned black box. The study evaluates the performance of our solution in three different scenarios, each involving autonomous agent navigation functionalities. This evaluation includes a thorough examination of accountability and explainability metrics, demonstrating the effectiveness of our approach in using accountable data from robot actions to obtain coherent, accurate and understandable explanations, even when facing challenges inherent in the use of autonomous agents in real-world scenarios.
>
---
#### [replaced 014] OpenLKA: An Open Dataset of Lane Keeping Assist from Recent Car Models under Real-world Driving Conditions
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.09092v2](http://arxiv.org/pdf/2505.09092v2)**

> **作者:** Yuhang Wang; Abdulaziz Alhuraish; Shengming Yuan; Hao Zhou
>
> **摘要:** Lane Keeping Assist (LKA) is widely adopted in modern vehicles, yet its real-world performance remains underexplored due to proprietary systems and limited data access. This paper presents OpenLKA, the first open, large-scale dataset for LKA evaluation and improvement. It includes 400 hours of driving data from 62 production vehicle models, collected through extensive road testing in Tampa, Florida and global contributions from the Comma.ai driving community. The dataset spans a wide range of challenging scenarios, including complex road geometries, degraded lane markings, adverse weather, lighting conditions and surrounding traffic. The dataset is multimodal, comprising: i) full CAN bus streams, decoded using custom reverse-engineered DBC files to extract key LKA events (e.g., system disengagements, lane detection failures); ii) synchronized high-resolution dash-cam video; iii) real-time outputs from Openpilot, providing accurate estimates of road curvature and lane positioning; iv) enhanced scene annotations generated by Vision Language Models, describing lane visibility, pavement quality, weather, lighting, and traffic conditions. By integrating vehicle-internal signals with high-fidelity perception and rich semantic context, OpenLKA provides a comprehensive platform for benchmarking the real-world performance of production LKA systems, identifying safety-critical operational scenarios, and assessing the readiness of current road infrastructure for autonomous driving. The dataset is publicly available at: https://github.com/OpenLKA/OpenLKA.
>
---
#### [replaced 015] Multimodal Fusion and Vision-Language Models: A Survey for Robot Vision
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02477v2](http://arxiv.org/pdf/2504.02477v2)**

> **作者:** Xiaofeng Han; Shunpeng Chen; Zenghuang Fu; Zhe Feng; Lue Fan; Dong An; Changwei Wang; Li Guo; Weiliang Meng; Xiaopeng Zhang; Rongtao Xu; Shibiao Xu
>
> **备注:** 27 pages, 11 figures, survey paper submitted to Information Fusion
>
> **摘要:** Robot vision has greatly benefited from advancements in multimodal fusion techniques and vision-language models (VLMs). We systematically review the applications of multimodal fusion in key robotic vision tasks, including semantic scene understanding, simultaneous localization and mapping (SLAM), 3D object detection, navigation and localization, and robot manipulation. We compare VLMs based on large language models (LLMs) with traditional multimodal fusion methods, analyzing their advantages, limitations, and synergies. Additionally, we conduct an in-depth analysis of commonly used datasets, evaluating their applicability and challenges in real-world robotic scenarios. Furthermore, we identify critical research challenges such as cross-modal alignment, efficient fusion strategies, real-time deployment, and domain adaptation, and propose future research directions, including self-supervised learning for robust multimodal representations, transformer-based fusion architectures, and scalable multimodal frameworks. Through a comprehensive review, comparative analysis, and forward-looking discussion, we provide a valuable reference for advancing multimodal perception and interaction in robotic vision. A comprehensive list of studies in this survey is available at https://github.com/Xiaofeng-Han-Res/MF-RV.
>
---
#### [replaced 016] TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.11069v2](http://arxiv.org/pdf/2507.11069v2)**

> **作者:** Jeongyun Kim; Seunghoon Jeong; Giseop Kim; Myung-Hwan Jeon; Eunji Jun; Ayoung Kim
>
> **摘要:** Understanding the 3D geometry of transparent objects from RGB images is challenging due to their inherent physical properties, such as reflection and refraction. To address these difficulties, especially in scenarios with sparse views and dynamic environments, we introduce TRAN-D, a novel 2D Gaussian Splatting-based depth reconstruction method for transparent objects. Our key insight lies in separating transparent objects from the background, enabling focused optimization of Gaussians corresponding to the object. We mitigate artifacts with an object-aware loss that places Gaussians in obscured regions, ensuring coverage of invisible surfaces while reducing overfitting. Furthermore, we incorporate a physics-based simulation that refines the reconstruction in just a few seconds, effectively handling object removal and chain-reaction movement of remaining objects without the need for rescanning. TRAN-D is evaluated on both synthetic and real-world sequences, and it consistently demonstrated robust improvements over existing GS-based state-of-the-art methods. In comparison with baselines, TRAN-D reduces the mean absolute error by over 39% for the synthetic TRansPose sequences. Furthermore, despite being updated using only one image, TRAN-D reaches a {\delta} < 2.5 cm accuracy of 48.46%, over 1.5 times that of baselines, which uses six images. Code and more results are available at https://jeongyun0609.github.io/TRAN-D/.
>
---
#### [replaced 017] SPARK: A Modular Benchmark for Humanoid Robot Safety
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2502.03132v2](http://arxiv.org/pdf/2502.03132v2)**

> **作者:** Yifan Sun; Rui Chen; Kai S. Yun; Yikuan Fang; Sebin Jung; Feihan Li; Bowei Li; Weiye Zhao; Changliu Liu
>
> **备注:** Presented at IFAC Symposium on Robotics
>
> **摘要:** This paper introduces the Safe Protective and Assistive Robot Kit (SPARK), a comprehensive benchmark designed to ensure safety in humanoid autonomy and teleoperation. Humanoid robots pose significant safety risks due to their physical capabilities of interacting with complex environments. The physical structures of humanoid robots further add complexity to the design of general safety solutions. To facilitate safe deployment of complex robot systems, SPARK can be used as a toolbox that comes with state-of-the-art safe control algorithms in a modular and composable robot control framework. Users can easily configure safety criteria and sensitivity levels to optimize the balance between safety and performance. To accelerate humanoid safety research and development, SPARK provides simulation benchmarks that compare safety approaches in a variety of environments, tasks, and robot models. Furthermore, SPARK allows quick deployment of synthesized safe controllers on real robots. For hardware deployment, SPARK supports Apple Vision Pro (AVP) or a Motion Capture System as external sensors, while offering interfaces for seamless integration with alternative hardware setups at the same time. This paper demonstrates SPARK's capability with both simulation experiments and case studies with a Unitree G1 humanoid robot. Leveraging these advantages of SPARK, users and researchers can significantly improve the safety of their humanoid systems as well as accelerate relevant research. The open source code is available at: https://github.com/intelligent-control-lab/spark.
>
---
#### [replaced 018] InterLoc: LiDAR-based Intersection Localization using Road Segmentation with Automated Evaluation Method
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.00512v3](http://arxiv.org/pdf/2505.00512v3)**

> **作者:** Nguyen Hoang Khoi Tran; Julie Stephany Berrio; Mao Shan; Zhenxing Ming; Stewart Worrall
>
> **摘要:** Online localization of road intersections is beneficial for autonomous vehicle localization, mapping and motion planning. Intersections offer strong landmarks for correcting vehicle pose estimation, anchoring new sensor data in up-to-date maps, and guiding vehicle routing in road network graphs. Despite this importance, intersection localization has not been widely studied, with existing methods either ignoring the rich semantic information already computed onboard or relying on scarce, hand-labeled intersection datasets. To close this gap, we present a novel LiDAR-based method for online vehicle-centric intersection localization. We detect the intersection candidates in a bird's eye view (BEV) representation formed by concatenating a sequence of semantic road scans. We then refine these candidates by analyzing the intersecting road branches and adjusting the intersection center point in a least-squares formulation. For evaluation, we introduce an automated pipeline that pairs localized intersection points with OpenStreetMap (OSM) intersection nodes using precise GNSS/INS ground-truth poses. Experiments on the SemanticKITTI dataset show that our method outperforms the latest learning-based baseline in accuracy and reliability. Sensitivity tests demonstrate the method's robustness to challenging segmentation errors, highlighting its applicability in the real world.
>
---
#### [replaced 019] Active Probing with Multimodal Predictions for Motion Planning
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.09822v2](http://arxiv.org/pdf/2507.09822v2)**

> **作者:** Darshan Gadginmath; Farhad Nawaz; Minjun Sung; Faizan M Tariq; Sangjae Bae; David Isele; Fabio Pasqualetti; Jovin D'sa
>
> **备注:** To appear at IROS '25. 8 pages. 3 tables. 6 figures
>
> **摘要:** Navigation in dynamic environments requires autonomous systems to reason about uncertainties in the behavior of other agents. In this paper, we introduce a unified framework that combines trajectory planning with multimodal predictions and active probing to enhance decision-making under uncertainty. We develop a novel risk metric that seamlessly integrates multimodal prediction uncertainties through mixture models. When these uncertainties follow a Gaussian mixture distribution, we prove that our risk metric admits a closed-form solution, and is always finite, thus ensuring analytical tractability. To reduce prediction ambiguity, we incorporate an active probing mechanism that strategically selects actions to improve its estimates of behavioral parameters of other agents, while simultaneously handling multimodal uncertainties. We extensively evaluate our framework in autonomous navigation scenarios using the MetaDrive simulation environment. Results demonstrate that our active probing approach successfully navigates complex traffic scenarios with uncertain predictions. Additionally, our framework shows robust performance across diverse traffic agent behavior models, indicating its broad applicability to real-world autonomous navigation challenges. Code and videos are available at https://darshangm.github.io/papers/active-probing-multimodal-predictions/.
>
---
#### [replaced 020] KISS-Matcher: Fast and Robust Point Cloud Registration Revisited
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.15615v3](http://arxiv.org/pdf/2409.15615v3)**

> **作者:** Hyungtae Lim; Daebeom Kim; Gunhee Shin; Jingnan Shi; Ignacio Vizzo; Hyun Myung; Jaesik Park; Luca Carlone
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** While global point cloud registration systems have advanced significantly in all aspects, many studies have focused on specific components, such as feature extraction, graph-theoretic pruning, or pose solvers. In this paper, we take a holistic view on the registration problem and develop an open-source and versatile C++ library for point cloud registration, called KISS-Matcher. KISS-Matcher combines a novel feature detector, Faster-PFH, that improves over the classical fast point feature histogram (FPFH). Moreover, it adopts a $k$-core-based graph-theoretic pruning to reduce the time complexity of rejecting outlier correspondences. Finally, it combines these modules in a complete, user-friendly, and ready-to-use pipeline. As verified by extensive experiments, KISS-Matcher has superior scalability and broad applicability, achieving a substantial speed-up compared to state-of-the-art outlier-robust registration pipelines while preserving accuracy. Our code will be available at https://github.com/MIT-SPARK/KISS-Matcher.
>
---
#### [replaced 021] MTF-Grasp: A Multi-tier Federated Learning Approach for Robotic Grasping
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.10158v2](http://arxiv.org/pdf/2507.10158v2)**

> **作者:** Obaidullah Zaland; Erik Elmroth; Monowar Bhuyan
>
> **备注:** The work is accepted for presentation at IEEE SMC 2025
>
> **摘要:** Federated Learning (FL) is a promising machine learning paradigm that enables participating devices to train privacy-preserved and collaborative models. FL has proven its benefits for robotic manipulation tasks. However, grasping tasks lack exploration in such settings where robots train a global model without moving data and ensuring data privacy. The main challenge is that each robot learns from data that is nonindependent and identically distributed (non-IID) and of low quantity. This exhibits performance degradation, particularly in robotic grasping. Thus, in this work, we propose MTF-Grasp, a multi-tier FL approach for robotic grasping, acknowledging the unique challenges posed by the non-IID data distribution across robots, including quantitative skewness. MTF-Grasp harnesses data quality and quantity across robots to select a set of "top-level" robots with better data distribution and higher sample count. It then utilizes top-level robots to train initial seed models and distribute them to the remaining "low-level" robots, reducing the risk of model performance degradation in low-level robots. Our approach outperforms the conventional FL setup by up to 8% on the quantity-skewed Cornell and Jacquard grasping datasets.
>
---
#### [replaced 022] RACER: Rational Artificial Intelligence Car-following-model Enhanced by Reality
- **分类: cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2312.07003v2](http://arxiv.org/pdf/2312.07003v2)**

> **作者:** Tianyi Li; Alexander Halatsis; Raphael Stern
>
> **摘要:** This paper introduces RACER, the Rational Artificial Intelligence Car-following model Enhanced by Reality, a cutting-edge deep learning car-following model, that satisfies partial derivative constraints, designed to predict Adaptive Cruise Control (ACC) driving behavior while staying theoretically feasible. Unlike conventional models, RACER effectively integrates Rational Driving Constraints (RDCs), crucial tenets of actual driving, resulting in strikingly accurate and realistic predictions. Against established models like the Optimal Velocity Relative Velocity (OVRV), a car-following Neural Network (NN), and a car-following Physics-Informed Neural Network (PINN), RACER excels across key metrics, such as acceleration, velocity, and spacing. Notably, it displays a perfect adherence to the RDCs, registering zero violations, in stark contrast to other models. This study highlights the immense value of incorporating physical constraints within AI models, especially for augmenting safety measures in transportation. It also paves the way for future research to test these models against human driving data, with the potential to guide safer and more rational driving behavior. The versatility of the proposed model, including its potential to incorporate additional derivative constraints and broader architectural applications, enhances its appeal and broadens its impact within the scientific community.
>
---
