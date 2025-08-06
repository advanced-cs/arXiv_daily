# 机器人 cs.RO

- **最新发布 47 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] Vision-based Perception System for Automated Delivery Robot-Pedestrians Interactions
- **分类: cs.RO; cs.LG**

- **简介: 该论文旨在开发基于单目视觉的自动化配送机器人-行人交互系统，解决导航安全性、效率与社会适配性问题。通过集成多目标检测、姿态估计和深度感知，显著提升了轨迹预测精度（IDF1）和多物体跟踪能力（MOTA），并验证了在复杂人群中的检测精度超过85%，实现了对弱势行人的有效识别与行为引导。**

- **链接: [http://arxiv.org/pdf/2508.03541v1](http://arxiv.org/pdf/2508.03541v1)**

> **作者:** Ergi Tushe; Bilal Farooq
>
> **摘要:** The integration of Automated Delivery Robots (ADRs) into pedestrian-heavy urban spaces introduces unique challenges in terms of safe, efficient, and socially acceptable navigation. We develop the complete pipeline for a single vision sensor based multi-pedestrian detection and tracking, pose estimation, and monocular depth perception. Leveraging the real-world MOT17 dataset sequences, this study demonstrates how integrating human-pose estimation and depth cues enhances pedestrian trajectory prediction and identity maintenance, even under occlusions and dense crowds. Results show measurable improvements, including up to a 10% increase in identity preservation (IDF1), a 7% improvement in multiobject tracking accuracy (MOTA), and consistently high detection precision exceeding 85%, even in challenging scenarios. Notably, the system identifies vulnerable pedestrian groups supporting more socially aware and inclusive robot behaviour.
>
---
#### [new 002] Estimation of Aerodynamics Forces in Dynamic Morphing Wing Flight
- **分类: cs.RO**

- **简介: 该论文旨在开发动态仿生翼飞行器的力估计方法，解决准确预测气动力问题，通过物理观察者和神经网络模型实现实时力测量并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.02984v1](http://arxiv.org/pdf/2508.02984v1)**

> **作者:** Bibek Gupta; Mintae Kim; Albert Park; Eric Sihite; Koushil Sreenath; Alireza Ramezani
>
> **摘要:** Accurate estimation of aerodynamic forces is essential for advancing the control, modeling, and design of flapping-wing aerial robots with dynamic morphing capabilities. In this paper, we investigate two distinct methodologies for force estimation on Aerobat, a bio-inspired flapping-wing platform designed to emulate the inertial and aerodynamic behaviors observed in bat flight. Our goal is to quantify aerodynamic force contributions during tethered flight, a crucial step toward closed-loop flight control. The first method is a physics-based observer derived from Hamiltonian mechanics that leverages the concept of conjugate momentum to infer external aerodynamic forces acting on the robot. This observer builds on the system's reduced-order dynamic model and utilizes real-time sensor data to estimate forces without requiring training data. The second method employs a neural network-based regression model, specifically a multi-layer perceptron (MLP), to learn a mapping from joint kinematics, flapping frequency, and environmental parameters to aerodynamic force outputs. We evaluate both estimators using a 6-axis load cell in a high-frequency data acquisition setup that enables fine-grained force measurements during periodic wingbeats. The conjugate momentum observer and the regression model demonstrate strong agreement across three force components (Fx, Fy, Fz).
>
---
#### [new 003] A novel autonomous microplastics surveying robot for beach environments
- **分类: cs.RO**

- **简介: 该论文提出了一款基于自主导航的微塑料探测机器人，旨在通过高精度扫描和实时化学分析解决海滩环境中微塑料污染的检测与分布问题。研究利用激光-红外传感器实现对沙地中的微塑料粒子分类识别。**

- **链接: [http://arxiv.org/pdf/2508.02952v1](http://arxiv.org/pdf/2508.02952v1)**

> **作者:** Hassan Iqbal; Kobiny Rex; Joseph Shirley; Carlos Baiz; Christian Claudel
>
> **备注:** 12 pages, 11 figures
>
> **摘要:** Microplastics, defined as plastic particles smaller than 5 millimeters, have become a pervasive environmental contaminant that accumulates on beaches due to wind patterns and tidal forcing. Detecting microplastics and mapping their concentration in the wild remains one of the primary challenges in addressing this environmental issue. This paper introduces a novel robotic platform that automatically detects and chemically analyzes microplastics on beach surfaces. This mobile manipulator system scans areas for microplastics using a camera mounted on the robotic arm's end effector. The system effectively segments candidate microplastic particles on sand surfaces even in the presence of organic matter such as leaves and clams. Once a candidate microplastic particle is detected, the system steers a near-infrared (NIR) spectroscopic sensor onto the particle using both NIR and visual feedback to chemically analyze it in real-time. Through experiments in lab and beach environments, the system is shown to achieve an excellent positional precision in manipulation control and high microplastic classification accuracy.
>
---
#### [new 004] AeroSafe: Mobile Indoor Air Purification using Aerosol Residence Time Analysis and Robotic Cough Emulator Testbed
- **分类: cs.RO; cs.AI**

- **简介: 该论文旨在通过机器人咳嗽模拟器与数字孪生技术优化室内空气质量，解决传统过滤器无法检测呼吸道颗粒物的问题，构建双代理物理模拟器并利用LSTM训练模型实现实时空气质量预测与干预策略。**

- **链接: [http://arxiv.org/pdf/2508.02947v1](http://arxiv.org/pdf/2508.02947v1)**

> **作者:** M Tanjid Hasan Tonmoy; Rahath Malladi; Kaustubh Singh; Forsad Al Hossain; Rajesh Gupta; Andrés E. Tejada-Martínez; Tauhidur Rahman
>
> **备注:** Accepted at IEEE International Conference on Robotics and Automation (ICRA) 2025. Author Accepted Manuscript
>
> **摘要:** Indoor air quality plays an essential role in the safety and well-being of occupants, especially in the context of airborne diseases. This paper introduces AeroSafe, a novel approach aimed at enhancing the efficacy of indoor air purification systems through a robotic cough emulator testbed and a digital-twins-based aerosol residence time analysis. Current portable air filters often overlook the concentrations of respiratory aerosols generated by coughs, posing a risk, particularly in high-exposure environments like healthcare facilities and public spaces. To address this gap, we present a robotic dual-agent physical emulator comprising a maneuverable mannequin simulating cough events and a portable air purifier autonomously responding to aerosols. The generated data from this emulator trains a digital twins model, combining a physics-based compartment model with a machine learning approach, using Long Short-Term Memory (LSTM) networks and graph convolution layers. Experimental results demonstrate the model's ability to predict aerosol concentration dynamics with a mean residence time prediction error within 35 seconds. The proposed system's real-time intervention strategies outperform static air filter placement, showcasing its potential in mitigating airborne pathogen risks.
>
---
#### [new 005] CookBench: A Long-Horizon Embodied Planning Benchmark for Complex Cooking Scenarios
- **分类: cs.RO**

- **简介: 该论文提出CookBench作为长时规划基准，解决复杂烹饪场景中短期任务与粗粒度动作的问题，通过Unity高精度模拟构建两阶段任务（意图识别+交互执行），细化动作到空间层级并提供统一工具集，分析LLM/VLM在复杂长期任务中的不足。**

- **链接: [http://arxiv.org/pdf/2508.03232v1](http://arxiv.org/pdf/2508.03232v1)**

> **作者:** Muzhen Cai; Xiubo Chen; Yining An; Jiaxin Zhang; Xuesong Wang; Wang Xu; Weinan Zhang; Ting Liu
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Embodied Planning is dedicated to the goal of creating agents capable of executing long-horizon tasks in complex physical worlds. However, existing embodied planning benchmarks frequently feature short-horizon tasks and coarse-grained action primitives. To address this challenge, we introduce CookBench, a benchmark for long-horizon planning in complex cooking scenarios. By leveraging a high-fidelity simulation environment built upon the powerful Unity game engine, we define frontier AI challenges in a complex, realistic environment. The core task in CookBench is designed as a two-stage process. First, in Intention Recognition, an agent needs to accurately parse a user's complex intent. Second, in Embodied Interaction, the agent should execute the identified cooking goal through a long-horizon, fine-grained sequence of physical actions. Unlike existing embodied planning benchmarks, we refine the action granularity to a spatial level that considers crucial operational information while abstracting away low-level robotic control. Besides, We provide a comprehensive toolset that encapsulates the simulator. Its unified API supports both macro-level operations, such as placing orders and purchasing ingredients, and a rich set of fine-grained embodied actions for physical interaction, enabling researchers to focus on high-level planning and decision-making. Furthermore, we present an in-depth analysis of state-of-the-art, closed-source Large Language Model and Vision-Language Model, revealing their major shortcomings and challenges posed by complex, long-horizon tasks. The full benchmark will be open-sourced to facilitate future research.
>
---
#### [new 006] Point2Act: Efficient 3D Distillation of Multimodal LLMs for Zero-Shot Context-Aware Grasping
- **分类: cs.RO**

- **简介: 该论文提出Point2Act，解决零样本上下文感知抓取中的3D动作定位问题，通过MLLM提取3D动作点并结合多视图融合补偿几何模糊，生成高效响应（20秒内）。**

- **链接: [http://arxiv.org/pdf/2508.03099v1](http://arxiv.org/pdf/2508.03099v1)**

> **作者:** Sang Min Kim; Hyeongjun Heo; Junho Kim; Yonghyeon Lee; Young Min Kim
>
> **摘要:** We propose Point2Act, which directly retrieves the 3D action point relevant for a contextually described task, leveraging Multimodal Large Language Models (MLLMs). Foundation models opened the possibility for generalist robots that can perform a zero-shot task following natural language descriptions within an unseen environment. While the semantics obtained from large-scale image and language datasets provide contextual understanding in 2D images, the rich yet nuanced features deduce blurry 2D regions and struggle to find precise 3D locations for actions. Our proposed 3D relevancy fields bypass the high-dimensional features and instead efficiently imbue lightweight 2D point-level guidance tailored to the task-specific action. The multi-view aggregation effectively compensates for misalignments due to geometric ambiguities, such as occlusion, or semantic uncertainties inherent in the language descriptions. The output region is highly localized, reasoning fine-grained 3D spatial context that can directly transfer to an explicit position for physical action at the on-the-fly reconstruction of the scene. Our full-stack pipeline, which includes capturing, MLLM querying, 3D reconstruction, and grasp pose extraction, generates spatially grounded responses in under 20 seconds, facilitating practical manipulation tasks. Project page: https://sangminkim-99.github.io/point2act/
>
---
#### [new 007] GACL: Grounded Adaptive Curriculum Learning with Active Task and Performance Monitoring
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出了一种名为GACL的任务学习框架，旨在解决传统机器人课程学习方法依赖人工设计的局限性，通过自主任务生成、动态性能监控和目标域适配技术，在轮式导航和四足行走等复杂环境中的任务成功率分别提升了6.8%和6.1%。**

- **链接: [http://arxiv.org/pdf/2508.02988v1](http://arxiv.org/pdf/2508.02988v1)**

> **作者:** Linji Wang; Zifan Xu; Peter Stone; Xuesu Xiao
>
> **备注:** 7 pages, IROS 2025
>
> **摘要:** Curriculum learning has emerged as a promising approach for training complex robotics tasks, yet current applications predominantly rely on manually designed curricula, which demand significant engineering effort and can suffer from subjective and suboptimal human design choices. While automated curriculum learning has shown success in simple domains like grid worlds and games where task distributions can be easily specified, robotics tasks present unique challenges: they require handling complex task spaces while maintaining relevance to target domain distributions that are only partially known through limited samples. To this end, we propose Grounded Adaptive Curriculum Learning, a framework specifically designed for robotics curriculum learning with three key innovations: (1) a task representation that consistently handles complex robot task design, (2) an active performance tracking mechanism that allows adaptive curriculum generation appropriate for the robot's current capabilities, and (3) a grounding approach that maintains target domain relevance through alternating sampling between reference and synthetic tasks. We validate GACL on wheeled navigation in constrained environments and quadruped locomotion in challenging 3D confined spaces, achieving 6.8% and 6.1% higher success rates, respectively, than state-of-the-art methods in each domain.
>
---
#### [new 008] CogniPlan: Uncertainty-Guided Path Planning with Conditional Generative Layout Prediction
- **分类: cs.RO**

- **简介: 该论文提出了一种基于条件生成模型的不确定性引导路径规划框架（CogniPlan），解决移动机器人在未知环境中自主探索与导航的问题，通过融合生成式布局预测与图注意力机制，实现了高效性和精度的提升。**

- **链接: [http://arxiv.org/pdf/2508.03027v1](http://arxiv.org/pdf/2508.03027v1)**

> **作者:** Yizhuo Wang; Haodong He; Jingsong Liang; Yuhong Cao; Ritabrata Chakraborty; Guillaume Sartoretti
>
> **备注:** Accepted for presentation at CORL 2025
>
> **摘要:** Path planning in unknown environments is a crucial yet inherently challenging capability for mobile robots, which primarily encompasses two coupled tasks: autonomous exploration and point-goal navigation. In both cases, the robot must perceive the environment, update its belief, and accurately estimate potential information gain on-the-fly to guide planning. In this work, we propose CogniPlan, a novel path planning framework that leverages multiple plausible layouts predicted by a COnditional GeNerative Inpainting model, mirroring how humans rely on cognitive maps during navigation. These predictions, based on the partially observed map and a set of layout conditioning vectors, enable our planner to reason effectively under uncertainty. We demonstrate strong synergy between generative image-based layout prediction and graph-attention-based path planning, allowing CogniPlan to combine the scalability of graph representations with the fidelity and predictiveness of occupancy maps, yielding notable performance gains in both exploration and navigation. We extensively evaluate CogniPlan on two datasets (hundreds of maps and realistic floor plans), consistently outperforming state-of-the-art planners. We further deploy it in a high-fidelity simulator and on hardware, showcasing its high-quality path planning and real-world applicability.
>
---
#### [new 009] Co-designing Zoomorphic Robot Concepts for Animal Welfare Education
- **分类: cs.RO**

- **简介: 该论文旨在通过参与式设计研讨会开发符合动物福利教育需求的定制化机器人概念，解决儿童与动物互动的安全性及教育性问题。研究发现关键要素包括视觉信号和自然形态，并提出分支故事板和交互叙事等创新活动，同时探讨共识达成的挑战。**

- **链接: [http://arxiv.org/pdf/2508.02898v1](http://arxiv.org/pdf/2508.02898v1)**

> **作者:** Isobel Voysey; Lynne Baillie; Joanne Williams; Michael Herrmann
>
> **摘要:** Animal welfare education could greatly benefit from customized robots to help children learn about animals and their behavior, and thereby promote positive, safe child-animal interactions. To this end, we ran Participatory Design workshops with animal welfare educators and children to identify key requirements for zoomorphic robots from their perspectives. Our findings encompass a zoomorphic robot's appearance, behavior, and features, as well as concepts for a narrative surrounding the robot. Through comparing and contrasting the two groups, we find the importance of: negative reactions to undesirable behavior from children; using the facial features and tail to provide cues signaling an animal's internal state; and a natural, furry appearance and texture. We also contribute some novel activities for Participatory Design with children, including branching storyboards inspired by thematic apperception tests and interactive narratives, and reflect on some of the key design challenges of achieving consensus between the groups, despite much overlap in their design concepts.
>
---
#### [new 010] Opti-Acoustic Scene Reconstruction in Highly Turbid Underwater Environments
- **分类: cs.RO**

- **简介: 该论文旨在解决高浑浊度水域中单目视觉重建不足的问题，提出基于光学声学的实时重建方法，通过避免点特征识别并匹配声呐与图像数据实现场景重建。**

- **链接: [http://arxiv.org/pdf/2508.03408v1](http://arxiv.org/pdf/2508.03408v1)**

> **作者:** Ivana Collado-Gonzalez; John McConnell; Paul Szenher; Brendan Englot
>
> **摘要:** Scene reconstruction is an essential capability for underwater robots navigating in close proximity to structures. Monocular vision-based reconstruction methods are unreliable in turbid waters and lack depth scale information. Sonars are robust to turbid water and non-uniform lighting conditions, however, they have low resolution and elevation ambiguity. This work proposes a real-time opti-acoustic scene reconstruction method that is specially optimized to work in turbid water. Our strategy avoids having to identify point features in visual data and instead identifies regions of interest in the data. We then match relevant regions in the image to corresponding sonar data. A reconstruction is obtained by leveraging range data from the sonar and elevation data from the camera image. Experimental comparisons against other vision-based and sonar-based approaches at varying turbidity levels, and field tests conducted in marina environments, validate the effectiveness of the proposed approach. We have made our code open-source to facilitate reproducibility and encourage community engagement.
>
---
#### [new 011] Inland-LOAM: Voxel-Based Structural Semantic Mapping for Inland Waterways
- **分类: cs.RO**

- **简介: 该研究提出Inland-LOAM，一种基于三维点云的水道结构语义映射框架，解决现有地图缺乏实时数据和LiDAR SLAM在水道中的失效问题，通过改进特征提取和水平面约束，将3D点云转化为2D语义地图，实现实时导航参数计算及岸线自动提取，验证其在真实场景中的有效性，代码和数据已公开。**

- **链接: [http://arxiv.org/pdf/2508.03672v1](http://arxiv.org/pdf/2508.03672v1)**

> **作者:** Zhongbi Luo; Yunjia Wang; Jan Swevers; Peter Slaets; Herman Bruyninckx
>
> **摘要:** Accurate geospatial information is crucial for safe, autonomous Inland Waterway Transport (IWT), as existing charts (IENC) lack real-time detail and conventional LiDAR SLAM fails in waterway environments. These challenges lead to vertical drift and non-semantic maps, hindering autonomous navigation. This paper introduces Inland-LOAM, a LiDAR SLAM framework for waterways. It uses an improved feature extraction and a water surface planar constraint to mitigate vertical drift. A novel pipeline transforms 3D point clouds into structured 2D semantic maps using voxel-based geometric analysis, enabling real-time computation of navigational parameters like bridge clearances. An automated module extracts shorelines and exports them into a lightweight, IENC-compatible format. Evaluations on a real-world dataset show Inland-LOAM achieves superior localization accuracy over state-of-the-art methods. The generated semantic maps and shorelines align with real-world conditions, providing reliable data for enhanced situational awareness. The code and dataset will be publicly available
>
---
#### [new 012] Optimizing Bipedal Locomotion for The 100m Dash With Comparison to Human Running
- **分类: cs.RO; cs.AI; I.2.9**

- **简介: 该论文旨在探讨如何通过优化Cassie机器人步态提升其在100米冲刺中的效率，解决传统机械与人类运行机制的效率差异问题。研究基于仿生学方法，对步态进行多速度范围优化，并通过实验与控制器整合验证其可行性，成功实现对世界纪录的突破。**

- **链接: [http://arxiv.org/pdf/2508.03070v1](http://arxiv.org/pdf/2508.03070v1)**

> **作者:** Devin Crowley; Jeremy Dao; Helei Duan; Kevin Green; Jonathan Hurst; Alan Fern
>
> **备注:** 7 pages, 7 figures, published by IEEE at ICRA 2023, pp. 12205-12211, see https://ieeexplore.ieee.org/document/10160436
>
> **摘要:** In this paper, we explore the space of running gaits for the bipedal robot Cassie. Our first contribution is to present an approach for optimizing gait efficiency across a spectrum of speeds with the aim of enabling extremely high-speed running on hardware. This raises the question of how the resulting gaits compare to human running mechanics, which are known to be highly efficient in comparison to quadrupeds. Our second contribution is to conduct this comparison based on established human biomechanical studies. We find that despite morphological differences between Cassie and humans, key properties of the gaits are highly similar across a wide range of speeds. Finally, our third contribution is to integrate the optimized running gaits into a full controller that satisfies the rules of the real-world task of the 100m dash, including starting and stopping from a standing position. We demonstrate this controller on hardware to establish the Guinness World Record for Fastest 100m by a Bipedal Robot.
>
---
#### [new 013] Safety-Aware Imitation Learning via MPC-Guided Disturbance Injection
- **分类: cs.RO**

- **简介: 该论文提出一种基于模型预测控制（MPC）的对抗性干扰注入技术，旨在通过设计时间的方式增强模仿学习的安全性，解决传统方法因学习误差导致的安全风险问题。研究工作包括：1）构建安全约束模型并进行数据注入；2）利用MPC模拟器处理高维动态系统；3）集成安全考量以提升仿真实验性能。**

- **链接: [http://arxiv.org/pdf/2508.03129v1](http://arxiv.org/pdf/2508.03129v1)**

> **作者:** Le Qiu; Yusuf Umut Ciftci; Somil Bansal
>
> **摘要:** Imitation Learning has provided a promising approach to learning complex robot behaviors from expert demonstrations. However, learned policies can make errors that lead to safety violations, which limits their deployment in safety-critical applications. We propose MPC-SafeGIL, a design-time approach that enhances the safety of imitation learning by injecting adversarial disturbances during expert demonstrations. This exposes the expert to a broader range of safety-critical scenarios and allows the imitation policy to learn robust recovery behaviors. Our method uses sampling-based Model Predictive Control (MPC) to approximate worst-case disturbances, making it scalable to high-dimensional and black-box dynamical systems. In contrast to prior work that relies on analytical models or interactive experts, MPC-SafeGIL integrates safety considerations directly into data collection. We validate our approach through extensive simulations including quadruped locomotion and visuomotor navigation and real-world experiments on a quadrotor, demonstrating improvements in both safety and task performance. See our website here: https://leqiu2003.github.io/MPCSafeGIL/
>
---
#### [new 014] Language as Cost: Proactive Hazard Mapping using VLM for Robot Navigation
- **分类: cs.RO**

- **简介: 该论文提出了一种基于零样本的语言-成本映射框架，解决了机器人在动态环境中的危险预测与规避问题。通过整合VLM对视觉场景的解读与风险评估，构建了预判性导航成本模型，有效提升了机器人在复杂动态场景下的安全性和导航成功率。**

- **链接: [http://arxiv.org/pdf/2508.03138v1](http://arxiv.org/pdf/2508.03138v1)**

> **作者:** Mintaek Oh; Chan Kim; Seung-Woo Seo; Seong-Woo Kim
>
> **备注:** Accepted at IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025. 8 pages, 7 figures
>
> **摘要:** Robots operating in human-centric or hazardous environments must proactively anticipate and mitigate dangers beyond basic obstacle detection. Traditional navigation systems often depend on static maps, which struggle to account for dynamic risks, such as a person emerging from a suddenly opening door. As a result, these systems tend to be reactive rather than anticipatory when handling dynamic hazards. Recent advancements in pre-trained large language models and vision-language models (VLMs) create new opportunities for proactive hazard avoidance. In this work, we propose a zero-shot language-as-cost mapping framework that leverages VLMs to interpret visual scenes, assess potential dynamic risks, and assign risk-aware navigation costs preemptively, enabling robots to anticipate hazards before they materialize. By integrating this language-based cost map with a geometric obstacle map, the robot not only identifies existing obstacles but also anticipates and proactively plans around potential hazards arising from environmental dynamics. Experiments in simulated and diverse dynamic environments demonstrate that the proposed method significantly improves navigation success rates and reduces hazard encounters, compared to reactive baseline planners. Code and supplementary materials are available at https://github.com/Taekmino/LaC.
>
---
#### [new 015] CollaBot: Vision-Language Guided Simultaneous Collaborative Manipulation
- **分类: cs.RO**

- **简介: 该论文提出了一种Vision-Language驱动的协同机器人操作框架，旨在解决传统单体任务对大物体移动的局限性，同时扩展至多机器人协作场景。其核心工作包括：1）利用SEEM进行目标对象场景分割与点云提取；2）将任务分解为局部抓取与全局协作机制；3）设计两阶段规划模块以确保轨迹安全性。实验表明该框架在不同规模和任务条件下可实现52%的成功率。**

- **链接: [http://arxiv.org/pdf/2508.03526v1](http://arxiv.org/pdf/2508.03526v1)**

> **作者:** Kun Song; Shentao Ma; Gaoming Chen; Ninglong Jin; Guangbao Zhao; Mingyu Ding; Zhenhua Xiong; Jia Pan
>
> **备注:** 9 pages,5 figures
>
> **摘要:** A central research topic in robotics is how to use this system to interact with the physical world. Traditional manipulation tasks primarily focus on small objects. However, in factory or home environments, there is often a need for the movement of large objects, such as moving tables. These tasks typically require multi-robot systems to work collaboratively. Previous research lacks a framework that can scale to arbitrary sizes of robots and generalize to various kinds of tasks. In this work, we propose CollaBot, a generalist framework for simultaneous collaborative manipulation. First, we use SEEM for scene segmentation and point cloud extraction of the target object. Then, we propose a collaborative grasping framework, which decomposes the task into local grasp pose generation and global collaboration. Finally, we design a 2-stage planning module that can generate collision-free trajectories to achieve this task. Experiments show a success rate of 52% across different numbers of robots, objects, and tasks, indicating the effectiveness of the proposed framework.
>
---
#### [new 016] SkeNa: Learning to Navigate Unseen Environments Based on Abstract Hand-Drawn Maps
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出了一种基于抽象手绘地图的导航任务（SkeNa），旨在解决未知环境下的导航问题。通过构建SoR数据集和SkeNavigator框架，利用RMD和DAGP提升导航精度，实现了对高抽象场景的高效导航。**

- **链接: [http://arxiv.org/pdf/2508.03053v1](http://arxiv.org/pdf/2508.03053v1)**

> **作者:** Haojun Xu; Jiaqi Xiang; Wu Wei; Jinyu Chen; Linqing Zhong; Linjiang Huang; Hongyu Yang; Si Liu
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** A typical human strategy for giving navigation guidance is to sketch route maps based on the environmental layout. Inspired by this, we introduce Sketch map-based visual Navigation (SkeNa), an embodied navigation task in which an agent must reach a goal in an unseen environment using only a hand-drawn sketch map as guidance. To support research for SkeNa, we present a large-scale dataset named SoR, comprising 54k trajectory and sketch map pairs across 71 indoor scenes. In SoR, we introduce two navigation validation sets with varying levels of abstraction in hand-drawn sketches, categorized based on their preservation of spatial scales in the environment, to facilitate future research. To construct SoR, we develop an automated sketch-generation pipeline that efficiently converts floor plans into hand-drawn representations. To solve SkeNa, we propose SkeNavigator, a navigation framework that aligns visual observations with hand-drawn maps to estimate navigation targets. It employs a Ray-based Map Descriptor (RMD) to enhance sketch map valid feature representation using equidistant sampling points and boundary distances. To improve alignment with visual observations, a Dual-Map Aligned Goal Predictor (DAGP) leverages the correspondence between sketch map features and on-site constructed exploration map features to predict goal position and guide navigation. SkeNavigator outperforms prior floor plan navigation methods by a large margin, improving SPL on the high-abstract validation set by 105% relatively. Our code and dataset will be released.
>
---
#### [new 017] Why Evolve When You Can Adapt? Post-Evolution Adaptation of Genetic Memory for On-the-Fly Control
- **分类: cs.RO; cs.NE**

- **简介: 该论文探讨了通过混合遗传算法（GA）与Hebbian学习机制，在实时动态环境中实现机器人控制器的自我适应能力。解决了传统遗传算法无法应对环境变化的问题，提出了一种基于在线适应性函数的零样本自适应控制方案，验证了其在T-迷宫导航任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.03600v1](http://arxiv.org/pdf/2508.03600v1)**

> **作者:** Hamze Hammami; Eva Denisa Barbulescu; Talal Shaikh; Mouayad Aldada; Muhammad Saad Munawar
>
> **备注:** This work was accepted for presentation at the ALIFE 2025 Conference in Kyoto, and will be published by MIT Press as part of the ALIFE 2025 proceedings
>
> **摘要:** Imagine a robot controller with the ability to adapt like human synapses, dynamically rewiring itself to overcome unforeseen challenges in real time. This paper proposes a novel zero-shot adaptation mechanism for evolutionary robotics, merging a standard Genetic Algorithm (GA) controller with online Hebbian plasticity. Inspired by biological systems, the method separates learning and memory, with the genotype acting as memory and Hebbian updates handling learning. In our approach, the fitness function is leveraged as a live scaling factor for Hebbian learning, enabling the robot's neural controller to adjust synaptic weights on-the-fly without additional training. This adds a dynamic adaptive layer that activates only during runtime to handle unexpected environmental changes. After the task, the robot 'forgets' the temporary adjustments and reverts to the original weights, preserving core knowledge. We validate this hybrid GA-Hebbian controller on an e-puck robot in a T-maze navigation task with changing light conditions and obstacles.
>
---
#### [new 018] Optimal Trajectory Planning in a Vertically Undulating Snake Locomotion using Contact-implicit Optimization
- **分类: cs.RO**

- **简介: 该论文旨在解决蛇形机器人在垂直起伏运动中的轨迹规划问题，利用接触隐性优化技术结合Moreau步进法，通过简化刚体动力学模型替代传统接触建模方法，从而克服复杂接触与控制分配挑战。**

- **链接: [http://arxiv.org/pdf/2508.02953v1](http://arxiv.org/pdf/2508.02953v1)**

> **作者:** Adarsh Salagame; Eric Sihite; Alireza Ramezani
>
> **摘要:** Contact-rich problems, such as snake robot locomotion, offer unexplored yet rich opportunities for optimization-based trajectory and acyclic contact planning. So far, a substantial body of control research has focused on emulating snake locomotion and replicating its distinctive movement patterns using shape functions that either ignore the complexity of interactions or focus on complex interactions with matter (e.g., burrowing movements). However, models and control frameworks that lie in between these two paradigms and are based on simple, fundamental rigid body dynamics, which alleviate the challenging contact and control allocation problems in snake locomotion, remain absent. This work makes meaningful contributions, substantiated by simulations and experiments, in the following directions: 1) introducing a reduced-order model based on Moreau's stepping-forward approach from differential inclusion mathematics, 2) verifying model accuracy, 3) experimental validation.
>
---
#### [new 019] Tunable Leg Stiffness in a Monopedal Hopper for Energy-Efficient Vertical Hopping Across Varying Ground Profiles
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文旨在解决如何通过调节腿刚度优化垂直跳跃能量效率的问题，提出HASTA机器人在不同地面条件下自主调整腿刚度的工作方案，并通过实验与仿真验证其可行性，为未来智能机器人腿部控制系统提供了理论支持。**

- **链接: [http://arxiv.org/pdf/2508.02873v1](http://arxiv.org/pdf/2508.02873v1)**

> **作者:** Rongqian Chen; Jun Kwon; Kefan Wu; Wei-Hsi Chen
>
> **备注:** 2025 IEEE International Conference on Robotics & Automation (ICRA)
>
> **摘要:** We present the design and implementation of HASTA (Hopper with Adjustable Stiffness for Terrain Adaptation), a vertical hopping robot with real-time tunable leg stiffness, aimed at optimizing energy efficiency across various ground profiles (a pair of ground stiffness and damping conditions). By adjusting leg stiffness, we aim to maximize apex hopping height, a key metric for energy-efficient vertical hopping. We hypothesize that softer legs perform better on soft, damped ground by minimizing penetration and energy loss, while stiffer legs excel on hard, less damped ground by reducing limb deformation and energy dissipation. Through experimental tests and simulations, we find the best leg stiffness within our selection for each combination of ground stiffness and damping, enabling the robot to achieve maximum steady-state hopping height with a constant energy input. These results support our hypothesis that tunable stiffness improves energy-efficient locomotion in controlled experimental conditions. In addition, the simulation provides insights that could aid in the future development of controllers for selecting leg stiffness.
>
---
#### [new 020] Aerobatic maneuvers in insect-scale flapping-wing aerial robots via deep-learned robust tube model predictive control
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **简介: 该论文旨在解决昆虫级飞行器在高动态和环境不确定性下的飞行控制问题，通过设计基于深度学习的模型预测控制器和神经网络实现仿生飞行控制，展示了750毫克昆虫级飞行器的卓越敏捷性和鲁棒性，实现了447%和255%的性能提升。**

- **链接: [http://arxiv.org/pdf/2508.03043v1](http://arxiv.org/pdf/2508.03043v1)**

> **作者:** Yi-Hsuan Hsiao; Andrea Tagliabue; Owen Matteson; Suhan Kim; Tong Zhao; Jonathan P. How; YuFeng Chen
>
> **备注:** 27 pages, 26 supplementary pages, 6 main figures, 16 supplementary figures, 1 table
>
> **摘要:** Aerial insects exhibit highly agile maneuvers such as sharp braking, saccades, and body flips under disturbance. In contrast, insect-scale aerial robots are limited to tracking non-aggressive trajectories with small body acceleration. This performance gap is contributed by a combination of low robot inertia, fast dynamics, uncertainty in flapping-wing aerodynamics, and high susceptibility to environmental disturbance. Executing highly dynamic maneuvers requires the generation of aggressive flight trajectories that push against the hardware limit and a high-rate feedback controller that accounts for model and environmental uncertainty. Here, through designing a deep-learned robust tube model predictive controller, we showcase insect-like flight agility and robustness in a 750-millgram flapping-wing robot. Our model predictive controller can track aggressive flight trajectories under disturbance. To achieve a high feedback rate in a compute-constrained real-time system, we design imitation learning methods to train a two-layer, fully connected neural network, which resembles insect flight control architecture consisting of central nervous system and motor neurons. Our robot demonstrates insect-like saccade movements with lateral speed and acceleration of 197 centimeters per second and 11.7 meters per second square, representing 447$\%$ and 255$\%$ improvement over prior results. The robot can also perform saccade maneuvers under 160 centimeters per second wind disturbance and large command-to-force mapping errors. Furthermore, it performs 10 consecutive body flips in 11 seconds - the most challenging maneuver among sub-gram flyers. These results represent a milestone in achieving insect-scale flight agility and inspire future investigations on sensing and compute autonomy.
>
---
#### [new 021] Physics-informed Neural Time Fields for Prehensile Object Manipulation
- **分类: cs.RO**

- **简介: 该论文提出物理引导神经时间场（PINN）用于解决预握物体操控问题，通过高效学习解Eikonal方程并实时规划轨迹，有效解决了传统方法效率低、依赖示例的局限性，已在模拟与现实场景中验证其优越性。**

- **链接: [http://arxiv.org/pdf/2508.02976v1](http://arxiv.org/pdf/2508.02976v1)**

> **作者:** Hanwen Ren; Ruiqi Ni; Ahmed H. Qureshi
>
> **摘要:** Object manipulation skills are necessary for robots operating in various daily-life scenarios, ranging from warehouses to hospitals. They allow the robots to manipulate the given object to their desired arrangement in the cluttered environment. The existing approaches to solving object manipulations are either inefficient sampling based techniques, require expert demonstrations, or learn by trial and error, making them less ideal for practical scenarios. In this paper, we propose a novel, multimodal physics-informed neural network (PINN) for solving object manipulation tasks. Our approach efficiently learns to solve the Eikonal equation without expert data and finds object manipulation trajectories fast in complex, cluttered environments. Our method is multimodal as it also reactively replans the robot's grasps during manipulation to achieve the desired object poses. We demonstrate our approach in both simulation and real-world scenarios and compare it against state-of-the-art baseline methods. The results indicate that our approach is effective across various objects, has efficient training compared to previous learning-based methods, and demonstrates high performance in planning time, trajectory length, and success rates. Our demonstration videos can be found at https://youtu.be/FaQLkTV9knI.
>
---
#### [new 022] LiGen: GAN-Augmented Spectral Fingerprinting for Indoor Positioning
- **分类: cs.RO; I.2.9; C.3**

- **简介: 该论文提出一种结合谱指纹与GAN数据增强的室内定位系统LiGen，旨在解决现有Wi-Fi系统受环境干扰导致的定位误差问题。通过设计PointGAN（条件坐标生成）和FreeGAN（弱定位标签）等方法，利用多层感知机优化模型性能，实现比传统系统更高的精度（50%以上），并证明其在复杂环境中的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.03024v1](http://arxiv.org/pdf/2508.03024v1)**

> **作者:** Jie Lin; Hsun-Yu Lee; Ho-Ming Li; Fang-Jing Wu
>
> **备注:** 6 pages, 10 figures
>
> **摘要:** Accurate and robust indoor localization is critical for smart building applications, yet existing Wi-Fi-based systems are often vulnerable to environmental conditions. This work presents a novel indoor localization system, called LiGen, that leverages the spectral intensity patterns of ambient light as fingerprints, offering a more stable and infrastructure-free alternative to radio signals. To address the limited spectral data, we design a data augmentation framework based on generative adversarial networks (GANs), featuring two variants: PointGAN, which generates fingerprints conditioned on coordinates, and FreeGAN, which uses a weak localization model to label unconditioned samples. Our positioning model, leveraging a Multi-Layer Perceptron (MLP) architecture to train on synthesized data, achieves submeter-level accuracy, outperforming Wi-Fi-based baselines by over 50\%. LiGen also demonstrates strong robustness in cluttered environments. To the best of our knowledge, this is the first system to combine spectral fingerprints with GAN-based data augmentation for indoor localization.
>
---
#### [new 023] Force-Compliance MPC and Robot-User CBFs for Interactive Navigation and User-Robot Safety in Hexapod Guide Robots
- **分类: cs.RO**

- **简介: 该论文旨在解决互动导航与用户-机器人安全问题，提出Force-Compliance MPC和Robot-User CBFs技术，通过力反馈控制实现复杂环境下的安全导航，利用Eight-Way DBSCAN算法降低计算复杂度，结合MBE和Kalman滤波优化障碍识别与轨迹预测，实现了Hexapod机器人在资源受限条件下的高效自主导航与用户保护。**

- **链接: [http://arxiv.org/pdf/2508.03246v1](http://arxiv.org/pdf/2508.03246v1)**

> **作者:** Zehua Fan; Feng Gao; Zhijun Chen; Yunpeng Yin; Limin Yang; Qingxing Xi; En Yang; Xuefeng Luo
>
> **摘要:** Guiding the visually impaired in complex environments requires real-time two-way interaction and safety assurance. We propose a Force-Compliance Model Predictive Control (FC-MPC) and Robot-User Control Barrier Functions (CBFs) for force-compliant navigation and obstacle avoidance in Hexapod guide robots. FC-MPC enables two-way interaction by estimating user-applied forces and moments using the robot's dynamic model and the recursive least squares (RLS) method, and then adjusting the robot's movements accordingly, while Robot-User CBFs ensure the safety of both the user and the robot by handling static and dynamic obstacles, and employ weighted slack variables to overcome feasibility issues in complex dynamic environments. We also adopt an Eight-Way Connected DBSCAN method for obstacle clustering, reducing computational complexity from O(n2) to approximately O(n), enabling real-time local perception on resource-limited on-board robot computers. Obstacles are modeled using Minimum Bounding Ellipses (MBEs), and their trajectories are predicted through Kalman filtering. Implemented on the HexGuide robot, the system seamlessly integrates force compliance, autonomous navigation, and obstacle avoidance. Experimental results demonstrate the system's ability to adapt to user force commands while guaranteeing user and robot safety simultaneously during navigation in complex environments.
>
---
#### [new 024] Hand-Eye Autonomous Delivery: Learning Humanoid Navigation, Locomotion and Reaching
- **分类: cs.RO**

- **简介: 该论文提出了一种模块化框架HEAD，用于实现人机协同的自主配送。通过将高阶规划控制目标位置/姿态，低级控制器学习追踪人体关键点，结合人类运动数据与Aria眼镜采集的数据进行联合训练，解决了机器人自主导航与动作执行的问题，并验证了其在复杂场景中的应用能力。**

- **链接: [http://arxiv.org/pdf/2508.03068v1](http://arxiv.org/pdf/2508.03068v1)**

> **作者:** Sirui Chen; Yufei Ye; Zi-Ang Cao; Jennifer Lew; Pei Xu; C. Karen Liu
>
> **摘要:** We propose Hand-Eye Autonomous Delivery (HEAD), a framework that learns navigation, locomotion, and reaching skills for humanoids, directly from human motion and vision perception data. We take a modular approach where the high-level planner commands the target position and orientation of the hands and eyes of the humanoid, delivered by the low-level policy that controls the whole-body movements. Specifically, the low-level whole-body controller learns to track the three points (eyes, left hand, and right hand) from existing large-scale human motion capture data while high-level policy learns from human data collected by Aria glasses. Our modular approach decouples the ego-centric vision perception from physical actions, promoting efficient learning and scalability to novel scenes. We evaluate our method both in simulation and in the real-world, demonstrating humanoid's capabilities to navigate and reach in complex environments designed for humans.
>
---
#### [new 025] Thruster-Enhanced Locomotion: A Decoupled Model Predictive Control with Learned Contact Residuals
- **分类: cs.RO**

- **简介: 该论文旨在探索基于分层控制架构（Raibert型控制器+MPC）的联合推进与姿态控制优化，解决传统MPC因低扭矩带宽限制导致的系统性能瓶颈，通过学习接触残差动态弥补地面冲击影响，验证了该方法在推力恢复和猫步走姿上的稳定性。**

- **链接: [http://arxiv.org/pdf/2508.03003v1](http://arxiv.org/pdf/2508.03003v1)**

> **作者:** Chenghao Wang; Alireza Ramezani
>
> **摘要:** Husky Carbon, a robot developed by Northeastern University, serves as a research platform to explore unification of posture manipulation and thrust vectoring. Unlike conventional quadrupeds, its joint actuators and thrusters enable enhanced control authority, facilitating thruster-assisted narrow-path walking. While a unified Model Predictive Control (MPC) framework optimizing both ground reaction forces and thruster forces could theoretically address this control problem, its feasibility is limited by the low torque-control bandwidth of the system's lightweight actuators. To overcome this challenge, we propose a decoupled control architecture: a Raibert-type controller governs legged locomotion using position-based control, while an MPC regulates the thrusters augmented by learned Contact Residual Dynamics (CRD) to account for leg-ground impacts. This separation bypasses the torque-control rate bottleneck while retaining the thruster MPC to explicitly account for leg-ground impact dynamics through learned residuals. We validate this approach through both simulation and hardware experiments, showing that the decoupled control architecture with CRD performs more stable behavior in terms of push recovery and cat-like walking gait compared to the decoupled controller without CRD.
>
---
#### [new 026] Online Learning for Vibration Suppression in Physical Robot Interaction using Power Tools
- **分类: cs.RO**

- **简介: 该论文研究了基于功率工具的在线振动抑制技术，旨在解决物理人机交互中复杂环境下的振动问题，通过改进的BMFLC算法与动态衰减机制提升系统性能，验证了其在仿真与实际实验中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.03559v1](http://arxiv.org/pdf/2508.03559v1)**

> **作者:** Gokhan Solak; Arash Ajoudani
>
> **备注:** Submitted, under review
>
> **摘要:** Vibration suppression is an important capability for collaborative robots deployed in challenging environments such as construction sites. We study the active suppression of vibration caused by external sources such as power tools. We adopt the band-limited multiple Fourier linear combiner (BMFLC) algorithm to learn the vibration online and counter it by feedforward force control. We propose the damped BMFLC method, extending BMFLC with a novel adaptive step-size approach that improves the convergence time and noise resistance. Our logistic function-based damping mechanism reduces the effect of noise and enables larger learning rates. We evaluate our method on extensive simulation experiments with realistic time-varying multi-frequency vibration and real-world physical interaction experiments. The simulation experiments show that our method improves the suppression rate in comparison to the original BMFLC and its recursive least squares and Kalman filter-based extensions. Furthermore, our method is far more efficient than the latter two. We further validate the effectiveness of our method in real-world polishing experiments. A supplementary video is available at https://youtu.be/ms6m-6JyVAI.
>
---
#### [new 027] Context-aware Risk Assessment and Its Application in Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文旨在开发一种动态、模块化且低开销的上下文感知风险评估框架（CRI），解决传统方法缺乏解释性与实时性问题，通过空间分区、RSS原理和概率融合策略，在Bench2Drive数据集上验证了其显著提升自动驾驶安全性（如降低碰撞率19%）。**

- **链接: [http://arxiv.org/pdf/2508.02919v1](http://arxiv.org/pdf/2508.02919v1)**

> **作者:** Boyang Tian; Weisong Shi
>
> **备注:** ITSC 2025, 7 pages
>
> **摘要:** Ensuring safety in autonomous driving requires precise, real-time risk assessment and adaptive behavior. Prior work on risk estimation either outputs coarse, global scene-level metrics lacking interpretability, proposes indicators without concrete integration into autonomous systems, or focuses narrowly on specific driving scenarios. We introduce the Context-aware Risk Index (CRI), a light-weight modular framework that quantifies directional risks based on object kinematics and spatial relationships, dynamically adjusting control commands in real time. CRI employs direction-aware spatial partitioning within a dynamic safety envelope using Responsibility-Sensitive Safety (RSS) principles, a hybrid probabilistic-max fusion strategy for risk aggregation, and an adaptive control policy for real-time behavior modulation. We evaluate CRI on the Bench2Drive benchmark comprising 220 safety-critical scenarios using a state-of-the-art end-to-end model Transfuser++ on challenging routes. Our collision-rate metrics show a 19\% reduction (p = 0.003) in vehicle collisions per failed route, a 20\% reduction (p = 0.004) in collisions per kilometer, a 17\% increase (p = 0.016) in composed driving score, and a statistically significant reduction in penalty scores (p = 0.013) with very low overhead (3.6 ms per decision cycle). These results demonstrate that CRI substantially improves safety and robustness in complex, risk-intensive environments while maintaining modularity and low runtime overhead.
>
---
#### [new 028] Model-agnostic Meta-learning for Adaptive Gait Phase and Terrain Geometry Estimation with Wearable Soft Sensors
- **分类: cs.RO**

- **简介: 该论文提出了一种基于模型无关的元学习框架，用于通过服装传感器估计人体步态和地形几何，解决了适应性与泛化性问题，整合了MAML技术以增强模型初始化并提高效率与精度。**

- **链接: [http://arxiv.org/pdf/2508.02930v1](http://arxiv.org/pdf/2508.02930v1)**

> **作者:** Zenan Zhu; Wenxi Chen; Pei-Chun Kao; Janelle Clark; Lily Behnke; Rebecca Kramer-Bottiglio; Holly Yanco; Yan Gu
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** This letter presents a model-agnostic meta-learning (MAML) based framework for simultaneous and accurate estimation of human gait phase and terrain geometry using a small set of fabric-based wearable soft sensors, with efficient adaptation to unseen subjects and strong generalization across different subjects and terrains. Compared to rigid alternatives such as inertial measurement units, fabric-based soft sensors improve comfort but introduce nonlinearities due to hysteresis, placement error, and fabric deformation. Moreover, inter-subject and inter-terrain variability, coupled with limited calibration data in real-world deployments, further complicate accurate estimation. To address these challenges, the proposed framework integrates MAML into a deep learning architecture to learn a generalizable model initialization that captures subject- and terrain-invariant structure. This initialization enables efficient adaptation (i.e., adaptation with only a small amount of calibration data and a few fine-tuning steps) to new users, while maintaining strong generalization (i.e., high estimation accuracy across subjects and terrains). Experiments on nine participants walking at various speeds over five terrain conditions demonstrate that the proposed framework outperforms baseline approaches in estimating gait phase, locomotion mode, and incline angle, with superior accuracy, adaptation efficiency, and generalization.
>
---
#### [new 029] Robot builds a robot's brain: AI generated drone command and control station hosted in the sky
- **分类: cs.RO**

- **简介: 该论文探讨了AI驱动的无人机控制系统开发，解决了传统代码开发效率低的问题，通过AI生成代码实现自定义功能控制，验证了其在实时性和安全性方面的优势。**

- **链接: [http://arxiv.org/pdf/2508.02962v1](http://arxiv.org/pdf/2508.02962v1)**

> **作者:** Peter Burke
>
> **摘要:** Advances in artificial intelligence (AI) including large language models (LLMs) and hybrid reasoning models present an opportunity to reimagine how autonomous robots such as drones are designed, developed, and validated. Here, we demonstrate a fully AI-generated drone control system: with minimal human input, an artificial intelligence (AI) model authored all the code for a real-time, self-hosted drone command and control platform, which was deployed and demonstrated on a real drone in flight as well as a simulated virtual drone in the cloud. The system enables real-time mapping, flight telemetry, autonomous mission planning and execution, and safety protocolsall orchestrated through a web interface hosted directly on the drone itself. Not a single line of code was written by a human. We quantitatively benchmark system performance, code complexity, and development speed against prior, human-coded architectures, finding that AI-generated code can deliver functionally complete command-and-control stacks at orders-of-magnitude faster development cycles, though with identifiable current limitations related to specific model context window and reasoning depth. Our analysis uncovers the practical boundaries of AI-driven robot control code generation at current model scales, as well as emergent strengths and failure modes in AI-generated robotics code. This work sets a precedent for the autonomous creation of robot control systems and, more broadly, suggests a new paradigm for robotics engineeringone in which future robots may be largely co-designed, developed, and verified by artificial intelligence. In this initial work, a robot built a robot's brain.
>
---
#### [new 030] DiWA: Diffusion Policy Adaptation with World Models
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出DiWA框架，解决扩散策略RL训练效率低与实际应用需求矛盾，通过世界模型实现零样本优化，突破传统模型自由方法的交互瓶颈，首次在实现实时机器人技能改进中验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.03645v1](http://arxiv.org/pdf/2508.03645v1)**

> **作者:** Akshay L Chandra; Iman Nematollahi; Chenguang Huang; Tim Welschehold; Wolfram Burgard; Abhinav Valada
>
> **备注:** Accepted at the 2025 Conference on Robot Learning (CoRL)
>
> **摘要:** Fine-tuning diffusion policies with reinforcement learning (RL) presents significant challenges. The long denoising sequence for each action prediction impedes effective reward propagation. Moreover, standard RL methods require millions of real-world interactions, posing a major bottleneck for practical fine-tuning. Although prior work frames the denoising process in diffusion policies as a Markov Decision Process to enable RL-based updates, its strong dependence on environment interaction remains highly inefficient. To bridge this gap, we introduce DiWA, a novel framework that leverages a world model for fine-tuning diffusion-based robotic skills entirely offline with reinforcement learning. Unlike model-free approaches that require millions of environment interactions to fine-tune a repertoire of robot skills, DiWA achieves effective adaptation using a world model trained once on a few hundred thousand offline play interactions. This results in dramatically improved sample efficiency, making the approach significantly more practical and safer for real-world robot learning. On the challenging CALVIN benchmark, DiWA improves performance across eight tasks using only offline adaptation, while requiring orders of magnitude fewer physical interactions than model-free baselines. To our knowledge, this is the first demonstration of fine-tuning diffusion policies for real-world robotic skills using an offline world model. We make the code publicly available at https://diwa.cs.uni-freiburg.de.
>
---
#### [new 031] Residual Neural Terminal Constraint for MPC-based Collision Avoidance in Dynamic Environments
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **简介: 该论文提出了一种基于残差网络的混合MPC局部规划器，用于动态环境下的碰撞规避任务，通过HJ值函数建模安全区域并优化实时安全性，实现了30%以上的成功率提升。**

- **链接: [http://arxiv.org/pdf/2508.03428v1](http://arxiv.org/pdf/2508.03428v1)**

> **作者:** Bojan Derajić; Mohamed-Khalil Bouzidi; Sebastian Bernhard; Wolfgang Hönig
>
> **摘要:** In this paper, we propose a hybrid MPC local planner that uses a learning-based approximation of a time-varying safe set, derived from local observations and applied as the MPC terminal constraint. This set can be represented as a zero-superlevel set of the value function computed via Hamilton-Jacobi (HJ) reachability analysis, which is infeasible in real-time. We exploit the property that the HJ value function can be expressed as a difference of the corresponding signed distance function (SDF) and a non-negative residual function. The residual component is modeled as a neural network with non-negative output and subtracted from the computed SDF, resulting in a real-time value function estimate that is at least as safe as the SDF by design. Additionally, we parametrize the neural residual by a hypernetwork to improve real-time performance and generalization properties. The proposed method is compared with three state-of-the-art methods in simulations and hardware experiments, achieving up to 30\% higher success rates compared to the best baseline while requiring a similar computational effort and producing high-quality (low travel-time) solutions.
>
---
#### [new 032] Theatre in the Loop: A Rehearsal-Based, Collaborative Workflow for Expressive Robotic Behaviours
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出了一种基于导演指导的戏剧化协作流程，旨在通过叙事目标引导机器人生成情感化肢体动作，解决机械限制并推动跨学科协作，为表达性机器人行为提供了实践模型。**

- **链接: [http://arxiv.org/pdf/2508.03514v1](http://arxiv.org/pdf/2508.03514v1)**

> **作者:** Pavlos Panagiotidis; Victor Zhi Heung Ngo; Sean Myatt; Roma Patel; Rachel Ramchurn; Alan Chamberlain; Ayse Kucukyilmaz
>
> **备注:** The paper is accepted for presentation to International Conference on Social Robotics + AI (https://icsr2025.eu/)
>
> **摘要:** In this paper, we propose theatre-in-the-loop, a framework for developing expressive robot behaviours tailored to artistic performance through a director-guided puppeteering workflow. Leveraging theatrical methods, we use narrative objectives to direct a puppeteer in generating improvised robotic gestures that convey specific emotions. These improvisations are captured and curated to build a dataset of reusable movement templates for standalone playback in future autonomous performances. Initial trials demonstrate the feasibility of this approach, illustrating how the workflow enables precise sculpting of robotic gestures into coherent emotional arcs while revealing challenges posed by the robot's mechanical constraints. We argue that this practice-led framework provides a model for interdisciplinary teams creating socially expressive robot behaviours, contributing to (1) theatre as an interactive training ground for human-robot interaction and (2) co-creation methodologies between humans and machines.
>
---
#### [new 033] Learning User Interaction Forces using Vision for a Soft Finger Exosuit
- **分类: cs.RO**

- **简介: 该论文旨在解决软手指外骨骼的接触力估计问题，通过图像学习方法开发了一种非侵入式解决方案，解决了物理建模与嵌入传感的复杂性，实现了对动态交互的实时力估计。**

- **链接: [http://arxiv.org/pdf/2508.02870v1](http://arxiv.org/pdf/2508.02870v1)**

> **作者:** Mohamed Irfan Refai; Abdulaziz Y. Alkayas; Anup Teejo Mathew; Federico Renda; Thomas George Thuruthel
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Wearable assistive devices are increasingly becoming softer. Modelling their interface with human tissue is necessary to capture transmission of dynamic assistance. However, their nonlinear and compliant nature makes both physical modeling and embedded sensing challenging. In this paper, we develop a image-based, learning-based framework to estimate distributed contact forces for a finger-exosuit system. We used the SoRoSim toolbox to generate a diverse dataset of exosuit geometries and actuation scenarios for training. The method accurately estimated interaction forces across multiple contact locations from low-resolution grayscale images, was able to generalize to unseen shapes and actuation levels, and remained robust under visual noise and contrast variations. We integrated the model into a feedback controller, and found that the vision-based estimator functions as a surrogate force sensor for closed-loop control. This approach could be used as a non-intrusive alternative for real-time force estimation for exosuits.
>
---
#### [new 034] UniFucGrasp: Human-Hand-Inspired Unified Functional Grasp Annotation Strategy and Dataset for Diverse Dexterous Hands
- **分类: cs.RO; cs.CV; eess.IV**

- **简介: 该论文提出一种基于生物模仿的统一功能抓握标注策略（UniFucGrasp），解决传统 Dexterous grasp 数据集仅侧重稳定性而忽视功能性抓握的问题。通过将自然人类动作映射到多样手结构并结合几何闭合力技术，构建了低成本高效的功能抓握数据集，并验证其提升抓握精度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.03339v1](http://arxiv.org/pdf/2508.03339v1)**

> **作者:** Haoran Lin; Wenrui Chen; Xianchi Chen; Fan Yang; Qiang Diao; Wenxin Xie; Sijie Wu; Kailun Yang; Maojun Li; Yaonan Wang
>
> **备注:** The project page is at https://haochen611.github.io/UFG
>
> **摘要:** Dexterous grasp datasets are vital for embodied intelligence, but mostly emphasize grasp stability, ignoring functional grasps needed for tasks like opening bottle caps or holding cup handles. Most rely on bulky, costly, and hard-to-control high-DOF Shadow Hands. Inspired by the human hand's underactuated mechanism, we establish UniFucGrasp, a universal functional grasp annotation strategy and dataset for multiple dexterous hand types. Based on biomimicry, it maps natural human motions to diverse hand structures and uses geometry-based force closure to ensure functional, stable, human-like grasps. This method supports low-cost, efficient collection of diverse, high-quality functional grasps. Finally, we establish the first multi-hand functional grasp dataset and provide a synthesis model to validate its effectiveness. Experiments on the UFG dataset, IsaacSim, and complex robotic tasks show that our method improves functional manipulation accuracy and grasp stability, enables efficient generalization across diverse robotic hands, and overcomes annotation cost and generalization challenges in dexterous grasping. The project page is at https://haochen611.github.io/UFG.
>
---
#### [new 035] Multimodal Human-Intent Modeling for Contextual Robot-to-Human Handovers of Arbitrary Objects
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决传统基于预设目标的机器人手部操作中缺乏上下文感知的人类意图理解问题。研究通过整合多模态指令（口头/非口头）和上下文信息，提出一种统一框架，实现对任意物体的动态手部动作生成与交互优化。**

- **链接: [http://arxiv.org/pdf/2508.02982v1](http://arxiv.org/pdf/2508.02982v1)**

> **作者:** Lucas Chen; Guna Avula; Hanwen Ren; Zixing Wang; Ahmed H. Qureshi
>
> **摘要:** Human-robot object handover is a crucial element for assistive robots that aim to help people in their daily lives, including elderly care, hospitals, and factory floors. The existing approaches to solving these tasks rely on pre-selected target objects and do not contextualize human implicit and explicit preferences for handover, limiting natural and smooth interaction between humans and robots. These preferences can be related to the target object selection from the cluttered environment and to the way the robot should grasp the selected object to facilitate desirable human grasping during handovers. Therefore, this paper presents a unified approach that selects target distant objects using human verbal and non-verbal commands and performs the handover operation by contextualizing human implicit and explicit preferences to generate robot grasps and compliant handover motion sequences. We evaluate our integrated framework and its components through real-world experiments and user studies with arbitrary daily-life objects. The results of these evaluations demonstrate the effectiveness of our proposed pipeline in handling object handover tasks by understanding human preferences. Our demonstration videos can be found at https://youtu.be/6z27B2INl-s.
>
---
#### [new 036] Enhancing Joint Human-AI Inference in Robot Missions: A Confidence-Based Approach
- **分类: cs.HC; cs.RO**

- **简介: 该论文旨在通过增强人类-AI联合推理来提升机器人任务性能，解决传统方法（如依赖AI推荐）导致的互补性不足问题。研究通过用户研究验证了AI信心对推理精度的影响，并提出最大信心的启发式方法，表明良好AI决策支持系统能促进团队协作。**

- **链接: [http://arxiv.org/pdf/2508.03293v1](http://arxiv.org/pdf/2508.03293v1)**

> **作者:** Duc-An Nguyen; Clara Colombatto; Steve Fleming; Ingmar Posner; Nick Hawes; Raunak Bhattacharyya
>
> **摘要:** Joint human-AI inference holds immense potential to improve outcomes in human-supervised robot missions. Current day missions are generally in the AI-assisted setting, where the human operator makes the final inference based on the AI recommendation. However, due to failures in human judgement on when to accept or reject the AI recommendation, complementarity is rarely achieved. We investigate joint human-AI inference where the inference made with higher confidence is selected. Through a user study with N=100 participants on a representative simulated robot teleoperation task, specifically studying the inference of robots' control delays we show that: a) Joint inference accuracy is higher and its extent is regulated by the confidence calibration of the AI agent, and b) Humans change their inferences based on AI recommendations and the extent and direction of this change is also regulated by the confidence calibration of the AI agent. Interestingly, our results show that pairing poorly-calibrated AI-DSS with humans hurts performance instead of helping the team, reiterating the need for AI-based decision support systems with good metacognitive sensitivity. To the best of our knowledge, our study presents the first application of a maximum-confidence-based heuristic for joint human-AI inference within a simulated robot teleoperation task.
>
---
#### [new 037] La La LiDAR: Large-Scale Layout Generation from LiDAR Data
- **分类: cs.CV; cs.RO**

- **简介: 该论文旨在解决传统扩散模型在LiDAR场景生成中的前景控制缺失问题，提出La La LiDAR框架通过语义增强场景图扩散和关系感知上下文条件进行结构化生成，并结合前景控制注入实现定制化对象放置与空间/语义一致性，同时引入Waymo-SG和nuScenes-SG作为数据集及新评估指标。**

- **链接: [http://arxiv.org/pdf/2508.03691v1](http://arxiv.org/pdf/2508.03691v1)**

> **作者:** Youquan Liu; Lingdong Kong; Weidong Yang; Xin Li; Ao Liang; Runnan Chen; Ben Fei; Tongliang Liu
>
> **备注:** Preprint; 10 pages, 6 figures, 7 tables
>
> **摘要:** Controllable generation of realistic LiDAR scenes is crucial for applications such as autonomous driving and robotics. While recent diffusion-based models achieve high-fidelity LiDAR generation, they lack explicit control over foreground objects and spatial relationships, limiting their usefulness for scenario simulation and safety validation. To address these limitations, we propose Large-scale Layout-guided LiDAR generation model ("La La LiDAR"), a novel layout-guided generative framework that introduces semantic-enhanced scene graph diffusion with relation-aware contextual conditioning for structured LiDAR layout generation, followed by foreground-aware control injection for complete scene generation. This enables customizable control over object placement while ensuring spatial and semantic consistency. To support our structured LiDAR generation, we introduce Waymo-SG and nuScenes-SG, two large-scale LiDAR scene graph datasets, along with new evaluation metrics for layout synthesis. Extensive experiments demonstrate that La La LiDAR achieves state-of-the-art performance in both LiDAR generation and downstream perception tasks, establishing a new benchmark for controllable 3D scene generation.
>
---
#### [new 038] Generating Light-based Fingerprints for Indoor Localization
- **分类: eess.SP; cs.RO; I.2.9; C.3**

- **简介: 该论文旨在解决室内定位中的信号干扰与覆盖问题，提出通过可见光通信（VLC）生成指纹并优化训练数据的方法，构建了两阶段框架提升定位精度20%。**

- **链接: [http://arxiv.org/pdf/2508.03011v1](http://arxiv.org/pdf/2508.03011v1)**

> **作者:** Hsun-Yu Lee; Jie Lin; Fang-Jing Wu
>
> **备注:** 5 pages, 12 figures; presented at the 2024 MC & WASN Conference (Best Paper Candidate)
>
> **摘要:** Accurate indoor localization underpins applications ranging from wayfinding and emergency response to asset tracking and smart-building services. Radio-frequency solutions (e.g. Wi-Fi, RFID, UWB) are widely adopted but remain vulnerable to multipath fading, interference, and uncontrollable coverage variation. We explore an orthogonal modality -- visible light communication (VLC) -- and demonstrate that the spectral signatures captured by a low-cost AS7341 sensor can serve as robust location fingerprints. We introduce a two-stage framework that (i) trains a multi-layer perceptron (MLP) on real spectral measurements and (ii) enlarges the training corpus with synthetic samples produced by TabGAN. The augmented dataset reduces the mean localization error from 62.9cm to 49.3cm -- a 20% improvement -- while requiring only 5% additional data-collection effort. Experimental results obtained on 42 reference points in a U-shaped laboratory confirm that GAN-based augmentation mitigates data-scarcity issues and enhances generalization.
>
---
#### [new 039] Beyond Policy Optimization: A Data Curation Flywheel for Sparse-Reward Long-Horizon Planning
- **分类: cs.AI; cs.RO**

- **简介: 该论文探讨了多轮协作规划中的稀疏奖励问题，提出BPO框架解决信用分配与计算开销两大挑战，通过数据飞轮机制实现高效长时推理模型开发。**

- **链接: [http://arxiv.org/pdf/2508.03018v1](http://arxiv.org/pdf/2508.03018v1)**

> **作者:** Yutong Wang; Pengliang Ji; Kaixin Li; Baolong Bi; Tao Feng; Guillaume Sartoretti
>
> **摘要:** Large Language Reasoning Models have demonstrated remarkable success on static tasks, yet their application to multi-round agentic planning in interactive environments faces two fundamental challenges. First, the intractable credit assignment problem renders conventional reinforcement learning ineffective in sparse-reward settings. Second, the computational overhead of verbose, step-by-step reasoning histories is prohibitive. To address these challenges, we propose BPO, a three-stage framework (bootstrapping, extrapolation, and refinement) that establishes a self-improving data flywheel to develop robust reasoning models for long-horizon, sparse-reward environments. Our framework first bootstraps efficient reasoning using the proposed planning quaternions with long-short chain-of-thought fusion. It then extrapolates to out-of-distribution tasks through complexity-stratified curriculum learning. Finally, the model iteratively refines itself by learning exclusively on experiences selected via reward-gated rejection sampling. Experiments on ALFWorld, ScienceWorld, and WebShop demonstrate that our approach achieves state-of-the-art with significant token efficiency, providing a new recipe for reasoning models in agentic planning.
>
---
#### [new 040] LRDDv2: Enhanced Long-Range Drone Detection Dataset with Range Information and Comprehensive Real-World Challenges
- **分类: cs.CV; cs.RO**

- **简介: 该论文旨在提升长距离无人机检测能力，解决现有数据不足及复杂环境下的挑战，通过添加范围信息和多样化图像，构建了LRDDv2数据库，支持算法优化与研究。**

- **链接: [http://arxiv.org/pdf/2508.03331v1](http://arxiv.org/pdf/2508.03331v1)**

> **作者:** Amirreza Rouhi; Sneh Patel; Noah McCarthy; Siddiqa Khan; Hadi Khorsand; Kaleb Lefkowitz; David K. Han
>
> **备注:** Accepted and presented at ISRR 2024
>
> **摘要:** The exponential growth in Unmanned Aerial Vehicles (UAVs) usage underscores the critical need of detecting them at extended distances to ensure safe operations, especially in densely populated areas. Despite the tremendous advances made in computer vision through deep learning, the detection of these small airborne objects remains a formidable challenge. While several datasets have been developed specifically for drone detection, the need for a more extensive and diverse collection of drone image data persists, particularly for long-range detection under varying environmental conditions. We introduce here the Long Range Drone Detection (LRDD) Version 2 dataset, comprising 39,516 meticulously annotated images, as a second release of the LRDD dataset released previously. The LRDDv2 dataset enhances the LRDDv1 by incorporating a greater variety of images, providing a more diverse and comprehensive resource for drone detection research. What sets LRDDv2 apart is its inclusion of target range information for over 8,000 images, making it possible to develop algorithms for drone range estimation. Tailored for long-range aerial object detection, the majority of LRDDv2's dataset consists of images capturing drones with 50 or fewer pixels in 1080p resolution. For access to the complete Long-Range Drone Detection Dataset (LRDD)v2, please visit https://research.coe.drexel.edu/ece/imaple/lrddv2/ .
>
---
#### [new 041] OmniShape: Zero-Shot Multi-Hypothesis Shape and Pose Estimation in the Real World
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出了一种零样本方法，用于从单个观测点估计物体的形状和姿态，分离形状填充与几何建模，利用条件扩散模型训练两个分布进行采样。**

- **链接: [http://arxiv.org/pdf/2508.03669v1](http://arxiv.org/pdf/2508.03669v1)**

> **作者:** Katherine Liu; Sergey Zakharov; Dian Chen; Takuya Ikeda; Greg Shakhnarovich; Adrien Gaidon; Rares Ambrus
>
> **备注:** 8 pages, 5 figures. This version has typo fixes on top of the version published at ICRA 2025
>
> **摘要:** We would like to estimate the pose and full shape of an object from a single observation, without assuming known 3D model or category. In this work, we propose OmniShape, the first method of its kind to enable probabilistic pose and shape estimation. OmniShape is based on the key insight that shape completion can be decoupled into two multi-modal distributions: one capturing how measurements project into a normalized object reference frame defined by the dataset and the other modelling a prior over object geometries represented as triplanar neural fields. By training separate conditional diffusion models for these two distributions, we enable sampling multiple hypotheses from the joint pose and shape distribution. OmniShape demonstrates compelling performance on challenging real world datasets. Project website: https://tri-ml.github.io/omnishape
>
---
#### [new 042] Can Large Language Models Identify Materials from Radar Signals?
- **分类: eess.SP; cs.ET; cs.RO**

- **简介: 该论文研究了大语言模型（LLMs）能否直接从雷达信号中识别材料，解决了现有方法受限于类别和数据收集的问题，提出LMMaterial通过物理信号处理与检索增强生成技术，实现了对多种材料的开放集识别。**

- **链接: [http://arxiv.org/pdf/2508.03120v1](http://arxiv.org/pdf/2508.03120v1)**

> **作者:** Jiangyou Zhu; Hongyu Deng; He Chen
>
> **摘要:** Accurately identifying the material composition of objects is a critical capability for AI robots powered by large language models (LLMs) to perform context-aware manipulation. Radar technologies offer a promising sensing modality for material recognition task. When combined with deep learning, radar technologies have demonstrated strong potential in identifying the material of various objects. However, existing radar-based solutions are often constrained to closed-set object categories and typically require task-specific data collection to train deep learning models, largely limiting their practical applicability. This raises an important question: Can we leverage the powerful reasoning capabilities of pre-trained LLMs to directly infer material composition from raw radar signals? Answering this question is non-trivial due to the inherent redundancy of radar signals and the fact that pre-trained LLMs have no prior exposure to raw radar data during training. To address this, we introduce LLMaterial, the first study to investigate the feasibility of using LLM to identify materials directly from radar signals. First, we introduce a physics-informed signal processing pipeline that distills high-redundancy radar raw data into a set of compact intermediate parameters that encapsulate the material's intrinsic characteristics. Second, we adopt a retrieval-augmented generation (RAG) strategy to provide the LLM with domain-specific knowledge, enabling it to interpret and reason over the extracted intermediate parameters. Leveraging this integration, the LLM is empowered to perform step-by-step reasoning on the condensed radar features, achieving open-set material recognition directly from raw radar signals. Preliminary results show that LLMaterial can effectively distinguish among a variety of common materials, highlighting its strong potential for real-world material identification applications.
>
---
#### [new 043] LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LiDARCrafter框架，解决生成式世界模型在LiDAR数据中的动态建模问题，通过自然语言指令解析场景图并生成结构化内容，实现4D LiDAR的时空连续性与精细编辑。**

- **链接: [http://arxiv.org/pdf/2508.03692v1](http://arxiv.org/pdf/2508.03692v1)**

> **作者:** Ao Liang; Youquan Liu; Yu Yang; Dongyue Lu; Linfeng Li; Lingdong Kong; Huaici Zhao; Wei Tsang Ooi
>
> **备注:** Preprint; 28 pages, 18 figures, 12 tables; Project Page at https://lidarcrafter.github.io
>
> **摘要:** Generative world models have become essential data engines for autonomous driving, yet most existing efforts focus on videos or occupancy grids, overlooking the unique LiDAR properties. Extending LiDAR generation to dynamic 4D world modeling presents challenges in controllability, temporal coherence, and evaluation standardization. To this end, we present LiDARCrafter, a unified framework for 4D LiDAR generation and editing. Given free-form natural language inputs, we parse instructions into ego-centric scene graphs, which condition a tri-branch diffusion network to generate object structures, motion trajectories, and geometry. These structured conditions enable diverse and fine-grained scene editing. Additionally, an autoregressive module generates temporally coherent 4D LiDAR sequences with smooth transitions. To support standardized evaluation, we establish a comprehensive benchmark with diverse metrics spanning scene-, object-, and sequence-level aspects. Experiments on the nuScenes dataset using this benchmark demonstrate that LiDARCrafter achieves state-of-the-art performance in fidelity, controllability, and temporal consistency across all levels, paving the way for data augmentation and simulation. The code and benchmark are released to the community.
>
---
#### [new 044] Veila: Panoramic LiDAR Generation from a Monocular RGB Image
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一个条件扩散框架（Veila），旨在从单眼RGB图像生成全景LiDAR数据，解决现有方法缺乏细粒度空间控制和跨模态一致性的问题，创新性地融合了CACM、GCMA和PFC机制，并引入交叉模态一致性指标进行评估。**

- **链接: [http://arxiv.org/pdf/2508.03690v1](http://arxiv.org/pdf/2508.03690v1)**

> **作者:** Youquan Liu; Lingdong Kong; Weidong Yang; Ao Liang; Jianxiong Gao; Yang Wu; Xiang Xu; Xin Li; Linfeng Li; Runnan Chen; Ben Fei
>
> **备注:** Preprint; 10 pages, 6 figures, 7 tables
>
> **摘要:** Realistic and controllable panoramic LiDAR data generation is critical for scalable 3D perception in autonomous driving and robotics. Existing methods either perform unconditional generation with poor controllability or adopt text-guided synthesis, which lacks fine-grained spatial control. Leveraging a monocular RGB image as a spatial control signal offers a scalable and low-cost alternative, which remains an open problem. However, it faces three core challenges: (i) semantic and depth cues from RGB are vary spatially, complicating reliable conditioning generation; (ii) modality gaps between RGB appearance and LiDAR geometry amplify alignment errors under noisy diffusion; and (iii) maintaining structural coherence between monocular RGB and panoramic LiDAR is challenging, particularly in non-overlap regions between images and LiDAR. To address these challenges, we propose Veila, a novel conditional diffusion framework that integrates: a Confidence-Aware Conditioning Mechanism (CACM) that strengthens RGB conditioning by adaptively balancing semantic and depth cues according to their local reliability; a Geometric Cross-Modal Alignment (GCMA) for robust RGB-LiDAR alignment under noisy diffusion; and a Panoramic Feature Coherence (PFC) for enforcing global structural consistency across monocular RGB and panoramic LiDAR. Additionally, we introduce two metrics, Cross-Modal Semantic Consistency and Cross-Modal Depth Consistency, to evaluate alignment quality across modalities. Experiments on nuScenes, SemanticKITTI, and our proposed KITTI-Weather benchmark demonstrate that Veila achieves state-of-the-art generation fidelity and cross-modal consistency, while enabling generative data augmentation that improves downstream LiDAR semantic segmentation.
>
---
#### [new 045] COFFEE: A Shadow-Resilient Real-Time Pose Estimator for Unknown Tumbling Asteroids using Sparse Neural Networks
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种实时姿态估计框架COFFEE，针对未知陨石轨迹建模任务，通过稀疏神经网络联合训练对抗自发光遮挡影子的影响，提升姿态估计精度并降低计算需求。**

- **链接: [http://arxiv.org/pdf/2508.03132v1](http://arxiv.org/pdf/2508.03132v1)**

> **作者:** Arion Zimmermann; Soon-Jo Chung; Fred Hadaegh
>
> **备注:** in Proc. 75th Int. Astronautical Congress (IAC-24), Milan, Italy, Oct. 2024
>
> **摘要:** The accurate state estimation of unknown bodies in space is a critical challenge with applications ranging from the tracking of space debris to the shape estimation of small bodies. A necessary enabler to this capability is to find and track features on a continuous stream of images. Existing methods, such as SIFT, ORB and AKAZE, achieve real-time but inaccurate pose estimates, whereas modern deep learning methods yield higher quality features at the cost of more demanding computational resources which might not be available on space-qualified hardware. Additionally, both classical and data-driven methods are not robust to the highly opaque self-cast shadows on the object of interest. We show that, as the target body rotates, these shadows may lead to large biases in the resulting pose estimates. For these objects, a bias in the real-time pose estimation algorithm may mislead the spacecraft's state estimator and cause a mission failure, especially if the body undergoes a chaotic tumbling motion. We present COFFEE, the Celestial Occlusion Fast FEature Extractor, a real-time pose estimation framework for asteroids designed to leverage prior information on the sun phase angle given by sun-tracking sensors commonly available onboard spacecraft. By associating salient contours to their projected shadows, a sparse set of features are detected, invariant to the motion of the shadows. A Sparse Neural Network followed by an attention-based Graph Neural Network feature matching model are then jointly trained to provide a set of correspondences between successive frames. The resulting pose estimation pipeline is found to be bias-free, more accurate than classical pose estimation pipelines and an order of magnitude faster than other state-of-the-art deep learning pipelines on synthetic data as well as on renderings of the tumbling asteroid Apophis.
>
---
#### [new 046] Following Route Instructions using Large Vision-Language Models: A Comparison between Low-level and Panoramic Action Spaces
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文探讨了基于大视觉语言模型的视觉-语言导航任务，旨在解决自主机器人如何根据自然语言指令导航陌生环境的问题。通过对比低级与全景动作空间，研究了off-the-shelf LVLMs在VLN任务中的有效性，发现其在R2R数据集上实现41%的成功率，表明其仍需进一步优化以匹配专门设计的导航模型。**

- **链接: [http://arxiv.org/pdf/2508.02917v1](http://arxiv.org/pdf/2508.02917v1)**

> **作者:** Vebjørn Haug Kåsene; Pierre Lison
>
> **备注:** This paper has been accepted to ICNSLP 2025
>
> **摘要:** Vision-and-Language Navigation (VLN) refers to the task of enabling autonomous robots to navigate unfamiliar environments by following natural language instructions. While recent Large Vision-Language Models (LVLMs) have shown promise in this task, most current VLM systems rely on models specifically designed and optimized for navigation, leaving the potential of off-the-shelf LVLMs underexplored. Furthermore, while older VLN approaches used low-level action spaces with egocentric views and atomic actions (such as "turn left" or "move forward"), newer models tend to favor panoramic action spaces with discrete navigable viewpoints. This paper investigates (1) whether off-the-shelf LVLMs (fine-tuned without architectural modifications or simulator-based training) can effectively support VLN tasks and (2) whether such models can support both low-level and panoramic action paradigms. To this end, we fine-tune the open-source model Qwen2.5-VL-3B-Instruct on the Room-to-Room (R2R) dataset and evaluate its empirical performance across both low-level and panoramic action spaces. The best resulting model achieves a 41% success rate on the R2R test set, demonstrating that while off-the-shelf LVLMs can learn to perform Vision-and-Language Navigation, they still lag behind models specifically designed for this task.
>
---
#### [new 047] Frequency Point Game Environment for UAVs via Expert Knowledge and Large Language Model
- **分类: cs.MA; cs.GT; cs.RO**

- **简介: 该论文提出基于专家知识和大语言模型的频率点游戏环境模型，用于模拟无人机干扰与对抗策略，解决谱竞争、路径规划等问题，通过迭代交互优化路径规划，提升抗干扰能力。**

- **链接: [http://arxiv.org/pdf/2508.02757v1](http://arxiv.org/pdf/2508.02757v1)**

> **作者:** Jingpu Yang
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) have made significant advancements in communication stability and security through techniques such as frequency hopping, signal spreading, and adaptive interference suppression. However, challenges remain in modeling spectrum competition, integrating expert knowledge, and predicting opponent behavior. To address these issues, we propose UAV-FPG (Unmanned Aerial Vehicle - Frequency Point Game), a game-theoretic environment model that simulates the dynamic interaction between interference and anti-interference strategies of opponent and ally UAVs in communication frequency bands. The model incorporates a prior expert knowledge base to optimize frequency selection and employs large language models for path planning, simulating a "strong adversary". Experimental results highlight the effectiveness of integrating the expert knowledge base and the large language model, with the latter significantly improving path planning in dynamic scenarios through iterative interactions, outperforming fixed-path strategies. UAV-FPG provides a robust platform for advancing anti-jamming strategies and intelligent decision-making in UAV communication systems.
>
---
## 更新

#### [replaced 001] Vision-Language Fusion for Real-Time Autonomous Driving: Goal-Centered Cross-Attention of Camera, HD-Map, & Waypoints
- **分类: cs.CV; cs.AI; cs.LG; cs.RO; I.4.8; I.2.10; I.2.6; C.3.3; I.4.9**

- **链接: [http://arxiv.org/pdf/2507.23064v2](http://arxiv.org/pdf/2507.23064v2)**

> **作者:** Santosh Patapati; Trisanth Srinivasan; Murari Ambati
>
> **备注:** 5 pages
>
> **摘要:** Autonomous cars need geometric accuracy and semantic understanding to navigate complex environments, yet most stacks handle them separately. We present XYZ-Drive, a single vision-language model that reads a front-camera frame, a 25m $\times$ 25m overhead map, and the next waypoint, then outputs steering and speed. A lightweight goal-centered cross-attention layer lets waypoint tokens highlight relevant image and map patches, supporting both action and textual explanations, before the fused tokens enter a partially fine-tuned LLaMA-3.2 11B model. On the MD-NEX Outdoor-Driving benchmark XYZ-Drive attains 95% success and 0.80 Success weighted by Path Length (SPL), surpassing PhysNav-DG by 15%. and halving collisions, all while significantly improving efficiency by using only a single branch. Sixteen ablations explain the gains. Removing any modality (vision, waypoint, map) drops success by up to 11%, confirming their complementary roles and rich connections. Replacing goal-centered attention with simple concatenation cuts 3% in performance, showing query-based fusion injects map knowledge more effectively. Keeping the transformer frozen loses 5%, showing the importance of fine-tuning when applying VLMs for specific tasks such as autonomous driving. Coarsening map resolution from 10 cm to 40 cm blurs lane edges and raises crash rate. Overall, these results demonstrate that early, token-level fusion of intent and map layout enables accurate, transparent, real-time driving.
>
---
#### [replaced 002] 16 Ways to Gallop: Energetics and Body Dynamics of High-Speed Quadrupedal Gaits
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.13716v2](http://arxiv.org/pdf/2503.13716v2)**

> **作者:** Yasser G. Alqaham; Jing Cheng; Zhenyu Gan
>
> **备注:** 7 pages, 6 figures, Accepted for IROS 2025
>
> **摘要:** Galloping is a common high-speed gait in both animals and quadrupedal robots, yet its energetic characteristics remain insufficiently explored. This study systematically analyzes a large number of possible galloping gaits by categorizing them based on the number of flight phases per stride and the phase relationships between the front and rear legs, following Hildebrand's framework for asymmetrical gaits. Using the A1 quadrupedal robot from Unitree, we model galloping dynamics as a hybrid dynamical system and employ trajectory optimization (TO) to minimize the cost of transport (CoT) across a range of speeds. Our results reveal that rotary and transverse gallop footfall sequences exhibit no fundamental energetic difference, despite variations in body yaw and roll motion. However, the number of flight phases significantly impacts energy efficiency: galloping with no flight phases is optimal at lower speeds, whereas galloping with two flight phases minimizes energy consumption at higher speeds. We validate these findings using a quadratic programming (QP)-based controller, developed in our previous work, in Gazebo simulations. These insights advance the understanding of quadrupedal locomotion energetics and may inform future legged robot designs for adaptive, energy-efficient gait transitions.
>
---
#### [replaced 003] Opt-in Camera: Person Identification in Video via UWB Localization and Its Application to Opt-in Systems
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.19891v2](http://arxiv.org/pdf/2409.19891v2)**

> **作者:** Matthew Ishige; Yasuhiro Yoshimura; Ryo Yonetani
>
> **备注:** IROS 2025
>
> **摘要:** This paper presents opt-in camera, a concept of privacy-preserving camera systems capable of recording only specific individuals in a crowd who explicitly consent to be recorded. Our system utilizes a mobile wireless communication tag attached to personal belongings as proof of opt-in and as a means of localizing tag carriers in video footage. Specifically, the on-ground positions of the wireless tag are first tracked over time using the unscented Kalman filter (UKF). The tag trajectory is then matched against visual tracking results for pedestrians found in videos to identify the tag carrier. Technically, we devise a dedicated trajectory matching technique based on constrained linear optimization, as well as a novel calibration technique that handles wireless tag-camera calibration and hyperparameter tuning for the UKF, which mitigates the non-line-of-sight (NLoS) issue in wireless localization. We implemented the proposed opt-in camera system using ultra-wideband (UWB) devices and an off-the-shelf webcam. Experimental results demonstrate that our system can perform opt-in recording of individuals in real-time at 10 fps, with reliable identification accuracy in crowds of 8-23 people in a confined space.
>
---
#### [replaced 004] Rethink Repeatable Measures of Robot Performance with Statistical Query
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2505.08216v2](http://arxiv.org/pdf/2505.08216v2)**

> **作者:** Bowen Weng; Linda Capito; Guillermo A. Castillo; Dylan Khor
>
> **摘要:** For a general standardized testing algorithm designed to evaluate a specific aspect of a robot's performance, several key expectations are commonly imposed. Beyond accuracy (i.e., closeness to a typically unknown ground-truth reference) and efficiency (i.e., feasibility within acceptable testing costs and equipment constraints), one particularly important attribute is repeatability. Repeatability refers to the ability to consistently obtain the same testing outcome when similar testing algorithms are executed on the same subject robot by different stakeholders, across different times or locations. However, achieving repeatable testing has become increasingly challenging as the components involved grow more complex, intelligent, diverse, and, most importantly, stochastic. While related efforts have addressed repeatability at ethical, hardware, and procedural levels, this study focuses specifically on repeatable testing at the algorithmic level. Specifically, we target the well-adopted class of testing algorithms in standardized evaluation: statistical query (SQ) algorithms (i.e., algorithms that estimate the expected value of a bounded function over a distribution using sampled data). We propose a lightweight, parameterized, and adaptive modification applicable to any SQ routine, whether based on Monte Carlo sampling, importance sampling, or adaptive importance sampling, that makes it provably repeatable, with guaranteed bounds on both accuracy and efficiency. We demonstrate the effectiveness of the proposed approach across three representative scenarios: (i) established and widely adopted standardized testing of manipulators, (ii) emerging intelligent testing algorithms for operational risk assessment in automated vehicles, and (iii) developing use cases involving command tracking performance evaluation of humanoid robots in locomotion tasks.
>
---
#### [replaced 005] DriveSOTIF: Advancing Perception SOTIF Through Multimodal Large Language Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.07084v2](http://arxiv.org/pdf/2505.07084v2)**

> **作者:** Shucheng Huang; Freda Shi; Chen Sun; Jiaming Zhong; Minghao Ning; Yufeng Yang; Yukun Lu; Hong Wang; Amir Khajepour
>
> **备注:** This work has been submitted to the IEEE for possible publication. V2 of the manuscript, submitted to IEEE-TVT;
>
> **摘要:** Human drivers possess spatial and causal intelligence, enabling them to perceive driving scenarios, anticipate hazards, and react to dynamic environments. In contrast, autonomous vehicles lack these abilities, making it challenging to manage perception-related Safety of the Intended Functionality (SOTIF) risks, especially under complex or unpredictable driving conditions. To address this gap, we propose fine-tuning multimodal large language models (MLLMs) on a customized dataset specifically designed to capture perception-related SOTIF scenarios. Benchmarking results show that fine-tuned MLLMs achieve an 11.8\% improvement in close-ended VQA accuracy and a 12.0\% increase in open-ended VQA scores compared to baseline models, while maintaining real-time performance with a 0.59-second average inference time per image. We validate our approach through real-world case studies in Canada and China, where fine-tuned models correctly identify safety risks that challenge even experienced human drivers. This work represents the first application of domain-specific MLLM fine-tuning for SOTIF domain in autonomous driving. The dataset and related resources are available at github.com/s95huang/DriveSOTIF.git
>
---
#### [replaced 006] Improving Drone Racing Performance Through Iterative Learning MPC
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2508.01103v2](http://arxiv.org/pdf/2508.01103v2)**

> **作者:** Haocheng Zhao; Niklas Schlüter; Lukas Brunke; Angela P. Schoellig
>
> **备注:** Accepted for oral presentation at IROS 2025
>
> **摘要:** Autonomous drone racing presents a challenging control problem, requiring real-time decision-making and robust handling of nonlinear system dynamics. While iterative learning model predictive control (LMPC) offers a promising framework for iterative performance improvement, its direct application to drone racing faces challenges like real-time compatibility or the trade-off between time-optimal and safe traversal. In this paper, we enhance LMPC with three key innovations: (1) an adaptive cost function that dynamically weights time-optimal tracking against centerline adherence, (2) a shifted local safe set to prevent excessive shortcutting and enable more robust iterative updates, and (3) a Cartesian-based formulation that accommodates safety constraints without the singularities or integration errors associated with Frenet-frame transformations. Results from extensive simulation and real-world experiments demonstrate that our improved algorithm can optimize initial trajectories generated by a wide range of controllers with varying levels of tuning for a maximum improvement in lap time by 60.85%. Even applied to the most aggressively tuned state-of-the-art model-based controller, MPCC++, on a real drone, a 6.05% improvement is still achieved. Overall, the proposed method pushes the drone toward faster traversal and avoids collisions in simulation and real-world experiments, making it a practical solution to improve the peak performance of drone racing.
>
---
#### [replaced 007] Equivariant Volumetric Grasping
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.18847v2](http://arxiv.org/pdf/2507.18847v2)**

> **作者:** Pinhao Song; Yutong Hu; Pengteng Li; Renaud Detry
>
> **备注:** 19 pages
>
> **摘要:** We propose a new volumetric grasp model that is equivariant to rotations around the vertical axis, leading to a significant improvement in sample efficiency. Our model employs a tri-plane volumetric feature representation -- i.e., the projection of 3D features onto three canonical planes. We introduce a novel tri-plane feature design in which features on the horizontal plane are equivariant to 90{\deg} rotations, while the sum of features from the other two planes remains invariant to the same transformations. This design is enabled by a new deformable steerable convolution, which combines the adaptability of deformable convolutions with the rotational equivariance of steerable ones. This allows the receptive field to adapt to local object geometry while preserving equivariance properties. We further develop equivariant adaptations of two state-of-the-art volumetric grasp planners, GIGA and IGD. Specifically, we derive a new equivariant formulation of IGD's deformable attention mechanism and propose an equivariant generative model of grasp orientations based on flow matching. We provide a detailed analytical justification of the proposed equivariance properties and validate our approach through extensive simulated and real-world experiments. Our results demonstrate that the proposed projection-based design significantly reduces both computational and memory costs. Moreover, the equivariant grasp models built on top of our tri-plane features consistently outperform their non-equivariant counterparts, achieving higher performance with only a modest computational overhead. Video and code can be viewed in: https://mousecpn.github.io/evg-page/
>
---
#### [replaced 008] Residual Koopman Model Predictive Control for Enhanced Vehicle Dynamics with Small On-Track Data Input
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.18396v2](http://arxiv.org/pdf/2507.18396v2)**

> **作者:** Yonghao Fu; Cheng Hu; Haokun Xiong; Zhanpeng Bao; Wenyuan Du; Edoardo Ghignone; Michele Magno; Lei Xie; Hongye Su
>
> **摘要:** In vehicle trajectory tracking tasks, the simplest approach is the Pure Pursuit (PP) Control. However, this single-point preview tracking strategy fails to consider vehicle model constraints, compromising driving safety. Model Predictive Control (MPC) as a widely adopted control method, optimizes control actions by incorporating mechanistic models and physical constraints. While its control performance critically depends on the accuracy of vehicle modeling. Traditional vehicle modeling approaches face inherent trade-offs between capturing nonlinear dynamics and maintaining computational efficiency, often resulting in reduced control performance. To address these challenges, this paper proposes Residual Koopman Model Predictive Control (RKMPC) framework. This method uses two linear MPC architecture to calculate control inputs: a Linear Model Predictive Control (LMPC) computes the baseline control input based on the vehicle kinematic model, and a neural network-based RKMPC calculates the compensation input. The final control command is obtained by adding these two components. This design preserves the reliability and interpretability of traditional mechanistic model while achieving performance optimization through residual modeling. This method has been validated on the Carsim-Matlab joint simulation platform and a physical 1:10 scale F1TENTH racing car. Experimental results show that RKMPC requires only 20% of the training data needed by traditional Koopman Model Predictive Control (KMPC) while delivering superior tracking performance. Compared to traditional LMPC, RKMPC reduces lateral error by 11.7%-22.1%, decreases heading error by 8.9%-15.8%, and improves front-wheel steering stability by up to 27.6%. The implementation code is available at: https://github.com/ZJU-DDRX/Residual Koopman.
>
---
#### [replaced 009] DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.20900v4](http://arxiv.org/pdf/2502.20900v4)**

> **作者:** Yifan Zhong; Xuchuan Huang; Ruochong Li; Ceyao Zhang; Zhang Chen; Tianrui Guan; Fanlian Zeng; Ka Num Lui; Yuyao Ye; Yitao Liang; Yaodong Yang; Yuanpei Chen
>
> **备注:** 19 pages, 11 figures
>
> **摘要:** Dexterous grasping remains a fundamental yet challenging problem in robotics. A general-purpose robot must be capable of grasping diverse objects in arbitrary scenarios. However, existing research typically relies on restrictive assumptions, such as single-object settings or limited environments, showing constrained generalization. We present DexGraspVLA, a hierarchical framework for robust generalization in language-guided general dexterous grasping and beyond. It utilizes a pre-trained Vision-Language model as the high-level planner and learns a diffusion-based low-level Action controller. The key insight to achieve generalization lies in iteratively transforming diverse language and visual inputs into domain-invariant representations via foundation models, where imitation learning can be effectively applied due to the alleviation of domain shift. Notably, our method achieves a 90+% dexterous grasping success rate under thousands of challenging unseen cluttered scenes. Empirical analysis confirms the consistency of internal model behavior across environmental variations, validating our design. DexGraspVLA also, for the first time, simultaneously demonstrates free-form long-horizon prompt execution, robustness to adversarial objects and human disturbance, and failure recovery. Extended application to nonprehensile grasping further proves its generality. Project website: https://dexgraspvla.github.io.
>
---
#### [replaced 010] Non-Prehensile Tool-Object Manipulation by Integrating LLM-Based Planning and Manoeuvrability-Driven Controls
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.06931v5](http://arxiv.org/pdf/2412.06931v5)**

> **作者:** Hoi-Yin Lee; Peng Zhou; Anqing Duan; Wanyu Ma; Chenguang Yang; David Navarro-Alarcon
>
> **摘要:** Being able to use tools is a widely recognised indicator of intelligence across species. Humans, for instance, have demonstrated mastery of tool use for over two million years. The ability to use tools is invaluable as it extends an organism's reach and enhances its capacity to interact with objects and the environment. Being able to understand the geometric-mechanical relations between the tools-objects-environments allows certain species (e.g., apes and crows) to reach food in narrow constrained spaces. The same principles of physical augmentation and its associated non-prehensile manipulation capabilities also apply to robotic systems. For example, by instrumenting them with different types of end-effectors, robots can (in principle) dexterously interact (e.g., push and flip) with objects of various shapes and masses akin to its biological counterpart. However, developing this type of manipulation skill is still an open research problem. Furthermore, the complexity of planning tool-object manipulation tasks, particularly in coordinating the actions of dual-arm robots, presents significant challenges. To address these complexities, we propose integrating Large Language Models (LLMs) to assist in planning and executing these intricate manipulations, thereby enhancing the robot's ability to perform in diverse scenarios.
>
---
#### [replaced 011] Real-Time Sense and Detect of Drones Using Deep Learning and Airborne LiDAR
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2310.09589v3](http://arxiv.org/pdf/2310.09589v3)**

> **作者:** Manduhu Manduhu; Alexander Dow; Petar Trslic; Gerard Dooly; Benjamin Blanck; James Riordan
>
> **摘要:** The safe operation of drone swarms beyond visual line of sight requires multiple safeguards to mitigate the risk of collision between drones flying in close-proximity scenarios. Cooperative navigation and flight coordination strategies that rely on pre-planned trajectories, constant %{satellite and network connectivity and reliable Global Navigation Satellite System (GNSS) positioning are brittle to failure. Drone embedded sense and detect offers a comprehensive mode of separation between drones for deconfliction and collision avoidance. This paper presents the first airborne LiDAR based solution for drone-swarm detection and localization using 3D deep learning model. It adapts an existing deep learning neural network to the air-to-air drone scenario by expanding the scan space vertically. A new sparse convolution is proposed and applied to accelerate the backbone layer, which is the most time-consuming part of the neural network. To collect training data of safety critical, close-proximity multi-drone operations, a scenario Digital Twin is used to augment real datasets with high fidelity synthetic data. The trained model achieves over 80% recall and 96% precision when tested on real-world datasets. By incorporating a tracking-by-detection algorithm the system can reliably monitor the separation distance of multiple drones in challenging environments.
>
---
#### [replaced 012] Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.17213v2](http://arxiv.org/pdf/2506.17213v2)**

> **作者:** Xiuyu Yang; Shuhan Tan; Philipp Krähenbühl
>
> **备注:** ICCV 2025. Project page: https://orangesodahub.github.io/InfGen Code: https://github.com/OrangeSodahub/infgen
>
> **摘要:** An ideal traffic simulator replicates the realistic long-term point-to-point trip that a self-driving system experiences during deployment. Prior models and benchmarks focus on closed-loop motion simulation for initial agents in a scene. This is problematic for long-term simulation. Agents enter and exit the scene as the ego vehicle enters new regions. We propose InfGen, a unified next-token prediction model that performs interleaved closed-loop motion simulation and scene generation. InfGen automatically switches between closed-loop motion simulation and scene generation mode. It enables stable long-term rollout simulation. InfGen performs at the state-of-the-art in short-term (9s) traffic simulation, and significantly outperforms all other methods in long-term (30s) simulation. The code and model of InfGen will be released at https://orangesodahub.github.io/InfGen
>
---
#### [replaced 013] Doppler-SLAM: Doppler-Aided Radar-Inertial and LiDAR-Inertial Simultaneous Localization and Mapping
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.11634v3](http://arxiv.org/pdf/2504.11634v3)**

> **作者:** Dong Wang; Hannes Haag; Daniel Casado Herraez; Stefan May; Cyrill Stachniss; Andreas Nüchter
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Simultaneous localization and mapping (SLAM) is a critical capability for autonomous systems. Traditional SLAM approaches, which often rely on visual or LiDAR sensors, face significant challenges in adverse conditions such as low light or featureless environments. To overcome these limitations, we propose a novel Doppler-aided radar-inertial and LiDAR-inertial SLAM framework that leverages the complementary strengths of 4D radar, FMCW LiDAR, and inertial measurement units. Our system integrates Doppler velocity measurements and spatial data into a tightly-coupled front-end and graph optimization back-end to provide enhanced ego velocity estimation, accurate odometry, and robust mapping. We also introduce a Doppler-based scan-matching technique to improve front-end odometry in dynamic environments. In addition, our framework incorporates an innovative online extrinsic calibration mechanism, utilizing Doppler velocity and loop closure to dynamically maintain sensor alignment. Extensive evaluations on both public and proprietary datasets show that our system significantly outperforms state-of-the-art radar-SLAM and LiDAR-SLAM frameworks in terms of accuracy and robustness. To encourage further research, the code of our Doppler-SLAM and our dataset are available at: https://github.com/Wayne-DWA/Doppler-SLAM.
>
---
#### [replaced 014] Simultaneous Pick and Place Detection by Combining SE(3) Diffusion Models with Differential Kinematics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.19502v2](http://arxiv.org/pdf/2504.19502v2)**

> **作者:** Tianyi Ko; Takuya Ikeda; Balazs Opra; Koichi Nishiwaki
>
> **备注:** Accepted for IROS2025
>
> **摘要:** Grasp detection methods typically target the detection of a set of free-floating hand poses that can grasp the object. However, not all of the detected grasp poses are executable due to physical constraints. Even though it is straightforward to filter invalid grasp poses in the post-process, such a two-staged approach is computationally inefficient, especially when the constraint is hard. In this work, we propose an approach to take the following two constraints into account during the grasp detection stage, namely, (i) the picked object must be able to be placed with a predefined configuration without in-hand manipulation (ii) it must be reachable by the robot under the joint limit and collision-avoidance constraints for both pick and place cases. Our key idea is to train an SE(3) grasp diffusion network to estimate the noise in the form of spatial velocity, and constrain the denoising process by a multi-target differential inverse kinematics with an inequality constraint, so that the states are guaranteed to be reachable and placement can be performed without collision. In addition to an improved success ratio, we experimentally confirmed that our approach is more efficient and consistent in computation time compared to a naive two-stage approach.
>
---
#### [replaced 015] 3DRot: 3D Rotation Augmentation for RGB-Based 3D Tasks
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.01423v2](http://arxiv.org/pdf/2508.01423v2)**

> **作者:** Shitian Yang; Deyu Li; Xiaoke Jiang; Lei Zhang
>
> **摘要:** RGB-based 3D tasks, e.g., 3D detection, depth estimation, 3D keypoint estimation, still suffer from scarce, expensive annotations and a thin augmentation toolbox, since most image transforms, including resize and rotation, disrupt geometric consistency. In this paper, we introduce 3DRot, a plug-and-play augmentation that rotates and mirrors images about the camera's optical center while synchronously updating RGB images, camera intrinsics, object poses, and 3D annotations to preserve projective geometry-achieving geometry-consistent rotations and reflections without relying on any scene depth. We validate 3DRot with a classical 3D task, monocular 3D detection. On SUN RGB-D dataset, 3DRot raises $IoU_{3D}$ from 43.21 to 44.51, cuts rotation error (ROT) from 22.91$^\circ$ to 20.93$^\circ$, and boosts $mAP_{0.5}$ from 35.70 to 38.11. As a comparison, Cube R-CNN adds 3 other datasets together with SUN RGB-D for monocular 3D estimation, with a similar mechanism and test dataset, increases $IoU_{3D}$ from 36.2 to 37.8, boosts $mAP_{0.5}$ from 34.7 to 35.4. Because it operates purely through camera-space transforms, 3DRot is readily transferable to other 3D tasks.
>
---
#### [replaced 016] Diffuse-CLoC: Guided Diffusion for Physics-based Character Look-ahead Control
- **分类: cs.GR; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.11801v3](http://arxiv.org/pdf/2503.11801v3)**

> **作者:** Xiaoyu Huang; Takara Truong; Yunbo Zhang; Fangzhou Yu; Jean Pierre Sleiman; Jessica Hodgins; Koushil Sreenath; Farbod Farshidian
>
> **摘要:** We present Diffuse-CLoC, a guided diffusion framework for physics-based look-ahead control that enables intuitive, steerable, and physically realistic motion generation. While existing kinematics motion generation with diffusion models offer intuitive steering capabilities with inference-time conditioning, they often fail to produce physically viable motions. In contrast, recent diffusion-based control policies have shown promise in generating physically realizable motion sequences, but the lack of kinematics prediction limits their steerability. Diffuse-CLoC addresses these challenges through a key insight: modeling the joint distribution of states and actions within a single diffusion model makes action generation steerable by conditioning it on the predicted states. This approach allows us to leverage established conditioning techniques from kinematic motion generation while producing physically realistic motions. As a result, we achieve planning capabilities without the need for a high-level planner. Our method handles a diverse set of unseen long-horizon downstream tasks through a single pre-trained model, including static and dynamic obstacle avoidance, motion in-betweening, and task-space control. Experimental results show that our method significantly outperforms the traditional hierarchical framework of high-level motion diffusion and low-level tracking.
>
---
#### [replaced 017] The Starlink Robot: A Platform and Dataset for Mobile Satellite Communication
- **分类: cs.RO; cs.NI**

- **链接: [http://arxiv.org/pdf/2506.19781v3](http://arxiv.org/pdf/2506.19781v3)**

> **作者:** Boyi Liu; Qianyi Zhang; Qiang Yang; Jianhao Jiao; Jagmohan Chauhan; Dimitrios Kanoulas
>
> **摘要:** The integration of satellite communication into mobile devices represents a paradigm shift in connectivity, yet the performance characteristics under motion and environmental occlusion remain poorly understood. We present the Starlink Robot, the first mobile robotic platform equipped with Starlink satellite internet, comprehensive sensor suite including upward-facing camera, LiDAR, and IMU, designed to systematically study satellite communication performance during movement. Our multi-modal dataset captures synchronized communication metrics, motion dynamics, sky visibility, and 3D environmental context across diverse scenarios including steady-state motion, variable speeds, and different occlusion conditions. This platform and dataset enable researchers to develop motion-aware communication protocols, predict connectivity disruptions, and optimize satellite communication for emerging mobile applications from smartphones to autonomous vehicles. In this work, we use LEOViz for real-time satellite tracking and data collection. The starlink robot project is available at https://github.com/StarlinkRobot.
>
---
#### [replaced 018] Breaking Imitation Bottlenecks: Reinforced Diffusion Powers Diverse Trajectory Generation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.04049v2](http://arxiv.org/pdf/2507.04049v2)**

> **作者:** Ziying Song; Lin Liu; Hongyu Pan; Bencheng Liao; Mingzhe Guo; Lei Yang; Yongchang Zhang; Shaoqing Xu; Caiyan Jia; Yadan Luo
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Most end-to-end autonomous driving methods rely on imitation learning from single expert demonstrations, often leading to conservative and homogeneous behaviors that limit generalization in complex real-world scenarios. In this work, we propose DIVER, an end-to-end driving framework that integrates reinforcement learning with diffusion-based generation to produce diverse and feasible trajectories. At the core of DIVER lies a reinforced diffusion-based generation mechanism. First, the model conditions on map elements and surrounding agents to generate multiple reference trajectories from a single ground-truth trajectory, alleviating the limitations of imitation learning that arise from relying solely on single expert demonstrations. Second, reinforcement learning is employed to guide the diffusion process, where reward-based supervision enforces safety and diversity constraints on the generated trajectories, thereby enhancing their practicality and generalization capability. Furthermore, to address the limitations of L2-based open-loop metrics in capturing trajectory diversity, we propose a novel Diversity metric to evaluate the diversity of multi-mode predictions.Extensive experiments on the closed-loop NAVSIM and Bench2Drive benchmarks, as well as the open-loop nuScenes dataset, demonstrate that DIVER significantly improves trajectory diversity, effectively addressing the mode collapse problem inherent in imitation learning.
>
---
#### [replaced 019] OWLed: Outlier-weighed Layerwise Pruning for Efficient Autonomous Driving Framework
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.07711v3](http://arxiv.org/pdf/2411.07711v3)**

> **作者:** Jiaxi Li; Lu Yin; Xilu Wang
>
> **备注:** IJCNN2025
>
> **摘要:** The integration of Large Language Models (LLMs) into autonomous driving systems offers promising enhancements in environmental understanding and decision-making. However, the substantial computational demands of deploying LLMs locally on vehicles render this approach unfeasible for real-world automotive applications. To address this challenge, we introduce OWLed, the Outlier-Weighed Layerwise Pruning for Efficient Autonomous Driving Framework that leverages outlier-weighted layerwise sparsity for model compression. Our method assigns non-uniform sparsity ratios to different layers based on the distribution of outlier features, significantly reducing the model size without the need for fine-tuning. To ensure the compressed model adapts well to autonomous driving tasks, we incorporate driving environment data into both the calibration and pruning processes. Our empirical studies reveal that the encoder component is more sensitive to pruning than the LLM, highlighting its critical role in the system. Experimental results demonstrate that OWLed outperforms existing methods in perception, action prediction, and language understanding while substantially lowering computational requirements. These findings underscore the potential of combining advanced pruning techniques with LLMs to develop efficient and robust autonomous driving systems capable of handling complex scenarios. Code will be made publicly available.
>
---
#### [replaced 020] Boosting Omnidirectional Stereo Matching with a Pre-trained Depth Foundation Model
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.23502v2](http://arxiv.org/pdf/2503.23502v2)**

> **作者:** Jannik Endres; Oliver Hahn; Charles Corbière; Simone Schaub-Meyer; Stefan Roth; Alexandre Alahi
>
> **备注:** Accepted at IROS 2025. Project page: https://vita-epfl.github.io/DFI-OmniStereo-website/
>
> **摘要:** Omnidirectional depth perception is essential for mobile robotics applications that require scene understanding across a full 360{\deg} field of view. Camera-based setups offer a cost-effective option by using stereo depth estimation to generate dense, high-resolution depth maps without relying on expensive active sensing. However, existing omnidirectional stereo matching approaches achieve only limited depth accuracy across diverse environments, depth ranges, and lighting conditions, due to the scarcity of real-world data. We present DFI-OmniStereo, a novel omnidirectional stereo matching method that leverages a large-scale pre-trained foundation model for relative monocular depth estimation within an iterative optimization-based stereo matching architecture. We introduce a dedicated two-stage training strategy to utilize the relative monocular depth features for our omnidirectional stereo matching before scale-invariant fine-tuning. DFI-OmniStereo achieves state-of-the-art results on the real-world Helvipad dataset, reducing disparity MAE by approximately 16% compared to the previous best omnidirectional stereo method.
>
---
#### [replaced 021] Unveiling the Potential of iMarkers: Invisible Fiducial Markers for Advanced Robotics
- **分类: cs.RO; cs.CV; I.2.10; I.2.9; I.4.8**

- **链接: [http://arxiv.org/pdf/2501.15505v3](http://arxiv.org/pdf/2501.15505v3)**

> **作者:** Ali Tourani; Deniz Isinsu Avsar; Hriday Bavle; Jose Luis Sanchez-Lopez; Jan Lagerwall; Holger Voos
>
> **备注:** 18 pages, 10 figures, 3 tables
>
> **摘要:** Fiducial markers are widely used in various robotics tasks, facilitating enhanced navigation, object recognition, and scene understanding. Despite their advantages for robots and Augmented Reality (AR) applications, they often disrupt the visual aesthetics of environments because they are visible to humans, making them unsuitable for non-intrusive use cases. To address this gap, this paper presents "iMarkers"-innovative, unobtrusive fiducial markers detectable exclusively by robots equipped with specialized sensors. These markers offer high flexibility in production, allowing customization of their visibility range and encoding algorithms to suit various demands. The paper also introduces the hardware designs and software algorithms developed for detecting iMarkers, highlighting their adaptability and robustness in the detection and recognition stages. Various evaluations have demonstrated the effectiveness of iMarkers compared to conventional (printed) and blended fiducial markers and confirmed their applicability in diverse robotics scenarios.
>
---
