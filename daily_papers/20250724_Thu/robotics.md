# 机器人 cs.RO

- **最新发布 50 篇**

- **更新 20 篇**

## 最新发布

#### [new 001] Analytical Formulation of Autonomous Vehicle Freeway Merging Control with State-Dependent Discharge Rates
- **分类: cs.RO**

- **简介: 该论文属于智能交通系统任务，旨在解决自动驾驶车辆在高速匝道合流时的效率与安全问题。通过引入状态依赖的放行率模型，推导出动态排队和延误指标，并建立碰撞风险函数。最终以动态规划模型优化合流位置和速度，实现延迟和风险最小化。**

- **链接: [http://arxiv.org/pdf/2507.16846v1](http://arxiv.org/pdf/2507.16846v1)**

> **作者:** Qing Tang; Xianbiao Hu
>
> **备注:** Accepted for publication in IEEE Transactions on Intelligent Transportation Systems (2025) as a regular paper (minor revision approved)
>
> **摘要:** The core of the freeway merging control problem lies in dynamic queue propagation and dissipation linked to merging vehicle behavior. Traditionally, queuing is modeled through demand-supply interactions with time varying demand and fixed capacity. However, field observations show flow rates decrease during congestion at freeway merges due to the impact of intersecting traffic, a factor overlooked in fundamental diagrams. This manuscript introduces an analytical approach to characterize and control the dynamic multi-stage merging of autonomous vehicles, prioritizing traffic efficiency and safety. For the first time, the effective discharge rate at the merging point, reduced by the multi-stage dynamic merging process, is analytically derived using a closed form formulation. Leveraging this expression, performance metrics such as queue length and traffic delay are derived as the first objective. Additionally, a crash risk function is established to quantitatively assess potential collisions during the merging process, serving as the second objective. Finally, the problem is formulated as a dynamic programming model to jointly minimize delay and crash risk, with the merging location and speed as decision variables. Given the terminal state, the ramp vehicle merging task is formulated as a recursive optimization problem, employing backward induction to find the minimum cost solution. Numerical experiments using the NGSIM dataset validate the derived effective discharge rate. The results indicate that the proposed model outperforms two benchmark algorithms, leading to a more efficient and safer merging process.
>
---
#### [new 002] InstructVLA: Vision-Language-Action Instruction Tuning from Understanding to Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉-语言-动作（VLA）建模任务，旨在解决现有模型在多模态推理与动作生成间的权衡问题。论文提出InstructVLA模型，通过VLA-IT训练方法，结合多模态数据与专家混合机制，提升任务理解与操作性能，实现更强的泛化与推理能力。**

- **链接: [http://arxiv.org/pdf/2507.17520v1](http://arxiv.org/pdf/2507.17520v1)**

> **作者:** Shuai Yang; Hao Li; Yilun Chen; Bin Wang; Yang Tian; Tai Wang; Hanqing Wang; Feng Zhao; Yiyi Liao; Jiangmiao Pang
>
> **备注:** 38 pages
>
> **摘要:** To operate effectively in the real world, robots must integrate multimodal reasoning with precise action generation. However, existing vision-language-action (VLA) models often sacrifice one for the other, narrow their abilities to task-specific manipulation data, and suffer catastrophic forgetting of pre-trained vision-language capabilities. To bridge this gap, we introduce InstructVLA, an end-to-end VLA model that preserves the flexible reasoning of large vision-language models (VLMs) while delivering leading manipulation performance. InstructVLA introduces a novel training paradigm, Vision-Language-Action Instruction Tuning (VLA-IT), which employs multimodal training with mixture-of-experts adaptation to jointly optimize textual reasoning and action generation on both standard VLM corpora and a curated 650K-sample VLA-IT dataset. On in-domain SimplerEnv tasks, InstructVLA achieves 30.5% improvement over SpatialVLA. To evaluate generalization, we introduce SimplerEnv-Instruct, an 80-task benchmark requiring closed-loop control and high-level instruction understanding, where it outperforms a fine-tuned OpenVLA by 92% and an action expert aided by GPT-4o by 29%. Additionally, InstructVLA surpasses baseline VLMs on multimodal tasks and exhibits inference-time scaling by leveraging textual reasoning to boost manipulation performance in both simulated and real-world settings. These results demonstrate InstructVLA's potential for bridging intuitive and steerable human-robot interaction with efficient policy learning.
>
---
#### [new 003] CA-Cut: Crop-Aligned Cutout for Data Augmentation to Learn More Robust Under-Canopy Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于农业机器人视觉导航任务，旨在解决复杂农田环境下模型泛化能力不足的问题。作者提出了一种新的数据增强方法CA-Cut，在训练中对作物行附近的区域进行随机遮挡，以提升模型在遮挡情况下的语义关键点预测鲁棒性。实验表明该方法显著减少了预测误差，并增强了跨环境的适应能力。**

- **链接: [http://arxiv.org/pdf/2507.17727v1](http://arxiv.org/pdf/2507.17727v1)**

> **作者:** Robel Mamo; Taeyeong Choi
>
> **备注:** Accepted for publication at the 12th European Conference on Mobile Robots (ECMR 2025)
>
> **摘要:** State-of-the-art visual under-canopy navigation methods are designed with deep learning-based perception models to distinguish traversable space from crop rows. While these models have demonstrated successful performance, they require large amounts of training data to ensure reliability in real-world field deployment. However, data collection is costly, demanding significant human resources for in-field sampling and annotation. To address this challenge, various data augmentation techniques are commonly employed during model training, such as color jittering, Gaussian blur, and horizontal flip, to diversify training data and enhance model robustness. In this paper, we hypothesize that utilizing only these augmentation techniques may lead to suboptimal performance, particularly in complex under-canopy environments with frequent occlusions, debris, and non-uniform spacing of crops. Instead, we propose a novel augmentation method, so-called Crop-Aligned Cutout (CA-Cut) which masks random regions out in input images that are spatially distributed around crop rows on the sides to encourage trained models to capture high-level contextual features even when fine-grained information is obstructed. Our extensive experiments with a public cornfield dataset demonstrate that masking-based augmentations are effective for simulating occlusions and significantly improving robustness in semantic keypoint predictions for visual navigation. In particular, we show that biasing the mask distribution toward crop rows in CA-Cut is critical for enhancing both prediction accuracy and generalizability across diverse environments achieving up to a 36.9% reduction in prediction error. In addition, we conduct ablation studies to determine the number of masks, the size of each mask, and the spatial distribution of masks to maximize overall performance.
>
---
#### [new 004] FAST-Calib: LiDAR-Camera Extrinsic Calibration in One Second
- **分类: cs.RO**

- **简介: 该论文属于传感器标定任务，旨在解决激光雷达（LiDAR）与相机之间的外参标定问题。作者提出FAST-Calib，利用3D靶标和边缘提取算法，实现快速、准确的自动标定，并通过椭圆拟合补偿激光雷达的边缘膨胀效应。实验验证了其在多种LiDAR和相机组合上的有效性与优越性能。**

- **链接: [http://arxiv.org/pdf/2507.17210v1](http://arxiv.org/pdf/2507.17210v1)**

> **作者:** Chunran Zheng; Fu Zhang
>
> **摘要:** This paper proposes FAST-Calib, a fast and user-friendly LiDAR-camera extrinsic calibration tool based on a custom-made 3D target. FAST-Calib supports both mechanical and solid-state LiDARs by leveraging an efficient and reliable edge extraction algorithm that is agnostic to LiDAR scan patterns. It also compensates for edge dilation artifacts caused by LiDAR spot spread through ellipse fitting, and supports joint optimization across multiple scenes. We validate FAST-Calib on three LiDAR models (Ouster, Avia, and Mid360), each paired with a wide-angle camera. Experimental results demonstrate superior accuracy and robustness compared to existing methods. With point-to-point registration errors consistently below 6.5mm and total processing time under 0.7s, FAST-Calib provides an efficient, accurate, and target-based automatic calibration pipeline. We have open-sourced our code and dataset on GitHub to benefit the robotics community.
>
---
#### [new 005] Leveraging multi-source and heterogeneous signals for fatigue detection
- **分类: cs.RO; cs.AI; 62H30; I.2**

- **简介: 该论文属于疲劳检测任务，旨在解决实际场景中因传感器受限而难以有效监测疲劳的问题。论文提出了一种多源异构疲劳检测框架，能自适应利用不同来源的数据，提升了在传感器受限情况下的疲劳检测效果。**

- **链接: [http://arxiv.org/pdf/2507.16859v1](http://arxiv.org/pdf/2507.16859v1)**

> **作者:** Luobin Cui; Yanlai Wu; Tang Ying; Weikai Li
>
> **备注:** 1figures,32pages
>
> **摘要:** Fatigue detection plays a critical role in safety-critical applications such as aviation, mining, and long-haul transport. However, most existing methods rely on high-end sensors and controlled environments, limiting their applicability in real world settings. This paper formally defines a practical yet underexplored problem setting for real world fatigue detection, where systems operating with context-appropriate sensors aim to leverage knowledge from differently instrumented sources including those using impractical sensors deployed in controlled environments. To tackle this challenge, we propose a heterogeneous and multi-source fatigue detection framework that adaptively utilizes the available modalities in the target domain while benefiting from the diverse configurations present in source domains. Our experiments, conducted using a realistic field-deployed sensor setup and two publicly available datasets, demonstrate the practicality, robustness, and improved generalization of our approach, paving the practical way for effective fatigue monitoring in sensor-constrained scenarios.
>
---
#### [new 006] Confidence Calibration in Vision-Language-Action Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究视觉-语言-动作（VLA）模型中的置信度校准问题，旨在提升机器人行为的可信度。通过基准测试、提示集成、时间维度分析和分动作维度校准等方法，探索如何使VLA模型在高任务成功率的同时具备可靠的不确定性估计。**

- **链接: [http://arxiv.org/pdf/2507.17383v1](http://arxiv.org/pdf/2507.17383v1)**

> **作者:** Thomas P Zollo; Richard Zemel
>
> **备注:** 34 pages, 19 figures
>
> **摘要:** Trustworthy robot behavior requires not only high levels of task success but also that the robot can reliably quantify how likely it is to succeed. To this end, we present the first systematic study of confidence calibration in vision-language-action (VLA) foundation models, which map visual observations and natural-language instructions to low-level robot motor commands. We begin with extensive benchmarking to understand the critical relationship between task success and calibration error across multiple datasets and VLA variants, finding that task performance and calibration are not in tension. Next, we introduce prompt ensembles for VLAs, a lightweight, Bayesian-inspired algorithm that averages confidence across paraphrased instructions and consistently improves calibration. We further analyze calibration over the task time horizon, showing that confidence is often most reliable after making some progress, suggesting natural points for risk-aware intervention. Finally, we reveal differential miscalibration across action dimensions and propose action-wise Platt scaling, a method to recalibrate each action dimension independently to produce better confidence estimates. Our aim in this study is to begin to develop the tools and conceptual understanding necessary to render VLAs both highly performant and highly trustworthy via reliable uncertainty quantification.
>
---
#### [new 007] Safety Assurance for Quadrotor Kinodynamic Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于无人机安全控制任务，旨在解决自主无人机在复杂环境中运动规划与控制时的安全保障问题。论文提出了一种结合高层路径规划与低层安全滤波的方法，确保系统在动态环境中的安全运行。**

- **链接: [http://arxiv.org/pdf/2507.17679v1](http://arxiv.org/pdf/2507.17679v1)**

> **作者:** Theodoros Tavoulareas; Marzia Cescon
>
> **备注:** Accepted for publication at 2025 Modeling, Estimation and Control Conference (MECC)
>
> **摘要:** Autonomous drones have gained considerable attention for applications in real-world scenarios, such as search and rescue, inspection, and delivery. As their use becomes ever more pervasive in civilian applications, failure to ensure safe operation can lead to physical damage to the system, environmental pollution, and even loss of human life. Recent work has demonstrated that motion planning techniques effectively generate a collision-free trajectory during navigation. However, these methods, while creating the motion plans, do not inherently consider the safe operational region of the system, leading to potential safety constraints violation during deployment. In this paper, we propose a method that leverages run time safety assurance in a kinodynamic motion planning scheme to satisfy the system's operational constraints. First, we use a sampling-based geometric planner to determine a high-level collision-free path within a user-defined space. Second, we design a low-level safety assurance filter to provide safety guarantees to the control input of a Linear Quadratic Regulator (LQR) designed with the purpose of trajectory tracking. We demonstrate our proposed approach in a restricted 3D simulation environment using a model of the Crazyflie 2.0 drone.
>
---
#### [new 008] MobileUse: A GUI Agent with Hierarchical Reflection for Autonomous Mobile Operation
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于移动智能体任务，旨在解决复杂任务执行中的长时序控制、错误恢复及冷启动问题。提出了MobileUse，通过分层反思架构实现多尺度错误检测与恢复，并引入主动探索模块以增强环境理解，显著提升任务成功率。**

- **链接: [http://arxiv.org/pdf/2507.16853v1](http://arxiv.org/pdf/2507.16853v1)**

> **作者:** Ning Li; Xiangmou Qu; Jiamu Zhou; Jun Wang; Muning Wen; Kounianhua Du; Xingyu Lou; Qiuying Peng; Jun Wang; Weinan Zhang
>
> **备注:** A technical report on a GUI agent based on multi-agent systems
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have enabled the development of mobile agents that can understand visual inputs and follow user instructions, unlocking new possibilities for automating complex tasks on mobile devices. However, applying these models to real-world mobile scenarios remains a significant challenge due to the long-horizon task execution, difficulty in error recovery, and the cold-start problem in unfamiliar environments. To address these challenges, we propose MobileUse, a GUI agent designed for robust and adaptive mobile task execution. To improve resilience in long-horizon tasks and dynamic environments, we introduce a hierarchical reflection architecture that enables the agent to self-monitor, detect, and recover from errors across multiple temporal scales-ranging from individual actions to overall task completion-while maintaining efficiency through a reflection-on-demand strategy. To tackle cold-start issues, we further introduce a proactive exploration module, which enriches the agent's understanding of the environment through self-planned exploration. Evaluations on AndroidWorld and AndroidLab benchmarks demonstrate that MobileUse establishes new state-of-the-art performance, achieving success rates of 62.9% and 44.2%, respectively. To facilitate real-world applications, we release an out-of-the-box toolkit for automated task execution on physical mobile devices, which is available at https://github.com/MadeAgents/mobile-use.
>
---
#### [new 009] Multi-Objective Trajectory Planning for a Robotic Arm in Curtain Wall Installation
- **分类: cs.RO**

- **简介: 该论文属于机器人轨迹规划任务，旨在解决建筑幕墙安装机器人在复杂环境中多目标轨迹优化的问题。作者设计了新型机械臂结构，并提出NSGA-III-FO算法以提升收敛效率，通过实验验证其在多目标优化中的有效性与实用性。**

- **链接: [http://arxiv.org/pdf/2507.17140v1](http://arxiv.org/pdf/2507.17140v1)**

> **作者:** Xiao Liu; Yunxiao Cheng; Weijun Wang; Tianlun Huang; Zhiyong Wang; Wei Feng
>
> **摘要:** In the context of labor shortages and rising costs, construction robots are regarded as the key to revolutionizing traditional construction methods and improving efficiency and quality in the construction industry. In order to ensure that construction robots can perform tasks efficiently and accurately in complex construction environments, traditional single-objective trajectory optimization methods are difficult to meet the complex requirements of the changing construction environment. Therefore, we propose a multi-objective trajectory optimization for the robotic arm used in the curtain wall installation. First, we design a robotic arm for curtain wall installation, integrating serial, parallel, and folding arm elements, while considering its physical properties and motion characteristics. In addition, this paper proposes an NSGA-III-FO algorithm (NSGA-III with Focused Operator, NSGA-III-FO) that incorporates a focus operator screening mechanism to accelerate the convergence of the algorithm towards the Pareto front, thereby effectively balancing the multi-objective constraints of construction robots. The proposed algorithm is tested against NSGA-III, MOEA/D, and MSOPS-II in ten consecutive trials on the DTLZ3 and WFG3 test functions, showing significantly better convergence efficiency than the other algorithms. Finally, we conduct two sets of experiments on the designed robotic arm platform, which confirm the efficiency and practicality of the NSGA-III-FO algorithm in solving multi-objective trajectory planning problems for curtain wall installation tasks.
>
---
#### [new 010] VLA-Touch: Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人感知与控制任务，旨在解决现有视觉-语言-动作模型缺乏触觉反馈导致在接触密集任务中效果不佳的问题。论文提出VLA-Touch，通过引入预训练触觉语言模型和扩散控制器，在不微调原有模型的前提下，融合双层次触觉反馈，提升任务规划效率与操作精度。**

- **链接: [http://arxiv.org/pdf/2507.17294v1](http://arxiv.org/pdf/2507.17294v1)**

> **作者:** Jianxin Bi; Kevin Yuchen Ma; Ce Hao; Mike Zheng Shou; Harold Soh
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** Tactile feedback is generally recognized to be crucial for effective interaction with the physical world. However, state-of-the-art Vision-Language-Action (VLA) models lack the ability to interpret and use tactile signals, limiting their effectiveness in contact-rich tasks. Incorporating tactile feedback into these systems is challenging due to the absence of large multi-modal datasets. We present VLA-Touch, an approach that enhances generalist robot policies with tactile sensing \emph{without fine-tuning} the base VLA. Our method introduces two key innovations: (1) a pipeline that leverages a pretrained tactile-language model that provides semantic tactile feedback for high-level task planning, and (2) a diffusion-based controller that refines VLA-generated actions with tactile signals for contact-rich manipulation. Through real-world experiments, we demonstrate that our dual-level integration of tactile feedback improves task planning efficiency while enhancing execution precision. Code is open-sourced at \href{https://github.com/jxbi1010/VLA-Touch}{this URL}.
>
---
#### [new 011] RAPTAR: Radar Radiation Pattern Acquisition through Automated Collaborative Robotics
- **分类: cs.RO**

- **简介: 该论文属于机器人自动化测量任务，旨在解决传统方法在雷达芯片天线方向图测量中的角度覆盖有限、依赖专用设备和手动校准的问题。论文设计了基于协作机器人的RAPTAR系统，实现无需消声室的三维方向图测量，具备高精度、高分辨率和良好的误差控制，适用于车载、无人机、AR/VR等复杂场景的雷达测试需求。**

- **链接: [http://arxiv.org/pdf/2507.16988v1](http://arxiv.org/pdf/2507.16988v1)**

> **作者:** Maaz Qureshi; Mohammad Omid Bagheri; Abdelrahman Elbadrawy; William Melek; George Shaker
>
> **备注:** 8 Pages, IEEE Journal
>
> **摘要:** Accurate characterization of modern on-chip antennas remains challenging, as current probe-station techniques offer limited angular coverage, rely on bespoke hardware, and require frequent manual alignment. This research introduces RAPTAR (Radiation Pattern Acquisition through Robotic Automation), a portable, state-of-the-art, and autonomous system based on collaborative robotics. RAPTAR enables 3D radiation-pattern measurement of integrated radar modules without dedicated anechoic facilities. The system is designed to address the challenges of testing radar modules mounted in diverse real-world configurations, including vehicles, UAVs, AR/VR headsets, and biomedical devices, where traditional measurement setups are impractical. A 7-degree-of-freedom Franka cobot holds the receiver probe and performs collision-free manipulation across a hemispherical spatial domain, guided by real-time motion planning and calibration accuracy with RMS error below 0.9 mm. The system achieves an angular resolution upto 2.5 degree and integrates seamlessly with RF instrumentation for near- and far-field power measurements. Experimental scans of a 60 GHz radar module show a mean absolute error of less than 2 dB compared to full-wave electromagnetic simulations ground truth. Benchmarking against baseline method demonstrates 36.5% lower mean absolute error, highlighting RAPTAR accuracy and repeatability.
>
---
#### [new 012] The Wilhelm Tell Dataset of Affordance Demonstrations
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于机器人感知任务，旨在解决机器人理解环境可操作性（Affordance）的问题。论文构建了一个包含第一/第三人称视频及元数据的新型数据集，记录人类操作行为，用于训练识别可操作性表现的感知系统。**

- **链接: [http://arxiv.org/pdf/2507.17401v1](http://arxiv.org/pdf/2507.17401v1)**

> **作者:** Rachel Ringe; Mihai Pomarlan; Nikolaos Tsiogkas; Stefano De Giorgis; Maria Hedblom; Rainer Malaka
>
> **备注:** \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Affordances - i.e. possibilities for action that an environment or objects in it provide - are important for robots operating in human environments to perceive. Existing approaches train such capabilities on annotated static images or shapes. This work presents a novel dataset for affordance learning of common household tasks. Unlike previous approaches, our dataset consists of video sequences demonstrating the tasks from first- and third-person perspectives, along with metadata about the affordances that are manifested in the task, and is aimed towards training perception systems to recognize affordance manifestations. The demonstrations were collected from several participants and in total record about seven hours of human activity. The variety of task performances also allows studying preparatory maneuvers that people may perform for a task, such as how they arrange their task space, which is also relevant for collaborative service robots.
>
---
#### [new 013] An Exploratory Study on Human-Robot Interaction using Semantics-based Situational Awareness
- **分类: cs.RO**

- **简介: 该论文研究基于语义的情境感知在人机交互中的应用，旨在提升人机团队在移动机器人任务中的表现。论文通过实验验证高语义信息对操作者工作负荷、情境感知信任度和反应时间的影响，结果表明语义信息能有效提升人机协作效率。**

- **链接: [http://arxiv.org/pdf/2507.17376v1](http://arxiv.org/pdf/2507.17376v1)**

> **作者:** Tianshu Ruan; Aniketh Ramesh; Rustam Stolkin; Manolis Chiou
>
> **摘要:** In this paper, we investigate the impact of high-level semantics (evaluation of the environment) on Human-Robot Teams (HRT) and Human-Robot Interaction (HRI) in the context of mobile robot deployments. Although semantics has been widely researched in AI, how high-level semantics can benefit the HRT paradigm is underexplored, often fuzzy, and intractable. We applied a semantics-based framework that could reveal different indicators of the environment (i.e. how much semantic information exists) in a mock-up disaster response mission. In such missions, semantics are crucial as the HRT should handle complex situations and respond quickly with correct decisions, where humans might have a high workload and stress. Especially when human operators need to shift their attention between robots and other tasks, they will struggle to build Situational Awareness (SA) quickly. The experiment suggests that the presented semantics: 1) alleviate the perceived workload of human operators; 2) increase the operator's trust in the SA; and 3) help to reduce the reaction time in switching the level of autonomy when needed. Additionally, we find that participants with higher trust in the system are encouraged by high-level semantics to use teleoperation mode more.
>
---
#### [new 014] Optimizing Delivery Logistics: Enhancing Speed and Safety with Drone Technology
- **分类: cs.RO**

- **简介: 该论文属于无人机物流任务，旨在解决最后一公里配送的速度与安全问题。作者设计了一种集成AI的无人机配送系统，优化路径规划、目标检测、包裹安全与实时追踪，并评估轻量AI模型与硬件配置，初步验证了系统的效率与合规性。**

- **链接: [http://arxiv.org/pdf/2507.17253v1](http://arxiv.org/pdf/2507.17253v1)**

> **作者:** Maharshi Shastri; Ujjval Shrivastav
>
> **摘要:** The increasing demand for fast and cost effective last mile delivery solutions has catalyzed significant advancements in drone based logistics. This research describes the development of an AI integrated drone delivery system, focusing on route optimization, object detection, secure package handling, and real time tracking. The proposed system leverages YOLOv4 Tiny for object detection, the NEO 6M GPS module for navigation, and the A7670 SIM module for real time communication. A comparative analysis of lightweight AI models and hardware components is conducted to determine the optimal configuration for real time UAV based delivery. Key challenges including battery efficiency, regulatory compliance, and security considerations are addressed through the integration of machine learning techniques, IoT devices, and encryption protocols. Preliminary studies demonstrate improvement in delivery time compared to conventional ground based logistics, along with high accuracy recipient authentication through facial recognition. The study also discusses ethical implications and societal acceptance of drone deliveries, ensuring compliance with FAA, EASA and DGCA regulatory standards. Note: This paper presents the architecture, design, and preliminary simulation results of the proposed system. Experimental results, simulation benchmarks, and deployment statistics are currently being acquired. A comprehensive analysis will be included in the extended version of this work.
>
---
#### [new 015] Event Detection for Active Lower Limb Prosthesis
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在提升下肢假肢的事件检测精度。通过研究十字韧带拉伸在步态周期中的变化，利用双髁膝设计结合LVDT传感器分析数据。实验发现韧带拉伸在特定步态阶段具有速度依赖性和稳定转折点，可用于预测触地和脚掌平放事件，从而优化动力假肢控制器的设计。**

- **链接: [http://arxiv.org/pdf/2507.17649v1](http://arxiv.org/pdf/2507.17649v1)**

> **作者:** J. D. Clark; P. Ellison
>
> **摘要:** Accurate event detection is key to the successful design of semi-passive and powered prosthetics. Kinematically, the natural knee is complex, with translation and rotation components that have a substantial impact on gait characteristics. When simplified to a pin joint, some of this behaviour is lost. This study investigates the role of cruciate ligament stretch in event detection. A bicondylar knee design was used, constrained by analogues of the anterior and posterior cruciate ligaments. This offers the ability to characterize knee kinematics by the stretch of the ligaments. The ligament stretch was recorded using LVDTs parallel to the ligaments of the Russell knee on a bent knee crutch. Which was used to capture data on a treadmill at 3 speeds. This study finds speed dependence within the stretch of the cruciate ligaments, prominently around 5\% and 80\% of the gait cycle for the posterior and anterior. The cycle profile remains consistent with speed; therefore, other static events such as the turning point feature at around 90\% and 95\% of the cycle, for the posterior and anterior, respectively, could be used as a predictive precursor for initial contact. Likewise at 90\% and 95\%, another pair of turning points that in this case could be used to predict foot flat. This concludes that the use of a bicondylar knee design could improve the detection of events during the gait cycle, and therefore could increase the accuracy of subsequent controllers for powered prosthetics.
>
---
#### [new 016] Dynamic Parameter Identification of a Curtain Wall Installation Robotic Arm
- **分类: cs.RO**

- **简介: 该论文旨在提高幕墙安装机器人操作的智能化水平。为解决传统方法效率和质量不足的问题，设计了一种液压驱动的机器人臂及动态参数识别方法。通过建立D-H模型、集成液压缸动力学并采用Stribeck摩擦模型，提出分层渐进参数识别策略，实现高精度动态参数识别，实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.17136v1](http://arxiv.org/pdf/2507.17136v1)**

> **作者:** Xiao Liu; Yunxiao Cheng; Weijun Wang; Tianlun Huang; Wei Feng
>
> **摘要:** In the construction industry, traditional methods fail to meet the modern demands for efficiency and quality. The curtain wall installation is a critical component of construction projects. We design a hydraulically driven robotic arm for curtain wall installation and a dynamic parameter identification method. We establish a Denavit-Hartenberg (D-H) model based on measured robotic arm structural parameters and integrate hydraulic cylinder dynamics to construct a composite parametric system driven by a Stribeck friction model. By designing high-signal-to-noise ratio displacement excitation signals for hydraulic cylinders and combining Fourier series to construct optimal excitation trajectories that satisfy joint constraints, this method effectively excites the characteristics of each parameter in the minimal parameter set of the dynamic model of the robotic arm. On this basis, a hierarchical progressive parameter identification strategy is proposed: least squares estimation is employed to separately identify and jointly calibrate the dynamic parameters of both the hydraulic cylinder and the robotic arm, yielding Stribeck model curves for each joint. Experimental validation on a robotic arm platform demonstrates residual standard deviations below 0.4 Nm between theoretical and measured joint torques, confirming high-precision dynamic parameter identification for the hydraulic-driven curtain wall installation robotic arm. This significantly contributes to enhancing the intelligence level of curtain wall installation operations.
>
---
#### [new 017] MARSCalib: Multi-robot, Automatic, Robust, Spherical Target-based Extrinsic Calibration in Field and Extraterrestrial Environments
- **分类: cs.RO**

- **简介: 该论文属于多机器人系统中的激光雷达-相机外参标定任务，旨在解决户外及外星环境下标定目标和传感器受干扰时的标定鲁棒性问题。论文提出了MARSCalib方法，通过提取图像中的椭圆中心和点云中的球心配对，计算变换矩阵，并引入新算法处理目标退化和传感器噪声，实现了在复杂环境下的鲁棒标定。**

- **链接: [http://arxiv.org/pdf/2507.17130v1](http://arxiv.org/pdf/2507.17130v1)**

> **作者:** Seokhwan Jeong; Hogyun Kim; Younggun Cho
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** This paper presents a novel spherical target-based LiDAR-camera extrinsic calibration method designed for outdoor environments with multi-robot systems, considering both target and sensor corruption. The method extracts the 2D ellipse center from the image and the 3D sphere center from the pointcloud, which are then paired to compute the transformation matrix. Specifically, the image is first decomposed using the Segment Anything Model (SAM). Then, a novel algorithm extracts an ellipse from a potentially corrupted sphere, and the extracted center of ellipse is corrected for errors caused by the perspective projection model. For the LiDAR pointcloud, points on the sphere tend to be highly noisy due to the absence of flat regions. To accurately extract the sphere from these noisy measurements, we apply a hierarchical weighted sum to the accumulated pointcloud. Through experiments, we demonstrated that the sphere can be robustly detected even under both types of corruption, outperforming other targets. We evaluated our method using three different types of LiDARs (spinning, solid-state, and non-repetitive) with cameras positioned in three different locations. Furthermore, we validated the robustness of our method to target corruption by experimenting with spheres subjected to various types of degradation. These experiments were conducted in both a planetary test and a field environment. Our code is available at https://github.com/sparolab/MARSCalib.
>
---
#### [new 018] Summarizing Normative Driving Behavior From Large-Scale NDS Datasets for Vehicle System Development
- **分类: cs.RO**

- **简介: 该论文旨在通过分析大规模自然驾驶数据（NDS），总结规范驾驶行为，用于辅助车辆系统开发。它属于数据分析与行为建模任务，解决如何从海量驾驶数据中提取关键驾驶行为特征的问题。论文处理了SHRP 2 NDS数据，分析五项驾驶指标，并开发可视化工具进行群体比较。**

- **链接: [http://arxiv.org/pdf/2507.16839v1](http://arxiv.org/pdf/2507.16839v1)**

> **作者:** Gregory Beale; Gibran Ali
>
> **备注:** Accepted to the 2025 IEEE International Conference on Intelligent Transportation Systems (ITSC 2025)
>
> **摘要:** This paper presents a methodology to process large-scale naturalistic driving studies (NDS) to describe the driving behavior for five vehicle metrics, including speed, speeding, lane keeping, following distance, and headway, contextualized by roadway characteristics, vehicle classes, and driver demographics. Such descriptions of normative driving behaviors can aid in the development of vehicle safety and intelligent transportation systems. The methodology is demonstrated using data from the Second Strategic Highway Research Program (SHRP 2) NDS, which includes over 34 million miles of driving across more than 3,400 drivers. Summaries of each driving metric were generated using vehicle, GPS, and forward radar data. Additionally, interactive online analytics tools were developed to visualize and compare driving behavior across groups through dynamic data selection and grouping. For example, among drivers on 65-mph roads for the SHRP 2 NDS, females aged 16-19 exceeded the speed limit by 7.5 to 15 mph slightly more often than their male counterparts, and younger drivers maintained headways under 1.5 seconds more frequently than older drivers. This work supports better vehicle systems and safer infrastructure by quantifying normative driving behaviors and offers a methodology for analyzing NDS datasets for cross group comparisons.
>
---
#### [new 019] KernelSOS for Global Sampling-Based Optimal Control and Estimation via Semidefinite Programming
- **分类: cs.RO**

- **简介: 该论文属于优化控制与估计任务，旨在解决存在局部极小值的非凸优化问题。作者提出KernelSOS方法，结合核方法与平方和优化，实现对非多项式、非参数问题的全局采样优化，并可作为局部求解器的有效初始化手段。**

- **链接: [http://arxiv.org/pdf/2507.17572v1](http://arxiv.org/pdf/2507.17572v1)**

> **作者:** Antoine Groudiev; Fabian Schramm; Éloïse Berthier; Justin Carpentier; Frederike Dümbgen
>
> **摘要:** Global optimization has gained attraction over the past decades, thanks to the development of both theoretical foundations and efficient numerical routines to cope with optimization problems of various complexities. Among recent methods, Kernel Sum of Squares (KernelSOS) appears as a powerful framework, leveraging the potential of sum of squares methods from the polynomial optimization community with the expressivity of kernel methods widely used in machine learning. This paper applies the kernel sum of squares framework for solving control and estimation problems, which exhibit poor local minima. We demonstrate that KernelSOS performs well on a selection of problems from both domains. In particular, we show that KernelSOS is competitive with other sum of squares approaches on estimation problems, while being applicable to non-polynomial and non-parametric formulations. The sample-based nature of KernelSOS allows us to apply it to trajectory optimization problems with an integrated simulator treated as a black box, both as a standalone method and as a powerful initialization method for local solvers, facilitating the discovery of better solutions.
>
---
#### [new 020] JAM: Keypoint-Guided Joint Prediction after Classification-Aware Marginal Proposal for Multi-Agent Interaction
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于自动驾驶中的多智能体交互轨迹预测任务。它旨在解决现有方法在生成低概率轨迹模式时质量较低的问题。论文提出了一种两阶段框架JAM，第一阶段通过分类轨迹类型生成多样化的轨迹提议，第二阶段结合场景上下文和关键路径点进行联合预测，从而提升多智能体交互预测的准确性。**

- **链接: [http://arxiv.org/pdf/2507.17152v1](http://arxiv.org/pdf/2507.17152v1)**

> **作者:** Fangze Lin; Ying He; Fei Yu; Hong Zhang
>
> **备注:** IROS 2025 Accepted
>
> **摘要:** Predicting the future motion of road participants is a critical task in autonomous driving. In this work, we address the challenge of low-quality generation of low-probability modes in multi-agent joint prediction. To tackle this issue, we propose a two-stage multi-agent interactive prediction framework named \textit{keypoint-guided joint prediction after classification-aware marginal proposal} (JAM). The first stage is modeled as a marginal prediction process, which classifies queries by trajectory type to encourage the model to learn all categories of trajectories, providing comprehensive mode information for the joint prediction module. The second stage is modeled as a joint prediction process, which takes the scene context and the marginal proposals from the first stage as inputs to learn the final joint distribution. We explicitly introduce key waypoints to guide the joint prediction module in better capturing and leveraging the critical information from the initial predicted trajectories. We conduct extensive experiments on the real-world Waymo Open Motion Dataset interactive prediction benchmark. The results show that our approach achieves competitive performance. In particular, in the framework comparison experiments, the proposed JAM outperforms other prediction frameworks and achieves state-of-the-art performance in interactive trajectory prediction. The code is available at https://github.com/LinFunster/JAM to facilitate future research.
>
---
#### [new 021] IndoorBEV: Joint Detection and Footprint Completion of Objects via Mask-based Prediction in Indoor Scenarios for Bird's-Eye View Perception
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于室内场景下的机器人感知任务，旨在解决复杂室内环境中多样物体检测与轮廓补全问题。传统方法在处理不规则形状、遮挡和动静态物体共存时表现不足。论文提出IndoorBEV方法，基于鸟瞰图（BEV）和掩码预测，联合检测物体并补全其轮廓，提升室内场景理解能力。**

- **链接: [http://arxiv.org/pdf/2507.17445v1](http://arxiv.org/pdf/2507.17445v1)**

> **作者:** Haichuan Li; Changda Tian; Panos Trahanias; Tomi Westerlund
>
> **摘要:** Detecting diverse objects within complex indoor 3D point clouds presents significant challenges for robotic perception, particularly with varied object shapes, clutter, and the co-existence of static and dynamic elements where traditional bounding box methods falter. To address these limitations, we propose IndoorBEV, a novel mask-based Bird's-Eye View (BEV) method for indoor mobile robots. In a BEV method, a 3D scene is projected into a 2D BEV grid which handles naturally occlusions and provides a consistent top-down view aiding to distinguish static obstacles from dynamic agents. The obtained 2D BEV results is directly usable to downstream robotic tasks like navigation, motion prediction, and planning. Our architecture utilizes an axis compact encoder and a window-based backbone to extract rich spatial features from this BEV map. A query-based decoder head then employs learned object queries to concurrently predict object classes and instance masks in the BEV space. This mask-centric formulation effectively captures the footprint of both static and dynamic objects regardless of their shape, offering a robust alternative to bounding box regression. We demonstrate the effectiveness of IndoorBEV on a custom indoor dataset featuring diverse object classes including static objects and dynamic elements like robots and miscellaneous items, showcasing its potential for robust indoor scene understanding.
>
---
#### [new 022] Shared Control of Holonomic Wheelchairs through Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于智能轮椅共享控制任务，旨在解决全向轮椅导航中用户操作不直观、安全性低的问题。作者提出一种基于强化学习的方法，将2D用户输入转化为33D运动，在保证安全的同时提升操作舒适性与顺滑度，并实现了真实场景中的应用验证。**

- **链接: [http://arxiv.org/pdf/2507.17055v1](http://arxiv.org/pdf/2507.17055v1)**

> **作者:** Jannis Bähler; Diego Paez-Granados; Jorge Peña-Queralta
>
> **摘要:** Smart electric wheelchairs can improve user experience by supporting the driver with shared control. State-of-the-art work showed the potential of shared control in improving safety in navigation for non-holonomic robots. However, for holonomic systems, current approaches often lead to unintuitive behavior for the user and fail to utilize the full potential of omnidirectional driving. Therefore, we propose a reinforcement learning-based method, which takes a 2D user input and outputs a 3D motion while ensuring user comfort and reducing cognitive load on the driver. Our approach is trained in Isaac Gym and tested in simulation in Gazebo. We compare different RL agent architectures and reward functions based on metrics considering cognitive load and user comfort. We show that our method ensures collision-free navigation while smartly orienting the wheelchair and showing better or competitive smoothness compared to a previous non-learning-based method. We further perform a sim-to-real transfer and demonstrate, to the best of our knowledge, the first real-world implementation of RL-based shared control for an omnidirectional mobility platform.
>
---
#### [new 023] Prolonging Tool Life: Learning Skillful Use of General-purpose Tools through Lifespan-guided Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人工具使用任务，旨在解决通用工具在不确定环境中使用时寿命短的问题。通过引入考虑工具寿命的强化学习框架，结合有限元分析与Miner法则估计剩余寿命，并采用自适应奖励归一化机制优化策略学习，实现了工具寿命的有效延长。**

- **链接: [http://arxiv.org/pdf/2507.17275v1](http://arxiv.org/pdf/2507.17275v1)**

> **作者:** Po-Yen Wu; Cheng-Yu Kuo; Yuki Kadokawa; Takamitsu Matsubara
>
> **备注:** Under review
>
> **摘要:** In inaccessible environments with uncertain task demands, robots often rely on general-purpose tools that lack predefined usage strategies. These tools are not tailored for particular operations, making their longevity highly sensitive to how they are used. This creates a fundamental challenge: how can a robot learn a tool-use policy that both completes the task and prolongs the tool's lifespan? In this work, we address this challenge by introducing a reinforcement learning (RL) framework that incorporates tool lifespan as a factor during policy optimization. Our framework leverages Finite Element Analysis (FEA) and Miner's Rule to estimate Remaining Useful Life (RUL) based on accumulated stress, and integrates the RUL into the RL reward to guide policy learning toward lifespan-guided behavior. To handle the fact that RUL can only be estimated after task execution, we introduce an Adaptive Reward Normalization (ARN) mechanism that dynamically adjusts reward scaling based on estimated RULs, ensuring stable learning signals. We validate our method across simulated and real-world tool use tasks, including Object-Moving and Door-Opening with multiple general-purpose tools. The learned policies consistently prolong tool lifespan (up to 8.01x in simulation) and transfer effectively to real-world settings, demonstrating the practical value of learning lifespan-guided tool use strategies.
>
---
#### [new 024] Multi-agent Reinforcement Learning for Robotized Coral Reef Sample Collection
- **分类: cs.RO**

- **简介: 论文任务为自主水下机器人珊瑚采样控制。解决问题：实现高效珊瑚礁样本采集。工作：构建基于数字孪生的强化学习环境，结合仿真与真实实验，采用游戏引擎、深度强化学习和实时水下动作捕捉实现零样本迁移策略。**

- **链接: [http://arxiv.org/pdf/2507.16941v1](http://arxiv.org/pdf/2507.16941v1)**

> **作者:** Daniel Correa; Tero Kaarlela; Jose Fuentes; Paulo Padrao; Alain Duran; Leonardo Bobadilla
>
> **摘要:** This paper presents a reinforcement learning (RL) environment for developing an autonomous underwater robotic coral sampling agent, a crucial coral reef conservation and research task. Using software-in-the-loop (SIL) and hardware-in-the-loop (HIL), an RL-trained artificial intelligence (AI) controller is developed using a digital twin (DT) in simulation and subsequently verified in physical experiments. An underwater motion capture (MOCAP) system provides real-time 3D position and orientation feedback during verification testing for precise synchronization between the digital and physical domains. A key novelty of this approach is the combined use of a general-purpose game engine for simulation, deep RL, and real-time underwater motion capture for an effective zero-shot sim-to-real strategy.
>
---
#### [new 025] Deformable Cluster Manipulation via Whole-Arm Policy Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人操控任务，旨在解决多自由度机械臂对可变形物体集群进行全臂操作的问题。论文提出了一种结合3D点云和本体感知触觉信息的无模型策略学习框架，采用分布状态表示与核均值嵌入提升训练效率，并设计了适用于清除遮挡的通用启发策略，实现了从仿真到真实环境的零样本迁移。**

- **链接: [http://arxiv.org/pdf/2507.17085v1](http://arxiv.org/pdf/2507.17085v1)**

> **作者:** Jayadeep Jacob; Wenzheng Zhang; Houston Warren; Paulo Borges; Tirthankar Bandyopadhyay; Fabio Ramos
>
> **摘要:** Manipulating clusters of deformable objects presents a substantial challenge with widespread applicability, but requires contact-rich whole-arm interactions. A potential solution must address the limited capacity for realistic model synthesis, high uncertainty in perception, and the lack of efficient spatial abstractions, among others. We propose a novel framework for learning model-free policies integrating two modalities: 3D point clouds and proprioceptive touch indicators, emphasising manipulation with full body contact awareness, going beyond traditional end-effector modes. Our reinforcement learning framework leverages a distributional state representation, aided by kernel mean embeddings, to achieve improved training efficiency and real-time inference. Furthermore, we propose a novel context-agnostic occlusion heuristic to clear deformables from a target region for exposure tasks. We deploy the framework in a power line clearance scenario and observe that the agent generates creative strategies leveraging multiple arm links for de-occlusion. Finally, we perform zero-shot sim-to-real policy transfer, allowing the arm to clear real branches with unknown occlusion patterns, unseen topology, and uncertain dynamics.
>
---
#### [new 026] Robot-mediated physical Human-Human Interaction in Neurorehabilitation: a position paper
- **分类: cs.RO**

- **简介: 该论文属于康复机器人研究任务，旨在解决神经康复中机器人系统缺乏治疗师临床经验与适应性的问题。作者提出“机器人辅助的人与人物理互动”新框架，结合机器人精准性与治疗师专业判断，整合多学科方法，推动康复机器人更自然、高效地辅助治疗。**

- **链接: [http://arxiv.org/pdf/2507.17561v1](http://arxiv.org/pdf/2507.17561v1)**

> **作者:** Lorenzo Vianello; Matthew Short; Julia Manczurowsky; Emek Barış Küçüktabak; Francesco Di Tommaso; Alessia Noccaro; Laura Bandini; Shoshana Clark; Alaina Fiorenza; Francesca Lunardini; Alberto Canton; Marta Gandolla; Alessandra L. G. Pedrocchi; Emilia Ambrosini; Manuel Murie-Fernandez; Carmen B. Roman; Jesus Tornero; Natacha Leon; Andrew Sawers; Jim Patton; Domenico Formica; Nevio Luigi Tagliamonte; Georg Rauter; Kilian Baur; Fabian Just; Christopher J. Hasson; Vesna D. Novak; Jose L. Pons
>
> **摘要:** Neurorehabilitation conventionally relies on the interaction between a patient and a physical therapist. Robotic systems can improve and enrich the physical feedback provided to patients after neurological injury, but they under-utilize the adaptability and clinical expertise of trained therapists. In this position paper, we advocate for a novel approach that integrates the therapist's clinical expertise and nuanced decision-making with the strength, accuracy, and repeatability of robotics: Robot-mediated physical Human-Human Interaction. This framework, which enables two individuals to physically interact through robotic devices, has been studied across diverse research groups and has recently emerged as a promising link between conventional manual therapy and rehabilitation robotics, harmonizing the strengths of both approaches. This paper presents the rationale of a multidisciplinary team-including engineers, doctors, and physical therapists-for conducting research that utilizes: a unified taxonomy to describe robot-mediated rehabilitation, a framework of interaction based on social psychology, and a technological approach that makes robotic systems seamless facilitators of natural human-human interaction.
>
---
#### [new 027] Reconfigurable Tendon-Driven Robots: Eliminating Inter-segmental Coupling via Independently Lockable Joints
- **分类: cs.RO**

- **简介: 论文提出了一种可重构肌腱驱动机器人（RTR），通过可独立锁定的关节消除传统肌腱驱动机器人（TDR）中因多段结构引起的段间耦合问题。该任务旨在提升TDR的控制精度与灵活性。研究设计了无需持续供电即可锁定的关节结构，实现了选择性驱动，简化了控制模型。通过仿真与实验验证了RTR在复杂环境中具有更好的可重构性与运动能力。**

- **链接: [http://arxiv.org/pdf/2507.17163v1](http://arxiv.org/pdf/2507.17163v1)**

> **作者:** Botao Lin; Shuang Song; Jiaole Wang
>
> **摘要:** With a slender redundant body, the tendon-driven robot (TDR) has a large workspace and great maneuverability while working in complex environments. TDR comprises multiple independently controlled robot segments, each with a set of driving tendons. While increasing the number of robot segments enhances dexterity and expands the workspace, this structural expansion also introduces intensified inter-segmental coupling. Therefore, achieving precise TDR control requires more complex models and additional motors. This paper presents a reconfigurable tendon-driven robot (RTR) equipped with innovative lockable joints. Each joint's state (locked/free) can be individually controlled through a pair of antagonistic tendons, and its structure eliminates the need for a continuous power supply to maintain the state. Operators can selectively actuate the targeted robot segments, and this scheme fundamentally eliminates the inter-segmental coupling, thereby avoiding the requirement for complex coordinated control between segments. The workspace of RTR has been simulated and compared with traditional TDRs' workspace, and RTR's advantages are further revealed. The kinematics and statics models of the RTR have been derived and validation experiments have been conducted. Demonstrations have been performed using a seven-joint RTR prototype to show its reconfigurability and moving ability in complex environments with an actuator pack comprising only six motors.
>
---
#### [new 028] Language-Conditioned Open-Vocabulary Mobile Manipulation with Pretrained Models
- **分类: cs.RO**

- **简介: 该论文属于移动操作任务，旨在解决开放词汇条件下机器人在家庭环境中操作新物体的问题。作者提出了LOVMM框架，结合大语言模型和视觉语言模型，实现根据自然语言指令完成多任务操作，表现出良好的零样本泛化能力和多任务学习效果。**

- **链接: [http://arxiv.org/pdf/2507.17379v1](http://arxiv.org/pdf/2507.17379v1)**

> **作者:** Shen Tan; Dong Zhou; Xiangyu Shao; Junqiao Wang; Guanghui Sun
>
> **备注:** IJCAI 2025
>
> **摘要:** Open-vocabulary mobile manipulation (OVMM) that involves the handling of novel and unseen objects across different workspaces remains a significant challenge for real-world robotic applications. In this paper, we propose a novel Language-conditioned Open-Vocabulary Mobile Manipulation framework, named LOVMM, incorporating the large language model (LLM) and vision-language model (VLM) to tackle various mobile manipulation tasks in household environments. Our approach is capable of solving various OVMM tasks with free-form natural language instructions (e.g. "toss the food boxes on the office room desk to the trash bin in the corner", and "pack the bottles from the bed to the box in the guestroom"). Extensive experiments simulated in complex household environments show strong zero-shot generalization and multi-task learning abilities of LOVMM. Moreover, our approach can also generalize to multiple tabletop manipulation tasks and achieve better success rates compared to other state-of-the-art methods.
>
---
#### [new 029] Towards Human-level Intelligence via Human-like Whole-Body Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文旨在通过模仿人类行为实现通用机器人智能，聚焦于类人全身操作任务。论文提出Astribot Suite框架，解决类人机器人在硬件设计、全身遥操作接口及学习算法三方面的核心挑战，推动机器人具备人类级别的灵活性和智能，完成多样环境中的日常任务。**

- **链接: [http://arxiv.org/pdf/2507.17141v1](http://arxiv.org/pdf/2507.17141v1)**

> **作者:** Guang Gao; Jianan Wang; Jinbo Zuo; Junnan Jiang; Jingfan Zhang; Xianwen Zeng; Yuejiang Zhu; Lianyang Ma; Ke Chen; Minhua Sheng; Ruirui Zhang; Zhaohui An
>
> **摘要:** Building general-purpose intelligent robots has long been a fundamental goal of robotics. A promising approach is to mirror the evolutionary trajectory of humans: learning through continuous interaction with the environment, with early progress driven by the imitation of human behaviors. Achieving this goal presents three core challenges: (1) designing safe robotic hardware with human-level physical capabilities; (2) developing an intuitive and scalable whole-body teleoperation interface for data collection; and (3) creating algorithms capable of learning whole-body visuomotor policies from human demonstrations. To address these challenges in a unified framework, we propose Astribot Suite, a robot learning suite for whole-body manipulation aimed at general daily tasks across diverse environments. We demonstrate the effectiveness of our system on a wide range of activities that require whole-body coordination, extensive reachability, human-level dexterity, and agility. Our results show that Astribot's cohesive integration of embodiment, teleoperation interface, and learning pipeline marks a significant step towards real-world, general-purpose whole-body robotic manipulation, laying the groundwork for the next generation of intelligent robots.
>
---
#### [new 030] Sensor-Space Based Robust Kinematic Control of Redundant Soft Manipulator by Learning
- **分类: cs.RO**

- **简介: 该论文属于软体机器人控制任务，旨在解决冗余软体机械臂在未知外力和受限环境中的运动控制难题。论文提出了基于传感器空间的模仿学习控制框架（SS-ILKC），结合强化学习与对抗模仿学习，实现对执行器饱和和环境约束的鲁棒控制，并通过仿真到现实的迁移机制实现零样本部署。**

- **链接: [http://arxiv.org/pdf/2507.16842v1](http://arxiv.org/pdf/2507.16842v1)**

> **作者:** Yinan Meng; Kun Qian; Jiong Yang; Renbo Su; Zhenhong Li; Charlie C. L. Wang
>
> **摘要:** The intrinsic compliance and high degree of freedom (DoF) of redundant soft manipulators facilitate safe interaction and flexible task execution. However, effective kinematic control remains highly challenging, as it must handle deformations caused by unknown external loads and avoid actuator saturation due to improper null-space regulation - particularly in confined environments. In this paper, we propose a Sensor-Space Imitation Learning Kinematic Control (SS-ILKC) framework to enable robust kinematic control under actuator saturation and restrictive environmental constraints. We employ a dual-learning strategy: a multi-goal sensor-space control framework based on reinforcement learning principle is trained in simulation to develop robust control policies for open spaces, while a generative adversarial imitation learning approach enables effective policy learning from sparse expert demonstrations for confined spaces. To enable zero-shot real-world deployment, a pre-processed sim-to-real transfer mechanism is proposed to mitigate the simulation-to-reality gap and accurately characterize actuator saturation limits. Experimental results demonstrate that our method can effectively control a pneumatically actuated soft manipulator, achieving precise path-following and object manipulation in confined environments under unknown loading conditions.
>
---
#### [new 031] AquaChat: An LLM-Guided ROV Framework for Adaptive Inspection of Aquaculture Net Pens
- **分类: cs.RO**

- **简介: 该论文属于水下机器人任务规划与控制领域，旨在解决传统水产养殖网箱检测中适应性差的问题。作者提出了AquaChat框架，结合大语言模型与ROV技术，实现自然语言指令解析、任务规划与自适应控制，提升检测灵活性与效率。**

- **链接: [http://arxiv.org/pdf/2507.16841v1](http://arxiv.org/pdf/2507.16841v1)**

> **作者:** Waseem Akram; Muhayy Ud Din; Abdelhaleem Saad; Irfan Hussain
>
> **摘要:** Inspection of aquaculture net pens is essential for maintaining the structural integrity, biosecurity, and operational efficiency of fish farming systems. Traditional inspection approaches rely on pre-programmed missions or manual control, offering limited adaptability to dynamic underwater conditions and user-specific demands. In this study, we propose AquaChat, a novel Remotely Operated Vehicle (ROV) framework that integrates Large Language Models (LLMs) for intelligent and adaptive net pen inspection. The system features a multi-layered architecture: (1) a high-level planning layer that interprets natural language user commands using an LLM to generate symbolic task plans; (2) a mid-level task manager that translates plans into ROV control sequences; and (3) a low-level motion control layer that executes navigation and inspection tasks with precision. Real-time feedback and event-triggered replanning enhance robustness in challenging aquaculture environments. The framework is validated through experiments in both simulated and controlled aquatic environments representative of aquaculture net pens. Results demonstrate improved task flexibility, inspection accuracy, and operational efficiency. AquaChat illustrates the potential of integrating language-based AI with marine robotics to enable intelligent, user-interactive inspection systems for sustainable aquaculture operations.
>
---
#### [new 032] Mobile Manipulation with Active Inference for Long-Horizon Rearrangement Tasks
- **分类: cs.RO**

- **简介: 该论文属于移动操作任务，旨在解决复杂、长视野任务中主动推理的应用问题。论文提出了一种分层的主动推理架构，结合高阶主动推理模型与全身主动推理控制器，实现技能组合、在线适应与任务失败恢复。方法在Habitat基准测试中表现优于现有技术。**

- **链接: [http://arxiv.org/pdf/2507.17338v1](http://arxiv.org/pdf/2507.17338v1)**

> **作者:** Corrado Pezzato; Ozan Çatal; Toon Van de Maele; Riddhi J. Pitliya; Tim Verbelen
>
> **摘要:** Despite growing interest in active inference for robotic control, its application to complex, long-horizon tasks remains untested. We address this gap by introducing a fully hierarchical active inference architecture for goal-directed behavior in realistic robotic settings. Our model combines a high-level active inference model that selects among discrete skills realized via a whole-body active inference controller. This unified approach enables flexible skill composition, online adaptability, and recovery from task failures without requiring offline training. Evaluated on the Habitat Benchmark for mobile manipulation, our method outperforms state-of-the-art baselines across the three long-horizon tasks, demonstrating for the first time that active inference can scale to the complexity of modern robotics benchmarks.
>
---
#### [new 033] HuNavSim 2.0
- **分类: cs.RO**

- **简介: 论文介绍了HuNavSim 2.0，一个用于模拟人类与机器人导航行为的开源工具。该工具基于ROS 2开发，支持与Gazebo或NVidia Isaac Sim等仿真平台集成，旨在促进人类感知机器人导航系统的研究。通过扩展行为树中的动作和条件集合，实现了更复杂和真实的人类行为模拟。**

- **链接: [http://arxiv.org/pdf/2507.17317v1](http://arxiv.org/pdf/2507.17317v1)**

> **作者:** Miguel Escudero-Jiménez; Noé Pérez-Higueras; Andrés Martínez-Silva; Fernando Caballero; Luis Merino
>
> **备注:** Preprint submitted to the 8th Iberian Robotics Conference (ROBOT 2025)
>
> **摘要:** This work presents a new iteration of the Human Navigation Simulator (HuNavSim), a novel open-source tool for the simulation of different human-agent navigation behaviors in scenarios with mobile robots. The tool, programmed under the ROS 2 framework, can be used together with different well-known robotics simulators such as Gazebo or NVidia Isaac Sim. The main goal is to facilitate the development and evaluation of human-aware robot navigation systems in simulation. In this new version, several features have been improved and new ones added, such as the extended set of actions and conditions that can be combined in Behavior Trees to compound complex and realistic human behaviors.
>
---
#### [new 034] When and Where Localization Fails: An Analysis of the Iterative Closest Point in Evolving Environment
- **分类: cs.RO**

- **简介: 该论文属于定位任务，旨在解决动态户外环境中基于激光雷达的短期重定位问题。作者构建了一个高分辨率、多时相数据集，并采用两种ICP算法评估定位鲁棒性，发现点到面对齐更稳定准确，尤其适用于特征稀疏或植被密集区域。**

- **链接: [http://arxiv.org/pdf/2507.17531v1](http://arxiv.org/pdf/2507.17531v1)**

> **作者:** Abdel-Raouf Dannaoui; Johann Laconte; Christophe Debain; Francois Pomerleau; Paul Checchin
>
> **备注:** 7 pages, 7 figures, proceedings in European Conference on Mobile Robots (ECMR) 2025
>
> **摘要:** Robust relocalization in dynamic outdoor environments remains a key challenge for autonomous systems relying on 3D lidar. While long-term localization has been widely studied, short-term environmental changes, occurring over days or weeks, remain underexplored despite their practical significance. To address this gap, we present a highresolution, short-term multi-temporal dataset collected weekly from February to April 2025 across natural and semi-urban settings. Each session includes high-density point cloud maps, 360 deg panoramic images, and trajectory data. Projected lidar scans, derived from the point cloud maps and modeled with sensor-accurate occlusions, are used to evaluate alignment accuracy against the ground truth using two Iterative Closest Point (ICP) variants: Point-to-Point and Point-to-Plane. Results show that Point-to-Plane offers significantly more stable and accurate registration, particularly in areas with sparse features or dense vegetation. This study provides a structured dataset for evaluating short-term localization robustness, a reproducible framework for analyzing scan-to-map alignment under noise, and a comparative evaluation of ICP performance in evolving outdoor environments. Our analysis underscores how local geometry and environmental variability affect localization success, offering insights for designing more resilient robotic systems.
>
---
#### [new 035] ResKACNNet: A Residual ChebyKAN Network for Inertial Odometry
- **分类: cs.RO**

- **简介: 该论文属于惯性里程计任务，旨在解决传统CNN方法难以捕捉IMU数据非线性运动特征和长期依赖的问题。作者提出ResChebyKAN网络，结合Chebyshev多项式和EKSA注意力模块，提升定位精度，并在多个数据集上验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.16865v1](http://arxiv.org/pdf/2507.16865v1)**

> **作者:** Shanshan Zhang; Tianshui Wen; Siyue Wang; Qi Zhang; Ziheng Zhou; Huiru Zheng; Lingxiang Zheng; Yu Yang
>
> **摘要:** Inertial Measurement Unit (IMU) has become a key technology for achieving low-cost and precise positioning. However, traditional CNN-based inertial positioning methods struggle to capture the nonlinear motion characteristics and long-term dependencies in IMU data. To address this limitation, we propose a novel inertial positioning network with a generic backbone called ResChebyKAN, which leverages the nonlinear approximation capabilities of Chebyshev polynomials to model complex motion patterns. Additionally, we introduce an Efficient Kernel-based Self-Attention (EKSA) module to effectively capture contextual information and enhance long-term dependency modeling. Experimental results on public datasets (e.g., RIDI, RoNIN, RNIN-VIO, OxIOD, IMUNet, and TLIO) demonstrate that our method reduces the absolute trajectory error by 3.79% to 42.32% compared to existing benchmark methods. Furthermore, we release a preprocessed dataset and empirically show that removing the gravity component from acceleration data significantly improves inertial positioning performance.
>
---
#### [new 036] Falconry-like palm landing by a flapping-wing drone based on the human gesture interaction and distance-aware flight planning
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决如何实现扑翼无人机与人类安全接触交互的问题。受猎鹰训练启发，作者设计了一种基于手势交互和距离感知飞行规划的掌心降落系统，首次实现了扑翼无人机与人类的接触式互动。**

- **链接: [http://arxiv.org/pdf/2507.17144v1](http://arxiv.org/pdf/2507.17144v1)**

> **作者:** Kazuki Numazato; Keiichiro Kan; Masaki Kitagawa; Yunong Li; Johannes Kubel; Moju Zhao
>
> **备注:** 8 pages, 14 figures
>
> **摘要:** Flapping-wing drones have attracted significant attention due to their biomimetic flight. They are considered more human-friendly due to their characteristics such as low noise and flexible wings, making them suitable for human-drone interactions. However, few studies have explored the practical interaction between humans and flapping-wing drones. On establishing a physical interaction system with flapping-wing drones, we can acquire inspirations from falconers who guide birds of prey to land on their arms. This interaction interprets the human body as a dynamic landing platform, which can be utilized in various scenarios such as crowded or spatially constrained environments. Thus, in this study, we propose a falconry-like interaction system in which a flapping-wing drone performs a palm landing motion on a human hand. To achieve a safe approach toward humans, we design a trajectory planning method that considers both physical and psychological factors of the human safety such as the drone's velocity and distance from the user. We use a commercial flapping platform with our implemented motion planning and conduct experiments to evaluate the palm landing performance and safety. The results demonstrate that our approach enables safe and smooth hand landing interactions. To the best of our knowledge, it is the first time to achieve a contact-based interaction between flapping-wing drones and humans.
>
---
#### [new 037] Terrain-Aware Adaptation for Two-Dimensional UAV Path Planners
- **分类: cs.RO**

- **简介: 该论文属于无人机路径规划任务，旨在解决现有二维路径规划算法在三维地形中重建效果差的问题。作者提出了一种地形感知的扩展算法DARP-3D，可调整无人机高度和相机角度，提升三维重建质量。**

- **链接: [http://arxiv.org/pdf/2507.17519v1](http://arxiv.org/pdf/2507.17519v1)**

> **作者:** Kostas Karakontis; Thanos Petsanis; Athanasios Ch. Kapoutsis; Pavlos Ch. Kapoutsis; Elias B. Kosmatopoulos
>
> **摘要:** Multi-UAV Coverage Path Planning (mCPP) algorithms in popular commercial software typically treat a Region of Interest (RoI) only as a 2D plane, ignoring important3D structure characteristics. This leads to incomplete 3Dreconstructions, especially around occluded or vertical surfaces. In this paper, we propose a modular algorithm that can extend commercial two-dimensional path planners to facilitate terrain-aware planning by adjusting altitude and camera orientations. To demonstrate it, we extend the well-known DARP (Divide Areas for Optimal Multi-Robot Coverage Path Planning) algorithm and produce DARP-3D. We present simulation results in multiple 3D environments and a real-world flight test using DJI hardware. Compared to baseline, our approach consistently captures improved 3D reconstructions, particularly in areas with significant vertical features. An open-source implementation of the algorithm is available here:https://github.com/konskara/TerraPlan
>
---
#### [new 038] Dynamic Modeling and Dimensional Optimization of Legged Mechanisms for Construction Robot
- **分类: cs.RO**

- **简介: 该论文属于机器人结构设计与优化任务，旨在解决施工机器人能耗高、负载能力弱和动态性能不足的问题。工作包括仿生腿结构设计、动力学建模、参数优化及仿真验证，有效降低了关节驱动力和能耗。**

- **链接: [http://arxiv.org/pdf/2507.17132v1](http://arxiv.org/pdf/2507.17132v1)**

> **作者:** Xiao Liu; Xianlong Yang; Weijun Wang; Wei Feng
>
> **摘要:** With the rapid development of the construction industry, issues such as harsh working environments, high-intensity and high-risk tasks, and labor shortages have become increasingly prominent. This drives higher demands for construction robots in terms of low energy consumption, high mobility, and high load capacity. This paper focuses on the design and optimization of leg structures for construction robots, aiming to improve their dynamic performance, reduce energy consumption, and enhance load-bearing capabilities. Firstly, based on the leg configuration of ants in nature, we design a structure for the robot's leg. Secondly, we propose a novel structural optimization method. Using the Lagrangian approach, a dynamic model of the leg was established. Combining the dynamic model with the leg's motion trajectory, we formulated multiple dynamic evaluation metrics and conducted a comprehensive optimization study on the geometric parameters of each leg segment. The results show that the optimized leg structure reduces peak joint torques and energy consumption by over 20%. Finally, dynamic simulation experiments were conducted using ADAMS. The results demonstrate a significant reduction in the driving power of each joint after optimization, validating the effectiveness and rationality of the proposed strategy. This study provides a theoretical foundation and technical support for the design of heavy-load, high-performance construction robots.
>
---
#### [new 039] Generalized Advantage Estimation for Distributional Policy Gradients
- **分类: cs.LG; cs.RO**

- **简介: 论文属于强化学习任务，旨在解决传统GAE无法处理分布强化学习中的价值分布问题。作者提出分布GAE（DGAE），结合最优传输理论和指数加权估计，有效衡量分布间的距离与方向差异，降低策略梯度估计的方差，并将其集成到三种策略梯度方法中进行实验验证。**

- **链接: [http://arxiv.org/pdf/2507.17530v1](http://arxiv.org/pdf/2507.17530v1)**

> **作者:** Shahil Shaik; Jonathon M. Smereka; Yue Wang
>
> **备注:** 6 pages, 3 figures, published at ACC 2025 Conference
>
> **摘要:** Generalized Advantage Estimation (GAE) has been used to mitigate the computational complexity of reinforcement learning (RL) by employing an exponentially weighted estimation of the advantage function to reduce the variance in policy gradient estimates. Despite its effectiveness, GAE is not designed to handle value distributions integral to distributional RL, which can capture the inherent stochasticity in systems and is hence more robust to system noises. To address this gap, we propose a novel approach that utilizes the optimal transport theory to introduce a Wasserstein-like directional metric, which measures both the distance and the directional discrepancies between probability distributions. Using the exponentially weighted estimation, we leverage this Wasserstein-like directional metric to derive distributional GAE (DGAE). Similar to traditional GAE, our proposed DGAE provides a low-variance advantage estimate with controlled bias, making it well-suited for policy gradient algorithms that rely on advantage estimation for policy updates. We integrated DGAE into three different policy gradient methods. Algorithms were evaluated across various OpenAI Gym environments and compared with the baselines with traditional GAE to assess the performance.
>
---
#### [new 040] From Scan to Action: Leveraging Realistic Scans for Embodied Scene Understanding
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于三维场景理解任务，旨在解决真实世界3D扫描数据因格式多样、工具不兼容等问题难以被有效利用的挑战。论文提出基于USD的统一标注整合方法，并通过LLM场景编辑与机器人仿真应用验证有效性，成功提升数据可用性与应用泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.17585v1](http://arxiv.org/pdf/2507.17585v1)**

> **作者:** Anna-Maria Halacheva; Jan-Nico Zaech; Sombit Dey; Luc Van Gool; Danda Pani Paudel
>
> **备注:** Accepted at the OpenSUN3D Workshop, CVPR 2025. This workshop paper is not included in the official CVPR proceedings
>
> **摘要:** Real-world 3D scene-level scans offer realism and can enable better real-world generalizability for downstream applications. However, challenges such as data volume, diverse annotation formats, and tool compatibility limit their use. This paper demonstrates a methodology to effectively leverage these scans and their annotations. We propose a unified annotation integration using USD, with application-specific USD flavors. We identify challenges in utilizing holistic real-world scan datasets and present mitigation strategies. The efficacy of our approach is demonstrated through two downstream applications: LLM-based scene editing, enabling effective LLM understanding and adaptation of the data (80% success), and robotic simulation, achieving an 87% success rate in policy learning.
>
---
#### [new 041] Budget Allocation Policies for Real-Time Multi-Agent Path Finding
- **分类: cs.MA; cs.AI; cs.RO**

- **简介: 该论文属于多智能体路径规划任务，旨在解决实时环境下多智能体路径冲突问题。现有方法未充分考虑规划预算分配，论文提出多种预算分配策略，并在PrP和MAPF-LNS2算法中验证。结果表明，相比共享预算方式，按代理分配预算能更高效解决拥堵场景，降低完成时间。**

- **链接: [http://arxiv.org/pdf/2507.16874v1](http://arxiv.org/pdf/2507.16874v1)**

> **作者:** Raz Beck; Roni Stern
>
> **备注:** 8 pages, 2 figures, 3 tables
>
> **摘要:** Multi-Agent Pathfinding (MAPF) is the problem of finding paths for a set of agents such that each agent reaches its desired destination while avoiding collisions with the other agents. Many MAPF solvers are designed to run offline, that is, first generate paths for all agents and then execute them. Real-Time MAPF (RT-MAPF) embodies a realistic MAPF setup in which one cannot wait until a complete path for each agent has been found before they start to move. Instead, planning and execution are interleaved, where the agents must commit to a fixed number of steps in a constant amount of computation time, referred to as the planning budget. Existing solutions to RT-MAPF iteratively call windowed versions of MAPF algorithms in every planning period, without explicitly considering the size of the planning budget. We address this gap and explore different policies for allocating the planning budget in windowed versions of standard MAPF algorithms, namely Prioritized Planning (PrP) and MAPF-LNS2. Our exploration shows that the baseline approach in which all agents draw from a shared planning budget pool is ineffective in over-constrained situations. Instead, policies that distribute the planning budget over the agents are able to solve more problems with a smaller makespan.
>
---
#### [new 042] IONext: Unlocking the Next Era of Inertial Odometry
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于惯性里程计任务，旨在提升定位精度与泛化能力。针对现有Transformer模型对局部运动敏感度低、CNN模型时序建模不足的问题，论文提出了IONext网络，结合Dual-wing Adaptive Dynamic Mixer和Spatio-Temporal Gating Unit模块，实现更优的全局与局部运动特征融合及时序建模。**

- **链接: [http://arxiv.org/pdf/2507.17089v1](http://arxiv.org/pdf/2507.17089v1)**

> **作者:** Shanshan Zhang; Siyue Wang; Tianshui Wen; Qi Zhang; Ziheng Zhou; Lingxiang Zheng; Yu Yang
>
> **摘要:** Researchers have increasingly adopted Transformer-based models for inertial odometry. While Transformers excel at modeling long-range dependencies, their limited sensitivity to local, fine-grained motion variations and lack of inherent inductive biases often hinder localization accuracy and generalization. Recent studies have shown that incorporating large-kernel convolutions and Transformer-inspired architectural designs into CNN can effectively expand the receptive field, thereby improving global motion perception. Motivated by these insights, we propose a novel CNN-based module called the Dual-wing Adaptive Dynamic Mixer (DADM), which adaptively captures both global motion patterns and local, fine-grained motion features from dynamic inputs. This module dynamically generates selective weights based on the input, enabling efficient multi-scale feature aggregation. To further improve temporal modeling, we introduce the Spatio-Temporal Gating Unit (STGU), which selectively extracts representative and task-relevant motion features in the temporal domain. This unit addresses the limitations of temporal modeling observed in existing CNN approaches. Built upon DADM and STGU, we present a new CNN-based inertial odometry backbone, named Next Era of Inertial Odometry (IONext). Extensive experiments on six public datasets demonstrate that IONext consistently outperforms state-of-the-art (SOTA) Transformer- and CNN-based methods. For instance, on the RNIN dataset, IONext reduces the average ATE by 10% and the average RTE by 12% compared to the representative model iMOT.
>
---
#### [new 043] Hierarchical Reinforcement Learning Framework for Adaptive Walking Control Using General Value Functions of Lower-Limb Sensor Signals
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于康复机器人任务，旨在解决下肢外骨骼自适应行走控制问题。研究采用分层强化学习框架，结合生物传感器信号的通用价值函数预测，提升复杂地形下外骨骼的决策与控制性能，从而增强行动障碍者的移动能力与自主性。**

- **链接: [http://arxiv.org/pdf/2507.16983v1](http://arxiv.org/pdf/2507.16983v1)**

> **作者:** Sonny T. Jones; Grange M. Simpson; Patrick M. Pilarski; Ashley N. Dalrymple
>
> **备注:** 5 pages, 3 figures, accepted at the 6th Multi-disciplinary Conference on Reinforcement Learning and Decision Making (RLDM2025), June 11-14, 2025
>
> **摘要:** Rehabilitation technology is a natural setting to study the shared learning and decision-making of human and machine agents. In this work, we explore the use of Hierarchical Reinforcement Learning (HRL) to develop adaptive control strategies for lower-limb exoskeletons, aiming to enhance mobility and autonomy for individuals with motor impairments. Inspired by prominent models of biological sensorimotor processing, our investigated HRL approach breaks down the complex task of exoskeleton control adaptation into a higher-level framework for terrain strategy adaptation and a lower-level framework for providing predictive information; this latter element is implemented via the continual learning of general value functions (GVFs). GVFs generated temporal abstractions of future signal values from multiple wearable lower-limb sensors, including electromyography, pressure insoles, and goniometers. We investigated two methods for incorporating actual and predicted sensor signals into a policy network with the intent to improve the decision-making capacity of the control system of a lower-limb exoskeleton during ambulation across varied terrains. As a key result, we found that the addition of predictions made from GVFs increased overall network accuracy. Terrain-specific performance increases were seen while walking on even ground, uneven ground, up and down ramps, and turns, terrains that are often misclassified without predictive information. This suggests that predictive information can aid decision-making during uncertainty, e.g., on terrains that have a high chance of being misclassified. This work, therefore, contributes new insights into the nuances of HRL and the future development of exoskeletons to facilitate safe transitioning and traversing across different walking environments.
>
---
#### [new 044] PIG-Nav: Key Insights for Pretrained Image Goal Navigation Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉导航任务，旨在提升预训练图像目标导航模型的泛化能力和零样本性能。论文提出了PIG-Nav，通过改进模型结构和引入辅助任务提升导航表现，并设计了新的数据预处理流程。在多个环境中实现了性能提升，同时减少了对标注数据的依赖。**

- **链接: [http://arxiv.org/pdf/2507.17220v1](http://arxiv.org/pdf/2507.17220v1)**

> **作者:** Jiansong Wan; Chengming Zhou; Jinkua Liu; Xiangge Huang; Xiaoyu Chen; Xiaohan Yi; Qisen Yang; Baiting Zhu; Xin-Qiang Cai; Lixing Liu; Rushuai Yang; Chuheng Zhang; Sherif Abdelfattah; Hayong Shin; Pushi Zhang; Li Zhao; Jiang Bian
>
> **摘要:** Recent studies have explored pretrained (foundation) models for vision-based robotic navigation, aiming to achieve generalizable navigation and positive transfer across diverse environments while enhancing zero-shot performance in unseen settings. In this work, we introduce PIG-Nav (Pretrained Image-Goal Navigation), a new approach that further investigates pretraining strategies for vision-based navigation models and contributes in two key areas. Model-wise, we identify two critical design choices that consistently improve the performance of pretrained navigation models: (1) integrating an early-fusion network structure to combine visual observations and goal images via appropriately pretrained Vision Transformer (ViT) image encoder, and (2) introducing suitable auxiliary tasks to enhance global navigation representation learning, thus further improving navigation performance. Dataset-wise, we propose a novel data preprocessing pipeline for efficiently labeling large-scale game video datasets for navigation model training. We demonstrate that augmenting existing open navigation datasets with diverse gameplay videos improves model performance. Our model achieves an average improvement of 22.6% in zero-shot settings and a 37.5% improvement in fine-tuning settings over existing visual navigation foundation models in two complex simulated environments and one real-world environment. These results advance the state-of-the-art in pretrained image-goal navigation models. Notably, our model maintains competitive performance while requiring significantly less fine-tuning data, highlighting its potential for real-world deployment with minimal labeled supervision.
>
---
#### [new 045] Monocular Semantic Scene Completion via Masked Recurrent Networks
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于计算机视觉任务，旨在解决单目语义场景补全问题。通过提出一种新的两阶段框架MonoMRN，结合Masked Sparse Gated Recurrent Unit和距离注意力投影，提升复杂场景中可见和遮挡区域的预测性能，并增强模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.17661v1](http://arxiv.org/pdf/2507.17661v1)**

> **作者:** Xuzhi Wang; Xinran Wu; Song Wang; Lingdong Kong; Ziping Zhao
>
> **备注:** ICCV 2025; 15 pages, 10 figures, 6 tables; Code at https://github.com/alanWXZ/MonoMRN
>
> **摘要:** Monocular Semantic Scene Completion (MSSC) aims to predict the voxel-wise occupancy and semantic category from a single-view RGB image. Existing methods adopt a single-stage framework that aims to simultaneously achieve visible region segmentation and occluded region hallucination, while also being affected by inaccurate depth estimation. Such methods often achieve suboptimal performance, especially in complex scenes. We propose a novel two-stage framework that decomposes MSSC into coarse MSSC followed by the Masked Recurrent Network. Specifically, we propose the Masked Sparse Gated Recurrent Unit (MS-GRU) which concentrates on the occupied regions by the proposed mask updating mechanism, and a sparse GRU design is proposed to reduce the computation cost. Additionally, we propose the distance attention projection to reduce projection errors by assigning different attention scores according to the distance to the observed surface. Experimental results demonstrate that our proposed unified framework, MonoMRN, effectively supports both indoor and outdoor scenes and achieves state-of-the-art performance on the NYUv2 and SemanticKITTI datasets. Furthermore, we conduct robustness analysis under various disturbances, highlighting the role of the Masked Recurrent Network in enhancing the model's resilience to such challenges. The source code is publicly available.
>
---
#### [new 046] Perspective-Invariant 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D目标检测任务，旨在解决非车载平台（如四足机器人、无人机）及跨平台场景下的检测问题。作者提出了Pi3DET数据集与跨平台适应框架，实现几何与特征级对齐，提升跨平台检测性能，推动通用3D感知系统发展。**

- **链接: [http://arxiv.org/pdf/2507.17665v1](http://arxiv.org/pdf/2507.17665v1)**

> **作者:** Ao Liang; Lingdong Kong; Dongyue Lu; Youquan Liu; Jian Fang; Huaici Zhao; Wei Tsang Ooi
>
> **备注:** ICCV 2025; 46 pages, 18 figures, 22 tables; Project Page at https://pi3det.github.io
>
> **摘要:** With the rise of robotics, LiDAR-based 3D object detection has garnered significant attention in both academia and industry. However, existing datasets and methods predominantly focus on vehicle-mounted platforms, leaving other autonomous platforms underexplored. To bridge this gap, we introduce Pi3DET, the first benchmark featuring LiDAR data and 3D bounding box annotations collected from multiple platforms: vehicle, quadruped, and drone, thereby facilitating research in 3D object detection for non-vehicle platforms as well as cross-platform 3D detection. Based on Pi3DET, we propose a novel cross-platform adaptation framework that transfers knowledge from the well-studied vehicle platform to other platforms. This framework achieves perspective-invariant 3D detection through robust alignment at both geometric and feature levels. Additionally, we establish a benchmark to evaluate the resilience and robustness of current 3D detectors in cross-platform scenarios, providing valuable insights for developing adaptive 3D perception systems. Extensive experiments validate the effectiveness of our approach on challenging cross-platform tasks, demonstrating substantial gains over existing adaptation methods. We hope this work paves the way for generalizable and unified 3D perception systems across diverse and complex environments. Our Pi3DET dataset, cross-platform benchmark suite, and annotation toolkit have been made publicly available.
>
---
#### [new 047] VLM-Guided Visual Place Recognition for Planet-Scale Geo-Localization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，旨在解决全球尺度下单图像地理定位问题。传统方法在可扩展性和准确性上存在不足，作者提出结合视觉语言模型（VLM）与检索式视觉地点识别（VPR）的混合框架，利用VLM生成先验信息引导检索空间，并通过重排序机制提升定位精度，取得了优于现有方法的表现。**

- **链接: [http://arxiv.org/pdf/2507.17455v1](http://arxiv.org/pdf/2507.17455v1)**

> **作者:** Sania Waheed; Na Min An; Michael Milford; Sarvapali D. Ramchurn; Shoaib Ehsan
>
> **摘要:** Geo-localization from a single image at planet scale (essentially an advanced or extreme version of the kidnapped robot problem) is a fundamental and challenging task in applications such as navigation, autonomous driving and disaster response due to the vast diversity of locations, environmental conditions, and scene variations. Traditional retrieval-based methods for geo-localization struggle with scalability and perceptual aliasing, while classification-based approaches lack generalization and require extensive training data. Recent advances in vision-language models (VLMs) offer a promising alternative by leveraging contextual understanding and reasoning. However, while VLMs achieve high accuracy, they are often prone to hallucinations and lack interpretability, making them unreliable as standalone solutions. In this work, we propose a novel hybrid geo-localization framework that combines the strengths of VLMs with retrieval-based visual place recognition (VPR) methods. Our approach first leverages a VLM to generate a prior, effectively guiding and constraining the retrieval search space. We then employ a retrieval step, followed by a re-ranking mechanism that selects the most geographically plausible matches based on feature similarity and proximity to the initially estimated coordinates. We evaluate our approach on multiple geo-localization benchmarks and show that it consistently outperforms prior state-of-the-art methods, particularly at street (up to 4.51%) and city level (up to 13.52%). Our results demonstrate that VLM-generated geographic priors in combination with VPR lead to scalable, robust, and accurate geo-localization systems.
>
---
#### [new 048] Talk2Event: Grounded Understanding of Dynamic Scenes from Event Cameras
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于事件相机与语言理解的跨模任务，旨在解决动态场景中基于语言的物体定位问题。作者构建了大规模数据集Talk2Event，并提出EventRefer模型，融合多属性信息实现更精准的感知。**

- **链接: [http://arxiv.org/pdf/2507.17664v1](http://arxiv.org/pdf/2507.17664v1)**

> **作者:** Lingdong Kong; Dongyue Lu; Ao Liang; Rong Li; Yuhao Dong; Tianshuai Hu; Lai Xing Ng; Wei Tsang Ooi; Benoit R. Cottereau
>
> **备注:** Preprint; 42 pages, 17 figures, 16 tables; Project Page at https://talk2event.github.io
>
> **摘要:** Event cameras offer microsecond-level latency and robustness to motion blur, making them ideal for understanding dynamic environments. Yet, connecting these asynchronous streams to human language remains an open challenge. We introduce Talk2Event, the first large-scale benchmark for language-driven object grounding in event-based perception. Built from real-world driving data, we provide over 30,000 validated referring expressions, each enriched with four grounding attributes -- appearance, status, relation to viewer, and relation to other objects -- bridging spatial, temporal, and relational reasoning. To fully exploit these cues, we propose EventRefer, an attribute-aware grounding framework that dynamically fuses multi-attribute representations through a Mixture of Event-Attribute Experts (MoEE). Our method adapts to different modalities and scene dynamics, achieving consistent gains over state-of-the-art baselines in event-only, frame-only, and event-frame fusion settings. We hope our dataset and approach will establish a foundation for advancing multimodal, temporally-aware, and language-driven perception in real-world robotics and autonomy.
>
---
#### [new 049] Evaluating Uncertainty and Quality of Visual Language Action-enabled Robots
- **分类: cs.SE; cs.RO**

- **简介: 该论文属于机器人视觉语言模型评估任务，旨在解决当前仅依赖任务成功率评估VLA模型的不足。论文提出了8个不确定性指标和5个质量指标，通过908次任务执行数据及专家评估验证，发现这些指标与人类评估有中强相关性，可有效评估任务质量和模型置信度。**

- **链接: [http://arxiv.org/pdf/2507.17049v1](http://arxiv.org/pdf/2507.17049v1)**

> **作者:** Pablo Valle; Chengjie Lu; Shaukat Ali; Aitor Arrieta
>
> **摘要:** Visual Language Action (VLA) models are a multi-modal class of Artificial Intelligence (AI) systems that integrate visual perception, natural language understanding, and action planning to enable agents to interpret their environment, comprehend instructions, and perform embodied tasks autonomously. Recently, significant progress has been made to advance this field. These kinds of models are typically evaluated through task success rates, which fail to capture the quality of task execution and the mode's confidence in its decisions. In this paper, we propose eight uncertainty metrics and five quality metrics specifically designed for VLA models for robotic manipulation tasks. We assess their effectiveness through a large-scale empirical study involving 908 successful task executions from three state-of-the-art VLA models across four representative robotic manipulation tasks. Human domain experts manually labeled task quality, allowing us to analyze the correlation between our proposed metrics and expert judgments. The results reveal that several metrics show moderate to strong correlation with human assessments, highlighting their utility for evaluating task quality and model confidence. Furthermore, we found that some of the metrics can discriminate between high-, medium-, and low-quality executions from unsuccessful tasks, which can be interesting when test oracles are not available. Our findings challenge the adequacy of current evaluation practices that rely solely on binary success rates and pave the way for improved real-time monitoring and adaptive enhancement of VLA-enabled robotic systems.
>
---
#### [new 050] PRIX: Learning to Plan from Raw Pixels for End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决现有端到端模型依赖昂贵传感器、计算复杂的问题。作者提出PRIX，一种仅使用摄像头原始像素进行路径规划的高效架构，无需显式BEV表示或LiDAR。核心创新为上下文感知重校准Transformer（CaRT），提升视觉特征鲁棒性，实验证明其在主流数据集上性能领先，且更轻量、快速，适合实际部署。**

- **链接: [http://arxiv.org/pdf/2507.17596v1](http://arxiv.org/pdf/2507.17596v1)**

> **作者:** Maciej K. Wozniak; Lianhang Liu; Yixi Cai; Patric Jensfelt
>
> **备注:** under review
>
> **摘要:** While end-to-end autonomous driving models show promising results, their practical deployment is often hindered by large model sizes, a reliance on expensive LiDAR sensors and computationally intensive BEV feature representations. This limits their scalability, especially for mass-market vehicles equipped only with cameras. To address these challenges, we propose PRIX (Plan from Raw Pixels). Our novel and efficient end-to-end driving architecture operates using only camera data, without explicit BEV representation and forgoing the need for LiDAR. PRIX leverages a visual feature extractor coupled with a generative planning head to predict safe trajectories from raw pixel inputs directly. A core component of our architecture is the Context-aware Recalibration Transformer (CaRT), a novel module designed to effectively enhance multi-level visual features for more robust planning. We demonstrate through comprehensive experiments that PRIX achieves state-of-the-art performance on the NavSim and nuScenes benchmarks, matching the capabilities of larger, multimodal diffusion planners while being significantly more efficient in terms of inference speed and model size, making it a practical solution for real-world deployment. Our work is open-source and the code will be at https://maxiuw.github.io/prix.
>
---
## 更新

#### [replaced 001] Hierarchical Learning-Enhanced MPC for Safe Crowd Navigation with Heterogeneous Constraints
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.09859v2](http://arxiv.org/pdf/2506.09859v2)**

> **作者:** Huajian Liu; Yixuan Feng; Wei Dong; Kunpeng Fan; Chao Wang; Yongzhuo Gao
>
> **摘要:** In this paper, we propose a novel hierarchical framework for robot navigation in dynamic environments with heterogeneous constraints. Our approach leverages a graph neural network trained via reinforcement learning (RL) to efficiently estimate the robot's cost-to-go, formulated as local goal recommendations. A spatio-temporal path-searching module, which accounts for kinematic constraints, is then employed to generate a reference trajectory to facilitate solving the non-convex optimization problem used for explicit constraint enforcement. More importantly, we introduce an incremental action-masking mechanism and a privileged learning strategy, enabling end-to-end training of the proposed planner. Both simulation and real-world experiments demonstrate that the proposed method effectively addresses local planning in complex dynamic environments, achieving state-of-the-art (SOTA) performance. Compared with existing learning-optimization hybrid methods, our approach eliminates the dependency on high-fidelity simulation environments, offering significant advantages in computational efficiency and training scalability. The code will be released as open-source upon acceptance of the paper.
>
---
#### [replaced 002] VL-Explore: Zero-shot Vision-Language Exploration and Target Discovery by Mobile Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.08791v2](http://arxiv.org/pdf/2502.08791v2)**

> **作者:** Yuxuan Zhang; Adnan Abdullah; Sanjeev J. Koppal; Md Jahidul Islam
>
> **备注:** V2, includes suppl as appendix
>
> **摘要:** Vision-language navigation (VLN) has emerged as a promising paradigm, enabling mobile robots to perform zero-shot inference and execute tasks without specific pre-programming. However, current systems often separate map exploration and path planning, with exploration relying on inefficient algorithms due to limited (partially observed) environmental information. In this paper, we present a novel navigation pipeline named "VL-Explore" for simultaneous exploration and target discovery in unknown environments, leveraging the capabilities of a vision-language model named CLIP. Our approach requires only monocular vision and operates without any prior map or knowledge about the target. For comprehensive evaluations, we designed a functional prototype of a UGV (unmanned ground vehicle) system named "Open Rover", a customized platform for general-purpose VLN tasks. We integrated and deployed the VL-Explore pipeline on Open Rover to evaluate its throughput, obstacle avoidance capability, and trajectory performance across various real-world scenarios. Experimental results demonstrate that VL-Explore consistently outperforms traditional map-traversal algorithms and achieves performance comparable to path-planning methods that depend on prior map and target knowledge. Notably, VL-Explore offers real-time active navigation without requiring pre-captured candidate images or pre-built node graphs, addressing key limitations of existing VLN pipelines.
>
---
#### [replaced 003] Towards Generalist Robot Learning from Internet Video: A Survey
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2404.19664v5](http://arxiv.org/pdf/2404.19664v5)**

> **作者:** Robert McCarthy; Daniel C. H. Tan; Dominik Schmidt; Fernando Acero; Nathan Herr; Yilun Du; Thomas G. Thuruthel; Zhibin Li
>
> **摘要:** Scaling deep learning to massive and diverse internet data has driven remarkable breakthroughs in domains such as video generation and natural language processing. Robot learning, however, has thus far failed to replicate this success and remains constrained by a scarcity of available data. Learning from videos (LfV) methods aim to address this data bottleneck by augmenting traditional robot data with large-scale internet video. This video data provides foundational information regarding physical dynamics, behaviours, and tasks, and can be highly informative for general-purpose robots. This survey systematically examines the emerging field of LfV. We first outline essential concepts, including detailing fundamental LfV challenges such as distribution shift and missing action labels in video data. Next, we comprehensively review current methods for extracting knowledge from large-scale internet video, overcoming LfV challenges, and improving robot learning through video-informed training. The survey concludes with a critical discussion of future opportunities. Here, we emphasize the need for scalable foundation model approaches that can leverage the full range of available internet video and enhance the learning of robot policies and dynamics models. Overall, the survey aims to inform and catalyse future LfV research, driving progress towards general-purpose robots.
>
---
#### [replaced 004] Active Probing with Multimodal Predictions for Motion Planning
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.09822v4](http://arxiv.org/pdf/2507.09822v4)**

> **作者:** Darshan Gadginmath; Farhad Nawaz; Minjun Sung; Faizan M Tariq; Sangjae Bae; David Isele; Fabio Pasqualetti; Jovin D'sa
>
> **备注:** To appear at IROS '25. 8 pages. 3 tables. 6 figures. Project page: https://darshangm.github.io/papers/active-probing-multimodal-predictions/
>
> **摘要:** Navigation in dynamic environments requires autonomous systems to reason about uncertainties in the behavior of other agents. In this paper, we introduce a unified framework that combines trajectory planning with multimodal predictions and active probing to enhance decision-making under uncertainty. We develop a novel risk metric that seamlessly integrates multimodal prediction uncertainties through mixture models. When these uncertainties follow a Gaussian mixture distribution, we prove that our risk metric admits a closed-form solution, and is always finite, thus ensuring analytical tractability. To reduce prediction ambiguity, we incorporate an active probing mechanism that strategically selects actions to improve its estimates of behavioral parameters of other agents, while simultaneously handling multimodal uncertainties. We extensively evaluate our framework in autonomous navigation scenarios using the MetaDrive simulation environment. Results demonstrate that our active probing approach successfully navigates complex traffic scenarios with uncertain predictions. Additionally, our framework shows robust performance across diverse traffic agent behavior models, indicating its broad applicability to real-world autonomous navigation challenges. Code and videos are available at https://darshangm.github.io/papers/active-probing-multimodal-predictions/.
>
---
#### [replaced 005] Rethinking Range-View LiDAR Segmentation in Adverse Weather
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08979v2](http://arxiv.org/pdf/2506.08979v2)**

> **作者:** Longyu Yang; Lu Zhang; Jun Liu; Yap-Peng Tan; Heng Tao Shen; Xiaofeng Zhu; Ping Hu
>
> **摘要:** LiDAR segmentation has emerged as an important task to enrich scene perception and understanding. Range-view-based methods have gained popularity due to their high computational efficiency and compatibility with real-time deployment. However, their generalized performance under adverse weather conditions remains underexplored, limiting their reliability in real-world environments. In this work, we identify and analyze the unique challenges that affect the generalization of range-view LiDAR segmentation in severe weather. To address these challenges, we propose a modular and lightweight framework that enhances robustness without altering the core architecture of existing models. Our method reformulates the initial stem block of standard range-view networks into two branches to process geometric attributes and reflectance intensity separately. Specifically, a Geometric Abnormality Suppression (GAS) module reduces the influence of weather-induced spatial noise, and a Reflectance Distortion Calibration (RDC) module corrects reflectance distortions through memory-guided adaptive instance normalization. The processed features are then fused and passed to the original segmentation pipeline. Extensive experiments on different benchmarks and baseline models demonstrate that our approach significantly improves generalization to adverse weather with minimal inference overhead, offering a practical and effective solution for real-world LiDAR segmentation.
>
---
#### [replaced 006] Onto-LLM-TAMP: Knowledge-oriented Task and Motion Planning using Large Language Models
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.07493v2](http://arxiv.org/pdf/2412.07493v2)**

> **作者:** Muhayy Ud Din; Jan Rosell; Waseem Akram; Isiah Zaplana; Maximo A Roa; Irfan Hussain
>
> **备注:** Submitted to knowledge based systems
>
> **摘要:** Performing complex manipulation tasks in dynamic environments requires efficient Task and Motion Planning (TAMP) approaches that combine high-level symbolic plans with low-level motion control. Advances in Large Language Models (LLMs), such as GPT-4, are transforming task planning by offering natural language as an intuitive and flexible way to describe tasks, generate symbolic plans, and reason. However, the effectiveness of LLM-based TAMP approaches is limited due to static and template-based prompting, which limits adaptability to dynamic environments and complex task contexts. To address these limitations, this work proposes a novel Onto-LLM-TAMP framework that employs knowledge-based reasoning to refine and expand user prompts with task-contextual reasoning and knowledge-based environment state descriptions. Integrating domain-specific knowledge into the prompt ensures semantically accurate and context-aware task plans. The proposed framework demonstrates its effectiveness by resolving semantic errors in symbolic plan generation, such as maintaining logical temporal goal ordering in scenarios involving hierarchical object placement. The proposed framework is validated through both simulation and real-world scenarios, demonstrating significant improvements over the baseline approach in terms of adaptability to dynamic environments and the generation of semantically correct task plans.
>
---
#### [replaced 007] X-MOBILITY: End-To-End Generalizable Navigation via World Modeling
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.17491v3](http://arxiv.org/pdf/2410.17491v3)**

> **作者:** Wei Liu; Huihua Zhao; Chenran Li; Joydeep Biswas; Billy Okal; Pulkit Goyal; Yan Chang; Soha Pouya
>
> **摘要:** General-purpose navigation in challenging environments remains a significant problem in robotics, with current state-of-the-art approaches facing myriad limitations. Classical approaches struggle with cluttered settings and require extensive tuning, while learning-based methods face difficulties generalizing to out-of-distribution environments. This paper introduces X-Mobility, an end-to-end generalizable navigation model that overcomes existing challenges by leveraging three key ideas. First, X-Mobility employs an auto-regressive world modeling architecture with a latent state space to capture world dynamics. Second, a diverse set of multi-head decoders enables the model to learn a rich state representation that correlates strongly with effective navigation skills. Third, by decoupling world modeling from action policy, our architecture can train effectively on a variety of data sources, both with and without expert policies: off-policy data allows the model to learn world dynamics, while on-policy data with supervisory control enables optimal action policy learning. Through extensive experiments, we demonstrate that X-Mobility not only generalizes effectively but also surpasses current state-of-the-art navigation approaches. Additionally, X-Mobility also achieves zero-shot Sim2Real transferability and shows strong potential for cross-embodiment generalization.
>
---
#### [replaced 008] Efficient Precision-Scalable Hardware for Microscaling (MX) Processing in Robotics Learning
- **分类: cs.AR; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.22404v2](http://arxiv.org/pdf/2505.22404v2)**

> **作者:** Stef Cuyckens; Xiaoling Yi; Nitish Satya Murthy; Chao Fang; Marian Verhelst
>
> **备注:** To appear in 2025 IEEE/ACM International Symposium on Low Power Electronics and Design (ISLPED 2025)
>
> **摘要:** Autonomous robots require efficient on-device learning to adapt to new environments without cloud dependency. For this edge training, Microscaling (MX) data types offer a promising solution by combining integer and floating-point representations with shared exponents, reducing energy consumption while maintaining accuracy. However, the state-of-the-art continuous learning processor, namely Dacapo, faces limitations with its MXINT-only support and inefficient vector-based grouping during backpropagation. In this paper, we present, to the best of our knowledge, the first work that addresses these limitations with two key innovations: (1) a precision-scalable arithmetic unit that supports all six MX data types by exploiting sub-word parallelism and unified integer and floating-point processing; and (2) support for square shared exponent groups to enable efficient weight handling during backpropagation, removing storage redundancy and quantization overhead. We evaluate our design against Dacapo under iso-peak-throughput on four robotics workloads in TSMC 16nm FinFET technology at 400MHz, reaching a 51% lower memory footprint, and 4x higher effective training throughput, while achieving comparable energy efficiency, enabling efficient robotics continual learning at the edge.
>
---
#### [replaced 009] Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.21427v2](http://arxiv.org/pdf/2506.21427v2)**

> **作者:** Prajwal Koirala; Cody Fleming
>
> **摘要:** Generative models such as diffusion and flow-matching offer expressive policies for offline reinforcement learning (RL) by capturing rich, multimodal action distributions, but their iterative sampling introduces high inference costs and training instability due to gradient propagation across sampling steps. We propose the \textit{Single-Step Completion Policy} (SSCP), a generative policy trained with an augmented flow-matching objective to predict direct completion vectors from intermediate flow samples, enabling accurate, one-shot action generation. In an off-policy actor-critic framework, SSCP combines the expressiveness of generative models with the training and inference efficiency of unimodal policies, without requiring long backpropagation chains. Our method scales effectively to offline, offline-to-online, and online RL settings, offering substantial gains in speed and adaptability over diffusion-based baselines. We further extend SSCP to goal-conditioned RL, enabling flat policies to exploit subgoal structures without explicit hierarchical inference. SSCP achieves strong results across standard offline RL and behavior cloning benchmarks, positioning it as a versatile, expressive, and efficient framework for deep RL and sequential decision-making.
>
---
#### [replaced 010] ICCO: Learning an Instruction-conditioned Coordinator for Language-guided Task-aligned Multi-robot Control
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.12122v2](http://arxiv.org/pdf/2503.12122v2)**

> **作者:** Yoshiki Yano; Kazuki Shibata; Maarten Kokshoorn; Takamitsu Matsubara
>
> **备注:** 8 pages, 9 figures, to be published in the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems
>
> **摘要:** Recent advances in Large Language Models (LLMs) have permitted the development of language-guided multi-robot systems, which allow robots to execute tasks based on natural language instructions. However, achieving effective coordination in distributed multi-agent environments remains challenging due to (1) misalignment between instructions and task requirements and (2) inconsistency in robot behaviors when they independently interpret ambiguous instructions. To address these challenges, we propose Instruction-Conditioned Coordinator (ICCO), a Multi-Agent Reinforcement Learning (MARL) framework designed to enhance coordination in language-guided multi-robot systems. ICCO consists of a Coordinator agent and multiple Local Agents, where the Coordinator generates Task-Aligned and Consistent Instructions (TACI) by integrating language instructions with environmental states, ensuring task alignment and behavioral consistency. The Coordinator and Local Agents are jointly trained to optimize a reward function that balances task efficiency and instruction following. A Consistency Enhancement Term is added to the learning objective to maximize mutual information between instructions and robot behaviors, further improving coordination. Simulation and real-world experiments validate the effectiveness of ICCO in achieving language-guided task-aligned multi-robot control. The demonstration can be found at https://yanoyoshiki.github.io/ICCO/.
>
---
#### [replaced 011] How to Adapt Control Barrier Functions? A Learning-Based Approach with Applications to a VTOL Quadplane
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.03038v3](http://arxiv.org/pdf/2504.03038v3)**

> **作者:** Taekyung Kim; Randal W. Beard; Dimitra Panagou
>
> **备注:** 2025 IEEE Conference on Decision and Control (CDC). Project page: https://www.taekyung.me/how-to-adapt-cbf
>
> **摘要:** In this paper, we present a novel theoretical framework for online adaptation of Control Barrier Function (CBF) parameters, i.e., of the class K functions included in the CBF condition, under input constraints. We introduce the concept of locally validated CBF parameters, which are adapted online to guarantee finite-horizon safety, based on conditions derived from Nagumo's theorem and tangent cone analysis. To identify these parameters online, we integrate a learning-based approach with an uncertainty-aware verification process that account for both epistemic and aleatoric uncertainties inherent in neural network predictions. Our method is demonstrated on a VTOL quadplane model during challenging transition and landing maneuvers, showcasing enhanced performance while maintaining safety.
>
---
#### [replaced 012] Robot Operation of Home Appliances by Reading User Manuals
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2505.20424v2](http://arxiv.org/pdf/2505.20424v2)**

> **作者:** Jian Zhang; Hanbo Zhang; Anxing Xiao; David Hsu
>
> **摘要:** Operating home appliances, among the most common tools in every household, is a critical capability for assistive home robots. This paper presents ApBot, a robot system that operates novel household appliances by "reading" their user manuals. ApBot faces multiple challenges: (i) infer goal-conditioned partial policies from their unstructured, textual descriptions in a user manual document, (ii) ground the policies to the appliance in the physical world, and (iii) execute the policies reliably over potentially many steps, despite compounding errors. To tackle these challenges, ApBot constructs a structured, symbolic model of an appliance from its manual, with the help of a large vision-language model (VLM). It grounds the symbolic actions visually to control panel elements. Finally, ApBot closes the loop by updating the model based on visual feedback. Our experiments show that across a wide range of simulated and real-world appliances, ApBot achieves consistent and statistically significant improvements in task success rate, compared with state-of-the-art large VLMs used directly as control policies. These results suggest that a structured internal representations plays an important role in robust robot operation of home appliances, especially, complex ones.
>
---
#### [replaced 013] Why Automate This? Exploring Correlations between Desire for Robotic Automation, Invested Time and Well-Being
- **分类: cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.06348v3](http://arxiv.org/pdf/2501.06348v3)**

> **作者:** Ruchira Ray; Leona Pang; Sanjana Srivastava; Li Fei-Fei; Samantha Shorey; Roberto Martín-Martín
>
> **备注:** 20 pages, 14 figures
>
> **摘要:** Understanding the motivations underlying the human inclination to automate tasks is vital to developing truly helpful robots integrated into daily life. Accordingly, we ask: are individuals more inclined to automate chores based on the time they consume or the feelings experienced while performing them? This study explores these preferences and whether they vary across different social groups (i.e., gender category and income level). Leveraging data from the BEHAVIOR-1K dataset, the American Time-Use Survey, and the American Time-Use Survey Well-Being Module, we investigate the relationship between the desire for automation, time spent on daily activities, and their associated feelings - Happiness, Meaningfulness, Sadness, Painfulness, Stressfulness, or Tiredness. Our key findings show that, despite common assumptions, time spent does not strongly relate to the desire for automation for the general population. For the feelings analyzed, only happiness and pain are key indicators. Significant differences by gender and economic level also emerged: Women prefer to automate stressful activities, whereas men prefer to automate those that make them unhappy; mid-income individuals prioritize automating less enjoyable and meaningful activities, while low and high-income show no significant correlations. We hope our research helps motivate technologies to develop robots that match the priorities of potential users, moving domestic robotics toward more socially relevant solutions. We open-source all the data, including an online tool that enables the community to replicate our analysis and explore additional trends at https://robin-lab.cs.utexas.edu/why-automate-this/.
>
---
#### [replaced 014] RoBridge: A Hierarchical Architecture Bridging Cognition and Execution for General Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01709v3](http://arxiv.org/pdf/2505.01709v3)**

> **作者:** Kaidong Zhang; Rongtao Xu; Pengzhen Ren; Junfan Lin; Hefeng Wu; Liang Lin; Xiaodan Liang
>
> **备注:** project page: https://abliao.github.io/RoBridge/
>
> **摘要:** Operating robots in open-ended scenarios with diverse tasks is a crucial research and application direction in robotics. While recent progress in natural language processing and large multimodal models has enhanced robots' ability to understand complex instructions, robot manipulation still faces the procedural skill dilemma and the declarative skill dilemma in open environments. Existing methods often compromise cognitive and executive capabilities. To address these challenges, in this paper, we propose RoBridge, a hierarchical intelligent architecture for general robotic manipulation. It consists of a high-level cognitive planner (HCP) based on a large-scale pre-trained vision-language model (VLM), an invariant operable representation (IOR) serving as a symbolic bridge, and a generalist embodied agent (GEA). RoBridge maintains the declarative skill of VLM and unleashes the procedural skill of reinforcement learning, effectively bridging the gap between cognition and execution. RoBridge demonstrates significant performance improvements over existing baselines, achieving a 75% success rate on new tasks and an 83% average success rate in sim-to-real generalization using only five real-world data samples per task. This work represents a significant step towards integrating cognitive reasoning with physical execution in robotic systems, offering a new paradigm for general robotic manipulation.
>
---
#### [replaced 015] ROADWork Dataset: Learning to Recognize, Observe, Analyze and Drive Through Work Zones
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2406.07661v2](http://arxiv.org/pdf/2406.07661v2)**

> **作者:** Anurag Ghosh; Shen Zheng; Robert Tamburo; Khiem Vuong; Juan Alvarez-Padilla; Hailiang Zhu; Michael Cardei; Nicholas Dunn; Christoph Mertz; Srinivasa G. Narasimhan
>
> **备注:** ICCV 2025 Accepted Paper
>
> **摘要:** Perceiving and autonomously navigating through work zones is a challenging and underexplored problem. Open datasets for this long-tailed scenario are scarce. We propose the ROADWork dataset to learn to recognize, observe, analyze, and drive through work zones. State-of-the-art foundation models fail when applied to work zones. Fine-tuning models on our dataset significantly improves perception and navigation in work zones. With ROADWork dataset, we discover new work zone images with higher precision (+32.5%) at a much higher rate (12.8$\times$) around the world. Open-vocabulary methods fail too, whereas fine-tuned detectors improve performance (+32.2 AP). Vision-Language Models (VLMs) struggle to describe work zones, but fine-tuning substantially improves performance (+36.7 SPICE). Beyond fine-tuning, we show the value of simple techniques. Video label propagation provides additional gains (+2.6 AP) for instance segmentation. While reading work zone signs, composing a detector and text spotter via crop-scaling improves performance +14.2% 1-NED). Composing work zone detections to provide context further reduces hallucinations (+3.9 SPICE) in VLMs. We predict navigational goals and compute drivable paths from work zone videos. Incorporating road work semantics ensures 53.6% goals have angular error (AE) < 0.5 (+9.9 %) and 75.3% pathways have AE < 0.5 (+8.1 %).
>
---
#### [replaced 016] Virtual Holonomic Constraints in Motion Planning: Revisiting Feasibility and Limitations
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.07983v2](http://arxiv.org/pdf/2505.07983v2)**

> **作者:** Maksim Surov
>
> **备注:** 17 pages, 3 figure
>
> **摘要:** This paper addresses the feasibility of virtual holonomic constraints (VHCs) in the context of motion planning for underactuated mechanical systems with a single degree of underactuation. While existing literature has established a widely accepted definition of VHC, we argue that this definition is overly restrictive and excludes a broad class of admissible trajectories from consideration. To illustrate this point, we analyze a periodic motion of the Planar Vertical Take-Off and Landing (PVTOL) aircraft that satisfies all standard motion planning requirements, including orbital stabilizability. However, for this solution -- as well as for a broad class of similar ones -- there exists no VHC that satisfies the conventional definition. We further provide a formal proof demonstrating that the conditions imposed by this definition necessarily fail for a broad class of trajectories of mechanical systems. These findings call for a reconsideration of the current definition of VHCs, with the potential to significantly broaden their applicability in motion planning.
>
---
#### [replaced 017] GEMINUS: Dual-aware Global and Scene-Adaptive Mixture-of-Experts for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.14456v3](http://arxiv.org/pdf/2507.14456v3)**

> **作者:** Chi Wan; Yixin Cui; Jiatong Du; Shuo Yang; Yulong Bai; Yanjun Huang
>
> **摘要:** End-to-end autonomous driving requires adaptive and robust handling of complex and diverse traffic environments. However, prevalent single-mode planning methods attempt to learn an overall policy while struggling to acquire diversified driving skills to handle diverse scenarios. Therefore, this paper proposes GEMINUS, a Mixture-of-Experts end-to-end autonomous driving framework featuring a Global Expert, a Scene-Adaptive Experts Group, and equipped with a Dual-aware Router. Specifically, the Global Expert is trained on the overall dataset, possessing robust performance. The Scene-Adaptive Experts are trained on corresponding scene subsets, achieving adaptive performance. The Dual-aware Router simultaneously considers scenario-level features and routing uncertainty to dynamically activate expert modules. Through the effective coupling of the Global Expert and the Scene-Adaptive Experts Group via the Dual-aware Router, GEMINUS achieves adaptive and robust performance in diverse scenarios. GEMINUS outperforms existing methods in the Bench2Drive closed-loop benchmark and achieves state-of-the-art performance in Driving Score and Success Rate, even with only monocular vision input. Furthermore, ablation studies demonstrate significant improvements over the original single-expert baseline: 7.67% in Driving Score, 22.06% in Success Rate, and 19.41% in MultiAbility-Mean. The code will be available at https://github.com/newbrains1/GEMINUS.
>
---
#### [replaced 018] Optimizing Design and Control Methods for Using Collaborative Robots in Upper-Limb Rehabilitation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2407.18661v2](http://arxiv.org/pdf/2407.18661v2)**

> **作者:** Dario Onfiani; Marco Caramaschi; Luigi Biagiotti; Fabio Pini
>
> **摘要:** In this paper, we address the development of a robotic rehabilitation system for the upper limbs based on collaborative end-effector solutions. The use of commercial collaborative robots offers significant advantages for this task, as they are optimized from an engineering perspective and ensure safe physical interaction with humans. However, they also come with noticeable drawbacks, such as the limited range of sizes available on the market and the standard control modes, which are primarily oriented towards industrial or service applications. To address these limitations, we propose an optimization-based design method to fully exploit the capability of the cobot in performing rehabilitation tasks. Additionally, we introduce a novel control architecture based on an admittance-type Virtual Fixture method, which constrains the motion of the robot along a prescribed path. This approach allows for an intuitive definition of the task to be performed via Programming by Demonstration and enables the system to operate both passively and actively. In passive mode, the system supports the patient during task execution with additional force, while in active mode, it opposes the motion with a braking force. Experimental results demonstrate the effectiveness of the proposed method.
>
---
#### [replaced 019] First, Learn What You Don't Know: Active Information Gathering for Driving at the Limits of Handling
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2411.00107v2](http://arxiv.org/pdf/2411.00107v2)**

> **作者:** Alexander Davydov; Franck Djeumou; Marcus Greiff; Makoto Suminaka; Michael Thompson; John Subosits; Thomas Lew
>
> **摘要:** Combining data-driven models that adapt online and model predictive control (MPC) has enabled effective control of nonlinear systems. However, when deployed on unstable systems, online adaptation may not be fast enough to ensure reliable simultaneous learning and control. For example, a controller on a vehicle executing highly dynamic maneuvers--such as drifting to avoid an obstacle--may push the vehicle's tires to their friction limits, destabilizing the vehicle and allowing modeling errors to quickly compound and cause a loss of control. To address this challenge, we present an active information gathering framework for identifying vehicle dynamics as quickly as possible. We propose an expressive vehicle dynamics model that leverages Bayesian last-layer meta-learning to enable rapid online adaptation. The model's uncertainty estimates are used to guide informative data collection and quickly improve the model prior to deployment. Dynamic drifting experiments on a Toyota Supra show that (i) the framework enables reliable control of a vehicle at the edge of stability, (ii) online adaptation alone may not suffice for zero-shot control and can lead to undesirable transient errors or spin-outs, and (iii) active data collection helps achieve reliable performance.
>
---
#### [replaced 020] LLM as a code generator in Agile Model Driven Development
- **分类: cs.AI; cs.ET; cs.RO; cs.SE**

- **链接: [http://arxiv.org/pdf/2410.18489v2](http://arxiv.org/pdf/2410.18489v2)**

> **作者:** Ahmed R. Sadik; Sebastian Brulin; Markus Olhofer
>
> **摘要:** Leveraging Large Language Models (LLM) like GPT4 in the auto generation of code represents a significant advancement, yet it is not without its challenges. The ambiguity inherent in natural language descriptions of software poses substantial obstacles to generating deployable, structured artifacts. This research champions Model Driven Development (MDD) as a viable strategy to overcome these challenges, proposing an Agile Model Driven Development (AMDD) approach that employs GPT4 as a code generator. This approach enhances the flexibility and scalability of the code auto generation process and offers agility that allows seamless adaptation to changes in models or deployment environments. We illustrate this by modeling a multi agent Unmanned Vehicle Fleet (UVF) system using the Unified Modeling Language (UML), significantly reducing model ambiguity by integrating the Object Constraint Language (OCL) for code structure meta modeling, and the FIPA ontology language for communication semantics meta modeling. Applying GPT4 auto generation capabilities yields Java and Python code that is compatible with the JADE and PADE frameworks, respectively. Our thorough evaluation of the auto generated code verifies its alignment with expected behaviors and identifies enhancements in agent interactions. Structurally, we assessed the complexity of code derived from a model constrained solely by OCL meta models, against that influenced by both OCL and FIPA ontology meta models. The results indicate that the ontology constrained meta model produces inherently more complex code, yet its cyclomatic complexity remains within manageable levels, suggesting that additional meta model constraints can be incorporated without exceeding the high risk threshold for complexity.
>
---
