# 机器人 cs.RO

- **最新发布 31 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist Vision-Language-Action Policies
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究面向具身智能的视觉-语言-动作（VLA）模型，旨在解决多源异构机器人数据在形态、动作空间等方面的异质性问题。提出HiMoE-VLA框架，采用分层专家混合结构自适应融合多样化数据，提升跨平台泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.05693v1](https://arxiv.org/pdf/2512.05693v1)**

> **作者:** Zhiying Du; Bei Liu; Yaobo Liang; Yichao Shen; Haidong Cao; Xiangyu Zheng; Zhiyuan Feng; Zuxuan Wu; Jiaolong Yang; Yu-Gang Jiang
>
> **摘要:** The development of foundation models for embodied intelligence critically depends on access to large-scale, high-quality robot demonstration data. Recent approaches have sought to address this challenge by training on large collections of heterogeneous robotic datasets. However, unlike vision or language data, robotic demonstrations exhibit substantial heterogeneity across embodiments and action spaces as well as other prominent variations such as senor configurations and action control frequencies. The lack of explicit designs for handling such heterogeneity causes existing methods to struggle with integrating diverse factors, thereby limiting their generalization and leading to degraded performance when transferred to new settings. In this paper, we present HiMoE-VLA, a novel vision-language-action (VLA) framework tailored to effectively handle diverse robotic data with heterogeneity. Specifically, we introduce a Hierarchical Mixture-of-Experts (HiMoE) architecture for the action module which adaptively handles multiple sources of heterogeneity across layers and gradually abstracts them into shared knowledge representations. Through extensive experimentation with simulation benchmarks and real-world robotic platforms, HiMoE-VLA demonstrates a consistent performance boost over existing VLA baselines, achieving higher accuracy and robust generalization across diverse robots and action spaces. The code and models are publicly available at https://github.com/ZhiyingDu/HiMoE-VLA.
>
---
#### [new 002] Wake Vectoring for Efficient Morphing Flight
- **分类: cs.RO**

- **简介: 该论文研究形态可变飞行机器人在飞行中重构时的推力损失问题，提出一种无电子元件的被动尾流偏转机制。通过在ATMO机器人中集成内部导流板，回收并向下引导旋翼尾流，恢复最多40%的垂直推力，提升变形过程中的飞行效率与控制能力。**

- **链接: [https://arxiv.org/pdf/2512.05211v1](https://arxiv.org/pdf/2512.05211v1)**

> **作者:** Ioannis Mandralis; Severin Schumacher; Morteza Gharib
>
> **摘要:** Morphing aerial robots have the potential to transform autonomous flight, enabling navigation through cluttered environments, perching, and seamless transitions between aerial and terrestrial locomotion. Yet mid-flight reconfiguration presents a critical aerodynamic challenge: tilting propulsors to achieve shape change reduces vertical thrust, undermining stability and control authority. Here, we introduce a passive wake vectoring mechanism that recovers lost thrust during morphing. Integrated into a novel robotic system, Aerially Transforming Morphobot (ATMO), internal deflectors intercept and redirect rotor wake downward, passively steering airflow momentum that would otherwise be wasted. This electronics-free solution achieves up to a 40% recovery of vertical thrust in configurations where no useful thrust would otherwise be produced, substantially extending hover and maneuvering capabilities during transformation. Our findings highlight a new direction for morphing aerial robot design, where passive aerodynamic structures, inspired by thrust vectoring in rockets and aircraft, enable efficient, agile flight without added mechanical complexity.
>
---
#### [new 003] Disturbance Compensation for Safe Kinematic Control of Robotic Systems with Closed Architecture
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对闭源机器人系统内环不可改、模型不确定的问题，提出一种外环可集成的扰动补偿方法，结合干扰抑制与安全控制，实现高精度、安全的运动跟踪。**

- **链接: [https://arxiv.org/pdf/2512.05292v1](https://arxiv.org/pdf/2512.05292v1)**

> **作者:** Fan Zhang; Jinfeng Chen; Joseph J. B. Mvogo Ahanda; Hanz Richter; Ge Lv; Bin Hu; Qin Lin
>
> **摘要:** In commercial robotic systems, it is common to encounter a closed inner-loop torque controller that is not user-modifiable. However, the outer-loop controller, which sends kinematic commands such as position or velocity for the inner-loop controller to track, is typically exposed to users. In this work, we focus on the development of an easily integrated add-on at the outer-loop layer by combining disturbance rejection control and robust control barrier function for high-performance tracking and safe control of the whole dynamic system of an industrial manipulator. This is particularly beneficial when 1) the inner-loop controller is imperfect, unmodifiable, and uncertain; and 2) the dynamic model exhibits significant uncertainty. Stability analysis, formal safety guarantee proof, and hardware experiments with a PUMA robotic manipulator are presented. Our solution demonstrates superior performance in terms of simplicity of implementation, robustness, tracking precision, and safety compared to the state of the art. Video: https://youtu.be/zw1tanvrV8Q
>
---
#### [new 004] Correspondence-Oriented Imitation Learning: Flexible Visuomotor Control with 3D Conditioning
- **分类: cs.RO**

- **简介: 该论文提出对应导向的模仿学习（COIL），用于灵活的视觉运动控制任务。旨在解决任务表示缺乏灵活性的问题，通过3D关键点对应关系定义任务，支持可变时空粒度，并设计了带时空注意力机制的条件策略，实现跨任务、物体和运动模式的泛化。**

- **链接: [https://arxiv.org/pdf/2512.05953v1](https://arxiv.org/pdf/2512.05953v1)**

> **作者:** Yunhao Cao; Zubin Bhaumik; Jessie Jia; Xingyi He; Kuan Fang
>
> **摘要:** We introduce Correspondence-Oriented Imitation Learning (COIL), a conditional policy learning framework for visuomotor control with a flexible task representation in 3D. At the core of our approach, each task is defined by the intended motion of keypoints selected on objects in the scene. Instead of assuming a fixed number of keypoints or uniformly spaced time intervals, COIL supports task specifications with variable spatial and temporal granularity, adapting to different user intents and task requirements. To robustly ground this correspondence-oriented task representation into actions, we design a conditional policy with a spatio-temporal attention mechanism that effectively fuses information across multiple input modalities. The policy is trained via a scalable self-supervised pipeline using demonstrations collected in simulation, with correspondence labels automatically generated in hindsight. COIL generalizes across tasks, objects, and motion patterns, achieving superior performance compared to prior methods on real-world manipulation tasks under both sparse and dense specifications.
>
---
#### [new 005] Global stability of vehicle-with-driver dynamics via Sum-of-Squares programming
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究车辆-驾驶员系统的全局稳定性，旨在估计系统吸引域内的安全不变集。通过Sum-of-Squares编程优化李雅普诺夫函数，结合状态安全约束，实现对稳定性和安全性兼具的区域估计，并验证于典型行驶场景。**

- **链接: [https://arxiv.org/pdf/2512.05806v1](https://arxiv.org/pdf/2512.05806v1)**

> **作者:** Martino Gulisano; Marco Gabiccini
>
> **备注:** 20 pages, 7 figures, 2 tables
>
> **摘要:** This work estimates safe invariant subsets of the Region of Attraction (ROA) for a seven-state vehicle-with-driver system, capturing both asymptotic stability and the influence of state-safety bounds along the system trajectory. Safe sets are computed by optimizing Lyapunov functions through an original iterative Sum-of-Squares (SOS) procedure. The method is first demonstrated on a two-state benchmark, where it accurately recovers a prescribed safe region as the 1-level set of a polynomial Lyapunov function. We then describe the distinguishing characteristics of the studied vehicle-with-driver system: the control dynamics mimic human driver behavior through a delayed preview-tracking model that, with suitable parameter choices, can also emulate digital controllers. To enable SOS optimization, a polynomial approximation of the nonlinear vehicle model is derived, together with its operating-envelope constraints. The framework is then applied to understeering and oversteering scenarios, and the estimated safe sets are compared with reference boundaries obtained from exhaustive simulations. The results show that SOS techniques can efficiently deliver Lyapunov-defined safe regions, supporting their potential use for real-time safety assessment, for example as a supervisory layer for active vehicle control.
>
---
#### [new 006] Training-Time Action Conditioning for Efficient Real-Time Chunking
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究实时分块控制任务，旨在降低视觉-语言-动作模型的推理延迟。提出训练时模拟延迟并直接条件化动作前缀，避免推理时图像修复带来的计算开销。方法简单高效，无需修改模型或运行系统，实验证明其在性能和速度上与现有方法相当但更节省计算。**

- **链接: [https://arxiv.org/pdf/2512.05964v1](https://arxiv.org/pdf/2512.05964v1)**

> **作者:** Kevin Black; Allen Z. Ren; Michael Equi; Sergey Levine
>
> **摘要:** Real-time chunking (RTC) enables vision-language-action models (VLAs) to generate smooth, reactive robot trajectories by asynchronously predicting action chunks and conditioning on previously committed actions via inference-time inpainting. However, this inpainting method introduces computational overhead that increases inference latency. In this work, we propose a simple alternative: simulating inference delay at training time and conditioning on action prefixes directly, eliminating any inference-time overhead. Our method requires no modifications to the model architecture or robot runtime, and can be implemented with only a few additional lines of code. In simulated experiments, we find that training-time RTC outperforms inference-time RTC at higher inference delays. In real-world experiments on box building and espresso making tasks with the $π_{0.6}$ VLA, we demonstrate that training-time RTC maintains both task performance and speed parity with inference-time RTC while being computationally cheaper. Our results suggest that training-time action conditioning is a practical drop-in replacement for inference-time inpainting in real-time robot control.
>
---
#### [new 007] SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对视觉语言模型缺乏物理动态理解的问题，提出SIMPACT框架，通过在测试时引入仿真增强VLM的物理推理能力，实现细粒度机器人操作任务的行动规划，提升在真实世界复杂操作任务中的表现。**

- **链接: [https://arxiv.org/pdf/2512.05955v1](https://arxiv.org/pdf/2512.05955v1)**

> **作者:** Haowen Liu; Shaoxiong Yao; Haonan Chen; Jiawei Gao; Jiayuan Mao; Jia-Bin Huang; Yilun Du
>
> **摘要:** Vision-Language Models (VLMs) exhibit remarkable common-sense and semantic reasoning capabilities. However, they lack a grounded understanding of physical dynamics. This limitation arises from training VLMs on static internet-scale visual-language data that contain no causal interactions or action-conditioned changes. Consequently, it remains challenging to leverage VLMs for fine-grained robotic manipulation tasks that require physical understanding, reasoning, and corresponding action planning. To overcome this, we present SIMPACT, a test-time, SIMulation-enabled ACTion Planning framework that equips VLMs with physical reasoning through simulation-in-the-loop world modeling, without requiring any additional training. From a single RGB-D observation, SIMPACT efficiently constructs physics simulations, enabling the VLM to propose informed actions, observe simulated rollouts, and iteratively refine its reasoning. By integrating language reasoning with physics prediction, our simulation-enabled VLM can understand contact dynamics and action outcomes in a physically grounded way. Our method demonstrates state-of-the-art performance on five challenging, real-world rigid-body and deformable manipulation tasks that require fine-grained physical reasoning, outperforming existing general-purpose robotic manipulation models. Our results demonstrate that embedding physics understanding via efficient simulation into VLM reasoning at test time offers a promising path towards generalizable embodied intelligence. Project webpage can be found at https://simpact-bot.github.io
>
---
#### [new 008] A Comprehensive Framework for Automated Quality Control in the Automotive Industry
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于工业质检任务，旨在解决汽车制造中铝压铸件表面缺陷检测的自动化问题。作者提出了一种基于双协作机器人与YOLO11n模型的视觉系统，结合优化成像与图像处理技术，实现高精度、低误检的实时缺陷识别与定位。**

- **链接: [https://arxiv.org/pdf/2512.05579v1](https://arxiv.org/pdf/2512.05579v1)**

> **作者:** Panagiota Moraiti; Panagiotis Giannikos; Athanasios Mastrogeorgiou; Panagiotis Mavridis; Linghao Zhou; Panagiotis Chatzakos
>
> **摘要:** This paper presents a cutting-edge robotic inspection solution designed to automate quality control in automotive manufacturing. The system integrates a pair of collaborative robots, each equipped with a high-resolution camera-based vision system to accurately detect and localize surface and thread defects in aluminum high-pressure die casting (HPDC) automotive components. In addition, specialized lenses and optimized lighting configurations are employed to ensure consistent and high-quality image acquisition. The YOLO11n deep learning model is utilized, incorporating additional enhancements such as image slicing, ensemble learning, and bounding-box merging to significantly improve performance and minimize false detections. Furthermore, image processing techniques are applied to estimate the extent of the detected defects. Experimental results demonstrate real-time performance with high accuracy across a wide variety of defects, while minimizing false detections. The proposed solution is promising and highly scalable, providing the flexibility to adapt to various production environments and meet the evolving demands of the automotive industry.
>
---
#### [new 009] Toward Efficient and Robust Behavior Models for Multi-Agent Driving Simulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多智能体驾驶仿真任务，旨在解决行为模型的效率与真实性平衡问题。作者提出实例中心的场景表示和对称上下文编码器，结合自适应奖励的逆强化学习方法，提升仿真效率、准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.05812v1](https://arxiv.org/pdf/2512.05812v1)**

> **作者:** Fabian Konstantinidis; Moritz Sackmann; Ulrich Hofmann; Christoph Stiller
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Scalable multi-agent driving simulation requires behavior models that are both realistic and computationally efficient. We address this by optimizing the behavior model that controls individual traffic participants. To improve efficiency, we adopt an instance-centric scene representation, where each traffic participant and map element is modeled in its own local coordinate frame. This design enables efficient, viewpoint-invariant scene encoding and allows static map tokens to be reused across simulation steps. To model interactions, we employ a query-centric symmetric context encoder with relative positional encodings between local frames. We use Adversarial Inverse Reinforcement Learning to learn the behavior model and propose an adaptive reward transformation that automatically balances robustness and realism during training. Experiments demonstrate that our approach scales efficiently with the number of tokens, significantly reducing training and inference times, while outperforming several agent-centric baselines in terms of positional accuracy and robustness.
>
---
#### [new 010] An Integrated System for WEEE Sorting Employing X-ray Imaging, AI-based Object Detection and Segmentation, and Delta Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文针对废旧电子产品中电池回收的安全隐患，提出一种融合X射线双能成像、AI检测分割与Delta机器人抓取的自动分拣系统，实现对多种WEEE中电池的精准识别与自主分离。**

- **链接: [https://arxiv.org/pdf/2512.05599v1](https://arxiv.org/pdf/2512.05599v1)**

> **作者:** Panagiotis Giannikos; Lampis Papakostas; Evangelos Katralis; Panagiotis Mavridis; George Chryssinas; Myrto Inglezou; Nikolaos Panagopoulos; Antonis Porichis; Athanasios Mastrogeorgiou; Panagiotis Chatzakos
>
> **摘要:** Battery recycling is becoming increasingly critical due to the rapid growth in battery usage and the limited availability of natural resources. Moreover, as battery energy densities continue to rise, improper handling during recycling poses significant safety hazards, including potential fires at recycling facilities. Numerous systems have been proposed for battery detection and removal from WEEE recycling lines, including X-ray and RGB-based visual inspection methods, typically driven by AI-powered object detection models (e.g., Mask R-CNN, YOLO, ResNets). Despite advances in optimizing detection techniques and model modifications, a fully autonomous solution capable of accurately identifying and sorting batteries across diverse WEEEs types has yet to be realized. In response to these challenges, we present our novel approach which integrates a specialized X-ray transmission dual energy imaging subsystem with advanced pre-processing algorithms, enabling high-contrast image reconstruction for effective differentiation of dense and thin materials in WEEE. Devices move along a conveyor belt through a high-resolution X-ray imaging system, where YOLO and U-Net models precisely detect and segment battery-containing items. An intelligent tracking and position estimation algorithm then guides a Delta robot equipped with a suction gripper to selectively extract and properly discard the targeted devices. The approach is validated in a photorealistic simulation environment developed in NVIDIA Isaac Sim and on the real setup.
>
---
#### [new 011] Bayesian Active Inference for Intelligent UAV Anti-Jamming and Adaptive Trajectory Planning
- **分类: cs.RO; cs.AI; eess.SP; eess.SY**

- **简介: 该论文研究无人机在干扰环境下的轨迹规划任务，旨在解决未知干扰源下的通信保持与路径优化问题。提出基于贝叶斯主动推理的分层框架，融合专家示范与概率建模，实现干扰预测、干扰源定位与自适应导航。**

- **链接: [https://arxiv.org/pdf/2512.05711v1](https://arxiv.org/pdf/2512.05711v1)**

> **作者:** Ali Krayani; Seyedeh Fatemeh Sadati; Lucio Marcenaro; Carlo Regazzoni
>
> **备注:** This paper has been accepted for the 2026 IEEE Consumer Communications & Networking Conference (IEEE CCNC 2026)
>
> **摘要:** This paper proposes a hierarchical trajectory planning framework for UAVs operating under adversarial jamming conditions. Leveraging Bayesian Active Inference, the approach combines expert-generated demonstrations with probabilistic generative modeling to encode high-level symbolic planning, low-level motion policies, and wireless signal feedback. During deployment, the UAV performs online inference to anticipate interference, localize jammers, and adapt its trajectory accordingly, without prior knowledge of jammer locations. Simulation results demonstrate that the proposed method achieves near-expert performance, significantly reducing communication interference and mission cost compared to model-free reinforcement learning baselines, while maintaining robust generalization in dynamic environments.
>
---
#### [new 012] Optimal Safety-Aware Scheduling for Multi-Agent Aerial 3D Printing with Utility Maximization under Dependency Constraints
- **分类: cs.RO**

- **简介: 该论文研究多无人机协同空中3D打印的任务规划问题，旨在解决任务依赖、安全冲突与资源约束下的调度优化。提出一种兼顾安全性、效用最大化与任务优先级的优化框架，并通过仿真验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.05815v1](https://arxiv.org/pdf/2512.05815v1)**

> **作者:** Marios-Nektarios Stamatopoulos; Shridhar Velhal; Avijit Banerjee; George Nikolakopoulos
>
> **摘要:** This article presents a novel coordination and task-planning framework to enable the simultaneous conflict-free collaboration of multiple unmanned aerial vehicles (UAVs) for aerial 3D printing. The proposed framework formulates an optimization problem that takes a construction mission divided into sub-tasks and a team of autonomous UAVs, along with limited volume and battery. It generates an optimal mission plan comprising task assignments and scheduling while accounting for task dependencies arising from the geometric and structural requirements of the 3D design, inter-UAV safety constraints, material usage, and total flight time of each UAV. The potential conflicts occurring during the simultaneous operation of the UAVs are addressed at a segment level by dynamically selecting the starting time and location of each task to guarantee collision-free parallel execution. An importance prioritization is proposed to accelerate the computation by guiding the solution toward more important tasks. Additionally, a utility maximization formulation is proposed to dynamically determine the optimal number of UAVs required for a given mission, balancing the trade-off between minimizing makespan and the deployment of excess agents. The proposed framework's effectiveness is evaluated through a Gazebo-based simulation setup, where agents are coordinated by a mission control module allocating the printing tasks based on the generated optimal scheduling plan while remaining within the material and battery constraints of each UAV.
>
---
#### [new 013] Physically-Based Simulation of Automotive LiDAR
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究自动驾驶激光雷达的物理仿真，旨在准确模拟飞行时间激光雷达的光学特性。通过结合近红外物理渲染与实验室测量，建模光束扩展、回波脉宽、环境光等效应，并针对实际系统进行参数标定与验证。**

- **链接: [https://arxiv.org/pdf/2512.05932v1](https://arxiv.org/pdf/2512.05932v1)**

> **作者:** L. Dudzik; M. Roschani; A. Sielemann; K. Trampert; J. Ziehn; J. Beyerer; C. Neumann
>
> **摘要:** We present an analytic model for simulating automotive time-of-flight (ToF) LiDAR that includes blooming, echo pulse width, and ambient light, along with steps to determine model parameters systematically through optical laboratory measurements. The model uses physically based rendering (PBR) in the near-infrared domain. It assumes single-bounce reflections and retroreflections over rasterized rendered images from shading or ray tracing, including light emitted from the sensor as well as stray light from other, non-correlated sources such as sunlight. Beams from the sensor and sensitivity of the receiving diodes are modeled with flexible beam steering patterns and with non-vanishing diameter. Different (all non-real time) computational approaches can be chosen based on system properties, computing capabilities, and desired output properties. Model parameters include system-specific properties, namely the physical spread of the LiDAR beam, combined with the sensitivity of the receiving diode; the intensity of the emitted light; the conversion between the intensity of reflected light and the echo pulse width; and scenario parameters such as environment lighting, positioning, and surface properties of the target(s) in the relevant infrared domain. System-specific properties of the model are determined from laboratory measurements of the photometric luminance on different target surfaces aligned with a goniometer at 0.01° resolution, which marks the best available resolution for measuring the beam pattern. The approach is calibrated for and tested on two automotive LiDAR systems, the Valeo Scala Gen. 2 and the Blickfeld Cube 1. Both systems differ notably in their properties and available interfaces, but the relevant model parameters could be extracted successfully.
>
---
#### [new 014] A Hyperspectral Imaging Guided Robotic Grasping System
- **分类: cs.RO**

- **简介: 该论文研究机器人抓取任务，旨在解决复杂环境中物体识别与抓取精度低的问题。提出PRISM成像机制和SpectralGrasp框架，融合高光谱图像的空谱信息，提升纺织品识别和分拣成功率，实现优于人类和RGB方法的性能。**

- **链接: [https://arxiv.org/pdf/2512.05578v1](https://arxiv.org/pdf/2512.05578v1)**

> **作者:** Zheng Sun; Zhipeng Dong; Shixiong Wang; Zhongyi Chu; Fei Chen
>
> **备注:** 8 pages, 7 figures, Accepted to IEEE Robotics and Automation Letters (RA-L) 2025
>
> **摘要:** Hyperspectral imaging is an advanced technique for precisely identifying and analyzing materials or objects. However, its integration with robotic grasping systems has so far been explored due to the deployment complexities and prohibitive costs. Within this paper, we introduce a novel hyperspectral imaging-guided robotic grasping system. The system consists of PRISM (Polyhedral Reflective Imaging Scanning Mechanism) and the SpectralGrasp framework. PRISM is designed to enable high-precision, distortion-free hyperspectral imaging while simplifying system integration and costs. SpectralGrasp generates robotic grasping strategies by effectively leveraging both the spatial and spectral information from hyperspectral images. The proposed system demonstrates substantial improvements in both textile recognition compared to human performance and sorting success rate compared to RGB-based methods. Additionally, a series of comparative experiments further validates the effectiveness of our system. The study highlights the potential benefits of integrating hyperspectral imaging with robotic grasping systems, showcasing enhanced recognition and grasping capabilities in complex and dynamic environments. The project is available at: https://zainzh.github.io/PRISM.
>
---
#### [new 015] 3D Path Planning for Robot-assisted Vertebroplasty from Arbitrary Bi-plane X-ray via Differentiable Rendering
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机器人辅助椎体成形术中的3D路径规划，旨在无需术前CT扫描。通过可微渲染与统计形状模型，利用双平面X光实现任意视角下的椎体重建与路径规划，解决了传统方法依赖CT配准的问题。**

- **链接: [https://arxiv.org/pdf/2512.05803v1](https://arxiv.org/pdf/2512.05803v1)**

> **作者:** Blanca Inigo; Benjamin D. Killeen; Rebecca Choi; Michelle Song; Ali Uneri; Majid Khan; Christopher Bailey; Axel Krieger; Mathias Unberath
>
> **摘要:** Robotic systems are transforming image-guided interventions by enhancing accuracy and minimizing radiation exposure. A significant challenge in robotic assistance lies in surgical path planning, which often relies on the registration of intraoperative 2D images with preoperative 3D CT scans. This requirement can be burdensome and costly, particularly in procedures like vertebroplasty, where preoperative CT scans are not routinely performed. To address this issue, we introduce a differentiable rendering-based framework for 3D transpedicular path planning utilizing bi-planar 2D X-rays. Our method integrates differentiable rendering with a vertebral atlas generated through a Statistical Shape Model (SSM) and employs a learned similarity loss to refine the SSM shape and pose dynamically, independent of fixed imaging geometries. We evaluated our framework in two stages: first, through vertebral reconstruction from orthogonal X-rays for benchmarking, and second, via clinician-in-the-loop path planning using arbitrary-view X-rays. Our results indicate that our method outperformed a normalized cross-correlation baseline in reconstruction metrics (DICE: 0.75 vs. 0.65) and achieved comparable performance to the state-of-the-art model ReVerteR (DICE: 0.77), while maintaining generalization to arbitrary views. Success rates for bipedicular planning reached 82% with synthetic data and 75% with cadaver data, exceeding the 66% and 31% rates of a 2D-to-3D baseline, respectively. In conclusion, our framework facilitates versatile, CT-free 3D path planning for robot-assisted vertebroplasty, effectively accommodating real-world imaging diversity without the need for preoperative CT scans.
>
---
#### [new 016] Scenario-aware Uncertainty Quantification for Trajectory Prediction with Statistical Guarantees
- **分类: cs.RO; eess.SY**

- **简介: 该论文属自动驾驶轨迹预测任务，旨在解决现有方法缺乏场景自适应不确定性量化的问题。作者提出一种场景感知框架，结合Frenet坐标系、CopulaCPTS校准与可靠性判别器，生成具统计保证的预测区间并评估轨迹可靠性。**

- **链接: [https://arxiv.org/pdf/2512.05682v1](https://arxiv.org/pdf/2512.05682v1)**

> **作者:** Yiming Shu; Jiahui Xu; Linghuan Kong; Fangni Zhang; Guodong Yin; Chen Sun
>
> **摘要:** Reliable uncertainty quantification in trajectory prediction is crucial for safety-critical autonomous driving systems, yet existing deep learning predictors lack uncertainty-aware frameworks adaptable to heterogeneous real-world scenarios. To bridge this gap, we propose a novel scenario-aware uncertainty quantification framework to provide the predicted trajectories with prediction intervals and reliability assessment. To begin with, predicted trajectories from the trained predictor and their ground truth are projected onto the map-derived reference routes within the Frenet coordinate system. We then employ CopulaCPTS as the conformal calibration method to generate temporal prediction intervals for distinct scenarios as the uncertainty measure. Building upon this, within the proposed trajectory reliability discriminator (TRD), mean error and calibrated confidence intervals are synergistically analyzed to establish reliability models for different scenarios. Subsequently, the risk-aware discriminator leverages a joint risk model that integrates longitudinal and lateral prediction intervals within the Frenet coordinate to identify critical points. This enables segmentation of trajectories into reliable and unreliable segments, holding the advantage of informing downstream planning modules with actionable reliability results. We evaluated our framework using the real-world nuPlan dataset, demonstrating its effectiveness in scenario-aware uncertainty quantification and reliability assessment across diverse driving contexts.
>
---
#### [new 017] Seabed-to-Sky Mapping of Maritime Environments with a Dual Orthogonal SONAR and LiDAR Sensor Suite
- **分类: cs.RO**

- **简介: 该论文针对海上环境“ seabed-to-sky”一体化建图任务，解决GNSS依赖与昂贵传感器问题。提出一种融合LiDAR-IMU与双正交前视声呐的系统，实现无GNSS的连续三维统一建图，并通过改进LIO-SAM实现实时稠密映射。**

- **链接: [https://arxiv.org/pdf/2512.05303v1](https://arxiv.org/pdf/2512.05303v1)**

> **作者:** Christian Westerdahl; Jonas Poulsen; Daniel Holmelund; Peter Nicholas Hansen; Fletcher Thompson; Roberto Galeazzi
>
> **摘要:** Critical maritime infrastructure increasingly demands situational awareness both above and below the surface, yet existing ''seabed-to-sky'' mapping pipelines either rely on GNSS (vulnerable to shadowing/spoofing) or expensive bathymetric sonars. We present a unified, GNSS-independent mapping system that fuses LiDAR-IMU with a dual, orthogonally mounted Forward Looking Sonars (FLS) to generate consistent seabed-to-sky maps from an Autonomous Surface Vehicle. On the acoustic side, we extend orthogonal wide-aperture fusion to handle arbitrary inter-sonar translations (enabling heterogeneous, non-co-located models) and extract a leading edge from each FLS to form line-scans. On the mapping side, we modify LIO-SAM to ingest both stereo-derived 3D sonar points and leading-edge line-scans at and between keyframes via motion-interpolated poses, allowing sparse acoustic updates to contribute continuously to a single factor-graph map. We validate the system on real-world data from Belvederekanalen (Copenhagen), demonstrating real-time operation with approx. 2.65 Hz map updates and approx. 2.85 Hz odometry while producing a unified 3D model that spans air-water domains.
>
---
#### [new 018] Real-time Remote Tracking and Autonomous Planning for Whale Rendezvous using Robots
- **分类: cs.RO**

- **简介: 该论文致力于实现海上 sperm whale 的实时远程追踪与机器人自主会合任务，解决多鲸声学跟踪、分布式通信与决策及远距离信号处理难题，提出基于模型的强化学习方法，结合传感器数据与鲸鱼潜水模型指导无人机导航，并通过实地实验与仿真验证系统有效性。**

- **链接: [https://arxiv.org/pdf/2512.05808v1](https://arxiv.org/pdf/2512.05808v1)**

> **作者:** Sushmita Bhattacharya; Ninad Jadhav; Hammad Izhar; Karen Li; Kevin George; Robert Wood; Stephanie Gil
>
> **摘要:** We introduce a system for real-time sperm whale rendezvous at sea using an autonomous uncrewed aerial vehicle. Our system employs model-based reinforcement learning that combines in situ sensor data with an empirical whale dive model to guide navigation decisions. Key challenges include (i) real-time acoustic tracking in the presence of multiple whales, (ii) distributed communication and decision-making for robot deployments, and (iii) on-board signal processing and long-range detection from fish-trackers. We evaluate our system by conducting rendezvous with sperm whales at sea in Dominica, performing hardware experiments on land, and running simulations using whale trajectories interpolated from marine biologists' surface observations.
>
---
#### [new 019] Spatiotemporal Tubes for Differential Drive Robots with Model Uncertainty
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对存在模型不确定性和外部干扰的差速驱动机器人，提出一种基于时空管（STT）的控制框架，解决满足时序可达-避障-停留（T-RAS）的任务。通过构建动态安全通道并设计闭环控制律，确保安全性与鲁棒性，无需在线优化，具有高效性与形式化保证。**

- **链接: [https://arxiv.org/pdf/2512.05495v1](https://arxiv.org/pdf/2512.05495v1)**

> **作者:** Ratnangshu Das; Ahan Basu; Christos Verginis; Pushpak Jagtap
>
> **摘要:** This paper presents a Spatiotemporal Tube (STT)-based control framework for differential-drive mobile robots with dynamic uncertainties and external disturbances, guaranteeing the satisfaction of Temporal Reach-Avoid-Stay (T-RAS) specifications. The approach employs circular STT, characterized by smoothly time-varying center and radius, to define dynamic safe corridors that guide the robot from the start region to the goal while avoiding obstacles. In particular, we first develop a sampling-based synthesis algorithm to construct a feasible STT that satisfies the prescribed timing and safety constraints with formal guarantees. To ensure that the robot remains confined within this tube, we then design analytically a closed-form, approximation-free control law. The resulting controller is computationally efficient, robust to disturbances and {model uncertainties}, and requires no model approximations or online optimization. The proposed framework is validated through simulation studies on a differential-drive robot and benchmarked against state-of-the-art methods, demonstrating superior robustness, accuracy, and computational efficiency.
>
---
#### [new 020] Search at Scale: Improving Numerical Conditioning of Ergodic Coverage Optimization for Multi-Scale Domains
- **分类: cs.RO**

- **简介: 该论文属于机器人覆盖路径规划任务，旨在解决多尺度域下遍历性覆盖优化中因数值缩放导致的数值不稳定问题。提出基于MMD度量的自适应、尺度无关优化方法，通过退火超参数和对数空间度量改进数值条件。**

- **链接: [https://arxiv.org/pdf/2512.05229v1](https://arxiv.org/pdf/2512.05229v1)**

> **作者:** Yanis Lahrach; Christian Hughes; Ian Abraham
>
> **摘要:** Recent methods in ergodic coverage planning have shown promise as tools that can adapt to a wide range of geometric coverage problems with general constraints, but are highly sensitive to the numerical scaling of the problem space. The underlying challenge is that the optimization formulation becomes brittle and numerically unstable with changing scales, especially under potentially nonlinear constraints that impose dynamic restrictions, due to the kernel-based formulation. This paper proposes to address this problem via the development of a scale-agnostic and adaptive ergodic coverage optimization method based on the maximum mean discrepancy metric (MMD). Our approach allows the optimizer to solve for the scale of differential constraints while annealing the hyperparameters to best suit the problem domain and ensure physical consistency. We also derive a variation of the ergodic metric in the log space, providing additional numerical conditioning without loss of performance. We compare our approach with existing coverage planning methods and demonstrate the utility of our approach on a wide range of coverage problems.
>
---
#### [new 021] State-Conditional Adversarial Learning: An Off-Policy Visual Domain Transfer Method for End-to-End Imitation Learning
- **分类: cs.RO**

- **简介: 该论文研究端到端模仿学习中的视觉域迁移，解决目标域数据稀缺、无专家、非策略的问题。提出状态条件对抗学习（SCAL），通过状态条件下的潜在分布对齐实现跨域迁移，在自动驾驶仿真中验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2512.05335v1](https://arxiv.org/pdf/2512.05335v1)**

> **作者:** Yuxiang Liu; Shengfan Cao
>
> **摘要:** We study visual domain transfer for end-to-end imitation learning in a realistic and challenging setting where target-domain data are strictly off-policy, expert-free, and scarce. We first provide a theoretical analysis showing that the target-domain imitation loss can be upper bounded by the source-domain loss plus a state-conditional latent KL divergence between source and target observation models. Guided by this result, we propose State- Conditional Adversarial Learning, an off-policy adversarial framework that aligns latent distributions conditioned on system state using a discriminator-based estimator of the conditional KL term. Experiments on visually diverse autonomous driving environments built on the BARC-CARLA simulator demonstrate that SCAL achieves robust transfer and strong sample efficiency.
>
---
#### [new 022] XR-DT: Extended Reality-Enhanced Digital Twin for Agentic Mobile Robots
- **分类: cs.RO; cs.AI; cs.HC; cs.MA; eess.SY**

- **简介: 该论文提出XR-DT框架，属人机交互任务，旨在提升移动机器人在共享空间中的安全与可信交互。通过融合扩展现实与数字孪生技术，结合多模态大模型推理与多智能体协同，实现人类意图理解、环境感知与机器人自适应决策的双向可解释交互。**

- **链接: [https://arxiv.org/pdf/2512.05270v1](https://arxiv.org/pdf/2512.05270v1)**

> **作者:** Tianyi Wang; Jiseop Byeon; Ahmad Yehia; Huihai Wang; Yiming Xu; Tianyi Zeng; Ziran Wang; Junfeng Jiao; Christian Claudel
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** As mobile robots increasingly operate alongside humans in shared workspaces, ensuring safe, efficient, and interpretable Human-Robot Interaction (HRI) has become a pressing challenge. While substantial progress has been devoted to human behavior prediction, limited attention has been paid to how humans perceive, interpret, and trust robots' inferences, impeding deployment in safety-critical and socially embedded environments. This paper presents XR-DT, an eXtended Reality-enhanced Digital Twin framework for agentic mobile robots, that bridges physical and virtual spaces to enable bi-directional understanding between humans and robots. Our hierarchical XR-DT architecture integrates virtual-, augmented-, and mixed-reality layers, fusing real-time sensor data, simulated environments in the Unity game engine, and human feedback captured through wearable AR devices. Within this framework, we design an agentic mobile robot system with a unified diffusion policy for context-aware task adaptation. We further propose a chain-of-thought prompting mechanism that allows multimodal large language models to reason over human instructions and environmental context, while leveraging an AutoGen-based multi-agent coordination layer to enhance robustness and collaboration in dynamic tasks. Initial experimental results demonstrate accurate human and robot trajectory prediction, validating the XR-DT framework's effectiveness in HRI tasks. By embedding human intention, environmental dynamics, and robot cognition into the XR-DT framework, our system enables interpretable, trustworthy, and adaptive HRI.
>
---
#### [new 023] Invariance Co-training for Robot Visual Generalization
- **分类: cs.RO; cs.AI**

- **简介: 该论文属机器人视觉泛化任务，旨在解决现有策略对视角、光照等变化敏感的问题。提出通过状态相似性和观测不变性辅助任务，协同训练真实演示与合成图像数据，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.05230v1](https://arxiv.org/pdf/2512.05230v1)**

> **作者:** Jonathan Yang; Chelsea Finn; Dorsa Sadigh
>
> **备注:** 14 pages, 10 figures
>
> **摘要:** Reasoning from diverse observations is a fundamental capability for generalist robot policies to operate in a wide range of environments. Despite recent advancements, many large-scale robotic policies still remain sensitive to key sources of observational variation such as changes in camera perspective, lighting, and the presence of distractor objects. We posit that the limited generalizability of these models arises from the substantial diversity required to robustly cover these quasistatic axes, coupled with the current scarcity of large-scale robotic datasets that exhibit rich variation across them. In this work, we propose to systematically examine what robots need to generalize across these challenging axes by introducing two key auxiliary tasks, state similarity and invariance to observational perturbations, applied to both demonstration data and static visual data. We then show that via these auxiliary tasks, leveraging both more-expensive robotic demonstration data and less-expensive, visually rich synthetic images generated from non-physics-based simulation (for example, Unreal Engine) can lead to substantial increases in generalization to unseen camera viewpoints, lighting configurations, and distractor conditions. Our results demonstrate that co-training on this diverse data improves performance by 18 percent over existing generative augmentation methods. For more information and videos, please visit https://invariance-cotraining.github.io
>
---
#### [new 024] Synset Signset Germany: a Synthetic Dataset for German Traffic Sign Recognition
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对交通标志识别任务，提出合成数据集Synset Signset Germany。结合GAN纹理生成与解析式场景建模，生成含真实磨损和光照变化的德国交通标志图像，支持XAI与鲁棒性测试，覆盖211类标志并提供丰富标注与元数据。**

- **链接: [https://arxiv.org/pdf/2512.05936v1](https://arxiv.org/pdf/2512.05936v1)**

> **作者:** Anne Sielemann; Lena Loercher; Max-Lion Schumacher; Stefan Wolf; Masoud Roschani; Jens Ziehn
>
> **备注:** 8 pages, 8 figures, 3 tables
>
> **摘要:** In this paper, we present a synthesis pipeline and dataset for training / testing data in the task of traffic sign recognition that combines the advantages of data-driven and analytical modeling: GAN-based texture generation enables data-driven dirt and wear artifacts, rendering unique and realistic traffic sign surfaces, while the analytical scene modulation achieves physically correct lighting and allows detailed parameterization. In particular, the latter opens up applications in the context of explainable AI (XAI) and robustness tests due to the possibility of evaluating the sensitivity to parameter changes, which we demonstrate with experiments. Our resulting synthetic traffic sign recognition dataset Synset Signset Germany contains a total of 105500 images of 211 different German traffic sign classes, including newly published (2020) and thus comparatively rare traffic signs. In addition to a mask and a segmentation image, we also provide extensive metadata including the stochastically selected environment and imaging effect parameters for each image. We evaluate the degree of realism of Synset Signset Germany on the real-world German Traffic Sign Recognition Benchmark (GTSRB) and in comparison to CATERED, a state-of-the-art synthetic traffic sign recognition dataset.
>
---
#### [new 025] World Models That Know When They Don't Know: Controllable Video Generation with Calibrated Uncertainty
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文研究可控视频生成中的不确定性量化，旨在解决模型幻觉问题。提出C3方法，通过评分规则、潜在空间不确定性估计与像素级映射，实现细粒度、校准的置信度预测，支持分布内校准与异常检测。**

- **链接: [https://arxiv.org/pdf/2512.05927v1](https://arxiv.org/pdf/2512.05927v1)**

> **作者:** Zhiting Mei; Tenny Yin; Micah Baker; Ola Shorinwa; Anirudha Majumdar
>
> **摘要:** Recent advances in generative video models have led to significant breakthroughs in high-fidelity video synthesis, specifically in controllable video generation where the generated video is conditioned on text and action inputs, e.g., in instruction-guided video editing and world modeling in robotics. Despite these exceptional capabilities, controllable video models often hallucinate - generating future video frames that are misaligned with physical reality - which raises serious concerns in many tasks such as robot policy evaluation and planning. However, state-of-the-art video models lack the ability to assess and express their confidence, impeding hallucination mitigation. To rigorously address this challenge, we propose C3, an uncertainty quantification (UQ) method for training continuous-scale calibrated controllable video models for dense confidence estimation at the subpatch level, precisely localizing the uncertainty in each generated video frame. Our UQ method introduces three core innovations to empower video models to estimate their uncertainty. First, our method develops a novel framework that trains video models for correctness and calibration via strictly proper scoring rules. Second, we estimate the video model's uncertainty in latent space, avoiding training instability and prohibitive training costs associated with pixel-space approaches. Third, we map the dense latent-space uncertainty to interpretable pixel-level uncertainty in the RGB space for intuitive visualization, providing high-resolution uncertainty heatmaps that identify untrustworthy regions. Through extensive experiments on large-scale robot learning datasets (Bridge and DROID) and real-world evaluations, we demonstrate that our method not only provides calibrated uncertainty estimates within the training distribution, but also enables effective out-of-distribution detection.
>
---
#### [new 026] ARCAS: An Augmented Reality Collision Avoidance System with SLAM-Based Tracking for Enhancing VRU Safety
- **分类: eess.SY; cs.AR; cs.CV; cs.ET; cs.RO; eess.IV**

- **简介: 该论文提出ARCAS，一种基于SLAM和LiDAR的增强现实碰撞预警系统，旨在提升弱势道路使用者（VRU）安全性。通过AR头显实时叠加3D警示信息，解决混合交通中VRU碰撞风险问题，实验证明显著提升了反应时间与安全裕度。**

- **链接: [https://arxiv.org/pdf/2512.05299v1](https://arxiv.org/pdf/2512.05299v1)**

> **作者:** Ahmad Yehia; Jiseop Byeon; Tianyi Wang; Huihai Wang; Yiming Xu; Junfeng Jiao; Christian Claudel
>
> **备注:** 8 pages, 3 figures, 1 table
>
> **摘要:** Vulnerable road users (VRUs) face high collision risks in mixed traffic, yet most existing safety systems prioritize driver or vehicle assistance over direct VRU support. This paper presents ARCAS, a real-time augmented reality collision avoidance system that provides personalized spatial alerts to VRUs via wearable AR headsets. By fusing roadside 360-degree 3D LiDAR with SLAM-based headset tracking and an automatic 3D calibration procedure, ARCAS accurately overlays world-locked 3D bounding boxes and directional arrows onto approaching hazards in the user's passthrough view. The system also enables multi-headset coordination through shared world anchoring. Evaluated in real-world pedestrian interactions with e-scooters and vehicles (180 trials), ARCAS nearly doubled pedestrians' time-to-collision and increased counterparts' reaction margins by up to 4x compared to unaided-eye conditions. Results validate the feasibility and effectiveness of LiDAR-driven AR guidance and highlight the potential of wearable AR as a promising next-generation safety tool for urban mobility.
>
---
#### [new 027] A Residual Variance Matching Recursive Least Squares Filter for Real-time UAV Terrain Following
- **分类: eess.SP; cs.RO; stat.ML**

- **简介: 该论文针对无人机在野火巡逻中地形跟随的实时航点估计问题，提出一种残差方差匹配递归最小二乘（RVM-RLS）滤波方法，通过自适应调节提升非线性时变系统下的估计精度，显著提高飞行安全与火灾检测能力。**

- **链接: [https://arxiv.org/pdf/2512.05918v1](https://arxiv.org/pdf/2512.05918v1)**

> **作者:** Xiaobo Wu; Youmin Zhang
>
> **摘要:** Accurate real-time waypoints estimation for the UAV-based online Terrain Following during wildfire patrol missions is critical to ensuring flight safety and enabling wildfire detection. However, existing real-time filtering algorithms struggle to maintain accurate waypoints under measurement noise in nonlinear and time-varying systems, posing risks of flight instability and missed wildfire detections during UAV-based terrain following. To address this issue, a Residual Variance Matching Recursive Least Squares (RVM-RLS) filter, guided by a Residual Variance Matching Estimation (RVME) criterion, is proposed to adaptively estimate the real-time waypoints of nonlinear, time-varying UAV-based terrain following systems. The proposed method is validated using a UAV-based online terrain following system within a simulated terrain environment. Experimental results show that the RVM-RLS filter improves waypoints estimation accuracy by approximately 88$\%$ compared with benchmark algorithms across multiple evaluation metrics. These findings demonstrate both the methodological advances in real-time filtering and the practical potential of the RVM-RLS filter for UAV-based online wildfire patrol.
>
---
#### [new 028] Measuring the Effect of Background on Classification and Feature Importance in Deep Learning for AV Perception
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文研究自动驾驶感知中深度学习模型的可解释性，旨在量化背景特征对分类和特征重要性的影响。通过构建六种不同背景相关性和相机变化的合成交通标志数据集，分析模型是否依赖物体或背景进行分类，揭示训练域变化下背景特征的重要性变化。**

- **链接: [https://arxiv.org/pdf/2512.05937v1](https://arxiv.org/pdf/2512.05937v1)**

> **作者:** Anne Sielemann; Valentin Barner; Stefan Wolf; Masoud Roschani; Jens Ziehn; Juergen Beyerer
>
> **备注:** 8 pages, 2 figures, 7 tables
>
> **摘要:** Common approaches to explainable AI (XAI) for deep learning focus on analyzing the importance of input features on the classification task in a given model: saliency methods like SHAP and GradCAM are used to measure the impact of spatial regions of the input image on the classification result. Combined with ground truth information about the location of the object in the input image (e.g., a binary mask), it is determined whether object pixels had a high impact on the classification result, or whether the classification focused on background pixels. The former is considered to be a sign of a healthy classifier, whereas the latter is assumed to suggest overfitting on spurious correlations. A major challenge, however, is that these intuitive interpretations are difficult to test quantitatively, and hence the output of such explanations lacks an explanation itself. One particular reason is that correlations in real-world data are difficult to avoid, and whether they are spurious or legitimate is debatable. Synthetic data in turn can facilitate to actively enable or disable correlations where desired but often lack a sufficient quantification of realism and stochastic properties. [...] Therefore, we systematically generate six synthetic datasets for the task of traffic sign recognition, which differ only in their degree of camera variation and background correlation [...] to quantify the isolated influence of background correlation, different levels of camera variation, and considered traffic sign shapes on the classification performance, as well as background feature importance. [...] Results include a quantification of when and how much background features gain importance to support the classification task based on changes in the training domain [...]. Download: synset.de/datasets/synset-signset-ger/background-effect
>
---
#### [new 029] AREA3D: Active Reconstruction Agent with Unified Feed-Forward 3D Perception and Vision-Language Guidance
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文研究主动3D重建任务，旨在解决传统方法依赖手工几何规则导致视图冗余的问题。作者提出AREA3D，结合前馈3D感知与视觉-语言引导，解耦不确定性建模并引入语义指导，提升稀疏视角下的重建精度。**

- **链接: [https://arxiv.org/pdf/2512.05131v1](https://arxiv.org/pdf/2512.05131v1)**

> **作者:** Tianling Xu; Shengzhe Gan; Leslie Gu; Yuelei Li; Fangneng Zhan; Hanspeter Pfister
>
> **备注:** Under review
>
> **摘要:** Active 3D reconstruction enables an agent to autonomously select viewpoints to efficiently obtain accurate and complete scene geometry, rather than passively reconstructing scenes from pre-collected images. However, existing active reconstruction methods often rely on hand-crafted geometric heuristics, which can lead to redundant observations without substantially improving reconstruction quality. To address this limitation, we propose AREA3D, an active reconstruction agent that leverages feed-forward 3D reconstruction models and vision-language guidance. Our framework decouples view-uncertainty modeling from the underlying feed-forward reconstructor, enabling precise uncertainty estimation without expensive online optimization. In addition, an integrated vision-language model provides high-level semantic guidance, encouraging informative and diverse viewpoints beyond purely geometric cues. Extensive experiments on both scene-level and object-level benchmarks demonstrate that AREA3D achieves state-of-the-art reconstruction accuracy, particularly in the sparse-view regime. Code will be made available at: https://github.com/TianlingXu/AREA3D .
>
---
#### [new 030] Label-Efficient Point Cloud Segmentation with Active Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究3D点云语义分割中的标签高效标注问题，提出一种基于2D网格划分和模型集成不确定性的主动学习方法，有效减少标注成本，在多个数据集上性能优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.05759v1](https://arxiv.org/pdf/2512.05759v1)**

> **作者:** Johannes Meyer; Jasper Hoffmann; Felix Schulz; Dominik Merkle; Daniel Buescher; Alexander Reiterer; Joschka Boedecker; Wolfram Burgard
>
> **摘要:** Semantic segmentation of 3D point cloud data often comes with high annotation costs. Active learning automates the process of selecting which data to annotate, reducing the total amount of annotation needed to achieve satisfactory performance. Recent approaches to active learning for 3D point clouds are often based on sophisticated heuristics for both, splitting point clouds into annotatable regions and selecting the most beneficial for further neural network training. In this work, we propose a novel and easy-to-implement strategy to separate the point cloud into annotatable regions. In our approach, we utilize a 2D grid to subdivide the point cloud into columns. To identify the next data to be annotated, we employ a network ensemble to estimate the uncertainty in the network output. We evaluate our method on the S3DIS dataset, the Toronto-3D dataset, and a large-scale urban 3D point cloud of the city of Freiburg, which we labeled in parts manually. The extensive evaluation shows that our method yields performance on par with, or even better than, complex state-of-the-art methods on all datasets. Furthermore, we provide results suggesting that in the context of point clouds the annotated area can be a more meaningful measure for active learning algorithms than the number of annotated points.
>
---
#### [new 031] Two-Stage Camera Calibration Method for Multi-Camera Systems Using Scene Geometry
- **分类: eess.IV; cs.RO**

- **简介: 该论文属于多相机标定任务，旨在解决无精确地图、无法布设标定物等实际条件下标定困难的问题。提出两阶段方法：第一阶段利用自然几何特征估计内参和姿态初值，第二阶段通过交互调整投影视域实现精确定位，仅需单张静态图像即可完成系统标定。**

- **链接: [https://arxiv.org/pdf/2512.05171v1](https://arxiv.org/pdf/2512.05171v1)**

> **作者:** Aleksandr Abramov
>
> **摘要:** Calibration of multi-camera systems is a key task for accurate object tracking. However, it remains a challenging problem in real-world conditions, where traditional methods are not applicable due to the lack of accurate floor plans, physical access to place calibration patterns, or synchronized video streams. This paper presents a novel two-stage calibration method that overcomes these limitations. In the first stage, partial calibration of individual cameras is performed based on an operator's annotation of natural geometric primitives (parallel, perpendicular, and vertical lines, or line segments of equal length). This allows estimating key parameters (roll, pitch, focal length) and projecting the camera's Effective Field of View (EFOV) onto the horizontal plane in a base 3D coordinate system. In the second stage, precise system calibration is achieved through interactive manipulation of the projected EFOV polygons. The operator adjusts their position, scale, and rotation to align them with the floor plan or, in its absence, using virtual calibration elements projected onto all cameras in the system. This determines the remaining extrinsic parameters (camera position and yaw). Calibration requires only a static image from each camera, eliminating the need for physical access or synchronized video. The method is implemented as a practical web service. Comparative analysis and demonstration videos confirm the method's applicability, accuracy, and flexibility, enabling the deployment of precise multi-camera tracking systems in scenarios previously considered infeasible.
>
---
## 更新

#### [replaced 001] Multi-Modal Data-Efficient 3D Scene Understanding for Autonomous Driving
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文研究自动驾驶中的多模态半监督3D场景理解，旨在减少对标注LiDAR数据的依赖。提出LaserMix++框架，融合激光雷达与相机数据，通过跨模态交互、特征蒸馏和语言引导生成辅助监督，提升数据利用效率，在少标签下显著优于全监督方法。**

- **链接: [https://arxiv.org/pdf/2405.05258v3](https://arxiv.org/pdf/2405.05258v3)**

> **作者:** Lingdong Kong; Xiang Xu; Jiawei Ren; Wenwei Zhang; Liang Pan; Kai Chen; Wei Tsang Ooi; Ziwei Liu
>
> **备注:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
>
> **摘要:** Efficient data utilization is crucial for advancing 3D scene understanding in autonomous driving, where reliance on heavily human-annotated LiDAR point clouds challenges fully supervised methods. Addressing this, our study extends into semi-supervised learning for LiDAR semantic segmentation, leveraging the intrinsic spatial priors of driving scenes and multi-sensor complements to augment the efficacy of unlabeled datasets. We introduce LaserMix++, an evolved framework that integrates laser beam manipulations from disparate LiDAR scans and incorporates LiDAR-camera correspondences to further assist data-efficient learning. Our framework is tailored to enhance 3D scene consistency regularization by incorporating multi-modality, including 1) multi-modal LaserMix operation for fine-grained cross-sensor interactions; 2) camera-to-LiDAR feature distillation that enhances LiDAR feature learning; and 3) language-driven knowledge guidance generating auxiliary supervisions using open-vocabulary models. The versatility of LaserMix++ enables applications across LiDAR representations, establishing it as a universally applicable solution. Our framework is rigorously validated through theoretical analysis and extensive experiments on popular driving perception datasets. Results demonstrate that LaserMix++ markedly outperforms fully supervised alternatives, achieving comparable accuracy with five times fewer annotations and significantly improving the supervised-only baselines. This substantial advancement underscores the potential of semi-supervised approaches in reducing the reliance on extensive labeled data in LiDAR-based 3D scene understanding systems.
>
---
#### [replaced 002] ReSem3D: Refinable 3D Spatial Constraints via Fine-Grained Semantic Grounding for Generalizable Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.HC; cs.LG**

- **简介: 该论文研究机器人操作任务，旨在解决语义驱动3D空间约束中粒度粗、缺乏闭环规划和鲁棒性差的问题。提出ReSem3D框架，结合MLLM与VFM实现细粒度语义接地，分阶段构建层次化3D约束，并实时优化控制，提升泛化性与适应性。**

- **链接: [https://arxiv.org/pdf/2507.18262v3](https://arxiv.org/pdf/2507.18262v3)**

> **作者:** Chenyu Su; Weiwei Shang; Chen Qian; Fei Zhang; Shuang Cong
>
> **备注:** 12 pages,9 figures
>
> **摘要:** Semantics-driven 3D spatial constraints align highlevel semantic representations with low-level action spaces, facilitating the unification of task understanding and execution in robotic manipulation. The synergistic reasoning of Multimodal Large Language Models (MLLMs) and Vision Foundation Models (VFMs) enables cross-modal 3D spatial constraint construction. Nevertheless, existing methods have three key limitations: (1) coarse semantic granularity in constraint modeling, (2) lack of real-time closed-loop planning, (3) compromised robustness in semantically diverse environments. To address these challenges, we propose ReSem3D, a unified manipulation framework for semantically diverse environments, leveraging the synergy between VFMs and MLLMs to achieve fine-grained visual grounding and dynamically constructs hierarchical 3D spatial constraints for real-time manipulation. Specifically, the framework is driven by hierarchical recursive reasoning in MLLMs, which interact with VFMs to automatically construct 3D spatial constraints from natural language instructions and RGB-D observations in two stages: part-level extraction and region-level refinement. Subsequently, these constraints are encoded as real-time optimization objectives in joint space, enabling reactive behavior to dynamic disturbances. Extensive simulation and real-world experiments are conducted in semantically rich household and sparse chemical lab environments. The results demonstrate that ReSem3D performs diverse manipulation tasks under zero-shot conditions, exhibiting strong adaptability and generalization. Code and videos are available at https://github.com/scy-v/ReSem3D and https://resem3d.github.io.
>
---
#### [replaced 003] IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.RO**

- **简介: 该论文聚焦VLM驱动具身智能体在家庭任务中的交互安全评估。针对现有静态评测无法捕捉动态风险的问题，提出IS-Bench，首个支持多模态、过程导向的交互安全评测基准，包含161个场景与388种安全风险，揭示当前模型缺乏交互安全意识，并推动更安全AI系统发展。**

- **链接: [https://arxiv.org/pdf/2506.16402v3](https://arxiv.org/pdf/2506.16402v3)**

> **作者:** Xiaoya Lu; Zeren Chen; Xuhao Hu; Yijin Zhou; Weichen Zhang; Dongrui Liu; Lu Sheng; Jing Shao
>
> **摘要:** Flawed planning from VLM-driven embodied agents poses significant safety hazards, hindering their deployment in real-world household tasks. However, existing static, non-interactive evaluation paradigms fail to adequately assess risks within these interactive environments, since they cannot simulate dynamic risks that emerge from an agent's actions and rely on unreliable post-hoc evaluations that ignore unsafe intermediate steps. To bridge this critical gap, we propose evaluating an agent's interactive safety: its ability to perceive emergent risks and execute mitigation steps in the correct procedural order. We thus present IS-Bench, the first multi-modal benchmark designed for interactive safety, featuring 161 challenging scenarios with 388 unique safety risks instantiated in a high-fidelity simulator. Crucially, it facilitates a novel process-oriented evaluation that verifies whether risk mitigation actions are performed before/after specific risk-prone steps. Extensive experiments on leading VLMs, including the GPT-4o and Gemini-2.5 series, reveal that current agents lack interactive safety awareness, and that while safety-aware Chain-of-Thought can improve performance, it often compromises task completion. By highlighting these critical limitations, IS-Bench provides a foundation for developing safer and more reliable embodied AI systems. Code and data are released under https://github.com/AI45Lab/IS-Bench.
>
---
#### [replaced 004] GEX: Democratizing Dexterity with Fully-Actuated Dexterous Hand and Exoskeleton Glove
- **分类: cs.RO**

- **简介: 该论文提出GEX系统，解决灵巧操作机器人成本高、控制精度低的问题。通过全驱动的机械手与外骨骼手套构建闭环遥操作框架，实现低成本、高保真的灵巧运动捕捉与复现，推动具身AI与技能迁移研究。**

- **链接: [https://arxiv.org/pdf/2506.04982v2](https://arxiv.org/pdf/2506.04982v2)**

> **作者:** Yunlong Dong; Xing Liu; Jun Wan; Zelin Deng
>
> **摘要:** This paper introduces GEX, an innovative low-cost dexterous manipulation system that combines the GX11 tri-finger anthropomorphic hand (11 DoF) with the EX12 tri-finger exoskeleton glove (12 DoF), forming a closed-loop teleoperation framework through kinematic retargeting for high-fidelity control. Both components employ modular 3D-printed finger designs, achieving ultra-low manufacturing costs while maintaining full actuation capabilities. Departing from conventional tendon-driven or underactuated approaches, our electromechanical system integrates independent joint motors across all 23 DoF, ensuring complete state observability and accurate kinematic modeling. This full-actuation architecture enables precise bidirectional kinematic calculations, substantially enhancing kinematic retargeting fidelity between the exoskeleton and robotic hand. The proposed system bridges the cost-performance gap in dexterous manipulation research, providing an accessible platform for acquiring high-quality demonstration data to advance embodied AI and dexterous robotic skill transfer learning.
>
---
#### [replaced 005] Momentum-constrained Hybrid Heuristic Trajectory Optimization Framework with Residual-enhanced DRL for Visually Impaired Scenarios
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究面向视障辅助导航的轨迹规划问题，提出一种动量约束混合启发式框架（MHHTOF），结合残差增强DRL与双阶段成本建模，在保证轨迹平滑性与安全性的同时，提升优化效率与训练稳定性。**

- **链接: [https://arxiv.org/pdf/2509.15582v2](https://arxiv.org/pdf/2509.15582v2)**

> **作者:** Yuting Zeng; Zhiwen Zheng; You Zhou; JiaLing Xiao; Yongbin Yu; Manping Fan; Bo Gong; Liyong Ren
>
> **备注:** Upon further internal evaluation, we found that the current version does not adequately represent the clarity and completeness that we intend for this work. To avoid possible misunderstanding caused by this preliminary form, we request withdrawal. A refined version will be prepared privately before any further dissemination
>
> **摘要:** This paper proposes a momentum-constrained hybrid heuristic trajectory optimization framework (MHHTOF) tailored for assistive navigation in visually impaired scenarios, integrating trajectory sampling generation, optimization and evaluation with residual-enhanced deep reinforcement learning (DRL). In the first stage, heuristic trajectory sampling cluster (HTSC) is generated in the Frenet coordinate system using third-order interpolation with fifth-order polynomials and momentum-constrained trajectory optimization (MTO) constraints to ensure smoothness and feasibility. After first stage cost evaluation, the second stage leverages a residual-enhanced actor-critic network with LSTM-based temporal feature modeling to adaptively refine trajectory selection in the Cartesian coordinate system. A dual-stage cost modeling mechanism (DCMM) with weight transfer aligns semantic priorities across stages, supporting human-centered optimization. Experimental results demonstrate that the proposed LSTM-ResB-PPO achieves significantly faster convergence, attaining stable policy performance in approximately half the training iterations required by the PPO baseline, while simultaneously enhancing both reward outcomes and training stability. Compared to baseline method, the selected model reduces average cost and cost variance by 30.3% and 53.3%, and lowers ego and obstacle risks by over 77%. These findings validate the framework's effectiveness in enhancing robustness, safety, and real-time feasibility in complex assistive planning tasks.
>
---
#### [replaced 006] Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究第一视角下手部运动预测任务，旨在解决现有方法预测目标单一、模态差异大、手头运动耦合等问题。提出Uni-Hand框架，通过多模态融合、双分支扩散模型和目标指示机制，实现2D/3D手部关键点与交互状态的多目标预测，并验证其在下游任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.12878v3](https://arxiv.org/pdf/2511.12878v3)**

> **作者:** Junyi Ma; Wentao Bao; Jingyi Xu; Guanzhong Sun; Yu Zheng; Erhang Zhang; Xieyuanli Chen; Hesheng Wang
>
> **备注:** Extended journal version of MMTwin (IROS'25). Code and data: https://github.com/IRMVLab/UniHand
>
> **摘要:** Forecasting how human hands move in egocentric views is critical for applications like augmented reality and human-robot policy transfer. Recently, several hand trajectory prediction (HTP) methods have been developed to generate future possible hand waypoints, which still suffer from insufficient prediction targets, inherent modality gaps, entangled hand-head motion, and limited validation in downstream tasks. To address these limitations, we present a universal hand motion forecasting framework considering multi-modal input, multi-dimensional and multi-target prediction patterns, and multi-task affordances for downstream applications. We harmonize multiple modalities by vision-language fusion, global context incorporation, and task-aware text embedding injection, to forecast hand waypoints in both 2D and 3D spaces. A novel dual-branch diffusion is proposed to concurrently predict human head and hand movements, capturing their motion synergy in egocentric vision. By introducing target indicators, the prediction model can forecast the specific joint waypoints of the wrist or the fingers, besides the widely studied hand center points. In addition, we enable Uni-Hand to additionally predict hand-object interaction states (contact/separation) to facilitate downstream tasks better. As the first work to incorporate downstream task evaluation in the literature, we build novel benchmarks to assess the real-world applicability of hand motion forecasting algorithms. The experimental results on multiple publicly available datasets and our newly proposed benchmarks demonstrate that Uni-Hand achieves the state-of-the-art performance in multi-dimensional and multi-target hand motion forecasting. Extensive validation in multiple downstream tasks also presents its impressive human-robot policy transfer to enable robotic manipulation, and effective feature enhancement for action anticipation/recognition.
>
---
#### [replaced 007] SAT: Dynamic Spatial Aptitude Training for Multimodal Language Models
- **分类: cs.CV; cs.AI; cs.GR; cs.RO**

- **简介: 该论文聚焦多模态语言模型的空间推理能力，旨在提升其对静态与动态空间关系的理解。作者构建了基于3D仿真的SAT数据集，通过模拟生成高质量标注的静态和动态空间问答数据，有效增强了模型在真实场景中的空间认知表现。**

- **链接: [https://arxiv.org/pdf/2412.07755v3](https://arxiv.org/pdf/2412.07755v3)**

> **作者:** Arijit Ray; Jiafei Duan; Ellis Brown; Reuben Tan; Dina Bashkirova; Rose Hendrix; Kiana Ehsani; Aniruddha Kembhavi; Bryan A. Plummer; Ranjay Krishna; Kuo-Hao Zeng; Kate Saenko
>
> **备注:** Accepted to COLM 2025. Project webpage: https://arijitray.com/SAT/
>
> **摘要:** Reasoning about motion and space is a fundamental cognitive capability that is required by multiple real-world applications. While many studies highlight that large multimodal language models (MLMs) struggle to reason about space, they only focus on static spatial relationships, and not dynamic awareness of motion and space, i.e., reasoning about the effect of egocentric and object motions on spatial relationships. Manually annotating such object and camera movements is expensive. Hence, we introduce SAT, a simulated spatial aptitude training dataset utilizing 3D simulators, comprising both static and dynamic spatial reasoning across 175K question-answer (QA) pairs and 20K scenes. Complementing this, we also construct a small (150 image-QAs) yet challenging dynamic spatial test set using real-world images. Leveraging our SAT datasets and 6 existing static spatial benchmarks, we systematically investigate what improves both static and dynamic spatial awareness. Our results reveal that simulations are surprisingly effective at imparting spatial aptitude to MLMs that translate to real images. We show that perfect annotations in simulation are more effective than existing approaches of pseudo-annotating real images. For instance, SAT training improves a LLaVA-13B model by an average 11% and a LLaVA-Video-7B model by an average 8% on multiple spatial benchmarks, including our real-image dynamic test set and spatial reasoning on long videos -- even outperforming some large proprietary models. While reasoning over static relationships improves with synthetic training data, there is still considerable room for improvement for dynamic reasoning questions.
>
---
#### [replaced 008] A neural signed configuration distance function for path planning of picking manipulators
- **分类: cs.RO**

- **简介: 该论文针对抓取机械臂路径规划中碰撞检测耗时的问题，提出一种神经符号构型距离函数（nSCDF），用隐式障碍表示生成无碰撞球体，构建基于球的多查询规划器，快速生成安全走廊并优化路径。**

- **链接: [https://arxiv.org/pdf/2502.16205v2](https://arxiv.org/pdf/2502.16205v2)**

> **作者:** Bernhard Wullt; Mikael Norrlöf; Per Mattsson; Thomas B. Schön
>
> **摘要:** Picking manipulators are task specific robots, with fewer degrees of freedom compared to general-purpose manipulators, and are heavily used in industry. The efficiency of the picking robots is highly dependent on the path planning solution, which is commonly done using sampling-based multi-query methods. The planner is robustly able to solve the problem, but its heavy use of collision-detection limits the planning capabilities for online use. We approach this problem by presenting a novel implicit obstacle representation for path planning, a neural signed configuration distance function (nSCDF), which allows us to form collision-free balls in the configuration space. We use the ball representation to re-formulate a state of the art multi-query path planner, i.e., instead of points, we use balls in the graph. Our planner returns a collision-free corridor, which allows us to use convex programming to produce optimized paths. From our numerical experiments, we observe that our planner produces paths that are close to those from an asymptotically optimal path planner, in significantly less time.
>
---
#### [replaced 009] STATE-NAV: Stability-Aware Traversability Estimation for Bipedal Navigation on Rough Terrain
- **分类: cs.RO**

- **简介: 该论文研究双足机器人在非平坦地形上的导航任务，解决传统方法忽略运动稳定性的 traversability 估计问题。提出首个基于学习的稳定性感知 traversability 估计网络 TravFormer 与风险敏感导航框架，实现更鲁棒、高效的双足行走。**

- **链接: [https://arxiv.org/pdf/2506.01046v4](https://arxiv.org/pdf/2506.01046v4)**

> **作者:** Ziwon Yoon; Lawrence Y. Zhu; Jingxi Lu; Lu Gan; Ye Zhao
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Bipedal robots have advantages in maneuvering human-centered environments, but face greater failure risk compared to other stable mobile platforms such as wheeled or quadrupedal robots. While learning-based traversability has been widely studied for these platforms, bipedal traversability has instead relied on manually designed rules with limited consideration of locomotion stability on rough terrain. In this work, we present the first learning-based traversability estimation and risk-sensitive navigation framework for bipedal robots operating in diverse, uneven environments. TravFormer, a transformer-based neural network, is trained to predict bipedal instability with uncertainty, enabling risk-aware and adaptive planning. Based on the network, we define traversability as stability-aware command velocity-the fastest command velocity that keeps instability below a user-defined limit. This velocity-based traversability is integrated into a hierarchical planner that combines traversability-informed Rapid Random Tree Star (TravRRT*) for time-efficient planning and Model Predictive Control (MPC) for safe execution. We validate our method in MuJoCo simulation and the real world, demonstrating improved navigation performance, with enhanced robustness and time efficiency across varying terrains compared to existing methods.
>
---
#### [replaced 010] LLM-Driven Corrective Robot Operation Code Generation with Static Text-Based Simulation
- **分类: cs.RO**

- **简介: 该论文研究LLM生成机器人操作代码的可靠性问题，提出一种基于静态文本模拟的纠错框架。通过让LLM模拟代码执行并生成语义观测，实现无需物理实验或仿真器的高效反馈，提升代码生成准确性和部署便捷性。**

- **链接: [https://arxiv.org/pdf/2512.02002v2](https://arxiv.org/pdf/2512.02002v2)**

> **作者:** Wenhao Wang; Yi Rong; Yanyan Li; Long Jiao; Jiawei Yuan
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** Recent advances in Large language models (LLMs) have demonstrated their promising capabilities of generating robot operation code to enable LLM-driven robots. To enhance the reliability of operation code generated by LLMs, corrective designs with feedback from the observation of executing code have been increasingly adopted in existing research. However, the code execution in these designs relies on either a physical experiment or a customized simulation environment, which limits their deployment due to the high configuration effort of the environment and the potential long execution time. In this paper, we explore the possibility of directly leveraging LLM to enable static simulation of robot operation code, and then leverage it to design a new reliable LLM-driven corrective robot operation code generation framework. Our framework configures the LLM as a static simulator with enhanced capabilities that reliably simulate robot code execution by interpreting actions, reasoning over state transitions, analyzing execution outcomes, and generating semantic observations that accurately capture trajectory dynamics. To validate the performance of our framework, we performed experiments on various operation tasks for different robots, including UAVs and small ground vehicles. The experiment results not only demonstrated the high accuracy of our static text-based simulation but also the reliable code generation of our LLM-driven corrective framework, which achieves a comparable performance with state-of-the-art research while does not rely on dynamic code execution using physical experiments or simulators.
>
---
#### [replaced 011] Adaptive Keyframe Selection for Scalable 3D Scene Reconstruction in Dynamic Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究动态环境中可扩展的3D场景重建，旨在解决传统关键帧选择方法在实时感知中的数据瓶颈问题。作者提出一种自适应关键帧选择方法，结合误差评估与动量更新机制，提升重建质量，适用于机器人在复杂动态场景中的应用。**

- **链接: [https://arxiv.org/pdf/2510.23928v2](https://arxiv.org/pdf/2510.23928v2)**

> **作者:** Raman Jha; Yang Zhou; Giuseppe Loianno
>
> **备注:** Accepted at ROBOVIS 2026
>
> **摘要:** In this paper, we propose an adaptive keyframe selection method for improved 3D scene reconstruction in dynamic environments. The proposed method integrates two complementary modules: an error-based selection module utilizing photometric and structural similarity (SSIM) errors, and a momentum-based update module that dynamically adjusts keyframe selection thresholds according to scene motion dynamics. By dynamically curating the most informative frames, our approach addresses a key data bottleneck in real-time perception. This allows for the creation of high-quality 3D world representations from a compressed data stream, a critical step towards scalable robot learning and deployment in complex, dynamic environments. Experimental results demonstrate significant improvements over traditional static keyframe selection strategies, such as fixed temporal intervals or uniform frame skipping. These findings highlight a meaningful advancement toward adaptive perception systems that can dynamically respond to complex and evolving visual scenes. We evaluate our proposed adaptive keyframe selection module on two recent state-of-the-art 3D reconstruction networks, Spann3r and CUT3R, and observe consistent improvements in reconstruction quality across both frameworks. Furthermore, an extensive ablation study confirms the effectiveness of each individual component in our method, underlining their contribution to the overall performance gains.
>
---
#### [replaced 012] Perspective-Invariant 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究LiDAR-based 3D目标检测，针对非车载平台数据缺失和跨平台适应问题，构建了多平台基准Pi3DET，提出几何与特征级对齐的跨平台自适应框架，实现视角不变的3D检测，推动通用3D感知系统发展。**

- **链接: [https://arxiv.org/pdf/2507.17665v2](https://arxiv.org/pdf/2507.17665v2)**

> **作者:** Ao Liang; Lingdong Kong; Dongyue Lu; Youquan Liu; Jian Fang; Huaici Zhao; Wei Tsang Ooi
>
> **备注:** ICCV 2025; 54 pages, 18 figures, 22 tables; Project Page at https://pi3det.github.io
>
> **摘要:** With the rise of robotics, LiDAR-based 3D object detection has garnered significant attention in both academia and industry. However, existing datasets and methods predominantly focus on vehicle-mounted platforms, leaving other autonomous platforms underexplored. To bridge this gap, we introduce Pi3DET, the first benchmark featuring LiDAR data and 3D bounding box annotations collected from multiple platforms: vehicle, quadruped, and drone, thereby facilitating research in 3D object detection for non-vehicle platforms as well as cross-platform 3D detection. Based on Pi3DET, we propose a novel cross-platform adaptation framework that transfers knowledge from the well-studied vehicle platform to other platforms. This framework achieves perspective-invariant 3D detection through robust alignment at both geometric and feature levels. Additionally, we establish a benchmark to evaluate the resilience and robustness of current 3D detectors in cross-platform scenarios, providing valuable insights for developing adaptive 3D perception systems. Extensive experiments validate the effectiveness of our approach on challenging cross-platform tasks, demonstrating substantial gains over existing adaptation methods. We hope this work paves the way for generalizable and unified 3D perception systems across diverse and complex environments. Our Pi3DET dataset, cross-platform benchmark suite, and annotation toolkit have been made publicly available.
>
---
#### [replaced 013] Real-Time Execution of Action Chunking Flow Policies
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对视觉语言动作模型在实时控制中的高延迟问题，提出一种名为RTC的推理时算法，实现动作分块策略的平滑异步执行。无需重训练，即可在模拟和真实双臂操作任务中显著提升实时性与成功率。**

- **链接: [https://arxiv.org/pdf/2506.07339v2](https://arxiv.org/pdf/2506.07339v2)**

> **作者:** Kevin Black; Manuel Y. Galliker; Sergey Levine
>
> **备注:** published in NeurIPS 2025
>
> **摘要:** Modern AI systems, especially those interacting with the physical world, increasingly require real-time performance. However, the high latency of state-of-the-art generalist models, including recent vision-language action models (VLAs), poses a significant challenge. While action chunking has enabled temporal consistency in high-frequency control tasks, it does not fully address the latency problem, leading to pauses or out-of-distribution jerky movements at chunk boundaries. This paper presents a novel inference-time algorithm that enables smooth asynchronous execution of action chunking policies. Our method, real-time chunking (RTC), is applicable to any diffusion- or flow-based VLA out of the box with no re-training. It generates the next action chunk while executing the current one, "freezing" actions guaranteed to execute and "inpainting" the rest. To test RTC, we introduce a new benchmark of 12 highly dynamic tasks in the Kinetix simulator, as well as evaluate 6 challenging real-world bimanual manipulation tasks. Results demonstrate that RTC is fast, performant, and uniquely robust to inference delay, significantly improving task throughput and enabling high success rates in precise tasks $\unicode{x2013}$ such as lighting a match $\unicode{x2013}$ even in the presence of significant latency. See https://pi.website/research/real_time_chunking for videos.
>
---
#### [replaced 014] WiSER-X: Wireless Signals-based Efficient Decentralized Multi-Robot Exploration without Explicit Information Exchange
- **分类: cs.RO**

- **简介: 该论文研究多机器人协同探索任务，解决通信受限下避免覆盖重叠与协作效率问题。提出WiSER-X算法，利用无线信号估计相对位置，实现去中心化探索、异步终止与容错，无需共享地图，兼顾低重叠与高覆盖率。**

- **链接: [https://arxiv.org/pdf/2412.19876v2](https://arxiv.org/pdf/2412.19876v2)**

> **作者:** Ninad Jadhav; Meghna Behari; Robert J. Wood; Stephanie Gil
>
> **摘要:** We introduce a Wireless Signal based Efficient multi-Robot eXploration (WiSER-X) algorithm applicable to a decentralized team of robots exploring an unknown environment with communication bandwidth constraints. WiSER-X relies only on local inter-robot relative position estimates, that can be obtained by exchanging signal pings from onboard sensors such as WiFi, Ultra-Wide Band, amongst others, to inform the exploration decisions of individual robots to minimize redundant coverage overlaps. Furthermore, WiSER-X also enables asynchronous termination without requiring a shared map between the robots. It also adapts to heterogeneous robot behaviors and even complete failures in unknown environment while ensuring complete coverage. Simulations show that WiSER-X leads to 58% lower overlap than a zero-information-sharing baseline algorithm-1 and only 23% more overlap than a full-information-sharing algorithm baseline algorithm-2. Hardware experiments further validate the feasibility of WiSER-X using full onboard sensing.
>
---
#### [replaced 015] MIGHTY: Hermite Spline-based Efficient Trajectory Planning
- **分类: cs.RO**

- **简介: 该论文针对轨迹规划中硬约束方法计算量大、软约束方法优化不充分的问题，提出MIGHTY——一种基于Hermite样条的高效时空联合优化方法，在仿真和实机测试中实现了更快的计算与飞行速度。**

- **链接: [https://arxiv.org/pdf/2511.10822v2](https://arxiv.org/pdf/2511.10822v2)**

> **作者:** Kota Kondo; Yuwei Wu; Vijay Kumar; Jonathan P. How
>
> **备注:** 9 pages, 10 figures
>
> **摘要:** Hard-constraint trajectory planners often rely on commercial solvers and demand substantial computational resources. Existing soft-constraint methods achieve faster computation, but either (1) decouple spatial and temporal optimization or (2) restrict the search space. To overcome these limitations, we introduce MIGHTY, a Hermite spline-based planner that performs spatiotemporal optimization while fully leveraging the continuous search space of a spline. In simulation, MIGHTY achieves a 9.3% reduction in computation time and a 13.1% reduction in travel time over state-of-the-art baselines, with a 100% success rate. In hardware, MIGHTY completes multiple high-speed flights up to 6.7 m/s in a cluttered static environment and long-duration flights with dynamically added obstacles.
>
---
#### [replaced 016] SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control
- **分类: cs.RO; cs.AI; cs.CV; cs.GR; eess.SY**

- **简介: 该论文聚焦 humanoid 控制任务，旨在解决现有控制器泛化性差、依赖人工奖励的问题。提出 SONIC，通过扩大模型、数据与计算规模，利用动作捕捉数据进行运动跟踪，学习通用运动先验，并支持多种输入接口与实际任务迁移。**

- **链接: [https://arxiv.org/pdf/2511.07820v2](https://arxiv.org/pdf/2511.07820v2)**

> **作者:** Zhengyi Luo; Ye Yuan; Tingwu Wang; Chenran Li; Sirui Chen; Fernando Castañeda; Zi-Ang Cao; Jiefeng Li; David Minor; Qingwei Ben; Xingye Da; Runyu Ding; Cyrus Hogg; Lina Song; Edy Lim; Eugene Jeong; Tairan He; Haoru Xue; Wenli Xiao; Zi Wang; Simon Yuen; Jan Kautz; Yan Chang; Umar Iqbal; Linxi "Jim" Fan; Yuke Zhu
>
> **备注:** Project page: https://nvlabs.github.io/SONIC/
>
> **摘要:** Despite the rise of billion-parameter foundation models trained across thousands of GPUs, similar scaling gains have not been shown for humanoid control. Current neural controllers for humanoids remain modest in size, target a limited set of behaviors, and are trained on a handful of GPUs over several days. We show that scaling up model capacity, data, and compute yields a generalist humanoid controller capable of creating natural and robust whole-body movements. Specifically, we posit motion tracking as a natural and scalable task for humanoid control, leveraging dense supervision from diverse motion-capture data to acquire human motion priors without manual reward engineering. We build a foundation model for motion tracking by scaling along three axes: network size (from 1.2M to 42M parameters), dataset volume (over 100M frames, 700 hours of high-quality motion data), and compute (9k GPU hours). Beyond demonstrating the benefits of scale, we show the practical utility of our model through two mechanisms: (1) a real-time universal kinematic planner that bridges motion tracking to downstream task execution, enabling natural and interactive control, and (2) a unified token space that supports various motion input interfaces, such as VR teleoperation devices, human videos, and vision-language-action (VLA) models, all using the same policy. Scaling motion tracking exhibits favorable properties: performance improves steadily with increased compute and data diversity, and learned representations generalize to unseen motions, establishing motion tracking at scale as a practical foundation for humanoid control.
>
---
#### [replaced 017] Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究视觉-语言-动作（VLA）模型，旨在解决现有模型参数多、依赖大量机器人数据、泛化差的问题。作者提出轻量级模型Evo-1，通过新架构和两阶段训练，在不依赖机器人预训练的情况下保持语义对齐，提升性能与部署效率。**

- **链接: [https://arxiv.org/pdf/2511.04555v2](https://arxiv.org/pdf/2511.04555v2)**

> **作者:** Tao Lin; Yilei Zhong; Yuxin Du; Jingjing Zhang; Jiting Liu; Yinxinyu Chen; Encheng Gu; Ziyan Liu; Hongyi Cai; Yanwen Zou; Lixing Zou; Zhaoye Zhou; Gen Li; Bo Zhao
>
> **备注:** Github: https://github.com/MINT-SJTU/Evo-1
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a powerful framework that unifies perception, language, and control, enabling robots to perform diverse tasks through multimodal understanding. However, current VLA models typically contain massive parameters and rely heavily on large-scale robot data pretraining, leading to high computational costs during training, as well as limited deployability for real-time inference. Moreover, most training paradigms often degrade the perceptual representations of the vision-language backbone, resulting in overfitting and poor generalization to downstream tasks. In this work, we present Evo-1, a lightweight VLA model that reduces computation and improves deployment efficiency, while maintaining strong performance without pretraining on robot data. Evo-1 builds on a native multimodal Vision-Language model (VLM), incorporating a novel cross-modulated diffusion transformer along with an optimized integration module, together forming an effective architecture. We further introduce a two-stage training paradigm that progressively aligns action with perception, preserving the representations of the VLM. Notably, with only 0.77 billion parameters, Evo-1 achieves state-of-the-art results on the Meta-World and RoboTwin suite, surpassing the previous best models by 12.4% and 6.9%, respectively, and also attains a competitive result of 94.8% on LIBERO. In real-world evaluations, Evo-1 attains a 78% success rate with high inference frequency and low memory overhead, outperforming all baseline methods. We release code, data, and model weights to facilitate future research on lightweight and efficient VLA models.
>
---
#### [replaced 018] Learning Visually Interpretable Oscillator Networks for Soft Continuum Robots from Video
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文研究软连续体机器人动力学建模，旨在提升数据驱动方法的物理可解释性。提出注意力广播解码器（ABCD）生成像素级注意力图，结合2D振荡器网络实现动态过程的可视化，无需先验知识即可学习紧凑、可解释的模型，并显著提高预测精度。**

- **链接: [https://arxiv.org/pdf/2511.18322v2](https://arxiv.org/pdf/2511.18322v2)**

> **作者:** Henrik Krauss; Johann Licher; Naoya Takeishi; Annika Raatz; Takehisa Yairi
>
> **备注:** Dataset available at: https://zenodo.org/records/17812071
>
> **摘要:** Data-driven learning of soft continuum robot (SCR) dynamics from high-dimensional observations offers flexibility but often lacks physical interpretability, while model-based approaches require prior knowledge and can be computationally expensive. We bridge this gap by introducing (1) the Attention Broadcast Decoder (ABCD), a plug-and-play module for autoencoder-based latent dynamics learning that generates pixel-accurate attention maps localizing each latent dimension's contribution while filtering static backgrounds. (2) By coupling these attention maps to 2D oscillator networks, we enable direct on-image visualization of learned dynamics (masses, stiffness, and forces) without prior knowledge. We validate our approach on single- and double-segment SCRs, demonstrating that ABCD-based models significantly improve multi-step prediction accuracy: 5.7x error reduction for Koopman operators and 3.5x for oscillator networks on the two-segment robot. The learned oscillator network autonomously discovers a chain structure of oscillators. Unlike standard methods, ABCD models enable smooth latent space extrapolation beyond training data. This fully data-driven approach yields compact, physically interpretable models suitable for control applications.
>
---
#### [replaced 019] Semantic Communication and Control Co-Design for Multi-Objective Distinct Dynamics
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文研究语义通信与控制联合设计，旨在降低通信开销并提升多目标动态系统的预测与控制性能。提出基于Koopman算子的逻辑语义自编码器框架，结合STL编码控制规则，实现状态线性化建模，显著减少通信量并提高精度。**

- **链接: [https://arxiv.org/pdf/2410.02303v2](https://arxiv.org/pdf/2410.02303v2)**

> **作者:** Abanoub M. Girgis; Hyowoon Seo; Mehdi Bennis
>
> **摘要:** This letter introduces a machine-learning approach to learning the semantic dynamics of correlated systems with different control rules and dynamics. By leveraging the Koopman operator in an autoencoder (AE) framework, the system's state evolution is linearized in the latent space using a dynamic semantic Koopman (DSK) model, capturing the baseline semantic dynamics. Signal temporal logic (STL) is incorporated through a logical semantic Koopman (LSK) model to encode system-specific control rules. These models form the proposed logical Koopman AE framework that reduces communication costs while improving state prediction accuracy and control performance, showing a 91.65% reduction in communication samples and significant performance gains in simulation.
>
---
#### [replaced 020] H-GAR: A Hierarchical Interaction Framework via Goal-Driven Observation-Action Refinement for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文研究机器人操作中的视频与动作预测任务，旨在解决现有方法因忽略目标导向导致的语义错位问题。作者提出H-GAR框架，通过目标引导的粗到精观察-动作协同优化，实现更准确、连贯的操作预测。**

- **链接: [https://arxiv.org/pdf/2511.17079v2](https://arxiv.org/pdf/2511.17079v2)**

> **作者:** Yijie Zhu; Rui Shao; Ziyang Liu; Jie He; Jizhihui Liu; Jiuru Wang; Zitong Yu
>
> **备注:** Accepted to AAAI 2026 (Oral), Project Page: https://github.com/JiuTian-VL/H-GAR
>
> **摘要:** Unified video and action prediction models hold great potential for robotic manipulation, as future observations offer contextual cues for planning, while actions reveal how interactions shape the environment. However, most existing approaches treat observation and action generation in a monolithic and goal-agnostic manner, often leading to semantically misaligned predictions and incoherent behaviors. To this end, we propose H-GAR, a Hierarchical interaction framework via Goal-driven observation-Action Refinement.To anchor prediction to the task objective, H-GAR first produces a goal observation and a coarse action sketch that outline a high-level route toward the goal. To enable explicit interaction between observation and action under the guidance of the goal observation for more coherent decision-making, we devise two synergistic modules. (1) Goal-Conditioned Observation Synthesizer (GOS) synthesizes intermediate observations based on the coarse-grained actions and the predicted goal observation. (2) Interaction-Aware Action Refiner (IAAR) refines coarse actions into fine-grained, goal-consistent actions by leveraging feedback from the intermediate observations and a Historical Action Memory Bank that encodes prior actions to ensure temporal consistency. By integrating goal grounding with explicit action-observation interaction in a coarse-to-fine manner, H-GAR enables more accurate manipulation. Extensive experiments on both simulation and real-world robotic manipulation tasks demonstrate that H-GAR achieves state-of-the-art performance.
>
---
#### [replaced 021] Point-PNG: Conditional Pseudo-Negatives Generation for Point Cloud Pre-Training
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于点云自监督学习任务，旨在解决现有方法因不变性坍塌导致变换信息丢失的问题。作者提出Point-PNG框架，通过条件伪负样本生成和COPE网络，显式惩罚不变性坍塌，提升表示的判别性和变换敏感性。**

- **链接: [https://arxiv.org/pdf/2409.15832v3](https://arxiv.org/pdf/2409.15832v3)**

> **作者:** Sutharsan Mahendren; Saimunur Rahman; Piotr Koniusz; Tharindu Fernando; Sridha Sridharan; Clinton Fookes; Peyman Moghadam
>
> **备注:** Accepted for publication in IEEE ACCESS
>
> **摘要:** We propose Point-PNG, a novel self-supervised learning framework that generates conditional pseudo-negatives in the latent space to learn point cloud representations that are both discriminative and transformation-sensitive. Conventional self-supervised learning methods focus on achieving invariance, discarding transformation-specific information. Recent approaches incorporate transformation sensitivity by explicitly modeling relationships between original and transformed inputs. However, they often suffer from an invariant-collapse phenomenon, where the predictor degenerates into identity mappings, resulting in latent representations with limited variation across transformations. To address this, we propose Point-PNG that explicitly penalizes invariant collapse through pseudo-negatives generation, enabling the network to capture richer transformation cues while preserving discriminative representations. To this end, we introduce a parametric network, COnditional Pseudo-Negatives Embedding (COPE), which learns localized displacements induced by transformations within the latent space. A key challenge arises when jointly training COPE with the MAE, as it tends to converge to trivial identity mappings. To overcome this, we design a loss function based on pseudo-negatives conditioned on the transformation, which penalizes such trivial invariant solutions and enforces meaningful representation learning. We validate Point-PNG on shape classification and relative pose estimation tasks, showing competitive performance on ModelNet40 and ScanObjectNN under challenging evaluation protocols, and achieving superior accuracy in relative pose estimation compared to supervised baselines.
>
---
