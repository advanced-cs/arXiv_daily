# 机器人 cs.RO

- **最新发布 46 篇**

- **更新 35 篇**

## 最新发布

#### [new 001] Differential Barometric Altimetry for Submeter Vertical Localization and Floor Recognition Indoors
- **分类: cs.RO**

- **简介: 该论文属于室内定位任务，解决垂直定位与楼层识别问题。通过差分气压传感实现亚米级精度的垂直定位和100%楼层识别。**

- **链接: [https://arxiv.org/pdf/2601.02184v1](https://arxiv.org/pdf/2601.02184v1)**

> **作者:** Yuhang Zhang; Sören Schwertfeger
>
> **摘要:** Accurate altitude estimation and reliable floor recognition are critical for mobile robot localization and navigation within complex multi-storey environments. In this paper, we present a robust, low-cost vertical estimation framework leveraging differential barometric sensing integrated within a fully ROS-compliant software package. Our system simultaneously publishes real-time altitude data from both a stationary base station and a mobile sensor, enabling precise and drift-free vertical localization. Empirical evaluations conducted in challenging scenarios -- such as fully enclosed stairwells and elevators, demonstrate that our proposed barometric pipeline achieves sub-meter vertical accuracy (RMSE: 0.29 m) and perfect (100%) floor-level identification. In contrast, our results confirm that standalone height estimates, obtained solely from visual- or LiDAR-based SLAM odometry, are insufficient for reliable vertical localization. The proposed ROS-compatible barometric module thus provides a practical and cost-effective solution for robust vertical awareness in real-world robotic deployments. The implementation of our method is released as open source at https://github.com/witsir/differential-barometric.
>
---
#### [new 002] Explicit World Models for Reliable Human-Robot Collaboration
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机协作任务，旨在解决感知噪声和指令模糊带来的可靠性问题。通过构建显式世界模型，使机器人更好地理解并响应人类意图。**

- **链接: [https://arxiv.org/pdf/2601.01705v1](https://arxiv.org/pdf/2601.01705v1)**

> **作者:** Kenneth Kwok; Basura Fernando; Qianli Xu; Vigneshwaran Subbaraju; Dongkyu Choi; Boon Kiat Quek
>
> **备注:** Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
>
> **摘要:** This paper addresses the topic of robustness under sensing noise, ambiguous instructions, and human-robot interaction. We take a radically different tack to the issue of reliable embodied AI: instead of focusing on formal verification methods aimed at achieving model predictability and robustness, we emphasise the dynamic, ambiguous and subjective nature of human-robot interactions that requires embodied AI systems to perceive, interpret, and respond to human intentions in a manner that is consistent, comprehensible and aligned with human expectations. We argue that when embodied agents operate in human environments that are inherently social, multimodal, and fluid, reliability is contextually determined and only has meaning in relation to the goals and expectations of humans involved in the interaction. This calls for a fundamentally different approach to achieving reliable embodied AI that is centred on building and updating an accessible "explicit world model" representing the common ground between human and AI, that is used to align robot behaviours with human expectations.
>
---
#### [new 003] Action-Sketcher: From Reasoning to Action via Visual Sketches for Long-Horizon Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出Action-Sketcher框架，解决长周期机器人操作中的空间歧义和时间鲁棒性问题，通过视觉草图实现语言与场景几何的连接。**

- **链接: [https://arxiv.org/pdf/2601.01618v1](https://arxiv.org/pdf/2601.01618v1)**

> **作者:** Huajie Tan; Peterson Co; Yijie Xu; Shanyu Rong; Yuheng Ji; Cheng Chi; Xiansheng Chen; Qiongyu Zhang; Zhongxia Zhao; Pengwei Wang; Zhongyuan Wang; Shanghang Zhang
>
> **备注:** 26 pages, 14 figures
>
> **摘要:** Long-horizon robotic manipulation is increasingly important for real-world deployment, requiring spatial disambiguation in complex layouts and temporal resilience under dynamic interaction. However, existing end-to-end and hierarchical Vision-Language-Action (VLA) policies often rely on text-only cues while keeping plan intent latent, which undermines referential grounding in cluttered or underspecified scenes, impedes effective task decomposition of long-horizon goals with close-loop interaction, and limits causal explanation by obscuring the rationale behind action choices. To address these issues, we first introduce Visual Sketch, an implausible visual intermediate that renders points, boxes, arrows, and typed relations in the robot's current views to externalize spatial intent, connect language to scene geometry. Building on Visual Sketch, we present Action-Sketcher, a VLA framework that operates in a cyclic See-Think-Sketch-Act workflow coordinated by adaptive token-gated strategy for reasoning triggers, sketch revision, and action issuance, thereby supporting reactive corrections and human interaction while preserving real-time action prediction. To enable scalable training and evaluation, we curate diverse corpus with interleaved images, text, Visual Sketch supervision, and action sequences, and train Action-Sketcher with a multi-stage curriculum recipe that combines interleaved sequence alignment for modality unification, language-to-sketch consistency for precise linguistic grounding, and imitation learning augmented with sketch-to-action reinforcement for robustness. Extensive experiments on cluttered scenes and multi-object tasks, in simulation and on real-world tasks, show improved long-horizon success, stronger robustness to dynamic scene changes, and enhanced interpretability via editable sketches and step-wise plans. Project website: https://action-sketcher.github.io
>
---
#### [new 004] CycleVLA: Proactive Self-Correcting Vision-Language-Action Models via Subtask Backtracking and Minimum Bayes Risk Decoding
- **分类: cs.RO**

- **简介: 该论文提出CycleVLA，解决机器人执行中的故障检测与自纠正问题，通过子任务回溯和最小贝叶斯风险解码实现主动预防性纠错。**

- **链接: [https://arxiv.org/pdf/2601.02295v1](https://arxiv.org/pdf/2601.02295v1)**

> **作者:** Chenyang Ma; Guangyu Yang; Kai Lu; Shitong Xu; Bill Byrne; Niki Trigoni; Andrew Markham
>
> **备注:** Project Page: https://dannymcy.github.io/cyclevla/
>
> **摘要:** Current work on robot failure detection and correction typically operate in a post hoc manner, analyzing errors and applying corrections only after failures occur. This work introduces CycleVLA, a system that equips Vision-Language-Action models (VLAs) with proactive self-correction, the capability to anticipate incipient failures and recover before they fully manifest during execution. CycleVLA achieves this by integrating a progress-aware VLA that flags critical subtask transition points where failures most frequently occur, a VLM-based failure predictor and planner that triggers subtask backtracking upon predicted failure, and a test-time scaling strategy based on Minimum Bayes Risk (MBR) decoding to improve retry success after backtracking. Extensive experiments show that CycleVLA improves performance for both well-trained and under-trained VLAs, and that MBR serves as an effective zero-shot test-time scaling strategy for VLAs. Project Page: https://dannymcy.github.io/cyclevla/
>
---
#### [new 005] SAHA: Supervised Autonomous HArvester for selective forest thinning
- **分类: cs.RO**

- **简介: 该论文提出SAHA系统，用于解决森林择伐的自动化问题。通过改进机器人平台，实现自主导航与精准作业，提升森林管理效率。**

- **链接: [https://arxiv.org/pdf/2601.01282v1](https://arxiv.org/pdf/2601.01282v1)**

> **作者:** Fang Nan; Meher Malladi; Qingqing Li; Fan Yang; Joonas Juola; Tiziano Guadagnino; Jens Behley; Cesar Cadena; Cyrill Stachniss; Marco Hutter
>
> **摘要:** Forestry plays a vital role in our society, creating significant ecological, economic, and recreational value. Efficient forest management involves labor-intensive and complex operations. One essential task for maintaining forest health and productivity is selective thinning, which requires skilled operators to remove specific trees to create optimal growing conditions for the remaining ones. In this work, we present a solution based on a small-scale robotic harvester (SAHA) designed for executing this task with supervised autonomy. We build on a 4.5-ton harvester platform and implement key hardware modifications for perception and automatic control. We implement learning- and model-based approaches for precise control of hydraulic actuators, accurate navigation through cluttered environments, robust state estimation, and reliable semantic estimation of terrain traversability. Integrating state-of-the-art techniques in perception, planning, and control, our robotic harvester can autonomously navigate forest environments and reach targeted trees for selective thinning. We present experimental results from extensive field trials over kilometer-long autonomous missions in northern European forests, demonstrating the harvester's ability to operate in real forests. We analyze the performance and provide the lessons learned for advancing robotic forest management.
>
---
#### [new 006] Simulations and Advancements in MRI-Guided Power-Driven Ferric Tools for Wireless Therapeutic Interventions
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于医疗机器人任务，旨在解决MRI环境下无线治疗设备的精准控制问题。通过开发计算系统，实现血管路径规划与安全导航，提升介入治疗的精度与安全性。**

- **链接: [https://arxiv.org/pdf/2601.01726v1](https://arxiv.org/pdf/2601.01726v1)**

> **作者:** Wenhui Chu; Aobo Jin; Hardik A. Gohel
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Designing a robotic system that functions effectively within the specific environment of a Magnetic Resonance Imaging (MRI) scanner requires solving numerous technical issues, such as maintaining the robot's precision and stability under strong magnetic fields. This research focuses on enhancing MRI's role in medical imaging, especially in its application to guide intravascular interventions using robot-assisted devices. A newly developed computational system is introduced, designed for seamless integration with the MRI scanner, including a computational unit and user interface. This system processes MR images to delineate the vascular network, establishing virtual paths and boundaries within vessels to prevent procedural damage. Key findings reveal the system's capability to create tailored magnetic field gradient patterns for device control, considering the vessel's geometry and safety norms, and adapting to different blood flow characteristics for finer navigation. Additionally, the system's modeling aspect assesses the safety and feasibility of navigating pre-set vascular paths. Conclusively, this system, based on the Qt framework and C/C++, with specialized software modules, represents a major step forward in merging imaging technology with robotic aid, significantly enhancing precision and safety in intravascular procedures.
>
---
#### [new 007] Topological Mapping and Navigation using a Monocular Camera based on AnyLoc
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，解决单目相机下的拓扑地图构建与导航问题。通过AnyLoc提取关键帧描述符，建立拓扑关系，实现高效路径规划与闭环检测。**

- **链接: [https://arxiv.org/pdf/2601.01067v1](https://arxiv.org/pdf/2601.01067v1)**

> **作者:** Wenzheng Zhang; Yoshitaka Hara; Sousuke Nakamura
>
> **备注:** Published in Proc. IEEE CASE 2025. 7 pages, 11 figures
>
> **摘要:** This paper proposes a method for topological mapping and navigation using a monocular camera. Based on AnyLoc, keyframes are converted into descriptors to construct topological relationships, enabling loop detection and map building. Unlike metric maps, topological maps simplify path planning and navigation by representing environments with key nodes instead of precise coordinates. Actions for visual navigation are determined by comparing segmented images with the image associated with target nodes. The system relies solely on a monocular camera, ensuring fast map building and navigation using key nodes. Experiments show effective loop detection and navigation in real and simulation environments without pre-training. Compared to a ResNet-based method, this approach improves success rates by 60.2% on average while reducing time and space costs, offering a lightweight solution for robot and human navigation in various scenarios.
>
---
#### [new 008] Online Estimation and Manipulation of Articulated Objects
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在解决未知铰接物体的自主操控问题。通过融合视觉先验与运动感知，提出一种基于螺旋理论的在线估计方法，提升机器人对未知物体的识别与操作能力。**

- **链接: [https://arxiv.org/pdf/2601.01438v1](https://arxiv.org/pdf/2601.01438v1)**

> **作者:** Russell Buchanan; Adrian Röfer; João Moura; Abhinav Valada; Sethu Vijayakumar
>
> **备注:** This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this article is published in Autonomous Robots, and is available online at [Link will be updated when available]
>
> **摘要:** From refrigerators to kitchen drawers, humans interact with articulated objects effortlessly every day while completing household chores. For automating these tasks, service robots must be capable of manipulating arbitrary articulated objects. Recent deep learning methods have been shown to predict valuable priors on the affordance of articulated objects from vision. In contrast, many other works estimate object articulations by observing the articulation motion, but this requires the robot to already be capable of manipulating the object. In this article, we propose a novel approach combining these methods by using a factor graph for online estimation of articulation which fuses learned visual priors and proprioceptive sensing during interaction into an analytical model of articulation based on Screw Theory. With our method, a robotic system makes an initial prediction of articulation from vision before touching the object, and then quickly updates the estimate from kinematic and force sensing during manipulation. We evaluate our method extensively in both simulations and real-world robotic manipulation experiments. We demonstrate several closed-loop estimation and manipulation experiments in which the robot was capable of opening previously unseen drawers. In real hardware experiments, the robot achieved a 75% success rate for autonomous opening of unknown articulated objects.
>
---
#### [new 009] AIMS: An Adaptive Integration of Multi-Sensor Measurements for Quadrupedal Robot Localization
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，旨在解决四足机器人在狭窄隧道环境中定位精度低的问题。通过融合LiDAR、IMU和腿里程计数据，提出自适应融合方法AIMS以提高定位准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.01561v1](https://arxiv.org/pdf/2601.01561v1)**

> **作者:** Yujian Qiu; Yuqiu Mu; Wen Yang; Hao Zhu
>
> **摘要:** This paper addresses the problem of accurate localization for quadrupedal robots operating in narrow tunnel-like environments. Due to the long and homogeneous characteristics of such scenarios, LiDAR measurements often provide weak geometric constraints, making traditional sensor fusion methods susceptible to accumulated motion estimation errors. To address these challenges, we propose AIMS, an adaptive LiDAR-IMU-leg odometry fusion method for robust quadrupedal robot localization in degenerate environments. The proposed method is formulated within an error-state Kalman filtering framework, where LiDAR and leg odometry measurements are integrated with IMU-based state prediction, and measurement noise covariance matrices are adaptively adjusted based on online degeneracy-aware reliability assessment. Experimental results obtained in narrow corridor environments demonstrate that the proposed method improves localization accuracy and robustness compared with state-of-the-art approaches.
>
---
#### [new 010] Vision-Based Early Fault Diagnosis and Self-Recovery for Strawberry Harvesting Robots
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于农业机器人任务，解决草莓采摘机器人的视觉感知与抓取故障问题，提出SRR-Net模型和自恢复策略，提升采摘稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2601.02085v1](https://arxiv.org/pdf/2601.02085v1)**

> **作者:** Meili Sun; Chunjiang Zhao; Lichao Yang; Hao Liu; Shimin Hu; Ya Xiong
>
> **摘要:** Strawberry harvesting robots faced persistent challenges such as low integration of visual perception, fruit-gripper misalignment, empty grasping, and strawberry slippage from the gripper due to insufficient gripping force, all of which compromised harvesting stability and efficiency in orchard environments. To overcome these issues, this paper proposed a visual fault diagnosis and self-recovery framework that integrated multi-task perception with corrective control strategies. At the core of this framework was SRR-Net, an end-to-end multi-task perception model that simultaneously performed strawberry detection, segmentation, and ripeness estimation, thereby unifying visual perception with fault diagnosis. Based on this integrated perception, a relative error compensation method based on the simultaneous target-gripper detection was designed to address positional misalignment, correcting deviations when error exceeded the tolerance threshold. To mitigate empty grasping and fruit-slippage faults, an early abort strategy was implemented. A micro-optical camera embedded in the end-effector provided real-time visual feedback, enabling grasp detection during the deflating stage and strawberry slip prediction during snap-off through MobileNet V3-Small classifier and a time-series LSTM classifier. Experiments demonstrated that SRR-Net maintained high perception accuracy. For detection, it achieved a precision of 0.895 and recall of 0.813 on strawberries, and 0.972/0.958 on hands. In segmentation, it yielded a precision of 0.887 and recall of 0.747 for strawberries, and 0.974/0.947 for hands. For ripeness estimation, SRR-Net attained a mean absolute error of 0.035, while simultaneously supporting multi-task perception and sustaining a competitive inference speed of 163.35 FPS.
>
---
#### [new 011] From Perception to Symbolic Task Planning: Vision-Language Guided Human-Robot Collaborative Structured Assembly
- **分类: cs.RO**

- **简介: 该论文属于人机协同装配任务，解决感知噪声和人类干预下的状态估计与任务规划问题。提出双模块框架，结合视觉语言模型与知识驱动规划，提升动态环境下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.00978v1](https://arxiv.org/pdf/2601.00978v1)**

> **作者:** Yanyi Chen; Min Deng
>
> **摘要:** Human-robot collaboration (HRC) in structured assembly requires reliable state estimation and adaptive task planning under noisy perception and human interventions. To address these challenges, we introduce a design-grounded human-aware planning framework for human-robot collaborative structured assembly. The framework comprises two coupled modules. Module I, Perception-to-Symbolic State (PSS), employs vision-language models (VLMs) based agents to align RGB-D observations with design specifications and domain knowledge, synthesizing verifiable symbolic assembly states. It outputs validated installed and uninstalled component sets for online state tracking. Module II, Human-Aware Planning and Replanning (HPR), performs task-level multi-robot assignment and updates the plan only when the observed state deviates from the expected execution outcome. It applies a minimal-change replanning rule to selectively revise task assignments and preserve plan stability even under human interventions. We validate the framework on a 27-component timber-frame assembly. The PSS module achieves 97% state synthesis accuracy, and the HPR module maintains feasible task progression across diverse HRC scenarios. Results indicate that integrating VLM-based perception with knowledge-driven planning improves robustness of state estimation and task planning under dynamic conditions.
>
---
#### [new 012] AlignDrive: Aligned Lateral-Longitudinal Planning for End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决路径与速度协调不足的问题。提出一种级联框架，将纵向规划基于路径，提升协同性和安全性。**

- **链接: [https://arxiv.org/pdf/2601.01762v1](https://arxiv.org/pdf/2601.01762v1)**

> **作者:** Yanhao Wu; Haoyang Zhang; Fei He; Rui Wu; Congpei Qiu; Liang Gao; Wei Ke; Tong Zhang
>
> **备注:** underreview
>
> **摘要:** End-to-end autonomous driving has rapidly progressed, enabling joint perception and planning in complex environments. In the planning stage, state-of-the-art (SOTA) end-to-end autonomous driving models decouple planning into parallel lateral and longitudinal predictions. While effective, this parallel design can lead to i) coordination failures between the planned path and speed, and ii) underutilization of the drive path as a prior for longitudinal planning, thus redundantly encoding static information. To address this, we propose a novel cascaded framework that explicitly conditions longitudinal planning on the drive path, enabling coordinated and collision-aware lateral and longitudinal planning. Specifically, we introduce a path-conditioned formulation that explicitly incorporates the drive path into longitudinal planning. Building on this, the model predicts longitudinal displacements along the drive path rather than full 2D trajectory waypoints. This design simplifies longitudinal reasoning and more tightly couples it with lateral planning. Additionally, we introduce a planning-oriented data augmentation strategy that simulates rare safety-critical events, such as vehicle cut-ins, by adding agents and relabeling longitudinal targets to avoid collision. Evaluated on the challenging Bench2Drive benchmark, our method sets a new SOTA, achieving a driving score of 89.07 and a success rate of 73.18%, demonstrating significantly improved coordination and safety
>
---
#### [new 013] VISO: Robust Underwater Visual-Inertial-Sonar SLAM with Photometric Rendering for Dense 3D Reconstruction
- **分类: cs.RO**

- **简介: 该论文属于水下SLAM任务，旨在解决水下视觉定位不准和三维重建精度低的问题。提出VISO系统，融合视觉、惯性与声呐数据，提升定位精度与重建质量。**

- **链接: [https://arxiv.org/pdf/2601.01144v1](https://arxiv.org/pdf/2601.01144v1)**

> **作者:** Shu Pan; Simon Archieri; Ahmet Cinar; Jonatan Scharff Willners; Ignacio Carlucho; Yvan Petillot
>
> **摘要:** Visual challenges in underwater environments significantly hinder the accuracy of vision-based localisation and the high-fidelity dense reconstruction. In this paper, we propose VISO, a robust underwater SLAM system that fuses a stereo camera, an inertial measurement unit (IMU), and a 3D sonar to achieve accurate 6-DoF localisation and enable efficient dense 3D reconstruction with high photometric fidelity. We introduce a coarse-to-fine online calibration approach for extrinsic parameters estimation between the 3D sonar and the camera. Additionally, a photometric rendering strategy is proposed for the 3D sonar point cloud to enrich the sonar map with visual information. Extensive experiments in a laboratory tank and an open lake demonstrate that VISO surpasses current state-of-the-art underwater and visual-based SLAM algorithms in terms of localisation robustness and accuracy, while also exhibiting real-time dense 3D reconstruction performance comparable to the offline dense mapping method.
>
---
#### [new 014] SingingBot: An Avatar-Driven System for Robotic Face Singing Performance
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于情感机器人面部演唱任务，旨在解决机器人在歌唱中连续情感表达与一致性问题。提出一种基于虚拟形象的框架，生成生动的演唱形象并映射到机器人，提升情感表现力。**

- **链接: [https://arxiv.org/pdf/2601.02125v1](https://arxiv.org/pdf/2601.02125v1)**

> **作者:** Zhuoxiong Xu; Xuanchen Li; Yuhao Cheng; Fei Xu; Yichao Yan; Xiaokang Yang
>
> **摘要:** Equipping robotic faces with singing capabilities is crucial for empathetic Human-Robot Interaction. However, existing robotic face driving research primarily focuses on conversations or mimicking static expressions, struggling to meet the high demands for continuous emotional expression and coherence in singing. To address this, we propose a novel avatar-driven framework for appealing robotic singing. We first leverage portrait video generation models embedded with extensive human priors to synthesize vivid singing avatars, providing reliable expression and emotion guidance. Subsequently, these facial features are transferred to the robot via semantic-oriented mapping functions that span a wide expression space. Furthermore, to quantitatively evaluate the emotional richness of robotic singing, we propose the Emotion Dynamic Range metric to measure the emotional breadth within the Valence-Arousal space, revealing that a broad emotional spectrum is crucial for appealing performances. Comprehensive experiments prove that our method achieves rich emotional expressions while maintaining lip-audio synchronization, significantly outperforming existing approaches.
>
---
#### [new 015] DemoBot: Efficient Learning of Bimanual Manipulation with Dexterous Hands From Third-Person Human Videos
- **分类: cs.RO**

- **简介: 该论文提出DemoBot，用于从人类视频中学习双臂精细操作技能。解决如何从单个未标注视频高效获取复杂操作任务的问题，通过视频处理和强化学习框架实现。**

- **链接: [https://arxiv.org/pdf/2601.01651v1](https://arxiv.org/pdf/2601.01651v1)**

> **作者:** Yucheng Xu; Xiaofeng Mao; Elle Miller; Xinyu Yi; Yang Li; Zhibin Li; Robert B. Fisher
>
> **摘要:** This work presents DemoBot, a learning framework that enables a dual-arm, multi-finger robotic system to acquire complex manipulation skills from a single unannotated RGB-D video demonstration. The method extracts structured motion trajectories of both hands and objects from raw video data. These trajectories serve as motion priors for a novel reinforcement learning (RL) pipeline that learns to refine them through contact-rich interactions, thereby eliminating the need to learn from scratch. To address the challenge of learning long-horizon manipulation skills, we introduce: (1) Temporal-segment based RL to enforce temporal alignment of the current state with demonstrations; (2) Success-Gated Reset strategy to balance the refinement of readily acquired skills and the exploration of subsequent task stages; and (3) Event-Driven Reward curriculum with adaptive thresholding to guide the RL learning of high-precision manipulation. The novel video processing and RL framework successfully achieved long-horizon synchronous and asynchronous bimanual assembly tasks, offering a scalable approach for direct skill acquisition from human videos.
>
---
#### [new 016] VisuoTactile 6D Pose Estimation of an In-Hand Object using Vision and Tactile Sensor Data
- **分类: cs.RO**

- **简介: 该论文属于6D姿态估计任务，旨在解决机器人抓取物体时因遮挡导致的定位困难。通过融合视觉与触觉数据，提升物体姿态估计精度。**

- **链接: [https://arxiv.org/pdf/2601.01675v1](https://arxiv.org/pdf/2601.01675v1)**

> **作者:** Snehal s. Dikhale; Karankumar Patel; Daksh Dhingra; Itoshi Naramura; Akinobu Hayashi; Soshi Iba; Nawid Jamali
>
> **备注:** Accepted for publication in IEEE Robotics and Automation Letters (RA-L), January 2022. Presented at ICRA 2022. This is the author's version of the manuscript
>
> **摘要:** Knowledge of the 6D pose of an object can benefit in-hand object manipulation. In-hand 6D object pose estimation is challenging because of heavy occlusion produced by the robot's grippers, which can have an adverse effect on methods that rely on vision data only. Many robots are equipped with tactile sensors at their fingertips that could be used to complement vision data. In this paper, we present a method that uses both tactile and vision data to estimate the pose of an object grasped in a robot's hand. To address challenges like lack of standard representation for tactile data and sensor fusion, we propose the use of point clouds to represent object surfaces in contact with the tactile sensor and present a network architecture based on pixel-wise dense fusion. We also extend NVIDIA's Deep Learning Dataset Synthesizer to produce synthetic photo-realistic vision data and corresponding tactile point clouds. Results suggest that using tactile data in addition to vision data improves the 6D pose estimate, and our network generalizes successfully from synthetic training to real physical robots.
>
---
#### [new 017] Simulations of MRI Guided and Powered Ferric Applicators for Tetherless Delivery of Therapeutic Interventions
- **分类: cs.RO; cs.CV; eess.SY**

- **简介: 该论文属于医学机器人领域，旨在解决MRI引导下无导线治疗设备的术前规划问题。通过构建计算平台，实现血管路径模拟与安全评估。**

- **链接: [https://arxiv.org/pdf/2601.00981v1](https://arxiv.org/pdf/2601.00981v1)**

> **作者:** Wenhui Chu; Khang Tran; Nikolaos V. Tsekos
>
> **备注:** 9 pages, 8 figures, published in ICBBB 2022
>
> **摘要:** Magnetic Resonance Imaging (MRI) is a well-established modality for pre-operative planning and is also explored for intra-operative guidance of procedures such as intravascular interventions. Among the experimental robot-assisted technologies, the magnetic field gradients of the MRI scanner are used to power and maneuver ferromagnetic applicators for accessing sites in the patient's body via the vascular network. In this work, we propose a computational platform for preoperative planning and modeling of MRI-powered applicators inside blood vessels. This platform was implemented as a two-way data and command pipeline that links the MRI scanner, the computational core, and the operator. The platform first processes multi-slice MR data to extract the vascular bed and then fits a virtual corridor inside the vessel. This corridor serves as a virtual fixture (VF), a forbidden region for the applicators to avoid vessel perforation or collision. The geometric features of the vessel centerline, the VF, and MRI safety compliance (dB/dt, max available gradient) are then used to generate magnetic field gradient waveforms. Different blood flow profiles can be user-selected, and those parameters are used for modeling the applicator's maneuvering. The modeling module further generates cues about whether the selected vascular path can be safely maneuvered. Given future experimental studies that require a real-time operation, the platform was implemented on the Qt framework (C/C++) with software modules performing specific tasks running on dedicated threads: PID controller, generation of VF, generation of MR gradient waveforms.
>
---
#### [new 018] Towards reliable subsea object recovery: a simulation study of an auv with a suction-actuated end effector
- **分类: cs.RO**

- **简介: 该论文属于深海自主物体回收任务，旨在解决极端环境下自主操作难题。通过仿真验证了AUV与吸力末端执行器的协同控制方法。**

- **链接: [https://arxiv.org/pdf/2601.01106v1](https://arxiv.org/pdf/2601.01106v1)**

> **作者:** Michele Grimaldi; Yosaku Maeda; Hitoshi Kakami; Ignacio Carlucho; Yvan Petillot; Tomoya Inoue
>
> **摘要:** Autonomous object recovery in the hadal zone is challenging due to extreme hydrostatic pressure, limited visibility and currents, and the need for precise manipulation at full ocean depth. Field experimentation in such environments is costly, high-risk, and constrained by limited vehicle availability, making early validation of autonomous behaviors difficult. This paper presents a simulation-based study of a complete autonomous subsea object recovery mission using a Hadal Small Vehicle (HSV) equipped with a three-degree-of-freedom robotic arm and a suction-actuated end effector. The Stonefish simulator is used to model realistic vehicle dynamics, hydrodynamic disturbances, sensing, and interaction with a target object under hadal-like conditions. The control framework combines a world-frame PID controller for vehicle navigation and stabilization with an inverse-kinematics-based manipulator controller augmented by acceleration feed-forward, enabling coordinated vehicle - manipulator operation. In simulation, the HSV autonomously descends from the sea surface to 6,000 m, performs structured seafloor coverage, detects a target object, and executes a suction-based recovery. The results demonstrate that high-fidelity simulation provides an effective and low-risk means of evaluating autonomous deep-sea intervention behaviors prior to field deployment.
>
---
#### [new 019] EduSim-LLM: An Educational Platform Integrating Large Language Models and Robotic Simulation for Beginners
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决自然语言控制机器人的问题。通过集成大语言模型与机器人仿真，构建语言驱动控制模型，提升机器人操作的易用性与准确性。**

- **链接: [https://arxiv.org/pdf/2601.01196v1](https://arxiv.org/pdf/2601.01196v1)**

> **作者:** Shenqi Lu; Liangwei Zhang
>
> **摘要:** In recent years, the rapid development of Large Language Models (LLMs) has significantly enhanced natural language understanding and human-computer interaction, creating new opportunities in the field of robotics. However, the integration of natural language understanding into robotic control is an important challenge in the rapid development of human-robot interaction and intelligent automation industries. This challenge hinders intuitive human control over complex robotic systems, limiting their educational and practical accessibility. To address this, we present the EduSim-LLM, an educational platform that integrates LLMs with robot simulation and constructs a language-drive control model that translates natural language instructions into executable robot behavior sequences in CoppeliaSim. We design two human-robot interaction models: direct control and autonomous control, conduct systematic simulations based on multiple language models, and evaluate multi-robot collaboration, motion planning, and manipulation capabilities. Experiential results show that LLMs can reliably convert natural language into structured robot actions; after applying prompt-engineering templates instruction-parsing accuracy improves significantly; as task complexity increases, overall accuracy rate exceeds 88.9% in the highest complexity tests.
>
---
#### [new 020] Latent Space Reinforcement Learning for Multi-Robot Exploration
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人探索任务，解决环境映射与协作规划问题。通过自编码器压缩地图，结合强化学习实现高效多机协同，提升系统扩展性与适应性。**

- **链接: [https://arxiv.org/pdf/2601.01139v1](https://arxiv.org/pdf/2601.01139v1)**

> **作者:** Sriram Rajasekar; Ashwini Ratnoo
>
> **摘要:** Autonomous mapping of unknown environments is a critical challenge, particularly in scenarios where time is limited. Multi-agent systems can enhance efficiency through collaboration, but the scalability of motion-planning algorithms remains a key limitation. Reinforcement learning has been explored as a solution, but existing approaches are constrained by the limited input size required for effective learning, restricting their applicability to discrete environments. This work addresses that limitation by leveraging autoencoders to perform dimensionality reduction, compressing high-fidelity occupancy maps into latent state vectors while preserving essential spatial information. Additionally, we introduce a novel procedural generation algorithm based on Perlin noise, designed to generate topologically complex training environments that simulate asteroid fields, caves and forests. These environments are used for training the autoencoder and the navigation algorithm using a hierarchical deep reinforcement learning framework for decentralized coordination. We introduce a weighted consensus mechanism that modulates reliance on shared data via a tuneable trust parameter, ensuring robustness to accumulation of errors. Experimental results demonstrate that the proposed system scales effectively with number of agents and generalizes well to unfamiliar, structurally distinct environments and is resilient in communication-constrained settings.
>
---
#### [new 021] Genie Sim 3.0 : A High-Fidelity Comprehensive Simulation Platform for Humanoid Robot
- **分类: cs.RO**

- **简介: 该论文提出Genie Sim 3.0，解决机器人学习数据不足与仿真基准不统一问题，通过LLM生成高保真场景，构建大规模合成数据集，支持高效策略训练与评估。**

- **链接: [https://arxiv.org/pdf/2601.02078v1](https://arxiv.org/pdf/2601.02078v1)**

> **作者:** Chenghao Yin; Da Huang; Di Yang; Jichao Wang; Nanshu Zhao; Chen Xu; Wenjun Sun; Linjie Hou; Zhijun Li; Junhui Wu; Zhaobo Liu; Zhen Xiao; Sheng Zhang; Lei Bao; Rui Feng; Zhenquan Pang; Jiayu Li; Qian Wang; Maoqing Yao
>
> **摘要:** The development of robust and generalizable robot learning models is critically contingent upon the availability of large-scale, diverse training data and reliable evaluation benchmarks. Collecting data in the physical world poses prohibitive costs and scalability challenges, and prevailing simulation benchmarks frequently suffer from fragmentation, narrow scope, or insufficient fidelity to enable effective sim-to-real transfer. To address these challenges, we introduce Genie Sim 3.0, a unified simulation platform for robotic manipulation. We present Genie Sim Generator, a large language model (LLM)-powered tool that constructs high-fidelity scenes from natural language instructions. Its principal strength resides in rapid and multi-dimensional generalization, facilitating the synthesis of diverse environments to support scalable data collection and robust policy evaluation. We introduce the first benchmark that pioneers the application of LLM for automated evaluation. It leverages LLM to mass-generate evaluation scenarios and employs Vision-Language Model (VLM) to establish an automated assessment pipeline. We also release an open-source dataset comprising more than 10,000 hours of synthetic data across over 200 tasks. Through systematic experimentation, we validate the robust zero-shot sim-to-real transfer capability of our open-source dataset, demonstrating that synthetic data can server as an effective substitute for real-world data under controlled conditions for scalable policy training. For code and dataset details, please refer to: https://github.com/AgibotTech/genie_sim.
>
---
#### [new 022] DisCo-FLoc: Using Dual-Level Visual-Geometric Contrasts to Disambiguate Depth-Aware Visual Floorplan Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉楼图定位任务，旨在解决简约楼图中因重复结构导致的定位模糊问题。提出DisCo-FLoc方法，通过双级视觉-几何对比消除歧义，提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.01822v1](https://arxiv.org/pdf/2601.01822v1)**

> **作者:** Shiyong Meng; Tao Zou; Bolei Chen; Chaoxu Mu; Jianxin Wang
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Since floorplan data is readily available, long-term persistent, and robust to changes in visual appearance, visual Floorplan Localization (FLoc) has garnered significant attention. Existing methods either ingeniously match geometric priors or utilize sparse semantics to reduce FLoc uncertainty. However, they still suffer from ambiguous FLoc caused by repetitive structures within minimalist floorplans. Moreover, expensive but limited semantic annotations restrict their applicability. To address these issues, we propose DisCo-FLoc, which utilizes dual-level visual-geometric Contrasts to Disambiguate depth-aware visual Floc, without requiring additional semantic labels. Our solution begins with a ray regression predictor tailored for ray-casting-based FLoc, predicting a series of FLoc candidates using depth estimation expertise. In addition, a novel contrastive learning method with position-level and orientation-level constraints is proposed to strictly match depth-aware visual features with the corresponding geometric structures in the floorplan. Such matches can effectively eliminate FLoc ambiguity and select the optimal imaging pose from FLoc candidates. Exhaustive comparative studies on two standard visual Floc benchmarks demonstrate that our method outperforms the state-of-the-art semantic-based method, achieving significant improvements in both robustness and accuracy.
>
---
#### [new 023] From Metrics to Meaning: Insights from a Mixed-Methods Field Experiment on Retail Robot Deployment
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于服务机器人部署研究，旨在分析机器人与顾客、员工的互动效果。通过混合方法实验，探讨机器人对零售流程的影响，提出优化建议。**

- **链接: [https://arxiv.org/pdf/2601.01946v1](https://arxiv.org/pdf/2601.01946v1)**

> **作者:** Sichao Song; Yuki Okafuji; Takuya Iwamoto; Jun Baba; Hiroshi Ishiguro
>
> **摘要:** We report a mixed-methods field experiment of a conversational service robot deployed under everyday staffing discretion in a live bedding store. Over 12 days we alternated three conditions--Baseline (no robot), Robot-only, and Robot+Fixture--and video-annotated the service funnel from passersby to purchase. An explanatory sequential design then used six post-experiment staff interviews to interpret the quantitative patterns. Quantitatively, the robot increased stopping per passerby (highest with the fixture), yet clerk-led downstream steps per stopper--clerk approach, store entry, assisted experience, and purchase--decreased. Interviews explained this divergence: clerks avoided interrupting ongoing robot-customer talk, struggled with ambiguous timing amid conversational latency, and noted child-centered attraction that often satisfied curiosity at the doorway. The fixture amplified visibility but also anchored encounters at the threshold, creating a well-defined micro-space where needs could ``close'' without moving inside. We synthesize these strands into an integrative account from the initial show of interest on the part of a customer to their entering the store and derive actionable guidance. The results advance the understanding of interactions between customers, staff members, and the robot and offer practical recommendations for deploying service robots in high-touch retail.
>
---
#### [new 024] ORION: Option-Regularized Deep Reinforcement Learning for Cooperative Multi-Agent Online Navigation
- **分类: cs.RO**

- **简介: 该论文属于多智能体协作导航任务，解决部分已知环境下的协同导航问题。提出ORION框架，通过深度强化学习实现动态环境中的高效协作与不确定性降低。**

- **链接: [https://arxiv.org/pdf/2601.01155v1](https://arxiv.org/pdf/2601.01155v1)**

> **作者:** Zhang Shizhe; Liang Jingsong; Zhou Zhitao; Ye Shuhan; Wang Yizhuo; Tan Ming Siang Derek; Chiun Jimmy; Cao Yuhong; Sartoretti Guillaume
>
> **摘要:** Existing methods for multi-agent navigation typically assume fully known environments, offering limited support for partially known scenarios such as warehouses or factory floors. There, agents may need to plan trajectories that balance their own path optimality with their ability to collect and share information about the environment that can help their teammates reach their own goals. To these ends, we propose ORION, a novel deep reinforcement learning framework for cooperative multi-agent online navigation in partially known environments. Starting from an imperfect prior map, ORION trains agents to make decentralized decisions, coordinate to reach their individual targets, and actively reduce map uncertainty by sharing online observations in a closed perception-action loop. We first design a shared graph encoder that fuses prior map with online perception into a unified representation, providing robust state embeddings under dynamic map discrepancies. At the core of ORION is an option-critic framework that learns to reason about a set of high-level cooperative modes that translate into sequences of low-level actions, allowing agents to switch between individual navigation and team-level exploration adaptively. We further introduce a dual-stage cooperation strategy that enables agents to assist teammates under map uncertainty, thereby reducing the overall makespan. Across extensive maze-like maps and large-scale warehouse environments, our simulation results show that ORION achieves high-quality, real-time decentralized cooperation over varying team sizes, outperforming state-of-the-art classical and learning-based baselines. Finally, we validate ORION on physical robot teams, demonstrating its robustness and practicality for real-world cooperative navigation.
>
---
#### [new 025] HanoiWorld : A Joint Embedding Predictive Architecture BasedWorld Model for Autonomous Vehicle Controller
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出HanoiWorld，基于JEPA的自动驾驶控制器世界模型，解决数据需求高、安全性不足等问题，通过RNN实现安全驾驶规划。**

- **链接: [https://arxiv.org/pdf/2601.01577v1](https://arxiv.org/pdf/2601.01577v1)**

> **作者:** Tran Tien Dat; Nguyen Hai An; Nguyen Khanh Viet Dung; Nguyen Duy Duc
>
> **摘要:** Current attempts of Reinforcement Learning for Autonomous Controller are data-demanding while the results are under-performed, unstable, and unable to grasp and anchor on the concept of safety, and over-concentrating on noise features due to the nature of pixel reconstruction. While current Self-Supervised Learningapproachs that learning on high-dimensional representations by leveraging the JointEmbedding Predictive Architecture (JEPA) are interesting and an effective alternative, as the idea mimics the natural ability of the human brain in acquiring new skill usingimagination and minimal samples of observations. This study introduces Hanoi-World, a JEPA-based world model that using recurrent neural network (RNN) formaking longterm horizontal planning with effective inference time. Experimentsconducted on the Highway-Env package with difference enviroment showcase the effective capability of making a driving plan while safety-awareness, with considerablecollision rate in comparison with SOTA baselines
>
---
#### [new 026] Value Vision-Language-Action Planning & Search
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在分布偏移下的脆弱性问题。通过引入带价值函数的MCTS搜索框架V-VLAPS，提升成功率并减少仿真次数。**

- **链接: [https://arxiv.org/pdf/2601.00969v1](https://arxiv.org/pdf/2601.00969v1)**

> **作者:** Ali Salamatian; Ke; Ren; Kieran Pattison; Cyrus Neary
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as powerful generalist policies for robotic manipulation, yet they remain fundamentally limited by their reliance on behavior cloning, leading to brittleness under distribution shift. While augmenting pretrained models with test-time search algorithms like Monte Carlo Tree Search (MCTS) can mitigate these failures, existing formulations rely solely on the VLA prior for guidance, lacking a grounded estimate of expected future return. Consequently, when the prior is inaccurate, the planner can only correct action selection via the exploration term, which requires extensive simulation to become effective. To address this limitation, we introduce Value Vision-Language-Action Planning and Search (V-VLAPS), a framework that augments MCTS with a lightweight, learnable value function. By training a simple multilayer perceptron (MLP) on the latent representations of a fixed VLA backbone (Octo), we provide the search with an explicit success signal that biases action selection toward high-value regions. We evaluate V-VLAPS on the LIBERO robotic manipulation suite, demonstrating that our value-guided search improves success rates by over 5 percentage points while reducing the average number of MCTS simulations by 5-15 percent compared to baselines that rely only on the VLA prior.
>
---
#### [new 027] Deep Robust Koopman Learning from Noisy Data
- **分类: cs.RO**

- **简介: 该论文属于系统建模任务，旨在解决噪声数据下Koopman算子估计偏差问题。通过自编码器架构联合学习提升函数和低偏差算子，提高预测与控制的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.01971v1](https://arxiv.org/pdf/2601.01971v1)**

> **作者:** Aditya Singh; Rajpal Singh; Jishnu Keshavan
>
> **摘要:** Koopman operator theory has emerged as a leading data-driven approach that relies on a judicious choice of observable functions to realize global linear representations of nonlinear systems in the lifted observable space. However, real-world data is often noisy, making it difficult to obtain an accurate and unbiased approximation of the Koopman operator. The Koopman operator generated from noisy datasets is typically corrupted by noise-induced bias that severely degrades prediction and downstream tracking performance. In order to address this drawback, this paper proposes a novel autoencoder-based neural architecture to jointly learn the appropriate lifting functions and the reduced-bias Koopman operator from noisy data. The architecture initially learns the Koopman basis functions that are consistent for both the forward and backward temporal dynamics of the system. Subsequently, by utilizing the learned forward and backward temporal dynamics, the Koopman operator is synthesized with a reduced bias making the method more robust to noise compared to existing techniques. Theoretical analysis is used to demonstrate significant bias reduction in the presence of training noise. Dynamics prediction and tracking control simulations are conducted for multiple serial manipulator arms, including performance comparisons with leading alternative designs, to demonstrate its robustness under various noise levels. Experimental studies with the Franka FR3 7-DoF manipulator arm are further used to demonstrate the effectiveness of the proposed approach in a practical setting.
>
---
#### [new 028] Learning Diffusion Policy from Primitive Skills for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决扩散策略与动作生成不一致的问题。通过引入基础技能，提出SDP模型，实现更准确的技能引导控制。**

- **链接: [https://arxiv.org/pdf/2601.01948v1](https://arxiv.org/pdf/2601.01948v1)**

> **作者:** Zhihao Gu; Ming Yang; Difan Zou; Dong Xu
>
> **备注:** Accepted to AAAI2026
>
> **摘要:** Diffusion policies (DP) have recently shown great promise for generating actions in robotic manipulation. However, existing approaches often rely on global instructions to produce short-term control signals, which can result in misalignment in action generation. We conjecture that the primitive skills, referred to as fine-grained, short-horizon manipulations, such as ``move up'' and ``open the gripper'', provide a more intuitive and effective interface for robot learning. To bridge this gap, we propose SDP, a skill-conditioned DP that integrates interpretable skill learning with conditional action planning. SDP abstracts eight reusable primitive skills across tasks and employs a vision-language model to extract discrete representations from visual observations and language instructions. Based on them, a lightweight router network is designed to assign a desired primitive skill for each state, which helps construct a single-skill policy to generate skill-aligned actions. By decomposing complex tasks into a sequence of primitive skills and selecting a single-skill policy, SDP ensures skill-consistent behavior across diverse tasks. Extensive experiments on two challenging simulation benchmarks and real-world robot deployments demonstrate that SDP consistently outperforms SOTA methods, providing a new paradigm for skill-based robot learning with diffusion policies.
>
---
#### [new 029] CausalNav: A Long-term Embodied Navigation System for Autonomous Mobile Robots in Dynamic Outdoor Scenarios
- **分类: cs.RO**

- **简介: 该论文提出CausalNav，解决动态户外环境中的长期语义导航问题。通过构建多层级语义场景图，结合实时感知与离线地图，实现高效、稳定的自主导航。**

- **链接: [https://arxiv.org/pdf/2601.01872v1](https://arxiv.org/pdf/2601.01872v1)**

> **作者:** Hongbo Duan; Shangyi Luo; Zhiyuan Deng; Yanbo Chen; Yuanhao Chiang; Yi Liu; Fangming Liu; Xueqian Wang
>
> **备注:** Accepted by IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Autonomous language-guided navigation in large-scale outdoor environments remains a key challenge in mobile robotics, due to difficulties in semantic reasoning, dynamic conditions, and long-term stability. We propose CausalNav, the first scene graph-based semantic navigation framework tailored for dynamic outdoor environments. We construct a multi-level semantic scene graph using LLMs, referred to as the Embodied Graph, that hierarchically integrates coarse-grained map data with fine-grained object entities. The constructed graph serves as a retrievable knowledge base for Retrieval-Augmented Generation (RAG), enabling semantic navigation and long-range planning under open-vocabulary queries. By fusing real-time perception with offline map data, the Embodied Graph supports robust navigation across varying spatial granularities in dynamic outdoor environments. Dynamic objects are explicitly handled in both the scene graph construction and hierarchical planning modules. The Embodied Graph is continuously updated within a temporal window to reflect environmental changes and support real-time semantic navigation. Extensive experiments in both simulation and real-world settings demonstrate superior robustness and efficiency.
>
---
#### [new 030] DST-Calib: A Dual-Path, Self-Supervised, Target-Free LiDAR-Camera Extrinsic Calibration Network
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于LiDAR-相机外参标定任务，解决传统方法依赖标定目标和静态场景的问题。提出一种自监督、双路径的在线标定网络，提升泛化性和适应性。**

- **链接: [https://arxiv.org/pdf/2601.01188v1](https://arxiv.org/pdf/2601.01188v1)**

> **作者:** Zhiwei Huang; Yanwei Fu; Yi Zhou; Xieyuanli Chen; Qijun Chen; Rui Fan
>
> **摘要:** LiDAR-camera extrinsic calibration is essential for multi-modal data fusion in robotic perception systems. However, existing approaches typically rely on handcrafted calibration targets (e.g., checkerboards) or specific, static scene types, limiting their adaptability and deployment in real-world autonomous and robotic applications. This article presents the first self-supervised LiDAR-camera extrinsic calibration network that operates in an online fashion and eliminates the need for specific calibration targets. We first identify a significant generalization degradation problem in prior methods, caused by the conventional single-sided data augmentation strategy. To overcome this limitation, we propose a novel double-sided data augmentation technique that generates multi-perspective camera views using estimated depth maps, thereby enhancing robustness and diversity during training. Built upon this augmentation strategy, we design a dual-path, self-supervised calibration framework that reduces the dependence on high-precision ground truth labels and supports fully adaptive online calibration. Furthermore, to improve cross-modal feature association, we replace the traditional dual-branch feature extraction design with a difference map construction process that explicitly correlates LiDAR and camera features. This not only enhances calibration accuracy but also reduces model complexity. Extensive experiments conducted on five public benchmark datasets, as well as our own recorded dataset, demonstrate that the proposed method significantly outperforms existing approaches in terms of generalizability.
>
---
#### [new 031] What you reward is what you learn: Comparing rewards for online speech policy optimization in public HRI
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在优化公共场景下服务机器人的语音策略。通过在线学习和多臂老虎机方法，比较不同奖励机制对交互效果的影响，提出实际部署的设计建议。**

- **链接: [https://arxiv.org/pdf/2601.01969v1](https://arxiv.org/pdf/2601.01969v1)**

> **作者:** Sichao Song; Yuki Okafuji; Kaito Ariu; Amy Koike
>
> **摘要:** Designing policies that are both efficient and acceptable for conversational service robots in open and diverse environments is non-trivial. Unlike fixed, hand-tuned parameters, online learning can adapt to non-stationary conditions. In this paper, we study how to adapt a social robot's speech policy in the wild. During a 12-day in-situ deployment with over 1,400 public encounters, we cast online policy optimization as a multi-armed bandit problem and use Thompson sampling to select among six actions defined by speech rate (slow/normal/fast) and verbosity (concise/detailed). We compare three complementary binary rewards--Ru (user rating), Rc (conversation closure), and Rt (>=2 turns)--and show that each induces distinct arm distributions and interaction behaviors. We complement the online results with offline evaluations that analyze contextual factors (e.g., crowd level, group size) using video-annotated data. Taken together, we distill ready-to-use design lessons for deploying online optimization of speech policies in real public HRI settings.
>
---
#### [new 032] Realistic adversarial scenario generation via human-like pedestrian model for autonomous vehicle control parameter optimisation
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于自动驾驶安全测试任务，旨在解决传统模拟场景过于复杂而缺乏真实性的难题。通过引入类人行人模型生成更真实的对抗场景，优化AV控制策略。**

- **链接: [https://arxiv.org/pdf/2601.02082v1](https://arxiv.org/pdf/2601.02082v1)**

> **作者:** Yueyang Wang; Mehmet Dogar; Gustav Markkula
>
> **摘要:** Autonomous vehicles (AVs) are rapidly advancing and are expected to play a central role in future mobility. Ensuring their safe deployment requires reliable interaction with other road users, not least pedestrians. Direct testing on public roads is costly and unsafe for rare but critical interactions, making simulation a practical alternative. Within simulation-based testing, adversarial scenarios are widely used to probe safety limits, but many prioritise difficulty over realism, producing exaggerated behaviours which may result in AV controllers that are overly conservative. We propose an alternative method, instead using a cognitively inspired pedestrian model featuring both inter-individual and intra-individual variability to generate behaviourally plausible adversarial scenarios. We provide a proof of concept demonstration of this method's potential for AV control optimisation, in closed-loop testing and tuning of an AV controller. Our results show that replacing the rule-based CARLA pedestrian with the human-like model yields more realistic gap acceptance patterns and smoother vehicle decelerations. Unsafe interactions occur only for certain pedestrian individuals and conditions, underscoring the importance of human variability in AV testing. Adversarial scenarios generated by this model can be used to optimise AV control towards safer and more efficient behaviour. Overall, this work illustrates how incorporating human-like road user models into simulation-based adversarial testing can enhance the credibility of AV evaluation and provide a practical basis to behaviourally informed controller optimisation.
>
---
#### [new 033] Scalable Data-Driven Reachability Analysis and Control via Koopman Operators with Conformal Coverage Guarantees
- **分类: eess.SY; cs.AI; cs.LG; cs.RO; math.OC**

- **简介: 该论文属于安全验证任务，解决未知非线性动态系统的概率安全性问题。通过Koopman算子和神经网络构建线性模型，设计控制器并利用合规预测保证覆盖范围。**

- **链接: [https://arxiv.org/pdf/2601.01076v1](https://arxiv.org/pdf/2601.01076v1)**

> **作者:** Devesh Nath; Haoran Yin; Glen Chou
>
> **备注:** Under review, 28 pages, 12 figures
>
> **摘要:** We propose a scalable reachability-based framework for probabilistic, data-driven safety verification of unknown nonlinear dynamics. We use Koopman theory with a neural network (NN) lifting function to learn an approximate linear representation of the dynamics and design linear controllers in this space to enable closed-loop tracking of a reference trajectory distribution. Closed-loop reachable sets are efficiently computed in the lifted space and mapped back to the original state space via NN verification tools. To capture model mismatch between the Koopman dynamics and the true system, we apply conformal prediction to produce statistically-valid error bounds that inflate the reachable sets to ensure the true trajectories are contained with a user-specified probability. These bounds generalize across references, enabling reuse without recomputation. Results on high-dimensional MuJoCo tasks (11D Hopper, 28D Swimmer) and 12D quadcopters show improved reachable set coverage rate, computational efficiency, and conservativeness over existing methods.
>
---
#### [new 034] Analyzing the Shopping Journey: Computing Shelf Browsing Visits in a Physical Retail Store
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于行为分析任务，旨在通过算法识别顾客在实体店的货架浏览行为，解决如何理解购物者意图的问题。工作包括算法开发、模型校准与评估，以及对浏览模式与购买关系的分析。**

- **链接: [https://arxiv.org/pdf/2601.00928v1](https://arxiv.org/pdf/2601.00928v1)**

> **作者:** Luis Yoichi Morales; Francesco Zanlungo; David M. Woollard
>
> **摘要:** Motivated by recent challenges in the deployment of robots into customer-facing roles within retail, this work introduces a study of customer activity in physical stores as a step toward autonomous understanding of shopper intent. We introduce an algorithm that computes shoppers' ``shelf visits'' -- capturing their browsing behavior in the store. Shelf visits are extracted from trajectories obtained via machine vision-based 3D tracking and overhead cameras. We perform two independent calibrations of the shelf visit algorithm, using distinct sets of trajectories (consisting of 8138 and 15129 trajectories), collected in different stores and labeled by human reviewers. The calibrated models are then evaluated on trajectories held out of the calibration process both from the same store on which calibration was performed and from the other store. An analysis of the results shows that the algorithm can recognize customers' browsing activity when evaluated in an environment different from the one on which calibration was performed. We then use the model to analyze the customers' ``browsing patterns'' on a large set of trajectories and their relation to actual purchases in the stores. Finally, we discuss how shelf browsing information could be used for retail planning and in the domain of human-robot interaction scenarios.
>
---
#### [new 035] Contractive Diffusion Policies: Robust Action Diffusion via Contractive Score-Based Sampling with Differential Equations
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决扩散策略在连续控制中的稳定性问题。通过引入收缩扩散策略，提升对误差的鲁棒性并减少动作方差。**

- **链接: [https://arxiv.org/pdf/2601.01003v1](https://arxiv.org/pdf/2601.01003v1)**

> **作者:** Amin Abyaneh; Charlotte Morissette; Mohamad H. Danesh; Anas El Houssaini; David Meger; Gregory Dudek; Hsiu-Chin Lin
>
> **备注:** Under review at ICLR 2026
>
> **摘要:** Diffusion policies have emerged as powerful generative models for offline policy learning, whose sampling process can be rigorously characterized by a score function guiding a Stochastic Differential Equation (SDE). However, the same score-based SDE modeling that grants diffusion policies the flexibility to learn diverse behavior also incurs solver and score-matching errors, large data requirements, and inconsistencies in action generation. While less critical in image generation, these inaccuracies compound and lead to failure in continuous control settings. We introduce Contractive Diffusion Policies (CDPs) to induce contractive behavior in the diffusion sampling dynamics. Contraction pulls nearby flows closer to enhance robustness against solver and score-matching errors while reducing unwanted action variance. We develop an in-depth theoretical analysis along with a practical implementation recipe to incorporate CDPs into existing diffusion policy architectures with minimal modification and computational cost. We evaluate CDPs for offline learning by conducting extensive experiments in simulation and real-world settings. Across benchmarks, CDPs often outperform baseline policies, with pronounced benefits under data scarcity.
>
---
#### [new 036] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶领域，旨在解决生成式视频世界模型的评估问题。提出DrivingGen基准，涵盖多样数据和新指标，评估视觉真实、轨迹合理性等，推动可靠模型发展。**

- **链接: [https://arxiv.org/pdf/2601.01528v1](https://arxiv.org/pdf/2601.01528v1)**

> **作者:** Yang Zhou; Hao Shao; Letian Wang; Zhuofan Zong; Hongsheng Li; Steven L. Waslander
>
> **备注:** 10 pages, 4 figures; Project Website: https://drivinggen-bench.github.io/
>
> **摘要:** Video generation models, as one form of world models, have emerged as one of the most exciting frontiers in AI, promising agents the ability to imagine the future by modeling the temporal evolution of complex scenes. In autonomous driving, this vision gives rise to driving world models: generative simulators that imagine ego and agent futures, enabling scalable simulation, safe testing of corner cases, and rich synthetic data generation. Yet, despite fast-growing research activity, the field lacks a rigorous benchmark to measure progress and guide priorities. Existing evaluations remain limited: generic video metrics overlook safety-critical imaging factors; trajectory plausibility is rarely quantified; temporal and agent-level consistency is neglected; and controllability with respect to ego conditioning is ignored. Moreover, current datasets fail to cover the diversity of conditions required for real-world deployment. To address these gaps, we present DrivingGen, the first comprehensive benchmark for generative driving world models. DrivingGen combines a diverse evaluation dataset curated from both driving datasets and internet-scale video sources, spanning varied weather, time of day, geographic regions, and complex maneuvers, with a suite of new metrics that jointly assess visual realism, trajectory plausibility, temporal coherence, and controllability. Benchmarking 14 state-of-the-art models reveals clear trade-offs: general models look better but break physics, while driving-specific ones capture motion realistically but lag in visual quality. DrivingGen offers a unified evaluation framework to foster reliable, controllable, and deployable driving world models, enabling scalable simulation, planning, and data-driven decision-making.
>
---
#### [new 037] LiveBo: Empowering Non-Chinese Speaking Students through AI-Driven Real-Life Scenarios in Cantonese
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于语言教育任务，旨在解决非中文学生学习粤语的困难。通过AI驱动的实景模拟和社交机器人LiveBo提升学习者参与度和语言能力。**

- **链接: [https://arxiv.org/pdf/2601.01227v1](https://arxiv.org/pdf/2601.01227v1)**

> **作者:** Ka Yan Fung; Kwong Chiu Fung; Yuxing Tao; Tze Leung Rick Lui; Kuen Fung Sin
>
> **摘要:** Language learning is a multifaceted process. Insufficient vocabulary can hinder communication and lead to demotivation. For non-Chinese speaking (NCS) students, learning Traditional Chinese (Cantonese) poses distinct challenges, particularly due to the complexity of converting spoken and written forms. To address this issue, this study examines the effectiveness of real-life scenario simulations integrated with interactive social robots in enhancing NCS student engagement and language acquisition. The research employs a quasi-experimental design involving NCS students who interact with an AI-driven, robot-assisted language learning system, LiveBo. The study aims to assess the impact of this innovative approach on active participation and motivation. Data are collected through proficiency tests, questionnaires and semi-structured interviews. Findings indicate that NCS students experience positive improvements in behavioural and emotional engagement, motivation and learning outcomes, highlighting the potential of integrating novel technologies in language education. We plan to compare with the control group in the future. This study highlights the significance of interactive and immersive learning experiences in promoting motivation and enhancing language acquisition among NCS students.
>
---
#### [new 038] Value-guided action planning with JEPA world models
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决JEPA模型在行动规划中的性能不足问题。通过优化表示空间，提升规划效果。**

- **链接: [https://arxiv.org/pdf/2601.00844v1](https://arxiv.org/pdf/2601.00844v1)**

> **作者:** Matthieu Destrade; Oumayma Bounou; Quentin Le Lidec; Jean Ponce; Yann LeCun
>
> **备注:** Presented as a poster at the World Modeling Workshop 2026, Mila
>
> **摘要:** Building deep learning models that can reason about their environment requires capturing its underlying dynamics. Joint-Embedded Predictive Architectures (JEPA) provide a promising framework to model such dynamics by learning representations and predictors through a self-supervised prediction objective. However, their ability to support effective action planning remains limited. We propose an approach to enhance planning with JEPA world models by shaping their representation space so that the negative goal-conditioned value function for a reaching cost in a given environment is approximated by a distance (or quasi-distance) between state embeddings. We introduce a practical method to enforce this constraint during training and show that it leads to significantly improved planning performance compared to standard JEPA models on simple control tasks.
>
---
#### [new 039] Real-Time Lane Detection via Efficient Feature Alignment and Covariance Optimization for Low-Power Embedded Systems
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于实时车道检测任务，针对嵌入式系统中计算资源有限的问题，提出CDO模块提升检测精度，无需增加计算复杂度。**

- **链接: [https://arxiv.org/pdf/2601.01696v1](https://arxiv.org/pdf/2601.01696v1)**

> **作者:** Yian Liu; Xiong Wang; Ping Xu; Lei Zhu; Ming Yan; Linyun Xue
>
> **摘要:** Real-time lane detection in embedded systems encounters significant challenges due to subtle and sparse visual signals in RGB images, often constrained by limited computational resources and power consumption. Although deep learning models for lane detection categorized into segmentation-based, anchor-based, and curve-based methods there remains a scarcity of universally applicable optimization techniques tailored for low-power embedded environments. To overcome this, we propose an innovative Covariance Distribution Optimization (CDO) module specifically designed for efficient, real-time applications. The CDO module aligns lane feature distributions closely with ground-truth labels, significantly enhancing detection accuracy without increasing computational complexity. Evaluations were conducted on six diverse models across all three method categories, including two optimized for real-time applications and four state-of-the-art (SOTA) models, tested comprehensively on three major datasets: CULane, TuSimple, and LLAMAS. Experimental results demonstrate accuracy improvements ranging from 0.01% to 1.5%. The proposed CDO module is characterized by ease of integration into existing systems without structural modifications and utilizes existing model parameters to facilitate ongoing training, thus offering substantial benefits in performance, power efficiency, and operational flexibility in embedded systems.
>
---
#### [new 040] MotiBo: The Impact of Interactive Digital Storytelling Robots on Student Motivation through Self-Determination Theory
- **分类: cs.HC; cs.MM; cs.RO**

- **简介: 该论文属于教育技术任务，旨在解决传统 storytelling 缺乏互动性的问题。通过设计 MotiBo 系统，比较不同教学方式对学生参与度的影响。**

- **链接: [https://arxiv.org/pdf/2601.01218v1](https://arxiv.org/pdf/2601.01218v1)**

> **作者:** Ka Yan Fung; Tze Leung Rick Lui; Yuxing Tao; Kuen Fung Sin
>
> **摘要:** Creativity is increasingly recognized as an important skill in education, and storytelling can enhance motivation and engagement among students. However, conventional storytelling methods often lack the interactive elements necessary to engage students. To this end, this study examines the impact of an interactive digital storytelling system incorporating a human-like robot on student engagement and creativity. The study aims to compare engagement levels across three modalities: paper-based, PowerPoint, and robot-assisted storytelling, MotiBo. Utilizing a quasi-experimental design, this work involves three groups of students who interact with the storytelling system over a five-day learning. Findings reveal that students using MotiBo exhibit statistically significant improvement in behavioural and cognitive engagement compared to those using traditional methods. These results suggest that the integration of novel technologies can effectively enhance the learning experience, ultimately promoting creativity and self-learning ability in educational settings. Future research will investigate the long-term effects of these technologies on learning outcomes and explore their potential for broader applications in diverse educational contexts.
>
---
#### [new 041] Sampling Strategy Design for Model Predictive Path Integral Control on Legged Robot Locomotion
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决MPPI在腿式机器人运动中的应用问题。通过设计不同采样策略，提升控制性能与效率。**

- **链接: [https://arxiv.org/pdf/2601.01409v1](https://arxiv.org/pdf/2601.01409v1)**

> **作者:** Chuyuan Tao; Fanxin Wang; Haolong Jiang; Jia He; Yiyang Chen; Qinglei Bu
>
> **摘要:** Model Predictive Path Integral (MPPI) control has emerged as a powerful sampling-based optimal control method for complex, nonlinear, and high-dimensional systems. However, directly applying MPPI to legged robotic systems presents several challenges. This paper systematically investigates the role of sampling strategy design within the MPPI framework for legged robot locomotion. Based upon the idea of structured control parameterization, we explore and compare multiple sampling strategies within the framework, including both unstructured and spline-based approaches. Through extensive simulations on a quadruped robot platform, we evaluate how different sampling strategies affect control smoothness, task performance, robustness, and sample efficiency. The results provide new insights into the practical implications of sampling design for deploying MPPI on complex legged systems.
>
---
#### [new 042] Bridging Language Gaps: Utilizing Interactive Robots to Teach Cantonese in Real-Life Contexts for Newly-Arrived Children
- **分类: cs.ET; cs.HC; cs.RO**

- **简介: 论文探讨使用互动机器人教学，帮助新来港学生学习粤语，解决语言障碍和文化融入问题。任务属于教育技术领域，旨在提升学习参与度与语言能力。**

- **链接: [https://arxiv.org/pdf/2601.01234v1](https://arxiv.org/pdf/2601.01234v1)**

> **作者:** Ka-Yan Fung; Yuxing Tao; Tze-Leung; Rick Lui; Kuen-Fung Sin
>
> **摘要:** Hong Kong's education system is notably multicultural, including local, non-Chinese-speaking, and newly arrived students (NAS) (Mandarine Chinese-speaking). NAS can guess the meaning of vocabulary but cannot speak out, presenting unique challenges for them, particularly language barriers and cultural differences. These challenges hinder their academic success and social integration, leading to feelings of isolation and demotivation. Current resources often fail to address the emotional well-being of these students and predominantly focus on English language acquisition, leaving a gap in support for learning Cantonese and navigating the local cultural landscape. This study explores the effectiveness of an interactive robot, Boon Boon, in teaching Cantonese through real-life contexts to enhance NAS children learning engagement and motivation. The research questions are: (1) How does interactive robot-empowered scenario learning influence the learning engagement and motivation of NAS in learning Cantonese? and (2) What is the impact of a robot-empowered scenario learning system on the Cantonese language proficiency of NAS? Fourteen children are invited to participate in a four-day learning program with Boon Boon. The preliminary result indicated that Boon Boon drove students' attention to learning and academic achievement. Future research will focus on long-term assessments of robot-empowered learning's effectiveness and explore the scalability of this approach across diverse educational settings and cultural backgrounds.
>
---
#### [new 043] Dichotomous Diffusion Policy Optimization
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出DIPOLE算法，解决扩散策略在强化学习中训练不稳定和计算复杂的问题。通过分解策略为最大化与最小化两部分，实现稳定且可控的策略优化。**

- **链接: [https://arxiv.org/pdf/2601.00898v1](https://arxiv.org/pdf/2601.00898v1)**

> **作者:** Ruiming Liang; Yinan Zheng; Kexin Zheng; Tianyi Tan; Jianxiong Li; Liyuan Mao; Zhihao Wang; Guang Chen; Hangjun Ye; Jingjing Liu; Jinqiao Wang; Xianyuan Zhan
>
> **摘要:** Diffusion-based policies have gained growing popularity in solving a wide range of decision-making tasks due to their superior expressiveness and controllable generation during inference. However, effectively training large diffusion policies using reinforcement learning (RL) remains challenging. Existing methods either suffer from unstable training due to directly maximizing value objectives, or face computational issues due to relying on crude Gaussian likelihood approximation, which requires a large amount of sufficiently small denoising steps. In this work, we propose DIPOLE (Dichotomous diffusion Policy improvement), a novel RL algorithm designed for stable and controllable diffusion policy optimization. We begin by revisiting the KL-regularized objective in RL, which offers a desirable weighted regression objective for diffusion policy extraction, but often struggles to balance greediness and stability. We then formulate a greedified policy regularization scheme, which naturally enables decomposing the optimal policy into a pair of stably learned dichotomous policies: one aims at reward maximization, and the other focuses on reward minimization. Under such a design, optimized actions can be generated by linearly combining the scores of dichotomous policies during inference, thereby enabling flexible control over the level of greediness.Evaluations in offline and offline-to-online RL settings on ExORL and OGBench demonstrate the effectiveness of our approach. We also use DIPOLE to train a large vision-language-action (VLA) model for end-to-end autonomous driving (AD) and evaluate it on the large-scale real-world AD benchmark NAVSIM, highlighting its potential for complex real-world applications.
>
---
#### [new 044] VIT-Ped: Visionary Intention Transformer for Pedestrian Behavior Analysis
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于行人行为分析任务，旨在预测行人意图以提升自动驾驶安全性。提出基于Transformer的算法，融合多模态数据，在JAAD数据集上取得优异效果。**

- **链接: [https://arxiv.org/pdf/2601.01989v1](https://arxiv.org/pdf/2601.01989v1)**

> **作者:** Aly R. Elkammar; Karim M. Gamaleldin; Catherine M. Elias
>
> **摘要:** Pedestrian Intention prediction is one of the key technologies in the transition from level 3 to level 4 autonomous driving. To understand pedestrian crossing behaviour, several elements and features should be taken into consideration to make the roads of tomorrow safer for everybody. We introduce a transformer / video vision transformer based algorithm of different sizes which uses different data modalities .We evaluated our algorithms on popular pedestrian behaviour dataset, JAAD, and have reached SOTA performance and passed the SOTA in metrics like Accuracy, AUC and F1-score. The advantages brought by different model design choices are investigated via extensive ablation studies.
>
---
#### [new 045] Real-Time LiDAR Point Cloud Densification for Low-Latency Spatial Data Transmission
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于三维重建任务，旨在解决LiDAR点云稀疏和实时处理问题。通过融合多传感器数据，提出一种快速点云增强方法，实现高分辨率、低延迟的深度图生成。**

- **链接: [https://arxiv.org/pdf/2601.01210v1](https://arxiv.org/pdf/2601.01210v1)**

> **作者:** Kazuhiko Murasaki; Shunsuke Konagai; Masakatsu Aoki; Taiga Yoshida; Ryuichi Tanida
>
> **摘要:** To realize low-latency spatial transmission system for immersive telepresence, there are two major problems: capturing dynamic 3D scene densely and processing them in real time. LiDAR sensors capture 3D in real time, but produce sparce point clouds. Therefore, this paper presents a high-speed LiDAR point cloud densification method to generate dense 3D scene with minimal latency, addressing the need for on-the-fly depth completion while maintaining real-time performance. Our approach combines multiple LiDAR inputs with high-resolution color images and applies a joint bilateral filtering strategy implemented through a convolutional neural network architecture. Experiments demonstrate that the proposed method produces dense depth maps at full HD resolution in real time (30 fps), which is over 15x faster than a recent training-based depth completion approach. The resulting dense point clouds exhibit accurate geometry without multiview inconsistencies or ghosting artifacts.
>
---
#### [new 046] PyBatchRender: A Python Library for Batched 3D Rendering at Up to One Million FPS
- **分类: cs.GR; cs.AI; cs.PF; cs.RO**

- **简介: 该论文提出PyBatchRender，解决高帧率3D渲染难题，用于强化学习中的高效环境模拟。**

- **链接: [https://arxiv.org/pdf/2601.01288v1](https://arxiv.org/pdf/2601.01288v1)**

> **作者:** Evgenii Rudakov; Jonathan Shock; Benjamin Ultan Cowley
>
> **摘要:** Reinforcement learning from pixels is often bottlenecked by the performance and complexity of 3D rendered environments. Researchers face a trade-off between high-speed, low-level engines and slower, more accessible Python frameworks. To address this, we introduce PyBatchRender, a Python library for high-throughput, batched 3D rendering that achieves over 1 million FPS on simple scenes. Built on the Panda3D game engine, it utilizes its mature ecosystem while enhancing performance through optimized batched rendering for up to 1000X speedups. Designed as a physics-agnostic renderer for reinforcement learning from pixels, PyBatchRender offers greater flexibility than dedicated libraries, simpler setup than typical game-engine wrappers, and speeds rivaling state-of-the-art C++ engines like Madrona. Users can create custom scenes entirely in Python with tens of lines of code, enabling rapid prototyping for scalable AI training. Open-source and easy to integrate, it serves to democratize high-performance 3D simulation for researchers and developers. The library is available at https://github.com/dolphin-in-a-coma/PyBatchRender.
>
---
## 更新

#### [replaced 001] Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决传统系统在复杂场景下的局限性。通过分析视觉-语言-动作模型，提出两种框架并总结挑战与方向。**

- **链接: [https://arxiv.org/pdf/2512.16760v2](https://arxiv.org/pdf/2512.16760v2)**

> **作者:** Tianshuai Hu; Xiaolu Liu; Song Wang; Yiyao Zhu; Ao Liang; Lingdong Kong; Guoyang Zhao; Zeying Gong; Jun Cen; Zhiyu Huang; Xiaoshuai Hao; Linfeng Li; Hang Song; Xiangtai Li; Jun Ma; Shaojie Shen; Jianke Zhu; Dacheng Tao; Ziwei Liu; Junwei Liang
>
> **备注:** Survey; 47 pages, 7 figures, 9 tables; GitHub Repo at https://github.com/worldbench/awesome-vla-for-ad
>
> **摘要:** Autonomous driving has long relied on modular "Perception-Decision-Action" pipelines, where hand-crafted interfaces and rule-based components often break down in complex or long-tailed scenarios. Their cascaded design further propagates perception errors, degrading downstream planning and control. Vision-Action (VA) models address some limitations by learning direct mappings from visual inputs to actions, but they remain opaque, sensitive to distribution shifts, and lack structured reasoning or instruction-following capabilities. Recent progress in Large Language Models (LLMs) and multimodal learning has motivated the emergence of Vision-Language-Action (VLA) frameworks, which integrate perception with language-grounded decision making. By unifying visual understanding, linguistic reasoning, and actionable outputs, VLAs offer a pathway toward more interpretable, generalizable, and human-aligned driving policies. This work provides a structured characterization of the emerging VLA landscape for autonomous driving. We trace the evolution from early VA approaches to modern VLA frameworks and organize existing methods into two principal paradigms: End-to-End VLA, which integrates perception, reasoning, and planning within a single model, and Dual-System VLA, which separates slow deliberation (via VLMs) from fast, safety-critical execution (via planners). Within these paradigms, we further distinguish subclasses such as textual vs. numerical action generators and explicit vs. implicit guidance mechanisms. We also summarize representative datasets and benchmarks for evaluating VLA-based driving systems and highlight key challenges and open directions, including robustness, interpretability, and instruction fidelity. Overall, this work aims to establish a coherent foundation for advancing human-compatible autonomous driving systems.
>
---
#### [replaced 002] Interconnection and Damping Assignment Passivity-Based Control using Sparse Neural ODEs
- **分类: cs.RO**

- **简介: 该论文属于控制理论领域，解决IDA-PBC控制器设计中难以解析求解匹配PDE的问题。通过神经ODE和稀疏学习方法，实现复杂任务的控制器设计与稳定性分析。**

- **链接: [https://arxiv.org/pdf/2512.06935v2](https://arxiv.org/pdf/2512.06935v2)**

> **作者:** Nicolò Botteghi; Owen Brook; Urban Fasel; Federico Califano
>
> **摘要:** Interconnection and Damping Assignment Passivity-Based Control (IDA-PBC) is a nonlinear control technique that assigns a port-Hamiltonian (pH) structure to a controlled system using a state-feedback law. While IDA-PBC has been extensively studied and applied to many systems, its practical implementation often remains confined to academic examples and, almost exclusively, to stabilization tasks. The main limitation of IDA-PBC stems from the complexity of analytically solving a set of partial differential equations (PDEs), referred to as the matching conditions, which enforce the pH structure of the closed-loop system. However, this is extremely challenging, especially for complex physical systems and tasks. In this work, we propose a novel numerical approach for designing IDA-PBC controllers without solving the matching PDEs exactly. We cast the IDA-PBC problem as the learning of a neural ordinary differential equation. In particular, we rely on sparse dictionary learning to parametrize the desired closed-loop system as a sparse linear combination of nonlinear state-dependent functions. Optimization of the controller parameters is achieved by solving a multi-objective optimization problem whose cost function is composed of a generic task-dependent cost and a matching condition-dependent cost. Our numerical results show that the proposed method enables (i) IDA-PBC to be applicable to complex tasks beyond stabilization, such as the discovery of periodic oscillatory behaviors, (ii) the derivation of closed-form expressions of the controlled system, including residual terms in case of approximate matching, and (iii) stability analysis of the learned controller.
>
---
#### [replaced 003] General Dynamic Goal Recognition using Goal-Conditioned and Meta Reinforcement Learning
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于目标识别任务，解决动态环境中实时适应的目标识别问题。提出两种方法，利用无模型目标条件强化学习和元强化学习，实现快速适应与高精度识别。**

- **链接: [https://arxiv.org/pdf/2505.09737v2](https://arxiv.org/pdf/2505.09737v2)**

> **作者:** Osher Elhadad; Owen Morrissey; Reuth Mirsky
>
> **备注:** Accepted for publication at AAMAS 2026
>
> **摘要:** Understanding an agent's goal through its behavior is a common AI problem called Goal Recognition (GR). This task becomes particularly challenging in dynamic environments where goals are numerous and ever-changing. We introduce the General Dynamic Goal Recognition (GDGR) problem, a broader definition of GR aimed at real-time adaptation of GR systems. This paper presents two novel approaches to tackle GDGR: (1) GC-AURA, generalizing to new goals using Model-Free Goal-Conditioned Reinforcement Learning, and (2) Meta-AURA, adapting to novel environments with Meta-Reinforcement Learning. We evaluate these methods across diverse environments, demonstrating their ability to achieve rapid adaptation and high GR accuracy under dynamic and noisy conditions. This work is a significant step forward in enabling GR in dynamic and unpredictable real-world environments.
>
---
#### [replaced 004] Towards Balanced Behavior Cloning from Imbalanced Datasets
- **分类: cs.RO**

- **简介: 该论文属于模仿学习任务，旨在解决数据不平衡导致的策略偏差问题。通过重新加权数据集，提升多任务行为的模仿效果。**

- **链接: [https://arxiv.org/pdf/2508.06319v2](https://arxiv.org/pdf/2508.06319v2)**

> **作者:** Sagar Parekh; Heramb Nemlekar; Dylan P. Losey
>
> **摘要:** Robots should be able to learn complex behaviors from human demonstrations. In practice, these human-provided datasets are inevitably imbalanced: i.e., the human demonstrates some subtasks more frequently than others. State-of-the-art methods default to treating each element of the human's dataset as equally important. So if -- for instance -- the majority of the human's data focuses on reaching a goal, and only a few state-action pairs move to avoid an obstacle, the learning algorithm will place greater emphasis on goal reaching. More generally, misalignment between the relative amounts of data and the importance of that data causes fundamental problems for imitation learning approaches. In this paper we analyze and develop learning methods that automatically account for mixed datasets. We formally prove that imbalanced data leads to imbalanced policies when each state-action pair is weighted equally; these policies emulate the most represented behaviors, and not the human's complex, multi-task demonstrations. We next explore algorithms that rebalance offline datasets (i.e., reweight the importance of different state-action pairs) without human oversight. Reweighting the dataset can enhance the overall policy performance. However, there is no free lunch: each method for autonomously rebalancing brings its own pros and cons. We formulate these advantages and disadvantages, helping other researchers identify when each type of approach is most appropriate. We conclude by introducing a novel meta-gradient rebalancing algorithm that addresses the primary limitations behind existing approaches. Our experiments show that dataset rebalancing leads to better downstream learning, improving the performance of general imitation learning algorithms without requiring additional data collection. See our project website: https://collab.me.vt.edu/data_curation/.
>
---
#### [replaced 005] Stochastic Online Optimization for Cyber-Physical and Robotic Systems
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于在线优化任务，解决 cyber-physical 和机器人系统中的随机优化问题，提出基于梯度的框架并分析其收敛性。**

- **链接: [https://arxiv.org/pdf/2404.05318v2](https://arxiv.org/pdf/2404.05318v2)**

> **作者:** Hao Ma; Melanie Zeilinger; Michael Muehlebach
>
> **备注:** 46 pages, 16 figures
>
> **摘要:** We propose a novel gradient-based online optimization framework for solving stochastic programming problems that frequently arise in the context of cyber-physical and robotic systems. Our problem formulation accommodates constraints that model the evolution of a cyber-physical system, which has, in general, a continuous state and action space, is nonlinear, and where the state is only partially observed. We also incorporate an approximate model of the dynamics as prior knowledge into the learning process and show that even rough estimates of the dynamics can significantly improve the convergence of our algorithms. Our online optimization framework encompasses both gradient descent and quasi-Newton methods, and we provide a unified convergence analysis of our algorithms in a non-convex setting. We also characterize the impact of modeling errors in the system dynamics on the convergence rate of the algorithms. Finally, we evaluate our algorithms in simulations of a flexible beam, a four-legged walking robot, and in real-world experiments with a ping-pong playing robot.
>
---
#### [replaced 006] No Minima, No Collisions: Combining Modulation and Control Barrier Function Strategies for Feasible Dynamic Collision Avoidance
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于动态避障任务，解决CBF-QP存在局部极小值的问题。通过结合调制与CBF方法，提出MCBF-QP框架，有效消除非期望平衡点。**

- **链接: [https://arxiv.org/pdf/2502.14238v3](https://arxiv.org/pdf/2502.14238v3)**

> **作者:** Yifan Xue; Nadia Figueroa
>
> **摘要:** Control Barrier Function Quadratic Programs (CBF-QPs) have become a central tool for real-time safety-critical control due to their applicability to general control-affine systems and their ability to enforce constraints through optimization. Yet, they often generate trajectories with undesirable local minima that prevent convergence to goals. On the other hand, Modulation of Dynamical Systems (Mod-DS) methods (including normal, reference, and on-manifold variants) reshape nominal vector fields geometrically and achieve obstacle avoidance with few or even no local minima. However, Mod-DS provides no straightforward mechanism for handling input constraints and remains largely restricted to fully actuated systems. In this paper, we revisit the theoretical foundations of both approaches and show that, despite their seemingly different constructions, the normal Mod-DS is a special case of the CBF-QP, and the reference Mod-DS is linked to the CBF-QP through a single shared equation. These connections motivate our Modulated CBF-QP (MCBF-QP) framework, which introduces reference and on-manifold modulation variants that reduce or fully eliminate the spurious equilibria inherent to CBF-QPs for general control-affine systems operating in dynamic, cluttered environments. We validate the proposed controllers in simulated hospital settings and in real-world experiments on fully actuated Ridgeback robots and underactuated Fetch platforms. Across all evaluations, Modulated CBF-QPs consistently outperform standard CBF-QPs on every performance metric.
>
---
#### [replaced 007] Multimodal Classification Network Guided Trajectory Planning for Four-Wheel Independent Steering Autonomous Parking Considering Obstacle Attributes
- **分类: cs.RO**

- **简介: 该论文属于自主泊车任务，解决4WIS车辆在复杂环境中的轨迹规划问题。通过融合感知与规划，引入障碍物属性处理和风险模型，提升路径效率与安全性。**

- **链接: [https://arxiv.org/pdf/2512.18836v2](https://arxiv.org/pdf/2512.18836v2)**

> **作者:** Jingjia Teng; Yang Li; Yougang Bian; Manjiang Hu; Yingbai Hu; Guofa Li; Jianqiang Wang
>
> **摘要:** Four-wheel Independent Steering (4WIS) vehicles have attracted increasing attention for their superior maneuverability. Human drivers typically choose to cross or drive over the low-profile obstacles (e.g., plastic bags) to efficiently navigate through narrow spaces, while existing planners neglect obstacle attributes, leading to suboptimal efficiency or planning failures. To address this issue, we propose a novel multimodal trajectory planning framework that employs a neural network for scene perception, combines 4WIS hybrid A* search to generate a warm start, and utilizes an optimal control problem (OCP) for trajectory optimization. Specifically, a multimodal perception network fusing visual information and vehicle states is employed to capture semantic and contextual scene understanding, enabling the planner to adapt the strategy according to scene complexity (hard or easy task). For hard tasks, guided points are introduced to decompose complex tasks into local subtasks, improving the search efficiency. The multiple steering modes of 4WIS vehicles, Ackermann, diagonal, and zero-turn, are also incorporated as kinematically feasible motion primitives. Moreover, a hierarchical obstacle handling strategy, which categorizes obstacles as "non-traversable", "crossable", and "drive-over", is incorporated into the node expansion process, explicitly linking obstacle attributes to planning actions to enable efficient decisions. Furthermore, to address dynamic obstacles with motion uncertainty, we introduce a probabilistic risk field model, constructing risk-aware driving corridors that serve as linear collision constraints in OCP. Experimental results demonstrate the proposed framework's effectiveness in generating safe, efficient, and smooth trajectories for 4WIS vehicles, especially in constrained environments.
>
---
#### [replaced 008] FORTE: Tactile Force and Slip Sensing on Compliant Fingers for Delicate Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决脆弱物体抓取中的力控与滑动检测问题。提出FORTE系统，通过3D打印结构实现低延迟力和滑动反馈，提升抓取精度与安全性。**

- **链接: [https://arxiv.org/pdf/2506.18960v3](https://arxiv.org/pdf/2506.18960v3)**

> **作者:** Siqi Shang; Mingyo Seo; Yuke Zhu; Lillian Chin
>
> **摘要:** Handling fragile objects remains a major challenge for robotic manipulation. Tactile sensing and soft robotics can improve delicate object handling, but typically involve high integration complexity or slow response times. We address these issues through FORTE, an easy-to-fabricate tactile sensing system. FORTE uses 3D-printed fin-ray grippers with internal air channels to provide low-latency force and slip feedback. This feedback allows us to apply just enough force to grasp objects without damaging them. We accurately estimate grasping forces from 0-8 N with an average error of 0.2 N, and detect slip events within 100 ms of occurring. FORTE can grasp a wide range of slippery, fragile, and deformable objects, including raspberries and potato chips with 92% success and achieves 93% accuracy in detecting slip events. These results highlight FORTE's potential as a robust solution for delicate robotic manipulation. Project page: https://merge-lab.github.io/FORTE/
>
---
#### [replaced 009] Stairway to Success: An Online Floor-Aware Zero-Shot Object-Goal Navigation Framework via LLM-Driven Coarse-to-Fine Exploration
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决多楼层环境下无地图、零样本目标导航问题。提出ASCENT框架，结合多层抽象与粗到细推理，实现在线楼层感知导航。**

- **链接: [https://arxiv.org/pdf/2505.23019v4](https://arxiv.org/pdf/2505.23019v4)**

> **作者:** Zeying Gong; Rong Li; Tianshuai Hu; Ronghe Qiu; Lingdong Kong; Lingfeng Zhang; Guoyang Zhao; Yiyi Ding; Junwei Liang
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RAL). Project Page at https://zeying-gong.github.io/projects/ascent
>
> **摘要:** Deployable service and delivery robots struggle to navigate multi-floor buildings to reach object goals, as existing systems fail due to single-floor assumptions and requirements for offline, globally consistent maps. Multi-floor environments pose unique challenges including cross-floor transitions and vertical spatial reasoning, especially navigating unknown buildings. Object-Goal Navigation benchmarks like HM3D and MP3D also capture this multi-floor reality, yet current methods lack support for online, floor-aware navigation. To bridge this gap, we propose \textbf{\textit{ASCENT}}, an online framework for Zero-Shot Object-Goal Navigation that enables robots to operate without pre-built maps or retraining on new object categories. It introduces: (1) a \textbf{Multi-Floor Abstraction} module that dynamically constructs hierarchical representations with stair-aware obstacle mapping and cross-floor topology modeling, and (2) a \textbf{Coarse-to-Fine Reasoning} module that combines frontier ranking with LLM-driven contextual analysis for multi-floor navigation decisions. We evaluate on HM3D and MP3D benchmarks, outperforming state-of-the-art zero-shot approaches, and demonstrate real-world deployment on a quadruped robot.
>
---
#### [replaced 010] Volume-Consistent Kneading-Based Deformation Manufacturing for Material-Efficient Shaping
- **分类: cs.RO**

- **简介: 该论文属于制造领域，旨在解决传统制造中的材料浪费和表面质量问题。提出一种体积一致的揉捏成形方法，实现高效、低废的三维变形制造。**

- **链接: [https://arxiv.org/pdf/2511.22042v2](https://arxiv.org/pdf/2511.22042v2)**

> **作者:** Lei Li; Jiale Gong; Ziyang Li; Hong Wang
>
> **备注:** 39 pages, 31 figures
>
> **摘要:** Conventional subtractive manufacturing inevitably involves material loss during geometric realization, while additive manufacturing still suffers from limitations in surface quality, process continuity, and productivity when fabricating complex geometries. To address these challenges, this paper proposes a volume-consistent kneading-based forming method for plastic materials, enabling continuous and controllable three-dimensional deformation under mass conservation. An integrated kneading-based manufacturing system is developed, in which geometry-aware kneading command generation, layer-wise kneading execution, and in-process point-cloud scanning are tightly coupled to form a closed-loop workflow of scanning, forming, and feedback compensation. Target geometries are analyzed through layer-wise point-cloud processing and classified into enveloping and non-enveloping types. Accordingly, an Envelope Shaping First strategy and a Similar Gradient Method are adopted to ensure stable material flow and continuous deformation. An RMSE-based compensation scheme is further introduced to correct systematic geometric deviations induced by elastic rebound and material redistribution. Experimental validation on five representative geometries demonstrates high geometric fidelity, with material utilization consistently exceeding 98%. The results indicate that kneading-based forming provides a promising alternative manufacturing paradigm for low-waste, customizable production.
>
---
#### [replaced 011] RoboMirror: Understand Before You Imitate for Video to Humanoid Locomotion
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RoboMirror，解决视频到人形机器人运动的控制问题，通过视觉理解生成合理动作，无需姿态重建。**

- **链接: [https://arxiv.org/pdf/2512.23649v3](https://arxiv.org/pdf/2512.23649v3)**

> **作者:** Zhe Li; Cheng Chi; Boan Zhu; Yangyang Wei; Shuanghao Bai; Yuheng Ji; Yibo Peng; Tao Huang; Pengwei Wang; Zhongyuan Wang; S. -H. Gary Chan; Chang Xu; Shanghang Zhang
>
> **摘要:** Humans learn locomotion through visual observation, interpreting visual content first before imitating actions. However, state-of-the-art humanoid locomotion systems rely on either curated motion capture trajectories or sparse text commands, leaving a critical gap between visual understanding and control. Text-to-motion methods suffer from semantic sparsity and staged pipeline errors, while video-based approaches only perform mechanical pose mimicry without genuine visual understanding. We propose RoboMirror, the first retargeting-free video-to-locomotion framework embodying "understand before you imitate". Leveraging VLMs, it distills raw egocentric/third-person videos into visual motion intents, which directly condition a diffusion-based policy to generate physically plausible, semantically aligned locomotion without explicit pose reconstruction or retargeting. Extensive experiments validate the effectiveness of RoboMirror, it enables telepresence via egocentric videos, drastically reduces third-person control latency by 80%, and achieves a 3.7% higher task success rate than baselines. By reframing humanoid control around video understanding, we bridge the visual understanding and action gap.
>
---
#### [replaced 012] ManiBox: Enhancing Embodied Spatial Generalization via Scalable Simulation Data Generations
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出ManiBox框架，解决机器人操作中的空间泛化问题。通过模拟数据生成和边界框引导，提升真实场景下的操作能力。**

- **链接: [https://arxiv.org/pdf/2411.01850v3](https://arxiv.org/pdf/2411.01850v3)**

> **作者:** Hengkai Tan; Xuezhou Xu; Chengyang Ying; Xinyi Mao; Zeyuan Wang; Songming Liu; Xingxing Zhang; Zhizhong Su; Hang Su; Jun Zhu
>
> **摘要:** Embodied agents require robust spatial intelligence to execute precise real-world manipulations. However, this remains a significant challenge, as current methods often struggle to accurately position objects in space. Collecting extensive data can help address this issue by enhancing the agent's spatial understanding. Nonetheless, obtaining such data with real robots is prohibitively expensive, and relying on simulation data frequently leads to visual generalization gaps during real-world deployment. To tackle these challenges, we propose ManiBox, a novel bounding-box-guided framework. By decoupling perception from policy generalization, ManiBox effectively reduces the Sim2Real gap, leverages Internet-scale data, and scales our policy data collection in simulation. Specifically, within ManiBox, the RL teacher policy efficiently generates scalable simulation data. The student policy is distilled from this data and takes bounding boxes as input, which is proven sufficient for determining objects' spatial positions, thus enabling zero-shot transfer to real robots. Comprehensive evaluations in both simulated and real-world environments demonstrate that ManiBox exhibits strong spatial generalization and adaptability across various manipulation tasks and settings. Furthermore, our empirical study provides preliminary verification of spatial scaling laws, i.e., the amount of data required for spatial generalization scales with spatial volume following a power-law relationship. At a given spatial volume level, the success rate of manipulation tasks follows Michaelis-Menten kinetics with respect to data volume, exhibiting a saturation effect as data increases. Our videos and code are available at https://thkkk.github.io/manibox
>
---
#### [replaced 013] Phase-based Nonlinear Model Predictive Control for Humanoid Walking Stabilization with Single and Double Support Time Adjustments
- **分类: cs.RO**

- **简介: 该论文属于机器人行走稳定性控制任务，旨在解决 humanoid 在单双支撑阶段的动态平衡问题。通过相位一致的非线性模型预测控制框架，优化 ZMP、步态和支撑时间，提升行走稳定性。**

- **链接: [https://arxiv.org/pdf/2506.03856v2](https://arxiv.org/pdf/2506.03856v2)**

> **作者:** Kwanwoo Lee; Gyeongjae Park; Myeong-Ju Kim; Jaeheung Park
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** The contact sequence of humanoid walking consists of single and double support phases (SSP and DSP), and their coordination through proper duration and dynamic transition based on the robot's state is crucial for maintaining walking stability. Numerous studies have investigated phase duration optimization as an effective means of improving walking stability. This paper presents a phase-based Nonlinear Model Predictive Control (NMPC) framework that jointly optimizes Zero Moment Point (ZMP) modulation, step location, SSP duration (step timing), and DSP duration within a single formulation. Specifically, the proposed framework reformulates the nonlinear DCM (Divergent Component of Motion) error dynamics into a phase-consistent representation and incorporates them as dynamic constraints within the NMPC. The proposed framework also guarantees ZMP input continuity during contact-phase transitions and disables footstep updates during the DSP, thereby enabling dynamically reliable balancing control regardless of whether the robot is in SSP or DSP. The effectiveness of the proposed method is validated through extensive simulation and hardware experiments, demonstrating improved balance performance under external disturbances.
>
---
#### [replaced 014] SPARC: Spine with Prismatic and Revolute Compliance for Quadruped Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出SPARC，一种用于四足机器人的脊柱模块，解决腿部运动中柔性控制问题。通过结合旋转与直线运动，实现可编程阻抗控制，提升行走效率。**

- **链接: [https://arxiv.org/pdf/2510.01984v2](https://arxiv.org/pdf/2510.01984v2)**

> **作者:** Yue Wang
>
> **摘要:** We present SPARC, a compact, open-source 3-DoF sagittal-plane spine module that combines revolute (pitch) and prismatic (axial) motion with programmable task-space impedance for quadruped robots. The system integrates three torque-controlled actuators, a 1~kHz control board, and protected power electronics in a 1.26~kg package. A floating-base impedance controller with dynamics compensation renders tunable spring-damper behavior along horizontal, vertical, and rotational axes. Benchtop experiments validate the approach: quasi-static tests demonstrate linear force-displacement characteristics with commanded horizontal stiffness spanning 300--700~N/m ($\leq 1.5\%$ error, $R^2 \geq 0.992$), while dynamic trials confirm predictable damping behavior across multiple settings. Furthermore, simulation studies reveal that adapting spine stiffness based on terrain slope and stride length significantly reduces the cost of transport. SPARC provides an open-source platform for systematic studies of spine compliance in legged locomotion, with complete hardware and firmware resources available for community use. Github repository: https://github.com/YueWang996/sparc.git
>
---
#### [replaced 015] RoboBPP: Benchmarking Robotic Online Bin Packing with Physics-based Simulation
- **分类: cs.RO**

- **简介: 该论文属于机器人在线装箱任务，旨在解决工业物流中物理可行性评估的问题。通过构建物理仿真基准系统RoboBPP，整合真实数据与评估指标，提升算法实用性。**

- **链接: [https://arxiv.org/pdf/2512.04415v3](https://arxiv.org/pdf/2512.04415v3)**

> **作者:** Zhoufeng Wang; Hang Zhao; Juzhan Xu; Shishun Zhang; Zeyu Xiong; Ruizhen Hu; Chenyang Zhu; Zecui Zeng; Kai Xu
>
> **备注:** Under review at the International Journal of Robotics Research (IJRR)
>
> **摘要:** Physical feasibility in 3D bin packing is a key requirement in modern industrial logistics and robotic automation. With the growing adoption of industrial automation, online bin packing has gained increasing attention. However, inconsistencies in problem settings, test datasets, and evaluation metrics have hindered progress in the field, and there is a lack of a comprehensive benchmarking system. Direct testing on real hardware is costly, and building a realistic simulation environment is also challenging. To address these limitations, we introduce RoboBPP, a benchmarking system designed for robotic online bin packing. RoboBPP integrates a physics-based simulator to assess physical feasibility. In our simulation environment, we introduce a robotic arm and boxes at real-world scales to replicate real industrial packing workflows. By simulating conditions that arise in real industrial applications, we ensure that evaluated algorithms are practically deployable. In addition, prior studies often rely on synthetic datasets whose distributions differ from real-world industrial data. To address this issue, we collect three datasets from real industrial workflows, including assembly-line production, logistics packing, and furniture manufacturing. The benchmark comprises three carefully designed test settings and extends existing evaluation metrics with new metrics for structural stability and operational safety. We design a scoring system and derive a range of insights from the evaluation results. RoboBPP is fully open-source and is equipped with visualization tools and an online leaderboard, providing a reproducible and extensible foundation for future research and industrial applications (https://robot-bin-packing-benchmark.github.io).
>
---
#### [replaced 016] Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于开放词汇移动操作任务，解决机器人基座放置问题。通过融合语义与几何信息，提出一种零样本框架，提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2511.06240v2](https://arxiv.org/pdf/2511.06240v2)**

> **作者:** Tzu-Jung Lin; Jia-Fong Yeh; Hung-Ting Su; Chung-Yi Lin; Yi-Ting Chen; Winston H. Hsu
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** In open-vocabulary mobile manipulation (OVMM), task success often hinges on the selection of an appropriate base placement for the robot. Existing approaches typically navigate to proximity-based regions without considering affordances, resulting in frequent manipulation failures. We propose Affordance-Guided Coarse-to-Fine Exploration, a zero-shot framework for base placement that integrates semantic understanding from vision-language models (VLMs) with geometric feasibility through an iterative optimization process. Our method constructs cross-modal representations, namely Affordance RGB and Obstacle Map+, to align semantics with spatial context. This enables reasoning that extends beyond the egocentric limitations of RGB perception. To ensure interaction is guided by task-relevant affordances, we leverage coarse semantic priors from VLMs to guide the search toward task-relevant regions and refine placements with geometric constraints, thereby reducing the risk of convergence to local optima. Evaluated on five diverse open-vocabulary mobile manipulation tasks, our system achieves an 85% success rate, significantly outperforming classical geometric planners and VLM-based methods. This demonstrates the promise of affordance-aware and multimodal reasoning for generalizable, instruction-conditioned planning in OVMM.
>
---
#### [replaced 017] Compositional Diffusion with Guided Search for Long-Horizon Planning
- **分类: cs.RO**

- **简介: 该论文属于长期规划任务，解决组合生成模型中模式平均导致的不一致问题。提出CDGS方法，在扩散过程中嵌入搜索，提升全局一致性与局部可行性。**

- **链接: [https://arxiv.org/pdf/2601.00126v2](https://arxiv.org/pdf/2601.00126v2)**

> **作者:** Utkarsh A Mishra; David He; Yongxin Chen; Danfei Xu
>
> **备注:** 38 pages, 18 figures
>
> **摘要:** Generative models have emerged as powerful tools for planning, with compositional approaches offering particular promise for modeling long-horizon task distributions by composing together local, modular generative models. This compositional paradigm spans diverse domains, from multi-step manipulation planning to panoramic image synthesis to long video generation. However, compositional generative models face a critical challenge: when local distributions are multimodal, existing composition methods average incompatible modes, producing plans that are neither locally feasible nor globally coherent. We propose Compositional Diffusion with Guided Search (CDGS), which addresses this mode averaging problem by embedding search directly within the diffusion denoising process. Our method explores diverse combinations of local modes through population-based sampling, prunes infeasible candidates using likelihood-based filtering, and enforces global consistency through iterative resampling between overlapping segments. CDGS matches oracle performance on seven robot manipulation tasks, outperforming baselines that lack compositionality or require long-horizon training data. The approach generalizes across domains, enabling coherent text-guided panoramic images and long videos through effective local-to-global message passing. More details: https://cdgsearch.github.io/
>
---
#### [replaced 018] SurgWorld: Learning Surgical Robot Policies from Videos via World Modeling
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于手术机器人领域，解决数据稀缺问题。通过构建SurgWorld模型和SATA数据集，生成合成视频动作数据，提升手术机器人自主学习效果。**

- **链接: [https://arxiv.org/pdf/2512.23162v3](https://arxiv.org/pdf/2512.23162v3)**

> **作者:** Yufan He; Pengfei Guo; Mengya Xu; Zhaoshuo Li; Andriy Myronenko; Dillan Imans; Bingjie Liu; Dongren Yang; Mingxue Gu; Yongnan Ji; Yueming Jin; Ren Zhao; Baiyong Shen; Daguang Xu
>
> **摘要:** Data scarcity remains a fundamental barrier to achieving fully autonomous surgical robots. While large scale vision language action (VLA) models have shown impressive generalization in household and industrial manipulation by leveraging paired video action data from diverse domains, surgical robotics suffers from the paucity of datasets that include both visual observations and accurate robot kinematics. In contrast, vast corpora of surgical videos exist, but they lack corresponding action labels, preventing direct application of imitation learning or VLA training. In this work, we aim to alleviate this problem by learning policy models from SurgWorld, a world model designed for surgical physical AI. We curated the Surgical Action Text Alignment (SATA) dataset with detailed action description specifically for surgical robots. Then we built SurgeWorld based on the most advanced physical AI world model and SATA. It's able to generate diverse, generalizable and realistic surgery videos. We are also the first to use an inverse dynamics model to infer pseudokinematics from synthetic surgical videos, producing synthetic paired video action data. We demonstrate that a surgical VLA policy trained with these augmented data significantly outperforms models trained only on real demonstrations on a real surgical robot platform. Our approach offers a scalable path toward autonomous surgical skill acquisition by leveraging the abundance of unlabeled surgical video and generative world modeling, thus opening the door to generalizable and data efficient surgical robot policies.
>
---
#### [replaced 019] VL-LN Bench: Towards Long-horizon Goal-oriented Navigation with Active Dialogs
- **分类: cs.RO**

- **简介: 该论文提出IIGN任务，解决真实场景下导航指令模糊的问题，通过主动对话增强导航模型，构建VL-LN基准进行评估。**

- **链接: [https://arxiv.org/pdf/2512.22342v3](https://arxiv.org/pdf/2512.22342v3)**

> **作者:** Wensi Huang; Shaohao Zhu; Meng Wei; Jinming Xu; Xihui Liu; Hanqing Wang; Tai Wang; Feng Zhao; Jiangmiao Pang
>
> **摘要:** In most existing embodied navigation tasks, instructions are well-defined and unambiguous, such as instruction following and object searching. Under this idealized setting, agents are required solely to produce effective navigation outputs conditioned on vision and language inputs. However, real-world navigation instructions are often vague and ambiguous, requiring the agent to resolve uncertainty and infer user intent through active dialog. To address this gap, we propose Interactive Instance Goal Navigation (IIGN), a task that requires agents not only to generate navigation actions but also to produce language outputs via active dialog, thereby aligning more closely with practical settings. IIGN extends Instance Goal Navigation (IGN) by allowing agents to freely consult an oracle in natural language while navigating. Building on this task, we present the Vision Language-Language Navigation (VL-LN) benchmark, which provides a large-scale, automatically generated dataset and a comprehensive evaluation protocol for training and assessing dialog-enabled navigation models. VL-LN comprises over 41k long-horizon dialog-augmented trajectories for training and an automatic evaluation protocol with an oracle capable of responding to agent queries. Using this benchmark, we train a navigation model equipped with dialog capabilities and show that it achieves significant improvements over the baselines. Extensive experiments and analyses further demonstrate the effectiveness and reliability of VL-LN for advancing research on dialog-enabled embodied navigation. Code and dataset: https://0309hws.github.io/VL-LN.github.io/
>
---
#### [replaced 020] LIMOncello: Iterated Error-State Kalman Filter on the SGal(3) Manifold for Fast LiDAR-Inertial Odometry
- **分类: cs.RO**

- **简介: 该论文提出LIMOncello，用于LiDAR-Inertial Odometry任务，解决低可观测性下的位姿估计问题，采用SGal(3)流形和迭代误差状态卡尔曼滤波提升精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.19567v2](https://arxiv.org/pdf/2512.19567v2)**

> **作者:** Carlos Pérez-Ruiz; Joan Solà
>
> **摘要:** This work introduces LIMOncello, a tightly coupled LiDAR-Inertial Odometry system that models 6-DoF motion on the $\mathrm{SGal}(3)$ manifold within an iterated error-state Kalman filter backend. Compared to state representations defined on $\mathrm{SO}(3)\times\mathbb{R}^6$, the use of $\mathrm{SGal}(3)$ provides a coherent and numerically stable discrete-time propagation model that helps limit drift in low-observability conditions. LIMOncello also includes a lightweight incremental i-Octree mapping backend that enables faster updates and substantially lower memory usage than incremental kd-tree style map structures, without relying on locality-restricted search heuristics. Experiments on multiple real-world datasets show that LIMOncello achieves competitive accuracy while improving robustness in geometrically sparse environments. The system maintains real-time performance with stable memory growth and is released as an extensible open-source implementation at https://github.com/CPerezRuiz335/LIMOncello.
>
---
#### [replaced 021] RNBF: Real-Time RGB-D Based Neural Barrier Functions for Safe Robotic Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人安全导航任务，解决未知环境中实时避障问题。通过在线构建神经SDF，实现无预训练的视觉导航，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2505.02294v3](https://arxiv.org/pdf/2505.02294v3)**

> **作者:** Satyajeet Das; Yifan Xue; Haoming Li; Nadia Figueroa
>
> **摘要:** Autonomous safe navigation in unstructured and novel environments poses significant challenges, especially when environment information can only be provided through low-cost vision sensors. Although safe reactive approaches have been proposed to ensure robot safety in complex environments, many base their theory off the assumption that the robot has prior knowledge on obstacle locations and geometries. In this paper, we present a real-time, vision-based framework that constructs continuous, first-order differentiable Signed Distance Fields (SDFs) of unknown environments entirely online, without any pre-training, and is fully compatible with established SDF-based reactive controllers. To achieve robust performance under practical sensing conditions, our approach explicitly accounts for noise in affordable RGB-D cameras, refining the neural SDF representation online for smoother geometry and stable gradient estimates. We validate the proposed method in simulation and real-world experiments using a Fetch robot. Videos and supplementary material are available at https://satyajeetburla.github.io/rnbf/.
>
---
#### [replaced 022] MOON: Multi-Objective Optimization-Driven Object-Goal Navigation Using a Variable-Horizon Set-Orienteering Planner
- **分类: cs.RO**

- **简介: 该论文属于目标导航任务，解决复杂环境中多目标优化问题。提出MOON框架，结合地标编码、导航增强和变时域规划，实现高效全局路径规划。**

- **链接: [https://arxiv.org/pdf/2505.12752v3](https://arxiv.org/pdf/2505.12752v3)**

> **作者:** Daigo Nakajima; Kanji Tanaka; Daiki Iwata; Kouki Terashima
>
> **备注:** 9 pages, 7 figures, technical report
>
> **摘要:** This paper proposes MOON (Multi-Objective Optimization-driven Object-goal Navigation), a novel framework designed for efficient navigation in large-scale, complex indoor environments. While existing methods often rely on local heuristics, they frequently fail to address the strategic trade-offs between competing objectives in vast areas. To overcome this, we formulate the task as a multi-objective optimization problem (MOO) that balances frontier-based exploration with the exploitation of observed landmarks. Our prototype integrates three key pillars: (1) QOM [IROS05] for discriminative landmark encoding; (2) StructNav [RSS23] to enhance the navigation pipeline; and (3) a variable-horizon Set Orienteering Problem (SOP) formulation for globally coherent planning. To further support the framework's scalability, we provide a detailed theoretical foundation for the budget-constrained SOP formulation and the data-driven mode-switching strategy that enables long-horizon resource allocation. Additionally, we introduce a high-speed neural planner that distills the expert solver into a transformer-based model, reducing decision latency by a factor of nearly 10 while maintaining high planning quality.
>
---
#### [replaced 023] RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言导航任务，旨在解决机器人在复杂3D场景中准确理解空间指代和动态推理的问题。提出RoboRefer模型，结合深度编码与强化学习，提升空间理解与多步推理能力。**

- **链接: [https://arxiv.org/pdf/2506.04308v4](https://arxiv.org/pdf/2506.04308v4)**

> **作者:** Enshen Zhou; Jingkun An; Cheng Chi; Yi Han; Shanyu Rong; Chi Zhang; Pengwei Wang; Zhongyuan Wang; Tiejun Huang; Lu Sheng; Shanghang Zhang
>
> **备注:** Accepted by NeurIPS 2025. Project page: https://zhoues.github.io/RoboRefer/
>
> **摘要:** Spatial referring is a fundamental capability of embodied robots to interact with the 3D physical world. However, even with the powerful pretrained vision language models (VLMs), recent approaches are still not qualified to accurately understand the complex 3D scenes and dynamically reason about the instruction-indicated locations for interaction. To this end, we propose RoboRefer, a 3D-aware VLM that can first achieve precise spatial understanding by integrating a disentangled but dedicated depth encoder via supervised fine-tuning (SFT). Moreover, RoboRefer advances generalized multi-step spatial reasoning via reinforcement fine-tuning (RFT), with metric-sensitive process reward functions tailored for spatial referring tasks. To support SFT and RFT training, we introduce RefSpatial, a large-scale dataset of 20M QA pairs (2x prior), covering 31 spatial relations (vs. 15 prior) and supporting complex reasoning processes (up to 5 steps). In addition, we introduce RefSpatial-Bench, a challenging benchmark filling the gap in evaluating spatial referring with multi-step reasoning. Experiments show that SFT-trained RoboRefer achieves state-of-the-art spatial understanding, with an average success rate of 89.6%. RFT-trained RoboRefer further outperforms all other baselines by a large margin, even surpassing Gemini-2.5-Pro by 17.4% in average accuracy on RefSpatial-Bench. Notably, RoboRefer can be integrated with various control policies to execute long-horizon, dynamic tasks across diverse robots (e,g., UR5, G1 humanoid) in cluttered real-world scenes.
>
---
#### [replaced 024] Foundation models on the bridge: Semantic hazard detection and safety maneuvers for maritime autonomy with vision-language models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于海上自主任务，解决船舶在异常情况下的安全应对问题。通过视觉语言模型实现语义感知，设计快速 fallback 动作选择器，确保符合IMO规范。**

- **链接: [https://arxiv.org/pdf/2512.24470v2](https://arxiv.org/pdf/2512.24470v2)**

> **作者:** Kim Alexander Christensen; Andreas Gudahl Tufte; Alexey Gusev; Rohan Sinha; Milan Ganai; Ole Andreas Alsos; Marco Pavone; Martin Steinert
>
> **备注:** 17 pages without bibliography or appendix. The main paper has 16 figures. Paper webpage can be found at https://kimachristensen.github.io/bridge_policy/
>
> **摘要:** The draft IMO MASS Code requires autonomous and remotely supervised maritime vessels to detect departures from their operational design domain, enter a predefined fallback that notifies the operator, permit immediate human override, and avoid changing the voyage plan without approval. Meeting these obligations in the alert-to-takeover gap calls for a short-horizon, human-overridable fallback maneuver. Classical maritime autonomy stacks struggle when the correct action depends on meaning (e.g., diver-down flag means people in the water, fire close by means hazard). We argue (i) that vision-language models (VLMs) provide semantic awareness for such out-of-distribution situations, and (ii) that a fast-slow anomaly pipeline with a short-horizon, human-overridable fallback maneuver makes this practical in the handover window. We introduce Semantic Lookout, a camera-only, candidate-constrained VLM fallback maneuver selector that selects one cautious action (or station-keeping) from water-valid, world-anchored trajectories under continuous human authority. On 40 harbor scenes we measure per-call scene understanding and latency, alignment with human consensus (model majority-of-three voting), short-horizon risk-relief on fire hazard scenes, and an on-water alert->fallback maneuver->operator handover. Sub-10 s models retain most of the awareness of slower state-of-the-art models. The fallback maneuver selector outperforms geometry-only baselines and increases standoff distance on fire scenes. A field run verifies end-to-end operation. These results support VLMs as semantic fallback maneuver selectors compatible with the draft IMO MASS Code, within practical latency budgets, and motivate future work on domain-adapted, hybrid autonomy that pairs foundation-model semantics with multi-sensor bird's-eye-view perception and short-horizon replanning. Website: kimachristensen.github.io/bridge_policy
>
---
#### [replaced 025] H2R: A Human-to-Robot Data Augmentation for Robot Pre-training from Videos
- **分类: cs.RO**

- **简介: 该论文提出H2R方法，解决机器人学习中人类与机器人视觉差异问题。通过数据增强，将人类视频转换为机器人视角数据，提升机器人泛化能力。**

- **链接: [https://arxiv.org/pdf/2505.11920v3](https://arxiv.org/pdf/2505.11920v3)**

> **作者:** Guangrun Li; Yaoxu Lyu; Zhuoyang Liu; Chengkai Hou; Jieyu Zhang; Shanghang Zhang
>
> **摘要:** Large-scale pre-training using videos has proven effective for robot learning. However, the models pre-trained on such data can be suboptimal for robot learning due to the significant visual gap between human hands and those of different robots. To remedy this, we propose H2R, a simple data augmentation technique that detects human hand keypoints, synthesizes robot motions in simulation, and composites rendered robots into egocentric videos. This process explicitly bridges the visual gap between human and robot embodiments during pre-training. We apply H2R to augment large-scale egocentric human video datasets such as Ego4D and SSv2, replacing human hands with simulated robotic arms to generate robot-centric training data. Based on this, we construct and release a family of 1M-scale datasets covering multiple robot embodiments (UR5 with gripper/Leaphand, Franka) and data sources (SSv2, Ego4D). To verify the effectiveness of the augmentation pipeline, we introduce a CLIP-based image-text similarity metric that quantitatively evaluates the semantic fidelity of robot-rendered frames to the original human actions. We validate H2R across three simulation benchmarks: Robomimic, RLBench and PushT and real-world manipulation tasks with a UR5 robot equipped with Gripper and Leaphand end-effectors. H2R consistently improves downstream success rates, yielding gains of 5.0%-10.2% in simulation and 6.7%-23.3% in real-world tasks across various visual encoders and policy learning methods. These results indicate that H2R improves the generalization ability of robotic policies by mitigating the visual discrepancies between human and robot domains.
>
---
#### [replaced 026] Multi-Robot Data-Free Continual Communicative Learning (CCL) from Black-Box Visual Place Recognition Models
- **分类: cs.RO**

- **简介: 该论文属于多机器人视觉定位任务，解决未知机器人间知识共享问题。通过通信框架CCL，无需数据直接提升机器人定位能力。**

- **链接: [https://arxiv.org/pdf/2503.02256v2](https://arxiv.org/pdf/2503.02256v2)**

> **作者:** Kenta Tsukahara; Kanji Tanaka; Daiki Iwata; Jonathan Tay Yu Liang
>
> **备注:** 6 pages, 4 figures, technical report
>
> **摘要:** In emerging multi-robot societies, heterogeneous agents must continually extract and integrate local knowledge from one another through communication, even when their internal models are completely opaque. Existing approaches to continual or collaborative learning for visual place recognition (VPR) largely assume white-box access to model parameters or shared training datasets, which is unrealistic when robots encounter unknown peers in the wild. This paper introduces \emph{Continual Communicative Learning (CCL)}, a data-free multi-robot framework in which a traveler robot (student) continually improves its VPR capability by communicating with black-box teacher models via a constrained query--response channel. We repurpose Membership Inference Attacks (MIA), originally developed as privacy attacks on machine learning models, as a constructive communication primitive to reconstruct pseudo-training sets from black-box VPR teachers without accessing their parameters or raw data. To overcome the intrinsic communication bottleneck caused by the low sampling efficiency of black-box MIA, we propose a prior-based query strategy that leverages the student's own VPR prior to focus queries on informative regions of the embedding space, thereby reducing the knowledge transfer (KT) cost. Experimental results on a standard multi-session VPR benchmark demonstrate that the proposed CCL framework yields substantial performance gains for low-performing robots under modest communication budgets, highlighting CCL as a promising building block for scalable and fault-tolerant multi-robot systems.
>
---
#### [replaced 027] Aerial World Model for Long-horizon Visual Generation and Navigation in 3D Space
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于无人机导航任务，旨在解决长距离视觉生成与导航中语义信息不足的问题。提出ANWM模型，通过预测未来视觉并结合语义评估轨迹，提升导航成功率。**

- **链接: [https://arxiv.org/pdf/2512.21887v2](https://arxiv.org/pdf/2512.21887v2)**

> **作者:** Weichen Zhang; Peizhi Tang; Xin Zeng; Fanhang Man; Shiquan Yu; Zichao Dai; Baining Zhao; Hongjin Chen; Yu Shang; Wei Wu; Chen Gao; Xinlei Chen; Xin Wang; Yong Li; Wenwu Zhu
>
> **摘要:** Unmanned aerial vehicles (UAVs) have emerged as powerful embodied agents. One of the core abilities is autonomous navigation in large-scale three-dimensional environments. Existing navigation policies, however, are typically optimized for low-level objectives such as obstacle avoidance and trajectory smoothness, lacking the ability to incorporate high-level semantics into planning. To bridge this gap, we propose ANWM, an aerial navigation world model that predicts future visual observations conditioned on past frames and actions, thereby enabling agents to rank candidate trajectories by their semantic plausibility and navigational utility. ANWM is trained on 4-DoF UAV trajectories and introduces a physics-inspired module: Future Frame Projection (FFP), which projects past frames into future viewpoints to provide coarse geometric priors. This module mitigates representational uncertainty in long-distance visual generation and captures the mapping between 3D trajectories and egocentric observations. Empirical results demonstrate that ANWM significantly outperforms existing world models in long-distance visual forecasting and improves UAV navigation success rates in large-scale environments.
>
---
#### [replaced 028] AdaVLN: Towards Visual Language Navigation in Continuous Indoor Environments with Moving Humans
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决动态室内环境中人类障碍物带来的导航挑战。提出AdaVLN框架及相应数据集，增强导航模型的适应性。**

- **链接: [https://arxiv.org/pdf/2411.18539v2](https://arxiv.org/pdf/2411.18539v2)**

> **作者:** Dillon Loh; Tomasz Bednarz; Xinxing Xia; Frank Guan
>
> **摘要:** Visual Language Navigation is a task that challenges robots to navigate in realistic environments based on natural language instructions. While previous research has largely focused on static settings, real-world navigation must often contend with dynamic human obstacles. Hence, we propose an extension to the task, termed Adaptive Visual Language Navigation (AdaVLN), which seeks to narrow this gap. AdaVLN requires robots to navigate complex 3D indoor environments populated with dynamically moving human obstacles, adding a layer of complexity to navigation tasks that mimic the real-world. To support exploration of this task, we also present AdaVLN simulator and AdaR2R datasets. The AdaVLN simulator enables easy inclusion of fully animated human models directly into common datasets like Matterport3D. We also introduce a "freeze-time" mechanism for both the navigation task and simulator, which pauses world state updates during agent inference, enabling fair comparisons and experimental reproducibility across different hardware. We evaluate several baseline models on this task, analyze the unique challenges introduced by AdaVLN, and demonstrate its potential to bridge the sim-to-real gap in VLN research.
>
---
#### [replaced 029] LSRE: Latent Semantic Rule Encoding for Real-Time Semantic Risk Detection in Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶中的语义安全检测任务，解决实时语义风险识别问题。提出LSRE框架，将语言规则编码到潜在空间，实现高效风险评估。**

- **链接: [https://arxiv.org/pdf/2512.24712v2](https://arxiv.org/pdf/2512.24712v2)**

> **作者:** Qian Cheng; Weitao Zhou; Cheng Jing; Nanshan Deng; Junze Wen; Zhaoyang Liu; Kun Jiang; Diange Yang
>
> **摘要:** Real-world autonomous driving must adhere to complex human social rules that extend beyond legally codified traffic regulations. Many of these semantic constraints, such as yielding to emergency vehicles, complying with traffic officers' gestures, or stopping for school buses, are intuitive for humans yet difficult to encode explicitly. Although large vision-language models (VLMs) can interpret such semantics, their inference cost makes them impractical for real-time deployment. This work proposes LSRE, a Latent Semantic Rule Encoding framework that converts sparsely sampled VLM judgments into decision boundaries within the latent space of a recurrent world model. By encoding language-defined safety semantics into a lightweight latent classifier, LSRE enables real-time semantic risk assessment at 10 Hz without per-frame VLM queries. Experiments on six semantic-failure scenarios in CARLA demonstrate that LSRE attains semantic risk detection accuracy comparable to a large VLM baseline, while providing substantially earlier hazard anticipation and maintaining low computational latency. LSRE further generalizes to rarely seen semantic-similar test cases, indicating that language-guided latent classification offers an effective and deployable mechanism for semantic safety monitoring in autonomous driving.
>
---
#### [replaced 030] P2U-SLAM: A Monocular Wide-FoV SLAM System Based on Point Uncertainty and Pose Uncertainty
- **分类: cs.RO; cs.CV; eess.IV**

- **简介: 该论文提出P2U-SLAM，解决宽视场角视觉SLAM中的长期定位性能问题。通过引入点不确定性与位姿不确定性，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2409.10143v2](https://arxiv.org/pdf/2409.10143v2)**

> **作者:** Yufan Zhang; Kailun Yang; Ze Wang; Kaiwei Wang
>
> **备注:** Accepted to IEEE Transactions on Intelligent Transportation Systems (T-ITS). The source code will be made publicly available at https://github.com/BambValley/P2U-SLAM
>
> **摘要:** This paper presents P2U-SLAM, a visual Simultaneous Localization And Mapping (SLAM) system with a wide Field of View (FoV) camera, which utilizes pose uncertainty and point uncertainty. While the wide FoV enables considerable repetitive observations of historical map points for matching cross-view features, the data properties of the historical map points and the poses of historical keyframes have changed during the optimization process. The neglect of data property changes results in the lack of partial information matrices in optimization, increasing the risk of long-term positioning performance degradation. The purpose of our research is to mitigate the risks posed by wide-FoV visual input to the SLAM system. Based on the conditional probability model, this work reveals the definite impacts of the above data properties changes on the optimization process, concretizes these impacts as point uncertainty and pose uncertainty, and gives their specific mathematical form. P2U-SLAM embeds point uncertainty into the tracking module and pose uncertainty into the local mapping module respectively, and updates these uncertainties after each optimization operation including local mapping, map merging, and loop closing. We present an exhaustive evaluation on 27 sequences from two popular public datasets with wide-FoV visual input. P2U-SLAM shows excellent performance compared with other state-of-the-art methods. The source code will be made publicly available at https://github.com/BambValley/P2U-SLAM.
>
---
#### [replaced 031] How Robot Dogs See the Unseeable: Improving Visual Interpretability via Peering for Exploratory Robots
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉感知任务，旨在解决机器人在植被环境中因遮挡导致的视觉理解问题。通过模仿昆虫的“窥视”动作，结合信号处理与大模型，提升机器人在部分遮挡下的感知能力。**

- **链接: [https://arxiv.org/pdf/2511.16262v4](https://arxiv.org/pdf/2511.16262v4)**

> **作者:** Oliver Bimber; Karl Dietrich von Ellenrieder; Michael Haller; Rakesh John Amala Arokia Nathan; Gianni Lunardi; Mohamed Youssef; Marco Camurri; Santos Miguel Orozco Soto; Jeremy E. Niven
>
> **摘要:** In vegetated environments, such as forests, exploratory robots play a vital role in navigating complex, cluttered environments where human access is limited and traditional equipment struggles. Visual occlusion from obstacles, such as foliage, can severely obstruct a robot's sensors, impairing scene understanding. We show that "peering", a characteristic side-to-side movement used by insects to overcome their visual limitations, can also allow robots to markedly improve visual reasoning under partial occlusion. This is accomplished by applying core signal processing principles, specifically optical synthetic aperture sensing, together with the vision reasoning capabilities of modern large multimodal models. Peering enables real-time, high-resolution, and wavelength-independent perception, which is crucial for vision-based scene understanding across a wide range of applications. The approach is low-cost and immediately deployable on any camera-equipped robot. We investigated different peering motions and occlusion masking strategies, demonstrating that, unlike peering, state-of-the-art multi-view 3D vision techniques fail in these conditions due to their high susceptibility to occlusion. Our experiments were carried out on an industrial-grade quadrupedal robot. However, the ability to peer is not limited to such platforms, but potentially also applicable to bipedal, hexapod, wheeled, or crawling platforms. Robots that can effectively see through partial occlusion will gain superior perception abilities - including enhanced scene understanding, situational awareness, camouflage breaking, and advanced navigation in complex environments.
>
---
#### [replaced 032] Evaluation of Impression Difference of a Domestic Mobile Manipulator with Autonomous and/or Remote Control in Fetch-and-Carry Tasks
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究家庭服务机器人在取送任务中，不同控制模式对用户感知的影响。旨在解决人机交互中的代理权分配问题，通过实验评估远程控制、自主控制及混合模式的效果。**

- **链接: [https://arxiv.org/pdf/2512.24029v2](https://arxiv.org/pdf/2512.24029v2)**

> **作者:** Takashi Yamamoto; Hiroaki Yaguchi; Shohei Kato; Hiroyuki Okada
>
> **备注:** Published in Advanced Robotics (2020). v2 updates Abstract/Comments (metadata only); paper content unchanged. Please cite: Advanced Robotics 34(20):1291-1308, 2020. https://doi.org/10.1080/01691864.2020.1780152
>
> **摘要:** A single service robot can present two distinct agencies: its onboard autonomy and an operator-mediated agency, yet users experience them through one physical body. We formalize this dual-agency structure as a User-Robot-Operator triad in an autonomous remote-control setting that integrates teleoperation with autonomous execution and human-in-the-loop remote assistance. Prior to the recent surge of language-based and multimodal interfaces, we developed and evaluated an early-stage prototype in 2020 that combined natural-language text chat with a sketch-based interface enabling freehand on-image annotation over the robot's live camera view to support remote intervention. We evaluated three modes - remote control via teleoperation, autonomous control, and autonomous remote control (a hybrid mode representing different levels of autonomy) - in controlled fetch-and-carry mobile manipulation tasks using a domestic mobile manipulator, the Human Support Robot (HSR), on a World Robot Summit 2020 rule-compliant test field. The results show systematic mode-dependent differences in user-rated affinity and perceived security, indicating that switching or blending agency within one robot measurably shapes human impressions in Human-Robot Interaction (HRI). These findings provide empirical guidance for designing human-in-the-loop mobile manipulation in domestic physical tasks.
>
---
#### [replaced 033] Standing Tall: Sim to Real Fall Classification and Lead Time Prediction for Bipedal Robots
- **分类: cs.RO**

- **简介: 该论文属于双足机器人防跌落任务，解决实时跌倒分类与预测问题。在Digit机器人上实现并优化算法，提升准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2506.01141v3](https://arxiv.org/pdf/2506.01141v3)**

> **作者:** Gokul Prabhakaran; Jessy W. Grizzle; M. Eva Mungai
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper extends a previously proposed fall prediction algorithm to a real-time (online) setting, with implementations in both hardware and simulation. The system is validated on the full-sized bipedal robot Digit, where the real-time version achieves performance comparable to the offline implementation while maintaining a zero false positive rate, an average lead time (defined as the difference between the true and predicted fall time) of 1.1s (well above the required minimum of 0.2s), and a maximum lead time error of just 0.03s. It also achieves a high recovery rate of 0.97, demonstrating its effectiveness in real-world deployment. In addition to the real-time implementation, this work identifies key limitations of the original algorithm, particularly under omnidirectional faults, and introduces a fine-tuned strategy to improve robustness. The enhanced algorithm shows measurable improvements across all evaluated metrics, including a 0.05 reduction in average false positive rate and a 1.19s decrease in the maximum error of the average predicted lead time.
>
---
#### [replaced 034] Nonlinear Oscillatory Response of Automated Vehicle Car-following: Theoretical Analysis with Traffic State and Control Input Limits
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于自动驾驶车辆跟车控制任务，解决非线性振荡响应分析问题。通过描述函数方法，考虑交通状态和控制输入限制，建立理论分析框架，提升稳定性分析精度。**

- **链接: [https://arxiv.org/pdf/2505.24029v3](https://arxiv.org/pdf/2505.24029v3)**

> **作者:** Sixu Li; Yang Zhou
>
> **摘要:** This paper presents a framework grounded in the theory of describing function (DF) and incremental-input DF to theoretically analyze the nonlinear oscillatory response of automated vehicles (AVs) car-following (CF) amidst traffic oscillations, considering the limits of traffic state and control input. While prevailing approaches largely ignore these limits (i.e., saturation of acceleration/deceleration and speed) and focus on linear string stability analysis, this framework establishes a basis for theoretically analyzing the frequency response of AV systems with nonlinearities imposed by these limits. To this end, trajectories of CF pairs are decomposed into nominal and oscillatory trajectories, subsequently, the controlled AV system is repositioned within the oscillatory trajectory coordinates. Built on this base, DFs are employed to approximate the frequency responses of nonlinear saturation components by using their first harmonic output, thereby capturing the associated amplification ratio and phase shift. Considering the closed-loop nature of AV control systems, where system states and control input mutually influence each other, amplification ratios and phase shifts are balanced within the loop to ensure consistency. This balancing process may render multiple solutions, hence the incremental-input DF is further applied to identify the reasonable ones. The proposed method is validated by estimations from Simulink, and further comparisons with prevailing methods are conducted. Results confirm the alignment of our framework with Simulink results and exhibit its superior accuracy in analysis compared to the prevailing methods. Furthermore, the framework proves valuable in string stability analysis, especially when conventional linear methods offer misleading insights.
>
---
#### [replaced 035] Do You Have Freestyle? Expressive Humanoid Locomotion via Audio Control
- **分类: cs.RO**

- **简介: 该论文属于人形机器人运动生成任务，解决机器人缺乏音乐和语音驱动的即兴表现能力问题。提出RoboPerform框架，直接从音频生成运动，无需显式重建，提升响应速度与准确性。**

- **链接: [https://arxiv.org/pdf/2512.23650v2](https://arxiv.org/pdf/2512.23650v2)**

> **作者:** Zhe Li; Cheng Chi; Yangyang Wei; Boan Zhu; Tao Huang; Zhenguo Sun; Yibo Peng; Pengwei Wang; Zhongyuan Wang; Fangzhou Liu; Chang Xu; Shanghang Zhang
>
> **摘要:** Humans intuitively move to sound, but current humanoid robots lack expressive improvisational capabilities, confined to predefined motions or sparse commands. Generating motion from audio and then retargeting it to robots relies on explicit motion reconstruction, leading to cascaded errors, high latency, and disjointed acoustic-actuation mapping. We propose RoboPerform, the first unified audio-to-locomotion framework that can directly generate music-driven dance and speech-driven co-speech gestures from audio. Guided by the core principle of "motion = content + style", the framework treats audio as implicit style signals and eliminates the need for explicit motion reconstruction. RoboPerform integrates a ResMoE teacher policy for adapting to diverse motion patterns and a diffusion-based student policy for audio style injection. This retargeting-free design ensures low latency and high fidelity. Experimental validation shows that RoboPerform achieves promising results in physical plausibility and audio alignment, successfully transforming robots into responsive performers capable of reacting to audio.
>
---
