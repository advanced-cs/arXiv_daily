# 机器人 cs.RO

- **最新发布 21 篇**

- **更新 16 篇**

## 最新发布

#### [new 001] Load-Based Variable Transmission Mechanism for Robotic Applications
- **分类: cs.RO**

- **简介: 该论文提出一种基于负载的被动变传动比机制（LBVT），旨在解决机器人关节需动态适应外力但传统变传动系统复杂、需额外驱动的问题。通过预紧弹簧与四杆机构实现扭矩阈值触发的自动传动比调节，仿真验证其在18 N以上可提升传动比达40%。**

- **链接: [https://arxiv.org/pdf/2512.15448v1](https://arxiv.org/pdf/2512.15448v1)**

> **作者:** Sinan Emre; Victor Barasuol; Matteo Villa; Claudio Semini
>
> **备注:** 22nd International Conference on Advanced Robotics (ICAR 2025)
>
> **摘要:** This paper presents a Load-Based Variable Transmission (LBVT) mechanism designed to enhance robotic actuation by dynamically adjusting the transmission ratio in response to external torque demands. Unlike existing variable transmission systems that require additional actuators for active control, the proposed LBVT mechanism leverages a pre-tensioned spring and a four-bar linkage to passively modify the transmission ratio, thereby reducing the complexity of robot joint actuation systems. The effectiveness of the LBVT mechanism is evaluated through simulation-based analyses. The results confirm that the system achieves up to a 40 percent increase in transmission ratio upon reaching a predefined torque threshold, effectively amplifying joint torque when required without additional actuation. Furthermore, the simulations demonstrate a torque amplification effect triggered when the applied force exceeds 18 N, highlighting the system ability to autonomously respond to varying load conditions. This research contributes to the development of lightweight, efficient, and adaptive transmission systems for robotic applications, particularly in legged robots where dynamic torque adaptation is critical.
>
---
#### [new 002] VLA-AN: An Efficient and Onboard Vision-Language-Action Framework for Aerial Navigation in Complex Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出VLA-AN框架，面向无人机在复杂环境中的自主导航任务，解决域偏移、时序推理弱、生成式动作不安全及机载部署难四大问题；通过3D-GS建模、三阶段训练、轻量安全动作模块和深度部署优化，实现高效、安全、实时的端到端闭环导航。**

- **链接: [https://arxiv.org/pdf/2512.15258v1](https://arxiv.org/pdf/2512.15258v1)**

> **作者:** Yuze Wu; Mo Zhu; Xingxing Li; Yuheng Du; Yuxin Fan; Wenjun Li; Xin Zhou; Fei Gao
>
> **摘要:** This paper proposes VLA-AN, an efficient and onboard Vision-Language-Action (VLA) framework dedicated to autonomous drone navigation in complex environments. VLA-AN addresses four major limitations of existing large aerial navigation models: the data domain gap, insufficient temporal navigation with reasoning, safety issues with generative action policies, and onboard deployment constraints. First, we construct a high-fidelity dataset utilizing 3D Gaussian Splatting (3D-GS) to effectively bridge the domain gap. Second, we introduce a progressive three-stage training framework that sequentially reinforces scene comprehension, core flight skills, and complex navigation capabilities. Third, we design a lightweight, real-time action module coupled with geometric safety correction. This module ensures fast, collision-free, and stable command generation, mitigating the safety risks inherent in stochastic generative policies. Finally, through deep optimization of the onboard deployment pipeline, VLA-AN achieves a robust real-time 8.3x improvement in inference throughput on resource-constrained UAVs. Extensive experiments demonstrate that VLA-AN significantly improves spatial grounding, scene reasoning, and long-horizon navigation, achieving a maximum single-task success rate of 98.1%, and providing an efficient, practical solution for realizing full-chain closed-loop autonomy in lightweight aerial robots.
>
---
#### [new 003] OMCL: Open-vocabulary Monte Carlo Localization
- **分类: cs.RO**

- **简介: 该论文属机器人定位任务，解决多模态地图与视觉观测鲁棒匹配问题。提出OMCL方法，将开放词汇视觉语言特征引入蒙特卡洛定位，支持跨模态观测-地图关联，并支持自然语言初始化全局定位。在室内外数据集上验证了泛化性。**

- **链接: [https://arxiv.org/pdf/2512.15557v1](https://arxiv.org/pdf/2512.15557v1)**

> **作者:** Evgenii Kruzhkov; Raphael Memmesheimer; Sven Behnke
>
> **备注:** Accepted to IEEE RA-L
>
> **摘要:** Robust robot localization is an important prerequisite for navigation planning. If the environment map was created from different sensors, robot measurements must be robustly associated with map features. In this work, we extend Monte Carlo Localization using vision-language features. These open-vocabulary features enable to robustly compute the likelihood of visual observations, given a camera pose and a 3D map created from posed RGB-D images or aligned point clouds. The abstract vision-language features enable to associate observations and map elements from different modalities. Global localization can be initialized by natural language descriptions of the objects present in the vicinity of locations. We evaluate our approach using Matterport3D and Replica for indoor scenes and demonstrate generalization on SemanticKITTI for outdoor scenes.
>
---
#### [new 004] NAP3D: NeRF Assisted 3D-3D Pose Alignment for Autonomous Vehicles
- **分类: cs.RO**

- **简介: 该论文提出NAP3D方法，属自动驾驶车辆定位任务，旨在解决长期运行中因传感器噪声和漂移导致的位姿累积误差问题。它利用预训练NeRF与当前深度图进行3D-3D点对齐，实现无需回环、不依赖重访的位姿精调，提升几何一致性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.15080v1](https://arxiv.org/pdf/2512.15080v1)**

> **作者:** Gaurav Bansal
>
> **备注:** 10 pages, 5 figures, 2 tables
>
> **摘要:** Accurate localization is essential for autonomous vehicles, yet sensor noise and drift over time can lead to significant pose estimation errors, particularly in long-horizon environments. A common strategy for correcting accumulated error is visual loop closure in SLAM, which adjusts the pose graph when the agent revisits previously mapped locations. These techniques typically rely on identifying visual mappings between the current view and previously observed scenes and often require fusing data from multiple sensors. In contrast, this work introduces NeRF-Assisted 3D-3D Pose Alignment (NAP3D), a complementary approach that leverages 3D-3D correspondences between the agent's current depth image and a pre-trained Neural Radiance Field (NeRF). By directly aligning 3D points from the observed scene with synthesized points from the NeRF, NAP3D refines the estimated pose even from novel viewpoints, without relying on revisiting previously observed locations. This robust 3D-3D formulation provides advantages over conventional 2D-3D localization methods while remaining comparable in accuracy and applicability. Experiments demonstrate that NAP3D achieves camera pose correction within 5 cm on a custom dataset, robustly outperforming a 2D-3D Perspective-N-Point baseline. On TUM RGB-D, NAP3D consistently improves 3D alignment RMSE by approximately 6 cm compared to this baseline given varying noise, despite PnP achieving lower raw rotation and translation parameter error in some regimes, highlighting NAP3D's improved geometric consistency in 3D space. By providing a lightweight, dataset-agnostic tool, NAP3D complements existing SLAM and localization pipelines when traditional loop closure is unavailable.
>
---
#### [new 005] Infrastructure-based Autonomous Mobile Robots for Internal Logistics -- Challenges and Future Perspectives
- **分类: cs.RO**

- **简介: 该论文属系统架构与应用研究任务，旨在解决工业场景中AMR系统可扩展性、鲁棒性与人机协同不足问题。提出基础设施赋能的AMR参考架构，整合外部感知、边缘云与车载智能，并在重型车辆制造厂实证验证，辅以UX评估。**

- **链接: [https://arxiv.org/pdf/2512.15215v1](https://arxiv.org/pdf/2512.15215v1)**

> **作者:** Erik Brorsson; Kristian Ceder; Ze Zhang; Sabino Francesco Roselli; Endre Erős; Martin Dahl; Beatrice Alenljung; Jessica Lindblom; Thanh Bui; Emmanuel Dean; Lennart Svensson; Kristofer Bengtsson; Per-Lage Götvall; Knut Åkesson
>
> **摘要:** The adoption of Autonomous Mobile Robots (AMRs) for internal logistics is accelerating, with most solutions emphasizing decentralized, onboard intelligence. While AMRs in indoor environments like factories can be supported by infrastructure, involving external sensors and computational resources, such systems remain underexplored in the literature. This paper presents a comprehensive overview of infrastructure-based AMR systems, outlining key opportunities and challenges. To support this, we introduce a reference architecture combining infrastructure-based sensing, on-premise cloud computing, and onboard autonomy. Based on the architecture, we review core technologies for localization, perception, and planning. We demonstrate the approach in a real-world deployment in a heavy-vehicle manufacturing environment and summarize findings from a user experience (UX) evaluation. Our aim is to provide a holistic foundation for future development of scalable, robust, and human-compatible AMR systems in complex industrial environments.
>
---
#### [new 006] mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出mimic-video模型，属机器人控制任务，旨在解决VLAs缺乏物理因果理解、依赖大量专家数据的问题。它用预训练视频模型联合建模语义与动态，并设计流匹配动作解码器作为逆动力学模型，提升样本效率与收敛速度。**

- **链接: [https://arxiv.org/pdf/2512.15692v1](https://arxiv.org/pdf/2512.15692v1)**

> **作者:** Jonas Pai; Liam Achenbach; Victoriano Montesinos; Benedek Forrai; Oier Mees; Elvis Nava
>
> **摘要:** Prevailing Vision-Language-Action Models (VLAs) for robotic manipulation are built upon vision-language backbones pretrained on large-scale, but disconnected static web data. As a result, despite improved semantic generalization, the policy must implicitly infer complex physical dynamics and temporal dependencies solely from robot trajectories. This reliance creates an unsustainable data burden, necessitating continuous, large-scale expert data collection to compensate for the lack of innate physical understanding. We contend that while vision-language pretraining effectively captures semantic priors, it remains blind to physical causality. A more effective paradigm leverages video to jointly capture semantics and visual dynamics during pretraining, thereby isolating the remaining task of low-level control. To this end, we introduce \model, a novel Video-Action Model (VAM) that pairs a pretrained Internet-scale video model with a flow matching-based action decoder conditioned on its latent representations. The decoder serves as an Inverse Dynamics Model (IDM), generating low-level robot actions from the latent representation of video-space action plans. Our extensive evaluation shows that our approach achieves state-of-the-art performance on simulated and real-world robotic manipulation tasks, improving sample efficiency by 10x and convergence speed by 2x compared to traditional VLA architectures.
>
---
#### [new 007] HERO: Hierarchical Traversable 3D Scene Graphs for Embodied Navigation Among Movable Obstacles
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属具身导航任务，解决静态场景图无法处理可移动障碍物导致的可达性差问题。提出 HERO 框架，构建分层可通行3D场景图，将可操作障碍物建模为可穿越路径，提升导航效率与可达性。**

- **链接: [https://arxiv.org/pdf/2512.15047v1](https://arxiv.org/pdf/2512.15047v1)**

> **作者:** Yunheng Wang; Yixiao Feng; Yuetong Fang; Shuning Zhang; Tan Jing; Jian Li; Xiangrui Jiang; Renjing Xu
>
> **摘要:** 3D Scene Graphs (3DSGs) constitute a powerful representation of the physical world, distinguished by their abilities to explicitly model the complex spatial, semantic, and functional relationships between entities, rendering a foundational understanding that enables agents to interact intelligently with their environment and execute versatile behaviors. Embodied navigation, as a crucial component of such capabilities, leverages the compact and expressive nature of 3DSGs to enable long-horizon reasoning and planning in complex, large-scale environments. However, prior works rely on a static-world assumption, defining traversable space solely based on static spatial layouts and thereby treating interactable obstacles as non-traversable. This fundamental limitation severely undermines their effectiveness in real-world scenarios, leading to limited reachability, low efficiency, and inferior extensibility. To address these issues, we propose HERO, a novel framework for constructing Hierarchical Traversable 3DSGs, that redefines traversability by modeling operable obstacles as pathways, capturing their physical interactivity, functional semantics, and the scene's relational hierarchy. The results show that, relative to its baseline, HERO reduces PL by 35.1% in partially obstructed environments and increases SR by 79.4% in fully obstructed ones, demonstrating substantially higher efficiency and reachability.
>
---
#### [new 008] A Network-Based Framework for Modeling and Analyzing Human-Robot Coordination Strategies
- **分类: cs.RO; cs.HC**

- **简介: 该论文属人机协同设计任务，旨在解决现有框架难以支持概念设计阶段对协调动态性进行推理的问题。作者提出一种融合功能建模与图论的网络化计算框架，显式刻画协调需求及其时序演化，并通过灾害机器人案例验证其在权衡分析与协作能力识别中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.15282v1](https://arxiv.org/pdf/2512.15282v1)**

> **作者:** Martijn IJtsma; Salvatore Hargis
>
> **备注:** Under review at IEEE Transactions on Human-Machine Systems. 12 pages, 5 figures
>
> **摘要:** Studies of human-robot interaction in dynamic and unstructured environments show that as more advanced robotic capabilities are deployed, the need for cooperative competencies to support collaboration with human problem-holders increases. Designing human-robot systems to meet these demands requires an explicit understanding of the work functions and constraints that shape the feasibility of alternative joint work strategies. Yet existing human-robot interaction frameworks either emphasize computational support for real-time execution or rely on static representations for design, offering limited support for reasoning about coordination dynamics during early-stage conceptual design. To address this gap, this article presents a novel computational framework for analyzing joint work strategies in human-robot systems by integrating techniques from functional modeling with graph-theoretic representations. The framework characterizes collective work in terms of the relationships among system functions and the physical and informational structure of the work environment, while explicitly capturing how coordination demands evolve over time. Its use during conceptual design is demonstrated through a case study in disaster robotics, which shows how the framework can be used to support early trade-space exploration of human-robot coordination strategies and to identify cooperative competencies that support flexible management of coordination overhead. These results show how the framework makes coordination demands and their temporal evolution explicit, supporting design-time reasoning about cooperative competency requirements and work demands prior to implementation.
>
---
#### [new 009] Breathe with Me: Synchronizing Biosignals for User Embodiment in Robots
- **分类: cs.RO; cs.HC**

- **简介: 该论文探索通过实时同步用户呼吸（embreathment）增强其在机器人中的具身感。属人机交互任务，旨在解决传统视觉-运动同步外的生理信号整合问题。作者开展被试内实验，对比呼吸同步/非同步控制机械臂的效果，证实同步显著提升身体所有权并更受偏好。**

- **链接: [https://arxiv.org/pdf/2512.14952v1](https://arxiv.org/pdf/2512.14952v1)**

> **作者:** Iddo Yehoshua Wald; Amber Maimon; Shiyao Zhang; Dennis Küster; Robert Porzel; Tanja Schultz; Rainer Malaka
>
> **备注:** Accepted to appear in the ACM/IEEE International Conference on Human-Robot Interaction (HRI '26), Edinburgh, United Kingdom. Iddo Yehoshua Wald and Amber Maimon contributed equally
>
> **摘要:** Embodiment of users within robotic systems has been explored in human-robot interaction, most often in telepresence and teleoperation. In these applications, synchronized visuomotor feedback can evoke a sense of body ownership and agency, contributing to the experience of embodiment. We extend this work by employing embreathment, the representation of the user's own breath in real time, as a means for enhancing user embodiment experience in robots. In a within-subjects experiment, participants controlled a robotic arm, while its movements were either synchronized or non-synchronized with their own breath. Synchrony was shown to significantly increase body ownership, and was preferred by most participants. We propose the representation of physiological signals as a novel interoceptive pathway for human-robot interaction, and discuss implications for telepresence, prosthetics, collaboration with robots, and shared autonomy.
>
---
#### [new 010] ISS Policy : Scalable Diffusion Policy with Implicit Scene Supervision
- **分类: cs.RO**

- **简介: 该论文提出ISS Policy，一种基于点云的3D扩散策略模型，旨在解决视觉模仿学习中忽略3D场景结构导致泛化差、训练低效的问题。通过引入隐式场景监督模块，约束动作预测符合几何演化，提升鲁棒性与可扩展性，在仿真与真实机器人任务中均达SOTA。**

- **链接: [https://arxiv.org/pdf/2512.15020v1](https://arxiv.org/pdf/2512.15020v1)**

> **作者:** Wenlong Xia; Jinhao Zhang; Ce Zhang; Yaojia Wang; Youmin Gong; Jie Mei
>
> **摘要:** Vision-based imitation learning has enabled impressive robotic manipulation skills, but its reliance on object appearance while ignoring the underlying 3D scene structure leads to low training efficiency and poor generalization. To address these challenges, we introduce \emph{Implicit Scene Supervision (ISS) Policy}, a 3D visuomotor DiT-based diffusion policy that predicts sequences of continuous actions from point cloud observations. We extend DiT with a novel implicit scene supervision module that encourages the model to produce outputs consistent with the scene's geometric evolution, thereby improving the performance and robustness of the policy. Notably, ISS Policy achieves state-of-the-art performance on both single-arm manipulation tasks (MetaWorld) and dexterous hand manipulation (Adroit). In real-world experiments, it also demonstrates strong generalization and robustness. Additional ablation studies show that our method scales effectively with both data and parameters. Code and videos will be released.
>
---
#### [new 011] BEV-Patch-PF: Particle Filtering with BEV-Aerial Feature Matching for Off-Road Geo-Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出BEV-Patch-PF，一种无GPS的越野场景顺序地理定位方法。针对密集树冠和阴影下定位失效问题，融合车载BEV特征图与航拍特征图，通过粒子滤波匹配实现鲁棒实时定位，在真实数据集上显著降低轨迹误差。**

- **链接: [https://arxiv.org/pdf/2512.15111v1](https://arxiv.org/pdf/2512.15111v1)**

> **作者:** Dongmyeong Lee; Jesse Quattrociocchi; Christian Ellis; Rwik Rana; Amanda Adkins; Adam Uccello; Garrett Warnell; Joydeep Biswas
>
> **摘要:** We propose BEV-Patch-PF, a GPS-free sequential geo-localization system that integrates a particle filter with learned bird's-eye-view (BEV) and aerial feature maps. From onboard RGB and depth images, we construct a BEV feature map. For each 3-DoF particle pose hypothesis, we crop the corresponding patch from an aerial feature map computed from a local aerial image queried around the approximate location. BEV-Patch-PF computes a per-particle log-likelihood by matching the BEV feature to the aerial patch feature. On two real-world off-road datasets, our method achieves 7.5x lower absolute trajectory error (ATE) on seen routes and 7.0x lower ATE on unseen routes than a retrieval-based baseline, while maintaining accuracy under dense canopy and shadow. The system runs in real time at 10 Hz on an NVIDIA Tesla T4, enabling practical robot deployment.
>
---
#### [new 012] Remotely Detectable Robot Policy Watermarking
- **分类: cs.RO; cs.CR; cs.LG; eess.SY**

- **简介: 该论文提出CoNoCo方法，解决机器人策略水印的远程检测难题：在仅能获取外部观测（如视频）的条件下，通过嵌入频谱信号实现非侵入式所有权验证。它保证策略性能不变，并在仿真与真实机器人上验证了跨模态鲁棒检测效果。**

- **链接: [https://arxiv.org/pdf/2512.15379v1](https://arxiv.org/pdf/2512.15379v1)**

> **作者:** Michael Amir; Manon Flageat; Amanda Prorok
>
> **摘要:** The success of machine learning for real-world robotic systems has created a new form of intellectual property: the trained policy. This raises a critical need for novel methods that verify ownership and detect unauthorized, possibly unsafe misuse. While watermarking is established in other domains, physical policies present a unique challenge: remote detection. Existing methods assume access to the robot's internal state, but auditors are often limited to external observations (e.g., video footage). This ``Physical Observation Gap'' means the watermark must be detected from signals that are noisy, asynchronous, and filtered by unknown system dynamics. We formalize this challenge using the concept of a \textit{glimpse sequence}, and introduce Colored Noise Coherency (CoNoCo), the first watermarking strategy designed for remote detection. CoNoCo embeds a spectral signal into the robot's motions by leveraging the policy's inherent stochasticity. To show it does not degrade performance, we prove CoNoCo preserves the marginal action distribution. Our experiments demonstrate strong, robust detection across various remote modalities, including motion capture and side-way/top-down video footage, in both simulated and real-world robot experiments. This work provides a necessary step toward protecting intellectual property in robotics, offering the first method for validating the provenance of physical policies non-invasively, using purely remote observations.
>
---
#### [new 013] An Open Toolkit for Underwater Field Robotics
- **分类: cs.RO**

- **简介: 该论文属水下机器人硬件开源任务，旨在解决水下操纵系统成本高、封闭、难复现的问题。作者开发了开源的水下机器人关节（URJ）硬件 toolkit，含防水关节、控制电子与ROS2软件，并经40米实测验证，支持多种水下操作设备。**

- **链接: [https://arxiv.org/pdf/2512.15597v1](https://arxiv.org/pdf/2512.15597v1)**

> **作者:** Giacomo Picardi; Saverio Iacoponi; Matias Carandell; Jorge Aguirregomezcorta; Mrudul Chellapurath; Joaquin del Rio; Marcello Calisti; Iacopo Aguzzi
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Underwater robotics is becoming increasingly important for marine science, environmental monitoring, and subsea industrial operations, yet the development of underwater manipulation and actuation systems remains restricted by high costs, proprietary designs, and limited access to modular, research-oriented hardware. While open-source initiatives have democratized vehicle construction and control software, a substantial gap persists for joint-actuated systems-particularly those requiring waterproof, feedback-enabled actuation suitable for manipulators, grippers, and bioinspired devices. As a result, many research groups face lengthy development cycles, limited reproducibility, and difficulty transitioning laboratory prototypes to field-ready platforms. To address this gap, we introduce an open, cost-effective hardware and software toolkit for underwater manipulation research. The toolkit includes a depth-rated Underwater Robotic Joint (URJ) with early leakage detection, compact control and power management electronics, and a ROS2-based software stack for sensing and multi-mode actuation. All CAD models, fabrication files, PCB sources, firmware, and ROS2 packages are openly released, enabling local manufacturing, modification, and community-driven improvement. The toolkit has undergone extensive laboratory testing and multiple field deployments, demonstrating reliable operation up to 40 m depth across diverse applications, including a 3-DoF underwater manipulator, a tendon-driven soft gripper, and an underactuated sediment sampler. These results validate the robustness, versatility, and reusability of the toolkit for real marine environments. By providing a fully open, field-tested platform, this work aims to lower the barrier to entry for underwater manipulation research, improve reproducibility, and accelerate innovation in underwater field robotics.
>
---
#### [new 014] GuangMing-Explorer: A Four-Legged Robot Platform for Autonomous Exploration in General Environments
- **分类: cs.RO**

- **简介: 该论文提出GuangMing-Explorer四足机器人平台，面向通用环境的自主探索任务。旨在解决现有系统软硬件割裂、缺乏完整实用方案的问题。工作包括平台整体设计（硬件、软件栈、算法部署）及真实环境实验验证。**

- **链接: [https://arxiv.org/pdf/2512.15309v1](https://arxiv.org/pdf/2512.15309v1)**

> **作者:** Kai Zhang; Shoubin Chen; Dong Li; Baiyang Zhang; Tao Huang; Zehao Wu; Jiasheng Chen; Bo Zhang
>
> **备注:** 6 pages, published in ICUS2025
>
> **摘要:** Autonomous exploration is a fundamental capability that tightly integrates perception, planning, control, and motion execution. It plays a critical role in a wide range of applications, including indoor target search, mapping of extreme environments, resource exploration, etc. Despite significant progress in individual components, a holistic and practical description of a completely autonomous exploration system, encompassing both hardware and software, remains scarce. In this paper, we present GuangMing-Explorer, a fully integrated autonomous exploration platform designed for robust operation across diverse environments. We provide a comprehensive overview of the system architecture, including hardware design, software stack, algorithm deployment, and experimental configuration. Extensive real-world experiments demonstrate the platform's effectiveness and efficiency in executing autonomous exploration tasks, highlighting its potential for practical deployment in complex and unstructured environments.
>
---
#### [new 015] MiVLA: Towards Generalizable Vision-Language-Action Model with Human-Robot Mutual Imitation Pre-training
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MiVLA模型，属视觉-语言-动作（VLA）任务，旨在解决现有VLAs因视角、外观和形态差异导致的跨人机泛化能力弱问题。通过人类与机器人双向行为模仿预训练，利用手/臂运动学对齐，融合真实人类视频与仿真机器人数据，显著提升跨平台泛化性能。**

- **链接: [https://arxiv.org/pdf/2512.15411v1](https://arxiv.org/pdf/2512.15411v1)**

> **作者:** Zhenhan Yin; Xuanhan Wang; Jiahao Jiang; Kaiyuan Deng; Pengqi Chen; Shuangle Li; Chong Liu; Xing Xu; ingkuan Song; Lianli Gao; Heng Tao Shen
>
> **摘要:** While leveraging abundant human videos and simulated robot data poses a scalable solution to the scarcity of real-world robot data, the generalization capability of existing vision-language-action models (VLAs) remains limited by mismatches in camera views, visual appearance, and embodiment morphologies. To overcome this limitation, we propose MiVLA, a generalizable VLA empowered by human-robot mutual imitation pre-training, which leverages inherent behavioral similarity between human hands and robotic arms to build a foundation of strong behavioral priors for both human actions and robotic control. Specifically, our method utilizes kinematic rules with left/right hand coordinate systems for bidirectional alignment between human and robot action spaces. Given human or simulated robot demonstrations, MiVLA is trained to forecast behavior trajectories for one embodiment, and imitate behaviors for another one unseen in the demonstration. Based on this mutual imitation, it integrates the behavioral fidelity of real-world human data with the manipulative diversity of simulated robot data into a unified model, thereby enhancing the generalization capability for downstream tasks. Extensive experiments conducted on both simulation and real-world platforms with three robots (ARX, PiPer and LocoMan), demonstrate that MiVLA achieves strong improved generalization capability, outperforming state-of-the-art VLAs (e.g., $\boldsymbolπ_{0}$, $\boldsymbolπ_{0.5}$ and H-RDT) by 25% in simulation, and 14% in real-world robot control tasks.
>
---
#### [new 016] EPSM: A Novel Metric to Evaluate the Safety of Environmental Perception in Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出EPSM——一种面向自动驾驶环境感知安全性的新型评估指标。针对传统精度类指标忽视安全风险的问题，它联合建模目标与车道检测任务，设计轻量级对象安全度量和考虑任务关联的车道安全度量，生成统一可解释的安全评分，并在DeepAccident数据集上验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.15195v1](https://arxiv.org/pdf/2512.15195v1)**

> **作者:** Jörg Gamerdinger; Sven Teufel; Stephan Amann; Lukas Marc Listl; Oliver Bringmann
>
> **备注:** Submitted at IEEE IV 2026
>
> **摘要:** Extensive evaluation of perception systems is crucial for ensuring the safety of intelligent vehicles in complex driving scenarios. Conventional performance metrics such as precision, recall and the F1-score assess the overall detection accuracy, but they do not consider the safety-relevant aspects of perception. Consequently, perception systems that achieve high scores in these metrics may still cause misdetections that could lead to severe accidents. Therefore, it is important to evaluate not only the overall performance of perception systems, but also their safety. We therefore introduce a novel safety metric for jointly evaluating the most critical perception tasks, object and lane detection. Our proposed framework integrates a new, lightweight object safety metric that quantifies the potential risk associated with object detection errors, as well as an lane safety metric including the interdependence between both tasks that can occur in safety evaluation. The resulting combined safety score provides a unified, interpretable measure of perception safety performance. Using the DeepAccident dataset, we demonstrate that our approach identifies safety critical perception errors that conventional performance metrics fail to capture. Our findings emphasize the importance of safety-centric evaluation methods for perception systems in autonomous driving.
>
---
#### [new 017] I am here for you": How relational conversational AI appeals to adolescents, especially those who are socially and emotionally vulnerable
- **分类: cs.HC; cs.AI; cs.RO**

- **简介: 该论文属人机交互与AI伦理研究，旨在探究对话风格对青少年AI依赖的影响。通过284组青少年-家长在线实验，比较关系型（拟人化）与透明型（明确非人）聊天机器人回应方式，发现关系型风格更易被脆弱青少年偏好，提升拟人感与情感亲近，但也增加情感依赖风险。**

- **链接: [https://arxiv.org/pdf/2512.15117v1](https://arxiv.org/pdf/2512.15117v1)**

> **作者:** Pilyoung Kim; Yun Xie; Sujin Yang
>
> **摘要:** General-purpose conversational AI chatbots and AI companions increasingly provide young adolescents with emotionally supportive conversations, raising questions about how conversational style shapes anthropomorphism and emotional reliance. In a preregistered online experiment with 284 adolescent-parent dyads, youth aged 11-15 and their parents read two matched transcripts in which a chatbot responded to an everyday social problem using either a relational style (first-person, affiliative, commitment language) or a transparent style (explicit nonhumanness, informational tone). Adolescents more often preferred the relational than the transparent style, whereas parents were more likely to prefer transparent style than adolescents. Adolescents rated the relational chatbot as more human-like, likable, trustworthy and emotionally close, while perceiving both styles as similarly helpful. Adolescents who preferred relational style had lower family and peer relationship quality and higher stress and anxiety than those preferring transparent style or both chatbots. These findings identify conversational style as a key design lever for youth AI safety, showing that relational framing heightens anthropomorphism, trust and emotional closeness and can be especially appealing to socially and emotionally vulnerable adolescents, who may be at increased risk for emotional reliance on conversational AI.
>
---
#### [new 018] Photorealistic Phantom Roads in Real Scenes: Disentangling 3D Hallucinations from Physical Geometry
- **分类: cs.CV; cs.RO**

- **简介: 该论文聚焦单目深度估计任务，解决深度模型因语义先验导致的“3D幻境”（在平面区域误估非平面结构）这一安全风险。工作包括：构建首个真实幻觉基准3D-Mirage；提出拉普拉斯评估指标（DCS/CCS）；设计参数高效方法“接地自蒸馏”强制幻觉区域平面性。**

- **链接: [https://arxiv.org/pdf/2512.15423v1](https://arxiv.org/pdf/2512.15423v1)**

> **作者:** Hoang Nguyen; Xiaohao Xu; Xiaonan Huang
>
> **摘要:** Monocular depth foundation models achieve remarkable generalization by learning large-scale semantic priors, but this creates a critical vulnerability: they hallucinate illusory 3D structures from geometrically planar but perceptually ambiguous inputs. We term this failure the 3D Mirage. This paper introduces the first end-to-end framework to probe, quantify, and tame this unquantified safety risk. To probe, we present 3D-Mirage, the first benchmark of real-world illusions (e.g., street art) with precise planar-region annotations and context-restricted crops. To quantify, we propose a Laplacian-based evaluation framework with two metrics: the Deviation Composite Score (DCS) for spurious non-planarity and the Confusion Composite Score (CCS) for contextual instability. To tame this failure, we introduce Grounded Self-Distillation, a parameter-efficient strategy that surgically enforces planarity on illusion ROIs while using a frozen teacher to preserve background knowledge, thus avoiding catastrophic forgetting. Our work provides the essential tools to diagnose and mitigate this phenomenon, urging a necessary shift in MDE evaluation from pixel-wise accuracy to structural and contextual robustness. Our code and benchmark will be publicly available to foster this exciting research direction.
>
---
#### [new 019] SocialNav-MoE: A Mixture-of-Experts Vision Language Model for Socially Compliant Navigation with Reinforcement Fine-Tuning
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向机器人社会合规导航任务，解决现有方法重安全轻社交、大模型难实时部署的问题。提出轻量级MoE架构SocialNav-MoE，结合强化微调与语义相似性奖励（SSR），并系统评估小语言模型、路由策略及视觉编码器组合，在SNEI数据集上实现精度与效率的平衡。**

- **链接: [https://arxiv.org/pdf/2512.14757v1](https://arxiv.org/pdf/2512.14757v1)**

> **作者:** Tomohito Kawabata; Xinyu Zhang; Ling Xiao
>
> **摘要:** For robots navigating in human-populated environments, safety and social compliance are equally critical, yet prior work has mostly emphasized safety. Socially compliant navigation that accounts for human comfort, social norms, and contextual appropriateness remains underexplored. Vision language models (VLMs) show promise for this task; however, large-scale models incur substantial computational overhead, leading to higher inference latency and energy consumption, which makes them unsuitable for real-time deployment on resource-constrained robotic platforms. To address this issue, we investigate the effectiveness of small VLM and propose SocialNav-MoE, an efficient Mixture-of-Experts vision language model for socially compliant navigation with reinforcement fine-tuning (RFT). We further introduce a semantic similarity reward (SSR) to effectively leverage RFT for enhancing the decision-making capabilities. Additionally, we study the effectiveness of different small language model types (Phi, Qwen, and StableLM), routing strategies, and vision encoders (CLIP vs. SigLIP, frozen vs. fine-tuned). Experiments on the SNEI dataset demonstrate that SocialNav-MoE achieves an excellent balance between navigation accuracy and efficiency. The proposed SSR function is more effective than hard-level and character-level rewards. Source code will be released upon acceptance.
>
---
#### [new 020] Criticality Metrics for Relevance Classification in Safety Evaluation of Object Detection in Automated Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属自动驾驶安全评估任务，旨在解决物体检测中相关性（关键性）分类不准确的问题。作者系统综述并实证评估现有关键性指标，提出双向关键性评分与多指标聚合新策略，在DeepAccident数据集上将关键性分类准确率提升达100%。**

- **链接: [https://arxiv.org/pdf/2512.15181v1](https://arxiv.org/pdf/2512.15181v1)**

> **作者:** Jörg Gamerdinger; Sven Teufel; Stephan Amann; Oliver Bringmann
>
> **备注:** Accepted at IEEE ICVES 2025
>
> **摘要:** Ensuring safety is the primary objective of automated driving, which necessitates a comprehensive and accurate perception of the environment. While numerous performance evaluation metrics exist for assessing perception capabilities, incorporating safety-specific metrics is essential to reliably evaluate object detection systems. A key component for safety evaluation is the ability to distinguish between relevant and non-relevant objects - a challenge addressed by criticality or relevance metrics. This paper presents the first in-depth analysis of criticality metrics for safety evaluation of object detection systems. Through a comprehensive review of existing literature, we identify and assess a range of applicable metrics. Their effectiveness is empirically validated using the DeepAccident dataset, which features a variety of safety-critical scenarios. To enhance evaluation accuracy, we propose two novel application strategies: bidirectional criticality rating and multi-metric aggregation. Our approach demonstrates up to a 100% improvement in terms of criticality classification accuracy, highlighting its potential to significantly advance the safety evaluation of object detection systems in automated vehicles.
>
---
#### [new 021] QuantGraph: A Receding-Horizon Quantum Graph Solver
- **分类: quant-ph; cs.RO; eess.SY; physics.comp-ph**

- **简介: 该论文提出QuantGraph，一种面向图优化的量子增强框架。旨在解决动态规划在大规模图问题中计算复杂度高的问题。工作包括：两阶段Grover自适应搜索（局部阈值剪枝+全局精调），并嵌入滚动时域模型预测控制以提升鲁棒性与精度。**

- **链接: [https://arxiv.org/pdf/2512.15476v1](https://arxiv.org/pdf/2512.15476v1)**

> **作者:** Pranav Vaidhyanathan; Aristotelis Papatheodorou; David R. M. Arvidsson-Shukur; Mark T. Mitchison; Natalia Ares; Ioannis Havoutis
>
> **备注:** P.Vaidhyanathan and A. Papatheodorou contributed equally to this work. 11 pages, 4 figures, 1 table, 2 algorithms
>
> **摘要:** Dynamic programming is a cornerstone of graph-based optimization. While effective, it scales unfavorably with problem size. In this work, we present QuantGraph, a two-stage quantum-enhanced framework that casts local and global graph-optimization problems as quantum searches over discrete trajectory spaces. The solver is designed to operate efficiently by first finding a sequence of locally optimal transitions in the graph (local stage), without considering full trajectories. The accumulated cost of these transitions acts as a threshold that prunes the search space (up to 60% reduction for certain examples). The subsequent global stage, based on this threshold, refines the solution. Both stages utilize variants of the Grover-adaptive-search algorithm. To achieve scalability and robustness, we draw on principles from control theory and embed QuantGraph's global stage within a receding-horizon model-predictive-control scheme. This classical layer stabilizes and guides the quantum search, improving precision and reducing computational burden. In practice, the resulting closed-loop system exhibits robust behavior and lower overall complexity. Notably, for a fixed query budget, QuantGraph attains a 2x increase in control-discretization precision while still benefiting from Grover-search's inherent quadratic speedup compared to classical methods.
>
---
## 更新

#### [replaced 001] Embodied Co-Design for Rapidly Evolving Agents: Taxonomy, Frontiers, and Challenges
- **分类: cs.RO; cs.AI; cs.ET; eess.SY**

- **简介: 该论文是一篇综述任务，旨在系统梳理“具身协同设计”（ECD）这一新兴范式。它解决ECD缺乏统一框架的问题，提出分层分类体系（含脑、体、环境三要素及四类框架），整合百余项研究，评述基准、应用与挑战，并开源配套项目。**

- **链接: [https://arxiv.org/pdf/2512.04770v2](https://arxiv.org/pdf/2512.04770v2)**

> **作者:** Yuxing Wang; Zhiyu Chen; Tiantian Zhang; Qiyue Yin; Yongzhe Chang; Zhiheng Li; Liang Wang; Xueqian Wang
>
> **摘要:** Brain-body co-evolution enables animals to develop complex behaviors in their environments. Inspired by this biological synergy, embodied co-design (ECD) has emerged as a transformative paradigm for creating intelligent agents-from virtual creatures to physical robots-by jointly optimizing their morphologies and controllers rather than treating control in isolation. This integrated approach facilitates richer environmental interactions and robust task performance. In this survey, we provide a systematic overview of recent advances in ECD. We first formalize the concept of ECD and position it within related fields. We then introduce a hierarchical taxonomy: a lower layer that breaks down agent design into three fundamental components-controlling brain, body morphology, and task environment-and an upper layer that integrates these components into four major ECD frameworks: bi-level, single-level, generative, and open-ended. This taxonomy allows us to synthesize insights from more than one hundred recent studies. We further review notable benchmarks, datasets, and applications in both simulated and real-world scenarios. Finally, we identify significant challenges and offer insights into promising future research directions. A project associated with this survey has been created at https://github.com/Yuxing-Wang-THU/SurveyBrainBody.
>
---
#### [replaced 002] Registering the 4D Millimeter Wave Radar Point Clouds Via Generalized Method of Moments
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对4D毫米波雷达点云稀疏、噪声大导致配准困难的问题，提出基于广义矩估计（GMM）的无对应点配准方法，无需显式点对匹配，具有理论一致性。实验表明其精度与鲁棒性优于基准，媲美激光雷达方案。**

- **链接: [https://arxiv.org/pdf/2508.02187v2](https://arxiv.org/pdf/2508.02187v2)**

> **作者:** Xingyi Li; Han Zhang; Ziliang Wang; Yukai Yang; Weidong Chen
>
> **摘要:** 4D millimeter wave radars (4D radars) are new emerging sensors that provide point clouds of objects with both position and radial velocity measurements. Compared to LiDARs, they are more affordable and reliable sensors for robots' perception under extreme weather conditions. On the other hand, point cloud registration is an essential perception module that provides robot's pose feedback information in applications such as Simultaneous Localization and Mapping (SLAM). Nevertheless, the 4D radar point clouds are sparse and noisy compared to those of LiDAR, and hence we shall confront great challenges in registering the radar point clouds. To address this issue, we propose a point cloud registration framework for 4D radars based on Generalized Method of Moments. The method does not require explicit point-to-point correspondences between the source and target point clouds, which is difficult to compute for sparse 4D radar point clouds. Moreover, we show the consistency of the proposed method. Experiments on both synthetic and real-world datasets show that our approach achieves higher accuracy and robustness than benchmarks, and the accuracy is even comparable to LiDAR-based frameworks.
>
---
#### [replaced 003] Event Camera Meets Mobile Embodied Perception: Abstraction, Algorithm, Acceleration, Application
- **分类: cs.RO; cs.CV**

- **简介: 该论文是一篇综述，旨在解决事件相机在资源受限移动设备上高精度、低延迟感知的挑战。工作包括梳理2014–2025年文献，系统总结事件抽象、算法、软硬件加速及应用（如VO、跟踪、光流、3D重建），并指出未来方向与开源资源。**

- **链接: [https://arxiv.org/pdf/2503.22943v4](https://arxiv.org/pdf/2503.22943v4)**

> **作者:** Haoyang Wang; Ruishan Guo; Pengtao Ma; Ciyu Ruan; Xinyu Luo; Wenhua Ding; Tianyang Zhong; Jingao Xu; Yunhao Liu; Xinlei Chen
>
> **备注:** Accepted by ACM CSUR,35 pages
>
> **摘要:** With the increasing complexity of mobile device applications, these devices are evolving toward high agility. This shift imposes new demands on mobile sensing, particularly in achieving high-accuracy and low-latency. Event-based vision has emerged as a disruptive paradigm, offering high temporal resolution and low latency, making it well-suited for high-accuracy and low-latency sensing tasks on high-agility platforms. However, the presence of substantial noisy events, lack of stable, persistent semantic information, and large data volume pose challenges for event-based data processing on resource-constrained mobile devices. This paper surveys the literature from 2014 to 2025 and presents a comprehensive overview of event-based mobile sensing, encompassing its fundamental principles, event \textit{abstraction} methods, \textit{algorithm} advancements, and both hardware and software \textit{acceleration} strategies. We discuss key \textit{applications} of event cameras in mobile sensing, including visual odometry, object tracking, optical flow, and 3D reconstruction, while highlighting challenges associated with event data processing, sensor fusion, and real-time deployment. Furthermore, we outline future research directions, such as improving the event camera with advanced optics, leveraging neuromorphic computing for efficient processing, and integrating bio-inspired algorithms. To support ongoing research, we provide an open-source \textit{Online Sheet} with recent developments. We hope this survey serves as a reference, facilitating the adoption of event-based vision across diverse applications.
>
---
#### [replaced 004] Safety with Agency: Human-Centered Safety Filter with Application to AI-Assisted Motorsports
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出人类中心安全滤波器（HCSF），用于共享自主系统（如AI辅助赛车），解决安全与人类代理权难以兼顾的问题。通过学习神经安全价值函数，构建无需系统动力学知识的Q-CBF安全约束，实现平滑、最小干预式安全防护，并在高保真赛车模拟中验证其有效性。**

- **链接: [https://arxiv.org/pdf/2504.11717v4](https://arxiv.org/pdf/2504.11717v4)**

> **作者:** Donggeon David Oh; Justin Lidard; Haimin Hu; Himani Sinhmar; Elle Lazarski; Deepak Gopinath; Emily S. Sumner; Jonathan A. DeCastro; Guy Rosman; Naomi Ehrich Leonard; Jaime Fernández Fisac
>
> **备注:** Accepted to Robotics: Science and Systems (R:SS) 2025, 22 pages, 16 figures, 7 tables Updates for v4: typos in Appendix Subsection A revised
>
> **摘要:** We propose a human-centered safety filter (HCSF) for shared autonomy that significantly enhances system safety without compromising human agency. Our HCSF is built on a neural safety value function, which we first learn scalably through black-box interactions and then use at deployment to enforce a novel state-action control barrier function (Q-CBF) safety constraint. Since this Q-CBF safety filter does not require any knowledge of the system dynamics for both synthesis and runtime safety monitoring and intervention, our method applies readily to complex, black-box shared autonomy systems. Notably, our HCSF's CBF-based interventions modify the human's actions minimally and smoothly, avoiding the abrupt, last-moment corrections delivered by many conventional safety filters. We validate our approach in a comprehensive in-person user study using Assetto Corsa-a high-fidelity car racing simulator with black-box dynamics-to assess robustness in "driving on the edge" scenarios. We compare both trajectory data and drivers' perceptions of our HCSF assistance against unassisted driving and a conventional safety filter. Experimental results show that 1) compared to having no assistance, our HCSF improves both safety and user satisfaction without compromising human agency or comfort, and 2) relative to a conventional safety filter, our proposed HCSF boosts human agency, comfort, and satisfaction while maintaining robustness.
>
---
#### [replaced 005] Dexterous Manipulation through Imitation Learning: A Survey
- **分类: cs.RO; cs.LG**

- **简介: 该论文是一篇综述，聚焦于基于模仿学习的灵巧操作任务。旨在解决传统模型方法泛化差、强化学习样本效率低的问题。工作包括梳理IL在灵巧操作中的方法、进展、挑战及未来方向。**

- **链接: [https://arxiv.org/pdf/2504.03515v5](https://arxiv.org/pdf/2504.03515v5)**

> **作者:** Shan An; Ziyu Meng; Chao Tang; Yuning Zhou; Tengyu Liu; Fangqiang Ding; Shufang Zhang; Yao Mu; Ran Song; Wei Zhang; Zeng-Guang Hou; Hong Zhang
>
> **备注:** 32pages, 6 figures, 9 tables
>
> **摘要:** Dexterous manipulation, which refers to the ability of a robotic hand or multi-fingered end-effector to skillfully control, reorient, and manipulate objects through precise, coordinated finger movements and adaptive force modulation, enables complex interactions similar to human hand dexterity. With recent advances in robotics and machine learning, there is a growing demand for these systems to operate in complex and unstructured environments. Traditional model-based approaches struggle to generalize across tasks and object variations due to the high dimensionality and complex contact dynamics of dexterous manipulation. Although model-free methods such as reinforcement learning (RL) show promise, they require extensive training, large-scale interaction data, and carefully designed rewards for stability and effectiveness. Imitation learning (IL) offers an alternative by allowing robots to acquire dexterous manipulation skills directly from expert demonstrations, capturing fine-grained coordination and contact dynamics while bypassing the need for explicit modeling and large-scale trial-and-error. This survey provides an overview of dexterous manipulation methods based on imitation learning, details recent advances, and addresses key challenges in the field. Additionally, it explores potential research directions to enhance IL-driven dexterous manipulation. Our goal is to offer researchers and practitioners a comprehensive introduction to this rapidly evolving domain.
>
---
#### [replaced 006] Fast and Continual Learning for Hybrid Control Policies using Generalized Benders Decomposition
- **分类: cs.RO**

- **简介: 该论文面向混合整数模型预测控制（MPC）实时求解任务，旨在解决其因组合复杂性导致的求解速度不足问题。提出基于广义Benders分解（GBD）的在线学习求解器：通过在线枚举、缓存稀疏可行性割平面实现快速暖启动，并设计高效主问题算法，在少数据下逼近Gurobi性能。**

- **链接: [https://arxiv.org/pdf/2401.00917v3](https://arxiv.org/pdf/2401.00917v3)**

> **作者:** Xuan Lin
>
> **备注:** This paper has been withdrawn by the author. It has been superseded by a significantly updated version available at arXiv:2406.00780
>
> **摘要:** Hybrid model predictive control with both continuous and discrete variables is widely applicable to robotic control tasks, especially those involving contact with the environment. Due to the combinatorial complexity, the solving speed of hybrid MPC can be insufficient for real-time applications. In this paper, we proposed a hybrid MPC solver based on Generalized Benders Decomposition (GBD). The algorithm enumerates and stores cutting planes online inside a finite buffer. After a short cold-start phase, the stored cuts provide warm-starts for the new problem instances to enhance the solving speed. Despite the disturbance and randomly changing environment, the solving speed maintains. Leveraging on the sparsity of feasibility cuts, we also propose a fast algorithm for Benders master problems. Our solver is validated through controlling a cart-pole system with randomly moving soft contact walls, and a free-flying robot navigating around obstacles. The results show that with significantly less data than previous works, the solver reaches competitive speeds to the off-the-shelf solver Gurobi despite the Python overhead.
>
---
#### [replaced 007] Supervisory Measurement-Guided Noise Covariance Estimation
- **分类: cs.RO**

- **简介: 该论文属状态估计任务，旨在解决传感器噪声协方差难以准确标定的问题。提出一种监督测量引导的双层优化方法：下层用增广不变EKF估计轨迹，上层基于贝叶斯联合似然分解并行优化协方差，提升效率与精度。**

- **链接: [https://arxiv.org/pdf/2510.24508v2](https://arxiv.org/pdf/2510.24508v2)**

> **作者:** Haoying Li; Yifan Peng; Xinghan Li; Junfeng Wu
>
> **摘要:** Reliable state estimation hinges on accurate specification of sensor noise covariances, which weigh heterogeneous measurements. In practice, these covariances are difficult to identify due to environmental variability, front-end preprocessing, and other reasons. We address this by formulating noise covariance estimation as a bilevel optimization that, from a Bayesian perspective, factorizes the joint likelihood of so-called odometry and supervisory measurements, thereby balancing information utilization with computational efficiency. The factorization converts the nested Bayesian dependency into a chain structure, enabling efficient parallel computation: at the lower level, an invariant extended Kalman filter with state augmentation estimates trajectories, while a derivative filter computes analytical gradients in parallel for upper-level gradient updates. The upper level refines the covariance to guide the lower-level estimation. Experiments on synthetic and real-world datasets show that our method achieves higher efficiency over existing baselines.
>
---
#### [replaced 008] Integration of UWB Radar on Mobile Robots for Continuous Obstacle and Environment Mapping
- **分类: cs.RO**

- **简介: 该论文属机器人环境感知任务，旨在解决弱光、烟雾等视觉失效场景下的障碍物检测与建图问题。提出基于移动机器人搭载UWB雷达的无基础设施建图方法，设计CIR分析、目标识别、滤波与聚类三步处理流程，实现高精度障碍物检测与映射。**

- **链接: [https://arxiv.org/pdf/2512.01018v2](https://arxiv.org/pdf/2512.01018v2)**

> **作者:** Adelina Giurea; Stijn Luchie; Dieter Coppens; Jeroen Hoebeke; Eli De Poorter
>
> **备注:** This paper has been submitted to IEEE Access Journal and is currently undergoing review
>
> **摘要:** This paper presents an infrastructure-free approach for obstacle detection and environmental mapping using ultra-wideband (UWB) radar mounted on a mobile robotic platform. Traditional sensing modalities such as visual cameras and Light Detection and Ranging (LiDAR) fail in environments with poor visibility due to darkness, smoke, or reflective surfaces. In these visioned-impaired conditions, UWB radar offers a promising alternative. To this end, this work explores the suitability of robot-mounted UWB radar for environmental mapping in dynamic, anchor-free scenarios. The study investigates how different materials (metal, concrete and plywood) and UWB radio channels (5 and 9) influence the Channel Impulse Response (CIR). Furthermore, a processing pipeline is proposed to achieve reliable mapping of detected obstacles, consisting of 3 steps: (i) target identification (based on CIR peak detection), (ii) filtering (based on peak properties, signal-to-noise score, and phase-difference of arrival), and (iii) clustering (based on distance estimation and angle-of-arrival estimation). The proposed approach successfully reduces noise and multipath effects, resulting in an obstacle detection precision of at least 90.71% and a recall of 88.40% on channel 9 even when detecting low-reflective materials such as concrete. This work offers a foundation for further development of UWB-based localisation and mapping (SLAM) systems that do not rely on visual features and, unlike conventional UWB localisation systems, do not require on fixed anchor nodes for triangulation.
>
---
#### [replaced 009] Context Representation via Action-Free Transformer encoder-decoder for Meta Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属元强化学习任务，旨在解决上下文自适应元RL中任务推断依赖动作、导致策略耦合的问题。提出CRAFT模型，仅用状态-奖励序列推断任务表征，解耦任务推断与策略优化，提升泛化与适应效率。**

- **链接: [https://arxiv.org/pdf/2512.14057v2](https://arxiv.org/pdf/2512.14057v2)**

> **作者:** Amir M. Soufi Enayati; Homayoun Honari; Homayoun Najjaran
>
> **摘要:** Reinforcement learning (RL) enables robots to operate in uncertain environments, but standard approaches often struggle with poor generalization to unseen tasks. Context-adaptive meta reinforcement learning addresses these limitations by conditioning on the task representation, yet they mostly rely on complete action information in the experience making task inference tightly coupled to a specific policy. This paper introduces Context Representation via Action Free Transformer encoder decoder (CRAFT), a belief model that infers task representations solely from sequences of states and rewards. By removing the dependence on actions, CRAFT decouples task inference from policy optimization, supports modular training, and leverages amortized variational inference for scalable belief updates. Built on a transformer encoder decoder with rotary positional embeddings, the model captures long range temporal dependencies and robustly encodes both parametric and non-parametric task variations. Experiments on the MetaWorld ML-10 robotic manipulation benchmark show that CRAFT achieves faster adaptation, improved generalization, and more effective exploration compared to context adaptive meta--RL baselines. These findings highlight the potential of action-free inference as a foundation for scalable RL in robotic control.
>
---
#### [replaced 010] Off-Road Navigation via Implicit Neural Representation of Terrain Traversability
- **分类: cs.RO**

- **简介: 该论文属于自主越野导航任务，旨在解决传统采样式规划器视野短、无法联合优化路径几何与速度的问题。提出TRAIL框架，用隐式神经表征建模地形可通行性，并结合梯度优化方法联合调整轨迹形状与速度剖面。**

- **链接: [https://arxiv.org/pdf/2511.18183v2](https://arxiv.org/pdf/2511.18183v2)**

> **作者:** Yixuan Jia; Qingyuan Li; Jonathan P. How
>
> **备注:** 9 pages
>
> **摘要:** Autonomous off-road navigation requires robots to estimate terrain traversability from onboard sensors and plan accordingly. Conventional approaches typically rely on sampling-based planners such as MPPI to generate short-term control actions that aim to minimize traversal time and risk measures derived from the traversability estimates. These planners can react quickly but optimize only over a short look-ahead window, limiting their ability to reason about the full path geometry, which is important for navigating in challenging off-road environments. Moreover, they lack the ability to adjust speed based on the terrain bumpiness, which is important for smooth navigation on challenging terrains. In this paper, we introduce TRAIL (Traversability with an Implicit Learned Representation), an off-road navigation framework that leverages an implicit neural representation to continuously parameterize terrain properties. This representation yields spatial gradients that enable integration with a novel gradient-based trajectory optimization method that adapts the path geometry and speed profile based on terrain traversability.
>
---
#### [replaced 011] Demonstration Sidetracks: Categorizing Systematic Non-Optimality in Human Demonstrations
- **分类: cs.RO; cs.LG**

- **简介: 该论文属人机交互与机器人学习交叉任务，旨在解决LfD中人类示范非最优行为被误视为随机噪声的问题。作者通过40人实验识别出四类系统性“示范旁轨”（如探索、失误等），分析其时空规律及接口影响，并开源全部数据。**

- **链接: [https://arxiv.org/pdf/2506.11262v2](https://arxiv.org/pdf/2506.11262v2)**

> **作者:** Shijie Fang; Hang Yu; Qidi Fang; Reuben M. Aronson; Elaine S. Short
>
> **摘要:** Learning from Demonstration (LfD) is a popular approach for robots to acquire new skills, but most LfD methods suffer from imperfections in human demonstrations. Prior work typically treats these suboptimalities as random noise. In this paper we study non-optimal behaviors in non-expert demonstrations and show that they are systematic, forming what we call demonstration sidetracks. Using a public space study with 40 participants performing a long-horizon robot task, we recreated the setup in simulation and annotated all demonstrations. We identify four types of sidetracks (Exploration, Mistake, Alignment, Pause) and one control pattern (one-dimension control). Sidetracks appear frequently across participants, and their temporal and spatial distribution is tied to task context. We also find that users' control patterns depend on the control interface. These insights point to the need for better models of suboptimal demonstrations to improve LfD algorithms and bridge the gap between lab training and real-world deployment. All demonstrations, infrastructure, and annotations are available at https://github.com/AABL-Lab/Human-Demonstration-Sidetracks.
>
---
#### [replaced 012] SignBot: Learning Human-to-Humanoid Sign Language Interaction
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出SignBot框架，解决听障人士与人形机器人间的自然手语交互问题。工作包括：手语动作重定向、基于学习的运动控制、生成式交互（翻译/响应/生成），支持多机器人与数据集，在仿真和真实场景验证有效性。**

- **链接: [https://arxiv.org/pdf/2505.24266v3](https://arxiv.org/pdf/2505.24266v3)**

> **作者:** Guanren Qiao; Sixu Lin; Ronglai Zuo; Zhizheng Wu; Kui Jia; Guiliang Liu
>
> **摘要:** Sign language is a natural and visual form of language that uses movements and expressions to convey meaning, serving as a crucial means of communication for individuals who are deaf or hard-of-hearing (DHH). However, the number of people proficient in sign language remains limited, highlighting the need for technological advancements to bridge communication gaps and foster interactions with minorities. Based on recent advancements in embodied humanoid robots, we propose SignBot, a novel framework for human-robot sign language interaction. SignBot integrates a cerebellum-inspired motion control component and a cerebral-oriented module for comprehension and interaction. Specifically, SignBot consists of: 1) Motion Retargeting, which converts human sign language datasets into robot-compatible kinematics; 2) Motion Control, which leverages a learning-based paradigm to develop a robust humanoid control policy for tracking sign language gestures; and 3) Generative Interaction, which incorporates translator, responser, and generator of sign language, thereby enabling natural and effective communication between robots and humans. Simulation and real-world experimental results demonstrate that SignBot can effectively facilitate human-robot interaction and perform sign language motions with diverse robots and datasets. SignBot represents a significant advancement in automatic sign language interaction on embodied humanoid robot platforms, providing a promising solution to improve communication accessibility for the DHH community.
>
---
#### [replaced 013] New Location Science Models with Applications to UAV-Based Disaster Relief
- **分类: cs.RO; math.OC**

- **简介: 该论文属灾害应急响应任务，旨在解决灾后通信中断下无人机（UAV）快速定位与高效作业问题。提出融合风场影响与UAV异构性的新型位置优化模型——SFT问题，推广经典Sylvester问题，显著提升救援效率。**

- **链接: [https://arxiv.org/pdf/2510.15229v2](https://arxiv.org/pdf/2510.15229v2)**

> **作者:** Sina Kazemdehbashi; Yanchao Liu; Boris S. Mordukhovich
>
> **摘要:** Natural and human-made disasters can cause severe devastation and claim thousands of lives worldwide. Therefore, developing efficient methods for disaster response and management is a critical task for relief teams. One of the most essential components of effective response is the rapid collection of information about affected areas, damages, and victims. More data translates into better coordination, faster rescue operations, and ultimately, more lives saved. However, in some disasters, such as earthquakes, the communication infrastructure is often partially or completely destroyed, making it extremely difficult for victims to send distress signals and for rescue teams to locate and assist them in time. Unmanned Aerial Vehicles (UAVs) have emerged as valuable tools in such scenarios. In particular, a fleet of UAVs can be dispatched from a mobile station to the affected area to facilitate data collection and establish temporary communication networks. Nevertheless, real-world deployment of UAVs faces several challenges, with adverse weather conditions--especially wind--being among the most significant. To address this, we develop a novel mathematical framework to determine the optimal location of a mobile UAV station while explicitly accounting for the heterogeneity of the UAVs and the effect of wind. In particular, we generalize the Sylvester problem to introduce the Sylvester-Fermat-Torricelli (SFT) problem, which captures complex factors such as wind influence, UAV heterogeneity, and back-and-forth motion within a unified framework. The proposed framework enhances the practicality of UAV-based disaster response planning by accounting for real-world factors such as wind and UAV heterogeneity. Experimental results demonstrate that it can reduce wasted operational time by up to 84%, making post-disaster missions significantly more efficient and effective.
>
---
#### [replaced 014] CaFe-TeleVision: A Coarse-to-Fine Teleoperation System with Immersive Situated Visualization for Enhanced Ergonomics
- **分类: cs.RO**

- **简介: 该论文提出CaFe-TeleVision系统，面向远程机器人操作任务，解决现有系统效率低、人机工效差、多视角认知负荷高等问题。工作包括：设计粗粒度到细粒度的协同控制机制以平衡效率与人体工学，并引入按需情境化可视化降低视觉认知负担。在双臂操作任务中验证了其显著提升成功率、速度与用户舒适度。**

- **链接: [https://arxiv.org/pdf/2512.14270v2](https://arxiv.org/pdf/2512.14270v2)**

> **作者:** Zixin Tang; Yiming Chen; Quentin Rouxel; Dianxi Li; Shuang Wu; Fei Chen
>
> **备注:** Project webpage: https://clover-cuhk.github.io/cafe_television/ Code: https://github.com/Zixin-Tang/CaFe-TeleVision
>
> **摘要:** Teleoperation presents a promising paradigm for remote control and robot proprioceptive data collection. Despite recent progress, current teleoperation systems still suffer from limitations in efficiency and ergonomics, particularly in challenging scenarios. In this paper, we propose CaFe-TeleVision, a coarse-to-fine teleoperation system with immersive situated visualization for enhanced ergonomics. At its core, a coarse-to-fine control mechanism is proposed in the retargeting module to bridge workspace disparities, jointly optimizing efficiency and physical ergonomics. To stream immersive feedback with adequate visual cues for human vision systems, an on-demand situated visualization technique is integrated in the perception module, which reduces the cognitive load for multi-view processing. The system is built on a humanoid collaborative robot and validated with six challenging bimanual manipulation tasks. User study among 24 participants confirms that CaFe-TeleVision enhances ergonomics with statistical significance, indicating a lower task load and a higher user acceptance during teleoperation. Quantitative results also validate the superior performance of our system across six tasks, surpassing comparative methods by up to 28.89% in success rate and accelerating by 26.81% in completion time. Project webpage: https://clover-cuhk.github.io/cafe_television/
>
---
#### [replaced 015] History-Enhanced Two-Stage Transformer for Aerial Vision-and-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向空中视觉-语言导航（AVLN）任务，解决无人机在大尺度城市中依语言指令准确定位目标时，难以兼顾全局环境推理与局部场景理解的问题。提出历史增强的两阶段Transformer（HETT），通过粗粒度定位与细粒度动作优化的级联框架，并引入历史网格地图增强空间记忆，显著提升导航性能。**

- **链接: [https://arxiv.org/pdf/2512.14222v2](https://arxiv.org/pdf/2512.14222v2)**

> **作者:** Xichen Ding; Jianzhe Gao; Cong Pan; Wenguan Wang; Jie Qin
>
> **摘要:** Aerial Vision-and-Language Navigation (AVLN) requires Unmanned Aerial Vehicle (UAV) agents to localize targets in large-scale urban environments based on linguistic instructions. While successful navigation demands both global environmental reasoning and local scene comprehension, existing UAV agents typically adopt mono-granularity frameworks that struggle to balance these two aspects. To address this limitation, this work proposes a History-Enhanced Two-Stage Transformer (HETT) framework, which integrates the two aspects through a coarse-to-fine navigation pipeline. Specifically, HETT first predicts coarse-grained target positions by fusing spatial landmarks and historical context, then refines actions via fine-grained visual analysis. In addition, a historical grid map is designed to dynamically aggregate visual features into a structured spatial memory, enhancing comprehensive scene awareness. Additionally, the CityNav dataset annotations are manually refined to enhance data quality. Experiments on the refined CityNav dataset show that HETT delivers significant performance gains, while extensive ablation studies further verify the effectiveness of each component.
>
---
#### [replaced 016] ProbeMDE: Uncertainty-Guided Active Proprioception for Monocular Depth Estimation in Surgical Robotics
- **分类: cs.RO**

- **简介: 该论文属单目深度估计任务，旨在解决手术场景中因无纹理、镜面反射等导致的深度预测不准与高不确定性问题。提出ProbeMDE框架：利用模型集成量化不确定性，结合SVGD主动选择最优触点，融合稀疏本体感知测量提升精度，减少测量次数。**

- **链接: [https://arxiv.org/pdf/2512.11773v2](https://arxiv.org/pdf/2512.11773v2)**

> **作者:** Britton Jordan; Jordan Thompson; Jesse F. d'Almeida; Hao Li; Nithesh Kumar; Susheela Sharma Stern; Ipek Oguz; Robert J. Webster; Daniel Brown; Alan Kuntz; James Ferguson
>
> **备注:** 9 pages, 5 figures. Project page: https://brittonjordan.github.io/probe_mde/
>
> **摘要:** Monocular depth estimation (MDE) provides a useful tool for robotic perception, but its predictions are often uncertain and inaccurate in challenging environments such as surgical scenes where textureless surfaces, specular reflections, and occlusions are common. To address this, we propose ProbeMDE, a cost-aware active sensing framework that combines RGB images with sparse proprioceptive measurements for MDE. Our approach utilizes an ensemble of MDE models to predict dense depth maps conditioned on both RGB images and on a sparse set of known depth measurements obtained via proprioception, where the robot has touched the environment in a known configuration. We quantify predictive uncertainty via the ensemble's variance and measure the gradient of the uncertainty with respect to candidate measurement locations. To prevent mode collapse while selecting maximally informative locations to propriocept (touch), we leverage Stein Variational Gradient Descent (SVGD) over this gradient map. We validate our method in both simulated and physical experiments on central airway obstruction surgical phantoms. Our results demonstrate that our approach outperforms baseline methods across standard depth estimation metrics, achieving higher accuracy while minimizing the number of required proprioceptive measurements. Project page: https://brittonjordan.github.io/probe_mde/
>
---
