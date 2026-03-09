# 机器人 cs.RO

- **最新发布 72 篇**

- **更新 52 篇**

## 最新发布

#### [new 001] DexEMG: Towards Dexterous Teleoperation System via EMG2Pose Generalization
- **分类: cs.RO**

- **简介: 该论文属于机器人遥操作任务，旨在解决传统系统在性能与便携性间的矛盾。通过sEMG信号实现手部动作的精准控制与泛化，提出DexEMG系统提升机器人操作的灵活性和适用性。**

- **链接: [https://arxiv.org/pdf/2603.05861](https://arxiv.org/pdf/2603.05861)**

> **作者:** Qianyou Zhao; Wenqiao Li; Chiyu Wang; Kaifeng Zhang
>
> **摘要:** High-fidelity teleoperation of dexterous robotic hands is essential for bringing robots into unstructured domestic environments. However, existing teleoperation systems often face a trade-off between performance and portability: vision-based capture systems are constrained by costs and line-of-sight requirements, while mechanical exoskeletons are bulky and physically restrictive. In this paper, we present DexEMG, a lightweight and cost-effective teleoperation system leveraging surface electromyography (sEMG) to bridge the gap between human intent and robotic execution. We first collect a synchronized dataset of sEMG signals and hand poses via a MoCap glove to train EMG2Pose, a neural network capable of continuously predicting hand kinematics directly from muscle activity. To ensure seamless control, we develop a robust hand retargeting algorithm that maps the predicted poses onto a multi-fingered dexterous hand in real-time. Experimental results demonstrate that DexEMG achieves high precision in diverse teleoperation tasks. Notably, our system exhibits strong generalization capabilities across novel objects and complex environments without the need for intensive individual-specific recalibration. This work offers a scalable and intuitive interface for both general-purpose robotic manipulation and assistive technologies.
>
---
#### [new 002] MagRobot:An Open Simulator for Magnetically Navigated Robots
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人领域，旨在解决磁导航系统设计依赖实验、缺乏统一测试环境的问题。提出一个开源模拟平台，支持磁驱动与跟踪系统的仿真与优化。**

- **链接: [https://arxiv.org/pdf/2603.05992](https://arxiv.org/pdf/2603.05992)**

> **作者:** Heng Wang; Haoyu Song; Jiatao Zheng; Yuxiang Han; Kunli Wang
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** Magnetic navigation systems, including magnetic tracking systems and magnetic actuation systems, have shown great potential for occlusion-free localization and remote control of intracorporeal medical devices and robots in minimally invasive medicine, such as capsule endoscopy and cardiovascular intervention. However, the design of magnetically navigated robots remains heavily reliant on experimental prototyping, which is time-consuming and costly. Furthermore, there is a lack of a consistent experimental environment to compare and benchmark the hardware and algorithms across different magnetic navigation systems. To address these challenges, we propose the first universal open-source simulation platform to facilitate research, design and benchmarking of magnetically navigated robots. Our simulator features an intuitive graphical user interface that enables the user to efficiently design, visualize, and analyze magnetic navigation systems for both rigid and soft robots. The proposed simulator is versatile, which can simulate both magnetic actuation and magnetic tracking tasks in diverse medical applications that involve deformable anatomies. The proposed simulator provides an open development environment, where the user can load third-party anatomical models and customize both hardware and algorithms of magnetic navigation systems. The fidelity of the simulator is validated using both phantom and ex vivo experiments of magnetic navigation of a continuum robot and a capsule robot with diverse magnetic actuation setups. Three use cases of the simulator, i.e., bronchoscopy, endovascular intervention, and gastrointestinal endoscopy, are implemented to demonstrate the functionality of the simulator. It is shown that the configuration and algorithms of magnetic navigation systems can be flexibly designed and optimized for better performance using the simulator.
>
---
#### [new 003] PRISM: Personalized Refinement of Imitation Skills for Manipulation via Human Instructions
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出PRISM，用于机器人操作中通过人类指令优化模仿策略。解决如何将模仿学习与强化学习结合，提升策略的适应性和数据效率。通过迭代生成奖励函数和引入人类反馈实现策略 refinement。**

- **链接: [https://arxiv.org/pdf/2603.05574](https://arxiv.org/pdf/2603.05574)**

> **作者:** Arnau Boix-Granell; Alberto San-Miguel-Tello; Magí Dalmau-Moreno; Néstor García
>
> **备注:** 10 pages, 3 figures, Accepted for publication at European Robotics Forum 2026
>
> **摘要:** This paper presents PRISM: an instruction-conditioned refinement method for imitation policies in robotic manipulation. This approach bridges Imitation Learning (IL) and Reinforcement Learning (RL) frameworks into a seamless pipeline, such that an imitation policy on a broad generic task, generated from a set of user-guided demonstrations, can be refined through reinforcement to generate new unseen fine-grain behaviours. The refinement process follows the Eureka paradigm, where reward functions for RL are iteratively generated from an initial natural-language task description. Presented approach, builds on top of this mechanism to adapt a refined IL policy of a generic task to new goal configurations and the introduction of constraints by adding also human feedback correction on intermediate rollouts, enabling policy reusability and therefore data efficiency. Results for a pick-and-place task in a simulated scenario show that proposed method outperforms policies without human feedback, improving robustness on deployment and reducing computational burden.
>
---
#### [new 004] Towards Robotic Lake Maintenance: Integrating SONAR and Satellite Data to Assist Human Operators
- **分类: cs.RO**

- **简介: 该论文属于水域生态维护任务，旨在解决人工湖中水生植物过度生长的问题。通过整合卫星与声呐数据，实现精准定位与高效收割。**

- **链接: [https://arxiv.org/pdf/2603.06266](https://arxiv.org/pdf/2603.06266)**

> **作者:** Ahmed H. Elsayed; Christoph Manss; Tarek A. El-Mihoub; Andrej Lejman; Frederic Stahl
>
> **备注:** Accepted to and presented at the 2026 IEEE International Conference on Mechatronics and Robotics Engineering (ICMRE)
>
> **摘要:** Artificial Water Bodies (AWBs) are human-made systems that require continuous monitoring due to their artificial biological processes. These systems demand regular maintenance to manage their ecosystems effectively. As a result of these artificial conditions, underwater vegetation can grow rapidly and must be harvested to preserve the ecological balance. This paper proposes a two-step approach to support targeted weed harvesting for the maintenance of artificial lakes. The first step is the initial detection of Submerged Aquatic Vegetation (SAV), also referred to in this paper as areas of interest, is performed using satellite-derived indices, specifically the Aquatic Plants and Algae (APA) index, which highlights submerged vegetation in water bodies. Subsequently, an Unmanned Surface Vehicle (USV) equipped with multibeam SOund NAvigation and Ranging (SONAR) performs high-resolution bathymetric mapping to locate and quantify aquatic vegetation precisely. This two-stage approach offers an effective human-robot collaboration, where satellite data guides the USV missions and boat skippers leverage detailed SONAR maps for targeted harvesting. This setup narrows the search space and reduces manual workload from human operators, making the harvesting process less labour-intensive for operators. Preliminary results demonstrate the feasibility of integrating satellite imagery and underwater acoustic sensing to improve vegetation management in artificial lakes.
>
---
#### [new 005] HarvestFlex: Strawberry Harvesting via Vision-Language-Action Policy Adaptation in the Wild
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于农业机器人任务，解决温室草莓采摘问题。通过视觉-语言-动作策略迁移，构建闭环系统，提升采摘成功率与效率。**

- **链接: [https://arxiv.org/pdf/2603.05982](https://arxiv.org/pdf/2603.05982)**

> **作者:** Ziyang Zhao; Shuheng Wang; Zhonghua Miao; Ya Xiong
>
> **摘要:** This work presents the first study on transferring vision-language-action (VLA) policies to real greenhouse tabletop strawberry harvesting, a long-horizon, unstructured task challenged by occlusion and specular reflections. We built an end-to-end closed-loop system on the HarvestFlex platform using three-view RGB sensing (two fixed scene views plus a wrist-mounted view) and intentionally avoided depth clouds and explicit geometric calibration. We collected 3.71 h of VR teleoperated demonstrations (227 episodes) and fine-tuned pi_0, pi_0.5, and WALL-OSS with full fine-tuning and LoRA. Under a unified 50 trials real-greenhouse protocol and metrics spanning completion, pi_0.5 with full fine-tuning achieved success rate of 74.0% with 32.6 s/pick and damage rate of 4.1%. Asynchronous inference-control decoupling further improved performance over synchronous deployment. Results showed non-trivial closed-loop picking with fewer than four hours of real data, while remaining limited by close-range observability loss and contact-dynamics mismatch. A demonstration video is available at: this https URL.
>
---
#### [new 006] Data Analogies Enable Efficient Cross-Embodiment Transfer
- **分类: cs.RO**

- **简介: 该论文研究跨机器人配置的迁移学习任务，旨在提升泛化机器人策略的性能。通过分析不同形式的演示数据，发现数据类比能有效促进迁移，优于单纯增加数据量。**

- **链接: [https://arxiv.org/pdf/2603.06450](https://arxiv.org/pdf/2603.06450)**

> **作者:** Jonathan Yang; Chelsea Finn; Dorsa Sadigh
>
> **备注:** 14 pages, 11 Figures, 6 Tables
>
> **摘要:** Generalist robot policies are trained on demonstrations collected across a wide variety of robots, scenes, and viewpoints. Yet it remains unclear how to best organize and scale such heterogeneous data so that it genuinely improves performance in a given target setting. In this work, we ask: what form of demonstration data is most useful for enabling transfer across robot set-ups? We conduct controlled experiments that vary end-effector morphology, robot platform appearance, and camera perspective, and compare the effects of simply scaling the number of demonstrations against systematically broadening the diversity in different ways. Our simulated experiments show that while perceptual shifts such as viewpoint benefit most from broad diversity, morphology shifts benefit far less from unstructured diversity and instead see the largest gains from data analogies, i.e. paired demonstrations that align scenes, tasks, and/or trajectories across different embodiments. Informed by the simulation results, we improve real-world cross-embodiment transfer success by an average of $22.5\%$ over large-scale, unpaired datasets by changing only the composition of the data.
>
---
#### [new 007] Unified Learning of Temporal Task Structure and Action Timing for Bimanual Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决双臂协作中时间结构与动作时序的统一学习问题。通过结合符号与非符号时间约束，生成精确时序的执行计划。**

- **链接: [https://arxiv.org/pdf/2603.06538](https://arxiv.org/pdf/2603.06538)**

> **作者:** Christian Dreher; Patrick Dormanns; Andre Meixner; Tamim Asfour
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Temporal task structure is fundamental for bimanual manipulation: a robot must not only know that one action precedes or overlaps another, but also when each action should occur and how long it should take. While symbolic temporal relations enable high-level reasoning about task structure and alternative execution sequences, concrete timing parameters are equally essential for coordinating two hands at the execution level. Existing approaches address these two levels in isolation, leaving a gap between high-level task planning and low-level movement synchronization. This work presents an approach for learning both symbolic and subsymbolic temporal task constraints from human demonstrations and deriving executable, temporally parametrized plans for bimanual manipulation. Our contributions are (i) a 3-dimensional representation of timings between two actions with methods based on multivariate Gaussian Mixture Models to represent temporal relationships between actions on a subsymbolic level, (ii) a method based on the Davis-Putnam-Logemann-Loveland (DPLL) algorithm that finds and ranks all contradiction-free assignments of Allen relations to action pairs, representing different modes of a task, and (iii) an optimization-based planning system that combines the identified symbolic and subsymbolic temporal task constraints to derive temporally parametrized plans for robot execution. We evaluate our approach on several datasets, demonstrating that our method generates temporally parametrized plans closer to human demonstrations than the most characteristic demonstration baseline.
>
---
#### [new 008] RACAS: Controlling Diverse Robots With a Single Agentic System
- **分类: cs.RO; cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出RACAS系统，解决跨平台机器人控制问题。通过自然语言通信的模块实现通用控制，无需修改代码或模型，验证了其在多种机器人上的有效性。**

- **链接: [https://arxiv.org/pdf/2603.05621](https://arxiv.org/pdf/2603.05621)**

> **作者:** Dylan R. Ashley; Jan Przepióra; Yimeng Chen; Ali Abualsaud; Nurzhan Yesmagambet; Shinkyu Park; Eric Feron; Jürgen Schmidhuber
>
> **备注:** 7 pages in main text + 1 page of appendices + 1 page of references, 5 figures in main text + 1 figure in appendices, 2 tables in main text
>
> **摘要:** Many robotic platforms expose an API through which external software can command their actuators and read their sensors. However, transitioning from these low-level interfaces to high-level autonomous behaviour requires a complicated pipeline, whose components demand distinct areas of expertise. Existing approaches to bridging this gap either require retraining for every new embodiment or have only been validated across structurally similar platforms. We introduce RACAS (Robot-Agnostic Control via Agentic Systems), a cooperative agentic architecture in which three LLM/VLM-based modules (Monitors, a Controller, and a Memory Curator) communicate exclusively through natural language to provide closed-loop robot control. RACAS requires only a natural language description of the robot, a definition of available actions, and a task specification; no source code, model weights, or reward functions need to be modified to move between platforms. We evaluate RACAS on several tasks using a wheeled ground robot, a recently published novel multi-jointed robotic limb, and an underwater vehicle. RACAS consistently solved all assigned tasks across these radically different platforms, demonstrating the potential of agentic AI to substantially reduce the barrier to prototyping robotic solutions.
>
---
#### [new 009] Improved hopping control on slopes for small robots using spring mass modeling
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人控制任务，旨在解决跳跃机器人在斜坡上失衡的问题。通过弹簧质量模型分析并提出调整着陆角度和施加修正力矩的方法，提升跳跃稳定性。**

- **链接: [https://arxiv.org/pdf/2603.05902](https://arxiv.org/pdf/2603.05902)**

> **作者:** Heston Roberts; Pronoy Sarker; Sm Ashikul Islam; Min Gyu Kim
>
> **摘要:** Hopping robots often lose balance on slopes because the tilted ground creates unwanted rotation at landing. This work analyzes that effect using a simple spring mass model and identifies how slope induced impulses destabilize the robot. To address this, we introduce two straightforward fixes, adjusting the bodys touchdown angle based on the slope and applying a small corrective torque before takeoff. Together, these steps effectively cancel the unwanted rotation caused by inclined terrain, allowing the robot to land smoothly and maintain stable hopping even on steep slopes. Moreover, the proposed method remains simple enough to implement on low cost robotic platforms without requiring complex sensing or computation. By combining this analytical model with minimal control actions, this approach provides a practical path toward reliable hopping on uneven terrain. The results from simulation confirm that even small slope aware adjustments can dramatically improve landing stability, making the technique suitable for future autonomous field robots that must navigate natural environments such as hills, rubble, and irregular outdoor landscapes.
>
---
#### [new 010] KISS-IMU: Self-supervised Inertial Odometry with Motion-balanced Learning and Uncertainty-aware Inference
- **分类: cs.RO**

- **简介: 该论文属于惯性里程计任务，解决传统方法依赖真实标签的问题。通过自监督学习，结合LiDAR进行姿态优化，提升IMU的鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.06205](https://arxiv.org/pdf/2603.06205)**

> **作者:** Jiwon Choi; Hogyun Kim; Geonmo Yang; Juhui Lee; Younggun Cho
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** Inertial measurement units (IMUs), which provide high-frequency linear acceleration and angular velocity measurements, serve as fundamental sensing modalities in robotic systems. Recent advances in deep neural networks have led to remarkable progress in inertial odometry. However, the heavy reliance on ground truth data during training fundamentally limits scalability and generalization to unseen and diverse environments. We propose KISS-IMU, a novel self-supervised inertial odometry framework that eliminates ground truth dependency by leveraging simple LiDAR-based ICP registration and pose graph optimization as a supervisory signal. Our approach embodies two key principles: keeping the IMU stable through motion-aware balanced training and keeping the IMU strong through uncertainty-driven adaptive weighting during inference. To evaluate performance across diverse motion patterns and scenarios, we conducted comprehensive experiments on various real-world platforms, including quadruped robots. Importantly, we train only the IMU network in a self-supervised manner, with LiDAR serving solely as a lightweight supervisory signal rather than requiring additional learnable processes. This design enables the framework to ensure robustness without relying on joint multi-modal learning or ground truth supervision. The supplementary materials are available at this https URL.
>
---
#### [new 011] Environment-Aware Path Generation for Robotic Additive Manufacturing of Structures
- **分类: cs.RO**

- **简介: 该论文属于机器人增材制造路径规划任务，旨在解决动态环境中在线生成路径的问题。提出环境感知路径生成框架，评估不同算法在复杂障碍下的性能。**

- **链接: [https://arxiv.org/pdf/2603.05748](https://arxiv.org/pdf/2603.05748)**

> **作者:** Mahsa Rabiei; Reza Moini
>
> **摘要:** Robotic Additive Manufacturing (AM) has emerged as a scalable and customizable construction method in the last decade. However, current AM design methods rely on pre-conceived (A priori) toolpath of the structure, often developed via offline slicing software. Moreover, considering the dynamic construction environments involving obstacles on terrestrial and extraterrestrial environments, there is a need for online path generation methods. Here, an environment-aware path generation framework (PGF) is proposed for the first time in which structures are designed in an online fashion by utilizing four path planning (PP) algorithms (two search-based and two sampling-based). To evaluate the performance of the proposed PGF in different obstacle arrangements (periodic, random) for two types of structures (closed and open), structural (path roughness, turns, offset, Root Mean Square Error (RMSE), deviation) and computational (run time) performance metrics are developed. Most challenging environments (i.e., dense with high number of obstacles) are considered to saturate the feasibility limits of PP algorithms. The capability of each of the four path planners used in the PGF in finding a feasible path is assessed. Finally, the effectiveness of the proposed structural performance metrics is evaluated individually and comparatively, and most essential metrics necessary for evaluation of toolpath of the resulting structures are prescribed. Consequently, the most promising path planners in challenging environments are identified for robotic additive manufacturing applications.
>
---
#### [new 012] Multi-Robot Trajectory Planning via Constrained Bayesian Optimization and Local Cost Map Learning with STL-Based Conflict Resolution
- **分类: cs.RO**

- **简介: 该论文属于多机器人路径规划任务，解决在STL约束下生成高效安全轨迹的问题。提出cBOT和STL-KCBS方法，结合贝叶斯优化与STL冲突解析，提升规划效率与适应性。**

- **链接: [https://arxiv.org/pdf/2603.05767](https://arxiv.org/pdf/2603.05767)**

> **作者:** Sourav Raxit; Abdullah Al Redwan Newaz; Jose Fuentes; Paulo Padrao; Ana Cavalcanti; Leonardo Bobadilla
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** We address multi-robot motion planning under Signal Temporal Logic (STL) specifications with kinodynamic constraints. Exact approaches face scalability bottlenecks and limited adaptability, while conventional sampling-based methods require excessive samples to construct optimal trajectories. We propose a two-stage framework integrating sampling-based online learning with formal STL reasoning. At the single-robot level, our constrained Bayesian Optimization-based Tree search (cBOT) planner uses a Gaussian process as a surrogate model to learn local cost maps and feasibility constraints, generating shorter collision-free trajectories with fewer samples. At the multi-robot level, our STL-enhanced Kinodynamic Conflict-Based Search (STL-KCBS) algorithm incorporates STL monitoring into conflict detection and resolution, ensuring specification satisfaction while maintaining scalability and probabilistic completeness. Benchmarking demonstrates improved trajectory efficiency and safety over existing methods. Real-world experiments with autonomous surface vehicles validate robustness and practical applicability in uncertain environments. The STLcBOT Planner will be released as an open-source package, and videos of real-world and simulated experiments are available at this https URL.
>
---
#### [new 013] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决现有方法感知效率低和推理不聚焦的问题。提出ProFocus框架，结合大语言模型和视觉语言模型，实现主动感知和聚焦推理。**

- **链接: [https://arxiv.org/pdf/2603.05530](https://arxiv.org/pdf/2603.05530)**

> **作者:** Wei Xue; Mingcheng Li; Xuecheng Wu; Jingqun Tang; Dingkang Yang; Lihua Zhang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Vision-and-Language Navigation (VLN) requires agents to accurately perceive complex visual environments and reason over navigation instructions and histories. However, existing methods passively process redundant visual inputs and treat all historical contexts indiscriminately, resulting in inefficient perception and unfocused reasoning. To address these challenges, we propose \textbf{ProFocus}, a training-free progressive framework that unifies \underline{Pro}active Perception and \underline{Focus}ed Reasoning through collaboration between large language models (LLMs) and vision-language models (VLMs). For proactive perception, ProFocus transforms panoramic observations into structured ego-centric semantic maps, enabling the orchestration agent to identify missing visual information needed for reliable decision-making, and to generate targeted visual queries with corresponding focus regions that guide the perception agent to acquire the required observations. For focused reasoning, we propose Branch-Diverse Monte Carlo Tree Search (BD-MCTS) to identify top-$k$ high-value waypoints from extensive historical candidates. The decision agent focuses reasoning on the historical contexts associated with these waypoints, rather than considering all historical waypoints equally. Extensive experiments validate the effectiveness of ProFocus, achieving state-of-the-art performance among zero-shot methods on R2R and REVERIE benchmarks.
>
---
#### [new 014] RODEO: RObotic DEcentralized Organization
- **分类: cs.RO**

- **简介: 该论文提出RODEO框架，解决机器人自主性与责任审计问题。通过区块链技术实现机器人任务协调与支付，确保操作可审计。**

- **链接: [https://arxiv.org/pdf/2603.06058](https://arxiv.org/pdf/2603.06058)**

> **作者:** Milan Groshev; Eduardo Castelló Ferrer
>
> **备注:** 8 pages, 6 figures, Accepted at IEEE International Conference on Robotics & Automation (2026)
>
> **摘要:** Robots are improving their autonomy with minimal human supervision. However, auditable actions, transparent decision processes, and new human-robot interaction models are still missing requirements to achieve extended robot autonomy. To tackle these challenges, we propose RODEO (RObotic DEcentralized Organization), a blockchain-based framework that integrates trust and accountability mechanisms for robots. This paper formalizes Decentralized Autonomous Organizations (DAOs) for service robots. First, it provides a ROS-ETH bridge between the DAO and the robots. Second, it offers templates that enable organizations (e.g., companies, universities) to integrate service robots into their operations. Third, it provides proof-verification mechanisms that allow robot actions to be auditable. In our experimental setup, a mobile robot was deployed as a trash collector in a lab scenario. The robot collects trash and uses a smart bin to sort and dispose of it correctly. Then, the robot submits a proof of the successful operation and is compensated in DAO tokens. Finally, the robot re-invests the acquired funds to purchase battery charging services. Data collected in a three day experiment show that the robot doubled its income and reinvested funds to extend its operating time. The proof validation times of approximately one minute ensured verifiable task execution, while the accumulated robot income successfully funded up to 88 hours of future autonomous operation. The results of this research give insights about how robots and organizations can coordinate tasks and payments with auditable execution proofs and on-chain settlement.
>
---
#### [new 015] AnyCamVLA: Zero-Shot Camera Adaptation for Viewpoint Robust Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉-语言-动作模型的适应任务，解决相机视角变化导致的性能下降问题。提出无需额外数据的零样本相机适配方法，提升模型在不同视角下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.05868](https://arxiv.org/pdf/2603.05868)**

> **作者:** Hyeongjun Heo; Seungyeon Woo; Sang Min Kim; Junho Kim; Junho Lee; Yonghyeon Lee; Young Min Kim
>
> **备注:** Under review, Project Page: this https URL
>
> **摘要:** Despite remarkable progress in Vision-Language-Action models (VLAs) for robot manipulation, these large pre-trained models require fine-tuning to be deployed in specific environments. These fine-tuned models are highly sensitive to camera viewpoint changes that frequently occur in unstructured environments. In this paper, we propose a zero-shot camera adaptation framework without additional demonstration data, policy fine-tuning, or architectural modification. Our key idea is to virtually adjust test-time camera observations to match the training camera configuration in real-time. For that, we use a recent feed-forward novel view synthesis model which outputs high-quality target view images, handling both extrinsic and intrinsic parameters. This plug-and-play approach preserves the pre-trained capabilities of VLAs and applies to any RGB-based policy. Through extensive experiments on the LIBERO benchmark, our method consistently outperforms baselines that use data augmentation for policy fine-tuning or additional 3D-aware features for visual input. We further validate that our approach constantly enhances viewpoint robustness in real-world robotic manipulation scenarios, including settings with varying camera extrinsics, intrinsics, and freely moving handheld cameras.
>
---
#### [new 016] Task-Level Decisions to Gait Level Control: A Hierarchical Policy Approach for Quadruped Navigation
- **分类: cs.RO**

- **简介: 该论文研究四足机器人导航任务，解决高/低层次决策不匹配及环境变化带来的不稳定问题。提出分层策略TDGC，实现任务决策与步态控制的高效协同。**

- **链接: [https://arxiv.org/pdf/2603.05783](https://arxiv.org/pdf/2603.05783)**

> **作者:** Sijia Li; Haoyu Wang; Shenghai Yuan; Yizhuo Yang; Thien-Minh Nguyen
>
> **备注:** Submitted to IROS 2026
>
> **摘要:** Real-world quadruped navigation is constrained by a scale mismatch between high-level navigation decisions and low-level gait execution, as well as by instabilities under out-of-distribution environmental changes. Such variations challenge sim-to-real transfer and can trigger falls when policies lack explicit interfaces for adaptation. In this paper, we present a hierarchical policy architecture for quadrupedal navigation, termed Task-level Decision to Gait Control (TDGC). A low-level policy, trained with reinforcement learning in simulation, delivers gait-conditioned locomotion and maps task requirements to a compact set of controllable behavior parameters, enabling robust mode generation and smooth switching. A high-level policy makes task-centric decisions from sparse semantic or geometric terrain cues and translates them into low-level targets, forming a traceable decision pipeline without dense maps or high-resolution terrain reconstruction. Different from end-to-end approaches, our architecture provides explicit interfaces for deployment-time tuning, fault diagnosis, and policy refinement. We introduce a structured curriculum with performance-driven progression that expands environmental difficulty and disturbance ranges. Experiments show higher task success rates on mixed terrains and out-of-distribution tests.
>
---
#### [new 017] Restoring Linguistic Grounding in VLA Models via Train-Free Attention Recalibration
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型研究，解决OOC指令下模型依赖视觉而非语言的问题。提出IGAR方法，在不重新训练的情况下提升语言指导的准确性。**

- **链接: [https://arxiv.org/pdf/2603.06001](https://arxiv.org/pdf/2603.06001)**

> **作者:** Ninghao Zhang; Bin Zhu; Shijie Zhou; Jingjing Chen
>
> **摘要:** Vision-Language-Action (VLA) models enable robots to perform manipulation tasks directly from natural language instructions and are increasingly viewed as a foundation for generalist robotic policies. However, their reliability under Out-of-Distribution (OOD) instructions remains underexplored. In this paper, we reveal a critical failure mode in which VLA policies continue executing visually plausible actions even when the language instruction contradicts the scene. We refer to this phenomenon as linguistic blindness, where VLA policies prioritize visual priors over instruction semantics during action generation. To systematically analyze this issue, we introduce ICBench, a diagnostic benchmark constructed from the LIBERO dataset that probes language-action coupling by injecting controlled OOD instruction contradictions while keeping the visual environment unchanged. Evaluations on three representative VLA architectures, including Pi0, Pi0.5 and OpenVLA OFT, show that these models frequently succeed at tasks despite logically impossible instructions, revealing a strong visual bias in action generation. To mitigate this issue, we propose Instruction-Guided Attention Recalibration (IGAR), a train-free inference-time mechanism that rebalances attention distributions to restore the influence of language instructions. IGAR operates without retraining or architectural modification and can be directly applied to existing VLA models. Experiments across 30 LIBERO tasks demonstrate that IGAR substantially reduces erroneous execution under OOD contradictory instructions while preserving baseline task performance. We additionally validate the approach on a real Franka robotic arm, where IGAR effectively prevents manipulation triggered by inconsistent instructions.
>
---
#### [new 018] Multimodal Behavior Tree Generation: A Small Vision-Language Model for Robot Task Planning
- **分类: cs.RO**

- **简介: 该论文属于机器人任务规划领域，解决视觉-语言模型与行为树生成结合的问题。通过构建数据集并微调模型，生成可执行的行为树，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.06084](https://arxiv.org/pdf/2603.06084)**

> **作者:** Cristiano Battistini; Riccardo Andrea Izzo; Gianluca Bardaro; Matteo Matteucci
>
> **摘要:** Large and small language models have been widely used for robotic task planning. At the same time, vision-language models (VLMs) have successfully tackled problems such as image captioning, scene understanding, and visual question answering. In this work, we combine these two approaches by deploying a compact, open-source multimodal model to generate behavior trees for robotic task planning. The main obstacle to achieving this goal is the lack of an existing dataset that links visual observations and instructions to executable behavior trees. We propose a method to construct such a dataset starting from existing robotic episodes (i.e., Open X-Embodiment), in which a large model serves as a teacher in a multi-stage generation pipeline. We use this dataset to fine-tune VLMs ranging from 500M to 4B parameters via parameter-efficient fine-tuning (PEFT). The generated behavior trees, compatible with the this http URL library, are evaluated both offline, using structural and lexical metrics, and online through the execution of household tasks in a state-of-the-art embodied simulator. Our results demonstrate that our fine-tuned 4B-parameter VLM approaches the performance of state-of-the-art closed-source models, achieving an 87\% success rate while requiring only a fraction of the computational resources.
>
---
#### [new 019] Contact-Grounded Policy: Dexterous Visuotactile Policy with Generative Contact Grounding
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决精细操作中的接触感知问题。通过预测机器人状态和触觉反馈，生成可执行的控制目标，提升操作精度与灵活性。**

- **链接: [https://arxiv.org/pdf/2603.05687](https://arxiv.org/pdf/2603.05687)**

> **作者:** Zhengtong Xu; Yeping Wang; Ben Abbatematteo; Jom Preechayasomboon; Sonny Chan; Nick Colonnese; Amirhossein H. Memar
>
> **摘要:** Contact-Grounded Policy (CGP) enables fine-grained, contact-rich dexterous manipulation by grounding multi-point contacts through predicting the actual robot state and tactile feedback, and by using a learned contact-consistency mapping to convert these predictions into controller-executable targets for a compliance controller. CGP supports both dense tactile arrays and vision-based tactile sensors mounted on the hand. We collect demonstrations via teleoperation in both simulation and on a physical robot, and evaluate CGP across multiple dexterous manipulation tasks.
>
---
#### [new 020] Expert Knowledge-driven Reinforcement Learning for Autonomous Racing via Trajectory Guidance and Dynamics Constraints
- **分类: cs.RO**

- **简介: 该论文属于自主赛车任务，旨在解决强化学习在高动态环境中的不稳定和不安全问题。通过引入专家知识和动态约束，提升车辆的性能与安全性。**

- **链接: [https://arxiv.org/pdf/2603.05842](https://arxiv.org/pdf/2603.05842)**

> **作者:** Bo Leng; Weiqi Zhang; Zhuoren Li; Lu Xiong; Guizhe Jin; Ran Yu; Chen Lv
>
> **摘要:** Reinforcement learning has demonstrated significant potential in the field of autonomous driving. However, it suffers from defects such as training instability and unsafe action outputs when faced with autonomous racing environments characterized by high dynamics and strong nonlinearities. To this end, this paper proposes a trajectory guidance and dynamics constraints Reinforcement Learning (TraD-RL) method for autonomous racing. The key features of this method are as follows: 1) leveraging the prior expert racing line to construct an augmented state representation and facilitate reward shaping, thereby integrating domain knowledge to stabilize early-stage policy learning; 2) embedding explicit vehicle dynamic priors into a safe operating envelope formulated via control barrier functions to enable safety-constrained learning; and 3) adopting a multi-stage curriculum learning strategy that shifts from expert-guided learning to autonomous exploration, allowing the learned policy to surpass expert-level performance. The proposed method is evaluated in a high-fidelity simulation environment modeled after the Tempelhof Airport Street Circuit. Experimental results demonstrate that TraD-RL effectively improves both lap speed and driving stability of the autonomous racing vehicle, achieving a synergistic optimization of racing performance and safety.
>
---
#### [new 021] EmboAlign: Aligning Video Generation with Compositional Constraints for Zero-Shot Manipulation
- **分类: cs.RO**

- **简介: 该论文提出\method{}，解决视频生成模型在零样本机器人操作中的物理合理性问题。通过结合视觉语言模型的结构化空间推理，提升操作任务的成功率。**

- **链接: [https://arxiv.org/pdf/2603.05757](https://arxiv.org/pdf/2603.05757)**

> **作者:** Gehao Zhang; Zhenyang Ni; Payal Mohapatra; Han Liu; Ruohan Zhang; Qi Zhu
>
> **摘要:** Video generative models (VGMs) pretrained on large-scale internet data can produce temporally coherent rollout videos that capture rich object dynamics, offering a compelling foundation for zero-shot robotic manipulation. However, VGMs often produce physically implausible rollouts, and converting their pixel-space motion into robot actions through geometric retargeting further introduces cumulative errors from imperfect depth estimation and keypoint tracking. To address these challenges, we present \method{}, a data-free framework that aligns VGM outputs with compositional constraints generated by vision-language models (VLMs) at inference time. The key insight is that VLMs offer a capability complementary to VGMs: structured spatial reasoning that can identify the physical constraints critical to the success and safety of manipulation execution. Given a language instruction, \method{} uses a VLM to automatically extract a set of compositional constraints capturing task-specific requirements, which are then applied at two stages: (1) constraint-guided rollout selection, which scores and filters a batch of VGM rollouts to retain the most physically plausible candidate, and (2) constraint-based trajectory optimization, which uses the selected rollout as initialization and refines the robot trajectory under the same constraint set to correct retargeting errors. We evaluate \method{} on six real-robot manipulation tasks requiring precise, constraint-sensitive execution, improving the overall success rate by 43.3\% points over the strongest baseline without any task-specific training data.
>
---
#### [new 022] Digital-Twin Losses for Lane-Compliant Trajectory Prediction at Urban Intersections
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于交通场景下的轨迹预测任务，旨在提升城市交叉口多智能体轨迹预测的安全性与合规性。通过结合数字孪生与多损失函数，优化预测模型以减少违规和碰撞风险。**

- **链接: [https://arxiv.org/pdf/2603.05546](https://arxiv.org/pdf/2603.05546)**

> **作者:** Kuo-Yi Chao; Erik Leo Haß; Melina Gegg; Jiajie Zhang; Ralph Raßhofer; Alois Christian Knoll
>
> **备注:** 7 pages, 2 figures, conference
>
> **摘要:** Accurate and safety-conscious trajectory prediction is a key technology for intelligent transportation systems, especially in V2X-enabled urban environments with complex multi-agent interactions. In this paper, we created a digital twin-driven V2X trajectory prediction pipeline that jointly leverages cooperative perception from vehicles and infrastructure to forecast multi-agent motion at signalized intersections. The proposed model combines a Bi-LSTM-based generator with a structured training objective consisting of a standard mean squared error (MSE) loss and a novel twin loss. The twin loss encodes infrastructure constraints, collision avoidance, diversity across predicted modes, and rule-based priors derived from the digital twin. While the MSE term ensures point-wise accuracy, the twin loss penalizes traffic rule violations, predicted collisions, and mode collapse, guiding the model toward scene-consistent and safety-compliant predictions. We train and evaluate our approach on real-world V2X data sent from the intersection to the vehicle and collected in urban corridors. In addition to standard trajectory metrics (ADE, FDE), we introduce ITS-relevant safety indicators, including infrastructure and rule violation rates. Experimental results demonstrate that the proposed training scheme significantly reduces critical violations while maintaining comparable prediction accuracy and real-time performance, highlighting the potential of digital twin-driven multi-loss learning for V2X-enabled intelligent transportation systems.
>
---
#### [new 023] Uncertainty-Aware Adaptive Dynamics For Underwater Vehicle-Manipulator Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于水下机器人动力学建模任务，解决水下车辆-机械臂系统中参数不确定性问题，提出一种自适应动态模型，实现精确在线估计与控制。**

- **链接: [https://arxiv.org/pdf/2603.06548](https://arxiv.org/pdf/2603.06548)**

> **作者:** Edward Morgan; Nenyi K Dadson; Corina Barbalata
>
> **摘要:** Accurate and adaptive dynamic models are critical for underwater vehicle-manipulator systems where hydrodynamic effects induce time-varying parameters. This paper introduces a novel uncertainty-aware adaptive dynamics model framework that remains linear in lumped vehicle and manipulator parameters, and embeds convex physical consistency constraints during online estimation. Moving horizon estimation is used to stack horizon regressors, enforce realizable inertia, damping, friction, and hydrostatics, and quantify uncertainty from parameter evolution. Experiments on a BlueROV2 Heavy with a 4-DOF manipulator demonstrate rapid convergence and calibrated predictions. Manipulator fits achieve R2 = 0.88 to 0.98 with slopes near unity, while vehicle surge, heave, and roll are reproduced with good fidelity under stronger coupling and noise. Median solver time is approximately 0.023 s per update, confirming online feasibility. A comparison against a fixed parameter model shows consistent reductions in MAE and RMSE across degrees of freedom. Results indicate physically plausible parameters and confidence intervals with near 100% coverage, enabling reliable feedforward control and simulation in underwater environments.
>
---
#### [new 024] Open-Source Based and ETSI Compliant Cooperative, Connected, and Automated Mini-Cars
- **分类: cs.RO; cs.NI**

- **简介: 本文提出一种1:10比例的协作自动驾驶微型车平台，用于测试自动化、联网和协作技术。旨在降低实际实验成本，解决仿真与现实之间的鸿沟问题。**

- **链接: [https://arxiv.org/pdf/2603.06343](https://arxiv.org/pdf/2603.06343)**

> **作者:** Lorenzo Farina; Federico Gavioli; Salvatore Iandolo; Francesco Moretti; Giuseppe Perrone; Matteo Piccoli; Francesco Raviglione; Marco Rapelli; Antonio Solida; Paolo Burgio; Carlo Augusto Grazia; Alessandro Bazzi
>
> **备注:** 5 pages, 6 figures
>
> **摘要:** The automotive sector is following a revolutionary path from vehicles controlled by humans to vehicles that will be fully automated, fully connected, and ultimately fully cooperative. Along this road, new cooperative algorithms and protocols will be designed and field tested, which represents a great challenge in terms of costs. In this context, in particular, moving from simulations to practical experiments requires huge investments that are not always affordable and may become a barrier in some cases. To solve this issue and provide the community with an intermediate step, we here propose the use of 1:10 scaled cooperative, autonomous, and connected mini-cars. The mini-car is equipped with a Jetson Orin board running the open Robot Operating System 2 (ROS2), sensors for autonomous operations, and a Raspberry Pi board for connectivity mounting the open source Open Stack for Car (OScar). A key aspect of the proposal is the use of OScar, which implements a full ETSI cooperative-intelligent transport systems (C-ITS) compliant stack. The feasibility and potential of the proposed platform is here demonstrated through the implementation of a case study where the Day-1 intersection collision warning (ICW) application is implemented and validated.
>
---
#### [new 025] Underactuated multimodal jumping robot for extraterrestrial exploration
- **分类: cs.RO**

- **简介: 该论文属于机器人探索任务，旨在解决低重力环境下多模式移动问题。设计了一种仅用两台反应轮控制的欠驱动单足机器人，实现滚动、跳跃和自正功能。**

- **链接: [https://arxiv.org/pdf/2603.06525](https://arxiv.org/pdf/2603.06525)**

> **作者:** Neil R. Wagner; Justin K. Yim
>
> **备注:** 8 pages, 14 figures, Accepted for ICRA 2026
>
> **摘要:** We present a rolling and jumping underactuated monopedal robot designed to explore multimodal locomotion on low-gravity bodies. It uses only two reaction wheels to control its spatial orientation with two controllers: a balancing controller which can aim the robot's jump direction on the ground, and an aerial reorientation controller which can aim the robot's leg for landing after flight. We demonstrate rolling, targeted jumping and landing, and self-righting using only three actuators total, keeping system size to 0.33m and 1.25kg. Simple switching between locomotion modes enables the system to deal with differing landscapes and environmental conditions.
>
---
#### [new 026] Iterative Convex Optimization with Control Barrier Functions for Obstacle Avoidance among Polytopes
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决多边形障碍物避障问题。通过构建线性控制屏障函数，提出一种迭代凸优化框架，实现高效安全的轨迹规划。**

- **链接: [https://arxiv.org/pdf/2603.05916](https://arxiv.org/pdf/2603.05916)**

> **作者:** Shuo Liu; Zhe Huang; Calin A. Belta
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Obstacle avoidance of polytopic obstacles by polytopic robots is a challenging problem in optimization-based control and trajectory planning. Many existing methods rely on smooth geometric approximations, such as hyperspheres or ellipsoids, which allow differentiable distance expressions but distort the true geometry and restrict the feasible set. Other approaches integrate exact polytope distances into nonlinear model predictive control (MPC), resulting in nonconvex programs that limit real-time performance. In this paper, we construct linear discrete-time control barrier function (DCBF) constraints by deriving supporting hyperplanes from exact closest-point computations between convex polytopes. We then propose a novel iterative convex MPC-DCBF framework, where local linearization of system dynamics and robot geometry ensures convexity of the finite-horizon optimization at each iteration. The resulting formulation reduces computational complexity and enables fast online implementation for safety-critical control and trajectory planning of general nonlinear dynamics. The framework extends to multi-robot and three-dimensional environments. Numerical experiments demonstrate collision-free navigation in cluttered maze scenarios with millisecond-level solve times.
>
---
#### [new 027] Terrain characterization and locomotion adaptation in a small-scale lizard-inspired robot
- **分类: cs.RO**

- **简介: 该论文属于小尺度机器人自主导航任务，解决复杂地形适应问题。设计SILA Bot，通过感知与控制策略实现地形自适应运动。**

- **链接: [https://arxiv.org/pdf/2603.05837](https://arxiv.org/pdf/2603.05837)**

> **作者:** Duncan Andrews; Landon Zimmerman; Evan Martin; Joe DiGennaro; Baxi Chong
>
> **备注:** 7 pages. 9 figures. IROS 2026 Conference
>
> **摘要:** Unlike their large-scale counterparts, small-scale robots are largely confined to laboratory environments and are rarely deployed in real-world settings. As robot size decreases, robot-terrain interactions fundamentally change; however, there remains a lack of systematic understanding of what sensory information small-scale robots should acquire and how they should respond when traversing complex natural terrains. To address these challenges, we develop a Small-scale, Intelligent, Lizard-inspired, Adaptive Robot (SILA Bot) capable of adapting to diverse substrates. We use granular media of varying depths as a controlled yet representative terrain paradigm. We show that the optimal body movement pattern (ranging from standing-wave bending that assists limb retraction on flat ground to traveling-wave undulation that generates thrust in deep granular media) can be parameterized and approximated as a linear function of granular depth. Furthermore, proprioceptive signals, such as joint torque, provide sufficient information to estimate granular depth via a K-Nearest Neighbors classifier, achieving 95% accuracy. Leveraging these relationships, we design a simple linear feedback controller that modulates body phase and substantially improves locomotion performance on terrains with unknown depth. Together, these results establish a principled framework for perception and control in small-scale locomotion and enable effective terrain-adaptive locomotion while maintaining low computational complexity.
>
---
#### [new 028] SG-DOR: Learning Scene Graphs with Direction-Conditioned Occlusion Reasoning for Pepper Plants
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SG-DOR框架，解决密集作物中果实采摘的遮挡关系问题，通过场景图建模物理连接与方向遮挡，提升干预规划效果。**

- **链接: [https://arxiv.org/pdf/2603.06512](https://arxiv.org/pdf/2603.06512)**

> **作者:** Rohit Menon; Niklas Mueller-Goldingen; Sicong Pan; Gokul Krishna Chenchani; Maren Bennewitz
>
> **摘要:** Robotic harvesting in dense crop canopies requires effective interventions that depend not only on geometry, but also on explicit, direction-conditioned relations identifying which organs obstruct a target fruit. We present SG-DOR (Scene Graphs with Direction-Conditioned Occlusion Reasoning), a relational framework that, given instance-segmented organ point clouds, infers a scene graph encoding physical attachments and direction-conditioned occlusion. We introduce an occlusion ranking task for retrieving and ranking candidate leaves for a target fruit and approach direction, and propose a direction-aware graph neural architecture with per-fruit leaf-set attention and union-level aggregation. Experiments on a multi-plant synthetic pepper dataset show improved occlusion prediction (F1=0.73, NDCG@3=0.85) and attachment inference (edge F1=0.83) over strong ablations, yielding a structured relational signal for downstream intervention planning.
>
---
#### [new 029] A Unified Low-Dimensional Design Embedding for Joint Optimization of Shape, Material, and Actuation in Soft Robots
- **分类: cs.RO**

- **简介: 该论文属于软体机器人设计任务，解决形状、材料和驱动联合优化问题。提出一种低维统一嵌入方法，实现高效协同设计。**

- **链接: [https://arxiv.org/pdf/2603.06497](https://arxiv.org/pdf/2603.06497)**

> **作者:** Vittorio Candiello; Manuel Mekkattu; Mike Y. Michelis; Robert K. Katzschmann
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Soft robots achieve functionality through tight coupling among geometry, material composition, and actuation. As a result, effective design optimization requires these three aspects to be considered jointly rather than in isolation. This coupling is computationally challenging: nonlinear large-deformation mechanics increase simulation cost, while contact, collision handling, and non-smooth state transitions limit the applicability of standard gradient-based approaches. We introduce a smooth, low-dimensional design embedding for soft robots that unifies shape morphing, multi-material distribution, and actuation within a single structured parameter space. Shape variation is modeled through continuous deformation maps of a reference geometry, while material properties are encoded as spatial fields. Both are constructed from shared basis functions. This representation enables expressive co-design while drastically reducing the dimensionality of the search space. In our experiments, we show that design expressiveness increases with the number of basis functions, unlike comparable neural network encodings whose representational capacity does not scale predictably with parameter count. We further show that joint co-optimization of shape, material, and actuation using our unified embedding consistently outperforms sequential strategies. All experiments are performed independently of the underlying simulator, confirming compatibility with black-box simulation pipelines. Across multiple dynamic tasks, the proposed embedding surpasses neural network and voxel-based baseline parameterizations while using significantly fewer design parameters. Together, these findings demonstrate that structuring the design space itself enables efficient co-design of soft robots.
>
---
#### [new 030] Moving Through Clutter: Scaling Data Collection and Benchmarking for 3D Scene-Aware Humanoid Locomotion via Virtual Reality
- **分类: cs.RO**

- **简介: 该论文属于人形机器人在复杂三维环境中的运动任务，旨在解决 cluttered 环境下运动规划与控制问题，通过VR收集数据并构建基准测试集。**

- **链接: [https://arxiv.org/pdf/2603.05993](https://arxiv.org/pdf/2603.05993)**

> **作者:** Beichen Wang; Yuanjie Lu; Linji Wang; Liuchuan Yu; Xuesu Xiao
>
> **摘要:** Recent advances in humanoid locomotion have enabled dynamic behaviors such as dancing, martial arts, and parkour, yet these capabilities are predominantly demonstrated in open, flat, and obstacle-free settings. In contrast, real-world environments such as homes, offices, and public spaces, are densely cluttered, three-dimensional, and geometrically constrained, requiring scene-aware whole-body coordination, precise balance control, and reasoning over spatial constraints imposed by furniture and household objects. However, humanoid locomotion in cluttered 3D environments remains underexplored, and no public dataset systematically couples full-body human locomotion with the scene geometry that shapes it. To address this gap, we present Moving Through Clutter (MTC), an opensource Virtual Reality (VR) based data collection and evaluation framework for scene-aware humanoid locomotion in cluttered environments. Our system procedurally generates scenes with controllable clutter levels and captures embodiment-consistent, whole-body human motion through immersive VR navigation, which is then automatically retargeted to a humanoid robot model. We further introduce benchmarks that quantify environment clutter level and locomotion performance, including stability and collision safety. Using this framework, we compile a dataset of 348 trajectories across 145 diverse 3D cluttered scenes. The dataset provides a foundation for studying geometry-induced adaptation in humanoid locomotion and developing scene-aware planning and control methods.
>
---
#### [new 031] Safe Consensus of Cooperative Manipulation with Hierarchical Event-Triggered Control Barrier Functions
- **分类: cs.RO**

- **简介: 该论文属于多机械臂协同操作任务，解决动态环境中安全协作控制问题。提出分层事件触发控制屏障函数框架，实现高效安全的协调控制。**

- **链接: [https://arxiv.org/pdf/2603.06356](https://arxiv.org/pdf/2603.06356)**

> **作者:** Simiao Zhuang; Bingkun Huang; Zewen Yang
>
> **备注:** 8 pages
>
> **摘要:** Cooperative transport and manipulation of heavy or bulky payloads by multiple manipulators requires coordinated formation tracking, while simultaneously enforcing strict safety constraints in varying environments with limited communication and real-time computation budgets. This paper presents a distributed control framework that achieves consensus coordination with safety guarantees via hierarchical event-triggered control barrier functions (CBFs). We first develop a consensus-based protocol that relies solely on local neighbor information to enforce both translational and rotational consistency in task space. Building on this coordination layer, we propose a three-level hierarchical event-triggered safety architecture with CBFs, which is integrated with a risk-aware leader selection and smooth switching strategy to reduce online computation. The proposed approach is validated through real-world hardware experiments using two Franka manipulators operating with static obstacles, as well as comprehensive simulations demonstrating scalable multi-arm cooperation with dynamic obstacles. Results demonstrate higher precision cooperation under strict safety constraints, achieving substantially reduced computational cost and communication frequency compared to baseline methods.
>
---
#### [new 032] Hierarchical Latent Action Model
- **分类: cs.RO**

- **简介: 该论文提出HiLAM，解决长时序动作建模问题，通过层次化结构发现高阶技能，提升动态技能识别效果。**

- **链接: [https://arxiv.org/pdf/2603.05815](https://arxiv.org/pdf/2603.05815)**

> **作者:** Hanjung Kim; Lerrel Pinto; Seon Joo Kim
>
> **备注:** ICLR 2026 Workshop - 2nd Workshop on World Models: Understanding, Modelling and Scaling
>
> **摘要:** Latent Action Models (LAMs) enable learning from actionless data for applications ranging from robotic control to interactive world models. However, existing LAMs typically focus on short-horizon frame transitions and capture low-level motion while overlooking longer-term temporal structure. In contrast, actionless videos often contain temporally extended and high-level skills. We present HiLAM, a hierarchical latent action model that discovers latent skills by modeling long-term temporal information. To capture these dependencies across long horizons, we utilize a pretrained LAM as a low-level extractor. This architecture aggregates latent action sequences, which contain the underlying dynamic patterns of the video, into high-level latent skills. Our experiments demonstrate that HiLAM improves over the baseline and exhibits robust dynamic skill discovery.
>
---
#### [new 033] CFEAR-Teach-and-Repeat: Fast and Accurate Radar-only Localization
- **分类: cs.RO**

- **简介: 该论文属于定位任务，解决恶劣天气下自主导航的可靠定位问题。提出CFEAR-TR方法，利用单雷达实现快速准确的定位。**

- **链接: [https://arxiv.org/pdf/2603.06501](https://arxiv.org/pdf/2603.06501)**

> **作者:** Maximilian Hilger; Daniel Adolfsson; Ralf Becker; Henrik Andreasson; Achim J. Lilienthal
>
> **备注:** This paper has been accepted for publication in the IEEE International Conference on Robotics and Automation (ICRA), 2026
>
> **摘要:** Reliable localization in prior maps is essential for autonomous navigation, particularly under adverse weather, where optical sensors may fail. We present CFEAR-TR, a teach-and-repeat localization pipeline using a single spinning radar, which is designed for easily deployable, lightweight, and robust navigation in adverse conditions. Our method localizes by jointly aligning live scans to both stored scans from the teach mapping pass, and to a sliding window of recent live keyframes. This ensures accurate and robust pose estimation across different seasons and weather phenomena. Radar scans are represented using a sparse set of oriented surface points, computed from Doppler-compensated measurements. The map is stored in a pose graph that is traversed during localization. Experiments on the held-out test sequences from the Boreas dataset show that CFEAR-TR can localize with an accuracy as low as 0.117 m and 0.096°, corresponding to improvements of up to 63% over the previous state of the art, while running efficiently at 29 Hz. These results substantially narrow the gap to lidar-level localization, particularly in heading estimation. We make the C++ implementation of our work available to the community.
>
---
#### [new 034] Dual-Agent Multiple-Model Reinforcement Learning for Event-Triggered Human-Robot Co-Adaptation in Decoupled Task Spaces
- **分类: cs.RO**

- **简介: 该论文研究康复机器人控制任务，解决传统控制导致的轨迹振荡问题。提出DAMMRL框架，通过事件触发机制优化人机协同，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.06163](https://arxiv.org/pdf/2603.06163)**

> **作者:** Yaqi Li; Zhengqi Han; Huifang Liu; Steven W.Su
>
> **摘要:** This paper presents a shared-control rehabilitation policy for a custom 6-degree-of-freedom (6-DoF) upper-limb robot that decomposes complex reaching tasks into decoupled spatial axes. The patient governs the primary reaching direction using binary commands, while the robot autonomously manages orthogonal corrective motions. Because traditional fixed-frequency control often induces trajectory oscillations due to variable inverse-kinematics execution times, an event-driven progression strategy is proposed. This architecture triggers subsequent control actions only when the end-effector enters an admission sphere centred on the immediate target waypoint, and was validated in a semi-virtual setup linking a physical pressure sensor to a MuJoCo simulation. To optimise human--robot co-adaptation safely and efficiently, this study introduces Dual Agent Multiple Model Reinforcement Learning (DAMMRL). This framework discretises decision characteristics: the human agent selects the admission sphere radius to reflect their inherent speed--accuracy trade-off, while the robot agent dynamically adjusts its 3D Cartesian step magnitudes to complement the user's cognitive state. Trained in simulation and deployed across mixed environments, this event-triggered DAMMRL approach effectively suppresses waypoint chatter, balances spatial precision with temporal efficiency, and significantly improves success rates in object acquisition tasks.
>
---
#### [new 035] Safe-Night VLA: Seeing the Unseen via Thermal-Perceptive Vision-Language-Action Models for Safety-Critical Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决传统视觉语言动作模型感知局限与安全约束不足的问题。通过引入热感知模块和安全过滤机制，提升机器人在复杂环境中的安全操控能力。**

- **链接: [https://arxiv.org/pdf/2603.05754](https://arxiv.org/pdf/2603.05754)**

> **作者:** Dian Yu; Qingchuan Zhou; Bingkun Huang; Majid Khadiv; Zewen Yang
>
> **摘要:** Current Vision-Language-Action (VLA) models rely primarily on RGB perception, preventing them from capturing modalities such as thermal signals that are imperceptible to conventional visual sensors. Moreover, end-to-end generative policies lack explicit safety constraints, making them fragile when encountering obstacles and novel scenarios outside the training distribution. To address these limitations, we propose Safe-Night VLA, a multimodal manipulation framework that enables robots to see the unseen while enforcing rigorous safety constraints for thermal-aware manipulation in unstructured environments. Specifically, Safe-Night VLA integrates long-wave infrared thermal perception into a pre-trained vision-language backbone, enabling semantic reasoning grounded in thermodynamic properties. To ensure safe execution under out-of-distribution conditions, we incorporate a safety filter via control barrier functions, which provide deterministic workspace constraint enforcement during policy execution. We validate our framework through real-world experiments on a Franka manipulator, introducing a novel evaluation paradigm featuring temperature-conditioned manipulation, subsurface target localization, and reflection disambiguation, while maintaining constrained execution at inference time. Results demonstrate that Safe-Night VLA outperforms RGB-only baselines and provide empirical evidence that foundation models can effectively leverage non-visible physical modalities for robust manipulation.
>
---
#### [new 036] SuperSuit: An Isomorphic Bimodal Interface for Scalable Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文提出SuperSuit，解决移动机械臂长周期演示数据获取难题。通过双模态接口实现高效数据采集，提升移动操作的可扩展性。**

- **链接: [https://arxiv.org/pdf/2603.06280](https://arxiv.org/pdf/2603.06280)**

> **作者:** Tongqing Chen; Hang Wu; Jiasen Wang; Xiaotao Li; Zhu Jin; Lu Fang
>
> **摘要:** High-quality, long-horizon demonstrations are essential for embodied AI, yet acquiring such data for tightly coupled wheeled mobile manipulators remains a fundamental bottleneck. Unlike fixed-base systems, mobile manipulators require continuous coordination between $SE(2)$ locomotion and precise manipulation, exposing limitations in existing teleoperation and wearable interfaces. We present \textbf{SuperSuit}, a bimodal data acquisition framework that supports both robot-in-the-loop teleoperation and active demonstration under a shared kinematic interface. Both modalities produce structurally identical joint-space trajectories, enabling direct data mixing without modifying downstream policies. For locomotion, SuperSuit maps natural human stepping to continuous planar base velocities, eliminating discrete command switches. For manipulation, it employs a strictly isomorphic wearable arm in both modes, while policy training is formulated in a shift-invariant delta-joint representation to mitigate calibration offsets and structural compliance without inverse kinematics. Real-world experiments on long-horizon mobile manipulation tasks show 2.6$\times$ higher demonstration throughput in active mode compared to a teleoperation baseline, comparable policy performance when substituting teleoperation data with active demonstrations at fixed dataset size, and monotonic performance improvement as active data volume increases. These results indicate that consistent kinematic representations across collection modalities enable scalable data acquisition for long-horizon mobile manipulation.
>
---
#### [new 037] CDF-Glove: A Cable-Driven Force Feedback Glove for Dexterous Teleoperation
- **分类: cs.RO**

- **简介: 该论文属于遥操作任务，旨在解决灵巧操作中缺乏触觉反馈的问题。提出轻量低成本的CDF-Glove，实现高精度力反馈，提升操作质量。**

- **链接: [https://arxiv.org/pdf/2603.05804](https://arxiv.org/pdf/2603.05804)**

> **作者:** Huayue Liang; Ruochong Li; Yaodong Yang; Long Zeng; Yuanpei Chen; Xueqian Wang
>
> **摘要:** High-quality teleoperated demonstrations are a primary bottleneck for imitation learning (IL) in dexterous manipulation. However, haptic feedback provides operators with real-time contact information, enabling real-time finger posture adjustments, and thereby improving demonstration quality. Existing dexterous teleoperation platforms typically omit haptic feedback and remain bulky and expensive. We introduce CDF-Glove, a lightweight and low cost cable-driven force-feedback glove. The real-time state is available for 20 finger degrees of freedom (DoF), of which 16 are directly sensed and 4 are passively coupled (inferred from kinematic constraints). We develop a kinematic model and control stack for the glove, and validate them across multiple robotic hands with diverse kinematics and DoF. The CDF-Glove achieves distal joint repeatability of 0.4 degrees, and delivers about 200 ms force feedback latency, yielding a 4x improvement in task success rate relative to no-feedback teleoperation. We collect two bimanual teleoperation datasets, on which we train and evaluate Diffusion Policy baselines. Compared to kinesthetic teaching, the policies trained in our teleoperated demonstrations increase the average success rate by 55% and reduce the mean completion time by approximately 15.2 seconds (a 47.2% relative reduction). In particular, the CDF-Glove costs approximately US$230. The code and designs are released as open source at this https URL.
>
---
#### [new 038] Vision-Language System using Open-Source LLMs for Gestures in Medical Interpreter Robots
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于医疗交互任务，旨在解决跨语言沟通中非语言信号的识别与生成问题。通过构建视觉-语言框架，实现对特定语用行为的检测及相应手势生成。**

- **链接: [https://arxiv.org/pdf/2603.05751](https://arxiv.org/pdf/2603.05751)**

> **作者:** Thanh-Tung Ngo; Emma Murphy; Robert J. Ross
>
> **摘要:** Effective communication is vital in healthcare, especially across language barriers, where non-verbal cues and gestures are critical. This paper presents a privacy-preserving vision-language framework for medical interpreter robots that detects specific speech acts (consent and instruction) and generates corresponding robotic gestures. Built on locally deployed open-source models, the system utilizes a Large Language Model (LLM) with few-shot prompting for intent detection. We also introduce a novel dataset of clinical conversations annotated for speech acts and paired with gesture clips. Our identification module achieved 0.90 accuracy, 0.93 weighted precision, and a 0.91 weighted F1-Score. Our approach significantly improves computational efficiency and, in user studies, outperforms the speech-gesture generation baseline in human-likeness while maintaining comparable appropriateness.
>
---
#### [new 039] TEGA: A Tactile-Enhanced Grasping Assistant for Assistive Robotics via Sensor Fusion and Closed-Loop Haptic Feedback
- **分类: cs.RO**

- **简介: 该论文属于辅助机器人领域，解决抓取力调节不足的问题，通过融合肌电信号与触觉感知，实现闭环触觉反馈，提升抓握稳定性。**

- **链接: [https://arxiv.org/pdf/2603.05552](https://arxiv.org/pdf/2603.05552)**

> **作者:** Hengxu You; Tianyu Zhou; Fang Xu; Kaleb Smith; Eric Jing Du
>
> **备注:** Accepted to include in ICRA 2026
>
> **摘要:** Recent advances in teleoperation have enabled sophisticated manipulation of dexterous robotic hands, with most systems concentrating on guiding finger positions to achieve desired grasp configurations. However, while accurate finger positioning is essential, it often overlooks the equally critical task of grasp force modulation, vital for handling objects of diverse hardness, texture, and shape. This limitation poses a significant challenge for users, especially individuals with upper limb disabilities who lack natural tactile feedback and rely on indirect cues to infer appropriate force levels. To address this gap, We present the tactile enhanced grasping assistant (TEGA), a closed loop assistive teleoperation framework that fuses EMG based intent2force inference with visuotactile sensing mapped into real time vibrotactile feedback via a wearable haptic vest, enabling intuitive, proportional force adjustment during manipulation. A wearable haptic vest delivers real time tactile feedback, allowing users to dynamically refine grasp force during manipulation. User studies confirm that the system substantially improves grasp stability and task success, underscoring its potential for assistive robotic applications.
>
---
#### [new 040] Fly360: Omnidirectional Obstacle Avoidance within Drone View
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于无人机避障任务，解决全向避障问题。针对传统方法视角受限的问题，提出Fly360框架，通过全景感知与轻量决策网络实现稳定避障。**

- **链接: [https://arxiv.org/pdf/2603.06573](https://arxiv.org/pdf/2603.06573)**

> **作者:** Xiangkai Zhang; Dizhe Zhang; WenZhuo Cao; Zhaoliang Wan; Yingjie Niu; Lu Qi; Xu Yang; Zhiyong Liu
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** Obstacle avoidance in unmanned aerial vehicles (UAVs), as a fundamental capability, has gained increasing attention with the growing focus on spatial intelligence. However, current obstacle-avoidance methods mainly depend on limited field-of-view sensors and are ill-suited for UAV scenarios which require full-spatial awareness when the movement direction differs from the UAV's heading. This limitation motivates us to explore omnidirectional obstacle avoidance for panoramic drones with full-view perception. We first study an under explored problem setting in which a UAV must generate collision-free motion in environments with obstacles from arbitrary directions, and then construct a benchmark that consists of three representative flight tasks. Based on such settings, we propose Fly360, a two-stage perception-decision pipeline with a fixed random-yaw training strategy. At the perception stage, panoramic RGB observations are input and converted into depth maps as a robust intermediate representation. For the policy network, it is lightweight and used to output body-frame velocity commands from depth inputs. Extensive simulation and real-world experiments demonstrate that Fly360 achieves stable omnidirectional obstacle avoidance and outperforms forward-view baselines across all tasks. Our model is available at this https URL
>
---
#### [new 041] Relational Semantic Reasoning on 3D Scene Graphs for Open World Interactive Object Search
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于开放世界交互式物体搜索任务，解决传统方法在语义关系理解与实时性上的不足，提出SCOUT方法通过3D场景图进行高效探索。**

- **链接: [https://arxiv.org/pdf/2603.05642](https://arxiv.org/pdf/2603.05642)**

> **作者:** Imen Mahdi; Matteo Cassinelli; Fabien Despinoy; Tim Welschehold; Abhinav Valada
>
> **摘要:** Open-world interactive object search in household environments requires understanding semantic relationships between objects and their surrounding context to guide exploration efficiently. Prior methods either rely on vision-language embeddings similarity, which does not reliably capture task-relevant relational semantics, or large language models (LLMs), which are too slow and costly for real-time deployment. We introduce SCOUT: Scene Graph-Based Exploration with Learned Utility for Open-World Interactive Object Search, a novel method that searches directly over 3D scene graphs by assigning utility scores to rooms, frontiers, and objects using relational exploration heuristics such as room-object containment and object-object co-occurrence. To make this practical without sacrificing open-vocabulary generalization, we propose an offline procedural distillation framework that extracts structured relational knowledge from LLMs into lightweight models for on-robot inference. Furthermore, we present SymSearch, a scalable symbolic benchmark for evaluating semantic reasoning in interactive object search tasks. Extensive evaluations across symbolic and simulation environments show that SCOUT outperforms embedding similarity-based methods and matches LLM-level performance while remaining computationally efficient. Finally, real-world experiments demonstrate effective transfer to physical environments, enabling open-world interactive object search under realistic sensing and navigation constraints.
>
---
#### [new 042] TADPO: Reinforcement Learning Goes Off-road
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于自主驾驶任务，解决越野环境下的长距离规划与控制问题。提出TADPO算法，结合师生策略优化，实现高速越野的端到端强化学习系统，并成功进行真实车辆测试。**

- **链接: [https://arxiv.org/pdf/2603.05995](https://arxiv.org/pdf/2603.05995)**

> **作者:** Zhouchonghao Wu; Raymond Song; Vedant Mundheda; Luis E. Navarro-Serment; Christof Schoenborn; Jeff Schneider
>
> **备注:** 8 pages, 5 figures, 2 tables. Accepted at ICRA 2026
>
> **摘要:** Off-road autonomous driving poses significant challenges such as navigating unmapped, variable terrain with uncertain and diverse dynamics. Addressing these challenges requires effective long-horizon planning and adaptable control. Reinforcement Learning (RL) offers a promising solution by learning control policies directly from interaction. However, because off-road driving is a long-horizon task with low-signal rewards, standard RL methods are challenging to apply in this setting. We introduce TADPO, a novel policy gradient formulation that extends Proximal Policy Optimization (PPO), leveraging off-policy trajectories for teacher guidance and on-policy trajectories for student exploration. Building on this, we develop a vision-based, end-to-end RL system for high-speed off-road driving, capable of navigating extreme slopes and obstacle-rich terrain. We demonstrate our performance in simulation and, importantly, zero-shot sim-to-real transfer on a full-scale off-road vehicle. To our knowledge, this work represents the first deployment of RL-based policies on a full-scale off-road platform.
>
---
#### [new 043] A Hazard-Informed Data Pipeline for Robotics Physical Safety
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人安全领域，旨在解决物理安全问题。通过构建基于危害的框架，整合资产声明、漏洞枚举和合成数据生成，提升机器学习模型的安全性。**

- **链接: [https://arxiv.org/pdf/2603.06130](https://arxiv.org/pdf/2603.06130)**

> **作者:** Alexei Odinokov; Rostislav Yavorskiy
>
> **备注:** 4th International Conference on Automation and Mechatronics Engineering (ICAME 2026)
>
> **摘要:** This report presents a structured Robotics Physical Safety Framework based on explicit asset declaration, systematic vulnerability enumeration, and hazard-driven synthetic data generation. The approach bridges classical risk engineering with modern machine learning pipelines, enabling safety envelope learning grounded in a formalized hazard ontology. The key contribution of this framework is the alignment between classical safety engineering, digital twin simulation, synthetic data generation, and machine learning model training.
>
---
#### [new 044] Few-Shot Neural Differentiable Simulator: Real-to-Sim Rigid-Contact Modeling
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决物理模拟与真实世界数据不匹配的问题。通过结合分析模型与图神经网络，实现少量真实数据下的高精度仿真与优化。**

- **链接: [https://arxiv.org/pdf/2603.06218](https://arxiv.org/pdf/2603.06218)**

> **作者:** Zhenhao Huang; Siyuan Luo; Bingyang Zhou; Ziqiu Zeng; Jason Pho; Fan Shi
>
> **摘要:** Accurate physics simulation is essential for robotic learning and control, yet analytical simulators often fail to capture complex contact dynamics, while learning-based simulators typically require large amounts of costly real-world data. To bridge this gap, we propose a few-shot real-to-sim approach that combines the physical consistency of analytical formulations with the representational capacity of graph neural network (GNN)-based models. Using only a small amount of real-world data, our method calibrates analytical simulators to generate large-scale synthetic datasets that capture diverse contact interactions. On this foundation, we introduce a mesh-based GNN that implicitly models rigid-body forward dynamics and derive surrogate gradients for collision detection, achieving full differentiability. Experimental results demonstrate that our approach enables learning-based simulators to outperform differentiable baselines in replicating real-world trajectories. In addition, the differentiable design supports gradient-based optimization, which we validate through simulation-based policy learning in multi-object interaction scenarios. Extensive experiments show that our framework not only improves simulation fidelity with minimal supervision but also increases the efficiency of policy learning. Taken together, these findings suggest that differentiable simulation with few-shot real-world grounding provides a powerful direction for advancing future robotic manipulation and control.
>
---
#### [new 045] Swooper: Learning High-Speed Aerial Grasping With a Simple Gripper
- **分类: cs.RO**

- **简介: 该论文属于高速空中抓取任务，解决飞行控制与抓取协调难题。通过深度强化学习方法，设计简单 gripper 实现高效抓取。**

- **链接: [https://arxiv.org/pdf/2603.05935](https://arxiv.org/pdf/2603.05935)**

> **作者:** Ziken Huang; Xinze Niu; Bowen Chai; Renbiao Jin; Danping Zou
>
> **摘要:** High-speed aerial grasping presents significant challenges due to the high demands on precise, responsive flight control and coordinated gripper manipulation. In this work, we propose Swooper, a deep reinforcement learning (DRL) based approach that achieves both precise flight control and active gripper control using a single lightweight neural network policy. Training such a policy directly via DRL is nontrivial due to the complexity of coordinating flight and grasping. To address this, we adopt a two-stage learning strategy: we first pre-train a flight control policy, and then fine-tune it to acquire grasping skills. With the carefully designed reward functions and training framework, the entire training process completes in under 60 minutes on a standard desktop with an Nvidia RTX 3060 GPU. To validate the trained policy in the real world, we develop a lightweight quadrotor grasping platform equipped with a simple off-the-shelf gripper, and deploy the policy in a zero-shot manner on the onboard Raspberry Pi 4B computer, where each inference takes only about 1.0 ms. In 25 real-world trials, our policy achieves an 84% grasp success rate and grasping speeds of up to 1.5 m/s without any fine-tuning. This matches the robustness and agility of state-of-the-art classical systems with sophisticated grippers, highlighting the capability of DRL for learning a robust control policy that seamlessly integrates high-speed flight and grasping. The supplementary video is available for more results. Video: this https URL.
>
---
#### [new 046] RFM-HRI : A Multimodal Dataset of Medical Robot Failure, User Reaction and Recovery Preferences for Item Retrieval Tasks
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决医疗机器人故障下用户反应与恢复偏好问题。通过构建RFM-HRI数据集，分析用户在不同故障类型下的情绪和行为反应，为故障恢复提供依据。**

- **链接: [https://arxiv.org/pdf/2603.05641](https://arxiv.org/pdf/2603.05641)**

> **作者:** Yashika Batra; Giuliano Pioldi; Promise Ekpo; Arman Sayatqyzy; Purnjay Maruur; Shalom Otieno; Kevin Ching; Angelique Taylor
>
> **摘要:** While robots deployed in real-world environments inevitably experience interaction failures, understanding how users respond through verbal and non-verbal behaviors remains under-explored in human-robot interaction (HRI). This gap is particularly significant in healthcare-inspired settings, where interaction failures can directly affect task performance and user trust. We present the Robot Failures in Medical HRI (RFM-HRI) Dataset, a multimodal dataset capturing dyadic interactions between humans and robots embodied in crash carts, where communication failures are systematically induced during item retrieval tasks. Through Wizard-of-Oz studies with 41 participants across laboratory and hospital settings, we recorded responses to four failure types (speech, timing, comprehension, and search) derived from three years of crash-cart robot interaction data. The dataset contains 214 interaction samples including facial action units, head pose, speech transcripts, and post-interaction self-reports. Our analysis shows that failures significantly degrade affective valence and reduce perceived control compared to successful interactions. Failures are strongly associated with confusion, annoyance, and frustration, while successful interactions are characterized by surprise, relief, and confidence in task completion. Emotional responses also evolve across repeated failures, with confusion decreasing and frustration increasing over time. This work contributes (1) a publicly available multimodal dataset (RFM-HRI), (2) analysis of user responses to different failure types and preferred recovery strategies, and (3) a crash-cart retrieval scenario enabling systematic comparison of recovery strategies with implications for safety-critical failure recovery. Our findings provide a foundation for failure detection and recovery methods in embodied HRI.
>
---
#### [new 047] PROBE: Probabilistic Occupancy BEV Encoding with Analytical Translation Robustness for 3D Place Recognition
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出PROBE，一种基于贝叶斯的LiDAR场景识别方法，解决位移鲁棒性问题，通过概率建模提升跨传感器和多会话的识别精度。**

- **链接: [https://arxiv.org/pdf/2603.05965](https://arxiv.org/pdf/2603.05965)**

> **作者:** Jinseop Lee; Byoungho Lee; Gichul Yoo
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** We present PROBE (PRobabilistic Occupancy BEV Encoding), a learning-free LiDAR place recognition descriptor that models each BEV cell's occupancy as a Bernoulli random variable. Rather than relying on discrete point-cloud perturbations, PROBE analytically marginalizes over continuous Cartesian translations via the polar Jacobian, yielding a distance-adaptive angular uncertainty $\sigma_\theta = \sigma_t / r$ in $\mathcal{O}(R \times S)$ time. The primary parameter $\sigma_t$ represents the expected translational uncertainty in meters, a sensor-independent physical quantity allowing cross-sensor generalization without per-dataset tuning. Pairwise similarity combines a Bernoulli-KL Jaccard with exponential uncertainty gating and FFT-based height cosine similarity for rotation alignment. Evaluated on four datasets spanning four diverse LiDAR types, PROBE achieves the highest accuracy among handcrafted descriptors in multi-session evaluation and competitive single-session performance against both handcrafted and supervised baselines. The source code and supplementary materials are available at this https URL.
>
---
#### [new 048] OpenHEART: Opening Heterogeneous Articulated Objects with a Legged Manipulator
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决腿式机械臂打开异构铰接物体的问题。提出SAFE和ArtIEst方法，提升泛化能力和操作效率。**

- **链接: [https://arxiv.org/pdf/2603.05830](https://arxiv.org/pdf/2603.05830)**

> **作者:** Seonghyeon Lim; Hyeonwoo Lee; Seunghyun Lee; I Made Aswin Nahrendra; Hyun Myung
>
> **备注:** 8 pages
>
> **摘要:** Legged manipulators offer high mobility and versatile manipulation. However, robust interaction with heterogeneous articulated objects, such as doors, drawers, and cabinets, remains challenging because of the diverse articulation types of the objects and the complex dynamics of the legged robot. Existing reinforcement learning (RL)-based approaches often rely on high-dimensional sensory inputs, leading to sample inefficiency. In this paper, we propose a robust and sample-efficient framework for opening heterogeneous articulated objects with a legged manipulator. In particular, we propose Sampling-based Abstracted Feature Extraction (SAFE), which encodes handle and panel geometry into a compact low-dimensional representation, improving cross-domain generalization. Additionally, Articulation Information Estimator (ArtIEst) is introduced to adaptively mix proprioception with exteroception to estimate opening direction and range of motion for each object. The proposed framework was deployed to manipulate various heterogeneous articulated objects in simulation and real-world robot systems. Videos can be found on the project website: this https URL
>
---
#### [new 049] TransMASK: Masked State Representation through Learned Transformation
- **分类: cs.RO**

- **简介: 该论文提出TransMASK方法，用于学习忽略无关环境状态的机器人控制策略，提升泛化能力。属于模仿学习任务，解决环境变化下的策略鲁棒性问题。**

- **链接: [https://arxiv.org/pdf/2603.05670](https://arxiv.org/pdf/2603.05670)**

> **作者:** Sagar Parekh; Preston Culbertson; Dylan P. Losey
>
> **摘要:** Humans train robots to complete tasks in one environment, and expect robots to perform those same tasks in new environments. As humans, we know which aspects of the environment (i.e., the state) are relevant to the task. But there are also things that do not matter; e.g., the color of the table or the presence of clutter in the background. Ideally, the robot's policy learns to ignore these irrelevant state components. Achieving this invariance improves generalization: the robot knows not to factor irrelevant variables into its control decisions, making the policy more robust to environment changes. In this paper we therefore propose a self-supervised method to learn a mask which, when multiplied by the observed state, transforms that state into a latent representation that is biased towards relevant elements. Our method -- which we call TransMASK -- can be combined with a variety of imitation learning frameworks (such as diffusion policies) without any additional labels or alterations to the loss function. To achieve this, we recognize that the learned policy updates to better match the human's true policy. This true policy only depends on the relevant parts of the state; hence, as the gradients pass back through the learned policy and our proposed mask, they increase the value for elements that cause the robot to better imitate the human. We can therefore train TransMASK at the same time as we learn the policy. By normalizing the magnitude of each row in TransMASK, we force the mask to align with the Jacobian of the expert policy: columns that correspond to relevant states have large magnitudes, while columns for irrelevant states approach zero magnitude. We compare our approach to other methods that extract relevant states for downstream imitation learning. See our project website: this https URL
>
---
#### [new 050] Proprioceptive Shape Estimation of Tensegrity Manipulators Using Energy Minimisation
- **分类: cs.RO**

- **简介: 该论文属于形状估计任务，旨在解决连续弯曲张力结构机械臂的形状感知问题。通过仅使用每个杆件的倾斜角信息，实现高精度、稳定的形状估计。**

- **链接: [https://arxiv.org/pdf/2603.05976](https://arxiv.org/pdf/2603.05976)**

> **作者:** Tufail Ahmad Bhat; Shuhei Ikemoto
>
> **备注:** 8 pages, 10 figures, IEEE ICRA 2026
>
> **摘要:** Shape estimation is fundamental for controlling continuously bending tensegrity manipulators, yet achieving it remains a challenge. Although using exteroceptive sensors makes the implementation straightforward, it is costly and limited to specific environments. Proprioceptive approaches, by contrast, do not suffer from these limitations. So far, several methods have been proposed; however, to our knowledge, there are no proven examples of large-scale tensegrity structures used as manipulators. This paper demonstrates that shape estimation of the entire tensegrity manipulator can be achieved using only the inclination angle information relative to gravity for each strut. Inclination angle information is intrinsic sensory data that can be obtained simply by attaching an inertial measurement unit (IMU) to each strut. Experiments conducted on a five-layer tensegrity manipulator with 20 struts and a total length of 1160 mm demonstrate that the proposed method can estimate the shape with an accuracy of 2.1 \% of the total manipulator length, from arbitrary initial conditions under both static conditions and maintains stable shape estimation under external disturbances.
>
---
#### [new 051] History-Conditioned Spatio-Temporal Visual Token Pruning for Efficient Vision-Language Navigation
- **分类: cs.RO**

- **简介: 该论文针对视觉语言导航任务，解决VLA模型计算成本高、延迟大的问题。提出一种无需训练的时空视觉标记剪枝方法，提升推理效率并保持导航精度。**

- **链接: [https://arxiv.org/pdf/2603.06480](https://arxiv.org/pdf/2603.06480)**

> **作者:** Qitong Wang; Yijun Liang; Ming Li; Tianyi Zhou; Christopher Rasmussen
>
> **摘要:** Vision-Language Navigation (VLN) enables robots to follow natural-language instructions in visually grounded environments, serving as a key capability for embodied robotic systems. Recent Vision-Language-Action (VLA) models have demonstrated strong navigation performance, but their high computational cost introduces latency that limits real-time deployment. We propose a training-free spatio-temporal vision token pruning framework tailored to VLA-based VLN. We apply spatial token selection to the current view, alongside spatio-temporal compression for historical memories, enabling efficient long-horizon inference while reducing redundant computation. Leveraging attention-based token importance and query-guided spatio-temporal filtering, the proposed approach preserves navigation-relevant information without retraining or modifying pretrained models, allowing plug-and-play integration into existing VLA systems. Through experiments on standard VLN benchmarks, we confirm that our method significantly outperforms existing pruning strategies. It successfully preserves superior navigation accuracy under extreme pruning scenarios, all while maintaining the highly competitive inference efficiency. Real-world deployment on a Unitree Go2 quadruped robot further validates reliable and low-latency instruction-following navigation under practical robotic constraints. We hope this work helps bridge the gap between large-scale multimodal modeling and efficient, real-time embodied deployment in robotic navigation systems.
>
---
#### [new 052] Lifelong Embodied Navigation Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究终身具身导航学习任务，解决导航代理在持续学习中遗忘旧知识的问题。提出Uni-Walker框架，分离共享与特定导航知识，提升长期学习能力。**

- **链接: [https://arxiv.org/pdf/2603.06073](https://arxiv.org/pdf/2603.06073)**

> **作者:** Xudong Wang; Jiahua Dong; Baichen Liu; Qi Lyu; Lianqing Liu; Zhi Han
>
> **备注:** 24 pages, 7 figures
>
> **摘要:** Embodied navigation agents powered by large language models have shown strong performance on individual tasks but struggle to continually acquire new navigation skills, which suffer from catastrophic forgetting. We formalize this challenge as lifelong embodied navigation learning (LENL), where an agent is required to adapt to a sequence of navigation tasks spanning multiple scenes and diverse user instruction styles, while retaining previously learned knowledge. To tackle this problem, we propose Uni-Walker, a lifelong embodied navigation framework that decouples navigation knowledge into task-shared and task-specific components with Decoder Extension LoRA (DE-LoRA). To learn the shared knowledge, we design a knowledge inheritance strategy and an experts co-activation strategy to facilitate shared knowledge transfer and refinement across multiple navigation tasks. To learn the specific knowledge, we propose an expert subspace orthogonality constraint together and a navigation-specific chain-of-thought reasoning mechanism to capture specific knowledge and enhance instruction-style understanding. Extensive experiments demonstrate the superiority of Uni-Walker for building universal navigation agents with lifelong learning.
>
---
#### [new 053] Control Lyapunov Functions for Underactuated Soft Robots
- **分类: cs.RO**

- **简介: 该论文研究软体机器人的任务空间控制问题，针对欠驱动和执行器限制带来的稳定性挑战，提出一种基于控制李雅普诺夫函数的控制框架，提升控制精度和收敛性。**

- **链接: [https://arxiv.org/pdf/2603.05638](https://arxiv.org/pdf/2603.05638)**

> **作者:** Huy Pham; Zach J. Patterson
>
> **备注:** 8 pages, 5 figures, 2 tables. Submitted for publication to a conference
>
> **摘要:** Soft and soft-rigid hybrid robots are inherently underactuated and operate under tight actuator limits, making task-space control with stability guarantees challenging. Common nonlinear strategies for soft robots (e.g., those based on PD control) often rely on the assumption of full actuation with no actuator limits. This paper presents a general control framework for task-space regulation and tracking of underactuated soft robots under bounded inputs. The method enforces a rapidly exponentially stabilizing control Lyapunov function as a convex inequality constraint while simultaneously satisfying underactuated full-body dynamics and actuator bounds. We validate the approach in simulation on several platforms spanning increasing underactuation: a simple two link tendon-driven "finger", a trimmed helicoid manipulator, and a highly underactuated spiral robot. We compare against a number of baseline methods from the literature. Results show improved task-space accuracy and consistent Lyapunov convergence under input limits, achieving superior set-point and trajectory-tracking performance.
>
---
#### [new 054] DreamToNav: Generalizable Navigation for Robots via Generative Video Planning
- **分类: cs.RO**

- **简介: 该论文提出DreamToNav，属于机器人导航任务，解决传统导航依赖预设路径的问题。通过生成视频预测实现直观的人机交互控制，提升导航的通用性和准确性。**

- **链接: [https://arxiv.org/pdf/2603.06190](https://arxiv.org/pdf/2603.06190)**

> **作者:** Valerii Serpiva; Jeffrin Sam; Chidera Simon; Hajira Amjad; Iana Zhura; Artem Lykov; Dzmitry Tsetserukou
>
> **备注:** Submitted to conference
>
> **摘要:** We present DreamToNav, a novel autonomous robot framework that uses generative video models to enable intuitive, human-in-the-loop control. Instead of relying on rigid waypoint navigation, users provide natural language prompts (e.g. ``Follow the person carefully''), which the system translates into executable motion. Our pipeline first employs Qwen 2.5-VL-7B-Instruct to refine vague user instructions into precise visual descriptions. These descriptions condition NVIDIA Cosmos 2.5, a state-of-the-art video foundation model, to synthesize a physically consistent video sequence of the robot performing the task. From this synthetic video, we extract a valid kinematic path using visual pose estimation, robot detection and trajectory recovery. By treating video generation as a planning engine, DreamToNav allows robots to visually "dream" complex behaviors before executing them, providing a unified framework for obstacle avoidance and goal-directed navigation without task-specific engineering. We evaluate the approach on both a wheeled mobile robot and a quadruped robot in indoor navigation tasks. DreamToNav achieves a success rate of 76.7%, with final goal errors typically within 0.05-0.10 m and trajectory tracking errors below 0.15 m. These results demonstrate that trajectories extracted from generative video predictions can be reliably executed on physical robots across different locomotion platforms.
>
---
#### [new 055] Control Barrier Corridors: From Safety Functions to Safe Sets
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于安全控制任务，解决机器人在复杂环境中安全运动的问题。提出控制屏障走廊，统一了控制屏障函数与安全路径方法，实现安全目标选择与持续路径跟踪。**

- **链接: [https://arxiv.org/pdf/2603.06494](https://arxiv.org/pdf/2603.06494)**

> **作者:** Ömür Arslan; Nikolay Atanasov
>
> **备注:** 12 pages, 6 figures, an extended preprint version of a conference paper
>
> **摘要:** Safe autonomy is a critical requirement and a key enabler for robots to operate safely in unstructured complex environments. Control barrier functions and safe motion corridors are two widely used but technically distinct safety methods, functional and geometric, respectively, for safe motion planning and control. Control barrier functions are applied to the safety filtering of control inputs to limit the decay rate of system safety, whereas safe motion corridors are geometrically constructed to define a local safe zone around the system state for use in motion optimization and reference-governor design. This paper introduces a new notion of control barrier corridors, which unifies these two approaches by converting control barrier functions into local safe goal regions for reference goal selection in feedback control systems. We show, with examples on fully actuated systems, kinematic unicycles, and linear output regulation systems, that individual state safety can be extended locally over control barrier corridors for convex barrier functions, provided the control convergence rate matches the barrier decay rate, highlighting a trade-off between safety and reactiveness. Such safe control barrier corridors enable safely reachable persistent goal selection over continuously changing barrier corridors during system motion, which we demonstrate for verifiably safe and persistent path following in autonomous exploration of unknown environments.
>
---
#### [new 056] Task Parameter Extrapolation via Learning Inverse Tasks from Forward Demonstrations
- **分类: cs.RO**

- **简介: 该论文属于机器人学习领域，解决技能策略在新条件下的泛化问题。通过逆任务学习与正向示范结合，实现高效知识迁移，提升策略在复杂环境中的适应能力。**

- **链接: [https://arxiv.org/pdf/2603.05576](https://arxiv.org/pdf/2603.05576)**

> **作者:** Serdar Bahar; Fatih Dogangun; Matteo Saveriano; Yukie Nagai; Emre Ugur
>
> **摘要:** Generalizing skill policies to novel conditions remains a key challenge in robot learning. Imitation learning methods, while data-efficient, are largely confined to the training region and consistently fail on input data outside it, leading to unpredictable policy failures. Alternatively, transfer learning approaches offer methods for trajectory generation robust to both changes in environment or tasks, but they remain data-hungry and lack accuracy in zero-shot generalization. We address these challenges by framing the problem in the context of task inversion learning and proposing a novel joint learning approach to achieve accurate and efficient knowledge transfer. Our method constructs a common representation of the forward and inverse tasks, and leverages auxiliary forward demonstrations from novel configurations to successfully execute the corresponding inverse tasks, without any direct supervision. We show the extrapolation capabilities of our framework via ablation studies and experiments in simulated and real-world environments that require complex manipulation skills with a diverse set of objects and tools, where we outperform diffusion-based alternatives.
>
---
#### [new 057] How to Model Your Crazyflie Brushless
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决Crazyflie Brushless的建模与控制问题。工作包括建立动力学模型，验证其准确性，并用于强化学习控制器训练。**

- **链接: [https://arxiv.org/pdf/2603.05944](https://arxiv.org/pdf/2603.05944)**

> **作者:** Alexander Gräfe; Christoph Scherer; Wolfgang Hönig; Sebastian Trimpe
>
> **摘要:** The Crazyflie quadcopter is widely recognized as a leading platform for nano-quadcopter research. In early 2025, the Crazyflie Brushless was introduced, featuring brushless motors that provide around 50% more thrust compared to the brushed motors of its predecessor, the Crazyflie 2.1. This advancement has opened new opportunities for research in agile nano-quadcopter control. To support researchers utilizing this new platform, this work presents a dynamics model of the Crazyflie Brushless and identifies its key parameters. Through simulations and hardware analyses, we assess the accuracy of our model. We furthermore demonstrate its suitability for reinforcement learning applications by training an end-to-end neural network position controller and learning a backflip controller capable of executing two complete rotations with a vertical movement of just 1.8 meters. This showcases the model's ability to facilitate the learning of controllers and acrobatic maneuvers that successfully transfer from simulation to hardware. Utilizing this application, we investigate the impact of domain randomization on control performance, offering valuable insights into bridging the sim-to-real gap with the presented model. We have open-sourced the entire project, enabling users of the Crazyflie Brushless to swiftly implement and test their own controllers on an accurate simulation platform.
>
---
#### [new 058] Introducing the transitional autonomous vehicle lane-changing dataset: Empirical Experiments
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自动驾驶研究任务，旨在解决tAV在变道过程中的交互行为分析问题。通过构建高精度数据集，开展控制实验，分析tAV变道及响应动态。**

- **链接: [https://arxiv.org/pdf/2603.05716](https://arxiv.org/pdf/2603.05716)**

> **作者:** Abhinav Sharma; Zijun He; Danjue Chen
>
> **摘要:** Transitional autonomous vehicles (tAVs), which operate beyond SAE Level 1-2 automation but short of full autonomy, are increasingly sharing the road with human-driven vehicles (HDVs). As these systems interact during complex maneuvers such as lane changes, new patterns may emerge with implications for traffic stability and safety. Assessing these dynamics, particularly during mandatory lane changes, requires high-resolution trajectory data, yet datasets capturing tAV lane-changing behavior are scarce. This study introduces the North Carolina Transitional Autonomous Vehicle Lane-Changing (NC-tALC) Dataset, a high-fidelity trajectory dataset designed to characterize tAV interactions during lane-changing maneuvers. The dataset includes two controlled experimental series. In the first, tAV lane-changing experiments, a tAV executes lane changes in the presence of adaptive cruise control (ACC) equipped target vehicles, enabling analysis of lane-changing execution. In the second, tAV responding experiments, two tAVs act as followers and respond to cut-in maneuvers initiated by another tAV, enabling analysis of follower response dynamics. The dataset contains 152 trials (72 lane-changing and 80 responding trials) sampled at 20 Hz with centimeter-level RTK-GPS accuracy. The NC-tALC dataset provides a rigorous empirical foundation for evaluating tAV decision-making and interaction dynamics in controlled mandatory lane-changing scenarios.
>
---
#### [new 059] Sticky-Glance: Robust Intent Recognition for Human Robot Collaboration via Single-Glance
- **分类: cs.RO**

- **简介: 该论文属于人机协作中的意图识别任务，旨在解决多物体环境下 gaze 识别的鲁棒性问题。提出一种基于单次凝视的稳定意图识别方法，提升动态目标跟踪和静态目标选择的准确性。**

- **链接: [https://arxiv.org/pdf/2603.06121](https://arxiv.org/pdf/2603.06121)**

> **作者:** Yuzhi Lai; Shenghai Yuan; Peizheng Li; Andreas Zell
>
> **摘要:** Gaze is a valuable means of communication for impaired people with extremely limited motor capabilities. However, robust gaze-based intent recognition in multi-object environments is challenging due to gaze noise, micro-saccades, viewpoint changes, and dynamic objects. To address this, we propose an object-centric gaze grounding framework that stabilizes intent through a sticky-glance algorithm, jointly modeling geometric distance and direction trends. The inferred intent remains anchored to the object even under short glances with minimal 3 gaze samples, achieving a tracking rate of 0.94 for dynamic targets and selection accuracy of 0.98 for static targets. We further introduce a continuous shared control and multi-modal interaction paradigm, enabling high-readiness control and human-in-loop feedback, thereby reducing task duration for nearly 10 \%. Experiments across dynamic tracking, multi-perspective alignment, a baseline comparison, user studies, and ablation studies demonstrate improved robustness, efficiency, and reduced workload compared to representative baselines.
>
---
#### [new 060] From Decoupled to Coupled: Robustness Verification for Learning-based Keypoint Detection with Joint Specifications
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于关键点检测任务，解决模型对输入扰动的鲁棒性验证问题。提出一种耦合验证框架，通过联合约束所有关键点的偏差，提升验证效果。**

- **链接: [https://arxiv.org/pdf/2603.05604](https://arxiv.org/pdf/2603.05604)**

> **作者:** Xusheng Luo; Changliu Liu
>
> **备注:** 21 pages, 4 figures, 9 tables. arXiv admin note: text overlap with arXiv:2408.00117
>
> **摘要:** Keypoint detection underpins many vision tasks, including pose estimation, viewpoint recovery, and 3D reconstruction, yet modern neural models remain vulnerable to small input perturbations. Despite its importance, formal robustness verification for keypoint detectors is largely unexplored due to high-dimensional inputs and continuous coordinate outputs. We propose the first coupled robustness verification framework for heatmap-based keypoint detectors that bounds the joint deviation across all keypoints, capturing their interdependencies and downstream task requirements. Unlike prior decoupled, classification-style approaches that verify each keypoint independently and yield conservative guarantees, our method verifies collective behavior. We formulate verification as a falsification problem using a mixed-integer linear program (MILP) that combines reachable heatmap sets with a polytope encoding joint deviation constraints. Infeasibility certifies robustness, while feasibility provides counterexamples, and we prove the method is sound: if it certifies the model as robust, then the keypoint detection model is guaranteed to be robust. Experiments show that our coupled approach achieves high verified rates and remains effective under strict error thresholds where decoupled methods fail.
>
---
#### [new 061] NOVA: Next-step Open-Vocabulary Autoregression for 3D Multi-Object Tracking in Autonomous Driving
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出NOVA，解决自动驾驶中3D多目标跟踪的开放词汇问题。通过生成式时空语义建模，提升对未知目标的跟踪性能。**

- **链接: [https://arxiv.org/pdf/2603.06254](https://arxiv.org/pdf/2603.06254)**

> **作者:** Kai Luo; Xu Wang; Rui Fan; Kailun Yang
>
> **备注:** Code will be available at this https URL
>
> **摘要:** Generalizing across unknown targets is critical for open-world perception, yet existing 3D Multi-Object Tracking (3D MOT) pipelines remain limited by closed-set assumptions and ``semantic-blind'' heuristics. To address this, we propose Next-step Open-Vocabulary Autoregression (NOVA), an innovative paradigm that shifts 3D tracking from traditional fragmented distance-based matching toward generative spatio-temporal semantic modeling. NOVA reformulates 3D trajectories as structured spatio-temporal semantic sequences, enabling the simultaneous encoding of physical motion continuity and deep linguistic priors. By leveraging the autoregressive capabilities of Large Language Models (LLMs), we transform the tracking task into a principled process of next-step sequence completion. This mechanism allows the model to explicitly utilize the hierarchical structure of language space to resolve fine-grained semantic ambiguities and maintain identity consistency across complex long-range sequences through high-level commonsense reasoning. Extensive experiments on nuScenes, V2X-Seq-SPD, and KITTI demonstrate the superior performance of NOVA. Notably, on the nuScenes dataset, NOVA achieves an AMOTA of 22.41% for Novel categories, yielding a significant 20.21% absolute improvement over the baseline. These gains are realized through a compact 0.5B autoregressive model. Code will be available at this https URL.
>
---
#### [new 062] An Embodied Companion for Visual Storytelling
- **分类: cs.HC; cs.AI; cs.GR; cs.RO**

- **简介: 该论文提出一种结合绘画机器人与大语言模型的艺术协作系统，旨在解决AI在艺术创作中从工具向合作伙伴的转变问题。通过实时互动实现共同创作。**

- **链接: [https://arxiv.org/pdf/2603.05511](https://arxiv.org/pdf/2603.05511)**

> **作者:** Patrick Tresset; Markus Wulfmeier
>
> **备注:** 35 pages, 18 figures
>
> **摘要:** As artificial intelligence shifts from pure tool for delegation toward agentic collaboration, its use in the arts can shift beyond the exploration of machine autonomy toward synergistic co-creation. While our earlier robotic works utilized automation to distance the artist's intent from the final mark, we present Companion: an artistic apparatus that integrates a drawing robot with Large Language Models (LLMs) to re-center human-machine presence. By leveraging in-context learning and real-time tool use, the system engages in bidirectional interaction via speech and sketching. This approach transforms the robot from a passive executor into a playful co-creative partner capable of driving shared visual storytelling into unexpected aesthetic territories. To validate this collaborative shift, we employed the Consensual Assessment Technique (CAT) with a panel of seven art-world experts. Results confirm that the system produces works with a distinct aesthetic identity and professional exhibition merit, demonstrating the potential of AI as a highly capable artistic collaborator.
>
---
#### [new 063] Systematic Evaluation of Novel View Synthesis for Video Place Recognition
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视频位置识别任务，旨在评估合成新视角对VPR的影响。通过实验验证，发现少量合成视角能提升识别效果，视角变化大小不如添加数量和数据类型重要。**

- **链接: [https://arxiv.org/pdf/2603.05876](https://arxiv.org/pdf/2603.05876)**

> **作者:** Muhammad Zawad Mahmud; Samiha Islam; Damian Lyons
>
> **备注:** Submitted to IEEE IROS 2026
>
> **摘要:** The generation of synthetic novel views has the potential to positively impact robot navigation in several ways. In image-based navigation, a novel overhead view generated from a scene taken by a ground robot could be used to guide an aerial robot to that location. In Video Place Recognition (VPR), novel views of ground locations from the air can be added that enable a UAV to identify places seen by the ground robot, and similarly, overhead views can be used to generate novel ground views. This paper presents a systematic evaluation of synthetic novel views in VPR using five public VPR image databases and seven typical image similarity methods. We show that for small synthetic additions, novel views improve VPR recognition statistics. We find that for larger additions, the magnitude of viewpoint change is less important than the number of views added and the type of imagery in the dataset.
>
---
#### [new 064] FTSplat: Feed-forward Triangle Splatting Network
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D重建任务，旨在解决传统方法依赖优化、效率低的问题。提出FTSplat框架，直接生成连续三角面片，实现高效且兼容模拟的重建。**

- **链接: [https://arxiv.org/pdf/2603.05932](https://arxiv.org/pdf/2603.05932)**

> **作者:** Xiong Jinlin; Li Can; Shen Jiawei; Qi Zhigang; Sun Lei; Zhao Dongyang
>
> **摘要:** High-fidelity three-dimensional (3D) reconstruction is essential for robotics and simulation. While Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) achieve impressive rendering quality, their reliance on time-consuming per-scene optimization limits real-time deployment. Emerging feed-forward Gaussian splatting methods improve efficiency but often lack explicit, manifold geometry required for direct simulation. To address these limitations, we propose a feed-forward framework for triangle primitive generation that directly predicts continuous triangle surfaces from calibrated multi-view images. Our method produces simulation-ready models in a single forward pass, obviating the need for per-scene optimization or post-processing. We introduce a pixel-aligned triangle generation module and incorporate relative 3D point cloud supervision to enhance geometric learning stability and consistency. Experiments demonstrate that our method achieves efficient reconstruction while maintaining seamless compatibility with standard graphics and robotic simulators.
>
---
#### [new 065] RoboLayout: Differentiable 3D Scene Generation for Embodied Agents
- **分类: cs.AI; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出RoboLayout，解决3D场景生成中满足实体代理交互需求的问题。通过引入可达性约束和局部优化，提升场景可操作性，适用于多种代理类型。**

- **链接: [https://arxiv.org/pdf/2603.05522](https://arxiv.org/pdf/2603.05522)**

> **作者:** Ali Shamsaddinlou
>
> **摘要:** Recent advances in vision language models (VLMs) have shown strong potential for spatial reasoning and 3D scene layout generation from open-ended language instructions. However, generating layouts that are not only semantically coherent but also feasible for interaction by embodied agents remains challenging, particularly in physically constrained indoor environments. In this paper, RoboLayout is introduced as an extension of LayoutVLM that augments the original framework with agent-aware reasoning and improved optimization stability. RoboLayout integrates explicit reachability constraints into a differentiable layout optimization process, enabling the generation of layouts that are navigable and actionable by embodied agents. Importantly, the agent abstraction is not limited to a specific robot platform and can represent diverse entities with distinct physical capabilities, such as service robots, warehouse robots, humans of different age groups, or animals, allowing environment design to be tailored to the intended agent. In addition, a local refinement stage is proposed that selectively reoptimizes problematic object placements while keeping the remainder of the scene fixed, improving convergence efficiency without increasing global optimization iterations. Overall, RoboLayout preserves the strong semantic alignment and physical plausibility of LayoutVLM while enhancing applicability to agent-centric indoor scene generation, as demonstrated by experimental results across diverse scene configurations.
>
---
#### [new 066] Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对VLA模型在自动驾驶中的探索不足问题，提出Curious-VLA框架，通过两阶段设计提升探索能力，优化性能。**

- **链接: [https://arxiv.org/pdf/2603.06049](https://arxiv.org/pdf/2603.06049)**

> **作者:** Canyu Chen; Yuguang Yang; Zhewen Tan; Yizhi Wang; Ruiyi Zhan; Haiyan Liu; Xuanyao Mao; Jason Bao; Xinyue Tang; Linlin Yang; Bingchuan Sun; Yan Wang; Baochang Zhang
>
> **备注:** Accepted by CVPR2026 findings
>
> **摘要:** We identify a fundamental Narrow Policy limitation undermining the performance of autonomous VLA models, where driving Imitation Learning (IL) tends to collapse exploration and limit the potential of subsequent Reinforcement Learning (RL) stages, which often saturate prematurely due to insufficient feedback diversity. Thereby, we propose Curious-VLA, a framework that alleviates the exploit-explore dilemma through a two-stage design. During IL, we introduce a Feasible Trajectory Expansion (FTE) strategy to generate multiple physically valid trajectories and a step-wise normalized trajectory representation to adapt this diverse data. In the RL stage, we present Adaptive Diversity-Aware Sampling (ADAS) that prioritizes high-diversity samples and introduce Spanning Driving Reward (SDR) with a focal style weighting to amplify reward's value span for improving sensitivity to driving quality. On the Navsim benchmark, Curious-VLA achieves SoTA results (PDMS 90.3, EPDMS 85.4) and a Best-of-N PDMS of 94.8, demonstrating its effectiveness in unlocking the exploratory potential of VLA models. Code: this https URL.
>
---
#### [new 067] VG3S: Visual Geometry Grounded Gaussian Splatting for Semantic Occupancy Prediction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D语义占用预测任务，解决视觉主导方法中几何信息不足的问题。通过引入视觉基础模型的几何先验，提出VG3S框架提升预测精度。**

- **链接: [https://arxiv.org/pdf/2603.06210](https://arxiv.org/pdf/2603.06210)**

> **作者:** Xiaoyang Yan; Muleilan Pei; Shaojie Shen
>
> **摘要:** 3D semantic occupancy prediction has become a crucial perception task for comprehensive scene understanding in autonomous driving. While recent advances have explored 3D Gaussian splatting for occupancy modeling to substantially reduce computational overhead, the generation of high-quality 3D Gaussians relies heavily on accurate geometric cues, which are often insufficient in purely vision-centric paradigms. To bridge this gap, we advocate for injecting the strong geometric grounding capability from Vision Foundation Models (VFMs) into occupancy prediction. In this regard, we introduce Visual Geometry Grounded Gaussian Splatting (VG3S), a novel framework that empowers Gaussian-based occupancy prediction with cross-view 3D geometric grounding. Specifically, to fully exploit the rich 3D geometric priors from a frozen VFM, we propose a plug-and-play hierarchical geometric feature adapter, which can effectively transform generic VFM tokens via feature aggregation, task-specific alignment, and multi-scale restructuring. Extensive experiments on the nuScenes occupancy benchmark demonstrate that VG3S achieves remarkable improvements of 12.6% in IoU and 7.5% in mIoU over the baseline. Furthermore, we show that VG3S generalizes seamlessly across diverse VFMs, consistently enhancing occupancy prediction accuracy and firmly underscoring the immense value of integrating priors derived from powerful, pre-trained geometry-grounded VFMs.
>
---
#### [new 068] TaPD: Temporal-adaptive Progressive Distillation for Observation-Adaptive Trajectory Forecasting in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶中的轨迹预测任务，解决观测长度不固定导致的性能下降问题。提出TaPD框架，通过时间自适应的渐进式蒸馏和时序补全模块，提升不同观测长度下的预测效果。**

- **链接: [https://arxiv.org/pdf/2603.06231](https://arxiv.org/pdf/2603.06231)**

> **作者:** Mingyu Fan; Yi Liu; Hao Zhou; Deheng Qian; Mohammad Haziq Khan; Matthias Raetsch
>
> **摘要:** Trajectory prediction is essential for autonomous driving, enabling vehicles to anticipate the motion of surrounding agents to support safe planning. However, most existing predictors assume fixed-length histories and suffer substantial performance degradation when observations are variable or extremely short in real-world settings (e.g., due to occlusion or a limited sensing range). We propose TaPD (Temporal-adaptive Progressive Distillation), a unified plug-and-play framework for observation-adaptive trajectory forecasting under variable history lengths. TaPD comprises two cooperative modules: an Observation-Adaptive Forecaster (OAF) for future prediction and a Temporal Backfilling Module (TBM) for explicit reconstruction of the past. OAF is built on progressive knowledge distillation (PKD), which transfers motion pattern knowledge from long-horizon "teachers" to short-horizon "students" via hierarchical feature regression, enabling short observations to recover richer motion context. We further introduce a cosine-annealed distillation weighting scheme to balance forecasting supervision and feature alignment, improving optimization stability and cross-length consistency. For extremely short histories where implicit alignment is insufficient, TBM backfills missing historical segments conditioned on scene evolution, producing context-rich trajectories that strengthen PKD and thereby improve OAF. We employ a decoupled pretrain-reconstruct-finetune protocol to preserve real-motion priors while adapting to backfilled inputs. Extensive experiments on Argoverse 1 and Argoverse 2 show that TaPD consistently outperforms strong baselines across all observation lengths, delivers especially large gains under very short inputs, and improves other predictors (e.g., HiVT) in a plug-and-play manner. Code will be available at this https URL.
>
---
#### [new 069] Can we Trust Unreliable Voxels? Exploring 3D Semantic Occupancy Prediction under Label Noise
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于3D语义占用预测任务，解决标签噪声下的模型可靠性问题。提出DPR-Occ框架，提升噪声环境下的性能。**

- **链接: [https://arxiv.org/pdf/2603.06279](https://arxiv.org/pdf/2603.06279)**

> **作者:** Wenxin Li; Kunyu Peng; Di Wen; Junwei Zheng; Jiale Wei; Mengfei Duan; Yuheng Zhang; Rui Fan; Kailun Yang
>
> **备注:** The benchmark and source code will be made publicly available at this https URL
>
> **摘要:** 3D semantic occupancy prediction is a cornerstone of robotic perception, yet real-world voxel annotations are inherently corrupted by structural artifacts and dynamic trailing effects. This raises a critical but underexplored question: can autonomous systems safely rely on such unreliable occupancy supervision? To systematically investigate this issue, we establish OccNL, the first benchmark dedicated to 3D occupancy under occupancy-asymmetric and dynamic trailing noise. Our analysis reveals a fundamental domain gap: state-of-the-art 2D label noise learning strategies collapse catastrophically in sparse 3D voxel spaces, exposing a critical vulnerability in existing paradigms. To address this challenge, we propose DPR-Occ, a principled label noise-robust framework that constructs reliable supervision through dual-source partial label reasoning. By synergizing temporal model memory with representation-level structural affinity, DPR-Occ dynamically expands and prunes candidate label sets to preserve true semantics while suppressing noise propagation. Extensive experiments on SemanticKITTI demonstrate that DPR-Occ prevents geometric and semantic collapse under extreme corruption. Notably, even at 90% label noise, our method achieves significant performance gains (up to 2.57% mIoU and 13.91% IoU) over existing label noise learning baselines adapted to the 3D occupancy prediction task. By bridging label noise learning and 3D perception, OccNL and DPR-Occ provide a reliable foundation for safety-critical robotic perception in dynamic environments. The benchmark and source code will be made publicly available at this https URL.
>
---
#### [new 070] Spatial Calibration of Diffuse LiDARs
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于LiDAR与RGB图像的标定任务，解决 diffuse LiDAR 与 RGB 图像的 spatial calibration 问题，通过恢复像素级响应图实现两者对齐与融合。**

- **链接: [https://arxiv.org/pdf/2603.06531](https://arxiv.org/pdf/2603.06531)**

> **作者:** Nikhil Behari; Ramesh Raskar
>
> **摘要:** Diffuse direct time-of-flight LiDARs report per-pixel depth histograms formed by aggregating photon returns over a wide instantaneous field of view, violating the single-ray assumption behind standard LiDAR-RGB calibration. We present a simple spatial calibration procedure that estimates, for each diffuse LiDAR pixel, its footprint (effective support region) and relative spatial sensitivity in a co-located RGB image plane. Using a scanned retroreflective patch with background subtraction, we recover per-pixel response maps that provide an explicit LiDAR-to-RGB correspondence for cross-modal alignment and fusion. We demonstrate the method on the ams OSRAM TMF8828.
>
---
#### [new 071] Transforming Omnidirectional RGB-LiDAR data into 3D Gaussian Splatting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D重建任务，解决从废弃的RGB-LiDAR数据生成高质量数字孪生的问题。通过优化数据处理流程，提升3DGS效果。**

- **链接: [https://arxiv.org/pdf/2603.06061](https://arxiv.org/pdf/2603.06061)**

> **作者:** Semin Bae; Hansol Lim; Jongseong Brad Choi
>
> **备注:** This work has been submitted to the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) for possible publication
>
> **摘要:** The demand for large-scale digital twins is rapidly growing in robotics and autonomous driving. However, constructing these environments with 3D Gaussian Splatting (3DGS) usually requires expensive, purpose-built data collection. Meanwhile, deployed platforms routinely collect extensive omnidirectional RGB and LiDAR logs, but a significant portion of these sensor data is directly discarded or strictly underutilized due to transmission constraints and the lack of scalable reuse pipeline. In this paper, we present an omnidirectional RGB-LiDAR reuse pipeline that transforms these archived logs into robust initialization assets for 3DGS. Direct conversion of such raw logs introduces practical bottlenecks: inherent non-linear distortion leads to unreliable Structure-from-Motion (SfM) tracking, and dense, unorganized LiDAR clouds cause computational overhead during 3DGS optimization. To overcome these challenges, our pipeline strategically integrates an ERP-to-cubemap conversion module for deterministic spatial anchoring, alongside PRISM-a color stratified downsampling strategy. By bridging these multi-modal inputs via Fast Point Feature Histograms (FPFH) based global registration and Iterative Closest Point (ICP), our pipeline successfully repurposes a considerable fraction of discarded data into usable SfM geometry. Furthermore, our LiDAR-reinforced initialization consistently enhances the final 3DGS rendering fidelity in structurally complex scenes compared to vision-only baselines. Ultimately, this work provides a deterministic workflow for creating simulation-grade digital twins from standard archived sensor logs.
>
---
#### [new 072] BEVLM: Distilling Semantic Knowledge from LLMs into Bird's-Eye View Representations
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决多视角视觉处理与语义理解不一致的问题。通过BEVLM框架，将LLMs的语义知识注入BEV表示，提升空间推理与驾驶性能。**

- **链接: [https://arxiv.org/pdf/2603.06576](https://arxiv.org/pdf/2603.06576)**

> **作者:** Thomas Monninger; Shaoyuan Xie; Qi Alfred Chen; Sihao Ding
>
> **备注:** 4 figures, 6 tables in the main paper, 32 pages in total
>
> **摘要:** The integration of Large Language Models (LLMs) into autonomous driving has attracted growing interest for their strong reasoning and semantic understanding abilities, which are essential for handling complex decision-making and long-tail scenarios. However, existing methods typically feed LLMs with tokens from multi-view and multi-frame images independently, leading to redundant computation and limited spatial consistency. This separation in visual processing hinders accurate 3D spatial reasoning and fails to maintain geometric coherence across views. On the other hand, Bird's-Eye View (BEV) representations learned from geometrically annotated tasks (e.g., object detection) provide spatial structure but lack the semantic richness of foundation vision encoders. To bridge this gap, we propose BEVLM, a framework that connects a spatially consistent and semantically distilled BEV representation with LLMs. Through extensive experiments, we show that BEVLM enables LLMs to reason more effectively in cross-view driving scenes, improving accuracy by 46%, by leveraging BEV features as unified inputs. Furthermore, by distilling semantic knowledge from LLMs into BEV representations, BEVLM significantly improves closed-loop end-to-end driving performance by 29% in safety-critical scenarios.
>
---
## 更新

#### [replaced 001] Safe Autonomous Lane Changing: Planning with Dynamic Risk Fields and Time-Varying Convex Space Generation
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于自动驾驶轨迹规划任务，解决复杂场景下的安全变道问题。通过构建动态风险场和时变凸可行空间，结合约束iLQR算法，实现安全、高效、平顺的轨迹生成。**

- **链接: [https://arxiv.org/pdf/2511.22829](https://arxiv.org/pdf/2511.22829)**

> **作者:** Yijun Lu; Zhihao Lin; Zhen Tian
>
> **摘要:** This paper presents a novel trajectory planning pipeline for complex driving scenarios like autonomous lane changing, by integrating risk-aware planning with guaranteed collision avoidance into a unified optimization framework. We first construct a dynamic risk fields (DRF) that captures both the static and dynamic collision risks from surrounding vehicles. Then, we develop a rigorous strategy for generating time-varying convex feasible spaces that ensure kinematic feasibility and safety requirements. The trajectory planning problem is formulated as a finite-horizon optimal control problem and solved using a constrained iterative Linear Quadratic Regulator (iLQR) algorithm that jointly optimizes trajectory smoothness, control effort, and risk exposure while maintaining strict feasibility. Extensive simulations demonstrate that our method outperforms traditional approaches in terms of safety and efficiency, achieving collision-free trajectories with shorter lane-changing distances (28.59 m) and times (2.84 s) while maintaining smooth and comfortable acceleration patterns. In dense roundabout environments the planner further demonstrates robust adaptability, producing larger safety margins, lower jerk, and superior curvature smoothness compared with APF, MPC, and RRT based baselines. These results confirm that the integrated DRF with convex feasible space and constrained iLQR solver provides a balanced solution for safe, efficient, and comfortable trajectory generation in dynamic and interactive traffic scenarios.
>
---
#### [replaced 002] Large-Language-Model-Guided State Estimation for Partially Observable Task and Motion Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人任务与运动规划领域，解决部分可观测环境下的规划问题。通过引入大语言模型引导的常识知识，提升状态估计效率，减少规划时间。**

- **链接: [https://arxiv.org/pdf/2603.03704](https://arxiv.org/pdf/2603.03704)**

> **作者:** Yoonwoo Kim; Raghav Arora; Roberto Martín-Martín; Peter Stone; Ben Abbatematteo; Yoonchang Sung
>
> **摘要:** Robot planning in partially observable environments, where not all objects are known or visible, is a challenging problem, as it requires reasoning under uncertainty through partially observable Markov decision processes. During the execution of a computed plan, a robot may unexpectedly observe task-irrelevant objects, which are typically ignored by naive planners. In this work, we propose incorporating two types of common-sense knowledge: (1) certain objects are more likely to be found in specific locations; and (2) similar objects are likely to be co-located, while dissimilar objects are less likely to be found together. Manually engineering such knowledge is complex, so we explore leveraging the powerful common-sense reasoning capabilities of large language models (LLMs). Our planning and execution framework, CoCo-TAMP, introduces a hierarchical state estimation that uses LLM-guided information to shape the belief over task-relevant objects, enabling efficient solutions to long-horizon task and motion planning problems. In experiments, CoCo-TAMP achieves an average reduction of 62.7% in planning and execution time in simulation, and 72.6% in real-world demonstrations, compared to a baseline that does not incorporate either type of common-sense knowledge.
>
---
#### [replaced 003] Indicating Robot Vision Capabilities with Augmented Reality
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机协作任务，旨在解决人类对机器人视野认知不准确的问题。通过设计AR视觉指示器，提升人类对机器人视觉能力的正确理解。**

- **链接: [https://arxiv.org/pdf/2511.03550](https://arxiv.org/pdf/2511.03550)**

> **作者:** Hong Wang; Ridhima Phatak; James Ocampo; Zhao Han
>
> **摘要:** Research indicates that humans can mistakenly assume that robots and humans have the same field of view, possessing an inaccurate mental model of robots. This misperception may lead to failures during human-robot collaboration tasks where robots might be asked to complete impossible tasks about out-of-view objects. The issue is more severe when robots do not have a chance to scan the scene to update their world model while focusing on assigned tasks. To help align humans' mental models of robots' vision capabilities, we propose four field-of-view indicators in augmented reality and conducted a human-subjects experiment (N=41) to evaluate them in a collaborative assembly task regarding accuracy, confidence, task efficiency, and workload. These indicators span a spectrum of positions: two at robot's eye and head space -- deepening eye socket and adding blocks to two sides of the eyes (i.e., egocentric), and two anchoring in the robot's task space -- adding extended blocks from the sides of eyes to the table and placing blocks directly on the tables (i.e., allocentric). Results showed that, when placed directly in the task space, the allocentric indicator yields the highest accuracy, although with a delay in interpreting the robot's field of view. When placed at the robot's eyes, the egocentric indicator of deeper eye sockets, possible for physical alteration, also increased accuracy. In all indicators, participants' confidence was high while cognitive load remained low. Finally, we contribute six guidelines for practitioners to apply our augmented reality indicators or physical alterations to align humans' mental models with robots' vision capabilities.
>
---
#### [replaced 004] VISO: Robust Underwater Visual-Inertial-Sonar SLAM with Photometric Rendering for Dense 3D Reconstruction
- **分类: cs.RO**

- **简介: 该论文属于水下SLAM任务，旨在解决水下视觉定位不准和三维重建精度低的问题。提出VISO系统，融合视觉、惯性与声呐数据，提升定位精度与重建质量。**

- **链接: [https://arxiv.org/pdf/2601.01144](https://arxiv.org/pdf/2601.01144)**

> **作者:** Shu Pan; Simon Archieri; Ahmet Cinar; Jonatan Scharff Willners; Ignacio Carlucho; Yvan Petillot
>
> **摘要:** Visual challenges in underwater environments significantly hinder the accuracy of vision-based localisation and the high-fidelity dense reconstruction. In this paper, we propose VISO, a robust underwater SLAM system that fuses a stereo camera, an inertial measurement unit (IMU), and a 3D sonar to achieve accurate 6-DoF localisation and enable efficient dense 3D reconstruction with high photometric fidelity. We introduce a coarse-to-fine online calibration approach for extrinsic parameters estimation between the 3D sonar and the camera. Additionally, a photometric rendering strategy is proposed for the 3D sonar point cloud to enrich the sonar map with visual information. Extensive experiments in a laboratory tank and an open lake demonstrate that VISO surpasses current state-of-the-art underwater and visual-based SLAM algorithms in terms of localisation robustness and accuracy, while also exhibiting real-time dense 3D reconstruction performance comparable to the offline dense mapping method.
>
---
#### [replaced 005] AURASeg: Attention-guided Upsampling with Residual-Assistive Boundary Refinement for Onboard Robot Drivable-Area Segmentation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人可行驶区域分割任务，解决边界精度不足问题。提出AURASeg框架，结合残差边界精修和注意力上采样，提升分割效果并支持边缘部署。**

- **链接: [https://arxiv.org/pdf/2510.21536](https://arxiv.org/pdf/2510.21536)**

> **作者:** Narendhiran Vijayakumar; Sridevi. M
>
> **备注:** 6 pages, 4 figures, 4 tables
>
> **摘要:** Free space ground segmentation is essential to navigate autonomous robots, recognize drivable zones, and traverse efficiently. Fine-grained features remain challenging for existing segmentation models, particularly for robots in indoor, outdoor and road-scene environments. These difficulties arise from ineffective multi-scale processing, sub-optimal boundary refinement, and limited feature representation. To address this, we propose Attention-guided Upsampling with Residual-Assistive Boundary Refinement (AURASeg), a ground-plane drivable area segmentation framework designed to improve boundary precision while preserving strong region accuracy under edge-deployment constraints. Built on ResNet backbone, we propose (i) a Residual Boundary Refinement Module (RBRM) that enhances edge delineation through boundary-assistive feature refinement, and (ii) Attention Progressive Upsampling Decoder (APUD) blocks that fuse multi-level features using residual fusion of attention modules; additionally, we integrate (iii) a lightweight ASPPLite module to capture multi-scale context with minimal overhead. Extensive experiments on CARL-D, the Ground Mobile Robot Perception (GMRPD) dataset, and a custom Gazebo indoor dataset show that AURASeg consistently outperforms strong baselines, with notable gains in boundary metrics. Finally, we demonstrate on-device deployment on a Jetson Nano powered Kobuki TurtleBot, validating practical edge-inference feasibility. Code is omitted for anonymity and will be released upon acceptance.
>
---
#### [replaced 006] Robustness-Aware Tool Selection and Manipulation Planning with Learned Energy-Informed Guidance
- **分类: cs.RO**

- **简介: 该论文属于机器人工具使用任务，解决鲁棒性不足的问题。通过能量驱动的指标，联合选择工具并规划抗干扰的操作轨迹，提升操作鲁棒性。**

- **链接: [https://arxiv.org/pdf/2506.03362](https://arxiv.org/pdf/2506.03362)**

> **作者:** Yifei Dong; Yan Zhang; Sylvain Calinon; Florian T. Pokorny
>
> **备注:** IEEE International Conference on Robotics and Automation (ICRA), 2026
>
> **摘要:** Humans subconsciously choose robust ways of selecting and using tools, for example, choosing a ladle over a flat spatula to serve meatballs. However, robustness under external disturbances remains underexplored in robotic tool-use planning. This paper presents a robustness-aware method that jointly selects tools and plans contact-rich manipulation trajectories, explicitly optimizing for robustness against disturbances. At the core of our method is an energy-based robustness metric that guides the planner toward robust manipulation behaviors. We formulate a hierarchical optimization pipeline that first identifies a tool and configuration that optimizes robustness, and then plans a corresponding manipulation trajectory that maintains robustness throughout execution. We evaluate our method across three representative tool-use tasks. Simulation and real-world results demonstrate that our method consistently selects robust tools and generates disturbance-resilient manipulation plans.
>
---
#### [replaced 007] Bridging Simulation and Usability: A User-Friendly Framework for Scenario Generation in CARLA
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶仿真任务，旨在解决场景生成工具门槛高的问题。提出一个无代码框架，简化场景创建与管理，提升仿真验证的可访问性。**

- **链接: [https://arxiv.org/pdf/2507.19883](https://arxiv.org/pdf/2507.19883)**

> **作者:** Ahmed Abouelazm; Mohammad Mahmoud; Conrad Walter; Oleksandr Shchetsura; Erne Hussong; Helen Gremmelmaier; J. Marius Zöllner
>
> **备注:** Paper is accepted in IEEE International Automated Vehicle Validation Conference (IAVVC 2025)
>
> **摘要:** Autonomous driving promises safer roads, reduced congestion, and improved mobility, yet validating these systems across diverse conditions remains a major challenge. Real-world testing is expensive, time-consuming, and sometimes unsafe, making large-scale validation impractical. In contrast, simulation environments offer a scalable and cost-effective alternative for rigorous verification and validation. A critical component of the validation process is scenario generation, which involves designing and configuring traffic scenarios to evaluate autonomous systems' responses to various events and uncertainties. However, existing scenario generation tools often require programming knowledge, limiting accessibility for non-technical users. To address this limitation, we present an interactive, no-code framework for scenario generation. Our framework features a graphical interface that enables users to create, modify, save, load, and execute scenarios without needing coding expertise or detailed simulation knowledge. Unlike script-based tools such as Scenic or ScenarioRunner, our approach lowers the barrier to entry and supports a broader user base. Central to our framework is a graph-based scenario representation that facilitates structured management, supports both manual and automated generation, and enables integration with deep learning-based scenario and behavior generation methods. In automated mode, the framework can randomly sample parameters such as actor types, behaviors, and environmental conditions, allowing the generation of diverse and realistic test datasets. By simplifying the scenario generation process, this framework supports more efficient testing workflows and increases the accessibility of simulation-based validation for researchers, engineers, and policymakers.
>
---
#### [replaced 008] Decision-Driven Semantic Object Exploration for Legged Robots via Confidence-Calibrated Perception and Topological Subgoal Selection
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人导航任务，解决腿式机器人在开放环境中基于语义的探索问题。通过语义证据仲裁、拓扑记忆和子目标选择，提升探索决策的准确性与效果。**

- **链接: [https://arxiv.org/pdf/2509.20739](https://arxiv.org/pdf/2509.20739)**

> **作者:** Guoyang Zhao; Yudong Li; Weiqing Qi; Kai Zhang; Bonan Liu; Kai Chen; Haoang Li; Jun Ma
>
> **摘要:** Conventional navigation pipelines for legged robots remain largely geometry-centric, relying on dense SLAM representations that are fragile under rapid motion and offer limited support for semantic decision making in open-world exploration. In this work, we focus on decision-driven semantic object exploration, where the primary challenge is not map consistency but how noisy and heterogeneous semantic observations can be transformed into stable and executable exploration decisions. We propose a vision-based approach that explicitly addresses this problem through confidence-calibrated semantic evidence arbitration, a controlled-growth semantic topological memory, and a semantic utility-driven subgoal selection mechanism. These components enable the robot to accumulate task-relevant semantic knowledge over time and select exploration targets that balance semantic relevance, reliability, and reachability, without requiring dense geometric reconstruction. Extensive experiments in both simulation and real-world environments demonstrate that the proposed mechanisms consistently improve the quality of semantic decision inputs, subgoal selection accuracy, and overall exploration performance on legged robots.
>
---
#### [replaced 009] ExpReS-VLA: Specializing Vision-Language-Action Models Through Experience Replay and Retrieval
- **分类: cs.RO**

- **简介: 该论文提出ExpReS-VLA，解决VLA模型在特定环境下的适应问题，通过经验回放和检索增强实现快速微调，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2511.06202](https://arxiv.org/pdf/2511.06202)**

> **作者:** Shahram Najam Syed; Yatharth Ahuja; Arthur Jakobsson; Jeff Ichnowski
>
> **备注:** 8 pages, 4 figures, 3 tables, accepted to International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Vision-Language-Action (VLA) models like OpenVLA demonstrate impressive zero-shot generalization across robotic manipulation tasks but struggle to adapt to specific deployment environments where consistent high performance on a limited set of tasks is more valuable than broad generalization. We present EXPierence replayed, REtrieval augmented, Specialized VLA (ExpReS-VLA), a method that enables rapid on-device adaptation of pre-trained VLAs to target domains while preventing catastrophic forgetting through compressed experience replay and retrieval-augmented generation. Our approach maintains a memory-efficient buffer by storing extracted embeddings from OpenVLA's frozen vision backbone, reducing storage requirements by 97% compared to raw image-action pairs. During deployment, ExpReS-VLA retrieves the $k$ most similar past experiences using cosine similarity to augment training batches, while a prioritized experience replay buffer preserves recently successful trajectories. To leverage failed attempts, we introduce Thresholded Hybrid Contrastive Loss (THCL), enabling the model to learn from both successful and unsuccessful demonstrations. Experiments on the LIBERO benchmark show improvements from 82.6% to 93.1% on spatial reasoning and 61% to 72.3% on long-horizon tasks over base OpenVLA, with gains across architectures including $\pi_0$ (+3.2 points) and OpenVLA-OFT (+1.7 points). Physical robot experiments across five tasks demonstrate 98% success on both in-distribution and out-of-distribution conditions, improving from 84.7% and 32% respectively for naive fine-tuning. Adaptation completes in 31 seconds using 12 demonstrations on a single RTX 5090.
>
---
#### [replaced 010] C*: A Coverage Path Planning Algorithm for Unknown Environments using Rapidly Covering Graphs
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于路径规划任务，解决未知环境下的全覆盖问题。提出C*算法，通过RCG构建高效覆盖路径，提升覆盖率和效率。**

- **链接: [https://arxiv.org/pdf/2505.13782](https://arxiv.org/pdf/2505.13782)**

> **作者:** Zongyuan Shen; James P. Wilson; Shalabh Gupta
>
> **摘要:** The paper presents a novel sample-based algorithm, called C*, for real-time coverage path planning (CPP) of unknown environments. C* is built upon the concept of a Rapidly Covering Graph (RCG), which is incrementally constructed during robot navigation via progressive sampling of the search space. By using efficient sampling and pruning techniques, the RCG is constructed to be a minimum-sufficient graph, where its nodes and edges form the potential waypoints and segments of the coverage trajectory, respectively. The RCG tracks the coverage progress, generates the coverage trajectory and helps the robot to escape from the dead-end situations. To minimize coverage time, C* produces the desired back-and-forth coverage pattern, while adapting to the TSP-based optimal coverage of local isolated regions, called coverage holes, which are surrounded by obstacles and covered regions. It is analytically proven that C* provides complete coverage of unknown environments. The algorithmic simplicity and low computational complexity of C* make it easy to implement and suitable for real-time on-board applications. The performance of C* is validated by 1) extensive high-fidelity simulations and 2) laboratory experiments using an autonomous robot. C* yields near optimal trajectories, and a comparative evaluation with seven existing CPP methods demonstrates significant improvements in performance in terms of coverage time, number of turns, trajectory length, and overlap ratio, while preventing the formation of coverage holes. Finally, C* is comparatively evaluated on two different CPP applications using 1) energy-constrained robots and 2) multi-robot teams.
>
---
#### [replaced 011] GLIDE: A Coordinated Aerial-Ground Framework for Search and Rescue in Unknown Environments
- **分类: cs.RO**

- **简介: 该论文属于搜索与救援任务，解决未知环境中快速定位受害者和安全导航问题。通过协同无人机与地面车实现目标引导和地形侦察。**

- **链接: [https://arxiv.org/pdf/2509.14210](https://arxiv.org/pdf/2509.14210)**

> **作者:** Seth Farrell; Chenghao Li; Hesam Mojtahedi; Henrik I. Christensen
>
> **摘要:** We present a cooperative aerial-ground search-and-rescue (SAR) framework that pairs two unmanned aerial vehicles (UAVs) with an unmanned ground vehicle (UGV) to achieve rapid victim localization and obstacle-aware navigation in unknown environments. We dub this framework Guided Long-horizon Integrated Drone Escort (GLIDE), highlighting the UGV's reliance on UAV guidance for long-horizon planning. In our framework, a goal-searching UAV executes real-time onboard victim detection and georeferencing to nominate goals for the ground platform, while a terrain-scouting UAV flies ahead of the UGV's planned route to provide mid-level traversability updates. The UGV fuses aerial cues with local sensing to perform time-efficient A* planning and continuous replanning as information arrives. Additionally, we present a hardware demonstration (using a GEM e6 golf cart as the UGV and two X500 UAVs) to evaluate end-to-end SAR mission performance and include simulation ablations to assess the planning stack in isolation from detection. Empirical results demonstrate that explicit role separation across UAVs, coupled with terrain scouting and guided planning, improves reach time and navigation safety in time-critical SAR missions.
>
---
#### [replaced 012] Language Conditioning Improves Accuracy of Aircraft Goal Prediction in Non-Towered Airspace
- **分类: cs.RO**

- **简介: 该论文属于航空意图预测任务，解决非塔台空域中自主飞行器的轨迹预测问题。通过融合语言理解和空间推理，提升目标位置预测准确性。**

- **链接: [https://arxiv.org/pdf/2509.14063](https://arxiv.org/pdf/2509.14063)**

> **作者:** Sundhar Vinodh Sangeetha; Chih-Yuan Chiu; Sarah H.Q. Li; Shreyas Kousik
>
> **备注:** The last two authors advised equally. Accepted to the 2026 IEEE International Conference on Robotics and Automation. 8 pages, 6 figures
>
> **摘要:** Autonomous aircraft must safely operate in non-towered airspace, where coordination relies on voice-based communication among human pilots. Safe operation requires an aircraft to predict the intent, and corresponding goal location, of other aircraft. This paper introduces a multimodal framework for aircraft goal prediction that integrates natural language understanding with spatial reasoning to improve autonomous decision-making in such environments. We leverage automatic speech recognition and large language models to transcribe and interpret pilot radio calls, identify aircraft, and extract discrete intent labels. These intent labels are fused with observed trajectories to condition a temporal convolutional network and Gaussian mixture model for probabilistic goal prediction. Our method significantly reduces goal prediction error compared to baselines that rely solely on motion history, demonstrating that language-conditioned prediction increases prediction accuracy. Experiments on a real-world dataset from a non-towered airport validate the approach and highlight its potential to enable socially aware, language-conditioned robotic motion planning.
>
---
#### [replaced 013] Taxonomy-aware Dynamic Motion Generation on Hyperbolic Manifolds
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人运动生成任务，旨在解决运动与层次结构脱节的问题。通过引入超球面流形和分类先验，提出GPHDM模型，生成结构化且物理一致的运动。**

- **链接: [https://arxiv.org/pdf/2509.21281](https://arxiv.org/pdf/2509.21281)**

> **作者:** Luis Augenstein; Noémie Jaquier; Tamim Asfour; Leonel Rozo
>
> **备注:** Accepted for publication in IEEE Conference on Robotics and Automation (ICRA), 8 pages, 6 figures, 1 table
>
> **摘要:** Human-like motion generation for robots often draws inspiration from biomechanical studies, which often categorize complex human motions into hierarchical taxonomies. While these taxonomies provide rich structural information about how movements relate to one another, this information is frequently overlooked in motion generation models, leading to a disconnect between the generated motions and their underlying hierarchical structure. This paper introduces the \ac{gphdm}, a novel approach that learns latent representations preserving both the hierarchical structure of motions and their temporal dynamics to ensure physical consistency. Our model achieves this by extending the dynamics prior of the Gaussian Process Dynamical Model (GPDM) to the hyperbolic manifold and integrating it with taxonomy-aware inductive biases. Building on this geometry- and taxonomy-aware frameworks, we propose three novel mechanisms for generating motions that are both taxonomically-structured and physically-consistent: two probabilistic recursive approaches and a method based on pullback-metric geodesics. Experiments on generating realistic motion sequences on the hand grasping taxonomy show that the proposed GPHDM faithfully encodes the underlying taxonomy and temporal dynamics, and it generates novel physically-consistent trajectories.
>
---
#### [replaced 014] AIM-SLAM: Dense Monocular SLAM via Adaptive and Informative Multi-View Keyframe Prioritization with Foundation Model
- **分类: cs.RO**

- **简介: 该论文提出AIM-SLAM，解决单目SLAM中密集重建的视图选择问题。通过自适应多视角关键帧优先级策略，提升定位与重建精度。**

- **链接: [https://arxiv.org/pdf/2603.05097](https://arxiv.org/pdf/2603.05097)**

> **作者:** Jinwoo Jeon; Dong-Uk Seo; Eungchang Mason Lee; Hyun Myung
>
> **备注:** 8 pages
>
> **摘要:** Recent advances in geometric foundation models have emerged as a promising alternative for addressing the challenge of dense reconstruction in monocular visual simultaneous localization and mapping (SLAM). Although geometric foundation models enable SLAM to leverage variable input views, the previous methods remain confined to two-view pairs or fixed-length inputs without sufficient deliberation of geometric context for view selection. To tackle this problem, we propose AIM-SLAM, a dense monocular SLAM framework that exploits an adaptive and informative multi-view keyframe prioritization with dense pointmap predictions from visual geometry grounded transformer (VGGT). Specifically, we introduce the selective information- and geometric-aware multi-view adaptation (SIGMA) module, which employs voxel overlap and information gain to retrieve a candidate set of keyframes and adaptively determine its size. Furthermore, we formulate a joint multi-view Sim(3) optimization that enforces consistent alignment across selected views, substantially improving pose estimation accuracy. The effectiveness of AIM-SLAM is demonstrated on real-world datasets, where it achieves state-of-the-art performance in both pose estimation and dense reconstruction. Our system supports ROS integration, with code is available at this https URL.
>
---
#### [replaced 015] OA-Bug: An Olfactory-Auditory Augmented Bug Algorithm for Swarm Robots in a Denied Environment
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于自主机器人协同探索任务，旨在解决在无GNSS等辅助条件下的环境搜索问题。提出OA-Bug算法，结合嗅觉与听觉信号提升搜索效率。**

- **链接: [https://arxiv.org/pdf/2209.14007](https://arxiv.org/pdf/2209.14007)**

> **作者:** Siqi Tan; Xiaoya Zhang; Jingyao Li; Ruitao Jing; Mufan Zhao; Yang Liu; Quan Quan
>
> **备注:** 7 pages, 6 figures, accepted by 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Searching in a denied environment is challenging for swarm robots as no assistance from GNSS, mapping, data sharing, and central processing is allowed. However, using olfactory and auditory signals to cooperate like animals could be an important way to improve the collaboration of swarm robots. In this paper, an Olfactory-Auditory augmented Bug algorithm (OA-Bug) is proposed for a swarm of autonomous robots to explore a denied environment. A simulation environment is built to measure the performance of OA-Bug. The coverage of the search task can reach 96.93% using OA-Bug, which is significantly improved compared with a similar algorithm, SGBA. Furthermore, experiments are conducted on real swarm robots to prove the validity of OA-Bug. Results show that OA-Bug can improve the performance of swarm robots in a denied environment. Video: this https URL.
>
---
#### [replaced 016] ROSflight 2.0: Lean ROS 2-Based Autopilot for Unmanned Aerial Vehicles
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于无人机自主控制任务，旨在降低UAV研究门槛。通过升级ROS1至ROS2，提升系统模块化与可用性，支持硬件与仿真环境，实现高效控制。**

- **链接: [https://arxiv.org/pdf/2510.00995](https://arxiv.org/pdf/2510.00995)**

> **作者:** Jacob Moore; Phil Tokumaru; Ian Reid; Brandon Sutherland; Joseph Ritchie; Gabe Snow; Tim McLain
>
> **备注:** Submitted to the 2026 International Conference on Unmanned Aerial Systems
>
> **摘要:** ROSflight is a lean, open-source autopilot ecosystem for unmanned aerial vehicles (UAVs). Designed by researchers for researchers, it is built to lower the barrier to entry to UAV research and accelerate the transition from simulation to hardware experiments by maintaining a lean (not full-featured), well-documented, and modular codebase. This publication builds on previous treatments and describes significant additions to the architecture that improve the modularity and usability of ROSflight, including the transition from ROS 1 to ROS 2, supported hardware, low-level actuator mixing, and the simulation environment. We believe that these changes improve the usability of ROSflight and enable ROSflight to accelerate research in areas like advanced-air mobility. Hardware results are provided, showing that ROSflight is able to control a multirotor over a serial connection at 400 Hz while closing all control loops on the companion computer.
>
---
#### [replaced 017] EchoVLA: Synergistic Declarative Memory for VLA-Driven Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文提出EchoVLA，解决移动操作中记忆与推理不足的问题。通过引入场景和情节记忆，提升视觉-语言-动作模型的性能。**

- **链接: [https://arxiv.org/pdf/2511.18112](https://arxiv.org/pdf/2511.18112)**

> **作者:** Min Lin; Xiwen Liang; Bingqian Lin; Liu Jingzhi; Zijian Jiao; Kehan Li; Yu Sun; Weijia Liufu; Yuhan Ma; Yuecheng Liu; Shen Zhao; Yuzheng Zhuang; Xiaodan Liang
>
> **摘要:** Recent progress in Vision-Language-Action (VLA) models has enabled embodied agents to interpret multimodal instructions and perform complex tasks. However, existing VLAs are mostly confined to short-horizon, table-top manipulation, lacking the memory and reasoning capability required for mobile manipulation, where agents must coordinate navigation and manipulation under changing spatial contexts. In this work, we present EchoVLA, a memory-aware VLA model for mobile manipulation. EchoVLA incorporates a synergistic declarative memory inspired by the human brain, consisting of a scene memory that maintains a collection of spatial-semantic maps and an episodic memory that stores task-level experiences with multimodal contextual features. The two memories are individually stored, updated, and retrieved based on current observations, task history, and instructions, and their retrieved representations are fused via coarse- and fine-grained attention to guide base-arm diffusion policies. To support large-scale training, we further introduce MoMani, an automated benchmark that generates expert-level trajectories through multimodal large language model (MLLM)-guided planning and feedback-driven refinement, supplemented with real-robot demonstrations. Comprehensive simulated and real-world results demonstrate that EchoVLA substantially improves overall performance, e.g., it achieves the highest success rates of 0.52 on manipulation/navigation tasks and 0.31 on mobile manipulation tasks in simulation, exceeding the strong baseline $\pi_{0.5}$ by +0.20 and +0.11, respectively.
>
---
#### [replaced 018] Phys4D: Fine-Grained Physics-Consistent 4D Modeling from Video Diffusion
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出Phys4D，解决视频生成中物理一致性不足的问题，通过三阶段训练提升4D场景的物理合理性。**

- **链接: [https://arxiv.org/pdf/2603.03485](https://arxiv.org/pdf/2603.03485)**

> **作者:** Haoran Lu; Shang Wu; Jianshu Zhang; Maojiang Su; Guo Ye; Chenwei Xu; Lie Lu; Pranav Maneriker; Fan Du; Manling Li; Zhaoran Wang; Han Liu
>
> **摘要:** Recent video diffusion models have achieved impressive capabilities as large-scale generative world models. However, these models often struggle with fine-grained physical consistency, exhibiting physically implausible dynamics over time. In this work, we present \textbf{Phys4D}, a pipeline for learning physics-consistent 4D world representations from video diffusion models. Phys4D adopts \textbf{a three-stage training paradigm} that progressively lifts appearance-driven video diffusion models into physics-consistent 4D world representations. We first bootstrap robust geometry and motion representations through large-scale pseudo-supervised pretraining, establishing a foundation for 4D scene modeling. We then perform physics-grounded supervised fine-tuning using simulation-generated data, enforcing temporally consistent 4D dynamics. Finally, we apply simulation-grounded reinforcement learning to correct residual physical violations that are difficult to capture through explicit supervision. To evaluate fine-grained physical consistency beyond appearance-based metrics, we introduce a set of \textbf{4D world consistency evaluation} that probe geometric coherence, motion stability, and long-horizon physical plausibility. Experimental results demonstrate that Phys4D substantially improves fine-grained spatiotemporal and physical consistency compared to appearance-driven baselines, while maintaining strong generative performance. Our project page is available at this https URL
>
---
#### [replaced 019] Integrated Hierarchical Decision-Making in Inverse Kinematic Planning and Control
- **分类: cs.RO**

- **简介: 该论文属于机器人逆运动学规划与控制任务，解决复杂非线性决策问题。提出一种高效集成的稀疏分层非线性规划方法，提升计算效率与准确性。**

- **链接: [https://arxiv.org/pdf/2412.01324](https://arxiv.org/pdf/2412.01324)**

> **作者:** Kai Pfeiffer; Quan Zhang; Yuqing Chen; Gordon Boateng; Yuquan Wang; Vincent Bonnet; Aberrahmane Kheddar
>
> **摘要:** This work presents a novel and efficient non-linear programming framework that tightly integrates hierarchical decision-making with inverse kinematic planning and control. Decision-making plays a central role in many aspects of robotics, from sparse inverse kinematic control with a minimal number of joints, to inverse kinematic planning while simultaneously selecting a discrete end-effector location from multiple candidates. Current approaches often rely on heavy computations using mixed-integer non-linear programming, separate decision-making from inverse kinematics (some times approximated by reachability methods), or employ efficient but less accurate $\ell_1$-norm formulations of linear sparse programming, without addressing the underlying non-linear problem formulations. In contrast, the proposed sparse hierarchical non-linear programming solver is efficient, versatile, and accurate by exploiting sparse hierarchical structure and leveraging the rarely used $\ell_0$-norm in robotics. The solver efficiently addresses complex non-linear hierarchical decision-making problems, such as inverse kinematic planning with simultaneous prioritized selection of end-effector locations from a large set of candidates, or inverse kinematic control with simultaneous selection of bi-manual grasp locations on a randomly rotated box.
>
---
#### [replaced 020] Graph-based Online Lidar Odometry with Retrospective Map Refinement
- **分类: cs.RO**

- **简介: 该论文属于激光雷达里程计任务，旨在解决传统方法误差累积问题。通过多子图匹配与回溯优化，提升定位精度与一致性。**

- **链接: [https://arxiv.org/pdf/2503.21293](https://arxiv.org/pdf/2503.21293)**

> **作者:** Aaron Kurda; Simon Steuernagel; Marcus Baum
>
> **摘要:** Lidar-only odometry aims to estimate the trajectory of a mobile platform from a stream of lidar scans. Traditional scan-to map approaches register each scan against a single, evolving map, which propagates registration errors over time. To mitigate this, we propose a multitude-of-maps approach where the current scan is registered against multiple overlapping submaps instead of a single static map. By optimizing the resulting constraints in a pose graph, our method enables not only precise estimation of the current pose but also retrospective refinement of the submaps' anchor points, which improves short-term consistency and long-term accuracy. We demonstrate that our approach achieves competitive and often superior accuracy on a variety of automotive datasets while maintaining real-time performance. Ablation studies confirm the critical role of multiple registrations and retrospective refinement of the map as core factors for our accuracy gains. Code and raw results are available on our public GitHub at this https URL.
>
---
#### [replaced 021] Phys2Real: Fusing VLM Priors with Interactive Online Adaptation for Uncertainty-Aware Sim-to-Real Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，解决模拟到现实的迁移问题。通过融合视觉语言模型先验与在线适应，提升策略的不确定性感知能力，实现更高效的仿真到现实转移。**

- **链接: [https://arxiv.org/pdf/2510.11689](https://arxiv.org/pdf/2510.11689)**

> **作者:** Maggie Wang; Stephen Tian; Aiden Swann; Ola Shorinwa; Jiajun Wu; Mac Schwager
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Learning robotic manipulation policies directly in the real world can be expensive and time-consuming. While reinforcement learning (RL) policies trained in simulation present a scalable alternative, effective sim-to-real transfer remains challenging, particularly for tasks that require precise dynamics. To address this, we propose Phys2Real, a real-to-sim-to-real RL pipeline that combines vision-language model (VLM)-inferred physical parameter estimates with interactive adaptation through uncertainty-aware fusion. Our approach consists of three core components: (1) high-fidelity geometric reconstruction with 3D Gaussian splatting, (2) VLM-inferred prior distributions over physical parameters, and (3) online physical parameter estimation from interaction data. Phys2Real conditions policies on interpretable physical parameters, refining VLM predictions with online estimates via ensemble-based uncertainty quantification. On planar pushing tasks of a T-block with varying center of mass (CoM) and a hammer with an off-center mass distribution, Phys2Real achieves substantial improvements over a domain randomization baseline: 100% vs 79% success rate for the bottom-weighted T-block, 57% vs 23% in the challenging top-weighted T-block, and 15% faster average task completion for hammer pushing. Ablation studies indicate that the combination of VLM and interaction information is essential for success. Project website: this https URL.
>
---
#### [replaced 022] XR-DT: Extended Reality-Enhanced Digital Twin for Safe Motion Planning via Human-Aware Model Predictive Path Integral Control
- **分类: cs.RO; cs.AI; cs.HC; cs.MA; eess.SY**

- **简介: 该论文属于人机交互任务，旨在解决机器人在共享空间中安全导航的问题。通过XR-DT框架和HA-MPPI控制模型，实现人类行为预测与机器人路径规划的融合。**

- **链接: [https://arxiv.org/pdf/2512.05270](https://arxiv.org/pdf/2512.05270)**

> **作者:** Tianyi Wang; Jiseop Byeon; Ahmad Yehia; Yiming Xu; Jihyung Park; Tianyi Zeng; Sikai Chen; Ziran Wang; Junfeng Jiao; Christian Claudel
>
> **备注:** 8 pages, 6 figures, 3 tables
>
> **摘要:** As mobile robots increasingly operate alongside humans in shared workspaces, ensuring safe, efficient, and interpretable Human-Robot Interaction (HRI) has become a pressing challenge. While substantial progress has been devoted to human behavior prediction, limited attention has been paid to how humans perceive, interpret, and trust robots' inferences and how robots plan safe and efficient trajectories based on predicted human behaviors. To address these challenges, this paper presents XR-DT, an eXtended Reality-enhanced Digital Twin framework for mobile robots, which bridges physical and virtual spaces to enable bi-directional understanding between humans and robots. Our hierarchical XR-DT architecture integrates augmented-, virtual-, and mixed-reality layers, fusing real-time sensor data, simulated environments in the Unity game engine, and human feedback captured through wearable XR devices. Within this framework, we design a novel Human-Aware Model Predictive Path Integral (HA-MPPI) control model, an MPPI-based motion planner that incorporates ATLAS (Attention-based Trajectory Learning with Anticipatory Sensing), a multi-modal Transformer model designed for egocentric human trajectory prediction via XR headsets. Extensive real-world experimental results demonstrate accurate human trajectory prediction, and safe and efficient robot navigation, validating the HA-MPPI's effectiveness within the XR-DT framework. By embedding human behavior, environmental dynamics, and robot navigation into the XR-DT framework, our system enables interpretable, trustworthy, and adaptive HRI.
>
---
#### [replaced 023] Dependent Reachable Sets for the Constant Bearing Pursuit Strategy
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文属于控制理论任务，研究双智能体跟踪问题，通过常方位追击策略分析依赖可达集的几何特性，提出优化问题并进行仿真验证。**

- **链接: [https://arxiv.org/pdf/2512.00273](https://arxiv.org/pdf/2512.00273)**

> **作者:** Venkata Ramana Makkapati; Tulasi Ram Vechalapu; Vinodhini Comandur; Seth Hutchinson
>
> **备注:** This work has been submitted to a journal for possible publication
>
> **摘要:** This paper introduces a novel reachability problem for the scenario involving two agents, where one agent follows another agent using a feedback strategy. The geometry of the reachable set for an agent, termed \emph{dependent reachable set}, is characterized using the constant bearing pursuit strategy as a case study. Key theoretical results are presented that provide geometric bounds for the associated dependent reachable set. Simulation results are presented to empirically establish the shape of the dependent reachable set. In the process, an original optimization problem is formulated and analyzed for the constant bearing pursuit strategy.
>
---
#### [replaced 024] Safe Model Predictive Diffusion with Shielding
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决复杂系统生成安全、可行轨迹的问题。提出Safe MPD方法，结合模型扩散与安全屏蔽，确保轨迹安全可行。**

- **链接: [https://arxiv.org/pdf/2512.06261](https://arxiv.org/pdf/2512.06261)**

> **作者:** Taekyung Kim; Keyvan Majd; Hideki Okamoto; Bardh Hoxha; Dimitra Panagou; Georgios Fainekos
>
> **备注:** 2026 IEEE International Conference on Robotics and Automation (ICRA). Project page: this https URL
>
> **摘要:** Generating safe, kinodynamically feasible, and optimal trajectories for complex robotic systems is a central challenge in robotics. This paper presents Safe Model Predictive Diffusion (Safe MPD), a training-free diffusion planner that unifies a model-based diffusion framework with a safety shield to generate trajectories that are both kinodynamically feasible and safe by construction. By enforcing feasibility and safety on all samples during the denoising process, our method avoids the common pitfalls of post-processing corrections, such as computational intractability and loss of feasibility. We validate our approach on challenging non-convex planning problems, including kinematic and acceleration-controlled tractor-trailer systems. The results show that it substantially outperforms existing safety strategies in success rate and safety, while achieving sub-second computation times.
>
---
#### [replaced 025] Safe-SAGE: Social-Semantic Adaptive Guidance for Safe Engagement through Laplace-Modulated Poisson Safety Functions
- **分类: cs.RO**

- **简介: 该论文属于机器人安全导航任务，解决传统方法对障碍物语义理解不足的问题。提出Safe-SAGE框架，结合语义信息与安全控制，实现动态环境中的安全导航。**

- **链接: [https://arxiv.org/pdf/2603.05497](https://arxiv.org/pdf/2603.05497)**

> **作者:** Lizhi Yang; Ryan M. Bena; Meg Wilkinson; Gilbert Bahati; Andy Navarro Brenes; Ryan K. Cosner; Aaron D. Ames
>
> **备注:** 8 pages
>
> **摘要:** Traditional safety-critical control methods, such as control barrier functions, suffer from semantic blindness, exhibiting the same behavior around obstacles regardless of contextual significance. This limitation leads to the uniform treatment of all obstacles, despite their differing semantic meanings. We present Safe-SAGE (Social-Semantic Adaptive Guidance for Safe Engagement), a unified framework that bridges the gap between high-level semantic understanding and low-level safety-critical control through a Poisson safety function (PSF) modulated using a Laplace guidance field. Our approach perceives the environment by fusing multi-sensor point clouds with vision-based instance segmentation and persistent object tracking to maintain up-to-date semantics beyond the camera's field of view. A multi-layer safety filter is then used to modulate system inputs to achieve safe navigation using this semantic understanding of the environment. This safety filter consists of both a model predictive control layer and a control barrier function layer. Both layers utilize the PSF and flux modulation of the guidance field to introduce varying levels of conservatism and multi-agent passing norms for different obstacles in the environment. Our framework enables legged robots to safely navigate semantically rich, dynamic environments with context-dependent safety margins.
>
---
#### [replaced 026] Bi-AQUA: Bilateral Control-Based Imitation Learning for Underwater Robot Arms via Lighting-Aware Action Chunking with Transformers
- **分类: cs.RO**

- **简介: 该论文提出Bi-AQUA，解决水下机器人操作中光照变化导致的视觉控制难题。通过光照感知的Transformer方法，提升水下机械臂的抓取与操作能力。**

- **链接: [https://arxiv.org/pdf/2511.16050](https://arxiv.org/pdf/2511.16050)**

> **作者:** Takeru Tsunoori; Masato Kobayashi; Yuki Uranishi
>
> **摘要:** Underwater robotic manipulation remains challenging because lighting variation, color attenuation, scattering, and reduced visibility can severely degrade visuomotor policies. We present Bi-AQUA, the first underwater bilateral control-based imitation learning framework for robot arms that explicitly models lighting within the policy. Bi-AQUA integrates transformer-based bilateral action chunking with a hierarchical lighting-aware design composed of a label-free Lighting Encoder, FiLM-based visual feature modulation, and a lighting token for action conditioning. This design enables adaptation to static and dynamically changing underwater illumination while preserving the force-sensitive advantages of bilateral control, which are particularly important in long-horizon and contact-rich manipulation. Real-world experiments on underwater pick-and-place, drawer closing, and peg extraction tasks show that Bi-AQUA outperforms a bilateral baseline without lighting modeling and achieves robust performance under seen, unseen, and changing lighting conditions. These results highlight the importance of combining explicit lighting modeling with force-aware bilateral imitation learning for reliable underwater manipulation. For additional material, please check: this https URL
>
---
#### [replaced 027] CAVER: Curious Audiovisual Exploring Robot
- **分类: cs.RO**

- **简介: 该论文提出CAVER机器人，解决多模态感知与交互问题，通过音频视觉表征提升物体识别和动作模仿能力。**

- **链接: [https://arxiv.org/pdf/2511.07619](https://arxiv.org/pdf/2511.07619)**

> **作者:** Luca Macesanu; Boueny Folefack; Samik Singh; Ruchira Ray; Ben Abbatematteo; Roberto Martín-Martín
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Multimodal audiovisual perception can enable new avenues for robotic manipulation, from better material classification to the imitation of demonstrations for which only audio signals are available (e.g., playing a tune by ear). However, to unlock such multimodal potential, robots need to learn the correlations between an object's visual appearance and the sound it generates when they interact with it. Such an active sensorimotor experience requires new interaction capabilities, representations, and exploration methods to guide the robot in efficiently building increasingly rich audiovisual knowledge. In this work, we present CAVER, a novel robot that builds and utilizes rich audiovisual representations of objects. CAVER includes three novel contributions: 1) a novel 3D printed end-effector, attachable to parallel grippers, that excites objects' audio responses, 2) an audiovisual representation that combines local and global appearance information with sound features, and 3) an exploration algorithm that uses and builds the audiovisual representation in a curiosity-driven manner that prioritizes interacting with high uncertainty objects to obtain good coverage of surprising audio with fewer interactions. We demonstrate that CAVER builds rich representations in different scenarios more efficiently than several exploration baselines, and that the learned audiovisual representation leads to significant improvements in material classification and the imitation of audio-only human demonstrations. this https URL
>
---
#### [replaced 028] FindAnything: Open-Vocabulary and Object-Centric Mapping for Robot Exploration in Any Environment
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出FindAnything，解决机器人在未知环境中实时语义映射的问题。通过融合视觉语言信息，实现高效、可扩展的语义地图构建。**

- **链接: [https://arxiv.org/pdf/2504.08603](https://arxiv.org/pdf/2504.08603)**

> **作者:** Sebastián Barbas Laina; Simon Boche; Sotiris Papatheodorou; Simon Schaefer; Jaehyung Jung; Helen Oleynikova; Stefan Leutenegger
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Geometrically accurate and semantically expressive map representations have proven invaluable for robot deployment and task planning in unknown environments. Nevertheless, real-time, open-vocabulary semantic understanding of large-scale unknown environments still presents open challenges, mainly due to computational requirements. In this paper we present FindAnything, an open-world mapping framework that incorporates vision-language information into dense volumetric submaps. Thanks to the use of vision-language features, FindAnything combines pure geometric and open-vocabulary semantic information for a higher level of understanding. It proposes an efficient storage of open-vocabulary information through the aggregation of features at the object level. Pixelwise vision-language features are aggregated based on eSAM segments, which are in turn integrated into object-centric volumetric submaps, providing a mapping from open-vocabulary queries to 3D geometry that is scalable also in terms of memory usage. We demonstrate that FindAnything performs on par with the state-of-the-art in terms of semantic accuracy while being substantially faster and more memory-efficient, allowing its deployment in large-scale environments and on resourceconstrained devices, such as MAVs. We show that the real-time capabilities of FindAnything make it useful for downstream tasks, such as autonomous MAV exploration in a simulated Search and Rescue scenario. Project Page: this https URL.
>
---
#### [replaced 029] ROS-related Robotic Systems Development with V-model-based Application of MeROS Metamodel
- **分类: cs.RO; cs.SE**

- **简介: 该论文属于机器人系统工程任务，旨在解决ROS系统在协调与管理上的难题。通过结合MBSE与MeROS模型，提出基于V模型的结构化方法，提升系统设计的可追踪性与验证能力。**

- **链接: [https://arxiv.org/pdf/2506.08706](https://arxiv.org/pdf/2506.08706)**

> **作者:** Tomasz Winiarski; Jan Kaniuka; Daniel Giełdowski; Jakub Ostrysz; Krystian Radlak; Dmytro Kushnir
>
> **备注:** 22 pages
>
> **摘要:** Systems built on the Robot Operating System (ROS) are increasingly easy to assemble, yet hard to govern and reliably coordinate. Beyond the sheer number of subsystems involved, the difficulty stems from their diversity and interaction depth. In this paper, we use a compact heterogeneous robotic system (HeROS), combining mobile and manipulation capabilities, as a demonstration vehicle under dynamically changing tasks. Notably, all its subsystems are powered by ROS. The use of compatible interfaces and other ROS integration capabilities simplifies the construction of such systems. However, this only addresses part of the complexity: the semantic coherence and structural traceability are even more important for precise coordination and call for deliberate engineering methods. The Model-Based Systems Engineering (MBSE) discipline, which emerged from the experience of complexity management in large-scale engineering domains, offers the methodological foundations needed. Despite their strengths in complementary aspects of robotics systems engineering, the lack of a unified approach to integrate ROS and MBSE hinders the full potential of these tools. Motivated by the anticipated impact of such a synergy in robotics practice, we propose a structured methodology based on MeROS - a SysML metamodel created specifically to put the ROS-based systems into the focus of the MBSE workflow. As its methodological backbone, we adapt the well-known V-model to this context, illustrating how complex robotic systems can be designed with traceability and validation capabilities embedded into their lifecycle using practices familiar to engineering teams.
>
---
#### [replaced 030] Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）模型训练任务，旨在解决仿真与真实环境协同训练效果有限的问题。提出基于强化学习的协同训练框架，提升真实机器人部署效果和泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.12628](https://arxiv.org/pdf/2602.12628)**

> **作者:** Liangzhi Shi; Shuaihang Chen; Feng Gao; Yinuo Chen; Kang Chen; Tonghe Zhang; Hongzhi Zang; Weinan Zhang; Chao Yu; Yu Wang
>
> **摘要:** Simulation offers a scalable and low-cost way to enrich vision-language-action (VLA) training, reducing reliance on expensive real-robot demonstrations. However, most sim-real co-training methods rely on supervised fine-tuning (SFT), which treats simulation as a static source of demonstrations and does not exploit large-scale closed-loop interaction. Consequently, real-world gains and generalization are often limited. In this paper, we propose an \underline{\textit{RL}}-based sim-real \underline{\textit{Co}}-training \modify{(RL-Co)} framework that leverages interactive simulation while preserving real-world capabilities. Our method follows a generic two-stage design: we first warm-start the policy with SFT on a mixture of real and simulated demonstrations, then fine-tune it with reinforcement learning in simulation while adding an auxiliary supervised loss on real-world data to anchor the policy and mitigate catastrophic forgetting. We evaluate our framework on four real-world tabletop manipulation tasks using two representative VLA architectures, OpenVLA and $\pi_{0.5}$, and observe consistent improvements over real-only fine-tuning and SFT-based co-training, including +24% real-world success on OpenVLA and +20% on $\pi_{0.5}$. Beyond higher success rates, RL co-training yields stronger generalization to unseen task variations and substantially improved real-world data efficiency, providing a practical and scalable pathway for leveraging simulation to enhance real-robot deployment.
>
---
#### [replaced 031] APEX: Learning Adaptive High-Platform Traversal for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文属于人形机器人高平台行走任务，解决现有方法难以安全攀爬超过腿长平台的问题，提出APEX系统实现感知驱动的多技能自主 traversal。**

- **链接: [https://arxiv.org/pdf/2602.11143](https://arxiv.org/pdf/2602.11143)**

> **作者:** Yikai Wang; Tingxuan Leng; Changyi Lin; Shiqi Liu; Shir Simon; Bingqing Chen; Jonathan Francis; Ding Zhao
>
> **备注:** Project Website: this https URL
>
> **摘要:** Humanoid locomotion has advanced rapidly with deep reinforcement learning (DRL), enabling robust feet-based traversal over uneven terrain. Yet platforms beyond leg length remain largely out of reach because current RL training paradigms often converge to jumping-like solutions that are high-impact, torque-limited, and unsafe for real-world deployment. To address this gap, we propose APEX, a system for perceptive, climbing-based high-platform traversal that composes terrain-conditioned behaviors: climb-up and climb-down at vertical edges, walking or crawling on the platform, and stand-up and lie-down for posture reconfiguration. Central to our approach is a generalized ratchet progress reward for learning contact-rich, goal-reaching maneuvers. It tracks the best-so-far task progress and penalizes non-improving steps, providing dense yet velocity-free supervision that enables efficient exploration under strong safety regularization. Based on this formulation, we train LiDAR-based full-body maneuver policies and reduce the sim-to-real perception gap through a dual strategy: modeling mapping artifacts during training and applying filtering and inpainting to elevation maps during deployment. Finally, we distill all six skills into a single policy that autonomously selects behaviors and transitions based on local geometry and commands. Experiments on a 29-DoF Unitree G1 humanoid demonstrate zero-shot sim-to-real traversal of 0.8 meter platforms (approximately 114% of leg length), with robust adaptation to platform height and initial pose, as well as smooth and stable multi-skill transitions.
>
---
#### [replaced 032] Whole-Body Model-Predictive Control of Legged Robots with MuJoCo
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人控制任务，旨在解决腿式机器人全身模型预测控制问题。通过iLQR算法与MuJoCo动力学结合，实现高效实时控制，并验证其在真实环境中的有效性。**

- **链接: [https://arxiv.org/pdf/2503.04613](https://arxiv.org/pdf/2503.04613)**

> **作者:** John Z. Zhang; Taylor A. Howell; Zeji Yi; Chaoyi Pan; Guanya Shi; Guannan Qu; Tom Erez; Yuval Tassa; Zachary Manchester
>
> **备注:** to appear at ICRA 2026
>
> **摘要:** We demonstrate the surprising real-world effectiveness of a very simple approach to whole-body model-predictive control (MPC) of quadruped and humanoid robots: the iterative LQR (iLQR) algorithm with MuJoCo dynamics and finite-difference approximated derivatives. Building upon the previous success of model-based behavior synthesis and control of locomotion and manipulation tasks with MuJoCo in simulation, we show that these policies can easily generalize to the real world with few sim-to-real considerations. Our baseline method achieves real-time whole-body MPC on a variety of hardware experiments, including dynamic quadruped locomotion, quadruped walking on two legs, and full-sized humanoid bipedal locomotion. We hope this easy-to-reproduce hardware baseline lowers the barrier to entry for real-world whole-body MPC research and contributes to accelerating research velocity in the community. Our code and experiment videos will be available online at:this https URL
>
---
#### [replaced 033] FALCON: Future-Aware Learning with Contextual Object-Centric Pretraining for UAV Action Recognition
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出FALCON，解决无人机视频动作识别中的空间不平衡问题。通过对象感知的预训练方法，提升识别准确率并加快推理速度。**

- **链接: [https://arxiv.org/pdf/2409.18300](https://arxiv.org/pdf/2409.18300)**

> **作者:** Ruiqi Xian; Xiyang Wu; Tianrui Guan; Xijun Wang; Boqing Gong; Dinesh Manocha
>
> **摘要:** We introduce FALCON, a unified self-supervised video pretraining approach for UAV action recognition from raw RGB aerial footage, requiring no additional preprocessing at inference. UAV videos exhibit severe spatial imbalance: large, cluttered backgrounds dominate the field of view, causing reconstruction-based pretraining to waste capacity on uninformative regions and under-learn action-relevant human/object cues. FALCON addresses this by integrating object-aware masked autoencoding with object-centric dual-horizon future reconstruction. Using detections only during pretraining, we construct objectness priors that (i) enforce balanced token visibility during masking and (ii) concentrate reconstruction supervision on action-relevant regions, preventing learning from being dominated by background appearance. To promote temporal dynamics learning, we further reconstruct short- and long-horizon future content within an object-centric supervision region, injecting anticipatory temporal supervision that is robust to noisy aerial context. Across UAV benchmarks, FALCON improves top-1 accuracy by 2.9\% on NEC-Drone and 5.8\% on UAV-Human with a ViT-B backbone, while achieving 2$\times$--5$\times$ faster inference than supervised approaches that rely on heavy test-time augmentation.
>
---
#### [replaced 034] Real-Time Learning of Predictive Dynamic Obstacle Models for Robotic Motion Planning
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于机器人运动规划任务，解决如何实时学习非线性预测模型的问题。通过改进的Hankel-DMD方法实现动态障碍物的去噪与预测。**

- **链接: [https://arxiv.org/pdf/2511.00814](https://arxiv.org/pdf/2511.00814)**

> **作者:** Stella Kombo; Masih Haseli; Skylar X. Wei; Joel W. Burdick
>
> **备注:** 10 pages, 6 figures, submitted to IEEE International Conference on Robotics and Automation (ICRA) 2025
>
> **摘要:** Autonomous systems often must predict the motions of nearby agents from partial and noisy data. This paper asks and answers the question: "can we learn, in real-time, a nonlinear predictive model of another agent's motions?" Our online framework denoises and forecasts such dynamics using a modified sliding-window Hankel Dynamic Mode Decomposition (Hankel-DMD). Partial noisy measurements are embedded into a Hankel matrix, while an associated Page matrix enables singular-value hard thresholding (SVHT) to estimate the effective rank. A Cadzow projection enforces structured low-rank consistency, yielding a denoised trajectory and local noise variance estimates. From this representation, a time-varying Hankel-DMD lifted linear predictor is constructed for multi-step forecasts. The residual analysis provides variance-tracking signals that can support downstream estimators and risk-aware planning. We validate the approach in simulation under Gaussian and heavy-tailed noise, and experimentally on a dynamic crane testbed. Results show that the method achieves stable variance-aware denoising and short-horizon prediction suitable for integration into real-time control frameworks.
>
---
#### [replaced 035] Diverse and Adaptive Behavior Curriculum for Autonomous Driving: A Student-Teacher Framework with Multi-Agent RL
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于自主驾驶领域，旨在解决RL训练中场景单一、泛化能力差的问题。提出学生-教师框架，通过多智能体RL生成多样化交通行为，提升驾驶策略的鲁棒性与覆盖性。**

- **链接: [https://arxiv.org/pdf/2507.19146](https://arxiv.org/pdf/2507.19146)**

> **作者:** Ahmed Abouelazm; Johannes Ratz; Philip Schörner; J. Marius Zöllner
>
> **备注:** First and Second authors contributed equally; Paper accepted in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Autonomous driving faces challenges in navigating complex real-world traffic, requiring safe handling of both common and critical scenarios. Reinforcement learning (RL), a prominent method in end-to-end driving, enables agents to learn through trial and error in simulation. However, RL training often relies on rule-based traffic scenarios, limiting generalization. Additionally, current scenario generation methods focus heavily on critical scenarios, neglecting a balance with routine driving behaviors. Curriculum learning, which progressively trains agents on increasingly complex tasks, is a promising approach to improving the robustness and coverage of RL driving policies. However, existing research mainly emphasizes manually designed curricula, focusing on scenery and actor placement rather than traffic behavior dynamics. This work introduces a novel student-teacher framework for automatic curriculum learning. The teacher, a graph-based multi-agent RL component, adaptively generates traffic behaviors across diverse difficulty levels. An adaptive mechanism adjusts task difficulty based on student performance, ensuring exposure to behaviors ranging from common to critical. The student, though exchangeable, is realized as a deep RL agent with partial observability, reflecting real-world perception constraints. Results demonstrate the teacher's ability to generate diverse traffic behaviors. The student, trained with automatic curricula, outperformed agents trained on rule-based traffic, achieving higher rewards and exhibiting balanced, assertive driving.
>
---
#### [replaced 036] OmniDP: Beyond-FOV Large-Workspace Humanoid Manipulation with Omnidirectional 3D Perception
- **分类: cs.RO**

- **简介: 该论文属于人形机器人操作任务，旨在解决大工作空间下的感知局限问题。通过提出OmniDP方法，利用LiDAR实现360°三维感知，提升机器人在复杂环境中的操作能力。**

- **链接: [https://arxiv.org/pdf/2603.05355](https://arxiv.org/pdf/2603.05355)**

> **作者:** Pei Qu; Zheng Li; Yufei Jia; Ziyun Liu; Liang Zhu; Haoang Li; Jinni Zhou; Jun Ma
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** The deployment of humanoid robots for dexterous manipulation in unstructured environments remains challenging due to perceptual limitations that constrain the effective workspace. In scenarios where physical constraints prevent the robot from repositioning itself, maintaining omnidirectional awareness becomes far more critical than color or semantic this http URL recent advances in visuomotor policy learning have improved manipulation capabilities, conventional RGB-D solutions suffer from narrow fields of view (FOV) and self-occlusion, requiring frequent base movements that introduce motion uncertainty and safety risks. Existing approaches to expanding perception, including active vision systems and third-view cameras, introduce mechanical complexity, calibration dependencies, and latency that hinder reliable real-time performance. In this work, We propose OmniDP, an end-to-end LiDAR-driven 3D visuomotor policy that enables robust manipulation in large workspaces. Our method processes panoramic point clouds through a Time-Aware Attention Pooling mechanism, efficiently encoding sparse 3D data while capturing temporal dependencies. This 360° perception allows the robot to interact with objects across wide areas without frequent repositioning. To support policy learning, we develop a whole-body teleoperation system for efficient data collection on full-body coordination. Extensive experiments in simulation and real-world environments show that OmniDP achieves robust performance in large-workspace and cluttered scenarios, outperforming baselines that rely on egocentric depth cameras.
>
---
#### [replaced 037] CAPS: Context-Aware Priority Sampling for Enhanced Imitation Learning in Autonomous Driving
- **分类: cs.LG; cs.RO**

- **简介: 论文提出CAPS方法，用于增强自动驾驶中的模仿学习数据效率。解决数据不平衡问题，通过VQ-VAE提取结构化表示，按簇优先采样，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2503.01650](https://arxiv.org/pdf/2503.01650)**

> **作者:** Hamidreza Mirkhani; Behzad Khamidehi; Ehsan Ahmadi; Mohammed Elmahgiubi; Weize Zhang; Fazel Arasteh; Umar Rajguru; Kasra Rezaee; Dongfeng Bai
>
> **备注:** Accepted at IEEE International Conference on Robotics & Automation (ICRA 2026)
>
> **摘要:** In this paper, we introduce Context-Aware Priority Sampling (CAPS), a novel method designed to enhance data efficiency in learning-based autonomous driving systems. CAPS addresses the challenge of imbalanced datasets in imitation learning by leveraging Vector Quantized Variational Autoencoders (VQ-VAEs). In this way, we can get structured and interpretable data representations, which help to reveal meaningful patterns in the data. These patterns are used to group the data into clusters, with each sample being assigned a cluster ID. The cluster IDs are then used to re-balance the dataset, ensuring that rare yet valuable samples receive higher priority during training. We evaluate our method through closed-loop experiments in the CARLA simulator. The results on Bench2Drive scenarios demonstrate the effectiveness of CAPS in enhancing model generalization, with substantial improvements in both driving score and success rate.
>
---
#### [replaced 038] Learning Robust Control Policies for Inverted Pose on Miniature Blimp Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决微型飞艇机器人在倒置姿态下的稳定控制问题。通过构建仿真环境、训练控制策略并设计映射层，实现高效可靠的倒置控制。**

- **链接: [https://arxiv.org/pdf/2602.23972](https://arxiv.org/pdf/2602.23972)**

> **作者:** Yuanlin Yang; Lin Hong; Fumin Zhang
>
> **备注:** Accepted in ICRA 2026
>
> **摘要:** The ability to achieve and maintain inverted poses is essential for unlocking the full agility of miniature blimp robots (MBRs). However, developing reliable inverted control strategies for MBRs remains challenging due to their complex and underactuated dynamics. To address this challenge, we propose a novel framework that enables robust control policy learning for inverted pose on MBRs. The proposed framework consists of three core stages. First, a high-fidelity three-dimensional (3D) simulation environment is constructed and calibrated using real-world MBR motion data. Second, a robust inverted control policy is trained in simulation using a modified Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm combined with a domain randomization strategy. Third, a mapping layer is designed to bridge the sim-to-real gap and facilitate real-world deployment of the learned policy. Comprehensive evaluations in the simulation environment demonstrate that the learned policy achieves a higher success rate compared to the energy-shaping controller. Furthermore, experimental results confirm that the learned policy with a mapping layer enables an MBR to achieve and maintain a fully inverted pose in real-world settings.
>
---
#### [replaced 039] (MGS)$^2$-Net: Unifying Micro-Geometric Scale and Macro-Geometric Structure for Cross-View Geo-Localization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于跨视角地理定位任务，旨在解决航空图像与卫星图之间几何错位导致的定位不准确问题。提出(MGS)$^2$框架，融合宏观结构与微观尺度处理，提升定位精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.10704](https://arxiv.org/pdf/2602.10704)**

> **作者:** Minglei Li; Mengfan He; Chunyu Li; Chao Chen; Xingyu Shao; Ziyang Meng
>
> **摘要:** Cross-view geo-localization (CVGL) is pivotal for GNSS-denied UAV navigation but remains brittle under the drastic geometric misalignment between oblique aerial views and orthographic satellite references. Existing methods predominantly operate within a 2D manifold, neglecting the underlying 3D geometry where view-dependent vertical facades (macro-structure) and scale variations (micro-scale) severely corrupt feature alignment. To bridge this gap, we propose (MGS)$^2$, a geometry-grounded framework. The core of our innovation is the Macro-Geometric Structure Filtering (MGSF) module. Unlike pixel-wise matching sensitive to noise, MGSF leverages dilated geometric gradients to physically filter out high-frequency facade artifacts while enhancing the view-invariant horizontal plane, directly addressing the domain shift. To guarantee robust input for this structural filtering, we explicitly incorporate a Micro-Geometric Scale Adaptation (MGSA) module. MGSA utilizes depth priors to dynamically rectify scale discrepancies via multi-branch feature fusion. Furthermore, a Geometric-Appearance Contrastive Distillation (GACD) loss is designed to strictly discriminate against oblique occlusions. Extensive experiments demonstrate that (MGS)$^2$ achieves state-of-the-art performance, recording a Recall@1 of 97.5\% on University-1652 and 97.02\% on SUES-200. Furthermore, the framework exhibits superior cross-dataset generalization against geometric ambiguity. The code is available at: \href{this https URL}{this https URL}.
>
---
#### [replaced 040] ROSER: Few-Shot Robotic Sequence Retrieval for Scalable Robot Learning
- **分类: cs.RO**

- **简介: 该论文提出ROSER，解决机器人学习中任务标注数据稀缺的问题。通过少样本序列检索，从连续日志中提取任务相关片段，提升数据利用率。**

- **链接: [https://arxiv.org/pdf/2603.01474](https://arxiv.org/pdf/2603.01474)**

> **作者:** Zillur Rahman; Eddison Pham; Alejandro Daniel Noel; Cristian Meo
>
> **备注:** 2026 ICLR DATA-FM Workshop
>
> **摘要:** A critical bottleneck in robot learning is the scarcity of task-labeled, segmented training data, despite the abundance of large-scale robotic datasets recorded as long, continuous interaction logs. Existing datasets contain vast amounts of diverse behaviors, yet remain structurally incompatible with modern learning frameworks that require cleanly segmented, task-specific trajectories. We address this data utilization crisis by formalizing robotic sequence retrieval: the task of extracting reusable, task-centric segments from unlabeled logs using only a few reference examples. We introduce ROSER, a lightweight few-shot retrieval framework that learns task-agnostic metric spaces over temporal windows, enabling accurate retrieval with as few as 3-5 demonstrations, without any task-specific training required. To validate our approach, we establish comprehensive evaluation protocols and benchmark ROSER against classical alignment methods, learned embeddings, and language model baselines across three large-scale datasets (e.g., LIBERO, DROID, and nuScenes). Our experiments demonstrate that ROSER consistently outperforms all prior methods in both accuracy and efficiency, achieving sub-millisecond per-match inference while maintaining superior distributional alignment. By reframing data curation as few-shot retrieval, ROSER provides a practical pathway to unlock underutilized robotic datasets, fundamentally improving data availability for robot learning.
>
---
#### [replaced 041] Contact-Safe Reinforcement Learning with ProMP Reparameterization and Energy Awareness
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决复杂环境下的安全操作问题。通过结合PPO与运动基元，提出能量安全框架，提升轨迹一致性与接触安全性。**

- **链接: [https://arxiv.org/pdf/2511.13459](https://arxiv.org/pdf/2511.13459)**

> **作者:** Bingkun Huang; Yuhe Gong; Zewen Yang; Tianyu Ren; Luis Figueredo
>
> **备注:** 8 pages
>
> **摘要:** Reinforcement learning (RL) approaches based on Markov Decision Processes (MDPs) are predominantly applied in the robot joint space, often relying on limited task-specific information and partial awareness of the 3D environment. In contrast, episodic RL has demonstrated advantages over traditional MDP-based methods in terms of trajectory consistency, task awareness, and overall performance in complex robotic tasks. Moreover, traditional step-wise and episodic RL methods often neglect the contact-rich information inherent in task-space manipulation, especially considering the contact-safety and robustness. In this work, contact-rich manipulation tasks are tackled using a task-space, energy-safe framework, where reliable and safe task-space trajectories are generated through the combination of Proximal Policy Optimization (PPO) and movement primitives. Furthermore, an energy-aware Cartesian Impedance Controller objective is incorporated within the proposed framework to ensure safe interactions between the robot and the environment. Our experimental results demonstrate that the proposed framework outperforms existing methods in handling tasks on various types of surfaces in 3D environments, achieving high success rates as well as smooth trajectories and energy-safe interactions.
>
---
#### [replaced 042] MiDAS: A Multimodal Data Acquisition System and Dataset for Robot-Assisted Minimally Invasive Surgery
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出MiDAS系统，解决机器人辅助手术中多模态数据获取难题。通过非侵入方式采集手势、脚踏和视频数据，实现跨平台数据同步与分析。**

- **链接: [https://arxiv.org/pdf/2602.12407](https://arxiv.org/pdf/2602.12407)**

> **作者:** Keshara Weerasinghe; Seyed Hamid Reza Roodabeh; Andrew Hawkins; Zhaomeng Zhang; Zachary Schrader; Homa Alemzadeh
>
> **备注:** 29 pages, 17 figures
>
> **摘要:** Background: Robot-assisted minimally invasive surgery (RMIS) research increasingly relies on multimodal data, yet access to proprietary robot telemetry remains a major barrier. We introduce MiDAS, an open-source, platform-agnostic system enabling time-synchronized, non-invasive multimodal data acquisition across surgical robotic platforms. Methods: MiDAS integrates electromagnetic and RGB-D hand tracking, foot pedal sensing, and surgical video capturing without requiring proprietary robot interfaces. We validated MiDAS on the open-source Raven-II and the clinical da Vinci Xi by collecting multimodal datasets of peg transfer and hernia repair suturing tasks performed by surgical residents. Correlation analysis and downstream gesture recognition experiments were conducted. Results: External hand and foot sensing closely approximated internal robot kinematics and non-invasive motion signals achieved gesture recognition performance comparable to proprietary telemetry. Conclusion: MiDAS enables reproducible multimodal RMIS data collection and is released with annotated datasets, including the first multimodal dataset capturing hernia repair suturing on high-fidelity simulation models.
>
---
#### [replaced 043] RoboPocket: Improve Robot Policies Instantly with Your Phone
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出RoboPocket系统，解决机器人策略训练效率低的问题。通过AR视觉预测和远程微调，实现无需机器人即可快速优化策略，提升数据与样本效率。**

- **链接: [https://arxiv.org/pdf/2603.05504](https://arxiv.org/pdf/2603.05504)**

> **作者:** Junjie Fang; Wendi Chen; Han Xue; Fangyuan Zhou; Tian Le; Yi Wang; Yuting Zhang; Jun Lv; Chuan Wen; Cewu Lu
>
> **备注:** Project page: this https URL
>
> **摘要:** Scaling imitation learning is fundamentally constrained by the efficiency of data collection. While handheld interfaces have emerged as a scalable solution for in-the-wild data acquisition, they predominantly operate in an open-loop manner: operators blindly collect demonstrations without knowing the underlying policy's weaknesses, leading to inefficient coverage of critical state distributions. Conversely, interactive methods like DAgger effectively address covariate shift but rely on physical robot execution, which is costly and difficult to scale. To reconcile this trade-off, we introduce RoboPocket, a portable system that enables Robot-Free Instant Policy Iteration using single consumer smartphones. Its core innovation is a Remote Inference framework that visualizes the policy's predicted trajectory via Augmented Reality (AR) Visual Foresight. This immersive feedback allows collectors to proactively identify potential failures and focus data collection on the policy's weak regions without requiring a physical robot. Furthermore, we implement an asynchronous Online Finetuning pipeline that continuously updates the policy with incoming data, effectively closing the learning loop in minutes. Extensive experiments demonstrate that RoboPocket adheres to data scaling laws and doubles the data efficiency compared to offline scaling strategies, overcoming their long-standing efficiency bottleneck. Moreover, our instant iteration loop also boosts sample efficiency by up to 2$\times$ in distributed environments a small number of interactive corrections per person. Project page and videos: this https URL.
>
---
#### [replaced 044] Push Anything: Single- and Multi-Object Pushing From First Sight with Contact-Implicit MPC
- **分类: cs.RO**

- **简介: 该论文属于机器人非抓取操作任务，解决物体推搡中的接触复杂性问题。通过改进的CI-MPC算法，实现多物体精准推搡控制。**

- **链接: [https://arxiv.org/pdf/2510.19974](https://arxiv.org/pdf/2510.19974)**

> **作者:** Hien Bui; Yufeiyang Gao; Haoran Yang; Eric Cui; Siddhant Mody; Brian Acosta; Thomas Stephen Felix; Bibit Bianchini; Michael Posa
>
> **备注:** Presented at ICRA 2026; 8 pages, 8 figures. Hien Bui, Yufeiyang Gao, and Haoran Yang contributed equally to this work
>
> **摘要:** Non-prehensile manipulation of diverse objects remains a core challenge in robotics, driven by unknown physical properties and the complexity of contact-rich interactions. Recent advances in contact-implicit model predictive control (CI-MPC), with contact reasoning embedded directly in the trajectory optimization, have shown promise in tackling the task efficiently and robustly. However, demonstrations have been limited to narrowly curated examples. In this work, we showcase the broader capabilities of CI-MPC through precise planar pushing tasks over a wide range of object geometries, including multi-object domains. These scenarios demand reasoning over numerous inter-object and object-environment contacts to strategically manipulate and de-clutter the environment, challenges that were intractable for prior CI-MPC methods. To achieve this, we introduce Consensus Complementarity Control Plus (C3+), an enhanced CI-MPC algorithm integrated into a complete pipeline spanning object scanning, mesh reconstruction, and hardware execution. Compared to its predecessor C3, C3+ achieves substantially faster solve times, enabling real-time performance even in multi-object pushing tasks. On hardware, our system achieves overall 98% success rate across 33 objects, reaching pose goals within tight tolerances. The average time-to-goal is approximately 0.5, 1.6, 3.2, and 5.3 minutes for 1-, 2-, 3-, and 4-object tasks, respectively. Project page: this https URL.
>
---
#### [replaced 045] VEGA: Electric Vehicle Navigation Agent via Physics-Informed Neural Operator and Proximal Policy Optimization
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出VEGA系统，解决电动车路径规划与充电优化问题。结合物理信息神经算子和强化学习，实现高效能量感知的导航。**

- **链接: [https://arxiv.org/pdf/2509.13386](https://arxiv.org/pdf/2509.13386)**

> **作者:** Hansol Lim; Minhyeok Im; Jonathan Boyack; Jee Won Lee; Jongseong Brad Choi
>
> **备注:** This work has been submitted to the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) for possible publication
>
> **摘要:** We present VEGA, a vehicle-adaptive energy-aware routing system for electric vehicles (EVs) that integrates physics-informed parameter estimation with RL-based charge-aware path planning. VEGA consists of two copupled modules: (1) a physics-informed neural operator (PINO) that estimates vehicle-specific physical parameters-drag, rolling resistance, mass, motor and regenerative-braking efficiencies, and auxiliary load-from short windows of onboard speed and acceleration data; (2) a Proximal Policy Optimization (PPO) agent that navigates a charger-annotated road graph, jointly selecting routes and charging stops under state-of-charge constraints. The agent is initialized via behavior cloning from an A* teacher and fine-tuned with cirriculum-guided PPO on the full U.S. highway network with Tesla Supercharger locations. On a cross-country San Francisco-to-New York route (~4,860km), VEGA produces a feasible 20-stop plan with 56.12h total trip time and minimum SoC 11.41%. Against the controlled Energy-aware A* baseline, the distance and driving-time gaps are small (-8.49km and +0.37h), while inference is >20x faster. The learned policy generalizes without retraining to road networks in France and Japan.
>
---
#### [replaced 046] Generative Predictive Control: Flow Matching Policies for Dynamic and Difficult-to-Demonstrate Tasks
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文提出生成式预测控制，解决动态任务难以演示的问题。通过生成模型与预测控制结合，实现快速动态任务的策略学习。**

- **链接: [https://arxiv.org/pdf/2502.13406](https://arxiv.org/pdf/2502.13406)**

> **作者:** Vince Kurtz; Joel W. Burdick
>
> **备注:** ICRA 2026
>
> **摘要:** Generative control policies have recently unlocked major progress in robotics. These methods produce action sequences via diffusion or flow matching, with training data provided by demonstrations. But existing methods come with two key limitations: they require expert demonstrations, which can be difficult to obtain, and they are limited to relatively slow, quasi-static tasks. In this paper, we leverage a tight connection between sampling-based predictive control and generative modeling to address each of these issues. In particular, we introduce generative predictive control, a supervised learning framework for tasks with fast dynamics that are easy to simulate but difficult to demonstrate. We then show how trained flow-matching policies can be warm-started at inference time, maintaining temporal consistency and enabling high-frequency feedback. We believe that generative predictive control offers a complementary approach to existing behavior cloning methods, and hope that it paves the way toward generalist policies that extend beyond quasi-static demonstration-oriented tasks.
>
---
#### [replaced 047] ROSplane 2.0: A Fixed-Wing Autopilot for Research
- **分类: cs.RO; eess.SY**

- **简介: 该论文介绍ROSplane 2.0，一个用于固定翼无人机研究的开源自主系统。旨在解决 UAV 研究中集成新技术困难的问题，通过提供模块化、易用的框架加速研究进程。**

- **链接: [https://arxiv.org/pdf/2510.01041](https://arxiv.org/pdf/2510.01041)**

> **作者:** Ian Reid; Joseph Ritchie; Jacob Moore; Brandon Sutherland; Gabe Snow; Phillip Tokumaru; Tim McLain
>
> **备注:** Submitted to the 2026 International Conference on Unmanned Aerial Systems
>
> **摘要:** Unmanned aerial vehicle (UAV) research requires the integration of cutting-edge technology into existing autopilot frameworks. This process can be arduous, requiring extensive resources, time, and detailed knowledge of the existing system. ROSplane is a lean, open-source fixed-wing autonomy stack built by researchers for researchers. It is designed to accelerate research by providing clearly defined interfaces with an easily modifiable framework. Built around ROS 2, ROSplane allows for rapid integration of low or high-level control, path planning, or estimation algorithms. A focus on lean, easily-understood code and extensive documentation lowers the barrier to entry for researchers. Recent developments to ROSplane improve its capacity to accelerate UAV research, including the transition from ROS 1 to ROS 2, enhanced estimation and control algorithms, increased modularity, and an improved aerodynamic modeling pipeline. This aerodynamic modeling pipeline significantly reduces the effort of transitioning from simulation to real-world testing without requiring costly system identification or computational fluid dynamics tools. ROSplane's architecture reduces the effort required to integrate new research tools and methods, expediting hardware experimentation.
>
---
#### [replaced 048] RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶解释任务，解决数据稀缺和泛化能力不足问题，提出RAG-Driver模型提升驾驶决策的可解释性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2402.10828](https://arxiv.org/pdf/2402.10828)**

> **作者:** Jianhao Yuan; Shuyang Sun; Daniel Omeiza; Bo Zhao; Paul Newman; Lars Kunze; Matthew Gadd
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** We need to trust robots that use often opaque AI methods. They need to explain themselves to us, and we need to trust their explanation. In this regard, explainability plays a critical role in trustworthy autonomous decision-making to foster transparency and acceptance among end users, especially in complex autonomous driving. Recent advancements in Multi-Modal Large Language models (MLLMs) have shown promising potential in enhancing the explainability as a driving agent by producing control predictions along with natural language explanations. However, severe data scarcity due to expensive annotation costs and significant domain gaps between different datasets makes the development of a robust and generalisable system an extremely challenging task. Moreover, the prohibitively expensive training requirements of MLLM and the unsolved problem of catastrophic forgetting further limit their generalisability post-deployment. To address these challenges, we present RAG-Driver, a novel retrieval-augmented multi-modal large language model that leverages in-context learning for high-performance, explainable, and generalisable autonomous driving. By grounding in retrieved expert demonstration, we empirically validate that RAG-Driver achieves state-of-the-art performance in producing driving action explanations, justifications, and control signal prediction. More importantly, it exhibits exceptional zero-shot generalisation capabilities to unseen environments without further training endeavours.
>
---
#### [replaced 049] Sample-Based Hybrid Mode Control: Asymptotically Optimal Switching of Algorithmic and Non-Differentiable Control Modes
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决混合控制模式切换问题。通过样本方法优化模式选择与切换，实现高效、稳定的控制策略。**

- **链接: [https://arxiv.org/pdf/2510.19074](https://arxiv.org/pdf/2510.19074)**

> **作者:** Yilang Liu; Haoxiang You; Ian Abraham
>
> **摘要:** This paper investigates a sample-based solution to the hybrid mode control problem across non-differentiable and algorithmic hybrid modes. Our approach reasons about a set of hybrid control modes as an integer-based optimization problem where we select what mode to apply, when to switch to another mode, and the duration for which we are in a given control mode. A sample-based variation is derived to efficiently search the integer domain for optimal solutions. We find our formulation yields strong performance guarantees that can be applied to a number of robotics-related tasks. In addition, our approach is able to synthesize complex algorithms and policies to compound behaviors and achieve challenging tasks. Last, we demonstrate the effectiveness of our approach in real-world robotic examples that require reactive switching between long-term planning and high-frequency control.
>
---
#### [replaced 050] ROScopter: A Multirotor Autopilot based on ROSflight 2.0
- **分类: cs.RO**

- **简介: 该论文提出ROScopter，一个基于ROSflight 2.0的多旋翼自动驾驶仪，旨在简化研究代码的仿真与硬件测试，解决代码模块化和易用性问题。**

- **链接: [https://arxiv.org/pdf/2603.05404](https://arxiv.org/pdf/2603.05404)**

> **作者:** Jacob Moore; Ian Reid; Phil Tokumaru; Randy Beard; Tim McLain
>
> **备注:** Submitted to the 2026 International Conference on Unmanned Aerial Systems
>
> **摘要:** ROScopter is a lean multirotor autopilot built for researchers. ROScopter seeks to accelerate simulation and hardware testing of research code with an architecture that is both easy to understand and simple to modify. ROScopter is designed to interface with ROSflight 2.0 and runs entirely on an onboard flight computer, leveraging the features of ROS 2 to improve modularity. This work describes the architecture of ROScopter and how it can be used to test application code in both simulated and hardware environments. Hardware results of the default ROScopter behavior are presented, showing that ROScopter achieves similar performance to another state-of-the-art autopilot for basic waypoint-following maneuvers, but with a significantly reduced and more modular code-base.
>
---
#### [replaced 051] InsSo3D: Inertial Navigation System and 3D Sonar SLAM for turbid environment inspection
- **分类: cs.RO**

- **简介: 该论文属于SLAM任务，解决浑浊环境下水下结构的精准定位与建图问题。结合3D声呐和惯性导航系统，提出InsSo3D方法，有效纠正里程计偏差，提升水下环境的测绘精度。**

- **链接: [https://arxiv.org/pdf/2601.05805](https://arxiv.org/pdf/2601.05805)**

> **作者:** Simon Archieri; Ahmet Cinar; Shu Pan; Jonatan Scharff Willners; Michele Grimaldi; Ignacio Carlucho; Yvan Petillot
>
> **摘要:** This paper presents InsSo3D, an accurate and efficient method for large-scale 3D Simultaneous Localisation and Mapping (SLAM) using a 3D Sonar and an Inertial Navigation System (INS). Unlike traditional sonar, which produces 2D images containing range and azimuth information but lacks elevation information, 3D Sonar produces a 3D point cloud, which therefore does not suffer from elevation ambiguity. We introduce a robust and modern SLAM framework adapted to the 3D Sonar data using INS as prior, detecting loop closure and performing pose graph optimisation. We evaluated InsSo3D performance inside a test tank with access to ground truth data and in an outdoor flooded quarry. Comparisons to reference trajectories and maps obtained from an underwater motion tracking system and visual Structure From Motion (SFM) demonstrate that InsSo3D efficiently corrects odometry drift. The average trajectory error is below 21cm during a 50-minute-long mission, producing a map of 10m by 20m with a 9cm average reconstruction error, enabling safe inspection of natural or artificial underwater structures even in murky water conditions.
>
---
#### [replaced 052] Symmetry-Breaking in Multi-Agent Navigation: Winding Number-Aware MPC with a Learned Topological Strategy
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多智能体导航任务，解决对称性导致的死锁问题。提出WNumMPC方法，通过拓扑不变量和强化学习实现有效避障与协同。**

- **链接: [https://arxiv.org/pdf/2511.15239](https://arxiv.org/pdf/2511.15239)**

> **作者:** Tomoki Nakao; Kazumi Kasaura; Tadashi Kozuno
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** In distributed multi-agent navigation without explicit communication, agents can fall into symmetry-induced deadlocks because each agent must autonomously decide how to pass others. To address this problem, we propose WNumMPC, a hierarchical navigation method that quantifies cooperative symmetry-breaking strategies via a topological invariant, the winding number, and learns such strategies through reinforcement learning. The learning-based Planner outputs continuous-valued signed target winding numbers and dynamic importance weights to prioritize critical interactions in dense crossings. Then, the model-based Controller generates collision-free and efficient motions based on the strategy and weights provided by the Planner. Simulation and real-world robot experiments indicate that WNumMPC effectively avoids deadlocks and collisions and achieves better performance than the baselines, particularly in dense and symmetry-prone scenarios. These experiments also suggest that explicitly leveraging winding numbers yields robust sim-to-real transfer with minimal performance degradation. The code for the experiments is available at this https URL.
>
---
