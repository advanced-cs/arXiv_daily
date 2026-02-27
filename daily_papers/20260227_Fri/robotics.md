# 机器人 cs.RO

- **最新发布 43 篇**

- **更新 19 篇**

## 最新发布

#### [new 001] DigiArm: An Anthropomorphic 3D-Printed Prosthetic Hand with Enhanced Dexterity for Typing Tasks
- **分类: cs.RO**

- **简介: 该论文属于康复工程任务，旨在解决传统假肢缺乏精细操作能力的问题。研究设计了一种低成本3D打印仿生手，提升键盘输入和钢琴演奏等精细动作的控制能力。**

- **链接: [https://arxiv.org/pdf/2602.23017v1](https://arxiv.org/pdf/2602.23017v1)**

> **作者:** Dean Zadok; Tom Naamani; Yuval Bar-Ratson; Elisha Barash; Oren Salzman; Alon Wolf; Alex M. Bronstein; Nili Krausz
>
> **摘要:** Despite recent advancements, existing prosthetic limbs are unable to replicate the dexterity and intuitive control of the human hand. Current control systems for prosthetic hands are often limited to grasping, and commercial prosthetic hands lack the precision needed for dexterous manipulation or applications that require fine finger motions. Thus, there is a critical need for accessible and replicable prosthetic designs that enable individuals to interact with electronic devices and perform precise finger pressing, such as keyboard typing or piano playing, while preserving current prosthetic capabilities. This paper presents a low-cost, lightweight, 3D-printed robotic prosthetic hand, specifically engineered for enhanced dexterity with electronic devices such as a computer keyboard or piano, as well as general object manipulation. The robotic hand features a mechanism to adjust finger abduction/adduction spacing, a 2-D wrist with the inclusion of controlled ulnar/radial deviation optimized for typing, and control of independent finger pressing. We conducted a study to demonstrate how participants can use the robotic hand to perform keyboard typing and piano playing in real time, with different levels of finger and wrist motion. This supports the notion that our proposed design can allow for the execution of key typing motions more effectively than before, aiming to enhance the functionality of prosthetic hands.
>
---
#### [new 002] Hierarchical Trajectory Planning of Floating-Base Multi-Link Robot for Maneuvering in Confined Environments
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人路径规划任务，解决浮动基多连杆机器人在受限环境中的轨迹规划问题，提出分层规划框架，实现动态可行、避障的轨迹生成。**

- **链接: [https://arxiv.org/pdf/2602.22459v1](https://arxiv.org/pdf/2602.22459v1)**

> **作者:** Yicheng Chen; Jinjie Li; Haokun Liu; Zicheng Luo; Kotaro Kaneko; Moju Zhao
>
> **备注:** Accepted to IEEE T-ASE; DOI pending
>
> **摘要:** Floating-base multi-link robots can change their shape during flight, making them well-suited for applications in confined environments such as autonomous inspection and search and rescue. However, trajectory planning for such systems remains an open challenge because the problem lies in a high-dimensional, constraint-rich space where collision avoidance must be addressed together with kinematic limits and dynamic feasibility. This work introduces a hierarchical trajectory planning framework that integrates global guidance with configuration-aware local optimization. First, we exploit the dual nature of these robots - the root link as a rigid body for guidance and the articulated joints for flexibility - to generate global anchor states that decompose the planning problem into tractable segments. Second, we design a local trajectory planner that optimizes each segment in parallel with differentiable objectives and constraints, systematically enforcing kinematic feasibility and maintaining dynamic feasibility by avoiding control singularities. Third, we implement a complete system that directly processes point-cloud data, eliminating the need for handcrafted obstacle models. Extensive simulations and real-world experiments confirm that this framework enables an articulated aerial robot to exploit its morphology for maneuvering that rigid robots cannot achieve. To the best of our knowledge, this is the first planning framework for floating-base multi-link robots that has been demonstrated on a real robot to generate continuous, collision-free, and dynamically feasible trajectories directly from raw point-cloud inputs, without relying on handcrafted obstacle models.
>
---
#### [new 003] Metamorphic Testing of Vision-Language Action-Enabled Robots
- **分类: cs.RO; cs.SE**

- **简介: 该论文属于机器人测试任务，解决VLA模型的测试难题。通过元测试方法，提出可泛化的测试关系，有效检测故障，无需依赖具体测试用例。**

- **链接: [https://arxiv.org/pdf/2602.22579v1](https://arxiv.org/pdf/2602.22579v1)**

> **作者:** Pablo Valle; Sergio Segura; Shaukat Ali; Aitor Arrieta
>
> **摘要:** Vision-Language-Action (VLA) models are multimodal robotic task controllers that, given an instruction and visual inputs, produce a sequence of low-level control actions (or motor commands) enabling a robot to execute the requested task in the physical environment. These systems face the test oracle problem from multiple perspectives. On the one hand, a test oracle must be defined for each instruction prompt, which is a complex and non-generalizable approach. On the other hand, current state-of-the-art oracles typically capture symbolic representations of the world (e.g., robot and object states), enabling the correctness evaluation of a task, but fail to assess other critical aspects, such as the quality with which VLA-enabled robots perform a task. In this paper, we explore whether Metamorphic Testing (MT) can alleviate the test oracle problem in this context. To do so, we propose two metamorphic relation patterns and five metamorphic relations to assess whether changes to the test inputs impact the original trajectory of the VLA-enabled robots. An empirical study involving five VLA models, two simulated robots, and four robotic tasks shows that MT can effectively alleviate the test oracle problem by automatically detecting diverse types of failures, including, but not limited to, uncompleted tasks. More importantly, the proposed MRs are generalizable, making the proposed approach applicable across different VLA models, robots, and tasks, even in the absence of test oracles.
>
---
#### [new 004] Detection and Recognition: A Pairwise Interaction Framework for Mobile Service Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人社交感知任务，旨在解决移动服务机器人在人群环境中理解人类互动的问题。通过构建一对人交互框架，提升机器人导航的社交意识与安全性。**

- **链接: [https://arxiv.org/pdf/2602.22346v1](https://arxiv.org/pdf/2602.22346v1)**

> **作者:** Mengyu Liang; Sarah Gillet Schlegel; Iolanda Leite
>
> **摘要:** Autonomous mobile service robots, like lawnmowers or cleaning robots, operating in human-populated environments need to reason about local human-human interactions to support safe and socially aware navigation while fulfilling their tasks. For such robots, interaction understanding is not primarily a fine-grained recognition problem, but a perception problem under limited sensing quality and computational resources. Many existing approaches focus on holistic group activity recognition, which often requires complex and large models which may not be necessary for mobile service robots. Others use pairwise interaction methods which commonly rely on skeletal representations but their use in outdoor environments remains challenging. In this work, we argue that pairwise human interaction constitute a minimal yet sufficient perceptual unit for robot-centric social understanding. We study the problem of identifying interacting person pairs and classifying coarse-grained interaction behaviors sufficient for downstream group-level reasoning and service robot decision-making. To this end, we adopt a two-stage framework in which candidate interacting pairs are first identified based on lightweight geometric and motion cues, and interaction types are subsequently classified using a relation network. We evaluate the proposed approach on the JRDB dataset, where it achieves sufficient accuracy with reduced computational cost and model size compared to appearance-based methods. Additional experiments on the Collective Activity Dataset and zero shot test on a lawnmower-collected dataset further illustrate the generality of the proposed framework. These results suggest that pairwise geometric and motion cues provide a practical basis for interaction perception on mobile service robot providing a promising method for integration into mobile robot navigation stacks in future work. Code will be released soon
>
---
#### [new 005] Automated Robotic Needle Puncture for Percutaneous Dilatational Tracheostomy
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决PDT针刺精度低的问题。通过开发自动化机器人系统，提高针刺准确性并避免损伤。**

- **链接: [https://arxiv.org/pdf/2602.22952v1](https://arxiv.org/pdf/2602.22952v1)**

> **作者:** Yuan Tang; Bruno V. Adorno; Brendan A. McGrath; Andrew Weightman
>
> **摘要:** Percutaneous dilatational tracheostomy (PDT) is frequently performed on patients in intensive care units for prolonged mechanical ventilation. The needle puncture, as the most critical step of PDT, could lead to adverse consequences such as major bleeding and posterior tracheal wall perforation if performed inaccurately. Current practices of PDT puncture are all performed manually with no navigation assistance, which leads to large position and angular errors (5 mm and 30 degree). To improve the accuracy and reduce the difficulty of the PDT procedure, we propose a system that automates the needle insertion using a velocity-controlled robotic manipulator. Guided using pose data from two electromagnetic sensors, one at the needle tip and the other inside the trachea, the robotic system uses an adaptive constrained controller to adapt the uncertain kinematic parameters online and avoid collisions with the patient's body and tissues near the target. Simulations were performed to validate the controller's implementation, and then four hundred PDT punctures were performed on a mannequin to evaluate the position and angular accuracy. The absolute median puncture position error was 1.7 mm (IQR: 1.9 mm) and midline deviation was 4.13 degree (IQR: 4.55 degree), measured by the sensor inside the trachea. The small deviations from the nominal puncture in a simulated experimental setup and formal guarantees of collision-free insertions suggest the feasibility of the robotic PDT puncture.
>
---
#### [new 006] EgoAVFlow: Robot Policy Learning with Active Vision from Human Egocentric Videos via 3D Flow
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，解决从人类第一视角视频中学习操作和主动视觉的问题。通过3D流表示实现有效可见性维护与操作。**

- **链接: [https://arxiv.org/pdf/2602.22461v1](https://arxiv.org/pdf/2602.22461v1)**

> **作者:** Daesol Cho; Youngseok Jang; Danfei Xu; Sehoon Ha
>
> **摘要:** Egocentric human videos provide a scalable source of manipulation demonstrations; however, deploying them on robots requires active viewpoint control to maintain task-critical visibility, which human viewpoint imitation often fails to provide due to human-specific priors. We propose EgoAVFlow, which learns manipulation and active vision from egocentric videos through a shared 3D flow representation that supports geometric visibility reasoning and transfers without robot demonstrations. EgoAVFlow uses diffusion models to predict robot actions, future 3D flow, and camera trajectories, and refines viewpoints at test time with reward-maximizing denoising under a visibility-aware reward computed from predicted motion and scene geometry. Real-world experiments under actively changing viewpoints show that EgoAVFlow consistently outperforms prior human-demo-based baselines, demonstrating effective visibility maintenance and robust manipulation without robot demonstrations.
>
---
#### [new 007] Robust Helicopter Ship Deck Landing With Guaranteed Timing Using Shrinking-Horizon Model Predictive Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自主直升机着舰任务，解决动态环境下精准着陆问题。通过SHMPC和辅助控制器，实现时间约束下的鲁棒着陆控制。**

- **链接: [https://arxiv.org/pdf/2602.22714v1](https://arxiv.org/pdf/2602.22714v1)**

> **作者:** Philipp Schitz; Paolo Mercorelli; Johann C. Dauer
>
> **备注:** This version was submitted to the American Control Conference 2026 and has been accepted
>
> **摘要:** We present a runtime efficient algorithm for autonomous helicopter landings on moving ship decks based on Shrinking-Horizon Model Predictive Control (SHMPC). First, a suitable planning model capturing the relevant aspects of the full nonlinear helicopter dynamics is derived. Next, we use the SHMPC together with a touchdown controller stage to ensure a pre-specified maneuver time and an associated landing time window despite the presence of disturbances. A high disturbance rejection performance is achieved by designing an ancillary controller with disturbance feedback. Thus, given a target position and time, a safe landing with suitable terminal conditions is be guaranteed if the initial optimization problem is feasible. The efficacy of our approach is shown in simulation where all maneuvers achieve a high landing precision in strong winds while satisfying timing and operational constraints with maximum computation times in the millisecond range.
>
---
#### [new 008] SignVLA: A Gloss-Free Vision-Language-Action Framework for Real-Time Sign Language-Guided Robotic Manipulation
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文提出SignVLA，一种无需词素的视觉-语言-动作框架，用于实时手语引导机器人操作。解决手语到指令的映射问题，通过几何归一化和时间平滑处理手势流，实现可靠交互。**

- **链接: [https://arxiv.org/pdf/2602.22514v1](https://arxiv.org/pdf/2602.22514v1)**

> **作者:** Xinyu Tan; Ningwei Bai; Harry Gardener; Zhengyang Zhong; Luoyu Zhang; Liuhaichen Yang; Zhekai Duan; Monkgogi Galeitsiwe; Zezhi Tang
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** We present, to our knowledge, the first sign language-driven Vision-Language-Action (VLA) framework for intuitive and inclusive human-robot interaction. Unlike conventional approaches that rely on gloss annotations as intermediate supervision, the proposed system adopts a gloss-free paradigm and directly maps visual sign gestures to semantic instructions. This design reduces annotation cost and avoids the information loss introduced by gloss representations, enabling more natural and scalable multimodal interaction. In this work, we focus on a real-time alphabet-level finger-spelling interface that provides a robust and low-latency communication channel for robotic control. Compared with large-scale continuous sign language recognition, alphabet-level interaction offers improved reliability, interpretability, and deployment feasibility in safety-critical embodied environments. The proposed pipeline transforms continuous gesture streams into coherent language commands through geometric normalization, temporal smoothing, and lexical refinement, ensuring stable and consistent interaction. Furthermore, the framework is designed to support future integration of transformer-based gloss-free sign language models, enabling scalable word-level and sentence-level semantic understanding. Experimental results demonstrate the effectiveness of the proposed system in grounding sign-derived instructions into precise robotic actions under diverse interaction scenarios. These results highlight the potential of the framework to advance accessible, scalable, and multimodal embodied intelligence.
>
---
#### [new 009] Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究扩散模型在端到端自动驾驶中的应用，解决真实场景下规划性能不足的问题。通过大量实车数据和测试，提出HDP框架，提升驾驶性能。**

- **链接: [https://arxiv.org/pdf/2602.22801v1](https://arxiv.org/pdf/2602.22801v1)**

> **作者:** Yinan Zheng; Tianyi Tan; Bin Huang; Enguang Liu; Ruiming Liang; Jianlin Zhang; Jianwei Cui; Guang Chen; Kun Ma; Hangjun Ye; Long Chen; Ya-Qin Zhang; Xianyuan Zhan; Jingjing Liu
>
> **摘要:** Diffusion models have become a popular choice for decision-making tasks in robotics, and more recently, are also being considered for solving autonomous driving tasks. However, their applications and evaluations in autonomous driving remain limited to simulation-based or laboratory settings. The full strength of diffusion models for large-scale, complex real-world settings, such as End-to-End Autonomous Driving (E2E AD), remains underexplored. In this study, we conducted a systematic and large-scale investigation to unleash the potential of the diffusion models as planners for E2E AD, based on a tremendous amount of real-vehicle data and road testing. Through comprehensive and carefully controlled studies, we identify key insights into the diffusion loss space, trajectory representation, and data scaling that significantly impact E2E planning performance. Moreover, we also provide an effective reinforcement learning post-training strategy to further enhance the safety of the learned planner. The resulting diffusion-based learning framework, Hyper Diffusion Planner} (HDP), is deployed on a real-vehicle platform and evaluated across 6 urban driving scenarios and 200 km of real-world testing, achieving a notable 10x performance improvement over the base model. Our work demonstrates that diffusion models, when properly designed and trained, can serve as effective and scalable E2E AD planners for complex, real-world autonomous driving tasks.
>
---
#### [new 010] Performance and Experimental Analysis of Strain-based Models for Continuum Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人建模任务，旨在评估应变模型在连续机器人中的性能。通过实验验证第三阶应变插值方法的形状重建能力，对比现有模型，提升建模精度与效率。**

- **链接: [https://arxiv.org/pdf/2602.22854v1](https://arxiv.org/pdf/2602.22854v1)**

> **作者:** Annika Delucchi; Vincenzo Di Paola; Andreas Müller; and Matteo Zoppi
>
> **摘要:** Although strain-based models have been widely adopted in robotics, no comparison beyond the uniform bending test is commonly recognized to assess their performance. In addition, the increasing effort in prototyping continuum robots highlights the need to assess the applicability of these models and the necessity of comprehensive performance evaluation. To address this gap, this work investigates the shape reconstruction abilities of a third-order strain interpolation method, examining its ability to capture both individual and combined deformation effects. These results are compared and discussed against the Geometric-Variable Strain approach. Subsequently, simulation results are experimentally verified by reshaping a slender rod while recording the resulting configurations using cameras. The rod configuration is imposed using a manipulator displacing one of its tips and extracted through reflective markers, without the aid of any other external sensor -- i.e. strain gauges or wrench sensors placed along the rod. The experiments demonstrate good agreement between the model predictions and observed shapes, with average error of 0.58% of the rod length and average computational time of 0.32s per configuration, outperforming existing models.
>
---
#### [new 011] Designing Robots for Families: In-Situ Prototyping for Contextual Reminders on Family Routines
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决机器人在家庭中有效支持日常 routines 的问题。通过与家庭合作设计机器人行为，进行实地测试，探索机器人在家庭动态中的角色与挑战。**

- **链接: [https://arxiv.org/pdf/2602.22628v1](https://arxiv.org/pdf/2602.22628v1)**

> **作者:** Michael F. Xu; Enhui Zhao; Yawen Zhang; Joseph E. Michaelis; Sarah Sebo; Bilge Mutlu
>
> **备注:** Proceedings of the 21st ACM/IEEE International Conference on Human Robot Interaction (HRI 2026)
>
> **摘要:** Robots are increasingly entering the daily lives of families, yet their successful integration into domestic life remains a challenge. We explore family routines as a critical entry point for understanding how robots might find a sustainable role in everyday family settings. Together with each of the ten families, we co-designed robot interactions and behaviors, and a plan for the robot to support their chosen routines, accounting for contextual factors such as timing, participants, locations, and the activities in the environment. We then designed, prototyped, and deployed a mobile social robot as a four-day, in-home user study. Families welcomed the robot's reminders, with parents especially appreciating the offloading of some reminding tasks. At the same time, interviews revealed tensions around timing, authority, and family dynamics, highlighting the complexity of integrating robots into households beyond the immediate task of reminders. Based on these insights, we offer design implications for robot-facilitated contextual reminders and discuss broader considerations for designing robots for family settings.
>
---
#### [new 012] Interface-Aware Trajectory Reconstruction of Limited Demonstrations for Robot Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决接口限制导致的运动不优问题。通过轨迹重构算法，将受限演示提升至完整控制空间，提高运动效率与用户意图契合度。**

- **链接: [https://arxiv.org/pdf/2602.23287v1](https://arxiv.org/pdf/2602.23287v1)**

> **作者:** Demiana R. Barsoum; Mahdieh Nejati Javaremi; Larisa Y. C. Loke; Brenna D. Argall
>
> **备注:** 13 pages, 8 figures, to appear in the proceedings of the 2026 Human-Robot Interaction (HRI) Conference
>
> **摘要:** Assistive robots offer agency to humans with severe motor impairments. Often, these users control high-DoF robots through low-dimensional interfaces, such as using a 1-D sip-and-puff interface to operate a 6-DoF robotic arm. This mismatch results in having access to only a subset of control dimensions at a given time, imposing unintended and artificial constraints on robot motion. As a result, interface-limited demonstrations embed suboptimal motions that reflect interface restrictions rather than user intent. To address this, we present a trajectory reconstruction algorithm that reasons about task, environment, and interface constraints to lift demonstrations into the robot's full control space. We evaluate our approach using real-world demonstrations of ADL-inspired tasks performed via a 2-D joystick and 1-D sip-and-puff control interface, teleoperating two distinct 7-DoF robotic arms. Analyses of the reconstructed demonstrations and derived control policies show that lifted trajectories are faster and more efficient than their interface-constrained counterparts while respecting user preferences.
>
---
#### [new 013] Does the testing environment matter? Carsickness across on-road, test-track, and driving simulator conditions
- **分类: cs.RO; cs.ET**

- **简介: 该论文属于人因工程任务，旨在研究不同驾驶环境对晕动症的影响。通过对比真实道路、测试场地和驾驶模拟器中的晕动症表现，发现模拟器环境下的晕动症较低，因其无法完全再现低频运动。**

- **链接: [https://arxiv.org/pdf/2602.22671v1](https://arxiv.org/pdf/2602.22671v1)**

> **作者:** Georgios Papaioannou; Barys Shyrokau
>
> **摘要:** Carsickness has gained significant attention with the rise of automated vehicles, prompting extensive research across on-road, test-track, and driving simulator environments to understand its occurrence and develop mitigation strategies. However, the lack of carsickness standardization complicates comparisons across studies and environments. Previous works demonstrate measurement validity between two setups at most (e.g., on-road vs. driving simulator), leaving gaps in multi-environment comparisons. This study investigates the recreation of an on-road motion sickness exposure - previously replicated on a test track - using a motion-based driving simulator. Twenty-eight participants performed an eyes-off-road non-driving task while reporting motion sickness using the Misery Scale during the experiment and the Motion Sickness Assessment Questionnaire afterward. Psychological factors known to influence motion sickness were also assessed. The results present subjective and objective measurements for motion sickness across the considered environments. In this paper, acceleration measurements, objective metrics and subjective motion sickness ratings across environments are compared, highlighting key differences in sickness occurrence for simulator-based research validity. Significantly lower motion sickness scores are reported in the simulator compared to on-road and test-track conditions, due to its limited working envelope to reproduce low-frequency (<0.5 Hz) motions, which are the most provocative for motion sickness.
>
---
#### [new 014] GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人抓取任务，旨在提升模仿学习策略的抓取精度和泛化能力。通过引入先验知识的扩散策略，提高抓取动作的准确性和适应性。**

- **链接: [https://arxiv.org/pdf/2602.22862v1](https://arxiv.org/pdf/2602.22862v1)**

> **作者:** Enda Xiang; Haoxiang Ma; Xinzhu Ma; Zicheng Liu; Di Huang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** This paper focuses on enhancing the grasping precision and generalization of manipulation policies learned via imitation learning. Diffusion-based policy learning methods have recently become the mainstream approach for robotic manipulation tasks. As grasping is a critical subtask in manipulation, the ability of imitation-learned policies to execute precise and generalizable grasps merits particular attention. Existing imitation learning techniques for grasping often suffer from imprecise grasp executions, limited spatial generalization, and poor object generalization. To address these challenges, we incorporate grasp prior knowledge into the diffusion policy framework. In particular, we employ a latent diffusion policy to guide action chunk decoding with grasp pose prior, ensuring that generated motion trajectories adhere closely to feasible grasp configurations. Furthermore, we introduce a self-supervised reconstruction objective during diffusion to embed the graspness prior: at each reverse diffusion step, we reconstruct wrist-camera images back-projected the graspness from the intermediate representations. Both simulation and real robot experiments demonstrate that our approach significantly outperforms baseline methods and exhibits strong dynamic grasping capabilities.
>
---
#### [new 015] When to Act, Ask, or Learn: Uncertainty-Aware Policy Steering
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人行为适应任务，解决VLM过自信导致的策略选择问题。提出UPS框架，联合考虑语义不确定性和动作可行性，选择执行、询问或干预策略，提升系统持续学习能力。**

- **链接: [https://arxiv.org/pdf/2602.22474v1](https://arxiv.org/pdf/2602.22474v1)**

> **作者:** Jessie Yuan; Yilin Wu; Andrea Bajcsy
>
> **摘要:** Policy steering is an emerging way to adapt robot behaviors at deployment-time: a learned verifier analyzes low-level action samples proposed by a pre-trained policy (e.g., diffusion policy) and selects only those aligned with the task. While Vision-Language Models (VLMs) are promising general-purpose verifiers due to their reasoning capabilities, existing frameworks often assume these models are well-calibrated. In practice, the overconfident judgment from VLM can degrade the steering performance under both high-level semantic uncertainty in task specifications and low-level action uncertainty or incapability of the pre-trained policy. We propose uncertainty-aware policy steering (UPS), a framework that jointly reasons about semantic task uncertainty and low-level action feasibility, and selects an uncertainty resolution strategy: execute a high-confidence action, clarify task ambiguity via natural language queries, or ask for action interventions to correct the low-level policy when it is deemed incapable at the task. We leverage conformal prediction to calibrate the composition of the VLM and the pre-trained base policy, providing statistical assurances that the verifier selects the correct strategy. After collecting interventions during deployment, we employ residual learning to improve the capability of the pre-trained policy, enabling the system to learn continually but with minimal expensive human feedback. We demonstrate our framework through experiments in simulation and on hardware, showing that UPS can disentangle confident, ambiguous, and incapable scenarios and minimizes expensive user interventions compared to uncalibrated baselines and prior human- or robot-gated continual learning approaches. Videos can be found at https://jessie-yuan.github.io/ups/
>
---
#### [new 016] SPARR: Simulation-based Policies with Asymmetric Real-world Residuals for Assembly
- **分类: cs.RO**

- **简介: 该论文提出SPARR方法，解决机器人装配中的sim-to-real差距问题。结合仿真训练的基策略与真实世界的残差策略，提升装配成功率与适应性。**

- **链接: [https://arxiv.org/pdf/2602.23253v1](https://arxiv.org/pdf/2602.23253v1)**

> **作者:** Yijie Guo; Iretiayo Akinola; Lars Johannsmeier; Hugo Hadfield; Abhishek Gupta; Yashraj Narang
>
> **摘要:** Robotic assembly presents a long-standing challenge due to its requirement for precise, contact-rich manipulation. While simulation-based learning has enabled the development of robust assembly policies, their performance often degrades when deployed in real-world settings due to the sim-to-real gap. Conversely, real-world reinforcement learning (RL) methods avoid the sim-to-real gap, but rely heavily on human supervision and lack generalization ability to environmental changes. In this work, we propose a hybrid approach that combines a simulation-trained base policy with a real-world residual policy to efficiently adapt to real-world variations. The base policy, trained in simulation using low-level state observations and dense rewards, provides strong priors for initial behavior. The residual policy, learned in the real world using visual observations and sparse rewards, compensates for discrepancies in dynamics and sensor noise. Extensive real-world experiments demonstrate that our method, SPARR, achieves near-perfect success rates across diverse two-part assembly tasks. Compared to the state-of-the-art zero-shot sim-to-real methods, SPARR improves success rates by 38.4% while reducing cycle time by 29.7%. Moreover, SPARR requires no human expertise, in contrast to the state-of-the-art real-world RL approaches that depend heavily on human supervision.
>
---
#### [new 017] SODA-CitrON: Static Object Data Association by Clustering Multi-Modal Sensor Detections Online
- **分类: cs.RO**

- **简介: 该论文属于静态目标数据关联任务，解决多模态传感器检测的在线融合与跟踪问题。提出SODA-CitrON方法，实现静态目标的聚类与持续跟踪。**

- **链接: [https://arxiv.org/pdf/2602.22243v1](https://arxiv.org/pdf/2602.22243v1)**

> **作者:** Jan Nausner; Kilian Wohlleben; Michael Hubner
>
> **备注:** 8 pages, 5 figures; Submitted to the 2026 International Conference on Information Fusion (FUSION 2026). Under review
>
> **摘要:** The online fusion and tracking of static objects from heterogeneous sensor detections is a fundamental problem in robotics, autonomous systems, and environmental mapping. Although classical data association approaches such as JPDA are well suited for dynamic targets, they are less effective for static objects observed intermittently and with heterogeneous uncertainties, where motion models provide minimal discriminative with respect to clutter. In this paper, we propose a novel method for static object data association by clustering multi-modal sensor detections online (SODA-CitrON), while simultaneously estimating positions and maintaining persistent tracks for an unknown number of objects. The proposed unsupervised machine learning approach operates in a fully online manner and handles temporally uncorrelated and multi-sensor measurements. Additionally, it has a worst-case loglinear complexity in the number of sensor detections while providing full output explainability. We evaluate the proposed approach in different Monte Carlo simulation scenarios and compare it against state-of-the-art methods, including Bayesian filtering, DBSTREAM clustering, and JPDA. The results demonstrate that SODA-CitrON consistently outperforms the compared methods in terms of F1 score, position RMSE, MOTP, and MOTA in the static object mapping scenarios studied.
>
---
#### [new 018] Pixel2Catch: Multi-Agent Sim-to-Real Transfer for Agile Manipulation with a Single RGB Camera
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决单目视觉下物体运动感知与控制问题。通过多智能体强化学习，实现从仿真到现实的灵活操作迁移。**

- **链接: [https://arxiv.org/pdf/2602.22733v1](https://arxiv.org/pdf/2602.22733v1)**

> **作者:** Seongyong Kim; Junhyeon Cho; Kang-Won Lee; Soo-Chul Lim
>
> **摘要:** To catch a thrown object, a robot must be able to perceive the object's motion and generate control actions in a timely manner. Rather than explicitly estimating the object's 3D position, this work focuses on a novel approach that recognizes object motion using pixel-level visual information extracted from a single RGB image. Such visual cues capture changes in the object's position and scale, allowing the policy to reason about the object's motion. Furthermore, to achieve stable learning in a high-DoF system composed of a robot arm equipped with a multi-fingered hand, we design a heterogeneous multi-agent reinforcement learning framework that defines the arm and hand as independent agents with distinct roles. Each agent is trained cooperatively using role-specific observations and rewards, and the learned policies are successfully transferred from simulation to the real world.
>
---
#### [new 019] Rethinking the Practicality of Vision-language-action Model: A Comprehensive Benchmark and An Improved Baseline
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在提升其实用性。针对参数过大、预训练成本高、适用性差的问题，提出基准和轻量模型LLaVA-VLA，实现高效部署与广泛适用。**

- **链接: [https://arxiv.org/pdf/2602.22663v1](https://arxiv.org/pdf/2602.22663v1)**

> **作者:** Wenxuan Song; Jiayi Chen; Xiaoquan Sun; Huashuo Lei; Yikai Qin; Wei Zhao; Pengxiang Ding; Han Zhao; Tongxin Wang; Pengxu Hou; Zhide Zhong; Haodong Yan; Donglin Wang; Jun Ma; Haoang Li
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a generalist robotic agent. However, existing VLAs are hindered by excessive parameter scales, prohibitive pre-training requirements, and limited applicability to diverse embodiments. To improve the practicality of VLAs, we propose a comprehensive benchmark and an improved baseline. First, we propose CEBench, a new benchmark spanning diverse embodiments in both simulation and the real world with consideration of domain randomization. We collect 14.4k simulated trajectories and 1.6k real-world expert-curated trajectories to support training on CEBench. Second, using CEBench as our testbed, we study three critical aspects of VLAs' practicality and offer several key findings. Informed by these findings, we introduce LLaVA-VLA, a lightweight yet powerful VLA designed for practical deployment on consumer-grade GPUs. Architecturally, it integrates a compact VLM backbone with multi-view perception, proprioceptive tokenization, and action chunking. To eliminate reliance on costly pre-training, LLaVA-VLA adopts a two-stage training paradigm including post-training and fine-tuning. Furthermore, LLaVA-VLA extends the action space to unify navigation and manipulation. Experiments across embodiments demonstrate the capabilities of generalization and versatility of LLaVA-VLA , while real-world mobile manipulation experiments establish it as the first end-to-end VLA model for mobile manipulation. We will open-source all datasets, codes, and checkpoints upon acceptance to foster reproducibility and future research.
>
---
#### [new 020] Marinarium: a New Arena to Bring Maritime Robotics Closer to Shore
- **分类: cs.RO**

- **简介: 该论文介绍Marinarium，一个用于水下和太空仿生机器人实验的设施，解决仿真与现实环境差距问题，通过模块化设计实现多领域测试与验证。**

- **链接: [https://arxiv.org/pdf/2602.23053v1](https://arxiv.org/pdf/2602.23053v1)**

> **作者:** Ignacio Torroba; David Dorner; Victor Nan Fernandez-Ayala; Mart Kartasev; Joris Verhagen; Elias Krantz; Gregorio Marchesini; Carl Ljung; Pedro Roque; Chelsea Sidrane; Linda Van der Spaa; Nicola De Carli; Petter Ogren; Christer Fuglesang; Jana Tumova; Dimos V. Dimarogonas; Ivan Stenius
>
> **摘要:** This paper presents the Marinarium, a modular and stand-alone underwater research facility designed to provide a realistic testbed for maritime and space-analog robotic experimentation in a resource-efficient manner. The Marinarium combines a fully instrumented underwater and aerial operational volume, extendable via a retractable roof for real-weather conditions, a digital twin in the SMaRCSim simulator and tight integration with a space robotics laboratory. All of these result from design choices aimed at bridging simulation, laboratory validation, and field conditions. We compare the Marinarium to similar existing infrastructures and illustrate how its design enables a set of experiments in four open research areas within field robotics. First, we exploit high-fidelity dynamics data from the tank to demonstrate the potential of learning-based system identification approaches applied to underwater vehicles. We further highlight the versatility of the multi-domain operating volume via a rendezvous mission with a heterogeneous fleet of robots across underwater, surface, and air. We then illustrate how the presented digital twin can be utilized to reduce the reality gap in underwater simulation. Finally, we demonstrate the potential of underwater surrogates for spacecraft navigation validation by executing spatiotemporally identical inspection tasks on a planar space-robot emulator and a neutrally buoyant \gls{rov}. In this work, by sharing the insights obtained and rationale behind the design and construction of the Marinarium, we hope to provide the field robotics research community with a blueprint for bridging the gap between controlled and real offshore and space robotics experimentation.
>
---
#### [new 021] Grasp, Slide, Roll: Comparative Analysis of Contact Modes for Tactile-Based Shape Reconstruction
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉感知任务，旨在提升物体形状重建效率。通过比较不同接触模式，优化触觉数据获取，减少物理接触次数并提高精度。**

- **链接: [https://arxiv.org/pdf/2602.23206v1](https://arxiv.org/pdf/2602.23206v1)**

> **作者:** Chung Hee Kim; Shivani Kamtikar; Tye Brady; Taskin Padir; Joshua Migdal
>
> **备注:** 8 pages, 11 figures, Accepted by ICRA 2026
>
> **摘要:** Tactile sensing allows robots to gather detailed geometric information about objects through physical interaction, complementing vision-based approaches. However, efficiently acquiring useful tactile data remains challenging due to the time-consuming nature of physical contact and the need to strategically choose contact locations that maximize information gain while minimizing physical interactions. This paper studies how different contact modes affect object shape reconstruction using a tactile-enabled dexterous gripper. We compare three contact interaction modes: grasp-releasing, sliding induced by finger-grazing, and palm-rolling. These contact modes are combined with an information-theoretic exploration framework that guides subsequent sampling locations using a shape completion model. Our results show that the improved tactile sensing efficiency of finger-grazing and palm-rolling translates into faster convergence in shape reconstruction, requiring 34% fewer physical interactions while improving reconstruction accuracy by 55%. We validate our approach using a UR5e robot arm equipped with an Inspire-Robots Dexterous Hand, showing robust performance across primitive object geometries.
>
---
#### [new 022] Considering Perspectives for Automated Driving Ethics: Collective Risk in Vehicular Motion Planning
- **分类: cs.RO**

- **简介: 论文探讨自动驾驶车辆在决策中应考虑所有道路使用者的风险视角，而非仅从自身出发。任务是解决伦理问题，通过多视角风险平衡提升整体交通安全性与公平性。**

- **链接: [https://arxiv.org/pdf/2602.22940v1](https://arxiv.org/pdf/2602.22940v1)**

> **作者:** Leon Tolksdorf; Arturo Tejada; Christian Birkner; Nathan van de Wouw
>
> **备注:** 17 pages, 6 figures, 2 tables
>
> **摘要:** Recent automated vehicle (AV) motion planning strategies evolve around minimizing risk in road traffic. However, they exclusively consider risk from the AV's perspective and, as such, do not address the ethicality of its decisions for other road users. We argue that this does not reduce the risk of each road user, as risk may be different from the perspective of each road user. Indeed, minimizing the risk from the AV's perspective may not imply that the risk from the perspective of other road users is also being minimized; in fact, it may even increase. To test this hypothesis, we propose an AV motion planning strategy that supports switching risk minimization strategies between all road user perspectives. We find that the risk from the perspective of other road users can generally be considered different to the risk from the AV's perspective. Taking a collective risk perspective, i.e., balancing the risks of all road users, we observe an AV that minimizes overall traffic risk the best, while putting itself at slightly higher risk for the benefit of others, which is consistent with human driving behavior. In addition, adopting a collective risk minimization strategy can also be beneficial to the AV's travel efficiency by acting assertively when other road users maintain a low risk estimate of the AV. Yet, the AV drives conservatively when its planned actions are less predictable to other road users, i.e., associated with high risk. We argue that such behavior is a form of self-reflection and a natural prerequisite for socially acceptable AV behavior. We conclude that to facilitate ethicality in road traffic that includes AVs, the risk-perspective of each road user must be considered in the decision-making of AVs.
>
---
#### [new 023] An Empirical Analysis of Cooperative Perception for Occlusion Risk Mitigation
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶感知任务，解决遮挡风险评估问题。提出RTL指标，分析V2X部署效果，优化通信框架以提升安全性能。**

- **链接: [https://arxiv.org/pdf/2602.23051v1](https://arxiv.org/pdf/2602.23051v1)**

> **作者:** Aihong Wang; Tenghui Xie; Fuxi Wen; Jun Li
>
> **备注:** Accepted for publication in IEEE Internet of Things Journal (Regular Article), 2026. DOI: 10.1109/JIOT.2026.3668184
>
> **摘要:** Occlusions present a significant challenge for connected and automated vehicles, as they can obscure critical road users from perception systems. Traditional risk metrics often fail to capture the cumulative nature of these threats over time adequately. In this paper, we propose a novel and universal risk assessment metric, the Risk of Tracking Loss (RTL), which aggregates instantaneous risk intensity throughout occluded periods. This provides a holistic risk profile that encompasses both high-intensity, short-term threats and prolonged exposure. Utilizing diverse and high-fidelity real-world datasets, a large-scale statistical analysis is conducted to characterize occlusion risk and validate the effectiveness of the proposed metric. The metric is applied to evaluate different vehicle-to-everything (V2X) deployment strategies. Our study shows that full V2X penetration theoretically eliminates this risk, the reduction is highly nonlinear; a substantial statistical benefit requires a high penetration threshold of 75-90%. To overcome this limitation, we propose a novel asymmetric communication framework that allows even non-connected vehicles to receive warnings. Experimental results demonstrate that this paradigm achieves better risk mitigation performance. We found that our approach at 25% penetration outperforms the traditional symmetric model at 75%, and benefits saturate at only 50% penetration. This work provides a crucial risk assessment metric and a cost-effective, strategic roadmap for accelerating the safety benefits of V2X deployment.
>
---
#### [new 024] LeRobot: An Open-Source Library for End-to-End Robot Learning
- **分类: cs.RO**

- **简介: 该论文提出开源库LeRobot，解决机器人学习工具碎片化问题，整合从数据收集到算法实现的完整流程，支持高效、可扩展的端到端机器人学习。**

- **链接: [https://arxiv.org/pdf/2602.22818v1](https://arxiv.org/pdf/2602.22818v1)**

> **作者:** Remi Cadene; Simon Aliberts; Francesco Capuano; Michel Aractingi; Adil Zouitine; Pepijn Kooijmans; Jade Choghari; Martino Russi; Caroline Pascal; Steven Palma; Mustafa Shukor; Jess Moss; Alexander Soare; Dana Aubakirova; Quentin Lhoest; Quentin Gallouédec; Thomas Wolf
>
> **备注:** https://github.com/huggingface/lerobot
>
> **摘要:** Robotics is undergoing a significant transformation powered by advances in high-level control techniques based on machine learning, giving rise to the field of robot learning. Recent progress in robot learning has been accelerated by the increasing availability of affordable teleoperation systems, large-scale openly available datasets, and scalable learning-based methods. However, development in the field of robot learning is often slowed by fragmented, closed-source tools designed to only address specific sub-components within the robotics stack. In this paper, we present \texttt{lerobot}, an open-source library that integrates across the entire robot learning stack, from low-level middleware communication for motor controls to large-scale dataset collection, storage and streaming. The library is designed with a strong focus on real-world robotics, supporting accessible hardware platforms while remaining extensible to new embodiments. It also supports efficient implementations for various state-of-the-art robot learning algorithms from multiple prominent paradigms, as well as a generalized asynchronous inference stack. Unlike traditional pipelines which heavily rely on hand-crafted techniques, \texttt{lerobot} emphasizes scalable learning approaches that improve directly with more data and compute. Designed for accessibility, scalability, and openness, \texttt{lerobot} lowers the barrier to entry for researchers and practitioners to robotics while providing a platform for reproducible, state-of-the-art robot learning.
>
---
#### [new 025] Bayesian Preference Elicitation: Human-In-The-Loop Optimization of An Active Prosthesis
- **分类: cs.RO**

- **简介: 该论文属于人机协同优化任务，旨在解决假肢调参效率低、不贴合用户需求的问题，通过贝叶斯偏好优化方法实现个性化控制。**

- **链接: [https://arxiv.org/pdf/2602.22922v1](https://arxiv.org/pdf/2602.22922v1)**

> **作者:** Sophia Taddei; Wouter Koppen; Eligia Alfio; Stefano Nuzzo; Louis Flynn; Maria Alejandra Diaz; Sebastian Rojas Gonzalez; Tom Dhaene; Kevin De Pauw; Ivo Couckuyt; Tom Verstraten
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Tuning active prostheses for people with amputation is time-consuming and relies on metrics that may not fully reflect user needs. We introduce a human-in-the-loop optimization (HILO) approach that leverages direct user preferences to personalize a standard four-parameter prosthesis controller efficiently. Our method employs preference-based Multiobjective Bayesian Optimization that uses a state-or-the-art acquisition function especially designed for preference learning, and includes two algorithmic variants: a discrete version (\textit{EUBO-LineCoSpar}), and a continuous version (\textit{BPE4Prost}). Simulation results on benchmark functions and real-application trials demonstrate efficient convergence, robust preference elicitation, and measurable biomechanical improvements, illustrating the potential of preference-driven tuning for user-centered prosthesis control.
>
---
#### [new 026] Towards Intelligible Human-Robot Interaction: An Active Inference Approach to Occluded Pedestrian Scenarios
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决遮挡行人带来的安全问题。通过主动推理框架，结合RBPF和CEM-MPPI方法，提升决策的准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2602.23109v1](https://arxiv.org/pdf/2602.23109v1)**

> **作者:** Kai Chen; Yuyao Huang; Guang Chen
>
> **备注:** 14 pages, 6 figures, Proceedings of the 2026 ACM/IEEE International Conference on Human-Robot Interaction (HRI'26)
>
> **摘要:** The sudden appearance of occluded pedestrians presents a critical safety challenge in autonomous driving. Conventional rule-based or purely data-driven approaches struggle with the inherent high uncertainty of these long-tail scenarios. To tackle this challenge, we propose a novel framework grounded in Active Inference, which endows the agent with a human-like, belief-driven mechanism. Our framework leverages a Rao-Blackwellized Particle Filter (RBPF) to efficiently estimate the pedestrian's hybrid state. To emulate human-like cognitive processes under uncertainty, we introduce a Conditional Belief Reset mechanism and a Hypothesis Injection technique to explicitly model beliefs about the pedestrian's multiple latent intentions. Planning is achieved via a Cross-Entropy Method (CEM) enhanced Model Predictive Path Integral (MPPI) controller, which synergizes the efficient, iterative search of CEM with the inherent robustness of MPPI. Simulation experiments demonstrate that our approach significantly reduces the collision rate compared to reactive, rule-based, and reinforcement learning (RL) baselines, while also exhibiting explainable and human-like driving behavior that reflects the agent's internal belief state.
>
---
#### [new 027] Sapling-NeRF: Geo-Localised Sapling Reconstruction in Forests for Ecological Monitoring
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于生态监测任务，旨在解决森林幼树三维重建与精准定位问题。通过融合NeRF、LiDAR SLAM和GNSS，实现幼树的高精度结构重建与长期跟踪。**

- **链接: [https://arxiv.org/pdf/2602.22731v1](https://arxiv.org/pdf/2602.22731v1)**

> **作者:** Miguel Ángel Muñoz-Bañón; Nived Chebrolu; Sruthi M. Krishna Moorthy; Yifu Tao; Fernando Torres; Roberto Salguero-Gómez; Maurice Fallon
>
> **摘要:** Saplings are key indicators of forest regeneration and overall forest health. However, their fine-scale architectural traits are difficult to capture with existing 3D sensing methods, which make quantitative evaluation difficult. Terrestrial Laser Scanners (TLS), Mobile Laser Scanners (MLS), or traditional photogrammetry approaches poorly reconstruct thin branches, dense foliage, and lack the scale consistency needed for long-term monitoring. Implicit 3D reconstruction methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) are promising alternatives, but cannot recover the true scale of a scene and lack any means to be accurately geo-localised. In this paper, we present a pipeline which fuses NeRF, LiDAR SLAM, and GNSS to enable repeatable, geo-localised ecological monitoring of saplings. Our system proposes a three-level representation: (i) coarse Earth-frame localisation using GNSS, (ii) LiDAR-based SLAM for centimetre-accurate localisation and reconstruction, and (iii) NeRF-derived object-centric dense reconstruction of individual saplings. This approach enables repeatable quantitative evaluation and long-term monitoring of sapling traits. Our experiments in forest plots in Wytham Woods (Oxford, UK) and Evo (Finland) show that stem height, branching patterns, and leaf-to-wood ratios can be captured with increased accuracy as compared to TLS. We demonstrate that accurate stem skeletons and leaf distributions can be measured for saplings with heights between 0.5m and 2m in situ, giving ecologists access to richer structural and quantitative data for analysing forest dynamics.
>
---
#### [new 028] InCoM: Intent-Driven Perception and Structured Coordination for Whole-Body Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人移动操作任务，解决整体身体控制耦合与感知注意力分配问题。提出InCoM框架，通过意图驱动的感知和结构化协调提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2602.23024v1](https://arxiv.org/pdf/2602.23024v1)**

> **作者:** Jiahao Liu; Cui Wenbo; Haoran Li; Dongbin Zhao
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Whole-body mobile manipulation is a fundamental capability for general-purpose robotic agents, requiring both coordinated control of the mobile base and manipulator and robust perception under dynamically changing viewpoints. However, existing approaches face two key challenges: strong coupling between base and arm actions complicates whole-body control optimization, and perceptual attention is often poorly allocated as viewpoints shift during mobile manipulation. We propose InCoM, an intent-driven perception and structured coordination framework for whole-body mobile manipulation. InCoM infers latent motion intent to dynamically reweight multi-scale perceptual features, enabling stage-adaptive allocation of perceptual attention. To support robust cross-modal perception, InCoM further incorporates a geometric-semantic structured alignment mechanism that enhances multimodal correspondence. On the control side, we design a decoupled coordinated flow matching action decoder that explicitly models coordinated base-arm action generation, alleviating optimization difficulties caused by control coupling. Without access to privileged perceptual information, InCoM outperforms state-of-the-art methods on three ManiSkill-HAB scenarios by 28.2%, 26.1%, and 23.6% in success rate, demonstrating strong effectiveness for whole-body mobile manipulation.
>
---
#### [new 029] DySL-VLA: Efficient Vision-Language-Action Model Inference via Dynamic-Static Layer-Skipping for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文提出DySL-VLA模型，解决机器人操作中VLA模型计算成本高的问题，通过动态跳过非关键层提升效率。**

- **链接: [https://arxiv.org/pdf/2602.22896v1](https://arxiv.org/pdf/2602.22896v1)**

> **作者:** Zebin Yang; Yijiahao Qi; Tong Xie; Bo Yu; Shaoshan Liu; Meng Li
>
> **备注:** DAC 2026
>
> **摘要:** Vision-Language-Action (VLA) models have shown remarkable success in robotic tasks like manipulation by fusing a language model's reasoning with a vision model's 3D understanding. However, their high computational cost remains a major obstacle for real-world applications that require real-time performance. We observe that the actions within a task have varying levels of importance: critical steps demand high precision, while less important ones can tolerate more variance. Leveraging this insight, we propose DySL-VLA, a novel framework that addresses computational cost by dynamically skipping VLA layers based on each action's importance. DySL-VLA categorizes its layers into two types: informative layers, which are consistently executed, and incremental layers, which can be selectively skipped. To intelligently skip layers without sacrificing accuracy, we invent a prior-post skipping guidance mechanism to determine when to initiate layer-skipping. We also propose a skip-aware two-stage knowledge distillation algorithm to efficiently train a standard VLA into a DySL-VLA. Our experiments indicate that DySL-VLA achieves 2.1% improvement in success length over Deer-VLA on the Calvin dataset, while simultaneously reducing trainable parameters by a factor of 85.7 and providing a 3.75x speedup relative to the RoboFlamingo baseline at iso-accuracy. Our code is available on https://github.com/PKU-SEC-Lab/DYSL_VLA.
>
---
#### [new 030] SCOPE: Skeleton Graph-Based Computation-Efficient Framework for Autonomous UAV Exploration
- **分类: cs.RO**

- **简介: 该论文属于自主无人机探索任务，解决资源受限设备上计算延迟和轨迹振荡问题。提出SCOPE框架，通过骨架图和分层规划提升效率。**

- **链接: [https://arxiv.org/pdf/2602.22707v1](https://arxiv.org/pdf/2602.22707v1)**

> **作者:** Kai Li; Shengtao Zheng; Linkun Xiu; Yuze Sheng; Xiao-Ping Zhang; Dongyue Huang; Xinlei Chen
>
> **备注:** This paper has been accepted for publication in the IEEE ROBOTICS AND AUTOMATION LETTERS (RA-L). Please cite the paper using appropriate formats
>
> **摘要:** Autonomous exploration in unknown environments is key for mobile robots, helping them perceive, map, and make decisions in complex areas. However, current methods often rely on frequent global optimization, suffering from high computational latency and trajectory oscillation, especially on resource-constrained edge devices. To address these limitations, we propose SCOPE, a novel framework that incrementally constructs a real-time skeletal graph and introduces Implicit Unknown Region Analysis for efficient spatial reasoning. The planning layer adopts a hierarchical on-demand strategy: the Proximal Planner generates smooth, high-frequency local trajectories, while the Region-Sequence Planner is activated only when necessary to optimize global visitation order. Comparative evaluations in simulation demonstrate that SCOPE achieves competitive exploration performance comparable to state-of-the-art global planners, while reducing computational cost by an average of 86.9%. Real-world experiments further validate the system's robustness and low latency in practical scenarios.
>
---
#### [new 031] A Perspective on Open Challenges in Deformable Object Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决变形物体操控中的挑战，如感知、建模与控制问题。工作包括回顾最新技术，探索多模态感知和强化学习等方法，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2602.22998v1](https://arxiv.org/pdf/2602.22998v1)**

> **作者:** Ryan Paul McKennaa; John Oyekan
>
> **备注:** 28 pages, 7 Figures
>
> **摘要:** Deformable object manipulation (DOM) represents a critical challenge in robotics, with applications spanning healthcare, manufacturing, food processing, and beyond. Unlike rigid objects, deformable objects exhibit infinite dimensionality, dynamic shape changes, and complex interactions with their environment, posing significant hurdles for perception, modeling, and control. This paper reviews the state of the art in DOM, focusing on key challenges such as occlusion handling, task generalization, and scalable, real-time solutions. It highlights advancements in multimodal perception systems, including the integration of multi-camera setups, active vision, and tactile sensing, which collectively address occlusion and improve adaptability in unstructured environments. Cutting-edge developments in physically informed reinforcement learning (RL) and differentiable simulations are explored, showcasing their impact on efficiency, precision, and scalability. The review also emphasizes the potential of simulated expert demonstrations and generative neural networks to standardize task specifications and bridge the simulation-to-reality gap. Finally, future directions are proposed, including the adoption of graph neural networks for high-level decision-making and the creation of comprehensive datasets to enhance DOM's real-world applicability. By addressing these challenges, DOM research can pave the way for versatile robotic systems capable of handling diverse and dynamic tasks with deformable objects.
>
---
#### [new 032] Simple Models, Real Swimming: Digital Twins for Tendon-Driven Underwater Robots
- **分类: cs.RO**

- **简介: 该论文属于软体机器人控制任务，旨在解决水下机器人运动模拟与控制难题。通过构建简单流体模型，实现高效仿真与学习，提升机器人运动性能。**

- **链接: [https://arxiv.org/pdf/2602.23283v1](https://arxiv.org/pdf/2602.23283v1)**

> **作者:** Mike Y. Michelis; Nana Obayashi; Josie Hughes; Robert K. Katzschmann
>
> **摘要:** Mimicking the graceful motion of swimming animals remains a core challenge in soft robotics due to the complexity of fluid-structure interaction and the difficulty of controlling soft, biomimetic bodies. Existing modeling approaches are often computationally expensive and impractical for complex control or reinforcement learning needed for realistic motions to emerge in robotic systems. In this work, we present a tendon-driven fish robot modeled in an efficient underwater swimmer environment using a simplified, stateless hydrodynamics formulation implemented in the widespread robotics framework MuJoCo. With just two real-world swimming trajectories, we identify five fluid parameters that allow a matching to experimental behavior and generalize across a range of actuation frequencies. We show that this stateless fluid model can generalize to unseen actuation and outperform classical analytical models such as the elongated body theory. This simulation environment runs faster than real-time and can easily enable downstream learning algorithms such as reinforcement learning for target tracking, reaching a 93% success rate. Due to the simplicity and ease of use of the model and our open-source simulation environment, our results show that even simple, stateless models -- when carefully matched to physical data -- can serve as effective digital twins for soft underwater robots, opening up new directions for scalable learning and control in aquatic environments.
>
---
#### [new 033] GeoWorld: Geometric World Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GeoWorld，解决视觉规划中的长期预测和几何结构保留问题。通过超球面JEPA和几何强化学习，提升多步规划效果。**

- **链接: [https://arxiv.org/pdf/2602.23058v1](https://arxiv.org/pdf/2602.23058v1)**

> **作者:** Zeyu Zhang; Danning Li; Ian Reid; Richard Hartley
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Energy-based predictive world models provide a powerful approach for multi-step visual planning by reasoning over latent energy landscapes rather than generating pixels. However, existing approaches face two major challenges: (i) their latent representations are typically learned in Euclidean space, neglecting the underlying geometric and hierarchical structure among states, and (ii) they struggle with long-horizon prediction, which leads to rapid degradation across extended rollouts. To address these challenges, we introduce GeoWorld, a geometric world model that preserves geometric structure and hierarchical relations through a Hyperbolic JEPA, which maps latent representations from Euclidean space onto hyperbolic manifolds. We further introduce Geometric Reinforcement Learning for energy-based optimization, enabling stable multi-step planning in hyperbolic latent space. Extensive experiments on CrossTask and COIN demonstrate around 3% SR improvement in 3-step planning and 2% SR improvement in 4-step planning compared to the state-of-the-art V-JEPA 2. Project website: https://steve-zeyu-zhang.github.io/GeoWorld.
>
---
#### [new 034] Physics Informed Viscous Value Representations
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习领域，解决离线目标条件策略的价值估计问题。通过引入物理信息的正则化，基于HJB方程的粘性解提升模型的几何一致性与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.23280v1](https://arxiv.org/pdf/2602.23280v1)**

> **作者:** Hrishikesh Viswanath; Juanwu Lu; S. Talha Bukhari; Damon Conover; Ziran Wang; Aniket Bera
>
> **摘要:** Offline goal-conditioned reinforcement learning (GCRL) learns goal-conditioned policies from static pre-collected datasets. However, accurate value estimation remains a challenge due to the limited coverage of the state-action space. Recent physics-informed approaches have sought to address this by imposing physical and geometric constraints on the value function through regularization defined over first-order partial differential equations (PDEs), such as the Eikonal equation. However, these formulations can often be ill-posed in complex, high-dimensional environments. In this work, we propose a physics-informed regularization derived from the viscosity solution of the Hamilton-Jacobi-Bellman (HJB) equation. By providing a physics-based inductive bias, our approach grounds the learning process in optimal control theory, explicitly regularizing and bounding updates during value iterations. Furthermore, we leverage the Feynman-Kac theorem to recast the PDE solution as an expectation, enabling a tractable Monte Carlo estimation of the objective that avoids numerical instability in higher-order gradients. Experiments demonstrate that our method improves geometric consistency, making it broadly applicable to navigation and high-dimensional, complex manipulation tasks. Open-source codes are available at https://github.com/HrishikeshVish/phys-fk-value-GCRL.
>
---
#### [new 035] UniScale: Unified Scale-Aware 3D Reconstruction for Multi-View Understanding via Prior Injection for Robotic Perception
- **分类: cs.CV; cs.RO**

- **简介: 论文提出UniScale，用于机器人感知的多视角3D重建任务，解决环境结构准确提取问题。通过统一模型联合估计相机参数、深度和场景尺度，结合几何先验提升性能。**

- **链接: [https://arxiv.org/pdf/2602.23224v1](https://arxiv.org/pdf/2602.23224v1)**

> **作者:** Mohammad Mahdavian; Gordon Tan; Binbin Xu; Yuan Ren; Dongfeng Bai; Bingbing Liu
>
> **摘要:** We present UniScale, a unified, scale-aware multi-view 3D reconstruction framework for robotic applications that flexibly integrates geometric priors through a modular, semantically informed design. In vision-based robotic navigation, the accurate extraction of environmental structure from raw image sequences is critical for downstream tasks. UniScale addresses this challenge with a single feed-forward network that jointly estimates camera intrinsics and extrinsics, scale-invariant depth and point maps, and the metric scale of a scene from multi-view images, while optionally incorporating auxiliary geometric priors when available. By combining global contextual reasoning with camera-aware feature representations, UniScale is able to recover the metric-scale of the scene. In robotic settings where camera intrinsics are known, they can be easily incorporated to improve performance, with additional gains obtained when camera poses are also available. This co-design enables robust, metric-aware 3D reconstruction within a single unified model. Importantly, UniScale does not require training from scratch, and leverages world priors exhibited in pre-existing models without geometric encoding strategies, making it particularly suitable for resource-constrained robotic teams. We evaluate UniScale on multiple benchmarks, demonstrating strong generalization and consistent performance across diverse environments. We will release our implementation upon acceptance.
>
---
#### [new 036] Relational Appliances: A Robot in the Refrigerator for Home-Based Health Promotion
- **分类: cs.HC; cs.RO**

- **简介: 论文提出“关系性家电”概念，通过冰箱中的拟人机器人促进健康饮食。任务是探索家用设备在健康促进中的应用，解决如何通过互动设备影响用户饮食行为的问题。工作包括设计并测试一个具有社交互动功能的机器人原型。**

- **链接: [https://arxiv.org/pdf/2602.22542v1](https://arxiv.org/pdf/2602.22542v1)**

> **作者:** Timothy Bickmore; Mehdi Arjmand; Yunus Terzioglu
>
> **摘要:** Kitchen appliances are frequently used domestic artifacts situated at the point of everyday dietary decision making, making them a promising but underexplored site for health promotion. We explore the concept of relational appliances: everyday household devices designed as embodied social actors that engage users through ongoing, personalized interaction. We focus on the refrigerator, whose unique affordances, including a fixed, sensor-rich environment, private interaction space, and close coupling to food items, support contextualized, conversational engagement during snack choices. We present an initial exploration of this concept through a pilot study deploying an anthropomorphic robotic head inside a household refrigerator. In a home-lab apartment, participants repeatedly retrieved snacks during simulated TV "commercial breaks" while interacting with a human-sized robotic head. Participants were randomized to either a health-promotion condition, in which the robot made healthy snack recommendations, or a social-chat control condition. Outcomes included compliance with recommendations, nutritional quality of selected snacks, and psychosocial measures related to acceptance of the robot. Results suggest that participants found the robot persuasive, socially engaging, and increasingly natural over time, often describing it as helpful, aware, and companionable. Most participants reported greater awareness of their snack decisions and expressed interest in having such a robot in their own home. We discuss implications for designing relational appliances that leverage anthropomorphism, trust, and long-term human-technology relationships for home-based health promotion.
>
---
#### [new 037] WaterVideoQA: ASV-Centric Perception and Rule-Compliant Reasoning via Multi-Modal Agents
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出WaterVideoQA基准和NaviMind系统，解决自主水面航行器在动态水域环境中的认知与决策问题，提升其安全性和准确性。**

- **链接: [https://arxiv.org/pdf/2602.22923v1](https://arxiv.org/pdf/2602.22923v1)**

> **作者:** Runwei Guan; Shaofeng Liang; Ningwei Ouyang; Weichen Fei; Shanliang Yao; Wei Dai; Chenhao Ge; Penglei Sun; Xiaohui Zhu; Tao Huang; Ryan Wen Liu; Hui Xiong
>
> **备注:** 11 pages,8 figures
>
> **摘要:** While autonomous navigation has achieved remarkable success in passive perception (e.g., object detection and segmentation), it remains fundamentally constrained by a void in knowledge-driven, interactive environmental cognition. In the high-stakes domain of maritime navigation, the ability to bridge the gap between raw visual perception and complex cognitive reasoning is not merely an enhancement but a critical prerequisite for Autonomous Surface Vessels to execute safe and precise maneuvers. To this end, we present WaterVideoQA, the first large-scale, comprehensive Video Question Answering benchmark specifically engineered for all-waterway environments. This benchmark encompasses 3,029 video clips across six distinct waterway categories, integrating multifaceted variables such as volatile lighting and dynamic weather to rigorously stress-test ASV capabilities across a five-tier hierarchical cognitive framework. Furthermore, we introduce NaviMind, a pioneering multi-agent neuro-symbolic system designed for open-ended maritime reasoning. By synergizing Adaptive Semantic Routing, Situation-Aware Hierarchical Reasoning, and Autonomous Self-Reflective Verification, NaviMind transitions ASVs from superficial pattern matching to regulation-compliant, interpretable decision-making. Experimental results demonstrate that our framework significantly transcends existing baselines, establishing a new paradigm for intelligent, trustworthy interaction in dynamic maritime environments.
>
---
#### [new 038] CWM: Contrastive World Models for Action Feasibility Learning in Embodied Agent Pipelines
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于具身智能任务，解决动作可行性评分问题。通过对比世界模型（CWM）提升动作筛选效果，优于传统监督微调方法。**

- **链接: [https://arxiv.org/pdf/2602.22452v1](https://arxiv.org/pdf/2602.22452v1)**

> **作者:** Chayan Banerjee
>
> **摘要:** A reliable action feasibility scorer is a critical bottleneck in embodied agent pipelines: before any planning or reasoning occurs, the agent must identify which candidate actions are physically executable in the current state. Existing approaches use supervised fine-tuning (SFT) to train action scorers, but SFT treats each candidate independently and does not explicitly teach the model to discriminate between actions that are physically correct and those that are subtly wrong. We propose the Contrastive World Model (CWM), which fine-tunes a large language model (LLM) as an action scorer using an InfoNCE contrastive objective with hard-mined negative examples. The key idea is to push valid actions away from invalid ones in scoring space, with special emphasis on hard negatives: semantically similar but physically incompatible candidates. We evaluate CWM on the ScienceWorld benchmark through two studies. First, an intrinsic affordance evaluation on 605 hard-negative test pairs shows that CWM outperforms SFT by +6.76 percentage points on Precision@1 for minimal-edit negatives -- cases where a single word changes the physical outcome -- and achieves a higher AUC-ROC (0.929 vs. 0.906). Second, a live filter characterisation study measures how well CWM ranks gold-path actions against all valid environment actions during task execution. Under out-of-distribution stress conditions, CWM maintains a significantly better safety margin (-2.39) than SFT (-3.96), indicating that the gold action is ranked closer to the top. These results support the hypothesis that contrastive training induces representations that capture physical feasibility more faithfully than SFT alone.
>
---
#### [new 039] FLIGHT: Fibonacci Lattice-based Inference for Geometric Heading in real-Time
- **分类: cs.CV; cs.CG; cs.RO**

- **简介: 该论文属于视觉定位任务，旨在解决单目视频中相机运动估计的问题。提出基于斐波那契格网的霍夫变换方法，提升在噪声和异常值下的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2602.23115v1](https://arxiv.org/pdf/2602.23115v1)**

> **作者:** David Dirnfeld; Fabien Delattre; Pedro Miraldo; Erik Learned-Miller
>
> **摘要:** Estimating camera motion from monocular video is a fundamental problem in computer vision, central to tasks such as SLAM, visual odometry, and structure-from-motion. Existing methods that recover the camera's heading under known rotation, whether from an IMU or an optimization algorithm, tend to perform well in low-noise, low-outlier conditions, but often decrease in accuracy or become computationally expensive as noise and outlier levels increase. To address these limitations, we propose a novel generalization of the Hough transform on the unit sphere (S(2)) to estimate the camera's heading. First, the method extracts correspondences between two frames and generates a great circle of directions compatible with each pair of correspondences. Then, by discretizing the unit sphere using a Fibonacci lattice as bin centers, each great circle casts votes for a range of directions, ensuring that features unaffected by noise or dynamic objects vote consistently for the correct motion direction. Experimental results on three datasets demonstrate that the proposed method is on the Pareto frontier of accuracy versus efficiency. Additionally, experiments on SLAM show that the proposed method reduces RMSE by correcting the heading during camera pose initialization.
>
---
#### [new 040] Evaluating Zero-Shot and One-Shot Adaptation of Small Language Models in Leader-Follower Interaction
- **分类: cs.HC; cs.AI; cs.LG; cs.RO; eess.SY**

- **简介: 该论文属于角色分类任务，旨在解决资源受限机器人在人机交互中实时角色分配的问题。通过构建数据集并测试小语言模型的零样本和单样本适应策略，验证了微调方法的有效性。**

- **链接: [https://arxiv.org/pdf/2602.23312v1](https://arxiv.org/pdf/2602.23312v1)**

> **作者:** Rafael R. Baptista; André de Lima Salgado; Ricardo V. Godoy; Marcelo Becker; Thiago Boaventura; Gustavo J. G. Lahr
>
> **摘要:** Leader-follower interaction is an important paradigm in human-robot interaction (HRI). Yet, assigning roles in real time remains challenging for resource-constrained mobile and assistive robots. While large language models (LLMs) have shown promise for natural communication, their size and latency limit on-device deployment. Small language models (SLMs) offer a potential alternative, but their effectiveness for role classification in HRI has not been systematically evaluated. In this paper, we present a benchmark of SLMs for leader-follower communication, introducing a novel dataset derived from a published database and augmented with synthetic samples to capture interaction-specific dynamics. We investigate two adaptation strategies: prompt engineering and fine-tuning, studied under zero-shot and one-shot interaction modes, compared with an untrained baseline. Experiments with Qwen2.5-0.5B reveal that zero-shot fine-tuning achieves robust classification performance (86.66% accuracy) while maintaining low latency (22.2 ms per sample), significantly outperforming baseline and prompt-engineered approaches. However, results also indicate a performance degradation in one-shot modes, where increased context length challenges the model's architectural capacity. These findings demonstrate that fine-tuned SLMs provide an effective solution for direct role assignment, while highlighting critical trade-offs between dialogue complexity and classification reliability on the edge.
>
---
#### [new 041] Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出LaGS方法，用于4D全景占据跟踪任务，解决动态环境中同时获取精确几何与时间关联的问题。通过融合多视角信息到3D体素网格，实现高效场景理解。**

- **链接: [https://arxiv.org/pdf/2602.23172v1](https://arxiv.org/pdf/2602.23172v1)**

> **作者:** Maximilian Luz; Rohit Mohan; Thomas Nürnberg; Yakov Miron; Daniele Cattaneo; Abhinav Valada
>
> **摘要:** Capturing 4D spatiotemporal surroundings is crucial for the safe and reliable operation of robots in dynamic environments. However, most existing methods address only one side of the problem: they either provide coarse geometric tracking via bounding boxes, or detailed 3D structures like voxel-based occupancy that lack explicit temporal association. In this work, we present Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking (LaGS) that advances spatiotemporal scene understanding in a holistic direction. Our approach incorporates camera-based end-to-end tracking with mask-based multi-view panoptic occupancy prediction, and addresses the key challenge of efficiently aggregating multi-view information into 3D voxel grids via a novel latent Gaussian splatting approach. Specifically, we first fuse observations into 3D Gaussians that serve as a sparse point-centric latent representation of the 3D scene, and then splat the aggregated features onto a 3D voxel grid that is decoded by a mask-based segmentation head. We evaluate LaGS on the Occ3D nuScenes and Waymo datasets, achieving state-of-the-art performance for 4D panoptic occupancy tracking. We make our code available at https://lags.cs.uni-freiburg.de/.
>
---
#### [new 042] Risk-Aware World Model Predictive Control for Generalizable End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自主驾驶任务，解决E2E-AD在罕见场景下的泛化与安全性问题。提出RaWMPC框架，通过世界模型和风险评估实现安全决策。**

- **链接: [https://arxiv.org/pdf/2602.23259v1](https://arxiv.org/pdf/2602.23259v1)**

> **作者:** Jiangxin Sun; Feng Xue; Teng Long; Chang Liu; Jian-Fang Hu; Wei-Shi Zheng; Nicu Sebe
>
> **摘要:** With advances in imitation learning (IL) and large-scale driving datasets, end-to-end autonomous driving (E2E-AD) has made great progress recently. Currently, IL-based methods have become a mainstream paradigm: models rely on standard driving behaviors given by experts, and learn to minimize the discrepancy between their actions and expert actions. However, this objective of "only driving like the expert" suffers from limited generalization: when encountering rare or unseen long-tail scenarios outside the distribution of expert demonstrations, models tend to produce unsafe decisions in the absence of prior experience. This raises a fundamental question: Can an E2E-AD system make reliable decisions without any expert action supervision? Motivated by this, we propose a unified framework named Risk-aware World Model Predictive Control (RaWMPC) to address this generalization dilemma through robust control, without reliance on expert demonstrations. Practically, RaWMPC leverages a world model to predict the consequences of multiple candidate actions and selects low-risk actions through explicit risk evaluation. To endow the world model with the ability to predict the outcomes of risky driving behaviors, we design a risk-aware interaction strategy that systematically exposes the world model to hazardous behaviors, making catastrophic outcomes predictable and thus avoidable. Furthermore, to generate low-risk candidate actions at test time, we introduce a self-evaluation distillation method to distill riskavoidance capabilities from the well-trained world model into a generative action proposal network without any expert demonstration. Extensive experiments show that RaWMPC outperforms state-of-the-art methods in both in-distribution and out-of-distribution scenarios, while providing superior decision interpretability.
>
---
#### [new 043] Motion-aware Event Suppression for Event Cameras
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种运动感知事件抑制框架，用于过滤事件相机中由IMOs和自运动引起的噪声。任务是提升事件流处理的准确性与效率，通过实时分割与预测运动实现动态事件的提前抑制。**

- **链接: [https://arxiv.org/pdf/2602.23204v1](https://arxiv.org/pdf/2602.23204v1)**

> **作者:** Roberto Pellerito; Nico Messikommer; Giovanni Cioffi; Marco Cannici; Davide Scaramuzza
>
> **摘要:** In this work, we introduce the first framework for Motion-aware Event Suppression, which learns to filter events triggered by IMOs and ego-motion in real time. Our model jointly segments IMOs in the current event stream while predicting their future motion, enabling anticipatory suppression of dynamic events before they occur. Our lightweight architecture achieves 173 Hz inference on consumer-grade GPUs with less than 1 GB of memory usage, outperforming previous state-of-the-art methods on the challenging EVIMO benchmark by 67\% in segmentation accuracy while operating at a 53\% higher inference rate. Moreover, we demonstrate significant benefits for downstream applications: our method accelerates Vision Transformer inference by 83\% via token pruning and improves event-based visual odometry accuracy, reducing Absolute Trajectory Error (ATE) by 13\%.
>
---
## 更新

#### [replaced 001] DropVLA: An Action-Level Backdoor Attack on Vision--Language--Action Models
- **分类: cs.CR; cs.AI; cs.RO**

- **简介: 该论文属于安全领域，研究VLA模型的后门攻击问题。提出DropVLA攻击方法，在有限数据污染下实现对特定动作的隐蔽控制，保持正常任务性能。**

- **链接: [https://arxiv.org/pdf/2510.10932v2](https://arxiv.org/pdf/2510.10932v2)**

> **作者:** Zonghuan Xu; Xiang Zheng; Xingjun Ma; Yu-Gang Jiang
>
> **备注:** 8 pages, 6 tables, 3 figures. Under review
>
> **摘要:** Vision-Language-Action (VLA) models map multimodal perception and language instructions to executable robot actions, making them particularly vulnerable to behavioral backdoor manipulation: a hidden trigger introduced during training can induce unintended physical actions while nominal task performance remains intact. Prior work on VLA backdoors primarily studies untargeted attacks or task-level hijacking, leaving fine-grained control over individual actions largely unexplored. In this work, we present DropVLA, an action-level backdoor attack that forces a reusable action primitive (e.g., open_gripper) to execute at attacker-chosen decision points under a realistic pipeline-black-box setting with limited data-poisoning access, using a window-consistent relabeling scheme for chunked fine-tuning. On OpenVLA-7B evaluated with LIBERO, vision-only poisoning achieves 98.67%-99.83% attack success rate (ASR) with only 0.31% poisoned episodes while preserving 98.50%-99.17% clean-task retention, and successfully triggers the targeted action within 25 control steps at 500 Hz (0.05 s). Text-only triggers are unstable at low poisoning budgets, and combining text with vision provides no consistent ASR improvement over vision-only attacks. The backdoor remains robust to moderate trigger variations and transfers across evaluation suites (96.27%, 99.09%), whereas text-only largely fails (0.72%). We further validate physical-world feasibility on a 7-DoF Franka arm with pi0-fast, demonstrating non-trivial attack efficacy under camera-relative motion that induces image-plane trigger drift. These results reveal that VLA models can be covertly steered at the granularity of safety-critical actions with minimal poisoning and without observable degradation of nominal performance.
>
---
#### [replaced 002] STL-Based Motion Planning and Uncertainty-Aware Risk Analysis for Human-Robot Collaboration with a Multi-Rotor Aerial Vehicle
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，解决多旋翼无人机的安全路径规划与风险分析问题。通过STL编码任务目标，优化轨迹并评估不确定性风险，提升协作安全性与效率。**

- **链接: [https://arxiv.org/pdf/2509.10692v2](https://arxiv.org/pdf/2509.10692v2)**

> **作者:** Giuseppe Silano; Amr Afifi; Martin Saska; Antonio Franchi
>
> **备注:** 45 pages, 14 figures
>
> **摘要:** This paper presents a novel approach to motion planning and risk analysis for enhancing human-robot collaboration using a Multi-Rotor Aerial Vehicle (MRAV). The proposed method uses Signal Temporal Logic (STL) to encode key mission objectives, such as safety, timing, and human preferences, with a strong focus on ergonomics and comfort. An optimization framework generates dynamically feasible trajectories while considering the MRAV's physical constraints. Given the nonlinear and non-convex nature of the problem, smooth approximations and gradient-based techniques assist in handling the problem's computational complexity. Additionally, an uncertainty-aware risk analysis is incorporated to assess potential deviations from the mission specifications, providing insights into the likelihood of mission success under uncertain conditions. Further, an event-triggered replanning strategy is implemented to respond to unforeseen events and external disturbances. The approach is validated through MATLAB and Gazebo simulations, using an object handover task in a mock-up environment inspired by power line maintenance scenarios. The results highlight the method's effectiveness in achieving safe, efficient, and resilient human-robot collaboration.
>
---
#### [replaced 003] ST-GS: Vision-Based 3D Semantic Occupancy Prediction with Spatial-Temporal Gaussian Splatting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D语义占用预测任务，旨在解决多视角空间交互不足和多帧时间一致性差的问题。提出ST-GS框架，增强空间与时间建模能力。**

- **链接: [https://arxiv.org/pdf/2509.16552v2](https://arxiv.org/pdf/2509.16552v2)**

> **作者:** Xiaoyang Yan; Muleilan Pei; Shaojie Shen
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** 3D occupancy prediction is critical for comprehensive scene understanding in vision-centric autonomous driving. Recent advances have explored utilizing 3D semantic Gaussians to model occupancy while reducing computational overhead, but they remain constrained by insufficient multi-view spatial interaction and limited multi-frame temporal consistency. To overcome these issues, in this paper, we propose a novel Spatial-Temporal Gaussian Splatting (ST-GS) framework to enhance both spatial and temporal modeling in existing Gaussian-based pipelines. Specifically, we develop a guidance-informed spatial aggregation strategy within a dual-mode attention mechanism to strengthen spatial interaction in Gaussian representations. Furthermore, we introduce a geometry-aware temporal fusion scheme that effectively leverages historical context to improve temporal continuity in scene completion. Extensive experiments on the large-scale nuScenes occupancy prediction benchmark showcase that our proposed approach not only achieves state-of-the-art performance but also delivers markedly better temporal consistency compared to existing Gaussian-based methods.
>
---
#### [replaced 004] NMPCM: Nonlinear Model Predictive Control on Resource-Constrained Microcontrollers
- **分类: cs.RO**

- **简介: 该论文属于控制任务，解决在资源受限微控制器上实现非线性模型预测控制（NMPC）的问题，提出一种高效方法用于四旋翼无人机控制。**

- **链接: [https://arxiv.org/pdf/2507.21259v2](https://arxiv.org/pdf/2507.21259v2)**

> **作者:** Van Chung Nguyen; Pratik Walunj; Chuong Le; An Duy Nguyen; Hung Manh La
>
> **摘要:** Nonlinear Model Predictive Control (NMPC) is a powerful approach for controlling highly dynamic robotic systems, as it accounts for system dynamics and optimizes control inputs at each step. However, its high computational complexity makes implementation on resource-constrained microcontrollers impractical. While recent studies have demonstrated the feasibility of Model Predictive Control (MPC) with linearized dynamics on microcontrollers, applying full NMPC remains a significant challenge. This work presents an efficient solution for generating and deploying NMPC on microcontrollers (NMPCM) to control quadrotor UAVs. The proposed method optimizes computational efficiency while maintaining high control accuracy. Simulations in Gazebo/ROS and real-world experiments validate the effectiveness of the approach, demonstrating its capability to achieve high-frequency NMPC execution in real-time systems. The code is available at: https://github.com/aralab-unr/NMPCM.
>
---
#### [replaced 005] Multi-robot LiDAR SLAM: a practical case study in underground tunnel environments
- **分类: cs.RO**

- **简介: 该论文属于多机器人SLAM任务，旨在解决地下隧道环境中激光雷达定位与建图的挑战，重点改进回环检测的误报问题。**

- **链接: [https://arxiv.org/pdf/2507.21553v4](https://arxiv.org/pdf/2507.21553v4)**

> **作者:** Federica Di Lauro; Domenico G. Sorrenti; Miguel Angel Sotelo
>
> **备注:** 14 pages, 14 figures
>
> **摘要:** Multi-robot SLAM aims at localizing and building a map with multiple robots, interacting with each other. In the work described in this article, we analyze the pipeline of a decentralized LiDAR SLAM system to study the current limitations of the state of the art, and we discover a significant source of failures, i.e., that the loop detection is the source of too many false positives. We therefore develop and propose a new heuristic to overcome these limitations. The environment taken as reference in this work is the highly challenging case of underground tunnels. We also highlight potential new research areas still under-explored.
>
---
#### [replaced 006] SignBot: Learning Human-to-Humanoid Sign Language Interaction
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出SignBot，用于人机手语交互任务，解决聋哑人群沟通障碍问题。通过运动重定向、控制和生成交互模块，实现机器人准确执行手语。**

- **链接: [https://arxiv.org/pdf/2505.24266v4](https://arxiv.org/pdf/2505.24266v4)**

> **作者:** Guanren Qiao; Sixu Lin; Ronglai Zuo; Zhizheng Wu; Kui Jia; Guiliang Liu
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Sign language is a natural and visual form of language that uses movements and expressions to convey meaning, serving as a crucial means of communication for individuals who are deaf or hard-of-hearing (DHH). However, the number of people proficient in sign language remains limited, highlighting the need for technological advancements to bridge communication gaps and foster interactions with minorities. Based on recent advancements in embodied humanoid robots, we propose SignBot, a novel framework for human-robot sign language interaction. SignBot integrates a cerebellum-inspired motion control component and a cerebral-oriented module for comprehension and interaction. Specifically, SignBot consists of: 1) Motion Retargeting, which converts human sign language datasets into robot-compatible kinematics; 2) Motion Control, which leverages a learning-based paradigm to develop a robust humanoid control policy for tracking sign language gestures; and 3) Generative Interaction, which incorporates translator, responser, and generator of sign language, thereby enabling natural and effective communication between robots and humans. Simulation and real-world experimental results demonstrate that SignBot can effectively facilitate human-robot interaction and perform sign language motions with diverse robots and datasets. SignBot represents a significant advancement in automatic sign language interaction on embodied humanoid robot platforms, providing a promising solution to improve communication accessibility for the DHH community.
>
---
#### [replaced 007] From Prompts to Printable Models: Support-Effective 3D Generation via Offset Direct Preference Optimization
- **分类: cs.RO**

- **简介: 该论文属于3D生成任务，旨在解决模型生成的几何体难以打印的问题。通过优化减少支撑结构需求，提升打印效率与可持续性。**

- **链接: [https://arxiv.org/pdf/2511.16434v2](https://arxiv.org/pdf/2511.16434v2)**

> **作者:** Chenming Wu; Xiaofan Li; Chengkai Dai
>
> **备注:** Accepted by IEEE Robotics and Automation Letters 2026, preprint version by authors
>
> **摘要:** Current text-to-3D models prioritize visual fidelity but often neglect physical fabricability, resulting in geometries requiring excessive support structures. This paper introduces SEG (\textit{\underline{S}upport-\underline{E}ffective \underline{G}eneration}), a novel framework that integrates Direct Preference Optimization with an Offset (ODPO) into the 3D generation pipeline to directly optimize models for minimal support material usage. By incorporating support structure simulation into the training process, SEG encourages the generation of geometries that inherently require fewer supports, thus reducing material waste and production time. We demonstrate SEG's effectiveness through extensive experiments on two benchmark datasets, Thingi10k-Val and GPT-3DP-Val, showing that SEG significantly outperforms baseline models such as TRELLIS, DPO, and DRO in terms of support volume reduction and printability. Qualitative results further reveal that SEG maintains high fidelity to input prompts while minimizing the need for support structures. Our findings highlight the potential of SEG to transform 3D printing by directly optimizing models during the generative process, paving the way for more sustainable and efficient digital fabrication practices.
>
---
#### [replaced 008] Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D重建任务，解决高速飞行无人机的运动模糊和位姿漂移问题。通过融合事件流和模糊图像，优化NeRF并提升位姿估计精度。**

- **链接: [https://arxiv.org/pdf/2602.21101v2](https://arxiv.org/pdf/2602.21101v2)**

> **作者:** Rong Zou; Marco Cannici; Davide Scaramuzza
>
> **摘要:** Fast-flying aerial robots promise rapid inspection under limited battery constraints, with direct applications in infrastructure inspection, terrain exploration, and search and rescue. However, high speeds lead to severe motion blur in images and induce significant drift and noise in pose estimates, making dense 3D reconstruction with Neural Radiance Fields (NeRFs) particularly challenging due to their high sensitivity to such degradations. In this work, we present a unified framework that leverages asynchronous event streams alongside motion-blurred frames to reconstruct high-fidelity radiance fields from agile drone flights. By embedding event-image fusion into NeRF optimization and jointly refining event-based visual-inertial odometry priors using both event and frame modalities, our method recovers sharp radiance fields and accurate camera trajectories without ground-truth supervision. We validate our approach on both synthetic data and real-world sequences captured by a fast-flying drone. Despite highly dynamic drone flights, where RGB frames are severely degraded by motion blur and pose priors become unreliable, our method reconstructs high-fidelity radiance fields and preserves fine scene details, delivering a performance gain of over 50% on real-world data compared to state-of-the-art methods.
>
---
#### [replaced 009] Sparse Imagination for Efficient Visual World Model Planning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉世界模型规划任务，旨在解决机器人决策中的计算资源限制问题。通过稀疏想象方法，减少预测过程中的token数量，提升效率并保持控制精度。**

- **链接: [https://arxiv.org/pdf/2506.01392v2](https://arxiv.org/pdf/2506.01392v2)**

> **作者:** Junha Chun; Youngjoon Jeong; Taesup Kim
>
> **备注:** Accepted to ICLR 2026; Project Page: https://nikriz1.github.io/sparse_imagination/
>
> **摘要:** World model based planning has significantly improved decision-making in complex environments by enabling agents to simulate future states and make informed choices. This computational burden is particularly restrictive in robotics, where resources are severely constrained. To address this limitation, we propose a Sparse Imagination for Efficient Visual World Model Planning, which enhances computational efficiency by reducing the number of tokens processed during forward prediction. Our method leverages a sparsely trained vision-based world model based on transformers with randomized grouped attention strategy, allowing the model to flexibly adjust the number of tokens processed based on the computational resource. By enabling sparse imagination during latent rollout, our approach significantly accelerates planning while maintaining high control fidelity. Experimental results demonstrate that sparse imagination preserves task performance while dramatically improving inference efficiency. This general technique for visual planning is applicable from simple test-time trajectory optimization to complex real-world tasks with the latest VLAs, enabling the deployment of world models in real-time scenarios.
>
---
#### [replaced 010] Super LiDAR Intensity for Robotic Perception
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决低成本LiDAR数据稀疏的问题。通过构建密集LiDAR强度图像，提升环境感知能力。**

- **链接: [https://arxiv.org/pdf/2508.10398v2](https://arxiv.org/pdf/2508.10398v2)**

> **作者:** Wei Gao; Jie Zhang; Mingle Zhao; Zhiyuan Zhang; Shu Kong; Maani Ghaffari; Dezhen Song; Cheng-Zhong Xu; Hui Kong
>
> **备注:** IEEE Robotics and Automation Letters (RA-L), 2026 (https://ieeexplore.ieee.org/document/11395610). The dataset and code are available at: (https://github.com/IMRL/Super-LiDAR-Intensity)
>
> **摘要:** Conventionally, human intuition defines vision as a modality of passive optical sensing, relying on ambient light to perceive the environment. However, active optical sensing, which involves emitting and receiving signals, offers unique advantages by capturing both radiometric and geometric properties of the environment, independent of external illumination conditions. This work focuses on advancing active optical sensing using Light Detection and Ranging (LiDAR), which captures intensity data, enabling the estimation of surface reflectance that remains invariant under varying illumination. Such properties are crucial for robotic perception tasks, including detection, recognition, segmentation, and Simultaneous Localization and Mapping (SLAM). A key challenge with low-cost LiDARs lies in the sparsity of scan data, which limits their broader application. To address this limitation, this work introduces an innovative framework for generating dense LiDAR intensity images from sparse data, leveraging the unique attributes of non-repeating scanning LiDAR (NRS-LiDAR). We tackle critical challenges, including intensity calibration and the transition from static to dynamic scene domains, facilitating the reconstruction of dense intensity images in real-world settings. The key contributions of this work include a comprehensive dataset for LiDAR intensity image densification, a densification network tailored for NRS-LiDAR, and diverse applications such as loop closure and traffic lane detection using the generated dense intensity images. Experimental results validate the efficacy of the proposed approach, which successfully integrates computer vision techniques with LiDAR data processing, enhancing the applicability of low-cost LiDAR systems and establishing a novel paradigm for robotic vision via active optical sensing--LiDAR as a Camera.
>
---
#### [replaced 011] Hierarchical LLM-Based Multi-Agent Framework with Prompt Optimization for Multi-Robot Task Planning
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文属于多机器人任务规划任务，解决自然语言指令到可执行动作的转换问题。通过分层LLM框架与提示优化，提升规划准确性和成功率。**

- **链接: [https://arxiv.org/pdf/2602.21670v2](https://arxiv.org/pdf/2602.21670v2)**

> **作者:** Tomoya Kawabe; Rin Takano
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026. 8 pages, 2 figures
>
> **摘要:** Multi-robot task planning requires decomposing natural-language instructions into executable actions for heterogeneous robot teams. Conventional Planning Domain Definition Language (PDDL) planners provide rigorous guarantees but struggle to handle ambiguous or long-horizon missions, while large language models (LLMs) can interpret instructions and propose plans but may hallucinate or produce infeasible actions. We present a hierarchical multi-agent LLM-based planner with prompt optimization: an upper layer decomposes tasks and assigns them to lower-layer agents, which generate PDDL problems solved by a classical planner. When plans fail, the system applies TextGrad-inspired textual-gradient updates to optimize each agent's prompt and thereby improve planning accuracy. In addition, meta-prompts are learned and shared across agents within the same layer, enabling efficient prompt optimization in multi-agent settings. On the MAT-THOR benchmark, our planner achieves success rates of 0.95 on compound tasks, 0.84 on complex tasks, and 0.60 on vague tasks, improving over the previous state-of-the-art LaMMA-P by 2, 7, and 15 percentage points respectively. An ablation study shows that the hierarchical structure, prompt optimization, and meta-prompt sharing contribute roughly +59, +37, and +4 percentage points to the overall success rate.
>
---
#### [replaced 012] Time-Varying Formation Tracking Control of Wheeled Mobile Robots With Region Constraint: A Generalized Udwadia-Kalaba Framework
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人协同控制任务，解决具有区域约束的时变编队跟踪问题。通过广义Udwadia-Kalaba框架设计控制器，确保机器人安全运行。**

- **链接: [https://arxiv.org/pdf/2512.07137v2](https://arxiv.org/pdf/2512.07137v2)**

> **作者:** Yijie Kang; Yuqing Hao; Qingyun Wang; Guanrong Chen
>
> **备注:** 17 pages,9 figures
>
> **摘要:** In this article, the time-varying formation tracking control of wheeled mobile robots with region constraint is investigated from a generalized Udwadia-Kalaba framework. The communication network is modeled as a directed and weighted graph that has a spanning tree with the leader being the root. By reformulating the time-varying formation tracking control objective as an equality constrained equation and transforming the region constraint by a diffeomorphism, the time-varying formation tracking controller with the region constraint is designed under the generalized Udwadia-Kalaba framework. Compared with the existing works on time-varying formation tracking control, the region constraint is taken into account in this paper, which ensures the safety of the robots. Finally, the feasibility of the proposed control strategy is illustrated through some numerical simulations.
>
---
#### [replaced 013] A spherical amplitude-phase formulation for 3-D adaptive line-of-sight (ALOS) guidance with USGES stability guarantees
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文属于路径跟踪任务，解决3D自适应视线制导的稳定性问题。通过引入球面幅相模型，简化稳定性证明并扩展至复杂三维路径。**

- **链接: [https://arxiv.org/pdf/2505.08344v2](https://arxiv.org/pdf/2505.08344v2)**

> **作者:** Erlend M. Coates; Thor I. Fossen
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** A recently proposed 3-D adaptive line-of-sight (ALOS) path-following algorithm addressed coupled motion dynamics of marine craft, aircraft and uncrewed vehicles under environmental disturbances such as wind, waves and ocean currents. Stability analysis established uniform semi-global exponential stability (USGES) using a body-velocity-based amplitude-phase representation of the North-East-Down kinematic differential equations. However, the analysis is limited to straight-line paths, and restrictive assumptions are needed to ensure convergence of the vertical crab angle estimation error to zero. In this paper, we revisit the ALOS framework and introduce a novel spherical amplitude-phase design model that uses an alternative definition of the vertical crab angle. Our proposed formulation enables a significantly simplified stability proof, while retaining the USGES property for straight-line paths, removing restrictive assumptions on constant altitude/depth or zero horizontal crab angle, and remaining valid for general 3-D motion with nonzero roll, pitch and flight-path angles. We also show that the USGES result extends to a class of curved 3-D paths.
>
---
#### [replaced 014] A Pragmatic VLA Foundation Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出一种视觉-语言-动作基础模型LingBot-VLA，旨在提升机器人操作的泛化能力和成本效率，通过大量真实数据训练并在多个平台上验证其性能。**

- **链接: [https://arxiv.org/pdf/2601.18692v2](https://arxiv.org/pdf/2601.18692v2)**

> **作者:** Wei Wu; Fan Lu; Yunnan Wang; Shuai Yang; Shi Liu; Fangjing Wang; Qian Zhu; He Sun; Yong Wang; Shuailei Ma; Yiyu Ren; Kejia Zhang; Hui Yu; Jingmei Zhao; Shuai Zhou; Zhenqi Qiu; Houlong Xiong; Ziyu Wang; Zechen Wang; Ran Cheng; Yong-Lu Li; Yongtao Huang; Xing Zhu; Yujun Shen; Kecheng Zheng
>
> **备注:** Project Webpage: https://technology.robbyant.com/lingbot-vla/, Code: https://github.com/Robbyant/lingbot-vla/, GM-100: https://huggingface.co/datasets/robbyant/lingbot-GM-100
>
> **摘要:** Offering great potential in robotic manipulation, a capable Vision-Language-Action (VLA) foundation model is expected to faithfully generalize across tasks and platforms while ensuring cost efficiency (e.g., data and GPU hours required for adaptation). To this end, we develop LingBot-VLA with around 20,000 hours of real-world data from 9 popular dual-arm robot configurations. Through a systematic assessment on 3 robotic platforms, each completing 100 tasks with 130 post-training episodes per task, our model achieves clear superiority over competitors, showcasing its strong performance and broad generalizability. We have also built an efficient codebase, which delivers a throughput of 261 samples per second with an 8-GPU training setup, representing a 1.5~2.8$\times$ (depending on the relied VLM base model) speedup over existing VLA-oriented codebases. The above features ensure that our model is well-suited for real-world deployment. To advance the field of robot learning, we provide open access to the code, base model, and benchmark data, with a focus on enabling more challenging tasks and promoting sound evaluation standards.
>
---
#### [replaced 015] SCREP: Scene Coordinate Regression and Evidential Learning-based Perception-Aware Trajectory Generation
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决GPS拒收环境下的轨迹生成问题。通过结合场景坐标回归与证据学习，提出感知-aware的轨迹规划方法，提升定位精度与可靠性。**

- **链接: [https://arxiv.org/pdf/2507.07467v2](https://arxiv.org/pdf/2507.07467v2)**

> **作者:** Juyeop Han; Lukas Lao Beyer; Guilherme V. Cavalheiro; Sertac Karaman
>
> **摘要:** Autonomous flight in GPS-denied indoor spaces requires trajectories that keep visual-localization error tightly bounded across varied missions. Map-based visual localization methods such as feature matching require computationally intensive map reconstruction and have feature-storage scalability issues, especially for large environments. Scene coordinate regression (SCR) provides an efficient learning-based alternative that directly predicts3D coordinates for every pixel, enabling absolute pose estimation with significant potential for onboard roboticsapplications. We present a perception-aware trajectory planner that couples an evidential learning-based SCR poseestimator with a receding-horizon trajectory optimizer. The optimizer steers the onboard camera toward reliablescene coordinates with low uncertainty, while a fixed-lag smoother fuses the low-rate SCR pose estimates with high-rate IMU data to provide a high-quality, high-rate pose estimate. In simulation, our planner reduces translationand rotation RMSE by at least 4.9% and 30.8% relative to baselines, respectively. Hardware-in-the-loop experiments validate the feasibility of our proposed trajectory planner under close-to-real deployment conditions.
>
---
#### [replaced 016] PPT: Pretraining with Pseudo-Labeled Trajectories for Motion Forecasting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于运动预测任务，解决数据标注成本高、可扩展性差的问题。通过伪标签轨迹预训练，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2412.06491v3](https://arxiv.org/pdf/2412.06491v3)**

> **作者:** Yihong Xu; Yuan Yin; Éloi Zablocki; Tuan-Hung Vu; Alexandre Boulch; Matthieu Cord
>
> **备注:** 8 pages, 6 figures, accepted to ICRA 2026
>
> **摘要:** Accurately predicting how agents move in dynamic scenes is essential for safe autonomous driving. State-of-the-art motion forecasting models rely on datasets with manually annotated or post-processed trajectories. However, building these datasets is costly, generally manual, hard to scale, and lacks reproducibility. They also introduce domain gaps that limit generalization across environments. We introduce PPT (Pretraining with Pseudo-labeled Trajectories), a simple and scalable pretraining framework that uses unprocessed and diverse trajectories automatically generated from off-the-shelf 3D detectors and tracking. Unlike data annotation pipelines aiming for clean, single-label annotations, PPT is a pretraining framework embracing off-the-shelf trajectories as useful signals for learning robust representations. With optional finetuning on a small amount of labeled data, models pretrained with PPT achieve strong performance across standard benchmarks, particularly in low-data regimes, and in cross-domain, end-to-end, and multi-class settings. PPT is easy to implement and improves generalization in motion forecasting.
>
---
#### [replaced 017] SplatSDF: Boosting SDF-NeRF via Architecture-Level Fusion with Gaussian Splats
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文属于环境建模任务，旨在解决SDF-NeRF训练慢、收敛难的问题。通过融合3D高斯点云，提升收敛速度与精度。**

- **链接: [https://arxiv.org/pdf/2411.15468v2](https://arxiv.org/pdf/2411.15468v2)**

> **作者:** Runfa Blark Li; Keito Suzuki; Bang Du; Ki Myung Brian Lee; Nikolay Atanasov; Truong Nguyen
>
> **摘要:** Signed distance-radiance field (SDF-NeRF) is a promising environment representation that offers both photo-realistic rendering and geometric reasoning such as proximity queries for collision avoidance. However, the slow training speed and convergence of SDF-NeRF hinder their use in practical robotic systems. We propose SplatSDF, a novel SDF-NeRF architecture that accelerates convergence using 3D Gaussian splats (3DGS), which can be quickly pre-trained. Unlike prior approaches that introduce a consistency loss between separate 3DGS and SDF-NeRF models, SplatSDF directly fuses 3DGS at an architectural level by consuming it as an input to SDF-NeRF during training. This is achieved using a novel sparse 3DGS fusion strategy that injects neural embeddings of 3DGS into SDF-NeRF around the object surface, while also permitting inference without 3DGS for minimal operation. Experimental results show SplatSDF achieves 3X faster convergence to the same geometric accuracy than the best baseline, and outperforms state-of-the-art SDF-NeRF methods in terms of chamfer distance and peak signal to noise ratio, unlike consistency loss-based approaches that in fact provide limited gains. We also present computational techniques for accelerating gradient and Hessian steps by 3X. We expect these improvements will contribute to deploying SDF-NeRF on practical systems.
>
---
#### [replaced 018] Spatially anchored Tactile Awareness for Robust Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人灵巧操作任务，解决视觉-触觉学习方法在亚毫米精度任务中的不足。提出SaTA框架，通过空间锚定触觉感知提升几何推理能力，无需物体模型即可精准控制。**

- **链接: [https://arxiv.org/pdf/2510.14647v2](https://arxiv.org/pdf/2510.14647v2)**

> **作者:** Jialei Huang; Yang Ye; Yuanqing Gong; Xuezhou Zhu; Yang Gao; Kaifeng Zhang
>
> **备注:** 8 pages
>
> **摘要:** Dexterous manipulation requires precise geometric reasoning, yet existing visuo-tactile learning methods struggle with sub-millimeter precision tasks that are routine for traditional model-based approaches. We identify a key limitation: while tactile sensors provide rich contact information, current learning frameworks fail to effectively leverage both the perceptual richness of tactile signals and their spatial relationship with hand kinematics. We believe an ideal tactile representation should explicitly ground contact measurements in a stable reference frame while preserving detailed sensory information, enabling policies to not only detect contact occurrence but also precisely infer object geometry in the hand's coordinate system. We introduce SaTA (Spatially-anchored Tactile Awareness for dexterous manipulation), an end-to-end policy framework that explicitly anchors tactile features to the hand's kinematic frame through forward kinematics, enabling accurate geometric reasoning without requiring object models or explicit pose estimation. Our key insight is that spatially grounded tactile representations allow policies to not only detect contact occurrence but also precisely infer object geometry in the hand's coordinate system. We validate SaTA on challenging dexterous manipulation tasks, including bimanual USB-C mating in free space, a task demanding sub-millimeter alignment precision, as well as light bulb installation requiring precise thread engagement and rotational control, and card sliding that demands delicate force modulation and angular precision. These tasks represent significant challenges for learning-based methods due to their stringent precision requirements. Across multiple benchmarks, SaTA significantly outperforms strong visuo-tactile baselines, improving success rates by up to 30 percentage while reducing task completion times by 27 percentage.
>
---
#### [replaced 019] DreamWaQ++: Obstacle-Aware Quadrupedal Locomotion With Resilient Multi-Modal Reinforcement Learning
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于四足机器人运动控制任务，旨在解决复杂环境中运动鲁棒性不足的问题。通过融合本体感知与外部感知的多模态强化学习方法，提升机器人在崎岖地形和陡坡上的敏捷运动能力。**

- **链接: [https://arxiv.org/pdf/2409.19709v2](https://arxiv.org/pdf/2409.19709v2)**

> **作者:** I Made Aswin Nahrendra; Byeongho Yu; Minho Oh; Dongkyu Lee; Seunghyun Lee; Hyeonwoo Lee; Hyungtae Lim; Hyun Myung
>
> **备注:** IEEE Transactions on Robotics 2026. Project site is available at https://dreamwaqpp.github.io
>
> **摘要:** Quadrupedal robots hold promising potential for applications in navigating cluttered environments with resilience akin to their animal counterparts. However, their floating base configuration makes them vulnerable to real-world uncertainties, yielding substantial challenges in their locomotion control. Deep reinforcement learning has become one of the plausible alternatives for realizing a robust locomotion controller. However, the approaches that rely solely on proprioception sacrifice collision-free locomotion because they require front-feet contact to detect the presence of stairs to adapt the locomotion gait. Meanwhile, incorporating exteroception necessitates a precisely modeled map observed by exteroceptive sensors over a period of time. Therefore, this work proposes a novel method to fuse proprioception and exteroception featuring a resilient multi-modal reinforcement learning. The proposed method yields a controller that showcases agile locomotion performance on a quadrupedal robot over a myriad of real-world courses, including rough terrains, steep slopes, and high-rise stairs, while retaining its robustness against out-of-distribution situations.
>
---
