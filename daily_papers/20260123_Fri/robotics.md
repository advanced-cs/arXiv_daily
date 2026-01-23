# 机器人 cs.RO

- **最新发布 28 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] PUMA: Perception-driven Unified Foothold Prior for Mobility Augmented Quadruped Parkour
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于四足机器人公园跑任务，解决环境感知与足部选择问题。提出PUMA框架，整合视觉与足部先验，提升机器人实时适应能力。**

- **链接: [https://arxiv.org/pdf/2601.15995v1](https://arxiv.org/pdf/2601.15995v1)**

> **作者:** Liang Wang; Kanzhong Yao; Yang Liu; Weikai Qin; Jun Wu; Zhe Sun; Qiuguo Zhu
>
> **摘要:** Parkour tasks for quadrupeds have emerged as a promising benchmark for agile locomotion. While human athletes can effectively perceive environmental characteristics to select appropriate footholds for obstacle traversal, endowing legged robots with similar perceptual reasoning remains a significant challenge. Existing methods often rely on hierarchical controllers that follow pre-computed footholds, thereby constraining the robot's real-time adaptability and the exploratory potential of reinforcement learning. To overcome these challenges, we present PUMA, an end-to-end learning framework that integrates visual perception and foothold priors into a single-stage training process. This approach leverages terrain features to estimate egocentric polar foothold priors, composed of relative distance and heading, guiding the robot in active posture adaptation for parkour tasks. Extensive experiments conducted in simulation and real-world environments across various discrete complex terrains, demonstrate PUMA's exceptional agility and robustness in challenging scenarios.
>
---
#### [new 002] Collision-Free Humanoid Traversal in Cluttered Indoor Scenes
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决复杂室内场景中人形机器人避障问题。提出HumanoidPF表示方法，提升强化学习的避障能力，并通过混合场景生成实现泛化与真实世界部署。**

- **链接: [https://arxiv.org/pdf/2601.16035v1](https://arxiv.org/pdf/2601.16035v1)**

> **作者:** Han Xue; Sikai Liang; Zhikai Zhang; Zicheng Zeng; Yun Liu; Yunrui Lian; Jilong Wang; Qingtao Liu; Xuesong Shi; Li Yi
>
> **摘要:** We study the problem of collision-free humanoid traversal in cluttered indoor scenes, such as hurdling over objects scattered on the floor, crouching under low-hanging obstacles, or squeezing through narrow passages. To achieve this goal, the humanoid needs to map its perception of surrounding obstacles with diverse spatial layouts and geometries to the corresponding traversal skills. However, the lack of an effective representation that captures humanoid-obstacle relationships during collision avoidance makes directly learning such mappings difficult. We therefore propose Humanoid Potential Field (HumanoidPF), which encodes these relationships as collision-free motion directions, significantly facilitating RL-based traversal skill learning. We also find that HumanoidPF exhibits a surprisingly negligible sim-to-real gap as a perceptual representation. To further enable generalizable traversal skills through diverse and challenging cluttered indoor scenes, we further propose a hybrid scene generation method, incorporating crops of realistic 3D indoor scenes and procedurally synthesized obstacles. We successfully transfer our policy to the real world and develop a teleoperation system where users could command the humanoid to traverse in cluttered indoor scenes with just a single click. Extensive experiments are conducted in both simulation and the real world to validate the effectiveness of our method. Demos and code can be found in our website: https://axian12138.github.io/CAT/.
>
---
#### [new 003] Improve the autonomy of the SE2(3) group based Extended Kalman Filter for Integrated Navigation: Theoretical Analysis
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于导航模型优化任务，旨在解决高精度导航中SE2(3)框架的自主性问题，通过理论分析提出改进模型方法。**

- **链接: [https://arxiv.org/pdf/2601.16062v1](https://arxiv.org/pdf/2601.16062v1)**

> **作者:** Jiarui Cui; Maosong Wang; Wenqi Wu; Peiqi Li; Xianfei Pan
>
> **摘要:** One of core advantages of the SE2(3) Lie group framework for navigation modeling lies in the autonomy of error propagation. Current research on Lie group based extended Kalman filters has demonstrated that error propagation autonomy holds in low-precision applications, such as in micro electromechanical system (MEMS) based integrated navigation without considering earth rotation and inertial device biases. However, in high-precision navigation state estimation, maintaining autonomy is extremely difficult when considering with earth rotation and inertial device biases. This paper presents the theoretical analysis on the autonomy of SE2(3) group based high-precision navigation models under inertial, earth and world frame respectively. Through theoretical analysis, we find that the limitation of the traditional, trivial SE2(3) group navigation modeling method is that the presence of Coriolis force terms introduced by velocity in non-inertial frame. Therefore, a construction method for SE2(3) group navigation models is proposed, which brings the navigation models closer to full autonomy.
>
---
#### [new 004] IVRA: Improving Visual-Token Relations for Robot Action Policy with Training-Free Hint-Based Guidance
- **分类: cs.RO**

- **简介: 该论文提出IVRA方法，解决视觉-语言-动作模型中空间线索弱化的问题。通过引入内置视觉编码器的亲和提示，提升机器人操作策略的精确性。**

- **链接: [https://arxiv.org/pdf/2601.16207v1](https://arxiv.org/pdf/2601.16207v1)**

> **作者:** Jongwoo Park; Kanchana Ranasinghe; Jinhyeok Jang; Cristina Mata; Yoo Sung Jang; Michael S Ryoo
>
> **摘要:** Many Vision-Language-Action (VLA) models flatten image patches into a 1D token sequence, weakening the 2D spatial cues needed for precise manipulation. We introduce IVRA, a lightweight, training-free method that improves spatial understanding by exploiting affinity hints already available in the model's built-in vision encoder, without requiring any external encoder or retraining. IVRA selectively injects these affinity signals into a language-model layer in which instance-level features reside. This inference-time intervention realigns visual-token interactions and better preserves geometric structure while keeping all model parameters fixed. We demonstrate the generality of IVRA by applying it to diverse VLA architectures (LLaRA, OpenVLA, and FLOWER) across simulated benchmarks spanning both 2D and 3D manipulation (VIMA and LIBERO) and on various real-robot tasks. On 2D VIMA, IVRA improves average success by +4.2% over the baseline LLaRA in a low-data regime. On 3D LIBERO, it yields consistent gains over the OpenVLA and FLOWER baselines, including improvements when baseline accuracy is near saturation (96.3% to 97.1%). All code and models will be released publicly. Visualizations are available at: jongwoopark7978.github.io/IVRA
>
---
#### [new 005] AION: Aerial Indoor Object-Goal Navigation Using Dual-Policy Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于室内目标导航任务，解决空中平台在未知环境中自主导航至目标物体的问题。提出AION框架，采用双策略强化学习方法提升探索与导航效率及安全性。**

- **链接: [https://arxiv.org/pdf/2601.15614v1](https://arxiv.org/pdf/2601.15614v1)**

> **作者:** Zichen Yan; Yuchen Hou; Shenao Wang; Yichao Gao; Rui Huang; Lin Zhao
>
> **摘要:** Object-Goal Navigation (ObjectNav) requires an agent to autonomously explore an unknown environment and navigate toward target objects specified by a semantic label. While prior work has primarily studied zero-shot ObjectNav under 2D locomotion, extending it to aerial platforms with 3D locomotion capability remains underexplored. Aerial robots offer superior maneuverability and search efficiency, but they also introduce new challenges in spatial perception, dynamic control, and safety assurance. In this paper, we propose AION for vision-based aerial ObjectNav without relying on external localization or global maps. AION is an end-to-end dual-policy reinforcement learning (RL) framework that decouples exploration and goal-reaching behaviors into two specialized policies. We evaluate AION on the AI2-THOR benchmark and further assess its real-time performance in IsaacSim using high-fidelity drone models. Experimental results show that AION achieves superior performance across comprehensive evaluation metrics in exploration, navigation efficiency, and safety. The video can be found at https://youtu.be/TgsUm6bb7zg.
>
---
#### [new 006] Designing Persuasive Social Robots for Health Behavior Change: A Systematic Review of Behavior Change Strategies and Evaluation Methods
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于健康行为干预研究，旨在解决社会机器人设计与评估方法不足的问题。通过系统综述分析现有研究中的行为改变策略和评估方法，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2601.15309v1](https://arxiv.org/pdf/2601.15309v1)**

> **作者:** Jiaxin Xu; Chao Zhang; Raymond H. Cuijpers; Wijnand A. IJsselsteijn
>
> **备注:** Accepted to HRI 2026
>
> **摘要:** Social robots are increasingly applied as health behavior change interventions, yet actionable knowledge to guide their design and evaluation remains limited. This systematic review synthesizes (1) the behavior change strategies used in existing HRI studies employing social robots to promote health behavior change, and (2) the evaluation methods applied to assess behavior change outcomes. Relevant literature was identified through systematic database searches and hand searches. Analysis of 39 studies revealed four overarching categories of behavior change strategies: coaching strategies, counseling strategies, social influence strategies, and persuasion-enhancing strategies. These strategies highlight the unique affordances of social robots as behavior change interventions and offer valuable design heuristics. The review also identified key characteristics of current evaluation practices, including study designs, settings, durations, and outcome measures, on the basis of which we propose several directions for future HRI research.
>
---
#### [new 007] A Beacon Based Solution for Autonomous UUVs GNSS-Denied Stealthy Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于水下导航任务，解决GNSS拒止环境下的UUV隐蔽导航问题。通过部署信标网络，实现UUV的精确定位与路径规划。**

- **链接: [https://arxiv.org/pdf/2601.15802v1](https://arxiv.org/pdf/2601.15802v1)**

> **作者:** Alexandre Albore; Humbert Fiorino; Damien Pellier
>
> **备注:** 8 pages. IEEE TechDefense 2025
>
> **摘要:** Autonomous Unmanned Underwater Vehicles (UUVs) enable military and civilian covert operations in coastal areas without relying on support vessels or Global Navigation Satellite Systems (GNSS). Such operations are critical when surface access is not possible and stealthy navigation is required in restricted environments such as protected zones or dangerous areas under access ban. GNSS denied navigation is then essential to maintaining concealment as surfacing could expose UUVs to detection. To ensure a precise fleet positioning a constellation of beacons deployed by aerial or surface drones establish a synthetic landmark network that will guide the fleet of UUVs along an optimized path from the continental shelf to the goal on the shore. These beacons either submerged or floating emit acoustic signals for UUV localisation and navigation. A hierarchical planner generates an adaptive route for the drones executing primitive actions while continuously monitoring and replanning as needed to maintain trajectory accuracy.
>
---
#### [new 008] Neural Collision Detection for Multi-arm Laparoscopy Surgical Robots Through Learning-from-Simulation
- **分类: cs.RO**

- **简介: 该论文属于多机械臂腹腔镜手术机器人碰撞检测任务，旨在提升手术安全性与效率。通过分析建模、仿真与深度学习结合，实现精准的碰撞检测与距离估计。**

- **链接: [https://arxiv.org/pdf/2601.15459v1](https://arxiv.org/pdf/2601.15459v1)**

> **作者:** Sarvin Ghiasi; Majid Roshanfar; Jake Barralet; Liane S. Feldman; Amir Hooshiar
>
> **摘要:** This study presents an integrated framework for enhancing the safety and operational efficiency of robotic arms in laparoscopic surgery by addressing key challenges in collision detection and minimum distance estimation. By combining analytical modeling, real-time simulation, and machine learning, the framework offers a robust solution for ensuring safe robotic operations. An analytical model was developed to estimate the minimum distances between robotic arms based on their joint configurations, offering precise theoretical calculations that serve as both a validation tool and a benchmark. To complement this, a 3D simulation environment was created to model two 7-DOF Kinova robotic arms, generating a diverse dataset of configurations for collision detection and distance estimation. Using these insights, a deep neural network model was trained with joint actuators of robot arms and relative positions as inputs, achieving a mean absolute error of 282.2 mm and an R-squared value of 0.85. The close alignment between predicted and actual distances highlights the network's accuracy and its ability to generalize spatial relationships. This work demonstrates the effectiveness of combining analytical precision with machine learning algorithms to enhance the precision and reliability of robotic systems.
>
---
#### [new 009] A Universal Large Language Model -- Drone Command and Control Interface
- **分类: cs.RO**

- **简介: 该论文属于无人机控制任务，旨在解决LLM与无人机接口不通用的问题。通过MCP协议开发了通用、易用的无人机控制接口。**

- **链接: [https://arxiv.org/pdf/2601.15486v1](https://arxiv.org/pdf/2601.15486v1)**

> **作者:** Javier N. Ramos-Silva; Peter J. Burke
>
> **摘要:** The use of artificial intelligence (AI) for drone control can have a transformative impact on drone capabilities, especially when real world information can be integrated with drone sensing, command, and control, part of a growing field of physical AI. Large language models (LLMs) can be advantageous if trained at scale on general knowledge, but especially and in particular when the training data includes information such as detailed map geography topology of the entire planet, as well as the ability to access real time situational data such as weather. However, challenges remain in the interface between drones and LLMs in general, with each application requiring a tedious, labor intensive effort to connect the LLM trained knowledge to drone command and control. Here, we solve that problem, using an interface strategy that is LLM agnostic and drone agnostic, providing the first universal, versatile, comprehensive and easy to use drone control interface. We do this using the new model context protocol (MCP) standard, an open standard that provides a universal way for AI systems to access external data, tools, and services. We develop and deploy a cloud based Linux machine hosting an MCP server that supports the Mavlink protocol, an ubiquitous drone control language used almost universally by millions of drones including Ardupilot and PX4 framework.We demonstrate flight control of a real unmanned aerial vehicle. In further testing, we demonstrate extensive flight planning and control capability in a simulated drone, integrated with a Google Maps MCP server for up to date, real time navigation information. This demonstrates a universal approach to integration of LLMs with drone command and control, a paradigm that leverages and exploits virtually all of modern AI industry with drone technology in an easy to use interface that translates natural language to drone control.
>
---
#### [new 010] Preparation and Motion Study of Magnetically Driven Micro Soft Robot Mimicking the Cownose Ray
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于微机器人研究任务，旨在解决微型机器人在狭窄水下环境中的驱动与控制问题。设计并测试了一种磁驱动的仿鳐鱼软体机器人，探索磁场参数对其运动性能的影响。**

- **链接: [https://arxiv.org/pdf/2601.15349v1](https://arxiv.org/pdf/2601.15349v1)**

> **作者:** Jiaqing Chang; Song Gao; Chaowei Dong; zhaobang Li; Yang Liu
>
> **备注:** These experiments lay an important foundation for the study of tether-free control of underwater micro-soft robots. Furthermore, this research provides important references for the fields of biomimetic robots and magnetically controlled micro-soft robots
>
> **摘要:** In narrow, unstructured underwater environments such as environmental monitoring and minimally invasive medical procedures, micro soft robots exhibit unique advantages due to their flexible movement capabilities and small size. At the same time, applying bionic technology to the structural design of micro soft robots can significantly improve their swimming performance. However, limited by their miniaturization, these robots are difficult to power internally and usually adopt a wireless power supply method. This study designs and fabricates a magnetically responsive, cownose ray-inspired micro soft robot based on the swimming principle of the cownose ray. The robot is made of a certain proportion of NdFeB and PDMS. Then, a three-dimensional Helmholtz coil is used to generate an oscillating harmonic magnetic field to conduct swimming experiments on the robot, exploring the influence of magnetic field parameters on the robot's swimming performance. The experimental results show that the swimming speed is the fastest at B = 5 mT and f = 11 Hz, reaching 5.25 mm/s, which is about 0.5 body lengths per second. In addition, by adjusting the current direction and frequency of the coil, the robot can perform different swimming modes such as straight swimming, turning swimming, and directional swimming. By employing a stepwise adjustment method, the impact of response errors on the robot's trajectory can be effectively reduced. This study demonstrates a method for magnetically driven micro soft robots, laying a foundation for the application of wireless-driven robots in underwater narrow spaces.
>
---
#### [new 011] D-Optimality-Guided Reinforcement Learning for Efficient Open-Loop Calibration of a 3-DOF Ankle Rehabilitation Robot
- **分类: cs.RO**

- **简介: 该论文属于机器人标定任务，解决多自由度康复机器人精准对齐问题。通过D-优化强化学习方法，高效选择校准姿态，提升参数估计精度与一致性。**

- **链接: [https://arxiv.org/pdf/2601.15707v1](https://arxiv.org/pdf/2601.15707v1)**

> **作者:** Qifan Hu; Branko Celler; Weidong Mu; Steven W. Su
>
> **摘要:** Accurate alignment of multi-degree-of-freedom rehabilitation robots is essential for safe and effective patient training. This paper proposes a two-stage calibration framework for a self-designed three-degree-of-freedom (3-DOF) ankle rehabilitation robot. First, a Kronecker-product-based open-loop calibration method is developed to cast the input-output alignment into a linear parameter identification problem, which in turn defines the associated experimental design objective through the resulting information matrix. Building on this formulation, calibration posture selection is posed as a combinatorial design-of-experiments problem guided by a D-optimality criterion, i.e., selecting a small subset of postures that maximises the determinant of the information matrix. To enable practical selection under constraints, a Proximal Policy Optimization (PPO) agent is trained in simulation to choose 4 informative postures from a candidate set of 50. Across simulation and real-robot evaluations, the learned policy consistently yields substantially more informative posture combinations than random selection: the mean determinant of the information matrix achieved by PPO is reported to be more than two orders of magnitude higher with reduced variance. In addition, real-world results indicate that a parameter vector identified from only four D-optimality-guided postures provides stronger cross-episode prediction consistency than estimates obtained from a larger but unstructured set of 50 postures. The proposed framework therefore improves calibration efficiency while maintaining robust parameter estimation, offering practical guidance for high-precision alignment of multi-DOF rehabilitation robots.
>
---
#### [new 012] DextER: Language-driven Dexterous Grasp Generation with Embodied Reasoning
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DextER，解决语言驱动的灵巧抓取问题，通过具身推理生成抓取策略，提升抓取成功率和意图对齐。**

- **链接: [https://arxiv.org/pdf/2601.16046v1](https://arxiv.org/pdf/2601.16046v1)**

> **作者:** Junha Lee; Eunha Park; Minsu Cho
>
> **摘要:** Language-driven dexterous grasp generation requires the models to understand task semantics, 3D geometry, and complex hand-object interactions. While vision-language models have been applied to this problem, existing approaches directly map observations to grasp parameters without intermediate reasoning about physical interactions. We present DextER, Dexterous Grasp Generation with Embodied Reasoning, which introduces contact-based embodied reasoning for multi-finger manipulation. Our key insight is that predicting which hand links contact where on the object surface provides an embodiment-aware intermediate representation bridging task semantics with physical constraints. DextER autoregressively generates embodied contact tokens specifying which finger links contact where on the object surface, followed by grasp tokens encoding the hand configuration. On DexGYS, DextER achieves 67.14% success rate, outperforming state-of-the-art by 3.83%p with 96.4% improvement in intention alignment. We also demonstrate steerable generation through partial contact specification, providing fine-grained control over grasp synthesis.
>
---
#### [new 013] Improve the autonomy of the SE2(3) group based Extended Kalman Filter for Integrated Navigation: Application
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于导航系统任务，旨在提升SE2(3)框架下导航模型的自主性。通过实验与仿真验证改进后的高精度导航模型性能。**

- **链接: [https://arxiv.org/pdf/2601.16078v1](https://arxiv.org/pdf/2601.16078v1)**

> **作者:** Jiarui Cui; Maosong Wang; Wenqi Wu; Peiqi Li; Xianfei Pan
>
> **摘要:** One of the core advantages of SE2(3) Lie group framework for navigation modeling lies in the autonomy of error propagation. In the previous paper, the theoretical analysis of autonomy property of navigation model in inertial, earth and world frames was given. A construction method for SE2(3) group navigation model is proposed to improve the non-inertial navigation model toward full autonomy. This paper serves as a counterpart to previous paper and conducts the real-world strapdown inertial navigation system (SINS)/odometer(ODO) experiments as well as Monte-Carlo simulations to demonstrate the performance of improved SE2(3) group based high-precision navigation models.
>
---
#### [new 014] Learning a Unified Latent Space for Cross-Embodiment Robot Control
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决跨形态人形机器人运动迁移问题。通过学习统一潜在空间，实现不同机器人间的运动泛化与直接部署。**

- **链接: [https://arxiv.org/pdf/2601.15419v1](https://arxiv.org/pdf/2601.15419v1)**

> **作者:** Yashuai Yan; Dongheui Lee
>
> **摘要:** We present a scalable framework for cross-embodiment humanoid robot control by learning a shared latent representation that unifies motion across humans and diverse humanoid platforms, including single-arm, dual-arm, and legged humanoid robots. Our method proceeds in two stages: first, we construct a decoupled latent space that captures localized motion patterns across different body parts using contrastive learning, enabling accurate and flexible motion retargeting even across robots with diverse morphologies. To enhance alignment between embodiments, we introduce tailored similarity metrics that combine joint rotation and end-effector positioning for critical segments, such as arms. Then, we train a goal-conditioned control policy directly within this latent space using only human data. Leveraging a conditional variational autoencoder, our policy learns to predict latent space displacements guided by intended goal directions. We show that the trained policy can be directly deployed on multiple robots without any adaptation. Furthermore, our method supports the efficient addition of new robots to the latent space by learning only a lightweight, robot-specific embedding layer. The learned latent policies can also be directly applied to the new robots. Experimental results demonstrate that our approach enables robust, scalable, and embodiment-agnostic robot control across a wide range of humanoid platforms.
>
---
#### [new 015] Airflow Source Seeking on Small Quadrotors Using a Single Flow Sensor
- **分类: cs.RO**

- **简介: 该论文属于环境感知任务，旨在解决小四旋翼无人机在受限空间中追踪污染源的问题。通过使用定制流传感器，实现基于气流方向的源寻踪算法。**

- **链接: [https://arxiv.org/pdf/2601.15607v1](https://arxiv.org/pdf/2601.15607v1)**

> **作者:** Lenworth Thomas; Tjaden Bridges; Sarah Bergbreiter
>
> **摘要:** As environmental disasters happen more frequently and severely, seeking the source of pollutants or harmful particulates using plume tracking becomes even more important. Plume tracking on small quadrotors would allow these systems to operate around humans and fly in more confined spaces, but can be challenging due to poor sensitivity and long response times from gas sensors that fit on small quadrotors. In this work, we present an approach to complement chemical plume tracking with airflow source-seeking behavior using a custom flow sensor that can sense both airflow magnitude and direction on small quadrotors < 100 g. We use this sensor to implement a modified version of the `Cast and Surge' algorithm that takes advantage of flow direction sensing to find and navigate towards flow sources. A series of characterization experiments verified that the system can detect airflow while in flight and reorient the quadrotor toward the airflow. Several trials with random starting locations and orientations were used to show that our source-seeking algorithm can reliably find a flow source. This work aims to provide a foundation for future platforms that can use flow sensors in concert with other sensors to enable richer plume tracking data collection and source-seeking.
>
---
#### [new 016] DualShield: Safe Model Predictive Diffusion via Reachability Analysis for Interactive Autonomous Driving
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属于自动驾驶任务，解决扩散模型在动态环境中的安全性问题。通过DualShield框架，结合HJ可达性分析，提升规划安全性和效率。**

- **链接: [https://arxiv.org/pdf/2601.15729v1](https://arxiv.org/pdf/2601.15729v1)**

> **作者:** Rui Yang; Lei Zheng; Ruoyu Yao; Jun Ma
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Diffusion models have emerged as a powerful approach for multimodal motion planning in autonomous driving. However, their practical deployment is typically hindered by the inherent difficulty in enforcing vehicle dynamics and a critical reliance on accurate predictions of other agents, making them prone to safety issues under uncertain interactions. To address these limitations, we introduce DualShield, a planning and control framework that leverages Hamilton-Jacobi (HJ) reachability value functions in a dual capacity. First, the value functions act as proactive guidance, steering the diffusion denoising process towards safe and dynamically feasible regions. Second, they form a reactive safety shield using control barrier-value functions (CBVFs) to modify the executed actions and ensure safety. This dual mechanism preserves the rich exploration capabilities of diffusion models while providing principled safety assurance under uncertain and even adversarial interactions. Simulations in challenging unprotected U-turn scenarios demonstrate that DualShield significantly improves both safety and task efficiency compared to leading methods from different planning paradigms under uncertainty.
>
---
#### [new 017] A Mobile Magnetic Manipulation Platform for Gastrointestinal Navigation with Deep Reinforcement Learning Control
- **分类: cs.RO**

- **简介: 该论文属于磁控机器人导航任务，旨在解决传统磁控系统控制难、建模复杂的问题。通过深度强化学习和移动磁阵平台，实现精准、快速的胃肠道磁性胶囊控制。**

- **链接: [https://arxiv.org/pdf/2601.15545v1](https://arxiv.org/pdf/2601.15545v1)**

> **作者:** Zhifan Yan; Chang Liu; Yiyang Jiang; Wenxuan Zheng; Xinhao Chen; Axel Krieger
>
> **摘要:** Targeted drug delivery in the gastrointestinal (GI) tract using magnetic robots offers a promising alternative to systemic treatments. However, controlling these robots is a major challenge. Stationary magnetic systems have a limited workspace, while mobile systems (e.g., coils on a robotic arm) suffer from a "model-calibration bottleneck", requiring complex, pre-calibrated physical models that are time-consuming to create and computationally expensive. This paper presents a compact, low-cost mobile magnetic manipulation platform that overcomes this limitation using Deep Reinforcement Learning (DRL). Our system features a compact four-electromagnet array mounted on a UR5 collaborative robot. A Soft Actor-Critic (SAC)-based control strategy is trained through a sim-to-real pipeline, enabling effective policy deployment within 15 minutes and significantly reducing setup time. We validated the platform by controlling a 7-mm magnetic capsule along 2D trajectories. Our DRL-based controller achieved a root-mean-square error (RMSE) of 1.18~mm for a square path and 1.50~mm for a circular path. We also demonstrated successful tracking over a clinically relevant, 30 cm * 20 cm workspace. This work demonstrates a rapidly deployable, model-free control framework capable of precise magnetic manipulation in a large workspace,validated using a 2D GI phantom.
>
---
#### [new 018] TeNet: Text-to-Network for Compact Policy Synthesis
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出TeNet，用于从自然语言生成紧凑机器人策略，解决资源受限的实时控制问题。通过文本条件超网络实现高效政策合成。**

- **链接: [https://arxiv.org/pdf/2601.15912v1](https://arxiv.org/pdf/2601.15912v1)**

> **作者:** Ariyan Bighashdel; Kevin Sebastian Luck
>
> **摘要:** Robots that follow natural-language instructions often either plan at a high level using hand-designed interfaces or rely on large end-to-end models that are difficult to deploy for real-time control. We propose TeNet (Text-to-Network), a framework for instantiating compact, task-specific robot policies directly from natural language descriptions. TeNet conditions a hypernetwork on text embeddings produced by a pretrained large language model (LLM) to generate a fully executable policy, which then operates solely on low-dimensional state inputs at high control frequencies. By using the language only once at the policy instantiation time, TeNet inherits the general knowledge and paraphrasing robustness of pretrained LLMs while remaining lightweight and efficient at execution time. To improve generalization, we optionally ground language in behavior during training by aligning text embeddings with demonstrated actions, while requiring no demonstrations at inference time. Experiments on MuJoCo and Meta-World benchmarks show that TeNet produces policies that are orders of magnitude smaller than sequence-based baselines, while achieving strong performance in both multi-task and meta-learning settings and supporting high-frequency control. These results show that text-conditioned hypernetworks offer a practical way to build compact, language-driven controllers for ressource-constrained robot control tasks with real-time requirements.
>
---
#### [new 019] Glove2UAV: A Wearable IMU-Based Glove for Intuitive Control of UAV
- **分类: cs.RO**

- **简介: 论文提出Glove2UAV，一种基于IMU的手套设备，用于直观控制无人机。任务是实现安全、实时的无人机操控，解决手势与飞行指令映射问题。工作包括设计轻量处理流程和集成触觉反馈。**

- **链接: [https://arxiv.org/pdf/2601.15775v1](https://arxiv.org/pdf/2601.15775v1)**

> **作者:** Amir Habel; Ivan Snegirev; Elizaveta Semenyakina; Miguel Altamirano Cabrera; Jeffrin Sam; Fawad Mehboob; Roohan Ahmed Khan; Muhammad Ahsan Mustafa; Dzmitry Tsetserukou
>
> **备注:** This paper has been accepted for publication at LBR of HRI 2026 conference
>
> **摘要:** This paper presents Glove2UAV, a wearable IMU-glove interface for intuitive UAV control through hand and finger gestures, augmented with vibrotactile warnings for exceeding predefined speed thresholds. To promote safer and more predictable interaction in dynamic flight, Glove2UAV is designed as a lightweight and easily deployable wearable interface intended for real-time operation. Glove2UAV streams inertial measurements in real time and estimates palm and finger orientations using a compact processing pipeline that combines median-based outlier suppression with Madgwick-based orientation estimation. The resulting motion estimations are mapped to a small set of control primitives for directional flight (forward/backward and lateral motion) and, when supported by the platform, to object-interaction commands. Vibrotactile feedback is triggered when flight speed exceeds predefined threshold values, providing an additional alert channel during operation. We validate real-time feasibility by synchronizing glove signals with UAV telemetry in both simulation and real-world flights. The results show fast gesture-based command execution, stable coupling between gesture dynamics and platform motion, correct operation of the core command set in our trials, and timely delivery of vibratile warning cues.
>
---
#### [new 020] Accurate Calibration and Robust LiDAR-Inertial Odometry for Spinning Actuated LiDAR Systems
- **分类: cs.RO**

- **简介: 该论文属于LiDAR定位任务，解决校准与定位鲁棒性问题，提出无目标校准方法和自适应LiDAR-惯性里程计。**

- **链接: [https://arxiv.org/pdf/2601.15946v1](https://arxiv.org/pdf/2601.15946v1)**

> **作者:** Zijie Chen; Xiaowei Liu; Yong Xu; Shenghai Yuan; Jianping Li; Lihua Xie
>
> **备注:** This article has been accepted for publication in IEEE Robotics and Automation Letters (RA-L). Personal use is permitted. All other uses require IEEE permission
>
> **摘要:** Accurate calibration and robust localization are fundamental for downstream tasks in spinning actuated LiDAR applications. Existing methods, however, require parameterizing extrinsic parameters based on different mounting configurations, limiting their generalizability. Additionally, spinning actuated LiDAR inevitably scans featureless regions, which complicates the balance between scanning coverage and localization robustness. To address these challenges, this letter presents a targetless LiDAR-motor calibration (LM-Calibr) on the basis of the Denavit-Hartenberg convention and an environmental adaptive LiDAR-inertial odometry (EVA-LIO). LM-Calibr supports calibration of LiDAR-motor systems with various mounting configurations. Extensive experiments demonstrate its accuracy and convergence across different scenarios, mounting angles, and initial values. Additionally, EVA-LIO adaptively selects downsample rates and map resolutions according to spatial scale. This adaptivity enables the actuator to operate at maximum speed, thereby enhancing scanning completeness while ensuring robust localization, even when LiDAR briefly scans featureless areas. The source code and hardware design are available on GitHub: \textcolor{blue}{\href{https://github.com/zijiechenrobotics/lm_calibr}{github.com/zijiechenrobotics/lm\_calibr}}. The video is available at \textcolor{blue}{\href{https://youtu.be/cZyyrkmeoSk}{youtu.be/cZyyrkmeoSk}}
>
---
#### [new 021] Point Bridge: 3D Representations for Cross Domain Policy Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决仿真到现实的策略迁移问题。通过点云表示和VLM提取特征，实现无需视觉对齐的跨域策略学习。**

- **链接: [https://arxiv.org/pdf/2601.16212v1](https://arxiv.org/pdf/2601.16212v1)**

> **作者:** Siddhant Haldar; Lars Johannsmeier; Lerrel Pinto; Abhishek Gupta; Dieter Fox; Yashraj Narang; Ajay Mandlekar
>
> **摘要:** Robot foundation models are beginning to deliver on the promise of generalist robotic agents, yet progress remains constrained by the scarcity of large-scale real-world manipulation datasets. Simulation and synthetic data generation offer a scalable alternative, but their usefulness is limited by the visual domain gap between simulation and reality. In this work, we present Point Bridge, a framework that leverages unified, domain-agnostic point-based representations to unlock synthetic datasets for zero-shot sim-to-real policy transfer, without explicit visual or object-level alignment. Point Bridge combines automated point-based representation extraction via Vision-Language Models (VLMs), transformer-based policy learning, and efficient inference-time pipelines to train capable real-world manipulation agents using only synthetic data. With additional co-training on small sets of real demonstrations, Point Bridge further improves performance, substantially outperforming prior vision-based sim-and-real co-training methods. It achieves up to 44% gains in zero-shot sim-to-real transfer and up to 66% with limited real data across both single-task and multitask settings. Videos of the robot are best viewed at: https://pointbridge3d.github.io/
>
---
#### [new 022] CompliantVLA-adaptor: VLM-Guided Variable Impedance Action for Safe Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决接触密集场景下的安全控制问题。通过引入视觉语言模型指导的可变阻抗控制，提升机器人的安全性和操作效果。**

- **链接: [https://arxiv.org/pdf/2601.15541v1](https://arxiv.org/pdf/2601.15541v1)**

> **作者:** Heng Zhang; Wei-Hsing Huang; Qiyi Tong; Gokhan Solak; Puze Liu; Sheng Liu; Jan Peters; Arash Ajoudani
>
> **备注:** under review
>
> **摘要:** We propose a CompliantVLA-adaptor that augments the state-of-the-art Vision-Language-Action (VLA) models with vision-language model (VLM)-informed context-aware variable impedance control (VIC) to improve the safety and effectiveness of contact-rich robotic manipulation tasks. Existing VLA systems (e.g., RDT, Pi0, OpenVLA-oft) typically output position, but lack force-aware adaptation, leading to unsafe or failed interactions in physical tasks involving contact, compliance, or uncertainty. In the proposed CompliantVLA-adaptor, a VLM interprets task context from images and natural language to adapt the stiffness and damping parameters of a VIC controller. These parameters are further regulated using real-time force/torque feedback to ensure interaction forces remain within safe thresholds. We demonstrate that our method outperforms the VLA baselines on a suite of complex contact-rich tasks, both in simulation and on real hardware, with improved success rates and reduced force violations. The overall success rate across all tasks increases from 9.86\% to 17.29\%, presenting a promising path towards safe contact-rich manipulation using VLAs. We release our code, prompts, and force-torque-impedance-scenario context datasets at https://sites.google.com/view/compliantvla.
>
---
#### [new 023] Efficiently Learning Robust Torque-based Locomotion Through Reinforcement with Model-Based Supervision
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在解决 bipedal locomotion 在真实环境中的鲁棒性问题。通过结合模型控制与强化学习，提升行走适应性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.16109v1](https://arxiv.org/pdf/2601.16109v1)**

> **作者:** Yashuai Yan; Tobias Egle; Christian Ott; Dongheui Lee
>
> **摘要:** We propose a control framework that integrates model-based bipedal locomotion with residual reinforcement learning (RL) to achieve robust and adaptive walking in the presence of real-world uncertainties. Our approach leverages a model-based controller, comprising a Divergent Component of Motion (DCM) trajectory planner and a whole-body controller, as a reliable base policy. To address the uncertainties of inaccurate dynamics modeling and sensor noise, we introduce a residual policy trained through RL with domain randomization. Crucially, we employ a model-based oracle policy, which has privileged access to ground-truth dynamics during training, to supervise the residual policy via a novel supervised loss. This supervision enables the policy to efficiently learn corrective behaviors that compensate for unmodeled effects without extensive reward shaping. Our method demonstrates improved robustness and generalization across a range of randomized conditions, offering a scalable solution for sim-to-real transfer in bipedal locomotion.
>
---
#### [new 024] Social Robotics for Disabled Students: An Empirical Investigation of Embodiment, Roles and Interaction
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决残疾学生在高等教育中的支持障碍。通过比较不同机器人角色和形态，评估其对信息获取和社交体验的影响。**

- **链接: [https://arxiv.org/pdf/2601.15293v1](https://arxiv.org/pdf/2601.15293v1)**

> **作者:** Alva Markelius; Fethiye Irmak Doğan; Julie Bailey; Guy Laban; Jenny L. Gibson; Hatice Gunes
>
> **备注:** Preprint. Accepted at ACM IEEE International Conference on Human Robot Interaction 2026, Edinburgh, Scotland, UK
>
> **摘要:** Institutional and social barriers in higher education often prevent students with disabilities from effectively accessing support, including lengthy procedures, insufficient information, and high social-emotional demands. This study empirically explores how disabled students perceive robot-based support, comparing two interaction roles, one information based (signposting) and one disclosure based (sounding board), and two embodiment types (physical robot/disembodied voice agent). Participants assessed these systems across five dimensions: perceived understanding, social energy demands, information access/clarity, task difficulty, and data privacy concerns. The main findings of the study reveal that the physical robot was perceived as more understanding than the voice-only agent, with embodiment significantly shaping perceptions of sociability, animacy, and privacy. We also analyse differences between disability types. These results provide critical insights into the potential of social robots to mitigate accessibility barriers in higher education, while highlighting ethical, social and technical challenges.
>
---
#### [new 025] Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出Cosmos Policy，用于将预训练视频模型直接适配为机器人策略，解决视觉-动作控制与规划问题，通过单阶段微调实现高效动作生成和未来状态预测。**

- **链接: [https://arxiv.org/pdf/2601.16163v1](https://arxiv.org/pdf/2601.16163v1)**

> **作者:** Moo Jin Kim; Yihuai Gao; Tsung-Yi Lin; Yen-Chen Lin; Yunhao Ge; Grace Lam; Percy Liang; Shuran Song; Ming-Yu Liu; Chelsea Finn; Jinwei Gu
>
> **摘要:** Recent video generation models demonstrate remarkable ability to capture complex physical interactions and scene evolution over time. To leverage their spatiotemporal priors, robotics works have adapted video models for policy learning but introduce complexity by requiring multiple stages of post-training and new architectural components for action generation. In this work, we introduce Cosmos Policy, a simple approach for adapting a large pretrained video model (Cosmos-Predict2) into an effective robot policy through a single stage of post-training on the robot demonstration data collected on the target platform, with no architectural modifications. Cosmos Policy learns to directly generate robot actions encoded as latent frames within the video model's latent diffusion process, harnessing the model's pretrained priors and core learning algorithm to capture complex action distributions. Additionally, Cosmos Policy generates future state images and values (expected cumulative rewards), which are similarly encoded as latent frames, enabling test-time planning of action trajectories with higher likelihood of success. In our evaluations, Cosmos Policy achieves state-of-the-art performance on the LIBERO and RoboCasa simulation benchmarks (98.5% and 67.1% average success rates, respectively) and the highest average score in challenging real-world bimanual manipulation tasks, outperforming strong diffusion policies trained from scratch, video model-based policies, and state-of-the-art vision-language-action models fine-tuned on the same robot demonstrations. Furthermore, given policy rollout data, Cosmos Policy can learn from experience to refine its world model and value function and leverage model-based planning to achieve even higher success rates in challenging tasks. We release code, models, and training data at https://research.nvidia.com/labs/dir/cosmos-policy/
>
---
#### [new 026] Keyframe-Based Feed-Forward Visual Odometry
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉里程计任务，解决传统方法与基础模型结合中的效率与性能问题。提出基于关键帧的前馈视觉里程计，利用强化学习自适应选择关键帧，提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2601.16020v1](https://arxiv.org/pdf/2601.16020v1)**

> **作者:** Weichen Dai; Wenhan Su; Da Kong; Yuhang Ming; Wanzeng Kong
>
> **摘要:** The emergence of visual foundation models has revolutionized visual odometry~(VO) and SLAM, enabling pose estimation and dense reconstruction within a single feed-forward network. However, unlike traditional pipelines that leverage keyframe methods to enhance efficiency and accuracy, current foundation model based methods, such as VGGT-Long, typically process raw image sequences indiscriminately. This leads to computational redundancy and degraded performance caused by low inter-frame parallax, which provides limited contextual stereo information. Integrating traditional geometric heuristics into these methods is non-trivial, as their performance depends on high-dimensional latent representations rather than explicit geometric metrics. To bridge this gap, we propose a novel keyframe-based feed-forward VO. Instead of relying on hand-crafted rules, our approach employs reinforcement learning to derive an adaptive keyframe policy in a data-driven manner, aligning selection with the intrinsic characteristics of the underlying foundation model. We train our agent on TartanAir dataset and conduct extensive evaluations across several real-world datasets. Experimental results demonstrate that the proposed method achieves consistent and substantial improvements over state-of-the-art feed-forward VO methods.
>
---
#### [new 027] DTP: A Simple yet Effective Distracting Token Pruning Framework for Vision-Language Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言动作模型任务，解决模型过度关注无关图像区域的问题。提出DTP框架动态去除干扰令牌，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2601.16065v1](https://arxiv.org/pdf/2601.16065v1)**

> **作者:** Chenyang Li; Jieyuan Liu; Bin Li; Bo Gao; Yilin Yuan; Yangfan He; Yuchen Li; Jingqun Tang
>
> **摘要:** Vision-Language Action (VLA) models have shown remarkable progress in robotic manipulation by leveraging the powerful perception abilities of Vision-Language Models (VLMs) to understand environments and directly output actions. However, by default, VLA models may overly attend to image tokens in the task-irrelevant region, which we describe as 'distracting tokens'. This behavior can disturb the model from the generation of the desired action tokens in each step, affecting the success rate of tasks. In this paper, we introduce a simple yet effective plug-and-play Distracting Token Pruning (DTP) framework, which dynamically detects and prunes these distracting image tokens. By correcting the model's visual attention patterns, we aim to improve the task success rate, as well as exploring the performance upper boundaries of the model without altering its original architecture or adding additional inputs. Experiments on the SIMPLER Benchmark (Li et al., 2024) show that our method consistently achieving relative improvements in task success rates across different types of novel VLA models, demonstrating generalizability to transformer-based VLAs. Further analysis reveals a negative correlation between the task success rate and the amount of attentions in the task-irrelevant region for all models tested, highlighting a common phenomenon of VLA models that could guide future research. We also publish our code at: https://anonymous.4open.science/r/CBD3.
>
---
#### [new 028] MapViT: A Two-Stage ViT-Based Framework for Real-Time Radio Quality Map Prediction in Dynamic Environments
- **分类: cs.NI; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出MapViT，用于实时预测动态环境中的无线电质量图。解决机器人在复杂环境中感知与导航的问题，通过两阶段ViT框架提升预测精度与效率。**

- **链接: [https://arxiv.org/pdf/2601.15578v1](https://arxiv.org/pdf/2601.15578v1)**

> **作者:** Cyril Shih-Huan Hsu; Xi Li; Lanfranco Zanzi; Zhiheng Yang; Chrysa Papagianni; Xavier Costa Pérez
>
> **备注:** This paper has been accepted for publication at IEEE International Conference on Communications (ICC) 2026
>
> **摘要:** Recent advancements in mobile and wireless networks are unlocking the full potential of robotic autonomy, enabling robots to take advantage of ultra-low latency, high data throughput, and ubiquitous connectivity. However, for robots to navigate and operate seamlessly, efficiently and reliably, they must have an accurate understanding of both their surrounding environment and the quality of radio signals. Achieving this in highly dynamic and ever-changing environments remains a challenging and largely unsolved problem. In this paper, we introduce MapViT, a two-stage Vision Transformer (ViT)-based framework inspired by the success of pre-train and fine-tune paradigm for Large Language Models (LLMs). MapViT is designed to predict both environmental changes and expected radio signal quality. We evaluate the framework using a set of representative Machine Learning (ML) models, analyzing their respective strengths and limitations across different scenarios. Experimental results demonstrate that the proposed two-stage pipeline enables real-time prediction, with the ViT-based implementation achieving a strong balance between accuracy and computational efficiency. This makes MapViT a promising solution for energy- and resource-constrained platforms such as mobile robots. Moreover, the geometry foundation model derived from the self-supervised pre-training stage improves data efficiency and transferability, enabling effective downstream predictions even with limited labeled data. Overall, this work lays the foundation for next-generation digital twin ecosystems, and it paves the way for a new class of ML foundation models driving multi-modal intelligence in future 6G-enabled systems.
>
---
## 更新

#### [replaced 001] VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于多智能体协作任务，旨在解决动态环境中视觉驱动的协作问题。提出VIKI-Bench基准和VIKI-R框架，提升多机器人协作性能。**

- **链接: [https://arxiv.org/pdf/2506.09049v3](https://arxiv.org/pdf/2506.09049v3)**

> **作者:** Li Kang; Xiufeng Song; Heng Zhou; Yiran Qin; Jie Yang; Xiaohong Liu; Philip Torr; Lei Bai; Zhenfei Yin
>
> **备注:** Accepted by NeurIPS 2025 Track on Datasets and Benchmarks. Project page: https://faceong.github.io/VIKI-R/
>
> **摘要:** Coordinating multiple embodied agents in dynamic environments remains a core challenge in artificial intelligence, requiring both perception-driven reasoning and scalable cooperation strategies. While recent works have leveraged large language models (LLMs) for multi-agent planning, a few have begun to explore vision-language models (VLMs) for visual reasoning. However, these VLM-based approaches remain limited in their support for diverse embodiment types. In this work, we introduce VIKI-Bench, the first hierarchical benchmark tailored for embodied multi-agent cooperation, featuring three structured levels: agent activation, task planning, and trajectory perception. VIKI-Bench includes diverse robot embodiments, multi-view visual observations, and structured supervision signals to evaluate reasoning grounded in visual inputs. To demonstrate the utility of VIKI-Bench, we propose VIKI-R, a two-stage framework that fine-tunes a pretrained vision-language model (VLM) using Chain-of-Thought annotated demonstrations, followed by reinforcement learning under multi-level reward signals. Our extensive experiments show that VIKI-R significantly outperforms baselines method across all task levels. Furthermore, we show that reinforcement learning enables the emergence of compositional cooperation patterns among heterogeneous agents. Together, VIKI-Bench and VIKI-R offer a unified testbed and method for advancing multi-agent, visual-driven cooperation in embodied AI systems.
>
---
#### [replaced 002] Reinforcement Learning Compensated Model Predictive Control for Off-road Driving on Unknown Deformable Terrain
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自主驾驶任务，解决未知变形地形下的高速控制问题。通过结合强化学习与模型预测控制，提升控制性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2408.09253v2](https://arxiv.org/pdf/2408.09253v2)**

> **作者:** Prakhar Gupta; Jonathon M. Smereka; Yunyi Jia
>
> **备注:** Submitted to IEEE Transactions on Intelligent Vehicles as a Regular Paper; was withdrawn in March 2025. A revised version of this manuscript was submitted to ACC 2025 review as a regular paper in Sep 2025
>
> **摘要:** This study presents an Actor-Critic reinforcement learning Compensated Model Predictive Controller (AC2MPC) designed for high-speed, off-road autonomous driving on deformable terrains. Addressing the difficulty of modeling unknown tire-terrain interaction and ensuring real-time control feasibility and performance, this framework integrates deep reinforcement learning with a model predictive controller to manage unmodeled nonlinear dynamics. We evaluate the controller framework over constant and varying velocity profiles using high-fidelity simulator Project Chrono. Our findings demonstrate that our controller statistically outperforms standalone model-based and learning-based controllers over three unknown terrains that represent sandy deformable track, sandy and rocky track and cohesive clay-like deformable soil track. Despite varied and previously unseen terrain characteristics, this framework generalized well enough to track longitudinal reference speeds with the least error. Furthermore, this framework required significantly less training data compared to purely learning based controller, converging in fewer steps while delivering better performance. Even when under-trained, this controller outperformed the standalone controllers, highlighting its potential for safer and more efficient real-world deployment.
>
---
#### [replaced 003] BayesianVLA: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在新指令和复杂任务中的泛化问题。通过引入贝叶斯分解和潜在动作查询，提升语言指导的准确性。**

- **链接: [https://arxiv.org/pdf/2601.15197v2](https://arxiv.org/pdf/2601.15197v2)**

> **作者:** Shijie Lian; Bin Yu; Xiaopeng Lin; Laurence T. Yang; Zhaolong Shen; Changti Wu; Yuzhuo Miao; Cong Huang; Kai Chen
>
> **摘要:** Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose BayesianVLA, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $π(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, BayesianVLA significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.
>
---
#### [replaced 004] PAD-TRO: Projection-Augmented Diffusion for Direct Trajectory Optimization
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于轨迹优化任务，解决动态可行性问题。提出PAD-TRO方法，通过模型扩散直接生成状态序列，并引入无梯度投影机制，提升无人机导航成功率。**

- **链接: [https://arxiv.org/pdf/2510.04436v2](https://arxiv.org/pdf/2510.04436v2)**

> **作者:** Jushan Chen; Santiago Paternain
>
> **备注:** Accepted for publication at the 2026 American Control Conference
>
> **摘要:** Recently, diffusion models have gained popularity and attention in trajectory optimization due to their capability of modeling multi-modal probability distributions. However, addressing nonlinear equality constraints, i.e, dynamic feasibility, remains a great challenge in diffusion-based trajectory optimization. Recent diffusion-based trajectory optimization frameworks rely on a single-shooting style approach where the denoised control sequence is applied to forward propagate the dynamical system, which cannot explicitly enforce constraints on the states and frequently leads to sub-optimal solutions. In this work, we propose a novel direct trajectory optimization approach via model-based diffusion, which directly generates a sequence of states. To ensure dynamic feasibility, we propose a gradient-free projection mechanism that is incorporated into the reverse diffusion process. Our results show that, compared to a recent state-of-the-art baseline, our approach leads to zero dynamic feasibility error and approximately 4x higher success rate in a quadrotor waypoint navigation scenario involving dense static obstacles.
>
---
#### [replaced 005] Data-driven tool wear prediction in milling, based on a process-integrated single-sensor approach
- **分类: cs.LG; cs.RO; eess.SP**

- **简介: 该论文属于工具磨损预测任务，旨在解决传统方法依赖多传感器、数据量大的问题。通过单传感器数据和迁移学习，提升模型泛化能力，验证了多种模型的有效性。**

- **链接: [https://arxiv.org/pdf/2412.19950v5](https://arxiv.org/pdf/2412.19950v5)**

> **作者:** Eric Hirsch; Christian Friedrich
>
> **备注:** This work is a preprint and has been submitted for possible publication,14 pages, 12 figures
>
> **摘要:** Accurate tool wear prediction is essential for maintaining productivity and minimizing costs in machining. However, the complex nature of the tool wear process poses significant challenges to achieving reliable predictions. This study explores data-driven methods, in particular deep learning, for tool wear prediction. Traditional data-driven approaches often focus on a single process, relying on multi-sensor setups and extensive data generation, which limits generalization to new settings. Moreover, multi-sensor integration is often impractical in industrial environments. To address these limitations, this research investigates the transferability of predictive models using minimal training data, validated across two processes. Furthermore, it uses a simple setup with a single acceleration sensor to establish a low-cost data generation approach that facilitates the generalization of models to other processes via transfer learning. The study evaluates several machine learning models, including transformer-inspired convolutional neural networks (CNN), long short-term memory networks (LSTM), support vector machines (SVM), and decision trees, trained on different input formats such as feature vectors and short-time Fourier transform (STFT). The performance of the models is evaluated on two machines and on different amounts of training data, including scenarios with significantly reduced datasets, providing insight into their effectiveness under constrained data conditions. The results demonstrate the potential of specific models and configurations for effective tool wear prediction, contributing to the development of more adaptable and efficient predictive maintenance strategies in machining. Notably, the ConvNeXt model has an exceptional performance, achieving 99.1\% accuracy in identifying tool wear using data from only four milling tools operated until they are worn.
>
---
#### [replaced 006] Multi-Layered Reasoning from a Single Viewpoint for Learning See-Through Grasping
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决无外部传感器的抓取问题。提出VBSeeThruP架构，通过单目视觉实现多模态感知与反应式抓取。**

- **链接: [https://arxiv.org/pdf/2312.09822v5](https://arxiv.org/pdf/2312.09822v5)**

> **作者:** Fang Wan; Chaoyang Song
>
> **备注:** 39 pages, 13 figures, 2 tables, for supplementary videos, see https://bionicdl.ancorasir.com/?p=1658, for opensourced codes, see https://github.com/ancorasir/SeeThruFinger
>
> **摘要:** Sensory substitution enables biological systems to perceive stimuli that are typically perceived by another organ, which is inspirational for physical agents. Multimodal perception of intrinsic and extrinsic interactions is critical in building an intelligent robot that learns. This study presents a Vision-based See-Through Perception (VBSeeThruP) architecture that simultaneously perceives multiple intrinsic and extrinsic modalities from a single visual input, in a markerless manner, all packed into a soft robotic finger using the Soft Polyhedral Network design. It is generally applicable to miniature vision systems placed beneath deformable networks with a see-through design, capturing real-time images of the network's physical interactions induced by contact-based events, overlaid on the visual scene of the external environment, as demonstrated in the ablation study. We present the VBSeeThruP's capability for learning reactive grasping without using external cameras or dedicated force and torque sensors on the fingertips. Using the inpainted scene and the deformation mask, we further demonstrate the multimodal performance of the VBSeeThruP architecture to simultaneously achieve various perceptions, including but not limited to scene inpainting, object detection, depth sensing, scene segmentation, masked deformation tracking, 6D force/torque sensing, and contact event detection, all within a single sensory input from the in-finger vision markerlessly.
>
---
#### [replaced 007] Who Is Responsible? Self-Adaptation Under Multiple Concurrent Uncertainties With Unknown Sources in Complex ROS-Based Systems
- **分类: cs.RO**

- **简介: 该论文属于机器人自适应任务，解决复杂ROS系统中多并发不确定性问题。提出基于MAPE-K的自适应方法，处理多源不确定性、级联故障及多种应对策略，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2504.20477v2](https://arxiv.org/pdf/2504.20477v2)**

> **作者:** Andreas Wiedholz; Rafael Paintner; Julian Gleißner; Alwin Hoffmann; Tobias Huber
>
> **备注:** submitted to ACSOS 2025
>
> **摘要:** Robotic systems increasingly operate in dynamic, unpredictable environments, where tightly coupled sensors and software modules increase the probability of a single fault cascading across components and admitting multiple plausible strategies to resolve the underlying uncertainty. Most existing self-adaptive approaches that have been applied to robotics assume predefined one-to-one uncertainty-to-adaptation mappings. We present a ROS2-based self-adaptive approach building upon the MAPE-K feedback loop that addresses (1) multiple simultaneous uncertainties with differing criticality, (2) cascading uncertainties across components, and (3) multiple plausible resolving strategies per detected symptom. Central to our approach is an adaptation rule set which lets designers specify uncertainty patterns, assign criticality levels, and enumerate multiple plausible adaptation strategies. This rule set, combined with an automatically extracted live ROS2 dependency graph, enables lightweight root-cause analysis and strategy ranking to prioritize minimal and effective adaptations. Evaluations on an underwater robot scenario and a perception use case show that our approach can identify root causes among concurrent uncertainties, favours inexpensive adaptations, reduces unnecessary adaptations, and achieves performance comparable to existing baselines designed for sequential uncertainties. The code is publicly available.
>
---
#### [replaced 008] Sigma: The Key for Vision-Language-Action Models toward Telepathic Alignment
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出Sigma模型，解决视觉-语言-动作系统中语义与控制间缺乏动态思维空间的问题，通过架构设计实现感知与动作的意念对齐。**

- **链接: [https://arxiv.org/pdf/2512.00783v3](https://arxiv.org/pdf/2512.00783v3)**

> **作者:** Libo Wang
>
> **备注:** The Sigma model has been open-sourced on Hugging Face. Weights, dataset, some scripts, and logs are all available. The link is: https://huggingface.co/Veltraxor/Sigma
>
> **摘要:** To address a fundamental limitation in cognitive systems, namely the absence of a time-updatable mediating thought space between semantics and continuous control, this work constructs and trains a vision-language-action model termed Sigma, deployed on a single RTX 4090. The model is built upon the open-source pi0.5_base backbone, with the svla_so101_pickplace dataset preprocessed into a structured training corpus. An independently designed VLA architecture is introduced to integrate deep semantic understanding with associative reasoning, enabling telepathic-style alignment between perception and action. Training proceeds through iterative optimization of data preprocessing, LoRA-based fine-tuning, and inference-stage adapter design. Evaluation is conducted using offline closed-loop replay, comparing Sigma against the untuned pi0.5_base under identical data conditions. Experimental results indicate a consistent reduction in control MSE across vector-, fragment-, and trajectory-level scales, while preserving the stability of the telepathy norm and semantic-text alignment quality. These findings demonstrate that mind-responsive alignment control can be quantitatively achieved through semantic and associative architectural integration without retraining the base model, providing a reproducible pathway for semantic alignment and intention-driven behavior.
>
---
#### [replaced 009] ProbeMDE: Uncertainty-Guided Active Proprioception for Monocular Depth Estimation in Surgical Robotics
- **分类: cs.RO**

- **简介: 该论文属于单目深度估计任务，旨在解决手术场景中深度预测不确定和不准确的问题。通过结合RGB图像与稀疏本体感觉测量，利用集成模型和SVGD优化测量位置，提高精度并减少测量次数。**

- **链接: [https://arxiv.org/pdf/2512.11773v3](https://arxiv.org/pdf/2512.11773v3)**

> **作者:** Britton Jordan; Jordan Thompson; Jesse F. d'Almeida; Hao Li; Nithesh Kumar; Susheela Sharma Stern; James Ferguson; Ipek Oguz; Robert J. Webster; Daniel Brown; Alan Kuntz
>
> **备注:** 9 pages, 5 figures. Project page: https://brittonjordan.github.io/probe_mde/
>
> **摘要:** Monocular depth estimation (MDE) provides a useful tool for robotic perception, but its predictions are often uncertain and inaccurate in challenging environments such as surgical scenes where textureless surfaces, specular reflections, and occlusions are common. To address this, we propose ProbeMDE, a cost-aware active sensing framework that combines RGB images with sparse proprioceptive measurements for MDE. Our approach utilizes an ensemble of MDE models to predict dense depth maps conditioned on both RGB images and on a sparse set of known depth measurements obtained via proprioception, where the robot has touched the environment in a known configuration. We quantify predictive uncertainty via the ensemble's variance and measure the gradient of the uncertainty with respect to candidate measurement locations. To prevent mode collapse while selecting maximally informative locations to propriocept (touch), we leverage Stein Variational Gradient Descent (SVGD) over this gradient map. We validate our method in both simulated and physical experiments on central airway obstruction surgical phantoms. Our results demonstrate that our approach outperforms baseline methods across standard depth estimation metrics, achieving higher accuracy while minimizing the number of required proprioceptive measurements. Project page: https://brittonjordan.github.io/probe_mde/
>
---
#### [replaced 010] Towards Natural Language Environment: Understanding Seamless Natural-Language-Based Human-Multi-Robot Interactions
- **分类: cs.HC; cs.RO**

- **简介: 该论文探讨自然语言环境下的多机器人协作问题，属于人机交互任务。研究旨在理解人类与多机器人通过自然语言协调的机制，通过分析设计空间和角色扮演实验，提出相关设计建议。**

- **链接: [https://arxiv.org/pdf/2601.13338v2](https://arxiv.org/pdf/2601.13338v2)**

> **作者:** Ziyi Liu; Xinyi Wang; Shao-Kang Hsia; Chenfei Zhu; Zhengzhe Zhu; Xiyun Hu; Anastasia Kouvaras Ostrowski; Karthik Ramani
>
> **摘要:** As multiple robots are expected to coexist in future households, natural language is increasingly envisioned as a primary medium for human-robot and robot-robot communication. This paper introduces the concept of a Natural Language Environment (NLE), defined as an interaction space in which humans and multiple heterogeneous robots coordinate primarily through natural language. Rather than proposing a deployable system, this work aims to explore the design space of such environments. We first synthesize prior work on language-based human-robot interaction to derive a preliminary design space for NLEs. We then conduct a role-playing study in virtual reality to investigate how people conceptualize, negotiate, and coordinate human-multi-robot interactions within this imagined environment. Based on qualitative and quantitative analysis, we refine the preliminary design space and derive design implications that highlight key tensions and opportunities around task coordination dominance, robot autonomy, and robot personality in Natural Language Environments.
>
---
