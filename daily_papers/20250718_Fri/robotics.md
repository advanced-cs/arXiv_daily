# 机器人 cs.RO

- **最新发布 39 篇**

- **更新 20 篇**

## 最新发布

#### [new 001] osmAG-LLM: Zero-Shot Open-Vocabulary Object Navigation via Semantic Maps and Large Language Models Reasoning
- **分类: cs.RO**

- **简介: 该论文属于开放词汇目标导航任务，解决动态环境中对象位置不确定的问题。通过结合语义地图与大语言模型推理，实现更鲁棒的导航效果。**

- **链接: [http://arxiv.org/pdf/2507.12753v1](http://arxiv.org/pdf/2507.12753v1)**

> **作者:** Fujing Xie; Sören Schwertfeger; Hermann Blum
>
> **摘要:** Recent open-vocabulary robot mapping methods enrich dense geometric maps with pre-trained visual-language features, achieving a high level of detail and guiding robots to find objects specified by open-vocabulary language queries. While the issue of scalability for such approaches has received some attention, another fundamental problem is that high-detail object mapping quickly becomes outdated, as objects get moved around a lot. In this work, we develop a mapping and navigation system for object-goal navigation that, from the ground up, considers the possibilities that a queried object can have moved, or may not be mapped at all. Instead of striving for high-fidelity mapping detail, we consider that the main purpose of a map is to provide environment grounding and context, which we combine with the semantic priors of LLMs to reason about object locations and deploy an active, online approach to navigate to the objects. Through simulated and real-world experiments we find that our approach tends to have higher retrieval success at shorter path lengths for static objects and by far outperforms prior approaches in cases of dynamic or unmapped object queries. We provide our code and dataset at: https://anonymous.4open.science/r/osmAG-LLM.
>
---
#### [new 002] Robustness Requirement Coverage using a Situation Coverage Approach for Vision-based AI Systems
- **分类: cs.RO**

- **简介: 该论文属于AI系统安全任务，解决传感器退化导致的感知可靠性问题，通过结合噪声因子分析与情境覆盖，提出系统化提取鲁棒性安全需求的方法。**

- **链接: [http://arxiv.org/pdf/2507.12986v1](http://arxiv.org/pdf/2507.12986v1)**

> **作者:** Sepeedeh Shahbeigi; Nawshin Mannan Proma; Victoria Hodge; Richard Hawkins; Boda Li; Valentina Donzella
>
> **备注:** 4 pages, 1 figure
>
> **摘要:** AI-based robots and vehicles are expected to operate safely in complex and dynamic environments, even in the presence of component degradation. In such systems, perception relies on sensors such as cameras to capture environmental data, which is then processed by AI models to support decision-making. However, degradation in sensor performance directly impacts input data quality and can impair AI inference. Specifying safety requirements for all possible sensor degradation scenarios leads to unmanageable complexity and inevitable gaps. In this position paper, we present a novel framework that integrates camera noise factor identification with situation coverage analysis to systematically elicit robustness-related safety requirements for AI-based perception systems. We focus specifically on camera degradation in the automotive domain. Building on an existing framework for identifying degradation modes, we propose involving domain, sensor, and safety experts, and incorporating Operational Design Domain specifications to extend the degradation model by incorporating noise factors relevant to AI performance. Situation coverage analysis is then applied to identify representative operational contexts. This work marks an initial step toward integrating noise factor analysis and situational coverage to support principled formulation and completeness assessment of robustness requirements for camera-based AI perception.
>
---
#### [new 003] Refining Motion for Peak Performance: Identifying Optimal Gait Parameters for Energy-Efficient Quadrupedal Bounding
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在解决 quadrupedal 机器人能耗问题。通过优化步态参数提升能量效率，进行了仿真与实验验证。**

- **链接: [http://arxiv.org/pdf/2507.12751v1](http://arxiv.org/pdf/2507.12751v1)**

> **作者:** Yasser G. Alqaham; Jing Cheng; Zhenyu Gan
>
> **备注:** Published in the ACC 2025 Conference proceedings
>
> **摘要:** Energy efficiency is a critical factor in the performance and autonomy of quadrupedal robots. While previous research has focused on mechanical design and actuation improvements, the impact of gait parameters on energetics has been less explored. In this paper, we hypothesize that gait parameters, specifically duty factor, phase shift, and stride duration, are key determinants of energy consumption in quadrupedal locomotion. To test this hypothesis, we modeled the Unitree A1 quadrupedal robot and developed a locomotion controller capable of independently adjusting these gait parameters. Simulations of bounding gaits were conducted in Gazebo across a range of gait parameters at three different speeds: low, medium, and high. Experimental tests were also performed to validate the simulation results. The findings demonstrate that optimizing gait parameters can lead to significant reductions in energy consumption, enhancing the overall efficiency of quadrupedal locomotion. This work contributes to the advancement of energy-efficient control strategies for legged robots, offering insights directly applicable to commercially available platforms.
>
---
#### [new 004] Aligning Humans and Robots via Reinforcement Learning from Implicit Human Feedback
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于强化学习任务，旨在解决稀疏奖励下策略学习困难的问题。通过利用EEG信号提供隐式反馈，实现人机协作的高效学习。**

- **链接: [http://arxiv.org/pdf/2507.13171v1](http://arxiv.org/pdf/2507.13171v1)**

> **作者:** Suzie Kim; Hye-Bin Shin; Seong-Whan Lee
>
> **摘要:** Conventional reinforcement learning (RL) ap proaches often struggle to learn effective policies under sparse reward conditions, necessitating the manual design of complex, task-specific reward functions. To address this limitation, rein forcement learning from human feedback (RLHF) has emerged as a promising strategy that complements hand-crafted rewards with human-derived evaluation signals. However, most existing RLHF methods depend on explicit feedback mechanisms such as button presses or preference labels, which disrupt the natural interaction process and impose a substantial cognitive load on the user. We propose a novel reinforcement learning from implicit human feedback (RLIHF) framework that utilizes non-invasive electroencephalography (EEG) signals, specifically error-related potentials (ErrPs), to provide continuous, implicit feedback without requiring explicit user intervention. The proposed method adopts a pre-trained decoder to transform raw EEG signals into probabilistic reward components, en abling effective policy learning even in the presence of sparse external rewards. We evaluate our approach in a simulation environment built on the MuJoCo physics engine, using a Kinova Gen2 robotic arm to perform a complex pick-and-place task that requires avoiding obstacles while manipulating target objects. The results show that agents trained with decoded EEG feedback achieve performance comparable to those trained with dense, manually designed rewards. These findings validate the potential of using implicit neural feedback for scalable and human-aligned reinforcement learning in interactive robotics.
>
---
#### [new 005] DEMONSTRATE: Zero-shot Language to Robotic Control via Multi-task Demonstration Learning
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人控制任务，解决如何利用语言指令进行零样本控制的问题。通过多任务示范学习，减少对复杂数学表达的依赖，提升控制效果与可靠性。**

- **链接: [http://arxiv.org/pdf/2507.12855v1](http://arxiv.org/pdf/2507.12855v1)**

> **作者:** Rahel Rickenbach; Bruce Lee; René Zurbrügg; Carmen Amo Alonso; Melanie N. Zeilinger
>
> **摘要:** The integration of large language models (LLMs) with control systems has demonstrated significant potential in various settings, such as task completion with a robotic manipulator. A main reason for this success is the ability of LLMs to perform in-context learning, which, however, strongly relies on the design of task examples, closely related to the target tasks. Consequently, employing LLMs to formulate optimal control problems often requires task examples that contain explicit mathematical expressions, designed by trained engineers. Furthermore, there is often no principled way to evaluate for hallucination before task execution. To address these challenges, we propose DEMONSTRATE, a novel methodology that avoids the use of LLMs for complex optimization problem generations, and instead only relies on the embedding representations of task descriptions. To do this, we leverage tools from inverse optimal control to replace in-context prompt examples with task demonstrations, as well as the concept of multitask learning, which ensures target and example task similarity by construction. Given the fact that hardware demonstrations can easily be collected using teleoperation or guidance of the robot, our approach significantly reduces the reliance on engineering expertise for designing in-context examples. Furthermore, the enforced multitask structure enables learning from few demonstrations and assessment of hallucinations prior to task execution. We demonstrate the effectiveness of our method through simulation and hardware experiments involving a robotic arm tasked with tabletop manipulation.
>
---
#### [new 006] LaViPlan : Language-Guided Visual Path Planning with RLVR
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于自动驾驶任务，解决VLM在OOD场景下的视觉-语言-动作不一致问题，提出LaViPlan框架优化决策准确性。**

- **链接: [http://arxiv.org/pdf/2507.12911v1](http://arxiv.org/pdf/2507.12911v1)**

> **作者:** Hayeon Oh
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Out-of-distribution (OOD) scenarios in autonomous driving refer to situations that deviate from the training domain, often leading to unexpected and potentially hazardous behavior from planners that lack prior exposure to such cases. Recently, Vision-Language Models (VLMs) have been introduced into autonomous driving research for their promising generalization capabilities in OOD settings. Early studies demonstrated that VLMs could recognize OOD scenarios and generate user-level decisions such as "go straight" or "turn right." However, a new challenge has emerged due to the misalignment between the VLM's high-level decisions or visual reasoning expressed in language, and the low-level predicted trajectories interpreted as actions. In this paper, we propose LaViPlan, a framework that leverages Reinforcement Learning with Verifiable Rewards (RLVR) to optimize VLMs using planning-oriented metrics. This approach addresses the vision-language-action misalignment observed in existing VLMs fine-tuned via supervised learning, which can recognize driving scenarios but often produce context-unaware decisions. Experimental results demonstrate that our method improves situational awareness and decision-making under OOD conditions, highlighting its potential to mitigate the misalignment issue. This work introduces a promising post-training paradigm for VLM agents in the context of autonomous driving.
>
---
#### [new 007] Signal Temporal Logic Compliant Co-design of Planning and Control
- **分类: cs.RO**

- **简介: 该论文属于自主机器人任务，解决如何生成符合STL规范的运动规划问题。通过整合轨迹规划与控制，构建满足时空约束的运动方案。**

- **链接: [http://arxiv.org/pdf/2507.13225v1](http://arxiv.org/pdf/2507.13225v1)**

> **作者:** Manas Sashank Juvvi; Tushar Dilip Kurne; Vaishnavi J; Shishir Kolathaya; Pushpak Jagtap
>
> **摘要:** This work presents a novel co-design strategy that integrates trajectory planning and control to handle STL-based tasks in autonomous robots. The method consists of two phases: $(i)$ learning spatio-temporal motion primitives to encapsulate the inherent robot-specific constraints and $(ii)$ constructing an STL-compliant motion plan from these primitives. Initially, we employ reinforcement learning to construct a library of control policies that perform trajectories described by the motion primitives. Then, we map motion primitives to spatio-temporal characteristics. Subsequently, we present a sampling-based STL-compliant motion planning strategy tailored to meet the STL specification. The proposed model-free approach, which generates feasible STL-compliant motion plans across various environments, is validated on differential-drive and quadruped robots across various STL specifications. Demonstration videos are available at https://tinyurl.com/m6zp7rsm.
>
---
#### [new 008] ASC-SW: Atrous strip convolution network with sliding windows for visual-assisted map navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，解决地面障碍物检测问题。提出ASC-SW框架，结合轻量网络和滑动窗口模块，提升检测精度与速度。**

- **链接: [http://arxiv.org/pdf/2507.12744v1](http://arxiv.org/pdf/2507.12744v1)**

> **作者:** Cheng Liu; Fan Zhu; Yaoyu Zhuang Zhinan Chen Jiefeng Tang
>
> **摘要:** With the rapid development of lightweight visual neural network architectures, traditional high-performance vision models have undergone significant compression, greatly improving their computational efficiency and energy consumption ratio. This makes them feasible for deployment on resource-constrained edge computing devices. We propose a visual-assisted navigation framework called Atrous Strip Convolution-Sliding Window (ASC-SW), which leverages a depth camera and a lightweight visual neural network to assist map-based mobile robot navigation. This framework compensates for the inability of traditional light detection and range (LiDAR) sensors to detect ground-level obstacles such as ground-level wires. We introduce a lightweight and efficient segmentation model, Atrous Strip Convolution Network (ASCnet), for detecting deformable linear objects (DLOs). MobileNetV2 is used as the backbone network, and Atrous Strip Convolution Spatial Pyramid Pooling (ASCSPP) is designed to extract DLO features more effectively. Atrous Strip Convolution is integrated into ASCSPP to accurately identify the linear structure of DLOs with low computational cost. Additionally, a Sliding Window (SW) post-processing module is proposed to denoise the output in complex environments, improving recognition accuracy. Our method strikes a balance between inference speed and segmentation performance. It achieves a mean Intersection over Union (Miou) score of 75.3% on a self-built dataset and reaches 9.3 FPS inference speed on the Jetson Orin Nano edge device. Overall, our approach outperforms existing DLO detection models and has been successfully validated on a physical robotic platform.
>
---
#### [new 009] Learning to Predict Mobile Robot Stability in Off-Road Environments
- **分类: cs.RO**

- **简介: 该论文属于移动机器人稳定性预测任务，解决越野环境下稳定性评估问题。通过IMU数据和视觉信息，利用神经网络估计机器人稳定性，无需精确地形或力感知。**

- **链接: [http://arxiv.org/pdf/2507.12731v1](http://arxiv.org/pdf/2507.12731v1)**

> **作者:** Nathaniel Rose; Arif Ahmed; Emanuel Gutierrez-Cornejo; Parikshit Maini
>
> **备注:** Nathaniel Rose and Arif Ahmed contributed equally to this work. Accepted poster for RSS 2025 Workshop on Resilient Off-road Autonomous Robotics. 8 pages, 8 figures, 1 table
>
> **摘要:** Navigating in off-road environments for wheeled mobile robots is challenging due to dynamic and rugged terrain. Traditional physics-based stability metrics, such as Static Stability Margin (SSM) or Zero Moment Point (ZMP) require knowledge of contact forces, terrain geometry, and the robot's precise center-of-mass that are difficult to measure accurately in real-world field conditions. In this work, we propose a learning-based approach to estimate robot platform stability directly from proprioceptive data using a lightweight neural network, IMUnet. Our method enables data-driven inference of robot stability without requiring an explicit terrain model or force sensing. We also develop a novel vision-based ArUco tracking method to compute a scalar score to quantify robot platform stability called C3 score. The score captures image-space perturbations over time as a proxy for physical instability and is used as a training signal for the neural network based model. As a pilot study, we evaluate our approach on data collected across multiple terrain types and speeds and demonstrate generalization to previously unseen conditions. These initial results highlight the potential of using IMU and robot velocity as inputs to estimate platform stability. The proposed method finds application in gating robot tasks such as precision actuation and sensing, especially for mobile manipulation tasks in agricultural and space applications. Our learning method also provides a supervision mechanism for perception based traversability estimation and planning.
>
---
#### [new 010] FFI-VTR: Lightweight and Robust Visual Teach and Repeat Navigation based on Feature Flow Indicator and Probabilistic Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉导航任务，解决无精确定位下的高效鲁棒导航问题。提出FFI-VTR方法，利用特征流和概率规划实现轻量级导航。**

- **链接: [http://arxiv.org/pdf/2507.12800v1](http://arxiv.org/pdf/2507.12800v1)**

> **作者:** Jikai Wang; Yunqi Cheng; Zonghai Chen
>
> **摘要:** Though visual and repeat navigation is a convenient solution for mobile robot self-navigation, achieving balance between efficiency and robustness in task environment still remains challenges. In this paper, we propose a novel visual and repeat robotic autonomous navigation method that requires no accurate localization and dense reconstruction modules, which makes our system featured by lightweight and robustness. Firstly, feature flow is introduced and we develop a qualitative mapping between feature flow and robot's motion, in which feature flow is defined as pixel location bias between matched features. Based on the mapping model, the map outputted by the teaching phase is represented as a keyframe graph, in which the feature flow on the edge encodes the relative motion between adjacent keyframes. Secondly, the visual repeating navigation is essentially modeled as a feature flow minimization problem between current observation and the map keyframe. To drive the robot to consistently reduce the feature flow between current frame and map keyframes without accurate localization, a probabilistic motion planning is developed based on our qualitative feature flow-motion mapping indicator. Extensive experiments using our mobile platform demonstrates that our proposed method is lightweight, robust, and superior to baselines. The source code has been made public at https://github.com/wangjks/FFI-VTR to benefit the community.
>
---
#### [new 011] Efficient Online Learning and Adaptive Planning for Robotic Information Gathering Based on Streaming Data
- **分类: cs.RO**

- **简介: 该论文属于机器人信息收集任务，解决未知或动态环境中高效路径规划问题。提出基于流式稀疏高斯过程的自适应规划方法，提升实时性能。**

- **链接: [http://arxiv.org/pdf/2507.13053v1](http://arxiv.org/pdf/2507.13053v1)**

> **作者:** Sanjeev Ramkumar Sudha; Joel Jose; Erlend M. Coates
>
> **摘要:** Robotic information gathering (RIG) techniques refer to methods where mobile robots are used to acquire data about the physical environment with a suite of sensors. Informative planning is an important part of RIG where the goal is to find sequences of actions or paths that maximize efficiency or the quality of information collected. Many existing solutions solve this problem by assuming that the environment is known in advance. However, real environments could be unknown or time-varying, and adaptive informative planning remains an active area of research. Adaptive planning and incremental online mapping are required for mapping initially unknown or varying spatial fields. Gaussian process (GP) regression is a widely used technique in RIG for mapping continuous spatial fields. However, it falls short in many applications as its real-time performance does not scale well to large datasets. To address these challenges, this paper proposes an efficient adaptive informative planning approach for mapping continuous scalar fields with GPs with streaming sparse GPs. Simulation experiments are performed with a synthetic dataset and compared against existing benchmarks. Finally, it is also verified with a real-world dataset to further validate the efficacy of the proposed method. Results show that our method achieves similar mapping accuracy to the baselines while reducing computational complexity for longer missions.
>
---
#### [new 012] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出FOUNDER框架，将基础模型与世界模型结合，解决无奖励的具身任务决策问题，通过映射实现目标条件策略学习。**

- **链接: [http://arxiv.org/pdf/2507.12496v1](http://arxiv.org/pdf/2507.12496v1)**

> **作者:** Yucen Wang; Rui Yu; Shenghua Wan; Le Gan; De-Chuan Zhan
>
> **备注:** Accepted by Forty-Second International Conference on Machine Learning (ICML 2025)
>
> **摘要:** Foundation Models (FMs) and World Models (WMs) offer complementary strengths in task generalization at different levels. In this work, we propose FOUNDER, a framework that integrates the generalizable knowledge embedded in FMs with the dynamic modeling capabilities of WMs to enable open-ended task solving in embodied environments in a reward-free manner. We learn a mapping function that grounds FM representations in the WM state space, effectively inferring the agent's physical states in the world simulator from external observations. This mapping enables the learning of a goal-conditioned policy through imagination during behavior learning, with the mapped task serving as the goal state. Our method leverages the predicted temporal distance to the goal state as an informative reward signal. FOUNDER demonstrates superior performance on various multi-task offline visual control benchmarks, excelling in capturing the deep-level semantics of tasks specified by text or videos, particularly in scenarios involving complex observations or domain gaps where prior methods struggle. The consistency of our learned reward function with the ground-truth reward is also empirically validated. Our project website is https://sites.google.com/view/founder-rl.
>
---
#### [new 013] Few-shot transfer of tool-use skills using human demonstrations with proximity and tactile sensing
- **分类: cs.RO**

- **简介: 该论文属于机器人工具使用任务，旨在解决少样本迁移问题。通过结合触觉和接近传感器，利用人类示范进行策略微调，提升机器人在不同环境中的工具操作能力。**

- **链接: [http://arxiv.org/pdf/2507.13200v1](http://arxiv.org/pdf/2507.13200v1)**

> **作者:** Marina Y. Aoyama; Sethu Vijayakumar; Tetsuya Narita
>
> **备注:** 8 pages, 9 figures, IEEE Robotics and Automation Letters
>
> **摘要:** Tools extend the manipulation abilities of robots, much like they do for humans. Despite human expertise in tool manipulation, teaching robots these skills faces challenges. The complexity arises from the interplay of two simultaneous points of contact: one between the robot and the tool, and another between the tool and the environment. Tactile and proximity sensors play a crucial role in identifying these complex contacts. However, learning tool manipulation using these sensors remains challenging due to limited real-world data and the large sim-to-real gap. To address this, we propose a few-shot tool-use skill transfer framework using multimodal sensing. The framework involves pre-training the base policy to capture contact states common in tool-use skills in simulation and fine-tuning it with human demonstrations collected in the real-world target domain to bridge the domain gap. We validate that this framework enables teaching surface-following tasks using tools with diverse physical and geometric properties with a small number of demonstrations on the Franka Emika robot arm. Our analysis suggests that the robot acquires new tool-use skills by transferring the ability to recognise tool-environment contact relationships from pre-trained to fine-tuned policies. Additionally, combining proximity and tactile sensors enhances the identification of contact states and environmental geometry.
>
---
#### [new 014] ZipMPC: Compressed Context-Dependent MPC Cost via Imitation Learning
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于控制领域，解决MPC计算负担过重的问题。通过模仿学习构建压缩的上下文相关成本函数，提升长时目标优化和实时性能。**

- **链接: [http://arxiv.org/pdf/2507.13088v1](http://arxiv.org/pdf/2507.13088v1)**

> **作者:** Rahel Rickenbach; Alan A. Lahoud; Erik Schaffernicht; Melanie N. Zeilinger; Johannes A. Stork
>
> **摘要:** The computational burden of model predictive control (MPC) limits its application on real-time systems, such as robots, and often requires the use of short prediction horizons. This not only affects the control performance, but also increases the difficulty of designing MPC cost functions that reflect the desired long-term objective. This paper proposes ZipMPC, a method that imitates a long-horizon MPC behaviour by learning a compressed and context-dependent cost function for a short-horizon MPC. It improves performance over alternative methods, such as approximate explicit MPC and automatic cost parameter tuning, in particular in terms of i) optimizing the long term objective; ii) maintaining computational costs comparable to a short-horizon MPC; iii) ensuring constraint satisfaction; and iv) generalizing control behaviour to environments not observed during training. For this purpose, ZipMPC leverages the concept of differentiable MPC with neural networks to propagate gradients of the imitation loss through the MPC optimization. We validate our proposed method in simulation and real-world experiments on autonomous racing. ZipMPC consistently completes laps faster than selected baselines, achieving lap times close to the long-horizon MPC baseline. In challenging scenarios where the short-horizon MPC baseline fails to complete a lap, ZipMPC is able to do so. In particular, these performance gains are also observed on tracks unseen during training.
>
---
#### [new 015] Physically Based Neural LiDAR Resimulation
- **分类: cs.RO; cs.CV; cs.GR; eess.IV**

- **简介: 该论文属于LiDAR模拟任务，解决现有方法对传感器特性建模不足的问题，通过建模滚动快门、激光功率变化等实现更精确的LiDAR仿真。**

- **链接: [http://arxiv.org/pdf/2507.12489v1](http://arxiv.org/pdf/2507.12489v1)**

> **作者:** Richard Marcus; Marc Stamminger
>
> **备注:** Accepted at ITSC 2025, Gold Coast Australia
>
> **摘要:** Methods for Novel View Synthesis (NVS) have recently found traction in the field of LiDAR simulation and large-scale 3D scene reconstruction. While solutions for faster rendering or handling dynamic scenes have been proposed, LiDAR specific effects remain insufficiently addressed. By explicitly modeling sensor characteristics such as rolling shutter, laser power variations, and intensity falloff, our method achieves more accurate LiDAR simulation compared to existing techniques. We demonstrate the effectiveness of our approach through quantitative and qualitative comparisons with state-of-the-art methods, as well as ablation studies that highlight the importance of each sensor model component. Beyond that, we show that our approach exhibits advanced resimulation capabilities, such as generating high resolution LiDAR scans in the camera perspective. Our code and the resulting dataset are available at https://github.com/richardmarcus/PBNLiDAR.
>
---
#### [new 016] GraspGen: A Diffusion-based Framework for 6-DOF Grasping with On-Generator Training
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于6-DOF抓取任务，解决抓取方法泛化性差的问题。提出GraspGen框架，结合扩散模型与判别器，提升抓取生成效果。**

- **链接: [http://arxiv.org/pdf/2507.13097v1](http://arxiv.org/pdf/2507.13097v1)**

> **作者:** Adithyavairavan Murali; Balakumar Sundaralingam; Yu-Wei Chao; Wentao Yuan; Jun Yamada; Mark Carlson; Fabio Ramos; Stan Birchfield; Dieter Fox; Clemens Eppner
>
> **摘要:** Grasping is a fundamental robot skill, yet despite significant research advancements, learning-based 6-DOF grasping approaches are still not turnkey and struggle to generalize across different embodiments and in-the-wild settings. We build upon the recent success on modeling the object-centric grasp generation process as an iterative diffusion process. Our proposed framework, GraspGen, consists of a DiffusionTransformer architecture that enhances grasp generation, paired with an efficient discriminator to score and filter sampled grasps. We introduce a novel and performant on-generator training recipe for the discriminator. To scale GraspGen to both objects and grippers, we release a new simulated dataset consisting of over 53 million grasps. We demonstrate that GraspGen outperforms prior methods in simulations with singulated objects across different grippers, achieves state-of-the-art performance on the FetchBench grasping benchmark, and performs well on a real robot with noisy visual observations.
>
---
#### [new 017] VLMgineer: Vision Language Models as Robotic Toolsmiths
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出VLMgineer框架，利用视觉语言模型进行机器人工具设计与任务执行，解决自动化工具发明问题。**

- **链接: [http://arxiv.org/pdf/2507.12644v1](http://arxiv.org/pdf/2507.12644v1)**

> **作者:** George Jiayuan Gao; Tianyu Li; Junyao Shi; Yihan Li; Zizhe Zhang; Nadia Figueroa; Dinesh Jayaraman
>
> **备注:** Project Website: https://vlmgineer.github.io/release
>
> **摘要:** Tool design and use reflect the ability to understand and manipulate the physical world through creativity, planning, and foresight. As such, these capabilities are often regarded as measurable indicators of intelligence across biological species. While much of today's research on robotic intelligence focuses on generating better controllers, inventing smarter tools offers a complementary form of physical intelligence: shifting the onus of problem-solving onto the tool's design. Given the vast and impressive common-sense, reasoning, and creative capabilities of today's foundation models, we investigate whether these models can provide useful priors to automatically design and effectively wield such tools? We present VLMgineer, a framework that harnesses the code generation abilities of vision language models (VLMs) together with evolutionary search to iteratively co-design physical tools and the action plans that operate them to perform a task. We evaluate VLMgineer on a diverse new benchmark of everyday manipulation scenarios that demand creative tool design and use. Across this suite, VLMgineer consistently discovers tools and policies that solve tasks more effectively and innovatively, transforming challenging robotics problems into straightforward executions. It also outperforms VLM-generated designs from human specifications and existing human-crafted tools for everyday tasks. To facilitate future research on automated tool invention, we will release our benchmark and code.
>
---
#### [new 018] Enter the Mind Palace: Reasoning and Planning for Long-term Active Embodied Question Answering
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究长期主动具身问答（LA-EQA）任务，解决机器人在长时间内结合记忆与探索回答复杂问题的问题。提出结构化记忆系统和信息价值停止准则，提升问答准确性和效率。**

- **链接: [http://arxiv.org/pdf/2507.12846v1](http://arxiv.org/pdf/2507.12846v1)**

> **作者:** Muhammad Fadhil Ginting; Dong-Ki Kim; Xiangyun Meng; Andrzej Reinke; Bandi Jai Krishna; Navid Kayhani; Oriana Peltzer; David D. Fan; Amirreza Shaban; Sung-Kyun Kim; Mykel J. Kochenderfer; Ali-akbar Agha-mohammadi; Shayegan Omidshafiei
>
> **摘要:** As robots become increasingly capable of operating over extended periods -- spanning days, weeks, and even months -- they are expected to accumulate knowledge of their environments and leverage this experience to assist humans more effectively. This paper studies the problem of Long-term Active Embodied Question Answering (LA-EQA), a new task in which a robot must both recall past experiences and actively explore its environment to answer complex, temporally-grounded questions. Unlike traditional EQA settings, which typically focus either on understanding the present environment alone or on recalling a single past observation, LA-EQA challenges an agent to reason over past, present, and possible future states, deciding when to explore, when to consult its memory, and when to stop gathering observations and provide a final answer. Standard EQA approaches based on large models struggle in this setting due to limited context windows, absence of persistent memory, and an inability to combine memory recall with active exploration. To address this, we propose a structured memory system for robots, inspired by the mind palace method from cognitive science. Our method encodes episodic experiences as scene-graph-based world instances, forming a reasoning and planning algorithm that enables targeted memory retrieval and guided navigation. To balance the exploration-recall trade-off, we introduce value-of-information-based stopping criteria that determines when the agent has gathered sufficient information. We evaluate our method on real-world experiments and introduce a new benchmark that spans popular simulation environments and actual industrial sites. Our approach significantly outperforms state-of-the-art baselines, yielding substantial gains in both answer accuracy and exploration efficiency.
>
---
#### [new 019] Non-differentiable Reward Optimization for Diffusion-based Autonomous Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于自主机器人运动规划任务，解决扩散模型难以优化非可导目标（如安全性和有效性）的问题，提出基于强化学习的训练方法提升性能。**

- **链接: [http://arxiv.org/pdf/2507.12977v1](http://arxiv.org/pdf/2507.12977v1)**

> **作者:** Giwon Lee; Daehee Park; Jaewoo Jeong; Kuk-Jin Yoon
>
> **备注:** Accepted at IROS 2025
>
> **摘要:** Safe and effective motion planning is crucial for autonomous robots. Diffusion models excel at capturing complex agent interactions, a fundamental aspect of decision-making in dynamic environments. Recent studies have successfully applied diffusion models to motion planning, demonstrating their competence in handling complex scenarios and accurately predicting multi-modal future trajectories. Despite their effectiveness, diffusion models have limitations in training objectives, as they approximate data distributions rather than explicitly capturing the underlying decision-making dynamics. However, the crux of motion planning lies in non-differentiable downstream objectives, such as safety (collision avoidance) and effectiveness (goal-reaching), which conventional learning algorithms cannot directly optimize. In this paper, we propose a reinforcement learning-based training scheme for diffusion motion planning models, enabling them to effectively learn non-differentiable objectives that explicitly measure safety and effectiveness. Specifically, we introduce a reward-weighted dynamic thresholding algorithm to shape a dense reward signal, facilitating more effective training and outperforming models trained with differentiable objectives. State-of-the-art performance on pedestrian datasets (CrowdNav, ETH-UCY) compared to various baselines demonstrates the versatility of our approach for safe and effective motion planning.
>
---
#### [new 020] Evaluating Reinforcement Learning Algorithms for Navigation in Simulated Robotic Quadrupeds: A Comparative Study Inspired by Guide Dog Behaviour
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人导航任务，旨在解决自主路径规划与避障问题。通过比较三种强化学习算法，提升模拟四足机器人的导航能力。**

- **链接: [http://arxiv.org/pdf/2507.13277v1](http://arxiv.org/pdf/2507.13277v1)**

> **作者:** Emma M. A. Harrison
>
> **摘要:** Robots are increasingly integrated across industries, particularly in healthcare. However, many valuable applications for quadrupedal robots remain overlooked. This research explores the effectiveness of three reinforcement learning algorithms in training a simulated quadruped robot for autonomous navigation and obstacle avoidance. The goal is to develop a robotic guide dog simulation capable of path following and obstacle avoidance, with long-term potential for real-world assistance to guide dogs and visually impaired individuals. It also seeks to expand research into medical 'pets', including robotic guide and alert dogs. A comparative analysis of thirteen related research papers shaped key evaluation criteria, including collision detection, pathfinding algorithms, sensor usage, robot type, and simulation platforms. The study focuses on sensor inputs, collision frequency, reward signals, and learning progression to determine which algorithm best supports robotic navigation in complex environments. Custom-made environments were used to ensure fair evaluation of all three algorithms under controlled conditions, allowing consistent data collection. Results show that Proximal Policy Optimization (PPO) outperformed Deep Q-Network (DQN) and Q-learning across all metrics, particularly in average and median steps to goal per episode. By analysing these results, this study contributes to robotic navigation, AI and medical robotics, offering insights into the feasibility of AI-driven quadruped mobility and its role in assistive robotics.
>
---
#### [new 021] MoCap2GT: A High-Precision Ground Truth Estimator for SLAM Benchmarking Based on Motion Capture and IMU Fusion
- **分类: cs.RO**

- **简介: 该论文属于SLAM基准评估任务，解决MoCap系统生成的地面真实轨迹精度不足的问题。通过融合MoCap与IMU数据，提出MoCap2GT方法提升轨迹估计精度。**

- **链接: [http://arxiv.org/pdf/2507.12920v1](http://arxiv.org/pdf/2507.12920v1)**

> **作者:** Zichao Shu; Shitao Bei; Jicheng Dai; Lijun Li; Zetao Chen
>
> **摘要:** Marker-based optical motion capture (MoCap) systems are widely used to provide ground truth (GT) trajectories for benchmarking SLAM algorithms. However, the accuracy of MoCap-based GT trajectories is mainly affected by two factors: spatiotemporal calibration errors between the MoCap system and the device under test (DUT), and inherent MoCap jitter. Consequently, existing benchmarks focus primarily on absolute translation error, as accurate assessment of rotation and inter-frame errors remains challenging, hindering thorough SLAM evaluation. This paper proposes MoCap2GT, a joint optimization approach that integrates MoCap data and inertial measurement unit (IMU) measurements from the DUT for generating high-precision GT trajectories. MoCap2GT includes a robust state initializer to ensure global convergence, introduces a higher-order B-spline pose parameterization on the SE(3) manifold with variable time offset to effectively model MoCap factors, and employs a degeneracy-aware measurement rejection strategy to enhance estimation accuracy. Experimental results demonstrate that MoCap2GT outperforms existing methods and significantly contributes to precise SLAM benchmarking. The source code is available at https://anonymous.4open.science/r/mocap2gt (temporarily hosted anonymously for double-blind review).
>
---
#### [new 022] ReAL-AD: Towards Human-Like Reasoning in End-to-End Autonomous Driving
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于自动驾驶任务，旨在解决端到端驾驶中缺乏人类层次推理的问题。通过引入三层认知模型和视觉语言模型，提升决策的可解释性和安全性。**

- **链接: [http://arxiv.org/pdf/2507.12499v1](http://arxiv.org/pdf/2507.12499v1)**

> **作者:** Yuhang Lu; Jiadong Tu; Yuexin Ma; Xinge Zhu
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** End-to-end autonomous driving has emerged as a promising approach to unify perception, prediction, and planning within a single framework, reducing information loss and improving adaptability. However, existing methods often rely on fixed and sparse trajectory supervision, limiting their ability to capture the hierarchical reasoning process that human drivers naturally employ. To bridge this gap, we propose ReAL-AD, a Reasoning-Augmented Learning framework that structures decision-making in autonomous driving based on the three-tier human cognitive model: Driving Strategy, Driving Decision, and Driving Operation, where Vision-Language Models (VLMs) are incorporated to enhance situational awareness and structured reasoning across these levels. Specifically, we introduce: (1) the Strategic Reasoning Injector, which formulates high-level driving strategies by interpreting complex traffic contexts from VLM-generated insights; (2) the Tactical Reasoning Integrator, which refines strategic intent into interpretable tactical choices such as lane changes, overtaking, and speed adjustments; and (3) the Hierarchical Trajectory Decoder, which progressively translates tactical decisions into precise control actions for smooth and human-like trajectory execution. Extensive evaluations show that integrating our framework improves planning accuracy and safety by over 30%, making end-to-end autonomous driving more interpretable and aligned with human-like hierarchical reasoning. The project page can be found at: \href{https://4dvlab.github.io/project_page/realad}{\texttt{4dvlab.github.io/project\_page/realad}}
>
---
#### [new 023] Latent Policy Steering with Embodiment-Agnostic Pretrained World Models
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人视觉运动策略学习任务，旨在减少真实世界数据收集需求。通过预训练世界模型和潜在策略引导方法，提升小样本下的策略性能。**

- **链接: [http://arxiv.org/pdf/2507.13340v1](http://arxiv.org/pdf/2507.13340v1)**

> **作者:** Yiqi Wang; Mrinal Verghese; Jeff Schneider
>
> **摘要:** Learning visuomotor policies via imitation has proven effective across a wide range of robotic domains. However, the performance of these policies is heavily dependent on the number of training demonstrations, which requires expensive data collection in the real world. In this work, we aim to reduce data collection efforts when learning visuomotor robot policies by leveraging existing or cost-effective data from a wide range of embodiments, such as public robot datasets and the datasets of humans playing with objects (human data from play). Our approach leverages two key insights. First, we use optic flow as an embodiment-agnostic action representation to train a World Model (WM) across multi-embodiment datasets, and finetune it on a small amount of robot data from the target embodiment. Second, we develop a method, Latent Policy Steering (LPS), to improve the output of a behavior-cloned policy by searching in the latent space of the WM for better action sequences. In real world experiments, we observe significant improvements in the performance of policies trained with a small amount of data (over 50% relative improvement with 30 demonstrations and over 20% relative improvement with 50 demonstrations) by combining the policy with a WM pretrained on two thousand episodes sampled from the existing Open X-embodiment dataset across different robots or a cost-effective human dataset from play.
>
---
#### [new 024] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言导航任务，旨在解决虚拟与物理环境间的差距问题。通过构建VLN-PE平台，评估不同方法在物理机器人中的表现，揭示实际部署挑战。**

- **链接: [http://arxiv.org/pdf/2507.13019v1](http://arxiv.org/pdf/2507.13019v1)**

> **作者:** Liuyi Wang; Xinyuan Xia; Hui Zhao; Hanqing Wang; Tai Wang; Yilun Chen; Chengju Liu; Qijun Chen; Jiangmiao Pang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recent Vision-and-Language Navigation (VLN) advancements are promising, but their idealized assumptions about robot movement and control fail to reflect physically embodied deployment challenges. To bridge this gap, we introduce VLN-PE, a physically realistic VLN platform supporting humanoid, quadruped, and wheeled robots. For the first time, we systematically evaluate several ego-centric VLN methods in physical robotic settings across different technical pipelines, including classification models for single-step discrete action prediction, a diffusion model for dense waypoint prediction, and a train-free, map-based large language model (LLM) integrated with path planning. Our results reveal significant performance degradation due to limited robot observation space, environmental lighting variations, and physical challenges like collisions and falls. This also exposes locomotion constraints for legged robots in complex environments. VLN-PE is highly extensible, allowing seamless integration of new scenes beyond MP3D, thereby enabling more comprehensive VLN evaluation. Despite the weak generalization of current models in physical deployment, VLN-PE provides a new pathway for improving cross-embodiment's overall adaptability. We hope our findings and tools inspire the community to rethink VLN limitations and advance robust, practical VLN models. The code is available at https://crystalsixone.github.io/vln_pe.github.io/.
>
---
#### [new 025] MoistureMapper: An Autonomous Mobile Robot for High-Resolution Soil Moisture Mapping at Scale
- **分类: cs.RO**

- **简介: 该论文属于土壤湿度监测任务，旨在解决高分辨率土壤湿度映射的规模化问题。研究设计并部署了自主机器人MoistureMapper，结合TDR传感器和自适应采样策略，提升测量效率与精度。**

- **链接: [http://arxiv.org/pdf/2507.12716v1](http://arxiv.org/pdf/2507.12716v1)**

> **作者:** Nathaniel Rose; Hannah Chuang; Manuel A Andrade-Rodriguez; Rishi Parashar; Dani Or; Parikshit Maini
>
> **备注:** Accepted by 2025 IEEE 21st International Conference on Automation Science and Engineering. 8 pages, 10 figures, 2 tables
>
> **摘要:** Soil moisture is a quantity of interest in many application areas including agriculture and climate modeling. Existing methods are not suitable for scale applications due to large deployment costs in high-resolution sensing applications such as for variable irrigation. In this work, we design, build and field deploy an autonomous mobile robot, MoistureMapper, for soil moisture sensing. The robot is equipped with Time Domain Reflectometry (TDR) sensors and a direct push drill mechanism for deploying the sensor to measure volumetric water content in the soil. Additionally, we implement and evaluate multiple adaptive sampling strategies based on a Gaussian Process based modeling to build a spatial mapping of moisture distribution in the soil. We present results from large scale computational simulations and proof-of-concept deployment on the field. The adaptive sampling approach outperforms a greedy benchmark approach and results in up to 30\% reduction in travel distance and 5\% reduction in variance in the reconstructed moisture maps. Link to video showing field experiments: https://youtu.be/S4bJ4tRzObg
>
---
#### [new 026] What Can Robots Teach Us About Trust and Reliance? An interdisciplinary dialogue between Social Sciences and Social Robotics
- **分类: cs.RO**

- **简介: 该论文属于跨学科研究任务，旨在解决人与机器人之间信任机制的理解问题。通过结合社会学与机器人学，探讨信任的形成与表现。**

- **链接: [http://arxiv.org/pdf/2507.13041v1](http://arxiv.org/pdf/2507.13041v1)**

> **作者:** Julien Wacquez; Elisabetta Zibetti; Joffrey Becker; Lorenzo Aloe; Fabio Amadio; Salvatore Anzalone; Lola Cañamero; Serena Ivaldi
>
> **摘要:** As robots find their way into more and more aspects of everyday life, questions around trust are becoming increasingly important. What does it mean to trust a robot? And how should we think about trust in relationships that involve both humans and non-human agents? While the field of Human-Robot Interaction (HRI) has made trust a central topic, the concept is often approached in fragmented ways. At the same time, established work in sociology, where trust has long been a key theme, is rarely brought into conversation with developments in robotics. This article argues that we need a more interdisciplinary approach. By drawing on insights from both social sciences and social robotics, we explore how trust is shaped, tested and made visible. Our goal is to open up a dialogue between disciplines and help build a more grounded and adaptable framework for understanding trust in the evolving world of human-robot interaction.
>
---
#### [new 027] Public Evaluation on Potential Social Impacts of Fully Autonomous Cybernetic Avatars for Physical Support in Daily-Life Environments: Large-Scale Demonstration and Survey at Avatar Land
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于社会影响评估任务，旨在研究全自主赛博化身在日常生活中的潜在社会影响。通过大规模演示和调查，分析公众对全自主赛博化身的接受度与担忧。**

- **链接: [http://arxiv.org/pdf/2507.12741v1](http://arxiv.org/pdf/2507.12741v1)**

> **作者:** Lotfi El Hafi; Kazuma Onishi; Shoichi Hasegawa; Akira Oyama; Tomochika Ishikawa; Masashi Osada; Carl Tornberg; Ryoma Kado; Kento Murata; Saki Hashimoto; Sebastian Carrera Villalobos; Akira Taniguchi; Gustavo Alfonso Garcia Ricardez; Yoshinobu Hagiwara; Tatsuya Aoki; Kensuke Iwata; Takato Horii; Yukiko Horikawa; Takahiro Miyashita; Tadahiro Taniguchi; Hiroshi Ishiguro
>
> **备注:** Accepted for presentation at the 2025 IEEE International Conference on Advanced Robotics and its Social Impacts (ARSO), Osaka, Japan
>
> **摘要:** Cybernetic avatars (CAs) are key components of an avatar-symbiotic society, enabling individuals to overcome physical limitations through virtual agents and robotic assistants. While semi-autonomous CAs intermittently require human teleoperation and supervision, the deployment of fully autonomous CAs remains a challenge. This study evaluates public perception and potential social impacts of fully autonomous CAs for physical support in daily life. To this end, we conducted a large-scale demonstration and survey during Avatar Land, a 19-day public event in Osaka, Japan, where fully autonomous robotic CAs, alongside semi-autonomous CAs, performed daily object retrieval tasks. Specifically, we analyzed responses from 2,285 visitors who engaged with various CAs, including a subset of 333 participants who interacted with fully autonomous CAs and shared their perceptions and concerns through a survey questionnaire. The survey results indicate interest in CAs for physical support in daily life and at work. However, concerns were raised regarding task execution reliability. In contrast, cost and human-like interaction were not dominant concerns. Project page: https://lotfielhafi.github.io/FACA-Survey/.
>
---
#### [new 028] MR-LDM -- The Merge-Reactive Longitudinal Decision Model: Game Theoretic Human Decision Modeling for Interactive Sim Agents
- **分类: cs.AI; cs.GT; cs.MA; cs.RO**

- **简介: 该论文属于自动驾驶领域，解决高速公路变道场景中模拟驾驶员行为的问题。通过构建博弈论决策模型，提升仿真环境的真实性与可解释性。**

- **链接: [http://arxiv.org/pdf/2507.12494v1](http://arxiv.org/pdf/2507.12494v1)**

> **作者:** Dustin Holley; Jovin D'sa; Hossein Nourkhiz Mahjoub; Gibran Ali
>
> **备注:** 8 pages
>
> **摘要:** Enhancing simulation environments to replicate real-world driver behavior, i.e., more humanlike sim agents, is essential for developing autonomous vehicle technology. In the context of highway merging, previous works have studied the operational-level yielding dynamics of lag vehicles in response to a merging car at highway on-ramps. Other works focusing on tactical decision modeling generally consider limited action sets or utilize payoff functions with large parameter sets and limited payoff bounds. In this work, we aim to improve the simulation of the highway merge scenario by targeting a game theoretic model for tactical decision-making with improved payoff functions and lag actions. We couple this with an underlying dynamics model to have a unified decision and dynamics model that can capture merging interactions and simulate more realistic interactions in an explainable and interpretable fashion. The proposed model demonstrated good reproducibility of complex interactions when validated on a real-world dataset. The model was finally integrated into a high fidelity simulation environment and confirmed to have adequate computation time efficiency for use in large-scale simulations to support autonomous vehicle development.
>
---
#### [new 029] MindJourney: Test-Time Scaling with World Models for Spatial Reasoning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于空间推理任务，解决VLM在3D场景理解上的不足。通过结合世界模型与VLM，实现测试时的动态推理，提升空间任务性能。**

- **链接: [http://arxiv.org/pdf/2507.12508v1](http://arxiv.org/pdf/2507.12508v1)**

> **作者:** Yuncong Yang; Jiageng Liu; Zheyuan Zhang; Siyuan Zhou; Reuben Tan; Jianwei Yang; Yilun Du; Chuang Gan
>
> **备注:** Project Page: https://umass-embodied-agi.github.io/MindJourney
>
> **摘要:** Spatial reasoning in 3D space is central to human cognition and indispensable for embodied tasks such as navigation and manipulation. However, state-of-the-art vision-language models (VLMs) struggle frequently with tasks as simple as anticipating how a scene will look after an egocentric motion: they perceive 2D images but lack an internal model of 3D dynamics. We therefore propose MindJourney, a test-time scaling framework that grants a VLM with this missing capability by coupling it to a controllable world model based on video diffusion. The VLM iteratively sketches a concise camera trajectory, while the world model synthesizes the corresponding view at each step. The VLM then reasons over this multi-view evidence gathered during the interactive exploration. Without any fine-tuning, our MindJourney achieves over an average 8% performance boost on the representative spatial reasoning benchmark SAT, showing that pairing VLMs with world models for test-time scaling offers a simple, plug-and-play route to robust 3D reasoning. Meanwhile, our method also improves upon the test-time inference VLMs trained through reinforcement learning, which demonstrates the potential of our method that utilizes world models for test-time scaling.
>
---
#### [new 030] $S^2M^2$: Scalable Stereo Matching Model for Reliable Depth Estimation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于立体匹配任务，解决跨分辨率和视差范围的通用性问题。提出S²M²模型，结合多尺度Transformer和新损失函数，实现高精度与高效性。**

- **链接: [http://arxiv.org/pdf/2507.13229v1](http://arxiv.org/pdf/2507.13229v1)**

> **作者:** Junhong Min; Youngpil Jeon; Jimin Kim; Minyong Choi
>
> **备注:** 8 pages, 5 figures, ICCV accepted paper
>
> **摘要:** The pursuit of a generalizable stereo matching model, capable of performing across varying resolutions and disparity ranges without dataset-specific fine-tuning, has revealed a fundamental trade-off. Iterative local search methods achieve high scores on constrained benchmarks, but their core mechanism inherently limits the global consistency required for true generalization. On the other hand, global matching architectures, while theoretically more robust, have been historically rendered infeasible by prohibitive computational and memory costs. We resolve this dilemma with $S^2M^2$: a global matching architecture that achieves both state-of-the-art accuracy and high efficiency without relying on cost volume filtering or deep refinement stacks. Our design integrates a multi-resolution transformer for robust long-range correspondence, trained with a novel loss function that concentrates probability on feasible matches. This approach enables a more robust joint estimation of disparity, occlusion, and confidence. $S^2M^2$ establishes a new state of the art on the Middlebury v3 and ETH3D benchmarks, significantly outperforming prior methods across most metrics while reconstructing high-quality details with competitive efficiency.
>
---
#### [new 031] Generalist Bimanual Manipulation via Foundation Video Diffusion Models
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于双臂机器人操作任务，解决数据稀缺和环境差异问题。通过视频扩散模型和掩码逆动力学模型，实现高效、泛化的动作预测与操作。**

- **链接: [http://arxiv.org/pdf/2507.12898v1](http://arxiv.org/pdf/2507.12898v1)**

> **作者:** Yao Feng; Hengkai Tan; Xinyi Mao; Guodong Liu; Shuhe Huang; Chendong Xiang; Hang Su; Jun Zhu
>
> **摘要:** Bimanual robotic manipulation, which involves the coordinated control of two robotic arms, is foundational for solving challenging tasks. Despite recent progress in general-purpose manipulation, data scarcity and embodiment heterogeneity remain serious obstacles to further scaling up in bimanual settings. In this paper, we introduce VIdeo Diffusion for Action Reasoning (VIDAR), a two-stage framework that leverages large-scale, diffusion-based video pre-training and a novel masked inverse dynamics model for action prediction. We pre-train the video diffusion model on 750K multi-view videos from three real-world bimanual robot platforms, utilizing a unified observation space that encodes robot, camera, task, and scene contexts. Our masked inverse dynamics model learns masks to extract action-relevant information from generated trajectories without requiring pixel-level labels, and the masks can effectively generalize to unseen backgrounds. Our experiments demonstrate that with only 20 minutes of human demonstrations on an unseen robot platform (only 1% of typical data requirements), VIDAR generalizes to unseen tasks and backgrounds with strong semantic understanding, surpassing state-of-the-art methods. Our findings highlight the potential of video foundation models, coupled with masked action prediction, to enable scalable and generalizable robotic manipulation in diverse real-world settings.
>
---
#### [new 032] VITA: Vision-to-Action Flow Matching Policy
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出VITA，一种基于视觉到动作流匹配的策略，用于解决视觉-动作控制任务中的模态差异和生成效率问题。通过结构化动作潜空间实现端到端学习，提升操控性能并降低延迟。**

- **链接: [http://arxiv.org/pdf/2507.13231v1](http://arxiv.org/pdf/2507.13231v1)**

> **作者:** Dechen Gao; Boqi Zhao; Andrew Lee; Ian Chuang; Hanchu Zhou; Hang Wang; Zhe Zhao; Junshan Zhang; Iman Soltani
>
> **备注:** Project page: https://ucd-dare.github.io/VITA/
>
> **摘要:** We present VITA, a Vision-To-Action flow matching policy that evolves latent visual representations into latent actions for visuomotor control. Traditional flow matching and diffusion policies sample from standard source distributions (e.g., Gaussian noise) and require additional conditioning mechanisms like cross-attention to condition action generation on visual information, creating time and space overheads. VITA proposes a novel paradigm that treats latent images as the flow source, learning an inherent mapping from vision to action while eliminating separate conditioning modules and preserving generative modeling capabilities. Learning flows between fundamentally different modalities like vision and action is challenging due to sparse action data lacking semantic structures and dimensional mismatches between high-dimensional visual representations and raw actions. We address this by creating a structured action latent space via an autoencoder as the flow matching target, up-sampling raw actions to match visual representation shapes. Crucially, we supervise flow matching with both encoder targets and final action outputs through flow latent decoding, which backpropagates action reconstruction loss through sequential flow matching ODE solving steps for effective end-to-end learning. Implemented as simple MLP layers, VITA is evaluated on challenging bi-manual manipulation tasks on the ALOHA platform, including 5 simulation and 2 real-world tasks. Despite its simplicity, MLP-only VITA outperforms or matches state-of-the-art generative policies while reducing inference latency by 50-130% compared to conventional flow matching policies requiring different conditioning mechanisms or complex architectures. To our knowledge, VITA is the first MLP-only flow matching policy capable of solving complex bi-manual manipulation tasks like those in ALOHA benchmarks.
>
---
#### [new 033] SE-VLN: A Self-Evolving Vision-Language Navigation Framework Based on Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决传统方法缺乏持续进化能力的问题。提出SE-VLN框架，通过记忆、推理和反思模块实现导航 agent 的自我演化。**

- **链接: [http://arxiv.org/pdf/2507.13152v1](http://arxiv.org/pdf/2507.13152v1)**

> **作者:** Xiangyu Dong; Haoran Zhao; Jiang Gao; Haozhou Li; Xiaoguang Ma; Yaoming Zhou; Fuhai Chen; Juan Liu
>
> **摘要:** Recent advances in vision-language navigation (VLN) were mainly attributed to emerging large language models (LLMs). These methods exhibited excellent generalization capabilities in instruction understanding and task reasoning. However, they were constrained by the fixed knowledge bases and reasoning abilities of LLMs, preventing fully incorporating experiential knowledge and thus resulting in a lack of efficient evolutionary capacity. To address this, we drew inspiration from the evolution capabilities of natural agents, and proposed a self-evolving VLN framework (SE-VLN) to endow VLN agents with the ability to continuously evolve during testing. To the best of our knowledge, it was the first time that an multimodal LLM-powered self-evolving VLN framework was proposed. Specifically, SE-VLN comprised three core modules, i.e., a hierarchical memory module to transfer successful and failure cases into reusable knowledge, a retrieval-augmented thought-based reasoning module to retrieve experience and enable multi-step decision-making, and a reflection module to realize continual evolution. Comprehensive tests illustrated that the SE-VLN achieved navigation success rates of 57% and 35.2% in unseen environments, representing absolute performance improvements of 23.9% and 15.0% over current state-of-the-art methods on R2R and REVERSE datasets, respectively. Moreover, the SE-VLN showed performance improvement with increasing experience repository, elucidating its great potential as a self-evolving agent framework for VLN.
>
---
#### [new 034] AnyPos: Automated Task-Agnostic Actions for Bimanual Manipulation
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于双臂操作任务，解决任务特定数据依赖问题。提出AnyPos模型与ATARA框架，实现无需任务指导的自动化动作学习，提升效率与成功率。**

- **链接: [http://arxiv.org/pdf/2507.12768v1](http://arxiv.org/pdf/2507.12768v1)**

> **作者:** Hengkai Tan; Yao Feng; Xinyi Mao; Shuhe Huang; Guodong Liu; Zhongkai Hao; Hang Su; Jun Zhu
>
> **摘要:** Vision-language-action (VLA) models have shown promise on task-conditioned control in complex settings such as bimanual manipulation. However, the heavy reliance on task-specific human demonstrations limits their generalization and incurs high data acquisition costs. In this work, we present a new notion of task-agnostic action paradigm that decouples action execution from task-specific conditioning, enhancing scalability, efficiency, and cost-effectiveness. To address the data collection challenges posed by this paradigm -- such as low coverage density, behavioral redundancy, and safety risks -- we introduce ATARA (Automated Task-Agnostic Random Actions), a scalable self-supervised framework that accelerates collection by over $ 30\times $ compared to human teleoperation. To further enable effective learning from task-agnostic data, which often suffers from distribution mismatch and irrelevant trajectories, we propose AnyPos, an inverse dynamics model equipped with Arm-Decoupled Estimation and a Direction-Aware Decoder (DAD). We additionally integrate a video-conditioned action validation module to verify the feasibility of learned policies across diverse manipulation tasks. Extensive experiments show that the AnyPos-ATARA pipeline yields a 51% improvement in test accuracy and achieves 30-40% higher success rates in downstream tasks such as lifting, pick-and-place, and clicking, using replay-based video validation. Project Page: https://embodiedfoundation.github.io/vidar_anypos
>
---
#### [new 035] DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉里程计任务，解决单目VO的鲁棒性与泛化性问题。通过融合DINOv2特征与几何信息，提出DINO-VO系统，提升定位精度与效率。**

- **链接: [http://arxiv.org/pdf/2507.13145v1](http://arxiv.org/pdf/2507.13145v1)**

> **作者:** Maulana Bisyir Azhari; David Hyunchul Shim
>
> **备注:** 8 pages, 6 figures. Accepted for publication in IEEE Robotics and Automation Letters (RA-L), July 2025
>
> **摘要:** Learning-based monocular visual odometry (VO) poses robustness, generalization, and efficiency challenges in robotics. Recent advances in visual foundation models, such as DINOv2, have improved robustness and generalization in various vision tasks, yet their integration in VO remains limited due to coarse feature granularity. In this paper, we present DINO-VO, a feature-based VO system leveraging DINOv2 visual foundation model for its sparse feature matching. To address the integration challenge, we propose a salient keypoints detector tailored to DINOv2's coarse features. Furthermore, we complement DINOv2's robust-semantic features with fine-grained geometric features, resulting in more localizable representations. Finally, a transformer-based matcher and differentiable pose estimation layer enable precise camera motion estimation by learning good matches. Against prior detector-descriptor networks like SuperPoint, DINO-VO demonstrates greater robustness in challenging environments. Furthermore, we show superior accuracy and generalization of the proposed feature descriptors against standalone DINOv2 coarse features. DINO-VO outperforms prior frame-to-frame VO methods on the TartanAir and KITTI datasets and is competitive on EuRoC dataset, while running efficiently at 72 FPS with less than 1GB of memory usage on a single GPU. Moreover, it performs competitively against Visual SLAM systems on outdoor driving scenarios, showcasing its generalization capabilities.
>
---
#### [new 036] Continuous Marine Tracking via Autonomous UAV Handoff
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于海洋动物跟踪任务，解决单无人机续航与环境干扰问题，通过多无人机协同实现连续跟踪。**

- **链接: [http://arxiv.org/pdf/2507.12763v1](http://arxiv.org/pdf/2507.12763v1)**

> **作者:** Heegyeong Kim; Alice James; Avishkar Seth; Endrowednes Kuantama; Jane Williamson; Yimeng Feng; Richard Han
>
> **备注:** 6 pages, 5 figures, to be published in DroNet '25: Proceedings of the 10th Workshop on Micro Aerial Vehicle Networks, Systems, and Applications
>
> **摘要:** This paper introduces an autonomous UAV vision system for continuous, real-time tracking of marine animals, specifically sharks, in dynamic marine environments. The system integrates an onboard computer with a stabilised RGB-D camera and a custom-trained OSTrack pipeline, enabling visual identification under challenging lighting, occlusion, and sea-state conditions. A key innovation is the inter-UAV handoff protocol, which enables seamless transfer of tracking responsibilities between drones, extending operational coverage beyond single-drone battery limitations. Performance is evaluated on a curated shark dataset of 5,200 frames, achieving a tracking success rate of 81.9\% during real-time flight control at 100 Hz, and robustness to occlusion, illumination variation, and background clutter. We present a seamless UAV handoff framework, where target transfer is attempted via high-confidence feature matching, achieving 82.9\% target coverage. These results confirm the viability of coordinated UAV operations for extended marine tracking and lay the groundwork for scalable, autonomous monitoring.
>
---
#### [new 037] Deep Bilinear Koopman Model for Real-Time Vehicle Control in Frenet Frame
- **分类: eess.SY; cs.LG; cs.RO; cs.SY; 93C10 (Primary), 93B40, 93C41, 68T07, 93B45 (Secondary); I.2.8; I.2.6; G.1.6; J.7**

- **简介: 该论文属于自动驾驶控制任务，解决车辆动态建模与实时控制问题。提出深度双线性Koopman模型，提升轨迹跟踪精度与实时性。**

- **链接: [http://arxiv.org/pdf/2507.12578v1](http://arxiv.org/pdf/2507.12578v1)**

> **作者:** Mohammad Abtahi; Farhang Motallebi Araghi; Navid Mojahed; Shima Nazari
>
> **备注:** 14 pages, 8 figures. This manuscript is under review with IEEE Transactions on Intelligent Vehicles
>
> **摘要:** Accurate modeling and control of autonomous vehicles remain a fundamental challenge due to the nonlinear and coupled nature of vehicle dynamics. While Koopman operator theory offers a framework for deploying powerful linear control techniques, learning a finite-dimensional invariant subspace for high-fidelity modeling continues to be an open problem. This paper presents a deep Koopman approach for modeling and control of vehicle dynamics within the curvilinear Frenet frame. The proposed framework uses a deep neural network architecture to simultaneously learn the Koopman operator and its associated invariant subspace from the data. Input-state bilinear interactions are captured by the algorithm while preserving convexity, which makes it suitable for real-time model predictive control (MPC) application. A multi-step prediction loss is utilized during training to ensure long-horizon prediction capability. To further enhance real-time trajectory tracking performance, the model is integrated with a cumulative error regulator (CER) module, which compensates for model mismatch by mitigating accumulated prediction errors. Closed-loop performance is evaluated through hardware-in-the-loop (HIL) experiments using a CarSim RT model as the target plant, with real-time validation conducted on a dSPACE SCALEXIO system. The proposed controller achieved significant reductions in tracking error relative to baseline controllers, confirming its suitability for real-time implementation in embedded autonomous vehicle systems.
>
---
#### [new 038] CubeSat Orbit Insertion Maneuvering Using J2 Perturbation
- **分类: astro-ph.EP; astro-ph.IM; cs.RO**

- **简介: 该论文属于航天任务，旨在解决CubeSat轨道插入中的燃料限制问题。通过利用J2扰动优化轨道参数，减少对推进系统的依赖。**

- **链接: [http://arxiv.org/pdf/2507.13017v1](http://arxiv.org/pdf/2507.13017v1)**

> **作者:** M. Amin Alandihallaj; M. Reza Emami
>
> **备注:** Pre-print of IEEE aeroconf paper
>
> **摘要:** The precise insertion of CubeSats into designated orbits is a complex task, primarily due to the limited propulsion capabilities and constrained fuel reserves onboard, which severely restrict the scope for large orbital corrections. This limitation necessitates the development of more efficient maneuvering techniques to ensure mission success. In this paper, we propose a maneuvering sequence that exploits the natural J2 perturbation caused by the Earth's oblateness. By utilizing the secular effects of this perturbation, it is possible to passively influence key orbital parameters such as the argument of perigee and the right ascension of the ascending node, thereby reducing the need for extensive propulsion-based corrections. The approach is designed to optimize the CubeSat's orbital insertion and minimize the total fuel required for trajectory adjustments, making it particularly suitable for fuel-constrained missions. The proposed methodology is validated through comprehensive numerical simulations that examine different initial orbital conditions and perturbation environments. Case studies are presented to demonstrate the effectiveness of the J2-augmented strategy in achieving accurate orbital insertion, showing a major reduction in fuel consumption compared to traditional methods. The results underscore the potential of this approach to extend the operational life and capabilities of CubeSats, offering a viable solution for future low-Earth orbit missions.
>
---
#### [new 039] Intelligent Virtual Sonographer (IVS): Enhancing Physician-Robot-Patient Communication
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决医生、机器人和患者之间的沟通问题。通过构建智能虚拟超声师（IVS），实现三方实时互动与信息传递。**

- **链接: [http://arxiv.org/pdf/2507.13052v1](http://arxiv.org/pdf/2507.13052v1)**

> **作者:** Tianyu Song; Feng Li; Yuan Bi; Angelos Karlas; Amir Yousefi; Daniela Branzan; Zhongliang Jiang; Ulrich Eck; Nassir Navab
>
> **备注:** Accepted at MICCAI 2025
>
> **摘要:** The advancement and maturity of large language models (LLMs) and robotics have unlocked vast potential for human-computer interaction, particularly in the field of robotic ultrasound. While existing research primarily focuses on either patient-robot or physician-robot interaction, the role of an intelligent virtual sonographer (IVS) bridging physician-robot-patient communication remains underexplored. This work introduces a conversational virtual agent in Extended Reality (XR) that facilitates real-time interaction between physicians, a robotic ultrasound system(RUS), and patients. The IVS agent communicates with physicians in a professional manner while offering empathetic explanations and reassurance to patients. Furthermore, it actively controls the RUS by executing physician commands and transparently relays these actions to the patient. By integrating LLM-powered dialogue with speech-to-text, text-to-speech, and robotic control, our system enhances the efficiency, clarity, and accessibility of robotic ultrasound acquisition. This work constitutes a first step toward understanding how IVS can bridge communication gaps in physician-robot-patient interaction, providing more control and therefore trust into physician-robot interaction while improving patient experience and acceptance of robotic ultrasound.
>
---
## 更新

#### [replaced 001] Fast Bilateral Teleoperation and Imitation Learning Using Sensorless Force Control via Accurate Dynamics Model
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.06174v3](http://arxiv.org/pdf/2507.06174v3)**

> **作者:** Koki Yamane; Yunhan Li; Masashi Konosu; Koki Inami; Junji Oaki; Sho Sakaino; Toshiaki Tsuji
>
> **备注:** 20 pages, 9 figures, Submitted to CoRL 2025
>
> **摘要:** In recent years, the advancement of imitation learning has led to increased interest in teleoperating low-cost manipulators to collect demonstration data. However, most existing systems rely on unilateral control, which only transmits target position values. While this approach is easy to implement and suitable for slow, non-contact tasks, it struggles with fast or contact-rich operations due to the absence of force feedback. This work demonstrates that fast teleoperation with force feedback is feasible even with force-sensorless, low-cost manipulators by leveraging 4-channel bilateral control. Based on accurately identified manipulator dynamics, our method integrates nonlinear terms compensation, velocity and external force estimation, and variable gain corresponding to inertial variation. Furthermore, using data collected by 4-channel bilateral control, we show that incorporating force information into both the input and output of learned policies improves performance in imitation learning. These results highlight the practical effectiveness of our system for high-fidelity teleoperation and data collection on affordable hardware.
>
---
#### [replaced 002] BEV-LIO(LC): BEV Image Assisted LiDAR-Inertial Odometry with Loop Closure
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.19242v2](http://arxiv.org/pdf/2502.19242v2)**

> **作者:** Haoxin Cai; Shenghai Yuan; Xinyi Li; Junfeng Guo; Jianqi Liu
>
> **摘要:** This work introduces BEV-LIO(LC), a novel LiDAR-Inertial Odometry (LIO) framework that combines Bird's Eye View (BEV) image representations of LiDAR data with geometry-based point cloud registration and incorporates loop closure (LC) through BEV image features. By normalizing point density, we project LiDAR point clouds into BEV images, thereby enabling efficient feature extraction and matching. A lightweight convolutional neural network (CNN) based feature extractor is employed to extract distinctive local and global descriptors from the BEV images. Local descriptors are used to match BEV images with FAST keypoints for reprojection error construction, while global descriptors facilitate loop closure detection. Reprojection error minimization is then integrated with point-to-plane registration within an iterated Extended Kalman Filter (iEKF). In the back-end, global descriptors are used to create a KD-tree-indexed keyframe database for accurate loop closure detection. When a loop closure is detected, Random Sample Consensus (RANSAC) computes a coarse transform from BEV image matching, which serves as the initial estimate for Iterative Closest Point (ICP). The refined transform is subsequently incorporated into a factor graph along with odometry factors, improving the global consistency of localization. Extensive experiments conducted in various scenarios with different LiDAR types demonstrate that BEV-LIO(LC) outperforms state-of-the-art methods, achieving competitive localization accuracy. Our code and video can be found at https://github.com/HxCa1/BEV-LIO-LC.
>
---
#### [replaced 003] A Roadmap for Climate-Relevant Robotics Research
- **分类: cs.RO; cs.AI; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.11623v2](http://arxiv.org/pdf/2507.11623v2)**

> **作者:** Alan Papalia; Charles Dawson; Laurentiu L. Anton; Norhan Magdy Bayomi; Bianca Champenois; Jung-Hoon Cho; Levi Cai; Joseph DelPreto; Kristen Edwards; Bilha-Catherine Githinji; Cameron Hickert; Vindula Jayawardana; Matthew Kramer; Shreyaa Raghavan; David Russell; Shide Salimi; Jingnan Shi; Soumya Sudhakar; Yanwei Wang; Shouyi Wang; Luca Carlone; Vijay Kumar; Daniela Rus; John E. Fernandez; Cathy Wu; George Kantor; Derek Young; Hanumant Singh
>
> **摘要:** Climate change is one of the defining challenges of the 21st century, and many in the robotics community are looking for ways to contribute. This paper presents a roadmap for climate-relevant robotics research, identifying high-impact opportunities for collaboration between roboticists and experts across climate domains such as energy, the built environment, transportation, industry, land use, and Earth sciences. These applications include problems such as energy systems optimization, construction, precision agriculture, building envelope retrofits, autonomous trucking, and large-scale environmental monitoring. Critically, we include opportunities to apply not only physical robots but also the broader robotics toolkit - including planning, perception, control, and estimation algorithms - to climate-relevant problems. A central goal of this roadmap is to inspire new research directions and collaboration by highlighting specific, actionable problems at the intersection of robotics and climate. This work represents a collaboration between robotics researchers and domain experts in various climate disciplines, and it serves as an invitation to the robotics community to bring their expertise to bear on urgent climate priorities.
>
---
#### [replaced 004] VertiSelector: Automatic Curriculum Learning for Wheeled Mobility on Vertically Challenging Terrain
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.17469v4](http://arxiv.org/pdf/2409.17469v4)**

> **作者:** Tong Xu; Chenhui Pan; Xuesu Xiao
>
> **摘要:** Reinforcement Learning (RL) has the potential to enable extreme off-road mobility by circumventing complex kinodynamic modeling, planning, and control by simulated end-to-end trial-and-error learning experiences. However, most RL methods are sample-inefficient when training in a large amount of manually designed simulation environments and struggle at generalizing to the real world. To address these issues, we introduce VertiSelector (VS), an automatic curriculum learning framework designed to enhance learning efficiency and generalization by selectively sampling training terrain. VS prioritizes vertically challenging terrain with higher Temporal Difference (TD) errors when revisited, thereby allowing robots to learn at the edge of their evolving capabilities. By dynamically adjusting the sampling focus, VS significantly boosts sample efficiency and generalization within the VW-Chrono simulator built on the Chrono multi-physics engine. Furthermore, we provide simulation and physical results using VS on a Verti-4-Wheeler platform. These results demonstrate that VS can achieve 23.08% improvement in terms of success rate by efficiently sampling during training and robustly generalizing to the real world.
>
---
#### [replaced 005] Online Adaptation of Terrain-Aware Dynamics for Planning in Unstructured Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.04484v2](http://arxiv.org/pdf/2506.04484v2)**

> **作者:** William Ward; Sarah Etter; Tyler Ingebrand; Christian Ellis; Adam J. Thorpe; Ufuk Topcu
>
> **备注:** Accepted to RSS-ROAR 2025
>
> **摘要:** Autonomous mobile robots operating in remote, unstructured environments must adapt to new, unpredictable terrains that can change rapidly during operation. In such scenarios, a critical challenge becomes estimating the robot's dynamics on changing terrain in order to enable reliable, accurate navigation and planning. We present a novel online adaptation approach for terrain-aware dynamics modeling and planning using function encoders. Our approach efficiently adapts to new terrains at runtime using limited online data without retraining or fine-tuning. By learning a set of neural network basis functions that span the robot dynamics on diverse terrains, we enable rapid online adaptation to new, unseen terrains and environments as a simple least-squares calculation. We demonstrate our approach for terrain adaptation in a Unity-based robotics simulator and show that the downstream controller has better empirical performance due to higher accuracy of the learned model. This leads to fewer collisions with obstacles while navigating in cluttered environments as compared to a neural ODE baseline.
>
---
#### [replaced 006] Force-Based Viscosity and Elasticity Measurements for Material Biomechanical Characterisation with a Collaborative Robotic Arm
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.11133v2](http://arxiv.org/pdf/2507.11133v2)**

> **作者:** Luca Beber; Edoardo Lamon; Giacomo Moretti; Matteo Saveriano; Luca Fambri; Luigi Palopoli; Daniele Fontanelli
>
> **摘要:** Diagnostic activities, such as ultrasound scans and palpation, are relatively low-cost. They play a crucial role in the early detection of health problems and in assessing their progression. However, they are also error-prone activities, which require highly skilled medical staff. The use of robotic solutions can be key to decreasing the inherent subjectivity of the results and reducing the waiting list. For a robot to perform palpation or ultrasound scans, it must effectively manage physical interactions with the human body, which greatly benefits from precise estimation of the patient's tissue biomechanical properties. This paper assesses the accuracy and precision of a robotic system in estimating the viscoelastic parameters of various materials, including some tests on ex vivo tissues as a preliminary proof-of-concept demonstration of the method's applicability to biological samples. The measurements are compared against a ground truth derived from silicone specimens with different viscoelastic properties, characterised using a high-precision instrument. Experimental results show that the robotic system's accuracy closely matches the ground truth, increasing confidence in the potential use of robots for such clinical applications.
>
---
#### [replaced 007] EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.12440v2](http://arxiv.org/pdf/2507.12440v2)**

> **作者:** Ruihan Yang; Qinxi Yu; Yecheng Wu; Rui Yan; Borui Li; An-Chieh Cheng; Xueyan Zou; Yunhao Fang; Hongxu Yin; Sifei Liu; Song Han; Yao Lu; Xiaolong Wang
>
> **备注:** More videos can be found on our website: https://rchalyang.github.io/EgoVLA
>
> **摘要:** Real robot data collection for imitation learning has led to significant advancements in robotic manipulation. However, the requirement for robot hardware in the process fundamentally constrains the scale of the data. In this paper, we explore training Vision-Language-Action (VLA) models using egocentric human videos. The benefit of using human videos is not only for their scale but more importantly for the richness of scenes and tasks. With a VLA trained on human video that predicts human wrist and hand actions, we can perform Inverse Kinematics and retargeting to convert the human actions to robot actions. We fine-tune the model using a few robot manipulation demonstrations to obtain the robot policy, namely EgoVLA. We propose a simulation benchmark called Ego Humanoid Manipulation Benchmark, where we design diverse bimanual manipulation tasks with demonstrations. We fine-tune and evaluate EgoVLA with Ego Humanoid Manipulation Benchmark and show significant improvements over baselines and ablate the importance of human data. Videos can be found on our website: https://rchalyang.github.io/EgoVLA
>
---
#### [replaced 008] The Role of Integrity Monitoring in Connected and Automated Vehicles: Current State-of-Practice and Future Directions
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2502.04874v3](http://arxiv.org/pdf/2502.04874v3)**

> **作者:** Saswat Priyadarshi Nayak; Matthew Barth
>
> **备注:** \c{opyright} 2022 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Positioning integrity refers to the trust in the performance of a navigation system. Accurate and reliable position information is needed to meet the requirements of connected and Automated Vehicle (CAV) applications, particularly in safety-critical scenarios. Receiver Autonomous Integrity Monitoring (RAIM) and its variants have been widely studied for Global Navigation Satellite System (GNSS)-based vehicle positioning, often fused with kinematic (e.g., Odometry) and perception sensors (e.g., camera). However, integrity monitoring (IM) for cooperative positioning solutions leveraging Vehicle-to-Everything (V2X) communication has received comparatively limited attention. This paper reviews existing research in the field of positioning IM and identifies various research gaps. Particular attention has been placed on identifying research that highlights cooperative IM methods. It also examines key automotive safety standards and public V2X datasets to map current research priorities and uncover critical gaps. Finally, the paper outlines promising future directions, highlighting research topics aimed at advancing and benchmarking positioning integrity.
>
---
#### [replaced 009] Sampling-Based Motion Planning with Discrete Configuration-Space Symmetries
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.00614v2](http://arxiv.org/pdf/2503.00614v2)**

> **作者:** Thomas Cohn; Russ Tedrake
>
> **备注:** Accepted to IROS 2025. 8 pages, 2 figures, 4 tables. Interactive results available at https://cohnt.github.io/projects/symmetries.html
>
> **摘要:** When planning motions in a configuration space that has underlying symmetries (e.g. when manipulating one or multiple symmetric objects), the ideal planning algorithm should take advantage of those symmetries to produce shorter trajectories. However, finite symmetries lead to complicated changes to the underlying topology of configuration space, preventing the use of standard algorithms. We demonstrate how the key primitives used for sampling-based planning can be efficiently implemented in spaces with finite symmetries. A rigorous theoretical analysis, building upon a study of the geometry of the configuration space, shows improvements in the sample complexity of several standard algorithms. Furthermore, a comprehensive slate of experiments demonstrates the practical improvements in both path length and runtime.
>
---
#### [replaced 010] Next-Gen Museum Guides: Autonomous Navigation and Visitor Interaction with an Agentic Robot
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.12273v2](http://arxiv.org/pdf/2507.12273v2)**

> **作者:** Luca Garello; Francesca Cocchella; Alessandra Sciutti; Manuel Catalano; Francesco Rea
>
> **摘要:** Autonomous robots are increasingly being tested into public spaces to enhance user experiences, particularly in cultural and educational settings. This paper presents the design, implementation, and evaluation of the autonomous museum guide robot Alter-Ego equipped with advanced navigation and interactive capabilities. The robot leverages state-of-the-art Large Language Models (LLMs) to provide real-time, context aware question-and-answer (Q&A) interactions, allowing visitors to engage in conversations about exhibits. It also employs robust simultaneous localization and mapping (SLAM) techniques, enabling seamless navigation through museum spaces and route adaptation based on user requests. The system was tested in a real museum environment with 34 participants, combining qualitative analysis of visitor-robot conversations and quantitative analysis of pre and post interaction surveys. Results showed that the robot was generally well-received and contributed to an engaging museum experience, despite some limitations in comprehension and responsiveness. This study sheds light on HRI in cultural spaces, highlighting not only the potential of AI-driven robotics to support accessibility and knowledge acquisition, but also the current limitations and challenges of deploying such technologies in complex, real-world environments.
>
---
#### [replaced 011] Learning Policies for Dynamic Coalition Formation in Multi-Robot Task Allocation
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2412.20397v3](http://arxiv.org/pdf/2412.20397v3)**

> **作者:** Lucas C. D. Bezerra; Ataíde M. G. dos Santos; Shinkyu Park
>
> **摘要:** We propose a decentralized, learning-based framework for dynamic coalition formation in Multi-Robot Task Allocation (MRTA). Our approach extends MAPPO by integrating spatial action maps, robot motion planning, intention sharing, and task allocation revision to enable effective and adaptive coalition formation. Extensive simulation studies confirm the effectiveness of our model, enabling each robot to rely solely on local information to learn timely revisions of task selections and form coalitions with other robots to complete collaborative tasks. The results also highlight the proposed framework's ability to handle large robot populations and adapt to scenarios with diverse task sets.
>
---
#### [replaced 012] GeoFlow-SLAM: A Robust Tightly-Coupled RGBD-Inertial and Legged Odometry Fusion SLAM for Dynamic Legged Robotics
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.14247v2](http://arxiv.org/pdf/2503.14247v2)**

> **作者:** Tingyang Xiao; Xiaolin Zhou; Liu Liu; Wei Sui; Wei Feng; Jiaxiong Qiu; Xinjie Wang; Zhizhong Su
>
> **备注:** 8 pages
>
> **摘要:** This paper presents GeoFlow-SLAM, a robust and effective Tightly-Coupled RGBD-inertial SLAM for legged robotics undergoing aggressive and high-frequency motions.By integrating geometric consistency, legged odometry constraints, and dual-stream optical flow (GeoFlow), our method addresses three critical challenges:feature matching and pose initialization failures during fast locomotion and visual feature scarcity in texture-less scenes.Specifically, in rapid motion scenarios, feature matching is notably enhanced by leveraging dual-stream optical flow, which combines prior map points and poses. Additionally, we propose a robust pose initialization method for fast locomotion and IMU error in legged robots, integrating IMU/Legged odometry, inter-frame Perspective-n-Point (PnP), and Generalized Iterative Closest Point (GICP). Furthermore, a novel optimization framework that tightly couples depth-to-map and GICP geometric constraints is first introduced to improve the robustness and accuracy in long-duration, visually texture-less environments. The proposed algorithms achieve state-of-the-art (SOTA) on collected legged robots and open-source datasets. To further promote research and development, the open-source datasets and code will be made publicly available at https://github.com/HorizonRobotics/geoflow-slam
>
---
#### [replaced 013] Robo-Platform: A Robotic System for Recording Sensors and Controlling Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2409.16595v3](http://arxiv.org/pdf/2409.16595v3)**

> **作者:** Masoud Dayani Najafabadi; Khoshnam Shojaei
>
> **备注:** Project repository: https://github.com/m-dayani/robo-platform Youtube Video: https://youtu.be/BTQ4yLB1bak Dataset: https://drive.google.com/drive/folders/1OZqdA1xa-SyJ64qL_TibqhtwhR1fWWrx?usp=sharing
>
> **摘要:** Mobile smartphones compactly provide sensors such as cameras, IMUs, GNSS measurement units, and wireless and wired communication channels required for robotics projects. They are affordable, portable, and programmable, which makes them ideal for testing, data acquisition, controlling mobile robots, and many other robotic applications. A robotic system is proposed in this paper, consisting of an Android phone, a microcontroller board attached to the phone via USB, and a remote wireless controller station. In the data acquisition mode, the Android device can record a dataset of a diverse configuration of multiple cameras, IMUs, GNSS units, and external USB ADC channels in the rawest format used for, but not limited to, pose estimation and scene reconstruction applications. In robot control mode, the Android phone, a microcontroller board, and other peripherals constitute the mobile or stationary robotic system. This system is controlled using a remote server connected over Wi-Fi or Bluetooth. Experiments show that although the SLAM and AR applications can utilize the acquired data, the proposed system can pave the way for more advanced algorithms for processing these noisy and sporadic measurements. Moreover, the characteristics of the communication media are studied, and two example robotic projects, which involve controlling a toy car and a quadcopter, are included.
>
---
#### [replaced 014] V-Max: A Reinforcement Learning Framework for Autonomous Driving
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.08388v3](http://arxiv.org/pdf/2503.08388v3)**

> **作者:** Valentin Charraut; Waël Doulazmi; Thomas Tournaire; Thibault Buhet
>
> **备注:** RLC 25 - Camera-ready
>
> **摘要:** Learning-based decision-making has the potential to enable generalizable Autonomous Driving (AD) policies, reducing the engineering overhead of rule-based approaches. Imitation Learning (IL) remains the dominant paradigm, benefiting from large-scale human demonstration datasets, but it suffers from inherent limitations such as distribution shift and imitation gaps. Reinforcement Learning (RL) presents a promising alternative, yet its adoption in AD remains limited due to the lack of standardized and efficient research frameworks. To this end, we introduce V-Max, an open research framework providing all the necessary tools to make RL practical for AD. V-Max is built on Waymax, a hardware-accelerated AD simulator designed for large-scale experimentation. We extend it using ScenarioNet's approach, enabling the fast simulation of diverse AD datasets.
>
---
#### [replaced 015] Human Demonstrations are Generalizable Knowledge for Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2312.02419v3](http://arxiv.org/pdf/2312.02419v3)**

> **作者:** Te Cui; Tianxing Zhou; Zicai Peng; Mengxiao Hu; Haoyang Lu; Haizhou Li; Guangyan Chen; Meiling Wang; Yufeng Yue
>
> **备注:** accepted for publication in lEEE/RSJ international Conference on Intelligent Robots and Systems (lROS 2025)
>
> **摘要:** Learning from human demonstrations is an emerging trend for designing intelligent robotic systems. However, previous methods typically regard videos as instructions, simply dividing them into action sequences for robotic repetition, which poses obstacles to generalization to diverse tasks or object instances. In this paper, we propose a different perspective, considering human demonstration videos not as mere instructions, but as a source of knowledge for robots. Motivated by this perspective and the remarkable comprehension and generalization capabilities exhibited by large language models (LLMs), we propose DigKnow, a method that DIstills Generalizable KNOWledge with a hierarchical structure. Specifically, DigKnow begins by converting human demonstration video frames into observation knowledge. This knowledge is then subjected to analysis to extract human action knowledge and further distilled into pattern knowledge compassing task and object instances, resulting in the acquisition of generalizable knowledge with a hierarchical structure. In settings with different tasks or object instances, DigKnow retrieves relevant knowledge for the current task and object instances. Subsequently, the LLM-based planner conducts planning based on the retrieved knowledge, and the policy executes actions in line with the plan to achieve the designated task. Utilizing the retrieved knowledge, we validate and rectify planning and execution outcomes, resulting in a substantial enhancement of the success rate. Experimental results across a range of tasks and scenes demonstrate the effectiveness of this approach in facilitating real-world robots to accomplish tasks with the knowledge derived from human demonstrations.
>
---
#### [replaced 016] Safety-Critical Human-Machine Shared Driving for Vehicle Collision Avoidance based on Hamilton-Jacobi reachability
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2502.10610v2](http://arxiv.org/pdf/2502.10610v2)**

> **作者:** Shiyue Zhao; Junzhi Zhang; Rui Zhou; Neda Masoud; Jianxiong Li; Helai Huang; Shijie Zhao
>
> **备注:** 36 pages, 15 figures, submitted to AAAP
>
> **摘要:** Road safety continues to be a pressing global issue, with vehicle collisions imposing significant human, societal, and economic burdens. Human-machine shared collision avoidance in critical collision scenarios aims to aid drivers' accident avoidance through intervening only when necessary. Existing methods count on replanning collision-free trajectories and imposing human-machine tracking, which usually interrupts the driver's intent and increases the risk of conflict. This paper introduces a Reachability-Aware Reinforcement Learning (RL) framework for shared control, guided by Hamilton-Jacobi (HJ) reachability analysis. Machine intervention is activated only when the vehicle approaches the Collision Avoidance Reachable Set (CARS), which represents states where collision is unavoidable. First, we precompute the reachability distributions and the CARS by solving the Bellman equation using offline data. To reduce human-machine conflicts, we develop a driver model for sudden obstacles and propose an authority allocation strategy considering key collision avoidance features. Finally, we train a RL agent to reduce human-machine conflicts while enforcing the hard constraint of avoiding entry into the CARS. The proposed method was tested on a real vehicle platform. Results show that the controller intervenes effectively near CARS to prevent collisions while maintaining improved original driving task performance. Robustness analysis further supports its flexibility across different driver attributes.
>
---
#### [replaced 017] Wearable Roller Rings to Augment In-Hand Manipulation through Active Surfaces
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2403.13132v4](http://arxiv.org/pdf/2403.13132v4)**

> **作者:** Hayden Webb; Podshara Chanrungmaneekul; Shenli Yuan; Kaiyu Hang
>
> **摘要:** In-hand manipulation is a crucial ability for reorienting and repositioning objects within grasps. The main challenges in this are not only the complexity of the computational models, but also the risks of grasp instability caused by active finger motions, such as rolling, sliding, breaking, and remaking contacts. This paper presents the development of the Roller Ring (RR), a modular robotic attachment with active surfaces that is wearable by both robot and human hands to manipulate without lifting a finger. By installing the angled RRs on hands, such that their spatial motions are not colinear, we derive a general differential motion model for manipulating objects. Our motion model shows that complete in-hand manipulation skill sets can be provided by as few as only 2 RRs through non-holonomic object motions, while more RRs can enable enhanced manipulation dexterity with fewer motion constraints. Through extensive experiments, we test the RRs on both a robot hand and a human hand to evaluate their manipulation capabilities. We show that the RRs can be employed to manipulate arbitrary object shapes to provide dexterous in-hand manipulation.
>
---
#### [replaced 018] DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.04447v2](http://arxiv.org/pdf/2507.04447v2)**

> **作者:** Wenyao Zhang; Hongsi Liu; Zekun Qi; Yunnan Wang; Xinqiang Yu; Jiazhao Zhang; Runpei Dong; Jiawei He; He Wang; Zhizheng Zhang; Li Yi; Wenjun Zeng; Xin Jin
>
> **摘要:** Recent advances in vision-language-action (VLA) models have shown promise in integrating image generation with action prediction to improve generalization and reasoning in robot manipulation. However, existing methods are limited to challenging image-based forecasting, which suffers from redundant information and lacks comprehensive and critical world knowledge, including dynamic, spatial and semantic information. To address these limitations, we propose DreamVLA, a novel VLA framework that integrates comprehensive world knowledge forecasting to enable inverse dynamics modeling, thereby establishing a perception-prediction-action loop for manipulation tasks. Specifically, DreamVLA introduces a dynamic-region-guided world knowledge prediction, integrated with the spatial and semantic cues, which provide compact yet comprehensive representations for action planning. This design aligns with how humans interact with the world by first forming abstract multimodal reasoning chains before acting. To mitigate interference among the dynamic, spatial and semantic information during training, we adopt a block-wise structured attention mechanism that masks their mutual attention, preventing information leakage and keeping each representation clean and disentangled. Moreover, to model the conditional distribution over future actions, we employ a diffusion-based transformer that disentangles action representations from shared latent features. Extensive experiments on both real-world and simulation environments demonstrate that DreamVLA achieves 76.7% success rate on real robot tasks and 4.44 average length on the CALVIN ABC-D benchmarks.
>
---
#### [replaced 019] Out-of-Distribution Recovery with Object-Centric Keypoint Inverse Policy for Visuomotor Imitation Learning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.03294v4](http://arxiv.org/pdf/2411.03294v4)**

> **作者:** George Jiayuan Gao; Tianyu Li; Nadia Figueroa
>
> **备注:** IROS 2025. Project Website: https://sites.google.com/view/ocr-penn
>
> **摘要:** We propose an object-centric recovery (OCR) framework to address the challenges of out-of-distribution (OOD) scenarios in visuomotor policy learning. Previous behavior cloning (BC) methods rely heavily on a large amount of labeled data coverage, failing in unfamiliar spatial states. Without relying on extra data collection, our approach learns a recovery policy constructed by an inverse policy inferred from the object keypoint manifold gradient in the original training data. The recovery policy serves as a simple add-on to any base visuomotor BC policy, agnostic to a specific method, guiding the system back towards the training distribution to ensure task success even in OOD situations. We demonstrate the effectiveness of our object-centric framework in both simulation and real robot experiments, achieving an improvement of 77.7\% over the base policy in OOD. Furthermore, we show OCR's capacity to autonomously collect demonstrations for continual learning. Overall, we believe this framework represents a step toward improving the robustness of visuomotor policies in real-world settings.
>
---
#### [replaced 020] TOP: Trajectory Optimization via Parallel Optimization towards Constant Time Complexity
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.10290v2](http://arxiv.org/pdf/2507.10290v2)**

> **作者:** Jiajun Yu; Nanhe Chen; Guodong Liu; Chao Xu; Fei Gao; Yanjun Cao
>
> **备注:** 8 pages, submitted to RA-L
>
> **摘要:** Optimization has been widely used to generate smooth trajectories for motion planning. However, existing trajectory optimization methods show weakness when dealing with large-scale long trajectories. Recent advances in parallel computing have accelerated optimization in some fields, but how to efficiently solve trajectory optimization via parallelism remains an open question. In this paper, we propose a novel trajectory optimization framework based on the Consensus Alternating Direction Method of Multipliers (CADMM) algorithm, which decomposes the trajectory into multiple segments and solves the subproblems in parallel. The proposed framework reduces the time complexity to O(1) per iteration to the number of segments, compared to O(N) of the state-of-the-art (SOTA) approaches. Furthermore, we introduce a closed-form solution that integrates convex linear and quadratic constraints to speed up the optimization, and we also present numerical solutions for general inequality constraints. A series of simulations and experiments demonstrate that our approach outperforms the SOTA approach in terms of efficiency and smoothness. Especially for a large-scale trajectory, with one hundred segments, achieving over a tenfold speedup. To fully explore the potential of our algorithm on modern parallel computing architectures, we deploy our framework on a GPU and show high performance with thousands of segments.
>
---
