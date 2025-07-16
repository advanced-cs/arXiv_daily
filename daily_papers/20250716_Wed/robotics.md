# 机器人 cs.RO

- **最新发布 39 篇**

- **更新 22 篇**

## 最新发布

#### [new 001] Learning to Move in Rhythm: Task-Conditioned Motion Policies with Orbital Stability Guarantees
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人运动控制任务，解决复杂周期性动作学习与任务迁移问题。通过引入OSMP框架，实现稳定周期运动的学习与多任务泛化。**

- **链接: [http://arxiv.org/pdf/2507.10602v1](http://arxiv.org/pdf/2507.10602v1)**

> **作者:** Maximilian Stölzle; T. Konstantin Rusch; Zach J. Patterson; Rodrigo Pérez-Dattari; Francesco Stella; Josie Hughes; Cosimo Della Santina; Daniela Rus
>
> **备注:** 73 pages
>
> **摘要:** Learning from demonstration provides a sample-efficient approach to acquiring complex behaviors, enabling robots to move robustly, compliantly, and with fluidity. In this context, Dynamic Motion Primitives offer built - in stability and robustness to disturbances but often struggle to capture complex periodic behaviors. Moreover, they are limited in their ability to interpolate between different tasks. These shortcomings substantially narrow their applicability, excluding a wide class of practically meaningful tasks such as locomotion and rhythmic tool use. In this work, we introduce Orbitally Stable Motion Primitives (OSMPs) - a framework that combines a learned diffeomorphic encoder with a supercritical Hopf bifurcation in latent space, enabling the accurate acquisition of periodic motions from demonstrations while ensuring formal guarantees of orbital stability and transverse contraction. Furthermore, by conditioning the bijective encoder on the task, we enable a single learned policy to represent multiple motion objectives, yielding consistent zero-shot generalization to unseen motion objectives within the training distribution. We validate the proposed approach through extensive simulation and real-world experiments across a diverse range of robotic platforms - from collaborative arms and soft manipulators to a bio-inspired rigid-soft turtle robot - demonstrating its versatility and effectiveness in consistently outperforming state-of-the-art baselines such as diffusion policies, among others.
>
---
#### [new 002] Object-Centric Mobile Manipulation through SAM2-Guided Perception and Imitation Learning
- **分类: cs.RO**

- **简介: 该论文属于移动操作任务，解决导航与操作解耦导致的性能问题。通过SAM2引导的感知和模仿学习，提升模型在不同角度下的任务泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.10899v1](http://arxiv.org/pdf/2507.10899v1)**

> **作者:** Wang Zhicheng; Satoshi Yagi; Satoshi Yamamori; Jun Morimoto
>
> **摘要:** Imitation learning for mobile manipulation is a key challenge in the field of robotic manipulation. However, current mobile manipulation frameworks typically decouple navigation and manipulation, executing manipulation only after reaching a certain location. This can lead to performance degradation when navigation is imprecise, especially due to misalignment in approach angles. To enable a mobile manipulator to perform the same task from diverse orientations, an essential capability for building general-purpose robotic models, we propose an object-centric method based on SAM2, a foundation model towards solving promptable visual segmentation in images, which incorporates manipulation orientation information into our model. Our approach enables consistent understanding of the same task from different orientations. We deploy the model on a custom-built mobile manipulator and evaluate it on a pick-and-place task under varied orientation angles. Compared to Action Chunking Transformer, our model maintains superior generalization when trained with demonstrations from varied approach angles. This work significantly enhances the generalization and robustness of imitation learning-based mobile manipulation systems.
>
---
#### [new 003] EquiContact: A Hierarchical SE(3) Vision-to-Force Equivariant Policy for Spatially Generalizable Contact-rich Tasks
- **分类: cs.RO**

- **简介: 该论文针对接触丰富的装配任务（如插销入孔），提出EquiContact框架，解决空间泛化问题。通过层次化策略实现SE(3)等变性，提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.10961v1](http://arxiv.org/pdf/2507.10961v1)**

> **作者:** Joohwan Seo; Arvind Kruthiventy; Soomi Lee; Megan Teng; Xiang Zhang; Seoyeon Choi; Jongeun Choi; Roberto Horowitz
>
> **备注:** Submitted to RA-L
>
> **摘要:** This paper presents a framework for learning vision-based robotic policies for contact-rich manipulation tasks that generalize spatially across task configurations. We focus on achieving robust spatial generalization of the policy for the peg-in-hole (PiH) task trained from a small number of demonstrations. We propose EquiContact, a hierarchical policy composed of a high-level vision planner (Diffusion Equivariant Descriptor Field, Diff-EDF) and a novel low-level compliant visuomotor policy (Geometric Compliant ACT, G-CompACT). G-CompACT operates using only localized observations (geometrically consistent error vectors (GCEV), force-torque readings, and wrist-mounted RGB images) and produces actions defined in the end-effector frame. Through these design choices, we show that the entire EquiContact pipeline is SE(3)-equivariant, from perception to force control. We also outline three key components for spatially generalizable contact-rich policies: compliance, localized policies, and induced equivariance. Real-world experiments on PiH tasks demonstrate a near-perfect success rate and robust generalization to unseen spatial configurations, validating the proposed framework and principles. The experimental videos can be found on the project website: https://sites.google.com/berkeley.edu/equicontact
>
---
#### [new 004] Multi-IMU Sensor Fusion for Legged Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人状态估计任务，解决腿式机器人在复杂环境下姿态和速度估计的误差问题。通过多IMU传感器融合与视觉数据结合，提升定位精度。**

- **链接: [http://arxiv.org/pdf/2507.11447v1](http://arxiv.org/pdf/2507.11447v1)**

> **作者:** Shuo Yang; John Z. Zhang; Ibrahima Sory Sow; Zachary Manchester
>
> **备注:** 16 pages
>
> **摘要:** This paper presents a state-estimation solution for legged robots that uses a set of low-cost, compact, and lightweight sensors to achieve low-drift pose and velocity estimation under challenging locomotion conditions. The key idea is to leverage multiple inertial measurement units on different links of the robot to correct a major error source in standard proprioceptive odometry. We fuse the inertial sensor information and joint encoder measurements in an extended Kalman filter, then combine the velocity estimate from this filter with camera data in a factor-graph-based sliding-window estimator to form a visual-inertial-leg odometry method. We validate our state estimator through comprehensive theoretical analysis and hardware experiments performed using real-world robot data collected during a variety of challenging locomotion tasks. Our algorithm consistently achieves minimal position deviation, even in scenarios involving substantial ground impact, foot slippage, and sudden body rotations. A C++ implementation, along with a large-scale dataset, is available at https://github.com/ShuoYangRobotics/Cerberus2.0.
>
---
#### [new 005] Ocean Diviner: A Diffusion-Augmented Reinforcement Learning for AUV Robust Control in the Underwater Tasks
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于水下自主航行器控制任务，解决轨迹规划与环境适应问题。提出一种融合扩散模型与强化学习的方法，提升AUV在复杂环境中的鲁棒性与适应性。**

- **链接: [http://arxiv.org/pdf/2507.11283v1](http://arxiv.org/pdf/2507.11283v1)**

> **作者:** Weiyi Liu; Jingzehua Xu; Guanwen Xie; Yi Li
>
> **摘要:** This paper presents a diffusion-augmented reinforcement learning (RL) approach for robust autonomous underwater vehicle (AUV) control, addressing key challenges in underwater trajectory planning and dynamic environment adaptation. The proposed method integrates three core innovations: (1) A diffusion-based trajectory generation framework that produces physically feasible multi-step trajectories, enhanced by a high-dimensional state encoding mechanism combining current observations with historical states and actions through a novel diffusion U-Net architecture, significantly improving long-horizon planning. (2) A sample-efficient hybrid learning architecture that synergizes diffusion-guided exploration with RL policy optimization, where the diffusion model generates diverse candidate actions and the RL critic selects optimal actions, achieving higher exploration efficiency and policy stability in dynamic underwater environments. Extensive simulation experiments validating the method's superior robustness and flexibility, outperforms conventional control methods in challenging marine conditions, offering enhanced adaptability and reliability for AUV operations in the underwater tasks.
>
---
#### [new 006] LLM-based ambiguity detection in natural language instructions for collaborative surgical robots
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于自然语言处理任务，旨在解决手术指令中的歧义问题。通过LLM框架检测指令歧义，提升人机协作安全性。**

- **链接: [http://arxiv.org/pdf/2507.11525v1](http://arxiv.org/pdf/2507.11525v1)**

> **作者:** Ana Davila; Jacinto Colan; Yasuhisa Hasegawa
>
> **备注:** Accepted at 2025 IEEE International Conference on Robot and Human Interactive Communication (ROMAN)
>
> **摘要:** Ambiguity in natural language instructions poses significant risks in safety-critical human-robot interaction, particularly in domains such as surgery. To address this, we propose a framework that uses Large Language Models (LLMs) for ambiguity detection specifically designed for collaborative surgical scenarios. Our method employs an ensemble of LLM evaluators, each configured with distinct prompting techniques to identify linguistic, contextual, procedural, and critical ambiguities. A chain-of-thought evaluator is included to systematically analyze instruction structure for potential issues. Individual evaluator assessments are synthesized through conformal prediction, which yields non-conformity scores based on comparison to a labeled calibration dataset. Evaluating Llama 3.2 11B and Gemma 3 12B, we observed classification accuracy exceeding 60% in differentiating ambiguous from unambiguous surgical instructions. Our approach improves the safety and reliability of human-robot collaboration in surgery by offering a mechanism to identify potentially ambiguous instructions before robot action.
>
---
#### [new 007] Robot Drummer: Learning Rhythmic Skills for Humanoid Drumming
- **分类: cs.RO**

- **简介: 该论文属于人形机器人音乐演奏任务，解决高精度鼓点控制与多肢体协调问题。通过强化学习训练机器人完成多样化的鼓谱，实现类似人类的鼓击策略。**

- **链接: [http://arxiv.org/pdf/2507.11498v1](http://arxiv.org/pdf/2507.11498v1)**

> **作者:** Asad Ali Shahid; Francesco Braghin; Loris Roveda
>
> **摘要:** Humanoid robots have seen remarkable advances in dexterity, balance, and locomotion, yet their role in expressive domains, such as music performance, remains largely unexplored. Musical tasks, like drumming, present unique challenges, including split-second timing, rapid contacts, and multi-limb coordination over pieces lasting minutes. In this paper, we introduce Robot Drummer, a humanoid system capable of expressive, high-precision drumming across a diverse repertoire of songs. We formulate humanoid drumming as sequential fulfillment of timed-contacts and transform drum scores in to a Rhythmic Contact Chain. To handle the long-horizon nature of musical performance, we decompose each piece into fixed-length segments and train a single policy across all segments in parallel using reinforcement learning. Through extensive experiments on over thirty popular rock, metal, and jazz tracks, our results demonstrate that Robot Drummer consistently achieves high F1 scores. The learned behaviors exhibit emergent human-like drumming strategies, such as cross-arm strikes, and adaptive sticks assignments, demonstrating the potential of reinforcement learning to bring humanoid robots into the domain of creative musical performance. Project page: \href{https://robot-drummer.github.io}{robot-drummer.github.io}
>
---
#### [new 008] Vision Language Action Models in Robotic Manipulation: A Systematic Review
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决视觉、语言与控制融合问题。通过分析102个VLA模型及相关数据集，提出分类框架与评估标准，推动通用机器人代理的发展。**

- **链接: [http://arxiv.org/pdf/2507.10672v1](http://arxiv.org/pdf/2507.10672v1)**

> **作者:** Muhayy Ud Din; Waseem Akram; Lyes Saad Saoud; Jan Rosell; Irfan Hussain
>
> **备注:** submitted to annual review in control
>
> **摘要:** Vision Language Action (VLA) models represent a transformative shift in robotics, with the aim of unifying visual perception, natural language understanding, and embodied control within a single learning framework. This review presents a comprehensive and forward-looking synthesis of the VLA paradigm, with a particular emphasis on robotic manipulation and instruction-driven autonomy. We comprehensively analyze 102 VLA models, 26 foundational datasets, and 12 simulation platforms that collectively shape the development and evaluation of VLAs models. These models are categorized into key architectural paradigms, each reflecting distinct strategies for integrating vision, language, and control in robotic systems. Foundational datasets are evaluated using a novel criterion based on task complexity, variety of modalities, and dataset scale, allowing a comparative analysis of their suitability for generalist policy learning. We introduce a two-dimensional characterization framework that organizes these datasets based on semantic richness and multimodal alignment, showing underexplored regions in the current data landscape. Simulation environments are evaluated for their effectiveness in generating large-scale data, as well as their ability to facilitate transfer from simulation to real-world settings and the variety of supported tasks. Using both academic and industrial contributions, we recognize ongoing challenges and outline strategic directions such as scalable pretraining protocols, modular architectural design, and robust multimodal alignment strategies. This review serves as both a technical reference and a conceptual roadmap for advancing embodiment and robotic control, providing insights that span from dataset generation to real world deployment of generalist robotic agents.
>
---
#### [new 009] Diffusion-Based Imaginative Coordination for Bimanual Manipulation
- **分类: cs.RO**

- **简介: 该论文属于双臂操作任务，旨在解决高维动作空间和复杂协调问题。提出一种基于扩散的联合视频与动作预测框架，提升操作成功率。**

- **链接: [http://arxiv.org/pdf/2507.11296v1](http://arxiv.org/pdf/2507.11296v1)**

> **作者:** Huilin Xu; Jian Ding; Jiakun Xu; Ruixiang Wang; Jun Chen; Jinjie Mai; Yanwei Fu; Bernard Ghanem; Feng Xu; Mohamed Elhoseiny
>
> **备注:** 15 pages, including 10 figures and 16 tables. Accepted at ICCV 2025
>
> **摘要:** Bimanual manipulation is crucial in robotics, enabling complex tasks in industrial automation and household services. However, it poses significant challenges due to the high-dimensional action space and intricate coordination requirements. While video prediction has been recently studied for representation learning and control, leveraging its ability to capture rich dynamic and behavioral information, its potential for enhancing bimanual coordination remains underexplored. To bridge this gap, we propose a unified diffusion-based framework for the joint optimization of video and action prediction. Specifically, we propose a multi-frame latent prediction strategy that encodes future states in a compressed latent space, preserving task-relevant features. Furthermore, we introduce a unidirectional attention mechanism where video prediction is conditioned on the action, while action prediction remains independent of video prediction. This design allows us to omit video prediction during inference, significantly enhancing efficiency. Experiments on two simulated benchmarks and a real-world setting demonstrate a significant improvement in the success rate over the strong baseline ACT using our method, achieving a \textbf{24.9\%} increase on ALOHA, an \textbf{11.1\%} increase on RoboTwin, and a \textbf{32.5\%} increase in real-world experiments. Our models and code are publicly available at https://github.com/return-sleep/Diffusion_based_imaginative_Coordination.
>
---
#### [new 010] Uncertainty Aware Mapping for Vision-Based Underwater Robots
- **分类: cs.RO**

- **简介: 该论文属于水下机器人视觉导航任务，解决环境建模中的不确定性问题。通过融合深度估计置信度，改进Voxblox框架，提升地图准确性。**

- **链接: [http://arxiv.org/pdf/2507.10991v1](http://arxiv.org/pdf/2507.10991v1)**

> **作者:** Abhimanyu Bhowmik; Mohit Singh; Madhushree Sannigrahi; Martin Ludvigsen; Kostas Alexis
>
> **备注:** Presented at the 2025 IEEE ICRA Workshop on Field Robotics
>
> **摘要:** Vision-based underwater robots can be useful in inspecting and exploring confined spaces where traditional sensors and preplanned paths cannot be followed. Sensor noise and situational change can cause significant uncertainty in environmental representation. Thus, this paper explores how to represent mapping inconsistency in vision-based sensing and incorporate depth estimation confidence into the mapping framework. The scene depth and the confidence are estimated using the RAFT-Stereo model and are integrated into a voxel-based mapping framework, Voxblox. Improvements in the existing Voxblox weight calculation and update mechanism are also proposed. Finally, a qualitative analysis of the proposed method is performed in a confined pool and in a pier in the Trondheim fjord. Experiments using an underwater robot demonstrated the change in uncertainty in the visualization.
>
---
#### [new 011] MPC-based Coarse-to-Fine Motion Planning for Robotic Object Transportation in Cluttered Environments
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人路径规划任务，解决 cluttered 环境下的物体搬运问题。通过融合视觉信息与MPC方法，实现动态、鲁棒的运动规划。**

- **链接: [http://arxiv.org/pdf/2507.11211v1](http://arxiv.org/pdf/2507.11211v1)**

> **作者:** Chen Cai; Ernesto Dickel Saraiva; Ya-jun Pan; Steven Liu
>
> **备注:** 10 pages, 5 figures, submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** This letter presents a novel coarse-to-fine motion planning framework for robotic manipulation in cluttered, unmodeled environments. The system integrates a dual-camera perception setup with a B-spline-based model predictive control (MPC) scheme. Initially, the planner generates feasible global trajectories from partial and uncertain observations. As new visual data are incrementally fused, both the environment model and motion planning are progressively refined. A vision-based cost function promotes target-driven exploration, while a refined kernel-perceptron collision detector enables efficient constraint updates for real-time planning. The framework accommodates closed-chain kinematics and supports dynamic replanning. Experiments on a multi-arm platform validate its robustness and adaptability under uncertainties and clutter.
>
---
#### [new 012] TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于透明物体深度重建任务，解决从稀疏视角RGB图像中准确恢复透明物体3D结构的问题。通过结合2D高斯点云与物理模拟，提升重建精度与动态场景适应性。**

- **链接: [http://arxiv.org/pdf/2507.11069v1](http://arxiv.org/pdf/2507.11069v1)**

> **作者:** Jeongyun Kim; Seunghoon Jeong; Giseop Kim; Myung-Hwan Jeon; Eunji Jun; Ayoung Kim
>
> **摘要:** Understanding the 3D geometry of transparent objects from RGB images is challenging due to their inherent physical properties, such as reflection and refraction. To address these difficulties, especially in scenarios with sparse views and dynamic environments, we introduce TRAN-D, a novel 2D Gaussian Splatting-based depth reconstruction method for transparent objects. Our key insight lies in separating transparent objects from the background, enabling focused optimization of Gaussians corresponding to the object. We mitigate artifacts with an object-aware loss that places Gaussians in obscured regions, ensuring coverage of invisible surfaces while reducing overfitting. Furthermore, we incorporate a physics-based simulation that refines the reconstruction in just a few seconds, effectively handling object removal and chain-reaction movement of remaining objects without the need for rescanning. TRAN-D is evaluated on both synthetic and real-world sequences, and it consistently demonstrated robust improvements over existing GS-based state-of-the-art methods. In comparison with baselines, TRAN-D reduces the mean absolute error by over 39% for the synthetic TRansPose sequences. Furthermore, despite being updated using only one image, TRAN-D reaches a {\delta} < 2.5 cm accuracy of 48.46%, over 1.5 times that of baselines, which uses six images. Code and more results are available at https://jeongyun0609.github.io/TRAN-D/.
>
---
#### [new 013] Enhancing Autonomous Manipulator Control with Human-in-loop for Uncertain Assembly Environments
- **分类: cs.RO**

- **简介: 该论文属于空间机器人任务，旨在提升月球环境下自主操作的可靠性。通过人机协同控制，解决不确定环境中的部署难题，实现精准柔性太阳能板展开。**

- **链接: [http://arxiv.org/pdf/2507.11006v1](http://arxiv.org/pdf/2507.11006v1)**

> **作者:** Ashutosh Mishra; Shreya Santra; Hazal Gozbasi; Kentaro Uno; Kazuya Yoshida
>
> **备注:** 6 pages, 7 figures. Manuscript accepted at the 2025 IEEE 21st International Conference on Automation Science and Engineering (CASE 2025)
>
> **摘要:** This study presents an advanced approach to enhance robotic manipulation in uncertain and challenging environments, with a focus on autonomous operations augmented by human-in-the-loop (HITL) control for lunar missions. By integrating human decision-making with autonomous robotic functions, the research improves task reliability and efficiency for space applications. The key task addressed is the autonomous deployment of flexible solar panels using an extendable ladder-like structure and a robotic manipulator with real-time feedback for precision. The manipulator relays position and force-torque data, enabling dynamic error detection and adaptive control during deployment. To mitigate the effects of sinkage, variable payload, and low-lighting conditions, efficient motion planning strategies are employed, supplemented by human control that allows operators to intervene in ambiguous scenarios. Digital twin simulation enhances system robustness by enabling continuous feedback, iterative task refinement, and seamless integration with the deployment pipeline. The system has been tested to validate its performance in simulated lunar conditions and ensure reliability in extreme lighting, variable terrain, changing payloads, and sensor limitations.
>
---
#### [new 014] Fast Non-Episodic Adaptive Tuning of Robot Controllers with Online Policy Optimization
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人控制任务，解决动态环境下控制器参数实时调整问题。提出M-GAPS算法，在单轨迹下实现快速策略优化，提升适应性和效率。**

- **链接: [http://arxiv.org/pdf/2507.10914v1](http://arxiv.org/pdf/2507.10914v1)**

> **作者:** James A. Preiss; Fengze Xie; Yiheng Lin; Adam Wierman; Yisong Yue
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** We study online algorithms to tune the parameters of a robot controller in a setting where the dynamics, policy class, and optimality objective are all time-varying. The system follows a single trajectory without episodes or state resets, and the time-varying information is not known in advance. Focusing on nonlinear geometric quadrotor controllers as a test case, we propose a practical implementation of a single-trajectory model-based online policy optimization algorithm, M-GAPS,along with reparameterizations of the quadrotor state space and policy class to improve the optimization landscape. In hardware experiments,we compare to model-based and model-free baselines that impose artificial episodes. We show that M-GAPS finds near-optimal parameters more quickly, especially when the episode length is not favorable. We also show that M-GAPS rapidly adapts to heavy unmodeled wind and payload disturbances, and achieves similar strong improvement on a 1:6-scale Ackermann-steered car. Our results demonstrate the hardware practicality of this emerging class of online policy optimization that offers significantly more flexibility than classic adaptive control, while being more stable and data-efficient than model-free reinforcement learning.
>
---
#### [new 015] Acting and Planning with Hierarchical Operational Models on a Mobile Robot: A Study with RAE+UPOM
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人任务执行领域，解决符号规划与实际控制不一致的问题，通过集成RAE+UPOM系统实现动作与规划的协同决策。**

- **链接: [http://arxiv.org/pdf/2507.11345v1](http://arxiv.org/pdf/2507.11345v1)**

> **作者:** Oscar Lima; Marc Vinci; Sunandita Patra; Sebastian Stock; Joachim Hertzberg; Martin Atzmueller; Malik Ghallab; Dana Nau; Paolo Traverso
>
> **备注:** Accepted in ECMR 2025 conference
>
> **摘要:** Robotic task execution faces challenges due to the inconsistency between symbolic planner models and the rich control structures actually running on the robot. In this paper, we present the first physical deployment of an integrated actor-planner system that shares hierarchical operational models for both acting and planning, interleaving the Reactive Acting Engine (RAE) with an anytime UCT-like Monte Carlo planner (UPOM). We implement RAE+UPOM on a mobile manipulator in a real-world deployment for an object collection task. Our experiments demonstrate robust task execution under action failures and sensor noise, and provide empirical insights into the interleaved acting-and-planning decision making process.
>
---
#### [new 016] All Eyes, no IMU: Learning Flight Attitude from Vision Alone
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于飞行控制任务，解决无惯性测量单元（IMU）的飞行姿态估计问题。通过视觉仅方法实现无人机姿态控制，使用事件相机和神经网络进行实时飞行调整。**

- **链接: [http://arxiv.org/pdf/2507.11302v1](http://arxiv.org/pdf/2507.11302v1)**

> **作者:** Jesse J. Hagenaars; Stein Stroobants; Sander M. Bohte; Guido C. H. E. De Croon
>
> **摘要:** Vision is an essential part of attitude control for many flying animals, some of which have no dedicated sense of gravity. Flying robots, on the other hand, typically depend heavily on accelerometers and gyroscopes for attitude stabilization. In this work, we present the first vision-only approach to flight control for use in generic environments. We show that a quadrotor drone equipped with a downward-facing event camera can estimate its attitude and rotation rate from just the event stream, enabling flight control without inertial sensors. Our approach uses a small recurrent convolutional neural network trained through supervised learning. Real-world flight tests demonstrate that our combination of event camera and low-latency neural network is capable of replacing the inertial measurement unit in a traditional flight control loop. Furthermore, we investigate the network's generalization across different environments, and the impact of memory and different fields of view. While networks with memory and access to horizon-like visual cues achieve best performance, variants with a narrower field of view achieve better relative generalization. Our work showcases vision-only flight control as a promising candidate for enabling autonomous, insect-scale flying robots.
>
---
#### [new 017] Mixed Discrete and Continuous Planning using Shortest Walks in Graphs of Convex Sets
- **分类: cs.RO**

- **简介: 该论文研究混合离散与连续规划问题，提出在凸集图中求最短路径的算法，用于机器人运动规划等任务，提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2507.10878v1](http://arxiv.org/pdf/2507.10878v1)**

> **作者:** Savva Morozov; Tobia Marcucci; Bernhard Paus Graesdal; Alexandre Amice; Pablo A. Parrilo; Russ Tedrake
>
> **备注:** 10 pages
>
> **摘要:** We study the Shortest-Walk Problem (SWP) in a Graph of Convex Sets (GCS). A GCS is a graph where each vertex is paired with a convex program, and each edge couples adjacent programs via additional costs and constraints. A walk in a GCS is a sequence of vertices connected by edges, where vertices may be repeated. The length of a walk is given by the cumulative optimal value of the corresponding convex programs. To solve the SWP in GCS, we first synthesize a piecewise-quadratic lower bound on the problem's cost-to-go function using semidefinite programming. Then we use this lower bound to guide an incremental-search algorithm that yields an approximate shortest walk. We show that the SWP in GCS is a natural language for many mixed discrete-continuous planning problems in robotics, unifying problems that typically require specialized solutions while delivering high performance and computational efficiency. We demonstrate this through experiments in collision-free motion planning, skill chaining, and optimal control of hybrid systems.
>
---
#### [new 018] SMART-Merge Planner: A Safe Merging and Real-Time Motion Planner for Autonomous Highway On-Ramp Merging
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于自动驾驶高速公路汇入任务，解决安全高效汇入问题。提出SMART-Merge规划器，通过优化成本函数和速度策略，实现快速可靠汇入。**

- **链接: [http://arxiv.org/pdf/2507.10968v1](http://arxiv.org/pdf/2507.10968v1)**

> **作者:** Toktam Mohammadnejad; Jovin D'sa; Behdad Chalaki; Hossein Nourkhiz Mahjoub; Ehsan Moradi-Pari
>
> **备注:** Accepted at IEEE ITSC 2025
>
> **摘要:** Merging onto a highway is a complex driving task that requires identifying a safe gap, adjusting speed, often interactions to create a merging gap, and completing the merge maneuver within a limited time window while maintaining safety and driving comfort. In this paper, we introduce a Safe Merging and Real-Time Merge (SMART-Merge) planner, a lattice-based motion planner designed to facilitate safe and comfortable forced merging. By deliberately adapting cost terms to the unique challenges of forced merging and introducing a desired speed heuristic, SMART-Merge planner enables the ego vehicle to merge successfully while minimizing the merge time. We verify the efficiency and effectiveness of the proposed merge planner through high-fidelity CarMaker simulations on hundreds of highway merge scenarios. Our proposed planner achieves the success rate of 100% as well as completes the merge maneuver in the shortest amount of time compared with the baselines, demonstrating our planner's capability to handle complex forced merge tasks and provide a reliable and robust solution for autonomous highway merge. The simulation result videos are available at https://sites.google.com/view/smart-merge-planner/home.
>
---
#### [new 019] Versatile and Generalizable Manipulation via Goal-Conditioned Reinforcement Learning with Grounded Object Detection
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在提升机器人在不同场景下的抓取能力。通过结合目标条件强化学习与物体检测模型，实现更通用和高效的抓取策略。**

- **链接: [http://arxiv.org/pdf/2507.10814v1](http://arxiv.org/pdf/2507.10814v1)**

> **作者:** Huiyi Wang; Fahim Shahriar; Alireza Azimi; Gautham Vasan; Rupam Mahmood; Colin Bellinger
>
> **备注:** 8 pages, 4 figures, 3 tables
>
> **摘要:** General-purpose robotic manipulation, including reach and grasp, is essential for deployment into households and workspaces involving diverse and evolving tasks. Recent advances propose using large pre-trained models, such as Large Language Models and object detectors, to boost robotic perception in reinforcement learning. These models, trained on large datasets via self-supervised learning, can process text prompts and identify diverse objects in scenes, an invaluable skill in RL where learning object interaction is resource-intensive. This study demonstrates how to integrate such models into Goal-Conditioned Reinforcement Learning to enable general and versatile robotic reach and grasp capabilities. We use a pre-trained object detection model to enable the agent to identify the object from a text prompt and generate a mask for goal conditioning. Mask-based goal conditioning provides object-agnostic cues, improving feature sharing and generalization. The effectiveness of the proposed framework is demonstrated in a simulated reach-and-grasp task, where the mask-based goal conditioning consistently maintains a $\sim$90\% success rate in grasping both in and out-of-distribution objects, while also ensuring faster convergence to higher returns.
>
---
#### [new 020] Development of an Autonomous Mobile Robotic System for Efficient and Precise Disinfection
- **分类: cs.RO**

- **简介: 该论文属于医疗自动化任务，旨在解决医院消毒效率低和资源不足的问题。提出一种自主机器人系统，针对病毒高发区进行高效紫外消毒，提升效果并缩短时间。**

- **链接: [http://arxiv.org/pdf/2507.11270v1](http://arxiv.org/pdf/2507.11270v1)**

> **作者:** Ting-Wei Ou; Jia-Hao Jiang; Guan-Lin Huang; Kuu-Young Young
>
> **备注:** Accepted to the IEEE International Conference on Systems, Man, and Cybernetics (SMC) 2025
>
> **摘要:** The COVID-19 pandemic has severely affected public health, healthcare systems, and daily life, especially amid resource shortages and limited workers. This crisis has underscored the urgent need for automation in hospital environments, particularly disinfection, which is crucial to controlling virus transmission and improving the safety of healthcare personnel and patients. Ultraviolet (UV) light disinfection, known for its high efficiency, has been widely adopted in hospital settings. However, most existing research focuses on maximizing UV coverage while paying little attention to the impact of human activity on virus distribution. To address this issue, we propose a mobile robotic system for UV disinfection focusing on the virus hotspot. The system prioritizes disinfection in high-risk areas and employs an approach for optimized UV dosage to ensure that all surfaces receive an adequate level of UV exposure while significantly reducing disinfection time. It not only improves disinfection efficiency but also minimizes unnecessary exposure in low-risk areas. In two representative hospital scenarios, our method achieves the same disinfection effectiveness while reducing disinfection time by 30.7% and 31.9%, respectively. The video of the experiment is available at: https://youtu.be/wHcWzOcoMPM.
>
---
#### [new 021] Human-Robot collaboration in surgery: Advances and challenges towards autonomous surgical assistants
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于医疗机器人领域，旨在解决人机协作手术中的技术挑战。通过系统综述分析了自主手术机器人的发展现状与问题，提出了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.11460v1](http://arxiv.org/pdf/2507.11460v1)**

> **作者:** Jacinto Colan; Ana Davila; Yutaro Yamada; Yasuhisa Hasegawa
>
> **备注:** Accepted at 2025 IEEE International Conference on Robot and Human Interactive Communication (ROMAN)
>
> **摘要:** Human-robot collaboration in surgery represents a significant area of research, driven by the increasing capability of autonomous robotic systems to assist surgeons in complex procedures. This systematic review examines the advancements and persistent challenges in the development of autonomous surgical robotic assistants (ASARs), focusing specifically on scenarios where robots provide meaningful and active support to human surgeons. Adhering to the PRISMA guidelines, a comprehensive literature search was conducted across the IEEE Xplore, Scopus, and Web of Science databases, resulting in the selection of 32 studies for detailed analysis. Two primary collaborative setups were identified: teleoperation-based assistance and direct hands-on interaction. The findings reveal a growing research emphasis on ASARs, with predominant applications currently in endoscope guidance, alongside emerging progress in autonomous tool manipulation. Several key challenges hinder wider adoption, including the alignment of robotic actions with human surgeon preferences, the necessity for procedural awareness within autonomous systems, the establishment of seamless human-robot information exchange, and the complexities of skill acquisition in shared workspaces. This review synthesizes current trends, identifies critical limitations, and outlines future research directions essential to improve the reliability, safety, and effectiveness of human-robot collaboration in surgical environments.
>
---
#### [new 022] Whom to Respond To? A Transformer-Based Model for Multi-Party Social Robot Interaction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多用户人机交互任务，解决社交机器人在多人场景中何时及如何回应的问题。提出基于Transformer的多任务框架，并设计新损失函数与数据集以提升决策效果。**

- **链接: [http://arxiv.org/pdf/2507.10960v1](http://arxiv.org/pdf/2507.10960v1)**

> **作者:** He Zhu; Ryo Miyoshi; Yuki Okafuji
>
> **摘要:** Prior human-robot interaction (HRI) research has primarily focused on single-user interactions, where robots do not need to consider the timing or recipient of their responses. However, in multi-party interactions, such as at malls and hospitals, social robots must understand the context and decide both when and to whom they should respond. In this paper, we propose a Transformer-based multi-task learning framework to improve the decision-making process of social robots, particularly in multi-user environments. Considering the characteristics of HRI, we propose two novel loss functions: one that enforces constraints on active speakers to improve scene modeling, and another that guides response selection towards utterances specifically directed at the robot. Additionally, we construct a novel multi-party HRI dataset that captures real-world complexities, such as gaze misalignment. Experimental results demonstrate that our model achieves state-of-the-art performance in respond decisions, outperforming existing heuristic-based and single-task approaches. Our findings contribute to the development of socially intelligent social robots capable of engaging in natural and context-aware multi-party interactions.
>
---
#### [new 023] LF: Online Multi-Robot Path Planning Meets Optimal Trajectory Control
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人路径规划任务，解决动态环境下的协同导航问题。工作包括设计异步协同框架LF，结合快速路径规划与轨迹控制，实现高效、可靠的多机器人运动。**

- **链接: [http://arxiv.org/pdf/2507.11464v1](http://arxiv.org/pdf/2507.11464v1)**

> **作者:** Ajay Shankar; Keisuke Okumura; Amanda Prorok
>
> **备注:** 9 pages; under review for IEEE Robotics & Automation - Letters (RA-L)
>
> **摘要:** We propose a multi-robot control paradigm to solve point-to-point navigation tasks for a team of holonomic robots with access to the full environment information. The framework invokes two processes asynchronously at high frequency: (i) a centralized, discrete, and full-horizon planner for computing collision- and deadlock-free paths rapidly, leveraging recent advances in multi-agent pathfinding (MAPF), and (ii) dynamics-aware, robot-wise optimal trajectory controllers that ensure all robots independently follow their assigned paths reliably. This hierarchical shift in planning representation from (i) discrete and coupled to (ii) continuous and decoupled domains enables the framework to maintain long-term scalable motion synthesis. As an instantiation of this idea, we present LF, which combines a fast state-of-the-art MAPF solver (LaCAM), and a robust feedback control stack (Freyja) for executing agile robot maneuvers. LF provides a robust and versatile mechanism for lifelong multi-robot navigation even under asynchronous and partial goal updates, and adapts to dynamic workspaces simply by quick replanning. We present various multirotor and ground robot demonstrations, including the deployment of 15 real multirotors with random, consecutive target updates while a person walks through the operational workspace.
>
---
#### [new 024] ILCL: Inverse Logic-Constraint Learning from Temporally Constrained Demonstrations
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决从示范中学习时间约束的问题。通过引入ILCL方法，结合遗传算法和逻辑约束强化学习，有效学习并迁移时间逻辑约束。**

- **链接: [http://arxiv.org/pdf/2507.11000v1](http://arxiv.org/pdf/2507.11000v1)**

> **作者:** Minwoo Cho; Jaehwi Jang; Daehyung Park
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** We aim to solve the problem of temporal-constraint learning from demonstrations to reproduce demonstration-like logic-constrained behaviors. Learning logic constraints is challenging due to the combinatorially large space of possible specifications and the ill-posed nature of non-Markovian constraints. To figure it out, we introduce a novel temporal-constraint learning method, which we call inverse logic-constraint learning (ILCL). Our method frames ICL as a two-player zero-sum game between 1) a genetic algorithm-based temporal-logic mining (GA-TL-Mining) and 2) logic-constrained reinforcement learning (Logic-CRL). GA-TL-Mining efficiently constructs syntax trees for parameterized truncated linear temporal logic (TLTL) without predefined templates. Subsequently, Logic-CRL finds a policy that maximizes task rewards under the constructed TLTL constraints via a novel constraint redistribution scheme. Our evaluations show ILCL outperforms state-of-the-art baselines in learning and transferring TL constraints on four temporally constrained tasks. We also demonstrate successful transfer to real-world peg-in-shallow-hole tasks.
>
---
#### [new 025] rt-RISeg: Real-Time Model-Free Robot Interactive Segmentation for Active Instance-Level Object Understanding
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于未知物体实例分割任务，解决传统方法依赖大规模数据导致泛化能力差的问题。提出rt-RISeg框架，通过机器人交互实时生成分割掩码，无需预训练模型。**

- **链接: [http://arxiv.org/pdf/2507.10776v1](http://arxiv.org/pdf/2507.10776v1)**

> **作者:** Howard H. Qian; Yiting Chen; Gaotian Wang; Podshara Chanrungmaneekul; Kaiyu Hang
>
> **备注:** 8 pages, IROS 2025, Interactive Perception, Segmentation, Robotics, Computer Vision
>
> **摘要:** Successful execution of dexterous robotic manipulation tasks in new environments, such as grasping, depends on the ability to proficiently segment unseen objects from the background and other objects. Previous works in unseen object instance segmentation (UOIS) train models on large-scale datasets, which often leads to overfitting on static visual features. This dependency results in poor generalization performance when confronted with out-of-distribution scenarios. To address this limitation, we rethink the task of UOIS based on the principle that vision is inherently interactive and occurs over time. We propose a novel real-time interactive perception framework, rt-RISeg, that continuously segments unseen objects by robot interactions and analysis of a designed body frame-invariant feature (BFIF). We demonstrate that the relative rotational and linear velocities of randomly sampled body frames, resulting from selected robot interactions, can be used to identify objects without any learned segmentation model. This fully self-contained segmentation pipeline generates and updates object segmentation masks throughout each robot interaction without the need to wait for an action to finish. We showcase the effectiveness of our proposed interactive perception method by achieving an average object segmentation accuracy rate 27.5% greater than state-of-the-art UOIS methods. Furthermore, although rt-RISeg is a standalone framework, we show that the autonomously generated segmentation masks can be used as prompts to vision foundation models for significantly improved performance.
>
---
#### [new 026] Force-Based Viscosity and Elasticity Measurements for Material Biomechanical Characterisation with a Collaborative Robotic Arm
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决人体组织力学特性测量问题。通过协作机械臂进行粘弹性参数估计，验证其准确性与临床应用潜力。**

- **链接: [http://arxiv.org/pdf/2507.11133v1](http://arxiv.org/pdf/2507.11133v1)**

> **作者:** Luca Beber; Edoardo Lamon; Giacomo Moretti; Matteo Saveriano; Luca Fambri; Luigi Palopoli; Daniele Fontanelli
>
> **摘要:** Diagnostic activities, such as ultrasound scans and palpation, are relatively low-cost. They play a crucial role in the early detection of health problems and in assessing their progression. However, they are also error-prone activities, which require highly skilled medical staff. The use of robotic solutions can be key to decreasing the inherent subjectivity of the results and reducing the waiting list. For a robot to perform palpation or ultrasound scans, it must effectively manage physical interactions with the human body, which greatly benefits from precise estimation of the patient's tissue biomechanical properties. This paper assesses the accuracy and precision of a robotic system in estimating the viscoelastic parameters of various materials, including some tests on ex vivo tissues as a preliminary proof-of-concept demonstration of the method's applicability to biological samples. The measurements are compared against a ground truth derived from silicone specimens with different viscoelastic properties, characterised using a high-precision instrument. Experimental results show that the robotic system's accuracy closely matches the ground truth, increasing confidence in the potential use of robots for such clinical applications.
>
---
#### [new 027] Closed Form Time Derivatives of the Equations of Motion of Rigid Body Systems
- **分类: cs.RO; cs.NA; math.DG; math.DS; math.GR; math.NA**

- **简介: 该论文属于机器人动力学领域，解决刚体系统运动方程时间导数的计算问题，提出了一种闭式解法，提供更直观的结构分析。**

- **链接: [http://arxiv.org/pdf/2507.11076v1](http://arxiv.org/pdf/2507.11076v1)**

> **作者:** Andreas Mueller; Shivesh Kumar
>
> **摘要:** Derivatives of equations of motion(EOM) describing the dynamics of rigid body systems are becoming increasingly relevant for the robotics community and find many applications in design and control of robotic systems. Controlling robots, and multibody systems comprising elastic components in particular, not only requires smooth trajectories but also the time derivatives of the control forces/torques, hence of the EOM. This paper presents the time derivatives of the EOM in closed form up to second-order as an alternative formulation to the existing recursive algorithms for this purpose, which provides a direct insight into the structure of the derivatives. The Lie group formulation for rigid body systems is used giving rise to very compact and easily parameterized equations.
>
---
#### [new 028] From Production Logistics to Smart Manufacturing: The Vision for a New RoboCup Industrial League
- **分类: cs.RO**

- **简介: 该论文属于工业机器人竞赛领域，旨在解决传统物流竞赛与现代智能制造脱节的问题。提出新竞赛框架，整合多机器人任务，提升相关性与挑战性。**

- **链接: [http://arxiv.org/pdf/2507.11402v1](http://arxiv.org/pdf/2507.11402v1)**

> **作者:** Supun Dissanayaka; Alexander Ferrein; Till Hofmann; Kosuke Nakajima; Mario Sanz-Lopez; Jesus Savage; Daniel Swoboda; Matteo Tschesche; Wataru Uemura; Tarik Viehmann; Shohei Yasuda
>
> **备注:** RoboCup Symposium 2025
>
> **摘要:** The RoboCup Logistics League is a RoboCup competition in a smart factory scenario that has focused on task planning, job scheduling, and multi-agent coordination. The focus on production logistics allowed teams to develop highly competitive strategies, but also meant that some recent developments in the context of smart manufacturing are not reflected in the competition, weakening its relevance over the years. In this paper, we describe the vision for the RoboCup Smart Manufacturing League, a new competition designed as a larger smart manufacturing scenario, reflecting all the major aspects of a modern factory. It will consist of several tracks that are initially independent but gradually combined into one smart manufacturing scenario. The new tracks will cover industrial robotics challenges such as assembly, human-robot collaboration, and humanoid robotics, but also retain a focus on production logistics. We expect the reenvisioned competition to be more attractive to newcomers and well-tried teams, while also shifting the focus to current and future challenges of industrial robotics.
>
---
#### [new 029] Learning to Tune Like an Expert: Interpretable and Scene-Aware Navigation via MLLM Reasoning and CVAE-Based Adaptation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于服务机器人导航任务，解决动态环境中导航系统泛化能力差的问题。通过多模态大语言模型和条件变分自编码器，实现可解释的场景感知导航与参数调优。**

- **链接: [http://arxiv.org/pdf/2507.11001v1](http://arxiv.org/pdf/2507.11001v1)**

> **作者:** Yanbo Wang; Zipeng Fang; Lei Zhao; Weidong Chen
>
> **摘要:** Service robots are increasingly deployed in diverse and dynamic environments, where both physical layouts and social contexts change over time and across locations. In these unstructured settings, conventional navigation systems that rely on fixed parameters often fail to generalize across scenarios, resulting in degraded performance and reduced social acceptance. Although recent approaches have leveraged reinforcement learning to enhance traditional planners, these methods often fail in real-world deployments due to poor generalization and limited simulation diversity, which hampers effective sim-to-real transfer. To tackle these issues, we present LE-Nav, an interpretable and scene-aware navigation framework that leverages multi-modal large language model reasoning and conditional variational autoencoders to adaptively tune planner hyperparameters. To achieve zero-shot scene understanding, we utilize one-shot exemplars and chain-of-thought prompting strategies. Additionally, a conditional variational autoencoder captures the mapping between natural language instructions and navigation hyperparameters, enabling expert-level tuning. Experiments show that LE-Nav can generate hyperparameters achieving human-level tuning across diverse planners and scenarios. Real-world navigation trials and a user study on a smart wheelchair platform demonstrate that it outperforms state-of-the-art methods on quantitative metrics such as success rate, efficiency, safety, and comfort, while receiving higher subjective scores for perceived safety and social acceptance. Code is available at https://github.com/Cavendish518/LE-Nav.
>
---
#### [new 030] RCG: Safety-Critical Scenario Generation for Robust Autonomous Driving via Real-World Crash Grounding
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶安全测试任务，旨在解决真实事故场景稀缺问题。通过整合碰撞语义的对抗扰动方法生成高风险且行为真实的测试场景。**

- **链接: [http://arxiv.org/pdf/2507.10749v1](http://arxiv.org/pdf/2507.10749v1)**

> **作者:** Benjamin Stoler; Juliet Yang; Jonathan Francis; Jean Oh
>
> **摘要:** Safety-critical scenarios are essential for training and evaluating autonomous driving (AD) systems, yet remain extremely rare in real-world driving datasets. To address this, we propose Real-world Crash Grounding (RCG), a scenario generation framework that integrates crash-informed semantics into adversarial perturbation pipelines. We construct a safety-aware behavior representation through contrastive pre-training on large-scale driving logs, followed by fine-tuning on a small, crash-rich dataset with approximate trajectory annotations extracted from video. This embedding captures semantic structure aligned with real-world accident behaviors and supports selection of adversary trajectories that are both high-risk and behaviorally realistic. We incorporate the resulting selection mechanism into two prior scenario generation pipelines, replacing their handcrafted scoring objectives with an embedding-based criterion. Experimental results show that ego agents trained against these generated scenarios achieve consistently higher downstream success rates, with an average improvement of 9.2% across seven evaluation settings. Qualitative and quantitative analyses further demonstrate that our approach produces more plausible and nuanced adversary behaviors, enabling more effective and realistic stress testing of AD systems. Code and tools will be released publicly.
>
---
#### [new 031] Unified Modeling and Structural Optimization of Multi-magnet Embedded Soft Continuum Robots for Enhanced Kinematic Performances
- **分类: cs.RO**

- **简介: 该论文属于软体机器人领域，旨在提升多磁铁嵌入式软连续机器人的运动性能。通过建立统一模型和优化框架，解决其运动学控制问题。**

- **链接: [http://arxiv.org/pdf/2507.10950v1](http://arxiv.org/pdf/2507.10950v1)**

> **作者:** Zhiwei Wu; Jiahao Luo; Siyi Wei; Jinhui Zhang
>
> **摘要:** This paper presents a unified modeling and optimization framework to enhance the kinematic performance of multi-magnet embedded soft continuum robots (MeSCRs). To this end, we establish a differentiable system formulation based on an extended pseudo-rigid-body model. This formulation enables analysis of the equilibrium well-posedness and the geometry of the induced configuration under magnetic actuation. In particular, we show that the maximum controllable degrees of freedom of a MeSCR equal twice the number of embedded magnets. We subsequently develop a structural optimization framework based on differential geometry that links classical kinematic measures (e.g., manipulability and dexterity) to the configuration of embedded magnets. The resulting optimization condition reveals that improving local performance requires structurally modulating the spectrum of the configuration space metric to counteract its distortion. Closed-form solutions for optimal magnet configurations are derived under representative conditions, and a gradient-based numerical method is proposed for general design scenarios. Simulation studies validate the effectiveness of the proposed framework.
>
---
#### [new 032] A Robust Controller based on Gaussian Processes for Robotic Manipulators with Unknown Uncertainty
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人控制任务，解决未知不确定下的轨迹跟踪问题。通过结合高斯过程回归与反馈线性化，设计鲁棒控制器以提高跟踪精度。**

- **链接: [http://arxiv.org/pdf/2507.11170v1](http://arxiv.org/pdf/2507.11170v1)**

> **作者:** Giulio Giacomuzzo; Mohamed Abdelwahab; Marco Calì; Alberto Dalla Libera; Ruggero Carli
>
> **摘要:** In this paper, we propose a novel learning-based robust feedback linearization strategy to ensure precise trajectory tracking for an important family of Lagrangian systems. We assume a nominal knowledge of the dynamics is given but no a-priori bounds on the model mismatch are available. In our approach, the key ingredient is the adoption of a regression framework based on Gaussian Processes (GPR) to estimate the model mismatch. This estimate is added to the outer loop of a classical feedback linearization scheme based on the nominal knowledge available. Then, to compensate for the residual uncertainty, we robustify the controller including an additional term whose size is designed based on the variance provided by the GPR framework. We proved that, with high probability, the proposed scheme is able to guarantee asymptotic tracking of a desired trajectory. We tested numerically our strategy on a 2 degrees of freedom planar robot.
>
---
#### [new 033] Exteroception through Proprioception Sensing through Improved Contact Modeling for Soft Growing Robots
- **分类: cs.RO**

- **简介: 该论文属于环境探索任务，旨在解决软体机器人在非结构化环境中定位与建图的问题。通过改进接触建模，实现基于本体感觉的外部感知。**

- **链接: [http://arxiv.org/pdf/2507.10694v1](http://arxiv.org/pdf/2507.10694v1)**

> **作者:** Francesco Fuentes; Serigne Diagne; Zachary Kingston; Laura H. Blumenschein
>
> **备注:** 22 pages, 21 figures, submitted to journal for potential publication
>
> **摘要:** Passive deformation due to compliance is a commonly used benefit of soft robots, providing opportunities to achieve robust actuation with few active degrees of freedom. Soft growing robots in particular have shown promise in navigation of unstructured environments due to their passive deformation. If their collisions and subsequent deformations can be better understood, soft robots could be used to understand the structure of the environment from direct tactile measurements. In this work, we propose the use of soft growing robots as mapping and exploration tools. We do this by first characterizing collision behavior during discrete turns, then leveraging this model to develop a geometry-based simulator that models robot trajectories in 2D environments. Finally, we demonstrate the model and simulator validity by mapping unknown environments using Monte Carlo sampling to estimate the optimal next deployment given current knowledge. Over both uniform and non-uniform environments, this selection method rapidly approaches ideal actions, showing the potential for soft growing robots in unstructured environment exploration and mapping.
>
---
#### [new 034] Comparison of Localization Algorithms between Reduced-Scale and Real-Sized Vehicles Using Visual and Inertial Sensors
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶定位任务，研究缩比车辆与全尺寸车辆在视觉和惯性定位算法上的精度差异，验证缩比车作为测试平台的可行性。**

- **链接: [http://arxiv.org/pdf/2507.11241v1](http://arxiv.org/pdf/2507.11241v1)**

> **作者:** Tobias Kern; Leon Tolksdorf; Christian Birkner
>
> **摘要:** Physically reduced-scale vehicles are emerging to accelerate the development of advanced automated driving functions. In this paper, we investigate the effects of scaling on self-localization accuracy with visual and visual-inertial algorithms using cameras and an inertial measurement unit (IMU). For this purpose, ROS2-compatible visual and visual-inertial algorithms are selected, and datasets are chosen as a baseline for real-sized vehicles. A test drive is conducted to record data of reduced-scale vehicles. We compare the selected localization algorithms, OpenVINS, VINS-Fusion, and RTAB-Map, in terms of their pose accuracy against the ground-truth and against data from real-sized vehicles. When comparing the implementation of the selected localization algorithms to real-sized vehicles, OpenVINS has the lowest average localization error. Although all selected localization algorithms have overlapping error ranges, OpenVINS also performs best when applied to a reduced-scale vehicle. When reduced-scale vehicles were compared to real-sized vehicles, minor differences were found in translational vehicle motion estimation accuracy. However, no significant differences were found when comparing the estimation accuracy of rotational vehicle motion, allowing RSVRs to be used as testing platforms for self-localization algorithms.
>
---
#### [new 035] GeoHopNet: Hopfield-Augmented Sparse Spatial Attention for Dynamic UAV Site Location Problem
- **分类: cs.LG; cs.AI; cs.NE; cs.RO; 90B06; I.2.8**

- **简介: 该论文属于动态无人机站点选址任务，解决大规模城市级位置问题。提出GeoHopNet模型，通过改进注意力机制和引入外部记忆模块，提升计算效率与求解质量。**

- **链接: [http://arxiv.org/pdf/2507.10636v1](http://arxiv.org/pdf/2507.10636v1)**

> **作者:** Jianing Zhi; Xinghua Li; Zidong Chen
>
> **备注:** 12 Pages, 5 Figures
>
> **摘要:** The rapid development of urban low-altitude unmanned aerial vehicle (UAV) economy poses new challenges for dynamic site selection of UAV landing points and supply stations. Traditional deep reinforcement learning methods face computational complexity bottlenecks, particularly with standard attention mechanisms, when handling large-scale urban-level location problems. This paper proposes GeoHopNet, a Hopfield-augmented sparse spatial attention network specifically designed for dynamic UAV site location problems. Our approach introduces four core innovations: (1) distance-biased multi-head attention mechanism that explicitly encodes spatial geometric information; (2) K-nearest neighbor sparse attention that reduces computational complexity from $O(N^2)$ to $O(NK)$; (3) a modern Hopfield external memory module; and (4) a memory regularization strategy. Experimental results demonstrate that GeoHopNet extends the boundary of solvable problem sizes. For large-scale instances with 1,000 nodes, where standard attention models become prohibitively slow (over 3 seconds per instance) and traditional solvers fail, GeoHopNet finds high-quality solutions (0.22\% optimality gap) in under 0.1 seconds. Compared to the state-of-the-art ADNet baseline on 100-node instances, our method improves solution quality by 22.2\% and is 1.8$\times$ faster.
>
---
#### [new 036] Offline Reinforcement Learning with Wasserstein Regularization via Optimal Transport Maps
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于离线强化学习任务，旨在解决分布偏移问题。通过引入Wasserstein正则化和输入凸神经网络，实现稳定策略学习。**

- **链接: [http://arxiv.org/pdf/2507.10843v1](http://arxiv.org/pdf/2507.10843v1)**

> **作者:** Motoki Omura; Yusuke Mukuta; Kazuki Ota; Takayuki Osa; Tatsuya Harada
>
> **备注:** Accepted at RLC 2025
>
> **摘要:** Offline reinforcement learning (RL) aims to learn an optimal policy from a static dataset, making it particularly valuable in scenarios where data collection is costly, such as robotics. A major challenge in offline RL is distributional shift, where the learned policy deviates from the dataset distribution, potentially leading to unreliable out-of-distribution actions. To mitigate this issue, regularization techniques have been employed. While many existing methods utilize density ratio-based measures, such as the $f$-divergence, for regularization, we propose an approach that utilizes the Wasserstein distance, which is robust to out-of-distribution data and captures the similarity between actions. Our method employs input-convex neural networks (ICNNs) to model optimal transport maps, enabling the computation of the Wasserstein distance in a discriminator-free manner, thereby avoiding adversarial training and ensuring stable learning. Our approach demonstrates comparable or superior performance to widely used existing methods on the D4RL benchmark dataset. The code is available at https://github.com/motokiomura/Q-DOT .
>
---
#### [new 037] A Learning Framework For Cooperative Collision Avoidance of UAV Swarms Leveraging Domain Knowledge
- **分类: cs.MA; cs.LG; cs.RO**

- **简介: 该论文属于无人机编队避障任务，解决大规模UAV协同避障问题。通过引入领域知识驱动的奖励机制，简化了MARL训练过程，提升适应复杂环境的能力。**

- **链接: [http://arxiv.org/pdf/2507.10913v1](http://arxiv.org/pdf/2507.10913v1)**

> **作者:** Shuangyao Huang; Haibo Zhang; Zhiyi Huang
>
> **备注:** Under review at AAAI 2026
>
> **摘要:** This paper presents a multi-agent reinforcement learning (MARL) framework for cooperative collision avoidance of UAV swarms leveraging domain knowledge-driven reward. The reward is derived from knowledge in the domain of image processing, approximating contours on a two-dimensional field. By modeling obstacles as maxima on the field, collisions are inherently avoided as contours never go through peaks or intersect. Additionally, counters are smooth and energy-efficient. Our framework enables training with large swarm sizes as the agent interaction is minimized and the need for complex credit assignment schemes or observation sharing mechanisms in state-of-the-art MARL approaches are eliminated. Moreover, UAVs obtain the ability to adapt to complex environments where contours may be non-viable or non-existent through intensive training. Extensive experiments are conducted to evaluate the performances of our framework against state-of-the-art MARL algorithms.
>
---
#### [new 038] Task-Oriented Human Grasp Synthesis via Context- and Task-Aware Diffusers
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于任务导向的人类抓取合成任务，旨在解决如何根据场景和任务生成准确的抓取姿态。工作包括构建任务感知的接触图并提出新数据集与评估指标。**

- **链接: [http://arxiv.org/pdf/2507.11287v1](http://arxiv.org/pdf/2507.11287v1)**

> **作者:** An-Lun Liu; Yu-Wei Chao; Yi-Ting Chen
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** In this paper, we study task-oriented human grasp synthesis, a new grasp synthesis task that demands both task and context awareness. At the core of our method is the task-aware contact maps. Unlike traditional contact maps that only reason about the manipulated object and its relation with the hand, our enhanced maps take into account scene and task information. This comprehensive map is critical for hand-object interaction, enabling accurate grasping poses that align with the task. We propose a two-stage pipeline that first constructs a task-aware contact map informed by the scene and task. In the subsequent stage, we use this contact map to synthesize task-oriented human grasps. We introduce a new dataset and a metric for the proposed task to evaluate our approach. Our experiments validate the importance of modeling both scene and task, demonstrating significant improvements over existing methods in both grasp quality and task performance. See our project page for more details: https://hcis-lab.github.io/TOHGS/
>
---
#### [new 039] CogDDN: A Cognitive Demand-Driven Navigation with Decision Optimization and Dual-Process Thinking
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于机器人导航任务，解决未知环境中基于用户意图的物体定位问题。提出CogDDN框架，结合双思维系统和决策优化，提升导航准确性与适应性。**

- **链接: [http://arxiv.org/pdf/2507.11334v1](http://arxiv.org/pdf/2507.11334v1)**

> **作者:** Yuehao Huang; Liang Liu; Shuangming Lei; Yukai Ma; Hao Su; Jianbiao Mei; Pengxiang Zhao; Yaqing Gu; Yong Liu; Jiajun Lv
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Mobile robots are increasingly required to navigate and interact within unknown and unstructured environments to meet human demands. Demand-driven navigation (DDN) enables robots to identify and locate objects based on implicit human intent, even when object locations are unknown. However, traditional data-driven DDN methods rely on pre-collected data for model training and decision-making, limiting their generalization capability in unseen scenarios. In this paper, we propose CogDDN, a VLM-based framework that emulates the human cognitive and learning mechanisms by integrating fast and slow thinking systems and selectively identifying key objects essential to fulfilling user demands. CogDDN identifies appropriate target objects by semantically aligning detected objects with the given instructions. Furthermore, it incorporates a dual-process decision-making module, comprising a Heuristic Process for rapid, efficient decisions and an Analytic Process that analyzes past errors, accumulates them in a knowledge base, and continuously improves performance. Chain of Thought (CoT) reasoning strengthens the decision-making process. Extensive closed-loop evaluations on the AI2Thor simulator with the ProcThor dataset show that CogDDN outperforms single-view camera-only methods by 15%, demonstrating significant improvements in navigation accuracy and adaptability. The project page is available at https://yuehaohuang.github.io/CogDDN/.
>
---
## 更新

#### [replaced 001] Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19417v2](http://arxiv.org/pdf/2502.19417v2)**

> **作者:** Lucy Xiaoyang Shi; Brian Ichter; Michael Equi; Liyiming Ke; Karl Pertsch; Quan Vuong; James Tanner; Anna Walling; Haohuan Wang; Niccolo Fusai; Adrian Li-Bell; Danny Driess; Lachy Groom; Sergey Levine; Chelsea Finn
>
> **备注:** ICML 2025
>
> **摘要:** Generalist robots that can perform a range of different tasks in open-world settings must be able to not only reason about the steps needed to accomplish their goals, but also process complex instructions, prompts, and even feedback during task execution. Intricate instructions (e.g., "Could you make me a vegetarian sandwich?" or "I don't like that one") require not just the ability to physically perform the individual steps, but the ability to situate complex commands and feedback in the physical world. In this work, we describe a system that uses vision-language models in a hierarchical structure, first reasoning over complex prompts and user feedback to deduce the most appropriate next step to fulfill the task, and then performing that step with low-level actions. In contrast to direct instruction following methods that can fulfill simple commands ("pick up the cup"), our system can reason through complex prompts and incorporate situated feedback during task execution ("that's not trash"). We evaluate our system across three robotic platforms, including single-arm, dual-arm, and dual-arm mobile robots, demonstrating its ability to handle tasks such as cleaning messy tables, making sandwiches, and grocery shopping. Videos are available at https://www.pi.website/research/hirobot
>
---
#### [replaced 002] Feature-Based vs. GAN-Based Learning from Demonstrations: When and Why
- **分类: cs.LG; cs.AI; cs.GR; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.05906v2](http://arxiv.org/pdf/2507.05906v2)**

> **作者:** Chenhao Li; Marco Hutter; Andreas Krause
>
> **摘要:** This survey provides a comparative analysis of feature-based and GAN-based approaches to learning from demonstrations, with a focus on the structure of reward functions and their implications for policy learning. Feature-based methods offer dense, interpretable rewards that excel at high-fidelity motion imitation, yet often require sophisticated representations of references and struggle with generalization in unstructured settings. GAN-based methods, in contrast, use implicit, distributional supervision that enables scalability and adaptation flexibility, but are prone to training instability and coarse reward signals. Recent advancements in both paradigms converge on the importance of structured motion representations, which enable smoother transitions, controllable synthesis, and improved task integration. We argue that the dichotomy between feature-based and GAN-based methods is increasingly nuanced: rather than one paradigm dominating the other, the choice should be guided by task-specific priorities such as fidelity, diversity, interpretability, and adaptability. This work outlines the algorithmic trade-offs and design considerations that underlie method selection, offering a framework for principled decision-making in learning from demonstrations.
>
---
#### [replaced 003] Reinforcement Learning with Action Chunking
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **链接: [http://arxiv.org/pdf/2507.07969v2](http://arxiv.org/pdf/2507.07969v2)**

> **作者:** Qiyang Li; Zhiyuan Zhou; Sergey Levine
>
> **备注:** 25 pages, 15 figures
>
> **摘要:** We present Q-chunking, a simple yet effective recipe for improving reinforcement learning (RL) algorithms for long-horizon, sparse-reward tasks. Our recipe is designed for the offline-to-online RL setting, where the goal is to leverage an offline prior dataset to maximize the sample-efficiency of online learning. Effective exploration and sample-efficient learning remain central challenges in this setting, as it is not obvious how the offline data should be utilized to acquire a good exploratory policy. Our key insight is that action chunking, a technique popularized in imitation learning where sequences of future actions are predicted rather than a single action at each timestep, can be applied to temporal difference (TD)-based RL methods to mitigate the exploration challenge. Q-chunking adopts action chunking by directly running RL in a 'chunked' action space, enabling the agent to (1) leverage temporally consistent behaviors from offline data for more effective online exploration and (2) use unbiased $n$-step backups for more stable and efficient TD learning. Our experimental results demonstrate that Q-chunking exhibits strong offline performance and online sample efficiency, outperforming prior best offline-to-online methods on a range of long-horizon, sparse-reward manipulation tasks.
>
---
#### [replaced 004] FLAF: Focal Line and Feature-constrained Active View Planning for Visual Teach and Repeat
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.03457v5](http://arxiv.org/pdf/2409.03457v5)**

> **作者:** Changfei Fu; Weinan Chen; Wenjun Xu; Hong Zhang
>
> **摘要:** This paper presents FLAF, a focal line and feature-constrained active view planning method for tracking failure avoidance in feature-based visual navigation of mobile robots. Our FLAF-based visual navigation is built upon a feature-based visual teach and repeat (VT\&R) framework, which supports many robotic applications by teaching a robot to navigate on various paths that cover a significant portion of daily autonomous navigation requirements. However, tracking failure in feature-based visual simultaneous localization and mapping (VSLAM) caused by textureless regions in human-made environments is still limiting VT\&R to be adopted in the real world. To address this problem, the proposed view planner is integrated into a feature-based visual SLAM system to build up an active VT\&R system that avoids tracking failure. In our system, a pan-tilt unit (PTU)-based active camera is mounted on the mobile robot. Using FLAF, the active camera-based VSLAM operates during the teaching phase to construct a complete path map and in the repeat phase to maintain stable localization. FLAF orients the robot toward more map points to avoid mapping failures during path learning and toward more feature-identifiable map points beneficial for localization while following the learned trajectory. Experiments in real scenarios demonstrate that FLAF outperforms the methods that do not consider feature-identifiability, and our active VT\&R system performs well in complex environments by effectively dealing with low-texture regions.
>
---
#### [replaced 005] Learning and Transferring Better with Depth Information in Visual Reinforcement Learning
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.09180v2](http://arxiv.org/pdf/2507.09180v2)**

> **作者:** Zichun Xu; Yuntao Li; Zhaomin Wang; Lei Zhuang; Guocai Yang; Jingdong Zhao
>
> **摘要:** Depth information is robust to scene appearance variations and inherently carries 3D spatial details. In this paper, a visual backbone based on the vision transformer is proposed to fuse RGB and depth modalities for enhancing generalization. Different modalities are first processed by separate CNN stems, and the combined convolutional features are delivered to the scalable vision transformer to obtain visual representations. Moreover, a contrastive unsupervised learning scheme is designed with masked and unmasked tokens to accelerate the sample efficiency during the reinforcement learning progress. For sim2real transfer, a flexible curriculum learning schedule is developed to deploy domain randomization over training processes.
>
---
#### [replaced 006] Context-Aware Deep Lagrangian Networks for Model Predictive Control
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.15249v2](http://arxiv.org/pdf/2506.15249v2)**

> **作者:** Lucas Schulze; Jan Peters; Oleg Arenz
>
> **备注:** Accepted to the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Controlling a robot based on physics-consistent dynamic models, such as Deep Lagrangian Networks (DeLaN), can improve the generalizability and interpretability of the resulting behavior. However, in complex environments, the number of objects to potentially interact with is vast, and their physical properties are often uncertain. This complexity makes it infeasible to employ a single global model. Therefore, we need to resort to online system identification of context-aware models that capture only the currently relevant aspects of the environment. While physical principles such as the conservation of energy may not hold across varying contexts, ensuring physical plausibility for any individual context-aware model can still be highly desirable, particularly when using it for receding horizon control methods such as model predictive control (MPC). Hence, in this work, we extend DeLaN to make it context-aware, combine it with a recurrent network for online system identification, and integrate it with an MPC for adaptive, physics-consistent control. We also combine DeLaN with a residual dynamics model to leverage the fact that a nominal model of the robot is typically available. We evaluate our method on a 7-DOF robot arm for trajectory tracking under varying loads. Our method reduces the end-effector tracking error by 39%, compared to a 21% improvement achieved by a baseline that uses an extended Kalman filter.
>
---
#### [replaced 007] A Survey: Learning Embodied Intelligence from Physical Simulators and World Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.00917v2](http://arxiv.org/pdf/2507.00917v2)**

> **作者:** Xiaoxiao Long; Qingrui Zhao; Kaiwen Zhang; Zihao Zhang; Dingrui Wang; Yumeng Liu; Zhengjie Shu; Yi Lu; Shouzheng Wang; Xinzhe Wei; Wei Li; Wei Yin; Yao Yao; Jia Pan; Qiu Shen; Ruigang Yang; Xun Cao; Qionghai Dai
>
> **备注:** 49pages, 25figures, 6tables, github repository avalible in https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey
>
> **摘要:** The pursuit of artificial general intelligence (AGI) has placed embodied intelligence at the forefront of robotics research. Embodied intelligence focuses on agents capable of perceiving, reasoning, and acting within the physical world. Achieving robust embodied intelligence requires not only advanced perception and control, but also the ability to ground abstract cognition in real-world interactions. Two foundational technologies, physical simulators and world models, have emerged as critical enablers in this quest. Physical simulators provide controlled, high-fidelity environments for training and evaluating robotic agents, allowing safe and efficient development of complex behaviors. In contrast, world models empower robots with internal representations of their surroundings, enabling predictive planning and adaptive decision-making beyond direct sensory input. This survey systematically reviews recent advances in learning embodied AI through the integration of physical simulators and world models. We analyze their complementary roles in enhancing autonomy, adaptability, and generalization in intelligent robots, and discuss the interplay between external simulation and internal modeling in bridging the gap between simulated training and real-world deployment. By synthesizing current progress and identifying open challenges, this survey aims to provide a comprehensive perspective on the path toward more capable and generalizable embodied AI systems. We also maintain an active repository that contains up-to-date literature and open-source projects at https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey.
>
---
#### [replaced 008] RoboBrain 2.0 Technical Report
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.02029v3](http://arxiv.org/pdf/2507.02029v3)**

> **作者:** BAAI RoboBrain Team; Mingyu Cao; Huajie Tan; Yuheng Ji; Minglan Lin; Zhiyu Li; Zhou Cao; Pengwei Wang; Enshen Zhou; Yi Han; Yingbo Tang; Xiangqi Xu; Wei Guo; Yaoxu Lyu; Yijie Xu; Jiayu Shi; Mengfei Du; Cheng Chi; Mengdi Zhao; Xiaoshuai Hao; Junkai Zhao; Xiaojie Zhang; Shanyu Rong; Huaihai Lyu; Zhengliang Cai; Yankai Fu; Ning Chen; Bolun Zhang; Lingfeng Zhang; Shuyi Zhang; Dong Liu; Xi Feng; Songjing Wang; Xiaodan Liu; Yance Jiao; Mengsi Lyu; Zhuo Chen; Chenrui He; Yulong Ao; Xue Sun; Zheqi He; Jingshu Zheng; Xi Yang; Donghai Shi; Kunchang Xie; Bochao Zhang; Shaokai Nie; Chunlei Men; Yonghua Lin; Zhongyuan Wang; Tiejun Huang; Shanghang Zhang
>
> **摘要:** We introduce RoboBrain 2.0, our latest generation of embodied vision-language foundation models, designed to unify perception, reasoning, and planning for complex embodied tasks in physical environments. It comes in two variants: a lightweight 7B model and a full-scale 32B model, featuring a heterogeneous architecture with a vision encoder and a language model. Despite its compact size, RoboBrain 2.0 achieves strong performance across a wide spectrum of embodied reasoning tasks. On both spatial and temporal benchmarks, the 32B variant achieves leading results, surpassing prior open-source and proprietary models. In particular, it supports key real-world embodied AI capabilities, including spatial understanding (e.g., affordance prediction, spatial referring, trajectory forecasting) and temporal decision-making (e.g., closed-loop interaction, multi-agent long-horizon planning, and scene graph updating). This report details the model architecture, data construction, multi-stage training strategies, infrastructure and practical applications. We hope RoboBrain 2.0 advances embodied AI research and serves as a practical step toward building generalist embodied agents. The code, checkpoint and benchmark are available at https://superrobobrain.github.io.
>
---
#### [replaced 009] Irrotational Contact Fields
- **分类: cs.RO; cs.CE; math-ph; math.MP**

- **链接: [http://arxiv.org/pdf/2312.03908v3](http://arxiv.org/pdf/2312.03908v3)**

> **作者:** Alejandro Castro; Xuchen Han; Joseph Masterjohn
>
> **备注:** 16 pages, 26 figures. The supplemental video is available publicly at https://youtu.be/FTUPYZ_8Xbk?si=MWndCUCGWMJsFnsO
>
> **摘要:** We present a framework for generating convex approximations of complex contact models, incorporating experimentally validated models like Hunt & Crossley coupled with Coulomb's law of friction alongside the principle of maximum dissipation. Our approach is robust across a wide range of stiffness values, making it suitable for both compliant surfaces and rigid approximations. We evaluate these approximations across a wide variety of test cases, detailing properties and limitations. We implement a fully differentiable solution in the open-source robotics toolkit, Drake. Our novel hybrid approach enables computation of gradients for complex geometric models while reusing factorizations from contact resolution. We demonstrate robust simulation of robotic tasks at interactive rates, with accurately resolved stiction and contact transitions, supporting effective sim-to-real transfer.
>
---
#### [replaced 010] MVCTrack: Boosting 3D Point Cloud Tracking via Multimodal-Guided Virtual Cues
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.02734v5](http://arxiv.org/pdf/2412.02734v5)**

> **作者:** Zhaofeng Hu; Sifan Zhou; Zhihang Yuan; Dawei Yang; Shibo Zhao; Ci-Jyun Liang
>
> **备注:** Accepted by ICRA 2025
>
> **摘要:** 3D single object tracking is essential in autonomous driving and robotics. Existing methods often struggle with sparse and incomplete point cloud scenarios. To address these limitations, we propose a Multimodal-guided Virtual Cues Projection (MVCP) scheme that generates virtual cues to enrich sparse point clouds. Additionally, we introduce an enhanced tracker MVCTrack based on the generated virtual cues. Specifically, the MVCP scheme seamlessly integrates RGB sensors into LiDAR-based systems, leveraging a set of 2D detections to create dense 3D virtual cues that significantly improve the sparsity of point clouds. These virtual cues can naturally integrate with existing LiDAR-based 3D trackers, yielding substantial performance gains. Extensive experiments demonstrate that our method achieves competitive performance on the NuScenes dataset.
>
---
#### [replaced 011] View Invariant Learning for Vision-Language Navigation in Continuous Environments
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.08831v2](http://arxiv.org/pdf/2507.08831v2)**

> **作者:** Josh Qixuan Sun; Xiaoying Xing; Huaiyuan Weng; Chul Min Yeum; Mark Crowley
>
> **备注:** Under review
>
> **摘要:** Vision-Language Navigation in Continuous Environments (VLNCE), where an agent follows instructions and moves freely to reach a destination, is a key research problem in embodied AI. However, most navigation policies are sensitive to viewpoint changes, i.e., variations in camera height and viewing angle that alter the agent's observation. In this paper, we introduce a generalized scenario, V2-VLNCE (VLNCE with Varied Viewpoints), and propose VIL (View Invariant Learning), a view-invariant post-training strategy that enhances the robustness of existing navigation policies to changes in camera viewpoint. VIL employs a contrastive learning framework to learn sparse and view-invariant features. Additionally, we introduce a teacher-student framework for the Waypoint Predictor Module, a core component of most VLNCE baselines, where a view-dependent teacher model distills knowledge into a view-invariant student model. We employ an end-to-end training paradigm to jointly optimize these components, thus eliminating the cost for individual module training. Empirical results show that our method outperforms state-of-the-art approaches on V2-VLNCE by 8-15% measured on Success Rate for two standard benchmark datasets R2R-CE and RxR-CE. Furthermore, we evaluate VIL under the standard VLNCE setting and find that, despite being trained for varied viewpoints, it often still improves performance. On the more challenging RxR-CE dataset, our method also achieved state-of-the-art performance across all metrics when compared to other map-free methods. This suggests that adding VIL does not diminish the standard viewpoint performance and can serve as a plug-and-play post-training method.
>
---
#### [replaced 012] Grasping a Handful: Sequential Multi-Object Dexterous Grasp Generation
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.22370v3](http://arxiv.org/pdf/2503.22370v3)**

> **作者:** Haofei Lu; Yifei Dong; Zehang Weng; Florian Pokorny; Jens Lundell; Danica Kragic
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** We introduce the sequential multi-object robotic grasp sampling algorithm SeqGrasp that can robustly synthesize stable grasps on diverse objects using the robotic hand's partial Degrees of Freedom (DoF). We use SeqGrasp to construct the large-scale Allegro Hand sequential grasping dataset SeqDataset and use it for training the diffusion-based sequential grasp generator SeqDiffuser. We experimentally evaluate SeqGrasp and SeqDiffuser against the state-of-the-art non-sequential multi-object grasp generation method MultiGrasp in simulation and on a real robot. The experimental results demonstrate that SeqGrasp and SeqDiffuser reach an 8.71%-43.33% higher grasp success rate than MultiGrasp. Furthermore, SeqDiffuser is approximately 1000 times faster at generating grasps than SeqGrasp and MultiGrasp.
>
---
#### [replaced 013] LVLM-MPC Collaboration for Autonomous Driving: A Safety-Aware and Task-Scalable Control Architecture
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2505.04980v2](http://arxiv.org/pdf/2505.04980v2)**

> **作者:** Kazuki Atsuta; Kohei Honda; Hiroyuki Okuda; Tatsuya Suzuki
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** This paper proposes a novel Large Vision-Language Model (LVLM) and Model Predictive Control (MPC) integration framework that delivers both task scalability and safety for Autonomous Driving (AD). LVLMs excel at high-level task planning across diverse driving scenarios. However, since these foundation models are not specifically designed for driving and their reasoning is not consistent with the feasibility of low-level motion planning, concerns remain regarding safety and smooth task switching. This paper integrates LVLMs with MPC Builder, which automatically generates MPCs on demand, based on symbolic task commands generated by the LVLM, while ensuring optimality and safety. The generated MPCs can strongly assist the execution or rejection of LVLM-driven task switching by providing feedback on the feasibility of the given tasks and generating task-switching-aware MPCs. Our approach provides a safe, flexible, and adaptable control framework, bridging the gap between cutting-edge foundation models and reliable vehicle operation. We demonstrate the effectiveness of our approach through a simulation experiment, showing that our system can safely and effectively handle highway driving while maintaining the flexibility and adaptability of LVLMs.
>
---
#### [replaced 014] Characterizing gaussian mixture of motion modes for skid-steer vehicle state estimation
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2505.00200v2](http://arxiv.org/pdf/2505.00200v2)**

> **作者:** Ameya Salvi; Mark Brudnak; Jonathon M. Smereka; Matthias Schmid; Venkat Krovi
>
> **摘要:** Skid-steered wheel mobile robots (SSWMRs) are characterized by the unique domination of the tire-terrain skidding for the robot to move. The lack of reliable friction models cascade into unreliable motion models, especially the reduced ordered variants used for state estimation and robot control. Ensemble modeling is an emerging research direction where the overall motion model is broken down into a family of local models to distribute the performance and resource requirement and provide a fast real-time prediction. To this end, a gaussian mixture model based modeling identification of model clusters is adopted and implemented within an interactive multiple model (IMM) based state estimation. The framework is adopted and implemented for angular velocity as the estimated state for a mid scaled skid-steered wheel mobile robot platform.
>
---
#### [replaced 015] Seeking to Collide: Online Safety-Critical Scenario Generation for Autonomous Driving with Retrieval Augmented Large Language Models
- **分类: cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.00972v2](http://arxiv.org/pdf/2505.00972v2)**

> **作者:** Yuewen Mei; Tong Nie; Jian Sun; Ye Tian
>
> **备注:** Accepted at IEEE ITSC 2025
>
> **摘要:** Simulation-based testing is crucial for validating autonomous vehicles (AVs), yet existing scenario generation methods either overfit to common driving patterns or operate in an offline, non-interactive manner that fails to expose rare, safety-critical corner cases. In this paper, we introduce an online, retrieval-augmented large language model (LLM) framework for generating safety-critical driving scenarios. Our method first employs an LLM-based behavior analyzer to infer the most dangerous intent of the background vehicle from the observed state, then queries additional LLM agents to synthesize feasible adversarial trajectories. To mitigate catastrophic forgetting and accelerate adaptation, we augment the framework with a dynamic memorization and retrieval bank of intent-planner pairs, automatically expanding its behavioral library when novel intents arise. Evaluations using the Waymo Open Motion Dataset demonstrate that our model reduces the mean minimum time-to-collision from 1.62 to 1.08 s and incurs a 75% collision rate, substantially outperforming baselines.
>
---
#### [replaced 016] RA-DP: Rapid Adaptive Diffusion Policy for Training-Free High-frequency Robotics Replanning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.04051v2](http://arxiv.org/pdf/2503.04051v2)**

> **作者:** Xi Ye; Rui Heng Yang; Jun Jin; Yinchuan Li; Amir Rasouli
>
> **备注:** Accepted by IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Diffusion models exhibit impressive scalability in robotic task learning, yet they struggle to adapt to novel, highly dynamic environments. This limitation primarily stems from their constrained replanning ability: they either operate at a low frequency due to a time-consuming iterative sampling process, or are unable to adapt to unforeseen feedback in case of rapid replanning. To address these challenges, we propose RA-DP, a novel diffusion policy framework with training-free high-frequency replanning ability that solves the above limitations in adapting to unforeseen dynamic environments. Specifically, our method integrates guidance signals which are often easily obtained in the new environment during the diffusion sampling process, and utilizes a novel action queue mechanism to generate replanned actions at every denoising step without retraining, thus forming a complete training-free framework for robot motion adaptation in unseen environments. Extensive evaluations have been conducted in both well-recognized simulation benchmarks and real robot tasks. Results show that RA-DP outperforms the state-of-the-art diffusion-based methods in terms of replanning frequency and success rate. Moreover, we show that our framework is theoretically compatible with any training-free guidance signal.
>
---
#### [replaced 017] Extending the Benefits of Parallel Elasticity across Multiple Actuation Tasks: A Geometric and Optimization-Based Approach
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.08889v4](http://arxiv.org/pdf/2409.08889v4)**

> **作者:** Kang Yang; Myia Dickens; James Schmiedeler; Edgar Bolívar-Nieto
>
> **摘要:** A spring in parallel with an effort source (e.g., electric motor or human muscle) can reduce its energy consumption and effort (i.e., torque or force) depending on the spring stiffness, spring preload, and actuation task. However, selecting the spring stiffness and preload that guarantees effort or energy reduction for an arbitrary set of tasks is a design challenge. This work formulates a convex optimization problem to guarantee that a parallel spring reduces the root-mean-square source effort or energy consumption for multiple tasks. Specifically, we guarantee the benefits across multiple tasks by enforcing a set of convex quadratic constraints in our optimization variables, the parallel spring stiffness and preload. These quadratic constraints are equivalent to ellipses in the stiffness and preload plane; any combination of stiffness and preload inside the ellipse represents a parallel spring that minimizes effort source or energy consumption with respect to an actuator without a spring. This geometric interpretation intuitively guides the stiffness and preload selection process. We analytically and experimentally prove the convex quadratic function of the spring stiffness and preload. As applications, we analyze the stiffness and preload selection of a parallel spring for a knee exoskeleton using human muscle as the effort source and a prosthetic ankle powered by electric motors. The source code associated with our framework is available as supplemental open-source software.
>
---
#### [replaced 018] Sim2Real Diffusion: Learning Cross-Domain Adaptive Representations for Transferable Autonomous Driving
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.00236v2](http://arxiv.org/pdf/2507.00236v2)**

> **作者:** Chinmay Vilas Samak; Tanmay Vilas Samak; Bing Li; Venkat Krovi
>
> **摘要:** Simulation-based design, optimization, and validation of autonomous driving algorithms have proven to be crucial for their improvement over the years. Nevertheless, the ultimate measure of effectiveness is their successful transition from simulation to reality (sim2real). However, existing sim2real transfer methods struggle to address the autonomy-oriented requirements of balancing: (i) conditioned domain adaptation, (ii) robust performance with limited examples, (iii) modularity in handling multiple domain representations, and (iv) real-time performance. To alleviate these pain points, we present a unified framework for learning cross-domain adaptive representations through conditional latent diffusion for sim2real transferable autonomous driving algorithms. Our framework offers options to leverage: (i) alternate foundation models, (ii) a few-shot fine-tuning pipeline, and (iii) textual as well as image prompts for mapping across given source and target domains. It is also capable of generating diverse high-quality samples when diffusing across parameter spaces such as times of day, weather conditions, seasons, and operational design domains. We systematically analyze the presented framework and report our findings in terms of performance benchmarks and ablation studies, with critical quantitative metrics as well as insightful qualitative examples and remarks. Additionally, we demonstrate the serviceability of sim2real diffusion for autonomous driving using a behavioral cloning case study. Our experiments indicate that the proposed framework is capable of bridging the perceptual sim2real gap by over 40%, which highlights the potential of diffusion models in sim2real transfer.
>
---
#### [replaced 019] BlueME: Robust Underwater Robot-to-Robot Communication Using Compact Magnetoelectric Antennas
- **分类: cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2411.09241v3](http://arxiv.org/pdf/2411.09241v3)**

> **作者:** Mehron Talebi; Sultan Mahmud; Adam Khalifa; Md Jahidul Islam
>
> **摘要:** We present the design, development, and experimental validation of BlueME, a compact magnetoelectric (ME) antenna array system for underwater robot-to-robot communication. BlueME employs ME antennas operating at their natural mechanical resonance frequency to efficiently transmit and receive very-low-frequency (VLF) electromagnetic signals underwater. We outline the design, simulation, fabrication, and integration of the proposed system on low-power embedded platforms focusing on portable and scalable applications. For performance evaluation, we deployed BlueME on an autonomous surface vehicle (ASV) and a remotely operated vehicle (ROV) in open-water field trials. Our tests demonstrate that BlueME maintains reliable signal transmission at distances beyond 200 meters while consuming only 1 watt of power. Field trials show that the system operates effectively in challenging underwater conditions such as turbidity, obstacles, and multipath interference -- that generally affect acoustics and optics. Our analysis also examines the impact of complete submersion on system performance and identifies key deployment considerations. This work represents the first practical underwater deployment of ME antennas outside the laboratory, and implements the largest VLF ME array system to date. BlueME demonstrates significant potential for marine robotics and automation in multi-robot cooperative systems and remote sensor networks.
>
---
#### [replaced 020] Online Intrinsic Rewards for Decision Making Agents from Large Language Model Feedback
- **分类: cs.LG; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.23022v3](http://arxiv.org/pdf/2410.23022v3)**

> **作者:** Qinqing Zheng; Mikael Henaff; Amy Zhang; Aditya Grover; Brandon Amos
>
> **摘要:** Automatically synthesizing dense rewards from natural language descriptions is a promising paradigm in reinforcement learning (RL), with applications to sparse reward problems, open-ended exploration, and hierarchical skill design. Recent works have made promising steps by exploiting the prior knowledge of large language models (LLMs). However, these approaches suffer from important limitations: they are either not scalable to problems requiring billions of environment samples, due to requiring LLM annotations for each observation, or they require a diverse offline dataset, which may not exist or be impossible to collect. In this work, we address these limitations through a combination of algorithmic and systems-level contributions. We propose ONI, a distributed architecture that simultaneously learns an RL policy and an intrinsic reward function using LLM feedback. Our approach annotates the agent's collected experience via an asynchronous LLM server, which is then distilled into an intrinsic reward model. We explore a range of algorithmic choices for reward modeling with varying complexity, including hashing, classification, and ranking models. Our approach achieves state-of-the-art performance across a range of challenging tasks from the NetHack Learning Environment, while removing the need for large offline datasets required by prior work. We make our code available at https://github.com/facebookresearch/oni .
>
---
#### [replaced 021] mmE-Loc: Facilitating Accurate Drone Landing with Ultra-High-Frequency Localization
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.09469v2](http://arxiv.org/pdf/2507.09469v2)**

> **作者:** Haoyang Wang; Jingao Xu; Xinyu Luo; Ting Zhang; Xuecheng Chen; Ruiyang Duan; Jialong Chen; Yunhao Liu; Jianfeng Zheng; Weijie Hong; Xinlei Chen
>
> **备注:** 17 pages, 34 figures. Journal extended version of arXiv:2502.14992
>
> **摘要:** For precise, efficient, and safe drone landings, ground platforms should real-time, accurately locate descending drones and guide them to designated spots. While mmWave sensing combined with cameras improves localization accuracy, lower sampling frequency of traditional frame cameras compared to mmWave radar creates bottlenecks in system throughput. In this work, we upgrade traditional frame camera with event camera, a novel sensor that harmonizes in sampling frequency with mmWave radar within ground platform setup, and introduce mmE-Loc, a high-precision, low-latency ground localization system designed for precise drone landings. To fully exploit the \textit{temporal consistency} and \textit{spatial complementarity} between these two modalities, we propose two innovative modules: \textit{(i)} the Consistency-instructed Collaborative Tracking module, which further leverages the drone's physical knowledge of periodic micro-motions and structure for accurate measurements extraction, and \textit{(ii)} the Graph-informed Adaptive Joint Optimization module, which integrates drone motion information for efficient sensor fusion and drone localization. Real-world experiments conducted in landing scenarios with a drone delivery company demonstrate that mmE-Loc significantly outperforms state-of-the-art methods in both accuracy and latency.
>
---
#### [replaced 022] SECURE: Semantics-aware Embodied Conversation under Unawareness for Lifelong Robot Learning
- **分类: cs.RO; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.17755v3](http://arxiv.org/pdf/2409.17755v3)**

> **作者:** Rimvydas Rubavicius; Peter David Fagan; Alex Lascarides; Subramanian Ramamoorthy
>
> **备注:** Published at 4th Conference on Lifelong Learning Agents (CoLLAs), 2025
>
> **摘要:** This paper addresses a challenging interactive task learning scenario we call rearrangement under unawareness: an agent must manipulate a rigid-body environment without knowing a key concept necessary for solving the task and must learn about it during deployment. For example, the user may ask to "put the two granny smith apples inside the basket", but the agent cannot correctly identify which objects in the environment are "granny smith" as the agent has not been exposed to such a concept before. We introduce SECURE, an interactive task learning policy designed to tackle such scenarios. The unique feature of SECURE is its ability to enable agents to engage in semantic analysis when processing embodied conversations and making decisions. Through embodied conversation, a SECURE agent adjusts its deficient domain model by engaging in dialogue to identify and learn about previously unforeseen possibilities. The SECURE agent learns from the user's embodied corrective feedback when mistakes are made and strategically engages in dialogue to uncover useful information about novel concepts relevant to the task. These capabilities enable the SECURE agent to generalize to new tasks with the acquired knowledge. We demonstrate in the simulated Blocksworld and the real-world apple manipulation environments that the SECURE agent, which solves such rearrangements under unawareness, is more data-efficient than agents that do not engage in embodied conversation or semantic analysis.
>
---
