# 机器人 cs.RO

- **最新发布 147 篇**

- **更新 76 篇**

## 最新发布

#### [new 001] Path-conditioned Reinforcement Learning-based Local Planning for Long-Range Navigation
- **分类: cs.RO**

- **简介: 该论文属于长期导航任务，旨在解决局部规划对全局路径依赖过强的问题。通过基于强化学习的策略，利用路径信息作为上下文引导，提升导航效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.13888](https://arxiv.org/pdf/2603.13888)**

> **作者:** Mateo Haro; Julia Richter; Fan Yang; Cesar Cadena; Marco Hutter
>
> **摘要:** Long-range navigation is commonly addressed through hierarchical pipelines in which a global planner generates a path, decomposed into waypoints, and followed sequentially by a local planner. These systems are sensitive to global path quality, as inaccurate remote sensing data can result in locally infeasible waypoints, which degrade local execution. At the same time, the limited global context available to the local planner hinders long-range efficiency. To address this issue, we propose a reinforcement learning-based local navigation policy that leverages path information as contextual guidance. The policy is conditioned on reference path observations and trained with a reward function mainly based on goal-reaching objectives, without any explicit path-following reward. Through this implicit conditioning, the policy learns to opportunistically exploit path information while remaining robust to misleading or degraded guidance. Experimental results show that the proposed approach significantly improves navigation efficiency when high-quality paths are available and maintains baseline-level performance when path observations are severely degraded or even non-existent. These properties make the method particularly well-suited for long-range navigation scenarios in which high-level plans are approximate and local execution must remain adaptive to uncertainty.
>
---
#### [new 002] Design of a Bio-Inspired Miniature Submarine for Low-Cost Water Quality Monitoring
- **分类: cs.RO**

- **简介: 论文提出一种低成本生物启发微型潜艇，用于水环境监测。解决传统设备成本高、体积大的问题，通过仿生推进和低成本组件实现稳定控制与采样。**

- **链接: [https://arxiv.org/pdf/2603.14244](https://arxiv.org/pdf/2603.14244)**

> **作者:** Quang Huy Vu; Quan Le; Manh Duong Phung
>
> **摘要:** Water quality monitoring is essential for protecting aquatic ecosystems and detecting environmental pollution. This paper presents the design and experimental validation of a bio-inspired miniature submarine for low-cost water quality monitoring. Inspired by the jet propulsion mechanism of squids, the proposed system employs pump-driven water jets for propulsion and steering, combined with a pump-based buoyancy control mechanism that enables both depth regulation and water sampling. The vehicle integrates low-cost, commercially available components including an ESP32 microcontroller, IMU, pressure sensor, GPS receiver, and LoRa communication module. The complete system can be constructed at a hardware cost of approximately $122.5, making it suitable for educational and environmental monitoring applications. Experimental validation was conducted through pool tests and field trials in a lake. During a 360 degrees rotation test, roll and pitch deviations remained within +/-2 degrees and +/-1.5 degrees, respectively, demonstrating stable attitude control. Steering experiments showed a heading step response with approximately 2 s rise time and 5 s settling time. Depth control experiments achieved a target depth of 2.5 m with steady-state error within +/-0.1 m. Field experiments further demonstrated reliable navigation and successful water sampling operations. The results confirm that the proposed platform provides a compact, stable, and cost-effective solution for small-scale aquatic environmental monitoring.
>
---
#### [new 003] TransCurriculum: Multi-Dimensional Curriculum Learning for Fast & Stable Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，解决高速行走中的稳定性和迁移损失问题。提出TransCurriculum方法，通过多维课程学习提升四足机器人运动性能。**

- **链接: [https://arxiv.org/pdf/2603.14156](https://arxiv.org/pdf/2603.14156)**

> **作者:** Prakhar Mishra; Amir Hossain Raj; Xuesu Xiao; Dinesh Manocha
>
> **摘要:** High-speed legged locomotion struggles with stability and transfer losses at higher command velocities during deployment. One reason is that most curricula vary difficulty along single axis, for example increase the range of command velocities, terrain difficulty, or domain parameters (e.g. friction or payload mass) using either fixed update rule or instantaneous rewards while ignoring how the history of robot training has evolved. We propose TransCurriculum, a transformer-based multi-dimensional curriculum learning approach for agile quadrupedal locomotion. TransCurriculum adapts to 3 axes, velocity command targets, terrain difficulty, and domain randomization parameters (friction and payload mass). Rather than feeding task reward history directly into the low-level control policy, our formulation exploits it at the curriculum level. A transformer-based teacher retrieves the sequence of rewards and uses it to predict future rewards, success rate, and learning progress to guide expansion of this multidimensional curriculum towards high performing task bins. Finally we validate our approach on the Unitree Go1 robot in simulation (Isaac Gym) and deploy it zero-shot on Go1 hardware. Our TransCurriculum policy achieves a maximum velocity of 6.3 m/s in simulation and outperforms prior curriculum baselines. We tested our TransCurriculum trained policy on terrains (carpets, slopes, tiles, concrete), achieving a forward velocity of 4.1 m/s on carpet surpassing the fastest curriculum methods by 18.8% and achieves maximum zero-shot value among all tested methods. Our multi-dimensional curriculum also reduces the transfer loss to 18% from 27% for command only curriculum, demonstrating the benefits of joint training over velocity, terrain and domain randomization dimension while keeping the task success rate of 80-90% on rigid indoor and outdoor surfaces.
>
---
#### [new 004] Multi-Mode Pneumatic Artificial Muscles Driven by Hybrid Positive-Negative Pressure
- **分类: cs.RO**

- **简介: 该论文属于软体机器人领域，旨在开发新型人工肌肉。通过混合正负压驱动的IN-FOAM结构，解决传统人工肌肉灵活性与可编程性不足的问题。工作包括结构设计、实验与理论分析，实现多种运动模式。**

- **链接: [https://arxiv.org/pdf/2603.15066](https://arxiv.org/pdf/2603.15066)**

> **作者:** Siyuan Feng; Ruoyu Feng; Shuguang Li
>
> **备注:** 20 pages, 17 figures. Published in IEEE Transactions on Robotics
>
> **摘要:** Artificial muscles embody human aspirations for engineering lifelike robotic movements. This paper introduces an architecture for Inflatable Fluid-Driven Origami-Inspired Artificial Muscles (IN-FOAMs). A typical IN-FOAM consists of an inflatable skeleton enclosed within an outer skin, which can be driven using a combination of positive and negative pressures (e.g., compressed air and vacuum). IN-FOAMs are manufactured using low-cost heat-sealable sheet materials through heat-pressing and heat-sealing processes. Thus, they can be ultra-thin when not actuated, making them flexible, lightweight, and portable. The skeleton patterns are programmable, enabling a variety of motions, including contracting, bending, twisting, and rotating, based on specific skeleton designs. We conducted comprehensive experimental, theoretical, and numerical studies to investigate IN-FOAM's basic mechanical behavior and properties. The results show that IN-FOAM's output force and contraction can be tuned through multiple operation modes with the applied hybrid positive-negative pressure. Additionally, we propose multilayer skeleton structures to enhance the contraction ratio further, and we demonstrate a multi-channel skeleton approach that allows the integration of multiple motion modes into a single IN-FOAM. These findings indicate that IN-FOAMs hold great potential for future applications in flexible wearable devices and compact soft robotic systems.
>
---
#### [new 005] Spatially Grounded Long-Horizon Task Planning in the Wild
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，解决VLM生成的计划缺乏空间可执行性的问题。提出GroundedPlanBench基准和V2GP方法，提升长时空中空间规划能力。**

- **链接: [https://arxiv.org/pdf/2603.13433](https://arxiv.org/pdf/2603.13433)**

> **作者:** Sehun Jung; HyunJee Song; Dong-Hee Kim; Reuben Tan; Jianfeng Gao; Yong Jae Lee; Donghyun Kim
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Recent advances in robot manipulation increasingly leverage Vision-Language Models (VLMs) for high-level reasoning, such as decomposing task instructions into sequential action plans expressed in natural language that guide downstream low-level motor execution. However, current benchmarks do not assess whether these plans are spatially executable, particularly in specifying the exact spatial locations where the robot should interact to execute the plan, limiting evaluation of real-world manipulation capability. To bridge this gap, we define a novel task of grounded planning and introduce GroundedPlanBench, a newly curated benchmark for spatially grounded long-horizon action planning in the wild. GroundedPlanBench jointly evaluates hierarchical sub-action planning and spatial action grounding (where to act), enabling systematic assessment of whether generated sub-actions are spatially executable for robot manipulation. We further introduce Video-to-Spatially Grounded Planning (V2GP), an automated data generation framework that leverages real-world robot video demonstrations to improve spatially grounded long-horizon planning. Our evaluations reveal that spatially grounded long-horizon planning remains a major bottleneck for current VLMs. Our results demonstrate that V2GP provides a promising approach for improving both action planning and spatial grounding performance, validated on our benchmark as well as through real-world robot manipulation experiments, advancing progress toward spatially actionable planning.
>
---
#### [new 006] Building Explicit World Model for Zero-Shot Open-World Object Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决开放世界中物体操作的零样本泛化问题。通过构建物理驱动的数字孪生环境，实现无需特定动作示范的策略探索与部署。**

- **链接: [https://arxiv.org/pdf/2603.13825](https://arxiv.org/pdf/2603.13825)**

> **作者:** Xiaotong Li; Gang Chen; Javier Alonso-Mora
>
> **摘要:** Open-world object manipulation remains a fundamental challenge in robotics. While Vision-Language-Action (VLA) models have demonstrated promising results, they rely heavily on large-scale robot action demonstrations, which are costly to collect and can hinder out-of-distribution generalization. In this paper, we propose an explicit-world-model-based framework for open-world manipulation that achieves zero-shot generalization by constructing a physically grounded digital twin of the environment. The framework integrates open-set perception, digital-twin reconstruction, sampling and evaluation of interaction strategies. By constructing a digital twin of the environment, our approach efficiently explores and evaluates manipulation strategies in physic-enabled simulator and reliably deploys the chosen strategy to the real world. Experimentally, the proposed framework is able to perform multiple open-set manipulation tasks without any task-specific action demonstrations, proving strong zero-shot generalization on both the task and object levels. Project Page: this https URL
>
---
#### [new 007] Optimal control of differentially flat underactuated planar robots in the perspective of oscillation mitigation
- **分类: cs.RO**

- **简介: 该论文研究如何通过最优控制与微分平坦控制结合，解决欠驱动平面机器人轨迹跟踪中的振荡问题。任务是提升控制精度与稳定性。工作包括理论分析与仿真验证。**

- **链接: [https://arxiv.org/pdf/2603.15528](https://arxiv.org/pdf/2603.15528)**

> **作者:** Stefano Lovato; Michele Tonan; Matteo Bottin; Matteo Massaro; Alberto Doria; Giulio Rosati
>
> **备注:** Accepted to European Control Conference (ECC 2026)
>
> **摘要:** Underactuated robots are characterized by a larger number of degrees of freedom than actuators and if they are designed with a specific mass distribution, they can be controlled by means of differential flatness theory. This structural property enables the development of lightweight and cost-effective robotic systems with enhanced dexterity. However, a key challenge lies in managing the passive joints, whose control demands precise and comprehensive dynamic modeling of the system. To simplify dynamic models, particularly for low-speed trajectories, friction is often neglected. While this assumption simplifies analysis and control design, it introduces residual oscillations of the end-effector about the target position. In this paper, the possibility of using optimal control along with differential flatness control is investigated to improve the tracking of the planned trajectories. First, the study was carried out through formal analysis, and then, it was validated by means of numerical simulations. Results highlight that optimal control can be used to plan the flat variables considering different (quadratic) performance indices: control effort, i.e. motor torque, and potential energy of the considered underactuated joint. Moreover, the minimization of potential energy can be used to design motion laws that are robust against variation of the stiffness and damping of the underactuated joint, thus reducing oscillations in the case of stiffness/damping mismatch.
>
---
#### [new 008] REFINE-DP: Diffusion Policy Fine-tuning for Humanoid Loco-manipulation via Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对人形机器人运动操作任务，解决高阶规划与低阶控制脱节导致的执行不稳定问题。通过强化学习微调扩散策略，提升任务成功率与运动质量。**

- **链接: [https://arxiv.org/pdf/2603.13707](https://arxiv.org/pdf/2603.13707)**

> **作者:** Zhaoyuan Gu; Yipu Chen; Zimeng Chai; Alfred Cueva; Thong Nguyen; Yifan Wu; Huishu Xue; Minji Kim; Isaac Legene; Fukang Liu; Matthew Kim; Ayan Barula; Yongxin Chen; Ye Zhao
>
> **摘要:** Humanoid loco-manipulation requires coordinated high-level motion plans with stable, low-level whole-body execution under complex robot-environment dynamics and long-horizon tasks. While diffusion policies (DPs) show promise for learning from demonstrations, deploying them on humanoids poses critical challenges: the motion planner trained offline is decoupled from the low-level controller, leading to poor command tracking, compounding distribution shift, and task failures. The common approach of scaling demonstration data is prohibitively expensive for high-dimensional humanoid systems. To address this challenge, we present REFINE-DP (REinforcement learning FINE-tuning of Diffusion Policy), a hierarchical framework that jointly optimizes a DP high-level planner and an RL-based low-level loco-manipulation controller. The DP is fine-tuned via a PPO-based diffusion policy gradient to improve task success rate, while the controller is simultaneously updated to accurately track the planner's evolving command distribution, reducing the distributional mismatch that degrades motion quality. We validate REFINE-DP on a humanoid robot performing loco-manipulation tasks, including door traversal and long-horizon object transport. REFINE-DP achieves an over $90\%$ success rate in simulation, even in out-of-distribution cases not seen in the pre-trained data, and enables smooth autonomous task execution in real-world dynamic environments. Our proposed method substantially outperforms pre-trained DP baselines and demonstrates that RL fine-tuning is key to reliable humanoid loco-manipulation. this https URL
>
---
#### [new 009] Towards Equitable Robotic Furnishing Agents for Aging-in-Place: ADL-Grounded Design Exploration
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决老年人日常活动困难问题。通过访谈与实验，提出一种具公平性的家庭机器人设计，以减轻认知与身体负担。**

- **链接: [https://arxiv.org/pdf/2603.14182](https://arxiv.org/pdf/2603.14182)**

> **作者:** Hansoo Lee; Changhee Seo; Subin Park; Sonya S. Kwak
>
> **备注:** Accepted at the ACM/IEEE International Conference on Human-Robot Interaction (HRI) 2026 Workshop: Equitable Robotics for Wellbeing (Eq-RW)
>
> **摘要:** In aging-in-place contexts, small difficulties in Activities of Daily Living (ADL) can accumulate, affecting well-being through fatigue, anxiety, reduced autonomy, and safety risks. This position paper argues that robotics for older adult wellbeing must move beyond "convenience features" and centre equity, justice, and responsibility. We conducted ADL-grounded semi-structured interviews with four adults in their 70s-80s, identifying recurrent challenges (finding/ organising items, taking medication, and transporting objects) and deriving requirements to reduce compounded cognitive-physical burden. Based on these insights, we propose an in-home robotic furnishing-agent concept leveraging computer vision and generative AI and LLMs for natural-language interaction, context-aware reminders, safe actuation, and user-centred transparency. We then report video-stimulated follow-up interviews with the same participants, highlighting preferences for confirmation before actuation, predictability, adjustable speed/autonomy, and multimodal feedback, as well as equity-related concerns. We conclude with open questions on evaluating and deploying equitable robotic wellbeing systems in real homes.
>
---
#### [new 010] Bots and Blocks: Presenting a project-based approach for robotics education
- **分类: cs.RO**

- **简介: 论文提出一种基于项目的机器人教育方法，旨在解决传统教学缺乏实践的问题。通过敏捷项目教授学生机器人编程与应用，提升其理论与实践能力。**

- **链接: [https://arxiv.org/pdf/2603.14529](https://arxiv.org/pdf/2603.14529)**

> **作者:** Tobias Geger; Dominique Briechle; Andreas Rausch
>
> **备注:** 12 pages, 3 figures, 23 references
>
> **摘要:** To prepare students for upcoming trends and challenges, it is important to teach them about the helpful and important aspects of modern technologies, such as robotics. However, classic study programs often fail to prepare students for working in the industry because of the lack of practical experience, caused by solely theoretical lecturing. The challenge is to teach both practical and theoretical skills interactively to improve the students' learning. In the scope of the paper, a project-based learning approach is proposed, where students are taught in an agile, semester-spanning project how to work with robots. This project is part of the applied computer science degree study program Digital Technologies. The paper presents the framework as well as an exemplary project featuring the development of a disassembly software ecosystem for hardware robots. In the project, the students are taught the programming of robots with the help of the Robot Operating System (ROS). To ensure the base qualifications, the students are taught in so-called schools, an interactive mix of lectures and exercises. At the beginning of the course, the basics of the technologies are covered, while the students work more and more in their team with the robot on a specific use case. The use case here is to automate the disassembly of build block assemblies.
>
---
#### [new 011] ToMPC: Task-oriented Model Predictive Control via ADMM for Safe Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出ToMPC框架，用于安全高效的机器人操作任务，解决开放空间中避障与交互问题，通过ADMM方法分解优化问题，提升操作效率与安全性。**

- **链接: [https://arxiv.org/pdf/2603.13944](https://arxiv.org/pdf/2603.13944)**

> **作者:** Xinyu Jia; Wenxin Wang; Jun Yang; Yongping Pan; Haoyong Yu
>
> **备注:** 8 pages, 10 figures, accepted by IEEE Robotics and Automation Letters (RAL)
>
> **摘要:** This paper proposes a task-oriented model predictive control (ToMPC) framework for safe and efficient robotic manipulation in open workspaces. The framework unifies collision-free motion and robot-environment interaction to address diverse scenarios. Additionally, it introduces task-oriented obstacle avoidance that leverages kinematic redundancy to enhance manipulation efficiency in obstructed environments. This complex optimization problem is solved by the alternating direction method of multipliers (ADMM), which decomposes the problem into two subproblems tackled by differential dynamic programming (DDP) and quadratic programming (QP), respectively. The effectiveness of this approach is validated in simulation and hardware experiments on a Franka Panda robotic manipulator. Results demonstrate that the framework can plan motion and/or force trajectories in real time, maximize the manipulation range while avoiding obstacles, and strictly adhere to safety-related hard constraints.
>
---
#### [new 012] TransDex: Pre-training Visuo-Tactile Policy with Point Cloud Reconstruction for Dexterous Manipulation of Transparent Objects
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在解决透明物体抓取中的自遮挡和深度信息丢失问题。提出TransDex方法，通过点云重建预训练实现视觉-触觉融合控制。**

- **链接: [https://arxiv.org/pdf/2603.13869](https://arxiv.org/pdf/2603.13869)**

> **作者:** Fengguan Li; Yifan Ma; Chen Qian; Wentao Rao; Weiwei Shang
>
> **备注:** Project page: this https URL
>
> **摘要:** Dexterous manipulation enables complex tasks but suffers from self-occlusion, severe depth noise, and depth information loss when manipulating transparent objects. To solve this problem, this paper proposes TransDex, a 3D visuo-tactile fusion motor policy based on point cloud reconstruction pre-training. Specifically, we first propose a self-supervised point cloud reconstruction pre-training approach based on Transformer. This method accurately recovers the 3D structure of objects from interactive point clouds of dexterous hands, even when random noise and large-scale masking are added. Building on this, TransDex is constructed in which perceptual encoding adopts a fine-grained hierarchical scheme and multi-round attention mechanisms adaptively fuse features of the robotic arm and dexterous hand to enable differentiated motion prediction. Results from transparent object manipulation experiments conducted on a real robotic system demonstrate that TransDex outperforms existing baseline methods. Further analysis validates the generalization capabilities of TransDex and the effectiveness of its individual components.
>
---
#### [new 013] From Scanning Guidelines to Action: A Robotic Ultrasound Agent with LLM-Based Reasoning
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决机器人超声扫描中依赖人工经验的问题。通过引入基于大语言模型的智能代理，实现自主、自适应的扫描流程。**

- **链接: [https://arxiv.org/pdf/2603.14393](https://arxiv.org/pdf/2603.14393)**

> **作者:** Yuan Bi; Yiping Zhou; Pei Liu; Feng Li; Zhongliang Jiang; Nassir Navab
>
> **备注:** Code: this https URL Video: this https URL
>
> **摘要:** Robotic ultrasound offers advantages over free-hand scanning, including improved reproducibility and reduced operator dependency. In clinical practice, US acquisition relies heavily on the sonographer's experience and situational judgment. When transferring this process to robotic systems, such expertise is often encoded explicitly through fixed procedures and task-specific models, yielding pipelines that can be difficult to adapt to new scanning tasks. In this work, we propose a unified framework for autonomous robotic US scanning that leverages a LLM-based agent to interpret US scanning guidelines and execute scans by dynamically invoking a set of provided software tools. Instead of encoding fixed scanning procedures, the LLM agent retrieves and reasons over guideline steps from scanning handbooks and adapts its planning decisions based on observations and the current scanning state. This enables the system to handle variable and decision-dependent workflows, such as adjusting scanning strategies, repeating steps, or selecting the appropriate next tool call in response to image quality or anatomical findings. Because the reasoning underlying tool selection is also critical for transparent and trustworthy planning, we further fine tune the LLM agent using a RL based strategy to improve both its reasoning quality and the correctness of tool selection and parameterization, while maintaining robust generalization to unseen guidelines and related tasks. We first validate the approach via verbal execution on 10 US scanning guidelines, assessing reasoning as well as tool selection and parameterization, and showing the benefit of RL fine tuning. We then demonstrate real world feasibility on robotic scanning of the gallbladder, spine, and kidney. Overall, the framework follows diverse guidelines and enables reliable autonomous scanning across multiple anatomical targets within a unified system.
>
---
#### [new 014] Learning Actionable Manipulation Recovery via Counterfactual Failure Synthesis
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决执行错误自主恢复问题。通过合成反事实失败轨迹，生成可用于训练的修复数据，提升机器人故障恢复能力。**

- **链接: [https://arxiv.org/pdf/2603.13528](https://arxiv.org/pdf/2603.13528)**

> **作者:** Dayou Li; Jiuzhou Lei; Hao Wang; Lulin Liu; Yunhao Yang; Zihan Wang; Bangya Liu; Minghui Zheng; Zhiwen Fan
>
> **摘要:** While recent foundation models have significantly advanced robotic manipulation, these systems still struggle to autonomously recover from execution errors. Current failure-learning paradigms rely on either costly and unsafe real-world data collection or simulator-based perturbations, which introduce a severe sim-to-real gap. Furthermore, existing visual analyzers predominantly output coarse, binary diagnoses rather than the executable, trajectory-level corrections required for actual recovery. To bridge the gap between failure diagnosis and actionable recovery, we introduce Dream2Fix, a framework that synthesizes photorealistic, counterfactual failure rollouts directly from successful real-world demonstrations. By perturbing actions within a generative world model, Dream2Fix creates paired failure-correction data without relying on simulators. To ensure the generated data is physically viable for robot learning, we implement a structured verification mechanism that strictly filters rollouts for task validity, visual coherence, and kinematic safety. This engine produces a high-fidelity dataset of over 120k paired samples. Using this dataset, we fine-tune a vision-language model to jointly predict failure types and precise recovery trajectories, mapping visual anomalies directly to corrective actions. Extensive real-world robotic experiments show our approach achieves state-of-the-art correction accuracy, improving from 19.7% to 81.3% over prior baselines, and successfully enables zero-shot closed-loop failure recovery in physical deployments.
>
---
#### [new 015] Learning from Mistakes: Post-Training for Driving VLA with Takeover Data
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，解决VLA模型在分布偏移下的安全性和性能问题。通过引入预接管语言监督和情景梦境强化学习，提升模型安全边际和驾驶表现。**

- **链接: [https://arxiv.org/pdf/2603.14972](https://arxiv.org/pdf/2603.14972)**

> **作者:** Yinfeng Gao; Deqing Liu; Qichao Zhang; Yupeng Zheng; Haochen Tian; Guang Li; Hangjun Ye; Long Chen; Da-Wei Ding; Dongbin Zhao
>
> **摘要:** Current Vision-Language-Action (VLA) paradigms in end-to-end autonomous driving rely on offline training from static datasets, leaving them vulnerable to distribution shift. Recent post-training methods use takeover data to mitigate this by augmenting the dataset with high-quality expert takeover samples, yet they suffer from two key limitations: supervision restricted to the period after the takeover moments leads to policies with limited safety margins, and passive preference optimization lacks active exploration for optimal performance. In this paper, we propose TakeVLA, a novel VLA post-training framework that overcomes these shortcomings through two complementary innovations. First, we introduce pre-takeover language supervision, which allows the VLA to learn from mistakes proactively. By explicitly teaching the model about what to do in error-prone situations, we cultivate a precautionary mindset that anticipates hazards early and substantially enlarges safety margins. Second, we propose Scenario Dreaming, a reinforcement fine-tuning paradigm that operates in reconstruceted takeover scenarios, encouraging active exploration beyond mere preference fitting. Experiments on the Bench2Drive benchmark demonstrate that TakeVLA achieves state-of-the-art closed-loop performance, surpassing the strong VLA baseline SimLingo by 4.93 in driving score, with an enhanced safety margin as evidenced by an 11.76% increase in average TTC.
>
---
#### [new 016] MorFiC: Fixing Value Miscalibration for Zero-Shot Quadruped Transfer
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，解决跨形态四足机器人零样本迁移问题。通过MorFiC方法，实现单策略在不同形态机器人上的有效迁移，提升稳定速度与运行距离。**

- **链接: [https://arxiv.org/pdf/2603.14554](https://arxiv.org/pdf/2603.14554)**

> **作者:** Prakhar Mishra; Amir Hossain Raj; Xuesu Xiao; Dinesh Manocha
>
> **摘要:** Generalizing learned locomotion policies across quadrupedal robots with different morphologies remain a challenge. Policies trained on a single robot often break when deployed on embodiments with different mass distributions, kinematics, joint limits, or actuation constraints, forcing per robot retraining. We present MorFiC, a reinforcement learning approach for zero-shot cross-morphology locomotion using a single shared policy. MorFiC resolves a key failure mode in multi-morphology actor-critic training: a shared critic tends to average incompatible value targets across embodiments, yielding miscalibrated advantages. To address this, MorFiC conditions the critic via morphology-aware modulation driven by robot physical and control parameters, generating morphology-specific value estimates within a shared network. Trained with a single source robot with morphology randomization in simulation, MorFiC can transfer to unseen robots and surpasses morphology-conditioned PPO baselines by improving stable average speed and longest stable run on multiple targets, including speed gains of +16.1% on A1, ~2x on Cheetah, and ~5x on B1. We additionally show that MorFiC reduces the value-prediction error variance across morphologies and stabilizes the advantage estimates, demonstrating that the improved value-function calibration corresponds to a stronger transfer performance. Finally, we demonstrate zero-shot deployment on two Unitree Go1 and Go2 robots without fine-tuning, indicating that critic-side conditioning is a practical approach for cross-morphology generalization.
>
---
#### [new 017] H-RINS: Hierarchical Tightly-coupled Radar-Inertial Navigation via Smoothing and Mapping
- **分类: cs.RO**

- **简介: 该论文属于导航定位任务，解决雷达-惯性系统易受漂移影响的问题。通过构建分层紧耦合框架，提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.14109](https://arxiv.org/pdf/2603.14109)**

> **作者:** Ali Alridha Abdulkarim; Mikhail Litvinov; Dzmitry Tsetserukou
>
> **备注:** 8 pages, 5 figures, Submitted to conference
>
> **摘要:** Millimeter-wave radar provides robust perception in visually degraded environments. However, radar-inertial state estimation is inherently susceptible to drift. Because radar yields only sparse, body-frame velocity measurements, it provides weak constraints on absolute orientation. Consequently, IMU biases remain poorly observable over the short time horizons typical of sliding-window filters. To address this fundamental observability challenge, we propose a tightly coupled, hierarchical radar-inertial factor graph framework. Our architecture decouples the estimation problem into a high-rate resetting graph and a persistent global graph. The resetting graph fuses IMU preintegration, radar velocities, and adaptive Zero-Velocity Updates (ZUPT) to generate the smooth, low-latency odometry required for real-time control. Concurrently, the persistent graph is a full-state factor graph maintaining the complete information of poses, velocities, and biases by fusing inertial data with keyframe-based geometric mapping and loop closures. Leveraging Incremental Smoothing and Mapping, the persistent graph can operate without explicit marginalization of variables, preserving their information while ensuring long-term bias observability. The cornerstone of our approach is a probabilistic tight-coupling mechanism: fully observable, optimized biases and their exact covariances are continuously injected from the persistent graph into the resetting graph's prior, effectively anchoring the high-rate estimator against integration drift. Extensive evaluations demonstrate our system achieves high accuracy with drift-reduced estimation at 27x real-time execution speeds. We release the implementation code and datasets upon the acceptance of the paper.
>
---
#### [new 018] BodyGuards: Escorting by Multiple Robots in Unknown Environment under Limited Communication
- **分类: cs.RO**

- **简介: 该论文提出BodyGuards框架，解决多机器人在受限通信下护送人类操作员的问题，通过协同探索与协调策略降低风险并缩短任务时间。**

- **链接: [https://arxiv.org/pdf/2603.15108](https://arxiv.org/pdf/2603.15108)**

> **作者:** Zhuoli Tian; Yanze Bao; Meng Guo
>
> **备注:** Accept by ICRA 2026
>
> **摘要:** Multi-robot systems are increasingly deployed in high-risk missions such as reconnaissance, disaster response, and subterranean operations. Protecting a human operator while navigating unknown and adversarial environments remains a critical challenge, especially when the communication among the operator and robots is restricted. Unlike existing collaborative exploration methods that aim for complete coverage, this work focuses on task-oriented exploration to minimize the navigation time of the operator to reach its goal while ensuring safety under adversarial threats. A novel escorting framework BodyGuards, is proposed to explicitly integrate seamlessly collaborative exploration, inter-robot-operator communication and escorting. The framework consists of three core components: (I) a dynamic movement strategy for the operator that maintains a local map with risk zones for proactive path planning; (II) a dual-mode robotic strategy combining frontier based exploration with optimized return events to balance exploration, threat detection, and intermittent communication; and (III) multi-robot coordination protocols that jointly plan exploration and information sharing for efficient escorting. Extensive human-in-the-loop simulations and hardware experiments demonstrate that the method significantly reduces operator risk and mission time, outperforming baselines in adversarial and constrained environments.
>
---
#### [new 019] Amortizing Trajectory Diffusion with Keyed Drift Fields
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于强化学习任务，旨在解决扩散模型轨迹规划计算成本高的问题。通过引入Keyed Drifting Policies（KDP），实现单步推理生成多样化轨迹，降低规划延迟。**

- **链接: [https://arxiv.org/pdf/2603.14056](https://arxiv.org/pdf/2603.14056)**

> **作者:** Gokul Puthumanaillam; Melkior Ornik
>
> **摘要:** Diffusion-based trajectory planners can synthesize rich, multimodal action sequences for offline reinforcement learning, but their iterative denoising incurs substantial inference-time cost, making closed-loop planning slow under tight compute budgets. We study the problem of achieving diffusion-like trajectory planning behavior with one-step inference, while retaining the ability to sample diverse candidate plans and condition on the current state in a receding-horizon control loop. Our key observation is that conditional trajectory generation fails under naïve distribution-matching objectives when the similarity measure used to align generated trajectories with the dataset is dominated by unconstrained future dimensions. In practice, this causes attraction toward average trajectories, collapses action diversity, and yields near-static behavior. Our key insight is that conditional generative planning requires a conditioning-aware notion of neighborhood: trajectory updates should be computed using distances in a compact key space that reflects the condition, while still applying updates in the full trajectory space. Building on this, we introduce Keyed Drifting Policies (KDP), a one-step trajectory generator trained with a drift-field objective that attracts generated trajectories toward condition-matched dataset windows and repels them from nearby generated samples, using a stop-gradient drifted target to amortize iterative refinement into training. At inference, the resulting policy produces a full trajectory window in a single forward pass. Across standard RL benchmarks and real-time hardware deployments, KDP achieves strong performance with one-step inference and substantially lower planning latency than diffusion sampling. Project website, code and videos: this https URL
>
---
#### [new 020] STL-SVPIO: Signal Temporal Logic guided Stein Variational Path Integral Optimization
- **分类: cs.RO**

- **简介: 该论文提出STL-SVPIO方法，用于解决复杂时空约束下的机器人轨迹规划问题，通过将STL转化为可微奖励机制，提升长时序任务的鲁棒性和效率。**

- **链接: [https://arxiv.org/pdf/2603.13333](https://arxiv.org/pdf/2603.13333)**

> **作者:** Hongrui Zheng; Zirui Zang; Ahmad Amine; Cristian Ioan Vasile; Rahul Mangharam
>
> **摘要:** Signal Temporal Logic (STL) enables formal specification of complex spatiotemporal constraints for robotic task planning. However, synthesizing long-horizon continuous control trajectories from complex STL specifications is fundamentally challenging due to the nested structure of STL robustness objectives. Existing solver-based methods, such as Mixed-Integer Linear Programming (MILP), suffer from exponential scaling, whereas sampling methods, such as Model-Predictive Path Integral control (MPPI), struggle with sparse, long-horizon costs. We introduce Signal Temporal Logic guided Stein Variational Path Integral Optimization (STL-SVPIO), which reframes STL as a globally informative, differentiable reward-shaping mechanism. By leveraging Stein Variational Gradient Descent and differentiable physics engines, STL-SVPIO transports a mutually repulsive swarm of control particles toward high robustness regions. Our method transforms sparse logical satisfaction into tractable variational inference, mitigating the severe local minima traps of standard gradient-based methods. We demonstrate that STL-SVPIO significantly outperforms existing methods in both robustness and efficiency for traditional STL tasks. Moreover, it solves complex long-horizon tasks, including multi-agent coordination with synchronization and queuing while baselines either fail to discover feasible solutions, or become computationally intractable. Finally, we use STL-SVPIO in agile robotic motion planning tasks with nonlinear dynamics, such as 7-DoF manipulation and half cheetah back flips to show the generalizability of our algorithm.
>
---
#### [new 021] Coordinate-Independent Robot Model Identification
- **分类: cs.RO**

- **简介: 该论文属于机器人建模任务，旨在解决传统方法依赖坐标系的问题。通过引入双度量权重，实现坐标无关的模型识别，提升精度。**

- **链接: [https://arxiv.org/pdf/2603.14656](https://arxiv.org/pdf/2603.14656)**

> **作者:** Yanhao Yang; Ross L. Hatton
>
> **备注:** 8 pages, 7 figures, supplementary video: this https URL
>
> **摘要:** Robot model identification is commonly performed by least-squares regression on inverse dynamics, but existing formulations measure residuals directly in coordinate force space and therefore depend on the chosen coordinate chart, units, and scaling. This paper proposes a coordinate-independent identification method that weights inverse-dynamics residuals by the dual metric induced by the system Riemannian metric. Using the force--velocity vector--covector duality, the dual metric provides a physically meaningful normalization of generalized forces, pulling coordinate residuals back into the ambient mechanical space and eliminating coordinate-induced bias. The resulting objective remains convex through an affine-metric and Schur-complement reformulation, and is compatible with physical-consistency constraints and geometric regularization. Experiments on an inertia-dominated Crazyflie--pendulum system and a drag-dominated LandSalp robot show improved identification accuracy, especially on shape coordinates, in both low-data and high-data settings.
>
---
#### [new 022] KiRAS: Keyframe Guided Self-Imitation for Robust and Adaptive Skill Learning in Quadruped Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人技能学习任务，旨在解决复杂地形下多技能泛化能力不足的问题。通过关键帧引导的自模仿方法，实现技能的鲁棒获取与平滑切换。**

- **链接: [https://arxiv.org/pdf/2603.15179](https://arxiv.org/pdf/2603.15179)**

> **作者:** Xiaoyi Wei; Peng Zhai; Jiaxin Tu; Yueqi Zhang; Yuqi Li; Zonghao Zhang; Hu Zhou; Lihua Zhang
>
> **备注:** Received by 2026 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** With advances in reinforcement learning and imitation learning, quadruped robots can acquire diverse skills within a single policy by imitating multiple skill-specific datasets. However, the lack of datasets on complex terrains limits the ability of such multi-skill policies to generalize effectively in unstructured environments. Inspired by animation, we adopt keyframes as minimal and universal skill representations, relaxing dataset constraints and enabling the integration of terrain adaptability with skill diversity. We propose Keyframe Guided Self-Imitation for Robust and Adaptive Skill Learning (KiRAS), an end-to-end framework for acquiring and transitioning between diverse skill primitives on complex terrains. KiRAS first learns diverse skills on flat terrain through keyframe-guided self-imitation, eliminating the need for expert datasets; then continues training the same policy network on rough terrains to enhance robustness. To eliminate catastrophic forgetting, a proficiency-based Skill Initialization Technique is introduced. Experiments on Solo-8 and Unitree Go1 robots show that KiRAS enables robust skill acquisition and smooth transitions across challenging terrains. This framework demonstrates its potential as a lightweight platform for multi-skill generation and dataset collection. It further enables flexible skill transitions that enhance locomotion on challenging terrains.
>
---
#### [new 023] MRPoS: Mixed Reality-Based Robot Navigation Interface Using Spatial Pointing and Speech with Large Language Model
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决传统手势操作机器人导航的不便。提出MRPoS系统，结合空间指针和语音交互，提升操作效率与便捷性。**

- **链接: [https://arxiv.org/pdf/2603.13313](https://arxiv.org/pdf/2603.13313)**

> **作者:** Eduardo Iglesius; Masato Kobayashi; Yuki Uranishi
>
> **摘要:** Recent advancements have made robot navigation more intuitive by transitioning from traditional 2D displays to spatially aware Mixed Reality (MR) systems. However, current MR interfaces often rely on manual "air tap" gestures for goal placement, which can be repetitive and physically demanding, especially for beginners. This paper proposes the Mixed Reality-Based Robot Navigation Interface using Spatial Pointing and Speech (MRPoS). This novel framework replaces complex hand gestures with a natural, multimodal interface combining spatial pointing with Large Language Model (LLM)-based speech interaction. By leveraging both information, the system translates verbal intent into navigation goals visualized by MR technology. Comprehensive experiments comparing MRPoS against conventional gesture-based systems demonstrate that our approach significantly reduces task completion time and workload, providing a more accessible and efficient interface. For additional material, please check: this https URL
>
---
#### [new 024] End-to-End O-RAN Testbed for Edge-AI-Enabled 5G/6G Connected Industrial Robotics
- **分类: cs.RO; cs.NI**

- **简介: 该论文属于工业机器人与5G/6G融合任务，旨在解决边缘AI服务在工业机器人中的应用问题，构建了基于O-RAN的端到端测试平台，并通过自主焊接场景验证了相关技术。**

- **链接: [https://arxiv.org/pdf/2603.13567](https://arxiv.org/pdf/2603.13567)**

> **作者:** Sasa Talosi; Vladimir Vincan; Srdjan Sobot; Goran Martic; Vladimir Morosev; Vukan Ninkovic; Dragisa Miskovic; Dejan Vukobratovic
>
> **备注:** Submitted to Global 6G Conference 2026
>
> **摘要:** Connected robotics is one of the principal use cases driving the transition towards more intelligent and capable 6G mobile cellular networks. Replacing wired connections with highly reliable, high-throughput, and low-latency 5G/6G radio interfaces enables robotic system mobility and the offloading of compute-intensive artificial intelligence (AI) models for robotic perception and control to servers located at the network edge. The transition towards Edge AI as a Service (E-AIaaS) simplifies on-site maintenance of robotic systems and reduces operational costs in industrial environments, while supporting flexible AI model life-cycle management and seamless upgrades of robotic functionalities over time. In this paper, we present a 5G/6G O-RAN-based end-to-end testbed that integrates E-AIaaS for connected industrial robotic applications. The objective is to design and deploy a generic experimental platform based on open technologies and interfaces, demonstrated through an E-AIaaS-enabled autonomous welding scenario. Within this scenario, the testbed is used to investigate trade-offs among different data acquisition, edge processing, and real-time streaming approaches for robotic perception, while supporting emerging paradigms such as semantic and goal-oriented communications.
>
---
#### [new 025] OCRA: Object-Centric Learning with 3D and Tactile Priors for Human-to-Robot Action Transfer
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出OCRA框架，用于从人类视频演示中学习机器人操作。解决人机动作迁移问题，通过融合3D和触觉信息提升操作鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.14401](https://arxiv.org/pdf/2603.14401)**

> **作者:** Kuanning Wang; Ke Fan; Yuqian Fu; Siyu Lin; Hu Luo; Daniel Seita; Yanwei Fu; Yu-Gang Jiang; Xiangyang Xue
>
> **备注:** Project page: this https URL
>
> **摘要:** We present OCRA, an Object-Centric framework for video-based human-to-Robot Action transfer that learns directly from human demonstration videos to enable robust manipulation. Object-centric learning emphasizes task-relevant objects and their interactions while filtering out irrelevant background, providing a natural and scalable way to teach robots. OCRA leverages multi-view RGB videos, the state-of-the-art 3D foundation model VGGT, and advanced detection and segmentation models to reconstruct object-centric 3D point clouds, capturing rich interactions between objects. To handle properties not easily perceived by vision alone, we incorporate tactile priors via a large-scale dataset of over one million tactile images. These 3D and tactile priors are fused through a multimodal module (ResFiLM) and fed into a Diffusion Policy to generate robust manipulation actions. Extensive experiments on both vision-only and visuo-tactile tasks show that OCRA significantly outperforms existing baselines and ablations, demonstrating its effectiveness for learning from human demonstration videos.
>
---
#### [new 026] End-to-End Dexterous Grasp Learning from Single-View Point Clouds via a Multi-Object Scene Dataset
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决多物体场景下灵巧抓取的问题。提出DGS-Net网络和数据集，提升抓取成功率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.15410](https://arxiv.org/pdf/2603.15410)**

> **作者:** Tao Geng; Dapeng Yang; Ziwei Liu; Le Zhang; Le Qi; WangYang Li; Yi Ren; Shan Luo; Fenglei Ni
>
> **备注:** 10 pages, 6 figures. Submitted to IEEE Transactions on Automation Science and Engineering (T-ASE)
>
> **摘要:** Dexterous grasping in multi-object scene constitutes a fundamental challenge in robotic manipulation. Current mainstream grasping datasets predominantly focus on single-object scenarios and predefined grasp configurations, often neglecting environmental interference and the modeling of dexterous pre-grasp gesture, thereby limiting their generalizability in real-world applications. To address this, we propose DGS-Net, an end-to-end grasp prediction network capable of learning dense grasp configurations from single-view point clouds in multi-object scene. Furthermore, we propose a two-stage grasp data generation strategy that progresses from dense single-object grasp synthesis to dense scene-level grasp generation. Our dataset comprises 307 objects, 240 multi-object scenes, and over 350k validated grasps. By explicitly modeling grasp offsets and pre-grasp configurations, the dataset provides more robust and accurate supervision for dexterous grasp learning. Experimental results show that DGS-Net achieves grasp success rates of 88.63\% in simulation and 78.98\% on a real robotic platform, while exhibiting lower penetration with a mean penetration depth of 0.375 mm and penetration volume of 559.45 mm^3, outperforming existing methods and demonstrating strong effectiveness and generalization capability. Our dataset is available at this https URL.
>
---
#### [new 027] Seeing Where to Deploy: Metric RGB-Based Traversability Analysis for Aerial-to-Ground Hidden Space Inspection
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于无人机与地面机器人协同任务，解决隐蔽空间部署区域识别问题。通过RGB重建和语义分析，构建可 traversability 地图，实现可靠部署。**

- **链接: [https://arxiv.org/pdf/2603.14639](https://arxiv.org/pdf/2603.14639)**

> **作者:** Seoyoung Lee; Shaekh Mohammad Shithil; Durgakant Pushp; Lantao Liu; Zhangyang Wang
>
> **摘要:** Inspection of confined infrastructure such as culverts often requires accessing hidden spaces whose entrances are reachable primarily from elevated viewpoints. Aerial-ground cooperation enables a UAV to deploy a compact UGV for interior exploration, but selecting a suitable deployment region from aerial observations requires metric terrain reasoning involving scale ambiguity, reconstruction uncertainty, and terrain semantics. We present a metric RGB-based geometric-semantic reconstruction and traversability analysis framework for aerial-to-ground hidden space inspection. A feed-forward multi-view RGB reconstruction backbone produces dense geometry, while temporally consistent semantic segmentation yields a 3D semantic map. To enable deployment-relevant measurements without LiDAR-based dense mapping, we introduce an embodied motion prior that recovers metric scale by enforcing consistency between predicted camera motion and onboard platform egomotion. From the metrically grounded reconstruction, we construct a confidence-aware geometric-semantic traversability map and evaluate candidate deployment zones under explicit reachability constraints. Experiments on a tethered UAV-UGV platform demonstrate reliable deployment-zone identification in hidden space scenarios.
>
---
#### [new 028] SmoothVLA: Aligning Vision-Language-Action Models with Physical Constraints via Intrinsic Smoothness Optimization
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操控任务，旨在解决VLA模型在训练中稳定性与探索性的矛盾。提出SmoothVLA框架，通过内在奖励优化提升运动平滑性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.13925](https://arxiv.org/pdf/2603.13925)**

> **作者:** Jiashun Li; Xiaoyu Shi; Hong Xie; Mingsheng Shang; Yun Lu
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a powerful paradigm for robotic manipulation. However, existing post-training methods face a dilemma between stability and exploration: Supervised Fine-Tuning (SFT) is constrained by demonstration quality and lacks generalization, whereas Reinforcement Learning (RL) improves exploration but often induces erratic, jittery trajectories that violate physical constraints. To bridge this gap, we propose SmoothVLA, a novel reinforcement learning fine-tuning framework that synergistically optimizes task performance and motion smoothness. The technical core is a physics-informed hybrid reward function that integrates binary sparse task rewards with a continuous dense term derived from trajectory jerk. Crucially, this reward is intrinsic, that computing directly from policy rollouts, without requiring extrinsic environment feedback or laborious reward engineering. Leveraging the Group Relative Policy Optimization (GRPO), SmoothVLA establishes trajectory smoothness as an explicit optimization prior, guiding the model toward physically feasible and stable control. Extensive experiments on the LIBERO benchmark demonstrate that SmoothVLA outperforms standard RL by 13.8\% in smoothness and significantly surpasses SFT in generalization across diverse tasks. Our work offers a scalable approach to aligning VLA models with physical-world constraints through intrinsic reward optimization.
>
---
#### [new 029] MoE-ACT: Scaling Multi-Task Bimanual Manipulation with Sparse Language-Conditioned Mixture-of-Experts Transformers
- **分类: cs.RO**

- **简介: 该论文提出MoE-ACT框架，解决多任务机械臂操作中的任务干扰问题。通过引入稀疏专家模块和语言条件机制，提升多任务泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.15265](https://arxiv.org/pdf/2603.15265)**

> **作者:** Kangjun Guo; Haichao Liu; Yanji Sun; Ruhan Zhao; Jinni Zhou; Jun Ma
>
> **摘要:** The ability of robots to handle multiple tasks under a unified policy is critical for deploying embodied intelligence in real-world household and industrial applications. However, out-of-distribution variation across tasks often causes severe task interference and negative transfer when training general robotic policies. To address this challenge, we propose a lightweight multi-task imitation learning framework for bimanual manipulation, termed Mixture-of-Experts-Enhanced Action Chunking Transformer (MoE-ACT), which integrates sparse Mixture-of-Experts (MoE) modules into the Transformer encoder of ACT. The MoE layer decomposes a unified task policy into independently invoked expert components. Through adaptive activation, it naturally decouples multi-task action distributions in latent space. During decoding, Feature-wise Linear Modulation (FiLM) dynamically modulates action tokens to improve consistency between action generation and task instructions. In parallel, multi-scale cross-attention enables the policy to simultaneously focus on both low-level and high-level semantic features, providing rich visual information for robotic manipulation. We further incorporate textual information, transitioning the framework from a purely vision-based model to a vision-centric, language-conditioned action generation system. Experimental validation in both simulation and a real-world dual-arm setup shows that MoE-ACT substantially improves multi-task performance. Specifically, MoE-ACT outperforms vanilla ACT by an average of 33% in success rate. These results indicate that MoE-ACT provides stronger robustness and generalization in complex multi-task bimanual manipulation environments. Our open-source project page can be found at this https URL.
>
---
#### [new 030] LiDAR-EVS: Enhance Extrapolated View Synthesis for 3D Gaussian Splatting with Pseudo-LiDAR Supervision
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出LiDAR-EVS，解决自动驾驶中LiDAR模拟在非训练轨迹上的泛化问题，通过伪LiDAR监督和空间正则化提升模拟效果。**

- **链接: [https://arxiv.org/pdf/2603.14763](https://arxiv.org/pdf/2603.14763)**

> **作者:** Yiming Huang; Xin Kang; Sipeng Zhang; Hongliang Ren; Weihua Zhang; Junjie Lai
>
> **备注:** 22 pages, 8 figures
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time LiDAR and camera synthesis in autonomous driving simulation. However, simulating LiDAR with 3DGS remains challenging for extrapolated views beyond the training trajectory, as existing methods are typically trained on single-traversal sensor scans, suffer from severe overfitting and poor generalization to novel ego-vehicle paths. To enable reliable simulation of LiDAR along unseen driving trajectories without external multi-pass data, we present LiDAR-EVS, a lightweight framework for robust extrapolated-view LiDAR simulation in autonomous driving. Designed to be plug-and-play, LiDAR-EVS readily extends to diverse LiDAR sensors and neural rendering baselines with minimal modification. Our framework comprises two key components: (1) pseudo extrapolated-view point cloud supervision with multi-frame LiDAR fusion, view transformation, occlusion curling, and intensity adjustment; (2) spatially-constrained dropout regularization that promotes robustness to diverse trajectory variations encountered in real-world driving. Extensive experiments demonstrate that LiDAR-EVS achieves SOTA performance on extrapolated-view LiDAR synthesis across three datasets, making it a promising tool for data-driven simulation, closed-loop evaluation, and synthetic data generation in autonomous driving systems.
>
---
#### [new 031] Physics-Informed Policy Optimization via Analytic Dynamics Regularization
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决机器人控制中策略样本效率低和物理不一致的问题。通过引入物理约束的正则化项，提升策略优化的效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.14469](https://arxiv.org/pdf/2603.14469)**

> **作者:** Namai Chandra; Liu Mohan; Zhihao Gu; Lin Wang
>
> **备注:** 11 pages, 8 figures. Submitted to ICML 2026
>
> **摘要:** Reinforcement learning (RL) has achieved strong performance in robotic control; however, state-of-the-art policy learning methods, such as actor-critic methods, still suffer from high sample complexity and often produce physically inconsistent actions. This limitation stems from neural policies implicitly rediscovering complex physics from data alone, despite accurate dynamics models being readily available in simulators. In this paper, we introduce a novel physics-informed RL framework, called PIPER, that seamlessly integrates physical constraints directly into neural policy optimization with analytical soft physics constraints. At the core of our method is the integration of a differentiable Lagrangian residual as a regularization term within the actor's objective. This residual, extracted from a robot's simulator description, subtly biases policy updates towards dynamically consistent solutions. Crucially, this physics integration is realized through an additional loss term during policy optimization, requiring no alterations to existing simulators or core RL algorithms. Extensive experiments demonstrate that our method significantly improves learning efficiency, stability, and control accuracy, establishing a new paradigm for efficient and physically consistent robotic control.
>
---
#### [new 032] User-Tailored Learning to Forecast Walking Modes for Exosuits
- **分类: cs.RO**

- **简介: 该论文属于行为识别任务，旨在解决外骨骼服装中用户行走模式的感知问题。通过惯性数据实现三种行走模式的估计，并支持在线模型自适应。**

- **链接: [https://arxiv.org/pdf/2603.15329](https://arxiv.org/pdf/2603.15329)**

> **作者:** Gabriele Abbate; Enrica Tricomi; Nathalie Gierden; Alessandro Giusti; Lorenzo Masia; Antonio Paolillo
>
> **摘要:** Assistive robotic devices, like soft lower-limb exoskeletons or exosuits, are widely spreading with the promise of helping people in everyday life. To make such systems adaptive to the variety of users wearing them, it is desirable to endow exosuits with advanced perception systems. However, exosuits have little sensory equipment because they need to be light and easy to wear. This paper presents a perception module based on machine learning that aims at estimating 3 walking modes (i.e., ascending or descending stairs and walking on level ground) of users wearing an exosuit. We tackle this perception problem using only inertial data from two sensors. Our approach provides an estimate for both future and past timesteps that supports control and enables a self-labeling procedure for online model adaptation. Indeed, we show that our estimate can label data acquired online and refine the model for new users. A thorough analysis carried out on real-life datasets shows the effectiveness of our user-tailored perception module. Finally, we integrate our system with the exosuit in a closed-loop controller, validating its performance in an online single-subject experiment.
>
---
#### [new 033] VIP-Loco: A Visually Guided Infinite Horizon Planning Framework for Legged Locomotion
- **分类: cs.RO**

- **简介: 该论文提出VIP-Loco框架，解决腿部机器人在复杂环境中的感知运动问题。融合视觉与强化学习，实现高效、鲁棒的运动规划。**

- **链接: [https://arxiv.org/pdf/2603.14345](https://arxiv.org/pdf/2603.14345)**

> **作者:** Aditya Shirwatkar; Satyam Gupta; Shishir Kolathaya
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Perceptive locomotion for legged robots requires anticipating and adapting to complex, dynamic environments. Model Predictive Control (MPC) serves as a strong baseline, providing interpretable motion planning with constraint enforcement, but struggles with high-dimensional perceptual inputs and rapidly changing terrain. In contrast, model-free Reinforcement Learning (RL) adapts well across visually challenging scenarios but lacks planning. To bridge this gap, we propose VIP-Loco, a framework that integrates vision-based scene understanding with RL and planning. During training, an internal model maps proprioceptive states and depth images into compact kinodynamic features used by the RL policy. At deployment, the learned models are used within an infinite-horizon MPC formulation, combining adaptability with structured planning. We validate VIP-Loco in simulation on challenging locomotion tasks, including slopes, stairs, crawling, tilting, gap jumping, and climbing, across three robot morphologies: a quadruped (Unitree Go1), a biped (Cassie), and a wheeled-biped (TronA1-W). Through ablations and comparisons with state-of-the-art methods, we show that VIP-Loco unifies planning and perception, enabling robust, interpretable locomotion in diverse environments.
>
---
#### [new 034] ImagiNav: Scalable Embodied Navigation via Generative Visual Prediction and Inverse Dynamics
- **分类: cs.RO**

- **简介: 该论文提出ImagiNav，解决开放世界中机器人通过自然语言导航的问题。通过分解指令、生成视频轨迹和逆动力学模型，实现无需机器人演示的零样本迁移导航。**

- **链接: [https://arxiv.org/pdf/2603.13833](https://arxiv.org/pdf/2603.13833)**

> **作者:** Jie Chen; Yuxin Cai; Yizhuo Wang; Ruofei Bai; Yuhong Cao; Jun Li; Yau Wei Yun; Guillaume Sartoretti
>
> **摘要:** Enabling robots to navigate open-world environments via natural language is critical for general-purpose autonomy. Yet, Vision-Language Navigation has relied on end-to-end policies trained on expensive, embodiment-specific robot data. While recent foundation models trained on vast simulation data show promise, the challenge of scaling and generalizing due to the limited scene diversity and visual fidelity in simulation persists. To address this gap, we propose ImagiNav, a novel modular paradigm that decouples visual planning from robot actuation, enabling the direct utilization of diverse in-the-wild navigation videos. Our framework operates as a hierarchy: a Vision-Language Model first decomposes instructions into textual subgoals; a finetuned generative video model then imagines the future video trajectory towards that subgoal; finally, an inverse dynamics model extracts the trajectory from the imagined video, which can then be tracked via a low-level controller. We additionally develop a scalable data pipeline of in-the-wild navigation videos auto-labeled via inverse dynamics and a pretrained Vision-Language Model. ImagiNav demonstrates strong zero-shot transfer to robot navigation without requiring robot demonstrations, paving the way for generalist robots that learn navigation directly from unlabeled, open-world data.
>
---
#### [new 035] Beyond Binary Success: Sample-Efficient and Statistically Rigorous Robot Policy Comparison
- **分类: cs.RO; stat.AP**

- **简介: 该论文属于机器人策略评估任务，解决真实世界测试资源不足的问题。提出一种高效、统计严谨的策略比较框架，支持多种评估指标，减少评估负担。**

- **链接: [https://arxiv.org/pdf/2603.13616](https://arxiv.org/pdf/2603.13616)**

> **作者:** David Snyder; Apurva Badithela; Nikolai Matni; George Pappas; Anirudha Majumdar; Masha Itkina; Haruki Nishimura
>
> **备注:** 12 + 9 pages, 2 + 5 figures,
>
> **摘要:** Generalist robot manipulation policies are becoming increasingly capable, but are limited in evaluation to a small number of hardware rollouts. This strong resource constraint in real-world testing necessitates both more informative performance measures and reliable and efficient evaluation procedures to properly assess model capabilities and benchmark progress in the field. This work presents a novel framework for robot policy comparison that is sample-efficient, statistically rigorous, and applicable to a broad set of evaluation metrics used in practice. Based on safe, anytime-valid inference (SAVI), our test procedure is sequential, allowing the evaluator to stop early when sufficient statistical evidence has accumulated to reach a decision at a pre-specified level of confidence. Unlike previous work developed for binary success, our unified approach addresses a wide range of informative metrics: from discrete partial credit task progress to continuous measures of episodic reward or trajectory smoothness, spanning both parametric and nonparametric comparison problems. Through extensive validation on simulated and real-world evaluation data, we demonstrate up to 70% reduction in evaluation burden compared to standard batch methods and up to 50% reduction compared to state-of-the-art sequential procedures designed for binary outcomes, with no loss of statistical rigor. Notably, our empirical results show that competing policies can be separated more quickly when using fine-grained task progress than binary success metrics.
>
---
#### [new 036] CORAL: COntextual Reasoning And Local Planning in A Hierarchical VLM Framework for Underwater Monitoring
- **分类: cs.RO**

- **简介: 该论文属于水下监测任务，解决AUV导航语义不足、碰撞多和路径误差问题。提出CORAL框架，分离高层语义与低层控制，提升覆盖率并减少碰撞。**

- **链接: [https://arxiv.org/pdf/2603.14786](https://arxiv.org/pdf/2603.14786)**

> **作者:** Zhenqi Wu; Yuanjie Lu; Xuesu Xiao; Xiaomin Lin
>
> **备注:** Submitted to IROS 2026
>
> **摘要:** Oyster reefs are critical ecosystem species that sustain biodiversity, filter water, and protect coastlines, yet they continue to decline globally. Restoring these ecosystems requires regular underwater monitoring to assess reef health, a task that remains costly, hazardous, and limited when performed by human divers. Autonomous underwater vehicles (AUVs) offer a promising alternative, but existing AUVs rely on geometry-based navigation that cannot interpret scene semantics. Recent vision-language models (VLMs) enable semantic reasoning for intelligent exploration, but existing VLM-driven systems adopt an end-to-end paradigm, introducing three key limitations. First, these systems require the VLM to generate every navigation decision, forcing frequent waits for inference. Second, VLMs cannot model robot dynamics, causing collisions in cluttered environments. Third, limited self-correction allows small deviations to accumulate into large path errors. To address these limitations, we propose CORAL, a framework that decouples high-level semantic reasoning from low-level reactive control. The VLM provides high-level exploration guidance by selecting waypoints, while a dynamics-based planner handles low-level collision-free execution. A geometric verification module validates waypoints and triggers replanning when needed. Compared with the previous state-of-the-art, CORAL improves coverage by 14.28% percentage points, or 17.85% relatively, reduces collisions by 100%, and requires 57% fewer VLM calls.
>
---
#### [new 037] GNIO: Gated Neural Inertial Odometry
- **分类: cs.RO**

- **简介: 该论文提出GNIO，解决惯性导航中的快速漂移问题。通过建模运动有效性与上下文，结合运动库和门控预测头，提升定位精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.15281](https://arxiv.org/pdf/2603.15281)**

> **作者:** Dapeng Feng; Yizhen Yin; Zhiqiang Chen; Yuhua Qi; Hongbo Chen
>
> **备注:** Submitted to IEEE Robotics and Automation Letters
>
> **摘要:** Inertial navigation using low-cost MEMS sensors is plagued by rapid drift due to sensor noise and bias instability. While recent data-driven approaches have made significant strides, they often struggle with micro-drifts during stationarity and mode fusion during complex motion transitions due to their reliance on fixed-window regression. In this work, we introduce Gated Neural Inertial Odometry (GNIO), a novel learning-based framework that explicitly models motion validity and context. We propose two key architectural innovations: \ding{182} a learnable Motion Bank that queries a global dictionary of motion patterns to provide semantic context beyond the local receptive field, and \ding{183} a Gated Prediction Head that decomposes displacement into magnitude and direction. This gating mechanism acts as a soft, differentiable Zero-Velocity Update (ZUPT), dynamically suppressing sensor noise during stationary periods while scaling predictions during dynamic motion. Extensive experiments across four public benchmarks demonstrate that GNIO significantly reduces position drift compared to state-of-the-art CNN and Transformer-based baselines. Notably, GNIO achieves a $60.21\%$ reduction in trajectory error on the OxIOD dataset and exhibits superior generalization in challenging scenarios involving frequent stops and irregular motion speeds.
>
---
#### [new 038] Towards Versatile Opti-Acoustic Sensor Fusion and Volumetric Mapping
- **分类: cs.RO**

- **简介: 该论文属于水下导航任务，解决浑浊环境中3D地图构建问题。融合声呐与相机数据，提升地图精度与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.14457](https://arxiv.org/pdf/2603.14457)**

> **作者:** Ivana Collado-Gonzalez; John McConnell; Brendan Englot
>
> **备注:** To appear at ICRA 2026 in Vienna, Austria
>
> **摘要:** Accurate 3D volumetric mapping is critical for autonomous underwater vehicles operating in obstacle-rich environments. Vision-based perception provides high-resolution data but fails in turbid conditions, while sonar is robust to lighting and turbidity but suffers from low resolution and elevation ambiguity. This paper presents a volumetric mapping framework that fuses a stereo sonar pair with a monocular camera to enable safe navigation under varying visibility conditions. Overlapping sonar fields of view resolve elevation ambiguity, producing fully defined 3D point clouds at each time step. The framework identifies regions of interest in camera images, associates them with corresponding sonar returns, and combines sonar range with camera-derived elevation cues to generate additional 3D points. Each 3D point is assigned a confidence value reflecting its reliability. These confidence-weighted points are fused using a Gaussian Process Volumetric Mapping framework that prioritizes the most reliable measurements. Experimental comparisons with other opti-acoustic and sonar-based approaches, along with field tests in a marina environment, demonstrate the method's effectiveness in capturing complex geometries and preserving critical information for robot navigation in both clear and turbid conditions. Our code is open-source to support community adoption.
>
---
#### [new 039] Data-Driven Physics Embedded Dynamics with Predictive Control and Reinforcement Learning for Quadrupeds
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于四足机器人运动控制任务，解决长期误差累积和可解释性差的问题，通过融合LNN与RL MPC提升动力学一致性与计算效率。**

- **链接: [https://arxiv.org/pdf/2603.14333](https://arxiv.org/pdf/2603.14333)**

> **作者:** Prakrut Kotecha; Aditya Shirwatkar; Shishir Kolathaya
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** State of the art quadrupedal locomotion approaches integrate Model Predictive Control (MPC) with Reinforcement Learning (RL), enabling complex motion capabilities with planning and terrain adaptive behaviors. However, they often face compounding errors over long horizons and have limited interpretability due to the absence of physical inductive biases. We address these issues by integrating Lagrangian Neural Networks (LNNs) into an RL MPC framework, enabling physically consistent dynamics learning. At deployment, our inverse dynamics infinite horizon MPC scheme avoids costly matrix inversions, improving computational efficiency by up to 4x with minimal loss of task performance. We validate our framework through multiple ablations of the proposed LNN and its variants. We show improved sample efficiency, reduced long-horizon error, and faster real time planning compared to unstructured neural dynamics. Lastly, we also test our framework on the Unitree Go1 robot to show real world viability.
>
---
#### [new 040] Load-Aware Locomotion Control for Humanoid Robots in Industrial Transportation Tasks
- **分类: cs.RO**

- **简介: 该论文研究工业场景下人形机器人负载运输任务，解决负载变化下的稳定行走问题。提出一种基于强化学习的负载感知控制框架，实现精准高度跟踪与稳定运动。**

- **链接: [https://arxiv.org/pdf/2603.14308](https://arxiv.org/pdf/2603.14308)**

> **作者:** Lequn Fu; Yijun Zhong; Xiao Li; Yibin Liu; Zhiyuan Xu; Jian Tang; Shiqi Li
>
> **备注:** This work has been submitted to the IEEE Transactions on Industrial Electronics for possible publication
>
> **摘要:** Humanoid robots deployed in industrial environments are required to perform load-carrying transportation tasks that tightly couple locomotion and manipulation. However, achieving stable and robust locomotion under varying payloads and upper-body motions is challenging due to dynamic coupling and partial observability. This paper presents a load-aware locomotion framework for industrial humanoids based on a decoupled yet coordinated loco-manipulation architecture. Lower-body locomotion is controlled via a reinforcement learning policy producing residual joint actions on kinematically derived nominal configurations. A kinematics-based locomotion reference with a height-conditioned joint-space offset guides learning, while a history-based state estimator infers base linear velocity and height and encodes residual load- and manipulation-induced disturbances in a compact latent representation. The framework is trained entirely in simulation and deployed on a full-size humanoid robot without fine-tuning. Simulation and real-world experiments demonstrate faster training, accurate height tracking, and stable loco-manipulation. Project page: this https URL
>
---
#### [new 041] A Dual Quaternion Framework for Collision Recovery of Quadrotor
- **分类: cs.RO**

- **简介: 该论文属于无人机碰撞恢复任务，解决传统接触模型在快速状态变化中出现的不一致问题，提出基于双四元数的框架，实现SE(3)流形上的精确碰撞建模与能量耗散控制。**

- **链接: [https://arxiv.org/pdf/2603.14698](https://arxiv.org/pdf/2603.14698)**

> **作者:** Valentin Gaucher; Wenlong Zhang
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Unmanned aerial vehicles (UAVs) operating in cluttered environments require accurate impact modeling to maintain stability. However, conventional contact models decouple linear and angular impulses, risking manifold inconsistency during rapid state transitions. This article presents a dual quaternion reset map that resolves rigid-body impacts directly on the SE(3) manifold. By operating on the unified spatial twist (linear and angular velocities as a single dual entity), our formulation is algebraically equivalent to the classical Newton impulse model while preserving manifold consistency during discrete state jumps. Building on this framework, we design a hybrid recovery controller that couples linear and angular momentum to ensure strict energy dissipation across impacts. Hardware-in-the-loop benchmarks demonstrate a 24% reduction in execution latency compared to an optimized matrix-based implementation. High-fidelity MuJoCo simulations validate the controller's robustness to complex contact dynamics, showing a 56.6% reduction in post-impact root-mean-square error (RMSE) and a 41.2% decrease in peak kinetic energy compared to decoupled recovery methods.
>
---
#### [new 042] OxyGen: Unified KV Cache Management for Vision-Language-Action Models under Multi-Task Parallelism
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多任务并行推理领域，解决视觉-语言-动作模型在设备端部署时的效率问题。通过统一KV缓存管理，提升推理速度与性能。**

- **链接: [https://arxiv.org/pdf/2603.14371](https://arxiv.org/pdf/2603.14371)**

> **作者:** Xiangyu Li; Huaizhi Tang; Xin Ding; Weijun Wang; Ting Cao; Yunxin Liu
>
> **备注:** Preprint
>
> **摘要:** Embodied AI agents increasingly require parallel execution of multiple tasks, such as manipulation, conversation, and memory construction, from shared observations under distinct time constraints. Recent Mixture-of-Transformers (MoT) Vision-Language-Action Models (VLAs) architecturally support such heterogeneous outputs, yet existing inference systems fail to achieve efficient multi-task parallelism for on-device deployment due to redundant computation and resource contention. We identify isolated KV cache management as the root cause. To address this, we propose unified KV cache management, an inference paradigm that treats KV cache as a first-class shared resource across tasks and over time. This abstraction enables two key optimizations: cross-task KV sharing eliminates redundant prefill of shared observations, while cross-frame continuous batching decouples variable-length language decoding from fixed-rate action generation across control cycles. We implement this paradigm for $\pi_{0.5}$, the most popular MoT VLA, and evaluate under representative robotic configurations. OxyGen achieves up to 3.7$\times$ speedup over isolated execution, delivering over 200 tokens/s language throughput and 70 Hz action frequency simultaneously without action quality degradation.
>
---
#### [new 043] A Unified Calibration Framework for Coordinate and Kinematic Parameters in Dual-Arm Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人标定任务，旨在解决双臂机器人坐标与运动学参数的联合标定问题。通过统一建模和优化，减少误差传播，提升协作精度。**

- **链接: [https://arxiv.org/pdf/2603.14809](https://arxiv.org/pdf/2603.14809)**

> **作者:** Tianyu Huang; Bohan Yang; Bin Li; Wenpan Li; Haoang Li; Wenlong Li; Yun-Hui Liu
>
> **备注:** 21 pages, 12 figures
>
> **摘要:** Precise collaboration in vision-based dual-arm robot systems requires accurate system calibration. Recent dual-robot calibration methods have achieved strong performance by simultaneously solving multiple coordinate transformations. However, these methods either treat kinematic errors as implicit noise or handle them through separated error modeling, resulting in non-negligible accumulated errors. In this paper, we present a novel framework for unified calibration of the coordinate transformations and kinematic parameters in both robot arms. Our key idea is to unify all the tightly coupled parameters within a single Lie-algebraic formulation. To this end, we construct a consolidated error model grounded in the product-of-exponentials formula, which naturally integrates the coordinate and kinematic parameters in twist forms. Our model introduces no artificial error separation and thus greatly mitigates the error propagation. In addition, we derive a closed-form analytical Jacobian from this model using Lie derivatives. By exploring the Jacobian rank property, we analyze the identifiability of all calibration parameters and show that our joint optimization is well-posed under mild conditions. This enables off-the-shelf iterative solvers to stably optimize these parameters on the manifold space. Besides, to ensure robust convergence of our joint optimization, we develop a certifiably correct algorithm for initializing the unknown coordinates. Relying on semidefinite relaxation, our algorithm can yield a reliable estimate whose near-global optimality can be verified a posteriori. Extensive experiments validate the superior accuracy of our approach over previous baselines under identical visual measurements. Meanwhile, our certifiable initialization consistently outperforms several coordinate-only baselines, proving its reliability as a starting point for joint optimization.
>
---
#### [new 044] Robust Sim-to-Real Cloth Untangling through Reduced-Resolution Observations via Adaptive Force-Difference Quantization
- **分类: cs.RO**

- **简介: 该论文属于机器人布料解缠任务，解决sim-to-real策略迁移中的现实差距问题。通过ADQ方法，利用粗粒度力变化模式提升迁移鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.13785](https://arxiv.org/pdf/2603.13785)**

> **作者:** Yoshihisa Tsurumine; Yuki Kadokawa; Kohei Hayashi; Christian Diehm; Takamitsu Matsubara
>
> **备注:** under review
>
> **摘要:** Robotic cloth untangling requires progressively disentangling fabric by adapting pulling actions to changing contact and tension conditions. Because large-scale real-world training is impractical due to cloth damage and hardware wear, sim-to-real policy transfer is a promising solution. However, cloth manipulation is highly sensitive to interaction dynamics, and policies that depend on precise force magnitudes often fail after transfer because similar force responses cannot be reproduced due to the reality gap. We observe that untangling is largely characterized by qualitative tension transitions rather than exact force values. This indicates that directly minimizing the sim-to-real gap in raw force measurements does not necessarily align with the task structure. We therefore hypothesize that emphasizing coarse force-change patterns while suppressing fine environment-dependent variations can improve robustness of sim-to-real transfer. Based on this insight, we propose Adaptive Force-Difference Quantization (ADQ), which reduces observation resolution by representing force inputs as discretized temporal differences and learning state-dependent quantization thresholds adaptively. This representation mitigates overfitting to environment-specific force characteristics and facilitates direct sim-to-real transfer. Experiments in both simulation and real-world cloth untangling demonstrate that ADQ achieves higher success rates and exhibits greater robustness in sim-to-real transfer than policies using raw force inputs. Supplementary video is available at this https URL
>
---
#### [new 045] Exploring the dynamic properties and motion reproducibility of a small upper-body humanoid robot with 13-DOF pneumatic actuation for data-driven control
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决高自由度气动机器人精确控制问题。通过构建13-DOF上肢机器人并验证其运动可重复性，设计数据驱动控制器提升轨迹跟踪性能。**

- **链接: [https://arxiv.org/pdf/2603.14787](https://arxiv.org/pdf/2603.14787)**

> **作者:** Hiroshi Atsuta; Hisashi Ishihara; Minoru Asada
>
> **备注:** 24 pages, 21 figures. Submitted to Advanced Robotics
>
> **摘要:** Pneumatically-actuated anthropomorphic robots with high degrees of freedom (DOF) offer significant potential for physical human-robot interaction. However, precise control of pneumatic actuators is challenging due to their inherent nonlinearities. This paper presents the development of a compact 13-DOF upper-body humanoid robot. To assess the feasibility of an effective controller, we first investigate its key dynamic properties, such as actuation time delays, and confirm that the system exhibits highly reproducible behavior. Leveraging this reproducibility, we implement a preliminary data-driven controller for a 4-DOF arm subsystem based on a multilayer perceptron with explicit time delay compensation. The network was trained on random movement data to generate pressure commands for tracking arbitrary trajectories. Comparative evaluations with a traditional PID controller demonstrate superior trajectory tracking performance, highlighting the potential of data-driven approaches for controlling complex, high-DOF pneumatic robots.
>
---
#### [new 046] Fabric Pneumatic Artificial Muscle-Based Head-Neck Exosuit: Design, Modeling, and Evaluation
- **分类: cs.RO**

- **简介: 该论文属于康复辅助设备设计任务，旨在解决头颈支持问题。通过设计柔性气动人工肌肉外骨骼，实现头颈运动辅助，提升患者舒适度与功能支持。**

- **链接: [https://arxiv.org/pdf/2603.13531](https://arxiv.org/pdf/2603.13531)**

> **作者:** Katalin Schäffer; Ian Bales; Haohan Zhang; Margaret McGuinness
>
> **备注:** Manuscript (8 pages, 5 tables, 7 figures) accepted to IEEE International Conference on Robotics and Automation 2026. Video attachment: this https URL
>
> **摘要:** Wearable exosuits assist human movement in tasks ranging from rehabilitation to daily activities; specifically, head-neck support is necessary for patients with certain neurological disorders. Rigid-link exoskeletons have shown to enable head-neck mobility compared to static braces, but their bulkiness and restrictive structure inspire designs using "soft" actuation methods. In this paper, we propose a fabric pneumatic artificial muscle-based exosuit design for head-neck support. We describe the design of our prototype and physics-based model, enabling us to derive actuator pressures required to compensate for gravitational load. Our modeled range of motion and workspace analysis indicate that the limited actuator lengths impose slight limitations (83% workspace coverage), and gravity compensation imposes a more significant limitation (43% workspace coverage). We introduce compression force along the neck as a novel, potentially comfort-related metric. We further apply our model to compare the torque output of various actuator placement configurations, allowing us to select a design with stability in lateral deviation and high axial rotation torques. The model correctly predicts trends in measured data where wrapping the actuators around the neck is not a significant factor. Our test dummy and human user demonstration confirm that the exosuit can provide functional head support and trajectory tracking, underscoring the potential of artificial muscle-based soft actuation for head-neck mobility assistance.
>
---
#### [new 047] One-Policy-Fits-All: Geometry-Aware Action Latents for Cross-Embodiment Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决跨具身泛化问题。通过构建共享潜空间和统一解码器，实现多具身策略联合训练，提升数据效率与技能迁移能力。**

- **链接: [https://arxiv.org/pdf/2603.14522](https://arxiv.org/pdf/2603.14522)**

> **作者:** Juncheng Mu; Sizhe Yang; Hojin Bae; Feiyu Jia; Qingwei Ben; Boyi Li; Huazhe Xu; Jiangmiao Pang
>
> **备注:** ICRA 2026
>
> **摘要:** Cross-embodiment manipulation is crucial for enhancing the scalability of robot manipulation and reducing the high cost of data collection. However, the significant differences between embodiments, such as variations in action spaces and structural disparities, pose challenges for joint training across multiple sources of data. To address this, we propose One-Policy-Fits-All (OPFA), a framework that enables learning a single, versatile policy across multiple embodiments. We first learn a Geometry-Aware Latent Representation (GaLR), which leverages 3D convolution networks and transformers to build a shared latent action space across different embodiments. Then we design a unified latent retargeting decoder that extracts embodiment-specific actions from the latent representations, without any embodiment-specific decoder tuning. OPFA enables end-to-end co-training of data from diverse embodiments, including various grippers and dexterous hands with arbitrary degrees of freedom, significantly improving data efficiency and reducing the cost of skill transfer. We conduct extensive experiments across 11 different end-effectors. The results demonstrate that OPFA significantly improves policy performance in diverse settings by leveraging heterogeneous embodiment data. For instance, cross-embodiment co-training can improve success rates by more than 50% compared to single-source training. Moreover, by adding only a few demonstrations from a new embodiment (e.g., eight), OPFA can achieve performance comparable to that of a well-trained model with 72 demonstrations.
>
---
#### [new 048] Safety-guaranteed and Goal-oriented Semantic Sensing, Communication, and Control for Robotics
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人系统任务，旨在解决无线连接机器人通信效率与安全问题。提出安全保障的语义通信方法，提升任务效果与安全性。**

- **链接: [https://arxiv.org/pdf/2603.13502](https://arxiv.org/pdf/2603.13502)**

> **作者:** Wenchao Wu; Shutong Chen; Wenjie Liu; Zhibo Pang; Yansha Deng
>
> **备注:** 7 pages. This paper has been submitted to the IEEE Communications Magazine
>
> **摘要:** Wirelessly-connected robotic system empowers robots with real-time intelligence by leveraging remote computing resources for decision-making. However, the data exchange between robots and base stations often overwhelms communication links, introducing latency that undermines real-time response. To tackle this, goal-oriented semantic communication (GSC) has been introduced into wirelessly-connected robotic systems to extract and transmit only goal-relevant semantic representations, enhancing communication efficiency and task effectiveness. However, existing GSC approaches focused primarily on optimizing effectiveness metrics while overlooking safety requirements, which should be treated as the top priority in real-world robotic systems. To bridge this gap, we propose safety-guaranteed and goal-oriented semantic communication for wirelessly-connected robotic system, aiming to maximize the robotic task effectiveness subject to practical operational safety requirements. We first summarize the general safety requirements and effectiveness metrics across typical robotic tasks, including robot arm grasping, unmanned aerial vehicle (UAV)-assisted tasks, and multi-robot exploration. We then systematically analyze the unique safety and effectiveness challenges faced by wirelessly-connected robotic system in sensing, communication, and control. Based on these, we further present potential safety-guaranteed and goal-oriented sensing, communication, and control solutions. Finally, a UAV target tracking case study validates that our proposed GSC solutions can significantly improve safety rate and tracking success rate by more than 2 times and 4.5 times, respectively.
>
---
#### [new 049] Tactile Modality Fusion for Vision-Language-Action Models
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决触觉信号融合问题。通过轻量级方法TacFiLM，将触觉信息融入VLA模型，提升接触操作性能。**

- **链接: [https://arxiv.org/pdf/2603.14604](https://arxiv.org/pdf/2603.14604)**

> **作者:** Charlotte Morissette; Amin Abyaneh; Wei-Di Chang; Anas Houssaini; David Meger; Hsiu-Chin Lin; Jonathan Tremblay; Gregory Dudek
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** We propose TacFiLM, a lightweight modality-fusion approach that integrates visual-tactile signals into vision-language-action (VLA) models. While recent advances in VLA models have introduced robot policies that are both generalizable and semantically grounded, these models mainly rely on vision-based perception. Vision alone, however, cannot capture the complex interaction dynamics that occur during contact-rich manipulation, including contact forces, surface friction, compliance, and shear. While recent attempts to integrate tactile signals into VLA models often increase complexity through token concatenation or large-scale pretraining, the heavy computational demands of behavioural models necessitate more lightweight fusion strategies. To address these challenges, TacFiLM outlines a post-training finetuning approach that conditions intermediate visual features on pretrained tactile representations using feature-wise linear modulation (FiLM). Experimental results on insertion tasks demonstrate consistent improvements in success rate, direct insertion performance, completion time, and force stability across both in-distribution and out-of-distribution tasks. Together, these results support our method as an effective approach to integrating tactile signals into VLA models, improving contact-rich manipulation behaviours.
>
---
#### [new 050] HapticVLA: Contact-Rich Manipulation via Vision-Language-Action Model without Inference-Time Tactile Sensing
- **分类: cs.RO**

- **简介: 该论文提出HapticVLA，解决无触觉传感下的高接触任务操控问题。通过离线学习和知识蒸馏，使模型具备触觉感知能力，提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2603.15257](https://arxiv.org/pdf/2603.15257)**

> **作者:** Konstantin Gubernatorov; Mikhail Sannikov; Ilya Mikhalchuk; Egor Kuznetsov; Makar Artemov; Ogunwoye Faith Ouwatobi; Marcelino Fernando; Artem Asanov; Ziang Guo; Dzmitry Tsetserukou
>
> **摘要:** Tactile sensing is a crucial capability for Vision-Language-Action (VLA) architectures, as it enables dexterous and safe manipulation in contact-rich tasks. However, reliance on dedicated tactile hardware increases cost and reduces reproducibility across robotic platforms. We argue that tactile-aware manipulation can be learned offline and deployed without direct haptic feedback at inference. To this end, we present HapticVLA, which proceeds in two tightly coupled stages: Safety-Aware Reward-Weighted Flow Matching (SA-RWFM) and Tactile Distillation (TD). SA-RWFM trains a flow-matching action expert that incorporates precomputed, safety-aware tactile rewards penalizing excessive grasping force and suboptimal grasping trajectories. TD further transfers this tactile-aware capability into a conventional VLA: we distill a compact tactile token from the SA-RWFM teacher and train a student VLA to predict that token from vision and state modalities, enabling tactile-aware action generation at inference without requiring on-board tactile sensors. This design preserves contact-rich tactile-aware reasoning within VLA while removing the need for on-board tactile sensors during deployment. On real-world experiments, HapticVLA achieves a mean success rate of 86.7%, consistently outperforming baseline VLAs - including versions provided with direct tactile feedback during inference.
>
---
#### [new 051] LineMaster Pro: A Low-Cost Intelligent Line Following Robot with PID Control and Ultrasonic Obstacle Avoidance for Educational Robotics
- **分类: cs.RO; cs.CV**

- **简介: 该论文介绍LineMaster Pro，一款低成本的智能循线机器人，解决传统设备昂贵且缺乏避障功能的问题。通过PID控制和超声波传感器实现精准导航与避障。**

- **链接: [https://arxiv.org/pdf/2603.13907](https://arxiv.org/pdf/2603.13907)**

> **作者:** Jeni Shahi; Abhishek Shah; A. S. M. Ahsanul Sarkar Akib
>
> **摘要:** Line following robots are fundamental platforms in robotics education, yet commercially available solutions remain prohibitively expensive ($150-300$) while lacking integrated obstacle detection capabilities essential for real-world applications. This paper presents LineMaster Pro, an intelligent low-cost line following robot implemented on an Arduino Nano platform that integrates dual TCRT5000 infrared sensors for precision line tracking, an HC-SR04 ultrasonic sensor for real-time obstacle detection, a digitally tuned PID controller with Ziegler-Nichols optimization, and a hierarchical finite state machine for robust obstacle avoidance. A systematic four-phase sensor calibration methodology ensures reliable operation across varying lighting and surface conditions. Experimental validation through 200 controlled trials and 72-hour continuous operation demonstrates mean tracking accuracy of 1.18 cm at 0.4 m/s (95\% CI [1.06, 1.30]), obstacle detection reliability of 96.7\% within 10-40 cm range with 0.7\% false positive rate, and 94\% successful recovery from path deviations. The PID implementation achieves 43\% improvement over conventional on-off control ($p<0.001$). At a total hardware cost of \$28.50 based on verified Bangladesh market prices, LineMaster Pro achieves a 94\% cost reduction compared to commercial alternatives, establishing a practical benchmark for accessible robotics education in resource-constrained environments.
>
---
#### [new 052] What Matters for Scalable and Robust Learning in End-to-End Driving Planners?
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决闭环驾驶学习的可扩展性和鲁棒性问题。通过分析架构模式，提出BevAD模型，提升驾驶性能与数据扩展能力。**

- **链接: [https://arxiv.org/pdf/2603.15185](https://arxiv.org/pdf/2603.15185)**

> **作者:** David Holtz; Niklas Hanselmann; Simon Doll; Marius Cordts; Bernt Schiele
>
> **备注:** To be published in CVPR Findings 2026
>
> **摘要:** End-to-end autonomous driving has gained significant attention for its potential to learn robust behavior in interactive scenarios and scale with data. Popular architectures often build on separate modules for perception and planning connected through latent representations, such as bird's eye view feature grids, to maintain end-to-end differentiability. This paradigm emerged mostly on open-loop datasets, with evaluation focusing not only on driving performance, but also intermediate perception tasks. Unfortunately, architectural advances that excel in open-loop often fail to translate to scalable learning of robust closed-loop driving. In this paper, we systematically re-examine the impact of common architectural patterns on closed-loop performance: (1) high-resolution perceptual representations, (2) disentangled trajectory representations, and (3) generative planning. Crucially, our analysis evaluates the combined impact of these patterns, revealing both unexpected limitations as well as underexplored synergies. Building on these insights, we introduce BevAD, a novel lightweight and highly scalable end-to-end driving architecture. BevAD achieves 72.7% success rate on the Bench2Drive benchmark and demonstrates strong data-scaling behavior using pure imitation learning. Our code and models are publicly available here: this https URL
>
---
#### [new 053] Stiffness Copilot: An Impedance Policy for Contact-Rich Teleoperation
- **分类: cs.RO**

- **简介: 该论文属于遥操作任务，解决接触密集操作中机器人阻抗选择难题。提出Stiffness Copilot，通过视觉在线调整阻抗，兼顾安全与效率。**

- **链接: [https://arxiv.org/pdf/2603.14068](https://arxiv.org/pdf/2603.14068)**

> **作者:** Yeping Wang; Zhengtong Xu; Pornthep Preechayasomboon; Ben Abbatematteo; Amirhossein H. Memar; Nick Colonnese; Sonny Chan
>
> **备注:** Project website: this https URL
>
> **摘要:** In teleoperation of contact-rich manipulation tasks, selecting robot impedance is critical but difficult. The robot must be compliant to avoid damaging the environment, but stiff to remain responsive and to apply force when needed. In this paper, we present Stiffness Copilot, a vision-based policy for shared-control teleoperation in which the operator commands robot pose and the policy adjusts robot impedance online. To train Stiffness Copilot, we first infer direction-dependent stiffness matrices in simulation using privileged contact information. We then use these matrices to supervise a lightweight vision policy that predicts robot stiffness from wrist-camera images and transfers zero-shot to real images at runtime. In a human-subject study, Stiffness Copilot achieved safety comparable to using a constant low stiffness while matching the efficiency of using a constant high stiffness.
>
---
#### [new 054] SAATT Nav: a Socially Aware Autonomous Transparent Transportation Navigation Framework for Wheelchairs
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主导航任务，旨在解决轮椅在社交环境中的安全与信任问题。提出SAATT Nav框架，结合大语言模型实现社会意识和决策透明，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.13698](https://arxiv.org/pdf/2603.13698)**

> **作者:** Yutong Zhang; Shaiv Y. Mehra; Bradley S. Duerstock; Juan P. Wachs
>
> **备注:** 8 pages, 4 figures, 2 tables, 1 algorithm. Submitted to IROS 2026
>
> **摘要:** While powered wheelchairs reduce physical fatigue as opposed to manual wheelchairs for individuals with mobility impairment, they demand high cognitive workload due to information processing, decision making and motor coordination. Current autonomous systems lack social awareness in navigation and transparency in decision-making, leading to decreased perceived safety and trust from the user and others in context. This work proposes Socially Aware Autonomous Transparent Transportation (SAATT) Navigation framework for wheelchairs as a potential solution. By implementing a Large Language Model (LLM) informed of user intent and capable of predicting other peoples' intent as a decision-maker for its local controller, it is able to detect and navigate social situations, such as passing pedestrians or a pair conversing. Furthermore, the LLM textually communicates its reasoning at each waypoint for transparency. In this experiment, it is compared against a standard global planner, a representative competing social navigation model, and an Ablation study in three simulated environments varied by social levels in eight metrics categorized under Safety, Social Compliance, Efficiency, and Comfort. Overall, SAATT Nav outperforms in most social situations and equivalently or only slightly worse in the remaining metrics, demonstrating the potential of a socially aware and transparent autonomous navigation system to assist wheelchair users.
>
---
#### [new 055] A Methodology for Dynamic Parameters Identification of 3-DOF Parallel Robots in Terms of Relevant Parameters
- **分类: cs.RO**

- **简介: 该论文属于动态参数识别任务，旨在解决3-DOF并联机器人参数难以准确识别的问题。通过简化模型并采用加权最小二乘法进行参数识别，提高模型的物理可行性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.15254](https://arxiv.org/pdf/2603.15254)**

> **作者:** Miguel Díaz-Rodríguez; Vicente Mata; Angel Valera; Alvaro Page
>
> **摘要:** The identification of dynamic parameters in mechanical systems is important for improving model-based control as well as for performing realistic dynamic simulations. Generally, when identification techniques are applied only a subset of so-called base parameters can be identified. More even, some of these parameters cannot be identified properly given that they have a small contribution to the robot dynamics and hence in the presence of noise in measurements and discrepancy in modeling, their quality of being identifiable decreases. For this reason, a strategy for dynamic parameter identification of fully parallel robots in terms of a subset called relevant parameters is put forward. The objective of the proposed methodology is to start from a full dynamic model, then simplification concerning the geometry of each link and, the symmetry due to legs of fully parallel robots, are carried out. After that, the identification is done by Weighted Least Squares. Then, with statistical considerations the model is reduced until the physical feasibility conditions are met. The application of the propose strategy has been experimentally tested on two difierent configurations of actual 3-DOF parallel robots. The response of the inverse and forward dynamics of the identified models agrees with experiments. In order to evaluate the forward dynamics response, an approach for obtaining the forward dynamics in terms of the relevant parameters is also proposed.
>
---
#### [new 056] Ego to World: Collaborative Spatial Reasoning in Embodied Systems via Reinforcement Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究多智能体系统中的空间推理问题，旨在提升从分散视角理解世界的能力。提出E2W基准和CoRL框架，通过强化学习实现跨视角协同感知与决策。**

- **链接: [https://arxiv.org/pdf/2603.14811](https://arxiv.org/pdf/2603.14811)**

> **作者:** Heng Zhou; Li Kang; Yiran Qin; Xiufeng Song; Ao Yu; Zilu Zhang; Haoming Song; Kaixin Xu; Yuchen Fan; Dongzhan Zhou; Xiaohong Liu; Ruimao Zhang; Philip Torr; Lei Bai; Zhenfei Yin
>
> **摘要:** Understanding the world from distributed, partial viewpoints is a fundamental challenge for embodied multi-agent systems. Each agent perceives the environment through an ego-centric view that is often limited by occlusion and ambiguity. To study this problem, we introduce the Ego-to-World (E2W) benchmark, which evaluates a vision-language model's ability to fuse heterogeneous viewpoints across three tasks: (i) global counting, (ii) relational location reasoning, and (iii) action-oriented grasping that requires predicting view-specific image coordinates. To address this setting, we propose CoRL, a two-stage framework that combines Chain-of-Thought supervised fine-tuning with reinforcement learning using Group-Relative Policy Optimization. Its core component, the Cross-View Spatial Reward (CVSR), provides dense task-aligned feedback by linking reasoning steps to visual evidence, ensuring coherent cross-view entity resolution, and guiding the model toward correct final predictions. Experiments on E2W show that CoRL consistently surpasses strong proprietary and open-source baselines on both reasoning and perception-grounding metrics, while ablations further confirm the necessity of each CVSR component. Beyond that, CoRL generalizes to external spatial reasoning benchmarks and enables effective real-world multi-robot manipulation with calibrated multi-camera rigs, demonstrating cross-view localization and successful grasp-and-place execution. Together, E2W and CoRL provide a principled foundation for learning world-centric scene understanding from distributed, ego-centric observations, advancing collaborative embodied AI.
>
---
#### [new 057] ViSA: Visited-State Augmentation for Generalized Goal-Space Contrastive Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于强化学习领域，针对目标条件强化学习中价值函数估计不准确的问题，提出ViSA方法，通过数据增强提升目标空间泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.14887](https://arxiv.org/pdf/2603.14887)**

> **作者:** Issa Nakamura; Tomoya Yamanokuchi; Yuki Kadokawa; Jia Qu; Shun Otsub; Ken Miyamoto; Shotaro Miwa; Takamitsu Matsubara
>
> **备注:** 8 pages, 7 figures, under Review
>
> **摘要:** Goal-Conditioned Reinforcement Learning (GCRL) is a framework for learning a policy that can reach arbitrarily given goals. In particular, Contrastive Reinforcement Learning (CRL) provides a framework for policy updates using an approximation of the value function estimated via contrastive learning, achieving higher sample efficiency compared to conventional methods. However, since CRL treats the visited state as a pseudo-goal during learning, it can accurately estimate the value function only for limited goals. To address this issue, we propose a novel data augmentation approach for CRL called ViSA (Visited-State Augmentation). ViSA consists of two components: 1) generating augmented state samples, with the aim of augmenting hard-to-visit state samples during on-policy exploration, and 2) learning consistent embedding space, which uses an augmented state as auxiliary information to regularize the embedding space by reformulating the objective function of the embedding space based on mutual information. We evaluate ViSA in simulation and real-world robotic tasks and show improved goal-space generalization, which permits accurate value estimation for hard-to-visit goals. Further details can be found on the project page: \href{this https URL}{\texttt{this https URL\_ViSA/}}
>
---
#### [new 058] Coupled Particle Filters for Robust Affordance Estimation
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，解决视觉和语义模糊下的 affordance 估计问题。通过耦合粒子滤波器提升 graspable 和 movable 区域的估计精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.15223](https://arxiv.org/pdf/2603.15223)**

> **作者:** Patrick Lowin; Vito Mengers; Oliver Brock
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Robotic affordance estimation is challenging due to visual, geometric, and semantic ambiguities in sensory input. We propose a method that disambiguates these signals using two coupled recursive estimators for sub-aspects of affordances: graspable and movable regions. Each estimator encodes property-specific regularities to reduce uncertainty, while their coupling enables bidirectional information exchange that focuses attention on regions where both agree, i.e., affordances. Evaluated on a real-world dataset, our method outperforms three recent affordance estimators (Where2Act, Hands-as-Probes, and HRP) by 308%, 245%, and 257% in precision, and remains robust under challenging conditions such as low light or cluttered environments. Furthermore, our method achieves a 70% success rate in our real-world evaluation. These results demonstrate that coupling complementary estimators yields precise, robust, and embodiment-appropriate affordance predictions.
>
---
#### [new 059] Sonar-MASt3R: Real-Time Opti-Acoustic Fusion in Turbid, Unstructured Environments
- **分类: cs.RO**

- **简介: 该论文属于水下感知任务，旨在解决浑浊环境中三维重建困难的问题。通过融合光学与声呐数据，提出Sonar-MASt3R方法，实现实时、鲁棒的三维重建。**

- **链接: [https://arxiv.org/pdf/2603.13585](https://arxiv.org/pdf/2603.13585)**

> **作者:** Amy Phung; Richard Camilli
>
> **备注:** This paper has been accepted for publication in ICRA 2026. Copyright IEEE
>
> **摘要:** Underwater intervention is an important capability in several marine domains, with numerous industrial, scientific, and defense applications. However, existing perception systems used during intervention operations rely on data from optical cameras, which limits capabilities in poor visibility or lighting conditions. Prior work has examined opti-acoustic fusion methods, which use sonar data to resolve the depth ambiguity of the camera data while using camera data to resolve the elevation angle ambiguity of the sonar data. However, existing methods cannot achieve dense 3D reconstructions in real-time, and few studies have reported results from applying these methods in a turbid environment. In this work, we propose the opti-acoustic fusion method Sonar-MASt3R, which uses MASt3R to extract dense correspondences from optical camera data in real-time and pairs it with geometric cues from an acoustic 3D reconstruction to ensure robustness in turbid conditions. Experimental results using data recorded from an opti-acoustic eye-in-hand configuration across turbidity values ranging from <0.5 to >12 NTU highlight this method's improved robustness to turbidity relative to baseline methods.
>
---
#### [new 060] SmallSatSim: A High-Fidelity Simulation and Training Toolkit for Microgravity Robotic Close Proximity Operations
- **分类: cs.RO**

- **简介: 该论文提出SmallSatSim工具，用于微重力环境下小卫星近距离操作的仿真与训练，解决机器人控制与规划难题。**

- **链接: [https://arxiv.org/pdf/2603.14598](https://arxiv.org/pdf/2603.14598)**

> **作者:** David Schwartz; Alexander Hansson; Sabrina Bodmer; David Sternberg; Oliver Jia-Richards; Keenan Albee
>
> **备注:** 7 pages, 7 figures
>
> **摘要:** Microgravity rendezvous and close proximity operations (RPO) is a growing area of interest for applications spanning in-space assembly and manufacturing (ISAM), orbital debris remediation, and small body exploration. Microgravity environments present unique challenges for robotic control and planning algorithms for new agile RPO mission scenarios like free-floating manipulation, planning under failure, and estimating high-fidelity dynamics of tumbling bodies. To facilitate the development and testing of novel RPO algorithms, we introduce SmallSatSim, a high-fidelity simulation toolkit that leverages the MuJoCo physics engine to accurately model small satellite RPO dynamics in local microgravity robotic free-flight settings, including under model disturbances and perturbations. The framework includes cutting edge out-of-the-box free-flyer control techniques. A GPU-accelerated pipeline using MuJoCo MJX and JAX is implemented for sampling- and learning-based simulation uses cases. SmallSatSim also supports configurable failure models, enabling the evaluation of safe control strategies under adversarial conditions. Visualization, logging, and GPU-enabled parallelization further enhance SmallSatSim's capability for RPO testing. We outline SmallSatSim's features and intended use cases, and demonstrate its use for robotic RPO planning and control. The open-sourced toolkit aims to accelerate research in autonomous, agile robotic small satellite operations.
>
---
#### [new 061] ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人接触操作任务，旨在解决传统位置控制在力感知和调节上的不足。提出ForceVLA2框架，实现混合力-位置控制与显式力感知，提升操作稳定性与成功率。**

- **链接: [https://arxiv.org/pdf/2603.15169](https://arxiv.org/pdf/2603.15169)**

> **作者:** Yang Li; Zhaxizhuoma; Hongru Jiang; Junjie Xia; Hongquan Zhang; Jinda Du; Yunsong Zhou; Jia Zeng; Ce Hao; Jieji Ren; Qiaojun Yu; Cewu Lu; Yu Qiao; Jiangmiao Pang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Embodied intelligence for contact-rich manipulation has predominantly relied on position control, while explicit awareness and regulation of interaction forces remain under-explored, limiting stability, precision, and robustness in real-world tasks. We propose ForceVLA2, an end-to-end vision-language-action framework that equips robots with hybrid force-position control and explicit force awareness. ForceVLA2 introduces force-based prompts into the VLM expert to construct force-aware task concepts across stages, and employs a Cross-Scale Mixture-of-Experts (MoE) in the action expert to adaptively fuse these concepts with real-time interaction forces for closed-loop hybrid force-position regulation. To support learning and evaluation, we construct ForceVLA2-Dataset, containing 1,000 trajectories over 5 contact-rich tasks, including wiping, pressing, and assembling, with multi-view images, task prompts, proprioceptive state, and force signals. Extensive experiments show that ForceVLA2 substantially improves success rates and reliability in contact-rich manipulation, outperforming pi0 and pi0.5 by 48.0% and 35.0%, respectively, across the 5 tasks, and mitigating common failure modes such as arm overload and unstable contact, thereby actively advancing force-aware interactive physical intelligence in VLAs. The project page is available at this https URL.
>
---
#### [new 062] EAAE: Energy-Aware Autonomous Exploration for UAVs in Unknown 3D Environments
- **分类: cs.RO**

- **简介: 该论文属于无人机自主探索任务，解决能量限制下的3D环境探索问题。提出EAAE框架，将能耗作为决策因素，优化轨迹以降低能耗并保持探索效率。**

- **链接: [https://arxiv.org/pdf/2603.15604](https://arxiv.org/pdf/2603.15604)**

> **作者:** Jacob Elskamp; Moji Shi; Leonard Bauersfeld; Davide Scaramuzza; Marija Popović
>
> **摘要:** Battery-powered multirotor unmanned aerial vehicles (UAVs) can rapidly map unknown environments, but mission performance is often limited by energy rather than geometry alone. Standard exploration policies that optimise for coverage or time can therefore waste energy through manoeuvre-heavy trajectories. In this paper, we address energy-aware autonomous 3D exploration for multirotor UAVs in initially unknown environments. We propose Energy-Aware Autonomous Exploration (EAAE), a modular frontier-based framework that makes energy an explicit decision variable during frontier selection. EAAE clusters frontiers into view-consistent regions, plans dynamically feasible candidate trajectories to the most informative clusters, and predicts their execution energy using an offline power estimation loop. The next target is then selected by minimising predicted trajectory energy while preserving exploration progress through a dual-layer planning architecture for safe execution. We evaluate EAAE in a full exploration pipeline with a rotor-speed-based power model across simulated 3D environments of increasing complexity. Compared to representative distance-based and information gain-based frontier baselines, EAAE consistently reduces total energy consumption while maintaining competitive exploration time and comparable map quality, providing a practical drop-in energy-aware layer for frontier exploration.
>
---
#### [new 063] ST-VLA: Enabling 4D-Aware Spatiotemporal Understanding for General Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决开放环境中语义、几何和时序动作推理的问题。提出ST-VLA框架，使用统一的3D-4D表示增强时空理解，提升操作鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.13788](https://arxiv.org/pdf/2603.13788)**

> **作者:** You Wu; Zixuan Chen; Cunxu Ou; Wenxuan Wang; Wenbo Huang; Lin Cao; Yangtao Chen; Weichao Qiu; Xingyue Quan; Jieqi Shi; Jing Huo; Yang Gao
>
> **备注:** 25 pages, under review
>
> **摘要:** Robotic manipulation in open-world environments requires reasoning across semantics, geometry, and long-horizon action dynamics. Existing hierarchical Vision-Language-Action (VLA) frameworks typically use 2D representations to connect high-level reasoning with low-level control, but lack depth awareness and temporal consistency, limiting robustness in complex 3D scenes. We propose ST-VLA, a hierarchical VLA framework using a unified 3D-4D representation to bridge perception and action. ST-VLA converts 2D guidance into 3D trajectories and generates smooth spatial masks that capture 4D spatio-temporal context, providing a stable interface between semantic reasoning and continuous control. To enable effective learning of such representations, we introduce ST-Human, a large-scale human manipulation dataset with 14 tasks and 300k episodes, annotated with 2D, 3D, and 4D supervision via a semi-automated pipeline. Using ST-Human, we train ST-VLM, a spatio-temporal vision-language model that generates spatially grounded and temporally coherent 3D representations to guide policy execution. The smooth spatial masks focus on task-relevant geometry and stabilize latent representations, enabling online replanning and long-horizon reasoning. Experiments on RLBench and real-world manipulation tasks show that \method significantly outperforms state-of-the-art baselines, improving zero-shot success rates by 44.6% and 30.3%. These results demonstrate that offloading spatio-temporal reasoning to VLMs with unified 3D-4D representations substantially improves robustness and generalization for open-world robotic manipulation. Project website: this https URL.
>
---
#### [new 064] AnoleVLA: Lightweight Vision-Language-Action Model with Deep State Space Models for Mobile Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于语言引导机器人操作任务，旨在解决资源受限环境下高效执行任务的问题。提出AnoleVLA模型，利用深度状态空间模型实现快速多模态处理。**

- **链接: [https://arxiv.org/pdf/2603.15046](https://arxiv.org/pdf/2603.15046)**

> **作者:** Yusuke Takagi; Motonari Kambara; Daichi Yashima; Koki Seno; Kento Tokura; Komei Sugiura
>
> **摘要:** In this study, we address the problem of language-guided robotic manipulation, where a robot is required to manipulate a wide range of objects based on visual observations and natural language instructions. This task is essential for service robots that operate in human environments, and requires safety, efficiency, and task-level generality. Although Vision-Language-Action models (VLAs) have demonstrated strong performance for this task, their deployment in resource-constrained environments remains challenging because of the computational cost of standard transformer backbones. To overcome this limitation, we propose AnoleVLA, a lightweight VLA that uses a deep state space model to process multimodal sequences efficiently. The model leverages its lightweight and fast sequential state modeling to process visual and textual inputs, which allows the robot to generate trajectories efficiently. We evaluated the proposed method in both simulation and physical experiments. Notably, in real-world evaluations, AnoleVLA outperformed a representative large-scale VLA by 21 points for the task success rate while achieving an inference speed approximately three times faster.
>
---
#### [new 065] NavGSim: High-Fidelity Gaussian Splatting Simulator for Large-Scale Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决大规模环境模拟难题。提出NavGSim，基于高斯泼溅技术生成高保真场景，并支持碰撞检测与多GPU开发。**

- **链接: [https://arxiv.org/pdf/2603.15186](https://arxiv.org/pdf/2603.15186)**

> **作者:** Jiahang Liu; Yuanxing Duan; Jiazhao Zhang; Minghan Li; Shaoan Wang; Zhizheng Zhang; He Wang
>
> **摘要:** Simulating realistic environments for robots is widely recognized as a critical challenge in robot learning, particularly in terms of rendering and physical simulation. This challenge becomes even more pronounced in navigation tasks, where trajectories often extend across multiple rooms or entire floors. In this work, we present NavGSim, a Gaussian Splatting-based simulator designed to generate high-fidelity, large-scale navigation environments. Built upon a hierarchical 3D Gaussian Splatting framework, NavGSim enables photorealistic rendering in expansive scenes spanning hundreds of square meters. To simulate navigation collisions, we introduce a Gaussian Splatting-based slice technique that directly extracts navigable areas from reconstructed Gaussians. Additionally, for ease of use, we provide comprehensive NavGSim APIs supporting multi-GPU development, including tools for custom scene reconstruction, robot configuration, policy training, and evaluation. To evaluate NavGSim's effectiveness, we train a Vision-Language-Action (VLA) model using trajectories collected from NavGSim and assess its performance in both simulated and real-world environments. Our results demonstrate that NavGSim significantly enhances the VLA model's scene understanding, enabling the policy to handle diverse navigation queries effectively.
>
---
#### [new 066] HALO:Closing Sim-to-Real Gap for Heavy-loaded Humanoid Agile Motion Skills via Differentiable Simulation
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决仿真到现实的迁移难题。针对负载未知的人形机器人，提出两阶段系统识别框架，提升策略在真实环境中的适应性与敏捷性。**

- **链接: [https://arxiv.org/pdf/2603.15084](https://arxiv.org/pdf/2603.15084)**

> **作者:** Xingyi Wang; Chenyun Zhang; Weiji Xie; Chao Yu; Wei Song; Chenjia Bai; Shiqiang Zhu
>
> **备注:** 9 pages, 5 figures, conference
>
> **摘要:** Humanoid robots deployed in real-world scenarios often need to carry unknown payloads, which introduce significant mismatch and degrade the effectiveness of simulation-to-reality reinforcement learning methods. To address this challenge, we propose a two-stage gradient-based system identification framework built on the differentiable simulator MuJoCo XLA. The first stage calibrates the nominal robot model using real-world data to reduce intrinsic sim-to-real discrepancies, while the second stage further identifies the mass distribution of the unknown payload. By explicitly reducing structured model bias prior to policy training, our approach enables zero-shot transfer of reinforcement learning policies to hardware under heavy-load conditions. Extensive simulation and real-world experiments demonstrate more precise parameter identification, improved motion tracking accuracy, and substantially enhanced agility and robustness compared to existing baselines. Project Page: this https URL
>
---
#### [new 067] A Real-Time Neuro-Symbolic Ethical Governor for Safe Decision Control in Autonomous Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于自主机器人安全控制任务，旨在解决伦理决策治理问题。通过融合神经符号方法，实现实时风险感知与伦理判断，提升操作安全性。**

- **链接: [https://arxiv.org/pdf/2603.14221](https://arxiv.org/pdf/2603.14221)**

> **作者:** Aueaphum Aueawatthanaphisut; Kuepon Aueawatthanaphisut
>
> **备注:** 6 pages, 6 figures, 5 equations
>
> **摘要:** Ethical decision governance has become a critical requirement for autonomous robotic systems operating in human-centered and safety-sensitive environments. This paper presents a real-time neuro-symbolic ethical governor designed to enable risk-aware supervisory control in autonomous robotic manipulation tasks. The proposed framework integrates transformer-based ethical reasoning with a probabilistic ethical risk field formulation and a threshold-based override control mechanism. language-grounded ethical intent inference capability is learned from natural language task descriptions using a fine-tuned DistilBERT model trained on the ETHICS commonsense dataset. A continuous ethical risk metric is subsequently derived from predicted unsafe action probability, confidence uncertainty, and probabilistic variance to support adaptive decision filtering. The effectiveness of the proposed approach is validated through simulated autonomous robot-arm task scenarios involving varying levels of human proximity and operational hazard. Experimental results demonstrate stable model convergence, reliable ethical risk discrimination, and improved safety-aware decision outcomes without significant degradation of task execution efficiency. The proposed neuro-symbolic architecture further provides enhanced interpretability compared with purely data-driven safety filters, enabling transparent ethical reasoning in real-time control loops. The findings suggest that ethical decision governance can be effectively modeled as a dynamic supervisory risk layer for autonomous robotic systems, with potential applicability to broader cyber-physical and assistive robotics domains.
>
---
#### [new 068] Physically Accurate Rigid-Body Dynamics in Particle-Based Simulation
- **分类: cs.RO**

- **简介: 该论文属于物理仿真任务，旨在解决粒子模拟中刚体动力学物理准确性不足的问题。通过改进PBD方法，提出PBD-R，提升仿真精度与计算效率。**

- **链接: [https://arxiv.org/pdf/2603.14634](https://arxiv.org/pdf/2603.14634)**

> **作者:** Ava Abderezaei; Nataliya Nechyporenko; Joseph Miceli; Gilberto Briscoe-Martinez; Alessandro Roncone
>
> **备注:** Submitted to IROS 2026
>
> **摘要:** Robotics demands simulation that can reason about the diversity of real-world physical interactions, from rigid to deformable objects and fluids. Current simulators address this by stitching together multiple subsolvers for different material types, resulting in a compositional architecture that complicates physical reasoning. Particle-based simulators offer a compelling alternative, representing all materials through a single unified formulation that enables seamless cross-material interactions. Among particle-based simulators, position-based dynamics (PBD) is a popular solver known for its computational efficiency and visual plausibility. However, its lack of physical accuracy has limited its adoption in robotics. To leverage the benefits of particle-based solvers while meeting the physical fidelity demands of robotics, we introduce PBD-R, a revised PBD formulation that enforces physically accurate rigid-body dynamics through a novel momentum-conservation constraint and a modified velocity update. Additionally, we introduce a solver-agnostic benchmark with analytical solutions to evaluate physical accuracy. Using this benchmark, we show that PBD-R significantly outperforms PBD and achieves competitive accuracy with MuJoCo while requiring less computation.
>
---
#### [new 069] URDF-Anything+: Autoregressive Articulated 3D Models Generation for Physical Simulation
- **分类: cs.RO**

- **简介: 该论文提出一种端到端的自回归框架，用于生成可执行的关节3D模型，解决从视觉输入重建关节物体的问题，提升物理模拟的准确性与真实性。**

- **链接: [https://arxiv.org/pdf/2603.14010](https://arxiv.org/pdf/2603.14010)**

> **作者:** Zhuangzhe Wu; Yue Xin; Chengkai Hou; Minghao Chen; Yaoxu Lyu; Jieyu Zhang; Shanghang Zhang
>
> **摘要:** Articulated objects are fundamental for robotics, simulation of physics, and interactive virtual environments. However, reconstructing them from visual input remains challenging, as it requires jointly inferring both part geometry and kinematic structure. We present, an end-to-end autoregressive framework that directly generates executable articulated object models from visual observations. Given image and object-level 3D cues, our method sequentially produces part geometries and their associated joint parameters, resulting in complete URDF models without reliance on multi-stage pipelines. The generation proceeds until the model determines that all parts have been produced, automatically inferring complete geometry and kinematics. Building on this capability, we enable a new Real-Follow-Sim paradigm, where high-fidelity digital twins constructed from visual observations allow policies trained and tested purely in simulation to transfer to real robots without online adaptation. Experiments on large-scale articulated object benchmarks and real-world robotic tasks demonstrate that outperforms prior methods in geometric reconstruction quality, joint parameter accuracy, and physical executability.
>
---
#### [new 070] Creating manufacturable blueprints for coarse-grained virtual robots
- **分类: cs.RO**

- **简介: 该论文属于机器人设计任务，旨在解决虚拟设计难以转化为可制造机器人的问题。通过自动化流程将抽象设计转换为完整蓝图，提升设计效率与可行性。**

- **链接: [https://arxiv.org/pdf/2603.13582](https://arxiv.org/pdf/2603.13582)**

> **作者:** Zihan Guo; Muhan Li; Shuzhe Zhang; Sam Kriegman
>
> **摘要:** Over the past three decades, countless embodied yet virtual agents have freely evolved inside computer simulations, but vanishingly few were realized as physical robots. This is because evolution was conducted at a level of abstraction that was convenient for freeform body generation (creation, mutation, recombination) but swept away almost all of the physical details of functional body parts. The resulting designs were crude and underdetermined, requiring considerable effort and expertise to convert into a manufacturable format. Here, we automate this mapping from simplified design spaces that are readily evolvable to complete blueprints that can be directly followed by a builder. The pipeline incrementally resolves manufacturing constraints by embedding the structural and functional semantics of motors, electronics, batteries, and wiring into the abstract virtual design. In lieu of evolution, a user-defined or AI-generated ``sketch'' of a body plan can also be fed as input to the pipeline, providing a versatile framework for accelerating the design of novel robots.
>
---
#### [new 071] Surgical Robot, Path Planning, Joint Space, Riemannian Manifolds
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人路径规划任务，解决手术机器人在有限空间内运动受限和角度限制问题。通过将位置转换为黎曼流形进行联合空间路径规划，降低关节运动范围。**

- **链接: [https://arxiv.org/pdf/2603.14852](https://arxiv.org/pdf/2603.14852)**

> **作者:** Yoshiki Yamamoto; Maina Sogabe; Shunichi Hirahara; Toshiki Kaisaki; Tetsuro Miyazaki; Kenji Kawashima
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** Robotic surgery for minimally invasive surgery can reduce the surgeon's workload by autonomously guiding robotic forceps. Movement of the robot is restricted around a fixed insertion port. The robot often encounters angle limitations during operation. Also, the surface of the abdominal cavity is non-concave, making it computationally expensive to find the desired this http URL this work, to solve these problems, we propose a method for path planning in joint space by transforming the position into a Riemannian manifold. An edge cost function is defined to search for a desired path in the joint space and reduce the range of motion of the joints. We found that the organ is mostly non-concave, making it easy to find the optimal path using gradient descent method. Experimental results demonstrated that the proposed method reduces the range of joint angle movement compared to calculations in position space.
>
---
#### [new 072] OmniClone: Engineering a Robust, All-Rounder Whole-Body Humanoid Teleoperation System
- **分类: cs.RO**

- **简介: 该论文提出OmniClone系统，解决人形机器人远程操控的鲁棒性与通用性问题，通过诊断基准优化训练数据和系统设计，实现高效、低成本的多技能控制。**

- **链接: [https://arxiv.org/pdf/2603.14327](https://arxiv.org/pdf/2603.14327)**

> **作者:** Yixuan Li; Le Ma; Yutang Lin; Yushi Du; Mengya Liu; Kaizhe Hu; Jieming Cui; Yixin Zhu; Wei Liang; Baoxiong Jia; Siyuan Huang
>
> **备注:** Website: this https URL
>
> **摘要:** Whole-body humanoid teleoperation enables humans to remotely control humanoid robots, serving as both a real-time operational tool and a scalable engine for collecting demonstrations for autonomous learning. Despite recent advances, existing systems are validated using aggregate metrics that conflate distinct motion regimes, masking critical failure modes. This lack of diagnostic granularity, compounded by tightly coupled and labor-intensive system configurations, hinders robust real-world deployment. A key open challenge is building a teleoperation system that is simultaneously robust, versatile, and affordable for practical use. Here we present OmniClone, a whole-body humanoid teleoperation system that achieves high-fidelity, multi-skill control on a single consumer GPU with modest data requirements. Central to our approach is OmniBench, a diagnostic benchmark that evaluates policies across stratified motion categories and difficulty levels on unseen motions, exposing the narrow specialization of prior systems. Guided by these diagnostics, we identify an optimized training data recipe and integrate system-level improvements: subject-agnostic retargeting and robust communication, that collectively reduce Mean Per-Joint Position Error (MPJPE) by over 66% while requiring orders-of-magnitude fewer computational resources than comparable methods. Crucially, OmniClone is control-source-agnostic: a single unified policy supports real-time teleoperation, generated motion playback, and Vision-Language-Action (VLA) models, while generalizing across operators of vastly different body proportions. By uniting diagnostic evaluation with practical engineering, OmniClone provides an accessible foundation for scalable humanoid teleoperation and autonomous learning.
>
---
#### [new 073] LPV-MPC for Lateral Control in Full-Scale Autonomous Racing
- **分类: cs.RO**

- **简介: 该论文属于自主赛车控制任务，解决高速下横向控制问题。提出LPV-MPC控制器，实现稳定高速行驶，分析车辆动态与控制性能。**

- **链接: [https://arxiv.org/pdf/2603.13732](https://arxiv.org/pdf/2603.13732)**

> **作者:** Hassan Jardali; Ihab S. Mohamed; Durgakant Pushp; Lantao Liu
>
> **摘要:** Autonomous racing has attracted significant attention recently, presenting challenges in selecting an optimal controller that operates within the onboard system's computational limits and meets operational constraints such as limited track time and high costs. This paper introduces a Linear Parameter-Varying Model Predictive Controller (LPV-MPC) for lateral control. Implemented on an IAC AV-24, the controller achieved stable performance at speeds exceeding 160 mph (71.5 m/s). We detail the controller design, the methodology for extracting model parameters, and key system-level and implementation considerations. Additionally, we report results from our final race run, providing a comprehensive analysis of both vehicle dynamics and controller performance. A Python implementation of the framework is available at: this https URL
>
---
#### [new 074] Fine-tuning is Not Enough: A Parallel Framework for Collaborative Imitation and Reinforcement Learning in End-to-end Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，旨在解决传统模仿学习性能受限的问题。通过提出PaIR-Drive框架，实现模仿学习与强化学习的并行协作，提升驾驶性能。**

- **链接: [https://arxiv.org/pdf/2603.13842](https://arxiv.org/pdf/2603.13842)**

> **作者:** Zhexi Lian; Haoran Wang; Xuerun Yan; Weimeng Lin; Xianhong Zhang; Yongyu Chen; Jia Hu
>
> **备注:** 8 pages, 7 figures, 6 tables
>
> **摘要:** End-to-end autonomous driving is typically built upon imitation learning (IL), yet its performance is constrained by the quality of human demonstrations. To overcome this limitation, recent methods incorporate reinforcement learning (RL) through sequential fine-tuning. However, such a paradigm remains suboptimal: sequential RL fine-tuning can introduce policy drift and often leads to a performance ceiling due to its dependence on the pretrained IL policy. To address these issues, we propose PaIR-Drive, a general Parallel framework for collaborative Imitation and Reinforcement learning in end-to-end autonomous driving. During training, PaIR-Drive separates IL and RL into two parallel branches with conflict-free training objectives, enabling fully collaborative optimization. This design eliminates the need to retrain RL when applying a new IL policy. During inference, RL leverages the IL policy to further optimize the final plan, allowing performance beyond prior knowledge of IL. Furthermore, we introduce a tree-structured trajectory neural sampler to group relative policy optimization (GRPO) in the RL branch, which enhances exploration capability. Extensive analysis on NAVSIMv1 and v2 benchmark demonstrates that PaIR-Drive achieves Competitive performance of 91.2 PDMS and 87.9 EPDMS, building upon Transfuser and DiffusionDrive IL baselines. PaIR-Drive consistently outperforms existing RL fine-tuning methods, and could even correct human experts' suboptimal behaviors. Qualitative results further confirm that PaIR-Drive can effectively explore and generate high-quality trajectories.
>
---
#### [new 075] From Folding Mechanics to Robotic Function: A Unified Modeling Framework for Compliant Origami
- **分类: cs.RO**

- **简介: 该论文属于机器人学与结构力学交叉领域，旨在解决柔性折纸结构的统一建模问题。通过引入离散微分几何框架，实现刚性折叠与弹性变形的统一描述，支持可编程控制与多物理场仿真。**

- **链接: [https://arxiv.org/pdf/2603.14900](https://arxiv.org/pdf/2603.14900)**

> **作者:** Bohan Zhang; Bo Wang; Huajiang Ouyang; Zhigang Wu; Haohao Bi; Jiawei Xu; Mingchao Liu; Weicheng Huang
>
> **备注:** 24 pages, 7 figures
>
> **摘要:** Origami inspired architectures offer a powerful route toward lightweight, reconfigurable, and programmable robotic systems. Yet, a unified mechanics framework capable of seamlessly bridging rigid folding, elastic deformation, and stability driven transitions in compliant origami remains lacking. Here, we introduce a geometry consistent modeling framework based on discrete differential geometry (DDG) that unifies panel elasticity and crease rotation within a single variational formulation. By embedding crease panel coupling directly into a mid edge geometric discretization, the framework naturally captures rigid folding limits, distributed bending, multistability, and nonlinear dynamic snap through within one mechanically consistent structure. This unified description enables programmable control of stability and deformation across rigid and compliant regimes, allowing origami structures to transition from static folding mechanisms to active robotic modules. An implicit dynamic formulation incorporating gravity, contact, friction, and magnetic actuation further supports strongly coupled multiphysics simulations. Through representative examples spanning single fold bifurcation, deployable Miura membranes, bistable Waterbomb modules, and Kresling based crawling robots, we demonstrate how geometry driven mechanics directly informs robotic functionality. This work establishes discrete differential geometry as a foundational design language for intelligent origami robotics, enabling predictive modeling, stability programming, and mechanics guided robotic actuation within a unified computational platform.
>
---
#### [new 076] GraspADMM: Improving Dexterous Grasp Synthesis via ADMM Optimization
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决高质灵巧抓取的合成问题。提出GraspADMM框架，结合ADMM优化，提升抓取的多样性、可行性与动态稳定性。**

- **链接: [https://arxiv.org/pdf/2603.13832](https://arxiv.org/pdf/2603.13832)**

> **作者:** Liangwang Ruan; Jiayi Chen; He Wang; Baoquan Chen
>
> **摘要:** Synthesizing high-quality dexterous grasps is a fundamental challenge in robot manipulation, requiring adherence to diversity, kinematic feasibility (valid hand-object contact without penetration), and dynamic stability (secure multi-contact forces). The recent framework Dexonomy successfully ensures broad grasp diversity through dense sampling and improves kinematic feasibility via a simulator-based refinement method that excels at resolving exact collisions. However, its reliance on fixed contact points restricts the hand's reachability and prevents the optimization of grasp metrics for dynamic stability. Conversely, purely gradient-based optimizers can maximize dynamic stability but rely on simplified contact approximations that inevitably cause physical penetrations. To bridge this gap, we propose GraspADMM, a novel grasp synthesis framework that preserves sampling-based diversity while improving kinematic feasibility and dynamic stability. By formulating the refinement stage using the Alternating Direction Method of Multipliers (ADMM), we decouple the target contact points on the object from the actual contact locations on the hand. This decomposition allows the pipeline to alternate between updating the target object points to directly maximize dynamic grasp metrics, and adjusting the hand pose to physically reach these targets while strictly respecting collision boundaries. Extensive experiments demonstrate that GraspADMM significantly outperforms state-of-the-art baselines, achieving a nearly 15\% absolute improvement in grasp success rate for type-unaware synthesis and roughly a 100\% relative improvement in type-aware synthesis. Furthermore, our approach maintains robust, physically plausible grasp generation even under extreme low-friction conditions.
>
---
#### [new 077] Latent Dynamics-Aware OOD Monitoring for Trajectory Prediction with Provable Guarantees
- **分类: cs.RO**

- **简介: 该论文属于轨迹预测中的OOD检测任务，旨在提升安全关键系统中预测模型的可靠性。通过构建HMM模型并改进MMD方法，实现快速且可靠的风险检测。**

- **链接: [https://arxiv.org/pdf/2603.14603](https://arxiv.org/pdf/2603.14603)**

> **作者:** Tongfei Guo; Lili Su
>
> **摘要:** In safety-critical Cyber-Physical Systems (CPS), accurate trajectory prediction provides vital guidance for downstream planning and control, yet although deep learning models achieve high-fidelity forecasts on validation data, their reliability degrades under out-of-distribution (OOD) scenarios caused by environmental uncertainty or rare traffic behaviors in real-world deployment; detecting such OOD events is challenging due to evolving traffic conditions and changing interaction patterns, while safety-critical applications demand formal guarantees on detection delay and false-alarm rates, motivating us-following recent work [1]-to formulate OOD monitoring for trajectory prediction as a quickest changepoint detection (QCD) problem that offers a principled statistical framework with established theory; we further observe that the real-world evolution of prediction errors under in-distribution (ID) conditions can be effectively modeled by a Hidden Markov Model (HMM), and by leveraging this structure we extend the cumulative Maximum Mean Discrepancy approach to enable detection without requiring explicit knowledge of the post-change distribution while still admitting provable guarantees on delay and false alarms, with experiments on three real-world driving datasets demonstrating reduced detection delay and robustness to heavy-tailed errors and unknown post-change conditions.
>
---
#### [new 078] Master Micro Residual Correction with Adaptive Tactile Fusion and Force-Mixed Control for Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人接触密集操作任务，旨在解决复杂交互动态下的控制难题。提出M2-ResiPolicy架构，结合高精度力控与触觉融合，提升操作稳定性和成功率。**

- **链接: [https://arxiv.org/pdf/2603.15152](https://arxiv.org/pdf/2603.15152)**

> **作者:** Xingting Li; Yifan Xie; Han Liu; Wei Hou; Guangyu Chen; Shoujie Li; Wenbo Ding
>
> **摘要:** Robotic contact-rich and fine-grained manipulation remains a significant challenge due to complex interaction dynamics and the competing requirements of multi-timescale control. While current visual imitation learning methods excel at long-horizon planning, they often fail to perceive critical interaction cues like friction variations or incipient slip, and struggle to balance global task coherence with local reactive feedback. To address these challenges, we propose M2-ResiPolicy, a novel Master-Micro residual control architecture that synergizes high-level action guidance with low-level correction. The framework consists of a Master-Guidance Policy (MGP) operating at 10 Hz, which generates temporally consistent action chunks via a diffusion-based backbone and employs a tactile-intensity-driven adaptive fusion mechanism to dynamically modulate perceptual weights between vision and touch. Simultaneously, a high-frequency (60 Hz) Micro-Residual Corrector (MRC) utilizes a lightweight GRU to provide real-time action compensation based on TCP wrench feedback. This policy is further integrated with a force-mixed PBIC execution layer, effectively regulating contact forces to ensure interaction safety. Experiments across several demanding tasks including fragile object grasping and precision insertion, demonstrate that M2-ResiPolicy significantly outperforms standard Diffusion Policy (DP) and state-of-the-art Reactive Diffusion Policy (RDP), achieving a 93\% damage-free success rate in chip grasping and superior force regulation stability.
>
---
#### [new 079] Implicit Maximum Likelihood Estimation for Real-time Generative Model Predictive Control
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决扩散模型推理速度慢的问题。通过引入IMLE方法，提升生成模型的推理速度，以适应实时控制需求。**

- **链接: [https://arxiv.org/pdf/2603.13733](https://arxiv.org/pdf/2603.13733)**

> **作者:** Grayson Lee; Minh Bui; Shuzi Zhou; Yankai Li; Mo Chen; Ke Li
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026. Project page: this https URL
>
> **摘要:** Diffusion-based models have recently shown strong performance in trajectory planning, as they are capable of capturing diverse, multimodal distributions of complex behaviors. A key limitation of these models is their slow inference speed, which results from the iterative denoising process. This makes them less suitable for real-time applications such as closed-loop model predictive control (MPC), where plans must be generated quickly and adapted continuously to a changing environment. In this paper, we investigate Implicit Maximum Likelihood Estimation (IMLE) as an alternative generative modeling approach for planning. IMLE offers strong mode coverage while enabling inference that is two orders of magnitude faster, making it particularly well suited for real-time MPC tasks. Our results demonstrate that IMLE achieves competitive performance on standard offline reinforcement learning benchmarks compared to the standard diffusion-based planner, while substantially improving planning speed in both open-loop and closed-loop settings. We further validate IMLE in a closed-loop human navigation scenario, operating in real-time, demonstrating how it enables rapid and adaptive plan generation in dynamic environments.
>
---
#### [new 080] CycleRL: Sim-to-Real Deep Reinforcement Learning for Robust Autonomous Bicycle Control
- **分类: cs.RO**

- **简介: 该论文属于自主自行车控制任务，旨在解决传统控制方法在非线性动态和现实不确定性下的适应性问题。提出CycleRL框架，通过深度强化学习实现仿真到现实的可靠控制。**

- **链接: [https://arxiv.org/pdf/2603.15013](https://arxiv.org/pdf/2603.15013)**

> **作者:** Gelu Liu; Teng Wang; Zhijie Wu; Junliang Wu; Songyuan Li; Xiangwei Zhu
>
> **备注:** 10 pages, 7 figures, 9 tables
>
> **摘要:** Autonomous bicycles offer a promising agile solution for urban mobility and last-mile logistics, however, conventional control strategies often struggle with their underactuated nonlinear dynamics, suffering from sensitivity to model mismatches and limited adaptability to real-world uncertainties. To address this, this paper presents CycleRL, the first sim-to-real deep reinforcement learning framework designed for robust autonomous bicycle control. Our approach trains an end-to-end neural control policy within the high-fidelity NVIDIA Isaac Sim environment, leveraging Proximal Policy Optimization (PPO) to circumvent the need for an explicit dynamics model. The framework features a composite reward function tailored for concurrent balance maintenance, velocity tracking, and steering control. Crucially, systematic domain randomization is employed to bridge the simulation-to-reality gap and facilitate direct transfer. In simulation, CycleRL achieves considerable performance, including a 99.90% balance success rate, a low steering tracking error of 1.15°, and a velocity tracking error of 0.18 m/s. These quantitative results, coupled with successful hardware transfer, validate DRL as an effective paradigm for autonomous bicycle control, offering superior adaptability over traditional methods. Video demonstrations are available at this https URL.
>
---
#### [new 081] GraspALL: Adaptive Structural Compensation from Illumination Variation for Robotic Garment Grasping in Any Low-Light Conditions
- **分类: cs.RO**

- **简介: 该论文属于机器人服装抓取任务，旨在解决低光环境下光照变化导致的结构特征退化问题。通过自适应融合RGB与非RGB模态特征，提升抓取准确性。**

- **链接: [https://arxiv.org/pdf/2603.14789](https://arxiv.org/pdf/2603.14789)**

> **作者:** Haifeng Zhong; Wenshuo Han; Zhouyu Wang; Runyang Feng; Fan Tang; Tong-Yee Lee; Zipei Fan; Ruihai Wu; Yuran Wang; Hao Dong; Hechang Chen; Hyung Jin Chang; Yixing Gao
>
> **摘要:** Achieving accurate garment grasping under dynamically changing illumination is crucial for all-day operation of service this http URL, the reduced illumination in low-light scenes severely degrades garment structural features, leading to a significant drop in grasping this http URL methods typically enhance RGB features by exploiting the illumination-invariant properties of non-RGB modalities, yet they overlook the varying dependence on non-RGB features under varying lighting conditions, which can introduce misaligned non-RGB cues and thereby weaken the model's adaptability to illumination changes when utilizing multimodal this http URL address this problem, we propose GraspALL, an illumination-structure interactive compensation this http URL innovation of GraspALL lies in encoding continuous illumination changes into quantitative references to guide adaptive feature fusion between RGB and non-RGB modalities according to varying lighting intensities, thereby generating illumination-consistent grasping this http URL on the self-built garment grasping dataset demonstrate that GraspALL improves grasping accuracy by 32-44% over baselines under diverse illumination conditions.
>
---
#### [new 082] Perception-Aware Autonomous Exploration in Feature-Limited Environments
- **分类: cs.RO**

- **简介: 该论文属于自主探索任务，解决特征稀疏环境下视觉惯性里程计性能下降的问题。通过感知-aware框架提升特征可观察性，优化轨迹以提高探索可靠性。**

- **链接: [https://arxiv.org/pdf/2603.15605](https://arxiv.org/pdf/2603.15605)**

> **作者:** Moji Shi; Rajitha de Silva; Hang Yu; Riccardo Polvara; Marija Popović
>
> **摘要:** Autonomous exploration in unknown environments typically relies on onboard state estimation for localisation and mapping. Existing exploration methods primarily maximise coverage efficiency, but often overlook that visual-inertial odometry (VIO) performance strongly depends on the availability of robust visual features. As a result, exploration policies can drive a robot into feature-sparse regions where tracking degrades, leading to odometry drift, corrupted maps, and mission failure. We propose a hierarchical perception-aware exploration framework for a stereo-equipped unmanned aerial vehicle (UAV) that explicitly couples exploration progress with feature observability. Our approach (i) associates each candidate frontier with an expected feature quality using a global feature map, and prioritises visually informative subgoals, and (ii) optimises a continuous yaw trajectory along the planned motion to maintain stable feature tracks. We evaluate our method in simulation across environments with varying texture levels and in real-world indoor experiments with largely textureless walls. Compared to baselines that ignore feature quality and/or do not optimise continuous yaw, our method maintains more reliable feature tracking, reduces odometry drift, and achieves on average 30\% higher coverage before the odometry error exceeds specified thresholds.
>
---
#### [new 083] Navigation beyond Wayfinding: Robots Collaborating with Visually Impaired Users for Environmental Interactions
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机协作导航任务，旨在解决盲人与机器人在环境交互中的精准定位与动态协调问题。工作包括设计一种可切换模式的协作系统，提升导航安全性与效率。**

- **链接: [https://arxiv.org/pdf/2603.14216](https://arxiv.org/pdf/2603.14216)**

> **作者:** Shaojun Cai; Nuwan Janaka; Ashwin Ram; Janidu Shehan; Yingjia Wan; Kotaro Hara; David Hsu
>
> **备注:** Accepted to ACM/IEEE HRI 2026, 10 pages, 6 figures
>
> **摘要:** Robotic guidance systems have shown promise in supporting blind and visually impaired (BVI) individuals with wayfinding and obstacle avoidance. However, most existing systems assume a clear path and do not support a critical aspect of navigation - environmental interactions that require manipulating objects to enable movement. These interactions are challenging for a human-robot pair because they demand (i) precise localization and manipulation of interaction targets (e.g., pressing elevator buttons) and (ii) dynamic coordination between the user's and robot's movements (e.g., pulling out a chair to sit). We present a collaborative human-robot approach that combines our robotic guide dog's precise sensing and localization capabilities with the user's ability to perform physical manipulation. The system alternates between two modes: lead mode, where the robot detects and guides the user to the target, and adaptation mode, where the robot adjusts its motion as the user interacts with the environment (e.g., opening a door). Evaluation results show that our system enables navigation that is safer, smoother, and more efficient than both a traditional white cane and a non-adaptive guiding system, with the performance gap widening as tasks demand higher precision in locating interaction targets. These findings highlight the promise of human-robot collaboration in advancing assistive technologies toward more generalizable and realistic navigation support.
>
---
#### [new 084] ArrayTac: A tactile display for simultaneous rendering of shape, stiffness and friction
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于触觉反馈任务，旨在解决多维触觉信息同步呈现的问题。研究提出ArrayTac系统，可同时模拟形状、刚度和摩擦，提升触觉仿真真实感。**

- **链接: [https://arxiv.org/pdf/2603.13829](https://arxiv.org/pdf/2603.13829)**

> **作者:** Tianhai Liang; Shiyi Guo; Baiye Cheng; Zhengrong Xue; Han Zhang; Huazhe Xu
>
> **摘要:** Human-computer interaction in the visual and auditory domains has achieved considerable maturity, yet machine-to-human tactile feedback remains underdeveloped. Existing tactile displays struggle to simultaneously render multiple tactile dimensions, such as shape, stiffness, and friction, which limits the realism of haptic simulation. Here, we present ArrayTac, a piezoelectric-driven tactile display capable of simultaneously rendering shape, stiffness, and friction to reproduce realistic haptic signals. The system comprises a 4x4 array of 16 actuator units, each employing a three-stage micro-lever mechanism to amplify the micrometer-scale displacement of the piezoelectric element, with Hall sensor-based closed-loop control at the end effector to enhance response speed and precision. We further implement two end-to-end pipelines: 1) a vision-to-touch framework that converts visual inputs into tactile signals using multimodal foundation models, and 2) a real-time tele-palpation system operating over distances of several thousand kilometers. In user studies, first-time participants accurately identify object shapes and physical properties with high success rates. In a tele-palpation experiment over 1,000km, untrained volunteers correctly identified both the number and type of tumors in a breast phantom with 100% accuracy and precisely localized their positions. The system pioneers a new pathway for high-fidelity haptic feedback by introducing the unprecedented capability to simultaneously render an object's shape, stiffness, and friction, delivering a holistic tactile experience that was previously unattainable.
>
---
#### [new 085] AeroGen: Agentic Drone Autonomy through Single-Shot Structured Prompting & Drone SDK
- **分类: cs.RO; cs.DC**

- **简介: 该论文提出AeroGen框架，解决无人机自主控制程序生成问题。通过结构化提示与SDK集成，提升LLM生成代码的正确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.14236](https://arxiv.org/pdf/2603.14236)**

> **作者:** Kautuk Astu; Yogesh Simmhan
>
> **摘要:** Designing correct UAV autonomy programs is challenging due to joint navigation, sensing and analytics requirements. While LLMs can generate code, their reliability for safety-critical UAVs remains uncertain. This paper presents AeroGen, an open-loop framework that enables consistently correct single-shot AI-generated drone control programs through structured guardrail prompting and integration with the AeroDaaS drone SDK. AeroGen encodes API descriptions, flight constraints and operational world rules directly into the system context prompt, enabling generic LLMs to produce constraint-aware code from user prompts, with minimal example code. We evaluate AeroGen across a diverse benchmark of 20 navigation tasks and 5 drone missions on urban, farm and inspection environments, using both imperative and declarative user prompts. AeroGen generates about 40 lines of AeroDaaS Python code in about 20s per mission, in both real-world and simulations, showing that structured prompting with a well-defined SDK improves robustness, correctness and deployability of LLM-generated drone autonomy programs.
>
---
#### [new 086] A Novel Camera-to-Robot Calibration Method for Vision-Based Floor Measurements
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉标定任务，解决相机与机器人坐标系对齐问题。通过设计参考板，融合激光跟踪与视觉测量，实现高精度标定。**

- **链接: [https://arxiv.org/pdf/2603.15126](https://arxiv.org/pdf/2603.15126)**

> **作者:** Jan Andre Rudolph; Dennis Haitz; Markus Ulrich
>
> **备注:** 8 pages; accepted for publication in the ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences
>
> **摘要:** A novel hand-eye calibration method for ground-observing mobile robots is proposed. While cameras on mobile robots are com- mon, they are rarely used for ground-observing measurement tasks. Laser trackers are increasingly used in robotics for precise localization. A referencing plate is designed to combine the two measurement modalities of laser-tracker 3D metrology and camera- based 2D imaging. It incorporates reflector nests for pose acquisition using a laser tracker and a camera calibration target that is observed by the robot-mounted camera. The procedure comprises estimating the plate pose, the plate-camera pose, and the robot pose, followed by computing the robot-camera transformation. Experiments indicate sub-millimeter repeatability.
>
---
#### [new 087] KoopmanFlow: Spectrally Decoupled Generative Control Policy via Koopman Structural Bias
- **分类: cs.RO**

- **简介: 该论文提出KoopmanFlow，解决机器人操作中稳定全局运动与高频局部修正难以同时建模的问题。通过结构先验和分枝生成机制，提升控制精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.13781](https://arxiv.org/pdf/2603.13781)**

> **作者:** Chengsi Yao; Ge Wang; Kai Kang; Shenhao Yan; Jiahao Yang; Fan Feng; Honghao Cai; Xianxian Zeng; Rongjun Chen; Yiming Zhao; Yatong Han; Xi Li
>
> **摘要:** Generative Control Policies (GCPs) show immense promise in robotic manipulation but struggle to simultaneously model stable global motions and high-frequency local corrections. While modern architectures extract multi-scale spatial features, their underlying Probability Flow ODEs apply a uniform temporal integration schedule. Compressed to a single step for real-time Receding Horizon Control (RHC), uniform ODE solvers mathematically smooth over sparse, high-frequency transients entangled within low-frequency steady states. To decouple these dynamics without accumulating pipelined errors, we introduce KoopmanFlow, a parameter-efficient generative policy guided by a Koopman-inspired structural inductive bias. Operating in a unified multimodal latent space with visual context, KoopmanFlow bifurcates generation at the terminal stage. Because visual conditioning occurs before spectral decomposition, both branches are visually guided yet temporally specialized. A macroscopic branch anchors slow-varying trajectories via single-step Consistency Training, while a transient branch uses Flow Matching to isolate high-frequency residuals stimulated by sudden visual cues (e.g., contacts or occlusions). Guided by an explicit spectral prior and optimized via a novel asymmetric consistency objective, KoopmanFlow establishes a fused co-training mechanism. This allows the variant branch to absorb localized dynamics without multi-stage error accumulation. Extensive experiments show KoopmanFlow significantly outperforms state-of-the-art baselines in contact-rich tasks requiring agile disturbance rejection. By trading a surplus latency buffer for a richer structural prior, KoopmanFlow achieves superior control fidelity and parameter efficiency within real-time deployment limits.
>
---
#### [new 088] On the Derivation of Tightly-Coupled LiDAR-Inertial Odometry with VoxelMap
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于LiDAR-Inertial Odometry任务，旨在通过VoxelMap和卡尔曼滤波框架实现精确的位姿估计，解决传感器融合中的几何建模与概率估计问题。**

- **链接: [https://arxiv.org/pdf/2603.15471](https://arxiv.org/pdf/2603.15471)**

> **作者:** Zhihao Zhan
>
> **摘要:** This note presents a concise mathematical formulation of tightly-coupled LiDAR-Inertial Odometry within an iterated error-state Kalman filter framework using a VoxelMap representation. Rather than proposing a new algorithm, it provides a clear and self-contained derivation that unifies the geometric modeling and probabilistic state estimation through consistent notation and explicit formulations. The document is intended to serve both as a technical reference and as an accessible entry point for a foundational understanding of the system architecture and estimation principles.
>
---
#### [new 089] Multi-Robot Coordination for Planning under Context Uncertainty
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究多机器人在上下文不确定环境中的协同规划问题。针对未知上下文导致的决策偏差，提出两阶段方法，先推断上下文，再按优先级规划路径，确保安全高效协作。**

- **链接: [https://arxiv.org/pdf/2603.13748](https://arxiv.org/pdf/2603.13748)**

> **作者:** Pulkit Rustagi; Kyle Hollins Wray; Sandhya Saisubramanian
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Real-world robots often operate in settings where objective priorities depend on the underlying context of operation. When the underlying context is unknown apriori, multiple robots may have to coordinate to gather informative observations to infer the context, since acting based on an incorrect context can lead to misaligned and unsafe behavior. Once the underlying true context is inferred, the robots optimize their task-specific objectives in the preference order induced by the context. We formalize this problem as a Multi-Robot Context-Uncertain Stochastic Shortest Path (MR-CUSSP), which captures context-relevant information at landmark states through joint observations. Our two-stage solution approach is composed of: (1) CIMOP (Coordinated Inference for Multi-Objective Planning) to compute plans that guide robots toward informative landmarks to efficiently infer the true context, and (2) LCBS (Lexicographic Conflict-Based Search) for collision-free multi-robot path planning with lexicographic objective preferences, induced by the context. We evaluate the algorithms using three simulated domains and demonstrate its practical applicability using five mobile robots in the salp domain setup.
>
---
#### [new 090] Zero-Shot Generalization from Motion Demonstrations to New Tasks
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，解决从有限演示中泛化到新任务的问题。通过构建高斯图，提出Stitching和Chaining框架，实现高效、稳定的动作迁移。**

- **链接: [https://arxiv.org/pdf/2603.15445](https://arxiv.org/pdf/2603.15445)**

> **作者:** Kilian Freitag; Alvin Combrink; Nadia Figueroa
>
> **摘要:** Learning motion policies from expert demonstrations is an essential paradigm in modern robotics. While end-to-end models aim for broad generalization, they require large datasets and computationally heavy inference. Conversely, learning dynamical systems (DS) provides fast, reactive, and provably stable control from very few demonstrations. However, existing DS learning methods typically model isolated tasks and struggle to reuse demonstrations for novel behaviors. In this work, we formalize the problem of combining isolated demonstrations within a shared workspace to enable generalization to unseen tasks. The Gaussian Graph is introduced, which reinterprets spatial components of learned motion primitives as discrete vertices with connections to one another. This formulation allows us to bridge continuous control with discrete graph search. We propose two frameworks leveraging this graph: Stitching, for constructing time-invariant DSs, and Chaining, giving a sequence-based DS for complex motions while retaining convergence guarantees. Simulations and real-robot experiments show that these methods successfully generalize to new tasks where baseline methods fail.
>
---
#### [new 091] LDHP: Library-Driven Hierarchical Planning for Non-prehensile Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文属于非抓取灵巧操作任务，解决传统方法执行不可行或泛化差的问题。提出LDHP框架，分层规划物体路径与抓取序列，确保可行性。**

- **链接: [https://arxiv.org/pdf/2603.13844](https://arxiv.org/pdf/2603.13844)**

> **作者:** Tierui He; Chao Zhao
>
> **备注:** 9 pages
>
> **摘要:** Non-prehensile manipulation is essential for handling thin, large, or otherwise ungraspable objects in unstructured settings. Prior planning and search-based methods often rely on ad-hoc manual designs or generate physically unrealizable motions by ignoring critical gripper properties, while training-based approaches are data-intensive and struggle to generalize to novel, out-of-distribution tasks. We propose a library-driven hierarchical planner (LDHP) that makes executability a first-class design goal: a top-tier contact-state planner proposes object-pose paths using MoveObject primitives, and a bottom-tier grasp planner synthesizes feasible grasp sequences with AdjustGrasp primitives; feasibility is certified by collision checks and quasi-static mechanics, and contact-sensitive segments are recovered via a bounded dichotomy refinement. This gripper-aware decomposition decouples object motion from grasp realizability, yields a task-agnostic pipeline that transfers across manipulation tasks and geometric variations without re-design, and exposes clean hooks for optional learned priors. Real-robot studies on zero-mobility lifting and slot insertion demonstrate consistent execution and robustness to shape and environment changes.
>
---
#### [new 092] R3DP: Real-Time 3D-Aware Policy for Embodied Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出R3DP，解决实时物体操作中的3D感知问题。通过整合大模型3D先验，提升任务成功率并保持实时性。**

- **链接: [https://arxiv.org/pdf/2603.14498](https://arxiv.org/pdf/2603.14498)**

> **作者:** Yuhao Zhang; Wanxi Dong; Yue Shi; Yi Liang; Jingnan Gao; Qiaochu Yang; Yaxing Lyu; Zhixuan Liang; Yibin Liu; Congsheng Xu; Xianda Guo; Wei Sui; Yaohui Jin; Xiaokang Yang; Yanyan Xu; Yao Mu
>
> **摘要:** Embodied manipulation requires accurate 3D understanding of objects and their spatial relations to plan and execute contact-rich actions. While large-scale 3D vision models provide strong priors, their computational cost incurs prohibitive latency for real-time control. We propose Real-time 3D-aware Policy (R3DP), which integrates powerful 3D priors into manipulation policies without sacrificing real-time performance. A core innovation of R3DP is the asynchronous fast-slow collaboration module, which seamlessly integrates large-scale 3D priors into the policy without compromising real-time performance. The system maintains real-time efficiency by querying the pre-trained slow system (VGGT) only on sparse key frames, while simultaneously employing a lightweight Temporal Feature Prediction Network (TFPNet) to predict features for all intermediate frames. By leveraging historical data to exploit temporal correlations, TFPNet explicitly improves task success rates through consistent feature estimation. Additionally, to enable more effective multi-view fusion, we introduce a Multi-View Feature Fuser (MVFF) that aggregates features across views by explicitly incorporating camera intrinsics and extrinsics. R3DP offers a plug-and-play solution for integrating large models into real-time inference systems. We evaluate R3DP against multiple baselines across different visual configurations. R3DP effectively harnesses large-scale 3D priors to achieve superior results, outperforming single-view and multi-view DP by 32.9% and 51.4% in average success rate, respectively. Furthermore, by decoupling heavy 3D reasoning from policy execution, R3DP achieves a 44.8% reduction in inference time compared to a naive DP+VGGT integration.
>
---
#### [new 093] Architecting Autonomy for Safe Microgravity Free-Flyer Inspection
- **分类: cs.RO**

- **简介: 论文研究小卫星在微重力环境下安全执行检查任务的自主架构设计，解决如何将任务需求转化为具体的规划与控制决策问题，提出了一套包含约束处理、安全机制和系统要求的自主框架。**

- **链接: [https://arxiv.org/pdf/2603.14524](https://arxiv.org/pdf/2603.14524)**

> **作者:** Keenan Albee; David C. Sternberg; Alexander Hansson; David Schwartz; Ritwik Majumdar; Oliver Jia-Richards
>
> **备注:** 10 pages, 6 figures, published in the Proceedings of the 2025 IEEE Aerospace Conference
>
> **摘要:** Small free-flying spacecraft can provide vital extravehicular activity (EVA) services like inspection and repair for future orbital outposts like the Lunar Gateway. Operating adjacent to delicate space station and microgravity targets, these spacecraft require formalization to describe the autonomy that a free-flyer inspection mission must provide. This work explores the transformation of general mission requirements for this class of free-flyer into a set of concrete decisions for the planning and control autonomy architectures that will power such missions. Flowing down from operator commands for inspection of important regions and mission time-criticality, a motion planning problem emerges that provides the basis for developing autonomy solutions. Unique constraints are considered such as velocity limitations, pointing, and keep-in/keep-out zones, with mission fallback techniques for providing hierarchical safety guarantees under model uncertainties and failure. Planning considerations such as cost function design and path vs. trajectory control are discussed. The typical inputs and outputs of the planning and control autonomy stack of such a mission are also provided. Notional system requirements such as solve times and propellant use are documented to inform planning and control design. The entire proposed autonomy framework for free-flyer inspection is realized in the SmallSatSim simulation environment, providing a reference example of free-flyer inspection autonomy. The proposed autonomy architecture serves as a blueprint for future implementations of small satellite autonomous inspection in proximity to mission-critical hardware, going beyond the existing literature in terms of both (1) providing realistic system requirements for an autonomous inspection mission and (2) translating these requirements into autonomy design decisions for inspection planning and control.
>
---
#### [new 094] AeroGrab: A Unified Framework for Aerial Grasping in Cluttered Environments
- **分类: cs.RO**

- **简介: 论文提出AeroGrab框架，解决杂乱环境中无人机抓取问题。整合目标识别、主动探索与抓取生成，提升抓取可靠性。**

- **链接: [https://arxiv.org/pdf/2603.15097](https://arxiv.org/pdf/2603.15097)**

> **作者:** Shivansh Pratap Singh; Naveen Sudheer Nair; Samaksh Ujjawal; Sarthak Mishra; Soham Patil; Rishabh Dev Yadav; Spandan Roy
>
> **摘要:** Reliable aerial grasping in cluttered environments remains challenging due to occlusions and collision risks. Existing aerial manipulation pipelines largely rely on centroid-based grasping and lack integration between the grasp pose generation models, active exploration, and language-level task specification, resulting in the absence of a complete end-to-end system. In this work, we present an integrated pipeline for reliable aerial grasping in cluttered environments. Given a scene and a language instruction, the system identifies the target object and actively explores it to gain better views of the object. During exploration, a grasp generation network predicts multiple 6-DoF grasp candidates for each view. Each candidate is evaluated using a collision-aware feasibility framework, and the overall best grasp is selected and executed using standard trajectory generation and control methods. Experiments in cluttered real-world scenarios demonstrate robust and reliable grasp execution, highlighting the effectiveness of combining active perception with feasibility-aware grasp selection for aerial manipulation.
>
---
#### [new 095] Data-Driven Autoregressive Power Prediction for GTernal Robots in the Robotarium
- **分类: cs.RO**

- **简介: 该论文属于机器人能量预测任务，解决多机器人系统中功率模型不准确的问题。通过分析大量数据，构建了基于历史功率的自回归预测模型，实现高精度实时预测。**

- **链接: [https://arxiv.org/pdf/2603.13908](https://arxiv.org/pdf/2603.13908)**

> **作者:** Yassin Abdelmeguid; Ammar Hasan
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Energy-aware algorithms for multi-robot systems require accurate power consumption models, yet existing approaches rely on kinematic approximations that fail to capture the complex dynamics of real hardware. We present a lightweight autoregressive predictor for the GTernal mobile robot platform deployed in the Georgia Tech Robotarium. Through analysis of 48,000 samples collected across six motion trials, we discover that power consumption exhibits strong temporal autocorrelation ($\rho_1 = 0.95$) that dominates kinematic effects. A 7,041-parameter multi-layer perceptron (MLP) achieves $R^2 = 0.90$ on held-out motion patterns by conditioning on recent power history, reaching the theoretical prediction ceiling imposed by measurement noise. Physical validation across seven robots in a collision avoidance scenario yields mean $R^2 = 0.87$, demonstrating zero-shot transfer to unseen robots and behaviors. The predictor runs in 224 $\mu$s per inference, enabling real-time deployment at 150$\times$ the platform's 30 Hz control rate. We release the trained model and dataset to support energy-aware multi-robot algorithm development.
>
---
#### [new 096] eNavi: Event-based Imitation Policies for Low-Light Indoor Mobile Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于室内机器人导航任务，解决低光环境下导航性能下降的问题。通过构建多模态数据集和融合RGB与事件流的策略，提升导航鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.14397](https://arxiv.org/pdf/2603.14397)**

> **作者:** Prithvi Jai Ramesh; Kaustav Chanda; Krishna Vinod; Joseph Raj Vishal; Yezhou Yang; Bharatesh Chakravarthi
>
> **摘要:** Event cameras provide high dynamic range and microsecond-level temporal resolution, making them well-suited for indoor robot navigation, where conventional RGB cameras degrade under fast motion or low-light conditions. Despite advances in event-based perception spanning detection, SLAM, and pose estimation, there remains limited research on end-to-end control policies that exploit the asynchronous nature of event streams. To address this gap, we introduce a real-world indoor person-following dataset collected using a TurtleBot 2 robot, featuring synchronized raw event streams, RGB frames, and expert control actions across multiple indoor maps, trajectories under both normal and low-light conditions. We further build a multimodal data preprocessing pipeline that temporally aligns event and RGB observations while reconstructing ground-truth actions from odometry to support high-quality imitation learning. Building on this dataset, we propose a late-fusion RGB-Event navigation policy that combines dual MobileNet encoders with a transformer-based fusion module trained via behavioral cloning. A systematic evaluation of RGB-only, Event-only, and RGB-Event fusion models across 12 training variations ranging from single-path imitation to general multi-path imitation shows that policies incorporating event data, particularly the fusion model, achieve improved robustness and lower action prediction error, especially in unseen low-light conditions where RGB-only models fail. We release the dataset, synchronization pipeline, and trained models at this https URL
>
---
#### [new 097] Vision-guided Autonomous Dual-arm Extraction Robot for Bell Pepper Harvesting
- **分类: cs.RO**

- **简介: 该论文属于农业机器人任务，旨在解决户外采摘甜椒的自动化问题。提出VADER系统，结合双臂协作与视觉感知，实现高效精准采摘。**

- **链接: [https://arxiv.org/pdf/2603.13987](https://arxiv.org/pdf/2603.13987)**

> **作者:** Kshitij Madhav Bhat; Tom Gao; Abhishek Mathur; Rohit Satishkumar; Francisco Yandun; Dominik Bauer; Nancy Pollard
>
> **备注:** 9 pages; first four authors have equal contribution
>
> **摘要:** Agricultural robotics has emerged as a critical solution to the labor shortages and rising costs associated with manual crop harvesting. Bell pepper harvesting, in particular, is a labor-intensive task, accounting for up to 50% of total production costs. While automated solutions have shown promise in controlled greenhouse environments, harvesting in unstructured outdoor farms remains an open challenge due to environmental variability and occlusion. This paper presents VADER (Vision-guided Autonomous Dual-arm Extraction Robot), a dual-arm mobile manipulation system designed specifically for the autonomous harvesting of bell peppers in outdoor environments. The system integrates a robust perception pipeline coupled with a dual-arm planning framework that coordinates a gripping arm and a cutting arm for extraction. We validate the system through trials in various realistic conditions, demonstrating a harvest success rate exceeding 60% with a cycle time of under 100 seconds per fruit, while also featuring a teleoperation fail-safe based on the GELLO teleoperation framework to ensure robustness. To support robust perception, we contribute a hierarchically structured dataset of over 3,200 images spanning indoor and outdoor domains, pairing wide-field scene images with close-up pepper images to enable a coarse-to-fine training strategy from fruit detection to high-precision pose estimation. The code and dataset will be made publicly available upon acceptance.
>
---
#### [new 098] Your Vision-Language-Action Model Already Has Attention Heads For Path Deviation Detection
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于导航任务，解决VLA模型因视觉推理幻觉导致的路径偏差问题。通过监控少量注意力头实现偏差检测，并引入轻量RL策略进行恢复。**

- **链接: [https://arxiv.org/pdf/2603.13782](https://arxiv.org/pdf/2603.13782)**

> **作者:** Jaehwan Jeong; Evelyn Zhu; Jinying Lin; Emmanuel Jaimes; Tuan-Anh Vu; Jungseock Joo; Sangpil Kim; M. Khalid Jawed
>
> **备注:** Keywords: Vision-Language Action (VLA), Reinforcement Learning (RL), Navigation Path Recovery, Robot Operating System (ROS)
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated strong potential for predicting semantic actions in navigation tasks, demonstrating the ability to reason over complex linguistic instructions and visual contexts. However, they are fundamentally hindered by visual-reasoning hallucinations that lead to trajectory deviations. Addressing this issue has conventionally required training external critic modules or relying on complex uncertainty heuristics. In this work, we discover that monitoring a few attention heads within a frozen VLA model can accurately detect path deviations without incurring additional computational overhead. We refer to these heads, which inherently capture the spatiotemporal causality between historical visual sequences and linguistic instructions, as Navigation Heads. Using these heads, we propose an intuitive, training-free anomaly-detection framework that monitors their signals to detect hallucinations in real time. Surprisingly, among over a thousand attention heads, a combination of just three is sufficient to achieve a 44.6 % deviation detection rate with a low false-positive rate of 11.7 %. Furthermore, upon detecting a deviation, we bypass the heavy VLA model and trigger a lightweight Reinforcement Learning (RL) policy to safely execute a shortest-path rollback. By integrating this entire detection-to-recovery pipeline onto a physical robot, we demonstrate its practical robustness. All source code will be publicly available.
>
---
#### [new 099] PerlAD: Towards Enhanced Closed-loop End-to-end Autonomous Driving with Pseudo-simulation-based Reinforcement Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决闭环执行中模仿学习与强化学习的不足。提出PerlAD方法，通过伪仿真和预测世界模型实现高效训练，提升驾驶性能。**

- **链接: [https://arxiv.org/pdf/2603.14908](https://arxiv.org/pdf/2603.14908)**

> **作者:** Yinfeng Gao; Qichao Zhang; Deqing Liu; Zhongpu Xia; Guang Li; Kun Ma; Guang Chen; Hangjun Ye; Long Chen; Da-Wei Ding; Dongbin Zhao
>
> **备注:** Accepted by IEEE RA-L. Submitted: 2025.12.2; Revised: 2026.2.4; Accepeted: 2026.3.7
>
> **摘要:** End-to-end autonomous driving policies based on Imitation Learning (IL) often struggle in closed-loop execution due to the misalignment between inadequate open-loop training objectives and real driving requirements. While Reinforcement Learning (RL) offers a solution by directly optimizing driving goals via reward signals, the rendering-based training environments introduce the rendering gap and are inefficient due to high computational costs. To overcome these challenges, we present a novel Pseudo-simulation-based RL method for closed-loop end-to-end autonomous driving, PerlAD. Based on offline datasets, PerlAD constructs a pseudo-simulation that operates in vector space, enabling efficient, rendering-free trial-and-error training. To bridge the gap between static datasets and dynamic closed-loop environments, PerlAD introduces a prediction world model that generates reactive agent trajectories conditioned on the ego vehicle's plan. Furthermore, to facilitate efficient planning, PerlAD utilizes a hierarchical decoupled planner that combines IL for lateral path generation and RL for longitudinal speed optimization. Comprehensive experimental results demonstrate that PerlAD achieves state-of-the-art performance on the Bench2Drive benchmark, surpassing the previous E2E RL method by 10.29% in Driving Score without requiring expensive online interactions. Additional evaluations on the DOS benchmark further confirm its reliability in handling safety-critical occlusion scenarios.
>
---
#### [new 100] CyboRacket: A Perception-to-Action Framework for Humanoid Racket Sports
- **分类: cs.RO**

- **简介: 该论文提出CyboRacket框架，解决人形机器人在网球击球任务中的感知-行动耦合问题，实现基于视觉的实时轨迹预测与稳定击球。**

- **链接: [https://arxiv.org/pdf/2603.14605](https://arxiv.org/pdf/2603.14605)**

> **作者:** Peng Ren; Chuan Qi; Haoyang Ge; Qiyuan Su; Xuguo He; Cong Huang; Pei Chi; Jiang Zhao; Kai Chen
>
> **摘要:** Dynamic ball-interaction tasks remain challenging for robots because they require tight perception-action coupling under limited reaction time. This challenge is especially pronounced in humanoid racket sports, where successful interception depends on accurate visual tracking, trajectory prediction, coordinated stepping, and stable whole-body striking. Existing robotic racket-sport systems often rely on external motion capture for state estimation or on task-specific low-level controllers that must be retrained across tasks and platforms. We present CyboRacket, a hierarchical perception-to-action framework for humanoid racket sports that integrates onboard visual perception, physics-based trajectory prediction, and large-scale pre-trained whole-body control. The framework uses onboard cameras to track the incoming object, predicts its future trajectory, and converts the estimated interception state into target end-effector and base-motion commands for whole-body execution by SONIC on the Unitree G1 humanoid robot. We evaluate the proposed framework in a vision-based humanoid tennis-hitting task. Experimental results demonstrate real-time visual tracking, trajectory prediction, and successful striking using purely onboard sensing.
>
---
#### [new 101] Exploration-assisted Bottleneck Transition Toward Robust and Data-efficient Deformable Object Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于可变形物体操作任务，旨在解决模仿学习在分布外状态下的失效问题。提出ExBot框架，通过瓶颈状态和双动作原语实现数据高效且鲁棒的操作。**

- **链接: [https://arxiv.org/pdf/2603.13756](https://arxiv.org/pdf/2603.13756)**

> **作者:** Yujiro Onishi; Ryo Takizawa; Yoshiyuki Ohmura; Yasuo Kuniyoshi
>
> **摘要:** Imitation learning has demonstrated impressive results in robotic manipulation but fails under out-of-distribution (OOD) states. This limitation is particularly critical in Deformable Object Manipulation (DOM), where the near-infinite possible configurations render comprehensive data collection infeasible. Although several methods address OOD states, they typically require exhaustive data or highly precise perception. Such requirements are often impractical for DOM owing to its inherent complexities, including self-occlusion. To address the OOD problem in DOM, we propose a novel framework, Exploration-assisted Bottleneck Transition for Deformable Object Manipulation (ExBot), which addresses the OOD challenge through two key advantages. First, we introduce bottleneck states, standardized configurations that serve as starting points for task execution. This enables the reconceptualization of OOD challenges as the problem of transitioning diverse initial states to these bottleneck states, significantly reducing demonstration requirements. Second, to account for imperfect perception, we partition the OOD state space based on recognizability and employ dual action primitives. This approach enables ExBot to manipulate even unrecognizable states without requiring accurate perception. By concentrating demonstrations around bottleneck states and leveraging exploration to alter perceptual conditions, ExBot achieves both data efficiency and robustness to severe OOD scenarios. Real-world experiments on rope and cloth manipulation demonstrate successful task completion from diverse OOD states, including severe self-occlusions.
>
---
#### [new 102] NavThinker: Action-Conditioned World Models for Coupled Prediction and Planning in Social Navigation
- **分类: cs.RO**

- **简介: 该论文属于社会导航任务，解决机器人在动态人类环境中安全行动的问题。提出NavThinker框架，结合世界模型与强化学习，实现预测与规划的耦合。**

- **链接: [https://arxiv.org/pdf/2603.15359](https://arxiv.org/pdf/2603.15359)**

> **作者:** Tianshuai Hu; Zeying Gong; Lingdong Kong; XiaoDong Mei; Yiyi Ding; Qi Zeng; Ao Liang; Rong Li; Yangyi Zhong; Junwei Liang
>
> **摘要:** Social navigation requires robots to act safely in dynamic human environments. Effective behavior demands thinking ahead: reasoning about how the scene and pedestrians evolve under different robot actions rather than reacting to current observations alone. This creates a coupled prediction-planning challenge, where robot actions and human motion mutually influence each other. To address this challenge, we propose NavThinker, a future-aware framework that couples an action-conditioned world model with on-policy reinforcement learning. The world model operates in the Depth Anything V2 patch feature space and performs autoregressive prediction of future scene geometry and human motion; multi-head decoders then produce future depth maps and human trajectories, yielding a future-aware state aligned with traversability and interaction risk. Crucially, we train the policy with DD-PPO while injecting world-model think-ahead signals via: (i) action-conditioned future features fused into the current observation embedding and (ii) social reward shaping from predicted human trajectories. Experiments on single- and multi-robot Social-HM3D show state-of-the-art navigation success, with zero-shot transfer to Social-MP3D and real-world deployment on a Unitree Go2, validating generalization and practical applicability. Webpage: this https URL.
>
---
#### [new 103] MA-VLCM: A Vision Language Critic Model for Value Estimation of Policies in Multi-Agent Team Settings
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多智能体强化学习任务，旨在解决集中式评论家样本效率低和泛化能力差的问题。通过引入预训练视觉-语言模型作为评论家，提升策略价值估计效果。**

- **链接: [https://arxiv.org/pdf/2603.15418](https://arxiv.org/pdf/2603.15418)**

> **作者:** Shahil Shaik; Aditya Parameshwaran; Anshul Nayak; Jonathon M. Smereka; Yue Wang
>
> **备注:** 7 pages, 6 figures
>
> **摘要:** Multi-agent reinforcement learning (MARL) commonly relies on a centralized critic to estimate the value function. However, learning such a critic from scratch is highly sample-inefficient and often lacks generalization across environments. At the same time, large vision-language-action models (VLAs) trained on internet-scale data exhibit strong multimodal reasoning and zero-shot generalization capabilities, yet directly deploying them for robotic execution remains computationally prohibitive, particularly in heterogeneous multi-robot systems with diverse embodiments and resource constraints. To address these challenges, we propose Multi-Agent Vision-Language-Critic Models (MA-VLCM), a framework that replaces the learned centralized critic in MARL with a pretrained vision-language model fine-tuned to evaluate multi-agent behavior. MA-VLCM acts as a centralized critic conditioned on natural language task descriptions, visual trajectory observations, and structured multi-agent state information. By eliminating critic learning during policy optimization, our approach significantly improves sample efficiency while producing compact execution policies suitable for deployment on resource-constrained robots. Results show good zero-shot return estimation on models with differing VLM backbones on in-distribution and out-of-distribution scenarios in multi-agent team settings
>
---
#### [new 104] See, Learn, Assist: Safe and Self-Paced Robotic Rehabilitation via Video-Based Learning from Demonstration
- **分类: cs.RO**

- **简介: 该论文属于机器人康复任务，旨在解决远程教学与安全康复训练问题。通过视频学习生成运动轨迹，结合动态控制策略，实现精准、安全的康复训练。**

- **链接: [https://arxiv.org/pdf/2603.14160](https://arxiv.org/pdf/2603.14160)**

> **作者:** Ali Alabbas; Camillo Murgia; Joanne Regan; Philip Long
>
> **摘要:** In this paper, we propose a novel framework that allows therapists to teach robot-assisted rehabilitation exercises remotely via RGB-D video. Our system encodes demonstrations as 6-DoF body-centric trajectories using Cartesian Dynamic Movement Primitives (DMPs), ensuring accurate posture-independent spatial generalization across diverse patient anatomies. Crucially, we execute these trajectories through a decoupled hybrid control architecture that constructs a spatially compliant virtual tunnel, paired with an effort-based temporal dilation mechanism. This architecture is applied to three distinct rehabilitation modalities: Passive, Active-Assisted, and Active-Resistive, by dynamically linking the exercise's execution phase to the patient's tangential force contribution. To guarantee safety, a Gaussian Mixture Regression (GMR) model is learned on-the-fly from the patient's own limb. This allows the detection of abnormal interaction forces and, if necessary, reverses the trajectory to prevent injury. Experimental validation demonstrates the system's precision, achieving an average trajectory reproduction error of 3.7cm and a range of motion (ROM) error of 5.5 degrees. Furthermore, dynamic interaction trials confirm that the controller successfully enforces effort-based progression while maintaining strict spatial path adherence against human disturbances.
>
---
#### [new 105] Bi-HIL: Bilateral Control-Based Multimodal Hierarchical Imitation Learning via Subtask-Level Progress Rate and Keyframe Memory for Long-Horizon Contact-Rich Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于长期高接触机器人操作任务，解决部分可观测性和子任务不稳定问题。提出Bi-HIL框架，结合关键帧记忆和子任务进度率，提升长时序操作的稳定性与协调性。**

- **链接: [https://arxiv.org/pdf/2603.13315](https://arxiv.org/pdf/2603.13315)**

> **作者:** Thanpimon Buamanee; Masato Kobayashi; Yuki Uranishi
>
> **摘要:** Long-horizon contact-rich robotic manipulation remains challenging due to partial observability and unstable subtask transitions under contact uncertainty. While hierarchical architectures improve temporal reasoning and bilateral imitation learning enables force-aware control, existing approaches often rely on flat policies that struggle with long-horizon coordination. We propose Bi-HIL, a bilateral control-based multimodal hierarchical imitation learning framework for long-horizon manipulation. Bi-HIL stabilizes hierarchical coordination by integrating keyframe memory with subtask-level progress rate that models phase progression within the active subtask and conditions both high- and low-level policies. We evaluate Bi-HIL on unimanual and bimanual real-robot tasks, demonstrating consistent improvements over flat and ablated variants. The results highlight the importance of explicitly modeling subtask progression together with force-aware control for robust long-horizon manipulation. For additional material, please check: this https URL
>
---
#### [new 106] RoCo Challenge at AAAI 2026: Benchmarking Robotic Collaborative Manipulation for Assembly Towards Industrial Automation
- **分类: cs.RO; cs.AI**

- **简介: 该论文介绍RoCo挑战，聚焦工业机器人协作装配任务，解决高精度行星齿轮箱组装问题。通过仿真与实测结合，评估多任务学习与故障恢复策略的有效性。**

- **链接: [https://arxiv.org/pdf/2603.15469](https://arxiv.org/pdf/2603.15469)**

> **作者:** Haichao Liu; Yuheng Zhou; Zhenyu Wu; Ziheng Ji; Ziyu Shan; Qianzhun Wang; Ruixuan Liu; Zhiyuan Yang; Yejun Gu; Shalman Khan; Shijun Yan; Jun Liu; Haiyue Zhu; Changliu Liu; Jianfei Yang; Jingbing Zhang; Ziwei Wang
>
> **备注:** 16 pages, 8 figures
>
> **摘要:** Embodied Artificial Intelligence (EAI) is rapidly developing, gradually subverting previous autonomous systems' paradigms from isolated perception to integrated, continuous action. This transition is highly significant for industrial robotic manipulation, promising to free human workers from repetitive, dangerous daily labor. To benchmark and advance this capability, we introduce the Robotic Collaborative Assembly Assistance (RoCo) Challenge with a dataset towards simulation and real-world assembly manipulation. Set against the backdrop of human-centered manufacturing, this challenge focuses on a high-precision planetary gearbox assembly task, a demanding yet highly representative operation in modern industry. Built upon a self-developed data collection, training, and evaluation system in Isaac Sim, and utilizing a dual-arm robot for real-world deployment, the challenge operates in two phases. The Simulation Round defines fine-grained task phases for step-wise scoring to handle the long-horizon nature of the assembly. The Real-World Round mirrors this evaluation with physical gearbox components and high-quality teleoperated datasets. The core tasks require assembling an epicyclic gearbox from scratch, including mounting three planet gears, a sun gear, and a ring gear. Attracting over 60 teams and 170+ participants from more than 10 countries, the challenge yielded highly effective solutions, most notably ARC-VLA and RoboCola. Results demonstrate that a dual-model framework for long-horizon multi-task learning is highly effective, and the strategic utilization of recovery-from-failure curriculum data is a critical insight for successful deployment. This report outlines the competition setup, evaluation approach, key findings, and future directions for industrial EAI. Our dataset, CAD files, code, and evaluation results can be found at: this https URL.
>
---
#### [new 107] From Passive Observer to Active Critic: Reinforcement Learning Elicits Process Reasoning for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于机器人操作任务，解决长期任务中过程监督不足的问题。通过引入PRIMO R1框架，将视频MLLM从被动观察者转为主动评价者，提升任务执行的准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.15600](https://arxiv.org/pdf/2603.15600)**

> **作者:** Yibin Liu; Yaxing Lyu; Daqi Gao; Zhixuan Liang; Weiliang Tang; Shilong Mu; Xiaokang Yang; Yao Mu
>
> **备注:** 31 pages
>
> **摘要:** Accurate process supervision remains a critical challenge for long-horizon robotic manipulation. A primary bottleneck is that current video MLLMs, trained primarily under a Supervised Fine-Tuning (SFT) paradigm, function as passive "Observers" that recognize ongoing events rather than evaluating the current state relative to the final task goal. In this paper, we introduce PRIMO R1 (Process Reasoning Induced Monitoring), a 7B framework that transforms video MLLMs into active "Critics". We leverage outcome-based Reinforcement Learning to incentivize explicit Chain-of-Thought generation for progress estimation. Furthermore, our architecture constructs a structured temporal input by explicitly anchoring the video sequence between initial and current state images. Supported by the proposed PRIMO Dataset and Benchmark, extensive experiments across diverse in-domain environments and out-of-domain real-world humanoid scenarios demonstrate that PRIMO R1 achieves state-of-the-art performance. Quantitatively, our 7B model achieves a 50% reduction in the mean absolute error of specialized reasoning baselines, demonstrating significant relative accuracy improvements over 72B-scale general MLLMs. Furthermore, PRIMO R1 exhibits strong zero-shot generalization on difficult failure detection tasks. We establish state-of-the-art performance on RoboFail benchmark with 67.0% accuracy, surpassing closed-source models like OpenAI o1 by 6.0%.
>
---
#### [new 108] GelSphere: An Omnidirectional Rolling Vision-Based Tactile Sensor for Online 3D Reconstruction and Normal Force Estimation
- **分类: cs.RO**

- **简介: 该论文提出GelSphere，一种用于在线3D重建和法向力估计的全向滚动视觉触觉传感器，解决传统传感器方向受限和易损问题。**

- **链接: [https://arxiv.org/pdf/2603.14104](https://arxiv.org/pdf/2603.14104)**

> **作者:** Seoyeon Lee; Mohammad Amin Mirzaee; Wenzhen Yuan
>
> **摘要:** We present GelSphere, a spherical vision-based tactile sensor designed for real-time continuous surface scanning. Unlike traditional vision-based tactile sensors that can only sense locally and are damaged when slid across surfaces, and cylindrical tactile sensors that can only roll along a fixed direction, our design enables omnidirectional rolling on surfaces. We accomplish this through our novel sensing system design, which has steel balls inside the sensor, forming a bearing layer between the gel and the rigid housing that allows rolling motion in all axes. The sensor streams tactile images through Wi-Fi, with online large-surface reconstruction capabilities. We present quantitative results for both reconstruction accuracy and image fusion performance. The results show that our sensor maintains geometric fidelity and high reconstruction accuracy even under multi-directional rolling, enabling uninterrupted surface scanning.
>
---
#### [new 109] Rationale Behind Human-Led Autonomous Truck Platooning
- **分类: cs.RO**

- **简介: 论文探讨人类主导的自动驾驶卡车编队技术，旨在解决全自动驾驶在复杂环境中的技术与社会挑战。通过分析事故数据和行业现状，提出人机协同编队方案，实现安全过渡到完全自动驾驶。**

- **链接: [https://arxiv.org/pdf/2603.13296](https://arxiv.org/pdf/2603.13296)**

> **作者:** Yukun Lu; Chenzhao Li; Xintong Jiang; Qiaoxuan Zhang
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Autonomous trucking has progressed rapidly in recent years, transitioning from early demonstrations to OEM-integrated commercial deployments. However, fully driverless freight operations across heterogeneous climates, infrastructure conditions, and regulatory environments remain technically and socially challenging. This paper presents a systematic rationale for human-led autonomous truck platooning as a pragmatic intermediate pathway. First, we analyze 53 major truck accidents across North America (2021-2026) and show that human-related factors remain the dominant contributors to severe crashes, highlighting both the need for advanced assistance/automated driving systems and the complexity of real-world driving environments. Second, we review recent industry developments and identify persistent limitations in long-tail edge cases, winter operations, remote-region logistics, and large-scale safety validation. Based on these findings, we argue that a human-in-the-loop (HiL) platooning architecture offers layered redundancy, adaptive judgment in uncertain conditions, and a scalable validation framework. Furthermore, the dual-use capability of follower vehicles enables an evolutionary transition from coordinated platooning to independent autonomous operation. Rather than representing a compromise, human-led platooning provides a technically grounded and societally aligned bridge toward large-scale autonomous freight deployment.
>
---
#### [new 110] Confusion-Aware In-Context-Learning for Vision-Language Models in Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决视觉语言模型在可混淆物体场景下的鲁棒性问题。通过提出CAICL方法，提升模型对混淆特征的识别能力。**

- **链接: [https://arxiv.org/pdf/2603.15134](https://arxiv.org/pdf/2603.15134)**

> **作者:** Yayun He; Zuheng Kang; Botao Zhao; Zhouyin Wu; Junqing Peng; Jianzong Wang
>
> **备注:** Accepted by the 29th International Conference on Computer Supported Cooperative Work in Design (CSCWD 2026)
>
> **摘要:** Vision-language models (VLMs) have significantly improved the generalization capabilities of robotic manipulation. However, VLM-based systems often suffer from a lack of robustness, leading to unpredictable errors, particularly in scenarios involving confusable objects. Our preliminary analysis reveals that these failures are mainly caused by shortcut learning problem inherently in VLMs, limiting their ability to accurately distinguish between confusable features. To this end, we propose Confusion-Aware In-Context Learning (CAICL), a method that enhances VLM performance in confusable scenarios for robotic manipulation. The approach begins with confusion localization and analysis, identifying potential sources of confusion. This information is then used as a prompt for the VLM to focus on features most likely to cause misidentification. Extensive experiments on the VIMA-Bench show that CAICL effectively addresses the shortcut learning issue, achieving a 85.5\% success rate and showing good stability across tasks with different degrees of generalization.
>
---
#### [new 111] ReMAP-DP: Reprojected Multi-view Aligned PointMaps for Diffusion Policy
- **分类: cs.RO**

- **简介: 该论文提出ReMAP-DP框架，解决机器人策略缺乏3D空间感知的问题，通过多视角对齐点云与扩散策略融合，提升精准操作能力。**

- **链接: [https://arxiv.org/pdf/2603.14977](https://arxiv.org/pdf/2603.14977)**

> **作者:** Xinzhang Yang; Renjun Wu; Jinyan Liu; Xuesong Li
>
> **摘要:** Generalist robot policies built upon 2D visual representations excel at semantic reasoning but inherently lack the explicit 3D spatial awareness required for high-precision tasks. Existing 3D integration methods struggle to bridge this gap due to the structural irregularity of sparse point clouds and the geometric distortion introduced by multi-view orthographic rendering. To overcome these barriers, we present ReMAP-DP, a novel framework synergizing standardized perspective reprojection with a structure-aware dual-stream diffusion policy. By coupling the re-projected views with pixel-aligned PointMaps, our dual-stream architecture leverages learnable modality embeddings to fuse frozen semantic features and explicit geometric descriptors, ensuring precise implicit patch-level alignment. Extensive experiments across simulation and real-world environments demonstrate ReMAP-DP's superior performance in diverse manipulation tasks. On RoboTwin 2.0, it attains a 59.3% average success rate, outperforming the DP3 baseline by +6.6%. On ManiSkill 3, our method yields a 28% improvement over DP3 on the geometrically challenging Stack Cube task. Furthermore, ReMAP-DP exhibits remarkable real-world robustness, executing high-precision and dynamic manipulations with superior data efficiency from only a handful of demonstrations. Project page is available at: this https URL
>
---
#### [new 112] Verification and Forward Invariance of Control Barrier Functions for Differential-Algebraic Systems
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于控制理论任务，解决DAE系统安全控制问题。针对CBF方法在DAE中的适用性不足，提出DAE-aware CBF，确保安全集前向不变性。**

- **链接: [https://arxiv.org/pdf/2603.13509](https://arxiv.org/pdf/2603.13509)**

> **作者:** Hongchao Zhang; Mohamad H. Kazma; Meiyi Ma; Taylor T. Johnson; Ahmad F. Taha
>
> **摘要:** Differential-algebraic equations (DAEs) arise in power networks, chemical processes, and multibody systems, where algebraic constraints encode physical conservation laws. The safety of such systems is critical, yet safe control is challenging because algebraic constraints restrict allowable state trajectories. Control barrier functions (CBFs) provide computationally efficient safety filters for ordinary differential equation (ODE) systems. However, existing CBF methods are not directly applicable to DAEs due to potential conflicts between the CBF condition and the constraint manifold. This paper introduces DAE-aware CBFs that incorporate the differential-algebraic structure through projected vector fields. We derive conditions that ensure forward invariance of safe sets while preserving algebraic constraints and extend the framework to higher-index DAEs. A systematic verification framework is developed, establishing necessary and sufficient conditions for geometric correctness and feasibility of DAE-aware CBFs. For polynomial systems, sum-of-squares certificates are provided, while for nonpolynomial and neural network candidates, satisfiability modulo theories are used for falsification. The approach is validated on wind turbine and flexible-link manipulator systems.
>
---
#### [new 113] Intelligent Control of Differential Drive Robots Subject to Unmodeled Dynamics with EKF-based State Estimation
- **分类: eess.SY; cs.LG; cs.RO**

- **简介: 该论文属于机器人控制任务，解决DDR在不确定环境下的状态估计与控制问题。融合EKF与神经网络，提升轨迹跟踪性能，减少速度误差。**

- **链接: [https://arxiv.org/pdf/2603.14940](https://arxiv.org/pdf/2603.14940)**

> **作者:** Amos Alwala; Yuchen Hu; Gabriel da Silva Lima; Wallace Moreira Bessa
>
> **摘要:** Reliable control and state estimation of differential drive robots (DDR) operating in dynamic and uncertain environments remains a challenge, particularly when system dynamics are partially unknown and sensor measurements are prone to degradation. This work introduces a unified control and state estimation framework that combines a Lyapunov-based nonlinear controller and Adaptive Neural Networks (ANN) with Extended Kalman Filter (EKF)-based multi-sensor fusion. The proposed controller leverages the universal approximation property of neural networks to model unknown nonlinearities in real time. An online adaptation scheme updates the weights of the radial basis function (RBF), the architecture chosen for the ANN. The learned dynamics are integrated into a feedback linearization (FBL) control law, for which theoretical guarantees of closed-loop stability and asymptotic convergence in a trajectory-tracking task are established through a Lyapunov-like stability analysis. To ensure robust state estimation, the EKF fuses inertial measurement unit (IMU) and odometry from monocular, 2D-LiDAR and wheel encoders. The fused state estimate drives the intelligent controller, ensuring consistent performance even under drift, wheel slip, sensor noise and failure. Gazebo simulations and real-world experiments are done using DDR, demonstrating the effectiveness of the approach in terms of improved velocity tracking performance with reduction in linear and angular velocity errors up to $53.91\%$ and $29.0\%$ in comparison to the baseline FBL.
>
---
#### [new 114] WorldVLM: Combining World Model Forecasting and Vision-Language Reasoning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决场景理解与动态预测的结合问题。提出WorldVLM融合视觉语言模型与世界模型，提升驾驶决策的上下文感知与预测能力。**

- **链接: [https://arxiv.org/pdf/2603.14497](https://arxiv.org/pdf/2603.14497)**

> **作者:** Stefan Englmeier; Katharina Winter; Fabian B. Flohr
>
> **备注:** 8 pages, 6 figures, 5 tables
>
> **摘要:** Autonomous driving systems depend on on models that can reason about high-level scene contexts and accurately predict the dynamics of their surrounding environment. Vision- Language Models (VLMs) have recently emerged as promising tools for decision-making and scene understanding, offering strong capabilities in contextual reasoning. However, their limited spatial comprehension constrains their effectiveness as end-to-end driving models. World Models (WM) internalize environmental dynamics to predict future scene evolution. Recently explored as ego-motion predictors and foundation models for autonomous driving, they represent a promising direction for addressing key challenges in the field, particularly enhancing generalization while maintaining dynamic prediction. To leverage the complementary strengths of context-based decision making and prediction, we propose WorldVLM: A hybrid architecture that unifies VLMs and WMs. In our design, the high-level VLM generates behavior commands to guide the driving WM, enabling interpretable and context-aware actions. We evaluate conditioning strategies and provide insights into the hybrid design challenges.
>
---
#### [new 115] Panoramic Affordance Prediction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于 affordance prediction 任务，旨在解决传统方法在全景图像中的性能问题。通过引入 PAP-12K 数据集和 PAP 框架，提升全景感知的准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.15558](https://arxiv.org/pdf/2603.15558)**

> **作者:** Zixin Zhang; Chenfei Liao; Hongfei Zhang; Harold Haodong Chen; Kanghao Chen; Zichen Wen; Litao Guo; Bin Ren; Xu Zheng; Yinchuan Li; Xuming Hu; Nicu Sebe; Ying-Cong Chen
>
> **摘要:** Affordance prediction serves as a critical bridge between perception and action in embodied AI. However, existing research is confined to pinhole camera models, which suffer from narrow Fields of View (FoV) and fragmented observations, often missing critical holistic environmental context. In this paper, we present the first exploration into Panoramic Affordance Prediction, utilizing 360-degree imagery to capture global spatial relationships and holistic scene understanding. To facilitate this novel task, we first introduce PAP-12K, a large-scale benchmark dataset containing over 1,000 ultra-high-resolution (12k, 11904 x 5952) panoramic images with over 12k carefully annotated QA pairs and affordance masks. Furthermore, we propose PAP, a training-free, coarse-to-fine pipeline inspired by the human foveal visual system to tackle the ultra-high resolution and severe distortion inherent in panoramic images. PAP employs recursive visual routing via grid prompting to progressively locate targets, applies an adaptive gaze mechanism to rectify local geometric distortions, and utilizes a cascaded grounding pipeline to extract precise instance-level masks. Experimental results on PAP-12K reveal that existing affordance prediction methods designed for standard perspective images suffer severe performance degradation and fail due to the unique challenges of panoramic vision. In contrast, PAP framework effectively overcomes these obstacles, significantly outperforming state-of-the-art baselines and highlighting the immense potential of panoramic perception for robust embodied intelligence.
>
---
#### [new 116] Real-Time Monocular Scene Analysis for UAV in Outdoor Environments
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于无人机实时场景分析任务，解决单目深度与语义估计问题。提出Co-SemDepth模型，并构建合成数据集提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.13368](https://arxiv.org/pdf/2603.13368)**

> **作者:** Yara AlaaEldin
>
> **摘要:** In this thesis, we leverage monocular cameras on aerial robots to predict depth and semantic maps in low-altitude unstructured environments. We propose a joint deep-learning architecture, named Co-SemDepth, that can perform the two tasks accurately and rapidly, and validate its effectiveness on a variety of datasets. The training of neural networks requires an abundance of annotated data, and in the UAV field, the availability of such data is limited. We introduce a new synthetic dataset in this thesis, TopAir that contains images captured with a nadir view in outdoor environments at different altitudes, helping to fill the gap. While using synthetic data for the training is convenient, it raises issues when shifting to the real domain for testing. We conduct an extensive analytical study to assess the effect of several factors on the synthetic-to-real generalization. Co-SemDepth and TaskPrompter models are used for comparison in this study. The results reveal a superior generalization performance for Co-SemDepth in depth estimation and for TaskPrompter in semantic segmentation. Also, our analysis allows us to determine which training datasets lead to a better generalization. Moreover, to help attenuate the gap between the synthetic and real domains, image style transfer techniques are explored on aerial images to convert from the synthetic to the realistic style. Cycle-GAN and Diffusion models are employed. The results reveal that diffusion models are better in the synthetic to real style transfer. In the end, we focus on the marine domain and address its challenges. Co-SemDepth is trained on a collected synthetic marine data, called MidSea, and tested on both synthetic and real data. The results reveal good generalization performance of Co-SemDepth when tested on real data from the SMD dataset while further enhancement is needed on the MIT dataset.
>
---
#### [new 117] Voronoi-based Second-order Descriptor with Whitened Metric in LiDAR Place Recognition
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于LiDAR位姿识别任务，解决第二阶池化方法在欧式距离中的不适配问题。通过结合Voronoi单元和白化机制，提升描述符的稳定性与区分度。**

- **链接: [https://arxiv.org/pdf/2603.14974](https://arxiv.org/pdf/2603.14974)**

> **作者:** Jaein Kim; Hee Bin Yoo; Dong-Sig Han; Byoung-Tak Zhang
>
> **备注:** Accepted at ICRA 26
>
> **摘要:** The pooling layer plays a vital role in aggregating local descriptors into the metrizable global descriptor in the LiDAR Place Recognition (LPR). In particular, the second-order pooling is capable of capturing higher-order interactions among local descriptors. However, its existing methods in the LPR adhere to conventional implementations and post-normalization, and incur the descriptor unsuitable for Euclidean distancing. Based on the recent interpretation that associates NetVLAD with the second-order statistics, we propose to integrate second-order pooling with the inductive bias from Voronoi cells. Our novel pooling method aggregates local descriptors to form the second-order matrix and whitens the global descriptor to implicitly measure the Mahalanobis distance while conserving the cluster property from Voronoi cells, addressing its numerical instability during learning with diverse techniques. We demonstrate its performance gains through the experiments conducted on the Oxford Robotcar and Wild-Places benchmarks and analyze the numerical effect of the proposed whitening algorithm.
>
---
#### [new 118] Semi-Automatic Flute Robot and Its Acoustic Sensing
- **分类: cs.HC; cs.RO; cs.SD**

- **简介: 该论文属于音乐机器人任务，旨在解决 flute 自动演奏与音区控制问题。设计了一种半自动长笛机器人，实现自动指法和低音区气流偏移辅助，提升演奏准确性与表现力。**

- **链接: [https://arxiv.org/pdf/2603.14180](https://arxiv.org/pdf/2603.14180)**

> **作者:** Hikari Kuriyama; Hiroaki Sonoda; Kouki Tomiyoshi; Gou Koutaki
>
> **备注:** This paper was submitted to a journal and received thorough reviews with high marks from the experts. Despite addressing three rounds of major revisions, it was ultimately rejected due to an unreasonable reviewer. We are uploading it here as a preprint
>
> **摘要:** Flute performance requires mastery of complex fingering combinations and register-dependent embouchure control, particularly jet offset adjustment for low-register production. Existing haptic and semi-automated systems do not address both aspects simultaneously through mechanical actuation. To our knowledge, no prior system fully automates fingering while mechanically assisting low-register tone production without requiring embouchure control. We developed a semi-automatic flute robot with an automatic fingering mechanism: fourteen servo motors actuate all keys via wire-based and rack-and-pinion drives in response to MIDI input, enabling performers to produce complete musical pieces through airflow alone. A jet offset assist mechanism rotates the head joint by a calibrated $22^\circ$ during low-register passages, shifting the jet offset toward a low-register configuration without modifying the instrument or embouchure. Fundamental frequency estimation confirmed correct pitch production across the chromatic range (C4--C7) and during musical performance. All key and lever movements were completed within 77.50~ms, corresponding to tempo capacity exceeding standard requirements. Harmonic analysis ($\Delta\mathrm{SPL} = \mathrm{SPL}_2 - \mathrm{SPL}_3$) showed a consistent increase in $\Delta$SPL for all low-register notes when activated, consistent with the intended jet offset shift. Head joint rotation completed within 40.00~ms. These results demonstrate mechanical feasibility of integrating automated fingering and register-dependent jet offset assistance under controlled conditions.
>
---
#### [new 119] Transformers As Generalizable Optimal Controllers
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于控制理论任务，旨在用Transformer学习通用最优控制器。解决如何通过单一控制器适配不同LTI系统的问题，通过训练Transformer策略实现近似最优反馈控制。**

- **链接: [https://arxiv.org/pdf/2603.14910](https://arxiv.org/pdf/2603.14910)**

> **作者:** Turki Bin Mohaya; Maitham F. AL-Sunni; John M. Dolan; Peter Seiler
>
> **备注:** 6 pages
>
> **摘要:** We study whether optimal state-feedback laws for a family of heterogeneous Multiple-Input, Multiple-Output (MIMO) Linear Time-Invariant (LTI) systems can be captured by a single learned controller. We train one transformer policy on LQR-generated trajectories from systems with different state and input dimensions, using a shared representation with standardization, padding, dimension encoding, and masked loss. The policy maps recent state history to control actions without requiring plant matrices at inference time. Across a broad set of systems, it achieves empirically small sub-optimality relative to Linear Quadratic Regulator (LQR), remains stabilizing under moderate parameter perturbations, and benefits from lightweight fine-tuning on unseen systems. These results support transformer policies as practical approximators of near-optimal feedback laws over structured linear-system families.
>
---
#### [new 120] Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于视觉状态表示学习任务，旨在解决动态环境中如何有效编码场景元素及其位置的问题。提出CroBo框架，通过全局到局部重建学习像素级场景理解，支持机器人决策。**

- **链接: [https://arxiv.org/pdf/2603.13904](https://arxiv.org/pdf/2603.13904)**

> **作者:** Seokmin Lee; Yunghee Lee; Byeonghyun Pak; Byeongju Woo
>
> **备注:** Preprint
>
> **摘要:** For robotic agents operating in dynamic environments, learning visual state representations from streaming video observations is essential for sequential decision making. Recent self-supervised learning methods have shown strong transferability across vision tasks, but they do not explicitly address what a good visual state should encode. We argue that effective visual states must capture what-is-where by jointly encoding the semantic identities of scene elements and their spatial locations, enabling reliable detection of subtle dynamics across observations. To this end, we propose CroBo, a visual state representation learning framework based on a global-to-local reconstruction objective. Given a reference observation compressed into a compact bottleneck token, CroBo learns to reconstruct heavily masked patches in a local target crop from sparse visible cues, using the global bottleneck token as context. This learning objective encourages the bottleneck token to encode a fine-grained representation of scene-wide semantic entities, including their identities, spatial locations, and configurations. As a result, the learned visual states reveal how scene elements move and interact over time, supporting sequential decision making. We evaluate CroBo on diverse vision-based robot policy learning benchmarks, where it achieves state-of-the-art performance. Reconstruction analyses and perceptual straightness experiments further show that the learned representations preserve pixel-level scene composition and encode what-moves-where across observations.
>
---
#### [new 121] Visualizing Critic Match Loss Landscapes for Interpretation of Online Reinforcement Learning Control Algorithms
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于在线强化学习任务，旨在解决算法性能不稳定的问题。通过可视化批评者损失景观，分析其优化行为，提升算法可解释性。**

- **链接: [https://arxiv.org/pdf/2603.14535](https://arxiv.org/pdf/2603.14535)**

> **作者:** Jingyi Liu; Jian Guo; Eberhard Gill
>
> **备注:** Revised manuscript, submitted to Acta Astronautica
>
> **摘要:** Reinforcement learning has proven its power on various occasions. However, its performance is not always guaranteed when system dynamics change. Instead, it largely relies on users' empirical experience. For reinforcement learning algorithms with an actor-critic structure, the critic neural network reflects the approximation and optimization process in the RL algorithm. Analyzing the performance of the critic neural network helps to understand the mechanism of the algorithm. To support systematic interpretation of such algorithms in dynamic control problems, this work proposes a critic match loss landscape visualization method for online reinforcement learning. The method constructs a loss landscape by projecting recorded critic parameter trajectories onto a low-dimensional linear subspace. The critic match loss is evaluated over the projected parameter grid using fixed reference state samples and temporal-difference targets. This yields a three-dimensional loss surface together with a two-dimensional optimization path that characterizes critic learning behavior. To extend analysis beyond visual inspection, quantitative landscape indices and a normalized system performance index are introduced, enabling structured comparison across different training outcomes. The approach is demonstrated using the Action-Dependent Heuristic Dynamic Programming algorithm on cart-pole and spacecraft attitude control tasks. Comparative analyses across projection methods and training stages reveal distinct landscape characteristics associated with stable convergence and unstable learning. The proposed framework enables both qualitative and quantitative interpretation of critic optimization behavior in online reinforcement learning.
>
---
#### [new 122] Kimodo: Scaling Controllable Human Motion Generation
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文属于人体运动生成任务，旨在解决数据量小导致的运动质量与控制精度不足问题。通过构建Kimodo模型，利用大规模动作捕捉数据实现高质量、可控制的人体运动生成。**

- **链接: [https://arxiv.org/pdf/2603.15546](https://arxiv.org/pdf/2603.15546)**

> **作者:** Davis Rempe; Mathis Petrovich; Ye Yuan; Haotian Zhang; Xue Bin Peng; Yifeng Jiang; Tingwu Wang; Umar Iqbal; David Minor; Michael de Ruyter; Jiefeng Li; Chen Tessler; Edy Lim; Eugene Jeong; Sam Wu; Ehsan Hassani; Michael Huang; Jin-Bey Yu; Chaeyeon Chung; Lina Song; Olivier Dionne; Jan Kautz; Simon Yuen; Sanja Fidler
>
> **备注:** Project page: this https URL
>
> **摘要:** High-quality human motion data is becoming increasingly important for applications in robotics, simulation, and entertainment. Recent generative models offer a potential data source, enabling human motion synthesis through intuitive inputs like text prompts or kinematic constraints on poses. However, the small scale of public mocap datasets has limited the motion quality, control accuracy, and generalization of these models. In this work, we introduce Kimodo, an expressive and controllable kinematic motion diffusion model trained on 700 hours of optical motion capture data. Our model generates high-quality motions while being easily controlled through text and a comprehensive suite of kinematic constraints including full-body keyframes, sparse joint positions/rotations, 2D waypoints, and dense 2D paths. This is enabled through a carefully designed motion representation and two-stage denoiser architecture that decomposes root and body prediction to minimize motion artifacts while allowing for flexible constraint conditioning. Experiments on the large-scale mocap dataset justify key design decisions and analyze how the scaling of dataset size and model size affect performance.
>
---
#### [new 123] HSImul3R: Physics-in-the-Loop Reconstruction of Simulation-Ready Human-Scene Interactions
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出HSImul3R，解决人体与场景交互的3D重建问题，通过物理约束优化，实现稳定仿真。**

- **链接: [https://arxiv.org/pdf/2603.15612](https://arxiv.org/pdf/2603.15612)**

> **作者:** Yukang Cao; Haozhe Xie; Fangzhou Hong; Long Zhuo; Zhaoxi Chen; Liang Pan; Ziwei Liu
>
> **备注:** this https URL
>
> **摘要:** We present HSImul3R, a unified framework for simulation-ready 3D reconstruction of human-scene interactions (HSI) from casual captures, including sparse-view images and monocular videos. Existing methods suffer from a perception-simulation gap: visually plausible reconstructions often violate physical constraints, leading to instability in physics engines and failure in embodied AI applications. To bridge this gap, we introduce a physically-grounded bi-directional optimization pipeline that treats the physics simulator as an active supervisor to jointly refine human dynamics and scene geometry. In the forward direction, we employ Scene-targeted Reinforcement Learning to optimize human motion under dual supervision of motion fidelity and contact stability. In the reverse direction, we propose Direct Simulation Reward Optimization, which leverages simulation feedback on gravitational stability and interaction success to refine scene geometry. We further present HSIBench, a new benchmark with diverse objects and interaction scenarios. Extensive experiments demonstrate that HSImul3R produces the first stable, simulation-ready HSI reconstructions and can be directly deployed to real-world humanoid robots.
>
---
#### [new 124] Deconfounded Lifelong Learning for Autonomous Driving via Dynamic Knowledge Spaces
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于自主驾驶任务，解决终身学习中的遗忘、知识迁移和混淆问题。提出DeLL框架，结合DPMM和因果调整，构建动态知识空间，提升模型适应性和性能。**

- **链接: [https://arxiv.org/pdf/2603.14354](https://arxiv.org/pdf/2603.14354)**

> **作者:** Jiayuan Du; Yuebing Song; Yiming Zhao; Xianghui Pan; Jiawei Lian; Yuchu Lu; Liuyi Wang; Chengju Liu; Qijun Chen
>
> **摘要:** End-to-End autonomous driving (E2E-AD) systems face challenges in lifelong learning, including catastrophic forgetting, difficulty in knowledge transfer across diverse scenarios, and spurious correlations between unobservable confounders and true driving intents. To address these issues, we propose DeLL, a Deconfounded Lifelong Learning framework that integrates a Dirichlet process mixture model (DPMM) with the front-door adjustment mechanism from causal inference. The DPMM is employed to construct two dynamic knowledge spaces: a trajectory knowledge space for clustering explicit driving behaviors and an implicit feature knowledge space for discovering latent driving abilities. Leveraging the non-parametric Bayesian nature of DPMM, our framework enables adaptive expansion and incremental updating of knowledge without predefining the number of clusters, thereby mitigating catastrophic forgetting. Meanwhile, the front-door adjustment mechanism utilizes the DPMM-derived knowledge as valid mediators to deconfound spurious correlations, such as those induced by sensor noise or environmental changes, and enhances the causal expressiveness of the learned representations. Additionally, we introduce an evolutionary trajectory decoder that enables non-autoregressive planning. To evaluate the lifelong learning performance of E2E-AD, we propose new evaluation protocols and metrics based on Bench2Drive. Extensive evaluations in the closed-loop CARLA simulator demonstrate that our framework significantly improves adaptability to new driving scenarios and overall driving performance, while effectively retaining previous acquired knowledge.
>
---
#### [new 125] Towards Generalizable Robotic Manipulation in Dynamic Environments
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于动态环境下的机器人操作任务，解决静态模型在动态场景中表现不佳的问题。通过构建DOMINO数据集和提出PUMA架构，提升模型的时空推理能力。**

- **链接: [https://arxiv.org/pdf/2603.15620](https://arxiv.org/pdf/2603.15620)**

> **作者:** Heng Fang; Shangru Li; Shuhan Wang; Xuanyang Xi; Dingkang Liang; Xiang Bai
>
> **摘要:** Vision-Language-Action (VLA) models excel in static manipulation but struggle in dynamic environments with moving targets. This performance gap primarily stems from a scarcity of dynamic manipulation datasets and the reliance of mainstream VLAs on single-frame observations, restricting their spatiotemporal reasoning capabilities. To address this, we introduce DOMINO, a large-scale dataset and benchmark for generalizable dynamic manipulation, featuring 35 tasks with hierarchical complexities, over 110K expert trajectories, and a multi-dimensional evaluation suite. Through comprehensive experiments, we systematically evaluate existing VLAs on dynamic tasks, explore effective training strategies for dynamic awareness, and validate the generalizability of dynamic data. Furthermore, we propose PUMA, a dynamics-aware VLA architecture. By integrating scene-centric historical optical flow and specialized world queries to implicitly forecast object-centric future states, PUMA couples history-aware perception with short-horizon prediction. Results demonstrate that PUMA achieves state-of-the-art performance, yielding a 6.3% absolute improvement in success rate over baselines. Moreover, we show that training on dynamic data fosters robust spatiotemporal representations that transfer to static tasks. All code and data are available at this https URL.
>
---
#### [new 126] Formalisms for Robotic Mission Specification and Execution: A Comparative Analysis
- **分类: cs.SE; cs.RO**

- **简介: 论文探讨了机器人任务规范的建模方法，比较了四种形式化方法的适用性。旨在解决多机器人系统任务描述不统一的问题，分析其表达能力和工具支持。**

- **链接: [https://arxiv.org/pdf/2603.15427](https://arxiv.org/pdf/2603.15427)**

> **作者:** Gianluca Filippone; Sara Pettinari; Patrizio Pelliccione
>
> **摘要:** Robots are increasingly deployed across diverse domains and designed for multi-purpose operation. As robotic systems grow in complexity and operate in dynamic environments, the need for structured, expressive, and scalable mission-specification approaches becomes critical, with mission specifications often defined in the field by domain experts rather than robotics specialists. However, there is no standard or widely accepted formalism for specifying missions in single- or multi-robot systems. A variety of formalisms, such as Behavior Trees, State Machines, Hierarchical Task Networks, and Business Process Model and Notation, have been adopted in robotics to varying degrees, each providing different levels of abstraction, expressiveness, and support for integration with human workflows and external devices. This paper presents a systematic analysis of these four formalisms with respect to their suitability for robot mission specification. Our study focuses on mission-level descriptions rather than robot software development. We analyze their underlying control structures and mission concepts, evaluate their expressiveness and limitations in modeling real-world missions, and assess the extent of available tool support. By comparing the formalisms and validating our findings with experts, we provide insights into their applicability, strengths, and shortcomings in robotic system modeling. The results aim to support practitioners and researchers in selecting appropriate modeling approaches for designing robust and adaptable robot and multi-robot missions.
>
---
#### [new 127] Global Truncated Loss Minimization for Robust and Threshold-Resilient Geometric Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于几何估计任务，旨在解决鲁棒性和阈值敏感性问题。提出GTM框架，通过全局优化截断损失，提升效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.14796](https://arxiv.org/pdf/2603.14796)**

> **作者:** Tianyu Huang; Liangzu Peng; Xinyue Zhang; Tongfan Guan; Jinhu Dong; Haoang Li; Laurent Kneip; Yun-Hui Liu
>
> **备注:** 19 pages, 10 figures
>
> **摘要:** To achieve outlier-robust geometric estimation, robust objective functions are generally employed to mitigate the influence of outliers. The widely used consensus maximization(CM) is highly robust when paired with global branch-and-bound(BnB) search. However, CM relies solely on inlier counts and is sensitive to the inlier threshold. Besides, the discrete nature of CM leads to loose bounds, necessitating extensive BnB iterations and computation cost. Truncated losses(TL), another continuous alternative, leverage residual information more effectively and could potentially overcome these issues. But to our knowledge, no prior work has systematically explored globally minimizing TL with BnB and its potential for enhanced threshold resilience or search efficiency. In this work, we propose GTM, the first unified BnB-based framework for globally-optimal TL loss minimization across diverse geometric problems. GTM involves a hybrid solving design: given an n-dimensional problem, it performs BnB search over an (n-1)-dimensional subspace while the remaining 1D variable is solved by bounding the objective function. Our hybrid design not only reduces the search space, but also enables us to derive Lipschitz-continuous bounding functions that are general, tight, and can be efficiently solved by a classic global Lipschitz solver named DIRECT, which brings further acceleration. We conduct a systematic evaluation on various BnB-based methods for CM and TL on the robust linear regression problem, showing that GTM enjoys remarkable threshold resilience and the highest efficiency compared to baseline methods. Furthermore, we apply GTM on different geometric estimation problems with diverse residual forms. Extensive experiments demonstrate that GTM achieves state-of-the-art outlier-robustness and threshold-resilience while maintaining high efficiency across these estimation tasks.
>
---
#### [new 128] A Loss Landscape Visualization Framework for Interpreting Reinforcement Learning: An ADHDP Case Study
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习解释任务，旨在解决算法内部行为难以理解的问题。提出一个框架，通过多视角分析价值估计、策略优化和时序差分信号的交互，以提升对ADHDP算法训练过程的理解。**

- **链接: [https://arxiv.org/pdf/2603.14600](https://arxiv.org/pdf/2603.14600)**

> **作者:** Jingyi Liu; Jian Guo; Eberhard Gill
>
> **备注:** Submitted to Acta Astronautica
>
> **摘要:** Reinforcement learning algorithms have been widely used in dynamic and control systems. However, interpreting their internal learning behavior remains a challenge. In the authors' previous work, a critic match loss landscape visualization method was proposed to study critic training. This study extends that method into a framework which provides a multi-perspective view of the learning dynamics, clarifying how value estimation, policy optimization, and temporal-difference (TD) signals interact during training. The proposed framework includes four complementary components; a three-dimensional reconstruction of the critic match loss surface that shows how TD targets shape the optimization geometry; an actor loss landscape under a frozen critic that reveals how the policy exploits that geometry; a trajectory combining time, Bellman error, and policy weights that indicates how updates move across the surface; and a state-TD map that identifies the state regions that drive those updates. The Action-Dependent Heuristic Dynamic Programming (ADHDP) algorithm for spacecraft attitude control is used as a case study. The framework is applied to compare several ADHDP variants and shows how training stabilizers and target updates change the optimization landscape and affect learning stability. Therefore, the proposed framework provides a systematic and interpretable tool for analyzing reinforcement learning behavior across algorithmic designs.
>
---
#### [new 129] Efficient Event Camera Volume System
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出EECVS框架，解决事件相机稀疏输出在机器人系统中的集成问题，通过自适应变换压缩提升重建质量和效率。**

- **链接: [https://arxiv.org/pdf/2603.14738](https://arxiv.org/pdf/2603.14738)**

> **作者:** Juan Camilo Soto; Ian Noronha; Saru Bharti; Upinder Kaur
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Event cameras promise low latency and high dynamic range, yet their sparse output challenges integration into standard robotic pipelines. We introduce \nameframew (Efficient Event Camera Volume System), a novel framework that models event streams as continuous-time Dirac impulse trains, enabling artifact-free compression through direct transform evaluation at event timestamps. Our key innovation combines density-driven adaptive selection among DCT, DTFT, and DWT transforms with transform-specific coefficient pruning strategies tailored to each domain's sparsity characteristics. The framework eliminates temporal binning artifacts while automatically adapting compression strategies based on real-time event density analysis. On EHPT-XC and MVSEC datasets, our framework achieves superior reconstruction fidelity with DTFT delivering the lowest earth mover distance. In downstream segmentation tasks, EECVS demonstrates robust generalization. Notably, our approach demonstrates exceptional cross-dataset generalization: when evaluated with EventSAM segmentation, EECVS achieves mean IoU 0.87 on MVSEC versus 0.44 for voxel grids at 24 channels, while remaining competitive on EHPT-XC. Our ROS2 implementation provides real-time deployment with DCT processing achieving 1.5 ms latency and 2.7X higher throughput than alternative transforms, establishing the first adaptive event compression framework that maintains both computational efficiency and superior generalization across diverse robotic scenarios.
>
---
#### [new 130] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation
- **分类: cs.CV; cs.LG; cs.RO; eess.IV**

- **简介: 该论文属于跨域全景语义分割任务，解决几何失真和语义不确定性问题。提出EDA-PSeg框架，通过EMA和GMA提升模型泛化能力和类别区分度。**

- **链接: [https://arxiv.org/pdf/2603.15475](https://arxiv.org/pdf/2603.15475)**

> **作者:** Yuanfan Zheng; Kunyu Peng; Xu Zheng; Kailun Yang
>
> **备注:** Accepted to CVPR 2026. The code is available at this https URL
>
> **摘要:** Cross-domain panoramic semantic segmentation has attracted growing interest as it enables comprehensive 360° scene understanding for real-world applications. However, it remains particularly challenging due to severe geometric Field of View (FoV) distortions and inconsistent open-set semantics across domains. In this work, we formulate an open-set domain adaptation setting, and propose Extrapolative Domain Adaptive Panoramic Segmentation (EDA-PSeg) framework that trains on local perspective views and tests on full 360° panoramic images, explicitly tackling both geometric FoV shifts across domains and semantic uncertainty arising from previously unseen classes. To this end, we propose the Euler-Margin Attention (EMA), which introduces an angular margin to enhance viewpoint-invariant semantic representation, while performing amplitude and phase modulation to improve generalization toward unseen classes. Additionally, we design the Graph Matching Adapter (GMA), which builds high-order graph relations to align shared semantics across FoV shifts while effectively separating novel categories through structural adaptation. Extensive experiments on four benchmark datasets under camera-shift, weather-condition, and open-set scenarios demonstrate that EDA-PSeg achieves state-of-the-art performance, robust generalization to diverse viewing geometries, and resilience under varying environmental conditions. The code is available at this https URL.
>
---
#### [new 131] Adapting Critic Match Loss Landscape Visualization to Off-policy Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决离线策略下批评者优化的几何分析问题。通过改进损失景观可视化方法，分析SAC算法的优化动态。**

- **链接: [https://arxiv.org/pdf/2603.14589](https://arxiv.org/pdf/2603.14589)**

> **作者:** Jingyi Liu; Jian Guo; Eberhard Gill
>
> **备注:** Revised manuscript, submitted to Astrodynamics
>
> **摘要:** This work extends an established critic match loss landscape visualization method from online to off-policy reinforcement learning (RL), aiming to reveal the optimization geometry behind critic learning. Off-policy RL differs from stepwise online actor-critic learning in its replay-based data flow and target computation. Based on these two structural differences, the critic match loss landscape visualization method is adapted to the Soft Actor-Critic (SAC) algorithm by aligning the loss evaluation with its batch-based data flow and target computation, using a fixed replay batch and precomputed critic targets from the selected policy. Critic parameters recorded during training are projected onto a principal component plane, where the critic match loss is evaluated to form a 3-D landscape with an overlaid 2-D optimization path. Applied to a spacecraft attitude control problem, the resulting landscapes are analyzed both qualitatively and quantitatively using sharpness, basin area, and local anisotropy metrics, together with temporal landscape snapshots. Comparisons between convergent SAC, divergent SAC, and divergent Action-Dependent Heuristic Dynamic Programming (ADHDP) cases reveal distinct geometric patterns and optimization behaviors under different algorithmic structures. The results demonstrate that the adapted critic match loss visualization framework serves as a geometric diagnostic tool for analyzing critic optimization dynamics in replay-based off-policy RL-based control problems.
>
---
#### [new 132] AerialVLA: A Vision-Language-Action Model for UAV Navigation via Minimalist End-to-End Control
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出AerialVLA，用于UAV导航的视觉-语言-动作模型，解决动态环境下的自主控制问题，通过端到端框架实现精准导航与着陆。**

- **链接: [https://arxiv.org/pdf/2603.14363](https://arxiv.org/pdf/2603.14363)**

> **作者:** Peng Xu; Zhengnan Deng; Jiayan Deng; Zonghua Gu; Shaohua Wan
>
> **备注:** 18 pages, 4 figures. Code and demo videos will be available at: this https URL
>
> **摘要:** Vision-Language Navigation (VLN) for Unmanned Aerial Vehicles (UAVs) demands complex visual interpretation and continuous control in dynamic 3D environments. Existing hierarchical approaches rely on dense oracle guidance or auxiliary object detectors, creating semantic gaps and limiting genuine autonomy. We propose AerialVLA, a minimalist end-to-end Vision-Language-Action framework mapping raw visual observations and fuzzy linguistic instructions directly to continuous physical control signals. First, we introduce a streamlined dual-view perception strategy that reduces visual redundancy while preserving essential cues for forward navigation and precise grounding, which additionally facilitates future simulation-to-reality transfer. To reclaim genuine autonomy, we deploy a fuzzy directional prompting mechanism derived solely from onboard sensors, completely eliminating the dependency on dense oracle guidance. Ultimately, we formulate a unified control space that integrates continuous 3-Degree-of-Freedom (3-DoF) kinematic commands with an intrinsic landing signal, freeing the agent from external object detectors for precision landing. Extensive experiments on the TravelUAV benchmark demonstrate that AerialVLA achieves state-of-the-art performance in seen environments. Furthermore, it exhibits superior generalization in unseen scenarios by achieving nearly three times the success rate of leading baselines, validating that a minimalist, autonomy-centric paradigm captures more robust visual-motor representations than complex modular systems.
>
---
#### [new 133] WestWorld: A Knowledge-Encoded Scalable Trajectory World Model for Diverse Robotic Systems
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出WestWorld，解决机器人轨迹建模的可扩展性与知识融合问题。通过系统感知专家和结构嵌入，提升零样本泛化与控制性能。**

- **链接: [https://arxiv.org/pdf/2603.14392](https://arxiv.org/pdf/2603.14392)**

> **作者:** Yuchen Wang; Jiangtao Kong; Sizhe Wei; Xiaochang Li; Haohong Lin; Hongjue Zhao; Tianyi Zhou; Lu Gan; Huajie Shao
>
> **摘要:** Trajectory world models play a crucial role in robotic dynamics learning, planning, and control. While recent works have explored trajectory world models for diverse robotic systems, they struggle to scale to a large number of distinct system dynamics and overlook domain knowledge of physical structures. To address these limitations, we introduce WestWorld, a knoWledge-Encoded Scalable Trajectory World model for diverse robotic systems. To tackle the scalability challenge, we propose a novel system-aware Mixture-of-Experts (Sys-MoE) that dynamically combines and routes specialized experts for different robotic systems via a learnable system embedding. To further enhance zero-shot generalization, we incorporate domain knowledge of robot physical structures by introducing a structural embedding that aligns trajectory representations with morphological information. After pretraining on 89 complex environments spanning diverse morphologies across both simulation and real-world settings, WestWorld achieves significant improvements over competitive baselines in zero- and few-shot trajectory prediction. Additionally, it shows strong scalability across a wide range of robotic environments and significantly improves performance on downstream model-based control for different robots. Finally, we deploy our model on a real-world Unitree Go1, where it demonstrates stable locomotion performance (see our demo on the website: this https URL). The code will be available upon publication.
>
---
#### [new 134] Thermal Image Refinement with Depth Estimation using Recurrent Networks for Monocular ORB-SLAM3
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉SLAM任务，旨在解决GPS-denied环境下无人机的定位与建图问题。通过单目热成像和递归网络实现深度估计与热力定位，提升低光条件下的导航性能。**

- **链接: [https://arxiv.org/pdf/2603.14998](https://arxiv.org/pdf/2603.14998)**

> **作者:** Hürkan Şahin; Huy Xuan Pham; Van Huyen Dang; Alper Yegenoglu; Erdal Kayacan
>
> **备注:** 8 pages, 8 figures, 2 table
>
> **摘要:** Autonomous navigation in GPS-denied and visually degraded environments remains challenging for unmanned aerial vehicles (UAVs). To this end, we investigate the use of a monocular thermal camera as a standalone sensor on a UAV platform for real-time depth estimation and simultaneous localization and mapping (SLAM). To extract depth information from thermal images, we propose a novel pipeline employing a lightweight supervised network with recurrent blocks (RBs) integrated to capture temporal dependencies, enabling more robust predictions. The network combines lightweight convolutional backbones with a thermal refinement network (T-RefNet) to refine raw thermal inputs and enhance feature visibility. The refined thermal images and predicted depth maps are integrated into ORB-SLAM3, enabling thermal-only localization. Unlike previous methods, the network is trained on a custom non-radiometric dataset, obviating the need for high-cost radiometric thermal cameras. Experimental results on datasets and UAV flights demonstrate competitive depth accuracy and robust SLAM performance under low-light conditions. On the radiometric VIVID++ (indoor-dark) dataset, our method achieves an absolute relative error of approximately 0.06, compared to baselines exceeding 0.11. In our non-radiometric indoor set, baseline errors remain above 0.24, whereas our approach remains below 0.10. Thermal-only ORB-SLAM3 maintains a mean trajectory error under 0.4 m.
>
---
#### [new 135] Geometry-Aware Set-Membership Multilateration: Directional Bounds and Anchor Selection
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究基于范围的定位中锚点选择问题，旨在提高定位精度。通过几何分析，提出E-和D型评分，并设计在线不确定性评估方法，提升定位集的准确性。**

- **链接: [https://arxiv.org/pdf/2603.14263](https://arxiv.org/pdf/2603.14263)**

> **作者:** Giuseppe C. Calafiore
>
> **摘要:** In this paper, we study anchor selection for range-based localization under unknown-but-bounded measurement errors. We start from the convex localization set $\X=\Xd\cap\Hset$ recently introduced in \cite{CalafioreSIAM}, where $\Xd$ is a polyhedron obtained from pairwise differences of squared-range equations between the unknown location $x$ and the anchors, and $\Hset$ is the intersection of upper-range hyperspheres. Our first goal is \emph{offline} design: we derive geometry-only E- and D-type scores from the centered scatter matrix $S(A)=AQ_mA\tran$, where $A$ collects the anchor coordinates and $Q_m=I_m-\frac{1}{m}\one\one\tran$ is the centering projector, showing that $\lambda_{\min}(S(A))$ controls worst-direction and diameter surrogates for the polyhedral certificate $\Xd$, while $\det S(A)$ controls principal-axis volume surrogates. Our second goal is \emph{online} uncertainty assessment for a selected subset of anchors: exploiting the special structure $\X=\Xd\cap\Hset$, we derive a simplex-aggregated enclosing ball for $\Hset$ and an exact support-function formula for $\Hset$, which lead to finite hybrid bounds for the actual localization set $\X$, even when the polyhedral certificate deteriorates. Numerical experiments are performed in two dimensions, showing that geometry-based subset selection is close to an oracle combinatorial search, that the D-score slightly dominates the E-score for the area-oriented metric considered here, and that the new $\Hset$-aware certificates track the realized size of the selected localization set closely.
>
---
#### [new 136] AutoMoT: A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决E2E系统中视觉-语言-动作统一与推理效率问题。提出AutoMoT框架，通过异步混合Transformer结构提升性能与效率。**

- **链接: [https://arxiv.org/pdf/2603.14851](https://arxiv.org/pdf/2603.14851)**

> **作者:** Wenhui Huang; Songyan Zhang; Qihang Huang; Zhidong Wang; Zhiqi Mao; Collister Chua; Zhan Chen; Long Chen; Chen Lv
>
> **摘要:** Integrating vision-language models (VLMs) into end-to-end (E2E) autonomous driving (AD) systems has shown promise in improving scene understanding. However, existing integration strategies suffer from several limitations: they either struggle to resolve distribution misalignment between reasoning and action spaces, underexploit the general reasoning capabilities of pretrained VLMs, or incur substantial inference latency during action policy generation, which degrades driving performance. To address these challenges, we propose \OURS in this work, an end-to-end AD framework that unifies reasoning and action generation within a single vision-language-action (VLA) model. Our approach leverages a mixture-of-transformer (MoT) architecture with joint attention sharing, which preserves the general reasoning capabilities of pre-trained VLMs while enabling efficient fast-slow inference through asynchronous execution at different task frequencies. Extensive experiments on multiple benchmarks, under both open- and closed-loop settings, demonstrate that \OURS achieves competitive performance compared to state-of-the-art methods. We further investigate the functional boundary of pre-trained VLMs in AD, examining when AD-tailored fine-tuning is necessary. Our results show that pre-trained VLMs can achieve competitive multi-task scene understanding performance through semantic prompting alone, while fine-tuning remains essential for action-level tasks such as decision-making and trajectory planning. We refer to \href{this https URL}{Project Page} for the demonstration videos and qualitative results.
>
---
#### [new 137] Benchmarking the Energy Cost of Assurance in Neuromorphic Edge Robotics
- **分类: cs.NE; cs.AR; cs.LG; cs.RO**

- **简介: 该论文属于边缘计算任务，解决可信AI在能量受限平台上的高效保障问题。通过设计事件驱动架构，提升系统鲁棒性同时降低能耗。**

- **链接: [https://arxiv.org/pdf/2603.13880](https://arxiv.org/pdf/2603.13880)**

> **作者:** Sylvester Kaczmarek
>
> **备注:** 6 pages, 4 figures. Accepted and presented at the STEAR 2026 Workshop on Sustainable and Trustworthy Edge AI for Robotics, HiPEAC 2026, Krakow, Poland
>
> **摘要:** Deploying trustworthy artificial intelligence on edge robotics imposes a difficult trade-off between high-assurance robustness and energy sustainability. Traditional defense mechanisms against adversarial attacks typically incur significant computational overhead, threatening the viability of power-constrained platforms in environments such as cislunar space. This paper quantifies the energy cost of assurance in event-driven neuromorphic systems. We benchmark the Hierarchical Temporal Defense (HTD) framework on the BrainChip Akida AKD1000 processor against a suite of adversarial temporal attacks. We demonstrate that unlike traditional deep learning defenses which often degrade efficiency significantly with increased robustness, the event-driven nature of the proposed architecture achieves a superior trade-off. The system reduces gradient-based adversarial success rates from 82.1% to 18.7% and temporal jitter success rates from 75.8% to 25.1%, while maintaining an energy consumption of approximately 45 microjoules per inference. We report a counter-intuitive reduction in dynamic power consumption in the fully defended configuration, attributed to volatility-gated plasticity mechanisms that induce higher network sparsity. These results provide empirical evidence that neuromorphic sparsity enables sustainable and high-assurance edge autonomy.
>
---
#### [new 138] Interp3R: Continuous-time 3D Geometry Estimation with Frames and Events
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Interp3R，解决3D几何估计中时间连续性不足的问题，通过融合事件数据与帧数据，实现任意时间点的深度和相机位姿估计。**

- **链接: [https://arxiv.org/pdf/2603.14528](https://arxiv.org/pdf/2603.14528)**

> **作者:** Shuang Guo; Filbert Febryanto; Lei Sun; Guillermo Gallego
>
> **备注:** 18 pages, 6 figures, 5 tables
>
> **摘要:** In recent years, 3D visual foundation models pioneered by pointmap-based approaches such as DUSt3R have attracted a lot of interest, achieving impressive accuracy and strong generalization across diverse scenes. However, these methods are inherently limited to recovering scene geometry only at the discrete time instants when images are captured, leaving the scene evolution during the blind time between consecutive frames largely unexplored. We introduce Interp3R, to the best of our knowledge the first method that enhances pointmap-based models to estimate depth and camera poses at arbitrary time instants. Interp3R leverages asynchronous event data to interpolate pointmaps produced by frame-based models, enabling temporally continuous geometric representations. Depth and camera poses are then jointly recovered by aligning the interpolated pointmaps together with those predicted by the underlying frame-based models into a consistent spatial framework. We train Interp3R exclusively on a synthetic dataset, yet demonstrate strong generalization across a wide range of synthetic and real-world benchmarks. Extensive experiments show that Interp3R outperforms by a considerable margin state-of-the-art baselines that follow a two-stage pipeline of 2D video frame interpolation followed by 3D geometry estimation.
>
---
#### [new 139] HiMemVLN: Enhancing Reliability of Open-Source Zero-Shot Vision-and-Language Navigation with Hierarchical Memory System
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决开放源代码模型在零样本导航中的可靠性问题。提出HiMemVLN框架，通过层次化记忆系统提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.14807](https://arxiv.org/pdf/2603.14807)**

> **作者:** Kailin Lyu; Kangyi Wu; Pengna Li; Xiuyu Hu; Qingyi Si; Cui Miao; Ning Yang; Zihang Wang; Long Xiao; Lianyu Hu; Jingyuan Sun; Ce Hao
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** LLM-based agents have demonstrated impressive zero-shot performance in vision-language navigation (VLN) tasks. However, most zero-shot methods primarily rely on closed-source LLMs as navigators, which face challenges related to high token costs and potential data leakage risks. Recent efforts have attempted to address this by using open-source LLMs combined with a spatiotemporal CoT framework, but they still fall far short compared to closed-source models. In this work, we identify a critical issue, Navigation Amnesia, through a detailed analysis of the navigation process. This issue leads to navigation failures and amplifies the gap between open-source and closed-source methods. To address this, we propose HiMemVLN, which incorporates a Hierarchical Memory System into a multimodal large model to enhance visual perception recall and long-term localization, mitigating the amnesia issue and improving the agent's navigation performance. Extensive experiments in both simulated and real-world environments demonstrate that HiMemVLN achieves nearly twice the performance of the open-source state-of-the-art method. The code is available at this https URL.
>
---
#### [new 140] Efficient Morphology-Control Co-Design via Stackelberg Proximal Policy Optimization
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文属于机器人学中的共设计任务，解决形态与控制策略协同优化问题。针对传统方法忽略控制适应动态的问题，提出Stackelberg PPO方法，提升优化效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.15388](https://arxiv.org/pdf/2603.15388)**

> **作者:** Yanning Dai; Yuhui Wang; Dylan R. Ashley; Jürgen Schmidhuber
>
> **备注:** presented at the Fourteenth International Conference on Learning Representations; 11 pages in main text + 3 pages of references + 23 pages of appendices, 5 figures in main text + 11 figures in appendices, 16 tables in appendices; accompanying website available at this https URL ; source code available at this https URL
>
> **摘要:** Morphology-control co-design concerns the coupled optimization of an agent's body structure and control policy. This problem exhibits a bi-level structure, where the control dynamically adapts to the morphology to maximize performance. Existing methods typically neglect the control's adaptation dynamics by adopting a single-level formulation that treats the control policy as fixed when optimizing morphology. This can lead to inefficient optimization, as morphology updates may be misaligned with control adaptation. In this paper, we revisit the co-design problem from a game-theoretic perspective, modeling the intrinsic coupling between morphology and control as a novel variant of a Stackelberg game. We propose Stackelberg Proximal Policy Optimization (Stackelberg PPO), which explicitly incorporates the control's adaptation dynamics into morphology optimization. By modeling this intrinsic coupling, our method aligns morphology updates with control adaptation, thereby stabilizing training and improving learning efficiency. Experiments across diverse co-design tasks demonstrate that Stackelberg PPO outperforms standard PPO in both stability and final performance, opening the way for dramatically more efficient robotics designs.
>
---
#### [new 141] Distributional Uncertainty and Adaptive Decision-Making in System
- **分类: math.OC; cs.RO; eess.SY**

- **简介: 该论文属于系统设计任务，解决复杂系统在不确定性下的协同设计问题。通过引入分布式不确定性模型，支持自适应决策和概率性设计分析。**

- **链接: [https://arxiv.org/pdf/2603.14047](https://arxiv.org/pdf/2603.14047)**

> **作者:** Yujun Huang; Gioele Zardini
>
> **摘要:** Complex engineered systems require coordinated design choices across heterogeneous components under multiple conflicting objectives and uncertain specifications. Monotone co-design provides a compositional framework for such problems by modeling each subsystem as a design problem: a feasible relation between provided functionalities and required resources in partially ordered sets. Existing uncertain co-design models rely on interval bounds, which support worst-case reasoning but cannot represent probabilistic risk or multi-stage adaptive decisions. We develop a distributional extension of co-design that models uncertain design outcomes as distributions over design problems and supports adaptive decision processes through Markov-kernel re-parameterizations. Using quasi-measurable and quasi-universal spaces, we show that the standard co-design interconnection operations remain compositional under this richer notion of uncertainty. We further introduce queries and observations that extract probabilistic design trade-offs, including feasibility probabilities, confidence bounds, and distributions of minimal required resources. A task-driven unmanned aerial vehicle case study illustrates how the framework captures risk-sensitive and information-dependent design choices that interval-based models cannot express.
>
---
#### [new 142] COT-FM: Cluster-wise Optimal Transport Flow Matching
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出COT-FM，用于改进流匹配模型的生成速度和质量。针对轨迹弯曲导致的误差问题，通过聚类目标样本并分配源分布，提升局部传输精度。**

- **链接: [https://arxiv.org/pdf/2603.13395](https://arxiv.org/pdf/2603.13395)**

> **作者:** Chiensheng Chiang; Kuan-Hsun Tu; Jia-Wei Liao; Cheng-Fu Chou; Tsung-Wei Ke
>
> **备注:** 18pages, CVPR 2026 accepted
>
> **摘要:** We introduce COT-FM, a general framework that reshapes the probability path in Flow Matching (FM) to achieve faster and more reliable generation. FM models often produce curved trajectories due to random or batchwise couplings, which increase discretization error and reduce sample quality. COT-FM fixes this by clustering target samples and assigning each cluster a dedicated source distribution obtained by reversing pretrained FM models. This divide-and-conquer strategy yields more accurate local transport and significantly straighter vector fields, all without changing the model architecture. As a plug-and-play approach, COT-FM consistently accelerates sampling and improves generation quality across 2D datasets, image generation benchmarks, and robotic manipulation tasks.
>
---
#### [new 143] VLA-Thinker: Boosting Vision-Language-Action Models through Thinking-with-Image Reasoning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出VLA-Thinker，解决视觉-语言-动作模型在长任务中被动处理视觉信息的问题。通过动态推理提升机器人操作性能。**

- **链接: [https://arxiv.org/pdf/2603.14523](https://arxiv.org/pdf/2603.14523)**

> **作者:** Chaoyang Wang; Wenrui Bao; Sicheng Gao; Bingxin Xu; Yu Tian; Yogesh S. Rawat; Yunhao Ge; Yuzhang Shang
>
> **备注:** We introduce VLA-Thinker, the first VLA model capable of thinking-with-image reasoning, which models visual perception as a dynamically invocable reasoning action, enabling Multimodal Embodied Chain-of-Thought
>
> **摘要:** Vision-Language-Action (VLA) models have shown promising capabilities for embodied intelligence, but most existing approaches rely on text-based chain-of-thought reasoning where visual inputs are treated as static context. This limits the ability of the model to actively revisit the environment and resolve ambiguities during long-horizon tasks. We propose VLA-Thinker, a thinking-with-image reasoning framework that models perception as a dynamically invocable reasoning action. To train such a system, we introduce a two-stage training pipeline consisting of (1) an SFT cold-start phase with curated visual Chain-of-Thought data to activate structured reasoning and tool-use behaviors, and (2) GRPO-based reinforcement learning to align complete reasoning-action trajectories with task-level success. Extensive experiments on LIBERO and RoboTwin 2.0 benchmarks demonstrate that VLA-Thinker significantly improves manipulation performance, achieving 97.5% success rate on LIBERO and strong gains across long-horizon robotic tasks. Project and Codes: this https URL .
>
---
#### [new 144] Encirclement Guaranteed Finite-Time Capture against Unknown Evader Strategies
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于多智能体协同控制任务，解决未知策略下追捕者包围并有限时间捕捉逃逸者的问题。提出策略确保包围与捕获，且给出捕获时间上界。**

- **链接: [https://arxiv.org/pdf/2603.15278](https://arxiv.org/pdf/2603.15278)**

> **作者:** Dinesh Patra; Prajakta Surve; Ashish R. Hota; Shaunak D. Bopardikar
>
> **摘要:** We consider a pursuit-evasion scenario involving a group of pursuers and a single evader in a two-dimensional unbounded environment. The pursuers aim to capture the evader in finite time while ensuring the evader remains enclosed within the convex hull of their positions until capture, without knowledge of the evader's heading angle. Prior works have addressed the problem of encirclement and capture separately in different contexts. In this paper, we present a class of strategies for the pursuers that guarantee capture in finite time while maintaining encirclement, irrespective of the evader's strategy. Furthermore, we derive an upper bound on the time to capture. Numerical results highlight the effectiveness of the proposed framework against a range of evader strategies.
>
---
#### [new 145] D-Compress: Detail-Preserving LiDAR Range Image Compression for Real-Time Streaming on Resource-Constrained Robots
- **分类: eess.IV; cs.RO**

- **简介: 该论文属于LiDAR点云压缩任务，旨在解决传统方法损失几何细节及动态带宽下率失真优化不足的问题。提出D-Compress框架，结合预测与小波变换，提升压缩效率与精度。**

- **链接: [https://arxiv.org/pdf/2603.13699](https://arxiv.org/pdf/2603.13699)**

> **作者:** Shengqian Wang; Chang Tu; He Chen
>
> **备注:** To appear in IEEE ICRA 2026
>
> **摘要:** Efficient 3D LiDAR point cloud compression (LPCC) and streaming are critical for edge server-assisted robotic systems, enabling real-time communication with compact data representations. A widely adopted approach represents LiDAR point clouds as range images, enabling the direct use of mature image and video compression codecs. However, because these codecs are designed with human visual perception in mind, they often compromise geometric details, which downgrades the performance of downstream robotic tasks such as mapping and object detection. Furthermore, rate-distortion optimization (RDO)-based rate control remains largely underexplored for range image compression (RIC) under dynamic bandwidth conditions. To address these limitations, we propose D-Compress, a new detail-preserving and fast RIC framework tailored for real-time streaming. D-Compress integrates both intra- and inter-frame prediction with an adaptive discrete wavelet transform approach for precise residual compression. Additionally, we introduce a new RDO-based rate control algorithm for RIC through new rate-distortion modeling. Extensive evaluations on various datasets demonstrate the superiority of D-Compress, which outperforms state-of-the-art (SOTA) compression methods in both geometric accuracy and downstream task performance, particularly at compression ratios exceeding 100x, while maintaining real-time execution on resource-constrained hardware. Moreover, evaluations under dynamic bandwidth conditions validate the robustness of its rate control mechanism.
>
---
#### [new 146] Egocentric World Model for Photorealistic Hand-Object Interaction Synthesis
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-动作交互任务，旨在解决真实感手物交互模拟问题。通过引入EgoHOI模型，从动作信号直接生成物理一致的交互过程，无需未来状态信息。**

- **链接: [https://arxiv.org/pdf/2603.13615](https://arxiv.org/pdf/2603.13615)**

> **作者:** Dayou Li; Lulin Liu; Bangya Liu; Shijie Zhou; Jiu Feng; Ziqi Lu; Minghui Zheng; Chenyu You; Zhiwen Fan
>
> **摘要:** To serve as a scalable data source for embodied AI, world models should act as true simulators that infer interaction dynamics strictly from user actions, rather than mere conditional video generators relying on privileged future object states. In this context, egocentric Human-Object Interaction (HOI) world models are critical for predicting physically grounded first-person rollouts. However, building such models is profoundly challenging due to rapid head motions, severe occlusions, and high-DoF hand articulations that abruptly alter contact topologies. Consequently, existing approaches often circumvent these physics challenges by resorting to conditional video generation with access to known future object trajectories. We introduce EgoHOI, an egocentric HOI world model that breaks away from this shortcut to simulate photorealistic, contact-consistent interactions from action signals alone. To ensure physical accuracy without future-state inputs, EgoHOI distills geometric and kinematic priors from 3D estimates into physics-informed embeddings. These embeddings regularize the egocentric rollouts toward physically valid dynamics. Experiments on the HOT3D dataset demonstrate consistent gains over strong baselines, and ablations validate the effectiveness of our physics-informed design.
>
---
#### [new 147] Seeking Physics in Diffusion Noise
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文研究视频扩散模型是否包含物理合理性信号，提出一种基于物理验证的轨迹选择方法，提升物理一致性并降低推理成本。**

- **链接: [https://arxiv.org/pdf/2603.14294](https://arxiv.org/pdf/2603.14294)**

> **作者:** Chujun Tang; Lei Zhong; Fangqiang Ding
>
> **备注:** 32 pages, 8 figures, 10 tables
>
> **摘要:** Do video diffusion models encode signals predictive of physical plausibility? We probe intermediate denoising representations of a pretrained Diffusion Transformer (DiT) and find that physically plausible and implausible videos are partially separable in mid-layer feature space across noise levels. This separability cannot be fully attributed to visual quality or generator identity, suggesting recoverable physics-related cues in frozen DiT features. Leveraging this observation, we introduce progressive trajectory selection, an inference-time strategy that scores parallel denoising trajectories at a few intermediate checkpoints using a lightweight physics verifier trained on frozen features, and prunes low-scoring candidates early. Extensive experiments on PhyGenBench demonstrate that our method improves physical consistency while reducing inference cost, achieving comparable results to Best-of-K sampling with substantially fewer denoising steps.
>
---
## 更新

#### [replaced 001] UMI-on-Air: Embodiment-Aware Guidance for Embodiment-Agnostic Visuomotor Policies
- **分类: cs.RO**

- **简介: 该论文属于视觉运动控制任务，旨在解决将通用操作策略迁移到受限机器人平台的问题。通过引入EADP方法，实现对不同机器人形态的感知与适应，提升执行效果。**

- **链接: [https://arxiv.org/pdf/2510.02614](https://arxiv.org/pdf/2510.02614)**

> **作者:** Harsh Gupta; Xiaofeng Guo; Huy Ha; Chuer Pan; Muqing Cao; Dongjae Lee; Sebastian Scherer; Shuran Song; Guanya Shi
>
> **备注:** Result videos can be found at this http URL
>
> **摘要:** We introduce UMI-on-Air, a framework for embodiment-aware deployment of embodiment-agnostic manipulation policies. Our approach leverages diverse, unconstrained human demonstrations collected with a handheld gripper (UMI) to train generalizable visuomotor policies. A central challenge in transferring these policies to constrained robotic embodiments-such as aerial manipulators-is the mismatch in control and robot dynamics, which often leads to out-of-distribution behaviors and poor execution. To address this, we propose Embodiment-Aware Diffusion Policy (EADP), which couples a high-level UMI policy with a low-level embodiment-specific controller at inference time. By integrating gradient feedback from the controller's tracking cost into the diffusion sampling process, our method steers trajectory generation towards dynamically feasible modes tailored to the deployment embodiment. This enables plug-and-play, embodiment-aware trajectory adaptation at test time. We validate our approach on multiple long-horizon and high-precision aerial manipulation tasks, showing improved success rates, efficiency, and robustness under disturbances compared to unguided diffusion baselines. Finally, we demonstrate deployment in previously unseen environments, using UMI demonstrations collected in the wild, highlighting a practical pathway for scaling generalizable manipulation skills across diverse-and even highly constrained-embodiments. All code, data, checkpoints, and result videos can be found at this http URL.
>
---
#### [replaced 002] Adaptive Sliding Mode Control for Vehicle Platoons with State-Dependent Friction Uncertainty
- **分类: cs.RO**

- **简介: 该论文属于车辆编队控制任务，旨在解决摩擦力不确定性带来的控制难题。提出一种自适应滑模控制器，以维持编队距离和速度，应对未知摩擦力影响。**

- **链接: [https://arxiv.org/pdf/2601.10724](https://arxiv.org/pdf/2601.10724)**

> **作者:** Rishabh Dev Yadav
>
> **备注:** Extended version based on the author MSc thesis. Related to an earlier IEEE ICAR 2021 publication
>
> **摘要:** Multi-robot formation control has various applications in domains such as vehicle troops, platoons, payload transportation, and surveillance. Maintaining formation in a vehicle platoon requires designing a suitable control scheme that can tackle external disturbances and uncertain system parameters while maintaining a predefined safe distance between the robots. A crucial challenge in this context is dealing with the unknown/uncertain friction forces between wheels and the ground, which vary with changes in road surface, wear in tires, and speed of the vehicle. Although state-of-the-art adaptive controllers can handle a priori bounded uncertainties, they struggle with accurately modeling and identifying frictional forces, which are often state-dependent and cannot be a priori bounded. This thesis proposes a new adaptive sliding mode controller for wheeled mobile robot-based vehicle platoons that can handle the unknown and complex behavior of frictional forces without prior knowledge of their parameters and structures. The controller uses the adaptive sliding mode control techniques to regulate the platoon's speed and maintain a predefined inter-robot distance, even in the presence of external disturbances and uncertain system parameters. This approach involves a two-stage process: first, the kinematic controller calculates the desired velocities based on the desired trajectory; and second, the dynamics model generates the commands to achieve the desired motion. By separating the kinematics and dynamics of the robot, this approach can simplify the control problem and allow for more efficient and robust control of the wheeled mobile robot.
>
---
#### [replaced 003] DiffusionRL: Efficient Training of Diffusion Policies for Robotic Grasping Using RL-Adapted Large-Scale Datasets
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决扩散模型在数据有限和场景适应上的问题。通过RL增强数据集并训练轻量扩散策略，提升抓取成功率。**

- **链接: [https://arxiv.org/pdf/2505.18876](https://arxiv.org/pdf/2505.18876)**

> **作者:** Maria Makarova; Qian Liu; Dzmitry Tsetserukou
>
> **摘要:** Diffusion models have been successfully applied in areas such as image, video, and audio generation. Recent works show their promise for sequential decision-making and dexterous manipulation, leveraging their ability to model complex action distributions. However, challenges persist due to the data limitations and scenario-specific adaptation needs. In this paper, we address these challenges by proposing an optimized approach to training diffusion policies using large, pre-built datasets that are enhanced using Reinforcement Learning (RL). Our end-to-end pipeline leverages RL-based enhancement of the DexGraspNet dataset, lightweight diffusion policy training on a dexterous manipulation task for a five-fingered robotic hand, and a pose sampling algorithm for validation. The pipeline achieved a high success rate of 80% for three DexGraspNet objects. By eliminating manual data collection, our approach lowers barriers to adopting diffusion models in robotics, enhancing generalization and robustness for real-world applications.
>
---
#### [replaced 004] REACT3D: Recovering Articulations for Interactive Physical 3D Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出REACT3D，解决静态3D场景交互化问题，通过检测、分割、估计关节等步骤生成可模拟的互动场景，提升场景理解研究效率。**

- **链接: [https://arxiv.org/pdf/2510.11340](https://arxiv.org/pdf/2510.11340)**

> **作者:** Zhao Huang; Boyang Sun; Alexandros Delitzas; Jiaqi Chen; Marc Pollefeys
>
> **备注:** 8 pages
>
> **摘要:** Interactive 3D scenes are increasingly vital for embodied intelligence, yet existing datasets remain limited due to the labor-intensive process of annotating part segmentation, kinematic types, and motion trajectories. We present REACT3D, a scalable zero-shot framework that converts static 3D scenes into simulation-ready interactive replicas with consistent geometry, enabling direct use in diverse downstream tasks. Our contributions include: (i) openable-object detection and segmentation to extract candidate movable parts from static scenes, (ii) articulation estimation that infers joint types and motion parameters, (iii) hidden-geometry completion followed by interactive object assembly, and (iv) interactive scene integration in widely supported formats to ensure compatibility with standard simulation platforms. We achieve state-of-the-art performance on detection/segmentation and articulation metrics across diverse indoor scenes, demonstrating the effectiveness of our framework and providing a practical foundation for scalable interactive scene generation, thereby lowering the barrier to large-scale research on articulated scene understanding. Our project page is this https URL
>
---
#### [replaced 005] World In Your Hands: A Large-Scale and Open-Source Ecosystem for Learning Human-Centric Manipulation in the Wild
- **分类: cs.RO**

- **简介: 该论文提出WIYH生态系统，解决人本操作数据不足问题，通过大规模真实场景数据提升机器人操作成功率。**

- **链接: [https://arxiv.org/pdf/2512.24310](https://arxiv.org/pdf/2512.24310)**

> **作者:** Yupeng Zheng; Jichao Peng; Weize Li; Yuhang Zheng; Xiang Li; Yujie Jin; Julong Wei; Guanhua Zhang; Ruiling Zheng; Ming Cao; Songen Gu; Zhenhong Zou; Kaige Li; Ke Wu; Mingmin Yang; Jiahao Liu; Pengfei Li; Hengjie Si; Feiyu Zhu; Wang Fu; Likun Wang; Ruiwen Yao; Jieru Zhao; Yilun Chen; Wenchao Ding
>
> **备注:** This dataset represents the first large-scale collection of real-world, human-centric multimodal data integrating vision, language, tactile sensing, and action (VLTA) Github: this https URL
>
> **摘要:** We introduce World In Your Hands (WIYH), a large-scale open-source ecosystem comprising over 1,000 hours of human manipulation data collected in-the-wild with millimeter-scale motion accuracy. Specifically, WIYH includes (1) the Oracle Suite, a wearable data collection kit with an auto-labeling pipeline for accurate motion capture; (2) the WIYH Dataset, featuring over 1,000 hours of multimodal manipulation data across hundreds of skills in diverse real-world scenarios; and (3) extensive annotations and benchmarks supporting tasks from perception to action. Furthermore, experiments based on the WIYH ecosystem show that integrating WIYH's human-centric data improves robotic manipulation success rates from 8% to 60% in cluttered scenes. World In Your Hands provides a foundation for advancing human-centric data collection and cross-embodiment policy learning. All data and hardware design will be open-source.
>
---
#### [replaced 006] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉与语言导航任务，旨在解决传统方法在感知和推理上的低效问题。提出ProFocus框架，通过主动感知和聚焦推理提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.05530](https://arxiv.org/pdf/2603.05530)**

> **作者:** Wei Xue; Mingcheng Li; Xuecheng Wu; Jingqun Tang; Dingkang Yang; Lihua Zhang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Vision-and-Language Navigation (VLN) requires agents to accurately perceive complex visual environments and reason over navigation instructions and histories. However, existing methods passively process redundant visual inputs and treat all historical contexts indiscriminately, resulting in inefficient perception and unfocused reasoning. To address these challenges, we propose \textbf{ProFocus}, a training-free progressive framework that unifies \underline{Pro}active Perception and \underline{Focus}ed Reasoning through collaboration between large language models (LLMs) and vision-language models (VLMs). For proactive perception, ProFocus transforms panoramic observations into structured ego-centric semantic maps, enabling the orchestration agent to identify missing visual information needed for reliable decision-making, and to generate targeted visual queries with corresponding focus regions that guide the perception agent to acquire the required observations. For focused reasoning, we propose Branch-Diverse Monte Carlo Tree Search (BD-MCTS) to identify top-$k$ high-value waypoints from extensive historical candidates. The decision agent focuses reasoning on the historical contexts associated with these waypoints, rather than considering all historical waypoints equally. Extensive experiments validate the effectiveness of ProFocus, achieving state-of-the-art performance among zero-shot methods on R2R and REVERIE benchmarks.
>
---
#### [replaced 007] STRIDE: Structured Lagrangian and Stochastic Residual Dynamics via Flow Matching
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出STRIDE框架，用于机器人动力学建模。任务是提升不确定环境下的预测准确性。解决传统模型与数据驱动方法的不足，通过分离刚体力学与随机交互效应，提高控制可靠性。**

- **链接: [https://arxiv.org/pdf/2603.08478](https://arxiv.org/pdf/2603.08478)**

> **作者:** Prakrut Kotecha; Ganga Nair B; Shishir Kolathaya
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Robotic systems operating in unstructured environments must operate under significant uncertainty arising from intermittent contacts, frictional variability, and unmodeled compliance. While recent model-free approaches have demonstrated impressive performance, many deployment settings still require predictive models that support planning, constraint handling, and online adaptation. Analytical rigid-body models provide strong physical structure but often fail to capture complex interaction effects, whereas purely data-driven models may violate physical consistency, exhibit data bias, and accumulate long-horizon drift. In this work, we propose STRIDE, a dynamics learning framework that explicitly separates conservative rigid-body mechanics from uncertain, effectively stochastic non-conservative interaction effects. The structured component is modeled using a Lagrangian Neural Network (LNN) to preserve energy-consistent inertial dynamics, while residual interaction forces are represented using Conditional Flow Matching (CFM) to capture multi-modal interaction phenomena. The two components are trained jointly end-to-end, enabling the model to retain physical structure while representing complex stochastic behavior. We evaluate STRIDE on systems of increasing complexity, including a pendulum, the Unitree Go1 quadruped, and the Unitree G1 humanoid. Results show 20% reduction in long-horizon prediction error and 30% reduction in contact force prediction error compared to deterministic residual baselines, supporting more reliable model-based control in uncertain robotic environments.
>
---
#### [replaced 008] SERFN: Sample-Efficient Real-World Dexterous Policy Fine-Tuning via Action-Chunked Critics and Normalizing Flows
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对现实世界中精细操作策略的微调问题，提出SERFN框架，利用归一化流和动作块评论家，提升样本效率与长期信用分配。**

- **链接: [https://arxiv.org/pdf/2602.09580](https://arxiv.org/pdf/2602.09580)**

> **作者:** Chenyu Yang; Denis Tarasov; Davide Liconti; Hehui Zheng; Robert K. Katzschmann
>
> **备注:** this https URL
>
> **摘要:** Real-world fine-tuning of dexterous manipulation policies remains challenging due to limited real-world interaction budgets and highly multimodal action distributions. Diffusion-based policies, while expressive, do not permit conservative likelihood-based updates during fine-tuning because action probabilities are intractable. In contrast, conventional Gaussian policies collapse under multimodality, particularly when actions are executed in chunks, and standard per-step critics fail to align with chunked execution, leading to poor credit assignment. We present SERFN, a sample-efficient off-policy fine-tuning framework with normalizing flow (NF) to address these challenges. The normalizing flow policy yields exact likelihoods for multimodal action chunks, allowing conservative, stable policy updates through likelihood regularization and thereby improving sample efficiency. An action-chunked critic evaluates entire action sequences, aligning value estimation with the policy's temporal structure and improving long-horizon credit assignment. To our knowledge, this is the first demonstration of a likelihood-based, multimodal generative policy combined with chunk-level value learning on real robotic hardware. We evaluate SERFN on two challenging dexterous manipulation tasks in the real world: cutting tape with scissors retrieved from a case, and in-hand cube rotation with a palm-down grasp -- both of which require precise, dexterous control over long horizons. On these tasks, SERFN achieves stable, sample-efficient adaptation where standard methods struggle.
>
---
#### [replaced 009] Balancing Safety and Optimality in Robot Path Planning: Algorithm and Metric
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人路径规划任务，解决安全与最优性之间的平衡问题。提出UPP算法，通过自适应权重动态优化路径，提升安全性同时保持较短路径。**

- **链接: [https://arxiv.org/pdf/2505.23197](https://arxiv.org/pdf/2505.23197)**

> **作者:** Jatin Kumar Arora; Soutrik Bandyopadhyay; Sunil Sulania; Shubhendu Bhasin
>
> **备注:** 26 pages
>
> **摘要:** Path planning for autonomous robots faces a fundamental trade-off between path length and obstacle clearance. While existing algorithms typically prioritize a single objective, we introduce the Unified Path Planner (UPP), a graph-search algorithm that dynamically balances safety and optimality via adaptive heuristic weighting. UPP employs a local inverse-distance safety field and auto-tunes its parameters based on real-time search progress, achieving provable suboptimality bounds while maintaining superior clearance. To enable rigorous evaluation, we introduce the OptiSafe index, a normalized metric that quantifies the trade-off between safety and optimality. Extensive evaluation across 10 environments shows that UPP achieves a 0.94 OptiSafe score in cluttered environments, compared with 0.22-0.85 for existing methods, with only 0.5-1% path-length overhead in simulation and a 100% success rate. Hardware validation on TurtleBot confirms practical advantages despite sim-to-real gaps.
>
---
#### [replaced 010] Lightweight 3D LiDAR-Based UAV Tracking: An Adaptive Extended Kalman Filtering Approach
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于无人机相对定位任务，解决小无人机在无GPS环境下精准跟踪的问题。通过自适应扩展卡尔曼滤波处理3D LiDAR数据，提升跟踪精度与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.09783](https://arxiv.org/pdf/2603.09783)**

> **作者:** Nivand Khosravi; Meysam Basiri; Rodrigo Ventura
>
> **备注:** Presented at the 19th International Conference on Intelligent Autonomous Systems, IAS-19, Genoa, Italy, June 30 to July 4, 2025. To appear in the Springer post-proceedings of the conference
>
> **摘要:** Accurate relative positioning is crucial for swarm aerial robotics, enabling coordinated flight and collision avoidance. Although vision-based tracking has been extensively studied, 3D LiDAR-based methods remain underutilized despite their robustness under varying lighting conditions. Existing systems often rely on bulky, power-intensive sensors, making them impractical for small UAVs with strict payload and energy constraints. This paper presents a lightweight LiDAR-based UAV tracking system incorporating an Adaptive Extended Kalman Filter (AEKF) framework. Our approach effectively addresses the challenges posed by sparse, noisy, and nonuniform point cloud data generated by non-repetitive scanning 3D LiDARs, ensuring reliable tracking while remaining suitable for small drones with strict payload constraints. Unlike conventional filtering techniques, the proposed method dynamically adjusts the noise covariance matrices using innovation and residual statistics, thereby enhancing tracking accuracy under real-world conditions. Additionally, a recovery mechanism ensures continuity of tracking during temporary detection failures caused by scattered LiDAR returns or occlusions. Experimental validation was performed using a Livox Mid-360 LiDAR mounted on a DJI F550 UAV in real-world flight scenarios. The proposed method demonstrated robust UAV tracking performance under sparse LiDAR returns and intermittent detections, consistently outperforming both standard Kalman filtering and particle filtering approaches during aggressive maneuvers. These results confirm that the framework enables reliable relative positioning in GPS-denied environments without the need for multi-sensor arrays or external infrastructure.
>
---
#### [replaced 011] Multimodal Belief-Space Covariance Steering with Active Probing and Influence for Interactive Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，解决复杂交通中不确定性下的安全交互问题。通过构建多模态信念模型、主动探测策略和风险评估层，实现更安全高效的决策。**

- **链接: [https://arxiv.org/pdf/2602.14540](https://arxiv.org/pdf/2602.14540)**

> **作者:** Devodita Chakravarty; John Dolan; Yiwei Lyu
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** Autonomous driving in complex traffic requires reasoning under uncertainty. Common approaches rely on prediction-based planning or risk-aware control, but these are typically treated in isolation, limiting their ability to capture the coupled nature of action and inference in interactive settings. This gap becomes especially critical in uncertain scenarios, where simply reacting to predictions can lead to unsafe maneuvers or overly conservative behavior. Our central insight is that safe interaction requires not only estimating human behavior but also shaping it when ambiguity poses risks. To this end, we introduce a hierarchical belief model that structures human behavior across coarse discrete intents and fine motion modes, updated via Bayesian inference for interpretable multi-resolution reasoning. On top of this, we develop an active probing strategy that identifies when multimodal ambiguity in human predictions may compromise safety and plans disambiguating actions that both reveal intent and gently steer human decisions toward safer outcomes. Finally, a runtime risk-evaluation layer based on Conditional Value-at-Risk (CVaR) ensures that all probing actions remain within human risk tolerance during influence. Our simulations in lane-merging and unsignaled intersection scenarios demonstrate that our approach achieves higher success rates and shorter completion times compared to existing methods. These results highlight the benefit of coupling belief inference, probing, and risk monitoring, yielding a principled and interpretable framework for planning under uncertainty.
>
---
#### [replaced 012] Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints
- **分类: cs.RO**

- **简介: 该论文属于机器人自主控制任务，解决人形机器人在真实场景中进行守门的挑战。通过强化学习框架，实现自然、动态的人机交互，提升机器人响应能力和动作流畅性。**

- **链接: [https://arxiv.org/pdf/2510.18002](https://arxiv.org/pdf/2510.18002)**

> **作者:** Junli Ren; Junfeng Long; Tao Huang; Huayi Wang; Zirui Wang; Feiyu Jia; Wentao Zhang; Jingbo Wang; Ping Luo; Jiangmiao Pang
>
> **摘要:** We present a reinforcement learning framework for autonomous goalkeeping with humanoid robots in real-world scenarios. While prior work has demonstrated similar capabilities on quadrupedal platforms, humanoid goalkeeping introduces two critical challenges: (1) generating natural, human-like whole-body motions, and (2) covering a wider guarding range with an equivalent response time. Unlike existing approaches that rely on separate teleoperation or fixed motion tracking for whole-body control, our method learns a single end-to-end RL policy, enabling fully autonomous, highly dynamic, and human-like robot-object interactions. To achieve this, we integrate multiple human motion priors conditioned on perceptual inputs into the RL training via an adversarial scheme. We demonstrate the effectiveness of our method through real-world experiments, where the humanoid robot successfully performs agile, autonomous, and naturalistic interceptions of fast-moving balls. In addition to goalkeeping, we demonstrate the generalization of our approach through tasks such as ball escaping and grabbing. Our work presents a practical and scalable solution for enabling highly dynamic interactions between robots and moving objects, advancing the field toward more adaptive and lifelike robotic behaviors.
>
---
#### [replaced 013] Persistent Autoregressive Mapping with Traffic Rules for Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，解决HD地图与交通规则持久性建模问题。提出PAMR框架，实现车道矢量与规则的联合构建与持续一致性维护。**

- **链接: [https://arxiv.org/pdf/2509.22756](https://arxiv.org/pdf/2509.22756)**

> **作者:** Shiyi Liang; Xinyuan Chang; Changjie Wu; Huiyuan Yan; Yifan Bai; Xinran Liu; Hang Zhang; Yujian Yuan; Shuang Zeng; Mu Xu; Xing Wei
>
> **备注:** AAAI2026
>
> **摘要:** Safe autonomous driving requires both accurate HD map construction and persistent awareness of traffic rules, even when their associated signs are no longer visible. However, existing methods either focus solely on geometric elements or treat rules as temporary classifications, failing to capture their persistent effectiveness across extended driving sequences. In this paper, we present PAMR (Persistent Autoregressive Mapping with Traffic Rules), a novel framework that performs autoregressive co-construction of lane vectors and traffic rules from visual observations. Our approach introduces two key mechanisms: Map-Rule Co-Construction for processing driving scenes in temporal segments, and Map-Rule Cache for maintaining rule consistency across these segments. To properly evaluate continuous and consistent map generation, we develop MapDRv2, featuring improved lane geometry annotations. Extensive experiments demonstrate that PAMR achieves superior performance in joint vector-rule mapping tasks, while maintaining persistent rule effectiveness throughout extended driving sequences.
>
---
#### [replaced 014] Decoupled Action Expert: Confining Task Knowledge to the Conditioning Pathway
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于视觉-语言-动作任务，解决动作生成中模型容量过大的问题。通过将任务知识限制在条件路径，验证了动作骨干可通用，提升了效率。**

- **链接: [https://arxiv.org/pdf/2511.12101](https://arxiv.org/pdf/2511.12101)**

> **作者:** Jian Zhou; Sihao Lin; Shuai Fu; Zerui Li; Gengze Zhou; Qi WU
>
> **摘要:** Many recent Vision-Language-Action models employ diffusion or flow-matching backbones with hundreds of millions of parameters for action generation. However, unlike image synthesis where the output spans millions of diverse pixels, a manipulation policy generates only short sequences of low-dimensional, physically correlated action values, a far simpler target that should not demand such capacity. We confirm this intuition and show that task-specific knowledge in these policies can be fully confined to the conditioning pathway, leaving the action backbone task-agnostic. To establish this, we introduce a decoupled training recipe: a general-purpose action head is first pretrained on observation-free forward-kinematics data, then frozen while only the conditioning pathway is trained for downstream tasks. Using Diffusion Policy as a testbed, we show that on both MimicGen and LIBERO, a single frozen backbone shared across all tasks matches normally trained counterparts. This confirms that the action expert encodes little task-specific knowledge. Ablations show that the specific pretraining signal (joint positions, end-effector poses, or no conditioning at all) has no effect on downstream performance, indicating that the backbone learns only general trajectory structure. Pushing this finding further, we replace the 244M U-Net in Diffusion Policy with a 5M-parameter MLP backbone that matches or exceeds its performance, calling into question the large capacity budgets allocated to action generation in current VLA designs.
>
---
#### [replaced 015] DiG-Net: Enhancing Human-Robot Interaction through Hyper-Range Dynamic Gesture Recognition in Assistive Robotics
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于手势识别任务，旨在解决远距离辅助人机交互中的手势识别问题。提出DiG-Net框架，提升远距离手势识别准确率。**

- **链接: [https://arxiv.org/pdf/2505.24786](https://arxiv.org/pdf/2505.24786)**

> **作者:** Eran Bamani Beeri; Eden Nissinman; Avishai Sintov
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2411.18413
>
> **摘要:** Dynamic hand gestures play a pivotal role in assistive human-robot interaction (HRI), facilitating intuitive, non-verbal communication, particularly for individuals with mobility constraints or those operating robots remotely. Current gesture recognition methods are mostly limited to short-range interactions, reducing their utility in scenarios demanding robust assistive communication from afar. In this paper, we present DiG-Net, the first dynamic gesture recognition framework enabling robust operation at hyper-range distances of up to 30 meters, specifically designed for assistive robotics to enhance accessibility and improve quality of life. Our proposed Distance-aware Gesture Network (DiG-Net) effectively combines Depth-Conditioned Deformable Alignment (DADA) blocks with Spatio-Temporal Graph modules, enabling robust processing and classification of gesture sequences captured under challenging conditions, including significant physical attenuation, reduced resolution, and dynamic gesture variations commonly experienced in real-world assistive environments. We further introduce the Radiometric Spatio-Temporal Depth Attenuation Loss (RSTDAL), shown to enhance learning and strengthen model robustness across varying distances. Our model demonstrates significant performance improvement over state-of-the-art gesture recognition frameworks, achieving a recognition accuracy of 97.3% on a diverse dataset with challenging hyper-range gestures. By effectively interpreting gestures from considerable distances, DiG-Net significantly enhances the usability of assistive robots in home healthcare, industrial safety, and remote assistance scenarios, enabling seamless and intuitive interactions for users regardless of physical limitations.
>
---
#### [replaced 016] Social Robots for People Living with Dementia: A Scoping Review on Deception from Design to Perception
- **分类: cs.HC; cs.CY; cs.RO**

- **简介: 该论文属于伦理与人机交互领域，探讨社会机器人在阿尔茨海默病护理中的欺骗问题。通过文献综述，分析设计线索与用户感知，识别可能引发欺骗的要素及用户反应模式。**

- **链接: [https://arxiv.org/pdf/2507.00963](https://arxiv.org/pdf/2507.00963)**

> **作者:** Fan Wang; Giulia Perugia; Yuan Feng; Wijnand IJsselsteijn
>
> **摘要:** As social robots are increasingly introduced into dementia care, their embodied and interactive design may blur the boundary between artificial and lifelike entities, raising ethical concerns about robotic deception. However, it remains unclear which specific design cues of social robots might lead to social robotic deception (SRD) in people living with dementia (PLwD), and which perceptions and responses of PLwD might indicate that SRD is taking place. To address these questions, we conducted a scoping review of 26 empirical studies reporting PLwD interacting with social robots. We identified three key design cue categories that might contribute to SRD and one that might break the illusion. However, the available literature does not provide sufficient evidence to determine which specific design cues lead to SRD. Thematic analysis of user responses reveals six recurring patterns in how PLwD perceive and respond to social robots. However, conceptual limitations in existing definitions of robotic deception make it difficult to identify when and to what extent deception actually occurs. Building on the results, we propose a dual-process interpretation that clarifies the cognitive basis of false beliefs in human-robot interaction and distinguishes SRD from anthropomorphism or emotional engagement.
>
---
#### [replaced 017] Interpretable Responsibility Sharing as a Heuristic for Task and Motion Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出一种名为IRS的启发式方法，用于家庭机器人任务与运动规划（TAMP），解决复杂任务执行效率问题。通过责任共享和辅助物体优化决策，提升规划效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2409.05586](https://arxiv.org/pdf/2409.05586)**

> **作者:** Arda Sarp Yenicesu; Sepehr Nourmohammadi; Berk Cicek; Ozgur S. Oguz
>
> **备注:** Accepted for the Special Issue "Planning and Learning for Autonomous Robotics" in Robotics and Autonomous Systems
>
> **摘要:** This article introduces a novel heuristic for Task and Motion Planning (TAMP) named Interpretable Responsibility Sharing (IRS), which enhances planning efficiency in domestic robots by leveraging human-constructed environments and inherent biases. Utilizing auxiliary objects (e.g., trays and pitchers), which are commonly found in household settings, IRS systematically incorporates these elements to simplify and optimize task execution. The heuristic is rooted in the novel concept of Responsibility Sharing (RS), where auxiliary objects share the task's responsibility with the embodied agent, dividing complex tasks into manageable sub-problems. This division not only reflects human usage patterns but also aids robots in navigating and manipulating within human spaces more effectively. By integrating Optimized Rule Synthesis (ORS) for decision-making, IRS ensures that the use of auxiliary objects is both strategic and context-aware, thereby improving the interpretability and effectiveness of robotic planning. Experiments conducted across various household tasks demonstrate that IRS significantly outperforms traditional methods by reducing the effort required in task execution and enhancing the overall decision-making process. This approach not only aligns with human intuitive methods but also offers a scalable solution adaptable to diverse domestic environments. Code is available at this https URL.
>
---
#### [replaced 018] RAG-3DSG: Enhancing 3D Scene Graphs with Re-Shot Guided Retrieval-Augmented Generation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于3D场景图构建任务，解决因遮挡和视角限制导致的语义不一致问题。提出RAG-3DSG方法，通过重新拍摄引导增强生成，提升场景表示的可靠性。**

- **链接: [https://arxiv.org/pdf/2601.10168](https://arxiv.org/pdf/2601.10168)**

> **作者:** Yue Chang; Rufeng Chen; Zhaofan Zhang; Yi Chen; Yifan Tian; Sihong Xie
>
> **摘要:** Open-vocabulary 3D Scene Graph (3DSG) can enhance various downstream tasks in robotics by leveraging structured semantic representations, yet current 3DSG construction methods suffer from semantic inconsistencies caused by noisy cross-image aggregation under occlusions and constrained viewpoints. To mitigate the impact of such inconsistency, we propose RAG-3DSG, which introduces re-shot guided uncertainty estimation. By measuring the semantic consistency between original limited viewpoints and re-shot optimal viewpoints, this method quantifies the underlying semantic ambiguity of each graph object. Based on this quantification, we devise an Object-level Retrieval-Augmented Generation (RAG) that leverages low-uncertainty objects as semantic anchors to retrieve more reliable contextual knowledge, enabling a Vision-Language Model to rectify the predictions of uncertain objects and optimize the final 3DSG. Extensive evaluations across three challenging benchmarks and real-world robot trials demonstrate that RAG-3DSG achieves superior recall and precision, effectively mitigating semantic noise to provide highly reliable scene representations for robotics tasks.
>
---
#### [replaced 019] SERN: Bandwidth-Adaptive Cross-Reality Synchronization for Simulation-Enhanced Robot Navigation
- **分类: cs.RO; cs.MA**

- **简介: 该论文提出SERN框架，解决多机器人在受限环境下的跨现实同步问题，通过虚拟孪生与物理机器人协同提升导航性能。**

- **链接: [https://arxiv.org/pdf/2410.16686](https://arxiv.org/pdf/2410.16686)**

> **作者:** Jumman Hossain; Emon Dey; Snehalraj Chugh; Masud Ahmed; MS Anwar; Abu-Zaher Faridee; Jason Hoppes; Theron Trout; Anjon Basak; Rafidh Chowdhury; Rishabh Mistry; Hyun Kim; Jade Freeman; Niranjan Suri; Adrienne Raglin; Carl Busart; Anuradha Ravi; Nirmalya Roy
>
> **摘要:** Cross reality integration of simulation and physical robots is a promising approach for multi-robot operations in contested environments, where communication may be intermittent, interference may be present, and observability may be degraded. We present SERN (Simulation-Enhanced Realistic Navigation), a framework that tightly couples a high-fidelity virtual twin with physical robots to support real-time collaborative decision making. SERN makes three main contributions. First, it builds a virtual twin from geospatial and sensor data and continuously corrects it using live robot telemetry. Second, it introduces a physics-aware synchronization pipeline that combines predictive modeling with adaptive PD control. Third, it provides a bandwidth-adaptive ROS bridge that prioritizes critical topics when communication links are constrained. We also introduce a multi-metric cost function that balances latency, reliability, computation, and bandwidth. Theoretically, we show that when the adaptive controller keeps the physical and virtual input mismatch small, synchronization error remains bounded under moderate packet loss and latency. Empirically, SERN reduces end-to-end message latency by 15% to 25% and processing load by about 15% compared with a standard ROS setup, while maintaining tight real-virtual alignment with less than 5 cm positional error and less than 2 degrees rotational error. In a navigation task, SERN achieves a 95% success rate, compared with 85% for a real-only setup and 70% for a simulation-only setup, while also requiring fewer interventions and less time to reach the goal. These results show that a simulation-enhanced cross-reality stack can improve situational awareness and multi-agent coordination in contested environments by enabling look-ahead planning in the virtual twin while using real sensor feedback to correct discrepancies.
>
---
#### [replaced 020] Concurrent Prehensile and Nonprehensile Manipulation: A Practical Approach to Multi-Stage Dexterous Tasks
- **分类: cs.RO**

- **简介: 该论文研究多阶段灵巧操作任务，解决机器人在真实世界中高效学习复杂抓取与操作行为的问题。通过分解演示为对象中心技能，实现高成功率的并发操作。**

- **链接: [https://arxiv.org/pdf/2603.11655](https://arxiv.org/pdf/2603.11655)**

> **作者:** Hao Jiang; Yue Wu; Yue Wang; Gaurav S. Sukhatme; Daniel Seita
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Dexterous hands enable concurrent prehensile and nonprehensile manipulation, such as holding one object while interacting with another, a capability essential for everyday tasks yet underexplored in robotics. Learning such long-horizon, contact-rich multi-stage behaviors is challenging because demonstrations are expensive to collect and end-to-end policies require substantial data to generalize across varied object geometries and placements. We present DexMulti, a sample-efficient approach for real-world dexterous multi-task manipulation that decomposes demonstrations into object-centric skills with well-defined temporal boundaries. Rather than learning monolithic policies, our method retrieves demonstrated skills based on current object geometry, aligns them to the observed object state using an uncertainty-aware estimator that tracks centroid and yaw, and executes them via a retrieve-align-execute paradigm. We evaluate on three multi-stage tasks requiring concurrent manipulation (Grasp + Pull, Grasp + Open, and Grasp + Grasp) across two dexterous hands (Allegro and LEAP) in over 1,000 real-world trials. Our approach achieves an average success rate of 66% on training objects with only 3-4 demonstrations per object, outperforming diffusion policy baselines by 2-3x while requiring far fewer demonstrations. Results demonstrate robust generalization to held-out objects and spatial variations up to +/-25 cm.
>
---
#### [replaced 021] H2R: A Human-to-Robot Data Augmentation for Robot Pre-training from Videos
- **分类: cs.RO**

- **简介: 该论文提出H2R，用于解决机器人预训练中人类与机器人视觉差异问题，通过数据增强提升机器人学习效果。**

- **链接: [https://arxiv.org/pdf/2505.11920](https://arxiv.org/pdf/2505.11920)**

> **作者:** Guangrun Li; Yaoxu Lyu; Zhuoyang Liu; Chengkai Hou; Jieyu Zhang; Shanghang Zhang
>
> **摘要:** Large-scale pre-training using egocentric human videos has proven effective for robot learning. However, the models pre-trained on such data can be suboptimal for robot learning due to the significant visual gap between human hands and those of different robots. To remedy this, we propose H2R, a human-to-robot data augmentation pipeline that converts egocentric human videos into robot-centric visual data. H2R estimates human hand pose from videos, retargets the motion to simulated robotic arms, removes human limbs via segmentation and inpainting, and composites rendered robot embodiments into the original frames with camera-aligned geometry. This process explicitly bridges the visual gap between human and robot embodiments during pre-training. We apply H2R to augment large-scale egocentric human video datasets such as Ego4D and SSv2. To verify the effectiveness of the augmentation pipeline, we introduce a CLIP-based image-text similarity metric that quantitatively evaluates the semantic fidelity of robot-rendered frames to the original human actions. We evaluate H2R through comprehensive experiments in both simulation and real-world settings. In simulation, H2R consistently improves downstream success rates across four benchmark suites-Robomimic, RLBench, PushT, and CortexBench-yielding gains of 1.3%-10.2% across different visual encoders and policy learning methods. In real-world experiments, H2R improves performance on UR5 and dual-arm Franka/UR5 manipulation platforms, achieving 3.3%-23.3% success rate gains across gripper-based, dexterous, and bimanual tasks. We further demonstrate the potential of H2R in cross-embodiment generalization and its compatibility with vision-language-action models. These results indicate that H2R improves the generalization ability of robotic policies by mitigating the visual discrepancies between human and robot domains.
>
---
#### [replaced 022] Federated Multi-Agent Mapping for Planetary Exploration
- **分类: cs.RO; cs.LG; cs.MA**

- **简介: 该论文属于行星探测任务，解决多智能体在带宽受限环境下高效共享地图数据的问题。提出联邦多智能体映射方法，通过隐式神经映射减少数据传输并提升地图收敛速度。**

- **链接: [https://arxiv.org/pdf/2404.02289](https://arxiv.org/pdf/2404.02289)**

> **作者:** Tiberiu-Ioan Szatmari; Abhishek Cauligi
>
> **备注:** 7 pages, 6 figures
>
> **摘要:** Multi-agent robotic exploration stands to play an important role in space exploration as the next generation of robotic systems ventures to far-flung environments. A key challenge in this new paradigm will be to effectively share and utilize the vast amount of data generated onboard while operating in bandwidth-constrained regimes typical of space missions. Federated learning (FL) is a promising tool for bridging this gap. Drawing inspiration from the upcoming CADRE Lunar rover mission, we propose a federated multi-agent mapping approach that jointly trains a global map model across agents without transmitting raw data. Our method leverages implicit neural mapping to generate parsimonious, adaptable representations, reducing data transmission by up to 93.8% compared to raw maps. Furthermore, we enhance this approach with meta-initialization on Earth-based traversability datasets to significantly accelerate map convergence; reducing iterations required to reach target performance by 80% compared to random initialization. We demonstrate the efficacy of our approach on Martian terrains and glacier datasets, achieving downstream path planning F1 scores as high as 0.95 while outperforming on map reconstruction losses.
>
---
#### [replaced 023] CLAIM: Camera-LiDAR Alignment with Intensity and Monodepth
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于相机与激光雷达标定任务，旨在解决两者数据对齐问题。提出CLAIM方法，通过结构和纹理损失优化变换矩阵，实现高效精准对齐。**

- **链接: [https://arxiv.org/pdf/2512.14001](https://arxiv.org/pdf/2512.14001)**

> **作者:** Zhuo Zhang; Yonghui Liu; Meijie Zhang; Feiyang Tan; Yikang Ding
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** In this paper, we unleash the potential of the powerful monodepth model in camera-LiDAR calibration and propose CLAIM, a novel method of aligning data from the camera and LiDAR. Given the initial guess and pairs of images and LiDAR point clouds, CLAIM utilizes a coarse-to-fine searching method to find the optimal transformation minimizing a patched Pearson correlation-based structure loss and a mutual information-based texture loss. These two losses serve as good metrics for camera-LiDAR alignment results and require no complicated steps of data processing, feature extraction, or feature matching like most methods, rendering our method simple and adaptive to most scenes. We validate CLAIM on public KITTI, Waymo, and MIAS-LCEC datasets, and the experimental results demonstrate its superior performance compared with the state-of-the-art methods. The code is available at this https URL.
>
---
#### [replaced 024] Graphite: A GPU-Accelerated Mixed-Precision Graph Optimization Framework
- **分类: cs.RO**

- **简介: 该论文提出Graphite，一个基于GPU的混合精度图优化框架，用于解决SLAM中的非线性最小二乘问题，提升优化速度与内存效率。**

- **链接: [https://arxiv.org/pdf/2509.26581](https://arxiv.org/pdf/2509.26581)**

> **作者:** Shishir Gopinath; Karthik Dantu; Steven Y. Ko
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** We present Graphite, a GPU-accelerated nonlinear least squares graph optimization framework. It provides a CUDA C++ interface to enable the sharing of code between a real-time application, such as a SLAM system, and its optimization tasks. The framework supports techniques to reduce memory usage, including in-place optimization, support for multiple floating point types and mixed-precision modes, and dynamically computed Jacobians. We evaluate Graphite on well-known bundle adjustment problems and find that it achieves similar performance to MegBA, a solver specialized for bundle adjustment, while maintaining generality and using less memory. We also apply Graphite to global visual-inertial bundle adjustment on maps generated from stereo-inertial SLAM datasets, and observe speed-ups of up to 59x compared to a CPU baseline. Our results indicate that our framework enables faster large-scale optimization on both desktop and resource-constrained devices.
>
---
#### [replaced 025] VL-Nav: A Neuro-Symbolic Approach for Reasoning-based Vision-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，旨在解决机器人在复杂指令下导航困难的问题。提出VL-Nav系统，结合神经符号方法提升任务分解与探索效率。**

- **链接: [https://arxiv.org/pdf/2502.00931](https://arxiv.org/pdf/2502.00931)**

> **作者:** Yi Du; Taimeng Fu; Zhipeng Zhao; Shaoshu Su; Zitong Zhan; Qiwei Du; Zhuoqun Chen; Bowen Li; Chen Wang
>
> **摘要:** Navigating unseen, large-scale environments based on complex and abstract human instructions remains a formidable challenge for autonomous mobile robots. Addressing this requires robots to infer implicit semantics and efficiently explore large-scale task spaces. However, existing methods, ranging from end-to-end learning to foundation model-based modular architectures, often lack the capability to decompose complex tasks or employ efficient exploration strategies, leading to robot aimless wandering or target recognition failures. To address these limitations, we propose VL-Nav, a neuro-symbolic (NeSy) vision-language navigation system. The proposed system intertwines neural reasoning with symbolic guidance through two core components: (1) a NeSy task planner that leverages a symbolic 3D scene graph and image memory system to enhance the vision language models' (VLMs) neural reasoning capabilities for task decomposition and replanning; and (2) a NeSy exploration system that couples neural semantic cues with the symbolic heuristic function to efficiently gather the task-related information while minimizing unnecessary repeat travel during exploration. Validated on the DARPA TIAMAT Challenge navigation tasks, our system achieved an 83.4% success rate (SR) in indoor environments and 75% in outdoor scenarios. VL-Nav achieved an 86.3% SR in real-world experiments, including a challenging 483-meter run. Finally, we validate the system with complex instructions in a 3D multi-floor scenario.
>
---
#### [replaced 026] MARVL: Multi-Stage Guidance for Robotic Manipulation via Vision-Language Models
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人操控任务，旨在解决稀疏奖励下策略学习效率低的问题。通过多阶段引导和视觉语言模型，提升奖励设计的准确性与任务执行效果。**

- **链接: [https://arxiv.org/pdf/2602.15872](https://arxiv.org/pdf/2602.15872)**

> **作者:** Xunlan Zhou; Xuanlin Chen; Shaowei Zhang; Xiangkun Li; ShengHua Wan; Xiaohai Hu; Lei Yuan; Le Gan; De-chuan Zhan
>
> **摘要:** Designing dense reward functions is pivotal for efficient robotic Reinforcement Learning (RL). However, most dense rewards rely on manual engineering, which fundamentally limits the scalability and automation of reinforcement learning. While Vision-Language Models (VLMs) offer a promising path to reward design, naive VLM rewards often misalign with task progress, struggle with spatial grounding, and show limited understanding of task semantics. To address these issues, we propose MARVL-Multi-stAge guidance for Robotic manipulation via Vision-Language models. MARVL fine-tunes a VLM for spatial and semantic consistency and decomposes tasks into multi-stage subtasks with task direction projection for trajectory sensitivity. Empirically, MARVL significantly outperforms existing VLM-reward methods on the Meta-World benchmark, demonstrating superior sample efficiency and robustness on sparse-reward manipulation tasks.
>
---
#### [replaced 027] RoboMD: Uncovering Robot Vulnerabilities through Semantic Potential Fields
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人安全任务，旨在解决现实环境中机器人操作策略的脆弱性问题。通过虚拟环境学习潜在场，识别并分析漏洞，提升机器人操作安全性与性能。**

- **链接: [https://arxiv.org/pdf/2412.02818](https://arxiv.org/pdf/2412.02818)**

> **作者:** Som Sagar; Jiafei Duan; Sreevishakh Vasudevan; Yifan Zhou; Heni Ben Amor; Dieter Fox; Ransalu Senanayake
>
> **备注:** 26 Pages, 20 figures
>
> **摘要:** Robot manipulation policies, while central to the promise of physical AI, are highly vulnerable in the presence of external variations in the real world. Diagnosing these vulnerabilities is hindered by two key challenges: (i) the relevant variations to test against are often unknown, and (ii) direct testing in the real world is costly and unsafe. We introduce a framework that tackles both issues by learning a separate deep reinforcement learning (deep RL) policy for vulnerability prediction through virtual runs on a continuous vision-language embedding trained with limited success-failure data. By treating this embedding space, which is rich in semantic and visual variations, as a potential field, the policy learns to move toward vulnerable regions while being repelled from success regions. This vulnerability prediction policy, trained on virtual rollouts, enables scalable and safe vulnerability analysis without expensive physical trials. By querying this policy, our framework builds a probabilistic vulnerability-likelihood map. Experiments across simulation benchmarks and a physical robot arm show that our framework uncovers up to 23% more unique vulnerabilities than state-of-the-art vision-language baselines, revealing subtle vulnerabilities overlooked by heuristic testing. Additionally, we show that fine-tuning the manipulation policy with the vulnerabilities discovered by our framework improves manipulation performance with much less fine-tuning data.
>
---
#### [replaced 028] ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning
- **分类: cs.AI; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出ExoPredicator框架，用于机器人长时程规划。任务是建模动态世界中的因果过程，解决环境变化与代理行为同步的问题。通过学习符号状态和因果过程，提升规划效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.26255](https://arxiv.org/pdf/2509.26255)**

> **作者:** Yichao Liang; Dat Nguyen; Cambridge Yang; Tianyang Li; Joshua B. Tenenbaum; Carl Edward Rasmussen; Adrian Weller; Zenna Tavares; Tom Silver; Kevin Ellis
>
> **备注:** ICLR 2026. The last two authors contributed equally in co-advising
>
> **摘要:** Long-horizon embodied planning is challenging because the world does not only change through an agent's actions: exogenous processes (e.g., water heating, dominoes cascading) unfold concurrently with the agent's actions. We propose a framework for abstract world models that jointly learns (i) symbolic state representations and (ii) causal processes for both endogenous actions and exogenous mechanisms. Each causal process models the time course of a stochastic cause-effect relation. We learn these world models from limited data via variational Bayesian inference combined with LLM proposals. Across five simulated tabletop robotics environments, the learned models enable fast planning that generalizes to held-out tasks with more objects and more complex goals, outperforming a range of baselines.
>
---
#### [replaced 029] A Modular Architecture Design for Autonomous Driving Racing in Controlled Environments
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自动驾驶任务，解决赛车在封闭环境中的自主驾驶问题。设计了模块化架构，包含感知、定位、路径规划和控制模块，提升系统可靠性与性能。**

- **链接: [https://arxiv.org/pdf/2512.03886](https://arxiv.org/pdf/2512.03886)**

> **作者:** Brais Fontan-Costas; M. Diaz-Cacho; Ruben Fernandez-Boullon; Manuel Alonso-Carracedo; Javier Perez-Robles
>
> **摘要:** This paper presents a modular autonomous driving architecture for Formula Student Driverless competition vehicles operating in closed-circuit environments. The perception module employs YOLOv11 for real-time traffic cone detection, achieving 0.93 mAP@0.5 on the FSOCO dataset, combined with neural stereo depth estimation from a ZED 2i camera for 3D cone localization with sub-0.5 m median error at distances up to 7 m. State estimation fuses RTK-GNSS positioning and IMU measurements through an Extended Kalman Filter (EKF) based on a kinematic bicycle model, achieving centimeter-level localization accuracy with a 12 cm improvement over raw GNSS. Path planning computes the racing line via cubic spline interpolation on ordered track boundaries and assigns speed profiles constrained by curvature and vehicle dynamics. A regulated pure pursuit controller tracks the planned trajectory with a dynamic lookahead parameterized by speed error. The complete pipeline is implemented as a modular ROS 2 architecture on an NVIDIA Jetson Orin NX platform, with each subsystem deployed as independent nodes communicating through a dual-computer configuration. Experimental validation combines real-world sensor evaluation with simulation-based end-to-end testing, where realistic sensor error distributions are injected to assess system-level performance under representative conditions.
>
---
#### [replaced 030] GM3: A General Physical Model for Micro-Mobility Vehicles
- **分类: cs.RO**

- **简介: 该论文属于车辆动力学建模任务，旨在解决MMV动态模拟中缺乏统一物理模型的问题。提出GM3模型，支持多种轮构型，并进行仿真验证。**

- **链接: [https://arxiv.org/pdf/2510.07807](https://arxiv.org/pdf/2510.07807)**

> **作者:** Grace Cai; Nithin Parepally; Laura Zheng; Ming C. Lin
>
> **摘要:** Modeling the dynamics of micro-mobility vehicles (MMV) is becoming increasingly important for training autonomous vehicle systems and building urban traffic simulations. However, mainstream tools rely on variants of the Kinematic Bicycle Model (KBM) or mode-specific physics that miss tire slip, load transfer, and rider/vehicle lean. To our knowledge, no unified, physics-based model captures these dynamics across the full range of common MMVs and wheel layouts. We propose the "Generalized Micro-mobility Model" (GM3), a tire-level formulation based on the tire brush representation that supports arbitrary wheel configurations, including single/double track and multi-wheel platforms. We introduce an interactive model-agnostic simulation framework that decouples vehicle/layout specification from dynamics to compare the GM3 with the KBM and other models, consisting of fixed step RK4 integration, human-in-the-loop and scripted control, real-time trajectory traces and logging for analysis. We also empirically validate the GM3 on the Stanford Drone Dataset's deathCircle (roundabout) scene for biker, skater, and cart classes.
>
---
#### [replaced 031] Multi-Robot Navigation in Social Mini-Games: Definitions, Taxonomy, and Algorithms
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人导航任务，旨在解决社交迷你游戏环境中的导航问题。提出统一分类和评估标准，梳理现有方法，明确其与传统导航系统的差异。**

- **链接: [https://arxiv.org/pdf/2508.13459](https://arxiv.org/pdf/2508.13459)**

> **作者:** Rohan Chandra; Shubham Singh; Wenhao Luo; Katia Sycara
>
> **备注:** Accepted for publication in Autonomous Robots 2026
>
> **摘要:** The "Last Mile Challenge" has long been considered an important, yet unsolved, challenge for autonomous vehicles, public service robots, and delivery robots. A central issue in this challenge is the ability of robots to navigate constrained and cluttered environments that have high agency (e.g., doorways, hallways, corridor intersections), often while competing for space with other robots and humans. We refer to these environments as "Social Mini-Games" (SMGs). Traditional navigation approaches designed for MRN do not perform well in SMGs, which has led to focused research on dedicated SMG solvers. However, publications on SMG navigation research make different assumptions, and have different objective functions (safety versus liveness). These assumptions and objectives are sometimes implicitly assumed or described informally. This makes it difficult to establish appropriate baselines for comparison in research papers, as well as making it difficult for practitioners to find the papers relevant to their concrete application. Such ad-hoc representation of the field also presents a barrier to new researchers wanting to start research in this area. SMG navigation research requires its own taxonomy, definitions, and evaluation protocols to guide effective research moving forward. This survey is the first to catalog SMG solvers using a well-defined and unified taxonomy and to classify existing methods accordingly. It also discusses the essential properties of SMG solvers, defines what SMGs are and how they appear in practice, outlines how to evaluate SMG solvers, and highlights the differences between SMG solvers and general navigation systems. The survey concludes with an overview of future directions and open challenges in the field. Our project is open-sourced at this https URL{this https URL.
>
---
#### [replaced 032] Beyond Frame-wise Tracking: A Trajectory-based Paradigm for Efficient Point Cloud Tracking
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于3D单目标跟踪任务，解决传统方法在计算成本与长期时序上下文之间的矛盾。提出TrajTrack框架，通过轨迹建模提升跟踪精度与效率。**

- **链接: [https://arxiv.org/pdf/2509.11453](https://arxiv.org/pdf/2509.11453)**

> **作者:** BaiChen Fan; Yuanxi Cui; Jian Li; Qin Wang; Shibo Zhao; Muqing Cao; Sifan Zhou
>
> **备注:** Acceptted in ICRA 2026
>
> **摘要:** LiDAR-based 3D single object tracking (3D SOT) is a critical task in robotics and autonomous systems. Existing methods typically follow frame-wise motion estimation or a sequence-based paradigm. However, the two-frame methods are efficient but lack long-term temporal context, making them vulnerable in sparse or occluded scenes, while sequence-based methods that process multiple point clouds gain robustness at a significant computational cost. To resolve this dilemma, we propose a novel trajectory-based paradigm and its instantiation, TrajTrack. TrajTrack is a lightweight framework that enhances a base two-frame tracker by implicitly learning motion continuity from historical bounding box trajectories alone-without requiring additional, costly point cloud inputs. It first generates a fast, explicit motion proposal and then uses an implicit motion modeling module to predict the future trajectory, which in turn refines and corrects the initial proposal. Extensive experiments on the large-scale NuScenes benchmark show that TrajTrack achieves new state-of-the-art performance, dramatically improving tracking precision by 3.02% over a strong baseline while running at 55 FPS. Besides, we also demonstrate the strong generalizability of TrajTrack across different base trackers. Code is available at this https URL.
>
---
#### [replaced 033] DyQ-VLA: Temporal-Dynamic-Aware Quantization for Embodied Vision-Language-Action Models
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型优化任务，解决静态量化在动态场景下的资源浪费与实时性不足问题，提出DyQ-VLA动态量化框架提升效率。**

- **链接: [https://arxiv.org/pdf/2603.07904](https://arxiv.org/pdf/2603.07904)**

> **作者:** Zihao Zheng; Hangyu Cao; Sicheng Tian; Jiayu Chen; Maoliang Li; Xinhao Sun; Hailong Zou; Zhaobo Zhang; Xuanzhe Liu; Donggang Cao; Hong Mei; Xiang Chen
>
> **摘要:** Vision-Language-Action (VLA) models are dominant in embodied intelligence but are constrained by inference overheads. While model quantization alleviates these bottlenecks for edge deployment, static quantization approaches remain suboptimal for VLAs due to two critical challenges: (1) Temporal-dynamic sensitivity, where fixed precision wastes resources by ignoring stage-varying error tolerances; and (2) Real-time allocation, where identifying real-time sensitivity to guide bit allocation remains unsolved. To address these challenges, we propose DyQ-VLA, a dynamic quantization framework for VLAs. Specifically, a sensitivity-aware switching strategy leverages real-time kinematic proxies to trigger the bit-width switch, while a kinematic-guided module dynamically allocates the optimal bit-width. Experiments show that DyQ-VLA requires only 30.9% of the original memory footprint while maintaining 99.5% of its original performance, achieving 1.49x simulation and up to 1.43x real-world speedups.
>
---
#### [replaced 034] GoalSwarm: Multi-UAV Semantic Coordination for Open-Vocabulary Object Navigation
- **分类: cs.RO**

- **简介: 该论文提出GoalSwarm，解决多无人机在未知环境中的开放词汇目标导航问题。通过轻量级语义地图和去中心化协作，实现高效目标识别与探索。**

- **链接: [https://arxiv.org/pdf/2603.12908](https://arxiv.org/pdf/2603.12908)**

> **作者:** MoniJesu Wonders James; Amir Atef Habel; Aleksey Fedoseev; Dzmitry Tsetserokou
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** Cooperative visual semantic navigation is a foundational capability for aerial robot teams operating in unknown environments. However, achieving robust open-vocabulary object-goal navigation remains challenging due to the computational constraints of deploying heavy perception models onboard and the complexity of decentralized multi-agent coordination. We present GoalSwarm, a fully decentralized multi-UAV framework for zero-shot semantic object-goal navigation. Each UAV collaboratively constructs a shared, lightweight 2D top-down semantic occupancy map by projecting depth observations from aerial vantage points, eliminating the computational burden of full 3D representations while preserving essential geometric and semantic structure. The core contributions of GoalSwarm are threefold: (1) integration of zero-shot foundation model -- SAM3 for open vocabulary detection and pixel-level segmentation, enabling open-vocabulary target identification without task-specific training; (2) a Bayesian Value Map that fuses multi-viewpoint detection confidences into a per-pixel goal-relevance distribution, enabling informed frontier scoring via Upper Confidence Bound (UCB) exploration; and (3) a decentralized coordination strategy combining semantic frontier extraction, cost-utility bidding with geodesic path costs, and spatial separation penalties to minimize redundant exploration across the swarm.
>
---
#### [replaced 035] TurboMap: GPU-Accelerated Local Mapping for Visual SLAM
- **分类: cs.RO**

- **简介: 该论文属于视觉SLAM任务，旨在解决实时局部建图的延迟问题。通过GPU加速和优化，提升局部映射效率，同时保持精度。**

- **链接: [https://arxiv.org/pdf/2511.02036](https://arxiv.org/pdf/2511.02036)**

> **作者:** Parsa Hosseininejad; Kimia Khabiri; Shishir Gopinath; Soudabeh Mohammadhashemi; Karthik Dantu; Steven Y. Ko
>
> **备注:** Submitted to IROS 2026
>
> **摘要:** In real-time Visual SLAM systems, local mapping must operate under strict latency constraints, as delays degrade map quality and increase the risk of tracking failure. GPU parallelization offers a promising way to reduce latency. However, parallelizing local mapping is challenging due to synchronized shared-state updates and the overhead of transferring large map data structures to the GPU. This paper presents TurboMap, a GPU-parallelized and CPU-optimized local mapping backend that holistically addresses these challenges. We restructure Map Point Creation to enable parallel Keypoint Correspondence Search on the GPU, redesign and parallelize Map Point Fusion, optimize Redundant Keyframe Culling on the CPU, and integrate a fast GPU-based Local Bundle Adjustment solver. To minimize data transfer and synchronization costs, we introduce persistent GPU-resident keyframe storage. Experiments on the EuRoC and TUM-VI datasets show average local mapping speedups of 1.3x and 1.6x, respectively, while preserving accuracy.
>
---
#### [replaced 036] Optimization-Based Robust Permissive Synthesis for Interval MDPs
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人决策任务，解决区间MDP下的鲁棒宽松合成问题。通过构建MILP模型，在不确定性下最大化策略灵活性，确保满足概率可达性要求。**

- **链接: [https://arxiv.org/pdf/2510.03481](https://arxiv.org/pdf/2510.03481)**

> **作者:** Khang Vo Huynh; David Parker; Lu Feng
>
> **摘要:** We present an optimization-based framework for robust permissive synthesis for Interval Markov Decision Processes (IMDPs), motivated by robotic decision-making under transition uncertainty. In many robotic systems, model inaccuracies and sensing noise lead to interval-valued transition probabilities. While robust IMDP synthesis typically yields a single policy and permissive synthesis assumes exact models, we show that robust permissive synthesis under interval uncertainty can be cast as a global mixed-integer linear program (MILP) that directly encodes robust Bellman constraints. The formulation maximizes a quantitative permissiveness metric (the number of enabled state-action pairs), while guaranteeing that every compliant strategy satisfies probabilistic reachability or expected reward specifications under all admissible transition realizations. To address the exponential complexity of vertex-based uncertainty representations, we derive a dualization-based encoding that eliminates explicit vertex enumeration and scales linearly with the number of successors. Experimental evaluation on four representative robotic benchmark domains demonstrates scalability to IMDPs with hundreds of thousands of states. The proposed framework provides a practical and general foundation for uncertainty-aware, flexibility-preserving controller synthesis in robotic systems.
>
---
#### [replaced 037] $χ_{0}$: Resource-Aware Robust Manipulation via Taming Distributional Inconsistencies
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人长期操作任务，解决分布不一致导致的误差累积问题。提出$\chi_{0}$框架，通过模型算术、阶段优势和训练部署对齐提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.09021](https://arxiv.org/pdf/2602.09021)**

> **作者:** Checheng Yu; Chonghao Sima; Gangcheng Jiang; Hai Zhang; Haoguang Mai; Hongyang Li; Huijie Wang; Jin Chen; Kaiyang Wu; Li Chen; Lirui Zhao; Modi Shi; Ping Luo; Qingwen Bu; Shijia Peng; Tianyu Li; Yibo Yuan
>
> **摘要:** High-reliability long-horizon robotic manipulation has traditionally relied on large-scale data and compute to understand complex real-world dynamics. However, we identify that the primary bottleneck to real-world robustness is not resource scale alone, but the distributional shift among the human demonstration distribution, the inductive bias learned by the policy, and the test-time execution distribution -- a systematic inconsistency that causes compounding errors in multi-stage tasks. To mitigate these inconsistencies, we propose $\chi_{0}$, a resource-efficient framework with effective modules designated to achieve production-level robustness in robotic manipulation. Our approach builds off three technical pillars: (i) Model Arithmetic, a weight-space merging strategy that efficiently soaks up diverse distributions of different demonstrations, varying from object appearance to state variations; (ii) Stage Advantage, a stage-aware advantage estimator that provides stable, dense progress signals, overcoming the numerical instability of prior non-stage approaches; and (iii) Train-Deploy Alignment, which bridges the distribution gap via spatio-temporal augmentation, heuristic DAgger corrections, and temporal chunk-wise smoothing. $\chi_{0}$ enables two sets of dual-arm robots to collaboratively orchestrate long-horizon garment manipulation, spanning tasks from flattening, folding, to hanging different clothes. Our method exhibits high-reliability autonomy; we are able to run the system from arbitrary initial state for consecutive 24 hours non-stop. Experiments validate that $\chi_{0}$ surpasses the state-of-the-art $\pi_{0.5}$ in success rate by nearly 250%, with only 20-hour data and 8 A100 GPUs. Code, data and models will be released to facilitate the community.
>
---
#### [replaced 038] Open-World Motion Forecasting
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于运动预测任务，解决真实场景中对象类别动态变化带来的挑战。提出一种端到端的增量学习框架，有效防止遗忘并适应新类别。**

- **链接: [https://arxiv.org/pdf/2603.09420](https://arxiv.org/pdf/2603.09420)**

> **作者:** Nicolas Schischka; Nikhil Gosala; B Ravi Kiran; Senthil Yogamani; Abhinav Valada
>
> **备注:** V2: Adapt author affiliation
>
> **摘要:** Motion forecasting aims to predict the future trajectories of dynamic agents in the scene, enabling autonomous vehicles to effectively reason about scene evolution. Existing approaches operate under the closed-world regime and assume fixed object taxonomy as well as access to high-quality perception. Therefore, they struggle in real-world settings where perception is imperfect and object taxonomy evolves over time. In this work, we bridge this fundamental gap by introducing open-world motion forecasting, a novel setting in which new object classes are sequentially introduced over time and future object trajectories are estimated directly from camera images. We tackle this setting by proposing the first end-to-end class-incremental motion forecasting framework to mitigate catastrophic forgetting while simultaneously learning to forecast newly introduced classes. When a new class is introduced, our framework employs a pseudo-labeling strategy to first generate motion forecasting pseudo-labels for all known classes which are then processed by a vision-language model to filter inconsistent and over-confident predictions. Parallelly, our approach further mitigates catastrophic forgetting by using a novel replay sampling strategy that leverages query feature variance to sample previous sequences with informative motion patterns. Extensive evaluation on the nuScenes and Argoverse 2 datasets demonstrates that our approach successfully resists catastrophic forgetting and maintains performance on previously learned classes while improving adaptation to novel ones. Further, we demonstrate that our approach supports zero-shot transfer to real-world driving and naturally extends to end-to-end class-incremental planning, enabling continual adaptation of the full autonomous driving system. We provide the code at this https URL.
>
---
#### [replaced 039] A Deconfounding Framework for Human Behavior Prediction: Enhancing Robotic Systems in Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文属于人类行为预测任务，旨在解决动态环境中因混杂因素导致的预测偏差问题。通过融合去混淆技术与时间序列方法，提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2410.20423](https://arxiv.org/pdf/2410.20423)**

> **作者:** Wentao Gao; Cheng Zhou
>
> **备注:** 7 pages, Under review
>
> **摘要:** Accurate prediction of human behavior is crucial for effective human-robot interaction (HRI) systems, especially in dynamic environments where real-time decisions are essential. This paper addresses the challenge of forecasting future human behavior using multivariate time series data from wearable sensors, which capture various aspects of human movement. The presence of hidden confounding factors in this data often leads to biased predictions, limiting the reliability of traditional models. To overcome this, we propose a robust predictive model that integrates deconfounding techniques with advanced time series prediction methods, enhancing the model's ability to isolate true causal relationships and improve prediction accuracy. Evaluation on real-world datasets demonstrates that our approach significantly outperforms traditional methods, providing a more reliable foundation for responsive and adaptive HRI systems.
>
---
#### [replaced 040] HoRD: Robust Humanoid Control via History-Conditioned Reinforcement Learning and Online Distillation
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出HoRD框架，解决人形机器人在动态变化环境中的鲁棒控制问题。通过历史条件强化学习和在线蒸馏，提升模型的适应能力与迁移性能。**

- **链接: [https://arxiv.org/pdf/2602.04412](https://arxiv.org/pdf/2602.04412)**

> **作者:** Puyue Wang; Jiawei Hu; Yan Gao; Junyan Wang; Yu Zhang; Gillian Dobbie; Tao Gu; Wafa Johal; Ting Dang; Hong Jia
>
> **摘要:** Humanoid robots can suffer significant performance drops under small changes in dynamics, task specifications, or environment setup. We propose HoRD, a two-stage learning framework for robust humanoid control under domain shift. First, we train a high-performance teacher policy via history-conditioned reinforcement learning, where the policy infers latent dynamics context from recent state--action trajectories to adapt online to diverse randomized dynamics. Second, we perform online distillation to transfer the teacher's robust control capabilities into a transformer-based student policy that operates on sparse root-relative 3D joint keypoint trajectories. By combining history-conditioned adaptation with online distillation, HoRD enables a single policy to adapt zero-shot to unseen domains without per-domain retraining. Extensive experiments show HoRD outperforms strong baselines in robustness and transfer, especially under unseen domains and external perturbations. Code and project page are available at this https URL.
>
---
#### [replaced 041] UniPrototype: Humn-Robot Skill Learning with Uniform Prototypes
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，旨在解决数据稀缺问题。通过共享运动基元，将人类技能知识迁移至机器人，提升学习效率和任务表现。**

- **链接: [https://arxiv.org/pdf/2509.23021](https://arxiv.org/pdf/2509.23021)**

> **作者:** Xiao Hu; Qi Yin; Yangming Shi; Yang Ye
>
> **备注:** This submission was uploaded in error and has been withdrawn. A substantial revision will need to be completed
>
> **摘要:** Data scarcity remains a fundamental challenge in robot learning. While human demonstrations benefit from abundant motion capture data and vast internet resources, robotic manipulation suffers from limited training examples. To bridge this gap between human and robot manipulation capabilities, we propose UniPrototype, a novel framework that enables effective knowledge transfer from human to robot domains via shared motion primitives. ur approach makes three key contributions: (1) We introduce a compositional prototype discovery mechanism with soft assignments, enabling multiple primitives to co-activate and thus capture blended and hierarchical skills; (2) We propose an adaptive prototype selection strategy that automatically adjusts the number of prototypes to match task complexity, ensuring scalable and efficient representation; (3) We demonstrate the effectiveness of our method through extensive experiments in both simulation environments and real-world robotic systems. Our results show that UniPrototype successfully transfers human manipulation knowledge to robots, significantly improving learning efficiency and task performance compared to existing this http URL code and dataset will be released upon acceptance at an anonymous repository.
>
---
#### [replaced 042] sim2art: Accurate Articulated Object Modeling from a Single Video using Synthetic Training Data Only
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于单视角视频中刚体物体建模任务，解决真实场景下复杂关节物体3D重建问题。通过合成数据训练，实现无需真实标注的高精度分割与关节参数估计。**

- **链接: [https://arxiv.org/pdf/2512.07698](https://arxiv.org/pdf/2512.07698)**

> **作者:** Arslan Artykov; Tom Ravaud; Corentin Sautier; Vincent Lepetit
>
> **摘要:** Understanding articulated objects from monocular video is a crucial yet challenging task in robotics and digital twin creation. Existing methods often rely on complex multi-view setups, high-fidelity object scans, or fragile long-term point tracks that frequently fail in casual real-world captures. In this paper, we present sim2art, a data-driven framework that recovers the 3D part segmentation and joint parameters of articulated objects from a single monocular video captured by a freely moving camera. Our core insight is a robust representation based on per-frame surface point sampling, which we augment with short-term scene flow and DINOv3 semantic features. Unlike previous works that depend on error-prone long-term correspondences, our representation is easy to obtain and exhibits a negligible difference between simulation and reality without requiring domain adaptation. Also, by construction, our method relies on single-viewpoint visibility, ensuring that the geometric representation remains consistent across synthetic and real data despite noise and occlusions. Leveraging a suitable Transformer-based architecture, sim2art is trained exclusively on synthetic data yet generalizes strongly to real-world sequences. To address the lack of standardized benchmarks in the field, we introduce two datasets featuring a significantly higher diversity of object categories and instances than prior work. Our evaluations show that sim2art effectively handles large camera motions and complex articulations, outperforming state-of-the-art optimization-based and tracking-dependent methods. sim2art offers a scalable solution that can be easily extended to new object categories without the need for cumbersome real-world annotations. Project webpage: this https URL
>
---
#### [replaced 043] Taxonomy and Trends in Reinforcement Learning for Robotics and Control Systems: A Structured Review
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于强化学习在机器人与控制系统中的应用研究，旨在梳理RL技术发展与应用趋势，解决智能机器人在复杂环境中的决策与控制问题。**

- **链接: [https://arxiv.org/pdf/2510.21758](https://arxiv.org/pdf/2510.21758)**

> **作者:** Kumater Ter; Abolanle Adetifa; Daniel Udekwe
>
> **摘要:** Reinforcement learning (RL) has become a foundational approach for enabling intelligent robotic behavior in dynamic and uncertain environments. This work presents an in-depth review of RL principles, advanced deep reinforcement learning (DRL) algorithms, and their integration into robotic and control systems. Beginning with the formalism of Markov Decision Processes (MDPs), the study outlines essential elements of the agent-environment interaction and explores core algorithmic strategies including actor-critic methods, value-based learning, and policy gradients. Emphasis is placed on modern DRL techniques such as DDPG, TD3, PPO, and SAC, which have shown promise in solving high-dimensional, continuous control tasks. A structured taxonomy is introduced to categorize RL applications across domains such as locomotion, manipulation, multi-agent coordination, and human-robot interaction, along with training methodologies and deployment readiness levels. The review synthesizes recent research efforts, highlighting technical trends, design patterns, and the growing maturity of RL in real-world robotics. Overall, this work aims to bridge theoretical advances with practical implementations, providing a consolidated perspective on the evolving role of RL in autonomous robotic systems.
>
---
#### [replaced 044] A Human-in-the-Loop Confidence-Aware Failure Recovery Framework for Modular Robot Policies
- **分类: cs.RO**

- **简介: 该论文属于机器人故障恢复任务，解决模块化机器人在不确定环境中的失败恢复问题。通过结合模块不确定性与人类干预成本，优化人机协作的故障恢复策略。**

- **链接: [https://arxiv.org/pdf/2602.10289](https://arxiv.org/pdf/2602.10289)**

> **作者:** Rohan Banerjee; Krishna Palempalli; Bohan Yang; Jiaying Fang; Alif Abdullah; Tom Silver; Sarah Dean; Tapomayukh Bhattacharjee
>
> **备注:** The second and third authors contributed equally. The last two authors advised equally
>
> **摘要:** Robots operating in unstructured human environments inevitably encounter failures, especially in robot caregiving scenarios. While humans can often help robots recover, excessive or poorly targeted queries impose unnecessary cognitive and physical workload on the human partner. We present a human-in-the-loop failure-recovery framework for modular robotic policies, where a policy is composed of distinct modules such as perception, planning, and control, any of which may fail and often require different forms of human feedback. Our framework integrates calibrated estimates of module-level uncertainty with models of human intervention cost to decide which module to query and when to query the human. It separates these two decisions: a module selector identifies the module most likely responsible for failure, and a querying algorithm determines whether to solicit human input or act autonomously. We evaluate several module-selection strategies and querying algorithms in controlled synthetic experiments, revealing trade-offs between recovery efficiency, robustness to system and user variables, and user workload. Finally, we deploy the framework on a robot-assisted bite acquisition system and demonstrate, in studies involving individuals with both emulated and real mobility limitations, that it improves recovery success while reducing the workload imposed on users. Our results highlight how explicitly reasoning about both robot uncertainty and human effort can enable more efficient and user-centered failure recovery in collaborative robots. Supplementary materials and videos can be found at: this http URL
>
---
#### [replaced 045] On transferring safety certificates across dynamical systems
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于控制安全领域，解决不同动力系统间安全证书迁移问题。提出tCBF框架，通过模拟函数和边界项实现安全约束的跨系统传递。**

- **链接: [https://arxiv.org/pdf/2602.03987](https://arxiv.org/pdf/2602.03987)**

> **作者:** Nikolaos Bousias; Charalampia Stamouli; Anastasios Tsiamis; George Pappas
>
> **摘要:** Control barrier functions (CBFs) provide a powerful tool for enforcing safety constraints in control systems, but their direct application to complex, high-dimensional dynamics is often challenging. In many settings, safety certificates are more naturally designed for simplified or alternative system models that do not exactly match the dynamics of interest. This paper addresses the problem of transferring safety guarantees between dynamical systems with mismatched dynamics. We propose a transferred control barrier function (tCBF) framework that enables safety constraints defined on one system to be systematically enforced on another system using a simulation function and an explicit margin term. The resulting transferred barrier accounts for model mismatch and induces a safety condition that can be enforced on the target system via a quadratic-program-based safety filter. The proposed approach is general and does not require the two systems to share the same state dimension or dynamics. We demonstrate the effectiveness of the framework on a quadrotor navigation task with the transferred barrier ensuring collision avoidance for the target system, while remaining minimally invasive to a nominal controller. These results highlight the potential of transferred control barrier functions as a general mechanism for enforcing safety across heterogeneous dynamical systems.
>
---
#### [replaced 046] From Fold to Function: Simulation-Driven Design of Origami Mechanisms
- **分类: cs.RO**

- **简介: 该论文属于机械设计任务，旨在解决origami机制模拟与优化问题。通过构建基于MuJoCo的仿真框架，实现origami结构的物理模拟与性能优化。**

- **链接: [https://arxiv.org/pdf/2511.10580](https://arxiv.org/pdf/2511.10580)**

> **作者:** Tianhui Han; Shashwat Singh; Sarvesh Patil; Zeynep Temel
>
> **备注:** 8 Pages, 9 Figures, Submitted to IEEE RoboSoft
>
> **摘要:** Origami-inspired mechanisms can transform flat sheets into functional three-dimensional dynamic structures that are lightweight, compact, and capable of complex motion. These properties make origami increasingly valuable in robotic and deployable systems. However, accurately simulating their folding behavior and interactions with the environment remains challenging. To address this, we present a design framework for origami mechanism simulation that utilizes MuJoCo's deformable-body capabilities. In our approach, origami sheets are represented as graphs of interconnected deformable elements with user-specified constraints such as creases and actuation, defined through an intuitive graphical user interface (GUI). This framework allows users to generate physically consistent simulations that capture both the geometric structure of origami mechanisms and their interactions with external objects and surfaces. We demonstrate our method's utility through a case study on an origami catapult, where design parameters are optimized in simulation using the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) and validated experimentally on physical prototypes. The optimized structure achieves improved throwing performance, illustrating how our system enables rapid, simulation-driven origami design, optimization, and analysis.
>
---
#### [replaced 047] RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RoboClaw框架，解决长周期机器人任务中的数据收集与策略执行问题。通过统一控制流程，提升任务稳定性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2603.11558](https://arxiv.org/pdf/2603.11558)**

> **作者:** Ruiying Li; Yunlang Zhou; YuYao Zhu; Kylin Chen; Jingyuan Wang; Sukai Wang; Kongtao Hu; Minhui Yu; Bowen Jiang; Zhan Su; Jiayao Ma; Xin He; Yongjian Shen; Yang Yang; Guanghui Ren; Maoqing Yao; Wenhao Wang; Yao Mu
>
> **摘要:** Vision-Language-Action (VLA) systems have shown strong potential for language-driven robotic manipulation. However, scaling them to long-horizon tasks remains challenging. Existing pipelines typically separate data collection, policy learning, and deployment, resulting in heavy reliance on manual environment resets and brittle multi-policy execution. We present RoboClaw, an agentic robotics framework that unifies data collection, policy learning, and task execution under a single VLM-driven controller. At the policy level, RoboClaw introduces Entangled Action Pairs (EAP), which couple forward manipulation behaviors with inverse recovery actions to form self-resetting loops for autonomous data collection. This mechanism enables continuous on-policy data acquisition and iterative policy refinement with minimal human intervention. During deployment, the same agent performs high-level reasoning and dynamically orchestrates learned policy primitives to accomplish long-horizon tasks. By maintaining consistent contextual semantics across collection and execution, RoboClaw reduces mismatch between the two phases and improves multi-policy robustness. Experiments in real-world manipulation tasks demonstrate improved stability and scalability compared to conventional open-loop pipelines, while significantly reducing human effort throughout the robot lifecycle, achieving a 25% improvement in success rate over baseline methods on long-horizon tasks and reducing human time investment by 53.7%.
>
---
#### [replaced 048] Pose Estimation of a Thruster-Driven Bioinspired Multi-Link Robot
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人姿态估计任务，解决自由漂浮生物启发多连杆机器人的位姿与形态同时估计问题。通过设计特定步态和滤波算法，提升估计精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2510.01485](https://arxiv.org/pdf/2510.01485)**

> **作者:** Nicholas B. Andrews; Yanhao Yang; Sofya Akhetova; Kristi A. Morgansen; Ross L. Hatton
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** This work demonstrates simultaneous pose (position and orientation) and shape estimation for a free-floating, bioinspired multi-link robot with unactuated joints, link-mounted thrusters for control, and a single gyroscope per link, resulting in an underactuated, minimally sensed platform. Because the inter-link joint angles are constrained, translation and rotation of the multi-link system requires cyclic, reciprocating actuation of the thrusters, referred to as a gait. Through a proof-of-concept hardware experiment and offline analysis, we show that the robot's shape can be reliably estimated using an Unscented Kalman Filter augmented with Gaussian process residual models to compensate for non-zero-mean, non-Gaussian noise, while the pose exhibits drift expected from gyroscope integration in the absence of absolute position measurements. Experimental results demonstrate that a Gaussian process model trained on a multi-gait dataset (forward, backward, left, right, and turning) performs comparably to one trained exclusively on forward-gait data, revealing an overlap in the gait input space, which can be exploited to reduce per-gait training data requirements while enhancing the filter's generalizability across multiple gaits. Lastly, we introduce a heuristic derived from the observability Gramian to correlate joint angle estimate quality with gait periodicity and thruster inputs, highlighting how control affects estimation quality.
>
---
#### [replaced 049] MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人导航任务，解决零样本环境下导航问题。提出M3DSG和MSGNav系统，保留视觉信息并提升导航性能。**

- **链接: [https://arxiv.org/pdf/2511.10376](https://arxiv.org/pdf/2511.10376)**

> **作者:** Xun Huang; Shijia Zhao; Yunxiang Wang; Xin Lu; Wanfa Zhang; Rongsheng Qu; Weixin Li; Yunhong Wang; Chenglu Wen
>
> **备注:** 18 pages, Accepted by CVPR 2026
>
> **摘要:** Embodied navigation is a fundamental capability for robotic agents operating. Real-world deployment requires open vocabulary generalization and low training overhead, motivating zero-shot methods rather than task-specific RL training. However, existing zero-shot methods that build explicit 3D scene graphs often compress rich visual observations into text-only relations, leading to high construction cost, irreversible loss of visual evidence, and constrained vocabularies. To address these limitations, we introduce the Multi-modal 3D Scene Graph (M3DSG), which preserves visual cues by replacing textual relational edges with dynamically assigned images. Built on M3DSG, we propose MSGNav, a zero-shot navigation system that includes a Key Subgraph Selection module for efficient reasoning, an Adaptive Vocabulary Update module for open vocabulary support, and a Closed-Loop Reasoning module for accurate exploration reasoning. Additionally, we further identify the last mile problem in zero-shot navigation determining the feasible target location with a suitable final viewpoint, and propose a Visibility-based Viewpoint Decision module to explicitly resolve it. Comprehensive experimental results demonstrate that MSGNav achieves state-of-the-art performance on the challenging GOAT-Bench and HM3D-ObjNav benchmark. The code will be publicly available at this https URL.
>
---
#### [replaced 050] Learning Dexterous Manipulation with Quantized Hand State
- **分类: cs.RO**

- **简介: 该论文属于机器人操控任务，旨在解决机械手与手臂协同控制难题。通过量化手部状态，提升学习效率与平衡性。**

- **链接: [https://arxiv.org/pdf/2509.17450](https://arxiv.org/pdf/2509.17450)**

> **作者:** Ying Feng; Hongjie Fang; Yinong He; Jingjing Chen; Chenxi Wang; Zihao He; Ruonan Liu; Cewu Lu
>
> **备注:** accepted by ICRA 2026
>
> **摘要:** Dexterous robotic hands enable robots to perform complex manipulations that require fine-grained control and adaptability. Achieving such manipulation is challenging because the high degrees of freedom tightly couple hand and arm motions, making learning and control difficult. Successful dexterous manipulation relies not only on precise hand motions, but also on accurate spatial positioning of the arm and coordinated arm-hand dynamics. However, most existing visuomotor policies represent arm and hand actions in a single combined space, which often causes high-dimensional hand actions to dominate the coupled action space and compromise arm control. To address this, we propose DQ-RISE, which quantizes hand states to simplify hand motion prediction while preserving essential patterns, and applies a continuous relaxation that allows arm actions to diffuse jointly with these compact hand states. This design enables the policy to learn arm-hand coordination from data while preventing hand actions from overwhelming the action space. Experiments show that DQ-RISE achieves more balanced and efficient learning, paving the way toward structured and generalizable dexterous manipulation. Project website: this http URL
>
---
#### [replaced 051] Dribble Master: Learning Agile Humanoid Dribbling through Legged Locomotion
- **分类: cs.RO**

- **简介: 该论文属于人形机器人足球控球任务，解决动态平衡与精准控球难题。提出两阶段课程学习框架，结合虚拟摄像模型和奖励机制，实现仿真到实物的高效迁移。**

- **链接: [https://arxiv.org/pdf/2505.12679](https://arxiv.org/pdf/2505.12679)**

> **作者:** Zhuoheng Wang; Jinyin Zhou; Qi Wu
>
> **摘要:** Humanoid soccer dribbling is a highly challenging task that demands dexterous ball manipulation while maintaining dynamic balance. Traditional rule-based methods often struggle to achieve accurate ball control due to their reliance on fixed walking patterns and limited adaptability to real-time ball dynamics. To address these challenges, we propose a two-stage curriculum learning framework that enables a humanoid robot to acquire dribbling skills without explicit dynamics or predefined trajectories. In the first stage, the robot learns basic locomotion skills; in the second stage, we fine-tune the policy for agile dribbling maneuvers. We further introduce a virtual camera model in simulation that simulates the field of view and perception constraints of the real robot, enabling realistic ball perception during training. We also design heuristic rewards to encourage active sensing, promoting a broader visual range for continuous ball perception. The policy is trained in simulation and successfully transferred to a physical humanoid robot. Experiment results demonstrate that our method enables effective ball manipulation, achieving flexible and visually appealing dribbling behaviors across multiple environments. This work highlights the potential of reinforcement learning in developing agile humanoid soccer robots. Additional details and videos are available at this https URL.
>
---
#### [replaced 052] Barrier-Riccati Synthesis for Nonlinear Safe Control with Expanded Region of Attraction
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于非线性安全控制任务，解决传统方法区域吸引力有限的问题，通过结合障碍状态与SDRE方法，扩展吸引域并保证安全稳定。**

- **链接: [https://arxiv.org/pdf/2504.15453](https://arxiv.org/pdf/2504.15453)**

> **作者:** Hassan Almubarak; Maitham F. AL-Sunni; Justin T. Dubbin; Nader Sadegh; John M. Dolan; Evangelos A. Theodorou
>
> **备注:** This work has been accepted for publication in the proceedings of the 2026 American Control Conference (ACC), New Orleans, Louisiana, USA
>
> **摘要:** We present a Riccati-based framework for safety-critical nonlinear control that integrates the barrier states (BaS) methodology with the State-Dependent Riccati Equation (SDRE) approach. The BaS formulation embeds safety constraints into the system dynamics via auxiliary states, enabling safety to be treated as a control objective. To overcome the limited region of attraction in linear BaS controllers, we extend the framework to nonlinear systems using SDRE synthesis applied to the barrier-augmented dynamics and derive a matrix inequality condition that certifies forward invariance of a large region of attraction and guarantees asymptotic safe stabilization. The resulting controller is computed online via pointwise Riccati solutions. We validate the method on an unstable constrained system and cluttered quadrotor navigation tasks, demonstrating improved constraint handling, scalability, and robustness near safety boundaries. This framework offers a principled and computationally tractable solution for synthesizing nonlinear safe feedback in safety-critical environments.
>
---
#### [replaced 053] Using VLM Reasoning to Constrain Task and Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于任务与运动规划领域，解决高阶任务规划与低阶运动规划间不匹配的问题。通过引入视觉语言模型进行常识推理，提前识别规划问题，提升规划效率。**

- **链接: [https://arxiv.org/pdf/2510.25548](https://arxiv.org/pdf/2510.25548)**

> **作者:** Muyang Yan; Miras Mengdibayev; Ardon Floros; Weihang Guo; Lydia E. Kavraki; Zachary Kingston
>
> **备注:** 9 pages, 7 figures, 1 table. Submitted to IROS 2026
>
> **摘要:** In task and motion planning, high-level task planning is done over an abstraction of the world to enable efficient search in long-horizon robotics problems. However, the feasibility of these task-level plans relies on the downward refinability of the abstraction into continuous motion. When a domain's refinability is poor, task-level plans that appear valid may ultimately fail during motion planning, requiring replanning and resulting in slower overall performance. Prior works mitigate this by encoding refinement issues as constraints to prune infeasible task plans. However, these approaches only add constraints upon refinement failure, expending significant search effort on infeasible branches. We propose VIZ-COAST, a method of leveraging the common-sense spatial reasoning of large pretrained Vision-Language Models to identify issues with downward refinement a priori, bypassing the need to fix these failures during planning. Experiments on three challenging TAMP domains show that our approach is able to extract plausible constraints from images and domain descriptions, drastically reducing planning times and, in some cases, eliminating downward refinement failures altogether, generalizing to a diverse range of instances from the broader domain.
>
---
#### [replaced 054] Hydrodynamic Performance Enhancement of Unmanned Underwater Gliders with Soft Robotic Morphing Wings for Agility Improvement
- **分类: cs.RO**

- **简介: 该论文属于水下机器人性能优化任务，旨在提升无人水下滑翔机的流体动力效率。通过设计软体可变形机翼，与传统刚性机翼对比，验证其在提高机动性和效率方面的优势。**

- **链接: [https://arxiv.org/pdf/2602.20054](https://arxiv.org/pdf/2602.20054)**

> **作者:** A. Giordano; G. De Meurichy; V. Telazzi; C. Mucignat; I. Lunati; D. A. L. M. Louchard; M. Iovieno; S. F. Armanini; M. Kovac
>
> **备注:** Conference paper accepted at 9th IEEE-RAS International Conference on Soft Robotics (RoboSoft 2026)
>
> **摘要:** This work assesses the hydrodynamic efficiency of Underwater Unmanned Vehicles (UUVs) equipped with soft morphing wings compared to conventional rigid wings. Unlike rigid wings, deformable counterparts can alter their aerodynamic properties on demand. Improvements in hydrodynamic efficiency extend a UUV's operational range and may determine mission feasibility. Structural and Computational Fluid Dynamics (CFD) simulations were conducted for both a soft morphing wing and a UUV incorporating it. The results show that a UUV employing soft wings achieves 9.75 percent higher overall efficiency than an equivalent vehicle with traditional rigid wings. These findings confirm the potential of soft robotics to enhance underwater vehicle performance, particularly in applications requiring pressure-agnostic operation.
>
---
#### [replaced 055] VLAD-Grasp: Zero-shot Grasp Detection via Vision-Language Models
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人抓取任务，解决传统方法依赖标注数据的问题。通过视觉-语言模型生成抓取目标，实现零样本抓取检测。**

- **链接: [https://arxiv.org/pdf/2511.05791](https://arxiv.org/pdf/2511.05791)**

> **作者:** Manav Kulshrestha; S. Talha Bukhari; Damon Conover; Aniket Bera
>
> **备注:** 8 pages, 4 figures, under review
>
> **摘要:** Robotic grasping is a fundamental capability for enabling autonomous manipulation, with usually infinite solutions. State-of-the-art approaches for grasping rely on learning from large-scale datasets comprising expert annotations of feasible grasps. Curating such datasets is challenging, and hence, learning-based methods are limited by the solution coverage of the dataset, and require retraining to handle novel objects. Towards this, we present VLAD-Grasp, a Vision-Language model Assisted zero-shot approach for Detecting Grasps. Our method (1) prompts a large vision-language model to generate a goal image where a virtual cylindrical proxy intersects the object's geometry, explicitly encoding an antipodal grasp axis in image space, then (2) predicts depth and segmentation to lift this generated image into 3D, and (3) aligns generated and observed object point clouds via principal components and correspondence-free optimization to recover an executable grasp pose. Unlike prior work, our approach is training-free and does not require curated grasp datasets, while achieving performance competitive with the state-of-the-art methods on the Cornell and Jacquard datasets. Furthermore, we demonstrate zero-shot generalization to real-world objects on a Franka Research 3 robot, highlighting vision-language models as powerful priors for robotic manipulation.
>
---
#### [replaced 056] ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine for Scalable Contact-Rich Robotics Simulation and Control
- **分类: cs.RO**

- **简介: 该论文提出ComFree-Sim，一个基于GPU并行的接触物理引擎，解决高密度接触场景下物理模拟效率低的问题。通过解析计算接触力，实现近线性扩展，提升仿真速度与控制性能。**

- **链接: [https://arxiv.org/pdf/2603.12185](https://arxiv.org/pdf/2603.12185)**

> **作者:** Chetan Borse; Zhixian Xie; Wei-Cheng Huang; Wanxin Jin
>
> **备注:** 9 pages
>
> **摘要:** Physics simulation for contact-rich robotics is often bottlenecked by contact resolution: mainstream engines enforce non-penetration and Coulomb friction via complementarity constraints or constrained optimization, requiring per-step iterative solves whose cost grows superlinearly with contact density. We present ComFree-Sim, a GPU-parallelized analytical contact physics engine built on complementarity-free contact modeling. ComFree-Sim computes contact impulses in closed form via an impedance-style prediction--correction update in the dual cone of Coulomb friction. Contact computation decouples across contact pairs and becomes separable across cone facets, mapping naturally to GPU kernels and yielding near-linear runtime scaling with the number of contacts. We further extend the formulation to a unified 6D contact model capturing tangential, torsional, and rolling friction, and introduce a practical dual-cone impedance heuristic. ComFree-Sim is implemented in Warp and exposed through a MuJoCo-compatible interface as a drop-in backend alternative to MuJoCo Warp (MJWarp). Experiments benchmark penetration, friction behaviors, stability, and simulation runtime scaling against MJWarp, demonstrating near-linear scaling and 2--3 times higher throughput in dense contact scenes with comparable physical fidelity. We deploy ComFree-Sim in real-time MPC for in-hand dexterous manipulation on a real-world multi-fingered LEAP hand and in dynamics-aware motion retargeting, demonstrating that low-latency simulation yields higher closed-loop success rates and enables practical high-frequency control in contact-rich tasks.
>
---
#### [replaced 057] Density Matrix-based Dynamics for Quantum Robotic Swarms
- **分类: cs.RO; quant-ph**

- **简介: 该论文属于量子机器人集群研究，旨在解决传统矩阵方法在大规模集群建模中的可扩展性问题。提出使用密度矩阵表示混合量子态，实现高效建模与局部信息提取。**

- **链接: [https://arxiv.org/pdf/2509.08002](https://arxiv.org/pdf/2509.08002)**

> **作者:** Maria Mannone; Mahathi Anand; Peppino Fazio; Abdalla Swikir
>
> **摘要:** In a robotic swarm, parameters such as position and proximity to the target can be described in terms of probability amplitudes. This idea led to recent studies on a quantum approach to the definition of the swarm, including a block-matrix representation. However, the size of such matrix-based representation increases drastically with the swarm size, making them impractical for large swarms. Hence, in this work, we propose a new approach for modeling robotic swarms and robotic networks by considering them as mixed quantum states that can be represented mathematically via density matrices. The size of such an approach only depends on the available degrees of freedom of the robot, and not its swarm size and thus scales well to large swarms. Moreover, it also enables the extraction of local information of the robots from the global swarm information contained in the density matrices, facilitating decentralized behavior that aligns with the collective swarm behavior. Our approach is validated on several simulations including large-scale swarms of up to 1000 robots. Finally, we provide some directions for future research that could potentially widen the impact of our approach.
>
---
#### [replaced 058] Optimal Modified Feedback Strategies in LQ Games under Control Imperfections
- **分类: cs.GT; cs.MA; cs.RO; eess.SY; math.OC**

- **简介: 该论文属于博弈论与控制领域的任务，解决LQ游戏中因控制不完善导致的策略偏差问题。通过构建补偿策略，提升受影响玩家的性能。**

- **链接: [https://arxiv.org/pdf/2503.19200](https://arxiv.org/pdf/2503.19200)**

> **作者:** Mahdis Rabbani; Navid Mojahed; Shima Nazari
>
> **备注:** 8 pages, 2 figures, Manuscript accepted to ACC 2026
>
> **摘要:** Game-theoretic approaches and Nash equilibrium have been widely applied across various engineering domains. However, practical challenges such as disturbances, delays, and actuator limitations can hinder the precise execution of Nash equilibrium strategies. This work investigates the impact of such implementation imperfections on game trajectories and players' costs in the context of a two-player finite-horizon linear quadratic (LQ) nonzero-sum game. Specifically, we analyze how small deviations by one player, measured or estimated at each stage affect the state trajectory and the other player's cost. To mitigate these effects, we construct a compensation law for the influenced player by augmenting the nominal game with the measurable deviation dynamics. The resulting policy is shown to be optimal within a causal affine policy class, and, for sufficiently small deviations, it locally outperforms the uncompensated equilibrium-derived feedback. Rigorous analysis and proofs are provided, and the effectiveness of the proposed approach is demonstrated through a representative numerical example.
>
---
#### [replaced 059] MoRoCo: An Online Topology-Adaptive Framework for Multi-Operator Multi-Robot Coordination under Restricted Communication
- **分类: cs.RO**

- **简介: 该论文属于多机器人协同任务，解决受限通信下的多操作员多机器人协调问题。提出MoRoCo框架，支持动态通信拓扑和在线请求处理。**

- **链接: [https://arxiv.org/pdf/2508.07657](https://arxiv.org/pdf/2508.07657)**

> **作者:** Zhuoli Tian; Yanze Bao; Yuyang Zhang; Meng Guo
>
> **备注:** 20 pages, 19 figures. Submitted to IEEE Transactions on Robotics (TRO)
>
> **摘要:** Fleets of autonomous robots are increasingly deployed with multiple human operators in communication-restricted environments for exploration and intervention tasks such as subterranean inspection, reconnaissance, and search-and-rescue. In these settings, communication is often limited to short-range ad-hoc links, making it difficult to coordinate exploration while supporting online human-fleet interactions. Existing work on multi-robot exploration largely focuses on information gathering itself, but pays limited attention to the fact that operators and robots issue time-critical requests during execution. These requests may require different communication structures, ranging from intermittent status delivery to sustained video streaming and teleoperation. To address this challenge, this paper presents MoRoCo, an online topology-adaptive framework for multi-operator multi-robot coordination under restricted communication. MoRoCo is built on a latency-bounded intermittent communication backbone that guarantees a prescribed delay for information collected by any robot to reach an operator, together with a detach-and-rejoin mechanism that enables online team resizing and topology reconfiguration. On top of this backbone, the framework instantiates request-consistent communication subgraphs to realize different modes of operator-robot interaction by jointly assigning robot roles, positions, and communication topology. It further supports the online decomposition and composition of these subgraphs using only local communication, allowing multiple requests to be serviced during exploration. The framework extends to heterogeneous fleets, multiple teams, and robot failures. Extensive human-in-the-loop simulations and hardware experiments demonstrate effective and reliable coordination under restricted communication.
>
---
#### [replaced 060] History-Aware Visuomotor Policy Learning via Point Tracking
- **分类: cs.RO**

- **简介: 该论文属于视觉运动控制任务，解决传统方法在处理长期依赖和重复状态时的不足。通过点跟踪构建对象中心的历史表示，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2509.17141](https://arxiv.org/pdf/2509.17141)**

> **作者:** Jingjing Chen; Hongjie Fang; Chenxi Wang; Shiquan Wang; Cewu Lu
>
> **备注:** accepted by ICRA 2026
>
> **摘要:** Many manipulation tasks require memory beyond the current observation, yet most visuomotor policies rely on the Markov assumption and thus struggle with repeated states or long-horizon dependencies. Existing methods attempt to extend observation horizons but remain insufficient for diverse memory requirements. To this end, we propose an object-centric history representation based on point tracking, which abstracts past observations into a compact and structured form that retains only essential task-relevant information. Tracked points are encoded and aggregated at the object level, yielding a compact history representation that can be seamlessly integrated into various visuomotor policies. Our design provides full history-awareness with high computational efficiency, leading to improved overall task performance and decision accuracy. Through extensive evaluations on diverse manipulation tasks, we show that our method addresses multiple facets of memory requirements - such as task stage identification, spatial memorization, and action counting, as well as longer-term demands like continuous and pre-loaded memory - and consistently outperforms both Markovian baselines and prior history-based approaches. Project website: this http URL
>
---
#### [replaced 061] Walking through Doors is Hard, even without Staircases: Universality and PSPACE-hardness of Planar Door Gadgets
- **分类: cs.CC; cs.RO**

- **简介: 该论文研究运动规划中的门装置，证明其可达性问题为PSPACE完全，简化了游戏复杂性证明，并扩展到多种门类型及3D游戏。**

- **链接: [https://arxiv.org/pdf/2006.01256](https://arxiv.org/pdf/2006.01256)**

> **作者:** MIT Gadgets Group; Jeffrey Bosboom; Erik D. Demaine; Jenny Diomidova; Dylan Hendrickson; Hayashi Layers; Jayson Lynch
>
> **备注:** 36 pages, 35 figures. All cases are now proved PSPACE-complete. New universality proofs. Earlier version published at FUN 2020
>
> **摘要:** An open-close door gadget has two states and three tunnels that can be traversed by an agent (player, robot, etc.): the "opening" and "closing" tunnels set the gadget's state to open and closed, respectively, while the "traverse" tunnel can be traversed if and only if the door is in the open state. We prove that it is PSPACE-complete to decide whether an agent can move from one location to another through a planar system of any such door gadget, removing the traditional need for crossover gadgets and thereby simplifying past PSPACE-hardness proofs of Lemmings and Nintendo games Super Mario Bros., Legend of Zelda, and Donkey Kong Country. Even stronger, we show that any gadget in the motion-planning-through-gadgets framework can be simulated by a planar system of door gadgets: the open-close door gadget is a universal gadget. We prove that these results hold for a variety of door gadgets. In particular, the opening, closing, and traverse tunnel locations can have an arbitrary cyclic order around the door; each tunnel can be directed or undirected; and the opening tunnel can instead be an optional button (with identical entrance and exit locations). Furthermore, we show the same hardness and universality results for two simpler types of door gadgets: self-closing door gadgets and symmetric self-closing door gadgets. Again we show that any self-closing door gadget planarly simulates any gadget, and thus the reachability motion planning problem is PSPACE-complete. Then we apply this framework to prove new PSPACE-hardness results for eight different 3D Mario video games and Sokobond.
>
---
#### [replaced 062] RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在解决现有策略缺乏记忆能力的问题。通过构建RMBench基准和提出Mem-0架构，评估并分析记忆设计对性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.01229](https://arxiv.org/pdf/2603.01229)**

> **作者:** Tianxing Chen; Yuran Wang; Mingleyang Li; Yan Qin; Hao Shi; Zixuan Li; Yifan Hu; Yingsheng Zhang; Kaixuan Wang; Yue Chen; Hongcheng Wang; Renjing Xu; Ruihai Wu; Yao Mu; Yaodong Yang; Hao Dong; Ping Luo
>
> **备注:** website: this https URL
>
> **摘要:** Robotic manipulation policies have made rapid progress in recent years, yet most existing approaches give limited consideration to memory capabilities. Consequently, they struggle to solve tasks that require reasoning over historical observations and maintaining task-relevant information over time, which are common requirements in real-world manipulation scenarios. Although several memory-aware policies have been proposed, systematic evaluation of memory-dependent manipulation remains underexplored, and the relationship between architectural design choices and memory performance is still not well understood. To address this gap, we introduce RMBench, a simulation benchmark comprising 9 manipulation tasks that span multiple levels of memory complexity, enabling systematic evaluation of policy memory capabilities. We further propose Mem-0, a modular manipulation policy with explicit memory components designed to support controlled ablation studies. Through extensive simulation and real-world experiments, we identify memory-related limitations in existing policies and provide empirical insights into how architectural design choices influence memory performance. The website is available at this https URL.
>
---
#### [replaced 063] SToRM: Supervised Token Reduction for Multi-modal LLMs toward efficient end-to-end autonomous driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出SToRM框架，解决多模态大语言模型在自动驾驶中计算资源消耗大的问题，通过令牌压缩实现高效端到端驾驶。**

- **链接: [https://arxiv.org/pdf/2602.11656](https://arxiv.org/pdf/2602.11656)**

> **作者:** Seo Hyun Kim; Jin Bok Park; Do Yeon Koo; Hogun Park; Il Yong Chun
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** In autonomous driving, end-to-end (E2E) driving systems that predict control commands directly from sensor data have achieved significant advancements. For safe driving in unexpected scenarios, these systems may additionally rely on human interventions such as natural language instructions. Using a multi-modal large language model (MLLM) facilitates human-vehicle interaction and can improve performance in such scenarios. However, this approach requires substantial computational resources due to its reliance on an LLM and numerous visual tokens from sensor inputs, which are limited in autonomous vehicles. Many MLLM studies have explored reducing visual tokens, but often suffer end-task performance degradation compared to using all tokens. To enable efficient E2E driving while maintaining performance comparable to using all tokens, this paper proposes the first Supervised Token Reduction framework for multi-modal LLMs (SToRM). The proposed framework consists of three key elements. First, a lightweight importance predictor with short-term sliding windows estimates token importance scores. Second, a supervised training approach uses an auxiliary path to obtain pseudo-supervision signals from an all-token LLM pass. Third, an anchor-context merging module partitions tokens into anchors and context tokens, and merges context tokens into relevant anchors to reduce redundancy while minimizing information loss. Experiments on the LangAuto benchmark show that SToRM outperforms state-of-the-art E2E driving MLLMs under the same reduced-token budget, maintaining all-token performance while reducing computational cost by up to 30x, and enabling real-time E2E driving on a standard GPU.
>
---
#### [replaced 064] SIL: Symbiotic Interactive Learning for Language-Conditioned Human-Agent Co-Adaptation
- **分类: cs.RO**

- **简介: 该论文提出SIL框架，解决人机协同适应问题，通过双向互动提升任务完成率和信念对齐。**

- **链接: [https://arxiv.org/pdf/2511.05203](https://arxiv.org/pdf/2511.05203)**

> **作者:** Linus Nwankwo; Bjoern Ellensohn; Christian Rauch; Elmar Rueckert
>
> **摘要:** Today's autonomous agents, largely driven by foundation models (FMs), can understand natural language instructions and solve long-horizon tasks with human-like reasoning. However, current human-robot interaction largely follows a one-way master-apprentice technique where the agent passively executes commands without reciprocal learning. This neglects the co-adaptive, multi-turn nature of everyday human interactions. We introduce symbiotic interactive learning (SIL), a bidirectional co-adaptation framework in a shared latent task space, where human and agent maintain joint belief states that evolve with interaction history. This enables proactive clarification, adaptive suggestions, and shared plan refinement. SIL leverages FMs for spatial perception and reasoning, together with a triplet-loss-trained neural encoder that grounds FMs' outputs into task-specific latent representations. To support long-term stability as tasks evolve, SIL uses episodic and semantic memory architectures, regularised via elastic weight consolidation to mitigate catastrophic forgetting. We evaluate SIL on simulated and real-world embodied tasks, including instruction following, information retrieval, query-oriented reasoning, and interactive dialogue, achieving a $90.4\%$ task completion rate and a belief alignment score of $\rho \approx 0.83$, an absolute improvement of about $20$ percentage points over the best ablations. Demos and resources: this https URL.
>
---
#### [replaced 065] Risk-Aware Obstacle Avoidance Algorithm for Real-Time Applications
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决动态环境中障碍物避让问题。提出一种融合概率建模与轨迹优化的算法，提升航行安全与适应性。**

- **链接: [https://arxiv.org/pdf/2602.09204](https://arxiv.org/pdf/2602.09204)**

> **作者:** Ozan Kaya; Emir Cem Gezer; Roger Skjetne; Ingrid Bouwer Utne
>
> **摘要:** Robust navigation in changing marine environments requires autonomous systems capable of perceiving, reasoning, and acting under uncertainty. This study introduces a hybrid risk-aware navigation architecture that integrates probabilistic modeling of obstacles along the vehicle path with smooth trajectory optimization for autonomous surface vessels. The system constructs probabilistic risk maps that capture both obstacle proximity and the behavior of dynamic objects. A risk-biased Rapidly Exploring Random Tree (RRT) planner leverages these maps to generate collision-free paths, which are subsequently refined using B-spline algorithms to ensure trajectory continuity. Three distinct RRT* rewiring modes are implemented based on the cost function: minimizing the path length, minimizing risk, and optimizing a combination of the path length and total risk. The framework is evaluated in experimental scenarios containing both static and dynamic obstacles. The results demonstrate the system's ability to navigate safely, maintain smooth trajectories, and dynamically adapt to changing environmental risks. Compared with conventional LIDAR or vision-only navigation approaches, the proposed method shows improvements in operational safety and autonomy, establishing it as a promising solution for risk-aware autonomous vehicle missions in uncertain and dynamic environments.
>
---
#### [replaced 066] HandelBot: Real-World Piano Playing via Fast Adaptation of Dexterous Robot Policies
- **分类: cs.RO**

- **简介: 该论文提出HandelBot，解决多指机器人高精度操作问题，通过模拟策略与快速适应结合，实现精确钢琴演奏。**

- **链接: [https://arxiv.org/pdf/2603.12243](https://arxiv.org/pdf/2603.12243)**

> **作者:** Amber Xie; Haozhi Qi; Dorsa Sadigh
>
> **备注:** Website: this https URL
>
> **摘要:** Mastering dexterous manipulation with multi-fingered hands has been a grand challenge in robotics for decades. Despite its potential, the difficulty of collecting high-quality data remains a primary bottleneck for high-precision tasks. While reinforcement learning and simulation-to-real-world transfer offer a promising alternative, the transferred policies often fail for tasks demanding millimeter-scale precision, such as bimanual piano playing. In this work, we introduce HandelBot, a framework that combines a simulation policy and rapid adaptation through a two-stage pipeline. Starting from a simulation-trained policy, we first apply a structured refinement stage to correct spatial alignments by adjusting lateral finger joints based on physical rollouts. Next, we use residual reinforcement learning to autonomously learn fine-grained corrective actions. Through extensive hardware experiments across five recognized songs, we demonstrate that HandelBot can successfully perform precise bimanual piano playing. Our system outperforms direct simulation deployment by a factor of 1.8x and requires only 30 minutes of physical interaction data.
>
---
#### [replaced 067] DynaFlow: Dynamics-embedded Flow Matching for Physically Consistent Motion Generation from State-only Demonstrations
- **分类: cs.RO**

- **简介: 该论文提出DynaFlow，解决从仅状态演示生成物理一致运动的问题。通过将可微模拟器嵌入流匹配模型，生成动作并确保物理可行性，实现端到端训练与真实机器人部署。**

- **链接: [https://arxiv.org/pdf/2509.19804](https://arxiv.org/pdf/2509.19804)**

> **作者:** Sowoo Lee; Dongyun Kang; Jaehyun Park; Hae-Won Park
>
> **备注:** 8 pages
>
> **摘要:** This paper introduces DynaFlow, a novel framework that embeds a differentiable simulator directly into a flow matching model. By generating trajectories in the action space and mapping them to dynamically feasible state trajectories via the simulator, DynaFlow ensures all outputs are physically consistent by construction. This end-to-end differentiable architecture enables training on state-only demonstrations, allowing the model to simultaneously generate physically consistent state trajectories while inferring the underlying action sequences required to produce them. We demonstrate the effectiveness of our approach through quantitative evaluations and showcase its real-world applicability by deploying the generated actions onto a physical Go1 quadruped robot. The robot successfully reproduces diverse gait present in the dataset, executes long-horizon motions in open-loop control and translates infeasible kinematic demonstrations into dynamically executable, stylistic behaviors. These hardware experiments validate that DynaFlow produces deployable, highly effective motions on real-world hardware from state-only demonstrations, effectively bridging the gap between kinematic data and real-world execution.
>
---
#### [replaced 068] PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 论文提出PhysMoDPO，解决文本控制下人形机器人运动的物理真实性和指令一致性问题。通过整合物理控制器和偏好优化，提升运动生成的物理合规性与任务准确性。**

- **链接: [https://arxiv.org/pdf/2603.13228](https://arxiv.org/pdf/2603.13228)**

> **作者:** Yangsong Zhang; Anujith Muraleedharan; Rikhat Akizhanov; Abdul Ahad Butt; Gül Varol; Pascal Fua; Fabio Pizzati; Ivan Laptev
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent progress in text-conditioned human motion generation has been largely driven by diffusion models trained on large-scale human motion data. Building on this progress, recent methods attempt to transfer such models for character animation and real robot control by applying a Whole-Body Controller (WBC) that converts diffusion-generated motions into executable trajectories. While WBC trajectories become compliant with physics, they may expose substantial deviations from original motion. To address this issue, we here propose PhysMoDPO, a Direct Preference Optimization framework. Unlike prior work that relies on hand-crafted physics-aware heuristics such as foot-sliding penalties, we integrate WBC into our training pipeline and optimize diffusion model such that the output of WBC becomes compliant both with physics and original text instructions. To train PhysMoDPO we deploy physics-based and task-specific rewards and use them to assign preference to synthesized trajectories. Our extensive experiments on text-to-motion and spatial control tasks demonstrate consistent improvements of PhysMoDPO in both physical realism and task-related metrics on simulated robots. Moreover, we demonstrate that PhysMoDPO results in significant improvements when applied to zero-shot motion transfer in simulation and for real-world deployment on a G1 humanoid robot.
>
---
#### [replaced 069] Eva-VLA: Evaluating Vision-Language-Action Models' Robustness Under Real-World Physical Variations
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人视觉-语言-动作模型的鲁棒性评估任务，旨在解决真实物理环境变化下的模型脆弱性问题。通过构建Eva-VLA框架，系统化分析并生成最坏场景以提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.18953](https://arxiv.org/pdf/2509.18953)**

> **作者:** Hanqing Liu; Shouwei Ruan; Jiahuan Long; Junqi Wu; Jiacheng Hou; Huili Tang; Tingsong Jiang; Weien Zhou; Wen Yao
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as promising solutions for robotic manipulation, yet their robustness to real-world physical variations remains critically underexplored. To bridge this gap, we propose Eva-VLA, the first unified framework to systematically evaluate the robustness of VLA models by formulating uncontrollable physical variations as continuous optimization problems. Specifically, our framework addresses two fundamental challenges in VLA models' physical robustness evaluation: 1) how to systematically characterize diverse physical perturbations encountered in real-world deployment while maintaining reproducibility, and 2) how to efficiently discover worst-case scenarios without incurring prohibitive real-world data collection costs. To tackle the first challenge, we decouple real-world variations into three key dimensions: 3D object transformations that affect spatial reasoning, illumination changes that challenge visual perception, and adversarial regions that disrupt scene understanding. For the second challenge, we introduce a continuous black-box optimization mechanism that maps these perturbations into a continuous parameter space, enabling the systematic exploration of worst-case scenarios. Extensive experiments validate the effectiveness of our approach. Notably, OpenVLA exhibits an average failure rate of over 90% across three physical variations on the LIBERO-Long task, exposing critical systemic fragilities. Furthermore, applying the generated worst-case scenarios during adversarial training quantifiably increases model robustness, validating the effectiveness of this approach. Our evaluation exposes the gap between laboratory and real-world conditions, while the Eva-VLA framework can serve as an effective data augmentation method to enhance the resilience of robotic manipulation systems.
>
---
#### [replaced 070] OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera
- **分类: cs.RO; cs.CV; eess.IV**

- **简介: 该论文提出OneOcc，解决腿部机器人3D语义占据预测问题，利用单全景相机实现鲁棒的环境感知。**

- **链接: [https://arxiv.org/pdf/2511.03571](https://arxiv.org/pdf/2511.03571)**

> **作者:** Hao Shi; Ze Wang; Shangwei Guo; Mengfei Duan; Song Wang; Teng Chen; Kailun Yang; Lin Wang; Kaiwei Wang
>
> **备注:** Accepted to CVPR 2026. Datasets and code will be publicly available at this https URL
>
> **摘要:** Robust 3D semantic occupancy is crucial for legged/humanoid robots, yet most semantic scene completion (SSC) systems target wheeled platforms with forward-facing sensors. We present OneOcc, a vision-only panoramic SSC framework designed for gait-introduced body jitter and 360° continuity. OneOcc combines: (i) Dual-Projection fusion (DP-ER) to exploit the annular panorama and its equirectangular unfolding, preserving 360° continuity and grid alignment; (ii) Bi-Grid Voxelization (BGV) to reason in Cartesian and cylindrical-polar spaces, reducing discretization bias and sharpening free/occupied boundaries; (iii) a lightweight decoder with Hierarchical AMoE-3D for dynamic multi-scale fusion and better long-range/occlusion reasoning; and (iv) plug-and-play Gait Displacement Compensation (GDC) learning feature-level motion correction without extra sensors. We also release two panoramic occupancy benchmarks: QuadOcc (real quadruped, first-person 360°) and Human360Occ (H3O) (CARLA human-ego 360° with RGB, Depth, semantic occupancy; standardized within-/cross-city splits). OneOcc sets a new state of the art on QuadOcc, outperforming strong vision baselines and remaining competitive with classical LiDAR baselines; on H3O it gains +3.83 mIoU (within-city) and +8.08 (cross-city). Modules are lightweight, enabling deployable full-surround perception for legged/humanoid robots. Datasets and code will be publicly available at this https URL.
>
---
#### [replaced 071] VLD: Visual Language Goal Distance for Reinforcement Learning Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人导航任务，旨在解决政策迁移困难和数据不足的问题。提出VLD学习框架，分离感知与策略学习，利用视觉-语言距离信号提升导航性能。**

- **链接: [https://arxiv.org/pdf/2512.07976](https://arxiv.org/pdf/2512.07976)**

> **作者:** Lazar Milikic; Manthan Patel; Jonas Frey
>
> **摘要:** Training end-to-end policies from image data to directly predict navigation actions for robotic systems has proven inherently difficult. Existing approaches often suffer from either the sim-to-real gap during policy transfer or a limited amount of training data with action labels. To address this problem, we introduce Vision-Language Distance (VLD) learning, a scalable framework for goal-conditioned navigation that decouples perception learning from policy learning. Instead of relying on raw sensory inputs during policy training, we first train a self-supervised distance-to-goal predictor on internet-scale video data. This predictor generalizes across both image- and text-based goals, providing a distance signal that can be minimized by a reinforcement learning (RL) policy. The RL policy can be trained entirely in simulation using privileged geometric distance signals, with injected noise to mimic the uncertainty of the trained distance predictor. At deployment, the policy consumes VLD predictions, inheriting semantic goal information-"where to go"-from large-scale visual training while retaining the robust low-level navigation behaviors learned in simulation. We propose using ordinal consistency to assess distance functions directly and demonstrate that VLD outperforms prior temporal distance approaches, such as ViNT and VIP. Experiments show that our decoupled design achieves competitive navigation performance in simulation with strong sim-to-real transfer, providing an alternative and, most importantly, scalable path toward reliable, multimodal navigation policies.
>
---
#### [replaced 072] TinyIO: Lightweight Reparameterized Inertial Odometry
- **分类: cs.RO**

- **简介: 该论文属于定位任务，旨在解决轻量级惯性里程计的高精度问题。提出TinyIO方法，通过多分支结构和双路径注意力机制，在减少参数量的同时提升性能。**

- **链接: [https://arxiv.org/pdf/2507.15293](https://arxiv.org/pdf/2507.15293)**

> **作者:** Shanshan Zhang; Siyue Wang; Mengzi Chen; Mengzhe Wang; Liqin Wu; Qi Zhang; Lingxiang Zheng
>
> **摘要:** Inertial odometry (IO) is a widely used approach for localization on mobile devices; however, obtaining a lightweight IO model that also achieves high accuracy remains challenging. To address this issue, we propose TinyIO, a lightweight IO method. During training, we adopt a multi-branch architecture to extract diverse motion features more effectively. At inference time, the trained multi-branch model is converted into an equivalent single-path architecture to reduce computational complexity. We further propose a Dual-Path Adaptive Attention mechanism (DPAA), which enhances TinyIO's perception of contextual motion along both channel and temporal dimensions with negligible additional parameters. Extensive experiments on public datasets demonstrate that our method attains a favorable trade-off between accuracy and model size. On the RoNIN dataset, TinyIO reduces the ATE by 23.53% compared with R-ResNet and decreases the parameter count by 3.68%.
>
---
#### [replaced 073] Register Any Point: Scaling 3D Point Cloud Registration by Flow Matching
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D点云配准任务，解决多视角点云对齐问题。通过条件生成方法直接生成配准点云，提升效率与全局一致性。**

- **链接: [https://arxiv.org/pdf/2512.01850](https://arxiv.org/pdf/2512.01850)**

> **作者:** Yue Pan; Tao Sun; Liyuan Zhu; Lucas Nunes; Iro Armeni; Jens Behley; Cyrill Stachniss
>
> **摘要:** Point cloud registration aligns multiple unposed point clouds into a common reference frame and is a core step for 3D reconstruction and robot localization without initial guess. In this work, we cast registration as conditional generation: a learned, continuous point-wise velocity field transports noisy points to a registered scene, from which the pose of each view is recovered. Unlike prior methods that perform correspondence matching to estimate pairwise transformations and then optimize a pose graph for multi-view registration, our model directly generates the registered point cloud, yielding both efficiency and point-level global consistency. By scaling the training data and conducting test-time rigidity enforcement, our approach achieves state-of-the-art results on existing pairwise registration benchmarks and on our proposed cross-domain multi-view registration benchmark. The superior zero-shot performance on this benchmark shows that our method generalizes across view counts, scene scales, and sensor modalities even with low overlap. Source code available at: this https URL.
>
---
#### [replaced 074] IRIS-SLAM: Unified Geo-Instance Representations for Robust Semantic Localization and Mapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出IRIS-SLAM，解决语义定位与建图中的几何-实例一致性问题，通过统一表示提升地图一致性和回环检测性能。**

- **链接: [https://arxiv.org/pdf/2602.18709](https://arxiv.org/pdf/2602.18709)**

> **作者:** Tingyang Xiao; Liu Liu; Wei Feng; Zhengyu Zou; Xiaolin Zhou; Wei Sui; Hao Li; Dingwen Zhang; Zhizhong Su
>
> **备注:** This version is being withdrawn because it was submitted without the final review and formal approval of all co-authors. The authors plan to resubmit a revised version once all internal approvals are secured
>
> **摘要:** Geometry foundation models have significantly advanced dense geometric SLAM, yet existing systems often lack deep semantic understanding and robust loop closure capabilities. Meanwhile, contemporary semantic mapping approaches are frequently hindered by decoupled architectures and fragile data association. We propose IRIS-SLAM, a novel RGB semantic SLAM system that leverages unified geometric-instance representations derived from an instance-extended foundation model. By extending a geometry foundation model to concurrently predict dense geometry and cross-view consistent instance embeddings, we enable a semantic-synergized association mechanism and instance-guided loop closure detection. Our approach effectively utilizes viewpoint-agnostic semantic anchors to bridge the gap between geometric reconstruction and open-vocabulary mapping. Experimental results demonstrate that IRIS-SLAM significantly outperforms state-of-the-art methods, particularly in map consistency and wide-baseline loop closure reliability.
>
---
#### [replaced 075] Hierarchical Diffusion Motion Planning with Task-Conditioned Uncertainty-Aware Priors
- **分类: cs.RO**

- **简介: 该论文提出一种分层扩散规划方法，用于机器人运动规划任务。解决传统方法在轨迹平滑性和任务语义编码上的不足，通过引入任务相关的高斯先验，提升规划成功率和轨迹质量。**

- **链接: [https://arxiv.org/pdf/2509.25685](https://arxiv.org/pdf/2509.25685)**

> **作者:** Amelie Minji Kim; Anqi Wu; Ye Zhao
>
> **摘要:** We propose a novel hierarchical diffusion planner that embeds task and motion structure directly into the noise model. Unlike standard diffusion-based planners that rely on zero-mean, isotropic Gaussian corruption, we introduce task-conditioned structured Gaussians whose means and covariances are derived from Gaussian Process Motion Planning (GPMP), explicitly encoding trajectory smoothness and task semantics in the prior. We first generalize the standard diffusion process to biased, non-isotropic corruption with closed-form forward and posterior expressions. Building on this formulation, our hierarchical design separates prior instantiation from trajectory denoising. At the upper level, the model predicts sparse, task-centric key states and their associated timings, which instantiate a structured Gaussian prior (mean and covariance). At the lower level, the full trajectory is denoised under this fixed prior, treating the upper-level outputs as noisy observations. Experiments on Maze2D goal-reaching and KUKA block stacking show consistently higher success rates and smoother trajectories than isotropic baselines, achieving dataset-level smoothness substantially earlier during training. Ablation studies further show that explicitly structuring the corruption process provides benefits beyond neural conditioning the denoising network alone. Overall, our approach concentrates the prior's probability mass near feasible and semantically meaningful trajectories. Our project page is available at this https URL.
>
---
#### [replaced 076] EMMA: Generalizing Real-World Robot Manipulation via Generative Visual Transfer
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出EMMA框架，解决机器人操作中数据获取困难的问题。通过生成式视觉迁移增强VLA模型，提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.22407](https://arxiv.org/pdf/2509.22407)**

> **作者:** Zhehao Dong; Xiaofeng Wang; Zheng Zhu; Yirui Wang; Yang Wang; Yukun Zhou; Boyuan Wang; Chaojun Ni; Runqi Ouyang; Wenkang Qin; Xinze Chen; Yun Ye; Guan Huang; Zhen Lu; Yue Yang
>
> **摘要:** The generalization of vision-language-action (VLA) models heavily relies on diverse training data. However, acquiring large-scale data for robot manipulation across varied object appearances is costly and labor-intensive. To address this limitation, we introduce Embodied Manipulation Media Adaptation (EMMA), a framework for augmenting VLA policies that combines a generative data engine with an effective training pipeline. We introduce DreamTransfer, a diffusion Transformer-based architecture for generating multi-view consistent and geometrically grounded embodied manipulation videos. DreamTransfer enables visual editing of robot videos through prompts, allowing for changes to the foreground, background, and lighting while preserving their 3D structure and geometric validity. We also utilize a hybrid training set of real and generated data and propose AdaMix to enhance the training process. AdaMix is a training strategy that adaptively weights samples according to policy performance to emphasize challenging samples. Comprehensive evaluations demonstrate that videos created by DreamTransfer yield substantial improvements over previous video generation techniques in multi-view consistency, geometric accuracy, and text-conditioning precision. We conduct extensive evaluations with a total of more than 1800 trials in both simulated and real-world robotic environments. In real-world robotic tasks with zero-shot visual settings, our framework achieves a relative performance increase of over 92% compared to training with real data alone, and improves by an additional 17% with AdaMix, demonstrating its efficacy in enhancing policy generalization.
>
---
