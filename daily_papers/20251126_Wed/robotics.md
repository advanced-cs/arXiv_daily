# 机器人 cs.RO

- **最新发布 48 篇**

- **更新 19 篇**

## 最新发布

#### [new 001] How Robot Kinematics Influence Human Performance in Virtual Robot-to-Human Handover Tasks
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究虚拟现实下人机交接任务中机器人运动学对人类表现的影响。针对人机协作中协调性差的问题，通过VR实验考察任务启动、伙伴形态、速度轨迹及旋转时机等因素，发现早期显著视觉提示和类人平滑轨迹可提升人类预测与同步能力，优化交互设计。**

- **链接: [https://arxiv.org/pdf/2511.20299v1](https://arxiv.org/pdf/2511.20299v1)**

> **作者:** Róisín Keenan; Joost C. Dessing
>
> **摘要:** Recent advancements in robotics have increased the possibilities for integrating robotic systems into human-involved workplaces, highlighting the need to examine and optimize human-robot coordination in collaborative settings. This study explores human-robot interactions during handover tasks using Virtual Reality (VR) to investigate differences in human motor performance across various task dynamics and robot kinematics. A VR-based robot handover simulation afforded safe and controlled assessments of human-robot interactions. In separate experiments, four potential influences on human performance were examined (1) control over task initiation and robot movement synchrony (temporal and spatiotemporal); (2) partner appearance (human versus robotic); (3) robot velocity profiles (minimum jerk, constant velocity, constant acceleration, and biphasic); and (4) the timing of rotational object motion. Findings across experiments emphasize humans benefit from robots providing early and salient visual information about task-relevant object motion, and advantages of human-like smooth robot trajectories. To varying degrees, these manipulations improved predictive accuracy and synchronization during interaction. This suggests that human-robot interactions should be designed to allow humans to leverage their natural capabilities for detecting biological motion, which conversely may reduce the need for costly robotic computations or added cognitive adaptation on the human side.
>
---
#### [new 002] Quality-guided UAV Surface Exploration for 3D Reconstruction
- **分类: cs.RO**

- **简介: 该论文针对自主无人机3D重建中的探索规划问题，提出一种基于重建质量的模块化下一最佳视角（NBV）框架。通过利用TSDF中的不确定性，自适应生成与选择视点，优化覆盖范围、重建质量和路径效率，显著优于传统方法。**

- **链接: [https://arxiv.org/pdf/2511.20353v1](https://arxiv.org/pdf/2511.20353v1)**

> **作者:** Benjamin Sportich; Kenza Boubakri; Olivier Simonin; Alessandro Renzaglia
>
> **摘要:** Reasons for mapping an unknown environment with autonomous robots are wide-ranging, but in practice, they are often overlooked when developing planning strategies. Rapid information gathering and comprehensive structural assessment of buildings have different requirements and therefore necessitate distinct methodologies. In this paper, we propose a novel modular Next-Best-View (NBV) planning framework for aerial robots that explicitly uses a reconstruction quality objective to guide the exploration planning. In particular, our approach introduces new and efficient methods for view generation and selection of viewpoint candidates that are adaptive to the user-defined quality requirements, fully exploiting the uncertainty encoded in a Truncated Signed Distance field (TSDF) representation of the environment. This results in informed and efficient exploration decisions tailored towards the predetermined objective. Finally, we validate our method via extensive simulations in realistic environments. We demonstrate that it successfully adjusts its behavior to the user goal while consistently outperforming conventional NBV strategies in terms of coverage, quality of the final 3D map and path efficiency.
>
---
#### [new 003] Collaborate sim and real: Robot Bin Packing Learning in Real-world and Physical Engine
- **分类: cs.RO**

- **简介: 该论文针对3D机器人装箱任务中因仿真与现实物理差异导致的堆叠不稳问题，提出一种协同仿真与真实数据的强化学习框架。通过域随机化增强仿真泛化性，并结合真实反馈微调模型，显著降低装箱坍塌率，提升实际部署效果。**

- **链接: [https://arxiv.org/pdf/2511.19932v1](https://arxiv.org/pdf/2511.19932v1)**

> **作者:** Lidi Zhang; Han Wu; Liyu Zhang; Ruofeng Liu; Haotian Wang; Chao Li; Desheng Zhang; Yunhuai Liu; Tian He
>
> **摘要:** The 3D bin packing problem, with its diverse industrial applications, has garnered significant research attention in recent years. Existing approaches typically model it as a discrete and static process, while real-world applications involve continuous gravity-driven interactions. This idealized simplification leads to infeasible deployments (e.g., unstable packing) in practice. Simulations with physical engine offer an opportunity to emulate continuous gravity effects, enabling the training of reinforcement learning (RL) agents to address such limitations and improve packing stability. However, a simulation-to-reality gap persists due to dynamic variations in physical properties of real-world objects, such as various friction coefficients, elasticity, and non-uniform weight distributions. To bridge this gap, we propose a hybrid RL framework that collaborates with physical simulation with real-world data feedback. Firstly, domain randomization is applied during simulation to expose agents to a spectrum of physical parameters, enhancing their generalization capability. Secondly, the RL agent is fine-tuned with real-world deployment feedback, further reducing collapse rates. Extensive experiments demonstrate that our method achieves lower collapse rates in both simulated and real-world scenarios. Large-scale deployments in logistics systems validate the practical effectiveness, with a 35\% reduction in packing collapse compared to baseline methods.
>
---
#### [new 004] CoC-VLA: Delving into Adversarial Domain Transfer for Explainable Autonomous Driving via Chain-of-Causality Visual-Language-Action Model
- **分类: cs.RO**

- **简介: 该论文针对自动驾驶中长尾场景泛化能力不足的问题，提出CoC-VLA框架，通过对抗迁移学习，将仿真数据中训练的复杂推理能力迁移到真实场景。利用链式因果视觉语言模型实现端到端推理，结合教师-学生架构与判别器，提升系统在罕见场景下的可解释性与性能。**

- **链接: [https://arxiv.org/pdf/2511.19914v1](https://arxiv.org/pdf/2511.19914v1)**

> **作者:** Dapeng Zhang; Fei Shen; Rui Zhao; Yinda Chen; Peng Zhi; Chenyang Li; Rui Zhou; Qingguo Zhou
>
> **摘要:** Autonomous driving represents a prominent application of artificial intelligence. Recent approaches have shifted from focusing solely on common scenarios to addressing complex, long-tail situations such as subtle human behaviors, traffic accidents, and non-compliant driving patterns. Given the demonstrated capabilities of large language models (LLMs) in understanding visual and natural language inputs and following instructions, recent methods have integrated LLMs into autonomous driving systems to enhance reasoning, interpretability, and performance across diverse scenarios. However, existing methods typically rely either on real-world data, which is suitable for industrial deployment, or on simulation data tailored to rare or hard case scenarios. Few approaches effectively integrate the complementary advantages of both data sources. To address this limitation, we propose a novel VLM-guided, end-to-end adversarial transfer framework for autonomous driving that transfers long-tail handling capabilities from simulation to real-world deployment, named CoC-VLA. The framework comprises a teacher VLM model, a student VLM model, and a discriminator. Both the teacher and student VLM models utilize a shared base architecture, termed the Chain-of-Causality Visual-Language Model (CoC VLM), which integrates temporal information via an end-to-end text adapter. This architecture supports chain-of-thought reasoning to infer complex driving logic. The teacher and student VLM models are pre-trained separately on simulated and real-world datasets. The discriminator is trained adversarially to facilitate the transfer of long-tail handling capabilities from simulated to real-world environments by the student VLM model, using a novel backpropagation strategy.
>
---
#### [new 005] Multi-Agent gatekeeper: Safe Flight Planning and Formation Control for Urban Air Mobility
- **分类: cs.RO**

- **简介: 该论文针对城市空中交通中多无人机编队飞行的安全问题，提出多智能体门卫框架。通过预计算安全轨迹备份集，实现领导者与跟随者在复杂三维环境中的可证明安全编队控制，解决在线规划缺乏安全性保障、离线规划适应性差的矛盾，验证了100%避障成功率及物理可行性。**

- **链接: [https://arxiv.org/pdf/2511.19691v1](https://arxiv.org/pdf/2511.19691v1)**

> **作者:** Thomas Marshall Vielmetti; Devansh R Agrawal; Dimitra Panagou
>
> **备注:** 13 pages, 4 figures, to appear AIAA SciTech 2026
>
> **摘要:** We present Multi-Agent gatekeeper, a framework that provides provable safety guarantees for leader-follower formation control in cluttered 3D environments. Existing methods face a trad-off: online planners and controllers lack formal safety guarantees, while offline planners lack adaptability to changes in the number of agents or desired formation. To address this gap, we propose a hybrid architecture where a single leader tracks a pre-computed, safe trajectory, which serves as a shared trajectory backup set for all follower agents. Followers execute a nominal formation-keeping tracking controller, and are guaranteed to remain safe by always possessing a known-safe backup maneuver along the leader's path. We formally prove this method ensures collision avoidance with both static obstacles and other agents. The primary contributions are: (1) the multi-agent gatekeeper algorithm, which extends our single-agent gatekeeper framework to multi-agent systems; (2) the trajectory backup set for provably safe inter-agent coordination for leader-follower formation control; and (3) the first application of the gatekeeper framework in a 3D environment. We demonstrate our approach in a simulated 3D urban environment, where it achieved a 100% collision-avoidance success rate across 100 randomized trials, significantly outperforming baseline CBF and NMPC methods. Finally, we demonstrate the physical feasibility of the resulting trajectories on a team of quadcopters.
>
---
#### [new 006] Metric, inertially aligned monocular state estimation via kinetodynamic priors
- **分类: cs.RO**

- **简介: 该论文针对柔性机器人系统中非刚体状态估计难题，提出基于动力学先验的单目视觉里程计方法。通过学习弹性变形模型与连续时间B样条运动建模，建立视觉加速度与形变加速度的物理关联，实现高精度位姿估计，并有效解决单目系统中尺度与重力恢复的病态问题。**

- **链接: [https://arxiv.org/pdf/2511.20496v1](https://arxiv.org/pdf/2511.20496v1)**

> **作者:** Jiaxin Liu; Min Li; Wanting Xu; Liang Li; Jiaqi Yang; Laurent Kneip
>
> **摘要:** Accurate state estimation for flexible robotic systems poses significant challenges, particular for platforms with dynamically deforming structures that invalidate rigid-body assumptions. This paper tackles this problem and allows to extend existing rigid-body pose estimation methods to non-rigid systems. Our approach hinges on two core assumptions: first, the elastic properties are captured by an injective deformation-force model, efficiently learned via a Multi-Layer Perceptron; second, we solve the platform's inherently smooth motion using continuous-time B-spline kinematic models. By continuously applying Newton's Second Law, our method establishes a physical link between visually-derived trajectory acceleration and predicted deformation-induced acceleration. We demonstrate that our approach not only enables robust and accurate pose estimation on non-rigid platforms, but that the properly modeled platform physics instigate inertial sensing properties. We demonstrate this feasibility on a simple spring-camera system, and show how it robustly resolves the typically ill-posed problem of metric scale and gravity recovery in monocular visual odometry.
>
---
#### [new 007] ShapeForce: Low-Cost Soft Robotic Wrist for Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文针对接触丰富操作中六轴力矩传感器成本高、易损的问题，提出低成本软腕装置ShapeForce。通过感知柔性核心形变，利用标记点姿态追踪获取力/力矩变化信号，无需校准与专用电子设备，实现高效接触反馈，显著降低硬件成本，性能接近传统传感器，适用于多种复杂抓取任务。**

- **链接: [https://arxiv.org/pdf/2511.19955v1](https://arxiv.org/pdf/2511.19955v1)**

> **作者:** Jinxuan Zhu; Zihao Yan; Yangyu Xiao; Jingxiang Guo; Chenrui Tie; Xinyi Cao; Yuhang Zheng; Lin Shao
>
> **摘要:** Contact feedback is essential for contact-rich robotic manipulation, as it allows the robot to detect subtle interaction changes and adjust its actions accordingly. Six-axis force-torque sensors are commonly used to obtain contact feedback, but their high cost and fragility have discouraged many researchers from adopting them in contact-rich tasks. To offer a more cost-efficient and easy-accessible source of contact feedback, we present ShapeForce, a low-cost, plug-and-play soft wrist that provides force-like signals for contact-rich robotic manipulation. Inspired by how humans rely on relative force changes in contact rather than precise force magnitudes, ShapeForce converts external force and torque into measurable deformations of its compliant core, which are then estimated via marker-based pose tracking and converted into force-like signals. Our design eliminates the need for calibration or specialized electronics to obtain exact values, and instead focuses on capturing force and torque changes sufficient for enabling contact-rich manipulation. Extensive experiments across diverse contact-rich tasks and manipulation policies demonstrate that ShapeForce delivers performance comparable to six-axis force-torque sensors at an extremely low cost.
>
---
#### [new 008] Human-Centered Cooperative Control Coupling Autonomous and Haptic Shared Control via Control Barrier Function
- **分类: cs.RO**

- **简介: 该论文针对遥操作中因操纵杆与人体动力学影响自主性能的问题，提出一种耦合无操纵杆自主控制与触觉共享控制的协作框架。通过控制屏障函数实时识别安全区域，仅在非安全区启用触觉反馈，提升操作精度与效率。**

- **链接: [https://arxiv.org/pdf/2511.19869v1](https://arxiv.org/pdf/2511.19869v1)**

> **作者:** Eito Sato; Takahiro Wada
>
> **摘要:** Haptic shared control (HSC) is effective in teleoperation when full autonomy is limited by uncertainty or sensing constraints. However, autonomous control performance achieved by maximizing HSC strength is limited because the dynamics of the joystick and human arm affect the robot's behavior. We propose a cooperative framework coupling a joystick-independent autonomous controller with HSC. A control barrier function ignores joystick inputs within a safe region determined by the human operator in real-time, while HSC is engaged otherwise. A pilot experiment on simulated tasks with tele-operated underwater robot in virtual environment demonstrated improved accuracy and reduced required time over conventional HSC.
>
---
#### [new 009] Kleinkram: Open Robotic Data Management
- **分类: cs.RO; cs.IR**

- **简介: 该论文提出Kleinkram，一个开源的本地化机器人数据管理平台，旨在解决海量非结构化机器人数据的存储、索引与共享难题。系统支持ROS bags、MCAP等格式，集成Docker工作流实现数据验证与分析，已管理超30TB数据，通过Web与CLI提升研究效率。**

- **链接: [https://arxiv.org/pdf/2511.20492v1](https://arxiv.org/pdf/2511.20492v1)**

> **作者:** Cyrill Püntener; Johann Schwabe; Dominique Garmier; Jonas Frey; Marco Hutter
>
> **备注:** for associated source code, see https://github.com/leggedrobotics/kleinkram
>
> **摘要:** We introduce Kleinkram, a free and open-source system designed to solve the challenge of managing massive, unstructured robotic datasets. Designed as a modular, on-premises cloud solution, Kleinkram enables scalable storage, indexing, and sharing of datasets, ranging from individual experiments to large-scale research collections. Kleinkram natively integrates with standard formats such as ROS bags and MCAP and utilises S3-compatible storage for flexibility. Beyond storage, Kleinkram features an integrated "Action Runner" that executes customizable Docker-based workflows for data validation, curation, and benchmarking. Kleinkram has successfully managed over 30 TB of data from diverse robotic systems, streamlining the research lifecycle through a modern web interface and a robust Command Line Interface (CLI).
>
---
#### [new 010] Active3D: Active High-Fidelity 3D Reconstruction via Hierarchical Uncertainty Quantification
- **分类: cs.RO**

- **简介: 该论文针对高保真3D重建中的不确定性与视点规划问题，提出Active3D框架。通过融合神经场与高斯原语的混合表示，构建分层不确定性体积，实现全局结构与局部细节的联合建模。基于不确定性驱动的视点选择与关键帧策略，优化重建效率与精度，显著提升真实场景下机器人感知的重建质量。**

- **链接: [https://arxiv.org/pdf/2511.20050v1](https://arxiv.org/pdf/2511.20050v1)**

> **作者:** Yan Li; Yingzhao Li; Gim Hee Lee
>
> **摘要:** In this paper, we present an active exploration framework for high-fidelity 3D reconstruction that incrementally builds a multi-level uncertainty space and selects next-best-views through an uncertainty-driven motion planner. We introduce a hybrid implicit-explicit representation that fuses neural fields with Gaussian primitives to jointly capture global structural priors and locally observed details. Based on this hybrid state, we derive a hierarchical uncertainty volume that quantifies both implicit global structure quality and explicit local surface confidence. To focus optimization on the most informative regions, we propose an uncertainty-driven keyframe selection strategy that anchors high-entropy viewpoints as sparse attention nodes, coupled with a viewpoint-space sliding window for uncertainty-aware local refinement. The planning module formulates next-best-view selection as an Expected Hybrid Information Gain problem and incorporates a risk-sensitive path planner to ensure efficient and safe exploration. Extensive experiments on challenging benchmarks demonstrate that our approach consistently achieves state-of-the-art accuracy, completeness, and rendering quality, highlighting its effectiveness for real-world active reconstruction and robotic perception tasks.
>
---
#### [new 011] Discover, Learn, and Reinforce: Scaling Vision-Language-Action Pretraining with Diverse RL-Generated Trajectories
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对视觉-语言-动作（VLA）模型预训练数据稀缺问题，提出DLR框架，通过信息论方法发现多种高成功率的机器人操作策略。解决了标准强化学习生成数据单一的问题，显著提升轨迹多样性与覆盖范围，实现更优的下游任务表现和可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.19528v1](https://arxiv.org/pdf/2511.19528v1)**

> **作者:** Rushuai Yang; Zhiyuan Feng; Tianxiang Zhang; Kaixin Wang; Chuheng Zhang; Li Zhao; Xiu Su; Yi Chen; Jiang Bian
>
> **摘要:** Scaling vision-language-action (VLA) model pre-training requires large volumes of diverse, high-quality manipulation trajectories. Most current data is obtained via human teleoperation, which is expensive and difficult to scale. Reinforcement learning (RL) methods learn useful skills through autonomous exploration, making them a viable approach for generating data. However, standard RL training collapses to a narrow execution pattern, limiting its utility for large-scale pre-training. We propose Discover, Lea rn and Reinforce (DLR), an information-theoretic pattern discovery framework that generates multiple distinct, high-success behavioral patterns for VLA pretraining. Empirically, DLR generates a markedly more diverse trajectory corpus on LIBERO. Specifically, it learns multiple distinct, high-success strategies for the same task where standard RL discovers only one, and hence it covers substantially broader regions of the state-action space. When adapted to unseen downstream task suites, VLA models pretrained on our diverse RL data surpass counterparts trained on equal-sized standard RL datasets. Moreover, DLR exhibits positive data-scaling behavior that single-pattern RL lacks. These results position multi-pattern RL as a practical, scalable data engine for embodied foundation models.
>
---
#### [new 012] Reinforcing Action Policies by Prophesying
- **分类: cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）政策在模仿学习中过拟合、泛化能力差的问题，提出ProphRL框架。通过预训练的世界模型Prophet和专为流形动作头设计的强化学习方法FA-GRPO与FlowScale，实现高效、稳定的后训练优化，在多个基准和真实机器人上显著提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2511.20633v1](https://arxiv.org/pdf/2511.20633v1)**

> **作者:** Jiahui Zhang; Ze Huang; Chun Gu; Zipei Ma; Li Zhang
>
> **备注:** https://LogosRoboticsGroup.github.io/ProphRL
>
> **摘要:** Vision-Language-Action (VLA) policies excel in aligning language, perception, and robot control. However, most VLAs are trained purely by imitation, which overfits to demonstrations, and is brittle under distribution shift. Reinforcement learning (RL) directly optimizes task reward and thus addresses this misalignment, but real-robot interaction is expensive and conventional simulators are hard to engineer and transfer. We address both data efficiency and optimization stability in VLA post-training via a learned world model and an RL procedure tailored to flow-based action heads. Specifically, we introduce Prophet, a unified action-to-video robot actuation pretrained across large-scale, heterogeneous robot data to learn reusable action-outcome dynamics. It is able to few-shot adapt to new robots, objects, and environments, yielding a rollout-ready simulator. Upon Prophet, we reinforce action policies with Flow-action-GRPO (FA-GRPO), which adapts Flow-GRPO to operate on VLA actions, and with FlowScale, a stepwise reweighting that rescales per-step gradients in the flow head. Together, Prophet, FA-GRPO, and FlowScale constitute ProphRL, a practical, data- and compute-efficient path to VLA post-training. Experiments show 5-17% success gains on public benchmarks and 24-30% gains on real robots across different VLA variants.
>
---
#### [new 013] Improved adaptive wind driven optimization algorithm for real-time path planning
- **分类: cs.RO**

- **简介: 该论文针对动态环境下机器人实时路径规划中适应性差、易早熟等问题，提出多层级自适应风驱动优化算法（MAWDO）。通过分层引导机制平衡探索与利用，显著提升搜索精度与稳定性，实现在复杂环境中的高效、平滑、无碰撞路径生成。**

- **链接: [https://arxiv.org/pdf/2511.20394v1](https://arxiv.org/pdf/2511.20394v1)**

> **作者:** Shiqian Liu; Azlan Mohd Zain; Le-le Mao
>
> **备注:** 23 pages, 4 figures
>
> **摘要:** Recently, path planning has achieved remarkable progress in enhancing global search capability and convergence accuracy through heuristic and learning-inspired optimization frameworks. However, real-time adaptability in dynamic environments remains a critical challenge for autonomous navigation, particularly when robots must generate collision-free, smooth, and efficient trajectories under complex constraints. By analyzing the difficulties in dynamic path planning, the Wind Driven Optimization (WDO) algorithm emerges as a promising framework owing to its physically interpretable search dynamics. Motivated by these observations, this work revisits the WDO principle and proposes a variant formulation, Multi-hierarchical adaptive wind driven optimization(MAWDO), that improves adaptability and robustness in time-varying environments. To mitigate instability and premature convergence, a hierarchical-guidance mechanism divides the population into multiple groups guided by individual, regional, and global leaders to balance exploration and exploitation. Extensive evaluations on sixteen benchmark functions show that MAWDO achieves superior optimization accuracy, convergence stability, and adaptability over state-of-the art metaheuristics. In dynamic path planning, MAWDO shortens the path length to 469.28 pixels, improving over Multi-strategy ensemble wind driven optimization(MEWDO), Adaptive wind driven optimization(AWDO) and WDO by 3.51\%, 11.63\% and 14.93\%, and achieves the smallest optimality gap (1.01) with smoothness 0.71 versus 13.50 and 15.67 for AWDO and WDO, leading to smoother, shorter, and collision-free trajectories that confirm its effectiveness for real-time path planning in complex environments.
>
---
#### [new 014] HAFO: Humanoid Force-Adaptive Control for Intense External Force Interaction Environments
- **分类: cs.RO**

- **简介: 该论文针对人形机器人在强外力干扰下难以实现稳定运动与精准操作的问题，提出HAFO框架。通过双智能体强化学习，耦合优化步态与上肢操控策略，利用弹簧阻尼模型建模外力并实现精细力控，使机器人在绳索拉力等强干扰下仍能保持稳定，显著提升抗干扰能力与任务执行精度。**

- **链接: [https://arxiv.org/pdf/2511.20275v1](https://arxiv.org/pdf/2511.20275v1)**

> **作者:** Chenhui Dong; Haozhe Xu; Wenhao Feng; Zhipeng Wang; Yanmin Zhou; Yifei Zhao; Bin He
>
> **摘要:** Reinforcement learning controllers have made impressive progress in humanoid locomotion and light load manipulation. However, achieving robust and precise motion with strong force interaction remains a significant challenge. Based on the above limitations, this paper proposes HAFO, a dual-agent reinforcement learning control framework that simultaneously optimizes both a robust locomotion strategy and a precise upper-body manipulation strategy through coupled training under external force interaction environments. Simultaneously, we explicitly model the external pulling disturbances through a spring-damper system and achieve fine-grained force control by manipulating the virtual spring. During this process, the reinforcement-learning policy spontaneously generates disturbance-rejection response by exploiting environmental feedback. Moreover, HAFO employs an asymmetric Actor-Critic framework in which the Critic-network access to privileged spring-damping forces guides the actor-network to learn a generalizable, robust policy for resisting external disturbances. The experimental results demonstrate that HAFO achieves stable control of humanoid robot under various strong force interactions, showing remarkable performance in load tasks and ensuring stable robot operation under rope tension disturbances. Project website: hafo-robot.github.io.
>
---
#### [new 015] ArtiBench and ArtiBrain: Benchmarking Generalizable Vision-Language Articulated Object Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对视觉语言引导的可动物体操作中的泛化难题，提出ArtiBench基准与ArtiBrain框架。通过多场景、多层次任务评估，揭示了跨部件、跨实例的挑战；ArtiBrain融合高层推理与自适应控制，利用视觉语言模型分解任务，结合几何关键帧与扩散模型实现精准、可解释的操作，并通过可行动能记忆库提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.20330v1](https://arxiv.org/pdf/2511.20330v1)**

> **作者:** Yuhan Wu; Tiantian Wei; Shuo Wang; ZhiChao Wang; Yanyong Zhang; Daniel Cremers; Yan Xia
>
> **摘要:** Interactive articulated manipulation requires long-horizon, multi-step interactions with appliances while maintaining physical consistency. Existing vision-language and diffusion-based policies struggle to generalize across parts, instances, and categories. We first introduce ArtiBench, a five-level benchmark covering kitchen, storage, office, and tool environments. ArtiBench enables structured evaluation from cross-part and cross-instance variation to long-horizon multi-object tasks, revealing the core generalization challenges of articulated object manipulation. Building on this benchmark, we propose ArtiBrain, a modular framework that unifies high-level reasoning with adaptive low-level control. ArtiBrain uses a VLM-based Task Reasoner (GPT-4.1) to decompose and validate subgoals, and employs a Hybrid Controller that combines geometry-aware keyframe execution with affordance-guided diffusion for precise and interpretable manipulation. An Affordance Memory Bank continually accumulates successful execution episodes and propagates part-level actionable affordances to unseen articulated parts and configurations. Extensive experiments on ArtiBench show that our ArtiBrain significantly outperforms state-of-the-art multimodal and diffusion-based methods in robustness and generalization. Code and dataset will be released upon acceptance.
>
---
#### [new 016] Development of a Testbed for Autonomous Vehicles: Integrating MPC Control with Monocular Camera Lane Detection
- **分类: cs.RO**

- **简介: 该论文针对自动驾驶车辆的轨迹跟踪问题，提出将单目相机车道检测与模型预测控制（MPC）结合的方法。通过边缘检测、滑动窗口和动态ROI提取车道线，基于自行车模型构建MPC控制器，在ROS Gazebo仿真中验证，使轨迹跟踪均方根误差降低27.65%，提升了精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2511.19655v1](https://arxiv.org/pdf/2511.19655v1)**

> **作者:** Shantanu Rahman; Nayeb Hasin; Mainul Islam
>
> **备注:** 49 pages, 23 figures
>
> **摘要:** Autonomous vehicles are becoming popular day by day not only for autonomous road traversal but also for industrial automation, farming and military. Most of the standard vehicles follow the Ackermann style steering mechanism. This has become to de facto standard for large and long faring vehicles. The local planner of an autonomous vehicle controls the low-level vehicle movement upon which the vehicle will perform its motor actuation. In our work, we focus on autonomous vehicles in road and perform experiments to analyze the effect of low-level controllers in the simulation and a real environment. To increase the precision and stability of trajectory tracking in autonomous cars, a novel method that combines lane identification with Model Predictive Control (MPC) is presented. The research focuses on camera-equipped autonomous vehicles and uses methods like edge recognition, sliding window-based straight-line identification for lane line extraction, and dynamic region of interest (ROI) extraction. Next, to follow the identified lane line, an MPC built on a bicycle vehicle dynamics model is created. A single-lane road simulation model is built using ROS Gazebo and tested in order to verify the controller's performance. The root mean square error between the optimal tracking trajectory and the target trajectory was reduced by 27.65% in the simulation results, demonstrating the high robustness and flexibility of the developed controller.
>
---
#### [new 017] Unifying Perception and Action: A Hybrid-Modality Pipeline with Implicit Visual Chain-of-Thought for Robotic Action Generation
- **分类: cs.RO**

- **简介: 该论文针对机器人动作生成中视觉与动作模态脱节及训练不稳的问题，提出VITA框架，通过隐式视觉思维链在共享离散潜空间中联合建模感知与控制，实现视觉预测与动作生成的协同优化，显著提升复杂环境下的任务成功率。**

- **链接: [https://arxiv.org/pdf/2511.19859v1](https://arxiv.org/pdf/2511.19859v1)**

> **作者:** Xiangkai Ma; Lekai Xing; Han Zhang; Wenzhong Li; Sanglu Lu
>
> **摘要:** Vision-Language-Action (VLA) models built upon Chain-of-Thought (CoT) have achieved remarkable success in advancing general-purpose robotic agents, owing to its significant perceptual comprehension. Recently, since text-only CoT struggles to adequately capture scene details in complex spatial environments, a highly promising strategy involves leveraging visual priors to guide robotic action generation. Nevertheless, these strategies face two inherent challenges: (i) a modality gap between visual observations and low-level actions, and (ii) unstable training due to competing objectives between visual prediction and action generation. To address these challenges, we propose a Vision-Integrated Trajectory Alignment (VITA) framework that learns a shared discrete latent space for vision and action, enabling joint modeling of perception and motor control. VITA introduces a implicit visual CoT: autoregressively generated tokens is simultaneously decoded into future frames predictions and robot actions, thereby internalizing visual dynamics as an inductive bias for motion planning. Extensive experiments on simulated and real-world environments demonstrate state-of-the-art performance. VITA improves 14.5\%, 9.6\% and 12.1\% over existing baselines on CALVIN, LIBERO and SimplerEnv. Furthermore, VITA attains an average success rate of 80.5\% across six real-world tasks, demonstrating its potential as a generalist robotic manipulation model.
>
---
#### [new 018] Online Learning-Enhanced High Order Adaptive Safety Control
- **分类: cs.RO**

- **简介: 该论文针对复杂动态系统中控制屏障函数（CBF）因模型误差导致安全失效的问题，提出一种基于神经微分方程的在线学习增强高阶自适应CBF方法。通过实时学习动态扰动，提升系统在风扰等不确定性下的安全性能，已在38g纳米四旋翼上验证，有效保持与障碍物的安全距离。**

- **链接: [https://arxiv.org/pdf/2511.19651v1](https://arxiv.org/pdf/2511.19651v1)**

> **作者:** Lishuo Pan; Mattia Catellani; Thales C. Silva; Lorenzo Sabattini; Nora Ayanian
>
> **备注:** 8 pages, 7 figures, submitted to RA-L
>
> **摘要:** Control barrier functions (CBFs) are an effective model-based tool to formally certify the safety of a system. With the growing complexity of modern control problems, CBFs have received increasing attention in both optimization-based and learning-based control communities as a safety filter, owing to their provable guarantees. However, success in transferring these guarantees to real-world systems is critically tied to model accuracy. For example, payloads or wind disturbances can significantly influence the dynamics of an aerial vehicle and invalidate the safety guarantee. In this work, we propose an efficient yet flexible online learning-enhanced high-order adaptive control barrier function using Neural ODEs. Our approach improves the safety of a CBF-certified system on the fly, even under complex time-varying model perturbations. In particular, we deploy our hybrid adaptive CBF controller on a 38g nano quadrotor, keeping a safe distance from the obstacle, against 18km/h wind.
>
---
#### [new 019] Robot-Powered Data Flywheels: Deploying Robots in the Wild for Continual Data Collection and Foundation Model Adaptation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出机器人驱动的数据飞轮框架，解决基础模型在真实世界中因训练数据不足而表现脆弱的问题。通过部署机器人自主采集真实场景数据，实现模型持续优化与任务效率提升。实验中机器人扫描图书馆书架，显著提升视觉语言模型的书籍识别与多语言OCR性能，同时节省人力。**

- **链接: [https://arxiv.org/pdf/2511.19647v1](https://arxiv.org/pdf/2511.19647v1)**

> **作者:** Jennifer Grannen; Michelle Pan; Kenneth Llontop; Cherie Ho; Mark Zolotas; Jeannette Bohg; Dorsa Sadigh
>
> **摘要:** Foundation models (FM) have unlocked powerful zero-shot capabilities in vision and language, yet their reliance on internet pretraining data leaves them brittle in unstructured, real-world settings. The messy, real-world data encountered during deployment (e.g. occluded or multilingual text) remains massively underrepresented in existing corpora. Robots, as embodied agents, are uniquely positioned to close this gap: they can act in physical environments to collect large-scale, real-world data that enriches FM training with precisely the examples current models lack. We introduce the Robot-Powered Data Flywheel, a framework that transforms robots from FM consumers into data generators. By deploying robots equipped with FMs in the wild, we enable a virtuous cycle: robots perform useful tasks while collecting real-world data that improves both domain-specific adaptation and domain-adjacent generalization. We instantiate this framework with Scanford, a mobile manipulator deployed in the East Asia Library for 2 weeks. Scanford autonomously scans shelves, identifies books using a vision-language model (VLM), and leverages the library catalog to label images without human annotation. This deployment both aids librarians and produces a dataset to finetune the underlying VLM, improving performance on the domain-specific in-the-wild library setting and on domain-adjacent multilingual OCR benchmarks. Using data collected from 2103 shelves, Scanford improves VLM performance on book identification from 32.0% to 71.8% and boosts domain-adjacent multilingual OCR from 24.8% to 46.6% (English) and 30.8% to 38.0% (Chinese), while saving an ~18.7 hrs of human time. These results highlight how robot-powered data flywheels can both reduce human effort in real deployments and unlock new pathways for continually adapting FMs to the messiness of reality. More details are at: https://scanford-robot.github.io
>
---
#### [new 020] Gated Uncertainty-Aware Runtime Dual Invariants for Neural Signal-Controlled Robotics
- **分类: cs.RO; cs.AI; cs.HC; cs.LG**

- **简介: 该论文针对神经信号控制机器人中的安全与可信问题，提出GUARDIAN框架，通过双层运行时监控与不确定性感知机制，在低精度解码下实现94-97%高安全率，支持实时验证与可审计追踪，解决神经信号控制系统的可靠性与信任难题。**

- **链接: [https://arxiv.org/pdf/2511.20570v1](https://arxiv.org/pdf/2511.20570v1)**

> **作者:** Tasha Kim; Oiwi Parker Jones
>
> **备注:** Embodied and Safe-Assured Robotic Systems workshop at NeurIPS 2025
>
> **摘要:** Safety-critical assistive systems that directly decode user intent from neural signals require rigorous guarantees of reliability and trust. We present GUARDIAN (Gated Uncertainty-Aware Runtime Dual Invariants), a framework for real-time neuro-symbolic verification for neural signal-controlled robotics. GUARDIAN enforces both logical safety and physiological trust by coupling confidence-calibrated brain signal decoding with symbolic goal grounding and dual-layer runtime monitoring. On the BNCI2014 motor imagery electroencephalogram (EEG) dataset with 9 subjects and 5,184 trials, the system performs at a high safety rate of 94-97% even with lightweight decoder architectures with low test accuracies (27-46%) and high ECE confidence miscalibration (0.22-0.41). We demonstrate 1.7x correct interventions in simulated noise testing versus at baseline. The monitor operates at 100Hz and sub-millisecond decision latency, making it practically viable for closed-loop neural signal-based systems. Across 21 ablation results, GUARDIAN exhibits a graduated response to signal degradation, and produces auditable traces from intent, plan to action, helping to link neural evidence to verifiable robot action.
>
---
#### [new 021] Whole-Body Inverse Dynamics MPC for Legged Loco-Manipulation
- **分类: cs.RO**

- **简介: 该论文针对足式机器人在执行抓取、推拉等任务时，需协调运动与力控的挑战，提出一种基于全阶逆动力学的模型预测控制框架。通过统一优化关节力矩，实现运动与力的联合规划与实时执行，在四足机器人上实现实时80Hz控制，成功完成推箱、擦白板等复杂操作。**

- **链接: [https://arxiv.org/pdf/2511.19709v1](https://arxiv.org/pdf/2511.19709v1)**

> **作者:** Lukas Molnar; Jin Cheng; Gabriele Fadini; Dongho Kang; Fatemeh Zargarbashi; Stelian Coros
>
> **备注:** 9 pages, 6 figures, to be published in IEEE Robotics and Automation Letters (Special Issue: Advancements in MPC and Learning Algorithms for Legged Robots)
>
> **摘要:** Loco-manipulation demands coordinated whole-body motion to manipulate objects effectively while maintaining locomotion stability, presenting significant challenges for both planning and control. In this work, we propose a whole-body model predictive control (MPC) framework that directly optimizes joint torques through full-order inverse dynamics, enabling unified motion and force planning and execution within a single predictive layer. This approach allows emergent, physically consistent whole-body behaviors that account for the system's dynamics and physical constraints. We implement our MPC formulation using open software frameworks (Pinocchio and CasADi), along with the state-of-the-art interior-point solver Fatrop. In real-world experiments on a Unitree B2 quadruped equipped with a Unitree Z1 manipulator arm, our MPC formulation achieves real-time performance at 80 Hz. We demonstrate loco-manipulation tasks that demand fine control over the end-effector's position and force to perform real-world interactions like pulling heavy loads, pushing boxes, and wiping whiteboards.
>
---
#### [new 022] A Virtual Mechanical Interaction Layer Enables Resilient Human-to-Robot Object Handovers
- **分类: cs.RO**

- **简介: 该论文研究人机物体交接任务，针对交接过程中物体姿态变化带来的不确定性问题，提出虚拟模型控制与增强现实结合的交互层，实现机器人动作的自适应与双向沟通。实验与用户研究验证了方法的鲁棒性与用户偏好。**

- **链接: [https://arxiv.org/pdf/2511.19543v1](https://arxiv.org/pdf/2511.19543v1)**

> **作者:** Omar Faris; Sławomir Tadeja; Fulvio Forni
>
> **摘要:** Object handover is a common form of interaction that is widely present in collaborative tasks. However, achieving it efficiently remains a challenge. We address the problem of ensuring resilient robotic actions that can adapt to complex changes in object pose during human-to-robot object handovers. We propose the use of Virtual Model Control to create an interaction layer that controls the robot and adapts to the dynamic changes in the handover process. Additionally, we propose the use of augmented reality to facilitate bidirectional communication between humans and robots during handovers. We assess the performance of our controller in a set of experiments that demonstrate its resilience to various sources of uncertainties, including complex changes to the object's pose during the handover. Finally, we performed a user study with 16 participants to understand human preferences for different robot control profiles and augmented reality visuals in object handovers. Our results showed a general preference for the proposed approach and revealed insights that can guide further development in adapting the interaction with the user.
>
---
#### [new 023] Hibikino-Musashi@Home 2025 Team Description Paper
- **分类: cs.RO**

- **简介: 该论文针对家庭服务机器人在复杂环境中的个性化适应问题，提出基于大模型的任务规划与脑启发记忆模型，构建开源仿真开发环境及数据生成工具，提升导航与任务执行的可复用性与智能化水平。**

- **链接: [https://arxiv.org/pdf/2511.20180v1](https://arxiv.org/pdf/2511.20180v1)**

> **作者:** Ryohei Kobayashi; Kosei Isomoto; Kosei Yamao; Soma Fumoto; Koshun Arimura; Naoki Yamaguchi; Akinobu Mizutani; Tomoya Shiba; Kouki Kimizuka; Yuta Ohno; Ryo Terashima; Hiromasa Yamaguchi; Tomoaki Fujino; Ryoga Maruno; Wataru Yoshimura; Kazuhito Mine; Tang Phu Thien Nhan; Yuga Yano; Yuichiro Tanaka; Takeshi Nishida; Takashi Morie; Hakaru Tamukoh
>
> **摘要:** This paper provides an overview of the techniques employed by Hibikino-Musashi@Home, which intends to participate in the domestic standard platform league. The team developed a dataset generator for training a robot vision system and an open-source development environment running on a Human Support Robot simulator. The large-language-model-powered task planner selects appropriate primitive skills to perform the task requested by the user. Moreover, the team has focused on research involving brain-inspired memory models for adaptation to individual home environments. This approach aims to provide intuitive and personalized assistance. Additionally, the team contributed to the reusability of the navigation system developed by Pumas in RoboCup2024. The team aimed to design a home service robot to assist humans in their homes and continuously attend competitions to evaluate and improve the developed system.
>
---
#### [new 024] Safe and Stable Neural Network Dynamical Systems for Robot Motion Planning
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对机器人运动规划中从示范学习安全稳定轨迹的挑战，提出S²-NNDS框架。通过联合学习神经动力系统与神经李雅普诺夫/屏障函数，实现对复杂非线性运动的建模，并利用分拆共形预测提供概率安全保证。在2D/3D数据集上验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2511.20593v1](https://arxiv.org/pdf/2511.20593v1)**

> **作者:** Allen Emmanuel Binny; Mahathi Anand; Hugo T. M. Kussaba; Lingyun Chen; Shreenabh Agrawal; Fares J. Abu-Dakka; Abdalla Swikir
>
> **摘要:** Learning safe and stable robot motions from demonstrations remains a challenge, especially in complex, nonlinear tasks involving dynamic, obstacle-rich environments. In this paper, we propose Safe and Stable Neural Network Dynamical Systems S$^2$-NNDS, a learning-from-demonstration framework that simultaneously learns expressive neural dynamical systems alongside neural Lyapunov stability and barrier safety certificates. Unlike traditional approaches with restrictive polynomial parameterizations, S$^2$-NNDS leverages neural networks to capture complex robot motions providing probabilistic guarantees through split conformal prediction in learned certificates. Experimental results on various 2D and 3D datasets -- including LASA handwriting and demonstrations recorded kinesthetically from the Franka Emika Panda robot -- validate S$^2$-NNDS effectiveness in learning robust, safe, and stable motions from potentially unsafe demonstrations.
>
---
#### [new 025] Flow-Based Path Planning for Multiple Homogenous UAVs for Outdoor Formation-Flying
- **分类: cs.RO**

- **简介: 该论文研究多无人机编队飞行中的无碰撞路径规划问题。针对同质无人机群，提出基于流网络的路径规划方法：构建GPS坐标流图，用图算法求最短路径，并通过福特-福克森法确保最大流（无碰撞）。通过仿真与实验验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2511.19653v1](https://arxiv.org/pdf/2511.19653v1)**

> **作者:** Mahmud Suhaimi Ibrahim; Shantanu Rahman; Muhammad Samin Hasan; Minhaj Uddin Ahmad; Abdullah Abrar
>
> **备注:** 9 pages, 15 figures, conference
>
> **摘要:** Collision-free path planning is the most crucial component in multi-UAV formation-flying (MFF). We use unlabeled homogenous quadcopters (UAVs) to demonstrate the use of a flow network to create complete (inter-UAV) collision-free paths. This procedure has three main parts: 1) Creating a flow network graph from physical GPS coordinates, 2) Finding a path of minimum cost (least distance) using any graph-based path-finding algorithm, and 3) Implementing the Ford-Fulkerson Method to find the paths with the maximum flow (no collision). Simulations of up to 64 UAVs were conducted for various formations, followed by a practical experiment with 3 quadcopters for testing physical plausibility and feasibility. The results of these tests show the efficacy of this method's ability to produce safe, collision-free paths.
>
---
#### [new 026] Power-Efficient Autonomous Mobile Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对自主移动机器人（AMR）的能效问题，提出pNav系统，通过联合优化软硬件子系统，解决功率消耗波动、环境感知局部性与协同控制难题。工作包括毫秒级功耗预测、实时导航局部性建模及软硬件动态协调，实测功耗降低38.1%且不影响导航性能。**

- **链接: [https://arxiv.org/pdf/2511.20467v1](https://arxiv.org/pdf/2511.20467v1)**

> **作者:** Liangkai Liu; Weisong Shi; Kang G. Shin
>
> **备注:** 13 pages, 16 figures
>
> **摘要:** This paper presents pNav, a novel power-management system that significantly enhances the power/energy-efficiency of Autonomous Mobile Robots (AMRs) by jointly optimizing their physical/mechanical and cyber subsystems. By profiling AMRs' power consumption, we identify three challenges in achieving CPS (cyber-physical system) power-efficiency that involve both cyber (C) and physical (P) subsystems: (1) variabilities of system power consumption breakdown, (2) environment-aware navigation locality, and (3) coordination of C and P subsystems. pNav takes a multi-faceted approach to achieve power-efficiency of AMRs. First, it integrates millisecond-level power consumption prediction for both C and P subsystems. Second, it includes novel real-time modeling and monitoring of spatial and temporal navigation localities for AMRs. Third, it supports dynamic coordination of AMR software (navigation, detection) and hardware (motors, DVFS driver) configurations. pNav is prototyped using the Robot Operating System (ROS) Navigation Stack, 2D LiDAR, and camera. Our in-depth evaluation with a real robot and Gazebo environments demonstrates a >96% accuracy in predicting power consumption and a 38.1% reduction in power consumption without compromising navigation accuracy and safety.
>
---
#### [new 027] Splatblox: Traversability-Aware Gaussian Splatting for Outdoor Robot Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Splatblox，用于户外复杂环境中的机器人自主导航。针对植被密集、障碍物不规则等问题，融合RGB与LiDAR数据，基于高斯点阵构建可通行性感知的欧氏有符号距离场，实现几何与语义联合建模，支持实时路径规划与避障，显著提升导航成功率与效率。**

- **链接: [https://arxiv.org/pdf/2511.18525v1](https://arxiv.org/pdf/2511.18525v1)**

> **作者:** Samarth Chopra; Jing Liang; Gershom Seneviratne; Yonghan Lee; Jaehoon Choi; Jianyu An; Stephen Cheng; Dinesh Manocha
>
> **备注:** Submitted to ICRA 2026
>
> **摘要:** We present Splatblox, a real-time system for autonomous navigation in outdoor environments with dense vegetation, irregular obstacles, and complex terrain. Our method fuses segmented RGB images and LiDAR point clouds using Gaussian Splatting to construct a traversability-aware Euclidean Signed Distance Field (ESDF) that jointly encodes geometry and semantics. Updated online, this field enables semantic reasoning to distinguish traversable vegetation (e.g., tall grass) from rigid obstacles (e.g., trees), while LiDAR ensures 360-degree geometric coverage for extended planning horizons. We validate Splatblox on a quadruped robot and demonstrate transfer to a wheeled platform. In field trials across vegetation-rich scenarios, it outperforms state-of-the-art methods with over 50% higher success rate, 40% fewer freezing incidents, 5% shorter paths, and up to 13% faster time to goal, while supporting long-range missions up to 100 meters. Experiment videos and more details can be found on our project page: https://splatblox.github.io
>
---
#### [new 028] Dynamic-ICP: Doppler-Aware Iterative Closest Point Registration for Dynamic Scenes
- **分类: cs.RO**

- **简介: 该论文针对动态场景中基于ICP的位姿估计失效问题，提出Dynamic-ICP框架。通过融合FMCW LiDAR的多普勒速度信息，实现对动态物体的检测与补偿，提升旋转稳定性和定位精度，无需外部传感器，可实时运行。**

- **链接: [https://arxiv.org/pdf/2511.20292v1](https://arxiv.org/pdf/2511.20292v1)**

> **作者:** Dong Wang; Daniel Casado Herraez; Stefan May; Andreas Nüchter
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Reliable odometry in highly dynamic environments remains challenging when it relies on ICP-based registration: ICP assumes near-static scenes and degrades in repetitive or low-texture geometry. We introduce Dynamic-ICP, a Doppler-aware registration framework. The method (i) estimates ego motion from per-point Doppler velocity via robust regression and builds a velocity filter, (ii) clusters dynamic objects and reconstructs object-wise translational velocities from ego-compensated radial measurements, (iii) predicts dynamic points with a constant-velocity model, and (iv) aligns scans using a compact objective that combines point-to-plane geometry residual with a translation-invariant, rotation-only Doppler residual. The approach requires no external sensors or sensor-vehicle calibration and operates directly on FMCW LiDAR range and Doppler velocities. We evaluate Dynamic-ICP on three datasets-HeRCULES, HeLiPR, AevaScenes-focusing on highly dynamic scenes. Dynamic-ICP consistently improves rotational stability and translation accuracy over the state-of-the-art methods. Our approach is also simple to integrate into existing pipelines, runs in real time, and provides a lightweight solution for robust registration in dynamic environments. To encourage further research, the code is available at: https://github.com/JMUWRobotics/Dynamic-ICP.
>
---
#### [new 029] Toward generic control for soft robotic systems
- **分类: cs.RO**

- **简介: 该论文针对软体机器人控制碎片化问题，提出基于控制柔顺性的通用控制框架。通过借鉴人类运动控制原理，将柔顺性从干扰转为可利用特性，实现跨形态、跨驱动方式的稳定、安全与可迁移控制，推动软体机器人统一控制方法的发展。**

- **链接: [https://arxiv.org/pdf/2511.20226v1](https://arxiv.org/pdf/2511.20226v1)**

> **作者:** Yu Sun; Yaosheng Deng; Wenjie Mei; Xiaogang Xiong; Yang Bai; Masaki Ogura; Zeyu Zhou; Mir Feroskhan; Michael Yu Wang; Qiyang Zuo; Yao Li; Yunjiang Lou
>
> **摘要:** Soft robotics has advanced rapidly, yet its control methods remain fragmented: different morphologies and actuation schemes still require task-specific controllers, hindering theoretical integration and large-scale deployment. A generic control framework is therefore essential, and a key obstacle lies in the persistent use of rigid-body control logic, which relies on precise models and strict low-level execution. Such a paradigm is effective for rigid robots but fails for soft robots, where the ability to tolerate and exploit approximate action representations, i.e., control compliance, is the basis of robustness and adaptability rather than a disturbance to be eliminated. Control should thus shift from suppressing compliance to explicitly exploiting it. Human motor control exemplifies this principle: instead of computing exact dynamics or issuing detailed muscle-level commands, it expresses intention through high-level movement tendencies, while reflexes and biomechanical mechanisms autonomously resolve local details. This architecture enables robustness, flexibility, and cross-task generalization. Motivated by this insight, we propose a generic soft-robot control framework grounded in control compliance and validate it across robots with diverse morphologies and actuation mechanisms. The results demonstrate stable, safe, and cross-platform transferable behavior, indicating that embracing control compliance, rather than resisting it, may provide a widely applicable foundation for unified soft-robot control.
>
---
#### [new 030] Reasoning-VLA: A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶中视觉-语言-动作（VLA）模型推理慢、泛化能力差的问题，提出Reasoning-VLA框架。通过可学习的动作查询与增强的视觉语言特征交互，实现并行连续动作生成，并整合多数据集构建标准化训练数据，结合监督与强化学习，显著提升性能、泛化性与推理速度。**

- **链接: [https://arxiv.org/pdf/2511.19912v1](https://arxiv.org/pdf/2511.19912v1)**

> **作者:** Dapeng Zhang; Zhenlong Yuan; Zhangquan Chen; Chih-Ting Liao; Yinda Chen; Fei Shen; Qingguo Zhou; Tat-Seng Chua
>
> **摘要:** Vision-Language-Action (VLA) models have recently shown strong decision-making capabilities in autonomous driving. However, existing VLAs often struggle with achieving efficient inference and generalizing to novel autonomous vehicle configurations and driving scenarios. In this paper, we propose Reasoning-VLA, a general and fast action-generation VLA framework. The proposed model employs a set of learnable action queries, initialized via Gaussian sampling from ground-truth trajectories within the training corpus. These learnable queries interact with reasoning-enhanced vision-language features to generate continuous action trajectories in parallel. To promote robust generalization, we consolidate eight publicly available autonomous driving datasets into a standardized, Chain-of-Thought reasoning-based, and easy-to-use data format for model training. Leveraging both supervised learning and reinforcement learning fine-tuning, extensive empirical evaluations across multiple benchmarks demonstrate that Reasoning-VLA achieves state-of-the-art performance, superior generalization capability, and the excellent inference speed reported to date.
>
---
#### [new 031] Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对开放世界具身智能中闭环评估难的问题，提出Wanderland框架，通过多传感器捕获与几何精确重建，实现高保真仿真。解决了视觉与几何模拟真实差距大、图像单模态方法性能差等问题，构建了可信赖的导航评估与3D重建基准。**

- **链接: [https://arxiv.org/pdf/2511.20620v1](https://arxiv.org/pdf/2511.20620v1)**

> **作者:** Xinhao Liu; Jiaqi Li; Youming Deng; Ruxin Chen; Yingjia Zhang; Yifei Ma; Li Guo; Yiming Li; Jing Zhang; Chen Feng
>
> **摘要:** Reproducible closed-loop evaluation remains a major bottleneck in Embodied AI such as visual navigation. A promising path forward is high-fidelity simulation that combines photorealistic sensor rendering with geometrically grounded interaction in complex, open-world urban environments. Although recent video-3DGS methods ease open-world scene capturing, they are still unsuitable for benchmarking due to large visual and geometric sim-to-real gaps. To address these challenges, we introduce Wanderland, a real-to-sim framework that features multi-sensor capture, reliable reconstruction, accurate geometry, and robust view synthesis. Using this pipeline, we curate a diverse dataset of indoor-outdoor urban scenes and systematically demonstrate how image-only pipelines scale poorly, how geometry quality impacts novel view synthesis, and how all of these adversely affect navigation policy learning and evaluation reliability. Beyond serving as a trusted testbed for embodied navigation, Wanderland's rich raw sensor data further allows benchmarking of 3D reconstruction and novel view synthesis models. Our work establishes a new foundation for reproducible research in open-world embodied AI. Project website is at https://ai4ce.github.io/wanderland/.
>
---
#### [new 032] Material-informed Gaussian Splatting for 3D World Reconstruction in a Digital Twin
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对数字孪生中的3D场景重建任务，解决传统LiDAR-camera融合方法依赖复杂校准、难以表征玻璃等材料的问题。提出纯相机方案：基于多视角图像用3D高斯溅射重建几何，结合视觉模型提取材质掩码，将材质标签投影至网格并赋予物理材质属性，实现高保真传感器模拟。**

- **链接: [https://arxiv.org/pdf/2511.20348v1](https://arxiv.org/pdf/2511.20348v1)**

> **作者:** João Malheiro Silva; Andy Huynh; Tong Duy Son; Holger Caesar
>
> **备注:** 8 pages, 5 figures. Submitted to IEEE Intelligent Vehicles Symposium (IV) 2026 for possible publication
>
> **摘要:** 3D reconstruction for Digital Twins often relies on LiDAR-based methods, which provide accurate geometry but lack the semantics and textures naturally captured by cameras. Traditional LiDAR-camera fusion approaches require complex calibration and still struggle with certain materials like glass, which are visible in images but poorly represented in point clouds. We propose a camera-only pipeline that reconstructs scenes using 3D Gaussian Splatting from multi-view images, extracts semantic material masks via vision models, converts Gaussian representations to mesh surfaces with projected material labels, and assigns physics-based material properties for accurate sensor simulation in modern graphics engines and simulators. This approach combines photorealistic reconstruction with physics-based material assignment, providing sensor simulation fidelity comparable to LiDAR-camera fusion while eliminating hardware complexity and calibration requirements. We validate our camera-only method using an internal dataset from an instrumented test vehicle, leveraging LiDAR as ground truth for reflectivity validation alongside image similarity metrics.
>
---
#### [new 033] GigaWorld-0: World Models as Data Engine to Empower Embodied AI
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GigaWorld-0，一个用于具身智能的统一世界模型数据引擎。针对真实交互数据稀缺且难获取的问题，构建视频与3D生成协同框架，实现高保真、物理可信、指令对齐的模拟数据生成。通过高效训练框架支持大规模训练，使VLA模型在无真实交互情况下显著提升机器人任务泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.19861v1](https://arxiv.org/pdf/2511.19861v1)**

> **作者:** GigaWorld Team; Angen Ye; Boyuan Wang; Chaojun Ni; Guan Huang; Guosheng Zhao; Haoyun Li; Jiagang Zhu; Kerui Li; Mengyuan Xu; Qiuping Deng; Siting Wang; Wenkang Qin; Xinze Chen; Xiaofeng Wang; Yankai Wang; Yu Cao; Yifan Chang; Yuan Xu; Yun Ye; Yang Wang; Yukun Zhou; Zhengyuan Zhang; Zhehao Dong; Zheng Zhu
>
> **备注:** Project Page: https://gigaworld0.github.io/
>
> **摘要:** World models are emerging as a foundational paradigm for scalable, data-efficient embodied AI. In this work, we present GigaWorld-0, a unified world model framework designed explicitly as a data engine for Vision-Language-Action (VLA) learning. GigaWorld-0 integrates two synergistic components: GigaWorld-0-Video, which leverages large-scale video generation to produce diverse, texture-rich, and temporally coherent embodied sequences under fine-grained control of appearance, camera viewpoint, and action semantics; and GigaWorld-0-3D, which combines 3D generative modeling, 3D Gaussian Splatting reconstruction, physically differentiable system identification, and executable motion planning to ensure geometric consistency and physical realism. Their joint optimization enables the scalable synthesis of embodied interaction data that is visually compelling, spatially coherent, physically plausible, and instruction-aligned. Training at scale is made feasible through our efficient GigaTrain framework, which exploits FP8-precision and sparse attention to drastically reduce memory and compute requirements. We conduct comprehensive evaluations showing that GigaWorld-0 generates high-quality, diverse, and controllable data across multiple dimensions. Critically, VLA model (e.g., GigaBrain-0) trained on GigaWorld-0-generated data achieve strong real-world performance, significantly improving generalization and task success on physical robots without any real-world interaction during training.
>
---
#### [new 034] MIMIC-MJX: Neuromechanical Emulation of Animal Behavior
- **分类: q-bio.NC; cs.AI; cs.RO**

- **简介: 该论文提出MIMIC-MJX框架，旨在从动物运动轨迹反推生物合理神经控制策略。针对仅靠运动学数据难以揭示神经控制机制的问题，通过训练神经控制器在物理仿真中驱动生物力学模型，重现真实运动轨迹，实现高效、准确且可泛化的神经控制建模，适用于行为分析与实验模拟。**

- **链接: [https://arxiv.org/pdf/2511.20532v1](https://arxiv.org/pdf/2511.20532v1)**

> **作者:** Charles Y. Zhang; Yuanjia Yang; Aidan Sirbu; Elliott T. T. Abe; Emil Wärnberg; Eric J. Leonardis; Diego E. Aldarondo; Adam Lee; Aaditya Prasad; Jason Foat; Kaiwen Bian; Joshua Park; Rusham Bhatt; Hutton Saunders; Akira Nagamori; Ayesha R. Thanawalla; Kee Wui Huang; Fabian Plum; Hendrik K. Beck; Steven W. Flavell; David Labonte; Blake A. Richards; Bingni W. Brunton; Eiman Azim; Bence P. Ölveczky; Talmo D. Pereira
>
> **摘要:** The primary output of the nervous system is movement and behavior. While recent advances have democratized pose tracking during complex behavior, kinematic trajectories alone provide only indirect access to the underlying control processes. Here we present MIMIC-MJX, a framework for learning biologically-plausible neural control policies from kinematics. MIMIC-MJX models the generative process of motor control by training neural controllers that learn to actuate biomechanically-realistic body models in physics simulation to reproduce real kinematic trajectories. We demonstrate that our implementation is accurate, fast, data-efficient, and generalizable to diverse animal body models. Policies trained with MIMIC-MJX can be utilized to both analyze neural control strategies and simulate behavioral experiments, illustrating its potential as an integrative modeling framework for neuroscience.
>
---
#### [new 035] A K-means Inspired Solution Framework for Large-Scale Multi-Traveling Salesman Problems
- **分类: eess.SY; cs.RO**

- **简介: 该论文针对大规模多旅行商问题（MTSP）的计算复杂度难题，提出一种受K-means启发的任务分配框架。通过将MTSP转化为空间约束分类问题，利用空间一致性实现快速路径成本估算与高效任务分组，显著降低计算开销。实验表明，该方法在1000代理、5000目标的大规模场景下仍能保持高解质量，有效支持无人系统大规模协同。**

- **链接: [https://arxiv.org/pdf/2511.19454v1](https://arxiv.org/pdf/2511.19454v1)**

> **作者:** Xiubin Chen
>
> **摘要:** The Multi-Traveling Salesman Problem (MTSP) is a commonly used mathematical model for multi-agent task allocation. However, as the number of agents and task targets increases, existing optimization-based methods often incur prohibitive computational costs, posing significant challenges to large-scale coordination in unmanned systems. To address this issue, this paper proposes a K-means-inspired task allocation framework that reformulates the MTSP as a spatially constrained classification process. By leveraging spatial coherence, the proposed method enables fast estimation of path costs and efficient task grouping, thereby fundamentally reducing overall computational complexity. Extensive simulation results demonstrate that the framework can maintain high solution quality even in extremely large-scale scenarios-for instance, in tasks involving 1000 agents and 5000 targets. The findings indicate that this "cluster-then-route" decomposition strategy offers an efficient and reliable solution for large-scale multi-agent task allocation.
>
---
#### [new 036] Prune-Then-Plan: Step-Level Calibration for Stable Frontier Exploration in Embodied Question Answering
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对具身问答（EQA）中的不稳定探索问题，提出Prune-Then-Plan框架。通过霍尔姆-邦尼费罗尼启发的剪枝去除不可信前缘选择，再由覆盖导向规划器决策，实现步级校准。有效缓解了大视觉语言模型导致的前沿振荡，提升导航效率与答案质量，在多个数据集上显著优于基线。**

- **链接: [https://arxiv.org/pdf/2511.19768v1](https://arxiv.org/pdf/2511.19768v1)**

> **作者:** Noah Frahm; Prakrut Patel; Yue Zhang; Shoubin Yu; Mohit Bansal; Roni Sengupta
>
> **备注:** webpage: https://noahfrahm.github.io/Prune-Then-Plan-project-page/
>
> **摘要:** Large vision-language models (VLMs) have improved embodied question answering (EQA) agents by providing strong semantic priors for open-vocabulary reasoning. However, when used directly for step-level exploration, VLMs often exhibit frontier oscillations, unstable back-and-forth movements caused by overconfidence and miscalibration, leading to inefficient navigation and degraded answer quality. We propose Prune-Then-Plan, a simple and effective framework that stabilizes exploration through step-level calibration. Instead of trusting raw VLM scores, our method prunes implausible frontier choices using a Holm-Bonferroni inspired pruning procedure and then delegates final decisions to a coverage-based planner. This separation converts overconfident predictions into conservative, interpretable actions by relying on human-level judgments to calibrate the step-level behavior of VLMs. Integrated into the 3D-Mem EQA framework, our approach achieves relative improvements of up to 49% and 33% in visually grounded SPL and LLM-Match metrics respectively over baselines. Overall, our method achieves better scene coverage under equal exploration budgets on both OpenEQA and EXPRESS-Bench datasets.
>
---
#### [new 037] Improved Linear-Time Construction of Minimal Dominating Set via Mobile Agents
- **分类: cs.DC; cs.DS; cs.MA; cs.RO**

- **简介: 该论文研究在匿名图中通过移动代理计算最小支配集（mDS）的任务。针对同步模型下已有算法复杂度不足的问题，提出两种新算法，实现O(n)时间、O(log n)内存的线性时间解，无需全局信息，并附带构建生成树和选举领导者的高效方法。**

- **链接: [https://arxiv.org/pdf/2511.19880v1](https://arxiv.org/pdf/2511.19880v1)**

> **作者:** Prabhat Kumar Chand; Anisur Rahaman Molla
>
> **摘要:** Mobile agents have emerged as a powerful framework for solving fundamental graph problems in distributed settings in recent times. These agents, modelled as autonomous physical or software entities, possess local computation power, finite memory and have the ability to traverse a graph, offering efficient solutions to a range of classical problems. In this work, we focus on the problem of computing a \emph{minimal dominating set} (mDS) in anonymous graphs using mobile agents. Building on the recently proposed optimal dispersion algorithm on the synchronous mobile agent model, we design two new algorithms that achieve a \emph{linear-time} solution for this problem in the synchronous setting. Specifically, given a connected $n$-node graph with $n$ agents initially placed in either rooted or arbitrary configurations, we show that an mDS can be computed in $O(n)$ rounds using only $O(\log n)$ bits of memory per agent, without using any prior knowledge of any global parameters. This improves upon the best-known complexity results in the literature over the same model. In addition, as natural by-products of our methodology, our algorithms also construct a spanning tree and elect a unique leader in $O(n)$ rounds, which are also important results of independent interest in the mobile-agent framework.
>
---
#### [new 038] VibraVerse: A Large-Scale Geometry-Acoustics Alignment Dataset for Physically-Consistent Multimodal Learning
- **分类: cs.AI; cs.CV; cs.GR; cs.RO**

- **简介: 论文提出VibraVerse，一个大规模几何-声学对齐数据集，解决现有多模态学习缺乏物理一致性的问题。通过建模物体几何、材料属性与振动发声的因果关系，构建从形状到声音的可解释映射，并设计CLASP框架实现跨模态物理一致对齐，推动可解释的声学引导感知。**

- **链接: [https://arxiv.org/pdf/2511.20422v1](https://arxiv.org/pdf/2511.20422v1)**

> **作者:** Bo Pang; Chenxi Xu; Jierui Ren; Guoping Wang; Sheng Li
>
> **摘要:** Understanding the physical world requires perceptual models grounded in physical laws rather than mere statistical correlations. However, existing multimodal learning frameworks, focused on vision and language, lack physical consistency and overlook the intrinsic causal relationships among an object's geometry, material, vibration modes, and the sounds it produces. We introduce VibraVerse, a large-scale geometry-acoustics alignment dataset that explicitly bridges the causal chain from 3D geometry -> physical attributes -> modal parameters -> acoustic signals. Each 3D model has explicit physical properties (density, Young's modulus, Poisson's ratio) and volumetric geometry, from which modal eigenfrequencies and eigenvectors are computed for impact sound synthesis under controlled excitations. To establish this coherence, we introduce CLASP, a contrastive learning framework for cross-modal alignment that preserves the causal correspondence between an object's physical structure and its acoustic response. This framework enforces physically consistent alignment across modalities, ensuring that every sample is coherent, traceable to the governing equations, and embedded within a unified representation space spanning shape, image, and sound. Built upon VibraVerse, we define a suite of benchmark tasks for geometry-to-sound prediction, sound-guided shape reconstruction, and cross-modal representation learning. Extensive validations on these tasks demonstrate that models trained on VibraVerse exhibit superior accuracy, interpretability, and generalization across modalities. These results establish VibraVerse as a benchmark for physically consistent and causally interpretable multimodal learning, providing a foundation for sound-guided embodied perception and a deeper understanding of the physical world. The dataset will be open-sourced.
>
---
#### [new 039] Maritime Small Object Detection from UAVs using Deep Learning with Altitude-Aware Dynamic Tiling
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对无人机在海上搜救中检测小目标困难的问题，提出一种高度感知的动态切片方法。通过根据飞行高度自适应调整图像切片大小和数量，提升小目标检测精度与推理速度。实验表明，该方法在保持高检测性能的同时，显著降低计算开销，适用于复杂多变的海上搜救场景。**

- **链接: [https://arxiv.org/pdf/2511.19728v1](https://arxiv.org/pdf/2511.19728v1)**

> **作者:** Sakib Ahmed; Oscar Pizarro
>
> **备注:** This is the author's accepted version of an article that has been published by IEEE. The final published version is available at IEEE Xplore
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are crucial in Search and Rescue (SAR) missions due to their ability to monitor vast maritime areas. However, small objects often remain difficult to detect from high altitudes due to low object-to-background pixel ratios. We propose an altitude-aware dynamic tiling method that scales and adaptively subdivides the image into tiles for enhanced small object detection. By integrating altitude-dependent scaling with an adaptive tiling factor, we reduce unnecessary computation while maintaining detection performance. Tested on the SeaDronesSee dataset [1] with YOLOv5 [2] and Slicing Aided Hyper Inference (SAHI) framework [3], our approach improves Mean Average Precision (mAP) for small objects by 38% compared to a baseline and achieves more than double the inference speed compared to static tiling. This approach enables more efficient and accurate UAV-based SAR operations under diverse conditions.
>
---
#### [new 040] IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants
- **分类: cs.CV; cs.AI; cs.HC; cs.RO**

- **简介: 该论文提出IndEgo数据集，用于工业场景下的第一人称助手研究。针对协作任务中多模态理解与错误检测难题，构建了包含3460段第一人称和1092段第三人称视频的数据集，涵盖多种工业任务，提供丰富标注与基准测试，推动协作任务理解、错误检测与推理问答的模型发展。**

- **链接: [https://arxiv.org/pdf/2511.19684v1](https://arxiv.org/pdf/2511.19684v1)**

> **作者:** Vivek Chavan; Yasmina Imgrund; Tung Dao; Sanwantri Bai; Bosong Wang; Ze Lu; Oliver Heimann; Jörg Krüger
>
> **备注:** Accepted to NeurIPS 2025 D&B Track. Project Page: https://indego-dataset.github.io/
>
> **摘要:** We introduce IndEgo, a multimodal egocentric and exocentric dataset addressing common industrial tasks, including assembly/disassembly, logistics and organisation, inspection and repair, woodworking, and others. The dataset contains 3,460 egocentric recordings (approximately 197 hours), along with 1,092 exocentric recordings (approximately 97 hours). A key focus of the dataset is collaborative work, where two workers jointly perform cognitively and physically intensive tasks. The egocentric recordings include rich multimodal data and added context via eye gaze, narration, sound, motion, and others. We provide detailed annotations (actions, summaries, mistake annotations, narrations), metadata, processed outputs (eye gaze, hand pose, semi-dense point cloud), and benchmarks on procedural and non-procedural task understanding, Mistake Detection, and reasoning-based Question Answering. Baseline evaluations for Mistake Detection, Question Answering and collaborative task understanding show that the dataset presents a challenge for the state-of-the-art multimodal models. Our dataset is available at: https://huggingface.co/datasets/FraunhoferIPK/IndEgo
>
---
#### [new 041] Strong Duality and Dual Ascent Approach to Continuous-Time Chance-Constrained Stochastic Optimal Control
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究连续时间随机最优控制中的机会约束问题，旨在通过强对偶性将概率约束转化为期望形式，利用路径积分方法进行对偶上升求解。解决了在不确定环境下保障状态约束满足概率的控制难题，并通过仿真验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2511.19451v1](https://arxiv.org/pdf/2511.19451v1)**

> **作者:** Apurva Patil; Alfredo Duarte; Fabrizio Bisetti; Takashi Tanaka
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2504.17154
>
> **摘要:** The paper addresses a continuous-time continuous-space chance-constrained stochastic optimal control (SOC) problem where the probability of failure to satisfy given state constraints is explicitly bounded. We leverage the notion of exit time from continuous-time stochastic calculus to formulate a chance-constrained SOC problem. Without any conservative approximation, the chance constraint is transformed into an expectation of an indicator function which can be incorporated into the cost function by considering a dual formulation. We then express the dual function in terms of the solution to a Hamilton-Jacobi-Bellman partial differential equation parameterized by the dual variable. Under a certain assumption on the system dynamics and cost function, it is shown that a strong duality holds between the primal chance-constrained problem and its dual. The Path integral approach is utilized to numerically solve the dual problem via gradient ascent using open-loop samples of system trajectories. We present simulation studies on chance-constrained motion planning for spatial navigation of mobile robots and the solution of the path integral approach is compared with that of the finite difference method.
>
---
#### [new 042] BRIC: Bridging Kinematic Plans and Physical Control at Test Time
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对长时序人体动作生成中运动规划与物理控制间的执行偏差问题，提出BRIC框架。通过测试时动态调整物理控制器并引入轻量级引导机制，实现对扩散模型生成动作的实时修正，提升动作的物理合理性与环境适应性，有效解决执行漂移问题。**

- **链接: [https://arxiv.org/pdf/2511.20431v1](https://arxiv.org/pdf/2511.20431v1)**

> **作者:** Dohun Lim; Minji Kim; Jaewoon Lim; Sungchan Kim
>
> **摘要:** We propose BRIC, a novel test-time adaptation (TTA) framework that enables long-term human motion generation by resolving execution discrepancies between diffusion-based kinematic motion planners and reinforcement learning-based physics controllers. While diffusion models can generate diverse and expressive motions conditioned on text and scene context, they often produce physically implausible outputs, leading to execution drift during simulation. To address this, BRIC dynamically adapts the physics controller to noisy motion plans at test time, while preserving pre-trained skills via a loss function that mitigates catastrophic forgetting. In addition, BRIC introduces a lightweight test-time guidance mechanism that steers the diffusion model in the signal space without updating its parameters. By combining both adaptation strategies, BRIC ensures consistent and physically plausible long-term executions across diverse environments in an effective and efficient manner. We validate the effectiveness of BRIC on a variety of long-term tasks, including motion composition, obstacle avoidance, and human-scene interaction, achieving state-of-the-art performance across all tasks.
>
---
#### [new 043] AVS: A Computational and Hierarchical Storage System for Autonomous Vehicles
- **分类: cs.DC; cs.DB; cs.OS; cs.RO**

- **简介: 该论文针对自动驾驶车辆海量异构数据存储与检索难题，提出AVS系统。通过计算与存储协同设计，实现模态感知压缩、冷热分层与日归档、轻量元数据索引，支持高效实时数据处理，在嵌入式平台验证了其低延迟、高效率与小资源占用优势。**

- **链接: [https://arxiv.org/pdf/2511.19453v1](https://arxiv.org/pdf/2511.19453v1)**

> **作者:** Yuxin Wang; Yuankai He; Weisong Shi
>
> **摘要:** Autonomous vehicles (AVs) are evolving into mobile computing platforms, equipped with powerful processors and diverse sensors that generate massive heterogeneous data, for example 14 TB per day. Supporting emerging third-party applications calls for a general-purpose, queryable onboard storage system. Yet today's data loggers and storage stacks in vehicles fail to deliver efficient data storage and retrieval. This paper presents AVS, an Autonomous Vehicle Storage system that co-designs computation with a hierarchical layout: modality-aware reduction and compression, hot-cold tiering with daily archival, and a lightweight metadata layer for indexing. The design is grounded with system-level benchmarks on AV data that cover SSD and HDD filesystems and embedded indexing, and is validated on embedded hardware with real L4 autonomous driving traces. The prototype delivers predictable real-time ingest, fast selective retrieval, and substantial footprint reduction under modest resource budgets. The work also outlines observations and next steps toward more scalable and longer deployments to motivate storage as a first-class component in AV stacks.
>
---
#### [new 044] Anytime-Feasible First-Order Optimization via Safe Sequential QCQP
- **分类: math.OC; cs.RO; eess.SY**

- **简介: 该论文针对非凸不等式约束优化问题，提出安全序列二次约束二次规划（SS-QCQP）算法，确保每步迭代均保持可行性。通过连续时间动力系统与自适应离散化设计，实现$O(1/t)$收敛率；进一步提出活动集变体提升可扩展性。实验验证其在多智能体控制中兼具可行性、收敛性与高解质量。**

- **链接: [https://arxiv.org/pdf/2511.19675v1](https://arxiv.org/pdf/2511.19675v1)**

> **作者:** Jiarui Wang; Mahyar Fazlyab
>
> **摘要:** This paper presents the Safe Sequential Quadratically Constrained Quadratic Programming (SS-QCQP) algorithm, a first-order method for smooth inequality-constrained nonconvex optimization that guarantees feasibility at every iteration. The method is derived from a continuous-time dynamical system whose vector field is obtained by solving a convex QCQP that enforces monotonic descent of the objective and forward invariance of the feasible set. The resulting continuous-time dynamics achieve an $O(1/t)$ convergence rate to first-order stationary points under standard constraint qualification conditions. We then propose a safeguarded Euler discretization with adaptive step-size selection that preserves this convergence rate while maintaining both descent and feasibility in discrete time. To enhance scalability, we develop an active-set variant (SS-QCQP-AS) that selectively enforces constraints near the boundary, substantially reducing computational cost without compromising theoretical guarantees. Numerical experiments on a multi-agent nonlinear optimal control problem demonstrate that SS-QCQP and SS-QCQP-AS maintain feasibility, exhibit the predicted convergence behavior, and deliver solution quality comparable to second-order solvers such as SQP and IPOPT.
>
---
#### [new 045] MAPS: Preserving Vision-Language Representations via Module-Wise Proximity Scheduling for Better Vision-Language-Action Generalization
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在微调时破坏预训练视觉-语言表征、损害泛化能力的问题，提出MAPS框架。通过模块级近邻调度，分阶段放松不同模块的约束，平衡稳定性与适应性，无需额外参数或数据，显著提升多场景下的性能。**

- **链接: [https://arxiv.org/pdf/2511.19878v1](https://arxiv.org/pdf/2511.19878v1)**

> **作者:** Chengyue Huang; Mellon M. Zhang; Robert Azarcon; Glen Chou; Zsolt Kira
>
> **摘要:** Vision-Language-Action (VLA) models inherit strong priors from pretrained Vision-Language Models (VLMs), but naive fine-tuning often disrupts these representations and harms generalization. Existing fixes -- freezing modules or applying uniform regularization -- either overconstrain adaptation or ignore the differing roles of VLA components. We present MAPS (Module-Wise Proximity Scheduling), the first robust fine-tuning framework for VLAs. Through systematic analysis, we uncover an empirical order in which proximity constraints should be relaxed to balance stability and flexibility. MAPS linearly schedules this relaxation, enabling visual encoders to stay close to their pretrained priors while action-oriented language layers adapt more freely. MAPS introduces no additional parameters or data, and can be seamlessly integrated into existing VLAs. Across MiniVLA-VQ, MiniVLA-OFT, OpenVLA-OFT, and challenging benchmarks such as SimplerEnv, CALVIN, LIBERO, as well as real-world evaluations on the Franka Emika Panda platform, MAPS consistently boosts both in-distribution and out-of-distribution performance (up to +30%). Our findings highlight empirically guided proximity to pretrained VLMs as a simple yet powerful principle for preserving broad generalization in VLM-to-VLA transfer.
>
---
#### [new 046] Map-World: Masked Action planning and Path-Integral World Model for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶多模态路径规划问题，提出MAP-World框架。通过掩码动作规划生成多样轨迹，结合路径加权世界模型，实现无需锚点或强化学习的端到端多模态规划，提升决策多样性与计算效率。**

- **链接: [https://arxiv.org/pdf/2511.20156v1](https://arxiv.org/pdf/2511.20156v1)**

> **作者:** Bin Hu; Zijian Lu; Haicheng Liao; Chengran Yuan; Bin Rao; Yongkang Li; Guofa Li; Zhiyong Cui; Cheng-zhong Xu; Zhenning Li
>
> **摘要:** Motion planning for autonomous driving must handle multiple plausible futures while remaining computationally efficient. Recent end-to-end systems and world-model-based planners predict rich multi-modal trajectories, but typically rely on handcrafted anchors or reinforcement learning to select a single best mode for training and control. This selection discards information about alternative futures and complicates optimization. We propose MAP-World, a prior-free multi-modal planning framework that couples masked action planning with a path-weighted world model. The Masked Action Planning (MAP) module treats future ego motion as masked sequence completion: past waypoints are encoded as visible tokens, future waypoints are represented as mask tokens, and a driving-intent path provides a coarse scaffold. A compact latent planning state is expanded into multiple trajectory queries with injected noise, yielding diverse, temporally consistent modes without anchor libraries or teacher policies. A lightweight world model then rolls out future BEV semantics conditioned on each candidate trajectory. During training, semantic losses are computed as an expectation over modes, using trajectory probabilities as discrete path weights, so the planner learns from the full distribution of plausible futures instead of a single selected path. On NAVSIM, our method matches anchor-based approaches and achieves state-of-the-art performance among world-model-based methods, while avoiding reinforcement learning and maintaining real-time inference latency.
>
---
#### [new 047] CostNav: A Navigation Benchmark for Cost-Aware Evaluation of Embodied Agents
- **分类: cs.AI; cs.CE; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出CostNav，一个面向具身智能体的成本感知导航基准，旨在解决现有研究仅关注任务成功率而忽视商业可行性的问题。通过构建包含硬件、能源、维护等成本与收益的经济模型，揭示了任务成功与商业盈利间的根本差异，并指出碰撞导致的维护成本是主要亏损来源，为导航算法的经济优化提供量化评估框架。**

- **链接: [https://arxiv.org/pdf/2511.20216v1](https://arxiv.org/pdf/2511.20216v1)**

> **作者:** Haebin Seong; Sungmin Kim; Minchan Kim; Yongjun Cho; Myunchul Joe; Suhwan Choi; Jaeyoon Jung; Jiyong Youn; Yoonshik Kim; Samwoo Seong; Yubeen Park; Youngjae Yu; Yunsung Lee
>
> **摘要:** Existing navigation benchmarks focus on task success metrics while overlooking economic viability -- critical for commercial deployment of autonomous delivery robots. We introduce \emph{CostNav}, a \textbf{Micro-Navigation Economic Testbed} that evaluates embodied agents through comprehensive cost-revenue analysis aligned with real-world business operations. CostNav models the complete economic lifecycle including hardware, training, energy, maintenance costs, and delivery revenue with service-level agreements, using industry-derived parameters. \textbf{To our knowledge, CostNav is the first work to quantitatively expose the gap between navigation research metrics and commercial viability}, revealing that optimizing for task success fundamentally differs from optimizing for economic deployment. Our cost model uses parameters derived from industry data sources (energy rates, delivery service pricing), and we project from a reduced-scale simulation to realistic deliveries. Under this projection, the baseline achieves 43.0\% SLA compliance but is \emph{not} commercially viable: yielding a loss of \$30.009 per run with no finite break-even point, because operating costs are dominated by collision-induced maintenance, which accounts for 99.7\% of per-run costs and highlights collision avoidance as a key optimization target. We demonstrate a learning-based on-device navigation baseline and establish a foundation for evaluating rule-based navigation, imitation learning, and cost-aware RL training. CostNav bridges the gap between navigation research and commercial deployment, enabling data-driven decisions about economic trade-offs across navigation paradigms.
>
---
#### [new 048] Learning Massively Multitask World Models for Continuous Control
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文提出一种多任务世界模型Newt，解决在线强化学习在连续控制中难以规模化的问题。通过200个多样化任务的预训练与在线联合优化，实现高效多任务学习与快速适应新任务，推动通用控制发展。**

- **链接: [https://arxiv.org/pdf/2511.19584v1](https://arxiv.org/pdf/2511.19584v1)**

> **作者:** Nicklas Hansen; Hao Su; Xiaolong Wang
>
> **备注:** Webpage: https://www.nicklashansen.com/NewtWM
>
> **摘要:** General-purpose control demands agents that act across many tasks and embodiments, yet research on reinforcement learning (RL) for continuous control remains dominated by single-task or offline regimes, reinforcing a view that online RL does not scale. Inspired by the foundation model recipe (large-scale pretraining followed by light RL) we ask whether a single agent can be trained on hundreds of tasks with online interaction. To accelerate research in this direction, we introduce a new benchmark with 200 diverse tasks spanning many domains and embodiments, each with language instructions, demonstrations, and optionally image observations. We then present \emph{Newt}, a language-conditioned multitask world model that is first pretrained on demonstrations to acquire task-aware representations and action priors, and then jointly optimized with online interaction across all tasks. Experiments show that Newt yields better multitask performance and data-efficiency than a set of strong baselines, exhibits strong open-loop control, and enables rapid adaptation to unseen tasks. We release our environments, demonstrations, code for training and evaluation, as well as 200+ checkpoints.
>
---
## 更新

#### [replaced 001] Continually Evolving Skill Knowledge in Vision Language Action Model
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对开放环境中机器人智能的持续技能学习问题，提出Stellar VLA框架，通过知识驱动的持续学习实现任务与技能的自监督演化。其双变体模型分别建模任务-知识空间与层级结构，减少标注依赖并提升复杂动作推理能力，在基准和真实任务中显著提升成功率。**

- **链接: [https://arxiv.org/pdf/2511.18085v2](https://arxiv.org/pdf/2511.18085v2)**

> **作者:** Yuxuan Wu; Guangming Wang; Zhiheng Yang; Maoqing Yao; Brian Sheil; Hesheng Wang
>
> **摘要:** Developing general robot intelligence in open environments requires continual skill learning. Recent Vision-Language-Action (VLA) models leverage massive pretraining data to support diverse manipulation tasks, but they still depend heavily on task-specific fine-tuning, revealing a lack of continual learning capability. Existing continual learning methods are also resource-intensive to scale to VLA models. We propose Stellar VLA, a knowledge-driven continual learning framework with two variants: T-Stellar, modeling task-centric knowledge space, and TS-Stellar, capturing hierarchical task-skill structure. Stellar VLA enables self-supervised knowledge evolution through joint learning of task latent representation and the knowledge space, reducing annotation needs. Knowledge-guided expert routing provide task specialization without extra network parameters, lowering training overhead. Experiments on the LIBERO benchmark and real-world tasks show over 50 percentage average improvement in final success rates relative to baselines. TS-Stellar further excels in complex action inference, and in-depth analyses verify effective knowledge retention and discovery. Our code will be released soon.
>
---
#### [replaced 002] From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文提出政策世界模型（PWM），解决自动驾驶中世界模型与规划分离的问题。通过动作无关的未来状态预测，实现状态-动作协同预测，提升规划可靠性。引入动态并行令牌生成机制，仅用前视摄像头即达到领先性能。**

- **链接: [https://arxiv.org/pdf/2510.19654v2](https://arxiv.org/pdf/2510.19654v2)**

> **作者:** Zhida Zhao; Talas Fu; Yifan Wang; Lijun Wang; Huchuan Lu
>
> **备注:** Accepted by NuerIPS 2025 (Poster)
>
> **摘要:** Despite remarkable progress in driving world models, their potential for autonomous systems remains largely untapped: the world models are mostly learned for world simulation and decoupled from trajectory planning. While recent efforts aim to unify world modeling and planning in a single framework, the synergistic facilitation mechanism of world modeling for planning still requires further exploration. In this work, we introduce a new driving paradigm named Policy World Model (PWM), which not only integrates world modeling and trajectory planning within a unified architecture, but is also able to benefit planning using the learned world knowledge through the proposed action-free future state forecasting scheme. Through collaborative state-action prediction, PWM can mimic the human-like anticipatory perception, yielding more reliable planning performance. To facilitate the efficiency of video forecasting, we further introduce a dynamically enhanced parallel token generation mechanism, equipped with a context-guided tokenizer and an adaptive dynamic focal loss. Despite utilizing only front camera input, our method matches or exceeds state-of-the-art approaches that rely on multi-view and multi-modal inputs. Code and model weights will be released at https://github.com/6550Zhao/Policy-World-Model.
>
---
#### [replaced 003] A Physics-informed Demonstration-guided Learning Framework for Granular Material Manipulation
- **分类: cs.RO**

- **简介: 该论文针对机器人操控颗粒材料时因物理特性复杂而难以建模的问题，提出基于可微分物理模拟器的示范引导学习框架。通过梯度优化生成非颗粒材料的示范数据，加速训练，实现高效、鲁棒的颗粒物搬运策略，在仿真与真实环境中均优于传统方法。**

- **链接: [https://arxiv.org/pdf/2406.09178v2](https://arxiv.org/pdf/2406.09178v2)**

> **作者:** Minglun Wei; Xintong Yang; Yu-Kun Lai; Seyed Amir Tafrishi; Ze Ji
>
> **备注:** Accepted as a regular paper by IEEE Transactions on Neural Networks and Learning Systems (TNNLS)
>
> **摘要:** Due to the complex physical properties of granular materials, research on robot learning for manipulating such materials predominantly either disregards the consideration of their physical characteristics or uses surrogate models to approximate their physical properties. Learning to manipulate granular materials based on physical information obtained through precise modelling remains an unsolved problem. In this paper, we propose to address this challenge by constructing a differentiable physics-based simulator for granular materials using the Taichi programming language and developing a learning framework accelerated by demonstrations generated through gradient-based optimisation on non-granular materials within our simulator, eliminating the costly data collection and model training of prior methods. Experimental results show that our method, with its flexible design, trains robust policies that are capable of executing the task of transporting granular materials in both simulated and real-world environments, beyond the capabilities of standard reinforcement learning, imitation learning, and prior task-specific granular manipulation methods.
>
---
#### [replaced 004] GigaBrain-0: A World Model-Powered Vision-Language-Action Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出GigaBrain-0，一种基于世界模型生成数据的视觉-语言-动作（VLA）基础模型，旨在解决通用机器人训练中真实数据收集成本高、可扩展性差的问题。通过生成多样化仿真数据，减少对真实数据依赖，提升跨任务泛化与政策鲁棒性，尤其在复杂操作任务中表现优异。**

- **链接: [https://arxiv.org/pdf/2510.19430v2](https://arxiv.org/pdf/2510.19430v2)**

> **作者:** GigaBrain Team; Angen Ye; Boyuan Wang; Chaojun Ni; Guan Huang; Guosheng Zhao; Haoyun Li; Jie Li; Jiagang Zhu; Lv Feng; Peng Li; Qiuping Deng; Runqi Ouyang; Wenkang Qin; Xinze Chen; Xiaofeng Wang; Yang Wang; Yifan Li; Yilong Li; Yiran Ding; Yuan Xu; Yun Ye; Yukun Zhou; Zhehao Dong; Zhenan Wang; Zhichao Liu; Zheng Zhu
>
> **备注:** https://gigabrain0.github.io/
>
> **摘要:** Training Vision-Language-Action (VLA) models for generalist robots typically requires large-scale real-world robot data, which is expensive and time-consuming to collect. The inefficiency of physical data collection severely limits the scalability, and generalization capacity of current VLA systems. To address this challenge, we introduce GigaBrain-0, a novel VLA foundation model empowered by world model-generated data (e.g., video generation, real2real transfer, human transfer, view transfer, sim2real transfer data). By leveraging world models to generate diverse data at scale, GigaBrain-0 significantly reduces reliance on real robot data while improving cross-task generalization. Our approach further improves policy robustness through RGBD input modeling and embodied Chain-of-Thought (CoT) supervision, enabling the model to reason about spatial geometry, object states, and long-horizon dependencies during task execution. This leads to substantial gains in real-world performance on dexterous, long-horizon, and mobile manipulation tasks. Extensive experiments demonstrate that GigaBrain-0 achieves superior generalization across variations in appearances (e.g., textures, colors), object placements, and camera viewpoints. Additionally, we present GigaBrain-0-Small, an optimized lightweight variant designed to run efficiently on devices such as the NVIDIA Jetson AGX Orin.
>
---
#### [replaced 005] OceanGym: A Benchmark Environment for Underwater Embodied Agents
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出OceanGym，首个面向水下具身智能体的综合性基准环境，旨在解决水下感知与决策难题。通过整合多模态大模型，构建涵盖八类任务的仿真平台，推动智能体在低可见度、强流等复杂环境下实现自主探索与长程目标达成，填补了水下智能研究空白。**

- **链接: [https://arxiv.org/pdf/2509.26536v2](https://arxiv.org/pdf/2509.26536v2)**

> **作者:** Yida Xue; Mingjun Mao; Xiangyuan Ru; Yuqi Zhu; Baochang Ren; Shuofei Qiao; Mengru Wang; Shumin Deng; Xinyu An; Ningyu Zhang; Ying Chen; Huajun Chen
>
> **备注:** Work in progress
>
> **摘要:** We introduce OceanGym, the first comprehensive benchmark for ocean underwater embodied agents, designed to advance AI in one of the most demanding real-world environments. Unlike terrestrial or aerial domains, underwater settings present extreme perceptual and decision-making challenges, including low visibility, dynamic ocean currents, making effective agent deployment exceptionally difficult. OceanGym encompasses eight realistic task domains and a unified agent framework driven by Multi-modal Large Language Models (MLLMs), which integrates perception, memory, and sequential decision-making. Agents are required to comprehend optical and sonar data, autonomously explore complex environments, and accomplish long-horizon objectives under these harsh conditions. Extensive experiments reveal substantial gaps between state-of-the-art MLLM-driven agents and human experts, highlighting the persistent difficulty of perception, planning, and adaptability in ocean underwater environments. By providing a high-fidelity, rigorously designed platform, OceanGym establishes a testbed for developing robust embodied AI and transferring these capabilities to real-world autonomous ocean underwater vehicles, marking a decisive step toward intelligent agents capable of operating in one of Earth's last unexplored frontiers. The code and data are available at https://github.com/OceanGPT/OceanGym.
>
---
#### [replaced 006] RLZero: Direct Policy Inference from Language Without In-Domain Supervision
- **分类: cs.AI; cs.GR; cs.LG; cs.RO**

- **简介: 该论文提出RLZero，解决零样本语言指令下直接生成行为策略的问题。针对传统方法需任务监督或测试时训练的缺陷，提出“想象-投影-模仿”框架：利用视频生成模型构想语言描述的场景，投影至目标环境，再通过预训练无监督强化学习代理直接模仿。首次实现无需领域监督的自然语言到行为的直接映射，可零样本生成复杂动作，如从YouTube视频中学习人类动作。**

- **链接: [https://arxiv.org/pdf/2412.05718v3](https://arxiv.org/pdf/2412.05718v3)**

> **作者:** Harshit Sikchi; Siddhant Agarwal; Pranaya Jajoo; Samyak Parajuli; Caleb Chuck; Max Rudolph; Peter Stone; Amy Zhang; Scott Niekum
>
> **备注:** NeurIPS 2025, 26 pages
>
> **摘要:** The reward hypothesis states that all goals and purposes can be understood as the maximization of a received scalar reward signal. However, in practice, defining such a reward signal is notoriously difficult, as humans are often unable to predict the optimal behavior corresponding to a reward function. Natural language offers an intuitive alternative for instructing reinforcement learning (RL) agents, yet previous language-conditioned approaches either require costly supervision or test-time training given a language instruction. In this work, we present a new approach that uses a pretrained RL agent trained using only unlabeled, offline interactions--without task-specific supervision or labeled trajectories--to get zero-shot test-time policy inference from arbitrary natural language instructions. We introduce a framework comprising three steps: imagine, project, and imitate. First, the agent imagines a sequence of observations corresponding to the provided language description using video generative models. Next, these imagined observations are projected into the target environment domain. Finally, an agent pretrained in the target environment with unsupervised RL instantly imitates the projected observation sequence through a closed-form solution. To the best of our knowledge, our method, RLZero, is the first approach to show direct language-to-behavior generation abilities on a variety of tasks and environments without any in-domain supervision. We further show that components of RLZero can be used to generate policies zero-shot from cross-embodied videos, such as those available on YouTube, even for complex embodiments like humanoids.
>
---
#### [replaced 007] Disentangled Control of Multi-Agent Systems
- **分类: eess.SY; cs.RO**

- **简介: 该论文针对多智能体系统中的动态耦合与去中心化控制难题，提出一种具收敛性保障的通用控制框架。解决了复杂拓扑下多智能体系统的解耦控制问题，实现了无近似化的时变密度覆盖控制与安全编队导航，支持多目标与实时应用。**

- **链接: [https://arxiv.org/pdf/2511.05900v2](https://arxiv.org/pdf/2511.05900v2)**

> **作者:** Ruoyu Lin; Gennaro Notomista; Magnus Egerstedt
>
> **摘要:** This paper develops a general framework for multi-agent control synthesis, which applies to a wide range of problems with convergence guarantees, regardless of the complexity of the underlying graph topology and the explicit time dependence of the objective function. The proposed framework systematically addresses a particularly challenging problem in multi-agent systems, i.e., decentralization of entangled dynamics among different agents, and it naturally supports multi-objective robotics and real-time implementations. To demonstrate its generality and effectiveness, the framework is implemented across three experiments, namely time-varying leader-follower formation control, decentralized coverage control for time-varying density functions without any approximations, which is a long-standing open problem, and safe formation navigation in dense environments.
>
---
#### [replaced 008] ARBoids: Adaptive Residual Reinforcement Learning With Boids Model for Cooperative Multi-USV Target Defense
- **分类: cs.LG; cs.CR; cs.RO**

- **简介: 该论文针对多无人艇协同防御任务中，面对高机动性攻击者时拦截困难的问题，提出ARBoids框架。融合Boids模型与深度强化学习，利用Boids提供高效协同基础，DRL学习残差策略以自适应优化防御行为，显著提升拦截性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2502.18549v3](https://arxiv.org/pdf/2502.18549v3)**

> **作者:** Jiyue Tao; Tongsheng Shen; Dexin Zhao; Feitian Zhang
>
> **摘要:** The target defense problem (TDP) for unmanned surface vehicles (USVs) concerns intercepting an adversarial USV before it breaches a designated target region, using one or more defending USVs. A particularly challenging scenario arises when the attacker exhibits superior maneuverability compared to the defenders, significantly complicating effective interception. To tackle this challenge, this letter introduces ARBoids, a novel adaptive residual reinforcement learning framework that integrates deep reinforcement learning (DRL) with the biologically inspired, force-based Boids model. Within this framework, the Boids model serves as a computationally efficient baseline policy for multi-agent coordination, while DRL learns a residual policy to adaptively refine and optimize the defenders' actions. The proposed approach is validated in a high-fidelity Gazebo simulation environment, demonstrating superior performance over traditional interception strategies, including pure force-based approaches and vanilla DRL policies. Furthermore, the learned policy exhibits strong adaptability to attackers with diverse maneuverability profiles, highlighting its robustness and generalization capability. The code of ARBoids will be released upon acceptance of this letter.
>
---
#### [replaced 009] SafePR: Unified Approach for Safe Parallel Robots by Contact Detection and Reaction with Redundancy Resolution
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对并联机器人在高速交互中的安全问题，提出SafePR统一方法。通过编码器与电机电流信息估计力，结合神经网络与粒子滤波实现接触检测与定位，区分碰撞与夹持，并利用冗余度规划避免自碰撞和类型II奇异点，实现在1.5m/s速度下25–275ms内安全响应，满足ISO/TS 15066标准。**

- **链接: [https://arxiv.org/pdf/2501.17773v3](https://arxiv.org/pdf/2501.17773v3)**

> **作者:** Aran Mohammad; Tim-Lukas Habich; Thomas Seel; Moritz Schappler
>
> **摘要:** Fast and safe motion is crucial for the successful deployment of physically interactive robots. Parallel robots (PRs) offer the potential for higher speeds while maintaining the same energy limits due to their low moving masses. However, they require methods for contact detection and reaction while avoiding singularities and self-collisions. We address this issue and present SafePR - a unified approach for the detection and localization, including the distinction between collision and clamping to perform a reaction that is safe for humans and feasible for PRs. Our approach uses information from the encoders and motor currents to estimate forces via a generalized-momentum observer. Neural networks and particle filters classify and localize the contacts. We introduce reactions with redundancy resolution to avoid self-collisions and type-II singularities. Our approach detected and terminated 72 real-world collision and clamping contacts with end-effector speeds of up to 1.5 m/s, each within 25-275 ms. The forces were below the thresholds from ISO/TS 15066. By using built-in sensors, SafePR enables safe interaction with already assembled PRs without the need for new hardware components.
>
---
#### [replaced 010] GRAM: Generalization in Deep RL with a Robust Adaptation Module
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文针对深度强化学习在真实场景中泛化能力不足的问题，提出GRAM框架，通过引入鲁棒适应模块，统一处理分布内与分布外环境动态，实现跨场景的强泛化性能。在仿真与四足机器人硬件实验中验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2412.04323v3](https://arxiv.org/pdf/2412.04323v3)**

> **作者:** James Queeney; Xiaoyi Cai; Alexander Schperberg; Radu Corcodel; Mouhacine Benosman; Jonathan P. How
>
> **备注:** Accepted for publication in IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** The reliable deployment of deep reinforcement learning in real-world settings requires the ability to generalize across a variety of conditions, including both in-distribution scenarios seen during training as well as novel out-of-distribution scenarios. In this work, we present a framework for dynamics generalization in deep reinforcement learning that unifies these two distinct types of generalization within a single architecture. We introduce a robust adaptation module that provides a mechanism for identifying and reacting to both in-distribution and out-of-distribution environment dynamics, along with a joint training pipeline that combines the goals of in-distribution adaptation and out-of-distribution robustness. Our algorithm GRAM achieves strong generalization performance across in-distribution and out-of-distribution scenarios upon deployment, which we demonstrate through extensive simulation and hardware locomotion experiments on a quadruped robot.
>
---
#### [replaced 011] Count Every Rotation and Every Rotation Counts: Exploring Drone Dynamics via Propeller Sensing
- **分类: cs.RO**

- **简介: 该论文针对无人机空中感知任务，解决地面非接触式感知精度低的问题。提出基于事件相机的\sysname系统，通过精准计数螺旋桨转速（Count Every Rotation）并利用转速推断飞行状态（Every Rotation Counts），实现毫秒级延迟、高精度转速估计与飞行指令识别，显著提升跟踪性能。**

- **链接: [https://arxiv.org/pdf/2511.13100v2](https://arxiv.org/pdf/2511.13100v2)**

> **作者:** Xuecheng Chen; Jingao Xu; Wenhua Ding; Haoyang Wang; Xinyu Luo; Ruiyang Duan; Jialong Chen; Xueqian Wang; Yunhao Liu; Xinlei Chen
>
> **摘要:** As drone-based applications proliferate, paramount contactless sensing of airborne drones from the ground becomes indispensable. This work demonstrates concentrating on propeller rotational speed will substantially improve drone sensing performance and proposes an event-camera-based solution, \sysname. \sysname features two components: \textit{Count Every Rotation} achieves accurate, real-time propeller speed estimation by mitigating ultra-high sensitivity of event cameras to environmental noise. \textit{Every Rotation Counts} leverages these speeds to infer both internal and external drone dynamics. Extensive evaluations in real-world drone delivery scenarios show that \sysname achieves a sensing latency of 3$ms$ and a rotational speed estimation error of merely 0.23\%. Additionally, \sysname infers drone flight commands with 96.5\% precision and improves drone tracking accuracy by over 22\% when combined with other sensing modalities. \textit{ Demo: {\color{blue}https://eventpro25.github.io/EventPro/.} }
>
---
#### [replaced 012] SVBRD-LLM: Self-Verifying Behavioral Rule Discovery for Autonomous Vehicle Identification
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出SVBRD-LLM框架，解决自动驾驶车辆行为识别问题。通过零样本提示从真实交通视频中自动发现、验证并应用可解释的行为规则，实现90.0%准确率的自动驾驶车辆识别，揭示其在速度控制、变道和加速度上的独特行为特征。**

- **链接: [https://arxiv.org/pdf/2511.14977v2](https://arxiv.org/pdf/2511.14977v2)**

> **作者:** Xiangyu Li; Zhaomiao Guo
>
> **摘要:** As more autonomous vehicles operate on public roads, understanding real-world behavior of autonomous vehicles is critical to analyzing traffic safety, making policies, and public acceptance. This paper proposes SVBRD-LLM, a framework that automatically discovers, verifies, and applies interpretable behavioral rules from real traffic videos through zero-shot prompt engineering. The framework extracts vehicle trajectories using YOLOv8 and ByteTrack, computes kinematic features, and employs GPT-5 zero-shot prompting to compare autonomous and human-driven vehicles, generating 35 structured behavioral rule hypotheses. These rules are tested on a validation set, iteratively refined based on failure cases to filter spurious correlations, and compiled into a high-confidence rule library. The framework is evaluated on an independent test set for speed change prediction, lane change prediction, and autonomous vehicle identification tasks. Experiments on over 1500 hours of real traffic videos show that the framework achieves 90.0% accuracy and 93.3% F1-score in autonomous vehicle identification. The discovered rules clearly reveal distinctive characteristics of autonomous vehicles in speed control smoothness, lane change conservativeness, and acceleration stability, with each rule accompanied by semantic description, applicable context, and validation confidence.
>
---
#### [replaced 013] LiHi-GS: LiDAR-Supervised Gaussian Splatting for Highway Driving Scene Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶中的高速公路场景重建任务，解决现有方法忽视高速场景与未充分利用LiDAR深度信息的问题。提出LiHi-GS，通过LiDAR监督提升3D场景重建精度，并支持LiDAR渲染，实现更真实的动态场景合成与编辑。**

- **链接: [https://arxiv.org/pdf/2412.15447v3](https://arxiv.org/pdf/2412.15447v3)**

> **作者:** Pou-Chun Kung; Xianling Zhang; Katherine A. Skinner; Nikita Jaipuria
>
> **备注:** RA-L 2025
>
> **摘要:** Photorealistic 3D scene reconstruction plays an important role in autonomous driving, enabling the generation of novel data from existing datasets to simulate safety-critical scenarios and expand training data without additional acquisition costs. Gaussian Splatting (GS) facilitates real-time, photorealistic rendering with an explicit 3D Gaussian representation of the scene, providing faster processing and more intuitive scene editing than the implicit Neural Radiance Fields (NeRFs). While extensive GS research has yielded promising advancements in autonomous driving applications, they overlook two critical aspects: First, existing methods mainly focus on low-speed and feature-rich urban scenes and ignore the fact that highway scenarios play a significant role in autonomous driving. Second, while LiDARs are commonplace in autonomous driving platforms, existing methods learn primarily from images and use LiDAR only for initial estimates or without precise sensor modeling, thus missing out on leveraging the rich depth information LiDAR offers and limiting the ability to synthesize LiDAR data. In this paper, we propose a novel GS method for dynamic scene synthesis and editing with improved scene reconstruction through LiDAR supervision and support for LiDAR rendering. Unlike prior works that are tested mostly on urban datasets, to the best of our knowledge, we are the first to focus on the more challenging and highly relevant highway scenes for autonomous driving, with sparse sensor views and monotone backgrounds. Visit our project page at: https://umautobots.github.io/lihi_gs
>
---
#### [replaced 014] An Efficient Closed-Form Solution to Full Visual-Inertial State Initialization
- **分类: cs.RO**

- **简介: 该论文针对视觉-惯性系统（VIO）的初始化任务，提出一种无需非线性优化的闭式解法。通过小角度与恒速近似，建立紧凑的解析模型，设计可观测性驱动的两阶段初始化方案，显著降低误差、缩短初始化时间与计算开销，实现高效可靠的初始状态估计。**

- **链接: [https://arxiv.org/pdf/2511.18910v2](https://arxiv.org/pdf/2511.18910v2)**

> **作者:** Samuel Cerezo; Seong Hun Lee; Javier Civera
>
> **备注:** 8 pages, 2 figures, 10 tables. Submitted to RA-L
>
> **摘要:** In this letter, we present a closed-form initialization method that recovers the full visual-inertial state without nonlinear optimization. Unlike previous approaches that rely on iterative solvers, our formulation yields analytical, easy-to-implement, and numerically stable solutions for reliable start-up. Our method builds on small-rotation and constant-velocity approximations, which keep the formulation compact while preserving the essential coupling between motion and inertial measurements. We further propose an observability-driven, two-stage initialization scheme that balances accuracy with initialization latency. Extensive experiments on the EuRoC dataset validate our assumptions: our method achieves 10-20% lower initialization error than optimization-based approaches, while using 4x shorter initialization windows and reducing computational cost by 5x.
>
---
#### [replaced 015] AutoFocus-IL: VLM-based Saliency Maps for Data-Efficient Visual Imitation Learning without Extra Human Annotations
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出AutoFocus-IL，一种基于视觉语言模型的视觉模仿学习方法，旨在提升数据效率与泛化能力。针对现有方法依赖人工标注或眼动数据的问题，该方法利用VLM自动生成时序显著性图，引导策略关注任务相关特征，抑制干扰因素，从而在无需额外人类标注的情况下实现更优性能。**

- **链接: [https://arxiv.org/pdf/2511.18617v2](https://arxiv.org/pdf/2511.18617v2)**

> **作者:** Litian Gong; Fatemeh Bahrani; Yutai Zhou; Amin Banayeeanzade; Jiachen Li; Erdem Bıyık
>
> **备注:** 8 pages, 6 figures. Code and datasets available at http://autofocus-il.github.io/
>
> **摘要:** AutoFocus-IL is a simple yet effective method to improve data efficiency and generalization in visual imitation learning by guiding policies to attend to task-relevant features rather than distractors and spurious correlations. Although saliency regularization has emerged as a promising way to achieve this, existing approaches typically require costly supervision such as human gaze data or manual saliency annotations. In contrast, AutoFocus-IL leverages vision-language models (VLMs) to automatically identify and track key objects in demonstrations, generating temporal saliency maps that highlight causal visual signals while suppressing distractors. These maps are then used to regularize behavior cloning policies, yielding stronger alignment between visual attention and task-relevant cues. Experiments in both the CARLA simulator and real-robot manipulation tasks demonstrate that AutoFocus-IL not only outperforms standard behavior cloning but also surpasses state-of-the-art baselines that assume privileged access to human supervision, such as gaze data. Code, datasets, and trained policy videos are available at https://AutoFocus-IL.github.io/.
>
---
#### [replaced 016] Hestia: Voxel-Face-Aware Hierarchical Next-Best-View Acquisition for Efficient 3D Reconstruction
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对3D重建中视点规划效率与鲁棒性不足的问题，提出Hestia框架。通过分层结构、面感知设计、多样数据集和近似贪婪策略，实现高效、实时的五自由度视点选择，显著提升覆盖率与重建精度，适用于真实场景。**

- **链接: [https://arxiv.org/pdf/2508.01014v3](https://arxiv.org/pdf/2508.01014v3)**

> **作者:** Cheng-You Lu; Zhuoli Zhuang; Nguyen Thanh Trung Le; Da Xiao; Yu-Cheng Chang; Thomas Do; Srinath Sridhar; Chin-teng Lin
>
> **备注:** Accepted to the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** Advances in 3D reconstruction and novel view synthesis have enabled efficient and photorealistic rendering. However, images for reconstruction are still either largely manual or constrained by simple preplanned trajectories. To address this issue, recent works propose generalizable next-best-view planners that do not require online learning. Nevertheless, robustness and performance remain limited across various shapes. Hence, this study introduces Voxel-Face-Aware Hierarchical Next-Best-View Acquisition for Efficient 3D Reconstruction (Hestia), which addresses the shortcomings of the reinforcement learning-based generalizable approaches for five-degree-of-freedom viewpoint prediction. Hestia systematically improves the planners through four components: a more diverse dataset to promote robustness, a hierarchical structure to manage the high-dimensional continuous action search space, a close-greedy strategy to mitigate spurious correlations, and a face-aware design to avoid overlooking geometry. Experimental results show that Hestia achieves non-marginal improvements, with at least a 4% gain in coverage ratio, while reducing Chamfer Distance by 50% and maintaining real-time inference. In addition, Hestia outperforms prior methods by at least 12% in coverage ratio with a 5-image budget and remains robust to object placement variations. Finally, we demonstrate that Hestia, as a next-best-view planner, is feasible for the real-world application. Our project page is https://johnnylu305.github.io/hestia web.
>
---
#### [replaced 017] FSR-VLN: Fast and Slow Reasoning for Vision-Language Navigation with Hierarchical Multi-modal Scene Graph
- **分类: cs.RO**

- **简介: 该论文针对视觉-语言导航（VLN）中长距离空间推理效率低、延迟高的问题，提出FSR-VLN框架。通过层级多模态场景图实现从粗到细的环境理解，结合快速匹配与慢速精炼的双阶段推理机制，显著提升导航成功率并降低82%响应时间，实现在人形机器人上的自然语言交互与实时导航。**

- **链接: [https://arxiv.org/pdf/2509.13733v3](https://arxiv.org/pdf/2509.13733v3)**

> **作者:** Xiaolin Zhou; Tingyang Xiao; Liu Liu; Yucheng Wang; Maiyue Chen; Xinrui Meng; Xinjie Wang; Wei Feng; Wei Sui; Zhizhong Su
>
> **备注:** Demo video are available at https://horizonrobotics.github.io/robot_lab/fsr-vln/
>
> **摘要:** Visual-Language Navigation (VLN) is a fundamental challenge in robotic systems, with broad applications for the deployment of embodied agents in real-world environments. Despite recent advances, existing approaches are limited in long-range spatial reasoning, often exhibiting low success rates and high inference latency, particularly in long-range navigation tasks. To address these limitations, we propose FSR-VLN, a vision-language navigation system that combines a Hierarchical Multi-modal Scene Graph (HMSG) with Fast-to-Slow Navigation Reasoning (FSR). The HMSG provides a multi-modal map representation supporting progressive retrieval, from coarse room-level localization to fine-grained goal view and object identification. Building on HMSG, FSR first performs fast matching to efficiently select candidate rooms, views, and objects, then applies VLM-driven refinement for final goal selection. We evaluated FSR-VLN across four comprehensive indoor datasets collected by humanoid robots, utilizing 87 instructions that encompass a diverse range of object categories. FSR-VLN achieves state-of-the-art (SOTA) performance in all datasets, measured by the retrieval success rate (RSR), while reducing the response time by 82% compared to VLM-based methods on tour videos by activating slow reasoning only when fast intuition fails. Furthermore, we integrate FSR-VLN with speech interaction, planning, and control modules on a Unitree-G1 humanoid robot, enabling natural language interaction and real-time navigation.
>
---
#### [replaced 018] Vision Language Models Can Parse Floor Plan Maps
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究视觉语言模型（VLM）在楼层平面图解析任务中的应用，旨在让模型理解地图的标签与几何结构，以支持移动机器人复杂室内导航。通过提示VLM生成九步导航计划，实验显示其成功率高达0.96，但大开放区域性能下降，论文据此提出改进方案。**

- **链接: [https://arxiv.org/pdf/2409.12842v2](https://arxiv.org/pdf/2409.12842v2)**

> **作者:** David DeFazio; Hrudayangam Mehta; Meng Wang; Ping Yang; Jeremy Blackburn; Shiqi Zhang
>
> **摘要:** Vision language models (VLMs) can simultaneously reason about images and texts to tackle many tasks, from visual question answering to image captioning. This paper focuses on map parsing, a novel task that is unexplored within the VLM context and particularly useful to mobile robots. Map parsing requires understanding not only the labels but also the geometric configurations of a map, i.e., what areas are like and how they are connected. To evaluate the performance of VLMs on map parsing, we prompt VLMs with floor plan maps to generate task plans for complex indoor navigation. Our results demonstrate the remarkable capability of VLMs in map parsing, with a success rate of 0.96 in tasks requiring a sequence of nine navigation actions, e.g., approaching and going through doors. Other than intuitive observations, e.g., VLMs do better in smaller maps and simpler navigation tasks, there was a very interesting observation that its performance drops in large open areas. We provide practical suggestions to address such challenges as validated by our experimental results. Webpage: https://sites.google.com/view/vlm-floorplan/
>
---
#### [replaced 019] Nauplius Optimisation for Autonomous Hydrodynamics
- **分类: cs.RO; cs.NE**

- **简介: 该论文针对水下无人机群在强流、有限通信等复杂环境下的自主部署问题，提出受藤壶幼体启发的NOAH算法。通过引入水流感知漂移、不可逆锚定及群体通信机制，解决了传统优化方法在水下环境中的可靠性难题，实现了高效、节能的水下集群部署。**

- **链接: [https://arxiv.org/pdf/2510.15350v2](https://arxiv.org/pdf/2510.15350v2)**

> **作者:** Shyalan Ramesh; Scott Mann; Alex Stumpf
>
> **摘要:** Autonomous Underwater vehicles must operate in strong currents, limited acoustic bandwidth, and persistent sensing requirements where conventional swarm optimisation methods are unreliable. This paper formulates an irreversible hydrodynamic deployment problem for Autonomous Underwater Vehicle (AUV) swarms and presents Nauplius Optimisation for Autonomous Hydrodynamics (NOAH), a novel nature-inspired swarm optimisation algorithm that combines current-aware drift, irreversible settlement in persistent sensing nodes, and colony-based communication. Drawing inspiration from the behaviour of barnacle nauplii, NOAH addresses the critical limitations of existing swarm algorithms by providing hydrodynamic awareness, irreversible anchoring mechanisms, and colony-based communication capabilities essential for underwater exploration missions. The algorithm establishes a comprehensive foundation for scalable and energy-efficient underwater swarm robotics with validated performance analysis. Validation studies demonstrate an 86% success rate for permanent anchoring scenarios, providing a unified formulation for hydrodynamic constraints and irreversible settlement behaviours with an empirical study under flow.
>
---
