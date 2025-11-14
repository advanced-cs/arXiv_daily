# 机器人 cs.RO

- **最新发布 26 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] Provably Safe Stein Variational Clarity-Aware Informative Planning
- **分类: cs.RO**

- **简介: 该论文针对动态环境中信息衰减不均与安全保障不足的问题，提出一种基于清晰度建模与Stein变分推断的规划框架，在优化信息获取轨迹的同时，通过低层过滤机制确保安全性，实现安全高效的自主信息采集。**

- **链接: [https://arxiv.org/pdf/2511.09836v1](https://arxiv.org/pdf/2511.09836v1)**

> **作者:** Kaleb Ben Naveed; Utkrisht Sahai; Anouck Girard; Dimitra Panagou
>
> **备注:** Submitted to Learning for Dynamics & Control Conference 2026. Paper Website: (https://usahai18.github.io/stein_clarity/)
>
> **摘要:** Autonomous robots are increasingly deployed for information-gathering tasks in environments that vary across space and time. Planning informative and safe trajectories in such settings is challenging because information decays when regions are not revisited. Most existing planners model information as static or uniformly decaying, ignoring environments where the decay rate varies spatially; those that model non-uniform decay often overlook how it evolves along the robot's motion, and almost all treat safety as a soft penalty. In this paper, we address these challenges. We model uncertainty in the environment using clarity, a normalized representation of differential entropy from our earlier work that captures how information improves through new measurements and decays over time when regions are not revisited. Building on this, we present Stein Variational Clarity-Aware Informative Planning, a framework that embeds clarity dynamics within trajectory optimization and enforces safety through a low-level filtering mechanism based on our earlier gatekeeper framework for safety verification. The planner performs Bayesian inference-based learning via Stein variational inference, refining a distribution over informative trajectories while filtering each nominal Stein informative trajectory to ensure safety. Hardware experiments and simulations across environments with varying decay rates and obstacles demonstrate consistent safety and reduced information deficits.
>
---
#### [new 002] nuPlan-R: A Closed-Loop Planning Benchmark for Autonomous Driving via Reactive Multi-Agent Simulation
- **分类: cs.RO; cs.AI**

- **简介: 论文提出nuPlan-R，一种基于学习型多智能体仿真的闭环规划基准，替代传统规则代理，提升交通行为真实性与多样性，更公平评估自动驾驶规划器性能，尤其凸显学习型规划器优势。**

- **链接: [https://arxiv.org/pdf/2511.10403v1](https://arxiv.org/pdf/2511.10403v1)**

> **作者:** Mingxing Peng; Ruoyu Yao; Xusen Guo; Jun Ma
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Recent advances in closed-loop planning benchmarks have significantly improved the evaluation of autonomous vehicles. However, existing benchmarks still rely on rule-based reactive agents such as the Intelligent Driver Model (IDM), which lack behavioral diversity and fail to capture realistic human interactions, leading to oversimplified traffic dynamics. To address these limitations, we present nuPlan-R, a new reactive closed-loop planning benchmark that integrates learning-based reactive multi-agent simulation into the nuPlan framework. Our benchmark replaces the rule-based IDM agents with noise-decoupled diffusion-based reactive agents and introduces an interaction-aware agent selection mechanism to ensure both realism and computational efficiency. Furthermore, we extend the benchmark with two additional metrics to enable a more comprehensive assessment of planning performance. Extensive experiments demonstrate that our reactive agent model produces more realistic, diverse, and human-like traffic behaviors, leading to a benchmark environment that better reflects real-world interactive driving. We further reimplement a collection of rule-based, learning-based, and hybrid planning approaches within our nuPlan-R benchmark, providing a clearer reflection of planner performance in complex interactive scenarios and better highlighting the advantages of learning-based planners in handling complex and dynamic scenarios. These results establish nuPlan-R as a new standard for fair, reactive, and realistic closed-loop planning evaluation. We will open-source the code for the new benchmark.
>
---
#### [new 003] Robot Crash Course: Learning Soft and Stylized Falling
- **分类: cs.RO; cs.LG**

- **简介: 该论文聚焦机器人跌倒控制任务，旨在减少跌倒时的物理损伤并实现可控的最终姿态。提出一种通用奖励函数与仿真采样策略，使双足机器人在强化学习中学会柔软、风格化的跌倒动作。**

- **链接: [https://arxiv.org/pdf/2511.10635v1](https://arxiv.org/pdf/2511.10635v1)**

> **作者:** Pascal Strauch; David Müller; Sammy Christen; Agon Serifi; Ruben Grandia; Espen Knoop; Moritz Bächer
>
> **摘要:** Despite recent advances in robust locomotion, bipedal robots operating in the real world remain at risk of falling. While most research focuses on preventing such events, we instead concentrate on the phenomenon of falling itself. Specifically, we aim to reduce physical damage to the robot while providing users with control over a robot's end pose. To this end, we propose a robot agnostic reward function that balances the achievement of a desired end pose with impact minimization and the protection of critical robot parts during reinforcement learning. To make the policy robust to a broad range of initial falling conditions and to enable the specification of an arbitrary and unseen end pose at inference time, we introduce a simulation-based sampling strategy of initial and end poses. Through simulated and real-world experiments, our work demonstrates that even bipedal robots can perform controlled, soft falls.
>
---
#### [new 004] RoboBenchMart: Benchmarking Robots in Retail Environment
- **分类: cs.RO; cs.AI**

- **简介: 论文提出RoboBenchMart，首个面向零售暗仓环境的机器人操作基准，解决现有基准过于简化的问题。工作包括构建复杂多层货架场景、生成轨迹与评估工具，并验证现有模型表现不佳，推动零售自动化研究。**

- **链接: [https://arxiv.org/pdf/2511.10276v1](https://arxiv.org/pdf/2511.10276v1)**

> **作者:** Konstantin Soshin; Alexander Krapukhin; Andrei Spiridonov; Denis Shepelev; Gregorii Bukhtuev; Andrey Kuznetsov; Vlad Shakhuro
>
> **摘要:** Most existing robotic manipulation benchmarks focus on simplified tabletop scenarios, typically involving a stationary robotic arm interacting with various objects on a flat surface. To address this limitation, we introduce RoboBenchMart, a more challenging and realistic benchmark designed for dark store environments, where robots must perform complex manipulation tasks with diverse grocery items. This setting presents significant challenges, including dense object clutter and varied spatial configurations -- with items positioned at different heights, depths, and in close proximity. By targeting the retail domain, our benchmark addresses a setting with strong potential for near-term automation impact. We demonstrate that current state-of-the-art generalist models struggle to solve even common retail tasks. To support further research, we release the RoboBenchMart suite, which includes a procedural store layout generator, a trajectory generation pipeline, evaluation tools and fine-tuned baseline models.
>
---
#### [new 005] Phantom Menace: Exploring and Enhancing the Robustness of VLA Models against Physical Sensor Attacks
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究视觉-语言-动作（VLA）模型对物理传感器攻击的鲁棒性，首次系统性提出“Real-Sim-Real”框架模拟摄像头与麦克风攻击，揭示其脆弱性，并提出对抗训练防御方法，提升模型在真实攻击下的安全性。**

- **链接: [https://arxiv.org/pdf/2511.10008v1](https://arxiv.org/pdf/2511.10008v1)**

> **作者:** Xuancun Lu; Jiaxiang Chen; Shilin Xiao; Zizhi Jin; Zhangrui Chen; Hanwen Yu; Bohan Qian; Ruochen Zhou; Xiaoyu Ji; Wenyuan Xu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Vision-Language-Action (VLA) models revolutionize robotic systems by enabling end-to-end perception-to-action pipelines that integrate multiple sensory modalities, such as visual signals processed by cameras and auditory signals captured by microphones. This multi-modality integration allows VLA models to interpret complex, real-world environments using diverse sensor data streams. Given the fact that VLA-based systems heavily rely on the sensory input, the security of VLA models against physical-world sensor attacks remains critically underexplored. To address this gap, we present the first systematic study of physical sensor attacks against VLAs, quantifying the influence of sensor attacks and investigating the defenses for VLA models. We introduce a novel ``Real-Sim-Real'' framework that automatically simulates physics-based sensor attack vectors, including six attacks targeting cameras and two targeting microphones, and validates them on real robotic systems. Through large-scale evaluations across various VLA architectures and tasks under varying attack parameters, we demonstrate significant vulnerabilities, with susceptibility patterns that reveal critical dependencies on task types and model designs. We further develop an adversarial-training-based defense that enhances VLA robustness against out-of-distribution physical perturbations caused by sensor attacks while preserving model performance. Our findings expose an urgent need for standardized robustness benchmarks and mitigation strategies to secure VLA deployments in safety-critical environments.
>
---
#### [new 006] LongComp: Long-Tail Compositional Zero-Shot Generalization for Robust Trajectory Prediction
- **分类: cs.RO**

- **简介: 该论文针对自动驾驶轨迹预测中的长尾组合零样本泛化问题，提出新评估设置与任务模块化门控网络，通过场景因子化和难度预测头，显著缩小分布外性能差距，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.10411v1](https://arxiv.org/pdf/2511.10411v1)**

> **作者:** Benjamin Stoler; Jonathan Francis; Jean Oh
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Methods for trajectory prediction in Autonomous Driving must contend with rare, safety-critical scenarios that make reliance on real-world data collection alone infeasible. To assess robustness under such conditions, we propose new long-tail evaluation settings that repartition datasets to create challenging out-of-distribution (OOD) test sets. We first introduce a safety-informed scenario factorization framework, which disentangles scenarios into discrete ego and social contexts. Building on analogies to compositional zero-shot image-labeling in Computer Vision, we then hold out novel context combinations to construct challenging closed-world and open-world settings. This process induces OOD performance gaps in future motion prediction of 5.0% and 14.7% in closed-world and open-world settings, respectively, relative to in-distribution performance for a state-of-the-art baseline. To improve generalization, we extend task-modular gating networks to operate within trajectory prediction models, and develop an auxiliary, difficulty-prediction head to refine internal representations. Our strategies jointly reduce the OOD performance gaps to 2.8% and 11.5% in the two settings, respectively, while still improving in-distribution performance.
>
---
#### [new 007] From Fold to Function: Dynamic Modeling and Simulation-Driven Design of Origami Mechanisms
- **分类: cs.RO**

- **简介: 该论文提出一种基于MuJoCo的原纸折结构仿真框架，解决其动态行为模拟难题，通过图形界面建模折痕与驱动，结合CMA-ES优化设计，并以折纸弹射器实验证明其有效性，实现仿真驱动的原纸折机构快速设计与分析。**

- **链接: [https://arxiv.org/pdf/2511.10580v1](https://arxiv.org/pdf/2511.10580v1)**

> **作者:** Tianhui Han; Shashwat Singh; Sarvesh Patil; Zeynep Temel
>
> **备注:** 8 Pages, 9 Figures, Submitted to IEEE RoboSoft
>
> **摘要:** Origami-inspired mechanisms can transform flat sheets into functional three-dimensional dynamic structures that are lightweight, compact, and capable of complex motion. These properties make origami increasingly valuable in robotic and deployable systems. However, accurately simulating their folding behavior and interactions with the environment remains challenging. To address this, we present a design framework for origami mechanism simulation that utilizes MuJoCo's deformable-body capabilities. In our approach, origami sheets are represented as graphs of interconnected deformable elements with user-specified constraints such as creases and actuation, defined through an intuitive graphical user interface (GUI). This framework allows users to generate physically consistent simulations that capture both the geometric structure of origami mechanisms and their interactions with external objects and surfaces. We demonstrate our method's utility through a case study on an origami catapult, where design parameters are optimized in simulation using the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) and validated experimentally on physical prototypes. The optimized structure achieves improved throwing performance, illustrating how our system enables rapid, simulation-driven origami design, optimization, and analysis.
>
---
#### [new 008] Audio-VLA: Adding Contact Audio Perception to Vision-Language-Action Model for Robotic Manipulation
- **分类: cs.RO; cs.SD**

- **简介: 论文提出Audio-VLA，将接触音频引入视觉-语言-动作模型，解决视觉仅靠无法感知操作动态过程的问题。通过多模态融合与音频增强仿真，提升机器人对操作过程的感知能力，并提出TCR指标评估动态过程表现。**

- **链接: [https://arxiv.org/pdf/2511.09958v1](https://arxiv.org/pdf/2511.09958v1)**

> **作者:** Xiangyi Wei; Haotian Zhang; Xinyi Cao; Siyu Xie; Weifeng Ge; Yang Li; Changbo Wang
>
> **摘要:** The Vision-Language-Action models (VLA) have achieved significant advances in robotic manipulation recently. However, vision-only VLA models create fundamental limitations, particularly in perceiving interactive and manipulation dynamic processes. This paper proposes Audio-VLA, a multimodal manipulation policy that leverages contact audio to perceive contact events and dynamic process feedback. Audio-VLA overcomes the vision-only constraints of VLA models. Additionally, this paper introduces the Task Completion Rate (TCR) metric to systematically evaluate dynamic operational processes. Audio-VLA employs pre-trained DINOv2 and SigLIP as visual encoders, AudioCLIP as the audio encoder, and Llama2 as the large language model backbone. We apply LoRA fine-tuning to these pre-trained modules to achieve robust cross-modal understanding of both visual and acoustic inputs. A multimodal projection layer aligns features from different modalities into the same feature space. Moreover RLBench and LIBERO simulation environments are enhanced by adding collision-based audio generation to provide realistic sound feedback during object interactions. Since current robotic manipulation evaluations focus on final outcomes rather than providing systematic assessment of dynamic operational processes, the proposed TCR metric measures how well robots perceive dynamic processes during manipulation, creating a more comprehensive evaluation metric. Extensive experiments on LIBERO, RLBench, and two real-world tasks demonstrate Audio-VLA's superior performance over vision-only comparative methods, while the TCR metric effectively quantifies dynamic process perception capabilities.
>
---
#### [new 009] A Study on Enhancing the Generalization Ability of Visuomotor Policies via Data Augmentation
- **分类: cs.RO**

- **简介: 该论文面向视觉运动策略的泛化问题，提出通过自动化生成多维度场景随机化数据（如相机位姿、光照、桌面纹理等）提升策略在零样本仿真实际迁移中的泛化能力，显著优于传统方法。**

- **链接: [https://arxiv.org/pdf/2511.09932v1](https://arxiv.org/pdf/2511.09932v1)**

> **作者:** Hanwen Wang
>
> **摘要:** The generalization ability of visuomotor policy is crucial, as a good policy should be deployable across diverse scenarios. Some methods can collect large amounts of trajectory augmentation data to train more generalizable imitation learning policies, aimed at handling the random placement of objects on the scene's horizontal plane. However, the data generated by these methods still lack diversity, which limits the generalization ability of the trained policy. To address this, we investigate the performance of policies trained by existing methods across different scene layout factors via automate the data generation for those factors that significantly impact generalization. We have created a more extensively randomized dataset that can be efficiently and automatically generated with only a small amount of human demonstration. The dataset covers five types of manipulators and two types of grippers, incorporating extensive randomization factors such as camera pose, lighting conditions, tabletop texture, and table height across six manipulation tasks. We found that all of these factors influence the generalization ability of the policy. Applying any form of randomization enhances policy generalization, with diverse trajectories particularly effective in bridging visual gap. Notably, we investigated on low-cost manipulator the effect of the scene randomization proposed in this work on enhancing the generalization capability of visuomotor policies for zero-shot sim-to-real transfer.
>
---
#### [new 010] Learning a Thousand Tasks in a Day
- **分类: cs.RO**

- **简介: 该论文研究机器人模仿学习，针对单任务需大量演示的问题，提出分解轨迹为对齐与交互阶段，并引入检索泛化，开发MT3方法，仅需单次演示即可学习千项新任务，大幅提升数据效率。**

- **链接: [https://arxiv.org/pdf/2511.10110v1](https://arxiv.org/pdf/2511.10110v1)**

> **作者:** Kamil Dreczkowski; Pietro Vitiello; Vitalis Vosylius; Edward Johns
>
> **备注:** This is the author's version of the work. It is posted here by permission of the AAAS for personal use, not for redistribution. The definitive version was published in Science Robotics on 12 November 2025, DOI: https://www.science.org/doi/10.1126/scirobotics.adv7594. Link to project website: https://www.robot-learning.uk/learning-1000-tasks
>
> **摘要:** Humans are remarkably efficient at learning tasks from demonstrations, but today's imitation learning methods for robot manipulation often require hundreds or thousands of demonstrations per task. We investigate two fundamental priors for improving learning efficiency: decomposing manipulation trajectories into sequential alignment and interaction phases, and retrieval-based generalisation. Through 3,450 real-world rollouts, we systematically study this decomposition. We compare different design choices for the alignment and interaction phases, and examine generalisation and scaling trends relative to today's dominant paradigm of behavioural cloning with a single-phase monolithic policy. In the few-demonstrations-per-task regime (<10 demonstrations), decomposition achieves an order of magnitude improvement in data efficiency over single-phase learning, with retrieval consistently outperforming behavioural cloning for both alignment and interaction. Building on these insights, we develop Multi-Task Trajectory Transfer (MT3), an imitation learning method based on decomposition and retrieval. MT3 learns everyday manipulation tasks from as little as a single demonstration each, whilst also generalising to novel object instances. This efficiency enables us to teach a robot 1,000 distinct everyday tasks in under 24 hours of human demonstrator time. Through 2,200 additional real-world rollouts, we reveal MT3's capabilities and limitations across different task families. Videos of our experiments can be found on at https://www.robot-learning.uk/learning-1000-tasks.
>
---
#### [new 011] A Shared-Autonomy Construction Robotic System for Overhead Works
- **分类: cs.RO; eess.SY**

- **简介: 该论文面向高空作业（如钻孔），构建了一套共享自主机器人系统，通过高斯溅射实时三维重建与神经配置空间屏障，实现动态环境下的安全遥操作，验证了硬件在钻孔、紧固等任务中的可行性。**

- **链接: [https://arxiv.org/pdf/2511.09695v1](https://arxiv.org/pdf/2511.09695v1)**

> **作者:** David Minkwan Kim; K. M. Brian Lee; Yong Hyeok Seo; Nikola Raicevic; Runfa Blark Li; Kehan Long; Chan Seon Yoon; Dong Min Kang; Byeong Jo Lim; Young Pyoung Kim; Nikolay Atanasov; Truong Nguyen; Se Woong Jun; Young Wook Kim
>
> **备注:** 4pages, 8 figures, ICRA construction workshop
>
> **摘要:** We present the ongoing development of a robotic system for overhead work such as ceiling drilling. The hardware platform comprises a mobile base with a two-stage lift, on which a bimanual torso is mounted with a custom-designed drilling end effector and RGB-D cameras. To support teleoperation in dynamic environments with limited visibility, we use Gaussian splatting for online 3D reconstruction and introduce motion parameters to model moving objects. For safe operation around dynamic obstacles, we developed a neural configuration-space barrier approach for planning and control. Initial feasibility studies demonstrate the capability of the hardware in drilling, bolting, and anchoring, and the software in safe teleoperation in a dynamic environment.
>
---
#### [new 012] A Robust Task-Level Control Architecture for Learned Dynamical Systems
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文针对机器人学习动力系统中任务执行偏差问题，提出L1-DS架构，融合名义控制器与L1自适应控制器，并引入窗口化DTW目标选择器，提升轨迹跟踪鲁棒性，适用于手写运动学习任务。**

- **链接: [https://arxiv.org/pdf/2511.09790v1](https://arxiv.org/pdf/2511.09790v1)**

> **作者:** Eshika Pathak; Ahmed Aboudonia; Sandeep Banik; Naira Hovakimyan
>
> **摘要:** Dynamical system (DS)-based learning from demonstration (LfD) is a powerful tool for generating motion plans in the operation (`task') space of robotic systems. However, the realization of the generated motion plans is often compromised by a ''task-execution mismatch'', where unmodeled dynamics, persistent disturbances, and system latency cause the robot's actual task-space state to diverge from the desired motion trajectory. We propose a novel task-level robust control architecture, L1-augmented Dynamical Systems (L1-DS), that explicitly handles the task-execution mismatch in tracking a nominal motion plan generated by any DS-based LfD scheme. Our framework augments any DS-based LfD model with a nominal stabilizing controller and an L1 adaptive controller. Furthermore, we introduce a windowed Dynamic Time Warping (DTW)-based target selector, which enables the nominal stabilizing controller to handle temporal misalignment for improved phase-consistent tracking. We demonstrate the efficacy of our architecture on the LASA and IROS handwriting datasets.
>
---
#### [new 013] Optimizing the flight path for a scouting Uncrewed Aerial Vehicle
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对灾后环境无结构化导致救援路径规划困难的问题，提出基于优化的无人机飞行路径规划方法，通过调整飞行高度最大化传感器覆盖面积并最小化数据不确定性，实现高效侦察任务。**

- **链接: [https://arxiv.org/pdf/2511.10598v1](https://arxiv.org/pdf/2511.10598v1)**

> **作者:** Raghav Adhikari; Sachet Khatiwada; Suman Poudel
>
> **备注:** This paper was prepared as an end of semester project for ME8710: Engineering Optimization, Clemson University. Consists of 7 pages and 8 figures
>
> **摘要:** Post-disaster situations pose unique navigation challenges. One of those challenges is the unstructured nature of the environment, which makes it hard to layout paths for rescue vehicles. We propose the use of Uncrewed Aerial Vehicle (UAV) in such scenario to perform reconnaissance across the environment. To accomplish this, we propose an optimization-based approach to plan a path for the UAV at optimal height where the sensors of the UAV can cover the most area and collect data with minimum uncertainty.
>
---
#### [new 014] PuffyBot: An Untethered Shape Morphing Robot for Multi-environment Locomotion
- **分类: cs.RO**

- **简介: 论文提出PuffyBot，一种可自主变形的无缆机器人，解决多环境（陆地/水下）运动适应问题。通过剪刀机构与 bell-crank 连杆实现体积与形态变化，支持爬行与游泳模式切换，采用TPU防水材料，实现双模浮力调控与两小时续航。**

- **链接: [https://arxiv.org/pdf/2511.09885v1](https://arxiv.org/pdf/2511.09885v1)**

> **作者:** Shashwat Singh; Zilin Si; Zeynep Temel
>
> **备注:** 8 pages, 10 figures, IEEE RoboSoft 2026
>
> **摘要:** Amphibians adapt their morphologies and motions to accommodate movement in both terrestrial and aquatic environments. Inspired by these biological features, we present PuffyBot, an untethered shape morphing robot capable of changing its body morphology to navigate multiple environments. Our robot design leverages a scissor-lift mechanism driven by a linear actuator as its primary structure to achieve shape morphing. The transformation enables a volume change from 255.00 cm3 to 423.75 cm3, modulating the buoyant force to counteract a downward force of 3.237 N due to 330 g mass of the robot. A bell-crank linkage is integrated with the scissor-lift mechanism, which adjusts the servo-actuated limbs by 90 degrees, allowing a seamless transition between crawling and swimming modes. The robot is fully waterproof, using thermoplastic polyurethane (TPU) fabric to ensure functionality in aquatic environments. The robot can operate untethered for two hours with an onboard battery of 1000 mA h. Our experimental results demonstrate multi-environment locomotion, including crawling on the land, crawling on the underwater floor, swimming on the water surface, and bimodal buoyancy adjustment to submerge underwater or resurface. These findings show the potential of shape morphing to create versatile and energy efficient robotic platforms suitable for diverse environments.
>
---
#### [new 015] Improving dependability in robotized bolting operations
- **分类: cs.RO**

- **简介: 该论文针对机器人拧螺栓操作中 autonomy 与故障管理不足的问题，提出一种融合精准力控、多模态人机交互与分层监督控制的可靠框架，提升故障检测与操作安全性，实验验证了其在管道法兰装配中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.10448v1](https://arxiv.org/pdf/2511.10448v1)**

> **作者:** Lorenzo Pagliara; Violeta Redondo; Enrico Ferrentino; Manuel Ferre; Pasquale Chiacchio
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** Bolting operations are critical in industrial assembly and in the maintenance of scientific facilities, requiring high precision and robustness to faults. Although robotic solutions have the potential to improve operational safety and effectiveness, current systems still lack reliable autonomy and fault management capabilities. To address this gap, we propose a control framework for dependable robotized bolting tasks and instantiate it on a specific robotic system. The system features a control architecture ensuring accurate driving torque control and active compliance throughout the entire operation, enabling safe interaction even under fault conditions. By designing a multimodal human-robot interface (HRI) providing real-time visualization of relevant system information and supporting seamless transitions between automatic and manual control, we improve operator situation awareness and fault detection capabilities. A high-level supervisor (SV) coordinates the execution and manages transitions between control modes, ensuring consistency with the supervisory control (SVC) paradigm, while preserving the human operator's authority. The system is validated in a representative bolting operation involving pipe flange joining, under several fault conditions. The results demonstrate improved fault detection capabilities, enhanced operator situational awareness, and accurate and compliant execution of the bolting operation. However, they also reveal the limitations of relying on a single camera to achieve full situational awareness.
>
---
#### [new 016] Baby Sophia: A Developmental Approach to Self-Exploration through Self-Touch and Hand Regard
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文提出Baby Sophia机器人框架，基于婴儿发育机制，利用内在奖励与课程学习，实现无监督的自触与手部注视学习，解决机器人自主多模态探索问题，模拟婴儿从随机动作到目的性行为的发展过程。**

- **链接: [https://arxiv.org/pdf/2511.09727v1](https://arxiv.org/pdf/2511.09727v1)**

> **作者:** Stelios Zarifis; Ioannis Chalkiadakis; Artemis Chardouveli; Vasiliki Moutzouri; Aggelos Sotirchos; Katerina Papadimitriou; Panagiotis Filntisis; Niki Efthymiou; Petros Maragos; Katerina Pastra
>
> **备注:** 5 pages, 3 tables
>
> **摘要:** Inspired by infant development, we propose a Reinforcement Learning (RL) framework for autonomous self-exploration in a robotic agent, Baby Sophia, using the BabyBench simulation environment. The agent learns self-touch and hand regard behaviors through intrinsic rewards that mimic an infant's curiosity-driven exploration of its own body. For self-touch, high-dimensional tactile inputs are transformed into compact, meaningful representations, enabling efficient learning. The agent then discovers new tactile contacts through intrinsic rewards and curriculum learning that encourage broad body coverage, balance, and generalization. For hand regard, visual features of the hands, such as skin-color and shape, are learned through motor babbling. Then, intrinsic rewards encourage the agent to perform novel hand motions, and follow its hands with its gaze. A curriculum learning setup from single-hand to dual-hand training allows the agent to reach complex visual-motor coordination. The results of this work demonstrate that purely curiosity-based signals, with no external supervision, can drive coordinated multimodal learning, imitating an infant's progression from random motor babbling to purposeful behaviors.
>
---
#### [new 017] Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对离线到在线强化学习中的行为覆盖不足与分布偏移问题，提出UEPO框架，通过多种子扩散策略、动态正则化和数据增强，实现高效、多样且鲁棒的机器人策略优化，在D4RL基准上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.10087v1](https://arxiv.org/pdf/2511.10087v1)**

> **作者:** Haidong Huang; Haiyue Zhu. Jiayu Song; Xixin Zhao; Yaohua Zhou; Jiayi Zhang; Yuze Zhai; Xiaocong Li
>
> **备注:** Accepted by NeurIPS 2025 Workshop on Embodied World Models for Decision Making
>
> **摘要:** Offline-to-online reinforcement learning (O2O-RL) has emerged as a promising paradigm for safe and efficient robotic policy deployment but suffers from two fundamental challenges: limited coverage of multimodal behaviors and distributional shifts during online adaptation. We propose UEPO, a unified generative framework inspired by large language model pretraining and fine-tuning strategies. Our contributions are threefold: (1) a multi-seed dynamics-aware diffusion policy that efficiently captures diverse modalities without training multiple models; (2) a dynamic divergence regularization mechanism that enforces physically meaningful policy diversity; and (3) a diffusion-based data augmentation module that enhances dynamics model generalization. On the D4RL benchmark, UEPO achieves +5.9\% absolute improvement over Uni-O4 on locomotion tasks and +12.4\% on dexterous manipulation, demonstrating strong generalization and scalability.
>
---
#### [new 018] Physics-informed Machine Learning for Static Friction Modeling in Robotic Manipulators Based on Kolmogorov-Arnold Networks
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出基于Kolmogorov-Arnold网络的物理信息机器学习方法，用于机器人关节静摩擦建模，解决传统模型依赖先验函数形式的问题，实现高精度、可解释的摩擦表达式自动提取。**

- **链接: [https://arxiv.org/pdf/2511.10079v1](https://arxiv.org/pdf/2511.10079v1)**

> **作者:** Yizheng Wang; Timon Rabczuk; Yinghua Liu
>
> **摘要:** Friction modeling plays a crucial role in achieving high-precision motion control in robotic operating systems. Traditional static friction models (such as the Stribeck model) are widely used due to their simple forms; however, they typically require predefined functional assumptions, which poses significant challenges when dealing with unknown functional structures. To address this issue, this paper proposes a physics-inspired machine learning approach based on the Kolmogorov Arnold Network (KAN) for static friction modeling of robotic joints. The method integrates spline activation functions with a symbolic regression mechanism, enabling model simplification and physical expression extraction through pruning and attribute scoring, while maintaining both high prediction accuracy and interpretability. We first validate the method's capability to accurately identify key parameters under known functional models, and further demonstrate its robustness and generalization ability under conditions with unknown functional structures and noisy data. Experiments conducted on both synthetic data and real friction data collected from a six-degree-of-freedom industrial manipulator show that the proposed method achieves a coefficient of determination greater than 0.95 across various tasks and successfully extracts concise and physically meaningful friction expressions. This study provides a new perspective for interpretable and data-driven robotic friction modeling with promising engineering applicability.
>
---
#### [new 019] DecARt Leg: Design and Evaluation of a Novel Humanoid Robot Leg with Decoupled Actuation for Agile Locomotion
- **分类: cs.RO**

- **简介: 该论文提出一种新型解耦驱动机器人腿DecARt Leg，旨在实现敏捷运动。通过准伸缩结构、前向膝关节和多连杆踝传动设计，解决传统腿机构耦合驱动响应慢的问题，并提出FAST指标进行性能评估，辅以仿真与实验验证。**

- **链接: [https://arxiv.org/pdf/2511.10021v1](https://arxiv.org/pdf/2511.10021v1)**

> **作者:** Egor Davydenko; Andrei Volchenkov; Vladimir Gerasimov; Roman Gorbachev
>
> **摘要:** In this paper, we propose a novel design of an electrically actuated robotic leg, called the DecARt (Decoupled Actuation Robot) Leg, aimed at performing agile locomotion. This design incorporates several new features, such as the use of a quasi-telescopic kinematic structure with rotational motors for decoupled actuation, a near-anthropomorphic leg appearance with a forward facing knee, and a novel multi-bar system for ankle torque transmission from motors placed above the knee. To analyze the agile locomotion capabilities of the design numerically, we propose a new descriptive metric, called the `Fastest Achievable Swing Time` (FAST), and perform a quantitative evaluation of the proposed design and compare it with other designs. Then we evaluate the performance of the DecARt Leg-based robot via extensive simulation and preliminary hardware experiments.
>
---
#### [new 020] ScaleADFG: Affordance-based Dexterous Functional Grasping via Scalable Dataset
- **分类: cs.RO**

- **简介: 论文提出ScaleADFG框架，解决机器人手与日常物体尺寸不匹配导致的灵巧抓取泛化难题，通过自动化构建多尺度工具使用抓取数据集（6万+抓取）和轻量级抓取生成网络，实现无需微调的零样本泛化与稳定抓取。**

- **链接: [https://arxiv.org/pdf/2511.09602v1](https://arxiv.org/pdf/2511.09602v1)**

> **作者:** Sizhe Wang; Yifan Yang; Yongkang Luo; Daheng Li; Wei Wei; Yan Zhang; Peiying Hu; Yunjin Fu; Haonan Duan; Jia Sun; Peng Wang
>
> **备注:** Accepted by IEEE Robotics and Automation Letters
>
> **摘要:** Dexterous functional tool-use grasping is essential for effective robotic manipulation of tools. However, existing approaches face significant challenges in efficiently constructing large-scale datasets and ensuring generalizability to everyday object scales. These issues primarily arise from size mismatches between robotic and human hands, and the diversity in real-world object scales. To address these limitations, we propose the ScaleADFG framework, which consists of a fully automated dataset construction pipeline and a lightweight grasp generation network. Our dataset introduce an affordance-based algorithm to synthesize diverse tool-use grasp configurations without expert demonstrations, allowing flexible object-hand size ratios and enabling large robotic hands (compared to human hands) to grasp everyday objects effectively. Additionally, we leverage pre-trained models to generate extensive 3D assets and facilitate efficient retrieval of object affordances. Our dataset comprising five object categories, each containing over 1,000 unique shapes with 15 scale variations. After filtering, the dataset includes over 60,000 grasps for each 2 dexterous robotic hands. On top of this dataset, we train a lightweight, single-stage grasp generation network with a notably simple loss design, eliminating the need for post-refinement. This demonstrates the critical importance of large-scale datasets and multi-scale object variant for effective training. Extensive experiments in simulation and on real robot confirm that the ScaleADFG framework exhibits strong adaptability to objects of varying scales, enhancing functional grasp stability, diversity, and generalizability. Moreover, our network exhibits effective zero-shot transfer to real-world objects. Project page is available at https://sizhe-wang.github.io/ScaleADFG_webpage
>
---
#### [new 021] SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **简介: SemanticVLA面向机器人操作任务，解决视觉冗余与语义对齐不足问题，提出语义对齐的稀疏化与增强框架，通过三模块协同提升效率与性能，在LIBERO基准上显著超越OpenVLA。**

- **链接: [https://arxiv.org/pdf/2511.10518v1](https://arxiv.org/pdf/2511.10518v1)**

> **作者:** Wei Li; Renshan Zhang; Rui Shao; Zhijian Fang; Kaiwen Zhou; Zhuotao Tian; Liqiang Nie
>
> **备注:** Accepted to AAAI 2026 (Oral), Project Page: https://github.com/JiuTian-VL/SemanticVLA
>
> **摘要:** Vision-Language-Action (VLA) models have advanced in robotic manipulation, yet practical deployment remains hindered by two key limitations: 1) perceptual redundancy, where irrelevant visual inputs are processed inefficiently, and 2) superficial instruction-vision alignment, which hampers semantic grounding of actions. In this paper, we propose SemanticVLA, a novel VLA framework that performs Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation. Specifically: 1) To sparsify redundant perception while preserving semantic alignment, Semantic-guided Dual Visual Pruner (SD-Pruner) performs: Instruction-driven Pruner (ID-Pruner) extracts global action cues and local semantic anchors in SigLIP; Spatial-aggregation Pruner (SA-Pruner) compacts geometry-rich features into task-adaptive tokens in DINOv2. 2) To exploit sparsified features and integrate semantics with spatial geometry, Semantic-complementary Hierarchical Fuser (SH-Fuser) fuses dense patches and sparse tokens across SigLIP and DINOv2 for coherent representation. 3) To enhance the transformation from perception to action, Semantic-conditioned Action Coupler (SA-Coupler) replaces the conventional observation-to-DoF approach, yielding more efficient and interpretable behavior modeling for manipulation tasks. Extensive experiments on simulation and real-world tasks show that SemanticVLA sets a new SOTA in both performance and efficiency. SemanticVLA surpasses OpenVLA on LIBERO benchmark by 21.1% in success rate, while reducing training cost and inference latency by 3.0-fold and 2.7-fold.SemanticVLA is open-sourced and publicly available at https://github.com/JiuTian-VL/SemanticVLA
>
---
#### [new 022] PALMS+: Modular Image-Based Floor Plan Localization Leveraging Depth Foundation Model
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: PALMS+提出一种无基础设施的视觉定位方法，利用单目深度模型重建3D点云，通过几何匹配实现高精度室内外定位，无需训练，显著优于PALMS和F3Loc。**

- **链接: [https://arxiv.org/pdf/2511.09724v1](https://arxiv.org/pdf/2511.09724v1)**

> **作者:** Yunqian Cheng; Benjamin Princen; Roberto Manduchi
>
> **备注:** Accepted to IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026, Application Track. Main paper: 8 pages, 5 figures. Supplementary material included
>
> **摘要:** Indoor localization in GPS-denied environments is crucial for applications like emergency response and assistive navigation. Vision-based methods such as PALMS enable infrastructure-free localization using only a floor plan and a stationary scan, but are limited by the short range of smartphone LiDAR and ambiguity in indoor layouts. We propose PALMS$+$, a modular, image-based system that addresses these challenges by reconstructing scale-aligned 3D point clouds from posed RGB images using a foundation monocular depth estimation model (Depth Pro), followed by geometric layout matching via convolution with the floor plan. PALMS$+$ outputs a posterior over the location and orientation, usable for direct or sequential localization. Evaluated on the Structured3D and a custom campus dataset consisting of 80 observations across four large campus buildings, PALMS$+$ outperforms PALMS and F3Loc in stationary localization accuracy -- without requiring any training. Furthermore, when integrated with a particle filter for sequential localization on 33 real-world trajectories, PALMS$+$ achieved lower localization errors compared to other methods, demonstrating robustness for camera-free tracking and its potential for infrastructure-free applications. Code and data are available at https://github.com/Head-inthe-Cloud/PALMS-Plane-based-Accessible-Indoor-Localization-Using-Mobile-Smartphones
>
---
#### [new 023] VISTA: A Vision and Intent-Aware Social Attention Framework for Multi-Agent Trajectory Prediction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: VISTA提出一种视觉与意图感知的社会注意力框架，用于多智能体轨迹预测，解决现有方法忽视长期目标与细粒度交互的问题，通过递归目标条件Transformer实现更真实、可解释且安全的轨迹生成。**

- **链接: [https://arxiv.org/pdf/2511.10203v1](https://arxiv.org/pdf/2511.10203v1)**

> **作者:** Stephane Da Silva Martins; Emanuel Aldea; Sylvie Le Hégarat-Mascle
>
> **备注:** Paper accepted at WACV 2026
>
> **摘要:** Multi-agent trajectory prediction is crucial for autonomous systems operating in dense, interactive environments. Existing methods often fail to jointly capture agents' long-term goals and their fine-grained social interactions, which leads to unrealistic multi-agent futures. We propose VISTA, a recursive goal-conditioned transformer for multi-agent trajectory forecasting. VISTA combines (i) a cross-attention fusion module that integrates long-horizon intent with past motion, (ii) a social-token attention mechanism for flexible interaction modeling across agents, and (iii) pairwise attention maps that make social influence patterns interpretable at inference time. Our model turns single-agent goal-conditioned prediction into a coherent multi-agent forecasting framework. Beyond standard displacement metrics, we evaluate trajectory collision rates as a measure of joint realism. On the high-density MADRAS benchmark and on SDD, VISTA achieves state-of-the-art accuracy and substantially fewer collisions. On MADRAS, it reduces the average collision rate of strong baselines from 2.14 to 0.03 percent, and on SDD it attains zero collisions while improving ADE, FDE, and minFDE. These results show that VISTA generates socially compliant, goal-aware, and interpretable trajectories, making it promising for safety-critical autonomous systems.
>
---
#### [new 024] Harnessing Bounded-Support Evolution Strategies for Policy Refinement
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文针对机器人策略优化中梯度噪声大的问题，提出TD-ES方法，利用有界三角噪声与秩排序估计，实现无梯度、并行化的策略精调。在PPO预训练后应用，显著提升成功率与稳定性。**

- **链接: [https://arxiv.org/pdf/2511.09923v1](https://arxiv.org/pdf/2511.09923v1)**

> **作者:** Ethan Hirschowitz; Fabio Ramos
>
> **备注:** 10 pages, 6 figures, to be published in Australasian Conference on Robotics and Automation (ACRA 2025)
>
> **摘要:** Improving competent robot policies with on-policy RL is often hampered by noisy, low-signal gradients. We revisit Evolution Strategies (ES) as a policy-gradient proxy and localize exploration with bounded, antithetic triangular perturbations, suitable for policy refinement. We propose Triangular-Distribution ES (TD-ES) which pairs bounded triangular noise with a centered-rank finite-difference estimator to deliver stable, parallelizable, gradient-free updates. In a two-stage pipeline -- PPO pretraining followed by TD-ES refinement -- this preserves early sample efficiency while enabling robust late-stage gains. Across a suite of robotic manipulation tasks, TD-ES raises success rates by 26.5% relative to PPO and greatly reduces variance, offering a simple, compute-light path to reliable refinement.
>
---
#### [new 025] MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation
- **分类: cs.CV; cs.RO**

- **简介: 论文提出MSGNav，面向零样本具身导航任务，解决传统3D场景图丢失视觉信息与词汇受限问题，引入多模态3D场景图（M3DSG）并设计四模块系统，实现开放词汇、高效推理与精准终点选择，性能达SOTA。**

- **链接: [https://arxiv.org/pdf/2511.10376v1](https://arxiv.org/pdf/2511.10376v1)**

> **作者:** Xun Huang; Shijia Zhao; Yunxiang Wang; Xin Lu; Wanfa Zhang; Rongsheng Qu; Weixin Li; Yunhong Wang; Chenglu Wen
>
> **备注:** 10 pages
>
> **摘要:** Embodied navigation is a fundamental capability for robotic agents operating. Real-world deployment requires open vocabulary generalization and low training overhead, motivating zero-shot methods rather than task-specific RL training. However, existing zero-shot methods that build explicit 3D scene graphs often compress rich visual observations into text-only relations, leading to high construction cost, irreversible loss of visual evidence, and constrained vocabularies. To address these limitations, we introduce the Multi-modal 3D Scene Graph (M3DSG), which preserves visual cues by replacing textual relational edges with dynamically assigned images. Built on M3DSG, we propose MSGNav, a zero-shot navigation system that includes a Key Subgraph Selection module for efficient reasoning, an Adaptive Vocabulary Update module for open vocabulary support, and a Closed-Loop Reasoning module for accurate exploration reasoning. Additionally, we further identify the last-mile problem in zero-shot navigation - determining the feasible target location with a suitable final viewpoint, and propose a Visibility-based Viewpoint Decision module to explicitly resolve it. Comprehensive experimental results demonstrate that MSGNav achieves state-of-the-art performance on GOAT-Bench and HM3D-OVON datasets. The open-source code will be publicly available.
>
---
#### [new 026] Safe Planning in Interactive Environments via Iterative Policy Updates and Adversarially Robust Conformal Prediction
- **分类: eess.SY; cs.RO**

- **简介: 该论文面向交互环境中的安全规划任务，解决策略更新导致分布偏移而失效的安全保证问题，提出基于对抗鲁棒共形预测的迭代框架，通过策略-轨迹敏感性分析动态调整安全边界，首次实现交互场景下的收敛性与安全保证。**

- **链接: [https://arxiv.org/pdf/2511.10586v1](https://arxiv.org/pdf/2511.10586v1)**

> **作者:** Omid Mirzaeedodangeh; Eliot Shekhtman; Nikolai Matni; Lars Lindemann
>
> **摘要:** Safe planning of an autonomous agent in interactive environments -- such as the control of a self-driving vehicle among pedestrians and human-controlled vehicles -- poses a major challenge as the behavior of the environment is unknown and reactive to the behavior of the autonomous agent. This coupling gives rise to interaction-driven distribution shifts where the autonomous agent's control policy may change the environment's behavior, thereby invalidating safety guarantees in existing work. Indeed, recent works have used conformal prediction (CP) to generate distribution-free safety guarantees using observed data of the environment. However, CP's assumption on data exchangeability is violated in interactive settings due to a circular dependency where a control policy update changes the environment's behavior, and vice versa. To address this gap, we propose an iterative framework that robustly maintains safety guarantees across policy updates by quantifying the potential impact of a planned policy update on the environment's behavior. We realize this via adversarially robust CP where we perform a regular CP step in each episode using observed data under the current policy, but then transfer safety guarantees across policy updates by analytically adjusting the CP result to account for distribution shifts. This adjustment is performed based on a policy-to-trajectory sensitivity analysis, resulting in a safe, episodic open-loop planner. We further conduct a contraction analysis of the system providing conditions under which both the CP results and the policy updates are guaranteed to converge. We empirically demonstrate these safety and convergence guarantees on a two-dimensional car-pedestrian case study. To the best of our knowledge, these are the first results that provide valid safety guarantees in such interactive settings.
>
---
## 更新

#### [replaced 001] Feedback-MPPI: Fast Sampling-Based MPC via Rollout Differentiation -- Adios low-level controllers
- **分类: cs.RO; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.14855v2](https://arxiv.org/pdf/2506.14855v2)**

> **作者:** Tommaso Belvedere; Michael Ziegltrum; Giulio Turrisi; Valerio Modugno
>
> **摘要:** Model Predictive Path Integral control is a powerful sampling-based approach suitable for complex robotic tasks due to its flexibility in handling nonlinear dynamics and non-convex costs. However, its applicability in real-time, highfrequency robotic control scenarios is limited by computational demands. This paper introduces Feedback-MPPI (F-MPPI), a novel framework that augments standard MPPI by computing local linear feedback gains derived from sensitivity analysis inspired by Riccati-based feedback used in gradient-based MPC. These gains allow for rapid closed-loop corrections around the current state without requiring full re-optimization at each timestep. We demonstrate the effectiveness of F-MPPI through simulations and real-world experiments on two robotic platforms: a quadrupedal robot performing dynamic locomotion on uneven terrain and a quadrotor executing aggressive maneuvers with onboard computation. Results illustrate that incorporating local feedback significantly improves control performance and stability, enabling robust, high-frequency operation suitable for complex robotic systems.
>
---
#### [replaced 002] Onboard Mission Replanning for Adaptive Cooperative Multi-Robot Systems
- **分类: cs.RO; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.06094v4](https://arxiv.org/pdf/2506.06094v4)**

> **作者:** Elim Kwan; Rehman Qureshi; Liam Fletcher; Colin Laganier; Victoria Nockles; Richard Walters
>
> **备注:** 9 pages, 5 figures, 1 table
>
> **摘要:** Cooperative autonomous robotic systems have significant potential for executing complex multi-task missions across space, air, ground, and maritime domains. But they commonly operate in remote, dynamic and hazardous environments, requiring rapid in-mission adaptation without reliance on fragile or slow communication links to centralised compute. Fast, on-board replanning algorithms are therefore needed to enhance resilience. Reinforcement Learning shows strong promise for efficiently solving mission planning tasks when formulated as Travelling Salesperson Problems (TSPs), but existing methods: 1) are unsuitable for replanning, where agents do not start at a single location; 2) do not allow cooperation between agents; 3) are unable to model tasks with variable durations; or 4) lack practical considerations for on-board deployment. Here we define the Cooperative Mission Replanning Problem as a novel variant of multiple TSP with adaptations to overcome these issues, and develop a new encoder/decoder-based model using Graph Attention Networks and Attention Models to solve it effectively and efficiently. Using a simple example of cooperative drones, we show our replanner consistently (90% of the time) maintains performance within 10% of the state-of-the-art LKH3 heuristic solver, whilst running 85-370 times faster on a Raspberry Pi. This work paves the way for increased resilience in autonomous multi-agent systems.
>
---
#### [replaced 003] Understanding while Exploring: Semantics-driven Active Mapping
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.00225v2](https://arxiv.org/pdf/2506.00225v2)**

> **作者:** Liyan Chen; Huangying Zhan; Hairong Yin; Yi Xu; Philippos Mordohai
>
> **摘要:** Effective robotic autonomy in unknown environments demands proactive exploration and precise understanding of both geometry and semantics. In this paper, we propose ActiveSGM, an active semantic mapping framework designed to predict the informativeness of potential observations before execution. Built upon a 3D Gaussian Splatting (3DGS) mapping backbone, our approach employs semantic and geometric uncertainty quantification, coupled with a sparse semantic representation, to guide exploration. By enabling robots to strategically select the most beneficial viewpoints, ActiveSGM efficiently enhances mapping completeness, accuracy, and robustness to noisy semantic data, ultimately supporting more adaptive scene exploration. Our experiments on the Replica and Matterport3D datasets highlight the effectiveness of ActiveSGM in active semantic mapping tasks.
>
---
#### [replaced 004] ATOM-CBF: Adaptive Safe Perception-Based Control under Out-of-Distribution Measurements
- **分类: cs.RO; eess.SY**

- **链接: [https://arxiv.org/pdf/2511.08741v2](https://arxiv.org/pdf/2511.08741v2)**

> **作者:** Kai S. Yun; Navid Azizan
>
> **摘要:** Ensuring the safety of real-world systems is challenging, especially when they rely on learned perception modules to infer the system state from high-dimensional sensor data. These perception modules are vulnerable to epistemic uncertainty, often failing when encountering out-of-distribution (OoD) measurements not seen during training. To address this gap, we introduce ATOM-CBF (Adaptive-To-OoD-Measurement Control Barrier Function), a novel safe control framework that explicitly computes and adapts to the epistemic uncertainty from OoD measurements, without the need for ground-truth labels or information on distribution shifts. Our approach features two key components: (1) an OoD-aware adaptive perception error margin and (2) a safety filter that integrates this adaptive error margin, enabling the filter to adjust its conservatism in real-time. We provide empirical validation in simulations, demonstrating that ATOM-CBF maintains safety for an F1Tenth vehicle with LiDAR scans and a quadruped robot with RGB images.
>
---
#### [replaced 005] VisualMimic: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.20322v2](https://arxiv.org/pdf/2509.20322v2)**

> **作者:** Shaofeng Yin; Yanjie Ze; Hong-Xing Yu; C. Karen Liu; Jiajun Wu
>
> **备注:** Website: https://visualmimic.github.io
>
> **摘要:** Humanoid loco-manipulation in unstructured environments demands tight integration of egocentric perception and whole-body control. However, existing approaches either depend on external motion capture systems or fail to generalize across diverse tasks. We introduce VisualMimic, a visual sim-to-real framework that unifies egocentric vision with hierarchical whole-body control for humanoid robots. VisualMimic combines a task-agnostic low-level keypoint tracker -- trained from human motion data via a teacher-student scheme -- with a task-specific high-level policy that generates keypoint commands from visual and proprioceptive input. To ensure stable training, we inject noise into the low-level policy and clip high-level actions using human motion statistics. VisualMimic enables zero-shot transfer of visuomotor policies trained in simulation to real humanoid robots, accomplishing a wide range of loco-manipulation tasks such as box lifting, pushing, football dribbling, and kicking. Beyond controlled laboratory settings, our policies also generalize robustly to outdoor environments. Videos are available at: https://visualmimic.github.io .
>
---
#### [replaced 006] Unlocking Efficient Vehicle Dynamics Modeling via Analytic World Models
- **分类: cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2502.10012v2](https://arxiv.org/pdf/2502.10012v2)**

> **作者:** Asen Nachkov; Danda Pani Paudel; Jan-Nico Zaech; Davide Scaramuzza; Luc Van Gool
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** Differentiable simulators represent an environment's dynamics as a differentiable function. Within robotics and autonomous driving, this property is used in Analytic Policy Gradients (APG), which relies on backpropagating through the dynamics to train accurate policies for diverse tasks. Here we show that differentiable simulation also has an important role in world modeling, where it can impart predictive, prescriptive, and counterfactual capabilities to an agent. Specifically, we design three novel task setups in which the differentiable dynamics are combined within an end-to-end computation graph not with a policy, but a state predictor. This allows us to learn relative odometry, optimal planners, and optimal inverse states. We collectively call these predictors Analytic World Models (AWMs) and demonstrate how differentiable simulation enables their efficient, end-to-end learning. In autonomous driving scenarios, they have broad applicability and can augment an agent's decision-making beyond reactive control.
>
---
#### [replaced 007] Depth Matters: Multimodal RGB-D Perception for Robust Autonomous Agents
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.16711v3](https://arxiv.org/pdf/2503.16711v3)**

> **作者:** Mihaela-Larisa Clement; Mónika Farsang; Felix Resch; Mihai-Teodor Stanusoiu; Radu Grosu
>
> **摘要:** Autonomous agents that rely purely on perception to make real-time control decisions require efficient and robust architectures. In this work, we demonstrate that augmenting RGB input with depth information significantly enhances our agents' ability to predict steering commands compared to using RGB alone. We benchmark lightweight recurrent controllers that leverage the fused RGB-D features for sequential decision-making. To train our models, we collect high-quality data using a small-scale autonomous car controlled by an expert driver via a physical steering wheel, capturing varying levels of steering difficulty. Our models were successfully deployed on real hardware and inherently avoided dynamic and static obstacles, under out-of-distribution conditions. Specifically, our findings reveal that the early fusion of depth data results in a highly robust controller, which remains effective even with frame drops and increased noise levels, without compromising the network's focus on the task.
>
---
#### [replaced 008] UniGS: Unified Geometry-Aware Gaussian Splatting for Multimodal Rendering
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2510.12174v2](https://arxiv.org/pdf/2510.12174v2)**

> **作者:** Yusen Xie; Zhenmin Huang; Jianhao Jiao; Dimitrios Kanoulas; Jun Ma
>
> **摘要:** In this paper, we propose UniGS, a unified map representation and differentiable framework for high-fidelity multimodal 3D reconstruction based on 3D Gaussian Splatting. Our framework integrates a CUDA-accelerated rasterization pipeline capable of rendering photo-realistic RGB images, geometrically accurate depth maps, consistent surface normals, and semantic logits simultaneously. We redesign the rasterization to render depth via differentiable ray-ellipsoid intersection rather than using Gaussian centers, enabling effective optimization of rotation and scale attribute through analytic depth gradients. Furthermore, we derive the analytic gradient formulation for surface normal rendering, ensuring geometric consistency among reconstructed 3D scenes. To improve computational and storage efficiency, we introduce a learnable attribute that enables differentiable pruning of Gaussians with minimal contribution during training. Quantitative and qualitative experiments demonstrate state-of-the-art reconstruction accuracy across all modalities, validating the efficacy of our geometry-aware paradigm. Source code and multimodal viewer will be available on GitHub.
>
---
#### [replaced 009] Special Unitary Parameterized Estimators of Rotation
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2411.13109v3](https://arxiv.org/pdf/2411.13109v3)**

> **作者:** Akshay Chandrasekhar
>
> **备注:** 32 pages; new algebraic formula for QuadMobiusAlg; three new benchmark experiments
>
> **摘要:** This paper revisits the topic of rotation estimation through the lens of special unitary matrices. We begin by reformulating Wahba's problem using $SU(2)$ to derive multiple solutions that yield linear constraints on corresponding quaternion parameters. We then explore applications of these constraints by formulating efficient methods for related problems. Finally, from this theoretical foundation, we propose two novel continuous representations for learning rotations in neural networks. Extensive experiments validate the effectiveness of the proposed methods.
>
---
#### [replaced 010] Improving Pre-Trained Vision-Language-Action Policies with Model-Based Search
- **分类: cs.RO; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.12211v2](https://arxiv.org/pdf/2508.12211v2)**

> **作者:** Cyrus Neary; Omar G. Younis; Artur Kuramshin; Ozgur Aslan; Glen Berseth
>
> **摘要:** Pre-trained vision-language-action (VLA) models offer a promising foundation for generalist robot policies, but often produce brittle behaviors or unsafe failures when deployed zero-shot in out-of-distribution scenarios. We present Vision-Language-Action Planning & Search (VLAPS) -- a novel framework and accompanying algorithms that embed model-based search into the inference procedure of pre-trained VLA policies to improve their performance on robotic tasks. Specifically, our method biases a modified Monte Carlo Tree Search (MCTS) algorithm -- run using a model of the target environment -- using action priors defined by the VLA policy. By using VLA-derived abstractions and priors in model-based search, VLAPS efficiently explores language-conditioned robotics tasks whose search spaces would otherwise be intractably large. Conversely, by integrating model-based search with the VLA policy's inference procedure, VLAPS yields behaviors that are more performant than those obtained by directly following the VLA policy's action predictions. VLAPS offers a principled framework to: i) control test-time compute in VLA models, ii) leverage a priori knowledge of the robotic environment, and iii) integrate established planning and reinforcement learning techniques into the VLA inference process. Across all experiments, VLAPS significantly outperforms VLA-only baselines on language-specified tasks that would otherwise be intractable for uninformed search algorithms, increasing success rates by as much as 67 percentage points.
>
---
#### [replaced 011] Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models
- **分类: cs.RO; cs.AI; cs.HC**

- **链接: [https://arxiv.org/pdf/2511.02162v3](https://arxiv.org/pdf/2511.02162v3)**

> **作者:** Alexander Htet Kyaw; Richa Gupta; Dhruv Shah; Anoop Sinha; Kory Mathewson; Stefanie Pender; Sachin Chitta; Yotto Koga; Faez Ahmed; Lawrence Sass; Randall Davis
>
> **备注:** Accepted to NeurIPS 2025, Conference on Neural Information Processing Systems, Creative AI Track
>
> **摘要:** Advances in 3D generative AI have enabled the creation of physical objects from text prompts, but challenges remain in creating objects involving multiple component types. We present a pipeline that integrates 3D generative AI with vision-language models (VLMs) to enable the robotic assembly of multi-component objects from natural language. Our method leverages VLMs for zero-shot, multi-modal reasoning about geometry and functionality to decompose AI-generated meshes into multi-component 3D models using predefined structural and panel components. We demonstrate that a VLM is capable of determining which mesh regions need panel components in addition to structural components, based on the object's geometry and functionality. Evaluation across test objects shows that users preferred the VLM-generated assignments 90.6% of the time, compared to 59.4% for rule-based and 2.5% for random assignment. Lastly, the system allows users to refine component assignments through conversational feedback, enabling greater human control and agency in making physical objects with generative AI and robotics.
>
---
#### [replaced 012] Keep on Going: Learning Robust Humanoid Motion Skills via Selective Adversarial Training
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2507.08303v3](https://arxiv.org/pdf/2507.08303v3)**

> **作者:** Yang Zhang; Zhanxiang Cao; Buqing Nie; Haoyang Li; Zhong Jiangwei; Qiao Sun; Xiaoyi Hu; Xiaokang Yang; Yue Gao
>
> **备注:** 13 pages, 10 figures, AAAI2026
>
> **摘要:** Humanoid robots are expected to operate reliably over long horizons while executing versatile whole-body skills. Yet Reinforcement Learning (RL) motion policies typically lose stability under prolonged operation, sensor/actuator noise, and real world disturbances. In this work, we propose a Selective Adversarial Attack for Robust Training (SA2RT) to enhance the robustness of motion skills. The adversary is learned to identify and sparsely perturb the most vulnerable states and actions under an attack-budget constraint, thereby exposing true weakness without inducing conservative overfitting. The resulting non-zero sum, alternating optimization continually strengthens the motion policy against the strongest discovered attacks. We validate our approach on the Unitree G1 humanoid robot across perceptive locomotion and whole-body control tasks. Experimental results show that adversarially trained policies improve the terrain traversal success rate by 40%, reduce the trajectory tracking error by 32%, and maintain long horizon mobility and tracking performance. Together, these results demonstrate that selective adversarial attacks are an effective driver for learning robust, long horizon humanoid motion skills.
>
---
#### [replaced 013] GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets
- **分类: cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2511.06471v2](https://arxiv.org/pdf/2511.06471v2)**

> **作者:** Jingtao Tang; Hang Ma
>
> **备注:** Accepted to AAAI-2026
>
> **摘要:** We study GCS-TSP, a new variant of the Traveling Salesman Problem (TSP) defined over a Graph of Convex Sets (GCS) -- a powerful representation for trajectory planning that decomposes the configuration space into convex regions connected by a sparse graph. In this setting, edge costs are not fixed but depend on the specific trajectory selected through each convex region, making classical TSP methods inapplicable. We introduce GHOST, a hierarchical framework that optimally solves the GCS-TSP by combining combinatorial tour search with convex trajectory optimization. GHOST systematically explores tours on a complete graph induced by the GCS, using a novel abstract-path-unfolding algorithm to compute admissible lower bounds that guide best-first search at both the high level (over tours) and the low level (over feasible GCS paths realizing the tour). These bounds provide strong pruning power, enabling efficient search while avoiding unnecessary convex optimization calls. We prove that GHOST guarantees optimality and present a bounded-suboptimal variant for time-critical scenarios. Experiments show that GHOST is orders-of-magnitude faster than unified mixed-integer convex programming baselines for simple cases and uniquely handles complex trajectory planning problems involving high-order continuity constraints and an incomplete GCS.
>
---
#### [replaced 014] BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2508.08241v4](https://arxiv.org/pdf/2508.08241v4)**

> **作者:** Qiayuan Liao; Takara E. Truong; Xiaoyu Huang; Yuman Gao; Guy Tevet; Koushil Sreenath; C. Karen Liu
>
> **备注:** Project page: https://beyondmimic.github.io/
>
> **摘要:** The human-like form of humanoid robots positions them uniquely to achieve the agility and versatility in motor skills that humans possess. Learning from human demonstrations offers a scalable approach to acquiring these capabilities. However, prior works either produce unnatural motions or rely on motion-specific tuning to achieve satisfactory naturalness. Furthermore, these methods are often motion- or goal-specific, lacking the versatility to compose diverse skills, especially when solving unseen tasks. We present BeyondMimic, a framework that scales to diverse motions and carries the versatility to compose them seamlessly in tackling unseen downstream tasks. At heart, a compact motion-tracking formulation enables mastering a wide range of radically agile behaviors, including aerial cartwheels, spin-kicks, flip-kicks, and sprinting, with a single setup and shared hyperparameters, all while achieving state-of-the-art human-like performance. Moving beyond the mere imitation of existing motions, we propose a unified latent diffusion model that empowers versatile goal specification, seamless task switching, and dynamic composition of these agile behaviors. Leveraging classifier guidance, a diffusion-specific technique for test-time optimization toward novel objectives, our model extends its capability to solve downstream tasks never encountered during training, including motion inpainting, joystick teleoperation, and obstacle avoidance, and transfers these skills zero-shot to real hardware. This work opens new frontiers for humanoid robots by pushing the limits of scalable human-like motor skill acquisition from human motion and advancing seamless motion synthesis that achieves generalization and versatility beyond training setups.
>
---
#### [replaced 015] Stochastic Adaptive Estimation in Polynomial Curvature Shape State Space for Continuum Robots
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2210.08427v5](https://arxiv.org/pdf/2210.08427v5)**

> **作者:** Guoqing Zhang; Long Wang
>
> **备注:** 20 pages. IEEE Transactions on Robotics - Accepted; this arXiv version corresponds to the final revision. Supplementary appendix provided as an ancillary PDF
>
> **摘要:** In continuum robotics, real-time robust shape estimation is crucial for planning and control tasks that involve physical manipulation in complex environments. In this paper, we present a novel stochastic observer-based shape estimation framework designed specifically for continuum robots. The shape state space is uniquely represented by the modal coefficients of a polynomial, enabled by leveraging polynomial curvature kinematics (PCK) to describe the curvature distribution along the arclength. Our framework processes noisy measurements from limited discrete position, orientation, or pose sensors to estimate the shape state robustly. We derive a novel noise-weighted observability matrix, providing a detailed assessment of observability variations under diverse sensor configurations. To overcome the limitations of a single model, our observer employs the Interacting Multiple Model (IMM) method, coupled with Extended Kalman Filters (EKFs), to mix polynomial curvature models of different orders. The IMM approach, rooted in Markov processes, effectively manages multiple model scenarios by dynamically adapting to different polynomial orders based on real-time model probabilities. This adaptability is key to ensuring robust shape estimation of the robot's behaviors under various conditions. Our comprehensive analysis, supported by both simulation studies and experimental validations, confirms the robustness and accuracy of our methods.
>
---
#### [replaced 016] ManipDreamer3D : Synthesizing Plausible Robotic Manipulation Video with Occupancy-aware 3D Trajectory
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.05314v2](https://arxiv.org/pdf/2509.05314v2)**

> **作者:** Ying Li; Xiaobao Wei; Xiaowei Chi; Yuming Li; Zhongyu Zhao; Hao Wang; Ningning Ma; Ming Lu; Sirui Han; Shanghang Zhang
>
> **备注:** 7pages; 7figures; 3 tables
>
> **摘要:** Data scarcity continues to be a major challenge in the field of robotic manipulation. Although diffusion models provide a promising solution for generating robotic manipulation videos, existing methods largely depend on 2D trajectories, which inherently face issues with 3D spatial ambiguity. In this work, we present a novel framework named ManipDreamer3D for generating plausible 3D-aware robotic manipulation videos from the input image and the text instruction. Our method combines 3D trajectory planning with a reconstructed 3D occupancy map created from a third-person perspective, along with a novel trajectory-to-video diffusion model. Specifically, ManipDreamer3D first reconstructs the 3D occupancy representation from the input image and then computes an optimized 3D end-effector trajectory, minimizing path length while avoiding collisions. Next, we employ a latent editing technique to create video sequences from the initial image latent and the optimized 3D trajectory. This process conditions our specially trained trajectory-to-video diffusion model to produce robotic pick-and-place videos. Our method generates robotic videos with autonomously planned plausible 3D trajectories, significantly reducing human intervention requirements. Experimental results demonstrate superior visual quality compared to existing methods.
>
---
#### [replaced 017] Towards Embodied Agentic AI: Review and Classification of LLM- and VLM-Driven Robot Autonomy and Interaction
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.05294v4](https://arxiv.org/pdf/2508.05294v4)**

> **作者:** Sahar Salimpour; Lei Fu; Kajetan Rachwał; Pascal Bertrand; Kevin O'Sullivan; Robert Jakob; Farhad Keramat; Leonardo Militano; Giovanni Toffetti; Harry Edelman; Jorge Peña Queralta
>
> **摘要:** Foundation models, including large language models (LLMs) and vision-language models (VLMs), have recently enabled novel approaches to robot autonomy and human-robot interfaces. In parallel, vision-language-action models (VLAs) or large behavior models (LBMs) are increasing the dexterity and capabilities of robotic systems. This survey paper reviews works that advance agentic applications and architectures, including initial efforts with GPT-style interfaces and more complex systems where AI agents function as coordinators, planners, perception actors, or generalist interfaces. Such agentic architectures allow robots to reason over natural language instructions, invoke APIs, plan task sequences, or assist in operations and diagnostics. In addition to peer-reviewed research, due to the fast-evolving nature of the field, we highlight and include community-driven projects, ROS packages, and industrial frameworks that show emerging trends. We propose a taxonomy for classifying model integration approaches and present a comparative analysis of the role that agents play in different solutions in today's literature.
>
---
