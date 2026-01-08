# 机器人 cs.RO

- **最新发布 26 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] A Vision-Language-Action Model with Visual Prompt for OFF-Road Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决非结构化地形中轨迹规划的难题。提出OFF-EMMA框架，通过视觉提示和推理策略提升规划精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.03519v1](https://arxiv.org/pdf/2601.03519v1)**

> **作者:** Liangdong Zhang; Yiming Nie; Haoyang Li; Fanjie Kong; Baobao Zhang; Shunxin Huang; Kai Fu; Chen Min; Liang Xiao
>
> **摘要:** Efficient trajectory planning in off-road terrains presents a formidable challenge for autonomous vehicles, often necessitating complex multi-step pipelines. However, traditional approaches exhibit limited adaptability in dynamic environments. To address these limitations, this paper proposes OFF-EMMA, a novel end-to-end multimodal framework designed to overcome the deficiencies of insufficient spatial perception and unstable reasoning in visual-language-action (VLA) models for off-road autonomous driving scenarios. The framework explicitly annotates input images through the design of a visual prompt block and introduces a chain-of-thought with self-consistency (COT-SC) reasoning strategy to enhance the accuracy and robustness of trajectory planning. The visual prompt block utilizes semantic segmentation masks as visual prompts, enhancing the spatial understanding ability of pre-trained visual-language models for complex terrains. The COT- SC strategy effectively mitigates the error impact of outliers on planning performance through a multi-path reasoning mechanism. Experimental results on the RELLIS-3D off-road dataset demonstrate that OFF-EMMA significantly outperforms existing methods, reducing the average L2 error of the Qwen backbone model by 13.3% and decreasing the failure rate from 16.52% to 6.56%.
>
---
#### [new 002] Lunar Rover Cargo Transport: Mission Concept and Field Test
- **分类: cs.RO**

- **简介: 该论文属于月球探测任务，旨在解决自动运输货物问题。通过Lidar Teach and Repeat技术实现自主导航与精准对接，进行了实地测试验证系统有效性。**

- **链接: [https://arxiv.org/pdf/2601.03371v1](https://arxiv.org/pdf/2601.03371v1)**

> **作者:** Alexander Krawciw; Nicolas Olmedo; Faizan Rehmatullah; Maxime Desjardins-Goulet; Pascal Toupin; Timothy D. Barfoot
>
> **备注:** 15 Pages, 13 Figures, to appear in IEEE Transactions on Field Robotics
>
> **摘要:** In future operations on the lunar surface, automated vehicles will be required to transport cargo between known locations. Such vehicles must be able to navigate precisely in safe regions to avoid natural hazards, human-constructed infrastructure, and dangerous dark shadows. Rovers must be able to park their cargo autonomously within a small tolerance to achieve a successful pickup and delivery. In this field test, Lidar Teach and Repeat provides an ideal autonomy solution for transporting cargo in this way. A one-tonne path-to-flight rover was driven in a semi-autonomous remote-control mode to create a network of safe paths. Once the route was taught, the rover immediately repeated the entire network of paths autonomously while carrying cargo. The closed-loop performance is accurate enough to align the vehicle to the cargo and pick it up. This field report describes a two-week deployment at the Canadian Space Agency's Analogue Terrain, culminating in a simulated lunar operation to evaluate the system's capabilities. Successful cargo collection and delivery were demonstrated in harsh environmental conditions.
>
---
#### [new 003] Dual-Attention Heterogeneous GNN for Multi-robot Collaborative Area Search via Deep Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于多机器人协同搜索任务，旨在解决探索与覆盖目标的动态平衡问题。提出DA-HGNN模型，通过异构图神经网络和双重注意力机制实现有效任务分配与决策。**

- **链接: [https://arxiv.org/pdf/2601.03686v1](https://arxiv.org/pdf/2601.03686v1)**

> **作者:** Lina Zhu; Jiyu Cheng; Yuehu Liu; Wei Zhang
>
> **摘要:** In multi-robot collaborative area search, a key challenge is to dynamically balance the two objectives of exploring unknown areas and covering specific targets to be rescued. Existing methods are often constrained by homogeneous graph representations, thus failing to model and balance these distinct tasks. To address this problem, we propose a Dual-Attention Heterogeneous Graph Neural Network (DA-HGNN) trained using deep reinforcement learning. Our method constructs a heterogeneous graph that incorporates three entity types: robot nodes, frontier nodes, and interesting nodes, as well as their historical states. The dual-attention mechanism comprises the relational-aware attention and type-aware attention operations. The relational-aware attention captures the complex spatio-temporal relationships among robots and candidate goals. Building on this relational-aware heterogeneous graph, the type-aware attention separately computes the relevance between robots and each goal type (frontiers vs. points of interest), thereby decoupling the exploration and coverage from the unified tasks. Extensive experiments conducted in interactive 3D scenarios within the iGibson simulator, leveraging the Gibson and MatterPort3D datasets, validate the superior scalability and generalization capability of the proposed approach.
>
---
#### [new 004] CLAP: Contrastive Latent Action Pretraining for Learning Vision-Language-Action Models from Human Videos
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出CLAP框架，解决从人类视频中学习视觉-语言-动作模型的难题，通过对比学习对齐视觉与本体感觉空间，提升机器人技能迁移效果。**

- **链接: [https://arxiv.org/pdf/2601.04061v1](https://arxiv.org/pdf/2601.04061v1)**

> **作者:** Chubin Zhang; Jianan Wang; Zifeng Gao; Yue Su; Tianru Dai; Cai Zhou; Jiwen Lu; Yansong Tang
>
> **备注:** Project page: https://lin-shan.com/CLAP/
>
> **摘要:** Generalist Vision-Language-Action models are currently hindered by the scarcity of robotic data compared to the abundance of human video demonstrations. Existing Latent Action Models attempt to leverage video data but often suffer from visual entanglement, capturing noise rather than manipulation skills. To address this, we propose Contrastive Latent Action Pretraining (CLAP), a framework that aligns the visual latent space from videos with a proprioceptive latent space from robot trajectories. By employing contrastive learning, CLAP maps video transitions onto a quantized, physically executable codebook. Building on this representation, we introduce a dual-formulation VLA framework offering both CLAP-NTP, an autoregressive model excelling at instruction following and object generalization, and CLAP-RF, a Rectified Flow-based policy designed for high-frequency, precise manipulation. Furthermore, we propose a Knowledge Matching (KM) regularization strategy to mitigate catastrophic forgetting during fine-tuning. Extensive experiments demonstrate that CLAP significantly outperforms strong baselines, enabling the effective transfer of skills from human videos to robotic execution. Project page: https://lin-shan.com/CLAP/.
>
---
#### [new 005] Towards Safe Autonomous Driving: A Real-Time Motion Planning Algorithm on Embedded Hardware
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决系统故障下的安全问题。提出一种实时轨迹规划算法，部署在嵌入式平台，确保故障时仍能安全运行。**

- **链接: [https://arxiv.org/pdf/2601.03904v1](https://arxiv.org/pdf/2601.03904v1)**

> **作者:** Korbinian Moller; Glenn Johannes Tungka; Lucas Jürgens; Johannes Betz
>
> **备注:** 7 pages, submitted to the IEEE Intelligent Vehicles Symposium (IV 2026), Detroit, MI, United States
>
> **摘要:** Ensuring the functional safety of Autonomous Vehicles (AVs) requires motion planning modules that not only operate within strict real-time constraints but also maintain controllability in case of system faults. Existing safeguarding concepts, such as Online Verification (OV), provide safety layers that detect infeasible planning outputs. However, they lack an active mechanism to ensure safe operation in the event that the main planner fails. This paper presents a first step toward an active safety extension for fail-operational Autonomous Driving (AD). We deploy a lightweight sampling-based trajectory planner on an automotive-grade, embedded platform running a Real-Time Operating System (RTOS). The planner continuously computes trajectories under constrained computational resources, forming the foundation for future emergency planning architectures. Experimental results demonstrate deterministic timing behavior with bounded latency and minimal jitter, validating the feasibility of trajectory planning on safety-certifiable hardware. The study highlights both the potential and the remaining challenges of integrating active fallback mechanisms as an integral part of next-generation safeguarding frameworks. The code is available at: https://github.com/TUM-AVS/real-time-motion-planning
>
---
#### [new 006] Integrating Sample Inheritance into Bayesian Optimization for Evolutionary Robotics
- **分类: cs.RO**

- **简介: 该论文属于进化机器人学任务，解决体脑协同优化问题。通过贝叶斯优化与样本继承提升控制器学习效率，减少预算需求。**

- **链接: [https://arxiv.org/pdf/2601.03813v1](https://arxiv.org/pdf/2601.03813v1)**

> **作者:** K. Ege de Bruin; Kyrre Glette; Kai Olav Ellefsen
>
> **摘要:** In evolutionary robotics, robot morphologies are designed automatically using evolutionary algorithms. This creates a body-brain optimization problem, where both morphology and control must be optimized together. A common approach is to include controller optimization for each morphology, but starting from scratch for every new body may require a high controller learning budget. We address this by using Bayesian optimization for controller optimization, exploiting its sample efficiency and strong exploration capabilities, and using sample inheritance as a form of Lamarckian inheritance. Under a deliberately low controller learning budget for each morphology, we investigate two types of sample inheritance: (1) transferring all the parent's samples to the offspring to be used as prior without evaluating them, and (2) reevaluating the parent's best samples on the offspring. Both are compared to a baseline without inheritance. Our results show that reevaluation performs best, with prior-based inheritance also outperforming no inheritance. Analysis reveals that while the learning budget is too low for a single morphology, generational inheritance compensates for this by accumulating learned adaptations across generations. Furthermore, inheritance mainly benefits offspring morphologies that are similar to their parents. Finally, we demonstrate the critical role of the environment, with more challenging environments resulting in more stable walking gaits. Our findings highlight that inheritance mechanisms can boost performance in evolutionary robotics without needing large learning budgets, offering an efficient path toward more capable robot design.
>
---
#### [new 007] Revisiting Continuous-Time Trajectory Estimation via Gaussian Processes and the Magnus Expansion
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于连续时间轨迹估计任务，旨在解决异步测量和Lie群状态表示问题。通过引入LTV GP与Magnus展开，构建更优雅的全局先验模型。**

- **链接: [https://arxiv.org/pdf/2601.03360v1](https://arxiv.org/pdf/2601.03360v1)**

> **作者:** Timothy Barfoot; Cedric Le Gentil; Sven Lilge
>
> **备注:** 21 pages, 12 figures
>
> **摘要:** Continuous-time state estimation has been shown to be an effective means of (i) handling asynchronous and high-rate measurements, (ii) introducing smoothness to the estimate, (iii) post hoc querying the estimate at times other than those of the measurements, and (iv) addressing certain observability issues related to scanning-while-moving sensors. A popular means of representing the trajectory in continuous time is via a Gaussian process (GP) prior, with the prior's mean and covariance functions generated by a linear time-varying (LTV) stochastic differential equation (SDE) driven by white noise. When the state comprises elements of Lie groups, previous works have resorted to a patchwork of local GPs each with a linear time-invariant SDE kernel, which while effective in practice, lacks theoretical elegance. Here we revisit the full LTV GP approach to continuous-time trajectory estimation, deriving a global GP prior on Lie groups via the Magnus expansion, which offers a more elegant and general solution. We provide a numerical comparison between the two approaches and discuss their relative merits.
>
---
#### [new 008] Wow, wo, val! A Comprehensive Embodied World Model Evaluation Turing Test
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于Embodied AI领域，旨在评估世界模型的生成能力和鲁棒性。通过构建基准测试Wow-wo-val，解决视频基础模型在感知、规划、执行等方面的真实世界适应性问题。**

- **链接: [https://arxiv.org/pdf/2601.04137v1](https://arxiv.org/pdf/2601.04137v1)**

> **作者:** Chun-Kai Fan; Xiaowei Chi; Xiaozhu Ju; Hao Li; Yong Bao; Yu-Kai Wang; Lizhang Chen; Zhiyuan Jiang; Kuangzhi Ge; Ying Li; Weishi Mi; Qingpo Wuwu; Peidong Jia; Yulin Luo; Kevin Zhang; Zhiyuan Qin; Yong Dai; Sirui Han; Yike Guo; Shanghang Zhang; Jian Tang
>
> **摘要:** As world models gain momentum in Embodied AI, an increasing number of works explore using video foundation models as predictive world models for downstream embodied tasks like 3D prediction or interactive generation. However, before exploring these downstream tasks, video foundation models still have two critical questions unanswered: (1) whether their generative generalization is sufficient to maintain perceptual fidelity in the eyes of human observers, and (2) whether they are robust enough to serve as a universal prior for real-world embodied agents. To provide a standardized framework for answering these questions, we introduce the Embodied Turing Test benchmark: WoW-World-Eval (Wow,wo,val). Building upon 609 robot manipulation data, Wow-wo-val examines five core abilities, including perception, planning, prediction, generalization, and execution. We propose a comprehensive evaluation protocol with 22 metrics to assess the models' generation ability, which achieves a high Pearson Correlation between the overall score and human preference (>0.93) and establishes a reliable foundation for the Human Turing Test. On Wow-wo-val, models achieve only 17.27 on long-horizon planning and at best 68.02 on physical consistency, indicating limited spatiotemporal consistency and physical reasoning. For the Inverse Dynamic Model Turing Test, we first use an IDM to evaluate the video foundation models' execution accuracy in the real world. However, most models collapse to $\approx$ 0% success, while WoW maintains a 40.74% success rate. These findings point to a noticeable gap between the generated videos and the real world, highlighting the urgency and necessity of benchmarking World Model in Embodied AI.
>
---
#### [new 009] From Score to Sound: An End-to-End MIDI-to-Motion Pipeline for Robotic Cello Performance
- **分类: cs.RO**

- **简介: 该论文属于机器人音乐表演任务，解决弦乐器演奏中精确控制与自动演奏问题。提出端到端MIDI到运动的管道，实现无动作捕捉的拟人演奏，并建立音乐图灵测试基准。**

- **链接: [https://arxiv.org/pdf/2601.03562v1](https://arxiv.org/pdf/2601.03562v1)**

> **作者:** Samantha Sudhoff; Pranesh Velmurugan; Jiashu Liu; Vincent Zhao; Yung-Hsiang Lu; Kristen Yeon-Ji Yun
>
> **摘要:** Robot musicians require precise control to obtain proper note accuracy, sound quality, and musical expression. Performance of string instruments, such as violin and cello, presents a significant challenge due to the precise control required over bow angle and pressure to produce the desired sound. While prior robotic cellists focus on accurate bowing trajectories, these works often rely on expensive motion capture techniques, and fail to sightread music in a human-like way. We propose a novel end-to-end MIDI score to robotic motion pipeline which converts musical input directly into collision-aware bowing motions for a UR5e robot cellist. Through use of Universal Robot Freedrive feature, our robotic musician can achieve human-like sound without the need for motion capture. Additionally, this work records live joint data via Real-Time Data Exchange (RTDE) as the robot plays, providing labeled robotic playing data from a collection of five standard pieces to the research community. To demonstrate the effectiveness of our method in comparison to human performers, we introduce the Musical Turing Test, in which a collection of 132 human participants evaluate our robot's performance against a human baseline. Human reference recordings are also released, enabling direct comparison for future studies. This evaluation technique establishes the first benchmark for robotic cello performance. Finally, we outline a residual reinforcement learning methodology to improve upon baseline robotic controls, highlighting future opportunities for improved string-crossing efficiency and sound quality.
>
---
#### [new 010] Locomotion Beyond Feet
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，旨在解决复杂地形下人形机器人全身运动问题。通过结合关键帧动画与强化学习，实现稳定、通用的运动控制。**

- **链接: [https://arxiv.org/pdf/2601.03607v1](https://arxiv.org/pdf/2601.03607v1)**

> **作者:** Tae Hoon Yang; Haochen Shi; Jiacheng Hu; Zhicong Zhang; Daniel Jiang; Weizhuo Wang; Yao He; Zhen Wu; Yuming Chen; Yifan Hou; Monroe Kennedy; Shuran Song; C. Karen Liu
>
> **备注:** Project website: https://locomotion-beyond-feet.github.io/
>
> **摘要:** Most locomotion methods for humanoid robots focus on leg-based gaits, yet natural bipeds frequently rely on hands, knees, and elbows to establish additional contacts for stability and support in complex environments. This paper introduces Locomotion Beyond Feet, a comprehensive system for whole-body humanoid locomotion across extremely challenging terrains, including low-clearance spaces under chairs, knee-high walls, knee-high platforms, and steep ascending and descending stairs. Our approach addresses two key challenges: contact-rich motion planning and generalization across diverse terrains. To this end, we combine physics-grounded keyframe animation with reinforcement learning. Keyframes encode human knowledge of motor skills, are embodiment-specific, and can be readily validated in simulation or on hardware, while reinforcement learning transforms these references into robust, physically accurate motions. We further employ a hierarchical framework consisting of terrain-specific motion-tracking policies, failure recovery mechanisms, and a vision-based skill planner. Real-world experiments demonstrate that Locomotion Beyond Feet achieves robust whole-body locomotion and generalizes across obstacle sizes, obstacle instances, and terrain sequences.
>
---
#### [new 011] An Event-Based Opto-Tactile Skin
- **分类: cs.RO**

- **简介: 该论文属于触觉传感任务，旨在开发一种基于事件的柔性触觉皮肤系统。通过集成DVS和柔性光学波导，实现高精度压力定位，并在数据稀疏时仍保持良好性能。**

- **链接: [https://arxiv.org/pdf/2601.03907v1](https://arxiv.org/pdf/2601.03907v1)**

> **作者:** Mohammadreza Koolani; Simeon Bamford; Petr Trunin; Simon F. Müller-Cleve; Matteo Lo Preti; Fulvio Mastrogiovanni; Lucia Beccai; Chiara Bartolozzi
>
> **备注:** Accepted for publication in Frontiers in Neuromorphic Engineering. 23 pages, 9 figures
>
> **摘要:** This paper presents a neuromorphic, event-driven tactile sensing system for soft, large-area skin, based on the Dynamic Vision Sensors (DVS) integrated with a flexible silicone optical waveguide skin. Instead of repetitively scanning embedded photoreceivers, this design uses a stereo vision setup comprising two DVS cameras looking sideways through the skin. Such a design produces events as changes in brightness are detected, and estimates press positions on the 2D skin surface through triangulation, utilizing Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to find the center of mass of contact events resulting from pressing actions. The system is evaluated over a 4620 mm2 probed area of the skin using a meander raster scan. Across 95 % of the presses visible to both cameras, the press localization achieved a Root-Mean-Squared Error (RMSE) of 4.66 mm. The results highlight the potential of this approach for wide-area flexible and responsive tactile sensors in soft robotics and interactive environments. Moreover, we examined how the system performs when the amount of event data is strongly reduced. Using stochastic down-sampling, the event stream was reduced to 1/1024 of its original size. Under this extreme reduction, the average localization error increased only slightly (from 4.66 mm to 9.33 mm), and the system still produced valid press localizations for 85 % of the trials. This reduction in pass rate is expected, as some presses no longer produce enough events to form a reliable cluster for triangulation. These results show that the sensing approach remains functional even with very sparse event data, which is promising for reducing power consumption and computational load in future implementations. The system exhibits a detection latency distribution with a characteristic width of 31 ms.
>
---
#### [new 012] Embedding Autonomous Agents in Resource-Constrained Robotic Platforms
- **分类: cs.RO; cs.AI**

- **简介: 论文探讨在资源受限的机器人平台上集成自主代理，解决其在动态环境中自主决策的问题。通过将AgentSpeak编程的代理与小型机器人结合，实现迷宫自主探索，验证了高效实时推理的可行性。**

- **链接: [https://arxiv.org/pdf/2601.04191v1](https://arxiv.org/pdf/2601.04191v1)**

> **作者:** Negar Halakou; Juan F. Gutierrez; Ye Sun; Han Jiang; Xueming Wu; Yilun Song; Andres Gomez
>
> **备注:** This is an open-access, author-archived version of a manuscript published in European Conference on Multi-Agent Systems 2025
>
> **摘要:** Many embedded devices operate under resource constraints and in dynamic environments, requiring local decision-making capabilities. Enabling devices to make independent decisions in such environments can improve the responsiveness of the system and reduce the dependence on constant external control. In this work, we integrate an autonomous agent, programmed using AgentSpeak, with a small two-wheeled robot that explores a maze using its own decision-making and sensor data. Experimental results show that the agent successfully solved the maze in 59 seconds using 287 reasoning cycles, with decision phases taking less than one millisecond. These results indicate that the reasoning process is efficient enough for real-time execution on resource-constrained hardware. This integration demonstrates how high-level agent-based control can be applied to resource-constrained embedded systems for autonomous operation.
>
---
#### [new 013] CoINS: Counterfactual Interactive Navigation via Skill-Aware VLM
- **分类: cs.RO**

- **简介: 该论文提出CoINS框架，解决机器人交互导航中缺乏物理能力理解的问题。通过技能感知的VLM和强化学习技能库，提升路径规划与环境操作能力。**

- **链接: [https://arxiv.org/pdf/2601.03956v1](https://arxiv.org/pdf/2601.03956v1)**

> **作者:** Kangjie Zhou; Zhejia Wen; Zhiyong Zhuo; Zike Yan; Pengying Wu; Ieng Hou U; Shuaiyang Li; Han Gao; Kang Ding; Wenhan Cao; Wei Pan; Chang Liu
>
> **备注:** 17 pages, 13 figures
>
> **摘要:** Recent Vision-Language Models (VLMs) have demonstrated significant potential in robotic planning. However, they typically function as semantic reasoners, lacking an intrinsic understanding of the specific robot's physical capabilities. This limitation is particularly critical in interactive navigation, where robots must actively modify cluttered environments to create traversable paths. Existing VLM-based navigators are predominantly confined to passive obstacle avoidance, failing to reason about when and how to interact with objects to clear blocked paths. To bridge this gap, we propose Counterfactual Interactive Navigation via Skill-aware VLM (CoINS), a hierarchical framework that integrates skill-aware reasoning and robust low-level execution. Specifically, we fine-tune a VLM, named InterNav-VLM, which incorporates skill affordance and concrete constraint parameters into the input context and grounds them into a metric-scale environmental representation. By internalizing the logic of counterfactual reasoning through fine-tuning on the proposed InterNav dataset, the model learns to implicitly evaluate the causal effects of object removal on navigation connectivity, thereby determining interaction necessity and target selection. To execute the generated high-level plans, we develop a comprehensive skill library through reinforcement learning, specifically introducing traversability-oriented strategies to manipulate diverse objects for path clearance. A systematic benchmark in Isaac Sim is proposed to evaluate both the reasoning and execution aspects of interactive navigation. Extensive simulations and real-world experiments demonstrate that CoINS significantly outperforms representative baselines, achieving a 17\% higher overall success rate and over 80\% improvement in complex long-horizon scenarios compared to the best-performing baseline
>
---
#### [new 014] Hierarchical GNN-Based Multi-Agent Learning for Dynamic Queue-Jump Lane and Emergency Vehicle Corridor Formation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于智能交通任务，旨在解决紧急车辆快速通行问题。通过构建分层GNN的多智能体强化学习框架，协调车辆形成应急通道，提升通行效率并降低碰撞风险。**

- **链接: [https://arxiv.org/pdf/2601.04177v1](https://arxiv.org/pdf/2601.04177v1)**

> **作者:** Haoran Su
>
> **备注:** 16 Pages, 5 Figures, 9 Tables, submitted to IEEE TITS
>
> **摘要:** Emergency vehicles require rapid passage through congested traffic, yet existing strategies fail to adapt to dynamic conditions. We propose a novel hierarchical graph neural network (GNN)-based multi-agent reinforcement learning framework to coordinate connected vehicles for emergency corridor formation. Our approach uses a high-level planner for global strategy and low-level controllers for trajectory execution, utilizing graph attention networks to scale with variable agent counts. Trained via Multi-Agent Proximal Policy Optimization (MAPPO), the system reduces emergency vehicle travel time by 28.3% compared to baselines and 44.6% compared to uncoordinated traffic in simulations. The design achieves near-zero collision rates (0.3%) while maintaining 81% of background traffic efficiency. Ablation and generalization studies confirm the framework's robustness across diverse scenarios. These results demonstrate the effectiveness of combining GNNs with hierarchical learning for intelligent transportation systems.
>
---
#### [new 015] PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出PointWorld，用于机器人操作的3D世界建模任务，解决如何从单张图像和动作预测3D位移的问题。通过3D点流表示动作，实现跨平台学习与实时控制。**

- **链接: [https://arxiv.org/pdf/2601.03782v1](https://arxiv.org/pdf/2601.03782v1)**

> **作者:** Wenlong Huang; Yu-Wei Chao; Arsalan Mousavian; Ming-Yu Liu; Dieter Fox; Kaichun Mo; Li Fei-Fei
>
> **摘要:** Humans anticipate, from a glance and a contemplated action of their bodies, how the 3D world will respond, a capability that is equally vital for robotic manipulation. We introduce PointWorld, a large pre-trained 3D world model that unifies state and action in a shared 3D space as 3D point flows: given one or few RGB-D images and a sequence of low-level robot action commands, PointWorld forecasts per-pixel displacements in 3D that respond to the given actions. By representing actions as 3D point flows instead of embodiment-specific action spaces (e.g., joint positions), this formulation directly conditions on physical geometries of robots while seamlessly integrating learning across embodiments. To train our 3D world model, we curate a large-scale dataset spanning real and simulated robotic manipulation in open-world environments, enabled by recent advances in 3D vision and simulated environments, totaling about 2M trajectories and 500 hours across a single-arm Franka and a bimanual humanoid. Through rigorous, large-scale empirical studies of backbones, action representations, learning objectives, partial observability, data mixtures, domain transfers, and scaling, we distill design principles for large-scale 3D world modeling. With a real-time (0.1s) inference speed, PointWorld can be efficiently integrated in the model-predictive control (MPC) framework for manipulation. We demonstrate that a single pre-trained checkpoint enables a real-world Franka robot to perform rigid-body pushing, deformable and articulated object manipulation, and tool use, without requiring any demonstrations or post-training and all from a single image captured in-the-wild. Project website at https://point-world.github.io/.
>
---
#### [new 016] Stable Language Guidance for Vision-Language-Action Models
- **分类: cs.RO; cs.CL**

- **简介: 该论文属于视觉-语言-动作模型任务，解决语言扰动导致的模型脆弱问题。提出RSS框架，通过分离视觉与语义信息提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.04052v1](https://arxiv.org/pdf/2601.04052v1)**

> **作者:** Zhihao Zhan; Yuhao Chen; Jiaying Zhou; Qinhan Lv; Hao Liu; Keze Wang; Liang Lin; Guangrun Wang
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated impressive capabilities in generalized robotic control; however, they remain notoriously brittle to linguistic perturbations. We identify a critical ``modality collapse'' phenomenon where strong visual priors overwhelm sparse linguistic signals, causing agents to overfit to specific instruction phrasings while ignoring the underlying semantic intent. To address this, we propose \textbf{Residual Semantic Steering (RSS)}, a probabilistic framework that disentangles physical affordance from semantic execution. RSS introduces two theoretical innovations: (1) \textbf{Monte Carlo Syntactic Integration}, which approximates the true semantic posterior via dense, LLM-driven distributional expansion, and (2) \textbf{Residual Affordance Steering}, a dual-stream decoding mechanism that explicitly isolates the causal influence of language by subtracting the visual affordance prior. Theoretical analysis suggests that RSS effectively maximizes the mutual information between action and intent while suppressing visual distractors. Empirical results across diverse manipulation benchmarks demonstrate that RSS achieves state-of-the-art robustness, maintaining performance even under adversarial linguistic perturbations.
>
---
#### [new 017] Towards Zero-Knowledge Task Planning via a Language-based Approach
- **分类: cs.RO**

- **简介: 该论文研究零知识任务规划问题，旨在无需任务特定知识生成执行序列。工作包括利用大语言模型分解指令、生成行为树，并在出错时动态调整，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2601.03398v1](https://arxiv.org/pdf/2601.03398v1)**

> **作者:** Liam Merz Hoffmeister; Brian Scassellati; Daniel Rakita
>
> **摘要:** In this work, we introduce and formalize the Zero-Knowledge Task Planning (ZKTP) problem, i.e., formulating a sequence of actions to achieve some goal without task-specific knowledge. Additionally, we present a first investigation and approach for ZKTP that leverages a large language model (LLM) to decompose natural language instructions into subtasks and generate behavior trees (BTs) for execution. If errors arise during task execution, the approach also uses an LLM to adjust the BTs on-the-fly in a refinement loop. Experimental validation in the AI2-THOR simulator demonstrate our approach's effectiveness in improving overall task performance compared to alternative approaches that leverage task-specific knowledge. Our work demonstrates the potential of LLMs to effectively address several aspects of the ZKTP problem, providing a robust framework for automated behavior generation with no task-specific setup.
>
---
#### [new 018] FIRE-VLM: A Vision-Language-Driven Reinforcement Learning Framework for UAV Wildfire Tracking in a Physics-Grounded Fire Digital Twin
- **分类: cs.RO**

- **简介: 该论文属于无人机火灾追踪任务，解决极端视觉下火灾监测问题。构建物理真实数字孪生环境，设计视觉语言引导的强化学习框架，提升火灾检测效率。**

- **链接: [https://arxiv.org/pdf/2601.03449v1](https://arxiv.org/pdf/2601.03449v1)**

> **作者:** Chris Webb; Mobin Habibpour; Mayamin Hamid Raha; Ali Reza Tavakkoli; Janice Coen; Fatemeh Afghah
>
> **摘要:** Wildfire monitoring demands autonomous systems capable of reasoning under extreme visual degradation, rapidly evolving physical dynamics, and scarce real-world training data. Existing UAV navigation approaches rely on simplified simulators and supervised perception pipelines, and lack embodied agents interacting with physically realistic fire environments. We introduce FIRE-VLM, the first end-to-end vision-language model (VLM) guided reinforcement learning (RL) framework trained entirely within a high-fidelity, physics-grounded wildfire digital twin. Built from USGS Digital Elevation Model (DEM) terrain, LANDFIRE fuel inventories, and semi-physical fire-spread solvers, this twin captures terrain-induced runs, wind-driven acceleration, smoke plume occlusion, and dynamic fuel consumption. Within this environment, a PPO agent with dual-view UAV sensing is guided by a CLIP-style VLM. Wildfire-specific semantic alignment scores, derived from a single prompt describing active fire and smoke plumes, are integrated as potential-based reward shaping signals. Our contributions are: (1) a GIS-to-simulation pipeline for constructing wildfire digital twins; (2) a VLM-guided RL agent for UAV firefront tracking; and (3) a wildfire-aware reward design that combines physical terms with VLM semantics. Across five digital-twin evaluation tasks, our VLM-guided policy reduces time-to-detection by up to 6 times, increases time-in-FOV, and is, to our knowledge, the first RL-based UAV wildfire monitoring system demonstrated in kilometer-scale, physics-grounded digital-twin fires.
>
---
#### [new 019] Generational Replacement and Learning for High-Performing and Diverse Populations in Evolvable Robots
- **分类: cs.RO**

- **简介: 该论文属于进化机器人领域，旨在解决形态与控制协同优化中的适应性差和多样性不足问题。通过结合代际替换与学习机制，提升种群多样性同时保持性能。**

- **链接: [https://arxiv.org/pdf/2601.03807v1](https://arxiv.org/pdf/2601.03807v1)**

> **作者:** K. Ege de Bruin; Kyrre Glette; Kai Olav Ellefsen
>
> **摘要:** Evolutionary Robotics offers the possibility to design robots to solve a specific task automatically by optimizing their morphology and control together. However, this co-optimization of body and control is challenging, because controllers need some time to adapt to the evolving morphology - which may make it difficult for new and promising designs to enter the evolving population. A solution to this is to add intra-life learning, defined as an additional controller optimization loop, to each individual in the evolving population. A related problem is the lack of diversity often seen in evolving populations as evolution narrows the search down to a few promising designs too quickly. This problem can be mitigated by implementing full generational replacement, where offspring robots replace the whole population. This solution for increasing diversity usually comes at the cost of lower performance compared to using elitism. In this work, we show that combining such generational replacement with intra-life learning can increase diversity while retaining performance. We also highlight the importance of performance metrics when studying learning in morphologically evolving robots, showing that evaluating according to function evaluations versus according to generations of evolution can give different conclusions.
>
---
#### [new 020] Cost-Effective Radar Sensors for Field-Based Water Level Monitoring with Sub-Centimeter Accuracy
- **分类: cs.RO**

- **简介: 该论文属于水位监测任务，旨在解决传统方法成本高、覆盖有限的问题。通过评估雷达传感器，结合统计滤波提升精度，实现低成本、高准确的水位监测。**

- **链接: [https://arxiv.org/pdf/2601.03447v1](https://arxiv.org/pdf/2601.03447v1)**

> **作者:** Anna Zavei-Boroda; J. Toby Minear; Kyle Harlow; Dusty Woods; Christoffer Heckman
>
> **备注:** 10 pages, 6 figures. Preliminary results presented as a poster at an academic conference
>
> **摘要:** Water level monitoring is critical for flood management, water resource allocation, and ecological assessment, yet traditional methods remain costly and limited in coverage. This work explores radar-based sensing as a low-cost alternative for water level estimation, leveraging its non-contact nature and robustness to environmental conditions. Commercial radar sensors are evaluated in real-world field tests, applying statistical filtering techniques to improve accuracy. Results show that a single radar sensor can achieve centimeter-scale precision with minimal calibration, making it a practical solution for autonomous water monitoring using drones and robotic platforms.
>
---
#### [new 021] Bayesian Monocular Depth Refinement via Neural Radiance Fields
- **分类: cs.CV; cs.GR; cs.LG; cs.RO**

- **简介: 该论文属于单目深度估计任务，旨在提升深度图的几何细节。通过结合NeRF与贝叶斯融合，改进现有方法平滑过度的问题。**

- **链接: [https://arxiv.org/pdf/2601.03869v1](https://arxiv.org/pdf/2601.03869v1)**

> **作者:** Arun Muthukkumar
>
> **备注:** IEEE 8th International Conference on Algorithms, Computing and Artificial Intelligence (ACAI 2025). Oral presentation; Best Presenter Award
>
> **摘要:** Monocular depth estimation has applications in many fields, such as autonomous navigation and extended reality, making it an essential computer vision task. However, current methods often produce smooth depth maps that lack the fine geometric detail needed for accurate scene understanding. We propose MDENeRF, an iterative framework that refines monocular depth estimates using depth information from Neural Radiance Fields (NeRFs). MDENeRF consists of three components: (1) an initial monocular estimate for global structure, (2) a NeRF trained on perturbed viewpoints, with per-pixel uncertainty, and (3) Bayesian fusion of the noisy monocular and NeRF depths. We derive NeRF uncertainty from the volume rendering process to iteratively inject high-frequency fine details. Meanwhile, our monocular prior maintains global structure. We demonstrate superior performance on key metrics and experiments using indoor scenes from the SUN RGB-D dataset.
>
---
#### [new 022] Modeling and Control for UAV with Off-center Slung Load
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究无人机带偏心吊载的建模与控制问题，通过新模型设计分层控制器，解决耦合动态带来的运动控制难题。**

- **链接: [https://arxiv.org/pdf/2601.03386v1](https://arxiv.org/pdf/2601.03386v1)**

> **作者:** Zongyang Lv; Yanmei Jia; Yongqing Liu; Alan F. Lynch; Qing Zhao; Yuhu Wu
>
> **摘要:** Unmanned aerial vehicle (UAV) with slung load system is a classic air transportation system. In practical applications, the suspension point of the slung load does not always align with the center of mass (CoM) of the UAV due to mission requirements or mechanical interference. This offset creates coupling in the system's nonlinear dynamics which leads to a complicated motion control problem. In existing research, modeling of the system are performed about the UAV's CoM. In this work we use the point of suspension instead. Based on the new model, a cascade control strategy is developed. In the middle-loop controller, the acceleration of the suspension point is used to regulate the swing angle of the slung load without the need for considering the coupling between the slung load and the UAV. Using the off-center reference frame, an inner-loop controller is designed to track the UAV's attitude without the need of simplification on the coupling effects. We prove local exponential stability of the closed-loop using Lyapunov approach. Finally, simulations and experiments are conducted to validate the proposed control system.
>
---
#### [new 023] A Reinforcement Learning-Based Model for Mapping and Goal-Directed Navigation Using Multiscale Place Fields
- **分类: cs.NE; cs.AI; cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决复杂环境下的自主导航问题。通过多尺度场所场模型和奖励机制，提升路径效率与学习速度。**

- **链接: [https://arxiv.org/pdf/2601.03520v1](https://arxiv.org/pdf/2601.03520v1)**

> **作者:** Bekarys Dukenbaev; Andrew Gerstenslager; Alexander Johnson; Ali A. Minai
>
> **备注:** 11 pages, 8 figures. Submitted to IEEE Transactions on Cognitive and Developmental Systems
>
> **摘要:** Autonomous navigation in complex and partially observable environments remains a central challenge in robotics. Several bio-inspired models of mapping and navigation based on place cells in the mammalian hippocampus have been proposed. This paper introduces a new robust model that employs parallel layers of place fields at multiple spatial scales, a replay-based reward mechanism, and dynamic scale fusion. Simulations show that the model improves path efficiency and accelerates learning compared to single-scale baselines, highlighting the value of multiscale spatial representations for adaptive robot navigation.
>
---
#### [new 024] CageDroneRF: A Large-Scale RF Benchmark and Toolkit for Drone Perception
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出CageDroneRF，解决RF无人机检测与识别数据集稀缺问题，通过真实数据与合成增强构建大规模基准，支持分类、开集识别和目标检测任务。**

- **链接: [https://arxiv.org/pdf/2601.03302v1](https://arxiv.org/pdf/2601.03302v1)**

> **作者:** Mohammad Rostami; Atik Faysal; Hongtao Xia; Hadi Kasasbeh; Ziang Gao; Huaxia Wang
>
> **摘要:** We present CageDroneRF (CDRF), a large-scale benchmark for Radio-Frequency (RF) drone detection and identification built from real-world captures and systematically generated synthetic variants. CDRF addresses the scarcity and limited diversity of existing RF datasets by coupling extensive raw recordings with a principled augmentation pipeline that (i) precisely controls Signal-to-Noise Ratio (SNR), (ii) injects interfering emitters, and (iii) applies frequency shifts with label-consistent bounding-box transformations for detection. This dataset spans a wide range of contemporary drone models, many unavailable in current public datasets, and acquisition conditions, derived from data collected at the Rowan University campus and within a controlled RF-cage facility. CDRF is released with interoperable open-source tools for data generation, preprocessing, augmentation, and evaluation that also operate on existing public benchmarks. CDRF enables standardized benchmarking for classification, open-set recognition, and object detection, supporting rigorous comparisons and reproducible pipelines. By releasing this comprehensive benchmark and tooling, CDRF aims to accelerate progress toward robust, generalizable RF perception models.
>
---
#### [new 025] Systematic Evaluation of Depth Backbones and Semantic Cues for Monocular Pseudo-LiDAR 3D Detection
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于单目3D目标检测任务，旨在提升单目伪LiDAR的性能。通过评估深度主干和语义线索的影响，发现深度模型和几何精度是关键。**

- **链接: [https://arxiv.org/pdf/2601.03617v1](https://arxiv.org/pdf/2601.03617v1)**

> **作者:** Samson Oseiwe Ajadalu
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Monocular 3D object detection offers a low-cost alternative to LiDAR, yet remains less accurate due to the difficulty of estimating metric depth from a single image. We systematically evaluate how depth backbones and feature engineering affect a monocular Pseudo-LiDAR pipeline on the KITTI validation split. Specifically, we compare NeWCRFs (supervised metric depth) against Depth Anything V2 Metric-Outdoor (Base) under an identical pseudo-LiDAR generation and PointRCNN detection protocol. NeWCRFs yields stronger downstream 3D detection, achieving 10.50\% AP$_{3D}$ at IoU$=0.7$ on the Moderate split using grayscale intensity (Exp~2). We further test point-cloud augmentations using appearance cues (grayscale intensity) and semantic cues (instance segmentation confidence). Contrary to the expectation that semantics would substantially close the gap, these features provide only marginal gains, and mask-based sampling can degrade performance by removing contextual geometry. Finally, we report a depth-accuracy-versus-distance diagnostic using ground-truth 2D boxes (including Ped/Cyc), highlighting that coarse depth correctness does not fully predict strict 3D IoU. Overall, under an off-the-shelf LiDAR detector, depth-backbone choice and geometric fidelity dominate performance, outweighing secondary feature injection.
>
---
#### [new 026] Choreographing a World of Dynamic Objects
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出CHORD方法，解决动态物体场景生成问题。通过视频生成模型提取运动信息，实现多物体4D动态合成，适用于机器人操作。**

- **链接: [https://arxiv.org/pdf/2601.04194v1](https://arxiv.org/pdf/2601.04194v1)**

> **作者:** Yanzhe Lyu; Chen Geng; Karthik Dharmarajan; Yunzhi Zhang; Hadi Alzayer; Shangzhe Wu; Jiajun Wu
>
> **摘要:** Dynamic objects in our physical 4D (3D + time) world are constantly evolving, deforming, and interacting with other objects, leading to diverse 4D scene dynamics. In this paper, we present a universal generative pipeline, CHORD, for CHOReographing Dynamic objects and scenes and synthesizing this type of phenomena. Traditional rule-based graphics pipelines to create these dynamics are based on category-specific heuristics, yet are labor-intensive and not scalable. Recent learning-based methods typically demand large-scale datasets, which may not cover all object categories in interest. Our approach instead inherits the universality from the video generative models by proposing a distillation-based pipeline to extract the rich Lagrangian motion information hidden in the Eulerian representations of 2D videos. Our method is universal, versatile, and category-agnostic. We demonstrate its effectiveness by conducting experiments to generate a diverse range of multi-body 4D dynamics, show its advantage compared to existing methods, and demonstrate its applicability in generating robotics manipulation policies. Project page: https://yanzhelyu.github.io/chord
>
---
## 更新

#### [replaced 001] Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出Alpamayo-R1，解决自动驾驶中长尾场景的推理与决策问题，通过融合因果推理和轨迹规划提升安全性和准确性。**

- **链接: [https://arxiv.org/pdf/2511.00088v2](https://arxiv.org/pdf/2511.00088v2)**

> **作者:** NVIDIA; :; Yan Wang; Wenjie Luo; Junjie Bai; Yulong Cao; Tong Che; Ke Chen; Yuxiao Chen; Jenna Diamond; Yifan Ding; Wenhao Ding; Liang Feng; Greg Heinrich; Jack Huang; Peter Karkus; Boyi Li; Pinyi Li; Tsung-Yi Lin; Dongran Liu; Ming-Yu Liu; Langechuan Liu; Zhijian Liu; Jason Lu; Yunxiang Mao; Pavlo Molchanov; Lindsey Pavao; Zhenghao Peng; Mike Ranzinger; Ed Schmerling; Shida Shen; Yunfei Shi; Sarah Tariq; Ran Tian; Tilman Wekel; Xinshuo Weng; Tianjun Xiao; Eric Yang; Xiaodong Yang; Yurong You; Xiaohui Zeng; Wenyuan Zhang; Boris Ivanovic; Marco Pavone
>
> **摘要:** End-to-end architectures trained via imitation learning have advanced autonomous driving by scaling model size and data, yet performance remains brittle in safety-critical long-tail scenarios where supervision is sparse and causal understanding is limited. We introduce Alpamayo-R1 (AR1), a vision-language-action model (VLA) that integrates Chain of Causation reasoning with trajectory planning for complex driving scenarios. Our approach features three key innovations: (1) the Chain of Causation (CoC) dataset, built through a hybrid auto-labeling and human-in-the-loop pipeline producing decision-grounded, causally linked reasoning traces aligned with driving behaviors; (2) a modular VLA architecture combining Cosmos-Reason, a vision-language model pre-trained for Physical AI, with a diffusion-based trajectory decoder that generates dynamically feasible trajectories in real time; (3) a multi-stage training strategy using supervised fine-tuning to elicit reasoning and reinforcement learning (RL) to enforce reasoning-action consistency and optimize reasoning quality. AR1 achieves up to a 12% improvement in planning accuracy on challenging cases compared to a trajectory-only baseline, with a 35% reduction in close encounter rate in closed-loop simulation. RL post-training improves reasoning quality by 45% and reasoning-action consistency by 37%. Model scaling from 0.5B to 7B parameters shows consistent improvements. On-vehicle road tests confirm real-time performance (99 ms latency) and successful urban deployment. By bridging interpretable reasoning with precise control, AR1 demonstrates a practical path towards Level 4 autonomous driving. Model weights are available at https://huggingface.co/nvidia/Alpamayo-R1-10B with inference code at https://github.com/NVlabs/alpamayo.
>
---
#### [replaced 002] From Human Intention to Action Prediction: Intention-Driven End-to-End Autonomous Driving
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于意图驱动的端到端自动驾驶任务，旨在解决现有系统难以理解人类高层意图的问题。工作包括构建基准数据集、提出新评估方法和两种解决方案。**

- **链接: [https://arxiv.org/pdf/2512.12302v2](https://arxiv.org/pdf/2512.12302v2)**

> **作者:** Huan Zheng; Yucheng Zhou; Tianyi Yan; Jiayi Su; Hongjun Chen; Dubing Chen; Xingtai Gui; Wencheng Han; Runzhou Tao; Zhongying Qiu; Jianfei Yang; Jianbing Shen
>
> **摘要:** While end-to-end autonomous driving has achieved remarkable progress in geometric control, current systems remain constrained by a command-following paradigm that relies on simple navigational instructions. Transitioning to genuinely intelligent agents requires the capability to interpret and fulfill high-level, abstract human intentions. However, this advancement is hindered by the lack of dedicated benchmarks and semantic-aware evaluation metrics. In this paper, we formally define the task of Intention-Driven End-to-End Autonomous Driving and present Intention-Drive, a comprehensive benchmark designed to bridge this gap. We construct a large-scale dataset featuring complex natural language intentions paired with high-fidelity sensor data. To overcome the limitations of conventional trajectory-based metrics, we introduce the Imagined Future Alignment (IFA), a novel evaluation protocol leveraging generative world models to assess the semantic fulfillment of human goals beyond mere geometric accuracy. Furthermore, we explore the solution space by proposing two distinct paradigms: an end-to-end vision-language planner and a hierarchical agent-based framework. The experiments reveal a critical dichotomy where existing models exhibit satisfactory driving stability but struggle significantly with intention fulfillment. Notably, the proposed frameworks demonstrate superior alignment with human intentions.
>
---
#### [replaced 003] OmniNav: A Unified Framework for Prospective Exploration and Visual-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决多类型导航与探索的统一问题。提出OmniNav框架，整合多种导航模式，提升成功率和泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.25687v3](https://arxiv.org/pdf/2509.25687v3)**

> **作者:** Xinda Xue; Junjun Hu; Minghua Luo; Shichao Xie; Jintao Chen; Zixun Xie; Kuichen Quan; Wei Guo; Mu Xu; Zedong Chu
>
> **摘要:** Embodied navigation presents a core challenge for intelligent robots, requiring the comprehension of visual environments, natural language instructions, and autonomous exploration. Existing models often fall short in offering a unified solution across diverse navigation paradigms, resulting in low success rates and limited generalization. We introduce OmniNav, a unified framework addressing instruct-goal, object-goal, point-goal navigation, and frontier-based exploration within a single architecture. Our approach features a lightweight, low-latency policy that accurately predicts continuous-space waypoints (coordinates and orientations). This policy surpasses action-chunk methods in precision and supports real-world deployment at control frequencies up to 5 Hz. Architecturally, OmniNav employs a fast-slow system design: a fast module generates waypoints using short-horizon visual context and subtasks, while a slow module performs deliberative planning with long-horizon observations and candidate frontiers to select subsequent subgoals and subtasks. This collaboration enhances path efficiency and maintains trajectory coherence, particularly in exploration and memory-intensive scenarios. Crucially, we identify that the primary bottleneck isn't merely navigation policy learning, but a robust understanding of general instructions and objects. To boost generalization, OmniNav integrates large-scale, general-purpose training datasets, including those for image captioning and visual recognition, into a joint multi-task regimen. This significantly improves success rates and robustness. Extensive experiments confirm OmniNav's state-of-the-art performance across various navigation benchmarks, with real-world deployment further validating its efficacy. OmniNav provides practical insights for embodied navigation, charting a scalable path towards versatile, highly generalizable robotic intelligence.
>
---
#### [replaced 004] The Combined Problem of Online Task Assignment and Lifelong Path Finding in Logistics Warehouses: Rule-Based Systems Matter
- **分类: cs.MA; cs.RO**

- **简介: 该论文研究物流仓库中在线任务分配与长期路径规划的联合问题，旨在提升系统效率。通过规则系统与自动化搜索方法，优化任务分配与路径规划，提高时间与经济效率。**

- **链接: [https://arxiv.org/pdf/2502.07332v2](https://arxiv.org/pdf/2502.07332v2)**

> **作者:** Fengming Zhu; Weijia Xu; Yifei Guo; Fangzhen Lin
>
> **备注:** In Proceedings ICLP 2025, arXiv:2601.00047
>
> **摘要:** We study the combined problem of online task assignment and lifelong path finding, which is crucial for the logistics industries. However, most literature either (1) focuses on lifelong path finding assuming a given task assigner, or (2) studies the offline version of this problem where tasks are known in advance. We argue that, to maximize the system throughput, the online version that integrates these two components should be tackled directly. To this end, we introduce a formal framework of the combined problem and its solution concept. Then, we design a rule-based lifelong planner under a practical robot model that works well even in environments with severe local congestion. Upon that, we automate the search for the task assigner with respect to the underlying path planner. Simulation experiments conducted in warehouse scenarios at Meituan, one of the largest shopping platforms in China, demonstrate that (a)in terms of time efficiency, our system requires only 83.77% of the execution time needed for the currently deployed system at Meituan, outperforming other SOTA algorithms by 8.09%; (b)in terms of economic efficiency, ours can achieve the same throughput with only 60% of the agents currently in use. The code and demos are available at https://github.com/Fernadoo/Online-TAPF.
>
---
#### [replaced 005] Real-time Sampling-based Model Predictive Control based on Reverse Kullback-Leibler Divergence and Its Adaptive Acceleration
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决采样模型预测控制实时性差的问题。提出基于反Kullback-Leibler散度的方法，提升收敛性和实时性。**

- **链接: [https://arxiv.org/pdf/2212.04298v2](https://arxiv.org/pdf/2212.04298v2)**

> **作者:** Taisuke Kobayashi; Kota Fukumoto
>
> **备注:** 18 pages, 16 figures
>
> **摘要:** Sampling-based model predictive control (MPC) has the potential for use in a wide variety of robotic systems. However, its unstable updates and poor convergence render it unsuitable for real-time control of robotic systems. This study addresses this challenge with a novel approach from reverse Kullback-Leibler divergence, which has a mode-seeking property and is likely to find one of the locally optimal solutions early. Using this approach, a weighted maximum likelihood estimation with positive and negative weights is obtained and solved using the mirror descent (MD) algorithm. Negative weights eliminate unnecessary actions, but a practical implementation needs to be designed to avoid interference with positive and negative updates based on rejection sampling. In addition, Nesterov's acceleration method for the proposed MD is modified to improve heuristic step size adaptive to the noise estimated in update amounts. Real-time simulations show that the proposed method can solve a wider variety of tasks statistically than the conventional method. In addition, higher degrees-of-freedom tasks can be solved by the improved acceleration even with a CPU only. The real-world applicability of the proposed method is also demonstrated by optimizing the operability in a variable impedance control of a force-driven mobile robot. https://youtu.be/D8bFMzct1XM
>
---
#### [replaced 006] Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人强化学习任务，解决离线模型基础RL在真实机器人中应用的挑战。提出RWM-U模型，结合MOPO-PPO，提升数据效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2504.16680v2](https://arxiv.org/pdf/2504.16680v2)**

> **作者:** Chenhao Li; Andreas Krause; Marco Hutter
>
> **摘要:** Reinforcement Learning (RL) has achieved impressive results in robotics, yet high-performing pipelines remain highly task-specific, with little reuse of prior data. Offline Model-based RL (MBRL) offers greater data efficiency by training policies entirely from existing datasets, but suffers from compounding errors and distribution shift in long-horizon rollouts. Although existing methods have shown success in controlled simulation benchmarks, robustly applying them to the noisy, biased, and partially observed datasets typical of real-world robotics remains challenging. We present a principled pipeline for making offline MBRL effective on physical robots. Our RWM-U extends autoregressive world models with epistemic uncertainty estimation, enabling temporally consistent multi-step rollouts with uncertainty effectively propagated over long horizons. We combine RWM-U with MOPO-PPO, which adapts uncertainty-penalized policy optimization to the stable, on-policy PPO framework for real-world control. We evaluate our approach on diverse manipulation and locomotion tasks in simulation and on real quadruped and humanoid, training policies entirely from offline datasets. The resulting policies consistently outperform model-free and uncertainty-unaware model-based baselines, and fusing real-world data in model learning further yields robust policies that surpass online model-free baselines trained solely in simulation.
>
---
#### [replaced 007] Adaptive Anomaly Recovery for Telemanipulation: A Diffusion Model Approach to Vision-Based Tracking
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人遥操作任务，解决视觉跟踪中的异常问题。通过引入扩散模型和帧差检测技术，提升跟踪稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2503.09632v2](https://arxiv.org/pdf/2503.09632v2)**

> **作者:** Haoyang Wang; Haoran Guo; Lingfeng Tao; Zhengxiong Li
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Dexterous telemanipulation critically relies on the continuous and stable tracking of the human operator's commands to ensure robust operation. Vison-based tracking methods are widely used but have low stability due to anomalies such as occlusions, inadequate lighting, and loss of sight. Traditional filtering, regression, and interpolation methods are commonly used to compensate for explicit information such as angles and positions. These approaches are restricted to low-dimensional data and often result in information loss compared to the original high-dimensional image and video data. Recent advances in diffusion-based approaches, which can operate on high-dimensional data, have achieved remarkable success in video reconstruction and generation. However, these methods have not been fully explored in continuous control tasks in robotics. This work introduces the Diffusion-Enhanced Telemanipulation (DET) framework, which incorporates the Frame-Difference Detection (FDD) technique to identify and segment anomalies in video streams. These anomalous clips are replaced after reconstruction using diffusion models, ensuring robust telemanipulation performance under challenging visual conditions. We validated this approach in various anomaly scenarios and compared it with the baseline methods. Experiments show that DET achieves an average RMSE reduction of 17.2% compared to the cubic spline and 51.1% compared to FFT-based interpolation for different occlusion durations.
>
---
#### [replaced 008] Tackling the Kidnapped Robot Problem via Sparse Feasible Hypothesis Sampling and Reliable Batched Multi-Stage Inference
- **分类: cs.RO**

- **简介: 该论文解决机器人定位中的绑架问题，提出一种高效全局重定位框架，通过LiDAR与地图匹配实现快速可靠定位。**

- **链接: [https://arxiv.org/pdf/2511.01219v4](https://arxiv.org/pdf/2511.01219v4)**

> **作者:** Muhua Zhang; Lei Ma; Ying Wu; Kai Shen; Deqing Huang; Henry Leung
>
> **备注:** 10 pages, 8 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper addresses the Kidnapped Robot Problem (KRP), a core localization challenge of relocalizing a robot in a known map without prior pose estimate when localization loss or at SLAM initialization. For this purpose, a passive 2-D global relocalization framework is proposed. It estimates the global pose efficiently and reliably from a single LiDAR scan and an occupancy grid map while the robot remains stationary, thereby enhancing the long-term autonomy of mobile robots. The proposed framework casts global relocalization as a non-convex problem and solves it via the multi-hypothesis scheme with batched multi-stage inference and early termination, balancing completeness and efficiency. The Rapidly-exploring Random Tree (RRT), under traversability constraints, asymptotically covers the reachable space to generate sparse, uniformly distributed feasible positional hypotheses, fundamentally reducing the sampling space. The hypotheses are preliminarily ordered by the proposed Scan Mean Absolute Difference (SMAD), a coarse beam-error level metric that facilitates the early termination by prioritizing high-likelihood candidates. The SMAD computation is optimized for non-panoramic scans. The Translation-Affinity Scan-to-Map Alignment Metric (TAM) is proposed for reliable orientation selection at hypothesized positions and accurate final pose evaluation to mitigate degradation in conventional likelihood-field metrics under translational uncertainty induced by sparse hypotheses, as well as non-panoramic LiDAR scan and environmental changes. Real-world experiments on a resource-constrained mobile robot with non-panoramic LiDAR scans show that the proposed framework achieves competitive performance in both global relocalization success rate and computational efficiency.
>
---
#### [replaced 009] Physics-Driven Data Generation for Contact-Rich Manipulation via Trajectory Optimization
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 该论文属于机器人接触操作任务，解决数据生成难题。通过物理模拟与优化方法，生成高质量、跨平台的机器人操作数据，提升策略迁移能力。**

- **链接: [https://arxiv.org/pdf/2502.20382v2](https://arxiv.org/pdf/2502.20382v2)**

> **作者:** Lujie Yang; H. J. Terry Suh; Tong Zhao; Bernhard Paus Graesdal; Tarik Kelestemur; Jiuguang Wang; Tao Pang; Russ Tedrake
>
> **摘要:** We present a low-cost data generation pipeline that integrates physics-based simulation, human demonstrations, and model-based planning to efficiently generate large-scale, high-quality datasets for contact-rich robotic manipulation tasks. Starting with a small number of embodiment-flexible human demonstrations collected in a virtual reality simulation environment, the pipeline refines these demonstrations using optimization-based kinematic retargeting and trajectory optimization to adapt them across various robot embodiments and physical parameters. This process yields a diverse, physically consistent dataset that enables cross-embodiment data transfer, and offers the potential to reuse legacy datasets collected under different hardware configurations or physical parameters. We validate the pipeline's effectiveness by training diffusion policies from the generated datasets for challenging contact-rich manipulation tasks across multiple robot embodiments, including a floating Allegro hand and bimanual robot arms. The trained policies are deployed zero-shot on hardware for bimanual iiwa arms, achieving high success rates with minimal human input. Project website: https://lujieyang.github.io/physicsgen/.
>
---
#### [replaced 010] Supercomputing for High-speed Avoidance and Reactive Planning in Robots
- **分类: cs.RO; cs.DC**

- **简介: 该论文属于机器人实时避障任务，解决高速环境下机器人反应慢的问题。通过将计算任务卸载至HPC，实现毫秒级响应，提升避障效率。**

- **链接: [https://arxiv.org/pdf/2509.19486v3](https://arxiv.org/pdf/2509.19486v3)**

> **作者:** Kieran S. Lachmansingh; José R. González-Estrada; Jacob Chisholm; Ryan E. Grant; Matthew K. X. J. Pan
>
> **备注:** Error in the graph size calculation, recalculated and resubmitted
>
> **摘要:** This paper presents SHARP (Supercomputing for High-speed Avoidance and Reactive Planning), a proof-of-concept study demonstrating how high-performance computing (HPC) can enable millisecond-scale responsiveness in robotic control. While modern robots face increasing demands for reactivity in human-robot shared workspaces, onboard processors are constrained by size, power, and cost. Offloading to HPC offers massive parallelism for trajectory planning, but its feasibility for real-time robotics remains uncertain due to network latency and jitter. We evaluate SHARP in a stress-test scenario where a 7-DOF manipulator must dodge high-speed foam projectiles. Using a hash-distributed multi-goal A* search implemented with MPI on both local and remote HPC clusters, the system achieves mean planning latencies of 22.9 ms (local) and 30.0 ms (remote, ~300 km away), with avoidance success rates of 84% and 88%, respectively. These results show that when round-trip latency remains within the tens-of-milliseconds regime, HPC-side computation is no longer the bottleneck, enabling avoidance well below human reaction times. The SHARP results motivate hybrid control architectures: low-level reflexes remain onboard for safety, while bursty, high-throughput planning tasks are offloaded to HPC for scalability. By reporting per-stage timing and success rates, this study provides a reproducible template for assessing real-time feasibility of HPC-driven robotics. Collectively, SHARP reframes HPC offloading as a viable pathway toward dependable, reactive robots in dynamic environments.
>
---
#### [replaced 011] Real-time Velocity Profile Optimization for Time-Optimal Maneuvering with Generic Acceleration Constraints
- **分类: cs.RO; eess.SY; math.OC**

- **简介: 该论文属于机器人轨迹规划任务，解决在通用加速度约束下实时计算最优速度剖面的问题。提出FBGA算法，在保证精度的同时显著提升计算效率。**

- **链接: [https://arxiv.org/pdf/2509.26428v2](https://arxiv.org/pdf/2509.26428v2)**

> **作者:** Mattia Piazza; Mattia Piccinini; Sebastiano Taddei; Francesco Biral; Enrico Bertolazzi
>
> **摘要:** The computation of time-optimal velocity profiles along prescribed paths, subject to generic acceleration constraints, is a crucial problem in robot trajectory planning, with particular relevance to autonomous racing. However, the existing methods either support arbitrary acceleration constraints at high computational cost or use conservative box constraints for computational efficiency. We propose FBGA, a new \underline{F}orward-\underline{B}ackward algorithm with \underline{G}eneric \underline{A}cceleration constraints, which achieves both high accuracy and low computation time. FBGA operates forward and backward passes to maximize the velocity profile in short, discretized path segments, while satisfying user-defined performance limits. Tested on five racetracks and two vehicle classes, FBGA handles complex, non-convex acceleration constraints with custom formulations. Its maneuvers and lap times closely match optimal control baselines (within $0.11\%$-$0.36\%$), while being up to three orders of magnitude faster. FBGA maintains high accuracy even with coarse discretization, making it well-suited for online multi-query trajectory planning. Our open-source \texttt{C++} implementation is available at: https://anonymous.4open.science/r/FB_public_RAL.
>
---
