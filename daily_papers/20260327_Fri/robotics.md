# 机器人 cs.RO

- **最新发布 42 篇**

- **更新 27 篇**

## 最新发布

#### [new 001] ThermoAct:Thermal-Aware Vision-Language-Action Models for Robotic Perception and Decision-Making
- **分类: cs.RO**

- **简介: 该论文提出ThermoAct框架，将热力信息融入视觉-语言-动作模型，解决机器人感知与决策中的安全与效率问题，提升任务成功率和环境安全性。**

- **链接: [https://arxiv.org/pdf/2603.25044](https://arxiv.org/pdf/2603.25044)**

> **作者:** Young-Chae Son; Dae-Kwan Ko; Yoon-Ji Choi; Soo-Chul Lim
>
> **摘要:** In recent human-robot collaboration environments, there is a growing focus on integrating diverse sensor data beyond visual information to enable safer and more intelligent task execution. Although thermal data can be crucial for enhancing robot safety and operational efficiency, its integration has been relatively overlooked in prior research. This paper proposes a novel Vision-Language-Action (VLA) framework that incorporates thermal information for robot task execution. The proposed system leverages a Vision-Language Model (VLM) as a high-level planner to interpret complex natural language commands and decompose them into simpler sub-tasks. This approach facilitates efficient data collection and robust reasoning for complex operations. Unlike conventional methods that rely solely on visual data, our approach integrates thermal information, enabling the robot to perceive physical properties and proactively ensure environmental safety. Experimental results from real-world task scenarios validate the feasibility of our proposed framework, suggesting its potential to enhance task success rates and safety compared to existing vision-based systems.
>
---
#### [new 002] CROSS: A Mixture-of-Experts Reinforcement Learning Framework for Generalizable Large-Scale Traffic Signal Control
- **分类: cs.RO**

- **简介: 该论文属于交通信号控制任务，旨在解决大规模交通信号控制的泛化难题。通过引入Mixture-of-Experts框架和预测对比聚类模块，提升对多样化交通场景的适应能力。**

- **链接: [https://arxiv.org/pdf/2603.24930](https://arxiv.org/pdf/2603.24930)**

> **作者:** Xibei Chen; Yifeng Zhang; Yuxiang Xiao; Mingfeng Fan; Maonan Wang; Guillaume Sartoretti
>
> **摘要:** Recent advances in robotics, automation, and artificial intelligence have enabled urban traffic systems to operate with increasing autonomy towards future smart cities, powered in part by the development of adaptive traffic signal control (ATSC), which dynamically optimizes signal phases to mitigate congestion and optimize traffic. However, achieving effective and generalizable large-scale ATSC remains a significant challenge due to the diverse intersection topologies and highly dynamic, complex traffic demand patterns across the network. Existing RL-based methods typically use a single shared policy for all scenarios, whose limited representational capacity makes it difficult to capture diverse traffic dynamics and generalize to unseen environments. To address these challenges, we propose CROSS, a novel Mixture-of-Experts (MoE)-based decentralized RL framework for generalizable ATSC. We first introduce a Predictive Contrastive Clustering (PCC) module that forecasts short-term state transitions to identify latent traffic patterns, followed by clustering and contrastive learning to enhance pattern-level representation. We further design a Scenario-Adaptive MoE module that augments a shared policy with multiple experts, thus enabling adaptive specialization and more flexible scenario-specific strategies. We conduct extensive experiments in the SUMO simulator on both synthetic and real-world traffic datasets. Compared with state-of-the-art baselines, CROSS achieves superior performance and generalization through improved representation of diverse traffic scenarios.
>
---
#### [new 003] Learning Rollout from Sampling:An R1-Style Tokenized Traffic Simulation Model
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于交通模拟任务，旨在提升自动驾驶评估的多样性与真实性。通过引入基于运动标记熵的强化学习方法，解决传统方法探索不足的问题，实现更安全、多样的多智能体行为模拟。**

- **链接: [https://arxiv.org/pdf/2603.24989](https://arxiv.org/pdf/2603.24989)**

> **作者:** Ziyan Wang; Peng Chen; Ding Li; Chiwei Li; Qichao Zhang; Zhongpu Xia; Guizhen Yu
>
> **摘要:** Learning diverse and high-fidelity traffic simulations from human driving demonstrations is crucial for autonomous driving evaluation. The recent next-token prediction (NTP) paradigm, widely adopted in large language models (LLMs), has been applied to traffic simulation and achieves iterative improvements via supervised fine-tuning (SFT). However, such methods limit active exploration of potentially valuable motion tokens, particularly in suboptimal regions. Entropy patterns provide a promising perspective for enabling exploration driven by motion token uncertainty. Motivated by this insight, we propose a novel tokenized traffic simulation policy, R1Sim, which represents an initial attempt to explore reinforcement learning based on motion token entropy patterns, and systematically analyzes the impact of different motion tokens on simulation outcomes. Specifically, we introduce an entropy-guided adaptive sampling mechanism that focuses on previously overlooked motion tokens with high uncertainty yet high potential. We further optimize motion behaviors using Group Relative Policy Optimization (GRPO), guided by a safety-aware reward design. Overall, these components enable a balanced exploration-exploitation trade-off through diverse high-uncertainty sampling and group-wise comparative estimation, resulting in realistic, safe, and diverse multi-agent behaviors. Extensive experiments on the Waymo Sim Agent benchmark demonstrate that R1Sim achieves competitive performance compared to state-of-the-art methods.
>
---
#### [new 004] Persistent Robot World Models: Stabilizing Multi-Step Rollouts via Reinforcement Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉预测任务，解决多步预测中误差累积导致的视觉质量下降问题。通过强化学习优化模型，提升长期预测的稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.25685](https://arxiv.org/pdf/2603.25685)**

> **作者:** Jai Bardhan; Patrik Drozdik; Josef Sivic; Vladimir Petrik
>
> **备注:** 34 pages, 11 figures, 12 tables
>
> **摘要:** Action-conditioned robot world models generate future video frames of the manipulated scene given a robot action sequence, offering a promising alternative for simulating tasks that are difficult to model with traditional physics engines. However, these models are optimized for short-term prediction and break down when deployed autoregressively: each predicted clip feeds back as context for the next, causing errors to compound and visual quality to rapidly degrade. We address this through the following contributions. First, we introduce a reinforcement learning (RL) post-training scheme that trains the world model on its own autoregressive rollouts rather than on ground-truth histories. We achieve this by adapting a recent contrastive RL objective for diffusion models to our setting and show that its convergence guarantees carry over exactly. Second, we design a training protocol that generates and compares multiple candidate variable-length futures from the same rollout state, reinforcing higher-fidelity predictions over lower-fidelity ones. Third, we develop efficient, multi-view visual fidelity rewards that combine complementary perceptual metrics across camera views and are aggregated at the clip level for dense, low-variance training signal. Fourth, we show that our approach establishes a new state-of-the-art for rollout fidelity on the DROID dataset, outperforming the strongest baseline on all metrics (e.g., LPIPS reduced by 14% on external cameras, SSIM improved by 9.1% on the wrist camera), winning 98% of paired comparisons, and achieving an 80% preference rate in a blind human study.
>
---
#### [new 005] Visualizing Impedance Control in Augmented Reality for Teleoperation: Design and User Evaluation
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决无触觉反馈的远程操作中力控制不足的问题。通过AR可视化阻抗控制器，提升操作者对接触力的感知与调节能力。**

- **链接: [https://arxiv.org/pdf/2603.25418](https://arxiv.org/pdf/2603.25418)**

> **作者:** Gijs van den Brandt; Femke van Beek; Elena Torta
>
> **备注:** 6 pages, 5 figures, submitted to IEEE RO-MAN 2026
>
> **摘要:** Teleoperation for contact-rich manipulation remains challenging, especially when using low-cost, motion-only interfaces that provide no haptic feedback. Virtual reality controllers enable intuitive motion control but do not allow operators to directly perceive or regulate contact forces, limiting task performance. To address this, we propose an augmented reality (AR) visualization of the impedance controller's target pose and its displacement from each robot end effector. This visualization conveys the forces generated by the controller, providing operators with intuitive, real-time feedback without expensive haptic hardware. We evaluate the design in a dual-arm manipulation study with 17 participants who repeatedly reposition a box with and without the AR visualization. Results show that AR visualization reduces completion time by 24% for force-critical lifting tasks, with no significant effect on sliding tasks where precise force control is less critical. These findings indicate that making the impedance target visible through AR is a viable approach to improve human-robot interaction for contact-rich teleoperation.
>
---
#### [new 006] CTS-PLL: A Robust and Anytime Framework for Collaborative Task Sequencing and Multi-Agent Path Finding
- **分类: cs.RO**

- **简介: 该论文研究协同任务调度与多智能体路径规划问题，提出CTS-PLL框架，解决复杂环境下的碰撞避免与任务完成问题，通过改进的规划方法提升成功率和解决方案质量。**

- **链接: [https://arxiv.org/pdf/2603.25121](https://arxiv.org/pdf/2603.25121)**

> **作者:** Junkai Jiang; Yitao Xu; Ruochen Li; Shaobing Xu; Jianqiang Wang
>
> **备注:** 8 pages, 5 figures, under review
>
> **摘要:** The Collaborative Task Sequencing and Multi-Agent Path Finding (CTS-MAPF) problem requires agents to accomplish sequences of tasks while avoiding collisions, posing significant challenges due to its combinatorial complexity. This work introduces CTS-PLL, a hierarchical framework that extends the configuration-based CTS-MAPF planning paradigm with two key enhancements: a lock agents detection and release mechanism leveraging a complete planning method for local re-planning, and an anytime refinement procedure based on Large Neighborhood Search (LNS). These additions ensure robustness in dense environments and enable continuous improvement of solution quality. Extensive evaluations across sparse and dense benchmarks demonstrate that CTS-PLL achieves higher success rates and solution quality compared with existing methods, while maintaining competitive runtime efficiency. Real-world robot experiments further demonstrate the feasibility of the approach in practice.
>
---
#### [new 007] Wireless bioelectronics for untethered biohybrid robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于生物机器人领域，解决无线控制生物混合机器人的问题，通过无线电信号、光学刺激和神经肌肉整合实现远程操控。**

- **链接: [https://arxiv.org/pdf/2603.24959](https://arxiv.org/pdf/2603.24959)**

> **作者:** Hiroyuki Tetsuka; Minoru Hirano
>
> **摘要:** Biohybrid robots integrate living tissues with engineered artificial structures to achieve organism-inspired actuation and behavior. A persistent challenge is delivering stimulation and control signals without relying on tethered wiring or bulky hardware immersed in cell-culture media. Wireless bioelectronics addresses this limitation by enabling the remote transfer of control signals, typically via radio-frequency magnetic fields, to locally stimulate muscle tissues at tissue-electrode interfaces. In parallel, wireless optoelectronics enables remote control of optogenetically modified, muscle-based robots by embedding light emitters that initiate muscle actuation through light-gated ion channels. Further advances incorporate neuromuscular junctions, leveraging biological signal transduction to enable selective control of multiple actuators through wireless frequency- and time-division multiplexing. This perspective article summarizes recent advances in control strategies for biohybrid robots, namely, wireless electrical stimulation, wireless optical stimulation, and neuromuscular integration. Then this describes cross-cutting design principles and highlights a future direction, namely, co-integration of neural organoid-bioelectronics toward autonomous, closed-loop biohybrid robots.
>
---
#### [new 008] FODMP: Fast One-Step Diffusion of Movement Primitives Generation for Time-Dependent Robot Actions
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出FODMP，解决机器人时间依赖动作生成速度与精度的矛盾，通过单步解码提升推理速度，适用于实时控制。**

- **链接: [https://arxiv.org/pdf/2603.24806](https://arxiv.org/pdf/2603.24806)**

> **作者:** Xirui Shi; Arya Ebrahimi; Yi Hu; Jun Jin
>
> **摘要:** Diffusion models are increasingly used for robot learning, but current designs face a clear trade-off. Action-chunking diffusion policies like ManiCM are fast to run, yet they only predict short segments of motion. This makes them reactive, but unable to capture time-dependent motion primitives, such as following a spring-damper-like behavior with built-in dynamic profiles of acceleration and deceleration. Recently, Movement Primitive Diffusion (MPD) partially addresses this limitation by parameterizing full trajectories using Probabilistic Dynamic Movement Primitives (ProDMPs), thereby enabling the generation of temporally structured motions. Nevertheless, MPD integrates the motion decoder directly into a multi-step diffusion process, resulting in prohibitively high inference latency that limits its applicability in real-time control settings. We propose FODMP (Fast One-step Diffusion of Movement Primitives), a new framework that distills diffusion models into the ProDMPs trajectory parameter space and generates motion using a single-step decoder. FODMP retains the temporal structure of movement primitives while eliminating the inference bottleneck through single-step consistency distillation. This enables robots to execute time-dependent primitives at high inference speed, suitable for closed-loop vision-based control. On standard manipulation benchmarks (MetaWorld, ManiSkill), FODMP runs up to 10 times faster than MPD and 7 times faster than action-chunking diffusion policies, while matching or exceeding their success rates. Beyond speed, by generating fast acceleration-deceleration motion primitives, FODMP allows the robot to intercept and securely catch a fast-flying ball, whereas action-chunking diffusion policy and MPD respond too slowly for real-time interception.
>
---
#### [new 009] Integrated Multi-Drone Task Allocation, Sequencing, and Optimal Trajectory Generation in Obstacle-Rich 3D Environments
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文属于多无人机任务分配与路径规划任务，解决复杂3D环境中协同作业的路径生成问题。提出IMD-TAPP框架，整合任务分配、路径排序和轨迹优化，确保安全且高效完成任务。**

- **链接: [https://arxiv.org/pdf/2603.24908](https://arxiv.org/pdf/2603.24908)**

> **作者:** Yunes Alqudsi; Murat Makaraci
>
> **备注:** Resubmission following accepted appeal (MOD-78958). Resubmitting to cs.RO with cross-lists cs.MA and cs.AI as advised by arXiv Support
>
> **摘要:** Coordinating teams of aerial robots in cluttered three-dimensional (3D) environments requires a principled integration of discrete mission planning-deciding which robot serves which goals and in what order -- with continuous-time trajectory synthesis that enforces collision avoidance and dynamic feasibility. This paper introduces IMD-TAPP (Integrated Multi-Drone Task Allocation and Path Planning), an end-to-end framework that jointly addresses multi-goal allocation, tour sequencing, and safe trajectory generation for quadrotor teams operating in obstacle-rich spaces. IMD--TAPP first discretizes the workspace into a 3D navigation graph and computes obstacle-aware robot-to-goal and goal-to-goal travel costs via graph-search-based pathfinding. These costs are then embedded within an Injected Particle Swarm Optimization (IPSO) scheme, guided by multiple linear assignment, to efficiently explore coupled assignment/ordering alternatives and to minimize mission makespan. Finally, the resulting waypoint tours are transformed into time-parameterized minimum-snap trajectories through a generation-and-optimization routine equipped with iterative validation of obstacle clearance and inter-robot separation, triggering re-planning when safety margins are violated. Extensive MATLAB simulations across cluttered 3D scenarios demonstrate that IMD--TAPP consistently produces dynamically feasible, collision-free trajectories while achieving competitive completion times. In a representative case study with two drones serving multiple goals, the proposed approach attains a minimum mission time of 136~s while maintaining the required safety constraints throughout execution.
>
---
#### [new 010] Fast-dVLA: Accelerating Discrete Diffusion VLA to Real-Time Performance
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，旨在解决预训练VLA模型在微调中性能提升有限且成本高的问题。通过解耦辅助任务目标，提出一种高效微调方法，提升性能并降低计算开销。**

- **链接: [https://arxiv.org/pdf/2603.25661](https://arxiv.org/pdf/2603.25661)**

> **作者:** Wenxuan Song; Jiayi Chen; Shuai Chen; Jingbo Wang; Pengxiang Ding; Han Zhao; Yikai Qin; Xinhu Zheng; Donglin Wang; Yan Wang; Haoang Li
>
> **摘要:** This paper proposes a novel approach to address the challenge that pretrained VLA models often fail to effectively improve performance and reduce adaptation costs during standard supervised finetuning (SFT). Some advanced finetuning methods with auxiliary training objectives can improve performance and reduce the number of convergence steps. However, they typically incur significant computational overhead due to the additional losses from auxiliary tasks. To simultaneously achieve the enhanced capabilities of auxiliary training with the simplicity of standard SFT, we decouple the two objectives of auxiliary task training within the parameter space, namely, enhancing general capabilities and fitting task-specific action distributions. To deliver this goal, we only need to train the model to converge on a small-scale task set using two distinct training strategies. The difference between the resulting model parameters can then be interpreted as capability vectors provided by auxiliary tasks. These vectors are then merged with pretrained parameters to form a capability-enhanced meta model. Moreover, when standard SFT is augmented with a lightweight orthogonal regularization loss, the merged model attains performance comparable to auxiliary finetuned baselines with reduced computational overhead. Experimental results demonstrate that this approach is highly effective across diverse robot tasks. Project page: this https URL
>
---
#### [new 011] A Nonvolatile Switchable-polarity EPM Valve
- **分类: cs.RO**

- **简介: 该论文属于微流控与流体控制领域，解决传统阀门依赖持续供电和固定路径的问题。提出一种非易失性可极性切换的EPM阀，实现无需持续供电的稳定流体控制。**

- **链接: [https://arxiv.org/pdf/2603.24811](https://arxiv.org/pdf/2603.24811)**

> **作者:** Bingchao Wang; Jonah Mack; Francesco Giorgio-Serchi; Adam A. Stokes
>
> **摘要:** Scalable control of pneumatic and fluidic networks remains fundamentally constrained by architectures that require continuous power input, dense external control hardware, and fixed routing topologies. Current valve arrays rely on such continuous actuation and mechanically fixed routing, imposing substantial thermal and architectural overhead. Here, we introduce the Switchable-polarity ElectroPermanent Magnet (S-EPM), a fundamentally new bistable magnetic architecture that deterministically reverses its external magnetic polarity through transient electrical excitation. By reconfiguring internal flux pathways within a composite magnet assembly, the S-EPM establishes two stable, opposing magnetic configurations without requiring sustained power. We integrate this architecture into a compact pinch-valve to robustly control pneumatic and liquid media. This state-encoded magnetic control enables logic-embedded fluidic networks, including decoders, hierarchical distribution modules, and a nonvolatile six-port routing array. These systems provide address-based routing and programmable compositional control, offering features like individual port isolation that are impossible with standard mechanically coupled rotary valves. By embedding functionality in persistent magnetic states rather than continuous power or static plumbing, this work establishes a scalable foundation for digital fluidics and autonomous laboratory platforms.
>
---
#### [new 012] System Design for Maintaining Internal State Consistency in Long-Horizon Robotic Tabletop Games
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人交互任务，旨在解决长周期桌游中状态不一致问题。通过系统设计保持感知、执行与交互状态的一致性，提升整体可靠性。**

- **链接: [https://arxiv.org/pdf/2603.25405](https://arxiv.org/pdf/2603.25405)**

> **作者:** Guangyu Zhao; Ceyao Zhang; Chengdong Ma; Tao Wu; Yiyang Song; Haoxuan Ru; Yifan Zhong; Ruilin Yan; Lingfeng Li; Ruochong Li; Yu Li; Xuyuan Han; Yun Ding; Ruizhang Jiang; Xiaochuan Zhang; Yichao Li; Yuanpei Chen; Yaodong Yang; Yitao Liang
>
> **摘要:** Long-horizon tabletop games pose a distinct systems challenge for robotics: small perceptual or execution errors can invalidate accumulated task state, propagate across decision-making modules, and ultimately derail interaction. This paper studies how to maintain internal state consistency in turn-based, multi-human robotic tabletop games through deliberate system design rather than isolated component improvement. Using Mahjong as a representative long-horizon setting, we present an integrated architecture that explicitly maintains perceptual, execution, and interaction state, partitions high-level semantic reasoning from time-critical perception and control, and incorporates verified action primitives with tactile-triggered recovery to prevent premature state corruption. We further introduce interaction-level monitoring mechanisms to detect turn violations and hidden-information breaches that threaten execution assumptions. Beyond demonstrating complete-game operation, we provide an empirical characterization of failure modes, recovery effectiveness, cross-module error propagation, and hardware-algorithm trade-offs observed during deployment. Our results show that explicit partitioning, monitored state transitions, and recovery mechanisms are critical for sustaining executable consistency over extended play, whereas monolithic or unverified pipelines lead to measurable degradation in end-to-end reliability. The proposed system serves as an empirical platform for studying system-level design principles in long-horizon, turn-based interaction.
>
---
#### [new 013] Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文属于个性化自动驾驶任务，旨在解决现有系统无法适应用户驾驶习惯和语言指令的问题。提出DMW框架，通过用户嵌入和自然语言指导实现个性化行为生成。**

- **链接: [https://arxiv.org/pdf/2603.25740](https://arxiv.org/pdf/2603.25740)**

> **作者:** Zehao Wang; Huaide Jiang; Shuaiwu Dong; Yuping Wang; Hang Qiu; Jiachen Li
>
> **备注:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026); Project website: this https URL
>
> **摘要:** Human driving behavior is inherently personal, which is shaped by long-term habits and influenced by short-term intentions. Individuals differ in how they accelerate, brake, merge, yield, and overtake across diverse situations. However, existing end-to-end autonomous driving systems either optimize for generic objectives or rely on fixed driving modes, lacking the ability to adapt to individual preferences or interpret natural language intent. To address this gap, we propose Drive My Way (DMW), a personalized Vision-Language-Action (VLA) driving framework that aligns with users' long-term driving habits and adapts to real-time user instructions. DMW learns a user embedding from our personalized driving dataset collected across multiple real drivers and conditions the policy on this embedding during planning, while natural language instructions provide additional short-term guidance. Closed-loop evaluation on the Bench2Drive benchmark demonstrates that DMW improves style instruction adaptation, and user studies show that its generated behaviors are recognizable as each driver's own style, highlighting personalization as a key capability for human-centered autonomous driving. Our data and code are available at this https URL.
>
---
#### [new 014] COIN: Collaborative Interaction-Aware Multi-Agent Reinforcement Learning for Self-Driving Systems
- **分类: cs.RO**

- **简介: 该论文属于多智能体强化学习任务，旨在解决密集城市交通中自动驾驶车辆的协同问题。提出COIN框架，通过联合优化个体与全局目标，提升安全性和效率。**

- **链接: [https://arxiv.org/pdf/2603.24931](https://arxiv.org/pdf/2603.24931)**

> **作者:** Yifeng Zhang; Jieming Chen; Tingguang Zhou; Tanishq Duhan; Jianghong Dong; Yuhong Cao; Guillaume Sartoretti
>
> **摘要:** Multi-Agent Self-Driving (MASD) systems provide an effective solution for coordinating autonomous vehicles to reduce congestion and enhance both safety and operational efficiency in future intelligent transportation systems. Multi-Agent Reinforcement Learning (MARL) has emerged as a promising approach for developing advanced end-to-end MASD systems. However, achieving efficient and safe collaboration in dynamic MASD systems remains a significant challenge in dense scenarios with complex agent interactions. To address this challenge, we propose a novel collaborative(CO-) interaction-aware(-IN) MARL framework, named COIN. Specifically, we develop a new counterfactual individual-global twin delayed deep deterministic policy gradient (CIG-TD3) algorithm, crafted in a "centralized training, decentralized execution" (CTDE) manner, which aims to jointly optimize the individual objectives (navigation) and the global objectives (collaboration) of agents. We further introduce a dual-level interaction-aware centralized critic architecture that captures both local pairwise interactions and global system-level dependencies, enabling more accurate global value estimation and improved credit assignment for collaborative policy learning. We conduct extensive simulation experiments in dense urban traffic environments, which demonstrate that COIN consistently outperforms other advanced baseline methods in both safety and efficiency across various system sizes. These results highlight its superiority in complex and dynamic MASD scenarios, as further validated through real-world robot demonstrations. Supplementary videos are available at this https URL
>
---
#### [new 015] Can Users Specify Driving Speed? Bench2Drive-Speed: Benchmark and Baselines for Desired-Speed Conditioned Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主驾驶任务，解决用户自定义车速和变道指令的问题。通过构建数据集和基准模型，研究如何在不依赖额外数据的情况下实现速度控制与变道执行。**

- **链接: [https://arxiv.org/pdf/2603.25672](https://arxiv.org/pdf/2603.25672)**

> **作者:** Yuqian Shao; Xiaosong Jia; Langechuan Liu; Junchi Yan
>
> **备注:** Project page: this https URL
>
> **摘要:** End-to-end autonomous driving (E2E-AD) has achieved remarkable progress. However, one practical and useful function has been long overlooked: users may wish to customize the desired speed of the policy or specify whether to allow the autonomous vehicle to overtake. To bridge this gap, we present Bench2Drive-Speed, a benchmark with metrics, dataset, and baselines for desired-speed conditioned autonomous driving. We introduce explicit inputs of users' desired target-speed and overtake/follow instructions to driving policy models. We design quantitative metrics, including Speed-Adherence Score and Overtake Score, to measure how faithfully policies follow user specifications, while remaining compatible with standard autonomous driving metrics. To enable training of speed-conditioned policies, one approach is to collect expert demonstrations that strictly follow speed requirements, an expensive and unscalable process in the real world. An alternative is to adapt existing regular driving data by treating the speed observed in future frames as the target speed for training. To investigate this, we construct CustomizedSpeedDataset, composed of 2,100 clips annotated with experts demonstrations, enabling systematic investigation of supervision strategies. Our experiments show that, under proper re-annotation, models trained on regular driving data perform comparably to on expert demonstrations, suggesting that speed supervision can be introduced without additional complex real-world data collection. Furthermore, we find that while target-speed following can be achieved without degrading regular driving performance, executing overtaking commands remains challenging due to the inherent difficulty of interactive behaviors. All code, datasets and baselines are available at this https URL
>
---
#### [new 016] Dissimilarity-Based Persistent Coverage Control of Multi-Robot Systems for Improving Solar Irradiance Prediction Accuracy in Solar Thermal Power Plants
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于多机器人系统任务，旨在提升太阳能热电站的辐照度预测精度。通过构建差异图并设计持续覆盖控制算法，优化传感器布局以提高预测效果。**

- **链接: [https://arxiv.org/pdf/2603.25139](https://arxiv.org/pdf/2603.25139)**

> **作者:** Haruki Kawase; Taiga Sugawara; A. Daniel Carnerero
>
> **备注:** 8 pages, 6 figures, 5 tables
>
> **摘要:** Accurate forecasting of future solar irradiance is essential for the effective control of solar thermal power plants. Although various kriging-based methods have been proposed to address the prediction problem, these methods typically do not provide an appropriate sampling strategy to dynamically position mobile sensors for optimizing prediction accuracy in real time, which is critical for achieving accurate forecasts with a minimal number of sensors. This paper introduces a dissimilarity map derived from a kriging model and proposes a persistent coverage control algorithm that effectively guides agents toward regions where additional observations are required to improve prediction performance. By means of experiments using mobile robots, the proposed approach was shown to obtain more accurate predictions than the considered baselines under various emulated irradiance fields.
>
---
#### [new 017] Towards Generalizable Robotic Data Flywheel: High-Dimensional Factorization and Composition
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决数据多样性不足和数据效率低的问题。提出F-ACIL框架，通过结构化因子分解实现高效组合泛化。**

- **链接: [https://arxiv.org/pdf/2603.25583](https://arxiv.org/pdf/2603.25583)**

> **作者:** Yuyang Xiao; Yifei Zhou; Haoran Wang; Wenxuan Ou; Yuxiao Liu
>
> **摘要:** The lack of sufficiently diverse data, coupled with limited data efficiency, remains a major bottleneck for generalist robotic models, yet systematic strategies for collecting and curating such data are not fully explored. Task diversity arises from implicit factors that are sparsely distributed across multiple dimensions and are difficult to define explicitly. To address this challenge, we propose F-ACIL, a heuristic factor-aware compositional iterative learning framework that enables structured data factorization and promotes compositional generalization. F-ACIL decomposes the data distribution into structured factor spaces such as object, action, and environment. Based on the factorized formulation, we develop a factor-wise data collection and an iterative training paradigm that promotes compositional generalization over the high-dimensional factor space, leading to more effective utilization of real-world robotic demonstrations. With extensive real-world experiments, we show that F-ACIL can achieve more than 45% performance gains with 5-10$\times$ fewer demonstrations comparing to that of which without the strategy. The results suggest that structured factorization offers a practical pathway toward efficient compositional generalization in real-world robotic learning. We believe F-ACIL can inspire more systematic research on building generalizable robotic data flywheel strategies. More demonstrations can be found at: this https URL
>
---
#### [new 018] SoftMimicGen: A Data Generation System for Scalable Robot Learning in Deformable Object Manipulation
- **分类: cs.RO**

- **简介: 该论文提出SoftMimicGen，用于柔性物体操作的自动化数据生成系统，解决真实数据难以收集的问题。**

- **链接: [https://arxiv.org/pdf/2603.25725](https://arxiv.org/pdf/2603.25725)**

> **作者:** Masoud Moghani; Mahdi Azizian; Animesh Garg; Yuke Zhu; Sean Huver; Ajay Mandlekar
>
> **摘要:** Large-scale robot datasets have facilitated the learning of a wide range of robot manipulation skills, but these datasets remain difficult to collect and scale further, owing to the intractable amount of human time, effort, and cost required. Simulation and synthetic data generation have proven to be an effective alternative to fuel this need for data, especially with the advent of recent work showing that such synthetic datasets can dramatically reduce real-world data requirements and facilitate generalization to novel scenarios unseen in real-world demonstrations. However, this paradigm has been limited to rigid-body tasks, which are easy to simulate. Deformable object manipulation encompasses a large portion of real-world manipulation and remains a crucial gap to address towards increasing adoption of the synthetic simulation data paradigm. In this paper, we introduce SoftMimicGen, an automated data generation pipeline for deformable object manipulation tasks. We introduce a suite of high-fidelity simulation environments that encompasses a wide range of deformable objects (stuffed animal, rope, tissue, towel) and manipulation behaviors (high-precision threading, dynamic whipping, folding, pick-and-place), across four robot embodiments: a single-arm manipulator, bimanual arms, a humanoid, and a surgical robot. We apply SoftMimicGen to generate datasets across the task suite, train high-performing policies from the data, and systematically analyze the data generation system. Project website: \href{this https URL}{this http URL}.
>
---
#### [new 019] Connectivity-Aware Representations for Constrained Motion Planning via Multi-Scale Contrastive Learning
- **分类: cs.RO**

- **简介: 该论文属于运动规划任务，解决约束下路径规划效率低的问题。通过多尺度对比学习构建连通性感知表示，提升规划成功率并减少时间。**

- **链接: [https://arxiv.org/pdf/2603.25298](https://arxiv.org/pdf/2603.25298)**

> **作者:** Suhyun Jeon; Yumin Lim; Woo-Jeong Baek; Hyeonseo Kim; Suhan Park; Jaeheung Park
>
> **备注:** 8 pages, 5 figures, ICRA 2026
>
> **摘要:** The objective of constrained motion planning is to connect start and goal configurations while satisfying task-specific constraints. Motion planning becomes inefficient or infeasible when the configurations lie in disconnected regions, known as essentially mutually disconnected (EMD) components. Constraints further restrict feasible space to a lower-dimensional submanifold, while redundancy introduces additional complexity because a single end-effector pose admits infinitely many inverse kinematic solutions that may form discrete self-motion manifolds. This paper addresses these challenges by learning a connectivity-aware representation for selecting start and goal configurations prior to planning. Joint configurations are embedded into a latent space through multi-scale manifold learning across neighborhood ranges from local to global, and clustering generates pseudo-labels that supervise a contrastive learning framework. The proposed framework provides a connectivity-aware measure that biases the selection of start and goal configurations in connected regions, avoiding EMDs and yielding higher success rates with reduced planning time. Experiments on various manipulation tasks showed that our method achieves 1.9 times higher success rates and reduces the planning time by a factor of 0.43 compared to baselines.
>
---
#### [new 020] MMaDA-VLA: Large Diffusion Vision-Language-Action Model with Unified Multi-Modal Instruction and Generation
- **分类: cs.RO**

- **简介: 该论文提出MMaDA-VLA模型，解决机器人操作中视觉-语言-动作对齐问题，通过统一多模态生成框架提升长期一致性与任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.25406](https://arxiv.org/pdf/2603.25406)**

> **作者:** Yang Liu; Pengxiang Ding; Tengyue Jiang; Xudong Wang; Wenxuan Song; Minghui Lin; Han Zhao; Hongyin Zhang; Zifeng Zhuang; Wei Zhao; Siteng Huang; Jinkui Shi; Donglin Wang
>
> **摘要:** Vision-Language-Action (VLA) models aim to control robots for manipulation from visual observations and natural-language instructions. However, existing hierarchical and autoregressive paradigms often introduce architectural overhead, suffer from temporal inconsistency and long-horizon error accumulation, and lack a mechanism to capture environment dynamics without extra modules. To this end, we present MMaDA-VLA, a fully native pre-trained large diffusion VLA model that unifies multi-modal understanding and generation in a single framework. Our key idea is a native discrete diffusion formulation that embeds language, images, and continuous robot controls into one discrete token space and trains a single backbone with masked token denoising to jointly generate a future goal observation and an action chunk in parallel. Iterative denoising enables global, order-free refinement, improving long-horizon consistency while grounding actions in predicted future visual outcomes without auxiliary world models. Experiments across simulation benchmarks and real-world tasks show state-of-the-art performance, achieving 98.0% average success on LIBERO and 4.78 average length on CALVIN.
>
---
#### [new 021] Saranga: MilliWatt Ultrasound for Navigation in Visually Degraded Environments on Palm-Sized Aerial Robots
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决视觉受限环境下微型飞行器的障碍物感知问题。提出低功耗超声感知系统Saranga，结合物理降噪和深度学习方法提升探测性能。**

- **链接: [https://arxiv.org/pdf/2603.24699](https://arxiv.org/pdf/2603.24699)**

> **作者:** Manoj Velmurugan; Phillip Brush; Colin Balfour; Richard J. Przybyla; Nitin J. Sanket
>
> **摘要:** Tiny palm-sized aerial robots possess exceptional agility and cost-effectiveness in navigating confined and cluttered environments. However, their limited payload capacity directly constrains the sensing suite on-board the robot, thereby limiting critical navigational tasks in Global Positioning System (GPS)-denied wild scenes. Common methods for obstacle avoidance use cameras and LIght Detection And Ranging (LIDAR), which become ineffective in visually degraded conditions such as low visibility, dust, fog or darkness. Other sensors, such as RAdio Detection And Ranging (RADAR), have high power consumption, making them unsuitable for tiny aerial robots. Inspired by bats, we propose Saranga, a low-power ultrasound-based perception stack that localizes obstacles using a dual sonar array. We present two key solutions to combat the low Peak Signal-to-Noise Ratio of $-4.9$ decibels: physical noise reduction and a deep learning based denoising method. Firstly, we present a practical way to block propeller induced ultrasound noise on the weak echoes. The second solution is to train a neural network to utilize the \textcolor{black}{long horizon of ultrasound echoes} for finding signal patterns under high amounts of uncorrelated noise where classical methods were insufficient. We generalize to the real world by using a synthetic data generation pipeline and limited real noise data for training. We enable a palm-sized aerial robot to navigate in visually degraded conditions of dense fog, darkness, and snow in a cluttered environment with thin and transparent obstacles using only on-board sensing and computation. We provide extensive real world results to demonstrate the efficacy of our approach.
>
---
#### [new 022] A Mentalistic Interface for Probing Folk-Psychological Attribution to Non-Humanoid Robots
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于人机交互任务，旨在研究人们对非人形机器人的意图归因。通过构建实验平台，控制行为并改变解释框架，探讨语言和框架如何影响人类的意图立场判断。**

- **链接: [https://arxiv.org/pdf/2603.25646](https://arxiv.org/pdf/2603.25646)**

> **作者:** Giulio Pisaneschi; Pierpaolo Serio; Estelle Gerbier; Andrea Dan Ryals; Lorenzo Pollini; Mario G. C. A. Cimino
>
> **备注:** Preprint submitted to IEEE. 8 pages, 21 figures
>
> **摘要:** This paper presents an experimental platform for studying intentional-state attribution toward a non-humanoid robot. The system combines a simulated robot, realistic task environments, and large language model-based explanatory layers that can express the same behavior in mentalistic, teleological, or mechanistic terms. By holding behavior constant while varying the explanatory frame, the platform provides a controlled way to investigate how language and framing shape the adoption of the intentional stance in robotics.
>
---
#### [new 023] SafeGuard ASF: SR Agentic Humanoid Robot System for Autonomous Industrial Safety
- **分类: cs.RO**

- **简介: 该论文属于工业安全任务，旨在解决无人工厂中的自主危险检测问题。通过人形机器人系统实现火灾、温度异常和入侵者的自动识别与响应。**

- **链接: [https://arxiv.org/pdf/2603.25353](https://arxiv.org/pdf/2603.25353)**

> **作者:** Thanh Nguyen Canh; Thang Tran Viet; Thanh Tuan Tran; Ben Wei Lim
>
> **摘要:** The rise of unmanned ``dark factories'' operating without human presence demands autonomous safety systems capable of detecting and responding to multiple hazard types. We present SafeGuard ASF (Agentic Security Fleet), a comprehensive framework deploying humanoid robots for autonomous hazard detection in industrial environments. Our system integrates multi-modal perception (RGB-D imaging), a ReAct-based agentic reasoning framework, and learned locomotion policies on the Unitree G1 humanoid platform. We address three critical hazard scenarios: fire and smoke detection, abnormal temperature monitoring in pipelines, and intruder detection in restricted zones. Our perception pipeline achieves 94.2% mAP for fire or smoke detection with 127ms latency. We train multiple locomotion policies, including dance motion tracking and velocity control, using Unitree RL Lab with PPO, demonstrating stable convergence within 80,000 training iterations. We validate our system in both simulation and real-world environments, demonstrating autonomous patrol, human detection with visual perception, and obstacle avoidance capabilities. The proposed ToolOrchestra action framework enables structured decision-making through perception, reasoning, and actuation tools.
>
---
#### [new 024] A Minimum-Energy Control Approach for Redundant Mobile Manipulators in Physical Human-Robot Interaction Applications
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于人机交互任务，旨在解决冗余移动机械臂的能量优化问题。通过提出最小化整体动能的控制方法，提升系统性能。**

- **链接: [https://arxiv.org/pdf/2603.25259](https://arxiv.org/pdf/2603.25259)**

> **作者:** Davide Tebaldi; Niccolò Paradisi; Fabio Pini; Luigi Biagiotti
>
> **摘要:** Research on mobile manipulation systems that physically interact with humans has expanded rapidly in recent years, opening the way to tasks which could not be performed using fixed-base manipulators. Within this context, developing suitable control methodologies is essential since mobile manipulators introduce additional degrees of freedom, making the design of control approaches more challenging and more prone to performance optimization. This paper proposes a control approach for a mobile manipulator, composed of a mobile base equipped with a robotic arm mounted on the top, with the objective of minimizing the overall kinetic energy stored in the whole-body mobile manipulator in physical human-robot interaction applications. The approach is experimentally tested with reference to a peg-in-hole task, and the results demonstrate that the proposed approach reduces the overall kinetic energy stored in the whole-body robotic system and improves the system performance compared with the benchmark method.
>
---
#### [new 025] Bayesian Learning-Enhanced Navigation with Deep Smoothing for Inertial-Aided Navigation
- **分类: cs.RO**

- **简介: 该论文属于导航任务，解决惯性导航系统与GNSS融合中的位置偏差问题。提出BLENDS框架，结合贝叶斯学习与深度平滑，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2603.25364](https://arxiv.org/pdf/2603.25364)**

> **作者:** Nadav Cohen; Itzik Klein
>
> **摘要:** Accurate post-processing navigation is essential for applications such as survey and mapping, where the full measurement history can be exploited to refine past state estimates. Fixed-interval smoothing algorithms represent the theoretically optimal solution under Gaussian assumptions. However, loosely coupled INS/GNSS systems fundamentally inherit the systematic position bias of raw GNSS measurements, leaving a persistent accuracy gap that model-based smoothers cannot resolve. To address this limitation, we propose BLENDS, which integrates Bayesian learning with deep smoothing to enhance navigation performance. BLENDS is a a data-driven post-processing framework that augments the classical two-filter smoother with a transformer-based neural network. It learns to modify the filter covariance matrices and apply an additive correction to the smoothed error-state directly within the Bayesian framework. A novel Bayesian-consistent loss jointly supervises the smoothed mean and covariance, enforcing minimum-variance estimates while maintaining statistical consistency. BLENDS is evaluated on two real-world datasets spanning a mobile robot and a quadrotor. Across all unseen test trajectories, BLENDS achieves horizontal position improvements of up to 63% over the baseline forward EKF.
>
---
#### [new 026] UMBRELLA: Uncertainty-aware Multi-robot Reactive Coordination under Dynamic Temporal Logic Tasks
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究多机器人协同任务协调问题，解决动态目标下的不确定性与时间约束。提出UMBRELLA框架，结合蒙特卡洛树搜索与置信预测，优化任务执行效率。**

- **链接: [https://arxiv.org/pdf/2603.25395](https://arxiv.org/pdf/2603.25395)**

> **作者:** Qisheng Zhao; Meng Guo; Hengxuan Du; Lars Lindemann; Zhongkui Li
>
> **摘要:** Multi-robot systems can be extremely efficient for accomplishing team-wise tasks by acting concurrently and collaboratively. However, most existing methods either assume static task features or simply replan when environmental changes occur. This paper addresses the challenging problem of coordinating multi-robot systems for collaborative tasks involving dynamic and moving targets. We explicitly model the uncertainty in target motion prediction via Conformal Prediction(CP), while respecting the spatial-temporal constraints specified by Linear Temporal Logic (LTL). The proposed framework (UMBRELLA) combines the Monte Carlo Tree Search (MCTS) over partial plans with uncertainty-aware rollouts, and introduces a CP-based metric to guide and accelerate the search. The objective is to minimize the Conditional Value at Risk (CVaR) of the average makespan. For tasks released online, a receding-horizon planning scheme dynamically adjusts the assignments based on updated task specifications and motion predictions. Spatial and temporal constraints among the tasks are always ensured, and only partial synchronization is required for the collaborative tasks during online execution. Extensive large-scale simulations and hardware experiments demonstrate substantial reductions in both the average makespan and its variance by 23% and 71%, compared with static baselines.
>
---
#### [new 027] $π$, But Make It Fly: Physics-Guided Transfer of VLA Models to Aerial Manipulation
- **分类: cs.RO**

- **简介: 该论文研究将视觉-语言-动作模型迁移至空中操作任务，解决动力学差异与数据不足问题，通过物理引导和数据合成提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.25038](https://arxiv.org/pdf/2603.25038)**

> **作者:** Johnathan Tucker; Denis Liu; Aiden Swann; Allen Ren; Javier Yu; Jiankai Sun; Brandon Kim; Lachlain McGranahan; Quan Vuong; Mac Schwager
>
> **摘要:** Vision-Language-Action (VLA) models such as $\pi_0$ have demonstrated remarkable generalization across diverse fixed-base manipulators. However, transferring these foundation models to aerial platforms remains an open challenge due to the fundamental mismatch between the quasi-static dynamics of fixed-base arms and the underactuated, highly dynamic nature of flight. In this work, we introduce AirVLA, a system that investigates the transferability of manipulation-pretrained VLAs to aerial pick-and-place tasks. We find that while visual representations transfer effectively, the specific control dynamics required for flight do not. To bridge this "dynamics gap" without retraining the foundation model, we introduce a Payload-Aware Guidance mechanism that injects payload constraints directly into the policy's flow-matching sampling process. To overcome data scarcity, we further utilize a Gaussian Splatting pipeline to synthesize navigation training data. We evaluate our method through a cumulative 460 real-world experiments which demonstrate that this synthetic data is a key enabler of performance, unlocking 100% success in navigation tasks where directly fine-tuning on teleoperation data alone attains 81% success. Our inference-time intervention, Payload-Aware Guidance, increases real-world pick-and-place task success from 23% to 50%. Finally, we evaluate the model on a long-horizon compositional task, achieving a 62% overall success rate. These results suggest that pre-trained manipulation VLAs, with appropriate data augmentation and physics-informed guidance, can transfer to aerial manipulation and navigation, as well as the composition of these tasks.
>
---
#### [new 028] Temporally Decoupled Diffusion Planning for Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶运动规划任务，解决动态环境中短期与长期决策的平衡问题。提出TDDM模型，通过分段噪声处理实现时空解耦，提升轨迹生成效果。**

- **链接: [https://arxiv.org/pdf/2603.25462](https://arxiv.org/pdf/2603.25462)**

> **作者:** Xiang Li; Bikun Wang; John Zhang; Jianjun Wang
>
> **备注:** icaps
>
> **摘要:** Motion planning in dynamic urban environments requires balancing immediate safety with long-term goals. While diffusion models effectively capture multi-modal decision-making, existing approaches treat trajectories as monolithic entities, overlooking heterogeneous temporal dependencies where near-term plans are constrained by instantaneous dynamics and far-term plans by navigational goals. To address this, we propose Temporally Decoupled Diffusion Model (TDDM), which reformulates trajectory generation via a noise-as-mask paradigm. By partitioning trajectories into segments with independent noise levels, we implicitly treat high noise as information voids and weak noise as contextual cues. This compels the model to reconstruct corrupted near-term states by leveraging internal correlations with better-preserved temporal contexts. Architecturally, we introduce a Temporally Decoupled Adaptive Layer Normalization (TD-AdaLN) to inject segment-specific timesteps. During inference, our Asymmetric Temporal Classifier-Free Guidance utilizes weakly noised far-term priors to guide immediate path generation. Evaluations on the nuPlan benchmark show TDDM approaches or exceeds state-of-the-art baselines, particularly excelling in the challenging Test14-hard subset.
>
---
#### [new 029] Characterization of Constraints in Flexible Unknown Environments
- **分类: cs.RO**

- **简介: 该论文属于自主操作任务，解决未知环境中柔性约束物体的安全路径规划问题。通过实时识别约束和刚度，实现安全操作与探索。**

- **链接: [https://arxiv.org/pdf/2603.24813](https://arxiv.org/pdf/2603.24813)**

> **作者:** Samrat Bhattacharyya; Nabil Simaan
>
> **摘要:** This paper presents an online path planning algorithm for safe autonomous manipulation of a flexibly constrained object in an unknown environment. Methods for real time identification and characterization of perceived flexible constraints and global stiffness are presented. Used in tandem, these methods allow a robot to simultaneously explore, characterize, and manipulate an elastic system safely. Navigation without a-priori knowledge of the system is achieved using constraint exploration based on local force and position information. The perceived constraint stiffness is considered at multiple poses along an object's (system) trajectory. Using stiffness eigenvector information, global stiffness behavior is characterized and identified using an atlas of simple mechanical constraints, such as hinges and planar constraints. Validation of these algorithms is carried out by simulation and experimentally. The ability to recognize several common simple mechanical constraints (such as a flexible hinge) in real time, and to subsequently identify relevant screw parameters is demonstrated. These results suggest the feasibility of simultaneous global constrain/stiffness exploration and safe manipulation of flexibly constrained objects. We believe that this approach will eventually enable safe cooperative manipulation in applications such as organ retraction and manipulation during surgery
>
---
#### [new 030] Intelligent Navigation and Obstacle-Aware Fabrication for Mobile Additive Manufacturing Systems
- **分类: cs.RO**

- **简介: 该论文属于移动制造任务，解决MAMbots在动态环境中的导航与打印问题，提出集成导航与制造的实时控制框架，提升适应性与精度。**

- **链接: [https://arxiv.org/pdf/2603.25688](https://arxiv.org/pdf/2603.25688)**

> **作者:** Yifei Li; Ruizhe Fu; Huihang Liu; Guha Manogharan; Feng Ju; Ilya Kovalenko
>
> **备注:** 8 pages, 4 figures, conference
>
> **摘要:** As the demand for mass customization increases, manufacturing systems must become more flexible and adaptable to produce personalized products efficiently. Additive manufacturing (AM) enhances production adaptability by enabling on-demand fabrication of customized components directly from digital models, but its flexibility remains constrained by fixed equipment layouts. Integrating mobile robots addresses this limitation by allowing manufacturing resources to move and adapt to changing production requirements. Mobile AM Robots (MAMbots) combine AM with mobile robotics to produce and transport components within dynamic manufacturing environments. However, the dynamic manufacturing environments introduce challenges for MAMbots. Disturbances such as obstacles and uneven terrain can disrupt navigation stability, which in turn affects printing accuracy and surface quality. This work proposes a universal mobile printing-and-delivery platform that couples navigation and material deposition, addressing the limitations of earlier frameworks that treated these processes separately. A real-time control framework is developed to plan and control the robot's navigation, ensuring safe motion, obstacle avoidance, and path stability while maintaining print quality. The closed-loop integration of sensing, mobility, and manufacturing provides real-time feedback for motion and process control, enabling MAMbots to make autonomous decisions in dynamic environments. The framework is validated through simulations and real-world experiments that test its adaptability to trajectory variations and external disturbances. Coupled navigation and printing together enable MAMbots to plan safe, adaptive trajectories, improving flexibility and adaptability in manufacturing.
>
---
#### [new 031] IntentReact: Guiding Reactive Object-Centric Navigation via Topological Intent
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，解决机器人在部分可观测环境下有效导航的问题。提出IntentReact框架，结合全局拓扑意图与局部反应控制，提升导航效果。**

- **链接: [https://arxiv.org/pdf/2603.25382](https://arxiv.org/pdf/2603.25382)**

> **作者:** Yanmei Jiao; Anpeng Lu; Wenhan Hu; Rong Xiong; Yue Wang; Huajin Tang; Wen-an Zhang
>
> **摘要:** Object-goal visual navigation requires robots to reason over semantic structure and act effectively under partial observability. Recent approaches based on object-level topological maps enable long-horizon navigation without dense geometric reconstruction, but their execution remains limited by the gap between global topological guidance and local perception-driven control. In particular, local decisions are made solely from the current egocentric observation, without access to information beyond the robot's field of view. As a result, the robot may persist along its current heading even when initially oriented away from the goal, moving toward directions that do not decrease the global topological distance. In this work, we propose IntentReact, an intent-conditioned object-centric navigation framework that introduces a compact interface between global topological planning and reactive object-centric control. Our approach encodes global topological guidance as a low-dimensional directional signal, termed intent, which conditions a learned waypoint prediction policy to bias navigation toward topologically consistent progression. This design enables the robot to promptly reorient when local observations are misleading, guiding motion toward directions that decrease global topological distance while preserving the reactivity and robustness of object-centric control. We evaluate the proposed framework through extensive experiments, demonstrating improved navigation success and execution quality compared to prior object-centric navigation methods.
>
---
#### [new 032] SABER: A Stealthy Agentic Black-Box Attack Framework for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出SABER，一种针对视觉-语言-动作模型的隐蔽攻击框架，解决指令篡改导致机器人行为异常的问题，通过生成有效且微小的文本修改来降低任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.24935](https://arxiv.org/pdf/2603.24935)**

> **作者:** Xiyang Wu; Guangyao Shi; Qingzi Wang; Zongxia Li; Amrit Singh Bedi; Dinesh Manocha
>
> **摘要:** Vision-language-action (VLA) models enable robots to follow natural-language instructions grounded in visual observations, but the instruction channel also introduces a critical vulnerability: small textual perturbations can alter downstream robot behavior. Systematic robustness evaluation therefore requires a black-box attacker that can generate minimal yet effective instruction edits across diverse VLA models. To this end, we present SABER, an agent-centric approach for automatically generating instruction-based adversarial attacks on VLA models under bounded edit budgets. SABER uses a GRPO-trained ReAct attacker to generate small, plausible adversarial instruction edits using character-, token-, and prompt-level tools under a bounded edit budget that induces targeted behavioral degradation, including task failure, unnecessarily long execution, and increased constraint violations. On the LIBERO benchmark across six state-of-the-art VLA models, SABER reduces task success by 20.6%, increases action-sequence length by 55%, and raises constraint violations by 33%, while requiring 21.1% fewer tool calls and 54.7% fewer character edits than strong GPT-based baselines. These results show that small, plausible instruction edits are sufficient to substantially degrade robot execution, and that an agentic black-box pipeline offers a practical, scalable, and adaptive approach for red-teaming robotic foundation models.
>
---
#### [new 033] LILAC: Language-Conditioned Object-Centric Optical Flow for Open-Loop Trajectory Generation
- **分类: cs.RO**

- **简介: 该论文属于语言引导的机器人操作任务，解决从图像和自然语言指令生成轨迹的问题。提出LILAC模型，通过语义对齐和跨模态适配生成高质量光学流，提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2603.25481](https://arxiv.org/pdf/2603.25481)**

> **作者:** Motonari Kambara; Koki Seno; Tomoya Kaichi; Yanan Wang; Komei Sugiura
>
> **备注:** Accepted to IEEE RA-L
>
> **摘要:** We address language-conditioned robotic manipulation using flow-based trajectory generation, which enables training on human and web videos of object manipulation and requires only minimal embodiment-specific data. This task is challenging, as object trajectory generation from pre-manipulation images and natural language instructions requires appropriate instruction-flow alignment. To tackle this challenge, we propose the flow-based Language Instruction-guided open-Loop ACtion generator (LILAC). This flow-based Vision-Language-Action model (VLA) generates object-centric 2D optical flow from an RGB image and a natural language instruction, and converts the flow into a 6-DoF manipulator trajectory. LILAC incorporates two key components: Semantic Alignment Loss, which strengthens language conditioning to generate instruction-aligned optical flow, and Prompt-Conditioned Cross-Modal Adapter, which aligns learned visual prompts with image and text features to provide rich cues for flow generation. Experimentally, our method outperformed existing approaches in generated flow quality across multiple benchmarks. Furthermore, in physical object manipulation experiments using free-form instructions, LILAC demonstrated a superior task success rate compared to existing methods. The project page is available at this https URL.
>
---
#### [new 034] Integrating Deep RL and Bayesian Inference for ObjectNav in Mobile Robotics
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于移动机器人目标导航任务，旨在解决部分可观测环境下的物体搜索问题。融合贝叶斯推理与深度强化学习，提升搜索效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.25366](https://arxiv.org/pdf/2603.25366)**

> **作者:** João Castelo-Branco; José Santos-Victor; Alexandre Bernardino
>
> **备注:** Accepted and to be published in the ICARSC 2026 26th IEEE International Conference on Autonomous Robot Systems and Competitions
>
> **摘要:** Autonomous object search is challenging for mobile robots operating in indoor environments due to partial observability, perceptual uncertainty, and the need to trade off exploration and navigation efficiency. Classical probabilistic approaches explicitly represent uncertainty but typically rely on handcrafted action-selection heuristics, while deep reinforcement learning enables adaptive policies but often suffers from slow convergence and limited interpretability. This paper proposes a hybrid object-search framework that integrates Bayesian inference with deep reinforcement learning. The method maintains a spatial belief map over target locations, updated online through Bayesian inference from calibrated object detections, and trains a reinforcement learning policy to select navigation actions directly from this probabilistic representation. The approach is evaluated in realistic indoor simulation using Habitat 3.0 and compared against developed baseline strategies. Across two indoor environments, the proposed method improves success rate while reducing search effort. Overall, the results support the value of combining Bayesian belief estimation with learned action selection to achieve more efficient and reliable objectsearch behavior under partial observability.
>
---
#### [new 035] Towards Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale
- **分类: cs.RO**

- **简介: 该论文属于具身AI任务，旨在解决肌肉驱动模型的运动控制问题。提出MuscleMimic框架，实现大规模、生理真实的全身运动模仿学习。**

- **链接: [https://arxiv.org/pdf/2603.25544](https://arxiv.org/pdf/2603.25544)**

> **作者:** Chengkun Li; Cheryl Wang; Bianca Ziliotto; Merkourios Simos; Jozsef Kovecses; Guillaume Durandau; Alexander Mathis
>
> **摘要:** Learning motor control for muscle-driven musculoskeletal models is hindered by the computational cost of biomechanically accurate simulation and the scarcity of validated, open full-body models. Here we present MuscleMimic, an open-source framework for scalable motion imitation learning with physiologically realistic, muscle-actuated humanoids. MuscleMimic provides two validated musculoskeletal embodiments - a fixed-root upper-body model (126 muscles) for bimanual manipulation and a full-body model (416 muscles) for locomotion - together with a retargeting pipeline that maps SMPL-format motion capture data onto musculoskeletal structures while preserving kinematic and dynamic consistency. Leveraging massively parallel GPU simulation, the framework achieves order-of-magnitude training speedups over prior CPU-based approaches while maintaining comprehensive collision handling, enabling a single generalist policy to be trained on hundreds of diverse motions within days. The resulting policy faithfully reproduces a broad repertoire of human movements under full muscular control and can be fine-tuned to novel motions within hours. Biomechanical validation against experimental walking and running data demonstrates strong agreement in joint kinematics (mean correlation r = 0.90), while muscle activation analysis reveals both the promise and fundamental challenges of achieving physiological fidelity through kinematic imitation alone. By lowering the computational and data barriers to musculoskeletal simulation, MuscleMimic enables systematic model validation across diverse dynamic movements and broader participation in neuromuscular control research. Code, models, checkpoints, and retargeted datasets are available at: this https URL
>
---
#### [new 036] Accurate Surface and Reflectance Modelling from 3D Radar Data with Neural Radiance Fields
- **分类: cs.RO**

- **简介: 该论文属于3D重建任务，解决雷达数据稀疏噪声下的表面建模问题。提出神经隐式方法，联合建模几何与反射特性，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2603.25623](https://arxiv.org/pdf/2603.25623)**

> **作者:** Judith Treffler; Vladimír Kubelka; Henrik Andreasson; Martin Magnusson
>
> **摘要:** Robust scene representation is essential for autonomous systems to safely operate in challenging low-visibility environments. Radar has a clear advantage over cameras and lidars in these conditions due to its resilience to environmental factors such as fog, smoke, or dust. However, radar data is inherently sparse and noisy, making reliable 3D surface reconstruction challenging. To address these challenges, we propose a neural implicit approach for 3D mapping from radar point clouds, which jointly models scene geometry and view-dependent radar intensities. Our method leverages a memory-efficient hybrid feature encoding to learn a continuous Signed Distance Field (SDF) for surface reconstruction, while also capturing radar-specific reflective properties. We show that our approach produces smoother, more accurate 3D surface reconstructions compared to existing lidar-based reconstruction methods applied to radar data, and can reconstruct view-dependent radar intensities. We also show that in general, as input point clouds get sparser, neural implicit representations render more faithful surfaces, compared to traditional explicit SDFs and meshing techniques.
>
---
#### [new 037] IndustriConnect: MCP Adapters and Mock-First Evaluation for AI-Assisted Industrial Operations
- **分类: cs.SE; cs.RO; eess.SY**

- **简介: 该论文属于工业AI系统任务，解决AI与工业协议不兼容问题，通过MCP适配器实现协议转换，并进行模拟测试验证其可靠性与错误处理能力。**

- **链接: [https://arxiv.org/pdf/2603.24703](https://arxiv.org/pdf/2603.24703)**

> **作者:** Melwin Xavier; Melveena Jolly; Vaisakh M A; Midhun Xavier
>
> **摘要:** AI assistants can decompose multi-step workflows, but they do not natively speak industrial protocols such as Modbus, MQTT/Sparkplug B, or OPC UA, so this paper presents INDUSTRICONNECT, a prototype suite of Model Context Protocol (MCP) adapters that expose industrial operations as schema-discoverable AI tools while preserving protocol-specific connectivity and safety controls; the system uses a common response envelope and a mock-first workflow so adapter behavior can be exercised locally before connecting to plant equipment, and a deterministic benchmark covering normal, fault-injected, stress, and recovery scenarios evaluates the flagship adapters, comprising 870 runs (480 normal, 210 fault-injected, 120 stress, 60 recovery trials) and 2820 tool calls across 7 fault scenarios and 12 stress scenarios, where the normal suite achieved full success, the fault suite confirmed structured error handling with adapter-level uint16 range validation, the stress suite identified concurrency boundaries, and same-session recovery after endpoint restart is demonstrated for all three protocols, with results providing evidence spanning adapter correctness, concurrency behavior, and structured error handling for AI-assisted industrial operations.
>
---
#### [new 038] Towards automatic smoke detector inspection: Recognition of the smoke detectors in industrial facilities and preparation for future drone integration
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于目标检测任务，旨在自动识别工业设施中的烟雾探测器，以支持无人机巡检。研究比较了多种检测模型，并探索了数据增强策略，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2603.24850](https://arxiv.org/pdf/2603.24850)**

> **作者:** Lukas Kratochvila; Jakub Stefansky; Simon Bilik; Robert Rous; Tomas Zemcik; Michal Wolny; Frantisek Rusnak; Ondrej Cech; Karel Horak
>
> **摘要:** Fire safety consists of a complex pipeline, and it is a very important topic of concern. One of its frontal parts are the smoke detectors, which are supposed to provide an alarm prior to a massive fire appears. As they are often difficult to reach due to high ceilings or problematic locations, an automatic inspection system would be very beneficial as it could allow faster revisions, prevent workers from dangerous work in heights, and make the whole process cheaper. In this study, we present the smoke detector recognition part of the automatic inspection system, which could easily be integrated to the drone system. As part of our research, we compare two popular convolutional-based object detectors YOLOv11 and SSD widely used on embedded devices together with the state-of-the-art transformer-based RT-DETRv2 with the backbones of different sizes. Due to a complicated way of collecting a sufficient amount of data for training in the real-world environment, we also compare several training strategies using the real and semi-synthetic data together with various augmentation methods. To achieve a robust testing, all models were evaluated on two test datasets with an expected and difficult appearance of the smoke detectors including motion blur, small resolution, or not complete objects. The best performing detector is the YOLOv11n, which reaches the average mAP@0.5 score of 0.884. Our code, pretrained models and dataset are publicly available.
>
---
#### [new 039] LaMP: Learning Vision-Language-Action Policies with 3D Scene Flow as Latent Motion Prior
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LaMP框架，解决机器人操作中复杂3D交互的感知与控制问题。通过结合视觉-语言-动作和3D场景流，提升任务执行的鲁棒性与成功率。**

- **链接: [https://arxiv.org/pdf/2603.25399](https://arxiv.org/pdf/2603.25399)**

> **作者:** Xinkai Wang; Chenyi Wang; Yifu Xu; Mingzhe Ye; Fu-Cheng Zhang; Jialin Tian; Xinyu Zhan; Lifeng Zhu; Cewu Lu; Lixin Yang
>
> **摘要:** We introduce \textbf{LaMP}, a dual-expert Vision-Language-Action framework that embeds dense 3D scene flow as a latent motion prior for robotic manipulation. Existing VLA models regress actions directly from 2D semantic visual features, forcing them to learn complex 3D physical interactions implicitly. This implicit learning strategy degrades under unfamiliar spatial dynamics. LaMP addresses this limitation by aligning a flow-matching \emph{Motion Expert} with a policy-predicting \emph{Action Expert} through gated cross-attention. Specifically, the Motion Expert generates a one-step partially denoised 3D scene flow, and its hidden states condition the Action Expert without full multi-step reconstruction. We evaluate LaMP on the LIBERO, LIBERO-Plus, and SimplerEnv-WidowX simulation benchmarks as well as real-world experiments. LaMP consistently outperforms evaluated VLA baselines across LIBERO, LIBERO-Plus, and SimplerEnv-WidowX benchmarks, achieving the highest reported average success rates under the same training budgets. On LIBERO-Plus OOD perturbations, LaMP shows improved robustness with an average 9.7% gain over the strongest prior baseline. Our project page is available at this https URL.
>
---
#### [new 040] Modernising Reinforcement Learning-Based Navigation for Embodied Semantic Scene Graph Generation
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于具身语义场景图生成任务，解决有限动作预算下如何提升场景图质量与导航效率的问题。通过改进策略优化方法和动作表示，提升场景图完整性与安全性。**

- **链接: [https://arxiv.org/pdf/2603.25415](https://arxiv.org/pdf/2603.25415)**

> **作者:** Roman Kueble; Marco Hueller; Mrunmai Phatak; Rainer Lienhart; Joerg Haehner
>
> **摘要:** Semantic world models enable embodied agents to reason about objects, relations, and spatial context beyond purely geometric representations. In Organic Computing, such models are a key enabler for objective-driven self-adaptation under uncertainty and resource constraints. The core challenge is to acquire observations maximising model quality and downstream usefulness within a limited action budget. Semantic scene graphs (SSGs) provide a structured and compact representation for this purpose. However, constructing them within a finite action horizon requires exploration strategies that trade off information gain against navigation cost and decide when additional actions yield diminishing returns. This work presents a modular navigation component for Embodied Semantic Scene Graph Generation and modernises its decision-making by replacing the policy-optimisation method and revisiting the discrete action formulation. We study compact and finer-grained, larger discrete motion sets and compare a single-head policy over atomic actions with a factorised multi-head policy over action components. We evaluate curriculum learning and optional depth-based collision supervision, and assess SSG completeness, execution safety, and navigation behaviour. Results show that replacing the optimisation algorithm alone improves SSG completeness by 21\% relative to the baseline under identical reward shaping. Depth mainly affects execution safety (collision-free motion), while completeness remains largely unchanged. Combining modern optimisation with a finer-grained, factorised action representation yields the strongest overall completeness--efficiency trade-off.
>
---
#### [new 041] Vega: Learning to Drive with Natural Language Instructions
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出Vega模型，解决自动驾驶中根据自然语言指令进行决策的问题。构建了包含10万场景的InstructScene数据集，融合视觉、语言和动作模态，提升个性化驾驶能力。**

- **链接: [https://arxiv.org/pdf/2603.25741](https://arxiv.org/pdf/2603.25741)**

> **作者:** Sicheng Zuo; Yuxuan Li; Wenzhao Zheng; Zheng Zhu; Jie Zhou; Jiwen Lu
>
> **备注:** Code is available at this https URL
>
> **摘要:** Vision-language-action models have reshaped autonomous driving to incorporate languages into the decision-making process. However, most existing pipelines only utilize the language modality for scene descriptions or reasoning and lack the flexibility to follow diverse user instructions for personalized driving. To address this, we first construct a large-scale driving dataset (InstructScene) containing around 100,000 scenes annotated with diverse driving instructions with the corresponding trajectories. We then propose a unified Vision-Language-World-Action model, Vega, for instruction-based generation and planning. We employ the autoregressive paradigm to process visual inputs (vision) and language instructions (language) and the diffusion paradigm to generate future predictions (world modeling) and trajectories (action). We perform joint attention to enable interactions between the modalities and use individual projection layers for different modalities for more capabilities. Extensive experiments demonstrate that our method not only achieves superior planning performance but also exhibits strong instruction-following abilities, paving the way for more intelligent and personalized driving systems.
>
---
#### [new 042] The Competence Shadow: Theory and Bounds of AI Assistance in Safety Engineering
- **分类: cs.AI; cs.ET; cs.HC; cs.RO; cs.SE**

- **简介: 该论文属于安全工程与AI协作任务，解决AI辅助可能引入系统性盲点的问题。提出“能力阴影”概念，分析协作结构对安全分析的影响，强调协作设计的重要性。**

- **链接: [https://arxiv.org/pdf/2603.25197](https://arxiv.org/pdf/2603.25197)**

> **作者:** Umair Siddique
>
> **备注:** 8 Pages, 3 Figures, 2 table
>
> **摘要:** As AI assistants become integrated into safety engineering workflows for Physical AI systems, a critical question emerges: does AI assistance improve safety analysis quality, or introduce systematic blind spots that surface only through post-deployment incidents? This paper develops a formal framework for AI assistance in safety analysis. We first establish why safety engineering resists benchmark-driven evaluation: safety competence is irreducibly multidimensional, constrained by context-dependent correctness, inherent incompleteness, and legitimate expert disagreement. We formalize this through a five-dimensional competence framework capturing domain knowledge, standards expertise, operational experience, contextual understanding, and judgment. We introduce the competence shadow: the systematic narrowing of human reasoning induced by AI-generated safety analysis. The shadow is not what the AI presents, but what it prevents from being considered. We formalize four canonical human-AI collaboration structures and derive closed-form performance bounds, demonstrating that the competence shadow compounds multiplicatively to produce degradation far exceeding naive additive estimates. The central finding is that AI assistance in safety engineering is a collaboration design problem, not a software procurement decision. The same tool degrades or improves analysis quality depending entirely on how it is used. We derive non-degradation conditions for shadow-resistant workflows and call for a shift from tool qualification toward workflow qualification for trustworthy Physical AI.
>
---
## 更新

#### [replaced 001] T-araVLN: Translator for Agricultural Robotic Agents on Vision-and-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于农业视觉语言导航任务，旨在解决农业机器人依赖人工或固定轨道的问题。提出T-araVLN方法，通过指令翻译模块提升导航准确性。**

- **链接: [https://arxiv.org/pdf/2509.06644](https://arxiv.org/pdf/2509.06644)**

> **作者:** Xiaobei Zhao; Xingqi Lyu; Xin Chen; Xiang Li
>
> **摘要:** Agricultural robotic agents have been becoming useful helpers in a wide range of agricultural tasks. However, they still heavily rely on manual operations or fixed railways for movement. To address this limitation, the AgriVLN method and the A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the agricultural domain, enabling agents to navigate to the target positions following the natural language instructions. We observe that AgriVLN can effectively understands the simple instructions, but often misunderstands the complex ones. To bridge this gap, we propose the T-araVLN method, in which we build the instruction translator module to translate noisy and mistaken instructions into refined and precise representations. When evaluated on A2A, our T-araVLN successfully improves Success Rate (SR) from 0.47 to 0.63 and reduces Navigation Error (NE) from 2.91m to 2.28m, demonstrating the state-of-the-art performance in the agricultural VLN domain. Code: this https URL.
>
---
#### [replaced 002] Seeking Physics in Diffusion Noise
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文研究视频扩散模型是否包含物理合理性信号，提出方法在推理时通过物理验证器优化轨迹选择，提升物理一致性并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2603.14294](https://arxiv.org/pdf/2603.14294)**

> **作者:** Chujun Tang; Lei Zhong; Fangqiang Ding
>
> **备注:** 32 pages, 8 figures, 10 tables
>
> **摘要:** Do video diffusion models encode signals predictive of physical plausibility? We probe intermediate denoising representations of a pretrained Diffusion Transformer (DiT) and find that physically plausible and implausible videos are partially separable in mid-layer feature space across noise levels. This separability cannot be fully attributed to visual quality or generator identity, suggesting recoverable physics-related cues in frozen DiT features. Leveraging this observation, we introduce progressive trajectory selection, an inference-time strategy that scores parallel denoising trajectories at a few intermediate checkpoints using a lightweight physics verifier trained on frozen features, and prunes low-scoring candidates early. Extensive experiments on PhyGenBench demonstrate that our method improves physical consistency while reducing inference cost, achieving comparable results to Best-of-K sampling with substantially fewer denoising steps.
>
---
#### [replaced 003] Self-Supervised Multisensory Pretraining for Contact-Rich Robot Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人强化学习任务，旨在解决多感官环境下学习效率低和鲁棒性差的问题。提出MSDP框架，通过自监督预训练提升多感官表征，增强策略学习效果。**

- **链接: [https://arxiv.org/pdf/2511.14427](https://arxiv.org/pdf/2511.14427)**

> **作者:** Rickmer Krohn; Vignesh Prasad; Gabriele Tiboni; Georgia Chalvatzaki
>
> **备注:** 8 pages, 11 figures, Accepted at RA-L
>
> **摘要:** Effective contact-rich manipulation requires robots to synergistically leverage vision, force, and proprioception. However, Reinforcement Learning agents struggle to learn in such multisensory settings, especially amidst sensory noise and dynamic changes. We propose MultiSensory Dynamic Pretraining (MSDP), a novel framework for learning expressive multisensory representations tailored for task-oriented policy learning. MSDP is based on masked autoencoding and trains a transformer-based encoder by reconstructing multisensory observations from only a subset of sensor embeddings, leading to cross-modal prediction and sensor fusion. For downstream policy learning, we introduce a novel asymmetric architecture, where a cross-attention mechanism allows the critic to extract dynamic, task-specific features from the frozen embeddings, while the actor receives a stable pooled representation to guide its actions. Our method demonstrates accelerated learning and robust performance under diverse perturbations, including sensor noise, and changes in object dynamics. Evaluations in multiple challenging, contact-rich robot manipulation tasks in simulation and the real world showcase the effectiveness of MSDP. Our approach exhibits strong robustness to perturbations and achieves high success rates on the real robot with as few as 6,000 online interactions, offering a simple yet powerful solution for complex multisensory robotic control. Website: this https URL
>
---
#### [replaced 004] MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决模拟到现实的零样本迁移问题。通过大规模仿真数据训练，实现无需真实数据微调的高效操作。**

- **链接: [https://arxiv.org/pdf/2603.16861](https://arxiv.org/pdf/2603.16861)**

> **作者:** Abhay Deshpande; Maya Guru; Rose Hendrix; Snehal Jauhri; Ainaz Eftekhar; Rohun Tripathi; Max Argus; Jordi Salvador; Haoquan Fang; Matthew Wallingford; Wilbert Pumacay; Yejin Kim; Quinn Pfeifer; Ying-Chun Lee; Piper Wolters; Omar Rayyan; Mingtong Zhang; Jiafei Duan; Karen Farley; Winson Han; Eli Vanderbilt; Dieter Fox; Ali Farhadi; Georgia Chalvatzaki; Dhruv Shah; Ranjay Krishna
>
> **摘要:** A prevailing view in robot learning is that simulation alone is not enough; effective sim-to-real transfer is widely believed to require at least some real-world data collection or task-specific fine-tuning to bridge the gap between simulated and physical environments. We challenge that assumption. With sufficiently large-scale and diverse simulated synthetic training data, we show that zero-shot transfer to the real world is not only possible, but effective for both static and mobile manipulation. We introduce MolmoBot-Engine, a fully open-source pipeline for procedural data generation across robots, tasks, and diverse simulated environments in MolmoSpaces. With it, we release MolmoBot-Data, a dataset of 1.8 million expert trajectories for articulated object manipulation and pick-and-place tasks. We train three policy classes: MolmoBot, a Molmo2-based multi-frame vision-language model with a flow-matching action head; MolmoBot-Pi0, which replicates the $\pi_0$ architecture to enable direct comparison; and MolmoBot-SPOC, a lightweight policy suitable for edge deployment and amenable to RL fine-tuning. We evaluate on two robotic platforms: the Franka FR3 for tabletop manipulation tasks and the Rainbow Robotics RB-Y1 mobile manipulator for door opening, drawer manipulation, cabinet interaction, and mobile pick-and-place. Without any real-world fine-tuning, our policies achieve zero-shot transfer to unseen objects and environments. On tabletop pick-and-place, MolmoBot achieves a success rate of 79.2% in real world evaluations across 4 settings, outperforming $\pi_{0.5}$ at 39.2%. Our results demonstrate that procedural environment generation combined with diverse articulated assets can produce robust manipulation policies that generalize broadly to the real world. Technical website: this https URL
>
---
#### [replaced 005] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决长期记忆构建和误差累积问题。提出DecoVLN框架，通过优化记忆选择和修正策略提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.13133](https://arxiv.org/pdf/2603.13133)**

> **作者:** Zihao Xin; Wentong Li; Yixuan Jiang; Bin Wang; Runmin Cong; Jie Qin; Shengjun Huang
>
> **备注:** 16 pages, 8 figures, CVPR2026
>
> **摘要:** Vision-and-Language Navigation (VLN) requires agents to follow long-horizon instructions and navigate complex 3D environments. However, existing approaches face two major challenges: constructing an effective long-term memory bank and overcoming the compounding errors problem. To address these issues, we propose DecoVLN, an effective framework designed for robust streaming perception and closed-loop control in long-horizon navigation. First, we formulate long-term memory construction as an optimization problem and introduce adaptive refinement mechanism that selects frames from a historical candidate pool by iteratively optimizing a unified scoring function. This function jointly balances three key criteria: semantic relevance to the instruction, visual diversity from the selected memory, and temporal coverage of the historical trajectory. Second, to alleviate compounding errors, we introduce a state-action pair-level corrective finetuning strategy. By leveraging geodesic distance between states to precisely quantify deviation from the expert trajectory, the agent collects high-quality state-action pairs in the trusted region while filtering out the polluted data with low relevance. This improves both the efficiency and stability of error correction. Extensive experiments demonstrate the effectiveness of DecoVLN, and we have deployed it in real-world environments.
>
---
#### [replaced 006] 3D Dynamics-Aware Manipulation: Endowing Manipulation Policies with 3D Foresight
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，解决深度运动下操纵性能不足的问题。通过引入3D动态建模与策略学习，提升操纵政策的3D预见能力。**

- **链接: [https://arxiv.org/pdf/2502.10028](https://arxiv.org/pdf/2502.10028)**

> **作者:** Yuxin He; Ruihao Zhang; Xianzu Wu; Zhiyuan Zhang; Cheng Ding; Qiang Nie
>
> **备注:** ICRA 2026
>
> **摘要:** The incorporation of world modeling into manipulation policy learning has pushed the boundary of manipulation performance. However, existing efforts simply model the 2D visual dynamics, which is insufficient for robust manipulation when target tasks involve prominent depth-wise movement. To address this, we present a 3D dynamics-aware manipulation framework that seamlessly integrates 3D world modeling and policy learning. Three self-supervised learning tasks (current depth estimation, future RGB-D prediction, 3D flow prediction) are introduced within our framework, which complement each other and endow the policy model with 3D foresight. Extensive experiments on simulation and the real world show that 3D foresight can greatly boost the performance of manipulation policies without sacrificing inference speed. Code is available at this https URL.
>
---
#### [replaced 007] End-to-End Low-Level Neural Control of an Industrial-Grade 6D Magnetic Levitation System
- **分类: eess.SY; cs.AI; cs.RO**

- **简介: 该论文属于6D磁悬浮系统控制任务，旨在解决复杂动态下的稳定控制问题。通过端到端神经控制方法，直接从传感器数据生成电流指令，实现高效、鲁棒的控制。**

- **链接: [https://arxiv.org/pdf/2509.01388](https://arxiv.org/pdf/2509.01388)**

> **作者:** Philipp Hartmann; Jannick Stranghöner; Klaus Neumann
>
> **备注:** 8 pages, 7 figures, 2 tables
>
> **摘要:** Magnetic levitation is poised to revolutionize industrial automation by integrating flexible in-machine product transport and seamless manipulation. It is expected to become the standard drive technology for automated manufacturing. However, controlling such systems is inherently challenging due to their complex, unstable dynamics. Traditional control approaches, which rely on hand-crafted control engineering, typically yield robust but conservative solutions, with their performance closely tied to the expertise of the engineering team. In contrast, learning-based neural control presents a promising alternative. This paper presents the first neural controller for 6D magnetic levitation. Trained end-to-end on interaction data from a proprietary controller, it directly maps raw sensor data and 6D reference poses to coil current commands. The neural controller can effectively generalize to previously unseen situations while maintaining accurate and robust control. These results underscore the practical feasibility of learning-based neural control in complex physical systems and suggest a future where such a paradigm could enhance or even substitute traditional engineering approaches in demanding real-world applications. The trained neural controller, source code, and demonstration videos are publicly available at this https URL.
>
---
#### [replaced 008] RoboMatch: A Unified Mobile-Manipulation Teleoperation Platform with Auto-Matching Network Architecture for Long-Horizon Tasks
- **分类: cs.RO**

- **简介: 论文提出RoboMatch，解决动态环境中长周期移动操作任务的问题。设计统一控制界面与自动匹配网络，提升操作精度与效率，增强复杂任务的执行能力。**

- **链接: [https://arxiv.org/pdf/2509.08522](https://arxiv.org/pdf/2509.08522)**

> **作者:** Hanyu Liu; Yunsheng Ma; Jiaxin Huang; Keqiang Ren; Jiayi Wen; Yilin Zheng; Haoru Luan; Baishu Wan; Pan Li; Jiejun Hou; Zhihua Wang; Zhigong Song
>
> **备注:** Accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** This paper presents RoboMatch, a novel unified teleoperation platform for mobile manipulation with an auto-matching network architecture, designed to tackle long-horizon tasks in dynamic environments. Our system enhances teleoperation performance, data collection efficiency, task accuracy, and operational stability. The core of RoboMatch is a cockpit-style control interface that enables synchronous operation of the mobile base and dual arms, significantly improving control precision and data collection. Moreover, we introduce the Proprioceptive-Visual Enhanced Diffusion Policy (PVE-DP), which leverages Discrete Wavelet Transform (DWT) for multi-scale visual feature extraction and integrates high-precision IMUs at the end-effector to enrich proprioceptive feedback, substantially boosting fine manipulation performance. Furthermore, we propose an Auto-Matching Network (AMN) architecture that decomposes long-horizon tasks into logical sequences and dynamically assigns lightweight pre-trained models for distributed inference. Experimental results demonstrate that our approach improves data collection efficiency by over 20%, increases task success rates by 20-30% with PVE-DP, and enhances long-horizon inference performance by approximately 40% with AMN, offering a robust solution for complex manipulation tasks. Project website: this https URL
>
---
#### [replaced 009] Constant-Time Motion Planning with Manipulation Behaviors
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出B-CTMP算法，解决机器人在半结构化环境中高效安全地完成抓取和插入等操作的运动规划问题，实现快速、可靠的任务执行。**

- **链接: [https://arxiv.org/pdf/2512.00939](https://arxiv.org/pdf/2512.00939)**

> **作者:** Nayesha Gandotra; Itamar Mishani; Maxim Likhachev
>
> **备注:** In submission
>
> **摘要:** Recent progress in contact-rich robotic manipulation has been striking, yet most deployed systems remain confined to simple, scripted routines. One of the key barriers is the lack of motion planning algorithms that can provide verifiable guarantees for safety, efficiency and reliability. To address this, a family of algorithms called Constant-Time Motion Planning (CTMP) was introduced, which leverages a preprocessing phase to enable collision-free motion queries in a fixed, user-specified time budget (e.g., 10 milliseconds). However, existing CTMP methods do not explicitly incorporate the manipulation behaviors essential for object handling. To bridge this gap, we introduce the \textit{Behavioral Constant-Time Motion Planner} (B-CTMP), an algorithm that extends CTMP to solve a broad class of two-step manipulation tasks: (1) a collision-free motion to a behavior initiation state, followed by (2) execution of a manipulation behavior (such as grasping or insertion) to reach the goal. By precomputing compact data structures, B-CTMP guarantees constant-time query in mere milliseconds while ensuring completeness and successful task execution over a specified set of states. We evaluate B-CTMP on two canonical manipulation tasks, shelf picking and plug insertion, in simulation and on a real robot. Our results show that B-CTMP unifies collision-free planning and object manipulation within a single constant-time framework, providing provable guarantees of speed and success for manipulation in semi-structured environments.
>
---
#### [replaced 010] CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多摄像头3D目标检测任务，旨在解决模型在不同摄像头配置间泛化能力差的问题。提出CoIn3D框架，通过空间感知特征调制和相机感知数据增强提升跨配置性能。**

- **链接: [https://arxiv.org/pdf/2603.05042](https://arxiv.org/pdf/2603.05042)**

> **作者:** Zhaonian Kuang; Rui Ding; Haotian Wang; Xinhu Zheng; Meng Yang; Gang Hua
>
> **备注:** Accepted to CVPR 2026 main track
>
> **摘要:** Multi-camera 3D object detection (MC3D) has attracted increasing attention with the growing deployment of multi-sensor physical agents, such as robots and autonomous vehicles. However, MC3D models still struggle to generalize to unseen platforms with new multi-camera configurations. Current solutions simply employ a meta-camera for unified representation but lack comprehensive consideration. In this paper, we revisit this issue and identify that the devil lies in spatial prior discrepancies across source and target configurations, including different intrinsics, extrinsics, and array layouts. To address this, we propose CoIn3D, a generalizable MC3D framework that enables strong transferability from source configurations to unseen target ones. CoIn3D explicitly incorporates all identified spatial priors into both feature embedding and image observation through spatial-aware feature modulation (SFM) and camera-aware data augmentation (CDA), respectively. SFM enriches feature space by integrating four spatial representations, such as focal length, ground depth, ground gradient, and Plücker coordinate. CDA improves observation diversity under various configurations via a training-free dynamic novel-view image synthesis scheme. Extensive experiments demonstrate that CoIn3D achieves strong cross-configuration performance on landmark datasets such as NuScenes, Waymo, and Lyft, under three dominant MC3D paradigms represented by BEVDepth, BEVFormer, and PETR.
>
---
#### [replaced 011] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决失败诊断与学习问题。提出ViFailback框架，结合视觉符号进行故障分析与纠正，并发布相关数据集与基准测试。**

- **链接: [https://arxiv.org/pdf/2512.02787](https://arxiv.org/pdf/2512.02787)**

> **作者:** Xianchao Zeng; Xinyu Zhou; Youcheng Li; Jiayou Shi; Tianle Li; Liangming Chen; Lei Ren; Yong-Lu Li
>
> **备注:** Accepted by CVPR 2026. Project Website: this https URL
>
> **摘要:** Vision-Language-Action (VLA) models have recently achieved remarkable progress in robotic manipulation, yet they remain limited in failure diagnosis and learning from failures. Additionally, existing failure datasets are mostly generated programmatically in simulation, which limits their generalization to the real world. In light of these, we introduce ViFailback, a framework designed to diagnose robotic manipulation failures and provide both textual and visual correction guidance. Our framework utilizes explicit visual symbols to enhance annotation efficiency. We further release the ViFailback dataset, a large-scale collection of 58,126 Visual Question Answering (VQA) pairs along with their corresponding 5,202 real-world manipulation trajectories. Based on the dataset, we establish ViFailback-Bench, a benchmark of 11 fine-grained VQA tasks designed to assess the failure diagnosis and correction abilities of Vision-Language Models (VLMs), featuring ViFailback-Bench Lite for closed-ended and ViFailback-Bench Hard for open-ended evaluation. To demonstrate the effectiveness of our framework, we built the ViFailback-8B VLM, which not only achieves significant overall performance improvement on ViFailback-Bench but also generates visual symbols for corrective action guidance. Finally, by integrating ViFailback-8B with a VLA model, we conduct real-world robotic experiments demonstrating its ability to assist the VLA model in recovering from failures. Project Website: this https URL
>
---
#### [replaced 012] Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion
- **分类: cs.RO**

- **简介: 该论文属于四足机器人运动控制任务，解决sim-to-real迁移性和奖励过拟合问题。提出MoE策略与RoboGauge评估框架，提升复杂地形下的运动可靠性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.00678](https://arxiv.org/pdf/2602.00678)**

> **作者:** Tianyang Wu; Hanwei Guo; Yuhang Wang; Junshu Yang; Xinyang Sui; Jiayi Xie; Xingyu Chen; Zeyang Liu; Xuguang Lan
>
> **备注:** Project Page: this https URL
>
> **摘要:** Reinforcement learning has shown strong promise for quadrupedal agile locomotion, even with proprioception-only sensing. In practice, however, sim-to-real gap and reward overfitting in complex terrains can produce policies that fail to transfer, while physical validation remains risky and inefficient. To address these challenges, we introduce a unified framework encompassing a Mixture-of-Experts (MoE) locomotion policy for robust multi-terrain representation with RoboGauge, a predictive assessment suite that quantifies sim-to-real transferability. The MoE policy employs a gated set of specialist experts to decompose latent terrain and command modeling, achieving superior deployment robustness and generalization via proprioception alone. RoboGauge further provides multi-dimensional proprioception-based metrics via sim-to-sim tests over terrains, difficulty levels, and domain randomizations, enabling reliable MoE policy selection without extensive physical trials. Experiments on a Unitree Go2 demonstrate robust locomotion on unseen challenging terrains, including snow, sand, stairs, slopes, and 30 cm obstacles. In dedicated high-speed tests, the robot reaches 4 m/s and exhibits an emergent narrow-width gait associated with improved stability at high velocity.
>
---
#### [replaced 013] MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶轨迹生成任务，解决传统方法依赖离散锚点导致的效率与鲁棒性矛盾。提出MeanFuser，通过连续表示、均值流和自适应重建提升性能与速度。**

- **链接: [https://arxiv.org/pdf/2602.20060](https://arxiv.org/pdf/2602.20060)**

> **作者:** Junli Wang; Yinan Zheng; Xueyi Liu; Zebin Xing; Pengfei Li; Guang Li; Kun Ma; Guang Chen; Hangjun Ye; Zhongpu Xia; Long Chen; Qichao Zhang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Generative models have shown great potential in trajectory planning. Recent studies demonstrate that anchor-guided generative models are effective in modeling the uncertainty of driving behaviors and improving overall performance. However, these methods rely on discrete anchor vocabularies that must sufficiently cover the trajectory distribution during testing to ensure robustness, inducing an inherent trade-off between vocabulary size and model performance. To overcome this limitation, we propose MeanFuser, an end-to-end autonomous driving method that enhances both efficiency and robustness through three key designs. (1) We introduce Gaussian Mixture Noise (GMN) to guide generative sampling, enabling a continuous representation of the trajectory space and eliminating the dependency on discrete anchor vocabularies. (2) We adapt ``MeanFlow Identity" to end-to-end planning, which models the mean velocity field between GMN and trajectory distribution instead of the instantaneous velocity field used in vanilla flow matching methods, effectively eliminating numerical errors from ODE solvers and significantly accelerating inference. (3) We design a lightweight Adaptive Reconstruction Module (ARM) that enables the model to implicitly select from all sampled proposals or reconstruct a new trajectory when none is satisfactory via attention this http URL on the NAVSIM closed-loop benchmark demonstrate that MeanFuser achieves outstanding performance without the supervision of the PDM Score and exceptional inference efficiency, offering a robust and efficient solution for end-to-end autonomous driving. Our code and model are available at this https URL.
>
---
#### [replaced 014] Chance-Constrained Iterative Linear-Quadratic Stochastic Games
- **分类: cs.RO**

- **简介: 该论文属于多智能体决策任务，旨在解决不确定环境下安全约束问题。提出CCILQGames算法，利用对偶方法处理概率约束，提升自主驾驶场景下的策略安全性与交互性。**

- **链接: [https://arxiv.org/pdf/2203.01222](https://arxiv.org/pdf/2203.01222)**

> **作者:** Hai Zhong; Yutaka Shimizu; Jianyu Chen
>
> **备注:** Updated version of the published IEEE RA-L paper. Assumption 1 and strategy space definition revised to make the information structure explicit. Theorem 1 assumptions are more explict. No changes to algorithm or experimental results
>
> **摘要:** Dynamic game arises as a powerful paradigm for multi-robot planning, for which safety constraint satisfaction is crucial. Constrained stochastic games are of particular interest, as real-world robots need to operate and satisfy constraints under uncertainty. Existing methods for solving stochastic games handle chance constraints using exponential penalties with hand-tuned weights. However, finding a suitable penalty weight is nontrivial and requires trial and error. In this paper, we propose the chance-constrained iterative linear-quadratic stochastic games (CCILQGames) algorithm. CCILQGames solves chance-constrained stochastic games using the augmented Lagrangian method. We evaluate our algorithm in three autonomous driving scenarios, including merge, intersection, and roundabout. Experimental results and Monte Carlo tests show that CCILQGames can generate safe and interactive strategies in stochastic environments.
>
---
#### [replaced 015] LLM4AD: Large Language Models for Autonomous Driving -- Concept, Review, Benchmark, Experiments, and Future Trends
- **分类: cs.RO; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于自动驾驶任务，探讨如何将大语言模型应用于自动驾驶系统，解决感知、决策等关键问题。工作包括概念设计、综述、基准测试及实验验证。**

- **链接: [https://arxiv.org/pdf/2410.15281](https://arxiv.org/pdf/2410.15281)**

> **作者:** Can Cui; Yunsheng Ma; Sung-Yeon Park; Zichong Yang; Yupeng Zhou; Peiran Liu; Juanwu Lu; Juntong Peng; Jiaru Zhang; Ruqi Zhang; Lingxi Li; Yaobin Chen; Jitesh H. Panchal; Amr Abdelraouf; Rohit Gupta; Kyungtae Han; Ziran Wang
>
> **备注:** The paper was accepted by the Proceedings of the IEEE
>
> **摘要:** With the broader adoption and highly successful development of Large Language Models (LLMs), there has been growing interest and demand for applying LLMs to autonomous driving technology. Driven by their natural language understanding and reasoning capabilities, LLMs have the potential to enhance various aspects of autonomous driving systems, from perception and scene understanding to interactive decision-making. This paper first introduces the novel concept of designing Large Language Models for Autonomous Driving (LLM4AD), followed by a review of existing LLM4AD studies. Then, a comprehensive benchmark is proposed for evaluating the instruction-following and reasoning abilities of LLM4AD systems, which includes LaMPilot-Bench, CARLA Leaderboard 1.0 Benchmark in simulation and NuPlanQA for multi-view visual question answering. Furthermore, extensive real-world experiments are conducted on autonomous vehicle platforms, examining both on-cloud and on-edge LLM deployment for personalized decision-making and motion control. Next, the future trends of integrating language diffusion models into autonomous driving are explored, exemplified by the proposed ViLaD (Vision-Language Diffusion) framework. Finally, the main challenges of LLM4AD are discussed, including latency, deployment, security and privacy, safety, trust and transparency, and personalization.
>
---
#### [replaced 016] Proprioceptive Image: An Image Representation of Proprioceptive Data from Quadruped Robots for Contact Estimation Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人状态学习任务，旨在解决四足机器人接触估计问题。通过将本体感觉数据转化为二维图像，提升模型的预测精度和泛化能力。**

- **链接: [https://arxiv.org/pdf/2510.14612](https://arxiv.org/pdf/2510.14612)**

> **作者:** Gabriel Fischer Abati; João Carlos Virgolino Soares; Giulio Turrisi; Victor Barasuol; Claudio Semini
>
> **备注:** Accepted to the IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** This paper presents a novel approach for representing proprioceptive time-series data from quadruped robots as structured two-dimensional images, enabling the use of convolutional neural networks for learning locomotion-related tasks. The proposed method encodes temporal dynamics from multiple proprioceptive signals, such as joint positions, IMU readings, and foot velocities, while preserving the robot's morphological structure in the spatial arrangement of the image. This transformation captures inter-signal correlations and gait-dependent patterns, providing a richer feature space than direct time-series processing. We apply this concept in the problem of contact estimation, a key capability for stable and adaptive locomotion on diverse terrains. Experimental evaluations on both real-world datasets and simulated environments show that our image-based representation consistently enhances prediction accuracy and generalization over conventional sequence-based models, underscoring the potential of cross-modal encoding strategies for robotic state learning. Our method achieves superior performance on the contact dataset, improving contact state accuracy from 87.7% to 94.5% over the recently proposed MI-HGNN method, using a 15 times shorter window size.
>
---
#### [replaced 017] Towards Exploratory and Focused Manipulation with Bimanual Active Perception: A New Problem, Benchmark and Strategy
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Exploratory and Focused Manipulation（EFM）任务，解决视觉遮挡导致的信息缺失问题。构建了EFM-10基准和BAP策略，通过双臂协作提升操作性能。**

- **链接: [https://arxiv.org/pdf/2602.01939](https://arxiv.org/pdf/2602.01939)**

> **作者:** Yuxin He; Ruihao Zhang; Tianao Shen; Cheng Liu; Qiang Nie
>
> **备注:** ICRA 2026
>
> **摘要:** Recently, active vision has reemerged as an important concept for manipulation, since visual occlusion occurs more frequently when main cameras are mounted on the robot heads. We reflect on the visual occlusion issue and identify its essence as the absence of information useful for task completion. Inspired by this, we come up with the more fundamental problem of Exploratory and Focused Manipulation (EFM). The proposed problem is about actively collecting information to complete challenging manipulation tasks that require exploration or focus. As an initial attempt to address this problem, we establish the EFM-10 benchmark that consists of 4 categories of tasks that align with our definition (10 tasks in total). We further come up with a Bimanual Active Perception (BAP) strategy, which leverages one arm to provide active vision and another arm to provide force sensing while manipulating. Based on this idea, we collect a dataset named BAPData for the tasks in EFM-10. With the dataset, we successfully verify the effectiveness of the BAP strategy in an imitation learning manner. We hope that the EFM-10 benchmark along with the BAP strategy can become a cornerstone that facilitates future research towards this direction. Project website: this http URL.
>
---
#### [replaced 018] Lightweight Tracking Control for Computationally Constrained Aerial Systems with the Newton-Raphson Method
- **分类: cs.RO; eess.SY; math.OC**

- **简介: 该论文属于飞行器控制任务，旨在解决计算资源受限下的跟踪控制问题。提出基于牛顿-拉夫森方法的轻量级控制器，并通过实验验证其性能优于传统方法。**

- **链接: [https://arxiv.org/pdf/2508.14185](https://arxiv.org/pdf/2508.14185)**

> **作者:** Evanns Morales-Cuadrado; Luke Baird; Yorai Wardi; Samuel Coogan
>
> **摘要:** We investigate the performance of a lightweight tracking controller, based on a flow version of the Newton-Raphson method, applied to a miniature blimp and a mid-size quadrotor. This tracking technique admits theoretical performance guarantees for certain classes of systems and has been successfully applied in simulation studies and on mobile robots with simplified motion models. We evaluate the technique through real-world flight experiments on aerial hardware platforms subject to realistic deployment and onboard computational constraints. The technique's performance is assessed in comparison with established baseline control frameworks of feedback linearization for the blimp, and nonlinear model predictive control for both the quadrotor and the blimp. The performance metrics under consideration are (i) root mean square error of flight trajectories with respect to target trajectories, (ii) algorithms' computation times, and (iii) CPU energy consumption associated with the control algorithms. The experimental findings show that the Newton-Raphson-based tracking controller achieves competitive or superior tracking performance to the baseline methods with substantially reduced computation time and energy expenditure.
>
---
#### [replaced 019] Research on environment perception and behavior prediction of intelligent UAV based on semantic communication
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于智能无人机环境感知与行为预测任务，解决无人机在元宇宙中的高效通信与信息安全问题，提出强化学习、语义通信框架及区块链认证方案。**

- **链接: [https://arxiv.org/pdf/2501.04480](https://arxiv.org/pdf/2501.04480)**

> **作者:** Kechong Ren; Li Gao; Qi Guan
>
> **备注:** The author list of this manuscript is incorrect and incomplete. This version is an unauthorized early draft without approval from all authors
>
> **摘要:** The convergence of drone delivery systems, virtual worlds, and blockchain has transformed logistics and supply chain management, providing a fast, and environmentally friendly alternative to traditional ground transportation methods;Provide users with a real-world experience, virtual service providers need to collect up-to-the-minute delivery information from edge devices. To address this challenge, 1) a reinforcement learning approach is introduced to enable drones with fast training capabilities and the ability to autonomously adapt to new virtual scenarios for effective resource allocation.2) A semantic communication framework for meta-universes is proposed, which utilizes the extraction of semantic information to reduce the communication cost and incentivize the transmission of information for meta-universe services.3) In order to ensure that user information security, a lightweight authentication and key agreement scheme is designed between the drone and the user by introducing blockchain technology. In our experiments, the drone adaptation performance is improved by about 35\%, and the local offloading rate can reach 90\% with the increase of the number of base stations. The semantic communication system proposed in this paper is compared with the Cross Entropy baseline model. Introducing blockchain technology the throughput of the transaction is maintained at a stable value with different number of drones.
>
---
#### [replaced 020] Bi-HIL: Bilateral Control-Based Multimodal Hierarchical Imitation Learning via Subtask-Level Progress Rate and Keyframe Memory for Long-Horizon Contact-Rich Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于长期高接触机器人操作任务，解决部分可观测性和子任务不稳定问题。提出Bi-HIL框架，结合关键帧记忆与进度率，提升长时序协调能力。**

- **链接: [https://arxiv.org/pdf/2603.13315](https://arxiv.org/pdf/2603.13315)**

> **作者:** Thanpimon Buamanee; Masato Kobayashi; Yuki Uranishi
>
> **摘要:** Long-horizon contact-rich robotic manipulation remains challenging due to partial observability and unstable subtask transitions under contact uncertainty. While hierarchical architectures improve temporal reasoning and bilateral imitation learning enables force-aware control, existing approaches often rely on flat policies that struggle with long-horizon coordination. We propose Bi-HIL, a bilateral control-based multimodal hierarchical imitation learning framework for long-horizon manipulation. Bi-HIL stabilizes hierarchical coordination by integrating keyframe memory with subtask-level progress rate that models phase progression within the active subtask and conditions both high- and low-level policies. We evaluate Bi-HIL on unimanual and bimanual real-robot tasks, demonstrating consistent improvements over flat and ablated variants. The results highlight the importance of explicitly modeling subtask progression together with force-aware control for robust long-horizon manipulation. For additional material, please check: this https URL
>
---
#### [replaced 021] Diffusion Forcing for Multi-Agent Interaction Sequence Modeling
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多智能体交互序列建模任务，旨在解决复杂多人互动生成问题。提出MAGNet框架，实现灵活的多智能体运动生成与协调。**

- **链接: [https://arxiv.org/pdf/2512.17900](https://arxiv.org/pdf/2512.17900)**

> **作者:** Vongani H. Maluleke; Kie Horiuchi; Lea Wilken; Evonne Ng; Jitendra Malik; Angjoo Kanazawa
>
> **备注:** Project page: this https URL ; Code: this https URL
>
> **摘要:** Understanding and generating multi-person interactions is a fundamental challenge with broad implications for robotics and social computing. While humans naturally coordinate in groups, modeling such interactions remains difficult due to long temporal horizons, strong inter-agent dependencies, and variable group sizes. Existing motion generation methods are largely task-specific and do not generalize to flexible multi-agent generation. We introduce MAGNet (Multi-Agent Generative Network), a unified autoregressive diffusion framework for multi-agent motion generation that supports a wide range of interaction tasks through flexible conditioning and sampling. MAGNet performs dyadic and polyadic prediction, partner inpainting, partner prediction, and agentic generation all within a single model, and can autoregressively generate ultra-long sequences spanning hundreds of motion steps. We explicitly model inter-agent coupling during autoregressive denoising, enabling coherent coordination across agents. As a result, MAGNet captures both tightly synchronized activities (e.g., dancing, boxing) and loosely structured social interactions. Our approach performs on par with specialized methods on dyadic benchmarks while naturally extending to polyadic scenarios involving three or more interacting people. Please watch the supplemental video, where the temporal dynamics and spatial coordination of generated interactions are best appreciated. Project page: this https URL
>
---
#### [replaced 022] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model
- **分类: cs.RO**

- **简介: 该论文属于自然语言到交通场景生成任务，解决从文本生成合理交通场景的问题。提出TTSG框架，整合LLM实现多智能体行为规划与道路选择，提升场景多样性与安全性。**

- **链接: [https://arxiv.org/pdf/2409.09575](https://arxiv.org/pdf/2409.09575)**

> **作者:** Bo-Kai Ruan; Hao-Tang Tsui; Yung-Hui Li; Hong-Han Shuai
>
> **备注:** Accepted by WAD@CVPR2026
>
> **摘要:** Generating realistic and controllable traffic scenes from natural language can greatly enhance the development and evaluation of autonomous driving systems. However, this task poses unique challenges: (1) grounding free-form text into spatially valid and semantically coherent layouts, (2) composing scenarios without predefined locations, and (3) planning multi-agent behaviors and selecting roads that respect agents' configurations. To address these, we propose a modular framework, TTSG, comprising prompt analysis, road retrieval, agent planning, and a novel plan-aware road ranking algorithm to solve these challenges. While large language models (LLMs) are used as general planners, our design integrates them into a tightly controlled pipeline that enforces structure, feasibility, and scene diversity. Notably, our ranking strategy ensures consistency between agent actions and road geometry, enabling scene generation without predefined routes or spawn points. The framework supports both routine and safety-critical scenarios, as well as multi-stage event composition. Experiments on SafeBench demonstrate that our method achieves the lowest average collision rate (3.5\%) across three critical scenarios. Moreover, driving captioning models trained on our generated scenes improve action reasoning by over 30 CIDEr points. These results underscore our proposed framework for flexible, interpretable, and safety-oriented simulation.
>
---
#### [replaced 023] MIGHTY: Hermite Spline-based Efficient Trajectory Planning
- **分类: cs.RO**

- **简介: 该论文提出MIGHTY，一种基于赫尔米特样条的轨迹规划方法，解决高效且满足约束的路径规划问题，通过时空优化提升计算效率和飞行性能。**

- **链接: [https://arxiv.org/pdf/2511.10822](https://arxiv.org/pdf/2511.10822)**

> **作者:** Kota Kondo; Yuwei Wu; Vijay Kumar; Jonathan P. How
>
> **备注:** 10 pages, 12 figures
>
> **摘要:** Hard-constraint trajectory planners often rely on commercial solvers and demand substantial computational resources. Existing soft-constraint methods achieve faster computation, but either (1) decouple spatial and temporal optimization or (2) restrict the search space. To overcome these limitations, we introduce MIGHTY, a Hermite spline-based planner that performs spatiotemporal optimization while fully leveraging the continuous search space of a spline. In simulation, MIGHTY achieves a 9.3% reduction in computation time and a 13.1% reduction in travel time over state-of-the-art baselines, with a 100% success rate. In hardware, MIGHTY completes multiple high-speed flights up to 6.7 m/s in a cluttered static environment and long-duration flights with dynamically added obstacles.
>
---
#### [replaced 024] An MPC framework for efficient navigation of mobile robots in cluttered environments
- **分类: cs.RO; eess.SY; math.OC**

- **简介: 该论文属于移动机器人导航任务，解决复杂环境中高效路径规划与避障问题。提出一种集成最短路径规划的MPC框架，确保快速响应与安全导航。**

- **链接: [https://arxiv.org/pdf/2509.15917](https://arxiv.org/pdf/2509.15917)**

> **作者:** Johannes Köhler; Daniel Zhang; Raffaele Soloperto; Andrea Carron; Melanie Zeilinger
>
> **备注:** - Code available at: this https URL - Supplementary video: this https URL
>
> **摘要:** We present a model predictive control (MPC) framework for efficient navigation of mobile robots in cluttered environments. The proposed approach integrates a finite-segment shortest path planner into the finite-horizon trajectory optimization of the MPC. This formulation ensures convergence to dynamically selected targets and guarantees collision avoidance, even under general nonlinear dynamics and cluttered environments. The approach is validated through hardware experiments on a small ground robot, where a human operator dynamically assigns target locations that a robot should reach while avoiding obstacles. The robot reached new targets within 2-3 seconds and responded to new commands within 50 ms to 100 ms, immediately adjusting its motion even while still moving at high speeds toward a previous target.
>
---
#### [replaced 025] Joint Magnetometer-IMU Calibration via Maximum A Posteriori Estimation
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于传感器标定任务，旨在提高磁力计与惯性测量单元的联合标定精度与效率。通过最大后验估计方法，同时优化标定参数和姿态轨迹，实现更准确的标定结果。**

- **链接: [https://arxiv.org/pdf/2505.16662](https://arxiv.org/pdf/2505.16662)**

> **作者:** Chuan Huang; Gustaf Hendeby; Isaac Skog
>
> **备注:** Latest version
>
> **摘要:** This paper presents a new approach for jointly calibrating magnetometers and inertial measurement units, focusing on improving calibration accuracy and computational efficiency. The proposed method formulates the calibration problem as a maximum a posteriori estimation problem, treating both the calibration parameters and orientation trajectory of the sensors as unknowns. This formulation enables efficient optimization with closed-form derivatives. The method is compared against two state-of-the-art approaches in terms of computational complexity and estimation accuracy. Simulation results demonstrate that the proposed method achieves lower root mean square error in calibration parameters while maintaining competitive computational efficiency. Further validation through real-world experiments confirms the practical benefits of our approach: it effectively reduces position drift in a magnetic field-aided inertial navigation system by more than a factor of two on most datasets. Moreover, the proposed method calibrated 30 magnetometers in less than 2 minutes. The contributions include a new calibration method, an analysis of existing methods, and a comprehensive empirical evaluation. Datasets and algorithms are made publicly available to promote reproducible research.
>
---
#### [replaced 026] When Should a Robot Think? Resource-Aware Reasoning via Reinforcement Learning for Embodied Robotic Decision-Making
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人决策任务，解决 embodied 机器人在推理与行动间的资源分配问题。提出 RARRL 框架，通过强化学习实现动态推理控制，提升任务成功率与系统效率。**

- **链接: [https://arxiv.org/pdf/2603.16673](https://arxiv.org/pdf/2603.16673)**

> **作者:** Jun Liu; Pu Zhao; Zhenglun Kong; Xuan Shen; Peiyan Dong; Fan Yang; Lin Cui; Hao Tang; Geng Yuan; Wei Niu; Wenbin Zhang; Xue Lin; Gaowen Liu; Yanzhi Wang; Dong Huang
>
> **摘要:** Embodied robotic systems increasingly rely on large language model (LLM)-based agents to support high-level reasoning, planning, and decision-making during interactions with the environment. However, invoking LLM reasoning introduces substantial computational latency and resource overhead, which can interrupt action execution and reduce system reliability. Excessive reasoning may delay actions, while insufficient reasoning often leads to incorrect decisions and task failures. This raises a fundamental question for embodied agents: when should the agent reason, and when should it act? In this work, we propose RARRL (Resource-Aware Reasoning via Reinforcement Learning), a hierarchical framework for resource-aware orchestration of embodied agents. Rather than learning low-level control policies, RARRL learns a high-level orchestration policy that operates at the agent's decision-making layer. This policy enables the agent to adaptively determine whether to invoke reasoning, which reasoning role to employ, and how much computational budget to allocate based on current observations, execution history, and remaining resources. Extensive experiments, including evaluations with empirical latency profiles derived from the ALFRED benchmark, show that RARRL consistently improves task success rates while reducing execution latency and enhancing robustness compared with fixed or heuristic reasoning strategies. These results demonstrate that adaptive reasoning control is essential for building reliable and efficient embodied robotic agents.
>
---
#### [replaced 027] The Role of Consequential and Functional Sound in Human-Robot Interaction: Toward Audio Augmented Reality Interfaces
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，探讨机器人声音对人类感知的影响及增强现实音频的应用。研究解决如何优化机器人声音设计以提升交互体验，通过实验分析声音效果与空间音频定位能力。**

- **链接: [https://arxiv.org/pdf/2511.15956](https://arxiv.org/pdf/2511.15956)**

> **作者:** Aliyah Smith; Monroe Kennedy III
>
> **备注:** 29 pages, 11 figures
>
> **摘要:** Robot sound, encompassing both consequential operational noise and intentionally designed auditory cues, plays an important role in human-robot interaction (HRI). Developing a deeper understanding of how robot sounds influence human experience, and how technologies such as augmented reality (AR) modulate these effects, can enable the design of more socially acceptable robots and more effective, intuitive human-robot interfaces. In this work, we present a three-part mixed-methods study (N = 51) that investigates (i) the effects of consequential robot sounds on human perception under varying degrees of physical colocation, (ii) human accuracy in localizing spatial audio cues delivered via augmented reality, and (iii) the use of augmented spatial audio cues for functional and transformative communication during collaborative handover tasks, in comparison to non-AR sound designs. Contrary to prior findings, our results indicate that the consequential sounds of a Kinova Gen3 manipulator did not negatively affect participants' perceptions of the robot. Participants demonstrated high accuracy in localizing lateral spatial cues, whereas frontal cues proved more challenging, delineating conditions under which spatial auditory feedback is most effective. Qualitative findings further reveal that augmented spatial audio cues can simultaneously convey task-relevant information while fostering a sense of warmth and reducing user discomfort during interaction. Together, these findings elucidate the perceptual effects of consequential robot sound and position sound, particularly augmented spatial audio, as a meaningful yet underutilized design resource for human-robot interaction.
>
---
