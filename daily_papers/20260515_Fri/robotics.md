# 机器人 cs.RO

- **最新发布 38 篇**

- **更新 31 篇**

## 最新发布

#### [new 001] Reactive Planning based Control for Mobile Robots in Obstacle-Cluttered Environments
- **分类: cs.RO**

- **简介: 该论文属于移动机器人路径规划任务，解决在障碍物密集环境中从起点到目标点的无碰撞运动控制问题。提出一种基于反应式规划的控制策略（RPCS），结合局部轨迹调整和自适应跟踪控制实现有效导航。**

- **链接: [https://arxiv.org/pdf/2605.14232](https://arxiv.org/pdf/2605.14232)**

> **作者:** Li Tan; Junlin Xiong; Yan Wang; Wei Ren
>
> **备注:** 7 pages, 7 figures
>
> **摘要:** This paper addresses the motion control problem for mobile robots in obstacle-cluttered environments. The mobile robot has partial environment information only, and aims to move from an initial position to a target position without collisions. For this purpose, a reactive planning based control strategy (RPCS) is proposed. First, the initial and target positions are connected as a reference trajectory. Then, a reactive planning strategy (RPS) is developed to ensure the collision avoidance by modifying the reference trajectory locally based on the partial environment information. Next, an adaptive tracking control strategy (ATCS) is proposed to track the reference trajectory with potentially local modifications via the discretization techniques. Finally, the RPS and ATCS are combined to establish the RPCS, whose efficacy and advantages are illustrated by numerical examples.
>
---
#### [new 002] Let Robots Feel Your Touch: Visuo-Tactile Cortical Alignment for Embodied Mirror Resonance
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于多模态感知任务，旨在解决机器人缺乏触觉共鸣的问题。通过构建Mirror Touch Net，实现视觉与触觉表征的对齐，使机器人能预测触觉并产生共情反应。**

- **链接: [https://arxiv.org/pdf/2605.14571](https://arxiv.org/pdf/2605.14571)**

> **作者:** Tianfang Zhu; Ning An; Rui Wang; Jiasi Gao; Qingming Luo; Anan Li; Guyue Zhou
>
> **摘要:** Observing touch on another's body can elicit corresponding tactile sensations in the observer, a phenomenon termed mirror touch that supports empathy and social perception. This visuo-tactile resonance is thought to rely on structural correspondence between visual and somatosensory cortices, yet robotic systems lack computational frameworks that instantiate this principle. Here we demonstrate that cortical correspondence can be operationalized to endow robots with mirror touch. We introduce Mirror Touch Net, which imposes semantic, distributional and geometric alignment between visual and tactile representations through multi-level constraints, enabling prediction of millimetre-scale tactile signals across 1,140 taxels on a robotic hand from RGB images. Manifold analysis reveals that these constraints reshape visual representations into geometry consistent with the tactile manifold, reducing the complexity of cross-modal mapping. Extending this alignment framework to cross-domain observations of human hands enables tactile prediction and reflexive responses to observed human touch. Our results link a neural principle of visuo-tactile resonance to robotic perception, providing an explainable route towards anticipatory touch and empathic human-robot interaction. Code is available at this https URL.
>
---
#### [new 003] SeaVis: Modeling and Control of a Remotely Operated Towed Vehicle for Seabed Visualization and Mapping
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于水下机器人控制任务，旨在解决海床可视化与地图绘制中的稳定定位问题。提出SeaVis模型及改进的LQR控制器，提升控制精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.14683](https://arxiv.org/pdf/2605.14683)**

> **作者:** Abdelhakim Amer; Aske Alstrup; Frederik Rasmussen; Yury Brodskiy; Andriy Sarabakha; Erdal Kayacan
>
> **备注:** Accepted at IEEE/ASME AIM 2026
>
> **摘要:** High-resolution seafloor mapping necessitates stable and precise positioning for underwater robots. This paper introduces a novel mathematical model for SeaVis remotely operated towed vehicles (ROTVs) and develops a gain-scheduled linear-quadratic regulator (LQR) for robust depth and attitude control. We validate the approach in a high-fidelity simulation, benchmarking the LQR against a conventional PID controller over a challenging seabed profile. The presented results demonstrate the LQR's superior performance, with significantly enhanced robustness to disturbances, greater control efficiency, and substantially reduced flap actuation. The gain scheduling also confirms the controller's effectiveness across the full operational velocity range. The complete simulation environment and controller are open-sourced.
>
---
#### [new 004] Before the Body Moves: Learning Anticipatory Joint Intent for Language-Conditioned Humanoid Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人形机器人控制任务，旨在解决语言指令下机器人动作的前瞻性与实时性问题。提出DAJI框架，实现语言与动作的前瞻性联合意图学习。**

- **链接: [https://arxiv.org/pdf/2605.14417](https://arxiv.org/pdf/2605.14417)**

> **作者:** Haozhe Jia; Honglei Jin; Yuan Zhang; Youcheng Fan; Shaofeng Liang; Lei Wang; Shuxu Jin; Kuimou Yu; Zinuo Zhang; Jianfei Song; Wenshuo Chen; Yutao Yue
>
> **摘要:** Natural language is an intuitive interface for humanoid robots, yet streaming whole-body control requires control representations that are executable now and anticipatory of future physical transitions. Existing language-conditioned humanoid systems typically generate kinematic references that a low-level tracker must repair reactively, or use latent/action policies whose outputs do not explicitly encode upcoming contact changes, support transfers, and balance preparation. We propose \textbf{DAJI} (\emph{Dynamics-Aligned Joint Intent}), a hierarchical framework that learns an anticipatory joint-intent interface between language generation and closed-loop control. DAJI-Act distills a future-aware teacher into a deployable diffusion action policy through student-driven rollouts, while DAJI-Flow autoregressively generates future intent chunks from language and intent history. Experiments show that DAJI achieves strong results in anticipatory latent learning, single-instruction generation, and streaming instruction following, reaching 94.42\% rollout success on HumanML3D-style generation and 0.152 subsequence FID on BABEL.
>
---
#### [new 005] Distill: Uncovering the True Intent behind Human-Robot Communication
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决用户与机器人沟通中意图不明确的问题。提出Distill方法，优化任务描述，更准确理解用户真实意图。**

- **链接: [https://arxiv.org/pdf/2605.14262](https://arxiv.org/pdf/2605.14262)**

> **作者:** Ting Li; David Porfirio
>
> **备注:** 17 pages
>
> **摘要:** As robots become increasingly integrated into everyday environments, intuitive communication paradigms such as natural language and end-user programming have become indispensable for specifying autonomous robot behavior. However, these mechanisms are ineffective at fully capturing user intent: natural language is imprecise and ambiguous, whereas end-user programming can be overly specific. As a result, understanding what users truly mean when they interact with robots remains a central challenge for human-AI communication systems. To address this issue, we propose the Distill approach for human-robot communication interfaces. Given a task specification provided by the user, Distill (1) removes unnecessary steps; (2) generalizes the meaning behind individual steps; and (3) relaxes ordering constraints between steps. We implemented Distill on a web interface and, through a crowdsourcing study, demonstrated its ability to elicit and refine user intent from initial task specifications.
>
---
#### [new 006] Learning Cross-Coupled and Regime Dependent Dynamics for Aerial Manipulation
- **分类: cs.RO**

- **简介: 该论文属于空中机械臂控制任务，旨在解决强耦合、时变动态带来的建模难题。提出一种结构化编解码框架，实现动态的在线自适应学习。**

- **链接: [https://arxiv.org/pdf/2605.14805](https://arxiv.org/pdf/2605.14805)**

> **作者:** Rishabh Dev Yadav; Samaksh Ujjawal; Sihao Sun; Spandan Roy; Wei Pan
>
> **摘要:** Accurate dynamics models are critical for aerial manipulators operating under complex tasks such as payload transport. However, modeling these systems remains fundamentally challenging due to strong quadrotor-manipulator coupling, delayed aerodynamic interactions, and regime-dependent dynamics variations arising from payload changes and manipulator reconfiguration. These effects produce residual dynamics that are simultaneously cross-coupled, history-dependent, and nonstationary, causing both analytical models and purely offline learned models to degrade during deployment. To address these challenges, we propose a structured encoder-decoder framework for adaptive residual dynamics learning in aerial manipulators. The proposed nonlinear latent encoder captures cross-variable coupling and temporal dependencies from state-input histories, while a lightweight linear latent decoder enables online adaptation under regime-dependent nonstationary dynamics. The linear-in-parameter decoder structure permits closed-form Bayesian adaptation together with consistency-driven covariance inflation, enabling rapid and stable adaptation to both transient and slowly varying dynamics changes while remaining compatible with real-time model predictive control (MPC). Experimental results on a real aerial manipulation platform demonstrate improved residual prediction accuracy, faster adaptation under changing operating conditions, and enhanced MPC-based trajectory tracking performance. These results highlight the importance of jointly modeling coupled temporal dynamics and deployment-time nonstationarity for reliable aerial manipulation.
>
---
#### [new 007] Learning Direct Control Policies with Flow Matching for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主驾驶任务，解决控制轨迹生成问题。通过流匹配方法，直接输出加速度和曲率控制序列，实现低延迟、高泛化的实时决策。**

- **链接: [https://arxiv.org/pdf/2605.14832](https://arxiv.org/pdf/2605.14832)**

> **作者:** Marcello Ceresini; Federico Pirazzoli; Andrea Bertogalli; Lorenzo Cipelli; Filippo D'Addeo; Anthony Dell'Eva; Alessandro Paolo Capasso; Alberto Broggi
>
> **备注:** 16 pages, 6 figures, 2 tables. Accepted at IEEE ITSC 2026
>
> **摘要:** We present a flow-matching planner for autonomous driving that directly outputs actionable control trajectories defined by acceleration and curvature profiles. The model is conditioned on a bird's-eye-view (BEV) raster of the surrounding scene and generates control sequences in a small number of Ordinary Differential Equations (ODE) integration steps, enabling low-latency inference suitable for real-time closed-loop re-planning. We train exclusively on urban scenarios (real urban city streets, intersections and roundabouts of the city of Parma, Italy) collected from a 2D traffic simulator with reactive agents, and evaluate in closed-loop on both in-distribution and markedly out-of-distribution environments, including multi-lane highways and unseen urban scenarios. Our results show that the model generalizes reliably to these unseen conditions, maintaining stable closed-loop control and successfully completing scenarios that differ substantially from the training distribution. We attribute this to the BEV representation, which provides a geometry-centric view of the scene that is inherently less sensitive to distributional shifts, and to the flow-matching formulation, which learns a smooth vector field that degrades gracefully under distribution shift. We provide video demonstrations of closed-loop behavior at this https URL.
>
---
#### [new 008] Safety-Constrained Reinforcement Learning with Post-Training Reachability Verification for Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决复杂环境中安全策略的可靠性问题。通过CVaR约束优化和可达性验证，提升策略的安全性与可验证性。**

- **链接: [https://arxiv.org/pdf/2605.14174](https://arxiv.org/pdf/2605.14174)**

> **作者:** Qisong He; Xinmiao Huang; Jinwei Hu; Zhuoyun Li; Yi Dong; Changshun Wu; Xiaowei Huang
>
> **摘要:** Safe navigation for mobile robots demands policies that remain reliable under the high-consequence perception uncertainty of cluttered environments. Yet most existing safe reinforcement learning (RL) methods assess safety through average cumulative cost. Such metrics can mask dangerous tail-risk behaviors. To address this, we propose a framework that trains risk-sensitive policies through Conditional Value-at-Risk (CVaR) constrained optimization on an off-policy TD3 backbone and evaluates their safety margins post-training through neural network reachability verification. During training, the policy is optimized under CVaR constraints on cumulative costs, promoting sensitivity to high-cost tail outcomes rather than average behavior alone. After training, we compute action reachable sets under bounded observation uncertainty using Taylor Model analysis, yielding a safety rate metric that quantifies the proportion of evaluated states at which the policy's reachable action set remains within prescribed safety margins. A key finding is that policies trained with CVaR constraints maintain larger safety margins from obstacles across evaluated states. This makes them significantly more amenable to formal reachability verification. Experiments across ten navigation scenarios and six baselines show that our method achieves a 98.3\% success rate, the highest safety verification rate among all compared methods, while revealing that average cost rankings and reachability-based safety rankings can diverge. This indicates that reachability verification captures risks which are missed by empirical cost metrics alone. We further validate our approach on a physical Clearpath Jackal robot, demonstrating successful sim-to-real transfer.
>
---
#### [new 009] Hand-in-the-Loop: Improving Dexterous VLA via Seamless Interventional Correction
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出HandITL方法，解决高自由度机械手在视觉-语言-动作模型中因人类干预导致的配置突变问题，提升灵巧操作的稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2605.15157](https://arxiv.org/pdf/2605.15157)**

> **作者:** Zhuohang Li; Liqun Huang; Wei Xu; Zhengming Zhu; Nie Lin; Xiao Ma; Xinjun Sheng; Ruoshi Wen
>
> **摘要:** Vision-Language-Action (VLA) models are prone to compounding errors in dexterous manipulation, where high-dimensional action spaces and contact-rich dynamics amplify small policy deviations over long horizons. While Interactive Imitation Learning (IIL) can refine policies through human takeover data, applying it to high-degree-of-freedom (DoF) robotic hands remains challenging due to a command mismatch between human teleoperation and policy execution at the takeover moment, which causes abrupt robot-hand configuration changes, or "gesture jumps". We present Hand-in-the-Loop (HandITL), a seamless human-in-the-loop intervention method that blends human corrective intent with autonomous policy execution to avoid gesture jumps during bimanual dexterous manipulation. Compared with direct teleoperation takeover, HandITL reduces takeover jitter by 99.8% and preserves robust post-takeover manipulation, reducing grasp failures by 87.5% and mean completion time by 19.1%. We validate HandITL on tasks requiring bimanual coordination, tool use, and fine-grained long-horizon manipulation. When used to collect intervention data for policy refinement, HandITL yields policies that outperform those trained with standard teleoperation data by 19% on average across three long-horizon dexterous tasks.
>
---
#### [new 010] IntentVLA: Short-Horizon Intent Modeling for Aliased Robot Manipulation
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出IntentVLA，解决机器人操作中因短时意图模糊导致的执行不稳定问题。通过历史信息建模提升动作块生成稳定性，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2605.14712](https://arxiv.org/pdf/2605.14712)**

> **作者:** Shijie Lian; Bin Yu; Xiaopeng Lin; Zhaolong Shen; Laurence Tianruo Yang; Yurun Jin; Haishan Liu; Changti Wu; Hang Yuan; Cong Huang; Kai Chen
>
> **备注:** Code can be found in this https URL
>
> **摘要:** Robot imitation data are often multimodal: similar visual-language observations may be followed by different action chunks because human demonstrators act with different short-horizon intents, task phases, or recent context. Existing frame-conditioned VLA policies infer each chunk from the current observation and instruction alone, so under partial observability they may resample different intents across adjacent replanning steps, leading to inter-chunk conflict and unstable execution. We introduce IntentVLA, a history-conditioned VLA framework that encodes recent visual observations into a compact short-horizon intent representation and uses it to condition chunk generation. We further introduce AliasBench, a 12-task ambiguity-aware benchmark on RoboTwin2 with matched training data and evaluation environments that isolate short-horizon observation aliasing. Across AliasBench, SimplerEnv, LIBERO, and RoboCasa, IntentVLA improves rollout stability and outperforms strong VLA baselines
>
---
#### [new 011] CoCo-InEKF: State Estimation with Learned Contact Covariances in Dynamic, Contact-Rich Scenarios
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于机器人状态估计任务，解决动态接触场景下的状态估计问题。提出CoCo-InEKF方法，利用学习到的连续接触协方差替代二值接触状态，提升估计精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.15122](https://arxiv.org/pdf/2605.15122)**

> **作者:** Michael Baumgartner; David Müller; Agon Serifi; Ruben Grandia; Espen Knoop; Markus Gross; Moritz Bächer
>
> **备注:** RSS 2026
>
> **摘要:** Robust state estimation for highly dynamic motion of legged robots remains challenging, especially in dynamic, contact-rich scenarios. Traditional approaches often rely on binary contact states that fail to capture the nuances of partial contact or directional slippage. This paper presents CoCo-InEKF, a differentiable invariant extended Kalman filter that utilizes continuous contact velocity covariances instead of binary contact states. These learned covariances allow the method to dynamically modulate contact confidence, accounting for more nuanced conditions ranging from firm contact to directional slippage or no contact. To predict these covariances for a set of predefined contact candidate points, we employ a lightweight neural network trained end-to-end using a state-error loss. This approach eliminates the need for heuristic ground-truth contact labels. In addition, we propose an automated contact candidate selection procedure and demonstrate that our method is insensitive to their exact placement. Experiments on a bipedal robot demonstrate a superior accuracy-efficiency tradeoff for linear velocity estimation, as well as improved filter consistency compared to baseline methods. This enables the robust execution of challenging motions, including dancing and complex ground interactions -- both in simulation and in the real world.
>
---
#### [new 012] SR-Platform: An Agentic Pipeline for Natural Language-Driven Robot Simulation Environment Synthesis
- **分类: cs.RO**

- **简介: 该论文提出SR-Platform，解决机器人仿真环境生成难题。通过自然语言驱动，自动构建MuJoCo环境，降低人工干预，提升效率。**

- **链接: [https://arxiv.org/pdf/2605.14700](https://arxiv.org/pdf/2605.14700)**

> **作者:** Ben Wei Lim; Minh Duc Le; Thang Truong; Thanh Nguyen Canh
>
> **摘要:** Generating robot simulation environments remains a major bottleneck in simulation-based robot learning. Constructing a training-ready MuJoCo scene typically requires expertise in 3D asset modeling, MJCF specification, spatial layout, collision avoidance, and robot-model integration. We present SR-Platform, a production-deployed agentic system that converts free-form natural language descriptions into executable, physically valid MuJoCo environments. SR-Platform decomposes scene synthesis into four stages: an LLM-based orchestrator that converts user intent into a structured scene plan; an asset forge that retrieves cached assets or generates new 3D geometry through LLM-to-CadQuery synthesis; a layout architect that assigns object poses and verifies industrial constraints; and a bridge layer that assembles the final MJCF scene and merges the selected robot model. The system is deployed as a nine-service Docker stack with WebSocket progress streaming, MinIO-backed mesh storage, Qdrant-based semantic asset retrieval, Redis job state, and InfluxDB telemetry. Using 30 days of production telemetry covering 611 successful LLM calls, SR-Platform generates five-object scenes with a median end-to-end latency of approximately 50 s, while cache-accelerated scenes complete in approximately 30-40 s. The asset forge shows an 11.3% first-attempt retry rate with automatic recovery, and cached asset retrieval removes per-object LLM calls for previously generated object types. These results show that agentic scene synthesis can reduce the manual effort required to create diverse robot training environments, enabling users to produce executable MuJoCo scenes from plain English prompts in under one minute.
>
---
#### [new 013] CaMeRL: Collision-Aware and Memory-Enhanced Reinforcement Learning for UAV Navigation in Multi-Scale Obstacle Environments
- **分类: cs.RO**

- **简介: 该论文属于无人机导航任务，解决多尺度障碍物环境下的避障问题。提出CaMeRL框架，结合碰撞感知和记忆机制，提升对小障碍物的敏感性和大障碍物遮挡下的导航能力。**

- **链接: [https://arxiv.org/pdf/2605.14810](https://arxiv.org/pdf/2605.14810)**

> **作者:** Hong Hong; Feiyu Liao; Yongheng Liang; Boning Zhang; Haitao Wang; Hejun Wu
>
> **备注:** 8 pages, 7 figures. Submitted to IEEE Robotics and Automation Letters
>
> **摘要:** In obstacle avoidance navigation of unmanned aerial vehicles (UAVs), variations in obstacle scale have received strangely less attention than obstacle number or density. Existing methods typically extract purely geometric features from single-frame depth observations. Such representations tend to neglect small obstacles and lose spatial context under occlusions caused by large obstacles, leading to noticeable degradation in environments with multi-scale obstacles. To address this issue, we propose CaMeRL, a Collision-aware and Memory-enhanced Reinforcement Learning framework for UAV navigation. The collision-aware latent representation encodes risk-sensitive depth cues to preserve fine-grained obstacle structures, thereby improving sensitivity to small obstacles. The temporal memory module integrates observations across frames, mitigating partial observability caused by large-obstacle occlusions. We evaluate CaMeRL with multi-scale obstacles, including ultra-small and extra-large obstacle settings. Results show that CaMeRL outperforms state-of-the-art baselines across all scales, with success rate gains of 0.48 and 0.28 in the ultra-small and extra-large settings, respectively. More importantly, CaMeRL achieves reliable navigation in cluttered outdoor environments.
>
---
#### [new 014] SOCC-ICP: Semantics-Assisted Odometry based on Occupancy Grids and ICP
- **分类: cs.RO**

- **简介: 该论文提出SOCC-ICP，用于LiDAR里程计任务，解决未知环境中精准位姿估计问题，结合语义占据网格与ICP算法提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.15074](https://arxiv.org/pdf/2605.15074)**

> **作者:** Johannes Scherer; Sebastian Hirt; Henri Meeß
>
> **备注:** 9 pages, 3 figures, Accepted May 2026 for publication in IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Reliable pose estimation in previously unseen environments is a fundamental capability of autonomous systems. Existing LiDAR odometry methods typically employ point-, surfel-, or NDT-based map representations, which are distinct from the semantic occupancy grids commonly used for downstream tasks such as motion planning. We introduce SOCC-ICP, a semantics-assisted odometry framework that jointly performs Semantic OCCupancy grid mapping and LiDAR scan alignment. Each map voxel encodes geometric and semantic statistics, enabling adaptive point-to-point or point-to-plane ICP based on local planarity. Further, the occupancy grid naturally filters dynamic objects through raycasting-based free-space updates. Across diverse evaluation scenarios, SOCC-ICP achieves performance competitive with state-of-the-art LiDAR odometry and remains robust in geometrically degenerate environments, even in the absence of semantic cues. When semantic labels are available, integrating them into map construction, downsampling, and correspondence weighting yields further accuracy gains. By unifying odometry and semantic occupancy grid mapping within a single representation, SOCC-ICP eliminates redundant map structures and directly provides a map suitable for downstream robotic applications.
>
---
#### [new 015] Motion Planning for Autonomous Vehicles using Optimization over Graphs of Convex Sets
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自主车辆运动规划任务，旨在解决复杂环境中生成安全且动态可行轨迹的问题。通过图结构的凸集优化方法，提升计算效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.14199](https://arxiv.org/pdf/2605.14199)**

> **作者:** Matheus Wagner; Antônio Augusto Fröhlich
>
> **摘要:** Motion planning for autonomous vehicles requires generating collision-free and dynamically feasible trajectories in complex environments under real-time constraints. While nonlinear optimal control formulations provide high-fidelity solutions, they are computationally demanding and sensitive to initialization, whereas geometric planning methods scale well but often decouple path selection from trajectory optimization. This paper studies the extent to which optimization over Graphs of Convex Sets (GCS) can approximate solutions of nonlinear optimal control problems in the context of autonomous driving. The free space is represented as a finite union of convex regions organized as a directed graph, allowing nonconvex geometry to be handled through discrete connectivity decisions while maintaining convex trajectory constraints within each region. Vehicle motion is parameterized using Bezier curves for the spatial path and a polynomial time-scaling function for temporal evolution. Under small-slip and linear tire assumptions, a simplified dynamic bicycle model enables approximate enforcement of dynamic feasibility through convex constraints on trajectory derivatives. The approach is evaluated in CommonRoad scenarios involving static obstacle avoidance and lane-changing maneuvers, and is compared against a nonlinear discrete-time optimal control formulation. The results indicate that the GCS-based method generates collision-free and dynamically consistent trajectories that closely match those obtained from the nonlinear program, while exhibiting improved computational efficiency and reduced sensitivity to initialization. These findings suggest that GCS provides a structured approximation of nonlinear motion planning problems, capturing dominant geometric and dynamic effects while preserving convexity in the continuous relaxation.
>
---
#### [new 016] Exploring Bottlenecks in VLM-LLM Navigation: How 3D Scene Understanding Capability Impacts Zero-Shot VLN
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决3D场景理解对零样本VLN的影响。通过分析感知能力与导航性能的关系，提出统计上限并发现感知饱和现象，建议优化核心词汇和边界框比例。**

- **链接: [https://arxiv.org/pdf/2605.14801](https://arxiv.org/pdf/2605.14801)**

> **作者:** Ziyi Xia; Chaoran Xiong; Litao Wei; Xinhao Hu; Ling Pei
>
> **备注:** Accepted by ICRA Workshop MM-Spatial AI, Oral
>
> **摘要:** Zero-shot vision-and-language navigation (VLN) has gained significant attention due to its minimal data collection costs and inherent generalization. This paradigm is typically driven by the integration of pre-trained Vision-Language Models (VLMs) and Large Language Models (LLMs), where VLMs construct 3D scene graphs while LLMs handle high-level reasoning and decision-making. However, a critical bottleneck exists in this system: current 3D perception models prioritize pixel-level accuracy, directly conflicting with the strict computational limits and real-time efficiency demanded by embodied navigation. To address this gap, this paper quantifies the actual impact of 3D scene understanding capability on VLN performance. Based on typical VLM-LLM frameworks, we propose statistical success rate (SR) upper bounds for two core subsystems: 1) the slow LLM planner, which relies on topological mapping semantics, and 2) the fast reactive navigator, which utilizes spatial coordinates and bounding boxes to execute LLM decisions. Evaluations using state-of-the-art 3D scene understanding models validate our proposed bounds and reveal a perception saturation phenomenon, indicating that improvements in perception accuracy beyond a certain threshold yield diminishing returns in navigation success. Our findings suggest that 3D scene understanding for VLN should pivot away from strict pixel-level precision, prioritizing instead navigation-relevant core vocabularies and accurate bounding box proportions.
>
---
#### [new 017] Behavioral Data-Driven Optimal Trajectory Generation for Rotary Cranes
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自动控制任务，旨在减少旋转起重机负载摆动。通过数据驱动方法生成最优轨迹，无需精确模型，提升操作安全与效率。**

- **链接: [https://arxiv.org/pdf/2605.14944](https://arxiv.org/pdf/2605.14944)**

> **作者:** Iskandar Khemakhem; Manuel Zobel; Johannes Schüle; Oliver Sawodny; Naoki Uchiyama; Abdallah Farrage
>
> **摘要:** With the growth of the construction industry and the global shortage of skilled labor, the automation of crane control has become increasingly important for safe and efficient operations. A central challenge in automatic crane control is the reduction of load oscillations during motion, which is primarily addressed through appropriate slewing trajectories. In this context, classical model-based control methods rely on accurate dynamical models and expert tuning, and often struggle to meet safety and precision requirements, while many learning-based approaches require large data sets and significant computational resources. This paper proposes a behavioral data-driven framework for generating open-loop slewing trajectories for rotary cranes that suppress load sway while reducing operation time and energy consumption. The approach builds on Willems' fundamental lemma and its generalizations, to bypass explicit system modeling and operate directly on measured input-output data. A practical workflow is presented in this paper to reduce the need for expert knowledge. Despite the underactuated nature of the crane dynamics, the method identifies a nonparametric representation of the system behavior and generates smooth, optimal trajectories using limited data and convex optimization. The proposed trajectory generation method is validated on a laboratory crane setup and compared against an established model-based approach, achieving up to 35% reduction in load sway, 43% reduction in tracking error, and 50% reduction in travel time.
>
---
#### [new 018] Pelican-Unified 1.0: A Unified Embodied Intelligence Model for Understanding, Reasoning, Imagination and Action
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Pelican-Unified 1.0，一个统一的具身智能模型，解决多模态理解、推理、想象与行动的联合优化问题。通过单一模型实现多项能力，提升整体性能。**

- **链接: [https://arxiv.org/pdf/2605.15153](https://arxiv.org/pdf/2605.15153)**

> **作者:** Yi Zhang; Yinda Chen; Che Liu; Zeyuan Ding; Jin Xu; Shilong Zou; Junwei Liao; Jiayu Hu; Xiancong Ren; Xiaopeng Zhang; Yechi Liu; Haoyuan Shi; Zecong Tang; Haosong Sun; Renwen Cui; Kuishu Wu; Wenhai Liu; Yang Xu; Yingji Zhang; Yidong Wang; Senkang Hu; Jinpeng Lu; Nga Teng Chan; Yechen Wu; Yong Dai; Jian Tang; Xiaozhu Ju
>
> **摘要:** We present Pelican-Unified 1.0, the first embodied foundation model trained according to the principle of unification. Pelican-Unified 1.0 uses a single VLM as a unified understanding module, mapping scenes, instructions, visual contexts, and action histories into a shared semantic space. The same VLM also serves as a unified reasoning module, autoregressively producing task-, action-, and future-oriented chains of thought in a single forward pass and projecting the final hidden state into a dense latent variable. A Unified Future Generator (UFG) then conditions on this latent variable and jointly generates future videos and future actions through two modality-specific output heads within the same denoising process. The language, video, and action losses are all backpropagated into the shared representation, enabling the model to jointly optimize understanding, reasoning, imagination, and action during training, rather than training three isolated expert systems. Experiments demonstrate that unification does not imply compromise. With a single checkpoint, Pelican-Unified 1.0 achieves strong performance across all three capabilities: 64.7 on eight VLM benchmarks, the best among comparable-scale models; 66.03 on WorldArena, ranking first; and 93.5 on RoboTwin, the second-best average among compared action methods. These results show that the unified paradigm succeeds in preserving specialist strength while bringing understanding, reasoning, imagination, and action into one model.
>
---
#### [new 019] DSSP: Diffusion State Space Policy with Full-History Encoding
- **分类: cs.RO**

- **简介: 该论文提出DSSP，解决机器人操作中长序列任务的模糊性问题。通过全历史编码和层次化条件机制，提升策略效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.14598](https://arxiv.org/pdf/2605.14598)**

> **作者:** Zhiyuan Guan; Jianshu Hu; Han Fang; Yunpeng Jiang; Yize Huang; Shujia Li; Xiao Li; Yutong Ban
>
> **摘要:** Diffusion-based imitation learning has shown strong promise for robot manipulation. However, most existing policies condition only on the current observation or a short window of recent observations, limiting their ability to resolve history-dependent ambiguities in long-horizon tasks. To address this, we introduce DSSP, a history-conditioned Diffusion State Space Policy that enables efficient, full-history conditioning for robot manipulation. Leveraging the continuous sequence modeling properties of State Space Models (SSMs), our history encoder effectively compresses the entire observation stream into a compact context representation. To ensure this context preserves critical information regarding future state evolution, the encoder is optimized with a dynamics-aware auxiliary training objective. This high-level context representation is then seamlessly fused with recent state observations to form a hierarchical conditioning mechanism for action generation. Furthermore, to maintain architectural consistency and minimize GPU memory overhead, we also instantiate the diffusion backbone itself using an SSM. Extensive experiments across simulation benchmarks and real-world manipulation tasks show that DSSP achieves state-of-the-art performance with a significantly smaller model size, demonstrating superior efficiency of the hierarchical conditioning in capturing crucial information as the history length increases.
>
---
#### [new 020] Energy-Efficient Quadruped Locomotion with Compliant Feet
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人运动控制任务，旨在提升四足机器人行走效率。通过引入柔性足部，结合强化学习优化控制策略，实验表明适当柔顺度可降低能耗17%。**

- **链接: [https://arxiv.org/pdf/2605.14411](https://arxiv.org/pdf/2605.14411)**

> **作者:** Pramod Pal; Shishir Kolathaya; Ashitava Ghosal
>
> **备注:** 29 pages, 7 figures, supplemental videos link is mentioned in the paper
>
> **摘要:** Quadruped robots are often designed with rigid feet to simplify control and maintain stable contact during locomotion. While this approach is straightforward, it limits the ability of the legs to absorb impact forces and reuse stored elastic energy, leading to higher energy expenditure during locomotion. To explore whether compliant feet can provide an advantage, we integrate foot compliance into a reinforcement learning (RL) locomotion controller and study its effect on walking efficiency. In simulation, we train eight policies corresponding to eight different spring stiffness values and then cross-evaluate their performance by measuring mechanical energy consumed per meter traveled. In experiments done on a developed quadruped, the energy consumption for the intermediate stiffness spring is lower by ~ 17% when compared to a very stiff or a very flexible spring incorporated in the feet, with similar trends appearing in the simulation results. These results indicate that selecting an appropriate foot compliance can improve locomotion efficiency without destabilizing the robot during motion.
>
---
#### [new 021] Ergodic Imitation for Adaptive Exploration around Demonstrations
- **分类: cs.RO**

- **简介: 该论文属于机器人模仿学习任务，解决训练与部署条件不匹配导致的轨迹失效问题。提出自适应遍历模仿方法，结合演示生成适应性探索轨迹。**

- **链接: [https://arxiv.org/pdf/2605.13996](https://arxiv.org/pdf/2605.13996)**

> **作者:** Ziyi Xu; Cem Bilaloglu; Yiming Li; Sylvain Calinon
>
> **备注:** 4 pages, 3 figures
>
> **摘要:** In robotics, a common challenge in imitation learning is the mismatch between training and deployment conditions, caused, for example, by environmental changes or imperfect observation and control. When a robot follows a nominal trajectory under such mismatch, it may become stuck and fail to complete the task. This calls for adaptive online exploration strategies that remain grounded in demonstrations. To this end, we propose an adaptive ergodic imitation approach that constructs a target distribution from the geometry of the retrieved demonstrations and uses it to generate trajectories that adaptively interpolate between tracking and exploration. Our method extends ergodic control beyond its traditional role in area-coverage and search by incorporating demonstrations into a retrieval-based receding-horizon framework for adaptive imitation.
>
---
#### [new 022] FU-MPC: Frontier- and Uncertainty-Aware Model Predictive Control for Efficient and Accurate UAV Exploration with Motorized LiDAR
- **分类: cs.RO**

- **简介: 该论文属于无人机探索任务，解决未知环境中高效且准确的探索问题。提出FU-MPC方法，结合LiDAR旋转控制，提升探索效率与定位可靠性。**

- **链接: [https://arxiv.org/pdf/2605.14920](https://arxiv.org/pdf/2605.14920)**

> **作者:** Jianping Li; Pengfei Wan; Zhongyuan Liu; Yi Wang; Yiheng Chen; Xinhang Xu; Rui Jin; Boyu Zhou; Lihua Xie
>
> **摘要:** Efficient UAV exploration in unknown environments requires rapid coverage expansion while maintaining accurate and reliable localization, since safe navigation in complex scenes depends on consistent mapping and pose estimation. However, for conventional LiDAR-equipped UAVs, the observable region is tightly coupled with the UAV pose and motion. Expanding coverage often requires additional translational or rotational maneuvers, which can reduce exploration efficiency and increase the risk of localization degradation in geometrically challenging environments. Motorized rotating LiDARs provide a promising solution by actively adjusting the sensor viewing direction without changing the UAV motion, thereby introducing an additional sensing degree of freedom. Nevertheless, existing exploration systems rarely exploit this scanning freedom as an explicit decision variable linked to both exploration progress and localization quality. To address this gap, we develop a UAV platform equipped with an independently actuated rotating LiDAR and propose a hierarchical exploration framework. The global planner organizes frontiers into representative viewpoints and sequences them using topology-aware transition costs. Built upon this planner, FU-MPC serves as a local receding-horizon scan controller that optimizes LiDAR rotation along the predicted flight trajectory. The controller jointly considers frontier-aware exploration utility and direction-dependent localization uncertainty, while lightweight surrogate evaluation enables real-time onboard execution. Experiments in complex environments demonstrate that the proposed system improves exploration efficiency while maintaining robust localization performance compared with fixed-pattern scanning and uncertainty-only baselines. The project page can be found at this https URL.
>
---
#### [new 023] A Prototyping Framework for Distributed Control of Multi-Robot Systems
- **分类: cs.RO; cs.MA; eess.SY**

- **简介: 论文提出一种多机器人分布式控制原型框架，解决理论算法与实际测试之间的差距。采用SPMD模式，在单机上模拟分布式控制，验证非合作博弈算法的有效性。**

- **链接: [https://arxiv.org/pdf/2605.15049](https://arxiv.org/pdf/2605.15049)**

> **作者:** Junaid Ahmed Memon; Allan Andre Do Nascimento; Kostas Margellos; Antonis Papachristodoulou
>
> **备注:** Accepted at IFAC World Congress 2026
>
> **摘要:** This paper presents a prototyping framework for distributed control of multi-robot systems, aimed at bridging theory and practical testing of distributed optimization algorithms. Using the Single Program, Multiple Data (SPMD) paradigm, the framework emulates distributed control on a single computer, with each core running the same algorithm using local states and neighbour-to-neighbour communication. We demonstrate the framework on a four-quadrotor position-swapping task using a non-cooperative game-theoretic distributed algorithm. Computational time and trajectory data are compared across the supported dynamics levels: a point-mass model, a high-fidelity quadrotor model, and an experimental hardware testbed using Crazyflie quadcopters. The results show that the framework provides a low-cost and accessible approach for validating distributed algorithms.
>
---
#### [new 024] Chrono-Gymnasium: An Open-Source, Gymnasium-Compatible Distributed Simulation Framework
- **分类: cs.RO**

- **简介: 该论文提出Chrono-Gymnasium，解决高保真仿真计算开销大的问题，通过分布式框架提升机器人和机械系统仿真效率。**

- **链接: [https://arxiv.org/pdf/2605.14911](https://arxiv.org/pdf/2605.14911)**

> **作者:** Bocheng Zou; Harry Zhang; Khailanii Slaton; Jingquan Wang; Derrick Ruan; Huzaifa Mustafa Unjhawala; Radu Serban; Dan Negrut
>
> **摘要:** High-fidelity physics simulation is essential for closing the sim-to-real gap in robotics and complex mechanical systems. However, the computational overhead of high-fidelity engines often limits their use in data-intensive tasks like Reinforcement Learning (RL) and global optimization. We introduce Chrono-Gymnasium, a distributed computing framework that scales the high-fidelity multi-body dynamics of Project Chrono across large-scale computing clusters. Built upon the Ray framework, Chrono-Gymnasium provides a standardized Gymnasium interface, enabling seamless integration with modern machine learning libraries while providing built-in synchronization and messaging primitives for distributed execution. We demonstrate the framework's capabilities through two distinct case studies: (1) the training of an RL agent for autonomous robotic navigation in complex terrains, and (2) the Bayesian Optimization of a planetary lander's design parameters to ensure landing stability. Our results show that Chrono-Gymnasium reduces wall-clock time for high-fidelity simulations without sacrificing physical accuracy, offering a scalable path for the design and control of complex robotic systems.
>
---
#### [new 025] CLOVER: Closed-Loop Value Estimation \& Ranking for End-to-End Autonomous Driving Planning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出CLOVER框架，解决端到端自动驾驶规划中的训练-评估不匹配问题，通过生成多样轨迹并评分排序提升规划性能。**

- **链接: [https://arxiv.org/pdf/2605.15120](https://arxiv.org/pdf/2605.15120)**

> **作者:** Sining Ang; Yuguang Yang; Canyu Chen; Yan Wang
>
> **摘要:** End-to-end autonomous driving planners are commonly trained by imitating a single logged trajectory, yet evaluated by rule-based planning metrics that measure safety, feasibility, progress, and comfort. This creates a training--evaluation mismatch: trajectories close to the logged path may violate planning rules, while alternatives farther from the demonstration can remain valid and high-scoring. The mismatch is especially limiting for proposal-selection planners, whose performance depends on candidate-set coverage and scorer ranking quality. We propose CLOVER, a Closed-LOop Value Estimation and Ranking framework for end-to-end autonomous driving planning. CLOVER follows a lightweight generator--scorer formulation: a generator produces diverse candidate trajectories, and a scorer predicts planning-metric sub-scores to rank them at inference time. To expand proposal support beyond single-trajectory imitation, CLOVER constructs evaluator-filtered pseudo-expert trajectories and trains the generator with set-level coverage supervision. It then performs conservative closed-loop self-distillation: the scorer is fitted to true evaluator sub-scores on generated proposals, while the generator is refined toward teacher-selected top-$k$ and vector-Pareto targets with stability regularization. We analyze when an imperfect scorer can improve the generator, showing that scorer-mediated refinement is reliable when scorer-selected targets are enriched under the true evaluator and updates remain conservative. On NAVSIM, CLOVER achieves 94.5 PDMS and 90.4 EPDMS, establishing a new state of the art. On the more challenging NavHard split, it obtains 48.3 EPDMS, matching the strongest reported result. On supplementary nuScenes open-loop evaluation, CLOVER achieves the lowest L2 error and collision rate among compared methods. Code data will be released at this https URL.
>
---
#### [new 026] MAPLE: Latent Multi-Agent Play for End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MAPLE框架，解决端到端自动驾驶中的多智能体闭环控制问题，通过latent空间训练提升模型反应能力和场景适应性。**

- **链接: [https://arxiv.org/pdf/2605.14201](https://arxiv.org/pdf/2605.14201)**

> **作者:** Rajeev Yasarla; Deepti Hegde; Hsin-Pai Cheng; Shizhong Han; Yunxiao Shi; Meysam Sadeghigooghari; Hanno Ackermann; Litian Liu; Pranav Desai; Fatih Porikli; Mohammad Ghavamzadeh; Hong Cai
>
> **备注:** 19 pages, 9 figures, NeurIPS 2026 submission
>
> **摘要:** Vision-language-action (VLA) models are effective as end-to-end motion planners, but can be brittle when evaluated in closed-loop settings due to being trained under traditional imitation learning framework. Existing closed-loop supervision approaches lack scalability and fail to completely model a reactive environment. We propose MAPLE, a novel framework for reactive, multi-agent rollout of a dynamic driving scenario in the latent space of the VLA model. The ego vehicle and nearby traffic agents are independently controlled over multi-step horizons, while being reactive to other agents in the scene, enabling closed-loop training. MAPLE consists of two training stages: (1) supervised fine-tuning on the latent rollouts based on ground-truth trajectories, followed by (2) reinforcement learning with global and agent -specific rewards that encourage safety, progress, and interaction realism. We further propose diversity rewards that encourage the model to generate planning behaviors that may not be present in logged driving data. Notably, our closed-loop training framework is scalable and does not require external simulators, which can be computationally expensive to run and have limited visual fidelity to the real-world. MAPLE achieves state-of-the-art driving performance on Bench2Drive and demonstrates scalable, closed-loop multi-agent play for robust E2E autonomous driving systems.
>
---
#### [new 027] Towards Robotic Dexterous Hand Intelligence: A Survey
- **分类: cs.RO**

- **简介: 该论文属于机器人灵巧手研究领域，旨在系统梳理硬件、控制方法、数据与评估，解决研究分散、难以比较的问题。**

- **链接: [https://arxiv.org/pdf/2605.13925](https://arxiv.org/pdf/2605.13925)**

> **作者:** Weiguang Zhao; Xihao Guo; Tian Liang; Rui Zhang; Irwin King; Kaizhu Huang
>
> **摘要:** Robotic dexterous hands are central to contact-rich manipulation, with rapid progress driven by advances in hardware, sensing, control, simulation, and data generation. However, existing studies are often developed under different assumptions regarding hand embodiments, sensory configurations, task settings, training data, and evaluation protocols, making systematic comparison difficult and obscuring the developmental trajectory of the field. This survey provides a holistic review of dexterous hand research from four complementary aspects. First, we present a hardware-level analysis covering actuation, transmission, perception, and representative hand designs, highlighting the key trade-offs in force capability, compliance, bandwidth, integration, and system complexity. Furthermore, we review control and learning methods for dexterous manipulation from a methodological perspective, grouping representative works by major paradigms and tracing their evolution in chronological order. In addition, we consolidate datasets, modality design, and evaluation practices, which enables methodological progress to be interpreted together with the ways in which it is trained, benchmarked, and assessed. Finally, we discuss the major limitations of current dexterous hand research and summarize the corresponding future directions. By connecting hardware analysis, methodological development, data resources, and evaluation, this survey aims to provide a structured understanding of dexterous hand research and to clarify the most important open challenges for future study.
>
---
#### [new 028] Behavior Cloning for Active Perception with Low-Resolution Egocentric Vision
- **分类: cs.RO**

- **简介: 论文研究在结构化物体寻找任务中，利用行为克隆实现主动感知。解决低分辨率视角下机器人定位与抓取问题，通过直接从图像预测关节指令完成任务。**

- **链接: [https://arxiv.org/pdf/2605.14106](https://arxiv.org/pdf/2605.14106)**

> **作者:** Anthony Bilic; Chen Chen; Ladislau Bölöni
>
> **摘要:** We investigate whether behavior cloning is sufficient to produce active perception in a structured object-finding task. A low-cost robot arm equipped with a wrist-mounted egocentric RGB camera must reposition to center a partially visible plant before triggering a grasp signal, requiring actions that improve future observations. The model predicts joint commands directly from low-resolution RGB images under closed-loop control. We show that low-resolution egocentric vision is sufficient for reliable task completion and that predicting relative joint deltas substantially outperforms absolute joint position prediction in our setting. These results demonstrate that visually grounded active perception can emerge from behavior cloning in a reproducible setting.
>
---
#### [new 029] Systematic Discovery of Semantic Attacks in Online Map Construction through Conditional Diffusion
- **分类: cs.CV; cs.CR; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶安全任务，旨在发现绕过防御的语义攻击。通过扩散模型生成真实环境变化，干扰高精地图构建，验证了现有防御的不足。**

- **链接: [https://arxiv.org/pdf/2605.14396](https://arxiv.org/pdf/2605.14396)**

> **作者:** Chenyi Wang; Ruoyu Song; Raymond Muller; Jean-Philippe Monteuuis; Jonathan Petit; Z. Berkay Celik; Ryan Gerdes; Ming F. Li
>
> **摘要:** Autonomous vehicles depend on online HD map construction to perceive lane boundaries, dividers, and pedestrian crossings -- safety-critical road elements that directly govern motion planning. While existing pixel perturbation attacks can disrupt the mapping, they can be neutralized by standard adversarial defenses. We present MIRAGE, a framework for systematic discovery of semantic attacks that bypass adversarial defenses and degrade mapping predictions by finding plausible environmental variation (e.g. shadows, wet roads). MIRAGE exploits the latent manifold of real-world data learned by diffusion models, and searches for semantically mutated scenes neighboring the ground truth with the same road topology yet mislead the mapping predictions. We evaluate MIRAGE on nuScenes and demonstrate two attacks: (1) boundary removal, suppressing 57.7% of detections and corrupting 96% of planned trajectories; and (2) boundary injection, the only method that successfully injects fictitious boundaries, while pixel PGD and AdvPatch fail entirely. Both attacks remain potent under various adversarial defenses. We use two independent VLM judges to quantify realism, where MIRAGE passes as realistic 80--84% of the time (vs. 97--99% for clean nuScenes), while AdvPatch only 0--9%. Our findings expose a categorical gap in current adversarial defenses: semantic-level perturbations that manifest as legitimate environmental variation are substantially harder to mitigate than pixel-level perturbations.
>
---
#### [new 030] EARL: Towards a Unified Analysis-Guided Reinforcement Learning Framework for Egocentric Interaction Reasoning and Pixel Grounding
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出EARL框架，解决egocentric视觉中的交互推理和像素定位问题。通过分析引导的强化学习，提升准确性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.14742](https://arxiv.org/pdf/2605.14742)**

> **作者:** Yuejiao Su; Xinshen Zhang; Zhen Ye; Lei Yao; Lap-Pui Chau; Yi Wang
>
> **备注:** Accepted at ICML 2026. Project page: this https URL
>
> **摘要:** Understanding human--environment interactions from egocentric vision is essential for assistive robotics and embodied intelligent agents, yet existing multimodal large language models (MLLMs) still struggle with accurate interaction reasoning and fine-grained pixel grounding. To this end, this paper introduces EARL, an Egocentric Analysis-guided Reinforcement Learning framework that explicitly transfers coarse interaction semantics to query-oriented answering and grounding. Specifically, EARL adopts a two-stage parsing framework including coarse-grained interpretation and fine-grained response. The first stage holistically interprets egocentric interactions and generates a structured textual description. The second stage produces the textual answer and pixel-level mask in response to the user query. To bridge the two stages, we extract a global interaction descriptor as a semantic prior, which is integrated via a novel Analysis-guided Feature Synthesizer (AFS) for query-oriented reasoning. To optimize heterogeneous outputs, including textual answers, bounding boxes, and grounding masks, we design a multi-faceted reward function and train the response stage with GRPO. Experiments on Ego-IRGBench show that EARL achieves 65.48% cIoU for pixel grounding, outperforming previous RL-based methods by 8.37%, while OOD grounding results on EgoHOS indicate strong transferability to unseen egocentric grounding scenarios.
>
---
#### [new 031] Slot-MPC: Goal-Conditioned Model Predictive Control with Object-Centric Representations
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于机器人决策任务，旨在解决传统方法在新情境下泛化能力差的问题。通过引入基于物体的表示和模型预测控制，提升任务性能与规划效率。**

- **链接: [https://arxiv.org/pdf/2605.14937](https://arxiv.org/pdf/2605.14937)**

> **作者:** Jonathan Spieler; Angel Villar-Corrales; Sven Behnke
>
> **摘要:** Predictive world models enable agents to model scene dynamics and reason about the consequences of their actions. Inspired by human perception, object-centric world models capture scene dynamics using object-level representations, which can be used for downstream applications such as action planning. However, most object-centric world models and reinforcement learning (RL) approaches learn reactive policies that are fixed at inference time, limiting generalization to novel situations. We propose Slot-MPC, an object-centric world modeling framework that enables planning through Model Predictive Control (MPC). Slot-MPC leverages vision encoders to learn slot-based representations, which encode individual objects in the scene, and uses these structured representations to learn an action-conditioned object-centric dynamics model. At inference time, the learned dynamics model enables action planning via MPC, allowing agents to adapt to previously unseen situations. Since the learned world model is differentiable, we can use gradient-based MPC to directly optimize actions, which is computationally more efficient than relying on gradient-free, sampling-based MPC methods. Experiments on simulated robotic manipulation tasks show that Slot-MPC improves both task performance and planning efficiency compared to non-object-centric world model baselines. In the considered offline setting with limited state-action coverage, we find that gradient-based MPC performs better than gradient-free, sampling-based MPC. Our results demonstrate that explicitly structured, object-centric representations provide a strong inductive bias for controllable and generalizable decision-making. Code and additional results are available at this https URL.
>
---
#### [new 032] WarmPrior: Straightening Flow-Matching Policies with Temporal Priors
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于机器人控制领域，解决生成策略中的分布设计问题。通过引入基于动作历史的WarmPrior，提升任务成功率和样本效率。**

- **链接: [https://arxiv.org/pdf/2605.13959](https://arxiv.org/pdf/2605.13959)**

> **作者:** Sinjae Kang; Chanyoung Kim; Kaixin Wang; Li Zhao; Kimin Lee
>
> **摘要:** Generative policies based on diffusion and flow matching have become a dominant paradigm for visuomotor robotic control. We show that replacing the standard Gaussian source distribution with WarmPrior, a simple temporally grounded prior constructed from readily available recent action history, consistently improves success rates on robotic manipulation tasks. We trace this gain to markedly straighter probability paths, echoing the effect of optimal-transport couplings in Rectified Flow. Beyond standard behavior cloning, WarmPrior also reshapes the exploration distribution in prior-space reinforcement learning, improving both sample efficiency and final performance. Collectively, these results identify the source distribution as an important and underexplored design axis in generative robot control.
>
---
#### [new 033] SceneFunRI: Reasoning the Invisible for Task-Driven Functional Object Localization
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出SceneFunRI，解决任务驱动的不可见功能物体定位问题，通过空间推理提升模型在复杂场景中的理解能力。**

- **链接: [https://arxiv.org/pdf/2605.14704](https://arxiv.org/pdf/2605.14704)**

> **作者:** Posheng Chen; Powen Cheng; Gueter Josmy Faure; Hung-Ting Su; Winston H. Hsu
>
> **摘要:** In real-world scenes, target objects may reside in regions that are not visible. While humans can often infer the locations of occluded objects from context and commonsense knowledge, this capability remains a major challenge for vision-language models (VLMs). To address this gap, we introduce SceneFunRI, a benchmark for Reasoning the Invisible. Based on the SceneFun3D dataset, SceneFunRI formulates the task as a 2D spatial reasoning problem via a semi-automatic pipeline and comprises 855 instances. It requires models to infer the locations of invisible functional objects from task instructions and commonsense reasoning. The strongest baseline model (Gemini 3 Flash) only achieves an CAcc@75 of 15.20, an mIoU of 0.74, and a Dist of 28.65. We group our prompting analysis into three categories: Strong Instruction Prompting, Reasoning-based Prompting, and Spatial Process of Elimination (SPoE). These findings indicate that invisible-region reasoning remains an unstable capability in current VLMs, motivating future work on models that more tightly integrate task intent, commonsense priors, spatial grounding, and uncertainty-aware search.
>
---
#### [new 034] Vision-Based Runtime Monitoring under Varying Specifications using Semantic Latent Representations
- **分类: cs.LG; cs.CV; cs.RO; eess.SY**

- **简介: 该论文研究视觉驱动的运行时监控任务，解决在部分可观测环境下如何可靠验证ptSTL公式的问题。通过引入语义基和滚动预测监控方法，实现高效、可复用的实时安全验证。**

- **链接: [https://arxiv.org/pdf/2605.13923](https://arxiv.org/pdf/2605.13923)**

> **作者:** Bardh Hoxha; Oliver Schön; Hideki Okamoto; Lars Lindemann; Georgios Fainekos
>
> **摘要:** We study certified runtime monitoring of past-time signal temporal logic (ptSTL) from visual observations under partial observability. The monitor must infer safety-relevant quantities from images and provide finite-sample guarantees, while being \emph{reusable}: once trained and calibrated, it should certify any formula in a target fragment without per-formula retraining. For fragments induced by a finite dictionary of temporal atoms, we prove that the \emph{semantic basis}, the vector of atom robustness scores, is the minimum prediction target within the class of monotone, 1-Lipschitz reusable interfaces: any formula is evaluated by a deterministic decoder derived from the parse tree, and a single conformal calibration pass certifies the entire fragment with no union bound. We also introduce a \emph{rolling prediction monitor} that predicts only current predicate values and reconstructs temporal history online; this is easier to learn but grows conservative at long horizons. On a pedestrian-crossroad benchmark, rolling achieves tighter certified bounds at short horizons while the semantic-basis monitor is up to 4-times tighter at long horizons. We validate the presented monitors on real-world Waymo driving data, where both monitors satisfy the conformal coverage guarantee empirically.
>
---
#### [new 035] Articraft: An Agentic System for Scalable Articulated 3D Asset Generation
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文属于3D资产生成任务，旨在解决 articulated 3D 对象数据不足的问题。通过引入 agentic 系统 Articraft，利用 LLM 自动生成高质量 3D 资产。**

- **链接: [https://arxiv.org/pdf/2605.15187](https://arxiv.org/pdf/2605.15187)**

> **作者:** Matt Zhou; Ruining Li; Xiaoyang Lyu; Zhaomou Song; Zhening Huang; Chuanxia Zheng; Christian Rupprecht; Andrea Vedaldi; Shangzhe Wu
>
> **备注:** Project page: this https URL
>
> **摘要:** A bottleneck in learning to understand articulated 3D objects is the lack of large and diverse datasets. In this paper, we propose to leverage large language models (LLMs) to close this gap and generate articulated assets at scale. We reduce the problem of generating an articulated 3D asset to that of writing a program that builds it. We then introduce a new agentic system, Articraft, that writes such programs automatically. We design a programmatic interface and harness to help the LLM do so effectively. The LLM writes code against a domain-specific SDK for defining parts, composing geometry, specifying joints, and writing tests to validate the resulting assets. The harness exposes a restricted workspace and interface to the LLM, validates the resulting assets, and returns structured feedback. In this way, the LLM is not distracted by details such as authoring a URDF file or managing a complex software environment. We show that this produces higher-quality assets than both state-of-the-art articulated-asset generators and general-purpose coding agents. Using Articraft, we build Articraft-10K, a curated dataset of over 10K articulated assets spanning 245 categories, and show its utility both for training models of articulated assets and in downstream applications such as robotics simulation and virtual reality.
>
---
#### [new 036] DiffPhD: A Unified Differentiable Solver for Projective Heterogeneous Materials in Elastodynamics with Contact-Rich GPU-Acceleration
- **分类: cs.GR; cs.DC; cs.RO; math.NA**

- **简介: 该论文提出DiffPhD，解决软体物理模拟中的异质材料、大变形和接触问题，通过GPU加速实现高效且稳定的梯度计算。**

- **链接: [https://arxiv.org/pdf/2605.14526](https://arxiv.org/pdf/2605.14526)**

> **作者:** Shih-Yu Lai; Sung-Han Tien; Jui-I Huang; Yen-Chen Tseng; Yi-Ting Chiu; Siyuan Luo; Ziqiu Zeng; Fan Shi; Peter Yichen Chen; Tiantian Liu; Yu-Lun Liu; Bing-Yu Chen
>
> **摘要:** Differentiable simulation of soft bodies is a foundation for system identification, trajectory optimization, and Real2Sim transfer. Yet, existing methods such as the differentiable Projective Dynamics (DiffPD) struggle when faced with heterogeneous materials with extreme stiffness contrasts, hyperelasticity under large deformations, and contact-rich interactions, which are common scenarios in the real world. We present DiffPhD, a unified GPU-accelerated differentiable Projective Dynamics framework for heterogeneous materials that tackles these intertwined challenges simultaneously. Our key insight is a careful integration of: (i) stiffness-aware projective weights to embed heterogeneity into the global system; (ii) trust-region eigenvalue filtering lifted to the backward pass for stable hyperelastic gradients and a type-II Anderson Acceleration scheme with dual-gate convergence to stabilize forward iteration under large stiffness contrasts; and (iii) a unified GPU pipeline that reuses a single sparse factor across forward, backward, and contact computations, with stiffness-amplified Rayleigh damping folded into the same factor for heterogeneity-aware dissipation at zero recurring cost. DiffPhD achieves strict gradient accuracy while delivering up to an order-of-magnitude speedup over prior differentiable solvers on heterogeneous, hyperelastic, contact-rich benchmarks. Crucially, this speedup does not come at the cost of stability: DiffPhD remains convergent on stiffness contrasts up to 100x where prior PD solvers degrade. This unlocks end-to-end gradient-based optimization on regimes previously bottlenecked by either solver fragility or per-iteration cost -- shell--joint composite creatures, soft characters wielding stiff weapons, and soft-gripper robotic manipulation -- all handled within a single forward--backward pass.
>
---
#### [new 037] SToRe3D: Sparse Token Relevance in ViTs for Efficient Multi-View 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对多视角3D目标检测任务，解决ViTs在密集token处理中导致的高延迟问题。提出SToRe3D框架，通过稀疏化选择关键2D和3D信息，提升推理速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2605.14110](https://arxiv.org/pdf/2605.14110)**

> **作者:** Sandro Papais; Lezhou Feng; Charles Cossette; Lingting Ge
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Vision Transformers (ViTs) enable strong multi-view 3D detection but are limited by high inference latency from dense token and query processing across multiple views and large 3D regions. Existing sparsity methods, designed mainly for 2D vision, prune or merge image tokens but do not extend to full-model sparsity or address 3D object queries. We introduce SToRe3D, a relevance-aligned sparsity framework that jointly selects 2D image tokens and 3D object queries while storing filtered features for reactivation. Mutual 2D-3D relevance heads allocate compute to driving-critical content and preserve other embeddings. Evaluated on nuScenes and our new nuScenes-Relevance benchmark, SToRe3D achieves up to 3x faster inference with marginal accuracy loss, establishing real-time large-scale ViT-based 3D detection while maintaining accuracy on planning-critical agents.
>
---
#### [new 038] Evo-Depth: A Lightweight Depth-Enhanced Vision-Language-Action Model
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Evo-Depth，解决机器人操作中空间理解不足的问题。通过轻量级深度增强模块，提升视觉-语言-动作模型的定位精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.14950](https://arxiv.org/pdf/2605.14950)**

> **作者:** Tao Lin; Yuxin Du; Jiting Liu; Nuobei Zhu; Yunhe Li; Yuqian Fu; Yinxinyu Chen; Hongyi Cai; Zewei Ye; Bing Cheng; Kai Ye; Yiran Mao; Yilei Zhong; MingKang Dong; Junchi Yan; Gen Li; Bo Zhao
>
> **摘要:** Vision-Language-Action models have emerged as a promising paradigm for robotic manipulation by unifying perception, language grounding, and action generation. However, they often struggle in scenarios requiring precise spatial understanding, as current VLA models primarily rely on 2D visual representations that lack depth information and detailed spatial relationships. While recent approaches incorporate explicit 3D inputs such as depth maps or point clouds to address this issue, they often increase system complexity, require additional sensors, and remain vulnerable to sensing noise and reconstruction errors. Another line of work explores implicit 3D-aware spatial modeling directly from RGB observations without extra sensors, but it often relies on large geometry foundation models, resulting in higher training and deployment costs. To address these challenges, we propose Evo-Depth, a lightweight depth-enhanced VLA framework that enhances spatially grounded manipulation without relying on additional sensing hardware or compromising deployment efficiency. Evo-Depth employs a lightweight Implicit Depth Encoding Module to extract compact depth features from multi-view RGB images. These features are incorporated into vision-language representations through a Spatial Enhancement Module via depth-aware modulation, enabling efficient spatial-semantic enhancement. A Progressive Alignment Training strategy is further introduced to align the resulting depth-enhanced representations with downstream action learning. With only 0.9B parameters, Evo-Depth achieves superior performance across four simulation benchmarks. In real-world experiments, Evo-Depth attains the highest average success rate while also exhibiting the smallest model size, lowest GPU memory usage, and highest inference frequency among compared methods.
>
---
## 更新

#### [replaced 001] AutoMoT: A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出AutoMoT，一个统一的视觉-语言-动作模型，用于端到端自动驾驶，解决推理与行动空间不匹配、推理能力未充分利用及推理延迟问题。**

- **链接: [https://arxiv.org/pdf/2603.14851](https://arxiv.org/pdf/2603.14851)**

> **作者:** Wenhui Huang; Songyan Zhang; Qihang Huang; Zhidong Wang; Zhiqi Mao; Collister Chua; Zhan Chen; Long Chen; Chen Lv
>
> **摘要:** Integrating vision-language models (VLMs) into end-to-end (E2E) autonomous driving (AD) systems has shown promise in improving scene understanding. However, existing integration strategies suffer from several limitations: they either struggle to resolve distribution misalignment between reasoning and action spaces, underexploit the general reasoning capabilities of pretrained VLMs, or incur substantial inference latency during action policy generation, which degrades driving performance. To address these challenges, we propose AutoMoT in this work, an end-to-end AD framework that unifies reasoning and action generation within a single vision-language-action (VLA) model. Our approach leverages a mixture-of-transformer (MoT) architecture with joint attention sharing, which preserves the general reasoning capabilities of pre-trained VLMs while enabling efficient fast-slow inference through asynchronous execution at different task frequencies. Extensive experiments on multiple benchmarks, under both open- and closed-loop settings, demonstrate that AutoMoT achieves competitive performance compared to state-of-the-art methods. We further investigate the functional boundary of pre-trained VLMs in AD, examining when AD-tailored fine-tuning is necessary. Our results show that pre-trained VLMs can achieve competitive multi-task scene understanding performance through semantic prompting alone, while fine-tuning remains essential for action-level tasks such as decision-making and trajectory planning. We refer to this https URL for the demonstration videos and qualitative results.
>
---
#### [replaced 002] ActivePusher: Active Learning and Planning with Residual Physics for Nonprehensile Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于非抓取操作任务，旨在解决学习模型数据效率低和不确定性问题。提出ActivePusher框架，结合主动学习与残差物理建模，提升规划可靠性与成功率。**

- **链接: [https://arxiv.org/pdf/2506.04646](https://arxiv.org/pdf/2506.04646)**

> **作者:** Zhuoyun Zhong; Seyedali Golestaneh; Constantinos Chamzas
>
> **备注:** Accepted by the 2026 IEEE International Conference on Robotics & Automation (ICRA 2026)
>
> **摘要:** Planning with learned dynamics models offers a promising approach toward versatile real-world manipulation, particularly in nonprehensile settings such as pushing or rolling, where accurate analytical models are difficult to obtain. However, collecting training data for learning-based methods can be costly and inefficient, as it often relies on randomly sampled interactions that are not necessarily the most informative. Furthermore, learned models tend to exhibit high uncertainty in underexplored regions of the skill space, undermining the reliability of long-horizon planning. To address these challenges, we propose ActivePusher, a novel framework that combines residual-physics modeling with uncertainty-based active learning, to focus data acquisition on the most informative skill parameters. Additionally, ActivePusher seamlessly integrates with model-based kinodynamic planners, leveraging uncertainty estimates to bias control sampling toward more reliable actions. We evaluate our approach in both simulation and real-world environments, and demonstrate that it consistently improves data efficiency and achieves higher planning success rates in comparison to baseline methods. The source code is available at this https URL.
>
---
#### [replaced 003] Co-Me: Confidence-Guided Token Merging for Visual Geometric Transformers
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Co-Me，用于加速视觉几何Transformer。解决模型计算效率低的问题，通过置信度引导合并低置信度token，提升速度并保持性能。属于视觉Transformer优化任务。**

- **链接: [https://arxiv.org/pdf/2511.14751](https://arxiv.org/pdf/2511.14751)**

> **作者:** Yutian Chen; Yuheng Qiu; Ruogu Li; Ali Agha; Shayegan Omidshafiei; Jay Patrikar; Sebastian Scherer
>
> **摘要:** We propose Confidence-Guided Token Merging (Co-Me), an acceleration mechanism for visual geometric transformers without retraining or finetuning the base model. Co-Me distilled a light-weight confidence predictor to rank tokens by uncertainty and selectively merge low-confidence ones, effectively reducing computation while maintaining spatial coverage. Compared to similarity-based merging or pruning, the confidence signal in Co-Me reliably indicates regions emphasized by the transformer, enabling substantial acceleration without degrading performance. Co-Me applies seamlessly to various multi-view and streaming visual geometric transformers, achieving speedups that scale with sequence length. When applied to VGGT and Pi3, Co-Me achieves up to 21.5x and 20.4x speedup, making visual geometric transformers practical for real-time 3D perception and reconstruction.
>
---
#### [replaced 004] Terminal Matters: Kinodynamic Planning with a Terminal Cost and Learned Uncertainty in Belief State-Cost Space
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决在不确定性下可靠到达目标的问题。提出终端成本优化方法，提升规划可靠性与目标偏好。**

- **链接: [https://arxiv.org/pdf/2605.09046](https://arxiv.org/pdf/2605.09046)**

> **作者:** Zhuoyun Zhong; Seyedali Golestaneh; Constantinos Chamzas
>
> **摘要:** In many real-world robotic tasks, robots must generate dynamically feasible motions that reliably reach desired goals even under uncertainty. Yet existing sampling-based kinodynamic planners typically optimize accumulated trajectory costs and treat goal reaching as a feasibility check, rather than explicitly optimizing terminal-state quality, such as goal preference or goal-reaching reliability. In this work, we introduce a terminal-cost formulation for kinodynamic planning that allows terminal-state quality to be optimized alongside accumulated trajectory cost. We prove that AO-RRT, an asymptotically optimal kinodynamic planner, preserves its asymptotic optimality under this augmented objective. We further extend the formulation to belief space and prove that minimizing the Wasserstein distance between the terminal belief and the goal improves a lower bound on the probability of reaching the goal region. The resulting planner, KiTe, uses this terminal-cost objective to encode goal preferences and improve reliability under uncertainty. To support systems without analytical uncertainty models, we learn dynamics and process uncertainty directly from data and integrate the learned belief dynamics into planning. Experiments on Flappy Bird, Car Parking, and Planar Pushing show that KiTe consistently improves goal-reaching success under uncertainty. Real-world Planar Pushing experiments further demonstrate that KiTe can plan effectively with learned dynamics and uncertainty. Source code is available at this https URL.
>
---
#### [replaced 005] RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RoboLab，一个用于评估任务通用机器人策略泛化能力的仿真基准。解决现有基准泛化测试不足的问题，通过生成多样化任务和分析策略敏感性，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2604.09860](https://arxiv.org/pdf/2604.09860)**

> **作者:** Xuning Yang; Rishit Dagli; Alex Zook; Hugo Hadfield; Ankit Goyal; Stan Birchfield; Fabio Ramos; Jonathan Tremblay
>
> **摘要:** The pursuit of general-purpose robotics has yielded impressive foundation models, yet simulation-based benchmarking remains a bottleneck due to rapid performance saturation and a lack of true generalization testing. Existing benchmarks often exhibit significant domain overlap between training and evaluation, trivializing success rates and obscuring insights into robustness. We introduce RoboLab, a simulation benchmarking framework designed to address these challenges. Concretely, our framework is designed to answer two questions: (1) to what extent can we understand the performance of a real-world policy by analyzing its behavior in simulation, and (2) which factor most strongly affect policy behavior. First, RoboLab enables human-authored and LLM-enabled generation of scenes and tasks in a robot- and policy-agnostic manner within a high-fidelity simulation environment. We introduce an accompanying RoboLab-120 benchmark, consisting of 120 tasks categorized into three competency axes: visual, procedural, relational, across three difficulty levels. Second, we introduce a systematic analysis of real-world policies that quantify both their performance and the sensitivity of their behavior to controlled perturbations, exposing significant performance gap in current state-of-the-art models. By providing granular metrics and a scalable toolset, RoboLab offers a scalable framework for evaluating the true generalization capabilities of task-generalist robotic policies. Project website: this https URL.
>
---
#### [replaced 006] MonoSpheres: Large-Scale Monocular SLAM-Based UAV Exploration through Perception-Coupled Mapping and Planning
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决单目视觉下大范围未知环境探索问题。通过结合感知与规划，提升稀疏深度数据下的地图构建与路径规划能力。**

- **链接: [https://arxiv.org/pdf/2511.17299](https://arxiv.org/pdf/2511.17299)**

> **作者:** Tomáš Musil; Matěj Petrlík; Martin Saska
>
> **备注:** 8 pages, 9 figures, accepted to IEEE Robotics and Automation Letters
>
> **摘要:** Autonomous exploration of unknown environments is a key capability for mobile robots, but it is largely unsolved for robots equipped with only a single monocular camera and no dense range sensors. In this paper, we present a novel approach to monocular vision-based exploration that can safely cover large-scale unstructured indoor and outdoor 3D environments by explicitly accounting for the properties of a sparse monocular SLAM frontend in both mapping and planning. The mapping module solves the problems of sparse depth data, free-space gaps, and large depth uncertainty by oversampling free space in texture-sparse areas and keeping track of obstacle position uncertainty. The planning module handles the added free-space uncertainty through rapid replanning and perception-aware heading control. We further show that frontier-based exploration is possible with sparse monocular depth data when parallax requirements and the possibility of textureless surfaces are taken into account. We evaluate our approach extensively in diverse real-world and simulated environments, including ablation studies. To the best of the authors' knowledge, the proposed method is the first to achieve 3D monocular exploration in real-world unstructured outdoor environments. We open-source our implementation to support future research.
>
---
#### [replaced 007] Learning Dynamic Rope Manipulation Using Task-Level Iterative Learning Control
- **分类: cs.RO**

- **简介: 该论文研究动态绳索操作任务，解决如何通过少量示范数据实现绳索操控的问题。提出一种基于任务级迭代学习控制的方法，直接在硬件上学习，适用于多种绳索类型。**

- **链接: [https://arxiv.org/pdf/2602.21302](https://arxiv.org/pdf/2602.21302)**

> **作者:** Krishna Suresh; Chris Atkeson
>
> **备注:** Project website: this https URL
>
> **摘要:** We introduce a Task-Level Iterative Learning Control method for dynamic manipulation of ropes. We demonstrate this method on a non-planar rope manipulation task called the flying knot. Using a single human demonstration and a simplified rope model, the method learns directly on hardware without reliance on large amounts of demonstration data or massive amounts of simulation. At each iteration, the algorithm inverts a model of the robot and rope by solving a quadratic program to propagate task-space errors into action updates. We evaluate performance across 7 different kinds of ropes, including chain, latex surgical tubing, and braided and twisted ropes, ranging in thicknesses of 7--25\,mm and densities of 0.013--0.5\,kg/m. Learning achieves a 100\% success rate within 10 trials on all ropes. Furthermore, the method can successfully transfer between most rope types in 2--5 trials. this https URL
>
---
#### [replaced 008] Overcoming Dynamics-Blindness: Training-Free Pace-and-Path Correction for VLA Models
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言-动作（VLA）模型任务，解决模型对动态环境感知不足的问题。提出无需训练的Pace-and-Path Correction方法，提升模型在动态场景中的表现。**

- **链接: [https://arxiv.org/pdf/2605.11459](https://arxiv.org/pdf/2605.11459)**

> **作者:** Yanyan Zhang; Chaoda Song; Vikash Singh; Xinpeng Li; Kai Ye; Zhe Hu; Zhongzhu Pu; Yu Yin; Vipin Chaudhary
>
> **摘要:** Vision-Language-Action (VLA) models achieve remarkable flexibility and generalization beyond classical control paradigms. However, most prevailing VLAs are trained under a single-frame observation paradigm, which leaves them structurally blind to temporal dynamics. Consequently, these models degrade severely in non-stationary scenarios, even when trained or finetuned on dynamic datasets. Existing approaches either require expensive retraining or suffer from latency bottlenecks and poor temporal consistency across action chunks. We propose Pace-and-Path Correction, a training-free, closed-form inference-time operator that wraps any chunked-action VLA. From a single quadratic cost, joint minimization yields a unified solution that decomposes orthogonally into two distinct channels. The pace channel compresses execution along the planned direction, while the path channel applies an orthogonal spatial offset, jointly absorbing the perceived dynamics within the chunk window. We evaluate our approach on a comprehensive diagnostic benchmark MoveBench designed to isolate motion as the sole controlled variable. Empirical results demonstrate that our framework consistently outperforms state-of-the-art training-free wrappers and dynamic-adaptive methods and improves success rates by up to 28.8% and 25.9% in absolute terms over foundational VLA models in dynamic-only and static-dynamic mixed environments, respectively.
>
---
#### [replaced 009] Action Emergence from Streaming Intent
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决动作生成问题。通过引入Streaming Intent机制，实现基于场景理解的意图驱动动作生成，提升模型的可控性和多样性。**

- **链接: [https://arxiv.org/pdf/2605.12622](https://arxiv.org/pdf/2605.12622)**

> **作者:** Pengfei Jing; Victor Shea-Jay Huang; Hengtong Lu; Jifeng Dai; Yan Xie; Benjin Zhu
>
> **备注:** Project page: this https URL
>
> **摘要:** We formalize action emergence as a target capability for end-to-end autonomous driving: the ability to generate physically feasible, semantically appropriate, and safety-compliant actions in arbitrary, long-tail traffic scenes through scene-conditioned reasoning rather than retrieval or interpolation of learned scene-action mappings. We show that previous paradigms cannot deliver action emergence: autoregressive trajectory decoders collapse the inherently multimodal future into a single averaged output, while diffusion and flow-matching generators express multimodality but are not steerable by reasoned intent. We propose Streaming Intent as a concrete way to approach action emergence: a mechanism that makes driving intent (i) semantically streamed through a continuous chain-of-thought that causally derives the intent from scene understanding, and (ii) temporally streamed across clips so that intent commitments remain coherent along the driving horizon. We realize Streaming Intent in a VLA model we call SI (Streaming Intent). SI autoregressively decodes a four-step chain-of-thought and emits an intent token; the decoded intent then drives classifier-free guidance (CFG) on a flow-matching action head, requiring only two denoising steps to generate the final trajectory. On the Waymo End-to-End benchmark, SI achieves competitive aggregate performance, with an RFS score of 7.96 on the validation set and 7.74 on the test set. Beyond aggregate metrics, the model demonstrates -- to our knowledge for the first time in a fully end-to-end VLA -- intent-faithful controllability: for a fixed scene, varying the intent class at inference yields qualitatively distinct yet consistently high-quality plans, arising purely from data-driven learning without any pre-built trajectory bank or hand-coded post-hoc selector.
>
---
#### [replaced 010] SoFFT: Spatial Fourier Transform for Modeling Continuum Soft Robots
- **分类: cs.RO**

- **简介: 该论文属于软体机器人建模任务，旨在解决其高自由度带来的建模复杂问题。通过将机器人的骨架视为时空信号并应用傅里叶变换，实现更简洁准确的变形描述。**

- **链接: [https://arxiv.org/pdf/2502.17347](https://arxiv.org/pdf/2502.17347)**

> **作者:** Daniele Caradonna; Diego Bianchi; Franco Angelini; Egidio Falotico
>
> **摘要:** Continuum soft robots, composed of flexible materials, exhibit theoretically infinite degrees of freedom, enabling notable adaptability in unstructured environments. Cosserat Rod Theory has emerged as a prominent framework for modeling these robots efficiently, representing continuum soft robots as time-varying curves, known as backbones. In this work, we propose viewing the robot's backbone as a signal in space and time, applying the Fourier transform to describe its deformation compactly. This approach unifies existing modeling strategies within the Cosserat Rod Theory framework, offering insights into commonly used heuristic methods. Moreover, the Fourier transform enables the development of a data-driven methodology to experimentally capture the robot's deformation. The proposed approach is validated through numerical simulations and experiments on a real-world prototype, demonstrating a reduction in the degrees of freedom while preserving the accuracy of the deformation representation.
>
---
#### [replaced 011] MIMIC-D: Multi-modal Imitation for MultI-agent Coordination with Decentralized Diffusion Policies
- **分类: cs.RO**

- **简介: 该论文属于多智能体协作任务，旨在解决多模态行为学习中的协调问题。提出MIMIC-D框架，通过去中心化策略实现有效协作。**

- **链接: [https://arxiv.org/pdf/2509.14159](https://arxiv.org/pdf/2509.14159)**

> **作者:** Dayi Dong; Maulik Bhatt; Seoyeon Choi; Negar Mehr
>
> **备注:** 8 pages, 4 figures, 5 tables
>
> **摘要:** As robots become more integrated in society, their ability to coordinate with other robots and humans on multi-modal tasks (those with multiple valid solutions) is crucial. Such behaviors can be learned from expert demonstrations via imitation learning (IL), but when expert demonstrations are multi-modal, standard IL approaches usually average across modes or collapse to a single mode, preventing effective coordination. Being inspired by diffusion models' ability to capture complex multi-modal trajectory distributions in single-agent settings, we develop a diffusion-based framework for coordinated multi-modal behavior in multi-agent systems. However, existing multi-agent diffusion approaches typically require a centralized planner or explicit communication among agents. This assumption can fail in real-world scenarios where robots must operate independently or with agents like humans that they cannot directly communicate with. Therefore, we propose MIMIC-D, a joint training with decentralized execution paradigm for multi-modal multi-agent IL via diffusion. We jointly train all agents' policies with only local information to achieve implicit coordination. In simulation and hardware experiments, our method exhibits robust multi-modal coordination behavior in various tasks and environments, improving upon state-of-the-art baselines.
>
---
#### [replaced 012] HECTOR: Human-centric Hierarchical Coordination and Supervision of Robotic Fleets under Continual Temporal Tasks
- **分类: cs.RO; cs.MA**

- **简介: 该论文提出HECTOR系统，解决大规模机器人编队在持续时间任务中的协同与监督问题，通过分层架构实现人机高效交互与动态任务分配。**

- **链接: [https://arxiv.org/pdf/2604.10892](https://arxiv.org/pdf/2604.10892)**

> **作者:** Shen Wang; Yinhang Luo; Jie Li; Meng Guo
>
> **摘要:** Robotic fleets can be extremely efficient when working concurrently and collaboratively, e.g., for delivery, surveillance, search and rescue. However, it can be demanding or even impractical for an operator to directly control each robot. Thus, autonomy of the fleet and its online interaction with the operator are both essential, particularly in dynamic and partially unknown environments. The operator might need to add new tasks, cancel some tasks, change priorities and modify planning results. How to design the procedure for these interactions and efficient algorithms to fulfill these needs have been mostly neglected in the related literature. Thus, this work proposes a human-centric coordination and supervision scheme (HECTOR) for large-scale robotic fleets under continual and uncertain temporal tasks. It consists of three hierarchical layers: (I) the bidirectional and multimodal protocol of online human-fleet interaction, where the operator interacts with and supervises the whole fleet; (II) the rolling assignment of currently-known tasks to teams within a certain horizon, and (III) the dynamic coordination within a team given the detected subtasks during online execution. The overall mission can be as general as temporal logic formulas over collaborative actions. Such hierarchical structure allows human interaction and supervision at different granularities and triggering conditions, to both improve computational efficiency and reduce human effort. Extensive human-in-the-loop simulations are performed over heterogeneous fleets under various temporal tasks and environmental uncertainties.
>
---
#### [replaced 013] Safe Bayesian Optimization for Complex Control Systems via Additive Gaussian Processes
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于控制优化任务，解决多控制器调参中的安全性和效率问题。提出SafeCtrlBO方法，利用加性高斯过程减少样本需求，并通过边界扩展确保安全性。**

- **链接: [https://arxiv.org/pdf/2408.16307](https://arxiv.org/pdf/2408.16307)**

> **作者:** Hongxuan Wang; Xiaocong Li; Lihao Zheng; Adrish Bhaumik; Prahlad Vadakkepat
>
> **备注:** The shorter version has been accepted by IEEE Robotics and Automation Letters. This is the full version
>
> **摘要:** Automatic controller tuning is attractive for robotics and mechatronic systems whose dynamics are difficult to model accurately, but direct black-box optimization can be unsafe because each query is executed on the physical plant. Existing safe Bayesian optimization (BO) methods provide high-probability safety guarantees, yet their practical use in multi-loop control is limited by two coupled difficulties: the controller parameter space is often moderately high-dimensional, and hardware evaluations are too expensive to allow hundreds or thousands of exploratory trials. This paper proposes \textsc{SafeCtrlBO}, a safe BO method for simultaneously tuning multiple coupled controllers. The method uses additive Gaussian-process kernels to encode low-order structure across controller gains and reduce the sample complexity associated with dense full-dimensional kernels. It also replaces the expensive potential-expander computation used in \textsc{SafeOpt}-style exploration with a boundary-based expansion rule that preserves the intended safe-set expansion behavior under explicit geometric conditions and is validated empirically. Experiments on synthetic benchmarks and on a permanent magnet synchronous motor (PMSM) speed-control platform show that \textsc{SafeCtrlBO} reaches high-performing controller parameters with fewer hardware evaluations than representative safe BO baselines, while maintaining the prescribed high-probability safety criterion and avoiding violations of the hard signal-safety constraint in the hardware study. The code implementation is publicly available at this https URL.
>
---
#### [replaced 014] DIVER: Reinforced Diffusion Breaks Imitation Bottlenecks in End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决模仿学习导致的行为单一问题。提出DIVER框架，结合强化学习与扩散生成，提升轨迹多样性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2507.04049](https://arxiv.org/pdf/2507.04049)**

> **作者:** Ziying Song; Lin Liu; Hongyu Pan; Bencheng Liao; Mingzhe Guo; Lei Yang; Yongchang Zhang; Shaoqing Xu; Caiyan Jia; Yadan Luo
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** Most end-to-end autonomous driving methods rely on imitation learning from single expert demonstrations, often leading to conservative and homogeneous behaviors that limit generalization in complex real-world scenarios. In this work, we propose DIVER, an end-to-end driving framework that integrates reinforcement learning with diffusion-based generation to produce diverse and feasible trajectories. At the core of DIVER lies a reinforced diffusion-based generation mechanism. First, the model conditions on map elements and surrounding agents to generate multiple reference trajectories from a single ground-truth trajectory, alleviating the limitations of imitation learning that arise from relying solely on single expert demonstrations. Second, reinforcement learning is employed to guide the diffusion process, where reward-based supervision enforces safety and diversity constraints on the generated trajectories, thereby enhancing their practicality and generalization capability. Furthermore, to address the limitations of L2-based open-loop metrics in capturing trajectory diversity, we propose a novel Diversity metric to evaluate the diversity of multi-mode this http URL experiments on the closed-loop NAVSIM and Bench2Drive benchmarks, as well as the open-loop nuScenes dataset, demonstrate that DIVER significantly improves trajectory diversity, effectively addressing the mode collapse problem inherent in imitation learning.
>
---
#### [replaced 015] MemCompiler: Compile, Don't Inject -- State-Conditioned Memory for Embodied Agents
- **分类: cs.RO**

- **简介: 该论文属于具身智能任务，解决记忆系统与 agent 状态不匹配的问题。提出 MemCompiler，通过状态条件编译记忆，提升效果与效率。**

- **链接: [https://arxiv.org/pdf/2605.07594](https://arxiv.org/pdf/2605.07594)**

> **作者:** Xin Ding; Xinrui Wang; Yifan Yang; Hao Wu; Shiqi Jiang; Qianxi Zhang; Liang Mi; Hanxin Zhu; Kun Li; Yunxin Liu; Zhibo Chen; Ting Cao
>
> **摘要:** Existing memory systems for embodied agents typically inject retrieved memory as static context at episode start, a paradigm we term Ahead-of-time Monolithic Memory Injection (AMMI). However, this static design quickly becomes misaligned with the agent's evolving state and may degrade lightweight executors below the no-memory baseline. To address this, we propose MemCompiler, which reframes memory utilization as State-Conditioned Memory Compilation. A learned Memory Compiler reads a structured Brief State capturing the agent's current execution state and dynamically selects and compiles only relevant memory into executable guidance. This guidance is delivered through a text channel and a latent Soft-Mem channel that preserves perceptual information not expressible in text. Across Alf World, EmbodiedBench, and ScienceWorld, MemCompiler consistently improves over no-memory across open-source backbones (up to +129%), matches or approaches frontier closed-source systems, and reduces per-step latency by 60%, demonstrating that state-aware memory compilation improves both effectiveness and efficiency.
>
---
#### [replaced 016] Bellman Value Decomposition for Task Logic in Safe Optimal Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究安全最优控制中的任务逻辑问题，通过Bellman值分解提升复杂任务的性能。针对高维环境下的目标与安全规范组合难题，提出VDPPO方法，实现安全与活跃性的自动平衡。**

- **链接: [https://arxiv.org/pdf/2602.19532](https://arxiv.org/pdf/2602.19532)**

> **作者:** William Sharpless; Oswin So; Dylan Hirsch; Sylvia Herbert; Chuchu Fan
>
> **摘要:** Real-world tasks involve nuanced combinations of goal and safety specifications. In high dimensions, the challenge is exacerbated: formal automata become cumbersome, and the combination of sparse rewards tends to require laborious tuning. In this work, we consider the innate structure of the Bellman Value as a means to naturally organize the problem for improved automatic performance. Namely, we prove the Bellman Value for a complex task defined in temporal logic can be decomposed into a graph of Bellman Values, connected by a set of well-known Bellman equations (BEs): the Reach-Avoid BE, the Avoid BE, and a novel type, the Reach-Avoid-Loop BE. To solve the Value and optimal policy, we propose VDPPO, which embeds the decomposed Value graph into a two-layer neural net, bootstrapping the implicit dependencies. We conduct a variety of simulated and hardware experiments to test our method on complex, high-dimensional tasks involving heterogeneous teams and nonlinear dynamics. Ultimately, we find this approach greatly improves performance over existing baselines, balancing safety and liveness automatically.
>
---
#### [replaced 017] Bluetooth Phased-array Aided Inertial Navigation Using Factor Graphs: Experimental Verification
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于导航任务，解决GNSS失效下的定位问题。使用蓝牙相控阵和因子图优化，评估不同估计策略的性能。**

- **链接: [https://arxiv.org/pdf/2602.17407](https://arxiv.org/pdf/2602.17407)**

> **作者:** Glen Hjelmerud Mørkbak Sørensen; Torleiv H. Bryne; Kristoffer Gryte; Tor Arne Johansen
>
> **备注:** 6 pages, 5 figures, 2 tables. \c{opyright} 2026 the authors. This work has been accepted to IFAC for publication under a Creative Commons Licence CC-BY-NC-ND
>
> **摘要:** Phased-array Bluetooth systems have emerged as a low-cost alternative for performing aided inertial navigation in GNSS-denied use cases such as warehouse logistics, drone landings, and autonomous docking. Basing a navigation system off of commercial-off-the-shelf components may reduce the barrier of entry for phased-array radio navigation systems, albeit at the cost of significantly noisier measurements and relatively short feasible range. In this paper, we compare robust estimation strategies for a factor graph optimisation-based estimator using experimental data collected from multirotor drone flight. We evaluate performance in loss-of-GNSS scenarios when aided by Bluetooth angular measurements, as well as range or barometric pressure.
>
---
#### [replaced 018] MindVLA-U1: VLA Beats VA with Unified Streaming Architecture for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MindVLA-U1，解决自动驾驶中视觉-语言-动作模型的统一问题，通过流式架构提升规划效果与效率。**

- **链接: [https://arxiv.org/pdf/2605.12624](https://arxiv.org/pdf/2605.12624)**

> **作者:** Yuzhou Huang; Benjin Zhu; Hengtong Lu; Victor Shea-Jay Huang; Haiming Zhang; Wei Chen; Jifeng Dai; Yan Xie; Hongsheng Li
>
> **备注:** Work in progress. Project page: this https URL
>
> **摘要:** Autonomous driving has progressed from modular pipelines toward end-to-end unification, and Vision-Language-Action (VLA) models are a natural extension of this journey beyond Vision-to-Action (VA). In practice, driving VLAs have often trailed VA on planning quality, suggesting that the difficulty is not simply model scale but the interface through which semantic reasoning, temporal context, and continuous control are combined. We argue that this gap reflects how VLA has been built -- as isolated subtask improvements that fail to compose coherent driving capabilities -- rather than what VLA is. We present MindVLA-U1, the first unified streaming VLA architecture for autonomous driving. A unified VLM backbone produces AR language tokens (optional) and flow-matching continuous action trajectories in a single forward pass over one shared representation, preserving the natural output form of each modality. A full streaming design processes the driving video framewise rather than as fixed video-action chunks under costly temporal VLM modeling. Planned trajectories evolve smoothly across frames while a learned streaming memory channel carries temporal context and updates. The unified architecture enables fast/slow systems on dense & sparse MoT backbones via flexible self-attention context management, and exposes a measurable language-control path for action: language-predicted driving intents steers the action diffusion via classifier-free guidance (CFG), turning language-side intent into control signals for continuous action planning. On the long-tail WOD-E2E benchmark, MindVLA-U1 surpasses experienced human drivers for the first time (8.20 RFS vs. 8.13 GT RFS) with 2 diffusion steps, achieves state-of-the-art planning ADEs over prior VA/VLA by large margins, and matches VA latency (16 FPS vs. RAP's 18 FPS at 1B scale) while preserving natural language interfaces for human-vehicle interaction.
>
---
#### [replaced 019] Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出Robometer，解决机器人奖励模型泛化性差的问题，通过轨迹比较实现更有效的奖励学习。**

- **链接: [https://arxiv.org/pdf/2603.02115](https://arxiv.org/pdf/2603.02115)**

> **作者:** Anthony Liang; Yigit Korkmaz; Jiahui Zhang; Minyoung Hwang; Abrar Anwar; Sidhant Kaushik; Aditya Shah; Alex S. Huang; Luke Zettlemoyer; Dieter Fox; Yu Xiang; Anqi Li; Andreea Bobu; Abhishek Gupta; Stephen Tu; Erdem Biyik; Jesse Zhang
>
> **备注:** 33 pages, 17 figures
>
> **摘要:** General-purpose robot reward models are typically trained to predict absolute task progress from expert demonstrations, providing only local, frame-level supervision. While effective for expert demonstrations, this paradigm scales poorly to large-scale robotics datasets where failed and suboptimal trajectories are abundant and assigning dense progress labels is ambiguous. We introduce Robometer, a scalable reward modeling framework that combines intra-trajectory progress supervision with inter-trajectory preference supervision. Robometer is trained with a dual objective: a frame-level progress loss that anchors reward magnitude on expert data, and a trajectory-comparison preference loss that imposes global ordering constraints across trajectories of the same task, enabling effective learning from both real and augmented failed trajectories. To support this formulation at scale, we curate RBM-1M, a reward-learning dataset comprising over one million trajectories spanning diverse robot embodiments and tasks, including substantial suboptimal and failure data. Across benchmarks and real-world evaluations, Robometer learns more generalizable reward functions than prior methods and improves robot learning performance across a diverse set of downstream applications. Code, model weights, and videos at this https URL.
>
---
#### [replaced 020] D-VLA: A High-Concurrency Distributed Asynchronous Reinforcement Learning Framework for Vision-Language-Action Models
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出D-VLA框架，解决大规模视觉-语言-动作模型在分布式强化学习中的高并发与低延迟问题，通过架构优化提升训练效率和稳定性。**

- **链接: [https://arxiv.org/pdf/2605.13276](https://arxiv.org/pdf/2605.13276)**

> **作者:** Yucheng Guo; Yongjian Guo; Zhong Guan; Wen Huang; Haoran Sun; Haodong Yue; Xiaolong Xiang; Shuai Di; Zhen Sun; Luqiao Wang; Junwu Xiong; Yicheng Gong
>
> **摘要:** The rapid evolution of Embodied AI has enabled Vision-Language-Action (VLA) models to excel in multimodal perception and task execution. However, applying Reinforcement Learning (RL) to these massive models in large-scale distributed environments faces severe systemic bottlenecks, primarily due to the resource conflict between high-fidelity physical simulation and the intensive VRAM/bandwidth demands of deep learning. This conflict often leaves overall throughput constrained by execution-phase inefficiencies. To address these challenges, we propose D-VLA, a high-concurrency, low-latency distributed RL framework for large-scale embodied foundation models. D-VLA introduces "Plane Decoupling," physically isolating high-frequency training data from low-frequency weight control to eliminate interference between simulation and optimization. We further design a four-thread asynchronous "Swimlane" pipeline, enabling full parallel overlap of sampling, inference, gradient computation, and parameter distribution. Additionally, a dual-pool VRAM management model and topology-aware replication resolve memory fragmentation and optimize communication efficiency. Experiments on benchmarks like LIBERO show that D-VLA significantly outperforms mainstream RL frameworks in throughput and sampling efficiency for billion-parameter VLA models. In trillion-parameter scalability tests, our framework maintains exceptional stability and linear speedup, providing a robust system for high-performance general-purpose embodied agents.
>
---
#### [replaced 021] Driving Intents Amplify Planning-Oriented Reinforcement Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决连续动作策略在单次示范下的模式崩溃问题。提出DIAL框架，通过意图扩增和多意图GRPO提升策略多样性与性能。**

- **链接: [https://arxiv.org/pdf/2605.12625](https://arxiv.org/pdf/2605.12625)**

> **作者:** Hengtong Lu; Victor Shea-Jay Huang; Chengmin Yang; Pengfei Jing; Jifeng Dai; Yan Xie; Benjin Zhu
>
> **备注:** Project page: this https URL
>
> **摘要:** Continuous-action policies trained on a single demonstrated trajectory per scene suffer from mode collapse: samples cluster around the demonstrated maneuver and the policy cannot represent semantically distinct alternatives. Under preference-based evaluation, this caps best-of-N performance -- even oracle selection cannot recover what the sampling distribution does not contain. We introduce DIAL, a two-stage Driving-Intent-Amplified reinforcement Learning framework for preference-aligned continuous-action driving policies. In the first stage, DIAL conditions the flow-matching action head on a discrete intent label with classifier-free guidance (CFG), which expands the sampling distribution along distinct maneuver modes and breaks single-demonstration mode collapse. In the second stage, DIAL carries this expanded distribution into preference RL through multi-intent GRPO, which spans all intent classes within every preference group and prevents fine-tuning from re-collapsing around the currently preferred mode. Instantiated for end-to-end driving with eight rule-derived intents and evaluated on WOD-E2E: competitive Vision-to-Action (VA) and Vision-Language-Action (VLA) Supervised Finetuning (SFT) baselines plateau below the human-driven demonstration at best-of-128, with the strongest prior (RAP) capping at Rater Feedback Score (RFS) 8.5 even with best-of-64; intent-CFG sampling lifts this ceiling to RFS 9.14 at best-of-128, surpassing both the prior best (RAP 8.5) and the human-driven demonstration (8.13) for the first time; and multi-intent GRPO improves held-out RFS from 7.681 to 8.211, while every single-intent baseline peaks lower and degrades by training end. These results suggest that the bottleneck of preference RL on continuous-action policies trained from demonstrations is not only how to update the policy, but to expand and preserve the sampling distribution being optimized.
>
---
#### [replaced 022] VER: Vision Expert Transformer for Robot Learning via Foundation Distillation and Dynamic Routing
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出VER，用于机器人学习的视觉专家Transformer。解决多视觉模型泛化不足问题，通过知识蒸馏和动态路由选择专家，提升任务适应性与效率。**

- **链接: [https://arxiv.org/pdf/2510.05213](https://arxiv.org/pdf/2510.05213)**

> **作者:** Yixiao Wang; Mingxiao Huo; Zhixuan Liang; Yushi Du; Lingfeng Sun; Haotian Lin; Jinghuan Shang; Chensheng Peng; Mohit Bansal; Mingyu Ding; Masayoshi Tomizuka
>
> **摘要:** Pretrained vision foundation models (VFMs) advance robotic learning via rich visual representations, yet individual VFMs typically excel only in specific domains, limiting generality across tasks. Distilling multiple VFMs into a unified representation for policy can mitigate this limitation but often yields inflexible task-specific feature selection and requires costly full re-training to incorporate robot-domain knowledge. We propose VER, a Vision Expert transformer for Robot learning. During pretraining, VER distills multiple VFMs into a vision expert library. It then fine-tunes only a lightweight routing network (fewer than 0.4% of parameters) to dynamically select task-relevant experts from the pretrained library for downstream robot tasks. We further introduce Patchwise Expert Routing with Curriculum Top-K Annealing to improve both flexibility and precision of dynamic expert selection. Moreover, VER supports parameter-efficient finetuning for scalable expert utilization and adaptive robot-domain knowledge integration. Across 17 diverse robotic tasks and multiple policy heads, VER achieves state-of-the-art performance. We find that VER reduces large-norm outliers in task-irrelevant regions (e.g., background) and concentrates on task-critical regions. Visualizations and codes can be found in this https URL.
>
---
#### [replaced 023] Geometry-Aware Sampling-Based Motion Planning on Riemannian Manifolds
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决在非欧几里得配置空间中生成最优路径的问题。提出一种基于流形的采样规划方法，提升轨迹质量与计算效率。**

- **链接: [https://arxiv.org/pdf/2602.00992](https://arxiv.org/pdf/2602.00992)**

> **作者:** Phone Thiha Kyaw; Jonathan Kelly
>
> **备注:** Accepted to the 17th World Symposium on the Algorithmic Foundations of Robotics (WAFR), Oulu, Finland, Jun 15-17, 2026
>
> **摘要:** In many robot motion planning problems, task objectives and physical constraints induce non-Euclidean geometry on the configuration space, yet many planners operate using Euclidean distances that ignore this structure. We address the problem of planning collision-free motions that minimize length under configuration-dependent Riemannian metrics, corresponding to geodesics on the configuration manifold. Conventional numerical methods for computing such paths do not scale well to high-dimensional systems, while sampling-based planners trade scalability for geometric fidelity. To bridge this gap, we propose a sampling-based motion planning framework that operates directly on Riemannian manifolds. We introduce a computationally efficient midpoint-based approximation of the Riemannian geodesic distance and prove that it matches the true Riemannian distance with third-order accuracy. Building on this approximation, we design a local planner that traces the manifold using first-order retractions guided by Riemannian natural gradients. Experiments on a two-link planar arm and a 7-DoF Franka manipulator under a kinetic-energy metric, as well as on rigid-body planning in $\mathrm{SE}(2)$ with non-holonomic motion constraints, demonstrate that our approach consistently produces lower-cost trajectories than Euclidean-based planners and classical numerical geodesic-solver baselines.
>
---
#### [replaced 024] Sharing the Load: Autonomous Multi-Rover Cargo Transport
- **分类: cs.RO**

- **简介: 该论文属于多探测器协同运输任务，旨在解决月球货运路径重复与负载共享问题。通过分布式模型预测控制算法实现两辆 rover 共同运输货物，提升任务灵活性与效率。**

- **链接: [https://arxiv.org/pdf/2510.18766](https://arxiv.org/pdf/2510.18766)**

> **作者:** Alexander Krawciw; Luka Antonyshyn; Sven Lilge; Nicolas Olmedo; Faizan Rehmatullah; Maxime Desjardins-Goulet; Pascal Toupin; Timothy D. Barfoot
>
> **备注:** 19 pages, 14 figures, submitted to IEEE Transactions on Field Robotics
>
> **摘要:** A future lunar habitat, as part of the Artemis program, will require a significant amount of logistics infrastructure. Cargo that is transported to the Moon will need to be moved from a landing site to other key locations that may be up to 5 km away. Teach and repeat navigation is well suited to this task as utility rovers will need to repeat these cargo routes many times. One of the most significant challenges involves the modules that will be assembled together to form the habitat. Canada is studying potential Lunar Utility Vehicle (LUV) designs to carry these large payloads between the landing site and the location of the habitat. As the details of the cargo continue to evolve, using two, smaller LUVs to carry cargo together would provide high capacity and mission flexibility. In this paper, we develop and implement a distributed model-predictive controller that allows vehicles to carry cargo that is shared between them. The algorithm is compared to baselines in small-scale before being implemented onboard two 800 kg path-to-flight rovers and field tested carrying a 475 kg cargo between them. A custom cargo coupling decouples the kinematics of each vehicle while fully supporting the cargo's mass. In our field test, the rovers maintain a relative separation error of 9.2 cm and maximum error of 33.4 cm. This multi-vehicle control architecture retains the high-quality path tracking of lidar teach and repeat for each rover. We demonstrate that kinematic freedom of the vehicles allows a single controller to provide mission improvements for other operations as well.
>
---
#### [replaced 025] XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决动作生成与跨领域数据对齐问题。提出XR-1框架，通过统一视觉-运动编码实现多机器人、多任务的泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.02776](https://arxiv.org/pdf/2511.02776)**

> **作者:** Shichao Fan; Kun Wu; Zhengping Che; Xinhua Wang; Di Wu; Fei Liao; Ning Liu; Yixue Zhang; Zhen Zhao; Zhiyuan Xu; Meng Li; Qingjie Liu; Shanghang Zhang; Min Wan; Jian Tang
>
> **备注:** Accepted to ICML2026 as spotlight
>
> **摘要:** Recent progress in large-scale robotic datasets and vision-language models (VLMs) has advanced research on vision-language-action (VLA) models. However, existing VLA models still face two fundamental challenges: (i) producing precise low-level actions from high-dimensional observations, (ii) bridging domain gaps across heterogeneous data sources, including diverse robot embodiments and human demonstrations. Existing methods often encode latent variables from either visual dynamics or robotic actions to guide policy learning, but they fail to fully exploit the complementary multi-modal knowledge present in large-scale, heterogeneous datasets. In this work, we present X Robotic Model 1 (XR-1), a novel framework for versatile and scalable VLA learning across diverse robots, tasks, and environments. XR-1 introduces the \emph{Unified Vision-Motion Codes (UVMC)}, a discrete latent representation learned via a dual-branch VQ-VAE that jointly encodes visual dynamics and robotic motion. UVMC addresses these challenges by (i) serving as an intermediate representation between the observations and actions, and (ii) aligning multimodal dynamic information from heterogeneous data sources to capture complementary knowledge. To effectively exploit UVMC, we propose a three-stage training paradigm: (i) self-supervised UVMC learning, (ii) UVMC-guided pretraining on large-scale cross-embodiment robotic datasets, and (iii) task-specific post-training. We validate XR-1 through extensive real-world experiments with more than 14,000 rollouts on six different robot embodiments, spanning over 120 diverse manipulation tasks. XR-1 consistently outperforms state-of-the-art baselines such as $\pi_{0.5}$, $\pi_0$, RDT, UniVLA, and GR00T-N1.5 while demonstrating strong generalization to novel objects, background variations, distractors, and illumination changes. Our project is at this https URL.
>
---
#### [replaced 026] Adapting Dijkstra for Buffers and Unlimited Transfers
- **分类: cs.DS; cs.AI; cs.RO**

- **简介: 该论文属于公共交通路径规划任务，解决无限换乘下的路径优化问题。通过改进Dijkstra算法，提出TAD方法，有效处理缓冲时间，提升计算效率。**

- **链接: [https://arxiv.org/pdf/2603.11729](https://arxiv.org/pdf/2603.11729)**

> **作者:** Denys Katkalo; Andrii Rohovyi; Toby Walsh
>
> **备注:** v3: revised manuscript incorporating reviewer feedback (formal correctness proof, deployment trade-off discussion, route/tau_min definitions, dominance-inequality fix); editorial and layout polish
>
> **摘要:** In recent years, RAPTOR based algorithms have been considered the state-of-the-art for path-finding with unlimited transfers without preprocessing. However, this status largely stems from the evolution of routing research, where Dijkstra-based solutions were superseded by timetable-based algorithms without a systematic comparison. In this work, we revisit classical Dijkstra-based approaches for public transit routing with unlimited transfers and demonstrate that Time-Dependent Dijkstra (TD-Dijkstra) outperforms MR. However, efficient TD-Dijkstra implementations rely on filtering dominated connections during preprocessing, which assumes passengers can always switch to a faster connection. We show that this filtering is unsound when stops have buffer times, as it cannot distinguish between seated passengers who may continue without waiting and transferring passengers who must respect the buffer. To address this limitation, we introduce Transfer Aware Dijkstra (TAD), a modification that scans entire trip sequences rather than individual edges, correctly handling buffer times while maintaining performance advantages over MR. Our experiments on London and Switzerland networks show that we can achieve a greater than two time speed-up over MR while producing optimal results on both networks with and without buffer times.
>
---
#### [replaced 027] Optimal UGV-UAV Cooperative Partitioning and Inspection of Shortest Paths
- **分类: cs.RO**

- **简介: 该论文研究UGV与UAV协同路径规划问题，解决未知障碍环境下最优路径分割与检查。通过分析不同场景下的竞争比，提出优化策略以减少UGV行驶时间。**

- **链接: [https://arxiv.org/pdf/2604.25284](https://arxiv.org/pdf/2604.25284)**

> **作者:** Ninh Nguyen; Srinivas Akella
>
> **备注:** Withdrawn by the authors due to an error in Section V.D in the competitive-ratio proof for the UGV-UAV case. The proof incorrectly uses $1+2\frac{v_A}{v_G+v_A}(k-1)\le 2\frac{v_A}{v_G+v_A}k-1$, which does not hold in general and affects the stated bound
>
> **摘要:** We study cooperative shortest path planning for an unmanned ground vehicle (UGV) assisted by an unmanned aerial vehicle (UAV) in environments with unknown road blockages that are only discovered when a robot reaches the damaged point. This formulation generalizes the original Canadian Traveller Problem (CTP), which assumes a single ground vehicle and that the traversability status of all incident edges is revealed upon arrival at a vertex. We first analyze the case where the start and the goal are connected by $k$ disjoint paths, and prove that the worst-case competitive ratio $\rho$ for a single UGV is $2k-1$. With UAV assistance, and under the simplifying assumption of negligible initial transit and deadheading UAV costs, the ratio improves to $\rho = 2\frac{v_G}{v_A + v_G}k - 1$, where $v_G$ and $v_A$ denote the UGV and UAV speed, respectively. To address general graphs and non-negligible UAV initial transit and deadheading costs, we present an optimal path partitioning strategy that assigns path prefix inspection to the UGV and path suffix inspection to the UAV, and prove the optimality of the UAV inspection strategy on general graphs. We evaluate our algorithm by performing experiments on road networks from the world's 50 most populous cities, with randomized blockages, and show that the proposed method reduces UGV travel times by up to 30%.
>
---
#### [replaced 028] RoboWM-Bench: A Benchmark for Evaluating World Models in Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RoboWM-Bench，用于评估机器人操作中视频世界模型的物理可执行性。解决现有基准未系统评估行为可执行性的问题，通过模拟验证生成动作的有效性。**

- **链接: [https://arxiv.org/pdf/2604.19092](https://arxiv.org/pdf/2604.19092)**

> **作者:** Feng Jiang; Yang Chen; Kyle Xu; Yuchen Liu; Haifeng Wang; Zhenhao Shen; Jasper Lu; Shengze Huang; Yuanfei Wang; Chen Xie; Ruihai Wu
>
> **摘要:** Recent advances in large-scale video world models have enabled increasingly realistic future prediction, raising the prospect of using generated videos as scalable supervision for robot learning. However, for embodied manipulation, perceptual realism alone is not sufficient: generated interactions must also be physically consistent and executable by robotic agents. Existing benchmarks provide valuable assessments of visual quality and physical plausibility, but they do not systematically evaluate whether predicted behaviors can be translated into executable actions that complete manipulation tasks. We introduce RoboWM-Bench, a manipulation-centric benchmark for embodiment-grounded evaluation of video world models. RoboWM-Bench converts generated human-hand and robotic manipulation videos into embodied action sequences and validates them through execution in physically grounded simulation environments. Built on real-to-sim scene reconstruction and diverse manipulation tasks, RoboWM-Bench enables standardized, reproducible, and scalable evaluation of physical executability. Using RoboWM-Bench, we evaluate state-of-the-art video world models and observe that visual plausibility and embodied executability are not always aligned. Our analysis highlights several recurring factors that affect execution performance, including spatial reasoning, contact prediction, and non-physical geometric distortions, particularly in complex and long-horizon interactions. These findings provide a more fine-grained view of current model capabilities and underscore the value of embodiment-aware evaluation for guiding physically grounded world modeling in robotic manipulation.
>
---
#### [replaced 029] From Local Matches to Global Masks: Template-Guided Instance Detection and Segmentation in Open-World Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于开放场景下的实例检测与分割任务，解决在仅有模板图像情况下准确定位和分割新对象的问题。提出L2G-Det框架，通过局部匹配生成候选点并优化，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2603.03577](https://arxiv.org/pdf/2603.03577)**

> **作者:** Qifan Zhang; Sai Haneesh Allu; Jikai Wang; Yangxiao Lu; Yu Xiang
>
> **备注:** Accepted to Robotics: Science and Systems (RSS) 2026. Project page: this https URL
>
> **摘要:** Detecting and segmenting novel object instances in open-world environments is a fundamental problem in robotic perception. Given only a small set of template images, a robot must locate and segment a specific object instance in a cluttered, previously unseen scene. Existing proposal-based approaches are highly sensitive to proposal quality and often fail under occlusion and background clutter. We propose L2G-Det, a local-to-global instance detection framework that bypasses explicit object proposals by leveraging dense patch-level matching between templates and the query image. Locally matched patches generate candidate points, which are refined through a candidate selection module to suppress false positives. The filtered points are then used to prompt an augmented Segment Anything Model (SAM) with instance-specific object tokens, enabling reliable reconstruction of complete instance masks. Experiments demonstrate improved performance over proposal-based methods in challenging open-world settings.
>
---
#### [replaced 030] Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）任务，旨在解决2D图像限制空间理解的问题。通过引入3D点云增强模型性能，提出Any3D-VLA以克服数据稀缺和领域差异问题。**

- **链接: [https://arxiv.org/pdf/2602.00807](https://arxiv.org/pdf/2602.00807)**

> **作者:** Xianzhe Fan; Shengliang Deng; Xiaoyang Wu; Yuxiang Lu; Zhuoling Li; Mi Yan; Yujia Zhang; Zhizheng Zhang; He Wang; Hengshuang Zhao
>
> **备注:** ICML 2026
>
> **摘要:** Existing Vision-Language-Action (VLA) models typically take 2D images as visual input, which limits their spatial understanding in complex scenes. How can we incorporate 3D information to enhance VLA capabilities? We conduct a pilot study across different observation spaces and visual representations. The results show that explicitly lifting visual input into point clouds yields representations that better complement their corresponding 2D representations. To address the challenges of (1) scarce 3D data and (2) the domain gap induced by cross-environment differences and depth-scale biases, we propose Any3D-VLA. It unifies the simulator, sensor, and model-estimated point clouds within a training pipeline, constructs diverse inputs, and learns domain-agnostic 3D representations that are fused with the corresponding 2D representations. Simulation and real-world experiments demonstrate Any3D-VLA's advantages in improving performance and mitigating the domain gap. Our project homepage is available at this https URL.
>
---
#### [replaced 031] MALLVI: A Multi-Agent Framework for Integrated Generalized Robotics Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出MALLVI框架，解决机器人操作中的任务规划问题。通过多智能体协作实现闭环反馈，提升动态环境下的操作成功率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.16898](https://arxiv.org/pdf/2602.16898)**

> **作者:** Mehrshad Taji; Arad Mahdinezhad Kashani; Iman Ahmadi; AmirHossein Jadidi; Saina Kashani; Babak Khalaj
>
> **备注:** Some fundemental change in text and codebase. Will request a new submission later on
>
> **摘要:** Task planning for robotic manipulation with large language models (LLMs) is an emerging area. Prior approaches rely on specialized models, fine tuning, or prompt tuning, and often operate in an open loop manner without robust environmental feedback, making them fragile in dynamic settings. MALLVI presents a Multi Agent Large Language and Vision framework that enables closed-loop feedback driven robotic manipulation. Given a natural language instruction and an image of the environment, MALLVI generates executable atomic actions for a robot manipulator. After action execution, a Vision Language Model (VLM) evaluates environmental feedback and decides whether to repeat the process or proceed to the next step. Rather than using a single model, MALLVI coordinates specialized agents, Decomposer, Localizer, Thinker, and Reflector, to manage perception, localization, reasoning, and high level planning. An optional Descriptor agent provides visual memory of the initial state. The Reflector supports targeted error detection and recovery by reactivating only relevant agents, avoiding full replanning. Experiments in simulation and real-world settings show that iterative closed loop multi agent coordination improves generalization and increases success rates in zero shot manipulation tasks. Code available at this https URL .
>
---
