# 机器人 cs.RO

- **最新发布 28 篇**

- **更新 26 篇**

## 最新发布

#### [new 001] Alter-Art: Exploring Embodied Artistic Creation through a Robot Avatar
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，探讨机器人作为艺术创作载体的新型模式，解决艺术家与机器间互动不足的问题，通过机器人化身实现具身艺术创作。**

- **链接: [https://arxiv.org/pdf/2604.26473](https://arxiv.org/pdf/2604.26473)**

> **作者:** Do Won Park; Samuele Bordini; Giorgio Grioli; Manuel G. Catalano; Antonio Bicchi
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** As with every emerging technology, new tools in the hands of artists reshape the nature of artwork creation. Current frameworks for robotics in arts deploy the robot as an autonomous creator or a collaborator, thus leaving a certain gap between the human artist and the machine. Now, we stand at the dawn of an era where artists can escape physical limitations and reshape their creative identity by inhabiting an alternative body. This new paradigm allows artists not only to command a robot remotely, but also to {\it be} a robot, to see and feel through it, experiencing a new embodied reality. Unlike virtual reality, where art is created in a digital dimension, in this case art creation is still firmly grounded in the material world: clay molded by mechanical hands, paint swept across a canvas or gestures performed on a physical stage alongside human actors. Through the robot avatar Alter-Ego, we explore the Alter-Art paradigm in dance, theater, and painting; it integrates immersive teleoperation and compliant actuation to enable a first-person creative experience. Analyzing qualitative artistic feedback, we investigate how embodiment shapes creative agency, identity and interaction with the environment. Our findings suggest that artists rapidly develop a sense of presence within the robotic body. The robot's physical constraints influence the creative process, manifesting differently across artistic domains. We highlight embodiment as a central design principle, contributing to social robotics and expanding the possibilities for telepresence and accessible artistic expression.
>
---
#### [new 002] HiPAN: Hierarchical Posture-Adaptive Navigation for Quadruped Robots in Unstructured 3D Environments
- **分类: cs.RO**

- **简介: 该论文属于四足机器人导航任务，解决在非结构化3D环境中路径规划与姿态适应问题。提出HiPAN框架，结合分层策略与课程学习，提升导航成功率与效率。**

- **链接: [https://arxiv.org/pdf/2604.26504](https://arxiv.org/pdf/2604.26504)**

> **作者:** Jeil Jeong; Minsung Yoon; Seokryun Choi; Heechan Shin; Taegeun Yang; Sung-eui Yoon
>
> **备注:** Accepted to RA-L 2026 | Project page: this https URL
>
> **摘要:** Navigating quadruped robots in unstructured 3D environments poses significant challenges, requiring goal-directed motion, effective exploration to escape from local minima, and posture adaptation to traverse narrow, height-constrained spaces. Conventional approaches employ a sequential mapping-planning pipeline but suffer from accumulated perception errors and high computational overhead, restricting their applicability on resource-constrained platforms. To address these challenges, we propose Hierarchical Posture-Adaptive Navigation (HiPAN), a framework that operates directly on onboard depth images at deployment. HiPAN adopts a hierarchical design: a high-level policy generates strategic navigation commands (planar velocity and body posture), which are executed by a low-level, posture-adaptive locomotion controller. To mitigate myopic behaviors and facilitate long-horizon navigation, we introduce Path-Guided Curriculum Learning, which progressively extends the navigation horizon from reactive obstacle avoidance to strategic navigation. In simulation, HiPAN achieves higher navigation success rates and greater path efficiency than classical reactive planners and end-to-end baselines, while real-world experiments further validate its applicability across diverse, unstructured 3D environments.
>
---
#### [new 003] 3D Generation for Embodied AI and Robotic Simulation: A Survey
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D生成任务，旨在解决 embodied AI 和机器人仿真中的内容生成问题。工作包括分类3D生成在数据生成、仿真环境构建和 sim2real 桥接中的角色，并分析当前瓶颈。**

- **链接: [https://arxiv.org/pdf/2604.26509](https://arxiv.org/pdf/2604.26509)**

> **作者:** Tianwei Ye; Yifan Mao; Minwen Liao; Jian Liu; Chunchao Guo; Dazhao Du; Quanxin Shou; Fangqi Zhu; Song Guo
>
> **备注:** 26 pages, 11 figures, 8 tables. Project Page: this https URL
>
> **摘要:** Embodied AI and robotic systems increasingly depend on scalable, diverse, and physically grounded 3D content for simulation-based training and real-world deployment. While 3D generative modeling has advanced rapidly, embodied applications impose requirements far beyond visual realism: generated objects must carry kinematic structure and material properties, scenes must support interaction and task execution, and the resulting content must bridge the gap between simulation and reality. This survey presents the first survey of 3D generation for embodied AI and organizes the literature around three roles that 3D generation plays in embodied systems. In \emph{Data Generator}, 3D generation produces simulation-ready objects and assets, including articulated, physically grounded, and deformable content for downstream interaction; in \emph{Simulation Environments}, it constructs interactive and task-oriented worlds, spanning structure-aware, controllable, and agentic scene generation; and in \emph{Sim2Real Bridge}, it supports digital twin reconstruction, data augmentation, and synthetic demonstrations for downstream robot learning and real-world transfer. We also show that the field is shifting from visual realism toward interaction readiness, and we identify the main bottlenecks, including limited physical annotations, the gap between geometric quality and physical validity, fragmented evaluation, and the persistent sim-to-real divide, that must be addressed for 3D generation to become a dependable foundation for embodied intelligence. Our project page is at this https URL.
>
---
#### [new 004] Split over $n$ resource sharing problem: Are fewer capable agents better than many simpler ones?
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究多智能体系统中资源分配问题，探讨将资源集中于少数强智能体还是分散给多个简单智能体更优。通过模型分析与仿真，揭示不同资源分配策略对系统性能的影响。**

- **链接: [https://arxiv.org/pdf/2604.26374](https://arxiv.org/pdf/2604.26374)**

> **作者:** Karthik Soma; Mohamed S. Talamali; Genki Miyauchi; Giovanni Beltrame; Heiko Hamann; Roderich Gross
>
> **备注:** Short paper presented at the 15th International Conference on Swarm Intelligence (ANTS 2026)
>
> **摘要:** In multi-agent systems, should limited resources be concentrated into a few capable agents or distributed among many simpler ones? This work formulates the split over $n$ resource sharing problem where a group of $n$ agents equally shares a common resource (e.g., monetary budget, computational resources, physical size). We present a case study in multi-agent coverage where the area of the disk-shaped footprint of agents scales as $1/n$. A formal analysis reveals that the initial coverage rate grows with $n$. However, if the speed of agents decreases proportionally with their radii, groups of all sizes perform equally well, whereas if it decreases proportionally with their footprints, a single agent performs best. We also present computer simulations in which resource splitting increases the failure rates of individual agents. The models and findings help identify optimal distributiveness levels and inform the design of multi-agent systems under resource constraints.
>
---
#### [new 005] Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出X-WAM，解决机器人行动与4D世界建模的统一问题，通过视频先验和异步去噪提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2604.26694](https://arxiv.org/pdf/2604.26694)**

> **作者:** Jun Guo; Qiwei Li; Peiyan Li; Zilong Chen; Nan Sun; Yifei Su; Heyun Wang; Yuan Zhang; Xinghang Li; Huaping Liu
>
> **备注:** Project website: this https URL
>
> **摘要:** We propose X-WAM, a Unified 4D World Model that unifies real-time robotic action execution and high-fidelity 4D world synthesis (video + 3D reconstruction) in a single framework, addressing the critical limitations of prior unified world models (e.g., UWM) that only model 2D pixel-space and fail to balance action efficiency and world modeling quality. To leverage the strong visual priors of pretrained video diffusion models, X-WAM imagines the future world by predicting multi-view RGB-D videos, and obtains spatial information efficiently through a lightweight structural adaptation: replicating the final few blocks of the pretrained Diffusion Transformer into a dedicated depth prediction branch for the reconstruction of future spatial information. Moreover, we propose Asynchronous Noise Sampling (ANS) to jointly optimize generation quality and action decoding efficiency. ANS applies a specialized asynchronous denoising schedule during inference, which rapidly decodes actions with fewer steps to enable efficient real-time execution, while dedicating the full sequence of steps to generate high-fidelity video. Rather than entirely decoupling the timesteps during training, ANS samples from their joint distribution to align with the inference distribution. Pretrained on over 5,800 hours of robotic data, X-WAM achieves 79.2% and 90.7% average success rate on RoboCasa and RoboTwin 2.0 benchmarks, while producing high-fidelity 4D reconstruction and generation surpassing existing methods in both visual and geometric metrics.
>
---
#### [new 006] STARRY: Spatial-Temporal Action-Centric World Modeling for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出STARRY，用于机器人操作的时空动作中心世界建模方法，解决传统方法在动作相关时空结构建模不足的问题，通过联合去噪和注意力机制提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2604.26848](https://arxiv.org/pdf/2604.26848)**

> **作者:** Yuxuan Tian; Yurun Jin; Bin Yu; Yukun Shi; Hao Wu; Chi Harold Liu; Kai Chen; Cong Huang
>
> **备注:** 19 pages
>
> **摘要:** Robotic manipulation critically requires reasoning about future spatial-temporal interactions, yet existing VLA policies and world-model-enhanced policies do not fully model action-relevant spatial-temporal interaction structure. We propose STARRY, a world-model-enhanced action-generation policy that aligns spatial-temporal prediction with action generation. STARRY jointly denoises future spatial-temporal latents and action sequences, and introduces Geometry-Aware Selective Attention Modulation to convert predicted depth and end-effector geometry into token-aligned weights for selective action-attention modulation. On RoboTwin 2.0, STARRY achieves 93.82% / 93.30% average success under Clean and Randomized settings. Real-world experiments further improve average success from 42.5% to 70.8% over $\pi_{0.5}$, demonstrating the effectiveness of action-centric spatial-temporal world modeling for spatial-temporally demanding robotic action generation.
>
---
#### [new 007] Atomic-Probe Governance for Skill Updates in Compositional Robot Policies
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究组合机器人策略中的技能更新治理问题，提出原子质量探测器和混合选择器，以提升技能替换后的组合性能。**

- **链接: [https://arxiv.org/pdf/2604.26689](https://arxiv.org/pdf/2604.26689)**

> **作者:** Xue Qin; Simin Luan; John See; Cong Yang; Zhijun Li
>
> **备注:** 8 pages main text + appendix; 3 figures, 12 tables;
>
> **摘要:** Skill libraries in deployed robotic systems are continually updated through fine-tuning, fresh demonstrations, or domain adaptation, yet existing typed-composition methods (BLADE, SymSkill, Generative Skill Chaining) treat the library as frozen at test time and do not analyze how composition outcomes change when a skill is replaced. We introduce a paired-sampling cross-version swap protocol on robosuite manipulation tasks to characterize this dimension of compositional skill learning. On a dual-arm peg-in-hole task we discover a dominant-skill effect: one ECM achieves 86.7% atomic success rate while every other ECM is at or below 26.7%, and whether this dominant ECM enters a composition shifts the success rate by up to +50pp. We characterize the boundary on a simpler pick task where all atomic policies saturate at 100% and the effect is undefined. Across three tasks we further find that off-policy behavioral distance metrics fail to identify the dominant ECM, ruling out the natural cheap predictor. We propose an atomic-quality probe and a Hybrid Selector combining per-skill probes (zero per-decision cost) with selective composition revalidation (full cost), and characterize its Pareto frontier on 144 skill-update decisions. On T6 the atomic-only probe sits 23pp below full revalidation (64.6% vs 87.5% oracle match) at zero per-decision cost; a Hybrid Selector with m=10 closes most of that gap to ~12pp at 46% of full-revalidation cost. On the cross-task average over 144 events, atomic-only is within 3pp of full revalidation under a mixed-oracle caveat. The atomic-quality probe is, to our knowledge, the first principled, deployment-ready primitive for skill-update governance in compositional robot policies.
>
---
#### [new 008] Lights Out: A Nighttime UAV Localization Framework Using Thermal Imagery and Semantic 3D Maps
- **分类: cs.RO**

- **简介: 该论文属于无人机定位任务，解决夜间GNSS失效时的定位问题。通过热成像与语义3D地图的语义重投影实现精准定位。**

- **链接: [https://arxiv.org/pdf/2604.26201](https://arxiv.org/pdf/2604.26201)**

> **作者:** Ryan Allen; Melissa Greeff
>
> **备注:** 8 pages, 4 figures, accepted to ICUAS 2025
>
> **摘要:** Reliable backup localization for unmanned aerial vehicles (UAVs) operating in GNSS-denied nighttime conditions remains an open challenge due to the severe modality gap between daytime RGB maps and nighttime thermal imagery. This work presents a semantic reprojection framework for map-relative nighttime UAV localization by aligning segmented thermal observations with a globally referenced, semantically labeled 3D map constructed from daytime RGB data. Rather than relying on appearance-based correspondence, localization is formulated in a shared semantic domain and solved via a symmetric bidirectional reprojection objective with confusion-aware weighting to improve robustness under segmentation uncertainty. The approach is evaluated offline across 6.5 km of nighttime, real-world UAV flight trajectories in urban and semi-structured environments. Relative to RTK GNSS ground truth, the system achieves a bias-corrected RMSE2D of 2.18 m and a median RMSE2D of 1.52 m. Results show that localization performance is strongly correlated with the availability of semantic edge evidence and that large-error events are spatially localized to semantically ambiguous areas rather than uniformly distributed. These findings indicate that semantic reprojection offers a promising pathway toward globally referenced nighttime UAV localization using thermal imagery alone.
>
---
#### [new 009] Bi-Level Optimization for Contact and Motion Planning in Rope-Assisted Legged Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决绳索辅助足式机器人在垂直表面移动的路径与控制问题，通过双层优化框架协同选择着陆区域并优化绳索张力和腿部力。**

- **链接: [https://arxiv.org/pdf/2604.26910](https://arxiv.org/pdf/2604.26910)**

> **作者:** Ruben Malacarne; Ioannis Tsikelis; Enrico Mingo Hoffman; Michele Focchi
>
> **摘要:** This paper presents a planning pipeline framework for locomotion in rope-assisted robots climbing vertical surfaces. The proposed framework is formulated as a bi-level optimization scheme that addresses a mixed-integer problem: selecting feasible terrain regions for landing while simultaneously optimizing the control inputs, namely rope tensions and leg forces, and landing location. The outer level of the optimization is solved using the Cross-Entropy Method, while the inner level relies on gradient-based nonlinear optimization to compute dynamically feasible motions. The approach is validated on a novel climbing robot platform, ALPINE, across a variety of challenging terrain configurations.
>
---
#### [new 010] Stochastic Entanglement of Deterministic Origami Tentacles For Universal Robotic Gripping
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人抓取任务，旨在解决复杂形状物体的稳健抓取问题。通过设计具有确定性变形和随机缠绕的折纸机械臂，实现简单驱动下的通用抓取。**

- **链接: [https://arxiv.org/pdf/2604.26897](https://arxiv.org/pdf/2604.26897)**

> **作者:** Alec Boron; Bokun Zheng; Ziyang Zhou; Noel Naughton; Suyi Li
>
> **摘要:** Origami-inspired robotic grippers have shown promising potential for object manipulation tasks due to their compact volume and mechanical flexibility. However, robust capture of objects with random shapes in dynamic working environments often comes at the cost of additional actuation channels and control complexity. Here, we introduce a tendon-driven origami tentacle gripper capable of universal object gripping by exploiting a synergy between local, deterministic deformation programming and global, stochastic entanglements. Each origami tentacle is made by cutting thin Mylar sheets; It features carefully placed holes for routing an actuation tendon, origami creases for controlling the deformation, and a tapered shape. By tailoring these design features, one can prescribe the shrinking, bending, and twisting deformation, eventually creating deterministic coiling with a simple tendon pull. Then, when multiple coiling tentacles are placed in proximity, stochastic entanglement emerges, allowing the tentacles to braid, knot, and grip objects with random shapes. We derived a simulation model by integrating origami mechanics with Cosserat rods to correlate origami design, tendon deformation, and their collective gripping performance. Then, we experimentally tested how these coiling and entangling origami tentacles can grasp objects under gravity and in water. A stow-and-release deployment mechanism was also tested to simulate in-orbit grasping. Overall, the entertaining origami tentacle gripper presents a new strategy for robust object grasping with simple design and actuation.
>
---
#### [new 011] FlowS: One-Step Motion Prediction via Local Transport Conditioning
- **分类: cs.RO**

- **简介: 该论文属于运动预测任务，解决生成模型延迟高的问题。提出FlowS框架，通过局部运输条件实现单步预测，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.26065](https://arxiv.org/pdf/2604.26065)**

> **作者:** Leandro Di Bella; Adrian Munteanu; Bruno Cornelis
>
> **备注:** 8 pages
>
> **摘要:** Generative motion prediction must satisfy three simultaneous requirements for real-world autonomy: high accuracy, diverse multimodal futures, and strictly bounded latency. Diffusion models meet the first two but violate the third, requiring tens to hundreds of denoising steps. We identify a conditioning strategy that resolves this tension: \textit{single-step integration is accurate when the underlying transport problem is local}. A model that must both discover the correct behavioral mode and traverse a long displacement in one step accumulates large discretization errors; conditioning the base distribution to lie near plausible futures reduces the problem to short-range refinement, the regime where a single Euler step suffices. We instantiate this \emph{local transport conditioning} in FlowS, a conditional flow matching framework with two mechanisms. First, an online, scene-conditioned learned prior emits $K$ calibrated anchor trajectories per agent, each already near a plausible future, converting mode discovery into local correction. Second, a step-consistent displacement field enforces semigroup self-consistency, guaranteeing that a single step inherits multi-step accuracy. Crucially, anchoring this field at learned priors along straight-line paths yields a {stable, low-variance} training target, unlike prior self-consistency methods that suffer from {high-variance bootstrap} signals on curved diffusion paths. On the Waymo Open Motion Dataset, FlowS achieves state-of-the-art Soft mAP {(0.4804) and mAP (0.4703) with ensemble at 75\,FPS} with single-step inference, demonstrating that local transport conditioning makes one-step generative motion prediction practical for safety-critical autonomy. Code and pretrained models will be released upon acceptance.
>
---
#### [new 012] Walk With Me: Long-Horizon Social Navigation for Human-Centric Outdoor Assistance
- **分类: cs.RO**

- **简介: 该论文属于人机协作导航任务，解决户外长时社交导航问题。提出无地图框架Walk with Me，结合语义目标定位、路径规划与安全推理，实现高效户外导航。**

- **链接: [https://arxiv.org/pdf/2604.26839](https://arxiv.org/pdf/2604.26839)**

> **作者:** Lingfeng Zhang; Xiaoshuai Hao; Xizhou Bu; Yingbo Tang; Hongsheng Li; Jinghui Lu; Xiu-shen Wei; Jiayi Ma; Yu Liu; Jing Zhang; Hangjun Ye; Xiaojun Liang; Long Chen; Wenbo Ding
>
> **摘要:** Assisting humans in open-world outdoor environments requires robots to translate high-level natural-language intentions into safe, long-horizon, and socially compliant navigation behavior. Existing map-based methods rely on costly pre-built HD maps, while learning-based policies are mostly limited to indoor and short-horizon settings. To bridge this gap, we propose Walk with Me, a map-free framework for long-horizon social navigation from high-level human instructions. Walk with Me leverages GPS context and lightweight candidate points-of-interest from a public map API for semantic destination grounding and waypoint proposal. A High-Level Vision-Language Model grounds abstract instructions into concrete destinations and plans coarse waypoint sequences. During execution, an observation-aware routing mechanism determines whether the Low-Level Vision-Language-Action policy can handle the current situation or whether explicit safety reasoning from the High-Level VLM is needed. Routine segments are executed by the Low-Level VLA, while complex situations such as crowded crossings trigger high-level reasoning and stop-and-wait behavior when unsafe. By combining semantic intent grounding, map-free long-horizon planning, safety-aware reasoning, and low-level action generation, Walk with Me enables practical outdoor social navigation for human-centric assistance.
>
---
#### [new 013] LLM-Flax : Generalizable Robotic Task Planning via Neuro-Symbolic Approaches with Large Language Models
- **分类: cs.RO**

- **简介: 该论文提出LLM-Flax框架，解决机器人任务规划中手动规则制定和训练数据依赖的问题。通过三阶段方法，利用大语言模型实现自动化规则生成、故障恢复和零样本对象评分，提升规划效果。**

- **链接: [https://arxiv.org/pdf/2604.26569](https://arxiv.org/pdf/2604.26569)**

> **作者:** Seongmin Kim; Daegyu Lee
>
> **摘要:** Deploying a neuro-symbolic task planner on a new domain today requires significant manual effort: a domain expert must author relaxation and complementary rules, and hundreds of training problems must be solved to supervise a Graph Neural Network (GNN) object scorer. We propose LLM-Flax, a three-stage framework that eliminates all three sources of manual effort using a locally hosted LLM given only a PDDL domain file. Stage 1 automatically generates relaxation and complementary rules via structured prompting with format validation and self-correction. Stage 2 introduces LLM-guided failure recovery with a feasibility-gated budget policy that explicitly reserves API latency cost before each LLM call, preventing the downstream relaxation fallback from being starved. Stage 3 replaces the domain-trained GNN entirely with zero-shot LLM object importance scoring, requiring no training data. We evaluate all three stages on the MazeNamo benchmark across 10x10, 12x12, and 15x15 grids (8 benchmarks total). LLM-Flax achieves average SR 0.945 versus the manual baseline's 0.828 (+0.117), matching or outperforming manual rules on every one of the eight benchmarks. On 12x12 Expert, LLM-Flax attains SR 0.733 where the manual planner fails entirely (SR 0.000); on 15x15 Hard, it achieves SR 1.000 versus Manual's 0.900. Stage 3 demonstrates feasibility (SR 0.720 on 12x12 Hard with no training data) but faces a context-window bottleneck at scale, pointing to the primary open challenge for future work.
>
---
#### [new 014] Reactive Motion Generation via Phase-varying Neural Potential Functions
- **分类: cs.RO**

- **简介: 该论文属于运动生成任务，解决轨迹交叉时的运动不确定性问题。提出PNPF框架，通过相位变量优化动力系统，提升轨迹稳定性和抗干扰能力。**

- **链接: [https://arxiv.org/pdf/2604.26450](https://arxiv.org/pdf/2604.26450)**

> **作者:** Ahmet Tekden; Dimitrios Kanoulas; Aude Billard; Yasemin Bekiroglu
>
> **备注:** Accepted by IEEE Robotics and Automation Letters (RAL)
>
> **摘要:** Dynamical systems (DS) methods for Learning-from-Demonstration (LfD) provide stable, continuous policies from few demonstrations. First-order dynamical systems (DS) are effective for many point-to-point and periodic tasks, as long as a unique velocity is defined for each state. For tasks with intersections (e.g., drawing an "8"), extensions such as second-order dynamics or phase variables are often used. However, by incorporating velocity, second-order models become sensitive to disturbances near intersections, as velocity is used to disambiguate motion direction. Moreover, this disambiguation may fail when nearly identical position-velocity pairs correspond to different onward motions. In contrast, phase-based methods rely on open-loop time or phase variables, which limit their ability to recover after perturbations. We introduce Phase-varying Neural Potential Functions (PNPF), an LfD framework that conditions a potential function on a phase variable which is estimated directly from state progression, rather than on open-loop temporal inputs. This phase variable allows the system to handle state revisits, while the learned potential function generates local vector fields for reactive and stable control. PNPF generalizes effectively across point-to-point, periodic, and full 6D motion tasks, outperforms existing baselines on trajectories with intersections, and demonstrates robust performance in real-time robotic manipulation under external disturbances.
>
---
#### [new 015] Rule-based High-Level Coaching for Goal-Conditioned Reinforcement Learning in Search-and-Rescue UAV Missions Under Limited-Simulation Training
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究无人机在搜救任务中的决策问题，提出一种结合规则策略与强化学习的框架，解决有限仿真训练下的安全与效率问题。**

- **链接: [https://arxiv.org/pdf/2604.26833](https://arxiv.org/pdf/2604.26833)**

> **作者:** Mahya Ramezani; Holger Voos
>
> **摘要:** This paper presents a hierarchical decision-making framework for unmanned aerial vehicle (UAV) missions motivated by search-and-rescue (SAR) scenarios under limited simulation training. The framework combines a fixed rule-based high-level advisor with an online goal-conditioned low-level reinforcement learning (RL) controller. To stress-test early adaptation, we also consider a strict no-pretraining deployment regime. The high-level advisor is defined offline from a structured task specification and compiled into deterministic rules. It provides interpretable mission- and safety-aware guidance through recommended actions, avoided actions, and regime-dependent arbitration weights. The low-level controller learns online from task-defined dense rewards and reuses experience through a mode-aware prioritized replay mechanism augmented with rule-derived metadata. We evaluate the framework on two tasks: battery-aware multi-goal delivery and moving-target delivery in obstacle-rich environments. Across both tasks, the proposed method improves early safety and sample efficiency primarily by reducing collision terminations, while preserving the ability to adapt online to scenario-specific dynamics.
>
---
#### [new 016] A Scaled Three-Vehicle Platooning Platform
- **分类: cs.RO**

- **简介: 该论文属于车辆协同控制任务，旨在解决车队行驶中的路径跟踪稳定性问题。研究构建了一个缩比多车平台，用于实验验证协同控制与人机协作的自主性。**

- **链接: [https://arxiv.org/pdf/2604.25963](https://arxiv.org/pdf/2604.25963)**

> **作者:** Kaiyue Lu; Qiaoxuan Zhang; Yukun Lu
>
> **摘要:** Vehicle platooning has attracted increasing attention as a promising approach to improve traffic efficiency, energy consumption, and roadway safety through coordinated multi-vehicle operation. A key challenge in platooning lies in maintaining stable and accurate path tracking during dynamic maneuvers such as lane changes, where lateral deviations and heading disturbances generated by the lead vehicle may propagate downstream to following vehicles. Robust longitudinal and lateral control systems are therefore essential not only for individual vehicle tracking performance, but also for overall platoon stability. For experimental studies, the Intelligent Mobility and Robotics Lab (IMRL) develops a scaled multi-vehicle platform for autonomous platooning research, with a particular emphasis on cooperative control and human-in-the-loop autonomy. This platform consists of one human-operable lead vehicle and two autonomous followers, enabling controlled and repeatable experiments on leader-follower coordination. Compared with full-scale field testing, this scaled platform offers a safer, lower-cost, and more flexible environment for rapid prototyping, controller validation, and multi-agent autonomy studies, while providing stronger physical realism than purely simulation-based evaluations.
>
---
#### [new 017] Multi-Periodogram Velocity Estimation with Irregular Reference Signals for Robot-Aided ISAC
- **分类: cs.RO; cs.IT**

- **简介: 该论文属于机器人辅助的感知与通信任务，解决移动机器人在使用不规则参考信号时的测速问题。提出多周期图算法提升低信噪比下的测速性能。**

- **链接: [https://arxiv.org/pdf/2604.25974](https://arxiv.org/pdf/2604.25974)**

> **作者:** Yi Geng; Pan Cao; Ting Zeng; Yongqian Deng
>
> **备注:** Accepted by ICC2026
>
> **摘要:** This paper addresses velocity estimation within robot-aided integrated sensing and communications (ISAC), where mobile robots act as sensing nodes but can only opportunistically reuse irregular 5G/6G reference signals (RSs). We show that the velocity profile induced by such irregular time-domain patterns can be decomposed into a periodic-peak component and an amplitude-shaping (weighting) component. Leveraging this structure, we propose a multi-periodogram velocity estimation algorithm that is standard-compliant and does not require new sensing-dedicated RSs or 3GPP modifications. Simulation results demonstrate that, compared with conventional periodogram processing, the proposed method improves low-SNR robustness by achieving a 3 dB SNR gain at the 10% missed-detection rate and reducing false alarms by 51%.
>
---
#### [new 018] ATLAS: An Annotation Tool for Long-horizon Robotic Action Segmentation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ATLAS，用于长时程机器人动作分割的标注工具。解决传统工具在多模态数据同步和效率上的不足，支持动作边界标注与任务结果记录。**

- **链接: [https://arxiv.org/pdf/2604.26637](https://arxiv.org/pdf/2604.26637)**

> **作者:** Sergej Stanovcic; Daniel Sliwowski; Dongheui Lee
>
> **备注:** 7 pages, 2 figures, 2 tables
>
> **摘要:** Annotating long-horizon robotic demonstrations with precise temporal action boundaries is crucial for training and evaluating action segmentation and manipulation policy learning methods. Existing annotation tools, however, are often limited: they are designed primarily for vision-only data, do not natively support synchronized visualization of robot-specific time-series signals (e.g., gripper state or force/torque), or require substantial effort to adapt to different dataset formats. In this paper, we introduce ATLAS, an annotation tool tailored for long-horizon robotic action segmentation. ATLAS provides time-synchronized visualization of multi-modal robotic data, including multi-view video and proprioceptive signals, and supports annotation of action boundaries, action labels, and task outcomes. The tool natively handles widely used robotics dataset formats such as ROS bags and the Reinforcement Learning Dataset (RLDS) format, and provides direct support for specific datasets such as REASSEMBLE. ATLAS can be easily extended to new formats via a modular dataset abstraction layer. Its keyboard-centric interface minimizes annotation effort and improves efficiency. In experiments on a contact-rich assembly task, ATLAS reduced the average per-action annotation time by at least 6% compared to ELAN, while the inclusion of time-series data improved temporal alignment with expert annotations by more than 2.8% and decreased boundary error fivefold compared to vision-only annotation tools.
>
---
#### [new 019] 2D and 3D Grasp Planners for the GET Asymmetrical Gripper
- **分类: cs.RO**

- **简介: 该论文属于抓取规划任务，旨在提升GET不对称夹爪的抓取性能。提出两种方法：基于单视角RGB-D图像的GET-2D-1.0和基于3D模型的GET-3D-1.0，通过实验验证其效果。**

- **链接: [https://arxiv.org/pdf/2604.26212](https://arxiv.org/pdf/2604.26212)**

> **作者:** Andrew Goldberg; Ethan Ransing; Anton Kourakin; Cael Magner; Edward H. Adelson; Ken Goldberg
>
> **摘要:** In this paper, we introduce GET-2D-1.0, a fast grasp planner for the GET asymmetrical gripper that operates from a single-view RGB-D image, using the Ferrari-Canny metric and a novel sampling strategy, and GET-3D-1.0, a mesh-based method using a 3D gripper model and ray-tracing. We evaluate both grasp planners against baselines with physical experiments, which suggest that GET-2D-1.0 can improve over a bounding box baseline by over 40% in lift success, shake survival, and force resistance. Experiments with GET-3D-1.0 suggest slight improvement compared to GET-2D-1.0 on lift success and shake survival, but are more computationally expensive, averaging 17 seconds of planning compared to 683 ms for GET-2D-1.0.
>
---
#### [new 020] FalconApp: Rapid iPhone Deployment of End-to-End Perception via Automatically Labeled Synthetic Data
- **分类: cs.RO**

- **简介: 该论文提出FalconApp，解决机器人感知中数据标注耗时问题，通过合成数据快速部署iPhone端的物体检测与位姿估计模型。**

- **链接: [https://arxiv.org/pdf/2604.25949](https://arxiv.org/pdf/2604.25949)**

> **作者:** Yan Miao; Will Shen; Sayan Mitra
>
> **摘要:** Reliable perception for robotics depends on large-scale labeled data, yet real-world datasets rely on heavy manual annotation and are time-consuming to produce. We present FalconApp, an iPhone app with an end-to-end frontend-backend pipeline that turns a short handheld capture of a rigid object into a perception module for mask detection and 6-DoF pose estimation. Our core contribution is a rapid mobile deployment pipeline paired with a photorealistic auto-labeling workflow: from a user-captured video of an object, FalconApp reconstructs an editable GSplat asset, composites it with diverse photorealistic backgrounds, renders synthetic images with ground-truth masks and poses, trains the perception module, and deploys it back to the iPhone frontend. Experiments across five rigid objects with diverse geometry and appearance show that FalconApp produces usable perception models with about 20 minutes of synthetic-data generation and training per object on average, around 30 ms end-to-end on-device latency on iPhone, and better overall pose accuracy than a PnP baseline on 4 / 5 objects in both simulation and real-world evaluation.
>
---
#### [new 021] STAR-Filter: Efficient Convex Free-Space Approximation via Starshaped Set Filtering in Noisy Environments
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决复杂环境中高效生成无碰撞空间的问题。提出STAR-Filter方法，通过星形集过滤快速生成凸区域，提升计算效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.26626](https://arxiv.org/pdf/2604.26626)**

> **作者:** Yuwei Wu; Yichen Zhao; Dexter Ong; Vijay Kumar
>
> **摘要:** Approximating collision-free space is fundamental to robot planning in complex environments. Convex geometric representations, such as polytopes and ellipsoids, are widely employed due to their structural properties, which can be easily integrated with convex optimization. Iterative optimization-based inflation methods can generate large volume polytopes in cluttered environments, but their efficiency degrades as the obstacle set becomes more complex or when sensor data are noisy. These methods are also sensitive to initialization and often rely on accurate geometric models. In this paper, we propose the STAR-Filter, a lightweight framework that employs starshaped set construction as a fast filter for convex region generation in collision-free space. By identifying obstacle points as active supporting constraints, the proposed method significantly reduces redundant computation while preserving feasibility and robustness to sensor noise. We provide theoretical and numerical analyses that characterize the structural properties of the starshaped set and proposed pipeline in environments of varying complexity. Simulation results show that the proposed framework achieves the lowest computation time and reduces conservativeness in polytope generation for real-world noisy and large-scale data. We demonstrate the effectiveness of the framework for Safe Flight Corridor (SFC) generation and agile quadrotor planning in noisy environments.
>
---
#### [new 022] Safe Navigation using Neural Radiance Fields via Reachable Sets
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于自主导航任务，解决复杂环境中安全路径规划问题。通过结合可达集与NeRF，利用约束优化控制实现安全导航。**

- **链接: [https://arxiv.org/pdf/2604.26899](https://arxiv.org/pdf/2604.26899)**

> **作者:** Omanshu Thapliyal; Malarvizhi Sankaranarayanasamy; Ravigopal Vennelakanti
>
> **备注:** 5 pages, 8 figures, 2026 4th International Conference on Mechatronics, Control and Robotics (ICMCR)
>
> **摘要:** Safe navigation in cluttered environments is an important challenge for autonomous systems. Robots navigating through obstacle ridden scenarios need to be able to navigate safely in the presence of obstacles, goals, and ego objects of varying geometries. In this work, reachable set representations of the robot's real-time capabilities in the state space can be utilized to capture safe navigation requirements. While neural radiance fields (NeRFs) are utilized to compute, store, and manipulate the volumetric representations of the obstacles, or ego vehicle, as needed. Constrained optimal control is employed to represent the resulting path planning problem, involving linear matrix inequality constraints. We present simulation results for path planning in the presence of numerous obstacles in two different scenarios. Safe navigation is demonstrated through using reachable sets in the corresponding constrained optimal control problems.
>
---
#### [new 023] Benchmarking the Safety of Large Language Models for Robotic Health Attendant Control
- **分类: cs.AI; cs.CY; cs.RO**

- **简介: 该论文属于安全评估任务，旨在解决LLM在医疗机器人控制中的安全性问题。通过测试和分析，揭示了模型安全性能的差异及影响因素。**

- **链接: [https://arxiv.org/pdf/2604.26577](https://arxiv.org/pdf/2604.26577)**

> **作者:** Mahiro Nakao; Kazuhiro Takemoto
>
> **备注:** 20 pages, 9 figures, 3 tables, 8 pages supplementary material
>
> **摘要:** Large language models (LLMs) are increasingly considered for deployment as the control component of robotic health attendants, yet their safety in this context remains poorly characterized. We introduce a dataset of 270 harmful instructions spanning nine prohibited behavior categories grounded in the American Medical Association Principles of Medical Ethics, and use it to evaluate 72 LLMs in a simulation environment based on the Robotic Health Attendant framework. The mean violation rate across all models was 54.4\%, with more than half exceeding 50\%, and violation rates varied substantially across behavior categories, with superficially plausible instructions such as device manipulation and emergency delay proving harder to refuse than overtly destructive ones. Model size and release date were the primary determinants of safety performance among open-weight models, and proprietary models were substantially safer than open-weight counterparts (median 23.7\% versus 72.8\%). Medical domain fine-tuning conferred no significant overall safety benefit, and a prompt-based defense strategy produced only a modest reduction in violation rates among the least safe models, leaving absolute violation rates at levels that would preclude safe clinical deployment. These findings demonstrate that safety evaluation must be treated as a first-class criterion in the development and deployment of LLMs for robotic health attendants.
>
---
#### [new 024] FruitProM-V2: Robust Probabilistic Maturity Estimation and Detection of Fruits and Vegetables
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于果实成熟度估计任务，旨在解决传统分类方法在相邻成熟阶段边界模糊的问题。通过将成熟度建模为连续变量并使用概率检测头进行预测，提升估计的可靠性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.26084](https://arxiv.org/pdf/2604.26084)**

> **作者:** Rahul Harsha Cheppally; Sidharth Rai; Sudan Baral; Benjamin Vail; Ajay Sharda
>
> **摘要:** Accurate fruit maturity identification is essential for determining harvest timing, as incorrect assessment directly affects yield and post-harvest quality. Although ripening is a continuous biological process, vision-based maturity estimation is typically formulated as a multi-class classification task, which imposes sharp boundaries between visually similar stages. To examine this limitation, we perform an annotation reliability study with two independent annotators on a held-out tomato dataset and observe disagreement concentrated near adjacent maturity stages. Motivated by this observation, we model maturity as a latent continuous variable and predict it probabilistically using a distributional detection head, converting the distribution into class probabilities through the cumulative distribution function (CDF). The proposed formulation maintains comparable performance to a standard detector under clean labels while better representing uncertainty. Furthermore, when controlled label noise is introduced during training, the probabilistic model demonstrates improved robustness relative to the baseline, indicating that explicitly modeling maturity uncertainty leads to more reliable visual maturity estimation.
>
---
#### [new 025] Three-Step Nav: A Hierarchical Global-Local Planner for Zero-Shot Vision-and-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决零样本导航中路径偏移、过早停止等问题。提出Three-Step Nav，通过全局、当前和回溯三步策略提升导航性能。**

- **链接: [https://arxiv.org/pdf/2604.26946](https://arxiv.org/pdf/2604.26946)**

> **作者:** Wanrong Zheng; Yunhao Ge; Laurent Itti
>
> **备注:** Accepted to AISTATS 2026. Code: this https URL
>
> **摘要:** Breakthrough progress in vision-based navigation through unknown environments has been achieved by using multimodal large language models (MLLMs). These models can plan a sequence of motions by evaluating the current view at each time step against the task and goal given to the agent. However, current zero-shot Vision-and-Language Navigation (VLN) agents powered by MLLMs still tend to drift off course, halt prematurely, and achieve low overall success rates. We propose Three-Step Nav to counteract these failures with a three-view protocol: First, "look forward" to extract global landmarks and sketch a coarse plan. Then, "look now" to align the current visual observation with the next sub-goal for fine-grained guidance. Finally, "look backward" audits the entire trajectory to correct accumulated drift before stopping. Requiring no gradient updates or task-specific fine-tuning, our planner drops into existing VLN pipelines with minimal overhead. Three-Step Nav achieves state-of-the-art zero-shot performance on the R2R-CE and RxR-CE dataset. Our code is available at this https URL.
>
---
#### [new 026] Persona-Based Process Design for Assistive Human-Robot Workplaces for Persons with Disabilities
- **分类: cs.HC; cs.RO; eess.SY**

- **简介: 该论文属于人机协作设计任务，旨在解决残疾人工作场所的个性化与通用性矛盾。通过基于角色的流程设计方法，生成适应不同用户需求的协作策略。**

- **链接: [https://arxiv.org/pdf/2604.26527](https://arxiv.org/pdf/2604.26527)**

> **作者:** Nils Mandischer; Daria Eckert and; Lars Mikelsons
>
> **备注:** Accepted at IEEE International Conference on Human-Machine Systems (ICHMS), Singapore, 2026
>
> **摘要:** Human-robot interaction is emerging as an important paradigm for integrating persons with disabilities into the workplace. While these systems can enable individuals to work, their design is mostly personalized, hindering widespread use beyond the individual user. The universal design paradigm is a central pillar of inclusive design, describing usability of systems by all. To incorporate universal design into process design for human-robot workplaces expert knowledge is required that is often not available. To simplify process design of human-robot workplaces, we propose a persona-based design approach. First, typical impairments prevalent in the workforce or particularly relevant for the processes are abstracted into personas with disabilities. The work process is subdivided into sequential actions. For each action and persona, strategies are developed to reach the action goal by a design thinking approach. The resulting actions are ordered by level of robot assistance, i.e. robot involvement, and implemented in a behavior tree. Therefore, the macro-behavior of the workplace may adapt to individual personas online. We demonstrate the method in a collaborative box folding process with a total of seven personas with disabilities. The persona-based process design shows promising results by generating more comprehensive process strategies while enabling adaptive behavior in the sense of universal design.
>
---
#### [new 027] Why Domain Matters: A Preliminary Study of Domain Effects in Underwater Object Detection
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于目标检测任务，旨在解决水下域迁移问题。针对现有基准未能反映真实场景因素的缺陷，提出一种基于图像、场景和采集特征的域标注框架，支持更准确的性能评估与故障分析。**

- **链接: [https://arxiv.org/pdf/2604.26174](https://arxiv.org/pdf/2604.26174)**

> **作者:** Melanie Wille; Dimity Miller; Tobias Fischer; Scarlett Raine
>
> **备注:** Poster Presentation at ICRA 2026 Workshop S2S
>
> **摘要:** Domain shift, where deviations between training and deployment data distributions degrade model performance, is a key challenge in underwater environments. Existing benchmarks testing performance for underwater domain shift simulate variability through synthetic style transfer. This fails to capture intrinsic scene factors such as visibility, illumination, scene composition, or acquisition factors, limiting analysis of real-world effects. We propose a labeling framework that defines underwater domains using measurable image, scene, and acquisition characteristics. Unlike prior benchmarks, it captures physically meaningful factors, enabling semantically consistent image grouping and supporting domain-specific evaluation of detection performance including failure analysis. We validate this on public datasets, showing systematic variations across domain factors and revealing hidden failure modes.
>
---
#### [new 028] Edge AI for Automotive Vulnerable Road User Safety: Deployable Detection via Knowledge Distillation
- **分类: cs.CV; cs.LG; cs.RO; eess.IV**

- **简介: 该论文属于目标检测任务，旨在解决边缘设备上VRU安全检测的模型精度与计算约束矛盾。通过知识蒸馏方法，将大模型知识迁移至小模型，实现高效准确检测。**

- **链接: [https://arxiv.org/pdf/2604.26857](https://arxiv.org/pdf/2604.26857)**

> **作者:** Akshay Karjol; Darrin M. Hanna
>
> **备注:** 6 pages, 3 figures
>
> **摘要:** Deploying accurate object detection for Vulnerable Road User (VRU) safety on edge hardware requires balancing model capacity against computational constraints. Large models achieve high accuracy but fail under INT8 quantization required for edge deployment, while small models sacrifice detection performance. This paper presents a knowledge distillation (KD) framework that trains a compact YOLOv8-S student (11.2M parameters) to mimic a YOLOv8-L teacher (43.7M parameters), achieving 3.9x compression while preserving quantization robustness. We evaluate on full-scale BDD100K (70K training images) with Post-Training Quantization to INT8. The teacher suffers catastrophic degradation under INT8 (-23% mAP), while the KD student retains accuracy (-5.6% mAP). Analysis reveals that KD transfers precision calibration rather than raw detection capacity: the KD student achieves 0.748 precision versus 0.653 for direct training at INT8, a 14.5% gain at equivalent recall, reducing false alarms by 44% versus the collapsed teacher. At INT8, the KD student exceeds the teacher's FP32 precision (0.748 vs. 0.718) in a model 3.9x smaller. These findings establish knowledge distillation as a requirement for deploying accurate, safety-critical VRU detection on edge hardware.
>
---
## 更新

#### [replaced 001] RetroMotion: Retrocausal Motion Forecasting Models are Instructable
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于多智能体运动预测任务，解决复杂场景下轨迹预测的不确定性问题。通过分解分布并引入回溯因果信息流，提升预测准确性与指令适应能力。**

- **链接: [https://arxiv.org/pdf/2505.20414](https://arxiv.org/pdf/2505.20414)**

> **作者:** Royden Wagner; Omer Sahin Tas; Felix Hauser; Marlon Steiner; Dominik Strutz; Abhishek Vivekanandan; Jaime Villa; Yinzhe Shen; Carlos Fernandez; Christoph Stiller
>
> **备注:** CVPRW26
>
> **摘要:** Motion forecasts of road users (i.e., agents) vary in complexity depending on the number of agents, scene constraints, and interactions. In particular, the output space of joint trajectory distributions grows exponentially with the number of agents. Therefore, we decompose multi-agent motion forecasts into (1) marginal distributions for all modeled agents and (2) joint distributions for interacting agents. Using a transformer model, we generate joint distributions by re-encoding marginal distributions followed by pairwise modeling. This incorporates a retrocausal flow of information from later points in marginal trajectories to earlier points in joint trajectories. For each time step, we model the positional uncertainty using compressed exponential power distributions. Notably, our method achieves strong results in the Waymo Interaction Prediction Challenge and generalizes well to the Argoverse 2 and V2X-Seq datasets. Additionally, our method provides an interface for issuing instructions. We show that standard motion forecasting training implicitly enables the model to follow instructions and adapt them to the scene context. GitHub repository: this https URL
>
---
#### [replaced 002] Geometric Inverse Flight Dynamics on SO(3) and Application to Tethered Fixed-Wing Aircraft
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究固定翼飞机的逆向飞行动力学问题，基于SO(3)几何框架，解决姿态与力平衡的协调控制，提出闭环轨迹到输入映射，应用于系留飞行分析。**

- **链接: [https://arxiv.org/pdf/2602.17166](https://arxiv.org/pdf/2602.17166)**

> **作者:** Antonio Franchi; Chiara Gabellieri
>
> **备注:** ACCEPTED ICUAS 2026
>
> **摘要:** We present a robotics-oriented, coordinate-free formulation of inverse flight dynamics for fixed-wing aircraft on SO(3). Translational force balance is written in the world frame and rotational dynamics in the body frame; aerodynamic directions (drag, lift, side) are defined geometrically, avoiding local attitude coordinates. Enforcing coordinated flight (no sideslip), we derive a closed-form trajectory-to-input map yielding the attitude, angular velocity, and thrust-angle-of-attack pair, and we recover the aerodynamic moment coefficients component-wise. Applying such a map to tethered flight on spherical parallels, we obtain analytic expressions for the required bank angle and identify a specific zero-bank locus where the tether tension exactly balances centrifugal effects, highlighting the decoupling between aerodynamic coordination and the apparent gravity vector. Under a simple lift/drag law, the minimal-thrust angle of attack admits a closed form. These pointwise quasi-steady inversion solutions become steady-flight trim when the trajectory and rotational dynamics are time-invariant. The framework bridges inverse simulation in aeronautics with geometric modeling in robotics, providing a rigorous building block for trajectory design and feasibility checks.
>
---
#### [replaced 003] Hybrid Diffusion for Simultaneous Symbolic and Continuous Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人长期任务规划领域，解决复杂决策与连续轨迹生成的难题。通过结合符号计划与连续轨迹的混合扩散模型，提升任务执行效果。**

- **链接: [https://arxiv.org/pdf/2509.21983](https://arxiv.org/pdf/2509.21983)**

> **作者:** Sigmund Hennum Høeg; Aksel Vaaler; Chaoqi Liu; Olav Egeland; Yilun Du
>
> **备注:** 10 pages, 11 figures. This work has been submitted to the IEEE for possible publication. See this https URL for the project website
>
> **摘要:** Constructing robots to accomplish long-horizon tasks is a long-standing challenge within artificial intelligence. Approaches using generative methods, particularly Diffusion Models, have gained attention due to their ability to model continuous robotic trajectories for planning and control. However, we show that these models struggle with long-horizon tasks that involve complex decision-making and, in general, are prone to confusing different modes of behavior, leading to failure. To remedy this, we propose to augment continuous trajectory generation by simultaneously generating a high-level symbolic plan. We show that this requires a novel mix of discrete variable diffusion and continuous diffusion, which dramatically outperforms the baselines. In addition, we illustrate how this hybrid diffusion process enables flexible trajectory synthesis, allowing us to condition synthesized actions on partial and complete symbolic conditions.
>
---
#### [replaced 004] CoFL: Continuous Flow Fields for Language-Conditioned Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出CoFL，用于语言引导的导航任务，解决传统系统依赖模块化流程和有限监督的问题。通过端到端学习连续流场，实现高效实时导航与闭环控制。**

- **链接: [https://arxiv.org/pdf/2603.02854](https://arxiv.org/pdf/2603.02854)**

> **作者:** Haokun Liu; Zhaoqi Ma; Yicheng Chen; Masaki Kitagawa; Wentao Zhang; Zicen Xiong; Jinjie Li; Moju Zhao
>
> **备注:** 18 pages, 13 figures
>
> **摘要:** Existing language-conditioned navigation systems typically rely on modular pipelines or trajectory generators, but the latter use each scene--instruction annotation mainly to supervise one start-conditioned rollout. To address these limitations, we present CoFL, an end-to-end policy that maps a bird's-eye view (BEV) observation and a language instruction to a continuous flow field for navigation. CoFL reformulates navigation as workspace-conditioned field learning rather than start-conditioned trajectory prediction: it learns local motion vectors at arbitrary BEV locations, turning each scene--instruction annotation into dense spatial control supervision. Trajectories are generated from any start by numerical integration of the predicted field, enabling simple real-time rollout and closed-loop recovery. To enable large-scale training and evaluation, we build a dataset of over 500k BEV image--instruction pairs, each procedurally annotated with a flow field and a trajectory derived from semantic maps built on Matterport3D and ScanNet. Evaluating on strictly unseen scenes, CoFL significantly outperforms modular Vision-Language Model (VLM)-based planners and trajectory generation policies in both navigation precision and safety, while maintaining real-time inference. Finally, we deploy CoFL zero-shot in real-world experiments with BEV observations across multiple layouts, maintaining feasible closed-loop control and a high success rate.
>
---
#### [replaced 005] Source-Free Bistable Fluidic Gripper for Size-Selective and Stiffness-Adaptive Grasping
- **分类: cs.RO**

- **简介: 该论文属于软体机器人抓取任务，解决传统软抓手依赖外部能源的问题。通过内部液体 redistributing 实现无源、自适应抓取。**

- **链接: [https://arxiv.org/pdf/2511.03691](https://arxiv.org/pdf/2511.03691)**

> **作者:** Zhihang Qin; Yueheng Zhang; Wan Su; Linxin Hou; Shenghao Zhou; Zhijun Chen; Yu Jun Tan; Cecilia Laschi
>
> **摘要:** Conventional fluid-driven soft grippers typically depend on external sources, which limit portability and long-term autonomy. This work introduces a self-contained soft gripper with fixed size that operates solely through internal liquid redistribution among three interconnected bistable snap-through chambers. When the top sensing chamber deforms upon contact, the displaced liquid triggers snap-through expansion of the grasping chambers, enabling stable and size-selective grasping without continuous energy input. The internal hydraulic feedback further allows passive adaptation of gripping pressure to object stiffness. This source-free and compact design opens new possibilities for lightweight, stiffness-adaptive fluid-driven manipulation in soft robotics, providing a feasible approach for targeted size-specific sampling and operation in underwater and field environments.
>
---
#### [replaced 006] The Alignment Flywheel: A Governance-Centric Hybrid MAS for Architecture-Agnostic Safety
- **分类: cs.MA; cs.LG; cs.RO**

- **简介: 该论文提出一种治理导向的混合多智能体系统架构，解决自主系统安全治理问题。通过解耦决策与安全，实现可审计、可更新的安全控制。**

- **链接: [https://arxiv.org/pdf/2603.02259](https://arxiv.org/pdf/2603.02259)**

> **作者:** Elias Malomgré; Pieter Simoens
>
> **备注:** Accepted for the EMAS workshop at AAMAS 2026
>
> **摘要:** Multi-agent systems provide mature methodologies for role decomposition, coordination, and normative governance, capabilities that remain essential as increasingly powerful autonomous decision components are embedded within agent-based systems. While learned and generative models substantially expand system capability, their safety behavior is often entangled with training, making it opaque, difficult to audit, and costly to update after deployment. This paper formalizes the Alignment Flywheel as a governance-centric hybrid MAS architecture that decouples decision generation from safety governance. A Proposer, representing any autonomous decision component, generates candidate trajectories, while a Safety Oracle returns raw safety signals through a stable interface. An enforcement layer applies explicit risk policy at runtime, and a governance MAS supervises the Oracle through auditing, uncertainty-driven verification, and versioned refinement. The central engineering principle is patch locality: many newly observed safety failures can be mitigated by updating the governed oracle artifact and its release pipeline rather than retracting or retraining the underlying decision component. The architecture is implementation-agnostic with respect to both the Proposer and the Safety Oracle, and specifies the roles, artifacts, protocols, and release semantics needed for runtime gating, audit intake, signed patching, and staged rollout across distributed deployments. The result is a hybrid MAS engineering framework for integrating highly capable but fallible autonomous systems under explicit, version-controlled, and auditable oversight.
>
---
#### [replaced 007] DC-Ada: Reward-Only Decentralized Sensor Adaptation for Heterogeneous Multi-Robot Teams
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文提出DC-Ada方法，解决异构多机器人团队在传感器差异下的适应问题，通过奖励优化实现去中心化感知适配。**

- **链接: [https://arxiv.org/pdf/2604.03905](https://arxiv.org/pdf/2604.03905)**

> **作者:** Saad Alqithami
>
> **摘要:** Heterogeneity is a defining feature of deployed multi-robot teams: platforms often differ in sensing modalities, ranges, fields of view, and failure patterns. Controllers trained under nominal sensing can degrade sharply when deployed on robots with missing or mismatched sensors, even when the task and action interface are unchanged. We present DC-Ada, a reward-only decentralized adaptation method that keeps a pretrained shared policy frozen and instead adapts compact per-robot observation transforms to map heterogeneous sensing into a fixed inference interface. DC-Ada is gradient-free and communication-minimal: it uses budgeted accept/reject random search with short common-random-number rollouts under a strict step budget. We evaluate DC-Ada against four baselines in a deterministic 2D multi-robot simulator covering warehouse logistics, search and rescue, and collaborative mapping, across four heterogeneity regimes (H0--H3) and five seeds with a matched budget of $200{,}000$ joint environment steps per run. Results show that heterogeneity can substantially degrade a frozen shared policy and that no single mitigation dominates across all tasks and metrics. Observation normalization is strongest for reward robustness in warehouse logistics and competitive in search and rescue, while the frozen shared policy is strongest for reward in collaborative mapping. DC-Ada offers a useful complementary operating point: it improves completion most clearly in severe coverage-based mapping while requiring only scalar team returns and no policy fine-tuning or persistent communication. These results position DC-Ada as a practical deploy-time adaptation method for heterogeneous teams.
>
---
#### [replaced 008] A Virtual Mechanical Interaction Layer Enables Resilient Human-to-Robot Object Handovers
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，解决物体交接中机器人适应性不足的问题。提出虚拟模型控制与增强现实技术，提升交接的鲁棒性和交互体验。**

- **链接: [https://arxiv.org/pdf/2511.19543](https://arxiv.org/pdf/2511.19543)**

> **作者:** Omar Faris; Sławomir Tadeja; Fulvio Forni
>
> **备注:** Accepted for publication in IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Object handover is a common form of interaction that is widely present in collaborative tasks. However, achieving it efficiently remains a challenge. We address the problem of ensuring resilient robotic actions that can adapt to complex changes in object pose during human-to-robot object handovers. We propose the use of Virtual Model Control to create an interaction layer that controls the robot and adapts to the dynamic changes in the handover process. Additionally, we propose the use of augmented reality to facilitate bidirectional communication between humans and robots during handovers. We assess the performance of our controller in a set of experiments that demonstrate its resilience to various sources of uncertainties, including complex changes to the object's pose during the handover. Finally, we performed a user study with 16 participants to understand human preferences for different robot control profiles and augmented reality visuals in object handovers. Our results showed a general preference for the proposed approach and revealed insights that can guide further development in adapting the interaction with the user.
>
---
#### [replaced 009] Bridging Discrete Planning and Continuous Execution for Redundant Robot
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决冗余机械臂在离散规划与连续执行间的问题。通过引入改进的规划策略和优化的逆运动学方法，提升路径质量与执行稳定性。**

- **链接: [https://arxiv.org/pdf/2604.02021](https://arxiv.org/pdf/2604.02021)**

> **作者:** Teng Yan; Yue Yu; Yihan Liu; Bingzhuo Zhong
>
> **备注:** 8 pages, 3 figures. Submitted to IFAC World Congress 2026
>
> **摘要:** Voxel-grid reinforcement learning is widely adopted for path planning in redundant manipulators due to its simplicity and reproducibility. However, direct execution through point-wise numerical inverse kinematics on 7-DoF arms often yields step-size jitter, abrupt joint transitions, and instability near singular configurations. This work proposes a bridging framework between discrete planning and continuous execution without modifying the discrete planner itself. On the planning side, step-normalized 26-neighbor Cartesian actions and a geometric tie-breaking mechanism are introduced to suppress unnecessary turns and eliminate step-size oscillations. On the execution side, a task-priority damped least-squares (TP-DLS) inverse kinematics layer is implemented. This layer treats end-effector position as a primary task, while posture and joint centering are handled as subordinate tasks projected into the null space, combined with trust-region clipping and joint velocity constraints. On a 7-DoF manipulator in random sparse, medium, and dense environments, this bridge raises planning success in dense scenes from about 0.58 to 1.00, shortens representative path length from roughly 1.53 m to 1.10 m, and while keeping end-effector error below 1 mm, reduces peak joint accelerations by over an order of magnitude, substantially improving the continuous execution quality of voxel-based RL paths on redundant manipulators.
>
---
#### [replaced 010] A Multimodal Depth-Aware Method For Embodied Reference Understanding
- **分类: cs.CV; cs.HC; cs.RO**

- **简介: 该论文属于Embodied Reference Understanding任务，旨在解决多目标场景下的指代消歧问题。通过结合语言模型、深度图和深度感知模块，提升目标检测的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2510.08278](https://arxiv.org/pdf/2510.08278)**

> **作者:** Fevziye Irem Eyiokur; Dogucan Yaman; Hazım Kemal Ekenel; Alexander Waibel
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Embodied Reference Understanding requires identifying a target object in a visual scene based on both language instructions and pointing cues. While prior works have shown progress in open-vocabulary object detection, they often fail in ambiguous scenarios where multiple candidate objects exist in the scene. To address these challenges, we propose a novel ERU framework that jointly leverages LLM-based data augmentation, depth-map modality, and a depth-aware decision module. This design enables robust integration of linguistic and embodied cues, improving disambiguation in complex or cluttered environments. Experimental results on two datasets demonstrate that our approach significantly outperforms existing baselines, achieving more accurate and reliable referent detection.
>
---
#### [replaced 011] EvolvingAgent: Curriculum Self-evolving Agent with Continual World Model for Long-Horizon Tasks
- **分类: cs.RO**

- **简介: 该论文提出EvolvingAgent，解决开放环境中长任务的自主完成问题。通过自规划、自控制和自反思模块，提升任务成功率并减少无效动作。**

- **链接: [https://arxiv.org/pdf/2502.05907](https://arxiv.org/pdf/2502.05907)**

> **作者:** Tongtong Feng; Xin Wang; Zekai Zhou; Ren Wang; Yuwei Zhan; Guangyao Li; Qing Li; Wenwu Zhu
>
> **摘要:** Completing Long-Horizon (LH) tasks in open-ended worlds is an important yet difficult problem for embodied agents. Existing approaches suffer from two key challenges: (1) they heavily rely on experiences obtained from human-created data or curricula, failing to autonomously update and select multimodal experiences, and (2) they may encounter catastrophic forgetting issues when faced with new tasks, failing to autonomously update world knowledge. To solve these challenges, this paper presents {\bf EvolvingAgent}, a curriculum self-evolving agent with a continual World Model (WM), which can autonomously complete various LH tasks across environments through self-planning, self-control, and self-reflection, without human intervention. Specifically, EvolvingAgent contains three modules, i.e., i) the experience-driven task planner, which uses an LLM along with multimodal experiences to convert LH tasks into executable sub-tasks; ii) the WM-guided action controller, which leverages WM to generate low-level actions and incorporates a self-verification mechanism to update multimodal experiences; iii) the Curriculum Learning (CL) -based reflector, which implements a two-stage CL algorithm to select multimodal experiences for task-adaptive WM updates. By building a planner-controller-reflector closed-loop dynamic, the continual WM for EvolvingAgent can autonomously update multimodal experiences and world knowledge. We conducted extensive experiments on Minecraft, compared with existing methods, EvolvingAgent can improve 111.74{\%} average success rate, reduce more than 6x ineffective actions, and generalize to the Atari environment with human-level performance.
>
---
#### [replaced 012] Explainable Representation of Finite-Memory Policies for POMDPs using Decision Trees
- **分类: cs.AI; cs.LG; cs.RO; eess.SY**

- **简介: 该论文属于强化学习领域，解决POMDP中有限记忆策略的可解释性问题。通过结合Mealy机与决策树，提出更简洁、易理解的策略表示方法。**

- **链接: [https://arxiv.org/pdf/2411.13365](https://arxiv.org/pdf/2411.13365)**

> **作者:** Muqsit Azeem; Debraj Chakraborty; Sudeep Kanav; Jan Kretinsky
>
> **备注:** Full version of the extended abstract accepted at AAMAS 2026
>
> **摘要:** Partially Observable Markov Decision Processes (POMDPs) are a fundamental framework for decision-making under uncertainty and partial observability. Since in general optimal policies may require infinite memory, they are hard to implement and often render most problems undecidable. Consequently, finite-memory policies are mostly considered instead. However, the algorithms for computing them are typically very complex, and so are the resulting policies. Facing the need for their explainability, we provide a representation of such policies, both (i) in an interpretable formalism and (ii) typically of smaller size, together yielding higher explainability. To that end, we combine models of Mealy machines and decision trees; the latter describing simple, stationary parts of the policies and the former describing how to switch among them. We design a translation for policies of the finite-state-controller (FSC) form from standard literature and show how our method smoothly generalizes to other variants of finite-memory policies. Further, we identify specific properties of recently used "attractor-based" policies, which allow us to construct yet simpler and smaller representations. Finally, we illustrate the higher explainability in a few case studies.
>
---
#### [replaced 013] ViTaPEs: Visuotactile Position Encodings for Cross-Modal Alignment in Multimodal Transformers
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出ViTaPEs，解决多模态对齐问题，通过双阶段位置编码提升视觉与触觉信息融合效果，增强模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2505.20032](https://arxiv.org/pdf/2505.20032)**

> **作者:** Fotios Lygerakis; Ozan Özdenizci; Elmar Rückert
>
> **摘要:** Tactile sensing provides local essential information that is complementary to visual perception, such as texture, compliance, and force. Despite recent advances in visuotactile representation learning, challenges remain in fusing these modalities and generalizing across tasks and environments without heavy reliance on pre-trained vision-language models. Moreover, existing methods do not study positional encodings, thereby overlooking the multi-stage spatial reasoning needed to capture fine-grained visuotactile correlations. We introduce ViTaPEs, a transformer-based architecture for learning task-agnostic visuotactile representations from paired vision and tactile inputs. Our key idea is a two-stage positional injection: local (modality-specific) positional encodings are added within each stream, and a global positional encoding is added on the joint token sequence immediately before attention, providing a shared positional vocabulary at the stage where cross-modal interaction occurs. We make the positional injection points explicit and conduct controlled ablations that isolate their effect before a token-wise nonlinearity versus immediately before self-attention. Experiments on multiple large-scale real-world datasets show that ViTaPEs not only surpasses state-of-the-art baselines across various recognition tasks but also demonstrates zero-shot generalization to unseen, out-of-domain scenarios. We further demonstrate the transfer-learning strength of \emph{ViTaPEs} in a robotic grasping task, where it outperforms state-of-the-art baselines in predicting grasp success. Project page: this https URL
>
---
#### [replaced 014] InCoM: Intent-Driven Perception and Structured Coordination for Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文针对移动操作任务，解决基座与机械臂控制耦合及感知注意力分配问题，提出InCoM框架，实现意图驱动的感知和结构化协调。**

- **链接: [https://arxiv.org/pdf/2602.23024](https://arxiv.org/pdf/2602.23024)**

> **作者:** Jiahao Liu; Cui Wenbo; Zhongpu Xia; Haoran Li; Dongbin Zhao
>
> **摘要:** Mobile manipulation is a fundamental capability for general-purpose robotic agents, requiring both coordinated control of the mobile base and manipulator and robust perception under dynamically changing viewpoints. However, existing approaches face two key challenges: strong coupling between base and arm actions complicates control optimization, and perceptual attention is often poorly allocated as viewpoints shift during mobile manipulation. We propose InCoM, an intent-driven perception and structured coordination framework for mobile manipulation. InCoM infers latent motion intent to dynamically reweight multi-scale perceptual features, enabling stage-adaptive allocation of perceptual attention. To support robust cross-modal perception, InCoM further incorporates a geometric-semantic structured alignment mechanism that enhances multimodal correspondence. On the control side, we design a decoupled coordinated flow matching action decoder that explicitly models coordinated base-arm action generation, alleviating optimization difficulties caused by control coupling. Experimental results demonstrate that InCoM significantly outperforms state-of-the-art methods, achieving success rate gains of 28.2%, 26.1%, and 23.6% across three ManiSkill-HAB scenarios without privileged information. Furthermore, its effectiveness is consistently validated in real-world mobile manipulation tasks, where InCoM maintains a superior success rate over existing baselines.
>
---
#### [replaced 015] Open-H-Embodiment: A Large-Scale Dataset for Enabling Foundation Models in Medical Robotics
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Open-H-Embodiment数据集，解决医疗机器人领域数据不足与封闭的问题，支持基础模型训练，提升机器人自主能力。**

- **链接: [https://arxiv.org/pdf/2604.21017](https://arxiv.org/pdf/2604.21017)**

> **作者:** Open-H-Embodiment Consortium; Nigel Nelson; Juo-Tung Chen; Jesse Haworth; Xinhao Chen; Lukas Zbinden; Dianye Huang; Alaa Eldin Abdelaal; Alberto Arezzo; Ayberk Acar; Farshid Alambeigi; Carlo Alberto Ammirati; Yunke Ao; Pablo David Aranda Rodriguez; Soofiyan Atar; Mattia Ballo; Noah Barnes; Federica Barontini; Filip Binkiewicz; Peter Black; Sebastian Bodenstedt; Leonardo Borgioli; Nikola Budjak; Benjamin Calmé; Fabio Carrillo; Nicola Cavalcanti; Changwei Chen; Haoxin Chen; Sihang Chen; Qihan Chen; Zhongyu Chen; Ziyang Chen; Shing Shin Cheng; Meiqing Cheng; Min Cheng; Zih-Yun Sarah Chiu; Xiangyu Chu; Camilo Correa-Gallego; Giulio Dagnino; Anton Deguet; Jacob Delgado; Jonathan C. DeLong; Kaizhong Deng; Alexander Dimitrakakis; Qingpeng Ding; Hao Ding; Giovanni Distefano; Daniel Donoho; Anqing Duan; Marco Esposito; Shane Farritor; Jad Fayad; Zahi Fayad; Mario Ferradosa; Filippo Filicori; Chelsea Finn; Philipp Fürnstahl; Jiawei Ge; Stamatia Giannarou; Xavier Giralt Ludevid; Frederic Giraud; Aditya Amit Godbole; Ken Goldberg; Antony Goldenberg; Diego Granero Marana; Xiaoqing Guo; Tamás Haidegger; Evan Hailey; Pascal Hansen; Ziyi Hao; Kush Hari; Kengo Hayashi; Jonathon Hawkins; Shelby Haworth; Ortrun Hellig; S. Duke Herrell; Zhouyang Hong; Andrew Howe; Junlei Hu; Zhaoyang Jacopo Hu; Ria Jain; Mohammad Rafiee Javazm; Howard Ji; Rui Ji; Jianmin Ji; Zhongliang Jiang; Dominic Jones; Jeffrey Jopling; Britton Jordan; Ran Ju; Michael Kam; Luoyao Kang; Fausto Kang; Siddhartha Kapuria; Peter Kazanzides; Sonika Kiehler; Ethan Kilmer; Ji Woong Kim; Przemysław Korzeniowski; Chandra Kuchi
>
> **备注:** Project website: this https URL
>
> **摘要:** Autonomous medical robots hold promise to improve patient outcomes, reduce provider workload, democratize access to care, and enable superhuman precision. However, autonomous medical robotics has been limited by a fundamental data problem: existing medical robotic datasets are small, single-embodiment, and rarely shared openly, restricting the development of foundation models that the field needs to advance. We introduce Open-H-Embodiment, the largest open dataset of medical robotic video with synchronized kinematics to date, spanning more than 49 institutions and multiple robotic platforms including the CMR Versius, Intuitive Surgical's da Vinci, da Vinci Research Kit (dVRK), Rob Surgical BiTrack, Virtual Incision's MIRA, Moon Surgical Maestro, and a variety of custom systems, spanning surgical manipulation, robotic ultrasound, and endoscopy procedures. We demonstrate the research enabled by this dataset through two foundation models. GR00T-H is the first open foundation vision-language-action model for medical robotics, which is the only evaluated model to achieve full end-to-end task completion on a structured suturing benchmark (25% of trials vs. 0% for all others) and achieves 64% average success across a 29-step ex vivo suturing sequence. We also train Cosmos-H-Surgical-Simulator, the first action-conditioned world model to enable multi-embodiment surgical simulation from a single checkpoint, spanning nine robotic platforms and supporting in silico policy evaluation and synthetic data generation for the medical domain. These results suggest that open, large-scale medical robot data collection can serve as critical infrastructure for the research community, enabling advances in robot learning, world modeling, and beyond.
>
---
#### [replaced 016] SD2AIL: Adversarial Imitation Learning from Synthetic Demonstrations via Diffusion Models
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于模仿学习任务，旨在解决专家示范数据收集困难的问题。通过扩散模型生成合成示范，并结合优先回放策略提升性能。**

- **链接: [https://arxiv.org/pdf/2512.18583](https://arxiv.org/pdf/2512.18583)**

> **作者:** Pengcheng Li; Qiang Fang; Tong Zhao; Yixing Lan; Xin Xu
>
> **备注:** This paper has the following problems: Limited novelty, not clearly differentiated from existing methods/concepts; The level of experimental validation is limited; Sufficient serious structural, language, or other issues that impact the comprehensibility of the manuscript
>
> **摘要:** Adversarial Imitation Learning (AIL) is a dominant framework in imitation learning that infers rewards from expert demonstrations to guide policy optimization. Although providing more expert demonstrations typically leads to improved performance and greater stability, collecting such demonstrations can be challenging in certain scenarios. Inspired by the success of diffusion models in data generation, we propose SD2AIL, which utilizes synthetic demonstrations via diffusion models. We first employ a diffusion model in the discriminator to generate synthetic demonstrations as pseudo-expert data that augment the expert demonstrations. To selectively replay the most valuable demonstrations from the large pool of (pseudo-) expert demonstrations, we further introduce a prioritized expert demonstration replay strategy (PEDR). The experimental results on simulation tasks demonstrate the effectiveness and robustness of our method. In particular, in the Hopper task, our method achieves an average return of 3441, surpassing the state-of-the-art method by 89. Our code will be available at this https URL.
>
---
#### [replaced 017] Neural-Geometric Tunnel Traversal: Localization-free UAV Flight with Tilted LiDARs
- **分类: cs.RO**

- **简介: 该论文属于无人机自主导航任务，解决在无GNSS信号的隧道中导航问题。通过结合LiDAR、几何方法和深度学习，实现方向调整与安全路径规划。**

- **链接: [https://arxiv.org/pdf/2404.09688](https://arxiv.org/pdf/2404.09688)**

> **作者:** Lorenzo Cano; Alejandro R. Mosteo; Danilo Tardioli
>
> **摘要:** Navigation of UAVs in challenging environments like tunnels or mines, where it is not possible to use GNSS methods to self-localize, illumination may be uneven or nonexistent, and wall features are likely to be scarce, is a complex task, especially if the navigation has to be done at high speed. In this paper we propose a novel proof-of-concept navigation technique for UAVs based on the use of LiDAR information through the joint use of geometric and machine-learning algorithms. The perceived information is processed by a deep neural network to establish the yaw of the UAV with respect to the tunnel's longitudinal axis, in order to adjust the direction of navigation. Additionally, a geometric method is used to compute the safest location inside the tunnel (i.e. the one that maximizes the distance to the closest obstacle). This information proves to be sufficient for simple yet effective navigation in straight and curved tunnels.
>
---
#### [replaced 018] Learning Vision-Based Omnidirectional Navigation: A Teacher-Student Approach Using Monocular Depth Estimation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人导航任务，解决工业环境中2D LiDAR传感器无法感知三维障碍物的问题。通过教师-学生框架，利用单目深度估计实现无LiDAR的全景导航。**

- **链接: [https://arxiv.org/pdf/2603.01999](https://arxiv.org/pdf/2603.01999)**

> **作者:** Jan Finke; Wayne Paul Martis; Adrian Schmelter; Lars Erbach; Christian Jestel; Marvin Wiedemann
>
> **摘要:** Reliable obstacle avoidance in industrial settings demands 3D scene understanding, but widely used 2D LiDAR sensors perceive only a single horizontal slice of the environment, missing critical obstacles above or below the scan plane. We present a teacher-student framework for vision-based mobile robot navigation that eliminates the need for LiDAR sensors. A teacher policy trained via Proximal Policy Optimization (PPO) in NVIDIA Isaac Lab leverages privileged 2D LiDAR observations that account for the full robot footprint to learn robust navigation. The learned behavior is distilled into a student policy that relies solely on monocular depth maps predicted by a fine-tuned Depth Anything V2 model from four RGB cameras. The complete inference pipeline, comprising monocular depth estimation (MDE), policy execution, and motor control, runs entirely onboard an NVIDIA Jetson Orin AGX mounted on a DJI RoboMaster platform, requiring no external computation for inference. In simulation, the student achieves success rates of 82-96.5%, consistently outperforming the standard 2D LiDAR teacher (50-89%). In real-world experiments, the MDE-based student outperforms the 2D LiDAR teacher when navigating around obstacles with complex 3D geometries, such as overhanging structures and low-profile objects, that fall outside the single scan plane of a 2D LiDAR.
>
---
#### [replaced 019] Variable Elimination in Hybrid Factor Graphs for Discrete-Continuous Inference & Estimation
- **分类: cs.RO**

- **简介: 该论文属于机器人学中的估计与推理任务，解决混合离散-连续变量的建模问题。提出一种新的混合因子图框架及变量消去算法，实现精确的联合估计与推断。**

- **链接: [https://arxiv.org/pdf/2601.00545](https://arxiv.org/pdf/2601.00545)**

> **作者:** Varun Agrawal; Frank Dellaert
>
> **摘要:** Many problems in robotics involve both continuous and discrete components, and modeling them together for estimation tasks has been a long standing and difficult problem. Hybrid Factor Graphs give us a mathematical framework to model these types of problems, however existing approaches for solving them are based on approximations. In this work, we propose a new framework for hybrid factor graphs along with a novel variable elimination algorithm to produce a hybrid Bayes network, which can be used for exact Maximum A Posteriori estimation and marginalization over both sets of variables. Our approach first develops a novel hybrid Gaussian factor which can connect to both discrete and continuous variables, and a hybrid conditional which can represent multiple continuous hypotheses conditioned on the discrete variables. Using these representations, we derive the process of hybrid variable elimination under the Conditional Linear Gaussian scheme, giving us exact posteriors as a hybrid Bayes network. To bound the number of discrete hypotheses, we use a tree-structured representation of the factors coupled with a simple pruning and probabilistic assignment scheme, which allows for tractable inference. We demonstrate the applicability of our framework on a large scale SLAM dataset and a real world pose graph optimization problem, both with ambiguous measurements which require discrete choices to be made for the most likely measurements. Our demonstrated results showcase the accuracy, generality, and simplicity of our hybrid factor graph framework.
>
---
#### [replaced 020] M2R2: MultiModal Robotic Representation for Temporal Action Segmentation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于时间动作分割任务，解决机器人多模态特征融合与复用问题，提出M2R2模型结合本体和外部感知信息，并引入新训练策略提升性能。**

- **链接: [https://arxiv.org/pdf/2504.18662](https://arxiv.org/pdf/2504.18662)**

> **作者:** Daniel Sliwowski; Dongheui Lee
>
> **备注:** 8 pages, 6 figures, 2 tables
>
> **摘要:** Temporal action segmentation (TAS) has long been a key area of research in both robotics and computer vision. In robotics, algorithms have primarily focused on leveraging proprioceptive information to determine skill boundaries, with recent approaches in surgical robotics incorporating vision. In contrast, computer vision typically relies on exteroceptive sensors, such as cameras. Existing multimodal TAS models in robotics integrate feature fusion within the model, making it difficult to reuse learned features across different models. Meanwhile, pretrained vision-only feature extractors commonly used in computer vision struggle in scenarios with limited object visibility. In this work, we address these challenges by proposing M2R2, a multimodal feature extractor tailored for TAS, which combines information from both proprioceptive and exteroceptive sensors. We introduce a novel training strategy that enables the reuse of learned features across multiple TAS models. Our method sets a new state-of-the-art performance on three robotic datasets REASSEMBLE, (Im)PerfectPour, and JIGSAWS. Additionally, we conduct an extensive ablation study to evaluate the contribution of different modalities in robotic TAS tasks.
>
---
#### [replaced 021] OnSiteVRU: A High-Resolution Trajectory Dataset for High-Density Vulnerable Road Users
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出OnSiteVRU数据集，用于高密度VRU轨迹研究，解决现有数据不足问题，提升自动驾驶系统安全性。**

- **链接: [https://arxiv.org/pdf/2503.23365](https://arxiv.org/pdf/2503.23365)**

> **作者:** Zhangcun Yan; Jianqiang Li; Peng Hang; Jian Sun
>
> **摘要:** With the acceleration of urbanization and the growth of transportation demands, the safety of vulnerable road users (VRUs, such as pedestrians and cyclists) in mixed traffic flows has become increasingly prominent, necessitating high-precision and diverse trajectory data to support the development and optimization of autonomous driving systems. However, existing datasets fall short in capturing the diversity and dynamics of VRU behaviors, making it difficult to meet the research demands of complex traffic environments. To address this gap, this study developed the OnSiteVRU datasets, which cover a variety of scenarios, including intersections, road segments, and urban villages. These datasets provide trajectory data for motor vehicles, electric bicycles, and human-powered bicycles, totaling approximately 17,429 trajectories with a precision of 0.04 seconds. The datasets integrate both aerial-view natural driving data and onboard real-time dynamic detection data, along with environmental information such as traffic signals, obstacles, and real-time maps, enabling a comprehensive reconstruction of interaction events. The results demonstrate that VRU\_Data outperforms traditional datasets in terms of VRU density and scene coverage, offering a more comprehensive representation of VRU behavioral characteristics. This provides critical support for traffic flow modeling, trajectory prediction, and autonomous driving virtual testing. The dataset is publicly available for download at: this https URL.
>
---
#### [replaced 022] Dynamically Extensible and Retractable Robotic Leg Linkages for Multi-task Execution in Search and Rescue Scenarios
- **分类: cs.RO**

- **简介: 该论文属于搜索与救援机器人领域，旨在解决地形适应与高力输出难以兼顾的问题。通过设计可伸缩的五杆连杆机构，实现腿部形态转换，提升机器人任务执行能力。**

- **链接: [https://arxiv.org/pdf/2511.10816](https://arxiv.org/pdf/2511.10816)**

> **作者:** William Harris; Lucas Yager; Syler Sylvester; Elizabeth Peiros; Micheal C. Yip
>
> **摘要:** Search and rescue (SAR) robots are required to quickly traverse terrain and perform high-force rescue tasks, necessitating both terrain adaptability and controlled high-force output. Few platforms exist today for SAR, and fewer still have the ability to cover both tasks of terrain adaptability and high-force output when performing extraction. While legged robots offer significant ability to traverse uneven terrain, they typically are unable to incorporate mechanisms that provide variable high-force outputs, unlike traditional wheel-based drive trains. This work introduces a novel concept for a dynamically extensible and retractable robot leg. Leveraging a dynamically extensible and retractable five-bar linkage design, it allows for mechanically switching between height-advantaged and force-advantaged configurations via a geometric transformation. A testbed evaluated leg performance across linkage geometries and operating modes, with empirical and analytical analyses conducted on stride length, force output, and stability. The results demonstrate that the morphing leg offers a promising path toward SAR robots that can both navigate terrain quickly and perform rescue tasks effectively.
>
---
#### [replaced 023] FASTER: Rethinking Real-Time Flow VLAs
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型的实时执行任务，解决反应延迟问题。通过提出FASTER方法，优化采样策略以缩短反应时间，提升机器人实时响应能力。**

- **链接: [https://arxiv.org/pdf/2603.19199](https://arxiv.org/pdf/2603.19199)**

> **作者:** Yuxiang Lu; Zhe Liu; Xianzhe Fan; Zhenya Yang; Jinghua Hou; Junyi Li; Kaixin Ding; Hengshuang Zhao
>
> **备注:** Project page: this https URL
>
> **摘要:** Real-time execution is crucial for deploying Vision-Language-Action (VLA) models in the physical world. Existing asynchronous inference methods primarily optimize trajectory smoothness, but neglect the critical latency in reacting to environmental changes. By rethinking the notion of reaction in action chunking policies, this paper presents a systematic analysis of the factors governing reaction time. We show that reaction time follows a uniform distribution determined jointly by the Time to First Action (TTFA) and the execution horizon. Moreover, we reveal that the standard practice of applying a constant schedule in flow-based VLAs can be inefficient and forces the system to complete all sampling steps before any movement can start, forming the bottleneck in reaction latency. To overcome this issue, we propose Fast Action Sampling for ImmediaTE Reaction (FASTER). By introducing a Horizon-Aware Schedule, FASTER adaptively prioritizes near-term actions during flow sampling, compressing the denoising of the immediate reaction by tenfold (e.g., in $\pi_{0.5}$ and X-VLA) into a single step, while preserving the quality of long-horizon trajectory. Coupled with a streaming client-server pipeline, FASTER substantially reduces the effective reaction latency on real robots, especially when deployed on consumer-grade GPUs. Real-world experiments, including a highly dynamic table tennis task, prove that FASTER unlocks unprecedented real-time responsiveness for generalist policies, enabling rapid generation of accurate and smooth trajectories.
>
---
#### [replaced 024] Distributional Stability of Tangent-Linearized Gaussian Inference on Smooth Manifolds
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究机器人中的流形上高斯推理的稳定性问题，解决非高斯性和几何依赖性难题，通过线性化方法推导出稳定界，并提供诊断工具。**

- **链接: [https://arxiv.org/pdf/2602.19179](https://arxiv.org/pdf/2602.19179)**

> **作者:** Junghoon Seo; Hakjin Lee; Jaehoon Sim
>
> **备注:** To appear in IEEE Robotics and Automation Letters (IEEE RA-L)
>
> **摘要:** Gaussian inference on smooth manifolds is central to robotics, but exact marginalization and conditioning are generally non-Gaussian and geometry-dependent. We study tangent-linearized Gaussian inference and derive explicit non-asymptotic $W_2$ stability bounds for projection marginalization and surface-measure conditioning. The bounds separate local second-order geometric distortion from nonlocal tail leakage and, for Gaussian inputs, yield closed-form diagnostics from $(\mu,\Sigma)$ and curvature/reach surrogates. Circle and planar-pushing experiments validate the predicted calibration transition near $\sqrt{\|\Sigma\|_{\mathrm{op}}}/R\approx 1/6$ and indicate that normal-direction uncertainty is the dominant failure mode when locality breaks. These diagnostics provide practical triggers for switching from single-chart linearization to multi-chart or sample-based manifold inference. Code and Jupyter notebooks are available at this https URL.
>
---
#### [replaced 025] VLN-Cache: Enabling Token Caching for VLN Models with Visual/Semantic Dynamics Awareness
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于视觉语言导航任务，解决大模型推理成本高的问题。提出VLN-Cache框架，通过视觉和语义动态感知实现高效token缓存。**

- **链接: [https://arxiv.org/pdf/2603.07080](https://arxiv.org/pdf/2603.07080)**

> **作者:** Zihao Zheng; Zhihao Mao; Xingyue Zhou; Jiayu Chen; Maoliang Li; Xinhao Sun; Hailong Zou; Zhaobo Zhang; Xuanzhe Liu; Donggang Cao; Hong Mei; Xiang Chen
>
> **摘要:** Vision-and-Language Navigation (VLN) increasingly relies on large vision-language models, but their inference cost conflicts with real-time deployment. Token caching is a promising training-free strategy that avoids redundant computation by reusing stable visual tokens across frames. However, existing methods assume a static camera and fixed semantic focus, assumptions that VLN fundamentally violates. We identify two failure modes: (1) visual dynamics, where viewpoint shift displaces token positions across frames, causing position-wise matching to pair misaligned content; (2) semantic dynamics, where token relevance shifts across task stages as navigation progresses, making cached states stale. We propose VLN-Cache, a visual-dynamic-aware and semantic-dynamic-aware caching framework that introduces view-aligned remapping to recover geometric correspondences and a task-relevance saliency filter to veto reuse at semantic transitions. A layer-adaptive entropy policy further balances the per-layer reuse budget. Experiments on the R2R-CE simulation benchmark show up to 1.52x speedup while maintaining competitive navigation success rates.
>
---
#### [replaced 026] R2RGEN: Real-to-Real 3D Data Generation for Spatially Generalized Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决空间泛化问题。通过生成真实世界3D数据，提升策略在不同空间配置下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.08547](https://arxiv.org/pdf/2510.08547)**

> **作者:** Xiuwei Xu; Angyuan Ma; Hankun Li; Bingyao Yu; Zheng Zhu; Jie Zhou; Jiwen Lu
>
> **备注:** Accepted to RSS 2026. Project page: this https URL
>
> **摘要:** Towards the aim of generalized robotic manipulation, spatial generalization is the most fundamental capability that requires the policy to work robustly under different spatial distribution of objects, environment and agent itself. To achieve this, substantial human demonstrations need to be collected to cover different spatial configurations for training a generalized visuomotor policy via imitation learning. Prior works explore a promising direction that leverages data generation to acquire abundant spatially diverse data from minimal source demonstrations. However, most approaches face significant sim-to-real gap and are often limited to constrained settings, such as fixed-base scenarios and predefined camera viewpoints. In this paper, we propose a real-to-real 3D data generation framework (R2RGen) that directly augments the pointcloud observation-action pairs to generate real-world data. R2RGen is simulator- and rendering-free, thus being efficient and plug-and-play. Specifically, we propose a unified three-stage framework, which (1) pre-processes source demonstrations under different camera setups in a shared 3D space with scene / trajectory parsing; (2) augments objects and robot's position with a group-wise backtracking strategy; (3) aligns the distribution of generated data with real-world 3D sensor using camera-aware post-processing. Empirically, R2RGen substantially enhances data efficiency on extensive experiments and demonstrates strong potential for scaling and application on mobile manipulation.
>
---
