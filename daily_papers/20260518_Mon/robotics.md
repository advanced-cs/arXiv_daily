# 机器人 cs.RO

- **最新发布 53 篇**

- **更新 29 篇**

## 最新发布

#### [new 001] NavRL++: A System-Level Framework for Improving Sim-to-Real Transfer in Reinforcement Learning-Based Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决强化学习中的sim-to-real迁移问题。提出NavRL++框架，包含训练策略和部署管道，提升真实环境下的导航性能。**

- **链接: [https://arxiv.org/pdf/2605.15559](https://arxiv.org/pdf/2605.15559)**

> **作者:** Zhefan Xu; Hanyu Jin; Kenji Shimada
>
> **备注:** 18 pages, 18 figures, 6 tables
>
> **摘要:** Recent years have witnessed significant progress in autonomous navigation using reinforcement learning. However, existing approaches largely emphasize reinforcement learning framework design, such as input representations, action spaces, and reward functions, while providing limited analysis of sim-to-real transfer and insufficient insight into how training strategies affect real-world deployment performance. To bridge this gap, we not only introduce an effective RL framework but also present a complete training and deployment pipeline, along with a systematic empirical study that disentangles the key factors affecting sim-to-real transfer in reinforcement learning-based navigation, including sensor noise, perception failures, system latency, and control response. Building on insights from this analysis, we introduce perturbation-aware fine-tuning, a post-training adaptation strategy that improves transfer robustness by explicitly accounting for empirically identified domain discrepancies. To further mitigate perception degradation and enhance control smoothness in real-world deployment, we propose a Transformer-based temporal reasoning policy that leverages short-horizon observation for navigation control. We quantitatively evaluate how individual sim-to-real perturbations and training design choices impact navigation performance across environments. Experimental results demonstrate that the proposed training strategy and policy architecture outperform learning-based baselines in both static and dynamic environments, while achieving performance comparable to optimization-based planners in static settings. We validate our approach through real-world deployment on multiple robotic platforms, including aerial and legged robots, across navigation-centric tasks such as exploration and inspection, demonstrating zero-shot sim-to-real transfer.
>
---
#### [new 002] HoloMotion-1 Technical Report
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出HoloMotion-1，用于零样本全身运动跟踪的类人机器人运动基础模型。解决运动捕捉数据局限性问题，通过混合数据训练提升泛化能力与跟踪精度。**

- **链接: [https://arxiv.org/pdf/2605.15336](https://arxiv.org/pdf/2605.15336)**

> **作者:** Maiyue Chen; Kaihui Wang; Bo Zhang; Xihan Ma; Zhiyuan Yang; Yi Ren; Qijun Huang; Zihao Zhu; Yucheng Wang; Zhizhong Su
>
> **备注:** 20 pages, 4 figures, 6 tables. Technical report
>
> **摘要:** In this report, we present HoloMotion-1, a humanoid motion foundation model for zero-shot whole-body motion tracking. A key innovation of HoloMotion-1 is to scale control-policy training with a large-scale hybrid motion corpus, where video-reconstructed motions from in-the-wild videos provide the dominant source of motion diversity, while curated motion-capture and in-house motion data provide higher-fidelity supervision and deployment-oriented coverage. This data regime enables HoloMotion-1 to move beyond conventional MoCap-only training and exposes the policy to substantially broader behaviors, capture conditions, and motion styles. Learning from such heterogeneous data introduces new challenges, including reconstruction noise, source-domain mismatch, uneven motion quality, and the need for temporal modeling under large behavioral variation. To address these challenges, HoloMotion-1 integrates large-capacity temporal modeling, a sparsely activated Mixture-of-Experts Transformer with KV-cache inference for real-time control, and a sequence-level training strategy that improves learning efficiency on extended motion sequences. Extensive experiments on multiple unseen motion benchmarks show that HoloMotion-1 generalizes robustly across diverse motion types and capture conditions, significantly improves tracking accuracy over prior methods, and transfers directly to a real humanoid robot without task-specific fine-tuning.
>
---
#### [new 003] Fast Expanding Safe Circular Regions for Efficient Local Path Planning
- **分类: cs.RO**

- **简介: 该论文属于机器人局部路径规划任务，旨在解决复杂环境中计算效率低的问题。提出一种基于几何的算法，通过扩展安全圆形区域实现快速导航。**

- **链接: [https://arxiv.org/pdf/2605.16009](https://arxiv.org/pdf/2605.16009)**

> **作者:** Scott Fredriksson; Akshit Saradagi; George Nikolakopoulos
>
> **备注:** Accepted by the IFAC World Congress 2026
>
> **摘要:** Local navigation is one of the fundamental problems in robot navigation, and numerous approaches have been proposed over the years, including methods such as the Dynamic Window Approach, Model Predictive Control, and more recently, Control Barrier Functions and machine learning based techniques. While these methods perform well in simple environments, many of them rely on optimization or learning based procedures that can struggle in more complex scenarios. In contrast, this article proposes a more geometric algorithmic approach that enables a local navigation method with faster computation times and longer planning horizons. The proposed method is based on the computation of a sequence of circular regions from a local LiDAR scan that expand in the direction of the goal and capture free local navigable space. The proposed method was implemented in the ROS2 framework and evaluated in a simulated environment.
>
---
#### [new 004] GAP: Geometric Anchor Pre-training for Data-Efficient Visuomotor Learning of Manipulation Tasks
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，解决少量专家示范下的视觉-运动策略学习问题。提出GAP方法，在预训练阶段规范空间适配器，提升策略学习的稳定性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.15836](https://arxiv.org/pdf/2605.15836)**

> **作者:** Davide Buoso; Andrea Protopapa; Stefano Di Carlo; Francesca Pistilli; Giuseppe Averta
>
> **备注:** Project webpage at this https URL
>
> **摘要:** Learning visuomotor policies from scarce expert demonstrations remains a core challenge in robotic manipulation. A primary hurdle lies in distilling high-dimensional RGB representations into control-relevant geometry without overfitting. While using frozen pre-trained Vision Foundation Models (VFMs) improves data efficiency, it also shifts most task adaptation onto a small spatial pooling module, which can latch onto task-irrelevant shortcuts and lose geometric grounding when finetuned with few data samples. More broadly, pre-trained visual representations used for policy learning have been observed to struggle under even minor scene perturbations, highlighting the need for robustness-oriented inductive biases. We propose Geometric Anchor Pre-training (GAP), a simple, action-free warm-up stage that regularizes the spatial adapter before downstream imitation learning. GAP pre-trains the pooling layer on a lightweight simulated proxy task where object masks are available at no cost, encouraging the adapter to produce keypoints that lie on the object, cover its spatial extent, and remain sharp and repeatable over time. This yields stable geometric anchors that provide a reliable coordinate interface for few-shot policy learning, while keeping the VFM frozen. We evaluate GAP on RoboMimic and ManiSkill under severe data scarcity (15-50 demonstrations) and domain shift. A simple adapter regularized with GAP consistently outperforms stronger attention-based poolers and end-to-end fine-tuning, achieving 62% success on RoboMimic Can with 15 demonstrations (+16% over AFA), 63% on the long-horizon high-precision Tool Hang task with 50 demonstrations, and 61% on ManiSkill StackCube with 30 demonstrations (+11% over full fine-tuning). The proxy stage is lightweight and fully decoupled from downstream tasks, making it practical to reuse across environments and manipulation skills.
>
---
#### [new 005] DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于刚性配准任务，解决噪声、部分重叠数据和实时处理难题。提出双空间方法，结合特征匹配与局部几何匹配优势，提升配准效率与精度。**

- **链接: [https://arxiv.org/pdf/2508.17034](https://arxiv.org/pdf/2508.17034)**

> **作者:** Jiayi Li; Yuxin Yao; Qiuhang Lu; Juyong Zhang
>
> **备注:** Accepted to CVPR 2026, Project page: this https URL
>
> **摘要:** Noisy, partially overlapping data and the need for real-time processing pose major challenges for rigid registration. Considering that feature-based matching can handle large transformation differences but suffers from limited accuracy, while local geometry-based matching can achieve fine-grained local alignment but relies heavily on a good initial transformation, we propose a novel dual-space paradigm to fully leverage the strengths of both approaches. First, we introduce an efficient filtering mechanism consisting of a computationally lightweight one-point RANSAC algorithm and a subsequent refinement module to eliminate unreliable feature-based correspondences. Subsequently, we treat the filtered correspondences as anchor points, extract geometric proxies, and formulate an effective objective function with a tailored solver to estimate the transformation. Experiments verify our method's effectiveness, as demonstrated by a 32x CPU-time speedup over MAC on KITTI with comparable accuracy. Project page: this https URL.
>
---
#### [new 006] Hybrid LLM-based Intelligent Framework for Robot Task Scheduling
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人任务调度领域，旨在提升施工机器人任务安排的效率与适应性。通过融合大语言模型，构建智能调度框架，解决传统方法在动态环境中的不足。**

- **链接: [https://arxiv.org/pdf/2605.15486](https://arxiv.org/pdf/2605.15486)**

> **作者:** Swayamjit Saha; Subhabrata Das; Haonan Duan; Xiao-Yang Liu
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** This study introduces intelligent frameworks that use Large Language Models (LLMs) to improve task scheduling for construction robots. The LLM is fed with key data about the desired task, such as agent action abilities, and the desired end goal to be achieved. A well-balanced allocation strategy is developed, optimizing both time efficiency and resource utilization. Our system utilizes a Natural Language Processing interface to streamline communication with construction professionals and adapt in real-time to unexpected site conditions. We concurrently use two LLM agents, specifically generator (GPT-4) and supervisor (Gemma 3/Llama 4/Mistral 7b) LLM agents to provide a more precise task schedule. We evaluate the proposed methodology using a straightforward scenario and provide metric scores to prove the efficacy of the frameworks. Our results highlight that the implementation of LLMs is crucial in construction operational tasks including robots.
>
---
#### [new 007] Dynamic Plasma Shape Control with Arbitrary Sensor Subsets
- **分类: cs.RO; eess.SY; physics.plasm-ph**

- **简介: 该论文属于等离子体控制任务，解决动态形状跟踪与传感器故障问题。提出一种强化学习代理，实现无需备用控制器的鲁棒控制，并在模拟和实验中验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.15935](https://arxiv.org/pdf/2605.15935)**

> **作者:** D. Sorokin; M. Stokolesov; A. Granovskiy; I. Prokofyev; E. Adishchev; M. Nurgaliev; E. Khayrutdinov; G. Subbotin; R. Clark; D. Orlov
>
> **摘要:** Plasma shape control in tokamaks requires a real-time controller that tracks dynamically changing shape targets while tolerating diagnostic failures. Classical approaches decompose the problem into equilibrium reconstruction followed by a linear controller, and assume a fixed, fully operational sensor set. We present a reinforcement learning agent that addresses both limitations simultaneously. The agent is trained in NSFsim, a high-fidelity tokamak simulator configured for DIII-D, on a curated dataset of 120 experimental plasma shapes. The shape targets are resampled as random step changes every 0.25 s, exposing the agent to diverse transitions across the full shape envelope. At test time the agent zero-shot tracks dynamic shape sequences; on a held-out static configuration in simulation it achieves a mean shape error of 2.01 cm, and dynamic trajectory following is demonstrated qualitatively in simulation and on the physical device. Diagnostic dropout randomly masks 30% of magnetic sensors per episode, yielding a single policy robust to arbitrary sensor subsets without backup controllers or mode-switching logic. An asymmetric actor-critic architecture with privileged equilibrium information improves value estimation under partial observability; an auxiliary shape reconstruction head on the actor enables end-to-end shape reconstruction from raw diagnostics and serves as an interpretability tool for policy analysis. The policy transfers to experimental DIII-D shots, where it directly commands the coil actuators on two dynamic shape maneuvers, and to the independent GSevolve simulator.
>
---
#### [new 008] Lamarckian Inheritance in Dynamic Environments: How Key Variables Affect Evolutionary Dynamics
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究进化机器人中拉马克遗传在动态环境中的作用，探讨其有效性受环境冲突性和可预测性影响。任务是优化机器人形态与控制器协同进化。**

- **链接: [https://arxiv.org/pdf/2605.15769](https://arxiv.org/pdf/2605.15769)**

> **作者:** K. Ege de Bruin; Kyrre Glette; Kai Olav Ellefsen
>
> **摘要:** The co-optimization of a robot's body and brain presents a coupled challenge: the morphology constrains which control strategies are effective, while the control determines how well the morphology performs. To address this, we combine morphology optimization as evolution with controller optimization as lifetime learning, utilizing Lamarckian inheritance to transfer learned controller parameters from parent to offspring. In dynamic environments, existing literature presents conflicting evidence: while traditional evolutionary theory often suggests Lamarckian inheritance lacks benefit, recent studies in evolutionary robotics indicate it can improve performance. We hypothesize that this is because previous works have not included all relevant variables with dynamic environments. In this work, we show that the benefit of Lamarckian inheritance depends on two variables: how conflicting the environmental changes are to robot control, and the predictability of those changes for the robotic agent. Using virtual soft robots and two different learning approaches, Bayesian optimization and reinforcement learning, we show that Lamarckian inheritance only underperforms Darwinian inheritance when the changes are both conflicting and unpredictable. We find that adding a sensor to detect environmental changes restores the benefits for Lamarckian inheritance in conflicting environments, by allowing robotic agents to predict the need for a different behavior, thereby generalizing their control.
>
---
#### [new 009] WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出WorldVLN，解决空中视觉语言导航任务中的动作预测问题。通过自回归模型预测环境状态变化并生成导航指令，提升导航成功率。**

- **链接: [https://arxiv.org/pdf/2605.15964](https://arxiv.org/pdf/2605.15964)**

> **作者:** Baining Zhao; Jiacheng Xu; Weicheng Feng; Xin Zhang; Zhaolu Wang; Haoyang Wang; Shilong Ji; Ziyou Wang; Jianjie Fang; Zhiheng Zheng; Weichen Zhang; Yu Shang; Wei Wu; Chen Gao; Xinlei Chen; Yong Li
>
> **摘要:** Aerial vision-language navigation (VLN) requires agents to follow natural-language instructions through closed-loop perception and action in 3D environments. We argue that aerial VLN can be formulated as a prediction-driven world-action problem: the agent should anticipate latent world evolution and act according to the predicted consequences. To this end, we propose WorldVLN, the first autoregressive world action model for aerial VLN. Unlike full-sequence video-generation world models that generate an entire visual clip, WorldVLN adapts a latent autoregressive video backbone to predict short-horizon world-state transitions and directly decodes them into executable waypoint actions. After each action segment is executed, newly received observations are encoded back into the autoregressive context, enabling closed-loop world-action prediction. We further introduce a two-stage training framework that first grounds the video prior in instruction-conditioned navigation dynamics and then develops Action-aware GRPO, the first reinforcement learning method tailored to autoregressive WAMs, to optimize waypoint decisions through their downstream rollout consequences. On public outdoor and indoor benchmarks, WorldVLN consistently outperforms existing Vision-Language-Action baselines with 12\%+ success-rate gains and larger advantages on challenging cases. It further transfers zero-shot to real drone deployment, suggesting that the proposed WorldVLN offers a promising route for spatial action tasks. Demos and code are available at this https URL.
>
---
#### [new 010] FLASH: Efficient Visuomotor Policy via Sparse Sampling
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出FLASH策略，解决机器人视觉-运动控制中的高延迟问题。通过连续多项式轨迹表示替代离散动作，提升推理速度与控制精度。**

- **链接: [https://arxiv.org/pdf/2605.15492](https://arxiv.org/pdf/2605.15492)**

> **作者:** Jiaqi Bai; Jindou Jia; Yuxuan Hu; Gen Li; Xiangyu Chen; Tuo An; Kuangji Zuo; Jianfei Yang
>
> **备注:** 19 pages, 10 figures
>
> **摘要:** Generative models such as diffusion and flow matching have become dominant paradigms for visuomotor policy learning, yet their reliance on iterative denoising incurs high inference latency incompatible with real-time robotic control. We present Fast Legendre-polynomial Action policy via Sparse History-anchored flow (FLASH Policy), which replaces discrete action-chunk generation with continuous Legendre polynomial trajectory representation. Specifically, by fitting expert demonstrations under sparse temporal sampling, FLASH enables a single inference to cover a significantly extended action horizon. To further accelerate generation, FLASH initiates the flow matching process from history polynomial coefficients rather than uninformative Gaussian noise, shortening the transport distance and enabling accurate single-step inference. Moreover, analytic polynomial differentiation directly provides desired velocity feed-forward signals to the torque controller without numerical approximation. Extensive experiments on five simulated and two real-world manipulation tasks demonstrate that FLASH achieves state-of-the-art success rates ($\ge 92\%$ across all tasks), a per-episode inference time of $31.40\,ms$ (up to $175\times$ faster than diffusion policies and $18\times$ faster than prior flow matching policies), up to $4\times$ faster training convergence than ACT, and $5\times$ to $7\times$ reduction in controller tracking error compared to discrete-action baselines.
>
---
#### [new 011] Where to Perch in a Tree: Vision-Guidance for Tree-Grasping Drones
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于目标定位任务，旨在解决树栖无人机的精准停靠问题。通过视觉算法评估树枝适合停靠的条件，如宽度、坡度和曲率，以提高无人机自主停靠能力。**

- **链接: [https://arxiv.org/pdf/2605.15430](https://arxiv.org/pdf/2605.15430)**

> **作者:** Alex Dunnett; Leonie Bottomley; Mirko Kovac; Basaran Bahadir Kocer
>
> **备注:** Work in progress version accepted to the Recent Advances in Robotic Perception for Forestry
>
> **摘要:** This study demonstrates a method to locate an ideal perch location on a tree for vision-guided autonomous tree-perching drones. Various image processing algorithms, including those used for machine learning, image segmentation and binary image morphology, are implemented to assess the shape and structure of a tree. Rather than identifying the closest available branch, this study builds on vision methods by evaluating the potential of each branch, determining its suitability for perching based on factors such as branch width, slope (angle to the horizontal) and curvature. For a given tree-perching drone and a dataset of more than 10,000 urban tree images taken from February to October in a subtropical and temperate monsoon climate, the proposed method successfully produces a result for 76% of feasible targets. A feasible target defined as a tree where the branch diameters are sufficiently thick and where the available perching space is at least equal to the width of a tendon-driven grasping claw. These successful preliminary results create a foundation from which a number of identified improvements and additional features can be developed to create a generalised method; this will involve the incorporation of supplementary data from depth perception and attitude sensors to enhance the branch assessment.
>
---
#### [new 012] KaRMA: A Kinematic Metric for Fine Manipulation Ability in Robotic Hands
- **分类: cs.RO**

- **简介: 该论文提出KaRMA，用于评估机器人手的精细操作能力，解决传统静态指标无法准确衡量动态操作的问题。通过分析滚动运动，量化物体位移和旋转范围。**

- **链接: [https://arxiv.org/pdf/2605.15548](https://arxiv.org/pdf/2605.15548)**

> **作者:** Martin Peticco; Pulkit Agrawal
>
> **摘要:** Traditional robotic hand metrics focus on static properties such as workspace, manipulability, and grasp stability. However, these metrics do not directly measure dexterity under the standard definition in robotic manipulation: the ability to continuously change an object's pose within the hand while maintaining contact from an initial grasp. We introduce Kinematic Rolling Manipulation Ability (KaRMA), a kinematic-only metric for fine manipulation that quantifies reachable in-hand translation and reorientation of a spherical test object within a two-finger precision pinch through feasible rolling motions. KaRMA enforces joint limits, collision constraints, rolling contact, and antipodal force feasibility, then investigates reachable in-hand object poses via breadth-first search over translation and rotation primitives. KaRMA reports three scores: translational coverage (KaRMA-T), rotational coverage (KaRMA-R), and sensitivity to the initial grasp (KaRMA-S). We evaluate KaRMA on 16 widely used robotic hands and compare against static baselines, showing that KaRMA separates hands that rank identically under static proxies, reveals translation-rotation tradeoffs invisible to existing baselines, and is qualitatively consistent with selected published task benchmarks where Jacobian-based metrics can be misleading.
>
---
#### [new 013] Wind-Aware Optimal Trajectory Planning for Efficient Gliding of Fixed-Wing Aerial Systems
- **分类: cs.RO**

- **简介: 该论文属于无人机轨迹规划任务，解决固定翼无人机在风扰和障碍物下的高效滑翔问题。通过非线性多成本轨迹规划方法，生成平滑轨迹并实现能量平衡控制。**

- **链接: [https://arxiv.org/pdf/2605.15619](https://arxiv.org/pdf/2605.15619)**

> **作者:** Luca Morando; Nishanth Bobbili; Giuseppe Loianno
>
> **备注:** Accepted for publication at IEEE International Conference on Robotics and Automation (ICRA 2026) held in Vienna
>
> **摘要:** Gliding offers small fixed-wing UAVs extended endurance and silent operation but requires accurate energy management, especially under wind disturbances and obstacle constraints. Traditional Total Energy Control Systems based controllers regulate the trade between potential and kinetic energy reactively, often requiring fine-tuning and trim-conditions knowledge. In this work, we shift the regulation to the planning level and present a nonlinear, multi-cost trajectory planner for small UAV gliders. The method generates $\mathcal{C}^3$ continuous trajectories based on Bernstein polynomials, mapped into control commands through differential flatness, and re-planned online to match experimentally derived sink polar curves. A simulated netto variometer is integrated into the optimization to estimate air mass motion, constraining the glide to energy-balanced states. Consecutive gliding trajectories are linked by cruising segments computed through trajectories initialized on Dubins path-based waypoints, enabling hybrid missions that combine powered and unpowered flight. The approach is validated in CFD simulations and real-world experiments with a fixed-wing platform, showing reliable stabilization of sink rate, airspeed, and glide ratio under wind gusts and in presence of obstacles.
>
---
#### [new 014] Adaptive Outer-Loop Control of Quadrotors via Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于无人机控制任务，解决DRL在真实环境中性能不足的问题。通过引入自适应控制架构和残差动力预测器，提升无人机对动态扰动的适应能力。**

- **链接: [https://arxiv.org/pdf/2605.16015](https://arxiv.org/pdf/2605.16015)**

> **作者:** Vishnu Saj; Sushi Vemuri; Dileep Kalathil; Moble Benedict
>
> **摘要:** Deep Reinforcement Learning (DRL) for quadrotor flight control typically relies on Domain Randomization (DR) for sim-to-real transfer, resulting in overly conservative policies that struggle with dynamic disturbances. To overcome this, we propose a novel adaptive control architecture that actively perceives and reacts to instantaneous perturbations. First, we train an optimal outer-loop policy, then replace its reliance on ground-truth disturbance data with a Residual Dynamics Predictor (RDP). The RDP estimates the external forces and moments acting on the aircraft in flight online using only the history of states and control actions. For seamless hardware transfer, we introduce a data-efficient linear calibration bridge and an online thrust correction mechanism that align the simulated latent space with reality using mere seconds of flight data. Real-world validations on a Crazyflie micro-quadrotor demonstrate that our adaptive controller significantly outperforms baselines, maintaining precise trajectory tracking under severe uncertainties including mass variations, asymmetric payloads, and dynamic slung loads
>
---
#### [new 015] Learning Dynamic Pick-and-Place for a Legged Manipulator
- **分类: cs.RO; cs.AI**

- **简介: 论文研究四足机械臂的动态抓取与放置任务，解决协调运动与精准操作难题。提出分层强化学习框架，实现重载和大范围作业的自适应控制。**

- **链接: [https://arxiv.org/pdf/2605.15713](https://arxiv.org/pdf/2605.15713)**

> **作者:** Moonkyu Jung; Jiseong Lee; Zhengmao He; Donghoon Youm; Juhyeok Mun; HyeongJun Kim; Hyunsik Oh; Donghyuk Choi; Jungwoo Hur; Jie Song; Jemin Hwangbo
>
> **备注:** Accepted to IEEE Robotics and Automation Letters 2026
>
> **摘要:** Legged manipulators extend robotic capabilities beyond static manipulation by integrating agile locomotion with versatile arm control. However, achieving precise manipulation while maintaining coordinated locomotion remains a major challenge. This work presents a hierarchical reinforcement learning framework for dynamic pick-and-place tasks using a quadruped equipped with a 6-DOF robotic arm. The framework incorporates an explicit mass estimation module enabling adaptive whole-body control for objects with varying weights. In simulation, the system achieves an 86.05% success rate with payloads up to 2.3 kg. The approach is further validated through real-world experiments across six representative scenarios with controlled variations in object physical properties (size and mass) and task heights. Specifically, within a wide vertical workspace ranging from ground level to 1.1~m-high tabletops, the system demonstrates an average success rate of 73.3% for payloads up to 1.3 kg, with an average execution time of 4.06 s. Unlike prior works that handle lightweight objects and execute pick-and-place motions with slow, piecewise motions, the proposed framework exploits concurrent locomotion and manipulation for dynamic, continuous execution. These results demonstrate the potential of quadrupedal mobile manipulators for adaptive, whole-body pick-and-place with heavier payloads and extended workspaces.
>
---
#### [new 016] PCASim: Promptable Closed-loop Adversarial Simulation for Urban Traffic Environment
- **分类: cs.RO**

- **简介: 论文提出PCASim，用于生成城市交通中的对抗性场景，提升自动驾驶安全性。任务是增强测试效果，解决传统方法不足，通过结合知识库与大模型生成定制化场景，并利用强化学习提高场景多样性与真实感。**

- **链接: [https://arxiv.org/pdf/2605.15654](https://arxiv.org/pdf/2605.15654)**

> **作者:** Chuancheng Zhang; Zhenhao Wang; Kaizheng Li; Yaran Lin; Qiang Guo; Bin Jiang
>
> **摘要:** Real-world autonomous driving, particularly in urban environments with numerous corner cases, requires rigorous testing to ensure product safety and robustness. However, few studies have explored integrating adversarial scenario generation with the training of safety agents in closed-loop testing, enabling efficient co-evolution and mutual enhancement of both. To address this challenge, an adversarial behavior knowledge repository is constructed by applying rule-based filtering to an open-source dataset, combined with knowledge retrieval modules tailored for simulation environments. A large language model (LLM) is employed to integrate knowledge-, data-, and adversarial-driven approaches, generating safety-critical traffic scenarios customized to user needs. Additionally, while evaluating the generated scenarios, we employ reinforcement learning models to train the behaviors of different types of vehicles, thereby enriching scenario diversity beyond existing datasets while preserving realism. Experimental results demonstrate that the proposed framework improves the accuracy of domain-specific language generation by 12\%. Moreover, the success rate of newly generated scenario transformations increases by 8\%, while obstacle-avoidance capability is enhanced by 30\%. For the complete manuscript, please refer to: this https URL
>
---
#### [new 017] Towards Trustworthy and Explainable AI for Perception Models: From Concept to Prototype Vehicle Deployment
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶感知任务，旨在解决深度神经网络的不透明性问题。提出可信AI感知模块，集成可解释性和不确定性估计，提升模型可靠性与安全性。**

- **链接: [https://arxiv.org/pdf/2605.16087](https://arxiv.org/pdf/2605.16087)**

> **作者:** Till Beemelmanns; Shayan Sharifi; Manas Mehrotra; Ayushman Choudhuri; Lutz Eckstein
>
> **备注:** Accepted for publication at IEEE ITSC 2026
>
> **摘要:** Deep Neural Networks have become the dominant solution for Autonomous Driving perception, but their opacity conflicts with emerging Trustworthy AI guidelines and complicates safety assurance, debugging, and human oversight. While theoretical frameworks for safe and Explainable AI (XAI) exist, concrete implementations of Trustworthy AI for 3D scene understanding remain scarce. We address this gap by proposing a Trustworthy AI perception module that is remarkably robust, integrates faithful explainability, and calibrated uncertainty estimates. Building on a transformer-based detector, we derive explanation from the attention mechanism at inference time and validate their faithfulness using perturbation-based consistency tests. We further integrate an uncertainty estimation and calibration module, and apply robustness-enhancing training methods. Experiments show faithful saliency behavior, improved robustness, and well-calibrated uncertainty estimates. Finally, we deploy these Trustworthy AI elements in a prototype vehicle and provide an XAI Interface that visualizes documentation artifacts, model uncertainty state, and saliency maps, demonstrating the feasibility of trustworthy perception monitoring in real time. Supplementary materials are available at this https URL .
>
---
#### [new 018] Reactive Robot-Centric Safety for Autonomous Navigation in Constrained and Dynamic Environments
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自主导航任务，解决动态环境中实时安全问题。通过集成3D LIDAR的CBF安全过滤器，实现碰撞避免与任务执行的平衡。**

- **链接: [https://arxiv.org/pdf/2605.15782](https://arxiv.org/pdf/2605.15782)**

> **作者:** Viswa Narayanan Sankaranarayanan; Vignesh K. Viswanathan; Akshit Saradagi; Sumeet Satpute; George Nikolakopoulos
>
> **备注:** 9 pages, 12 figures, currently under review
>
> **摘要:** In this work, we address the problem of ensuring real-time safety in autonomous robot navigation, in spatially constrained dynamic environments, by utilizing only onboard sensors. We present a real-time control architecture that integrates a 3D LIDAR perception-based composite control barrier function(CBF)-based safety filter directly into the autonomy pipeline. The proposed perception-driven framework enforces collision avoidance constraints dynamically from onboard point cloud data, thus allowing a large number of constraints to be handled at the control frequency, while remaining minimally invasive to nominal task execution. The safety region is defined as an ellipsoid in the body-frame, consistent with the geometry of the platform, which induces time-varying constraints in the world frame as the robot rotates; this effect is handled through a dedicated formulation of time-varying (CBF) for each LIDAR point. We validate the system through multiple field experiments in underground environments by utilizing a quadruped platform performing a visual inspection task, demonstrating reliable operation in the presence of dynamic obstacles, unsafe high-level references, abrupt localization anomalies, and while traversing through narrow corridors.
>
---
#### [new 019] Learning Sim-Grounded Policies for Bimanual Rope Manipulation from Human Teleoperation Data
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究双臂绳索操作任务，解决从人类遥控数据中学习有效策略的问题。通过比较视觉与状态基策略，发现状态空间更有利于泛化，提升数据效率。**

- **链接: [https://arxiv.org/pdf/2605.16043](https://arxiv.org/pdf/2605.16043)**

> **作者:** Gina Wigginghaus; Tim Missal; Berk Guler; Simon Manschitz; Jan Peters
>
> **备注:** Accepted to the Beyond Teleoperation Workshop at ICRA 2026, 5 pages, 2 figures
>
> **摘要:** Deformable Linear Objects (DLOs) such as ropes and cables are widely encountered in both household and industrial applications, yet remain challenging to manipulate due to their infinite-dimensional configuration space and frequent self-occlusion. Imitation learning from teleoperation offers a practical path to bimanual DLO manipulation, but its scalability is limited by human effort, making the choice of observation space critical for generalization from small datasets. In this study, we investigate whether the lack of generalization in egocentric visual policies for the knot-untangling task stems from the observation space itself, rather than from the policy architecture or data scale. We compare two Action Chunking with Transformers policies trained on the same bimanual teleoperation data: a vision-based policy conditioned on two egocentric RGB streams from wrist-mounted cameras, and a state-based policy conditioned on the DLO's 3D particle state, extracted from an initial observation via multi-view fusion and evolved in a particle-based eXtended Position-Based Dynamics simulation. Evaluated open-loop on an unseen rope configuration, the state-based policy outperforms its visual counterpart with a 30.8% reduction in L1 error when predicting the initial grasp-and-pull action, quantifying the observability gap between pixels and physics-consistent state, and pointing toward more data-efficient robot learning for the DLO manipulation task from limited human demonstrations.
>
---
#### [new 020] Designing for Robot Wranglers: A Synthesis of Literature and Practice
- **分类: cs.RO; cs.HC**

- **简介: 论文探讨机器人协调员角色，分析其工作内容与挑战，提出设计建议以支持该新兴职业。属于人机交互研究，解决如何有效支持机器人协调员的问题。**

- **链接: [https://arxiv.org/pdf/2605.15892](https://arxiv.org/pdf/2605.15892)**

> **作者:** David Porfirio; Ian McDermott; Hsin-Mei Chen; Satoru Satake; Takayuki Kanda; Thomas D. LaToza
>
> **备注:** Accepted for publication in the Proceedings of ACM Designing Interactive Systems (2026)
>
> **摘要:** Robots are increasingly present in human spaces, such as for conducting deliveries in hospitals, interacting with visitors at museums, and stocking items in warehouses. To ensure the seamless integration of robots into these spaces, a new role in human-robot interaction is emerging - the robot wrangler, namely an individual who is responsible for setting up, overseeing, and troubleshooting the robot. To understand the needs of this stakeholder, we conducted a scoping review that uncovered a typology of robot wrangling across the research literature, and discovered that wrangling is an umbrella term that collapses a highly complex and heterogeneous space of activities, often rendering this labor difficult to characterize and support. To further clarify and understand robot wrangling, we then reflected on our own firsthand and imagined experiences as robot wranglers within our own respective domains. Guided by the scoping review and our reflections, we devise a series of design implications for supporting wranglers directly as individuals and as members of a wider service ecology.
>
---
#### [new 021] MyoChallenge 2025: A New Benchmark for Human Athletic Intelligence
- **分类: cs.RO**

- **简介: 该论文介绍MyoChallenge 2025，旨在评估人类运动智能，解决人工智能在运动控制上的不足，通过高保真肌肉骨骼模型与机器学习算法进行体育动作基准测试。**

- **链接: [https://arxiv.org/pdf/2605.15650](https://arxiv.org/pdf/2605.15650)**

> **作者:** Cheryl Wang; Chun Kwang Tan; Balint K. Hodossy; Eric Lyu; Jun Guo; Wentao Zhao; Huaping Liu; Chengkun Li; Merkourios Simos; Bianca Ziliotto; Alexander Mathis; Siyuan Liu; Jiahao Chen; Shanlin Zhong; Bo Jiang; Ci Song; Yaoye Zhu; Chenhui Zuo; Yanan Sui; Mohamed Irfan Refai; Massimo Sartori; Guillaume Durandau; Vikash Kumar; Vittorio Caggiano
>
> **摘要:** Athletic performance represents the pinnacle of human motor intelligence, demanding rapid choices, precise control, agility, and coordinated physical execution. Replicating this seamless combination of capabilities remains elusive in current artificial intelligence and robotic systems. Concurrently, understanding the biological mastery of these movements is hindered because complex muscle coordination is rarely measured in vivo due to the limitations of physical equipment. To bridge this fundamental gap in understanding, MyoChallenge at NeurIPS 2025 established a pioneering benchmark for motor control intelligence in sports, leveraging high-fidelity musculoskeletal models within physics simulation combined with machine learning-driven algorithms. The competition introduces two distinct tracks emphasizing either upper or lower limbs control: a table tennis rally task utilizing a biomechanic upper limb composed of an arm with a hand and a trunk; and a soccer penalty kick using a biomechanic model of legs and a trunk. Marking the fourth iteration of the MyoChallenge series, this event attracted almost 70 teams and over 560 submissions globally, uniting a diverse community ranging from physicians and neuroscientists to machine learning experts. The competition facilitated the development of several state-of-the-art control algorithms for a musculoskeletal system capable of sports agility, leveraging techniques such as physics-based motion planners, on-policy behaviour cloning, hierarchical planning, and muscle synergies. By integrating standardized tasks and physiologically realistic models into the open-source framework of MyoSuite, MyoChallenge'25 serves as a reproducible and reusable testbed to accelerate interdisciplinary research across machine learning, biomechanics, sports science, and neuroscience. Project page: this https URL.
>
---
#### [new 022] A QUBO Formulation Framework for Kinematic Structure-Based Robot Design Optimization: A Robotic Hand Case Study
- **分类: cs.RO**

- **简介: 该论文属于机器人设计优化任务，解决如何将运动学结构转化为可优化问题。通过构建QUBO模型，结合经典与量子优化方法，实现机器人手的结构优化。**

- **链接: [https://arxiv.org/pdf/2605.15510](https://arxiv.org/pdf/2605.15510)**

> **作者:** HyoJae Kang; Yeong Jae Park; Jeongdo Ahn; Dongil Park
>
> **备注:** This manuscript has been submitted for possible publication. 14 pages, 5 figures
>
> **摘要:** This paper presents a quadratic unconstrained binary optimization-based formulation framework for robot design optimization using kinematic structure-level evaluation metrics. In the proposed framework, classical computation is used to evaluate design-dependent metrics while the resulting combinatorial selection problem is formulated in a structure compatible with quantum annealing-based optimization. A robotic hand is adopted as a representative case study, as its performance is determined by both the individual kinematic characteristics of each finger and interaction terms. The proposed formulation incorporates individual design rewards, overlap workspace interactions, one-hot constraint, and structural dependency penalties into a unified quadratic model. A 27-variable robotic hand design problem is constructed, and simulated annealing is used as a classical baseline to verify the feasibility of the formulation. Quantum annealing is further performed to examine the applicability of the proposed formulation to annealing-based hardware execution. The results show that feasible design combinations satisfying both one-hot selection and pairwise constraints can be obtained, with the observed objective-value range becoming narrower as the number of reads increases. In addition, the formulation process is discussed for other robotic systems. The proposed framework provides a generalized approach for transforming kinematic structure-based robot design problems into combinatorial optimization problems.
>
---
#### [new 023] PhysBrain 1.0 Technical Report
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在提升机器人物理理解。通过将人类视角视频转为物理常识监督，增强机器人泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.15298](https://arxiv.org/pdf/2605.15298)**

> **作者:** Shijie Lian; Bin Yu; Xiaopeng Lin; Changti Wu; Hang Yuan; Xiaolin Hu; Zhaolong Shen; Yuzhuo Miao; Haishan Liu; Yuxuan Tian; Yukun Shi; Cong Huang; Kai Chen
>
> **备注:** Project Page: this https URL
>
> **摘要:** Vision-language-action models have advanced rapidly, but robot trajectories alone provide limited coverage for learning broad physical understanding. PhysBrain 1.0 studies a complementary route: converting large-scale human egocentric video into structured physical commonsense supervision before robot adaptation. Our data engine extracts scene elements, spatial dynamics, action execution, and depth-aware relations, then turns them into question-answer supervision for training PhysBrain VLMs. The resulting physical priors are further transferred to VLA policies through a capability-preserving and language-sensitive adaptation design. Across multimodal QA benchmarks and embodied control benchmarks, including ERQA, PhysBench, SimplerEnv-WidowX, LIBERO, and RoboCasa, PhysBrain 1.0 achieves SOTA results and shows especially strong out-of-domain performance on SimplerEnv. These results suggest that scaling physical commonsense from human interaction video can provide an effective bridge from multimodal understanding to robot action.
>
---
#### [new 024] OHP-RL: Online Human Preference as Guidance in Reinforcement Learning for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文提出OHP-RL框架，用于机器人操作中的强化学习，解决人类干预信息利用不足的问题，通过偏好引导提升学习效率与安全性。**

- **链接: [https://arxiv.org/pdf/2605.15971](https://arxiv.org/pdf/2605.15971)**

> **作者:** Yunyang Mo; Jian Li; Qiwei Wu; Yihang Kang; Renjing Xu
>
> **摘要:** While reinforcement learning (RL) enables robots to acquire skills autonomously, its real-world deployment is severely limited by inefficient and unsafe exploration. Human-in-the-loop interventions offer a practical solution, yet existing methods typically exploit these interventions as auxiliary training signals, without fully capturing the richer information they provide about when and how autonomy should be guided. Human interventions often encode relative preferences over behavior under safety and task constraints, rather than prescribing exact actions to imitate. Motivated by this perspective, we propose Online Human Preference as Guidance in Reinforcement Learning (OHP-RL), a framework that leverages human interventions as preference information to guide policy learning. OHP-RL introduces a state-dependent preference gate that adaptively regulates when and to what extent human interventions should shape policy learning. This design enables the agent to benefit from intermittent and imperfect human feedback while preserving autonomous exploration and stable policy optimization. We evaluate OHP-RL on three challenging real-world contact-rich manipulation tasks on a Franka robot. Across all tasks, OHP-RL consistently achieves strong success rates, faster convergence, and substantially lower human intervention effort than prior approaches. Moreover, the learned policies exhibit more stable and human-aligned behavior throughout training.
>
---
#### [new 025] A Topology-Aware Spatiotemporal Handover Framework for Continuous Multi-UAV Tracking
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多无人机持续跟踪任务，解决轨迹碎片化导致的车辆身份丢失问题。提出一种基于拓扑的时空切换机制，实现全局身份一致跟踪。**

- **链接: [https://arxiv.org/pdf/2605.15779](https://arxiv.org/pdf/2605.15779)**

> **作者:** Jianlin Ye; Christos Kyrkou; Panayiotis Kolios
>
> **摘要:** The integration of Unmanned Aerial Vehicles(UAVs) into Intelligent Transportation Systems (ITS) offers synoptic visibility for traffic monitoring, yet scalable deployment is hindered by trajectory fragmentation, where vehicle identity persistence is lost across multi-UAV Fields of View (FOV). While state-of-the-art frameworks excel in optimizing local trajectory extraction and stability for single-drone imagery, they often function as isolated data silos that generate disjointed trajectories, thereby precluding network-level analysis such as Origin-Destination estimation. This paper presents a real-time Multi-Camera Multi-Vehicle Tracking (MCMT) system designed to handle global identity persistence. Addressing the visual ambiguity and computational cost of appearance-based Re-Identification (Re-ID) in nadir views, we introduce a lightweight Topology-Based Spatiotemporal Handover mechanism. We implement a high-throughput parallel pipeline leveraging YOLO11 and ByteTrack to process concurrent 4K streams. Our core contribution is a deterministic queue-based matching algorithm that utilizes geometric overlaps and virtual lane discretization to predictively manage identity handover via FIFO queues. Experimental results on complex urban environments, including intersections and merging traffic, demonstrate a Handover Success Rate (HOSR) of 99.8% in continuous traffic flows, significantly outperforming Re-ID baselines (74.1%) while validating edge deployment feasibility. The source code is available at this https URL.
>
---
#### [new 026] SkiP: When to Skip and When to Refine for Efficient Robot Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对机器人操作任务，解决动作预测效率问题。提出SkiP方法，通过跳过冗余步骤、重点优化关键步骤，提升效率并保持成功率。**

- **链接: [https://arxiv.org/pdf/2605.15536](https://arxiv.org/pdf/2605.15536)**

> **作者:** Mingtong Dai; Guanqi Peng; Yongjie Bai; Feng Yan; Chunjie Chen; Lingbo Liu; Liang Lin; Xinyu Wu
>
> **摘要:** Previous imitation learning policies predict future actions at every control step, whether in smooth motion phases or precise, contact-rich operation phases. This uniform treatment is wasteful: most steps in a manipulation trajectory traverse free space and carry little task-relevant information, while a small fraction of \emph{key} steps around contacts, grasps, and alignment demand dense, high-resolution prediction. We propose a novel \emph{action relabeling} mechanism: at each timestep in a skip segment, we replace the behavior cloning target with the action at the entrance of the next key segment, enabling the policy to leap over redundant steps in a single decision. The resulting \textbf{Skip Policy (SkiP)} dynamically leaps over skip segments and intensively refines actions in key segments, within a single unified network requiring no learned skip planner or hierarchical structure. To automatically partition demonstrations into key and skip segments without manual annotation, we introduce \emph{Motion Spectrum Keying} (MSK), a fast, task-agnostic procedure that detects local motion complexity from action signals. Extensive experiments across 72 simulated manipulation tasks and three real-robot tasks show that SkiP reduces executed steps by $15$--$40\%$ while matching or improving success rates across various policy backbones. Project page: \texttt{this https URL}.
>
---
#### [new 027] Beyond Collision Avoidance: Multi-Robot Yielding and Spatial Affordance in Emergency Evacuations
- **分类: cs.RO**

- **简介: 该论文属于多机器人协同任务，解决紧急疏散中机器人行为安全问题。通过实验分析不同避让策略，提出基于环境感知的语义导航方法。**

- **链接: [https://arxiv.org/pdf/2605.16115](https://arxiv.org/pdf/2605.16115)**

> **作者:** Ning Zhou; Edmund R. Hunt; Nikolai W.F. Bode
>
> **摘要:** As mobile service robots increasingly coexist with pedestrians, ensuring passively safe behaviour during confined emergency evacuations is critical. Existing multi-robot yielding strategies often focus solely on collision avoidance and macroscopic flow optimisation, overlooking environmental affordances and human spatial expectations. To bridge the gap between macroscopic theory and micro-level perception, we conducted a game-based virtual evacuation experiment (N=56). We investigated individual psychological responses to four multi-robot yielding strategies (Hide, LineEscape, Freeze, ShortestPath) across confined corridors with and without refuge niches. Our results establish a robust preference hierarchy (Hide > LineEscape > Freeze > ShortestPath), demonstrating that proactive space-yielding significantly outperforms freezing and efficiency-first approaches. Crucially, we found that environmental affordances heavily shape cognitive expectations. Actively utilising available niches amplifies the psychological comfort of proactive yielding (Hide). Conversely, failing to use an obvious niche (e.g., executing LineEscape) may trigger Expectation Violation. This is reflected in a drastically increased perceived cognitive delay, despite objectively unimpeded trajectories. Furthermore, prior robot interaction experience helps users decode complex social intents. Ultimately, this research demonstrates that safe human-robot interaction during emergencies must evolve from pure trajectory optimisation to semantically aware navigation. Future work will extend this framework to investigate complex interactions between robot swarms and pedestrian crowds.
>
---
#### [new 028] LAPS: Improving Incremental LiDAR Mapping using Active Pooling and Sampling for Neural Distance Fields
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于增量LiDAR建图任务，解决神经距离场在线优化中的灾难性遗忘问题。提出LAPS框架，通过主动采样和池化提升重建完整性和效率。**

- **链接: [https://arxiv.org/pdf/2605.15496](https://arxiv.org/pdf/2605.15496)**

> **作者:** Dongjae Lee; Wooseong Yang; Yifu Tao; Maurice Fallon; Ayoung Kim
>
> **备注:** accepted at RA-L 2026
>
> **摘要:** Neural distance fields offer a compact and continuous representation of 3D geometry, making them attractive for incremental LiDAR mapping. However, their online optimization is vulnerable to catastrophic forgetting, where new observations can degrade previously reconstructed geometry. Replay-based training is commonly used to address this issue, but existing methods typically rely on passive replay buffers and uniform sampling, which can waste memory on redundant observations and under-train poorly constrained regions. We propose LAPS, a replay management framework for incremental neural mapping that improves both replay retention and replay allocation during online updates. LAPS combines reliability-based active pooling to retain reliable historical samples under limited memory with uncertainty-guided active sampling to focus optimization on under-constrained regions. Experiments on synthetic and real-world benchmarks show that LAPS consistently improves reconstruction completeness while maintaining competitive geometric accuracy. On Oxford Spires, it improves recall by 4.66 pp and F1-score by 3.79 pp over PIN-SLAM on the Blenheim Palace 05 sequence. We release our open source implementation at: this https URL.
>
---
#### [new 029] Residual Reinforcement Learning for Robot Teleoperation under Stochastic Delays
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人遥操作任务，解决随机延迟导致的控制不稳定问题。通过融合LSTM和残差强化学习，提升控制性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.15480](https://arxiv.org/pdf/2605.15480)**

> **作者:** Kaize Deng; Zewen Yang
>
> **备注:** Accepted at 23rd IFAC World Congress 2026
>
> **摘要:** Stochastic communication delays in teleoperation introduce signal discontinuities that undermine control stability and degrade control performance. Consequently, the conventional reinforcement learning (RL) methods struggle with the delayed observations due to the delay-induced observations, leading to high-frequency chattering. To address this, we propose a hybrid control framework, delay-resilient RL, integrating a state estimator utilizing Long Short-Term Memory (LSTM) with a residual RL policy, which is resilient to stochastic delays. The LSTM reconstructs smooth, continuous state estimates from delayed observations, enabling the RL agent to learn a residual torque compensation policy that balances tracking accuracy with velocity smoothness. Experimental validation on Franka Panda robots demonstrates that our approach significantly outperforms the state-of-the-art baselines, ensuring robust and stable teleoperation even under high-variance stochastic delays.
>
---
#### [new 030] Task-Semantic Graph-Driven Distributed Agent Networking for Underwater Target Tracking
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于水下目标跟踪任务，解决MARL算法在AUV swarm中的评估与性能问题，提出STG-MAPPO算法及开源平台。**

- **链接: [https://arxiv.org/pdf/2605.15528](https://arxiv.org/pdf/2605.15528)**

> **作者:** Shengchao Zhu; Guangjie Han; Chuan Lin; Yu He
>
> **摘要:** Autonomous underwater vehicle (AUV) swarms are emerging as intelligent underwater networks, where each node must sense, communicate, process local data, and make decisions under severe acoustic constraints. Persistent underwater target tracking is a typical task with moving targets, changing communication topology, intermittent acoustic links, and limited observation for each AUV. Multi-agent reinforcement learning (MARL) is a natural candidate for distributed tracking, yet existing studies still lack a unified open-source platform for evaluating different MARL algorithms under six-degree-of-freedom AUV dynamics. In addition, policies trained with raw geometric states and low-level force actions often struggle to represent task phases, observation reliability, link quality, and local cooperation roles. This paper addresses these issues by developing an open-source MARL-AUV platform that integrates DI-engine with a six-degree-of-freedom underwater AUV target-tracking simulator. To the best of our knowledge, it is the first open platform that connects a public MARL training framework with physically modeled AUV swarm-based tasks, and provides a unified experimental protocol for fair training, testing, and comparison of representative RL and MARL algorithms. Based on this platform, we propose STG-MAPPO, a Semantic Task Graph-enhanced variant of Multi-Agent Proximal Policy Optimization. STG-MAPPO builds semantic policy inputs from tracking diagnostics, task phases, observation confidence, link availability, neighbor tracking quality, and local role advantage. A compact semantic task graph links communication-constrained network states to decentralized actor decisions, and a velocity-level action abstraction maps high-level cooperative decisions to executable six-degree-offreedom AUV control this http URL code is available at this https URL.
>
---
#### [new 031] Constrained MPC-Based Motion Planning for Morphing Quadrotors in Ultra-Narrow Passages under Limited Perception
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人运动规划任务，解决在感知受限环境下morphing quadrotors通过狭窄通道的问题。提出一种新型障碍物代价函数，提升导航效率与安全性。**

- **链接: [https://arxiv.org/pdf/2605.15999](https://arxiv.org/pdf/2605.15999)**

> **作者:** Harsh Modi; Xiao Liang; Minghui Zheng
>
> **摘要:** This paper introduces a motion planning framework to plan morphology and trajectory for morphing quadrotors under extremely constrained environments. We develop a novel obstacle avoidance cost function for nonlinear model predictive control (MPC) that enables navigation through extremely narrow gaps under limited perception from a 2D LiDAR. Classical artificial potential field-based costs typically have a high cost in narrow passages, artificially blocking the navigable path. In contrast, we propose a smooth exponential obstacle cost that preserves low traversal cost within narrow gaps while maintaining strong collision avoidance behavior. The formulation avoids hard activation thresholds and introduces a cost reduction factor to reduce the cost within narrow passages. Direct use of 2D LiDAR measurements in MPC allows navigation around arbitrarily shaped obstacles. The method is embedded within an acados-based nonlinear MPC framework. Simulation and experimental results demonstrate successful traversal of narrow corridors where typical repulsive cost functions would fail. The approach provides a computationally efficient and practical solution for navigating through tight spaces while maintaining safety from the obstacles. While we are implementing the framework on the morphing quadrotors, the cost function formulation is general-purpose for any mobile robot application, and is not limited to the morphing quadrotors. The implementation code is available at \href{this https URL}{Github Repo} and a short video is available at \href{this https URL}{Video Link}.
>
---
#### [new 032] A Reproducible and Physically Feasible Dynamic Parameter Identification Framework for a Low-Cost Robot Arm
- **分类: cs.RO**

- **简介: 该论文属于机器人动力学参数辨识任务，解决低成本机械臂动态模型的可重复性和物理可行性问题。通过简化模型、设计识别运动并结合优化方法，获得高精度且物理合理的动力学参数。**

- **链接: [https://arxiv.org/pdf/2605.15949](https://arxiv.org/pdf/2605.15949)**

> **作者:** Junji Oaki; Koki Yamane; Koki Inami; Sho Sakaino
>
> **备注:** 11 pages, 8 figures, 7 tables, and 1 algorithm
>
> **摘要:** This paper presents a reproducible and physically feasible dynamic parameter identification framework for CRANE-X7, a low-cost robot arm driven by modular smart actuators. To improve practical identifiability, products of inertia are removed according to approximate link symmetry, reducing the rigid-body model from 65 to 39 base parameters. Identification motions are hand-designed from structured single-joint and adjacent-joint primitives under practical joint-range limits. The proposed pipeline combines preprocessing, inverse-dynamics-regressor-based ordinary least squares (OLS), conditional semidefinite-programming (SDP) projection for feasibility recovery, and closed-loop input error (CLIE) refinement. Candidate solutions from 40 structured trajectories are analyzed in a common PCA space to select a statistically central representative model. Because statistical centrality alone does not ensure physical acceptability, the selected model is finally screened by an all-pose positive-definiteness audit of the inertia matrix and, when necessary, corrected by a localized post-CLIE SDP rescue step. Experiments show that the parameter cloud becomes progressively more concentrated from OLS to SDP and CLIE, while the final accepted model preserves high predictive accuracy on held-out validation motions. These results demonstrate a practical route to statistically coherent and physically feasible dynamic models for low-cost robot platforms.
>
---
#### [new 033] Structured Jacobian Construction for Motion Optimization with High-Order Time Derivatives in Multi-Link Systems
- **分类: cs.RO**

- **简介: 该论文属于运动优化任务，解决多体系统中高阶时间导数的雅可比计算问题。提出结构化雅可比方法，提升计算效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.15845](https://arxiv.org/pdf/2605.15845)**

> **作者:** Taiki Ishigaki; Ko Ayusawa; Eiichi Yoshida
>
> **摘要:** This paper presents a novel framework for Jacobian computation in motion optimization problems involving multi-link systems, where physical quantities are represented using higher-order time derivatives. In motion optimization of robots and humans, cost functions may incorporate higher-order time derivatives, such as jerk or the time variation of forces, to capture smoothness and perceptual characteristics, particularly in motion skill analysis and expressive behaviors, thereby necessitating Jacobian computations involving these quantities. However, such Jacobians are typically computed using numerical or automatic differentiation without explicitly exploiting the underlying multi-link structure, which can lead to increased computational cost and numerical instability. To address this limitation, we propose a structured Jacobian formulation for motion optimization, based on the comprehensive motion computation framework, in which physical quantities and their higher-order time derivatives are systematically represented along the multi-link structure. The proposed method systematically derives analytical expressions for Jacobians of kinematic and dynamic quantities, including momentum, forces, and joint torques, with respect to generalized coordinates and their higher-order derivatives. The resulting framework is applicable to both direct and inverse optimization. Through numerical experiments, we demonstrate that the proposed method improves computational efficiency compared to numerical and automatic differentiation, while achieving comparable accuracy. Furthermore, we demonstrate its effectiveness in inverse optimization by recovering cost function weights from motion data. Together, these results indicate that the proposed formulation provides a scalable and structured computational foundation for motion optimization involving higher-order time derivatives in multi-link systems.
>
---
#### [new 034] Terrain Consistent Reference-Guided RL for Humanoid Navigation Autonomy
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于人形机器人自主导航任务，解决如何在复杂地形中实现精确参考轨迹跟踪的问题。通过调整参考轨迹以适应地形，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2605.15517](https://arxiv.org/pdf/2605.15517)**

> **作者:** William D. Compton; Zachary Olkin; Aaron D. Ames
>
> **备注:** 8 pages, 4 figures, intended to submit to Humanoids 2026
>
> **摘要:** We present a method for training reference-guided, perceptive reinforcement learning locomotion policies for humanoid robots in which reference trajectories are modulated in training to be consistent with terrain geometry. Aiming to deploy our method with standard navigation autonomy infrastructure, we synthesize SE(2)-controllable reference trajectories inside the RL training loop, projecting desired footsteps onto valid footholds and adjusting swing-foot and center-of-mass trajectories to match the terrain. The resulting policy exposes a clean SE(2) velocity interface compatible with standard navigation planners. In simulation, environmentally-conditioned references significantly improve reference tracking performance compared to environment agnostic references. On hardware, we integrate the policy with an MPC + control barrier function planner and demonstrate long-horizon (>70m) closed-loop autonomous navigation on the Unitree G1 through outdoor environments containing rough terrain and consecutive flights of stairs, with all sensing and computation onboard.
>
---
#### [new 035] Health-Conditioned Vision-Language-Action Models for Malfunction-Aware Robot Control
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决机器人物理故障下的任务执行问题。通过引入健康条件的视觉-语言-动作模型，使机器人能适应关节退化等故障完成任务。**

- **链接: [https://arxiv.org/pdf/2605.16056](https://arxiv.org/pdf/2605.16056)**

> **作者:** Hüseyin Arslan; Özgür Erkent
>
> **备注:** VLA Pipelines Workshop at IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Research on Vision Language Action (VLA) models has been increasing rapidly in recent years. Although some of them focus on detecting, preventing, and recovering from task failures, they usually don't deal with adapting to robot's physical failures. In real-life scenarios, most robots face physical degradations in various ways such as joint degradation, actuator failure, or weak gripper. We introduce malfunction-aware (health-conditioned) VLA that takes a health vector as an input that gives information about robots' joints' operation angle and torque capability, and adapts its predictions to complete the tasks with the degraded joints. To achieve this, we inject a Health Projector module to the VLA-Adapter architecture and train it on malfunction robot data we collected on the LIBERO environment [1]. We collect 128 teleoperated episodes on Libero-Spatial tasks. Our results show that, with a very lightweight addition, the model can learn to operate successfully with different configurations of degraded joints which the default pretrained VLA-Adapter's Libero-Spatial-Pro model cannot. The code and dataset will be available soon at this https URL
>
---
#### [new 036] Diffusion Policy for Coordinated Control of a Nonholonomic Mobile Base and Dual Arms in Door Opening and Passing
- **分类: cs.RO**

- **简介: 该论文研究机器人协调控制任务，解决开门和通过过程中非完整移动平台与双臂的协同问题。通过扩散策略实现端到端控制，提升任务成功率和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.15352](https://arxiv.org/pdf/2605.15352)**

> **作者:** Shangqun Yu; Matthew En; Daniel Wu; Sangjun Park; Ziyi Zhou; Seyed Fakoorian; Donghyun Kim
>
> **摘要:** Opening heavy, self closing doors, especially those that require pulling remains a long standing challenge in robotics. Humans naturally employ both arms in a dexterous manner, rotating the handle, widening the gap, holding the door, switching arms when needed, and moving through while maintaining clearance. To replicate such behaviors, a robot must perform a long sequence of motions spanning multiple stages and interactions with different parts of the door. Traditional approaches rely on state machines that transition between manually defined stages (e.g., pulling after the knob is rotated, passing after the gap is sufficiently wide). While intuitive, these methods lack robustness, as hand crafted trajectories fail to generalize to the diversity of real world conditions without extensive engineering effort. Recent advances in imitation learning offer a scalable alternative, yet no existing visual action model has demonstrated simultaneous coordination of a nonholonomic base and dual arms for the complete door opening and passing task. In this paper, we tackle this complex, highly constrained problem using a diffusion based visuomotor control policy. Our results demonstrate that a single end to end policy can be learned to execute long horizon tasks requiring tight coordination between manipulation and locomotion. The resulting policy not only achieves a high success rate in opening and traversing damped pull doors but also demonstrates strong robustness to external disturbances capabilities that are difficult to realize with traditional methods.
>
---
#### [new 037] Hierarchical and Holistic Open-Vocabulary Functional 3D Scene Graphs for Indoor Spaces
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D场景理解任务，旨在解决功能性3D图结构构建中的覆盖不足与结构单一问题，通过引入新数据和优化方法提升场景图的准确性和层次性。**

- **链接: [https://arxiv.org/pdf/2605.15753](https://arxiv.org/pdf/2605.15753)**

> **作者:** Xinggang Hu; Chenyangguang Zhang; Alexandros Delitzas; Xiangkui Zhang; Marc Pollefeys; Francis Engelmann; Xiangyang Ji
>
> **摘要:** Functional 3D scene graphs offer a versatile and flexible representation for 3D scene understanding and robotic manipulation, defined by object nodes, interactive elements, and functional relationship edges. However, their potential remains underexplored due to the limited coverage of existing benchmarks and the overly straightforward design of previous pipelines, which primarily focus on large-scale furniture but lack of hierarchical structures. Therefore, in this work, we extend the benchmark coverage by introducing dense tabletop objects and explicit multi-level functional relationships. This expansion introduces critical challenges involving small-scale, dense, and similar instances, with lack of visual anchoring in relational reasoning, instance confusion during cross-frame fusion, and attribution uncertainty under dynamic viewpoints. To address these issues, we propose an open-vocabulary pipeline based on 2D visual grounding and 3D graph optimization. Specifically, we anchor fine-grained functional edges from 2D visual evidence, and associate nodes across frames in 3D using multiple cues. Furthermore, edge association is formulated as temporal graph optimization, integrating evidence accumulation, entropy regularization, and temporal smoothing to robustly determine the functional connections of each node. Finally, global hierarchy shaping is performed to recover the hierarchical graph structure. Extensive experiments demonstrate that the proposed method can reliably infer functional 3D scene graphs in challenging real-world scenes, thereby further unlocking their potential for practical applications.
>
---
#### [new 038] FocalPolicy: Frequency-Optimized Chunking and Locally Anchored Flow Matching for Coherent Visuomotor Policy
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于视觉-运动策略任务，旨在解决长时序动作的连贯性问题。提出FocalPolicy，结合频率优化分块与局部锚定流匹配，提升动作一致性与预测精度。**

- **链接: [https://arxiv.org/pdf/2605.15944](https://arxiv.org/pdf/2605.15944)**

> **作者:** Qian He; Zhenshuo Yang; Wenqi Liang; Chunhui Hao; Nicu Sebe; Jiandong Tian
>
> **摘要:** Visuomotor policies aim to learn complex manipulation tasks from expert demonstrations. However, generating smooth and coherent trajectories remains challenging, as it requires balancing proximal precision with distal foresight. Existing approaches typically focus on optimizing intra-chunk action distributions, often neglecting the inter-chunk coherence. Consequently, inter-chunk discontinuities significantly impede the learning of coherent long-horizon actions. To overcome this limitation and achieve a synergetic balance between precision and foresight, we propose FocalPolicy, a foresight-aware visuomotor policy that combines Frequency-Optimized Chunking with Locally Anchored flow matching. We introduce a foresight composite objective that supervises time-domain alignment within the proximal actions while regularizing frequency-domain structure over multiple future action chunks to improve cross-chunk coherence. To efficiently learn complex action distributions, we design locally anchored campling to enhance target signal propagation efficiency during consistency flow matching training. Extensive experiments demonstrate that FocalPolicy outperforms existing approaches and confirm the generalizability of our modules to other baselines. Project website: this https URL
>
---
#### [new 039] Propagating Unsafe Actions in LLM Controlled Multi-Robot Collaboration via Single Robot Compromise
- **分类: cs.RO; cs.CR**

- **简介: 该论文属于安全任务，研究LLM控制的多机器人协作中的安全风险。工作是提出一种通过单个机器人传播恶意指令的攻击方法，揭示多机器人系统的安全漏洞。**

- **链接: [https://arxiv.org/pdf/2605.15641](https://arxiv.org/pdf/2605.15641)**

> **作者:** Zhen Huang; Zhihuang Liu; Weishang Wu; Zhiping Cai
>
> **备注:** Accepted by the 35th International Joint Conference on Artificial Intelligence (IJCAI 2026). 9 pages, 4 figures, 3 tables
>
> **摘要:** Large language models (LLMs) are increasingly used as general planners in embodied intelligence, enabling high level coordination and low level task planning for both single robot and multi-robot collaboration. This increasing reliance on embodied LLM planners also raises critical security concerns, since misaligned or manipulated instructions can be translated into physical actions. Prior work has studied such threats in single robot settings, while security risks in LLM controlled multi-robot collaboration, especially those propagated through inter robot communication, remain largely unexplored. To bridge this gap, we propose a novel attack paradigm for multi-robot system in which the adversary interacts with only a single entry robot. The compromised robot then propagates malicious intent through peer communication, leading to coordinated unsafe actions across the system. Our evaluation, covering high risk dimensions of dereliction of duty, privacy compromise, and public safety hazards, reveals a persistent safety alignment gap in multi-robot planners. We quantify this process with three metrics, obedience, infectiousness, and stealthiness. Experiments demonstrate both persistent attacker control and rapid propagation: obedience reaches 1.00 in the strongest cases, and infectiousness rises to 0.90. Notably, the attack is highly efficient, requiring as few as 3.0 rounds to compromise all the robots while maintaining a stealthiness score of 0.81. Such risks are amplified when robots must resolve trade offs in critical situations, such as emergencies or conflicts of rights, because the coordination mechanism can unintentionally allow adversarial instructions to override safety requirements. The code is available at this https URL.
>
---
#### [new 040] DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo
- **分类: cs.RO**

- **简介: 该论文提出DexJoCo，一个用于评估灵巧操作的基准和工具包，解决灵巧手任务性能评估不足的问题。包含11项任务，涵盖工具使用、双臂协作等，旨在推动灵巧手机器人学习研究。**

- **链接: [https://arxiv.org/pdf/2605.16257](https://arxiv.org/pdf/2605.16257)**

> **作者:** Hanwen Wang; Weizhi Zhao; Xiangyu Wang; Siyuan Huang; He Lin; Boyuan Zheng; Rongtao Xu; Gang Wang; Yao Mu; He Wang; Lue Fan; Hongsheng Li; Zhaoxiang Zhang; Tieniu Tan
>
> **备注:** 8 pages, 6 figures, project page is available at: this https URL
>
> **摘要:** Achieving human-level manipulation requires dexterous robotic hands capable of complex object interactions. Advancing such capabilities further demands standardized benchmarks for systematic evaluation. However, existing dexterous benchmarks lack tasks that reflect the unique manipulation capabilities of dexterous hands over parallel grippers, as well as comprehensive evaluation pipelines. In this paper, we present DexJoCo, a benchmark and toolkit for task-oriented dexterous manipulation, comprising 11 functionally grounded tasks that evaluate tool-use, bimanual coordination, long-horizon execution, and reasoning. We develop a low-cost data collection system and collect 1.1K trajectories across these tasks, with support for domain randomization to assess robustness. We benchmark modern models under diverse settings, including visual and dynamics randomization, multi-task training, and action-head adaptation. Through extensive empirical analysis, we identify several important insights and common limitations of current policies in dexterous manipulation, highlighting key challenges for future research in dexterous hand robot learning. Project page available at: this https URL
>
---
#### [new 041] Feedback World Model Enables Precise Guidance of Diffusion Policy
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人决策任务，解决世界模型在分布外状态下预测不可靠的问题。提出反馈世界模型，通过实时反馈修正预测，提升预测精度和策略性能。**

- **链接: [https://arxiv.org/pdf/2605.15705](https://arxiv.org/pdf/2605.15705)**

> **作者:** Tuo An; Jindou Jia; Gen Li; Jingliang Li; Chuhao Zhou; Pengfei Liu; Bofan Lyu; Jiaqi Bai; Xinying Guo; Geng Li; Jianfei Yang
>
> **备注:** 21 pages, 9 figures
>
> **摘要:** World models aim to improve robotic decision making by predicting the consequences of actions. However, in practice, their predictions often become unreliable once the robot encounters states outside the training distribution, limiting their effectiveness at deployment. We observe that execution itself provides a natural but underutilized signal: after each action, the robot directly observes the true next state, revealing the mismatch between predicted and actual outcomes. Building on this insight, we propose feedback world model, a new paradigm that closes the loop between prediction and observation at inference time. Instead of treating the world model as a static open-loop predictor, our method maintains a lightweight feedback state that is updated online to iteratively correct future predictions, compensating for model errors using real-time observations without additional training data or parameter updates. We show that this process can be interpreted as a latent-space observer and admits convergence guarantees under mild conditions. We further introduce action-aware guidance to better translate corrected predictions into control by emphasizing action-controllable components while suppressing irrelevant variations. Experiments on LIBERO-Plus, Robomimic, and real-world manipulation tasks demonstrate that our method substantially improves both prediction accuracy and policy performance under distribution shift. In particular, it reduces world model prediction error by up to 76.4% and improves out-of-distribution (OOD) success rate by 30%. These results show that incorporating real-time feedback at inference time provides a simple yet powerful alternative to static world modeling.
>
---
#### [new 042] NIMO Controller: a self-driving laboratory orchestrator based on the Model Context Protocol
- **分类: cs.AI; cond-mat.mtrl-sci; cs.RO**

- **简介: 该论文属于人工智能与实验科学交叉任务，旨在解决SDL软件难以适配AI的问题。提出基于MCP的NIMO控制器，统一人机交互接口，提升SDL自动化水平。**

- **链接: [https://arxiv.org/pdf/2605.15227](https://arxiv.org/pdf/2605.15227)**

> **作者:** Naruki Yoshikawa; Ryo Tamura
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Self-driving laboratories (SDLs) have attracted increasing attention as a means of accelerating scientific discovery; however, developing SDL software remains technically demanding. To improve accessibility, orchestration software frameworks have been proposed to coordinate SDL components. Nevertheless, existing frameworks are primarily designed for human interaction and do not provide standardized interfaces suitable for AI agents. In this work, we propose an SDL software architecture based on the Model Context Protocol (MCP), in which all SDL functionalities are exposed through MCP servers. Following this design principle, we introduce an MCP-based SDL orchestrator, named NIMO Controller. It provides a visual programming interface automatically generated through MCP-based tool discovery, allowing human users to design experimental workflows without writing code. The same MCP backend can also be accessed by AI agents, providing a unified interface for both human users and AI agents. We demonstrate the proposed system through a case study on a color-matching SDL. The results validate the usability of the proposed MCP-based SDL architecture.
>
---
#### [new 043] DiLA: Disentangled Latent Action World Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出DiLA，解决LAMs中动作抽象与生成质量的平衡问题，通过内容-结构解耦实现高质量视频生成与语义动作空间。**

- **链接: [https://arxiv.org/pdf/2605.15725](https://arxiv.org/pdf/2605.15725)**

> **作者:** Tianqiu Zhang; Muyang Lyu; Yufan Zhang; Fang Fang; Si Wu
>
> **备注:** Project Page: this http URL
>
> **摘要:** Latent Action Models (LAMs) enable the learning of world models from unlabeled video by inferring abstract actions between consecutive frames. However, LAMs face a fundamental trade-off between action abstraction and generation fidelity. Existing methods typically circumvent this issue by using two-stage training with pre-trained world models or by limiting predictions to optical flow. In this paper, we introduce DiLA, a novel Disentangled Latent Action world model that aims to resolve this trade-off via content-structure disentanglement. Our key insight is that disentanglement and latent action learning are co-evolving: the predictive bottleneck inherent in latent action learning serves as a driving force for disentanglement, compelling the model to distill spatial layouts into the structure pathway while offloading visual details to a separate content pathway for generation. This synergy yields a continuous, semantically structured latent action space without compromising generative quality. DiLA achieves superior results in video generation quality, action transfer, visual planning, and manifold interpretability. These findings establish DiLA as a unified framework that simultaneously achieves high-level action abstraction and high-fidelity generation, advancing the frontier of self-supervised world model learning.
>
---
#### [new 044] Learning Bilevel Policies over Symbolic World Models for Long-Horizon Planning
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于长期规划任务，旨在解决机器人在复杂环境中执行长周期任务的难题。通过结合低层模仿学习与高层符号抽象，提出双层策略框架，提升规划效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.15975](https://arxiv.org/pdf/2605.15975)**

> **作者:** Dillon Z. Chen; Till Hofmann; Toryn Q. Klassen; Sheila A. McIlraith
>
> **摘要:** We tackle the challenge of building embodied AI agents that can reliably solve long-horizon planning problems. Imitation learning from demonstrations has shown itself to be effective in training robots to solve a diversity of complex tasks requiring fine motor control and manipulation over low-level (LL), continuous environments. Yet, it remains a difficult endeavour to generate long-horizon plans from imitation learning alone. In contrast, high-level (HL), symbolic abstractions facilitate efficient and interpretable long-horizon planning. We propose to combine the strengths of LL imitation learning for manipulation and control, and HL symbolic abstractions for long-horizon planning. We realise this idea via \emph{bilevel policies} of the form $(\pi^{\mathrm{hl}}, \pi^{\mathrm{ll}})$, consisting of a neural policy $\pi^{\mathrm{ll}}$ learned from LL demonstrations, and an HL symbolic policy $\pi^{\mathrm{hl}}$ that is constructed from symbolic abstractions of the LL demonstrations combined with inductive generalisation. We implement these ideas in the BISON system. Experiments on extended MetaWorld benchmarks demonstrate that BISON generalises to long horizons and problems with greater numbers of objects than those solved by VLA and end-to-end methods, and is more time and memory efficient in training and inference. Notably, when ignoring LL execution, BISON's HL policies can solve HL problems with 10,000 relevant objects in under a minute. Project page: this https URL
>
---
#### [new 045] CM-EVS: Sparse Panoramic RGB-D-Pose Data for Complete Scene Coverage
- **分类: cs.CV; cs.GR; cs.LG; cs.RO**

- **简介: 该论文提出CM-EVS数据集，解决3D视觉学习中场景覆盖不全、冗余及不一致的问题。通过COVER方法生成稀疏、一致的全景RGB-D-pose数据，提升场景覆盖效率与质量。**

- **链接: [https://arxiv.org/pdf/2605.15597](https://arxiv.org/pdf/2605.15597)**

> **作者:** Jiale Liu; Jungang Li; Jieming Yu; Xinglin Yu; Zihao Dongfang; Zongjian Ding; Kaifeng Ding; Yi Yang; Lidong Chen; Yang Zou; Shunwen Bai; Jiahuan Zhang; Haoran Huang; Shan Huang; Yudong Gao; Mingjun Cheng
>
> **备注:** 35 pages including appendix. Code and dataset: this https URL
>
> **摘要:** Modern 3D visual learning relies on observations sampled from metric 3D assets, yet existing scans, meshes, point clouds, simulations, and reconstructions do not directly provide a sparse, comparable, and geometry-consistent panoramic training interface. Dense trajectories duplicate nearby views, source-specific rendering policies yield heterogeneous annotations, and sparse heuristics may miss important regions or introduce depth-inconsistent observations. We study how to convert 3D assets into sparse panoramic RGB-D-pose data that preserves complete scene coverage with low redundancy and auditable provenance. We propose COVER (Coverage-Oriented Viewpoint curation with ERP Range-depth warping), a training-free ERP viewpoint curator that projects geometry observed from selected views into candidate ERP probes, scores incremental coverage, and penalizes depth conflicts. Under bounded proxy error, its greedy coverage proxy preserves the standard coverage-style approximation behavior up to an additive error term. Using COVER, we build CM-EVS (Coverage-curated Metric ERP View Set), a panoramic RGB-D-pose dataset with 36,373 curated ERP frames from 1,275 indoor scenes across Blender indoor, HM3D, and ScanNet++, complemented by outdoor panoramas from TartanGround and OB3D re-encoded into the same schema. Each frame provides full-sphere RGB, metric range depth, calibrated pose; COVER-produced indoor frames include per-step provenance logs. With a median of only 25 frames per indoor scene, CM-EVS covers all 13 unified room types while maintaining compact scene-level coverage. Experiments show that COVER improves the coverage-conflict trade-off, making CM-EVS a sparse, compact, and auditable RGB-D-pose resource for geometry-consistent panoramic 3D learning.
>
---
#### [new 046] Learn Where Outcomes Diverge: Efficient VLA RL via Probabilistic Chunk Masking
- **分类: cs.LG; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）强化学习任务，解决训练效率低的问题。通过概率性块掩码（PCM）优化梯度计算，提升训练速度并降低资源消耗。**

- **链接: [https://arxiv.org/pdf/2605.16154](https://arxiv.org/pdf/2605.16154)**

> **作者:** Vaidehi Bagaria; Nikshep Grampurohit; Pulkit Verma
>
> **摘要:** Reinforcement learning (RL) allows vision-language-action (VLA) policies to generalize beyond their training distribution by optimizing directly for task success, but post-training is computationally expensive. A natural response has been to speed rollout collection through faster simulators and world models. In GRPO-based VLA RL, we find that the dominant cost lies elsewhere: gradient computation accounts for approximately 78% of wall-clock time per step in our runs, while rollout collection accounts for only 21%. Gradient cost dominates because much of this computation is spent on phases that contribute little to learning. GRPO's learning signal is driven by advantage variance: only phases where successful and failed rollouts diverge produce learning signal. However, GRPO assigns the same advantage to every chunk in a rollout. As a result, actor-update compute is spent uniformly across the trajectory, including phases the policy already handles after pre-training and supervised fine-tuning. This paper presents Probabilistic Chunk Masking (PCM), a drop-in modification to GRPO that allocates gradient computation to a small, probabilistically selected subset of chunks per trajectory. PCM scores semantic phases using success-failure action variance, a rollout-derived proxy for per-phase gradient variance, and samples a fixed chunk budget with online-updated phase-level keep probabilities. We formalize per-phase gradient variance as the quantity determines where gradient computation is useful and show that success-failure action variance provides a measurable proxy for it. PCM requires no reward model or learned critic. On three LIBERO benchmarks, PCM matches the final success rate of standard GRPO while achieving 2.38 times wall-clock speedup, 4.8 times faster gradient updates, and 60% lower peak activation memory, while backpropagating through fewer than 20% of trajectory chunks.
>
---
#### [new 047] Mind Dreamer: Untethering Imagination via Active Latent Intervention on Latent Manifolds
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习领域，解决MBRL中因历史状态依赖导致的样本效率问题。提出Mind Dreamer框架，通过主动潜在干预提升想象力，提高稀疏奖励任务性能。**

- **链接: [https://arxiv.org/pdf/2605.16030](https://arxiv.org/pdf/2605.16030)**

> **作者:** Shaojun Xu; Xiaoling Zhou; Yihan Lin; Yapeng Meng; Xinglong Ji; Luping Shi; Rong Zhao
>
> **备注:** 34 pages, 7 figures
>
> **摘要:** Model-Based Reinforcement Learning (MBRL) leverages latent imagination for sample efficiency, yet remains constrained by Historical Tethering: imagination is typically initialized from observed states. This creates a learning asymmetry, where the world model's manifold discovery outpaces the policy's sparse-reward optimization. We propose Mind Dreamer (MD), a framework that operationalizes Active Latent Intervention (ALI) to transcend Markovian continuity. MD reformulates discovery as the minimization of a global Relay Manifold Expected Free Energy (R-EFE); by sampling initial states from a learned generator $s_0 \sim p_{gen}(\cdot)$ rather than the historical buffer, MD utilizes an adversarial generator to synthesize non-continuous latent jumps to epistemic blind spots that are physically plausible yet cognitively challenging. To resolve the credit assignment paradox across these spatial ruptures, we derive the Relay Value Function (RVF) and Relay Uncertainty Function (RUF). These potentials treat synthesized anchors as counterfactual intermediary states, propagating pragmatic and epistemic value through a principled Bellman-style formulation. Notably, we prove that uncertainty propagation across discontinuities necessitates a quadratic discount $\gamma^2$, establishing a formal epistemic horizon. Theoretically, MD approximates a variance-minimizing importance sampler that expands the manifold's spectral gap, reducing the hitting time to critical bottleneck states. Empirically, MD achieves a 1.67$\times$ average speedup over DreamerV3 on DeepMind Control Suite, reaching 8.8$\times$ in sparse-reward tasks.
>
---
#### [new 048] parallelcbf: A composable safety-filter and auditability framework for tensor-parallel reinforcement learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出ParallelCBF框架，解决安全约束强化学习的统一训练问题，整合并行环境、安全过滤、审计功能等，提升可复现性与安全性。**

- **链接: [https://arxiv.org/pdf/2605.15509](https://arxiv.org/pdf/2605.15509)**

> **作者:** Yijun Lu; Zilei Yang; Yuyin Ma
>
> **摘要:** While Isaac Lab provides massive parallel UAV simulation, OmniSafe and safe-control-gym provide constrained-RL benchmarks, and CBFKit provides control-barrier-function synthesis tooling, no existing framework unifies these capabilities for end-to-end safety-constrained training. ParallelCBF is the first framework to unify (i)~tensor-parallel UAV environments, (ii)~hard-gate CBF safety filters, (iii)~sharded BC-to-RL pipelines, and (iv)~first-class operational auditability -- pre-registration, watchdog registries, failure forensics, and dataset audits as composable APIs rather than user-implemented scripts. We release ParallelCBF v0.1.0 under Apache~2.0 with a four-layer composable API, a CPU PyTorch reference implementation of a dual-barrier (squared / linear-predictive) CBF, property-based safety invariance tests across vectorized batch sizes that complete in 1.67~s for the full 39-test suite, and a 31{,}415-episode behavior-cloning collection campaign whose curriculum mix, per-bucket yields, and dataset SHA-256 are auditable through the framework's own \texttt{ops} primitives. We report a representative end-to-end pipeline execution in which the framework's auditability layer halted a downstream training stage that did not meet pre-registered convergence criteria, preventing silent propagation of a degraded checkpoint -- an architectural property we argue is necessary, not merely useful, for reproducible empirical robotics research. The framework is installable via \texttt{pip install parallelcbf}; source and release artifacts are available at this https URL.
>
---
#### [new 049] IVGT: Implicit Visual Geometry Transformer for Neural Scene Representation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出IVGT，用于从无姿态多视角图像中重建连续一致的3D几何与外观，解决场景表示问题。**

- **链接: [https://arxiv.org/pdf/2605.16258](https://arxiv.org/pdf/2605.16258)**

> **作者:** Yuqi Wu; Tianyu Hu; Wenzhao Zheng; Yuanhui Huang; Haowen Sun; Jie Zhou; Jiwen Lu
>
> **备注:** Code: this https URL
>
> **摘要:** Reconstructing coherent 3D geometry and appearance from unposed multi-view images is a fundamental yet challenging problem in computer vision. Most existing visual geometry foundation models predict explicit geometry by regressing pixel-aligned pointmaps, often suffering from redundancy and limited geometric continuity. We propose IVGT, an Implicit Visual Geometry Transformer that implicitly models continuous and coherent geometry from pose-free multi-view images. This formulation learns a continuous neural scene representation in a canonical coordinate system and supports continuous spatial queries at any 3D positions, retrieving local features to predict signed distance (SDF) values and colors using lightweight decoders. It allows direct extraction of continuous and coherent surface geometry, enabling rendering of RGB images, depth maps, and surface normal maps from arbitrary viewpoints. We train IVGT via multi-dataset joint optimization with 2D supervision and 3D geometric regularization. IVGT demonstrates generalization across scenes and achieves strong performance on various tasks, including mesh and point cloud reconstruction, novel view synthesis, depth and surface normal estimation, and camera pose estimation.
>
---
#### [new 050] Optimizing Line Segment Inspection with Limited-Range Drones
- **分类: cs.CG; cs.DS; cs.RO; math.OC**

- **简介: 该论文属于无人机路径规划任务，解决有限续航下线段巡检的优化问题。通过设计算法，实现快速覆盖并最小化最大完成时间。**

- **链接: [https://arxiv.org/pdf/2605.15765](https://arxiv.org/pdf/2605.15765)**

> **作者:** José-Miguel Díaz-Báñez; José-Manuel Higes; Alina Kasiuk; Inmaculada Ventura
>
> **备注:** 28 pages, 14 figures
>
> **摘要:** Optimization problems with drones are widely studied in a variety of civilian tasks, mainly due to their ability to traverse rough terrains and to carry cameras and other sensors for surveillance tasks. The limited battery life of these aerial robots poses challenges in operational research. In this paper, we address the following optimization problem. We are given a set of line segments (e.g. tubes in a solar plant) to inspect by drones. The objective is to detect broken pipes using artificial intelligence and path planning must be carried out efficiently. On the one hand, the limited capacity of the batteries necessitates periodic visits (tours) to a fixed base station. However, it is desirable to allocate a set of tours for each drone to ensure that the segments are covered as quickly as possible, aiming to minimize the makespan, which is the maximum time spent by any drone. We are able to prove that this optimization problem is strongly NP-hard even when the segments are positioned on a line and the scenario involves only two drones. Then, approximation algorithms are proposed. Our computational experiments demonstrate that the proposed algorithm achieves near-optimal performance across diverse operational scenarios.
>
---
#### [new 051] Driving Through the Network: Performance and Workload Under Latency and Video Impairment
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机协同驾驶研究，旨在解决网络延迟与视频质量对远程驾驶性能的影响。通过实验分析不同延迟和比特率下的操作表现与生理反应，提出优化设计建议。**

- **链接: [https://arxiv.org/pdf/2605.15952](https://arxiv.org/pdf/2605.15952)**

> **作者:** Ines Trautmannsheimer; Ahmed Azab; Frank Diermeyer
>
> **备注:** Preprint of VEHITS 2026 : 12th International Conference on Vehicle Technology and Intelligent Transport Systems
>
> **摘要:** Teleoperation promises to extend the operational envelope of automated vehicles, yet it critically depends on network latency and video quality. We report a fixed-base driving-simulator study (N=25) with a 2x2 manipulation of added latency (100/300 ms) and bitrate (500/2000 kbit/s), plus a best-case baseline (0 ms added, 9000 kbit/s). We measured effective glass-to-glass (G2G) latency per condition (baseline approx. 413 ms; effective totals approx. 500-700 ms) and verified stable framerate and encoder settings. Multimodal measures covered performance (speed, steering reversals, crashes), oculomotor behavior (blink rate, fixation duration), physiology (RR interval, heart rate, skin conductance), and subjective workload. Latency and bitrate each increased operator load and modestly affected performance. Physiological measures (heart rate, RR interval) exhibited sub-additive interactions, whereas performance and oculomotor interactions were small or non-significant. Equivalence tests showed that 300 ms with 2000 kbit/s was velocity-equivalent to best-case (SESOI +/- 2 km/h), while 300 ms with 500 kbit/s was not. We argue that latency and video quality should be treated as largely independent design levers, and that physiology-aware adaptation can anticipate overload before safety is compromised.
>
---
#### [new 052] From Gridworlds to Warehouses: Adapting Lightweight One-shot Multi-Agent Pathfinding for AGVs
- **分类: cs.MA; cs.RO**

- **简介: 该论文属于多智能体路径规划任务，解决仓库中AGVs的路径规划问题。针对实际约束改进传统网格路径规划模型，并评估多种算法的性能。**

- **链接: [https://arxiv.org/pdf/2605.15799](https://arxiv.org/pdf/2605.15799)**

> **作者:** Hiroki Nagai; Keisuke Okumura
>
> **备注:** To be presented at IJCAI 2026
>
> **摘要:** Multi-agent pathfinding (MAPF) under one-shot planning is a core component of warehouse automation, yet classical formulations typically assume four-connected 2D grids with unit-time moves in four directions. To fill reality gaps while still being trackable with discrete combinatorial search, this work proposes a more practical counterpart tailored to differential-drive AGVs. We term this multi-agent warehouse pathfinding (MAWPF), featured with four constraints: (i) agent actions are restricted to straight motion and in-place rotation; (ii) rotations require multi-step costs; (iii) acceleration and deceleration are considered, and; (iv) follower collisions are prohibited to prevent rear-end crashes. To solve MAWPF efficiently, we adapt representative suboptimal MAPF algorithms-PP, LNS2, PIBT, and LaCAM-and conduct comprehensive benchmarking. Our experiments reveal that PP and LNS2 struggle to solve instances with many agents, while PIBT-based approaches achieve preferable scalability with increased solution cost. We believe that these constitute an important step toward adapting classical gridworld MAPF to operational warehouse setups.
>
---
#### [new 053] STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics-Physics Dual System
- **分类: cs.CV; cs.RO**

- **简介: 该论文聚焦于生成符合任务指令的仿真桌面场景。针对现有方法依赖语言模型导致的物理不合理的缺陷，提出STABLE系统，结合语义推理与物理修正模块，提升场景的物理合理性与任务一致性。**

- **链接: [https://arxiv.org/pdf/2605.16137](https://arxiv.org/pdf/2605.16137)**

> **作者:** Zhen Luo; Yixuan Yang; Xudong Xu; Jinkun Hao; Zhaoyang Lyu; Feng Zheng; Jiangmiao Pang; Yanwei Fu
>
> **备注:** ICML 2026
>
> **摘要:** Generating simulation-ready tabletop scenes from task instructions is an intriguing and promising research direction in the field of Embodied AI. However, existing task-to-scene generation methods rely exclusively on large language models (LLMs) to predict scene layouts, inevitably yielding object collisions or floating due to LLMs' inherent limitations in 3D spatial reasoning. In this paper, we present STABLE, a semantics-physics dual-system tailored for simulation-ready tabletop scene generation. STABLE consists of two complementary modules: (i) a Semantic Reasoner, a fine-tuned LLM trained on a structured tabletop scene dataset to generate coarse layouts from input task instructions, and (ii) a Physics Corrector, a physics-aware flow-based denoising model that outputs pose updates to refine layouts, which ensures the physical plausibility of scenes while preserves semantic alignment with task instructions. STABLE adopts a progressive generation paradigm: by alternating between the Semantic Reasoner and Physics Corrector, it incrementally expands the scene from task-critical objects to background objects. Experiments demonstrate that STABLE successfully generates simulation-ready tabletop scenes that strictly conform to task instructions and significantly enhances the physical validity of scenes over prior art.
>
---
## 更新

#### [replaced 001] Learning Structured Robot Policies from Vision-Language Models via Synthetic Neuro-Symbolic Supervision
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决传统方法缺乏可解释性的问题。通过神经符号方法，从多模态数据生成结构化机器人策略，实现零样本迁移。**

- **链接: [https://arxiv.org/pdf/2604.02812](https://arxiv.org/pdf/2604.02812)**

> **作者:** Alessandro Adami; Tommaso Tubaldo; Marco Todescato; Ruggero Carli; Pietro Falco
>
> **摘要:** Vision-Language Models (VLMs) have recently demonstrated strong capabilities in mapping multimodal observations to robot behaviors. However, most current approaches rely on end-to-end visuomotor policies that remain opaque and difficult to analyze, limiting their use in real-world robotic applications. In contrast, classical robotic systems often rely on structured policy representations that provide interpretability, modularity, and reactive execution. This work investigates how foundation models can be specialized to generate structured robot policies grounded in multimodal perception, bridging high-dimensional learning and symbolic control. We propose a neuro-symbolic approach in which a VLM synthesizes executable Behavior Tree policies from visual observations, natural language instructions, and structured system specifications. To enable scalable supervision without manual annotation, we introduce an automated pipeline that generates a synthetic multimodal dataset of domain-randomized scenes paired with instruction-policy examples produced by a foundation model. By decoupling structured task decomposition under constrained symbolic grammars from hardware-specific motor control, we demonstrate that a 12B-parameter model can learn structured spatial-symbolic mappings required for executable BT synthesis, solely through in-silico supervision. Real-world physical experiments on two heterogeneous robotic manipulators confirm that these structurally constrained policies achieve zero-shot transfer to real-world environments. The results emphasize that the data bottleneck in robotic planning can be bypassed by procedurally synthesizing high-fidelity, neuro-symbolic training data.
>
---
#### [replaced 002] Coordinated Diffusion: Generating Multi-Agent Behavior Without Multi-Agent Demonstrations
- **分类: cs.RO**

- **简介: 该论文属于多智能体协作任务，解决多智能体演示数据稀缺的问题。通过单智能体数据和成本函数，生成协调行为，提升数据效率。**

- **链接: [https://arxiv.org/pdf/2605.11485](https://arxiv.org/pdf/2605.11485)**

> **作者:** Lasse Peters; Laura Ferranti; Andrea Bajcsy; Javier Alonso-Mora
>
> **摘要:** Imitation learning powered by generative models has proven effective for modeling complex single-agent behaviors. However, teaching multi-agent systems, like multiple arms or vehicles, to coordinate through imitation learning is hindered by a fundamental data bottleneck: as the joint state-action space grows exponentially with the number of agents, collecting a sufficient amount of coordinated multi-agent demonstrations becomes extremely costly. In this work, we ask: how can we leverage single-agent demonstration data to learn multi-agent policies? We present Coordinated Diffusion (CoDi), a framework that couples independently trained single-agent diffusion policies through a user-defined multi-agent cost function, without requiring any coordinated demonstrations. We derive a new diffusion-based sampling scheme wherein the diffusion score function decomposes into independent, single-agent pre-trained base policies plus a cost-driven guidance term that coordinates these base policies into cohesive multi-agent behavior. We show that this guidance term can be estimated in a gradient-free manner, making CoDi applicable to black-box, non-differentiable cost functions without additional training. Theoretically and empirically, we analyze the conditions under which this composition can faithfully approximate a target multi-agent behavior. We find a complementary role for demonstration data versus the cost function: single-agent demonstrations must cover the support of the desired multi-agent behavior, while the cost function must promote desired behavior from this product of single-agent policies. Our results in simulation and hardware experiments of a two-arm manipulation task show that CoDi discovers robust coordinated behavior from single-agent data, is more data-efficient than multi-agent baselines, and highlights the importance of joint guidance, base policy support, and cost design.
>
---
#### [replaced 003] TACO: General Acrobatic Flight Control via Target-and-Command-Oriented Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于飞行控制任务，旨在解决传统方法无法在线调整参数的问题。提出TACO框架，实现通用机动控制与参数实时调整。**

- **链接: [https://arxiv.org/pdf/2503.01125](https://arxiv.org/pdf/2503.01125)**

> **作者:** Zikang Yin; Canlun Zheng; Shiliang Guo; Zhikun Wang; Shiyu Zhao
>
> **备注:** For the experiment video, please refer to this https URL
>
> **摘要:** Although acrobatic flight control has been studied extensively, one key limitation of the existing methods is that they are usually restricted to specific maneuver tasks and cannot change flight pattern parameters online. In this work, we propose a target-and-command-oriented reinforcement learning (TACO) framework, which can handle different maneuver tasks in a unified way and allows online parameter changes. Additionally, we propose a spectral normalization method with input-output rescaling to enhance the policy's temporal and spatial smoothness, independence, and symmetry, thereby overcoming the sim-to-real gap. We validate the TACO approach through extensive simulation and real-world experiments, demonstrating its capability to achieve high-speed circular flights and continuous multi-flips.
>
---
#### [replaced 004] Sampling-Based Global Optimal Control and Estimation via Semidefinite Programming
- **分类: cs.RO**

- **简介: 该论文属于控制与优化任务，解决全局最优控制与估计问题。通过KernelSOS方法结合半定规划，提升机器人控制和轨迹优化的效率与质量。**

- **链接: [https://arxiv.org/pdf/2507.17572](https://arxiv.org/pdf/2507.17572)**

> **作者:** Antoine Groudiev; Fabian Schramm; Éloïse Berthier; Justin Carpentier; Frederike Dümbgen
>
> **摘要:** Global optimization has gained attraction over the past decades, thanks to the development of both theoretical foundations and efficient numerical routines. Among recent advances, Kernel Sum of Squares (KernelSOS) provides a powerful theoretical framework, combining the expressivity of kernel methods with the guarantees of SOS optimization. In this paper, we take KernelSOS from theory to practice and demonstrate its use on challenging control and robotics problems. We identify and address the practical considerations required to make the method work in applied settings: restarting strategies, systematic calibration of hyperparameters, methods for recovering minimizers, and the combination with fast local solvers. As a proof of concept, the application of KernelSOS to robot localization highlights its competitiveness with existing SOS approaches that rely on heuristics and handcrafted reformulations to render the problem polynomial. Even in the high-dimensional, non-parametric setting of trajectory optimization with simulators treated as black boxes, we demonstrate how KernelSOS can be combined with fast local solvers to uncover higher-quality solutions without compromising overall runtimes.
>
---
#### [replaced 005] Efficiently Solving Mixed-Hierarchy Games with Quasi-Policy Approximations
- **分类: cs.GT; cs.RO**

- **简介: 该论文研究多机器人协同中的混合层次博弈问题，解决同时存在纳什和斯塔克尔伯格决策结构的协调难题。提出一种准策略近似方法，提升求解效率。**

- **链接: [https://arxiv.org/pdf/2602.01568](https://arxiv.org/pdf/2602.01568)**

> **作者:** Hamzah Khan; Dong Ho Lee; Jingqi Li; Tianyu Qiu; Christian Ellis; Jesse Milzman; Wesley Suttle; David Fridovich-Keil
>
> **摘要:** Multi-robot coordination often exhibits hierarchical structure, with some robots' decisions depending on the planned behaviors of others. While game theory provides a principled framework for such interactions, existing solvers struggle to handle mixed information structures that combine simultaneous (Nash) and hierarchical (Stackelberg) decision-making. We study N-robot forest-structured mixed-hierarchy games, in which each robot acts as a Stackelberg leader over its subtree while robots in different branches interact via Nash equilibria. We derive the Karush-Kuhn-Tucker (KKT) first-order optimality conditions for this class of games and show that they involve increasingly high-order derivatives of robots' best-response policies as the hierarchy depth grows, rendering a direct solution intractable. To overcome this challenge, we introduce a quasi-policy approximation that removes higher-order policy derivatives and develop an inexact Newton method for efficiently solving the resulting approximated KKT systems. We prove local exponential convergence of the proposed algorithm for games with non-quadratic objectives and nonlinear constraints. The approach is implemented in a highly optimized Julia library (this http URL) and evaluated in hardware and simulated multi-agent experiments, demonstrating real-time convergence for complex mixed-hierarchy information structures.
>
---
#### [replaced 006] HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations
- **分类: cs.RO**

- **简介: 该论文提出HoMMI框架，解决从人类演示中学习全身移动操作的问题。通过跨体感策略设计，缩小人机差异，实现长时序的协作与导航任务。**

- **链接: [https://arxiv.org/pdf/2603.03243](https://arxiv.org/pdf/2603.03243)**

> **作者:** Xiaomeng Xu; Jisang Park; Han Zhang; Eric Cousineau; Aditya Bhat; Jose Barreiros; Dian Wang; Jeannette Bohg; Shuran Song
>
> **摘要:** We present Whole-Body Mobile Manipulation Interface (HoMMI), a data collection and policy learning framework that learns whole-body mobile manipulation directly from robot-free human demonstrations. We augment UMI interfaces with egocentric sensing to capture the global context required for mobile manipulation, enabling portable, robot-free, and scalable data collection. However, naively incorporating egocentric sensing introduces a larger human-to-robot embodiment gap in both observation and action spaces, making policy transfer difficult. We explicitly bridge this gap with a cross-embodiment hand-eye policy design, including an embodiment agnostic visual representation; a relaxed head action representation; and a whole-body controller that realizes hand-eye trajectories through coordinated whole-body motion under robot-specific physical constraints. Together, these enable long-horizon mobile manipulation tasks requiring bimanual and whole-body coordination, navigation, and active perception. Results are best viewed on: this https URL
>
---
#### [replaced 007] FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出FlashSAC，一种快速稳定的离策略强化学习算法，用于高维机器人控制任务，解决传统方法在性能和效率上的不足。**

- **链接: [https://arxiv.org/pdf/2604.04539](https://arxiv.org/pdf/2604.04539)**

> **作者:** Donghu Kim; Youngdo Lee; Minho Park; Kinam Kim; I Made Aswin Nahendra; Takuma Seno; Sehee Min; Daniel Palenicek; Florian Vogt; Danica Kragic; Jan Peters; Jaegul Choo; Hojoon Lee
>
> **备注:** RSS'26
>
> **摘要:** Reinforcement learning (RL) is a core approach for robot control when expert demonstrations are unavailable. On-policy methods such as Proximal Policy Optimization (PPO) are widely used for their stability, but their reliance on narrowly distributed on-policy data limits accurate policy evaluation in high-dimensional state and action spaces. Off-policy methods can overcome this limitation by learning from a broader state-action distribution, yet suffer from slow convergence and instability, as fitting a value function over diverse data requires many gradient updates, causing critic errors to accumulate through bootstrapping. We present FlashSAC, a fast and stable off-policy RL algorithm built on Soft Actor-Critic. Motivated by scaling laws observed in supervised learning, FlashSAC sharply reduces gradient updates while compensating with larger models and higher data throughput. To maintain stability at increased scale, FlashSAC explicitly bounds weight, feature, and gradient norms, curbing critic error accumulation. Across over 60 tasks in 10 simulators, FlashSAC consistently outperforms PPO and strong off-policy baselines in both final performance and training efficiency, with the largest gains on high-dimensional tasks such as dexterous manipulation. In sim-to-real humanoid locomotion, FlashSAC reduces training time from hours to minutes, demonstrating the promise of off-policy RL for sim-to-real transfer.
>
---
#### [replaced 008] Approximating Global Contact-Implicit MPC via Sampling and Local Complementarity
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决接触丰富行为的实时全局优化问题。通过结合局部互补控制与全局采样，提出一种新型控制器，实现高效精准操作。**

- **链接: [https://arxiv.org/pdf/2505.13350](https://arxiv.org/pdf/2505.13350)**

> **作者:** Sharanya Venkatesh; Bibit Bianchini; Alp Aydinoglu; William Yang; Michael Posa
>
> **备注:** S.V. and B.B. contributed equally to this work. Accepted to RA-L 2025; presented at ICRA 2026. Project page: this https URL
>
> **摘要:** To achieve general-purpose dexterous manipulation, robots must rapidly devise and execute contact-rich behaviors. Existing model-based controllers are incapable of globally optimizing in real-time over the exponential number of possible contact sequences. Instead, recent progress in contact-implicit control has leveraged simpler models that, while still hybrid, make local approximations. However, the use of local models inherently limits the controller to only exploit nearby interactions, potentially requiring intervention to richly explore the space of possible contacts. We present a novel approach which leverages the strengths of local complementarity-based control in combination with low-dimensional, but global, sampling of possible end-effector locations. Our key insight is to consider a contact-free stage preceding a contact-rich stage at every control loop. Our algorithm, in parallel, samples end effector locations to which the contact-free stage can move the robot, then considers the cost predicted by contact-rich MPC local to each sampled location. The result is a globally-informed, contact-implicit controller capable of real-time dexterous manipulation. We demonstrate our controller on precise, non-prehensile manipulation of non-convex objects using a Franka Panda arm. Project page: this https URL
>
---
#### [replaced 009] RE-SAC: Disentangling aleatoric and epistemic risks in bus fleet control: A stable and robust ensemble DRL approach
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，解决公交调度中的不确定性问题。针对交通和乘客需求的随机性，提出RE-SAC方法，分离aleatoric与epistemic风险，提升策略稳定性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.18396](https://arxiv.org/pdf/2603.18396)**

> **作者:** Yifan Zhang; Liang Zheng
>
> **摘要:** Bus holding control is challenging due to stochastic traffic and passenger demand. While deep reinforcement learning (DRL) shows promise, standard actor-critic algorithms suffer from Q-value instability in volatile environments. A key source of this instability is the conflation of two distinct uncertainties: aleatoric uncertainty (irreducible noise) and epistemic uncertainty (data insufficiency). Treating these as a single risk leads to value underestimation in noisy states, causing catastrophic policy collapse. We propose a robust ensemble soft actor-critic (RE-SAC) framework to explicitly disentangle these uncertainties. RE-SAC applies Integral Probability Metric (IPM)-based weight regularization to the critic network to hedge against aleatoric risk, providing a smooth analytical lower bound for the robust Bellman operator without expensive inner-loop perturbations. To address epistemic risk, a diversified Q-ensemble penalizes overconfident value estimates in sparsely covered regions. This dual mechanism prevents the ensemble variance from misidentifying noise as a data gap, a failure mode identified in our ablation study. Experiments in a realistic bidirectional bus corridor simulation demonstrate that RE-SAC achieves the highest cumulative reward (approx. -0.4e6) compared to vanilla SAC (-0.55e6). Mahalanobis rareness analysis confirms that RE-SAC reduces Oracle Q-value estimation error by up to 62% in rare out-of-distribution states (MAE of 1647 vs. 4343), demonstrating superior robustness under high traffic variability.
>
---
#### [replaced 010] ProCompNav: Proactive Instance Navigation with Comparative Judgment for Ambiguous User Queries
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出ProCompNav，解决自然语言实例导航中用户查询模糊的问题。通过构建候选池并进行比较判断，减少用户输入，提升导航成功率。属于对话系统任务。**

- **链接: [https://arxiv.org/pdf/2605.06223](https://arxiv.org/pdf/2605.06223)**

> **作者:** Junhyuk Kwon; Seungjoon Lee; Hyejin Park; Kyle Min; Jungseul Ok
>
> **备注:** Project page: this https URL . Code: this https URL
>
> **摘要:** Natural-language instance navigation becomes challenging when the initial user request does not uniquely specify the target instance. A practical agent should reduce the user's burden by actively asking only the information needed to distinguish the target from similar distractors, rather than requiring a detailed description upfront. Existing approaches often fall short of this goal: they may stop at the first plausible candidate before sufficiently exploring alternatives, or, even after collecting multiple candidates, ask about the target's attributes derived from individual candidates rather than questions selected to distinguish candidates in the pool. As a result, despite the dialogue, the agent may still fail to distinguish the target from distractors, leading to premature decisions and lengthy user responses. We propose Proactive Instance Navigation with Comparative Judgment (ProCompNav), a two-stage framework that first constructs a candidate pool and then identifies the target through comparative judgment. At each round, ProCompNav extracts an attribute-value pair that splits the current pool, asks a binary yes/no question, and prunes all inconsistent candidates at once. This reframes disambiguation from open-ended target description to pool-level discriminative questioning, where each question is chosen to narrow the candidate set. On CoIN-Bench, ProCompNav improves Success Rate over interactive baselines with the same minimal input and non-interactive baselines with detailed descriptions, while substantially reducing Response Length. ProCompNav also achieves state-of-the-art Success Rate on TextNav, suggesting that comparative judgment is broadly useful for instance-level navigation among similar distractors. Code is available at this https URL.
>
---
#### [replaced 011] Vision-Based Safe Human-Robot Collaboration with Uncertainty Guarantees
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人机协作任务，旨在提升视觉引导下的人机协作安全性。通过融合不确定性估计与异常检测，提供可验证的安全保障。**

- **链接: [https://arxiv.org/pdf/2604.15221](https://arxiv.org/pdf/2604.15221)**

> **作者:** Jakob Thumm; Marian Frei; Tianle Ni; Matthias Althoff; Marco Pavone
>
> **摘要:** We propose a framework for vision-based human pose estimation and motion prediction that gives conformal prediction guarantees for certifiably safe human-robot collaboration. Our framework combines aleatoric uncertainty estimation with OOD detection for high probabilistic confidence. To integrate our pipeline in certifiable safety frameworks, we propose conformal prediction sets for human motion predictions with high, valid confidence. We evaluate our pipeline on recorded human motion data and a real-world human-robot collaboration setting.
>
---
#### [replaced 012] Hestia: Voxel-Face-Aware Hierarchical Next-Best-View Acquisition for Efficient 3D Reconstruction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D重建任务，解决视点选择效率与鲁棒性问题。提出Hestia方法，通过分层结构、面感知设计等提升覆盖率并降低误差。**

- **链接: [https://arxiv.org/pdf/2508.01014](https://arxiv.org/pdf/2508.01014)**

> **作者:** Cheng-You Lu; Zhuoli Zhuang; Nguyen Thanh Trung Le; Da Xiao; Yu-Cheng Chang; Thomas Do; Srinath Sridhar; Chin-teng Lin
>
> **备注:** Accepted to the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** Advances in 3D reconstruction and novel view synthesis have enabled efficient and photorealistic rendering. However, images for reconstruction are still either largely manual or constrained by simple preplanned trajectories. To address this issue, recent works propose generalizable next-best-view planners that do not require online learning. Nevertheless, robustness and performance remain limited across various shapes. Hence, this study introduces Voxel-Face-Aware Hierarchical Next-Best-View Acquisition for Efficient 3D Reconstruction (Hestia), which addresses the shortcomings of the reinforcement learning-based generalizable approaches for five-degree-of-freedom viewpoint prediction. Hestia systematically improves the planners through four components: a more diverse dataset to promote robustness, a hierarchical structure to manage the high-dimensional continuous action search space, a close-greedy strategy to mitigate spurious correlations, and a face-aware design to avoid overlooking geometry. Experimental results show that Hestia achieves non-marginal improvements, with at least a 4% gain in coverage ratio, while reducing Chamfer Distance by 50% and maintaining real-time inference. In addition, Hestia outperforms prior methods by at least 12% in coverage ratio with a 5-image budget and remains robust to object placement variations. Finally, we demonstrate that Hestia, as a next-best-view planner, is feasible for the real-world application. Our project page is this https URL web.
>
---
#### [replaced 013] ConsistNav: Closing the Action Consistency Gap in Zero-Shot Object Navigation with Semantic Executive Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于零样本目标导航任务，解决导航过程中动作不一致导致的失败问题。提出ConsistNav框架，通过语义执行机制提升导航稳定性与成功率。**

- **链接: [https://arxiv.org/pdf/2605.09869](https://arxiv.org/pdf/2605.09869)**

> **作者:** Haosen Wang; Zhenyang Li; Yinqiang Zhang; Zongqi He; Lutao Jiang; Kai Li; Yizhou Zhao; Liaoyuan Fan; Wenjian Hou; Tingbang Liang; Yibin Wen; Defeng Gu
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Zero-shot object navigation has advanced rapidly with open-vocabulary detectors, image--text models, and language-guided exploration. However, even after current methods detect a plausible target hypothesis, the agent may still oscillate between exploration and pursuit, or abandon the object near success. We identify this failure mode as an action consistency gap: semantic evidence is repeatedly reinterpreted at each step without persistent commitment across the episode. We introduce ConsistNav, a training-free zero-shot ObjectNav framework built around a semantic executive composed of three coordinated modules: Finite-State Executive Controller stages target pursuit through guarded semantic phases; Persistent Candidate Memory accumulates cross-frame target evidence into stable object hypotheses; and Stability-Aware Action Control suppresses rotational stagnation, ineffective pursuit, and unverified stopping. This design changes neither the detector nor the low-level planner; instead, it controls when semantic evidence should influence navigation and when it should be suppressed or revisited. We conduct extensive experiments on HM3D and MP3D, where ConsistNav achieves state-of-the-art results among compared zero-shot ObjectNav methods and improves SR by 11.4% and SPL by 7.9% over the controlled baseline on MP3D. Ablation studies and real-world deployment experiments further demonstrate the effectiveness and robustness of the proposed executive mechanism.
>
---
#### [replaced 014] The OncoReach Stylet for Brachytherapy: Design Evaluation and Pilot Study
- **分类: cs.RO**

- **简介: 该论文属于医疗设备设计任务，旨在解决传统直针限制手术路径的问题，通过开发可操控的OncoReach风格管，提升宫颈癌放射治疗的精准性。**

- **链接: [https://arxiv.org/pdf/2601.13529](https://arxiv.org/pdf/2601.13529)**

> **作者:** Pejman Kheradmand; Kent K. Yamamoto; Emma Webster; Keith Sowards; Gianna Hatheway; Katharine L. Jackson; Sabino Zani Jr.; Julie A. Raffi; Diandra N. Ayala-Peacock; Scott R. Silva; Joanna Deaton Bertram; Yash Chitalia
>
> **摘要:** Cervical cancer accounts for a significant portion of the global cancer burden among women. Interstitial brachytherapy (ISBT) is a standard procedure for treating cervical cancer; it involves placing a radioactive source through a straight hollow needle within or in close proximity to the tumor and surrounding tissue. However, the use of straight needles limits surgical planning to a linear needle path. We present the OncoReach stylet, a handheld, tendon-driven steerable stylet designed for compatibility with standard ISBT 15- and 13-gauge needles. Building upon our prior work, we evaluated design parameters like needle gauge, spherical joint count and spherical joint placement, including an asymmetric disk design to identify a configuration that maximizes bending compliance while retaining axial stiffness. Free space experiments quantified tip deflection across configurations, and a two-tube Cosserat rod model accurately predicted the centerline shape of the needle for most trials. The best performing configuration was integrated into a reusable handheld prototype that enables manual actuation. A patient-derived, multi-composite phantom model of the uterus and pelvis was developed to conduct a pilot study of the OncoReach steerable stylet with one expert user. Results showed the ability to steer from less-invasive, medial entry points to reach the lateral-most targets, underscoring the significance of steerable stylets.
>
---
#### [replaced 015] Sparse ActionGen: Accelerating Diffusion Policy with Real-time Pruning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉运动控制任务，旨在解决扩散策略生成动作速度慢的问题。提出SAG方法，通过实时剪枝和缓存复用实现高效动作生成。**

- **链接: [https://arxiv.org/pdf/2601.12894](https://arxiv.org/pdf/2601.12894)**

> **作者:** Kangye Ji; Jianbo Zhou; Yuan Meng; Ye Li; Hanyun Cui; Zhi Wang
>
> **摘要:** Diffusion Policy has dominated action generation due to its strong capabilities for modeling multi-modal action distributions, but its multi-step denoising processes make it impractical for real-time visuomotor control. Existing caching-based acceleration methods typically rely on $\textit{static}$ schedules that fail to adapt to the $\textit{dynamics}$ of robot-environment interactions, thereby leading to suboptimal performance. In this paper, we propose $\underline{\textbf{S}}$parse $\underline{\textbf{A}}$ction$\underline{\textbf{G}}$en ($\textbf{SAG}$) for extremely sparse action generation. To accommodate the iterative interactions, SAG customizes a rollout-adaptive prune-then-reuse mechanism that first identifies prunable computations globally and then reuses cached activations to substitute them during action diffusion. To capture the rollout dynamics, SAG parameterizes an observation-conditioned diffusion pruner for environment-aware adaptation and instantiates it with a highly parameter- and inference-efficient design for real-time prediction. Furthermore, SAG introduces a one-for-all reusing strategy that reuses activations across both timesteps and blocks in a zig-zag manner, minimizing the global redundancy. Extensive experiments on multiple robotic benchmarks demonstrate that SAG achieves up to 4$\times$ generation speedup without sacrificing performance. Project Page: this https URL.
>
---
#### [replaced 016] frax: Fast Robot Kinematics and Dynamics in JAX
- **分类: cs.RO**

- **简介: 该论文提出frax，一个基于JAX的机器人运动学与动力学库，解决多架构高性能计算问题，支持CPU、GPU和TPU，提升控制与学习效率。**

- **链接: [https://arxiv.org/pdf/2604.04310](https://arxiv.org/pdf/2604.04310)**

> **作者:** Daniel Morton; Marco Pavone
>
> **备注:** ICRA 2026 Workshop on Frontiers of Optimization for Robotics
>
> **摘要:** In robot control, planning, and learning, there is a need for rigid-body dynamics libraries that are highly performant, easy to use, and compatible with CPUs and accelerators. While existing libraries often excel at either low-latency CPU execution or high-throughput GPU workloads, few provide a unified framework that targets multiple architectures without compromising performance or ease-of-use. To address this, we introduce frax, a JAX-based library for robot kinematics and dynamics, providing a high-performance, pure-Python interface across CPU, GPU, and TPU. Via a fully-vectorized approach to robot dynamics, frax enables efficient real-time control and parallelization, while supporting automatic differentiation for optimization-based methods. On CPU, frax achieves low-microsecond computation times suitable for kilohertz control rates, outperforming common libraries in Python and approaching optimized C++ implementations. On GPU, the same code scales to thousands of instances, reaching upwards of 100 million dynamics evaluations per second. We validate performance on a Franka Panda manipulator and a Unitree G1 humanoid, and release frax as an open-source library.
>
---
#### [replaced 017] OpenFrontier: General Navigation with Visual-Language Grounded Frontiers
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人导航任务，解决复杂环境中高效导航问题。提出OpenFrontier框架，无需特定训练，通过视觉语义锚点实现高效导航。**

- **链接: [https://arxiv.org/pdf/2603.05377](https://arxiv.org/pdf/2603.05377)**

> **作者:** Esteban Padilla-Cerdio; Boyang Sun; Marc Pollefeys; Hermann Blum
>
> **摘要:** Open-world navigation requires robots to make decisions in complex everyday environments while adapting to flexible task requirements. Conventional navigation approaches often rely on dense 3D reconstruction and hand-crafted goal metrics, which limits their generalization across tasks and environments. Recent advances in vision-language navigation (VLN) and vision-language-action (VLA) models enable end-to-end policies conditioned on natural language, but typically require interactive training, large-scale data collection, or task-specific fine-tuning with a mobile agent. We formulate navigation as a sparse subgoal identification and reaching problem and observe that providing visual anchoring targets for high-level semantic priors enables highly efficient goal-conditioned navigation. Based on this insight, we select visual frontiers as semantic anchors and propose OpenFrontier, a navigation framework that requires no task-specific training or fine-tuning and seamlessly integrates diverse vision-language prior models. OpenFrontier enables efficient navigation with a lightweight system design, without dense 3D semantic mapping, task-specific policy training, or model fine-tuning. We evaluate OpenFrontier across multiple navigation benchmarks and demonstrate strong zero-shot performance, as well as effective real-world deployment on a mobile robot.
>
---
#### [replaced 018] Empowering Robot Teleoperation: Exploring the Synergies Between Devices and Manipulator Controllers in a Comparative Study
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决 teleoperation 设备与控制器匹配问题。通过实验分析不同设备与控制策略的协同效果，提升机器人操作性能。**

- **链接: [https://arxiv.org/pdf/2511.07720](https://arxiv.org/pdf/2511.07720)**

> **作者:** Yuxuan Zhao; Yuanchen Tang; Jindi Zhang; Hongyu Yu
>
> **摘要:** Robot learning empowers the robot system with human brain-like intelligence to autonomously acquire and adapt skills through experience, enhancing flexibility and adaptability in various environments. Aimed at achieving a similar level of capability in large language models (LLMs) for embodied intelligence, data quality plays a crucial role in training a foundational model with diverse robot skills. In this study, we investigate the collection of data for manipulation tasks using teleoperation devices. Different devices yield varying effects when paired with corresponding controller strategies, including position-based inverse kinematic (IK) control, torque-based inverse dynamic (ID) control, and optimization-based compliant control. Analysis of experimental results suggests the importance of the relationship between teleoperation devices and controllers for real tasks.
>
---
#### [replaced 019] CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出CLARE，解决机器人视觉-语言-动作模型的持续学习问题，无需存储旧数据，通过自适应模块扩展和动态路由机制，实现任务连续学习与知识保留。**

- **链接: [https://arxiv.org/pdf/2601.09512](https://arxiv.org/pdf/2601.09512)**

> **作者:** Ralf Römer; Yi Zhang; Yuming Li; Angela P. Schoellig
>
> **备注:** Accepted to IEEE Robotics and Automation Letters 2026. Project page: this https URL. 11 pages, 9 figures
>
> **摘要:** To teach robots complex manipulation tasks, a common approach is to fine-tune a pre-trained vision-language-action model (VLA) on task-specific data. However, since this recipe updates existing representations, it is unsuitable for long-term operation in the real world, where robots must continually adapt to new tasks and environments while retaining the knowledge they have already acquired. Existing continual learning methods for robotics commonly require storing previous data (exemplars), struggle with long task sequences, or rely on task identifiers for deployment. To address these limitations, we propose CLARE, a general, parameter-efficient framework for exemplar-free continual learning with VLAs. CLARE introduces lightweight modular adapters into selected VLA modules and autonomously expands the model only where necessary when learning a new task, guided by layer-wise feature similarity. During deployment, an autoencoder-based routing mechanism dynamically activates the most relevant adapters without requiring task labels. Through extensive experiments on the LIBERO benchmark and five real-world tasks, we show that CLARE achieves high performance on new tasks without catastrophic forgetting of earlier tasks, significantly outperforming even exemplar-based methods. Code, data, and videos are available at our website: this https URL.
>
---
#### [replaced 020] Whole-body motion planning and safety-critical control for aerial manipulation
- **分类: cs.RO**

- **简介: 该论文属于空中机械臂的运动规划与控制任务，解决复杂环境中安全轨迹生成问题。通过超椭球建模和最大余量规划，提升轨迹安全性与平滑性。**

- **链接: [https://arxiv.org/pdf/2511.02342](https://arxiv.org/pdf/2511.02342)**

> **作者:** Lin Yang; Jinwoo Lee; Domenico Campolo; H. Jin Kim; Jeonghyun Byun
>
> **备注:** Will be presented in 23rd IFAC World Congress 2026
>
> **摘要:** Aerial manipulation combines the maneuverability of multirotors with the dexterity of robotic arms to perform complex tasks in cluttered spaces. Yet planning safe, dynamically feasible trajectories remains difficult due to whole-body collision avoidance and the conservativeness of common geometric abstractions such as bounding boxes or ellipsoids. We present a whole-body motion planning and safety-critical control framework for aerial manipulators built on superquadrics (SQs). Using an SQ-plus-proxy representation, we model both the vehicle and obstacles with differentiable, geometry-accurate surfaces. Leveraging this representation, we introduce a maximum-clearance planner that fuses Voronoi diagrams with an equilibrium-manifold formulation to generate smooth, collision-aware trajectories. We further design a safety-critical controller that jointly enforces thrust limits and collision avoidance via high-order control barrier functions. In simulation, our approach outperforms sampling-based planners in cluttered environments, producing faster, safer, and smoother trajectories and exceeding ellipsoid-based baselines in geometric fidelity. Actual experiments on a physical aerial-manipulation platform confirm feasibility and robustness, demonstrating consistent performance across simulation and hardware settings. The video can be found at this https URL.
>
---
#### [replaced 021] Detecting Heel Strike and toe off Events Using Kinematic Methods and LSTM Models
- **分类: cs.RO**

- **简介: 该论文属于步态分析任务，旨在准确检测足跟触地和脚趾离地事件。通过比较多种运动学方法和LSTM模型，评估其在正常人群中的性能，为康复和外骨骼控制提供可靠方法。**

- **链接: [https://arxiv.org/pdf/2503.00794](https://arxiv.org/pdf/2503.00794)**

> **作者:** Longbin Zhang; Zhizhang Li; Xinyi Fu; Yi Xie; Xiaoyue Yan; Suiyuan Wang; Te Zhang; Hui Zhang; Kailun Yang; Tsung-Lin Wu; Prayook Jatesiktat; Ananda Sidarta; Wei Tech Ang
>
> **摘要:** Accurate gait event detection is crucial for gait analysis, rehabilitation, and assistive technology, particularly in exoskeleton control, where precise identification of stance and swing phases is essential. This study evaluated the performance of seven kinematics-based methods and a Long Short-Term Memory (LSTM) model for detecting heel strike and toe-off events across 4363 gait cycles from 588 able-bodied subjects. The results indicated that while the Zeni et al. method achieved the highest accuracy among kinematics-based approaches, other methods exhibited systematic biases or required dataset-specific tuning. The LSTM model performed comparably to Zeni et al., providing a data-driven alternative without systematic bias. These findings highlight the potential of deep learning-based approaches for gait event detection while emphasizing the need for further validation in clinical populations and across diverse gait conditions. Future research will explore the generalizability of these methods in pathological populations, such as individuals with post-stroke conditions and knee osteoarthritis, as well as their robustness across varied gait conditions and data collection settings to enhance their applicability in rehabilitation and exoskeleton control.
>
---
#### [replaced 022] Simultaneous State Estimation and Online Model Learning in a Soft Robotic System
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于状态估计与模型学习任务，解决软体机器人在未知状态下在线学习弯曲刚度模型的问题。通过结合粒子滤波与高斯过程，实现姿态估计与模型更新。**

- **链接: [https://arxiv.org/pdf/2602.14092](https://arxiv.org/pdf/2602.14092)**

> **作者:** Jan-Hendrik Ewering; Max Bartholdt; Simon F. G. Ehlers; Niklas Wahlström; Thomas B. Schön; Thomas Seel
>
> **备注:** 8 pages, 3 figures, 2 tables, contribution to the International Conference on Information Fusion 2026
>
> **摘要:** Operating complex real-world systems, such as soft robots, can benefit from precise predictive control schemes that require accurate state and model knowledge. This knowledge is typically not available in practical settings and must be inferred from noisy measurements. In particular, it is challenging to simultaneously estimate unknown states and learn a model online from sequentially arriving measurements. In this paper, we show how a recently proposed gray-box system identification tool enables the estimation of a soft robot's current pose while at the same time learning a bending stiffness model. For estimation and learning, we only need a nominal constant-curvature robot model and measurements of the robot's base reactions (e.g., base forces). The estimation scheme -- relying on a marginalized particle filter -- allows us to conveniently interface nominal constant-curvature equations with a Gaussian Process (GP) bending stiffness model to be learned. This, in contrast to estimation via a random walk over stiffness values, enables prediction of bending stiffness and improves overall model quality. We demonstrate, using a real-world soft robot, that the method learns a bending-stiffness model online while accurately estimating the robot's pose. Notably, reduced error in multi-step forward predictions indicates that the learned bending-stiffness GP improves overall model quality.
>
---
#### [replaced 023] An Introduction to Deep Reinforcement and Imitation Learning
- **分类: cs.RO; cs.LG**

- **简介: 本文介绍深度强化学习与模仿学习在具身智能体中的应用，旨在解决复杂序列决策问题。论文聚焦基础算法，提供深入理解。**

- **链接: [https://arxiv.org/pdf/2512.08052](https://arxiv.org/pdf/2512.08052)**

> **作者:** Pedro Santana
>
> **摘要:** Embodied agents, such as robots and virtual characters, must continuously select actions to execute tasks effectively, solving complex sequential decision-making problems. Given the difficulty of designing such controllers manually, learning-based approaches have emerged as promising alternatives, most notably Deep Reinforcement Learning (DRL) and Deep Imitation Learning (DIL). DRL leverages reward signals to optimize behavior, while DIL uses expert demonstrations to guide learning. This document introduces DRL and DIL in the context of embodied agents, adopting a concise, depth-first approach to the literature. It is self-contained, presenting all necessary mathematical and machine learning concepts as they are needed. It is not intended as a survey of the field; rather, it focuses on a small set of foundational algorithms and techniques, prioritizing in-depth understanding over broad coverage. The material ranges from Markov Decision Processes to REINFORCE and Proximal Policy Optimization (PPO) for DRL, and from Behavioral Cloning to Dataset Aggregation (DAgger) and Generative Adversarial Imitation Learning (GAIL) for DIL.
>
---
#### [replaced 024] Flatness-based trajectory planning for 3D overhead cranes with friction compensation and collision avoidance
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于3D天车轨迹规划任务，解决摩擦和碰撞问题。通过微分平坦性方法生成最优轨迹，考虑非线性摩擦与避障，提升运动速度与安全性。**

- **链接: [https://arxiv.org/pdf/2510.24457](https://arxiv.org/pdf/2510.24457)**

> **作者:** Jorge Vicente-Martinez; Edgar Ramirez-Laboreo
>
> **备注:** 6 pages, 8 figures. Final version, after peer review and acceptance, submitted to the 23rd IFAC World Congress
>
> **摘要:** This paper presents an optimal trajectory generation method for 3D overhead cranes by leveraging differential flatness. This framework enables the direct inclusion of complex physical and dynamic constraints, such as nonlinear friction and collision avoidance for both payload and rope. Our approach allows for aggressive movements by constraining payload swing only at the final point. A comparative simulation study validates our approach, demonstrating that neglecting dry friction leads to actuator saturation and collisions. The results show that friction modeling is a fundamental requirement for fast and safe crane trajectories.
>
---
#### [replaced 025] GSDrive: Reinforcing Driving Policies by Multi-mode Future Trajectory Probing with 3D Gaussian Splatting Environment
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，解决E2E驾驶策略优化问题。提出GSDrive框架，结合IL与RL，利用3DGS环境进行多模式轨迹预测与奖励 shaping，提升驾驶性能。**

- **链接: [https://arxiv.org/pdf/2604.28111](https://arxiv.org/pdf/2604.28111)**

> **作者:** Ziang Guo; Chen Min; Xuefeng Zhang; Yixiao Zhou; Shuo Wang; Sifa Zheng; Dzmitry Tsetserukou; Zufeng Zhang
>
> **备注:** 2nd version
>
> **摘要:** End-to-end (E2E) autonomous driving aims to directly map sensory observations to driving actions, but its real-world deployment is hindered by evolving data distributions and the high cost of continual annotation. While combining imitation learning (IL) and reinforcement learning (RL) is a common strategy for policy improvement, conventional RL training relies on delayed, event-based rewards, where policies learn only from catastrophic outcomes such as collisions, leading to premature convergence to suboptimal behaviors. To address these limitations, we propose GSDrive, a framework that uses a differentiable 3D Gaussian Splatting (3DGS) environment for future-aware trajectory probing and reward shaping in E2E driving. GSDrive first learns a multi-mode trajectory probe via IL and then uses RL to evaluate multiple candidate futures in the 3DGS environment, converting their simulated returns into dense shaping rewards for policy optimization. This yields a cyclic hybrid IL-RL training loop, where IL supplies structured future priors and RL provides interactive feedback for iterative refinement. Evaluated on the reconstructed nuScenes dataset, our method outperforms other simulation-based RL approaches in closed-loop experiments. Code is available at this https URL.
>
---
#### [replaced 026] Towards Robotic Dexterous Hand Intelligence: A Survey
- **分类: cs.RO**

- **简介: 该论文属于机器人灵巧手研究领域，旨在系统梳理硬件、控制方法、数据与评估，解决研究碎片化问题，明确未来方向。**

- **链接: [https://arxiv.org/pdf/2605.13925](https://arxiv.org/pdf/2605.13925)**

> **作者:** Weiguang Zhao; Tian Liang; Xihao Guo; Rui Zhang; Irwin King; Kaizhu Huang
>
> **摘要:** Robotic dexterous hands are central to contact-rich manipulation, with rapid progress driven by advances in hardware, sensing, control, simulation, and data generation. However, existing studies are often developed under different assumptions regarding hand embodiments, sensory configurations, task settings, training data, and evaluation protocols, making systematic comparison difficult and obscuring the developmental trajectory of the field. This survey provides a holistic review of dexterous hand research from four complementary aspects. First, we present a hardware-level analysis covering actuation, transmission, perception, and representative hand designs, highlighting the key trade-offs in force capability, compliance, bandwidth, integration, and system complexity. Furthermore, we review control and learning methods for dexterous manipulation from a methodological perspective, grouping representative works by major paradigms and tracing their evolution in chronological order. In addition, we consolidate datasets, modality design, and evaluation practices, which enables methodological progress to be interpreted together with the ways in which it is trained, benchmarked, and assessed. Finally, we discuss the major limitations of current dexterous hand research and summarize the corresponding future directions. By connecting hardware analysis, methodological development, data resources, and evaluation, this survey aims to provide a structured understanding of dexterous hand research and to clarify the most important open challenges for future study.
>
---
#### [replaced 027] QuickLAP: Quick Language-Action Preference Learning for Semi-Autonomous Systems
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出QuickLAP，用于半自主系统中融合语言与物理反馈的奖励学习任务，解决单一模态信息不完整的问题。通过贝叶斯框架实时整合语言和物理反馈，提升奖励学习效果。**

- **链接: [https://arxiv.org/pdf/2511.17855](https://arxiv.org/pdf/2511.17855)**

> **作者:** Jordan Abi Nader; David Lee; Nathaniel Dennler; Andreea Bobu
>
> **摘要:** Robots must learn from both what people do and what they say, but either modality alone is often incomplete: physical corrections are grounded but ambiguous in intent, while language expresses high-level goals but lacks physical grounding. We introduce QuickLAP: Quick Language-Action Preference learning, a Bayesian framework that fuses physical and language feedback to infer reward functions in real time. Our key insight is to treat language as a probabilistic observation over the user's latent preferences, clarifying which reward features matter and how physical corrections should be interpreted. QuickLAP uses Large Language Models (LLMs) to extract reward feature attention masks and preference shifts from free-form utterances, which it integrates with physical feedback in a closed-form update rule. This enables fast, real-time, and robust reward learning that handles ambiguous feedback. In a semi-autonomous driving simulator, QuickLAP reduces reward learning error by over 70% compared to physical-only and heuristic multimodal baselines. A 15-participant user study further validates our approach: participants found QuickLAP significantly more understandable and collaborative, and preferred its learned behavior over baselines. Code is available at this https URL.
>
---
#### [replaced 028] A Hierarchical Spatiotemporal Action Tokenizer for In-Context Imitation Learning in Robotics
- **分类: cs.RO**

- **简介: 该论文属于机器人模仿学习任务，旨在提升上下文模仿学习效果。提出一种分层时空动作分词器，通过多级聚类和时空信息融合，提高动作重构精度与性能。**

- **链接: [https://arxiv.org/pdf/2604.15215](https://arxiv.org/pdf/2604.15215)**

> **作者:** Fawad Javed Fateh; Ali Shah Ali; Murad Popattia; Usman Nizamani; Andrey Konin; M. Zeeshan Zia; Quoc-Huy Tran
>
> **摘要:** We present a novel hierarchical spatiotemporal action tokenizer for in-context imitation learning. We first propose a hierarchical approach, which consists of two successive levels of vector quantization. In particular, the lower level assigns input actions to fine-grained subclusters, while the higher level further maps fine-grained subclusters to clusters. Our hierarchical approach outperforms the non-hierarchical counterpart, while mainly exploiting spatial information by reconstructing input actions. Furthermore, we extend our approach by utilizing both spatial and temporal cues, forming a hierarchical spatiotemporal action tokenizer, namely HiST-AT. Specifically, our hierarchical spatiotemporal approach conducts multi-level clustering, while simultaneously recovering input actions and their associated timestamps. Finally, extensive evaluations on multiple simulation and real robotic manipulation benchmarks show that our approach establishes a new state-of-the-art performance in in-context imitation learning.
>
---
#### [replaced 029] CLOVER: Closed-Loop Value Estimation and Ranking for End-to-End Autonomous Driving Planning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出CLOVER框架，解决自动驾驶规划中的训练-评估不匹配问题，通过生成多样化轨迹并进行评分排序，提升规划性能。**

- **链接: [https://arxiv.org/pdf/2605.15120](https://arxiv.org/pdf/2605.15120)**

> **作者:** Sining Ang; Yuguang Yang; Canyu Chen; Yan Wang
>
> **摘要:** End-to-end autonomous driving planners are commonly trained by imitating a single logged trajectory, yet evaluated by rule-based planning metrics that measure safety, feasibility, progress, and comfort. This creates a training--evaluation mismatch: trajectories close to the logged path may violate planning rules, while alternatives farther from the demonstration can remain valid and high-scoring. The mismatch is especially limiting for proposal-selection planners, whose performance depends on candidate-set coverage and scorer ranking quality. We propose CLOVER, a Closed-LOop Value Estimation and Ranking framework for end-to-end autonomous driving planning. CLOVER follows a lightweight generator--scorer formulation: a generator produces diverse candidate trajectories, and a scorer predicts planning-metric sub-scores to rank them at inference time. To expand proposal support beyond single-trajectory imitation, CLOVER constructs evaluator-filtered pseudo-expert trajectories and trains the generator with set-level coverage supervision. It then performs conservative closed-loop self-distillation: the scorer is fitted to true evaluator sub-scores on generated proposals, while the generator is refined toward teacher-selected top-$k$ and vector-Pareto targets with stability regularization. We analyze when an imperfect scorer can improve the generator, showing that scorer-mediated refinement is reliable when scorer-selected targets are enriched under the true evaluator and updates remain conservative. On NAVSIM, CLOVER achieves 94.5 PDMS and 90.4 EPDMS, establishing a new state of the art. On the more challenging NavHard split, it obtains 48.3 EPDMS, matching the strongest reported result. On supplementary nuScenes open-loop evaluation, CLOVER achieves the lowest L2 error and collision rate among compared methods. Code data will be released at this https URL.
>
---
