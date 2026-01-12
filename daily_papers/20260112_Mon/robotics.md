# 机器人 cs.RO

- **最新发布 17 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] Motion Compensation for Real Time Ultrasound Scanning in Robotically Assisted Prostate Biopsy Procedures
- **分类: cs.RO**

- **简介: 该论文属于医学图像处理任务，旨在解决机器人辅助前列腺活检中的运动补偿问题。通过开发机器人系统，实现精准的前列腺超声扫描与重建，提高 biopsy 的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2601.05661v1](https://arxiv.org/pdf/2601.05661v1)**

> **作者:** Matija Markulin; Luka Matijević; Luka Siktar; Janko Jurdana; Branimir Caran; Marko Švaco; Filip Šuligoj; Bojan Šekoranja
>
> **备注:** Submitted for ICRA 2026
>
> **摘要:** Prostate cancer is one of the most common types of cancer in men. Its diagnosis by biopsy requires a high level of expertise and precision from the surgeon, so the results are highly operator-dependent. The aim of this work is to develop a robotic system for assisted ultrasound (US) examination of the prostate, a prebiopsy step that could reduce the dexterity requirements and enable faster, more accurate and more available prostate biopsy. We developed and validated a laboratory setup with a collaborative robotic arm that can autonomously scan a prostate phantom and attached the phantom to a medical robotic arm that mimics the patient's movements. The scanning robot keeps the relative position of the US probe and the prostate constant, ensuring a consistent and robust approach to reconstructing the prostate. To reconstruct the prostate, each slice is segmented to generate a series of prostate contours converted into a 3D point cloud used for biopsy planning. The average scan time of the prostate was 30 s, and the average 3D reconstruction of the prostate took 3 s. We performed four motion scenarios: the phantom was scanned in a stationary state (S), with horizontal motion (H), with vertical motion (V), and with a combination of the two (C). System validation is performed by registering the prostate point cloud reconstructions acquired during different motions (H, V, C) with those obtained in the stationary state. ICP registration with a threshold of 0.8 mm yields mean 83.2\% fitness and 0.35 mm RMSE for S-H registration, 84.1\% fitness and 0.37 mm RMSE for S-V registration and 79.4\% fitness and 0.37 mm RMSE for S-C registration. Due to the elastic and soft material properties of the prostate phantom, the maximum robot tracking error was 3 mm, which can be sufficient for prostate biopsy according to medical literature. The maximum delay in motion compensation was 0.5 s.
>
---
#### [new 002] Modular Autonomy with Conversational Interaction: An LLM-driven Framework for Decision Making in Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自主驾驶任务，解决人机交互中的语言指令映射问题。通过集成LLM与Autoware，实现自然语言命令的高效、安全执行。**

- **链接: [https://arxiv.org/pdf/2601.05806v1](https://arxiv.org/pdf/2601.05806v1)**

> **作者:** Marvin Seegert; Korbinian Moller; Johannes Betz
>
> **备注:** Submitted to the IEEE Intelligent Vehicles Symposium (IV 2026), Detroit, MI, United States
>
> **摘要:** Recent advancements in Large Language Models (LLMs) offer new opportunities to create natural language interfaces for Autonomous Driving Systems (ADSs), moving beyond rigid inputs. This paper addresses the challenge of mapping the complexity of human language to the structured action space of modular ADS software. We propose a framework that integrates an LLM-based interaction layer with Autoware, a widely used open-source software. This system enables passengers to issue high-level commands, from querying status information to modifying driving behavior. Our methodology is grounded in three key components: a taxonomization of interaction categories, an application-centric Domain Specific Language (DSL) for command translation, and a safety-preserving validation layer. A two-stage LLM architecture ensures high transparency by providing feedback based on the definitive execution status. Evaluation confirms the system's timing efficiency and translation robustness. Simulation successfully validated command execution across all five interaction categories. This work provides a foundation for extensible, DSL-assisted interaction in modular and safety-conscious autonomy stacks.
>
---
#### [new 003] Learning specifications for reactive synthesis with safety constraints
- **分类: cs.RO; cs.FL**

- **简介: 该论文属于机器人任务学习，解决动态环境中安全执行复杂任务的问题。通过学习形式化规范并结合多目标优化，生成满足安全约束的策略。**

- **链接: [https://arxiv.org/pdf/2601.05533v1](https://arxiv.org/pdf/2601.05533v1)**

> **作者:** Kandai Watanabe; Nicholas Renninger; Sriram Sankaranarayanan; Morteza Lahijanian
>
> **摘要:** This paper presents a novel approach to learning from demonstration that enables robots to autonomously execute complex tasks in dynamic environments. We model latent tasks as probabilistic formal languages and introduce a tailored reactive synthesis framework that balances robot costs with user task preferences. Our methodology focuses on safety-constrained learning and inferring formal task specifications as Probabilistic Deterministic Finite Automata (PDFA). We adapt existing evidence-driven state merging algorithms and incorporate safety requirements throughout the learning process to ensure that the learned PDFA always complies with safety constraints. Furthermore, we introduce a multi-objective reactive synthesis algorithm that generates deterministic strategies that are guaranteed to satisfy the PDFA task while optimizing the trade-offs between user preferences and robot costs, resulting in a Pareto front of optimal solutions. Our approach models the interaction as a two-player game between the robot and the environment, accounting for dynamic changes. We present a computationally-tractable value iteration algorithm to generate the Pareto front and the corresponding deterministic strategies. Comprehensive experimental results demonstrate the effectiveness of our algorithms across various robots and tasks, showing that the learned PDFA never includes unsafe behaviors and that synthesized strategies consistently achieve the task while meeting both the robot cost and user-preference requirements.
>
---
#### [new 004] Intent at a Glance: Gaze-Guided Robotic Manipulation via Foundation Models
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决如何通过眼神引导机器人执行操作的问题。工作是提出GAMMA系统，结合眼动追踪与视觉语言模型，实现意图理解与自主操作。**

- **链接: [https://arxiv.org/pdf/2601.05336v1](https://arxiv.org/pdf/2601.05336v1)**

> **作者:** Tracey Yee Hsin Tay; Xu Yan; Jonathan Ouyang; Daniel Wu; William Jiang; Jonathan Kao; Yuchen Cui
>
> **备注:** Accepted to 2025 RSS Robot Planning in the Era of Foundation Models (FM4RoboPlan) Workshop
>
> **摘要:** Designing intuitive interfaces for robotic control remains a central challenge in enabling effective human-robot interaction, particularly in assistive care settings. Eye gaze offers a fast, non-intrusive, and intent-rich input modality, making it an attractive channel for conveying user goals. In this work, we present GAMMA (Gaze Assisted Manipulation for Modular Autonomy), a system that leverages ego-centric gaze tracking and a vision-language model to infer user intent and autonomously execute robotic manipulation tasks. By contextualizing gaze fixations within the scene, the system maps visual attention to high-level semantic understanding, enabling skill selection and parameterization without task-specific training. We evaluate GAMMA on a range of table-top manipulation tasks and compare it against baseline gaze-based control without reasoning. Results demonstrate that GAMMA provides robust, intuitive, and generalizable control, highlighting the potential of combining foundation models and gaze for natural and scalable robot autonomy. Project website: https://gamma0.vercel.app/
>
---
#### [new 005] InsSo3D: Inertial Navigation System and 3D Sonar SLAM for turbid environment inspection
- **分类: cs.RO**

- **简介: 该论文属于SLAM任务，旨在解决浑浊环境中三维定位与建图问题。通过融合3D声呐和惯性导航系统，提出InsSo3D方法，有效纠正里程计误差，实现高精度地图构建。**

- **链接: [https://arxiv.org/pdf/2601.05805v1](https://arxiv.org/pdf/2601.05805v1)**

> **作者:** Simon Archieri; Ahmet Cinar; Shu Pan; Jonatan Scharff Willners; Michele Grimald; Ignacio Carlucho; Yvan Petillot
>
> **摘要:** This paper presents InsSo3D, an accurate and efficient method for large-scale 3D Simultaneous Localisation and Mapping (SLAM) using a 3D Sonar and an Inertial Navigation System (INS). Unlike traditional sonar, which produces 2D images containing range and azimuth information but lacks elevation information, 3D Sonar produces a 3D point cloud, which therefore does not suffer from elevation ambiguity. We introduce a robust and modern SLAM framework adapted to the 3D Sonar data using INS as prior, detecting loop closure and performing pose graph optimisation. We evaluated InsSo3D performance inside a test tank with access to ground truth data and in an outdoor flooded quarry. Comparisons to reference trajectories and maps obtained from an underwater motion tracking system and visual Structure From Motion (SFM) demonstrate that InsSo3D efficiently corrects odometry drift. The average trajectory error is below 21cm during a 50-minute-long mission, producing a map of 10m by 20m with a 9cm average reconstruction error, enabling safe inspection of natural or artificial underwater structures even in murky water conditions.
>
---
#### [new 006] Assembling Solar Panels by Dual Robot Arms Towards Full Autonomous Lunar Base Construction
- **分类: cs.RO**

- **简介: 论文研究了双机械臂自主组装太阳能板的任务，旨在解决月球基地建设中的自动化装配问题。通过集成视觉、控制与硬件系统，实现太阳能板的高效连接。**

- **链接: [https://arxiv.org/pdf/2601.05491v1](https://arxiv.org/pdf/2601.05491v1)**

> **作者:** Luca Nunziante; Kentaro Uno; Gustavo H. Diaz; Shreya Santra; Alessandro De Luca; Kazuya Yoshida
>
> **备注:** This is the authors' version of a paper accepted for publication in IEEE/SICE International Symposium on System Integration (SII), 2025, (c) IEEE
>
> **摘要:** Since the successful Apollo program, humanity is once again aiming to return to the Moon for scientific discovery, resource mining, and inhabitation. Upcoming decades focus on building a lunar outpost, with robotic systems playing a crucial role to safely and efficiently establish essential infrastructure such as solar power generating towers. Similar to the construction of the International Space Station (ISS), shipping necessary components via modules and assembling them in situ should be a practical scenario. In this context, this paper focuses on the integration of vision, control, and hardware systems within an autonomous sequence for a dual-arm robot system. We explore a perception and control pipeline specifically designed for assembling solar panel modules, one of the benchmark tasks. Ad hoc hardware was designed and tested in real-world experiments. A mock-up of modular solar panels and active-passive connectors are employed, with the control of this grappling fixture integrated into the proposed pipeline. The successful implementation of our method demonstrates that the two robot manipulators can effectively connect arbitrarily placed panels, highlighting the seamless integration of vision, control, and hardware systems in complex space applications.
>
---
#### [new 007] Intelligent Singularity Avoidance in UR10 Robotic Arm Path Planning Using Hybrid Fuzzy Logic and Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人路径规划任务，解决UR10机械臂在运动中遇到奇异点的问题。通过融合模糊逻辑与强化学习，实现有效避障和路径优化。**

- **链接: [https://arxiv.org/pdf/2601.05836v1](https://arxiv.org/pdf/2601.05836v1)**

> **作者:** Sheng-Kai Chen; Jyh-Horng Wu
>
> **备注:** Published in TANET 2025 (Paper No. T0404)
>
> **摘要:** This paper presents a comprehensive approach to singularity detection and avoidance in UR10 robotic arm path planning through the integration of fuzzy logic safety systems and reinforcement learning algorithms. The proposed system addresses critical challenges in robotic manipulation where singularities can cause loss of control and potential equipment damage. Our hybrid approach combines real-time singularity detection using manipulability measures, condition number analysis, and fuzzy logic decision-making with a stable reinforcement learning framework for adaptive path planning. Experimental results demonstrate a 90% success rate in reaching target positions while maintaining safe distances from singular configurations. The system integrates PyBullet simulation for training data collection and URSim connectivity for real-world deployment.
>
---
#### [new 008] TOSC: Task-Oriented Shape Completion for Open-World Dexterous Grasp Generation from Partial Point Clouds
- **分类: cs.RO**

- **简介: 该论文属于任务导向的形状补全与灵巧抓取任务，解决部分点云下抓取困难的问题。通过生成候选形状并优化，提升抓取效果。**

- **链接: [https://arxiv.org/pdf/2601.05499v1](https://arxiv.org/pdf/2601.05499v1)**

> **作者:** Weishang Wu; Yifei Shi; Zhiping Cai
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Task-oriented dexterous grasping remains challenging in robotic manipulations of open-world objects under severe partial observation, where significant missing data invalidates generic shape completion. In this paper, to overcome this limitation, we study Task-Oriented Shape Completion, a new task that focuses on completing the potential contact regions rather than the entire shape. We argue that shape completion for grasping should be explicitly guided by the downstream manipulation task. To achieve this, we first generate multiple task-oriented shape completion candidates by leveraging the zero-shot capabilities of object functional understanding from several pre-trained foundation models. A 3D discriminative autoencoder is then proposed to evaluate the plausibility of each generated candidate and optimize the most plausible one from a global perspective. A conditional flow-matching model named FlowGrasp is developed to generate task-oriented dexterous grasps from the optimized shape. Our method achieves state-of-the-art performance in task-oriented dexterous grasping and task-oriented shape completion, improving the Grasp Displacement and the Chamfer Distance over the state-of-the-art by 16.17\% and 55.26%, respectively. In particular, it shows good capabilities in grasping objects with severe missing data. It also demonstrates good generality in handling open-set categories and tasks.
>
---
#### [new 009] EvoQRE: Modeling Bounded Rationality in Safety-Critical Traffic Simulation via Evolutionary Quantal Response Equilibrium
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于交通仿真任务，旨在解决自动驾驶中人类驾驶员有限理性建模问题。通过引入EvoQRE框架，结合QRE与进化动态，提升仿真真实性和安全性。**

- **链接: [https://arxiv.org/pdf/2601.05653v1](https://arxiv.org/pdf/2601.05653v1)**

> **作者:** Phu-Hoa Pham; Chi-Nguyen Tran; Duy-Minh Dao-Sy; Phu-Quy Nguyen-Lam; Trung-Kiet Huynh
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Existing traffic simulation frameworks for autonomous vehicles typically rely on imitation learning or game-theoretic approaches that solve for Nash or coarse correlated equilibria, implicitly assuming perfectly rational agents. However, human drivers exhibit bounded rationality, making approximately optimal decisions under cognitive and perceptual constraints. We propose EvoQRE, a principled framework for modeling safety-critical traffic interactions as general-sum Markov games solved via Quantal Response Equilibrium (QRE) and evolutionary game dynamics. EvoQRE integrates a pre-trained generative world model with entropy-regularized replicator dynamics, capturing stochastic human behavior while maintaining equilibrium structure. We provide rigorous theoretical results, proving that the proposed dynamics converge to Logit-QRE under a two-timescale stochastic approximation with an explicit convergence rate of O(log k / k^{1/3}) under weak monotonicity assumptions. We further extend QRE to continuous action spaces using mixture-based and energy-based policy representations. Experiments on the Waymo Open Motion Dataset and nuPlan benchmark demonstrate that EvoQRE achieves state-of-the-art realism, improved safety metrics, and controllable generation of diverse safety-critical scenarios through interpretable rationality parameters.
>
---
#### [new 010] PRISM: Protocol Refinement through Intelligent Simulation Modeling
- **分类: cs.RO; cs.AI; cs.MA; q-bio.QM**

- **简介: 该论文提出PRISM框架，解决自动化实验协议设计与执行问题，通过语言模型生成并验证实验步骤，实现机器人自动操作。**

- **链接: [https://arxiv.org/pdf/2601.05356v1](https://arxiv.org/pdf/2601.05356v1)**

> **作者:** Brian Hsu; Priyanka V Setty; Rory M Butler; Ryan Lewis; Casey Stone; Rebecca Weinberg; Thomas Brettin; Rick Stevens; Ian Foster; Arvind Ramanathan
>
> **备注:** 43 pages, 8 figures, submitted to RSC Digital Discovery. Equal contribution: B. Hsu, P.V. Setty, R.M. Butler. Corresponding author: A. Ramanathan
>
> **摘要:** Automating experimental protocol design and execution remains as a fundamental bottleneck in realizing self-driving laboratories. We introduce PRISM (Protocol Refinement through Intelligent Simulation Modeling), a framework that automates the design, validation, and execution of experimental protocols on a laboratory platform composed of off-the-shelf robotic instruments. PRISM uses a set of language-model-based agents that work together to generate and refine experimental steps. The process begins with automatically gathering relevant procedures from web-based sources describing experimental workflows. These are converted into structured experimental steps (e.g., liquid handling steps, deck layout and other related operations) through a planning, critique, and validation loop. The finalized steps are translated into the Argonne MADSci protocol format, which provides a unified interface for coordinating multiple robotic instruments (Opentrons OT-2 liquid handler, PF400 arm, Azenta plate sealer and peeler) without requiring human intervention between steps. To evaluate protocol-generation performance, we benchmarked both single reasoning models and multi-agent workflow across constrained and open-ended prompting paradigms. The resulting protocols were validated in a digital-twin environment built in NVIDIA Omniverse to detect physical or sequencing errors before execution. Using Luna qPCR amplification and Cell Painting as case studies, we demonstrate PRISM as a practical end-to-end workflow that bridges language-based protocol generation, simulation-based validation, and automated robotic execution.
>
---
#### [new 011] Goal Force: Teaching Video Models To Accomplish Physics-Conditioned Goals
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视频生成任务，旨在解决视频模型目标定义困难的问题。通过引入力向量和动态过程作为目标，训练模型实现物理条件下的目标达成。**

- **链接: [https://arxiv.org/pdf/2601.05848v1](https://arxiv.org/pdf/2601.05848v1)**

> **作者:** Nate Gillman; Yinghua Zhou; Zitian Tang; Evan Luo; Arjan Chakravarthy; Daksh Aggarwal; Michael Freeman; Charles Herrmann; Chen Sun
>
> **备注:** Code and interactive demos at https://goal-force.github.io/
>
> **摘要:** Recent advancements in video generation have enabled the development of ``world models'' capable of simulating potential futures for robotics and planning. However, specifying precise goals for these models remains a challenge; text instructions are often too abstract to capture physical nuances, while target images are frequently infeasible to specify for dynamic tasks. To address this, we introduce Goal Force, a novel framework that allows users to define goals via explicit force vectors and intermediate dynamics, mirroring how humans conceptualize physical tasks. We train a video generation model on a curated dataset of synthetic causal primitives-such as elastic collisions and falling dominos-teaching it to propagate forces through time and space. Despite being trained on simple physics data, our model exhibits remarkable zero-shot generalization to complex, real-world scenarios, including tool manipulation and multi-object causal chains. Our results suggest that by grounding video generation in fundamental physical interactions, models can emerge as implicit neural physics simulators, enabling precise, physics-aware planning without reliance on external engines. We release all datasets, code, model weights, and interactive video demos at our project page.
>
---
#### [new 012] Inverting Non-Injective Functions with Twin Neural Network Regression
- **分类: cs.LG; cs.RO**

- **简介: 该论文研究非单射函数的逆问题，属于函数逆向建模任务。针对非单射函数不可逆的问题，提出一种基于孪生神经网络回归的方法，结合k近邻搜索，实现对非单射函数的确定性逆推。**

- **链接: [https://arxiv.org/pdf/2601.05378v1](https://arxiv.org/pdf/2601.05378v1)**

> **作者:** Sebastian J. Wetzel
>
> **摘要:** Non-injective functions are not invertible. However, non-injective functions can be restricted to sub-domains on which they are locally injective and surjective and thus invertible if the dimensionality between input and output spaces are the same. Further, even if the dimensionalities do not match it is often possible to choose a preferred solution from many possible solutions. Twin neural network regression is naturally capable of incorporating these properties to invert non-injective functions. Twin neural network regression is trained to predict adjustments to well known input variables $\mathbf{x}^{\text{anchor}}$ to obtain an estimate for an unknown $\mathbf{x}^{\text{new}}$ under a change of the target variable from $\mathbf{y}^{\text{anchor}}$ to $\mathbf{y}^{\text{new}}$. In combination with k-nearest neighbor search, I propose a deterministic framework that finds input parameters to a given target variable of non-injective functions. The method is demonstrated by inverting non-injective functions describing toy problems and robot arm control that are a) defined by data or b) known as mathematical formula.
>
---
#### [new 013] DexterCap: An Affordable and Automated System for Capturing Dexterous Hand-Object Manipulation
- **分类: cs.GR; cs.AI; cs.RO**

- **简介: 该论文属于手部动作捕捉任务，旨在解决复杂手物交互的精准捕捉问题。提出DexterCap系统，实现低成本、自动化的手部动作捕获与重建。**

- **链接: [https://arxiv.org/pdf/2601.05844v1](https://arxiv.org/pdf/2601.05844v1)**

> **作者:** Yutong Liang; Shiyi Xu; Yulong Zhang; Bowen Zhan; He Zhang; Libin Liu
>
> **备注:** 12 pages, 12 figures
>
> **摘要:** Capturing fine-grained hand-object interactions is challenging due to severe self-occlusion from closely spaced fingers and the subtlety of in-hand manipulation motions. Existing optical motion capture systems rely on expensive camera setups and extensive manual post-processing, while low-cost vision-based methods often suffer from reduced accuracy and reliability under occlusion. To address these challenges, we present DexterCap, a low-cost optical capture system for dexterous in-hand manipulation. DexterCap uses dense, character-coded marker patches to achieve robust tracking under severe self-occlusion, together with an automated reconstruction pipeline that requires minimal manual effort. With DexterCap, we introduce DexterHand, a dataset of fine-grained hand-object interactions covering diverse manipulation behaviors and objects, from simple primitives to complex articulated objects such as a Rubik's Cube. We release the dataset and code to support future research on dexterous hand-object interaction.
>
---
#### [new 014] Safety Not Found (404): Hidden Risks of LLM-Based Robotics Decision Making
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于机器人决策安全研究，旨在评估LLM在安全关键场景中的可靠性。通过设计任务测试其决策风险，发现模型存在严重安全隐患，指出当前LLM不适合直接用于安全系统。**

- **链接: [https://arxiv.org/pdf/2601.05529v1](https://arxiv.org/pdf/2601.05529v1)**

> **作者:** Jua Han; Jaeyoon Seo; Jungbin Min; Jean Oh; Jihie Kim
>
> **摘要:** One mistake by an AI system in a safety-critical setting can cost lives. As Large Language Models (LLMs) become integral to robotics decision-making, the physical dimension of risk grows; a single wrong instruction can directly endanger human safety. This paper addresses the urgent need to systematically evaluate LLM performance in scenarios where even minor errors are catastrophic. Through a qualitative evaluation of a fire evacuation scenario, we identified critical failure cases in LLM-based decision-making. Based on these, we designed seven tasks for quantitative assessment, categorized into: Complete Information, Incomplete Information, and Safety-Oriented Spatial Reasoning (SOSR). Complete information tasks utilize ASCII maps to minimize interpretation ambiguity and isolate spatial reasoning from visual processing. Incomplete information tasks require models to infer missing context, testing for spatial continuity versus hallucinations. SOSR tasks use natural language to evaluate safe decision-making in life-threatening contexts. We benchmark various LLMs and Vision-Language Models (VLMs) across these tasks. Beyond aggregate performance, we analyze the implications of a 1% failure rate, highlighting how "rare" errors escalate into catastrophic outcomes. Results reveal serious vulnerabilities: several models achieved a 0% success rate in ASCII navigation, while in a simulated fire drill, models instructed robots to move toward hazardous areas instead of emergency exits. Our findings lead to a sobering conclusion: current LLMs are not ready for direct deployment in safety-critical systems. A 99% accuracy rate is dangerously misleading in robotics, as it implies one out of every hundred executions could result in catastrophic harm. We demonstrate that even state-of-the-art models cannot guarantee safety, and absolute reliance on them creates unacceptable risks.
>
---
#### [new 015] Mobile Robot Localization Using a Novel Whisker-Like Sensor
- **分类: physics.app-ph; cs.RO**

- **简介: 该论文属于机器人定位任务，解决在视觉不可靠环境下精准定位问题。通过新型触须传感器和虚拟模型，实现接触点估计与机器人定位，误差小于7毫米。**

- **链接: [https://arxiv.org/pdf/2601.05612v1](https://arxiv.org/pdf/2601.05612v1)**

> **作者:** Prasanna K. Routray; Basak Sakcak; Steven M. LaValle; Manivannan M
>
> **摘要:** Whisker-like touch sensors offer unique advantages for short-range perception in environments where visual and long-range sensing are unreliable, such as confined, cluttered, or low-visibility settings. This paper presents a framework for estimating contact points and robot localization in a known planar environment using a single whisker sensor. We develop a family of virtual sensor models. Each model maps robot configurations to sensor observations and enables structured reasoning through the concept of preimages - the set of robot states consistent with a given observation. The notion of virtual sensor models serves as an abstraction to reason about state uncertainty without dependence on physical implementation. By combining sensor observations with a motion model, we estimate the contact point. Iterative estimation then enables reconstruction of obstacle boundaries. Furthermore, intersecting states inferred from current observations with forward-projected states from previous steps allow accurate robot localization without relying on vision or external systems. The framework supports both deterministic and possibilistic formulations and is validated through simulation and physical experiments using a low-cost, 3D printed, Hall-effect-based whisker sensor. Results demonstrate accurate contact estimation and localization with errors under 7 mm, demonstrating the potential of whisker-based sensing as a lightweight, adaptable complement to vision-based navigation.
>
---
#### [new 016] SceneFoundry: Generating Interactive Infinite 3D Worlds
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出SceneFoundry，用于生成可交互的大型3D场景，解决真实环境生成难题。属于环境生成任务，旨在为机器人学习提供物理真实的虚拟环境。**

- **链接: [https://arxiv.org/pdf/2601.05810v1](https://arxiv.org/pdf/2601.05810v1)**

> **作者:** ChunTeng Chen; YiChen Hsu; YiWen Liu; WeiFang Sun; TsaiChing Ni; ChunYi Lee; Min Sun; YuanFu Yang
>
> **备注:** 15 pages
>
> **摘要:** The ability to automatically generate large-scale, interactive, and physically realistic 3D environments is crucial for advancing robotic learning and embodied intelligence. However, existing generative approaches often fail to capture the functional complexity of real-world interiors, particularly those containing articulated objects with movable parts essential for manipulation and navigation. This paper presents SceneFoundry, a language-guided diffusion framework that generates apartment-scale 3D worlds with functionally articulated furniture and semantically diverse layouts for robotic training. From natural language prompts, an LLM module controls floor layout generation, while diffusion-based posterior sampling efficiently populates the scene with articulated assets from large-scale 3D repositories. To ensure physical usability, SceneFoundry employs differentiable guidance functions to regulate object quantity, prevent articulation collisions, and maintain sufficient walkable space for robotic navigation. Extensive experiments demonstrate that our framework generates structurally valid, semantically coherent, and functionally interactive environments across diverse scene types and conditions, enabling scalable embodied AI research.
>
---
#### [new 017] FlyPose: Towards Robust Human Pose Estimation From Aerial Views
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于人体姿态估计任务，解决从无人机视角准确检测和估计人体姿态的问题。通过多数据集训练提升模型性能，并发布新数据集FlyPose-104。**

- **链接: [https://arxiv.org/pdf/2601.05747v1](https://arxiv.org/pdf/2601.05747v1)**

> **作者:** Hassaan Farooq; Marvin Brenner; Peter St\ütz
>
> **备注:** 11 pages, 9 figures, IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are increasingly deployed in close proximity to humans for applications such as parcel delivery, traffic monitoring, disaster response and infrastructure inspections. Ensuring safe and reliable operation in these human-populated environments demands accurate perception of human poses and actions from an aerial viewpoint. This perspective challenges existing methods with low resolution, steep viewing angles and (self-)occlusion, especially if the application demands realtime feasibile models. We train and deploy FlyPose, a lightweight top-down human pose estimation pipeline for aerial imagery. Through multi-dataset training, we achieve an average improvement of 6.8 mAP in person detection across the test-sets of Manipal-UAV, VisDrone, HIT-UAV as well as our custom dataset. For 2D human pose estimation we report an improvement of 16.3 mAP on the challenging UAV-Human dataset. FlyPose runs with an inference latency of ~20 milliseconds including preprocessing on a Jetson Orin AGX Developer Kit and is deployed onboard a quadrotor UAV during flight experiments. We also publish FlyPose-104, a small but challenging aerial human pose estimation dataset, that includes manual annotations from difficult aerial perspectives: https://github.com/farooqhassaan/FlyPose.
>
---
## 更新

#### [replaced 001] Anomaly detection for generic failure monitoring in robotic assembly, screwing and manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人故障监测任务，旨在解决不同机器人控制策略和任务间的异常检测问题。通过分析多模态时间序列数据，评估多种自编码器方法的泛化能力与数据效率。**

- **链接: [https://arxiv.org/pdf/2509.26308v2](https://arxiv.org/pdf/2509.26308v2)**

> **作者:** Niklas Grambow; Lisa-Marie Fenner; Felipe Kempkes; Philip Hotz; Dingyuan Wan; Jörg Krüger; Kevin Haninger
>
> **摘要:** Out-of-distribution states in robot manipulation often lead to unpredictable robot behavior or task failure, limiting success rates and increasing risk of damage. Anomaly detection (AD) can identify deviations from expected patterns in data, which can be used to trigger failsafe behaviors and recovery strategies. Prior work has applied data-driven AD on time series data for specific robotic tasks, however the transferability of an AD approach between different robot control strategies and task types has not been shown. Leveraging time series data, such as force/torque signals, allows to directly capture robot-environment interactions, crucial for manipulation and online failure detection. Their broad availability, high sampling rates, and low dimensionality enable high temporal resolution and efficient processing. As robotic tasks can have widely signal characteristics and requirements, AD methods which can be applied in the same way to a wide range of tasks is needed, ideally with good data efficiency. We examine three industrial tasks, each presenting several anomalies. Test scenarios in robotic cabling, screwing, and sanding are built, and multi-modal time series data is gathered. Several autoencoder-based methods are compared, and we evaluate the generalization across different tasks and control methods (diffusion policy-, position-, and impedance-controlled). This allows us to validate the integration of AD in complex tasks involving tighter tolerances and variation from both the robot and its environment. Additionally, we evaluate data efficiency, detection latency, and task characteristics which support robust detection. The results indicate reliable detection with AUROC above 0.96 in failures in the cabling and screwing task, such as incorrect or misaligned parts. In the polishing task, only severe failures were reliably detected, while more subtle failures remained undetected.
>
---
#### [replaced 002] Volume-Consistent Kneading-Based Deformation Manufacturing for Material-Efficient Shaping
- **分类: cs.RO**

- **简介: 该论文属于制造任务，旨在解决传统制造中的材料浪费和表面质量问题。提出一种体积一致的揉捏成形方法，实现高效、低废的三维变形制造。**

- **链接: [https://arxiv.org/pdf/2511.22042v3](https://arxiv.org/pdf/2511.22042v3)**

> **作者:** Lei Li; Jiale Gong; Ziyang Li; Hong Wang
>
> **备注:** 39 pages, 31 figures
>
> **摘要:** Conventional subtractive manufacturing inevitably involves material loss during geometric realization, while additive manufacturing still suffers from limitations in surface quality, process continuity, and productivity when fabricating complex geometries. To address these challenges, this paper proposes a volume-consistent kneading-based forming method for plastic materials, enabling continuous and controllable three-dimensional deformation under mass conservation. An integrated kneading-based manufacturing system is developed, in which geometry-aware kneading command generation, layer-wise kneading execution, and in-process point-cloud scanning are tightly coupled to form a closed-loop workflow of scanning, forming, and feedback compensation. Target geometries are analyzed through layer-wise point-cloud processing and classified into enveloping and non-enveloping types. Accordingly, an Envelope Shaping First strategy and a Similar Gradient Method are adopted to ensure stable material flow and continuous deformation. An RMSE-based compensation scheme is further introduced to correct systematic geometric deviations induced by elastic rebound and material redistribution. Experimental validation on five representative geometries demonstrates high geometric fidelity, with material utilization consistently exceeding 98%. The results indicate that kneading-based forming provides a promising alternative manufacturing paradigm for low-waste, customizable production.
>
---
#### [replaced 003] Hierarchical GNN-Based Multi-Agent Learning for Dynamic Queue-Jump Lane and Emergency Vehicle Corridor Formation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于智能交通任务，旨在解决紧急车辆通行效率低的问题。通过构建基于GNN的多智能体学习框架，实现动态应急通道的高效形成与协调。**

- **链接: [https://arxiv.org/pdf/2601.04177v2](https://arxiv.org/pdf/2601.04177v2)**

> **作者:** Haoran Su
>
> **备注:** 16 Pages, 5 Figures, 9 Tables, submitted to IEEE TITS
>
> **摘要:** Emergency vehicles require rapid passage through congested traffic, yet existing strategies fail to adapt to dynamic conditions. We propose a novel hierarchical graph neural network (GNN)-based multi-agent reinforcement learning framework to coordinate connected vehicles for emergency corridor formation. Our approach uses a high-level planner for global strategy and low-level controllers for trajectory execution, utilizing graph attention networks to scale with variable agent counts. Trained via Multi-Agent Proximal Policy Optimization (MAPPO), the system reduces emergency vehicle travel time by 28.3% compared to baselines and 44.6% compared to uncoordinated traffic in simulations. The design achieves near-zero collision rates (0.3%) while maintaining 81% of background traffic efficiency. Ablation and generalization studies confirm the framework's robustness across diverse scenarios. These results demonstrate the effectiveness of combining GNNs with hierarchical learning for intelligent transportation systems.
>
---
#### [replaced 004] SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM与神经渲染交叉任务，旨在解决现有数据集无法涵盖两者挑战的问题。作者构建了SLAM&Render数据集，包含多模态传感器数据和精确运动信息，以评估相关方法。**

- **链接: [https://arxiv.org/pdf/2504.13713v5](https://arxiv.org/pdf/2504.13713v5)**

> **作者:** Samuel Cerezo; Gaetano Meli; Tomás Berriel Martins; Kirill Safronov; Javier Civera
>
> **备注:** 9 pages, 8 figures, submitted to The International Journal of Robotics Research (IJRR)
>
> **摘要:** Models and methods originally developed for Novel View Synthesis and Scene Rendering, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, are increasingly being adopted as representations in Simultaneous Localization and Mapping (SLAM). However, existing datasets fail to include the specific challenges of both fields, such as sequential operations and, in many settings, multi-modality in SLAM or generalization across viewpoints and illumination conditions in neural rendering. Additionally, the data are often collected using sensors which are handheld or mounted on drones or mobile robots, which complicates the accurate reproduction of sensor motions. To bridge these gaps, we introduce SLAM&Render, a novel dataset designed to benchmark methods in the intersection between SLAM, Novel View Rendering and Gaussian Splatting. Recorded with a robot manipulator, it uniquely includes 40 sequences with time-synchronized RGB-D images, IMU readings, robot kinematic data, and ground-truth pose streams. By releasing robot kinematic data, the dataset also enables the assessment of recent integrations of SLAM paradigms within robotic applications. The dataset features five setups with consumer and industrial objects under four controlled lighting conditions, each with separate training and test trajectories. All sequences are static with different levels of object rearrangements and occlusions. Our experimental results, obtained with several baselines from the literature, validate SLAM&Render as a relevant benchmark for this emerging research area.
>
---
#### [replaced 005] What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?
- **分类: cs.AI; cs.LG; cs.RO; stat.ML**

- **简介: 该论文研究基于联合嵌入预测世界模型的物理规划成功因素，旨在提升AI在新任务和环境中的泛化能力。通过分析模型结构、训练目标和规划算法，提出更优方案。**

- **链接: [https://arxiv.org/pdf/2512.24497v2](https://arxiv.org/pdf/2512.24497v2)**

> **作者:** Basile Terver; Tsung-Yen Yang; Jean Ponce; Adrien Bardes; Yann LeCun
>
> **备注:** V2 of the article: - Added AdaLN-zero - Added table comparing JEPA-WMs with baselines with std translating per-seed variability only, no variability across epochs - Reordered figures in main body of the paper
>
> **摘要:** A long-standing challenge in AI is to develop agents capable of solving a wide range of physical tasks and generalizing to new, unseen tasks and environments. A popular recent approach involves training a world model from state-action trajectories and subsequently use it with a planning algorithm to solve new tasks. Planning is commonly performed in the input space, but a recent family of methods has introduced planning algorithms that optimize in the learned representation space of the world model, with the promise that abstracting irrelevant details yields more efficient planning. In this work, we characterize models from this family as JEPA-WMs and investigate the technical choices that make algorithms from this class work. We propose a comprehensive study of several key components with the objective of finding the optimal approach within the family. We conducted experiments using both simulated environments and real-world robotic data, and studied how the model architecture, the training objective, and the planning algorithm affect planning success. We combine our findings to propose a model that outperforms two established baselines, DINO-WM and V-JEPA-2-AC, in both navigation and manipulation tasks. Code, data and checkpoints are available at https://github.com/facebookresearch/jepa-wms.
>
---
#### [replaced 006] On Steerability Factors for Growing Vine Robots
- **分类: cs.RO**

- **简介: 该论文研究软体机器人在复杂环境中的操控性问题，通过实验分析负载、压力等因素对弯曲能力的影响，优化设计提升移动性能。**

- **链接: [https://arxiv.org/pdf/2510.22504v2](https://arxiv.org/pdf/2510.22504v2)**

> **作者:** Ciera McFarland; Antonio Alvarez; Sarah Taher; Nathaniel Hanson; Margaret McGuinness
>
> **摘要:** Vine robots extend their tubular bodies by everting material from the tip, enabling navigation in complex environments with a minimalist soft body. Despite their promise for field applications, especially in the urban search and rescue domain, performance is constrained by the weight of attached sensors or tools, as well as other design and control choices. This work investigates how tip load, pressure, length, diameter, and fabrication method shape vine robot steerability--the ability to maneuver with controlled curvature--for robots that steer with series pouch motor-style pneumatic actuators. We conduct two groups of experiments: (1) studying tip load, chamber pressure, length, and diameter in a robot supporting itself against gravity, and (2) studying fabrication method and ratio of actuator to chamber pressure in a robot supported on the ground. Results show that steerability decreases with increasing tip load, is best at moderate chamber pressure, increases with length, and is largely unaffected by diameter. Robots with actuators attached on their exterior begin curving at low pressure ratios, but curvature saturates at high pressure ratios; those with actuators integrated into the robot body require higher pressure ratios to begin curving but achieve higher curvature overall. We demonstrate that robots optimized with these principles outperform those with ad hoc parameters in a mobility task that involves maximizing upward and horizontal curvatures.
>
---
#### [replaced 007] AURASeg: Attention Guided Upsampling with Residual Boundary-Assistive Refinement for Drivable-Area Segmentation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出AURASeg，用于道路区域分割任务，解决边界精度不足和特征表示有限的问题，通过引入RBRM和APUD模块提升分割效果。**

- **链接: [https://arxiv.org/pdf/2510.21536v2](https://arxiv.org/pdf/2510.21536v2)**

> **作者:** Narendhiran Vijayakumar; Sridevi. M
>
> **备注:** 6 pages, 4 figures, 4 tables
>
> **摘要:** Free space ground segmentation is essential to navigate autonomous robots, recognize drivable zones, and traverse efficiently. Fine-grained features remain challenging for existing segmentation models, particularly for robots in indoor and structured environments. These difficulties arise from ineffective multi-scale processing, suboptimal boundary refinement, and limited feature representation. To address this, we propose Attention-Guided Upsampling with Residual Boundary-Assistive Refinement (AURASeg), a ground-plane semantic segmentation framework designed to improve border precision while preserving strong region accuracy. Built on a ResNet-50 backbone, AURASeg introduces (i) a Residual Border Refinement Module (RBRM) that enhances edge delineation through boundary-assistive feature refinement, and (ii) Attention Progressive Upsampling Decoder (APUD) blocks that progressively fuse multi-level features during decoding. Additionally, we integrate a (iii) lightweight ASPPLite module to capture multi-scale context with minimal overhead. Extensive experiments on CARL-D, the Ground Mobile Robot Perception (GMRP) dataset, and a custom Gazebo indoor dataset show that AURASeg consistently outperforms strong baselines, with notable gains in boundary metrics. Finally, we demonstrate real-time deployment on a Kobuki TurtleBot, validating practical usability. The code is available at https://github.com/Narendhiranv04/AURASeg
>
---
#### [replaced 008] Model-free Adaptive Output Feedback Vibration Suppression in a Cantilever Beam
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于振动控制任务，旨在抑制未知扰动引起的悬臂梁振动。通过模型无关自适应控制方法，结合位移与加速度反馈，提升抑制效果。**

- **链接: [https://arxiv.org/pdf/2511.06084v3](https://arxiv.org/pdf/2511.06084v3)**

> **作者:** Juan Augusto Paredes Salazar; Ankit Goel
>
> **备注:** 16 pages, 14 figures, to be presented at Scitech 2026, uploaded new version that corrects some mistakes in the paper
>
> **摘要:** This paper presents a model-free adaptive control approach to suppress vibrations in a cantilevered beam excited by an unknown disturbance. The cantilevered beam under harmonic excitation is modeled using a lumped parameter approach. Based on retrospective cost optimization, a sampled-data adaptive controller is developed to suppress vibrations caused by external disturbances. Both displacement and acceleration measurements are considered for feedback. Since acceleration measurements are more sensitive to spillover, which excites higher frequency modes, a filter is developed to extract key displacement information from the acceleration data and enhance suppression performance. The vibration suppression performance is compared using both displacement and acceleration measurements.
>
---
#### [replaced 009] Low-Latency Event-Based Velocimetry for Quadrotor Control in a Narrow Pipe
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于无人机控制任务，旨在解决狭窄管道中悬停时的气动干扰问题。通过实时流场测量与神经网络估计扰动，实现闭环控制，提升飞行稳定性。**

- **链接: [https://arxiv.org/pdf/2507.15444v2](https://arxiv.org/pdf/2507.15444v2)**

> **作者:** Leonard Bauersfeld; Davide Scaramuzza
>
> **备注:** 19 pages
>
> **摘要:** Autonomous quadrotor flight in confined spaces such as pipes and tunnels presents significant challenges due to unsteady, self-induced aerodynamic disturbances. Very recent advances have enabled flight in such conditions, but they either rely on constant motion through the pipe to mitigate airflow recirculation effects or suffer from limited stability during hovering. In this work, we present the first closed-loop control system for quadrotors for hovering in narrow pipes that leverages real-time flow field measurements. We develop a low-latency, event-based smoke velocimetry method that estimates local airflow at high temporal resolution. This flow information is used by a disturbance estimator based on a recurrent convolutional neural network, which infers force and torque disturbances in real time. The estimated disturbances are integrated into a learning-based controller trained via reinforcement learning. The flow-feedback control proves particularly effective during lateral translation maneuvers in the pipe cross-section. There, the real-time disturbance information enables the controller to effectively counteract transient aerodynamic effects, thereby preventing collisions with the pipe wall. To the best of our knowledge, this work represents the first demonstration of an aerial robot with closed-loop control informed by real-time flow field measurements. This opens new directions for research on flight in aerodynamically complex environments. In addition, our work also sheds light on the characteristic flow structures that emerge during flight in narrow, circular pipes, providing new insights at the intersection of robotics and fluid dynamics.
>
---
#### [replaced 010] GR-Dexter Technical Report
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决多指灵巧手机器人在长时序操作中的泛化问题。通过硬件、数据与模型的综合设计，提升机器人在真实场景中的适应能力。**

- **链接: [https://arxiv.org/pdf/2512.24210v2](https://arxiv.org/pdf/2512.24210v2)**

> **作者:** Ruoshi Wen; Guangzeng Chen; Zhongren Cui; Min Du; Yang Gou; Zhigang Han; Liqun Huang; Mingyu Lei; Yunfei Li; Zhuohang Li; Wenlei Liu; Yuxiao Liu; Xiao Ma; Hao Niu; Yutao Ouyang; Zeyu Ren; Haixin Shi; Wei Xu; Haoxiang Zhang; Jiajun Zhang; Xiao Zhang; Liwei Zheng; Weiheng Zhong; Yifei Zhou; Zhengming Zhu; Hang Li
>
> **摘要:** Vision-language-action (VLA) models have enabled language-conditioned, long-horizon robot manipulation, but most existing systems are limited to grippers. Scaling VLA policies to bimanual robots with high degree-of-freedom (DoF) dexterous hands remains challenging due to the expanded action space, frequent hand-object occlusions, and the cost of collecting real-robot data. We present GR-Dexter, a holistic hardware-model-data framework for VLA-based generalist manipulation on a bimanual dexterous-hand robot. Our approach combines the design of a compact 21-DoF robotic hand, an intuitive bimanual teleoperation system for real-robot data collection, and a training recipe that leverages teleoperated robot trajectories together with large-scale vision-language and carefully curated cross-embodiment datasets. Across real-world evaluations spanning long-horizon everyday manipulation and generalizable pick-and-place, GR-Dexter achieves strong in-domain performance and improved robustness to unseen objects and unseen instructions. We hope GR-Dexter serves as a practical step toward generalist dexterous-hand robotic manipulation.
>
---
#### [replaced 011] PartDexTOG: Generating Dexterous Task-Oriented Grasping via Language-driven Part Analysis
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决复杂任务导向的精细抓取问题。通过语言驱动的部件分析生成有效抓取策略，提升机器人操作能力。**

- **链接: [https://arxiv.org/pdf/2505.12294v2](https://arxiv.org/pdf/2505.12294v2)**

> **作者:** Weishang Wu; Yifei Shi; Zhizhong Chen; Zhipong Cai
>
> **摘要:** Task-oriented grasping is a crucial yet challenging task in robotic manipulation. Despite the recent progress, few existing methods address task-oriented grasping with dexterous hands. Dexterous hands provide better precision and versatility, enabling robots to perform task-oriented grasping more effectively. In this paper, we argue that part analysis can enhance dexterous grasping by providing detailed information about the object's functionality. We propose PartDexTOG, a method that generates dexterous task-oriented grasps via language-driven part analysis. Taking a 3D object and a manipulation task represented by language as input, the method first generates the category-level and part-level grasp descriptions w.r.t the manipulation task by LLMs. Then, a category-part conditional diffusion model is developed to generate a dexterous grasp for each part, respectively, based on the generated descriptions. To select the most plausible combination of grasp and corresponding part from the generated ones, we propose a measure of geometric consistency between grasp and part. We show that our method greatly benefits from the open-world knowledge reasoning on object parts by LLMs, which naturally facilitates the learning of grasp generation on objects with different geometry and for different manipulation tasks. Our method ranks top on the OakInk-shape dataset over all previous methods, improving the Penetration Volume, the Grasp Displace, and the P-FID over the state-of-the-art by $3.58\%$, $2.87\%$, and $41.43\%$, respectively. Notably, it demonstrates good generality in handling novel categories and tasks.
>
---
#### [replaced 012] Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents
- **分类: cs.AI; cs.MA; cs.RO**

- **简介: 该论文研究密集环境中无指定代理的路径规划问题，提出BRaP模型并设计五种搜索算法，用于高效生成存储块重排方案。**

- **链接: [https://arxiv.org/pdf/2509.01022v2](https://arxiv.org/pdf/2509.01022v2)**

> **作者:** Bo Fu; Zhe Chen; Rahul Chandan; Alex Barbosa; Michael Caldara; Joey Durham; Federico Pecora
>
> **备注:** AAAI Conference on Artificial Intelligence (AAAI-26)
>
> **摘要:** We introduce the Block Rearrangement Problem (BRaP), a challenging component of large warehouse management which involves rearranging storage blocks within dense grids to achieve a goal state. We formally define the BRaP as a graph search problem. Building on intuitions from sliding puzzle problems, we propose five search-based solution algorithms, leveraging joint configuration space search, classical planning, multi-agent pathfinding, and expert heuristics. We evaluate the five approaches empirically for plan quality and scalability. Despite the exponential relation between search space size and block number, our methods demonstrate efficiency in creating rearrangement plans for deeply buried blocks in up to 80x80 grids.
>
---
#### [replaced 013] Grasp the Graph (GtG) 2.0: Ensemble of Graph Neural Networks for High-Precision Grasp Pose Detection in Clutter
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人抓取任务，解决杂乱环境中高精度抓取位姿检测问题。提出GtG 2.0框架，利用图神经网络集成方法提升抓取性能。**

- **链接: [https://arxiv.org/pdf/2505.02664v2](https://arxiv.org/pdf/2505.02664v2)**

> **作者:** Ali Rashidi Moghadam; Sayedmohammadreza Rastegari; Mehdi Tale Masouleh; Ahmad Kalhor
>
> **备注:** 20 pages
>
> **摘要:** Grasp pose detection in cluttered, real-world environments remains a significant challenge due to noisy and incomplete sensory data combined with complex object geometries. This paper introduces Grasp the Graph 2.0 (GtG 2.0) method, a lightweight yet highly effective hypothesis-and-test robotics grasping framework which leverages an ensemble of Graph Neural Networks for efficient geometric reasoning from point cloud data. Building on the success of GtG 1.0, which demonstrated the potential of Graph Neural Networks for grasp detection but was limited by assumptions of complete, noise-free point clouds and 4-Dof grasping, GtG 2.0 employs a conventional Grasp Pose Generator to efficiently produce 7-Dof grasp candidates. Candidates are assessed with an ensemble Graph Neural Network model which includes points within the gripper jaws (inside points) and surrounding contextual points (outside points). This improved representation boosts grasp detection performance over previous methods using the same generator. GtG 2.0 shows up to a 35% improvement in Average Precision on the GraspNet-1Billion benchmark compared to hypothesis-and-test and Graph Neural Network-based methods, ranking it among the top three frameworks. Experiments with a 3-Dof Delta Parallel robot and Kinect-v1 camera show a success rate of 91% and a clutter completion rate of 100%, demonstrating its flexibility and reliability.
>
---
#### [replaced 014] Closing the Reality Gap: Zero-Shot Sim-to-Real Deployment for Dexterous Force-Based Grasping and Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操控任务，旨在解决仿真到现实的部署难题。通过结合触觉与扭矩感知，提出高效模拟与校准方法，实现无需调优的零样本真实机械手操作。**

- **链接: [https://arxiv.org/pdf/2601.02778v2](https://arxiv.org/pdf/2601.02778v2)**

> **作者:** Zhe Zhao; Haoyu Dong; Zhengmao He; Yang Li; Xinyu Yi; Zhibin Li
>
> **摘要:** Human-like dexterous hands with multiple fingers offer human-level manipulation capabilities, but training control policies that can directly deploy on real hardware remains difficult due to contact-rich physics and imperfect actuation. We close this gap with a practical sim-to-real reinforcement learning (RL) framework that utilizes dense tactile feedback combined with joint torque sensing to explicitly regulate physical interactions. To enable effective sim-to-real transfer, we introduce (i) a computationally fast tactile simulation that computes distances between dense virtual tactile units and the object via parallel forward kinematics, providing high-rate, high-resolution touch signals needed by RL; (ii) a current-to-torque calibration that eliminates the need for torque sensors on dexterous hands by mapping motor current to joint torque; and (iii) actuator dynamics modeling to bridge the actuation gaps with randomization of non-ideal effects such as backlash, torque-speed saturation. Using an asymmetric actor-critic PPO pipeline trained entirely in simulation, our policies deploy directly to a five-finger hand. The resulting policies demonstrated two essential skills: (1) command-based, controllable grasp force tracking, and (2) reorientation of objects in the hand, both of which were robustly executed without fine-tuning on the robot. By combining tactile and torque in the observation space with effective sensing/actuation modeling, our system provides a practical solution to achieve reliable dexterous manipulation. To our knowledge, this is the first demonstration of controllable grasping on a multi-finger dexterous hand trained entirely in simulation and transferred zero-shot on real hardware.
>
---
#### [replaced 015] Dense 3D Displacement Estimation for Landslide Monitoring via Fusion of TLS Point Clouds and Embedded RGB Images
- **分类: cs.CV; cs.RO; eess.IV; physics.geo-ph**

- **简介: 该论文属于滑坡监测任务，解决传统方法难以获得高精度3D位移估计的问题，通过融合TLS点云与RGB图像，提出一种分层粗到细的位移估计方法。**

- **链接: [https://arxiv.org/pdf/2506.16265v2](https://arxiv.org/pdf/2506.16265v2)**

> **作者:** Zhaoyi Wang; Jemil Avers Butt; Shengyu Huang; Tomislav Medic; Andreas Wieser
>
> **备注:** Published in the International Journal of Applied Earth Observation and Geoinformation. 25 pages, 19 figures
>
> **摘要:** Landslide monitoring is essential for understanding geohazards and mitigating associated risks. Existing point cloud-based methods, however, typically rely on either geometric or radiometric information and often yield sparse or non-3D displacement estimates. In this paper, we propose a hierarchical partitioning-based coarse-to-fine approach that integrates 3D point clouds and co-registered RGB images to estimate dense 3D displacement vector fields. Patch-level matches are constructed using both 3D geometry and 2D image features, refined via geometric consistency checks, and followed by rigid transformation estimation per match. Experimental results on two real-world landslide datasets demonstrate that the proposed method produces 3D displacement estimates with high spatial coverage (79% and 97%) and accuracy. Deviations in displacement magnitude with respect to external measurements (total station or GNSS observations) are 0.15 m and 0.25 m on the two datasets, respectively, and only 0.07 m and 0.20 m compared to manually derived references, all below the mean scan resolutions (0.08 m and 0.30 m). Compared with the state-of-the-art method F2S3, the proposed approach improves spatial coverage while maintaining comparable accuracy. The proposed approach offers a practical and adaptable solution for TLS-based landslide monitoring and is extensible to other types of point clouds and monitoring tasks. The example data and source code are publicly available at https://github.com/gseg-ethz/fusion4landslide.
>
---
#### [replaced 016] A Photorealistic Dataset and Vision-Based Algorithm for Anomaly Detection During Proximity Operations in Lunar Orbit
- **分类: cs.RO**

- **简介: 该论文属于空间视觉异常检测任务，旨在解决月球轨道中光照变化导致的危险识别问题。通过构建合成数据集ALLO并提出MRAD算法提升检测效果。**

- **链接: [https://arxiv.org/pdf/2409.20435v5](https://arxiv.org/pdf/2409.20435v5)**

> **作者:** Selina Leveugle; Chang Won Lee; Svetlana Stolpner; Chris Langley; Paul Grouchy; Steven Waslander; Jonathan Kelly
>
> **备注:** In IEEE Robotics and Automation Letters (RA-L) and presented at the IEEE International Conference on Robotics and Automation, 1-5 June 2026, Vienna, Austria
>
> **摘要:** NASA's forthcoming Lunar Gateway space station, which will be uncrewed most of the time, will need to operate with an unprecedented level of autonomy. One key challenge is enabling the Canadarm3, the Gateway's external robotic system, to detect hazards in its environment using its onboard inspection cameras. This task is complicated by the extreme and variable lighting conditions in space. In this paper, we introduce the visual anomaly detection and localization task for the space domain and establish a benchmark based on a synthetic dataset called ALLO (Anomaly Localization in Lunar Orbit). We show that state-of-the-art visual anomaly detection methods often fail in the space domain, motivating the need for new approaches. To address this, we propose MRAD (Model Reference Anomaly Detection), a statistical algorithm that leverages the known pose of the Canadarm3 and a CAD model of the Gateway to generate reference images of the expected scene appearance. Anomalies are then identified as deviations from this model-generated reference. On the ALLO dataset, MRAD surpasses state-of-the-art anomaly detection algorithms, achieving an AP score of 62.9% at the pixel level and an AUROC score of 75.0% at the image level. Given the low tolerance for risk in space operations and the lack of domain-specific data, we emphasize the need for novel, robust, and accurate anomaly detection methods to handle the challenging visual conditions found in lunar orbit and beyond.
>
---
#### [replaced 017] iTeach: Interactive Teaching for Robot Perception using Mixed Reality
- **分类: cs.RO**

- **简介: 该论文提出iTeach系统，解决机器人在未知环境中感知模型适应问题。通过混合现实技术实现人机协作，提升机器人实时感知能力。**

- **链接: [https://arxiv.org/pdf/2410.09072v3](https://arxiv.org/pdf/2410.09072v3)**

> **作者:** Jishnu Jaykumar P; Cole Salvato; Vinaya Bomnale; Jikai Wang; Yu Xiang
>
> **摘要:** Robots deployed in the wild often encounter objects and scenes that break pre-trained perception models, yet adapting these models typically requires slow offline data collection, labeling, and retraining. We introduce iTeach, a human-in-the-loop system that enables robots to improve perception continuously as they explore new environments. A human sees the robot's predictions from its own viewpoint, corrects failures in real time, and the informed data drives iterative fine-tuning until performance is satisfactory. A mixed reality headset provides the interface, overlaying predictions in the user's view and enabling lightweight annotation via eye gaze and voice. Instead of tedious frame-by-frame labeling, a human guides the robot to scenes of choice and records short videos while interacting with objects. The human labels only the final frame, and a video segmentation model propagates labels across the sequence, converting seconds of input into dense supervision. The refined model is deployed immediately, closing the loop between human feedback and robot learning. We demonstrate iTeach on Unseen Object Instance Segmentation (UOIS), achieving consistent improvements over a pre-trained MSMFormer baseline on both our collected dataset and the SceneReplica benchmark, where it leads to higher grasping success, followed by a real-world demonstration of grasping unseen objects with a Fetch robot. By combining human judgment, efficient annotation, and on-the-fly refinement, iTeach provides a practical path toward perception systems that generalize robustly in diverse real-world conditions. Project page at https://irvlutd.github.io/iTeach
>
---
