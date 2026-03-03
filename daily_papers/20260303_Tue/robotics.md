# 机器人 cs.RO

- **最新发布 103 篇**

- **更新 62 篇**

## 最新发布

#### [new 001] From Transportation to Manipulation: Transforming Magnetic Levitation to Magnetic Robotics
- **分类: cs.RO**

- **简介: 该论文属于智能制造领域，旨在将磁悬浮技术从运输扩展到操作。解决传统磁悬浮系统功能单一的问题，通过设计六自由度机器人平台，实现精准定位与自主操作，提升制造灵活性和效率。**

- **链接: [https://arxiv.org/pdf/2603.01982](https://arxiv.org/pdf/2603.01982)**

> **作者:** Lara Bergmann; Noah Greis; Klaus Neumann
>
> **摘要:** Magnetic Levitation (MagLev) systems fundamentally increase the flexibility of in-machine material flow in industrial automation. Therefore, these systems enable dynamic throughput optimization, which is especially beneficial for high-mix low-volume manufacturing. Until now, MagLev installations have been used primarily for in-machine transport, while their potential for manipulation is largely unexplored. This paper introduces the 6D-Platform MagBot, a low-cost six degrees of freedom parallel kinematic that couples two movers into a composite robotic platform. Experiments show that the 6D-Platform MagBot achieves sub-millimeter positioning accuracy and supports fully autonomous pick up and drop off via a docking station, allowing rapid and repeatable reconfiguration of the machine. Relative to a single mover, the proposed platform substantially expands the reachable workspace, payload, and functional dexterity. By unifying transportation and manipulation, this work advances Magnetic Levitation towards Magnetic Robotics, enabling manufacturing solutions that are more agile, efficient, and adaptable.
>
---
#### [new 002] An Open-Source Modular Benchmark for Diffusion-Based Motion Planning in Closed-Loop Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，解决扩散模型在闭环系统中的评估问题。通过构建开源模块化基准，实现参数可配置和过程可观测，提升实际部署效果。**

- **链接: [https://arxiv.org/pdf/2603.01023](https://arxiv.org/pdf/2603.01023)**

> **作者:** Yun Li; Simon Thompson; Yidu Zhang; Ehsan Javanmardi; Manabu Tsukada
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Diffusion-based motion planners have achieved state-of-the-art results on benchmarks such as nuPlan, yet their evaluation within closed-loop production autonomous driving stacks remains largely unexplored. Existing evaluations abstract away ROS 2 communication latency and real-time scheduling constraints, while monolithic ONNX deployment freezes all solver parameters at export time. We present an open-source modular benchmark that addresses both gaps: using ONNX GraphSurgeon, we decompose a monolithic 18,398 node diffusion planner into three independently executable modules and reimplement the DPM-Solver++ denoising loop in native C++. Integrated as a ROS 2 node within Autoware, the open-source AD stack deployed on real vehicles worldwide, the system enables runtime-configurable solver parameters without model recompilation and per-step observability of the denoising process, breaking the black box of monolithic deployment. Unlike evaluations in standalone simulators such as CARLA, our benchmark operates within a production-grade stack and is validated through AWSIM closed-loop simulation. Through systematic comparison of DPM-Solver++ (first- and second-order) and DDIM across six step-count configurations (N in {3, 5, 7, 10, 15, 20}), we show that encoder caching yields a 3.2x latency reduction, and that second-order solving reduces FDE by 41% at N=3 compared to first-order. The complete codebase will be released as open-source, providing a direct path from simulation benchmarks to real-vehicle deployment.
>
---
#### [new 003] Beyond Static Instruction: A Multi-agent AI Framework for Adaptive Augmented Reality Robot Training
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于增强现实机器人训练任务，旨在解决静态界面无法适应不同学习者的问题。通过多智能体AI框架实现动态个性化教学。**

- **链接: [https://arxiv.org/pdf/2603.00016](https://arxiv.org/pdf/2603.00016)**

> **作者:** Nicolas Leins; Jana Gonnermann-Müller; Malte Teichmann; Sebastian Pokutta
>
> **摘要:** Augmented Reality (AR) offers powerful visualization capabilities for industrial robot training, yet current interfaces remain predominantly static, failing to account for learners' diverse cognitive profiles. In this paper, we present an AR application for robot training and propose a multi-agent AI framework for future integration that bridges the gap between static visualization and pedagogical intelligence. We report on the evaluation of the baseline AR interface with 36 participants performing a robotic pick-and-place task. While overall usability was high, notable disparities in task duration and learner characteristics highlighted the necessity for dynamic adaptation. To address this, we propose a multi-agent framework that orchestrates multiple components to perform complex preprocessing of multimodal inputs (e.g., voice, physiology, robot data) and adapt the AR application to the learner's needs. By utilizing autonomous Large Language Model (LLM) agents, the proposed system would dynamically adapt the learning environment based on advanced LLM reasoning in real-time.
>
---
#### [new 004] Wild-Drive: Off-Road Scene Captioning and Path Planning via Robust Multi-modal Routing and Efficient Large Language Model
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Wild-Drive框架，解决越野场景的场景描述与路径规划问题，通过多模态路由和高效大语言模型提升鲁棒性与透明度。**

- **链接: [https://arxiv.org/pdf/2603.00694](https://arxiv.org/pdf/2603.00694)**

> **作者:** Zihang Wang; Xu Li; Benwu Wang; Wenkai Zhu; Xieyuanli Chen; Dong Kong; Kailin Lyu; Yinan Du; Yiming Peng; Haoyang Che
>
> **摘要:** Explainability and transparent decision-making are essential for the safe deployment of autonomous driving systems. Scene captioning summarizes environmental conditions and risk factors in natural language, improving transparency, safety, and human--robot interaction. However, most existing approaches target structured urban scenarios; in off-road environments, they are vulnerable to single-modality degradations caused by rain, fog, snow, and darkness, and they lack a unified framework that jointly models structured scene captioning and path planning. To bridge this gap, we propose Wild-Drive, an efficient framework for off-road scene captioning and path planning. Wild-Drive adopts modern multimodal encoders and introduces a task-conditioned modality-routing bridge, MoRo-Former, to adaptively aggregate reliable information under degraded sensing. It then integrates an efficient large language model (LLM), together with a planning token and a gate recurrent unit (GRU) decoder, to generate structured captions and predict future trajectories. We also build the OR-C2P Benchmark, which covers structured off-road scene captioning and path planning under diverse sensor corruption conditions. Experiments on OR-C2P dataset and a self-collected dataset show that Wild-Drive outperforms prior LLM-based methods and remains more stable under degraded sensing. The code and benchmark will be publicly available at this https URL.
>
---
#### [new 005] RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在解决现有策略缺乏记忆能力的问题。通过构建RMBench基准和提出Mem-0策略，评估并分析记忆设计对性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.01229](https://arxiv.org/pdf/2603.01229)**

> **作者:** Tianxing Chen; Yuran Wang; Mingleyang Li; Yan Qin; Hao Shi; Zixuan Li; Yifan Hu; Yingsheng Zhang; Kaixuan Wang; Yue Chen; Hongcheng Wang; Renjing Xu; Ruihai Wu; Yao Mu; Yaodong Yang; Hao Dong; Ping Luo
>
> **备注:** website: this https URL
>
> **摘要:** Robotic manipulation policies have made rapid progress in recent years, yet most existing approaches give limited consideration to memory capabilities. Consequently, they struggle to solve tasks that require reasoning over historical observations and maintaining task-relevant information over time, which are common requirements in real-world manipulation scenarios. Although several memory-aware policies have been proposed, systematic evaluation of memory-dependent manipulation remains underexplored, and the relationship between architectural design choices and memory performance is still not well understood. To address this gap, we introduce RMBench, a simulation benchmark comprising 9 manipulation tasks that span multiple levels of memory complexity, enabling systematic evaluation of policy memory capabilities. We further propose Mem-0, a modular manipulation policy with explicit memory components designed to support controlled ablation studies. Through extensive simulation and real-world experiments, we identify memory-related limitations in existing policies and provide empirical insights into how architectural design choices influence memory performance. The website is available at this https URL.
>
---
#### [new 006] Tiny-DroNeRF: Tiny Neural Radiance Fields aboard Federated Learning-enabled Nano-drones
- **分类: cs.RO; cs.CV; eess.SY**

- **简介: 该论文属于无人机视觉任务，旨在解决纳米无人机在资源受限下进行3D场景重建的问题。工作包括设计轻量NeRF模型Tiny-DroNeRF，并结合联邦学习提升重建精度。**

- **链接: [https://arxiv.org/pdf/2603.01850](https://arxiv.org/pdf/2603.01850)**

> **作者:** Ilenia Carboni; Elia Cereda; Lorenzo Lamberti; Daniele Malpetti; Francesco Conti; Daniele Palossi
>
> **备注:** This paper has been accepted for publication in the IEEE ICRA 2026 conference. ©2026 IEEE
>
> **摘要:** Sub-30g nano-sized aerial robots can leverage their agility and form factor to autonomously explore cluttered and narrow environments, like in industrial inspection and search and rescue missions. However, the price for their tiny size is a strong limit in their resources, i.e., sub-100 mW microcontroller units (MCUs) delivering $\sim$100 GOps/s at best, and memory budgets well below 100 MB. Despite these strict constraints, we aim to enable complex vision-based tasks aboard nano-drones, such as dense 3D scene reconstruction: a key robotic task underlying fundamental capabilities like spatial awareness and motion planning. Top-performing 3D reconstruction methods leverage neural radiance fields (NeRF) models, which require GBs of memory and massive computation, usually delivered by high-end GPUs consuming 100s of Watts. Our work introduces Tiny-DroNeRF, a lightweight NeRF model, based on Instant-NGP, and optimized for running on a GAP9 ultra-low-power (ULP) MCU aboard our nano-drones. Then, we further empower our Tiny-DroNeRF by leveraging a collaborative federated learning scheme, which distributes the model training among multiple nano-drones. Our experimental results show a 96% reduction in Tiny-DroNeRF's memory footprint compared to Instant-NGP, with only a 5.7 dB drop in reconstruction accuracy. Finally, our federated learning scheme allows Tiny-DroNeRF to train with an amount of data otherwise impossible to keep in a single drone's memory, increasing the overall reconstruction accuracy. Ultimately, our work combines, for the first time, NeRF training on an ULP MCU with federated learning on nano-drones.
>
---
#### [new 007] A Safety-Aware Shared Autonomy Framework with BarrierIK Using Control Barrier Functions
- **分类: cs.RO**

- **简介: 该论文属于机器人安全控制任务，旨在解决共享自主系统在复杂环境中的安全性问题。通过在逆运动学层引入控制屏障函数，确保操作安全同时保持任务性能。**

- **链接: [https://arxiv.org/pdf/2603.01705](https://arxiv.org/pdf/2603.01705)**

> **作者:** Berk Guler; Kay Pompetzki; Yuanzheng Sun; Simon Manschitz; Jan Peters
>
> **备注:** Accepted on ICRA 2026, 9 pages, 5 figures
>
> **摘要:** Shared autonomy blends operator intent with autonomous assistance. In cluttered environments, linear blending can produce unsafe commands even when each source is individually collision-free. Many existing approaches model obstacle avoidance through potentials or cost terms, which only enforce safety as a soft constraint. In contrast, safety-critical control requires hard guarantees. We investigate the use of control barrier functions (CBFs) at the inverse kinematics (IK) layer of shared autonomy, targeting post-blend safety while preserving task performance. Our approach is evaluated in simulation on representative cluttered environments and in a VR teleoperation study comparing pure teleoperation with shared autonomy. Across conditions, employing CBFs at the IK layer reduces violation time and increases minimum clearance while maintaining task performance. In the user study, participants reported higher perceived safety and trust, lower interference, and an overall preference for shared autonomy with our safety filter. Additional materials available at this https URL.
>
---
#### [new 008] Hippo: High-performance Interior-Point and Projection-based Solver for Generic Constrained Trajectory Optimization
- **分类: cs.RO**

- **简介: 该论文提出Hippo，用于解决机器人轨迹优化问题，旨在提高计算效率和约束处理能力。**

- **链接: [https://arxiv.org/pdf/2603.00871](https://arxiv.org/pdf/2603.00871)**

> **作者:** Haizhou Zhao; Ludovic Righetti; Majid Khadiv
>
> **摘要:** Trajectory optimization is the core of modern model-based robotic control and motion planning. Existing trajectory optimizers, based on sequential quadratic programming (SQP) or differential dynamic programming (DDP), are often limited by their slow computation efficiency, low modeling flexibility, and poor convergence for complex tasks requiring hard constraints. In this paper, we introduce Hippo, a solver that can handle inequality constraints using the interior-point method (IPM) with an adaptive barrier update strategy and hard equality constraints via projection or IPM. Through extensive numerical benchmarks, we show that Hippo is a robust and efficient alternative to existing state-of-the-art solvers for difficult robotic trajectory optimization problems requiring high-quality solutions, such as locomotion and manipulation.
>
---
#### [new 009] LangGap: Diagnosing and Closing the Language Gap in Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言-动作模型研究，旨在解决模型忽视语言指令的问题。通过构建LangGap基准，进行语义扰动实验，验证数据增强对提升语言理解的有效性。**

- **链接: [https://arxiv.org/pdf/2603.00592](https://arxiv.org/pdf/2603.00592)**

> **作者:** Yuchen Hou; Lin Zhao
>
> **备注:** 7 pages, 3 figures. Code and benchmark will be available at this https URL
>
> **摘要:** Vision-Language-Action (VLA) models achieve over 95% success on standard benchmarks. However, through systematic experiments, we find that current state-of-the-art VLA models largely ignore language instructions. Prior work lacks: (1) systematic semantic perturbation diagnostics, (2) a benchmark that forces language understanding by design, and (3) linguistically diverse training data. This paper constructs the LangGap benchmark, based on a four-dimensional semantic perturbation method -- varying instruction semantics while keeping the tabletop layout fixed -- revealing language understanding deficits in {\pi}0.5. Existing benchmarks like LIBERO assign only one task per layout, underutilizing available objects and target locations; LangGap fully diversifies pick-and-place tasks under identical layouts, forcing models to truly understand language. Experiments show that targeted data augmentation can partially close the language gap -- success rate improves from 0% to 90% with single-task training, and 0% to 28% with multi-task training. However, as semantic diversity of extended tasks increases, model learning capacity proves severely insufficient; even trained tasks perform poorly. This reveals a fundamental challenge for VLA models in understanding diverse language instructions -- precisely the long-term value of LangGap.
>
---
#### [new 010] Compact Task-Aligned Imitation Learning for Laboratory Automation
- **分类: cs.RO**

- **简介: 该论文属于实验室自动化任务，解决传统方法成本高、灵活性差的问题。提出紧凑的模仿学习框架TVF-DiT，使用小模型实现高效自动化操作。**

- **链接: [https://arxiv.org/pdf/2603.01110](https://arxiv.org/pdf/2603.01110)**

> **作者:** Kanata Suzuki; Hanon Nakamurama; Kana Miyamoto; Tetsuya Ogata
>
> **摘要:** Robotic laboratory automation has traditionally relied on carefully engineered motion pipelines and task-specific hardware interfaces, resulting in high design cost and limited flexibility. While recent imitation learning techniques can generate general robot behaviors, their large model sizes often require high-performance computational resources, limiting applicability in practical laboratory environments. In this study, we propose a compact imitation learning framework for laboratory automation using small foundation models. The proposed method, TVF-DiT, aligns a self-supervised vision foundation model with a vision-language model through a compact adapter, and integrates them with a Diffusion Transformer-based action expert. The entire model consists of fewer than 500M parameters, enabling inference on low-VRAM GPUs. Experiments on three real-world laboratory tasks - test tube cleaning, test tube arrangement, and powder transfer - demonstrate an average success rate of 86.6%, significantly outperforming alternative lightweight baselines. Furthermore, detailed task prompts improve vision-language alignment and task performance. These results indicate that small foundation models, when properly aligned and integrated with diffusion-based policy learning, can effectively support practical laboratory automation with limited computational resources.
>
---
#### [new 011] Non-Markovian Long-Horizon Robot Manipulation via Keyframe Chaining
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人长期任务执行领域，解决长时序依赖问题。通过关键帧链式方法，提升视觉-语言-动作模型在非马尔可夫环境中的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.01465](https://arxiv.org/pdf/2603.01465)**

> **作者:** Yipeng Chen; Wentao Tan; Lei Zhu; Fengling Li; Jingjing Li; Guoli Yang; Heng Tao Shen
>
> **摘要:** Existing Vision-Language-Action (VLA) models often struggle to generalize to long-horizon tasks due to their heavy reliance on immediate observations. While recent studies incorporate retrieval mechanisms or extend context windows to handle procedural tasks, they often struggle to capture Non-Markovian dependencies, where optimal actions rely solely on specific past states rather than the current observation. To address this, we introduce Keyframe-Chaining VLA, a framework that extracts and links key historical frames to model long-horizon dependencies. Specifically, we propose an automatic keyframe selector that learns a discriminative embedding space, effectively identifying distinct state transitions. To capture task-critical information, we design a progress-aware query mechanism that dynamically retrieves historical frames based on their temporal relevance to the current execution phase. These selected keyframes are integrated into the VLA as interleaved visual tokens, explicitly grounding the policy in the long-horizon temporal context. Finally, we introduce a suite of four Non-Markovian manipulation tasks built upon the ManiSkill simulator to measure task success rates. Experimental results demonstrate that our method achieves superior performance, effectively tackling robot manipulation tasks characterized by long-horizon temporal dependencies. Code is available at this https URL.
>
---
#### [new 012] Fast Confidence-Aware Human Prediction via Hardware-accelerated Bayesian Inference for Safe Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人安全导航任务，解决人类行为预测问题。通过硬件加速的贝叶斯推断方法，实现高效、高精度的人类轨迹预测。**

- **链接: [https://arxiv.org/pdf/2603.01122](https://arxiv.org/pdf/2603.01122)**

> **作者:** Michael Lu; Minh Bui; Xubo Lyu; Mo Chen
>
> **摘要:** As robots increasingly integrate into everyday environments, ensuring their safe navigation around humans becomes imperative. Efficient and safe motion planning requires robots to account for human behavior, particularly in constrained spaces such as grocery stores or care homes, where interactions with multiple individuals are common. Prior research has employed Bayesian frameworks to model human rationality based on navigational intent, enabling the prediction of probabilistic trajectories for planning purposes. In this work, we present a simple yet novel approach for confidence-aware prediction that treats future predictions as particles. This framework is highly parallelized and accelerated on an graphics processing unit (GPU). As a result, this enables longer-term predictions at a frequency of 125 Hz and can be easily extended for multi-human predictions. Compared to existing methods, our implementation supports finer prediction time steps, yielding more granular trajectory forecasts. This enhanced resolution allows motion planners to respond effectively to subtle changes in human behavior. We validate our approach through real-world experiments, demonstrating a robot safely navigating among multiple humans with diverse navigational goals. Our results highlight the methods potential for robust and efficient human-robot coexistence in dynamic environments.
>
---
#### [new 013] Shape-Interpretable Visual Self-Modeling Enables Geometry-Aware Continuum Robot Control
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 该论文属于机器人控制任务，旨在解决连续机器人的几何感知与控制问题。通过构建可解释的三维形状模型，实现精准的形状与末端控制。**

- **链接: [https://arxiv.org/pdf/2603.01751](https://arxiv.org/pdf/2603.01751)**

> **作者:** Peng Yu; Xin Wang; Ning Tan
>
> **摘要:** Continuum robots possess high flexibility and redundancy, making them well suited for safe interaction in complex environments, yet their continuous deformation and nonlinear dynamics pose fundamental challenges to perception, modeling, and control. Existing vision-based control approaches often rely on end-to-end learning, achieving shape regulation without explicit awareness of robot geometry or its interaction with the environment. Here, we introduce a shape-interpretable visual self-modeling framework for continuum robots that enables geometry-aware control. Robot shapes are encoded from multi-view planar images using a Bezier-curve representation, transforming visual observations into a compact and physically meaningful shape space that uniquely characterizes the robot's three-dimensional configuration. Based on this representation, neural ordinary differential equations are employed to self-model both shape and end-effector dynamics directly from data, enabling hybrid shape-position control without analytical models or dense body markers. The explicit geometric structure of the learned shape space allows the robot to reason about its body and surroundings, supporting environment-aware behaviors such as obstacle avoidance and self-motion while maintaining end-effector objectives. Experiments on a cable-driven continuum robot demonstrate accurate shape-position regulation and tracking, with shape errors within 1.56% of image resolution and end-effector errors within 2% of robot length, as well as robust performance in constrained environments. By elevating visual shape representations from two-dimensional observations to an interpretable three-dimensional self-model, this work establishes a principled alternative to vision-based end-to-end control and advances autonomous, geometry-aware manipulation for continuum robots.
>
---
#### [new 014] Minimalist Compliance Control
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决安全交互中依赖力传感器的问题。提出一种仅使用电机信号的最小化顺应控制方法，无需力传感器或学习，实现稳定可靠的柔顺控制。**

- **链接: [https://arxiv.org/pdf/2603.00913](https://arxiv.org/pdf/2603.00913)**

> **作者:** Haochen Shi; Songbo Hu; Yifan Hou; Weizhuo Wang; Karen Liu; Shuran Song
>
> **备注:** Project website: this https URL
>
> **摘要:** Compliance control is essential for safe physical interaction, yet its adoption is limited by hardware requirements such as force torque sensors. While recent reinforcement learning approaches aim to bypass these constraints, they often suffer from sim-to-real gaps, lack safety guarantees, and add system complexity. We propose Minimalist Compliance Control, which enables compliant behavior using only motor current or voltage signals readily available in modern servos and quasi-direct-drive motors, without force sensors, current control, or learning. External wrenches are estimated from actuator signals and Jacobians and incorporated into a task-space admittance controller, preserving sufficient force measurement accuracy for stable and responsive compliance control. Our method is embodiment-agnostic and plug-and-play with diverse high-level planners. We validate our approach on a robot arm, a dexterous hand, and two humanoid robots across multiple contact-rich tasks, using vision-language models, imitation learning, and model-based planning. The results demonstrate robust, safe, and compliant interaction across embodiments and planning paradigms.
>
---
#### [new 015] Hybrid TD3: Overestimation Bias Analysis and Stable Policy Optimization for Hybrid Action Space
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，解决混合动作空间中的策略优化问题。提出Hybrid TD3算法，分析过估计偏差并提升训练稳定性。**

- **链接: [https://arxiv.org/pdf/2603.01302](https://arxiv.org/pdf/2603.01302)**

> **作者:** Thanh-Tuan Tran; Thanh Nguyen Canh; Nak Young Chong; Xiem HoangVan
>
> **摘要:** Reinforcement learning in discrete-continuous hybrid action spaces presents fundamental challenges for robotic manipulation, where high-level task decisions and low-level joint-space execution must be jointly optimized. Existing approaches either discretize continuous components or relax discrete choices into continuous approximations, which suffer from scalability limitations and training instability in high-dimensional action spaces and under domain randomization. In this paper, we propose Hybrid TD3, an extension of Twin Delayed Deep Deterministic Policy Gradient (TD3) that natively handles parameterized hybrid action spaces in a principled manner. We conduct a rigorous theoretical analysis of overestimation bias in hybrid action settings, deriving formal bounds under twin-critic architectures and establishing a complete bias ordering across five algorithmic variants. Building on this analysis, we introduce a weighted clipped Q-learning target that marginalizes over the discrete action distribution, achieving equivalent bias reduction to standard clipped minimization while improving policy smoothness. Experimental results demonstrate that Hybrid TD3 achieves superior training stability and competitive performance against state-of-the-art hybrid action baselines
>
---
#### [new 016] $π$-StepNFT: Wider Space Needs Finer Steps in Online RL for Flow-based VLAs
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于在线强化学习任务，解决流式视觉-语言-动作模型在多步采样中的似然不可处理问题。提出π-StepNFT框架，无需价值网络，提升泛化能力与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.02083](https://arxiv.org/pdf/2603.02083)**

> **作者:** Siting Wang; Xiaofeng Wang; Zheng Zhu; Minnan Pei; Xinyu Cui; Cheng Deng; Jian Zhao; Guan Huang; Haifeng Zhang; Jun Wang
>
> **摘要:** Flow-based vision-language-action (VLA) models excel in embodied control but suffer from intractable likelihoods during multi-step sampling, hindering online reinforcement learning. We propose \textbf{\textit{$\boldsymbol{\pi}$-StepNFT}} (Step-wise Negative-aware Fine-Tuning), a critic-and-likelihood-free framework that requires only a single forward pass per optimization step and eliminates auxiliary value networks. We identify that wider exploration spaces necessitate finer-grained, step-wise guidance for alignment. Empirically, $\pi$-StepNFT unlocks latent potential on LIBERO with competitive few-shot robustness. Moreover, it achieves superior generalization on ManiSkill, outperforming value-based baselines in OOD scenarios by preventing overfitting to multimodal features. This property offers a scalable solution promising for complex real-world applications.
>
---
#### [new 017] D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping
- **分类: cs.RO; cs.CV; cs.GR**

- **简介: 该论文属于机器人抓取任务，旨在解决仿真与现实间的动态差异问题。通过构建可微引擎，实现物体质量识别与抓取策略学习，提升抓取性能并缩小sim-to-real差距。**

- **链接: [https://arxiv.org/pdf/2603.01151](https://arxiv.org/pdf/2603.01151)**

> **作者:** Haozhe Lou; Mingtong Zhang; Haoran Geng; Hanyang Zhou; Sicheng He; Zhiyuan Gao; Siheng Zhao; Jiageng Mao; Pieter Abbeel; Jitendra Malik; Daniel Seita; Yue Wang
>
> **备注:** ICLR 2026 Poster
>
> **摘要:** Simulation provides a cost-effective and flexible platform for data generation and policy learning to develop robotic systems. However, bridging the gap between simulation and real-world dynamics remains a significant challenge, especially in physical parameter identification. In this work, we introduce a real-to-sim-to-real engine that leverages the Gaussian Splat representations to build a differentiable engine, enabling object mass identification from real-world visual observations and robot control signals, while enabling grasping policy learning simultaneously. Through optimizing the mass of the manipulated object, our method automatically builds high-fidelity and physically plausible digital twins. Additionally, we propose a novel approach to train force-aware grasping policies from limited data by transferring feasible human demonstrations into simulated robot demonstrations. Through comprehensive experiments, we demonstrate that our engine achieves accurate and robust performance in mass identification across various object geometries and mass values. Those optimized mass values facilitate force-aware policy learning, achieving superior and high performance in object grasping, effectively reducing the sim-to-real gap.
>
---
#### [new 018] EgoMoD: Predicting Global Maps of Dynamics from Local Egocentric Observations
- **分类: cs.RO**

- **简介: 该论文提出EgoMoD，用于从局部视角视频预测全局动态地图（MoD），解决动态环境中长期规划问题。通过局部观察预测全局运动趋势，提升导航预判能力。**

- **链接: [https://arxiv.org/pdf/2603.00167](https://arxiv.org/pdf/2603.00167)**

> **作者:** Iacopo Catalano; David Morilla-Cabello; Jorge Pena-Queralta; Eduardo Montijano
>
> **摘要:** Efficient navigation in dynamic environments requires anticipating how motion patterns evolve beyond the robot's immediate perceptual range, enabling preemptive rather than purely reactive planning in crowded scenes. Maps of Dynamics (MoDs) offer a structured representation of motion tendencies in space useful for long-term global planning, but constructing them traditionally requires global environment observations over extended periods of time. We introduce EgoMoD, the first approach that learns to predict future MoDs directly from short egocentric video clips collected during robot operation. Our method learns to infer environment-wide motion tendencies from local dynamic cues using a video- and pose-conditioned architecture trained with MoDs computed from external observations as privileged supervision, allowing local observations to serve as predictive signals of global motion structure. Thanks to this, we offer the capacity to forecast future motion dynamics over the whole environment rather than merely extend past patterns in the robot's field of view. Experiments in large simulated environments show that EgoMoD accurately predicts future MoDs under limited observability, while evaluation with real images showcases its zero-shot transferability to real systems.
>
---
#### [new 019] LAD-Drive: Bridging Language and Trajectory with Action-Aware Diffusion Transformers
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出LAD-Drive，解决自动驾驶中从语义意图生成连续轨迹的问题。通过分离意图与规划，结合动作解码和扩散模型，提升轨迹生成的准确性与安全性。**

- **链接: [https://arxiv.org/pdf/2603.02035](https://arxiv.org/pdf/2603.02035)**

> **作者:** Fabian Schmidt; Karol Fedurko; Markus Enzweiler; Abhinav Valada
>
> **摘要:** While multimodal large language models (MLLMs) provide advanced reasoning for autonomous driving, translating their discrete semantic knowledge into continuous trajectories remains a fundamental challenge. Existing methods often rely on unimodal planning heads that inherently limit their ability to represent multimodal driving behavior. Furthermore, most generative approaches frequently condition on one-hot encoded actions, discarding the nuanced navigational uncertainty critical for complex scenarios. To resolve these limitations, we introduce LAD-Drive, a generative framework that structurally disentangles high-level intention from low-level spatial planning. LAD-Drive employs an action decoder to infer a probabilistic meta-action distribution, establishing an explicit belief state that preserves the nuanced intent typically lost by one-hot encodings. This distribution, fused with the vehicle's kinematic state, conditions an action-aware diffusion decoder that utilizes a truncated denoising process to refine learned motion anchors into safe, kinematically feasible trajectories. Extensive evaluations on the LangAuto benchmark demonstrate that LAD-Drive achieves state-of-the-art results, outperforming competitive baselines by up to 59% in Driving Score while significantly reducing route deviations and collisions. We will publicly release the code and models on this https URL.
>
---
#### [new 020] Embedding Morphology into Transformers for Cross-Robot Policy Learning
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 该论文属于机器人学习任务，旨在解决跨机器人策略学习问题。通过在Transformer中引入形态信息，提升策略在不同机器人上的性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.00182](https://arxiv.org/pdf/2603.00182)**

> **作者:** Kei Suzuki; Jing Liu; Ye Wang; Chiori Hori; Matthew Brand; Diego Romeres; Toshiaki Koike-Akino
>
> **备注:** 17 pages, 8 figures (including appendix)
>
> **摘要:** Cross-robot policy learning -- training a single policy to perform well across multiple embodiments -- remains a central challenge in robot learning. Transformer-based policies, such as vision-language-action (VLA) models, are typically embodiment-agnostic and must infer kinematic structure purely from observations, which can reduce robustness across embodiments and even limit performance within a single embodiment. We propose an embodiment-aware transformer policy that injects morphology via three mechanisms: (1) kinematic tokens that factorize actions across joints and compress time through per-joint temporal chunking; (2) a topology-aware attention bias that encodes kinematic topology as an inductive bias in self-attention, encouraging message passing along kinematic edges; and (3) joint-attribute conditioning that augments topology with per-joint descriptors to capture semantics beyond connectivity. Across a range of embodiments, this structured integration consistently improves performance over a vanilla pi0.5 VLA baseline, indicating improved robustness both within an embodiment and across embodiments.
>
---
#### [new 021] UniHM: Unified Dexterous Hand Manipulation with Vision Language Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，解决如何通过自然语言指令实现灵活的手部操作问题。提出UniHM框架，结合视觉语言模型与物理约束，生成真实可行的抓取动作。**

- **链接: [https://arxiv.org/pdf/2603.00732](https://arxiv.org/pdf/2603.00732)**

> **作者:** Zhenhao Zhang; Jiaxin Liu; Ye Shi; Jingya Wang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Planning physically feasible dexterous hand manipulation is a central challenge in robotic manipulation and Embodied AI. Prior work typically relies on object-centric cues or precise hand-object interaction sequences, foregoing the rich, compositional guidance of open-vocabulary instruction. We introduce UniHM, the first framework for unified dexterous hand manipulation guided by free-form language commands. We propose a Unified Hand-Dexterous Tokenizer that maps heterogeneous dexterous-hand morphologies into a single shared codebook, improving cross-dexterous hand generalization and scalability to new morphologies. Our vision language action model is trained solely on human-object interaction data, eliminating the need for massive real-world teleoperation datasets, and demonstrates strong generalizability in producing human-like manipulation sequences from open-ended language instructions. To ensure physical realism, we introduce a physics-guided dynamic refinement module that performs segment-wise joint optimization under generative and temporal priors, yielding smooth and physically feasible manipulation sequences. Across multiple datasets and real-world evaluations, UniHM attains state-of-the-art results on both seen and unseen objects and trajectories, demonstrating strong generalization and high physical feasibility. Our project page at \href{this https URL}{this https URL}.
>
---
#### [new 022] Layered Safety: Enhancing Autonomous Collision Avoidance via Multistage CBF Safety Filters
- **分类: cs.RO**

- **简介: 该论文属于自主避障任务，解决动态环境中的安全轨迹规划问题。通过多阶段CBF安全滤波器，提升避障的可靠性和有效性。**

- **链接: [https://arxiv.org/pdf/2603.00338](https://arxiv.org/pdf/2603.00338)**

> **作者:** Erina Yamaguchi; Ryan M. Bena; Gilbert Bahati; Aaron D. Ames
>
> **摘要:** This paper presents a general end-to-end framework for constructing robust and reliable layered safety filters that can be leveraged to perform dynamic collision avoidance over a broad range of applications using only local perception data. Given a robot-centric point cloud, we begin by constructing an occupancy map which is used to synthesize a Poisson safety function (PSF). The resultant PSF is employed as a control barrier function (CBF) within two distinct safety filtering stages. In the first stage, we propose a predictive safety filter to compute optimal safe trajectories based on nominal potentially-unsafe commands. The resultant short-term plans are constrained to satisfy the CBF condition along a finite prediction horizon. In the second stage, instantaneous velocity commands are further refined by a real-time CBF-based safety filter and tracked by the full-order low-level robot controller. Assuming accurate tracking of velocity commands, we obtain formal guarantees of safety for the full-order system. We validate the optimality and robustness of our multistage architecture, in comparison to traditional single-stage safety filters, via a detailed Pareto analysis. We further demonstrate the effectiveness and generality of our collision avoidance methodology on multiple legged robot platforms across a variety of real-world dynamic scenarios.
>
---
#### [new 023] SFCo-Nav: Efficient Zero-Shot Visual Language Navigation via Collaboration of Slow LLM and Fast Attributed Graph Alignment
- **分类: cs.RO**

- **简介: 该论文提出SFCo-Nav，解决零样本视觉语言导航（VLN）中的高延迟和计算成本问题。通过慢速LLM与快速图对齐的协作，提升效率和实时性。**

- **链接: [https://arxiv.org/pdf/2603.01477](https://arxiv.org/pdf/2603.01477)**

> **作者:** Chaoran Xiong; Litao Wei; Xinhao Hu; Kehui Ma; Ziyi Xia; Zixin Jiang; Zhen Sun; Ling Pei
>
> **备注:** Accepted by 2026 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Recent advances in large vision-language models (VLMs) and large language models (LLMs) have enabled zero-shot approaches to visual language navigation (VLN), where an agent follows natural language instructions using only ego perception and reasoning. However, existing zero-shot methods typically construct a naive observation graph and perform per-step VLM-LLM inference on it, resulting in high latency and computation costs that limit real-time deployment. To address this, we present SFCo-Nav, an efficient zero-shot VLN framework inspired by the principle of slow-fast cognitive collaboration. SFCo-Nav integrates three key modules: 1) a slow LLM-based planner that produces a strategic chain of subgoals, each linked to an imagined object graph; 2) a fast reactive navigator for real-time object graph construction and subgoal execution; and 3) a lightweight asynchronous slow-fast bridge aligns advanced structured, attributed imagined and perceived graphs to estimate navigation confidence, triggering the slow LLM planner only when necessary. To the best of our knowledge, SFCo-Nav is the first slow-fast collaboration zero-shot VLN system supporting asynchronous LLM triggering according to the internal confidence. Evaluated on the public R2R and REVERIE benchmarks, SFCo-Nav matches or exceeds prior state-of-the-art zero-shot VLN success rates while cutting total token consumption per trajectory by over 50% and running more than 3.5 times faster. Finally, we demonstrate SFCo-Nav on a legged robot in a hotel suite, showcasing its efficiency and practicality in indoor environments.
>
---
#### [new 024] Optimal-Horizon Social Robot Navigation in Heterogeneous Crowds
- **分类: cs.RO**

- **简介: 该论文属于社会机器人导航任务，解决动态人群中的路径规划问题。针对传统MPC预测时 horizon 固定导致的适应性差问题，提出一种基于社交上下文的最优时 horizon 选择方法，提升导航效率与安全性。**

- **链接: [https://arxiv.org/pdf/2603.00507](https://arxiv.org/pdf/2603.00507)**

> **作者:** Jiamin Shi; Haolin Zhang; Yuchen Yan; Shitao Chen; Jingmin Xin; Nanning Zheng
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Navigating social robots in dense, dynamic crowds is challenging due to environmental uncertainty and complex human-robot interactions. While Model Predictive Control (MPC) offers strong real-time performance, its reliance on a fixed prediction horizon limits adaptability to changing environments and social dynamics. Furthermore, most MPC approaches treat pedestrians as homogeneous obstacles, ignoring social heterogeneity and cooperative or adversarial interactions, which often causes the Frozen Robot Problem in partially observable real-world environments. In this paper, we identify the planning horizon as a socially conditioned decision variable rather than a fixed design choice. Building on this insight, we propose an optimal-horizon social navigation framework that optimizes MPC foresight online according to inferred social context. A spatio-temporal Transformer infers pedestrian cooperation attributes from local trajectory observations, which serve as social priors for a reinforcement learning policy that optimally selects the prediction horizon under a task-driven objective. The resulting horizon-aware MPC incorporates socially conditioned safety constraints to balance navigation efficiency and interaction safety. Extensive simulations and real-world robot experiments demonstrate that optimal foresight selection is critical for robust social navigation in partially observable crowds. Compared to state-of-the-art baselines, the proposed approach achieves a 6.8\% improvement in success rate, reduces collisions by 50\%, and shortens navigation time by 19\%, with a low timeout rate of 0.8\%, validating the necessity of socially optimal planning horizons for efficient and safe robot navigation in crowded environments. Code and videos are available at Under Review.
>
---
#### [new 025] KERV: Kinematic-Rectified Speculative Decoding for Embodied VLA Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，解决VLA模型推理速度慢的问题。通过结合运动学预测与推测解码，提出KERV框架，提升速度并保持成功率。**

- **链接: [https://arxiv.org/pdf/2603.01581](https://arxiv.org/pdf/2603.01581)**

> **作者:** Zihao Zheng; Zhihao Mao; Maoliang Li; Jiayu Chen; Xinhao Sun; Zhaobo Zhang; Donggang Cao; Hong Mei; Xiang Chen
>
> **备注:** This paper has been accepted by DAC 2026
>
> **摘要:** Vision-Language-Action (VLA) models build a token-domain robot control paradigm, yet suffer from low speed. Speculative Decoding (SD) is an optimization strategy that can boost inference speed. Two key issues emerge when integrating VLA and SD: first, SD relies on re-inference to address token errors, which is computationally expensive; second, to mitigate token errors, the acceptance threshold in SD requires careful adjustment. Existing works fail to address the above two issues effectively. Meanwhile, as the bridge between AI and the physical world, existing embodied intelligence has overlooked the application of robotic kinematics. To address these issues, we innovatively combine token-domain VLA models with kinematic-domain prediction for SD, proposing a kinematic-rectified SD framework named KERV. We employ a kinematics-based Kalman Filter to predict actions and compensate for SD errors, avoiding costly re-inference. Moreover, we design a kinematics-based adjustment strategy to dynamically rectify the acceptance threshold, addressing the difficulty of threshold determination. Experimental results across diverse tasks and environments demonstrate that KERV achieves 27%~37% acceleration with nearly no Success Rate loss.
>
---
#### [new 026] MiniUGV$_2$: A Compact UAV-Deployable Tracked Ground Vehicle with Manipulation Capabilities
- **分类: cs.RO**

- **简介: 该论文属于机器人任务，解决无人机难以进入的隐蔽空间探索问题。设计了可部署的微型地面机器人，具备抓取和导航能力，提升混合空地机器人系统性能。**

- **链接: [https://arxiv.org/pdf/2603.00972](https://arxiv.org/pdf/2603.00972)**

> **作者:** Durgakant Pushp; Swapnil Kalhapure; Shaekh Mohammad Shithil; Lantao Liu
>
> **摘要:** Exploring and inspecting \emph{Hidden Spaces}, defined as environments whose entrances are accessible only to aerial robots but remain unexplored due to geometric constraints, limited flight time, and communication loss, remains a major challenge. We present miniUGV$_2$, a compact UAV-deployable tracked ground vehicle that extends UAV capabilities into confined environments. The system introduces dual articulated arms, integrated LiDAR and depth sensing, and modular electronics for enhanced autonomy. A novel tether module with an electro-permanent magnetic head enables safe deployment, retrieval, and optional detachment, thereby overcoming prior entanglement issues. Experiments demonstrate robust terrain navigation, self-righting, and manipulation of objects up to 3.5 kg, validating miniUGV$_2$ as a versatile platform for hybrid aerial-ground robotics.
>
---
#### [new 027] Closed-Loop Action Chunks with Dynamic Corrections for Training-Free Diffusion Policy
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操控任务，旨在解决动态场景下策略适应性差的问题。提出DCDP框架，通过动态修正实现快速响应和高适应性。**

- **链接: [https://arxiv.org/pdf/2603.01953](https://arxiv.org/pdf/2603.01953)**

> **作者:** Pengyuan Wu; Pingrui Zhang; Zhigang Wang; Dong Wang; Bin Zhao; Xuelong Li
>
> **备注:** Accepted by ICRA2026
>
> **摘要:** Diffusion-based policies have achieved remarkable results in robotic manipulation but often struggle to adapt rapidly in dynamic scenarios, leading to delayed responses or task failures. We present DCDP, a Dynamic Closed-Loop Diffusion Policy framework that integrates chunk-based action generation with real-time correction. DCDP integrates a self-supervised dynamic feature encoder, cross-attention fusion, and an asymmetric action encoder-decoder to inject environmental dynamics before action execution, achieving real-time closed-loop action correction and enhancing the system's adaptability in dynamic scenarios. In dynamic PushT simulations, DCDP improves adaptability by 19\% without retraining while requiring only 5\% additional computation. Its modular design enables plug-and-play integration, achieving both temporal coherence and real-time responsiveness in dynamic robotic scenarios, including real-world manipulation tasks. The project page is at: this https URL
>
---
#### [new 028] Pro-HOI: Perceptive Root-guided Humanoid-Object Interaction
- **分类: cs.RO**

- **简介: 该论文属于人形机器人操作任务，解决HOI可靠性问题。提出Pro-HOI框架，通过根轨迹引导实现稳定操作与导航，提升泛化与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.01126](https://arxiv.org/pdf/2603.01126)**

> **作者:** Yuhang Lin; Jiyuan Shi; Dewei Wang; Jipeng Kong; Yong Liu; Chenjia Bai; Xuelong Li
>
> **摘要:** Executing reliable Humanoid-Object Interaction (HOI) tasks for humanoid robots is hindered by the lack of generalized control interfaces and robust closed-loop perception mechanisms. In this work, we introduce Perceptive Root-guided Humanoid-Object Interaction, Pro-HOI, a generalizable framework for robust humanoid loco-manipulation. First, we collect box-carrying motions that are suitable for real-world deployment and optimize penetration artifacts through a Signed Distance Field loss. Second, we propose a novel training framework that conditions the policy on a desired root-trajectory while utilizing reference motion exclusively as a reward. This design not only eliminates the need for intricate reward tuning but also establishes root trajectory as a universal interface for high-level planners, enabling simultaneous navigation and loco-manipulation. Furthermore, to ensure operational reliability, we incorporate a persistent object estimation module. By fusing real-time detection with Digital Twin, this module allows the robot to autonomously detect slippage and trigger re-grasping maneuvers. Empirical validation on a Unitree G1 robot demonstrates that Pro-HOI significantly outperforms baselines in generalization and robustness, achieving reliable long-horizon execution in complex real-world scenarios.
>
---
#### [new 029] Keyframe-Guided Structured Rewards for Reinforcement Learning in Long-Horizon Laboratory Robotics
- **分类: cs.RO**

- **简介: 该论文属于长时程实验室机器人任务，解决奖励稀疏和多阶段约束问题。提出关键帧引导的奖励框架，提升精准操作成功率。**

- **链接: [https://arxiv.org/pdf/2603.00719](https://arxiv.org/pdf/2603.00719)**

> **作者:** Yibo Qiu; Shu'ang Sun; Haoliang Ye; Ronald X Xu; Mingzhai Sun
>
> **摘要:** Long-horizon precision manipulation in laboratory automation, such as pipette tip attachment and liquid transfer, requires policies that respect strict procedural logic while operating in continuous, high-dimensional state spaces. However, existing approaches struggle with reward sparsity, multi-stage structural constraints, and noisy or imperfect demonstrations, leading to inefficient exploration and unstable convergence. We propose a Keyframe-Guided Reward Generation Framework that automatically extracts kinematics-aware keyframes from demonstrations, generates stage-wise targets via a diffusion-based predictor in latent space, and constructs a geometric progress-based reward to guide online reinforcement learning. The framework integrates multi-view visual encoding, latent similarity-based progress tracking, and human-in-the-loop reinforcement fine-tuning on a Vision-Language-Action backbone to align policy optimization with the intrinsic stepwise logic of biological protocols. Across four real-world laboratory tasks, including high-precision pipette attachment and dynamic liquid transfer, our method achieves an average success rate of 82% after 40--60 minutes of online fine-tuning. Compared with HG-DAgger (42%) and Hil-ConRFT (47%), our approach demonstrates the effectiveness of structured keyframe-guided rewards in overcoming exploration bottlenecks and providing a scalable solution for high-precision, long-horizon robotic laboratory automation.
>
---
#### [new 030] DRIFT: Diffusion-based Rule-Inferred For Trajectories
- **分类: cs.RO**

- **简介: 该论文提出DRIFT，用于移动机器人轨迹生成任务，解决平滑性与精度的矛盾问题。通过结合图神经网络和时间感知机制，生成高保真轨迹。**

- **链接: [https://arxiv.org/pdf/2603.00936](https://arxiv.org/pdf/2603.00936)**

> **作者:** Jinyang Zhao; Handong Zheng; Yanjiu Zhong; Qiang Zhang; Yu Kang; Shunyu Wu
>
> **摘要:** Trajectory generation for mobile robots in unstructured environments faces a critical dilemma: balancing kinematic smoothness for safe execution with terminal precision for fine-grained tasks. Existing generative planners often struggle with this trade-off, yielding either smooth but imprecise paths or geometrically accurate but erratic motions. To address the aforementioned shortcomings, this article proposes DRIFT (Diffusion-based Rule-Inferred for Trajectories), a conditional diffusion framework designed to generate high-fidelity reference trajectories by integrating two complementary inductive biases. First, a Relational Inductive Bias, realized via a GNN-based Structured Scene Perception (SSP) module, encodes global topological constraints to ensure holistic smoothness. Second, a Temporal Attention Bias, implemented through a novel Graph-Conditioned Time-Aware GRU (GTGRU), dynamically attends to sparse obstacles and targets for precise local maneuvering. In the end, quantitative results demonstrate that DRIFT reconciles these conflicting objectives, achieving centimeter-level imitation fidelity (0.041m FDE) and competitive smoothness (27.19 Jerk). This balance yields highly executable reference plans for downstream control.
>
---
#### [new 031] HydroShear: Hydroelastic Shear Simulation for Tactile Sim-to-Real Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于触觉模拟到现实的强化学习任务，解决接触密集任务中模拟与现实差距大的问题。提出HydroShear模拟器，精确建模剪切力和接触动态，提升政策迁移效果。**

- **链接: [https://arxiv.org/pdf/2603.00446](https://arxiv.org/pdf/2603.00446)**

> **作者:** An Dang; Jayjun Lee; Mustafa Mukadam; X. Alice Wu; Bernadette Bucher; Manikantan Nambi; Nima Fazeli
>
> **备注:** Project page: this https URL
>
> **摘要:** In this paper, we address the problem of tactile sim-to-real policy transfer for contact-rich tasks. Existing methods primarily focus on vision-based sensors and emphasize image rendering quality while providing overly simplistic models of force and shear. Consequently, these models exhibit a large sim-to-real gap for many dexterous tasks. Here, we present HydroShear, a non-holonomic hydroelastic tactile simulator that advances the state-of-the-art by modeling: a) stick-slip transitions, b) path-dependent force and shear build up, and c) full SE(3) object-sensor interactions. HydroShear extends hydroelastic contact models using Signed Distance Functions (SDFs) to track the displacements of the on-surface points of an indenter during physical interaction with the sensor membrane. Our approach generates physics-based, computationally efficient force fields from arbitrary watertight geometries while remaining agnostic to the underlying physics engine. In experiments with GelSight Minis, HydroShear more faithfully reproduces real tactile shear compared to existing methods. This fidelity enables zero-shot sim-to-real transfer of reinforcement learning policies across four tasks: peg insertion, bin packing, book shelving for insertion, and drawer pulling for fine gripper control under slip. Our method achieves a 93% average success rate, outperforming policies trained on tactile images (34%) and alternative shear simulation methods (58%-61%).
>
---
#### [new 032] FATE: Closed-Loop Feasibility-Aware Task Generation with Active Repair for Physically Grounded Robotic Curricula
- **分类: cs.RO**

- **简介: 该论文提出FATE框架，解决生成任务中物理可行性不足的问题。通过闭环验证与主动修复，提升机器人任务课程的物理合理性。**

- **链接: [https://arxiv.org/pdf/2603.01505](https://arxiv.org/pdf/2603.01505)**

> **作者:** Bingchuan Wei; Bingqi Huang; Jingheng Ma; Zeyu zhang; Sen Cui
>
> **备注:** 16 Pages, 4 Figures
>
> **摘要:** Recent breakthroughs in generative simulation have harnessed Large Language Models (LLMs) to generate diverse robotic task curricula, yet these open-loop paradigms frequently produce linguistically coherent but physically infeasible goals, stemming from ungrounded task specifications or misaligned objective formulations. To address this critical limitation, we propose FATE (Feasibility-Aware Task gEneration), a closed-loop, self-correcting framework that reimagines task generation as an iterative validation-and-refinement process. Unlike conventional methods that decouple generation and verification into discrete stages, FATE embeds a generalist embodied agent directly into the generation loop to proactively guarantee the physical groundedness of the resulting curriculum. FATE instantiates a sequential auditing pipeline: it first validates static scene attributes (e.g., object affordances, layout compatibility) and subsequently verifies execution feasibility via simulated embodied interaction. Critical to its performance, upon detecting an infeasible task, FATE deploys an active repair module that autonomously adapts scene configurations or policy specifications, converting unworkable proposals into physically valid task instances. Extensive experiments validate that FATE generates semantically diverse, physically grounded task curricula while achieving a substantial reduction in execution failure rates relative to state-of-the-art generative baselines.
>
---
#### [new 033] Trust in Autonomous Human--Robot Collaboration: Effects of Responsive Interaction Policies
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究自主人机协作中的信任形成，比较了响应式与中性交互策略的影响，旨在理解信任在真实自主系统中的动态变化。**

- **链接: [https://arxiv.org/pdf/2603.00154](https://arxiv.org/pdf/2603.00154)**

> **作者:** Shauna Heron; Meng Cheng Lau
>
> **摘要:** Trust plays a central role in human--robot collaboration, yet its formation is rarely examined under the constraints of fully autonomous interaction. This pilot study investigated how interaction policy influences trust during in-person collaboration with a social robot operating without Wizard-of-Oz control or scripted repair. Participants completed a multi-stage collaborative task with a mobile robot that autonomously managed spoken-language dialogue, affect inference, and task progression. Two interaction policies were compared: a responsive policy, in which the robot proactively adapted its dialogue and assistance based on inferred interaction state, and a neutral, reactive policy, in which the robot provided only direct, task-relevant responses when prompted. Responsive interaction was associated with significantly higher post-interaction trust under viable communication conditions, despite no reliable differences in overall task accuracy. Sensitivity analyses indicated that affective and experiential components of trust were more sensitive to communication breakdown than evaluative judgments of reliability, and that as language-mediated interaction degraded, the trust advantage associated with responsiveness attenuated and ratings became less clearly interpretable as calibrated evaluations of collaborative competence. These findings suggest that trust in autonomous human--robot interaction emerges from process-level interaction dynamics and operates within constraints imposed by communication viability, highlighting the importance of evaluating trust under real autonomy conditions when designing interactive robotic systems.
>
---
#### [new 034] Multiview Progress Prediction of Robot Activities
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人动作进度预测任务，旨在解决单视角感知不足的问题，提出多视角架构以提升机器人操作任务中的动作理解能力。**

- **链接: [https://arxiv.org/pdf/2603.00151](https://arxiv.org/pdf/2603.00151)**

> **作者:** Elena Zoppellari; Federico Becattini; Marco Fiorucci; Lamberto Ballan
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** For robots to operate effectively and safely alongside humans, they must be able to understand the progress of ongoing actions. This ability, known as action progress prediction, is critical for tasks ranging from timely assistance to autonomous decision-making. However, modeling action progression in robotics has often been overlooked. Moreover, a single camera may be insufficient for understanding robot's ego-actions, as self-occlusion can significantly hinder perception and model performance. In this paper, we propose a multi-view architecture for action progress prediction in robot manipulation tasks. Experiments on Mobile ALOHA demonstrate the effectiveness of the proposed approach.
>
---
#### [new 035] From Dialogue to Execution: Mixture-of-Agents Assisted Interactive Planning for Behavior Tree-Based Long-Horizon Robot Execution
- **分类: cs.RO**

- **简介: 该论文属于机器人任务规划领域，解决长周期任务中交互效率低和计划管理难的问题。通过融合Mixture-of-Agents和行为树，提升规划效率与执行可靠性。**

- **链接: [https://arxiv.org/pdf/2603.01113](https://arxiv.org/pdf/2603.01113)**

> **作者:** Kanata Suzuki; Kazuki Hori; Haruka Miyoshi; Shuhei Kurita; Tetsuya Ogata
>
> **摘要:** Interactive task planning with large language models (LLMs) enables robots to generate high-level action plans from natural language instructions. However, in long-horizon tasks, such approaches often require many questions, increasing user burden. Moreover, flat plan representations become difficult to manage as task complexity grows. We propose a framework that integrates Mixture-of-Agents (MoA)-based proxy answering into interactive planning and generates Behavior Trees (BTs) for structured long-term execution. The MoA consists of multiple LLM-based expert agents that answer general or domain-specific questions when possible, reducing unnecessary human intervention. The resulting BT hierarchically represents task logic and enables retry mechanisms and dynamic switching among multiple robot policies. Experiments on cocktail-making tasks show that the proposed method reduces human response requirements by approximately 27% while maintaining structural and semantic similarity to fully human-answered BTs. Real-robot experiments on a smoothie-making task further demonstrate successful long-horizon execution with adaptive policy switching and recovery from action failures. These results indicate that MoA-assisted interactive planning improves dialogue efficiency while preserving execution quality in real-world robotic tasks.
>
---
#### [new 036] Learning Physics from Pretrained Video Models: A Multimodal Continuous and Sequential World Interaction Models for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决数据稀缺问题。通过预训练视频模型提取物理知识，构建连续交互框架，提升机器人操作性能。**

- **链接: [https://arxiv.org/pdf/2603.00110](https://arxiv.org/pdf/2603.00110)**

> **作者:** Zijian Song; Qichang Li; Sihan Qin; Yuhao Chen; Tianshui Chen; Liang Lin; Guangrun Wang
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** The scarcity of large-scale robotic data has motivated the repurposing of foundation models from other modalities for policy learning. In this work, we introduce PhysGen (Learning Physics from Pretrained Video Generation Models), a scalable continuous and sequential world interaction framework that leverages autoregressive video generation to solve robotic manipulation tasks. By treating the pretrained video model as a proxy for a physics simulator, PhysGen models the dynamic interplay between the external environment and robot actions. We introduce a multimodal continuous representation that unifies video and action into shared physical tokens, bridging the gap between discrete video generation and continuous robotic control. This approach enables the seamless transfer of implicit physical knowledge-such as object permanence and dynamics-from video pretraining to downstream this http URL ensure efficient convergence, we incorporate causal masking, inverse kinematics, Lookahead Multi-Token Prediction (L-MTP), and key-value (KV) caching. Experimental results on the Libero and ManiSkill benchmarks demonstrate that PhysGen consistently outperforms robust baselines, surpassing OpenVLA and WorldVLA by margins of 13.8% and 8.8%, respectively. Notably, in real-world scenarios, PhysGen matches the performance of large-scale action-pretrained models like $\pi_0$ without requiring prior action-specific pretraining, demonstrating superior capability in physically complex tasks such as grasping transparent objects. These findings validate the potential of extracting physical intuition from pretrained video generators to facilitate generalizable robotic manipulation.
>
---
#### [new 037] SaferPath: Hierarchical Visual Navigation with Learned Guidance and Safety-Constrained Control
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，旨在解决机器人在复杂环境中安全导航的问题。通过结合学习引导与安全约束控制，提升导航成功率并减少碰撞。**

- **链接: [https://arxiv.org/pdf/2603.01898](https://arxiv.org/pdf/2603.01898)**

> **作者:** Lingjie Zhang; Zeyu Jiang; Changhao Chen
>
> **备注:** ICRA 2026
>
> **摘要:** Visual navigation is a core capability for mobile robots, yet end-to-end learning-based methods often struggle with generalization and safety in unseen, cluttered, or narrow environments. These limitations are especially pronounced in dense indoor settings, where collisions are likely and end-to-end models frequently fail. To address this, we propose SaferPath, a hierarchical visual navigation framework that leverages learned guidance from existing end-to-end models and refines it through a safety-constrained optimization-control module. SaferPath transforms visual observations into a traversable-area map and refines guidance trajectories using Model Predictive Stein Variational Evolution Strategy (MP-SVES), efficiently generating safe trajectories in only a few iterations. The refined trajectories are tracked by an MPC controller, ensuring robust navigation in complex environments. Extensive experiments in scenarios with unseen obstacles, dense unstructured spaces, and narrow corridors demonstrate that SaferPath consistently improves success rates and reduces collisions, outperforming representative baselines such as ViNT and NoMaD, and enabling safe navigation in challenging real-world settings.
>
---
#### [new 038] Acoustic Sensing for Universal Jamming Grippers
- **分类: cs.RO; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于机器人感知任务，旨在解决传统传感器影响抓取性能的问题。通过声学传感实现柔性抓取器的自感知，提升物体识别与抓取能力。**

- **链接: [https://arxiv.org/pdf/2603.00351](https://arxiv.org/pdf/2603.00351)**

> **作者:** Lion Weber; Theodor Wienert; Martin Splettstößer; Alexander Koenig; Oliver Brock
>
> **备注:** Accepted at ICRA 2026, supplementary material under this https URL
>
> **摘要:** Universal jamming grippers excel at grasping unknown objects due to their compliant bodies. Traditional tactile sensors can compromise this compliance, reducing grasping performance. We present acoustic sensing as a form of morphological sensing, where the gripper's soft body itself becomes the sensor. A speaker and microphone are placed inside the gripper cavity, away from the deformable membrane, fully preserving compliance. Sound propagates through the gripper and object, encoding object properties, which are then reconstructed via machine learning. Our sensor achieves high spatial resolution in sensing object size (2.6 mm error) and orientation (0.6 deg error), remains robust to external noise levels of 80 dBA, and discriminates object materials (up to 100% accuracy) and 16 everyday objects (85.6% accuracy). We validate the sensor in a realistic tactile object sorting task, achieving 53 minutes of uninterrupted grasping and sensing, confirming the preserved grasping performance. Finally, we demonstrate that disentangled acoustic representations can be learned, improving robustness to irrelevant acoustic variations.
>
---
#### [new 039] I-Perceive: A Foundation Model for Active Perception with Language Instructions
- **分类: cs.RO**

- **简介: 该论文提出I-Perceive，解决开放意图下的主动感知问题，通过融合视觉语言与几何模型，实现基于自然语言指令的相机视角预测。**

- **链接: [https://arxiv.org/pdf/2603.00600](https://arxiv.org/pdf/2603.00600)**

> **作者:** Yongxi Huang; Zhuohang Wang; Wenjing Tang; Cewu Lu; Panpan Cai
>
> **摘要:** Active perception, the ability of a robot to proactively adjust its viewpoint to acquire task-relevant information, is essential for robust operation in unstructured real-world environments. While critical for downstream tasks such as manipulation, existing approaches have largely been confined to local settings (e.g., table-top scenes) with fixed perception objectives (e.g., occlusion reduction). Addressing active perception with open-ended intents in large-scale environments remains an open challenge. To bridge this gap, we propose I-Perceive, a foundation model for active perception conditioned on natural language instructions, designed for mobile manipulators and indoor environments. I-Perceive predicts camera views that follows open-ended language instructions, based on image-based scene contexts. By fusing a Vision-Language Model (VLM) backbone with a geometric foundation model, I-Perceive bridges semantic and geometric understanding, thus enabling effective reasoning for active perception. We train I-Perceive on a diverse dataset comprising real-world scene-scanning data and simulation data, both processed via an automated and scalable data generation pipeline. Experiments demonstrate that I-Perceive significantly outperforms state-of-the-art VLMs in both prediction accuracy and instruction following of generated camera views, and exhibits strong zero-shot generalization to novel scenes and tasks.
>
---
#### [new 040] (hu)Man vs. Machine: In the Future of Motorsport, can Autonomous Vehicles Compete?
- **分类: cs.RO**

- **简介: 论文探讨人类与自动驾驶车辆在赛车中的竞争可能性，分析技术与策略挑战，提出未来研究方向。属于科技评测任务，解决混合竞赛可行性问题。**

- **链接: [https://arxiv.org/pdf/2603.01560](https://arxiv.org/pdf/2603.01560)**

> **作者:** Armand Amaritei; Amber-Lily Blackadder; Sebastian Donnelly; Lora Hernandez; James Vine; Alexander Rast; Matthias Rolf; Andrew Bradley
>
> **摘要:** Motorsport has historically driven technological innovation in the automotive industry. Autonomous racing provides a proving ground to push the limits of performance of autonomous vehicle (AV) systems. In principle, AVs could be at least as fast, if not faster, than humans. However, human driven racing provides broader audience appeal thus far, and is more strategically challenging. Both provide opportunities to push each other even further technologically, yet competitions remain separate. This paper evaluates whether the future of motorsport could encompass joint competition between humans and AVs. Analysis of the current state of the art, as well as recent competition outcomes, shows that while technical performance has reached comparable levels, there are substantial challenges in racecraft, strategy and safety that need to be overcome. Outstanding issues involved in mixed human-AI racing, ranging from an initial assessment of critical factors such as system-level latencies, to effective planning and risk guarantees are explored. The crucial non-technical aspect of audience engagement and appeal regarding the changing character of motorsport is addressed. In the wider context of motorsport and AVs, this work outlines a proposed agenda for future research to 'keep pushing the possible', in the true spirit of motorsport.
>
---
#### [new 041] Designing Social Robots with Ethical, User-Adaptive Explainability in the Era of Foundation Models
- **分类: cs.RO**

- **简介: 论文探讨在大模型驱动的社交机器人中实现伦理化、用户自适应的解释性。属于人机交互任务，解决传统解释策略不足的问题，提出四点建议以提升解释的个性化与公平性。**

- **链接: [https://arxiv.org/pdf/2603.00102](https://arxiv.org/pdf/2603.00102)**

> **作者:** Fethiye Irmak Dogan; Alva Markelius; Hatice Gunes
>
> **备注:** Companion Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction
>
> **摘要:** Foundation models are increasingly embedded in social robots, mediating not only what they say and do but also how they adapt to users over time. This shift renders traditional ``one-size-fits-all'' explanation strategies especially problematic: generic justifications are now wrapped around behaviour produced by models trained on vast, heterogeneous, and opaque datasets. We argue that ethical, user-adapted explainability must be treated as a core design objective for foundation-model-driven social robotics. We first identify open challenges around explainability and ethical concerns that arise when both adaptation and explanation are delegated to foundation models. Building on this analysis, we propose four recommendations for moving towards user-adapted, modality-aware, and co-designed explanation strategies grounded in smaller, fairer datasets. An illustrative use case of an LLM-driven socially assistive robot demonstrates how these recommendations might be instantiated in a sensitive, real-world domain.
>
---
#### [new 042] HierKick: Hierarchical Reinforcement Learning for Vision-Guided Soccer Robot Control
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决足球机器人在复杂环境中的多时间尺度决策问题。提出HierKick框架，结合视觉与分层强化学习，实现高效运动控制。**

- **链接: [https://arxiv.org/pdf/2603.00948](https://arxiv.org/pdf/2603.00948)**

> **作者:** Yizhi Chen; Zheng Zhang; Zhanxiang Cao; Yihe Chen; Shengcheng Fu; Liyun Yan; Yang Zhang; Jiali Liu; Haoyang Li; Yue Gao
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** Controlling soccer robots involves multi-time-scale decision-making, which requires balancing long-term tactical planning and short-term motion execution. Traditional end-to-end reinforcement learning (RL) methods face challenges in complex dynamic environments. This paper proposes HierKick, a vision-guided soccer robot control framework based on dual-frequency hierarchical RL. The framework adopts a hierarchical control architecture featuring a 5 Hz high-level policy that integrates YOLOv8 for real-time detection and selects tasks via a coach model, and a pre-trained 50 Hz low-level controller for precise joint control. Through this architecture, the framework achieves the four steps of approaching, aligning, dribbling, and kicking. Experimental results show that the success rates of this framework are 95.2\% in IsaacGym, 89.8\% in Mujoco, and 80\% in the real world. HierKick provides an effective hierarchical paradigm for robot control in complex environments, extendable to multi-time-scale tasks, with its modular design and skill reuse offering a new path for intelligent robot control.
>
---
#### [new 043] A Deployable Bio-inspired Compliant Leg Design for Enhanced Leaping in Quadruped Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人设计任务，旨在解决四足机器人跳跃能力不足的问题。通过仿生设计，采用弹性材料提升跳跃高度，实现17.1%的性能提升。**

- **链接: [https://arxiv.org/pdf/2603.01128](https://arxiv.org/pdf/2603.01128)**

> **作者:** Yiyang Chen; Yuxin Liu; Jinzheng Zhou; Fanxin Wang; Qinglei Bu; Jie Sun; Yikun Cheng
>
> **摘要:** Quadruped robots are becoming increasingly essential for various applications, including industrial inspection and catastrophe search and rescue. These scenarios require robots to possess enhanced agility and obstacle-navigation skills. Nonetheless, the performance of current platforms is often constrained by insufficient peak motor power, limiting their ability to perform explosive jumps. To address this challenge, this paper proposes a bio-inspired method that emulates the energy-storage mechanism found in froghopper legs. We designed a Deployable Compliant Leg (DCL) utilizing a specialized 3D-printed elastic material, Polyether block amide (PEBA), featuring a lightweight internal lattice structure. This structure functions analogously to biological tendons, storing elastic energy during the robot's squatting phase and rapidly releasing it to augment motor output during the leap. The proposed mechanical design significantly enhances the robot's vertical jumping capability. Through finite element analysis (FEA) and experimental validation, we demonstrate a relative performance improvement of 17.1% in vertical jumping height.
>
---
#### [new 044] Spherical Latent Motion Prior for Physics-Based Simulated Humanoid Control
- **分类: cs.RO**

- **简介: 该论文属于物理模拟人形控制任务，旨在解决运动先验学习中的信息丢失和行为无效问题。提出SLMP方法，通过两阶段训练生成结构化潜在动作空间，实现稳定且语义有效的随机采样。**

- **链接: [https://arxiv.org/pdf/2603.01294](https://arxiv.org/pdf/2603.01294)**

> **作者:** Jing Tan; Weisheng Xu; Xiangrui Jiang; Jiaxi Zhang; Kun Yang; Kai Wu; Jiaqi Xiong; Shiting Chen; Yangfan Li; Yixiao Feng; Yuetong Fang; Yujia Zou; Yiqun Song; Renjing Xu
>
> **摘要:** Learning motion priors for physics-based humanoid control is an active research topic. Existing approaches mainly include variational autoencoders (VAE) and adversarial motion priors (AMP). VAE introduces information loss, and random latent sampling may sometimes produce invalid behaviors. AMP suffers from mode collapse and struggles to capture diverse motion skills. We present the Spherical Latent Motion Prior (SLMP), a two-stage method for learning motion priors. In the first stage, we train a high-quality motion tracking controller. In the second stage, we distill the tracking controller into a spherical latent space. A combination of distillation, a discriminator, and a discriminator-guided local semantic consistency constraint shapes a structured latent action space, allowing stable random sampling without information loss. To evaluate SLMP, we collect a two-hour human combat motion capture dataset and show that SLMP preserves fine motion detail without information loss, and random sampling yields semantically valid and stable behaviors. When applied to a two-agent physics-based combat task, SLMP produces human-like and physically plausible combat behaviors only using simple rule-based rewards. Furthermore, SLMP generalizes across different humanoid robot morphologies, demonstrating its transferability beyond a single simulated avatar.
>
---
#### [new 045] TGM-VLA: Task-Guided Mixup for Sampling-Efficient and Robust Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人模仿学习任务，解决数据质量差和训练效率低的问题。通过优化采样策略、引入颜色反转投影和任务引导混合技术，提升模型性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.00615](https://arxiv.org/pdf/2603.00615)**

> **作者:** Fanqi Pu; Lei Jiang; Wenming Yang
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** The performance of robotic imitation learning is fundamentally limited by data quality and training strategies. Prevalent sampling strategies on RLBench suffer from severe keyframe redundancy and imbalanced temporal distribution, leading to inefficient memory usage and unstable optimization. Moreover, reprojecting point clouds onto multi-view images with a black background--while more efficient than voxel-based methods--often causes dark objects to be indistinguishable and hard to manipulate. In this work, we propose a novel holistic framework that significantly improves both model performance and training efficiency. First, we redesign and optimize the keyframe sampling strategy, reducing memory consumption by 80% and accelerating training speed by 5x. Second, we augment the model with a color inversion projection branch--a simple yet effective module that resolves the ambiguity of dark objects. Finally, we propose a task-guided mixup technique that dynamically fuses point clouds and action heatmaps according to task instructions, greatly improving robustness to distractors and performance in multi-goal scenarios. Extensive experiments demonstrate that our method achieves state-of-the-art performance with a 90.5% success rate on RLBench and 68.8% on the COLOSSEUM benchmark under challenging interference conditions. Our code and checkpoints are available at this https URL.
>
---
#### [new 046] Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决离散路径点与连续运动不匹配的问题，提出NIAF方法实现连续动作函数回归。**

- **链接: [https://arxiv.org/pdf/2603.01766](https://arxiv.org/pdf/2603.01766)**

> **作者:** Haoyun Liu; Jianzhuang Zhao; Xinyuan Chang; Tianle Shi; Chuanzhang Meng; Jiayuan Tan; Feng Xiong; Tong Lin; Dongjie Huo; Mu Xu; SongLin Dong; Zhiheng Ma; Yihong Gong; Sheng Zhong
>
> **摘要:** Despite the rapid progress of Vision-Language-Action (VLA) models, the prevailing paradigm of predicting discrete waypoints remains fundamentally misaligned with the intrinsic continuity of physical motion. This discretization imposes rigid sampling rates, lacks high-order differentiability, and introduces quantization artifacts that hinder precise, compliant interaction. We propose Neural Implicit Action Fields (NIAF), a paradigm shift that reformulates action prediction from discrete waypoints to continuous action function regression. By utilizing an MLLM as a hierarchical spectral modulator over a learnable motion prior, NIAF synthesizes infinite-resolution trajectories as continuous-time manifolds. This formulation enables analytical differentiability, allowing for explicit supervision of velocity, acceleration, and jerk to ensure mathematical consistency and physical plausibility. Our approach achieves state-of-the-art results on CALVIN and LIBERO benchmarks across diverse backbones. Furthermore, real-world experiments demonstrate that NIAF enables stable impedance control, bridging the gap between high-level semantic understanding and low-level dynamic execution.
>
---
#### [new 047] TacMamba: A Tactile History Compression Adapter Bridging Fast Reflexes and Slow VLA Reasoning
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，解决视觉模糊下触觉与视觉信息融合问题。提出TacMamba架构，通过高效触觉压缩和双阶段训练提升操作精度与实时性。**

- **链接: [https://arxiv.org/pdf/2603.01700](https://arxiv.org/pdf/2603.01700)**

> **作者:** Zhenan Wang; Yanzhe Wang; Meixuan Ren; Peng Li; Yang Liu; Yifei Nie; Limin Long; Yun Ye; Xiaofeng Wang; Zhen Zhu; Huixu Dong
>
> **摘要:** In visually ambiguous manipulation such as detecting button click tactile feedback is often the sole source of ground truth. However, fusing tactile data poses a significant challenge due to a spatiotemporal mismatch: tactile perception requires high-frequency processing with long-horizon memory (System 1), whereas visual policies operate at low control frequencies (System 2). Existing architectures struggle to bridge this gap: Transformers are computationally prohibitive for high-frequency loops (>100Hz), while LSTMs suffer from forgetting over extended interaction histories. In this paper, we introduce TacMamba, a hierarchical architecture that aligns high-bandwidth tactile reflexes with low-frequency visual planning. Our approach comprises three core contributions: (1) a custom high-frequency tactile interface designed for flexible integration; (2) a Mamba-based Tactile History Compressor that encodes continuous force history into a compact state with O(1) inference latency (0.45 ms), enabling plug-and-play fusion with VLA models without joint pre-training and (3) a Tactile-Guided Dual-Stage Training strategy that leverages temporal discrimination for self-supervised representation learning and phase-uniform sampling to mitigate data sparsity. Experiments on discrete counting and implicit state switching demonstrate that TacMamba achieves 100% success rates, significantly outperforming the visual-only pi_0.5 baseline, while strictly satisfying hard real-time constraints.
>
---
#### [new 048] Zero-Shot Robotic Manipulation via 3D Gaussian Splatting-Enhanced Multimodal Retrieval-Augmented Generation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决零样本环境下物体泛化与姿态预测问题。通过构建多源知识库并引入3D高斯点云增强的多模态生成框架，提升模型的泛化能力和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.00500](https://arxiv.org/pdf/2603.00500)**

> **作者:** Zilong Xie; Jingyu Gong; Xin Tan; Zhizhong Zhang; Yuan Xie
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Existing end-to-end approaches of robotic manipulation often lack generalization to unseen objects or tasks due to limited data and poor interpretability. While recent Multimodal Large Language Models (MLLMs) demonstrate strong commonsense reasoning, they struggle with geometric and spatial understanding required for pose prediction. In this paper, we propose RobMRAG, a 3D Gaussian Splatting-Enhanced Multimodal Retrieval-Augmented Generation (MRAG) framework for zero-shot robotic manipulation. Specifically, we construct a multi-source manipulation knowledge base containing object contact frames, task completion frames, and pose parameters. During inference, a Hierarchical Multimodal Retrieval module first employs a three-priority hybrid retrieval strategy to find task-relevant object prototypes, then selects the geometrically closest reference example based on pixel-level similarity and Instance Matching Distance (IMD). We further introduce a 3D-Aware Pose Refinement module based on 3D Gaussian Splatting into the MRAG framework, which aligns the pose of the reference object to the target object in 3D space. The aligned results are reprojected onto the image plane and used as input to the MLLM to enhance the generation of the final pose parameters. Extensive experiments show that on a test set containing 30 categories of household objects, our method improves the success rate by 7.76% compared to the best-performing zero-shot baseline under the same setting, and by 6.54% compared to the state-of-the-art supervised baseline. Our results validate that RobMRAG effectively bridges the gap between high-level semantic reasoning and low-level geometric execution, enabling robotic systems that generalize to unseen objects while remaining inherently interpretable.
>
---
#### [new 049] Online Generation of Collision-Free Trajectories in Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决动态环境中生成无碰撞轨迹的问题。通过在线方法将几何路径转换为满足运动学约束的平滑轨迹，实现实时适应环境变化。**

- **链接: [https://arxiv.org/pdf/2603.00759](https://arxiv.org/pdf/2603.00759)**

> **作者:** Nermin Covic; Bakir Lacevic
>
> **备注:** Submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** In this paper, we present an online method for converting an arbitrary geometric path represented by a sequence of states, generated by any planner (e.g., sampling-based planners like RRT or PRM, search-based planners like ARA*, etc.), into a corresponding kinematically feasible, jerk-limited trajectory. The method generates a sequence of quintic/quartic splines that can be discretized at a user-specified control rate, and then streamed to a low-level robot controller. Our approach enables real-time adaptation to newly captured changes in the environment. It can also be re-invoked at any time instance to generate a new trajectory from the robot's current to a desired target state or sequence of states. We can guarantee that the trajectory will remain collision-free for a certain amount of time in dynamic environments, while allowing bounded geometric deviation from the original path. The kinematic constraints are taken into account, including limited jerk. We validate the approach in a comparative simulation study against the competing method, demonstrating favorable behavior w.r.t. smoothness, computational time, and real-time performance, particularly in scenarios with frequent changes of target states (up to 1 [kHz]). Experiments on a real robot demonstrate that the proposed approach can be used in real-world scenarios including human presence.
>
---
#### [new 050] Agent-Based Simulation of Trust Development in Human-Robot Teams: An Empirically-Validated Framework
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机协作研究，解决信任动态建模问题。构建了基于代理的模拟框架，验证了信任影响因素及性能关系。**

- **链接: [https://arxiv.org/pdf/2603.01189](https://arxiv.org/pdf/2603.01189)**

> **作者:** Ravi Kalluri
>
> **摘要:** This paper presents an empirically grounded agent-based model capturing trust dynamics, workload distribution, and collaborative performance in human-robot teams. The model, implemented in NetLogo 6.4.0, simulates teams of 2--10 agents performing tasks of varying complexity. We validate against Hancock et al.'s (2021) meta-analysis, achieving interval validity for 4 of 8 trust antecedent categories and strong ordinal validity (Spearman \r{ho}=0.833\rho = 0.833 \r{ho}=0.833). Sensitivity analysis using OFAT and full factorial designs (n=50n = 50 n=50 replications per condition) reveals robot reliability exhibits the strongest effect on trust ({\eta}2=0.35\eta^2 = 0.35 {\eta}2=0.35) and dominates task success ({\eta}2=0.93\eta^2 = 0.93 {\eta}2=0.93) and productivity ({\eta}2=0.89\eta^2 = 0.89 {\eta}2=0.89), consistent with meta-analytic findings. Trust asymmetry ratios ranged from 0.07 to 0.55 -- below the meta-analytic benchmark of 1.50 -- revealing that per-event asymmetry does not guarantee cumulative asymmetry when trust repair mechanisms remain active. Scenario analysis uncovered trust-performance decoupling: the Trust Recovery scenario achieved the highest productivity (4.29) despite the lowest trust (38.2), while the Unreliable Robot scenario produced the highest trust (73.2) despite the lowest task success (33.4\%), establishing calibration error as a critical diagnostic distinct from trust magnitude. Factorial ANOVA confirmed significant main effects for reliability, transparency, communication, and collaboration (p<.001p < .001 p<.001), explaining 45.4\% of trust variance. The open-source implementation provides an evidence-based tool for identifying overtrust and undertrust conditions prior to deployment.
>
---
#### [new 051] Towards Robot Skill Learning and Adaptation with Gaussian Processes
- **分类: cs.RO**

- **简介: 该论文属于机器人技能学习与适应任务，解决技能模型在环境变化下的适应问题。提出基于高斯过程的紧凑表达方法，实现鲁棒技能调整。**

- **链接: [https://arxiv.org/pdf/2603.01480](https://arxiv.org/pdf/2603.01480)**

> **作者:** A K M Nadimul Haque; Fouad Sukkar; Sheila Sujipto; Cedric Le Gentil; Marc G. Carmichael; Teresa Vidal-Calleja
>
> **摘要:** General robot skill adaptation requires expressive representations robust to varying task configurations. While recent learning-based skill adaptation methods refined via Reinforcement Learning (RL), have shown success, existing skill models often lack sufficient representational capacity for anything beyond minor environmental changes. In contrast, Gaussian Process (GP)-based skill modelling provides an expressive representation with useful analytical properties; however, adaptation of GP-based skills remains underexplored. This paper proposes a novel, robust skill adaptation framework that utilises GPs with sparse via-points for compact and expressive modelling. The model considers the trajectory's poses and leverages its first and second analytical derivatives to preserve the skill's kinematic profile. We present three adaptation methods to cater for the variability between initial and observed configurations. Firstly, an optimisation agent that adjusts the path's via-points while preserving the demonstration velocity. Second, a behaviour cloning agent trained to replicate output trajectories from the optimisation agent. Lastly, an RL agent that has learnt to modify via-points whilst maintaining the kinematic profile and enabling online capabilities. Evaluated across three tasks (drawer opening, cube-pushing and bar manipulation) in both simulation and hardware, our proposed methods outperform every benchmark in success rates. Furthermore, the results demonstrate that the GP-based representation enables all three methods to attain high cosine similarity and low velocity magnitude errors, indicating strong preservation of the kinematic profile. Overall, our formulation provides a compact representation capable of adapting to large deviations from a single demonstrated skill.
>
---
#### [new 052] Jailbreaking Embodied LLMs via Action-level Manipulation
- **分类: cs.RO**

- **简介: 该论文属于安全研究任务，旨在解决 embodied LLMs 的物理安全风险。工作是提出 Blindfold 框架，通过动作级操纵实现攻击，提升攻击成功率。**

- **链接: [https://arxiv.org/pdf/2603.01414](https://arxiv.org/pdf/2603.01414)**

> **作者:** Xinyu Huang; Qiang Yang; Leming Shen; Zijing Ma; Yuanqing Zheng
>
> **备注:** This paper has been officially accepted for ACM SenSys 2026
>
> **摘要:** Embodied Large Language Models (LLMs) enable AI agents to interact with the physical world through natural language instructions and actions. However, beyond the language-level risks inherent to LLMs themselves, embodied LLMs with real-world actuation introduce a new vulnerability: instructions that appear semantically benign may still lead to dangerous real-world consequences, revealing a fundamental misalignment between linguistic security and physical outcomes. In this paper, we introduce Blindfold, an automated attack framework that leverages the limited causal reasoning capabilities of embodied LLMs in real-world action contexts. Rather than iterative trial-and-error jailbreaking of black-box embodied LLMs, Blindfold adopts an Adversarial Proxy Planning strategy: it compromises a local surrogate LLM to perform action-level manipulations that appear semantically safe but could result in harmful physical effects when executed. Blindfold further conceals key malicious actions by injecting carefully crafted noise to evade detection by defense mechanisms, and it incorporates a rule-based verifier to improve the attack executability. Evaluations on both embodied AI simulators and a real-world 6DoF robotic arm show that Blindfold achieves up to 53% higher attack success rates than SOTA baselines, highlighting the urgent need to move beyond surface-level language censorship and toward consequence-aware defense mechanisms to secure embodied LLMs.
>
---
#### [new 053] A Novel Reconfigurable Dexterous Hand Based on Triple-Symmetric Bricard Parallel Mechanism
- **分类: cs.RO**

- **简介: 该论文属于机器人手部设计任务，旨在提升抓取的适应性与精度。通过引入三对称Bricard并联机构，实现可重构手掌，优化运动性能，增强操作稳定性与负载能力。**

- **链接: [https://arxiv.org/pdf/2603.00892](https://arxiv.org/pdf/2603.00892)**

> **作者:** Chunxu Tian; Zhichao Huang; Hongzeng Li; Bo Wang; Jinghao Jia; Yirui Sun; Dan Zhang
>
> **备注:** 8 pages, 14 figures, 2026 IEEE International Conference on Robotics & Automation
>
> **摘要:** This paper introduces a novel design for a robotic hand based on parallel mechanisms. The proposed hand uses a triple-symmetric Bricard linkage as its reconfigurable palm, enhancing adaptability to objects of varying shapes and sizes. Through topological and dimensional synthesis, the mechanism achieves a well-balanced degree of freedom and link configuration suitable for reconfigurable palm motion, balancing dexterity, stability, and load capacity. Furthermore, kinematic analysis is performed using screw theory and closed-loop constraints, and performance is evaluated based on workspace, stiffness, and motion/force transmission efficiency. Finally, a prototype is developed and tested through a series of grasping experiments, demonstrating the ability to perform stable and efficient manipulation across a wide range of objects. The results validate the effectiveness of the design in improving grasping versatility and operational precision, offering a promising solution for advanced robotic manipulation tasks.
>
---
#### [new 054] Optimal Solutions for the Moving Target Vehicle Routing Problem via Branch-and-Price with Relaxed Continuity
- **分类: cs.RO**

- **简介: 该论文研究MT-VRP任务，解决移动目标的路径规划问题。提出BPRC算法，通过改进的标签算法处理动态目标和时间依赖成本，实现更优解的快速求解。**

- **链接: [https://arxiv.org/pdf/2603.00663](https://arxiv.org/pdf/2603.00663)**

> **作者:** Anoop Bhat; Geordan Gutow; Zhongqiang Ren; Sivakumar Rathinam; Howie Choset
>
> **备注:** Accepted to ICAPS 2026
>
> **摘要:** The Moving Target Vehicle Routing Problem (MT-VRP) seeks trajectories for several agents that intercept a set of moving targets, subject to speed, time window, and capacity constraints. We introduce an exact algorithm, Branch-and-Price with Relaxed Continuity (BPRC), for the MT-VRP. The main challenge in a branch-and-price approach for the MT-VRP is the pricing subproblem, which is complicated by moving targets and time-dependent travel costs between targets. Our key contribution is a new labeling algorithm that solves this subproblem by means of a novel dominance criterion tailored for problems with moving targets. Numerical results on instances with up to 25 targets show that our algorithm finds optimal solutions more than an order of magnitude faster than a baseline based on previous work, showing particular strength in scenarios with limited agent capacities.
>
---
#### [new 055] TMR-VLA:Vision-Language-Action Model for Magnetic Motion Control of Tri-leg Silicone-based Soft Robot
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决磁控软体机器人导航问题。通过TMR-VLA系统，实现语言指令到电压控制的映射，提升机器人运动灵活性。**

- **链接: [https://arxiv.org/pdf/2603.00420](https://arxiv.org/pdf/2603.00420)**

> **作者:** Ruijie Tang; Chi Kit Ng; Kaixuan Wu; Long Bai; Guankun Wang; Yiming Huang; Yupeng Wang; Hongliang Ren
>
> **备注:** ICRA 2025
>
> **摘要:** In-vivo environments, magnetically actuated soft robots offer advantages such as wireless operation and precise control, showing promising potential for painless detection and therapeutic procedures. We developed a trileg magnetically driven soft robot (TMR) whose multi-legged design enables more flexible gaits and diverse motion patterns. For the silicone made of reconfigurable soft robots, its navigation ability can be separated into sequential motions, namely squatting, rotation, lifting a leg, walking and so on. Its motion and behavior depend on its bending shapes. To bridge motion type description and specific low-level voltage control, we introduced TMR-VLA, an end-to-end multi-modal system for a trileg magnetic soft robot capable of performing hybrid motion types, which is promising for developing a navigation ability by adapting its shape to language-constrained motion types. The TMR-VLA deploys embodied endoluminal localization ability from EndoVLA, and fuses sequential frames and natural language commands as input. Low-level voltage output is generated based on the current observation state and specific motion type description. The result shows the TMR-VLA can predict how the voltage applied to TMR will change the dynamics of a silicon-made soft robot. The TMR-VLA reached a 74% average success rate.
>
---
#### [new 056] Path Integral Particle Filtering for Hybrid Systems via Saltation Matrices
- **分类: cs.RO**

- **简介: 该论文属于状态估计任务，解决混合系统中接触事件的不确定性问题。通过路径积分优化控制与盐化矩阵，提出一种高效可靠的滤波算法。**

- **链接: [https://arxiv.org/pdf/2603.01176](https://arxiv.org/pdf/2603.01176)**

> **作者:** Karthik Shaji; Sreeranj Jayadevan; Bo Yuan; Hongzhe Yu; Yongxin Chen
>
> **摘要:** We present an optimal-control-based particle filtering method for state estimation in hybrid systems that undergo intermittent contact with their environments. We follow the path integral filtering framework that exploits the duality between the smoothing problem and optimal control. We leverage saltation matrices to map out the uncertainty propagation during contact events for hybrid systems. The resulting path integral optimal control problem allows for a state estimation algorithm robust to outlier effects, flexible to non-Gaussian noise distributions, that also handles the challenging contact dynamics in hybrid systems. This work offers a computationally efficient and reliable estimation algorithm for hybrid systems with stochastic dynamics. We also present extensive experimental results demonstrating that our approach consistently outperforms strong baselines across multiple settings.
>
---
#### [new 057] Learning Vision-Based Omnidirectional Navigation: A Teacher-Student Approach Using Monocular Depth Estimation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人导航任务，解决工业环境中2D LiDAR传感器无法检测三维障碍物的问题。通过教师-学生框架，利用单目深度估计实现无LiDAR的可靠导航。**

- **链接: [https://arxiv.org/pdf/2603.01999](https://arxiv.org/pdf/2603.01999)**

> **作者:** Jan Finke; Wayne Paul Martis; Adrian Schmelter; Lars Erbach; Christian Jestel; Marvin Wiedemann
>
> **摘要:** Reliable obstacle avoidance in industrial settings demands 3D scene understanding, but widely used 2D LiDAR sensors perceive only a single horizontal slice of the environment, missing critical obstacles above or below the scan plane. We present a teacher-student framework for vision-based mobile robot navigation that eliminates the need for LiDAR sensors. A teacher policy trained via Proximal Policy Optimization (PPO) in NVIDIA Isaac Lab leverages privileged 2D LiDAR observations that account for the full robot footprint to learn robust navigation. The learned behavior is distilled into a student policy that relies solely on monocular depth maps predicted by a fine-tuned Depth Anything V2 model from four RGB cameras. The complete inference pipeline, comprising monocular depth estimation (MDE), policy execution, and motor control, runs entirely onboard an NVIDIA Jetson Orin AGX mounted on a DJI RoboMaster platform, requiring no external computation for inference. In simulation, the student achieves success rates of 82-96.5%, consistently outperforming the standard 2D LiDAR teacher (50-89%). In real-world experiments, the MDE-based student outperforms the 2D LiDAR teacher when navigating around obstacles with complex 3D geometries, such as overhanging structures and low-profile objects, that fall outside the single scan plane of a 2D LiDAR.
>
---
#### [new 058] SSMG-Nav: Enhancing Lifelong Object Navigation with Semantic Skeleton Memory Graph
- **分类: cs.RO**

- **简介: 该论文属于服务机器人导航任务，解决长期环境下目标导航效率低的问题。提出SSMG-Nav框架，利用语义骨架记忆图提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.01813](https://arxiv.org/pdf/2603.01813)**

> **作者:** Haochen Niu; Lantao Zhang; Xingwu Ji; Rendong Ying; Peilin Liu; Fei Wen
>
> **备注:** Accepted by 2026 ICRA
>
> **摘要:** Navigating to out-of-sight targets from human instructions in unfamiliar environments is a core capability for service robots. Despite substantial progress, most approaches underutilize reusable, persistent memory, constraining performance in lifelong settings. Many are additionally limited to single-modality inputs and employ myopic greedy policies, which often induce inefficient back-and-forth maneuvers (BFMs). To address such limitations, we introduce SSMG-Nav, a framework for object navigation built on a \textit{Semantic Skeleton Memory Graph} (SSMG) that consolidates past observations into a spatially aligned, persistent memory anchored by topological keypoints (e.g., junctions, room centers). SSMG clusters nearby entities into subgraphs, unifying entity- and space-level semantics to yield a compact set of candidate destinations. To support multimodal targets (images, objects, and text), we integrate a vision-language model (VLM). For each subgraph, a multimodal prompt synthesized from memory guides the VLM to infer a target belief over destinations. A long-horizon planner then trades off this belief against traversability costs to produce a visit sequence that minimizes expected path length, thereby reducing backtracking. Extensive experiments on challenging lifelong benchmarks and standard ObjectNav benchmarks demonstrate that, compared to strong baselines, our method achieves higher success rates and greater path efficiency, validating the effectiveness of SSMG-Nav.
>
---
#### [new 059] Mean-Flow based One-Step Vision-Language-Action
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉-语言-动作（VLA）任务，旨在解决生成延迟问题。通过提出基于均值流的一步式方法，提升动作生成效率，实验证明其速度显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2603.01469](https://arxiv.org/pdf/2603.01469)**

> **作者:** Yang Chen; Xiaoguang Ma; Bin Zhao
>
> **摘要:** Recent advances in FlowMatching-based Vision-Language-Action (VLA) frameworks have demonstrated remarkable advantages in generating high-frequency action chunks, particularly for highly dexterous robotic manipulation tasks. Despite these notable achievements, their practical applications are constrained by prolonged generation latency, which stems from inherent iterative sampling requirements and architectural limitations. To address this critical bottleneck, we propose a Mean-Flow based One-Step VLA approach. Specifically, we resolve the noise-induced issues in the action generation process, thereby eliminating the consistency constraints inherent to conventional Flow-Matching methods. This significantly enhances generation efficiency and enables one-step action generation. Real-world robotic experiments show that the generation speed of the proposed Mean-Flow based One-Step VLA is 8.7 times and 83.9 times faster than that of SmolVLA and Diffusion Policy, respectively. These results elucidate its great potential as a high-efficiency backbone for VLA-based robotic manipulation.
>
---
#### [new 060] CHOP: Counterfactual Human Preference Labels Improve Obstacle Avoidance in Visuomotor Navigation Policies
- **分类: cs.RO**

- **简介: 该论文属于视觉运动导航任务，旨在提升机器人在复杂环境中的避障能力。通过引入基于人类偏好的反事实标签，优化导航策略，显著提高了安全性与路径效率。**

- **链接: [https://arxiv.org/pdf/2603.02004](https://arxiv.org/pdf/2603.02004)**

> **作者:** Gershom Seneviratne; Jianyu An; Vaibhav Shende; Sahire Ellahy; Yaxita Amin; Kondapi Manasanjani; Samarth Chopra; Jonathan Deepak Kannan; Dinesh Manocha
>
> **摘要:** Visuomotor navigation policies have shown strong perception-action coupling for embodied agents, yet they often struggle with safe navigation and dynamic obstacle avoidance in complex real-world environments. We introduce CHOP, a novel approach that leverages Counterfactual Human Preference Labels to align visuomotor navigation policies towards human intuition of safety and obstacle avoidance in navigation. In CHOP, for each visual observation, the robot's executed trajectory is included among a set of counterfactual navigation trajectories: alternative trajectories the robot could have followed under identical conditions. Human annotators provide pairwise preference labels over these trajectories based on anticipated outcomes such as collision risk and path efficiency. These aggregated preferences are then used to fine-tune visuomotor navigation policies, aligning their behavior with human preferences in navigation. Experiments on the SCAND dataset show that visuomotor navigation policies fine-tuned with CHOP reduce near-collision events by 49.7%, decrease deviation from human-preferred trajectories by 45.0%, and increase average obstacle clearance by 19.8% on average across multiple state-of-the-art models, compared to their pretrained baselines. These improvements transfer to real-world deployments on a Ghost Robotics Vision60 quadruped, where CHOP-aligned policies improve average goal success rates by 24.4%, increase minimum obstacle clearance by 6.8%, reduce collision and intervention events by 45.7%, and improve normalized path completion by 38.6% on average across navigation scenarios, compared to their pretrained baselines. Our results highlight the value of counterfactual preference supervision in bridging the gap between large-scale visuomotor policies and human-aligned, safety-aware embodied navigation.
>
---
#### [new 061] PEPA: a Persistently Autonomous Embodied Agent with Personalities
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人自主控制任务，旨在解决传统代理依赖外部指令的问题。通过引入人格特质，构建PEPA架构，实现自主目标生成与行为演化。**

- **链接: [https://arxiv.org/pdf/2603.00117](https://arxiv.org/pdf/2603.00117)**

> **作者:** Kaige Liu; Yang Li; Lijun Zhu; Weinan Zhang
>
> **摘要:** Living organisms exhibit persistent autonomy through internally generated goals and self-sustaining behavioral organization, yet current embodied agents remain driven by externally scripted objectives. This dependence on predefined task specifications limits their capacity for long-term deployment in dynamic, unstructured environments where continuous human intervention is impractical. We propose that personality traits provide an intrinsic organizational principle for achieving persistent autonomy. Analogous to genotypic biases shaping biological behavioral tendencies, personalities enable agents to autonomously generate goals and sustain behavioral evolution without external supervision. To realize this, we develop PEPA, a three-layer cognitive architecture that operates through three interacting systems: Sys3 autonomously synthesizes personality-aligned goals and refines them via episodic memory and daily self-reflection; Sys2 performs deliberative reasoning to translate goals into executable action plans; Sys1 grounds the agent in sensorimotor interaction, executing actions and recording experiences. We validate the framework through real-world deployment on a quadruped robot in a multi-floor office building. Operating without reliance on fixed task specifications, the robot autonomously arbitrates between user requests and personality-driven motivations, navigating elevators and exploring environments accordingly. Quantitative analysis across five distinct personality prototypes demonstrates stable, trait-aligned behaviors. The results confirm that personality-driven cognitive architectures enable sustained autonomous operation characteristic of persistent embodied systems. Code and demo videos are available at this https URL.
>
---
#### [new 062] Autonomous Block Assembly for Boom Cranes with Passive Joint Dynamics: Integrated Vision MPC Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于建筑自动化任务，解决吊装臂架在被动关节动力学下精确定位的问题。通过视觉与模型预测控制结合，实现自主块体抓取与避障装配。**

- **链接: [https://arxiv.org/pdf/2603.00103](https://arxiv.org/pdf/2603.00103)**

> **作者:** Gerald Ebmer; Minh Nhat Vu; Tobias Glück; Wolfgang Kemmetmüller
>
> **摘要:** This paper presents an autonomous control framework for articulated boom cranes performing prefabricated block assembly in construction environments. The key challenge addressed is precise placement control under passive joint dynamics that cause pendulum-like sway, complicating the accurate positioning of building components. Our integrated approach combines real-time vision-based pose estimation of building blocks, collision-aware B-spline path planning, and nonlinear model predictive control (NMPC) to achieve autonomous pickup, placement, and obstacle-avoidance assembly operations. The framework is validated on a laboratory-scale testbed that emulates crane kinematics and passive dynamics while enabling rapid experimentation. The collision-aware planner generates feasible B-spline references in real-time on CPU hardware with anytime performance, while the NMPC controller actively suppresses passive joint sway and tracks the planned trajectory under continuous vision feedback. Experimental results demonstrate autonomous block stacking and obstacle-avoidance assembly, with sway damping reducing settling times by more than an order of magnitude compared to uncontrolled passive dynamics, confirming the real-time feasibility of the integrated approach for construction automation.
>
---
#### [new 063] Certifiable Estimation with Factor Graphs
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人状态估计任务，旨在解决传统方法易陷入局部最优及凸松弛计算成本高的问题。通过将因子图与凸松弛结合，提出一种统一的可验证优化框架。**

- **链接: [https://arxiv.org/pdf/2603.01267](https://arxiv.org/pdf/2603.01267)**

> **作者:** Zhexin Xu; Nikolas R. Sanderson; Hanna Jiamei Zhang; David M. Rosen
>
> **摘要:** Factor graphs provide a convenient modular modeling language that enables practitioners to design and deploy high-performance robotic state estimation systems by composing simple, reusable building blocks. However, inference in these models is typically performed using local optimization methods that can converge to suboptimal solutions, a serious reliability concern in safety-critical applications. Conversely, certifiable estimators based on convex relaxation can recover verifiably globally optimal solutions in many practical settings, but the computational cost of solving their large-scale relaxations necessitates specialized, structure-exploiting solvers that require substantial expertise to implement, significantly hampering practical deployment. In this paper, we show that these two paradigms, which have thus far been treated as independent in the literature, can be naturally synthesized into a unified framework for certifiable factor graph optimization. The key insight is that factor graph structure is preserved under Shor's relaxation and Burer-Monteiro factorization: applying these transformations to a QCQP with an associated factor graph representation yields a lifted problem admitting a factor graph model with identical connectivity, in which variables and factors are simple one-to-one algebraic transformations of those in the original QCQP. This structural preservation enables the Riemannian Staircase methodology for certifiable estimation to be implemented using the same mature, highly-performant factor graph libraries and workflows already ubiquitously employed throughout robotics and computer vision, making certifiable estimation as straightforward to design and deploy as conventional factor graph inference.
>
---
#### [new 064] Planning Method for Skill-Based Control of Robots Using a PLC as Skill Trigger
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决技能编程中运动序列效率低的问题。通过引入优化的MoveContinuousSkill，提升执行效率。**

- **链接: [https://arxiv.org/pdf/2603.00555](https://arxiv.org/pdf/2603.00555)**

> **作者:** Andreas Gaugenrieder; Hari Hara Balasubramaniam; Jannik Möhrle; Rüdiger Daub
>
> **备注:** 6 pages, 3 figures, 2 tables, submitted to the 19th CIRP Conference on Intelligent Computation in Manufacturing Engineering - CIRP ICME '25, 16-18 July 2025, Ischia (Naples), Italy, has been officially accepted for publication in Procedia CIRP, ISSN: 2212-8271, where the Elsevier's copyright policy applies, and is currently in print
>
> **摘要:** Skill-based programming of robots provides a flexible approach for automation. Existing solutions neglect the optimization of motion sequences, leading to inefficiencies in execution. This work introduces a planning method that enhances skill-based robot programming by integrating motion sequence optimization. This optimization leads to a new MoveContinuousSkill. The software for executing the MoveContinuousSkill is implemented on a Programmable Logic Controller and applied across multiple robotic systems. Experimental results demonstrate a significant improvement in execution time through optimized motion sequence.
>
---
#### [new 065] Multimodal Adversarial Quality Policy for Safe Grasping
- **分类: cs.RO**

- **简介: 该论文属于安全抓取任务，解决DNN在HRI中的安全风险问题。提出MAQP框架，通过双模态优化和梯度平衡策略，提升RGBD模态下的抓取安全性。**

- **链接: [https://arxiv.org/pdf/2603.01479](https://arxiv.org/pdf/2603.01479)**

> **作者:** Kunlin Xie Chenghao Li Haolan Zhang; Nak Young Chong
>
> **备注:** submitted
>
> **摘要:** Vision-guided robot grasping based on Deep Neural Networks (DNNs) generalizes well but poses safety risks in the Human-Robot Interaction (HRI). Recent works solved it by designing benign adversarial attacks and patches with RGB modality, yet depth-independent characteristics limit their effectiveness on RGBD modality. In this work, we propose the Multimodal Adversarial Quality Policy (MAQP) to realize multimodal safe grasping. Our framework introduces two key components. First, the Heterogeneous Dual-Patch Optimization Scheme (HDPOS) mitigates the distribution discrepancy between RGB and depth modalities in patch generation by adopting modality-specific initialization strategies, employing a Gaussian distribution for depth patches and a uniform distribution for RGB patches, while jointly optimizing both modalities under a unified objective function. Second, the Gradient-Level Modality Balancing Strategy (GLMBS) is designed to resolve the optimization imbalance from RGB and Depth patches in patch shape adaptation by reweighting gradient contributions based on per-channel sensitivity analysis and applying distance-adaptive perturbation bounds. We conduct extensive experiments on the benchmark datasets and a cobot, showing the effectiveness of MAQP.
>
---
#### [new 066] PhysGraph: Physically-Grounded Graph-Transformer Policies for Bimanual Dexterous Hand-Tool-Object Manipulation
- **分类: cs.RO**

- **简介: 该论文属于双臂精细操作任务，解决工具使用中的高维状态和复杂接触动力学问题。提出PhysGraph模型，通过图结构表示系统，提升操作精度与成功率。**

- **链接: [https://arxiv.org/pdf/2603.01436](https://arxiv.org/pdf/2603.01436)**

> **作者:** Runfa Blark Li; David Kim; Xinshuang Liu; Keito Suzuki; Dwait Bhatt; Nikola Raicevic; Xin Lin; Ki Myung Brian Lee; Nikolay Atanasov; Truong Nguyen
>
> **摘要:** Bimanual dexterous manipulation for tool use remains a formidable challenge in robotics due to the high-dimensional state space and complicated contact dynamics. Existing methods naively represent the entire system state as a single configuration vector, disregarding the rich structural and topological information inherent to articulated hands. We present PhysGraph, a physically-grounded graph transformer policy designed explicitly for challenging bimanual hand-tool-object manipulation. Unlike prior works, we represent the bimanual system as a kinematic graph and introduce per-link tokenization to preserve fine-grained local state information. We propose a physically-grounded bias generator that injects structural priors directly into the attention mechanism, including kinematic spatial distance, dynamic contact states, geometric proximity, and anatomical properties. This allows the policy to explicitly reason about physical interactions rather than learning them implicitly from sparse rewards. Extensive experiments show that PhysGraph significantly outperforms baseline - ManipTrans in manipulation precision and task success rates while using only 51% of the parameters of ManipTrans. Furthermore, the inherent topological flexibility of our architecture shows qualitative zero-shot transfer to unseen tool/object geometries, and is sufficiently general to be trained on three robotic hands (Shadow, Allegro, Inspire).
>
---
#### [new 067] SurgFusion-Net: Diversified Adaptive Multimodal Fusion Network for Surgical Skill Assessment
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于手术技能评估任务，解决多模态数据融合难题，提出SurgFusion-Net和DRA方法，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2603.00108](https://arxiv.org/pdf/2603.00108)**

> **作者:** Runlong He; Freweini M. Tesfai; Matthew W. E. Boal; Nazir Sirajudeen; Dimitrios Anastasiou; Jialang Xu; Mobarak I. Hoque; Philip J. Edwards; John D. Kelly; Ashwin Sridhar; Abdolrahim Kadkhodamohammadi; Dhivya Chandrasekaran; Matthew J. Clarkson; Danail Stoyanov; Nader Francis; Evangelos B. Mazomenos
>
> **摘要:** Robotic-assisted surgery (RAS) is established in clinical practice, and automated surgical skill assessment utilizing multimodal data offers transformative potential for surgical analytics and education. However, developing effective multimodal methods remains challenging due to the task complexity, limited annotated datasets and insufficient techniques for cross-modal information fusion. Existing state-of-the-art relies exclusively on RGB video and only applies on dry-lab settings, failing to address the significant domain gap between controlled simulation and real clinical cases, where the surgical environment together with camera and tissue motion introduce substantial complexities. This work introduces SurgFusion-Net and Divergence Regulated Attention (DRA), an innovative fusion strategy for multimodal surgical skill assessment. We contribute two first-of-their-kind clinical datasets: the RAH-skill dataset containing 279,691 RGB frames from 37 videos of Robot-assisted Hysterectomy (RAH), and the RARP-skill dataset containing 70,661 RGB frames from 33 videos of Robot-Assisted Radical Prostatectomy (RARP). Both datasets include M-GEARS skill annotations, corresponding optical flow and tool segmentation masks. DRA incorporates adaptive dual attention and diversity-promoting multi-head attention to fuse multimodal information, from three modalities, based on surgical context, enhancing assessment accuracy and reliability. Validated on the JIGSAWS benchmark, RAH-skill, and RARP-skill datasets, our approach outperforms recent baselines with SCC improvements of 0.02 in LOSO, 0.04 in LOUO across JIGSAWS tasks, and 0.0538 and 0.0493 gains on RAH-skill and RARP-skill, respectively.
>
---
#### [new 068] riMESA: Consensus ADMM for Real-World Collaborative SLAM
- **分类: cs.RO**

- **简介: 该论文提出riMESA，解决多机器人协同SLAM中的通信限制、异常数据和实时性问题，采用共识ADMM方法实现高效分布式优化。**

- **链接: [https://arxiv.org/pdf/2603.01178](https://arxiv.org/pdf/2603.01178)**

> **作者:** Daniel McGann; Michael Kaess
>
> **摘要:** Collaborative Simultaneous Localization and Mapping (C-SLAM) is a fundamental capability for multi-robot teams as it enables downstream tasks like planning and navigation. However, existing C-SLAM back-end algorithms that are required to solve this problem struggle to address the practical realities of real-world deployments (i.e. communication limitations, outlier measurements, and online operation). In this paper we propose Robust Incremental Manifold Edge-based Separable ADMM (riMESA) -- a robust, incremental, and distributed C-SLAM back-end that is resilient to outliers, reliable in the face of limited communication, and can compute accurate state estimates for a multi-robot team in real-time. Through the development of riMESA, we, more broadly, make an argument for the use of Consensus Alternating Direction Method of Multipliers as a theoretical foundation for distributed optimization tasks in robotics like C-SLAM due to its flexibility, accuracy, and fast convergence. We conclude this work with an in-depth evaluation of riMESA on a variety of C-SLAM problem scenarios and communication network conditions using both synthetic and real-world C-SLAM data. These experiments demonstrate that riMESA is able to generalize across conditions, produce accurate state estimates, operate in real-time, and outperform the accuracy of prior works by a factor >7x on real-world datasets.
>
---
#### [new 069] Test-Driven Agentic Framework for Reliable Robot Controller
- **分类: cs.RO; eess.SY**

- **链接: [https://arxiv.org/pdf/2603.00455](https://arxiv.org/pdf/2603.00455)**

> **作者:** Shivanshu Tripathi; Reza Akbarian Bafghi; Maziar Raissi
>
> **摘要:** In this work, we present a test-driven, agentic framework for synthesizing a deployable low-level robot controller for navigation tasks. Given a 2D map with an image of an ultrasonic sensor-based robot, or a 3D robotic simulation environment, our framework iteratively refines the generated controller code using diagnostic feedback from structured test suites to achieve task success. We propose a dual-tier repair strategy to refine the generated code that alternates between prompt-level refinement and direct code editing. We evaluate the approach across 2D navigation tasks and 3D navigation in the Webots simulator. Experimental results show that test-driven synthesis substantially improves controller reliability and robustness over one-shot controller generation, especially when the initial prompt is underspecified. The source code and demonstration videos are available at: this https URL.
>
---
#### [new 070] ACDC: Adaptive Curriculum Planning with Dynamic Contrastive Control for Goal-Conditioned Reinforcement Learning in Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作中的目标条件强化学习任务，旨在解决现有方法依赖经验优先级导致性能不佳的问题。提出ACDC框架，结合自适应课程规划与动态对比控制，提升学习效率和任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.02104](https://arxiv.org/pdf/2603.02104)**

> **作者:** Xuerui Wang; Guangyu Ren; Tianhong Dai; Bintao Hu; Shuangyao Huang; Wenzhang Zhang; Hengyan Liu
>
> **备注:** 13 pages (including references and appendix), 12 figures. Accepted to ICAPS 2026. Code available at this https URL
>
> **摘要:** Goal-conditioned reinforcement learning has shown considerable potential in robotic manipulation; however, existing approaches remain limited by their reliance on prioritizing collected experience, resulting in suboptimal performance across diverse tasks. Inspired by human learning behaviors, we propose a more comprehensive learning paradigm, ACDC, which integrates multidimensional Adaptive Curriculum (AC) Planning with Dynamic Contrastive (DC) Control to guide the agent along a well-designed learning trajectory. More specifically, at the planning level, the AC component schedules the learning curriculum by dynamically balancing diversity-driven exploration and quality-driven exploitation based on the agent's success rate and training progress. At the control level, the DC component implements the curriculum plan through norm-constrained contrastive learning, enabling magnitude-guided experience selection aligned with the current curriculum focus. Extensive experiments on challenging robotic manipulation tasks demonstrate that ACDC consistently outperforms the state-of-the-art baselines in both sample efficiency and final task success rate.
>
---
#### [new 071] Modeling PWM-Time-SOC Interaction in a Simulated Robot
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人能源管理任务，旨在解决电池SOC预测问题。通过仿真和模型构建，研究PWM与SOC的关系，实现能量高效规划。**

- **链接: [https://arxiv.org/pdf/2603.00319](https://arxiv.org/pdf/2603.00319)**

> **作者:** Vidyut Pradeep; Shirantha Welikala
>
> **摘要:** Accurate prediction of battery state of charge is needed for autonomous robots to plan movements without using up all available power. This work develops a physics and data-informed model from a simulation that predicts SOC depletion as a function of time and PWM duty cycle for a simulated 4-wheel Arduino robot. A forward-motion simulation incorporating motor electrical characteristics (resistance, inductance, back-EMF, torque constant) and mechanical dynamics (mass, drag, rolling resistance, wheel radius) was used to generate SOC time-series data across PWM values from 1-100%. Sparse Identification of Nonlinear Dynamics (SINDy), combined with least-squares regression, was applied to construct a unified nonlinear model that captures SOC(t, p). The framework allows for energy-aware planning for similar robots and can be extended to incorporate arbitrary initial SOC levels and environment-dependent parameters for real-world deployment.
>
---
#### [new 072] B$^2$F-Map: Crowd-sourced Mapping with Bayesian B-spline Fusion
- **分类: cs.RO**

- **简介: 该论文属于高精度地图生成任务，解决无先验HD地图下多车辆数据融合的不确定性问题。通过贝叶斯B样条融合方法，实现基于单目相机和低成本传感器的车道线地图构建。**

- **链接: [https://arxiv.org/pdf/2603.01673](https://arxiv.org/pdf/2603.01673)**

> **作者:** Yiping Xie; Yuxuan Xia; Erik Stenborg; Junsheng Fu; Axel Beauvisage; Gabriel E. Garcia; Tianyu Wu; Gustaf Hendeby
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Crowd-sourced mapping offers a scalable alternative to creating maps using traditional survey vehicles. Yet, existing methods either rely on prior high-definition (HD) maps or neglect uncertainties in the map fusion. In this work, we present a complete pipeline for HD map generation using production vehicles equipped only with a monocular camera, consumer-grade GNSS, and IMU. Our approach includes on-cloud localization using lightweight standard-definition maps, on-vehicle mapping via an extended object trajectory (EOT) Poisson multi-Bernoulli (PMB) filter with Gibbs sampling, and on-cloud multi-drive optimization and Bayesian map fusion. We represent the lane lines using B-splines, where each B-spline is parameterized by a sequence of Gaussian distributed control points, and propose a novel Bayesian fusion framework for B-spline trajectories with differing density representation, enabling principled handling of uncertainties. We evaluate our proposed approach, B$^2$F-Map, on large-scale real-world datasets collected across diverse driving conditions and demonstrate that our method is able to produce geometrically consistent lane-level maps.
>
---
#### [new 073] Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究 fisheye 相机在机器人操作中的应用，解决其对策略学习的影响问题。通过实验分析空间定位、场景泛化和硬件泛化，提出改进方法以提升性能。**

- **链接: [https://arxiv.org/pdf/2603.02139](https://arxiv.org/pdf/2603.02139)**

> **作者:** Han Xue; Nan Min; Xiaotong Liu; Wendi Chen; Yuan Fang; Jun Lv; Cewu Lu; Chuan Wen
>
> **备注:** 22 pages, 15 figures, Accecpted by CVPR 2026
>
> **摘要:** The adoption of fisheye cameras in robotic manipulation, driven by their exceptionally wide Field of View (FoV), is rapidly outpacing a systematic understanding of their downstream effects on policy learning. This paper presents the first comprehensive empirical study to bridge this gap, rigorously analyzing the properties of wrist-mounted fisheye cameras for imitation learning. Through extensive experiments in both simulation and the real world, we investigate three critical research questions: spatial localization, scene generalization, and hardware generalization. Our investigation reveals that: (1) The wide FoV significantly enhances spatial localization, but this benefit is critically contingent on the visual complexity of the environment. (2) Fisheye-trained policies, while prone to overfitting in simple scenes, unlock superior scene generalization when trained with sufficient environmental diversity. (3) While naive cross-camera transfer leads to failures, we identify the root cause as scale overfitting and demonstrate that hardware generalization performance can be improved with a simple Random Scale Augmentation (RSA) strategy. Collectively, our findings provide concrete, actionable guidance for the large-scale collection and effective use of fisheye datasets in robotic learning. More results and videos are available on this https URL
>
---
#### [new 074] Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons
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
#### [new 075] Learning Thermal-Aware Locomotion Policies for an Electrically-Actuated Quadruped Robot
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决电动四足机器人电机过热问题。通过将温度纳入强化学习策略，引入热约束奖励，提升机器人持续运行能力。**

- **链接: [https://arxiv.org/pdf/2603.01631](https://arxiv.org/pdf/2603.01631)**

> **作者:** Letian Qian; Yuhang Wan; Shuhan Wang; Xin Luo
>
> **摘要:** Electrically-actuated quadrupedal robots possess high mobility on complex terrains, but their motors tend to accumulate heat under high-torque cyclic loads, potentially triggering overheat protection and limiting long-duration tasks. This work proposes a thermal-aware control method that incorporates motor temperatures into reinforcement learning locomotion policies and introduces thermal-constraint rewards to prevent temperature exceedance. Real-world experiments on the Unitree A1 demonstrate that, under a fixed 3 kg payload, the baseline policy triggers overheat protection and stops within approximately 7 minutes, whereas the proposed method can operate continuously for over 27 minutes without thermal interruptions while maintaining comparable command-tracking performance, thereby enhancing sustainable operational capability.
>
---
#### [new 076] DAM-VLA: A Dynamic Action Model-Based Vision-Language-Action Framework for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文提出DAM-VLA框架，解决机器人在动态环境中执行复杂操作任务时的精准控制问题，通过融合视觉语言模型与扩散动作模型实现高效操作。**

- **链接: [https://arxiv.org/pdf/2603.00926](https://arxiv.org/pdf/2603.00926)**

> **作者:** Xiongfeng Peng; Jiaqian Yu; Dingzhe Li; Yixiang Jin; Lu Xu; Yamin Mao; Chao Zhang; Weiming Li; Sujin Jang; Dongwook Lee; Daehyun Ji
>
> **备注:** Accepted to ICRA2026
>
> **摘要:** In dynamic environments such as warehouses, hospitals, and homes, robots must seamlessly transition between gross motion and precise manipulations to complete complex tasks. However, current Vision-Language-Action (VLA) frameworks, largely adapted from pre-trained Vision-Language Models (VLMs), often struggle to reconcile general task adaptability with the specialized precision required for intricate manipulation. To address this challenge, we propose DAM-VLA, a dynamic action model-based VLA framework. DAM-VLA integrates VLM reasoning with diffusion-based action models specialized for arm and gripper control. Specifically, it introduces (i) an action routing mechanism, using task-specific visual and linguistic cues to select appropriate action models (e.g., arm movement or gripper manipulation), (ii) a dynamic action model that fuses high-level VLM cognition with low-level visual features to predict actions, and (iii) a dual-scale action weighting mechanism that enables dynamic coordination between the arm-movement and gripper-manipulation models. Across extensive evaluations, DAM-VLA achieves superior success rates compared to state-of-the-art VLA methods in simulated (SIMPLER, FurnitureBench) and real-world settings, showing robust generalization from standard pick-and-place to demanding long-horizon and contact-rich tasks.
>
---
#### [new 077] D-GVIO: A Buffer-Driven and Efficient Decentralized GNSS-Visual-Inertial State Estimator for Multi-Agent Systems
- **分类: cs.RO**

- **简介: 该论文属于多智能体协同定位任务，旨在解决资源受限平台上的实时性、鲁棒性和计算效率问题。提出D-GVIO框架，通过缓冲策略和L-IEKF实现高效分布式状态估计。**

- **链接: [https://arxiv.org/pdf/2603.01404](https://arxiv.org/pdf/2603.01404)**

> **作者:** Yarong Luo; Wentao Lu; Chi Guo
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Cooperative localization is essential for swarm applications like collaborative exploration and search-and-rescue missions. However, maintaining real-time capability, robustness, and computational efficiency on resource-constrained platforms presents significant challenges. To address these challenges, we propose D-GVIO, a buffer-driven and fully decentralized GNSS-Visual-Inertial Odometry (GVIO) framework that leverages a novel buffering strategy to support efficient and robust distributed state estimation. The proposed framework is characterized by four core mechanisms. Firstly, through covariance segmentation, covariance intersection and buffering strategy, we modularize propagation and update steps in distributed state estimation, significantly reducing computational and communication burdens. Secondly, the left-invariant extended Kalman filter (L-IEKF) is adopted for information fusion, which exhibits superior state estimation performance over the traditional extended Kalman filter (EKF) since its state transition matrix is independent of the system state. Thirdly, a buffer-based re-propagation strategy is employed to handle delayed measurements efficiently and accurately by leveraging the L-IEKF, eliminating the need for costly re-computation. Finally, an adaptive buffer-driven outlier detection method is proposed to dynamically cull GNSS outliers, enhancing robustness in GNSS-challenged environments.
>
---
#### [new 078] RAG-RUSS: A Retrieval-Augmented Robotic Ultrasound for Autonomous Carotid Examination
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决传统超声检查依赖操作者的问题。提出RAG-RUSS框架，实现自主、可解释的颈动脉检查。**

- **链接: [https://arxiv.org/pdf/2603.01153](https://arxiv.org/pdf/2603.01153)**

> **作者:** Dianye Huang; Ziping Cong; Nassir Navab; Zhongliang Jiang
>
> **备注:** Accepted by ICRA
>
> **摘要:** Robotic ultrasound (US) has recently attracted increasing attention as a means to overcome the limitations of conventional US examinations, such as the strong operator dependence. However, the decision-making process of existing methods is often either rule-based or relies on end-to-end learning models that operate as black boxes. This has been seen as a main limit for clinical acceptance and raises safety concerns for widespread adoption in routine practice. To tackle this challenge, we introduce the RAG-RUSS, an interpretable framework capable of performing a full carotid examination in accordance with the clinical workflow while explicitly explaining both the current stage and the next planned action. Furthermore, given the scarcity of medical data, we incorporate retrieval-augmented generation to enhance generalization and reduce dependence on large-scale training datasets. The method was trained on data acquired from 28 volunteers, while an additional four volumetric scans recorded from previously unseen volunteers were reserved for testing. The results demonstrate that the method can explain the current scanning stage and autonomously plan probe motions to complete the carotid examination, encompassing both transverse and longitudinal planes.
>
---
#### [new 079] Geometric Look-Angle Shaping Strategy for Enclosed Inspection
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于无人机自主巡检任务，旨在解决封闭区域巡检中的路径规划与引导问题。提出GLASS策略，通过几何约束的视角角设计，实现稳定、安全的巡检路径。**

- **链接: [https://arxiv.org/pdf/2603.00325](https://arxiv.org/pdf/2603.00325)**

> **作者:** Amit Shivam; Manuel C.R.M. Fernandes; Sergio Vinha; Fernando A.C.C. Fontes
>
> **备注:** Preprinted submitted to ICUAS 2026
>
> **摘要:** This paper introduces inspection through GLASS, a Geometric Look-Angle Shaping Strategy for enclosed regions using unmanned aerial vehicles. In doing so, the vehicles guidance command is constructed through a bounded, geometry-consistent shaping of the look angle relative to a desired standoff path. By embedding a smooth, hyperbolic-tangent-type shaping function within a polar geometric framework, GLASS ensures global existence of the guidance dynamics. It avoids the far-field limitations inherent to conventional formulations. Lyapunov stability analysis establishes asymptotic convergence to a prescribed inspection standoff under explicit curvature feasibility conditions, along with analytical settling-time characteristics. The proposed strategy incorporates maximum turn-rate constraints without inducing singularities throughout the workspace. High-fidelity six-degree-of-freedom quadrotor simulations demonstrate the effectiveness of GLASS in representative enclosed inspection scenarios, highlighting a practically viable guidance framework for autonomous enclosed inspection missions.
>
---
#### [new 080] Validation of Space Robotics in Underwater Environments via Disturbance Robustness Equivalency
- **分类: cs.RO**

- **简介: 该论文属于空间机器人验证任务，解决水下环境与微重力环境的动态差异问题。通过等价扰动鲁棒性，实现空间任务在水下的有效验证。**

- **链接: [https://arxiv.org/pdf/2603.00628](https://arxiv.org/pdf/2603.00628)**

> **作者:** Joris Verhagen; Elias Krantz; Chelsea Sidrane; David Dörner; Nicola De Carli; Pedro Roque; Huina Mao; Gunnar Tibert; Ivan Stenius; Christer Fuglesang; Dimos Dimarogonas; Jana Tumova
>
> **备注:** 8 pages, 5 figures, 1 table
>
> **摘要:** We present an experimental validation framework for space robotics that leverages underwater environments to approximate microgravity dynamics. While neutral buoyancy conditions make underwater robotics an excellent platform for space robotics validation, there are still dynamical and environmental differences that need to be overcome. Given a high-level space mission specification, expressed in terms of a Signal Temporal Logic specification, we overcome these differences via the notion of maximal disturbance robustness of the mission. We formulate the motion planning problem such that the original space mission and the validation mission achieve the same disturbance robustness degree. The validation platform then executes its mission plan using a near-identical control strategy to the space mission where the closed-loop controller considers the spacecraft dynamics. Evaluating our validation framework relies on estimating disturbances during execution and comparing them to the disturbance robustness degree, providing practical evidence of operation in the space environment. Our evaluation features a dual-experiment setup: an underwater robot operating under near-neutral buoyancy conditions to validate the planning and control strategy of either an experimental planar spacecraft platform or a CubeSat in a high-fidelity space dynamics simulator.
>
---
#### [new 081] Real-Time Thermal-Inertial Odometry on Embedded Hardware for High-Speed GPS-Denied Flight
- **分类: cs.RO**

- **简介: 该论文属于视觉惯性导航任务，解决高速、无GPS环境下的定位问题。融合热成像与惯性数据，提升飞行稳定性与精度。**

- **链接: [https://arxiv.org/pdf/2603.02114](https://arxiv.org/pdf/2603.02114)**

> **作者:** Austin Stone; Mark Petersen; Cammy Peterson
>
> **摘要:** We present a real-time monocular thermal-inertial odometry system designed for high-velocity, GPS-denied flight on embedded hardware. The system fuses measurements from a FLIR Boson+ 640 longwave infrared camera, a high-rate IMU, a laser range finder, a barometer, and a magnetometer within a fixed-lag factor graph. To sustain reliable feature tracks under motion blur, low contrast, and rapid viewpoint changes, we employ a lightweight thermal-optimized front-end with multi-stage feature filtering. Laser range finder measurements provide per-feature depth priors that stabilize scale during weakly observable motion. High-rate inertial data is first pre-filtered using a Chebyshev Type II infinite impulse response (IIR) filter and then preintegrated, improving robustness to airframe vibrations during aggressive maneuvers. To address barometric altitude errors induced at high airspeeds, we train an uncertainty-aware gated recurrent unit (GRU) network that models the temporal dynamics of static pressure distortion, outperforming polynomial and multi-layer perceptron (MLP) baselines. Integrated on an NVIDIA Jetson Xavier NX, the complete system supports closed-loop quadrotor flight at 30 m/s with drift under 2% over kilometer-scale trajectories. These contributions expand the operational envelope of thermal-inertial navigation, enabling reliable high-speed flight in visually degraded and GPS-denied environments.
>
---
#### [new 082] A User Study on the Suitability of Teleoperation Interfaces for Primitive Manipulation Tasks
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互领域，研究 teleoperation 接口对基础操作任务的适用性。旨在比较 3D 鼠标与 VR 控制器在不同任务中的表现，以优化接口设计。**

- **链接: [https://arxiv.org/pdf/2603.00020](https://arxiv.org/pdf/2603.00020)**

> **作者:** Jun Aoki; Shunki Itadera
>
> **备注:** Accepted at 21st ACM/IEEE International Conference on Human-Robot Interaction (HRI'26), Late Breaking Report
>
> **摘要:** The application of teleoperation to control robotic arms has been widely explored, and user-friendly teleoperation systems have been studied for facilitating higher performance and lower operational burden. To investigate the dominant factors in a practical teleoperation system, this study focused on the characteristics of an interface used to operate a robotic arm. The usability of an interface depends on the characteristics of the manipulation tasks to be completed; however, systematic comparisons of different interfaces across different tasks remain limited. In this study, we compared two widely used teleoperation interfaces, a 3D mouse and a VR controller, for two simple yet broadly applicable tasks with a six-degree-of-freedom (6DoF) robotic arm: repetitively pushing buttons and rotating knobs. Participants (N = 23) controlled a robotic arm with 6DoF to push buttons and rotate knobs as many times as possible in 3-minute trials. Each trial was followed by a NASA-TLX workload rating. The results showed a clear connection between the interface and task performance: the VR controller yielded higher performance for pushing buttons, whereas the 3D mouse performed better and was less demanding for knob rotation. These findings highlight the importance of considering dominant motion primitives of the task when designing practical teleoperation interfaces.
>
---
#### [new 083] ROSER: Few-Shot Robotic Sequence Retrieval for Scalable Robot Learning
- **分类: cs.RO**

- **简介: 该论文提出ROSER，解决机器人学习中任务标注数据稀缺的问题。通过少样本序列检索，从连续日志中提取任务相关片段，提升数据利用率。**

- **链接: [https://arxiv.org/pdf/2603.01474](https://arxiv.org/pdf/2603.01474)**

> **作者:** Zillur Rahman; Eddison Pham; Alejandro Daniel Noel; Cristian Meo
>
> **备注:** 2026 ICLR DATA-FM Workshop
>
> **摘要:** A critical bottleneck in robot learning is the scarcity of task-labeled, segmented training data, despite the abundance of large-scale robotic datasets recorded as long, continuous interaction logs. Existing datasets contain vast amounts of diverse behaviors, yet remain structurally incompatible with modern learning frameworks that require cleanly segmented, task-specific trajectories. We address this data utilization crisis by formalizing robotic sequence retrieval: the task of extracting reusable, task-centric segments from unlabeled logs using only a few reference examples. We introduce ROSER, a lightweight few-shot retrieval framework that learns task-agnostic metric spaces over temporal windows, enabling accurate retrieval with as few as 3-5 demonstrations, without any task-specific training required. To validate our approach, we establish comprehensive evaluation protocols and benchmark ROSER against classical alignment methods, learned embeddings, and language model baselines across three large-scale datasets (e.g., LIBERO, DROID, and nuScenes). Our experiments demonstrate that ROSER consistently outperforms all prior methods in both accuracy and efficiency, achieving sub-millisecond per-match inference while maintaining superior distributional alignment. By reframing data curation as few-shot retrieval, ROSER provides a practical pathway to unlock underutilized robotic datasets, fundamentally improving data availability for robot learning.
>
---
#### [new 084] AI-IO: An Aerodynamics-Inspired Real-Time Inertial Odometry for Quadrotors
- **分类: cs.RO**

- **简介: 该论文属于无人机导航任务，解决惯性里程计的精度与泛化问题。通过引入旋翼速度数据和Transformer架构，提升速度预测准确率，并结合EKF实现实时高精度定位。**

- **链接: [https://arxiv.org/pdf/2603.00597](https://arxiv.org/pdf/2603.00597)**

> **作者:** Jiahao Cui; Feng Yu; Linzuo Zhang; Yu Hu; Danping Zou
>
> **备注:** 8 pages, 8 figures, 2026 IEEE International Conference on Robotics(ICRA 2026)
>
> **摘要:** Inertial Odometry (IO) has gained attention in quadrotor applications due to its sole reliance on inertial measurement units (IMUs), attributed to its lightweight design, low cost, and robust performance across diverse environments. However, most existing learning-based inertial odometry systems for quadrotors either use only IMU data or include additional dynamics-related inputs such as thrust, but still lack a principled formulation of the underlying physical model to be learned. This lack of interpretability hampers the model's ability to generalize and often limits its accuracy. In this work, we approach the inertial odometry learning problem from a different perspective. Inspired by the aerodynamics model and IMU measurement model, we identify the key physical quantity--rotor speed measurements required for inertial odometry and design a transformer-based inertial odometry. By incorporating rotor speed measurements, the proposed model improves velocity prediction accuracy by 36.9%. Furthermore, the transformer architecture more effectively exploits temporal dependencies for denoising and aerodynamics modeling, yielding an additional 22.4% accuracy gain over previous results. To support evaluation, we also provide a real-world quadrotor flight dataset capturing IMU measurements and rotor speed for high-speed motion. Finally, combined with an uncertainty-aware extended Kalman filter (EKF), our framework is validated across multiple datasets and real-time systems, demonstrating superior accuracy, generalization, and real-time performance. We share the code and data to promote further research (this https URL).
>
---
#### [new 085] Unifying Language-Action Understanding and Generation for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决语言指令与动作输出对齐不足及生成效率低的问题。提出LinkVLA模型，通过统一语言与动作编码、引入辅助任务和改进生成方法提升性能。**

- **链接: [https://arxiv.org/pdf/2603.01441](https://arxiv.org/pdf/2603.01441)**

> **作者:** Xinyang Wang; Qian Liu; Wenjie Ding; Zhao Yang; Wei Li; Chang Liu; Bailin Li; Kun Zhan; Xianpeng Lang; Wei Chen
>
> **摘要:** Vision-Language-Action (VLA) models are emerging as a promising paradigm for end-to-end autonomous driving, valued for their potential to leverage world knowledge and reason about complex driving scenes. However, existing methods suffer from two critical limitations: a persistent misalignment between language instructions and action outputs, and the inherent inefficiency of typical auto-regressive action generation. In this paper, we introduce LinkVLA, a novel architecture that directly addresses these challenges to enhance both alignment and efficiency. First, we establish a structural link by unifying language and action tokens into a shared discrete codebook, processed within a single multi-modal model. This structurally enforces cross-modal consistency from the ground up. Second, to create a deep semantic link, we introduce an auxiliary action understanding objective that trains the model to generate descriptive captions from trajectories, fostering a bidirectional language-action mapping. Finally, we replace the slow, step-by-step generation with a two-step coarse-to-fine generation method C2F that efficiently decodes the action sequence, saving 86% inference time. Experiments on closed-loop driving benchmarks show consistent gains in instruction following accuracy and driving performance, alongside reduced inference latency.
>
---
#### [new 086] LEAR: Learning Edge-Aware Representations for Event-to-LiDAR Localization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，解决事件相机与LiDAR点云对齐问题。提出LEAR框架，联合估计边缘结构和深度光流，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2603.01839](https://arxiv.org/pdf/2603.01839)**

> **作者:** Kuangyi Chen; Jun Zhang; Yuxi Hu; Yi Zhou; Friedrich Fraundorfer
>
> **摘要:** Event cameras offer high-temporal-resolution sensing that remains reliable under high-speed motion and challenging lighting, making them promising for localization from LiDAR point clouds in GPS-denied and visually degraded environments. However, aligning sparse, asynchronous events with dense LiDAR maps is fundamentally ill-posed, as direct correspondence estimation suffers from modality gaps. We propose LEAR, a dual-task learning framework that jointly estimates edge structures and dense event-depth flow fields to bridge the sensing-modality divide. Instead of treating edges as a post-hoc aid, LEAR couples them with flow estimation through a cross-modal fusion mechanism that injects modality-invariant geometric cues into the motion representation, and an iterative refinement strategy that enforces mutual consistency between the two tasks over multiple update steps. This synergy produces edge-aware, depth-aligned flow fields that enable more robust and accurate pose recovery via Perspective-n-Point (PnP) solvers. On several popular and challenging datasets, LEAR achieves superior performance over the best prior method. The source code, trained models, and demo videos are made publicly available online.
>
---
#### [new 087] Scaling Tasks, Not Samples: Mastering Humanoid Control through Multi-Task Model-Based Reinforcement Learning
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决机器人多技能学习效率低的问题。通过多任务模型强化学习，提升样本效率和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.01452](https://arxiv.org/pdf/2603.01452)**

> **作者:** Shaohuai Liu; Weirui Ye; Yilun Du; Le Xie
>
> **摘要:** Developing generalist robots capable of mastering diverse skills remains a central challenge in embodied AI. While recent progress emphasizes scaling model parameters and offline datasets, such approaches are limited in robotics, where learning requires active interaction. We argue that effective online learning should scale the \emph{number of tasks}, rather than the number of samples per task. This regime reveals a structural advantage of model-based reinforcement learning (MBRL). Because physical dynamics are invariant across tasks, a shared world model can aggregate multi-task experience to learn robust, task-agnostic representations. In contrast, model-free methods suffer from gradient interference when tasks demand conflicting actions in similar states. Task diversity therefore acts as a regularizer for MBRL, improving dynamics learning and sample efficiency. We instantiate this idea with \textbf{EfficientZero-Multitask (EZ-M)}, a sample-efficient multi-task MBRL algorithm for online learning. Evaluated on \textbf{HumanoidBench}, a challenging whole-body control benchmark, EZ-M achieves state-of-the-art performance with significantly higher sample efficiency than strong baselines, without extreme parameter scaling. These results establish task scaling as a critical axis for scalable robotic learning. The project website is available \href{this https URL}{here}.
>
---
#### [new 088] Align and Filter: Improving Performance in Asynchronous On-Policy RL
- **分类: cs.LG; cs.AI; cs.RO; eess.SY**

- **简介: 该论文属于强化学习领域，针对异步策略更新中的策略滞后问题，提出一种基于总变分的优化方法以提升性能。**

- **链接: [https://arxiv.org/pdf/2603.01365](https://arxiv.org/pdf/2603.01365)**

> **作者:** Homayoun Honari; Roger Creus Castanyer; Michael Przystupa; Michael Noukhovitch; Pablo Samuel Castro; Glen Berseth
>
> **摘要:** Distributed training and increasing the gradient update frequency are practical strategies to accelerate learning and improve performance, but both exacerbate a central challenge: \textit{policy lag}, which is the mismatch between the behavior policy generating data and the learning policy being updated. Policy lag can hinder the scaling of on-policy learning algorithms to larger problems. In this paper, we identify the sources of policy lag caused by distributed learning and high update frequency. We use the findings to propose \textit{total Variation-based Advantage aligned Constrained policy Optimization (\methodacronym)} as a practical approach to mitigate policy lag. We empirically validate our method and show that it offers better robustness to policy lag in classic RL tasks and a modern RL for LLM math reasoning task.
>
---
#### [new 089] From Leaderboard to Deployment: Code Quality Challenges in AV Perception Repositories
- **分类: cs.CV; cs.LG; cs.RO; cs.SE**

- **简介: 该论文属于自动驾驶领域，旨在解决感知代码质量与部署需求之间的差距。通过分析178个模型，发现多数代码不满足生产标准，提出改进指南。**

- **链接: [https://arxiv.org/pdf/2603.02194](https://arxiv.org/pdf/2603.02194)**

> **作者:** Mateus Karvat; Bram Adams; Sidney Givigi
>
> **摘要:** Autonomous vehicle (AV) perception models are typically evaluated solely on benchmark performance metrics, with limited attention to code quality, production readiness and long-term maintainability. This creates a significant gap between research excellence and real-world deployment in safety-critical systems subject to international safety standards. To address this gap, we present the first large-scale empirical study of software quality in AV perception repositories, systematically analyzing 178 unique models from the KITTI and NuScenes 3D Object Detection leaderboards. Using static analysis tools (Pylint, Bandit, and Radon), we evaluated code errors, security vulnerabilities, maintainability, and development practices. Our findings revealed that only 7.3% of the studied repositories meet basic production-readiness criteria, defined as having zero critical errors and no high-severity security vulnerabilities. Security issues are highly concentrated, with the top five issues responsible for almost 80% of occurrences, which prompted us to develop a set of actionable guidelines to prevent them. Additionally, the adoption of Continuous Integration/Continuous Deployment pipelines was correlated with better code maintainability. Our findings highlight that leaderboard performance does not reflect production readiness and that targeted interventions could substantially improve the quality and safety of AV perception code.
>
---
#### [new 090] Information-Theoretic Framework for Self-Adapting Model Predictive Controllers
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于控制理论任务，旨在解决MPC适应性不足的问题。通过引入信息理论框架和数字孪生技术，提升MPC的实时自适应能力。**

- **链接: [https://arxiv.org/pdf/2603.01286](https://arxiv.org/pdf/2603.01286)**

> **作者:** Wael Hafez; Amir Nazeri
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Model Predictive Control (MPC) is a vital technique for autonomous systems, like Unmanned Aerial Vehicles (UAVs), enabling optimized motion planning. However, traditional MPC struggles to adapt to real-time changes such as dynamic obstacles and shifting system dynamics, lacking inherent mechanisms for self-monitoring and adaptive optimization. Here, we introduce Entanglement Learning (EL), an information-theoretic framework that enhances MPC adaptability through an Information Digital Twin (IDT). The IDT monitors and quantifies, in bits, the information flow between MPC inputs, control actions, and UAV behavior. By introducing new information-theoretic metrics we call entanglement metrics, it tracks variations in these dependencies. These metrics measure the mutual information between the optimizer's input, its control actions, and the resulting UAV dynamics, enabling a deeper understanding of their interrelationships. This allows the IDT to detect performance deviations and generate real-time adaptive signals to recalibrate MPC parameters, preserving stability. Unlike traditional MPC, which relies on error-based feedback, this dual-feedback approach leverages information flow for proactive adaptation to evolving conditions. Scalable and leveraging existing infrastructure, this framework improves MPC reliability and robustness across diverse scenarios, extending beyond UAV control to any MPC implementation requiring adaptive performance.
>
---
#### [new 091] Event-Only Drone Trajectory Forecasting with RPM-Modulated Kalman Filtering
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于无人机轨迹预测任务，解决传统方法依赖RGB图像或训练数据的问题。通过融合旋翼转速信息，提出一种基于Kalman滤波的事件相机驱动预测方法。**

- **链接: [https://arxiv.org/pdf/2603.01997](https://arxiv.org/pdf/2603.01997)**

> **作者:** Hari Prasanth S.M.; Pejman Habibiroudkenar; Eerik Alamikkotervo; Dimitrios Bouzoulas; Risto Ojala
>
> **备注:** Submitted to ICUAS 2026 conference
>
> **摘要:** Event cameras provide high-temporal-resolution visual sensing that is well suited for observing fast-moving aerial objects; however, their use for drone trajectory prediction remains limited. This work introduces an event-only drone forecasting method that exploits propeller-induced motion cues. Propeller rotational speed are extracted directly from raw event data and fused within an RPM-aware Kalman filtering framework. Evaluations on the FRED dataset show that the proposed method outperforms learning-based approaches and vanilla kalman filter in terms of average distance error and final distance error at 0.4s and 0.8s forecasting horizons. The results demonstrate robust and accurate short- and medium-horizon trajectory forecasting without reliance on RGB imagery or training data.
>
---
#### [new 092] Streaming Real-Time Trajectory Prediction Using Endpoint-Aware Modeling
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于轨迹预测任务，解决实时连续预测问题。提出一种轻量级方法，利用端点信息提升预测准确性与效率。**

- **链接: [https://arxiv.org/pdf/2603.01864](https://arxiv.org/pdf/2603.01864)**

> **作者:** Alexander Prutsch; David Schinagl; Horst Possegger
>
> **备注:** WACV 2026 Oral. Project Page at this https URL
>
> **摘要:** Future trajectories of neighboring traffic agents have a significant influence on the path planning and decision-making of autonomous vehicles. While trajectory forecasting is a well-studied field, research mainly focuses on snapshot-based prediction, where each scenario is treated independently of its global temporal context. However, real-world autonomous driving systems need to operate in a continuous setting, requiring real-time processing of data streams with low latency and consistent predictions over successive timesteps. We leverage this continuous setting to propose a lightweight yet highly accurate streaming-based trajectory forecasting approach. We integrate valuable information from previous predictions with a novel endpoint-aware modeling scheme. Our temporal context propagation uses the trajectory endpoints of the previous forecasts as anchors to extract targeted scenario context encodings. Our approach efficiently guides its scene encoder to extract highly relevant context information without needing refinement iterations or segment-wise decoding. Our experiments highlight that our approach effectively relays information across consecutive timesteps. Unlike methods using multi-stage refinement processing, our approach significantly reduces inference latency, making it well-suited for real-world deployment. We achieve state-of-the-art streaming trajectory prediction results on the Argoverse~2 multi-agent and single-agent benchmarks, while requiring substantially fewer resources.
>
---
#### [new 093] Monocular 3D Object Position Estimation with VLMs for Human-Robot Interaction
- **分类: cs.CV; cs.AI; cs.HC; cs.LG; cs.RO**

- **简介: 该论文属于人机交互中的3D物体位置估计任务，旨在通过单目图像和自然语言输入准确预测物体3D坐标，提升机器人交互能力。**

- **链接: [https://arxiv.org/pdf/2603.01224](https://arxiv.org/pdf/2603.01224)**

> **作者:** Ari Wahl; Dorian Gawlinski; David Przewozny; Paul Chojecki; Felix Bießmann; Sebastian Bosse
>
> **备注:** Accepted at Workshop on Integrating Image Processing with Large-Scale Language/Vision Models for Advanced Visual Understanding (LVLM) at IEEE International Conference on Image Processing (ICIP) 2025
>
> **摘要:** Pre-trained general-purpose Vision-Language Models (VLM) hold the potential to enhance intuitive human-machine interactions due to their rich world knowledge and 2D object detection capabilities. However, VLMs for 3D coordinates detection tasks are rare. In this work, we investigate interactive abilities of VLMs by returning 3D object positions given a monocular RGB image from a wrist-mounted camera, natural language input, and robot states. We collected and curated a heterogeneous dataset of more than 100,000 images and finetuned a VLM using QLoRA with a custom regression head. By implementing conditional routing, our model maintains its ability to process general visual queries while adding specialized 3D position estimation capabilities. Our results demonstrate robust predictive performance with a median MAE of 13 mm on the test set and a five-fold improvement over a simpler baseline without finetuning. In about 25% of the cases, predictions are within a range considered acceptable for the robot to interact with objects.
>
---
#### [new 094] Pri4R: Learning World Dynamics for Vision-Language-Action Models with Privileged 4D Representation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出Pri4R，解决VLA模型缺乏物理动态理解的问题，通过引入4D信息提升动作-世界交互能力。**

- **链接: [https://arxiv.org/pdf/2603.01549](https://arxiv.org/pdf/2603.01549)**

> **作者:** Jisoo Kim; Jungbin Cho; Sanghyeok Chu; Ananya Bal; Jinhyung Kim; Gunhee Lee; Sihaeng Lee; Seung Hwan Kim; Bohyung Han; Hyunmin Lee; Laszlo A. Jeni; Seungryong Kim
>
> **摘要:** Humans learn not only how their bodies move, but also how the surrounding world responds to their actions. In contrast, while recent Vision-Language-Action (VLA) models exhibit impressive semantic understanding, they often fail to capture the spatiotemporal dynamics governing physical interaction. In this paper, we introduce Pri4R, a simple yet effective approach that endows VLA models with an implicit understanding of world dynamics by leveraging privileged 4D information during training. Specifically, Pri4R augments VLAs with a lightweight point track head that predicts 3D point tracks. By injecting VLA features into this head to jointly predict future 3D trajectories, the model learns to incorporate evolving scene geometry within its shared representation space, enabling more physically aware context for precise control. Due to its architectural simplicity, Pri4R is compatible with dominant VLA design patterns with minimal changes. During inference, we run the model using the original VLA architecture unchanged; Pri4R adds no extra inputs, outputs, or computational overhead. Across simulation and real-world evaluations, Pri4R significantly improves performance on challenging manipulation tasks, including a +10% gain on LIBERO-Long and a +40% gain on RoboCasa. We further show that 3D point track prediction is an effective supervision target for learning action-world dynamics, and validate our design choices through extensive ablations.
>
---
#### [new 095] Integrating LTL Constraints into PPO for Safe Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.LO; cs.RO**

- **简介: 该论文属于安全强化学习任务，旨在解决如何将LTL安全约束融入PPO算法的问题。通过引入LTL约束和惩罚机制，提升策略的安全性与性能。**

- **链接: [https://arxiv.org/pdf/2603.01292](https://arxiv.org/pdf/2603.01292)**

> **作者:** Maifang Zhang; Hang Yu; Qian Zuo; Cheng Wang; Vaishak Belle; Fengxiang He
>
> **摘要:** This paper proposes Proximal Policy Optimization with Linear Temporal Logic Constraints (PPO-LTL), a framework that integrates safety constraints written in LTL into PPO for safe reinforcement learning. LTL constraints offer rigorous representations of complex safety requirements, such as regulations that broadly exist in robotics, enabling systematic monitoring of safety requirements. Violations against LTL constraints are monitored by limit-deterministic Büchi automata, and then translated by a logic-to-cost mechanism into penalty signals. The signals are further employed for guiding the policy optimization via the Lagrangian scheme. Extensive experiments on the Zones and CARLA environments show that our PPO-LTL can consistently reduce safety violations, while maintaining competitive performance, against the state-of-the-art methods. The code is at this https URL.
>
---
#### [new 096] Design Framework and Manufacturing of an Active Magnetic Bearing Spindle for Micro-Milling Applications
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于机械设计任务，旨在解决微铣削主轴在高速下的摩擦与热膨胀问题。通过构建系统化设计框架，实现高精度主动磁轴承主轴的制造。**

- **链接: [https://arxiv.org/pdf/2603.00169](https://arxiv.org/pdf/2603.00169)**

> **作者:** Kazi Sher Ahmed; Bekir Bediz
>
> **摘要:** Micro-milling spindles require high rotational speeds where conventional rolling element bearings face limitations such as friction and thermal expansion. Active magnetic bearings (AMBs) address these challenges by providing non-contact and lubrication-free operation at ultra-high speeds with the ability to actively regulate spindle dynamics. The existing literature on AMB spindles has mainly reported specific prototype realizations or control system implementations for specific spindle dynamics. Consequently, design knowledge remains fragmented across isolated successful studies. This paper addresses this gap by presenting a systematic and iterative framework to design and manufacture a micro-milling AMB spindle. The process involves a multidisciplinary design flow with a focus on critical practical aspects of manufacturing. The realized spindle is reported as a case study.
>
---
#### [new 097] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决大规模环境中探索效率低的问题。通过分析策略多样性影响，提出耦合策略优化方法，提升样本效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.01741](https://arxiv.org/pdf/2603.01741)**

> **作者:** Naoki Shitanda; Motoki Omura; Tatsuya Harada; Takayuki Osa
>
> **备注:** In ICLR 2026. Website at this https URL
>
> **摘要:** Scaling reinforcement learning to tens of thousands of parallel environments requires overcoming the limited exploration capacity of a single policy. Ensemble-based policy gradient methods, which employ multiple policies to collect diverse samples, have recently been proposed to promote exploration. However, merely broadening the exploration space does not always enhance learning capability, since excessive exploration can reduce exploration quality or compromise training stability. In this work, we theoretically analyze the impact of inter-policy diversity on learning efficiency in policy ensembles, and propose Coupled Policy Optimization which regulates diversity through KL constraints between policies. The proposed method enables effective exploration and outperforms strong baselines such as SAPG, PBT, and PPO across multiple tasks, including challenging dexterous manipulation, in terms of both sample efficiency and final performance. Furthermore, analysis of policy diversity and effective sample size during training reveals that follower policies naturally distribute around the leader, demonstrating the emergence of structured and efficient exploratory behavior. Our results indicate that diverse exploration under appropriate regulation is key to achieving stable and sample-efficient learning in ensemble policy gradient methods. Project page at this https URL .
>
---
#### [new 098] RoboGPU: Accelerating GPU Collision Detection for Robotics
- **分类: cs.AR; cs.RO**

- **简介: 论文提出RoboGPU，用于加速机器人碰撞检测。解决传统GPU在处理机器人运动规划时效率低的问题，通过引入RoboCore提升碰撞查询速度，同时支持多种机器人任务。**

- **链接: [https://arxiv.org/pdf/2603.01517](https://arxiv.org/pdf/2603.01517)**

> **作者:** Lufei Liu; Liwei Xue; Youssef Mohammed; Jocelyn Zhao; Yuan Hsi Chou; Tor M. Aamodt
>
> **摘要:** Autonomous robots are increasingly prevalent in our society, emerging in medical care, transportation vehicles, and home assistance. These robots rely on motion planning and collision detection to identify a sequence of movements allowing them to navigate to an end goal without colliding with the surrounding environment. While many specialized accelerators have been proposed to meet the real-time requirements of robotics planning tasks, they often lack the flexibility to adapt to the rapidly changing landscape of robotics and support future advancements. However, GPUs are well-positioned for robotics and we find that they can also tackle collision detection algorithms with enhancements to existing ray tracing accelerator (RTA) units. Unlike intersection tests in ray tracing, collision queries in robotics require control flow mechanisms to avoid unnecessary computations in each query. In this work, we explore and compare different architectural modifications to address the gaps of existing GPU RTAs. Our proposed RoboGPU architecture introduces a RoboCore that computes collision queries 3.1$\times$ faster than RTA implementations and 14.8$\times$ faster than a CUDA baseline. RoboCore is also useful for other robotics tasks, achieving 3.6$\times$ speedup on a state-of-the-art neural motion planner and 1.1$\times$ speedup on Monte Carlo Localization compared to a baseline GPU. RoboGPU matches the performance of dedicated hardware accelerators while being able to adapt to evolving motion planning algorithms and support classical algorithms.
>
---
#### [new 099] Smart Prism with Tilt Compensation for CAN bus on Mobile Machinery Using Robotic Total Stations
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于定位与导航任务，解决移动机械在复杂地形中因倾斜导致的定位误差问题。通过集成IMU和总站，实现实时倾斜补偿，提升参考轨迹精度。**

- **链接: [https://arxiv.org/pdf/2603.00320](https://arxiv.org/pdf/2603.00320)**

> **作者:** Sumesh Sharma; Marcel Moll; Timo Oksanen
>
> **摘要:** Accurate reference trajectories are required to validate autonomous agricultural robots and highly automated off-road vehicles under real-world field conditions. In practice, robotic total stations provide millimeter-level prism center coordinates, but the point of interest on the vehicle is typically displaced by a lever arm, ranging from decimeters to multiple meters. Roll and pitch motions, as typically observed in off-road machinery, therefore introduce horizontal point of interest errors far exceeding the measurement accuracy of robotic total stations observations. This paper presents the design, implementation, and validation of a Smart Prism prototype that augments a robotic total station prism with an inertial measurement unit to enable real-time tilt compensation. The prototype integrates an STM32H7 microcontroller and a Murata SCH16T-series IMU and estimates roll and pitch angles using an adaptive complementary filter. The tilt-compensated point of interest coordinates are obtained by transforming a calibrated lever arm from the body frame into the navigation frame and combining it with robotic total station prism positions. To support vehicle-side integration, the system can transmit prism and tilt-compensated point of interest coordinates on the Controller Area Network bus, allowing the point of interest to be treated as a virtual position sensor (e.g., co-located with a rear-axle reference point). Experiments with a fixed ground reference point, using a prism to point of interest lever arm of approximately 1.07m and manual roll/pitch excursions of up to 60 deg, yield three-dimensional root-mean-square errors between 2.9mm and 23.6mm across five test series. The results demonstrate that IMU-based tilt compensation enables reference measurements suitable for validating centimeter-level navigation systems under dynamic field conditions.
>
---
#### [new 100] Empirical Study of Gaze Behavior in Children and Young Adults Using Deep Neural Networks and Robot Implementation: A Comparative Analysis of Social Situations
- **分类: cs.CY; cs.HC; cs.RO**

- **简介: 该论文属于行为分析任务，旨在研究儿童与成人在社交情境中的注视行为差异，并通过深度学习模型和机器人实现评估其社会接受度。**

- **链接: [https://arxiv.org/pdf/2603.00074](https://arxiv.org/pdf/2603.00074)**

> **作者:** Ramtin Tabatabaei; Milad Hosseini; Ali Mohajerzarrinkelk; Ali F. Meghdari; Alireza Taheri
>
> **摘要:** In a preliminary exploratory study, our goal was to train deep neural network models to mimic children's and/or adults' gaze behavior in certain social situations to reach this objective. Additionally, we aim to identify potential differences in gaze behavior between these two age groups based on our participants' gaze data. Furthermore, we aimed to assess the practical effectiveness of our adult and children models by deploying them on a Nao robot in real-life settings. To achieve this, we first created two video clips, one animation and one live-action, to depict some social situations. Using an eye-tracking device, we collected eye-tracking data from 24 participants, including 12 children and 12 adults. Then, we utilized deep neural networks, specifically LSTM and Transformer Networks, to analyze and model the gaze patterns of each group of participants. Our results indicate that when the models attempted to predict people's locations (in the next frame), they had an accuracy in the range of 62%-70% with one attempt, which increased by ~20% when attempted twice (i.e. the two highest-ranked predicted labels as outputs). As expected, the result underscores that gaze behavior is not a wholly unique phenomenon. We obtained feedback from 57 new participants to evaluate the robot's functionality. These participants were asked to watch two videos of the robot's performance in each mode and then complete a comprehensive questionnaire. The questionnaire results indicate that the participants expressed satisfaction with the robot's interaction, including its attention, intelligence, and responsiveness to human actions. However, they did not perceive the robot as a social companion comparable to a human. This exploratory study tries to address/show potentials of the social acceptance of robots based on human nonverbal behavioral cues for future research.
>
---
#### [new 101] DriveCode: Domain Specific Numerical Encoding for LLM-Based Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决LLM中数值编码精度不足的问题。通过提出DriveCode方法，将数字映射为嵌入向量，提升轨迹预测与控制信号生成效果。**

- **链接: [https://arxiv.org/pdf/2603.00919](https://arxiv.org/pdf/2603.00919)**

> **作者:** Zhiye Wang; Yanbo Jiang; Rui Zhou; Bo Zhang; Fang Zhang; Zhenhua Xu; Yaqin Zhang; Jianqiang Wang
>
> **备注:** The project page is available at this https URL
>
> **摘要:** Large language models (LLMs) have shown great promise for autonomous driving. However, discretizing numbers into tokens limits precise numerical reasoning, fails to reflect the positional significance of digits in the training objective, and makes it difficult to achieve both decoding efficiency and numerical precision. These limitations affect both the processing of sensor measurements and the generation of precise control commands, creating a fundamental barrier for deploying LLM-based autonomous driving systems. In this paper, we introduce DriveCode, a novel numerical encoding method that represents numbers as dedicated embeddings rather than discrete text tokens. DriveCode employs a number projector to map numbers into the language model's hidden space, enabling seamless integration with visual and textual features in a unified multimodal sequence. Evaluated on OmniDrive, DriveGPT4, and DriveGPT4-V2 datasets, DriveCode demonstrates superior performance in trajectory prediction and control signal generation, confirming its effectiveness for LLM-based autonomous driving systems.
>
---
#### [new 102] SMR-Net:Robot Snap Detection Based on Multi-Scale Features and Self-Attention Network
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人抓取任务，解决复杂场景下快接件检测与定位问题。提出SMR-Net算法，融合多尺度特征与自注意力机制，提升检测精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.01036](https://arxiv.org/pdf/2603.01036)**

> **作者:** Kuanxu Hou
>
> **备注:** snap assembly, snap detection and localization, object detection, multi-scale feature fusion, self-attention
>
> **摘要:** In robot automated assembly, snap assembly precision and efficiency directly determine overall production quality. As a core prerequisite, snap detection and localization critically affect subsequent assembly success. Traditional visual methods suffer from poor robustness and large localization errors when handling complex scenarios (e.g., transparent or low-contrast snaps), failing to meet high-precision assembly demands. To address this, this paper designs a dedicated sensor and proposes SMR-Net, an self-attention-based multi-scale object detection algorithm, to synergistically enhance detection and localization performance. SMR-Net adopts an attention-enhanced multi-scale feature fusion architecture: raw sensor data is encoded via an attention-embedded feature extractor to strengthen key snap features and suppress noise; three multi-scale feature maps are processed in parallel with standard and dilated convolution for dimension unification while preserving resolution; an adaptive reweighting network dynamically assigns weights to fused features, generating fine representations integrating details and global semantics. Experimental results on Type A and Type B snap datasets show SMR-Net outperforms traditional Faster R-CNN significantly: Intersection over Union (IoU) improves by 6.52% and 5.8%, and mean Average Precision (mAP) increases by 2.8% and 1.5% respectively. This fully demonstrates the method's superiority in complex snap detection and localization tasks.
>
---
#### [new 103] Bimanual XR Specification of Relative and Absolute Assembly Hierarchies for Teleoperation
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于远程操作任务，解决如何高效指定装配约束的问题。通过双手交互定义相对与绝对的装配层次结构，提升机器人操作灵活性。**

- **链接: [https://arxiv.org/pdf/2603.01495](https://arxiv.org/pdf/2603.01495)**

> **作者:** Benjamin Yang; Xichen He; Charlie Zou; Jen-Shuo Liu; Barbara Tversky; Steven Feiner
>
> **摘要:** We present a bimanual XR interaction approach for specifying remote assembly tasks as hierarchies of relative and absolute object constraints that specify high-level teleoperation goals for robots. Grabbing one object in each hand creates a constraint group (visualized as a hull) and groups can be nested into hierarchies. Each group can be relative (with a robot-specifiable 6DoF pose) or absolute (with an author-specified fixed 6DoF pose) in relation to its parent. A relative group specifies a subassembly that can be constructed at a location chosen by the robot software for efficiency rather than mandated by the user.
>
---
## 更新

#### [replaced 001] Properties of Lyapunov Subcenter Manifolds in Conservative Mechanical Systems
- **分类: math.DS; cs.RO**

- **简介: 该论文研究保守机械系统中Lyapunov子中心流形的性质，解决非线性模态的控制应用问题，通过理论分析与数值验证揭示其动态特性。**

- **链接: [https://arxiv.org/pdf/2505.13064](https://arxiv.org/pdf/2505.13064)**

> **作者:** Yannik P. Wotte; Arne Sachtler; Alin Albu-Schäffer; Stefano Stramigioli; Cosimo Della Santina
>
> **备注:** 20 pages, 27 figures, submitted to Automatica
>
> **摘要:** Multi-body mechanical systems have rich internal dynamics, whose solutions can be exploited as efficient control targets. Yet, solutions non-trivially depend on system parameters, obscuring feasible properties for use as target trajectories. For periodic regulation tasks in robotics applications, we investigate properties of nonlinear normal modes (NNMs) collected in Lyapunov subcenter manifolds (LSMs) of conservative mechanical systems. Using a time-symmetry of conservative mechanical systems, we show that mild non-resonance conditions guarantee LSMs to be Eigenmanifolds, in which NNMs are guaranteed to oscillate between two points of zero velocity. We also prove the existence of a unique generator, which is a connected, 1D manifold that collects these points of zero velocity for a given Eigenmanifold. Furthermore, we show that an additional spatial symmetry provides LSMs with yet stronger properties of Rosenberg manifolds. Here all brake trajectories pass through a unique equilibrium configuration, which can be favorable for control applications. These theoretical results are numerically confirmed on two mechanical systems: a double pendulum and a 5-link pendulum.
>
---
#### [replaced 002] BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在边缘设备部署效率低的问题。通过设计1-bit模型BitVLA，实现高效且性能稳定的机器人控制。**

- **链接: [https://arxiv.org/pdf/2506.07530](https://arxiv.org/pdf/2506.07530)**

> **作者:** Hongyu Wang; Chuyan Xiong; Ruiping Wang; Xilin Chen
>
> **备注:** Work in progress
>
> **摘要:** Deploying powerful Vision-Language-Action (VLA) models on edge devices is limited by their massive size. In this paper, we take a deployment-oriented view of VLA training: we target efficiency through model design and optimization, rather than relying solely on post-hoc compression. Thus, we propose BitVLA, a fully native 1-bit VLA model for robotic manipulation, where every parameters is ternary, i.e., {-1,0,1}. BitVLA is built on the publicly available 1-bit LLM BitNet b1.58 2B4T, and is trained as a vision-language-action policy that inherits the compactness of 1-bit pretraining while retaining strong task performance. To further reduce the memory footprint of the vision backbone, we introduce Quantize-then-Distill, a post-training quantization-aware training strategy that compresses a full-precision vision encoder to 1.58-bit weights, while a full-precision teacher guides representation alignment during training. Across simulation benchmarks and real-world tasks, BitVLA matches the performance of the full-precision OpenVLA-OFT baseline, while reducing model memory by 11.0x and end-to-end latency by 4.4x. These results suggest a practical path toward training-time efficiency-accuracy co-design for embodied policies, enabling competitive manipulation capability on memory-constrained edge robotic platforms. We release the code in this https URL, model weights in this https URL.
>
---
#### [replaced 003] UNCLE-Grasp: Uncertainty-Aware Grasping of Leaf-Occluded Strawberries
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决部分遮挡下草莓抓取的几何不确定性问题。通过建模不确定性并评估抓取可行性，提升抓取可靠性。**

- **链接: [https://arxiv.org/pdf/2601.14492](https://arxiv.org/pdf/2601.14492)**

> **作者:** Malak Mansour; Ali Abouzeid; Zezhou Sun; Qinbo Sun; Dezhen Song; Abdalla Swikir
>
> **摘要:** Robotic strawberry harvesting remains challenging under partial occlusion, where leaf interference introduces significant geometric uncertainty and renders grasp decisions based on a single deterministic shape estimate unreliable. From a single partial observation, multiple incompatible 3D shape completions may be plausible, such that grasps deemed feasible on one completion can fail on another. This paper presents an uncertainty-aware grasping pipeline for partially occluded strawberries that explicitly models geometric uncertainty arising from both occlusion and learned shape completion. The proposed approach employs point cloud completion with Monte Carlo dropout to sample multiple shape hypotheses, generates candidate grasps for each completion, and evaluates grasp feasibility using physically grounded force-closure metrics. Rather than selecting a grasp from a single shape estimate, feasibility is aggregated across completions and a conservative lower confidence bound (LCB) criterion is used to decide whether grasping a strawberry should be attempted or safely abstained. The method is evaluated in simulation and on a physical robot under increasing levels of synthetic and real leaf occlusion. Experimental results demonstrate that uncertainty-aware decision making enables reliable abstention from high-risk grasp attempts under severe occlusion while maintaining robust grasp execution when geometric confidence is sufficient, outperforming deterministic baselines in both simulated and physical robot experiments.
>
---
#### [replaced 004] Safe and Optimal Variable Impedance Control via Certified Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决RL在动态阻抗控制中的不稳定和不安全问题。提出C-GMS框架，确保策略稳定性和物理可行性。**

- **链接: [https://arxiv.org/pdf/2511.16330](https://arxiv.org/pdf/2511.16330)**

> **作者:** Shreyas Kumar; Ravi Prakash
>
> **备注:** Accepted at ICRA 2026
>
> **摘要:** Reinforcement learning (RL) offers a powerful approach for robots to learn complex, collaborative skills by combining Dynamic Movement Primitives (DMPs) for motion and Variable Impedance Control (VIC) for compliant interaction. However, this model-free paradigm often risks instability and unsafe exploration due to the time-varying nature of impedance gains. This work introduces Certified Gaussian Manifold Sampling (C-GMS), a novel trajectory-centric RL framework that learns combined DMP and VIC policies while guaranteeing Lyapunov stability and actuator feasibility by construction. Our approach reframes policy exploration as sampling from a mathematically defined manifold of stable gain schedules. This ensures every policy rollout is guaranteed to be stable and physically realizable, thereby eliminating the need for reward penalties or post-hoc validation. Furthermore, we provide a theoretical guarantee that our approach ensures bounded tracking error even in the presence of bounded model errors and deployment-time uncertainties. We demonstrate the effectiveness of C-GMS in simulation and verify its efficacy on a real robot, paving the way for reliable autonomous interaction in complex environments.
>
---
#### [replaced 005] MorphArtGrasp: Morphology-Aware Cross-Embodiment Dexterous Hand Articulation Generation for Grasping
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多指灵巧抓取任务，解决跨形态手部关节生成问题。提出MorphArtGrasp框架，通过形态嵌入和特征抓取集，生成高效抓取动作。**

- **链接: [https://arxiv.org/pdf/2510.06068](https://arxiv.org/pdf/2510.06068)**

> **作者:** Heng Zhang; Kevin Yuchen Ma; Mike Zheng Shou; Weisi Lin; Yan Wu
>
> **摘要:** Dexterous grasping with multi-fingered hands remains challenging due to high-dimensional articulations and the cost of optimization-based pipelines. Existing end-to-end methods require training on large-scale datasets for specific hands, limiting their ability to generalize across different embodiments. We propose MorphArtGrasp, an eigengrasp-based, end-to-end framework for cross-embodiment grasp generation. From a hand's morphology description, we derive a morphology embedding and an eigengrasp set. Conditioned on these, together with the object point cloud and wrist pose, an amplitude predictor regresses articulation coefficients in a low-dimensional space, which are decoded into full joint articulations. Articulation learning is supervised with a Kinematic-Aware Articulation Loss (KAL) that emphasizes fingertip-relevant motions and injects morphology-specific structure. In simulation on unseen objects across three dexterous hands, MorphArtGrasp attains a 91.9% average grasp success rate with less than 0.4 seconds inference per grasp. With few-shot adaptation to an unseen hand, it achieves 85.6% success on unseen objects in simulation, and real-world experiments on this few-shot-generalized hand achieve an 87% success rate. The code and additional materials are available on our project website this https URL.
>
---
#### [replaced 006] Bridging Perception and Planning: Towards End-to-End Planning for Signal Temporal Logic Tasks
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究机器人在信号时序逻辑（STL）任务中的路径规划问题，提出S-MSP框架，直接从多视角视觉和STL规范生成可行轨迹，提升复杂环境下的任务执行能力。**

- **链接: [https://arxiv.org/pdf/2509.12813](https://arxiv.org/pdf/2509.12813)**

> **作者:** Bowen Ye; Junyue Huang; Yang Liu; Xiaozhen Qiao; Xiang Yin
>
> **摘要:** We investigate the task and motion planning problem for Signal Temporal Logic (STL) specifications in robotics. Existing STL methods rely on pre-defined maps or mobility representations, which are ineffective in unstructured real-world environments. We propose the \emph{Structured-MoE STL Planner} (\textbf{S-MSP}), a differentiable framework that maps synchronized multi-view camera observations and an STL specification directly to a feasible trajectory. S-MSP integrates STL constraints within a unified pipeline, trained with a composite loss that combines trajectory reconstruction and STL robustness. A \emph{structure-aware} Mixture-of-Experts (MoE) model enables horizon-aware specialization by projecting sub-tasks into temporally anchored embeddings. We evaluate S-MSP using a high-fidelity simulation of factory-logistics scenarios with temporally constrained tasks. Experiments show that S-MSP outperforms single-expert baselines in STL satisfaction and trajectory feasibility. A rule-based \emph{safety filter} at inference improves physical executability without compromising logical correctness, showcasing the practicality of the approach.
>
---
#### [replaced 007] Automated Action Generation based on Action Field for Robotic Garment Smoothing and Alignment
- **分类: cs.RO**

- **简介: 该论文属于机器人服装平整与对齐任务，解决布料变形和形状多样带来的操作难题。提出一种基于动作场的自动化动作生成方法，提升精度并减少计算时间。**

- **链接: [https://arxiv.org/pdf/2505.03537](https://arxiv.org/pdf/2505.03537)**

> **作者:** Hu Cheng; Fuyuki Tokuda; Kazuhiro Kosuge
>
> **备注:** Accepted by IEEE Transactions on Automation Science and Engineering
>
> **摘要:** Garment manipulation using robotic systems is a challenging task due to the diverse shapes and deformable nature of fabric. In this paper, we propose a novel method for robotic garment smoothing and alignment that significantly improves the accuracy while reducing computational time compared to previous approaches. Our method features an action generator that directly interprets scene images and generates pixel-wise end-effector action vectors using a neural network. The network also predicts a manipulation score map that ranks potential actions, allowing the system to select the most effective action. Extensive simulation experiments demonstrate that our method achieves higher smoothing and alignment performances and faster computation time than previous approaches. Real-world experiments show that the proposed method generalizes well to different garment types and successfully flattens garments.
>
---
#### [replaced 008] DDP-WM: Disentangled Dynamics Prediction for Efficient World Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DDP-WM，解决世界模型计算效率低的问题，通过解耦动态预测提升性能，适用于导航和操作等任务。**

- **链接: [https://arxiv.org/pdf/2602.01780](https://arxiv.org/pdf/2602.01780)**

> **作者:** Shicheng Yin; Kaixuan Yin; Weixing Chen; Yang Liu; Guanbin Li; Liang Lin
>
> **备注:** Efficient and high-fidelity world model. Code is available at this https URL
>
> **摘要:** World models are essential for autonomous robotic planning. However, the substantial computational overhead of existing dense Transformerbased models significantly hinders real-time deployment. To address this efficiency-performance bottleneck, we introduce DDP-WM, a novel world model centered on the principle of Disentangled Dynamics Prediction (DDP). We hypothesize that latent state evolution in observed scenes is heterogeneous and can be decomposed into sparse primary dynamics driven by physical interactions and secondary context-driven background updates. DDP-WM realizes this decomposition through an architecture that integrates efficient historical processing with dynamic localization to isolate primary dynamics. By employing a crossattention mechanism for background updates, the framework optimizes resource allocation and provides a smooth optimization landscape for planners. Extensive experiments demonstrate that DDP-WM achieves significant efficiency and performance across diverse tasks, including navigation, precise tabletop manipulation, and complex deformable or multi-body interactions. Specifically, on the challenging Push-T task, DDP-WM achieves an approximately 9 times inference speedup and improves the MPC success rate from 90% to98% compared to state-of-the-art dense models. The results establish a promising path for developing efficient, high-fidelity world models. Codes will be available at this https URL.
>
---
#### [replaced 009] SWITCH: Benchmarking Modeling and Handling of Tangible Interfaces in Long-horizon Embodied Scenarios
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出SWITCH基准，用于评估智能体在长期具身场景中处理有形接口的能力，解决部分可观测、因果推理和验证等挑战。**

- **链接: [https://arxiv.org/pdf/2511.17649](https://arxiv.org/pdf/2511.17649)**

> **作者:** Jieru Lin; Zhiwei Yu; Börje F. Karlsson
>
> **摘要:** Autonomous agents operating in the real world must interact continuously with existing physical and semantic infrastructure, track delayed consequences, and verify outcomes over time. Everyday environments are rich in tangible control interfaces (TCIs)-e.g., light switches, appliance panels, and embedded GUI-posing core challenges for lifelong embodied agents, including partial observability, causal reasoning across time, and failure-aware verification under real-world constraints. Yet, current benchmarks rarely consider such long-horizon interaction and causality requirements. We introduce SWITCH (Semantic World Interface Tasks for Control & Handling), an embodied, task-driven benchmark created through iterative releases to probe these gaps. Its first iteration, SWITCH-Basic, evaluates five complementary abilities-task-aware VQA, semantic UI grounding, action generation, state transition prediction, and result verification-under ego-centric RGB video input and device diversity across 351 tasks spanning 98 real devices/appliances. Results from commercial and open LMMMs reveal systematic failures, highlighting critical gaps for lifelong agent deployment. SWITCH provides data, code, and held-out splits to enable reproducible non-contaminated evaluation and community contributions toward more challenging future iterations of the benchmark and the creation of relevant training data. Benchmark resources are available at: this https URL.
>
---
#### [replaced 010] VLA-Reasoner: Empowering Vision-Language-Action Models with Reasoning via Online Monte Carlo Tree Search
- **分类: cs.RO**

- **简介: 该论文提出VLA-Reasoner，解决视觉-语言-动作模型在长任务中因累积误差导致的性能下降问题，通过在线蒙特卡洛树搜索增强推理能力。**

- **链接: [https://arxiv.org/pdf/2509.22643](https://arxiv.org/pdf/2509.22643)**

> **作者:** Wenkai Guo; Guanxing Lu; Haoyuan Deng; Zhenyu Wu; Yansong Tang; Ziwei Wang
>
> **备注:** 8 pages, 6 figures, Accepted by ICRA 2026
>
> **摘要:** Vision-Language-Action models (VLAs) achieve strong performance in general robotic manipulation tasks by scaling imitation learning. However, existing VLAs are limited to predicting short-sighted next-action, which struggle with long-horizon trajectory tasks due to incremental deviations. To address this problem, we propose a plug-in framework named \method that effectively empowers off-the-shelf VLAs with the capability of foreseeing future states via test-time scaling. Specifically, \method samples and rolls out possible action trajectories where involved actions are rationales to generate future states via a world model, which enables \method to foresee and reason potential outcomes and search for the optimal actions. We further leverage Monte Carlo Tree Search (MCTS) to improve search efficiency in large action spaces, where step-wise VLA predictions seed the root. Meanwhile, we introduce a confidence sampling mechanism based on Kernel Density Estimation (KDE), to enable efficient exploration in MCTS without redundant VLA queries. We evaluate intermediate states in MCTS via an offline value estimation strategy, to score predicted futures and correct deviations with long-term feedback. We conducted extensive experiments in both simulators and the real world, demonstrating that our proposed VLA-Reasoner achieves significant improvements over the state-of-the-art VLAs. Our method highlights a potential pathway toward scalable test-time computation of robotic manipulation. The project website is available at: this https URL.
>
---
#### [replaced 011] Model Predictive Adversarial Imitation Learning for Planning from Observation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于模仿学习任务，旨在解决从观察中规划的问题。通过结合逆强化学习与模型预测控制，提出一种端到端的规划方法，提升样本效率和泛化能力。**

- **链接: [https://arxiv.org/pdf/2507.21533](https://arxiv.org/pdf/2507.21533)**

> **作者:** Tyler Han; Yanda Bao; Bhaumik Mehta; Gabriel Guo; Anubhav Vishwakarma; Emily Kang; Sanghun Jung; Rosario Scalise; Jason Zhou; Bryan Xu; Byron Boots
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Human demonstration data is often ambiguous and incomplete, motivating imitation learning approaches that also exhibit reliable planning behavior. A common paradigm to perform planning-from-demonstration involves learning a reward function via Inverse Reinforcement Learning (IRL) then deploying this reward via Model Predictive Control (MPC). Towards unifying these methods, we derive a replacement of the policy in IRL with a planning-based agent. With connections to Adversarial Imitation Learning, this formulation enables end-to-end interactive learning of planners from observation-only demonstrations. In addition to benefits in interpretability, complexity, and safety, we study and observe significant improvements on sample efficiency, out-of-distribution generalization, and robustness. The study includes evaluations in both simulated control benchmarks and real-world navigation experiments using few-to-single observation-only demonstrations.
>
---
#### [replaced 012] BiNoMaP: Learning Category-Level Bimanual Non-Prehensile Manipulation Primitives
- **分类: cs.RO**

- **简介: 该论文属于机器人非抓取操作任务，解决复杂非抓取动作的泛化问题。提出BiNoMaP框架，通过视频学习双臂操作基元，实现跨平台、跨物体的通用操作能力。**

- **链接: [https://arxiv.org/pdf/2509.21256](https://arxiv.org/pdf/2509.21256)**

> **作者:** Huayi Zhou; Kui Jia
>
> **备注:** Under review. The project link is this https URL
>
> **摘要:** Non-prehensile manipulation, encompassing ungraspable actions such as pushing, poking, pivoting, and wrapping, remains underexplored due to its contact-rich and analytically intractable nature. We revisit this problem from two perspectives. First, instead of relying on single-arm setups or favorable environmental supports (e.g., walls or edges), we advocate a generalizable dual-arm configuration and establish a suite of Bimanual Non-prehensile Manipulation Primitives (BiNoMaP). Second, departing from prevailing RL-based approaches, we propose a three-stage, RL-free framework for learning structured non-prehensile skills. We begin by extracting bimanual hand motion trajectories from video demonstrations. Since these coarse trajectories suffer from perceptual noise and morphological discrepancies, we introduce a geometry-aware post-optimization algorithm to refine them into executable manipulation primitives consistent with predefined motion patterns. To enable category-level generalization, the learned primitives are further parameterized by object-relevant geometric attributes, primarily size, allowing adaptation to unseen instances with significant shape variations. Importantly, BiNoMaP supports cross-embodiment transfer: the same primitives can be deployed on two real-world dual-arm platforms with distinct kinematic configurations, without redesigning skill structures. Extensive real-robot experiments across diverse objects and spatial configurations demonstrate the effectiveness, efficiency, and strong generalization capability of our approach.
>
---
#### [replaced 013] Beyond Frame-wise Tracking: A Trajectory-based Paradigm for Efficient Point Cloud Tracking
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于3D单目标跟踪任务，解决传统方法在计算成本与长期时序上下文之间的矛盾。提出TrajTrack框架，通过轨迹学习提升跟踪精度与效率。**

- **链接: [https://arxiv.org/pdf/2509.11453](https://arxiv.org/pdf/2509.11453)**

> **作者:** BaiChen Fan; Yuanxi Cui; Jian Li; Qin Wang; Shibo Zhao; Muqing Cao; Sifan Zhou
>
> **备注:** Acceptted in ICRA 2026
>
> **摘要:** LiDAR-based 3D single object tracking (3D SOT) is a critical task in robotics and autonomous systems. Existing methods typically follow frame-wise motion estimation or a sequence-based paradigm. However, the two-frame methods are efficient but lack long-term temporal context, making them vulnerable in sparse or occluded scenes, while sequence-based methods that process multiple point clouds gain robustness at a significant computational cost. To resolve this dilemma, we propose a novel trajectory-based paradigm and its instantiation, TrajTrack. TrajTrack is a lightweight framework that enhances a base two-frame tracker by implicitly learning motion continuity from historical bounding box trajectories alone-without requiring additional, costly point cloud inputs. It first generates a fast, explicit motion proposal and then uses an implicit motion modeling module to predict the future trajectory, which in turn refines and corrects the initial proposal. Extensive experiments on the large-scale NuScenes benchmark show that TrajTrack achieves new state-of-the-art performance, dramatically improving tracking precision by 3.02% over a strong baseline while running at 55 FPS. Besides, we also demonstrate the strong generalizability of TrajTrack across different base trackers. Code is available at this https URL.
>
---
#### [replaced 014] Optimization of Edge Directions and Weights for Mixed Guidance Graphs in Lifelong Multi-Agent Path Finding
- **分类: cs.MA; cs.AI; cs.RO**

- **简介: 该论文属于多智能体路径规划任务，解决Lifelong MAPF中的引导问题。通过优化边方向和权重，提出MGGO方法实现更严格的路径引导。**

- **链接: [https://arxiv.org/pdf/2602.23468](https://arxiv.org/pdf/2602.23468)**

> **作者:** Yulun Zhang; Varun Bhatt; Matthew C. Fontaine; Stefanos Nikolaidis; Jiaoyang Li
>
> **摘要:** Multi-Agent Path Finding (MAPF) aims to move agents from their start to goal vertices on a graph. Lifelong MAPF (LMAPF) continuously assigns new goals to agents as they complete current ones. To guide agents' movement in LMAPF, prior works have proposed Guidance Graph Optimization (GGO) methods to optimize a guidance graph, which is a bidirected weighted graph whose directed edges represent moving and waiting actions with edge weights being action costs. Higher edge weights represent higher action costs. However, edge weights only provide soft guidance. An edge with a high weight only discourages agents from using it, instead of prohibiting agents from traversing it. In this paper, we explore the need to incorporate edge directions optimization into GGO, providing strict guidance. We generalize GGO to Mixed Guidance Graph Optimization (MGGO), presenting two MGGO methods capable of optimizing both edge weights and directions. The first optimizes edge directions and edge weights in two phases separately. The second applies Quality Diversity algorithms to optimize a neural network capable of generating edge directions and weights. We also incorporate traffic patterns relevant to edge directions into a GGO method, making it capable of generating edge-direction-aware guidance graphs.
>
---
#### [replaced 015] Learning Contact Dynamics through Touching: Action-conditional Graph Neural Networks for Robotic Peg Insertion
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人接触动力学建模任务，解决机器人插销过程中运动与力矩预测问题。通过改进图神经网络，实现自监督学习，提升预测精度。**

- **链接: [https://arxiv.org/pdf/2509.12151](https://arxiv.org/pdf/2509.12151)**

> **作者:** Zongyao Yi; Joachim Hertzberg; Martin Atzmueller
>
> **摘要:** We present a learnable physics-based predictive model that provides accurate motion and force-torque prediction of the robot end effector in contact-rich manipulation. The proposed model extends the state-of-the-art GNN-based simulator (FIGNet) with novel node and edge types, enabling action-conditional predictions for control and state estimation in the context of robotic peg insertion. Our model learns in a self-supervised manner, using only joint encoder and force-torque data while the robot is touching the environment. In simulation, the MPC agent using our model matches the performance of the same controller with the ground truth dynamics model in a challenging peg-in-hole task, while in the real-world experiment, our model achieves a 50$\%$ improvement in motion prediction accuracy and 3$\times$ increase in force-torque prediction precision over the baseline physics simulator. Finally, we apply the model to track the robot end effector with a particle filter during real-world peg insertion, demonstrating a practical application of its predictive accuracy.
>
---
#### [replaced 016] Developing Fundamental Diagrams for Urban Air Mobility Traffic Based on Physical Experiments
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于交通流分析任务，旨在研究城市空中交通的基本图。通过理论分析与物理实验，构建UAM流量模型，解决其在真实环境中的交通特性问题。**

- **链接: [https://arxiv.org/pdf/2512.21425](https://arxiv.org/pdf/2512.21425)**

> **作者:** Hang Zhou; Yuhui Zhai; Shiyu Shen; Yanfeng Ouyang; Xiaowei Shi; Xiaopeng Li
>
> **摘要:** Urban Air Mobility (UAM) is an emerging application of unmanned aerial vehicles that promises to reduce travel time and alleviate congestion in urban transportation systems. As drone density increases, UAM traffic is expected to experience congestion similar to that in ground traffic. However, the fundamental characteristics of UAM traffic, particularly under real-world operating conditions, remain largely unexplored. This study proposes a general framework for constructing the fundamental diagram (FD) of UAM traffic by integrating theoretical analysis with physical experiments. To the best of our knowledge, this is the first study to derive UAM FDs using real-world physical experiment data. On the theoretical side, we design two drone control laws for collision avoidance and develop simulation-based traffic generation methods to produce diverse UAM traffic scenarios. Based on Edie's definition, traffic flow theory is then applied with a near-stationary traffic condition filtering method to construct the FD. To account for real-world disturbances and modeling uncertainties, we further conduct physical experiments on a reduced-scale testbed using Bitcraze Crazyflie drones. Both simulation and physical experiment trajectory data are collected and organized into the UAMTra2Flow dataset, which is analyzed using the proposed framework. Preliminary results indicate that classical FD structures for ground transportation, especially the Underwood model, are applicable to UAM systems. Notably, FD curves obtained from physical experiments exhibit deviations from simulation-based results, highlighting the importance of experimental validation. Finally, results from the reduced-scale testbed are scaled to realistic operating conditions to provide practical insights for future UAM traffic systems. The dataset and code for this paper are publicly available at this https URL.
>
---
#### [replaced 017] Soft Pneumatic Grippers: Topology optimization, 3D-printing and Experimental validation
- **分类: cs.RO**

- **简介: 该论文属于软体机械臂设计任务，旨在解决软性气动夹爪的结构优化问题。通过拓扑优化、3D打印和实验验证，提升夹爪的抓取性能。**

- **链接: [https://arxiv.org/pdf/2511.19211](https://arxiv.org/pdf/2511.19211)**

> **作者:** Prabhat Kumar; Chandra Prakash; Josh Pinskier; David Howard; Matthijs Langelaar
>
> **备注:** 11 Figures
>
> **摘要:** This paper presents a systematic topology optimization framework for designing a soft pneumatic gripper (SPG), explicitly considering the design-dependent nature of the actuating load. The load is modeled using Darcy's law with an added drainage term. A 2D soft arm unit is optimized by formulating it as a compliant mechanism design problem using the robust formulation. The problem is posed as a min-max optimization, where the output deformations of blueprint and eroded designs are considered. A volume constraint is imposed on the blueprint part, while a strain-energy constraint is enforced on the eroded part. The MMA is employed to solve the optimization problem and obtain the optimized soft unit. Finite element analysis with the Ogden material model confirms that the optimized 2D unit outperforms a conventional rectangular design under pneumatic loading. The optimized 2D unit is extruded to obtain a 3D module, and ten such units are assembled to create a soft arm. Deformation profiles of the optimized arm are analysed under different pressure loads. Four arms are 3D-printed and integrated with a supporting structure to realize the proposed SPG. The gripping performance of the SPG is demonstrated on objects with different weights, sizes, stiffness, and shapes.
>
---
#### [replaced 018] DA-VPC: Disturbance-Aware Visual Predictive Control Scheme of Docking Maneuvers for Autonomous Trolley Collection
- **分类: cs.RO**

- **简介: 该论文属于自主机器人 docking 任务，旨在解决视觉导航中的高精度和环境干扰问题。通过 DA-VPC 控制方案提升 docking 稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2509.07413](https://arxiv.org/pdf/2509.07413)**

> **作者:** Yuhan Pang; Bingyi Xia; Zhe Zhang; Zhirui Sun; Peijia Xie; Bike Zhu; Wenjun Xu; Jiankun Wang
>
> **摘要:** Service robots have demonstrated significant potential for autonomous trolley collection and redistribution in public spaces like airports or warehouses to improve efficiency and reduce cost. Usually, a fully autonomous system for the collection and transportation of multiple trolleys is based on a Leader-Follower formation of mobile manipulators, where reliable docking maneuvers of the mobile base are essential to align trolleys into organized queues. However, developing a vision-based robotic docking system faces significant challenges: high precision requirements, environmental disturbances, and inherent robot constraints. To address these challenges, we propose a Disturbance-Aware Visual Predictive Control (DA-VPC) scheme that incorporates active infrared markers for robust feature extraction across diverse lighting conditions. This framework explicitly models nonholonomic kinematics and visibility constraints for image-based visual servoing (IBVS), solving the predictive control problem through optimization. It is augmented with an extended state observer (ESO) designed to counteract disturbances during trolley pushing, ensuring precise and stable docking. Experimental results across diverse environments demonstrate the robustness of this system, with quantitative evaluations confirming high docking accuracy.
>
---
#### [replaced 019] HiCrowd: Hierarchical Crowd Flow Alignment for Dense Human Environments
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决密集人群中的冻结问题。通过结合强化学习与模型预测控制，使机器人跟随人流安全移动，提升导航效率与安全性。**

- **链接: [https://arxiv.org/pdf/2602.05608](https://arxiv.org/pdf/2602.05608)**

> **作者:** Yufei Zhu; Shih-Min Yang; Martin Magnusson; Allan Wang
>
> **备注:** 2026 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Navigating through dense human crowds remains a significant challenge for mobile robots. A key issue is the freezing robot problem, where the robot struggles to find safe motions and becomes stuck within the crowd. To address this, we propose HiCrowd, a hierarchical framework that integrates reinforcement learning (RL) with model predictive control (MPC). HiCrowd leverages surrounding pedestrian motion as guidance, enabling the robot to align with compatible crowd flows. A high-level RL policy generates a follow point to align the robot with a suitable pedestrian group, while a low-level MPC safely tracks this guidance with short horizon planning. The method combines long-term crowd aware decision making with safe short-term execution. We evaluate HiCrowd against reactive and learning-based baselines in offline setting (replaying recorded human trajectories) and online setting (human trajectories are updated to react to the robot in simulation). Experiments on a real-world dataset and a synthetic crowd dataset show that our method outperforms in navigation efficiency and safety, while reducing freezing behaviors. Our results suggest that leveraging human motion as guidance, rather than treating humans solely as dynamic obstacles, provides a powerful principle for safe and efficient robot navigation in crowds.
>
---
#### [replaced 020] Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多智能体交通仿真任务，旨在解决训练与测试分布差异导致的泛化问题。通过引入R1风格的强化微调方法，提升模型与人类行为的一致性。**

- **链接: [https://arxiv.org/pdf/2509.23993](https://arxiv.org/pdf/2509.23993)**

> **作者:** Muleilan Pei; Shaoshuai Shi; Shaojie Shen
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Scalable and realistic simulation of multi-agent traffic behavior is critical for advancing autonomous driving technologies. Although existing data-driven simulators have made significant strides in this domain, they predominantly rely on supervised learning to align simulated distributions with real-world driving scenarios. A persistent challenge, however, lies in the distributional shift that arises between training and testing, which often undermines model generalization in unseen environments. To address this limitation, we propose SMART-R1, a novel R1-style reinforcement fine-tuning paradigm tailored for next-token prediction models to better align agent behavior with human preferences and evaluation metrics. Our approach introduces a metric-oriented policy optimization algorithm to improve distribution alignment and an iterative "SFT-RFT-SFT" training strategy that alternates between Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT) to maximize performance gains. Extensive experiments on the large-scale Waymo Open Motion Dataset (WOMD) validate the effectiveness of this simple yet powerful R1-style training framework in enhancing foundation models. The results on the Waymo Open Sim Agents Challenge (WOSAC) showcase that SMART-R1 achieves state-of-the-art performance with an overall realism meta score of 0.7858, ranking first on the leaderboard at the time of submission.
>
---
#### [replaced 021] AoE: Always-on Egocentric Human Video Collection for Embodied AI
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出AoE系统，用于低成本收集全身视角视频数据，解决Embodied AI数据稀缺问题。通过手机和云边协同架构实现高效、广泛的数据采集与处理。**

- **链接: [https://arxiv.org/pdf/2602.23893](https://arxiv.org/pdf/2602.23893)**

> **作者:** Bowen Yang; Zishuo Li; Yang Sun; Changtao Miao; Yifan Yang; Man Luo; Xiaotong Yan; Feng Jiang; Jinchuan Shi; Yankai Fu; Ning Chen; Junkai Zhao; Pengwei Wang; Guocai Yao; Shanghang Zhang; Hao Chen; Zhe Li; Kai Zhu
>
> **摘要:** Embodied foundation models require large-scale, high-quality real-world interaction data for pre-training and scaling. However, existing data collection methods suffer from high infrastructure costs, complex hardware dependencies, and limited interaction scope, making scalable expansion challenging. In fact, humans themselves are ideal physically embodied agents. Therefore, obtaining egocentric real-world interaction data from globally distributed "human agents" offers advantages of low cost and sustainability. To this end, we propose the Always-on Egocentric (AoE) data collection system, which aims to simplify hardware dependencies by leveraging humans themselves and their smartphones, enabling low-cost, highly efficient, and scene-agnostic real-world interaction data collection to address the challenge of data scarcity. Specifically, we first employ an ergonomic neck-mounted smartphone holder to enable low-barrier, large-scale egocentric data collection through a cloud-edge collaborative architecture. Second, we develop a cross-platform mobile APP that leverages on-device compute for real-time processing, while the cloud hosts automated labeling and filtering pipelines that transform raw videos into high-quality training data. Finally, the AoE system supports distributed Ego video data collection by anyone, anytime, and anywhere. We evaluate AoE on data preprocessing quality and downstream tasks, demonstrating that high-quality egocentric data significantly boosts real-world generalization.
>
---
#### [replaced 022] HIMM: Human-Inspired Long-Term Memory Modeling for Embodied Exploration and Question Answering
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于具身问答与探索任务，解决长时观察下记忆不足的问题。提出非参数记忆框架，分离情景记忆与语义记忆，提升探索效率和推理能力。**

- **链接: [https://arxiv.org/pdf/2602.15513](https://arxiv.org/pdf/2602.15513)**

> **作者:** Ji Li; Bo Wang; Jing Xia; Mingyi Li; Shiyan Hu
>
> **摘要:** Deploying Multimodal Large Language Models as the brain of embodied agents remains challenging, particularly under long-horizon observations and limited context budgets. Existing memory assisted methods often rely on textual summaries, which discard rich visual and spatial details and remain brittle in non-stationary environments. In this work, we propose a non-parametric memory framework that explicitly disentangles episodic and semantic memory for embodied exploration and question answering. Our retrieval-first, reasoning-assisted paradigm recalls episodic experiences via semantic similarity and verifies them through visual reasoning, enabling robust reuse of past observations without rigid geometric alignment. In parallel, we introduce a program-style rule extraction mechanism that converts experiences into structured, reusable semantic memory, facilitating cross-environment generalization. Extensive experiments demonstrate state-of-the-art performance on embodied question answering and exploration benchmarks, yielding a 7.3% gain in LLM-Match and an 11.4% gain in LLM MatchXSPL on A-EQA, as well as +7.7% success rate and +6.8% SPL on GOAT-Bench. Analyses reveal that our episodic memory primarily improves exploration efficiency, while semantic memory strengthens complex reasoning of embodied agents.
>
---
#### [replaced 023] Ctrl-World: A Controllable Generative World Model for Robot Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在解决通用机器人策略评估与提升的难题。提出可控多视角世界模型，支持长时序一致性和精细动作控制，提升策略性能。**

- **链接: [https://arxiv.org/pdf/2510.10125](https://arxiv.org/pdf/2510.10125)**

> **作者:** Yanjiang Guo; Lucy Xiaoyang Shi; Jianyu Chen; Chelsea Finn
>
> **备注:** 17 pages
>
> **摘要:** Generalist robot policies can now perform a wide range of manipulation skills, but evaluating and improving their ability with unfamiliar objects and instructions remains a significant challenge. Rigorous evaluation requires a large number of real-world rollouts, while systematic improvement demands additional corrective data with expert labels. Both of these processes are slow, costly, and difficult to scale. World models offer a promising, scalable alternative by enabling policies to rollout within imagination space. However, a key challenge is building a controllable world model that can handle multi-step interactions with generalist robot policies. This requires a world model compatible with modern generalist policies by supporting multi-view prediction, fine-grained action control, and consistent long-horizon interactions, which is not achieved by previous works. In this paper, we make a step forward by introducing a controllable multi-view world model that can be used to evaluate and improve the instruction-following ability of generalist robot policies. Our model maintains long-horizon consistency with a pose-conditioned memory retrieval mechanism and achieves precise action control through frame-level action conditioning. Trained on the DROID dataset (95k trajectories, 564 scenes), our model generates spatially and temporally consistent trajectories under novel scenarios and new camera placements for over 20 seconds. We show that our method can accurately rank policy performance without real-world robot rollouts. Moreover, by synthesizing successful trajectories in imagination and using them for supervised fine-tuning, our approach can improve policy success by 44.7\%.
>
---
#### [replaced 024] Embodied intelligent industrial robotics: Framework and techniques
- **分类: cs.RO**

- **简介: 本文提出一种面向工业场景的具身智能机器人框架（EIIR），解决传统技术在效率、精度等方面的不足。通过五个模块设计，提升机器人在复杂环境中的适应能力。**

- **链接: [https://arxiv.org/pdf/2505.09305](https://arxiv.org/pdf/2505.09305)**

> **作者:** Chaoran Zhang; Chenhao Zhang; Zhaobo Xu; Qinghongbing Xie; Jinliang Hou; Pingfa Feng; Long Zeng
>
> **备注:** 71 pages, 13 figures. The associated project can be found at this https URL
>
> **摘要:** The combination of embodied intelligence and robots has great prospects and is becoming increasingly common. In order to work more efficiently, accurately, reliably, and safely in industrial scenarios, robots should have at least general knowledge, working-environment knowledge, and operating-object knowledge. These pose significant challenges to existing embodied intelligent robotics (EIR) techniques. Thus, this paper first briefly reviews the history of industrial robotics and analyzes the limitations of mainstream EIR frameworks. Then, a new knowledge-driven technical framework of embodied intelligent industrial robotics (EIIR) is proposed for various industrial environments. It has five modules: a world model, a high-level task planner, a low-level skill controller, a simulator, and a physical system. The development of techniques related to each module are also thoroughly reviewed, and recent progress regarding their adaption to industrial applications are discussed. A case study of real-world assembly system is given to demonstrate the newly proposed EIIR framework's applicability and potentiality. Finally, the key challenges that EIIR encounters in industrial scenarios are summarized and future research directions are suggested. The authors believe that EIIR technology is shaping the next generation of industrial robotics and EIIR-based industrial systems supply a new technological paradigm for intelligent manufacturing. It is expected that this review could serve as a valuable reference for scholars and engineers that are interested in industrial embodied intelligence. Together, scholars can use this research to drive their rapid advancement and application of EIIR techniques. The authors would continue to track and contribute new studies in the project page this https URL
>
---
#### [replaced 025] Coordinated Control of Multiple Construction Machines Using LLM-Generated Behavior Trees with Flag-Based Synchronization
- **分类: cs.RO**

- **简介: 该论文属于施工机械协同控制任务，旨在解决人工设计行为树效率低的问题。通过LLM生成行为树并结合同步标志实现多机协调，提升自动化水平。**

- **链接: [https://arxiv.org/pdf/2602.01041](https://arxiv.org/pdf/2602.01041)**

> **作者:** Akinosuke Tsutsumi; Tomoya Itsuka; Yuichiro Kasahara; Tomoya Kouno; Kota Akinari; Genki Yamauchi; Daisuke Endo; Taro Abe; Takeshi Hashimoto; Keiji Nagatani; Ryo Kurazume
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Earthwork operations face increasing demand, while workforce aging creates a growing need for automation. ROS2-TMS for Construction, a Cyber-Physical System framework for construction machinery automation, has been proposed; however, its reliance on manually designed Behavior Trees (BTs) limits scalability in cooperative operations. Recent advances in Large Language Models (LLMs) offer new opportunities for automated task planning, yet most existing studies remain limited to simple robotic systems. This paper proposes an LLM-based workflow for automatic generation of BTs toward coordinated operation of construction machines. The method introduces synchronization flags managed through a Global Blackboard, enabling multiple BTs to share execution states and represent inter-machine dependencies. The workflow consists of Action Sequence generation and BTs generation using LLMs. Simulation experiments on 30 construction instruction scenarios achieved up to 93\% success rate in coordinated multi-machine tasks. Real-world experiments using an excavator and a dump truck further demonstrate successful cooperative execution, indicating the potential to reduce manual BTs design effort in construction automation. These results highlight the feasibility of applying LLM-driven task planning to practical earthwork automation.
>
---
#### [replaced 026] Digital and Robotic Twinning for Validation of Proximity Operations and Formation Flying
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于航天器GNC系统验证任务，解决空间环境下GNC性能验证难题，提出数字与机器人孪生框架，实现软硬件联合测试与性能评估。**

- **链接: [https://arxiv.org/pdf/2507.20034](https://arxiv.org/pdf/2507.20034)**

> **作者:** Z. Ahmed; E. Bates; P. Francesch Huc; S. Y. W. Low; A. Golan; T. Bell; A. Rizza; S. D'Amico
>
> **摘要:** Spacecraft Rendezvous, Proximity Operations (RPO), and Formation Flying (FF) rely on safety-critical guidance, navigation and control (GNC) that must satisfy stringent performance and robustness requirements. However, verifying GNC performance is challenging due to the complexity and inaccessibility of the space environment, necessitating a verification and validation (V\&V) process that bridges simulation and real-world behavior. This paper contributes a unified, closed-loop, end-to-end digital and robotic twinning framework that enables software- and hardware-in-the-loop testing of spacecraft GNC systems. The framework is designed for modularity and flexibility, supporting interchangeable sensing modalities, control algorithms, and operational regimes. The digital twin includes an event-driven faster-than-real-time simulation environment to support rapid prototyping. The architecture is augmented with hardware-based robotic testbeds from Stanford's Space Rendezvous Laboratory (SLAB): the GNSS and Radiofrequency Autonomous Navigation Testbed for Distributed Space Systems (GRAND) to validate RF-based navigation techniques, and the Testbed for Rendezvous and Optical Navigation (TRON) and Optical Stimulator (OS) to validate vision-based methods. The test article for this work is an integrated multi-modal GNC software stack developed at SLAB. This paper introduces the hybrid twinning framework, summarizes calibration and error characterization of the robotic testbeds, and evaluates GNC performance across multiple operational modes in a full-range RPO scenario in LEO. The results demonstrate consistency between software- and hardware-in-the-loop tests with clear explainability for deviations in performance, thus validating the hybrid twinning pipeline as a reliable framework for realistic assessment and verification of GNC systems.
>
---
#### [replaced 027] TinyIO: Lightweight Reparameterized Inertial Odometry
- **分类: cs.RO**

- **简介: 该论文属于定位任务，旨在解决轻量级惯性里程计的高精度问题。提出TinyIO，通过多分支结构和双路径注意力机制，在减少参数量的同时提升性能。**

- **链接: [https://arxiv.org/pdf/2507.15293](https://arxiv.org/pdf/2507.15293)**

> **作者:** Shanshan Zhang; Mengzi Chen; Siyue Wang; Mengzhe Wang; Liqin Wu; Qi Zhang; Tianshui Wen
>
> **摘要:** Inertial odometry (IO) is a widely used approach for localization on mobile devices; however, obtaining a lightweight IO model that also achieves high accuracy remains challenging. To address this issue, we propose TinyIO, a lightweight IO method. During training, we adopt a multi-branch architecture to extract diverse motion features more effectively. At inference time, the trained multi-branch model is converted into an equivalent single-path architecture to reduce computational complexity. We further propose a Dual-Path Adaptive Attention mechanism (DPAA), which enhances TinyIO's perception of contextual motion along both channel and temporal dimensions with negligible additional parameters. Extensive experiments on public datasets demonstrate that our method attains a favorable trade-off between accuracy and model size. On the RoNIN dataset, TinyIO reduces the ATE by 23.53% compared with R-ResNet and decreases the parameter count by 3.68%.
>
---
#### [replaced 028] Large Scale Robotic Material Handling: Learning, Planning, and Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究大型机械臂的自主物料搬运任务，解决效率低、安全性差的问题。提出基于强化学习的抓取点规划和轨迹控制模块，提升搬运精度与安全。**

- **链接: [https://arxiv.org/pdf/2508.09003](https://arxiv.org/pdf/2508.09003)**

> **作者:** Filippo A. Spinelli; Yifan Zhai; Fang Nan; Pascal Egli; Julian Nubert; Thilo Bleumer; Lukas Miller; Ferdinand Hofmann; Marco Hutter
>
> **备注:** Final version published in IEEE Transactions on Field Robotics. It includes additional experiments and comparisons with classical methods
>
> **摘要:** Bulk material handling involves the efficient and precise moving of large quantities of materials, a core operation in many industries, including cargo ship unloading, waste sorting, construction, and demolition. These repetitive, labor-intensive, and safety-critical operations are typically performed using large hydraulic material handlers equipped with underactuated grippers. In this work, we present a comprehensive framework for the autonomous execution of large-scale material handling tasks. The system integrates specialized modules for environment perception, pile attack point selection, path planning, and motion control. The main contributions of this work are two reinforcement learning-based modules: an attack point planner that selects optimal grasping locations on the material pile to maximize removal efficiency and minimize the number of scoops, and a robust trajectory following controller that addresses the precision and safety challenges associated with underactuated grippers in movement, while utilizing their free-swinging nature to release material through dynamic throwing. We validate our framework through real-world experiments on a 40 t material handler in a representative worksite, focusing on two key tasks: high-throughput bulk pile management and high-precision truck loading. Comparative evaluations against human operators demonstrate the system's effectiveness in terms of precision, repeatability, and operational safety. To the best of our knowledge, this is the first complete automation of material handling tasks on a full scale.
>
---
#### [replaced 029] Sample-efficient and Scalable Exploration in Continuous-Time RL
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文研究连续时间强化学习问题，旨在提升算法的样本效率和可扩展性。通过概率模型学习系统动态，提出COMBRL算法，有效处理有奖励和无奖励场景。**

- **链接: [https://arxiv.org/pdf/2510.24482](https://arxiv.org/pdf/2510.24482)**

> **作者:** Klemens Iten; Lenart Treven; Bhavya Sukhija; Florian Dörfler; Andreas Krause
>
> **备注:** 28 pages, 8 figures, 6 tables. Published as a conference paper at ICLR 2026
>
> **摘要:** Reinforcement learning algorithms are typically designed for discrete-time dynamics, even though the underlying real-world control systems are often continuous in time. In this paper, we study the problem of continuous-time reinforcement learning, where the unknown system dynamics are represented using nonlinear ordinary differential equations (ODEs). We leverage probabilistic models, such as Gaussian processes and Bayesian neural networks, to learn an uncertainty-aware model of the underlying ODE. Our algorithm, COMBRL, greedily maximizes a weighted sum of the extrinsic reward and model epistemic uncertainty. This yields a scalable and sample-efficient approach to continuous-time model-based RL. We show that COMBRL achieves sublinear regret in the reward-driven setting, and in the unsupervised RL setting (i.e., without extrinsic rewards), we provide a sample complexity bound. In our experiments, we evaluate COMBRL in both standard and unsupervised RL settings and demonstrate that it scales better, is more sample-efficient than prior methods, and outperforms baselines across several deep RL tasks.
>
---
#### [replaced 030] TwinRL-VLA: Digital Twin-Driven Reinforcement Learning for Real-World Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在真实环境中探索效率低的问题。提出TwinRL框架，通过数字孪生技术提升探索效率和成功率。**

- **链接: [https://arxiv.org/pdf/2602.09023](https://arxiv.org/pdf/2602.09023)**

> **作者:** Qinwen Xu; Jiaming Liu; Rui Zhou; Shaojun Shi; Nuowei Han; Zhuoyang Liu; Chenyang Gu; Shuo Gu; Yang Yue; Gao Huang; Wenzhao Zheng; Sirui Han; Peng Jia; Shanghang Zhang
>
> **摘要:** Despite strong generalization capabilities, Vision-Language-Action (VLA) models remain constrained by the high cost of expert demonstrations and insufficient real-world interaction. While online reinforcement learning (RL) has shown promise in improving general foundation models, applying RL to VLA manipulation in real-world settings is still hindered by low exploration efficiency and a restricted exploration space. Through systematic real-world experiments, we observe that the effective exploration space of online RL is closely tied to the data distribution of supervised fine-tuning (SFT). Motivated by this observation, we propose TwinRL, a digital twin-real-world collaborative RL framework designed to scale and guide exploration for VLA models. First, a high-fidelity digital twin is efficiently reconstructed from smartphone-captured scenes, enabling realistic bidirectional transfer between real and simulated environments. During the SFT warm-up stage, we introduce an exploration space expansion strategy using digital twins to broaden the support of the data trajectory distribution. Building on this enhanced initialization, we propose a sim-to-real guided exploration strategy to further accelerate online RL. Specifically, TwinRL performs efficient and parallel online RL in the digital twin prior to deployment, effectively bridging the gap between offline and online training stages. Subsequently, we exploit efficient digital twin sampling to identify failure-prone yet informative configurations, which are used to guide targeted human-in-the-loop rollouts on the real robot. In our experiments, TwinRL approaches 100% success in both in-distribution regions covered by real-world demonstrations and out-of-distribution regions, delivering at least a 30% speedup over prior real-world RL methods and requiring only about 20 minutes on average across four tasks.
>
---
#### [replaced 031] Tru-POMDP: Task Planning Under Uncertainty via Tree of Hypotheses and Open-Ended POMDPs
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于任务规划领域，解决家庭服务机器人在不确定性环境下的任务规划问题。通过结合LLM与POMDP，提出Tru-POMDP方法，提升规划成功率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2506.02860](https://arxiv.org/pdf/2506.02860)**

> **作者:** Wenjing Tang; Xinyu He; Yongxi Huang; Yunxiao Xiao; Cewu Lu; Panpan Cai
>
> **摘要:** Task planning under uncertainty is essential for home-service robots operating in the real world. Tasks involve ambiguous human instructions, hidden or unknown object locations, and open-vocabulary object types, leading to significant open-ended uncertainty and a boundlessly large planning space. To address these challenges, we propose Tru-POMDP, a planner that combines structured belief generation using Large Language Models (LLMs) with principled POMDP planning. Tru-POMDP introduces a hierarchical Tree of Hypotheses (TOH), which systematically queries an LLM to construct high-quality particle beliefs over possible world states and human goals. We further formulate an open-ended POMDP model that enables rigorous Bayesian belief tracking and efficient belief-space planning over these LLM-generated hypotheses. Experiments on complex object rearrangement tasks across diverse kitchen environments show that Tru-POMDP significantly outperforms state-of-the-art LLM-based and LLM-tree-search hybrid planners, achieving higher success rates with significantly better plans, stronger robustness to ambiguity and occlusion, and greater planning efficiency.
>
---
#### [replaced 032] DA-MMP: Learning Coordinated and Accurate Throwing with Dynamics-Aware Motion Manifold Primitives
- **分类: cs.RO**

- **简介: 该论文聚焦于动态抓取任务，解决轨迹规划与执行不一致的问题。提出DA-MMP框架，通过学习运动流形生成协调的投掷轨迹，提升实际操作的成功率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.23721](https://arxiv.org/pdf/2509.23721)**

> **作者:** Chi Chu; Huazhe Xu
>
> **备注:** Accepted to ICRA 2026. Project page: this https URL
>
> **摘要:** Dynamic manipulation is a key capability for advancing robot performance, enabling skills such as tossing. While recent learning-based approaches have pushed the field forward, most methods still rely on manually designed action parameterizations, limiting their ability to produce the highly coordinated motions required in complex tasks. Motion planning can generate feasible trajectories, but the dynamics gap-stemming from control inaccuracies, contact uncertainties, and aerodynamic effects-often causes large deviations between planned and executed trajectories. In this work, we propose Dynamics-Aware Motion Manifold Primitives (DA-MMP), a motion generation framework for goal-conditioned dynamic manipulation, and instantiate it on a challenging real-world ring-tossing task. Our approach extends motion manifold primitives to variable-length trajectories through a compact parameterization and learns a high-quality manifold from a large-scale dataset of planned motions. Building on this manifold, a conditional flow matching model is trained in the latent space with a small set of real-world trials, enabling the generation of throwing trajectories that account for execution dynamics. Experiments show that our method can generate coordinated and smooth motion trajectories for the ring-tossing task. In real-world evaluations, it achieves high success rates and even surpasses the performance of trained human experts. Moreover, it generalizes to novel targets beyond the training range, indicating that it successfully learns the underlying trajectory-dynamics mapping.
>
---
#### [replaced 033] Large Language Model-Assisted UAV Operations and Communications: A Multifaceted Survey and Tutorial
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于无人机与大语言模型融合的研究，旨在提升无人机的智能水平。通过整合LLMs，解决无人机在环境理解、任务推理和协同控制等方面的问题，提出统一框架并探讨应用与伦理挑战。**

- **链接: [https://arxiv.org/pdf/2602.19534](https://arxiv.org/pdf/2602.19534)**

> **作者:** Yousef Emami; Hao Zhou; Radha Reddy; Atefeh Hajijamali Arani; Biliang Wang; Kai Li; Luis Almeida; Zhu Han
>
> **备注:** 40 pages, 10 figures, 13 tables
>
> **摘要:** Uncrewed Aerial Vehicles (UAVs) are widely deployed across diverse applications due to their mobility and agility. Recent advances in Large Language Models (LLMs) offer a transformative opportunity to enhance UAV intelligence beyond conventional optimization-based and learning-based approaches. By integrating LLMs into UAV systems, advanced environmental understanding, swarm coordination, mobility optimization, and high-level task reasoning can be achieved, thereby allowing more adaptive and context-aware aerial operations. This survey systematically explores the intersection of LLMs and UAV technologies and proposes a unified framework that consolidates existing architectures, methodologies, and applications for UAVs. We first present a structured taxonomy of LLM adaptation techniques for UAVs, including pretraining, fine-tuning, Retrieval-Augmented Generation (RAG), and prompt engineering, along with key reasoning capabilities such as Chain-of-Thought (CoT) and In-Context Learning (ICL). We then examine LLM-assisted UAV communications and operations, covering navigation, mission planning, swarm control, safety, autonomy, and network management. After that, the survey further discusses Multimodal LLMs (MLLMs) for human-swarm interaction, perception-driven navigation, and collaborative control. Finally, we address ethical considerations, including bias, transparency, accountability, and Human-in-the-Loop (HITL) strategies, and outline future research directions. Overall, this work positions LLM-assisted UAVs as a foundation for intelligent and adaptive aerial systems.
>
---
#### [replaced 034] Goal Reaching with Eikonal-Constrained Hierarchical Quasimetric Reinforcement Learning
- **分类: cs.LG; cs.RO; eess.SY; stat.ML**

- **简介: 该论文属于目标导向强化学习任务，旨在解决奖励设计困难问题。提出Eik-QRL和Eik-HiQRL方法，提升导航与操作任务性能。**

- **链接: [https://arxiv.org/pdf/2512.12046](https://arxiv.org/pdf/2512.12046)**

> **作者:** Vittorio Giammarino; Ahmed H. Qureshi
>
> **摘要:** Goal-Conditioned Reinforcement Learning (GCRL) mitigates the difficulty of reward design by framing tasks as goal reaching rather than maximizing hand-crafted reward signals. In this setting, the optimal goal-conditioned value function naturally forms a quasimetric, motivating Quasimetric RL (QRL), which constrains value learning to quasimetric mappings and enforces local consistency through discrete, trajectory-based constraints. We propose Eikonal-Constrained Quasimetric RL (Eik-QRL), a continuous-time reformulation of QRL based on the Eikonal Partial Differential Equation (PDE). This PDE-based structure makes Eik-QRL trajectory-free, requiring only sampled states and goals, while improving out-of-distribution generalization. We provide theoretical guarantees for Eik-QRL and identify limitations that arise under complex dynamics. To address these challenges, we introduce Eik-Hierarchical QRL (Eik-HiQRL), which integrates Eik-QRL into a hierarchical decomposition. Empirically, Eik-HiQRL achieves state-of-the-art performance in offline goal-conditioned navigation and yields consistent gains over QRL in manipulation tasks, matching temporal-difference methods.
>
---
#### [replaced 035] Neuro-Symbolic Skill Discovery for Conditional Multi-Level Planning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于多级规划任务，旨在从少量低级轨迹中学习通用的高级符号技能。通过神经符号架构和多级规划管道，实现高效、长时序的任务规划与执行。**

- **链接: [https://arxiv.org/pdf/2410.10045](https://arxiv.org/pdf/2410.10045)**

> **作者:** Hakan Aktas; Yigit Yildirim; Ahmet Firat Gamsiz; Deniz Bilge Akkoc; Erhan Oztop; Emre Ugur
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** This paper proposes a novel learning architecture for acquiring generalizable high-level symbolic skills from a few unlabeled low-level skill trajectory demonstrations. The architecture involves neural networks for symbol discovery and low-level controller acquisition and a multi-level planning pipeline that utilizes the discovered symbols and the learned low-level controllers. The discovered action symbols are automatically interpreted using visual language models that are also responsible for generating high-level plans. While extracting high-level symbols, our model preserves the low-level information so that low-level action planning can be carried out by using gradient-based planning. To assess the efficacy of our method, we tested the high and low-level planning performance of our architecture by using simulated and real-world experiments across various tasks. The experiments have shown that our method is able to manipulate objects in unseen locations and plan and execute long-horizon tasks by using novel action sequences, even in highly cluttered environments when cued by only a few demonstrations that cover small regions of the environment.
>
---
#### [replaced 036] Return Augmented Decision Transformer for Off-Dynamics Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文研究离线非动态强化学习任务，解决源域数据迁移至目标域时的动力学差异问题。提出REAG方法，通过调整回报分布提升决策变换器性能。**

- **链接: [https://arxiv.org/pdf/2410.23450](https://arxiv.org/pdf/2410.23450)**

> **作者:** Ruhan Wang; Yu Yang; Zhishuai Liu; Dongruo Zhou; Pan Xu
>
> **备注:** 26 pages, 11 tables, 8 figures. Published in Transactions on Machine Learning Research (TMLR)
>
> **摘要:** We study offline off-dynamics reinforcement learning (RL) to utilize data from an easily accessible source domain to enhance policy learning in a target domain with limited data. Our approach centers on return-conditioned supervised learning (RCSL), particularly focusing on Decision Transformer (DT) type frameworks, which can predict actions conditioned on desired return guidance and complete trajectory history. Previous works address the dynamics shift problem by augmenting the reward in the trajectory from the source domain to match the optimal trajectory in the target domain. However, this strategy can not be directly applicable in RCSL owing to (1) the unique form of the RCSL policy class, which explicitly depends on the return, and (2) the absence of a straightforward representation of the optimal trajectory distribution. We propose the Return Augmented (REAG) method for DT type frameworks, where we augment the return in the source domain by aligning its distribution with that in the target domain. We provide the theoretical analysis demonstrating that the RCSL policy learned from REAG achieves the same level of suboptimality as would be obtained without a dynamics shift. We introduce two practical implementations REAG$_\text{Dara}^{*}$ and REAG$_\text{MV}^{*}$ respectively. Thorough experiments on D4RL datasets and various DT-type baselines demonstrate that our methods consistently enhance the performance of DT type frameworks in off-dynamics RL.
>
---
#### [replaced 037] OmniVLA: Physically-Grounded Multimodal VLA with Unified Multi-Sensor Perception for Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出OmniVLA，解决机器人操作中感知能力不足的问题。通过融合多传感器数据，提升空间智能与操作性能。**

- **链接: [https://arxiv.org/pdf/2511.01210](https://arxiv.org/pdf/2511.01210)**

> **作者:** Heyu Guo; Shanmu Wang; Ruichun Ma; Shiqi Jiang; Yasaman Ghasempour; Omid Abari; Baining Guo; Lili Qiu
>
> **备注:** Accepted by ICRA'26
>
> **摘要:** Vision-language-action (VLA) models have shown strong generalization for robotic action prediction through large-scale vision-language pretraining. However, most existing models rely solely on RGB cameras, limiting their perception and, consequently, manipulation capabilities. We present OmniVLA, an omni-modality VLA model that integrates novel sensing modalities for physically-grounded spatial intelligence beyond RGB perception. The core of our approach is the sensor-masked image, a unified representation that overlays spatially grounded and physically meaningful masks onto the RGB images, derived from sensors including an infrared camera, a mmWave radar, and a microphone array. This image-native unification keeps sensor input close to RGB statistics to facilitate training, provides a uniform interface across sensor hardware, and enables data-efficient learning with lightweight per-sensor projectors. Built on this, we present a multisensory vision-language-action model architecture and train the model based on an RGB-pretrained VLA backbone. We evaluate OmniVLA on challenging real-world tasks where sensor-modality perception guides the robotic manipulation. OmniVLA achieves an average task success rate of 84%, significantly outperforms both RGB-only and raw-sensor-input baseline models by 59% and 28% respectively, meanwhile showing higher learning efficiency and stronger generalization capability.
>
---
#### [replaced 038] Dense-Jump Flow Matching with Non-Uniform Time Scheduling for Robotic Policies: Mitigating Multi-Step Inference Degradation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人策略学习任务，解决多步推理退化问题。通过非均匀时间调度和密集跳跃集成，提升策略性能。**

- **链接: [https://arxiv.org/pdf/2509.13574](https://arxiv.org/pdf/2509.13574)**

> **作者:** Zidong Chen; Zihao Guo; Peng Wang; ThankGod Itua Egbe; Yan Lyu; Chenghao Qian
>
> **摘要:** Flow matching has emerged as a competitive framework for learning high-quality generative policies in robotics; however, we find that generalisation arises and saturates early along the flow trajectory, in accordance with recent findings in the literature. We further observe that increasing the number of Euler integration steps during inference counter-intuitively and universally degrades policy performance. We attribute this to (i) additional, uniformly spaced integration steps oversample the late-time region, thereby constraining actions towards the training trajectories and reducing generalisation; and (ii) the learned velocity field becoming non-Lipschitz as integration time approaches 1, causing instability. To address these issues, we propose a novel policy that utilises non-uniform time scheduling (e.g., U-shaped) during training, which emphasises both early and late temporal stages to regularise policy training, and a dense-jump integration schedule at inference, which uses a single-step integration to replace the multi-step integration beyond a jump point, to avoid unstable areas around 1. Essentially, our policy is an efficient one-step learner that still pushes forward performance through multi-step integration, yielding up to 23.7% performance gains over state-of-the-art baselines across diverse robotic tasks.
>
---
#### [replaced 039] Scalable Multi-Task Learning through Spiking Neural Networks with Adaptive Task-Switching Policy for Intelligent Autonomous Agents
- **分类: cs.NE; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于多任务学习领域，旨在解决自主智能体在资源受限下多任务训练中的任务干扰问题。提出SwitchMT方法，通过自适应任务切换策略提升性能与可扩展性。**

- **链接: [https://arxiv.org/pdf/2504.13541](https://arxiv.org/pdf/2504.13541)**

> **作者:** Rachmad Vidya Wicaksana Putra; Avaneesh Devkota; Muhammad Shafique
>
> **备注:** Accepted at the 63rd ACM/IEEE Design Automation Conference (DAC), July 26-29, 2026 in Long Beach, CA, USA
>
> **摘要:** Training resource-constrained autonomous agents on multiple tasks simultaneously is crucial for adapting to diverse real-world environments. Recent works employ reinforcement learning (RL) approach, but they still suffer from sub-optimal multi-task performance due to task interference. State-of-the-art works employ Spiking Neural Networks (SNNs) to improve RL-based multi-task learning and enable low-power/energy operations through network enhancements and spike-driven data stream processing. However, they rely on fixed task-switching intervals during its training, thus limiting its performance and scalability. To address this, we propose SwitchMT, a novel methodology that employs adaptive task-switching for effective, scalable, and simultaneous multi-task learning. SwitchMT employs the following key ideas: (1) leveraging a Deep Spiking Q-Network with active dendrites and dueling structure, that utilizes task-specific context signals to create specialized sub-networks; and (2) devising an adaptive task-switching policy that leverages both rewards and internal dynamics of the network parameters. Experimental results demonstrate that SwitchMT achieves competitive scores in multiple Atari games (i.e., Pong: -8.8, Breakout: 5.6, and Enduro: 355.2) and longer game episodes as compared to the state-of-the-art. These results also highlight the effectiveness of SwitchMT methodology in addressing task interference without increasing the network complexity, enabling intelligent autonomous agents with scalable multi-task learning capabilities.
>
---
#### [replaced 040] AssemMate: Graph-Based LLM for Robotic Assembly Assistance
- **分类: cs.RO**

- **简介: 该论文提出AssemMate，一种基于图的LLM系统，用于机器人装配辅助。解决传统方法在实时性和精确性上的不足，通过知识图谱提升装配任务规划与人机交互效率。**

- **链接: [https://arxiv.org/pdf/2509.11617](https://arxiv.org/pdf/2509.11617)**

> **作者:** Qi Zheng; Chaoran Zhang; Zijian Liang; EnTe Lin; Shubo Cui; Qinghongbing Xie; Zhaobo Xu; Long Zeng
>
> **摘要:** Large Language Model (LLM)-based robotic assembly assistance has gained significant research attention. It requires the injection of domain-specific knowledge to guide the assembly process through natural language interaction with humans. Despite some progress, existing methods represent knowledge in the form of natural language text. Due to the long context and redundant content, they struggle to meet the robots' requirements for real-time and precise reasoning. In order to bridge this gap, we present AssemMate, which utilizes the graph\textemdash a concise and accurate form of knowledge representation\textemdash as input. This graph-based LLM enables knowledge graph question answering (KGQA), supporting human-robot interaction and assembly task planning for specific products. Beyond interactive QA, AssemMate also supports sensing stacked scenes and executing grasping to assist with assembly. Specifically, a self-supervised Graph Convolutional Network (GCN) encodes knowledge graph entities and relations into a latent space and aligns them with LLM's representation, enabling the LLM to understand graph information. In addition, a vision-enhanced strategy is employed to address stacked scenes in grasping. Through training and evaluation, AssemMate outperforms existing methods, achieving 6.4\% higher accuracy, 3 times faster inference, and 28 times shorter context length, while demonstrating strong generalization ability on random graphs. And our approach further demonstrates superiority through robotic grasping experiments in both simulated and real-world settings. More details can be found on the project page: this https URL
>
---
#### [replaced 041] RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RoboPARA框架，解决双臂机器人任务并行性优化问题，通过两阶段方法提升多任务处理效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2506.06683](https://arxiv.org/pdf/2506.06683)**

> **作者:** Shiying Duan; Pei Ren; Nanxiang Jiang; Zhengping Che; Jian Tang; Zhaoxin Fan; Yifan Sun; Wenjun Wu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Dual-arm robots play a crucial role in improving efficiency and flexibility in complex multitasking this http URL existing methods have achieved promising results in task planning, they often fail to fully optimize task parallelism, limiting the potential of dual-arm this http URL address this issue, we propose RoboPARA, a novel large language model (LLM)-driven framework for dual-arm task parallelism this http URL employs a two-stage process: (1) Dependency Graph-based Planning Candidates Generation, which constructs directed acyclic graphs (DAGs) to model task dependencies and eliminate redundancy, and (2) Graph Re-Traversal-based Dual-Arm Parallel Planning, which optimizes DAG traversal to maximize parallelism while maintaining task this http URL addition, we introduce the Cross-Scenario Dual-Arm Parallel Task dataset (X-DAPT dataset), the first dataset specifically designed to evaluate dual-arm task parallelism across diverse scenarios and difficulty this http URL experiments demonstrate that RoboPARA significantly outperforms existing planning methods, achieving higher efficiency and reliability, particularly in complex task this http URL code is publicly available at this https URL.
>
---
#### [replaced 042] Endowing Embodied Agents with Spatial Reasoning Capabilities for Vision-and-Language Navigation
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决机器人在真实环境中空间感知不足导致的幻觉问题。提出BrainNav框架，结合双地图和双定向策略，提升导航准确性与适应性。**

- **链接: [https://arxiv.org/pdf/2504.08806](https://arxiv.org/pdf/2504.08806)**

> **作者:** Qianqian Bai; Zhongpu Chen; Ling Luo; Huaming Du; Yuqian Lei; Ziyun Jiao
>
> **摘要:** Enhancing the spatial perception capabilities of mobile robots is crucial for achieving embodied Vision-and-Language Navigation (VLN). Although significant progress has been made in simulated environments, directly transferring these capabilities to real-world scenarios often results in severe hallucination phenomena, causing robots to lose effective spatial awareness. To address this issue, we propose BrainNav, a bio-inspired spatial cognitive navigation framework inspired by biological spatial cognition theories and cognitive map theory. BrainNav integrates dual-map (coordinate map and topological map) and dual-orientation (relative orientation and absolute orientation) strategies, enabling real-time navigation through dynamic scene capture and path planning. Its five core modules-Hippocampal Memory Hub, Visual Cortex Perception Engine, Parietal Spatial Constructor, Prefrontal Decision Center, and Cerebellar Motion Execution Unit-mimic biological cognitive functions to reduce spatial hallucinations and enhance adaptability. Validated in a zero-shot real-world lab environment using the Limo Pro robot, BrainNav, compatible with GPT-4, outperforms existing State-of-the-Art (SOTA) Vision-and-Language Navigation in Continuous Environments (VLN-CE) methods without fine-tuning.
>
---
#### [replaced 043] Robust Finetuning of Vision-Language-Action Robot Policies via Parameter Merging
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，旨在解决微调过程中泛化能力下降的问题。通过权重插值保持通用策略的泛化能力，同时学习新技能。**

- **链接: [https://arxiv.org/pdf/2512.08333](https://arxiv.org/pdf/2512.08333)**

> **作者:** Yajat Yadav; Zhiyuan Zhou; Andrew Wagenmaker; Karl Pertsch; Sergey Levine
>
> **摘要:** Generalist robot policies, trained on large and diverse datasets, have demonstrated the ability to generalize across a wide spectrum of behaviors, enabling a single policy to act in varied real-world environments. However, they still fall short on new tasks not covered in the training data. When finetuned on limited demonstrations of a new task, these policies often overfit to the specific demonstrations--not only losing their prior abilities to solve a wide variety of generalist tasks but also failing to generalize within the new task itself. In this work, we aim to develop a method that preserves the generalization capabilities of the generalist policy during finetuning, allowing a single policy to robustly incorporate a new skill into its repertoire. Our goal is a single policy that both learns to generalize to variations of the new task and retains the broad competencies gained from pretraining. We show that this can be achieved through a simple yet effective strategy: interpolating the weights of a finetuned model with that of the pretrained model. We show, across extensive simulated and real-world experiments, that such model merging produces a single model that inherits the generalist abilities of the base model and learns to solve the new task robustly, outperforming both the pretrained and finetuned model on out-of-distribution variations of the new task. Moreover, we show that model merging performance scales with the amount of pretraining data, and enables continual acquisition of new skills in a lifelong learning setting, without sacrificing previously learned generalist abilities.
>
---
#### [replaced 044] $\rm{A}^{\rm{SAR}}$: $\varepsilon$-Optimal Graph Search for Minimum Expected-Detection-Time Paths with Path Budget Constraints for Search and Rescue (SAR)
- **分类: cs.RO**

- **简介: 该论文属于搜索路径规划任务，解决SAR中在路径预算约束下寻找最短期望探测时间路径的问题。提出$\rm{A}^{\rm{SAR}}$算法，保证解的最优性近似。**

- **链接: [https://arxiv.org/pdf/2511.10792](https://arxiv.org/pdf/2511.10792)**

> **作者:** Eric Mugford; Jonathan D. Gammell
>
> **备注:** IEEE International Conference on Robotics and Automation (ICRA) 2026, 8 pages, 4 figures, 2 tables. The corresponding video can be found at this https URL
>
> **摘要:** Searches are conducted to find missing persons and/or objects given uncertain information, imperfect observers and large search areas in Search and Rescue (SAR). In many scenarios, such as Maritime SAR, expected survival times are short and optimal search could increase the likelihood of success. This optimization problem is complex for nontrivial problems given its probabilistic nature. Stochastic optimization methods search large problems by nondeterministically sampling the space to reduce the effective size of the problem. This has been used in SAR planning to search otherwise intractably large problems but the stochastic nature provides no formal guarantees on the quality of solutions found in finite time. This paper instead presents $\rm{A}^{\rm{SAR}}$, an $\varepsilon$-optimal search algorithm for SAR planning. It calculates a heuristic to bound the search space and uses graph-search methods to find solutions that are formally guaranteed to be within a user-specified factor, $\varepsilon$, of the optimal solution. It finds better solutions faster than existing optimization approaches in operational simulations. It is also demonstrated with a real-world field trial on Lake Ontario, Canada, where it was used to locate a drifting manikin in only 150s.
>
---
#### [replaced 045] Physically Ground Commonsense Knowledge for Articulated Object Manipulation with Analytic Concepts
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决将语义层面的常识知识与物理世界有效结合的问题。通过引入分析概念作为桥梁，实现物理驱动的物体操作控制。**

- **链接: [https://arxiv.org/pdf/2503.23348](https://arxiv.org/pdf/2503.23348)**

> **作者:** Jiude Wei; Yuxuan Li; Cewu Lu; Jianhua Sun
>
> **摘要:** We humans rely on a wide range of commonsense knowledge to interact with an extensive number and categories of objects in the physical world. Likewise, such commonsense knowledge is also crucial for robots to successfully develop generalized object manipulation skills. While recent advancements in Multi-modal Large Language Models (MLLMs) have showcased their impressive capabilities in acquiring commonsense knowledge and conducting commonsense reasoning, effectively grounding this semantic-level knowledge produced by MLLMs to the physical world to thoroughly guide robots in generalized articulated object manipulation remains a challenge that has not been sufficiently addressed. To this end, we introduce analytic concepts, procedurally defined upon mathematical symbolism that can be directly computed and simulated by machines. By leveraging the analytic concepts as a bridge between the semantic-level knowledge inferred by MLLMs and the physical world where real robots operate, we can figure out the knowledge of object structure and functionality with physics-informed representations, and then use the physically grounded knowledge to instruct robot control policies for generalized and accurate articulated object manipulation. Extensive experiments in both real world and simulation demonstrate the superiority of our approach.
>
---
#### [replaced 046] High-Performance Dual-Arm Task and Motion Planning for Tabletop Rearrangement
- **分类: cs.RO**

- **简介: 该论文提出SDAR框架，解决双臂桌面上物体重新排列任务。通过整合任务规划与运动规划，提升操作效率与成功率。**

- **链接: [https://arxiv.org/pdf/2512.08206](https://arxiv.org/pdf/2512.08206)**

> **作者:** Duo Zhang; Junshan Huang; Jingjin Yu
>
> **备注:** ICRA 2026 Submission
>
> **摘要:** We propose Synchronous Dual-Arm Rearrangement Planner (SDAR), a task and motion planning (TAMP) framework for tabletop rearrangement, where two robot arms equipped with 2-finger grippers must work together in close proximity to rearrange objects whose start and goal configurations are strongly entangled. To tackle such challenges, SDAR tightly knit together its dependency-driven task planner (SDAR-T) and synchronous dual-arm motion planner (SDAR-M), to intelligently sift through a large number of possible task and motion plans. Specifically, SDAR-T applies a simple yet effective strategy to decompose the global object dependency graph induced by the rearrangement task, to produce more optimal dual-arm task plans than solutions derived from optimal task plans for a single arm. Leveraging state-of-the-art GPU SIMD-based motion planning tools, SDAR-M employs a layered motion planning strategy to sift through many task plans for the best synchronous dual-arm motion plan while ensuring high levels of success rate. Comprehensive evaluation demonstrates that SDAR delivers a 100% success rate in solving complex, non-monotone, long-horizon tabletop rearrangement tasks with solution quality far exceeding the previous state-of-the-art. Experiments on two UR-5e arms further confirm SDAR directly and reliably transfers to robot hardware. Source code and supplementary materials are available at this https URL.
>
---
#### [replaced 047] ULC: A Unified and Fine-Grained Controller for Humanoid Loco-Manipulation
- **分类: cs.RO**

- **简介: 该论文属于人形机器人运动与操作任务，解决子系统协调不足的问题。提出统一控制器ULC，实现全身协同控制，提升跟踪精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2507.06905](https://arxiv.org/pdf/2507.06905)**

> **作者:** Wandong Sun; Luying Feng; Baoshi Cao; Yang Liu; Yaochu Jin; Zongwu Xie
>
> **摘要:** Loco-Manipulation for humanoid robots aims to enable robots to integrate mobility with upper-body tracking capabilities. Most existing approaches adopt hierarchical architectures that decompose control into isolated upper-body (manipulation) and lower-body (locomotion) policies. While this decomposition reduces training complexity, it inherently limits coordination between subsystems and contradicts the unified whole-body control exhibited by humans. We demonstrate that a single unified policy can achieve a combination of tracking accuracy, large workspace, and robustness for humanoid loco-manipulation. We propose the Unified Loco-Manipulation Controller (ULC), a single-policy framework that simultaneously tracks root velocity, root height, torso rotation, and dual-arm joint positions in an end-to-end manner, proving the feasibility of unified control without sacrificing performance. We achieve this unified control through key technologies: sequence skill acquisition for progressive learning complexity, residual action modeling for fine-grained control adjustments, command polynomial interpolation for smooth motion transitions, random delay release for robustness to deploy variations, load randomization for generalization to external disturbances, and center-of-gravity tracking for providing explicit policy gradients to maintain stability. We validate our method on the Unitree G1 humanoid robot with 3-DOF (degrees-of-freedom) waist. Compared with strong baselines, ULC shows better tracking performance to disentangled methods and demonstrating larger workspace coverage. The unified dual-arm tracking enables precise manipulation under external loads while maintaining coordinated whole-body control for complex loco-manipulation tasks.
>
---
#### [replaced 048] UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出UrbanVerse，用于构建高保真城市仿真环境，解决AI代理训练中场景多样性与真实性的不足。通过视频生成物理交互场景，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2510.15018](https://arxiv.org/pdf/2510.15018)**

> **作者:** Mingxuan Liu; Honglin He; Elisa Ricci; Wayne Wu; Bolei Zhou
>
> **备注:** Accepted to ICLR 2026. Project page: this https URL
>
> **摘要:** Urban embodied AI agents, ranging from delivery robots to quadrupeds, are increasingly populating our cities, navigating chaotic streets to provide last-mile connectivity. Training such agents requires diverse, high-fidelity urban environments to scale, yet existing human-crafted or procedurally generated simulation scenes either lack scalability or fail to capture real-world complexity. We introduce UrbanVerse, a data-driven real-to-sim system that converts crowd-sourced city-tour videos into physics-aware, interactive simulation scenes. UrbanVerse consists of: (i) UrbanVerse-100K, a repository of 100k+ annotated urban 3D assets with semantic and physical attributes, and (ii) UrbanVerse-Gen, an automatic pipeline that extracts scene layouts from video and instantiates metric-scale 3D simulations using retrieved assets. Running in IsaacSim, UrbanVerse offers 160 high-quality constructed scenes from 24 countries, along with a curated benchmark of 10 artist-designed test scenes. Experiments show that UrbanVerse scenes preserve real-world semantics and layouts, achieving human-evaluated realism comparable to manually crafted scenes. In urban navigation, policies trained in UrbanVerse exhibit scaling power laws and strong generalization, improving success by +6.3% in simulation and +30.1% in zero-shot sim-to-real transfer comparing to prior methods, accomplishing a 300 m real-world mission with only two interventions.
>
---
#### [replaced 049] CRISP: Contact-Guided Real2Sim from Monocular Video with Planar Scene Primitives
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出CRISP，用于从单目视频中恢复可模拟的人体运动和场景几何，解决人体与场景交互的物理合理性问题。通过平面基元拟合和强化学习确保重建的准确性与真实性。**

- **链接: [https://arxiv.org/pdf/2512.14696](https://arxiv.org/pdf/2512.14696)**

> **作者:** Zihan Wang; Jiashun Wang; Jeff Tan; Yiwen Zhao; Jessica Hodgins; Shubham Tulsiani; Deva Ramanan
>
> **备注:** Published at ICLR 2026. Project page: this https URL
>
> **摘要:** We introduce CRISP, a method that recovers simulatable human motion and scene geometry from monocular video. Prior work on joint human-scene reconstruction relies on data-driven priors and joint optimization with no physics in the loop, or recovers noisy geometry with artifacts that cause motion tracking policies with scene interactions to fail. In contrast, our key insight is to recover convex, clean, and simulation-ready geometry by fitting planar primitives to a point cloud reconstruction of the scene, via a simple clustering pipeline over depth, normals, and flow. To reconstruct scene geometry that might be occluded during interactions, we make use of human-scene contact modeling (e.g., we use human posture to reconstruct the occluded seat of a chair). Finally, we ensure that human and scene reconstructions are physically-plausible by using them to drive a humanoid controller via reinforcement learning. Our approach reduces motion tracking failure rates from 55.2\% to 6.9\% on human-centric video benchmarks (EMDB, PROX), while delivering a 43\% faster RL simulation throughput. We further validate it on in-the-wild videos including casually-captured videos, Internet videos, and even Sora-generated videos. This demonstrates CRISP's ability to generate physically-valid human motion and interaction environments at scale, greatly advancing real-to-sim applications for robotics and AR/VR.
>
---
#### [replaced 050] Query-Based Adaptive Aggregation for Multi-Dataset Joint Training Toward Universal Visual Place Recognition
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，旨在解决多数据集联合训练中特征聚合能力不足的问题。提出QAA方法，通过查询自适应聚合提升模型泛化性与性能。**

- **链接: [https://arxiv.org/pdf/2507.03831](https://arxiv.org/pdf/2507.03831)**

> **作者:** Jiuhong Xiao; Yang Zhou; Giuseppe Loianno
>
> **备注:** 8 pages, 4 figures, accepted at ICRA 2026
>
> **摘要:** Deep learning methods for Visual Place Recognition (VPR) have advanced significantly, largely driven by large-scale datasets. However, most existing approaches are trained on a single dataset, which can introduce dataset-specific inductive biases and limit model generalization. While multi-dataset joint training offers a promising solution for developing universal VPR models, divergences among training datasets can saturate the limited information capacity in feature aggregation layers, leading to suboptimal performance. To address these challenges, we propose Query-based Adaptive Aggregation (QAA), a novel feature aggregation technique that leverages learned queries as reference codebooks to effectively enhance information capacity without significant computational or parameter complexity. We show that computing the Cross-query Similarity (CS) between query-level image features and reference codebooks provides a simple yet effective way to generate robust descriptors. Our results demonstrate that QAA outperforms state-of-the-art models, achieving balanced generalization across diverse datasets while maintaining peak performance comparable to dataset-specific models. Ablation studies further explore QAA's mechanisms and scalability. Visualizations reveal that the learned queries exhibit diverse attention patterns across datasets. Project page: this http URL.
>
---
#### [replaced 051] Design and Control of a Compact Series Elastic Actuator Module for Robots in MRI Scanners
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决MRI环境中力控执行器不足的问题。设计了一种紧凑的串联弹性执行器模块，并开发了基于扰动观测器的控制器，实现精准扭矩控制。**

- **链接: [https://arxiv.org/pdf/2406.07670](https://arxiv.org/pdf/2406.07670)**

> **作者:** Binghan He; Naichen Zhao; David Y. Guo; Charles H. Paxson; Alfredo De Goyeneche; Michael Lustig; Chunlei Liu; Ronald S. Fearing
>
> **摘要:** Robotic assistance has broadened the capabilities of magnetic resonance imaging (MRI)-guided medical interventions, yet force-controlled actuators tailored for MRI environments remain limited. In this study, we present a novel MRI-compatible rotary series elastic actuator (SEA) module that employs velocity-sourced ultrasonic motors for force-controlled operation within MRI scanners. Unlike prior MRI-compatible SEA designs, our module uses a transmission force sensing SEA architecture, with four off-the-shelf compression springs placed between the gearbox and motor housings. To enable precise torque control, we develop a controller based on a disturbance observer, specifically designed for velocity-sourced motors. This controller improves torque regulation, even under varying external impedance, enhancing the actuator's suitability for MRI-guided medical interventions. Experimental validation confirms effective torque control in both 3 Tesla MRI and non-MRI settings, achieving a 5% settling time of 0.05 seconds and steady-state error within 2.5% of the actuator's maximum output torque. Notably, the controller maintains consistent performance across both low and high impedance conditions.
>
---
#### [replaced 052] Decentralized Multi-Robot Obstacle Detection and Tracking in a Maritime Scenario
- **分类: cs.RO**

- **简介: 该论文属于多机器人协同任务，解决海上障碍物检测与跟踪问题。通过无人机与水面船协作，采用分布式框架提升跟踪精度与一致性。**

- **链接: [https://arxiv.org/pdf/2602.12012](https://arxiv.org/pdf/2602.12012)**

> **作者:** Muhammad Farhan Ahmed; Vincent Frémont
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** Autonomous aerial-surface robot teams offer a scalable solution for maritime monitoring, but deployment remains difficult due to water-induced visual artifacts and bandwidth-limited coordination. This paper presents a decentralized multi-robot framework to detect and track floating containers using multiple UAVs cooperating with an autonomous surface vessel. Each UAV runs a YOLOv8 detector augmented with stereo disparity and maintains per-target EKF tracks with uncertainty-aware data association. Robots exchange compact track summaries that are fused conservatively using Covariance Intersection, preserving estimator consistency under unknown cross-correlations. An information-driven allocator assigns targets and selects UAV hover viewpoints by trading expected uncertainty reduction in travel effort and safety separation. Implemented in ROS, the proposed system is validated in simulations and compared with representative tracking and fusion baselines, showing improved identity continuity and localization accuracy with modest communication overhead.
>
---
#### [replaced 053] COMRES-VLM: Coordinated Multi-Robot Exploration and Search using Vision Language Models
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文属于多机器人协同探索与搜索任务，解决传统方法协调不足的问题。通过引入视觉语言模型，提升机器人系统的智能协作与目标搜索效率。**

- **链接: [https://arxiv.org/pdf/2509.26324](https://arxiv.org/pdf/2509.26324)**

> **作者:** Ruiyang Wang; Hao-Lun Hsu; David Hunt; Jiwoo Kim; Shaocheng Luo; Miroslav Pajic
>
> **摘要:** Autonomous exploration and object search in unknown indoor environments remain challenging for multi-robot systems (MRS). Traditional approaches often rely on greedy frontier assignment strategies with limited inter-robot coordination. In this work, we present Coordinated Multi-Robot Exploration and Search using Vision Language Models (COMRES-VLM), a novel framework that leverages Vision Language Models (VLMs) for intelligent coordination of MRS tasked with efficient exploration and target object search. COMRES-VLM integrates real-time frontier cluster extraction and topological skeleton analysis with VLM reasoning over shared occupancy maps, robot states, and optional natural language priors, in order to generate globally consistent waypoint assignments. Extensive experiments in large-scale simulated indoor environments with up to six robots demonstrate that COMRES-VLM consistently outperforms state-of-the-art coordination methods, including Capacitated Vehicle Routing Problem (CVRP) and Voronoi-based planners, achieving 10.2\% faster exploration completion and 55.7\% higher object search efficiency. Notably, COMRES-VLM enables natural language-based object search capabilities, allowing human operators to provide high-level semantic guidance that traditional algorithms cannot interpret.
>
---
#### [replaced 054] Robust Differentiable Collision Detection for General Objects
- **分类: cs.RO**

- **简介: 本文提出一种鲁棒的可微碰撞检测框架，解决传统方法非可微导致优化受限的问题，支持凸凹物体，提升抓取等任务性能。**

- **链接: [https://arxiv.org/pdf/2511.06267](https://arxiv.org/pdf/2511.06267)**

> **作者:** Jiayi Chen; Wei Zhao; Liangwang Ruan; Baoquan Chen; He Wang
>
> **摘要:** Collision detection is a core component of robotics applications such as simulation, control, and planning. Traditional algorithms like GJK+EPA compute witness points (i.e., the closest or deepest-penetration pairs between two objects) but are inherently non-differentiable, preventing gradient flow and limiting gradient-based optimization in contact-rich tasks such as grasping and manipulation. Recent work introduced efficient first-order randomized smoothing to make witness points differentiable; however, their direction-based formulation is restricted to convex objects and lacks robustness for complex geometries. In this work, we propose a robust and efficient differentiable collision detection framework that supports both convex and concave objects across diverse scales and configurations. Our method introduces distance-based first-order randomized smoothing, adaptive sampling, and equivalent gradient transport for robust and informative gradient computation. Experiments on complex meshes from DexGraspNet and Objaverse show significant improvements over existing baselines. Finally, we demonstrate a direct application of our method for dexterous grasp synthesis to refine the grasp quality. The code is available at this https URL.
>
---
#### [replaced 055] A Quantitative Comparison of Centralised and Distributed Reinforcement Learning-Based Control for Soft Robotic Arms
- **分类: cs.RO**

- **简介: 该论文属于强化学习控制任务，比较集中式与分布式策略在软机械臂控制中的效果，解决不同控制结构性能差异问题，通过仿真实验评估多种场景下的表现。**

- **链接: [https://arxiv.org/pdf/2511.02192](https://arxiv.org/pdf/2511.02192)**

> **作者:** Linxin Hou; Qirui Wu; Zhihang Qin; Neil Banerjee; Yongxin Guo; Cecilia Laschi
>
> **备注:** 7 pages, 4 figures, 2 tables, accepted by RoboSoft 2026
>
> **摘要:** This paper presents a quantitative comparison between centralised and distributed multi-agent reinforcement learning (MARL) architectures for controlling a soft robotic arm modelled as a Cosserat rod in simulation. Using PyElastica and the OpenAI Gym interface, we train both a global Proximal Policy Optimisation (PPO) controller and a Multi-Agent PPO (MAPPO) under identical budgets. Both approaches are based on the arm having $n$ number of controlled sections. The study systematically varies $n$ and evaluates the performance of the arm to reach a fixed target in three scenarios: default baseline condition, recovery from external disturbance, and adaptation to actuator failure. Quantitative metrics used for the evaluation are mean action magnitude, mean final distance, mean episode length, and success rate. The results show that there are no significant benefits of the distributed policy when the number of controlled sections $n\le4$. In very simple systems, when $n\le2$, the centralised policy outperforms the distributed one. When $n$ increases to $4< n\le 12$, the distributed policy shows a high sample efficiency. In these systems, distributed policy promotes a stronger success rate, resilience, and robustness under local observability and yields faster convergence given the same sample size. However, centralised policies achieve much higher time efficiency during training as it takes much less time to train the same size of samples. These findings highlight the trade-offs between centralised and distributed policy in reinforcement learning-based control for soft robotic systems and provide actionable design guidance for future sim-to-real transfer in soft rod-like manipulators.
>
---
#### [replaced 056] SLAP: Shortcut Learning for Abstract Planning
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出SLAP方法，解决长周期决策中稀疏奖励和连续状态动作的问题。通过自动发现抽象动作，提升任务成功率和计划效率。属于强化学习与机器人任务规划领域。**

- **链接: [https://arxiv.org/pdf/2511.01107](https://arxiv.org/pdf/2511.01107)**

> **作者:** Y. Isabel Liu; Bowen Li; Benjamin Eysenbach; Tom Silver
>
> **备注:** Published at the International Conference on Learning Representations (ICLR) 2026. Code available at this https URL
>
> **摘要:** Long-horizon decision-making with sparse rewards and continuous states and actions remains a fundamental challenge in AI and robotics. Task and motion planning (TAMP) is a model-based framework that addresses this challenge by planning hierarchically with abstract actions (options). These options are manually defined, limiting the agent to behaviors that we as human engineers know how to program (pick, place, move). In this work, we propose Shortcut Learning for Abstract Planning (SLAP), a method that leverages existing TAMP options to automatically discover new ones. Our key idea is to use model-free reinforcement learning (RL) to learn shortcuts in the abstract planning graph induced by the existing options in TAMP. Without any additional assumptions or inputs, shortcut learning leads to shorter solutions than pure planning, and higher task success rates than flat and hierarchical RL. Qualitatively, SLAP discovers dynamic physical improvisations (e.g., slap, wiggle, wipe) that differ significantly from the manually-defined ones. In experiments in four simulated robotic environments, we show that SLAP solves and generalizes to a wide range of tasks, reducing overall plan lengths by over 50% and consistently outperforming planning and RL baselines.
>
---
#### [replaced 057] CAIMAN: Causal Action Influence Detection for Sample-efficient Loco-manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出CAIMAN框架，解决腿部机器人在非预抓取操作中的样本效率问题。通过因果动作影响机制，提升物体推动技能学习效果。**

- **链接: [https://arxiv.org/pdf/2502.00835](https://arxiv.org/pdf/2502.00835)**

> **作者:** Yuanchen Yuan; Jin Cheng; Núria Armengol Urpí; Stelian Coros
>
> **摘要:** Enabling legged robots to perform non-prehensile loco-manipulation is crucial for enhancing their versatility. Learning behaviors such as whole-body object pushing often requires sophisticated planning strategies or extensive task-specific reward shaping, especially in unstructured environments. In this work, we present CAIMAN, a practical reinforcement learning framework that encourages the agent to gain control over other entities in the environment. CAIMAN leverages causal action influence as an intrinsic motivation objective, allowing legged robots to efficiently acquire object pushing skills even under sparse task rewards. We employ a hierarchical control strategy, combining a low-level locomotion module with a high-level policy that generates task-relevant velocity commands and is trained to maximize the intrinsic reward. To estimate causal action influence, we learn the dynamics of the environment by integrating a kinematic prior with data collected during training. We empirically demonstrate CAIMAN's superior sample efficiency and adaptability to diverse scenarios in simulation, as well as its successful transfer to real-world systems without further fine-tuning. A video demo is available at this https URL.
>
---
#### [replaced 058] OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning
- **分类: cs.RO**

- **简介: 该论文提出OneTwoVLA，一个统一的视觉-语言-动作模型，解决机器人任务执行中推理与行动协同的问题。通过自适应切换模式提升任务规划、错误恢复等能力。**

- **链接: [https://arxiv.org/pdf/2505.11917](https://arxiv.org/pdf/2505.11917)**

> **作者:** Fanqi Lin; Ruiqian Nai; Yingdong Hu; Jiacheng You; Junming Zhao; Yang Gao
>
> **摘要:** General-purpose robots capable of performing diverse tasks require synergistic reasoning and acting capabilities. However, recent dual-system approaches, which separate high-level reasoning from low-level acting, often suffer from challenges such as limited mutual understanding of capabilities between systems and latency issues. This paper introduces OneTwoVLA, a single unified vision-language-action model that can perform both acting (System One) and reasoning (System Two). Crucially, OneTwoVLA adaptively switches between two modes: explicitly reasoning at critical moments during task execution, and generating actions based on the most recent reasoning at other times. To further unlock OneTwoVLA's reasoning and generalization capabilities, we design a scalable pipeline for synthesizing embodied reasoning-centric vision-language data, used for co-training with robot data. We validate OneTwoVLA's effectiveness through extensive experiments, highlighting its superior performance across four key capabilities: long-horizon task planning, error detection and recovery, natural human-robot interaction, and generalizable visual grounding, enabling the model to perform long-horizon, highly dexterous manipulation tasks such as making hotpot or mixing cocktails.
>
---
#### [replaced 059] ExtremControl: Low-Latency Humanoid Teleoperation with Direct Extremity Control
- **分类: cs.RO**

- **简介: 该论文属于人形机器人远程操作任务，旨在解决高延迟导致响应慢的问题。通过直接控制肢体姿态和速度前馈，实现低延迟的实时操作。**

- **链接: [https://arxiv.org/pdf/2602.11321](https://arxiv.org/pdf/2602.11321)**

> **作者:** Ziyan Xiong; Lixing Fang; Junyun Huang; Kashu Yamazaki; Hao Zhang; Chuang Gan
>
> **备注:** Project website: this https URL
>
> **摘要:** Building a low-latency humanoid teleoperation system is essential for collecting diverse reactive and dynamic demonstrations. However, existing approaches rely on heavily pre-processed human-to-humanoid motion retargeting and position-only PD control, resulting in substantial latency that severely limits responsiveness and prevents tasks requiring rapid feedback and fast reactions. To address this problem, we propose ExtremControl, a low latency whole-body control framework that: (1) operates directly on SE(3) poses of selected rigid links, primarily humanoid extremities, to avoid full-body retargeting; (2) utilizes a Cartesian-space mapping to directly convert human motion to humanoid link targets; and (3) incorporates velocity feedforward control at low level to support highly responsive behavior under rapidly changing control interfaces. We further provide a unified theoretical formulation of ExtremControl and systematically validate its effectiveness through experiments in both simulation and real-world environments. Building on ExtremControl, we implement a low-latency humanoid teleoperation system that supports both optical motion capture and VR-based motion tracking, achieving end-to-end latency as low as 50ms and enabling highly responsive behaviors such as ping-pong ball balancing, juggling, and real-time return, thereby substantially surpassing the 200ms latency limit observed in prior work.
>
---
#### [replaced 060] V-MORALS: Visual Morse Graph-Aided Estimation of Regions of Attraction in a Learned Latent Space
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人安全分析任务，旨在解决仅依赖传感器数据时无法准确估计吸引域的问题。工作是提出V-MORALS方法，通过学习潜在空间生成Morse图以计算吸引域。**

- **链接: [https://arxiv.org/pdf/2602.23524](https://arxiv.org/pdf/2602.23524)**

> **作者:** Faiz Aladin; Ashwin Balasubramanian; Lars Lindemann; Daniel Seita
>
> **摘要:** Reachability analysis has become increasingly important in robotics to distinguish safe from unsafe states. Unfortunately, existing reachability and safety analysis methods often fall short, as they typically require known system dynamics or large datasets to estimate accurate system models, are computationally expensive, and assume full state information. A recent method, called MORALS, aims to address these shortcomings by using topological tools to estimate Regions of Attraction (ROA) in a low-dimensional latent space. However, MORALS still relies on full state knowledge and has not been studied when only sensor measurements are available. This paper presents Visual Morse Graph-Aided Estimation of Regions of Attraction in a Learned Latent Space (V-MORALS). V-MORALS takes in a dataset of image-based trajectories of a system under a given controller, and learns a latent space for reachability analysis. Using this learned latent space, our method is able to generate well-defined Morse Graphs, from which we can compute ROAs for various systems and controllers. V-MORALS provides capabilities similar to the original MORALS architecture without relying on state knowledge, and using only high-level sensor data. Our project website is at: this https URL.
>
---
#### [replaced 061] ISS Policy : Scalable Diffusion Policy with Implicit Scene Supervision
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决视觉模仿学习中依赖外观、忽视3D结构的问题。提出ISS Policy，通过隐式场景监督提升策略性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.15020](https://arxiv.org/pdf/2512.15020)**

> **作者:** Wenlong Xia; Jinhao Zhang; Ce Zhang; Yaojia Wang; Huizhe Li; Youmin Gong; Jie Mei
>
> **摘要:** Vision-based imitation learning has enabled impressive robotic manipulation skills, but its reliance on object appearance while ignoring the underlying 3D scene structure leads to low training efficiency and poor generalization. To address these challenges, we introduce \emph{Implicit Scene Supervision (ISS) Policy}, a 3D visuomotor DiT-based diffusion policy that predicts sequences of continuous actions from point cloud observations. We extend DiT with a novel implicit scene supervision module that encourages the model to produce outputs consistent with the scene's geometric evolution, thereby improving the performance and robustness of the policy. Notably, ISS Policy achieves state-of-the-art performance on both single-arm manipulation tasks (MetaWorld) and dexterous hand manipulation (Adroit). In real-world experiments, it also demonstrates strong generalization and robustness. Additional ablation studies show that our method scales effectively with both data and parameters. Code and videos will be released.
>
---
#### [replaced 062] Viability-Preserving Passive Torque Control
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决传统控制器在外部扰动下可能引发安全问题的问题。通过可行性理论预计算安全状态集，并结合二次规划框架确保机器人始终处于安全状态。**

- **链接: [https://arxiv.org/pdf/2510.03367](https://arxiv.org/pdf/2510.03367)**

> **作者:** Zizhe Zhang; Yicong Wang; Zhiquan Zhang; Tianyu Li; Nadia Figueroa
>
> **备注:** 8 pages, 7 figures, Project Website: this https URL
>
> **摘要:** Conventional passivity-based torque controllers for manipulators are typically unconstrained, which can lead to safety violations under external perturbations. In this paper, we employ viability theory to pre-compute safe sets in the state-space of joint positions and velocities. These viable sets, constructed via data-driven and analytical methods for self-collision avoidance, external object collision avoidance and joint-position and joint-velocity limits, provide constraints on joint accelerations and thus joint torques via the robot dynamics. A quadratic programming-based control framework enforces these constraints on a passive controller tracking a dynamical system, ensuring the robot states remain within the safe set in an infinite time horizon. We validate the proposed approach through simulations and hardware experiments on a 7-DoF Franka Emika manipulator. In comparison to a baseline constrained passive controller, our method operates at higher control-loop rates and yields smoother trajectories.
>
---
