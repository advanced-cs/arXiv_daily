# 机器人 cs.RO

- **最新发布 26 篇**

- **更新 25 篇**

## 最新发布

#### [new 001] Robotic capabilities framework: A boundary object and intermediate-level knowledge artifact for co-designing robotic processes
- **分类: cs.RO**

- **简介: 该论文提出“机器人能力框架”，旨在解决人机协作设计中跨学科知识脱节问题。通过构建一个中间层知识工具，促进技术与实践者对话，明确任务分工，支持协同设计，推动更公平、高效的未来工作模式。**

- **链接: [https://arxiv.org/pdf/2512.02549v1](https://arxiv.org/pdf/2512.02549v1)**

> **作者:** Alessandro Ianniello; Dave Murray-Rust; Sara Muscolo; Olger Siebinga; Nicky Mol; Denis Zatyagov; Eva Verhoef; Deborah Forster; David Abbink
>
> **摘要:** As robots become more adaptable, responsive, and capable of interacting with humans, the design of effective human-robot collaboration becomes critical. Yet, this design process is typically led by monodisciplinary approaches, often overlooking interdisciplinary knowledge and the experiential knowledge of workers who will ultimately share tasks with these systems. To address this gap, we introduce the robotic capabilities framework, a vocabulary that enables transdisciplinary collaborations to meaningfully shape the future of work when robotic systems are integrated into the workplace. Rather than focusing on the internal workings of robots, the framework centers discussion on high-level capabilities, supporting dialogue around which elements of a task should remain human-led and which can be delegated to robots. We developed the framework through reflexive and iterative processes, and applied it in two distinct settings: by engaging roboticists in describing existing commercial robots using its vocabulary, and through a design activity with students working on robotics-related projects. The framework emerges as an intermediate-level knowledge artifact and a boundary object that bridges technical and experiential domains, guiding designers, empowering workers, and contributing to more just and collaborative futures of work.
>
---
#### [new 002] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中失败诊断与学习难题，提出ViFailback框架，通过视觉符号提升故障诊断效率。构建了包含58,126个VQA对的真实世界操作数据集，并建立细粒度评估基准ViFailback-Bench。基于此，训练出ViFailback-8B模型，可生成可视化纠正指导，实现在真实场景中辅助机器人从失败中恢复。**

- **链接: [https://arxiv.org/pdf/2512.02787v1](https://arxiv.org/pdf/2512.02787v1)**

> **作者:** Xianchao Zeng; Xinyu Zhou; Youcheng Li; Jiayou Shi; Tianle Li; Liangming Chen; Lei Ren; Yong-Lu Li
>
> **摘要:** Vision-Language-Action (VLA) models have recently achieved remarkable progress in robotic manipulation, yet they remain limited in failure diagnosis and learning from failures. Additionally, existing failure datasets are mostly generated programmatically in simulation, which limits their generalization to the real world. In light of these, we introduce ViFailback, a framework designed to diagnose robotic manipulation failures and provide both textual and visual correction guidance. Our framework utilizes explicit visual symbols to enhance annotation efficiency. We further release the ViFailback dataset, a large-scale collection of 58,126 Visual Question Answering (VQA) pairs along with their corresponding 5,202 real-world manipulation trajectories. Based on the dataset, we establish ViFailback-Bench, a benchmark of 11 fine-grained VQA tasks designed to assess the failure diagnosis and correction abilities of Vision-Language Models (VLMs), featuring ViFailback-Bench Lite for closed-ended and ViFailback-Bench Hard for open-ended evaluation. To demonstrate the effectiveness of our framework, we built the ViFailback-8B VLM, which not only achieves significant overall performance improvement on ViFailback-Bench but also generates visual symbols for corrective action guidance. Finally, by integrating ViFailback-8B with a VLA model, we conduct real-world robotic experiments demonstrating its ability to assist the VLA model in recovering from failures. Project Website: https://x1nyuzhou.github.io/vifailback.github.io/
>
---
#### [new 003] Experimental Characterization of Fingertip Trajectory following for a 3-DoF Series-Parallel Hybrid Robotic Finger
- **分类: cs.RO**

- **简介: 该论文研究紧凑型多自由度机器人手指的任务空间轨迹跟踪问题。针对现有研究中精密轨迹控制实验稀缺的挑战，设计并实验验证了一种3-自由度串联-并联混合机械结构，实现毫米级精度的指尖轨迹跟踪，为灵巧操作提供重要基准。**

- **链接: [https://arxiv.org/pdf/2512.02951v1](https://arxiv.org/pdf/2512.02951v1)**

> **作者:** Nicholas Baiata; Nilanjan Chakraborty
>
> **摘要:** Task-space control of robotic fingers is a critical enabler of dexterous manipulation, as manipulation objectives are most naturally specified in terms of fingertip motions and applied forces rather than individual joint angles. While task-space planning and control have been extensively studied for larger, arm-scale manipulators, demonstrations of precise task-space trajectory tracking in compact, multi-DoF robotic fingers remain scarce. In this paper, we present the physical prototyping and experimental characterization of a three-degree-of-freedom, linkage-driven, series-parallel robotic finger with analytic forward kinematics and a closed-form Jacobian. A resolved motion rate control (RMRC) scheme is implemented to achieve closed-loop task-space trajectory tracking. We experimentally evaluate the fingertip tracking performance across a variety of trajectories, including straight lines, circles, and more complex curves, and report millimeter-level accuracy. To the best of our knowledge, this work provides one of the first systematic experimental demonstrations of precise task-space trajectory tracking in a linkage-driven robotic finger, thereby establishing a benchmark for future designs aimed at dexterous in-hand manipulation.
>
---
#### [new 004] Phase-Adaptive LLM Framework with Multi-Stage Validation for Construction Robot Task Allocation: A Systematic Benchmark Against Traditional Optimization Algorithms
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究构建机器人任务分配问题，针对传统优化算法在复杂场景中灵活性不足的问题，提出基于LLM的相位自适应框架LTAA。通过多阶段验证与动态提示，实现高效任务分配，显著降低计算开销，并在真实数据集上优于传统方法，兼具可解释性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.02810v1](https://arxiv.org/pdf/2512.02810v1)**

> **作者:** Shyam prasad reddy Kaitha; Hongrui Yu
>
> **摘要:** Multi-robot task allocation in construction automation has traditionally relied on optimization methods such as Dynamic Programming and Reinforcement Learning. This research introduces the LangGraph-based Task Allocation Agent (LTAA), an LLM-driven framework that integrates phase-adaptive allocation strategies, multi-stage validation with hierarchical retries, and dynamic prompting for efficient robot coordination. Although recent LLM approaches show potential for construction robotics, they largely lack rigorous validation and benchmarking against established algorithms. This paper presents the first systematic comparison of LLM-based task allocation with traditional methods in construction scenarios.The study validates LLM feasibility through SMART-LLM replication and addresses implementation challenges using a Self-Corrective Agent Architecture. LTAA leverages natural-language reasoning combined with structured validation mechanisms, achieving major computational gains reducing token usage by 94.6% and allocation time by 86% through dynamic prompting. The framework adjusts its strategy across phases: emphasizing execution feasibility early and workload balance in later allocations.The authors evaluate LTAA against Dynamic Programming, Q-learning, and Deep Q-Network (DQN) baselines using construction operations from the TEACh human-robot collaboration dataset. In the Heavy Excels setting, where robots have strong task specializations, LTAA achieves 77% task completion with superior workload balance, outperforming all traditional methods. These findings show that LLM-based reasoning with structured validation can match established optimization algorithms while offering additional advantages such as interpretability, adaptability, and the ability to update task logic without retraining.
>
---
#### [new 005] Video2Act: A Dual-System Video Diffusion Policy with Robotic Spatio-Motional Modeling
- **分类: cs.RO**

- **简介: 该论文针对机器人视觉-语言-动作（VLA）任务，解决视频扩散模型中运动信息利用不充分的问题。提出Video2Act框架，通过提取前景边界与帧间运动变化，作为条件输入扩散变压器动作头，实现空间与运动感知的协同决策。采用异步双系统设计提升推理效率，显著提升模拟与真实场景下的成功率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.03044v1](https://arxiv.org/pdf/2512.03044v1)**

> **作者:** Yueru Jia; Jiaming Liu; Shengbang Liu; Rui Zhou; Wanhe Yu; Yuyang Yan; Xiaowei Chi; Yandong Guo; Boxin Shi; Shanghang Zhang
>
> **摘要:** Robust perception and dynamics modeling are fundamental to real-world robotic policy learning. Recent methods employ video diffusion models (VDMs) to enhance robotic policies, improving their understanding and modeling of the physical world. However, existing approaches overlook the coherent and physically consistent motion representations inherently encoded across frames in VDMs. To this end, we propose Video2Act, a framework that efficiently guides robotic action learning by explicitly integrating spatial and motion-aware representations. Building on the inherent representations of VDMs, we extract foreground boundaries and inter-frame motion variations while filtering out background noise and task-irrelevant biases. These refined representations are then used as additional conditioning inputs to a diffusion transformer (DiT) action head, enabling it to reason about what to manipulate and how to move. To mitigate inference inefficiency, we propose an asynchronous dual-system design, where the VDM functions as the slow System 2 and the DiT head as the fast System 1, working collaboratively to generate adaptive actions. By providing motion-aware conditions to System 1, Video2Act maintains stable manipulation even with low-frequency updates from the VDM. For evaluation, Video2Act surpasses previous state-of-the-art VLA methods by 7.7% in simulation and 21.7% in real-world tasks in terms of average success rate, further exhibiting strong generalization capabilities.
>
---
#### [new 006] Steering Vision-Language-Action Models as Anti-Exploration: A Test-Time Scaling Approach
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在下游任务中因预训练数据冗余导致的推理不稳定性问题，提出TACO框架。通过测试时缩放与轻量伪计数验证，引导模型选择高置信度动作，抑制分布偏移，提升推理稳定性和成功率，无需梯度更新，适用于扩散类VLA模型。**

- **链接: [https://arxiv.org/pdf/2512.02834v1](https://arxiv.org/pdf/2512.02834v1)**

> **作者:** Siyuan Yang; Yang Zhang; Haoran He; Ling Pan; Xiu Li; Chenjia Bai; Xuelong Li
>
> **备注:** The first two authors contributed equally. Yang Zhang leads the whole project
>
> **摘要:** Vision-Language-Action (VLA) models, trained via flow-matching or diffusion objectives, excel at learning complex behaviors from large-scale, multi-modal datasets (e.g., human teleoperation, scripted policies). However, since VLAs incorporate diverse data modes in the pre-training stage, and the finetuning dataset often contains demonstration data collected in a kinematically suboptimal or undesirable way, it exists redundant action modes that are irrelevant to the success action modes of the downstream task. Specifically, we observe a critical inference-time fragility among various sampled noises after supervised finetuning of pre-trained VLAs. In this paper, we attribute this instability to the distribution shift between the VLA policy and the policy induced by stable success modes of the downstream task dataset. Thus, we propose \textbf{TACO}, a test-time-scaling (TTS) framework that applies a lightweight pseudo-count estimator as a high-fidelity verifier of action chunks. The VLA models integrated with TACO can execute the actions with maximum pseudo-count from all sampled action chunks, thereby preventing distribution shifts while preserving the generalization ability of VLAs since the constraint is applied only during inference. Our method resembles the classical anti-exploration principle in offline reinforcement learning (RL), and being gradient-free, it incurs significant computational benefits compared to RL update, especially for flow or diffusion-based VLAs which are difficult to perform RL update due to denoising process. Extensive experiments across four simulation benchmarks (RoboTwin2.0, Robotwin, LIBERO, SimplerEnv) and a dual-arm platform demonstrate that our method significantly improves the inference stability and success rates in downstream-task adaptations.
>
---
#### [new 007] VLA Models Are More Generalizable Than You Think: Revisiting Physical and Spatial Modeling
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究视觉-语言-动作（VLA）模型的泛化能力，针对其在新视角下性能下降的问题，指出根源在于空间建模错位。提出轻量级一 shot 适配框架：FTM通过全局仿射变换调整视觉特征，FLA采用低秩更新优化ViT编码器，显著提升视角鲁棒性，仅用少量参数即达接近全量微调效果。**

- **链接: [https://arxiv.org/pdf/2512.02902v1](https://arxiv.org/pdf/2512.02902v1)**

> **作者:** Weiqi Li; Quande Zhang; Ruifeng Zhai; Liang Lin; Guangrun Wang
>
> **摘要:** Vision-language-action (VLA) models achieve strong in-distribution performance but degrade sharply under novel camera viewpoints and visual perturbations. We show that this brittleness primarily arises from misalignment in Spatial Modeling, rather than Physical Modeling. To address this, we propose a one-shot adaptation framework that recalibrates visual representations through lightweight, learnable updates. Our first method, Feature Token Modulation (FTM), applies a global affine transformation to visual tokens and improves Libero viewpoint accuracy from 48.5% to 87.1% with only 4K parameters. Building on this, Feature Linear Adaptation (FLA) introduces low-rank updates to the ViT encoder, achieving 90.8% success with 4.7M parameters -- matching LoRA-scale finetuning at far lower cost. Together, these results reveal substantial untapped robustness in pretrained VLA models and demonstrate that targeted, minimal visual adaptation is sufficient to restore viewpoint generalization.
>
---
#### [new 008] VIGS-SLAM: Visual Inertial Gaussian Splatting SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出VIGS-SLAM，一种视觉惯性3D高斯点云SLAM系统，解决纯视觉方法在运动模糊、低纹理等条件下性能下降的问题。通过融合视觉与惯性信息，在统一优化框架中联合估计相机位姿、深度和IMU状态，实现鲁棒实时跟踪与高质量重建。**

- **链接: [https://arxiv.org/pdf/2512.02293v1](https://arxiv.org/pdf/2512.02293v1)**

> **作者:** Zihan Zhu; Wei Zhang; Norbert Haala; Marc Pollefeys; Daniel Barath
>
> **备注:** Project page: https://vigs-slam.github.io
>
> **摘要:** We present VIGS-SLAM, a visual-inertial 3D Gaussian Splatting SLAM system that achieves robust real-time tracking and high-fidelity reconstruction. Although recent 3DGS-based SLAM methods achieve dense and photorealistic mapping, their purely visual design degrades under motion blur, low texture, and exposure variations. Our method tightly couples visual and inertial cues within a unified optimization framework, jointly refining camera poses, depths, and IMU states. It features robust IMU initialization, time-varying bias modeling, and loop closure with consistent Gaussian updates. Experiments on four challenging datasets demonstrate our superiority over state-of-the-art methods. Project page: https://vigs-slam.github.io
>
---
#### [new 009] SwarmDiffusion: End-To-End Traversability-Guided Diffusion for Embodiment-Agnostic Navigation of Heterogeneous Robots
- **分类: cs.RO**

- **简介: 该论文针对异构机器人自主导航中视觉可通行性估计与轨迹生成分离、依赖提示工程及泛化差的问题，提出SwarmDiffusion模型。通过无提示的端到端扩散框架，联合预测可通行性并生成可行轨迹，采用随机采样与贝塞尔平滑构建无规划器轨迹，实现跨平台迁移与快速推理，显著提升导航成功率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.02851v1](https://arxiv.org/pdf/2512.02851v1)**

> **作者:** Iana Zhura; Sausar Karaf; Faryal Batool; Nipun Dhananjaya Weerakkodi Mudalige; Valerii Serpiva; Ali Alridha Abdulkarim; Aleksey Fedoseev; Didar Seyidov; Amjad Hajira; Dzmitry Tsetserukou
>
> **备注:** This work has been submitted for publication and is currently under review
>
> **摘要:** Visual traversability estimation is critical for autonomous navigation, but existing VLM-based methods rely on hand-crafted prompts, generalize poorly across embodiments, and output only traversability maps, leaving trajectory generation to slow external planners. We propose SwarmDiffusion, a lightweight end-to-end diffusion model that jointly predicts traversability and generates a feasible trajectory from a single RGB image. To remove the need for annotated or planner-produced paths, we introduce a planner-free trajectory construction pipeline based on randomized waypoint sampling, Bezier smoothing, and regularization enforcing connectivity, safety, directionality, and path thinness. This enables learning stable motion priors without demonstrations. SwarmDiffusion leverages VLM-derived supervision without prompt engineering and conditions the diffusion process on a compact embodiment state, producing physically consistent, traversable paths that transfer across different robot platforms. Across indoor environments and two embodiments (quadruped and aerial), the method achieves 80-100\% navigation success and 0.09 s inference, and adapts to a new robot using only-500 additional visual samples. It generalizes reliably to unseen environments in simulation and real-world trials, offering a scalable, prompt-free approach to unified traversability reasoning and trajectory generation.
>
---
#### [new 010] Reinforcement Learning for Robotic Safe Control with Force Sensing
- **分类: cs.RO**

- **简介: 该论文研究机器人在非结构化环境中安全操控任务，针对传统方法与强化学习在稳定性、可靠性及仿真到现实迁移中的安全隐患，引入力觉与触觉感知，提升策略适应性与安全性。实验表明，该方法在推物任务中更高效可靠，适用于多种实际应用。**

- **链接: [https://arxiv.org/pdf/2512.02022v1](https://arxiv.org/pdf/2512.02022v1)**

> **作者:** Nan Lin; Linrui Zhang; Yuxuan Chen; Zhenrui Chen; Yujun Zhu; Ruoxi Chen; Peichen Wu; Xiaoping Chen
>
> **摘要:** For the task with complicated manipulation in unstructured environments, traditional hand-coded methods are ineffective, while reinforcement learning can provide more general and useful policy. Although the reinforcement learning is able to obtain impressive results, its stability and reliability is hard to guarantee, which would cause the potential safety threats. Besides, the transfer from simulation to real world also will lead in unpredictable situations. To enhance the safety and reliability of robots, we introduce the force and haptic perception into reinforcement learning. Force and tactual sensation play key roles in robotic dynamic control and human-robot interaction. We demonstrate that the force-based reinforcement learning method can be more adaptive to environment, especially in sim-to-real transfer. Experimental results show in object pushing task, our strategy is safer and more efficient in both simulation and real world, thus it holds prospects for a wide variety of robotic applications.
>
---
#### [new 011] AID: Agent Intent from Diffusion for Multi-Agent Informative Path Planning
- **分类: cs.RO**

- **简介: 该论文针对多智能体信息路径规划（MAIPP）任务，解决环境信念动态演化下的协同效率问题。提出AID框架，基于扩散模型实现非自回归的长期轨迹生成，通过两阶段训练继承专家策略并优化协作，显著提升信息获取效率与执行速度，支持大规模多智能体扩展。**

- **链接: [https://arxiv.org/pdf/2512.02535v1](https://arxiv.org/pdf/2512.02535v1)**

> **作者:** Jeric Lew; Yuhong Cao; Derek Ming Siang Tan; Guillaume Sartoretti
>
> **摘要:** Information gathering in large-scale or time-critical scenarios (e.g., environmental monitoring, search and rescue) requires broad coverage within limited time budgets, motivating the use of multi-agent systems. These scenarios are commonly formulated as multi-agent informative path planning (MAIPP), where multiple agents must coordinate to maximize information gain while operating under budget constraints. A central challenge in MAIPP is ensuring effective coordination while the belief over the environment evolves with incoming measurements. Recent learning-based approaches address this by using distributions over future positions as "intent" to support coordination. However, these autoregressive intent predictors are computationally expensive and prone to compounding errors. Inspired by the effectiveness of diffusion models as expressive, long-horizon policies, we propose AID, a fully decentralized MAIPP framework that leverages diffusion models to generate long-term trajectories in a non-autoregressive manner. AID first performs behavior cloning on trajectories produced by existing MAIPP planners and then fine-tunes the policy using reinforcement learning via Diffusion Policy Policy Optimization (DPPO). This two-stage pipeline enables the policy to inherit expert behavior while learning improved coordination through online reward feedback. Experiments demonstrate that AID consistently improves upon the MAIPP planners it is trained from, achieving up to 4x faster execution and 17% increased information gain, while scaling effectively to larger numbers of agents. Our implementation is publicly available at https://github.com/marmotlab/AID.
>
---
#### [new 012] SAM2Grasp: Resolve Multi-modal Grasping via Prompt-conditioned Temporal Action Prediction
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人抓取中的多模态问题，提出SAM2Grasp框架。通过将任务重构为提示条件下的时序动作预测，利用SAM2的视觉时序追踪能力，仅训练轻量级动作头，实现对指定目标的稳定、唯一抓取轨迹预测，显著提升复杂场景下多物体抓取性能。**

- **链接: [https://arxiv.org/pdf/2512.02609v1](https://arxiv.org/pdf/2512.02609v1)**

> **作者:** Shengkai Wu; Jinrong Yang; Wenqiu Luo; Linfeng Gao; Chaohui Shang; Meiyu Zhi; Mingshan Sun; Fangping Yang; Liangliang Ren; Yong Zhao
>
> **摘要:** Imitation learning for robotic grasping is often plagued by the multimodal problem: when a scene contains multiple valid targets, demonstrations of grasping different objects create conflicting training signals. Standard imitation learning policies fail by averaging these distinct actions into a single, invalid action. In this paper, we introduce SAM2Grasp, a novel framework that resolves this issue by reformulating the task as a uni-modal, prompt-conditioned prediction problem. Our method leverages the frozen SAM2 model to use its powerful visual temporal tracking capability and introduces a lightweight, trainable action head that operates in parallel with its native segmentation head. This design allows for training only the small action head on pre-computed temporal-visual features from SAM2. During inference, an initial prompt, such as a bounding box provided by an upstream object detection model, designates the specific object to be grasped. This prompt conditions the action head to predict a unique, unambiguous grasp trajectory for that object alone. In all subsequent video frames, SAM2's built-in temporal tracking capability automatically maintains stable tracking of the selected object, enabling our model to continuously predict the grasp trajectory from the video stream without further external guidance. This temporal-prompted approach effectively eliminates ambiguity from the visuomotor policy. We demonstrate through extensive experiments that SAM2Grasp achieves state-of-the-art performance in cluttered, multi-object grasping tasks.
>
---
#### [new 013] CogDrive: Cognition-Driven Multimodal Prediction-Planning Fusion for Safe Autonomy
- **分类: cs.RO; cs.MA**

- **简介: 该论文针对复杂交通中安全自主驾驶问题，提出CogDrive框架，融合认知驱动的多模态预测与安全导向规划。通过拓扑运动语义与关系编码建模交互模式，实现稀疏、不平衡行为的精准预测；结合应急响应机制优化安全轨迹，提升长时预测与动态重规划能力。在Argoverse2和INTERACTION数据集上验证了其高精度与可靠性。**

- **链接: [https://arxiv.org/pdf/2512.02777v1](https://arxiv.org/pdf/2512.02777v1)**

> **作者:** Heye Huang; Yibin Yang; Mingfeng Fan; Haoran Wang; Xiaocong Zhao; Jianqiang Wang
>
> **备注:** 25 pages, 6 figures
>
> **摘要:** Safe autonomous driving in mixed traffic requires a unified understanding of multimodal interactions and dynamic planning under uncertainty. Existing learning based approaches struggle to capture rare but safety critical behaviors, while rule based systems often lack adaptability in complex interactions. To address these limitations, CogDrive introduces a cognition driven multimodal prediction and planning framework that integrates explicit modal reasoning with safety aware trajectory optimization. The prediction module adopts cognitive representations of interaction modes based on topological motion semantics and nearest neighbor relational encoding. With a differentiable modal loss and multimodal Gaussian decoding, CogDrive learns sparse and unbalanced interaction behaviors and improves long horizon trajectory prediction. The planning module incorporates an emergency response concept and optimizes safety stabilized trajectories, where short term consistent branches ensure safety during replanning cycles and long term branches support smooth and collision free motion under low probability switching modes. Experiments on Argoverse2 and INTERACTION datasets show that CogDrive achieves strong performance in trajectory accuracy and miss rate, while closed loop simulations confirm adaptive behavior in merge and intersection scenarios. By combining cognitive multimodal prediction with safety oriented planning, CogDrive offers an interpretable and reliable paradigm for safe autonomy in complex traffic.
>
---
#### [new 014] VLM as Strategist: Adaptive Generation of Safety-critical Testing Scenarios via Guided Diffusion
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对自动驾驶系统测试中安全关键场景稀缺且难以生成的问题，提出融合视觉语言模型（VLM）与自适应引导扩散模型的框架。通过三层架构实现从目标设定到动态场景生成的闭环控制，高效生成高保真、交互性强的安全关键测试场景。**

- **链接: [https://arxiv.org/pdf/2512.02844v1](https://arxiv.org/pdf/2512.02844v1)**

> **作者:** Xinzheng Wu; Junyi Chen; Naiting Zhong; Yong Shen
>
> **备注:** 25 pages, 9 figures
>
> **摘要:** The safe deployment of autonomous driving systems (ADSs) relies on comprehensive testing and evaluation. However, safety-critical scenarios that can effectively expose system vulnerabilities are extremely sparse in the real world. Existing scenario generation methods face challenges in efficiently constructing long-tail scenarios that ensure fidelity, criticality, and interactivity, while particularly lacking real-time dynamic response capabilities to the vehicle under test (VUT). To address these challenges, this paper proposes a safety-critical testing scenario generation framework that integrates the high-level semantic understanding capabilities of Vision Language Models (VLMs) with the fine-grained generation capabilities of adaptive guided diffusion models. The framework establishes a three-layer hierarchical architecture comprising a strategic layer for VLM-directed scenario generation objective determination, a tactical layer for guidance function formulation, and an operational layer for guided diffusion execution. We first establish a high-quality fundamental diffusion model that learns the data distribution of real driving scenarios. Next, we design an adaptive guided diffusion method that enables real-time, precise control of background vehicles (BVs) in closed-loop simulation. The VLM is then incorporated to autonomously generate scenario generation objectives and guidance functions through deep scenario understanding and risk reasoning, ultimately guiding the diffusion model to achieve VLM-directed scenario generation. Experimental results demonstrate that the proposed method can efficiently generate realistic, diverse, and highly interactive safety-critical testing scenarios. Furthermore, case studies validate the adaptability and VLM-directed generation performance of the proposed method.
>
---
#### [new 015] RoboWheel: A Data Engine from Real-World Human Demonstrations for Cross-Embodiment Robotic Learning
- **分类: cs.RO**

- **简介: 该论文提出RoboWheel，一个从真人手物交互视频生成跨形态机器人学习数据的引擎。针对真实世界动作数据难以通用的问题，通过高精度重建接触丰富的轨迹并物理约束优化，实现单目视觉输入到多机器人形态的灵活重定向，构建可扩展的仿真增强数据集，验证了HOI数据在模仿学习中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.02729v1](https://arxiv.org/pdf/2512.02729v1)**

> **作者:** Yuhong Zhang; Zihan Gao; Shengpeng Li; Ling-Hao Chen; Kaisheng Liu; Runqing Cheng; Xiao Lin; Junjia Liu; Zhuoheng Li; Jingyi Feng; Ziyan He; Jintian Lin; Zheyan Huang; Zhifang Liu; Haoqian Wang
>
> **备注:** 27 Pages, 21 figures
>
> **摘要:** We introduce Robowheel, a data engine that converts human hand object interaction (HOI) videos into training-ready supervision for cross morphology robotic learning. From monocular RGB or RGB-D inputs, we perform high precision HOI reconstruction and enforce physical plausibility via a reinforcement learning (RL) optimizer that refines hand object relative poses under contact and penetration constraints. The reconstructed, contact rich trajectories are then retargeted to cross-embodiments, robot arms with simple end effectors, dexterous hands, and humanoids, yielding executable actions and rollouts. To scale coverage, we build a simulation-augmented framework on Isaac Sim with diverse domain randomization (embodiments, trajectories, object retrieval, background textures, hand motion mirroring), which enriches the distributions of trajectories and observations while preserving spatial relationships and physical plausibility. The entire data pipeline forms an end to end pipeline from video,reconstruction,retargeting,augmentation data acquisition. We validate the data on mainstream vision language action (VLA) and imitation learning architectures, demonstrating that trajectories produced by our pipeline are as stable as those from teleoperation and yield comparable continual performance gains. To our knowledge, this provides the first quantitative evidence that HOI modalities can serve as effective supervision for robotic learning. Compared with teleoperation, Robowheel is lightweight, a single monocular RGB(D) camera is sufficient to extract a universal, embodiment agnostic motion representation that could be flexibly retargeted across embodiments. We further assemble a large scale multimodal dataset combining multi-camera captures, monocular videos, and public HOI corpora for training and evaluating embodied models.
>
---
#### [new 016] Vehicle Dynamics Embedded World Models for Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对自动驾驶中世界模型对车辆动态变化鲁棒性不足的问题，提出VDD方法，通过解耦自车动力学与环境动态建模，提升跨车型泛化能力。引入部署时策略调整（PAD）和训练时策略增强（PAT），显著改善驾驶性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.02417v1](https://arxiv.org/pdf/2512.02417v1)**

> **作者:** Huiqian Li; Wei Pan; Haodong Zhang; Jin Huang; Zhihua Zhong
>
> **摘要:** World models have gained significant attention as a promising approach for autonomous driving. By emulating human-like perception and decision-making processes, these models can predict and adapt to dynamic environments. Existing methods typically map high-dimensional observations into compact latent spaces and learn optimal policies within these latent representations. However, prior work usually jointly learns ego-vehicle dynamics and environmental transition dynamics from the image input, leading to inefficiencies and a lack of robustness to variations in vehicle dynamics. To address these issues, we propose the Vehicle Dynamics embedded Dreamer (VDD) method, which decouples the modeling of ego-vehicle dynamics from environmental transition dynamics. This separation allows the world model to generalize effectively across vehicles with diverse parameters. Additionally, we introduce two strategies to further enhance the robustness of the learned policy: Policy Adjustment during Deployment (PAD) and Policy Augmentation during Training (PAT). Comprehensive experiments in simulated environments demonstrate that the proposed model significantly improves both driving performance and robustness to variations in vehicle dynamics, outperforming existing approaches.
>
---
#### [new 017] Robust Geospatial Coordination of Multi-Agent Communications Networks Under Attrition
- **分类: cs.RO; cs.MA; eess.SY**

- **简介: 该论文针对应急响应中无人机通信网络易受损的问题，提出鲁棒抗毁任务组网（RTNUA）任务。为解决多无人机网络在高毁伤环境下的连通性维持难题，提出ΦIREMAN算法，利用物理启发势场实现主动冗余与故障恢复，显著提升网络可靠性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.02079v1](https://arxiv.org/pdf/2512.02079v1)**

> **作者:** Jonathan S. Kent; Eliana Stefani; Brian K. Plancher
>
> **备注:** 8 pages, 5 figures, 4 tables, submitted to IEEE RA-L
>
> **摘要:** Fast, efficient, robust communication during wildfire and other emergency responses is critical. One way to achieve this is by coordinating swarms of autonomous aerial vehicles carrying communications equipment to form an ad-hoc network connecting emergency response personnel to both each other and central command. However, operating in such extreme environments may lead to individual networking agents being damaged or rendered inoperable, which could bring down the network and interrupt communications. To overcome this challenge and enable multi-agent UAV networking in difficult environments, this paper introduces and formalizes the problem of Robust Task Networking Under Attrition (RTNUA), which extends connectivity maintenance in multi-robot systems to explicitly address proactive redundancy and attrition recovery. We introduce Physics-Informed Robust Employment of Multi-Agent Networks ($Φ$IREMAN), a topological algorithm leveraging physics-inspired potential fields to solve this problem. Through simulation across 25 problem configurations, $Φ$IREMAN consistently outperforms the DCCRS baseline, and on large-scale problems with up to 100 tasks and 500 drones, maintains $>99.9\%$ task uptime despite substantial attrition, demonstrating both effectiveness and scalability.
>
---
#### [new 018] SMP: Reusable Score-Matching Motion Priors for Physics-Based Character Control
- **分类: cs.GR; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出SMP，一种可重用的基于评分匹配的运动先验方法，用于物理驱动角色控制。针对传统对抗性运动先验需为每种控制器重新训练、难以复用的问题，SMP利用预训练的运动扩散模型与评分蒸馏采样，构建任务无关的通用奖励函数，支持跨任务复用与风格组合，实现高质量自然运动生成。**

- **链接: [https://arxiv.org/pdf/2512.03028v1](https://arxiv.org/pdf/2512.03028v1)**

> **作者:** Yuxuan Mu; Ziyu Zhang; Yi Shi; Minami Matsumoto; Kotaro Imamura; Guy Tevet; Chuan Guo; Michael Taylor; Chang Shu; Pengcheng Xi; Xue Bin Peng
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** Data-driven motion priors that can guide agents toward producing naturalistic behaviors play a pivotal role in creating life-like virtual characters. Adversarial imitation learning has been a highly effective method for learning motion priors from reference motion data. However, adversarial priors, with few exceptions, need to be retrained for each new controller, thereby limiting their reusability and necessitating the retention of the reference motion data when training on downstream tasks. In this work, we present Score-Matching Motion Priors (SMP), which leverages pre-trained motion diffusion models and score distillation sampling (SDS) to create reusable task-agnostic motion priors. SMPs can be pre-trained on a motion dataset, independent of any control policy or task. Once trained, SMPs can be kept frozen and reused as general-purpose reward functions to train policies to produce naturalistic behaviors for downstream tasks. We show that a general motion prior trained on large-scale datasets can be repurposed into a variety of style-specific priors. Furthermore SMP can compose different styles to synthesize new styles not present in the original dataset. Our method produces high-quality motion comparable to state-of-the-art adversarial imitation learning methods through reusable and modular motion priors. We demonstrate the effectiveness of SMP across a diverse suite of control tasks with physically simulated humanoid characters. Video demo available at https://youtu.be/ravlZJteS20
>
---
#### [new 019] SurfFill: Completion of LiDAR Point Clouds via Gaussian Surfel Splatting
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出SurfFill，一种基于高斯表面元的LiDAR点云补全方法。针对LiDAR在细结构和暗材质处漏采的问题，利用点云密度变化识别模糊区域，结合高斯表面元优化，在局部区域生成补充点，实现高质量补全。适用于大尺度场景，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.03010v1](https://arxiv.org/pdf/2512.03010v1)**

> **作者:** Svenja Strobel; Matthias Innmann; Bernhard Egger; Marc Stamminger; Linus Franke
>
> **备注:** Project page: https://lfranke.github.io/surffill
>
> **摘要:** LiDAR-captured point clouds are often considered the gold standard in active 3D reconstruction. While their accuracy is exceptional in flat regions, the capturing is susceptible to miss small geometric structures and may fail with dark, absorbent materials. Alternatively, capturing multiple photos of the scene and applying 3D photogrammetry can infer these details as they often represent feature-rich regions. However, the accuracy of LiDAR for featureless regions is rarely reached. Therefore, we suggest combining the strengths of LiDAR and camera-based capture by introducing SurfFill: a Gaussian surfel-based LiDAR completion scheme. We analyze LiDAR capturings and attribute LiDAR beam divergence as a main factor for artifacts, manifesting mostly at thin structures and edges. We use this insight to introduce an ambiguity heuristic for completed scans by evaluating the change in density in the point cloud. This allows us to identify points close to missed areas, which we can then use to grow additional points from to complete the scan. For this point growing, we constrain Gaussian surfel reconstruction [Huang et al. 2024] to focus optimization and densification on these ambiguous areas. Finally, Gaussian primitives of the reconstruction in ambiguous areas are extracted and sampled for points to complete the point cloud. To address the challenges of large-scale reconstruction, we extend this pipeline with a divide-and-conquer scheme for building-sized point cloud completion. We evaluate on the task of LiDAR point cloud completion of synthetic and real-world scenes and find that our method outperforms previous reconstruction methods.
>
---
#### [new 020] Polar Perspectives: Evaluating 2-D LiDAR Projections for Robust Place Recognition with Visual Foundation Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究LiDAR点云投影对视觉基础模型进行地点识别的影响，旨在提升2D投影在复杂环境下的鲁棒性与判别力。通过构建可控的检索流程，系统评估不同投影方式性能，验证了优化投影可有效替代端到端3D学习，适用于实时自主系统。**

- **链接: [https://arxiv.org/pdf/2512.02897v1](https://arxiv.org/pdf/2512.02897v1)**

> **作者:** Pierpaolo Serio; Giulio Pisaneschi; Andrea Dan Ryals; Vincenzo Infantino; Lorenzo Gentilini; Valentina Donzella; Lorenzo Pollini
>
> **备注:** 13 Pages, 5 Figures, 2 Tables Under Review
>
> **摘要:** This work presents a systematic investigation into how alternative LiDAR-to-image projections affect metric place recognition when coupled with a state-of-the-art vision foundation model. We introduce a modular retrieval pipeline that controls for backbone, aggregation, and evaluation protocol, thereby isolating the influence of the 2-D projection itself. Using consistent geometric and structural channels across multiple datasets and deployment scenarios, we identify the projection characteristics that most strongly determine discriminative power, robustness to environmental variation, and suitability for real-time autonomy. Experiments with different datasets, including integration into an operational place recognition policy, validate the practical relevance of these findings and demonstrate that carefully designed projections can serve as an effective surrogate for end-to-end 3-D learning in LiDAR place recognition.
>
---
#### [new 021] U4D: Uncertainty-Aware 4D World Modeling from LiDAR Sequences
- **分类: cs.CV; cs.RO**

- **简介: 该论文聚焦于自动驾驶与具身AI中的4D LiDAR世界建模任务，针对现有方法忽略场景不确定性导致生成伪影的问题，提出U4D框架。通过分割模型生成不确定性图，分“难到易”两阶段建模，并引入时空混合块增强时序一致性，实现更真实、稳定的4D环境重建。**

- **链接: [https://arxiv.org/pdf/2512.02982v1](https://arxiv.org/pdf/2512.02982v1)**

> **作者:** Xiang Xu; Ao Liang; Youquan Liu; Linfeng Li; Lingdong Kong; Ziwei Liu; Qingshan Liu
>
> **备注:** Preprint; 19 pages, 7 figures, 8 tables
>
> **摘要:** Modeling dynamic 3D environments from LiDAR sequences is central to building reliable 4D worlds for autonomous driving and embodied AI. Existing generative frameworks, however, often treat all spatial regions uniformly, overlooking the varying uncertainty across real-world scenes. This uniform generation leads to artifacts in complex or ambiguous regions, limiting realism and temporal stability. In this work, we present U4D, an uncertainty-aware framework for 4D LiDAR world modeling. Our approach first estimates spatial uncertainty maps from a pretrained segmentation model to localize semantically challenging regions. It then performs generation in a "hard-to-easy" manner through two sequential stages: (1) uncertainty-region modeling, which reconstructs high-entropy regions with fine geometric fidelity, and (2) uncertainty-conditioned completion, which synthesizes the remaining areas under learned structural priors. To further ensure temporal coherence, U4D incorporates a mixture of spatio-temporal (MoST) block that adaptively fuses spatial and temporal representations during diffusion. Extensive experiments show that U4D produces geometrically faithful and temporally consistent LiDAR sequences, advancing the reliability of 4D world modeling for autonomous perception and simulation.
>
---
#### [new 022] BEVDilation: LiDAR-Centric Multi-Modal Fusion for 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对3D目标检测中多模态融合因传感器几何精度差异导致性能下降的问题，提出LiDAR-centric的BEVDilation框架。通过图像特征作为隐式引导，缓解深度估计误差带来的空间错位，并利用图像先验增强点云稀疏性与语义信息，提升检测精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.02972v1](https://arxiv.org/pdf/2512.02972v1)**

> **作者:** Guowen Zhang; Chenhang He; Liyi Chen; Lei Zhang
>
> **备注:** Accept by AAAI26
>
> **摘要:** Integrating LiDAR and camera information in the bird's eye view (BEV) representation has demonstrated its effectiveness in 3D object detection. However, because of the fundamental disparity in geometric accuracy between these sensors, indiscriminate fusion in previous methods often leads to degraded performance. In this paper, we propose BEVDilation, a novel LiDAR-centric framework that prioritizes LiDAR information in the fusion. By formulating image BEV features as implicit guidance rather than naive concatenation, our strategy effectively alleviates the spatial misalignment caused by image depth estimation errors. Furthermore, the image guidance can effectively help the LiDAR-centric paradigm to address the sparsity and semantic limitations of point clouds. Specifically, we propose a Sparse Voxel Dilation Block that mitigates the inherent point sparsity by densifying foreground voxels through image priors. Moreover, we introduce a Semantic-Guided BEV Dilation Block to enhance the LiDAR feature diffusion processing with image semantic guidance and long-range context capture. On the challenging nuScenes benchmark, BEVDilation achieves better performance than state-of-the-art methods while maintaining competitive computational efficiency. Importantly, our LiDAR-centric strategy demonstrates greater robustness to depth noise compared to naive fusion. The source code is available at https://github.com/gwenzhang/BEVDilation.
>
---
#### [new 023] On the Convergence of Density-Based Predictive Control for Multi-Agent Non-Uniform Area Coverage
- **分类: eess.SY; cs.RO**

- **简介: 该论文提出密度基预测控制（DPC），解决多智能体非均匀区域覆盖问题。针对传统均匀覆盖忽略区域优先级的缺陷，DPC基于最优传输理论，通过参考分布引导智能体分配覆盖资源，实现高优先区高效覆盖。研究分析收敛性，推导最优控制律，并设计数值方法，仿真验证其优越性。**

- **链接: [https://arxiv.org/pdf/2512.02367v1](https://arxiv.org/pdf/2512.02367v1)**

> **作者:** Sungjun Seo; Kooktae Lee
>
> **备注:** Accepted for publication in ASME JDSMC
>
> **摘要:** This paper presents Density-based Predictive Control (DPC), a novel multi-agent control strategy for efficient non-uniform area coverage, grounded in optimal transport theory. In large-scale scenarios such as search and rescue or environmental monitoring, traditional uniform coverage fails to account for varying regional priorities. DPC leverages a pre-constructed reference distribution to allocate agents' coverage efforts, spending more time in high-priority or densely sampled regions. We analyze convergence conditions using the Wasserstein distance, derive an analytic optimal control law for unconstrained cases, and propose a numerical method for constrained scenarios. Simulations on first-order dynamics and linearized quadrotor models demonstrate that DPC achieves trajectories closely matching the non-uniform reference distribution, outperforming existing coverage methods.
>
---
#### [new 024] nuScenes Revisited: Progress and Challenges in Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文回顾nuScenes数据集的构建与影响，分析其在自动驾驶中的关键作用。针对数据集标准缺失与多模态融合挑战，系统梳理其技术细节、扩展版本及对后续研究的影响，总结主流方法与任务进展，为自动驾驶研究提供全面综述。**

- **链接: [https://arxiv.org/pdf/2512.02448v1](https://arxiv.org/pdf/2512.02448v1)**

> **作者:** Whye Kit Fong; Venice Erin Liong; Kok Seang Tan; Holger Caesar
>
> **备注:** 18 pages, 17 figures
>
> **摘要:** Autonomous Vehicles (AV) and Advanced Driver Assistance Systems (ADAS) have been revolutionized by Deep Learning. As a data-driven approach, Deep Learning relies on vast amounts of driving data, typically labeled in great detail. As a result, datasets, alongside hardware and algorithms, are foundational building blocks for the development of AVs. In this work we revisit one of the most widely used autonomous driving datasets: the nuScenes dataset. nuScenes exemplifies key trends in AV development, being the first dataset to include radar data, to feature diverse urban driving scenes from two continents, and to be collected using a fully autonomous vehicle operating on public roads, while also promoting multi-modal sensor fusion, standardized benchmarks, and a broad range of tasks including perception, localization \& mapping, prediction and planning. We provide an unprecedented look into the creation of nuScenes, as well as its extensions nuImages and Panoptic nuScenes, summarizing many technical details that have hitherto not been revealed in academic publications. Furthermore, we trace how the influence of nuScenes impacted a large number of other datasets that were released later and how it defined numerous standards that are used by the community to this day. Finally, we present an overview of both official and unofficial tasks using the nuScenes dataset and review major methodological developments, thereby offering a comprehensive survey of the autonomous driving literature, with a particular focus on nuScenes.
>
---
#### [new 025] Reframing Human-Robot Interaction Through Extended Reality: Unlocking Safer, Smarter, and More Empathic Interactions with Virtual Robots and Foundation Models
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互（HRI）领域，旨在通过扩展现实（XR）与基础模型（FM）融合，解决物理机器人局限性问题。提出虚拟机器人可实现安全、智能、共情交互，突破硬件限制，支持多模态感知与长期适应。研究构建了以用户为中心的伦理化XR代理框架，涵盖评估、生态与社会设计，推动更高效、自适应的未来人机交互范式。**

- **链接: [https://arxiv.org/pdf/2512.02569v1](https://arxiv.org/pdf/2512.02569v1)**

> **作者:** Yuchong Zhang; Yong Ma; Danica Kragic
>
> **备注:** This paper is under review
>
> **摘要:** This perspective reframes human-robot interaction (HRI) through extended reality (XR), arguing that virtual robots powered by large foundation models (FMs) can serve as cognitively grounded, empathic agents. Unlike physical robots, XR-native agents are unbound by hardware constraints and can be instantiated, adapted, and scaled on demand, while still affording embodiment and co-presence. We synthesize work across XR, HRI, and cognitive AI to show how such agents can support safety-critical scenarios, socially and cognitively empathic interaction across domains, and outreaching physical capabilities with XR and AI integration. We then discuss how multimodal large FMs (e.g., large language model, large vision model, and vision-language model) enable context-aware reasoning, affect-sensitive situations, and long-term adaptation, positioning virtual robots as cognitive and empathic mediators rather than mere simulation assets. At the same time, we highlight challenges and potential risks, including overtrust, cultural and representational bias, privacy concerns around biometric sensing, and data governance and transparency. The paper concludes by outlining a research agenda for human-centered, ethically grounded XR agents - emphasizing multi-layered evaluation frameworks, multi-user ecosystems, mixed virtual-physical embodiment, and societal and ethical design practices to envision XR-based virtual agents powered by FMs as reshaping future HRI into a more efficient and adaptive paradigm.
>
---
#### [new 026] Property-Guided Cyber-Physical Reduction and Surrogation for Safety Analysis in Robotic Vehicles
- **分类: cs.CR; cs.RO**

- **简介: 该论文针对机器人车辆系统安全分析难题，提出属性引导的系统降维与代理执行方法。通过提取与安全属性相关的控制逻辑与物理动态，构建轻量级代理模型，实现高效可扩展的故障检测。实验验证其在降低仿真成本的同时，能精准发现安全缺陷，为复杂系统的语义验证提供新路径。**

- **链接: [https://arxiv.org/pdf/2512.02270v1](https://arxiv.org/pdf/2512.02270v1)**

> **作者:** Nazmus Shakib Sayom; Luis Garcia
>
> **备注:** Accepted at EAI SmartSP 2025 (EAI International Conference on Security and Privacy in Cyber-Physical Systems and Smart Vehicles), Springer LNICST. The code repository is available here: https://doi.org/10.5281/zenodo.17497068
>
> **摘要:** We propose a methodology for falsifying safety properties in robotic vehicle systems through property-guided reduction and surrogate execution. By isolating only the control logic and physical dynamics relevant to a given specification, we construct lightweight surrogate models that preserve property-relevant behaviors while eliminating unrelated system complexity. This enables scalable falsification via trace analysis and temporal logic oracles. We demonstrate the approach on a drone control system containing a known safety flaw. The surrogate replicates failure conditions at a fraction of the simulation cost, and a property-guided fuzzer efficiently discovers semantic violations. Our results suggest that controller reduction, when coupled with logic-aware test generation, provides a practical and scalable path toward semantic verification of cyber-physical systems.
>
---
## 更新

#### [replaced 001] A Practical Guide for Incorporating Symmetry in Diffusion Policy
- **分类: cs.RO**

- **简介: 该论文研究如何在扩散策略中融入对称性以提升样本效率与泛化能力。针对等变网络实现复杂的问题，提出三种轻量级方法：相对动作、眼手感知、帧平均特征提取。实验证明，结合不变表示与等变特征提取可显著提升性能，接近全等变架构但大幅简化实现。**

- **链接: [https://arxiv.org/pdf/2505.13431v3](https://arxiv.org/pdf/2505.13431v3)**

> **作者:** Dian Wang; Boce Hu; Shuran Song; Robin Walters; Robert Platt
>
> **备注:** NeurIPS 2025
>
> **摘要:** Recently, equivariant neural networks for policy learning have shown promising improvements in sample efficiency and generalization, however, their wide adoption faces substantial barriers due to implementation complexity. Equivariant architectures typically require specialized mathematical formulations and custom network design, posing significant challenges when integrating with modern policy frameworks like diffusion-based models. In this paper, we explore a number of straightforward and practical approaches to incorporate symmetry benefits into diffusion policies without the overhead of full equivariant designs. Specifically, we investigate (i) invariant representations via relative trajectory actions and eye-in-hand perception, (ii) integrating equivariant vision encoders, and (iii) symmetric feature extraction with pretrained encoders using Frame Averaging. We first prove that combining eye-in-hand perception with relative or delta action parameterization yields inherent SE(3)-invariance, thus improving policy generalization. We then perform a systematic experimental study on those design choices for integrating symmetry in diffusion policies, and conclude that an invariant representation with equivariant feature extraction significantly improves the policy performance. Our method achieves performance on par with or exceeding fully equivariant architectures while greatly simplifying implementation.
>
---
#### [replaced 002] LAP: Fast LAtent Diffusion Planner with Fine-Grained Feature Distillation for Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文针对自动驾驶中扩散模型推理慢、低层运动细节干扰高层语义的问题，提出LAP框架。通过在VAE隐空间中分离意图与运动，实现单步生成高质量轨迹，结合细粒度特征蒸馏提升场景理解，显著提升规划效率与性能。**

- **链接: [https://arxiv.org/pdf/2512.00470v2](https://arxiv.org/pdf/2512.00470v2)**

> **作者:** Jinhao Zhang; Wenlong Xia; Zhexuan Zhou; Youmin Gong; Jie Mei
>
> **摘要:** Diffusion models have demonstrated strong capabilities for modeling human-like driving behaviors in autonomous driving, but their iterative sampling process induces substantial latency, and operating directly on raw trajectory points forces the model to spend capacity on low-level kinematics, rather than high-level multi-modal semantics. To address these limitations, we propose LAtent Planner (LAP), a framework that plans in a VAE-learned latent space that disentangles high-level intents from low-level kinematics, enabling our planner to capture rich, multi-modal driving strategies. We further introduce a fine-grained feature distillation mechanism to guide a better interaction and fusion between the high-level semantic planning space and the vectorized scene context. Notably, LAP can produce high-quality plans in one single denoising step, substantially reducing computational overhead. Through extensive evaluations on the large-scale nuPlan benchmark, LAP achieves state-of-the-art closed-loop performance among learning-based planning methods, while demonstrating an inference speed-up of at most 10 times over previous SOTA approaches. Code will be released at: https://github.com/jhz1192/Latent-Planner.
>
---
#### [replaced 003] Image-Based Relocalization and Alignment for Long-Term Monitoring of Dynamic Underwater Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对长期动态水下环境监测中视觉定位困难的问题，提出融合VPR、特征匹配与图像分割的重定位与对齐方法。旨在实现水下场景的精准回溯与变化分析，并构建首个大规模跨时序水下VPR基准数据集SQUIDLE+，推动自动化水下生态监测发展。**

- **链接: [https://arxiv.org/pdf/2503.04096v2](https://arxiv.org/pdf/2503.04096v2)**

> **作者:** Beverley Gorry; Tobias Fischer; Michael Milford; Alejandro Fontan
>
> **摘要:** Effective monitoring of underwater ecosystems is crucial for tracking environmental changes, guiding conservation efforts, and ensuring long-term ecosystem health. However, automating underwater ecosystem management with robotic platforms remains challenging due to the complexities of underwater imagery, which pose significant difficulties for traditional visual localization methods. We propose an integrated pipeline that combines Visual Place Recognition (VPR), feature matching, and image segmentation on video-derived images. This method enables robust identification of revisited areas, estimation of rigid transformations, and downstream analysis of ecosystem changes. Furthermore, we introduce the SQUIDLE+ VPR Benchmark-the first large-scale underwater VPR benchmark designed to leverage an extensive collection of unstructured data from multiple robotic platforms, spanning time intervals from days to years. The dataset encompasses diverse trajectories, arbitrary overlap and diverse seafloor types captured under varying environmental conditions, including differences in depth, lighting, and turbidity. Our code is available at: https://github.com/bev-gorry/underloc
>
---
#### [replaced 004] ST-Booster: An Iterative SpatioTemporal Perception Booster for Vision-and-Language Navigation in Continuous Environments
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究视觉-语言导航（VLN-CE）任务，针对连续环境中视觉记忆异构与三维结构噪声问题，提出ST-Booster模型。通过分层时空编码、多粒度对齐融合与价值引导路径生成，实现指令感知的迭代优化，显著提升复杂环境下的导航性能。**

- **链接: [https://arxiv.org/pdf/2504.09843v2](https://arxiv.org/pdf/2504.09843v2)**

> **作者:** Lu Yue; Dongliang Zhou; Liang Xie; Erwei Yin; Feitian Zhang
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Vision-and-Language Navigation in Continuous Environments (VLN-CE) requires agents to navigate unknown, continuous spaces based on natural language instructions. Compared to discrete settings, VLN-CE poses two core perception challenges. First, the absence of predefined observation points leads to heterogeneous visual memories and weakened global spatial correlations. Second, cumulative reconstruction errors in three-dimensional scenes introduce structural noise, impairing local feature perception. To address these challenges, this paper proposes ST-Booster, an iterative spatiotemporal booster that enhances navigation performance through multi-granularity perception and instruction-aware reasoning. ST-Booster consists of three key modules -- Hierarchical SpatioTemporal Encoding (HSTE), Multi-Granularity Aligned Fusion (MGAF), and ValueGuided Waypoint Generation (VGWG). HSTE encodes long-term global memory using topological graphs and captures shortterm local details via grid maps. MGAF aligns these dualmap representations with instructions through geometry-aware knowledge fusion. The resulting representations are iteratively refined through pretraining tasks. During reasoning, VGWG generates Guided Attention Heatmaps (GAHs) to explicitly model environment-instruction relevance and optimize waypoint selection. Extensive comparative experiments and performance analyses are conducted, demonstrating that ST-Booster outperforms existing state-of-the-art methods, particularly in complex, disturbance-prone environments.
>
---
#### [replaced 005] Agentic UAVs: LLM-Driven Autonomy with Integrated Tool-Calling and Cognitive Reasoning
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出Agentic UAV框架，解决传统无人机在动态任务中自主性不足、缺乏上下文推理与系统集成的问题。通过五层架构融合LLM推理与工具调用，实现基于GPT-4和Gemma 3的智能决策。在搜救仿真中显著提升检测率与决策准确率，验证了其在增强自主性与系统协同方面的有效性。**

- **链接: [https://arxiv.org/pdf/2509.13352v2](https://arxiv.org/pdf/2509.13352v2)**

> **作者:** Anis Koubaa; Khaled Gabr
>
> **备注:** 17 pages, 2 figure
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are increasingly used in defense, surveillance, and disaster response, yet most systems still operate at SAE Level 2 to 3 autonomy. Their dependence on rule-based control and narrow AI limits adaptability in dynamic and uncertain missions. Current UAV architectures lack context-aware reasoning, autonomous decision-making, and integration with external systems. Importantly, none make use of Large Language Model (LLM) agents with tool-calling for real-time knowledge access. This paper introduces the Agentic UAVs framework, a five-layer architecture consisting of Perception, Reasoning, Action, Integration, and Learning. The framework enhances UAV autonomy through LLM-driven reasoning, database querying, and interaction with third-party systems. A prototype built with ROS 2 and Gazebo combines YOLOv11 for object detection with GPT-4 for reasoning and a locally deployed Gemma 3 model. In simulated search-and-rescue scenarios, agentic UAVs achieved higher detection confidence (0.79 compared to 0.72), improved person detection rates (91% compared to 75%), and a major increase in correct action recommendations (92% compared to 4.5%). These results show that modest computational overhead can enable significantly higher levels of autonomy and system-level integration.
>
---
#### [replaced 006] Efficient Policy Optimization in Robust Constrained MDPs with Iteration Complexity Guarantees
- **分类: cs.LG; cs.AI; cs.RO; eess.SY**

- **简介: 该论文研究鲁棒约束马尔可夫决策过程（RCMDP）下的高效策略优化问题，旨在在模型不确定下同时最大化奖励并满足约束。针对传统方法无法处理强对偶性缺失及最坏情况模型差异的问题，提出新算法，通过分阶段优化约束与奖励函数，在$O(ε^{-2})$次迭代内实现近优且可行的策略，无需二分搜索，显著提升计算效率。**

- **链接: [https://arxiv.org/pdf/2505.19238v2](https://arxiv.org/pdf/2505.19238v2)**

> **作者:** Sourav Ganguly; Arnob Ghosh; Kishan Panaganti; Adam Wierman
>
> **摘要:** Constrained decision-making is essential for designing safe policies in real-world control systems, yet simulated environments often fail to capture real-world adversities. We consider the problem of learning a policy that will maximize the cumulative reward while satisfying a constraint, even when there is a mismatch between the real model and an accessible simulator/nominal model. In particular, we consider the robust constrained Markov decision problem (RCMDP) where an agent needs to maximize the reward and satisfy the constraint against the worst possible stochastic model under the uncertainty set centered around an unknown nominal model. Primal-dual methods, effective for standard constrained MDP (CMDP), are not applicable here because of the lack of the strong duality property. Further, one cannot apply the standard robust value-iteration based approach on the composite value function either as the worst case models may be different for the reward value function and the constraint value function. We propose a novel technique that effectively minimizes the constraint value function--to satisfy the constraints; on the other hand, when all the constraints are satisfied, it can simply maximize the robust reward value function. We prove that such an algorithm finds a policy with at most $ε$ sub-optimality and feasible policy after $O(ε^{-2})$ iterations. In contrast to the state-of-the-art method, we do not need to employ a binary search, thus, we reduce the computation time by at least 4x for smaller value of discount factor ($γ$) and by at least 6x for larger value of $γ$.
>
---
#### [replaced 007] Is Image-based Object Pose Estimation Ready to Support Grasping?
- **分类: cs.RO**

- **简介: 该论文针对基于单张RGB图像的6-DoF物体位姿估计任务，评估其在机器人抓取中的实用性。通过物理仿真环境测试五种开源估计算法，验证其能否作为抓取唯一感知源，揭示现有方法在实际应用中的局限性与潜力。**

- **链接: [https://arxiv.org/pdf/2512.01856v2](https://arxiv.org/pdf/2512.01856v2)**

> **作者:** Eric C. Joyce; Qianwen Zhao; Nathaniel Burgdorfer; Long Wang; Philippos Mordohai
>
> **摘要:** We present a framework for evaluating 6-DoF instance-level object pose estimators, focusing on those that require a single RGB (not RGB-D) image as input. Besides gaining intuition about how accurate these estimators are, we are interested in the degree to which they can serve as the sole perception mechanism for robotic grasping. To assess this, we perform grasping trials in a physics-based simulator, using image-based pose estimates to guide a parallel gripper and an underactuated robotic hand in picking up 3D models of objects. Our experiments on a subset of the BOP (Benchmark for 6D Object Pose Estimation) dataset compare five open-source object pose estimators and provide insights that were missing from the literature.
>
---
#### [replaced 008] WARPD: World model Assisted Reactive Policy Diffusion
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文针对机器人控制中扩散模型推理慢、长时序误差累积的问题，提出WARPD方法。通过学习参数空间的策略分布，直接生成闭环策略权重，实现更长动作时域、更强鲁棒性与更低计算开销，显著提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2410.14040v3](https://arxiv.org/pdf/2410.14040v3)**

> **作者:** Shashank Hegde; Satyajeet Das; Gautam Salhotra; Gaurav S. Sukhatme
>
> **摘要:** With the increasing availability of open-source robotic data, imitation learning has become a promising approach for both manipulation and locomotion. Diffusion models are now widely used to train large, generalized policies that predict controls or trajectories, leveraging their ability to model multimodal action distributions. However, this generality comes at the cost of larger model sizes and slower inference, an acute limitation for robotic tasks requiring high control frequencies. Moreover, Diffusion Policy (DP), a popular trajectory-generation approach, suffers from a trade-off between performance and action horizon: fewer diffusion queries lead to larger trajectory chunks, which in turn accumulate tracking errors. To overcome these challenges, we introduce WARPD (World model Assisted Reactive Policy Diffusion), a method that generates closed-loop policies (weights for neural policies) directly, instead of open-loop trajectories. By learning behavioral distributions in parameter space rather than trajectory space, WARPD offers two major advantages: (1) extended action horizons with robustness to perturbations, while maintaining high task performance, and (2) significantly reduced inference costs. Empirically, WARPD outperforms DP in long-horizon and perturbed environments, and achieves multitask performance on par with DP while requiring only ~ 1/45th of the inference-time FLOPs per step.
>
---
#### [replaced 009] Multi-User Personalisation in Human-Robot Interaction: Resolving Preference Conflicts Using Gradual Argumentation
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对多用户人机交互中的偏好冲突问题，提出MUP-QBAF框架，基于定量双极论证模型动态融合用户偏好与环境观测，实现多用户个性化适应。通过真实场景案例验证了其在冲突调解中的有效性与可解释性，为复杂情境下机器人决策提供了结构化方法。**

- **链接: [https://arxiv.org/pdf/2511.03576v2](https://arxiv.org/pdf/2511.03576v2)**

> **作者:** Aniol Civit; Antonio Andriella; Carles Sierra; Guillem Alenyà
>
> **备注:** Preprint submitted to a journal
>
> **摘要:** While personalisation in Human-Robot Interaction (HRI) has advanced significantly, most existing approaches focus on single-user adaptation, overlooking scenarios involving multiple stakeholders with potentially conflicting preferences. To address this, we propose the Multi-User Preferences Quantitative Bipolar Argumentation Framework (MUP-QBAF), a novel multi-user personalisation framework based on Quantitative Bipolar Argumentation Frameworks (QBAFs) that explicitly models and resolves multi-user preference conflicts. Unlike prior work in Argumentation Frameworks, which typically assumes static inputs, our approach is tailored to robotics: it incorporates both users' arguments and the robot's dynamic observations of the environment, allowing the system to adapt over time and respond to changing contexts. Preferences, both positive and negative, are represented as arguments whose strength is recalculated iteratively based on new information. The framework's properties and capabilities are presented and validated through a realistic case study, where an assistive robot mediates between the conflicting preferences of a caregiver and a care recipient during a frailty assessment task. This evaluation further includes a sensitivity analysis of argument base scores, demonstrating how preference outcomes can be shaped by user input and contextual observations. By offering a transparent, structured, and context-sensitive approach to resolving competing user preferences, this work advances the field of multi-user HRI. It provides a principled alternative to data-driven methods, enabling robots to navigate conflicts in real-world environments.
>
---
#### [replaced 010] SPARK: Sim-ready Part-level Articulated Reconstruction with VLM Knowledge
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SPARK框架，解决从单张RGB图像生成可模拟的关节式3D物体问题。利用视觉语言模型提取粗略URDF参数并生成部件参考图，结合扩散变换器生成一致的部件与完整形状，并通过可微正向运动学与渲染优化关节参数，实现高保真、物理一致的仿真资产生成。**

- **链接: [https://arxiv.org/pdf/2512.01629v2](https://arxiv.org/pdf/2512.01629v2)**

> **作者:** Yumeng He; Ying Jiang; Jiayin Lu; Yin Yang; Chenfanfu Jiang
>
> **备注:** Project page: https://heyumeng.com/SPARK/index.html. 17 pages, 7 figures
>
> **摘要:** Articulated 3D objects are critical for embodied AI, robotics, and interactive scene understanding, yet creating simulation-ready assets remains labor-intensive and requires expert modeling of part hierarchies and motion structures. We introduce SPARK, a framework for reconstructing physically consistent, kinematic part-level articulated objects from a single RGB image. Given an input image, we first leverage VLMs to extract coarse URDF parameters and generate part-level reference images. We then integrate the part-image guidance and the inferred structure graph into a generative diffusion transformer to synthesize consistent part and complete shapes of articulated objects. To further refine the URDF parameters, we incorporate differentiable forward kinematics and differentiable rendering to optimize joint types, axes, and origins under VLM-generated open-state supervision. Extensive experiments show that SPARK produces high-quality, simulation-ready articulated assets across diverse categories, enabling downstream applications such as robotic manipulation and interaction modeling. Project page: https://heyumeng.com/SPARK/index.html.
>
---
#### [replaced 011] MIMIC-MJX: Neuromechanical Emulation of Animal Behavior
- **分类: q-bio.NC; cs.AI; cs.RO**

- **简介: 该论文提出MIMIC-MJX框架，旨在从动物运动轨迹反推生物合理的神经控制策略。针对仅靠运动学数据难以揭示神经控制机制的问题，通过训练神经控制器驱动物理仿真中的生物力学模型，重现真实运动轨迹，实现对神经控制策略的建模与分析。**

- **链接: [https://arxiv.org/pdf/2511.20532v2](https://arxiv.org/pdf/2511.20532v2)**

> **作者:** Charles Y. Zhang; Yuanjia Yang; Aidan Sirbu; Elliott T. T. Abe; Emil Wärnberg; Eric J. Leonardis; Diego E. Aldarondo; Adam Lee; Aaditya Prasad; Jason Foat; Kaiwen Bian; Joshua Park; Rusham Bhatt; Hutton Saunders; Akira Nagamori; Ayesha R. Thanawalla; Kee Wui Huang; Fabian Plum; Hendrik K. Beck; Steven W. Flavell; David Labonte; Blake A. Richards; Bingni W. Brunton; Eiman Azim; Bence P. Ölveczky; Talmo D. Pereira
>
> **备注:** Corrected LaTeX issues. Project page available at https://mimic-mjx.talmolab.org
>
> **摘要:** The primary output of the nervous system is movement and behavior. While recent advances have democratized pose tracking during complex behavior, kinematic trajectories alone provide only indirect access to the underlying control processes. Here we present MIMIC-MJX, a framework for learning biologically-plausible neural control policies from kinematics. MIMIC-MJX models the generative process of motor control by training neural controllers that learn to actuate biomechanically-realistic body models in physics simulation to reproduce real kinematic trajectories. We demonstrate that our implementation is accurate, fast, data-efficient, and generalizable to diverse animal body models. Policies trained with MIMIC-MJX can be utilized to both analyze neural control strategies and simulate behavioral experiments, illustrating its potential as an integrative modeling framework for neuroscience.
>
---
#### [replaced 012] Guardian: Detecting Robotic Planning and Execution Errors with Vision-Language Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中故障检测与恢复难题，提出一种自动合成失败数据的方法，生成多样化的规划与执行错误。基于此构建三个新基准，训练出Guardian模型，实现高精度故障检测与细粒度推理，在仿真与真实机器人上均显著提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2512.01946v2](https://arxiv.org/pdf/2512.01946v2)**

> **作者:** Paul Pacaud; Ricardo Garcia; Shizhe Chen; Cordelia Schmid
>
> **备注:** Code, Data, and Models available at https://www.di.ens.fr/willow/research/guardian/. The paper contains 8 pages, 9 figures, 6 tables
>
> **摘要:** Robust robotic manipulation requires reliable failure detection and recovery. Although current Vision-Language Models (VLMs) show promise, their accuracy and generalization are limited by the scarcity of failure data. To address this data gap, we propose an automatic robot failure synthesis approach that procedurally perturbs successful trajectories to generate diverse planning and execution failures. This method produces not only binary classification labels but also fine-grained failure categories and step-by-step reasoning traces in both simulation and the real world. With it, we construct three new failure detection benchmarks: RLBench-Fail, BridgeDataV2-Fail, and UR5-Fail, substantially expanding the diversity and scale of existing failure datasets. We then train Guardian, a VLM with multi-view images for detailed failure reasoning and detection. Guardian achieves state-of-the-art performance on both existing and newly introduced benchmarks. It also effectively improves task success rates when integrated into a state-of-the-art manipulation system in simulation and real robots, demonstrating the impact of our generated failure data. Code, Data, and Models available at https://www.di.ens.fr/willow/research/guardian/.
>
---
#### [replaced 013] OpenGVL -- Benchmarking Visual Temporal Progress for Data Curation
- **分类: cs.RO; cs.CL**

- **简介: 该论文提出OpenGVL，一个用于评估视觉时间进度的基准，解决机器人数据稀缺与大规模数据标注难题。通过对比开源与闭源模型在任务进度预测中的表现，揭示开源模型显著落后，并展示其在自动化数据筛选中的应用价值。**

- **链接: [https://arxiv.org/pdf/2509.17321v3](https://arxiv.org/pdf/2509.17321v3)**

> **作者:** Paweł Budzianowski; Emilia Wiśnios; Gracjan Góral; Igor Kulakov; Viktor Petrenko; Krzysztof Walas
>
> **备注:** Workshop on Making Sense of Data in Robotics: Composition, Curation, and Interpretability at Scale at CoRL 2025
>
> **摘要:** Data scarcity remains one of the most limiting factors in driving progress in robotics. However, the amount of available robotics data in the wild is growing exponentially, creating new opportunities for large-scale data utilization. Reliable temporal task completion prediction could help automatically annotate and curate this data at scale. The Generative Value Learning (GVL) approach was recently proposed, leveraging the knowledge embedded in vision-language models (VLMs) to predict task progress from visual observations. Building upon GVL, we propose OpenGVL, a comprehensive benchmark for estimating task progress across diverse challenging manipulation tasks involving both robotic and human embodiments. We evaluate the capabilities of publicly available open-source foundation models, showing that open-source model families significantly underperform closed-source counterparts, achieving only approximately $70\%$ of their performance on temporal progress prediction tasks. Furthermore, we demonstrate how OpenGVL can serve as a practical tool for automated data curation and filtering, enabling efficient quality assessment of large-scale robotics datasets. We release the benchmark along with the complete codebase at \href{github.com/budzianowski/opengvl}{OpenGVL}.
>
---
#### [replaced 014] Forecasting in Offline Reinforcement Learning for Non-stationary Environments
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文针对非平稳环境下的离线强化学习问题，提出FORL框架。通过条件扩散生成候选状态与零样本时间序列模型，应对突发、时变偏移导致的观测不全问题，提升代理在未知非平稳环境中的初始性能。**

- **链接: [https://arxiv.org/pdf/2512.01987v2](https://arxiv.org/pdf/2512.01987v2)**

> **作者:** Suzan Ece Ada; Georg Martius; Emre Ugur; Erhan Oztop
>
> **备注:** The Thirty-Ninth Annual Conference on Neural Information Processing Systems, NeurIPS 2025
>
> **摘要:** Offline Reinforcement Learning (RL) provides a promising avenue for training policies from pre-collected datasets when gathering additional interaction data is infeasible. However, existing offline RL methods often assume stationarity or only consider synthetic perturbations at test time, assumptions that often fail in real-world scenarios characterized by abrupt, time-varying offsets. These offsets can lead to partial observability, causing agents to misperceive their true state and degrade performance. To overcome this challenge, we introduce Forecasting in Non-stationary Offline RL (FORL), a framework that unifies (i) conditional diffusion-based candidate state generation, trained without presupposing any specific pattern of future non-stationarity, and (ii) zero-shot time-series foundation models. FORL targets environments prone to unexpected, potentially non-Markovian offsets, requiring robust agent performance from the onset of each episode. Empirical evaluations on offline RL benchmarks, augmented with real-world time-series data to simulate realistic non-stationarity, demonstrate that FORL consistently improves performance compared to competitive baselines. By integrating zero-shot forecasting with the agent's experience, we aim to bridge the gap between offline RL and the complexities of real-world, non-stationary environments.
>
---
#### [replaced 015] ViTaMIn-B: A Reliable and Efficient Visuo-Tactile Bimanual Manipulation Interface
- **分类: cs.RO**

- **简介: 该论文针对双臂触觉操作中感知与定位不精准的问题，提出ViTaMIn-B系统。设计柔性触觉传感器DuoTact，实现高分辨率接触几何捕捉；通过重建3D点云提升跨传感器泛化性；采用Meta Quest控制器实现无漂移的6-DoF双臂位姿追踪，显著提升数据采集效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2511.05858v2](https://arxiv.org/pdf/2511.05858v2)**

> **作者:** Chuanyu Li; Chaoyi Liu; Daotan Wang; Shuyu Zhang; Lusong Li; Zecui Zeng; Fangchen Liu; Jing Xu; Rui Chen
>
> **备注:** Project page: https://chuanyune.github.io/ViTaMIn-B_page/
>
> **摘要:** Handheld devices have opened up unprecedented opportunities to collect large-scale, high-quality demonstrations efficiently. However, existing systems often lack robust tactile sensing or reliable pose tracking to handle complex interaction scenarios, especially for bimanual and contact-rich tasks. In this work, we propose ViTaMIn-B, a more capable and efficient handheld data collection system for such tasks. We first design DuoTact, a novel compliant visuo-tactile sensor built with a flexible frame to withstand large contact forces during manipulation while capturing high-resolution contact geometry. To enhance the cross-sensor generalizability, we propose reconstructing the sensor's global deformation as a 3D point cloud and using it as the policy input. We further develop a robust, unified 6-DoF bimanual pose acquisition process using Meta Quest controllers, which eliminates the trajectory drift issue in common SLAM-based methods. Comprehensive user studies confirm the efficiency and high usability of ViTaMIn-B among novice and expert operators. Furthermore, experiments on four bimanual manipulation tasks demonstrate its superior task performance relative to existing systems. Project page: https://chuanyune.github.io/ViTaMIn-B_page/
>
---
#### [replaced 016] EMMA: Scaling Mobile Manipulation via Egocentric Human Data
- **分类: cs.RO**

- **简介: 该论文提出EMMA框架，解决移动操作技能学习中依赖昂贵机器人遥操作的问题。通过联合训练人类第一人称动作数据与静态机器人数据，实现无需移动遥操作的端到端策略学习，实现在真实任务中媲美遥操作基线的表现，并展现良好泛化性与数据规模扩展性。**

- **链接: [https://arxiv.org/pdf/2509.04443v2](https://arxiv.org/pdf/2509.04443v2)**

> **作者:** Lawrence Y. Zhu; Pranav Kuppili; Ryan Punamiya; Patcharapong Aphiwetsa; Dhruv Patel; Simar Kareer; Sehoon Ha; Danfei Xu
>
> **摘要:** Scaling mobile manipulation imitation learning is bottlenecked by expensive mobile robot teleoperation. We present Egocentric Mobile MAnipulation (EMMA), an end-to-end framework training mobile manipulation policies from human mobile manipulation data with static robot data, sidestepping mobile teleoperation. To accomplish this, we co-train human full-body motion data with static robot data. In our experiments across three real-world tasks, EMMA demonstrates comparable performance to baselines trained on teleoperated mobile robot data (Mobile ALOHA), achieving higher or equivalent task performance in full task success. We find that EMMA is able to generalize to new spatial configurations and scenes, and we observe positive performance scaling as we increase the hours of human data, opening new avenues for scalable robotic learning in real-world environments. Details of this project can be found at https://ego-moma.github.io/.
>
---
#### [replaced 017] Large Language Models for Robotics: A Survey
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于综述任务，旨在探讨大语言模型（LLM）在机器人领域的应用。针对提升机器人智能、人机交互与自主性的问题，系统梳理了LLM在感知、决策、控制及跨模块协同中的技术进展，总结应用现状并展望未来挑战。**

- **链接: [https://arxiv.org/pdf/2311.07226v2](https://arxiv.org/pdf/2311.07226v2)**

> **作者:** Fanlong Zeng; Wensheng Gan; Zezheng Huai; Lichao Sun; Hechang Chen; Yongheng Wang; Ning Liu; Philip S. Yu
>
> **备注:** Preprint. 9 figures, 7 tables
>
> **摘要:** The human ability to learn, generalize, and control complex manipulation tasks through multi-modality feedback suggests a unique capability, which we refer to as dexterity intelligence. Understanding and assessing this intelligence is a complex task. Amidst the swift progress and extensive proliferation of large language models (LLMs), their applications in the field of robotics have garnered increasing attention. LLMs possess the ability to process and generate natural language, facilitating efficient interaction and collaboration with robots. Researchers and engineers in the field of robotics have recognized the immense potential of LLMs in enhancing robot intelligence, human-robot interaction, and autonomy. Therefore, this comprehensive review aims to summarize the applications of LLMs in robotics, delving into their impact and contributions to key areas such as robot control, perception, decision-making, and planning. This survey first provides an overview of the background and development of LLMs for robotics, followed by a discussion of their benefits and recent advancements in LLM-based robotic models. It then explores various techniques, employed in perception, decision-making, control, and interaction, as well as cross-module coordination in practical tasks. Finally, we review current applications of LLMs in robotics and outline potential challenges they may face in the near future. Embodied intelligence represents the future of intelligent systems, and LLM-based robotics is one of the most promising yet challenging paths toward achieving it.
>
---
#### [replaced 018] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在动态决策中忽视历史上下文的问题，提出AVA-VLA框架。通过引入基于信念状态的主动视觉注意力机制，利用递归状态动态聚焦关键视觉信息，提升序列决策能力。在LIBERO、CALVIN等基准上实现领先性能，并验证了真实机器人平台上的有效性与仿真到现实的迁移能力。**

- **链接: [https://arxiv.org/pdf/2511.18960v2](https://arxiv.org/pdf/2511.18960v2)**

> **作者:** Lei Xiao; Jifeng Li; Juntao Gao; Feiyang Ye; Yan Jin; Jingjing Qian; Jing Zhang; Yong Wu; Xiaoyuan Yu
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in embodied AI tasks. However, existing VLA models, often built upon Vision-Language Models (VLMs), typically process dense visual inputs independently at each timestep. This approach implicitly models the task as a Markov Decision Process (MDP). However, this history-agnostic design is suboptimal for effective visual token processing in dynamic sequential decision-making, as it fails to leverage the context of history. To address this limitation, we reformulate the problem from a Partially Observable Markov Decision Process (POMDP) perspective and propose a novel framework named AVA-VLA. Inspired by the POMDP that the action generation should be conditioned on the belief state. AVA-VLA introduces Active Visual Attention (AVA) to dynamically modulate visual processing. It achieves this by leveraging the recurrent state, which is a neural approximation of the agent's belief state derived from the previous decision step. Specifically, the AVA module uses the recurrent state to compute the soft weights to actively process task-relevant visual tokens based on its historical context. Comprehensive evaluations demonstrate that AVA-VLA achieves state-of-the-art performance across popular robotic benchmarks, including LIBERO and CALVIN. Furthermore, real-world deployments on a dual-arm robot platform validate the framework's practical applicability and robust sim-to-real transferability.
>
---
#### [replaced 019] Learning-based 3D Reconstruction in Autonomous Driving: A Comprehensive Survey
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的学习型3D重建任务，旨在解决环境精确建模难题。系统综述了相关技术演进与应用，分析了方法分类、挑战及研究趋势，指出当前研究在车载验证与安全评估方面披露不足，并提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2503.14537v5](https://arxiv.org/pdf/2503.14537v5)**

> **作者:** Liewen Liao; Weihao Yan; Wang Xu; Ming Yang; Songan Zhang; H. Eric Tseng
>
> **备注:** Published in IEEE Trans. on Intelligent Transportation Systems
>
> **摘要:** Learning-based 3D reconstruction has emerged as a transformative technique in autonomous driving, enabling precise modeling of environments through advanced neural representations. It has inspired pioneering solutions for vital tasks in autonomous driving, such as dense mapping and closed-loop simulation, as well as comprehensive scene feature for driving scene understanding and reasoning. Given the rapid growth in related research, this survey provides a comprehensive review of both technical evolutions and practical applications in autonomous driving. We begin with an introduction to the preliminaries of learning-based 3D reconstruction to provide a solid technical background foundation, then progress to a rigorous, multi-dimensional examination of cutting-edge methodologies, systematically organized according to the distinctive technical requirements and fundamental challenges of autonomous driving. Through analyzing and summarizing development trends and cutting-edge research, we identify existing technical challenges, along with insufficient disclosure of on-board validation and safety verification details in the current literature, and ultimately suggest potential directions to guide future studies.
>
---
#### [replaced 020] Sigma: The Key for Vision-Language-Action Models toward Telepathic Alignment
- **分类: cs.LG; cs.RO**

- **简介: 该论文针对人形机器人认知系统中语义与连续控制间缺乏可时更新的思维空间问题，提出名为Sigma的视觉-语言-动作模型。基于pi05_base模型，通过数据预处理、LoRA微调与推理适配器优化，实现无需重训练的意图驱动控制，显著降低控制误差，保持语义对齐，推动了人机“心灵感应”式交互。**

- **链接: [https://arxiv.org/pdf/2512.00783v2](https://arxiv.org/pdf/2512.00783v2)**

> **作者:** Libo Wang
>
> **备注:** The Sigma model has been open-sourced on Hugging Face. Weights, dataset, some scripts, and logs are all available. The link is: https://huggingface.co/Veltraxor/Sigma
>
> **摘要:** To address the gap in humanoid robot cognitive systems regarding the lack of a time-updable mediating thought space between semantics and continuous control, this study constructs and trains a VLA model named "Sigma" that runs on a single RTX 4090. It uses the open-source pi05_base model as a foundation and preprocesses svla_so101_pickplace into a training dataset. The researcher independently designed an architecture for a vision-language-action model that combines deep semantic understanding and association to achieve telepathic communication. The training process involved repeated optimizations of data preprocessing, LoRA fine-tuning, and the inference-stage adapter. The experiment employed offline closed-loop replay, comparing Sigma with the untuned pure pi05_base model under data conditions. Results showed that Sigma exhibited a stable decrease in control MSE across vector, fragment, and entire trajectory timescales, while maintaining the telepathy norm and semantic-text alignment quality unchanged. It demonstrates that mind-responsive alignment control is quantified through an architecture that combines deep understanding of semantics and association without retraining the base model, which provides reproducible experience for semantic alignment and intention-driven behavior in humanoid robots.
>
---
#### [replaced 021] DYNEMO-SLAM: Dynamic Entity and Motion-Aware 3D Scene Graph SLAM
- **分类: cs.RO**

- **简介: 该论文提出DYNEMO-SLAM，一种基于3D场景图的SLAM框架，旨在解决动态环境中传统SLAM因忽略动态物体而导致精度下降的问题。通过引入语义运动先验与动态实体感知约束，联合优化机器人轨迹、动态物体位姿及环境结构，实现对动态场景的鲁棒建图与定位，显著提升复杂环境下的性能。**

- **链接: [https://arxiv.org/pdf/2503.02050v2](https://arxiv.org/pdf/2503.02050v2)**

> **作者:** Marco Giberna; Muhammad Shaheer; Miguel Fernandez-Cortizas; Jose Andres Millan-Romera; Jose Luis Sanchez-Lopez; Holger Voos
>
> **备注:** 8 pages, 4 figures, 5 tables
>
> **摘要:** Robots operating in dynamic environments face significant challenges due to the presence of moving agents and displaced objects. Traditional SLAM systems typically assume a static world or treat dynamic as outliers, discarding their information to preserve map consistency. As a result, they cannot exploit dynamic entities as persistent landmarks, do not model and exploit their motion over time, and therefore quickly degrade in highly cluttered environments with few reliable static features. This paper presents a novel 3D scene graph-based SLAM framework that addresses the challenge of modeling and estimating the pose of dynamic entities into the SLAM backend. Our framework incorporates semantic motion priors and dynamic entity-aware constraints to jointly optimize the robot trajectory, dynamic entity poses, and the surrounding environment structure within a unified graph formulation. In parallel, a dynamic keyframe selection policy and a semantic loop-closure prefiltering step enable the system to remain robust and effective in highly dynamic environments by continuously adapting to scene changes and filtering inconsistent observations. The simulation and real-world experimental results show a 49.97% reduction in ATE compared to the baseline method employed, demonstrating the effectiveness of incorporating dynamic entities and estimating their poses for improved robustness and richer scene representation in complex scenarios while maintaining real-time performance.
>
---
#### [replaced 022] GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出GR-RL框架，针对长时程精细操作中人类示范噪声与次优问题，通过离线强化学习筛选有效轨迹、引入形态对称增强泛化性，并在线学习噪声预测以提升精度。首次实现83.3%成功率的自主系鞋带任务。**

- **链接: [https://arxiv.org/pdf/2512.01801v2](https://arxiv.org/pdf/2512.01801v2)**

> **作者:** Yunfei Li; Xiao Ma; Jiafeng Xu; Yu Cui; Zhongren Cui; Zhigang Han; Liqun Huang; Tao Kong; Yuxiao Liu; Hao Niu; Wanli Peng; Jingchao Qiao; Zeyu Ren; Haixin Shi; Zhi Su; Jiawen Tian; Yuyang Xiao; Shenyu Zhang; Liwei Zheng; Hang Li; Yonghui Wu
>
> **摘要:** We present GR-RL, a robotic learning framework that turns a generalist vision-language-action (VLA) policy into a highly capable specialist for long-horizon dexterous manipulation. Assuming the optimality of human demonstrations is core to existing VLA policies. However, we claim that in highly dexterous and precise manipulation tasks, human demonstrations are noisy and suboptimal. GR-RL proposes a multi-stage training pipeline that filters, augments, and reinforces the demonstrations by reinforcement learning. First, GR-RL learns a vision-language-conditioned task progress, filters the demonstration trajectories, and only keeps the transitions that contribute positively to the progress. Specifically, we show that by directly applying offline RL with sparse reward, the resulting $Q$-values can be treated as a robust progress function. Next, we introduce morphological symmetry augmentation that greatly improves the generalization and performance of GR-RL. Lastly, to better align the VLA policy with its deployment behaviors for high-precision control, we perform online RL by learning a latent space noise predictor. With this pipeline, GR-RL is, to our knowledge, the first learning-based policy that can autonomously lace up a shoe by threading shoelaces through multiple eyelets with an 83.3% success rate, a task requiring long-horizon reasoning, millimeter-level precision, and compliant soft-body interaction. We hope GR-RL provides a step toward enabling generalist robot foundations models to specialize into reliable real-world experts.
>
---
#### [replaced 023] LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LiDARCrafter，面向动态4D LiDAR场景生成与编辑任务。针对现有方法在可控性、时序一致性及评估标准上的不足，构建基于语言指令的统一框架，通过三分支扩散网络生成物体结构、运动轨迹与几何，并结合自回归模块实现时序连贯生成。建立标准化评估基准，实现在nuScenes数据集上的先进性能。**

- **链接: [https://arxiv.org/pdf/2508.03692v3](https://arxiv.org/pdf/2508.03692v3)**

> **作者:** Ao Liang; Youquan Liu; Yu Yang; Dongyue Lu; Linfeng Li; Lingdong Kong; Huaici Zhao; Wei Tsang Ooi
>
> **备注:** AAAI 2026 Oral Presentation; 38 pages, 18 figures, 12 tables; Project Page at https://lidarcrafter.github.io
>
> **摘要:** Generative world models have become essential data engines for autonomous driving, yet most existing efforts focus on videos or occupancy grids, overlooking the unique LiDAR properties. Extending LiDAR generation to dynamic 4D world modeling presents challenges in controllability, temporal coherence, and evaluation standardization. To this end, we present LiDARCrafter, a unified framework for 4D LiDAR generation and editing. Given free-form natural language inputs, we parse instructions into ego-centric scene graphs, which condition a tri-branch diffusion network to generate object structures, motion trajectories, and geometry. These structured conditions enable diverse and fine-grained scene editing. Additionally, an autoregressive module generates temporally coherent 4D LiDAR sequences with smooth transitions. To support standardized evaluation, we establish a comprehensive benchmark with diverse metrics spanning scene-, object-, and sequence-level aspects. Experiments on the nuScenes dataset using this benchmark demonstrate that LiDARCrafter achieves state-of-the-art performance in fidelity, controllability, and temporal consistency across all levels, paving the way for data augmentation and simulation. The code and benchmark are released to the community.
>
---
#### [replaced 024] Predicting Human Perceptions of Robot Performance During Navigation Tasks
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于人机交互中的感知预测任务，旨在无需中断交互地预测人类对机器人导航性能的感知。研究构建了SEAN TOGETHER数据集，分析非语言行为（如面部表情、空间行为）与感知的关系，发现空间特征对预测至关重要，且机器学习模型在二分类上显著优于人类，具备良好泛化能力。**

- **链接: [https://arxiv.org/pdf/2310.11590v3](https://arxiv.org/pdf/2310.11590v3)**

> **作者:** Qiping Zhang; Nathan Tsoi; Mofeed Nagib; Booyeon Choi; Jie Tan; Hao-Tien Lewis Chiang; Marynel Vázquez
>
> **摘要:** Understanding human perceptions of robot performance is crucial for designing socially intelligent robots that can adapt to human expectations. Current approaches often rely on surveys, which can disrupt ongoing human-robot interactions. As an alternative, we explore predicting people's perceptions of robot performance using non-verbal behavioral cues and machine learning techniques. We contribute the SEAN TOGETHER Dataset consisting of observations of an interaction between a person and a mobile robot in Virtual Reality, together with perceptions of robot performance provided by users on a 5-point scale. We then analyze how well humans and supervised learning techniques can predict perceived robot performance based on different observation types (like facial expression and spatial behavior features). Our results suggest that facial expressions alone provide useful information, but in the navigation scenarios that we considered, reasoning about spatial features in context is critical for the prediction task. Also, supervised learning techniques outperformed humans' predictions in most cases. Further, when predicting robot performance as a binary classification task on unseen users' data, the F1-Score of machine learning models more than doubled that of predictions on a 5-point scale. This suggested good generalization capabilities, particularly in identifying performance directionality over exact ratings. Based on these findings, we conducted a real-world demonstration where a mobile robot uses a machine learning model to predict how a human who follows it perceives it. Finally, we discuss the implications of our results for implementing these supervised learning models in real-world navigation. Our work paves the path to automatically enhancing robot behavior based on observations of users and inferences about their perceptions of a robot.
>
---
#### [replaced 025] Learning Massively Multitask World Models for Continuous Control
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文研究连续控制中的多任务强化学习问题，旨在实现单一智能体在数百个任务间的高效适应。提出新基准与语言条件的多任务世界模型Newt，通过演示预训练和在线联合优化，提升数据效率与泛化能力，支持快速零样本迁移。**

- **链接: [https://arxiv.org/pdf/2511.19584v2](https://arxiv.org/pdf/2511.19584v2)**

> **作者:** Nicklas Hansen; Hao Su; Xiaolong Wang
>
> **备注:** Webpage: https://www.nicklashansen.com/NewtWM
>
> **摘要:** General-purpose control demands agents that act across many tasks and embodiments, yet research on reinforcement learning (RL) for continuous control remains dominated by single-task or offline regimes, reinforcing a view that online RL does not scale. Inspired by the foundation model recipe (large-scale pretraining followed by light RL) we ask whether a single agent can be trained on hundreds of tasks with online interaction. To accelerate research in this direction, we introduce a new benchmark with 200 diverse tasks spanning many domains and embodiments, each with language instructions, demonstrations, and optionally image observations. We then present \emph{Newt}, a language-conditioned multitask world model that is first pretrained on demonstrations to acquire task-aware representations and action priors, and then jointly optimized with online interaction across all tasks. Experiments show that Newt yields better multitask performance and data-efficiency than a set of strong baselines, exhibits strong open-loop control, and enables rapid adaptation to unseen tasks. We release our environments, demonstrations, code for training and evaluation, as well as 200+ checkpoints.
>
---
