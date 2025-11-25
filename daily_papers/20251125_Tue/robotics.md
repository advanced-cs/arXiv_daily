# 机器人 cs.RO

- **最新发布 84 篇**

- **更新 38 篇**

## 最新发布

#### [new 001] EchoVLA: Robotic Vision-Language-Action Model with Synergistic Declarative Memory for Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文针对长时序移动操作任务中智能体缺乏记忆与推理能力的问题，提出EchoVLA模型。通过融合场景记忆与情景记忆的协同声明式记忆机制，增强对空间语义与任务经验的建模，并结合多模态注意力引导扩散策略。构建MoMani基准实现大规模训练评估，显著提升仿真与真实场景下的长时序操作性能。**

- **链接: [https://arxiv.org/pdf/2511.18112v1](https://arxiv.org/pdf/2511.18112v1)**

> **作者:** Min Lin; Xiwen Liang; Bingqian Lin; Liu Jingzhi; Zijian Jiao; Kehan Li; Yuhan Ma; Yuecheng Liu; Shen Zhao; Yuzheng Zhuang; Xiaodan Liang
>
> **摘要:** Recent progress in Vision-Language-Action (VLA) models has enabled embodied agents to interpret multimodal instructions and perform complex tasks. However, existing VLAs are mostly confined to short-horizon, table-top manipulation, lacking the memory and reasoning capability required for long-horizon mobile manipulation, where agents must coordinate navigation and manipulation under changing spatial contexts. In this work, we present EchoVLA, a memory-aware VLA model for long-horizon mobile manipulation. EchoVLA incorporates a synergistic declarative memory inspired by the human brain, consisting of a scene memory that maintains a collection of spatial-semantic maps and an episodic memory that stores task-level experiences with multimodal contextual features. During both training and inference, the two memories are individually stored, updated, and retrieved based on current observations, task history, and instructions, and their retrieved representations are fused via coarse- and fine-grained attention to guide mobile-arm diffusion policies. To support large-scale training and evaluation, we further introduce MoMani, an automated benchmark that generates expert-level long-horizon trajectories through multimodal large language model (MLLM)-guided planning and feedback-driven refinement, supplemented with real-robot demonstrations. Experiments in simulated and real-world settings show that EchoVLA improves long-horizon performance, reaching 0.52 SR on manipulation/navigation and 0.31 on mobile manipulation, exceeding $π_{0.5}$ by +0.08 and +0.11.
>
---
#### [new 002] SafeFall: Learning Protective Control for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文针对双足机器人易跌倒导致硬件损坏的问题，提出SafeFall框架。通过轻量级GRU跌倒预测器与强化学习保护策略，实现跌倒前的主动防护。在全尺寸机器人上验证，显著降低冲击力与关节扭矩，减少99.3%脆弱部件碰撞，提升安全性与部署可行性。**

- **链接: [https://arxiv.org/pdf/2511.18509v1](https://arxiv.org/pdf/2511.18509v1)**

> **作者:** Ziyu Meng; Tengyu Liu; Le Ma; Yingying Wu; Ran Song; Wei Zhang; Siyuan Huang
>
> **摘要:** Bipedal locomotion makes humanoid robots inherently prone to falls, causing catastrophic damage to the expensive sensors, actuators, and structural components of full-scale robots. To address this critical barrier to real-world deployment, we present \method, a framework that learns to predict imminent, unavoidable falls and execute protective maneuvers to minimize hardware damage. SafeFall is designed to operate seamlessly alongside existing nominal controller, ensuring no interference during normal operation. It combines two synergistic components: a lightweight, GRU-based fall predictor that continuously monitors the robot's state, and a reinforcement learning policy for damage mitigation. The protective policy remains dormant until the predictor identifies a fall as unavoidable, at which point it activates to take control and execute a damage-minimizing response. This policy is trained with a novel, damage-aware reward function that incorporates the robot's specific structural vulnerabilities, learning to shield critical components like the head and hands while absorbing energy with more robust parts of its body. Validated on a full-scale Unitree G1 humanoid, SafeFall demonstrated significant performance improvements over unprotected falls. It reduced peak contact forces by 68.3\%, peak joint torques by 78.4\%, and eliminated 99.3\% of collisions with vulnerable components. By enabling humanoids to fail safely, SafeFall provides a crucial safety net that allows for more aggressive experiments and accelerates the deployment of these robots in complex, real-world environments.
>
---
#### [new 003] AutoOdom: Learning Auto-regressive Proprioceptive Odometry for Legged Locomotion
- **分类: cs.RO**

- **简介: 该论文针对腿式机器人在无GPS、视觉受限环境下的精准本体感知里程计问题，提出AutoOdom系统。通过两阶段训练：先用仿真数据学习复杂动态，再用少量真实数据进行自回归优化，提升模型对传感器噪声的鲁棒性与仿真实现到现实的迁移能力，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.18857v1](https://arxiv.org/pdf/2511.18857v1)**

> **作者:** Changsheng Luo; Yushi Wang; Wenhan Cai; Mingguo Zhao
>
> **摘要:** Accurate proprioceptive odometry is fundamental for legged robot navigation in GPS-denied and visually degraded environments where conventional visual odometry systems fail. Current approaches face critical limitations: analytical filtering methods suffer from modeling uncertainties and cumulative drift, hybrid learning-filtering approaches remain constrained by their analytical components, while pure learning-based methods struggle with simulation-to-reality transfer and demand extensive real-world data collection. This paper introduces AutoOdom, a novel autoregressive proprioceptive odometry system that overcomes these challenges through an innovative two-stage training paradigm. Stage 1 employs large-scale simulation data to learn complex nonlinear dynamics and rapidly changing contact states inherent in legged locomotion, while Stage 2 introduces an autoregressive enhancement mechanism using limited real-world data to effectively bridge the sim-to-real gap. The key innovation lies in our autoregressive training approach, where the model learns from its own predictions to develop resilience against sensor noise and improve robustness in highly dynamic environments. Comprehensive experimental validation on the Booster T1 humanoid robot demonstrates that AutoOdom significantly outperforms state-of-the-art methods across all evaluation metrics, achieving 57.2% improvement in absolute trajectory error, 59.2% improvement in Umeyama-aligned error, and 36.2% improvement in relative pose error compared to the Legolas baseline. Extensive ablation studies provide critical insights into sensor modality selection and temporal modeling, revealing counterintuitive findings about IMU acceleration data and validating our systematic design choices for robust proprioceptive odometry in challenging locomotion scenarios.
>
---
#### [new 004] Skypilot: Fine-Tuning LLM with Physical Grounding for AAV Coverage Search
- **分类: cs.RO**

- **简介: 该论文针对自主飞行器（AAV）覆盖搜索任务，解决大模型缺乏物理实体约束导致的推理幻觉与不可复现问题。提出Skypilot框架，通过蒙特卡洛树搜索（MCTS）实现语言模型物理接地，构建含生成、重生成等操作的多样化动作空间，并基于物理反馈优化奖励函数。进一步在2.3万条MCTS样本上微调Qwen3-4B，显著提升推理效率与解质量。**

- **链接: [https://arxiv.org/pdf/2511.18270v1](https://arxiv.org/pdf/2511.18270v1)**

> **作者:** Zhongkai Chen; Yihao Sun; Chao Yan; Han Zhou; Xiaojia Xiang; Jie Jiang
>
> **摘要:** Autonomous aerial vehicles (AAVs) have played a pivotal role in coverage operations and search missions. Recent advances in large language models (LLMs) offer promising opportunities to augment AAV intelligence. These advances help address complex challenges like area coverage optimization, dynamic path planning, and adaptive decision-making. However, the absence of physical grounding in LLMs leads to hallucination and reproducibility problems in spatial reasoning and decision-making. To tackle these issues, we present Skypilot, an LLM-enhanced two-stage framework that grounds language models in physical reality by integrating monte carlo tree search (MCTS). In the first stage, we introduce a diversified action space that encompasses generate, regenerate, fine-tune, and evaluate operations, coupled with physics-informed reward functions to ensure trajectory feasibility. In the second stage, we fine-tune Qwen3-4B on 23,000 MCTS-generated samples, achieving substantial inference acceleration while maintaining solution quality. Extensive numerical simulations and real-world flight experiments validate the efficiency and superiority of our proposed approach. Detailed information and experimental results are accessible at https://sky-pilot.top.
>
---
#### [new 005] SENTINEL: A Fully End-to-End Language-Action Model for Humanoid Whole Body Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出SENTINEL，一个端到端的人形机器人全身控制语言-动作模型。针对现有系统语言理解与物理执行脱节的问题，构建仿真运动数据集，直接将语言指令和本体感知映射为低层动作，通过流匹配生成动作块并可微调，实现语义理解强、执行稳定，支持多模态输入。**

- **链接: [https://arxiv.org/pdf/2511.19236v1](https://arxiv.org/pdf/2511.19236v1)**

> **作者:** Yuxuan Wang; Haobin Jiang; Shiqing Yao; Ziluo Ding; Zongqing Lu
>
> **备注:** 23 pages, 8 figures, 11 tables
>
> **摘要:** Existing humanoid control systems often rely on teleoperation or modular generation pipelines that separate language understanding from physical execution. However, the former is entirely human-driven, and the latter lacks tight alignment between language commands and physical behaviors. In this paper, we present SENTINEL, a fully end-to-end language-action model for humanoid whole-body control. We construct a large-scale dataset by tracking human motions in simulation using a pretrained whole body controller, combined with their text annotations. The model directly maps language commands and proprioceptive inputs to low-level actions without any intermediate representation. The model generates action chunks using flow matching, which can be subsequently refined by a residual action head for real-world deployment. Our method exhibits strong semantic understanding and stable execution on humanoid robots in both simulation and real-world deployment, and also supports multi-modal extensions by converting inputs into texts.
>
---
#### [new 006] How to Train Your Latent Control Barrier Function: Smooth Safety Filtering Under Hard-to-Model Constraints
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对视觉运动控制中的安全约束问题，提出LatentCBF方法。解决现有方法因离散切换导致性能下降的问题，通过平滑边际函数和融合双策略数据训练，实现连续安全过滤，显著提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2511.18606v1](https://arxiv.org/pdf/2511.18606v1)**

> **作者:** Kensuke Nakamura; Arun L. Bishop; Steven Man; Aaron M. Johnson; Zachary Manchester; Andrea Bajcsy
>
> **备注:** 3 figures, 10 tables, 22 pages
>
> **摘要:** Latent safety filters extend Hamilton-Jacobi (HJ) reachability to operate on latent state representations and dynamics learned directly from high-dimensional observations, enabling safe visuomotor control under hard-to-model constraints. However, existing methods implement "least-restrictive" filtering that discretely switch between nominal and safety policies, potentially undermining the task performance that makes modern visuomotor policies valuable. While reachability value functions can, in principle, be adapted to be control barrier functions (CBFs) for smooth optimization-based filtering, we theoretically and empirically show that current latent-space learning methods produce fundamentally incompatible value functions. We identify two sources of incompatibility: First, in HJ reachability, failures are encoded via a "margin function" in latent space, whose sign indicates whether or not a latent is in the constraint set. However, representing the margin function as a classifier yields saturated value functions that exhibit discontinuous jumps. We prove that the value function's Lipschitz constant scales linearly with the margin function's Lipschitz constant, revealing that smooth CBFs require smooth margins. Second, reinforcement learning (RL) approximations trained solely on safety policy data yield inaccurate value estimates for nominal policy actions, precisely where CBF filtering needs them. We propose the LatentCBF, which addresses both challenges through gradient penalties that lead to smooth margin functions without additional labeling, and a value-training procedure that mixes data from both nominal and safety policy distributions. Experiments on simulated benchmarks and hardware with a vision-based manipulation policy demonstrate that LatentCBF enables smooth safety filtering while doubling the task-completion rate over prior switching methods.
>
---
#### [new 007] Mixture of Horizons in Action Chunking
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对视觉-语言-动作模型在机器人操作中因固定动作时域（horizon）导致的长期规划与精细控制权衡问题，提出混合时域（MoH）策略。通过并行处理多时域动作片段并动态融合，实现长程前瞻与短程精度兼顾，显著提升复杂任务性能与推理效率。**

- **链接: [https://arxiv.org/pdf/2511.19433v1](https://arxiv.org/pdf/2511.19433v1)**

> **作者:** Dong Jing; Gang Wang; Jiaqi Liu; Weiliang Tang; Zelong Sun; Yunchao Yao; Zhenyu Wei; Yunhui Liu; Zhiwu Lu; Mingyu Ding
>
> **备注:** 15 pages, 14 figures
>
> **摘要:** Vision-language-action (VLA) models have shown remarkable capabilities in robotic manipulation, but their performance is sensitive to the $\textbf{action chunk length}$ used during training, termed $\textbf{horizon}$. Our empirical study reveals an inherent trade-off: longer horizons provide stronger global foresight but degrade fine-grained accuracy, while shorter ones sharpen local control yet struggle on long-term tasks, implying fixed choice of single horizons being suboptimal. To mitigate the trade-off, we propose a $\textbf{mixture of horizons (MoH)}$ strategy. MoH rearranges the action chunk into several segments with different horizons, processes them in parallel with a shared action transformer, and fuses outputs with a light linear gate. It has three appealing benefits. 1) MoH exploits long-term foresight and short-term precision jointly within a single model, improving both performance and generalizability to complex tasks. 2) MoH is plug-and-play for full-attention action modules with minimal training or inference overhead. 3) MoH enables dynamic inference with adaptive horizons, which selects stable actions through cross-horizon consensus, achieving 2.5$\times$ higher throughput than baselines while preserving superior performance. Extensive experiments over flow-based policies $π_0$, $π_{0.5}$, and one-step regression policy $π_{\text{reg}}$ demonstrate that MoH yields consistent and significant gains on both simulations and real-world tasks. Notably, under mixed-task setting, $π_{0.5}$ with MoH reaches a new state-of-the-art with 99$\%$ average success rate on LIBERO after only $30k$ training iterations. Project page: https://github.com/Timsty1/MixtureOfHorizons
>
---
#### [new 008] Soft pneumatic grippers: Topology optimization, 3D-printing and experimental validation
- **分类: cs.RO**

- **简介: 该论文针对软体气动夹爪的性能优化问题，提出一种考虑自加载特性的拓扑优化框架。通过结合达西定律与鲁棒优化，设计高效2D柔性单元，并拓展为3D模块，最终实现多单元集成的夹爪。实验验证了其在不同工况下的优异抓取性能。**

- **链接: [https://arxiv.org/pdf/2511.19211v1](https://arxiv.org/pdf/2511.19211v1)**

> **作者:** Prabhat Kumar; Chandra Prakash; Josh Pinskier; David Howard; Matthijs Langelaar
>
> **备注:** 9 Figures
>
> **摘要:** This paper presents a systematic topology optimization framework for designing a soft pneumatic gripper (SPG), explicitly considering the design-dependent nature of the actuating load. The load is modeled using Darcy's law with an added drainage term. A 2D soft arm unit is optimized by formulating it as a compliant mechanism design problem using the robust formulation. The problem is posed as a min-max optimization, where the output deformations of blueprint and eroded designs are considered. A volume constraint is imposed on the blueprint part, while a strain-energy constraint is enforced on the eroded part. The MMA is employed to solve the optimization problem and obtain the optimized soft unit. Finite element analysis with the Ogden material model confirms that the optimized 2D unit outperforms a conventional rectangular design under pneumatic loading. The optimized 2D unit is extruded to obtain a 3D module, and ten such units are assembled to create a soft arm. Deformation profiles of the optimized arm are analysed under different pressure loads. Four arms are 3D-printed and integrated with a supporting structure to realize the proposed SPG. The gripping performance of the SPG is demonstrated on objects with different weights, sizes, stiffness, and shapes.
>
---
#### [new 009] A Coordinated Dual-Arm Framework for Delicate Snap-Fit Assemblies
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对精密卡扣装配中接触检测滞后与冲击力过大问题，提出基于双臂协作的实时响应框架。通过轻量神经网络SnapNet实现仅凭本体感知信号的卡扣到位实时检测，并结合事件触发的阻抗调节策略，提升装配精度与安全性。实验验证了高检测准确率（>96%召回率）与峰值冲击力降低30%的效果。**

- **链接: [https://arxiv.org/pdf/2511.18153v1](https://arxiv.org/pdf/2511.18153v1)**

> **作者:** Shreyas Kumar; Barat S; Debojit Das; Yug Desai; Siddhi Jain; Rajesh Kumar; Harish J. Palanthandalam-Madapusi
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** Delicate snap-fit assemblies, such as inserting a lens into an eye-wear frame or during electronics assembly, demand timely engagement detection and rapid force attenuation to prevent overshoot-induced component damage or assembly failure. We address these challenges with two key contributions. First, we introduce SnapNet, a lightweight neural network that detects snap-fit engagement from joint-velocity transients in real-time, showing that reliable detection can be achieved using proprioceptive signals without external sensors. Second, we present a dynamical-systems-based dual-arm coordination framework that integrates SnapNet driven detection with an event-triggered impedance modulation, enabling accurate alignment and compliant insertion during delicate snap-fit assemblies. Experiments across diverse geometries on a heterogeneous bimanual platform demonstrate high detection accuracy (over 96% recall) and up to a 30% reduction in peak impact forces compared to standard impedance control.
>
---
#### [new 010] Head Stabilization for Wheeled Bipedal Robots via Force-Estimation-Based Admittance Control
- **分类: cs.RO**

- **简介: 该论文针对轮式双足机器人在不平地形上头部不稳定的问题，提出基于力估计的阻抗控制方法。通过模型估算地面反作用力，实现头部在世界坐标系中的主动稳定，提升传感器精度与载荷安全。实验验证了算法实时性与地形适应性。**

- **链接: [https://arxiv.org/pdf/2511.18712v1](https://arxiv.org/pdf/2511.18712v1)**

> **作者:** Tianyu Wang; Chunxiang Yan; Xuanhong Liao; Tao Zhang; Ping Wang; Cong Wen; Dingchuan Liu; Haowen Yu; Ximin Lyu
>
> **摘要:** Wheeled bipedal robots are emerging as flexible platforms for field exploration. However, head instability induced by uneven terrain can degrade the accuracy of onboard sensors or damage fragile payloads. Existing research primarily focuses on stabilizing the mobile platform but overlooks active stabilization of the head in the world frame, resulting in vertical oscillations that undermine overall stability. To address this challenge, we developed a model-based ground force estimation method for our 6-degree-of-freedom wheeled bipedal robot. Leveraging these force estimates, we implemented an admittance control algorithm to enhance terrain adaptability. Simulation experiments validated the real-time performance of the force estimator and the robot's robustness when traversing uneven terrain.
>
---
#### [new 011] Stable Multi-Drone GNSS Tracking System for Marine Robots
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对海洋机器人在水下GNSS信号失效导致定位困难的问题，提出一种基于多无人机的稳定GNSS追踪系统。通过视觉检测、轻量级多目标跟踪与GNSS三角测量融合，结合信心加权EKF和跨无人机ID对齐算法，实现水面及近水面机器人实时、鲁棒的高精度定位，有效克服传统方法误差累积与依赖基础设施的缺陷。**

- **链接: [https://arxiv.org/pdf/2511.18694v1](https://arxiv.org/pdf/2511.18694v1)**

> **作者:** Shuo Wen; Edwin Meriaux; Mariana Sosa Guzmán; Zhizun Wang; Junming Shi; Gregory Dudek
>
> **摘要:** Accurate localization is essential for marine robotics, yet Global Navigation Satellite System (GNSS) signals are unreliable or unavailable even at a very short distance below the water surface. Traditional alternatives, such as inertial navigation, Doppler Velocity Loggers (DVL), SLAM, and acoustic methods, suffer from error accumulation, high computational demands, or infrastructure dependence. In this work, we present a scalable multi-drone GNSS-based tracking system for surface and near-surface marine robots. Our approach combines efficient visual detection, lightweight multi-object tracking, GNSS-based triangulation, and a confidence-weighted Extended Kalman Filter (EKF) to provide stable GNSS estimation in real time. We further introduce a cross-drone tracking ID alignment algorithm that enforces global consistency across views, enabling robust multi-robot tracking with redundant aerial coverage. We validate our system in diversified complex settings to show the scalability and robustness of the proposed algorithm.
>
---
#### [new 012] Compressor-VLA: Instruction-Guided Visual Token Compression for Efficient Robotic Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在机器人操作中因冗余视觉令牌导致的计算开销问题，提出Compressor-VLA框架。通过指令引导的语义任务压缩与空间细节保留模块，实现高效、任务相关的视觉信息压缩，显著降低计算量并提升实时性，验证了其在仿真到真实场景中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.18950v1](https://arxiv.org/pdf/2511.18950v1)**

> **作者:** Juntao Gao; Feiyang Ye; Jing Zhang; Wenjing Qian
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a powerful paradigm in Embodied AI. However, the significant computational overhead of processing redundant visual tokens remains a critical bottleneck for real-time robotic deployment. While standard token pruning techniques can alleviate this, these task-agnostic methods struggle to preserve task-critical visual information. To address this challenge, simultaneously preserving both the holistic context and fine-grained details for precise action, we propose Compressor-VLA, a novel hybrid instruction-conditioned token compression framework designed for efficient, task-oriented compression of visual information in VLA models. The proposed Compressor-VLA framework consists of two token compression modules: a Semantic Task Compressor (STC) that distills holistic, task-relevant context, and a Spatial Refinement Compressor (SRC) that preserves fine-grained spatial details. This compression is dynamically modulated by the natural language instruction, allowing for the adaptive condensation of task-relevant visual information. Experimentally, extensive evaluations demonstrate that Compressor-VLA achieves a competitive success rate on the LIBERO benchmark while reducing FLOPs by 59% and the visual token count by over 3x compared to its baseline. The real-robot deployments on a dual-arm robot platform validate the model's sim-to-real transferability and practical applicability. Moreover, qualitative analyses reveal that our instruction guidance dynamically steers the model's perceptual focus toward task-relevant objects, thereby validating the effectiveness of our approach.
>
---
#### [new 013] CNN-Based Camera Pose Estimation and Localisation of Scan Images for Aircraft Visual Inspection
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对飞机外部视觉检测中的相机位姿估计与图像定位问题，提出一种无需基础设施、可现场部署的基于CNN的方法。通过域随机化生成合成数据并微调网络，结合飞机几何信息优化损失函数，实现高精度位姿估计（误差<0.24m，<2°），并设计完整扫描流程，适用于受限环境下的自动化检测。**

- **链接: [https://arxiv.org/pdf/2511.18702v1](https://arxiv.org/pdf/2511.18702v1)**

> **作者:** Xueyan Oh; Leonard Loh; Shaohui Foong; Zhong Bao Andy Koh; Kow Leong Ng; Poh Kang Tan; Pei Lin Pearlin Toh; U-Xuan Tan
>
> **备注:** 12 pages, 12 figures
>
> **摘要:** General Visual Inspection is a manual inspection process regularly used to detect and localise obvious damage on the exterior of commercial aircraft. There has been increasing demand to perform this process at the boarding gate to minimise the downtime of the aircraft and automating this process is desired to reduce the reliance on human labour. Automating this typically requires estimating a camera's pose with respect to the aircraft for initialisation but most existing localisation methods require infrastructure, which is very challenging in uncontrolled outdoor environments and within the limited turnover time (approximately 2 hours) on an airport tarmac. Additionally, many airlines and airports do not allow contact with the aircraft's surface or using UAVs for inspection between flights, and restrict access to commercial aircraft. Hence, this paper proposes an on-site method that is infrastructure-free and easy to deploy for estimating a pan-tilt-zoom camera's pose and localising scan images. This method initialises using the same pan-tilt-zoom camera used for the inspection task by utilising a Deep Convolutional Neural Network fine-tuned on only synthetic images to predict its own pose. We apply domain randomisation to generate the dataset for fine-tuning the network and modify its loss function by leveraging aircraft geometry to improve accuracy. We also propose a workflow for initialisation, scan path planning, and precise localisation of images captured from a pan-tilt-zoom camera. We evaluate and demonstrate our approach through experiments with real aircraft, achieving root-mean-square camera pose estimation errors of less than 0.24 m and 2 degrees for all real scenes.
>
---
#### [new 014] Asynchronous Distributed Multi-Robot Motion Planning Under Imperfect Communication
- **分类: cs.RO**

- **简介: 该论文针对多机器人系统在通信延迟下的协同运动规划问题，提出一种考虑时延的分布式优化方法（DA-ADMM）。通过动态调整惩罚参数以应对延迟信息，提升算法在复杂环境中的鲁棒性与成功率，显著优于传统固定参数或残差平衡方法。**

- **链接: [https://arxiv.org/pdf/2511.18703v1](https://arxiv.org/pdf/2511.18703v1)**

> **作者:** Ardalan Tajbakhsh; Augustinos Saravanos; James Zhu; Evangelos A. Theodorou; Lorenz T. Biegler; Aaron M. Johnson
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** This paper addresses the challenge of coordinating multi-robot systems under realistic communication delays using distributed optimization. We focus on consensus ADMM as a scalable framework for generating collision-free, dynamically feasible motion plans in both trajectory optimization and receding-horizon control settings. In practice, however, these algorithms are sensitive to penalty tuning or adaptation schemes (e.g. residual balancing and adaptive parameter heuristics) that do not explicitly consider delays. To address this, we introduce a Delay-Aware ADMM (DA-ADMM) variant that adapts penalty parameters based on real-time delay statistics, allowing agents to down-weight stale information and prioritize recent updates during consensus and dual updates. Through extensive simulations in 2D and 3D environments with double-integrator, Dubins-car, and drone dynamics, we show that DA-ADMM significantly improves robustness, success rate, and solution quality compared to fixed-parameter, residual-balancing, and fixed-constraint baselines. Our results highlight that performance degradation is not solely determined by delay length or frequency, but by the optimizer's ability to contextually reason over delayed information. The proposed DA-ADMM achieves consistently better coordination performance across a wide range of delay conditions, offering a principled and efficient mechanism for resilient multi-robot motion planning under imperfect communication.
>
---
#### [new 015] Vision-Guided Optic Flow Navigation for Small Lunar Missions
- **分类: cs.RO; astro-ph.IM**

- **简介: 该论文针对小规模月球任务中受限的资源条件，提出一种基于视觉光流与激光测距的轻量级自主导航方法。通过融合平面与球面地形模型，利用稀疏光流和最小二乘法实现高精度运动估计，在真实感月球图像上验证了其在复杂与典型地形下的低误差表现，适用于实时嵌入式系统。**

- **链接: [https://arxiv.org/pdf/2511.17720v1](https://arxiv.org/pdf/2511.17720v1)**

> **作者:** Sean Cowan; Pietro Fanti; Leon B. S. Williams; Chit Hong Yam; Kaneyasu Asakuma; Yuichiro Nada; Dario Izzo
>
> **摘要:** Private lunar missions are faced with the challenge of robust autonomous navigation while operating under stringent constraints on mass, power, and computational resources. This work proposes a motion-field inversion framework that uses optical flow and rangefinder-based depth estimation as a lightweight CPU-based solution for egomotion estimation during lunar descent. We extend classical optical flow formulations by integrating them with depth modeling strategies tailored to the geometry for lunar/planetary approach, descent, and landing, specifically, planar and spherical terrain approximations parameterized by a laser rangefinder. Motion field inversion is performed through a least-squares framework, using sparse optical flow features extracted via the pyramidal Lucas-Kanade algorithm. We verify our approach using synthetically generated lunar images over the challenging terrain of the lunar south pole, using CPU budgets compatible with small lunar landers. The results demonstrate accurate velocity estimation from approach to landing, with sub-10% error for complex terrain and on the order of 1% for more typical terrain, as well as performances suitable for real-time applications. This framework shows promise for enabling robust, lightweight on-board navigation for small lunar missions.
>
---
#### [new 016] AUTOSAR AP and ROS 2 Collaboration Framework
- **分类: cs.RO; cs.SE**

- **简介: 该论文针对自动驾驶领域研究与开发平台脱节问题，提出AUTOSAR AP与ROS 2的协同框架。通过DDS与SOME/IP协议转换，实现两者通信互通，并自动生成功能配置文件，提升集成效率与系统兼容性。**

- **链接: [https://arxiv.org/pdf/2511.17540v1](https://arxiv.org/pdf/2511.17540v1)**

> **作者:** Ryudai Iwakami; Bo Peng; Hiroyuki Hanyu; Tasuku Ishigooka; Takuya Azumi
>
> **备注:** 9 pages. This version includes minor \lstlisting configuration adjustments for successful compilation. The page count is now nine pages due to the addition of author information. There are no other significant changes to the content or layout. Originally published at Euromicro Conference DSD 2024
>
> **摘要:** The field of autonomous vehicle research is advancing rapidly, necessitating platforms that meet real-time performance, safety, and security requirements for practical deployment. AUTOSAR Adaptive Platform (AUTOSAR AP) is widely adopted in development to meet these criteria; however, licensing constraints and tool implementation challenges limit its use in research. Conversely, Robot Operating System 2 (ROS 2) is predominantly used in research within the autonomous driving domain, leading to a disparity between research and development platforms that hinders swift commercialization. This paper proposes a collaboration framework that enables AUTOSAR AP and ROS 2 to communicate with each other using a Data Distribution Service for Real-Time Systems (DDS). In contrast, AUTOSAR AP uses Scalable service-Oriented Middleware over IP (SOME/IP) for communication. The proposed framework bridges these protocol differences, ensuring seamless interaction between the two platforms. We validate the functionality and performance of our bridge converter through empirical analysis, demonstrating its efficiency in conversion time and ease of integration with ROS 2 tools. Furthermore, the availability of the proposed collaboration framework is improved by automatically generating a configuration file for the proposed bridge converter.
>
---
#### [new 017] Accelerating Reinforcement Learning via Error-Related Human Brain Signals
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究如何利用脑电（EEG）中的错误相关电位加速复杂机器人操作任务的强化学习。针对高维操控中稀疏奖励导致学习慢的问题，提出将解码的神经反馈用于奖励塑形，通过实验验证其能显著提升学习效率与成功率，并证明方法对个体差异具有鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.18878v1](https://arxiv.org/pdf/2511.18878v1)**

> **作者:** Suzie Kim; Hye-Bin Shin; Hyo-Jeong Jang
>
> **摘要:** In this work, we investigate how implicit neural feed back can accelerate reinforcement learning in complex robotic manipulation settings. While prior electroencephalogram (EEG) guided reinforcement learning studies have primarily focused on navigation or low-dimensional locomotion tasks, we aim to understand whether such neural evaluative signals can improve policy learning in high-dimensional manipulation tasks involving obstacles and precise end-effector control. We integrate error related potentials decoded from offline-trained EEG classifiers into reward shaping and systematically evaluate the impact of human-feedback weighting. Experiments on a 7-DoF manipulator in an obstacle-rich reaching environment show that neural feedback accelerates reinforcement learning and, depending on the human-feedback weighting, can yield task success rates that at times exceed those of sparse-reward baselines. Moreover, when applying the best-performing feedback weighting across all sub jects, we observe consistent acceleration of reinforcement learning relative to the sparse-reward setting. Furthermore, leave-one subject-out evaluations confirm that the proposed framework remains robust despite the intrinsic inter-individual variability in EEG decodability. Our findings demonstrate that EEG-based reinforcement learning can scale beyond locomotion tasks and provide a viable pathway for human-aligned manipulation skill acquisition.
>
---
#### [new 018] RoboArmGS: High-Quality Robotic Arm Splatting via Bézier Curve Refinement
- **分类: cs.RO**

- **简介: 该论文针对机器人臂数字资产构建中的高精度建模问题，提出RoboArmGS，通过可学习的贝塞尔曲线修正URDF运动偏差，提升真实运动建模与渲染质量。工作包括设计运动修正模块和构建RoboArm4D数据集，实现更准确的动态3D高斯表示。**

- **链接: [https://arxiv.org/pdf/2511.17961v1](https://arxiv.org/pdf/2511.17961v1)**

> **作者:** Hao Wang; Xiaobao Wei; Ying Li; Qingpo Wuwu; Dongli Wu; Jiajun Cao; Ming Lu; Wenzhao Zheng; Shanghang Zhang
>
> **摘要:** Building high-quality digital assets of robotic arms is crucial yet challenging for the Real2Sim2Real pipeline. Current approaches naively bind static 3D Gaussians according to URDF links, forcing them to follow an URDF-rigged motion passively. However, real-world arm motion is noisy, and the idealized URDF-rigged motion cannot accurately model it, leading to severe rendering artifacts in 3D Gaussians. To address these challenges, we propose RoboArmGS, a novel hybrid representation that refines the URDF-rigged motion with learnable Bézier curves, enabling more accurate real-world motion modeling. To be more specific, we present a learnable Bézier Curve motion refiner that corrects per-joint residuals to address mismatches between real-world motion and URDF-rigged motion. RoboArmGS enables the learning of more accurate real-world motion while achieving a coherent binding of 3D Gaussians across arm parts. To support future research, we contribute a carefully collected dataset named RoboArm4D, which comprises several widely used robotic arms for evaluating the quality of building high-quality digital assets. We evaluate our approach on RoboArm4D, and RoboArmGS achieves state-of-the-art performance in real-world motion modeling and rendering quality. The code and dataset will be released.
>
---
#### [new 019] End-to-end Autonomous Vehicle Following System using Monocular Fisheye Camera
- **分类: cs.RO**

- **简介: 该论文提出一种基于单目鱼眼相机的端到端自动驾驶跟车系统，旨在解决传统车队编队依赖车道线和高成本传感器、适用场景受限的问题。通过引入语义掩码缓解多帧数据融合中的因果混淆，并设计动态采样机制精准追踪前车轨迹，实现在多种真实场景下的高效跟车，显著优于传统多阶段方法。**

- **链接: [https://arxiv.org/pdf/2511.19011v1](https://arxiv.org/pdf/2511.19011v1)**

> **作者:** Jiale Zhang; Yeqiang Qian; Tong Qin; Mingyang Jiang; Siyuan Chen; Ming Yang
>
> **摘要:** The increase in vehicle ownership has led to increased traffic congestion, more accidents, and higher carbon emissions. Vehicle platooning is a promising solution to address these issues by improving road capacity and reducing fuel consumption. However, existing platooning systems face challenges such as reliance on lane markings and expensive high-precision sensors, which limits their general applicability. To address these issues, we propose a vehicle following framework that expands its capability from restricted scenarios to general scenario applications using only a camera. This is achieved through our newly proposed end-to-end method, which improves overall driving performance. The method incorporates a semantic mask to address causal confusion in multi-frame data fusion. Additionally, we introduce a dynamic sampling mechanism to precisely track the trajectories of preceding vehicles. Extensive closed-loop validation in real-world vehicle experiments demonstrates the system's ability to follow vehicles in various scenarios, outperforming traditional multi-stage algorithms. This makes it a promising solution for cost-effective autonomous vehicle platooning. A complete real-world vehicle experiment is available at https://youtu.be/zL1bcVb9kqQ.
>
---
#### [new 020] MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent
- **分类: cs.RO**

- **简介: 该论文针对多技能视觉-语言-动作（VLA）模型合并难题，提出MergeVLA架构。通过稀疏激活的LoRA适配器和仅跨注意力块的行动专家设计，解决参数分歧与任务信息扩散问题，实现高效模型合并。在多个机器人任务中表现优于或等同于单任务微调模型，支持跨任务、跨设备泛化。**

- **链接: [https://arxiv.org/pdf/2511.18810v1](https://arxiv.org/pdf/2511.18810v1)**

> **作者:** Yuxia Fu; Zhizhen Zhang; Yuqi Zhang; Zijian Wang; Zi Huang; Yadan Luo
>
> **摘要:** Recent Vision-Language-Action (VLA) models reformulate vision-language models by tuning them with millions of robotic demonstrations. While they perform well when fine-tuned for a single embodiment or task family, extending them to multi-skill settings remains challenging: directly merging VLA experts trained on different tasks results in near-zero success rates. This raises a fundamental question: what prevents VLAs from mastering multiple skills within one model? With an empirical decomposition of learnable parameters during VLA fine-tuning, we identify two key sources of non-mergeability: (1) Finetuning drives LoRA adapters in the VLM backbone toward divergent, task-specific directions beyond the capacity of existing merging methods to unify. (2) Action experts develop inter-block dependencies through self-attention feedback, causing task information to spread across layers and preventing modular recombination. To address these challenges, we present MergeVLA, a merging-oriented VLA architecture that preserves mergeability by design. MergeVLA introduces sparsely activated LoRA adapters via task masks to retain consistent parameters and reduce irreconcilable conflicts in the VLM. Its action expert replaces self-attention with cross-attention-only blocks to keep specialization localized and composable. When the task is unknown, it uses a test-time task router to adaptively select the appropriate task mask and expert head from the initial observation, enabling unsupervised task inference. Across LIBERO, LIBERO-Plus, RoboTwin, and multi-task experiments on the real SO101 robotic arm, MergeVLA achieves performance comparable to or even exceeding individually finetuned experts, demonstrating robust generalization across tasks, embodiments, and environments.
>
---
#### [new 021] A Unified Multi-Dynamics Framework for Perception-Oriented Modeling in Tendon-Driven Continuum Robots
- **分类: cs.RO**

- **简介: 该论文针对腱驱动连续机器人依赖外部传感器导致硬件复杂、难扩展的问题，提出统一多动力学框架，融合电机电气、卷筒与机器人本体动力学，利用内在电机信号实现感知。通过建模与实验验证，实现了接触检测、主动感知及物体尺寸估计，实现了无需外部传感的物理可解释交互感知。**

- **链接: [https://arxiv.org/pdf/2511.18088v1](https://arxiv.org/pdf/2511.18088v1)**

> **作者:** Ibrahim Alsarraj; Yuhao Wang; Abdalla Swikir; Cesare Stefanini; Dezhen Song; Zhanchi Wang; Ke Wu
>
> **摘要:** Tendon-driven continuum robots offer intrinsically safe and contact-rich interactions owing to their kinematic redundancy and structural compliance. However, their perception often depends on external sensors, which increase hardware complexity and limit scalability. This work introduces a unified multi-dynamics modeling framework for tendon-driven continuum robotic systems, exemplified by a spiral-inspired robot named Spirob. The framework integrates motor electrical dynamics, motor-winch dynamics, and continuum robot dynamics into a coherent system model. Within this framework, motor signals such as current and angular displacement are modeled to expose the electromechanical signatures of external interactions, enabling perception grounded in intrinsic dynamics. The model captures and validates key physical behaviors of the real system, including actuation hysteresis and self-contact at motion limits. Building on this foundation, the framework is applied to environmental interaction: first for passive contact detection, verified experimentally against simulation data; then for active contact sensing, where control and perception strategies from simulation are successfully applied to the real robot; and finally for object size estimation, where a policy learned in simulation is directly deployed on hardware. The results demonstrate that the proposed framework provides a physically grounded way to interpret interaction signatures from intrinsic motor signals in tendon-driven continuum robots.
>
---
#### [new 022] Learning Visually Interpretable Oscillator Networks for Soft Continuum Robots from Video
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文针对软连续体机器人动力学建模中数据驱动方法缺乏物理可解释性的难题，提出基于注意力广播解码器（ABCD）的自编码器框架。通过生成像素级注意力图并耦合2D振荡器网络，实现无需先验知识的动力学参数可视化与高精度多步预测，显著提升模型可解释性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.18322v1](https://arxiv.org/pdf/2511.18322v1)**

> **作者:** Henrik Krauss; Johann Licher; Naoya Takeishi; Annika Raatz; Takehisa Yairi
>
> **摘要:** Data-driven learning of soft continuum robot (SCR) dynamics from high-dimensional observations offers flexibility but often lacks physical interpretability, while model-based approaches require prior knowledge and can be computationally expensive. We bridge this gap by introducing (1) the Attention Broadcast Decoder (ABCD), a plug-and-play module for autoencoder-based latent dynamics learning that generates pixel-accurate attention maps localizing each latent dimension's contribution while filtering static backgrounds. (2) By coupling these attention maps to 2D oscillator networks, we enable direct on-image visualization of learned dynamics (masses, stiffness, and forces) without prior knowledge. We validate our approach on single- and double-segment SCRs, demonstrating that ABCD-based models significantly improve multi-step prediction accuracy: 5.7x error reduction for Koopman operators and 3.5x for oscillator networks on the two-segment robot. The learned oscillator network autonomously discovers a chain structure of oscillators. Unlike standard methods, ABCD models enable smooth latent space extrapolation beyond training data. This fully data-driven approach yields compact, physically interpretable models suitable for control applications.
>
---
#### [new 023] See, Plan, Cut: MPC-Based Autonomous Volumetric Robotic Laser Surgery with OCT Guidance
- **分类: cs.RO**

- **简介: 该论文提出RATS系统，解决机器人激光手术中缺乏体积规划与术中反馈的问题。融合OCT与多模态成像，建立高精度激光-组织作用模型，采用基于采样的模型预测控制（MPC）实现闭环自主切割，可实时避让深层结构，提升精度与安全性。**

- **链接: [https://arxiv.org/pdf/2511.17777v1](https://arxiv.org/pdf/2511.17777v1)**

> **作者:** Ravi Prakash; Vincent Y. Wang; Arpit Mishra; Devi Yuliarti; Pei Zhong; Ryan P. McNabb; Patrick J. Codd; Leila J. Bridgeman
>
> **备注:** 9 pages, 8 figures
>
> **摘要:** Robotic laser systems offer the potential for sub-millimeter, non-contact, high-precision tissue resection, yet existing platforms lack volumetric planning and intraoperative feedback. We present RATS (Robot-Assisted Tissue Surgery), an intelligent opto-mechanical, optical coherence tomography (OCT)-guided robotic platform designed for autonomous volumetric soft tissue resection in surgical applications. RATS integrates macro-scale RGB-D imaging, micro-scale OCT, and a fiber-coupled surgical laser, calibrated through a novel multistage alignment pipeline that achieves OCT-to-laser calibration accuracy of 0.161+-0.031mm on tissue phantoms and ex vivo porcine tissue. A super-Gaussian laser-tissue interaction (LTI) model characterizes ablation crater morphology with an average RMSE of 0.231+-0.121mm, outperforming Gaussian baselines. A sampling-based model predictive control (MPC) framework operates directly on OCT voxel data to generate constraint-aware resection trajectories with closed-loop feedback, achieving 0.842mm RMSE and improving intersection-over-union agreement by 64.8% compared to feedforward execution. With OCT, RATS detects subsurface structures and modifies the planner's objective to preserve them, demonstrating clinical feasibility.
>
---
#### [new 024] AIRHILT: A Human-in-the-Loop Testbed for Multimodal Conflict Detection in Aviation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出AIRHILT，一个用于航空多模态冲突检测的人机协同仿真测试平台。针对飞行员与空管协同中多源信息融合难、响应延迟等问题，构建了集成语音、视觉与ADS-B数据的可扩展环境，支持模型快速接入与评估。通过参考流水线验证，实现平均7.7秒首告警，推动航空安全研究的可复现性。**

- **链接: [https://arxiv.org/pdf/2511.18718v1](https://arxiv.org/pdf/2511.18718v1)**

> **作者:** Omar Garib; Jayaprakash D. Kambhampaty; Olivia J. Pinon Fischer; Dimitri N. Mavris
>
> **备注:** 9 pages, 4 figures, 1 table, 1 algorithm
>
> **摘要:** We introduce AIRHILT (Aviation Integrated Reasoning, Human-in-the-Loop Testbed), a modular and lightweight simulation environment designed to evaluate multimodal pilot and air traffic control (ATC) assistance systems for aviation conflict detection. Built on the open-source Godot engine, AIRHILT synchronizes pilot and ATC radio communications, visual scene understanding from camera streams, and ADS-B surveillance data within a unified, scalable platform. The environment supports pilot- and controller-in-the-loop interactions, providing a comprehensive scenario suite covering both terminal area and en route operational conflicts, including communication errors and procedural mistakes. AIRHILT offers standardized JSON-based interfaces that enable researchers to easily integrate, swap, and evaluate automatic speech recognition (ASR), visual detection, decision-making, and text-to-speech (TTS) models. We demonstrate AIRHILT through a reference pipeline incorporating fine-tuned Whisper ASR, YOLO-based visual detection, ADS-B-based conflict logic, and GPT-OSS-20B structured reasoning, and present preliminary results from representative runway-overlap scenarios, where the assistant achieves an average time-to-first-warning of approximately 7.7 s, with average ASR and vision latencies of approximately 5.9 s and 0.4 s, respectively. The AIRHILT environment and scenario suite are openly available, supporting reproducible research on multimodal situational awareness and conflict detection in aviation; code and scenarios are available at https://github.com/ogarib3/airhilt.
>
---
#### [new 025] Expanding the Workspace of Electromagnetic Navigation Systems Using Dynamic Feedback for Single- and Multi-agent Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究磁导航系统的工作空间扩展问题，针对其受功率与热限制的瓶颈，提出基于动态反馈的运动中心控制策略。通过优化电流分配、实时姿态估计与高带宽组件，显著降低所需电流（0.1–0.2 A），实现单/多机器人稳定操控，拓展至50 cm距离，提升临床适用性。**

- **链接: [https://arxiv.org/pdf/2511.18486v1](https://arxiv.org/pdf/2511.18486v1)**

> **作者:** Jasan Zughaibi; Denis von Arx; Maurus Derungs; Florian Heemeyer; Luca A. Antonelli; Quentin Boehler; Michael Muehlebach; Bradley J. Nelson
>
> **摘要:** Electromagnetic navigation systems (eMNS) enable a number of magnetically guided surgical procedures. A challenge in magnetically manipulating surgical tools is that the effective workspace of an eMNS is often severely constrained by power and thermal limits. We show that system-level control design significantly expands this workspace by reducing the currents needed to achieve a desired motion. We identified five key system approaches that enable this expansion: (i) motion-centric torque/force objectives, (ii) energy-optimal current allocation, (iii) real-time pose estimation, (iv) dynamic feedback, and (v) high-bandwidth eMNS components. As a result, we stabilize a 3D inverted pendulum on an eight-coil OctoMag eMNS with significantly lower currents (0.1-0.2 A vs. 8-14 A), by replacing a field-centric field-alignment strategy with a motion-centric torque/force-based approach. We generalize to multi-agent control by simultaneously stabilizing two inverted pendulums within a shared workspace, exploiting magnetic-field nonlinearity and coil redundancy for independent actuation. A structured analysis compares the electromagnetic workspaces of both paradigms and examines current-allocation strategies that map motion objectives to coil currents. Cross-platform evaluation of the clinically oriented Navion eMNS further demonstrates substantial workspace expansion by maintaining stable balancing at distances up to 50 cm from the coils. The results demonstrate that feedback is a practical path to scalable, efficient, and clinically relevant magnetic manipulation.
>
---
#### [new 026] Translating Cultural Choreography from Humanoid Forms to Robotic Arm
- **分类: cs.RO; cs.HC**

- **简介: 该论文针对机器人臂舞蹈中文化语义丢失的问题，提出ROPERA框架，通过符号化姿势编码与解码，实现六自由度机械臂对昆曲《牡丹亭》姿态的精准复现。解决文化动作在跨形态机器人间传递时的语义保真难题，实现可移植的编排工作流。**

- **链接: [https://arxiv.org/pdf/2511.17603v1](https://arxiv.org/pdf/2511.17603v1)**

> **作者:** Chelsea-Xi Chen; Zhe Zhang; Aven-Le Zhou
>
> **摘要:** Robotic arm choreography often reproduces trajectories while missing cultural semantics. This study examines whether symbolic posture transfer with joint space compatible notation can preserve semantic fidelity on a six-degree-of-freedom arm and remain portable across morphologies. We implement ROPERA, a three-stage pipeline for encoding culturally codified postures, composing symbolic sequences, and decoding to servo commands. A scene from Kunqu opera, \textit{The Peony Pavilion}, serves as the material for evaluation. The procedure includes corpus-based posture selection, symbolic scoring, direct joint angle execution, and a visual layer with light painting and costume-informed colors. Results indicate reproducible execution with intended timing and cultural legibility reported by experts and audiences. The study points to non-anthropocentric cultural preservation and portable authoring workflows. Future work will design dance-informed transition profiles, extend the notation to locomotion with haptic, musical, and spatial cues, and test portability across platforms.
>
---
#### [new 027] Switch-JustDance: Benchmarking Whole Body Motion Tracking Policies Using a Commercial Console Game
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Switch-JustDance，一种基于任天堂Switch游戏《舞力全开》的低成本、可复现的人形机器人全身运动控制基准测试方法。针对现有评估缺乏真实场景对比与人类基准的问题，利用游戏实时评分系统量化机器人表现，验证了其可靠性与敏感性，并对三种先进控制器进行实机评测。**

- **链接: [https://arxiv.org/pdf/2511.17925v1](https://arxiv.org/pdf/2511.17925v1)**

> **作者:** Jeonghwan Kim; Wontaek Kim; Yidan Lu; Jin Cheng; Fatemeh Zargarbashi; Zicheng Zeng; Zekun Qi; Zhiyang Dou; Nitish Sontakke; Donghoon Baek; Sehoon Ha; Tianyu Li
>
> **摘要:** Recent advances in whole-body robot control have enabled humanoid and legged robots to perform increasingly agile and coordinated motions. However, standardized benchmarks for evaluating these capabilities in real-world settings, and in direct comparison to humans, remain scarce. Existing evaluations often rely on pre-collected human motion datasets or simulation-based experiments, which limit reproducibility, overlook hardware factors, and hinder fair human-robot comparisons. We present Switch-JustDance, a low-cost and reproducible benchmarking pipeline that leverages motion-sensing console games, Just Dance on the Nintendo Switch, to evaluate robot whole-body control. Using Just Dance on the Nintendo Switch as a representative platform, Switch-JustDance converts in-game choreography into robot-executable motions through streaming, motion reconstruction, and motion retargeting modules and enables users to evaluate controller performance through the game's built-in scoring system. We first validate the evaluation properties of Just Dance, analyzing its reliability, validity, sensitivity, and potential sources of bias. Our results show that the platform provides consistent and interpretable performance measures, making it a suitable tool for benchmarking embodied AI. Building on this foundation, we benchmark three state-of-the-art humanoid whole-body controllers on hardware and provide insights into their relative strengths and limitations.
>
---
#### [new 028] Implicit Neural Field-Based Process Planning for Multi-Axis Manufacturing: Direct Control over Collision Avoidance and Toolpath Geometry
- **分类: cs.RO**

- **简介: 该论文针对多轴制造中路径规划的碰撞避让与轨迹几何控制问题，提出基于隐式神经场的统一可微框架。通过将层生成与刀具路径设计联合优化，实现直接碰撞检测与轨迹形状控制，提升制造精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2511.17578v1](https://arxiv.org/pdf/2511.17578v1)**

> **作者:** Neelotpal Dutta; Tianyu Zhang; Tao Liu; Yongxue Chen; Charlie C. L. Wang
>
> **摘要:** Existing curved-layer-based process planning methods for multi-axis manufacturing address collisions only indirectly and generate toolpaths in a post-processing step, leaving toolpath geometry uncontrolled during optimization. We present an implicit neural field-based framework for multi-axis process planning that overcomes these limitations by embedding both layer generation and toolpath design within a single differentiable pipeline. Using sinusoidally activated neural networks to represent layers and toolpaths as implicit fields, our method enables direct evaluation of field values and derivatives at any spatial point, thereby allowing explicit collision avoidance and joint optimization of manufacturing layers and toolpaths. We further investigate how network hyperparameters and objective definitions influence singularity behavior and topology transitions, offering built-in mechanisms for regularization and stability control. The proposed approach is demonstrated on examples in both additive and subtractive manufacturing, validating its generality and effectiveness.
>
---
#### [new 029] Explicit Bounds on the Hausdorff Distance for Truncated mRPI Sets via Norm-Dependent Contraction Rates
- **分类: cs.RO; eess.SY; math.DS**

- **简介: 该论文研究鲁棒正不变集的截断误差问题，旨在提供首个显式闭式上界来量化有限时域与无限时域最小鲁棒正不变集间的豪斯多夫距离。通过引入依赖范数的收缩率，建立了误差衰减的解析表达式，并证明范数选择可加速收敛，提升模型预测控制等应用的计算效率与精度。**

- **链接: [https://arxiv.org/pdf/2511.18374v1](https://arxiv.org/pdf/2511.18374v1)**

> **作者:** Jiaxun Sun
>
> **摘要:** This paper establishes the first explicit and closed-form upper bound on the Hausdorff distance between the truncated minimal robust positively invariant (mRPI) set and its infinite-horizon limit. While existing mRPI approximations guarantee asymptotic convergence through geometric or norm-based arguments, none provides a computable expression that quantifies the truncation error for a given horizon. We show that the error satisfies \( d_H(\mathcal{E}_N,\mathcal{E}_\infty) \le r_W\,γ^{N+1}/(1-γ), \) where $γ<1$ is the induced-norm contraction factor and $r_W$ depends only on the disturbance set. The bound is fully analytic, requires no iterative set computations, and directly characterizes the decay rate of the truncated Minkowski series. We further demonstrate that the choice of vector norm serves as a design parameter that accelerates convergence, enabling substantially tighter horizon selection for robust invariant-set computations and tube-based MPC. Numerical experiments validate the sharpness, scalability, and practical relevance of the proposed bound.
>
---
#### [new 030] Anti-Jamming based on Null-Steering Antennas and Intelligent UAV Swarm Behavior
- **分类: cs.RO; cs.NI; eess.SY**

- **简介: 该论文研究无人机蜂群在干扰下的通信抗扰问题。针对无线链路易受干扰导致任务失败的难题，提出融合遗传算法、监督学习与强化学习的优化框架，结合零点抑制天线与自适应运动模型，实现动态避障、稳定通信与高效协同，验证了系统在复杂环境下的鲁棒性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.18086v1](https://arxiv.org/pdf/2511.18086v1)**

> **作者:** Miguel Lourenço; António Grilo
>
> **备注:** 10 pages
>
> **摘要:** Unmanned Aerial Vehicle (UAV) swarms represent a key advancement in autonomous systems, enabling coordinated missions through inter-UAV communication. However, their reliance on wireless links makes them vulnerable to jamming, which can disrupt coordination and mission success. This work investigates whether a UAV swarm can effectively overcome jamming while maintaining communication and mission efficiency. To address this, a unified optimization framework combining Genetic Algorithms (GA), Supervised Learning (SL), and Reinforcement Learning (RL) is proposed. The mission model, structured into epochs and timeslots, allows dynamic path planning, antenna orientation, and swarm formation while progressively enforcing collision rules. Null-steering antennas enhance resilience by directing antenna nulls toward interference sources. Results show that the GA achieved stable, collision-free trajectories but with high computational cost. SL models replicated GA-based configurations but struggled to generalize under dynamic or constrained settings. RL, trained via Proximal Policy Optimization (PPO), demonstrated adaptability and real-time decision-making with consistent communication and lower computational demand. Additionally, the Adaptive Movement Model generalized UAV motion to arbitrary directions through a rotation-based mechanism, validating the scalability of the proposed system. Overall, UAV swarms equipped with null-steering antennas and guided by intelligent optimization algorithms effectively mitigate jamming while maintaining communication stability, formation cohesion, and collision safety. The proposed framework establishes a unified, flexible, and reproducible basis for future research on resilient swarm communication systems.
>
---
#### [new 031] APULSE: A Scalable Hybrid Algorithm for the RCSPP on Large-Scale Dense Graphs
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对大规模密集图上的资源约束最短路径问题（RCSPP），提出混合算法APULSE。通过结合A*启发式搜索、脉冲式剪枝与时间分桶策略，显著提升求解效率与鲁棒性，在无人车路径规划等场景中实现近优解的快速计算，有效解决现有方法在大尺度问题上失效的瓶颈。**

- **链接: [https://arxiv.org/pdf/2511.18236v1](https://arxiv.org/pdf/2511.18236v1)**

> **作者:** Nuno Soares; António Grilo
>
> **备注:** 9 pages
>
> **摘要:** The resource-constrained shortest path problem (RCSPP) is a fundamental NP-hard optimization challenge with broad applications, from network routing to autonomous navigation. This problem involves finding a path that minimizes a primary cost subject to a budget on a secondary resource. While various RCSPP solvers exist, they often face critical scalability limitations when applied to the large, dense graphs characteristic of complex, real-world scenarios, making them impractical for time-critical planning. This challenge is particularly acute in domains like mission planning for unmanned ground vehicles (UGVs), which demand solutions on large-scale terrain graphs. This paper introduces APULSE, a hybrid label-setting algorithm designed to efficiently solve the RCSPP on such challenging graphs. APULSE integrates a best-first search guided by an A* heuristic with aggressive, Pulse-style pruning mechanisms and a time-bucketing strategy for effective state-space reduction. A computational study, using a large-scale UGV planning scenario, benchmarks APULSE against state-of-the-art algorithms. The results demonstrate that APULSE consistently finds near-optimal solutions while being orders of magnitude faster and more robust, particularly on large problem instances where competing methods fail. This superior scalability establishes APULSE as an effective solution for RCSPP in complex, large-scale environments, enabling capabilities such as interactive decision support and dynamic replanning.
>
---
#### [new 032] SM$^2$ITH: Safe Mobile Manipulation with Interactive Human Prediction via Task-Hierarchical Bilevel Model Predictive Control
- **分类: cs.RO**

- **简介: 该论文针对移动操作机器人在动态人机交互环境中的安全协同问题，提出SM²ITH框架。通过任务分层双层模型预测控制，融合交互式人类行为预测，实现机器人与人类的实时安全协作。实验验证了其在复杂任务中的优越性。**

- **链接: [https://arxiv.org/pdf/2511.17798v1](https://arxiv.org/pdf/2511.17798v1)**

> **作者:** Francesco D'Orazio; Sepehr Samavi; Xintong Du; Siqi Zhou; Giuseppe Oriolo; Angela P. Schoellig
>
> **摘要:** Mobile manipulators are designed to perform complex sequences of navigation and manipulation tasks in human-centered environments. While recent optimization-based methods such as Hierarchical Task Model Predictive Control (HTMPC) enable efficient multitask execution with strict task priorities, they have so far been applied mainly to static or structured scenarios. Extending these approaches to dynamic human-centered environments requires predictive models that capture how humans react to the actions of the robot. This work introduces Safe Mobile Manipulation with Interactive Human Prediction via Task-Hierarchical Bilevel Model Predictive Control (SM$^2$ITH), a unified framework that combines HTMPC with interactive human motion prediction through bilevel optimization that jointly accounts for robot and human dynamics. The framework is validated on two different mobile manipulators, the Stretch 3 and the Ridgeback-UR10, across three experimental settings: (i) delivery tasks with different navigation and manipulation priorities, (ii) sequential pick-and-place tasks with different human motion prediction models, and (iii) interactions involving adversarial human behavior. Our results highlight how interactive prediction enables safe and efficient coordination, outperforming baselines that rely on weighted objectives or open-loop human models.
>
---
#### [new 033] MobileVLA-R1: Reinforcing Vision-Language-Action for Mobile Robots
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对四足机器人视觉-语言-动作（VLA）任务中语义理解与连续控制难以对齐的问题，提出MobileVLA-R1框架。通过构建多粒度思维链数据集，采用两阶段训练提升推理一致性与控制稳定性，在仿真与真实场景均实现显著性能提升。**

- **链接: [https://arxiv.org/pdf/2511.17889v1](https://arxiv.org/pdf/2511.17889v1)**

> **作者:** Ting Huang; Dongjian Li; Rui Yang; Zeyu Zhang; Zida Yang; Hao Tang
>
> **摘要:** Grounding natural-language instructions into continuous control for quadruped robots remains a fundamental challenge in vision language action. Existing methods struggle to bridge high-level semantic reasoning and low-level actuation, leading to unstable grounding and weak generalization in the real world. To address these issues, we present MobileVLA-R1, a unified vision-language-action framework that enables explicit reasoning and continuous control for quadruped robots. We construct MobileVLA-CoT, a large-scale dataset of multi-granularity chain-of-thought (CoT) for embodied trajectories, providing structured reasoning supervision for alignment. Built upon this foundation, we introduce a two-stage training paradigm that combines supervised CoT alignment with GRPO reinforcement learning to enhance reasoning consistency, control stability, and long-horizon execution. Extensive evaluations on VLN and VLA tasks demonstrate superior performance over strong baselines, with approximately a 5% improvement. Real-world deployment on a quadruped robot validates robust performance in complex environments. Code: https://github.com/AIGeeksGroup/MobileVLA-R1. Website: https://aigeeksgroup.github.io/MobileVLA-R1.
>
---
#### [new 034] Continually Evolving Skill Knowledge in Vision Language Action Model
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对开放环境中机器人智能的持续技能学习问题，提出Stellar VLA框架，通过知识驱动的持续学习机制，实现任务与技能结构的自监督演化。其核心是减少对标注数据和额外参数的依赖，提升模型在复杂任务中的泛化与知识保留能力，显著提升成功率。**

- **链接: [https://arxiv.org/pdf/2511.18085v1](https://arxiv.org/pdf/2511.18085v1)**

> **作者:** Yuxuan Wu; Guangming Wang; Zhiheng Yang; Maoqing Yao; Brian Sheil; Hesheng Wang
>
> **摘要:** Developing general robot intelligence in open environments requires continual skill learning. Recent Vision-Language-Action (VLA) models leverage massive pretraining data to support diverse manipulation tasks, but they still depend heavily on task-specific fine-tuning, revealing a lack of continual learning capability. Existing continual learning methods are also resource-intensive to scale to VLA models. We propose Stellar VLA, a knowledge-driven continual learning framework with two variants: T-Stellar, modeling task-centric knowledge space, and TS-Stellar, capturing hierarchical task-skill structure. Stellar VLA enables self-supervised knowledge evolution through joint learning of task latent representation and the knowledge space, reducing annotation needs. Knowledge-guided expert routing provide task specialization without extra network parameters, lowering training overhead.Experiments on the LIBERO benchmark and real-world tasks show over 50 percentage average improvement in final success rates relative to baselines. TS-Stellar further excels in complex action inference, and in-depth analyses verify effective knowledge retention and discovery. Our code will be released soon.
>
---
#### [new 035] SkillWrapper: Generative Predicate Invention for Skill Abstraction
- **分类: cs.RO**

- **简介: 该论文针对自主智能体在长时序任务中泛化能力不足的问题，提出SkillWrapper方法，通过生成式谓词发明从视觉输入中抽象黑箱技能的符号表示，实现可证明完备的规划。工作包括构建形式化理论与基于基础模型的主动数据收集，使机器人能在真实世界中用抽象表示解决未见的复杂任务。**

- **链接: [https://arxiv.org/pdf/2511.18203v1](https://arxiv.org/pdf/2511.18203v1)**

> **作者:** Ziyi Yang; Benned Hedegaard; Ahmed Jaafar; Yichen Wei; Skye Thompson; Shreyas S. Raman; Haotian Fu; Stefanie Tellex; George Konidaris; David Paulius; Naman Shah
>
> **摘要:** Generalizing from individual skill executions to solving long-horizon tasks remains a core challenge in building autonomous agents. A promising direction is learning high-level, symbolic abstractions of the low-level skills of the agents, enabling reasoning and planning independent of the low-level state space. Among possible high-level representations, object-centric skill abstraction with symbolic predicates has been proven to be efficient because of its compatibility with domain-independent planners. Recent advances in foundation models have made it possible to generate symbolic predicates that operate on raw sensory inputs, a process we call generative predicate invention, to facilitate downstream abstraction learning. However, it remains unclear which formal properties the learned representations must satisfy, and how they can be learned to guarantee these properties. In this paper, we address both questions by presenting a formal theory of generative predicate invention for skill abstraction, resulting in symbolic operators that can be used for provably sound and complete planning. Within this framework, we propose SkillWrapper, a method that leverages foundation models to actively collect robot data and learn human-interpretable, plannable representations of black-box skills, using only RGB image observations. Our extensive empirical evaluation in simulation and on real robots shows that SkillWrapper learns abstract representations that enable solving unseen, long-horizon tasks in the real world with black-box skills.
>
---
#### [new 036] Off-Road Navigation via Implicit Neural Representation of Terrain Traversability
- **分类: cs.RO**

- **简介: 该论文针对自主非结构化环境导航任务，解决传统方法因短视规划与固定速度导致的路径优化不足问题。提出TRAIL框架，利用隐式神经表示连续建模地形可通行性，结合梯度优化实现路径形状与速度曲线的协同调整，提升复杂地形下的导航效率与平稳性。**

- **链接: [https://arxiv.org/pdf/2511.18183v1](https://arxiv.org/pdf/2511.18183v1)**

> **作者:** Yixuan Jia; Qingyuan Li; Jonathan P. How
>
> **备注:** 9 pages
>
> **摘要:** Autonomous off-road navigation requires robots to estimate terrain traversability from onboard sensors and plan accordingly. Conventional approaches typically rely on sampling-based planners such as MPPI to generate short-term control actions that aim to minimize traversal time and risk measures derived from the traversability estimates. These planners can react quickly but optimize only over a short look-ahead window, limiting their ability to reason about the full path geometry, which is important for navigating in challenging off-road environments. Moreover, they lack the ability to adjust speed based on the terrain bumpiness, which is important for smooth navigation on challenging terrains. In this paper, we introduce TRAIL (Traversability with an Implicit Learned Representation), an off-road navigation framework that leverages an implicit neural representation to continuously parameterize terrain properties. This representation yields spatial gradients that enable integration with a novel gradient-based trajectory optimization method that adapts the path geometry and speed profile based on terrain traversability.
>
---
#### [new 037] AIA-UltraNeRF:Acoustic-Impedance-Aware Neural Radiance Field with Hash Encodings for Robotic Ultrasound Reconstruction and Localization
- **分类: cs.RO**

- **简介: 该论文针对机器人超声成像中的重建与定位问题，提出AIA-UltraNeRF模型，融合声阻抗感知与哈希编码，实现高效3D超声地图重建与快速定位。通过双监督网络与离线初始位置检索，提升速度与精度，支持无操作员扫描。**

- **链接: [https://arxiv.org/pdf/2511.18293v1](https://arxiv.org/pdf/2511.18293v1)**

> **作者:** Shuai Zhang; Jingsong Mu; Cancan Zhao; Leiqi Tian; Zhijun Xing; Bo Ouyang; Xiang Li
>
> **摘要:** Neural radiance field (NeRF) is a promising approach for reconstruction and new view synthesis. However, previous NeRF-based reconstruction methods overlook the critical role of acoustic impedance in ultrasound imaging. Localization methods face challenges related to local minima due to the selection of initial poses. In this study, we design a robotic ultrasound system (RUSS) with an acoustic-impedance-aware ultrasound NeRF (AIA-UltraNeRF) to decouple the scanning and diagnostic processes. Specifically, AIA-UltraNeRF models a continuous function of hash-encoded spatial coordinates for the 3D ultrasound map, allowing for the storage of acoustic impedance without dense sampling. This approach accelerates both reconstruction and inference speeds. We then propose a dual-supervised network that leverages teacher and student models to hash-encode the rendered ultrasound images from the reconstructed map. AIA-UltraNeRF retrieves the most similar hash values without the need to render images again, providing an offline initial image position for localization. Moreover, we develop a RUSS with a spherical remote center of motion mechanism to hold the probe, implementing operator-independent scanning modes that separate image acquisition from diagnostic workflows. Experimental results on a phantom and human subjects demonstrate the effectiveness of acoustic impedance in implicitly characterizing the color of ultrasound images. AIAUltraNeRF achieves both reconstruction and localization with inference speeds that are 9.9 faster than those of vanilla NeRF.
>
---
#### [new 038] GVD-TG: Topological Graph based on Fast Hierarchical GVD Sampling for Robot Exploration
- **分类: cs.RO**

- **简介: 该论文针对机器人探索中实时更新高精度拓扑地图的难题，提出基于广义Voronoi图（GVD）的拓扑图构建方法。通过多粒度采样、连通性约束聚类与形态学扩张的前沿提取，提升地图准确性与探索效率，有效避免路径回溯，增强系统灵活性。**

- **链接: [https://arxiv.org/pdf/2511.18708v1](https://arxiv.org/pdf/2511.18708v1)**

> **作者:** Yanbin Li; Canran Xiao; Shenghai Yuan; Peilai Yu; Ziruo Li; Zhiguo Zhang; Wenzheng Chi; Wei Zhang
>
> **备注:** 12 pages, 10 figures
>
> **摘要:** Topological maps are more suitable than metric maps for robotic exploration tasks. However, real-time updating of accurate and detail-rich environmental topological maps remains a challenge. This paper presents a topological map updating method based on the Generalized Voronoi Diagram (GVD). First, the newly observed areas are denoised to avoid low-efficiency GVD nodes misleading the topological structure. Subsequently, a multi-granularity hierarchical GVD generation method is designed to control the sampling granularity at both global and local levels. This not only ensures the accuracy of the topological structure but also enhances the ability to capture detail features, reduces the probability of path backtracking, and ensures no overlap between GVDs through the maintenance of a coverage map, thereby improving GVD utilization efficiency. Second, a node clustering method with connectivity constraints and a connectivity method based on a switching mechanism are designed to avoid the generation of unreachable nodes and erroneous nodes caused by obstacle attraction. A special cache structure is used to store all connectivity information, thereby improving exploration efficiency. Finally, to address the issue of frontiers misjudgment caused by obstacles within the scope of GVD units, a frontiers extraction method based on morphological dilation is designed to effectively ensure the reachability of frontiers. On this basis, a lightweight cost function is used to assess and switch to the next viewpoint in real time. This allows the robot to quickly adjust its strategy when signs of path backtracking appear, thereby escaping the predicament and increasing exploration flexibility. And the performance of system for exploration task is verified through comparative tests with SOTA methods.
>
---
#### [new 039] AFT: Appearance-Based Feature Tracking for Markerless and Training-Free Shape Reconstruction of Soft Robots
- **分类: cs.RO**

- **简介: 该论文针对软体机器人形状重建任务，解决现有视觉方法依赖标记、训练数据或复杂设备的问题。提出一种基于表面外观的无标记、无训练框架，利用自然表面特征实现分层匹配，实现实时、鲁棒的形状跟踪，可在多种环境下稳定运行，提升部署灵活性与成本效益。**

- **链接: [https://arxiv.org/pdf/2511.18215v1](https://arxiv.org/pdf/2511.18215v1)**

> **作者:** Shangyuan Yuan; Preston Fairchild; Yu Mei; Xinyu Zhou; Xiaobo Tan
>
> **摘要:** Accurate shape reconstruction is essential for precise control and reliable operation of soft robots. Compared to sensor-based approaches, vision-based methods offer advantages in cost, simplicity, and ease of deployment. However, existing vision-based methods often rely on complex camera setups, specific backgrounds, or large-scale training datasets, limiting their practicality in real-world scenarios. In this work, we propose a vision-based, markerless, and training-free framework for soft robot shape reconstruction that directly leverages the robot's natural surface appearance. These surface features act as implicit visual markers, enabling a hierarchical matching strategy that decouples local partition alignment from global kinematic optimization. Requiring only an initial 3D reconstruction and kinematic alignment, our method achieves real-time shape tracking across diverse environments while maintaining robustness to occlusions and variations in camera viewpoints. Experimental validation on a continuum soft robot demonstrates an average tip error of 2.6% during real-time operation, as well as stable performance in practical closed-loop control tasks. These results highlight the potential of the proposed approach for reliable, low-cost deployment in dynamic real-world settings.
>
---
#### [new 040] Learning Diffusion Policies for Robotic Manipulation of Timber Joinery under Fabrication Uncertainty
- **分类: cs.RO**

- **简介: 该论文研究机器人在建造不确定性下对木制榫卯接头的接触敏感装配任务。针对制造误差带来的挑战，提出基于扩散策略的学习方法，在两阶段实验中验证了其性能与鲁棒性，成功实现75%平均成功率（最大偏差10mm），展示了其在复杂装配任务中的泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.17774v1](https://arxiv.org/pdf/2511.17774v1)**

> **作者:** Salma Mozaffari; Daniel Ruan; William van den Bogert; Nima Fazeli; Sigrid Adriaenssens; Arash Adel
>
> **摘要:** Construction uncertainties such as fabrication inaccuracies and material imperfections pose a significant challenge to contact-rich robotic manipulation by hindering precise and robust assembly. In this paper, we explore the performance and robustness of diffusion policy learning as a promising solution for contact-sensitive robotic assembly at construction scale, using timber mortise and tenon joints as a case study. A two-phase study is conducted: first, to evaluate policy performance and applicability; second, to assess robustness in handling fabrication uncertainties simulated as randomized perturbations to the mortise position. The best-performing policy achieved a total average success rate of 75% with perturbations up to 10 mm, including 100% success in unperturbed cases. The results demonstrate the potential of sensory-motor diffusion policies to generalize to a wide range of complex, contact-rich assembly tasks across construction and manufacturing, advancing robotic construction under uncertainty and contributing to safer, more efficient building practices.
>
---
#### [new 041] LEARN: Learning End-to-End Aerial Resource-Constrained Multi-Robot Navigation
- **分类: cs.RO; cs.LG; cs.MA**

- **简介: 该论文针对纳米无人机在资源受限条件下多机导航难题，提出轻量级两阶段强化学习框架LEARN。结合低分辨率传感器与紧凑策略网络，在仿真和实测中实现高效安全飞行，显著降低资源消耗并提升性能。**

- **链接: [https://arxiv.org/pdf/2511.17765v1](https://arxiv.org/pdf/2511.17765v1)**

> **作者:** Darren Chiu; Zhehui Huang; Ruohai Ge; Gaurav S. Sukhatme
>
> **备注:** 20 pages, 15 figures
>
> **摘要:** Nano-UAV teams offer great agility yet face severe navigation challenges due to constrained onboard sensing, communication, and computation. Existing approaches rely on high-resolution vision or compute-intensive planners, rendering them infeasible for these platforms. We introduce LEARN, a lightweight, two-stage safety-guided reinforcement learning (RL) framework for multi-UAV navigation in cluttered spaces. Our system combines low-resolution Time-of-Flight (ToF) sensors and a simple motion planner with a compact, attention-based RL policy. In simulation, LEARN outperforms two state-of-the-art planners by $10\%$ while using substantially fewer resources. We demonstrate LEARN's viability on six Crazyflie quadrotors, achieving fully onboard flight in diverse indoor and outdoor environments at speeds up to $2.0 m/s$ and traversing $0.2 m$ gaps.
>
---
#### [new 042] AutoFocus-IL: VLM-based Saliency Maps for Data-Efficient Visual Imitation Learning without Extra Human Annotations
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对视觉模仿学习中的数据效率与泛化能力问题，提出AutoFocus-IL方法。通过利用视觉语言模型自动生成时序显著性图，引导策略关注任务相关特征，抑制干扰因素，无需额外人工标注。实验表明其性能优于标准行为克隆及依赖人类监督的先进方法。**

- **链接: [https://arxiv.org/pdf/2511.18617v1](https://arxiv.org/pdf/2511.18617v1)**

> **作者:** Litian Gong; Fatemeh Bahrani; Yutai Zhou; Amin Banayeeanzade; Jiachen Li; Erdem Biyik
>
> **备注:** 8 pages, 6 figures. Code and datasets available at http://autofocus-il.github.io/
>
> **摘要:** AutoFocus-IL is a simple yet effective method to improve data efficiency and generalization in visual imitation learning by guiding policies to attend to task-relevant features rather than distractors and spurious correlations. Although saliency regularization has emerged as a promising way to achieve this, existing approaches typically require costly supervision such as human gaze data or manual saliency annotations. In contrast, AutoFocus-IL leverages vision-language models (VLMs) to automatically identify and track key objects in demonstrations, generating temporal saliency maps that highlight causal visual signals while suppressing distractors. These maps are then used to regularize behavior cloning policies, yielding stronger alignment between visual attention and task-relevant cues. Experiments in both the CARLA simulator and real-robot manipulation tasks demonstrate that AutoFocus-IL not only outperforms standard behavior cloning but also surpasses state-of-the-art baselines that assume privileged access to human supervision, such as gaze data. Code, datasets, and trained policy videos are available at https://AutoFocus-IL.github.io/.
>
---
#### [new 043] An Analysis of Constraint-Based Multi-Agent Pathfinding Algorithms
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文研究约束型多智能体路径规划（MAPF）算法，针对冲突避免策略中的约束分类问题，分析保守与激进约束在不同场景下的表现。通过实验对比CBS与CBSw/P算法在混合网格-路网表示下的性能，提出基于问题特征的约束选择决策流程，为MAPF与多机器人运动规划提供设计指导。**

- **链接: [https://arxiv.org/pdf/2511.18604v1](https://arxiv.org/pdf/2511.18604v1)**

> **作者:** Hannah Lee; James D. Motes; Marco Morales; Nancy M. Amato
>
> **摘要:** This study informs the design of future multi-agent pathfinding (MAPF) and multi-robot motion planning (MRMP) algorithms by guiding choices based on constraint classification for constraint-based search algorithms. We categorize constraints as conservative or aggressive and provide insights into their search behavior, focusing specifically on vanilla Conflict-Based Search (CBS) and Conflict-Based Search with Priorities (CBSw/P). Under a hybrid grid-roadmap representation with varying resolution, we observe that aggressive (priority constraint) formulations tend to solve more instances as agent count or resolution increases, whereas conservative (motion constraint) formulations yield stronger solution quality when both succeed. Findings are synthesized in a decision flowchart, aiding users in selecting suitable constraints. Recommendations extend to Multi-Robot Motion Planning (MRMP), emphasizing the importance of considering topological features alongside problem, solution, and representation features. A comprehensive exploration of the study, including raw data and map performance, is available in our public GitHub Repository: https://GitHub.com/hannahjmlee/constraint-mapf-analysis
>
---
#### [new 044] L1 Sample Flow for Efficient Visuomotor Learning
- **分类: cs.RO**

- **简介: 该论文针对机器人视觉-运动学习中的效率与多模态分布建模难题，提出L1 Flow方法。通过将流匹配重构为基于L1损失的样本预测，仅用两步采样即可生成精准动作序列，兼顾了流模型的多模态表达能力与L1回归的高效性，显著提升训练与推理速度。**

- **链接: [https://arxiv.org/pdf/2511.17898v1](https://arxiv.org/pdf/2511.17898v1)**

> **作者:** Weixi Song; Zhetao Chen; Tao Xu; Xianchao Zeng; Xinyu Zhou; Lixin Yang; Donglin Wang; Cewu Lu; Yong-Lu Li
>
> **摘要:** Denoising-based models, such as diffusion and flow matching, have been a critical component of robotic manipulation for their strong distribution-fitting and scaling capacity. Concurrently, several works have demonstrated that simple learning objectives, such as L1 regression, can achieve performance comparable to denoising-based methods on certain tasks, while offering faster convergence and inference. In this paper, we focus on how to combine the advantages of these two paradigms: retaining the ability of denoising models to capture multi-modal distributions and avoid mode collapse while achieving the efficiency of the L1 regression objective. To achieve this vision, we reformulate the original v-prediction flow matching and transform it into sample-prediction with the L1 training objective. We empirically show that the multi-modality can be expressed via a single ODE step. Thus, we propose \textbf{L1 Flow}, a two-step sampling schedule that generates a suboptimal action sequence via a single integration step and then reconstructs the precise action sequence through a single prediction. The proposed method largely retains the advantages of flow matching while reducing the iterative neural function evaluations to merely two and mitigating the potential performance degradation associated with direct sample regression. We evaluate our method with varying baselines and benchmarks, including 8 tasks in MimicGen, 5 tasks in RoboMimic \& PushT Bench, and one task in the real-world scenario. The results show the advantages of the proposed method with regard to training efficiency, inference speed, and overall performance. \href{https://song-wx.github.io/l1flow.github.io/}{Project Website.}
>
---
#### [new 045] Dreaming Falcon: Physics-Informed Model-Based Reinforcement Learning for Quadcopters
- **分类: cs.RO**

- **简介: 该论文针对四旋翼无人机在动态环境中的控制鲁棒性问题，提出基于物理信息的世界模型。通过将动力学建模为自由体系统并结合6-DOF Runge-Kutta积分器，提升模型泛化能力。相比传统RNN模型，该方法虽在训练数据上表现相近，但显著改善了新轨迹下的状态预测稳定性，助力策略收敛。**

- **链接: [https://arxiv.org/pdf/2511.18243v1](https://arxiv.org/pdf/2511.18243v1)**

> **作者:** Eashan Vytla; Bhavanishankar Kalavakolanu; Andrew Perrault; Matthew McCrink
>
> **摘要:** Current control algorithms for aerial robots struggle with robustness in dynamic environments and adverse conditions. Model-based reinforcement learning (RL) has shown strong potential in handling these challenges while remaining sample-efficient. Additionally, Dreamer has demonstrated that online model-based RL can be achieved using a recurrent world model trained on replay buffer data. However, applying Dreamer to aerial systems has been quite challenging due to its sample inefficiency and poor generalization of dynamics models. Our work explores a physics-informed approach to world model learning and improves policy performance. The world model treats the quadcopter as a free-body system and predicts the net forces and moments acting on it, which are then passed through a 6-DOF Runge-Kutta integrator (RK4) to predict future state rollouts. In this paper, we compare this physics-informed method to a standard RNN-based world model. Although both models perform well on the training data, we observed that they fail to generalize to new trajectories, leading to rapid divergence in state rollouts, preventing policy convergence.
>
---
#### [new 046] SAFE-SMART: Safety Analysis and Formal Evaluation using STL Metrics for Autonomous RoboTs
- **分类: cs.RO**

- **简介: 该论文针对学习型自主机器人的安全评估难题，提出基于STL的后验安全分析方法。通过将人类安全规则转为STL规范，量化评估模型行为的合规性，生成TRV和LRV指标，驱动模型迭代优化。在虚拟驾驶与真实机器人导航中均显著提升安全性与任务完成率。**

- **链接: [https://arxiv.org/pdf/2511.17781v1](https://arxiv.org/pdf/2511.17781v1)**

> **作者:** Kristy Sakano; Jianyu An; Dinesh Manocha; Huan Xu
>
> **摘要:** We present a novel, regulator-driven approach for post hoc safety evaluation of learning-based, black-box autonomous mobile robots, ensuring ongoing compliance with evolving, human-defined safety rules. In our iterative workflow, human safety requirements are translated by regulators into Signal Temporal Logic (STL) specifications. Rollout traces from the black-box model are externally verified for compliance, yielding quantitative safety metrics, Total Robustness Value (TRV) and Largest Robustness Value (LRV), which measure average and worst-case specification adherence. These metrics inform targeted retraining and iterative improvement by model designers. We apply our method across two different applications: a virtual driving scenario and an autonomous mobile robot navigating a complex environment, and observe statistically significant improvements across both scenarios. In the virtual driving scenario, we see a 177% increase in traces adhering to the simulation speed limit, a 1138% increase in traces minimizing off-road driving, and a 16% increase in traces successfully reaching the goal within the time limit. In the autonomous navigation scenario, there is a 300% increase in traces avoiding sharp turns, a 200% increase in traces reaching the goal within the time limit, and a 49% increase in traces minimizing time spent near obstacles. Finally, we validate our approach on a TurtleBot3 robot in the real world, and demonstrate improved obstacle navigation with safety buffers.
>
---
#### [new 047] Reference-Free Sampling-Based Model Predictive Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出一种无参考的采样式模型预测控制框架，用于四足与人形机器人的自主运动规划。针对传统方法依赖预设步态和复杂训练的问题，通过双空间样条参数化，在少量采样下实现自适应接触策略，直接优化高层目标，生成多样运动模式，如跳跃、翻滚与平衡，可在普通CPU上实时运行。**

- **链接: [https://arxiv.org/pdf/2511.19204v1](https://arxiv.org/pdf/2511.19204v1)**

> **作者:** Fabian Schramm; Pierre Fabre; Nicolas Perrin-Gilbert; Justin Carpentier
>
> **摘要:** We present a sampling-based model predictive control (MPC) framework that enables emergent locomotion without relying on handcrafted gait patterns or predefined contact sequences. Our method discovers diverse motion patterns, ranging from trotting to galloping, robust standing policies, jumping, and handstand balancing, purely through the optimization of high-level objectives. Building on model predictive path integral (MPPI), we propose a dual-space spline parameterization that operates on position and velocity control points. Our approach enables contact-making and contact-breaking strategies that adapt automatically to task requirements, requiring only a limited number of sampled trajectories. This sample efficiency allows us to achieve real-time control on standard CPU hardware, eliminating the need for GPU acceleration typically required by other state-of-the-art MPPI methods. We validate our approach on the Go2 quadrupedal robot, demonstrating various emergent gaits and basic jumping capabilities. In simulation, we further showcase more complex behaviors, such as backflips, dynamic handstand balancing and locomotion on a Humanoid, all without requiring reference tracking or offline pre-training.
>
---
#### [new 048] Unobservable Subspace Evolution and Alignment for Consistent Visual-Inertial Navigation
- **分类: cs.RO**

- **简介: 该论文针对视觉惯性导航系统（VINS）中的不一致性问题，提出Unobservable Subspace Evolution（USE）分析框架，揭示了估计步骤中不可观测子空间的动态演化机制。基于此，提出Unobservable Subspace Alignment（USA）方法，通过选择性干预导致失配的步骤，实现高精度、低复杂度的一致性优化。**

- **链接: [https://arxiv.org/pdf/2511.17992v1](https://arxiv.org/pdf/2511.17992v1)**

> **作者:** Chungeng Tian; Fenghua He; Ning Hao
>
> **备注:** 20 pages, 16 figures
>
> **摘要:** The inconsistency issue in the Visual-Inertial Navigation System (VINS) is a long-standing and fundamental challenge. While existing studies primarily attribute the inconsistency to observability mismatch, these analyses are often based on simplified theoretical formulations that consider only prediction and SLAM correction. Such formulations fail to cover the non-standard estimation steps, such as MSCKF correction and delayed initialization, which are critical for practical VINS estimators. Furthermore, the lack of a comprehensive understanding of how inconsistency dynamically emerges across estimation steps has hindered the development of precise and efficient solutions. As a result, current approaches often face a trade-off between estimator accuracy, consistency, and implementation complexity. To address these limitations, this paper proposes a novel analysis framework termed Unobservable Subspace Evolution (USE), which systematically characterizes how the unobservable subspace evolves throughout the entire estimation pipeline by explicitly tracking changes in its evaluation points. This perspective sheds new light on how individual estimation steps contribute to inconsistency. Our analysis reveals that observability misalignment induced by certain steps is the antecedent of observability mismatch. Guided by this insight, we propose a simple yet effective solution paradigm, Unobservable Subspace Alignment (USA), which eliminates inconsistency by selectively intervening only in those estimation steps that induce misalignment. We design two USA methods: transformation-based and re-evaluation-based, both offering accurate and computationally lightweight solutions. Extensive simulations and real-world experiments validate the effectiveness of the proposed methods.
>
---
#### [new 049] Rethinking Intermediate Representation for VLM-based Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文针对视觉语言模型（VLM）在机器人操作中生成可执行中间表示的难题，提出语义组装表示SEAM，通过分解为语义操作词典与语法结构，兼顾VLM理解性与任务泛化能力。结合新提出的少样本检索增强分割方法，实现快速精准的细粒度物体定位，显著提升真实场景下的操作性能。**

- **链接: [https://arxiv.org/pdf/2511.19315v1](https://arxiv.org/pdf/2511.19315v1)**

> **作者:** Weiliang Tang; Jialin Gao; Jia-Hui Pan; Gang Wang; Li Erran Li; Yunhui Liu; Mingyu Ding; Pheng-Ann Heng; Chi-Wing Fu
>
> **摘要:** Vision-Language Model (VLM) is an important component to enable robust robot manipulation. Yet, using it to translate human instructions into an action-resolvable intermediate representation often needs a tradeoff between VLM-comprehensibility and generalizability. Inspired by context-free grammar, we design the Semantic Assembly representation named SEAM, by decomposing the intermediate representation into vocabulary and grammar. Doing so leads us to a concise vocabulary of semantically-rich operations and a VLM-friendly grammar for handling diverse unseen tasks. In addition, we design a new open-vocabulary segmentation paradigm with a retrieval-augmented few-shot learning strategy to localize fine-grained object parts for manipulation, effectively with the shortest inference time over all state-of-the-art parallel works. Also, we formulate new metrics for action-generalizability and VLM-comprehensibility, demonstrating the compelling performance of SEAM over mainstream representations on both aspects. Extensive real-world experiments further manifest its SOTA performance under varying settings and tasks.
>
---
#### [new 050] Autonomous Surface Selection For Manipulator-Based UV Disinfection In Hospitals Using Foundation Models
- **分类: cs.RO**

- **简介: 该论文针对医院中机械臂紫外消毒的表面自动选择问题，提出基于基础模型的方法，无需训练即可实现目标表面精准分割。通过视觉语言模型辅助分割优化，有效排除细小非目标物体，提升分割准确率至92%以上，降低人为干预与误照射风险，推动自动化消毒落地。**

- **链接: [https://arxiv.org/pdf/2511.18709v1](https://arxiv.org/pdf/2511.18709v1)**

> **作者:** Xueyan Oh; Jonathan Her; Zhixiang Ong; Brandon Koh; Yun Hann Tan; U-Xuan Tan
>
> **备注:** 7 pages, 7 figures; This paper has been accepted by IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Ultraviolet (UV) germicidal radiation is an established non-contact method for surface disinfection in medical environments. Traditional approaches require substantial human intervention to define disinfection areas, complicating automation, while deep learning-based methods often need extensive fine-tuning and large datasets, which can be impractical for large-scale deployment. Additionally, these methods often do not address scene understanding for partial surface disinfection, which is crucial for avoiding unintended UV exposure. We propose a solution that leverages foundation models to simplify surface selection for manipulator-based UV disinfection, reducing human involvement and removing the need for model training. Additionally, we propose a VLM-assisted segmentation refinement to detect and exclude thin and small non-target objects, showing that this reduces mis-segmentation errors. Our approach achieves over 92\% success rate in correctly segmenting target and non-target surfaces, and real-world experiments with a manipulator and simulated UV light demonstrate its practical potential for real-world applications.
>
---
#### [new 051] An Efficient Closed-Form Solution to Full Visual-Inertial State Initialization
- **分类: cs.RO**

- **简介: 该论文针对视觉惯性系统（VIO）的初始化问题，提出一种无需非线性优化的闭式解法。通过小旋转与匀速假设，构建紧凑且数值稳定的解析解，并设计可观测性驱动的两阶段初始化流程，显著降低误差、缩短初始化时间与计算开销。**

- **链接: [https://arxiv.org/pdf/2511.18910v1](https://arxiv.org/pdf/2511.18910v1)**

> **作者:** Samuel Cerezo; Seong Hun Lee; Javier Civera
>
> **备注:** 8 pages, 2 figures, 10 tables. Submitted to RA-L
>
> **摘要:** In this letter, we present a closed-form initialization method that recovers the full visual-inertial state without nonlinear optimization. Unlike previous approaches that rely on iterative solvers, our formulation yields analytical, easy-to-implement, and numerically stable solutions for reliable start-up. Our method builds on small-rotation and constant-velocity approximations, which keep the formulation compact while preserving the essential coupling between motion and inertial measurements. We further propose an observability-driven, two-stage initialization scheme that balances accuracy with initialization latency. Extensive experiments on the EuRoC dataset validate our assumptions: our method achieves 10-20% lower initialization error than optimization-based approaches, while using 4x shorter initialization windows and reducing computational cost by 5x.
>
---
#### [new 052] Splatblox: Traversability-Aware Gaussian Splatting for Outdoor Robot Navigation
- **分类: cs.RO**

- **简介: 该论文提出Splatblox，用于户外机器人自主导航任务。针对复杂植被与不规则障碍环境下的路径规划难题，融合RGB图像与LiDAR点云，基于高斯点阵构建可实时更新的语义可通行欧式符号距离场，实现几何与语义联合感知，显著提升导航成功率与效率。**

- **链接: [https://arxiv.org/pdf/2511.18525v1](https://arxiv.org/pdf/2511.18525v1)**

> **作者:** Samarth Chopra; Jing Liang; Gershom Seneviratne; Yonghan Lee; Jaehoon Choi; Jianyu An; Stephen Cheng; Dinesh Manocha
>
> **备注:** Submitted to ICRA 2026
>
> **摘要:** We present Splatblox, a real-time system for autonomous navigation in outdoor environments with dense vegetation, irregular obstacles, and complex terrain. Our method fuses segmented RGB images and LiDAR point clouds using Gaussian Splatting to construct a traversability-aware Euclidean Signed Distance Field (ESDF) that jointly encodes geometry and semantics. Updated online, this field enables semantic reasoning to distinguish traversable vegetation (e.g., tall grass) from rigid obstacles (e.g., trees), while LiDAR ensures 360-degree geometric coverage for extended planning horizons. We validate Splatblox on a quadruped robot and demonstrate transfer to a wheeled platform. In field trials across vegetation-rich scenarios, it outperforms state-of-the-art methods with over 50% higher success rate, 40% fewer freezing incidents, 5% shorter paths, and up to 13% faster time to goal, while supporting long-range missions up to 100 meters. Experiment videos and more details can be found on our project page: https://splatblox.github.io
>
---
#### [new 053] Robot joint characterisation and control using a magneto-optical rotary encoder
- **分类: cs.RO; physics.optics**

- **简介: 该论文提出一种基于磁光效应的紧凑型旋转编码器，用于机器人转轴的精确位置检测。针对传统编码器成本高、体积大的问题，采用双程光路与非均匀磁体结构，实现360°连续旋转测量，分辨率0.3°，最高转速370°/s，提供低成本、高可靠的替代方案。**

- **链接: [https://arxiv.org/pdf/2511.17608v1](https://arxiv.org/pdf/2511.17608v1)**

> **作者:** Yunlong Guo; John Canning; Zenon Chaczko; Gang-Ding Peng
>
> **摘要:** A robust and compact magneto-optical rotary encoder for the characterisation of robotic rotary joints is demonstrated. The system employs magnetic field-induced optical attenuation in a double-pass configuration using rotating nonuniform magnets around an optical circulator operating in reflection. The encoder tracks continuous 360° rotation with rotation sweep rates from ν = 135 °/s to ν = 370 °/s, and an angular resolution of Δθ = 0.3°. This offers a low-cost and reliable alternative to conventional robot rotation encoders while maintaining competitive performance.
>
---
#### [new 054] Online Learning-Enhanced Lie Algebraic MPC for Robust Trajectory Tracking of Autonomous Surface Vehicles
- **分类: cs.RO**

- **简介: 该论文针对自主水面航行器在风浪等未知扰动下的轨迹跟踪难题，提出一种融合在线学习与李群凸误差模型预测控制的算法。通过实时补偿扰动，实现高精度、鲁棒且高效的轨迹跟踪，验证了方法在仿真与实测中的优越性。**

- **链接: [https://arxiv.org/pdf/2511.18683v1](https://arxiv.org/pdf/2511.18683v1)**

> **作者:** Yinan Dong; Ziyu Xu; Tsimafei Lazouski; Sangli Teng; Maani Ghaffari
>
> **摘要:** Autonomous surface vehicles (ASVs) are easily influenced by environmental disturbances such as wind and waves, making accurate trajectory tracking a persistent challenge in dynamic marine conditions. In this paper, we propose an efficient controller for trajectory tracking of marine vehicles under unknown disturbances by combining a convex error-state MPC on the Lie group with an online learning module to compensate for these disturbances in real time. This design enables adaptive and robust control while maintaining computational efficiency. Extensive evaluations in numerical simulations, the Virtual RobotX (VRX) simulator, and real-world field experiments demonstrate that our method achieves superior tracking accuracy under various disturbance scenarios compared with existing approaches.
>
---
#### [new 055] Object-centric Task Representation and Transfer using Diffused Orientation Fields
- **分类: cs.RO**

- **简介: 该论文针对机器人在曲面物体上迁移操作任务的难题，提出基于扩散方向场（DOF）的局部参考框架表示方法。通过在线计算点云数据中的平滑方向场，将任务表达为随位置变化的局部坐标系，仅需稀疏关键点对应即可实现跨形状任务迁移，成功应用于连续交互任务如检测、切割和剥离。**

- **链接: [https://arxiv.org/pdf/2511.18563v1](https://arxiv.org/pdf/2511.18563v1)**

> **作者:** Cem Bilaloglu; Tobias Löw; Sylvain Calinon
>
> **摘要:** Curved objects pose a fundamental challenge for skill transfer in robotics: unlike planar surfaces, they do not admit a global reference frame. As a result, task-relevant directions such as "toward" or "along" the surface vary with position and geometry, making object-centric tasks difficult to transfer across shapes. To address this, we introduce an approach using Diffused Orientation Fields (DOF), a smooth representation of local reference frames, for transfer learning of tasks across curved objects. By expressing manipulation tasks in these smoothly varying local frames, we reduce the problem of transferring tasks across curved objects to establishing sparse keypoint correspondences. DOF is computed online from raw point cloud data using diffusion processes governed by partial differential equations, conditioned on keypoints. We evaluate DOF under geometric, topological, and localization perturbations, and demonstrate successful transfer of tasks requiring continuous physical interaction such as inspection, slicing, and peeling across varied objects. We provide our open-source codes at our website https://github.com/idiap/diffused_fields_robotics
>
---
#### [new 056] Enhancing UAV Search under Occlusion using Next Best View Planning
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对无人机在高遮挡环境（如密林）中搜索时视野受限的问题，提出一种改进的“下一最佳视角”规划方法。通过引入可见性与几何双重启发式策略，优化相机视角选择，提升遮挡环境下目标检测率与覆盖效率，显著增强搜救任务的搜索性能。**

- **链接: [https://arxiv.org/pdf/2511.18353v1](https://arxiv.org/pdf/2511.18353v1)**

> **作者:** Sigrid Helene Strand; Thomas Wiedemann; Bram Burczek; Dmitriy Shutin
>
> **备注:** Submitted to IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
>
> **摘要:** Search and rescue missions are often critical following sudden natural disasters or in high-risk environmental situations. The most challenging search and rescue missions involve difficult-to-access terrains, such as dense forests with high occlusion. Deploying unmanned aerial vehicles for exploration can significantly enhance search effectiveness, facilitate access to challenging environments, and reduce search time. However, in dense forests, the effectiveness of unmanned aerial vehicles depends on their ability to capture clear views of the ground, necessitating a robust search strategy to optimize camera positioning and perspective. This work presents an optimized planning strategy and an efficient algorithm for the next best view problem in occluded environments. Two novel optimization heuristics, a geometry heuristic, and a visibility heuristic, are proposed to enhance search performance by selecting optimal camera viewpoints. Comparative evaluations in both simulated and real-world settings reveal that the visibility heuristic achieves greater performance, identifying over 90% of hidden objects in simulated forests and offering 10% better detection rates than the geometry heuristic. Additionally, real-world experiments demonstrate that the visibility heuristic provides better coverage under the canopy, highlighting its potential for improving search and rescue missions in occluded environments.
>
---
#### [new 057] SP-VINS: A Hybrid Stereo Visual Inertial Navigation System based on Implicit Environmental Map
- **分类: cs.RO**

- **简介: 该论文提出SP-VINS，一种基于隐式环境地图的混合立体视觉惯性导航系统，旨在解决传统滤波式VINS长期定位精度下降的问题。通过关键帧与2D特征构建隐式地图，结合地标重投影与射线约束的混合残差滤波框架，并实现相机-IMU外参在线标定，显著提升长时间高精度定位性能。**

- **链接: [https://arxiv.org/pdf/2511.18756v1](https://arxiv.org/pdf/2511.18756v1)**

> **作者:** Xueyu Du; Lilian Zhang; Fuan Duan; Xincan Luo; Maosong Wang; Wenqi Wu; JunMao
>
> **摘要:** Filter-based visual inertial navigation system (VINS) has attracted mobile-robot researchers for the good balance between accuracy and efficiency, but its limited mapping quality hampers long-term high-accuracy state estimation. To this end, we first propose a novel filter-based stereo VINS, differing from traditional simultaneous localization and mapping (SLAM) systems based on 3D map, which performs efficient loop closure constraints with implicit environmental map composed of keyframes and 2D keypoints. Secondly, we proposed a hybrid residual filter framework that combines landmark reprojection and ray constraints to construct a unified Jacobian matrix for measurement updates. Finally, considering the degraded environment, we incorporated the camera-IMU extrinsic parameters into visual description to achieve online calibration. Benchmark experiments demonstrate that the proposed SP-VINS achieves high computational efficiency while maintaining long-term high-accuracy localization performance, and is superior to existing state-of-the-art (SOTA) methods.
>
---
#### [new 058] MicCheck: Repurposing Off-the-Shelf Pin Microphones for Easy and Low-Cost Contact Sensing
- **分类: cs.RO**

- **简介: 该论文针对机器人操作中触觉感知不足的问题，提出MicCheck方法，利用廉价蓝牙麦克风作为接触传感器。通过简单改装，实现低成本、易集成的声学接触感知，有效支持材料分类与复杂操作任务，提升了模仿学习的性能。**

- **链接: [https://arxiv.org/pdf/2511.18299v1](https://arxiv.org/pdf/2511.18299v1)**

> **作者:** Steven Oh; Tai Inui; Magdeline Kuan; Jia-Yeu Lin
>
> **摘要:** Robotic manipulation tasks are contact-rich, yet most imitation learning (IL) approaches rely primarily on vision, which struggles to capture stiffness, roughness, slip, and other fine interaction cues. Tactile signals can address this gap, but existing sensors often require expensive, delicate, or integration-heavy hardware. In this work, we introduce MicCheck, a plug-and-play acoustic sensing approach that repurposes an off-the-shelf Bluetooth pin microphone as a low-cost contact sensor. The microphone clips into a 3D-printed gripper insert and streams audio via a standard USB receiver, requiring no custom electronics or drivers. Despite its simplicity, the microphone provides signals informative enough for both perception and control. In material classification, it achieves 92.9% accuracy on a 10-class benchmark across four interaction types (tap, knock, slow press, drag). For manipulation, integrating pin microphone into an IL pipeline with open source hardware improves the success rate on picking and pouring task from 0.40 to 0.80 and enables reliable execution of contact-rich skills such as unplugging and sound-based sorting. Compared with high-resolution tactile sensors, pin microphones trade spatial detail for cost and ease of integration, offering a practical pathway for deploying acoustic contact sensing in low-cost robot setups.
>
---
#### [new 059] Multi-Agent Monocular Dense SLAM With 3D Reconstruction Priors
- **分类: cs.RO**

- **简介: 该论文提出首个多智能体单目稠密SLAM系统，解决单目SLAM计算成本高、难以扩展至多机协同的问题。通过引入3D重建先验与基于回环检测的映射融合机制，实现高效、高精度的全局一致性建图，显著提升计算效率，保持与先进方法相当的映射精度。**

- **链接: [https://arxiv.org/pdf/2511.19031v1](https://arxiv.org/pdf/2511.19031v1)**

> **作者:** Haihang Wu; Yuchen Zhou
>
> **摘要:** Monocular Simultaneous Localization and Mapping (SLAM) aims to estimate a robot's pose while simultaneously reconstructing an unknown 3D scene using a single camera. While existing monocular SLAM systems generate detailed 3D geometry through dense scene representations, they are computationally expensive due to the need for iterative optimization. To address this challenge, MASt3R-SLAM utilizes learned 3D reconstruction priors, enabling more efficient and accurate estimation of both 3D structures and camera poses. However, MASt3R-SLAM is limited to single-agent operation. In this paper, we extend MASt3R-SLAM to introduce the first multi-agent monocular dense SLAM system. Each agent performs local SLAM using a 3D reconstruction prior, and their individual maps are fused into a globally consistent map through a loop-closure-based map fusion mechanism. Our approach improves computational efficiency compared to state-of-the-art methods, while maintaining similar mapping accuracy when evaluated on real-world datasets.
>
---
#### [new 060] Deployment Dynamics and Optimization of Novel Space Antenna Deployable Mechanism
- **分类: cs.RO**

- **简介: 该论文针对空间大口径天线难以装入小型运载火箭的问题，提出一种新型三杆折叠桁架机构（TSDTM）。通过几何建模、运动学与动力学分析，结合仿真与优化算法，实现高效部署与轻量化设计。采用AI方法优化材料与结构参数，显著提升预测精度，验证了智能算法在空间结构设计中的应用潜力。**

- **链接: [https://arxiv.org/pdf/2511.19377v1](https://arxiv.org/pdf/2511.19377v1)**

> **作者:** Mamoon Aamir; Mariyam Sattar; Naveed Ur Rehman Junejo; Aqsa Zafar Abbasi
>
> **摘要:** Given the increasing need for large aperture antennas in space missions, the difficulty of fitting such structures into small launch vehicles has prompted the design of deployable antenna systems. The thesis introduces a new Triple Scissors Deployable Truss Mechanism (TSDTM) for space antenna missions. The new mechanism is to be stowed during launch and efficiently deploy in orbit, offering maximum aperture size while taking up minimal launch volume. The thesis covers the entire design process from geometric modeling, kinematic analysis with screw theory and Newtonian approaches, dynamic analysis by eigenvalue and simulation methods, and verification with SolidWorks. In addition, optimization routines were coded based on Support Vector Machines for material choice in LEO environments and machine learning method for geometric setup. The TSDTM presented has enhanced structural dynamics with good comparison between simulation and analytical predictions. The structure optimized proved highly accurate, with a deviation of just 1.94% between machine learning-predicted and simulated natural frequencies, demonstrating the potential of incorporating AI-based methods in space structural design.
>
---
#### [new 061] Efficient Optimization of a Permanent Magnet Array for a Stable 2D Trap
- **分类: cs.RO**

- **简介: 该论文针对微型机器人磁控中的稳定力场难题，提出一种基于永久磁阵列的2D稳定磁阱。通过GPU加速优化算法，快速计算最优磁体角度，实现对毫米级机器人在20–120mm范围内的精准控制，验证了复杂轨迹追踪能力，并展示了可扩展至100磁体的高效性。**

- **链接: [https://arxiv.org/pdf/2511.19201v1](https://arxiv.org/pdf/2511.19201v1)**

> **作者:** Ann-Sophia Müller; Moonkwang Jeong; Jiyuan Tian; Meng Zhang; Tian Qiu
>
> **备注:** 6 pages, 6 figures, IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Untethered magnetic manipulation of biomedical millirobots has a high potential for minimally invasive surgical applications. However, it is still challenging to exert high actuation forces on the small robots over a large distance. Permanent magnets offer stronger magnetic torques and forces than electromagnetic coils, however, feedback control is more difficult. As proven by Earnshaw's theorem, it is not possible to achieve a stable magnetic trap in 3D by static permanent magnets. Here, we report a stable 2D magnetic force trap by an array of permanent magnets to control a millirobot. The trap is located in an open space with a tunable distance to the magnet array in the range of 20 - 120mm, which is relevant to human anatomical scales. The design is achieved by a novel GPU-accelerated optimization algorithm that uses mean squared error (MSE) and Adam optimizer to efficiently compute the optimal angles for any number of magnets in the array. The algorithm is verified using numerical simulation and physical experiments with an array of two magnets. A millirobot is successfully trapped and controlled to follow a complex trajectory. The algorithm demonstrates high scalability by optimizing the angles for 100 magnets in under three seconds. Moreover, the optimization workflow can be adapted to optimize a permanent magnet array to achieve the desired force vector fields.
>
---
#### [new 062] Analysis of Deep-Learning Methods in an ISO/TS 15066-Compliant Human-Robot Safety Framework
- **分类: cs.RO**

- **简介: 该论文针对协作机器人在ISO/TS 15066标准下因保守速度限制导致效率低的问题，提出一种基于深度学习的人机安全框架（HRSF），通过识别人体不同部位实现动态速度调整，提升安全性与效率。实验表明，可减少15%的周期时间。**

- **链接: [https://arxiv.org/pdf/2511.19094v1](https://arxiv.org/pdf/2511.19094v1)**

> **作者:** David Bricher; Andreas Mueller
>
> **备注:** MDPI Sensors, published 22 November 2025
>
> **摘要:** Over the last years collaborative robots have gained great success in manufacturing applications where human and robot work together in close proximity. However, current ISO/TS-15066-compliant implementations often limit the efficiency of collaborative tasks due to conservative speed restrictions. For this reason, this paper introduces a deep-learning-based human-robot-safety framework (HRSF) that aims at a dynamical adaptation of robot velocities depending on the separation distance between human and robot while respecting maximum biomechanical force and pressure limits. The applicability of the framework was investigated for four different deep learning approaches that can be used for human body extraction: human body recognition, human body segmentation, human pose estimation, and human body part segmentation. Unlike conventional industrial safety systems, the proposed HRSF differentiates individual human body parts from other objects, enabling optimized robot process execution. Experiments demonstrated a quantitative reduction in cycle time of up to 15% compared to conventional safety technology.
>
---
#### [new 063] Time-aware Motion Planning in Dynamic Environments with Conformal Prediction
- **分类: cs.RO**

- **简介: 该论文针对动态环境中安全导航问题，提出基于可信预测（Conformal Prediction）的时空感知运动规划框架。通过全局与局部规划器结合，实现无分布假设的安全保障，并引入自适应分位数机制优化不确定性量化，提升轨迹可行性与响应能力。**

- **链接: [https://arxiv.org/pdf/2511.18170v1](https://arxiv.org/pdf/2511.18170v1)**

> **作者:** Kaier Liang; Licheng Luo; Yixuan Wang; Mingyu Cai; Cristian Ioan Vasile
>
> **摘要:** Safe navigation in dynamic environments remains challenging due to uncertain obstacle behaviors and the lack of formal prediction guarantees. We propose two motion planning frameworks that leverage conformal prediction (CP): a global planner that integrates Safe Interval Path Planning (SIPP) for uncertainty-aware trajectory generation, and a local planner that performs online reactive planning. The global planner offers distribution-free safety guarantees for long-horizon navigation, while the local planner mitigates inaccuracies in obstacle trajectory predictions through adaptive CP, enabling robust and responsive motion in dynamic environments. To further enhance trajectory feasibility, we introduce an adaptive quantile mechanism in the CP-based uncertainty quantification. Instead of using a fixed confidence level, the quantile is automatically tuned to the optimal value that preserves trajectory feasibility, allowing the planner to adaptively tighten safety margins in regions with higher uncertainty. We validate the proposed framework through numerical experiments conducted in dynamic and cluttered environments. The project page is available at https://time-aware-planning.github.io
>
---
#### [new 064] Autonomous Docking of Multi-Rotor UAVs on Blimps under the Influence of Wind Gusts
- **分类: cs.RO**

- **简介: 该论文研究多旋翼无人机在风扰下自主对接飞艇的任务。针对飞艇易受风 gust 影响导致轨迹偏移的问题，提出基于时序卷积网络的风扰预测方法与新型模型预测控制策略，实现无碰撞、精确对接。通过仿真与实验证明方法有效性，首次实现飞艇上无人机自主对接的实机验证。**

- **链接: [https://arxiv.org/pdf/2511.19135v1](https://arxiv.org/pdf/2511.19135v1)**

> **作者:** Pascal Goldschmid; Aamir Ahmad
>
> **备注:** 13 pages, 8 figures, 8 tables
>
> **摘要:** Multi-rotor UAVs face limited flight time due to battery constraints. Autonomous docking on blimps with onboard battery recharging and data offloading offers a promising solution for extended UAV missions. However, the vulnerability of blimps to wind gusts causes trajectory deviations, requiring precise, obstacle-aware docking strategies. To this end, this work introduces two key novelties: (i) a temporal convolutional network that predicts blimp responses to wind gusts, enabling rapid gust detection and estimation of points where the wind gust effect has subsided; (ii) a model predictive controller (MPC) that leverages these predictions to compute collision-free trajectories for docking, enabled by a novel obstacle avoidance method for close-range manoeuvres near the blimp. Simulation results show our method outperforms a baseline constant-velocity model of the blimp significantly across different scenarios. We further validate the approach in real-world experiments, demonstrating the first autonomous multi-rotor docking control strategy on blimps shown outside simulation. Source code is available here https://github.com/robot-perception-group/multi_rotor_airship_docking.
>
---
#### [new 065] Observer Actor: Active Vision Imitation Learning with Sparse View Gaussian Splatting
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出Observer Actor（ObAct）框架，用于主动视觉模仿学习。针对机器人操作中视角遮挡问题，通过动态分配观察者与执行者角色，利用3D高斯点云实现虚拟视角探索，优化观测视角，提升策略鲁棒性。实验表明，相比固定摄像头，性能显著提升。**

- **链接: [https://arxiv.org/pdf/2511.18140v1](https://arxiv.org/pdf/2511.18140v1)**

> **作者:** Yilong Wang; Cheng Qian; Ruomeng Fan; Edward Johns
>
> **备注:** Videos are available on our project webpage at https://obact.github.io
>
> **摘要:** We propose Observer Actor (ObAct), a novel framework for active vision imitation learning in which the observer moves to optimal visual observations for the actor. We study ObAct on a dual-arm robotic system equipped with wrist-mounted cameras. At test time, ObAct dynamically assigns observer and actor roles: the observer arm constructs a 3D Gaussian Splatting (3DGS) representation from three images, virtually explores this to find an optimal camera pose, then moves to this pose; the actor arm then executes a policy using the observer's observations. This formulation enhances the clarity and visibility of both the object and the gripper in the policy's observations. As a result, we enable the training of ambidextrous policies on observations that remain closer to the occlusion-free training distribution, leading to more robust policies. We study this formulation with two existing imitation learning methods -- trajectory transfer and behavior cloning -- and experiments show that ObAct significantly outperforms static-camera setups: trajectory transfer improves by 145% without occlusion and 233% with occlusion, while behavior cloning improves by 75% and 143%, respectively. Videos are available at https://obact.github.io.
>
---
#### [new 066] Connectivity-Preserving Multi-Agent Area Coverage via Optimal-Transport-Based Density-Driven Optimal Control (D2OC)
- **分类: eess.SY; cs.RO**

- **简介: 该论文针对多智能体非均匀区域覆盖任务，解决现有密度驱动方法无法保证通信连通性的问题。提出基于最优传输的连通性保持型密度驱动最优控制（D2OC）方法，通过平滑连通性惩罚项，在保持严格凸性与分布式实现的同时，确保通信连续，提升覆盖质量与收敛速度。**

- **链接: [https://arxiv.org/pdf/2511.18579v1](https://arxiv.org/pdf/2511.18579v1)**

> **作者:** Kooktae Lee; Ethan Brook
>
> **备注:** Under review in IEEE Control Systems Letters (LCSS). 6 pages
>
> **摘要:** Multi-agent systems play a central role in area coverage tasks across search-and-rescue, environmental monitoring, and precision agriculture. Achieving non-uniform coverage, where spatial priorities vary across the domain, requires coordinating agents while respecting dynamic and communication constraints. Density-driven approaches can distribute agents according to a prescribed reference density, but existing methods do not ensure connectivity. This limitation often leads to communication loss, reduced coordination, and degraded coverage performance. This letter introduces a connectivity-preserving extension of the Density-Driven Optimal Control (D2OC) framework. The coverage objective, defined using the Wasserstein distance between the agent distribution and the reference density, admits a convex quadratic program formulation. Communication constraints are incorporated through a smooth connectivity penalty, which maintains strict convexity, supports distributed implementation, and preserves inter-agent communication without imposing rigid formations. Simulation studies show that the proposed method consistently maintains connectivity, improves convergence speed, and enhances non-uniform coverage quality compared with density-driven schemes that do not incorporate explicit connectivity considerations.
>
---
#### [new 067] Target-Bench: Can World Models Achieve Mapless Path Planning with Semantic Targets?
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对世界模型在无地图路径规划中的能力不足问题，提出Target-Bench基准，评估模型在真实环境中的语义目标导航性能。通过450段视频与真实轨迹数据，量化评估生成视频的路径规划能力，验证了现有模型表现有限，并展示微调可显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.17792v1](https://arxiv.org/pdf/2511.17792v1)**

> **作者:** Dingrui Wang; Hongyuan Ye; Zhihao Liang; Zhexiao Sun; Zhaowei Lu; Yuchen Zhang; Yuyu Zhao; Yuan Gao; Marvin Seegert; Finn Schäfer; Haotong Qin; Wei Li; Luigi Palmieri; Felix Jahncke; Mattia Piccinini; Johannes Betz
>
> **备注:** 10 pages
>
> **摘要:** While recent world models generate highly realistic videos, their ability to perform robot path planning remains unclear and unquantified. We introduce Target-Bench, the first benchmark specifically designed to evaluate world models on mapless path planning toward semantic targets in real-world environments. Target-Bench provides 450 robot-collected video sequences spanning 45 semantic categories with SLAM-based ground truth trajectories. Our evaluation pipeline recovers camera motion from generated videos and measures planning performance using five complementary metrics that quantify target-reaching capability, trajectory accuracy, and directional consistency. We evaluate state-of-the-art models including Sora 2, Veo 3.1, and the Wan series. The best off-the-shelf model (Wan2.2-Flash) achieves only 0.299 overall score, revealing significant limitations in current world models for robotic planning tasks. We show that fine-tuning an open-source 5B-parameter model on only 325 scenarios from our dataset achieves 0.345 overall score -- an improvement of more than 400% over its base version (0.066) and 15% higher than the best off-the-shelf model. We will open-source the code and dataset.
>
---
#### [new 068] Rad-GS: Radar-Vision Integration for 3D Gaussian Splatting SLAM in Outdoor Environments
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Rad-GS，一种用于室外大场景的雷达-视觉融合4D SLAM系统，利用3D高斯作为可微分空间表示。通过融合雷达点云与多普勒信息，实现动态物体掩码，减少渲染伪影并提升定位精度；结合非同步图像优化全局高斯表示，增强纹理一致性和新视角合成质量；采用全局八叉树与针对性管理策略，降低噪声与内存消耗，实现了千米级真实场景重建。**

- **链接: [https://arxiv.org/pdf/2511.16091v1](https://arxiv.org/pdf/2511.16091v1)**

> **作者:** Renxiang Xiao; Wei Liu; Yuanfan Zhang; Yushuai Chen; Jinming Chen; Zilu Wang; Liang Hu
>
> **摘要:** We present Rad-GS, a 4D radar-camera SLAM system designed for kilometer-scale outdoor environments, utilizing 3D Gaussian as a differentiable spatial representation. Rad-GS combines the advantages of raw radar point cloud with Doppler information and geometrically enhanced point cloud to guide dynamic object masking in synchronized images, thereby alleviating rendering artifacts and improving localization accuracy. Additionally, unsynchronized image frames are leveraged to globally refine the 3D Gaussian representation, enhancing texture consistency and novel view synthesis fidelity. Furthermore, the global octree structure coupled with a targeted Gaussian primitive management strategy further suppresses noise and significantly reduces memory consumption in large-scale environments. Extensive experiments and ablation studies demonstrate that Rad-GS achieves performance comparable to traditional 3D Gaussian methods based on camera or LiDAR inputs, highlighting the feasibility of robust outdoor mapping using 4D mmWave radar. Real-world reconstruction at kilometer scale validates the potential of Rad-GS for large-scale scene reconstruction.
>
---
#### [new 069] SWITCH: Benchmarking Modeling and Handling of Tangible Interfaces in Long-horizon Embodied Scenarios
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出SWITCH基准，针对长时序具身智能中的实体控制接口（TCI）建模与操作问题。解决现有基准缺乏对视觉证据、因果预测与结果验证的评估难题。通过351个真实设备任务，评估模型在视觉问答、语义定位、动作生成等五方面能力，推动更鲁棒的具身智能发展。**

- **链接: [https://arxiv.org/pdf/2511.17649v1](https://arxiv.org/pdf/2511.17649v1)**

> **作者:** Jieru Lin; Zhiwei Yu; Börje F. Karlsson
>
> **摘要:** Autonomous intelligence requires not only perception and reasoning, but critically, effective interaction with the existing world and its infrastructure. Everyday environments are rich in tangible control interfaces (TCIs), e.g., light switches, appliance panels, and embedded GUIs, that demand commonsense and physics reasoning, but also causal prediction and outcome verification in time and space (e.g., delayed heating, remote lights). Moreover, failures here have potential safety implications, yet current benchmarks rarely test grounding, partial observability (video), or post-hoc verification in situated settings. We introduce SWITCH (Semantic World Interface Tasks for Control and Handling), an embodied, task-driven benchmark created through iterative releases to probe these gaps. Its first iteration, SWITCH-Basic, evaluates five complementary abilities:task-aware VQA, semantic UI grounding, action generation, state-transition prediction, and result verification, under egocentric RGB video input and device diversity. Across 351 tasks spanning 98 real devices and appliances, commercial and open LMMMs exhibit inconsistent performance even on single-step interactions, often over-relying on textual cues and under-using visual or video evidence (and high aggregate scores can mask such failures). SWITCH provides data, code, and held-out splits to enable reproducible evaluation and community contributions toward more challenging future iterations of the benchmark and the creation of training datasets. Benchmark resources are available at: https://github.com/BAAI-Agents/SWITCH.
>
---
#### [new 070] Categorical Equivariant Deep Learning: Category-Equivariant Neural Networks and Universal Approximation Theorems
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出范畴等变深度学习框架（CENN），统一了群、偏序集、图与层叠神经网络的等变性。通过范畴论与径向测度构建线性与非线性层，证明了有限深度CENN在连续等变变换空间中稠密，实现了广义等变万能逼近定理，拓展了等变深度学习至非几何对称性。**

- **链接: [https://arxiv.org/pdf/2511.18417v1](https://arxiv.org/pdf/2511.18417v1)**

> **作者:** Yoshihiro Maruyama
>
> **摘要:** We develop a theory of category-equivariant neural networks (CENNs) that unifies group/groupoid-equivariant networks, poset/lattice-equivariant networks, graph and sheaf neural networks. Equivariance is formulated as naturality in a topological category with Radon measures, formulating linear and nonlinear layers in the categorical setup. We prove the equivariant universal approximation theorem in the general setting: the class of finite-depth CENNs is dense in the space of continuous equivariant transformations. We instantiate the framework for groups/groupoids, posets/lattices, graphs and cellular sheaves, deriving universal approximation theorems for them in a systematic manner. Categorical equivariant deep learning thus allows us to expand the horizons of equivariant deep learning beyond group actions, encompassing not only geometric symmetries but also contextual and compositional symmetries.
>
---
#### [new 071] QAL: A Loss for Recall Precision Balance in 3D Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对3D重建中召回与精度难以平衡的问题，提出质量感知损失（QAL），通过分离召回与精度控制，提升模型对细结构和稀疏区域的覆盖能力。实验表明，QAL在多个数据集和架构上均显著优于传统损失函数，且增强机器人抓取性能，具备良好的泛化性与实用性。**

- **链接: [https://arxiv.org/pdf/2511.17824v1](https://arxiv.org/pdf/2511.17824v1)**

> **作者:** Pranay Meshram; Yash Turkar; Kartikeya Singh; Praveen Raj Masilamani; Charuvahan Adhivarahan; Karthik Dantu
>
> **备注:** Accepted to WACV 2026. Camera-ready version to appear
>
> **摘要:** Volumetric learning underpins many 3D vision tasks such as completion, reconstruction, and mesh generation, yet training objectives still rely on Chamfer Distance (CD) or Earth Mover's Distance (EMD), which fail to balance recall and precision. We propose Quality-Aware Loss (QAL), a drop-in replacement for CD/EMD that combines a coverage-weighted nearest-neighbor term with an uncovered-ground-truth attraction term, explicitly decoupling recall and precision into tunable components. Across diverse pipelines, QAL achieves consistent coverage gains, improving by an average of +4.3 pts over CD and +2.8 pts over the best alternatives. Though modest in percentage, these improvements reliably recover thin structures and under-represented regions that CD/EMD overlook. Extensive ablations confirm stable performance across hyperparameters and across output resolutions, while full retraining on PCN and ShapeNet demonstrates generalization across datasets and backbones. Moreover, QAL-trained completions yield higher grasp scores under GraspNet evaluation, showing that improved coverage translates directly into more reliable robotic manipulation. QAL thus offers a principled, interpretable, and practical objective for robust 3D vision and safety-critical robotics pipelines
>
---
#### [new 072] Leveraging LLMs for reward function design in reinforcement learning control tasks
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文针对强化学习中奖励函数设计耗时、依赖人工经验的问题，提出LEARN-Opt框架。通过文本描述自动生成并评估奖励函数，无需预设指标或环境代码，实现无监督优化。实验表明其性能媲美先进方法，且可利用低成本LLM高效生成高质量奖励函数。**

- **链接: [https://arxiv.org/pdf/2511.19355v1](https://arxiv.org/pdf/2511.19355v1)**

> **作者:** Franklin Cardenoso; Wouter Caarls
>
> **摘要:** The challenge of designing effective reward functions in reinforcement learning (RL) represents a significant bottleneck, often requiring extensive human expertise and being time-consuming. Previous work and recent advancements in large language models (LLMs) have demonstrated their potential for automating the generation of reward functions. However, existing methodologies often require preliminary evaluation metrics, human-engineered feedback for the refinement process, or the use of environmental source code as context. To address these limitations, this paper introduces LEARN-Opt (LLM-based Evaluator and Analyzer for Reward functioN Optimization). This LLM-based, fully autonomous, and model-agnostic framework eliminates the need for preliminary metrics and environmental source code as context to generate, execute, and evaluate reward function candidates from textual descriptions of systems and task objectives. LEARN-Opt's main contribution lies in its ability to autonomously derive performance metrics directly from the system description and the task objective, enabling unsupervised evaluation and selection of reward functions. Our experiments indicate that LEARN-Opt achieves performance comparable to or better to that of state-of-the-art methods, such as EUREKA, while requiring less prior knowledge. We find that automated reward design is a high-variance problem, where the average-case candidate fails, requiring a multi-run approach to find the best candidates. Finally, we show that LEARN-Opt can unlock the potential of low-cost LLMs to find high-performing candidates that are comparable to, or even better than, those of larger models. This demonstrated performance affirms its potential to generate high-quality reward functions without requiring any preliminary human-defined metrics, thereby reducing engineering overhead and enhancing generalizability.
>
---
#### [new 073] CUS-GS: A Compact Unified Structured Gaussian Splatting Framework for Multimodal Scene Representation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出CUS-GS，一种紧凑的统一结构化高斯点阵框架，用于多模态场景表示。针对现有方法在语义理解与几何建模间的割裂问题，通过体素锚点结构融合多模态语义特征与3D几何，实现高效、一致的场景表征，显著提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2511.17904v1](https://arxiv.org/pdf/2511.17904v1)**

> **作者:** Yuhang Ming; Chenxin Fang; Xingyuan Yu; Fan Zhang; Weichen Dai; Wanzeng Kong; Guofeng Zhang
>
> **备注:** 15 pages, 8 figures, 4 tables
>
> **摘要:** Recent advances in Gaussian Splatting based 3D scene representation have shown two major trends: semantics-oriented approaches that focus on high-level understanding but lack explicit 3D geometry modeling, and structure-oriented approaches that capture spatial structures yet provide limited semantic abstraction. To bridge this gap, we present CUS-GS, a compact unified structured Gaussian Splatting representation, which connects multimodal semantic features with structured 3D geometry. Specifically, we design a voxelized anchor structure that constructs a spatial scaffold, while extracting multimodal semantic features from a set of foundation models (e.g., CLIP, DINOv2, SEEM). Moreover, we introduce a multimodal latent feature allocation mechanism to unify appearance, geometry, and semantics across heterogeneous feature spaces, ensuring a consistent representation across multiple foundation models. Finally, we propose a feature-aware significance evaluation strategy to dynamically guide anchor growing and pruning, effectively removing redundant or invalid anchors while maintaining semantic integrity. Extensive experiments show that CUS-GS achieves competitive performance compared to state-of-the-art methods using as few as 6M parameters - an order of magnitude smaller than the closest rival at 35M - highlighting the excellent trade off between performance and model efficiency of the proposed framework.
>
---
#### [new 074] scipy.spatial.transform: Differentiable Framework-Agnostic 3D Transformations in Python
- **分类: cs.LG; cs.RO**

- **简介: 该论文针对3D刚体变换在不同框架下实现不兼容的问题，提出一个框架无关的可微分3D变换库。通过支持JAX、PyTorch等数组库，实现GPU/TPU加速与自动微分，提升机器人、视觉等领域中旋转与平移计算的鲁棒性与效率。**

- **链接: [https://arxiv.org/pdf/2511.18157v1](https://arxiv.org/pdf/2511.18157v1)**

> **作者:** Martin Schuck; Alexander von Rohr; Angela P. Schoellig
>
> **备注:** Accepted as oral at the 1st Workshop on Differentiable Systems and Scientific Machine Learning @ EurIPS 2025
>
> **摘要:** Three-dimensional rigid-body transforms, i.e. rotations and translations, are central to modern differentiable machine learning pipelines in robotics, vision, and simulation. However, numerically robust and mathematically correct implementations, particularly on SO(3), are error-prone due to issues such as axis conventions, normalizations, composition consistency and subtle errors that only appear in edge cases. SciPy's spatial.transform module is a rigorously tested Python implementation. However, it historically only supported NumPy, limiting adoption in GPU-accelerated and autodiff-based workflows. We present a complete overhaul of SciPy's spatial.transform functionality that makes it compatible with any array library implementing the Python array API, including JAX, PyTorch, and CuPy. The revised implementation preserves the established SciPy interface while enabling GPU/TPU execution, JIT compilation, vectorized batching, and differentiation via native autodiff of the chosen backend. We demonstrate how this foundation supports differentiable scientific computing through two case studies: (i) scalability of 3D transforms and rotations and (ii) a JAX drone simulation that leverages SciPy's Rotation for accurate integration of rotational dynamics. Our contributions have been merged into SciPy main and will ship in the next release, providing a framework-agnostic, production-grade basis for 3D spatial math in differentiable systems and ML.
>
---
#### [new 075] Three-Dimensional Anatomical Data Generation Based on Artificial Neural Networks
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对手术规划中3D anatomical数据获取难的问题，提出基于神经网络的自动化生成方法。利用生物仿生水凝胶前列腺模型与定制超声扫描，结合3D GAN生成多样化的3D模型，实现高精度图像分割与三维重建，解决真实数据获取受限及软组织成像困难问题。**

- **链接: [https://arxiv.org/pdf/2511.19198v1](https://arxiv.org/pdf/2511.19198v1)**

> **作者:** Ann-Sophia Müller; Moonkwang Jeong; Meng Zhang; Jiyuan Tian; Arkadiusz Miernik; Stefanie Speidel; Tian Qiu
>
> **备注:** 6 pages, 4 figures, 1 table, IEEE International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Surgical planning and training based on machine learning requires a large amount of 3D anatomical models reconstructed from medical imaging, which is currently one of the major bottlenecks. Obtaining these data from real patients and during surgery is very demanding, if even possible, due to legal, ethical, and technical challenges. It is especially difficult for soft tissue organs with poor imaging contrast, such as the prostate. To overcome these challenges, we present a novel workflow for automated 3D anatomical data generation using data obtained from physical organ models. We additionally use a 3D Generative Adversarial Network (GAN) to obtain a manifold of 3D models useful for other downstream machine learning tasks that rely on 3D data. We demonstrate our workflow using an artificial prostate model made of biomimetic hydrogels with imaging contrast in multiple zones. This is used to physically simulate endoscopic surgery. For evaluation and 3D data generation, we place it into a customized ultrasound scanner that records the prostate before and after the procedure. A neural network is trained to segment the recorded ultrasound images, which outperforms conventional, non-learning-based computer vision techniques in terms of intersection over union (IoU). Based on the segmentations, a 3D mesh model is reconstructed, and performance feedback is provided.
>
---
#### [new 076] ArticFlow: Generative Simulation of Articulated Mechanisms
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出ArticFlow，一种用于生成可动机械结构的两阶段流匹配框架。针对动作依赖变形与数据稀缺难题，通过联合隐空间流与点流，实现动作可控的高质量生成与仿真，支持形态插值与跨动作泛化，在MuJoCo上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.17883v1](https://arxiv.org/pdf/2511.17883v1)**

> **作者:** Jiong Lin; Jinchen Ruan; Hod Lipson
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Recent advances in generative models have produced strong results for static 3D shapes, whereas articulated 3D generation remains challenging due to action-dependent deformations and limited datasets. We introduce ArticFlow, a two-stage flow matching framework that learns a controllable velocity field from noise to target point sets under explicit action control. ArticFlow couples (i) a latent flow that transports noise to a shape-prior code and (ii) a point flow that transports points conditioned on the action and the shape prior, enabling a single model to represent diverse articulated categories and generalize across actions. On MuJoCo Menagerie, ArticFlow functions both as a generative model and as a neural simulator: it predicts action-conditioned kinematics from a compact prior and synthesizes novel morphologies via latent interpolation. Compared with object-specific simulators and an action-conditioned variant of static point-cloud generators, ArticFlow achieves higher kinematic accuracy and better shape quality. Results show that action-conditioned flow matching is a practical route to controllable and high-quality articulated mechanism generation.
>
---
#### [new 077] First-order Sobolev Reinforcement Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出一种基于一阶Sobolev范数的强化学习方法，通过强制值函数与目标函数在值和梯度上的一致性，提升TD学习的精度。解决了传统TD学习仅匹配值而忽略局部几何的问题。工作是将导数一致性作为梯度目标融入现有算法（如DDPG、SAC），实现更快收敛与更稳定策略更新。**

- **链接: [https://arxiv.org/pdf/2511.19165v1](https://arxiv.org/pdf/2511.19165v1)**

> **作者:** Fabian Schramm; Nicolas Perrin-Gilbert; Justin Carpentier
>
> **备注:** Workshop paper at Differentiable Systems and Scientific Machine Learning, EurIPS 2025
>
> **摘要:** We propose a refinement of temporal-difference learning that enforces first-order Bellman consistency: the learned value function is trained to match not only the Bellman targets in value but also their derivatives with respect to states and actions. By differentiating the Bellman backup through differentiable dynamics, we obtain analytically consistent gradient targets. Incorporating these into the critic objective using a Sobolev-type loss encourages the critic to align with both the value and local geometry of the target function. This first-order TD matching principle can be seamlessly integrated into existing algorithms, such as Q-learning or actor-critic methods (e.g., DDPG, SAC), potentially leading to faster critic convergence and more stable policy gradients without altering their overall structure.
>
---
#### [new 078] ActDistill: General Action-Guided Self-Derived Distillation for Efficient Vision-Language-Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在机器人操作中计算开销大、推理延迟高的问题，提出ActDistill框架。通过动作引导的自蒸馏机制，将大型VLA模型的动作预测能力迁移到轻量级学生模型，利用图结构建模动作演化并引入动态路由，实现高效推理。实验表明，该方法在保持高性能的同时，计算量减少超50%，速度提升达1.67倍。**

- **链接: [https://arxiv.org/pdf/2511.18082v1](https://arxiv.org/pdf/2511.18082v1)**

> **作者:** Wencheng Ye; Tianshi Wang; Lei Zhu; Fengling Li; Guoli Yang
>
> **摘要:** Recent Vision-Language-Action (VLA) models have shown impressive flexibility and generalization, yet their deployment in robotic manipulation remains limited by heavy computational overhead and inference latency. In this work, we present ActDistill, a general action-guided self-derived distillation framework that transfers the action prediction capability of any existing VLA model to a lightweight counterpart. Unlike previous efficiency strategies that primarily emphasize vision-language correlations, ActDistill leverages action priors to guide knowledge transfer and model compression, achieving action-oriented efficiency for VLA models. Specifically, we employ a well-trained VLA model as the teacher and introduce a graph-structured encapsulation strategy to explicitly model the hierarchical evolution of action prediction. The student model, derived from the graph-encapsulated teacher, is further equipped with a dynamic router that adaptively selects computation paths based on action prediction demands, guided by hierarchical graph-informed supervision to ensure smooth and efficient evolution. During inference, graph-related auxiliary components are removed, allowing the student to execute only dynamically routed layers and predict high-precision actions with minimal computation and latency. Experiments on embodied benchmarks demonstrate that ActDistill achieves comparable or superior performance to full-scale VLA models while reducing computation by over 50% with up to 1.67 times speedup, thereby establishing a general paradigm toward efficient embodied intelligence.
>
---
#### [new 079] PhysGS: Bayesian-Inferred Gaussian Splatting for Physical Property Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出PhysGS，一种基于贝叶斯推理的3D高斯点阵方法，用于从视觉和语言先验中估计密集的物理属性（如摩擦、硬度等）。针对现有3D重建忽略物理属性的问题，该方法通过迭代更新材料信念并建模不确定性，实现空间连续的物理属性估计，在多个数据集上显著提升估计精度。**

- **链接: [https://arxiv.org/pdf/2511.18570v1](https://arxiv.org/pdf/2511.18570v1)**

> **作者:** Samarth Chopra; Jing Liang; Gershom Seneviratne; Dinesh Manocha
>
> **备注:** Submitted to CVPR 2026
>
> **摘要:** Understanding physical properties such as friction, stiffness, hardness, and material composition is essential for enabling robots to interact safely and effectively with their surroundings. However, existing 3D reconstruction methods focus on geometry and appearance and cannot infer these underlying physical properties. We present PhysGS, a Bayesian-inferred extension of 3D Gaussian Splatting that estimates dense, per-point physical properties from visual cues and vision--language priors. We formulate property estimation as Bayesian inference over Gaussian splats, where material and property beliefs are iteratively refined as new observations arrive. PhysGS also models aleatoric and epistemic uncertainties, enabling uncertainty-aware object and scene interpretation. Across object-scale (ABO-500), indoor, and outdoor real-world datasets, PhysGS improves accuracy of the mass estimation by up to 22.8%, reduces Shore hardness error by up to 61.2%, and lowers kinetic friction error by up to 18.1% compared to deterministic baselines. Our results demonstrate that PhysGS unifies 3D reconstruction, uncertainty modeling, and physical reasoning in a single, spatially continuous framework for dense physical property estimation. Additional results are available at https://samchopra2003.github.io/physgs.
>
---
#### [new 080] GContextFormer: A global context-aware hybrid multi-head attention approach with scaled additive aggregation for multimodal trajectory prediction
- **分类: cs.AI; cs.CV; cs.LG; cs.MA; cs.RO; cs.SI**

- **简介: 该论文针对无图多模态轨迹预测任务，解决地图依赖模型数据成本高及无图方法缺乏全局上下文导致意图错配的问题。提出GContextFormer，通过全局感知的混合注意力与缩放加性聚合，增强模式间意图对齐，实现更鲁棒、可解释的多路径预测。**

- **链接: [https://arxiv.org/pdf/2511.18874v1](https://arxiv.org/pdf/2511.18874v1)**

> **作者:** Yuzhi Chen; Yuanchang Xie; Lei Zhao; Pan Liu; Yajie Zou; Chen Wang
>
> **摘要:** Multimodal trajectory prediction generates multiple plausible future trajectories to address vehicle motion uncertainty from intention ambiguity and execution variability. However, HD map-dependent models suffer from costly data acquisition, delayed updates, and vulnerability to corrupted inputs, causing prediction failures. Map-free approaches lack global context, with pairwise attention over-amplifying straight patterns while suppressing transitional patterns, resulting in motion-intention misalignment. This paper proposes GContextFormer, a plug-and-play encoder-decoder architecture with global context-aware hybrid attention and scaled additive aggregation achieving intention-aligned multimodal prediction without map reliance. The Motion-Aware Encoder builds scene-level intention prior via bounded scaled additive aggregation over mode-embedded trajectory tokens and refines per-mode representations under shared global context, mitigating inter-mode suppression and promoting intention alignment. The Hierarchical Interaction Decoder decomposes social reasoning into dual-pathway cross-attention: a standard pathway ensures uniform geometric coverage over agent-mode pairs while a neighbor-context-enhanced pathway emphasizes salient interactions, with gating module mediating their contributions to maintain coverage-focus balance. Experiments on eight highway-ramp scenarios from TOD-VT dataset show GContextFormer outperforms state-of-the-art baselines. Compared to existing transformer models, GContextFormer achieves greater robustness and concentrated improvements in high-curvature and transition zones via spatial distributions. Interpretability is achieved through motion mode distinctions and neighbor context modulation exposing reasoning attribution. The modular architecture supports extensibility toward cross-domain multimodal reasoning tasks. Source: https://fenghy-chen.github.io/sources/.
>
---
#### [new 081] Multi-Agent Coordination in Autonomous Vehicle Routing: A Simulation-Based Study of Communication, Memory, and Routing Loops
- **分类: cs.MA; cs.LG; cs.RO**

- **简介: 该论文研究自主车辆协同路径规划中的路由循环问题，针对无记忆的通信重规划导致效率下降的问题，提出轻量级对象记忆管理（OMM）机制，通过共享障碍物黑名单减少冗余计算。实验表明，OMM可降低75.7%平均行驶时间，显著提升系统稳定性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.17656v1](https://arxiv.org/pdf/2511.17656v1)**

> **作者:** KM Khalid Saifullah; Daniel Palmer
>
> **摘要:** Multi-agent coordination is critical for next-generation autonomous vehicle (AV) systems, yet naive implementations of communication-based rerouting can lead to catastrophic performance degradation. This study investigates a fundamental problem in decentralized multi-agent navigation: routing loops, where vehicles without persistent obstacle memory become trapped in cycles of inefficient path recalculation. Through systematic simulation experiments involving 72 unique configurations across varying vehicle densities (15, 35, 55 vehicles) and obstacle frequencies (6, 20 obstacles), we demonstrate that memory-less reactive rerouting increases average travel time by up to 682% compared to baseline conditions. To address this, we introduce Object Memory Management (OMM), a lightweight mechanism enabling agents to retain and share knowledge of previously encountered obstacles. OMM operates by maintaining a distributed blacklist of blocked nodes, which each agent consults during Dijkstra-based path recalculation, effectively preventing redundant routing attempts. Our results show that OMM-enabled coordination reduces average travel time by 75.7% and wait time by 88% compared to memory-less systems, while requiring only 1.67 route recalculations per vehicle versus 9.83 in memory-less scenarios. This work provides empirical evidence that persistent, shared memory is not merely beneficial but essential for robust multi-agent coordination in dynamic environments. The findings have implications beyond autonomous vehicles, informing the design of decentralized systems in robotics, network routing, and distributed AI. We provide a comprehensive experimental analysis, including detailed scenario breakdowns, scalability assessments, and visual documentation of the routing loop phenomenon, demonstrating OMM's critical role in preventing detrimental feedback cycles in cooperative multi-agent systems.
>
---
#### [new 082] Beyond Description: Cognitively Benchmarking Fine-Grained Action for Embodied Agents
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对具身智能体在复杂物理环境中执行精细动作的能力评估问题，提出CFG-Bench基准，涵盖四类认知能力的多模态问答数据。通过系统评测发现主流多模态大模型在细粒度动作理解与高阶推理上存在显著不足，并验证了针对性微调可有效提升性能。**

- **链接: [https://arxiv.org/pdf/2511.18685v1](https://arxiv.org/pdf/2511.18685v1)**

> **作者:** Dayong Liu; Chao Xu; Weihong Chen; Suyu Zhang; Juncheng Wang; Jiankang Deng; Baigui Sun; Yang Liu
>
> **摘要:** Multimodal Large Language Models (MLLMs) show promising results as decision-making engines for embodied agents operating in complex, physical environments. However, existing benchmarks often prioritize high-level planning or spatial reasoning, leaving the fine-grained action intelligence required for embodied physical interaction underexplored. To address this gap, we introduce CFG-Bench, a new benchmark designed to systematically evaluate this crucial capability. CFG-Bench consists of 1,368 curated videos paired with 19,562 three-modalities question-answer pairs targeting four cognitive abilities: 1) Physical Interaction, 2) Temporal-Causal Relation, 3) Intentional Understanding, and 4) Evaluative Judgment. Together, these dimensions provide a systematic framework for assessing a model's ability to translate visual observations into actionable knowledge, moving beyond mere surface-level recognition. Our comprehensive evaluation on CFG-Bench reveals that leading MLLMs struggle to produce detailed instructions for physical interactions and exhibit profound limitations in the higher-order reasoning of intention and evaluation. Moreover, supervised fine-tuning (SFT) on our data demonstrates that teaching an MLLMs to articulate fine-grained actions directly translates to significant performance gains on established embodied benchmarks. Our analysis highlights these limitations and offers insights for developing more capable and grounded embodied agents.
>
---
#### [new 083] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在动态决策中忽视历史上下文的问题，提出AVA-VLA框架。通过引入基于信念状态的主动视觉注意力机制，利用递归状态动态聚焦关键视觉信息，将任务从MDP重构为POMDP，显著提升模型在机器人任务中的表现与真实世界迁移能力。**

- **链接: [https://arxiv.org/pdf/2511.18960v1](https://arxiv.org/pdf/2511.18960v1)**

> **作者:** Lei Xiao; Jifeng Li; Juntao Gao; Feiyang Ye; Yan Jin; Jingjing Qian; Jing Zhang; Yong Wu; Xiaoyuan Yu
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in embodied AI tasks. However, existing VLA models, often built upon Vision-Language Models (VLMs), typically process dense visual inputs independently at each timestep. This approach implicitly models the task as a Markov Decision Process (MDP). However, this history-agnostic design is suboptimal for effective visual token processing in dynamic sequential decision-making, as it fails to leverage the context of history. To address this limitation, we reformulate the problem from a Partially Observable Markov Decision Process (POMDP) perspective and propose a novel framework named AVA-VLA. Inspired by the POMDP that the action generation should be conditioned on the belief state. AVA-VLA introduces Active Visual Attention (AVA) to dynamically modulate visual processing. It achieves this by leveraging the recurrent state, which is a neural approximation of the agent's belief state derived from the previous decision step. Specifically, the AVA module uses the recurrent state to compute the soft weights to actively process task-relevant visual tokens based on its historical context. Comprehensive evaluations demonstrate that AVA-VLA achieves state-of-the-art performance across popular robotic benchmarks, including LIBERO and CALVIN. Furthermore, real-world deployments on a dual-arm robot platform validate the framework's practical applicability and robust sim-to-real transferability.
>
---
#### [new 084] QuickLAP: Quick Language-Action Preference Learning for Autonomous Driving Agents
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出QuickLAP，一种融合语言与物理反馈的快速奖励学习框架，用于自动驾驶代理。针对单一反馈模态不完整的问题，通过贝叶斯方法实时融合语言与动作反馈，利用LLM解析语言中的偏好信息，提升奖励学习准确性与可解释性。在模拟驾驶中显著降低误差，并获用户偏好验证。**

- **链接: [https://arxiv.org/pdf/2511.17855v1](https://arxiv.org/pdf/2511.17855v1)**

> **作者:** Jordan Abi Nader; David Lee; Nathaniel Dennler; Andreea Bobu
>
> **摘要:** Robots must learn from both what people do and what they say, but either modality alone is often incomplete: physical corrections are grounded but ambiguous in intent, while language expresses high-level goals but lacks physical grounding. We introduce QuickLAP: Quick Language-Action Preference learning, a Bayesian framework that fuses physical and language feedback to infer reward functions in real time. Our key insight is to treat language as a probabilistic observation over the user's latent preferences, clarifying which reward features matter and how physical corrections should be interpreted. QuickLAP uses Large Language Models (LLMs) to extract reward feature attention masks and preference shifts from free-form utterances, which it integrates with physical feedback in a closed-form update rule. This enables fast, real-time, and robust reward learning that handles ambiguous feedback. In a semi-autonomous driving simulator, QuickLAP reduces reward learning error by over 70% compared to physical-only and heuristic multimodal baselines. A 15-participant user study further validates our approach: participants found QuickLAP significantly more understandable and collaborative, and preferred its learned behavior over baselines. Code is available at https://github.com/MIT-CLEAR-Lab/QuickLAP.
>
---
## 更新

#### [replaced 001] Ionospheric and Plasmaspheric Delay Characterization for Lunar Terrestrial GNSS Receivers with Global Core Plasma Model
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2510.10059v2](https://arxiv.org/pdf/2510.10059v2)**

> **作者:** Keidai Iiyama; Grace Gao
>
> **备注:** Submitted NAVIGATION: Journal of the Institute of Navigation
>
> **摘要:** Recent advancements in lunar positioning, navigation, and timing (PNT) have demonstrated that terrestrial GNSS signals, including weak sidelobe transmissions, can be exploited for lunar spacecraft positioning and timing. While GNSS-based navigation at the Moon has been validated recently, unmodeled ionospheric and plasmaspheric delays remain a significant error source, particularly given the unique signal geometry and extended propagation paths. This paper characterizes these delays using the Global Core Plasma Model (GCPM) and a custom low-cost ray-tracing algorithm that iteratively solves for bent signal paths. We simulate first-, second-, and third-order group delays, as well as excess path length from ray bending, for GNSS signals received at both lunar orbit and the lunar south pole under varying solar and geomagnetic conditions. Results show that mean group delays are typically on the order of 1 m, but can exceed 100 m for low-altitude ray paths during high solar activity, while bending delays are generally smaller but non-negligible for low-altitude ray paths. We also quantify the influence of signal frequency, geomagnetic $K_p$ index, and solar R12 index. These findings inform the design of robust positioning and timing algorithms that utilize terrestrial GNSS signals.
>
---
#### [replaced 002] Simultaneous Localization and 3D-Semi Dense Mapping for Micro Drones Using Monocular Camera and Inertial Sensors
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2511.14335v2](https://arxiv.org/pdf/2511.14335v2)**

> **作者:** Jeryes Danial; Yosi Ben Asher; Itzik Klein
>
> **摘要:** Monocular simultaneous localization and mapping (SLAM) algorithms estimate drone poses and build a 3D map using a single camera. Current algorithms include sparse methods that lack detailed geometry, while learning-driven approaches produce dense maps but are computationally intensive. Monocular SLAM also faces scale ambiguities, which affect its accuracy. To address these challenges, we propose an edge-aware lightweight monocular SLAM system combining sparse keypoint-based pose estimation with dense edge reconstruction. Our method employs deep learning-based depth prediction and edge detection, followed by optimization to refine keypoints and edges for geometric consistency, without relying on global loop closure or heavy neural computations. We fuse inertial data with vision by using an extended Kalman filter to resolve scale ambiguity and improve accuracy. The system operates in real time on low-power platforms, as demonstrated on a DJI Tello drone with a monocular camera and inertial sensors. In addition, we demonstrate robust autonomous navigation and obstacle avoidance in indoor corridors and on the TUM RGBD dataset. Our approach offers an effective, practical solution to real-time mapping and navigation in resource-constrained environments.
>
---
#### [replaced 003] RynnVLA-002: A Unified Vision-Language-Action and World Model
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2511.17502v2](https://arxiv.org/pdf/2511.17502v2)**

> **作者:** Jun Cen; Siteng Huang; Yuqian Yuan; Kehan Li; Hangjie Yuan; Chaohui Yu; Yuming Jiang; Jiayan Guo; Xin Li; Hao Luo; Fan Wang; Deli Zhao; Hao Chen
>
> **摘要:** We introduce RynnVLA-002, a unified Vision-Language-Action (VLA) and world model. The world model leverages action and visual inputs to predict future image states, learning the underlying physics of the environment to refine action generation. Conversely, the VLA model produces subsequent actions from image observations, enhancing visual understanding and supporting the world model's image generation. The unified framework of RynnVLA-002 enables joint learning of environmental dynamics and action planning. Our experiments show that RynnVLA-002 surpasses individual VLA and world models, demonstrating their mutual enhancement. We evaluate RynnVLA-002 in both simulation and real-world robot tasks. RynnVLA-002 achieves 97.4% success rate on the LIBERO simulation benchmark without pretraining, while in real-world LeRobot experiments, its integrated world model boosts the overall success rate by 50%.
>
---
#### [replaced 004] Multi-Timescale Hierarchical Reinforcement Learning for Unified Behavior and Control of Autonomous Driving
- **分类: cs.RO; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.23771v3](https://arxiv.org/pdf/2506.23771v3)**

> **作者:** Guizhe Jin; Zhuoren Li; Bo Leng; Ran Yu; Lu Xiong; Chen Sun
>
> **备注:** 8 pages, accepted for publication in IEEE Robotics and Automation Letters (RAL)
>
> **摘要:** Reinforcement Learning (RL) is increasingly used in autonomous driving (AD) and shows clear advantages. However, most RL-based AD methods overlook policy structure design. An RL policy that only outputs short-timescale vehicle control commands results in fluctuating driving behavior due to fluctuations in network outputs, while one that only outputs long-timescale driving goals cannot achieve unified optimality of driving behavior and control. Therefore, we propose a multi-timescale hierarchical reinforcement learning approach. Our approach adopts a hierarchical policy structure, where high- and low-level RL policies are unified-trained to produce long-timescale motion guidance and short-timescale control commands, respectively. Therein, motion guidance is explicitly represented by hybrid actions to capture multimodal driving behaviors on structured road and support incremental low-level extend-state updates. Additionally, a hierarchical safety mechanism is designed to ensure multi-timescale safety. Evaluation in simulator-based and HighD dataset-based highway multi-lane scenarios demonstrates that our approach significantly improves AD performance, effectively increasing driving efficiency, action consistency and safety.
>
---
#### [replaced 005] M2R2: MultiModal Robotic Representation for Temporal Action Segmentation
- **分类: cs.RO; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.18662v2](https://arxiv.org/pdf/2504.18662v2)**

> **作者:** Daniel Sliwowski; Dongheui Lee
>
> **备注:** 8 pages, 6 figures, 2 tables
>
> **摘要:** Temporal action segmentation (TAS) has long been a key area of research in both robotics and computer vision. In robotics, algorithms have primarily focused on leveraging proprioceptive information to determine skill boundaries, with recent approaches in surgical robotics incorporating vision. In contrast, computer vision typically relies on exteroceptive sensors, such as cameras. Existing multimodal TAS models in robotics integrate feature fusion within the model, making it difficult to reuse learned features across different models. Meanwhile, pretrained vision-only feature extractors commonly used in computer vision struggle in scenarios with limited object visibility. In this work, we address these challenges by proposing M2R2, a multimodal feature extractor tailored for TAS, which combines information from both proprioceptive and exteroceptive sensors. We introduce a novel pretraining strategy that enables the reuse of learned features across multiple TAS models. Our method achieves state-of-the-art performance on the REASSEMBLE dataset, a challenging multimodal robotic assembly dataset, outperforming existing robotic action segmentation models by 46.6%. Additionally, we conduct an extensive ablation study to evaluate the contribution of different modalities in robotic TAS tasks.
>
---
#### [replaced 006] MonoMPC: Monocular Vision Based Navigation with Learned Collision Model and Risk-Aware Model Predictive Control
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2508.07387v2](https://arxiv.org/pdf/2508.07387v2)**

> **作者:** Basant Sharma; Prajyot Jadhav; Pranjal Paul; K. Madhava Krishna; Arun Kumar Singh
>
> **摘要:** Navigating unknown environments with a single RGB camera is challenging, as the lack of depth information prevents reliable collision-checking. While some methods use estimated depth to build collision maps, we found that depth estimates from vision foundation models are too noisy for zero-shot navigation in cluttered environments. We propose an alternative approach: instead of using noisy estimated depth for direct collision-checking, we use it as a rich context input to a learned collision model. This model predicts the distribution of minimum obstacle clearance that the robot can expect for a given control sequence. At inference, these predictions inform a risk-aware MPC planner that minimizes estimated collision risk. We proposed a joint learning pipeline that co-trains the collision model and risk metric using both safe and unsafe trajectories. Crucially, our joint-training ensures well calibrated uncertainty in our collision model that improves navigation in highly cluttered environments. Consequently, real-world experiments show reductions in collision-rate and improvements in goal reaching and speed over several strong baselines.
>
---
#### [replaced 007] DeepFleet: Multi-Agent Foundation Models for Mobile Robots
- **分类: cs.RO; cs.MA**

- **链接: [https://arxiv.org/pdf/2508.08574v2](https://arxiv.org/pdf/2508.08574v2)**

> **作者:** Ameya Agaskar; Sriram Siva; William Pickering; Kyle O'Brien; Charles Kekeh; Ang Li; Brianna Gallo Sarker; Alicia Chua; Mayur Nemade; Charun Thattai; Jiaming Di; Isaac Iyengar; Ramya Dharoor; Dino Kirouani; Jimmy Erskine; Tamir Hegazy; Scott Niekum; Usman A. Khan; Federico Pecora; Joseph W. Durham
>
> **备注:** 27 pages, 10 figures, 2 tables
>
> **摘要:** We introduce DeepFleet, a suite of foundation models designed to support coordination and planning for large-scale mobile robot fleets. These models are trained on fleet movement data, including robot positions, goals, and interactions, from hundreds of thousands of robots in Amazon warehouses worldwide. DeepFleet consists of four architectures that each embody a distinct inductive bias and collectively explore key points in the design space for multi-agent foundation models: the robot-centric (RC) model is an autoregressive decision transformer operating on neighborhoods of individual robots; the robot-floor (RF) model uses a transformer with cross-attention between robots and the warehouse floor; the image-floor (IF) model applies convolutional encoding to a multi-channel image representation of the full fleet; and the graph-floor (GF) model combines temporal attention with graph neural networks for spatial relationships. In this paper, we describe these models and present our evaluation of the impact of these design choices on prediction task performance. We find that the robot-centric and graph-floor models, which both use asynchronous robot state updates and incorporate the localized structure of robot interactions, show the most promise. We also present experiments that show that these two models can make effective use of larger warehouses operation datasets as the models are scaled up.
>
---
#### [replaced 008] Towards Sensor Data Abstraction of Autonomous Vehicle Perception Systems
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2105.06896v3](https://arxiv.org/pdf/2105.06896v3)**

> **作者:** Hannes Reichert; Lukas Lang; Kevin Rösch; Daniel Bogdoll; Konrad Doll; Bernhard Sick; Hans-Christian Reuss; Christoph Stiller; J. Marius Zöllner
>
> **备注:** Hannes Reichert, Lukas Lang, Kevin Rösch and Daniel Bogdoll contributed equally. Accepted for publication at ISC2 2021
>
> **摘要:** Full-stack autonomous driving perception modules usually consist of data-driven models based on multiple sensor modalities. However, these models might be biased to the sensor setup used for data acquisition. This bias can seriously impair the perception models' transferability to new sensor setups, which continuously occur due to the market's competitive nature. We envision sensor data abstraction as an interface between sensor data and machine learning applications for highly automated vehicles (HAD). For this purpose, we review the primary sensor modalities, camera, lidar, and radar, published in autonomous-driving related datasets, examine single sensor abstraction and abstraction of sensor setups, and identify critical paths towards an abstraction of sensor data from multiple perception configurations.
>
---
#### [replaced 009] Greedy Heuristics for Sampling-Based Motion Planning in High-Dimensional State Spaces
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2405.03411v3](https://arxiv.org/pdf/2405.03411v3)**

> **作者:** Phone Thiha Kyaw; Anh Vu Le; Rajesh Elara Mohan; Jonathan Kelly
>
> **备注:** Submitted to the Autonomous Robots journal
>
> **摘要:** Informed sampling techniques accelerate the convergence of sampling-based motion planners by biasing sampling toward regions of the state space that are most likely to yield better solutions. However, when the current solution path contains redundant or tortuous segments, the resulting informed subset may remain unnecessarily large, slowing convergence. Our prior work addressed this issue by introducing the greedy informed set, which reduces the sampling region based on the maximum heuristic cost along the current solution path. In this article, we formally characterize the behavior of the greedy informed set within Rapidly-exploring Random Tree (RRT*)-like planners and analyze how greedy sampling affects exploration and asymptotic optimality. We then present Greedy RRT* (G-RRT*), a bi-directional anytime variant of RRT* that leverages the greedy informed set to focus sampling in the most promising regions of the search space. Experiments on abstract planning benchmarks, manipulation tasks from the MotionBenchMaker dataset, and a dual-arm Barrett WAM problem demonstrate that G-RRT* rapidly finds initial solutions and converges asymptotically to optimal paths, outperforming state-of-the-art sampling-based planners.
>
---
#### [replaced 010] Continuous Gaussian Process Pre-Optimization for Asynchronous Event-Inertial Odometry
- **分类: cs.RO; eess.SY**

- **链接: [https://arxiv.org/pdf/2412.08909v2](https://arxiv.org/pdf/2412.08909v2)**

> **作者:** Zhixiang Wang; Xudong Li; Yizhai Zhang; Fan Zhang; Panfeng Huang
>
> **备注:** 8pages
>
> **摘要:** Event cameras, as bio-inspired sensors, are asynchronously triggered with high-temporal resolution compared to intensity cameras. Recent work has focused on fusing the event measurements with inertial measurements to enable ego-motion estimation in high-speed and HDR environments. However, existing methods predominantly rely on IMU preintegration designed mainly for synchronous sensors and discrete-time frameworks. In this paper, we propose a continuous-time preintegration method based on the Temporal Gaussian Process (TGP) called GPO. Concretely, we model the preintegration as a time-indexed motion trajectory and leverage an efficient two-step optimization to initialize the precision preintegration pseudo-measurements. Our method realizes a linear and constant time cost for initialization and query, respectively. To further validate the proposal, we leverage the GPO to design an asynchronous event-inertial odometry and compare with other asynchronous fusion schemes within the same odometry system. Experiments conducted on both public and own-collected datasets demonstrate that the proposed GPO offers significant advantages in terms of precision and efficiency, outperforming existing approaches in handling asynchronous sensor fusion.
>
---
#### [replaced 011] Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models
- **分类: cs.RO; cs.AI; cs.HC**

- **链接: [https://arxiv.org/pdf/2511.02162v4](https://arxiv.org/pdf/2511.02162v4)**

> **作者:** Alexander Htet Kyaw; Richa Gupta; Dhruv Shah; Anoop Sinha; Kory Mathewson; Stefanie Pender; Sachin Chitta; Yotto Koga; Faez Ahmed; Lawrence Sass; Randall Davis
>
> **备注:** Accepted to NeurIPS 2025, Conference on Neural Information Processing Systems, Creative AI Track
>
> **摘要:** Advances in 3D generative AI have enabled the creation of physical objects from text prompts, but challenges remain in creating objects involving multiple component types. We present a pipeline that integrates 3D generative AI with vision-language models (VLMs) to enable the robotic assembly of multi-component objects from natural language. Our method leverages VLMs for zero-shot, multi-modal reasoning about geometry and functionality to decompose AI-generated meshes into multi-component 3D models using predefined structural and panel components. We demonstrate that a VLM is capable of determining which mesh regions need panel components in addition to structural components, based on the object's geometry and functionality. Evaluation across test objects shows that users preferred the VLM-generated assignments 90.6% of the time, compared to 59.4% for rule-based and 2.5% for random assignment. Lastly, the system allows users to refine component assignments through conversational feedback, enabling greater human control and agency in making physical objects with generative AI and robotics.
>
---
#### [replaced 012] Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2511.17373v2](https://arxiv.org/pdf/2511.17373v2)**

> **作者:** Yixuan Pan; Ruoyi Qiao; Li Chen; Kashyap Chitta; Liang Pan; Haoguang Mai; Qingwen Bu; Hao Zhao; Cunyuan Zheng; Ping Luo; Hongyang Li
>
> **摘要:** Humanoid robots are envisioned to perform a wide range of tasks in human-centered environments, requiring controllers that combine agility with robust balance. Recent advances in locomotion and whole-body tracking have enabled impressive progress in either agile dynamic skills or stability-critical behaviors, but existing methods remain specialized, focusing on one capability while compromising the other. In this work, we introduce AMS (Agility Meets Stability), the first framework that unifies both dynamic motion tracking and extreme balance maintenance in a single policy. Our key insight is to leverage heterogeneous data sources: human motion capture datasets that provide rich, agile behaviors, and physically constrained synthetic balance motions that capture stability configurations. To reconcile the divergent optimization goals of agility and stability, we design a hybrid reward scheme that applies general tracking objectives across all data while injecting balance-specific priors only into synthetic motions. Further, an adaptive learning strategy with performance-driven sampling and motion-specific reward shaping enables efficient training across diverse motion distributions. We validate AMS extensively in simulation and on a real Unitree G1 humanoid. Experiments demonstrate that a single policy can execute agile skills such as dancing and running, while also performing zero-shot extreme balance motions like Ip Man's Squat, highlighting AMS as a versatile control paradigm for future humanoid applications.
>
---
#### [replaced 013] Evo-0: Vision-Language-Action Model with Implicit Spatial Understanding
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.00416v3](https://arxiv.org/pdf/2507.00416v3)**

> **作者:** Tao Lin; Gen Li; Yilei Zhong; Yanwen Zou; Yuxin Du; Jiting Liu; Encheng Gu; Bo Zhao
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising framework for enabling generalist robots capable of perceiving, reasoning, and acting in the real world. These models usually build upon pretrained Vision-Language Models (VLMs), which excel at semantic understanding due to large-scale image and text pretraining. However, existing VLMs typically lack precise spatial understanding capabilities, as they are primarily tuned on 2D image-text pairs without 3D supervision. To address this limitation, recent approaches have incorporated explicit 3D inputs such as point clouds or depth maps, but this necessitates additional depth sensors or pre-trained depth estimation models, which may yield defective results. In contrast, our work introduces a plug-and-play module that implicitly incorporates 3D geometry features into VLA models by leveraging an off-the-shelf visual geometry foundation model. This integration provides the model with depth-aware visual representations, improving its ability to understand the geometric structure of the scene and the spatial relationships among objects from RGB images alone. We evaluate our method on a set of spatially challenging tasks in both simulation and the real world. Extensive evaluations show that our method significantly improves the performance of state-of-the-art VLA models across diverse scenarios.
>
---
#### [replaced 014] Extremum Seeking Controlled Wiggling for Tactile Insertion
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2410.02595v2](https://arxiv.org/pdf/2410.02595v2)**

> **作者:** Levi Burner; Pavan Mantripragada; Gabriele M. Caddeo; Lorenzo Natale; Cornelia Fermüller; Yiannis Aloimonos
>
> **备注:** 8 pages, 6 figures, 4 tables
>
> **摘要:** When humans perform complex insertion tasks such as pushing a cup into a cupboard, routing a cable, or putting a key in a lock, they wiggle the object and adapt the process through tactile feedback. A similar robotic approach has not been developed. We study an extremum seeking control law that wiggles end effector pose to maximize insertion depth while minimizing strain measured by a GelSight Mini sensor. Evaluation is conducted on four keys featuring complex geometry and five assembly tasks featuring basic geometry. On keys, the algorithm achieves 71% success rate over 120 trials with 6-DOF perturbations, 84% over 240 trials with 1-DOF perturbations, and 75% over 40 trials initialized with vision. It significantly outperforms a baseline optimizer, CMA-ES, that replaces wiggling with random sampling. When tested on a state-of-the-art assembly benchmark featuring basic geometry, it achieves 98% over 50 vision-initialized trials. The benchmark's most similar baseline, which was trained on the objects, achieved 86%. These results, realized without contact modeling or learning, show that closed loop wiggling based on tactile feedback is a robust paradigm for robotic insertion.
>
---
#### [replaced 015] DragonFly: Single mmWave Radar 3D Localization of Highly Dynamic Tags in GPS-Denied Environments
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2507.04602v2](https://arxiv.org/pdf/2507.04602v2)**

> **作者:** Skanda Harisha; Jimmy G. D. Hester; Aline Eid
>
> **备注:** Accepted in ACM Mobicom'25
>
> **摘要:** The accurate localization and tracking of dynamic targets, such as equipment, people, vehicles, drones, robots, and the assets that they interact with in GPS-denied indoor environments is critical to enabling safe and efficient operations in the next generation of spatially aware industrial facilities. This paper presents DragonFly , a 3D localization system of highly dynamic backscatter tags using a single MIMO mmWave radar. The system delivers the first demonstration of a mmWave backscatter system capable of exploiting the capabilities of MIMO radars for the 3D localization of mmID tags moving at high speeds and accelerations at long ranges by introducing a critical Doppler disambiguation algorithm and a fully integrated cross-polarized dielectric lens-based mmID tag consuming a mere 68 uW. DragonFly was extensively evaluated in static and dynamic configurations, including on a flying quadcopter, and benchmarked against multiple baselines, demonstrating its ability to track the positions of multiple tags with a median 3D accuracy of 12 cm at speeds and acceleration on the order of 10 m/s and 4 m/s^2 and at ranges of up to 50m.
>
---
#### [replaced 016] Anomaly Detection in Autonomous Driving: A Survey
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2204.07974v2](https://arxiv.org/pdf/2204.07974v2)**

> **作者:** Daniel Bogdoll; Maximilian Nitsche; J. Marius Zöllner
>
> **备注:** Daniel Bogdoll and Maximilian Nitsche contributed equally. Accepted for publication at CVPR 2022 WAD workshop
>
> **摘要:** Nowadays, there are outstanding strides towards a future with autonomous vehicles on our roads. While the perception of autonomous vehicles performs well under closed-set conditions, they still struggle to handle the unexpected. This survey provides an extensive overview of anomaly detection techniques based on camera, lidar, radar, multimodal and abstract object level data. We provide a systematization including detection approach, corner case level, ability for an online application, and further attributes. We outline the state-of-the-art and point out current research gaps.
>
---
#### [replaced 017] A Target-based Multi-LiDAR Multi-Camera Extrinsic Calibration System
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.16621v2](https://arxiv.org/pdf/2507.16621v2)**

> **作者:** Lorenzo Gentilini; Pierpaolo Serio; Valentina Donzella; Lorenzo Pollini
>
> **备注:** RiTA 2025 Accepted, 13 Pages, 6 Figures and 2 Tables
>
> **摘要:** Extrinsic Calibration represents the cornerstone of autonomous driving. Its accuracy plays a crucial role in the perception pipeline, as any errors can have implications for the safety of the vehicle. Modern sensor systems collect different types of data from the environment, making it harder to align the data. To this end, we propose a target-based extrinsic calibration system tailored for a multi-LiDAR and multi-camera sensor suite. This system enables cross-calibration between LiDARs and cameras with limited prior knowledge using a custom ChArUco board and a tailored nonlinear optimization method. We test the system with real-world data gathered in a warehouse. Results demonstrated the effectiveness of the proposed method, highlighting the feasibility of a unique pipeline tailored for various types of sensors.
>
---
#### [replaced 018] Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2501.10100v4](https://arxiv.org/pdf/2501.10100v4)**

> **作者:** Chenhao Li; Andreas Krause; Marco Hutter
>
> **摘要:** Learning robust and generalizable world models is crucial for enabling efficient and scalable robotic control in real-world environments. In this work, we introduce a novel framework for learning world models that accurately capture complex, partially observable, and stochastic dynamics. The proposed method employs a dual-autoregressive mechanism and self-supervised training to achieve reliable long-horizon predictions without relying on domain-specific inductive biases, ensuring adaptability across diverse robotic tasks. We further propose a policy optimization framework that leverages world models for efficient training in imagined environments and seamless deployment in real-world systems. This work advances model-based reinforcement learning by addressing the challenges of long-horizon prediction, error accumulation, and sim-to-real transfer. By providing a scalable and robust framework, the introduced methods pave the way for adaptive and efficient robotic systems in real-world applications.
>
---
#### [replaced 019] Learning Primitive Embodied World Models: Towards Scalable Robotic Learning
- **分类: cs.RO; cs.AI; cs.MM**

- **链接: [https://arxiv.org/pdf/2508.20840v3](https://arxiv.org/pdf/2508.20840v3)**

> **作者:** Qiao Sun; Liujia Yang; Wei Tang; Wei Huang; Kaixin Xu; Yongchao Chen; Mingyu Liu; Jiange Yang; Haoyi Zhu; Yating Wang; Tong He; Yilun Chen; Xili Dai; Nanyang Ye; Qinying Gu
>
> **摘要:** While video-generation-based embodied world models have gained increasing attention, their reliance on large-scale embodied interaction data remains a key bottleneck. The scarcity, difficulty of collection, and high dimensionality of embodied data fundamentally limit the alignment granularity between language and actions and exacerbate the challenge of long-horizon video generation--hindering generative models from achieving a "GPT moment" in the embodied domain. There is a naive observation: the diversity of embodied data far exceeds the relatively small space of possible primitive motions. Based on this insight, we propose a novel paradigm for world modeling--Primitive Embodied World Models (PEWM). By restricting video generation to fixed short horizons, our approach 1) enables fine-grained alignment between linguistic concepts and visual representations of robotic actions, 2) reduces learning complexity, 3) improves data efficiency in embodied data collection, and 4) decreases inference latency. By equipping with a modular Vision-Language Model (VLM) planner and a Start-Goal heatmap Guidance mechanism (SGG), PEWM further enables flexible closed-loop control and supports compositional generalization of primitive-level policies over extended, complex tasks. Our framework leverages the spatiotemporal vision priors in video models and the semantic awareness of VLMs to bridge the gap between fine-grained physical interaction and high-level reasoning, paving the way toward scalable, interpretable, and general-purpose embodied intelligence.
>
---
#### [replaced 020] ResAD: Normalized Residual Trajectory Modeling for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2510.08562v2](https://arxiv.org/pdf/2510.08562v2)**

> **作者:** Zhiyu Zheng; Shaoyu Chen; Haoran Yin; Xinbang Zhang; Jialv Zou; Xinggang Wang; Qian Zhang; Lefei Zhang
>
> **摘要:** End-to-end autonomous driving (E2EAD) systems, which learn to predict future trajectories directly from sensor data, are fundamentally challenged by the inherent spatio-temporal imbalance of trajectory data. This imbalance creates a significant optimization burden, causing models to learn spurious correlations instead of robust driving logic, while also prioritizing uncertain, distant predictions, thereby compromising immediate safety. To address these issues, we propose ResAD, a novel Normalized Residual Trajectory Modeling framework. Instead of predicting the future trajectory directly, our approach reframes and simplifies the learning task by predicting the residual deviation from a deterministic inertial reference. This inertial reference serves as a strong physical prior, compelling the model to move beyond simple pattern-matching and instead focus its capacity on learning the necessary, context-driven deviations (e.g., traffic rules, obstacles) from this default, inertially-guided path. To mitigate the optimization imbalance caused by uncertain, long-term horizons, ResAD further incorporates Point-wise Normalization of the predicted residual. This technique re-weights the optimization objective, preventing large-magnitude errors associated with distant, uncertain waypoints from dominating the learning signal. On the NAVSIM v1 and v2 benchmarks, ResAD achieves state-of-the-art results of 88.8 PDMS and 85.5 EPDMS with only two denoising steps, demonstrating that ResAD significantly simplifies the learning task and improves planning performance. The code will be released to facilitate further research.
>
---
#### [replaced 021] Advancing Autonomous Driving: DepthSense with Radar and Spatial Attention
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2109.05265v4](https://arxiv.org/pdf/2109.05265v4)**

> **作者:** Muhamamd Ishfaq Hussain; Zubia Naz; Muhammad Aasim Rafique; Moongu Jeon
>
> **摘要:** Depth perception is crucial for spatial understanding and has traditionally been achieved through stereoscopic imaging. However, the precision of depth estimation using stereoscopic methods depends on the accurate calibration of binocular vision sensors. Monocular cameras, while more accessible, often suffer from reduced accuracy, especially under challenging imaging conditions. Optical sensors, too, face limitations in adverse environments, leading researchers to explore radar technology as a reliable alternative. Although radar provides coarse but accurate signals, its integration with fine-grained monocular camera data remains underexplored. In this research, we propose DepthSense, a novel radar-assisted monocular depth enhancement approach. DepthSense employs an encoder-decoder architecture, a Radar Residual Network, feature fusion with a spatial attention mechanism, and an ordinal regression layer to deliver precise depth estimations. We conducted extensive experiments on the nuScenes dataset to validate the effectiveness of DepthSense. Our methodology not only surpasses existing approaches in quantitative performance but also reduces parameter complexity and inference times. Our findings demonstrate that DepthSense represents a significant advancement over traditional stereo methods, offering a robust and efficient solution for depth estimation in autonomous driving. By leveraging the complementary strengths of radar and monocular camera data, DepthSense sets a new benchmark in the field, paving the way for more reliable and accurate spatial perception systems.
>
---
#### [replaced 022] Unreal Robotics Lab: A High-Fidelity Robotics Simulator with Advanced Physics and Rendering
- **分类: cs.RO; cs.CV; cs.GR; cs.LG**

- **链接: [https://arxiv.org/pdf/2504.14135v2](https://arxiv.org/pdf/2504.14135v2)**

> **作者:** Jonathan Embley-Riches; Jianwei Liu; Simon Julier; Dimitrios Kanoulas
>
> **摘要:** High-fidelity simulation is essential for robotics research, enabling safe and efficient testing of perception, control, and navigation algorithms. However, achieving both photorealistic rendering and accurate physics modeling remains a challenge. This paper presents a novel simulation framework, the Unreal Robotics Lab (URL), that integrates the advanced rendering capabilities of the Unreal Engine with MuJoCo's high-precision physics simulation. Our approach enables realistic robotic perception while maintaining accurate physical interactions, facilitating benchmarking and dataset generation for vision-based robotics applications. The system supports complex environmental effects, such as smoke, fire, and water dynamics, which are critical to evaluating robotic performance under adverse conditions. We benchmark visual navigation and SLAM methods within our framework, demonstrating its utility for testing real-world robustness in controlled yet diverse scenarios. By bridging the gap between physics accuracy and photorealistic rendering, our framework provides a powerful tool for advancing robotics research and sim-to-real transfer. Our open-source framework is available at https://unrealroboticslab.github.io/.
>
---
#### [replaced 023] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.06659v3](https://arxiv.org/pdf/2506.06659v3)**

> **作者:** Wenhao Yao; Zhenxin Li; Shiyi Lan; Zi Wang; Xinglong Sun; Jose M. Alvarez; Zuxuan Wu
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Autonomous vehicles must navigate safely in complex driving environments. Imitating a single expert trajectory, as in regression-based approaches, usually does not explicitly assess the safety of the predicted trajectory. Selection-based methods address this by generating and scoring multiple trajectory candidates and predicting the safety score for each. However, they face optimization challenges in precisely selecting the best option from thousands of candidates and distinguishing subtle but safety-critical differences, especially in rare and challenging scenarios. We propose DriveSuprim to overcome these challenges and advance the selection-based paradigm through a coarse-to-fine paradigm for progressive candidate filtering, a rotation-based augmentation method to improve robustness in out-of-distribution scenarios, and a self-distillation framework to stabilize training. DriveSuprim achieves state-of-the-art performance, reaching 93.5% PDMS in NAVSIM v1 and 87.1% EPDMS in NAVSIM v2 without extra data, with 83.02 Driving Score and 60.00 Success Rate on the Bench2Drive benchmark, demonstrating superior planning capabilities in various driving scenarios.
>
---
#### [replaced 024] Autonomous Vehicle Path Planning by Searching With Differentiable Simulation
- **分类: cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2511.11043v2](https://arxiv.org/pdf/2511.11043v2)**

> **作者:** Asen Nachkov; Jan-Nico Zaech; Danda Pani Paudel; Xi Wang; Luc Van Gool
>
> **摘要:** Planning allows an agent to safely refine its actions before executing them in the real world. In autonomous driving, this is crucial to avoid collisions and navigate in complex, dense traffic scenarios. One way to plan is to search for the best action sequence. However, this is challenging when all necessary components - policy, next-state predictor, and critic - have to be learned. Here we propose Differentiable Simulation for Search (DSS), a framework that leverages the differentiable simulator Waymax as both a next state predictor and a critic. It relies on the simulator's hardcoded dynamics, making state predictions highly accurate, while utilizing the simulator's differentiability to effectively search across action sequences. Our DSS agent optimizes its actions using gradient descent over imagined future trajectories. We show experimentally that DSS - the combination of planning gradients and stochastic search - significantly improves tracking and path planning accuracy compared to sequence prediction, imitation learning, model-free RL, and other planning methods.
>
---
#### [replaced 025] Leverage Cross-Attention for End-to-End Open-Vocabulary Panoptic Reconstruction
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2501.01119v2](https://arxiv.org/pdf/2501.01119v2)**

> **作者:** Xuan Yu; Yuxuan Xie; Yili Liu; Haojian Lu; Rong Xiong; Yiyi Liao; Yue Wang
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** Open-vocabulary panoptic reconstruction offers comprehensive scene understanding, enabling advances in embodied robotics and photorealistic simulation. In this paper, we propose PanopticRecon++, an end-to-end method that formulates panoptic reconstruction through a novel cross-attention perspective. This perspective models the relationship between 3D instances (as queries) and the scene's 3D embedding field (as keys) through their attention map. Unlike existing methods that separate the optimization of queries and keys or overlook spatial proximity, PanopticRecon++ introduces learnable 3D Gaussians as instance queries. This formulation injects 3D spatial priors to preserve proximity while maintaining end-to-end optimizability. Moreover, this query formulation facilitates the alignment of 2D open-vocabulary instance IDs across frames by leveraging optimal linear assignment with instance masks rendered from the queries. Additionally, we ensure semantic-instance segmentation consistency by fusing query-based instance segmentation probabilities with semantic probabilities in a novel panoptic head supervised by a panoptic loss. During training, the number of instance query tokens dynamically adapts to match the number of objects. PanopticRecon++ shows competitive performance in terms of 3D and 2D segmentation and reconstruction performance on both simulation and real-world datasets, and demonstrates a user case as a robot simulator. Our project website is at: https://yuxuan1206.github.io/panopticrecon_pp/
>
---
#### [replaced 026] Unified Generation-Refinement Planning: Bridging Guided Flow Matching and Sampling-Based MPC for Social Navigation
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2508.01192v2](https://arxiv.org/pdf/2508.01192v2)**

> **作者:** Kazuki Mizuta; Karen Leung
>
> **摘要:** Planning safe and effective robot behavior in dynamic, human-centric environments remains a core challenge due to the need to handle multimodal uncertainty, adapt in real-time, and ensure safety. Optimization-based planners offer explicit constraint handling but performance relies on initialization quality. Learning-based planners better capture multimodal possible solutions but struggle to enforce constraints such as safety. In this paper, we introduce a unified generation-refinement framework bridging learning and optimization with a novel reward-guided conditional flow matching (CFM) model and model predictive path integral (MPPI) control. Our key innovation is in the incorporation of a bidirectional information exchange: samples from a reward-guided CFM model provide informed priors for MPPI refinement, while the optimal trajectory from MPPI warm-starts the next CFM generation. Using autonomous social navigation as a motivating application, we demonstrate that our approach can flexibly adapt to dynamic environments to satisfy safety requirements in real-time.
>
---
#### [replaced 027] RGBSQGrasp: Inferring Local Superquadric Primitives from Single RGB Image for Graspability-Aware Bin Picking
- **分类: cs.RO; eess.SY**

- **链接: [https://arxiv.org/pdf/2503.02387v3](https://arxiv.org/pdf/2503.02387v3)**

> **作者:** Yifeng Xu; Fan Zhu; Ye Li; Sebastian Ren; Xiaonan Huang; Yuhao Chen
>
> **备注:** 8 pages, 6 figures, IROS2025 RGMCW Best Workshop Paper
>
> **摘要:** Bin picking is a challenging robotic task due to occlusions and physical constraints that limit visual information for object recognition and grasping. Existing approaches often rely on known CAD models or prior object geometries, restricting generalization to novel or unknown objects. Other methods directly regress grasp poses from RGB-D data without object priors, but the inherent noise in depth sensing and the lack of object understanding make grasp synthesis and evaluation more difficult. Superquadrics (SQ) offer a compact, interpretable shape representation that captures the physical and graspability understanding of objects. However, recovering them from limited viewpoints is challenging, as existing methods rely on multiple perspectives for near-complete point cloud reconstruction, limiting their effectiveness in bin-picking. To address these challenges, we propose \textbf{RGBSQGrasp}, a grasping framework that leverages superquadric shape primitives and foundation metric depth estimation models to infer grasp poses from a monocular RGB camera -- eliminating the need for depth sensors. Our framework integrates a universal, cross-platform dataset generation pipeline, a foundation model-based object point cloud estimation module, a global-local superquadric fitting network, and an SQ-guided grasp pose sampling module. By integrating these components, RGBSQGrasp reliably infers grasp poses through geometric reasoning, enhancing grasp stability and adaptability to unseen objects. Real-world robotic experiments demonstrate a 92% grasp success rate, highlighting the effectiveness of RGBSQGrasp in packed bin-picking environments.
>
---
#### [replaced 028] Occlusion-Aware Contingency Safety-Critical Planning for Autonomous Driving
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2502.06359v2](https://arxiv.org/pdf/2502.06359v2)**

> **作者:** Lei Zheng; Rui Yang; Minzhe Zheng; Zengqi Peng; Michael Yu Wang; Jun Ma
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** Ensuring safe driving while maintaining travel efficiency for autonomous vehicles in dynamic and occluded environments is a critical challenge. This paper proposes an occlusion-aware contingency safety-critical planning approach for real-time autonomous driving. Leveraging reachability analysis for risk assessment, forward reachable sets of phantom vehicles are used to derive risk-aware dynamic velocity boundaries. These velocity boundaries are incorporated into a biconvex nonlinear programming (NLP) formulation that formally enforces safety using spatiotemporal barrier constraints, while simultaneously optimizing exploration and fallback trajectories within a receding horizon planning framework. To enable real-time computation and coordination between trajectories, we employ the consensus alternating direction method of multipliers (ADMM) to decompose the biconvex NLP problem into low-dimensional convex subproblems. The effectiveness of the proposed approach is validated through simulations and real-world experiments in occluded intersections. Experimental results demonstrate enhanced safety and improved travel efficiency, enabling real-time safe trajectory generation in dynamic occluded intersections under varying obstacle conditions. The project page is available at https://zack4417.github.io/oacp-website/.
>
---
#### [replaced 029] Description of Corner Cases in Automated Driving: Goals and Challenges
- **分类: cs.LG; cs.RO**

- **链接: [https://arxiv.org/pdf/2109.09607v4](https://arxiv.org/pdf/2109.09607v4)**

> **作者:** Daniel Bogdoll; Jasmin Breitenstein; Florian Heidecker; Maarten Bieshaar; Bernhard Sick; Tim Fingscheidt; J. Marius Zöllner
>
> **备注:** Daniel Bogdoll, Jasmin Breitenstein and Florian Heidecker contributed equally. Accepted for publication at ICCV 2021 ERCVAD Workshop
>
> **摘要:** Scaling the distribution of automated vehicles requires handling various unexpected and possibly dangerous situations, termed corner cases (CC). Since many modules of automated driving systems are based on machine learning (ML), CC are an essential part of the data for their development. However, there is only a limited amount of CC data in large-scale data collections, which makes them challenging in the context of ML. With a better understanding of CC, offline applications, e.g., dataset analysis, and online methods, e.g., improved performance of automated driving systems, can be improved. While there are knowledge-based descriptions and taxonomies for CC, there is little research on machine-interpretable descriptions. In this extended abstract, we will give a brief overview of the challenges and goals of such a description.
>
---
#### [replaced 030] Situationally-Aware Dynamics Learning
- **分类: cs.RO; cs.AI; cs.LG; math.OC**

- **链接: [https://arxiv.org/pdf/2505.19574v2](https://arxiv.org/pdf/2505.19574v2)**

> **作者:** Alejandro Murillo-Gonzalez; Lantao Liu
>
> **摘要:** Autonomous robots operating in complex, unstructured environments face significant challenges due to latent, unobserved factors that obscure their understanding of both their internal state and the external world. Addressing this challenge would enable robots to develop a more profound grasp of their operational context. To tackle this, we propose a novel framework for online learning of hidden state representations, with which the robots can adapt in real-time to uncertain and dynamic conditions that would otherwise be ambiguous and result in suboptimal or erroneous behaviors. Our approach is formalized as a Generalized Hidden Parameter Markov Decision Process, which explicitly models the influence of unobserved parameters on both transition dynamics and reward structures. Our core innovation lies in learning online the joint distribution of state transitions, which serves as an expressive representation of latent ego- and environmental-factors. This probabilistic approach supports the identification and adaptation to different operational situations, improving robustness and safety. Through a multivariate extension of Bayesian Online Changepoint Detection, our method segments changes in the underlying data generating process governing the robot's dynamics. The robot's transition model is then informed with a symbolic representation of the current situation derived from the joint distribution of latest state transitions, enabling adaptive and context-aware decision-making. To showcase the real-world effectiveness, we validate our approach in the challenging task of unstructured terrain navigation, where unmodeled and unmeasured terrain characteristics can significantly impact the robot's motion. Extensive experiments in both simulation and real world reveal significant improvements in data efficiency, policy performance, and the emergence of safer, adaptive navigation strategies.
>
---
#### [replaced 031] AsynEIO: Asynchronous Monocular Event-Inertial Odometry Using Gaussian Process Regression
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2411.12175v2](https://arxiv.org/pdf/2411.12175v2)**

> **作者:** Zhixiang Wang; Xudong Li; Yizhai Zhang; Fan Zhang; Panfeng Huang
>
> **备注:** 20 pages, 20 figures
>
> **摘要:** Event cameras, when combined with inertial sensors, show significant potential for motion estimation in challenging scenarios, such as high-speed maneuvers and low-light environments. There are many methods for producing such estimations, but most boil down to a synchronous discrete-time fusion problem. However, the asynchronous nature of event cameras and their unique fusion mechanism with inertial sensors remain underexplored. In this paper, we introduce a monocular event-inertial odometry method called AsynEIO, designed to fuse asynchronous event and inertial data within a unified Gaussian Process (GP) regression framework. Our approach incorporates an event-driven frontend that tracks feature trajectories directly from raw event streams at a high temporal resolution. These tracked feature trajectories, along with various inertial factors, are integrated into the same GP regression framework to enable asynchronous fusion. With deriving analytical residual Jacobians and noise models, our method constructs a factor graph that is iteratively optimized and pruned using a sliding-window optimizer. Comparative assessments highlight the performance of different inertial fusion strategies, suggesting optimal choices for varying conditions. Experimental results on both public datasets and our own event-inertial sequences indicate that AsynEIO outperforms existing methods, especially in high-speed and low-illumination scenarios.
>
---
#### [replaced 032] Learning to Drive Anywhere with Model-Based Reannotation
- **分类: cs.RO; cs.CV; cs.LG; eess.SY**

- **链接: [https://arxiv.org/pdf/2505.05592v3](https://arxiv.org/pdf/2505.05592v3)**

> **作者:** Noriaki Hirose; Lydia Ignatova; Kyle Stachowicz; Catherine Glossop; Sergey Levine; Dhruv Shah
>
> **备注:** 9 pages, 8 figures, 6 tables
>
> **摘要:** Developing broadly generalizable visual navigation policies for robots is a significant challenge, primarily constrained by the availability of large-scale, diverse training data. While curated datasets collected by researchers offer high quality, their limited size restricts policy generalization. To overcome this, we explore leveraging abundant, passively collected data sources, including large volumes of crowd-sourced teleoperation data and unlabeled YouTube videos, despite their potential for lower quality or missing action labels. We propose Model-Based ReAnnotation (MBRA), a framework that utilizes a learned short-horizon, model-based expert model to relabel or generate high-quality actions for these passive datasets. This relabeled data is then distilled into LogoNav, a long-horizon navigation policy conditioned on visual goals or GPS waypoints. We demonstrate that LogoNav, trained using MBRA-processed data, achieves state-of-the-art performance, enabling robust navigation over distances exceeding 300 meters in previously unseen indoor and outdoor environments. Our extensive real-world evaluations, conducted across a fleet of robots (including quadrupeds) in six cities on three continents, validate the policy's ability to generalize and navigate effectively even amidst pedestrians in crowded settings.
>
---
#### [replaced 033] Monocular Person Localization under Camera Ego-motion
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2503.02916v2](https://arxiv.org/pdf/2503.02916v2)**

> **作者:** Yu Zhan; Hanjing Ye; Hong Zhang
>
> **备注:** Accepted by IROS2025. Project page: https://medlartea.github.io/rpf-quadruped/
>
> **摘要:** Localizing a person from a moving monocular camera is critical for Human-Robot Interaction (HRI). To estimate the 3D human position from a 2D image, existing methods either depend on the geometric assumption of a fixed camera or use a position regression model trained on datasets containing little camera ego-motion. These methods are vulnerable to severe camera ego-motion, resulting in inaccurate person localization. We consider person localization as a part of a pose estimation problem. By representing a human with a four-point model, our method jointly estimates the 2D camera attitude and the person's 3D location through optimization. Evaluations on both public datasets and real robot experiments demonstrate our method outperforms baselines in person localization accuracy. Our method is further implemented into a person-following system and deployed on an agile quadruped robot.
>
---
#### [replaced 034] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.05186v4](https://arxiv.org/pdf/2508.05186v4)**

> **作者:** Yongjie Bai; Zhouxia Wang; Yang Liu; Kaijun Luo; Yifan Wen; Mingtong Dai; Weixing Chen; Ziliang Chen; Lingbo Liu; Guanbin Li; Liang Lin
>
> **备注:** 24 pages, 15 figures, project page: https://hcplab-sysu.github.io/TAVP
>
> **摘要:** Recent vision-language-action (VLA) models for multi-task robotic manipulation commonly rely on static viewpoints and shared visual encoders, which limit 3D perception and cause task interference, hindering robustness and generalization. In this work, we propose Task-aware Virtual View Exploration (TVVE), a framework designed to overcome these challenges by integrating virtual view exploration with task-specific representation learning. TVVE employs an efficient exploration policy, accelerated by a novel pseudo-environment, to acquire informative views. Furthermore, we introduce a Task-aware Mixture-of-Experts (TaskMoE) visual encoder to disentangle features across different tasks, boosting both representation fidelity and task generalization. By learning to see the world in a task-aware way, TVVE generates more complete and discriminative visual representations, demonstrating significantly enhanced action prediction across a wide array of manipulation challenges. To further validate the robustness and generalization capability of TVVE under out-of-distribution (OOD) settings, we construct a challenging benchmark, RLBench-OG, covering various visual perturbations and camera pose variations. Extensive experiments on RLBench and RLBench-OG show that our TVVE achieves superior performance over state-of-the-art approaches. In real-robot experiments, TVVE demonstrates exceptional performance and generalizes robustly in multiple OOD settings, including visual disturbances and unseen instructions. Visual results and code are provided at: https://hcplab-sysu.github.io/TAVP.
>
---
#### [replaced 035] Yummy Operations Robot Initiative: Autonomous Cooking System Utilizing a Modular Robotic Kitchen and a Dual-Arm Proprioceptive Manipulator
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2405.11094v4](https://arxiv.org/pdf/2405.11094v4)**

> **作者:** Donghun Noh; Hyunwoo Nam; Kyle Gillespie; Yeting Liu; Dennis Hong
>
> **摘要:** This paper presents Yummy Operations Robot Initiative (YORI), a proprioceptive dual-arm robotic system that demonstrates autonomous multi-dish cooking for scalable food service applications. YORI integrates a dual-arm manipulator equipped with proprioceptive actuators, custom-designed tools, appliances, and a structured kitchen environment to address the complexities of cooking tasks. The proprioceptive actuators enable fast, precise, force-controlled movements while mitigating the risks associated with cooking-related impacts. The system's modular kitchen design and flexible tool-changing mechanism support simultaneous multi-dish preparation through torque control and optimization-based motion planning and scheduling. A comprehensive scheduling framework with dynamic rescheduling ensures reliable adaptation to new orders and delays. The system was publicly validated through live demonstrations, reliably preparing steak-frites across multiple convention sessions. This paper details YORI's design and explores future directions in kitchen optimization, task planning, and food quality control, demonstrating its potential as a scalable robotic cooking solution. A system introduction and cooking videos are available online.
>
---
#### [replaced 036] Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2508.10936v2](https://arxiv.org/pdf/2508.10936v2)**

> **作者:** Cheng Chen; Hao Huang; Saurabh Bagchi
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** Collaborative perception enables connected vehicles to share information, overcoming occlusions and extending the limited sensing range inherent in single-agent (non-collaborative) systems. Existing vision-only methods for 3D semantic occupancy prediction commonly rely on dense 3D voxels, which incur high communication costs, or 2D planar features, which require accurate depth estimation or additional supervision, limiting their applicability to collaborative scenarios. To address these challenges, we propose the first approach leveraging sparse 3D semantic Gaussian splatting for collaborative 3D semantic occupancy prediction. By sharing and fusing intermediate Gaussian primitives, our method provides three benefits: a neighborhood-based cross-agent fusion that removes duplicates and suppresses noisy or inconsistent Gaussians; a joint encoding of geometry and semantics in each primitive, which reduces reliance on depth supervision and allows simple rigid alignment; and sparse, object-centric messages that preserve structural information while reducing communication volume. Extensive experiments demonstrate that our approach outperforms single-agent perception and baseline collaborative methods by +8.42 and +3.28 points in mIoU, and +5.11 and +22.41 points in IoU, respectively. When further reducing the number of transmitted Gaussians, our method still achieves a +1.9 improvement in mIoU, using only 34.6% communication volume, highlighting robust performance under limited communication budgets.
>
---
#### [replaced 037] FalconWing: An Ultra-Light Indoor Fixed-Wing UAV Platform for Vision-Based Autonomy
- **分类: cs.RO; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.01383v3](https://arxiv.org/pdf/2505.01383v3)**

> **作者:** Yan Miao; Will Shen; Hang Cui; Sayan Mitra
>
> **摘要:** We introduce FalconWing, an ultra-light (150 g) indoor fixed-wing UAV platform for vision-based autonomy. Controlled indoor environment enables year-round repeatable UAV experiment but imposes strict weight and maneuverability limits on the UAV, motivating our ultra-light FalconWing design. FalconWing couples a lightweight hardware stack (137g airframe with a 9g camera) and offboard computation with a software stack featuring a photorealistic 3D Gaussian Splat (GSplat) simulator for developing and evaluating vision-based controllers. We validate FalconWing on two challenging vision-based aerial case studies. In the leader-follower case study, our best vision-based controller, trained via imitation learning on GSplat-rendered data augmented with domain randomization, achieves 100% tracking success across 3 types of leader maneuvers over 30 trials and shows robustness to leader's appearance shifts in simulation. In the autonomous landing case study, our vision-based controller trained purely in simulation transfers zero-shot to real hardware, achieving an 80% success rate over ten landing trials. We will release hardware designs, GSplat scenes, and dynamics models upon publication to make FalconWing an open-source flight kit for engineering students and research labs.
>
---
#### [replaced 038] Doppler Correspondence: Non-Iterative Scan Matching With Doppler Velocity-Based Correspondence
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2502.11461v3](https://arxiv.org/pdf/2502.11461v3)**

> **作者:** Jiwoo Kim; Geunsik Bae; Changseung Kim; Jinwoo Lee; Woojae Shin; Hyondong Oh
>
> **摘要:** Achieving successful scan matching is essential for LiDAR odometry. However, in challenging environments with adverse weather conditions or repetitive geometric patterns, LiDAR odometry performance is degraded due to incorrect scan matching. Recently, the emergence of frequency-modulated continuous wave 4D LiDAR and 4D radar technologies has provided the potential to address these unfavorable conditions. The term 4D refers to point cloud data characterized by range, azimuth, and elevation along with Doppler velocity. Although 4D data is available, most scan matching methods for 4D LiDAR and 4D radar still establish correspondence by repeatedly identifying the closest points between consecutive scans, overlooking the Doppler information. This paper introduces, for the first time, a simple Doppler velocity-based correspondence -- Doppler Correspondence -- that is invariant to translation and small rotation of the sensor, with its geometric and kinematic foundations. Extensive experiments demonstrate that the proposed method enables the direct matching of consecutive point clouds without an iterative process, making it computationally efficient. Additionally, it provides a more robust correspondence estimation in environments with repetitive geometric patterns.The implementation of our proposed method is publicly available at https://github.com/Tars0523/Doppler Correspondence.
>
---
